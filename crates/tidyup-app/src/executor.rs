//! Apply a set of review decisions to the filesystem.
//!
//! The executor is the only place in tidyup that actually *moves bytes*. It
//! honors two invariants from `CLAUDE.md`:
//!
//! 1. **Every move is preceded by a backup.** Loose proposals shelve the
//!    original via `BackupStore::shelve` before the filesystem rename; if
//!    shelving fails, the move is aborted for that proposal.
//! 2. **Bundles move atomically or not at all.** Bundles shelve the entire
//!    subtree via `shelve_bundle` and then execute a single `rename()` of
//!    the bundle root. Same-volume renames are atomic by POSIX/NTFS guarantee;
//!    cross-volume renames fall back to copy-verify-delete of the subtree,
//!    rolling back staged data on any failure.
//!
//! Decisions that are `Reject` are skipped. `Override { new_target }` wins over
//! the proposal's `proposed_path`.
//!
//! The executor does not decide *whether* to apply a change — it just executes
//! the subset the caller has pre-approved. Review logic stays in the review
//! handler; apply logic stays here.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context};
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter, ReviewHandler};
use tidyup_core::storage::{BackupStore, ChangeLog};
use tidyup_core::Result;
use tidyup_domain::{BundleProposal, ChangeProposal, FileId, IndexedFile, Phase, ReviewDecision};
use uuid::Uuid;
use walkdir::WalkDir;

/// Summary of what the executor did during an apply pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApplyReport {
    pub applied: usize,
    pub skipped: usize,
    pub failed: usize,
    pub bundles_applied: usize,
    pub bundles_skipped: usize,
    pub bundles_failed: usize,
}

/// Collection of dependencies the executor needs.
#[allow(missing_debug_implementations)]
pub struct ExecutorDeps<'a> {
    pub change_log: &'a dyn ChangeLog,
    pub backup_store: &'a dyn BackupStore,
    pub progress: &'a dyn ProgressReporter,
}

/// Apply a list of loose `ChangeProposal`s according to a matching list of
/// `ReviewDecision`s.
///
/// `decisions` is assumed to cover the same proposals by id — each decision's
/// `proposal_id` (or the single `Uuid` variant) must map to exactly one
/// proposal. Unmatched decisions are ignored; proposals with no matching
/// decision are skipped.
///
/// `dry_run = true` produces status updates and reports but does not touch the
/// filesystem or write shelf entries.
pub async fn apply_loose_decisions(
    proposals: &[ChangeProposal],
    decisions: &[ReviewDecision],
    deps: &ExecutorDeps<'_>,
    dry_run: bool,
) -> Result<ApplyReport> {
    let by_id: HashMap<Uuid, &ChangeProposal> = proposals.iter().map(|p| (p.id, p)).collect();
    #[allow(clippy::cast_possible_truncation)]
    let total = decisions.len() as u64;
    deps.progress
        .phase_started(Phase::Applying, Some(total))
        .await;

    let mut report = ApplyReport::default();
    for (idx, decision) in decisions.iter().enumerate() {
        let (proposal_id, target_override) = match decision {
            ReviewDecision::Approve(id) => (*id, None),
            ReviewDecision::Override {
                proposal_id,
                new_target,
            } => (*proposal_id, Some(new_target.clone())),
            ReviewDecision::Reject(_) => {
                report.skipped += 1;
                continue;
            }
        };

        let Some(proposal) = by_id.get(&proposal_id) else {
            report.skipped += 1;
            continue;
        };

        let target = target_override
            .clone()
            .unwrap_or_else(|| proposal.proposed_path.clone());

        match apply_single(proposal, &target, deps, dry_run).await {
            Ok(()) => {
                report.applied += 1;
                deps.progress
                    .item_completed(
                        Phase::Applying,
                        ProgressItem {
                            label: proposal.original_path.display().to_string(),
                            #[allow(clippy::cast_possible_truncation)]
                            current: (idx as u64) + 1,
                            total: Some(total),
                        },
                    )
                    .await;
            }
            Err(e) => {
                report.failed += 1;
                deps.progress
                    .message(
                        Level::Warn,
                        &format!("apply failed for {}: {e}", proposal.original_path.display()),
                    )
                    .await;
            }
        }
    }

    deps.progress.phase_finished(Phase::Applying).await;
    Ok(report)
}

/// Apply a list of bundles. A bundle is applied atomically: either every
/// member moves or no member moves.
///
/// Bundle-level review is a Phase-6+ port addition; for v0.1 this function
/// accepts the full `auto_apply` list (the caller decides — CLI uses
/// `--yes` + a confidence threshold, else surfaces a warning and the bundle
/// stays pending).
pub async fn apply_bundles(
    bundles: &[BundleProposal],
    auto_apply: &[Uuid],
    deps: &ExecutorDeps<'_>,
    dry_run: bool,
) -> Result<ApplyReport> {
    let mut report = ApplyReport::default();
    if bundles.is_empty() {
        return Ok(report);
    }
    for bundle in bundles {
        if !auto_apply.contains(&bundle.id) {
            report.bundles_skipped += 1;
            continue;
        }
        // Two atomic strategies: directory bundles (code projects, etc.) move by
        // a single root rename; file-set bundles (photo bursts, music albums,
        // document series) are clustered loose siblings with no shared root, so
        // each member moves individually with all-or-nothing rollback.
        let result = if bundle.kind.moves_as_file_set() {
            apply_file_set_bundle(bundle, deps, dry_run).await
        } else {
            apply_bundle_atomic(bundle, deps, dry_run).await
        };
        match result {
            Ok(()) => report.bundles_applied += 1,
            Err(e) => {
                report.bundles_failed += 1;
                deps.progress
                    .message(
                        Level::Warn,
                        &format!("bundle apply failed for {}: {e}", bundle.root.display()),
                    )
                    .await;
            }
        }
    }
    Ok(report)
}

async fn apply_single(
    proposal: &ChangeProposal,
    target: &Path,
    deps: &ExecutorDeps<'_>,
    dry_run: bool,
) -> Result<()> {
    let source = proposal.original_path.clone();
    if !source.exists() {
        return Err(anyhow!("source missing: {}", source.display()));
    }
    if dry_run {
        // Dry run: count in the report but never mutate FS or DB state.
        return Ok(());
    }

    // Shelve before any filesystem mutation.
    let indexed = indexed_stub(&source, proposal.file_id.clone())?;
    deps.backup_store
        .shelve(&indexed, proposal.id)
        .await
        .with_context(|| format!("shelving {}", source.display()))?;

    ensure_parent(target)?;
    move_path(&source, target)
        .with_context(|| format!("moving {} -> {}", source.display(), target.display()))?;

    deps.change_log.mark_applied(proposal.id).await?;
    Ok(())
}

async fn apply_bundle_atomic(
    bundle: &BundleProposal,
    deps: &ExecutorDeps<'_>,
    dry_run: bool,
) -> Result<()> {
    let source = bundle.root.clone();
    let leaf = source
        .file_name()
        .ok_or_else(|| anyhow!("bundle root has no file_name: {}", source.display()))?;
    let target = bundle.target_parent.join(leaf);

    if !source.exists() {
        return Err(anyhow!("bundle root missing: {}", source.display()));
    }
    if dry_run {
        return Ok(());
    }

    deps.backup_store
        .shelve_bundle(&source, bundle.id)
        .await
        .with_context(|| format!("shelving bundle {}", source.display()))?;

    ensure_parent(&target)?;
    move_path(&source, &target)
        .with_context(|| format!("moving bundle {} -> {}", source.display(), target.display()))?;

    deps.change_log.mark_bundle_applied(bundle.id).await?;
    Ok(())
}

/// Apply a **file-set** bundle: move each member to its own `proposed_path`,
/// atomically as a group. Unlike [`apply_bundle_atomic`] (a single directory
/// rename), the members are clustered loose siblings with no shared root, so
/// each is shelved + moved individually and **every completed move is reversed
/// if any member fails** — the bundle relocates whole or not at all.
///
/// Each member is shelved keyed by its own proposal id (the same id
/// [`crate::rollback`] looks up), so a later `rollback` restores the originals.
async fn apply_file_set_bundle(
    bundle: &BundleProposal,
    deps: &ExecutorDeps<'_>,
    dry_run: bool,
) -> Result<()> {
    // Pre-flight: every source must exist and no target may be occupied, so the
    // common failure modes are caught before we touch the filesystem.
    for member in &bundle.members {
        if !member.original_path.exists() {
            return Err(anyhow!(
                "file-set member missing: {}",
                member.original_path.display()
            ));
        }
        if member.proposed_path.exists() {
            return Err(anyhow!(
                "refusing to overwrite existing target: {}",
                member.proposed_path.display()
            ));
        }
    }
    if dry_run {
        return Ok(());
    }

    // Completed (dst, src) moves, for reverse-on-failure rollback (LIFO).
    let mut moved: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    for member in &bundle.members {
        let src = member.original_path.clone();
        let dst = member.proposed_path.clone();

        let indexed = indexed_stub(&src, member.file_id.clone())?;
        if let Err(e) = deps.backup_store.shelve(&indexed, member.id).await {
            reverse_moves(&moved);
            return Err(e).with_context(|| format!("shelving {}", src.display()));
        }

        if let Err(e) = ensure_parent(&dst).and_then(|()| move_path(&src, &dst)) {
            reverse_moves(&moved);
            return Err(e)
                .with_context(|| format!("moving {} -> {}", src.display(), dst.display()));
        }
        moved.push((dst, src));
    }

    deps.change_log.mark_bundle_applied(bundle.id).await?;
    Ok(())
}

/// Best-effort reversal of completed file-set moves, used only on the failure
/// path (LIFO order). Errors are swallowed — the originals are also preserved
/// on the backup shelf, so `rollback` can still recover even if a reverse fails.
fn reverse_moves(moved: &[(std::path::PathBuf, std::path::PathBuf)]) {
    for (dst, src) in moved.iter().rev() {
        let _ = move_path(dst, src);
    }
}

/// Fast-path same-volume rename with a cross-volume copy-verify-delete fallback.
fn move_path(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if dst.exists() {
        return Err(anyhow!(
            "refusing to overwrite existing target: {}",
            dst.display(),
        ));
    }
    match std::fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if is_cross_device(&e) => copy_verify_delete(src, dst),
        Err(e) => Err(e).with_context(|| format!("rename {} -> {}", src.display(), dst.display())),
    }
}

fn is_cross_device(e: &std::io::Error) -> bool {
    // Portable detection via ErrorKind::CrossesDevices landed in stable; older
    // rustcs require inspecting the raw os error. Accept either.
    #[allow(unused_variables)]
    let kind = e.kind();
    #[cfg(unix)]
    {
        if e.raw_os_error() == Some(libc_exdev()) {
            return true;
        }
    }
    // String-match fallback for platforms where kind doesn't surface the distinction.
    e.to_string().to_lowercase().contains("cross-device")
}

#[cfg(unix)]
const fn libc_exdev() -> i32 {
    // EXDEV is 18 on Linux, 18 on macOS, 18 on FreeBSD. Hard-code to avoid a
    // libc dep.
    18
}

/// Copy subtree (or file), verify via BLAKE3 hash, then delete source. If
/// anything fails, staged data at the destination is removed; originals remain
/// untouched.
fn copy_verify_delete(src: &Path, dst: &Path) -> anyhow::Result<()> {
    let copy_result: anyhow::Result<()> = (|| {
        if src.is_dir() {
            copy_dir_recursive(src, dst)?;
        } else {
            if let Some(parent) = dst.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("mkdir {}", parent.display()))?;
                }
            }
            std::fs::copy(src, dst)
                .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
        }
        verify_tree(src, dst)?;
        Ok(())
    })();

    if let Err(e) = copy_result {
        // Roll back staged data; ignore best-effort cleanup errors.
        if dst.is_dir() {
            let _ = std::fs::remove_dir_all(dst);
        } else if dst.exists() {
            let _ = std::fs::remove_file(dst);
        }
        return Err(e);
    }

    // Only remove originals after successful verification.
    if src.is_dir() {
        std::fs::remove_dir_all(src)
            .with_context(|| format!("removing original {}", src.display()))?;
    } else {
        std::fs::remove_file(src)
            .with_context(|| format!("removing original {}", src.display()))?;
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dst).with_context(|| format!("mkdir {}", dst.display()))?;
    for entry in WalkDir::new(src).min_depth(1) {
        let entry = entry.with_context(|| format!("walking {}", src.display()))?;
        let relative = entry
            .path()
            .strip_prefix(src)
            .context("strip_prefix during recursive copy")?;
        let target = dst.join(relative);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target)
                .with_context(|| format!("mkdir {}", target.display()))?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("mkdir {}", parent.display()))?;
            }
            std::fs::copy(entry.path(), &target).with_context(|| {
                format!("copy {} -> {}", entry.path().display(), target.display())
            })?;
        }
    }
    Ok(())
}

/// Verify that every file in `src` has an identical BLAKE3 hash at the same
/// relative position under `dst`. Missing / divergent files fail the copy.
fn verify_tree(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if src.is_file() {
        let a = blake3_of(src)?;
        let b = blake3_of(dst)?;
        if a != b {
            return Err(anyhow!(
                "hash mismatch after copy: {} vs {}",
                src.display(),
                dst.display(),
            ));
        }
        return Ok(());
    }
    for entry in WalkDir::new(src).min_depth(1) {
        let entry = entry.with_context(|| format!("walking {}", src.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let relative = entry.path().strip_prefix(src).context("strip_prefix")?;
        let copied = dst.join(relative);
        if !copied.exists() {
            return Err(anyhow!("missing in staged copy: {}", copied.display()));
        }
        let a = blake3_of(entry.path())?;
        let b = blake3_of(&copied)?;
        if a != b {
            return Err(anyhow!(
                "hash mismatch after copy: {} vs {}",
                entry.path().display(),
                copied.display(),
            ));
        }
    }
    Ok(())
}

fn blake3_of(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn ensure_parent(target: &Path) -> anyhow::Result<()> {
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
    }
    Ok(())
}

/// Build a minimal `IndexedFile` suitable for `BackupStore::shelve`. The shelf
/// only needs `path` + `name`; other fields are stubs sized from current
/// metadata.
fn indexed_stub(source: &Path, file_id: Option<FileId>) -> anyhow::Result<IndexedFile> {
    let metadata =
        std::fs::metadata(source).with_context(|| format!("stat {}", source.display()))?;
    let name = source
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("file has no name: {}", source.display()))?
        .to_string();
    let extension = source
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_default();
    Ok(IndexedFile {
        id: file_id.unwrap_or_default(),
        path: source.to_path_buf(),
        name,
        extension,
        mime_type: "application/octet-stream".to_string(),
        size_bytes: metadata.len(),
        content_hash: tidyup_domain::ContentHash(String::new()),
        indexed_at: chrono::Utc::now(),
    })
}

/// Threshold-only bundle selection used on the `--yes` path.
///
/// - `auto_approve_all = true` (i.e. `--yes`): approve bundles with confidence ≥
///   `min_confidence`; skip the rest. This mirrors how `--yes` auto-approves
///   loose moves above a confidence threshold.
/// - `auto_approve_all = false`: approve nothing. Callers that want interactive
///   per-bundle review go through [`select_bundle_decisions`] instead.
///
/// Bundles are atomic aggregates, so the decision is binary per bundle — there
/// is no per-member selection and no `Override` (members carry their own paths
/// and never receive rename proposals).
#[must_use]
pub fn select_auto_applied_bundles(
    bundles: &[BundleProposal],
    auto_approve_all: bool,
    min_confidence: f32,
) -> Vec<Uuid> {
    if !auto_approve_all {
        return Vec::new();
    }
    bundles
        .iter()
        .filter(|b| b.confidence >= min_confidence)
        .map(|b| b.id)
        .collect()
}

/// Decide which bundles to apply, honouring both the `--yes` threshold path and
/// interactive per-bundle review.
///
/// - `auto_approve_all = true` (`--yes`): non-interactive — approve bundles
///   clearing `min_confidence` via [`select_auto_applied_bundles`]. The bundle
///   review handler is not consulted, consistent with `--yes` skipping prompts.
/// - `auto_approve_all = false`: delegate to [`ReviewHandler::review_bundles`].
///   The CLI's interactive handler prompts per bundle; the default trait impl
///   (UI today, test stubs) approves nothing, so every bundle stays pending —
///   exactly the pre-bundle-review behaviour.
///
/// Returns the ids the user (or threshold) approved. Renames are never involved:
/// bundle members move as-is, so there is nothing to surface for rename review.
///
/// # Errors
/// Propagates errors from the review handler.
pub async fn select_bundle_decisions(
    bundles: &[BundleProposal],
    auto_approve_all: bool,
    min_confidence: f32,
    review: &dyn ReviewHandler,
) -> Result<Vec<Uuid>> {
    if bundles.is_empty() {
        return Ok(Vec::new());
    }
    if auto_approve_all {
        return Ok(select_auto_applied_bundles(bundles, true, min_confidence));
    }
    review.review_bundles(bundles.to_vec()).await
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tempfile::TempDir;
    use tidyup_core::frontend::Level;
    use tidyup_core::Result as CoreResult;
    use tidyup_domain::{ChangeStatus, ChangeType};

    struct NullProgress;
    #[async_trait]
    impl ProgressReporter for NullProgress {
        async fn phase_started(&self, _p: Phase, _t: Option<u64>) {}
        async fn item_completed(&self, _p: Phase, _i: ProgressItem) {}
        async fn phase_finished(&self, _p: Phase) {}
        async fn message(&self, _l: Level, _m: &str) {}
    }

    struct RecordingLog {
        applied: Mutex<Vec<Uuid>>,
        applied_bundles: Mutex<Vec<Uuid>>,
    }

    impl RecordingLog {
        const fn new() -> Self {
            Self {
                applied: Mutex::new(Vec::new()),
                applied_bundles: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ChangeLog for RecordingLog {
        async fn record_proposal(&self, _p: &ChangeProposal, _run: Option<Uuid>) -> CoreResult<()> {
            Ok(())
        }
        async fn mark_applied(&self, id: Uuid) -> CoreResult<()> {
            self.applied.lock().unwrap().push(id);
            Ok(())
        }
        async fn mark_unshelved(&self, _id: Uuid) -> CoreResult<()> {
            Ok(())
        }
        async fn pending(&self) -> CoreResult<Vec<ChangeProposal>> {
            Ok(Vec::new())
        }
        async fn record_bundle(&self, _b: &BundleProposal, _run: Option<Uuid>) -> CoreResult<()> {
            Ok(())
        }
        async fn mark_bundle_applied(&self, id: Uuid) -> CoreResult<()> {
            self.applied_bundles.lock().unwrap().push(id);
            Ok(())
        }
        async fn mark_bundle_unshelved(&self, _id: Uuid) -> CoreResult<()> {
            Ok(())
        }
        async fn pending_bundles(&self) -> CoreResult<Vec<BundleProposal>> {
            Ok(Vec::new())
        }
        async fn applied_proposals_for_run(&self, _run: Uuid) -> CoreResult<Vec<ChangeProposal>> {
            Ok(Vec::new())
        }
        async fn applied_bundles_for_run(&self, _run: Uuid) -> CoreResult<Vec<BundleProposal>> {
            Ok(Vec::new())
        }
    }

    struct NoopBackup {
        shelved: Mutex<Vec<Uuid>>,
    }

    impl NoopBackup {
        const fn new() -> Self {
            Self {
                shelved: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl BackupStore for NoopBackup {
        async fn shelve(
            &self,
            file: &IndexedFile,
            change_id: Uuid,
        ) -> CoreResult<tidyup_domain::BackupRecord> {
            self.shelved.lock().unwrap().push(change_id);
            Ok(tidyup_domain::BackupRecord {
                id: Uuid::new_v4(),
                change_id,
                original_path: file.path.clone(),
                backup_path: PathBuf::from("/tmp/shelf/mock"),
                shelved_at: chrono::Utc::now(),
                unshelved_at: None,
                status: tidyup_domain::BackupStatus::Shelved,
            })
        }
        async fn shelve_bundle(
            &self,
            root: &Path,
            bundle_id: Uuid,
        ) -> CoreResult<tidyup_domain::BackupRecord> {
            self.shelved.lock().unwrap().push(bundle_id);
            Ok(tidyup_domain::BackupRecord {
                id: Uuid::new_v4(),
                change_id: bundle_id,
                original_path: root.to_path_buf(),
                backup_path: PathBuf::from("/tmp/shelf/mock"),
                shelved_at: chrono::Utc::now(),
                unshelved_at: None,
                status: tidyup_domain::BackupStatus::Shelved,
            })
        }
        async fn restore(&self, _record: &tidyup_domain::BackupRecord) -> CoreResult<()> {
            Ok(())
        }
        async fn find_by_change_id(
            &self,
            _change_id: Uuid,
        ) -> CoreResult<Option<tidyup_domain::BackupRecord>> {
            Ok(None)
        }
        async fn prune_older_than_days(&self, _days: u32) -> CoreResult<usize> {
            Ok(0)
        }
    }

    fn sample_proposal(src: PathBuf, dst: &Path) -> ChangeProposal {
        ChangeProposal {
            id: Uuid::new_v4(),
            file_id: Some(FileId::new()),
            change_type: ChangeType::Move,
            original_path: src,
            proposed_path: dst.to_path_buf(),
            proposed_name: dst
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("name")
                .to_string(),
            confidence: 0.95,
            reasoning: "t".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: chrono::Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: Some(0.95),
            rename_mismatch_score: None,
        }
    }

    #[tokio::test]
    async fn apply_loose_decisions_moves_files_and_shelves() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("x.txt");
        std::fs::write(&src, b"hello").unwrap();
        let dst = dir.path().join("out/x.txt");

        let proposal = sample_proposal(src.clone(), &dst);
        let log = RecordingLog::new();
        let shelf = NoopBackup::new();
        let deps = ExecutorDeps {
            change_log: &log,
            backup_store: &shelf,
            progress: &NullProgress,
        };

        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        let report =
            apply_loose_decisions(std::slice::from_ref(&proposal), &decisions, &deps, false)
                .await
                .unwrap();

        assert_eq!(report.applied, 1);
        assert!(!src.exists(), "source must be gone");
        assert!(dst.exists(), "destination must exist");
        assert_eq!(std::fs::read(&dst).unwrap(), b"hello");
        assert!(shelf.shelved.lock().unwrap().contains(&proposal.id));
        assert!(log.applied.lock().unwrap().contains(&proposal.id));
    }

    #[tokio::test]
    async fn apply_loose_decisions_honors_override_target() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("a.txt");
        std::fs::write(&src, b"a").unwrap();
        let default_dst = dir.path().join("default/a.txt");
        let override_dst = dir.path().join("chosen/b.txt");

        let proposal = sample_proposal(src.clone(), &default_dst);
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let decisions = vec![ReviewDecision::Override {
            proposal_id: proposal.id,
            new_target: override_dst.clone(),
        }];
        let report = apply_loose_decisions(&[proposal], &decisions, &deps, false)
            .await
            .unwrap();
        assert_eq!(report.applied, 1);
        assert!(override_dst.exists());
    }

    #[tokio::test]
    async fn apply_loose_decisions_respects_dry_run() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("p.txt");
        std::fs::write(&src, b"p").unwrap();
        let dst = dir.path().join("out/p.txt");

        let proposal = sample_proposal(src.clone(), &dst);
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        let report = apply_loose_decisions(&[proposal], &decisions, &deps, true)
            .await
            .unwrap();
        assert_eq!(report.applied, 1);
        assert!(src.exists(), "dry-run must not touch source");
        assert!(!dst.exists(), "dry-run must not touch destination");
    }

    #[tokio::test]
    async fn apply_loose_decisions_rejects_skip() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("s.txt");
        std::fs::write(&src, b"s").unwrap();
        let dst = dir.path().join("out/s.txt");

        let proposal = sample_proposal(src.clone(), &dst);
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let decisions = vec![ReviewDecision::Reject(proposal.id)];
        let report = apply_loose_decisions(&[proposal], &decisions, &deps, false)
            .await
            .unwrap();
        assert_eq!(report.skipped, 1);
        assert!(src.exists());
        assert!(!dst.exists());
    }

    #[tokio::test]
    async fn select_auto_applied_bundles_requires_yes_and_threshold() {
        let low = BundleProposal {
            id: Uuid::new_v4(),
            root: PathBuf::from("/a"),
            kind: tidyup_domain::BundleKind::Generic,
            target_parent: PathBuf::from("/target"),
            members: vec![],
            confidence: 0.3,
            reasoning: "t".into(),
            status: ChangeStatus::Pending,
            created_at: chrono::Utc::now(),
            applied_at: None,
        };
        let high = BundleProposal {
            confidence: 0.9,
            id: Uuid::new_v4(),
            ..low.clone()
        };
        let bundles = vec![low, high.clone()];

        assert_eq!(select_auto_applied_bundles(&bundles, false, 0.5).len(), 0);
        let ids = select_auto_applied_bundles(&bundles, true, 0.5);
        assert_eq!(ids, vec![high.id]);
    }

    fn sample_bundle(confidence: f32) -> BundleProposal {
        BundleProposal {
            id: Uuid::new_v4(),
            root: PathBuf::from("/a"),
            kind: tidyup_domain::BundleKind::Generic,
            target_parent: PathBuf::from("/target"),
            members: vec![],
            confidence,
            reasoning: "t".into(),
            status: ChangeStatus::Pending,
            created_at: chrono::Utc::now(),
            applied_at: None,
        }
    }

    /// Records the bundle ids it was shown and approves a configurable subset.
    struct RecordingBundleReview {
        approve: Vec<Uuid>,
        seen: Mutex<Vec<Uuid>>,
    }

    #[async_trait]
    impl ReviewHandler for RecordingBundleReview {
        async fn review(&self, _p: Vec<ChangeProposal>) -> CoreResult<Vec<ReviewDecision>> {
            Ok(Vec::new())
        }
        async fn review_bundles(&self, bundles: Vec<BundleProposal>) -> CoreResult<Vec<Uuid>> {
            self.seen
                .lock()
                .unwrap()
                .extend(bundles.iter().map(|b| b.id));
            Ok(self.approve.clone())
        }
    }

    #[tokio::test]
    async fn select_bundle_decisions_yes_path_uses_threshold_not_handler() {
        let low = sample_bundle(0.3);
        let high = sample_bundle(0.9);
        let bundles = vec![low, high.clone()];
        // A handler that would approve everything — but --yes must NOT consult it.
        let reviewer = RecordingBundleReview {
            approve: bundles.iter().map(|b| b.id).collect(),
            seen: Mutex::new(Vec::new()),
        };

        let ids = select_bundle_decisions(&bundles, true, 0.5, &reviewer)
            .await
            .unwrap();

        assert_eq!(
            ids,
            vec![high.id],
            "only the high-confidence bundle clears the threshold"
        );
        assert!(
            reviewer.seen.lock().unwrap().is_empty(),
            "review_bundles must not be called on the --yes path",
        );
    }

    #[tokio::test]
    async fn select_bundle_decisions_interactive_path_delegates_to_handler() {
        let a = sample_bundle(0.1);
        let b = sample_bundle(0.99);
        let bundles = vec![a.clone(), b];
        // Interactive handler approves only the *low*-confidence bundle, proving
        // the decision comes from the handler, not a confidence threshold.
        let reviewer = RecordingBundleReview {
            approve: vec![a.id],
            seen: Mutex::new(Vec::new()),
        };

        let ids = select_bundle_decisions(&bundles, false, 0.85, &reviewer)
            .await
            .unwrap();

        assert_eq!(ids, vec![a.id]);
        assert_eq!(
            reviewer.seen.lock().unwrap().len(),
            2,
            "handler sees every bundle"
        );
    }

    #[tokio::test]
    async fn select_bundle_decisions_empty_short_circuits() {
        let reviewer = RecordingBundleReview {
            approve: vec![Uuid::new_v4()],
            seen: Mutex::new(Vec::new()),
        };
        let ids = select_bundle_decisions(&[], false, 0.85, &reviewer)
            .await
            .unwrap();
        assert!(ids.is_empty());
        assert!(
            reviewer.seen.lock().unwrap().is_empty(),
            "no bundles → handler not consulted",
        );
    }

    #[tokio::test]
    async fn default_review_bundles_approves_nothing() {
        // A handler that only implements `review` inherits the trait default for
        // `review_bundles` (approve nothing) — the pre-bundle-review behaviour.
        struct LooseOnly;
        #[async_trait]
        impl ReviewHandler for LooseOnly {
            async fn review(&self, _p: Vec<ChangeProposal>) -> CoreResult<Vec<ReviewDecision>> {
                Ok(Vec::new())
            }
        }
        let bundles = vec![sample_bundle(0.99)];
        let ids = select_bundle_decisions(&bundles, false, 0.85, &LooseOnly)
            .await
            .unwrap();
        assert!(ids.is_empty(), "default review_bundles holds every bundle");
    }

    fn file_set_bundle(members: Vec<ChangeProposal>, target_parent: PathBuf) -> BundleProposal {
        BundleProposal::new(
            PathBuf::from("/src/cluster"),
            tidyup_domain::BundleKind::PhotoBurst,
            target_parent,
            members,
            0.9,
            "photo burst".into(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn apply_file_set_bundle_moves_each_member_and_shelves() {
        let dir = TempDir::new().unwrap();
        let src1 = dir.path().join("IMG_001.jpg");
        let src2 = dir.path().join("IMG_002.jpg");
        std::fs::write(&src1, b"one").unwrap();
        std::fs::write(&src2, b"two").unwrap();
        let dst1 = dir.path().join("Photos/Bursts/x/IMG_001.jpg");
        let dst2 = dir.path().join("Photos/Bursts/x/IMG_002.jpg");

        let members = vec![
            sample_proposal(src1.clone(), &dst1),
            sample_proposal(src2.clone(), &dst2),
        ];
        let bundle = file_set_bundle(members, dir.path().join("Photos/Bursts/x"));

        let log = RecordingLog::new();
        let shelf = NoopBackup::new();
        let deps = ExecutorDeps {
            change_log: &log,
            backup_store: &shelf,
            progress: &NullProgress,
        };

        let report = apply_bundles(std::slice::from_ref(&bundle), &[bundle.id], &deps, false)
            .await
            .unwrap();

        assert_eq!(report.bundles_applied, 1);
        assert!(!src1.exists() && !src2.exists(), "originals must be moved");
        assert!(
            dst1.exists() && dst2.exists(),
            "members must land at targets"
        );
        assert_eq!(std::fs::read(&dst1).unwrap(), b"one");
        // Each member is shelved by its OWN id (the id rollback looks up).
        let shelved = shelf.shelved.lock().unwrap().clone();
        assert!(shelved.contains(&bundle.members[0].id));
        assert!(shelved.contains(&bundle.members[1].id));
        assert!(log.applied_bundles.lock().unwrap().contains(&bundle.id));
    }

    #[tokio::test]
    async fn apply_file_set_bundle_is_atomic_when_a_target_is_occupied() {
        let dir = TempDir::new().unwrap();
        let src1 = dir.path().join("a.jpg");
        let src2 = dir.path().join("b.jpg");
        std::fs::write(&src1, b"a").unwrap();
        std::fs::write(&src2, b"b").unwrap();
        let dst1 = dir.path().join("out/a.jpg");
        let dst2 = dir.path().join("out/b.jpg");
        // Occupy the second target so pre-flight refuses the whole bundle.
        std::fs::create_dir_all(dst2.parent().unwrap()).unwrap();
        std::fs::write(&dst2, b"existing").unwrap();

        let members = vec![
            sample_proposal(src1.clone(), &dst1),
            sample_proposal(src2.clone(), &dst2),
        ];
        let bundle = file_set_bundle(members, dir.path().join("out"));

        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let report = apply_bundles(std::slice::from_ref(&bundle), &[bundle.id], &deps, false)
            .await
            .unwrap();

        assert_eq!(report.bundles_failed, 1);
        // All-or-nothing: neither member moved, no partial state.
        assert!(src1.exists() && src2.exists(), "no member may move");
        assert!(!dst1.exists(), "first target must stay untouched");
        assert_eq!(std::fs::read(&dst2).unwrap(), b"existing");
    }

    #[tokio::test]
    async fn apply_file_set_bundle_dry_run_touches_nothing() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("c.jpg");
        std::fs::write(&src, b"c").unwrap();
        let dst = dir.path().join("out/c.jpg");
        let bundle = file_set_bundle(
            vec![sample_proposal(src.clone(), &dst)],
            dir.path().join("out"),
        );
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let report = apply_bundles(std::slice::from_ref(&bundle), &[bundle.id], &deps, true)
            .await
            .unwrap();
        assert_eq!(report.bundles_applied, 1);
        assert!(src.exists(), "dry-run must not move the source");
        assert!(!dst.exists(), "dry-run must not create the target");
    }
}
