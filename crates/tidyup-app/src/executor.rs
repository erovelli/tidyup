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
use tidyup_domain::{
    BundleKind, BundleProposal, ChangeProposal, ChangeType, FileId, IndexedFile, Phase,
    ReviewDecision,
};
use uuid::Uuid;
use walkdir::WalkDir;

/// Minimum raw semantic cosine for non-interactive approval of opaque
/// structural bundles. Soft/file-set collections remain review-only until
/// their action-specific confidence is calibrated.
pub const DEFAULT_BUNDLE_MIN_CONFIDENCE: f32 = 0.50;

/// Summary of one apply pass.
///
/// During a dry-run, successful previews count as applied dispositions even
/// though this executor leaves the filesystem and persistence layers untouched.
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

    // Proactive duplicate-destination guard: if two approved decisions resolve
    // to the same target path, moving them in sequence would clobber the first.
    // Detect it up front and refuse the whole apply rather than silently losing
    // a file. (Rejected decisions don't contribute a target.)
    check_no_duplicate_targets(decisions, &by_id)?;

    let total = u64::try_from(decisions.len()).unwrap_or(u64::MAX);
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
            ReviewDecision::Reject(id) => {
                report.skipped = report.skipped.saturating_add(1);
                // Persist the review outcome so the rejected proposal leaves the
                // pending set (else it re-surfaces on every run). Only on a real
                // apply — a dry-run is a pure preview that mutates no state.
                if !dry_run {
                    if let Err(e) = deps.change_log.mark_rejected(*id).await {
                        deps.progress
                            .message(
                                Level::Warn,
                                &format!("recording rejection for {id} failed: {e}"),
                            )
                            .await;
                    }
                }
                continue;
            }
        };

        let Some(proposal) = by_id.get(&proposal_id) else {
            report.skipped = report.skipped.saturating_add(1);
            continue;
        };

        let target = target_override
            .clone()
            .unwrap_or_else(|| proposal.proposed_path.clone());

        match apply_single(proposal, &target, deps, dry_run).await {
            Ok(()) => {
                report.applied = report.applied.saturating_add(1);
                deps.progress
                    .item_completed(
                        Phase::Applying,
                        ProgressItem {
                            label: proposal.original_path.display().to_string(),
                            current: u64::try_from(idx).unwrap_or(u64::MAX).saturating_add(1),
                            total: Some(total),
                        },
                    )
                    .await;
            }
            Err(e) => {
                report.failed = report.failed.saturating_add(1);
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
    approved: &[BundleProposal],
    deps: &ExecutorDeps<'_>,
    dry_run: bool,
) -> Result<ApplyReport> {
    let mut report = ApplyReport::default();
    if bundles.is_empty() {
        return Ok(report);
    }
    let approved_by_id: HashMap<Uuid, &BundleProposal> =
        approved.iter().map(|bundle| (bundle.id, bundle)).collect();
    for original in bundles {
        let Some(reviewed) = approved_by_id.get(&original.id).copied() else {
            report.bundles_skipped = report.bundles_skipped.saturating_add(1);
            continue;
        };
        let bundle = match reconcile_reviewed_bundle(original, reviewed) {
            Ok(bundle) => bundle,
            Err(e) => {
                report.bundles_failed = report.bundles_failed.saturating_add(1);
                deps.progress
                    .message(
                        Level::Warn,
                        &format!(
                            "bundle review rejected for {}: {e}",
                            original.root.display()
                        ),
                    )
                    .await;
                continue;
            }
        };
        // Two atomic strategies: directory bundles (code projects, etc.) move by
        // a single root rename; file-set bundles (photo bursts, music albums,
        // document series) are clustered loose siblings with no shared root, so
        // each member moves individually with all-or-nothing rollback.
        let result = if bundle.kind.moves_as_file_set() {
            apply_file_set_bundle(&bundle, deps, dry_run).await
        } else {
            apply_bundle_atomic(&bundle, deps, dry_run).await
        };
        match result {
            Ok(()) => {
                report.bundles_applied = report.bundles_applied.saturating_add(1);
            }
            Err(e) => {
                report.bundles_failed = report.bundles_failed.saturating_add(1);
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

/// Reconcile a frontend-returned bundle with the persisted proposal.
///
/// A semantic-collection label and its member basenames are editable. Every
/// identity, source, hash, score, status, and destination-parent field remains
/// authoritative from `original`; structural bundles have no editable fields.
/// The returned aggregate is rebuilt in original member order so a frontend
/// cannot add, drop, substitute, or reorder execution inputs.
fn reconcile_reviewed_bundle(
    original: &BundleProposal,
    reviewed: &BundleProposal,
) -> Result<BundleProposal> {
    validate_immutable_bundle_fields(original, reviewed)?;

    let reviewed_label = match (&original.kind, &reviewed.kind) {
        (BundleKind::SemanticCollection { .. }, BundleKind::SemanticCollection { label }) => {
            validate_path_component(label, "collection label")?;
            Some(label.as_str())
        }
        (original_kind, reviewed_kind) if original_kind == reviewed_kind => None,
        _ => {
            return Err(anyhow!(
                "frontend changed immutable bundle kind for {}",
                original.id
            ));
        }
    };

    if reviewed.members.len() != original.members.len() {
        return Err(anyhow!(
            "frontend changed bundle member count for {} (expected {}, received {})",
            original.id,
            original.members.len(),
            reviewed.members.len()
        ));
    }
    let reviewed_by_id: HashMap<Uuid, &ChangeProposal> = reviewed
        .members
        .iter()
        .map(|member| (member.id, member))
        .collect();
    if reviewed_by_id.len() != reviewed.members.len() {
        return Err(anyhow!(
            "frontend returned duplicate member ids for bundle {}",
            original.id
        ));
    }

    let mut reconciled = original.clone();
    reconciled.kind = reviewed.kind.clone();
    let mut members = Vec::with_capacity(original.members.len());
    for original_member in &original.members {
        let reviewed_member = reviewed_by_id
            .get(&original_member.id)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "frontend replaced or removed member {} from bundle {}",
                    original_member.id,
                    original.id
                )
            })?;
        members.push(reconcile_reviewed_member(
            original_member,
            reviewed_member,
            original,
            reviewed_label,
        )?);
    }
    reconciled.members = members;
    Ok(reconciled)
}

fn validate_immutable_bundle_fields(
    original: &BundleProposal,
    reviewed: &BundleProposal,
) -> Result<()> {
    if reviewed.id != original.id
        || reviewed.root != original.root
        || reviewed.target_parent != original.target_parent
        || reviewed.confidence.to_bits() != original.confidence.to_bits()
        || reviewed.reasoning != original.reasoning
        || reviewed.status != original.status
        || reviewed.created_at != original.created_at
        || reviewed.applied_at != original.applied_at
    {
        return Err(anyhow!(
            "frontend changed immutable bundle fields for {}",
            original.id
        ));
    }
    Ok(())
}

fn reconcile_reviewed_member(
    original_member: &ChangeProposal,
    reviewed_member: &ChangeProposal,
    original_bundle: &BundleProposal,
    reviewed_label: Option<&str>,
) -> Result<ChangeProposal> {
    validate_immutable_member_fields(original_member, reviewed_member, original_bundle.id)?;
    let Some(label) = reviewed_label else {
        if reviewed_member.change_type != original_member.change_type
            || reviewed_member.proposed_name != original_member.proposed_name
            || reviewed_member.proposed_path != original_member.proposed_path
        {
            return Err(anyhow!(
                "frontend edited a structural bundle member in bundle {}",
                original_bundle.id
            ));
        }
        return Ok(original_member.clone());
    };

    validate_path_component(&reviewed_member.proposed_name, "member filename")?;
    let expected_path = original_bundle
        .target_parent
        .join(label)
        .join(&reviewed_member.proposed_name);
    if !reviewed_member
        .proposed_path
        .starts_with(&original_bundle.target_parent)
    {
        return Err(anyhow!(
            "reviewed member target escapes bundle parent: {}",
            reviewed_member.proposed_path.display()
        ));
    }
    if reviewed_member.proposed_path != expected_path {
        return Err(anyhow!(
            "reviewed member target must be collection label plus filename: expected {}, received {}",
            expected_path.display(),
            reviewed_member.proposed_path.display()
        ));
    }
    let original_name = original_member
        .original_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let expected_change = if original_name == reviewed_member.proposed_name {
        ChangeType::Move
    } else {
        ChangeType::RenameAndMove
    };
    if reviewed_member.change_type != expected_change {
        return Err(anyhow!(
            "reviewed member change type does not match its edited filename"
        ));
    }

    let mut member = original_member.clone();
    member.change_type = expected_change;
    member
        .proposed_name
        .clone_from(&reviewed_member.proposed_name);
    member
        .proposed_path
        .clone_from(&reviewed_member.proposed_path);
    Ok(member)
}

fn validate_immutable_member_fields(
    original: &ChangeProposal,
    reviewed: &ChangeProposal,
    bundle_id: Uuid,
) -> Result<()> {
    if reviewed.id != original.id
        || reviewed.file_id != original.file_id
        || reviewed.original_path != original.original_path
        || reviewed.confidence.to_bits() != original.confidence.to_bits()
        || reviewed.reasoning != original.reasoning
        || reviewed.needs_review != original.needs_review
        || reviewed.status != original.status
        || reviewed.created_at != original.created_at
        || reviewed.applied_at != original.applied_at
        || reviewed.bundle_id != original.bundle_id
        || reviewed.classification_confidence.map(f32::to_bits)
            != original.classification_confidence.map(f32::to_bits)
        || reviewed.rename_mismatch_score.map(f32::to_bits)
            != original.rename_mismatch_score.map(f32::to_bits)
        || reviewed.content_hash != original.content_hash
    {
        return Err(anyhow!(
            "frontend changed immutable fields for member {} in bundle {bundle_id}",
            original.id
        ));
    }
    Ok(())
}

fn validate_path_component(value: &str, field: &str) -> Result<()> {
    let path = Path::new(value);
    if value.trim().is_empty()
        || path.file_name().and_then(|name| name.to_str()) != Some(value)
        || path.components().count() != 1
    {
        return Err(anyhow!("{field} must be one non-empty path component"));
    }
    Ok(())
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
    // TOCTOU guard: refuse to move a file that changed since it was classified.
    verify_unchanged(&source, proposal.content_hash.as_deref())?;
    if dry_run {
        // Dry run: count in the report but never mutate FS or DB state.
        return Ok(());
    }

    // Order matters for crash consistency:
    //   1. shelve (safety net — the original is preserved before anything moves)
    //   2. mark_applied (write-ahead journal — records intent BEFORE the move)
    //   3. move
    // A crash between 2 and 3 leaves the proposal marked applied but not yet
    // moved; rollback's precheck sees the destination absent and the original
    // still in place and treats the restore as a safe no-op. The reverse order
    // (move then mark) would strand a moved-but-unjournaled file that rollback
    // — which enumerates *applied* changes — could never reach.
    let indexed = indexed_stub(&source, proposal.file_id.clone())?;
    deps.backup_store
        .shelve(&indexed, proposal.id)
        .await
        .with_context(|| format!("shelving {}", source.display()))?;

    deps.change_log.mark_applied(proposal.id).await?;

    ensure_parent(target)?;
    move_path(&source, target)
        .with_context(|| format!("moving {} -> {}", source.display(), target.display()))?;

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

    // Write-ahead journal: shelve, then mark applied, then move (see
    // `apply_single`). A crash after the mark but before the atomic rename
    // leaves the bundle marked applied yet un-moved; rollback no-ops it (the
    // destination is absent, the original still present).
    deps.backup_store
        .shelve_bundle(&source, bundle.id)
        .await
        .with_context(|| format!("shelving bundle {}", source.display()))?;

    deps.change_log.mark_bundle_applied(bundle.id).await?;

    ensure_parent(&target)?;
    move_path(&source, &target)
        .with_context(|| format!("moving bundle {} -> {}", source.display(), target.display()))?;

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
    // Pre-flight: every source must exist, be unchanged since scan, and no
    // target may be occupied (including collisions *within* the bundle), so the
    // common failure modes are caught before we touch the filesystem.
    let mut seen_targets: HashMap<&Path, &Path> = HashMap::new();
    for member in &bundle.members {
        if !member.original_path.exists() {
            return Err(anyhow!(
                "file-set member missing: {}",
                member.original_path.display()
            ));
        }
        verify_unchanged(&member.original_path, member.content_hash.as_deref())?;
        if member.proposed_path.exists() {
            return Err(anyhow!(
                "refusing to overwrite existing target: {}",
                member.proposed_path.display()
            ));
        }
        if let Some(other) = seen_targets.insert(&member.proposed_path, &member.original_path) {
            return Err(anyhow!(
                "two bundle members target the same path {}: {} and {}",
                member.proposed_path.display(),
                other.display(),
                member.original_path.display(),
            ));
        }
    }
    if dry_run {
        return Ok(());
    }

    // Write-ahead journal for crash consistency: mark the bundle applied BEFORE
    // moving any member. A hard kill mid-loop then leaves the bundle marked
    // applied with only some members shelved+moved; rollback enumerates the
    // bundle (because it is applied), restores the moved members from their
    // shelf records, and skips members that were never shelved (still at their
    // origin). Without this write-ahead mark, a crash before the final
    // mark_bundle_applied would strand moved members that rollback could never
    // reach. Caught (non-crash) failures still reverse in-process below and
    // undo the mark, preserving apply-time atomicity for the common case.
    deps.change_log.mark_bundle_applied(bundle.id).await?;

    // Completed (dst, src) moves, for reverse-on-failure rollback (LIFO).
    let mut moved: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    for member in &bundle.members {
        let src = member.original_path.clone();
        let dst = member.proposed_path.clone();

        let indexed = indexed_stub(&src, member.file_id.clone())?;
        if let Err(e) = deps.backup_store.shelve(&indexed, member.id).await {
            reverse_moves(&moved);
            let _ = deps.change_log.mark_bundle_unshelved(bundle.id).await;
            return Err(e).with_context(|| format!("shelving {}", src.display()));
        }

        if let Err(e) = ensure_parent(&dst).and_then(|()| move_path(&src, &dst)) {
            reverse_moves(&moved);
            let _ = deps.change_log.mark_bundle_unshelved(bundle.id).await;
            return Err(e)
                .with_context(|| format!("moving {} -> {}", src.display(), dst.display()));
        }
        moved.push((dst, src));
    }

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
    // A cross-volume `rename` fails with a platform-specific OS error.
    // `io::ErrorKind::CrossesDevices` is still unstable (feature `io_error_more`),
    // so we match the raw OS error per platform, with a string fallback for any
    // exotic target. Unix: `EXDEV`. Windows: `ERROR_NOT_SAME_DEVICE`.
    #[cfg(unix)]
    if e.raw_os_error() == Some(EXDEV) {
        return true;
    }
    #[cfg(windows)]
    if e.raw_os_error() == Some(ERROR_NOT_SAME_DEVICE) {
        return true;
    }
    let msg = e.to_string().to_lowercase();
    msg.contains("cross-device") || msg.contains("different disk")
}

/// `EXDEV` — "cross-device link". 18 on Linux, macOS, and FreeBSD; hard-coded to
/// avoid a `libc` dependency.
#[cfg(unix)]
const EXDEV: i32 = 18;

/// `ERROR_NOT_SAME_DEVICE` — Win32 error surfaced by `rename` (`MoveFileEx`) when
/// source and destination are on different volumes. Without this branch,
/// cross-volume moves would fail outright on Windows instead of falling back to
/// copy-verify-delete.
#[cfg(windows)]
const ERROR_NOT_SAME_DEVICE: i32 = 17;

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
        let a = blake3_stream(src)?;
        let b = blake3_stream(dst)?;
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
        let a = blake3_stream(entry.path())?;
        let b = blake3_stream(&copied)?;
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

/// Streaming BLAKE3 of a file, hex-encoded. Reads in fixed-size chunks so a
/// multi-GB file never lands in memory at once — used for the cross-volume
/// copy verify and the apply-time TOCTOU re-hash, both of which run over
/// arbitrarily large user files.
fn blake3_stream(path: &Path) -> anyhow::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = match file.read(&mut buf) {
            Ok(n) => n,
            // Retry on EINTR like `std::fs::read` does, so a signal delivered
            // mid-read doesn't spuriously fail a verify/TOCTOU re-hash.
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        if n == 0 {
            break;
        }
        let chunk = buf
            .get(..n)
            .ok_or_else(|| anyhow::anyhow!("file read exceeded hashing buffer"))?;
        hasher.update(chunk);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// TOCTOU guard for apply: re-hash `source` and confirm it still matches the
/// digest captured when the proposal was built. A file edited or replaced in
/// the review gap is refused rather than moved under a stale classification.
/// `expected == None` (hash couldn't be computed at scan time) skips the check.
fn verify_unchanged(source: &Path, expected: Option<&str>) -> anyhow::Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    // Stream the re-hash so verifying a multi-GB file doesn't slurp it into RAM.
    let actual = blake3_stream(source).with_context(|| {
        format!(
            "re-hashing {} for apply-time verification",
            source.display()
        )
    })?;
    if actual != expected {
        return Err(anyhow!(
            "source changed since it was reviewed (content hash mismatch): {} — \
             re-scan before applying",
            source.display(),
        ));
    }
    Ok(())
}

/// Refuse an apply where two approved decisions resolve to the same target
/// path. Applying them in sequence would clobber whichever landed first; this
/// catches it before any filesystem mutation. Only `Approve`/`Override`
/// contribute a target — `Reject` moves nothing.
fn check_no_duplicate_targets(
    decisions: &[ReviewDecision],
    by_id: &HashMap<Uuid, &ChangeProposal>,
) -> Result<()> {
    let mut seen: HashMap<&Path, Uuid> = HashMap::new();
    for decision in decisions {
        let (id, target): (Uuid, &Path) = match decision {
            ReviewDecision::Approve(id) => {
                let Some(p) = by_id.get(id) else { continue };
                (*id, p.proposed_path.as_path())
            }
            ReviewDecision::Override {
                proposal_id,
                new_target,
            } => {
                // Mirror the apply loop, which skips Overrides whose id doesn't
                // match a proposal — such a decision moves nothing, so it can't
                // collide with anything.
                if !by_id.contains_key(proposal_id) {
                    continue;
                }
                (*proposal_id, new_target.as_path())
            }
            ReviewDecision::Reject(_) => continue,
        };
        if let Some(other) = seen.insert(target, id) {
            return Err(anyhow!(
                "two approved changes target the same path {} (proposals {other} and {id}); \
                 refusing to apply to avoid overwriting one with the other",
                target.display(),
            ));
        }
    }
    Ok(())
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
/// - `auto_approve_all = true` (i.e. `--yes`): approve recognized opaque
///   structural bundles with confidence ≥ `min_confidence`; skip soft/file-set
///   collections and the rest. This precision-first policy remains until
///   action-specific calibration establishes a safe collection threshold.
/// - `auto_approve_all = false`: approve nothing. Callers that want interactive
///   per-bundle review go through [`select_bundle_decisions`] instead.
///
/// Bundles are atomic aggregates, so the decision is binary per bundle — there
/// is no per-member selection and no `Override` (members carry their own paths
/// and are never selected independently). Bundles containing member renames
/// require interactive review and are excluded from this path. Soft/file-set
/// collections are held because their raw semantic confidence is not calibrated
/// for unattended application. The legacy `Generic` kind is also held if read
/// from persisted data; current scans do not create generic directory envelopes.
#[must_use]
pub fn select_auto_applied_bundles(
    bundles: &[BundleProposal],
    auto_approve_all: bool,
    min_confidence: f32,
) -> Vec<BundleProposal> {
    if !auto_approve_all {
        return Vec::new();
    }
    bundles
        .iter()
        .filter(|b| {
            !b.kind.moves_as_file_set()
                && !matches!(&b.kind, BundleKind::Generic)
                && b.confidence >= min_confidence
                && !b.members.iter().any(|member| {
                    matches!(
                        member.change_type,
                        ChangeType::Rename | ChangeType::RenameAndMove
                    )
                })
        })
        .cloned()
        .collect()
}

/// Validate selected destinations before any filesystem mutation.
///
/// Paths are folded conservatively for case-insensitive filesystems; this may
/// hold a proposal that would be distinct on a case-sensitive volume, but it
/// can never permit an overwrite on the user's default macOS volume.
///
/// Existing parent directories are allowed, while an existing final target is
/// rejected. Bundle directory destinations reserve their root once; file-set
/// collections reserve each member target.
pub fn validate_destination_ledger(
    proposals: &[ChangeProposal],
    decisions: &[ReviewDecision],
    approved_bundles: &[BundleProposal],
) -> Result<()> {
    let by_id: HashMap<Uuid, &ChangeProposal> = proposals.iter().map(|p| (p.id, p)).collect();
    let mut reserved: Vec<(Vec<String>, String)> = Vec::new();

    for decision in decisions {
        let (id, target, source) = match decision {
            ReviewDecision::Approve(id) => {
                let Some(proposal) = by_id.get(id) else {
                    continue;
                };
                (
                    *id,
                    proposal.proposed_path.as_path(),
                    proposal.original_path.as_path(),
                )
            }
            ReviewDecision::Override {
                proposal_id,
                new_target,
            } => {
                let Some(proposal) = by_id.get(proposal_id) else {
                    continue;
                };
                (
                    *proposal_id,
                    new_target.as_path(),
                    proposal.original_path.as_path(),
                )
            }
            ReviewDecision::Reject(_) => continue,
        };
        reserve_destination(&mut reserved, id, target, source)?;
    }

    for bundle in approved_bundles {
        if bundle.kind.moves_as_file_set() {
            for member in &bundle.members {
                reserve_destination(
                    &mut reserved,
                    member.id,
                    &member.proposed_path,
                    &member.original_path,
                )?;
            }
        } else {
            let leaf = bundle
                .root
                .file_name()
                .ok_or_else(|| anyhow!("bundle root has no filename: {}", bundle.root.display()))?;
            reserve_destination(
                &mut reserved,
                bundle.id,
                &bundle.target_parent.join(leaf),
                &bundle.root,
            )?;
        }
    }
    Ok(())
}

fn reserve_destination(
    reserved: &mut Vec<(Vec<String>, String)>,
    id: Uuid,
    target: &Path,
    source: &Path,
) -> Result<()> {
    let key = normalized_destination(target);
    for (other_key, other) in reserved.iter() {
        let overlap =
            key == *other_key || key.starts_with(other_key) || other_key.starts_with(&key);
        if overlap {
            return Err(anyhow!(
                "approved operations have overlapping destinations at {} ({} and {}); refusing to apply",
                target.display(),
                other,
                id,
            ));
        }
    }
    // A target that exists *because it is the source* is a no-op, not a
    // collision. Scan reorganizes in place, so an override back to the file's
    // current location must not abort the whole plan; the destination is still
    // reserved so nothing else can claim it.
    if key != normalized_destination(source) && target.exists() {
        return Err(anyhow!(
            "approved operation targets an existing path {}; refusing to overwrite",
            target.display()
        ));
    }
    reserved.push((key, id.to_string()));
    Ok(())
}

fn normalized_destination(path: &Path) -> Vec<String> {
    // Component-wise comparison catches ancestor/descendant conflicts without
    // confusing names such as `foo` and `foobar`. Lower-casing is conservative
    // for case-insensitive macOS/Windows volumes. Unicode normalization is not
    // performed here, so canonically equivalent decomposed/composed spellings
    // remain a known limitation and are still subject to the exact filesystem
    // guard at apply time.
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().to_lowercase())
        .collect()
}

/// Decide which bundles to apply, honouring both the `--yes` threshold path and
/// interactive per-bundle review.
///
/// - `auto_approve_all = true` (`--yes`): non-interactive — approve recognized
///   opaque structural bundles clearing `min_confidence` via
///   [`select_auto_applied_bundles`]. Soft/file-set collections remain pending
///   until calibrated and explicitly reviewed.
/// - `auto_approve_all = false`: delegate to [`ReviewHandler::review_bundles`].
///   The CLI prompts per bundle and the desktop exposes bundle decisions in its
///   complete-plan surface. A frontend that relies on the default trait method
///   approves nothing, so its bundles stay pending.
///
/// Returns the proposals the user (or threshold) approved. The threshold path
/// applies only recognized opaque structural bundles; any generic, soft, or
/// rename-bearing bundle remains pending until a frontend reviews it explicitly.
///
/// # Errors
/// Propagates errors from the review handler.
pub async fn select_bundle_decisions(
    bundles: &[BundleProposal],
    auto_approve_all: bool,
    min_confidence: f32,
    review: &dyn ReviewHandler,
) -> Result<Vec<BundleProposal>> {
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
    use tidyup_domain::ChangeStatus;

    #[test]
    fn copy_verify_delete_relocates_file_and_removes_original() {
        // Drives the cross-volume path directly (no real second volume needed):
        // copy -> streamed-hash verify_tree -> delete original.
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src.bin");
        let dst = dir.path().join("nested/dst.bin");
        // Larger than one 64 KiB hash chunk so the streaming loop iterates.
        let payload = vec![0xABu8; 200 * 1024];
        std::fs::write(&src, &payload).unwrap();

        copy_verify_delete(&src, &dst).unwrap();

        assert!(!src.exists(), "original must be removed after verify");
        assert_eq!(std::fs::read(&dst).unwrap(), payload, "content preserved");
    }

    #[test]
    fn copy_verify_delete_relocates_subtree() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("bundle");
        let dst = dir.path().join("moved");
        std::fs::create_dir_all(src.join("a")).unwrap();
        std::fs::write(src.join("a/one.txt"), b"one").unwrap();
        std::fs::write(src.join("two.txt"), vec![7u8; 130 * 1024]).unwrap();

        copy_verify_delete(&src, &dst).unwrap();

        assert!(!src.exists(), "original subtree removed");
        assert_eq!(std::fs::read(dst.join("a/one.txt")).unwrap(), b"one");
        assert_eq!(
            std::fs::read(dst.join("two.txt")).unwrap().len(),
            130 * 1024
        );
    }

    #[test]
    fn cross_device_detected_from_os_error() {
        // A real cross-volume rename surfaces the platform errno; is_cross_device
        // must route it to copy-verify-delete rather than failing the move.
        #[cfg(unix)]
        {
            let exdev = std::io::Error::from_raw_os_error(EXDEV);
            assert!(is_cross_device(&exdev), "EXDEV must count as cross-device");
        }
        #[cfg(windows)]
        {
            let not_same = std::io::Error::from_raw_os_error(ERROR_NOT_SAME_DEVICE);
            assert!(is_cross_device(&not_same));
        }
        // A non-cross-device OS error (errno 13) must NOT trigger the fallback.
        assert!(!is_cross_device(&std::io::Error::from_raw_os_error(13)));
        // String fallback for exotic platforms whose errno we don't hard-code.
        assert!(is_cross_device(&std::io::Error::other(
            "Invalid cross-device link"
        )));
    }

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
        rejected: Mutex<Vec<Uuid>>,
    }

    impl RecordingLog {
        const fn new() -> Self {
            Self {
                applied: Mutex::new(Vec::new()),
                applied_bundles: Mutex::new(Vec::new()),
                rejected: Mutex::new(Vec::new()),
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
        async fn mark_rejected(&self, id: Uuid) -> CoreResult<()> {
            self.rejected.lock().unwrap().push(id);
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
                content_hash: None,
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
                content_hash: None,
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
        async fn precheck_restore(
            &self,
            _record: &tidyup_domain::BackupRecord,
            _destination: &Path,
        ) -> CoreResult<tidyup_domain::RestorePrecheck> {
            Ok(tidyup_domain::RestorePrecheck::Ready)
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
            content_hash: None,
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
        let log = RecordingLog::new();
        let shelf = NoopBackup::new();
        let deps = ExecutorDeps {
            change_log: &log,
            backup_store: &shelf,
            progress: &NullProgress,
        };
        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        let report = apply_loose_decisions(&[proposal], &decisions, &deps, true)
            .await
            .unwrap();
        assert_eq!(report.applied, 1, "dry-run reports what would apply");
        assert!(src.exists(), "dry-run must not touch source");
        assert!(!dst.exists(), "dry-run must not touch destination");
        assert!(shelf.shelved.lock().unwrap().is_empty());
        assert!(log.applied.lock().unwrap().is_empty());
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
    async fn apply_aborts_when_source_changed_since_scan() {
        // Proposal carries the scan-time hash of "before"; the file is edited to
        // "after edit" before apply. The TOCTOU guard must refuse the move.
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("doc.txt");
        std::fs::write(&src, b"before").unwrap();
        let dst = dir.path().join("out/doc.txt");

        let mut proposal = sample_proposal(src.clone(), &dst);
        proposal.content_hash = Some(blake3::hash(b"before").to_hex().to_string());
        // The user edits the file after review.
        std::fs::write(&src, b"after edit").unwrap();

        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        let report = apply_loose_decisions(&[proposal], &decisions, &deps, false)
            .await
            .unwrap();

        assert_eq!(report.failed, 1, "changed source must fail apply");
        assert_eq!(report.applied, 0);
        assert!(src.exists(), "the edited source is left in place");
        assert!(!dst.exists(), "nothing moved");
        assert_eq!(std::fs::read(&src).unwrap(), b"after edit");
    }

    #[tokio::test]
    async fn apply_proceeds_when_source_unchanged() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("doc.txt");
        std::fs::write(&src, b"stable").unwrap();
        let dst = dir.path().join("out/doc.txt");

        let mut proposal = sample_proposal(src.clone(), &dst);
        proposal.content_hash = Some(blake3::hash(b"stable").to_hex().to_string());

        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        let report = apply_loose_decisions(&[proposal], &decisions, &deps, false)
            .await
            .unwrap();

        assert_eq!(report.applied, 1, "unchanged source applies normally");
        assert!(dst.exists());
    }

    #[tokio::test]
    async fn apply_refuses_duplicate_destinations() {
        // Two approved proposals resolve to the SAME target — refuse the whole
        // apply before moving either, so neither clobbers the other.
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, b"aaa").unwrap();
        std::fs::write(&b, b"bbb").unwrap();
        let shared_dst = dir.path().join("out/merged.txt");

        let pa = sample_proposal(a.clone(), &shared_dst);
        let pb = sample_proposal(b.clone(), &shared_dst);
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let decisions = vec![
            ReviewDecision::Approve(pa.id),
            ReviewDecision::Approve(pb.id),
        ];
        let err = apply_loose_decisions(&[pa, pb], &decisions, &deps, false)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("same path"),
            "expected duplicate-destination error, got: {err}"
        );
        // Nothing moved — both sources remain.
        assert!(a.exists() && b.exists());
        assert!(!shared_dst.exists());
    }

    #[tokio::test]
    async fn override_target_collision_is_detected() {
        // A default target and an Override that collide are also caught.
        let dir = TempDir::new().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, b"aaa").unwrap();
        std::fs::write(&b, b"bbb").unwrap();
        let dst_a = dir.path().join("out/a.txt");
        let dst_b = dir.path().join("out/b.txt");

        let pa = sample_proposal(a.clone(), &dst_a);
        let pb = sample_proposal(b.clone(), &dst_b);
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        // Override pb onto pa's target.
        let decisions = vec![
            ReviewDecision::Approve(pa.id),
            ReviewDecision::Override {
                proposal_id: pb.id,
                new_target: dst_a.clone(),
            },
        ];
        let err = apply_loose_decisions(&[pa, pb], &decisions, &deps, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("same path"), "got: {err}");
    }

    #[tokio::test]
    async fn select_auto_applied_bundles_requires_yes_and_threshold() {
        let low = sample_bundle(0.3);
        let typical_raw_cosine = BundleProposal {
            confidence: 0.55,
            id: Uuid::new_v4(),
            ..low.clone()
        };
        let bundles = vec![low, typical_raw_cosine.clone()];

        assert_eq!(
            select_auto_applied_bundles(&bundles, false, DEFAULT_BUNDLE_MIN_CONFIDENCE).len(),
            0
        );
        let ids = select_auto_applied_bundles(&bundles, true, DEFAULT_BUNDLE_MIN_CONFIDENCE);
        assert_eq!(
            ids.iter().map(|bundle| bundle.id).collect::<Vec<_>>(),
            vec![typical_raw_cosine.id],
            "a structural bundle clears the configured raw-cosine threshold",
        );
    }

    #[test]
    fn soft_and_generic_bundles_require_explicit_review() {
        let mut semantic = sample_bundle(0.99);
        semantic.kind = BundleKind::SemanticCollection {
            label: "example".to_string(),
        };
        let mut photo_burst = sample_bundle(0.99);
        photo_burst.kind = BundleKind::PhotoBurst;
        let mut generic = sample_bundle(0.99);
        generic.kind = BundleKind::Generic;

        assert!(
            select_auto_applied_bundles(&[semantic, photo_burst, generic], true, 0.0).is_empty()
        );

        let structural = sample_bundle(0.99);
        assert_eq!(
            select_auto_applied_bundles(std::slice::from_ref(&structural), true, 0.50)
                .iter()
                .map(|bundle| bundle.id)
                .collect::<Vec<_>>(),
            vec![structural.id],
            "recognized opaque structural bundles remain eligible after clearing the threshold",
        );
    }

    #[test]
    fn select_auto_applied_bundles_holds_member_renames() {
        let mut renamed = sample_bundle(0.95);
        let mut member = sample_proposal(
            PathBuf::from("/a/screenshot.png"),
            Path::new("/target/project_homepage.png"),
        );
        member.change_type = ChangeType::RenameAndMove;
        member.proposed_name = "project_homepage.png".to_string();
        member.bundle_id = Some(renamed.id);
        renamed.members.push(member);

        assert!(select_auto_applied_bundles(&[renamed], true, 0.50).is_empty());
    }

    fn sample_bundle(confidence: f32) -> BundleProposal {
        BundleProposal {
            id: Uuid::new_v4(),
            root: PathBuf::from("/a"),
            kind: BundleKind::RustCrate,
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
        async fn review_bundles(
            &self,
            bundles: Vec<BundleProposal>,
        ) -> CoreResult<Vec<BundleProposal>> {
            self.seen
                .lock()
                .unwrap()
                .extend(bundles.iter().map(|b| b.id));
            Ok(bundles
                .into_iter()
                .filter(|bundle| self.approve.contains(&bundle.id))
                .collect())
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
            ids.iter().map(|bundle| bundle.id).collect::<Vec<_>>(),
            vec![high.id],
            "only the high-confidence bundle clears the threshold"
        );
        assert!(
            reviewer.seen.lock().unwrap().is_empty(),
            "review_bundles must not be called on the --yes path",
        );
    }

    #[test]
    fn generic_envelopes_require_explicit_review() {
        let generic = BundleProposal {
            kind: BundleKind::Generic,
            ..sample_bundle(0.99)
        };
        assert!(select_auto_applied_bundles(&[generic], true, 0.0).is_empty());
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

        assert_eq!(
            ids.iter().map(|bundle| bundle.id).collect::<Vec<_>>(),
            vec![a.id]
        );
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
            BundleKind::PhotoBurst,
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

        let report = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            false,
        )
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
    async fn apply_bundles_refuses_frontend_that_drops_a_member() {
        let dir = TempDir::new().unwrap();
        let src1 = dir.path().join("a.jpg");
        let src2 = dir.path().join("b.jpg");
        std::fs::write(&src1, b"a").unwrap();
        std::fs::write(&src2, b"b").unwrap();
        let dst1 = dir.path().join("out/a.jpg");
        let dst2 = dir.path().join("out/b.jpg");
        let original = file_set_bundle(
            vec![
                sample_proposal(src1.clone(), &dst1),
                sample_proposal(src2.clone(), &dst2),
            ],
            dir.path().join("out"),
        );
        let mut reviewed = original.clone();
        reviewed.members.pop();
        let shelf = NoopBackup::new();
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &shelf,
            progress: &NullProgress,
        };

        let report = apply_bundles(&[original], &[reviewed], &deps, false)
            .await
            .unwrap();

        assert_eq!(report.bundles_failed, 1);
        assert_eq!(report.bundles_applied, 0);
        assert!(src1.exists() && src2.exists(), "no member may move");
        assert!(!dst1.exists() && !dst2.exists());
        assert!(
            shelf.shelved.lock().unwrap().is_empty(),
            "reconciliation must fail before shelving"
        );
    }

    #[tokio::test]
    async fn apply_bundles_accepts_semantic_label_and_filename_edits() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("Screenshot.png");
        std::fs::write(&src, b"screen").unwrap();
        let target_parent = dir.path().join("out");
        let original_target = target_parent.join("screenshots/Screenshot.png");
        let original = BundleProposal::new(
            dir.path().to_path_buf(),
            BundleKind::SemanticCollection {
                label: "screenshots".to_string(),
            },
            target_parent.clone(),
            vec![sample_proposal(src.clone(), &original_target)],
            0.8,
            "shared OCR evidence".to_string(),
        )
        .unwrap();
        let mut reviewed = original.clone();
        reviewed.kind = BundleKind::SemanticCollection {
            label: "project".to_string(),
        };
        reviewed.members[0].change_type = ChangeType::RenameAndMove;
        reviewed.members[0].proposed_name = "project_homepage.png".to_string();
        let edited_target = target_parent.join("project/project_homepage.png");
        reviewed.members[0].proposed_path.clone_from(&edited_target);
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };

        let report = apply_bundles(&[original], &[reviewed], &deps, false)
            .await
            .unwrap();

        assert_eq!(report.bundles_applied, 1);
        assert!(!src.exists());
        assert_eq!(std::fs::read(edited_target).unwrap(), b"screen");
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
        let report = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            false,
        )
        .await
        .unwrap();

        assert_eq!(report.bundles_failed, 1);
        // All-or-nothing: neither member moved, no partial state.
        assert!(src1.exists() && src2.exists(), "no member may move");
        assert!(!dst1.exists(), "first target must stay untouched");
        assert_eq!(std::fs::read(&dst2).unwrap(), b"existing");
    }

    #[tokio::test]
    async fn apply_file_set_bundle_reverses_completed_moves_on_mid_bundle_failure() {
        // Member 1 moves cleanly; member 2's move fails mid-bundle (its target
        // parent is occupied by a regular file, so create_dir_all errors). The
        // executor must reverse the completed move so no partial bundle state
        // survives — member 1 is back at its origin.
        let dir = TempDir::new().unwrap();
        let src1 = dir.path().join("IMG_001.jpg");
        let src2 = dir.path().join("IMG_002.jpg");
        std::fs::write(&src1, b"one").unwrap();
        std::fs::write(&src2, b"two").unwrap();
        let dst1 = dir.path().join("out/IMG_001.jpg");
        // src2's destination sits under a path that is a FILE, not a dir.
        let blocker = dir.path().join("blocked");
        std::fs::write(&blocker, b"i am a file").unwrap();
        let dst2 = blocker.join("sub/IMG_002.jpg");

        let members = vec![
            sample_proposal(src1.clone(), &dst1),
            sample_proposal(src2.clone(), &dst2),
        ];
        let bundle = file_set_bundle(members, dir.path().join("out"));
        let shelf = NoopBackup::new();
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &shelf,
            progress: &NullProgress,
        };

        let report = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            false,
        )
        .await
        .unwrap();

        assert_eq!(report.bundles_failed, 1, "the bundle apply fails");
        assert_eq!(report.bundles_applied, 0);
        // All-or-nothing: member 1's completed move was reversed.
        assert!(src1.exists(), "member 1 must be restored to its origin");
        assert!(!dst1.exists(), "member 1's destination must be undone");
        assert!(src2.exists(), "member 2 never moved");
        assert_eq!(std::fs::read(&src1).unwrap(), b"one");
        // Both were shelved before their moves, so rollback could also recover.
        let shelved = shelf.shelved.lock().unwrap().clone();
        assert!(shelved.contains(&bundle.members[0].id));
    }

    #[tokio::test]
    async fn apply_file_set_bundle_refuses_two_members_targeting_same_path() {
        // Within-bundle duplicate-target guard: two members resolve to the same
        // destination. Pre-flight must refuse before moving anything.
        let dir = TempDir::new().unwrap();
        let src1 = dir.path().join("a.jpg");
        let src2 = dir.path().join("b.jpg");
        std::fs::write(&src1, b"a").unwrap();
        std::fs::write(&src2, b"b").unwrap();
        let shared = dir.path().join("out/dup.jpg");

        let members = vec![
            sample_proposal(src1.clone(), &shared),
            sample_proposal(src2.clone(), &shared),
        ];
        let bundle = file_set_bundle(members, dir.path().join("out"));
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let report = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            false,
        )
        .await
        .unwrap();

        assert_eq!(report.bundles_failed, 1);
        assert_eq!(report.bundles_applied, 0);
        // Nothing moved — pre-flight caught it before any filesystem mutation.
        assert!(src1.exists() && src2.exists());
        assert!(!shared.exists());
    }

    #[tokio::test]
    async fn apply_file_set_bundle_aborts_when_a_member_changed_since_scan() {
        // Member-level TOCTOU: one member carries a scan-time hash; its file is
        // edited before apply. Pre-flight must refuse the whole bundle.
        let dir = TempDir::new().unwrap();
        let src1 = dir.path().join("a.jpg");
        let src2 = dir.path().join("b.jpg");
        std::fs::write(&src1, b"a-before").unwrap();
        std::fs::write(&src2, b"b").unwrap();
        let dst1 = dir.path().join("out/a.jpg");
        let dst2 = dir.path().join("out/b.jpg");

        let mut m1 = sample_proposal(src1.clone(), &dst1);
        m1.content_hash = Some(blake3::hash(b"a-before").to_hex().to_string());
        let m2 = sample_proposal(src2.clone(), &dst2);
        // The user edits member 1 after review.
        std::fs::write(&src1, b"a-EDITED").unwrap();

        let bundle = file_set_bundle(vec![m1, m2], dir.path().join("out"));
        let deps = ExecutorDeps {
            change_log: &RecordingLog::new(),
            backup_store: &NoopBackup::new(),
            progress: &NullProgress,
        };
        let report = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            false,
        )
        .await
        .unwrap();

        assert_eq!(report.bundles_failed, 1, "changed member fails the bundle");
        assert_eq!(report.bundles_applied, 0);
        // All-or-nothing: neither member moved.
        assert!(src1.exists() && src2.exists());
        assert!(!dst1.exists() && !dst2.exists());
    }

    #[tokio::test]
    async fn apply_file_set_bundle_dry_run_reports_same_disposition_as_real_apply() {
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
        let report = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            true,
        )
        .await
        .unwrap();
        assert_eq!(
            report.bundles_applied, 1,
            "dry-run reports what would apply"
        );
        assert!(src.exists(), "dry-run must not move the source");
        assert!(!dst.exists(), "dry-run must not create the target");

        let applied = apply_bundles(
            std::slice::from_ref(&bundle),
            std::slice::from_ref(&bundle),
            &deps,
            false,
        )
        .await
        .unwrap();
        assert_eq!(report.bundles_applied, applied.bundles_applied);
        assert!(!src.exists());
        assert!(dst.exists());
    }

    #[test]
    fn destination_ledger_catches_cross_case_collisions() {
        let dir = TempDir::new().unwrap();
        let source_a = dir.path().join("a.txt");
        let source_b = dir.path().join("b.txt");
        let target = dir.path().join("Organized/item.txt");
        let first = sample_proposal(source_a, &target);
        let second_target = dir.path().join("Organized/ITEM.txt");
        let second = sample_proposal(source_b, &second_target);
        let decisions = vec![
            ReviewDecision::Approve(first.id),
            ReviewDecision::Approve(second.id),
        ];
        let error = validate_destination_ledger(&[first, second], &decisions, &[])
            .expect_err("case-folded targets must be held before apply");
        assert!(error.to_string().contains("overlapping destinations"));
    }

    #[test]
    fn destination_ledger_catches_loose_target_inside_bundle_target() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("incoming/readme.txt");
        let bundle_root = dir.path().join("incoming/project");
        let target_parent = dir.path().join("organized");
        let bundle_member_target = target_parent.join("project/readme.txt");
        let loose = sample_proposal(source, &bundle_member_target);
        let structural_member =
            sample_proposal(bundle_root.join("readme.txt"), &bundle_member_target);
        let structural = BundleProposal::new(
            bundle_root,
            BundleKind::RustCrate,
            target_parent,
            vec![structural_member],
            0.9,
            "project".to_string(),
        )
        .unwrap();
        let decisions = vec![ReviewDecision::Approve(loose.id)];
        let error =
            validate_destination_ledger(&[loose], &decisions, std::slice::from_ref(&structural))
                .expect_err("a member cannot target inside a structural bundle root");
        assert!(error.to_string().contains("overlapping destinations"));
    }

    #[test]
    fn destination_ledger_rejects_existing_final_target() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("source.txt");
        let target = dir.path().join("organized/existing.txt");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(&target, b"already here").unwrap();
        let proposal = sample_proposal(source, &target);
        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        let error = validate_destination_ledger(&[proposal], &decisions, &[])
            .expect_err("existing final target must be held");
        assert!(error.to_string().contains("existing path"));
    }

    /// A target that exists because it *is* the source is a no-op, not a
    /// collision. Scan reorganizes in place, so without this the ledger would
    /// abort an entire run over one file that needed nothing done to it.
    #[test]
    fn destination_ledger_allows_target_equal_to_source() {
        let dir = TempDir::new().unwrap();
        let settled = dir.path().join("organized/settled.txt");
        std::fs::create_dir_all(settled.parent().unwrap()).unwrap();
        std::fs::write(&settled, b"already filed").unwrap();
        let proposal = sample_proposal(settled.clone(), &settled);
        let decisions = vec![ReviewDecision::Approve(proposal.id)];
        validate_destination_ledger(&[proposal], &decisions, &[])
            .expect("a file already at its destination must not fail the plan");
    }

    /// The self-target exemption is narrow: two different sources resolving to
    /// one path is still a collision, and an unrelated occupant still blocks.
    #[test]
    fn destination_ledger_still_rejects_other_sources_targeting_a_live_path() {
        let dir = TempDir::new().unwrap();
        let settled = dir.path().join("organized/settled.txt");
        std::fs::create_dir_all(settled.parent().unwrap()).unwrap();
        std::fs::write(&settled, b"already filed").unwrap();
        let intruder = sample_proposal(dir.path().join("elsewhere.txt"), &settled);
        let decisions = vec![ReviewDecision::Approve(intruder.id)];
        let error = validate_destination_ledger(&[intruder], &decisions, &[])
            .expect_err("a different source must not overwrite an occupied path");
        assert!(error.to_string().contains("existing path"));
    }
}
