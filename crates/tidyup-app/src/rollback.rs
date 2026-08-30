//! Rollback service — restore a previous run from the backup shelf.
//!
//! Rollback proceeds in two passes:
//!
//! 1. **Bundles first.** Every applied bundle for the run is found; its shelf
//!    record is looked up by `change_id == bundle.id`; the destination
//!    subtree is removed; the shelf copy is restored to `bundle.root`. All
//!    members are then marked `Unshelved` via `mark_bundle_unshelved`.
//!
//! 2. **Loose proposals.** For each applied loose proposal in the run, look up
//!    the shelf record by `change_id == proposal.id`, delete the destination
//!    path, copy the shelved original back into place, and mark the proposal
//!    `Unshelved`.
//!
//! Rollback never destroys data. Before touching any destination, the shelf
//! copy is verified intact and both the live destination and the original slot
//! are hash-compared against it (via [`BackupStore::precheck_restore`]). Three
//! outcomes per item:
//!
//! - **Restored** — shelf intact, destination unedited (or already gone),
//!   original slot free (or already holding the shelved content): the
//!   destination is renamed aside, the original copied back from the shelf,
//!   and only then is the displaced destination removed. If the restore copy
//!   fails mid-way, the displaced destination is renamed back — a failed
//!   restore never leaves the data solely on the shelf.
//! - **Conflict** — the destination was edited after apply, or a *new* file
//!   occupies the original slot: nothing is touched and the item is tallied as
//!   a conflict, so the user's data survives.
//! - **Failure** — a missing shelf record, missing/corrupt shelf copy, or
//!   filesystem error: the destination is preserved (pre-restore failures
//!   leave it untouched; mid-restore failures rename the displaced copy back)
//!   and the item is tallied as a failure.
//!
//! File-set bundles precheck **every** member before restoring any: one edited
//! member conflicts the whole bundle (never restore some members while an
//! edited sibling stays). A mid-restore I/O failure leaves already-restored
//! members restored (their backup rows are `Unshelved`); a retry recognises
//! those as done and converges to full restoration — partial progress is
//! never bricked and nothing is lost.
//!
//! Partial rollback is honest and user-visible via [`RollbackReport`]. The run
//! is flipped to `RolledBack` **only when every item restored cleanly** (no
//! conflicts, no failures); otherwise the run keeps its prior state so a later
//! `rollback` retries just the still-applied items — the conflicted/failed ones
//! stay shelved and `Applied`. (`PartiallyRolledBack` isn't a `RunState` yet.)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::Result;
use tidyup_domain::{
    BackupRecord, BackupStatus, BundleProposal, ChangeProposal, Phase, RestorePrecheck, RunRecord,
    RunState,
};
use uuid::Uuid;

use crate::ServiceContext;

#[allow(missing_debug_implementations)]
pub struct RollbackService {
    ctx: Arc<ServiceContext>,
}

/// Per-item result of a rollback attempt. `Conflict` carries a short human
/// description of why the item was deliberately left in place.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RestoreOutcome {
    Restored,
    Conflict(&'static str),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackReport {
    pub run_id: Uuid,
    pub restored: usize,
    pub bundles_restored: usize,
    pub failures: usize,
    /// Items skipped because their destination was modified after apply.
    /// Restoring nothing for these preserves the user's edits. Non-zero means
    /// the run was *not* flipped to `RolledBack` and can be retried.
    #[serde(default)]
    pub conflicts: usize,
}

impl RollbackService {
    #[must_use]
    pub fn new(ctx: Arc<ServiceContext>) -> Self {
        Self { ctx }
    }

    /// Roll back every applied change in `run_id` by restoring from the
    /// backup shelf.
    ///
    /// # Errors
    /// Propagates storage errors. Individual restore failures are tallied into
    /// [`RollbackReport::failures`] rather than aborting the whole rollback.
    pub async fn rollback_run(
        &self,
        run_id: Uuid,
        progress: &dyn ProgressReporter,
    ) -> Result<RollbackReport> {
        progress.phase_started(Phase::Rollback, None).await;

        let bundles = self.ctx.change_log.applied_bundles_for_run(run_id).await?;
        let proposals = self
            .ctx
            .change_log
            .applied_proposals_for_run(run_id)
            .await?;

        let mut report = RollbackReport {
            run_id,
            restored: 0,
            bundles_restored: 0,
            failures: 0,
            conflicts: 0,
        };

        // Bundles first — atomic per-bundle restore.
        for bundle in &bundles {
            match self.rollback_bundle(bundle, progress).await {
                Ok(RestoreOutcome::Restored) => {
                    report.bundles_restored = report.bundles_restored.saturating_add(1);
                }
                Ok(RestoreOutcome::Conflict(why)) => {
                    report.conflicts = report.conflicts.saturating_add(1);
                    progress
                        .message(
                            Level::Warn,
                            &format!(
                                "bundle rollback skipped for {}: {why} — left untouched to \
                                 preserve your data",
                                bundle.root.display()
                            ),
                        )
                        .await;
                }
                Err(e) => {
                    report.failures = report.failures.saturating_add(1);
                    progress
                        .message(
                            Level::Warn,
                            &format!("bundle rollback failed for {}: {e}", bundle.root.display()),
                        )
                        .await;
                }
            }
        }

        for proposal in &proposals {
            match self.rollback_proposal(proposal, progress).await {
                Ok(RestoreOutcome::Restored) => {
                    report.restored = report.restored.saturating_add(1);
                }
                Ok(RestoreOutcome::Conflict(why)) => {
                    report.conflicts = report.conflicts.saturating_add(1);
                    progress
                        .message(
                            Level::Warn,
                            &format!(
                                "rollback skipped for {}: {why} — left untouched to preserve \
                                 your data",
                                proposal.original_path.display()
                            ),
                        )
                        .await;
                }
                Err(e) => {
                    report.failures = report.failures.saturating_add(1);
                    progress
                        .message(
                            Level::Warn,
                            &format!(
                                "rollback failed for {}: {e}",
                                proposal.original_path.display()
                            ),
                        )
                        .await;
                }
            }
        }

        // Flip to RolledBack only when everything restored cleanly. If any item
        // conflicted or errored, keep the run's prior state so a later rollback
        // retries the still-applied items — and so re-running a fully-successful
        // rollback stays a no-op instead of masking failures.
        if report.failures == 0 && report.conflicts == 0 {
            self.ctx
                .run_log
                .finish_run(run_id, RunState::RolledBack)
                .await?;
        }

        progress.phase_finished(Phase::Rollback).await;
        Ok(report)
    }

    async fn rollback_proposal(
        &self,
        proposal: &ChangeProposal,
        progress: &dyn ProgressReporter,
    ) -> Result<RestoreOutcome> {
        let record = match self
            .resolve_record(proposal.id, &proposal.original_path)
            .await?
        {
            RecordState::Restorable(record) => record,
            RecordState::AlreadyRestored | RecordState::NeverMoved => {
                // AlreadyRestored: a crash between restore and the status update
                // left the original back in place. NeverMoved: a crash after the
                // write-ahead journal but before shelve+move left it at its
                // origin. Either way there is nothing to move — heal the
                // proposal status and count it restored.
                self.ctx.change_log.mark_unshelved(proposal.id).await?;
                return Ok(RestoreOutcome::Restored);
            }
        };

        // Never touch the destination unverified. If the shelf is missing this
        // errors (destination untouched); if the destination was edited or the
        // original slot holds new content it returns Conflict (nothing touched).
        match self
            .ctx
            .backup_store
            .precheck_restore(&record, &proposal.proposed_path)
            .await?
        {
            RestorePrecheck::Ready => {}
            RestorePrecheck::ShelfUnusable(why) => return Err(anyhow::anyhow!(why)),
            RestorePrecheck::DestinationModified => {
                return Ok(RestoreOutcome::Conflict(CONFLICT_DEST_MODIFIED))
            }
            RestorePrecheck::OriginalOccupied => {
                return Ok(RestoreOutcome::Conflict(CONFLICT_ORIGINAL_OCCUPIED))
            }
        }

        self.restore_with_displacement(&record, &proposal.proposed_path)
            .await?;
        self.ctx.change_log.mark_unshelved(proposal.id).await?;

        progress
            .item_completed(
                Phase::Rollback,
                ProgressItem {
                    label: proposal.original_path.display().to_string(),
                    current: 1,
                    total: None,
                },
            )
            .await;
        Ok(RestoreOutcome::Restored)
    }

    /// Restore `record` while keeping the destination recoverable throughout:
    /// rename the destination aside, copy the shelf content back to the
    /// original path, and only then remove the displaced copy. A failure in
    /// the restore copy renames the displaced destination back, so no step of
    /// a failed rollback ever leaves the data solely on the shelf.
    async fn restore_with_displacement(
        &self,
        record: &BackupRecord,
        destination: &Path,
    ) -> Result<()> {
        let displaced = displace_destination(destination)?;
        if let Err(e) = self.ctx.backup_store.restore(record).await {
            if let Some(temp) = &displaced {
                if let Err(undo) = std::fs::rename(temp, destination) {
                    return Err(anyhow::anyhow!(
                        "restore failed ({e}); renaming displaced destination back also \
                         failed ({undo}) — the moved content is preserved at {}",
                        temp.display(),
                    ));
                }
            }
            return Err(e);
        }
        if let Some(temp) = &displaced {
            // The original is restored; the displaced destination is now a
            // duplicate. Best-effort removal — a leftover temp wastes space but
            // never loses data.
            let removed = if temp.is_dir() {
                std::fs::remove_dir_all(temp)
            } else {
                std::fs::remove_file(temp)
            };
            if let Err(e) = removed {
                tracing::warn!(
                    "could not remove displaced rollback copy {}: {e}",
                    temp.display()
                );
            }
        }
        Ok(())
    }

    /// Look up the backup record for `change_id` and classify what a retrying
    /// rollback should do with it.
    async fn resolve_record(&self, change_id: Uuid, original: &Path) -> Result<RecordState> {
        let Some(record) = self.ctx.backup_store.find_by_change_id(change_id).await? else {
            // No shelf record. Under write-ahead journaling a change is shelved
            // before it is marked applied, so the only way an *applied* change
            // has no shelf record is a crash after the journal write but before
            // shelve+move — the original never moved. If it's still there,
            // there is nothing to restore; if it's gone, the state is
            // genuinely inconsistent and we surface it.
            if original.exists() {
                return Ok(RecordState::NeverMoved);
            }
            return Err(anyhow::anyhow!(
                "no shelved backup for change {change_id} and the original {} is missing",
                original.display(),
            ));
        };
        match record.status {
            BackupStatus::Shelved => Ok(RecordState::Restorable(record)),
            BackupStatus::Unshelved => {
                if original.exists() {
                    Ok(RecordState::AlreadyRestored)
                } else {
                    Err(anyhow::anyhow!(
                        "backup for {} was already consumed but the original is missing — \
                         restore it by hand from {}",
                        original.display(),
                        record.backup_path.display(),
                    ))
                }
            }
            BackupStatus::Expired => Err(anyhow::anyhow!(
                "backup for {} expired and its shelf content was pruned",
                original.display(),
            )),
        }
    }

    async fn rollback_bundle(
        &self,
        bundle: &BundleProposal,
        progress: &dyn ProgressReporter,
    ) -> Result<RestoreOutcome> {
        // File-set bundles (photo bursts, music albums, document series) were
        // applied by moving each member individually, so they restore the same
        // way — per member, by the member's own shelf record.
        if bundle.kind.moves_as_file_set() {
            return self.rollback_file_set_bundle(bundle, progress).await;
        }

        let record = match self.resolve_record(bundle.id, &bundle.root).await? {
            RecordState::Restorable(record) => record,
            RecordState::AlreadyRestored | RecordState::NeverMoved => {
                self.ctx.change_log.mark_bundle_unshelved(bundle.id).await?;
                return Ok(RestoreOutcome::Restored);
            }
        };

        let leaf = bundle
            .root
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("bundle root has no file_name"))?;
        let moved_to = bundle.target_parent.join(leaf);

        match self
            .ctx
            .backup_store
            .precheck_restore(&record, &moved_to)
            .await?
        {
            RestorePrecheck::Ready => {}
            RestorePrecheck::ShelfUnusable(why) => return Err(anyhow::anyhow!(why)),
            RestorePrecheck::DestinationModified => {
                return Ok(RestoreOutcome::Conflict(CONFLICT_DEST_MODIFIED))
            }
            RestorePrecheck::OriginalOccupied => {
                return Ok(RestoreOutcome::Conflict(CONFLICT_ORIGINAL_OCCUPIED))
            }
        }

        self.restore_with_displacement(&record, &moved_to).await?;
        self.ctx.change_log.mark_bundle_unshelved(bundle.id).await?;

        progress
            .item_completed(
                Phase::Rollback,
                ProgressItem {
                    label: bundle.root.display().to_string(),
                    current: 1,
                    total: None,
                },
            )
            .await;
        Ok(RestoreOutcome::Restored)
    }

    /// Roll back a file-set bundle as a unit. Every member's shelf and
    /// destination are prechecked *first*: if any destination was edited (or
    /// any original slot holds new content), the whole bundle is reported as a
    /// conflict and nothing is touched — we never restore some members while
    /// leaving an edited sibling in place. Only when every member is `Ready`
    /// do we restore, each via [`Self::restore_with_displacement`].
    ///
    /// A mid-restore I/O failure leaves the members restored so far in place
    /// (their backup rows are already `Unshelved`) and propagates the error
    /// before `mark_bundle_unshelved`; the bundle stays `Applied` and a retry
    /// recognises the already-restored members (`Unshelved` + original
    /// present) and completes the rest — partial progress converges to full
    /// restoration instead of bricking.
    ///
    /// This also recovers a bundle whose *apply* was interrupted by a crash:
    /// write-ahead journaling marks the bundle applied before moving any member,
    /// so a partially-applied bundle enumerates here; members that were never
    /// shelved (still at their origin) resolve to `NeverMoved` and are skipped,
    /// while the moved members restore normally.
    async fn rollback_file_set_bundle(
        &self,
        bundle: &BundleProposal,
        progress: &dyn ProgressReporter,
    ) -> Result<RestoreOutcome> {
        // Phase 1: resolve + precheck every member before mutating anything.
        let mut prepared: Vec<(BackupRecord, &ChangeProposal)> =
            Vec::with_capacity(bundle.members.len());
        for member in &bundle.members {
            let record = match self
                .resolve_record(member.id, &member.original_path)
                .await?
            {
                RecordState::Restorable(record) => record,
                // AlreadyRestored: restored by an earlier partial rollback.
                // NeverMoved: a crash left the bundle journaled applied before
                // this member was shelved+moved (it's still at its origin).
                // Either way, skip it.
                RecordState::AlreadyRestored | RecordState::NeverMoved => continue,
            };
            match self
                .ctx
                .backup_store
                .precheck_restore(&record, &member.proposed_path)
                .await?
            {
                RestorePrecheck::Ready => prepared.push((record, member)),
                RestorePrecheck::ShelfUnusable(why) => return Err(anyhow::anyhow!(why)),
                RestorePrecheck::DestinationModified => {
                    return Ok(RestoreOutcome::Conflict(CONFLICT_MEMBER_MODIFIED))
                }
                RestorePrecheck::OriginalOccupied => {
                    return Ok(RestoreOutcome::Conflict(CONFLICT_MEMBER_ORIGINAL_OCCUPIED))
                }
            }
        }

        // Phase 2: every pending member is safe — restore them, then mark the
        // bundle unshelved. An error here propagates before the mark, so the
        // bundle stays retryable (see the docstring's convergence contract).
        for (record, member) in &prepared {
            self.restore_with_displacement(record, &member.proposed_path)
                .await
                .with_context(|| {
                    format!(
                        "restoring file-set member {} — completed members stay restored; \
                         re-run rollback to finish the bundle",
                        member.original_path.display(),
                    )
                })?;
        }
        self.ctx.change_log.mark_bundle_unshelved(bundle.id).await?;
        progress
            .item_completed(
                Phase::Rollback,
                ProgressItem {
                    label: format!("{} ({} files)", bundle.kind.as_str(), bundle.members.len()),
                    current: 1,
                    total: None,
                },
            )
            .await;
        Ok(RestoreOutcome::Restored)
    }

    /// List recorded runs, most recent first.
    ///
    /// # Errors
    /// Propagates storage failures.
    pub async fn list_runs(&self) -> Result<Vec<RunRecord>> {
        self.ctx.run_log.list_runs().await
    }

    /// Expire shelved backups older than `days`, marking them `Expired` and
    /// best-effort removing their content from the shelf. Returns the count
    /// pruned. A thin wrapper over the [`BackupStore`](tidyup_core::storage::BackupStore)
    /// port so the CLI doesn't reach into the context directly.
    ///
    /// # Errors
    /// Propagates storage failures.
    pub async fn prune_backups(&self, days: u32) -> Result<usize> {
        self.ctx.backup_store.prune_older_than_days(days).await
    }

    /// Delete cached semantic artifacts older than `days` and return the row
    /// count removed. This cache is reconstructible and independent of the
    /// backup shelf's rollback guarantees.
    ///
    /// # Errors
    /// Propagates storage failures.
    pub async fn prune_semantic_artifacts(&self, days: u32) -> Result<usize> {
        self.ctx
            .file_index
            .prune_semantic_artifacts_older_than_days(days)
            .await
    }
}

/// What a retrying rollback should do with a change's backup record.
#[derive(Debug)]
enum RecordState {
    /// Still shelved — restore it (after precheck).
    Restorable(BackupRecord),
    /// Already restored by an earlier (possibly partial) rollback.
    AlreadyRestored,
    /// No shelf record and the original is still in place — the change was
    /// journaled applied (write-ahead) but a crash killed the process before it
    /// was shelved/moved. Nothing to restore; the file never left its origin.
    NeverMoved,
}

const CONFLICT_DEST_MODIFIED: &str = "destination modified since apply";
const CONFLICT_ORIGINAL_OCCUPIED: &str = "original location now holds different content";
const CONFLICT_MEMBER_MODIFIED: &str =
    "a bundle member's destination was modified since apply (bundles restore all-or-nothing)";
const CONFLICT_MEMBER_ORIGINAL_OCCUPIED: &str =
    "a bundle member's original location now holds different content (bundles restore \
     all-or-nothing)";

/// Rename a rollback destination (file or subtree) aside to a unique sibling
/// temp path, returning it. `Ok(None)` when the destination no longer exists.
/// A sibling rename stays on the same volume, so it is atomic and cheap; the
/// displaced copy is removed only after the shelf restore succeeds, and renamed
/// back if it fails.
fn displace_destination(path: &Path) -> Result<Option<PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    let name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("destination has no file name: {}", path.display()))?
        .to_string_lossy()
        .into_owned();
    let temp = path.with_file_name(format!(
        ".{name}.tidyup-restore-{}",
        Uuid::new_v4().simple()
    ));
    std::fs::rename(path, &temp).map_err(|e| {
        anyhow::anyhow!(
            "displacing destination {} -> {}: {e}",
            path.display(),
            temp.display(),
        )
    })?;
    Ok(Some(temp))
}
