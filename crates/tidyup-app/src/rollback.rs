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
//! A proposal or bundle that cannot be rolled back (missing shelf record,
//! target already moved again, filesystem error) is logged as a failure and
//! the rollback continues — partial rollback is honest and user-visible via
//! the returned [`RollbackReport`].
//!
//! The run record's state is updated to `RolledBack` on success and
//! `PartiallyRolledBack` isn't represented yet — when failures occur we still
//! flip the run to `RolledBack` but surface `failures > 0` in the report so
//! the user can rerun the command or inspect by hand.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::Result;
use tidyup_domain::{BundleProposal, ChangeProposal, Phase, RunRecord, RunState};
use uuid::Uuid;

use crate::ServiceContext;

#[allow(missing_debug_implementations)]
pub struct RollbackService {
    ctx: Arc<ServiceContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackReport {
    pub run_id: Uuid,
    pub restored: usize,
    pub bundles_restored: usize,
    pub failures: usize,
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
        };

        // Bundles first — atomic per-bundle restore.
        for bundle in &bundles {
            match self.rollback_bundle(bundle, progress).await {
                Ok(()) => report.bundles_restored += 1,
                Err(e) => {
                    report.failures += 1;
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
                Ok(()) => report.restored += 1,
                Err(e) => {
                    report.failures += 1;
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

        // Mark the run regardless — rollback is idempotent, and a partial
        // success is still a rollback attempt worth recording.
        self.ctx
            .run_log
            .finish_run(run_id, RunState::RolledBack)
            .await?;

        progress.phase_finished(Phase::Rollback).await;
        Ok(report)
    }

    async fn rollback_proposal(
        &self,
        proposal: &ChangeProposal,
        progress: &dyn ProgressReporter,
    ) -> Result<()> {
        let record = self
            .ctx
            .backup_store
            .find_by_change_id(proposal.id)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no shelved backup for change {} ({})",
                    proposal.id,
                    proposal.original_path.display(),
                )
            })?;

        // Remove the moved file at its destination if it's still there — the
        // restore copies from shelf to original_path, and leaving the dest in
        // place would silently leave two copies.
        if proposal.proposed_path.exists() {
            if proposal.proposed_path.is_dir() {
                std::fs::remove_dir_all(&proposal.proposed_path)?;
            } else {
                std::fs::remove_file(&proposal.proposed_path)?;
            }
        }

        self.ctx.backup_store.restore(&record).await?;
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
        Ok(())
    }

    async fn rollback_bundle(
        &self,
        bundle: &BundleProposal,
        progress: &dyn ProgressReporter,
    ) -> Result<()> {
        let record = self
            .ctx
            .backup_store
            .find_by_change_id(bundle.id)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no shelved backup for bundle {} ({})",
                    bundle.id,
                    bundle.root.display(),
                )
            })?;

        let leaf = bundle
            .root
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("bundle root has no file_name"))?;
        let moved_to = bundle.target_parent.join(leaf);
        if moved_to.exists() {
            if moved_to.is_dir() {
                std::fs::remove_dir_all(&moved_to)?;
            } else {
                std::fs::remove_file(&moved_to)?;
            }
        }

        self.ctx.backup_store.restore(&record).await?;
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
        Ok(())
    }

    /// List recorded runs, most recent first.
    ///
    /// # Errors
    /// Propagates storage failures.
    pub async fn list_runs(&self) -> Result<Vec<RunRecord>> {
        self.ctx.run_log.list_runs().await
    }
}
