//! Migration service — moves files from a source tree into a user-defined target tree.
//!
//! Same handle drives CLI (`tidyup migrate`) and UI ("Migrate" button).

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::frontend::Level;
use tidyup_core::{ProgressReporter, Result, ReviewHandler};
use tidyup_domain::{RunMode, RunRecord, RunState};
use tidyup_pipeline::{migration::run_migration, profiler};
use uuid::Uuid;

use crate::executor::{
    apply_bundles, apply_loose_decisions, select_auto_applied_bundles, ApplyReport, ExecutorDeps,
};
use crate::ServiceContext;

#[allow(missing_debug_implementations)]
pub struct MigrationService {
    ctx: Arc<ServiceContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRequest {
    pub source: std::path::PathBuf,
    pub target: std::path::PathBuf,
    pub dry_run: bool,
    /// When true, auto-apply bundles whose confidence clears
    /// `bundle_min_confidence`. Set by the CLI only if `--yes`.
    #[serde(default)]
    pub auto_approve_bundles: bool,
    /// Lower bound on confidence for auto-applied bundles.
    #[serde(default = "default_bundle_confidence")]
    pub bundle_min_confidence: f32,
}

const fn default_bundle_confidence() -> f32 {
    0.85
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub proposed: usize,
    pub bundles: usize,
    pub unclassified: usize,
    pub approved: usize,
    pub applied: usize,
    pub skipped: usize,
    pub failed: usize,
    pub bundles_applied: usize,
    pub bundles_skipped: usize,
    pub bundles_failed: usize,
    pub run_id: Uuid,
}

impl MigrationService {
    #[must_use]
    pub fn new(ctx: Arc<ServiceContext>) -> Self {
        Self { ctx }
    }

    /// Run a full migration. Single entry point both frontends call.
    ///
    /// 1. Scans target tree, builds folder profiles (`Phase::ProfilingTarget`).
    /// 2. Classifies each source file (`Phase::Classifying`).
    /// 3. Persists proposals + bundles via [`ChangeLog`], tagged with a run id.
    /// 4. Calls `review.review(proposals)` for loose proposals.
    /// 5. Shelves and moves approved proposals atomically (per-file or per-bundle).
    ///
    /// [`ChangeLog`]: tidyup_core::storage::ChangeLog
    ///
    /// # Errors
    /// Propagates profiling, pipeline, storage, review, and apply errors.
    pub async fn run(
        &self,
        request: MigrationRequest,
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
    ) -> Result<MigrationReport> {
        let run = RunRecord::begin(
            RunMode::Migrate,
            request.source.clone(),
            Some(request.target.clone()),
        );
        let run_id = run.id;
        self.ctx.run_log.record_run(&run).await?;

        let result = self.try_run(&request, progress, review, run_id).await;

        match &result {
            Ok(_) => {
                self.ctx
                    .run_log
                    .finish_run(run_id, RunState::Completed)
                    .await?;
            }
            Err(_) => {
                let _ = self.ctx.run_log.finish_run(run_id, RunState::Failed).await;
            }
        }
        result
    }

    async fn try_run(
        &self,
        request: &MigrationRequest,
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
        run_id: Uuid,
    ) -> Result<MigrationReport> {
        progress
            .phase_started(tidyup_domain::Phase::ProfilingTarget, None)
            .await;
        let target_scan = profiler::scan_target(&request.target)?;
        let profile_cache =
            profiler::build_profile_cache(&target_scan, self.ctx.embeddings.as_ref()).await?;
        progress
            .phase_finished(tidyup_domain::Phase::ProfilingTarget)
            .await;

        let outcome = run_migration(
            &request.source,
            &profile_cache,
            self.ctx.embeddings.as_ref(),
            &self.ctx.extractors,
            &tidyup_domain::ClassifierConfig::default(),
            progress,
        )
        .await?;

        for proposal in &outcome.proposals {
            self.ctx
                .change_log
                .record_proposal(proposal, Some(run_id))
                .await?;
        }
        for bundle in &outcome.bundles {
            self.ctx
                .change_log
                .record_bundle(bundle, Some(run_id))
                .await?;
        }

        let decisions = if outcome.proposals.is_empty() {
            Vec::new()
        } else {
            review.review(outcome.proposals.clone()).await?
        };
        let approved = decisions
            .iter()
            .filter(|d| {
                matches!(
                    d,
                    tidyup_domain::ReviewDecision::Approve(_)
                        | tidyup_domain::ReviewDecision::Override { .. }
                )
            })
            .count();

        let deps = ExecutorDeps {
            change_log: self.ctx.change_log.as_ref(),
            backup_store: self.ctx.backup_store.as_ref(),
            progress,
        };

        let loose_report: ApplyReport = if decisions.is_empty() {
            ApplyReport::default()
        } else {
            apply_loose_decisions(&outcome.proposals, &decisions, &deps, request.dry_run).await?
        };

        let auto_apply_ids = select_auto_applied_bundles(
            &outcome.bundles,
            request.auto_approve_bundles,
            request.bundle_min_confidence,
        );
        if !outcome.bundles.is_empty() && auto_apply_ids.is_empty() {
            progress
                .message(
                    Level::Info,
                    &format!(
                        "{} bundle(s) held for review; run with --yes to auto-apply those above {:.2} confidence",
                        outcome.bundles.len(),
                        request.bundle_min_confidence,
                    ),
                )
                .await;
        }
        let bundle_report =
            apply_bundles(&outcome.bundles, &auto_apply_ids, &deps, request.dry_run).await?;

        Ok(MigrationReport {
            proposed: outcome.proposals.len(),
            bundles: outcome.bundles.len(),
            unclassified: outcome.unclassified.len(),
            approved,
            applied: loose_report.applied,
            skipped: loose_report.skipped,
            failed: loose_report.failed,
            bundles_applied: bundle_report.bundles_applied,
            bundles_skipped: bundle_report.bundles_skipped,
            bundles_failed: bundle_report.bundles_failed,
            run_id,
        })
    }

    /// Indexing-only pass. Out of scope for v0.1 — indexing happens implicitly
    /// inside `run_migration`. Exposed for future `tidyup status` flows.
    ///
    /// # Errors
    /// Always errors — intentionally unwired until a `status` subcommand lands.
    pub async fn index(&self, root: &Path, progress: &dyn ProgressReporter) -> Result<usize> {
        let _ = (&self.ctx, root, progress);
        anyhow::bail!("stand-alone indexing pass is not part of the v0.1 scope")
    }
}
