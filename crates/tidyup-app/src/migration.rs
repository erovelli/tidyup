//! Migration service — moves files from a source tree into a user-defined target tree.
//!
//! Same handle drives CLI (`tidyup migrate`) and UI ("Migrate" button).
//!
//! # Phase 4 scope
//!
//! The service produces and persists proposals end-to-end. The **apply** step
//! (shelving + filesystem moves) is Phase 5 — this service stops at review.
//! That keeps the privacy / safety surface smaller while the pipeline
//! stabilises.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::{ProgressReporter, Result, ReviewHandler};
use tidyup_pipeline::{migration::run_migration, profiler};
use uuid::Uuid;

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
    /// 2. Indexes source tree (`Phase::Indexing`).
    /// 3. Classifies each source file (`Phase::Classifying`).
    /// 4. Persists proposals + bundles via [`ChangeLog`].
    /// 5. Calls `review.review(proposals)` for loose proposals.
    ///
    /// Apply (shelving + moves) is Phase 5. `applied` stays 0 here.
    ///
    /// [`ChangeLog`]: tidyup_core::storage::ChangeLog
    ///
    /// # Errors
    /// Propagates profiling, pipeline, storage, and review errors.
    pub async fn run(
        &self,
        request: MigrationRequest,
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
    ) -> Result<MigrationReport> {
        let run_id = Uuid::new_v4();

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
            self.ctx.change_log.record_proposal(proposal).await?;
        }
        for bundle in &outcome.bundles {
            self.ctx.change_log.record_bundle(bundle).await?;
        }

        let decisions = if outcome.proposals.is_empty() {
            Vec::new()
        } else {
            review.review(outcome.proposals.clone()).await?
        };
        let mut approved = 0usize;
        let mut skipped = 0usize;
        for d in &decisions {
            match d {
                tidyup_domain::ReviewDecision::Approve(_)
                | tidyup_domain::ReviewDecision::Override { .. } => approved += 1,
                tidyup_domain::ReviewDecision::Reject(_) => skipped += 1,
            }
        }

        // Apply: Phase 5.
        let _ = request.dry_run;

        Ok(MigrationReport {
            proposed: outcome.proposals.len(),
            bundles: outcome.bundles.len(),
            unclassified: outcome.unclassified.len(),
            approved,
            applied: 0,
            skipped,
            failed: 0,
            run_id,
        })
    }

    /// Indexing-only pass. Useful for `tidyup status` or UI initial load.
    ///
    /// Phase-5 territory — returns an error until the indexer wiring lands.
    ///
    /// # Errors
    /// Always errors until Phase 5.
    pub async fn index(&self, root: &Path, progress: &dyn ProgressReporter) -> Result<usize> {
        let _ = (&self.ctx, root, progress);
        anyhow::bail!("indexing-only pass not wired until Phase 5")
    }
}
