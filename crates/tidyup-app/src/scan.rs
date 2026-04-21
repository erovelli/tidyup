//! Scan service — classify files in place against a fixed taxonomy.
//!
//! Contrast with `migration`: scan targets a *taxonomy* (categories); migration targets
//! an existing folder *hierarchy*. Both produce `ChangeProposal`s that use the same
//! review flow.
//!
//! # Flow
//!
//! 1. Produce proposals via [`tidyup_pipeline::scan::run_scan`].
//! 2. Persist proposals + bundles to the change log, tagged with a fresh `run_id`.
//! 3. Review loose proposals via the supplied [`ReviewHandler`].
//! 4. Apply approved loose proposals via the [`crate::executor`] (shelve → move →
//!    mark applied).
//! 5. Auto-apply bundles iff `--yes` + confidence threshold; otherwise leave them
//!    pending with a message (bundle review UX is Phase 6+).

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::frontend::Level;
use tidyup_core::{ProgressReporter, Result, ReviewHandler};
use tidyup_domain::{RunMode, RunRecord, RunState};
use tidyup_pipeline::scan::{
    run_scan, AudioContext, ImageContext, MultimodalContext, ScanCandidate,
};
use uuid::Uuid;

use crate::executor::{
    apply_bundles, apply_loose_decisions, select_auto_applied_bundles, ApplyReport, ExecutorDeps,
};
use crate::ServiceContext;

#[allow(missing_debug_implementations)]
pub struct ScanService {
    ctx: Arc<ServiceContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanRequest {
    pub root: std::path::PathBuf,
    pub taxonomy_path: Option<std::path::PathBuf>,
    pub dry_run: bool,
    /// When true, auto-apply bundles whose confidence clears
    /// `bundle_min_confidence`. Required for bundles to apply in v0.1 (no
    /// interactive bundle review yet). Set by the CLI only if `--yes` is passed.
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
pub struct ScanReport {
    /// Loose (non-bundle) proposals produced.
    pub proposed: usize,
    /// Bundles produced.
    pub bundles: usize,
    /// Files the cascade couldn't classify.
    pub unclassified: usize,
    /// Loose proposals the user approved in review.
    pub approved: usize,
    /// Loose proposals successfully moved (post-review + shelve).
    pub applied: usize,
    /// Loose proposals skipped (reject, not approved).
    pub skipped: usize,
    /// Loose proposals whose move failed after approval.
    pub failed: usize,
    /// Bundles successfully moved atomically.
    pub bundles_applied: usize,
    /// Bundles skipped (held for review).
    pub bundles_skipped: usize,
    /// Bundle moves that failed.
    pub bundles_failed: usize,
    pub run_id: Uuid,
}

impl ScanService {
    #[must_use]
    pub fn new(ctx: Arc<ServiceContext>) -> Self {
        Self { ctx }
    }

    /// Run the scan pipeline end-to-end.
    ///
    /// `candidates` is the precomputed scan taxonomy — produced by the caller
    /// (typically `tidyup-embeddings-ort::taxonomy::default_taxonomy` +
    /// the embedding backend). Kept out of [`ScanRequest`] so the request
    /// stays serializable for UI state and so the service doesn't hardcode a
    /// specific embedding-backend crate.
    ///
    /// `image_candidates` / `audio_candidates` are the per-modality scan
    /// taxonomies, embedded in the modality-specific space (`SigLIP` / `CLAP`).
    /// Pass `&[]` when the modality backend isn't loaded — the routing
    /// short-circuits and the file falls through to the text Tier 2 path.
    ///
    /// # Errors
    /// Propagates pipeline, storage, review, and apply errors.
    pub async fn run(
        &self,
        request: ScanRequest,
        candidates: &[ScanCandidate],
        image_candidates: &[ScanCandidate],
        audio_candidates: &[ScanCandidate],
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
    ) -> Result<ScanReport> {
        let run = RunRecord::begin(RunMode::Scan, request.root.clone(), None);
        let run_id = run.id;
        self.ctx.run_log.record_run(&run).await?;

        let outcome_result = self
            .try_run(
                &request,
                candidates,
                image_candidates,
                audio_candidates,
                progress,
                review,
                run_id,
            )
            .await;

        match &outcome_result {
            Ok(_) => {
                self.ctx
                    .run_log
                    .finish_run(run_id, RunState::Completed)
                    .await?;
            }
            Err(_) => {
                // Best-effort: record terminal state but don't mask original error.
                let _ = self.ctx.run_log.finish_run(run_id, RunState::Failed).await;
            }
        }

        outcome_result
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn try_run(
        &self,
        request: &ScanRequest,
        candidates: &[ScanCandidate],
        image_candidates: &[ScanCandidate],
        audio_candidates: &[ScanCandidate],
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
        run_id: Uuid,
    ) -> Result<ScanReport> {
        let output_root = request.root.clone();

        let multimodal = MultimodalContext {
            image: self
                .ctx
                .image_embeddings
                .as_ref()
                .filter(|_| !image_candidates.is_empty())
                .map(|backend| ImageContext {
                    backend: backend.as_ref(),
                    candidates: image_candidates,
                }),
            audio: self
                .ctx
                .audio_embeddings
                .as_ref()
                .filter(|_| !audio_candidates.is_empty())
                .map(|backend| AudioContext {
                    backend: backend.as_ref(),
                    candidates: audio_candidates,
                }),
        };

        let outcome = run_scan(
            &request.root,
            &output_root,
            candidates,
            self.ctx.embeddings.as_ref(),
            &multimodal,
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

        Ok(ScanReport {
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

    /// Single-file classification — not wired in v0.1. The scan pipeline is
    /// batch-oriented and always scopes against a source root.
    ///
    /// # Errors
    /// Always errors.
    pub async fn classify_one(&self, file: &Path) -> Result<tidyup_domain::ChangeProposal> {
        let _ = (&self.ctx, file);
        anyhow::bail!("single-file classification is not part of the v0.1 scope")
    }
}
