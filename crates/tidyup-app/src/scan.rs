//! Scan service — classify files in place against a fixed taxonomy.
//!
//! Contrast with `migration`: scan targets a *taxonomy* (categories); migration targets
//! an existing folder *hierarchy*. Both produce `ChangeProposal`s that use the same
//! review flow.
//!
//! # Phase 4 scope
//!
//! This service produces and persists proposals. The **apply** step (shelving
//! originals + executing filesystem moves) is Phase 5 and is intentionally absent
//! — the service ends the flow at review. That keeps the privacy / safety surface
//! smaller while the pipeline stabilises.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::{ProgressReporter, Result, ReviewHandler};
use tidyup_pipeline::scan::{run_scan, ScanCandidate};
use uuid::Uuid;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    /// Loose (non-bundle) proposals produced.
    pub proposed: usize,
    /// Bundles produced.
    pub bundles: usize,
    /// Files the cascade couldn't classify.
    pub unclassified: usize,
    /// Loose proposals the user approved in review. (Apply is Phase 5 — so
    /// `applied == 0` until then.)
    pub approved: usize,
    pub applied: usize,
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
    /// # Errors
    /// Propagates pipeline, storage, and review errors.
    pub async fn run(
        &self,
        request: ScanRequest,
        candidates: &[ScanCandidate],
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
    ) -> Result<ScanReport> {
        let run_id = Uuid::new_v4();
        let output_root = request.root.clone();

        let outcome = run_scan(
            &request.root,
            &output_root,
            candidates,
            self.ctx.embeddings.as_ref(),
            &self.ctx.extractors,
            &tidyup_domain::ClassifierConfig::default(),
            progress,
        )
        .await?;

        // Persist. Loose proposals first, then bundles (each atomic with its members).
        for proposal in &outcome.proposals {
            self.ctx.change_log.record_proposal(proposal).await?;
        }
        for bundle in &outcome.bundles {
            self.ctx.change_log.record_bundle(bundle).await?;
        }

        // Review loose proposals. Bundle surfacing to a frontend is not yet
        // defined at the port layer — for now they short-circuit review and
        // remain in Pending state in the change log.
        let decisions = if outcome.proposals.is_empty() {
            Vec::new()
        } else {
            review.review(outcome.proposals.clone()).await?
        };
        let approved = decisions
            .iter()
            .filter(|d| matches!(d, tidyup_domain::ReviewDecision::Approve(_)))
            .count();

        // Apply: Phase 5. For now `applied == 0` regardless of `dry_run`.
        let _ = request.dry_run;

        Ok(ScanReport {
            proposed: outcome.proposals.len(),
            bundles: outcome.bundles.len(),
            unclassified: outcome.unclassified.len(),
            approved,
            applied: 0,
            run_id,
        })
    }

    /// Stub retained for UI call sites; fully-implemented one-file classification
    /// is Phase 5 and requires threading a single [`ScanCandidate`] slice in.
    ///
    /// # Errors
    /// Always errors — this is a deliberate "not yet wired" boundary.
    pub async fn classify_one(&self, file: &Path) -> Result<tidyup_domain::ChangeProposal> {
        let _ = (&self.ctx, file);
        anyhow::bail!("single-file classification not wired until Phase 5")
    }
}
