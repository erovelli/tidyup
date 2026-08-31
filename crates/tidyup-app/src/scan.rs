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
//! 3. Interactive frontends review the complete loose+bundle plan through
//!    [`ReviewHandler::review_all`]. Under `--yes`, loose move-only proposals
//!    and eligible opaque structural bundles follow separate restricted
//!    thresholds; file-set/generic bundles and renames remain pending.
//! 4. Apply approved changes via the [`crate::executor`] using
//!    shelve → write-ahead applied mark → move journaling.
//!    Bundles move atomically or stay pending — never partially.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::frontend::Level;
use tidyup_core::{ProgressReporter, Result, ReviewHandler, ReviewOutcome};
use tidyup_domain::{
    CapabilityManifest, FileProcessingRole, FileProcessingStage, RunMode, RunRecord, RunState,
};
use tidyup_pipeline::scan::{
    run_scan, AudioContext, ImageContext, MultimodalContext, ScanCandidate,
};
use uuid::Uuid;

use crate::executor::{
    apply_bundles, apply_envelope_capacity_limits, apply_loose_decisions,
    redirect_conflicting_directory_envelopes, select_bundle_decisions, validate_destination_ledger,
    ApplyReport, ExecutorDeps, DEFAULT_BUNDLE_MIN_CONFIDENCE,
};
use crate::processing::{attach_indexed_identities, record_source_outcomes, report_indexing};
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
    /// When true (the `--yes` path), auto-apply eligible opaque structural
    /// bundles whose confidence clears `bundle_min_confidence` without
    /// prompting. Soft/file-set and generic bundles remain review-only. When
    /// false, bundles are surfaced to the `ReviewHandler` for interactive
    /// per-bundle approval. Set by the CLI only if `--yes` is passed.
    #[serde(default)]
    pub auto_approve_bundles: bool,
    /// Lower bound on confidence for auto-applied bundles.
    #[serde(default = "default_bundle_confidence")]
    pub bundle_min_confidence: f32,
}

const fn default_bundle_confidence() -> f32 {
    DEFAULT_BUNDLE_MIN_CONFIDENCE
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    /// Readable files whose stable identity was persisted for this run.
    pub indexed: usize,
    /// Paths discovered but not readable/hashable during identity indexing.
    pub indexing_failed: usize,
    /// Loose (non-bundle) proposals produced.
    pub proposed: usize,
    /// Bundles produced.
    pub bundles: usize,
    /// Files the cascade couldn't classify.
    pub unclassified: usize,
    /// Files already at the destination the classifier chose. Successfully
    /// classified, no operation proposed — scan sorts in place.
    pub already_in_place: usize,
    /// Images the per-directory clustering work cap excluded from
    /// visual-collection discovery. They are still classified individually;
    /// what they lost was the chance to be grouped.
    pub visual_candidates_over_cap: usize,
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
    /// Semantic/extraction facilities actually available to this run.
    pub capabilities: CapabilityManifest,
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
    /// short-circuits and the file falls through to the general text path.
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
        let run = RunRecord::begin(RunMode::Scan, request.root.clone(), None)
            .with_capabilities(self.ctx.capabilities.clone());
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
            Err(error) => {
                // Best-effort: record terminal state but don't mask original error.
                let _ = self
                    .ctx
                    .file_index
                    .fail_incomplete_processing(
                        run_id,
                        FileProcessingRole::Source,
                        FileProcessingStage::Planning,
                        &format!("run failed before a terminal file outcome: {error}"),
                    )
                    .await;
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

        progress
            .message(
                Level::Info,
                &format!("capabilities: {}", self.ctx.capabilities.summary()),
            )
            .await;

        // Identity indexing is deliberately completed before semantic work so
        // the persistent index remains the source of truth even when a later
        // extractor, model, or router fails. The pipeline does not depend on
        // SQLite; it writes through the FileIndex port.
        let indexed = tidyup_pipeline::indexing::index_directory(
            &request.root,
            self.ctx.file_index.as_ref(),
            run_id,
            FileProcessingRole::Source,
        )
        .await?;
        report_indexing(&indexed, progress).await;
        progress
            .message(
                Level::Info,
                &format!(
                    "indexed {} file(s); {} indexing failure(s)",
                    indexed.indexed.len(),
                    indexed.failed
                ),
            )
            .await;

        let image_concepts = if let Some(backend) = self.ctx.image_embeddings.as_deref() {
            match tidyup_pipeline::semantic::prepare_visual_concepts(backend).await {
                Ok(concepts) => concepts,
                Err(error) => {
                    progress
                        .message(
                            Level::Warn,
                            &format!(
                                "visual concept preparation failed; semantic image renames disabled: {error}"
                            ),
                        )
                        .await;
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        let multimodal = MultimodalContext {
            image: self
                .ctx
                .image_embeddings
                .as_ref()
                .filter(|_| !image_candidates.is_empty())
                .map(|backend| ImageContext {
                    backend: backend.as_ref(),
                    candidates: image_candidates,
                    concepts: &image_concepts,
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
            artifact_store: Some(self.ctx.file_index.as_ref()),
        };

        let text_backend = self.ctx.text.as_deref();
        // The classifier config (rename thresholds + optional reranker) is
        // materialised from the loaded TidyupConfig at context-build time; see
        // `classifier_config_for`.
        // Reuse the identities indexing just streamed: the proposal keeps the
        // FileId and the BLAKE3 already computed instead of re-hashing.
        let identities = tidyup_pipeline::indexing::SourceIdentities::new(&indexed.indexed);
        let mut outcome = run_scan(
            &request.root,
            &identities,
            &output_root,
            candidates,
            self.ctx.embeddings.as_ref(),
            &multimodal,
            text_backend,
            &self.ctx.extractors,
            &self.ctx.classifier,
            progress,
        )
        .await?;

        let redirected = redirect_conflicting_directory_envelopes(&mut outcome.bundles);
        if redirected > 0 {
            progress
                .message(
                    Level::Warn,
                    &format!(
                        "{redirected} directory envelope(s) target an existing folder and were redirected to a temporal import parent for review",
                    ),
                )
                .await;
        }

        let capacity = apply_envelope_capacity_limits(
            &mut outcome.bundles,
            self.ctx.classifier.directory_envelopes.backup_warn_bytes,
            self.ctx
                .classifier
                .directory_envelopes
                .backup_hard_limit_bytes,
        );
        if capacity.warning_count > 0 {
            progress
                .message(
                    Level::Warn,
                    &format!(
                        "{} directory envelope backup(s) exceed the configured size warning; inspect available disk space before applying",
                        capacity.warning_count,
                    ),
                )
                .await;
        }
        if capacity.held_count > 0 {
            progress
                .message(
                    Level::Warn,
                    &format!(
                        "{} directory envelope backup(s) exceed the unattended size limit and require explicit approval",
                        capacity.held_count,
                    ),
                )
                .await;
        }

        attach_indexed_identities(
            &indexed.indexed,
            &mut outcome.proposals,
            &mut outcome.bundles,
        );
        record_source_outcomes(
            self.ctx.file_index.as_ref(),
            run_id,
            &indexed.indexed,
            &outcome.proposals,
            &outcome.bundles,
            &outcome.unclassified,
            &outcome.already_in_place,
        )
        .await?;

        for proposal in &outcome.proposals {
            self.ctx
                .change_log
                .record_proposal(proposal, Some(run_id))
                .await?;
        }

        let review_outcome = if request.auto_approve_bundles {
            let decisions = if outcome.proposals.is_empty() {
                Vec::new()
            } else {
                review.review(outcome.proposals.clone()).await?
            };
            let approved_bundles = select_bundle_decisions(
                &outcome.bundles,
                true,
                request.bundle_min_confidence,
                review,
            )
            .await?;
            ReviewOutcome {
                decisions,
                approved_bundles,
            }
        } else {
            review
                .review_all(outcome.proposals.clone(), outcome.bundles.clone())
                .await?
        };
        let decisions = review_outcome.decisions;
        let approved_bundles = review_outcome.approved_bundles;
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

        validate_destination_ledger(&outcome.proposals, &decisions, &approved_bundles)?;

        let loose_report: ApplyReport = if decisions.is_empty() {
            ApplyReport::default()
        } else {
            apply_loose_decisions(&outcome.proposals, &decisions, &deps, request.dry_run).await?
        };

        if !outcome.bundles.is_empty() && approved_bundles.is_empty() {
            let held = outcome.bundles.len();
            let detail = if request.auto_approve_bundles {
                format!(
                    "none cleared the {:.2} auto-apply threshold",
                    request.bundle_min_confidence
                )
            } else {
                "none approved in review".to_string()
            };
            progress
                .message(Level::Info, &format!("{held} bundle(s) held; {detail}"))
                .await;
        }
        for original in &outcome.bundles {
            let reviewed = approved_bundles
                .iter()
                .find(|bundle| bundle.id == original.id)
                .unwrap_or(original);
            self.ctx
                .change_log
                .record_bundle(reviewed, Some(run_id))
                .await?;
        }
        let bundle_report =
            apply_bundles(&outcome.bundles, &approved_bundles, &deps, request.dry_run).await?;

        Ok(ScanReport {
            indexed: indexed.indexed.len(),
            indexing_failed: indexed.failed,
            proposed: outcome.proposals.len(),
            bundles: outcome.bundles.len(),
            unclassified: outcome.unclassified.len(),
            already_in_place: outcome.already_in_place.len(),
            visual_candidates_over_cap: outcome.visual_candidates_over_cap,
            approved,
            applied: loose_report.applied,
            skipped: loose_report.skipped,
            failed: loose_report.failed,
            bundles_applied: bundle_report.bundles_applied,
            bundles_skipped: bundle_report.bundles_skipped,
            bundles_failed: bundle_report.bundles_failed,
            run_id,
            capabilities: self.ctx.capabilities.clone(),
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
