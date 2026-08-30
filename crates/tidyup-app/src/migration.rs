//! Migration service — moves files from a source tree into a user-defined target tree.
//!
//! Same handle drives CLI (`tidyup migrate`) and UI ("Migrate" button).

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::frontend::Level;
use tidyup_core::{ProgressReporter, Result, ReviewHandler, ReviewOutcome};
use tidyup_domain::{
    CapabilityManifest, FileProcessingRole, FileProcessingStage, RunMode, RunRecord, RunState,
};
use tidyup_pipeline::migration::{run_migration, MigrationMultimodal};
use tidyup_pipeline::profiler::{self, MultimodalProfilers};
use uuid::Uuid;

use crate::executor::{
    apply_bundles, apply_loose_decisions, select_bundle_decisions, validate_destination_ledger,
    ApplyReport, ExecutorDeps, DEFAULT_BUNDLE_MIN_CONFIDENCE,
};
use crate::processing::{
    attach_indexed_identities, record_source_outcomes, record_target_profiled, report_indexing,
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
    /// When true (the `--yes` path), auto-apply eligible opaque structural
    /// bundles whose confidence clears `bundle_min_confidence` without
    /// prompting. Soft/file-set and generic bundles remain review-only. When
    /// false, bundles are surfaced to the `ReviewHandler` for interactive
    /// per-bundle approval. Set by the CLI only if `--yes`.
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
pub struct MigrationReport {
    pub source_indexed: usize,
    pub target_indexed: usize,
    pub indexing_failed: usize,
    pub proposed: usize,
    pub bundles: usize,
    pub unclassified: usize,
    /// Images the per-directory clustering work cap excluded from
    /// visual-collection discovery. They are still classified individually;
    /// what they lost was the chance to be grouped.
    pub visual_candidates_over_cap: usize,
    pub approved: usize,
    pub applied: usize,
    pub skipped: usize,
    pub failed: usize,
    pub bundles_applied: usize,
    pub bundles_skipped: usize,
    pub bundles_failed: usize,
    pub run_id: Uuid,
    pub capabilities: CapabilityManifest,
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
    /// 4. Calls `review.review_all(proposals, bundles)` for an interactive
    ///    complete plan, or the restricted threshold paths under `--yes`.
    /// 5. Shelves, journals, and moves approved proposals atomically
    ///    (per-file or per-bundle).
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
        // Overlap guard: a source nested under the target (or vice versa) would
        // move files into their own subtree — duplicating or endlessly
        // re-routing them. Refuse before recording a run.
        guard_source_target_overlap(&request.source, &request.target)?;

        let run = RunRecord::begin(
            RunMode::Migrate,
            request.source.clone(),
            Some(request.target.clone()),
        )
        .with_capabilities(self.ctx.capabilities.clone());
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
            Err(error) => {
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
        result
    }

    #[allow(clippy::too_many_lines)]
    async fn try_run(
        &self,
        request: &MigrationRequest,
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
        run_id: Uuid,
    ) -> Result<MigrationReport> {
        progress
            .message(
                Level::Info,
                &format!("capabilities: {}", self.ctx.capabilities.summary()),
            )
            .await;
        // Persist identity for both sides of a migration before semantic
        // profiling/classification. This makes unchanged files reusable and
        // ensures failures remain visible in the durable index.
        let target_indexed = tidyup_pipeline::indexing::index_directory(
            &request.target,
            self.ctx.file_index.as_ref(),
            run_id,
            FileProcessingRole::TargetProfile,
        )
        .await?;
        report_indexing(&target_indexed, progress).await;
        progress
            .phase_started(tidyup_domain::Phase::ProfilingTarget, None)
            .await;
        let target_scan = profiler::scan_target(&request.target)?;
        // Profile-building signals beyond the always-present name embedding:
        // - the extractors populate the text `content_centroid` from each target
        //   folder's documents (so migration classifies by contents, not just
        //   folder names);
        // - the optional cross-modal backends populate image/audio centroids
        //   when their bundles are loaded.
        // Source files then route against the matching-space centroid; missing
        // centroids fall back to the name/text path.
        let hierarchy_labels = target_scan
            .leaf_folders
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .map(str::to_string)
            .collect::<Vec<_>>();
        let image_concepts = if let Some(backend) = self.ctx.image_embeddings.as_deref() {
            match tidyup_pipeline::semantic::prepare_visual_concepts_with_labels(
                backend,
                &hierarchy_labels,
            )
            .await
            {
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
        let multimodal = MigrationMultimodal {
            image: self.ctx.image_embeddings.as_deref(),
            audio: self.ctx.audio_embeddings.as_deref(),
            image_concepts: &image_concepts,
            artifact_store: Some(self.ctx.file_index.as_ref()),
        };
        let profilers = MultimodalProfilers {
            image: multimodal.image,
            audio: multimodal.audio,
            extractors: &self.ctx.extractors,
            artifact_store: Some(self.ctx.file_index.as_ref()),
        };
        let profile_cache = profiler::build_profile_cache_multimodal(
            &target_scan,
            self.ctx.embeddings.as_ref(),
            profilers,
        )
        .await?;
        record_target_profiled(
            self.ctx.file_index.as_ref(),
            run_id,
            &target_indexed.indexed,
        )
        .await?;
        progress
            .phase_finished(tidyup_domain::Phase::ProfilingTarget)
            .await;

        let source_indexed = tidyup_pipeline::indexing::index_directory(
            &request.source,
            self.ctx.file_index.as_ref(),
            run_id,
            FileProcessingRole::Source,
        )
        .await?;
        report_indexing(&source_indexed, progress).await;
        progress
            .message(
                Level::Info,
                &format!(
                    "indexed {} target file(s) and {} source file(s)",
                    target_indexed.indexed.len(),
                    source_indexed.indexed.len()
                ),
            )
            .await;

        let text_backend = self.ctx.text.as_deref();
        // Classifier config (rename thresholds + optional reranker) is
        // materialised from the loaded TidyupConfig at context-build time.
        // Reuse the identities indexing just streamed: the proposal keeps the
        // FileId and the BLAKE3 already computed instead of re-hashing.
        let identities = tidyup_pipeline::indexing::SourceIdentities::new(&source_indexed.indexed);
        let mut outcome = run_migration(
            &request.source,
            &identities,
            &profile_cache,
            self.ctx.embeddings.as_ref(),
            text_backend,
            multimodal,
            &self.ctx.extractors,
            &self.ctx.classifier,
            progress,
        )
        .await?;

        attach_indexed_identities(
            &source_indexed.indexed,
            &mut outcome.proposals,
            &mut outcome.bundles,
        );
        record_source_outcomes(
            self.ctx.file_index.as_ref(),
            run_id,
            &source_indexed.indexed,
            &outcome.proposals,
            &outcome.bundles,
            &outcome.unclassified,
            // Migration moves every source file into the target tree, so no
            // file can resolve to the location it already occupies.
            &[],
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

        Ok(MigrationReport {
            source_indexed: source_indexed.indexed.len(),
            target_indexed: target_indexed.indexed.len(),
            indexing_failed: source_indexed.failed.saturating_add(target_indexed.failed),
            proposed: outcome.proposals.len(),
            bundles: outcome.bundles.len(),
            unclassified: outcome.unclassified.len(),
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

/// Reject a migration whose source and target trees overlap. Moving files from
/// a source nested inside the target (or a target nested inside the source)
/// would relocate files into their own subtree — duplicating them or feeding
/// them back into the classifier on a later run. Equal roots are also refused.
///
/// Paths are compared after best-effort canonicalisation so `.`/`..`/symlink
/// spellings of the same directory are caught; when a path can't be
/// canonicalised (e.g. doesn't exist yet) the lexical form is used.
fn guard_source_target_overlap(source: &Path, target: &Path) -> Result<()> {
    let s = canonical_or_lexical(source);
    let t = canonical_or_lexical(target);
    if s == t {
        return Err(anyhow::anyhow!(
            "source and target are the same directory ({}); nothing to migrate into",
            source.display(),
        ));
    }
    if t.starts_with(&s) {
        return Err(anyhow::anyhow!(
            "target {} is inside source {}; migrating would move files into their own subtree",
            target.display(),
            source.display(),
        ));
    }
    if s.starts_with(&t) {
        return Err(anyhow::anyhow!(
            "source {} is inside target {}; migrating would move files within the target tree",
            source.display(),
            target.display(),
        ));
    }
    Ok(())
}

fn canonical_or_lexical(p: &Path) -> std::path::PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}
