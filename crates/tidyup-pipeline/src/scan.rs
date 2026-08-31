//! Scan-mode pipeline — classify a source tree against a fixed taxonomy.
//!
//! The scan pipeline uses semantic routing with a *taxonomy* as target:
//! each file is compared against every candidate category. Contrast with the migration pipeline in
//! [`crate::migration`], which classifies against an arbitrary existing
//! folder hierarchy.
//!
//! # Routing flow
//!
//! 1. **Bundle detection** (in the scanner) carves out opaque subtrees; scan
//!    mode preserves their internal paths while routing their aggregate
//!    semantic evidence.
//! 2. **Semantic embeddings**: caller-supplied [`ScanCandidate`]s provide
//!    pre-computed description embeddings; the pipeline embeds the file
//!    name/local context and extracted content as independent evidence and
//!    picks the highest-cosine candidate. Review is required below the
//!    `embedding_threshold` (default 0.35) with `ambiguity_gap`.
//! 3. **LLM fallback** (optional): when a [`TextBackend`] is provided
//!    and deterministic embedding routing lands in the review zone (below threshold or inside the
//!    ambiguity gap), the LLM classifies the content and the resulting
//!    `summary + category + tags` is re-embedded and re-ranked against the
//!    same candidate list. If the re-ranked top scores higher than the original,
//!    we adopt it with legacy [`Tier::Llm`] provenance. The activation gate is the
//!    caller passing `Some(text_backend)` — this module is feature-flag-free
//!    by design.
//! 4. **Uncertain results**: surface to review via `needs_review = true`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tidyup_core::extractor::ContentExtractor;
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::inference::{
    AudioEmbeddingBackend, EmbeddingBackend, FileModality, ImageEmbeddingBackend, TextBackend,
};
use tidyup_domain::change::{ChangeProposal, ChangeStatus, ChangeType};
use tidyup_domain::{BundleProposal, ClassifierConfig, Phase};

use crate::naming::{uniquify_bundle_member_names, RenameProposal};
use crate::scanner::{self, DetectedBundle};
use crate::semantic::{gate_grounded_rename, GroundedConcept, SemanticRunCache};
use crate::spine::{
    build_proposal as build_shared_proposal, cosine, file_modality, find_year, gate_rename,
    gate_rename_with_thresholds, normalize_semantic_text,
};
use crate::yake;
use uuid::Uuid;

/// A classification target: one leaf in the scan taxonomy.
///
/// The `embedding` must be L2-normalized and produced by the same backend
/// the caller passes to [`run_scan`]. Callers typically get these from
/// `tidyup_embeddings_ort::taxonomy` — the pipeline keeps the shape
/// dep-free so it stays trait-object-only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanCandidate {
    /// Taxonomy folder path, e.g. `"Finance/Taxes/"`. Trailing slash required.
    pub folder_path: String,
    /// Rich description used to compute [`Self::embedding`] — retained for
    /// proposal reasoning only (not re-embedded).
    pub description: String,
    /// When `true`, the pipeline appends a year subdirectory if one can be
    /// extracted from the filename or content.
    pub temporal: bool,
    /// L2-normalized embedding of [`Self::description`].
    pub embedding: Vec<f32>,
}

/// Optional multimodal embedding backends + per-modality candidate lists.
///
/// When a backend is present, the pipeline routes files of
/// that modality through it instead of (or in addition to) text routing.
/// Each candidate list embeds in the **modality-specific** latent space and
/// is NOT interchangeable with the text-only `ScanCandidate` list passed at
/// the top level — the pipeline keeps them separate so a misconfigured
/// caller can't compute a meaningless cross-space cosine.
///
/// `Default::default()` returns an empty context so text-only callers do not
/// need to construct modality-specific state.
#[derive(Default)]
#[allow(missing_debug_implementations)] // trait objects don't implement Debug
pub struct MultimodalContext<'a> {
    pub image: Option<ImageContext<'a>>,
    pub audio: Option<AudioContext<'a>>,
    /// Optional persistent semantic-artifact store. Cache failures degrade to
    /// fresh inference and never fail classification.
    pub artifact_store: Option<&'a dyn tidyup_core::storage::FileIndex>,
}

/// Image-side classification context: SigLIP-style backend + image taxonomy.
#[allow(missing_debug_implementations)]
pub struct ImageContext<'a> {
    pub backend: &'a dyn ImageEmbeddingBackend,
    pub candidates: &'a [ScanCandidate],
    /// General visible concepts, pre-embedded once in the image backend's text
    /// space. They support grounded naming and collection labels but never map
    /// directly to a destination.
    pub concepts: &'a [GroundedConcept],
}

/// Audio-side classification context: CLAP-style backend + audio taxonomy.
#[allow(missing_debug_implementations)]
pub struct AudioContext<'a> {
    pub backend: &'a dyn AudioEmbeddingBackend,
    pub candidates: &'a [ScanCandidate],
}

/// Output of one scan pass.
#[derive(Debug, Clone)]
pub struct ScanOutcome {
    pub proposals: Vec<ChangeProposal>,
    pub bundles: Vec<BundleProposal>,
    /// Files the semantic router could not classify. Listed here so callers can
    /// surface them rather than silently drop.
    pub unclassified: Vec<PathBuf>,
    /// Files already sitting at the destination the classifier chose.
    ///
    /// Scan reorganizes in place, so a correctly filed file resolves to its own
    /// current path. That is a successful classification, not a move: emitting
    /// it as a proposal would fill review with no-ops and — because the
    /// destination ledger refuses an approved target that already exists —
    /// abort the whole plan over a file that needed nothing done to it.
    pub already_in_place: Vec<PathBuf>,
    /// Images excluded from visual-collection discovery by the per-directory
    /// work cap. See [`crate::clustering::ClusterOutcome`].
    pub visual_candidates_over_cap: usize,
}

/// Drive the scan pipeline end-to-end.
///
/// `output_root` is the destination prefix applied to each taxonomy leaf
/// path — typically the source root itself (to reorganize in-place) or a
/// user-specified directory.
///
/// `multimodal` carries optional image/audio backends and their per-
/// modality candidate lists. Pass [`MultimodalContext::default()`] for the
/// text-only path.
///
/// `text_backend` is the optional LLM reranker. Pass `None` (the default-build
/// case) to use deterministic embeddings only; pass `Some` and low-confidence
/// embedding verdicts get a chance to be re-ranked through an LLM-cleaned query.
/// The pipeline never invokes the backend when the embedding result already cleared its
/// thresholds, so the cost is paid only on hard cases.
///
/// # Errors
/// Propagates source-read and embedding-backend failures. Per-file extraction
/// or classification errors are logged via `progress.message(Level::Warn, …)`
/// and surfaced through [`ScanOutcome::unclassified`].
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub async fn run_scan(
    source_root: &Path,
    identities: &crate::indexing::SourceIdentities,
    output_root: &Path,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    multimodal: &MultimodalContext<'_>,
    text_backend: Option<&dyn TextBackend>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
    progress: &dyn ProgressReporter,
) -> Result<ScanOutcome> {
    let semantic_cache = SemanticRunCache::new(multimodal.artifact_store);
    progress.phase_started(Phase::Clustering, None).await;
    let tree =
        scanner::scan_with_policy(source_root, &config.directory_envelopes, &config.discovery);
    // Content clustering: group loose siblings into photo bursts / music albums
    // / document series. Runs after the structural scanner; these move as
    // file-sets (each member individually, atomically) — see
    // `BundleKind::moves_as_file_set`.
    let clustered = crate::clustering::cluster_loose_semantic(
        &tree.loose_files,
        extractors,
        &crate::clustering::ClusterConfig::default(),
        crate::clustering::SemanticClusterContext {
            image: multimodal.image.as_ref().map(|context| context.backend),
            image_concepts: multimodal
                .image
                .as_ref()
                .map_or(&[], |context| context.concepts),
            cache: &semantic_cache,
            identities,
        },
    )
    .await;
    progress.phase_finished(Phase::Clustering).await;

    let content_bundles = clustered.bundles;
    let loose_files = clustered.loose;
    let mut outcome = ScanOutcome {
        proposals: Vec::new(),
        bundles: Vec::new(),
        unclassified: Vec::new(),
        already_in_place: Vec::new(),
        visual_candidates_over_cap: clustered.visual_candidates_over_cap,
    };

    // Bundles and loose files share one Classifying phase with one running
    // counter. Both bundle loops below classify member files, and they used to
    // run between `phase_finished(Indexing)` and `phase_started(Classifying)`
    // with no progress events at all — the longest unreported stretch of the
    // run, displayed by a frontend as whatever phase had last been announced.
    let bundle_count = tree.bundles.len().saturating_add(content_bundles.len());
    let classify_total =
        u64::try_from(bundle_count.saturating_add(loose_files.len())).unwrap_or(u64::MAX);
    progress
        .phase_started(Phase::Classifying, Some(classify_total))
        .await;
    let mut classified_count: u64 = 0;

    // Structural bundles preserve their internal layout and names.
    for bundle in &tree.bundles {
        match build_bundle_proposal(
            bundle,
            identities,
            output_root,
            candidates,
            embeddings,
            extractors,
            config.directory_envelopes.sample_cap,
            config,
        )
        .await
        {
            Ok(bp) => outcome.bundles.push(bp),
            Err(e) => {
                outcome.unclassified.extend(bundle.members.iter().cloned());
                progress
                    .message(
                        Level::Warn,
                        &format!("bundle proposal failed for {}: {e}", bundle.root.display()),
                    )
                    .await;
            }
        }
        classified_count = classified_count.saturating_add(1);
        progress
            .item_completed(
                Phase::Classifying,
                ProgressItem {
                    label: bundle.root.display().to_string(),
                    current: classified_count,
                    total: Some(classify_total),
                },
            )
            .await;
    }
    // Content clusters classify each loose member so semantic collections can
    // carry evidence-backed screenshot renames while remaining atomic.
    for bundle in &content_bundles {
        match build_content_bundle_proposal(
            bundle,
            identities,
            output_root,
            candidates,
            embeddings,
            multimodal,
            text_backend,
            extractors,
            config,
            &semantic_cache,
        )
        .await
        {
            Ok(bp) => outcome.bundles.push(bp),
            Err(e) => {
                outcome.unclassified.extend(bundle.members.iter().cloned());
                progress
                    .message(
                        Level::Warn,
                        &format!(
                            "content bundle proposal failed for {}: {e}",
                            bundle.root.display()
                        ),
                    )
                    .await;
            }
        }
        classified_count = classified_count.saturating_add(1);
        progress
            .item_completed(
                Phase::Classifying,
                ProgressItem {
                    label: bundle.root.display().to_string(),
                    current: classified_count,
                    total: Some(classify_total),
                },
            )
            .await;
    }

    for path in &loose_files {
        match classify_file(
            path,
            identities,
            candidates,
            embeddings,
            multimodal,
            text_backend,
            extractors,
            config,
            &semantic_cache,
        )
        .await
        {
            Ok(Some(classified)) => {
                let mut proposal = scan_proposal(path, output_root, &classified, identities);
                // Report calibrated confidence (no-op under the default Identity
                // calibration; applies Platt scaling when a fitted set is set).
                proposal.confidence = config.calibration.calibrate(proposal.confidence);
                if proposal.proposed_path == *path {
                    // Scan is in-place housekeeping: the interesting output is
                    // what sits in the wrong place. A file the classifier would
                    // put exactly where it already is needs no operation, and
                    // proposing one would make the destination ledger reject the
                    // run for targeting an existing path.
                    outcome.already_in_place.push(path.clone());
                } else {
                    outcome.proposals.push(proposal);
                }
            }
            Ok(None) => {
                outcome.unclassified.push(path.clone());
            }
            Err(e) => {
                progress
                    .message(
                        Level::Warn,
                        &format!("classify failed for {}: {e}", path.display()),
                    )
                    .await;
                outcome.unclassified.push(path.clone());
            }
        }
        classified_count = classified_count.saturating_add(1);
        progress
            .item_completed(
                Phase::Classifying,
                ProgressItem {
                    label: path.display().to_string(),
                    current: classified_count,
                    total: Some(classify_total),
                },
            )
            .await;
    }

    progress.phase_finished(Phase::Classifying).await;
    Ok(outcome)
}

/// Internal classification verdict for one loose file.
#[derive(Debug, Clone)]
struct ClassifiedFile {
    folder_path: String,
    confidence: f32,
    reasoning: String,
    needs_review: bool,
    year: Option<i32>,
    temporal: bool,
    rename: RenameProposal,
    classification_confidence: Option<f32>,
    rename_mismatch_score: Option<f32>,
}

/// Classify a single loose file using semantic evidence and optional LLM reranking.
///
/// Optional LLM reranking fires only when (a) embedding routing produced a `needs_review` result, (b) a
/// `text_backend` was supplied by the caller, and (c)
/// `config.enable_llm_fallback` is true. Otherwise the embedding result stands.
///
/// Returns `Ok(None)` when there are no destination candidates.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn classify_file(
    path: &Path,
    identities: &crate::indexing::SourceIdentities,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    multimodal: &MultimodalContext<'_>,
    text_backend: Option<&dyn TextBackend>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
    semantic_cache: &SemanticRunCache<'_>,
) -> Result<Option<ClassifiedFile>> {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();

    // Extract early so content can contribute semantic evidence.
    let mime = identities.mime_type_or_detect(path).await;
    let extracted = semantic_cache
        .extract(path, mime.as_deref(), extractors)
        .await;
    let effective_mime = extracted
        .as_ref()
        .map(|e| e.mime.clone())
        .or_else(|| mime.clone());

    // Content-sniffed MIME selects a compatible semantic capability, never a
    // destination. All destination choices below come from model similarity.
    let modality = file_modality(effective_mime.as_deref());
    if matches!(modality, FileModality::Image) {
        if let Some(img_ctx) = multimodal.image.as_ref() {
            if let Some(verdict) = classify_image(
                path,
                img_ctx,
                embeddings,
                effective_mime.as_deref(),
                extracted.as_ref(),
                config,
                semantic_cache,
            )
            .await?
            {
                return Ok(Some(verdict));
            }
        }
    } else if matches!(modality, FileModality::Audio) {
        if let Some(aud_ctx) = multimodal.audio.as_ref() {
            if let Some(verdict) = classify_audio(
                path,
                aud_ctx,
                effective_mime.as_deref(),
                extracted.as_ref(),
                config,
            )
            .await?
            {
                return Ok(Some(verdict));
            }
        }
    }

    // General semantic fallback. Independent name and content evidence prevent
    // a long document body from drowning out a meaningful filename, while no
    // channel maps directly to a folder. Missing content redistributes its
    // configured weight to the remaining name/path/MIME evidence.
    let text = extracted
        .as_ref()
        .and_then(|content| content.text.as_deref())
        .unwrap_or_default();
    let (best_idx, best_score, gap, semantic_query) = semantic_best_match(
        path,
        text,
        effective_mime.as_deref(),
        candidates,
        embeddings,
        &config.weights,
    )
    .await?;
    if best_idx.is_none() {
        return Ok(None);
    }
    let idx = best_idx.unwrap_or(0);
    let mut chosen_idx = idx;
    let mut chosen_score = best_score;
    let mut chosen_gap = gap;
    let mut tier3_used = false;
    let mut tier3_model: Option<String> = None;
    let needs_review = best_score < config.embedding_threshold || gap < config.ambiguity_gap;

    // Optional LLM reranking only fires when deterministic semantic routing is
    // uncertain, the caller wired in a text backend, and configuration permits
    // it. Confident local-model results never pay the fallback cost.
    if needs_review && config.enable_llm_fallback {
        if let Some(backend) = text_backend {
            let llm_evidence = if text.trim().is_empty() {
                semantic_query.as_str()
            } else {
                text
            };
            match rerank_scan_candidates(backend, embeddings, llm_evidence, &filename, candidates)
                .await
            {
                Ok(Some((llm_idx, llm_score, llm_gap, model_id))) if llm_score > chosen_score => {
                    chosen_idx = llm_idx;
                    chosen_score = llm_score;
                    chosen_gap = llm_gap;
                    tier3_used = true;
                    tier3_model = Some(model_id);
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "tier3 llm fallback failed; staying with tier2");
                }
            }
        }
    }

    let candidate = candidates
        .get(chosen_idx)
        .ok_or_else(|| anyhow::anyhow!("selected classification candidate is out of bounds"))?;
    let final_needs_review =
        chosen_score < config.embedding_threshold || chosen_gap < config.ambiguity_gap;

    let content_text = (!text.trim().is_empty()).then_some(text);
    let year = year_from_path_and_text(path, content_text);
    let keywords = content_text
        .map(|body| yake::extract_keywords(body, 8))
        .unwrap_or_default();
    let metadata_json = extracted
        .as_ref()
        .map_or(serde_json::Value::Null, |e| e.metadata.clone());
    // Rename gating uses the deterministic semantic score (`best_score`), not
    // the post-rerank routing score (`chosen_score`). The optional LLM is only
    // consulted for uncertain placement and must not create a rename path.
    let rename = gate_rename(
        path,
        &metadata_json,
        &keywords,
        year,
        best_score,
        embeddings,
        content_text,
        &filename,
        config,
    )
    .await?;

    let reasoning = if tier3_used {
        let model = tier3_model.as_deref().unwrap_or("unknown");
        format!(
            "tier3 llm-rerank: cos={chosen_score:.3} gap={chosen_gap:.3} \
             tier2_cos={best_score:.3} llm={model} embed={}",
            embeddings.model_id(),
        )
    } else {
        format!(
            "tier2 embedding: cos={chosen_score:.3} gap={chosen_gap:.3} model={}",
            embeddings.model_id(),
        )
    };

    Ok(Some(ClassifiedFile {
        folder_path: candidate.folder_path.clone(),
        confidence: chosen_score,
        reasoning,
        needs_review: final_needs_review,
        year,
        temporal: candidate.temporal,
        rename: rename.proposal,
        // The deterministic semantic score that gated the rename, before any
        // optional LLM reranking.
        classification_confidence: Some(best_score),
        rename_mismatch_score: rename.mismatch_score,
    }))
}

async fn semantic_best_match(
    path: &Path,
    body: &str,
    mime: Option<&str>,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    weights: &tidyup_domain::ScoreWeights,
) -> Result<(Option<usize>, f32, f32, String)> {
    if candidates.is_empty() {
        return Ok((None, 0.0, 0.0, String::new()));
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let parent = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let name_evidence = format!(
        "{} parent context {} media type {}",
        normalize_semantic_text(filename),
        normalize_semantic_text(parent),
        mime.unwrap_or("unknown")
    );
    let mut evidence = vec![(name_evidence, weights.name)];
    if !body.trim().is_empty() {
        evidence.push((body.to_string(), weights.centroid));
    }
    let total_weight: f32 = evidence.iter().map(|(_, weight)| *weight).sum();
    let texts: Vec<&str> = evidence.iter().map(|(text, _)| text.as_str()).collect();
    let vectors = embeddings.embed_texts(&texts).await?;
    let mut ranked: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let score = vectors
                .iter()
                .zip(evidence.iter())
                .map(|(vector, (_, weight))| cosine(vector, &candidate.embedding) * weight)
                .sum::<f32>()
                / total_weight.max(f32::EPSILON);
            (index, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let Some((index, score)) = ranked.first().copied() else {
        return Ok((None, 0.0, 0.0, String::new()));
    };
    let gap = ranked.get(1).map_or(score, |(_, second)| score - second);
    let query = evidence
        .iter()
        .map(|(text, _)| text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Ok((Some(index), score, gap, query))
}

/// Optional rerank: ask the LLM to classify the content, then re-embed its summary +
/// category + tags as a richer query and re-rank the candidate list. Returns
/// `(idx, score, gap, model_id)` for the new top, or `None` when the LLM
/// produced no usable output.
///
/// We deliberately ignore the LLM's `suggested_name`: rename proposals use
/// exact evidence or scored local concept retrieval, never LLM-authored text.
async fn rerank_scan_candidates(
    text_backend: &dyn TextBackend,
    embeddings: &dyn EmbeddingBackend,
    text: &str,
    filename: &str,
    candidates: &[ScanCandidate],
) -> Result<Option<(usize, f32, f32, String)>> {
    // Keep fallback prompts comfortably inside small local models' context
    // windows. Extractors may return very long PDFs; truncating on a character
    // boundary preserves deterministic, valid UTF-8 input.
    let bounded_text = crate::text_util::char_prefix(text, 12_000);
    let Some((query, model_id)) =
        crate::spine::tier3_rerank(text_backend, bounded_text, filename).await?
    else {
        return Ok(None);
    };
    let llm_embedding = embeddings.embed_text(&query).await?;
    let (idx, score, gap) = best_match(&llm_embedding, candidates);
    Ok(idx.map(|i| (i, score, gap, model_id)))
}

/// Build a single dense query string from a [`ContentClassification`]. The
// The shared helper builds the LLM query from summary, category, and tags.
/// File size in bytes, or `0` if it can't be stat'd (treated as "small enough";
/// the subsequent read then falls through safely).
async fn file_len(path: &Path) -> u64 {
    tokio::fs::metadata(path).await.map_or(0, |m| m.len())
}

/// Select a compatible semantic capability from the centralized MIME result.
// Unknown/application types use the general text/context path.
/// Image-modality routing — `SigLIP` cross-modal cosine against image taxonomy.
///
/// Returns `Ok(None)` when the image fails to read or the candidate list is
/// empty. The caller falls back to the general text path in that case.
async fn classify_image(
    path: &Path,
    ctx: &ImageContext<'_>,
    text_embeddings: &dyn EmbeddingBackend,
    mime: Option<&str>,
    extracted: Option<&tidyup_core::extractor::ExtractedContent>,
    config: &ClassifierConfig,
    semantic_cache: &SemanticRunCache<'_>,
) -> Result<Option<ClassifiedFile>> {
    if ctx.candidates.is_empty() {
        return Ok(None);
    }
    let mime_str = mime.unwrap_or("application/octet-stream");
    let Ok(Some(embedding)) = semantic_cache
        .image_embedding(path, mime_str, ctx.backend)
        .await
    else {
        return Ok(None);
    };
    let (best_idx, best_score, gap) = best_match(embedding.as_slice(), ctx.candidates);
    let Some(idx) = best_idx else {
        return Ok(None);
    };
    let Some(candidate) = ctx.candidates.get(idx) else {
        return Ok(None);
    };
    let needs_review =
        best_score < config.image_embedding_threshold || gap < config.image_ambiguity_gap;
    let year = year_from_path_and_text(path, extracted.and_then(|e| e.text.as_deref()));
    let mut grounded =
        gate_grounded_rename(path, embedding.as_slice(), ctx.concepts, best_score, config);
    // Screenshots often contain stronger local OCR evidence than the bounded
    // visual concept bank. Keep SigLIP for destination routing, but allow the
    // exact extracted text to author a review-only filename.
    if let Some(extracted) = extracted.filter(|value| {
        value
            .metadata
            .get("ocr_text")
            .and_then(serde_json::Value::as_str)
            .is_some()
    }) {
        if let Some(content_text) = extracted
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
        {
            let keywords = yake::extract_keywords(content_text, 8);
            let filename = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            let ocr = gate_rename_with_thresholds(
                path,
                &extracted.metadata,
                &keywords,
                year,
                best_score,
                text_embeddings,
                Some(content_text),
                filename,
                config.image_embedding_threshold,
                config.rename.min_ocr_mismatch_score,
            )
            .await?;
            if !matches!(ocr.proposal, RenameProposal::Keep) {
                grounded.proposal = ocr.proposal;
                grounded.mismatch_score = ocr.mismatch_score;
            }
        }
    }
    let concept_reasoning = grounded
        .concepts
        .iter()
        .map(|concept| format!("{}:{:.3}", concept.label, concept.score))
        .collect::<Vec<_>>()
        .join(",");
    Ok(Some(ClassifiedFile {
        folder_path: candidate.folder_path.clone(),
        confidence: best_score,
        reasoning: format!(
            "tier2 image: cos={best_score:.3} gap={gap:.3} model={} concepts=[{concept_reasoning}]",
            ctx.backend.model_id(),
        ),
        needs_review: needs_review || !matches!(grounded.proposal, RenameProposal::Keep),
        year,
        temporal: candidate.temporal,
        rename: grounded.proposal,
        classification_confidence: Some(best_score),
        rename_mismatch_score: grounded.mismatch_score,
    }))
}

/// Audio-modality routing — `CLAP` cross-modal cosine against audio taxonomy.
async fn classify_audio(
    path: &Path,
    ctx: &AudioContext<'_>,
    mime: Option<&str>,
    extracted: Option<&tidyup_core::extractor::ExtractedContent>,
    config: &ClassifierConfig,
) -> Result<Option<ClassifiedFile>> {
    if ctx.candidates.is_empty() {
        return Ok(None);
    }
    // Bound the read: oversized audio falls through to other semantic evidence
    // rather than being slurped whole into memory for embedding.
    if file_len(path).await > tidyup_extract::MAX_DOCUMENT_BYTES {
        return Ok(None);
    }
    let Ok(bytes) = tokio::fs::read(path).await else {
        return Ok(None);
    };
    let mime_str = mime.unwrap_or("application/octet-stream");
    let Ok(embedding) = ctx.backend.embed_audio(&bytes, mime_str).await else {
        return Ok(None);
    };
    let (best_idx, best_score, gap) = best_match(&embedding, ctx.candidates);
    let Some(idx) = best_idx else {
        return Ok(None);
    };
    let Some(candidate) = ctx.candidates.get(idx) else {
        return Ok(None);
    };
    let needs_review = best_score < config.embedding_threshold || gap < config.ambiguity_gap;
    let year = year_from_path_and_text(path, extracted.and_then(|e| e.text.as_deref()));
    Ok(Some(ClassifiedFile {
        folder_path: candidate.folder_path.clone(),
        confidence: best_score,
        reasoning: format!(
            "tier2 audio: cos={best_score:.3} gap={gap:.3} model={}",
            ctx.backend.model_id(),
        ),
        needs_review,
        year,
        temporal: candidate.temporal,
        rename: RenameProposal::Keep,
        classification_confidence: Some(best_score),
        rename_mismatch_score: None,
    }))
}

/// Run the rename cascade and apply the two-signal gate from `RenameConfig`.
///
/// Returns `RenameProposal::Keep` (and no mismatch score) when either gate
/// fails, so downstream code can treat "Keep" uniformly without reading the
/// thresholds itself. The raw mismatch score is surfaced when it was
/// computed, so the proposal can log it even on Keep.
fn best_match(vec: &[f32], candidates: &[ScanCandidate]) -> (Option<usize>, f32, f32) {
    if candidates.is_empty() {
        return (None, 0.0, 0.0);
    }
    let mut top = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    let mut top_idx: Option<usize> = None;
    for (i, c) in candidates.iter().enumerate() {
        let s = cosine(vec, &c.embedding);
        if s > top {
            second = top;
            top = s;
            top_idx = Some(i);
        } else if s > second {
            second = s;
        }
    }
    let gap = if second == f32::NEG_INFINITY {
        top
    } else {
        top - second
    };
    (top_idx, top, gap)
}

fn year_from_path_and_text(path: &Path, text: Option<&str>) -> Option<i32> {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if let Some(y) = find_year(filename) {
        return Some(y);
    }
    text.and_then(|t| find_year(crate::text_util::char_prefix(t, 1000)))
}

fn destination_dir(
    output_root: &Path,
    folder_path: &str,
    year: Option<i32>,
    temporal: bool,
) -> PathBuf {
    let mut dest = output_root.to_path_buf();
    // Trim trailing slash for consistent join behavior.
    let trimmed = folder_path.trim_end_matches('/');
    if !trimmed.is_empty() {
        dest.push(trimmed);
    }
    if temporal {
        if let Some(y) = year {
            dest.push(y.to_string());
        }
    }
    dest
}

fn scan_proposal(
    source: &Path,
    output_root: &Path,
    classified: &ClassifiedFile,
    identities: &crate::indexing::SourceIdentities,
) -> ChangeProposal {
    build_shared_proposal(
        source,
        &destination_dir(
            output_root,
            &classified.folder_path,
            classified.year,
            classified.temporal,
        ),
        &classified.rename,
        classified.confidence,
        classified.reasoning.clone(),
        classified.needs_review,
        classified.classification_confidence,
        classified.rename_mismatch_score,
        identities,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn build_bundle_proposal(
    bundle: &DetectedBundle,
    identities: &crate::indexing::SourceIdentities,
    output_root: &Path,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    extractors: &[Arc<dyn ContentExtractor>],
    evidence_sample_cap: usize,
    config: &ClassifierConfig,
) -> Result<BundleProposal> {
    let leaf = bundle
        .root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("bundle");
    let member_names = bundle
        .members
        .iter()
        .take(24)
        .filter_map(|member| member.file_name().and_then(|name| name.to_str()))
        .collect::<Vec<_>>()
        .join(" ");
    let extracted_evidence = crate::envelopes::aggregate_semantic_evidence(
        &bundle.members,
        extractors,
        evidence_sample_cap,
    )
    .await;
    let query = format!(
        "{}; {}; {}",
        normalize_semantic_text(leaf),
        normalize_semantic_text(&member_names),
        normalize_semantic_text(&extracted_evidence),
    );
    let embedding = embeddings.embed_text(&query).await?;
    let (candidate_index, raw_confidence, gap) = best_match(&embedding, candidates);
    let uncertain =
        raw_confidence < config.bundle_embedding_threshold || gap < config.bundle_ambiguity_gap;
    let confidence = if uncertain { 0.0 } else { raw_confidence };
    let target_parent = if uncertain {
        output_root.to_path_buf()
    } else {
        let candidate = candidate_index
            .and_then(|index| candidates.get(index))
            .ok_or_else(|| {
                anyhow::anyhow!("no semantic destination candidate for bundle {leaf}")
            })?;
        output_root.join(candidate.folder_path.trim_end_matches('/'))
    };

    // Directory bundles keep their subtree under a folder named after the root
    // (members relocate relative to the root). File-set clusters (photo bursts,
    // music albums, document series) have no meaningful subtree — they collapse
    // flat into a subfolder named by the cluster (`target_subdir`).
    let flat = bundle.target_subdir.is_some();
    let subfolder = bundle.target_subdir.clone().unwrap_or_else(|| {
        bundle
            .root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("bundle")
            .to_string()
    });
    let bundle_target_root = target_parent.join(&subfolder);

    let mut members = Vec::with_capacity(bundle.members.len());
    for m in &bundle.members {
        let proposed_path = if flat {
            bundle_target_root.join(m.file_name().unwrap_or_else(|| m.as_os_str()))
        } else {
            let rel = m.strip_prefix(&bundle.root).unwrap_or(m);
            bundle_target_root.join(rel)
        };
        let name = m
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        members.push(ChangeProposal {
            id: Uuid::new_v4(),
            file_id: identities.file_id(m),
            change_type: ChangeType::Move,
            original_path: m.clone(),
            proposed_path,
            proposed_name: name,
            confidence,
            reasoning: bundle.reasoning.clone(),
            needs_review: uncertain
                || bundle
                    .envelope
                    .as_ref()
                    .is_some_and(|envelope| envelope.requires_review),
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: None, // stamped by BundleProposal::new
            classification_confidence: None,
            rename_mismatch_score: None,
            content_hash: identities.content_hash(m),
        });
    }

    let proposal = BundleProposal::new(
        bundle.root.clone(),
        bundle.kind.clone(),
        target_parent,
        members,
        confidence,
        if uncertain {
            format!(
                "{}; bundle routing abstained: cos={raw_confidence:.3} gap={gap:.3}",
                bundle.reasoning
            )
        } else {
            format!(
                "{}; semantic bundle routing: cos={confidence:.3} gap={gap:.3}",
                bundle.reasoning
            )
        },
    )?;
    match &bundle.envelope {
        Some(envelope) => proposal.with_envelope(envelope.clone()).map_err(Into::into),
        None => Ok(proposal),
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
async fn build_content_bundle_proposal(
    bundle: &DetectedBundle,
    identities: &crate::indexing::SourceIdentities,
    output_root: &Path,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    multimodal: &MultimodalContext<'_>,
    text_backend: Option<&dyn TextBackend>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
    semantic_cache: &SemanticRunCache<'_>,
) -> Result<BundleProposal> {
    let label = bundle.target_subdir.clone().unwrap_or_else(|| {
        bundle
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("collection")
            .to_string()
    });
    let allow_renames = bundle.kind.allows_member_renames();
    let mut proposals = Vec::with_capacity(bundle.members.len());
    let mut confidence_sum = 0.0_f32;

    for member in &bundle.members {
        let classified = classify_file(
            member,
            identities,
            candidates,
            embeddings,
            multimodal,
            text_backend,
            extractors,
            config,
            semantic_cache,
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("no classification for {}", member.display()))?;
        let mut proposal = scan_proposal(member, output_root, &classified, identities);
        confidence_sum += proposal.confidence;
        if !allow_renames {
            proposal.change_type = ChangeType::Move;
            proposal.proposed_name = member
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            proposal.rename_mismatch_score = None;
        }
        proposal.reasoning = format!("{}; {}", bundle.reasoning, proposal.reasoning);
        proposals.push(proposal);
    }

    // Route the collection from its semantic identity and evidence-backed
    // member names. Generic source names (timestamps, camera counters, etc.)
    // are deliberately omitted once OCR has supplied a descriptive rename;
    // otherwise formatting noise can overwhelm the meaning of a small set.
    let member_evidence = proposals
        .iter()
        .map(|proposal| semantic_filename(&proposal.proposed_name))
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
    let collection_query = format!(
        "cohesive collection named {}; member artifacts: {}",
        normalize_semantic_text(&label),
        member_evidence
    );
    let collection_embedding = embeddings.embed_text(&collection_query).await?;
    let (candidate_index, mut collection_confidence, mut collection_gap) =
        best_match(&collection_embedding, candidates);
    let mut chosen_index = candidate_index;
    let mut routing_tier = "semantic embedding";
    let uncertain = collection_confidence < config.bundle_embedding_threshold
        || collection_gap < config.bundle_ambiguity_gap;
    if uncertain && config.enable_llm_fallback {
        if let Some(backend) = text_backend {
            if let Ok(Some((llm_index, llm_score, llm_gap, _))) =
                rerank_scan_candidates(backend, embeddings, &collection_query, &label, candidates)
                    .await
            {
                if llm_score > collection_confidence {
                    chosen_index = Some(llm_index);
                    collection_confidence = llm_score;
                    collection_gap = llm_gap;
                    routing_tier = "LLM-refined semantic embedding";
                }
            }
        }
    }
    let abstained = collection_confidence < config.bundle_embedding_threshold
        || collection_gap < config.bundle_ambiguity_gap;
    let target_parent = if abstained {
        output_root.to_path_buf()
    } else {
        let taxonomy = chosen_index
            .and_then(|index| candidates.get(index))
            .map(|candidate| candidate.folder_path.as_str())
            .ok_or_else(|| anyhow::anyhow!("no semantic destination for collection {label}"))?;
        output_root.join(taxonomy.trim_end_matches('/'))
    };
    let collection_root = target_parent.join(&label);
    for proposal in &mut proposals {
        proposal.proposed_path = collection_root.join(&proposal.proposed_name);
    }
    uniquify_bundle_member_names(&mut proposals)?;

    let count = u16::try_from(proposals.len()).unwrap_or(u16::MAX);
    let member_confidence = if count == 0 {
        0.0
    } else {
        confidence_sum / f32::from(count)
    };
    let confidence = if abstained {
        0.0
    } else {
        member_confidence.min(collection_confidence)
    };
    Ok(BundleProposal::new(
        bundle.root.clone(),
        bundle.kind.clone(),
        target_parent,
        proposals,
        confidence,
        format!(
            "{}; {routing_tier} collection routing: cos={collection_confidence:.3} gap={collection_gap:.3}{}",
            bundle.reasoning,
            if abstained { "; abstained to source root" } else { "" }
        ),
    )?)
}

fn semantic_filename(filename: &str) -> String {
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(filename);
    normalize_semantic_text(stem)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::fs;
    use tempfile::TempDir;
    use tidyup_core::extractor::ExtractedContent;
    use tidyup_core::frontend::Level;
    use tidyup_domain::BundleKind;

    struct NullProgress;
    #[async_trait]
    impl ProgressReporter for NullProgress {
        async fn phase_started(&self, _p: Phase, _t: Option<u64>) {}
        async fn item_completed(&self, _p: Phase, _i: ProgressItem) {}
        async fn phase_finished(&self, _p: Phase) {}
        async fn message(&self, _l: Level, _m: &str) {}
    }

    /// Captures the progress stream so tests can assert what a frontend would
    /// actually have been able to display.
    #[derive(Default)]
    struct RecordingProgress {
        started: std::sync::Mutex<Vec<(Phase, Option<u64>)>>,
        items: std::sync::Mutex<Vec<(Phase, u64, Option<u64>)>>,
    }

    #[async_trait]
    impl ProgressReporter for RecordingProgress {
        async fn phase_started(&self, p: Phase, t: Option<u64>) {
            self.started.lock().unwrap().push((p, t));
        }
        async fn item_completed(&self, p: Phase, i: ProgressItem) {
            self.items.lock().unwrap().push((p, i.current, i.total));
        }
        async fn phase_finished(&self, _p: Phase) {}
        async fn message(&self, _l: Level, _m: &str) {}
    }

    /// Deterministic embedder: sums byte values modulo 7 buckets.
    struct BucketEmbeddings;
    #[async_trait]
    impl EmbeddingBackend for BucketEmbeddings {
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            let mut v = vec![0.0_f32; 7];
            for (i, b) in text.bytes().enumerate() {
                v[i % 7] += f32::from(b);
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= norm;
            }
            Ok(v)
        }
        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed_text(t).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize {
            7
        }
        fn model_id(&self) -> &'static str {
            "bucket"
        }
    }

    struct FailingEmbeddings;
    #[async_trait]
    impl EmbeddingBackend for FailingEmbeddings {
        async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            anyhow::bail!("synthetic embedding failure")
        }

        async fn embed_texts(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            anyhow::bail!("synthetic embedding failure")
        }

        fn dimensions(&self) -> usize {
            7
        }

        fn model_id(&self) -> &'static str {
            "failing"
        }
    }

    struct ConstantEmbeddings;
    #[async_trait]
    impl EmbeddingBackend for ConstantEmbeddings {
        async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0])
        }

        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0]; texts.len()])
        }

        fn dimensions(&self) -> usize {
            1
        }

        fn model_id(&self) -> &'static str {
            "constant"
        }
    }

    /// Extractor that reads a file as UTF-8.
    struct PlainExtractor;
    #[async_trait]
    impl ContentExtractor for PlainExtractor {
        fn supports(&self, _path: &Path, _mime: Option<&str>) -> bool {
            true
        }
        async fn extract(&self, path: &Path) -> tidyup_core::Result<ExtractedContent> {
            let bytes = tokio::fs::read(path).await?;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            Ok(ExtractedContent {
                text: Some(text),
                mime: "text/plain".to_string(),
                metadata: serde_json::json!({}),
            })
        }
    }

    async fn sample_candidates(eb: &BucketEmbeddings) -> Vec<ScanCandidate> {
        let specs = [
            (
                "Finance/Taxes/",
                "tax return W-2 1099 1040 IRS refund withholding",
                true,
            ),
            (
                "Photos/",
                "jpeg raw photograph camera exif landscape portrait",
                false,
            ),
            (
                "Code/",
                "source code rust python javascript compile function",
                false,
            ),
        ];
        let mut out = Vec::new();
        for (p, d, t) in &specs {
            let emb = eb.embed_text(d).await.unwrap();
            out.push(ScanCandidate {
                folder_path: (*p).to_string(),
                description: (*d).to_string(),
                temporal: *t,
                embedding: emb,
            });
        }
        out
    }

    /// Semantic fixture with no suffix knowledge: dimensions represent topic
    /// meaning only, so tests fail if routing starts depending on file type.
    struct TopicEmbeddings;
    #[async_trait]
    impl EmbeddingBackend for TopicEmbeddings {
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            let lower = text.to_ascii_lowercase();
            Ok(vec![
                if lower.contains("tax") || lower.contains("invoice") {
                    1.0
                } else {
                    0.0
                },
                if lower.contains("vacation") || lower.contains("portrait") {
                    1.0
                } else {
                    0.0
                },
                if lower.contains("software") || lower.contains("project") {
                    1.0
                } else {
                    0.0
                },
            ])
        }

        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut vectors = Vec::with_capacity(texts.len());
            for text in texts {
                vectors.push(self.embed_text(text).await?);
            }
            Ok(vectors)
        }

        fn dimensions(&self) -> usize {
            3
        }

        fn model_id(&self) -> &'static str {
            "topic-fixture"
        }
    }

    async fn topic_candidates(embeddings: &TopicEmbeddings) -> Vec<ScanCandidate> {
        let specs = [
            ("Money/", "tax invoice", false),
            ("Memories/", "vacation portrait", false),
            ("Builds/", "software project", false),
        ];
        let mut candidates = Vec::new();
        for (path, description, temporal) in specs {
            candidates.push(ScanCandidate {
                folder_path: path.to_string(),
                description: description.to_string(),
                temporal,
                embedding: embeddings.embed_text(description).await.unwrap(),
            });
        }
        candidates
    }

    /// Scan sorts in place, so a file already sitting in the folder the
    /// classifier picks needs no operation. Emitting a proposal for it would
    /// both fill review with no-ops and make the destination ledger abort the
    /// run for targeting a path that already exists.
    #[tokio::test]
    async fn file_already_at_its_destination_produces_no_proposal() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("Money")).unwrap();
        // Already filed correctly: routes to `Money/`, and is already there.
        let settled = td.path().join("Money").join("filed.blob");
        fs::write(&settled, b"tax invoice statement").unwrap();
        // Misfiled: routes to `Builds/`, currently sitting under `Money/`.
        let misplaced = td.path().join("Money").join("stray.blob");
        fs::write(&misplaced, b"software project milestone").unwrap();

        let embeddings = TopicEmbeddings;
        let candidates = topic_candidates(&embeddings).await;
        let extractors: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &embeddings,
            &MultimodalContext::default(),
            None,
            &extractors,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.already_in_place, vec![settled]);
        assert_eq!(out.proposals.len(), 1, "only the misplaced file moves");
        assert_eq!(out.proposals[0].original_path, misplaced);
        assert!(out.proposals[0]
            .proposed_path
            .starts_with(td.path().join("Builds")));
    }

    /// Bundle classification used to run between `phase_finished(Indexing)`
    /// and `phase_started(Classifying)` while emitting nothing at all, so a
    /// frontend displayed a stale phase over the longest stretch of the run and
    /// an ordinary scan looked like it had hung. Every classified unit —
    /// bundles included — must now be covered by one continuous counter.
    #[tokio::test]
    async fn every_classified_unit_reports_progress() {
        let td = TempDir::new().unwrap();
        // A marker-root structural bundle plus loose siblings.
        fs::create_dir_all(td.path().join("proj/src")).unwrap();
        fs::write(td.path().join("proj/Cargo.toml"), b"[package]").unwrap();
        fs::write(td.path().join("proj/src/main.rs"), b"fn main() {}").unwrap();
        fs::write(td.path().join("tax invoice.blob"), b"tax invoice statement").unwrap();
        fs::write(
            td.path().join("build notes.blob"),
            b"software project milestone",
        )
        .unwrap();

        let embeddings = TopicEmbeddings;
        let candidates = topic_candidates(&embeddings).await;
        let extractors: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let progress = RecordingProgress::default();
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &embeddings,
            &MultimodalContext::default(),
            None,
            &extractors,
            &ClassifierConfig::default(),
            &progress,
        )
        .await
        .unwrap();

        let started = progress.started.lock().unwrap().clone();
        // Discovery/clustering must not masquerade as identity indexing.
        assert!(
            started.iter().any(|(p, _)| *p == Phase::Clustering),
            "discovery + clustering needs its own phase, got {started:?}"
        );
        assert!(
            !started.iter().any(|(p, _)| *p == Phase::Indexing),
            "the pipeline no longer reports Indexing; that is the app-layer pass"
        );

        let declared = started
            .iter()
            .find(|(p, _)| *p == Phase::Classifying)
            .and_then(|(_, total)| *total)
            .expect("Classifying must declare a total so a progress bar can render");

        let units = out.bundles.len()
            + out.proposals.len()
            + out.unclassified.len()
            + out.already_in_place.len();
        assert_eq!(
            declared,
            u64::try_from(units).unwrap(),
            "the declared total must cover bundles as well as loose files"
        );

        // One event per unit, strictly increasing, ending exactly at the total:
        // no silent stretch where a frontend has nothing to show.
        let items = progress.items.lock().unwrap().clone();
        let ticks: Vec<u64> = items
            .iter()
            .filter(|(p, _, _)| *p == Phase::Classifying)
            .map(|(_, current, _)| *current)
            .collect();
        assert_eq!(
            ticks,
            (1..=declared).collect::<Vec<_>>(),
            "progress must advance once per unit with no gaps"
        );
        assert!(
            items.iter().all(|(_, _, total)| *total == Some(declared)),
            "every tick must carry the same denominator"
        );
    }

    #[tokio::test]
    async fn identical_suffixes_route_by_semantics_not_type() {
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("first.blob"), b"tax invoice statement").unwrap();
        fs::write(td.path().join("second.blob"), b"software project milestone").unwrap();

        let embeddings = TopicEmbeddings;
        let candidates = topic_candidates(&embeddings).await;
        let extractors: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &embeddings,
            &MultimodalContext::default(),
            None,
            &extractors,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 2);
        let first = out
            .proposals
            .iter()
            .find(|proposal| proposal.original_path.ends_with("first.blob"))
            .unwrap();
        let second = out
            .proposals
            .iter()
            .find(|proposal| proposal.original_path.ends_with("second.blob"))
            .unwrap();
        assert!(first.proposed_path.starts_with(td.path().join("Money")));
        assert!(second.proposed_path.starts_with(td.path().join("Builds")));
    }

    #[tokio::test]
    async fn different_suffixes_converge_on_shared_semantics() {
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("alpha.odd"), b"vacation portrait").unwrap();
        fs::write(td.path().join("beta.weird"), b"vacation portrait").unwrap();

        let embeddings = TopicEmbeddings;
        let candidates = topic_candidates(&embeddings).await;
        let extractors: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &embeddings,
            &MultimodalContext::default(),
            None,
            &extractors,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 2);
        assert!(out.proposals.iter().all(|proposal| proposal
            .proposed_path
            .starts_with(td.path().join("Memories"))));
    }

    #[tokio::test]
    async fn semantic_ranking_resolves_source_without_extension_routing() {
        let td = TempDir::new().unwrap();
        // Loose file, not bundled (no Cargo.toml at root).
        fs::write(td.path().join("helpers.rs"), b"fn main() {}").unwrap();

        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            None,
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(p.proposed_path.to_string_lossy().contains("Code"));
        assert!(p.reasoning.contains("tier2 embedding"));
    }

    #[tokio::test]
    async fn bundle_is_emitted_for_cargo_project() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("myproj/src")).unwrap();
        fs::write(
            td.path().join("myproj/Cargo.toml"),
            b"[package]\nname='x'\n",
        )
        .unwrap();
        fs::write(td.path().join("myproj/src/main.rs"), b"fn main() {}").unwrap();

        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            None,
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.bundles.len(), 1);
        let bundle = &out.bundles[0];
        assert_eq!(bundle.kind, BundleKind::DirectoryEnvelope);
        assert!(bundle
            .envelope
            .as_ref()
            .is_some_and(|metadata| !metadata.provenance.is_empty()));
        assert_eq!(bundle.members.len(), 2);
        assert!(bundle.members.iter().any(|member| {
            member.original_path.ends_with("myproj/src/main.rs")
                && member.proposed_path.ends_with("myproj/src/main.rs")
                && member.change_type == ChangeType::Move
        }));
        assert!(bundle.members.iter().any(|member| {
            member.original_path.ends_with("myproj/Cargo.toml")
                && member.proposed_path.ends_with("myproj/Cargo.toml")
                && member.change_type == ChangeType::Move
        }));
        // Bundles bypass per-file classification.
        assert!(out.proposals.is_empty());
    }

    #[tokio::test]
    async fn failed_bundle_proposals_return_every_member_as_unclassified() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("myproj/src")).unwrap();
        fs::write(
            td.path().join("myproj/Cargo.toml"),
            b"[package]\nname='x'\n",
        )
        .unwrap();
        fs::write(td.path().join("myproj/src/main.rs"), b"fn main() {}").unwrap();
        fs::write(
            td.path().join("atomsnotelectrons_submission.txt"),
            b"submission",
        )
        .unwrap();
        fs::write(
            td.path().join("atomsnotelectrons_testbench.png"),
            b"testbench",
        )
        .unwrap();
        let extractors: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let candidates = sample_candidates(&BucketEmbeddings).await;

        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &FailingEmbeddings,
            &MultimodalContext::default(),
            None,
            &extractors,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();

        assert!(out.bundles.is_empty());
        assert_eq!(out.unclassified.len(), 4);
    }

    #[tokio::test]
    async fn temporal_category_appends_year_subdir() {
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("2024_tax_return.pdf"), b"form 1040 taxes").unwrap();

        // Use synthetic tax candidate that matches the file content by bucket.
        // Destination selection is semantic; the suffix only selects extraction.
        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            None,
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 1);
        // If it landed in Finance/Taxes (temporal), the year directory should appear.
        let p = &out.proposals[0];
        let ds = p.proposed_path.to_string_lossy();
        if p.proposed_path.starts_with(td.path().join("Finance/Taxes")) {
            assert!(ds.contains("2024"), "expected year subdir in {ds}");
        }
    }

    #[tokio::test]
    async fn no_classification_goes_to_unclassified() {
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("weird.xyzqw"), b"").unwrap();
        let eb = BucketEmbeddings;
        // Empty candidate list → Tier 2 can't fire, and extension is unknown.
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &[],
            &eb,
            &MultimodalContext::default(),
            None,
            &[],
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 0);
        assert_eq!(out.unclassified.len(), 1);
    }

    #[tokio::test]
    async fn ocr_rename_must_clear_classification_confidence_gate() {
        let metadata = serde_json::json!({
            "ocr_text": "README.MD\nTESTBENCH\nLEADERBOARD"
        });
        let gated = gate_rename(
            Path::new("/source/Screenshot.png"),
            &metadata,
            &[],
            None,
            0.31,
            &ConstantEmbeddings,
            Some("README.MD TESTBENCH LEADERBOARD"),
            "Screenshot.png",
            &ClassifierConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(gated.proposal, RenameProposal::Keep);
    }

    #[tokio::test]
    async fn below_threshold_is_flagged_for_review() {
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("mystery.dat"), b"opaque binary content").unwrap();
        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        // Force thresholds very high to ensure review.
        let cfg = ClassifierConfig {
            embedding_threshold: 0.99,
            ambiguity_gap: 0.50,
            ..ClassifierConfig::default()
        };
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            None,
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 1);
        assert!(out.proposals[0].needs_review);
    }

    /// Toy SigLIP-style backend: bucket-based image embedding + same-sized
    /// text embedding, so cosine matches across modalities.
    struct BucketImageBackend;
    #[async_trait]
    impl ImageEmbeddingBackend for BucketImageBackend {
        async fn embed_image(&self, bytes: &[u8], _mime: &str) -> Result<Vec<f32>> {
            let mut v = vec![0.0_f32; 7];
            for (i, b) in bytes.iter().enumerate() {
                v[i % 7] += f32::from(*b);
            }
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= n;
            }
            Ok(v)
        }
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            let mut v = vec![0.0_f32; 7];
            for (i, b) in text.bytes().enumerate() {
                v[i % 7] += f32::from(b);
            }
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= n;
            }
            Ok(v)
        }
        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed_text(t).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize {
            7
        }
        fn model_id(&self) -> &'static str {
            "bucket-image"
        }
    }

    struct GroundedImageBackend;

    #[async_trait]
    impl ImageEmbeddingBackend for GroundedImageBackend {
        async fn embed_image(&self, _bytes: &[u8], _mime: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            if text.contains("cat") {
                Ok(vec![1.0, 0.0])
            } else if text.contains("beach") {
                Ok(vec![0.8, 0.2])
            } else {
                Ok(vec![0.0, 1.0])
            }
        }

        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut vectors = Vec::with_capacity(texts.len());
            for text in texts {
                vectors.push(self.embed_text(text).await?);
            }
            Ok(vectors)
        }

        fn dimensions(&self) -> usize {
            2
        }

        fn model_id(&self) -> &'static str {
            "grounded-image"
        }
    }

    #[tokio::test]
    async fn non_textual_image_receives_grounded_semantic_rename() {
        let temp = TempDir::new().unwrap();
        let image_path = temp.path().join("image_2348985fg.png");
        image::RgbImage::new(4, 4).save(&image_path).unwrap();
        let backend = GroundedImageBackend;
        let candidates = vec![ScanCandidate {
            folder_path: "Photos/Pets/".to_string(),
            description: "a photograph of a cat at a beach".to_string(),
            temporal: true,
            embedding: vec![1.0, 0.0],
        }];
        let concepts = vec![
            GroundedConcept {
                label: "cat".to_string(),
                prompt: "a photograph of a cat".to_string(),
                family: "animal".to_string(),
                embedding: vec![1.0, 0.0],
            },
            GroundedConcept {
                label: "beach".to_string(),
                prompt: "a photograph of a beach".to_string(),
                family: "scene".to_string(),
                embedding: vec![0.8, 0.2],
            },
            GroundedConcept {
                label: "office".to_string(),
                prompt: "a photograph of an office".to_string(),
                family: "place".to_string(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let multimodal = MultimodalContext {
            image: Some(ImageContext {
                backend: &backend,
                candidates: &candidates,
                concepts: &concepts,
            }),
            audio: None,
            artifact_store: None,
        };
        let config = ClassifierConfig {
            embedding_threshold: 0.30,
            ambiguity_gap: 0.01,
            ..ClassifierConfig::default()
        };
        let outcome = run_scan(
            temp.path(),
            &crate::indexing::SourceIdentities::default(),
            temp.path(),
            &[],
            &BucketEmbeddings,
            &multimodal,
            None,
            &[],
            &config,
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(outcome.proposals.len(), 1);
        let proposal = &outcome.proposals[0];
        assert_eq!(proposal.change_type, ChangeType::RenameAndMove);
        assert_eq!(proposal.proposed_name, "cat_on_beach.png");
        assert!(
            proposal.needs_review,
            "semantic renames always require review"
        );
        assert!(proposal.reasoning.contains("cat:1.000"));
    }

    #[tokio::test]
    async fn image_modality_routes_through_image_backend_when_present() {
        let td = TempDir::new().unwrap();
        // Generate a tiny PNG so MIME detection identifies it as image/.
        let img_path = td.path().join("snapshot.png");
        let img = image::RgbImage::new(4, 4);
        img.save(&img_path).unwrap();

        let img_be = BucketImageBackend;
        let candidates = vec![ScanCandidate {
            folder_path: "Photos/".to_string(),
            description: "a photograph".to_string(),
            temporal: true,
            embedding: img_be.embed_text("a photograph").await.unwrap(),
        }];
        let img_ctx = ImageContext {
            backend: &img_be,
            candidates: &candidates,
            concepts: &[],
        };
        let multimodal = MultimodalContext {
            image: Some(img_ctx),
            audio: None,
            artifact_store: None,
        };

        let eb = BucketEmbeddings;
        // Empty text candidates ensure any proposal came from the image path.
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };
        // No extractors: the test PlainExtractor would falsely advertise
        // text/plain MIME and short-circuit modality routing. With no
        // extractor, the `mime` from `tidyup_extract::mime::detect` (which
        // sniffs PNG headers correctly) drives modality detection.
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &[],
            &eb,
            &multimodal,
            None,
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(
            p.reasoning.contains("tier2 image"),
            "expected image tier reasoning, got {}",
            p.reasoning,
        );
        assert!(p.proposed_path.to_string_lossy().contains("Photos"));
    }

    #[tokio::test]
    async fn image_reaches_tier2_at_default_threshold_when_backend_present() {
        // Regression signal: a recognized image reaches the image embedding
        // backend without any extension-to-destination short circuit.
        let td = TempDir::new().unwrap();
        let img_path = td.path().join("snapshot.png");
        image::RgbImage::new(4, 4).save(&img_path).unwrap();

        let img_be = BucketImageBackend;
        let candidates = vec![ScanCandidate {
            folder_path: "Photos/".to_string(),
            description: "a photograph".to_string(),
            temporal: true,
            embedding: img_be.embed_text("a photograph").await.unwrap(),
        }];
        let img_ctx = ImageContext {
            backend: &img_be,
            candidates: &candidates,
            concepts: &[],
        };
        let multimodal = MultimodalContext {
            image: Some(img_ctx),
            audio: None,
            artifact_store: None,
        };

        let eb = BucketEmbeddings;
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![];
        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &[],
            &eb,
            &multimodal,
            None,
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 1);
        assert!(
            out.proposals[0].reasoning.contains("tier2 image"),
            "the image must reach the semantic image backend; got {}",
            out.proposals[0].reasoning,
        );
    }

    #[test]
    fn unknown_mime_uses_general_semantic_path() {
        let m = file_modality(None);
        assert!(matches!(m, FileModality::Text));
    }

    #[test]
    fn file_modality_classifies_audio_mime() {
        let m = file_modality(Some("audio/mpeg"));
        assert!(matches!(m, FileModality::Audio));
    }

    #[test]
    fn destination_dir_temporal_year_append() {
        let root = PathBuf::from("/out");
        let d = destination_dir(&root, "Finance/Taxes/", Some(2024), true);
        assert_eq!(d, PathBuf::from("/out/Finance/Taxes/2024"));
    }

    #[test]
    fn destination_dir_non_temporal_ignores_year() {
        let root = PathBuf::from("/out");
        let d = destination_dir(&root, "Photos/", Some(2024), false);
        assert_eq!(d, PathBuf::from("/out/Photos"));
    }

    #[test]
    fn year_extraction_does_not_panic_on_multibyte_text() {
        // The 1000-byte content truncation must land on a char boundary. A
        // string of 3-byte codepoints puts byte 1000 mid-character, which a
        // naive `&text[..1000]` would panic on. Also exercise a non-ASCII path.
        let filler = "字".repeat(400); // 1200 bytes; byte 1000 lands mid-codepoint
                                       // Year is within the first 1000 bytes; the multibyte filler pushes the
                                       // truncation boundary into the middle of a codepoint.
        let text = format!("tax year 2021 {filler}");
        let path = PathBuf::from("/inbox/écritureΩ/문서.txt");
        // Must not panic; the in-window year is found despite the multibyte body.
        assert_eq!(year_from_path_and_text(&path, Some(&text)), Some(2021));
        // A purely multibyte body with no year returns None, still no panic.
        assert_eq!(year_from_path_and_text(&path, Some(&filler)), None);
    }

    #[test]
    fn find_year_edge_cases() {
        assert_eq!(find_year("2024.pdf"), Some(2024));
        assert_eq!(find_year("report-2039-final.pdf"), Some(2039));
        assert_eq!(find_year("file20249.pdf"), None);
        assert_eq!(find_year("file12024.pdf"), None);
        assert_eq!(find_year("1999"), None);
        assert_eq!(find_year("2040"), None);
    }

    /// Stub `TextBackend` returning a fixed `ContentClassification`. Used to
    /// drive the Tier 3 path deterministically without loading a real LLM.
    struct StubTextBackend {
        category: &'static str,
        tags: Vec<&'static str>,
        summary: &'static str,
    }
    #[async_trait]
    impl TextBackend for StubTextBackend {
        async fn classify_text(
            &self,
            _text: &str,
            _filename: &str,
        ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
            Ok(tidyup_core::inference::ContentClassification {
                category: self.category.to_string(),
                tags: self.tags.iter().map(|s| (*s).to_string()).collect(),
                summary: self.summary.to_string(),
                suggested_name: None,
            })
        }
        async fn classify_audio(
            &self,
            _filename: &str,
            _metadata: &str,
        ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
            panic!("audio path not exercised by these tests")
        }
        async fn classify_video(
            &self,
            _filename: &str,
            _frame_captions: &[String],
        ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
            panic!("video path not exercised by these tests")
        }
        async fn classify_image_description(
            &self,
            _filename: &str,
            _description: &str,
        ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
            panic!("image-desc path not exercised by these tests")
        }
        async fn complete(
            &self,
            _prompt: &str,
            _opts: &tidyup_core::inference::GenerationOptions,
        ) -> tidyup_core::Result<String> {
            panic!("complete not exercised by these tests")
        }
        fn model_id(&self) -> &'static str {
            "stub-llm"
        }
    }

    #[tokio::test]
    async fn tier3_llm_rerank_replaces_uncertain_tier2_verdict() {
        // Setup: a file whose raw bytes embed poorly against any candidate,
        // forcing Tier 2 below threshold. The stub LLM's summary embeds well
        // against the Finance/Taxes candidate, so Tier 3 should override.
        let td = TempDir::new().unwrap();
        // Content chosen so its bucket-embedding cosine with the Finance
        // description is mediocre — it's mostly punctuation/short words that
        // don't share many bytes with the tax keyword soup.
        fs::write(td.path().join("anonymous.dat"), b"x x x x x x").unwrap();

        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];

        // High thresholds force Tier 2 to mark needs_review even on its top hit.
        let cfg = ClassifierConfig {
            embedding_threshold: 0.99,
            ambiguity_gap: 0.50,
            enable_llm_fallback: true,
            ..ClassifierConfig::default()
        };

        // category + tags are prepended to the LLM query in production; here
        // we leave them empty so the query is exactly the Finance candidate's
        // description. This isolates the test from the toy bucket embedder's
        // byte-collision behaviour and keeps the assertion about *wiring*,
        // not embedder fidelity.
        let llm = StubTextBackend {
            category: "",
            tags: vec![],
            summary: "tax return W-2 1099 1040 IRS refund withholding",
        };

        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            Some(&llm),
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(
            p.reasoning.contains("tier3 llm-rerank"),
            "expected tier3 reasoning, got {}",
            p.reasoning,
        );
        assert!(
            p.proposed_path.to_string_lossy().contains("Finance"),
            "expected Finance route, got {:?}",
            p.proposed_path,
        );
    }

    #[tokio::test]
    async fn tier3_reroute_gates_rename_on_tier2_confidence_not_rerank() {
        // Regression: the rename gate must be driven by Tier 2's confidence, not
        // the post-Tier-3 rerank score. Tier 3 only fires when Tier 2 was
        // uncertain and can only raise the routing score — so a rerouted file
        // must never earn a rename. With the pre-fix wiring (feeding the boosted
        // `chosen_score` into the gate and reporting it as
        // `classification_confidence`), the reported gate score equalled the
        // routing confidence; after the fix it stays the (lower) Tier-2 score.
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("anonymous.dat"), b"x x x x x x").unwrap();

        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.99,
            ambiguity_gap: 0.50,
            enable_llm_fallback: true,
            ..ClassifierConfig::default()
        };
        let llm = StubTextBackend {
            category: "",
            tags: vec![],
            summary: "tax return W-2 1099 1040 IRS refund withholding",
        };

        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            Some(&llm),
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        let p = &out.proposals[0];
        assert!(
            p.reasoning.contains("tier3 llm-rerank"),
            "precondition: Tier 3 must have rerouted, got {}",
            p.reasoning,
        );
        // The rename-gate score is Tier 2's, strictly below the boosted routing
        // confidence — this is the crux of the fix.
        let gate_score = p.classification_confidence.unwrap();
        assert!(
            gate_score < p.confidence,
            "rename gate score {gate_score} must be the (lower) Tier-2 score, \
             not the boosted routing confidence {}",
            p.confidence,
        );
        // A Tier-3 reroute never produces a rename (Tier 2 was uncertain, so the
        // gate's classification-confidence threshold can't be met).
        assert_eq!(
            p.change_type,
            ChangeType::Move,
            "Tier-3 reroute must not produce a rename"
        );
    }

    #[tokio::test]
    async fn tier3_disabled_in_config_does_not_call_llm() {
        // If `enable_llm_fallback = false`, the LLM is never consulted even
        // when a backend is supplied. Stub returns a category that *would*
        // re-route — the test asserts it does not.
        struct ExplodingTextBackend;
        #[async_trait]
        impl TextBackend for ExplodingTextBackend {
            async fn classify_text(
                &self,
                _text: &str,
                _filename: &str,
            ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
                panic!("Tier 3 must not be invoked when enable_llm_fallback is false");
            }
            async fn classify_audio(
                &self,
                _f: &str,
                _m: &str,
            ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
                panic!("audio path not exercised by this test")
            }
            async fn classify_video(
                &self,
                _f: &str,
                _c: &[String],
            ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
                panic!("video path not exercised by this test")
            }
            async fn classify_image_description(
                &self,
                _f: &str,
                _d: &str,
            ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
                panic!("image-description path not exercised by this test")
            }
            async fn complete(
                &self,
                _p: &str,
                _o: &tidyup_core::inference::GenerationOptions,
            ) -> tidyup_core::Result<String> {
                panic!("completion path not exercised by this test")
            }
            fn model_id(&self) -> &'static str {
                "exploding"
            }
        }

        let td = TempDir::new().unwrap();
        fs::write(td.path().join("anonymous.dat"), b"x x x x x x").unwrap();

        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.99,
            ambiguity_gap: 0.50,
            enable_llm_fallback: false,
            ..ClassifierConfig::default()
        };

        let out = run_scan(
            td.path(),
            &crate::indexing::SourceIdentities::default(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            Some(&ExplodingTextBackend),
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        assert!(out.proposals[0].needs_review);
        assert!(out.proposals[0].reasoning.contains("tier2"));
    }
}
