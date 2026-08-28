//! Migration-mode pipeline — classify a source tree against an existing
//! destination hierarchy described by a [`ProfileCache`].
//!
//! Contrast with [`crate::scan`]: scan-mode routes into a fixed taxonomy;
//! migration-mode routes into whatever structure the user has already built.
//! Composite semantic scoring combines similarity to each folder's
//! `name_embedding` and `content_centroid`. File
//! type may select a compatible inference backend, but never a destination.
//!
//! # Composite score
//!
//! Per [`ScoreWeights`]. The profiler populates `content_centroid` from each target
//! folder's documents when extractors are supplied; when a folder has no text
//! documents (or extractors weren't supplied) its `content_centroid` is `None`
//! and the centroid weight is redistributed to `name` so the composite stays in
//! `[0, 1]`.
//!
//! # Tier 3
//!
//! When a caller supplies `Some(text_backend)` and Tier 2's top profile lands
//! in the review zone, the LLM classifies the content and the resulting
//! `summary + category + tags` is re-embedded and re-ranked against the same
//! profile cache under the same scoring rules. The LLM-reranked top is
//! adopted only if it scores higher than Tier 2's. The verdict's
//! `resolved_at` is set to [`Tier::Llm`] when this fires. The activation gate
//! is the caller passing `Some(text_backend)` — this module is
//! feature-flag-free by design.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use tidyup_core::extractor::ContentExtractor;
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::inference::{
    AudioEmbeddingBackend, EmbeddingBackend, FileModality, ImageEmbeddingBackend, TextBackend,
};
use tidyup_domain::change::{ChangeProposal, ChangeStatus, ChangeType};
use tidyup_domain::migration::{
    Candidate, ClassificationResult, ScoreBreakdown, ScoreWeights, Tier,
};
use tidyup_domain::{BundleProposal, ClassifierConfig, FolderProfile, Phase, ProfileCache};
use uuid::Uuid;

use crate::naming::{propose_rename, RenameProposal, RenameSource};
use crate::scanner::{self, DetectedBundle};
use crate::text_util::char_prefix;
use crate::yake;

/// Optional cross-modal backends for routing image/audio source files.
///
/// Both `None` on the default install — image and audio files then fall through
/// to the text Tier 2 path (today's behaviour). When present, image source
/// files route against folder image centroids and audio against audio centroids.
///
/// Latent-space isolation: the image backend is only ever used to embed image
/// source files for comparison against [`FolderProfile::image_centroid`], and
/// audio likewise against [`FolderProfile::audio_centroid`]. There is no path
/// that compares one modality's embedding against another's centroid or against
/// the text `name_embedding`.
#[derive(Default, Clone, Copy)]
#[allow(missing_debug_implementations)] // trait objects don't implement Debug
pub struct MigrationMultimodal<'a> {
    pub image: Option<&'a dyn ImageEmbeddingBackend>,
    pub audio: Option<&'a dyn AudioEmbeddingBackend>,
}

/// Output of one migration pass.
#[derive(Debug, Clone)]
pub struct MigrationOutcome {
    pub proposals: Vec<ChangeProposal>,
    pub bundles: Vec<BundleProposal>,
    /// Per-file classification results — kept alongside the proposals so
    /// callers can inspect score breakdowns / runner-ups without re-running
    /// the cascade.
    pub classifications: Vec<ClassificationResult>,
    /// Files the cascade couldn't place (no extractable content, empty
    /// profile cache, or below-threshold with too small an ambiguity gap and
    /// no reliable semantic evidence).
    pub unclassified: Vec<PathBuf>,
}

/// Drive the migration cascade end-to-end.
///
/// `text_backend` is the optional Tier 3 LLM. Pass `None` and the cascade
/// stops at Tier 2; pass `Some` and low-confidence Tier 2 verdicts get a
/// chance to be re-ranked through an LLM-cleaned query against the same
/// folder profiles.
///
/// `multimodal` carries optional image/audio backends. When present, image and
/// audio source files are routed against the folders' cross-modal centroids
/// (`SigLIP` / `CLAP`) instead of the text path; when absent (the default
/// install) those files fall through to the text Tier 2 cascade unchanged.
///
/// # Errors
/// Propagates source-read and embedding-backend failures. Per-file
/// extraction / classification failures are logged via `progress.message`
/// and surface through [`MigrationOutcome::unclassified`].
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub async fn run_migration(
    source_root: &Path,
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
    text_backend: Option<&dyn TextBackend>,
    multimodal: MigrationMultimodal<'_>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
    progress: &dyn ProgressReporter,
) -> Result<MigrationOutcome> {
    progress.phase_started(Phase::Indexing, None).await;
    let tree = scanner::scan(source_root);
    let (content_bundles, loose_files) = crate::clustering::cluster_loose(
        &tree.loose_files,
        extractors,
        &crate::clustering::ClusterConfig::default(),
    )
    .await;
    progress.phase_finished(Phase::Indexing).await;

    let mut outcome = MigrationOutcome {
        proposals: Vec::new(),
        bundles: Vec::new(),
        classifications: Vec::new(),
        unclassified: Vec::new(),
    };

    // Bundles first — they bypass Tier 2 per-file classification but we still
    // place them under the best-matching leaf folder (by leaf name).
    for bundle in &tree.bundles {
        match build_bundle_proposal(bundle, profiles, embeddings).await {
            Ok(bp) => outcome.bundles.push(bp),
            Err(e) => {
                progress
                    .message(
                        Level::Warn,
                        &format!("bundle proposal failed for {}: {e}", bundle.root.display()),
                    )
                    .await;
            }
        }
    }

    // Semantic collections and other loose-file clusters need per-member
    // extraction/classification before their aggregate destination and optional
    // evidence-backed renames can be proposed.
    for bundle in &content_bundles {
        match build_content_bundle_proposal(
            bundle,
            profiles,
            embeddings,
            text_backend,
            multimodal,
            extractors,
            config,
        )
        .await
        {
            Ok(bp) => outcome.bundles.push(bp),
            Err(e) => {
                progress
                    .message(
                        Level::Warn,
                        &format!(
                            "semantic collection proposal failed for {}: {e}",
                            bundle.root.display()
                        ),
                    )
                    .await;
            }
        }
    }

    let total = u64::try_from(loose_files.len()).unwrap_or(u64::MAX);
    progress
        .phase_started(Phase::Classifying, Some(total))
        .await;

    for (idx, path) in loose_files.iter().enumerate() {
        match classify_file(
            path,
            profiles,
            embeddings,
            text_backend,
            multimodal,
            extractors,
            config,
        )
        .await
        {
            Ok(Some(verdict)) => {
                let mut proposal = build_proposal(path, &verdict);
                // Calibrated confidence (no-op under the default Identity).
                proposal.confidence = config.calibration.calibrate(proposal.confidence);
                outcome.proposals.push(proposal);
                outcome.classifications.push(verdict.result);
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
        progress
            .item_completed(
                Phase::Classifying,
                ProgressItem {
                    label: path.display().to_string(),
                    current: u64::try_from(idx).unwrap_or(u64::MAX).saturating_add(1),
                    total: Some(total),
                },
            )
            .await;
    }

    progress.phase_finished(Phase::Classifying).await;
    Ok(outcome)
}

/// Per-file verdict: the full `ClassificationResult` plus the rename proposal
/// and the scoring breakdown needed to build a `ChangeProposal`.
struct Verdict {
    result: ClassificationResult,
    rename: RenameProposal,
    destination_folder: PathBuf,
    confidence: f32,
    reasoning: String,
    classification_confidence: Option<f32>,
    rename_mismatch_score: Option<f32>,
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn classify_file(
    path: &Path,
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
    text_backend: Option<&dyn TextBackend>,
    multimodal: MigrationMultimodal<'_>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
) -> Result<Option<Verdict>> {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();

    let mime = tidyup_extract::mime::detect(path).await;
    let extracted = match tidyup_extract::router::pick(extractors, path, mime.as_deref()) {
        Some(ex) => ex.extract(path).await.ok(),
        None => None,
    };
    let effective_mime = extracted
        .as_ref()
        .map(|e| e.mime.clone())
        .or_else(|| mime.clone());

    // File type selects only a compatible semantic backend. It never maps to a
    // destination folder.
    let modality = file_modality(path, effective_mime.as_deref());
    // Cross-modal image/audio files route against the folders'
    // image/audio centroids when the matching backend is loaded. A miss
    // (backend absent, no folder has a centroid, unreadable/oversized file)
    // falls through to the text semantic path.
    if let Some(verdict) = classify_modality_file(
        path,
        modality,
        effective_mime.as_deref(),
        multimodal,
        profiles,
        config,
    )
    .await
    {
        return Ok(Some(verdict));
    }

    // General semantic scoring against all leaf profiles. Files with no
    // extractable body still carry filename, local path context, and detected
    // media type as model inputs; none of those inputs directly chooses a
    // destination.
    let text = extracted
        .as_ref()
        .and_then(|content| content.text.as_deref())
        .unwrap_or_default();
    let semantic_query = semantic_evidence_query(path, text, effective_mime.as_deref());
    let content_embedding = embeddings.embed_text(&semantic_query).await?;

    let ranked = rank_profiles(&content_embedding, profiles, path, &config.weights);
    if ranked.is_empty() {
        return Ok(None);
    }

    let Some((tier2_folder, tier2_score, tier2_breakdown)) = ranked.first().cloned() else {
        return Ok(None);
    };
    let tier2_gap = ranked
        .get(1)
        .map_or(tier2_score, |(_, score, _)| tier2_score - score);
    let tier2_needs_review =
        tier2_score < config.embedding_threshold || tier2_gap < config.ambiguity_gap;

    // Tier 3 — only fires when Tier 2 was uncertain and a backend is wired.
    let mut chosen_folder = tier2_folder.clone();
    let mut chosen_score = tier2_score;
    let mut chosen_gap = tier2_gap;
    let mut chosen_breakdown = tier2_breakdown.clone();
    let mut chosen_ranked = ranked.clone();
    let mut tier3_used = false;
    let mut tier3_model: Option<String> = None;

    if tier2_needs_review && config.enable_llm_fallback {
        if let Some(backend) = text_backend {
            match tier3_rerank(
                backend,
                embeddings,
                profiles,
                path,
                if text.trim().is_empty() {
                    &semantic_query
                } else {
                    text
                },
                &filename,
                &config.weights,
            )
            .await
            {
                Ok(Some((rerank, model_id))) => {
                    if let Some((f, s, b)) = rerank.first().cloned() {
                        if s > chosen_score {
                            let g = rerank.get(1).map_or(s, |(_, score, _)| s - score);
                            chosen_folder = f;
                            chosen_score = s;
                            chosen_gap = g;
                            chosen_breakdown = b;
                            chosen_ranked = rerank;
                            tier3_used = true;
                            tier3_model = Some(model_id);
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "tier3 llm fallback failed; staying with tier2");
                }
            }
        }
    }

    let needs_review =
        chosen_score < config.embedding_threshold || chosen_gap < config.ambiguity_gap;

    let metadata_json = extracted
        .as_ref()
        .map_or(serde_json::Value::Null, |e| e.metadata.clone());
    let content_text = (!text.trim().is_empty()).then_some(text);
    let keywords = content_text
        .map(|body| yake::extract_keywords(body, 8))
        .unwrap_or_default();
    let year = content_text.and_then(|body| find_year(char_prefix(body, 1000)));
    // Rename gate is driven by Tier 2's confidence (`tier2_score`), NOT the
    // post-Tier-3 rerank (`chosen_score`) — see the scan-mode gate and the
    // "Tier 3 reroutes never produce renames" invariant in CLAUDE.md.
    let rename = gate_rename(
        path,
        &metadata_json,
        &keywords,
        year,
        tier2_score,
        embeddings,
        content_text,
        &filename,
        config,
    )
    .await?;

    let candidates_out: Vec<Candidate> = chosen_ranked
        .iter()
        .take(5)
        .map(|(folder, score, breakdown)| Candidate {
            folder: folder.clone(),
            score: *score,
            score_breakdown: breakdown.clone(),
        })
        .collect();

    let resolved_at = if tier3_used {
        Tier::Llm
    } else {
        Tier::Embedding
    };
    let reasoning = if tier3_used {
        let model = tier3_model.as_deref().unwrap_or("unknown");
        format!(
            "tier3 llm-rerank: top={chosen_score:.3} gap={chosen_gap:.3} \
             tier2_top={tier2_score:.3} llm={model} embed={}",
            embeddings.model_id(),
        )
    } else {
        format!(
            "tier2 composite: top={chosen_score:.3} gap={chosen_gap:.3} name={:.3} centroid={}",
            chosen_breakdown.name_similarity,
            chosen_breakdown
                .centroid_similarity
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}")),
        )
    };

    Ok(Some(Verdict {
        result: ClassificationResult {
            source_file: path.to_path_buf(),
            candidates: candidates_out,
            resolved_at,
            needs_review,
            suggested_rename: match &rename.proposal {
                RenameProposal::Rename { name, .. } => Some(name.clone()),
                RenameProposal::Keep => None,
            },
        },
        rename: rename.proposal,
        destination_folder: chosen_folder,
        confidence: chosen_score,
        reasoning,
        // The Tier-2 sub-score that gated the rename (not the Tier-3 rerank).
        classification_confidence: Some(tier2_score),
        rename_mismatch_score: rename.mismatch_score,
    }))
}

fn semantic_evidence_query(path: &Path, body: &str, mime: Option<&str>) -> String {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let parent = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    format!(
        "filename {} parent context {} media type {} content {}",
        normalize_semantic_text(filename),
        normalize_semantic_text(parent),
        mime.unwrap_or("unknown"),
        body
    )
}

fn normalize_semantic_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect()
}

/// Tier 3 for migration mode: ask the LLM to classify the content, then
/// re-rank profiles using an embedding of `summary + category + tags` instead
/// of the raw content.
///
/// Returns the new ranked list (same shape as [`rank_profiles`]) plus the
/// backend's model id, or `None` when the LLM produced no usable output. We
/// deliberately ignore the LLM's `suggested_name` — rename proposals are
/// extractive only (per project rename policy).
#[allow(clippy::too_many_arguments)]
async fn tier3_rerank(
    text_backend: &dyn TextBackend,
    embeddings: &dyn EmbeddingBackend,
    profiles: &ProfileCache,
    source_path: &Path,
    text: &str,
    filename: &str,
    weights: &ScoreWeights,
) -> Result<Option<(Vec<(PathBuf, f32, ScoreBreakdown)>, String)>> {
    // Small local fallback models have finite context windows; bound long PDF
    // and document evidence without ever splitting a UTF-8 code point.
    let classification = text_backend
        .classify_text(char_prefix(text, 12_000), filename)
        .await?;
    let model_id = text_backend.model_id().to_string();
    let query = build_llm_query(&classification);
    if query.is_empty() {
        return Ok(None);
    }
    let llm_embedding = embeddings.embed_text(&query).await?;
    let ranked = rank_profiles(&llm_embedding, profiles, source_path, weights);
    if ranked.is_empty() {
        return Ok(None);
    }
    Ok(Some((ranked, model_id)))
}

/// Build a single dense query string from a [`ContentClassification`]. Mirrors
/// the scan-mode helper — same idea: combine the LLM's structured output into
/// one string that embeds well against folder name/centroid descriptions.
fn build_llm_query(c: &tidyup_core::inference::ContentClassification) -> String {
    let mut parts = Vec::with_capacity(3);
    if !c.category.is_empty() {
        parts.push(c.category.clone());
    }
    if !c.tags.is_empty() {
        parts.push(c.tags.join(" "));
    }
    if !c.summary.is_empty() {
        parts.push(c.summary.clone());
    }
    parts.join(" ")
}

/// Cross-modal Tier 2 for one file: if it's an image/audio file and the
/// matching backend is loaded, embed it and rank against the folders' centroids
/// in that modality's latent space. Returns `None` (fall through to text) when
/// the modality has no backend, the file can't be read or embedded, or no
/// folder carries a centroid in that space.
async fn classify_modality_file(
    path: &Path,
    modality: FileModality,
    mime: Option<&str>,
    multimodal: MigrationMultimodal<'_>,
    profiles: &ProfileCache,
    config: &ClassifierConfig,
) -> Option<Verdict> {
    let mime_str = mime.unwrap_or("application/octet-stream");
    // Bound the read: oversized media falls through to text/path evidence
    // rather than being slurped whole into memory for embedding.
    if tokio::fs::metadata(path).await.map_or(0, |m| m.len()) > tidyup_extract::MAX_DOCUMENT_BYTES {
        return None;
    }
    let bytes = tokio::fs::read(path).await.ok()?;
    match modality {
        FileModality::Image => {
            let backend = multimodal.image?;
            let embedding = backend.embed_image(&bytes, mime_str).await.ok()?;
            rank_centroids(
                path,
                &embedding,
                profiles,
                config,
                |p| p.image_centroid.as_deref(),
                "image",
                backend.model_id(),
            )
        }
        FileModality::Audio => {
            let backend = multimodal.audio?;
            let embedding = backend.embed_audio(&bytes, mime_str).await.ok()?;
            rank_centroids(
                path,
                &embedding,
                profiles,
                config,
                |p| p.audio_centroid.as_deref(),
                "audio",
                backend.model_id(),
            )
        }
        _ => None,
    }
}

/// Rank leaf folders that carry a centroid in `select`'s latent space by cosine
/// against `embedding`, returning a [`Verdict`] for the best match.
///
/// `select` extracts the modality-appropriate centroid from a profile
/// (`image_centroid` or `audio_centroid`) — `embedding` must come from the same
/// backend, so the cosine is meaningful and never crosses latent spaces.
/// Returns `None` when no leaf folder has a centroid in this space, so the
/// caller falls through to the text Tier 2 path.
///
/// Bundle members and renames are irrelevant here: this is a loose-file
/// placement, and image/audio renames stay on the (text) EXIF/metadata path
/// exactly as in scan mode, so the verdict is always `RenameProposal::Keep`.
fn rank_centroids(
    path: &Path,
    embedding: &[f32],
    profiles: &ProfileCache,
    config: &ClassifierConfig,
    select: impl Fn(&FolderProfile) -> Option<&[f32]>,
    modality_label: &str,
    model_id: &str,
) -> Option<Verdict> {
    let mut ranked: Vec<(PathBuf, f32)> = Vec::new();
    for folder in &profiles.last_scan.leaf_folders {
        if let Some(profile) = profiles.profiles.get(folder) {
            if let Some(centroid) = select(profile) {
                let score = cosine(embedding, centroid).max(0.0);
                ranked.push((folder.clone(), score));
            }
        }
    }
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (folder, score) = ranked.first().cloned()?;
    let gap = ranked.get(1).map_or(score, |(_, next)| score - next);
    let needs_review = score < config.embedding_threshold || gap < config.ambiguity_gap;

    let candidates: Vec<Candidate> = ranked
        .iter()
        .take(5)
        .map(|(f, s)| Candidate {
            folder: f.clone(),
            score: *s,
            score_breakdown: ScoreBreakdown {
                name_similarity: 0.0,
                centroid_similarity: Some(*s),
            },
        })
        .collect();

    Some(Verdict {
        result: ClassificationResult {
            source_file: path.to_path_buf(),
            candidates,
            resolved_at: Tier::Embedding,
            needs_review,
            suggested_rename: None,
        },
        rename: RenameProposal::Keep,
        destination_folder: folder,
        confidence: score,
        reasoning: format!(
            "tier2 {modality_label}-centroid: cos={score:.3} gap={gap:.3} model={model_id}"
        ),
        classification_confidence: Some(score),
        rename_mismatch_score: None,
    })
}

/// Decide a file's modality from MIME + extension. Mirrors the scan pipeline's
/// helper; kept local so migration mode doesn't depend on scan internals.
fn file_modality(path: &Path, mime: Option<&str>) -> FileModality {
    if let Some(m) = mime {
        if m.starts_with("image/") {
            return FileModality::Image;
        }
        if m.starts_with("audio/") {
            return FileModality::Audio;
        }
        if m.starts_with("video/") {
            return FileModality::Video;
        }
        if m.starts_with("text/") || m == "application/pdf" {
            return FileModality::Text;
        }
    }
    let Some(ext) = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
    else {
        return FileModality::Skip;
    };
    match ext.as_str() {
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" | "webp" | "ico" | "avif"
        | "jxl" | "heic" | "heif" | "raw" | "cr2" | "cr3" | "nef" | "arw" | "orf" | "dng"
        | "rw2" | "raf" => FileModality::Image,
        "mp3" | "flac" | "m4a" | "wav" | "ogg" | "opus" | "aiff" | "aif" | "ape" | "wma"
        | "alac" | "aac" | "mka" => FileModality::Audio,
        "mp4" | "mov" | "mkv" | "avi" | "wmv" | "flv" | "webm" | "m4v" | "mpg" | "mpeg" => {
            FileModality::Video
        }
        _ => FileModality::Text,
    }
}

/// Rank every leaf profile by composite score against a content embedding.
/// Returns `(folder_path, score, breakdown)` sorted top-first.
fn rank_profiles(
    content_embedding: &[f32],
    profiles: &ProfileCache,
    source_path: &Path,
    weights: &ScoreWeights,
) -> Vec<(PathBuf, f32, ScoreBreakdown)> {
    let mut out = Vec::new();
    for path in &profiles.last_scan.leaf_folders {
        let Some(profile) = profiles.profiles.get(path) else {
            continue;
        };
        let breakdown = score_profile(content_embedding, profile, source_path);
        let score = composite(&breakdown, weights);
        out.push((path.clone(), score, breakdown));
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn score_profile(
    content_embedding: &[f32],
    profile: &FolderProfile,
    _source_path: &Path,
) -> ScoreBreakdown {
    let name_similarity = cosine(content_embedding, &profile.name_embedding).max(0.0);
    let centroid_similarity = profile
        .content_centroid
        .as_deref()
        .map(|c| cosine(content_embedding, c).max(0.0));
    ScoreBreakdown {
        name_similarity,
        centroid_similarity,
    }
}

/// Combine sub-scores into a composite confidence in `[0, 1]`.
///
/// When `centroid_similarity` is `None` (common in v0.1), its weight is
/// redistributed onto `name_similarity` so the scale doesn't shrink.
fn composite(b: &ScoreBreakdown, w: &ScoreWeights) -> f32 {
    let (name_w, centroid_term) = b
        .centroid_similarity
        .map_or((w.name + w.centroid, 0.0), |c| (w.name, w.centroid * c));
    name_w.mul_add(b.name_similarity, centroid_term)
}

struct GatedRename {
    proposal: RenameProposal,
    mismatch_score: Option<f32>,
}

#[allow(clippy::too_many_arguments)]
async fn gate_rename(
    path: &Path,
    metadata: &serde_json::Value,
    keywords: &[yake::Keyword],
    year: Option<i32>,
    classification_confidence: f32,
    embeddings: &dyn EmbeddingBackend,
    content_text: Option<&str>,
    filename: &str,
    config: &ClassifierConfig,
) -> Result<GatedRename> {
    let proposal = propose_rename(path, metadata, keywords, year);
    if matches!(proposal, RenameProposal::Keep) {
        return Ok(GatedRename {
            proposal,
            mismatch_score: None,
        });
    }
    let ocr_evidence = matches!(
        &proposal,
        RenameProposal::Rename {
            source: RenameSource::Ocr,
            ..
        }
    );
    if classification_confidence < config.rename.min_classification_confidence && !ocr_evidence {
        return Ok(GatedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: None,
        });
    }
    let Some(content_text) = content_text else {
        return Ok(GatedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: None,
        });
    };
    let filename_vec = embeddings.embed_text(filename).await?;
    let content_vec = embeddings.embed_text(content_text).await?;
    let cos = cosine(&filename_vec, &content_vec);
    let mismatch = 1.0_f32 - cos;
    if mismatch < config.rename.min_mismatch_score && !ocr_evidence {
        return Ok(GatedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: Some(mismatch),
        });
    }
    Ok(GatedRename {
        proposal,
        mismatch_score: Some(mismatch),
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn find_year(s: &str) -> Option<i32> {
    crate::text_util::find_year(s)
}

fn build_proposal(source: &Path, v: &Verdict) -> ChangeProposal {
    let final_name = match &v.rename {
        RenameProposal::Rename { name, .. } => name.clone(),
        RenameProposal::Keep => source
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string(),
    };
    let proposed_path = v.destination_folder.join(&final_name);
    let change_type = match v.rename {
        RenameProposal::Rename { .. } => ChangeType::RenameAndMove,
        RenameProposal::Keep => ChangeType::Move,
    };
    ChangeProposal {
        id: Uuid::new_v4(),
        file_id: None,
        change_type,
        original_path: source.to_path_buf(),
        proposed_path,
        proposed_name: final_name,
        confidence: v.confidence,
        reasoning: v.reasoning.clone(),
        needs_review: v.result.needs_review,
        status: ChangeStatus::Pending,
        created_at: Utc::now(),
        applied_at: None,
        bundle_id: None,
        classification_confidence: v.classification_confidence,
        rename_mismatch_score: v.rename_mismatch_score,
        content_hash: crate::hashing::content_hash_of(source),
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_content_bundle_proposal(
    bundle: &DetectedBundle,
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
    text_backend: Option<&dyn TextBackend>,
    multimodal: MigrationMultimodal<'_>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
) -> Result<BundleProposal> {
    let mut classified = Vec::with_capacity(bundle.members.len());
    for member in &bundle.members {
        let verdict = classify_file(
            member,
            profiles,
            embeddings,
            text_backend,
            multimodal,
            extractors,
            config,
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("no classification for {}", member.display()))?;
        classified.push((member.clone(), verdict));
    }

    let label = bundle.target_subdir.clone().unwrap_or_else(|| {
        bundle
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("collection")
            .to_string()
    });
    let target_parent = choose_collection_target(
        bundle,
        &label,
        &classified,
        profiles,
        embeddings,
        text_backend,
        config,
    )
    .await
    .ok_or_else(|| anyhow::anyhow!("no semantic destination for collection {label}"))?;
    let collection_root = target_parent.join(&label);
    let allow_renames = bundle.kind.allows_member_renames();

    let mut confidence_sum = 0.0_f32;
    let mut proposals = Vec::with_capacity(classified.len());
    for (source, verdict) in classified {
        confidence_sum += verdict.confidence;
        let mut proposal = build_proposal(&source, &verdict);
        if !allow_renames {
            proposal.change_type = ChangeType::Move;
            proposal.proposed_name = source
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            proposal.rename_mismatch_score = None;
        }
        proposal.proposed_path = collection_root.join(&proposal.proposed_name);
        proposal.reasoning = format!("{}; {}", bundle.reasoning, proposal.reasoning);
        proposals.push(proposal);
    }

    let count = u16::try_from(proposals.len()).unwrap_or(u16::MAX);
    let confidence = if count == 0 {
        0.0
    } else {
        confidence_sum / f32::from(count)
    };
    Ok(BundleProposal::new(
        bundle.root.clone(),
        bundle.kind.clone(),
        target_parent,
        proposals,
        confidence,
        bundle.reasoning.clone(),
    )?)
}

async fn choose_collection_target(
    bundle: &DetectedBundle,
    label: &str,
    classified: &[(PathBuf, Verdict)],
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
    text_backend: Option<&dyn TextBackend>,
    config: &ClassifierConfig,
) -> Option<PathBuf> {
    let first = classified.first()?.1.destination_folder.clone();
    if classified
        .iter()
        .all(|(_, verdict)| verdict.destination_folder == first)
    {
        return Some(first);
    }

    let filenames = classified
        .iter()
        .filter_map(|(path, verdict)| match &verdict.rename {
            RenameProposal::Rename { name, .. } => Path::new(name)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_owned),
            RenameProposal::Keep => path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_owned),
        })
        .map(|name| normalize_semantic_text(&name))
        .collect::<Vec<_>>()
        .join("; ");
    let query = format!(
        "cohesive collection named {}; member artifacts: {filenames}",
        normalize_semantic_text(label)
    );
    let embedding = embeddings.embed_text(&query).await.ok()?;
    let ranked = rank_profiles(&embedding, profiles, &bundle.root, &config.weights);
    let (mut folder, score, _) = ranked.first()?.clone();
    let gap = ranked.get(1).map_or(score, |(_, second, _)| score - second);
    let uncertain = score < config.embedding_threshold || gap < config.ambiguity_gap;
    if uncertain && config.enable_llm_fallback {
        if let Some(backend) = text_backend {
            if let Ok(Some((reranked, _))) = tier3_rerank(
                backend,
                embeddings,
                profiles,
                &bundle.root,
                &query,
                label,
                &config.weights,
            )
            .await
            {
                if let Some((llm_folder, llm_score, _)) = reranked.first() {
                    if *llm_score > score {
                        folder.clone_from(llm_folder);
                    }
                }
            }
        }
    }
    Some(folder)
}

/// Bundle placement: embed structural and naming evidence and pick the
/// top-scoring learned profile. Bundle detection controls atomicity only.
async fn build_bundle_proposal(
    bundle: &DetectedBundle,
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
) -> Result<BundleProposal> {
    let leaf_name = bundle
        .root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("bundle")
        .to_string();

    let target_parent = pick_bundle_target(bundle, profiles, embeddings, &leaf_name)
        .await
        .ok_or_else(|| anyhow::anyhow!("no semantic destination for bundle {leaf_name}"))?;

    let bundle_target_root = target_parent.join(&leaf_name);

    let mut members = Vec::with_capacity(bundle.members.len());
    for m in &bundle.members {
        let rel = m.strip_prefix(&bundle.root).unwrap_or(m);
        let proposed_path = bundle_target_root.join(rel);
        let name = m
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        members.push(ChangeProposal {
            id: Uuid::new_v4(),
            file_id: None,
            change_type: ChangeType::Move,
            original_path: m.clone(),
            proposed_path,
            proposed_name: name,
            confidence: 0.90,
            reasoning: bundle.reasoning.clone(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: None,
            rename_mismatch_score: None,
            content_hash: crate::hashing::content_hash_of(m),
        });
    }

    Ok(BundleProposal::new(
        bundle.root.clone(),
        bundle.kind.clone(),
        target_parent,
        members,
        0.90,
        bundle.reasoning.clone(),
    )?)
}

async fn pick_bundle_target(
    bundle: &DetectedBundle,
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
    leaf_name: &str,
) -> Option<PathBuf> {
    if profiles.last_scan.leaf_folders.is_empty() {
        return None;
    }
    // Bundle kind controls atomicity, not placement. Placement is ranked from
    // the kind label, collection name, and member names against the learned
    // target-folder profiles.
    let members = bundle
        .members
        .iter()
        .take(24)
        .filter_map(|member| member.file_name().and_then(|name| name.to_str()))
        .collect::<Vec<_>>()
        .join(" ");
    let query = format!(
        "atomic collection kind {} name {} members {}",
        bundle.kind.as_str(),
        leaf_name,
        members
    );
    let query_embedding = embeddings.embed_text(&query).await.ok()?;
    let mut best: Option<(PathBuf, f32)> = None;
    for path in &profiles.last_scan.leaf_folders {
        if let Some(profile) = profiles.profiles.get(path) {
            let score = cosine(&query_embedding, &profile.name_embedding);
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((path.clone(), score));
            }
        }
    }
    best.map(|(p, _)| p)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::fs;
    use std::time::SystemTime;
    use tempfile::TempDir;
    use tidyup_core::extractor::ExtractedContent;
    use tidyup_domain::migration::{FolderMetadata, FolderNode, OrganizationType, TargetScan};
    use tidyup_domain::BundleKind;

    struct NullProgress;
    #[async_trait]
    impl ProgressReporter for NullProgress {
        async fn phase_started(&self, _p: Phase, _t: Option<u64>) {}
        async fn item_completed(&self, _p: Phase, _i: ProgressItem) {}
        async fn phase_finished(&self, _p: Phase) {}
        async fn message(&self, _l: Level, _m: &str) {}
    }

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

    fn make_node(path: &Path, name: &str, dominant: &[&str]) -> FolderNode {
        let mut ext_counts: HashMap<String, u32> = HashMap::new();
        for e in dominant {
            ext_counts.insert((*e).to_string(), 10);
        }
        FolderNode {
            path: path.to_path_buf(),
            name: name.to_string(),
            path_segments: vec![name.to_string()],
            depth: 1,
            children: vec![],
            sibling_names: vec![],
            metadata: FolderMetadata {
                file_count: 10,
                recursive_file_count: 10,
                extension_counts: ext_counts,
                dominant_extensions: dominant.iter().map(|e| (*e).to_string()).collect(),
                date_range: None,
                avg_file_size: 1024,
                has_children: false,
                content_hash: String::new(),
                scanned_at: SystemTime::now(),
            },
        }
    }

    async fn make_profile(
        path: &Path,
        description: &str,
        dominant: &[&str],
        eb: &BucketEmbeddings,
    ) -> FolderProfile {
        let emb = eb.embed_text(description).await.unwrap();
        FolderProfile {
            path: path.to_path_buf(),
            name_embedding: emb,
            content_centroid: None,
            centroid_sample_count: 0,
            image_centroid: None,
            image_centroid_sample_count: 0,
            audio_centroid: None,
            audio_centroid_sample_count: 0,
            metadata: make_node(path, "x", dominant).metadata,
            organization_type: OrganizationType::Semantic,
            profile_confidence: 0.9,
            last_updated: SystemTime::now(),
        }
    }

    async fn sample_cache(target_root: &Path, eb: &BucketEmbeddings) -> ProfileCache {
        let finance = target_root.join("Finance");
        let code = target_root.join("Code");
        let photos = target_root.join("Photos");
        fs::create_dir_all(&finance).unwrap();
        fs::create_dir_all(&code).unwrap();
        fs::create_dir_all(&photos).unwrap();

        let mut profiles: HashMap<PathBuf, FolderProfile> = HashMap::new();
        profiles.insert(
            finance.clone(),
            make_profile(
                &finance,
                "tax return W-2 1099 1040 IRS refund withholding",
                &[".pdf"],
                eb,
            )
            .await,
        );
        profiles.insert(
            code.clone(),
            make_profile(
                &code,
                "source code rust python javascript compile function",
                &[".rs", ".py"],
                eb,
            )
            .await,
        );
        profiles.insert(
            photos.clone(),
            make_profile(
                &photos,
                "jpeg raw photograph camera exif landscape portrait",
                &[".jpg", ".png"],
                eb,
            )
            .await,
        );

        let mut nodes: HashMap<PathBuf, FolderNode> = HashMap::new();
        nodes.insert(finance.clone(), make_node(&finance, "Finance", &[".pdf"]));
        nodes.insert(code.clone(), make_node(&code, "Code", &[".rs", ".py"]));
        nodes.insert(
            photos.clone(),
            make_node(&photos, "Photos", &[".jpg", ".png"]),
        );

        let scan = TargetScan {
            root: target_root.to_path_buf(),
            nodes,
            leaf_folders: vec![finance, code, photos],
            scan_timestamp: SystemTime::now(),
        };

        ProfileCache {
            target_root: target_root.to_path_buf(),
            model_id: "bucket".to_string(),
            embedding_dim: 7,
            profiles,
            last_scan: scan,
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn loose_file_classified_into_leaf_folder() {
        let src = TempDir::new().unwrap();
        let tgt = TempDir::new().unwrap();
        fs::write(src.path().join("notes.txt"), b"hello world").unwrap();

        let eb = BucketEmbeddings;
        let profiles = sample_cache(tgt.path(), &eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };

        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal::default(),
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        // Destination must be one of the leaf folders we registered.
        let leaves: Vec<_> = profiles
            .last_scan
            .leaf_folders
            .iter()
            .map(PathBuf::as_path)
            .collect();
        assert!(
            leaves.iter().any(|l| p.proposed_path.starts_with(l)),
            "proposed_path {:?} not under any registered leaf",
            p.proposed_path
        );
    }

    #[tokio::test]
    async fn semantic_ranking_routes_source_into_a_profile_leaf() {
        let src = TempDir::new().unwrap();
        let tgt = TempDir::new().unwrap();
        fs::write(src.path().join("main.rs"), b"fn main() {}").unwrap();

        let eb = BucketEmbeddings;
        let profiles = sample_cache(tgt.path(), &eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];

        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal::default(),
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(profiles
            .last_scan
            .leaf_folders
            .iter()
            .any(|leaf| p.proposed_path.starts_with(leaf)));
        assert!(p.reasoning.contains("tier2 composite"));
    }

    #[tokio::test]
    async fn bundle_routed_via_profile_similarity() {
        let src = TempDir::new().unwrap();
        let tgt = TempDir::new().unwrap();
        fs::create_dir_all(src.path().join("myproj/src")).unwrap();
        fs::write(
            src.path().join("myproj/Cargo.toml"),
            b"[package]\nname='x'\n",
        )
        .unwrap();
        fs::write(src.path().join("myproj/src/main.rs"), b"fn main() {}").unwrap();

        let eb = BucketEmbeddings;
        let profiles = sample_cache(tgt.path(), &eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];

        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal::default(),
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.bundles.len(), 1);
        let b = &out.bundles[0];
        assert_eq!(b.kind, BundleKind::RustCrate);
        // target_parent is one of the registered leaves.
        assert!(profiles
            .last_scan
            .leaf_folders
            .iter()
            .any(|l| l == &b.target_parent));
        assert_eq!(b.members.len(), 2);
    }

    #[tokio::test]
    async fn empty_cache_yields_unclassified_for_loose_files() {
        let src = TempDir::new().unwrap();
        let tgt = TempDir::new().unwrap();
        fs::write(src.path().join("weird.xyz"), b"opaque").unwrap();

        let eb = BucketEmbeddings;
        // Empty cache (no leaves).
        let profiles = ProfileCache {
            target_root: tgt.path().to_path_buf(),
            model_id: "bucket".to_string(),
            embedding_dim: 7,
            profiles: HashMap::new(),
            last_scan: TargetScan {
                root: tgt.path().to_path_buf(),
                nodes: HashMap::new(),
                leaf_folders: vec![],
                scan_timestamp: SystemTime::now(),
            },
            created_at: SystemTime::now(),
            last_updated: SystemTime::now(),
        };
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];

        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal::default(),
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 0);
        assert_eq!(out.unclassified.len(), 1);
    }

    /// Stub `TextBackend` for Tier 3 tests. Returns a fixed
    /// `ContentClassification`; non-text methods are unreachable.
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
            "stub-llm"
        }
    }

    #[tokio::test]
    async fn tier3_llm_rerank_overrides_uncertain_tier2() {
        let src = TempDir::new().unwrap();
        let tgt = TempDir::new().unwrap();
        // Content embeds poorly against the Finance profile via raw bytes,
        // forcing Tier 2 needs_review. The stub LLM emits a Finance-shaped
        // summary, which re-embeds strongly against the Finance profile.
        fs::write(src.path().join("anonymous.dat"), b"x x x x x x").unwrap();

        let eb = BucketEmbeddings;
        let profiles = sample_cache(tgt.path(), &eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];

        let cfg = ClassifierConfig {
            embedding_threshold: 0.99,
            ambiguity_gap: 0.50,
            enable_llm_fallback: true,
            ..ClassifierConfig::default()
        };
        // Empty category/tags so the LLM query is exactly the Finance
        // profile's description — keeps the assertion about *wiring*, not
        // embedder fidelity (the bucket embedder is byte-collision-prone).
        let llm = StubTextBackend {
            category: "",
            tags: vec![],
            summary: "tax return W-2 1099 1040 IRS refund withholding",
        };

        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            Some(&llm),
            MigrationMultimodal::default(),
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
            p.proposed_path.starts_with(tgt.path().join("Finance")),
            "expected Finance route, got {:?}",
            p.proposed_path,
        );
        assert_eq!(out.classifications.len(), 1);
        assert_eq!(out.classifications[0].resolved_at, Tier::Llm);
    }

    #[tokio::test]
    async fn tier3_reroute_gates_rename_on_tier2_confidence_not_rerank() {
        // Migration-mode counterpart of the scan-mode regression: the rename
        // gate must read Tier 2's confidence (`tier2_score`), not the boosted
        // post-Tier-3 `chosen_score`. classification_confidence on the proposal
        // is therefore the (lower) Tier-2 score, strictly below the routing
        // confidence after a reroute. Pre-fix, both equalled chosen_score.
        let src = TempDir::new().unwrap();
        let tgt = TempDir::new().unwrap();
        fs::write(src.path().join("anonymous.dat"), b"x x x x x x").unwrap();

        let eb = BucketEmbeddings;
        let profiles = sample_cache(tgt.path(), &eb).await;
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

        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            Some(&llm),
            MigrationMultimodal::default(),
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
        let gate_score = p.classification_confidence.unwrap();
        assert!(
            gate_score < p.confidence,
            "rename gate score {gate_score} must be the (lower) Tier-2 score, \
             not the boosted routing confidence {}",
            p.confidence,
        );
        assert_eq!(
            p.change_type,
            ChangeType::Move,
            "Tier-3 reroute must not produce a rename"
        );
    }

    #[test]
    fn composite_redistributes_centroid_weight_when_absent() {
        let w = ScoreWeights::default();
        let with_centroid = ScoreBreakdown {
            name_similarity: 0.5,
            centroid_similarity: Some(0.5),
        };
        let without_centroid = ScoreBreakdown {
            name_similarity: 0.5,
            centroid_similarity: None,
        };
        let a = composite(&with_centroid, &w);
        let b = composite(&without_centroid, &w);
        // Both live weights sum to one, and the absent centroid weight moves
        // onto the name signal.
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn composite_ceiling_can_clear_default_rename_gate() {
        let ceiling = composite(
            &ScoreBreakdown {
                name_similarity: 1.0,
                centroid_similarity: Some(1.0),
            },
            &ScoreWeights::default(),
        );
        assert!(
            ceiling >= tidyup_domain::RenameConfig::default().min_classification_confidence,
            "composite ceiling {ceiling} must reach the rename confidence gate",
        );
        assert!((ceiling - 1.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Phase 8 — migration-mode multimodal centroids
    // -----------------------------------------------------------------------

    /// Toy cross-modal image backend: embeds image bytes and text into the same
    /// 4-bucket L2-normalized space, so a folder centroid built from sample
    /// images is comparable to a source image's embedding. Deterministic on the
    /// leading bytes, so "similar" fixtures land near each other.
    struct BucketImageBackend;
    #[async_trait]
    impl ImageEmbeddingBackend for BucketImageBackend {
        async fn embed_image(&self, bytes: &[u8], _mime: &str) -> Result<Vec<f32>> {
            Ok(bucket4(bytes))
        }
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(bucket4(text.as_bytes()))
        }
        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| bucket4(t.as_bytes())).collect())
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn model_id(&self) -> &'static str {
            "bucket-image"
        }
    }

    fn bucket4(bytes: &[u8]) -> Vec<f32> {
        let mut v = vec![0.0_f32; 4];
        for (i, b) in bytes.iter().enumerate() {
            v[i % 4] += f32::from(*b);
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= n;
        }
        v
    }

    /// Two folders, one with an image centroid: an image source file matching
    /// that folder's images routes there via the image-centroid path, and the
    /// reasoning records it came from the cross-modal tier — not text.
    #[tokio::test]
    async fn image_source_routes_against_image_centroid() {
        use crate::profiler::{build_profile_cache_multimodal, scan_target, MultimodalProfilers};

        let tgt = TempDir::new().unwrap();
        // Photos/ holds a couple of "images" (PNG bytes). Docs/ holds none.
        let photos = tgt.path().join("Photos");
        let docs = tgt.path().join("Docs");
        fs::create_dir_all(&photos).unwrap();
        fs::create_dir_all(&docs).unwrap();
        let png = make_png_bytes();
        fs::write(photos.join("a.png"), &png).unwrap();
        fs::write(photos.join("b.png"), &png).unwrap();
        fs::write(docs.join("note.txt"), b"just some prose here").unwrap();

        let eb = BucketEmbeddings;
        let img = BucketImageBackend;
        let scan = scan_target(tgt.path()).unwrap();
        let profiles = build_profile_cache_multimodal(
            &scan,
            &eb,
            MultimodalProfilers {
                image: Some(&img),
                audio: None,
                extractors: &[],
            },
        )
        .await
        .unwrap();

        // Photos got an image centroid; Docs did not.
        assert!(
            profiles.profiles[&photos].image_centroid.is_some(),
            "Photos should have an image centroid built from its 2 images",
        );
        assert_eq!(profiles.profiles[&photos].image_centroid_sample_count, 2);
        assert!(
            profiles.profiles[&docs].image_centroid.is_none(),
            "Docs has no images → no image centroid",
        );

        // A source image identical to the Photos fixtures must route to Photos
        // via the image-centroid tier. No extractors: a text extractor would
        // advertise text/plain and short-circuit modality detection — we want
        // the real PNG MIME (sniffed by `tidyup_extract::mime::detect`) to drive
        // the image branch.
        let src = TempDir::new().unwrap();
        fs::write(src.path().join("vacation.png"), &png).unwrap();

        let ex: Vec<Arc<dyn ContentExtractor>> = vec![];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };
        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal {
                image: Some(&img),
                audio: None,
            },
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(
            p.reasoning.contains("tier2 image-centroid"),
            "expected image-centroid routing, got: {}",
            p.reasoning,
        );
        assert!(
            p.proposed_path.starts_with(&photos),
            "image should route to Photos, got {:?}",
            p.proposed_path,
        );
        assert_eq!(out.classifications[0].resolved_at, Tier::Embedding);
    }

    /// Regression signal (migration mode): a recognized image routes through
    /// the image centroid whenever that semantic backend is available.
    #[tokio::test]
    async fn image_reaches_centroid_at_default_threshold_when_backend_present() {
        use crate::profiler::{build_profile_cache_multimodal, scan_target, MultimodalProfilers};

        let tgt = TempDir::new().unwrap();
        let photos = tgt.path().join("Photos");
        fs::create_dir_all(&photos).unwrap();
        let png = make_png_bytes();
        fs::write(photos.join("a.png"), &png).unwrap();
        fs::write(photos.join("b.png"), &png).unwrap();

        let eb = BucketEmbeddings;
        let img = BucketImageBackend;
        let scan = scan_target(tgt.path()).unwrap();
        let profiles = build_profile_cache_multimodal(
            &scan,
            &eb,
            MultimodalProfilers {
                image: Some(&img),
                audio: None,
                extractors: &[],
            },
        )
        .await
        .unwrap();

        let src = TempDir::new().unwrap();
        fs::write(src.path().join("vacation.png"), &png).unwrap();
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };
        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal {
                image: Some(&img),
                audio: None,
            },
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        assert!(
            out.proposals[0].reasoning.contains("tier2 image-centroid"),
            "the image must reach the centroid path; got: {}",
            out.proposals[0].reasoning,
        );
        assert_eq!(out.classifications[0].resolved_at, Tier::Embedding);
    }

    /// With no folder carrying an image centroid (backend present at classify
    /// time but profiles built without one), an image source file falls through
    /// to the text Tier 2 path — it must never be force-placed by cross-comparing
    /// the image embedding against a text `name_embedding`. Here the image has no
    /// extractable text either, so the honest outcome is "unclassified": no
    /// proposal is fabricated across latent spaces.
    #[tokio::test]
    async fn image_source_falls_through_when_no_centroid() {
        let tgt = TempDir::new().unwrap();
        let eb = BucketEmbeddings;
        // Text-only profiles: no image centroids anywhere.
        let profiles = sample_cache(tgt.path(), &eb).await;

        let src = TempDir::new().unwrap();
        fs::write(src.path().join("photo.png"), make_png_bytes()).unwrap();

        let img = BucketImageBackend;
        // No extractor means real PNG MIME drives modality routing and there is
        // no text evidence when the image backend has no matching centroid.
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![];
        let cfg = ClassifierConfig {
            embedding_threshold: 0.0,
            ambiguity_gap: 0.0,
            ..ClassifierConfig::default()
        };
        let out = run_migration(
            src.path(),
            &profiles,
            &eb,
            None,
            MigrationMultimodal {
                image: Some(&img),
                audio: None,
            },
            &ex,
            &cfg,
            &NullProgress,
        )
        .await
        .unwrap();

        // No image centroid to match → the image-centroid branch returns None
        // and the file falls through to generalized text-space semantic
        // evidence. It must never compare an image embedding with a text-space
        // folder vector.
        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(
            !p.reasoning.contains("image-centroid"),
            "must not route via the image tier when no image centroid exists, got: {}",
            p.reasoning,
        );
        assert!(p.reasoning.contains("tier2 composite"));
    }

    /// A minimal valid PNG so `tidyup_extract::mime::detect` identifies the
    /// fixture as `image/` and the modality router sends it down the image path.
    fn make_png_bytes() -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        let img = image::RgbImage::new(2, 2);
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }
}
