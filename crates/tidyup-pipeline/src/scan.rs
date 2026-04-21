//! Scan-mode pipeline — classify a source tree against a fixed taxonomy.
//!
//! The scan pipeline runs the three-tier cascade with a *taxonomy* as target:
//! each file ends up proposed under a well-known category folder (`Finance/`,
//! `Photos/`, `Code/`, …). Contrast with the migration pipeline in
//! [`crate::migration`], which classifies against an arbitrary existing
//! folder hierarchy.
//!
//! # Cascade
//!
//! 1. **Bundle detection** (in the scanner) carves out opaque subtrees; scan
//!    mode routes each [`BundleKind`] to a default taxonomy folder by kind.
//! 2. **Tier 1 — heuristics** (`[crate::heuristics]`): extension / MIME /
//!    marker filename. Fires at `heuristic_threshold` (default 0.60).
//! 3. **Tier 2 — embeddings**: caller-supplied [`ScanCandidate`]s provide
//!    pre-computed description embeddings; the pipeline embeds the file
//!    content and picks the highest-cosine candidate. Fires at
//!    `embedding_threshold` (default 0.35) with `ambiguity_gap`.
//! 4. **Below thresholds**: surface to review via `needs_review = true`.
//!    Tier 3 LLM fallback is out of scope for this module — the migration
//!    pipeline owns the feature-gated call-out.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tidyup_core::extractor::ContentExtractor;
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::inference::{
    AudioEmbeddingBackend, EmbeddingBackend, FileModality, ImageEmbeddingBackend,
};
use tidyup_domain::change::{ChangeProposal, ChangeStatus, ChangeType};
use tidyup_domain::{BundleKind, BundleProposal, ClassifierConfig, Phase};
use uuid::Uuid;

use crate::heuristics::{self, HeuristicMatch};
use crate::naming::{propose_rename, RenameProposal};
use crate::scanner::{self, DetectedBundle};
use crate::yake;

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

/// Optional multimodal Tier 2 backends + per-modality candidate lists.
///
/// Phase 7 wiring: when a backend is present, the pipeline routes files of
/// that modality through it instead of (or in addition to) the text cascade.
/// Each candidate list embeds in the **modality-specific** latent space and
/// is NOT interchangeable with the text-only `ScanCandidate` list passed at
/// the top level — the pipeline keeps them separate so a misconfigured
/// caller can't compute a meaningless cross-space cosine.
///
/// `Default::default()` returns an empty context so legacy text-only
/// callers don't have to know about Phase 7.
#[derive(Default)]
#[allow(missing_debug_implementations)] // trait objects don't implement Debug
pub struct MultimodalContext<'a> {
    pub image: Option<ImageContext<'a>>,
    pub audio: Option<AudioContext<'a>>,
}

/// Image-side classification context: SigLIP-style backend + image taxonomy.
#[allow(missing_debug_implementations)]
pub struct ImageContext<'a> {
    pub backend: &'a dyn ImageEmbeddingBackend,
    pub candidates: &'a [ScanCandidate],
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
    /// Files the cascade couldn't classify at all (heuristic miss + no
    /// extractable content). Listed here so callers can surface them rather
    /// than silently drop.
    pub unclassified: Vec<PathBuf>,
}

/// Drive the scan cascade end-to-end.
///
/// `output_root` is the destination prefix applied to each taxonomy leaf
/// path — typically the source root itself (to reorganize in-place) or a
/// user-specified directory.
///
/// `multimodal` carries optional Phase 7 image/audio backends + their per-
/// modality candidate lists. Pass [`MultimodalContext::default()`] for the
/// text-only cascade.
///
/// # Errors
/// Propagates source-read and embedding-backend failures. Per-file extraction
/// or classification errors are logged via `progress.message(Level::Warn, …)`
/// and surfaced through [`ScanOutcome::unclassified`].
#[allow(clippy::too_many_arguments)]
pub async fn run_scan(
    source_root: &Path,
    output_root: &Path,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    multimodal: &MultimodalContext<'_>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
    progress: &dyn ProgressReporter,
) -> Result<ScanOutcome> {
    progress.phase_started(Phase::Indexing, None).await;
    let tree = scanner::scan(source_root);
    progress.phase_finished(Phase::Indexing).await;

    let mut outcome = ScanOutcome {
        proposals: Vec::new(),
        bundles: Vec::new(),
        unclassified: Vec::new(),
    };

    // Bundles first — they don't pass through Tier 2.
    for bundle in &tree.bundles {
        match build_bundle_proposal(bundle, output_root) {
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

    #[allow(clippy::cast_possible_truncation)]
    let total = tree.loose_files.len() as u64;
    progress
        .phase_started(Phase::Classifying, Some(total))
        .await;

    for (idx, path) in tree.loose_files.iter().enumerate() {
        match classify_file(path, candidates, embeddings, multimodal, extractors, config).await {
            Ok(Some(classified)) => {
                let proposal = build_proposal(path, output_root, &classified);
                outcome.proposals.push(proposal);
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
                    #[allow(clippy::cast_possible_truncation)]
                    current: (idx as u64) + 1,
                    total: Some(total),
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

/// Classify a single loose file through Tiers 1→2 and return the best verdict.
///
/// Returns `Ok(None)` when no tier produced any signal (typically: unknown
/// extension and extraction failure).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn classify_file(
    path: &Path,
    candidates: &[ScanCandidate],
    embeddings: &dyn EmbeddingBackend,
    multimodal: &MultimodalContext<'_>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
) -> Result<Option<ClassifiedFile>> {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();

    // Try to extract early — both Tier 1 (MIME) and Tier 2 (content) want it.
    let mime = tidyup_extract::mime::detect(path).await;
    let extracted = match tidyup_extract::router::pick(extractors, path, mime.as_deref()) {
        Some(ext) => ext.extract(path).await.ok(),
        None => None,
    };
    let effective_mime = extracted
        .as_ref()
        .map(|e| e.mime.clone())
        .or_else(|| mime.clone());

    // Tier 1.
    if let Some(hit) = heuristics::classify(path, effective_mime.as_deref()) {
        if hit.confidence >= config.heuristic_threshold {
            let year =
                year_from_path_and_text(path, extracted.as_ref().and_then(|e| e.text.as_deref()));
            let keywords = extracted
                .as_ref()
                .and_then(|e| e.text.as_deref())
                .map(|t| yake::extract_keywords(t, 8))
                .unwrap_or_default();
            let metadata_json = extracted
                .as_ref()
                .map_or(serde_json::Value::Null, |e| e.metadata.clone());
            let rename = gate_rename(
                path,
                &metadata_json,
                &keywords,
                year,
                hit.confidence,
                embeddings,
                extracted.as_ref().and_then(|e| e.text.as_deref()),
                &filename,
                config,
            )
            .await?;
            return Ok(Some(ClassifiedFile {
                folder_path: hit.taxonomy_path.to_string(),
                confidence: hit.confidence,
                reasoning: format!("tier1 heuristic: {}", hit.reason),
                needs_review: false,
                year,
                temporal: candidates
                    .iter()
                    .find(|c| c.folder_path == hit.taxonomy_path)
                    .is_some_and(|c| c.temporal),
                rename: rename.proposal,
                classification_confidence: Some(hit.confidence),
                rename_mismatch_score: rename.mismatch_score,
            }));
        }
    }

    // Tier 2 — modality-aware. Image/audio files with the corresponding
    // backend present go through cross-modal classification (SigLIP / CLAP).
    // Fall through to text classification for everything else, or if the
    // modality backend is absent.
    let modality = file_modality(path, effective_mime.as_deref());
    if matches!(modality, FileModality::Image) {
        if let Some(img_ctx) = multimodal.image.as_ref() {
            if let Some(verdict) = classify_image(
                path,
                img_ctx,
                effective_mime.as_deref(),
                extracted.as_ref(),
                config,
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

    // Tier 2 text path. Requires extracted text.
    let text = extracted.as_ref().and_then(|e| e.text.as_deref());
    let Some(text) = text else {
        // No text → no embedding → we can only report Tier 1 if it fired at
        // any confidence, even below the threshold. Emit weak proposal flagged
        // for review.
        if let Some(hit) = heuristics::classify(path, effective_mime.as_deref()) {
            return Ok(Some(weak_heuristic(&hit, path)));
        }
        return Ok(None);
    };

    let embedding = embeddings.embed_text(text).await?;
    let (best_idx, best_score, gap) = best_match(&embedding, candidates);
    if best_idx.is_none() {
        return Ok(None);
    }
    let idx = best_idx.unwrap_or(0);
    let candidate = &candidates[idx];
    let needs_review = best_score < config.embedding_threshold || gap < config.ambiguity_gap;

    let year = year_from_path_and_text(path, Some(text));
    let keywords = yake::extract_keywords(text, 8);
    let metadata_json = extracted
        .as_ref()
        .map_or(serde_json::Value::Null, |e| e.metadata.clone());
    let rename = gate_rename(
        path,
        &metadata_json,
        &keywords,
        year,
        best_score,
        embeddings,
        Some(text),
        &filename,
        config,
    )
    .await?;

    Ok(Some(ClassifiedFile {
        folder_path: candidate.folder_path.clone(),
        confidence: best_score,
        reasoning: format!(
            "tier2 embedding: cos={best_score:.3} gap={gap:.3} model={}",
            embeddings.model_id(),
        ),
        needs_review,
        year,
        temporal: candidate.temporal,
        rename: rename.proposal,
        classification_confidence: Some(best_score),
        rename_mismatch_score: rename.mismatch_score,
    }))
}

/// Decide a file's modality from its extension + MIME. Used to pick which
/// Tier 2 backend handles it. Mirrors the classification in
/// `tidyup-extract` but produces the [`FileModality`] enum the inference
/// crate exposes.
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

/// Image-modality Tier 2 — `SigLIP` cross-modal cosine against image taxonomy.
///
/// Returns `Ok(None)` when the image fails to read or the candidate list is
/// empty. The caller falls back to the text Tier 2 path in that case.
async fn classify_image(
    path: &Path,
    ctx: &ImageContext<'_>,
    mime: Option<&str>,
    extracted: Option<&tidyup_core::extractor::ExtractedContent>,
    config: &ClassifierConfig,
) -> Result<Option<ClassifiedFile>> {
    if ctx.candidates.is_empty() {
        return Ok(None);
    }
    let Ok(bytes) = tokio::fs::read(path).await else {
        return Ok(None);
    };
    let mime_str = mime.unwrap_or("application/octet-stream");
    let Ok(embedding) = ctx.backend.embed_image(&bytes, mime_str).await else {
        return Ok(None);
    };
    let (best_idx, best_score, gap) = best_match(&embedding, ctx.candidates);
    let Some(idx) = best_idx else {
        return Ok(None);
    };
    let candidate = &ctx.candidates[idx];
    let needs_review = best_score < config.embedding_threshold || gap < config.ambiguity_gap;
    let year = year_from_path_and_text(path, extracted.and_then(|e| e.text.as_deref()));
    Ok(Some(ClassifiedFile {
        folder_path: candidate.folder_path.clone(),
        confidence: best_score,
        reasoning: format!(
            "tier2 image: cos={best_score:.3} gap={gap:.3} model={}",
            ctx.backend.model_id(),
        ),
        needs_review,
        year,
        temporal: candidate.temporal,
        // Image-side rename gating is intentionally not wired in v0.1 of
        // Phase 7 — extractive renames for images come from the existing
        // EXIF metadata path, which the text Tier 2 cascade already covers.
        // Surfacing image filenames as-is keeps the cross-modal cut conservative.
        rename: RenameProposal::Keep,
        classification_confidence: Some(best_score),
        rename_mismatch_score: None,
    }))
}

/// Audio-modality Tier 2 — `CLAP` cross-modal cosine against audio taxonomy.
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
    let candidate = &ctx.candidates[idx];
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

fn weak_heuristic(hit: &HeuristicMatch, _path: &Path) -> ClassifiedFile {
    ClassifiedFile {
        folder_path: hit.taxonomy_path.to_string(),
        confidence: hit.confidence,
        reasoning: format!("tier1 heuristic (below threshold): {}", hit.reason),
        needs_review: true,
        year: None,
        temporal: false,
        rename: RenameProposal::Keep,
        classification_confidence: Some(hit.confidence),
        rename_mismatch_score: None,
    }
}

struct GatedRename {
    proposal: RenameProposal,
    mismatch_score: Option<f32>,
}

/// Run the rename cascade and apply the two-signal gate from `RenameConfig`.
///
/// Returns `RenameProposal::Keep` (and no mismatch score) when either gate
/// fails, so downstream code can treat "Keep" uniformly without reading the
/// thresholds itself. The raw mismatch score is surfaced when it was
/// computed, so the proposal can log it even on Keep.
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

    if classification_confidence < config.rename.min_classification_confidence {
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

    if mismatch < config.rename.min_mismatch_score {
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
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

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
    text.and_then(|t| find_year(&t[..t.len().min(1000)]))
}

fn find_year(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    if bytes.len() < 4 {
        return None;
    }
    for i in 0..=bytes.len() - 4 {
        if bytes[i] == b'2'
            && bytes[i + 1] == b'0'
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            let after_ok = i + 4 >= bytes.len() || !bytes[i + 4].is_ascii_digit();
            if before_ok && after_ok {
                if let Ok(y) = s[i..i + 4].parse::<i32>() {
                    if (2000..=2039).contains(&y) {
                        return Some(y);
                    }
                }
            }
        }
    }
    None
}

fn build_proposal(source: &Path, output_root: &Path, c: &ClassifiedFile) -> ChangeProposal {
    let final_name = match &c.rename {
        RenameProposal::Rename { name, .. } => name.clone(),
        RenameProposal::Keep => source
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string(),
    };
    let destination_dir = destination_dir(output_root, &c.folder_path, c.year, c.temporal);
    let proposed_path = destination_dir.join(&final_name);

    let change_type = match c.rename {
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
        confidence: c.confidence,
        reasoning: c.reasoning.clone(),
        needs_review: c.needs_review,
        status: ChangeStatus::Pending,
        created_at: Utc::now(),
        applied_at: None,
        bundle_id: None,
        classification_confidence: c.classification_confidence,
        rename_mismatch_score: c.rename_mismatch_score,
    }
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

/// Default taxonomy placement for a bundle kind in scan mode.
const fn bundle_taxonomy(kind: &BundleKind) -> &'static str {
    match kind {
        BundleKind::GitRepository
        | BundleKind::NodeProject
        | BundleKind::RustCrate
        | BundleKind::PythonProject
        | BundleKind::XcodeProject
        | BundleKind::AndroidStudioProject => "Code/Projects/",
        BundleKind::JupyterNotebookSet => "Code/Notebooks/",
        BundleKind::PhotoBurst => "Photos/Bursts/",
        BundleKind::MusicAlbum => "Music/Albums/",
        BundleKind::DocumentSeries { .. } => "Documents/Series/",
        BundleKind::Generic => "Archives/",
    }
}

fn build_bundle_proposal(bundle: &DetectedBundle, output_root: &Path) -> Result<BundleProposal> {
    let leaf = bundle
        .root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("bundle")
        .to_string();
    let taxonomy = bundle_taxonomy(&bundle.kind);
    let target_parent = output_root.join(taxonomy.trim_end_matches('/'));
    let bundle_target_root = target_parent.join(&leaf);

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
            bundle_id: None, // stamped by BundleProposal::new
            classification_confidence: None,
            rename_mismatch_score: None,
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

    struct NullProgress;
    #[async_trait]
    impl ProgressReporter for NullProgress {
        async fn phase_started(&self, _p: Phase, _t: Option<u64>) {}
        async fn item_completed(&self, _p: Phase, _i: ProgressItem) {}
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

    #[tokio::test]
    async fn heuristic_resolves_rust_source() {
        let td = TempDir::new().unwrap();
        // Loose file, not bundled (no Cargo.toml at root).
        fs::write(td.path().join("helpers.rs"), b"fn main() {}").unwrap();

        let eb = BucketEmbeddings;
        let candidates = sample_candidates(&eb).await;
        let ex: Vec<Arc<dyn ContentExtractor>> = vec![Arc::new(PlainExtractor)];
        let out = run_scan(
            td.path(),
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(p.proposed_path.to_string_lossy().contains("Code"));
        assert!(p.reasoning.contains("tier1"));
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
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();
        assert_eq!(out.bundles.len(), 1);
        let bundle = &out.bundles[0];
        assert_eq!(bundle.kind, BundleKind::RustCrate);
        assert_eq!(bundle.members.len(), 2);
        // Bundles bypass per-file classification.
        assert!(out.proposals.is_empty());
    }

    #[tokio::test]
    async fn temporal_category_appends_year_subdir() {
        let td = TempDir::new().unwrap();
        fs::write(td.path().join("2024_tax_return.pdf"), b"form 1040 taxes").unwrap();

        // Use synthetic tax candidate that matches the file content by bucket.
        // .pdf extension isn't in heuristics, so this routes through Tier 2.
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
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
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
            td.path(),
            &[],
            &eb,
            &MultimodalContext::default(),
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
            td.path(),
            &candidates,
            &eb,
            &MultimodalContext::default(),
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
        };
        let multimodal = MultimodalContext {
            image: Some(img_ctx),
            audio: None,
        };

        let eb = BucketEmbeddings;
        // Empty text candidates so we know any proposal came from the image
        // path. Tier 1 will fire (PNG → Photos/) — verify the proposal is
        // produced even with that path; the modality routing kicks in only
        // when Tier 1 misses or the heuristic threshold isn't cleared.
        let cfg = ClassifierConfig {
            heuristic_threshold: 0.99, // force Tier 1 to miss
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
            td.path(),
            &[],
            &eb,
            &multimodal,
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

    #[test]
    fn file_modality_classifies_image_extension() {
        let m = file_modality(Path::new("/tmp/photo.HEIC"), None);
        assert!(matches!(m, FileModality::Image));
    }

    #[test]
    fn file_modality_classifies_audio_mime() {
        let m = file_modality(Path::new("/tmp/clip"), Some("audio/mpeg"));
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
    fn bundle_taxonomy_maps_each_kind() {
        assert_eq!(bundle_taxonomy(&BundleKind::RustCrate), "Code/Projects/");
        assert_eq!(bundle_taxonomy(&BundleKind::NodeProject), "Code/Projects/");
        assert_eq!(
            bundle_taxonomy(&BundleKind::JupyterNotebookSet),
            "Code/Notebooks/",
        );
        assert_eq!(bundle_taxonomy(&BundleKind::PhotoBurst), "Photos/Bursts/");
        assert_eq!(bundle_taxonomy(&BundleKind::MusicAlbum), "Music/Albums/");
        assert_eq!(
            bundle_taxonomy(&BundleKind::DocumentSeries {
                pattern: "x".into(),
            }),
            "Documents/Series/",
        );
        assert_eq!(bundle_taxonomy(&BundleKind::Generic), "Archives/");
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
}
