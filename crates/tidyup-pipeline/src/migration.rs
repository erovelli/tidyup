//! Migration-mode pipeline — classify a source tree against an existing
//! destination hierarchy described by a [`ProfileCache`].
//!
//! Contrast with [`crate::scan`]: scan-mode routes into a fixed taxonomy;
//! migration-mode routes into whatever structure the user has already built.
//! Tier 1 still fires (extension / MIME / marker filenames), but Tier 2
//! composite scoring is the primary signal, combining similarity to each
//! folder's `name_embedding` and `content_centroid`, plus a metadata score
//! and a hierarchy adjustment.
//!
//! # Composite score
//!
//! Per [`ScoreWeights`] (defaults `0.25 name + 0.55 centroid + 0.10 metadata +
//! 0.10 hierarchy`). When a folder's `content_centroid` is `None` (the v0.1
//! default), the centroid weight is redistributed to `name` so the composite
//! stays in `[0, 1]`.
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
use tidyup_core::inference::{EmbeddingBackend, TextBackend};
use tidyup_domain::change::{ChangeProposal, ChangeStatus, ChangeType};
use tidyup_domain::migration::{
    Candidate, ClassificationResult, ScoreBreakdown, ScoreWeights, Tier,
};
use tidyup_domain::{
    BundleKind, BundleProposal, ClassifierConfig, FolderProfile, Phase, ProfileCache,
};
use uuid::Uuid;

use crate::heuristics::{self, HeuristicMatch};
use crate::naming::{propose_rename, RenameProposal};
use crate::scanner::{self, DetectedBundle};
use crate::yake;

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
    /// no heuristic fallback).
    pub unclassified: Vec<PathBuf>,
}

/// Drive the migration cascade end-to-end.
///
/// `text_backend` is the optional Tier 3 LLM. Pass `None` and the cascade
/// stops at Tier 2; pass `Some` and low-confidence Tier 2 verdicts get a
/// chance to be re-ranked through an LLM-cleaned query against the same
/// folder profiles.
///
/// # Errors
/// Propagates source-read and embedding-backend failures. Per-file
/// extraction / classification failures are logged via `progress.message`
/// and surface through [`MigrationOutcome::unclassified`].
#[allow(clippy::too_many_arguments)]
pub async fn run_migration(
    source_root: &Path,
    profiles: &ProfileCache,
    embeddings: &dyn EmbeddingBackend,
    text_backend: Option<&dyn TextBackend>,
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClassifierConfig,
    progress: &dyn ProgressReporter,
) -> Result<MigrationOutcome> {
    progress.phase_started(Phase::Indexing, None).await;
    let tree = scanner::scan(source_root);
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

    #[allow(clippy::cast_possible_truncation)]
    let total = tree.loose_files.len() as u64;
    progress
        .phase_started(Phase::Classifying, Some(total))
        .await;

    for (idx, path) in tree.loose_files.iter().enumerate() {
        match classify_file(path, profiles, embeddings, text_backend, extractors, config).await {
            Ok(Some(verdict)) => {
                let proposal = build_proposal(path, &verdict);
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

    // Tier 1.
    //
    // Heuristics give us a class label (e.g. `"Code/"`), but in migration
    // mode we need an actual folder path. Use the heuristic's taxonomy
    // string as a soft routing signal: pick the profile leaf whose path
    // *contains* that label case-insensitively. If none matches, fall
    // through to Tier 2.
    if let Some(hit) = heuristics::classify(path, effective_mime.as_deref()) {
        if hit.confidence >= config.heuristic_threshold {
            if let Some(folder) = route_heuristic(&hit, profiles) {
                let text = extracted.as_ref().and_then(|e| e.text.as_deref());
                let metadata_json = extracted
                    .as_ref()
                    .map_or(serde_json::Value::Null, |e| e.metadata.clone());
                let keywords = text
                    .map(|t| yake::extract_keywords(t, 8))
                    .unwrap_or_default();
                let year = text.and_then(|t| find_year(&t[..t.len().min(1000)]));
                let rename = gate_rename(
                    path,
                    &metadata_json,
                    &keywords,
                    year,
                    hit.confidence,
                    embeddings,
                    text,
                    &filename,
                    config,
                )
                .await?;
                return Ok(Some(Verdict {
                    result: ClassificationResult {
                        source_file: path.to_path_buf(),
                        candidates: vec![Candidate {
                            folder: folder.clone(),
                            score: hit.confidence,
                            score_breakdown: ScoreBreakdown {
                                name_similarity: 0.0,
                                centroid_similarity: None,
                                metadata_score: 0.0,
                                hierarchy_adjustment: 0.0,
                            },
                        }],
                        resolved_at: Tier::Heuristic,
                        needs_review: false,
                        suggested_rename: match &rename.proposal {
                            RenameProposal::Rename { name, .. } => Some(name.clone()),
                            RenameProposal::Keep => None,
                        },
                    },
                    rename: rename.proposal,
                    destination_folder: folder,
                    confidence: hit.confidence,
                    reasoning: format!("tier1 heuristic: {}", hit.reason),
                    classification_confidence: Some(hit.confidence),
                    rename_mismatch_score: rename.mismatch_score,
                }));
            }
        }
    }

    // Tier 2 — composite scoring against all leaf profiles.
    let Some(text) = extracted.as_ref().and_then(|e| e.text.as_deref()) else {
        // No content → fall back to the heuristic if any, flagged for review.
        if let Some(hit) = heuristics::classify(path, effective_mime.as_deref()) {
            if let Some(folder) = route_heuristic(&hit, profiles) {
                return Ok(Some(weak_heuristic(&hit, folder, path)));
            }
        }
        return Ok(None);
    };

    let content_embedding = embeddings.embed_text(text).await?;

    let ranked = rank_profiles(&content_embedding, profiles, path, &config.weights);
    if ranked.is_empty() {
        return Ok(None);
    }

    let (tier2_folder, tier2_score, tier2_breakdown) = ranked[0].clone();
    let tier2_gap = if ranked.len() >= 2 {
        tier2_score - ranked[1].1
    } else {
        tier2_score
    };
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
                text,
                &filename,
                &config.weights,
            )
            .await
            {
                Ok(Some((rerank, model_id))) if rerank[0].1 > chosen_score => {
                    let (f, s, b) = rerank[0].clone();
                    let g = if rerank.len() >= 2 {
                        s - rerank[1].1
                    } else {
                        s
                    };
                    chosen_folder = f;
                    chosen_score = s;
                    chosen_gap = g;
                    chosen_breakdown = b;
                    chosen_ranked = rerank;
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

    let needs_review =
        chosen_score < config.embedding_threshold || chosen_gap < config.ambiguity_gap;

    let metadata_json = extracted
        .as_ref()
        .map_or(serde_json::Value::Null, |e| e.metadata.clone());
    let keywords = yake::extract_keywords(text, 8);
    let year = find_year(&text[..text.len().min(1000)]);
    let rename = gate_rename(
        path,
        &metadata_json,
        &keywords,
        year,
        chosen_score,
        embeddings,
        Some(text),
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
            "tier2 composite: top={chosen_score:.3} gap={chosen_gap:.3} name={:.3} centroid={} \
             metadata={:.3} hierarchy={:.3}",
            chosen_breakdown.name_similarity,
            chosen_breakdown
                .centroid_similarity
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}")),
            chosen_breakdown.metadata_score,
            chosen_breakdown.hierarchy_adjustment,
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
        classification_confidence: Some(chosen_score),
        rename_mismatch_score: rename.mismatch_score,
    }))
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
    let classification = text_backend.classify_text(text, filename).await?;
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

fn weak_heuristic(hit: &HeuristicMatch, folder: PathBuf, path: &Path) -> Verdict {
    Verdict {
        result: ClassificationResult {
            source_file: path.to_path_buf(),
            candidates: vec![Candidate {
                folder: folder.clone(),
                score: hit.confidence,
                score_breakdown: ScoreBreakdown {
                    name_similarity: 0.0,
                    centroid_similarity: None,
                    metadata_score: 0.0,
                    hierarchy_adjustment: 0.0,
                },
            }],
            resolved_at: Tier::Heuristic,
            needs_review: true,
            suggested_rename: None,
        },
        rename: RenameProposal::Keep,
        destination_folder: folder,
        confidence: hit.confidence,
        reasoning: format!("tier1 heuristic (below threshold): {}", hit.reason),
        classification_confidence: Some(hit.confidence),
        rename_mismatch_score: None,
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
    source_path: &Path,
) -> ScoreBreakdown {
    let name_similarity = cosine(content_embedding, &profile.name_embedding).max(0.0);
    let centroid_similarity = profile
        .content_centroid
        .as_deref()
        .map(|c| cosine(content_embedding, c).max(0.0));
    let metadata_score = metadata_compatibility(source_path, profile);
    ScoreBreakdown {
        name_similarity,
        centroid_similarity,
        metadata_score,
        hierarchy_adjustment: 0.0,
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
    let metadata_term = w.metadata.mul_add(b.metadata_score, centroid_term);
    let hierarchy_term = w.hierarchy.mul_add(b.hierarchy_adjustment, metadata_term);
    name_w.mul_add(b.name_similarity, hierarchy_term)
}

fn metadata_compatibility(source_path: &Path, profile: &FolderProfile) -> f32 {
    let Some(ext) = source_path.extension().and_then(|s| s.to_str()) else {
        return 0.0;
    };
    let dotted = format!(".{}", ext.to_ascii_lowercase());
    if profile.metadata.dominant_extensions.contains(&dotted) {
        1.0
    } else if profile.metadata.extension_counts.contains_key(&dotted) {
        0.5
    } else {
        0.0
    }
}

fn route_heuristic(hit: &HeuristicMatch, profiles: &ProfileCache) -> Option<PathBuf> {
    let label = hit
        .taxonomy_path
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(hit.taxonomy_path)
        .to_ascii_lowercase();
    if label.is_empty() {
        return None;
    }
    profiles
        .last_scan
        .leaf_folders
        .iter()
        .find(|p| {
            p.components()
                .next_back()
                .and_then(|c| c.as_os_str().to_str())
                .is_some_and(|name| name.to_ascii_lowercase().contains(&label))
        })
        .cloned()
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
    }
}

/// Bundle placement: embed the bundle's leaf name and pick the top-scoring
/// profile by `name_embedding` cosine. Fall back to a taxonomy default when
/// the profile cache is empty.
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
        .unwrap_or_else(|| {
            profiles
                .target_root
                .join(default_bundle_taxonomy(&bundle.kind))
        });

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
    // Build a single description string from the bundle kind + root name.
    let query = format!("{} {}", default_bundle_taxonomy(&bundle.kind), leaf_name);
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

const fn default_bundle_taxonomy(kind: &BundleKind) -> &'static str {
    match kind {
        BundleKind::GitRepository
        | BundleKind::NodeProject
        | BundleKind::RustCrate
        | BundleKind::PythonProject
        | BundleKind::XcodeProject
        | BundleKind::AndroidStudioProject => "Code/Projects",
        BundleKind::JupyterNotebookSet => "Code/Notebooks",
        BundleKind::PhotoBurst => "Photos/Bursts",
        BundleKind::MusicAlbum => "Music/Albums",
        BundleKind::DocumentSeries { .. } => "Documents/Series",
        BundleKind::Generic => "Archives",
    }
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

        let out = run_migration(src.path(), &profiles, &eb, None, &ex, &cfg, &NullProgress)
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
    async fn heuristic_routes_rust_source_to_code_folder() {
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
            &ex,
            &ClassifierConfig::default(),
            &NullProgress,
        )
        .await
        .unwrap();

        assert_eq!(out.proposals.len(), 1);
        let p = &out.proposals[0];
        assert!(p.proposed_path.starts_with(tgt.path().join("Code")));
        assert!(p.reasoning.contains("tier1"));
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
            unreachable!()
        }
        async fn classify_video(
            &self,
            _f: &str,
            _c: &[String],
        ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
            unreachable!()
        }
        async fn classify_image_description(
            &self,
            _f: &str,
            _d: &str,
        ) -> tidyup_core::Result<tidyup_core::inference::ContentClassification> {
            unreachable!()
        }
        async fn complete(
            &self,
            _p: &str,
            _o: &tidyup_core::inference::GenerationOptions,
        ) -> tidyup_core::Result<String> {
            unreachable!()
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

    #[test]
    fn composite_redistributes_centroid_weight_when_absent() {
        let w = ScoreWeights::default();
        let with_centroid = ScoreBreakdown {
            name_similarity: 0.5,
            centroid_similarity: Some(0.5),
            metadata_score: 0.0,
            hierarchy_adjustment: 0.0,
        };
        let without_centroid = ScoreBreakdown {
            name_similarity: 0.5,
            centroid_similarity: None,
            metadata_score: 0.0,
            hierarchy_adjustment: 0.0,
        };
        let a = composite(&with_centroid, &w);
        let b = composite(&without_centroid, &w);
        // With centroid: 0.25*0.5 + 0.55*0.5 = 0.4
        // Without centroid: (0.25+0.55)*0.5 = 0.4
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn metadata_compatibility_boosts_dominant_extensions() {
        let eb_emb = vec![0.0_f32; 7];
        let profile = FolderProfile {
            path: PathBuf::from("/tgt/Code"),
            name_embedding: eb_emb,
            content_centroid: None,
            centroid_sample_count: 0,
            metadata: make_node(Path::new("/tgt/Code"), "Code", &[".rs"]).metadata,
            organization_type: OrganizationType::Semantic,
            profile_confidence: 1.0,
            last_updated: SystemTime::now(),
        };
        assert_eq!(metadata_compatibility(Path::new("main.rs"), &profile), 1.0,);
        // Non-dominant but present.
        assert_eq!(metadata_compatibility(Path::new("main.txt"), &profile), 0.0,);
    }

    #[test]
    fn route_heuristic_matches_by_label() {
        let eb = BucketEmbeddings;
        let rt = tokio::runtime::Runtime::new().unwrap();
        let tgt = TempDir::new().unwrap();
        let profiles = rt.block_on(sample_cache(tgt.path(), &eb));
        let hit = HeuristicMatch {
            taxonomy_path: "Code/",
            confidence: 0.9,
            reason: "rust source",
        };
        let routed = route_heuristic(&hit, &profiles).unwrap();
        assert_eq!(routed, tgt.path().join("Code"));
    }

    #[test]
    fn default_bundle_taxonomy_mapping() {
        assert_eq!(
            default_bundle_taxonomy(&BundleKind::RustCrate),
            "Code/Projects"
        );
        assert_eq!(
            default_bundle_taxonomy(&BundleKind::PhotoBurst),
            "Photos/Bursts"
        );
        assert_eq!(default_bundle_taxonomy(&BundleKind::Generic), "Archives");
    }
}
