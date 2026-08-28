//! Migration-mode types: target-tree scans, folder profiles, classification results,
//! and the migration plan/run structures.
//!
//! These types drive the migration pipeline (moving files from a source directory into
//! an existing target hierarchy), distinct from the scan-mode pipeline that classifies
//! against a fixed taxonomy.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Target structure scanning types
// ---------------------------------------------------------------------------

/// A single directory node in the target tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderNode {
    /// Absolute path to this directory.
    pub path: PathBuf,
    /// Folder name (final path component).
    pub name: String,
    /// Full path segments from target root, e.g., `["House Stuff", "Mortgage"]`.
    pub path_segments: Vec<String>,
    /// Depth from target root (root children = 1).
    pub depth: u32,
    /// Direct child folder paths.
    pub children: Vec<PathBuf>,
    /// Sibling folder names (other children of this node's parent).
    pub sibling_names: Vec<String>,
    /// Metadata snapshot at scan time.
    pub metadata: FolderMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderMetadata {
    /// Number of direct files (not recursive).
    pub file_count: u32,
    /// Number of files recursively.
    pub recursive_file_count: u32,
    /// Frequency map of file extensions, e.g., `{".pdf": 12, ".jpg": 45}`.
    pub extension_counts: HashMap<String, u32>,
    /// Dominant extensions (top 3 by count).
    pub dominant_extensions: Vec<String>,
    /// Date range of files (earliest modified, latest modified).
    pub date_range: Option<(SystemTime, SystemTime)>,
    /// Mean file size in bytes.
    pub avg_file_size: u64,
    /// Whether this folder has subdirectories.
    pub has_children: bool,
    /// Hash of direct file listing for change detection.
    pub content_hash: String,
    /// Timestamp of this scan.
    pub scanned_at: SystemTime,
}

/// Top-level scan result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetScan {
    pub root: PathBuf,
    pub nodes: HashMap<PathBuf, FolderNode>,
    /// Folders with no child directories (classification targets).
    pub leaf_folders: Vec<PathBuf>,
    pub scan_timestamp: SystemTime,
}

/// Result of comparing two scans for incremental updates.
#[derive(Debug, Clone)]
pub struct ScanDiff {
    /// Folders present in current scan but not in previous.
    pub added: Vec<PathBuf>,
    /// Folders present in previous scan but not in current.
    pub removed: Vec<PathBuf>,
    /// Folders present in both but with different `content_hash`.
    pub modified: Vec<PathBuf>,
    /// Folders present in both with identical `content_hash`.
    pub unchanged: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// Folder profiling types
// ---------------------------------------------------------------------------

/// Semantic profile for a target folder.
///
/// # Latent-space isolation
///
/// The three centroid-style fields live in **disjoint** latent spaces and must
/// never be cosine-compared against each other:
/// - [`name_embedding`](Self::name_embedding) and
///   [`content_centroid`](Self::content_centroid) — text space (`bge-small`).
/// - [`image_centroid`](Self::image_centroid) — cross-modal image space
///   (`SigLIP`); compare only against `SigLIP` image embeddings.
/// - [`audio_centroid`](Self::audio_centroid) — cross-modal audio space
///   (`CLAP`); compare only against `CLAP` audio embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderProfile {
    /// Absolute path to the folder.
    pub path: PathBuf,
    /// Embedding of synthesized natural-language description (text space).
    pub name_embedding: Vec<f32>,
    /// Mean text embedding of sampled document bodies within this folder (text
    /// space, same backend as `name_embedding`). `None` when the folder holds no
    /// extractable text documents or the profiler was run without extractors.
    pub content_centroid: Option<Vec<f32>>,
    /// Number of files that contributed to the content centroid.
    pub centroid_sample_count: u32,
    /// Mean `SigLIP` image embedding of sampled images directly in this folder.
    /// `None` when no image backend was available at profile time or the folder
    /// holds no images. Compare only against `SigLIP` image embeddings.
    #[serde(default)]
    pub image_centroid: Option<Vec<f32>>,
    /// Number of images that contributed to [`image_centroid`](Self::image_centroid).
    #[serde(default)]
    pub image_centroid_sample_count: u32,
    /// Mean `CLAP` audio embedding of sampled audio directly in this folder.
    /// `None` when no audio backend was available at profile time or the folder
    /// holds no audio. Compare only against `CLAP` audio embeddings.
    #[serde(default)]
    pub audio_centroid: Option<Vec<f32>>,
    /// Number of audio files that contributed to [`audio_centroid`](Self::audio_centroid).
    #[serde(default)]
    pub audio_centroid_sample_count: u32,
    /// Structural metadata.
    pub metadata: FolderMetadata,
    /// Detected organizational dimension.
    pub organization_type: OrganizationType,
    /// Confidence in the profile (0.0-1.0).
    pub profile_confidence: f32,
    /// Timestamp of last profile update.
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrganizationType {
    /// Folder organized by topic/content type.
    Semantic,
    /// Folder organized by date.
    DateBased { pattern: DatePattern },
    /// Folder organized by project or entity name.
    ProjectBased,
    /// Folder organized by workflow status.
    StatusBased,
    /// Cannot determine; treat as semantic.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatePattern {
    Year,
    Quarter,
    Month,
    Week,
}

/// All folder profiles for a target root, built per migration run.
///
/// Serializable for a future on-disk cache, but not currently persisted or
/// reloaded — every run rebuilds it in full (see `CLAUDE.md`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCache {
    /// Target root this cache belongs to.
    pub target_root: PathBuf,
    /// Embedding model identifier used to generate these profiles.
    pub model_id: String,
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// All folder profiles, keyed by absolute path.
    pub profiles: HashMap<PathBuf, FolderProfile>,
    /// Scan state at time of last full profile build.
    pub last_scan: TargetScan,
    /// Cache creation timestamp.
    pub created_at: SystemTime,
    /// Timestamp of the most recent (currently always full) profile build.
    /// Reserved for incremental rebuilds, which are not yet wired.
    pub last_updated: SystemTime,
}

// ---------------------------------------------------------------------------
// Classification result types
// ---------------------------------------------------------------------------

/// Result of classifying a single source file against target profiles.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Source file being classified.
    pub source_file: PathBuf,
    /// Ordered list of candidate destinations, best first.
    pub candidates: Vec<Candidate>,
    /// Which tier produced the final classification.
    pub resolved_at: Tier,
    /// Whether this classification needs user review.
    pub needs_review: bool,
    /// Optional new filename (only if Tier 3 was invoked for renaming).
    pub suggested_rename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    /// Target folder path.
    pub folder: PathBuf,
    /// Composite confidence score `[0.0, 1.0]`.
    pub score: f32,
    /// Breakdown of how the score was computed.
    pub score_breakdown: ScoreBreakdown,
}

#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    /// Similarity to folder name embedding.
    pub name_similarity: f32,
    /// Similarity to folder content centroid.
    pub centroid_similarity: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    Embedding,
    Llm,
}

// ---------------------------------------------------------------------------
// Classifier config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    /// Semantic embedding auto-classify threshold.
    pub embedding_threshold: f32,
    /// Semantic ambiguity gap threshold.
    pub ambiguity_gap: f32,
    /// Whether to invoke Tier 3 (LLM) for ambiguous files. Defaults to `false`
    /// (privacy-preserving): activation is materialised from the layered config
    /// only under the three-gate model (cargo feature + config bool +
    /// per-invocation flag), never from this default alone.
    pub enable_llm_fallback: bool,
    /// Composite score weights.
    pub weights: ScoreWeights,
    /// Rename proposal thresholds.
    pub rename: RenameConfig,
    /// Confidence calibration applied to reported classification confidence.
    /// Default [`Calibration::Identity`] (raw scores) — see [`Calibration`].
    pub calibration: Calibration,
}

#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub name: f32,
    pub centroid: f32,
}

/// Thresholds gating rename proposals. Both signals must clear their threshold before a
/// rename is surfaced to review. Renames never auto-apply, even under `--yes`.
///
/// - `min_classification_confidence`: lower bound on Tier-2 classification confidence.
/// - `min_mismatch_score`: lower bound on `1.0 - cosine(embed(filename), content_embedding)`.
#[derive(Debug, Clone)]
pub struct RenameConfig {
    pub min_classification_confidence: f32,
    pub min_mismatch_score: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            embedding_threshold: 0.35,
            ambiguity_gap: 0.05,
            // Privacy default: Tier 3 stays off unless the layered config +
            // three-gate activation explicitly turns it on. A dead
            // `enable_llm_renaming` field used to default true here — it was
            // never read and contradicted the extractive-only rename invariant.
            enable_llm_fallback: false,
            weights: ScoreWeights::default(),
            rename: RenameConfig::default(),
            calibration: Calibration::default(),
        }
    }
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            // Preserve the former 25:55 ratio while normalizing the two live
            // semantic signals onto the full confidence scale.
            name: 0.3125,
            centroid: 0.6875,
        }
    }
}

impl Default for RenameConfig {
    fn default() -> Self {
        Self {
            min_classification_confidence: 0.85,
            min_mismatch_score: 0.60,
        }
    }
}

/// Maps a raw classifier score to a calibrated confidence in `[0, 1]`.
///
/// `Identity` is the v0.1 default — confidence is reported as the raw
/// weighted-cosine, uncalibrated (a true no-op). `Platt { a, b }` applies
/// logistic (Platt) scaling `sigmoid(a·raw + b)`, fit offline against a labeled
/// corpus (`cargo xtask eval --calibrate`). Calibration is opt-in: the shipped
/// default stays `Identity` until a corpus-fit parameter set exists, so the
/// "raw weighted-cosine, not calibrated" v0.1 promise holds by default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Calibration {
    /// No calibration — report the raw score unchanged.
    #[default]
    Identity,
    /// Logistic (Platt) scaling: `sigmoid(a · raw + b)`.
    Platt { a: f64, b: f64 },
}

impl Calibration {
    /// Map a raw score to a calibrated confidence. `Identity` returns the input
    /// unchanged; `Platt` applies logistic scaling (output always in `[0, 1]`).
    #[must_use]
    // The logistic result is bounded to `[0, 1]`; narrowing it to the public
    // score type cannot overflow or change its sign.
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    pub fn calibrate(&self, raw: f32) -> f32 {
        match *self {
            Self::Identity => raw,
            Self::Platt { a, b } => {
                let z = a.mul_add(f64::from(raw), b);
                (1.0 / (1.0 + (-z).exp())) as f32
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn organization_type_serde_roundtrip() {
        let types = vec![
            OrganizationType::Semantic,
            OrganizationType::DateBased {
                pattern: DatePattern::Year,
            },
            OrganizationType::ProjectBased,
            OrganizationType::StatusBased,
            OrganizationType::Unknown,
        ];
        for ot in types {
            let json = serde_json::to_string(&ot).unwrap();
            let back: OrganizationType = serde_json::from_str(&json).unwrap();
            assert_eq!(ot, back);
        }
    }

    #[test]
    fn tier_serde_roundtrip() {
        for t in [Tier::Embedding, Tier::Llm] {
            let json = serde_json::to_string(&t).unwrap();
            let back: Tier = serde_json::from_str(&json).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn default_classifier_config() {
        let config = ClassifierConfig::default();
        assert!((config.embedding_threshold - 0.35).abs() < f32::EPSILON);
        let w = &config.weights;
        let total = w.name + w.centroid;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn default_rename_config_matches_spec() {
        let r = RenameConfig::default();
        assert!((r.min_classification_confidence - 0.85).abs() < f32::EPSILON);
        assert!((r.min_mismatch_score - 0.60).abs() < f32::EPSILON);
    }

    #[test]
    fn rename_config_thresholds_in_unit_range() {
        let r = RenameConfig::default();
        assert!((0.0..=1.0).contains(&r.min_classification_confidence));
        assert!((0.0..=1.0).contains(&r.min_mismatch_score));
    }
}
