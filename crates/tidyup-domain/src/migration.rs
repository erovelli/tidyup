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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderProfile {
    /// Absolute path to the folder.
    pub path: PathBuf,
    /// Embedding of synthesized natural-language description.
    pub name_embedding: Vec<f32>,
    /// Mean embedding of sampled file contents within this folder.
    pub content_centroid: Option<Vec<f32>>,
    /// Number of files that contributed to the content centroid.
    pub centroid_sample_count: u32,
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

/// On-disk cache of all folder profiles for a target root.
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
    /// Last incremental update timestamp.
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
    /// Metadata compatibility score.
    pub metadata_score: f32,
    /// Hierarchical coherence adjustment.
    pub hierarchy_adjustment: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    Heuristic,
    Embedding,
    Llm,
}

// ---------------------------------------------------------------------------
// Migration plan types
// ---------------------------------------------------------------------------

/// Complete set of proposed file moves from source to target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Unique run identifier.
    pub run_id: String,
    /// Source root.
    pub source_root: PathBuf,
    /// Target root.
    pub target_root: PathBuf,
    /// Timestamp of plan creation.
    pub created_at: SystemTime,
    /// All proposed moves.
    pub moves: Vec<ProposedMove>,
    /// Files that could not be classified.
    pub unclassified: Vec<PathBuf>,
    /// Summary statistics.
    pub stats: PlanStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedMove {
    /// Source file path.
    pub source: PathBuf,
    /// Proposed destination path (full path including filename).
    pub destination: PathBuf,
    /// New filename, if renaming was applied.
    pub renamed_to: Option<String>,
    /// Classification tier and confidence.
    pub tier: Tier,
    pub confidence: f32,
    /// Runner-up destination if any.
    pub runner_up: Option<(PathBuf, f32)>,
    /// User approval status.
    pub status: MoveStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveStatus {
    Pending,
    Approved,
    Rejected,
    ManualOverride { new_destination: PathBuf },
    Completed,
    Failed { error: String },
    RolledBack,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlanStats {
    pub total_files: u32,
    pub tier1_resolved: u32,
    pub tier2_resolved: u32,
    pub tier3_resolved: u32,
    pub unclassified: u32,
    pub needs_review: u32,
    pub avg_confidence: f32,
}

// ---------------------------------------------------------------------------
// Migration execution types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRun {
    pub run_id: String,
    pub source_root: PathBuf,
    pub target_root: PathBuf,
    pub started_at: SystemTime,
    pub completed_at: Option<SystemTime>,
    pub status: RunStatus,
    pub executed_moves: Vec<ExecutedMove>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutedMove {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub file_size: u64,
    pub moved_at: SystemTime,
    pub rolled_back: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    InProgress,
    Completed,
    PartiallyCompleted { completed: u32, failed: u32 },
    RolledBack,
    PartiallyRolledBack,
}

// ---------------------------------------------------------------------------
// Classifier config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    /// Tier 1 auto-classify threshold.
    pub heuristic_threshold: f32,
    /// Tier 2 auto-classify threshold.
    pub embedding_threshold: f32,
    /// Tier 2 ambiguity gap threshold.
    pub ambiguity_gap: f32,
    /// Whether to invoke Tier 3 for ambiguous files.
    pub enable_llm_fallback: bool,
    /// Whether to invoke Tier 3 for vague filenames.
    pub enable_llm_renaming: bool,
    /// Composite score weights.
    pub weights: ScoreWeights,
    /// Rename proposal thresholds.
    pub rename: RenameConfig,
}

#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub name: f32,
    pub centroid: f32,
    pub metadata: f32,
    pub hierarchy: f32,
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
            heuristic_threshold: 0.60,
            embedding_threshold: 0.35,
            ambiguity_gap: 0.05,
            enable_llm_fallback: true,
            enable_llm_renaming: true,
            weights: ScoreWeights::default(),
            rename: RenameConfig::default(),
        }
    }
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            name: 0.25,
            centroid: 0.55,
            metadata: 0.10,
            hierarchy: 0.10,
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
    fn move_status_serde_roundtrip() {
        let statuses = vec![
            MoveStatus::Pending,
            MoveStatus::Approved,
            MoveStatus::Rejected,
            MoveStatus::Completed,
            MoveStatus::Failed {
                error: "disk full".to_string(),
            },
            MoveStatus::RolledBack,
            MoveStatus::ManualOverride {
                new_destination: PathBuf::from("/new/path"),
            },
        ];
        for s in statuses {
            let json = serde_json::to_string(&s).unwrap();
            let back: MoveStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn tier_serde_roundtrip() {
        for t in [Tier::Heuristic, Tier::Embedding, Tier::Llm] {
            let json = serde_json::to_string(&t).unwrap();
            let back: Tier = serde_json::from_str(&json).unwrap();
            assert_eq!(t, back);
        }
    }

    #[test]
    fn default_classifier_config() {
        let config = ClassifierConfig::default();
        assert!((config.heuristic_threshold - 0.60).abs() < f32::EPSILON);
        assert!((config.embedding_threshold - 0.35).abs() < f32::EPSILON);
        let w = &config.weights;
        let total = w.name + w.centroid + w.metadata + w.hierarchy;
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
