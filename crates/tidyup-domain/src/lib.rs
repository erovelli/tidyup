//! Pure domain types. No I/O, no async, no dependencies on other tidyup crates.
//!
//! These types form the contract between every other crate in the workspace.
//! Keep minimal and stable.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for a file tracked by tidyup. Derived from content hash + path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub Uuid);

impl FileId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for FileId {
    fn default() -> Self {
        Self::new()
    }
}

/// Confidence score in `[0.0, 1.0]` attached to classifier output.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(pub f32);

impl Confidence {
    pub const LOW: Self = Self(0.33);
    pub const MEDIUM: Self = Self(0.66);
    pub const HIGH: Self = Self(0.85);
}

/// A file observed on disk with extracted metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    pub id: FileId,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub content_hash: String,
    pub mime: Option<String>,
    pub modified_at: DateTime<Utc>,
    pub extracted_text: Option<String>,
}

/// A proposed change to be reviewed by the user before execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeProposal {
    pub id: Uuid,
    pub file_id: FileId,
    pub source_path: PathBuf,
    pub target_path: PathBuf,
    pub rationale: String,
    pub confidence: Confidence,
    pub tier: ClassificationTier,
}

/// Which tier produced a proposal. Useful for debugging and UI display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationTier {
    Heuristics,
    Embeddings,
    Llm,
    Manual,
}

/// User's decision on a single proposal during review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approve(Uuid),
    Reject(Uuid),
    Override {
        proposal_id: Uuid,
        new_target: PathBuf,
    },
}

/// Semantic profile of a target folder, learned during target-tree scanning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderProfile {
    pub path: PathBuf,
    pub name_embedding: Vec<f32>,
    pub content_centroid: Option<Vec<f32>>,
    pub sample_count: usize,
    pub org_type: OrgType,
}

/// How a folder is organised. Detected during scanning, influences classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrgType {
    Semantic,
    DateBased,
    ProjectBased,
    StatusBased,
    Unknown,
}

/// Backup record for rollback support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecord {
    pub id: Uuid,
    pub file_id: FileId,
    pub original_path: PathBuf,
    pub shelved_path: PathBuf,
    pub created_at: DateTime<Utc>,
}

/// Phases emitted to frontends during a run. Enables a single progress contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Indexing,
    Extracting,
    ProfilingTarget,
    Classifying,
    AwaitingReview,
    Applying,
    Rollback,
}

#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),
}
