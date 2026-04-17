//! Pure domain types. No I/O, no async, no dependencies on other tidyup crates.
//!
//! These types form the contract between every other crate in the workspace.
//! Keep minimal and stable — a breaking change here ripples to every caller.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod backup;
pub mod change;
pub mod file;
pub mod migration;

pub use backup::{BackupRecord, BackupStatus};
pub use change::{ChangeProposal, ChangeStatus, ChangeType, ParseError};
pub use file::{ContentHash, FileId, IndexedFile};
pub use migration::{
    Candidate, ClassificationResult, ClassifierConfig, DatePattern, ExecutedMove, FolderMetadata,
    FolderNode, FolderProfile, MigrationPlan, MigrationRun, MoveStatus, OrganizationType,
    PlanStats, ProfileCache, ProposedMove, RunStatus, ScanDiff, ScoreBreakdown, ScoreWeights,
    TargetScan, Tier,
};

/// Phases emitted to frontends during a run. Drives the single progress contract
/// shared between CLI (`indicatif`) and UI (Dioxus signals).
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

/// User's decision on a single proposal during review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approve(Uuid),
    Reject(Uuid),
    Override { proposal_id: Uuid, new_target: PathBuf },
}
