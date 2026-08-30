//! Pure domain types. No I/O, no async, no dependencies on other tidyup crates.
//!
//! These types form the contract between every other crate in the workspace.
//! Keep minimal and stable — a breaking change here ripples to every caller.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod backup;
pub mod bundle;
pub mod change;
pub mod file;
pub mod migration;
pub mod run;

pub use backup::{BackupRecord, BackupStatus, RestorePrecheck};
pub use bundle::{BundleError, BundleKind, BundleProposal};
pub use change::{ChangeProposal, ChangeStatus, ChangeType, ParseError};
pub use file::{
    ContentHash, FileId, FileProcessingRecord, FileProcessingRole, FileProcessingStage,
    FileProcessingState, IndexedFile, SemanticArtifact,
};
pub use migration::{
    Calibration, Candidate, ClassificationResult, ClassifierConfig, DatePattern, FolderMetadata,
    FolderNode, FolderProfile, OrganizationType, ProfileCache, RenameConfig, ScanDiff,
    ScoreBreakdown, ScoreWeights, TargetScan, Tier,
};
pub use run::{
    CapabilityEntry, CapabilityKind, CapabilityManifest, CapabilityStatus, RunMode, RunRecord,
    RunState,
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

/// Build the semantic text embedded to classify a file, from its extracted
/// body and filename.
///
/// This is the canonical construction used by the standalone embedding
/// classifier and its offline evaluation harness. The runtime scan pipeline
/// keeps name and body as separately weighted evidence channels. The model
/// token-truncates internally, so no character cap is applied here.
#[must_use]
pub fn classification_query(body: &str, filename: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        filename.to_string()
    } else if filename.is_empty() {
        body.to_string()
    } else {
        format!("{filename} {body}")
    }
}

/// User's decision on a single proposal during review.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approve(Uuid),
    Reject(Uuid),
    Override {
        proposal_id: Uuid,
        new_target: PathBuf,
    },
}

#[cfg(test)]
mod tests {
    use super::classification_query;

    #[test]
    fn query_prepends_filename_to_body() {
        assert_eq!(
            classification_query("tax return", "scan.pdf"),
            "scan.pdf tax return"
        );
    }

    #[test]
    fn query_falls_back_to_filename_when_body_empty() {
        assert_eq!(classification_query("", "photo.jpg"), "photo.jpg");
        assert_eq!(classification_query("   ", "photo.jpg"), "photo.jpg");
    }

    #[test]
    fn query_is_body_when_filename_empty() {
        assert_eq!(classification_query("body text", ""), "body text");
    }

    #[test]
    fn query_trims_body() {
        assert_eq!(classification_query("  hi  ", "a.txt"), "a.txt hi");
    }
}
