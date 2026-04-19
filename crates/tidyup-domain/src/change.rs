//! Proposed and applied changes. Flows through the review pipeline.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::file::FileId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    Rename,
    Move,
    RenameAndMove,
}

impl ChangeType {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Rename => "Rename",
            Self::Move => "Move",
            Self::RenameAndMove => "RenameAndMove",
        }
    }

    /// Human-readable label for UI display.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Rename => "Rename",
            Self::Move => "Move",
            Self::RenameAndMove => "Rename + Move",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "Rename" => Ok(Self::Rename),
            "Move" => Ok(Self::Move),
            "RenameAndMove" => Ok(Self::RenameAndMove),
            other => Err(ParseError::UnknownChangeType(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeStatus {
    Pending,
    Approved,
    Rejected,
    Applied,
    Unshelved,
}

impl ChangeStatus {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Approved => "Approved",
            Self::Rejected => "Rejected",
            Self::Applied => "Applied",
            Self::Unshelved => "Unshelved",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "Pending" => Ok(Self::Pending),
            "Approved" => Ok(Self::Approved),
            "Rejected" => Ok(Self::Rejected),
            "Applied" => Ok(Self::Applied),
            "Unshelved" => Ok(Self::Unshelved),
            other => Err(ParseError::UnknownChangeStatus(other.to_string())),
        }
    }
}

/// A proposed rename/move for a single file. Never applied without a `ReviewDecision::Approve`.
///
/// Bundle membership: when `bundle_id` is `Some`, this proposal moves a single member of a
/// `BundleProposal` and is never approved, applied, or rolled back independently of its bundle.
/// Members always have `change_type == ChangeType::Move` — bundle members never receive rename
/// suggestions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangeProposal {
    pub id: Uuid,
    pub file_id: Option<FileId>,
    pub change_type: ChangeType,
    pub original_path: PathBuf,
    pub proposed_path: PathBuf,
    pub proposed_name: String,
    pub confidence: f32,
    pub reasoning: String,
    pub needs_review: bool,
    pub status: ChangeStatus,
    pub created_at: DateTime<Utc>,
    pub applied_at: Option<DateTime<Utc>>,
    /// FK to the owning `BundleProposal`. `None` for loose (non-bundle) proposals.
    #[serde(default)]
    pub bundle_id: Option<Uuid>,
    /// Tier-2 classification sub-score used (with `rename_mismatch_score`) to gate rename
    /// proposals. `None` when classification wasn't run (e.g. bundle members, heuristic-only).
    #[serde(default)]
    pub classification_confidence: Option<f32>,
    /// `1.0 - cosine(embed(filename_as_text), content_embedding)`. Drives rename proposals
    /// jointly with `classification_confidence`. `None` when no rename was considered.
    #[serde(default)]
    pub rename_mismatch_score: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unknown ChangeType: {0}")]
    UnknownChangeType(String),
    #[error("unknown ChangeStatus: {0}")]
    UnknownChangeStatus(String),
    #[error("unknown BackupStatus: {0}")]
    UnknownBackupStatus(String),
    #[error("unknown BundleKind: {0}")]
    UnknownBundleKind(String),
    #[error("unknown RunMode: {0}")]
    UnknownRunMode(String),
    #[error("unknown RunState: {0}")]
    UnknownRunState(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn change_type_roundtrip() {
        for ct in [
            ChangeType::Rename,
            ChangeType::Move,
            ChangeType::RenameAndMove,
        ] {
            let s = ct.as_str();
            let back = ChangeType::parse(s).unwrap();
            assert_eq!(ct, back);
        }
    }

    #[test]
    fn change_status_roundtrip() {
        for cs in [
            ChangeStatus::Pending,
            ChangeStatus::Approved,
            ChangeStatus::Rejected,
            ChangeStatus::Applied,
            ChangeStatus::Unshelved,
        ] {
            let s = cs.as_str();
            let back = ChangeStatus::parse(s).unwrap();
            assert_eq!(cs, back);
        }
    }

    #[test]
    fn proposal_serde_roundtrip() {
        let p = ChangeProposal {
            id: Uuid::new_v4(),
            file_id: Some(FileId(Uuid::new_v4())),
            change_type: ChangeType::RenameAndMove,
            original_path: PathBuf::from("/docs/scan001.pdf"),
            proposed_path: PathBuf::from("/docs/Finance/tax-return.pdf"),
            proposed_name: "tax-return.pdf".to_string(),
            confidence: 0.87,
            reasoning: "Contains tax forms".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: Some(0.91),
            rename_mismatch_score: Some(0.72),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: ChangeProposal = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn proposal_bundle_member_serde_roundtrip() {
        let bundle_id = Uuid::new_v4();
        let p = ChangeProposal {
            id: Uuid::new_v4(),
            file_id: Some(FileId(Uuid::new_v4())),
            change_type: ChangeType::Move,
            original_path: PathBuf::from("/src/my-project/src/main.rs"),
            proposed_path: PathBuf::from("/code/my-project/src/main.rs"),
            proposed_name: "main.rs".to_string(),
            confidence: 0.95,
            reasoning: "Bundle member".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: Some(bundle_id),
            classification_confidence: None,
            rename_mismatch_score: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: ChangeProposal = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.bundle_id, Some(bundle_id));
    }
}
