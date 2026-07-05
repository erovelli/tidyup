//! Backup records — every applied change shelves the original for rollback.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::change::ParseError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStatus {
    Shelved,
    Unshelved,
    Expired,
}

impl BackupStatus {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Shelved => "Shelved",
            Self::Unshelved => "Unshelved",
            Self::Expired => "Expired",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "Shelved" => Ok(Self::Shelved),
            "Unshelved" => Ok(Self::Unshelved),
            "Expired" => Ok(Self::Expired),
            other => Err(ParseError::UnknownBackupStatus(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackupRecord {
    pub id: Uuid,
    pub change_id: Uuid,
    pub original_path: PathBuf,
    pub backup_path: PathBuf,
    pub shelved_at: DateTime<Utc>,
    pub unshelved_at: Option<DateTime<Utc>>,
    pub status: BackupStatus,
    /// Content digest of the shelved original at shelve time. For a single file
    /// this is the BLAKE3 of its bytes; for a bundle subtree it is a canonical
    /// tree digest. `None` for records written before content hashing shipped —
    /// rollback then falls back to comparing the live destination against the
    /// shelf copy directly. Used to (a) detect shelf corruption and (b) detect a
    /// destination edited after apply, so rollback never destroys user edits.
    #[serde(default)]
    pub content_hash: Option<String>,
}

/// Verdict on whether a shelved backup can be safely restored over its
/// destination.
///
/// Returned by the storage layer's restore-precheck so the rollback caller
/// never has to re-derive the hashing algorithm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorePrecheck {
    /// The shelf copy is intact and the destination still matches what was
    /// shelved (or the destination is already gone). Safe to delete the
    /// destination and restore the original.
    Ready,
    /// The shelf copy is missing or corrupt — restoring would itself lose data.
    /// The caller must not touch the destination. Carries a human-readable
    /// reason.
    ShelfUnusable(String),
    /// The destination was modified after the move (its content no longer
    /// matches the shelved original). Deleting it would destroy the user's
    /// edits, so the caller must skip and report a conflict.
    DestinationModified,
    /// The original location is occupied by content that does not match the
    /// shelved original — e.g. the user saved a *new* file into the vacated
    /// slot after apply. Restoring would overwrite it, so the caller must skip
    /// and report a conflict.
    OriginalOccupied,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn backup_status_roundtrip() {
        for bs in [
            BackupStatus::Shelved,
            BackupStatus::Unshelved,
            BackupStatus::Expired,
        ] {
            let s = bs.as_str();
            let back = BackupStatus::parse(s).unwrap();
            assert_eq!(bs, back);
        }
    }

    #[test]
    fn backup_record_serde_roundtrip() {
        let b = BackupRecord {
            id: Uuid::new_v4(),
            change_id: Uuid::new_v4(),
            original_path: PathBuf::from("/docs/scan001.pdf"),
            backup_path: PathBuf::from("/backups/2024-01-01/abc/scan001.pdf"),
            shelved_at: Utc::now(),
            unshelved_at: None,
            status: BackupStatus::Shelved,
            content_hash: Some("af1349b9f5f9a1a6a0404dea36dcc949".to_string()),
        };
        let json = serde_json::to_string(&b).unwrap();
        let back: BackupRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn backup_record_deserialises_without_content_hash() {
        // Records written before content hashing shipped have no `content_hash`
        // field; `#[serde(default)]` must accept them as `None`.
        let legacy = r#"{
            "id":"00000000-0000-0000-0000-000000000001",
            "change_id":"00000000-0000-0000-0000-000000000002",
            "original_path":"/a","backup_path":"/b",
            "shelved_at":"2024-01-01T00:00:00Z","unshelved_at":null,
            "status":"Shelved"
        }"#;
        let back: BackupRecord = serde_json::from_str(legacy).unwrap();
        assert_eq!(back.content_hash, None);
    }

    #[test]
    fn restore_precheck_roundtrip() {
        for v in [
            RestorePrecheck::Ready,
            RestorePrecheck::ShelfUnusable("missing".to_string()),
            RestorePrecheck::DestinationModified,
            RestorePrecheck::OriginalOccupied,
        ] {
            let json = serde_json::to_string(&v).unwrap();
            let back: RestorePrecheck = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }
}
