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
        };
        let json = serde_json::to_string(&b).unwrap();
        let back: BackupRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(b, back);
    }
}
