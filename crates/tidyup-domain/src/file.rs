//! File identity and indexed-file records.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::change::ParseError;

/// Stable per-file identifier. Preserved across re-scans via upsert on path.
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

/// BLAKE3 content hash, hex-encoded.
///
/// BLAKE3 is ~2–3× faster than SHA-256, pure Rust, and cryptographically strong. It
/// content-addresses files: captured on each `ChangeProposal` for the apply-time TOCTOU
/// guard and stored in the `files` table's indexed `content_hash` column. Versioned
/// semantic media artifacts also use it as their stable cross-path cache identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

/// A file observed on disk. The canonical record written to the `FileIndex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedFile {
    pub id: FileId,
    pub path: PathBuf,
    pub name: String,
    pub extension: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub content_hash: ContentHash,
    pub indexed_at: DateTime<Utc>,
}

/// How a path participates in a particular run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileProcessingRole {
    /// A source item eligible for organization.
    Source,
    /// An existing target item used only to learn a migration hierarchy.
    TargetProfile,
}

impl FileProcessingRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "Source",
            Self::TargetProfile => "TargetProfile",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ParseError> {
        match value {
            "Source" => Ok(Self::Source),
            "TargetProfile" => Ok(Self::TargetProfile),
            other => Err(ParseError::UnknownFileProcessingRole(other.to_string())),
        }
    }
}

/// Pipeline stage responsible for the current per-run file state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileProcessingStage {
    Discovery,
    Indexing,
    Profiling,
    Classification,
    Planning,
}

impl FileProcessingStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Discovery => "Discovery",
            Self::Indexing => "Indexing",
            Self::Profiling => "Profiling",
            Self::Classification => "Classification",
            Self::Planning => "Planning",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ParseError> {
        match value {
            "Discovery" => Ok(Self::Discovery),
            "Indexing" => Ok(Self::Indexing),
            "Profiling" => Ok(Self::Profiling),
            "Classification" => Ok(Self::Classification),
            "Planning" => Ok(Self::Planning),
            other => Err(ParseError::UnknownFileProcessingStage(other.to_string())),
        }
    }
}

/// Durable processing outcome for one path in one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileProcessingState {
    Discovered,
    Indexed,
    Classified,
    Unclassified,
    Deferred,
    Failed,
}

impl FileProcessingState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "Discovered",
            Self::Indexed => "Indexed",
            Self::Classified => "Classified",
            Self::Unclassified => "Unclassified",
            Self::Deferred => "Deferred",
            Self::Failed => "Failed",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ParseError> {
        match value {
            "Discovered" => Ok(Self::Discovered),
            "Indexed" => Ok(Self::Indexed),
            "Classified" => Ok(Self::Classified),
            "Unclassified" => Ok(Self::Unclassified),
            "Deferred" => Ok(Self::Deferred),
            "Failed" => Ok(Self::Failed),
            other => Err(ParseError::UnknownFileProcessingState(other.to_string())),
        }
    }
}

/// Run-scoped processing record. `file_id` is absent when identity indexing
/// failed before a canonical [`IndexedFile`] could be produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileProcessingRecord {
    pub run_id: Uuid,
    pub path: PathBuf,
    pub file_id: Option<FileId>,
    pub role: FileProcessingRole,
    pub stage: FileProcessingStage,
    pub state: FileProcessingState,
    pub reason: Option<String>,
    pub updated_at: DateTime<Utc>,
}

/// Versioned semantic representation cached by content identity.
///
/// `latent_space` is an explicit model-family label such as `image` or
/// `audio`; consumers must still require an exact `model_id` and
/// `preprocessing_version` match before reuse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticArtifact {
    pub content_hash: ContentHash,
    pub model_id: String,
    pub preprocessing_version: String,
    pub latent_space: String,
    pub embedding: Vec<f32>,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn file_id_roundtrip() {
        let id = FileId(Uuid::new_v4());
        let json = serde_json::to_string(&id).unwrap();
        let back: FileId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn content_hash_roundtrip() {
        let hash = ContentHash("abc123".to_string());
        let json = serde_json::to_string(&hash).unwrap();
        let back: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(hash, back);
    }

    #[test]
    fn indexed_file_roundtrip() {
        let file = IndexedFile {
            id: FileId(Uuid::new_v4()),
            path: PathBuf::from("/tmp/test.pdf"),
            name: "test.pdf".to_string(),
            extension: "pdf".to_string(),
            mime_type: "application/pdf".to_string(),
            size_bytes: 1024,
            content_hash: ContentHash("deadbeef".to_string()),
            indexed_at: Utc::now(),
        };
        let json = serde_json::to_string(&file).unwrap();
        let back: IndexedFile = serde_json::from_str(&json).unwrap();
        assert_eq!(file, back);
    }

    #[test]
    fn semantic_artifact_roundtrip() {
        let artifact = SemanticArtifact {
            content_hash: ContentHash("abc123".to_string()),
            model_id: "siglip".to_string(),
            preprocessing_version: "image-v1".to_string(),
            latent_space: "image".to_string(),
            embedding: vec![1.0, 0.0],
            updated_at: Utc::now(),
        };
        let json = serde_json::to_string(&artifact).unwrap();
        let back: SemanticArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(artifact, back);
    }

    #[test]
    fn processing_enums_roundtrip() {
        for role in [
            FileProcessingRole::Source,
            FileProcessingRole::TargetProfile,
        ] {
            assert_eq!(FileProcessingRole::parse(role.as_str()).unwrap(), role);
        }
        for stage in [
            FileProcessingStage::Discovery,
            FileProcessingStage::Indexing,
            FileProcessingStage::Profiling,
            FileProcessingStage::Classification,
            FileProcessingStage::Planning,
        ] {
            assert_eq!(FileProcessingStage::parse(stage.as_str()).unwrap(), stage);
        }
        for state in [
            FileProcessingState::Discovered,
            FileProcessingState::Indexed,
            FileProcessingState::Classified,
            FileProcessingState::Unclassified,
            FileProcessingState::Deferred,
            FileProcessingState::Failed,
        ] {
            assert_eq!(FileProcessingState::parse(state.as_str()).unwrap(), state);
        }
    }
}
