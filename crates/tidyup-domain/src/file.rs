//! File identity and indexed-file records.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
/// BLAKE3 is ~2–3× faster than SHA-256, pure Rust, and cryptographically strong. Used across
/// the workspace for content-addressing (`FileIndex` keying, `ProfileCache` invalidation).
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
}
