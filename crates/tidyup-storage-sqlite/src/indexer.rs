//! Directory indexer — walk a tree, hash contents with BLAKE3, upsert into a [`FileIndex`].
//!
//! The indexer is the first stage of every operational mode: before bundle detection or
//! classification, we need a `FileIndex` populated with every observed path, its
//! content hash, inferred MIME type, and metadata snapshot. It's a pure I/O + hashing
//! step with no inference dependencies.
//!
//! # Hashing
//!
//! BLAKE3 is used per the workspace policy (see `CLAUDE.md`). It's ~2–3× faster than
//! SHA-256 and pure Rust. The hash is hex-encoded (lowercase) for SQL storage.
//!
//! # MIME detection
//!
//! Two-step:
//! 1. [`infer::get`] sniffs magic bytes — authoritative when it fires (matches the
//!    file header, not the extension).
//! 2. Fall back to [`mime_guess::from_path`] when sniffing returns `None`.
//!
//! This matches the docorg behaviour and guards against mislabeled extensions without
//! pulling `file(1)` or libmagic.
//!
//! # Concurrency
//!
//! Each file's read + hash + mime sniff runs in [`tokio::task::spawn_blocking`] so the
//! async runtime isn't stalled by disk I/O or CPU-bound hashing of large files. Upserts
//! happen sequentially because the `FileIndex` trait is async but the sqlite default
//! impl already wraps writes in its own `spawn_blocking`.
//!
//! # Dotfile skipping
//!
//! Entries with any path component starting with `.` are skipped. This avoids indexing
//! `.git`, `.cache`, `.DS_Store`, `node_modules/.bin`, etc. — and crucially prevents
//! descending into a `.git` directory whose enclosing tree will be picked up by bundle
//! detection later anyway.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use tidyup_core::storage::FileIndex;
use tidyup_domain::{ContentHash, FileId, IndexedFile};
use uuid::Uuid;
use walkdir::WalkDir;

/// Walk `root` recursively, index every file, upsert each into `index`.
///
/// Returns the full set of [`IndexedFile`]s observed this pass. Skips entries that
/// cannot be read (logged via `tracing::warn`) and files under dotted path components.
///
/// The returned records carry the `FileId` assigned this pass. When a record for the
/// path already exists in the index, the `FileIndex::upsert` contract preserves the
/// original `FileId` — use [`FileIndex::by_path`] afterwards to recover the canonical
/// id if needed.
pub async fn index_directory(root: &Path, index: &dyn FileIndex) -> Result<Vec<IndexedFile>> {
    let entries = collect_entries(root);

    let mut indexed = Vec::with_capacity(entries.len());
    for path in entries {
        if let Some(file) = describe(path).await? {
            index
                .upsert(&file)
                .await
                .with_context(|| format!("upserting {}", file.path.display()))?;
            indexed.push(file);
        }
    }

    Ok(indexed)
}

fn collect_entries(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_dot_entry(e))
        .filter_map(|entry| match entry {
            Ok(e) => Some(e),
            Err(err) => {
                tracing::warn!("walk error under {}: {err}", root.display());
                None
            }
        })
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .collect()
}

/// Reject entries whose **own** name starts with `.`. Uses `DirEntry::depth` to skip
/// the walk root itself — otherwise temp dirs named `.tmpXXXX` (macOS default) or
/// `.cache` trees would never be walked.
fn is_dot_entry(entry: &walkdir::DirEntry) -> bool {
    if entry.depth() == 0 {
        return false;
    }
    entry.file_name().to_string_lossy().starts_with('.')
}

async fn describe(path: PathBuf) -> Result<Option<IndexedFile>> {
    tokio::task::spawn_blocking(move || describe_blocking(&path))
        .await
        .context("joining describe task")
}

fn describe_blocking(path: &Path) -> Option<IndexedFile> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("skipping {}: {e}", path.display());
            return None;
        }
    };
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("metadata unreadable for {}: {e}", path.display());
            return None;
        }
    };

    let hash = ContentHash(blake3::hash(&bytes).to_hex().to_string());
    let mime = infer::get(&bytes).map_or_else(
        || {
            mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string()
        },
        |t| t.mime_type().to_string(),
    );

    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let extension = path
        .extension()
        .map_or_else(String::new, |e| e.to_string_lossy().into_owned());

    Some(IndexedFile {
        id: FileId(Uuid::new_v4()),
        path: path.to_path_buf(),
        name,
        extension,
        mime_type: mime,
        size_bytes: metadata.len(),
        content_hash: hash,
        indexed_at: Utc::now(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::SqliteStore;
    use std::fs;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, bytes: &[u8]) -> PathBuf {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, bytes).unwrap();
        p
    }

    #[tokio::test]
    async fn blake3_hashes_match_domain_contract() {
        let dir = TempDir::new().unwrap();
        let p = write(dir.path(), "hello.txt", b"hello world");
        let store = SqliteStore::open_in_memory().unwrap();

        let indexed = index_directory(dir.path(), &store).await.unwrap();
        assert_eq!(indexed.len(), 1);

        let expected = blake3::hash(b"hello world").to_hex().to_string();
        assert_eq!(indexed[0].content_hash.0, expected);
        assert_eq!(indexed[0].path, p);
        assert_eq!(indexed[0].size_bytes, 11);
    }

    #[tokio::test]
    async fn skips_dotfiles_and_dot_subtrees() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "keep.txt", b"visible");
        write(dir.path(), ".hidden", b"nope");
        write(dir.path(), ".git/config", b"also nope");
        write(dir.path(), "sub/.cache/data", b"still nope");

        let store = SqliteStore::open_in_memory().unwrap();
        let indexed = index_directory(dir.path(), &store).await.unwrap();

        let names: Vec<_> = indexed.iter().map(|f| f.name.clone()).collect();
        assert_eq!(names, ["keep.txt"]);
    }

    #[tokio::test]
    async fn upserts_are_visible_via_by_path() {
        let dir = TempDir::new().unwrap();
        let p = write(dir.path(), "doc.pdf", b"%PDF-1.7\nbody");
        let store = SqliteStore::open_in_memory().unwrap();

        index_directory(dir.path(), &store).await.unwrap();

        let got = store.by_path(&p).await.unwrap().unwrap();
        assert_eq!(got.name, "doc.pdf");
        assert_eq!(got.extension, "pdf");
    }

    #[tokio::test]
    async fn content_sniff_overrides_extension_when_available() {
        // A real JPEG SOI header under a misleading .txt extension — infer should
        // catch this and return image/jpeg rather than the extension's text/plain.
        let dir = TempDir::new().unwrap();
        let jpeg_soi = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F'];
        write(dir.path(), "lie.txt", &jpeg_soi);

        let store = SqliteStore::open_in_memory().unwrap();
        let indexed = index_directory(dir.path(), &store).await.unwrap();
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].mime_type, "image/jpeg");
    }

    #[tokio::test]
    async fn rescan_preserves_file_id() {
        let dir = TempDir::new().unwrap();
        let p = write(dir.path(), "stable.txt", b"first");
        let store = SqliteStore::open_in_memory().unwrap();

        index_directory(dir.path(), &store).await.unwrap();
        let first_id = store.by_path(&p).await.unwrap().unwrap().id;

        // Modify the file so content_hash + indexed_at change, then rescan.
        fs::write(&p, b"second").unwrap();
        index_directory(dir.path(), &store).await.unwrap();

        let after = store.by_path(&p).await.unwrap().unwrap();
        assert_eq!(after.id, first_id, "FileId must survive re-index");
        assert_ne!(
            after.content_hash.0,
            blake3::hash(b"first").to_hex().to_string()
        );
    }
}
