//! Persistent, format-neutral discovery and identity indexing.
//!
//! Indexing deliberately stops at identity: MIME sniffing selects a decoder,
//! while placement and naming remain semantic pipeline concerns. Keeping this
//! pass in the pipeline lets both scan and migration populate the same
//! `FileIndex` without coupling the application layer to `SQLite`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use tidyup_core::storage::FileIndex;
use tidyup_domain::{
    ContentHash, FileId, FileProcessingRecord, FileProcessingRole, FileProcessingStage,
    FileProcessingState, IndexedFile,
};
use uuid::Uuid;
use walkdir::WalkDir;

/// Result of one identity-indexing pass.
#[derive(Debug, Clone, Default)]
pub struct IndexSummary {
    pub indexed: Vec<IndexedFile>,
    pub failed: usize,
}

/// Walk `root`, upsert every readable regular file, and persist a run-scoped
/// state for every path the walker can identify.
///
/// Identity failures retain the path, stage, and reason even when no
/// [`IndexedFile`] or [`FileId`] can be produced.
pub async fn index_directory(
    root: &Path,
    index: &dyn FileIndex,
    run_id: Uuid,
    role: FileProcessingRole,
) -> Result<IndexSummary> {
    let (paths, discovery_failures) = collect_entries(root);
    let mut summary = IndexSummary {
        failed: discovery_failures.len(),
        ..IndexSummary::default()
    };
    let discovery_records = discovery_failures
        .into_iter()
        .map(|(path, reason)| {
            processing_record(
                run_id,
                path,
                None,
                role,
                FileProcessingStage::Discovery,
                FileProcessingState::Failed,
                Some(reason),
            )
        })
        .chain(paths.iter().cloned().map(|path| {
            processing_record(
                run_id,
                path,
                None,
                role,
                FileProcessingStage::Discovery,
                FileProcessingState::Discovered,
                None,
            )
        }))
        .collect::<Vec<_>>();
    index.put_processing_records(&discovery_records).await?;

    let mut indexing_records = Vec::with_capacity(paths.len());
    for path in paths {
        let record = match describe(path.clone()).await? {
            Ok(record) => record,
            Err(reason) => {
                tracing::warn!("indexer: unable to index {}: {reason}", path.display());
                indexing_records.push(processing_record(
                    run_id,
                    path,
                    None,
                    role,
                    FileProcessingStage::Indexing,
                    FileProcessingState::Failed,
                    Some(reason),
                ));
                summary.failed = summary.failed.saturating_add(1);
                continue;
            }
        };
        index
            .upsert(&record)
            .await
            .with_context(|| format!("upserting indexed file {}", record.path.display()))?;
        let canonical = index.by_path(&record.path).await?.ok_or_else(|| {
            anyhow::anyhow!(
                "indexed path was not readable after upsert: {}",
                record.path.display()
            )
        })?;
        indexing_records.push(processing_record(
            run_id,
            canonical.path.clone(),
            Some(canonical.id.clone()),
            role,
            FileProcessingStage::Indexing,
            FileProcessingState::Indexed,
            None,
        ));
        summary.indexed.push(canonical);
    }
    index.put_processing_records(&indexing_records).await?;
    Ok(summary)
}

fn collect_entries(root: &Path) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
    let mut paths = Vec::new();
    let mut failures = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        match entry {
            Ok(entry) if entry.file_type().is_file() => paths.push(entry.into_path()),
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(
                    "indexer: unable to read entry under {}: {error}",
                    root.display()
                );
                if let Some(path) = error.path() {
                    failures.push((path.to_path_buf(), error.to_string()));
                }
            }
        }
    }
    (paths, failures)
}

async fn describe(path: PathBuf) -> Result<std::result::Result<IndexedFile, String>> {
    tokio::task::spawn_blocking(move || describe_blocking(&path))
        .await
        .context("joining identity indexing task")
}

fn describe_blocking(path: &Path) -> std::result::Result<IndexedFile, String> {
    let metadata = std::fs::metadata(path).map_err(|error| format!("stat failed: {error}"))?;
    let mut file = std::fs::File::open(path).map_err(|error| format!("open failed: {error}"))?;
    // MIME sniffing only needs a bounded prefix; hashing remains streaming so
    // large downloads never allocate their entire contents.
    let mut prefix = vec![0_u8; 64 * 1024];
    let prefix_len = std::io::Read::read(&mut file, &mut prefix)
        .map_err(|error| format!("initial read failed: {error}"))?;
    prefix.truncate(prefix_len);
    let mut hasher = blake3::Hasher::new();
    hasher.update(&prefix);
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let length = std::io::Read::read(&mut file, &mut buffer)
            .map_err(|error| format!("hash read failed: {error}"))?;
        if length == 0 {
            break;
        }
        let Some(chunk) = buffer.get(..length) else {
            return Err("invalid read length while hashing".to_string());
        };
        hasher.update(chunk);
    }
    let mime_type = infer::get(&prefix).map_or_else(
        || {
            mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string()
        },
        |kind| kind.mime_type().to_string(),
    );
    Ok(IndexedFile {
        id: FileId(Uuid::new_v4()),
        path: path.to_path_buf(),
        name: path
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
        extension: path.extension().map_or_else(String::new, |extension| {
            extension.to_string_lossy().into_owned()
        }),
        mime_type,
        size_bytes: metadata.len(),
        content_hash: ContentHash(hasher.finalize().to_hex().to_string()),
        indexed_at: Utc::now(),
    })
}

#[allow(clippy::too_many_arguments)]
fn processing_record(
    run_id: Uuid,
    path: PathBuf,
    file_id: Option<FileId>,
    role: FileProcessingRole,
    lifecycle_step: FileProcessingStage,
    lifecycle_status: FileProcessingState,
    reason: Option<String>,
) -> FileProcessingRecord {
    FileProcessingRecord {
        run_id,
        path,
        file_id,
        role,
        stage: lifecycle_step,
        state: lifecycle_status,
        reason,
        updated_at: Utc::now(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    use tidyup_storage_sqlite::SqliteStore;

    async fn run(store: &SqliteStore, root: &Path) -> (Uuid, IndexSummary) {
        use tidyup_core::storage::RunLog;
        use tidyup_domain::{RunMode, RunRecord};

        let record = RunRecord::begin(RunMode::Scan, root.to_path_buf(), None);
        store.record_run(&record).await.unwrap();
        let summary = index_directory(root, store, record.id, FileProcessingRole::Source)
            .await
            .unwrap();
        (record.id, summary)
    }

    fn write(root: &Path, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, bytes).unwrap();
        path
    }

    #[tokio::test]
    async fn streams_blake3_hash_and_upserts_record() {
        let dir = TempDir::new().unwrap();
        let path = write(dir.path(), "hello.txt", b"hello world");
        let store = SqliteStore::open_in_memory().unwrap();

        let (_, indexed) = run(&store, dir.path()).await;
        assert_eq!(indexed.indexed.len(), 1);
        assert_eq!(
            indexed.indexed[0].content_hash.0,
            blake3::hash(b"hello world").to_hex().to_string()
        );
        assert_eq!(store.by_path(&path).await.unwrap().unwrap().path, path);
    }

    #[tokio::test]
    async fn accounts_for_dotfiles_and_dot_subtrees() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "keep.txt", b"visible");
        write(dir.path(), ".hidden", b"nope");
        write(dir.path(), ".git/config", b"also nope");
        write(dir.path(), "sub/.cache/data", b"still nope");
        let store = SqliteStore::open_in_memory().unwrap();

        let (_, indexed) = run(&store, dir.path()).await;
        let mut names = indexed
            .indexed
            .iter()
            .map(|file| file.name.as_str())
            .collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(names, [".hidden", "config", "data", "keep.txt"]);
    }

    #[tokio::test]
    async fn magic_mime_wins_over_misleading_extension() {
        let dir = TempDir::new().unwrap();
        let jpeg_soi = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F'];
        write(dir.path(), "lie.txt", &jpeg_soi);
        let store = SqliteStore::open_in_memory().unwrap();

        let (_, indexed) = run(&store, dir.path()).await;
        assert_eq!(indexed.indexed[0].mime_type, "image/jpeg");
    }

    #[tokio::test]
    async fn rescans_preserve_file_id() {
        let dir = TempDir::new().unwrap();
        let path = write(dir.path(), "stable.txt", b"first");
        let store = SqliteStore::open_in_memory().unwrap();
        let (_, _) = run(&store, dir.path()).await;
        let first_id = store.by_path(&path).await.unwrap().unwrap().id;

        fs::write(&path, b"second").unwrap();
        let (_, _) = run(&store, dir.path()).await;
        let after = store.by_path(&path).await.unwrap().unwrap();
        assert_eq!(after.id, first_id);
        assert_ne!(
            after.content_hash.0,
            blake3::hash(b"first").to_hex().to_string()
        );
    }
}
