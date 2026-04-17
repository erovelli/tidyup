//! `FileIndex` implementation on `SqliteStore`.
//!
//! `upsert` preserves the existing `FileId` on path conflict via `SQLite`'s
//! `ON CONFLICT(path) DO UPDATE` — the caller-supplied `id` is used only for brand-new rows.
//! This is the load-bearing invariant for cross-scan stability.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use rusqlite::{params, Row};
use tidyup_core::storage::FileIndex;
use tidyup_domain::{ContentHash, FileId, IndexedFile};
use uuid::Uuid;

use crate::SqliteStore;

const COLS: &str = "id, path, name, extension, mime_type, size_bytes, content_hash, indexed_at";

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", p.display()))
}

/// Escape SQL LIKE metacharacters (`%`, `_`, `\`) so prefix matching is literal.
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if matches!(ch, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn row_to_file(row: &Row<'_>) -> rusqlite::Result<IndexedFile> {
    let id_str: String = row.get("id")?;
    let path_str: String = row.get("path")?;
    let hash: String = row.get("content_hash")?;
    let size: i64 = row.get("size_bytes")?;
    let id = Uuid::parse_str(&id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    #[allow(clippy::cast_sign_loss)]
    let size_bytes = if size < 0 { 0 } else { size as u64 };
    Ok(IndexedFile {
        id: FileId(id),
        path: PathBuf::from(path_str),
        name: row.get("name")?,
        extension: row.get("extension")?,
        mime_type: row.get("mime_type")?,
        size_bytes,
        content_hash: ContentHash(hash),
        indexed_at: row.get("indexed_at")?,
    })
}

#[async_trait]
impl FileIndex for SqliteStore {
    async fn upsert(&self, record: &IndexedFile) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let record = record.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let size_bytes =
                i64::try_from(record.size_bytes).context("file size_bytes exceeds i64 range")?;
            let path = path_str(&record.path)?;
            {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        "INSERT INTO files (id, path, name, extension, mime_type, size_bytes, \
                         content_hash, indexed_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                         ON CONFLICT(path) DO UPDATE SET \
                            name = excluded.name, \
                            extension = excluded.extension, \
                            mime_type = excluded.mime_type, \
                            size_bytes = excluded.size_bytes, \
                            content_hash = excluded.content_hash, \
                            indexed_at = excluded.indexed_at",
                        params![
                            record.id.0.to_string(),
                            path,
                            record.name,
                            record.extension,
                            record.mime_type,
                            size_bytes,
                            record.content_hash.0,
                            record.indexed_at,
                        ],
                    )
                    .context("upserting indexed file")?;
            }
            Ok(())
        })
        .await
        .context("join upsert task")??;
        Ok(())
    }

    async fn get(&self, id: &FileId) -> tidyup_core::Result<Option<IndexedFile>> {
        let conn = self.conn();
        let id_str = id.0.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<IndexedFile>> {
            let row = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt = guard.prepare(&format!("SELECT {COLS} FROM files WHERE id = ?1"))?;
                stmt.query_row(params![id_str], row_to_file)
                    .map(Some)
                    .or_else(|e| match e {
                        rusqlite::Error::QueryReturnedNoRows => Ok(None),
                        other => Err(other),
                    })?
            };
            Ok(row)
        })
        .await
        .context("join get task")??;
        Ok(result)
    }

    async fn by_path(&self, path: &Path) -> tidyup_core::Result<Option<IndexedFile>> {
        let conn = self.conn();
        let path = path_str(path)?.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<IndexedFile>> {
            let row = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt =
                    guard.prepare(&format!("SELECT {COLS} FROM files WHERE path = ?1"))?;
                stmt.query_row(params![path], row_to_file)
                    .map(Some)
                    .or_else(|e| match e {
                        rusqlite::Error::QueryReturnedNoRows => Ok(None),
                        other => Err(other),
                    })?
            };
            Ok(row)
        })
        .await
        .context("join by_path task")??;
        Ok(result)
    }

    async fn list_under(&self, root: &Path) -> tidyup_core::Result<Vec<IndexedFile>> {
        let conn = self.conn();
        let root_str = path_str(root)?.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<IndexedFile>> {
            let rows = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let prefix = format!("{}/%", escape_like(&root_str));
                let mut stmt = guard.prepare(&format!(
                    "SELECT {COLS} FROM files WHERE path = ?1 OR path LIKE ?2 ESCAPE '\\' \
                     ORDER BY path"
                ))?;
                let fetched: Vec<IndexedFile> = stmt
                    .query_map(params![root_str, prefix], row_to_file)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                fetched
            };
            Ok(rows)
        })
        .await
        .context("join list_under task")??;
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_file(path: &str, id: Uuid) -> IndexedFile {
        IndexedFile {
            id: FileId(id),
            path: PathBuf::from(path),
            name: path.rsplit('/').next().unwrap_or("").to_string(),
            extension: "rs".to_string(),
            mime_type: "text/x-rust".to_string(),
            size_bytes: 42,
            content_hash: ContentHash("deadbeef".to_string()),
            indexed_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn upsert_preserves_file_id_across_rescans() {
        let store = SqliteStore::open_in_memory().unwrap();
        let first_id = Uuid::new_v4();
        store
            .upsert(&sample_file("/code/main.rs", first_id))
            .await
            .unwrap();

        let second_id = Uuid::new_v4();
        assert_ne!(first_id, second_id);
        store
            .upsert(&sample_file("/code/main.rs", second_id))
            .await
            .unwrap();

        let found = store
            .by_path(Path::new("/code/main.rs"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(found.id.0, first_id, "upsert must preserve original FileId");
    }

    #[tokio::test]
    async fn by_path_and_get_return_none_when_absent() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert!(store.by_path(Path::new("/nope")).await.unwrap().is_none());
        assert!(store.get(&FileId(Uuid::new_v4())).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_under_returns_recursive_subtree() {
        let store = SqliteStore::open_in_memory().unwrap();
        store
            .upsert(&sample_file("/code/a.rs", Uuid::new_v4()))
            .await
            .unwrap();
        store
            .upsert(&sample_file("/code/sub/b.rs", Uuid::new_v4()))
            .await
            .unwrap();
        store
            .upsert(&sample_file("/other/c.rs", Uuid::new_v4()))
            .await
            .unwrap();

        let listed = store.list_under(Path::new("/code")).await.unwrap();
        let paths: Vec<_> = listed
            .iter()
            .map(|f| f.path.display().to_string())
            .collect();
        assert_eq!(paths, ["/code/a.rs", "/code/sub/b.rs"]);
    }
}
