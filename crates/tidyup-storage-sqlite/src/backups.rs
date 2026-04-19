//! `BackupStore` implementation.
//!
//! Shelf layout: `<backup_root>/<YYYY-MM-DD>/<id>/<name>`, where `id` is the originating
//! proposal id (a `ChangeProposal::id` for single files, a `BundleProposal::id` for
//! subtrees). Single-file shelves land as `<name>` — a file. Bundle shelves land as
//! `<root_basename>/...` — a directory recursively mirroring the source subtree.
//!
//! `BackupRecord::change_id` holds whichever id the caller supplied; the schema doesn't
//! distinguish, matching the domain (`change_id: Uuid`) which intentionally treats a
//! bundle-level change and a loose change uniformly.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{params, Row, Transaction};
use tidyup_core::storage::BackupStore;
use tidyup_domain::{BackupRecord, BackupStatus, IndexedFile};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::SqliteStore;

const BACKUP_COLS: &str =
    "id, change_id, original_path, backup_path, shelved_at, unshelved_at, status";

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", p.display()))
}

fn parse_uuid(s: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn row_to_backup(row: &Row<'_>) -> rusqlite::Result<BackupRecord> {
    let id = parse_uuid(&row.get::<_, String>("id")?)?;
    let change_id = parse_uuid(&row.get::<_, String>("change_id")?)?;
    let status_s: String = row.get("status")?;
    let status = BackupStatus::parse(&status_s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(BackupRecord {
        id,
        change_id,
        original_path: PathBuf::from(row.get::<_, String>("original_path")?),
        backup_path: PathBuf::from(row.get::<_, String>("backup_path")?),
        shelved_at: row.get::<_, DateTime<Utc>>("shelved_at")?,
        unshelved_at: row.get::<_, Option<DateTime<Utc>>>("unshelved_at")?,
        status,
    })
}

fn insert_backup(tx: &Transaction<'_>, r: &BackupRecord) -> Result<()> {
    tx.execute(
        &format!(
            "INSERT INTO backups ({BACKUP_COLS}) VALUES \
             (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
        ),
        params![
            r.id.to_string(),
            r.change_id.to_string(),
            path_str(&r.original_path)?,
            path_str(&r.backup_path)?,
            r.shelved_at,
            r.unshelved_at,
            r.status.as_str(),
        ],
    )
    .context("inserting backup record")?;
    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).with_context(|| format!("mkdir {}", dst.display()))?;
    for entry in WalkDir::new(src).min_depth(1) {
        let entry = entry.with_context(|| format!("walking {}", src.display()))?;
        let relative = entry
            .path()
            .strip_prefix(src)
            .context("computing relative path during recursive copy")?;
        let target = dst.join(relative);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target)
                .with_context(|| format!("mkdir {}", target.display()))?;
        } else {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("mkdir {}", parent.display()))?;
            }
            std::fs::copy(entry.path(), &target).with_context(|| {
                format!("copy {} -> {}", entry.path().display(), target.display())
            })?;
        }
    }
    Ok(())
}

fn shelf_dir(backup_root: &Path, shelved_at: DateTime<Utc>, id: Uuid) -> PathBuf {
    let date = shelved_at.format("%Y-%m-%d").to_string();
    backup_root.join(date).join(id.to_string())
}

fn require_backup_root(store: &SqliteStore) -> Result<std::sync::Arc<PathBuf>> {
    store.backup_root().ok_or_else(|| {
        anyhow!("BackupStore requires a backup_root; call SqliteStore::with_backup_root")
    })
}

#[async_trait]
impl BackupStore for SqliteStore {
    async fn shelve(
        &self,
        file: &IndexedFile,
        change_id: Uuid,
    ) -> tidyup_core::Result<BackupRecord> {
        let backup_root = require_backup_root(self)?;
        let conn = self.conn();
        let original_path = file.path.clone();
        let name = original_path
            .file_name()
            .ok_or_else(|| anyhow!("file has no name: {}", original_path.display()))?
            .to_string_lossy()
            .into_owned();
        let result = tokio::task::spawn_blocking(move || -> Result<BackupRecord> {
            let shelved_at = Utc::now();
            let target_dir = shelf_dir(backup_root.as_path(), shelved_at, change_id);
            std::fs::create_dir_all(&target_dir)
                .with_context(|| format!("mkdir shelf {}", target_dir.display()))?;
            let backup_path = target_dir.join(&name);
            std::fs::copy(&original_path, &backup_path).with_context(|| {
                format!(
                    "copy {} -> {}",
                    original_path.display(),
                    backup_path.display()
                )
            })?;
            let record = BackupRecord {
                id: Uuid::new_v4(),
                change_id,
                original_path,
                backup_path,
                shelved_at,
                unshelved_at: None,
                status: BackupStatus::Shelved,
            };
            {
                let mut guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let tx = guard.transaction()?;
                insert_backup(&tx, &record)?;
                tx.commit()?;
            }
            Ok(record)
        })
        .await
        .context("join shelve task")??;
        Ok(result)
    }

    async fn shelve_bundle(
        &self,
        root: &Path,
        bundle_id: Uuid,
    ) -> tidyup_core::Result<BackupRecord> {
        let backup_root = require_backup_root(self)?;
        let conn = self.conn();
        let original_path = root.to_path_buf();
        let name = original_path
            .file_name()
            .ok_or_else(|| anyhow!("bundle root has no name: {}", original_path.display()))?
            .to_string_lossy()
            .into_owned();
        let result = tokio::task::spawn_blocking(move || -> Result<BackupRecord> {
            let shelved_at = Utc::now();
            let target_dir = shelf_dir(backup_root.as_path(), shelved_at, bundle_id);
            let backup_path = target_dir.join(&name);
            copy_dir_recursive(&original_path, &backup_path)?;
            let record = BackupRecord {
                id: Uuid::new_v4(),
                change_id: bundle_id,
                original_path,
                backup_path,
                shelved_at,
                unshelved_at: None,
                status: BackupStatus::Shelved,
            };
            {
                let mut guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let tx = guard.transaction()?;
                insert_backup(&tx, &record)?;
                tx.commit()?;
            }
            Ok(record)
        })
        .await
        .context("join shelve_bundle task")??;
        Ok(result)
    }

    async fn find_by_change_id(
        &self,
        change_id: Uuid,
    ) -> tidyup_core::Result<Option<BackupRecord>> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<BackupRecord>> {
            let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
            let mut stmt = guard.prepare(&format!(
                "SELECT {BACKUP_COLS} FROM backups \
                 WHERE change_id = ?1 AND status = ?2 \
                 ORDER BY shelved_at DESC LIMIT 1"
            ))?;
            let fetched = stmt
                .query_row(
                    params![change_id.to_string(), BackupStatus::Shelved.as_str()],
                    row_to_backup,
                )
                .ok();
            Ok(fetched)
        })
        .await
        .context("join find_by_change_id task")??;
        Ok(result)
    }

    async fn restore(&self, record: &BackupRecord) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let record = record.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(parent) = record.original_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("mkdir {}", parent.display()))?;
                }
            }
            if record.backup_path.is_dir() {
                copy_dir_recursive(&record.backup_path, &record.original_path)?;
            } else {
                std::fs::copy(&record.backup_path, &record.original_path).with_context(|| {
                    format!(
                        "copy {} -> {}",
                        record.backup_path.display(),
                        record.original_path.display()
                    )
                })?;
            }
            let now = Utc::now();
            {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        "UPDATE backups SET status = ?1, unshelved_at = ?2 WHERE id = ?3",
                        params![BackupStatus::Unshelved.as_str(), now, record.id.to_string()],
                    )
                    .context("marking backup unshelved")?;
            }
            Ok(())
        })
        .await
        .context("join restore task")??;
        Ok(())
    }

    async fn prune_older_than_days(&self, days: u32) -> tidyup_core::Result<usize> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<usize> {
            let cutoff = Utc::now() - chrono::Duration::days(i64::from(days));
            let shelved = BackupStatus::Shelved.as_str();
            let expired = BackupStatus::Expired.as_str();
            let victims: Vec<PathBuf> = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt = guard.prepare(
                    "SELECT backup_path FROM backups \
                     WHERE status = ?1 AND shelved_at < ?2",
                )?;
                let rows = stmt.query_map(params![shelved, cutoff], |r| {
                    let p: String = r.get(0)?;
                    Ok(PathBuf::from(p))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            for path in &victims {
                // Best-effort: disk failures don't block marking rows expired. A stale shelf
                // entry wastes space but never breaks correctness — restore of an expired
                // record is already invalid.
                if path.is_dir() {
                    let _ = std::fs::remove_dir_all(path);
                } else {
                    let _ = std::fs::remove_file(path);
                }
            }
            let updated = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        "UPDATE backups SET status = ?1 \
                         WHERE status = ?2 AND shelved_at < ?3",
                        params![expired, shelved, cutoff],
                    )
                    .context("marking backups expired")?
            };
            Ok(updated)
        })
        .await
        .context("join prune task")??;
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;
    use tidyup_domain::{ContentHash, FileId, IndexedFile};

    fn write_file(tmp: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let path = tmp.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn sample_indexed(path: PathBuf) -> IndexedFile {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        IndexedFile {
            id: FileId(Uuid::new_v4()),
            path,
            name,
            extension: "txt".to_string(),
            mime_type: "text/plain".to_string(),
            size_bytes: 0,
            content_hash: ContentHash("deadbeef".to_string()),
            indexed_at: Utc::now(),
        }
    }

    fn store_with_backup_root(dir: &TempDir) -> SqliteStore {
        let db = dir.path().join("t.db");
        let shelf = dir.path().join("shelf");
        std::fs::create_dir_all(&shelf).unwrap();
        SqliteStore::open(&db).unwrap().with_backup_root(shelf)
    }

    #[tokio::test]
    async fn shelve_copies_file_and_records_row() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let src = write_file(dir.path(), "scan.pdf", b"pdf-bytes");
        let change_id = Uuid::new_v4();
        let record = store
            .shelve(&sample_indexed(src.clone()), change_id)
            .await
            .unwrap();

        assert_eq!(record.change_id, change_id);
        assert_eq!(record.status, BackupStatus::Shelved);
        assert!(record.backup_path.exists(), "shelved file must exist");
        assert_eq!(std::fs::read(&record.backup_path).unwrap(), b"pdf-bytes");

        // Row roundtrips cleanly.
        let conn = store.conn();
        let id = record.id.to_string();
        let roundtripped = tokio::task::spawn_blocking(move || {
            let guard = conn.lock().unwrap();
            let mut stmt = guard
                .prepare(&format!("SELECT {BACKUP_COLS} FROM backups WHERE id = ?1"))
                .unwrap();
            stmt.query_row(params![id], row_to_backup).unwrap()
        })
        .await
        .unwrap();
        assert_eq!(roundtripped, record);
    }

    #[tokio::test]
    async fn shelve_without_backup_root_errors() {
        let dir = TempDir::new().unwrap();
        let store = SqliteStore::open(&dir.path().join("t.db")).unwrap();
        let src = write_file(dir.path(), "x.txt", b"x");
        let err = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("backup_root"), "got: {err}");
    }

    #[tokio::test]
    async fn shelve_bundle_copies_subtree() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let root = dir.path().join("proj");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("Cargo.toml"), b"[package]\nname=\"x\"").unwrap();
        std::fs::write(root.join("src/main.rs"), b"fn main() {}").unwrap();

        let bundle_id = Uuid::new_v4();
        let record = store.shelve_bundle(&root, bundle_id).await.unwrap();

        assert_eq!(record.change_id, bundle_id);
        assert!(
            record.backup_path.is_dir(),
            "bundle shelve must produce a dir"
        );
        assert!(record.backup_path.join("Cargo.toml").exists());
        assert!(record.backup_path.join("src/main.rs").exists());
    }

    #[tokio::test]
    async fn restore_copies_file_back_to_original_path() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let src = write_file(dir.path(), "note.txt", b"hello");
        let record = store
            .shelve(&sample_indexed(src.clone()), Uuid::new_v4())
            .await
            .unwrap();
        std::fs::remove_file(&src).unwrap();
        assert!(!src.exists());

        store.restore(&record).await.unwrap();
        assert_eq!(std::fs::read(&src).unwrap(), b"hello");
    }

    #[tokio::test]
    async fn restore_marks_record_unshelved() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let src = write_file(dir.path(), "a.txt", b"a");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        store.restore(&record).await.unwrap();

        let conn = store.conn();
        let id = record.id.to_string();
        let fetched = tokio::task::spawn_blocking(move || {
            let guard = conn.lock().unwrap();
            let mut stmt = guard
                .prepare(&format!("SELECT {BACKUP_COLS} FROM backups WHERE id = ?1"))
                .unwrap();
            stmt.query_row(params![id], row_to_backup).unwrap()
        })
        .await
        .unwrap();
        assert_eq!(fetched.status, BackupStatus::Unshelved);
        assert!(fetched.unshelved_at.is_some());
    }

    #[tokio::test]
    async fn prune_expires_old_shelves_and_removes_from_disk() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let src = write_file(dir.path(), "old.txt", b"old");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();

        // Backdate the shelved_at directly in SQL to simulate an old entry.
        {
            let conn = store.conn();
            let id = record.id.to_string();
            let old = Utc::now() - chrono::Duration::days(60);
            tokio::task::spawn_blocking(move || {
                let guard = conn.lock().unwrap();
                guard
                    .execute(
                        "UPDATE backups SET shelved_at = ?1 WHERE id = ?2",
                        params![old, id],
                    )
                    .unwrap();
            })
            .await
            .unwrap();
        }

        let n = store.prune_older_than_days(30).await.unwrap();
        assert_eq!(n, 1);
        assert!(!record.backup_path.exists(), "pruned file must be removed");

        // Second run finds nothing new.
        let n2 = store.prune_older_than_days(30).await.unwrap();
        assert_eq!(n2, 0);
    }

    #[tokio::test]
    async fn find_by_change_id_returns_only_shelved() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let src = write_file(dir.path(), "x.txt", b"x");
        let change_id = Uuid::new_v4();
        let shelved = store.shelve(&sample_indexed(src), change_id).await.unwrap();

        let found = store.find_by_change_id(change_id).await.unwrap().unwrap();
        assert_eq!(found.id, shelved.id);

        // After restore, status flips to Unshelved and lookup returns None.
        store.restore(&shelved).await.unwrap();
        assert!(store.find_by_change_id(change_id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn find_by_change_id_returns_none_when_missing() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        assert!(store
            .find_by_change_id(Uuid::new_v4())
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn prune_respects_cutoff() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let fresh = write_file(dir.path(), "fresh.txt", b"x");
        store
            .shelve(&sample_indexed(fresh.clone()), Uuid::new_v4())
            .await
            .unwrap();
        let n = store.prune_older_than_days(30).await.unwrap();
        assert_eq!(n, 0, "recently-shelved backup must not be pruned");
    }
}
