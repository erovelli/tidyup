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
use tidyup_domain::{BackupRecord, BackupStatus, IndexedFile, RestorePrecheck};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::SqliteStore;

const BACKUP_COLS: &str =
    "id, change_id, original_path, backup_path, shelved_at, unshelved_at, status, content_hash";

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
        content_hash: row.get::<_, Option<String>>("content_hash")?,
    })
}

fn insert_backup(tx: &Transaction<'_>, r: &BackupRecord) -> Result<()> {
    tx.execute(
        &format!(
            "INSERT INTO backups ({BACKUP_COLS}) VALUES \
             (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"
        ),
        params![
            r.id.to_string(),
            r.change_id.to_string(),
            path_str(&r.original_path)?,
            path_str(&r.backup_path)?,
            r.shelved_at,
            r.unshelved_at,
            r.status.as_str(),
            r.content_hash,
        ],
    )
    .context("inserting backup record")?;
    Ok(())
}

/// Streaming BLAKE3 of a single file. Reads in fixed-size chunks so a multi-GB
/// shelf entry never lands in memory at once.
fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open for hashing {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read for hashing {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Canonical content digest of a directory subtree: every entry's relative
/// path plus a kind-tagged content token, sorted and folded into one hash.
/// Order-independent and stable across the shelf copy and the live destination
/// (both hashed relative to their own root), so a bundle whose destination was
/// edited after apply produces a different digest.
///
/// Entry semantics mirror how [`copy_dir_recursive`] materialises a shelf copy:
/// - Anything that resolves to a file (`Path::is_file`, which **follows
///   symlinks** — just like the `std::fs::copy` used at shelve time) hashes as
///   `F:<blake3 of followed content>`. A symlink-to-file at the destination
///   therefore matches the regular file the shelf stored for it.
/// - Directories (including symlinks to directories) contribute a `D` token,
///   so a user-added empty directory changes the digest.
/// - Anything else (e.g. a broken symlink) contributes an `L` token, so its
///   appearance or disappearance is also detected.
fn tree_hash(root: &Path) -> Result<String> {
    let mut entries: Vec<(String, String)> = Vec::new();
    for entry in WalkDir::new(root).min_depth(1).sort_by_file_name() {
        let entry = entry.with_context(|| format!("walking {}", root.display()))?;
        let path = entry.path();
        let relative = path
            .strip_prefix(root)
            .context("strip_prefix during tree hash")?;
        // Normalise separators so the digest is stable regardless of platform.
        let rel = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let token = if path.is_file() {
            format!("F:{}", hash_file(path)?)
        } else if path.is_dir() {
            "D".to_string()
        } else {
            "L".to_string()
        };
        entries.push((rel, token));
    }
    entries.sort();
    let mut hasher = blake3::Hasher::new();
    for (rel, token) in entries {
        hasher.update(rel.as_bytes());
        hasher.update(b"\0");
        hasher.update(token.as_bytes());
        hasher.update(b"\n");
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Content digest of a path, dispatching to [`tree_hash`] for directories and
/// [`hash_file`] for files. This is the single algorithm shared by shelve-time
/// recording and rollback-time verification.
fn hash_path(path: &Path) -> Result<String> {
    if path.is_dir() {
        tree_hash(path)
    } else {
        hash_file(path)
    }
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
            // Hash the shelved copy (identical bytes to the original, and the
            // exact bytes rollback verifies against later).
            let content_hash = Some(hash_file(&backup_path)?);
            let record = BackupRecord {
                id: Uuid::new_v4(),
                change_id,
                original_path,
                backup_path,
                shelved_at,
                unshelved_at: None,
                status: BackupStatus::Shelved,
                content_hash,
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
            let content_hash = Some(tree_hash(&backup_path)?);
            let record = BackupRecord {
                id: Uuid::new_v4(),
                change_id: bundle_id,
                original_path,
                backup_path,
                shelved_at,
                unshelved_at: None,
                status: BackupStatus::Shelved,
                content_hash,
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
            // Latest record regardless of status: callers branch on
            // `record.status` — a partially-rolled-back bundle needs to see its
            // already-`Unshelved` members to retry to completion instead of
            // erroring on "no shelved backup" forever.
            let mut stmt = guard.prepare(&format!(
                "SELECT {BACKUP_COLS} FROM backups \
                 WHERE change_id = ?1 \
                 ORDER BY shelved_at DESC LIMIT 1"
            ))?;
            let fetched = stmt
                .query_row(params![change_id.to_string()], row_to_backup)
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

    async fn precheck_restore(
        &self,
        record: &BackupRecord,
        destination: &Path,
    ) -> tidyup_core::Result<RestorePrecheck> {
        let record = record.clone();
        let destination = destination.to_path_buf();
        let verdict = tokio::task::spawn_blocking(move || -> Result<RestorePrecheck> {
            // 1. The shelf copy must exist — without it, restore has nothing to
            //    put back, so deleting the destination would lose data outright.
            if !record.backup_path.exists() {
                return Ok(RestorePrecheck::ShelfUnusable(format!(
                    "shelf copy missing: {}",
                    record.backup_path.display()
                )));
            }
            // 2. The shelf copy must be intact. Recompute its digest; if a
            //    stored digest exists and disagrees, the shelf was corrupted and
            //    can't be trusted as the restore source.
            let shelf_hash = hash_path(&record.backup_path)
                .with_context(|| format!("hashing shelf {}", record.backup_path.display()))?;
            if let Some(stored) = &record.content_hash {
                if stored != &shelf_hash {
                    return Ok(RestorePrecheck::ShelfUnusable(format!(
                        "shelf copy corrupt (digest changed): {}",
                        record.backup_path.display()
                    )));
                }
            }
            // 3. The original slot must be free or already hold the shelved
            //    content. Restore copies shelf → original unconditionally, so a
            //    NEW file the user saved into the vacated slot would be
            //    overwritten — refuse instead. (An original that matches the
            //    shelf is fine: re-restoring it is a no-op.)
            if record.original_path.exists() {
                let original_hash = hash_path(&record.original_path).with_context(|| {
                    format!("hashing original slot {}", record.original_path.display())
                })?;
                if original_hash != shelf_hash {
                    return Ok(RestorePrecheck::OriginalOccupied);
                }
            }
            // 4. If the destination is already gone, there is nothing to clobber
            //    — restoring the shelf copy is safe.
            if !destination.exists() {
                return Ok(RestorePrecheck::Ready);
            }
            // 5. Compare the live destination against the shelved bytes. A match
            //    means the move's target is untouched and safe to replace; a
            //    mismatch means the user edited it post-apply — never delete.
            let dest_hash = hash_path(&destination)
                .with_context(|| format!("hashing destination {}", destination.display()))?;
            if dest_hash == shelf_hash {
                Ok(RestorePrecheck::Ready)
            } else {
                Ok(RestorePrecheck::DestinationModified)
            }
        })
        .await
        .context("join precheck_restore task")??;
        Ok(verdict)
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
    async fn find_by_change_id_returns_latest_record_any_status() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);

        let src = write_file(dir.path(), "x.txt", b"x");
        let change_id = Uuid::new_v4();
        let shelved = store.shelve(&sample_indexed(src), change_id).await.unwrap();

        let found = store.find_by_change_id(change_id).await.unwrap().unwrap();
        assert_eq!(found.id, shelved.id);
        assert_eq!(found.status, BackupStatus::Shelved);

        // After restore the record stays visible with status Unshelved, so a
        // retrying rollback can recognise already-restored members instead of
        // erroring on "no shelved backup" forever.
        store.restore(&shelved).await.unwrap();
        let after = store.find_by_change_id(change_id).await.unwrap().unwrap();
        assert_eq!(after.id, shelved.id);
        assert_eq!(after.status, BackupStatus::Unshelved);
    }

    #[tokio::test]
    async fn shelve_records_content_hash() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "h.txt", b"hash me");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        assert_eq!(
            record.content_hash,
            Some(blake3::hash(b"hash me").to_hex().to_string()),
        );
    }

    #[tokio::test]
    async fn precheck_ready_when_destination_matches_shelf() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        // The "destination" is a moved copy with identical bytes.
        let dest = write_file(dir.path(), "moved.txt", b"payload");
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::Ready,
        );
    }

    #[tokio::test]
    async fn precheck_flags_edited_destination() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        let dest = write_file(dir.path(), "moved.txt", b"payload EDITED");
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::DestinationModified,
        );
    }

    #[tokio::test]
    async fn precheck_ready_when_destination_missing() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        let dest = dir.path().join("gone.txt");
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::Ready,
        );
    }

    #[tokio::test]
    async fn precheck_reports_missing_shelf() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        std::fs::remove_file(&record.backup_path).unwrap();
        let dest = write_file(dir.path(), "moved.txt", b"payload");
        assert!(matches!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::ShelfUnusable(_),
        ));
    }

    #[tokio::test]
    async fn precheck_reports_corrupt_shelf() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        // Corrupt the shelf copy so its recomputed digest no longer matches.
        std::fs::write(&record.backup_path, b"tampered").unwrap();
        let dest = write_file(dir.path(), "moved.txt", b"payload");
        assert!(matches!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::ShelfUnusable(_),
        ));
    }

    #[tokio::test]
    async fn precheck_flags_new_file_in_original_slot() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src.clone()), Uuid::new_v4())
            .await
            .unwrap();
        // Simulate the apply-move, then the user saving a NEW file into the
        // vacated original slot.
        let dest = write_file(dir.path(), "moved.txt", b"payload");
        std::fs::write(&src, b"BRAND NEW FILE, not the original").unwrap();
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::OriginalOccupied,
        );
    }

    #[tokio::test]
    async fn precheck_ready_when_original_slot_matches_shelf() {
        // Re-restoring over an original that already holds the shelved content
        // is a harmless no-op, not a conflict.
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let src = write_file(dir.path(), "orig.txt", b"payload");
        let record = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        // Original still in place (as after a healed partial rollback).
        let dest = dir.path().join("moved.txt");
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::Ready,
        );
    }

    #[tokio::test]
    async fn precheck_detects_empty_dir_added_to_bundle_destination() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let root = dir.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.txt"), b"a").unwrap();
        let record = store.shelve_bundle(&root, Uuid::new_v4()).await.unwrap();
        std::fs::remove_dir_all(&root).unwrap(); // simulate the move

        let dest = dir.path().join("moved");
        copy_dir_recursive(&record.backup_path, &dest).unwrap();
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::Ready,
        );
        // A user-created empty directory inside the moved bundle must flip the
        // verdict — deleting the destination would destroy it.
        std::fs::create_dir(dest.join("notes")).unwrap();
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::DestinationModified,
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn precheck_ready_for_pristine_bundle_containing_symlink() {
        // shelve_bundle materialises a symlink-to-file as a regular file
        // (fs::copy follows links); the destination keeps the symlink after a
        // same-volume rename. tree_hash follows links the same way fs::copy
        // does, so a pristine destination must precheck Ready, not
        // DestinationModified-forever.
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let root = dir.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("real.txt"), b"linked content").unwrap();
        std::os::unix::fs::symlink(root.join("real.txt"), root.join("link.txt")).unwrap();

        let record = store.shelve_bundle(&root, Uuid::new_v4()).await.unwrap();
        // Simulate the same-volume rename apply: the destination subtree keeps
        // the symlink exactly as the source had it.
        let dest = dir.path().join("moved");
        std::fs::rename(&root, &dest).unwrap();
        // Repair the link target (it pointed into the old root).
        std::fs::remove_file(dest.join("link.txt")).unwrap();
        std::os::unix::fs::symlink(dest.join("real.txt"), dest.join("link.txt")).unwrap();

        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::Ready,
        );
    }

    #[tokio::test]
    async fn legacy_database_gains_content_hash_column_and_precheck_falls_back() {
        // A database created before content hashing shipped must (a) migrate
        // via ALTER TABLE on open and (b) still precheck correctly for its old
        // records (content_hash = NULL → compare destination against the shelf
        // bytes directly).
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("legacy.db");
        {
            let conn = rusqlite::Connection::open(&db).unwrap();
            // The pre-WP0 backups schema, verbatim minus content_hash.
            conn.execute_batch(
                "CREATE TABLE backups (
                    id             TEXT PRIMARY KEY,
                    change_id      TEXT NOT NULL,
                    original_path  TEXT NOT NULL,
                    backup_path    TEXT NOT NULL,
                    shelved_at     TEXT NOT NULL,
                    unshelved_at   TEXT,
                    status         TEXT NOT NULL
                );",
            )
            .unwrap();
            // A legacy row written by the old code.
            let shelf_file = dir.path().join("legacy-shelf.txt");
            std::fs::write(&shelf_file, b"legacy bytes").unwrap();
            conn.execute(
                "INSERT INTO backups (id, change_id, original_path, backup_path, shelved_at, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    Uuid::new_v4().to_string(),
                    Uuid::new_v4().to_string(),
                    dir.path().join("orig.txt").to_str().unwrap(),
                    shelf_file.to_str().unwrap(),
                    Utc::now(),
                    "Shelved",
                ],
            )
            .unwrap();
        }

        // Re-open through SqliteStore: the additive migration must fire.
        let store = SqliteStore::open(&db)
            .unwrap()
            .with_backup_root(dir.path().join("shelf"));
        std::fs::create_dir_all(dir.path().join("shelf")).unwrap();

        // The legacy row hydrates with content_hash = None...
        let conn = store.conn();
        let record = tokio::task::spawn_blocking(move || {
            let guard = conn.lock().unwrap();
            let mut stmt = guard
                .prepare(&format!("SELECT {BACKUP_COLS} FROM backups LIMIT 1"))
                .unwrap();
            stmt.query_row([], row_to_backup).unwrap()
        })
        .await
        .unwrap();
        assert_eq!(record.content_hash, None);

        // ...and precheck falls back to direct shelf-vs-destination compare.
        let matching = write_file(dir.path(), "dest-match.txt", b"legacy bytes");
        assert_eq!(
            store.precheck_restore(&record, &matching).await.unwrap(),
            RestorePrecheck::Ready,
        );
        let edited = write_file(dir.path(), "dest-edit.txt", b"edited after apply");
        assert_eq!(
            store.precheck_restore(&record, &edited).await.unwrap(),
            RestorePrecheck::DestinationModified,
        );

        // New shelves into the migrated DB record a hash (8-param insert works).
        let src = write_file(dir.path(), "new.txt", b"new");
        let rec2 = store
            .shelve(&sample_indexed(src), Uuid::new_v4())
            .await
            .unwrap();
        assert!(rec2.content_hash.is_some());
    }

    #[tokio::test]
    async fn precheck_detects_edited_bundle_member() {
        let dir = TempDir::new().unwrap();
        let store = store_with_backup_root(&dir);
        let root = dir.path().join("proj");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("Cargo.toml"), b"[package]").unwrap();
        std::fs::write(root.join("src/main.rs"), b"fn main() {}").unwrap();
        let record = store.shelve_bundle(&root, Uuid::new_v4()).await.unwrap();

        // A destination subtree that mirrors the shelf exactly → Ready.
        let dest = dir.path().join("moved");
        copy_dir_recursive(&record.backup_path, &dest).unwrap();
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::Ready,
        );
        // Edit a member → DestinationModified.
        std::fs::write(dest.join("src/main.rs"), b"fn main() { edited() }").unwrap();
        assert_eq!(
            store.precheck_restore(&record, &dest).await.unwrap(),
            RestorePrecheck::DestinationModified,
        );
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
