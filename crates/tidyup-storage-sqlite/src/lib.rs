//! `SQLite` implementations of the storage port traits.
//!
//! One `SqliteStore` owns a single `Arc<Mutex<Connection>>` and implements `FileIndex`,
//! `ChangeLog`, and `BackupStore`. rusqlite is synchronous — every trait method wraps the
//! locked connection in `tokio::task::spawn_blocking` so the async runtime isn't blocked.
//!
//! On `open`:
//! - `PRAGMA journal_mode = WAL` for concurrent readers during writes.
//! - `PRAGMA foreign_keys = ON` (per-connection) to honor `ON DELETE CASCADE`.
//! - Migrations applied idempotently.
//!
//! `BackupStore` methods need a shelf directory; set it with [`SqliteStore::with_backup_root`].
//! Calling a backup method without one configured returns an error.

// Every query holds the connection mutex for the duration of the spawn_blocking task.
// Clippy's "inline the lock" suggestion breaks multi-statement queries where a
// Statement borrows the guard across prepare + query_row/query_map.
#![allow(clippy::significant_drop_tightening)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::Connection;

mod backups;
mod changes;
mod files;
pub mod indexer;
mod runs;
mod schema;

pub use indexer::index_directory;

/// Default storage backend. Cheaply cloneable — the connection is shared.
#[derive(Clone, Debug)]
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    backup_root: Option<Arc<PathBuf>>,
}

impl SqliteStore {
    /// Open (or create) a database at `path`, run migrations, and enable WAL + FK enforcement.
    pub fn open(path: &Path) -> Result<Self> {
        let mut conn = Connection::open(path)
            .with_context(|| format!("opening sqlite database at {}", path.display()))?;
        configure(&conn)?;
        schema::apply(&mut conn).context("applying schema migrations")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            backup_root: None,
        })
    }

    /// In-memory store, useful for tests. No WAL (not supported for `:memory:`).
    pub fn open_in_memory() -> Result<Self> {
        let mut conn = Connection::open_in_memory().context("opening in-memory sqlite")?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        schema::apply(&mut conn).context("applying schema migrations")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            backup_root: None,
        })
    }

    /// Configure where shelved backups live on disk. Required before any `BackupStore` call.
    #[must_use]
    pub fn with_backup_root(mut self, root: PathBuf) -> Self {
        self.backup_root = Some(Arc::new(root));
        self
    }

    fn conn(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }

    fn backup_root(&self) -> Option<Arc<PathBuf>> {
        self.backup_root.as_ref().map(Arc::clone)
    }
}

fn configure(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;\n\
         PRAGMA foreign_keys = ON;\n\
         PRAGMA synchronous = NORMAL;",
    )
    .context("configuring sqlite pragmas")?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn open_in_memory_runs_migrations() {
        let store = SqliteStore::open_in_memory().unwrap();
        let count: i64 = {
            let conn = store.conn.lock().unwrap();
            conn.query_row(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name IN \
                 ('files', 'runs', 'bundles', 'change_proposals', 'backups')",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(count, 5);
    }

    #[test]
    fn foreign_keys_enabled() {
        let store = SqliteStore::open_in_memory().unwrap();
        let on: i64 = {
            let conn = store.conn.lock().unwrap();
            conn.query_row("PRAGMA foreign_keys", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(on, 1);
    }
}
