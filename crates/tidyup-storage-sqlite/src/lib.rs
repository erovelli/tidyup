//! `SQLite` implementations of the storage port traits.
//!
//! One `SqliteStore` owns a single `Arc<Mutex<Connection>>` and implements `FileIndex`,
//! `ChangeLog`, and (in a follow-up) `BackupStore`. rusqlite is synchronous — every trait
//! method wraps the locked connection in `tokio::task::spawn_blocking` so the async runtime
//! isn't blocked.
//!
//! On `open`:
//! - `PRAGMA journal_mode = WAL` for concurrent readers during writes.
//! - `PRAGMA foreign_keys = ON` (per-connection) to honor `ON DELETE CASCADE`.
//! - Migrations applied idempotently.

// Every query holds the connection mutex for the duration of the spawn_blocking task.
// Clippy's "inline the lock" suggestion breaks multi-statement queries where a
// Statement borrows the guard across prepare + query_row/query_map.
#![allow(clippy::significant_drop_tightening)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::Connection;

mod changes;
mod files;
mod schema;

/// Default storage backend. Cheaply cloneable — the connection is shared.
#[derive(Clone, Debug)]
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
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
        })
    }

    /// In-memory store, useful for tests. No WAL (not supported for `:memory:`).
    pub fn open_in_memory() -> Result<Self> {
        let mut conn = Connection::open_in_memory().context("opening in-memory sqlite")?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        schema::apply(&mut conn).context("applying schema migrations")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn conn(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
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
                 ('files', 'bundles', 'change_proposals', 'backups')",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(count, 4);
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
