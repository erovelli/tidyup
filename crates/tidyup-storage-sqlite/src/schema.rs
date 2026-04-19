//! Schema + migrations. Applied once per connection on open.
//!
//! Design notes:
//! - UUIDs stored as TEXT (hyphenated, uppercase-insensitive).
//! - `DateTime<Utc>` mapped by rusqlite's `chrono` feature (ISO 8601).
//! - `BundleKind` stored as JSON so the `DocumentSeries { pattern }` payload roundtrips cleanly.
//! - Flat enums (`ChangeType`, `ChangeStatus`, `BackupStatus`) stored as their `as_str()`.
//! - `PRAGMA foreign_keys = ON` is *per connection* and is set in `SqliteStore::open`.

use rusqlite::Connection;

const CREATE_FILES: &str = r"
CREATE TABLE IF NOT EXISTS files (
    id            TEXT PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    extension     TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    content_hash  TEXT NOT NULL,
    indexed_at    TEXT NOT NULL
);
";

const CREATE_FILES_HASH_IDX: &str =
    "CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);";

const CREATE_RUNS: &str = r"
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    mode          TEXT NOT NULL,
    source_root   TEXT NOT NULL,
    target_root   TEXT,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    state         TEXT NOT NULL
);
";

const CREATE_BUNDLES: &str = r"
CREATE TABLE IF NOT EXISTS bundles (
    id             TEXT PRIMARY KEY,
    root           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    target_parent  TEXT NOT NULL,
    status         TEXT NOT NULL,
    reasoning      TEXT NOT NULL,
    confidence     REAL NOT NULL,
    created_at     TEXT NOT NULL,
    applied_at     TEXT,
    run_id         TEXT REFERENCES runs(id)
);
";

const CREATE_CHANGE_PROPOSALS: &str = r"
CREATE TABLE IF NOT EXISTS change_proposals (
    id                         TEXT PRIMARY KEY,
    file_id                    TEXT REFERENCES files(id),
    change_type                TEXT NOT NULL,
    original_path              TEXT NOT NULL,
    proposed_path              TEXT NOT NULL,
    proposed_name              TEXT NOT NULL,
    confidence                 REAL NOT NULL,
    reasoning                  TEXT NOT NULL,
    needs_review               INTEGER NOT NULL,
    status                     TEXT NOT NULL,
    created_at                 TEXT NOT NULL,
    applied_at                 TEXT,
    bundle_id                  TEXT REFERENCES bundles(id) ON DELETE CASCADE,
    classification_confidence  REAL,
    rename_mismatch_score      REAL,
    run_id                     TEXT REFERENCES runs(id)
);
";

const CREATE_CHANGES_BUNDLE_IDX: &str =
    "CREATE INDEX IF NOT EXISTS idx_changes_bundle_id ON change_proposals(bundle_id);";

const CREATE_CHANGES_STATUS_IDX: &str =
    "CREATE INDEX IF NOT EXISTS idx_changes_status ON change_proposals(status);";

const CREATE_CHANGES_RUN_IDX: &str =
    "CREATE INDEX IF NOT EXISTS idx_changes_run_id ON change_proposals(run_id);";

const CREATE_BUNDLES_RUN_IDX: &str =
    "CREATE INDEX IF NOT EXISTS idx_bundles_run_id ON bundles(run_id);";

const CREATE_BACKUPS: &str = r"
CREATE TABLE IF NOT EXISTS backups (
    id             TEXT PRIMARY KEY,
    change_id      TEXT NOT NULL,
    original_path  TEXT NOT NULL,
    backup_path    TEXT NOT NULL,
    shelved_at     TEXT NOT NULL,
    unshelved_at   TEXT,
    status         TEXT NOT NULL
);
";

/// Apply every migration inside a single transaction.
#[allow(clippy::redundant_pub_crate)]
pub(super) fn apply(conn: &mut Connection) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;
    tx.execute_batch(
        &[
            CREATE_FILES,
            CREATE_FILES_HASH_IDX,
            CREATE_RUNS,
            CREATE_BUNDLES,
            CREATE_CHANGE_PROPOSALS,
            CREATE_CHANGES_BUNDLE_IDX,
            CREATE_CHANGES_STATUS_IDX,
            CREATE_CHANGES_RUN_IDX,
            CREATE_BUNDLES_RUN_IDX,
            CREATE_BACKUPS,
        ]
        .join("\n"),
    )?;
    tx.commit()
}
