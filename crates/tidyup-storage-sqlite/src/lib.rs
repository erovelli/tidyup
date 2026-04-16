//! `SQLite` implementations of the storage port traits.
//!
//! Uses WAL mode, upsert-on-path-conflict. Ports: `FileIndex`, `ChangeLog`, `BackupStore`.

// TODO: SqliteFileIndex, SqliteChangeLog, FsBackupStore (shelf directory + retention).
