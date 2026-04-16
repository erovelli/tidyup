//! Storage backend ports — file index, change log, backup store.

use std::path::Path;

use async_trait::async_trait;
use tidyup_domain::{BackupRecord, ChangeProposal, FileId, FileRecord};

use crate::Result;

/// Persistent index of observed files. Default impl: `SQLite` (`tidyup-storage-sqlite`).
#[async_trait]
pub trait FileIndex: Send + Sync {
    async fn upsert(&self, record: &FileRecord) -> Result<()>;
    async fn get(&self, id: &FileId) -> Result<Option<FileRecord>>;
    async fn by_path(&self, path: &Path) -> Result<Option<FileRecord>>;
    async fn list_under(&self, root: &Path) -> Result<Vec<FileRecord>>;
}

/// Append-only log of proposed and applied changes. Drives diff view + audit trail.
#[async_trait]
pub trait ChangeLog: Send + Sync {
    async fn record_proposal(&self, proposal: &ChangeProposal) -> Result<()>;
    async fn mark_applied(&self, proposal_id: uuid::Uuid) -> Result<()>;
    async fn pending(&self) -> Result<Vec<ChangeProposal>>;
}

/// Backup store — shelf-style temporary storage for rollback.
#[async_trait]
pub trait BackupStore: Send + Sync {
    async fn shelve(&self, file: &FileRecord) -> Result<BackupRecord>;
    async fn restore(&self, record: &BackupRecord) -> Result<()>;
    async fn prune_older_than_days(&self, days: u32) -> Result<usize>;
}
