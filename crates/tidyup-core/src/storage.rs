//! Storage backend ports — file index, change log, backup store.

use std::path::Path;

use async_trait::async_trait;
use tidyup_domain::{BackupRecord, ChangeProposal, FileId, IndexedFile};

use crate::Result;

/// Persistent index of observed files. Default impl: `SQLite` (`tidyup-storage-sqlite`).
#[async_trait]
pub trait FileIndex: Send + Sync {
    async fn upsert(&self, record: &IndexedFile) -> Result<()>;
    async fn get(&self, id: &FileId) -> Result<Option<IndexedFile>>;
    async fn by_path(&self, path: &Path) -> Result<Option<IndexedFile>>;
    async fn list_under(&self, root: &Path) -> Result<Vec<IndexedFile>>;
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
    async fn shelve(&self, file: &IndexedFile) -> Result<BackupRecord>;
    async fn restore(&self, record: &BackupRecord) -> Result<()>;
    async fn prune_older_than_days(&self, days: u32) -> Result<usize>;
}
