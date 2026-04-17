//! Storage backend ports — file index, change log, backup store.

use std::path::Path;

use async_trait::async_trait;
use tidyup_domain::{BackupRecord, BundleProposal, ChangeProposal, FileId, IndexedFile};
use uuid::Uuid;

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
///
/// Bundles are first-class: a `BundleProposal` and all its members are recorded together in a
/// single transaction, and [`pending`](Self::pending) exposes **loose** proposals only (those
/// not belonging to a bundle). Bundle members are never surfaced outside their bundle — use
/// [`pending_bundles`](Self::pending_bundles) to retrieve them.
#[async_trait]
pub trait ChangeLog: Send + Sync {
    /// Record a loose (non-bundle) proposal. Callers should not pass proposals with
    /// `bundle_id == Some(_)` here — those flow through [`record_bundle`](Self::record_bundle).
    async fn record_proposal(&self, proposal: &ChangeProposal) -> Result<()>;
    async fn mark_applied(&self, proposal_id: Uuid) -> Result<()>;
    /// Pending **loose** proposals only (`bundle_id IS NULL`).
    async fn pending(&self) -> Result<Vec<ChangeProposal>>;

    /// Record a bundle and all its members atomically.
    async fn record_bundle(&self, bundle: &BundleProposal) -> Result<()>;
    /// Mark a bundle and all its member proposals as applied, atomically.
    async fn mark_bundle_applied(&self, bundle_id: Uuid) -> Result<()>;
    /// Pending bundles with their members hydrated.
    async fn pending_bundles(&self) -> Result<Vec<BundleProposal>>;
}

/// Backup store — shelf-style temporary storage for rollback.
///
/// Bundles shelve as a single subtree so rollback is atomic with the bundle itself — never
/// per-member. The returned [`BackupRecord::change_id`] holds the originating
/// [`ChangeProposal::id`] for [`shelve`](Self::shelve) and the
/// [`BundleProposal::id`] for [`shelve_bundle`](Self::shelve_bundle).
#[async_trait]
pub trait BackupStore: Send + Sync {
    /// Copy the original file to the shelf and record the backup. Called before a single-file
    /// move is applied; if it fails, the move must not proceed.
    async fn shelve(&self, file: &IndexedFile, change_id: Uuid) -> Result<BackupRecord>;

    /// Copy the entire bundle subtree (recursive) to the shelf and record one backup. Called
    /// before a bundle move is applied. The returned record's `backup_path` points at the
    /// shelved subtree root.
    async fn shelve_bundle(&self, root: &Path, bundle_id: Uuid) -> Result<BackupRecord>;

    /// Restore a shelved backup to `original_path`. Handles files and subtrees uniformly.
    async fn restore(&self, record: &BackupRecord) -> Result<()>;

    /// Expire backups older than `days`: marks rows as [`BackupStatus::Expired`] and best-effort
    /// removes the shelved content from disk. Returns the number of records expired.
    ///
    /// [`BackupStatus::Expired`]: tidyup_domain::BackupStatus::Expired
    async fn prune_older_than_days(&self, days: u32) -> Result<usize>;
}
