//! Storage backend ports — file index, change log, backup store, run log.

use std::path::Path;

use async_trait::async_trait;
use tidyup_domain::{
    BackupRecord, BundleProposal, ChangeProposal, FileId, IndexedFile, RestorePrecheck, RunRecord,
    RunState,
};
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
///
/// Proposals may be associated with a [`RunRecord`](tidyup_domain::RunRecord) via
/// `run_id`. When provided, rollback can enumerate every applied change for that run.
#[async_trait]
pub trait ChangeLog: Send + Sync {
    /// Record a loose (non-bundle) proposal. Callers should not pass proposals with
    /// `bundle_id == Some(_)` here — those flow through [`record_bundle`](Self::record_bundle).
    ///
    /// `run_id` optionally ties this proposal to a run record for rollback lookup.
    async fn record_proposal(&self, proposal: &ChangeProposal, run_id: Option<Uuid>) -> Result<()>;
    async fn mark_applied(&self, proposal_id: Uuid) -> Result<()>;
    /// Record a review rejection so the proposal leaves the pending set. Without
    /// this a rejected proposal stays `Pending` forever and re-surfaces from
    /// [`pending`](Self::pending) on every run. A no-op for unknown ids.
    async fn mark_rejected(&self, proposal_id: Uuid) -> Result<()>;
    /// Mark a proposal as rolled back (originals restored from the shelf).
    async fn mark_unshelved(&self, proposal_id: Uuid) -> Result<()>;
    /// Pending **loose** proposals only (`bundle_id IS NULL`).
    async fn pending(&self) -> Result<Vec<ChangeProposal>>;

    /// Record a bundle and all its members atomically.
    async fn record_bundle(&self, bundle: &BundleProposal, run_id: Option<Uuid>) -> Result<()>;
    /// Mark a bundle and all its member proposals as applied, atomically.
    async fn mark_bundle_applied(&self, bundle_id: Uuid) -> Result<()>;
    /// Mark a bundle and every member as rolled back (originals restored).
    async fn mark_bundle_unshelved(&self, bundle_id: Uuid) -> Result<()>;
    /// Pending bundles with their members hydrated.
    async fn pending_bundles(&self) -> Result<Vec<BundleProposal>>;

    /// Applied (not-yet-rolled-back) loose proposals that belong to `run_id`.
    async fn applied_proposals_for_run(&self, run_id: Uuid) -> Result<Vec<ChangeProposal>>;

    /// Applied (not-yet-rolled-back) bundles that belong to `run_id`, with members hydrated.
    async fn applied_bundles_for_run(&self, run_id: Uuid) -> Result<Vec<BundleProposal>>;
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

    /// Decide whether `record`'s destination can be safely deleted and restored.
    ///
    /// Rollback must never destroy data: this recomputes the shelf copy's digest
    /// (rejecting a missing or corrupt shelf) and compares the live `destination`
    /// against it, so a destination edited after the move is reported as a
    /// conflict instead of being clobbered. `destination` is the path the change
    /// moved the original *to* (a file for loose proposals, a subtree root for
    /// directory bundles). Owning the algorithm here keeps the rollback caller
    /// from re-deriving the tree-hash.
    async fn precheck_restore(
        &self,
        record: &BackupRecord,
        destination: &Path,
    ) -> Result<RestorePrecheck>;

    /// Look up the backup record for a given change or bundle id.
    ///
    /// Returns the single most-recent record with matching `change_id`
    /// **regardless of status**, or `None` if the change was never shelved.
    /// Callers branch on [`BackupRecord::status`]: `Shelved` is restorable;
    /// `Unshelved` means already restored (a retrying rollback treats it as
    /// done); `Expired` means the shelf content was pruned.
    async fn find_by_change_id(&self, change_id: Uuid) -> Result<Option<BackupRecord>>;

    /// Expire backups older than `days`: marks rows as [`BackupStatus::Expired`] and best-effort
    /// removes the shelved content from disk. Returns the number of records expired.
    ///
    /// [`BackupStatus::Expired`]: tidyup_domain::BackupStatus::Expired
    async fn prune_older_than_days(&self, days: u32) -> Result<usize>;
}

/// Persistent log of scan/migration runs. Drives `rollback <run_id>` and `list-runs`.
#[async_trait]
pub trait RunLog: Send + Sync {
    /// Record a newly-begun run (usually with `state = InProgress`).
    async fn record_run(&self, run: &RunRecord) -> Result<()>;
    /// Update a run's terminal state (`Completed` / `RolledBack` / `Failed`) and set `completed_at`.
    async fn finish_run(&self, run_id: Uuid, state: RunState) -> Result<()>;
    /// Look up a run by id. Returns `None` if unknown.
    async fn get_run(&self, run_id: Uuid) -> Result<Option<RunRecord>>;
    /// List runs, most recently started first.
    async fn list_runs(&self) -> Result<Vec<RunRecord>>;
}
