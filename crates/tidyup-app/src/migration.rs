//! Migration service — moves files from a source tree into a user-defined target tree.
//!
//! Same handle drives CLI (`tidyup migrate`) and UI ("Migrate" button).

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::{ProgressReporter, Result, ReviewHandler};

use crate::ServiceContext;

#[allow(missing_debug_implementations)] // ctx holds trait objects
pub struct MigrationService {
    ctx: Arc<ServiceContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRequest {
    pub source: std::path::PathBuf,
    pub target: std::path::PathBuf,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub proposed: usize,
    pub approved: usize,
    pub applied: usize,
    pub skipped: usize,
    pub failed: usize,
    pub run_id: uuid::Uuid,
}

impl MigrationService {
    pub fn new(ctx: Arc<ServiceContext>) -> Self {
        Self { ctx }
    }

    /// Run a full migration. This is the single entry point both frontends call.
    ///
    /// The service:
    /// 1. Indexes source tree (emits `Phase::Indexing` progress).
    /// 2. Scans target tree, builds folder profiles (`Phase::ProfilingTarget`).
    /// 3. Classifies each source file (`Phase::Classifying`).
    /// 4. Calls `review.review(proposals)` — frontend-specific UX.
    /// 5. Applies approved moves (`Phase::Applying`), unless `dry_run`.
    pub async fn run(
        &self,
        request: MigrationRequest,
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
    ) -> Result<MigrationReport> {
        let _ = (&self.ctx, request, progress, review);
        // TODO: delegate to tidyup_pipeline::migration::run(&self.ctx, ...)
        // Pipeline emits progress events via `progress`, hands proposals to `review`,
        // executes approvals via ctx.backup_store + filesystem.
        anyhow::bail!("not yet implemented")
    }

    /// Indexing-only pass. Useful for `tidyup status` or UI initial load.
    pub async fn index(&self, root: &Path, progress: &dyn ProgressReporter) -> Result<usize> {
        let _ = (&self.ctx, root, progress);
        anyhow::bail!("not yet implemented")
    }
}
