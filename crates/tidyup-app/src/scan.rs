//! Scan service — classify files in place against a fixed or config-loaded taxonomy.
//!
//! Contrast with `migration`: scan targets a *taxonomy* (categories); migration targets
//! an existing folder *hierarchy*. Both produce `ChangeProposal`s that use the same
//! review flow.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tidyup_core::{ProgressReporter, Result, ReviewHandler};

use crate::ServiceContext;

#[allow(missing_debug_implementations)]
pub struct ScanService {
    ctx: Arc<ServiceContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanRequest {
    pub root: std::path::PathBuf,
    pub taxonomy_path: Option<std::path::PathBuf>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    pub scanned: usize,
    pub classified: usize,
    pub applied: usize,
    pub run_id: uuid::Uuid,
}

impl ScanService {
    pub fn new(ctx: Arc<ServiceContext>) -> Self {
        Self { ctx }
    }

    pub async fn run(
        &self,
        request: ScanRequest,
        progress: &dyn ProgressReporter,
        review: &dyn ReviewHandler,
    ) -> Result<ScanReport> {
        let _ = (&self.ctx, request, progress, review);
        anyhow::bail!("not yet implemented")
    }

    pub async fn classify_one(&self, file: &Path) -> Result<tidyup_domain::ChangeProposal> {
        let _ = (&self.ctx, file);
        anyhow::bail!("not yet implemented")
    }
}
