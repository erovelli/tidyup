//! Rollback service — restore a previous migration run from backup shelf.

use std::sync::Arc;

use tidyup_core::{ProgressReporter, Result};

use crate::ServiceContext;

#[allow(missing_debug_implementations)]
pub struct RollbackService {
    ctx: Arc<ServiceContext>,
}

impl RollbackService {
    pub fn new(ctx: Arc<ServiceContext>) -> Self {
        Self { ctx }
    }

    pub async fn rollback_run(
        &self,
        run_id: uuid::Uuid,
        progress: &dyn ProgressReporter,
    ) -> Result<usize> {
        let _ = (&self.ctx, run_id, progress);
        anyhow::bail!("not yet implemented")
    }

    pub async fn list_runs(&self) -> Result<Vec<uuid::Uuid>> {
        anyhow::bail!("not yet implemented")
    }
}
