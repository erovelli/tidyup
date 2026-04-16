//! CLI [`ReviewHandler`](tidyup_core::frontend::ReviewHandler) impls:
//! - [`AutoApproveHandler`] — `--yes` or `--min-confidence` threshold.
//! - [`InteractiveHandler`] — prompt-per-file (or `ratatui` full-screen diff later).

use async_trait::async_trait;
use tidyup_core::{frontend::ReviewHandler, Result};
use tidyup_domain::{ChangeProposal, ReviewDecision};

pub(crate) struct AutoApproveHandler {
    pub(crate) min_confidence: f32,
}

#[async_trait]
impl ReviewHandler for AutoApproveHandler {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
        Ok(proposals
            .into_iter()
            .map(|p| {
                if p.confidence.0 >= self.min_confidence {
                    ReviewDecision::Approve(p.id)
                } else {
                    ReviewDecision::Reject(p.id)
                }
            })
            .collect())
    }
}

pub(crate) struct InteractiveHandler;

#[async_trait]
impl ReviewHandler for InteractiveHandler {
    async fn review(&self, _proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
        // TODO: prompt-per-file; later, ratatui diff view.
        Ok(Vec::new())
    }
}
