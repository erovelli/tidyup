//! Frontend ports — the seam between application services and CLI/UI frontends.
//!
//! Both `tidyup-cli` and `tidyup-ui` implement these traits. Application services
//! never depend on a concrete frontend — they call through these trait objects, so
//! the same service logic powers both delivery mechanisms.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tidyup_domain::{ChangeProposal, Phase, ReviewDecision};

use crate::Result;

/// A single progress event emitted during a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressItem {
    pub label: String,
    pub current: u64,
    pub total: Option<u64>,
}

/// Severity of a message emitted to the frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

/// Streaming progress reporter. CLI wraps `indicatif`; UI updates Dioxus signals.
///
/// Implementations must be cheap to call from tight loops. They should not block
/// the caller on I/O — buffer or drop if necessary.
#[async_trait]
pub trait ProgressReporter: Send + Sync {
    async fn phase_started(&self, phase: Phase, total: Option<u64>);
    async fn item_completed(&self, phase: Phase, item: ProgressItem);
    async fn phase_finished(&self, phase: Phase);
    async fn message(&self, level: Level, msg: &str);
}

/// Review strategy: how the frontend gathers decisions on classification proposals.
///
/// Implementations:
/// - **CLI interactive** — ratatui or prompt-per-file (`tidyup migrate --interactive`)
/// - **CLI auto**        — accept-all, reject-below-threshold, etc. (`--yes`, `--min-confidence`)
/// - **UI**              — the diff-view page, returning the full decision set when user clicks Apply
#[async_trait]
pub trait ReviewHandler: Send + Sync {
    /// Present all proposals. Return one decision per proposal.
    ///
    /// Implementations may batch (UI: show all, collect) or stream (CLI: prompt per item).
    /// The contract is the same from the service's perspective.
    async fn review(&self, proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>>;
}

/// Supplies runtime configuration to services. Abstracts over file-backed config (CLI)
/// vs. in-memory/user-mutated config (UI settings page).
pub trait ConfigProvider: Send + Sync {
    fn get(&self, key: &str) -> Option<String>;
}
