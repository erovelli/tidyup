//! CLI [`ProgressReporter`] impl — wraps `indicatif` `MultiProgress` for human mode,
//! or emits line-delimited JSON events for `--json` mode.

use async_trait::async_trait;
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_domain::Phase;

pub(crate) struct CliReporter {
    // Read once indicatif wiring lands; kept to lock in the public shape.
    #[allow(dead_code)]
    pub(crate) json: bool,
    // TODO: multi: indicatif::MultiProgress, bars per phase
}

#[async_trait]
impl ProgressReporter for CliReporter {
    async fn phase_started(&self, _phase: Phase, _total: Option<u64>) {}
    async fn item_completed(&self, _phase: Phase, _item: ProgressItem) {}
    async fn phase_finished(&self, _phase: Phase) {}
    async fn message(&self, _level: Level, _msg: &str) {}
}
