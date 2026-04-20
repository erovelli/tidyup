//! Dioxus [`ProgressReporter`] impl.
//!
//! Mirrors the CLI's `indicatif`-backed reporter in shape, but writes phase
//! and progress events into [`SignalBundle`] fields instead. Any component
//! subscribed to the relevant signal re-renders on update — the same contract
//! as indicatif's progress bars, just via Dioxus' reactivity.
//!
//! Implementation notes:
//!
//! * Writes happen from the tokio task the service runs on. `SyncSignal<T>`
//!   is `Send + Sync` when `T: Send + Sync`, so cross-task writes are sound.
//! * `messages` is capped at 200 lines so a chatty run doesn't grow the
//!   retained set unboundedly — the head is dropped, not the tail.
//! * [`Phase::AwaitingReview`] is a pure status marker; the actual review
//!   transition is driven by [`DioxusReviewHandler`](crate::review).
#![allow(clippy::large_types_passed_by_value)]

use async_trait::async_trait;
use dioxus::prelude::*;
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_domain::Phase;

use crate::state::{LogMessage, SignalBundle};

const MAX_MESSAGES: usize = 200;

#[allow(missing_debug_implementations)]
pub(crate) struct DioxusReporter {
    signals: SignalBundle,
}

impl DioxusReporter {
    pub(crate) const fn new(signals: SignalBundle) -> Self {
        Self { signals }
    }
}

#[async_trait]
impl ProgressReporter for DioxusReporter {
    async fn phase_started(&self, phase: Phase, total: Option<u64>) {
        let mut phase_sig = self.signals.phase;
        let mut current = self.signals.progress_current;
        let mut total_sig = self.signals.progress_total;
        let mut label = self.signals.progress_label;
        phase_sig.set(Some(phase));
        current.set(0);
        total_sig.set(total);
        label.set(String::new());
    }

    async fn item_completed(&self, _phase: Phase, item: ProgressItem) {
        let mut current = self.signals.progress_current;
        let mut total_sig = self.signals.progress_total;
        let mut label = self.signals.progress_label;
        current.set(item.current);
        if item.total.is_some() {
            total_sig.set(item.total);
        }
        label.set(item.label);
    }

    async fn phase_finished(&self, _phase: Phase) {
        // Leave the phase signal set to the last value; the UI resets it when
        // the service call returns. We just clear the transient label.
        let mut label = self.signals.progress_label;
        label.set(String::new());
    }

    async fn message(&self, level: Level, msg: &str) {
        let mut messages = self.signals.messages;
        let entry = LogMessage {
            level,
            text: msg.to_string(),
        };
        messages.with_mut(|list| {
            list.push(entry);
            if list.len() > MAX_MESSAGES {
                let drop_n = list.len() - MAX_MESSAGES;
                list.drain(0..drop_n);
            }
        });
    }
}
