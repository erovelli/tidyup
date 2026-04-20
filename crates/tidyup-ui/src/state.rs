//! Shared reactive state for the Dioxus desktop UI.
//!
//! A single [`SharedState`] is held as context at the app root. It bundles:
//!
//! * The per-field signals components observe to render (`phase`, `messages`,
//!   `proposals`, …). These are `Copy` and trivially move into event handlers.
//! * A non-signal mutex that holds the [`oneshot::Sender`] the background
//!   `ScanService`/`MigrationService` is awaiting during review. The UI takes
//!   this sender when the user clicks *Apply* and sends the collected
//!   decisions, unblocking the service call.
//!
//! Signals use [`SyncStorage`] so the reporter and review handler — both
//! required to be `Send + Sync` by the `tidyup-core` port traits — can hold
//! the bundle and write to it from any tokio task.
//!
//! **Signal ownership**: every signal is created in [`ScopeId::ROOT`] via
//! [`Signal::new_maybe_sync_in_scope`]. This matters because long-running
//! services run via [`spawn_forever`], which attaches their tasks to the root
//! scope. Reading a signal from a non-descendant scope trips dioxus' lifetime
//! warnings and can (and does) lose updates. Anchoring at the root makes every
//! other scope a descendant.
//!
//! Keeping the review sender outside the signal graph avoids needing a
//! clonable trait object where a one-shot channel fits naturally.

use std::collections::HashMap;
use std::sync::Arc;

use dioxus::prelude::*;
use dioxus_core::ScopeId;
use tidyup_app::{MigrationReport, RollbackReport, ScanReport};
use tidyup_core::frontend::Level;
use tidyup_domain::{BundleProposal, ChangeProposal, Phase, ReviewDecision, RunRecord};
use tokio::sync::{oneshot, Mutex};
use uuid::Uuid;

/// A single message from `ProgressReporter::message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogMessage {
    pub(crate) level: Level,
    pub(crate) text: String,
}

/// Snapshot of the last completed `ScanService`/`MigrationService` run, used
/// to render the post-apply summary panel.
#[derive(Debug, Clone)]
pub(crate) enum LastReport {
    Scan(ScanReport),
    Migration(MigrationReport),
    Rollback(RollbackReport),
}

/// Which service is currently in flight (if any). Disables re-entry buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Busy {
    Idle,
    Scanning,
    Migrating,
    RollingBack,
}

/// Every [`SyncSignal`] the UI reads. `Copy` so event handlers and spawned
/// tasks can move it in cheaply; `SyncStorage` so writes from any thread are
/// safe, which is what the `Send + Sync` bound on the frontend ports requires.
#[derive(Clone, Copy, PartialEq)]
#[allow(missing_debug_implementations)]
pub(crate) struct SignalBundle {
    pub(crate) phase: SyncSignal<Option<Phase>>,
    pub(crate) progress_current: SyncSignal<u64>,
    pub(crate) progress_total: SyncSignal<Option<u64>>,
    pub(crate) progress_label: SyncSignal<String>,
    pub(crate) messages: SyncSignal<Vec<LogMessage>>,
    pub(crate) proposals: SyncSignal<Vec<ChangeProposal>>,
    pub(crate) bundles: SyncSignal<Vec<BundleProposal>>,
    pub(crate) decisions: SyncSignal<HashMap<Uuid, ReviewDecision>>,
    pub(crate) review_pending: SyncSignal<bool>,
    pub(crate) busy: SyncSignal<Busy>,
    pub(crate) last_report: SyncSignal<Option<LastReport>>,
    pub(crate) runs: SyncSignal<Vec<RunRecord>>,
    pub(crate) error: SyncSignal<Option<String>>,
    pub(crate) model_ready: SyncSignal<Option<bool>>,
}

/// Non-signal state: the pending review's sender, held across an await inside
/// the `ScanService::run` future. Accessed from both the Dioxus render thread
/// (when the user clicks *Apply*) and the service's tokio task (when review
/// starts), so a `tokio::Mutex` is the right primitive.
pub(crate) type ReviewSlot = Arc<Mutex<Option<oneshot::Sender<Vec<ReviewDecision>>>>>;

/// Top-level shared state provided at the app root and consumed by every page.
///
/// `PartialEq` is implemented via pointer identity on `review_slot` so that
/// `SharedState` can be used as a dioxus `#[component]` prop. Signals already
/// compare by generational-pointer identity via their own `PartialEq`.
#[derive(Clone)]
#[allow(missing_debug_implementations)]
pub(crate) struct SharedState {
    pub(crate) signals: SignalBundle,
    pub(crate) review_slot: ReviewSlot,
}

impl PartialEq for SharedState {
    fn eq(&self, other: &Self) -> bool {
        self.signals == other.signals && Arc::ptr_eq(&self.review_slot, &other.review_slot)
    }
}

impl SharedState {
    /// Build the bundle with every signal anchored to [`ScopeId::ROOT`].
    ///
    /// This is **not a hook**: we use [`Signal::new_maybe_sync_in_scope`]
    /// directly, which bypasses the per-scope hook list and makes the signal's
    /// owner independent of the caller's scope. Callers should wrap the call
    /// in [`use_hook`](dioxus::prelude::use_hook) at the app root to run it
    /// exactly once per app lifetime.
    ///
    /// Why root scope: `spawn_forever` tasks live on the root scope. Reading a
    /// root-scope signal from a root-scope task is a descendant read; reading
    /// an App-scope signal from that same task trips dioxus'
    /// `copy_value_hoisted` warning and risks dropped updates.
    pub(crate) fn new_at_root() -> Self {
        let signals = SignalBundle {
            phase: Signal::new_maybe_sync_in_scope(None, ScopeId::ROOT),
            progress_current: Signal::new_maybe_sync_in_scope(0_u64, ScopeId::ROOT),
            progress_total: Signal::new_maybe_sync_in_scope(None, ScopeId::ROOT),
            progress_label: Signal::new_maybe_sync_in_scope(String::new(), ScopeId::ROOT),
            messages: Signal::new_maybe_sync_in_scope(Vec::new(), ScopeId::ROOT),
            proposals: Signal::new_maybe_sync_in_scope(Vec::new(), ScopeId::ROOT),
            bundles: Signal::new_maybe_sync_in_scope(Vec::new(), ScopeId::ROOT),
            decisions: Signal::new_maybe_sync_in_scope(HashMap::new(), ScopeId::ROOT),
            review_pending: Signal::new_maybe_sync_in_scope(false, ScopeId::ROOT),
            busy: Signal::new_maybe_sync_in_scope(Busy::Idle, ScopeId::ROOT),
            last_report: Signal::new_maybe_sync_in_scope(None, ScopeId::ROOT),
            runs: Signal::new_maybe_sync_in_scope(Vec::new(), ScopeId::ROOT),
            error: Signal::new_maybe_sync_in_scope(None, ScopeId::ROOT),
            model_ready: Signal::new_maybe_sync_in_scope(None, ScopeId::ROOT),
        };
        Self {
            signals,
            review_slot: Arc::new(Mutex::new(None)),
        }
    }
}
