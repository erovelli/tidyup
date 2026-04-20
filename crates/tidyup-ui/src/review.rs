//! Dioxus [`ReviewHandler`] impl.
//!
//! The CLI's interactive handler blocks on `term.read_key()`; the UI's blocks
//! on a [`tokio::sync::oneshot`] the UI fulfils when the user clicks *Apply*
//! on the review page.
//!
//! Flow per [`review`](ReviewHandler::review) call:
//!
//! 1. Stash `proposals` (and any pending bundles already on the signal) into
//!    the UI state and mark `review_pending = true`.
//! 2. Install a fresh oneshot sender into [`ReviewSlot`]; park the receiver.
//! 3. Await the receiver. The app shell observes `review_pending` and routes
//!    to `/review`; the review page drives approve/reject per item and, on
//!    *Apply*, takes the sender out of the slot and sends the collected
//!    decisions.
//! 4. On completion, clear `review_pending` and the stashed proposal list so
//!    the UI returns to a clean state before the service applies moves.
//!
//! Default behaviour mirrors the CLI: unseen proposals count as *reject*,
//! never *approve*. Renames are surfaced but treated identically — the user
//! must explicitly approve each one. Renames never auto-apply per
//! `CLAUDE.md` → "Rename policy".
#![allow(clippy::large_types_passed_by_value)]

use async_trait::async_trait;
use dioxus::prelude::*;
use tidyup_core::{frontend::ReviewHandler, Result};
use tidyup_domain::{ChangeProposal, ReviewDecision};
use tokio::sync::oneshot;

use crate::state::{ReviewSlot, SignalBundle};

#[allow(missing_debug_implementations)]
pub(crate) struct DioxusReviewHandler {
    signals: SignalBundle,
    slot: ReviewSlot,
}

impl DioxusReviewHandler {
    pub(crate) const fn new(signals: SignalBundle, slot: ReviewSlot) -> Self {
        Self { signals, slot }
    }
}

#[async_trait]
impl ReviewHandler for DioxusReviewHandler {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
        if proposals.is_empty() {
            return Ok(Vec::new());
        }

        // Leave the map empty so the UI can distinguish "user has not decided
        // this one yet" from "user explicitly rejected it".
        // `execute_with_threshold` fills untouched proposals at submit time;
        // submit_review sends the final, complete decision set to the service.
        let mut decisions_sig = self.signals.decisions;
        decisions_sig.with_mut(std::collections::HashMap::clear);

        let mut proposals_sig = self.signals.proposals;
        proposals_sig.set(proposals);

        let (tx, rx) = oneshot::channel::<Vec<ReviewDecision>>();
        {
            let mut guard = self.slot.lock().await;
            *guard = Some(tx);
        }

        let mut pending = self.signals.review_pending;
        pending.set(true);

        let decisions = rx
            .await
            .map_err(|e| anyhow::anyhow!("review cancelled: {e}"))?;

        // Review complete — clear the transient UI state before the service
        // moves on to apply.
        pending.set(false);
        proposals_sig.set(Vec::new());
        decisions_sig.with_mut(std::collections::HashMap::clear);

        Ok(decisions)
    }
}
