#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::missing_const_for_fn)]

//! Tidyup desktop UI entry point.
//!
//! Mirror of the CLI: build the same `ServiceContext`, but implement
//! `ProgressReporter` and `ReviewHandler` as Dioxus-signal-driven adapters.
//! The service calls are identical — that is the plug-and-play seam.
//!
//! Deferred behind the CLI in the v0.x release plan.

fn main() {
    // TODO:
    // - launch Dioxus desktop
    // - pages/: dashboard, diff_view, backup_browser, settings, onboarding
    // - reporter.rs: DioxusReporter { phase_signal, progress_signal }
    // - review.rs:   DioxusReviewHandler — resolves a oneshot channel when user clicks Apply
}
