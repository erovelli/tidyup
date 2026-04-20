// Binary crate — `pub(crate)` on private modules conflicts with
// clippy::redundant_pub_crate. Silence it; rustc's `unreachable_pub` is the
// better check for binaries.
#![allow(clippy::redundant_pub_crate)]
// Dioxus' `rsx!` expansion triggers a spurious `unused_qualifications` on
// event-handler attribute names on stable Rust 1.90.
#![allow(unused_qualifications)]

//! Tidyup desktop UI entry point.
//!
//! Mirrors the CLI: build the same `ServiceContext` and call the same
//! `tidyup-app` services, but supply Dioxus-signal-backed `ProgressReporter`
//! and `ReviewHandler` implementations. The plug-and-play seam lives in
//! `tidyup-app`; this binary never reimplements business logic.

mod context;
mod pages;
mod reporter;
mod review;
mod state;

use dioxus::prelude::*;

use crate::pages::{Dashboard, Review, Runs, Settings};
use crate::state::SharedState;

const THEME_CSS: Asset = asset!("/assets/theme.css");

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    dioxus::launch(App);
}

#[derive(Clone, Routable, PartialEq, Debug)]
enum Route {
    #[layout(AppShell)]
    #[route("/")]
    Dashboard {},
    #[route("/review")]
    Review {},
    #[route("/runs")]
    Runs {},
    #[route("/settings")]
    Settings {},
}

#[component]
fn App() -> Element {
    // `SharedState::new_at_root` creates signals pinned to `ScopeId::ROOT` via
    // `Signal::new_maybe_sync_in_scope` — not via `use_signal_sync`. That lets
    // `spawn_forever` tasks (also on the root scope) read/write the signals
    // without tripping the `copy_value_hoisted` warning.
    //
    // Wrapped in `use_hook` so it runs exactly once per App lifetime; the
    // cached `SharedState` is cloned on every subsequent render. Inner handles
    // (signals, Arc) stay stable.
    let state = use_hook(SharedState::new_at_root);
    provide_context(state);

    rsx! {
        document::Stylesheet { href: THEME_CSS }
        Router::<Route> {}
    }
}

#[component]
fn AppShell() -> Element {
    let state = use_context::<SharedState>();
    let pending_sig = state.signals.review_pending;
    let nav = use_navigator();

    // When the service asks for review, flip to the review page. Only the
    // `review_pending` signal is subscribed here — `nav.replace` with the
    // same route is a no-op, so we don't need to read the current route.
    use_effect(move || {
        if *pending_sig.read() {
            nav.replace(Route::Review {});
        }
    });

    rsx! {
        div {
            class: "app-shell",
            aside {
                class: "sidebar",
                h1 { class: "app-title", "tidyup" }
                nav {
                    class: "nav",
                    NavItem { to: Route::Dashboard {}, label: "Dashboard" }
                    NavItem { to: Route::Review {},    label: "Review" }
                    NavItem { to: Route::Runs {},      label: "Runs" }
                    NavItem { to: Route::Settings {},  label: "Settings" }
                }
            }
            main {
                class: "content",
                Outlet::<Route> {}
            }
        }
    }
}

#[component]
fn NavItem(to: Route, label: &'static str) -> Element {
    let route: Route = use_route();
    let active = std::mem::discriminant(&route) == std::mem::discriminant(&to);
    let class = if active {
        "nav-item active"
    } else {
        "nav-item"
    };
    rsx! {
        Link {
            to: to,
            class: "{class}",
            "{label}"
        }
    }
}
