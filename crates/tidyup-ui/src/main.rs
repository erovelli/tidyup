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

use dioxus::desktop::tao::dpi::LogicalSize;
use dioxus::desktop::tao::window::Icon;
use dioxus::desktop::{icon_from_memory, Config, WindowBuilder};
use dioxus::prelude::*;

use crate::pages::{Dashboard, Review, Runs, Settings};
use crate::state::SharedState;

/// The stylesheet is compiled into the binary rather than referenced through
/// the `asset!` macro.
///
/// `asset!` resolves to a hashed URL that only exists once the `dx` CLI has
/// collected assets into a bundle. A plain `cargo run`/`cargo build` binary —
/// the command `README.md` documents — produced an unresolvable href, so the
/// webview silently fell back to its default stylesheet and the whole app
/// rendered as unstyled serif HTML with a non-animating spinner. Embedding the
/// file removes the build-tool dependency and makes the documented command
/// produce the design in `DESIGN.md`.
const THEME_CSS: &str = include_str!("../assets/theme.css");

/// The window icon, embedded from the workspace-level brand assets.
///
/// `assets/brand/` is the single source of truth for the mark — `README.md`,
/// `Dioxus.toml`, and the freedesktop entry all read the same files — so this
/// reaches out of the crate rather than keeping a second copy in sync. That is
/// safe only because `tidyup-ui` is `publish = false`; a publishable crate
/// cannot `include_bytes!` outside its own package directory.
const APP_ICON_PNG: &[u8] = include_bytes!("../../../assets/brand/icon-256.png");

/// User-facing application name.
///
/// Deliberately distinct from the `tidyup-desktop` binary name, which exists to
/// disambiguate from the `tidyup` CLI binary at the `cargo run` level and is not
/// meant to be read by users.
///
/// This must be set in code. dioxus-desktop builds its default window from
/// `dioxus_cli_config::app_title()` and falls back to the literal `"Dioxus App"`
/// when that is unset — and it is always unset under a plain `cargo build` /
/// `cargo run`, which is the command `README.md` documents. `Dioxus.toml` feeds
/// `app_title()` only when the app is built through the `dx` CLI, so the
/// manifest alone would leave the documented build path titled "Dioxus App".
/// Don't drop this in favour of the manifest.
const WINDOW_TITLE: &str = "Tidyup";

/// Decode the embedded window icon, degrading to the platform default.
///
/// Workspace lints forbid `unwrap`/`expect`/`panic`, and a window icon is
/// cosmetic: a corrupt asset must not stop the app from starting. This mirrors
/// how a missing Swift toolchain degrades OCR to a warning rather than breaking
/// the build.
fn window_icon() -> Option<Icon> {
    match icon_from_memory::<Icon>(APP_ICON_PNG) {
        Ok(icon) => Some(icon),
        Err(e) => {
            tracing::warn!("could not decode the bundled window icon: {e}");
            None
        }
    }
}

/// Build the desktop window configuration.
///
/// Without this, dioxus-desktop titles the window "Dioxus App" and falls back to
/// *its own* logo for the window icon (`webview.rs` substitutes `default_icon()`
/// whenever `window_icon` is unset).
fn desktop_config() -> Config {
    let window = WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        // The review diff is a two-column layout with drawn connectors between
        // the columns; below roughly 960x640 the columns overlap the connector
        // gutter, so the floor is a layout constraint rather than a preference.
        .with_inner_size(LogicalSize::new(1280.0, 860.0))
        .with_min_inner_size(LogicalSize::new(960.0, 640.0));

    // `with_window` replaces the whole builder, so anything that writes through
    // to window fields — `with_icon` does — has to come after it.
    let mut config = Config::new()
        .with_window(window)
        // Paint the host surface `--surface` so launch doesn't flash white
        // before the webview paints the stylesheet.
        .with_background_color((0xf9, 0xf9, 0xf9, 0xff));

    if let Some(icon) = window_icon() {
        config = config.with_icon(icon);
    }

    config
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    LaunchBuilder::desktop()
        .with_cfg(desktop_config())
        .launch(App);
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
    let mut llm_fallback_active = state.signals.llm_fallback_active;
    use_future(move || async move {
        llm_fallback_active.set(state::load_llm_fallback_prearm().await);
    });
    provide_context(state);

    rsx! {
        document::Style { {THEME_CSS} }
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
