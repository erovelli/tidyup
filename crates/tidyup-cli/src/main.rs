// Binary crate — private modules use `pub(crate)` for explicitness, which conflicts
// with clippy::redundant_pub_crate. Silence it; the rustc unreachable_pub lint is more
// semantically correct for binaries.
#![allow(clippy::redundant_pub_crate)]
// Stubs don't await yet; remove once commands dispatch through real services.
#![allow(clippy::unused_async)]
#![allow(clippy::missing_const_for_fn)]

//! Tidyup CLI entry point.
//!
//! The CLI's job is narrow:
//! 1. Parse args (clap).
//! 2. Load config.
//! 3. Build the `ServiceContext` via the backend registry.
//! 4. Supply a CLI-flavored `ProgressReporter` (indicatif) and `ReviewHandler`
//!    (interactive prompts, or `--yes` auto-approver).
//! 5. Call `tidyup_app::*Service`.
//!
//! All business logic lives in `tidyup-app` / `tidyup-pipeline` — the CLI is a
//! thin adapter. The UI binary is the same shape with different reporter/review
//! impls.

mod commands;
mod reporter;
mod review;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "tidyup", version, about = "On-device AI file organizer")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Suppress interactive prompts; auto-approve anything above `--min-confidence`.
    #[arg(long, global = true)]
    yes: bool,

    /// Emit JSON events instead of human-readable progress (for scripting).
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Migrate files from SOURCE into an existing target hierarchy.
    Migrate {
        source: std::path::PathBuf,
        target: std::path::PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    /// Classify files in place against a taxonomy.
    Scan {
        root: std::path::PathBuf,
        #[arg(long)]
        taxonomy: Option<std::path::PathBuf>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Roll back a previous run by ID.
    Rollback { run_id: uuid::Uuid },
    /// Show/edit config.
    Config,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let cli = Cli::parse();
    commands::dispatch(cli).await
}
