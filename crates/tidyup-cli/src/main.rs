// Binary crate — private modules use `pub(crate)` for explicitness, which conflicts
// with clippy::redundant_pub_crate. Silence it; the rustc unreachable_pub lint is more
// semantically correct for binaries.
#![allow(clippy::redundant_pub_crate)]

//! Tidyup CLI entry point.
//!
//! The CLI's job is narrow:
//! 1. Parse args (clap).
//! 2. Load config.
//! 3. Build the `ServiceContext` (storage + embeddings + extractors) via
//!    [`context::build`].
//! 4. Supply a CLI-flavored `ProgressReporter` (indicatif) and `ReviewHandler`
//!    (interactive prompts, or `--yes` auto-approver).
//! 5. Call `tidyup_app::*Service`.
//!
//! All business logic lives in `tidyup-app` / `tidyup-pipeline` — the CLI is a
//! thin adapter. The UI binary is the same shape with different reporter/review
//! impls.

mod commands;
mod context;
mod reporter;
mod review;

use clap::{Parser, Subcommand};

// Four global flags (yes/json/llm_fallback/remote) is fine for a CLI struct;
// clippy's bool-count lint targets domain types, not arg parsers.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Debug)]
#[command(name = "tidyup", version, about = "On-device AI file organizer")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Suppress interactive prompts; auto-approve anything above the internal
    /// confidence threshold. Renames never auto-apply, even under `--yes`.
    #[arg(long, global = true)]
    yes: bool,

    /// Emit JSON events instead of human-readable progress (for scripting).
    #[arg(long, global = true)]
    json: bool,

    /// Activate the local LLM Tier 3 fallback (mistralrs).
    ///
    /// Power-user opt-in. Triple-gated: requires `--features llm-fallback`
    /// at build time, `[inference] llm_fallback = true` in config, and this
    /// flag (or `TIDYUP_LLM_FALLBACK=1`) at invocation. Default builds and
    /// default invocations remain LLM-silent.
    #[arg(long, global = true, env = "TIDYUP_LLM_FALLBACK")]
    llm_fallback: bool,

    /// Activate the remote Tier 3 backend (`OpenAI`-compatible / Anthropic /
    /// Ollama).
    ///
    /// Power-user opt-in. Triple-gated: requires `--features remote` at
    /// build time, an `[inference.remote]` section in config, and this flag
    /// (or `TIDYUP_REMOTE=1`) at invocation.
    #[arg(long, global = true, env = "TIDYUP_REMOTE")]
    remote: bool,
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
    /// Roll back a previous run by ID, or list recorded runs with `--list`.
    Rollback {
        /// Run ID to roll back. Required unless `--list` is passed.
        run_id: Option<uuid::Uuid>,
        /// List recorded runs instead of rolling one back.
        #[arg(long)]
        list: bool,
    },
    /// Show current config (file path + parsed values).
    Config,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    commands::dispatch(cli).await
}
