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
mod watch;

use clap::{Parser, Subcommand};

// Four global flags (yes/json/llm_fallback/remote) is fine for a CLI struct;
// clippy's bool-count lint targets domain types, not arg parsers.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Debug)]
#[command(name = "tidyup", version, about = "On-device AI file organizer")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Suppress interactive prompts; auto-approve loose move-only proposals
    /// above `[classifier] min_confidence` and recognized opaque structural
    /// bundles above the application's separate threshold. Soft/file-set and
    /// generic bundles remain review-only, as do all renames.
    #[arg(long, global = true)]
    yes: bool,

    /// Emit JSON events instead of human-readable progress (for scripting).
    #[arg(long, global = true)]
    json: bool,

    /// Activate the optional local LLM reranker (mistralrs).
    ///
    /// Power-user opt-in. Triple-gated: requires `--features llm-fallback`
    /// at build time, `[inference] llm_fallback = true` in config, and this
    /// flag (or `TIDYUP_LLM_FALLBACK=1`) at invocation. Default builds and
    /// default invocations remain LLM-silent.
    ///
    /// The env var accepts `1`, `true`, `yes`, or `on` (case-insensitive).
    #[arg(long, global = true)]
    llm_fallback: bool,

    /// Activate the optional remote reranker (`OpenAI`-compatible endpoint).
    ///
    /// Power-user opt-in. Triple-gated: requires `--features remote` at
    /// build time, an `[inference.remote]` section in config, and this flag
    /// (or `TIDYUP_REMOTE=1`) at invocation. The env var accepts the same
    /// boolish values as `TIDYUP_LLM_FALLBACK`. Only the OpenAI-compatible
    /// endpoint is selectable from config today; Anthropic/Ollama adapters exist
    /// in `tidyup-inference-remote` but aren't yet wired.
    #[arg(long, global = true)]
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
        /// Path to a custom taxonomy TOML file (array of `[[entry]]` tables with
        /// `path`, `description`, optional `temporal`). Overrides the built-in
        /// text taxonomy; image/audio taxonomies stay default. Omit to use the
        /// built-in taxonomy.
        #[arg(long)]
        taxonomy: Option<std::path::PathBuf>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Watch SOURCE and report proposals (dry-run) on every change.
    ///
    /// Advisory only: it never moves files — run `scan` to apply. The model loads
    /// once and is reused across rescans. Press Ctrl-C to stop.
    Watch {
        root: std::path::PathBuf,
        /// Custom taxonomy TOML file (same format as `scan --taxonomy`).
        #[arg(long)]
        taxonomy: Option<std::path::PathBuf>,
        /// Debounce window (ms) for coalescing rapid changes into one rescan.
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
    },
    /// Roll back a previous run by ID, or list recorded runs with `--list`.
    Rollback {
        /// Run ID to roll back. Required unless `--list` is passed.
        run_id: Option<uuid::Uuid>,
        /// List recorded runs instead of rolling one back.
        #[arg(long)]
        list: bool,
    },
    /// Prune shelved backups and semantic cache entries past the retention window.
    ///
    /// Expires backups and removes reconstructible semantic artifacts past the
    /// TTL. Defaults to `[storage] backup_retention_days` (30 unless configured).
    /// Once a backup is pruned, its run can no longer be rolled back.
    Prune {
        /// Override the retention window, in days.
        #[arg(long)]
        days: Option<u32>,
    },
    /// Show a status summary: data dir, model, retention, and recent runs.
    Status,
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
