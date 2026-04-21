//! Workspace automation. Run via `cargo xtask <task>`.
//!
//! Replaces ad-hoc shell scripts. Keeps the build story portable across Linux,
//! macOS, and Windows with no bash dependency.

mod models;
mod privacy;

use std::process::{Command, ExitCode};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "Workspace automation")]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Run the full CI suite: fmt, clippy, tests, deny, feature matrix.
    Ci,
    /// Format all crates.
    Fmt,
    /// Clippy with workspace lint policy.
    Lint,
    /// Run `cargo deny` checks (requires `cargo install cargo-deny`).
    Deny,
    /// Run `cargo hack --feature-powerset check` to catch feature breakage.
    FeatureMatrix,
    /// Download model bundles into the platform cache directory.
    ///
    /// Used by packagers and developers. The default tidyup CLI binary is
    /// network-silent by design (see `CLAUDE.md`) — this xtask is the
    /// sanctioned one-shot installer.
    ///
    /// By default downloads only the text embedding bundle (`bge-small`).
    /// Pass `--siglip` to additionally fetch the Phase 7 image encoder, or
    /// `--clap` for the audio encoder. `--multimodal` is shorthand for
    /// `--siglip --clap`.
    DownloadModels {
        /// Overwrite existing files if they are present.
        #[arg(long)]
        force: bool,
        /// Also download the `SigLIP` image encoder bundle.
        #[arg(long)]
        siglip: bool,
        /// Also download the `CLAP` audio encoder bundle.
        #[arg(long)]
        clap: bool,
        /// Convenience: enables both `--siglip` and `--clap`.
        #[arg(long)]
        multimodal: bool,
    },
    /// Assert the default `tidyup-cli` dep graph contains no network or LLM
    /// deps. Runs `cargo tree -p tidyup-cli -e normal` and fails if any
    /// banned crate appears. Embedded in `ci`.
    CheckPrivacy,
}

fn main() -> ExitCode {
    match Cli::parse().task.run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask failed: {e:#}");
            ExitCode::FAILURE
        }
    }
}

impl Task {
    fn run(self) -> Result<()> {
        match self {
            Self::Ci => {
                privacy::check()?;
                sh(&["fmt", "--all", "--", "--check"])?;
                run_ci_lints()?;
                run_ci_tests()?;
                Ok(())
            }
            Self::Fmt => sh(&["fmt", "--all"]),
            Self::Lint => run_ci_lints(),
            Self::Deny => ext("cargo-deny", &["check"]),
            Self::FeatureMatrix => ext("cargo-hack", &["hack", "--feature-powerset", "check"]),
            Self::DownloadModels {
                force,
                siglip,
                clap,
                multimodal,
            } => {
                let want_siglip = siglip || multimodal;
                let want_clap = clap || multimodal;
                models::download(force, want_siglip, want_clap)
            }
            Self::CheckPrivacy => privacy::check(),
        }
    }
}

/// Feature combinations exercised by `cargo xtask ci`.
///
/// `--all-features` isn't usable workspace-wide because the cli exposes
/// `llm-metal` / `llm-cuda` accelerator pass-throughs that require
/// platform-specific toolchains (Metal on macOS, `nvcc`/`cudarc` for CUDA).
/// Neither can succeed on every CI host, so we run a layered matrix instead:
/// workspace-default for regression coverage, cli with the two opt-in inference
/// features, and extract with all format features. Accelerator passthroughs
/// remain available to end users via `cargo build --features llm-metal`.
fn run_ci_lints() -> Result<()> {
    sh(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ])?;
    sh(&[
        "clippy",
        "-p",
        "tidyup-cli",
        "--all-targets",
        "--features",
        "llm-fallback,remote",
        "--",
        "-D",
        "warnings",
    ])?;
    sh(&[
        "clippy",
        "-p",
        "tidyup-extract",
        "--all-targets",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ])?;
    Ok(())
}

fn run_ci_tests() -> Result<()> {
    sh(&["test", "--workspace"])?;
    sh(&[
        "test",
        "-p",
        "tidyup-cli",
        "--features",
        "llm-fallback,remote",
    ])?;
    sh(&["test", "-p", "tidyup-extract", "--all-features"])?;
    Ok(())
}

fn sh(args: &[&str]) -> Result<()> {
    let status = Command::new(env!("CARGO"))
        .args(args)
        .status()
        .with_context(|| format!("failed to spawn cargo {args:?}"))?;
    if !status.success() {
        bail!("cargo {args:?} exited with {status}");
    }
    Ok(())
}

fn ext(bin: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(bin)
        .args(args)
        .status()
        .with_context(|| format!("failed to run {bin}; install it with `cargo install {bin}`"))?;
    if !status.success() {
        bail!("{bin} {args:?} exited with {status}");
    }
    Ok(())
}
