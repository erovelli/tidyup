//! Workspace automation. Run via `cargo xtask <task>`.
//!
//! Replaces ad-hoc shell scripts. Keeps the build story portable across Linux,
//! macOS, and Windows with no bash dependency.

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
                sh(&["fmt", "--all", "--", "--check"])?;
                sh(&[
                    "clippy",
                    "--workspace",
                    "--all-targets",
                    "--all-features",
                    "--",
                    "-D",
                    "warnings",
                ])?;
                sh(&["test", "--workspace", "--all-features"])?;
                Ok(())
            }
            Self::Fmt => sh(&["fmt", "--all"]),
            Self::Lint => sh(&[
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ]),
            Self::Deny => ext("cargo-deny", &["check"]),
            Self::FeatureMatrix => ext("cargo-hack", &["hack", "--feature-powerset", "check"]),
        }
    }
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
