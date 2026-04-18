//! `cargo xtask check-privacy` — enforce the default-build privacy invariant.
//!
//! The load-bearing promise in `CLAUDE.md` is that
//! `cargo build -p tidyup-cli` (no `--features`) produces a binary with no
//! HTTP client and no LLM inference linked. This task runs
//! `cargo tree -p tidyup-cli -e normal` on the default feature set and fails
//! if any banned crate name appears in the dependency graph.
//!
//! Keep this list in sync with the privacy model doc.

use std::process::Command;

use anyhow::{bail, Context, Result};

/// Crates that must NOT appear in the default `tidyup-cli` dep graph.
///
/// Network-capable crates (`reqwest`, `hyper`, `rustls`) are the primary
/// concern — the default build must be airplane-mode-compatible. LLM crates
/// (`mistralrs`, `candle-core`, `hf-hub`) must also stay out.
const BANNED: &[&str] = &[
    "reqwest",
    "hyper",
    "rustls",
    "mistralrs",
    "candle-core",
    "hf-hub",
];

#[allow(unreachable_pub)]
pub fn check() -> Result<()> {
    println!("xtask check-privacy: inspecting default tidyup-cli dep graph");

    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "-p",
            "tidyup-cli",
            "-e",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .context("spawn cargo tree")?;
    if !output.status.success() {
        bail!(
            "cargo tree failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let tree = String::from_utf8(output.stdout).context("cargo tree output is not UTF-8")?;

    let mut hits = Vec::new();
    for line in tree.lines() {
        let name = line.split_whitespace().next().unwrap_or("");
        if BANNED.contains(&name) {
            hits.push(line.trim().to_owned());
        }
    }

    if !hits.is_empty() {
        eprintln!("Default tidyup-cli dep graph contains banned crates:");
        for hit in &hits {
            eprintln!("  · {hit}");
        }
        bail!(
            "privacy invariant violated: {} banned dep(s) leaked into the default build",
            hits.len(),
        );
    }

    println!("  default graph clean — none of {BANNED:?} present");
    Ok(())
}
