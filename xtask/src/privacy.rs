//! `cargo xtask check-privacy` — enforce the default-build privacy invariant.
//!
//! The load-bearing promise in `CLAUDE.md` is that the default builds produce
//! binaries with no HTTP client and no LLM inference linked. This task runs
//! `cargo tree -e normal` on the default feature set of each shipped binary and
//! fails if any banned crate name appears in the dependency graph:
//!
//! * `tidyup-cli` — network-silent **and** LLM-silent (airplane-mode default).
//! * `tidyup-ui`  — LLM-silent. Network crates aren't banned here: it's a
//!   webview desktop app, so network-silence is not one of its promises; the
//!   load-bearing UI guarantee is that the default build doesn't bundle the LLM.
//!
//! Keep these lists in sync with the privacy model doc.

use std::process::Command;

use anyhow::{bail, Context, Result};

/// LLM-inference crates that must never appear in a default build.
const LLM: &[&str] = &["mistralrs", "candle-core", "hf-hub"];

/// Network crates that must stay out of the airplane-mode default CLI build.
const NETWORK: &[&str] = &["reqwest", "hyper", "rustls"];

#[allow(unreachable_pub)]
pub fn check() -> Result<()> {
    // CLI default build: network-silent and LLM-silent.
    let cli_banned: Vec<&str> = NETWORK.iter().chain(LLM).copied().collect();
    check_crate("tidyup-cli", &cli_banned)?;
    // UI default build: LLM-silent (Tier 3 is a power-user `--features` opt-in).
    check_crate("tidyup-ui", LLM)?;
    Ok(())
}

/// Resolve `crate_name`'s default-feature normal dep graph and fail if any
/// `banned` crate appears in it.
fn check_crate(crate_name: &str, banned: &[&str]) -> Result<()> {
    println!("xtask check-privacy: inspecting default {crate_name} dep graph");

    let output = Command::new(env!("CARGO"))
        .args(["tree", "-p", crate_name, "-e", "normal", "--prefix", "none"])
        .output()
        .context("spawn cargo tree")?;
    if !output.status.success() {
        bail!(
            "cargo tree failed for {crate_name} ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let tree = String::from_utf8(output.stdout).context("cargo tree output is not UTF-8")?;

    let mut hits = Vec::new();
    for line in tree.lines() {
        let name = line.split_whitespace().next().unwrap_or("");
        if banned.contains(&name) {
            hits.push(line.trim().to_owned());
        }
    }

    if !hits.is_empty() {
        eprintln!("Default {crate_name} dep graph contains banned crates:");
        for hit in &hits {
            eprintln!("  · {hit}");
        }
        bail!(
            "privacy invariant violated: {} banned dep(s) leaked into the default {crate_name} build",
            hits.len(),
        );
    }

    println!("  {crate_name} default graph clean — none of {banned:?} present");
    Ok(())
}
