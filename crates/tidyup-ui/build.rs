//! Stamps the build's git revision into the binary for the Settings → About
//! card.
//!
//! Nothing here may fail the build. A release tarball, a vendored source drop,
//! or any checkout without a `.git` directory has no revision to report, and
//! that is a normal condition rather than an error — the UI reads the value
//! through `option_env!` and renders "unknown" when it is absent. This mirrors
//! `tidyup-extract`'s build script, which degrades a missing Swift toolchain to
//! a warning rather than breaking the workspace.

use std::{path::Path, process::Command};

fn main() {
    // Re-stamp when HEAD moves. `.git/HEAD` covers commits and checkouts;
    // `packed-refs` covers the case where the branch ref is packed rather than
    // loose. Both are declared only when they exist, since naming a path that
    // never appears would make cargo rebuild this crate on every invocation.
    for path in [".git/HEAD", ".git/packed-refs"] {
        let from_workspace_root = Path::new("../..").join(path);
        if from_workspace_root.exists() {
            println!("cargo:rerun-if-changed={}", from_workspace_root.display());
        }
    }

    if let Some(sha) = git_short_sha() {
        println!("cargo:rustc-env=TIDYUP_GIT_SHA={sha}");
    }
}

/// The short revision of `HEAD`, with a `-dirty` marker when the working tree
/// has uncommitted changes.
///
/// Returns `None` for any failure at all — git missing from PATH, no
/// repository, a git that errors — because none of those are worth failing a
/// build over.
fn git_short_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let sha = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    if sha.is_empty() {
        return None;
    }

    Some(if is_dirty() {
        format!("{sha}-dirty")
    } else {
        sha
    })
}

/// Whether tracked files differ from `HEAD`.
///
/// A non-empty `git status --porcelain` means the binary does not correspond to
/// the reported revision, which is precisely the situation a bug report needs
/// to disclose. An unavailable git is reported as clean rather than dirty: it
/// is the same "no revision information" case the caller already handles.
fn is_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .is_ok_and(|out| out.status.success() && !out.stdout.is_empty())
}
