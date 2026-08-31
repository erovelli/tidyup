//! Stamps the build's git revision into the binary for the Settings → About
//! card.
//!
//! Nothing here may fail the build. A release tarball, a vendored source drop,
//! or any checkout without a `.git` directory has no revision to report, and
//! that is a normal condition rather than an error — the UI reads the value
//! through `option_env!` and renders "unknown" when it is absent. This mirrors
//! `tidyup-extract`'s build script, which degrades a missing Swift toolchain to
//! a warning rather than breaking the workspace.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let workspace = workspace_root();
    configure_git_reruns(&workspace);

    if let Some(sha) = git_short_sha(&workspace) {
        println!("cargo:rustc-env=TIDYUP_GIT_SHA={sha}");
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("../.."))
}

/// Re-run for every input that can change the reported revision or dirty flag.
///
/// Watching the resolved symbolic ref is essential: `.git/HEAD` itself is
/// unchanged when a branch receives a new commit. The tracked-file directives
/// make the `-dirty` suffix current even when Cargo otherwise reuses a prior
/// build-script result.
fn configure_git_reruns(workspace: &Path) {
    let Some(git_dir) = git_dir(workspace) else {
        return;
    };
    rerun_if_exists(&git_dir.join("HEAD"));
    rerun_if_exists(&git_dir.join("packed-refs"));
    rerun_if_exists(&git_dir.join("index"));

    if let Ok(head) = std::fs::read_to_string(git_dir.join("HEAD")) {
        if let Some(reference) = head.trim().strip_prefix("ref: ") {
            rerun_if_exists(&git_dir.join(reference));
        }
    }

    let Ok(output) = Command::new("git")
        .args(["-C", &workspace.display().to_string(), "ls-files", "-z"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    for path in String::from_utf8_lossy(&output.stdout).split('\0') {
        if !path.is_empty() {
            rerun_if_exists(&workspace.join(path));
        }
    }
}

fn rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn git_dir(workspace: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .args([
            "-C",
            &workspace.display().to_string(),
            "rev-parse",
            "--git-dir",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(raw.trim());
    Some(if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    })
}

/// The short revision of `HEAD`, with a `-dirty` marker when the working tree
/// has uncommitted changes.
///
/// Returns `None` for any failure at all — git missing from PATH, no
/// repository, a git that errors — because none of those are worth failing a
/// build over.
fn git_short_sha(workspace: &Path) -> Option<String> {
    let output = Command::new("git")
        .args([
            "-C",
            &workspace.display().to_string(),
            "rev-parse",
            "--short",
            "HEAD",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let sha = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    if sha.is_empty() {
        return None;
    }

    Some(if is_dirty(workspace) {
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
fn is_dirty(workspace: &Path) -> bool {
    Command::new("git")
        .args([
            "-C",
            &workspace.display().to_string(),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ])
        .output()
        .is_ok_and(|out| out.status.success() && !out.stdout.is_empty())
}
