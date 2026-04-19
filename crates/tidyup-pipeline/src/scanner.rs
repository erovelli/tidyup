//! Source-directory walker with bundle detection.
//!
//! The first pass of every pipeline run. The scanner walks the source tree,
//! marks self-contained subtrees (`.git/`, `Cargo.toml`, `package.json`,
//! `pyproject.toml`, `*.xcodeproj`, `build.gradle`, clusters of `.ipynb`
//! files) as [`DetectedBundle`]s, and collects every other file as a loose
//! entry.
//!
//! # Opacity
//!
//! Once a subtree is identified as a bundle, the scanner **does not descend**
//! for per-file classification. All files under the bundle root are captured
//! as `members` and flow to the apply layer as an atomic `BundleProposal`
//! (assembled by the pipeline from the scanner output). This mirrors the
//! "bundles move all-or-nothing" invariant in `CLAUDE.md`.
//!
//! # Precedence
//!
//! When multiple markers coexist in the same directory the most specific
//! wins — `Cargo.toml` outranks `.git/` because a Rust crate's relevance to
//! a user is "a crate", not "a repository". Directories with no marker but
//! containing *two or more* sibling `.ipynb` files become
//! [`BundleKind::JupyterNotebookSet`], which is the weakest hard-bundle
//! signal.
//!
//! # Scope
//!
//! v0.1 covers marker-file bundles only. EXIF-clustered photo bursts and
//! ID3-clustered music albums are deferred — they require content reads the
//! scanner intentionally avoids, and will land alongside the corresponding
//! extractors. Soft `DocumentSeries` detection (filename families like
//! `invoice-2024-01.pdf`, `invoice-2024-02.pdf`) is also deferred; the
//! rename cascade still handles those files individually in the meantime.

use std::fs;
use std::path::{Path, PathBuf};

use tidyup_domain::bundle::BundleKind;
use walkdir::WalkDir;

/// Output of a single source-tree scan. Consumed by the pipeline to emit
/// `BundleProposal`s for bundles and per-file classification for loose files.
#[derive(Debug, Clone)]
pub struct ScanTree {
    pub root: PathBuf,
    pub bundles: Vec<DetectedBundle>,
    pub loose_files: Vec<PathBuf>,
}

/// A subtree the scanner has marked as an atomic move unit.
#[derive(Debug, Clone)]
pub struct DetectedBundle {
    /// Directory containing the bundle marker (e.g. the dir with `Cargo.toml`).
    pub root: PathBuf,
    pub kind: BundleKind,
    /// Every regular file under [`Self::root`], recursively. Dotfiles and
    /// editor/VCS noise are preserved — the bundle is opaque.
    pub members: Vec<PathBuf>,
    /// Human-readable explanation of the detection, for proposal reasoning
    /// and audit logs.
    pub reasoning: String,
}

/// Walk `root`, classify each directory as bundle-root or transparent, and
/// collect loose files from transparent directories.
///
/// Symlinks are never followed to avoid cycles and escape from the source
/// tree. Unreadable subdirectories are logged and skipped rather than
/// aborting the scan — a single permission error on a home directory
/// shouldn't kill the whole run.
///
/// Errors on descendants (permission denied, unreadable entries) are
/// swallowed and logged via `tracing::warn!` rather than aborting the scan.
#[must_use]
pub fn scan(root: &Path) -> ScanTree {
    let mut tree = ScanTree {
        root: root.to_path_buf(),
        bundles: Vec::new(),
        loose_files: Vec::new(),
    };
    scan_dir(root, &mut tree);
    tree
}

fn scan_dir(dir: &Path, tree: &mut ScanTree) {
    if let Some((kind, reason)) = detect_bundle(dir) {
        let members = collect_members(dir);
        tree.bundles.push(DetectedBundle {
            root: dir.to_path_buf(),
            kind,
            members,
            reasoning: reason.to_string(),
        });
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!("scanner: unable to read {}: {e}", dir.display());
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("scanner: bad entry in {}: {e}", dir.display());
                continue;
            }
        };
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    "scanner: file_type failed for {}: {e}",
                    entry.path().display()
                );
                continue;
            }
        };
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            scan_dir(&path, tree);
        } else if file_type.is_file() && !is_noise(&path) {
            tree.loose_files.push(path);
        }
    }
}

/// Inspect the direct children of `dir` and decide whether it roots a bundle.
///
/// Returns `Some((kind, reason))` on match. Precedence runs from most-specific
/// marker to least; see module docs for rationale.
fn detect_bundle(dir: &Path) -> Option<(BundleKind, &'static str)> {
    let entries = fs::read_dir(dir).ok()?;

    let mut has_cargo_toml = false;
    let mut has_package_json = false;
    let mut has_pyproject = false;
    let mut has_setup_py = false;
    let mut has_gradle = false;
    let mut has_dot_git = false;
    let mut has_xcodeproj = false;
    let mut notebook_count: u32 = 0;

    for entry in entries.flatten() {
        let name_os = entry.file_name();
        let Some(name) = name_os.to_str() else {
            continue;
        };
        let Ok(ft) = entry.file_type() else {
            continue;
        };

        if ft.is_dir() {
            if name == ".git" {
                has_dot_git = true;
            } else if name.ends_with(".xcodeproj") {
                has_xcodeproj = true;
            }
        } else if ft.is_file() {
            match name {
                "Cargo.toml" => has_cargo_toml = true,
                "package.json" => has_package_json = true,
                "pyproject.toml" => has_pyproject = true,
                "setup.py" | "setup.cfg" => has_setup_py = true,
                "settings.gradle" | "settings.gradle.kts" | "build.gradle" | "build.gradle.kts" => {
                    has_gradle = true;
                }
                other
                    if Path::new(other)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("ipynb")) =>
                {
                    notebook_count += 1;
                }
                _ => {}
            }
        }
    }

    if has_cargo_toml {
        return Some((BundleKind::RustCrate, "Cargo.toml manifest at root"));
    }
    if has_package_json {
        return Some((BundleKind::NodeProject, "package.json manifest at root"));
    }
    if has_pyproject || has_setup_py {
        return Some((
            BundleKind::PythonProject,
            "pyproject.toml or setup.py at root",
        ));
    }
    if has_xcodeproj {
        return Some((BundleKind::XcodeProject, ".xcodeproj directory at root"));
    }
    if has_gradle {
        return Some((
            BundleKind::AndroidStudioProject,
            "Gradle build files at root",
        ));
    }
    if has_dot_git {
        return Some((BundleKind::GitRepository, ".git directory at root"));
    }
    if notebook_count >= 2 {
        return Some((
            BundleKind::JupyterNotebookSet,
            "two or more sibling .ipynb notebooks",
        ));
    }
    None
}

fn collect_members(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .collect()
}

/// Noise files that shouldn't surface in classification — OS metadata, icon
/// caches, folder-preview artifacts. Bundle walks intentionally preserve
/// these because they're part of the atomic subtree; loose scans skip them.
fn is_noise(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| {
            matches!(
                name,
                ".DS_Store" | "Thumbs.db" | "desktop.ini" | ".localized",
            )
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn touch(dir: &Path, rel: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, b"").unwrap();
    }

    #[test]
    fn empty_directory_scans_to_empty_tree() {
        let td = TempDir::new().unwrap();
        let tree = scan(td.path());
        assert!(tree.bundles.is_empty());
        assert!(tree.loose_files.is_empty());
    }

    #[test]
    fn loose_files_are_collected() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "a.txt");
        touch(td.path(), "sub/b.md");
        touch(td.path(), "sub/nested/c.pdf");
        let tree = scan(td.path());
        assert!(tree.bundles.is_empty());
        assert_eq!(tree.loose_files.len(), 3);
    }

    #[test]
    fn cargo_toml_detects_rust_crate() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "proj/Cargo.toml");
        touch(td.path(), "proj/src/main.rs");
        touch(td.path(), "proj/src/lib.rs");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::RustCrate);
        assert!(tree.loose_files.is_empty());
        assert_eq!(tree.bundles[0].members.len(), 3);
    }

    #[test]
    fn package_json_detects_node_project() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "app/package.json");
        touch(td.path(), "app/index.js");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::NodeProject);
    }

    #[test]
    fn pyproject_detects_python_project() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "pkg/pyproject.toml");
        touch(td.path(), "pkg/module.py");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::PythonProject);
    }

    #[test]
    fn setup_py_detects_python_project() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "pkg/setup.py");
        touch(td.path(), "pkg/pkg/__init__.py");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::PythonProject);
    }

    #[test]
    fn dot_git_alone_detects_git_repository() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "repo/.git/HEAD");
        touch(td.path(), "repo/README.md");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::GitRepository);
    }

    #[test]
    fn cargo_toml_outranks_dot_git() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "repo/.git/HEAD");
        touch(td.path(), "repo/Cargo.toml");
        touch(td.path(), "repo/src/lib.rs");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::RustCrate);
    }

    #[test]
    fn xcodeproj_detects_xcode_project() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("app/MyApp.xcodeproj")).unwrap();
        touch(td.path(), "app/MyApp.xcodeproj/project.pbxproj");
        touch(td.path(), "app/Main.swift");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::XcodeProject);
    }

    #[test]
    fn settings_gradle_detects_android_project() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "app/settings.gradle");
        touch(td.path(), "app/app/build.gradle");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::AndroidStudioProject);
    }

    #[test]
    fn two_ipynb_neighbors_detect_notebook_set() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "nb/intro.ipynb");
        touch(td.path(), "nb/analysis.ipynb");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::JupyterNotebookSet);
    }

    #[test]
    fn single_ipynb_does_not_detect_notebook_set() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "nb/lone.ipynb");
        let tree = scan(td.path());
        assert!(tree.bundles.is_empty());
        assert_eq!(tree.loose_files.len(), 1);
    }

    #[test]
    fn bundle_subtrees_are_not_descended() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "proj/Cargo.toml");
        touch(td.path(), "proj/src/main.rs");
        touch(td.path(), "proj/src/sub/other.rs");
        touch(td.path(), "loose.txt");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.loose_files.len(), 1);
        assert_eq!(
            tree.loose_files[0]
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap(),
            "loose.txt",
        );
    }

    #[test]
    fn nested_bundles_do_not_emit_outer_bundle() {
        // A directory containing a bundle subdir is NOT itself a bundle unless
        // it has its own marker. The outer dir should be transparent.
        let td = TempDir::new().unwrap();
        touch(td.path(), "projects/one/Cargo.toml");
        touch(td.path(), "projects/one/src/lib.rs");
        touch(td.path(), "projects/two/package.json");
        touch(td.path(), "projects/two/index.js");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 2);
        assert!(tree.loose_files.is_empty());
        let kinds: Vec<&BundleKind> = tree.bundles.iter().map(|b| &b.kind).collect();
        assert!(kinds.contains(&&BundleKind::RustCrate));
        assert!(kinds.contains(&&BundleKind::NodeProject));
    }

    #[test]
    fn noise_files_are_skipped_in_loose() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "doc.txt");
        touch(td.path(), ".DS_Store");
        touch(td.path(), "sub/Thumbs.db");
        touch(td.path(), "sub/real.md");
        let tree = scan(td.path());
        assert_eq!(tree.loose_files.len(), 2);
        for p in &tree.loose_files {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap();
            assert!(name == "doc.txt" || name == "real.md");
        }
    }

    #[test]
    fn bundle_members_include_dotfiles() {
        // Inside a bundle everything is preserved — including .gitignore,
        // lockfiles, etc. The subtree is opaque.
        let td = TempDir::new().unwrap();
        touch(td.path(), "proj/Cargo.toml");
        touch(td.path(), "proj/Cargo.lock");
        touch(td.path(), "proj/.gitignore");
        touch(td.path(), "proj/src/main.rs");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].members.len(), 4);
    }
}
