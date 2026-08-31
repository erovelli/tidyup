//! Source-directory walker with bundle detection.
//!
//! The first pass of every pipeline run. The scanner walks the source tree,
//! marks recognized self-contained subtrees as [`DetectedBundle`]s, and
//! recursively collects ordinary files as loose entries. Unmarked directories
//! are transparent to discovery so nested projects still win at their own
//! marker roots and ordinary descendants reach semantic classification.
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
//! When multiple markers coexist in the same directory the most specific wins.
//! Directories with no marker but containing *two or more* sibling `.ipynb`
//! files become [`BundleKind::JupyterNotebookSet`]. Unknown directories are
//! traversed rather than turned into opaque bundles; preserving or adapting
//! their relative placement is a later planning concern, not a walk boundary.
//!
//! # Scope
//!
//! This module covers **marker-file (directory) bundles only** — it
//! deliberately performs no content reads. Content-clustered bundles
//! (EXIF photo bursts, ID3 music albums, filename `DocumentSeries`) are a
//! separate pass over the loose files in [`crate::clustering`], which runs
//! after this scanner and uses the content extractors. Those clusters move as
//! file-sets (each member individually, atomically) rather than by renaming a
//! shared root directory.

use std::fs;
use std::path::{Path, PathBuf};

use tidyup_domain::bundle::BundleKind;
use tidyup_domain::{DirectoryEnvelope, DirectoryEnvelopeConfig, DiscoveryConfig};
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
    /// For file-set clusters (photo bursts, music albums, document series), the
    /// subfolder name the members are grouped under at the destination (e.g.
    /// `"Burst 2024-01-15"`, an album title, or a filename family). `None` for
    /// directory bundles, which preserve their existing subtree layout.
    pub target_subdir: Option<String>,
    /// Hierarchy-selected envelope metadata. `None` for legacy marker bundles
    /// and file-set clusters.
    pub envelope: Option<DirectoryEnvelope>,
}

/// Walk `root`, stop at recognized bundle roots, and recursively collect all
/// other regular files as loose entries.
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

/// Scan using the configured hierarchy policy.
///
/// The legacy scanner remains available for compatibility and calibration. When
/// generalized envelopes are enabled, parent-first discovery selects
/// non-overlapping directory roots and exposes only files from confidently
/// heterogeneous containers to loose classification and clustering.
#[must_use]
pub fn scan_with_policy(
    root: &Path,
    config: &DirectoryEnvelopeConfig,
    discovery: &DiscoveryConfig,
) -> ScanTree {
    if !config.enabled {
        return scan_with_discovery(root, discovery);
    }
    let discovered = crate::envelopes::discover(root, config, discovery);
    ScanTree {
        root: root.to_path_buf(),
        bundles: discovered
            .envelopes
            .into_iter()
            .map(|envelope| DetectedBundle {
                root: envelope.root,
                kind: BundleKind::DirectoryEnvelope,
                members: envelope.members,
                reasoning: "hierarchy-selected directory envelope".to_string(),
                target_subdir: None,
                envelope: Some(envelope.metadata),
            })
            .collect(),
        loose_files: discovered.loose_files,
    }
}

fn scan_with_discovery(root: &Path, discovery: &DiscoveryConfig) -> ScanTree {
    let mut tree = ScanTree {
        root: root.to_path_buf(),
        bundles: Vec::new(),
        loose_files: Vec::new(),
    };
    scan_dir_with_discovery(root, &mut tree, discovery);
    tree
}

fn scan_dir(dir: &Path, tree: &mut ScanTree) {
    scan_dir_with_discovery(dir, tree, &DiscoveryConfig::default());
}

fn scan_dir_with_discovery(dir: &Path, tree: &mut ScanTree, discovery: &DiscoveryConfig) {
    if let Some((kind, reason)) = detect_bundle(dir) {
        let members = collect_members(dir);
        tree.bundles.push(DetectedBundle {
            root: dir.to_path_buf(),
            kind,
            members,
            reasoning: reason.to_string(),
            target_subdir: None,
            envelope: None,
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
        if is_ignored(&path, discovery) {
            continue;
        }
        if file_type.is_dir() {
            scan_dir_with_discovery(&path, tree, discovery);
        } else if file_type.is_file() {
            tree.loose_files.push(path);
        }
    }
}

/// Inspect the direct children of `dir` and decide whether it roots a bundle.
///
/// Returns `Some((kind, reason))` on match. Precedence runs from most-specific
/// marker to least; see module docs for rationale.
/// Return descriptive marker evidence for a directory.
///
/// Generalized directory envelopes consume this as provenance only; marker
/// matching must never become a destination-routing rule.
pub(crate) fn detect_bundle(dir: &Path) -> Option<(BundleKind, &'static str)> {
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
                    notebook_count = notebook_count.saturating_add(1);
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
pub(crate) fn is_ignored(path: &Path, config: &DiscoveryConfig) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    config.ignore_names.iter().any(|ignored| ignored == name)
        || (!config.include_hidden && name.starts_with('.'))
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
    fn files_inside_unknown_directories_remain_loose() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "a.txt");
        touch(td.path(), "sub/b.md");
        touch(td.path(), "sub/nested/c.pdf");
        let tree = scan(td.path());
        assert!(tree.bundles.is_empty());
        assert_eq!(tree.loose_files.len(), 3);
        assert!(tree.loose_files.iter().any(|path| path.ends_with("a.txt")));
        assert!(tree
            .loose_files
            .iter()
            .any(|path| path.ends_with("sub/b.md")));
        assert!(tree
            .loose_files
            .iter()
            .any(|path| path.ends_with("sub/nested/c.pdf")));
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
        assert_eq!(tree.loose_files, [td.path().join("nb/lone.ipynb")]);
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
    fn sibling_projects_under_an_intermediate_directory_are_detected_separately() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "projects/one/Cargo.toml");
        touch(td.path(), "projects/one/src/lib.rs");
        touch(td.path(), "projects/two/package.json");
        touch(td.path(), "projects/two/index.js");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 2);
        assert!(tree.loose_files.is_empty());
        assert!(tree.bundles.iter().any(|bundle| {
            bundle.root.ends_with("projects/one") && bundle.kind == BundleKind::RustCrate
        }));
        assert!(tree.bundles.iter().any(|bundle| {
            bundle.root.ends_with("projects/two") && bundle.kind == BundleKind::NodeProject
        }));
    }

    #[test]
    fn project_nested_multiple_levels_below_root_is_detected() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "dev/repos/myproject/Cargo.toml");
        touch(td.path(), "dev/repos/myproject/src/main.rs");
        let tree = scan(td.path());
        assert_eq!(tree.bundles.len(), 1);
        assert_eq!(tree.bundles[0].kind, BundleKind::RustCrate);
        assert!(tree.bundles[0].root.ends_with("dev/repos/myproject"));
        assert_eq!(tree.bundles[0].members.len(), 2);
        assert!(tree.loose_files.is_empty());
    }

    #[test]
    fn ordinary_files_in_plain_subdirectory_are_individual_loose_inputs() {
        let td = TempDir::new().unwrap();
        touch(td.path(), "plainfolder/notes.txt");
        touch(td.path(), "plainfolder/agenda.txt");
        let tree = scan(td.path());
        assert!(tree.bundles.is_empty());
        assert_eq!(tree.loose_files.len(), 2);
        assert!(tree
            .loose_files
            .iter()
            .all(|path| path.parent() == Some(td.path().join("plainfolder").as_path())));
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
        assert!(tree
            .loose_files
            .iter()
            .any(|path| path.ends_with("doc.txt")));
        assert!(tree
            .loose_files
            .iter()
            .any(|path| path.ends_with("sub/real.md")));
        assert!(tree.bundles.is_empty());
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
