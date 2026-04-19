//! Target-tree profiler — walks the destination hierarchy and produces a
//! [`TargetScan`] + [`ProfileCache`] consumed by the migration-mode cascade.
//!
//! # What a profile is
//!
//! Each directory in the target tree becomes a [`FolderProfile`] composed of:
//!
//! - **`name_embedding`** — embedding of a synthesized description combining
//!   folder name, path segments, dominant extensions, and file count. Always
//!   present. This is what lets a freshly-created empty folder still receive
//!   classifications on first sight.
//! - **`content_centroid`** — mean of per-file content embeddings for files
//!   directly in the folder. Optional in v0.1: only populated when a caller
//!   supplies per-file embeddings. The migration cascade gracefully weights
//!   around a missing centroid (see `scoring::score_candidate`).
//! - **`metadata`** — [`FolderMetadata`] snapshot: extension counts, file
//!   counts, date range, BLAKE3 content hash (for cache invalidation).
//! - **`organization_type`** — [`OrganizationType`] detected from child-folder
//!   names: year/quarter/month buckets, workflow-status folders, or plain
//!   semantic groupings.
//!
//! # Caching
//!
//! The profile cache invalidates by `FolderMetadata.content_hash`, not by
//! timestamp — consistent with the rule in `CLAUDE.md`. A
//! [`ScanDiff`](tidyup_domain::migration::ScanDiff) computed against a prior
//! scan tells callers which profiles need rebuilding.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use tidyup_core::inference::EmbeddingBackend;
use tidyup_domain::migration::{
    DatePattern, FolderMetadata, FolderNode, FolderProfile, OrganizationType, ProfileCache,
    ScanDiff, TargetScan,
};

/// Walk `root` and build a [`TargetScan`] describing the folder hierarchy.
///
/// Symlinks are not followed. Unreadable descendants are logged and skipped.
///
/// # Errors
/// Returns `io::Error` if `root` itself cannot be read. Per-descendant
/// failures are swallowed and logged.
pub fn scan_target(root: &Path) -> io::Result<TargetScan> {
    let root = root.to_path_buf();
    let mut nodes: HashMap<PathBuf, FolderNode> = HashMap::new();
    let mut leaf_folders: Vec<PathBuf> = Vec::new();

    // Seed with the root itself so callers get a profile for it, too.
    let root_node = build_node(&root, &root, &[])?;
    let has_children = root_node.metadata.has_children;
    nodes.insert(root.clone(), root_node);
    if !has_children {
        leaf_folders.push(root.clone());
    }

    collect_nodes(&root, &root, &[], &mut nodes, &mut leaf_folders);

    Ok(TargetScan {
        root,
        nodes,
        leaf_folders,
        scan_timestamp: SystemTime::now(),
    })
}

fn collect_nodes(
    root: &Path,
    dir: &Path,
    parent_segments: &[String],
    nodes: &mut HashMap<PathBuf, FolderNode>,
    leaves: &mut Vec<PathBuf>,
) {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::warn!("profiler: unable to read {}: {e}", dir.display());
            return;
        }
    };

    let mut current_segments = parent_segments.to_vec();
    if dir != root {
        if let Some(name) = dir.file_name().and_then(|s| s.to_str()) {
            current_segments.push(name.to_string());
        }
    }

    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() || !ft.is_dir() {
            continue;
        }
        let path = entry.path();
        let node = match build_node(root, &path, &current_segments) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("profiler: build_node failed for {}: {e}", path.display());
                continue;
            }
        };
        let is_leaf = !node.metadata.has_children;
        nodes.insert(path.clone(), node);
        if is_leaf {
            leaves.push(path.clone());
        }
        collect_nodes(root, &path, &current_segments, nodes, leaves);
    }
}

fn build_node(root: &Path, dir: &Path, parent_segments: &[String]) -> io::Result<FolderNode> {
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();

    let depth = if dir == root {
        0
    } else {
        u32::try_from(parent_segments.len() + 1).unwrap_or(u32::MAX)
    };

    let mut path_segments = parent_segments.to_vec();
    if dir != root {
        path_segments.push(name.clone());
    }

    let mut children: Vec<PathBuf> = Vec::new();
    let mut file_count: u32 = 0;
    let mut extension_counts: HashMap<String, u32> = HashMap::new();
    let mut total_size: u64 = 0;
    let mut size_count: u64 = 0;
    let mut earliest: Option<SystemTime> = None;
    let mut latest: Option<SystemTime> = None;
    let mut hasher = blake3::Hasher::new();
    let mut names: Vec<String> = Vec::new();

    for entry in fs::read_dir(dir)?.flatten() {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        let path = entry.path();
        if ft.is_dir() {
            children.push(path);
        } else if ft.is_file() {
            file_count = file_count.saturating_add(1);
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                names.push(name.to_string());
            }
            if let Some(ext) = path
                .extension()
                .and_then(|s| s.to_str())
                .map(str::to_ascii_lowercase)
            {
                *extension_counts.entry(format!(".{ext}")).or_insert(0) += 1;
            }
            if let Ok(meta) = entry.metadata() {
                total_size = total_size.saturating_add(meta.len());
                size_count = size_count.saturating_add(1);
                if let Ok(modified) = meta.modified() {
                    earliest = Some(earliest.map_or(modified, |e| e.min(modified)));
                    latest = Some(latest.map_or(modified, |l| l.max(modified)));
                }
            }
        }
    }

    names.sort();
    for n in &names {
        hasher.update(n.as_bytes());
        hasher.update(b"\0");
    }

    let recursive_file_count = count_files_recursively(dir);
    let dominant_extensions = top_extensions(&extension_counts, 3);
    let sibling_names = sibling_names(dir);
    let has_children = !children.is_empty();
    #[allow(clippy::cast_possible_truncation)]
    let avg_file_size = if size_count == 0 {
        0
    } else {
        total_size / size_count
    };

    let metadata = FolderMetadata {
        file_count,
        recursive_file_count,
        extension_counts,
        dominant_extensions,
        date_range: earliest.zip(latest),
        avg_file_size,
        has_children,
        content_hash: hasher.finalize().to_hex().to_string(),
        scanned_at: SystemTime::now(),
    };

    Ok(FolderNode {
        path: dir.to_path_buf(),
        name,
        path_segments,
        depth,
        children,
        sibling_names,
        metadata,
    })
}

fn count_files_recursively(dir: &Path) -> u32 {
    let mut count: u32 = 0;
    for entry in walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
    {
        if entry.file_type().is_file() {
            count = count.saturating_add(1);
        }
    }
    count
}

fn sibling_names(dir: &Path) -> Vec<String> {
    let Some(parent) = dir.parent() else {
        return Vec::new();
    };
    let Some(self_name) = dir.file_name() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(parent) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .filter_map(|e| {
            let name = e.file_name();
            if name == self_name {
                None
            } else {
                name.to_str().map(str::to_string)
            }
        })
        .collect()
}

fn top_extensions(counts: &HashMap<String, u32>, k: usize) -> Vec<String> {
    let mut pairs: Vec<(&String, &u32)> = counts.iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    pairs.into_iter().take(k).map(|(e, _)| e.clone()).collect()
}

// ---------------------------------------------------------------------------
// Organization-type detection
// ---------------------------------------------------------------------------

const STATUS_WORDS: &[&str] = &[
    "todo",
    "doing",
    "done",
    "active",
    "archive",
    "archived",
    "draft",
    "drafts",
    "pending",
    "review",
    "reviewed",
    "complete",
    "completed",
    "inbox",
    "backlog",
    "in-progress",
    "wip",
];

/// Detect the organisational dimension of a folder from its child-folder names.
///
/// A folder counts as date-based if ≥50 % of its direct children match a
/// consistent date-bucket shape. Status-based fires on any workflow-term
/// hit in children. A folder with no children defaults to `Semantic`.
#[must_use]
pub fn detect_organization<S: std::hash::BuildHasher>(
    node: &FolderNode,
    all_nodes: &HashMap<PathBuf, FolderNode, S>,
) -> OrganizationType {
    let child_names: Vec<String> = node
        .children
        .iter()
        .filter_map(|p| all_nodes.get(p).map(|n| n.name.to_ascii_lowercase()))
        .collect();
    if child_names.is_empty() {
        return OrganizationType::Semantic;
    }

    let total = child_names.len();

    let year_hits = child_names.iter().filter(|n| is_year_bucket(n)).count();
    if year_hits * 2 >= total {
        return OrganizationType::DateBased {
            pattern: DatePattern::Year,
        };
    }

    let quarter_hits = child_names.iter().filter(|n| is_quarter_bucket(n)).count();
    if quarter_hits * 2 >= total {
        return OrganizationType::DateBased {
            pattern: DatePattern::Quarter,
        };
    }

    let month_hits = child_names.iter().filter(|n| is_month_bucket(n)).count();
    if month_hits * 2 >= total {
        return OrganizationType::DateBased {
            pattern: DatePattern::Month,
        };
    }

    if child_names
        .iter()
        .any(|n| STATUS_WORDS.contains(&n.as_str()))
    {
        return OrganizationType::StatusBased;
    }

    OrganizationType::Semantic
}

fn is_year_bucket(name: &str) -> bool {
    let trimmed = name.trim_matches(|c: char| c == '_' || c == '-');
    trimmed.len() == 4
        && trimmed.chars().all(|c| c.is_ascii_digit())
        && trimmed
            .parse::<i32>()
            .ok()
            .is_some_and(|y| (1900..=2100).contains(&y))
}

fn is_quarter_bucket(name: &str) -> bool {
    // 2024-Q1, 2024_Q1, Q1-2024, Q1_2024
    let parts: Vec<&str> = name.split(['-', '_']).collect();
    if parts.len() != 2 {
        return false;
    }
    let (a, b) = (parts[0], parts[1]);
    is_year_token(a) && is_quarter_token(b) || is_quarter_token(a) && is_year_token(b)
}

fn is_month_bucket(name: &str) -> bool {
    // 2024-01, 2024_01, 01-2024, jan-2024, 2024-jan
    let parts: Vec<&str> = name.split(['-', '_']).collect();
    if parts.len() != 2 {
        return false;
    }
    let (a, b) = (parts[0], parts[1]);
    (is_year_token(a) && is_month_token(b)) || (is_month_token(a) && is_year_token(b))
}

fn is_year_token(s: &str) -> bool {
    s.len() == 4 && s.chars().all(|c| c.is_ascii_digit())
}

fn is_quarter_token(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    matches!(s.as_str(), "q1" | "q2" | "q3" | "q4")
}

fn is_month_token(s: &str) -> bool {
    if s.len() == 2 && s.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(m) = s.parse::<u8>() {
            return (1..=12).contains(&m);
        }
    }
    matches!(
        s,
        "jan"
            | "feb"
            | "mar"
            | "apr"
            | "may"
            | "jun"
            | "jul"
            | "aug"
            | "sep"
            | "oct"
            | "nov"
            | "dec"
            | "january"
            | "february"
            | "march"
            | "april"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
    )
}

// ---------------------------------------------------------------------------
// Description synthesis + embedding
// ---------------------------------------------------------------------------

/// Compose the natural-language description fed into the folder name embedding.
///
/// Shape: `"<segments joined with /> — folder containing <N> files; dominant
/// extensions: .ext, .ext, .ext."`. Empty folders drop the extension clause.
///
/// The description is deliberately descriptive, not imperative — it's encoded
/// by the same BGE model that embeds file content, and symmetric phrasing
/// keeps the two sides of the cosine comparable.
#[must_use]
pub fn synthesize_description(node: &FolderNode) -> String {
    let path_display = if node.path_segments.is_empty() {
        node.name.clone()
    } else {
        node.path_segments.join(" / ")
    };

    let count = node.metadata.file_count;
    let ext_clause = if node.metadata.dominant_extensions.is_empty() {
        String::new()
    } else {
        format!(
            "; dominant extensions: {}",
            node.metadata.dominant_extensions.join(", "),
        )
    };

    if count == 0 {
        format!("{path_display} — folder")
    } else {
        format!("{path_display} — folder containing {count} files{ext_clause}")
    }
}

/// Build a [`ProfileCache`] from a [`TargetScan`].
///
/// For each folder, synthesizes a description, embeds it, and assembles the
/// [`FolderProfile`]. `content_centroid` is left as `None` in v0.1 — it's a
/// pure additive signal, and the migration scorer weights around its absence
/// (see `migration` module).
///
/// # Errors
/// Propagates embedding backend failures.
pub async fn build_profile_cache(
    scan: &TargetScan,
    embeddings: &dyn EmbeddingBackend,
) -> Result<ProfileCache> {
    // Batch-embed all descriptions in insertion order for determinism.
    let mut paths: Vec<PathBuf> = scan.nodes.keys().cloned().collect();
    paths.sort();

    let descriptions: Vec<String> = paths
        .iter()
        .filter_map(|p| scan.nodes.get(p).map(synthesize_description))
        .collect();
    let refs: Vec<&str> = descriptions.iter().map(String::as_str).collect();

    let mut name_vectors: Vec<Vec<f32>> = Vec::with_capacity(refs.len());
    for chunk in refs.chunks(32) {
        let mut batch = embeddings.embed_texts(chunk).await?;
        name_vectors.append(&mut batch);
    }

    let now = SystemTime::now();
    let mut profiles: HashMap<PathBuf, FolderProfile> = HashMap::with_capacity(paths.len());
    for (path, name_embedding) in paths.into_iter().zip(name_vectors.into_iter()) {
        let Some(node) = scan.nodes.get(&path) else {
            continue;
        };
        let organization_type = detect_organization(node, &scan.nodes);
        let profile_confidence = estimate_profile_confidence(node);
        profiles.insert(
            path.clone(),
            FolderProfile {
                path,
                name_embedding,
                content_centroid: None,
                centroid_sample_count: 0,
                metadata: node.metadata.clone(),
                organization_type,
                profile_confidence,
                last_updated: now,
            },
        );
    }

    Ok(ProfileCache {
        target_root: scan.root.clone(),
        model_id: embeddings.model_id().to_string(),
        embedding_dim: embeddings.dimensions(),
        profiles,
        last_scan: scan.clone(),
        created_at: now,
        last_updated: now,
    })
}

/// Heuristic profile confidence in `[0.0, 1.0]` used as a tiebreaker during
/// classification. A folder with many files and a descriptive name gives a
/// strong signal; a nearly-empty folder gives a weak one.
#[must_use]
pub fn estimate_profile_confidence(node: &FolderNode) -> f32 {
    let name_len = node.name.chars().count();
    let name_score = if name_len == 0 {
        0.0
    } else if name_len < 3 {
        0.3
    } else if name_len < 6 {
        0.6
    } else {
        1.0
    };

    let files = f32::from(u16::try_from(node.metadata.file_count.min(50)).unwrap_or(u16::MAX));
    let file_score = (files / 50.0).min(1.0);

    0.6_f32.mul_add(name_score, 0.4 * file_score)
}

// ---------------------------------------------------------------------------
// Scan diffing for incremental profile rebuilds
// ---------------------------------------------------------------------------

/// Compare two [`TargetScan`]s and emit the set of folder changes.
///
/// `modified` contains paths whose `content_hash` changed; this is the signal
/// an incremental profiler uses to decide which folders need re-embedding.
#[must_use]
pub fn diff_scans(previous: &TargetScan, current: &TargetScan) -> ScanDiff {
    let prev_paths: HashSet<&PathBuf> = previous.nodes.keys().collect();
    let curr_paths: HashSet<&PathBuf> = current.nodes.keys().collect();

    let added: Vec<PathBuf> = curr_paths
        .difference(&prev_paths)
        .map(|p| (*p).clone())
        .collect();
    let removed: Vec<PathBuf> = prev_paths
        .difference(&curr_paths)
        .map(|p| (*p).clone())
        .collect();

    let mut modified: Vec<PathBuf> = Vec::new();
    let mut unchanged: Vec<PathBuf> = Vec::new();
    for path in prev_paths.intersection(&curr_paths) {
        let prev_hash = previous.nodes.get(*path).map(|n| &n.metadata.content_hash);
        let curr_hash = current.nodes.get(*path).map(|n| &n.metadata.content_hash);
        if prev_hash == curr_hash {
            unchanged.push((*path).clone());
        } else {
            modified.push((*path).clone());
        }
    }

    ScanDiff {
        added,
        removed,
        modified,
        unchanged,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::fs;
    use tempfile::TempDir;

    fn touch(dir: &Path, rel: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, b"x").unwrap();
    }

    /// Deterministic stub backend: produces a length-5 vector seeded by the
    /// first char of the input plus its length. Enough structure for tests
    /// that compare whether two descriptions embed differently; no real
    /// meaning attached.
    struct FakeEmbeddings;

    #[async_trait]
    impl EmbeddingBackend for FakeEmbeddings {
        async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            #[allow(clippy::cast_precision_loss)]
            let c = text.chars().next().map_or(0.0, |c| c as u32 as f32);
            #[allow(clippy::cast_precision_loss)]
            let len = text.len() as f32;
            Ok(vec![c, len, c + len, c * 0.01, len * 0.01])
        }
        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed_text(t).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize {
            5
        }
        fn model_id(&self) -> &'static str {
            "fake"
        }
    }

    #[test]
    fn scan_empty_root() {
        let td = TempDir::new().unwrap();
        let scan = scan_target(td.path()).unwrap();
        assert_eq!(scan.nodes.len(), 1);
        assert_eq!(scan.leaf_folders.len(), 1);
    }

    #[test]
    fn scan_flat_hierarchy() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("Finance")).unwrap();
        fs::create_dir_all(td.path().join("Photos")).unwrap();
        touch(td.path(), "Finance/tax.pdf");
        touch(td.path(), "Photos/vacation.jpg");
        let scan = scan_target(td.path()).unwrap();
        // root + 2 children.
        assert_eq!(scan.nodes.len(), 3);
        assert_eq!(scan.leaf_folders.len(), 2);
    }

    #[test]
    fn scan_nested_hierarchy_captures_depth() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("a/b/c")).unwrap();
        touch(td.path(), "a/b/c/file.txt");
        let scan = scan_target(td.path()).unwrap();
        let c = scan.nodes.get(&td.path().join("a/b/c")).unwrap();
        assert_eq!(c.depth, 3);
        assert_eq!(c.path_segments, vec!["a", "b", "c"]);
    }

    #[test]
    fn metadata_captures_extension_counts() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("docs")).unwrap();
        touch(td.path(), "docs/a.pdf");
        touch(td.path(), "docs/b.pdf");
        touch(td.path(), "docs/c.md");
        let scan = scan_target(td.path()).unwrap();
        let docs = scan.nodes.get(&td.path().join("docs")).unwrap();
        assert_eq!(docs.metadata.file_count, 3);
        assert_eq!(docs.metadata.extension_counts.get(".pdf"), Some(&2));
        assert_eq!(docs.metadata.extension_counts.get(".md"), Some(&1));
        assert_eq!(docs.metadata.dominant_extensions[0], ".pdf");
    }

    #[test]
    fn content_hash_deterministic_on_same_content() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("dir")).unwrap();
        touch(td.path(), "dir/a.txt");
        touch(td.path(), "dir/b.txt");
        let scan1 = scan_target(td.path()).unwrap();
        let scan2 = scan_target(td.path()).unwrap();
        let h1 = &scan1
            .nodes
            .get(&td.path().join("dir"))
            .unwrap()
            .metadata
            .content_hash;
        let h2 = &scan2
            .nodes
            .get(&td.path().join("dir"))
            .unwrap()
            .metadata
            .content_hash;
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_changes_when_file_added() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("dir")).unwrap();
        touch(td.path(), "dir/a.txt");
        let scan1 = scan_target(td.path()).unwrap();
        touch(td.path(), "dir/b.txt");
        let scan2 = scan_target(td.path()).unwrap();
        let h1 = &scan1
            .nodes
            .get(&td.path().join("dir"))
            .unwrap()
            .metadata
            .content_hash;
        let h2 = &scan2
            .nodes
            .get(&td.path().join("dir"))
            .unwrap()
            .metadata
            .content_hash;
        assert_ne!(h1, h2);
    }

    #[test]
    fn year_organization_detected() {
        let td = TempDir::new().unwrap();
        for y in ["2021", "2022", "2023", "2024"] {
            fs::create_dir_all(td.path().join("Taxes").join(y)).unwrap();
        }
        let scan = scan_target(td.path()).unwrap();
        let taxes = scan.nodes.get(&td.path().join("Taxes")).unwrap();
        assert_eq!(
            detect_organization(taxes, &scan.nodes),
            OrganizationType::DateBased {
                pattern: DatePattern::Year,
            },
        );
    }

    #[test]
    fn quarter_organization_detected() {
        let td = TempDir::new().unwrap();
        for q in ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"] {
            fs::create_dir_all(td.path().join("Board").join(q)).unwrap();
        }
        let scan = scan_target(td.path()).unwrap();
        let board = scan.nodes.get(&td.path().join("Board")).unwrap();
        assert_eq!(
            detect_organization(board, &scan.nodes),
            OrganizationType::DateBased {
                pattern: DatePattern::Quarter,
            },
        );
    }

    #[test]
    fn status_organization_detected() {
        let td = TempDir::new().unwrap();
        for s in ["Active", "Done", "Backlog"] {
            fs::create_dir_all(td.path().join("Projects").join(s)).unwrap();
        }
        let scan = scan_target(td.path()).unwrap();
        let projects = scan.nodes.get(&td.path().join("Projects")).unwrap();
        assert_eq!(
            detect_organization(projects, &scan.nodes),
            OrganizationType::StatusBased,
        );
    }

    #[test]
    fn mixed_organization_falls_back_to_semantic() {
        let td = TempDir::new().unwrap();
        for name in ["Research", "Contracts", "Receipts"] {
            fs::create_dir_all(td.path().join("Work").join(name)).unwrap();
        }
        let scan = scan_target(td.path()).unwrap();
        let work = scan.nodes.get(&td.path().join("Work")).unwrap();
        assert_eq!(
            detect_organization(work, &scan.nodes),
            OrganizationType::Semantic,
        );
    }

    #[test]
    fn synthesize_description_includes_path_and_counts() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("Finance/Taxes")).unwrap();
        touch(td.path(), "Finance/Taxes/2024.pdf");
        let scan = scan_target(td.path()).unwrap();
        let node = scan.nodes.get(&td.path().join("Finance/Taxes")).unwrap();
        let desc = synthesize_description(node);
        assert!(desc.contains("Finance"));
        assert!(desc.contains("Taxes"));
        assert!(desc.contains(".pdf"));
        assert!(desc.contains("1 files"));
    }

    #[tokio::test]
    async fn build_profile_cache_populates_name_embeddings() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("Finance")).unwrap();
        touch(td.path(), "Finance/tax.pdf");
        let scan = scan_target(td.path()).unwrap();
        let cache = build_profile_cache(&scan, &FakeEmbeddings).await.unwrap();
        assert_eq!(cache.model_id, "fake");
        assert_eq!(cache.embedding_dim, 5);
        let finance = cache.profiles.get(&td.path().join("Finance")).unwrap();
        assert_eq!(finance.name_embedding.len(), 5);
        assert!(finance.content_centroid.is_none());
    }

    #[test]
    fn diff_detects_added_removed_modified() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("a")).unwrap();
        fs::create_dir_all(td.path().join("b")).unwrap();
        touch(td.path(), "a/x.txt");
        let prev = scan_target(td.path()).unwrap();

        fs::remove_dir_all(td.path().join("b")).unwrap();
        fs::create_dir_all(td.path().join("c")).unwrap();
        touch(td.path(), "a/y.txt"); // modifies "a"
        let curr = scan_target(td.path()).unwrap();

        let diff = diff_scans(&prev, &curr);
        assert!(diff.added.iter().any(|p| p.ends_with("c")));
        assert!(diff.removed.iter().any(|p| p.ends_with("b")));
        assert!(diff.modified.iter().any(|p| p.ends_with("a")));
    }

    #[test]
    fn profile_confidence_rewards_descriptive_name_and_file_count() {
        let td = TempDir::new().unwrap();
        fs::create_dir_all(td.path().join("Finance")).unwrap();
        for i in 0..10 {
            touch(td.path(), &format!("Finance/file{i}.pdf"));
        }
        fs::create_dir_all(td.path().join("x")).unwrap();
        let scan = scan_target(td.path()).unwrap();
        let rich = scan.nodes.get(&td.path().join("Finance")).unwrap();
        let poor = scan.nodes.get(&td.path().join("x")).unwrap();
        assert!(estimate_profile_confidence(rich) > estimate_profile_confidence(poor));
    }
}
