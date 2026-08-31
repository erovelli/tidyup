//! Hierarchical directory-boundary discovery.
//!
//! This module deliberately separates *whether a directory should stay
//! together* from *where it should be routed*.  Discovery is parent-first and
//! non-overlapping: once a parent is selected as an envelope, descendants are
//! evidence only.  Known project markers enrich the evidence but are never a
//! gate for selecting an envelope.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tidyup_domain::{
    DirectoryBoundary, DirectoryEnvelope, DirectoryEnvelopeConfig, DirectorySnapshot,
};
use walkdir::WalkDir;

use crate::hashing::content_hash_of;
use crate::scanner;

/// A selected non-overlapping directory boundary.
#[derive(Debug, Clone)]
pub struct DetectedEnvelope {
    pub root: PathBuf,
    /// Regular-file members are retained for current per-file processing
    /// accounting. The root snapshot remains authoritative for execution.
    pub members: Vec<PathBuf>,
    pub metadata: DirectoryEnvelope,
}

/// Result of parent-first hierarchy discovery.
#[derive(Debug, Clone, Default)]
pub struct EnvelopeDiscovery {
    pub envelopes: Vec<DetectedEnvelope>,
    /// Files exposed by directories confidently identified as containers.
    pub loose_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default)]
struct Node {
    path: PathBuf,
    files: Vec<PathBuf>,
    children: Vec<PathBuf>,
    unreadable: bool,
}

#[derive(Debug, Clone)]
struct Profile {
    cohesion: f32,
    boundary: BoundaryDecision,
    snapshot: DirectorySnapshot,
    provenance: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryDecision {
    Cohesive,
    Container,
    Uncertain,
}

/// Build a deterministic snapshot for a directory without following symlinks.
///
/// A failed entry remains part of the digest as an error token and marks the
/// result incomplete. Callers must hold incomplete snapshots from unattended
/// application rather than pretending the unseen subtree is unchanged.
#[must_use]
pub fn snapshot_directory(root: &Path) -> DirectorySnapshot {
    let mut tokens = Vec::new();
    let mut regular_files = 0_u64;
    let mut directories = 0_u64;
    let mut symlinks = 0_u64;
    let mut total_bytes = 0_u64;
    let mut complete = true;

    if !root.is_dir() {
        return DirectorySnapshot {
            digest: blake3::hash(b"missing-directory").to_hex().to_string(),
            regular_files,
            directories,
            symlinks,
            total_bytes,
            complete: false,
        };
    }

    for entry in WalkDir::new(root)
        .follow_links(false)
        .min_depth(1)
        .sort_by_file_name()
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                complete = false;
                let rel = error
                    .path()
                    .and_then(|path| path.strip_prefix(root).ok())
                    .map_or_else(|| "<unknown>".to_string(), relative_string);
                tokens.push(format!("E:{rel}"));
                continue;
            }
        };
        let path = entry.path();
        let rel = relative_string(path.strip_prefix(root).unwrap_or(path));
        let Ok(metadata) = std::fs::symlink_metadata(path) else {
            complete = false;
            tokens.push(format!("E:{rel}"));
            continue;
        };
        let mode = permission_mode(&metadata);
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            symlinks = symlinks.saturating_add(1);
            if let Ok(target) = std::fs::read_link(path) {
                tokens.push(format!("L:{rel}:{mode}:{}", target.display()));
            } else {
                complete = false;
                tokens.push(format!("L:{rel}:{mode}:<unreadable>"));
            }
        } else if file_type.is_dir() {
            directories = directories.saturating_add(1);
            tokens.push(format!("D:{rel}:{mode}"));
        } else if file_type.is_file() {
            regular_files = regular_files.saturating_add(1);
            total_bytes = total_bytes.saturating_add(metadata.len());
            if let Some(hash) = content_hash_of(path) {
                tokens.push(format!("F:{rel}:{mode}:{hash}"));
            } else {
                complete = false;
                tokens.push(format!("F:{rel}:{mode}:<unreadable>"));
            }
        } else {
            complete = false;
            tokens.push(format!("O:{rel}:{mode}"));
        }
    }

    tokens.sort_unstable();
    let mut hasher = blake3::Hasher::new();
    for token in tokens {
        hasher.update(token.as_bytes());
        hasher.update(b"\n");
    }
    DirectorySnapshot {
        digest: hasher.finalize().to_hex().to_string(),
        regular_files,
        directories,
        symlinks,
        total_bytes,
        complete,
    }
}

/// Select non-overlapping directory envelopes under `source_root`.
///
/// The requested root remains a container even if its contents are highly
/// coherent. This permits a Desktop-like root to expose unrelated immediate
/// files while still preserving a coherent nested directory.
#[must_use]
pub fn discover(source_root: &Path, config: &DirectoryEnvelopeConfig) -> EnvelopeDiscovery {
    let nodes = collect_nodes(source_root);
    let mut profiles = BTreeMap::new();
    let mut paths: Vec<PathBuf> = nodes.keys().cloned().collect();
    paths.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for path in paths {
        let Some(node) = nodes.get(&path) else {
            continue;
        };
        let profile = profile_node(node, &profiles, config);
        profiles.insert(path, profile);
    }

    let mut out = EnvelopeDiscovery::default();
    let Some(root) = nodes.get(source_root) else {
        return out;
    };
    out.loose_files.extend(root.files.iter().cloned());
    let mut children = root.children.clone();
    children.sort();
    for child in children {
        select(&child, &nodes, &profiles, &mut out);
    }
    out.loose_files.sort();
    out.envelopes.sort_by(|a, b| a.root.cmp(&b.root));
    out
}

fn select(
    path: &Path,
    nodes: &BTreeMap<PathBuf, Node>,
    profiles: &BTreeMap<PathBuf, Profile>,
    out: &mut EnvelopeDiscovery,
) {
    let Some(node) = nodes.get(path) else {
        return;
    };
    let Some(profile) = profiles.get(path) else {
        return;
    };
    if profile.boundary == BoundaryDecision::Container {
        out.loose_files.extend(node.files.iter().cloned());
        let mut children = node.children.clone();
        children.sort();
        for child in children {
            select(&child, nodes, profiles, out);
        }
        return;
    }

    let mut members = Vec::new();
    collect_regular_members(path, &mut members);
    let boundary = match profile.boundary {
        BoundaryDecision::Cohesive => DirectoryBoundary::Cohesive,
        BoundaryDecision::Uncertain | BoundaryDecision::Container => DirectoryBoundary::Uncertain,
    };
    let requires_review =
        matches!(boundary, DirectoryBoundary::Uncertain) || !profile.snapshot.complete;
    let evidence = vec![format!(
        "hierarchy cohesion={:.3}; {} regular files; {} directories; {} symlinks",
        profile.cohesion,
        profile.snapshot.regular_files,
        profile.snapshot.directories,
        profile.snapshot.symlinks,
    )];
    out.envelopes.push(DetectedEnvelope {
        root: path.to_path_buf(),
        members,
        metadata: DirectoryEnvelope {
            boundary,
            cohesion: profile.cohesion,
            snapshot: profile.snapshot.clone(),
            evidence,
            provenance: profile.provenance.clone(),
            requires_review,
        },
    });
}

fn collect_nodes(root: &Path) -> BTreeMap<PathBuf, Node> {
    let mut nodes = BTreeMap::new();
    collect_node(root, &mut nodes);
    nodes
}

fn collect_node(path: &Path, nodes: &mut BTreeMap<PathBuf, Node>) {
    let mut node = Node {
        path: path.to_path_buf(),
        ..Node::default()
    };
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(
                "directory-envelope: unable to read {}: {error}",
                path.display()
            );
            node.unreadable = true;
            nodes.insert(path.to_path_buf(), node);
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(
                    "directory-envelope: bad entry in {}: {error}",
                    path.display()
                );
                node.unreadable = true;
                continue;
            }
        };
        let entry_path = entry.path();
        match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => {
                node.children.push(entry_path.clone());
                collect_node(&entry_path, nodes);
            }
            Ok(file_type) if file_type.is_file() => node.files.push(entry_path),
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(
                    "directory-envelope: file type failed for {}: {error}",
                    entry_path.display()
                );
                node.unreadable = true;
            }
        }
    }
    node.files.sort();
    node.children.sort();
    nodes.insert(path.to_path_buf(), node);
}

fn profile_node(
    node: &Node,
    child_profiles: &BTreeMap<PathBuf, Profile>,
    config: &DirectoryEnvelopeConfig,
) -> Profile {
    let snapshot = snapshot_directory(&node.path);
    let marker = scanner::detect_bundle(&node.path)
        .map(|(kind, reason)| format!("{}: {reason}", kind.as_str()));
    let provenance = marker.iter().cloned().collect::<Vec<_>>();
    let labels = node
        .files
        .iter()
        .chain(node.children.iter())
        // File suffixes are deliberately excluded: a shared extension is not
        // evidence that two files belong to the same directory envelope.
        .filter_map(|path| {
            path.file_stem()
                .or_else(|| path.file_name())
                .and_then(|name| name.to_str())
        })
        .map(normalize_label)
        .filter(|label| !label.is_empty())
        .take(config.sample_cap)
        .collect::<Vec<_>>();
    let lexical = mean_pair_affinity(&labels);
    let child_affinity = mean_child_affinity(&node.children, child_profiles);
    let only_directories = !node.children.is_empty() && node.files.is_empty();
    let marker_bonus = if marker.is_some() { 0.30 } else { 0.0 };
    let base = if labels.is_empty() {
        0.50
    } else if labels.len() == 1 {
        0.65
    } else if only_directories {
        0.75_f32.mul_add(lexical.max(child_affinity), 0.25)
    } else {
        0.85_f32.mul_add(lexical.max(child_affinity), 0.15)
    };
    let cohesion = (base + marker_bonus).min(1.0);
    let boundary = if node.unreadable || !snapshot.complete {
        BoundaryDecision::Uncertain
    } else if cohesion >= config.cohesive_threshold {
        BoundaryDecision::Cohesive
    } else if cohesion <= config.container_threshold && labels.len() >= 2 {
        BoundaryDecision::Container
    } else {
        BoundaryDecision::Uncertain
    };
    Profile {
        cohesion,
        boundary,
        snapshot,
        provenance,
    }
}

fn collect_regular_members(root: &Path, out: &mut Vec<PathBuf>) {
    for entry in WalkDir::new(root)
        .follow_links(false)
        .min_depth(1)
        .into_iter()
        .flatten()
    {
        if entry.file_type().is_file() {
            out.push(entry.into_path());
        }
    }
    out.sort();
}

fn mean_child_affinity(children: &[PathBuf], profiles: &BTreeMap<PathBuf, Profile>) -> f32 {
    let values = children
        .iter()
        .filter_map(|path| profiles.get(path).map(|profile| profile.cohesion))
        .collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / usize_to_f32(values.len())
    }
}

fn mean_pair_affinity(labels: &[String]) -> f32 {
    if labels.len() < 2 {
        return 1.0;
    }
    let token_sets = labels
        .iter()
        .map(|label| label.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut total = 0.0_f32;
    let mut pairs = 0_u32;
    for (index, left) in token_sets.iter().enumerate() {
        for right in token_sets.iter().skip(index.saturating_add(1)) {
            let common = left.iter().filter(|token| right.contains(token)).count();
            let union = left
                .len()
                .saturating_add(right.len())
                .saturating_sub(common);
            total += if union == 0 {
                0.0
            } else {
                usize_to_f32(common) / usize_to_f32(union)
            };
            pairs = pairs.saturating_add(1);
        }
    }
    if pairs == 0 {
        0.0
    } else {
        total / usize_to_f32(usize::try_from(pairs).unwrap_or(usize::MAX))
    }
}

fn usize_to_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

fn normalize_label(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !token.chars().all(char::is_numeric))
        .collect::<Vec<_>>()
        .join(" ")
}

fn relative_string(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(unix)]
fn permission_mode(metadata: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode()
}

#[cfg(not(unix))]
const fn permission_mode(_metadata: &std::fs::Metadata) -> u32 {
    0
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(root: &Path, relative: &str, body: &[u8]) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    fn enabled() -> DirectoryEnvelopeConfig {
        DirectoryEnvelopeConfig {
            enabled: true,
            ..DirectoryEnvelopeConfig::default()
        }
    }

    #[test]
    fn snapshot_retains_empty_directories_and_hidden_files() {
        let temp = TempDir::new().unwrap();
        write(temp.path(), "world/.hidden", b"secret");
        std::fs::create_dir_all(temp.path().join("world/empty")).unwrap();
        let snapshot = snapshot_directory(&temp.path().join("world"));
        assert!(snapshot.complete);
        assert_eq!(snapshot.regular_files, 1);
        assert_eq!(snapshot.directories, 1);
    }

    #[test]
    fn cohesive_parent_wins_over_nested_children() {
        let temp = TempDir::new().unwrap();
        write(temp.path(), "Projects/project-alpha/main.rs", b"alpha");
        write(temp.path(), "Projects/project-beta/main.rs", b"beta");
        let found = discover(temp.path(), &enabled());
        assert_eq!(found.envelopes.len(), 1);
        assert!(found.envelopes[0].root.ends_with("Projects"));
    }

    #[test]
    fn heterogeneous_container_exposes_its_files() {
        let temp = TempDir::new().unwrap();
        write(temp.path(), "Misc/tax-return.pdf", b"tax");
        write(temp.path(), "Misc/beach-photo.jpg", b"photo");
        write(temp.path(), "Misc/meeting-notes.txt", b"notes");
        let found = discover(temp.path(), &enabled());
        assert!(found.envelopes.is_empty());
        assert_eq!(found.loose_files.len(), 3);
    }

    #[test]
    fn empty_directory_remains_an_uncertain_envelope() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("empty")).unwrap();
        let found = discover(temp.path(), &enabled());
        assert_eq!(found.envelopes.len(), 1);
        assert!(found.envelopes[0].metadata.requires_review);
        assert_eq!(found.envelopes[0].members.len(), 0);
    }
}
