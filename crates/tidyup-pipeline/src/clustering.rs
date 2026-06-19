//! Content-based clustering of loose sibling files into atomic bundles.
//!
//! A pipeline pass that runs **after** the structural-marker [`scanner`] over
//! its loose files. Within each directory it groups siblings into:
//!
//! - [`BundleKind::PhotoBurst`] — images whose EXIF capture times fall within a
//!   short window (a rapid sequence shot).
//! - [`BundleKind::MusicAlbum`] — audio files sharing an ID3 album tag.
//! - [`BundleKind::DocumentSeries`] — files whose names form a numeric family
//!   (`invoice-2024-01`, `invoice-2024-02`, …).
//!
//! Unlike marker bundles these have **no shared directory** to relocate, so the
//! executor moves their members individually (all-or-nothing) — see
//! [`BundleKind::moves_as_file_set`]. Each detected cluster carries a
//! `target_subdir` (burst date / album title / family stem) the members are
//! grouped under at the destination.
//!
//! EXIF/ID3 reads go through the supplied [`ContentExtractor`]s; when the image
//! or audio extractor isn't wired (or a file has no usable metadata), that file
//! simply isn't clustered — it falls through to per-file classification. The
//! pure grouping helpers ([`family_key`], [`group_by_key`],
//! [`group_by_time_window`]) take no extractors and carry the unit tests.
//!
//! [`scanner`]: crate::scanner

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDateTime;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_domain::bundle::BundleKind;

use crate::scanner::DetectedBundle;

/// Tunables for content clustering. Thresholds are deliberately conservative —
/// a misfired cluster is more annoying than a few un-grouped files, since the
/// alternative (per-file moves) is always available.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Minimum images in a time window to count as a photo burst.
    pub min_burst: usize,
    /// Max gap (seconds) between consecutive shots in a burst.
    pub burst_window_secs: i64,
    /// Minimum tracks sharing an album tag to count as a music album.
    pub min_album: usize,
    /// Minimum files in a filename family to count as a document series.
    pub min_series: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            min_burst: 3,
            burst_window_secs: 60,
            min_album: 3,
            min_series: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Modality {
    Image,
    Audio,
    Other,
}

const IMAGE_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "heic", "heif", "raw", "cr2", "cr3",
    "nef", "arw", "dng", "orf", "rw2", "raf",
];
const AUDIO_EXTS: &[&str] = &[
    "mp3", "flac", "m4a", "wav", "ogg", "opus", "aiff", "aif", "ape", "wma", "alac", "aac",
];

fn modality(path: &Path) -> Modality {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if IMAGE_EXTS.contains(&ext.as_str()) {
        Modality::Image
    } else if AUDIO_EXTS.contains(&ext.as_str()) {
        Modality::Audio
    } else {
        Modality::Other
    }
}

/// Cluster loose files into content bundles, returning the detected bundles plus
/// the leftover loose files (those absorbed into no cluster).
///
/// Clustering is per-directory: only siblings can form a bundle.
pub async fn cluster_loose(
    loose: &[PathBuf],
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClusterConfig,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    // Group inputs by parent directory (BTreeMap for deterministic order).
    let mut by_dir: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for path in loose {
        let parent = path.parent().map_or_else(PathBuf::new, Path::to_path_buf);
        by_dir.entry(parent).or_default().push(path.clone());
    }

    let mut bundles = Vec::new();
    let mut leftovers = Vec::new();
    for (dir, files) in by_dir {
        let (dir_bundles, dir_leftover) = cluster_dir(&dir, &files, extractors, config).await;
        bundles.extend(dir_bundles);
        leftovers.extend(dir_leftover);
    }
    (bundles, leftovers)
}

async fn cluster_dir(
    dir: &Path,
    files: &[PathBuf],
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClusterConfig,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    let mut images = Vec::new();
    let mut audio = Vec::new();
    let mut others = Vec::new();
    for f in files {
        match modality(f) {
            Modality::Image => images.push(f.clone()),
            Modality::Audio => audio.push(f.clone()),
            Modality::Other => others.push(f.clone()),
        }
    }

    let mut bundles = Vec::new();

    let (burst_bundles, burst_left) = cluster_photo_bursts(dir, &images, extractors, config).await;
    bundles.extend(burst_bundles);

    let (album_bundles, album_left) = cluster_music_albums(dir, &audio, extractors, config).await;
    bundles.extend(album_bundles);

    // Document series runs over everything not already clustered.
    let mut series_input = burst_left;
    series_input.extend(album_left);
    series_input.extend(others);
    let (series_bundles, leftover) = cluster_document_series(dir, &series_input, config);
    bundles.extend(series_bundles);

    (bundles, leftover)
}

// ---------------------------------------------------------------------------
// Photo bursts (EXIF capture-time clustering)
// ---------------------------------------------------------------------------

async fn cluster_photo_bursts(
    dir: &Path,
    images: &[PathBuf],
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClusterConfig,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    if images.len() < config.min_burst {
        return (Vec::new(), images.to_vec());
    }

    let mut timed: Vec<(PathBuf, i64)> = Vec::new();
    let mut leftover: Vec<PathBuf> = Vec::new();
    for img in images {
        match exif_timestamp(img, extractors).await {
            Some(ts) => timed.push((img.clone(), ts)),
            None => leftover.push(img.clone()),
        }
    }

    let (groups, ungrouped) =
        group_by_time_window(timed, config.burst_window_secs, config.min_burst);
    leftover.extend(ungrouped);

    let mut bundles = Vec::new();
    for group in groups {
        let earliest = group.iter().map(|(_, t)| *t).min().unwrap_or(0);
        let label = format_burst_label(earliest);
        let members: Vec<PathBuf> = group.into_iter().map(|(p, _)| p).collect();
        let reasoning = format!(
            "{} photos captured within {}s (EXIF burst)",
            members.len(),
            config.burst_window_secs,
        );
        bundles.push(make_bundle(
            dir,
            BundleKind::PhotoBurst,
            members,
            &label,
            reasoning,
        ));
    }
    (bundles, leftover)
}

async fn exif_timestamp(path: &Path, extractors: &[Arc<dyn ContentExtractor>]) -> Option<i64> {
    let content = extract(path, extractors).await?;
    let date = content.metadata.get("exif")?.get("date")?.as_str()?;
    parse_exif_datetime(date)
}

/// Parse an EXIF `DateTimeOriginal` string (`"YYYY:MM:DD HH:MM:SS"`) to a unix
/// timestamp. EXIF has no timezone, so it's interpreted as UTC — fine for
/// *relative* burst windowing.
fn parse_exif_datetime(s: &str) -> Option<i64> {
    let dt = NaiveDateTime::parse_from_str(s.trim(), "%Y:%m:%d %H:%M:%S").ok()?;
    Some(dt.and_utc().timestamp())
}

fn format_burst_label(unix_secs: i64) -> String {
    chrono::DateTime::from_timestamp(unix_secs, 0).map_or_else(
        || "Burst".to_string(),
        |dt| format!("Burst {}", dt.format("%Y-%m-%d")),
    )
}

/// Greedily group time-stamped items where consecutive gaps are within
/// `window_secs`. Returns the groups of size `>= min` plus the leftover paths.
fn group_by_time_window(
    mut items: Vec<(PathBuf, i64)>,
    window_secs: i64,
    min: usize,
) -> (Vec<Vec<(PathBuf, i64)>>, Vec<PathBuf>) {
    items.sort_by_key(|(_, t)| *t);

    let mut groups: Vec<Vec<(PathBuf, i64)>> = Vec::new();
    let mut leftover: Vec<PathBuf> = Vec::new();
    let mut current: Vec<(PathBuf, i64)> = Vec::new();

    for item in items {
        if let Some((_, last_t)) = current.last() {
            if item.1 - *last_t > window_secs {
                flush_window(&mut current, &mut groups, &mut leftover, min);
            }
        }
        current.push(item);
    }
    flush_window(&mut current, &mut groups, &mut leftover, min);
    (groups, leftover)
}

fn flush_window(
    current: &mut Vec<(PathBuf, i64)>,
    groups: &mut Vec<Vec<(PathBuf, i64)>>,
    leftover: &mut Vec<PathBuf>,
    min: usize,
) {
    let group = std::mem::take(current);
    if group.len() >= min {
        groups.push(group);
    } else {
        leftover.extend(group.into_iter().map(|(p, _)| p));
    }
}

// ---------------------------------------------------------------------------
// Music albums (ID3 album-tag clustering)
// ---------------------------------------------------------------------------

async fn cluster_music_albums(
    dir: &Path,
    audio: &[PathBuf],
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClusterConfig,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    if audio.len() < config.min_album {
        return (Vec::new(), audio.to_vec());
    }

    let mut keyed: Vec<(PathBuf, String)> = Vec::new();
    let mut leftover: Vec<PathBuf> = Vec::new();
    for track in audio {
        match album_tag(track, extractors).await {
            Some(album) => keyed.push((track.clone(), album)),
            None => leftover.push(track.clone()),
        }
    }

    let (groups, ungrouped) = group_by_key(keyed, config.min_album);
    leftover.extend(ungrouped);

    let mut bundles = Vec::new();
    for (album, members) in groups {
        let reasoning = format!("{} tracks sharing album \"{album}\"", members.len());
        bundles.push(make_bundle(
            dir,
            BundleKind::MusicAlbum,
            members,
            &album,
            reasoning,
        ));
    }
    (bundles, leftover)
}

async fn album_tag(path: &Path, extractors: &[Arc<dyn ContentExtractor>]) -> Option<String> {
    let content = extract(path, extractors).await?;
    let album = content.metadata.get("tags")?.get("album")?.as_str()?.trim();
    if album.is_empty() {
        None
    } else {
        Some(album.to_string())
    }
}

// ---------------------------------------------------------------------------
// Document series (filename-family clustering — no content reads)
// ---------------------------------------------------------------------------

fn cluster_document_series(
    dir: &Path,
    files: &[PathBuf],
    config: &ClusterConfig,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    let mut keyed: Vec<(PathBuf, String)> = Vec::new();
    let mut leftover: Vec<PathBuf> = Vec::new();
    for f in files {
        match family_key(f) {
            Some(key) => keyed.push((f.clone(), key)),
            None => leftover.push(f.clone()),
        }
    }

    let (groups, ungrouped) = group_by_key(keyed, config.min_series);
    leftover.extend(ungrouped);

    let mut bundles = Vec::new();
    for (family, members) in groups {
        let reasoning = format!(
            "{} files in the \"{family}\" filename series",
            members.len()
        );
        bundles.push(make_bundle(
            dir,
            BundleKind::DocumentSeries {
                pattern: family.clone(),
            },
            members,
            &family,
            reasoning,
        ));
    }
    (bundles, leftover)
}

/// Derive a family key from a filename by stripping a trailing numeric suffix.
///
/// `invoice-2024-01.pdf` -> `Some("invoice-2024")`; `report.pdf` (no numeric
/// suffix) -> `None`; `001.pdf` (all digits) -> `None`. Returning `None` keeps
/// non-sequenced files out of series clustering.
fn family_key(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let no_digits = stem.trim_end_matches(|c: char| c.is_ascii_digit());
    let trimmed = no_digits.trim_end_matches(['-', '_', ' ', '.']);
    // Require that a numeric suffix was actually stripped and something remains.
    if trimmed.is_empty() || trimmed.len() == stem.len() {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

/// Group `(path, key)` pairs by key, keeping only groups of size `>= min`.
/// Returns the kept groups (sorted by key) and the leftover paths.
fn group_by_key(
    items: Vec<(PathBuf, String)>,
    min: usize,
) -> (Vec<(String, Vec<PathBuf>)>, Vec<PathBuf>) {
    let mut map: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for (path, key) in items {
        map.entry(key).or_default().push(path);
    }

    let mut groups = Vec::new();
    let mut leftover = Vec::new();
    for (key, mut paths) in map {
        if paths.len() >= min {
            paths.sort();
            groups.push((key, paths));
        } else {
            leftover.extend(paths);
        }
    }
    (groups, leftover)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Run the first extractor that claims `path` and return its content. `None`
/// when no extractor supports the file or extraction fails — either way the
/// file just doesn't cluster.
async fn extract(
    path: &Path,
    extractors: &[Arc<dyn ContentExtractor>],
) -> Option<ExtractedContent> {
    let extractor = extractors.iter().find(|e| e.supports(path, None))?;
    extractor.extract(path).await.ok()
}

fn make_bundle(
    dir: &Path,
    kind: BundleKind,
    mut members: Vec<PathBuf>,
    subdir: &str,
    reasoning: String,
) -> DetectedBundle {
    members.sort();
    DetectedBundle {
        root: dir.to_path_buf(),
        kind,
        members,
        reasoning,
        target_subdir: Some(sanitize_subdir(subdir)),
    }
}

/// Make a cluster label safe to use as a single directory name.
fn sanitize_subdir(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = mapped.trim();
    if trimmed.is_empty() {
        "cluster".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn family_key_strips_numeric_suffix() {
        assert_eq!(
            family_key(&p("/d/invoice-2024-01.pdf")),
            Some("invoice-2024".to_string())
        );
        assert_eq!(family_key(&p("/d/IMG_0001.jpg")), Some("img".to_string()));
    }

    #[test]
    fn family_key_rejects_non_sequenced_names() {
        assert_eq!(family_key(&p("/d/report.pdf")), None); // no numeric suffix
        assert_eq!(family_key(&p("/d/001.pdf")), None); // all digits
    }

    #[test]
    fn group_by_key_keeps_only_large_enough_groups() {
        let items = vec![
            (p("a1"), "fam".to_string()),
            (p("a2"), "fam".to_string()),
            (p("a3"), "fam".to_string()),
            (p("b1"), "lone".to_string()),
        ];
        let (groups, leftover) = group_by_key(items, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "fam");
        assert_eq!(groups[0].1.len(), 3);
        assert_eq!(leftover, vec![p("b1")]);
    }

    #[test]
    fn parse_exif_datetime_roundtrips() {
        let ts = parse_exif_datetime("2024:01:15 10:30:45").unwrap();
        // 2024-01-15T10:30:45Z
        assert_eq!(ts, 1_705_314_645);
        assert!(parse_exif_datetime("not a date").is_none());
    }

    #[test]
    fn group_by_time_window_splits_on_gaps() {
        // Three shots within 5s, then a 4th 200s later (alone).
        let items = vec![
            (p("a"), 1000),
            (p("b"), 1002),
            (p("c"), 1004),
            (p("d"), 1300),
        ];
        let (groups, leftover) = group_by_time_window(items, 60, 3);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
        assert_eq!(leftover, vec![p("d")]);
    }

    #[test]
    fn sanitize_subdir_replaces_unsafe_chars() {
        assert_eq!(sanitize_subdir("Dark Side / Moon"), "Dark Side _ Moon");
        assert_eq!(sanitize_subdir("   "), "cluster");
    }

    #[tokio::test]
    async fn cluster_loose_detects_document_series_without_extractors() {
        // No extractors wired → bursts/albums can't form; filename families do.
        let files = vec![
            p("/inbox/invoice-01.pdf"),
            p("/inbox/invoice-02.pdf"),
            p("/inbox/invoice-03.pdf"),
            p("/inbox/taxes.pdf"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert_eq!(bundles.len(), 1);
        let b = &bundles[0];
        assert!(matches!(b.kind, BundleKind::DocumentSeries { .. }));
        assert!(b.kind.moves_as_file_set());
        assert_eq!(b.members.len(), 3);
        assert_eq!(b.target_subdir.as_deref(), Some("invoice"));
        assert_eq!(leftover, vec![p("/inbox/taxes.pdf")]);
    }

    #[tokio::test]
    async fn cluster_loose_leaves_small_groups_loose() {
        let files = vec![p("/d/invoice-01.pdf"), p("/d/invoice-02.pdf")];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert!(bundles.is_empty());
        assert_eq!(leftover.len(), 2);
    }

    #[tokio::test]
    async fn cluster_loose_keeps_clusters_per_directory() {
        // Same family name in two directories must not merge across them.
        let files = vec![
            p("/a/page-1.txt"),
            p("/a/page-2.txt"),
            p("/a/page-3.txt"),
            p("/b/page-1.txt"),
            p("/b/page-2.txt"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        // /a has 3 (clusters), /b has 2 (stays loose).
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].members.len(), 3);
        assert_eq!(leftover.len(), 2);
    }
}
