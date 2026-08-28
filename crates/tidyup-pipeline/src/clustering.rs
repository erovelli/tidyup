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

use std::collections::{BTreeMap, HashSet};
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
const MODEL_EXTS: &[&str] = &["stl", "obj", "3mf", "step", "stp", "fbx"];

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
    // High-precision semantic collections run before modality-specific passes:
    // a shared rare entity stem is meaningful across extensions (for example
    // `atomsnotelectrons_submission.txt` + `atomsnotelectrons_testbench.png`).
    // OCR-similar generic screenshots can then attach to that anchored set.
    let (mut bundles, semantic_left) = cluster_semantic_collections(dir, files, extractors).await;

    let mut images = Vec::new();
    let mut audio = Vec::new();
    let mut others = Vec::new();
    for f in &semantic_left {
        match modality(f) {
            Modality::Image => images.push(f.clone()),
            Modality::Audio => audio.push(f.clone()),
            Modality::Other => others.push(f.clone()),
        }
    }

    let (burst_bundles, burst_left) = cluster_photo_bursts(dir, &images, extractors, config).await;
    bundles.extend(burst_bundles);

    let (album_bundles, album_left) = cluster_music_albums(dir, &audio, extractors, config).await;
    bundles.extend(album_bundles);

    // Document series runs over NON-media loose files only. Media that failed
    // burst/album clustering — EXIF-less photos, untagged audio — must NOT be
    // swept into a DocumentSeries by filename family (e.g. IMG_0001.jpg,
    // IMG_0002.jpg): format-sequential media names are not semantic evidence
    // of a document series. Excluding them by modality is the structural guard;
    // they fall through for individual semantic classification.
    let (series_bundles, mut leftover) = cluster_document_series(dir, &others, config);
    bundles.extend(series_bundles);
    leftover.extend(burst_left);
    leftover.extend(album_left);

    (bundles, leftover)
}

// ---------------------------------------------------------------------------
// Semantic collections (cross-format filename + OCR evidence)
// ---------------------------------------------------------------------------

async fn cluster_semantic_collections(
    dir: &Path,
    files: &[PathBuf],
    extractors: &[Arc<dyn ContentExtractor>],
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    // Directory cohesion outranks repeated stems inside an assembly. A CAD
    // directory commonly contains the same part in multiple formats
    // (`Middle Piece.stl` + `Middle Piece.3mf`); grouping those pairs first
    // would fragment the assembly and could drop an unsupported format if its
    // smaller bundle later failed classification.
    if let Some(bundle) = cohesive_model_bundle(dir, files) {
        return (vec![bundle], Vec::new());
    }

    let by_entity = semantic_entity_groups(files);

    // OCR-capable extraction is comparatively expensive. Build the generic
    // screenshot evidence once, before considering any anchored entity group,
    // so G groups and S screenshots cost S extractions rather than G * S.
    let mut screenshot_evidence = Vec::new();
    for path in files.iter().filter(|path| is_generic_screenshot(path)) {
        let Some(content) = extract(path, extractors).await else {
            continue;
        };
        let Some(text) = content.text else {
            continue;
        };
        screenshot_evidence.push((path.clone(), semantic_tokens(&text)));
    }
    screenshot_evidence.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut consumed = HashSet::new();
    let mut bundles = Vec::new();
    for (key, mut members) in by_entity {
        if members.len() < 2 {
            continue;
        }
        members.sort();
        for member in &members {
            consumed.insert(member.clone());
        }

        // Expand an anchored entity group with generic screenshots whose
        // locally recognised text overlaps the group's image/text evidence.
        let mut evidence_tokens = HashSet::new();
        for member in &members {
            if let Some(content) = extract(member, extractors).await {
                if let Some(text) = content.text {
                    evidence_tokens.extend(semantic_tokens(&text));
                }
            }
        }
        for (screenshot, tokens) in &screenshot_evidence {
            if consumed.contains(screenshot) {
                continue;
            }
            if token_overlap(&evidence_tokens, tokens) >= 0.22 {
                evidence_tokens.extend(tokens.iter().cloned());
                consumed.insert(screenshot.clone());
                members.push(screenshot.clone());
            }
        }

        let label = display_collection_label(&key);
        let reasoning = format!(
            "{} related artifacts share the entity stem \"{key}\"; generic screenshots were attached only when local OCR evidence overlapped",
            members.len()
        );
        bundles.push(make_bundle(
            dir,
            BundleKind::SemanticCollection {
                label: label.clone(),
            },
            members,
            &label,
            reasoning,
        ));
    }

    // Preserve cohesive 3D assemblies found inside a named source directory.
    // This intentionally does not fire at a broad source root such as Desktop:
    // every member must share a model extension and the directory must itself
    // have a non-generic name.
    let remaining: Vec<PathBuf> = files
        .iter()
        .filter(|path| !consumed.contains(*path))
        .cloned()
        .collect();
    if is_cohesive_model_directory(dir, &remaining) {
        let label = dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("3d-model")
            .to_string();
        for member in &remaining {
            consumed.insert(member.clone());
        }
        bundles.push(make_bundle(
            dir,
            BundleKind::SemanticCollection {
                label: label.clone(),
            },
            remaining,
            &label,
            format!("cohesive 3D assembly preserved from source directory \"{label}\""),
        ));
    }

    let leftovers = files
        .iter()
        .filter(|path| !consumed.contains(*path))
        .cloned()
        .collect();
    (bundles, leftovers)
}

fn cohesive_model_bundle(dir: &Path, files: &[PathBuf]) -> Option<DetectedBundle> {
    if !is_cohesive_model_directory(dir, files) {
        return None;
    }
    let label = dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("3d-model")
        .to_string();
    Some(make_bundle(
        dir,
        BundleKind::SemanticCollection {
            label: label.clone(),
        },
        files.to_vec(),
        &label,
        format!("cohesive 3D assembly preserved from source directory \"{label}\""),
    ))
}

fn semantic_entity_groups(files: &[PathBuf]) -> BTreeMap<String, Vec<PathBuf>> {
    // Tokenization allocates lowercase strings. Do it once per file rather
    // than twice for every pair (which became prohibitive for large folders).
    let tokenized: Vec<Option<Vec<String>>> =
        files.iter().map(|path| filename_tokens(path)).collect();
    let mut candidates: BTreeMap<String, HashSet<PathBuf>> = BTreeMap::new();
    for (index, (left, left_tokens)) in files.iter().zip(&tokenized).enumerate() {
        for (right, right_tokens) in files.iter().zip(&tokenized).skip(index.saturating_add(1)) {
            let (Some(left_tokens), Some(right_tokens)) = (left_tokens, right_tokens) else {
                continue;
            };
            let Some(key) = shared_entity_key(left_tokens, right_tokens) else {
                continue;
            };
            let members = candidates.entry(key).or_default();
            members.insert(left.clone());
            members.insert(right.clone());
        }
    }

    // Prefer the largest, most specific groups and assign each file once.
    let mut ranked: Vec<(String, HashSet<PathBuf>)> = candidates.into_iter().collect();
    ranked.sort_by(|(left_key, left), (right_key, right)| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| right_key.len().cmp(&left_key.len()))
            .then_with(|| left_key.cmp(right_key))
    });
    let mut consumed = HashSet::new();
    let mut groups = BTreeMap::new();
    for (key, members) in ranked {
        let mut available: Vec<PathBuf> = members
            .into_iter()
            .filter(|path| !consumed.contains(path))
            .collect();
        if available.len() < 2 {
            continue;
        }
        available.sort();
        consumed.extend(available.iter().cloned());
        groups.insert(key, available);
    }
    groups
}

fn shared_entity_key(left_tokens: &[String], right_tokens: &[String]) -> Option<String> {
    if is_generic_screenshot_tokens(left_tokens) || is_generic_screenshot_tokens(right_tokens) {
        return None;
    }
    let shared = left_tokens
        .iter()
        .zip(right_tokens.iter())
        .take_while(|(left, right)| left == right)
        .count();
    if shared == 0 {
        return None;
    }
    let shared_tokens = left_tokens.get(..shared)?;
    // Dates, counters, and timestamps are correlation in naming convention,
    // not semantic evidence that files belong in one atomic collection.
    if shared_tokens
        .iter()
        .all(|token| token.chars().all(|ch| ch.is_ascii_digit()))
    {
        return None;
    }
    let key = shared_tokens.join("_");
    let first_token_len = left_tokens.first()?.len();
    // A long unique token or a multi-token prefix is stable across arbitrary
    // project vocabularies without enumerating roles such as "submission".
    (key.len() >= 8 && (shared > 1 || first_token_len >= 8)).then_some(key)
}

fn filename_tokens(path: &Path) -> Option<Vec<String>> {
    let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
    let tokens: Vec<String> = stem
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(ToString::to_string)
        .collect();
    (!tokens.is_empty()).then_some(tokens)
}

fn is_generic_screenshot(path: &Path) -> bool {
    filename_tokens(path).is_some_and(|tokens| is_generic_screenshot_tokens(&tokens))
}

fn is_generic_screenshot_tokens(tokens: &[String]) -> bool {
    tokens.first().is_some_and(|token| token == "screenshot")
        || matches!(tokens, [first, second, ..] if first == "screen" && second == "shot")
}

fn semantic_tokens(text: &str) -> HashSet<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "from", "this", "that", "your", "you", "are", "all", "back",
        "home", "what", "have", "has", "into", "not", "but", "was", "were",
    ];
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|token| token.len() >= 3 && !STOP.contains(&token.as_str()))
        .collect()
}

fn token_overlap(anchor: &HashSet<String>, candidate: &HashSet<String>) -> f32 {
    if anchor.is_empty() || candidate.is_empty() {
        return 0.0;
    }
    let shared = anchor.intersection(candidate).count();
    let denominator = anchor.len().min(candidate.len());
    let Ok(shared_u16) = u16::try_from(shared) else {
        return 0.0;
    };
    let Ok(denominator_u16) = u16::try_from(denominator) else {
        return 0.0;
    };
    f32::from(shared_u16) / f32::from(denominator_u16)
}

fn display_collection_label(key: &str) -> String {
    key.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars.next().map_or_else(String::new, |first| {
                format!("{}{}", first.to_ascii_uppercase(), chars.as_str())
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_cohesive_model_directory(dir: &Path, files: &[PathBuf]) -> bool {
    if files.len() < 2 {
        return false;
    }
    let generic_dir = dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_ascii_lowercase)
        .is_none_or(|name| matches!(name.as_str(), "desktop" | "downloads" | "documents"));
    if generic_dir {
        return false;
    }
    files.iter().all(|path| {
        path.extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| MODEL_EXTS.contains(&ext.to_ascii_lowercase().as_str()))
    })
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
            if item.1.saturating_sub(*last_t) > window_secs {
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
    async fn semantic_collection_links_cross_format_shared_entity() {
        let files = vec![
            p("/desktop/atomsnotelectrons_submission.txt"),
            p("/desktop/atomsnotelectrons_testbench.png"),
            p("/desktop/unrelated.pdf"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert_eq!(bundles.len(), 1);
        assert!(matches!(
            bundles[0].kind,
            BundleKind::SemanticCollection { .. }
        ));
        assert_eq!(bundles[0].members.len(), 2);
        assert_eq!(leftover, vec![p("/desktop/unrelated.pdf")]);
    }

    #[tokio::test]
    async fn semantic_collection_does_not_require_known_role_words() {
        let files = vec![
            p("/desktop/quasarforge_lantern.alpha"),
            p("/desktop/quasarforge_velvet.omega"),
            p("/desktop/unrelated.data"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].target_subdir.as_deref(), Some("Quasarforge"));
        assert_eq!(bundles[0].members.len(), 2);
        assert_eq!(leftover, vec![p("/desktop/unrelated.data")]);
    }

    #[tokio::test]
    async fn generic_screenshot_variants_do_not_form_entity_collections() {
        for files in [
            vec![
                p("/desktop/screenshot_1.png"),
                p("/desktop/screenshot_2.png"),
            ],
            vec![
                p("/desktop/Screen_Shot_1.png"),
                p("/desktop/Screen-Shot-2.png"),
            ],
        ] {
            let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
            assert!(bundles.is_empty());
            assert_eq!(leftover.len(), 2);
        }
    }

    #[tokio::test]
    async fn numeric_and_date_prefixes_do_not_form_entity_collections() {
        for files in [
            vec![
                p("/desktop/20240101_report.pdf"),
                p("/desktop/20240101_notes.pdf"),
            ],
            vec![
                p("/desktop/2026-08-25_notes.txt"),
                p("/desktop/2026-08-25_photo.png"),
            ],
        ] {
            let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
            assert!(bundles.is_empty());
            assert_eq!(leftover.len(), 2);
        }
    }

    #[tokio::test]
    async fn cohesive_model_directory_is_preserved() {
        let files = vec![
            p("/desktop/zprint/Back Piece.stl"),
            p("/desktop/zprint/Front Piece.stl"),
            p("/desktop/zprint/Middle Piece.stl"),
            p("/desktop/zprint/Middle Piece.3mf"),
            p("/desktop/zprint/Side Piece.stl"),
            p("/desktop/zprint/Side Piece.3mf"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert_eq!(bundles.len(), 1);
        assert!(leftover.is_empty());
        assert_eq!(bundles[0].target_subdir.as_deref(), Some("zprint"));
        assert_eq!(bundles[0].members.len(), 6);
    }

    #[tokio::test]
    async fn exif_less_photos_do_not_become_a_document_series() {
        // IMG_0001.jpg / _0002 / _0003 with no EXIF (no extractors → no capture
        // time) must NOT be swept into a DocumentSeries by filename family —
        // their counter pattern alone is not semantic evidence of a document
        // series. The modality guard excludes them from that structural pass.
        let files = vec![
            p("/dcim/IMG_0001.jpg"),
            p("/dcim/IMG_0002.jpg"),
            p("/dcim/IMG_0003.jpg"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert!(
            bundles.is_empty(),
            "EXIF-less photos must not form any bundle, got {:?}",
            bundles.iter().map(|b| &b.kind).collect::<Vec<_>>(),
        );
        assert_eq!(leftover.len(), 3, "all three fall through as loose files");
    }

    #[tokio::test]
    async fn untagged_audio_does_not_become_a_document_series() {
        // Same guard for audio: untagged tracks with a filename family must not
        // cluster as a DocumentSeries.
        let files = vec![
            p("/music/track-01.mp3"),
            p("/music/track-02.mp3"),
            p("/music/track-03.mp3"),
        ];
        let (bundles, leftover) = cluster_loose(&files, &[], &ClusterConfig::default()).await;
        assert!(bundles.is_empty(), "untagged audio must not form a bundle");
        assert_eq!(leftover.len(), 3);
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
