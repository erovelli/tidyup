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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::NaiveDateTime;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::inference::ImageEmbeddingBackend;
use tidyup_domain::bundle::BundleKind;

use crate::scanner::DetectedBundle;
use crate::semantic::{cosine, rank_concepts, GroundedConcept, SemanticRunCache};

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
    /// Minimum visually similar loose siblings in a semantic collection.
    pub min_semantic_cluster: usize,
    /// Complete-link cosine floor for every pair in a visual collection.
    pub semantic_similarity_threshold: f32,
    /// Minimum raw concept score used to ground a visual collection label.
    pub semantic_label_confidence: f32,
    /// Deterministic per-directory cap for the quadratic complete-link pass.
    /// Files beyond the cap stay loose and remain individually classifiable.
    pub max_semantic_candidates: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            min_burst: 3,
            burst_window_secs: 60,
            min_album: 3,
            min_series: 3,
            min_semantic_cluster: 3,
            semantic_similarity_threshold: 0.88,
            semantic_label_confidence: 0.30,
            max_semantic_candidates: 512,
        }
    }
}

/// Optional modality semantics for generalized content clustering.
#[derive(Clone, Copy)]
#[allow(missing_debug_implementations)]
pub struct SemanticClusterContext<'a> {
    pub image: Option<&'a dyn ImageEmbeddingBackend>,
    pub image_concepts: &'a [GroundedConcept],
    pub cache: &'a SemanticRunCache<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Modality {
    Image,
    Audio,
    Other,
}

// This list protects opaque 3D assemblies as bundles; it never chooses a
// destination or semantic label. Capability dispatch below uses MIME/content.
const MODEL_EXTS: &[&str] = &["stl", "obj", "3mf", "step", "stp", "fbx"];

fn modality_from_mime(mime: Option<&str>) -> Modality {
    if mime.is_some_and(|value| value.starts_with("image/")) {
        Modality::Image
    } else if mime.is_some_and(|value| value.starts_with("audio/")) {
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
    let cache = SemanticRunCache::default();
    cluster_loose_semantic(
        loose,
        extractors,
        config,
        SemanticClusterContext {
            image: None,
            image_concepts: &[],
            cache: &cache,
        },
    )
    .await
}

/// Cluster loose files with optional semantic embeddings. The caller-owned
/// cache is reused later by placement and naming.
pub async fn cluster_loose_semantic(
    loose: &[PathBuf],
    extractors: &[Arc<dyn ContentExtractor>],
    config: &ClusterConfig,
    semantic: SemanticClusterContext<'_>,
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
        let (dir_bundles, dir_leftover) =
            cluster_dir(&dir, &files, extractors, config, semantic).await;
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
    semantic: SemanticClusterContext<'_>,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    // High-precision semantic collections run before modality-specific passes:
    // a shared rare entity stem is meaningful across extensions (for example
    // `atomsnotelectrons_submission.txt` + `atomsnotelectrons_testbench.png`).
    // OCR-similar generic screenshots can then attach to that anchored set.
    let (mut bundles, semantic_left) =
        cluster_semantic_collections(dir, files, extractors, semantic.cache).await;
    let mut modalities = HashMap::with_capacity(semantic_left.len());
    for path in &semantic_left {
        let mime = tidyup_extract::mime::detect(path).await;
        modalities.insert(path.clone(), modality_from_mime(mime.as_deref()));
    }
    let mut images = Vec::new();
    let mut audio = Vec::new();
    let mut others = Vec::new();
    for f in &semantic_left {
        match modalities.get(f).copied().unwrap_or(Modality::Other) {
            Modality::Image => images.push(f.clone()),
            Modality::Audio => audio.push(f.clone()),
            Modality::Other => others.push(f.clone()),
        }
    }

    // Explicit capture-time metadata outranks visual similarity. A real burst
    // remains a PhotoBurst (and therefore preserves member names) even when a
    // loaded image encoder would also find its frames visually cohesive.
    let (burst_bundles, burst_left) = cluster_photo_bursts(dir, &images, extractors, config).await;
    bundles.extend(burst_bundles);

    let (visual_bundles, visual_left) =
        cluster_visual_neighbors(dir, &burst_left, config, semantic, &modalities).await;
    bundles.extend(visual_bundles);

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
    leftover.extend(visual_left);
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
    cache: &SemanticRunCache<'_>,
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
        let mime = tidyup_extract::mime::detect(path).await;
        let Some(content) = cache.extract(path, mime.as_deref(), extractors).await else {
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
            let mime = tidyup_extract::mime::detect(member).await;
            if let Some(content) = cache.extract(member, mime.as_deref(), extractors).await {
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

// ---------------------------------------------------------------------------
// General visual-neighbor collections
// ---------------------------------------------------------------------------

async fn cluster_visual_neighbors(
    dir: &Path,
    files: &[PathBuf],
    config: &ClusterConfig,
    semantic: SemanticClusterContext<'_>,
    modalities: &HashMap<PathBuf, Modality>,
) -> (Vec<DetectedBundle>, Vec<PathBuf>) {
    let Some(backend) = semantic.image else {
        return (Vec::new(), files.to_vec());
    };
    if semantic.image_concepts.is_empty() {
        return (Vec::new(), files.to_vec());
    }

    let (image_paths, candidate_count) =
        bounded_visual_candidates(files, modalities, config.max_semantic_candidates);
    if candidate_count > image_paths.len() {
        tracing::debug!(
            directory = %dir.display(),
            candidates = candidate_count,
            cap = config.max_semantic_candidates,
            "visual collection discovery capped; remaining files stay loose"
        );
    }

    let mut embedded = Vec::new();
    for path in &image_paths {
        let mime = tidyup_extract::mime::detect(path)
            .await
            .unwrap_or_else(|| "application/octet-stream".to_string());
        match semantic.cache.image_embedding(path, &mime, backend).await {
            Ok(Some(embedding)) => embedded.push((path.clone(), embedding)),
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "semantic image clustering skipped file");
            }
        }
    }
    embedded.sort_by(|(left, _), (right, _)| left.cmp(right));

    // Deterministic complete-link grouping. A candidate joins only when it is
    // close to every member, avoiding single-link chains through ambiguous
    // bridge images.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for candidate in 0..embedded.len() {
        let destination = groups.iter().position(|group| {
            let Some(candidate_embedding) = embedded.get(candidate) else {
                return false;
            };
            group.iter().all(|member| {
                embedded.get(*member).is_some_and(|member_embedding| {
                    cosine(
                        candidate_embedding.1.as_slice(),
                        member_embedding.1.as_slice(),
                    ) >= config.semantic_similarity_threshold
                })
            })
        });
        if let Some(index) = destination {
            if let Some(group) = groups.get_mut(index) {
                group.push(candidate);
            }
        } else {
            groups.push(vec![candidate]);
        }
    }

    let mut consumed = HashSet::new();
    let mut bundles = Vec::new();
    for group in groups
        .into_iter()
        .filter(|group| group.len() >= config.min_semantic_cluster)
    {
        let centroid = mean_embedding(
            group
                .iter()
                .filter_map(|index| embedded.get(*index).map(|item| item.1.as_slice())),
        );
        let ranked = rank_concepts(&centroid, semantic.image_concepts);
        let Some((label, evidence)) = grounded_collection_label(&ranked, config) else {
            continue;
        };
        let members: Vec<PathBuf> = group
            .iter()
            .filter_map(|index| embedded.get(*index).map(|item| item.0.clone()))
            .collect();
        let min_similarity = minimum_pair_similarity(&group, &embedded);
        consumed.extend(members.iter().cloned());
        bundles.push(make_bundle(
            dir,
            BundleKind::SemanticCollection {
                label: label.clone(),
            },
            members,
            &label,
            format!(
                "visually coherent collection; complete-link minimum cosine={min_similarity:.3}; grounded concepts={evidence}"
            ),
        ));
    }

    let leftovers = files
        .iter()
        .filter(|path| !consumed.contains(*path))
        .cloned()
        .collect();
    (bundles, leftovers)
}

fn bounded_visual_candidates(
    files: &[PathBuf],
    modalities: &HashMap<PathBuf, Modality>,
    cap: usize,
) -> (Vec<PathBuf>, usize) {
    let mut candidates = files
        .iter()
        .filter(|path| {
            modalities
                .get(*path)
                .is_some_and(|modality| *modality == Modality::Image)
        })
        .cloned()
        .collect::<Vec<_>>();
    candidates.sort();
    let count = candidates.len();
    candidates.truncate(cap);
    (candidates, count)
}

fn mean_embedding<'a>(vectors: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let vectors: Vec<&[f32]> = vectors.collect();
    let Some(first) = vectors.first() else {
        return Vec::new();
    };
    let mut mean = vec![0.0_f32; first.len()];
    for vector in &vectors {
        if vector.len() != mean.len() {
            return Vec::new();
        }
        for (sum, value) in mean.iter_mut().zip(*vector) {
            *sum += *value;
        }
    }
    let denominator = u16::try_from(vectors.len()).map_or(1.0, f32::from);
    for value in &mut mean {
        *value /= denominator;
    }
    let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut mean {
            *value /= norm;
        }
    }
    mean
}

fn grounded_collection_label(
    ranked: &[crate::semantic::ConceptMatch],
    config: &ClusterConfig,
) -> Option<(String, String)> {
    let first = ranked.first()?;
    if first.score < config.semantic_label_confidence {
        return None;
    }
    let mut selected = vec![first];
    if let Some(second) = ranked.get(1) {
        let third_score = ranked.get(2).map_or(0.0, |concept| concept.score);
        if second.score >= config.semantic_label_confidence && second.score - third_score >= 0.02 {
            selected.push(second);
        }
    }
    let label = selected
        .iter()
        .map(|concept| display_collection_label(&concept.label))
        .collect::<Vec<_>>()
        .join(" ");
    let evidence = selected
        .iter()
        .map(|concept| format!("{}:{:.3}", concept.label, concept.score))
        .collect::<Vec<_>>()
        .join(",");
    Some((label, evidence))
}

fn minimum_pair_similarity(group: &[usize], embedded: &[(PathBuf, Arc<Vec<f32>>)]) -> f32 {
    let mut minimum = 1.0_f32;
    for (offset, left) in group.iter().enumerate() {
        for right in group.iter().skip(offset.saturating_add(1)) {
            if let (Some(left_item), Some(right_item)) = (embedded.get(*left), embedded.get(*right))
            {
                minimum = minimum.min(cosine(left_item.1.as_slice(), right_item.1.as_slice()));
            }
        }
    }
    minimum
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

/// Parse an EXIF `DateTimeOriginal` string to a unix timestamp. The raw EXIF
/// representation uses `YYYY:MM:DD`, while `kamadak-exif` displays the value as
/// `YYYY-MM-DD`; accept both. EXIF has no timezone, so it's interpreted as UTC
/// — sufficient for relative burst windowing.
fn parse_exif_datetime(s: &str) -> Option<i64> {
    let value = s.trim();
    let dt = NaiveDateTime::parse_from_str(value, "%Y:%m:%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S"))
        .ok()?;
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
    use async_trait::async_trait;
    use tempfile::TempDir;
    use tidyup_extract::image::ImageExtractor;

    struct VisualBackend;

    #[async_trait]
    impl ImageEmbeddingBackend for VisualBackend {
        async fn embed_image(
            &self,
            image_bytes: &[u8],
            _mime: &str,
        ) -> tidyup_core::Result<Vec<f32>> {
            if image_bytes.first() == Some(&b'c') {
                Ok(vec![1.0, 0.0])
            } else {
                Ok(vec![0.0, 1.0])
            }
        }

        async fn embed_text(&self, text: &str) -> tidyup_core::Result<Vec<f32>> {
            if text.contains("cat") {
                Ok(vec![1.0, 0.0])
            } else {
                Ok(vec![0.0, 1.0])
            }
        }

        async fn embed_texts(&self, texts: &[&str]) -> tidyup_core::Result<Vec<Vec<f32>>> {
            let mut embeddings = Vec::with_capacity(texts.len());
            for text in texts {
                embeddings.push(self.embed_text(text).await?);
            }
            Ok(embeddings)
        }

        fn dimensions(&self) -> usize {
            2
        }

        fn model_id(&self) -> &'static str {
            "visual-test"
        }
    }

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    /// Add a little-endian EXIF `DateTimeOriginal` APP1 segment to the real,
    /// decodable JPEG fixture. This exercises the production image extractor
    /// rather than replacing the extractor boundary with a test double.
    fn jpeg_with_capture_time() -> Vec<u8> {
        let fixture = include_bytes!("../../tidyup-extract/tests/fixtures/sample.jpg");
        assert_eq!(&fixture[..2], &[0xff, 0xd8]);

        let mut app1 = vec![
            0xff, 0xe1, 0x00, 0x48, b'E', b'x', b'i', b'f', 0x00, 0x00, b'I', b'I', 0x2a, 0x00,
            0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x69, 0x87, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x03, 0x90, 0x02, 0x00,
            0x14, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        app1.extend_from_slice(b"2024:01:15 10:30:45\0");

        let capacity = fixture.len().saturating_add(app1.len());
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(&fixture[..2]);
        output.extend_from_slice(&app1);
        output.extend_from_slice(&fixture[2..]);
        output
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
        assert_eq!(
            parse_exif_datetime("2024-01-15 10:30:45"),
            Some(1_705_314_645)
        );
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
    async fn visual_neighbors_cluster_without_filename_evidence() {
        let temp = TempDir::new().unwrap();
        let files = ["IMG_0001.png", "DSC_9137.png", "random_44.png", "other.png"]
            .map(|name| temp.path().join(name));
        for path in &files[..3] {
            std::fs::write(path, b"cat beach fixture").unwrap();
        }
        std::fs::write(&files[3], b"x unrelated fixture").unwrap();

        let cache = SemanticRunCache::default();
        let concepts = vec![
            GroundedConcept {
                label: "cat".to_string(),
                prompt: "a photograph of a cat".to_string(),
                family: "animal".to_string(),
                embedding: vec![1.0, 0.0],
            },
            GroundedConcept {
                label: "beach".to_string(),
                prompt: "a photograph of a beach".to_string(),
                family: "scene".to_string(),
                embedding: vec![0.8, 0.2],
            },
        ];
        let (bundles, leftover) = cluster_loose_semantic(
            &files,
            &[],
            &ClusterConfig::default(),
            SemanticClusterContext {
                image: Some(&VisualBackend),
                image_concepts: &concepts,
                cache: &cache,
            },
        )
        .await;

        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].members.len(), 3);
        assert!(matches!(
            bundles[0].kind,
            BundleKind::SemanticCollection { .. }
        ));
        assert!(bundles[0].reasoning.contains("visually coherent"));
        assert_eq!(leftover, vec![files[3].clone()]);
    }

    #[tokio::test]
    async fn exif_photo_burst_preempts_visual_collection() {
        let temp = TempDir::new().unwrap();
        let files = ["alpha.jpg", "bravo.jpg", "charlie.jpg"].map(|name| temp.path().join(name));
        let jpeg = jpeg_with_capture_time();
        for path in &files {
            std::fs::write(path, &jpeg).unwrap();
        }

        let cache = SemanticRunCache::default();
        let concepts = vec![GroundedConcept {
            label: "photograph".to_string(),
            prompt: "a photograph".to_string(),
            family: "media".to_string(),
            embedding: vec![0.0, 1.0],
        }];
        let extractors: Vec<Arc<dyn ContentExtractor>> =
            vec![Arc::new(ImageExtractor::with_ocr(false, 0))];
        let extracted = extractors[0].extract(&files[0]).await.unwrap();
        assert!(
            extracted.metadata["exif"]["date"].is_string(),
            "fixture must expose DateTimeOriginal: {}",
            extracted.metadata
        );
        assert_eq!(
            extracted.metadata["exif"]["date"].as_str(),
            Some("2024-01-15 10:30:45")
        );
        let (bundles, leftover) = cluster_loose_semantic(
            &files,
            &extractors,
            &ClusterConfig::default(),
            SemanticClusterContext {
                image: Some(&VisualBackend),
                image_concepts: &concepts,
                cache: &cache,
            },
        )
        .await;

        assert_eq!(bundles.len(), 1);
        assert!(matches!(bundles[0].kind, BundleKind::PhotoBurst));
        assert_eq!(bundles[0].members.len(), 3);
        assert!(leftover.is_empty());
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
