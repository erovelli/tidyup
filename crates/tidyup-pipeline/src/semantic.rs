//! Shared semantic artifacts for one scan or migration run.
//!
//! A file is decoded and embedded at most once per model during a run. The
//! resulting vectors are reused by clustering, placement, and grounded naming.
//! Model-specific latent spaces remain in separate maps so callers cannot
//! accidentally compare BGE, `SigLIP`, and CLAP vectors.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use chrono::Utc;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::inference::{AudioEmbeddingBackend, ImageEmbeddingBackend};
use tidyup_core::storage::FileIndex;
use tidyup_domain::{ClassifierConfig, ContentHash, SemanticArtifact};

use crate::naming::{propose_grounded_rename, sanitize_filename, RenameProposal};

type EmbeddingCache = Mutex<HashMap<(PathBuf, String), Arc<Vec<f32>>>>;

/// A text concept pre-embedded in a modality's contrastive latent space.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedConcept {
    /// Stable filesystem-safe label used when composing a filename.
    pub label: String,
    /// Natural-language prompt used by the model's text tower.
    pub prompt: String,
    /// Data-defined compatibility family used to avoid composing redundant
    /// broad/narrow labels such as `cat_on_animal`.
    pub family: String,
    /// L2-normalized text embedding in the matching modality space.
    pub embedding: Vec<f32>,
}

/// A concept retrieved from file content, including its raw model score.
#[derive(Debug, Clone, PartialEq)]
pub struct ConceptMatch {
    pub label: String,
    pub score: f32,
    pub family: String,
}

/// Result of the shared grounded rename gate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroundedRename {
    pub proposal: RenameProposal,
    pub mismatch_score: Option<f32>,
    pub concepts: Vec<ConceptMatch>,
}

/// Per-run cache shared by extraction, clustering, classification, and naming.
///
/// This is the in-memory first layer of the semantic-artifact cache. Keys
/// include the backend model id so changing a model cannot reuse stale vectors.
/// When a `FileIndex` is supplied, an exact content-hash/model/preprocessing/
/// latent-space cache sits behind this run-local layer.
#[allow(missing_debug_implementations)]
pub struct SemanticRunCache<'a> {
    persistent: Option<&'a dyn FileIndex>,
    extracted: Mutex<HashMap<PathBuf, Option<ExtractedContent>>>,
    image_embeddings: EmbeddingCache,
    audio_embeddings: EmbeddingCache,
}

impl Default for SemanticRunCache<'_> {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<'a> SemanticRunCache<'a> {
    #[must_use]
    pub fn new(persistent: Option<&'a dyn FileIndex>) -> Self {
        Self {
            persistent,
            extracted: Mutex::new(HashMap::new()),
            image_embeddings: Mutex::new(HashMap::new()),
            audio_embeddings: Mutex::new(HashMap::new()),
        }
    }

    /// Detect and extract a file once during this run.
    pub async fn extract(
        &self,
        path: &Path,
        mime: Option<&str>,
        extractors: &[Arc<dyn ContentExtractor>],
    ) -> Option<ExtractedContent> {
        if let Ok(cache) = self.extracted.lock() {
            if let Some(cached) = cache.get(path) {
                return cached.clone();
            }
        }

        let extracted = match tidyup_extract::router::pick(extractors, path, mime) {
            Some(extractor) => extractor.extract(path).await.ok(),
            None => None,
        };
        if let Ok(mut cache) = self.extracted.lock() {
            cache.insert(path.to_path_buf(), extracted.clone());
        }
        extracted
    }

    /// Read and embed an image once for one backend model during this run.
    pub async fn image_embedding(
        &self,
        path: &Path,
        mime: &str,
        backend: &dyn ImageEmbeddingBackend,
    ) -> Result<Option<Arc<Vec<f32>>>> {
        let key = (path.to_path_buf(), backend.model_id().to_string());
        if let Ok(cache) = self.image_embeddings.lock() {
            if let Some(cached) = cache.get(&key) {
                return Ok(Some(Arc::clone(cached)));
            }
        }
        if file_too_large(path).await {
            return Ok(None);
        }
        let bytes = tokio::fs::read(path).await?;
        let content_hash = ContentHash(blake3::hash(&bytes).to_hex().to_string());
        if let Some(store) = self.persistent {
            match store
                .semantic_artifact(&content_hash, backend.model_id(), "image-v1", "image")
                .await
            {
                Ok(Some(artifact)) if artifact.embedding.len() == backend.dimensions() => {
                    let embedding = Arc::new(artifact.embedding);
                    if let Ok(mut cache) = self.image_embeddings.lock() {
                        cache.insert(key, Arc::clone(&embedding));
                    }
                    return Ok(Some(embedding));
                }
                Ok(_) => {}
                Err(error) => tracing::warn!(%error, "persistent image artifact lookup failed"),
            }
        }
        let embedding = Arc::new(backend.embed_image(&bytes, mime).await?);
        if let Ok(mut cache) = self.image_embeddings.lock() {
            cache.insert(key, Arc::clone(&embedding));
        }
        self.persist_embedding(
            content_hash,
            backend.model_id(),
            "image-v1",
            "image",
            embedding.as_slice(),
        )
        .await;
        Ok(Some(embedding))
    }

    /// Read and embed audio once for one backend model during this run.
    pub async fn audio_embedding(
        &self,
        path: &Path,
        mime: &str,
        backend: &dyn AudioEmbeddingBackend,
    ) -> Result<Option<Arc<Vec<f32>>>> {
        let key = (path.to_path_buf(), backend.model_id().to_string());
        if let Ok(cache) = self.audio_embeddings.lock() {
            if let Some(cached) = cache.get(&key) {
                return Ok(Some(Arc::clone(cached)));
            }
        }
        if file_too_large(path).await {
            return Ok(None);
        }
        let bytes = tokio::fs::read(path).await?;
        let content_hash = ContentHash(blake3::hash(&bytes).to_hex().to_string());
        if let Some(store) = self.persistent {
            match store
                .semantic_artifact(&content_hash, backend.model_id(), "audio-v1", "audio")
                .await
            {
                Ok(Some(artifact)) if artifact.embedding.len() == backend.dimensions() => {
                    let embedding = Arc::new(artifact.embedding);
                    if let Ok(mut cache) = self.audio_embeddings.lock() {
                        cache.insert(key, Arc::clone(&embedding));
                    }
                    return Ok(Some(embedding));
                }
                Ok(_) => {}
                Err(error) => tracing::warn!(%error, "persistent audio artifact lookup failed"),
            }
        }
        let embedding = Arc::new(backend.embed_audio(&bytes, mime).await?);
        if let Ok(mut cache) = self.audio_embeddings.lock() {
            cache.insert(key, Arc::clone(&embedding));
        }
        self.persist_embedding(
            content_hash,
            backend.model_id(),
            "audio-v1",
            "audio",
            embedding.as_slice(),
        )
        .await;
        Ok(Some(embedding))
    }

    async fn persist_embedding(
        &self,
        content_hash: ContentHash,
        model_id: &str,
        preprocessing_version: &str,
        latent_space: &str,
        embedding: &[f32],
    ) {
        let Some(store) = self.persistent else {
            return;
        };
        let artifact = SemanticArtifact {
            content_hash,
            model_id: model_id.to_string(),
            preprocessing_version: preprocessing_version.to_string(),
            latent_space: latent_space.to_string(),
            embedding: embedding.to_vec(),
            updated_at: Utc::now(),
        };
        if let Err(error) = store.put_semantic_artifact(&artifact).await {
            tracing::warn!(%error, "persistent semantic artifact write failed");
        }
    }
}

async fn file_too_large(path: &Path) -> bool {
    tokio::fs::metadata(path).await.map_or(0, |meta| meta.len())
        > tidyup_extract::MAX_DOCUMENT_BYTES
}

/// Build the bounded general visual concept bank in one batched text-tower call.
///
/// The bank is data, not routing control flow: concepts describe visible
/// entities/scenes and never map directly to destination paths.
pub async fn prepare_visual_concepts(
    backend: &dyn ImageEmbeddingBackend,
) -> Result<Vec<GroundedConcept>> {
    prepare_visual_concepts_with_labels(backend, &[]).await
}

/// Build the general visual bank plus dynamic labels learned from a target
/// hierarchy. Dynamic labels receive no destination mapping; they are merely
/// additional text queries in the same contrastive space.
pub async fn prepare_visual_concepts_with_labels(
    backend: &dyn ImageEmbeddingBackend,
    dynamic_labels: &[String],
) -> Result<Vec<GroundedConcept>> {
    let mut entries: Vec<(String, String, String)> = visual_concept_entries()
        .into_iter()
        .map(|(label, prompt, family)| (label.to_string(), prompt.to_string(), family.to_string()))
        .collect();
    let mut claimed = entries
        .iter()
        .map(|(label, _, _)| concept_key(label))
        .collect::<std::collections::HashSet<_>>();
    for dynamic in dynamic_labels {
        let label = sanitize_filename(dynamic);
        if label.is_empty() || !claimed.insert(concept_key(&label)) {
            continue;
        }
        entries.push((
            label.clone(),
            format!("an image associated with {dynamic}"),
            label,
        ));
    }
    let prompts: Vec<&str> = entries
        .iter()
        .map(|(_, prompt, _)| prompt.as_str())
        .collect();
    let embeddings = backend.embed_texts(&prompts).await?;
    Ok(entries
        .into_iter()
        .zip(embeddings)
        .map(|((label, prompt, family), embedding)| GroundedConcept {
            label,
            prompt,
            family,
            embedding,
        })
        .collect())
}

fn concept_key(label: &str) -> String {
    label.to_ascii_lowercase().trim_end_matches('s').to_string()
}

/// Rank concepts in the same latent space as `content_embedding`.
#[must_use]
pub fn rank_concepts(content_embedding: &[f32], concepts: &[GroundedConcept]) -> Vec<ConceptMatch> {
    let mut ranked: Vec<ConceptMatch> = concepts
        .iter()
        .map(|concept| ConceptMatch {
            label: concept.label.clone(),
            score: cosine(content_embedding, &concept.embedding).max(0.0),
            family: concept.family.clone(),
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.label.cmp(&right.label))
    });
    ranked
}

/// Apply destination, grounding, ambiguity, and filename-mismatch gates to a
/// non-textual semantic rename.
#[must_use]
pub(crate) fn gate_grounded_rename(
    path: &Path,
    content_embedding: &[f32],
    concepts: &[GroundedConcept],
    classification_confidence: f32,
    config: &ClassifierConfig,
) -> GroundedRename {
    if classification_confidence < config.embedding_threshold {
        return GroundedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: None,
            concepts: Vec::new(),
        };
    }
    let ranked = rank_concepts(content_embedding, concepts);
    let selected = select_grounded_concepts(&ranked, config);
    if selected.is_empty() {
        return GroundedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: None,
            concepts: Vec::new(),
        };
    }
    let mismatch = grounded_filename_mismatch(path, &selected);
    if mismatch < config.rename.min_grounded_mismatch {
        return GroundedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: Some(mismatch),
            concepts: selected,
        };
    }
    GroundedRename {
        proposal: propose_grounded_rename(path, &selected),
        mismatch_score: Some(mismatch),
        concepts: selected,
    }
}

fn select_grounded_concepts(
    ranked: &[ConceptMatch],
    config: &ClassifierConfig,
) -> Vec<ConceptMatch> {
    let Some(first) = ranked.first() else {
        return Vec::new();
    };
    if first.score < config.rename.min_grounding_confidence {
        return Vec::new();
    }
    let second_index = ranked
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, concept)| !concepts_redundant(first, concept))
        .map(|(index, _)| index);
    let second = second_index.and_then(|index| ranked.get(index));
    let third_score = second_index
        .and_then(|index| {
            ranked.iter().skip(index.saturating_add(1)).find(|concept| {
                !concepts_redundant(first, concept)
                    && second.is_none_or(|second| !concepts_redundant(second, concept))
            })
        })
        .map_or(0.0, |concept| concept.score);
    if let Some(second) = second {
        if second.score >= config.rename.min_grounding_confidence
            && second.score - third_score >= config.rename.min_grounding_gap
        {
            return vec![first.clone(), second.clone()];
        }
        if first.score - second.score < config.rename.min_grounding_gap {
            return Vec::new();
        }
    }
    vec![first.clone()]
}

fn concepts_redundant(left: &ConceptMatch, right: &ConceptMatch) -> bool {
    if left.family == right.family {
        return true;
    }
    let left = left.label.trim_end_matches('s');
    let right = right.label.trim_end_matches('s');
    left == right
        || (left.len() >= 4 && right.len() >= 4 && (left.contains(right) || right.contains(left)))
}

fn grounded_filename_mismatch(path: &Path, concepts: &[ConceptMatch]) -> f32 {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let tokens = stem
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<std::collections::HashSet<_>>();
    let matched = concepts
        .iter()
        .filter(|concept| tokens.contains(concept.label.as_str()))
        .count();
    let matched_u16 = u16::try_from(matched).unwrap_or(0);
    let count_u16 = u16::try_from(concepts.len()).unwrap_or(1);
    1.0 - f32::from(matched_u16) / f32::from(count_u16)
}

/// L2-safe cosine helper shared by semantic artifact consumers.
#[must_use]
pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

fn visual_concept_entries() -> Vec<(&'static str, &'static str, &'static str)> {
    include_str!("../assets/visual_concepts.tsv")
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let label = fields.next()?;
            let prompt = fields.next()?;
            let family = fields.next().unwrap_or(label);
            Some((label, prompt, family))
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    struct CountingImageBackend {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ImageEmbeddingBackend for CountingImageBackend {
        async fn embed_image(&self, _image_bytes: &[u8], _mime: &str) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![1.0, 0.0])
        }

        async fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0, 0.0]; texts.len()])
        }

        fn dimensions(&self) -> usize {
            2
        }

        fn model_id(&self) -> &'static str {
            "counting-image"
        }
    }

    #[test]
    fn concept_ranking_is_deterministic() {
        let concepts = vec![
            GroundedConcept {
                label: "cat".to_string(),
                prompt: "cat".to_string(),
                family: "animal".to_string(),
                embedding: vec![1.0, 0.0],
            },
            GroundedConcept {
                label: "animal".to_string(),
                prompt: "animal".to_string(),
                family: "animal".to_string(),
                embedding: vec![0.95, 0.05],
            },
            GroundedConcept {
                label: "beach".to_string(),
                prompt: "beach".to_string(),
                family: "scene".to_string(),
                embedding: vec![0.8, 0.2],
            },
        ];
        let ranked = rank_concepts(&[1.0, 0.0], &concepts);
        assert_eq!(ranked[0].label, "cat");
        assert_eq!(ranked[1].label, "animal");
    }

    #[test]
    fn concept_bank_contains_reference_case() {
        let labels: Vec<&str> = visual_concept_entries()
            .into_iter()
            .map(|(label, _, _)| label)
            .collect();
        assert!(labels.contains(&"cat"));
        assert!(labels.contains(&"beach"));
    }

    #[tokio::test]
    async fn dynamic_labels_do_not_duplicate_plural_builtin_concepts() {
        let backend = CountingImageBackend {
            calls: AtomicUsize::new(0),
        };
        let concepts = prepare_visual_concepts_with_labels(
            &backend,
            &["Cats".to_string(), "Pet Photos".to_string()],
        )
        .await
        .unwrap();
        assert!(concepts.iter().any(|concept| concept.label == "cat"));
        assert!(!concepts.iter().any(|concept| concept.label == "cats"));
        assert!(concepts.iter().any(|concept| concept.label == "pet_photos"));
    }

    #[test]
    fn grounded_gate_names_cat_on_beach_and_respects_mismatch() {
        let concepts = vec![
            GroundedConcept {
                label: "cat".to_string(),
                prompt: "cat".to_string(),
                family: "animal".to_string(),
                embedding: vec![1.0, 0.0],
            },
            GroundedConcept {
                label: "animal".to_string(),
                prompt: "animal".to_string(),
                family: "animal".to_string(),
                embedding: vec![0.95, 0.05],
            },
            GroundedConcept {
                label: "beach".to_string(),
                prompt: "beach".to_string(),
                family: "scene".to_string(),
                embedding: vec![0.8, 0.2],
            },
            GroundedConcept {
                label: "office".to_string(),
                prompt: "office".to_string(),
                family: "place".to_string(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let config = ClassifierConfig {
            embedding_threshold: 0.30,
            ambiguity_gap: 0.01,
            ..ClassifierConfig::default()
        };

        let unnamed = gate_grounded_rename(
            Path::new("/downloads/image_2348985fg.png"),
            &[1.0, 0.0],
            &concepts,
            0.90,
            &config,
        );
        assert_eq!(
            unnamed.proposal,
            RenameProposal::Rename {
                name: "cat_on_beach.png".to_string(),
                source: crate::naming::RenameSource::SemanticConcept,
            }
        );
        assert_eq!(unnamed.mismatch_score, Some(1.0));

        let descriptive = gate_grounded_rename(
            Path::new("/downloads/cat_on_beach.png"),
            &[1.0, 0.0],
            &concepts,
            0.90,
            &config,
        );
        assert_eq!(descriptive.proposal, RenameProposal::Keep);
        assert_eq!(descriptive.mismatch_score, Some(0.0));
    }

    #[tokio::test]
    async fn image_embedding_is_computed_once_per_run() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("image.png");
        std::fs::write(&path, b"small image fixture").unwrap();
        let backend = CountingImageBackend {
            calls: AtomicUsize::new(0),
        };
        let cache = SemanticRunCache::default();

        let first = cache
            .image_embedding(&path, "image/png", &backend)
            .await
            .unwrap();
        let second = cache
            .image_embedding(&path, "image/png", &backend)
            .await
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(backend.calls.load(Ordering::Relaxed), 1);
    }
}
