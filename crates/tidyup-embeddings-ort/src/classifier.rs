//! Taxonomy-backed embedding classifier — Tier 2 of the classification cascade.
//!
//! Pairs an [`OrtEmbeddings`] with a list of [`TaxonomyEntry`]s. On load, it
//! either restores pre-computed category embeddings from disk or recomputes
//! and caches them. `classify()` embeds the input text and returns the best-
//! matching folder plus a raw cosine similarity score.
//!
//! Scores are cosine similarity of L2-normalized vectors, so values sit in
//! `[-1.0, 1.0]` with 1.0 = identical direction. The thresholds in
//! `ClassifierConfig::embedding_threshold` (default 0.35) are tuned against
//! this scale.

use std::sync::Arc;

use anyhow::Result;

use tidyup_core::inference::EmbeddingBackend;

use crate::embeddings::OrtEmbeddings;
use crate::taxonomy::{default_taxonomy, TaxonomyCache, TaxonomyEntry};
use crate::util::{cosine_similarity, extract_year};

/// Result of embedding-based classification.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingClassification {
    /// Hierarchical folder path (e.g. `"Finance/Taxes/2024/"`).
    pub folder: String,
    /// Raw cosine similarity in `[-1.0, 1.0]`.
    pub confidence: f32,
    /// Index into the taxonomy (useful for diagnostics).
    pub entry_index: usize,
}

/// Classifier that scores file content against a taxonomy via cosine similarity.
pub struct EmbeddingClassifier {
    embeddings: Arc<OrtEmbeddings>,
    entries: Vec<TaxonomyEntry>,
    category_embeddings: Vec<Vec<f32>>,
}

impl std::fmt::Debug for EmbeddingClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingClassifier")
            .field("entry_count", &self.entries.len())
            .field("model_id", &self.embeddings.model_id())
            .finish_non_exhaustive()
    }
}

impl EmbeddingClassifier {
    /// Construct a classifier using the default taxonomy. Loads category
    /// embeddings from [`crate::paths::taxonomy_cache_path`] if the hash
    /// matches the current model + taxonomy; otherwise embeds and caches.
    ///
    /// # Errors
    /// Propagates embedding errors on cache miss.
    pub async fn with_default_taxonomy(embeddings: Arc<OrtEmbeddings>) -> Result<Self> {
        Self::new(embeddings, default_taxonomy()).await
    }

    /// Construct a classifier with a caller-supplied taxonomy.
    ///
    /// # Errors
    /// Propagates embedding errors on cache miss.
    pub async fn new(embeddings: Arc<OrtEmbeddings>, entries: Vec<TaxonomyEntry>) -> Result<Self> {
        let model_id = embeddings.model_id().to_string();
        let category_embeddings = Self::load_or_compute(&embeddings, &entries, &model_id).await?;
        Ok(Self {
            embeddings,
            entries,
            category_embeddings,
        })
    }

    async fn load_or_compute(
        embeddings: &Arc<OrtEmbeddings>,
        entries: &[TaxonomyEntry],
        model_id: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let cache_path = crate::paths::taxonomy_cache_path();

        if let Some(path) = &cache_path {
            if let Some(cache) = TaxonomyCache::load(path) {
                if cache.is_valid(model_id, entries) {
                    tracing::info!(
                        entries = cache.embeddings.len(),
                        "loaded taxonomy embeddings from cache",
                    );
                    return Ok(cache.embeddings);
                }
                tracing::debug!("taxonomy cache invalid — rebuilding");
            }
        }

        tracing::info!(
            entries = entries.len(),
            "pre-computing taxonomy embeddings (first run)",
        );
        let mut category_embeddings = Vec::with_capacity(entries.len());
        for chunk in entries.chunks(32) {
            let texts: Vec<&str> = chunk.iter().map(|e| e.description).collect();
            let mut batch = embeddings.embed_texts(&texts).await?;
            category_embeddings.append(&mut batch);
        }

        if let Some(path) = cache_path {
            let cache = TaxonomyCache {
                model_id: model_id.to_string(),
                entry_count: entries.len(),
                descriptions_hash: TaxonomyCache::compute_hash(entries),
                embeddings: category_embeddings.clone(),
            };
            if let Err(e) = cache.save(&path) {
                tracing::warn!("failed to save taxonomy cache: {e}");
            }
        }

        Ok(category_embeddings)
    }

    /// Classify file content.
    ///
    /// `text` is the extracted body; `filename` is the basename. A short
    /// excerpt is concatenated with the filename as the query — the filename
    /// alone often carries strong signal (e.g. `"2024_tax_return.pdf"`).
    ///
    /// # Errors
    /// Propagates embedding errors.
    pub async fn classify(&self, text: &str, filename: &str) -> Result<EmbeddingClassification> {
        let query = build_query(text, filename);
        let query_embedding = self.embeddings.embed_text(&query).await?;
        Ok(self.classify_embedding(&query_embedding, text, filename))
    }

    /// Classify from a pre-computed embedding — the fast path for batch jobs
    /// where content was already embedded in an earlier pass.
    #[must_use]
    pub fn classify_embedding(
        &self,
        embedding: &[f32],
        text: &str,
        filename: &str,
    ) -> EmbeddingClassification {
        let (best_idx, best_score) = rank_best(embedding, &self.category_embeddings);
        finalize(&self.entries, best_idx, best_score, text, filename)
    }

    /// Returns the ranked top-k candidates by cosine similarity.
    #[must_use]
    pub fn top_k(&self, embedding: &[f32], k: usize) -> Vec<EmbeddingClassification> {
        rank_top_k(embedding, &self.category_embeddings, k)
            .into_iter()
            .map(|(i, score)| EmbeddingClassification {
                folder: self.entries[i].path.to_string(),
                confidence: score,
                entry_index: i,
            })
            .collect()
    }

    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        self.embeddings.model_id()
    }

    /// Access the underlying embedding backend for callers that need to
    /// embed queries or content directly.
    #[must_use]
    pub const fn embeddings(&self) -> &Arc<OrtEmbeddings> {
        &self.embeddings
    }
}

/// Build the query string for classification: filename + content excerpt.
///
/// A 500-character excerpt keeps inference fast while giving enough context
/// to disambiguate categories that would collide on filename alone.
fn build_query(text: &str, filename: &str) -> String {
    if text.is_empty() {
        filename.to_string()
    } else {
        let cut = text.char_indices().nth(500).map_or(text.len(), |(i, _)| i);
        format!("{filename} {}", &text[..cut])
    }
}

/// Highest cosine-similarity index + score from a slice of category vectors.
///
/// Returns `(0, 0.0)` on an empty input so callers don't need to branch.
fn rank_best(embedding: &[f32], category_embeddings: &[Vec<f32>]) -> (usize, f32) {
    category_embeddings
        .iter()
        .enumerate()
        .map(|(i, cat)| (i, cosine_similarity(embedding, cat)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0))
}

/// Top-k category indices sorted by descending cosine similarity.
fn rank_top_k(embedding: &[f32], category_embeddings: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = category_embeddings
        .iter()
        .enumerate()
        .map(|(i, cat)| (i, cosine_similarity(embedding, cat)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Materialize a ranked index into a user-facing [`EmbeddingClassification`].
///
/// Resolves the `temporal` taxonomy flag by extracting a year from the text /
/// filename and appending it to the folder path.
fn finalize(
    entries: &[TaxonomyEntry],
    idx: usize,
    score: f32,
    text: &str,
    filename: &str,
) -> EmbeddingClassification {
    let entry = &entries[idx];
    let folder = if entry.temporal {
        extract_year(text, filename)
            .map_or_else(|| entry.path.to_string(), |y| format!("{}{y}/", entry.path))
    } else {
        entry.path.to_string()
    };
    EmbeddingClassification {
        folder,
        confidence: score,
        entry_index: idx,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn build_query_handles_empty_text() {
        assert_eq!(build_query("", "file.pdf"), "file.pdf");
    }

    #[test]
    fn build_query_truncates_text() {
        let text = "a".repeat(1000);
        let q = build_query(&text, "file.pdf");
        assert!(q.starts_with("file.pdf "));
        assert!(q.len() < 600);
    }

    #[test]
    fn build_query_preserves_utf8() {
        let text = "é".repeat(300);
        let q = build_query(&text, "file.pdf");
        assert!(q.is_char_boundary(q.len()));
    }

    // ---- ranking helpers ---------------------------------------------------
    //
    // These tests exercise the pure-math core of the classifier without
    // requiring an ONNX model. Vectors below are hand-designed L2-normalized
    // 3-dim unit vectors so cosine similarity reduces to the dot product.

    fn taxonomy_fixture() -> Vec<TaxonomyEntry> {
        vec![
            TaxonomyEntry {
                path: "Finance/Taxes/",
                description: "tax forms and returns",
                temporal: true,
            },
            TaxonomyEntry {
                path: "Work/Reports/",
                description: "work reports and memos",
                temporal: false,
            },
            TaxonomyEntry {
                path: "Travel/",
                description: "travel bookings",
                temporal: false,
            },
        ]
    }

    #[test]
    fn rank_best_picks_highest_cosine() {
        // cosine_similarity internally normalizes; query dotted with the
        // first basis vector gives ~0.99 once magnitudes are factored in.
        let categories = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![0.9, 0.1, 0.0];
        let (idx, score) = rank_best(&query, &categories);
        assert_eq!(idx, 0);
        assert!(score > 0.99);
    }

    #[test]
    fn rank_best_returns_default_on_empty() {
        assert_eq!(rank_best(&[1.0, 0.0, 0.0], &[]), (0, 0.0));
    }

    #[test]
    fn rank_top_k_is_sorted_descending() {
        let categories = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.6, 0.8, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let query = vec![1.0, 0.0, 0.0];
        let ranked = rank_top_k(&query, &categories, 3);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].0, 0);
        assert_eq!(ranked[1].0, 1);
        assert_eq!(ranked[2].0, 2);
        assert!(ranked[0].1 >= ranked[1].1);
        assert!(ranked[1].1 >= ranked[2].1);
    }

    #[test]
    fn rank_top_k_truncates_to_k() {
        let categories = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ranked = rank_top_k(&[1.0, 1.0, 1.0], &categories, 2);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn finalize_temporal_entry_appends_year() {
        let entries = taxonomy_fixture();
        let result = finalize(&entries, 0, 0.9, "", "2024_tax_return.pdf");
        assert_eq!(result.folder, "Finance/Taxes/2024/");
        assert_eq!(result.entry_index, 0);
    }

    #[test]
    fn finalize_non_temporal_entry_returns_path_as_is() {
        let entries = taxonomy_fixture();
        let result = finalize(&entries, 1, 0.8, "quarterly review", "review.pdf");
        assert_eq!(result.folder, "Work/Reports/");
    }

    #[test]
    fn finalize_temporal_entry_without_year_stays_bare() {
        let entries = taxonomy_fixture();
        let result = finalize(&entries, 0, 0.7, "plain text", "notes.md");
        assert_eq!(result.folder, "Finance/Taxes/");
    }

    #[test]
    fn confidence_is_bounded_for_normalized_vectors() {
        // Unit vectors -> cosine is in [-1, 1]. Exercise a few pairs.
        let categories = [vec![1.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0]];
        let (_, pos) = rank_best(&[1.0, 0.0, 0.0], &categories[..1]);
        let (_, neg) = rank_best(&[1.0, 0.0, 0.0], &categories[1..]);
        assert!((pos - 1.0).abs() < 1e-6);
        assert!((neg + 1.0).abs() < 1e-6);
    }
}
