//! `ServiceContext` construction — the backend wiring done once per invocation.
//!
//! This is where the CLI translates a parsed config + runtime flags into
//! concrete impls of the port traits consumed by the application services.
//!
//! # Privacy gates
//!
//! The default build links neither `tidyup-inference-mistralrs` nor
//! `tidyup-inference-remote`. When those features are compiled in, *inclusion*
//! does not imply *activation* — config-level `llm_fallback = true` /
//! `backends = ["remote-..."]` must still be joined with a per-invocation flag
//! (`--llm-fallback` / `--remote` or the matching env vars). See
//! `CLAUDE.md` → "Privacy model".
//!
//! Phase 5 ships without backend selection wiring — the default embedding-only
//! path is the only one exercised. A no-op `TextBackend` stands in for the
//! port; the pipeline doesn't need a text backend on the default cascade.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use tidyup_app::config::{resolve_data_dir, TidyupConfig};
use tidyup_app::ServiceContext;
use tidyup_core::inference::{ContentClassification, GenerationOptions, TextBackend};
use tidyup_embeddings_ort::{installation_instructions, verify_default_model, OrtEmbeddings};
use tidyup_storage_sqlite::SqliteStore;

/// Top-level build: given a parsed config, produce a ready-to-use
/// [`ServiceContext`] with the default storage + embedding backend and the
/// default content extractors.
///
/// `strict_model` controls the first-run behaviour: when `true` (the default
/// for migrate/scan), a missing or corrupt model bundle is a hard error with
/// user-facing installation instructions. Rollback may pass `false` since it
/// does not invoke the classifier.
///
/// # Errors
/// Propagates model-verification, database-open, and embedding-load errors.
pub(crate) async fn build(
    config: &TidyupConfig,
    strict_model: bool,
) -> Result<Arc<ServiceContext>> {
    let data_dir = resolve_data_dir(&config.storage)?;
    std::fs::create_dir_all(&data_dir)
        .with_context(|| format!("creating tidyup data dir {}", data_dir.display()))?;
    let db_path = data_dir.join("tidyup.db");
    let shelf = data_dir.join("backup");
    std::fs::create_dir_all(&shelf)
        .with_context(|| format!("creating backup shelf {}", shelf.display()))?;

    let store = SqliteStore::open(&db_path)
        .with_context(|| format!("opening sqlite at {}", db_path.display()))?
        .with_backup_root(shelf);

    let embeddings = if strict_model {
        verify_and_load_default_embeddings().await?
    } else {
        // Rollback doesn't classify — a degraded fallback is fine.
        match verify_and_load_default_embeddings().await {
            Ok(e) => e,
            Err(_) => Arc::new(NullEmbeddings),
        }
    };

    let extractors = default_extractors();

    Ok(Arc::new(ServiceContext {
        file_index: Arc::new(store.clone()),
        change_log: Arc::new(store.clone()),
        backup_store: Arc::new(store.clone()),
        run_log: Arc::new(store),
        text: Arc::new(NullTextBackend),
        embeddings,
        vision: None,
        extractors,
    }))
}

/// Verify model artifacts are present and load the embedding backend.
///
/// # Errors
/// Surfaces [`installation_instructions`] as the error message when artifacts
/// are missing, so the CLI can display actionable next steps.
pub(crate) async fn verify_and_load_default_embeddings(
) -> Result<Arc<dyn tidyup_core::inference::EmbeddingBackend>> {
    if let Err(e) = verify_default_model() {
        return Err(anyhow!("{e}\n\n{}", installation_instructions(),));
    }
    // Load synchronously — ORT init is cheap; `spawn_blocking` would add noise.
    let embeddings = tokio::task::spawn_blocking(OrtEmbeddings::load_default)
        .await
        .map_err(|e| anyhow!("embedding load join: {e}"))??;
    Ok(Arc::new(embeddings))
}

fn default_extractors() -> Vec<Arc<dyn tidyup_core::extractor::ContentExtractor>> {
    // Registration order matters: more-specific extractors first, plain text
    // as the catch-all fallback. The CLI picks up `tidyup-extract`'s default
    // features (`text`, `pdf`, `image`), so these modules are always in scope.
    vec![
        Arc::new(tidyup_extract::pdf::PdfExtractor),
        Arc::new(tidyup_extract::image::ImageExtractor),
        Arc::new(tidyup_extract::text::PlainTextExtractor),
    ]
}

/// Read-only config snapshot helper — placed here so command handlers can
/// surface where the TOML lives.
#[must_use]
pub(crate) fn describe_data_dir(config: &TidyupConfig) -> Option<String> {
    resolve_data_dir(&config.storage)
        .ok()
        .map(|p| p.display().to_string())
}

// ---------------------------------------------------------------------------
// Minimal stand-in backends used by the default path. A proper registry lands
// in Phase 6+; for v0.1 the classifier only consumes the embedding backend.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct NullEmbeddings;

#[async_trait]
impl tidyup_core::inference::EmbeddingBackend for NullEmbeddings {
    async fn embed_text(&self, _text: &str) -> tidyup_core::Result<Vec<f32>> {
        Err(anyhow!("embedding backend not loaded"))
    }
    async fn embed_texts(&self, _texts: &[&str]) -> tidyup_core::Result<Vec<Vec<f32>>> {
        Err(anyhow!("embedding backend not loaded"))
    }
    fn dimensions(&self) -> usize {
        0
    }
    fn model_id(&self) -> &'static str {
        "null"
    }
}

#[derive(Debug)]
struct NullTextBackend;

#[async_trait]
impl TextBackend for NullTextBackend {
    async fn classify_text(
        &self,
        _text: &str,
        _filename: &str,
    ) -> tidyup_core::Result<ContentClassification> {
        Err(anyhow!(
            "text backend not enabled (default build has no LLM)"
        ))
    }
    async fn classify_audio(
        &self,
        _filename: &str,
        _metadata: &str,
    ) -> tidyup_core::Result<ContentClassification> {
        Err(anyhow!("text backend not enabled"))
    }
    async fn classify_video(
        &self,
        _filename: &str,
        _frame_captions: &[String],
    ) -> tidyup_core::Result<ContentClassification> {
        Err(anyhow!("text backend not enabled"))
    }
    async fn classify_image_description(
        &self,
        _filename: &str,
        _description: &str,
    ) -> tidyup_core::Result<ContentClassification> {
        Err(anyhow!("text backend not enabled"))
    }
    async fn complete(
        &self,
        _prompt: &str,
        _opts: &GenerationOptions,
    ) -> tidyup_core::Result<String> {
        Err(anyhow!("text backend not enabled"))
    }
    fn model_id(&self) -> &'static str {
        "null"
    }
}

/// Build the scan taxonomy — taxonomy entries + pre-computed description
/// embeddings — into [`tidyup_pipeline::scan::ScanCandidate`]s.
///
/// The embedding backend is expected to be [`OrtEmbeddings`]-shaped (i.e. the
/// real on-disk model) for scan to produce useful output; the null backend
/// fails closed, which is exactly the right behaviour.
pub(crate) async fn build_default_scan_candidates(
    embeddings: &dyn tidyup_core::inference::EmbeddingBackend,
) -> Result<Vec<tidyup_pipeline::scan::ScanCandidate>> {
    let entries = tidyup_embeddings_ort::default_taxonomy();
    let texts: Vec<&str> = entries.iter().map(|e| e.description).collect();
    let vecs = embeddings.embed_texts(&texts).await?;
    let mut out = Vec::with_capacity(entries.len());
    for (entry, emb) in entries.into_iter().zip(vecs) {
        out.push(tidyup_pipeline::scan::ScanCandidate {
            folder_path: entry.path.to_string(),
            description: entry.description.to_string(),
            temporal: entry.temporal,
            embedding: emb,
        });
    }
    Ok(out)
}
