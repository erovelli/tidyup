//! `ServiceContext` construction for the desktop UI.
//!
//! Mirrors the CLI's `context.rs` in shape — the UI is a thin adapter at the
//! frontend seam, not a distinct application. Any divergence between this
//! module and the CLI's equivalent is a sign the wiring is drifting.
//!
//! # Privacy gates
//!
//! Same two-gate model as the CLI: power-user inference backends enter the
//! UI binary at compile-time via cargo features (none wired yet — the UI
//! ships with the default embedding-only path) and require per-invocation
//! activation at runtime. See `CLAUDE.md` → "Privacy model".

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use tidyup_app::config::{resolve_data_dir, TidyupConfig};
use tidyup_app::ServiceContext;
use tidyup_core::inference::{AudioEmbeddingBackend, ImageEmbeddingBackend};
use tidyup_embeddings_ort::{
    installation_instructions, verify_clap_model, verify_default_model, verify_siglip_model,
    ClapEmbeddings, OrtEmbeddings, SigLipEmbeddings,
};
use tidyup_storage_sqlite::SqliteStore;

/// Build a ready-to-use [`ServiceContext`] against the local data dir.
///
/// `strict_model = true` (scan/migrate path) fails fast with installer
/// instructions if the embedding model is missing. `strict_model = false`
/// (rollback / runs-listing path) falls back to a null embedding backend so
/// the UI can still browse run history without a model.
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
        match verify_and_load_default_embeddings().await {
            Ok(e) => e,
            Err(_) => Arc::new(NullEmbeddings),
        }
    };

    // Phase 7 multimodal — load SigLIP / CLAP if their bundles are present.
    // Mirrors the CLI's behaviour: missing bundles are not an error, image
    // and audio files just route through the text Tier 2 fallback.
    let image_embeddings = try_load_siglip();
    let audio_embeddings = try_load_clap();

    let extractors = default_extractors();

    Ok(Arc::new(ServiceContext {
        file_index: Arc::new(store.clone()),
        change_log: Arc::new(store.clone()),
        backup_store: Arc::new(store.clone()),
        run_log: Arc::new(store),
        // The UI has no Tier 3 activation surface yet — keeping `text: None`
        // upholds the privacy model (no per-invocation gate => no Tier 3).
        // A settings-page toggle is the natural next step.
        text: None,
        embeddings,
        vision: None,
        image_embeddings,
        audio_embeddings,
        extractors,
    }))
}

fn try_load_siglip() -> Option<Arc<dyn ImageEmbeddingBackend>> {
    if verify_siglip_model().is_err() {
        tracing::debug!("SigLIP bundle not present; image-modality Tier 2 disabled");
        return None;
    }
    match SigLipEmbeddings::load_default() {
        Ok(b) => {
            tracing::info!("SigLIP image encoder loaded");
            Some(Arc::new(b))
        }
        Err(e) => {
            tracing::warn!(error = %e, "SigLIP load failed");
            None
        }
    }
}

fn try_load_clap() -> Option<Arc<dyn AudioEmbeddingBackend>> {
    if verify_clap_model().is_err() {
        tracing::debug!("CLAP bundle not present; audio-modality Tier 2 disabled");
        return None;
    }
    match ClapEmbeddings::load_default() {
        Ok(b) => {
            tracing::info!("CLAP audio encoder loaded");
            Some(Arc::new(b))
        }
        Err(e) => {
            tracing::warn!(error = %e, "CLAP load failed");
            None
        }
    }
}

pub(crate) async fn verify_and_load_default_embeddings(
) -> Result<Arc<dyn tidyup_core::inference::EmbeddingBackend>> {
    if let Err(e) = verify_default_model() {
        return Err(anyhow!("{e}\n\n{}", installation_instructions()));
    }
    let embeddings = tokio::task::spawn_blocking(OrtEmbeddings::load_default)
        .await
        .map_err(|e| anyhow!("embedding load join: {e}"))??;
    Ok(Arc::new(embeddings))
}

fn default_extractors() -> Vec<Arc<dyn tidyup_core::extractor::ContentExtractor>> {
    vec![
        Arc::new(tidyup_extract::pdf::PdfExtractor),
        Arc::new(tidyup_extract::image::ImageExtractor),
        Arc::new(tidyup_extract::text::PlainTextExtractor),
    ]
}

/// Best-effort quick check used by the dashboard to short-circuit the model
/// installation banner before launching a scan/migrate. Does not load the
/// model — just verifies the artifacts are on disk.
pub(crate) fn quick_model_check() -> std::result::Result<(), String> {
    verify_default_model()
        .map(|_| ())
        .map_err(|e| format!("{e}\n\n{}", installation_instructions()))
}

/// Build scan-mode taxonomy candidates from the default taxonomy plus the
/// given embedding backend. Identical to the CLI helper.
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

/// Image-modality taxonomy candidates. Returns an empty vec when the `SigLIP`
/// backend isn't loaded.
///
/// # Errors
/// Propagates embedding backend failures.
pub(crate) async fn build_image_scan_candidates(
    image_backend: Option<&dyn ImageEmbeddingBackend>,
) -> Result<Vec<tidyup_pipeline::scan::ScanCandidate>> {
    let Some(backend) = image_backend else {
        return Ok(Vec::new());
    };
    let entries = tidyup_embeddings_ort::default_image_taxonomy();
    let texts: Vec<&str> = entries.iter().map(|e| e.description).collect();
    let vecs = backend.embed_texts(&texts).await?;
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

/// Audio-modality taxonomy candidates.
///
/// # Errors
/// Propagates embedding backend failures.
pub(crate) async fn build_audio_scan_candidates(
    audio_backend: Option<&dyn AudioEmbeddingBackend>,
) -> Result<Vec<tidyup_pipeline::scan::ScanCandidate>> {
    let Some(backend) = audio_backend else {
        return Ok(Vec::new());
    };
    let entries = tidyup_embeddings_ort::default_audio_taxonomy();
    let texts: Vec<&str> = entries.iter().map(|e| e.description).collect();
    let vecs = backend.embed_texts(&texts).await?;
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

// ---------------------------------------------------------------------------
// Null embedding backend — only used when the model bundle is missing on the
// rollback / runs-listing path. The text backend port is `Option`, so no null
// stand-in is needed there.
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
