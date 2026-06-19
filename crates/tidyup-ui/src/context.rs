//! `ServiceContext` construction for the desktop UI.
//!
//! Mirrors the CLI's `context.rs` in shape — the UI is a thin adapter at the
//! frontend seam, not a distinct application. Any divergence between this
//! module and the CLI's equivalent is a sign the wiring is drifting.
//!
//! # Privacy gates
//!
//! Same triple-gated model as the CLI for Tier 3 LLM fallback: the
//! `llm-fallback` cargo feature must be compiled in (gate 1), `[inference]
//! llm_fallback = true` must be set in config (gate 2), and the per-invocation
//! [`InferenceActivation`] — sourced from the Settings session toggle — must
//! request it (gate 3). With the default activation the context's
//! [`text`](tidyup_app::ServiceContext::text) field is `None` and Tier 3 is
//! never invoked, identical to a default build with no feature compiled in. The
//! UI deliberately does not surface the `remote` backend; that stays CLI-only.
//! See `CLAUDE.md` → "Privacy model".

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use tidyup_app::config::{resolve_data_dir, TidyupConfig};
use tidyup_app::ServiceContext;
use tidyup_core::inference::{AudioEmbeddingBackend, ImageEmbeddingBackend, TextBackend};
use tidyup_embeddings_ort::{
    installation_instructions, verify_clap_model, verify_default_model, verify_siglip_model,
    ClapEmbeddings, OrtEmbeddings, SigLipEmbeddings,
};
use tidyup_storage_sqlite::SqliteStore;

/// Per-invocation inference activation gate — the UI mirror of the CLI's
/// struct of the same name.
///
/// Only a build with the `llm-fallback` feature can meaningfully set
/// `llm_fallback = true`; with the default activation (all-false) the
/// [`ServiceContext::text`] field is `None` and Tier 3 stays off. The UI sources
/// this from the Settings session toggle, which is itself gated on the cargo
/// feature + the config bool, so the three gates align before this is ever
/// `true`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct InferenceActivation {
    /// Triple-gated activation for the local LLM fallback (mistralrs).
    pub llm_fallback: bool,
}

/// Build a ready-to-use [`ServiceContext`] against the local data dir.
///
/// `strict_model = true` (scan/migrate path) fails fast with installer
/// instructions if the embedding model is missing. `strict_model = false`
/// (rollback / runs-listing path) falls back to a null embedding backend so
/// the UI can still browse run history without a model.
///
/// `activation` carries the per-invocation Tier 3 gate. When `llm_fallback` is
/// `false` (the default) the [`ServiceContext::text`] field is `None`. The
/// rollback / runs path passes the default activation since it never classifies.
pub(crate) async fn build(
    config: &TidyupConfig,
    strict_model: bool,
    activation: InferenceActivation,
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

    // Tier 3 text backend, loaded sequentially after the embedding model per the
    // CLAUDE.md operational rule (concurrent model loads OOM on 8GB hosts). Stays
    // `None` unless all three privacy gates align — see `build_text_backend`.
    let text = build_text_backend(config, activation).await?;

    let extractors = default_extractors();

    Ok(Arc::new(ServiceContext {
        file_index: Arc::new(store.clone()),
        change_log: Arc::new(store.clone()),
        backup_store: Arc::new(store.clone()),
        run_log: Arc::new(store),
        text,
        embeddings,
        vision: None,
        image_embeddings,
        audio_embeddings,
        extractors,
    }))
}

/// Build the optional Tier 3 [`TextBackend`] from the per-invocation activation
/// + config. Returns `Ok(None)` for the privacy-preserving case where the
/// activation gate doesn't fire. Mirrors the CLI's `build_text_backend` (minus
/// the remote branch, which the UI does not surface).
async fn build_text_backend(
    config: &TidyupConfig,
    activation: InferenceActivation,
) -> Result<Option<Arc<dyn TextBackend>>> {
    if activation.llm_fallback {
        return build_llm_backend(config).await.map(Some);
    }
    Ok(None)
}

#[cfg(feature = "llm-fallback")]
async fn build_llm_backend(config: &TidyupConfig) -> Result<Arc<dyn TextBackend>> {
    if !config.inference.llm_fallback {
        return Err(anyhow!(
            "Tier 3 was requested but [inference] llm_fallback is false in config; \
             the privacy model requires both to enable Tier 3 LLM fallback"
        ));
    }
    let model_id = "Qwen/Qwen3-0.6B";
    tracing::info!(model_id, "loading mistralrs text backend (Tier 3)");
    let engine = tidyup_inference_mistralrs::MistralRsEngine::load(model_id)
        .await
        .context("loading mistralrs Tier 3 backend")?;
    Ok(engine as Arc<dyn TextBackend>)
}

#[cfg(not(feature = "llm-fallback"))]
#[allow(clippy::unused_async)] // signature mirrors the feature-on variant
async fn build_llm_backend(_config: &TidyupConfig) -> Result<Arc<dyn TextBackend>> {
    Err(anyhow!(
        "Tier 3 LLM fallback was requested but this desktop build was compiled \
         without the `llm-fallback` feature.\n\
         Rebuild with: cargo build --release -p tidyup-ui --features llm-fallback"
    ))
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
