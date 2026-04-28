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
//! [`InferenceActivation`] captures the per-invocation gate. The CLI parses
//! flags + env vars and builds it before calling [`build`]. With the default
//! activation (`llm_fallback: false`, `remote: false`) the context's
//! [`text`](tidyup_app::ServiceContext::text) field is `None` and Tier 3 is
//! never invoked — same shape as the default build with neither feature
//! compiled in.

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tidyup_app::config::{resolve_data_dir, TidyupConfig};
use tidyup_app::ServiceContext;
use tidyup_core::inference::{AudioEmbeddingBackend, ImageEmbeddingBackend, TextBackend};
use tidyup_embeddings_ort::{
    installation_instructions, verify_clap_model, verify_default_model, verify_siglip_model,
    ClapEmbeddings, OrtEmbeddings, SigLipEmbeddings,
};
use tidyup_storage_sqlite::SqliteStore;

/// Per-invocation inference activation gate.
///
/// Constructed in `commands::dispatch` from CLI flags + environment + config.
/// Only the cargo features compiled in can produce `true` values; with no
/// features the struct is always all-false and the loader treats the
/// activation as a no-op (Tier 3 stays off).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct InferenceActivation {
    /// Triple-gated activation for the local LLM fallback (mistralrs).
    pub llm_fallback: bool,
    /// Triple-gated activation for the remote text backend.
    pub remote: bool,
}

/// Top-level build: given a parsed config + activation gate, produce a
/// ready-to-use [`ServiceContext`].
///
/// `strict_model` controls the first-run behaviour: when `true` (the default
/// for migrate/scan), a missing or corrupt model bundle is a hard error with
/// user-facing installation instructions. Rollback may pass `false` since it
/// does not invoke the classifier.
///
/// `activation` carries the triple-gated per-invocation flags. When both
/// fields are `false` (the default), the [`ServiceContext::text`] field is
/// `None` and Tier 3 is never invoked — same shape as the default build with
/// neither LLM nor remote features compiled in. When activation requests a
/// backend the corresponding feature wasn't compiled in, the loader fails
/// fast with a rebuild hint.
///
/// # Errors
/// Propagates model-verification, database-open, and embedding-load errors.
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

    let embeddings: Arc<dyn tidyup_core::inference::EmbeddingBackend> = if strict_model {
        verify_and_load_default_embeddings().await?
    } else {
        // Rollback doesn't classify — a degraded fallback is fine.
        verify_and_load_default_embeddings()
            .await
            .unwrap_or_else(|_| Arc::new(NullEmbeddings))
    };

    // Phase 7: optional image/audio backends. Each is loaded only if its
    // bundle is present on disk — missing artifacts are NOT an error since
    // the default install path ships text-only. The pipeline gracefully
    // falls back when a backend is absent.
    let image_embeddings = try_load_siglip();
    let audio_embeddings = try_load_clap();

    // Tier 3 text backend. Loaded sequentially after the embedding model per
    // operational rule in CLAUDE.md (concurrent model loads OOM on 8GB hosts).
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

/// Build the optional Tier 3 [`TextBackend`] from the per-invocation
/// activation + config.
///
/// Precedence: `--remote` wins over `--llm-fallback` if both fire (a remote
/// endpoint is more likely to be the intentional choice when configured —
/// it's not the default for any path). Returns `Ok(None)` for the
/// privacy-preserving case where neither activation gate fires.
async fn build_text_backend(
    config: &TidyupConfig,
    activation: InferenceActivation,
) -> Result<Option<Arc<dyn TextBackend>>> {
    if activation.remote {
        return build_remote_backend(config).map(Some);
    }
    if activation.llm_fallback {
        return build_llm_backend(config).await.map(Some);
    }
    Ok(None)
}

#[cfg(feature = "llm-fallback")]
async fn build_llm_backend(config: &TidyupConfig) -> Result<Arc<dyn TextBackend>> {
    if !config.inference.llm_fallback {
        return Err(anyhow!(
            "--llm-fallback flag set but [inference] llm_fallback is false in config; \
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
        "--llm-fallback was passed but this binary was built without the `llm-fallback` feature.\n\
         Rebuild with: cargo build --release -p tidyup-cli --features llm-fallback"
    ))
}

#[cfg(feature = "remote")]
fn build_remote_backend(config: &TidyupConfig) -> Result<Arc<dyn TextBackend>> {
    use tidyup_inference_remote::{RemoteEndpoint, RemoteText};
    let remote_cfg = config.inference.remote.as_ref().ok_or_else(|| {
        anyhow!(
            "--remote flag set but [inference.remote] is missing from config. \
             Add an `[inference.remote]` section with `endpoint`, `api_key_env`, and `model`."
        )
    })?;
    let api_key = std::env::var(&remote_cfg.api_key_env).map_err(|_| {
        anyhow!(
            "remote backend requires the API key env var `{}` to be set",
            remote_cfg.api_key_env
        )
    })?;
    let endpoint = RemoteEndpoint::OpenAi {
        url: remote_cfg.endpoint.clone(),
        api_key,
        model: remote_cfg.model.clone(),
    };
    tracing::info!(model = %remote_cfg.model, "loading remote text backend (Tier 3)");
    let backend = RemoteText::new(endpoint).context("constructing remote text backend")?;
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "remote"))]
fn build_remote_backend(_config: &TidyupConfig) -> Result<Arc<dyn TextBackend>> {
    Err(anyhow!(
        "--remote was passed but this binary was built without the `remote` feature.\n\
         Rebuild with: cargo build --release -p tidyup-cli --features remote"
    ))
}

/// Best-effort load of the `SigLIP` image encoder. Returns `None` when the
/// bundle is missing or fails to load — the caller surfaces the absence as
/// a soft fallback to text-tier classification, not an error.
fn try_load_siglip() -> Option<Arc<dyn ImageEmbeddingBackend>> {
    if verify_siglip_model().is_err() {
        tracing::debug!("SigLIP bundle not present; image-modality Tier 2 disabled");
        return None;
    }
    match SigLipEmbeddings::load_default() {
        Ok(b) => {
            tracing::info!("SigLIP image encoder loaded — image-modality Tier 2 enabled");
            Some(Arc::new(b))
        }
        Err(e) => {
            tracing::warn!(error = %e, "SigLIP load failed; falling back to text Tier 2");
            None
        }
    }
}

/// Best-effort load of the `CLAP` audio encoder. Same fallback semantics as
/// [`try_load_siglip`].
fn try_load_clap() -> Option<Arc<dyn AudioEmbeddingBackend>> {
    if verify_clap_model().is_err() {
        tracing::debug!("CLAP bundle not present; audio-modality Tier 2 disabled");
        return None;
    }
    match ClapEmbeddings::load_default() {
        Ok(b) => {
            tracing::info!("CLAP audio encoder loaded — audio-modality Tier 2 enabled");
            Some(Arc::new(b))
        }
        Err(e) => {
            tracing::warn!(error = %e, "CLAP load failed; falling back to text Tier 2");
            None
        }
    }
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
// Null embedding backend — used only on the rollback path when the model
// bundle is missing. The text backend port is `Option`, so no null stand-in
// is needed there: absence is the privacy-preserving default.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct NullEmbeddings;

#[async_trait::async_trait]
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

/// Build the image-modality scan taxonomy. Caller passes the image backend
/// (typically `ctx.image_embeddings`); returns an empty vec when the backend
/// is `None`.
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

/// Build the audio-modality scan taxonomy. Symmetric with
/// [`build_image_scan_candidates`].
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
