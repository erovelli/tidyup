//! ONNX-runtime embedding backend for BGE-style BERT encoders.
//!
//! # Model contract
//!
//! Encoder must expose:
//! - Inputs: `input_ids`, `attention_mask`, `token_type_ids` — all `i64`
//!   tensors of shape `[batch, seq_len]`.
//! - Output: `last_hidden_state` — `f32` tensor of shape `[batch, seq_len, dim]`.
//!
//! Pooling is **CLS-token** (index 0), followed by **L2 normalization**.
//! This matches `BAAI/bge-small-en-v1.5` and compatible Qdrant-quantized variants.
//!
//! # Threading
//!
//! [`ort::session::Session`] is `Send + Sync`. The wrapper dispatches each
//! embed call through [`tokio::task::spawn_blocking`] because `Session::run`
//! is a blocking C++ call. Callers may share a single [`OrtEmbeddings`]
//! across tasks via [`std::sync::Arc`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use async_trait::async_trait;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::{
    EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use tidyup_core::inference::EmbeddingBackend;

use crate::util::l2_normalize;

/// Default maximum sequence length for BGE-small. The model's positional
/// embeddings are capped at 512.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Default output dimensionality for `bge-small-en-v1.5`.
pub const DEFAULT_EMBEDDING_DIMS: usize = 384;

/// Default model identifier, used as a cache key.
pub const DEFAULT_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";

/// Configuration for [`OrtEmbeddings::load`].
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Path to the tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Stable model identifier for cache invalidation and logs.
    pub model_id: String,
    /// Output embedding dimensionality.
    pub dims: usize,
    /// Maximum sequence length. Inputs are truncated to this length.
    pub max_seq_len: usize,
    /// Number of threads for intra-op parallelism. `None` = ORT default.
    pub intra_threads: Option<usize>,
}

impl Config {
    /// Defaults pointing at the conventional platform cache paths for
    /// `bge-small-en-v1.5`.
    ///
    /// # Errors
    /// Returns [`None`] if the platform cache directory is unavailable.
    #[must_use]
    pub fn default_bge_small() -> Option<Self> {
        let model_path = crate::paths::default_model_path()?;
        let tokenizer_path = crate::paths::default_tokenizer_path()?;
        Some(Self {
            model_path,
            tokenizer_path,
            model_id: DEFAULT_MODEL_ID.to_string(),
            dims: DEFAULT_EMBEDDING_DIMS,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            intra_threads: None,
        })
    }
}

/// ONNX embedding backend. Holds an initialized session + tokenizer and
/// produces L2-normalized CLS-pooled embeddings.
///
/// [`Session::run`] takes `&mut self` in `ort` 2.x — we serialize access
/// through a [`Mutex`]. Locks are only held inside `spawn_blocking`, so the
/// async executor never stalls on them.
pub struct OrtEmbeddings {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    dims: usize,
    model_id: String,
}

impl std::fmt::Debug for OrtEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtEmbeddings")
            .field("dims", &self.dims)
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl OrtEmbeddings {
    /// Load the encoder and tokenizer from disk.
    ///
    /// # Errors
    /// - Model or tokenizer file missing.
    /// - ONNX Runtime init or session build failure.
    /// - Tokenizer JSON parse failure.
    pub fn load(config: Config) -> Result<Self> {
        ensure_ort_initialized();

        if !config.model_path.exists() {
            return Err(anyhow::anyhow!(
                "Embedding model not found at {}. Download `{}` and place it at that path, \
                 or set TIDYUP_MODEL_CACHE to a directory containing it.",
                config.model_path.display(),
                config.model_id,
            ));
        }
        if !config.tokenizer_path.exists() {
            return Err(anyhow::anyhow!(
                "Tokenizer not found at {}. Download tokenizer.json for `{}` and place it there.",
                config.tokenizer_path.display(),
                config.model_id,
            ));
        }

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("ORT session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ORT optimization level: {e}"))?;
        if let Some(threads) = config.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|e| anyhow::anyhow!("ORT intra threads: {e}"))?;
        }
        let session = builder.commit_from_file(&config.model_path).map_err(|e| {
            anyhow::anyhow!("loading ONNX model at {}: {e}", config.model_path.display())
        })?;

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        configure_tokenizer(&mut tokenizer, config.max_seq_len);

        tracing::info!(
            model = %config.model_id,
            dims = config.dims,
            max_seq_len = config.max_seq_len,
            "embedding model loaded",
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            dims: config.dims,
            model_id: config.model_id,
        })
    }

    /// Load the default `bge-small-en-v1.5` model from the platform cache.
    ///
    /// # Errors
    /// Returns an error if the platform cache directory is unavailable or
    /// the model files are missing.
    pub fn load_default() -> Result<Self> {
        let config = Config::default_bge_small().ok_or_else(|| {
            anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
        })?;
        Self::load(config)
    }

    /// Synchronous embed — intended for use inside a `spawn_blocking` context
    /// or test code. Production async callers should go through
    /// [`EmbeddingBackend::embed_text`] / [`EmbeddingBackend::embed_texts`].
    ///
    /// Always returns L2-normalized vectors.
    ///
    /// # Errors
    /// Tokenization failure, ORT inference error, or tensor extraction failure.
    // The session mutex guard must stay live while we iterate `SessionOutputs` —
    // the extracted array view borrows from the session. Dropping the guard
    // earlier would invalidate those borrows.
    #[allow(clippy::significant_drop_tightening)]
    pub fn embed_sync(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let inputs: Vec<EncodeInput<'_>> = texts
            .iter()
            .map(|t| EncodeInput::Single((*t).into()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode_batch: {e}"))?;

        let batch = encodings.len();
        let seq_len = encodings
            .iter()
            .map(tokenizers::Encoding::len)
            .max()
            .unwrap_or(0);
        if seq_len == 0 {
            return Ok(vec![vec![0.0; self.dims]; batch]);
        }

        let mut ids = Vec::with_capacity(batch * seq_len);
        let mut mask = Vec::with_capacity(batch * seq_len);
        let mut tids = Vec::with_capacity(batch * seq_len);
        for enc in &encodings {
            push_padded(&mut ids, enc.get_ids(), seq_len, 0);
            push_padded(&mut mask, enc.get_attention_mask(), seq_len, 0);
            push_padded(&mut tids, enc.get_type_ids(), seq_len, 0);
        }

        let ids_arr =
            Array2::from_shape_vec((batch, seq_len), ids).context("build input_ids tensor")?;
        let mask_arr = Array2::from_shape_vec((batch, seq_len), mask)
            .context("build attention_mask tensor")?;
        let tids_arr = Array2::from_shape_vec((batch, seq_len), tids)
            .context("build token_type_ids tensor")?;

        let ids_tensor =
            Tensor::from_array(ids_arr).map_err(|e| anyhow::anyhow!("input_ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array(mask_arr)
            .map_err(|e| anyhow::anyhow!("attention_mask tensor: {e}"))?;
        let tids_tensor = Tensor::from_array(tids_arr)
            .map_err(|e| anyhow::anyhow!("token_type_ids tensor: {e}"))?;

        let out = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("ORT session mutex poisoned"))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor,
                    "token_type_ids" => tids_tensor,
                ])
                .map_err(|e| anyhow::anyhow!("ORT inference: {e}"))?;

            let hidden = outputs
                .get("last_hidden_state")
                .ok_or_else(|| anyhow::anyhow!("ORT output missing last_hidden_state"))?;
            let view = hidden
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("extract last_hidden_state: {e}"))?;
            let hidden_shape = view.shape();
            if hidden_shape.len() != 3 || hidden_shape[0] != batch || hidden_shape[2] != self.dims {
                return Err(anyhow::anyhow!(
                    "unexpected last_hidden_state shape: {hidden_shape:?} (expected [{batch}, _, {}])",
                    self.dims,
                ));
            }

            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                let mut row = Vec::with_capacity(self.dims);
                for d in 0..self.dims {
                    row.push(view[[b, 0, d]]);
                }
                l2_normalize(&mut row);
                out.push(row);
            }
            out
        };
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingBackend for OrtEmbeddings {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let session = self.session.clone();
        let tokenizer = self.tokenizer.clone();
        let dims = self.dims;
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            embed_sync_impl(&session, &tokenizer, &[&text], dims)
                .map(|mut v| v.pop().unwrap_or_default())
        })
        .await
        .context("embed_text join")?
    }

    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.session.clone();
        let tokenizer = self.tokenizer.clone();
        let dims = self.dims;
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            embed_sync_impl(&session, &tokenizer, &refs, dims)
        })
        .await
        .context("embed_texts join")?
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// Shared between the async path (needs a free function to move the Arcs) and
// the sync test path. Keep in step with `OrtEmbeddings::embed_sync`.
#[allow(clippy::significant_drop_tightening)]
fn embed_sync_impl(
    session: &Mutex<Session>,
    tokenizer: &Tokenizer,
    texts: &[&str],
    dims: usize,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let inputs: Vec<EncodeInput<'_>> = texts
        .iter()
        .map(|t| EncodeInput::Single((*t).into()))
        .collect();
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode_batch: {e}"))?;

    let batch = encodings.len();
    let seq_len = encodings
        .iter()
        .map(tokenizers::Encoding::len)
        .max()
        .unwrap_or(0);
    if seq_len == 0 {
        return Ok(vec![vec![0.0; dims]; batch]);
    }

    let mut ids = Vec::with_capacity(batch * seq_len);
    let mut mask = Vec::with_capacity(batch * seq_len);
    let mut tids = Vec::with_capacity(batch * seq_len);
    for enc in &encodings {
        push_padded(&mut ids, enc.get_ids(), seq_len, 0);
        push_padded(&mut mask, enc.get_attention_mask(), seq_len, 0);
        push_padded(&mut tids, enc.get_type_ids(), seq_len, 0);
    }

    let ids_arr =
        Array2::from_shape_vec((batch, seq_len), ids).context("build input_ids tensor")?;
    let mask_arr =
        Array2::from_shape_vec((batch, seq_len), mask).context("build attention_mask tensor")?;
    let tids_arr =
        Array2::from_shape_vec((batch, seq_len), tids).context("build token_type_ids tensor")?;

    let ids_tensor =
        Tensor::from_array(ids_arr).map_err(|e| anyhow::anyhow!("input_ids tensor: {e}"))?;
    let mask_tensor =
        Tensor::from_array(mask_arr).map_err(|e| anyhow::anyhow!("attention_mask tensor: {e}"))?;
    let tids_tensor =
        Tensor::from_array(tids_arr).map_err(|e| anyhow::anyhow!("token_type_ids tensor: {e}"))?;

    let out = {
        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("ORT session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => tids_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("ORT inference: {e}"))?;

        let hidden = outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow::anyhow!("ORT output missing last_hidden_state"))?;
        let view = hidden
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract last_hidden_state: {e}"))?;
        let hidden_shape = view.shape();
        if hidden_shape.len() != 3 || hidden_shape[0] != batch || hidden_shape[2] != dims {
            return Err(anyhow::anyhow!(
                "unexpected last_hidden_state shape: {hidden_shape:?} (expected [{batch}, _, {dims}])",
            ));
        }

        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut row = Vec::with_capacity(dims);
            for d in 0..dims {
                row.push(view[[b, 0, d]]);
            }
            l2_normalize(&mut row);
            out.push(row);
        }
        out
    };
    Ok(out)
}

fn configure_tokenizer(tokenizer: &mut Tokenizer, max_seq_len: usize) {
    let _ = tokenizer.with_truncation(Some(TruncationParams {
        direction: TruncationDirection::Right,
        max_length: max_seq_len,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    }));
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }));
}

fn push_padded(out: &mut Vec<i64>, src: &[u32], seq_len: usize, pad: i64) {
    for &v in src.iter().take(seq_len) {
        out.push(i64::from(v));
    }
    for _ in src.len().min(seq_len)..seq_len {
        out.push(pad);
    }
}

/// Ensure `ort::init()` runs exactly once per process. `commit` returns
/// `true` on first install, `false` if the global env options were already
/// set — either outcome is fine.
fn ensure_ort_initialized() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let _ = ort::init().with_name("tidyup").commit();
    });
}

/// Quick existence check without loading the model.
#[must_use]
pub fn model_available(model_path: &Path, tokenizer_path: &Path) -> bool {
    model_path.exists() && tokenizer_path.exists()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn config_default_bge_small_shape() {
        // Cannot assume a platform cache exists in all CI environments,
        // but we can at least sanity-check the shape when it is present.
        if let Some(cfg) = Config::default_bge_small() {
            assert_eq!(cfg.model_id, DEFAULT_MODEL_ID);
            assert_eq!(cfg.dims, DEFAULT_EMBEDDING_DIMS);
            assert_eq!(cfg.max_seq_len, DEFAULT_MAX_SEQ_LEN);
            assert!(cfg.model_path.ends_with("model.onnx"));
            assert!(cfg.tokenizer_path.ends_with("tokenizer.json"));
        }
    }

    #[test]
    fn load_missing_model_errors_clearly() {
        let cfg = Config {
            model_path: PathBuf::from("/definitely/does/not/exist/model.onnx"),
            tokenizer_path: PathBuf::from("/definitely/does/not/exist/tokenizer.json"),
            model_id: "test".into(),
            dims: 384,
            max_seq_len: 512,
            intra_threads: None,
        };
        let err = OrtEmbeddings::load(cfg).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("Embedding model not found"), "got: {s}");
    }

    #[test]
    fn push_padded_truncates_and_pads() {
        let mut out = Vec::new();
        push_padded(&mut out, &[1, 2, 3, 4, 5], 3, 0);
        assert_eq!(out, vec![1, 2, 3]);

        let mut out = Vec::new();
        push_padded(&mut out, &[1, 2], 5, 0);
        assert_eq!(out, vec![1, 2, 0, 0, 0]);
    }
}
