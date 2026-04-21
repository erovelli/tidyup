// `SigLIP` and other proper nouns appear frequently in this module; treating
// them as missing-backticks is noise.
#![allow(clippy::doc_markdown)]

//! SigLIP cross-modal image/text embedding backend — Phase 7 multimodal Tier 2
//! image classifier.
//!
//! # Model contract
//!
//! Two ONNX files (vision and text towers) must export a shared embedding
//! space (typically 768-dim for `siglip-base-patch16-224`). The vision tower
//! consumes `pixel_values` and the text tower consumes `input_ids`. Output
//! tensor names are matched tolerantly (`image_embeds` / `text_embeds` /
//! `pooler_output` / `last_hidden_state` are all probed in order).
//!
//! Vectors are L2-normalized post-pooling so consumers can dot-product them as
//! cosine similarity.
//!
//! # Preprocessing
//!
//! - **Image**: decode via `image` crate, resize to 224×224 with bilinear
//!   filtering, convert to f32 in `[0, 1]`, normalize per-channel with
//!   `mean = std = 0.5` (the SigLIP processor convention), reshape to NCHW.
//! - **Text**: tokenize, pad/truncate to 64 tokens, pass `input_ids` only
//!   (the text tower consumes the embedding lookup directly).
//!
//! # Threading
//!
//! Same shape as [`crate::OrtEmbeddings`]: each ONNX session is wrapped in a
//! `Mutex` and inference runs inside `spawn_blocking`. Vision and text
//! sessions are independent, so the two towers can run concurrently across
//! tasks.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use ndarray::{Array2, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::{
    EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use tidyup_core::inference::ImageEmbeddingBackend;

use crate::util::l2_normalize;

/// Default model identifier (SigLIP base, patch 16, 224×224).
pub const DEFAULT_MODEL_ID: &str = "google/siglip-base-patch16-224";

/// Default output dimensionality.
pub const DEFAULT_EMBEDDING_DIMS: usize = 768;

/// Image side length the vision tower expects.
pub const IMAGE_SIZE: u32 = 224;

/// Per-channel normalization mean (SigLIP convention).
const NORM_MEAN: [f32; 3] = [0.5, 0.5, 0.5];

/// Per-channel normalization standard deviation.
const NORM_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Default text-tower max sequence length. SigLIP base caps at 64.
pub const DEFAULT_MAX_SEQ_LEN: usize = 64;

/// Configuration for [`SigLipEmbeddings::load`].
#[derive(Debug, Clone)]
pub struct Config {
    pub vision_path: PathBuf,
    pub text_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub model_id: String,
    pub dims: usize,
    pub max_seq_len: usize,
    pub intra_threads: Option<usize>,
}

impl Config {
    /// Defaults pointing at the conventional platform cache paths.
    #[must_use]
    pub fn default_siglip_base() -> Option<Self> {
        Some(Self {
            vision_path: crate::paths::siglip_vision_path()?,
            text_path: crate::paths::siglip_text_path()?,
            tokenizer_path: crate::paths::siglip_tokenizer_path()?,
            model_id: DEFAULT_MODEL_ID.to_string(),
            dims: DEFAULT_EMBEDDING_DIMS,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            intra_threads: None,
        })
    }
}

/// SigLIP cross-modal image/text encoder.
pub struct SigLipEmbeddings {
    vision: Arc<Mutex<Session>>,
    text: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    dims: usize,
    model_id: String,
}

impl std::fmt::Debug for SigLipEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigLipEmbeddings")
            .field("dims", &self.dims)
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl SigLipEmbeddings {
    /// Load both towers and the tokenizer.
    ///
    /// # Errors
    /// Missing files, ORT init, or tokenizer parse failure.
    pub fn load(config: Config) -> Result<Self> {
        crate::embeddings::ensure_ort_initialized();

        if !config.vision_path.exists() {
            return Err(anyhow::anyhow!(
                "SigLIP vision model not found at {}. {}",
                config.vision_path.display(),
                "See `verify_siglip_model` for installation instructions.",
            ));
        }
        if !config.text_path.exists() {
            return Err(anyhow::anyhow!(
                "SigLIP text model not found at {}",
                config.text_path.display(),
            ));
        }
        if !config.tokenizer_path.exists() {
            return Err(anyhow::anyhow!(
                "SigLIP tokenizer not found at {}",
                config.tokenizer_path.display(),
            ));
        }

        let vision = build_session(&config.vision_path, config.intra_threads)?;
        let text = build_session(&config.text_path, config.intra_threads)?;

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load SigLIP tokenizer: {e}"))?;
        configure_tokenizer(&mut tokenizer, config.max_seq_len);

        tracing::info!(
            model = %config.model_id,
            dims = config.dims,
            image_size = IMAGE_SIZE,
            "SigLIP encoder loaded",
        );

        Ok(Self {
            vision: Arc::new(Mutex::new(vision)),
            text: Arc::new(Mutex::new(text)),
            tokenizer: Arc::new(tokenizer),
            dims: config.dims,
            model_id: config.model_id,
        })
    }

    /// Load the default `siglip-base-patch16-224` bundle from the platform
    /// cache.
    ///
    /// # Errors
    /// Returns an error if the cache directory is unavailable or the bundle
    /// is missing.
    pub fn load_default() -> Result<Self> {
        let config = Config::default_siglip_base().ok_or_else(|| {
            anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
        })?;
        Self::load(config)
    }
}

#[async_trait]
impl ImageEmbeddingBackend for SigLipEmbeddings {
    async fn embed_image(&self, image_bytes: &[u8], mime: &str) -> Result<Vec<f32>> {
        let session = self.vision.clone();
        let dims = self.dims;
        let bytes = image_bytes.to_vec();
        let mime = mime.to_string();
        tokio::task::spawn_blocking(move || embed_image_sync(&session, &bytes, &mime, dims))
            .await
            .context("embed_image join")?
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let session = self.text.clone();
        let tokenizer = self.tokenizer.clone();
        let dims = self.dims;
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            embed_text_sync(&session, &tokenizer, &[&text], dims)
                .map(|mut v| v.pop().unwrap_or_default())
        })
        .await
        .context("embed_text join")?
    }

    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.text.clone();
        let tokenizer = self.tokenizer.clone();
        let dims = self.dims;
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            embed_text_sync(&session, &tokenizer, &refs, dims)
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

/// Decode → resize → normalize → run vision tower → L2-normalize.
#[allow(clippy::significant_drop_tightening)]
fn embed_image_sync(
    session: &Mutex<Session>,
    image_bytes: &[u8],
    _mime: &str,
    dims: usize,
) -> Result<Vec<f32>> {
    let img = image::load_from_memory(image_bytes).context("decoding image bytes")?;
    let resized = img
        .resize_exact(
            IMAGE_SIZE,
            IMAGE_SIZE,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();

    let mut buf = vec![0.0_f32; 3 * (IMAGE_SIZE as usize) * (IMAGE_SIZE as usize)];
    let h = IMAGE_SIZE as usize;
    let w = IMAGE_SIZE as usize;
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let pixel = resized.get_pixel(x, y);
            let yi = y as usize;
            let xi = x as usize;
            for c in 0..3 {
                let v = f32::from(pixel.0[c]) / 255.0_f32;
                let v = (v - NORM_MEAN[c]) / NORM_STD[c];
                buf[c * h * w + yi * w + xi] = v;
            }
        }
    }

    let pixel_values =
        Array4::from_shape_vec((1, 3, h, w), buf).context("build pixel_values tensor shape")?;
    let pixel_tensor = Tensor::from_array(pixel_values)
        .map_err(|e| anyhow::anyhow!("pixel_values tensor: {e}"))?;

    let vec = {
        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("SigLIP vision session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs!["pixel_values" => pixel_tensor])
            .map_err(|e| anyhow::anyhow!("SigLIP vision inference: {e}"))?;
        extract_pooled(&outputs, dims, "vision")?
    };
    Ok(vec)
}

#[allow(clippy::significant_drop_tightening)]
fn embed_text_sync(
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
        .map_err(|e| anyhow::anyhow!("SigLIP tokenizer encode_batch: {e}"))?;

    let batch = encodings.len();
    let seq_len = encodings
        .iter()
        .map(tokenizers::Encoding::len)
        .max()
        .unwrap_or(0);
    if seq_len == 0 {
        return Ok(vec![vec![0.0; dims]; batch]);
    }

    let mut ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
    for enc in &encodings {
        let raw = enc.get_ids();
        for &v in raw.iter().take(seq_len) {
            ids.push(i64::from(v));
        }
        let pad_count = seq_len.saturating_sub(raw.len().min(seq_len));
        ids.extend(std::iter::repeat_n(0_i64, pad_count));
    }

    let ids_arr =
        Array2::from_shape_vec((batch, seq_len), ids).context("build input_ids tensor")?;
    let ids_tensor =
        Tensor::from_array(ids_arr).map_err(|e| anyhow::anyhow!("input_ids tensor: {e}"))?;

    let mut out: Vec<Vec<f32>> = {
        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("SigLIP text session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs!["input_ids" => ids_tensor])
            .map_err(|e| anyhow::anyhow!("SigLIP text inference: {e}"))?;
        extract_pooled_batch(&outputs, batch, dims, "text")?
    };

    for v in &mut out {
        l2_normalize(v);
    }
    Ok(out)
}

/// Tolerant pooled-vector extractor for the single-image path.
fn extract_pooled(
    outputs: &ort::session::SessionOutputs<'_>,
    dims: usize,
    side: &str,
) -> Result<Vec<f32>> {
    let mut batch = extract_pooled_batch(outputs, 1, dims, side)?;
    let mut v = batch.pop().unwrap_or_default();
    l2_normalize(&mut v);
    Ok(v)
}

/// Tolerant pooled-vector extractor for the batched-text path. Tries the
/// common SigLIP output names in order. Falls back to mean-pooling
/// `last_hidden_state` if no pooled tensor is exposed.
fn extract_pooled_batch(
    outputs: &ort::session::SessionOutputs<'_>,
    batch: usize,
    dims: usize,
    side: &str,
) -> Result<Vec<Vec<f32>>> {
    let preferred = ["image_embeds", "text_embeds", "pooler_output", "embeddings"];
    for name in preferred {
        if let Some(out) = outputs.get(name) {
            let view = out
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("extract {name}: {e}"))?;
            let shape = view.shape();
            if shape.len() == 2 && shape[0] == batch && shape[1] == dims {
                let mut rows = Vec::with_capacity(batch);
                for b in 0..batch {
                    let mut row = Vec::with_capacity(dims);
                    for d in 0..dims {
                        row.push(view[[b, d]]);
                    }
                    rows.push(row);
                }
                return Ok(rows);
            }
            return Err(anyhow::anyhow!(
                "SigLIP {side} output `{name}` has unexpected shape {shape:?} \
                 (expected [{batch}, {dims}])"
            ));
        }
    }

    if let Some(out) = outputs.get("last_hidden_state") {
        let view = out
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract last_hidden_state: {e}"))?;
        let shape = view.shape();
        if shape.len() != 3 || shape[0] != batch || shape[2] != dims {
            return Err(anyhow::anyhow!(
                "SigLIP {side} last_hidden_state shape {shape:?} \
                 (expected [{batch}, _, {dims}])"
            ));
        }
        let seq = shape[1];
        let mut rows = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut row = vec![0.0_f32; dims];
            for s in 0..seq {
                for d in 0..dims {
                    row[d] += view[[b, s, d]];
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let denom = seq as f32;
            for x in &mut row {
                *x /= denom.max(1.0);
            }
            rows.push(row);
        }
        return Ok(rows);
    }

    Err(anyhow::anyhow!(
        "SigLIP {side} ONNX output missing pooled vector \
         (looked for image_embeds/text_embeds/pooler_output/embeddings/last_hidden_state)"
    ))
}

fn build_session(path: &std::path::Path, intra_threads: Option<usize>) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("ORT session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("ORT optimization level: {e}"))?;
    if let Some(threads) = intra_threads {
        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("ORT intra threads: {e}"))?;
    }
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("loading ONNX model at {}: {e}", path.display()))
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn config_default_siglip_base_shape() {
        if let Some(cfg) = Config::default_siglip_base() {
            assert_eq!(cfg.model_id, DEFAULT_MODEL_ID);
            assert_eq!(cfg.dims, DEFAULT_EMBEDDING_DIMS);
            assert_eq!(cfg.max_seq_len, DEFAULT_MAX_SEQ_LEN);
            assert!(cfg.vision_path.ends_with("vision_model.onnx"));
            assert!(cfg.text_path.ends_with("text_model.onnx"));
        }
    }

    #[test]
    fn load_missing_files_errors_clearly() {
        let cfg = Config {
            vision_path: PathBuf::from("/no/such/vision_model.onnx"),
            text_path: PathBuf::from("/no/such/text_model.onnx"),
            tokenizer_path: PathBuf::from("/no/such/tokenizer.json"),
            model_id: "test".into(),
            dims: 768,
            max_seq_len: 64,
            intra_threads: None,
        };
        let err = SigLipEmbeddings::load(cfg).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("SigLIP vision model not found"), "got: {s}");
    }
}
