// `CLAP` and `Symphonia` are proper nouns repeated throughout; doc_markdown
// here is noise.
#![allow(clippy::doc_markdown)]

//! CLAP cross-modal audio/text embedding backend — Phase 7 multimodal Tier 2
//! audio classifier.
//!
//! # Model contract
//!
//! Two ONNX files (audio and text towers) export a shared embedding space
//! (typically 512-dim for `clap-htsat-unfused`). The text tower consumes
//! `input_ids` like SigLIP's text tower. The audio tower's input shape
//! depends on which export the user installed:
//!
//! - **Fused exports** (preprocessing-baked-in) accept raw waveform
//!   `input_features` of shape `[1, samples]` at 48 kHz mono.
//! - **Unfused exports** expect a precomputed mel spectrogram. Tidyup v0.1
//!   ships the fused-export path because mel-spec preprocessing in pure Rust
//!   is a substantial extra surface and the fused exports are widely
//!   available.
//!
//! Both audio and text outputs are L2-normalized post-pooling.
//!
//! # Audio preprocessing (fused export)
//!
//! - Decode container via `symphonia` (WAV/FLAC/MP3/AAC/OGG/M4A).
//! - Downmix to mono by averaging channels.
//! - Resample to 48 000 Hz (CLAP's training rate).
//! - Trim or zero-pad to [`AUDIO_LENGTH_SAMPLES`] samples (10 seconds).
//! - Pass as `input_features` `[1, AUDIO_LENGTH_SAMPLES]` `f32` in `[-1, 1]`.
//!
//! # Threading
//!
//! Same shape as [`crate::OrtEmbeddings`] — sessions wrapped in `Mutex`,
//! inference inside `spawn_blocking`.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tokenizers::{
    EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use tidyup_core::inference::AudioEmbeddingBackend;

use crate::util::l2_normalize;

/// Default model identifier (CLAP HTSAT-unfused via Xenova HF mirror).
pub const DEFAULT_MODEL_ID: &str = "laion/clap-htsat-unfused";

/// Default output dimensionality.
pub const DEFAULT_EMBEDDING_DIMS: usize = 512;

/// Sample rate the audio tower expects (Hz).
pub const TARGET_SAMPLE_RATE: u32 = 48_000;

/// Audio clip length in samples — 10 seconds at [`TARGET_SAMPLE_RATE`].
pub const AUDIO_LENGTH_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize) * 10;

/// Default text-tower max sequence length. CLAP base caps at 77 tokens
/// (CLIP-style).
pub const DEFAULT_MAX_SEQ_LEN: usize = 77;

/// Configuration for [`ClapEmbeddings::load`].
#[derive(Debug, Clone)]
pub struct Config {
    pub audio_path: PathBuf,
    pub text_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub model_id: String,
    pub dims: usize,
    pub max_seq_len: usize,
    pub intra_threads: Option<usize>,
}

impl Config {
    #[must_use]
    pub fn default_clap() -> Option<Self> {
        Some(Self {
            audio_path: crate::paths::clap_audio_path()?,
            text_path: crate::paths::clap_text_path()?,
            tokenizer_path: crate::paths::clap_tokenizer_path()?,
            model_id: DEFAULT_MODEL_ID.to_string(),
            dims: DEFAULT_EMBEDDING_DIMS,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            intra_threads: None,
        })
    }
}

/// CLAP cross-modal audio/text encoder.
pub struct ClapEmbeddings {
    audio: Arc<Mutex<Session>>,
    text: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    dims: usize,
    model_id: String,
}

impl std::fmt::Debug for ClapEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClapEmbeddings")
            .field("dims", &self.dims)
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl ClapEmbeddings {
    /// Load both towers and the tokenizer.
    ///
    /// # Errors
    /// Missing files, ORT init, tokenizer parse failure.
    pub fn load(config: Config) -> Result<Self> {
        crate::embeddings::ensure_ort_initialized();

        if !config.audio_path.exists() {
            return Err(anyhow::anyhow!(
                "CLAP audio model not found at {}",
                config.audio_path.display(),
            ));
        }
        if !config.text_path.exists() {
            return Err(anyhow::anyhow!(
                "CLAP text model not found at {}",
                config.text_path.display(),
            ));
        }
        if !config.tokenizer_path.exists() {
            return Err(anyhow::anyhow!(
                "CLAP tokenizer not found at {}",
                config.tokenizer_path.display(),
            ));
        }

        let audio = build_session(&config.audio_path, config.intra_threads)?;
        let text = build_session(&config.text_path, config.intra_threads)?;

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load CLAP tokenizer: {e}"))?;
        configure_tokenizer(&mut tokenizer, config.max_seq_len);

        tracing::info!(
            model = %config.model_id,
            dims = config.dims,
            sample_rate = TARGET_SAMPLE_RATE,
            "CLAP encoder loaded",
        );

        Ok(Self {
            audio: Arc::new(Mutex::new(audio)),
            text: Arc::new(Mutex::new(text)),
            tokenizer: Arc::new(tokenizer),
            dims: config.dims,
            model_id: config.model_id,
        })
    }

    /// Load the default CLAP bundle from the platform cache.
    ///
    /// # Errors
    /// Returns an error if the cache directory is unavailable or the bundle
    /// is missing.
    pub fn load_default() -> Result<Self> {
        let config = Config::default_clap().ok_or_else(|| {
            anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
        })?;
        Self::load(config)
    }
}

#[async_trait]
impl AudioEmbeddingBackend for ClapEmbeddings {
    async fn embed_audio(&self, audio_bytes: &[u8], mime: &str) -> Result<Vec<f32>> {
        let session = self.audio.clone();
        let dims = self.dims;
        let bytes = audio_bytes.to_vec();
        let mime = mime.to_string();
        tokio::task::spawn_blocking(move || embed_audio_sync(&session, &bytes, &mime, dims))
            .await
            .context("embed_audio join")?
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

/// Decode → resample to mono 48 kHz → trim/pad → run audio tower → L2-normalize.
#[allow(clippy::significant_drop_tightening)]
fn embed_audio_sync(
    session: &Mutex<Session>,
    audio_bytes: &[u8],
    mime: &str,
    dims: usize,
) -> Result<Vec<f32>> {
    let samples = decode_to_mono_48k(audio_bytes, mime)?;
    let prepared = pad_or_truncate(&samples, AUDIO_LENGTH_SAMPLES);

    let arr = Array2::from_shape_vec((1, AUDIO_LENGTH_SAMPLES), prepared)
        .context("build input_features tensor shape")?;
    let tensor =
        Tensor::from_array(arr).map_err(|e| anyhow::anyhow!("input_features tensor: {e}"))?;

    let vec = {
        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("CLAP audio session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs!["input_features" => tensor])
            .map_err(|e| anyhow::anyhow!("CLAP audio inference: {e}"))?;
        let mut batch = extract_pooled_batch(&outputs, 1, dims, "audio")?;
        let mut v = batch.pop().unwrap_or_default();
        l2_normalize(&mut v);
        v
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
        .map_err(|e| anyhow::anyhow!("CLAP tokenizer encode_batch: {e}"))?;

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
    let mut mask: Vec<i64> = Vec::with_capacity(batch * seq_len);
    for enc in &encodings {
        let raw = enc.get_ids();
        let raw_mask = enc.get_attention_mask();
        for &v in raw.iter().take(seq_len) {
            ids.push(i64::from(v));
        }
        let pad_ids = seq_len.saturating_sub(raw.len().min(seq_len));
        ids.extend(std::iter::repeat_n(0_i64, pad_ids));
        for &v in raw_mask.iter().take(seq_len) {
            mask.push(i64::from(v));
        }
        let pad_mask = seq_len.saturating_sub(raw_mask.len().min(seq_len));
        mask.extend(std::iter::repeat_n(0_i64, pad_mask));
    }

    let ids_arr =
        Array2::from_shape_vec((batch, seq_len), ids).context("build input_ids tensor")?;
    let mask_arr =
        Array2::from_shape_vec((batch, seq_len), mask).context("build attention_mask tensor")?;
    let ids_tensor =
        Tensor::from_array(ids_arr).map_err(|e| anyhow::anyhow!("input_ids tensor: {e}"))?;
    let mask_tensor =
        Tensor::from_array(mask_arr).map_err(|e| anyhow::anyhow!("attention_mask tensor: {e}"))?;

    let mut out: Vec<Vec<f32>> = {
        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("CLAP text session mutex poisoned"))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("CLAP text inference: {e}"))?;
        extract_pooled_batch(&outputs, batch, dims, "text")?
    };

    for v in &mut out {
        l2_normalize(v);
    }
    Ok(out)
}

fn extract_pooled_batch(
    outputs: &ort::session::SessionOutputs<'_>,
    batch: usize,
    dims: usize,
    side: &str,
) -> Result<Vec<Vec<f32>>> {
    let preferred = ["audio_embeds", "text_embeds", "pooler_output", "embeddings"];
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
                "CLAP {side} output `{name}` has unexpected shape {shape:?} \
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
                "CLAP {side} last_hidden_state shape {shape:?} \
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

    // Drop the leading `[B, mel_bins, time]` shape if the audio tower emits
    // a 3-D mel-feature passthrough. Mean-pool over time, then over mel bins
    // is meaningless; this branch only exists so a misconfigured ONNX surfaces
    // a clear error rather than a panic.
    if let Some(out) = outputs.get("input_features") {
        let view = out
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract input_features: {e}"))?;
        return Err(anyhow::anyhow!(
            "CLAP {side} returned only `input_features` (shape {:?}) — the loaded \
             ONNX is the audio-preprocessor passthrough, not the encoder. Re-export \
             the fused model or install the encoder ONNX.",
            view.shape(),
        ));
    }

    Err(anyhow::anyhow!(
        "CLAP {side} ONNX output missing pooled vector \
         (looked for audio_embeds/text_embeds/pooler_output/embeddings/last_hidden_state)"
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

/// Trim or zero-pad samples to the target length.
fn pad_or_truncate(samples: &[f32], target: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(target);
    if samples.len() >= target {
        out.extend_from_slice(&samples[..target]);
    } else {
        out.extend_from_slice(samples);
        out.resize(target, 0.0);
    }
    out
}

/// Decode arbitrary container/codec into a mono 48 kHz `f32` waveform.
///
/// Uses Symphonia's auto-probe so format detection works from bytes alone.
/// Resampling is the simple linear-interpolation kind — adequate for an
/// embedding model that already trained on diverse rates, not a substitute
/// for `rubato` or `samplerate` for production audio.
#[allow(clippy::too_many_lines)]
fn decode_to_mono_48k(audio_bytes: &[u8], mime: &str) -> Result<Vec<f32>> {
    let cursor = Cursor::new(audio_bytes.to_vec());
    let stream = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = mime_to_extension(mime) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probing audio format")?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("no default audio track in input"))?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let source_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("audio has no sample rate"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("building audio decoder")?;

    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(anyhow::anyhow!("decode packet: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!("decode frame: {e}")),
        };
        downmix_to_mono_into(&buf, &mut mono);

        if mono.len() >= AUDIO_LENGTH_SAMPLES.saturating_mul(48) / 48 {
            // We have enough at the source rate; bail to keep memory bounded.
            // (We use the source rate here; resampling reduces or grows it
            // below.)
            if mono.len() >= source_rate as usize * 30 {
                break;
            }
        }
    }

    if source_rate == TARGET_SAMPLE_RATE {
        Ok(mono)
    } else {
        Ok(linear_resample(&mono, source_rate, TARGET_SAMPLE_RATE))
    }
}

fn downmix_to_mono_into(buf: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().max(1);
            for f in 0..frames {
                let mut sum = 0.0_f32;
                for c in 0..chans {
                    sum += b.chan(c)[f];
                }
                #[allow(clippy::cast_precision_loss)]
                out.push(sum / chans as f32);
            }
        }
        AudioBufferRef::S16(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().max(1);
            for f in 0..frames {
                let mut sum = 0.0_f32;
                for c in 0..chans {
                    sum += f32::from(b.chan(c)[f]) / f32::from(i16::MAX);
                }
                #[allow(clippy::cast_precision_loss)]
                out.push(sum / chans as f32);
            }
        }
        AudioBufferRef::S32(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().max(1);
            #[allow(clippy::cast_precision_loss)]
            let denom = i32::MAX as f32;
            for f in 0..frames {
                let mut sum = 0.0_f32;
                for c in 0..chans {
                    #[allow(clippy::cast_precision_loss)]
                    let v = b.chan(c)[f] as f32;
                    sum += v / denom;
                }
                #[allow(clippy::cast_precision_loss)]
                out.push(sum / chans as f32);
            }
        }
        AudioBufferRef::U8(b) => {
            let frames = b.frames();
            let chans = b.spec().channels.count().max(1);
            for f in 0..frames {
                let mut sum = 0.0_f32;
                for c in 0..chans {
                    let v = (f32::from(b.chan(c)[f]) - 128.0_f32) / 128.0_f32;
                    sum += v;
                }
                #[allow(clippy::cast_precision_loss)]
                out.push(sum / chans as f32);
            }
        }
        // Other sample formats (S24, U16, U24, U32, F64) collapse to silence
        // for now — the mainstream codecs we care about emit S16/S32/F32.
        _ => {}
    }
}

/// Linear interpolation resampling. Adequate for embedding-model input where
/// the model has already absorbed sample-rate diversity in training; not a
/// substitute for windowed-sinc when audio fidelity matters.
fn linear_resample(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = f64::from(dst_rate) / f64::from(src_rate);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        #[allow(clippy::cast_precision_loss)]
        let src = (i as f64) / ratio;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(input.len() - 1);
        #[allow(clippy::cast_possible_truncation)]
        let frac = (src - src.floor()) as f32;
        let v = input[lo].mul_add(1.0 - frac, input[hi] * frac);
        out.push(v);
    }
    out
}

const fn mime_to_extension(mime: &str) -> Option<&'static str> {
    // Symphonia's `Hint` wants extension hints (without the dot). MIME →
    // ext mapping covers the common audio types; if unknown we let Symphonia
    // sniff the container format.
    match mime.as_bytes() {
        b"audio/wav" | b"audio/wave" | b"audio/x-wav" => Some("wav"),
        b"audio/flac" | b"audio/x-flac" => Some("flac"),
        b"audio/mpeg" => Some("mp3"),
        b"audio/aac" => Some("aac"),
        b"audio/ogg" | b"audio/vorbis" => Some("ogg"),
        b"audio/opus" => Some("opus"),
        b"audio/mp4" => Some("m4a"),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn pad_or_truncate_pads_short() {
        let v = pad_or_truncate(&[1.0, 2.0], 5);
        assert_eq!(v, vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_or_truncate_truncates_long() {
        let v = pad_or_truncate(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn linear_resample_no_op_when_equal_rate() {
        let v = linear_resample(&[0.0, 0.5, 1.0], 48_000, 48_000);
        assert_eq!(v, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn linear_resample_doubles_length_at_2x() {
        let v = linear_resample(&[0.0, 1.0, 0.0], 24_000, 48_000);
        // Length doubles; values lie on the linear interpolant.
        assert_eq!(v.len(), 6);
        assert_eq!(v[0], 0.0);
        assert!((v[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn config_default_clap_shape() {
        if let Some(cfg) = Config::default_clap() {
            assert_eq!(cfg.model_id, DEFAULT_MODEL_ID);
            assert_eq!(cfg.dims, DEFAULT_EMBEDDING_DIMS);
            assert!(cfg.audio_path.ends_with("audio_model.onnx"));
            assert!(cfg.text_path.ends_with("text_model.onnx"));
        }
    }

    #[test]
    fn load_missing_files_errors_clearly() {
        let cfg = Config {
            audio_path: PathBuf::from("/no/such/audio_model.onnx"),
            text_path: PathBuf::from("/no/such/text_model.onnx"),
            tokenizer_path: PathBuf::from("/no/such/tokenizer.json"),
            model_id: "test".into(),
            dims: 512,
            max_seq_len: 77,
            intra_threads: None,
        };
        let err = ClapEmbeddings::load(cfg).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("CLAP audio model not found"), "got: {s}");
    }

    #[test]
    fn mime_extension_lookup_known_formats() {
        assert_eq!(mime_to_extension("audio/wav"), Some("wav"));
        assert_eq!(mime_to_extension("audio/flac"), Some("flac"));
        assert_eq!(mime_to_extension("audio/mpeg"), Some("mp3"));
        assert_eq!(mime_to_extension("audio/x-unknown"), None);
    }
}
