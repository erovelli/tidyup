//! `mistralrs`-backed [`TextBackend`](tidyup_core::inference::TextBackend) and
//! [`VisionBackend`](tidyup_core::inference::VisionBackend).
//!
//! # Privacy model
//!
//! This crate is **excluded from default builds** via `optional = true` on the
//! `tidyup-cli` dep. The default release binary has no `mistralrs` / `candle` /
//! `hf-hub` in its dependency graph. Inclusion requires `--features
//! llm-fallback` at build time AND `[inference] llm_fallback = true` in config
//! AND `--llm-fallback` or `TIDYUP_LLM_FALLBACK=1` at runtime. See
//! `CLAUDE.md#privacy-model`.
//!
//! # Models
//!
//! - **Classifier** — text-only. Any mistralrs-compatible chat model id
//!   (default: `Qwen/Qwen3-0.6B` with ISQ Q4K). Loaded eagerly in
//!   [`MistralRsEngine::load`].
//! - **Vision** — `HuggingFaceTB/SmolVLM-256M-Instruct`. Loaded lazily on the
//!   first `caption_image` call so text-only workloads don't pay the vision
//!   memory cost.
//!
//! # Backend selection
//!
//! Accelerator selection (`CUDA` / `Metal` / `CPU`) is driven by the mistralrs
//! features enabled at build time ([`compiled_backends`], [`pick`]). For v0.1
//! forwarding is CPU-by-default; Metal / CUDA builds require explicit
//! `-F tidyup-inference-mistralrs/metal` (or `cuda`) when building
//! `tidyup-cli`.

use std::sync::Arc;

use async_trait::async_trait;
use image::DynamicImage;
use mistralrs::{
    IsqType, MultimodalModelBuilder, RequestBuilder, TextMessageRole, TextMessages,
    TextModelBuilder,
};
use tidyup_core::inference::{
    parse_content_classification, prompts, BackendKind, ContentClassification, GenerationOptions,
    TextBackend, VisionBackend,
};
use tidyup_core::Result;
use tokio::sync::OnceCell;

// ---------------------------------------------------------------------------
// MistralRsEngine — wraps two mistralrs models
// ---------------------------------------------------------------------------

/// `mistralrs`-backed inference engine implementing
/// [`TextBackend`] and [`VisionBackend`].
///
/// - **Classifier** — loaded eagerly at [`Self::load`]. Text-only chat model.
/// - **Vision** — SmolVLM-256M, loaded lazily on first image call via
///   [`OnceCell`] so text-only workloads don't pay the vision memory cost.
///
/// Audio classification uses extractor metadata (lofty) without a dedicated
/// model — see [`TextBackend::classify_audio`].
pub struct MistralRsEngine {
    classifier: mistralrs::Model,
    classifier_id: String,
    vision: OnceCell<mistralrs::Model>,
}

impl std::fmt::Debug for MistralRsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralRsEngine")
            .field("classifier_id", &self.classifier_id)
            .field("vision_loaded", &self.vision.initialized())
            .finish_non_exhaustive()
    }
}

impl MistralRsEngine {
    /// Load the classifier model. This is the only model loaded eagerly.
    ///
    /// Callers MUST await this to completion before spawning other model
    /// loads — concurrent model loads can OOM on 8 GB hosts
    /// (see `CLAUDE.md#operational-rules`).
    pub async fn load(model_id: &str) -> Result<Arc<Self>> {
        tracing::info!("Loading mistralrs classifier model: {model_id}");

        let classifier = TextModelBuilder::new(model_id)
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("mistralrs classifier load failed: {e}"))?;

        tracing::info!("mistralrs classifier model loaded successfully");
        Ok(Arc::new(Self {
            classifier,
            classifier_id: model_id.to_owned(),
            vision: OnceCell::new(),
        }))
    }

    /// Lazily load the vision model (SmolVLM-256M).
    async fn ensure_vision(&self) -> Result<&mistralrs::Model> {
        self.vision
            .get_or_try_init(|| async {
                tracing::info!("Loading vision model: {VISION_MODEL_ID}");
                let model = MultimodalModelBuilder::new(VISION_MODEL_ID)
                    .with_isq(IsqType::Q4K)
                    .with_logging()
                    .build()
                    .await
                    .map_err(|e| anyhow::anyhow!("mistralrs vision load failed: {e}"))?;
                tracing::info!("Vision model loaded successfully");
                Ok(model)
            })
            .await
    }

    async fn run_classification(
        &self,
        system: &str,
        user: String,
    ) -> Result<ContentClassification> {
        let messages = TextMessages::new()
            .add_message(TextMessageRole::System, system)
            .add_message(TextMessageRole::User, &user);

        let request = RequestBuilder::from(messages)
            .enable_thinking(false)
            .set_sampler_max_len(150)
            .set_sampler_temperature(0.1);

        let response = self
            .classifier
            .send_chat_request(request)
            .await
            .map_err(|e| anyhow::anyhow!("mistralrs classification call failed: {e}"))?;
        let content = extract_content(&response)?;
        parse_content_classification(&content)
    }
}

const VISION_MODEL_ID: &str = "HuggingFaceTB/SmolVLM-256M-Instruct";

// ---------------------------------------------------------------------------
// TextBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl TextBackend for MistralRsEngine {
    async fn classify_text(&self, text: &str, filename: &str) -> Result<ContentClassification> {
        let user = format!("Filename: {filename}\n\nContent:\n{text}");
        self.run_classification(prompts::TEXT_CLASSIFY_SYSTEM, user)
            .await
    }

    async fn classify_audio(
        &self,
        filename: &str,
        metadata: &str,
    ) -> Result<ContentClassification> {
        let user = format!("Filename: {filename}\nMetadata:\n{metadata}");
        self.run_classification(prompts::AUDIO_CLASSIFY_SYSTEM, user)
            .await
    }

    async fn classify_video(
        &self,
        filename: &str,
        frame_captions: &[String],
    ) -> Result<ContentClassification> {
        let captions = frame_captions
            .iter()
            .enumerate()
            .map(|(i, c)| format!("Frame {}: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!("Filename: {filename}\n\nFrame descriptions:\n{captions}");
        self.run_classification(prompts::VIDEO_CLASSIFY_SYSTEM, user)
            .await
    }

    async fn classify_image_description(
        &self,
        filename: &str,
        description: &str,
    ) -> Result<ContentClassification> {
        let user = format!("Filename: {filename}\nImage description: {description}");
        self.run_classification(prompts::IMAGE_CLASSIFY_SYSTEM, user)
            .await
    }

    async fn complete(&self, prompt: &str, opts: &GenerationOptions) -> Result<String> {
        let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let request = RequestBuilder::from(messages)
            .enable_thinking(false)
            .set_sampler_max_len(opts.max_tokens as usize)
            .set_sampler_temperature(f64::from(opts.temperature));

        let response = self
            .classifier
            .send_chat_request(request)
            .await
            .map_err(|e| anyhow::anyhow!("mistralrs completion call failed: {e}"))?;
        extract_content(&response)
    }

    fn model_id(&self) -> &str {
        &self.classifier_id
    }
}

// ---------------------------------------------------------------------------
// VisionBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl VisionBackend for MistralRsEngine {
    async fn caption_image(&self, image_bytes: &[u8], _mime: &str) -> Result<String> {
        let image: DynamicImage = image::load_from_memory(image_bytes)
            .map_err(|e| anyhow::anyhow!("decode image bytes: {e}"))?;

        let vision = self.ensure_vision().await?;

        let request = RequestBuilder::new()
            .add_image_message(
                TextMessageRole::User,
                prompts::IMAGE_CAPTION_PROMPT,
                vec![image],
            )
            .set_sampler_max_len(60)
            .set_sampler_temperature(0.1);

        let response = vision
            .send_chat_request(request)
            .await
            .map_err(|e| anyhow::anyhow!("mistralrs caption call failed: {e}"))?;
        extract_content(&response)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn model_id(&self) -> &str {
        VISION_MODEL_ID
    }
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

fn extract_content(response: &mistralrs::ChatCompletionResponse) -> Result<String> {
    response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .ok_or_else(|| anyhow::anyhow!("No content in mistralrs response"))
}

// ---------------------------------------------------------------------------
// Runtime backend selection — docorg's `inference/select.rs`, mistralrs scope
// ---------------------------------------------------------------------------

/// Backends that this build of `tidyup-inference-mistralrs` was compiled with.
///
/// Compile-time only — does not verify that the underlying driver / runtime is
/// actually present. Always includes [`BackendKind::Cpu`]. Excludes `Vulkan`
/// and `Rocm` — those live in the future `tidyup-inference-llamacpp` crate.
#[must_use]
pub fn compiled_backends() -> Vec<BackendKind> {
    let mut out = Vec::new();
    if cfg!(feature = "cuda") {
        out.push(BackendKind::Cuda);
    }
    if cfg!(feature = "metal") {
        out.push(BackendKind::Metal);
    }
    out.push(BackendKind::Cpu);
    out
}

/// Cheap runtime probe per backend kind — deliberately side-effect-free.
///
/// We want a quick "looks like this works" signal, not a full init. On a
/// probe-positive backend we still defer to [`MistralRsEngine::load`] to
/// surface any real failure.
#[must_use]
pub const fn probe(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Cpu => true,
        BackendKind::Metal => cfg!(all(target_os = "macos", feature = "metal")),
        BackendKind::Cuda => cfg!(feature = "cuda"),
        // Vulkan / ROCm are not served by mistralrs — reserved for the
        // future llamacpp backend.
        BackendKind::Vulkan | BackendKind::Rocm => false,
    }
}

/// Backends usable on this host right now. Intersection of
/// [`compiled_backends`] with [`probe`].
#[must_use]
pub fn detect_available() -> Vec<BackendKind> {
    compiled_backends()
        .into_iter()
        .filter(|k| probe(*k))
        .collect()
}

/// Pick the best usable backend. An explicit `preferred` hint wins iff it
/// survives [`probe`]; otherwise we fall through the auto-detect priority
/// order: CUDA → Metal → CPU.
#[must_use]
pub fn pick(preferred: Option<BackendKind>) -> BackendKind {
    if let Some(p) = preferred {
        if probe(p) {
            return p;
        }
        tracing::warn!(
            "Preferred backend {} not available; auto-detecting",
            p.label(),
        );
    }
    let available = detect_available();
    for &candidate in &[BackendKind::Cuda, BackendKind::Metal, BackendKind::Cpu] {
        if available.contains(&candidate) {
            return candidate;
        }
    }
    BackendKind::Cpu
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_compiled() {
        assert!(compiled_backends().contains(&BackendKind::Cpu));
    }

    #[test]
    fn cpu_is_always_available() {
        assert!(probe(BackendKind::Cpu));
        assert!(detect_available().contains(&BackendKind::Cpu));
    }

    #[test]
    fn vulkan_and_rocm_are_not_mistralrs_backends() {
        assert!(!probe(BackendKind::Vulkan));
        assert!(!probe(BackendKind::Rocm));
    }

    #[test]
    fn pick_falls_back_to_cpu_when_override_unreachable() {
        // ROCm is never a mistralrs backend, so pick() must fall back.
        let chosen = pick(Some(BackendKind::Rocm));
        assert!(detect_available().contains(&chosen));
    }

    #[test]
    fn pick_honors_cpu_override() {
        assert_eq!(pick(Some(BackendKind::Cpu)), BackendKind::Cpu);
    }
}
