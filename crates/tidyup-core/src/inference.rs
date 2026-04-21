//! Inference backend ports — LLM, vision, embedding backends, plus shared
//! classification types.
//!
//! # Layering
//!
//! - [`Classifier`] is the high-level port consumed by `tidyup-pipeline`. It
//!   produces a [`ClassificationResult`](tidyup_domain::migration::ClassificationResult)
//!   against a set of folder candidates.
//! - [`TextBackend`], [`VisionBackend`], [`EmbeddingBackend`] are low-level
//!   model adapters. Classifier implementations compose them.
//!
//! Backends advertise themselves at runtime via [`BackendCapabilities`] so a
//! registry can pick the best available one for a host — no cargo-feature
//! rebuild required to switch providers of an *included* backend family.
//! Feature flags gate whether a backend family is *compiled in* (see
//! `ARCHITECTURE.md` + `CLASSIFICATION.md`).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Content classification emitted by a [`TextBackend`].
///
/// All text backends are required to emit this exact shape so the rest of the
/// pipeline stays backend-agnostic. JSON parsing is tolerant via
/// [`parse_content_classification`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentClassification {
    pub category: String,
    pub tags: Vec<String>,
    pub summary: String,
    #[serde(default)]
    pub suggested_name: Option<String>,
}

/// Modality of a source file, derived from MIME + extension by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileModality {
    Text,
    Image,
    Video,
    Audio,
    /// File that should be skipped (e.g. `.DS_Store`, lock files).
    Skip,
}

/// Backend family — used for logs, diagnostics, and to gate expensive tiers.
///
/// The migration pipeline skips Tier 3 LLM fallback on [`BackendKind::Cpu`] by
/// default (CPU LLM inference is 25–50 s/file and not viable interactively).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cuda,
    Metal,
    Vulkan,
    Rocm,
    Cpu,
}

impl BackendKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Metal => "Metal",
            Self::Vulkan => "Vulkan",
            Self::Rocm => "ROCm",
            Self::Cpu => "CPU",
        }
    }

    /// Returns `true` if the backend uses GPU acceleration.
    #[must_use]
    pub const fn is_accelerated(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cuda" => Some(Self::Cuda),
            "metal" => Some(Self::Metal),
            "vulkan" => Some(Self::Vulkan),
            "rocm" | "hip" => Some(Self::Rocm),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// Backend capability modalities — used by a runtime registry to pick the best
/// backend for a request. A backend may advertise any subset.
///
/// `Embeddings` covers text-only embedding backends (e.g. `bge-small`).
/// `ImageEmbeddings` and `AudioEmbeddings` cover cross-modal contrastive
/// encoders (`SigLIP`, `CLAP`) — distinct because their latent spaces are
/// not interchangeable with each other or with text-only embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Vision,
    Embeddings,
    ImageEmbeddings,
    AudioEmbeddings,
}

/// Runtime capability descriptor. A registry uses this to pick the best backend
/// available on the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub name: String,
    pub modalities: Vec<Modality>,
    pub requires_network: bool,
    pub accelerator: Accelerator,
}

impl BackendCapabilities {
    #[must_use]
    pub fn supports(&self, modality: Modality) -> bool {
        self.modalities.contains(&modality)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Accelerator {
    Cpu,
    Metal,
    Cuda,
    Rocm,
    Vulkan,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub stop: Vec<String>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.2,
            stop: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend traits
// ---------------------------------------------------------------------------

/// Generative text classifier. Used for Tier 3 LLM fallback and by any
/// higher-level classifier that wants a natural-language opinion.
///
/// v0.1 exposes four classification entry points covering the four modalities.
/// Image / video paths go through a [`VisionBackend`] first to obtain captions
/// or per-frame descriptions.
#[async_trait]
pub trait TextBackend: Send + Sync {
    async fn classify_text(&self, text: &str, filename: &str) -> Result<ContentClassification>;

    async fn classify_audio(&self, filename: &str, metadata: &str)
        -> Result<ContentClassification>;

    async fn classify_video(
        &self,
        filename: &str,
        frame_captions: &[String],
    ) -> Result<ContentClassification>;

    async fn classify_image_description(
        &self,
        filename: &str,
        description: &str,
    ) -> Result<ContentClassification>;

    /// Raw generation — used for templated rename synthesis and ad-hoc
    /// prompts. Default implementations may share a chat model with the
    /// classify methods.
    async fn complete(&self, prompt: &str, opts: &GenerationOptions) -> Result<String>;

    /// Stable identifier for this backend's text model. Surfaced in logs and
    /// proposal reasoning.
    fn model_id(&self) -> &str;
}

/// Vision-language captioner. Returns a one-sentence description of an image;
/// the result is fed back into a [`TextBackend`] for category routing.
#[async_trait]
pub trait VisionBackend: Send + Sync {
    async fn caption_image(&self, image_bytes: &[u8], mime: &str) -> Result<String>;

    /// Whether this backend has a vision model loaded. A pipeline uses this
    /// to skip image captioning entirely on backends without a VLM.
    fn is_available(&self) -> bool {
        true
    }

    fn model_id(&self) -> &str;
}

/// Embedding generator for the Tier 2 cosine-similarity classifier and folder
/// profiling.
///
/// Embeddings MUST be L2-normalized — consumers treat them as unit vectors and
/// compute dot-products as cosine similarity directly.
#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Output dimensionality. `bge-small-en-v1.5` = 384.
    fn dimensions(&self) -> usize;

    /// Stable identifier. Used as a cache key for taxonomy and folder-profile
    /// caches; changing this invalidates them.
    fn model_id(&self) -> &str;
}

/// Cross-modal image embedding backend — produces L2-normalized vectors in a
/// shared image-text latent space (e.g. `SigLIP`).
///
/// The text and image methods produce vectors in the **same** latent space, so
/// `cosine(embed_image(bytes), embed_text("a photograph of a cat"))` is a
/// meaningful similarity score. This is the v0.1 mechanism for image-modality
/// classification: pre-embed taxonomy descriptions with `embed_text`, embed
/// the candidate image with `embed_image`, and rank by cosine similarity.
///
/// **Latent-space isolation.** These vectors are NOT comparable to those from
/// [`EmbeddingBackend`] — pairing them would compute meaningless cosines. The
/// pipeline keeps the backends separate to make this impossible to mis-wire.
#[async_trait]
pub trait ImageEmbeddingBackend: Send + Sync {
    /// Embed raw image bytes. `mime` is informational — implementations may
    /// use it to short-circuit decoding for unsupported formats.
    async fn embed_image(&self, image_bytes: &[u8], mime: &str) -> Result<Vec<f32>>;

    /// Embed a text query in the same latent space (`SigLIP` text tower).
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Batched text embedding for taxonomy precomputation.
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Output dimensionality (`SigLIP-base` = 768).
    fn dimensions(&self) -> usize;

    /// Stable identifier. Cache key for image-taxonomy embeddings.
    fn model_id(&self) -> &str;
}

/// Cross-modal audio embedding backend — produces L2-normalized vectors in a
/// shared audio-text latent space (e.g. `CLAP`).
///
/// Same shape as [`ImageEmbeddingBackend`] but for audio: pre-embed taxonomy
/// descriptions, embed the candidate audio, rank by cosine. Latent-space
/// isolation applies — these vectors are not comparable to text-only
/// [`EmbeddingBackend`] or [`ImageEmbeddingBackend`] vectors.
#[async_trait]
pub trait AudioEmbeddingBackend: Send + Sync {
    /// Embed raw audio bytes. Implementations are responsible for resampling,
    /// mel-spectrogram computation, and length truncation.
    async fn embed_audio(&self, audio_bytes: &[u8], mime: &str) -> Result<Vec<f32>>;

    /// Embed a text query in the same latent space (`CLAP` text tower).
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Batched text embedding for taxonomy precomputation.
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Output dimensionality (`CLAP-htsat` = 512).
    fn dimensions(&self) -> usize;

    /// Stable identifier.
    fn model_id(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Classifier port (consumed by tidyup-pipeline)
// ---------------------------------------------------------------------------

/// The classification port.
///
/// Implementations compose one or more backends (embeddings, optional LLM) to
/// turn extracted content into a [`ClassificationResult`](tidyup_domain::migration::ClassificationResult).
///
/// The port is intentionally simple — `text` + `filename` — so the same trait
/// serves both scan mode (classify against a fixed taxonomy) and migration
/// mode (classify against target folder profiles). Which candidates the
/// classifier scores against is an implementation detail.
#[async_trait]
pub trait Classifier: Send + Sync {
    /// Classify a file. `text` is the extracted body; `filename` is the basename.
    ///
    /// Implementations are expected to short-circuit on high-confidence
    /// matches and surface lower-confidence cases via
    /// [`ClassificationResult::needs_review`](tidyup_domain::migration::ClassificationResult).
    async fn classify(
        &self,
        text: &str,
        filename: &str,
    ) -> Result<tidyup_domain::migration::ClassificationResult>;

    /// Stable identifier used in proposal reasoning and logs.
    fn classifier_id(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Shared system prompts — used by any TextBackend that drives a chat model
// ---------------------------------------------------------------------------

pub mod prompts {
    pub const TEXT_CLASSIFY_SYSTEM: &str = r#"You are a file classifier. Given the text content of a file, respond with a JSON object containing:
- "category": one of [document, spreadsheet, code, config, data, presentation, ebook, correspondence, report, notes, legal, financial, medical, creative, academic, travel, real_estate, automotive, pet, recipe, fitness, government, resume, warranty, family, other]
- "tags": array of 3-5 descriptive tags
- "summary": one sentence summary of the file content
- "suggested_name": a clear, descriptive filename (keeping the original extension). Use lowercase_with_underscores. If relevant dates appear in the content (year, quarter, month, or specific date), incorporate them as a prefix in YYYY, YYYY_QN, YYYY-MM, or YYYY-MM-DD format. Examples: "2024_q1_earnings_report.pdf", "2025-03-15_lease_agreement.pdf", "tax_return_1040_2024.pdf". Only suggest a name if the current filename is vague, generic, or does not reflect the content.

Respond ONLY with the JSON object. No explanation."#;

    pub const IMAGE_CAPTION_PROMPT: &str =
        "Describe this image in one sentence. Focus on: what the image contains, the setting, and any visible text.";

    pub const IMAGE_CLASSIFY_SYSTEM: &str = r#"You are a file classifier. Given a description of an image file, respond with a JSON object containing:
- "category": one of [photo, screenshot, diagram, document_scan, artwork, meme, icon, logo, map, chart, medical_image, satellite, other]
- "tags": array of 3-5 descriptive tags
- "summary": one sentence summary
- "suggested_name": a clear, descriptive filename (keeping the original extension). Use lowercase_with_underscores. If a date is apparent (from EXIF, visible text, or context), use it as a prefix in YYYY-MM-DD format. Only suggest a name if the current filename is vague or generic (e.g. IMG_1234.jpg, scan.png).

Respond ONLY with the JSON object. No explanation."#;

    pub const AUDIO_CLASSIFY_SYSTEM: &str = r#"You are a file classifier. Given metadata and a transcript of an audio file, respond with a JSON object containing:
- "category": one of [music, podcast, voice_memo, audiobook, lecture, interview, sound_effect, ringtone, ambient, phone_call, other]
- "tags": array of 3-5 descriptive tags
- "summary": one sentence summary
- "suggested_name": a clear, descriptive filename (keeping the original extension). Use lowercase_with_underscores. Include artist/title if known. If a date is apparent, prefix with YYYY-MM-DD. Only suggest a name if the current filename is vague or generic.

Respond ONLY with the JSON object. No explanation."#;

    pub const VIDEO_CLASSIFY_SYSTEM: &str = r#"You are a file classifier. Given descriptions of video frames and an optional audio transcript, respond with a JSON object containing:
- "category": one of [clip, tutorial, presentation_recording, screen_recording, vlog, music_video, surveillance, animation, documentary, conversation, other]
- "tags": array of 3-5 descriptive tags
- "summary": one sentence summary
- "suggested_name": a clear, descriptive filename (keeping the original extension). Use lowercase_with_underscores. If a date is apparent, prefix with YYYY-MM-DD. Only suggest a name if the current filename is vague or generic.

Respond ONLY with the JSON object. No explanation."#;
}

// ---------------------------------------------------------------------------
// Tolerant JSON parser — strips <think> blocks, markdown fences, surrounding prose
// ---------------------------------------------------------------------------

/// Parse model output as a [`ContentClassification`].
///
/// Tolerant of `<think>...</think>` reasoning blocks, markdown code fences,
/// and surrounding prose. Implementations of [`TextBackend`] should pass raw
/// model responses through this rather than reimplementing parsing.
pub fn parse_content_classification(content: &str) -> Result<ContentClassification> {
    let stripped = content.find("</think>").map_or_else(
        || content.trim(),
        |pos| content[pos + "</think>".len()..].trim(),
    );

    if let Ok(c) = serde_json::from_str::<ContentClassification>(stripped) {
        return Ok(c);
    }

    let cleaned = stripped
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if let Ok(c) = serde_json::from_str::<ContentClassification>(cleaned) {
        return Ok(c);
    }

    if let (Some(start), Some(end)) = (cleaned.find('{'), cleaned.rfind('}')) {
        if end > start {
            if let Ok(c) = serde_json::from_str::<ContentClassification>(&cleaned[start..=end]) {
                return Ok(c);
            }
        }
    }

    Err(anyhow::anyhow!(
        "Failed to parse classification JSON\nRaw: {content}"
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_content_classification_plain() {
        let json =
            r#"{"category":"document","tags":["tax","form","1040"],"summary":"A tax form."}"#;
        let c = parse_content_classification(json).unwrap();
        assert_eq!(c.category, "document");
        assert_eq!(c.tags.len(), 3);
        assert!(c.suggested_name.is_none());
    }

    #[test]
    fn parse_content_classification_with_suggested_name() {
        let json = r#"{"category":"financial","tags":["tax","1040"],"summary":"Tax return.","suggested_name":"tax_return_1040_2024.pdf"}"#;
        let c = parse_content_classification(json).unwrap();
        assert_eq!(c.category, "financial");
        assert_eq!(
            c.suggested_name.as_deref(),
            Some("tax_return_1040_2024.pdf"),
        );
    }

    #[test]
    fn parse_content_classification_with_think() {
        let json = r#"<think>reasoning goes here</think>{"category":"photo","tags":["cat"],"summary":"A cat."}"#;
        let c = parse_content_classification(json).unwrap();
        assert_eq!(c.category, "photo");
    }

    #[test]
    fn parse_content_classification_fenced() {
        let json =
            "```json\n{\"category\":\"music\",\"tags\":[\"pop\"],\"summary\":\"A pop song.\"}\n```";
        let c = parse_content_classification(json).unwrap();
        assert_eq!(c.category, "music");
    }

    #[test]
    fn parse_content_classification_with_prose() {
        let json = "Here's the answer: {\"category\":\"document\",\"tags\":[\"a\",\"b\",\"c\"],\"summary\":\"x\"} done.";
        let c = parse_content_classification(json).unwrap();
        assert_eq!(c.category, "document");
    }

    #[test]
    fn backend_kind_parse_roundtrip() {
        for k in [
            BackendKind::Cuda,
            BackendKind::Metal,
            BackendKind::Vulkan,
            BackendKind::Rocm,
            BackendKind::Cpu,
        ] {
            assert_eq!(BackendKind::parse(k.label()), Some(k));
        }
        assert_eq!(BackendKind::parse("HIP"), Some(BackendKind::Rocm));
        assert_eq!(BackendKind::parse("nonsense"), None);
    }

    #[test]
    fn backend_kind_accelerated() {
        assert!(!BackendKind::Cpu.is_accelerated());
        assert!(BackendKind::Cuda.is_accelerated());
        assert!(BackendKind::Metal.is_accelerated());
        assert!(BackendKind::Vulkan.is_accelerated());
        assert!(BackendKind::Rocm.is_accelerated());
    }

    #[test]
    fn backend_capabilities_supports() {
        let caps = BackendCapabilities {
            name: "test".into(),
            modalities: vec![Modality::Text, Modality::Embeddings],
            requires_network: false,
            accelerator: Accelerator::Cpu,
        };
        assert!(caps.supports(Modality::Text));
        assert!(caps.supports(Modality::Embeddings));
        assert!(!caps.supports(Modality::Vision));
        assert!(!caps.supports(Modality::ImageEmbeddings));
        assert!(!caps.supports(Modality::AudioEmbeddings));
    }

    #[test]
    fn backend_capabilities_supports_multimodal() {
        let caps = BackendCapabilities {
            name: "siglip".into(),
            modalities: vec![Modality::ImageEmbeddings],
            requires_network: false,
            accelerator: Accelerator::Cpu,
        };
        assert!(caps.supports(Modality::ImageEmbeddings));
        assert!(!caps.supports(Modality::Embeddings));
    }
}
