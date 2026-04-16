//! Inference backend ports — LLM, vision, and embedding models.
//!
//! Implementations live in `tidyup-inference-*` crates (mistralrs, remote HTTP, etc.).
//! Selection happens at runtime via a registry — no cargo-feature rebuild required
//! to switch providers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Generic text-generation backend (used for LLM classification, rationales, naming).
#[async_trait]
pub trait TextBackend: Send + Sync {
    async fn complete(&self, prompt: &str, opts: &GenerationOptions) -> Result<String>;

    /// Identifier of the underlying model — surfaced to users in diffs and logs.
    fn model_id(&self) -> &str;
}

/// Vision-capable backend (image captioning, OCR hints).
#[async_trait]
pub trait VisionBackend: Send + Sync {
    async fn caption(&self, image_bytes: &[u8], mime: &str) -> Result<String>;
    fn model_id(&self) -> &str;
}

/// Embedding backend for semantic similarity (folder profiling, classification).
#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_id(&self) -> &str;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Modalities a backend can serve. A backend may advertise any subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Vision,
    Embeddings,
}

/// Runtime capability descriptor. A registry uses this to pick the best backend
/// available on the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Accelerator {
    Cpu,
    Metal,
    Cuda,
    Rocm,
    Vulkan,
    Remote,
}
