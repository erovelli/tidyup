//! ONNX-runtime
//! [`EmbeddingBackend`](tidyup_core::inference::EmbeddingBackend),
//! [`ImageEmbeddingBackend`](tidyup_core::inference::ImageEmbeddingBackend),
//! and [`AudioEmbeddingBackend`](tidyup_core::inference::AudioEmbeddingBackend)
//! implementations.
//!
//! # Default text model
//!
//! `bge-small-en-v1.5` (384-dim, 133 MB fp32 or ~35 MB Q8). English-only in
//! v0.1; multilingual encoders are a post-v0.1 roadmap item.
//!
//! # Phase 7 multimodal models
//!
//! - [`siglip`] — SigLIP-base cross-modal image/text encoder. Optional:
//!   loaded only when the bundle exists under
//!   `<cache>/tidyup/models/siglip-base-patch16-224/`.
//! - [`clap`]  — CLAP audio/text encoder. Optional: loaded only when the
//!   bundle exists under `<cache>/tidyup/models/clap-htsat-unfused/`.
//!
//! # Shape
//!
//! - [`OrtEmbeddings`] loads an ONNX encoder + `WordPiece` tokenizer and
//!   produces L2-normalized sentence embeddings via CLS-token pooling. It
//!   implements [`EmbeddingBackend`].
//! - [`EmbeddingClassifier`] pairs an [`OrtEmbeddings`] with a taxonomy of
//!   target folders. `classify()` returns the highest-cosine-similarity
//!   folder with a raw score in `[-1.0, 1.0]` (unit-normalized).
//! - [`taxonomy`] defines the default hierarchical taxonomy plus a disk
//!   cache keyed by BLAKE3 hash of descriptions + model id.
//! - [`util`] provides cosine similarity, L2 normalization, and year
//!   extraction helpers.

pub mod audio_taxonomy;
pub mod clap;
pub mod classifier;
pub mod embeddings;
pub mod image_taxonomy;
pub mod install;
pub mod paths;
pub mod siglip;
pub mod taxonomy;
pub mod util;

pub use audio_taxonomy::default_audio_taxonomy;
pub use clap::ClapEmbeddings;
pub use classifier::{EmbeddingClassification, EmbeddingClassifier};
pub use embeddings::{Config, OrtEmbeddings};
pub use image_taxonomy::default_image_taxonomy;
pub use install::{
    clap_installation_instructions, installation_instructions, siglip_installation_instructions,
    verify_clap_model, verify_default_model, verify_siglip_model, ArtifactSpec,
};
pub use siglip::SigLipEmbeddings;
pub use taxonomy::{default_taxonomy, TaxonomyEntry};
pub use util::{cosine_similarity, extract_year, l2_normalize};
