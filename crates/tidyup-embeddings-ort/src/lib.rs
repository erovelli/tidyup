//! ONNX-runtime [`EmbeddingBackend`](tidyup_core::inference::EmbeddingBackend)
//! — the default Tier 2 classifier.
//!
//! # Default model
//!
//! `bge-small-en-v1.5` (384-dim, 133 MB fp32 or ~35 MB Q8). English-only in
//! v0.1; multilingual encoders are a post-v0.1 roadmap item.
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

pub mod classifier;
pub mod embeddings;
pub mod install;
pub mod paths;
pub mod taxonomy;
pub mod util;

pub use classifier::{EmbeddingClassification, EmbeddingClassifier};
pub use embeddings::{Config, OrtEmbeddings};
pub use install::{installation_instructions, verify_default_model, ArtifactSpec};
pub use taxonomy::{default_taxonomy, TaxonomyEntry};
pub use util::{cosine_similarity, extract_year, l2_normalize};
