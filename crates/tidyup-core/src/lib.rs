//! Port traits for the tidyup hexagonal architecture.
//!
//! # Two families of ports
//!
//! **Frontend ports** (implemented by CLI/UI, consumed by [`tidyup-app`] services):
//! - [`ProgressReporter`] — streaming progress updates
//! - [`ReviewHandler`]    — collect user decisions on proposals
//! - [`ConfigProvider`]   — supply runtime configuration
//!
//! **Backend ports** (implemented by impl crates, consumed by the pipeline):
//! - [`inference`] — LLM, vision, embedding backends
//! - [`storage`]   — file index, change log, backup store
//! - [`extractor`] — content extractors per file type
//!
//! The key property: application services in [`tidyup-app`] depend *only* on these
//! traits. CLI and UI both drive the same services — they differ only in the
//! frontend-port implementations they supply.

pub mod extractor;
pub mod frontend;
pub mod inference;
pub mod storage;

pub use frontend::{ConfigProvider, ProgressReporter, ReviewHandler, ReviewOutcome};

/// Result alias used across port traits.
pub type Result<T> = std::result::Result<T, anyhow::Error>;
