//! Content extractor ports — per-file-type text/metadata extraction.
//!
//! Implementations live in `tidyup-extract`, gated behind cargo features for
//! heavier deps (pdf-extract, calamine, etc.).

use std::path::Path;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    pub text: Option<String>,
    pub mime: String,
    pub metadata: serde_json::Value,
}

/// Produces text + metadata from a file. Implementations should gracefully
/// return `text: None` for files they understand structurally but cannot
/// transcribe (e.g., binary blobs, encrypted PDFs).
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    fn supports(&self, path: &Path, mime: Option<&str>) -> bool;
    async fn extract(&self, path: &Path) -> Result<ExtractedContent>;
}
