//! PDF extractor.
//!
//! Uses `pdf-extract` to pull selectable text out of PDF documents. Scanned /
//! image-only PDFs (OCR required) produce a tiny amount of extracted text —
//! those surface as `text: None` with `metadata.scanned = true` so the
//! classifier can skip them or route to a different tier rather than embedding
//! noise.
//!
//! `pdf-extract` is synchronous and CPU-heavy; the work runs under
//! `tokio::task::spawn_blocking` so a slow document doesn't stall the pipeline
//! runtime.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::Result;

/// Cutoff for treating a document as scanned/image-only.
///
/// Documents below this many non-whitespace characters after extraction are
/// treated as scanned/image-only. Chosen to match docorg's historical cutoff
/// and large enough to filter title-page-only PDFs without missing genuine
/// short notes.
pub const MIN_EXTRACTED_CHARS: usize = 100;

/// Extractor for PDF documents.
///
/// `text` in the output is `Some` when extraction yielded more than
/// [`MIN_EXTRACTED_CHARS`] of non-whitespace characters; otherwise `None`
/// with `metadata.scanned = true`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PdfExtractor;

impl PdfExtractor {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContentExtractor for PdfExtractor {
    fn supports(&self, path: &Path, mime: Option<&str>) -> bool {
        if matches!(mime, Some("application/pdf")) {
            return true;
        }
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent> {
        let owned: PathBuf = path.to_path_buf();
        let result = tokio::task::spawn_blocking(move || pdf_extract::extract_text(&owned)).await?;

        let (text_opt, scanned, error) = match result {
            Ok(raw) => {
                let non_ws = raw.chars().filter(|c| !c.is_whitespace()).count();
                if non_ws >= MIN_EXTRACTED_CHARS {
                    (Some(raw), false, None)
                } else {
                    tracing::debug!(
                        "pdf extract: {} chars insufficient ({} < {MIN_EXTRACTED_CHARS}), treating as scanned",
                        path.display(),
                        non_ws,
                    );
                    (None, true, None)
                }
            }
            Err(e) => {
                tracing::warn!("pdf extract failed for {}: {e}", path.display());
                (None, false, Some(e.to_string()))
            }
        };

        let metadata = error.map_or_else(
            || serde_json::json!({ "scanned": scanned }),
            |err| serde_json::json!({ "scanned": scanned, "error": err }),
        );

        Ok(ExtractedContent {
            text: text_opt,
            mime: "application/pdf".to_string(),
            metadata,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn supports_by_mime() {
        let e = PdfExtractor::new();
        assert!(e.supports(Path::new("unknown"), Some("application/pdf")));
        assert!(!e.supports(Path::new("unknown"), Some("text/plain")));
    }

    #[test]
    fn supports_by_extension_case_insensitive() {
        let e = PdfExtractor::new();
        assert!(e.supports(Path::new("paper.pdf"), None));
        assert!(e.supports(Path::new("PAPER.PDF"), None));
        assert!(!e.supports(Path::new("paper.txt"), None));
    }

    #[tokio::test]
    async fn missing_file_reports_error_in_metadata() {
        let e = PdfExtractor::new();
        let out = e.extract(Path::new("/definitely/does/not/exist.pdf")).await;
        // The extract call itself returns Ok because failures are captured in
        // metadata rather than the Result — the pipeline wants best-effort
        // signal, not hard errors.
        let out = out.unwrap();
        assert_eq!(out.mime, "application/pdf");
        assert!(out.text.is_none());
        assert!(out.metadata.get("error").is_some());
    }
}
