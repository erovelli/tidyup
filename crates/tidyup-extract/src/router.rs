//! Extractor dispatch.
//!
//! The pipeline owns a `Vec<Arc<dyn ContentExtractor>>` registered at startup.
//! For each file it calls [`pick`], which walks the list and returns the first
//! extractor whose `supports(path, mime)` returns true. Order matters:
//! more-specific extractors should be registered first (e.g. `PdfExtractor`
//! before a generic fallback).

use std::path::Path;
use std::sync::Arc;

use tidyup_core::extractor::ContentExtractor;

/// Pick the first registered extractor that claims `path`.
///
/// `mime` should be supplied by the caller (typically from
/// [`crate::mime::detect`] or the file index) to avoid re-sniffing per
/// extractor.
pub fn pick<'a>(
    extractors: &'a [Arc<dyn ContentExtractor>],
    path: &Path,
    mime: Option<&str>,
) -> Option<&'a Arc<dyn ContentExtractor>> {
    extractors.iter().find(|e| e.supports(path, mime))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use tidyup_core::extractor::ExtractedContent;
    use tidyup_core::Result;

    #[derive(Debug)]
    struct Fixed {
        name: &'static str,
        matches: bool,
    }

    #[async_trait]
    impl ContentExtractor for Fixed {
        fn supports(&self, _path: &Path, _mime: Option<&str>) -> bool {
            self.matches
        }
        async fn extract(&self, _path: &Path) -> Result<ExtractedContent> {
            Ok(ExtractedContent {
                text: Some(self.name.to_string()),
                mime: "text/plain".to_string(),
                metadata: serde_json::Value::Null,
            })
        }
    }

    #[test]
    fn picks_first_matching_in_registration_order() {
        let a: Arc<dyn ContentExtractor> = Arc::new(Fixed {
            name: "a",
            matches: false,
        });
        let b: Arc<dyn ContentExtractor> = Arc::new(Fixed {
            name: "b",
            matches: true,
        });
        let c: Arc<dyn ContentExtractor> = Arc::new(Fixed {
            name: "c",
            matches: true,
        });
        let list = vec![a, b, c];

        let pick = pick(&list, Path::new("/nope"), None).unwrap();
        // We don't have Eq on the trait objects; round-trip through extract to identify.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let extracted = rt.block_on(pick.extract(Path::new("/nope"))).unwrap();
        assert_eq!(extracted.text.as_deref(), Some("b"));
    }

    #[test]
    fn none_when_no_extractor_claims_path() {
        let a: Arc<dyn ContentExtractor> = Arc::new(Fixed {
            name: "a",
            matches: false,
        });
        let list = vec![a];
        assert!(pick(&list, Path::new("/nope"), None).is_none());
    }
}
