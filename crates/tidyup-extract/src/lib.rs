//! Content extractors — one impl per supported format, gated by cargo features.
//!
//! The pipeline builds a `Vec<Arc<dyn ContentExtractor>>` at startup and dispatches
//! by calling `supports(path, mime)` in order.

// TODO: PlainTextExtractor, PdfExtractor, ExcelExtractor, ImageExtractor, AudioExtractor.
