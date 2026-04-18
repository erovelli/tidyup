//! Content extractors — one impl per supported format, gated by cargo features.
//!
//! The pipeline builds a `Vec<Arc<dyn ContentExtractor>>` at startup and
//! dispatches by calling [`router::pick`], which returns the first extractor
//! whose `supports(path, mime)` is true. MIME should be supplied by the caller
//! (typically from the file index or [`mime::detect`]) so each file is sniffed
//! at most once per pipeline pass.
//!
//! # Feature flags
//!
//! - `text`  (default) — [`text::PlainTextExtractor`]
//! - `pdf`            — `PdfExtractor` (future)
//! - `image`          — `ImageExtractor` (future)
//! - `excel`          — `ExcelExtractor` (future)
//! - `audio`          — `AudioExtractor` (future)
//!
//! Binaries built with `--no-default-features` still compile; they just lack
//! any extractor implementations and must register their own.

pub mod mime;
pub mod router;

#[cfg(feature = "text")]
pub mod text;

pub use router::pick;

// TODO(Phase 2): PdfExtractor, ExcelExtractor, ImageExtractor, AudioExtractor.
