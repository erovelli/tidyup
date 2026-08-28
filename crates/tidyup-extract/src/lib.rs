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
//! - `pdf`   (default) — [`pdf::PdfExtractor`]
//! - `image` (default) — [`image::ImageExtractor`]
//! - `excel`          — [`excel::ExcelExtractor`]
//! - `audio`          — [`audio::AudioExtractor`]
//!
//! Binaries built with `--no-default-features` still compile; they just lack
//! any extractor implementations and must register their own.

pub mod mime;

#[cfg(all(target_os = "macos", feature = "image"))]
mod macos_ocr;
pub mod router;

/// Upper bound on the size of a document handed to an in-memory parser.
///
/// `pdf_extract` and `calamine` load and parse the whole file, so above this
/// the extractor returns `text: None` and defers the file to review rather than
/// risk OOM — a real document is never this large, and the text extractor
/// already caps its own reads at [`text::MAX_BYTES`]. Sized to keep peak parse
/// memory bounded on the 8 GB machines the tool targets.
pub const MAX_DOCUMENT_BYTES: u64 = 128 * 1024 * 1024;

#[cfg(feature = "text")]
pub mod text;

#[cfg(feature = "pdf")]
pub mod pdf;

#[cfg(feature = "excel")]
pub mod excel;

#[cfg(feature = "image")]
pub mod image;

#[cfg(feature = "audio")]
pub mod audio;

pub use router::pick;
