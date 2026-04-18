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
pub mod router;

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
