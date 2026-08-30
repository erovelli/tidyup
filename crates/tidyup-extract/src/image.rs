//! Image extractor.
//!
//! Reads image dimensions via the `image` crate and EXIF metadata via
//! `kamadak-exif`. `text` is a flat `key: value` transcription of the salient
//! tags (camera make/model, timestamp, GPS, orientation) — the format mirrors
//! docorg so the classifier sees a stable prose-ish string rather than nested
//! JSON.
//! On macOS, optional Vision OCR is restricted to plausible screenshot names
//! under a configurable byte cap; ordinary photos never launch the helper.
//!
//! Dimensions use `image::image_dimensions`, which reads only the image header
//! for most formats. The `image` crate doesn't support HEIC/AVIF out of the
//! box in v0.1; those surface with `dimensions: null` and the EXIF block
//! (which *is* decodable via `kamadak-exif`) still populated when present.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::Result;

/// Extensions this extractor claims. Includes formats the `image` crate
/// cannot decode (HEIC, RAW) because EXIF extraction still works on those
/// and dimensions fall back to null.
const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "ico", "avif", "jxl", "heic",
    "heif", "raw", "cr2", "nef", "arw", "dng",
];

/// Default upper bound for an image passed whole to the Vision helper (20 MiB).
pub const DEFAULT_MAX_OCR_BYTES: u64 = 20 * 1024 * 1024;

/// Extractor for image files. Produces dimensions + EXIF metadata; no pixels.
#[derive(Debug, Clone, Copy)]
pub struct ImageExtractor {
    ocr_enabled: bool,
    max_ocr_bytes: u64,
}

impl Default for ImageExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageExtractor {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ocr_enabled: true,
            max_ocr_bytes: DEFAULT_MAX_OCR_BYTES,
        }
    }

    /// Configure bounded local OCR. A zero byte cap disables OCR;
    /// dimensions and EXIF extraction remain available in every case.
    #[must_use]
    pub const fn with_ocr(ocr_enabled: bool, max_ocr_bytes: u64) -> Self {
        Self {
            ocr_enabled,
            max_ocr_bytes,
        }
    }
}

#[async_trait]
impl ContentExtractor for ImageExtractor {
    fn supports(&self, path: &Path, mime: Option<&str>) -> bool {
        if let Some(m) = mime {
            if m.starts_with("image/") {
                return true;
            }
        }
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| IMAGE_EXTENSIONS.iter().any(|e| ext.eq_ignore_ascii_case(e)))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent> {
        let owned: PathBuf = path.to_path_buf();
        let ocr_enabled = self.ocr_enabled;
        let max_ocr_bytes = self.max_ocr_bytes;
        let probe =
            tokio::task::spawn_blocking(move || probe(&owned, ocr_enabled, max_ocr_bytes)).await?;

        let ImageProbe {
            dimensions,
            exif,
            ocr_text,
            error,
        } = probe;

        let mut text_parts = Vec::new();
        if !exif.is_empty() {
            text_parts.push(
                exif.iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
        if let Some(ocr) = &ocr_text {
            if !ocr.is_empty() {
                text_parts.push(ocr.clone());
            }
        }
        let text = (!text_parts.is_empty()).then(|| text_parts.join("\n"));

        let mut exif_obj = serde_json::Map::new();
        for (k, v) in &exif {
            exif_obj.insert((*k).to_string(), serde_json::Value::String(v.clone()));
        }

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "dimensions".to_string(),
            match dimensions {
                Some((w, h)) => serde_json::json!({ "width": w, "height": h }),
                None => serde_json::Value::Null,
            },
        );
        metadata.insert("exif".to_string(), serde_json::Value::Object(exif_obj));
        if let Some(ocr) = ocr_text {
            metadata.insert("ocr_text".to_string(), serde_json::Value::String(ocr));
        }
        if let Some(e) = error {
            metadata.insert("error".to_string(), serde_json::Value::String(e));
        }

        Ok(ExtractedContent {
            text,
            mime: mime_for_extension(path),
            metadata: serde_json::Value::Object(metadata),
        })
    }
}

struct ImageProbe {
    dimensions: Option<(u32, u32)>,
    exif: Vec<(&'static str, String)>,
    ocr_text: Option<String>,
    error: Option<String>,
}

// The OCR bounds are only read on macOS builds where the Vision helper
// compiled. Elsewhere they are inert, and `-D warnings` would otherwise reject
// them as unused; keep the real names so the macOS branch stays readable.
#[cfg_attr(
    not(all(target_os = "macos", macos_vision_ocr)),
    allow(unused_variables)
)]
fn probe(path: &Path, ocr_enabled: bool, max_ocr_bytes: u64) -> ImageProbe {
    let dimensions = image::image_dimensions(path).ok();

    #[cfg(all(target_os = "macos", macos_vision_ocr))]
    let ocr_text = should_run_ocr(path, ocr_enabled, max_ocr_bytes)
        .then(|| crate::macos_ocr::recognize(path).ok())
        .flatten()
        .filter(|text| !text.is_empty());
    #[cfg(not(all(target_os = "macos", macos_vision_ocr)))]
    let ocr_text = None;

    let (exif, error) = match extract_exif(path) {
        Ok(tags) => (tags, None),
        Err(e) => (Vec::new(), Some(e)),
    };

    ImageProbe {
        dimensions,
        exif,
        ocr_text,
        error,
    }
}

#[cfg(any(all(target_os = "macos", macos_vision_ocr), test))]
fn should_run_ocr(path: &Path, enabled: bool, max_bytes: u64) -> bool {
    enabled
        && max_bytes > 0
        && plausible_screenshot_name(path)
        && std::fs::metadata(path).is_ok_and(|metadata| metadata.len() <= max_bytes)
}

#[cfg(any(all(target_os = "macos", macos_vision_ocr), test))]
fn plausible_screenshot_name(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return false;
    };
    let tokens: Vec<String> = stem
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    tokens.first().is_some_and(|token| {
        ["screenshot", "screencapture", "screengrab"]
            .iter()
            .any(|prefix| token.starts_with(prefix))
    }) || matches!(tokens.as_slice(), [first, second, ..] if first == "screen" && matches!(second.as_str(), "shot" | "capture" | "grab"))
}

fn extract_exif(path: &Path) -> std::result::Result<Vec<(&'static str, String)>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut buf = std::io::BufReader::new(file);
    let reader = exif::Reader::new()
        .read_from_container(&mut buf)
        .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    for field in reader.fields() {
        let key: &'static str = match field.tag {
            exif::Tag::Make => "camera",
            exif::Tag::Model => "model",
            exif::Tag::DateTimeOriginal => "date",
            exif::Tag::GPSLatitude => "gps_lat",
            exif::Tag::GPSLongitude => "gps_lon",
            exif::Tag::Orientation => "orientation",
            _ => continue,
        };
        out.push((key, format!("{}", field.display_value())));
    }
    Ok(out)
}

fn mime_for_extension(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("jpg" | "jpeg") => "image/jpeg".to_string(),
        Some("png") => "image/png".to_string(),
        Some("gif") => "image/gif".to_string(),
        Some("bmp") => "image/bmp".to_string(),
        Some("tiff" | "tif") => "image/tiff".to_string(),
        Some("webp") => "image/webp".to_string(),
        Some("ico") => "image/x-icon".to_string(),
        Some("avif") => "image/avif".to_string(),
        Some("jxl") => "image/jxl".to_string(),
        Some("heic" | "heif") => "image/heic".to_string(),
        Some("cr2" | "nef" | "arw" | "dng" | "raw") => "image/x-raw".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn supports_by_mime() {
        let e = ImageExtractor::new();
        assert!(e.supports(Path::new("unknown"), Some("image/jpeg")));
        assert!(e.supports(Path::new("unknown"), Some("image/png")));
        assert!(!e.supports(Path::new("unknown"), Some("text/plain")));
    }

    #[test]
    fn supports_by_extension_case_insensitive() {
        let e = ImageExtractor::new();
        assert!(e.supports(Path::new("photo.JPG"), None));
        assert!(e.supports(Path::new("photo.heic"), None));
        assert!(e.supports(Path::new("raw.CR2"), None));
        assert!(!e.supports(Path::new("paper.pdf"), None));
    }

    #[tokio::test]
    async fn extracts_dimensions_from_tiny_png() {
        // Minimal valid 1x1 PNG built by the `image` crate.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.png");
        let img = image::RgbImage::new(2, 3);
        img.save(&path).unwrap();

        let e = ImageExtractor::new();
        let out = e.extract(&path).await.unwrap();
        assert_eq!(out.mime, "image/png");
        let dims = &out.metadata["dimensions"];
        assert_eq!(dims["width"], 2);
        assert_eq!(dims["height"], 3);
        // EXIF absent is not an error — freshly authored PNG has no EXIF.
        assert!(out.text.is_none());
    }

    #[tokio::test]
    async fn missing_file_returns_error_metadata() {
        let e = ImageExtractor::new();
        let out = e.extract(Path::new("/no/such.jpg")).await.unwrap();
        assert!(out.metadata["dimensions"].is_null());
        assert!(out.metadata.get("error").is_some());
    }

    #[test]
    fn ocr_is_limited_to_plausible_screenshots_under_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let screenshot = dir.path().join("Screen_Shot_1.png");
        let photo = dir.path().join("IMG_0001.png");
        std::fs::write(&screenshot, [0_u8; 16]).unwrap();
        std::fs::write(&photo, [0_u8; 16]).unwrap();

        assert!(should_run_ocr(&screenshot, true, 16));
        assert!(!should_run_ocr(&screenshot, false, 16));
        assert!(!should_run_ocr(&screenshot, true, 0));
        assert!(!should_run_ocr(&screenshot, true, 15));
        assert!(!should_run_ocr(&photo, true, 16));
    }

    #[test]
    fn screenshot_name_variants_are_recognized_without_extension_routing() {
        for name in [
            "Screenshot 2026-08-25.png",
            "screenshot_1.anything",
            "Screen-Shot-2",
            "screen_capture_3.tiff",
            "screencapture4.png",
        ] {
            assert!(plausible_screenshot_name(Path::new(name)), "{name}");
        }
        assert!(!plausible_screenshot_name(Path::new("holiday_photo.png")));
    }

    #[test]
    fn mime_mapping_roundtrip() {
        assert_eq!(mime_for_extension(Path::new("a.jpg")), "image/jpeg");
        assert_eq!(mime_for_extension(Path::new("a.jpeg")), "image/jpeg");
        assert_eq!(mime_for_extension(Path::new("a.heic")), "image/heic");
        assert_eq!(mime_for_extension(Path::new("a.cr2")), "image/x-raw");
    }
}
