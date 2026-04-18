//! Image extractor.
//!
//! Reads image dimensions via the `image` crate and EXIF metadata via
//! `kamadak-exif`. `text` is a flat `key: value` transcription of the salient
//! tags (camera make/model, timestamp, GPS, orientation) — the format mirrors
//! docorg so the classifier sees a stable prose-ish string rather than nested
//! JSON.
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

/// Extractor for image files. Produces dimensions + EXIF metadata; no pixels.
#[derive(Debug, Default, Clone, Copy)]
pub struct ImageExtractor;

impl ImageExtractor {
    #[must_use]
    pub const fn new() -> Self {
        Self
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
        let probe = tokio::task::spawn_blocking(move || probe(&owned)).await?;

        let ImageProbe {
            dimensions,
            exif,
            error,
        } = probe;

        let text = if exif.is_empty() {
            None
        } else {
            Some(
                exif.iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        };

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
    error: Option<String>,
}

fn probe(path: &Path) -> ImageProbe {
    let dimensions = image::image_dimensions(path).ok();

    let (exif, error) = match extract_exif(path) {
        Ok(tags) => (tags, None),
        Err(e) => (Vec::new(), Some(e)),
    };

    ImageProbe {
        dimensions,
        exif,
        error,
    }
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
    fn mime_mapping_roundtrip() {
        assert_eq!(mime_for_extension(Path::new("a.jpg")), "image/jpeg");
        assert_eq!(mime_for_extension(Path::new("a.jpeg")), "image/jpeg");
        assert_eq!(mime_for_extension(Path::new("a.heic")), "image/heic");
        assert_eq!(mime_for_extension(Path::new("a.cr2")), "image/x-raw");
    }
}
