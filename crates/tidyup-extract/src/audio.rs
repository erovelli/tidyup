//! Audio extractor.
//!
//! Reads container-level metadata (ID3 / MP4 / FLAC / Vorbis tags) and stream
//! properties (duration, bitrate, sample rate, channels) via `lofty`. The v0.1
//! classifier consumes the transcribed key-value text; raw samples are out of
//! scope until the audio encoder lands in Phase 6.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use lofty::file::{AudioFile, TaggedFileExt};
use lofty::tag::{Accessor, ItemKey};
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::Result;

/// Extensions handled by `lofty`. Video-audio hybrids (`m4v`, etc.) route to
/// a video extractor when that lands; audio-only here.
const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "wav", "flac", "aac", "ogg", "opus", "m4a", "wma", "aiff", "aif", "alac", "ape", "wv",
    "amr", "mka",
];

/// Extractor for audio files.
#[derive(Debug, Default, Clone, Copy)]
pub struct AudioExtractor;

impl AudioExtractor {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContentExtractor for AudioExtractor {
    fn supports(&self, path: &Path, mime: Option<&str>) -> bool {
        if let Some(m) = mime {
            if m.starts_with("audio/") {
                return true;
            }
        }
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| AUDIO_EXTENSIONS.iter().any(|e| ext.eq_ignore_ascii_case(e)))
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent> {
        let owned: PathBuf = path.to_path_buf();
        let probe = tokio::task::spawn_blocking(move || probe(&owned)).await?;

        let AudioProbe {
            tags,
            duration_secs,
            bitrate_kbps,
            sample_rate_hz,
            channels,
            error,
        } = probe;

        let mut lines: Vec<String> = tags
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>();
        if let Some(d) = duration_secs {
            lines.push(format!("duration: {d:.1}s"));
        }
        if let Some(br) = bitrate_kbps {
            lines.push(format!("bitrate: {br} kbps"));
        }
        if let Some(sr) = sample_rate_hz {
            lines.push(format!("sample_rate: {sr} Hz"));
        }
        if let Some(ch) = channels {
            lines.push(format!("channels: {ch}"));
        }

        let text = if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        };

        let mut tag_obj = serde_json::Map::new();
        for (k, v) in &tags {
            tag_obj.insert((*k).to_string(), serde_json::Value::String(v.clone()));
        }

        let mut metadata = serde_json::Map::new();
        metadata.insert("tags".to_string(), serde_json::Value::Object(tag_obj));
        metadata.insert(
            "duration_secs".to_string(),
            duration_secs
                .and_then(serde_json::Number::from_f64)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
        );
        metadata.insert(
            "bitrate_kbps".to_string(),
            bitrate_kbps.map_or(serde_json::Value::Null, Into::into),
        );
        metadata.insert(
            "sample_rate_hz".to_string(),
            sample_rate_hz.map_or(serde_json::Value::Null, Into::into),
        );
        metadata.insert(
            "channels".to_string(),
            channels.map_or(serde_json::Value::Null, Into::into),
        );
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

struct AudioProbe {
    tags: Vec<(&'static str, String)>,
    duration_secs: Option<f64>,
    bitrate_kbps: Option<u32>,
    sample_rate_hz: Option<u32>,
    channels: Option<u8>,
    error: Option<String>,
}

fn probe(path: &Path) -> AudioProbe {
    let tagged = match lofty::read_from_path(path) {
        Ok(t) => t,
        Err(e) => {
            return AudioProbe {
                tags: Vec::new(),
                duration_secs: None,
                bitrate_kbps: None,
                sample_rate_hz: None,
                channels: None,
                error: Some(e.to_string()),
            };
        }
    };

    let mut tags = Vec::new();
    if let Some(tag) = tagged.primary_tag().or_else(|| tagged.first_tag()) {
        if let Some(v) = tag.title() {
            tags.push(("title", v.into_owned()));
        }
        if let Some(v) = tag.artist() {
            tags.push(("artist", v.into_owned()));
        }
        if let Some(v) = tag.album() {
            tags.push(("album", v.into_owned()));
        }
        if let Some(v) = tag.genre() {
            tags.push(("genre", v.into_owned()));
        }
        if let Some(v) = tag.get_string(ItemKey::Year) {
            tags.push(("year", v.to_string()));
        } else if let Some(v) = tag.get_string(ItemKey::RecordingDate) {
            tags.push(("date", v.to_string()));
        }
        if let Some(v) = tag.get_string(ItemKey::AlbumArtist) {
            tags.push(("album_artist", v.to_string()));
        }
    }

    let props = tagged.properties();
    let dur = props.duration();
    let duration_secs = if dur.is_zero() {
        None
    } else {
        Some(dur.as_secs_f64())
    };

    AudioProbe {
        tags,
        duration_secs,
        bitrate_kbps: props.audio_bitrate(),
        sample_rate_hz: props.sample_rate(),
        channels: props.channels(),
        error: None,
    }
}

fn mime_for_extension(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("mp3") => "audio/mpeg".to_string(),
        Some("wav") => "audio/wav".to_string(),
        Some("flac") => "audio/flac".to_string(),
        Some("aac") => "audio/aac".to_string(),
        Some("ogg") => "audio/ogg".to_string(),
        Some("opus") => "audio/opus".to_string(),
        Some("m4a" | "alac") => "audio/mp4".to_string(),
        Some("wma") => "audio/x-ms-wma".to_string(),
        Some("aiff" | "aif") => "audio/aiff".to_string(),
        Some("mka") => "audio/x-matroska".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn supports_by_mime() {
        let e = AudioExtractor::new();
        assert!(e.supports(Path::new("unknown"), Some("audio/mpeg")));
        assert!(e.supports(Path::new("unknown"), Some("audio/flac")));
        assert!(!e.supports(Path::new("unknown"), Some("text/plain")));
    }

    #[test]
    fn supports_by_extension_case_insensitive() {
        let e = AudioExtractor::new();
        assert!(e.supports(Path::new("song.MP3"), None));
        assert!(e.supports(Path::new("song.flac"), None));
        assert!(!e.supports(Path::new("clip.mp4"), None));
    }

    #[tokio::test]
    async fn missing_file_returns_error_metadata() {
        let e = AudioExtractor::new();
        let out = e.extract(Path::new("/no/such.mp3")).await.unwrap();
        assert!(out.text.is_none());
        assert!(out.metadata.get("error").is_some());
        assert_eq!(out.mime, "audio/mpeg");
    }

    #[test]
    fn mime_mapping_roundtrip() {
        assert_eq!(mime_for_extension(Path::new("a.mp3")), "audio/mpeg");
        assert_eq!(mime_for_extension(Path::new("a.flac")), "audio/flac");
        assert_eq!(mime_for_extension(Path::new("a.m4a")), "audio/mp4");
    }
}
