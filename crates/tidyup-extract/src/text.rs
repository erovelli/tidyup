//! Plain-text extractor.
//!
//! Covers files whose MIME is `text/*` or whose extension names a textual
//! format that conventionally resolves to `application/*` (source code,
//! structured-data formats, shell scripts). For v0.1 classification the
//! Tier-2 embedding model only needs a prefix of the content, so the
//! extractor caps reads at [`MAX_BYTES`] — ample for signal extraction
//! without risking multi-GB log files blowing out memory.

use std::path::Path;

use async_trait::async_trait;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::Result;
use tokio::io::AsyncReadExt;

/// Maximum bytes read from a single file. Chosen as a safety valve on
/// pathological inputs (multi-GB logs, mislabeled binaries). Classification
/// needs a prefix, not the whole file.
pub const MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Extensions that are plain text but typically don't resolve to a `text/*`
/// MIME via `mime_guess`. Kept deliberately conservative — the goal is
/// recognising files this extractor can sensibly transcribe, not claiming
/// every textual format. Richer formats (PDF, DOCX, XLSX) route to their
/// dedicated extractors.
const TEXT_EXTENSIONS: &[&str] = &[
    // Source
    "rs",
    "py",
    "js",
    "mjs",
    "cjs",
    "ts",
    "tsx",
    "jsx",
    "go",
    "java",
    "kt",
    "kts",
    "swift",
    "c",
    "cc",
    "cpp",
    "cxx",
    "h",
    "hh",
    "hpp",
    "hxx",
    "m",
    "mm",
    "cs",
    "rb",
    "php",
    "pl",
    "pm",
    "lua",
    "r",
    "jl",
    "scala",
    "clj",
    "cljs",
    "ex",
    "exs",
    "erl",
    "hs",
    "ml",
    "mli",
    "fs",
    "fsi",
    "fsx",
    "dart",
    "zig",
    "nim",
    "v",
    "sol",
    "sql",
    // Shell / config
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "bat",
    "cmd",
    // Structured data / config
    "toml",
    "yaml",
    "yml",
    "json",
    "jsonc",
    "json5",
    "ini",
    "cfg",
    "conf",
    "env",
    "properties",
    "xml",
    "csv",
    "tsv",
    // Docs / notes
    "md",
    "markdown",
    "rst",
    "org",
    "tex",
    "bib",
    "asciidoc",
    "adoc",
    // Web
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "less",
    // Misc text
    "log",
    "txt",
    "text",
    "diff",
    "patch",
    "gitignore",
    "gitattributes",
    "editorconfig",
    "dockerfile",
];

/// Reads textual files as a UTF-8 string (with lossy decoding for invalid bytes).
///
/// Invalid UTF-8 is replaced with U+FFFD rather than failing — real-world files
/// routinely mix encodings and a best-effort transcription is more useful to
/// the classifier than a hard error.
#[derive(Debug, Default, Clone, Copy)]
pub struct PlainTextExtractor;

impl PlainTextExtractor {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// MIME prefixes that unambiguously mean "not plain text". If a caller-supplied
/// MIME matches one of these, we trust it over any textual extension — a
/// `.txt` file whose bytes sniff as `image/png` is a binary blob, regardless
/// of the filename.
const BINARY_MIME_PREFIXES: &[&str] = &["image/", "video/", "audio/", "font/"];

/// Specific `application/*` MIMEs that are binary. `application/*` also covers
/// many textual formats (json, toml, xml, javascript, …), so we blacklist
/// known-binary types rather than allow-listing textual ones.
const BINARY_APPLICATION_MIMES: &[&str] = &[
    "application/pdf",
    "application/zip",
    "application/x-tar",
    "application/gzip",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/octet-stream",
    "application/x-executable",
    "application/x-sharedlib",
    "application/x-mach-binary",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/msword",
    "application/vnd.ms-powerpoint",
];

fn mime_looks_binary(mime: &str) -> bool {
    BINARY_MIME_PREFIXES.iter().any(|p| mime.starts_with(p))
        || BINARY_APPLICATION_MIMES.contains(&mime)
}

#[async_trait]
impl ContentExtractor for PlainTextExtractor {
    fn supports(&self, path: &Path, mime: Option<&str>) -> bool {
        if let Some(m) = mime {
            if m.starts_with("text/") {
                return true;
            }
            if mime_looks_binary(m) {
                return false;
            }
            // `application/*` that isn't blacklisted (json, toml, xml, …)
            // falls through to the extension check below, which has the
            // allowlist of textual formats.
        }

        let ext = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase);
        if let Some(ext) = ext.as_deref() {
            return TEXT_EXTENSIONS.contains(&ext);
        }
        // Files like `Dockerfile`, `Makefile`, `LICENSE` have no extension but
        // are conventionally text. Match by filename.
        let stem = path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase);
        matches!(
            stem.as_deref(),
            Some("dockerfile" | "makefile" | "license" | "readme" | "changelog" | "authors")
        )
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut buf = Vec::new();
        (&mut file).take(MAX_BYTES).read_to_end(&mut buf).await?;

        let truncated = buf.len() as u64 == MAX_BYTES;
        let text = String::from_utf8_lossy(&buf).into_owned();
        let mime = crate::mime::detect(path)
            .await
            .unwrap_or_else(|| "text/plain".to_string());

        let metadata = serde_json::json!({
            "byte_count": buf.len(),
            "truncated": truncated,
        });

        Ok(ExtractedContent {
            text: Some(text),
            mime,
            metadata,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn supports_text_mime() {
        let e = PlainTextExtractor::new();
        assert!(e.supports(Path::new("whatever"), Some("text/plain")));
        assert!(e.supports(Path::new("whatever"), Some("text/markdown")));
    }

    #[test]
    fn supports_source_and_config_extensions() {
        let e = PlainTextExtractor::new();
        assert!(e.supports(Path::new("main.rs"), None));
        assert!(e.supports(Path::new("Config.TOML"), None));
        assert!(e.supports(Path::new("x.JSON"), Some("application/json")));
    }

    #[test]
    fn supports_extensionless_conventional_names() {
        let e = PlainTextExtractor::new();
        assert!(e.supports(Path::new("/repo/Dockerfile"), None));
        assert!(e.supports(Path::new("/repo/LICENSE"), None));
    }

    #[test]
    fn rejects_binary_extensions() {
        let e = PlainTextExtractor::new();
        assert!(!e.supports(Path::new("scan.pdf"), None));
        assert!(!e.supports(Path::new("photo.jpg"), None));
        assert!(!e.supports(Path::new("video.mp4"), None));
    }

    #[test]
    fn caller_supplied_binary_mime_overrides_textual_extension() {
        let e = PlainTextExtractor::new();
        // Bytes were sniffed as PNG despite the .txt filename — trust bytes.
        assert!(!e.supports(Path::new("decoy.txt"), Some("image/png")));
        assert!(!e.supports(Path::new("decoy.md"), Some("application/pdf")));
    }

    #[test]
    fn textual_application_mime_still_checks_extension() {
        let e = PlainTextExtractor::new();
        // application/json isn't blacklisted; extension carries the decision.
        assert!(e.supports(Path::new("data.json"), Some("application/json")));
        assert!(!e.supports(Path::new("data.bin"), Some("application/json")));
    }

    #[tokio::test]
    async fn extracts_plain_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("note.txt");
        tokio::fs::write(&path, b"hello, world").await.unwrap();

        let e = PlainTextExtractor::new();
        let out = e.extract(&path).await.unwrap();
        assert_eq!(out.text.as_deref(), Some("hello, world"));
        assert!(out.mime.starts_with("text/"), "mime = {}", out.mime);
        assert_eq!(out.metadata["byte_count"], 12);
        assert_eq!(out.metadata["truncated"], false);
    }

    #[tokio::test]
    async fn invalid_utf8_is_replaced_not_errored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.txt");
        // 0xFF is invalid UTF-8 start byte.
        tokio::fs::write(&path, [b'o', b'k', 0xFF, b'!'])
            .await
            .unwrap();

        let e = PlainTextExtractor::new();
        let out = e.extract(&path).await.unwrap();
        let text = out.text.unwrap();
        assert!(text.starts_with("ok"));
        assert!(text.contains('\u{FFFD}'));
    }

    #[tokio::test]
    async fn truncates_at_max_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.log");
        let size = usize::try_from(MAX_BYTES + 1024).unwrap();
        let big = vec![b'a'; size];
        tokio::fs::write(&path, &big).await.unwrap();

        let e = PlainTextExtractor::new();
        let out = e.extract(&path).await.unwrap();
        assert_eq!(u64::try_from(out.text.unwrap().len()).unwrap(), MAX_BYTES);
        assert_eq!(out.metadata["truncated"], true);
    }
}
