//! End-to-end check: a realistic extractor registry dispatches to
//! `PlainTextExtractor` for textual files and declines binary-looking files.
//! Simulates the shape `tidyup-pipeline` will use at startup.

#![cfg(feature = "text")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use tidyup_core::extractor::ContentExtractor;
use tidyup_extract::mime;
use tidyup_extract::router::pick;
use tidyup_extract::text::PlainTextExtractor;

fn registry() -> Vec<Arc<dyn ContentExtractor>> {
    vec![Arc::new(PlainTextExtractor::new())]
}

#[tokio::test]
async fn dispatches_markdown_file_through_plaintext() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readme.md");
    tokio::fs::write(&path, b"# Title\nsome prose")
        .await
        .unwrap();

    let extractors = registry();
    let mime = mime::detect(&path).await;
    let chosen = pick(&extractors, &path, mime.as_deref()).expect("should route to text");

    let out = chosen.extract(&path).await.unwrap();
    assert!(out.text.unwrap().contains("Title"));
    assert!(out.mime.starts_with("text/"));
}

#[tokio::test]
async fn dispatches_rust_source_without_text_mime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lib.rs");
    tokio::fs::write(&path, b"fn main() { println!(\"hi\"); }")
        .await
        .unwrap();

    let extractors = registry();
    // Rust source typically resolves to application/* or text/*; either way
    // the extension fallback in `supports()` catches it.
    let mime = mime::detect(&path).await;
    let chosen = pick(&extractors, &path, mime.as_deref()).expect("should route to text");

    let out = chosen.extract(&path).await.unwrap();
    assert!(out.text.unwrap().contains("println"));
}

#[tokio::test]
async fn declines_png_even_with_misleading_extension() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("decoy.txt");
    // PNG magic + junk. mime::detect returns image/png, which does not
    // start with text/; the extension is .txt (in TEXT_EXTENSIONS), so
    // `supports()` returns true from the MIME branch only if MIME is text/*.
    // Here MIME is image/png, and without MIME the extension would match.
    // This test pins the precedence: caller-supplied MIME from a real sniff
    // should steer away from PlainTextExtractor when the bytes disagree.
    let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    bytes.extend_from_slice(b"not really a png");
    tokio::fs::write(&path, &bytes).await.unwrap();

    let sniffed = mime::detect(&path).await;
    assert_eq!(sniffed.as_deref(), Some("image/png"));

    // With no other registered extractors, the router returns None for this
    // file when MIME is supplied — confirming PlainTextExtractor yields on
    // non-text MIME rather than swallowing binary content.
    let e = PlainTextExtractor::new();
    assert!(!e.supports(&path, sniffed.as_deref()));
}
