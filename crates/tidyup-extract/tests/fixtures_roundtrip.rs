//! Extractor happy-path checks against committed fixtures under
//! `tests/fixtures/`. The unit tests cover error handling and `supports()`
//! logic with in-memory inputs; these tests confirm each extractor actually
//! decodes real files of its target format end-to-end.
//!
//! Each test also runs the fixture through the registry (`mime::detect` +
//! `router::pick`) with the full default-feature extractor set, so we catch
//! regressions where a new extractor steals dispatch from an existing one.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
#[cfg(all(
    feature = "text",
    feature = "pdf",
    feature = "excel",
    feature = "image",
    feature = "audio"
))]
use std::sync::Arc;

#[cfg(any(
    feature = "pdf",
    feature = "excel",
    feature = "image",
    feature = "audio"
))]
use tidyup_core::extractor::ContentExtractor;
#[cfg(all(
    feature = "text",
    feature = "pdf",
    feature = "excel",
    feature = "image",
    feature = "audio"
))]
use tidyup_extract::{mime, router::pick};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[cfg(all(
    feature = "text",
    feature = "pdf",
    feature = "excel",
    feature = "image",
    feature = "audio"
))]
fn full_registry() -> Vec<Arc<dyn ContentExtractor>> {
    // Order mirrors the precedence we expect at runtime: format-specific
    // extractors first, plain text as a catch-all. The text extractor
    // deliberately declines binary MIMEs so this ordering is not load-bearing
    // for correctness — only for dispatch latency.
    vec![
        Arc::new(tidyup_extract::pdf::PdfExtractor::new()),
        Arc::new(tidyup_extract::excel::ExcelExtractor::new()),
        Arc::new(tidyup_extract::image::ImageExtractor::new()),
        Arc::new(tidyup_extract::audio::AudioExtractor::new()),
        Arc::new(tidyup_extract::text::PlainTextExtractor::new()),
    ]
}

#[cfg(feature = "pdf")]
#[tokio::test]
async fn pdf_fixture_extracts_selectable_text() {
    let path = fixture("sample.pdf");
    let e = tidyup_extract::pdf::PdfExtractor::new();
    let out = e.extract(&path).await.unwrap();
    assert_eq!(out.mime, "application/pdf");
    let text = out.text.expect("pdf fixture should produce text");
    assert!(
        text.to_lowercase().contains("tidyup"),
        "expected 'tidyup' in extracted text: {text}"
    );
    assert_eq!(out.metadata["scanned"], false);
}

#[cfg(feature = "excel")]
#[tokio::test]
async fn xlsx_fixture_extracts_cells_and_sheet_name() {
    let path = fixture("sample.xlsx");
    let e = tidyup_extract::excel::ExcelExtractor::new();
    let out = e.extract(&path).await.unwrap();
    assert_eq!(
        out.mime,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    );
    let text = out.text.expect("xlsx fixture should produce text");
    assert!(text.contains("Budget"), "sheet name missing: {text}");
    assert!(text.contains("Rent"), "first row missing: {text}");
    assert!(text.contains("1500"), "numeric cell missing: {text}");
    let sheet_names = out.metadata["sheet_names"].as_array().unwrap();
    assert_eq!(sheet_names.len(), 1);
    assert_eq!(sheet_names[0], "Budget");
}

#[cfg(feature = "image")]
#[tokio::test]
async fn jpg_fixture_extracts_exif_and_dimensions() {
    let path = fixture("sample.jpg");
    let e = tidyup_extract::image::ImageExtractor::new();
    let out = e.extract(&path).await.unwrap();
    assert_eq!(out.mime, "image/jpeg");
    let dims = &out.metadata["dimensions"];
    assert_eq!(dims["width"], 4);
    assert_eq!(dims["height"], 4);
    let exif = out.metadata["exif"].as_object().unwrap();
    assert!(
        exif.contains_key("camera"),
        "expected Make tag in EXIF: {exif:?}"
    );
    assert!(
        exif.contains_key("model"),
        "expected Model tag in EXIF: {exif:?}"
    );
    let text = out.text.unwrap();
    assert!(text.contains("camera:"));
    assert!(text.contains("model:"));
}

#[cfg(feature = "audio")]
#[tokio::test]
async fn mp3_fixture_extracts_id3_tags_and_properties() {
    let path = fixture("sample.mp3");
    let e = tidyup_extract::audio::AudioExtractor::new();
    let out = e.extract(&path).await.unwrap();
    assert_eq!(out.mime, "audio/mpeg");
    let tags = out.metadata["tags"].as_object().unwrap();
    assert_eq!(
        tags.get("title").and_then(|v| v.as_str()),
        Some("Test Song")
    );
    assert_eq!(
        tags.get("artist").and_then(|v| v.as_str()),
        Some("Test Artist")
    );
    assert_eq!(
        tags.get("album").and_then(|v| v.as_str()),
        Some("Test Album")
    );
    // Duration is ~0.1s in the fixture.
    let dur = out.metadata["duration_secs"].as_f64().unwrap();
    assert!(dur > 0.0 && dur < 1.0, "unexpected duration: {dur}");
    let text = out.text.unwrap();
    assert!(text.contains("title: Test Song"));
}

#[cfg(all(
    feature = "text",
    feature = "pdf",
    feature = "excel",
    feature = "image",
    feature = "audio"
))]
#[tokio::test]
async fn registry_routes_each_fixture_to_its_extractor() {
    struct Case {
        file: &'static str,
        expected_mime: &'static str,
    }
    let cases = [
        Case {
            file: "sample.pdf",
            expected_mime: "application/pdf",
        },
        Case {
            file: "sample.xlsx",
            expected_mime: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        },
        Case {
            file: "sample.jpg",
            expected_mime: "image/jpeg",
        },
        Case {
            file: "sample.mp3",
            expected_mime: "audio/mpeg",
        },
    ];

    let registry = full_registry();
    for case in cases {
        let path = fixture(case.file);
        let sniffed = mime::detect(&path).await;
        let chosen = pick(&registry, &path, sniffed.as_deref())
            .unwrap_or_else(|| panic!("no extractor routed for {}", case.file));
        let out = chosen.extract(&path).await.unwrap();
        assert_eq!(
            out.mime, case.expected_mime,
            "fixture {} routed to wrong extractor — mime mismatch",
            case.file
        );
    }
}
