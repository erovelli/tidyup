//! Spreadsheet extractor.
//!
//! Handles `.xlsx`, `.xls`, `.xlsb`, `.xlsm`, and `.ods` via the `calamine`
//! crate. Produces a plain-text transcription of each sheet's first
//! [`MAX_ROWS_PER_SHEET`] rows joined by tabs/newlines — enough structure for
//! the classifier to infer topic without dragging multi-megabyte workbooks
//! fully into memory.
//!
//! `calamine` opens workbooks synchronously and can scan entire ZIP archives
//! on large files; extraction runs on `spawn_blocking`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use calamine::Reader;
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::Result;

/// Rows sampled per sheet. A classifier prefix, not a full transcription.
pub const MAX_ROWS_PER_SHEET: usize = 50;

/// MIME types commonly seen for the supported spreadsheet formats. We accept
/// any of these; absent a matching MIME, the extension check carries.
const SPREADSHEET_MIMES: &[&str] = &[
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.ms-excel.sheet.macroenabled.12",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12",
    "application/vnd.oasis.opendocument.spreadsheet",
];

const SPREADSHEET_EXTENSIONS: &[&str] = &["xlsx", "xls", "xlsb", "xlsm", "ods"];

/// Extractor for Excel and ODS spreadsheets.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExcelExtractor;

impl ExcelExtractor {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ContentExtractor for ExcelExtractor {
    fn supports(&self, path: &Path, mime: Option<&str>) -> bool {
        if let Some(m) = mime {
            if SPREADSHEET_MIMES.contains(&m) {
                return true;
            }
        }
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| {
                SPREADSHEET_EXTENSIONS
                    .iter()
                    .any(|e| ext.eq_ignore_ascii_case(e))
            })
    }

    async fn extract(&self, path: &Path) -> Result<ExtractedContent> {
        let owned: PathBuf = path.to_path_buf();
        let result = tokio::task::spawn_blocking(move || transcribe_workbook(&owned)).await?;

        match result {
            Ok((text, sheet_names, rows_read)) => Ok(ExtractedContent {
                text: Some(text),
                mime: mime_for_extension(path),
                metadata: serde_json::json!({
                    "sheet_names": sheet_names,
                    "rows_read": rows_read,
                    "max_rows_per_sheet": MAX_ROWS_PER_SHEET,
                }),
            }),
            Err(e) => {
                tracing::warn!("excel extract failed for {}: {e}", path.display());
                Ok(ExtractedContent {
                    text: None,
                    mime: mime_for_extension(path),
                    metadata: serde_json::json!({ "error": e }),
                })
            }
        }
    }
}

fn transcribe_workbook(path: &Path) -> std::result::Result<(String, Vec<String>, usize), String> {
    let mut wb = calamine::open_workbook_auto(path).map_err(|e| e.to_string())?;

    let sheet_names: Vec<String> = wb.sheet_names();
    let mut text = String::new();
    let mut rows_read = 0usize;

    for sheet in &sheet_names {
        let range = match wb.worksheet_range(sheet) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("excel extract: skipping sheet {sheet}: {e}");
                continue;
            }
        };
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("# ");
        text.push_str(sheet);
        text.push('\n');
        for row in range.rows().take(MAX_ROWS_PER_SHEET) {
            let cells: Vec<String> = row.iter().map(|c| format!("{c}")).collect();
            text.push_str(&cells.join("\t"));
            text.push('\n');
            rows_read += 1;
        }
    }

    Ok((text, sheet_names, rows_read))
}

fn mime_for_extension(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("xlsx" | "xlsm") => {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string()
        }
        Some("xls") => "application/vnd.ms-excel".to_string(),
        Some("xlsb") => "application/vnd.ms-excel.sheet.binary.macroenabled.12".to_string(),
        Some("ods") => "application/vnd.oasis.opendocument.spreadsheet".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn supports_by_mime() {
        let e = ExcelExtractor::new();
        assert!(e.supports(
            Path::new("report"),
            Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        ));
        assert!(e.supports(Path::new("legacy"), Some("application/vnd.ms-excel")));
        assert!(!e.supports(Path::new("report"), Some("text/csv")));
    }

    #[test]
    fn supports_by_extension() {
        let e = ExcelExtractor::new();
        assert!(e.supports(Path::new("q3.XLSX"), None));
        assert!(e.supports(Path::new("old.xls"), None));
        assert!(e.supports(Path::new("open.ods"), None));
        assert!(!e.supports(Path::new("data.csv"), None));
    }

    #[tokio::test]
    async fn missing_file_returns_error_metadata() {
        let e = ExcelExtractor::new();
        let out = e
            .extract(Path::new("/no/such/workbook.xlsx"))
            .await
            .unwrap();
        assert!(out.text.is_none());
        assert!(out.metadata.get("error").is_some());
        assert_eq!(
            out.mime,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
    }

    #[test]
    fn mime_mapping_roundtrip() {
        assert_eq!(
            mime_for_extension(Path::new("a.xlsx")),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
        assert_eq!(
            mime_for_extension(Path::new("a.xls")),
            "application/vnd.ms-excel"
        );
        assert_eq!(
            mime_for_extension(Path::new("a.ods")),
            "application/vnd.oasis.opendocument.spreadsheet"
        );
    }
}
