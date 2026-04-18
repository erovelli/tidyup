//! MIME detection for extractor dispatch.
//!
//! Sniff a prefix of the file for magic bytes via [`infer`]; fall back to
//! [`mime_guess`] over the file extension. Mirrors the strategy used by
//! `tidyup-storage-sqlite`'s indexer so downstream code sees consistent
//! MIME strings regardless of whether it came from the index or a live sniff.

use std::path::Path;

use tokio::io::AsyncReadExt;

/// How many bytes to read for magic-byte sniffing. `infer` only inspects the
/// first few dozen bytes for its signatures; 8 KiB is overkill but cheap and
/// leaves room if signatures extend in future versions.
pub const SNIFF_BYTES: usize = 8 * 1024;

/// Detect the MIME type of a file using content sniffing with an extension
/// fallback. Returns `None` only if the file is unreadable and has no known
/// extension mapping.
///
/// Reads up to [`SNIFF_BYTES`] from the start of the file — never loads the
/// whole file.
pub async fn detect(path: &Path) -> Option<String> {
    let sniffed = read_prefix(path)
        .await
        .and_then(|buf| infer::get(&buf).map(|kind| kind.mime_type().to_string()));
    sniffed.or_else(|| {
        let guess = mime_guess::from_path(path).first()?;
        Some(guess.to_string())
    })
}

async fn read_prefix(path: &Path) -> Option<Vec<u8>> {
    let mut file = match tokio::fs::File::open(path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!("mime sniff: unable to open {}: {e}", path.display());
            return None;
        }
    };
    let mut buf = Vec::with_capacity(SNIFF_BYTES);
    let mut limited = (&mut file).take(SNIFF_BYTES as u64);
    if let Err(e) = limited.read_to_end(&mut buf).await {
        tracing::debug!("mime sniff: read failed for {}: {e}", path.display());
        return None;
    }
    Some(buf)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use std::io::Write as _;

    #[tokio::test]
    async fn png_magic_bytes_win_over_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mislabeled.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        // PNG magic number.
        f.write_all(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();
        f.write_all(b"rest of fake png").unwrap();

        let mime = detect(&path).await.unwrap();
        assert_eq!(mime, "image/png");
    }

    #[tokio::test]
    async fn falls_back_to_extension_when_sniff_is_inconclusive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.md");
        std::fs::write(&path, b"# heading\nbody").unwrap();

        let mime = detect(&path).await.unwrap();
        assert!(
            mime.starts_with("text/"),
            "expected text/* for .md, got {mime}"
        );
    }

    #[tokio::test]
    async fn missing_file_returns_none_when_extension_also_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gone.weirdext");
        let mime = detect(&path).await;
        assert!(mime.is_none(), "unexpected mime: {mime:?}");
    }
}
