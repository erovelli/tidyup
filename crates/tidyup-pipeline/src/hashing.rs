//! Content hashing for TOCTOU-safe apply.
//!
//! A proposal captures the source file's BLAKE3 at build (scan/migration) time
//! so the executor can re-verify at apply time that the file hasn't been edited
//! or replaced in the review gap. Hashing streams in fixed-size chunks so a
//! multi-GB file never lands in memory at once.

use std::io::Read;
use std::path::Path;

/// Streaming BLAKE3 of a file's bytes, hex-encoded.
///
/// Returns `None` on any I/O error (unreadable file, permission denied,
/// is-a-directory) — the caller then records no scan-time hash and the
/// apply-time TOCTOU guard is skipped for that proposal, preserving prior
/// behaviour rather than blocking the move.
#[must_use]
pub fn content_hash_of(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(_) => return None,
        };
    }
    Some(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn hashes_match_blake3_of_contents() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, b"content bytes").unwrap();
        assert_eq!(
            content_hash_of(&path),
            Some(blake3::hash(b"content bytes").to_hex().to_string()),
        );
    }

    #[test]
    fn changed_content_changes_hash() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.bin");
        std::fs::write(&path, b"before").unwrap();
        let a = content_hash_of(&path).unwrap();
        std::fs::write(&path, b"after edit").unwrap();
        let b = content_hash_of(&path).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn missing_file_is_none() {
        let dir = TempDir::new().unwrap();
        assert_eq!(content_hash_of(&dir.path().join("nope")), None);
    }
}
