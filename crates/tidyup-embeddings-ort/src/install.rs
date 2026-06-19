//! Model installation metadata and verification.
//!
//! The default binary is network-silent (see `CLAUDE.md` privacy model) — it
//! cannot download the ONNX model itself. Instead, this module provides the
//! URLs + checksums + paths that installer tooling (`cargo xtask
//! download-models`, Homebrew formula, OS packagers) uses to place the files
//! under the platform cache directory, plus a verification helper and a
//! user-facing instructions builder for when the model is absent.
//!
//! Placement is simple: `<cache>/tidyup/models/bge-small-en-v1.5/model.onnx`
//! and `<cache>/tidyup/models/bge-small-en-v1.5/tokenizer.json`. First-run
//! download over HTTP is feature-gated and lives outside this module — the
//! default build never links an HTTP client.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Metadata for one model artifact that tidyup needs on disk.
#[derive(Debug, Clone)]
pub struct ArtifactSpec {
    /// File basename written under the model directory.
    pub filename: &'static str,
    /// HTTPS download URL (used only by opt-in installer tooling).
    pub url: &'static str,
    /// Expected file size in bytes — cheap sanity check before hashing.
    pub size_bytes: u64,
    /// BLAKE3 hex digest of the expected file contents. Computed at packaging
    /// time; regenerate on model version bump.
    pub blake3_hex: &'static str,
}

/// A named bundle of model artifacts that install + verify operate on as a unit.
///
/// The single source of truth for which files a bundle needs, where they live,
/// and what they should hash to — shared by the runtime verifier and the
/// `cargo xtask download-models` / `verify-models` tooling so the two cannot
/// drift.
#[derive(Debug, Clone, Copy)]
pub struct BundleSpec {
    /// Human-readable bundle name, for logs and instructions.
    pub name: &'static str,
    /// Subdirectory under the model cache that holds the bundle's files.
    pub dir: &'static str,
    /// The artifacts that make up the bundle.
    pub artifacts: &'static [ArtifactSpec],
}

/// The default embedding model bundle: `bge-small-en-v1.5` ONNX + its
/// tokenizer.
///
/// URLs point at the canonical Hugging Face artifacts. The `size_bytes` /
/// `blake3_hex` fields are unpinned (`0` / empty) by default; [`verify_artifact`]
/// skips those checks when unset, so unpinned specs never break installs. To
/// pin: run `cargo xtask download-models`, which prints each file's BLAKE3 + size,
/// then paste them here — ideally after switching the URL to an immutable
/// `resolve/<commit-sha>/` revision so the pin stays valid.
pub const DEFAULT_MODEL_DIR: &str = "bge-small-en-v1.5";

/// ONNX encoder weights.
pub const DEFAULT_MODEL_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "model.onnx",
    url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
    size_bytes: 0,
    blake3_hex: "",
};

/// `WordPiece` tokenizer in `tokenizers`-crate JSON format.
pub const DEFAULT_TOKENIZER_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "tokenizer.json",
    url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json",
    size_bytes: 0,
    blake3_hex: "",
};

/// Every artifact in the default embedding bundle, in install order.
pub const DEFAULT_ARTIFACTS: &[ArtifactSpec] =
    &[DEFAULT_MODEL_ARTIFACT, DEFAULT_TOKENIZER_ARTIFACT];

/// The default embedding bundle (`bge-small-en-v1.5`).
pub const DEFAULT_BUNDLE: BundleSpec = BundleSpec {
    name: "bge-small-en-v1.5",
    dir: DEFAULT_MODEL_DIR,
    artifacts: DEFAULT_ARTIFACTS,
};

/// The model subdirectory inside the platform cache.
#[must_use]
pub fn default_model_directory() -> Option<PathBuf> {
    crate::paths::model_cache_dir().map(|d| d.join(DEFAULT_MODEL_DIR))
}

/// Verify one artifact on disk.
///
/// - Returns `Ok(())` if the file exists, its size matches (when non-zero in
///   the spec), and its BLAKE3 digest matches (when a digest is pinned).
/// - Returns `Err` with a user-actionable message otherwise.
///
/// # Errors
/// Missing file, wrong size, mismatched hash, or I/O failure.
pub fn verify_artifact(path: &Path, spec: &ArtifactSpec) -> Result<()> {
    if !path.exists() {
        return Err(anyhow::anyhow!("missing artifact: {}", path.display(),));
    }
    let metadata = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if spec.size_bytes != 0 && metadata.len() != spec.size_bytes {
        return Err(anyhow::anyhow!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            spec.size_bytes,
            metadata.len(),
        ));
    }
    if !spec.blake3_hex.is_empty() {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let actual = blake3::hash(&bytes).to_hex().to_string();
        if actual != spec.blake3_hex {
            return Err(anyhow::anyhow!(
                "{} checksum mismatch: expected {}, got {}",
                path.display(),
                spec.blake3_hex,
                actual,
            ));
        }
    }
    Ok(())
}

/// Compute the BLAKE3 hex digest and byte size of a file on disk.
///
/// Used by installer tooling to report the values a maintainer should pin into
/// the [`ArtifactSpec`]s (via `cargo xtask download-models`), and by
/// `verify-models` for diagnostics. Unlike [`verify_artifact`] — which is lazy
/// and only hashes when a digest is pinned — this always reads the whole file,
/// so reserve it for tooling, not the hot binary-load path.
///
/// # Errors
/// I/O failure reading the file.
pub fn artifact_digest(path: &Path) -> Result<(String, u64)> {
    let size = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let hash = blake3::hash(&bytes).to_hex().to_string();
    Ok((hash, size))
}

/// Verify every artifact in `bundle` against its on-disk file under the model
/// cache. Returns the bundle directory on success.
///
/// # Errors
/// Platform cache unavailable, or any artifact missing / wrong size /
/// mismatched checksum (the first failure short-circuits via
/// [`verify_artifact`]).
pub fn verify_bundle(bundle: &BundleSpec) -> Result<PathBuf> {
    let dir = crate::paths::model_cache_dir()
        .ok_or_else(|| {
            anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
        })?
        .join(bundle.dir);
    for spec in bundle.artifacts {
        verify_artifact(&dir.join(spec.filename), spec)?;
    }
    Ok(dir)
}

/// Verify that every artifact for the default model bundle is present and
/// intact. Returns the bundle directory on success.
///
/// # Errors
/// See [`verify_bundle`].
pub fn verify_default_model() -> Result<PathBuf> {
    verify_bundle(&DEFAULT_BUNDLE)
}

/// Human-readable instructions for installing the default model bundle by
/// hand, for builds of tidyup that don't ship an auto-installer.
#[must_use]
pub fn installation_instructions() -> String {
    let dir = default_model_directory().map_or_else(
        || "<platform cache>/tidyup/models/bge-small-en-v1.5/".to_string(),
        |d| format!("{}/", d.display()),
    );
    format!(
        "Missing embedding model. Place these two files under\n  {dir}\n\n  \
         - model.onnx     from {model}\n  \
         - tokenizer.json from {tok}\n\n\
         From a local checkout you can also run `cargo xtask download-models`.",
        model = DEFAULT_MODEL_ARTIFACT.url,
        tok = DEFAULT_TOKENIZER_ARTIFACT.url,
    )
}

// ---------------------------------------------------------------------------
// SigLIP (image / text) — Phase 7 multimodal Tier 2 image classifier.
// ---------------------------------------------------------------------------

/// Vision tower ONNX. Source: HF `nielsr/siglip-base-patch16-224` ONNX export.
pub const SIGLIP_VISION_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "vision_model.onnx",
    url:
        "https://huggingface.co/nielsr/siglip-base-patch16-224/resolve/main/onnx/vision_model.onnx",
    size_bytes: 0,
    blake3_hex: "",
};

/// Text tower ONNX (sentencepiece-style, but exported with WordPiece-shaped IO).
pub const SIGLIP_TEXT_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "text_model.onnx",
    url: "https://huggingface.co/nielsr/siglip-base-patch16-224/resolve/main/onnx/text_model.onnx",
    size_bytes: 0,
    blake3_hex: "",
};

/// Tokenizer JSON for the text tower.
pub const SIGLIP_TOKENIZER_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "tokenizer.json",
    url: "https://huggingface.co/nielsr/siglip-base-patch16-224/resolve/main/tokenizer.json",
    size_bytes: 0,
    blake3_hex: "",
};

/// Every artifact in the `SigLIP` image bundle, in install order.
pub const SIGLIP_ARTIFACTS: &[ArtifactSpec] = &[
    SIGLIP_VISION_ARTIFACT,
    SIGLIP_TEXT_ARTIFACT,
    SIGLIP_TOKENIZER_ARTIFACT,
];

/// The optional `SigLIP` image bundle.
pub const SIGLIP_BUNDLE: BundleSpec = BundleSpec {
    name: "siglip-base-patch16-224",
    dir: crate::paths::SIGLIP_DIR,
    artifacts: SIGLIP_ARTIFACTS,
};

/// Verify the `SigLIP` bundle is present. Returns the bundle directory on
/// success.
///
/// # Errors
/// Surfaces missing-artifact errors via [`verify_bundle`].
pub fn verify_siglip_model() -> Result<PathBuf> {
    verify_bundle(&SIGLIP_BUNDLE)
}

/// User-facing instructions for installing the `SigLIP` bundle by hand.
#[must_use]
pub fn siglip_installation_instructions() -> String {
    let dir = crate::paths::model_cache_dir().map_or_else(
        || {
            format!(
                "<platform cache>/tidyup/models/{}/",
                crate::paths::SIGLIP_DIR
            )
        },
        |d| format!("{}/", d.join(crate::paths::SIGLIP_DIR).display()),
    );
    format!(
        "Missing SigLIP image encoder (Phase 7 multimodal — optional).\n\
         Place these three files under\n  {dir}\n\n  \
         - vision_model.onnx from {vision}\n  \
         - text_model.onnx   from {text}\n  \
         - tokenizer.json    from {tok}\n\n\
         From a local checkout you can also run `cargo xtask download-models --siglip`.",
        vision = SIGLIP_VISION_ARTIFACT.url,
        text = SIGLIP_TEXT_ARTIFACT.url,
        tok = SIGLIP_TOKENIZER_ARTIFACT.url,
    )
}

// ---------------------------------------------------------------------------
// CLAP (audio / text) — Phase 7 multimodal Tier 2 audio classifier.
// ---------------------------------------------------------------------------

/// Audio tower ONNX. Source: HF `laion/clap-htsat-unfused` ONNX export.
pub const CLAP_AUDIO_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "audio_model.onnx",
    url: "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/main/onnx/audio_model.onnx",
    size_bytes: 0,
    blake3_hex: "",
};

/// Text tower ONNX.
pub const CLAP_TEXT_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "text_model.onnx",
    url: "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/main/onnx/text_model.onnx",
    size_bytes: 0,
    blake3_hex: "",
};

/// Tokenizer JSON for the text tower.
pub const CLAP_TOKENIZER_ARTIFACT: ArtifactSpec = ArtifactSpec {
    filename: "tokenizer.json",
    url: "https://huggingface.co/Xenova/clap-htsat-unfused/resolve/main/tokenizer.json",
    size_bytes: 0,
    blake3_hex: "",
};

/// Every artifact in the `CLAP` audio bundle, in install order.
pub const CLAP_ARTIFACTS: &[ArtifactSpec] = &[
    CLAP_AUDIO_ARTIFACT,
    CLAP_TEXT_ARTIFACT,
    CLAP_TOKENIZER_ARTIFACT,
];

/// The optional `CLAP` audio bundle.
pub const CLAP_BUNDLE: BundleSpec = BundleSpec {
    name: "clap-htsat-unfused",
    dir: crate::paths::CLAP_DIR,
    artifacts: CLAP_ARTIFACTS,
};

/// Verify the `CLAP` bundle is present. Returns the bundle directory on success.
///
/// # Errors
/// Surfaces missing-artifact errors via [`verify_bundle`].
pub fn verify_clap_model() -> Result<PathBuf> {
    verify_bundle(&CLAP_BUNDLE)
}

/// User-facing instructions for installing the `CLAP` bundle by hand.
#[must_use]
pub fn clap_installation_instructions() -> String {
    let dir = crate::paths::model_cache_dir().map_or_else(
        || format!("<platform cache>/tidyup/models/{}/", crate::paths::CLAP_DIR),
        |d| format!("{}/", d.join(crate::paths::CLAP_DIR).display()),
    );
    format!(
        "Missing CLAP audio encoder (Phase 7 multimodal — optional).\n\
         Place these three files under\n  {dir}\n\n  \
         - audio_model.onnx from {audio}\n  \
         - text_model.onnx  from {text}\n  \
         - tokenizer.json   from {tok}\n\n\
         From a local checkout you can also run `cargo xtask download-models --clap`.",
        audio = CLAP_AUDIO_ARTIFACT.url,
        text = CLAP_TEXT_ARTIFACT.url,
        tok = CLAP_TOKENIZER_ARTIFACT.url,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn missing_artifact_errors() {
        let spec = ArtifactSpec {
            filename: "model.onnx",
            url: "https://example.com",
            size_bytes: 0,
            blake3_hex: "",
        };
        let err = verify_artifact(Path::new("/no/such/path"), &spec).unwrap_err();
        assert!(format!("{err}").contains("missing artifact"));
    }

    #[test]
    fn size_mismatch_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"hello").unwrap();
        let spec = ArtifactSpec {
            filename: "file",
            url: "https://example.com",
            size_bytes: 42,
            blake3_hex: "",
        };
        let err = verify_artifact(&path, &spec).unwrap_err();
        assert!(format!("{err}").contains("size mismatch"));
    }

    #[test]
    fn checksum_mismatch_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"hello").unwrap();
        let spec = ArtifactSpec {
            filename: "file",
            url: "https://example.com",
            size_bytes: 0,
            blake3_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        };
        let err = verify_artifact(&path, &spec).unwrap_err();
        assert!(format!("{err}").contains("checksum mismatch"));
    }

    #[test]
    fn empty_checksum_skips_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"hello").unwrap();
        let spec = ArtifactSpec {
            filename: "file",
            url: "https://example.com",
            size_bytes: 0,
            blake3_hex: "",
        };
        verify_artifact(&path, &spec).unwrap();
    }

    #[test]
    fn instructions_mention_both_files() {
        let msg = installation_instructions();
        assert!(msg.contains("model.onnx"));
        assert!(msg.contains("tokenizer.json"));
        assert!(msg.contains("bge-small-en-v1.5"));
    }

    #[test]
    fn artifact_digest_reports_hash_and_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        std::fs::write(&path, b"tidyup integrity").unwrap();
        let (hash, size) = artifact_digest(&path).unwrap();
        assert_eq!(size, 16);
        assert_eq!(hash, blake3::hash(b"tidyup integrity").to_hex().to_string());
    }

    #[test]
    fn default_bundle_is_consistent() {
        assert_eq!(DEFAULT_BUNDLE.dir, DEFAULT_MODEL_DIR);
        assert_eq!(DEFAULT_BUNDLE.artifacts.len(), 2);
        let names: Vec<_> = DEFAULT_BUNDLE
            .artifacts
            .iter()
            .map(|a| a.filename)
            .collect();
        assert!(names.contains(&"model.onnx"));
        assert!(names.contains(&"tokenizer.json"));
    }

    #[test]
    fn every_bundle_artifact_is_a_huggingface_url() {
        assert_eq!(SIGLIP_BUNDLE.artifacts.len(), 3);
        assert_eq!(CLAP_BUNDLE.artifacts.len(), 3);
        for bundle in [DEFAULT_BUNDLE, SIGLIP_BUNDLE, CLAP_BUNDLE] {
            assert!(!bundle.dir.is_empty());
            for spec in bundle.artifacts {
                assert!(
                    spec.url.starts_with("https://huggingface.co/"),
                    "non-HF url: {}",
                    spec.url,
                );
                assert!(!spec.filename.is_empty());
            }
        }
    }
}
