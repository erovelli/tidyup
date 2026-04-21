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

/// The default embedding model bundle: `bge-small-en-v1.5` ONNX + its
/// tokenizer.
///
/// URLs point at the canonical Hugging Face artifacts. Size and checksum are
/// placeholders until the tooling that populates them lands; `verify_artifact`
/// skips hash comparison when `blake3_hex` is empty so placeholder values
/// don't break user builds while the model pipeline is wired up.
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

/// Verify that every artifact for the default model bundle is present and
/// intact. Returns the bundle directory on success.
///
/// # Errors
/// See [`verify_artifact`]; the first failure short-circuits.
pub fn verify_default_model() -> Result<PathBuf> {
    let dir = default_model_directory().ok_or_else(|| {
        anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
    })?;
    verify_artifact(
        &dir.join(DEFAULT_MODEL_ARTIFACT.filename),
        &DEFAULT_MODEL_ARTIFACT,
    )?;
    verify_artifact(
        &dir.join(DEFAULT_TOKENIZER_ARTIFACT.filename),
        &DEFAULT_TOKENIZER_ARTIFACT,
    )?;
    Ok(dir)
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

/// Verify the `SigLIP` bundle is present. Returns the bundle directory on
/// success.
///
/// # Errors
/// Surfaces missing-artifact errors via [`verify_artifact`].
pub fn verify_siglip_model() -> Result<PathBuf> {
    let dir = crate::paths::model_cache_dir()
        .ok_or_else(|| {
            anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
        })?
        .join(crate::paths::SIGLIP_DIR);
    verify_artifact(
        &dir.join(SIGLIP_VISION_ARTIFACT.filename),
        &SIGLIP_VISION_ARTIFACT,
    )?;
    verify_artifact(
        &dir.join(SIGLIP_TEXT_ARTIFACT.filename),
        &SIGLIP_TEXT_ARTIFACT,
    )?;
    verify_artifact(
        &dir.join(SIGLIP_TOKENIZER_ARTIFACT.filename),
        &SIGLIP_TOKENIZER_ARTIFACT,
    )?;
    Ok(dir)
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

/// Verify the `CLAP` bundle is present. Returns the bundle directory on success.
///
/// # Errors
/// Surfaces missing-artifact errors via [`verify_artifact`].
pub fn verify_clap_model() -> Result<PathBuf> {
    let dir = crate::paths::model_cache_dir()
        .ok_or_else(|| {
            anyhow::anyhow!("platform cache directory unavailable; set TIDYUP_MODEL_CACHE")
        })?
        .join(crate::paths::CLAP_DIR);
    verify_artifact(
        &dir.join(CLAP_AUDIO_ARTIFACT.filename),
        &CLAP_AUDIO_ARTIFACT,
    )?;
    verify_artifact(&dir.join(CLAP_TEXT_ARTIFACT.filename), &CLAP_TEXT_ARTIFACT)?;
    verify_artifact(
        &dir.join(CLAP_TOKENIZER_ARTIFACT.filename),
        &CLAP_TOKENIZER_ARTIFACT,
    )?;
    Ok(dir)
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
}
