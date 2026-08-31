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

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::Deserialize;

const MODEL_MANIFEST_SOURCE: &str = include_str!("../models.toml");

/// Stable manifest keys used by runtime and installer callers.
pub const DEFAULT_BUNDLE_KEY: &str = "text";
pub const SIGLIP_BUNDLE_KEY: &str = "siglip";
pub const CLAP_BUNDLE_KEY: &str = "clap";

/// Metadata for one model artifact that tidyup needs on disk.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSpec {
    /// Semantic role within the bundle (`model`, `vision`, `text`, ...).
    pub role: String,
    /// File basename written under the model directory.
    pub filename: String,
    /// HTTPS download URL (used only by opt-in installer tooling).
    pub url: String,
    /// Expected file size in bytes — cheap sanity check before hashing.
    pub size_bytes: u64,
    /// BLAKE3 hex digest of the expected file contents. Computed at packaging
    /// time; regenerate on model version bump.
    #[serde(rename = "blake3")]
    pub blake3_hex: String,
}

/// A named bundle of model artifacts that install + verify operate on as a unit.
///
/// The single source of truth for which files a bundle needs, where they live,
/// and what they should hash to — shared by the runtime verifier and the
/// `cargo xtask download-models` / `verify-models` tooling so the two cannot
/// drift.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleSpec {
    /// Stable internal lookup key.
    pub key: String,
    /// Human-readable bundle name, for logs and instructions.
    pub name: String,
    /// Subdirectory under the model cache that holds the bundle's files.
    #[serde(rename = "cache_dir")]
    pub dir: String,
    /// The artifacts that make up the bundle.
    #[serde(rename = "artifact")]
    pub artifacts: Vec<ArtifactSpec>,
    /// Upstream model identity represented by the local bundle.
    pub model_id: String,
    /// Shared latent dimensionality.
    pub dimensions: usize,
    /// Maximum tokenizer sequence length where applicable.
    pub max_sequence_length: usize,
    /// Token id used when padding text-tower inputs.
    pub pad_token_id: u32,
    /// Token string corresponding to `pad_token_id`.
    pub pad_token: String,
    /// Whether inputs must always be padded to `max_sequence_length`.
    pub pad_to_max_length: bool,
    /// Required square image side for vision bundles.
    #[serde(default)]
    pub image_size: Option<u32>,
    /// Auditable upstream repository.
    pub source: String,
    /// Immutable upstream revision.
    pub revision: String,
    /// Stored-weight precision/quantization.
    pub precision: String,
    /// Stable cache identity for preprocessing semantics. Change this whenever
    /// tokenization, padding, normalization, or decoding changes.
    pub preprocessing_version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelManifest {
    schema_version: u32,
    #[serde(rename = "bundle")]
    bundles: Vec<BundleSpec>,
}

static MODEL_MANIFEST: OnceLock<std::result::Result<ModelManifest, String>> = OnceLock::new();

fn manifest() -> Result<&'static ModelManifest> {
    MODEL_MANIFEST
        .get_or_init(|| {
            let parsed: ModelManifest = toml::from_str(MODEL_MANIFEST_SOURCE)
                .map_err(|error| format!("invalid embedded model manifest: {error}"))?;
            if parsed.schema_version != 1 {
                return Err(format!(
                    "unsupported model manifest schema {}",
                    parsed.schema_version
                ));
            }
            for bundle in &parsed.bundles {
                if bundle.revision.len() < 12 || bundle.artifacts.is_empty() {
                    return Err(format!("incomplete model bundle {}", bundle.key));
                }
                for artifact in &bundle.artifacts {
                    if !artifact
                        .url
                        .contains(&format!("/resolve/{}/", bundle.revision))
                    {
                        return Err(format!(
                            "artifact {} in {} is not pinned to revision {}",
                            artifact.role, bundle.key, bundle.revision
                        ));
                    }
                }
            }
            Ok(parsed)
        })
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))
}

/// Look up a checked-in model bundle by stable key.
pub fn model_bundle(key: &str) -> Result<&'static BundleSpec> {
    manifest()?
        .bundles
        .iter()
        .find(|bundle| bundle.key == key)
        .ok_or_else(|| anyhow::anyhow!("model manifest has no bundle keyed {key}"))
}

impl BundleSpec {
    /// Find an artifact by its semantic role.
    #[must_use]
    pub fn artifact(&self, role: &str) -> Option<&ArtifactSpec> {
        self.artifacts.iter().find(|artifact| artifact.role == role)
    }
}

/// The model subdirectory inside the platform cache.
#[must_use]
pub fn default_model_directory() -> Option<PathBuf> {
    let bundle = model_bundle(DEFAULT_BUNDLE_KEY).ok()?;
    crate::paths::model_cache_dir().map(|d| d.join(&bundle.dir))
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
        return Err(anyhow::anyhow!("missing artifact: {}", path.display()));
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
        let (actual, _) = artifact_digest(path)?;
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
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        let chunk = buffer
            .get(..read)
            .ok_or_else(|| anyhow::anyhow!("invalid read length for {}", path.display()))?;
        hasher.update(chunk);
    }
    Ok((hasher.finalize().to_hex().to_string(), size))
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
        .join(&bundle.dir);
    for spec in &bundle.artifacts {
        verify_artifact(&dir.join(&spec.filename), spec)?;
    }
    Ok(dir)
}

/// Verify that every artifact for the default model bundle is present and
/// intact. Returns the bundle directory on success.
///
/// # Errors
/// See [`verify_bundle`].
pub fn verify_default_model() -> Result<PathBuf> {
    verify_bundle(model_bundle(DEFAULT_BUNDLE_KEY)?)
}

/// Human-readable instructions for installing the default model bundle by
/// hand, for builds of tidyup that don't ship an auto-installer.
#[must_use]
pub fn installation_instructions() -> String {
    let Ok(bundle) = model_bundle(DEFAULT_BUNDLE_KEY) else {
        return "Invalid embedded model manifest; rebuild tidyup from a valid checkout."
            .to_string();
    };
    let Some(model) = bundle.artifact("model") else {
        return "Default model manifest is missing its model artifact.".to_string();
    };
    let Some(tokenizer) = bundle.artifact("tokenizer") else {
        return "Default model manifest is missing its tokenizer artifact.".to_string();
    };
    let dir = default_model_directory().map_or_else(
        || "<platform cache>/tidyup/models/bge-small-en-v1.5/".to_string(),
        |d| format!("{}/", d.display()),
    );
    format!(
        "Missing embedding model. Place these two files under\n  {dir}\n\n  \
         - model.onnx     from {model}\n  \
         - tokenizer.json from {tok}\n\n\
         From a local checkout you can also run `cargo xtask download-models`.",
        model = model.url,
        tok = tokenizer.url,
    )
}

/// Verify the `SigLIP` bundle is present. Returns the bundle directory on
/// success.
///
/// # Errors
/// Surfaces missing-artifact errors via [`verify_bundle`].
pub fn verify_siglip_model() -> Result<PathBuf> {
    verify_bundle(model_bundle(SIGLIP_BUNDLE_KEY)?)
}

/// User-facing instructions for installing the `SigLIP` bundle by hand.
#[must_use]
pub fn siglip_installation_instructions() -> String {
    let Ok(bundle) = model_bundle(SIGLIP_BUNDLE_KEY) else {
        return "Invalid embedded SigLIP model manifest; rebuild tidyup from a valid checkout."
            .to_string();
    };
    let dir = crate::paths::model_cache_dir().map_or_else(
        || format!("<platform cache>/tidyup/models/{}/", bundle.dir),
        |d| format!("{}/", d.join(&bundle.dir).display()),
    );
    let source_lines = bundle
        .artifacts
        .iter()
        .map(|artifact| format!("         - {} from {}", artifact.filename, artifact.url))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Missing SigLIP image encoder (Phase 7 multimodal — optional).\n\
         Place these files under\n  {dir}\n\n{source_lines}\n\n\
         From a local checkout you can also run `cargo xtask download-models --siglip`."
    )
}

// ---------------------------------------------------------------------------
// CLAP (audio / text) — local multimodal audio routing.
// ---------------------------------------------------------------------------

/// Verify the `CLAP` bundle is present. Returns the bundle directory on success.
///
/// # Errors
/// Surfaces missing-artifact errors via [`verify_bundle`].
pub fn verify_clap_model() -> Result<PathBuf> {
    verify_bundle(model_bundle(CLAP_BUNDLE_KEY)?)
}

/// User-facing instructions for installing the `CLAP` bundle by hand.
#[must_use]
pub fn clap_installation_instructions() -> String {
    let Ok(bundle) = model_bundle(CLAP_BUNDLE_KEY) else {
        return "Invalid embedded CLAP model manifest; rebuild tidyup from a valid checkout."
            .to_string();
    };
    let dir = crate::paths::model_cache_dir().map_or_else(
        || format!("<platform cache>/tidyup/models/{}/", bundle.dir),
        |d| format!("{}/", d.join(&bundle.dir).display()),
    );
    let source_lines = bundle
        .artifacts
        .iter()
        .map(|artifact| format!("         - {} from {}", artifact.filename, artifact.url))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Missing CLAP audio encoder (Phase 7 multimodal — optional).\n\
         Place these files under\n  {dir}\n\n{source_lines}\n\n\
         From a local checkout you can also run `cargo xtask download-models --clap`."
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn missing_artifact_errors() {
        let spec = ArtifactSpec {
            role: "model".to_string(),
            filename: "model.onnx".to_string(),
            url: "https://example.com".to_string(),
            size_bytes: 0,
            blake3_hex: String::new(),
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
            role: "model".to_string(),
            filename: "file".to_string(),
            url: "https://example.com".to_string(),
            size_bytes: 42,
            blake3_hex: String::new(),
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
            role: "model".to_string(),
            filename: "file".to_string(),
            url: "https://example.com".to_string(),
            size_bytes: 0,
            blake3_hex: "0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
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
            role: "model".to_string(),
            filename: "file".to_string(),
            url: "https://example.com".to_string(),
            size_bytes: 0,
            blake3_hex: String::new(),
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
        let bundle = model_bundle(DEFAULT_BUNDLE_KEY).unwrap();
        assert_eq!(bundle.dir, "bge-small-en-v1.5");
        assert_eq!(bundle.artifacts.len(), 2);
        let names: Vec<_> = bundle
            .artifacts
            .iter()
            .map(|a| a.filename.as_str())
            .collect();
        assert!(names.contains(&"model.onnx"));
        assert!(names.contains(&"tokenizer.json"));
    }

    #[test]
    fn every_bundle_artifact_is_a_huggingface_url() {
        for key in [DEFAULT_BUNDLE_KEY, SIGLIP_BUNDLE_KEY, CLAP_BUNDLE_KEY] {
            let bundle = model_bundle(key).unwrap();
            assert!(!bundle.dir.is_empty());
            assert!(!bundle.artifacts.is_empty());
            for spec in &bundle.artifacts {
                assert!(
                    spec.url.starts_with("https://huggingface.co/"),
                    "non-HF url: {}",
                    spec.url,
                );
                assert!(!spec.filename.is_empty());
            }
        }
    }

    #[test]
    fn manifest_references_are_immutable_and_runtime_metadata_is_present() {
        for key in [DEFAULT_BUNDLE_KEY, SIGLIP_BUNDLE_KEY, CLAP_BUNDLE_KEY] {
            let bundle = model_bundle(key).unwrap();
            assert!(!bundle.model_id.is_empty());
            assert!(bundle.dimensions > 0);
            assert!(bundle.max_sequence_length > 0);
            assert!(!bundle.source.is_empty());
            assert!(!bundle.precision.is_empty());
            for artifact in &bundle.artifacts {
                assert!(artifact
                    .url
                    .contains(&format!("/resolve/{}/", bundle.revision)));
            }
        }
    }
}
