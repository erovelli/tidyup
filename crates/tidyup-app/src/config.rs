//! Layered configuration: defaults → TOML file → environment overrides.
//!
//! Consumed identically by CLI and UI. Lives in `tidyup-app` because config has no
//! heavy deps of its own and is tightly coupled to service construction.
//!
//! # Layering
//!
//! 1. [`TidyupConfig::default`] establishes baseline values.
//! 2. [`load_from`] reads TOML over the defaults (serde `#[serde(default)]` on every
//!    struct lets partial files merge cleanly without `None`-ing missing sections).
//! 3. [`apply_env_overrides`] applies a small whitelist of environment variables —
//!    currently `TIDYUP_CONFIG_PATH` and `TIDYUP_DATA_DIR`.
//!
//! # What this module does **not** do (privacy model)
//!
//! Environment variables that *activate* privacy-sensitive features
//! (`TIDYUP_REMOTE`, `TIDYUP_LLM_FALLBACK`) are per-invocation gates evaluated by the
//! CLI layer, never written back into the persisted config. The config describes
//! *available* capabilities; the CLI decides whether to use them this run.
//!
//! First-run defaults deliberately never enable remote or LLM fallback. See
//! `CLAUDE.md` → "Privacy model".

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_OCR_BYTES: u64 = 20 * 1024 * 1024;

/// Top-level config. Every sub-section has `Default`, and `#[serde(default)]` lets
/// partial TOML files merge without erroring on missing sections.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TidyupConfig {
    pub storage: StorageConfig,
    pub classifier: ClassifierConfig,
    pub inference: InferenceConfig,
    pub extraction: ExtractionConfig,
    pub rename: RenameConfig,
    pub bundle_detection: BundleDetectionConfig,
    pub directory_envelopes: DirectoryEnvelopeConfig,
}

/// Resource bounds for local content extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractionConfig {
    /// Run Apple's local Vision OCR for plausible screenshot files on supported
    /// macOS builds. Dimensions and EXIF extraction are unaffected.
    pub ocr_enabled: bool,
    /// Maximum screenshot size handed whole to the OCR helper.
    pub ocr_max_bytes: u64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            ocr_enabled: true,
            ocr_max_bytes: DEFAULT_MAX_OCR_BYTES,
        }
    }
}

/// Where we keep the sqlite DB + shelved backups + downloaded models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    /// Override for the tidyup data root. `None` = use [`platform_data_path`].
    pub data_dir: Option<PathBuf>,
    /// Default 30 days; shelved backups older than this are eligible for pruning.
    pub backup_retention_days: u32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: None,
            backup_retention_days: 30,
        }
    }
}

/// Classification cascade. Semantic embeddings are always the primary path;
/// the optional LLM is activated separately by the privacy gates below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClassifierConfig {
    /// Legacy ordered tier list retained for config-file compatibility. New
    /// configs contain only `"embeddings"`; unknown old IDs are ignored.
    /// The `"llm"` tier requires `--features llm-fallback` at build time
    /// *and* `inference.llm_fallback = true` *and* a per-invocation activation flag.
    pub tiers: Vec<String>,
    /// Fallback auto-classify threshold for the composite score. Used when a tier
    /// doesn't provide its own threshold.
    pub min_confidence: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            tiers: vec!["embeddings".to_string()],
            min_confidence: 0.75,
        }
    }
}

/// Inference backend configuration. **Never** enables remote or LLM by default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceConfig {
    /// Ordered list of backend IDs to try at runtime. Known IDs:
    /// `"embeddings-ort"` (default, always available), `"mistralrs"`
    /// (requires `--features llm-fallback`), `"remote-openai"` / `"remote-anthropic"`
    /// / `"remote-ollama"` (requires `--features remote`).
    pub backends: Vec<String>,
    /// Allow the optional LLM reranker to run *if* the crate was compiled with
    /// `--features llm-fallback` and the per-invocation flag is set. Never enabled
    /// by default.
    pub llm_fallback: bool,
    /// Remote backend details. Populated has no effect unless `--features remote`
    /// is compiled and `--remote` / `TIDYUP_REMOTE=1` is set per-invocation.
    pub remote: Option<RemoteBackendConfig>,
    /// Default embedding backend settings.
    pub embedding: EmbeddingConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backends: vec!["embeddings-ort".to_string()],
            llm_fallback: false,
            remote: None,
            embedding: EmbeddingConfig::default(),
        }
    }
}

/// Remote backend wiring. Presence in config doesn't activate it — activation is a
/// per-invocation CLI gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteBackendConfig {
    pub endpoint: String,
    /// Environment variable the CLI reads for the API key. Avoids writing the key
    /// itself to disk.
    pub api_key_env: String,
    pub model: String,
}

/// Default-path embedding classifier settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Hugging Face model id or local path. Default: `bge-small-en-v1.5` — ~35 MB
    /// Q8 ONNX, executed locally through `ort` / ONNX Runtime.
    pub model_id: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "bge-small-en-v1.5".to_string(),
        }
    }
}

/// Thresholds gating rename proposals.
///
/// Both signals must clear their threshold before a rename is surfaced to review.
/// Renames never auto-apply, even under `--yes`. Semantic-collection members
/// may receive rename proposals, but the entire bundle then requires review.
///
/// Mirrors [`tidyup_domain::migration::RenameConfig`] on the TOML side; the pipeline
/// materialises the domain type from this when services are wired up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenameConfig {
    /// Lower bound on deterministic semantic-routing evidence. Default 0.85.
    pub min_classification_confidence: f32,
    /// Lower bound on `1.0 - cosine(embed(filename), content_embedding)`. Default 0.60.
    pub min_mismatch_score: f32,
    /// Lower bound on the fraction of retrieved non-textual concept labels
    /// absent from the filename. Separate scale from text cosine mismatch.
    pub min_grounded_mismatch: f32,
    /// Lower bound on a contrastive model's raw concept score for grounded
    /// image/audio naming. This is model-scale evidence, not a probability.
    pub min_grounding_confidence: f32,
    /// Required separation between the last selected concept and the next
    /// unselected concept.
    pub min_grounding_gap: f32,
}

impl Default for RenameConfig {
    fn default() -> Self {
        Self {
            min_classification_confidence: 0.85,
            min_mismatch_score: 0.60,
            min_grounded_mismatch: 0.60,
            min_grounding_confidence: 0.30,
            min_grounding_gap: 0.02,
        }
    }
}

/// Bundle detection toggles.
///
/// v0.1 ships with the compiled-in marker set (`.git`, `Cargo.toml`, `package.json`,
/// `pyproject.toml`, `*.xcodeproj`, `settings.gradle`/`build.gradle`, `.ipynb`
/// neighbours). `extra_markers` is intended to let users declare additional
/// *filename* markers without a rebuild; `soft_bundle_enabled`/`enabled` gate the
/// marker- and metadata-clustering passes (EXIF bursts, ID3 albums, filename
/// families).
///
/// **Not yet wired.** These fields are parsed and round-trip through the config
/// but the scan/migration pipelines don't yet consume them — bundle detection
/// currently always runs with the compiled-in marker set. Threading this section
/// into `scanner::scan` + `clustering::cluster_loose` is tracked as follow-up
/// (see the roadmap); `[rename]` and `[classifier]` config *are* consumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BundleDetectionConfig {
    pub enabled: bool,
    /// User-configured additional marker filenames (e.g. `"deno.json"`, `"flake.nix"`).
    pub extra_markers: Vec<String>,
    pub soft_bundle_enabled: bool,
}

impl Default for BundleDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extra_markers: Vec::new(),
            soft_bundle_enabled: true,
        }
    }
}

/// Policy and resource limits for hierarchy-selected directory envelopes.
///
/// This is intentionally separate from legacy marker detection. Markers may
/// contribute provenance to an envelope, but they never switch the policy on
/// or decide its destination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DirectoryEnvelopeConfig {
    /// Enable generalized parent-first boundary selection. Kept off until the
    /// full execution and rollback path is active for a release.
    pub enabled: bool,
    pub cohesive_threshold: f32,
    pub container_threshold: f32,
    pub sample_cap: usize,
    pub min_move_improvement: f32,
    pub backup_warn_bytes: u64,
    pub backup_hard_limit_bytes: u64,
}

impl Default for DirectoryEnvelopeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cohesive_threshold: 0.60,
            container_threshold: 0.35,
            sample_cap: 24,
            min_move_improvement: 0.05,
            backup_warn_bytes: 2 * 1024 * 1024 * 1024,
            backup_hard_limit_bytes: 10 * 1024 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Load / save
// ---------------------------------------------------------------------------

/// Load a config from `path`. Missing sections fall back to defaults.
pub fn load_from(path: &Path) -> Result<TidyupConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading config at {}", path.display()))?;
    let config: TidyupConfig =
        toml::from_str(&contents).with_context(|| format!("parsing {}", path.display()))?;
    Ok(config)
}

/// Load the layered config: platform default path → env-override path → defaults if absent,
/// then apply environment overrides.
///
/// Lookup precedence for the config file:
/// 1. `TIDYUP_CONFIG_PATH` env var if set.
/// 2. [`platform_config_path`].
///
/// If the resolved file doesn't exist, defaults are returned (this is the first-run case).
pub fn load() -> Result<TidyupConfig> {
    let path = resolved_config_path()?;
    let mut config = if path.exists() {
        load_from(&path)?
    } else {
        TidyupConfig::default()
    };
    apply_env_overrides(&mut config);
    Ok(config)
}

/// The config file path [`load`] will actually read.
///
/// `TIDYUP_CONFIG_PATH` if set, else [`platform_config_path`]. Callers that
/// *display* the config path (e.g. `tidyup config`) must use this, not
/// `platform_config_path` directly, so the shown path matches the file `load`
/// reads under an env override.
///
/// # Errors
/// Propagates [`platform_config_path`] failure when no env override is set.
pub fn resolved_config_path() -> Result<PathBuf> {
    resolve_config_path(config_path_from_env())
}

/// Pure resolver behind [`resolved_config_path`]: an explicit override wins,
/// otherwise fall back to the platform path. Split out so the precedence is
/// unit-testable without mutating process environment.
fn resolve_config_path(env_override: Option<PathBuf>) -> Result<PathBuf> {
    env_override.map_or_else(platform_config_path, Ok)
}

/// Persist `config` as pretty-printed TOML, creating parent directories if needed.
pub fn save(config: &TidyupConfig, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let contents = toml::to_string_pretty(config).context("serialising config")?;
    std::fs::write(path, contents).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn config_path_from_env() -> Option<PathBuf> {
    std::env::var_os("TIDYUP_CONFIG_PATH").map(PathBuf::from)
}

/// Apply the small whitelist of non-activating environment overrides.
///
/// This mutates `cfg` in place. Privacy-sensitive activation gates
/// (`TIDYUP_REMOTE`, `TIDYUP_LLM_FALLBACK`) are **not** handled here — those live
/// in the CLI's per-invocation gate logic.
pub fn apply_env_overrides(cfg: &mut TidyupConfig) {
    if let Some(v) = std::env::var_os("TIDYUP_DATA_DIR") {
        cfg.storage.data_dir = Some(PathBuf::from(v));
    }
}

// ---------------------------------------------------------------------------
// Platform paths
// ---------------------------------------------------------------------------

/// Platform-aware path for the TOML config file.
///
/// | Platform | Location |
/// |----------|----------|
/// | Linux    | `$XDG_CONFIG_HOME/tidyup/config.toml`  (usually `~/.config/tidyup/…`) |
/// | macOS    | `~/Library/Application Support/tidyup/config.toml` |
/// | Windows  | `%APPDATA%\tidyup\config.toml` |
pub fn platform_config_path() -> Result<PathBuf> {
    let base =
        dirs::config_dir().ok_or_else(|| anyhow!("cannot determine platform config directory"))?;
    Ok(base.join("tidyup").join("config.toml"))
}

/// Platform-aware path for the tidyup data root (sqlite DB + shelf + models).
pub fn platform_data_path() -> Result<PathBuf> {
    let base =
        dirs::data_dir().ok_or_else(|| anyhow!("cannot determine platform data directory"))?;
    Ok(base.join("tidyup"))
}

/// Default shelf (backup store) path under the data root.
pub fn platform_backup_path() -> Result<PathBuf> {
    Ok(platform_data_path()?.join("backup"))
}

/// Default model cache path under the data root.
pub fn platform_models_path() -> Result<PathBuf> {
    Ok(platform_data_path()?.join("models"))
}

/// Resolve the effective storage root by layering config over platform default.
///
/// Prefer this over touching `storage.data_dir` directly — it consolidates the
/// default-fallback logic into one place.
pub fn resolve_data_dir(cfg: &StorageConfig) -> Result<PathBuf> {
    cfg.data_dir.clone().map_or_else(platform_data_path, Ok)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn defaults_never_enable_remote_or_llm() {
        let cfg = TidyupConfig::default();
        assert!(
            cfg.inference.remote.is_none(),
            "remote must be off by default"
        );
        assert!(
            !cfg.inference.llm_fallback,
            "llm fallback must be off by default"
        );
        assert!(
            !cfg.classifier.tiers.iter().any(|t| t == "llm"),
            "default tiers must not include llm"
        );
        assert_eq!(cfg.classifier.tiers, ["embeddings"]);
        assert!(
            !cfg.inference
                .backends
                .iter()
                .any(|b| b.starts_with("remote-")),
            "default backends must not include remote"
        );
    }

    #[test]
    fn default_roundtrips_through_toml() {
        let cfg = TidyupConfig::default();
        let serialised = toml::to_string_pretty(&cfg).unwrap();
        let back: TidyupConfig = toml::from_str(&serialised).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn resolve_config_path_prefers_env_override() {
        // An explicit override is returned verbatim — this is the path
        // `tidyup config` must display so it matches what `load` reads.
        let override_path = PathBuf::from("/custom/tidyup.toml");
        assert_eq!(
            resolve_config_path(Some(override_path.clone())).unwrap(),
            override_path,
        );
        // With no override, resolution matches the bare platform path.
        assert_eq!(resolve_config_path(None).ok(), platform_config_path().ok());
    }

    #[test]
    fn partial_toml_uses_defaults_for_missing_sections() {
        let toml_str = r"
[storage]
backup_retention_days = 90
";
        let cfg: TidyupConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.storage.backup_retention_days, 90);
        // Unspecified sections should match defaults.
        assert_eq!(cfg.classifier, ClassifierConfig::default());
        assert_eq!(cfg.extraction, ExtractionConfig::default());
        assert_eq!(cfg.rename, RenameConfig::default());
        assert_eq!(cfg.bundle_detection, BundleDetectionConfig::default());
        assert!((cfg.rename.min_classification_confidence - 0.85).abs() < f32::EPSILON);
        assert!((cfg.rename.min_mismatch_score - 0.60).abs() < f32::EPSILON);
        assert!((cfg.rename.min_grounded_mismatch - 0.60).abs() < f32::EPSILON);
    }

    #[test]
    fn rename_section_round_trips_at_documented_defaults() {
        let cfg = TidyupConfig::default();
        let s = toml::to_string_pretty(&cfg).unwrap();
        assert!(s.contains("[rename]"));
        assert!(s.contains("min_classification_confidence = 0.85"));
        assert!(s.contains("min_mismatch_score = 0.6"));
        assert!(s.contains("min_grounded_mismatch = 0.6"));
    }

    #[test]
    fn extraction_section_round_trips_at_bounded_defaults() {
        let cfg = TidyupConfig::default();
        let serialised = toml::to_string_pretty(&cfg).unwrap();
        assert!(serialised.contains("[extraction]"));
        assert!(serialised.contains("ocr_enabled = true"));
        assert!(serialised.contains("ocr_max_bytes = 20971520"));
        let back: TidyupConfig = toml::from_str(&serialised).unwrap();
        assert_eq!(back.extraction, ExtractionConfig::default());
    }

    #[test]
    fn bundle_detection_section_present_in_serialised_defaults() {
        let s = toml::to_string_pretty(&TidyupConfig::default()).unwrap();
        assert!(s.contains("[bundle_detection]"));
        assert!(s.contains("enabled = true"));
        assert!(s.contains("soft_bundle_enabled = true"));
    }

    #[test]
    fn load_from_reads_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tidyup.toml");
        let custom = TidyupConfig {
            storage: StorageConfig {
                data_dir: Some(PathBuf::from("/mnt/tidyup")),
                backup_retention_days: 45,
            },
            ..TidyupConfig::default()
        };
        save(&custom, &path).unwrap();
        let loaded = load_from(&path).unwrap();
        assert_eq!(loaded, custom);
    }

    #[test]
    fn save_creates_parent_directories() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("a").join("b").join("config.toml");
        save(&TidyupConfig::default(), &nested).unwrap();
        assert!(nested.exists());
    }

    #[test]
    fn resolve_data_dir_prefers_override_when_set() {
        let cfg = StorageConfig {
            data_dir: Some(PathBuf::from("/data/override")),
            ..StorageConfig::default()
        };
        assert_eq!(
            resolve_data_dir(&cfg).unwrap(),
            PathBuf::from("/data/override")
        );
    }

    #[test]
    fn platform_paths_are_non_empty() {
        // These depend on the host so we only assert they resolve + mention tidyup.
        let config = platform_config_path().unwrap();
        let data = platform_data_path().unwrap();
        assert!(config.ends_with("tidyup/config.toml"));
        assert!(data.ends_with("tidyup"));
        assert!(platform_backup_path().unwrap().starts_with(&data));
        assert!(platform_models_path().unwrap().starts_with(&data));
    }

    #[test]
    fn unknown_keys_are_rejected_so_typos_surface_at_load() {
        // `deny_unknown_fields` keeps typos from silently doing nothing.
        let toml_str = r#"
[storage]
ddata_dir = "/tmp"
"#;
        let err = toml::from_str::<TidyupConfig>(toml_str).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown field"), "actual: {msg}");
    }
}
