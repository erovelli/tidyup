//! Layered configuration: defaults -> `~/.config/tidyup/config.toml` -> env -> overrides.
//!
//! Consumed identically by CLI and UI. Lives in `tidyup-app` because it has no
//! heavy deps of its own and is tightly coupled to service construction.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TidyupConfig {
    pub inference: InferenceConfig,
    pub classifier: ClassifierConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceConfig {
    /// Ordered list of backend IDs to try. First one whose capabilities match wins.
    /// Examples: `["mistralrs", "remote-openai"]`.
    pub backends: Vec<String>,
    pub remote: Option<RemoteBackendConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteBackendConfig {
    pub endpoint: String,
    pub api_key_env: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClassifierConfig {
    /// Ordered tiers to run. Valid values: `"heuristics"`, `"embeddings"`, `"llm"`.
    pub tiers: Vec<String>,
    pub min_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageConfig {
    pub data_dir: Option<std::path::PathBuf>,
    pub backup_retention_days: u32,
}

pub fn load() -> anyhow::Result<TidyupConfig> {
    // TODO: merge defaults -> file -> env -> overrides
    Ok(TidyupConfig::default())
}
