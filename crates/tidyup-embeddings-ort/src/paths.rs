//! Platform-aware cache paths for ONNX model artifacts.
//!
//! Resolution order:
//! 1. `TIDYUP_MODEL_CACHE` environment variable (explicit override).
//! 2. `dirs::cache_dir()` joined with `tidyup/models/`.
//!    - macOS:   `~/Library/Caches/tidyup/models/`
//!    - Linux:   `$XDG_CACHE_HOME/tidyup/models/` or `~/.cache/tidyup/models/`
//!    - Windows: `%LOCALAPPDATA%\tidyup\models\`
//!
//! If neither resolves, [`model_cache_dir`] returns `None` and callers
//! should surface a user-actionable error.

use std::path::PathBuf;

/// Directory that holds ONNX model files and the taxonomy embedding cache.
#[must_use]
pub fn model_cache_dir() -> Option<PathBuf> {
    if let Ok(override_path) = std::env::var("TIDYUP_MODEL_CACHE") {
        return Some(PathBuf::from(override_path));
    }
    dirs::cache_dir().map(|d| d.join("tidyup").join("models"))
}

/// Full path to the default `bge-small-en-v1.5` ONNX file.
#[must_use]
pub fn default_model_path() -> Option<PathBuf> {
    model_cache_dir().map(|d| d.join("bge-small-en-v1.5").join("model.onnx"))
}

/// Full path to the default `bge-small-en-v1.5` tokenizer.
#[must_use]
pub fn default_tokenizer_path() -> Option<PathBuf> {
    model_cache_dir().map(|d| d.join("bge-small-en-v1.5").join("tokenizer.json"))
}

/// Full path to the taxonomy embedding cache.
#[must_use]
pub fn taxonomy_cache_path() -> Option<PathBuf> {
    model_cache_dir().map(|d| d.join("taxonomy_embeddings.json"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn env_override_wins() {
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY-adjacent: set/unset env vars across threads is unsound in general
        // but tests run single-threaded per-process unless `--test-threads > 1`
        // and these setters are scoped. Use scoped override via a lock if parallelized later.
        std::env::set_var("TIDYUP_MODEL_CACHE", tmp.path());
        let dir = model_cache_dir().unwrap();
        assert_eq!(dir, tmp.path());
        std::env::remove_var("TIDYUP_MODEL_CACHE");
    }

    #[test]
    fn default_cache_dir_nonempty() {
        std::env::remove_var("TIDYUP_MODEL_CACHE");
        if let Some(dir) = model_cache_dir() {
            assert!(dir.ends_with("tidyup/models") || dir.ends_with("tidyup\\models"));
        }
    }
}
