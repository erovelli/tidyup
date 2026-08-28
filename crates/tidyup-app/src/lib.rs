// Stubs return `bail!` and don't yet `await` anything; remove these allows once the
// pipeline crate is wired in and the methods become genuinely async.
#![allow(clippy::unused_async)]
#![allow(clippy::missing_const_for_fn)]

//! Application services — the **plug-and-play handles** both CLI and UI call.
//!
//! # Architectural contract
//!
//! Every service here takes:
//! - Concrete backend implementations (inference, storage, extractor) as generics
//!   or trait objects — selected once at startup.
//! - **Frontend ports** (`&dyn ProgressReporter`, `&dyn ReviewHandler`) per call —
//!   so the same service method can be driven by a terminal session or a Dioxus
//!   component with zero code duplication.
//!
//! The CLI and UI each:
//! 1. Build the backend stack (via `tidyup-config` + registry).
//! 2. Construct the services once.
//! 3. Implement `ProgressReporter` + `ReviewHandler` their way.
//! 4. Call the services.
//!
//! That is the entire seam.

pub mod config;
pub mod executor;
pub mod migration;
pub mod rollback;
pub mod scan;

pub use migration::{MigrationReport, MigrationService};
pub use rollback::{RollbackReport, RollbackService};
pub use scan::{ScanReport, ScanService};

/// Materialise the pipeline's [`ClassifierConfig`](tidyup_domain::ClassifierConfig)
/// from the loaded [`TidyupConfig`](config::TidyupConfig) for a run.
///
/// - The `[rename]` TOML section drives the rename gate thresholds
///   (`min_classification_confidence`, `min_mismatch_score`).
/// - Tier 3 (`enable_llm_fallback`) is enabled **iff** a text backend was
///   activated (`text_present`): [`ServiceContext::text`] is `Some` only when
///   the three-gate model passed (cargo feature + config bool + per-invocation
///   flag), so this ties Tier 3 to activation while keeping it off by default.
/// - The semantic scoring thresholds (`embedding_threshold`, `ambiguity_gap`),
///   weights, and calibration keep their pipeline-tuned
///   domain defaults — the app-config `[classifier]` section's `min_confidence`
///   drives the `--yes` auto-approve threshold at the CLI, not these.
#[must_use]
pub fn classifier_config_for(
    cfg: &config::TidyupConfig,
    text_present: bool,
) -> tidyup_domain::ClassifierConfig {
    tidyup_domain::ClassifierConfig {
        enable_llm_fallback: text_present,
        rename: tidyup_domain::migration::RenameConfig {
            min_classification_confidence: cfg.rename.min_classification_confidence,
            min_mismatch_score: cfg.rename.min_mismatch_score,
        },
        ..tidyup_domain::ClassifierConfig::default()
    }
}

/// Bundle of backend handles a service needs. Constructed once per process.
///
/// Using `Arc<dyn Trait>` everywhere keeps the services object-safe and lets
/// the same instance be shared across CLI command handlers or UI components.
///
/// Phase 7 added the optional cross-modal embedding backends
/// ([`image_embeddings`](Self::image_embeddings) and
/// [`audio_embeddings`](Self::audio_embeddings)). Both are `Option<Arc<dyn …>>`
/// because the model artifacts ship out-of-band — the default install path
/// (text-only) leaves them `None` and the pipeline falls back to text-tier
/// classification for image/audio files.
///
/// [`text`](Self::text) is `Option` for the same reason on the LLM side: the
/// default build is LLM-silent (no `tidyup-inference-mistralrs` linked) and
/// the Tier-3 fallback only runs when the context builder both wires a real
/// [`TextBackend`] *and* the per-invocation activation gate fires. `None` is
/// the privacy-preserving default — see `CLAUDE.md` → "Privacy model".
#[allow(missing_debug_implementations)] // trait objects don't implement Debug
pub struct ServiceContext {
    pub file_index: std::sync::Arc<dyn tidyup_core::storage::FileIndex>,
    pub change_log: std::sync::Arc<dyn tidyup_core::storage::ChangeLog>,
    pub backup_store: std::sync::Arc<dyn tidyup_core::storage::BackupStore>,
    pub run_log: std::sync::Arc<dyn tidyup_core::storage::RunLog>,
    pub text: Option<std::sync::Arc<dyn tidyup_core::inference::TextBackend>>,
    pub embeddings: std::sync::Arc<dyn tidyup_core::inference::EmbeddingBackend>,
    pub vision: Option<std::sync::Arc<dyn tidyup_core::inference::VisionBackend>>,
    pub image_embeddings: Option<std::sync::Arc<dyn tidyup_core::inference::ImageEmbeddingBackend>>,
    pub audio_embeddings: Option<std::sync::Arc<dyn tidyup_core::inference::AudioEmbeddingBackend>>,
    pub extractors: Vec<std::sync::Arc<dyn tidyup_core::extractor::ContentExtractor>>,
    /// Classifier config materialised from the loaded `TidyupConfig` (rename
    /// thresholds, Tier-3 activation). Built once at context construction via
    /// [`classifier_config_for`]; the scan/migration services pass it straight
    /// to the pipeline. Test contexts can use `classifier_config_for(&cfg, …)`
    /// or a literal.
    pub classifier: tidyup_domain::ClassifierConfig,
}

#[cfg(test)]
mod classifier_config_tests {
    use super::config::{RenameConfig, TidyupConfig};

    #[test]
    fn tier3_enabled_only_when_text_backend_present() {
        // Regression guard: flipping the domain default to `false` must not
        // leave an activated backend dead. Tier 3 tracks `text_present`.
        let cfg = TidyupConfig::default();
        assert!(
            super::classifier_config_for(&cfg, true).enable_llm_fallback,
            "an activated text backend must enable Tier 3",
        );
        assert!(
            !super::classifier_config_for(&cfg, false).enable_llm_fallback,
            "no text backend must keep Tier 3 off (privacy default)",
        );
    }

    #[test]
    fn rename_thresholds_come_from_config() {
        let cfg = TidyupConfig {
            rename: RenameConfig {
                min_classification_confidence: 0.42,
                min_mismatch_score: 0.99,
            },
            ..TidyupConfig::default()
        };
        let cc = super::classifier_config_for(&cfg, false);
        assert!((cc.rename.min_classification_confidence - 0.42).abs() < f32::EPSILON);
        assert!((cc.rename.min_mismatch_score - 0.99).abs() < f32::EPSILON);
    }
}
