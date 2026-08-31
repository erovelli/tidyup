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
mod processing;
pub mod rollback;
pub mod scan;

pub use migration::{MigrationReport, MigrationService};
pub use rollback::{RollbackReport, RollbackService};
pub use scan::{ScanReport, ScanService};

/// Named runtime inputs used to construct a capability manifest.
///
/// Keeping activation facts in a struct prevents positional booleans from
/// being swapped when frontends wire their service context.
#[derive(Clone, Copy)]
#[allow(missing_debug_implementations)]
pub struct CapabilityManifestInput<'a> {
    pub embeddings: &'a dyn tidyup_core::inference::EmbeddingBackend,
    pub image_embeddings: Option<&'a dyn tidyup_core::inference::ImageEmbeddingBackend>,
    pub audio_embeddings: Option<&'a dyn tidyup_core::inference::AudioEmbeddingBackend>,
    pub text: Option<&'a dyn tidyup_core::inference::TextBackend>,
    pub vision: Option<&'a dyn tidyup_core::inference::VisionBackend>,
    pub extractors: &'a [std::sync::Arc<dyn tidyup_core::extractor::ContentExtractor>],
    pub ocr_enabled: bool,
    pub ocr_available: bool,
    pub text_requires_network: bool,
}

/// Build the inspectable capability manifest persisted with every operational
/// run. This records actual wired backends, not merely compiled features.
#[must_use]
pub fn capability_manifest_for(
    input: CapabilityManifestInput<'_>,
) -> tidyup_domain::CapabilityManifest {
    use tidyup_domain::{CapabilityEntry, CapabilityKind, CapabilityManifest, CapabilityStatus};

    let mut entries = vec![CapabilityEntry {
        kind: CapabilityKind::TextEmbedding,
        status: CapabilityStatus::Available,
        implementation: "embedding-backend".to_string(),
        model_id: Some(input.embeddings.model_id().to_string()),
        dimensions: Some(input.embeddings.dimensions()),
        requires_network: false,
        detail: None,
    }];
    entries.push(image_capability(input.image_embeddings));
    entries.push(audio_capability(input.audio_embeddings));
    entries.push(text_capability(input.text, input.text_requires_network));
    entries.push(vision_capability(input.vision, input.text_requires_network));
    entries.extend(input.extractors.iter().map(|extractor| CapabilityEntry {
        kind: CapabilityKind::ContentExtractor,
        status: CapabilityStatus::Available,
        implementation: extractor.capability_id().to_string(),
        model_id: None,
        dimensions: None,
        requires_network: false,
        detail: None,
    }));
    entries.push(ocr_capability(input.ocr_enabled, input.ocr_available));
    CapabilityManifest {
        schema_version: 1,
        entries,
    }
}

fn image_capability(
    backend: Option<&dyn tidyup_core::inference::ImageEmbeddingBackend>,
) -> tidyup_domain::CapabilityEntry {
    use tidyup_domain::{CapabilityEntry, CapabilityKind, CapabilityStatus};

    backend.map_or_else(
        || CapabilityEntry {
            kind: CapabilityKind::ImageEmbedding,
            status: CapabilityStatus::Unavailable,
            implementation: "image-embedding-backend".to_string(),
            model_id: None,
            dimensions: None,
            requires_network: false,
            detail: Some("verified image model bundle was not loaded".to_string()),
        },
        |backend| CapabilityEntry {
            kind: CapabilityKind::ImageEmbedding,
            status: CapabilityStatus::Available,
            implementation: "image-embedding-backend".to_string(),
            model_id: Some(backend.model_id().to_string()),
            dimensions: Some(backend.dimensions()),
            requires_network: false,
            detail: None,
        },
    )
}

fn audio_capability(
    backend: Option<&dyn tidyup_core::inference::AudioEmbeddingBackend>,
) -> tidyup_domain::CapabilityEntry {
    use tidyup_domain::{CapabilityEntry, CapabilityKind, CapabilityStatus};

    backend.map_or_else(
        || CapabilityEntry {
            kind: CapabilityKind::AudioEmbedding,
            status: CapabilityStatus::Unavailable,
            implementation: "audio-embedding-backend".to_string(),
            model_id: None,
            dimensions: None,
            requires_network: false,
            detail: Some("verified audio model bundle was not loaded".to_string()),
        },
        |backend| CapabilityEntry {
            kind: CapabilityKind::AudioEmbedding,
            status: CapabilityStatus::Available,
            implementation: "audio-embedding-backend".to_string(),
            model_id: Some(backend.model_id().to_string()),
            dimensions: Some(backend.dimensions()),
            requires_network: false,
            detail: None,
        },
    )
}

fn text_capability(
    backend: Option<&dyn tidyup_core::inference::TextBackend>,
    requires_network: bool,
) -> tidyup_domain::CapabilityEntry {
    use tidyup_domain::{CapabilityEntry, CapabilityKind, CapabilityStatus};

    backend.map_or_else(
        || CapabilityEntry {
            kind: CapabilityKind::GenerativeText,
            status: CapabilityStatus::Disabled,
            implementation: "llm-reranker".to_string(),
            model_id: None,
            dimensions: None,
            requires_network: false,
            detail: Some("optional three-gate fallback was not activated".to_string()),
        },
        |backend| CapabilityEntry {
            kind: CapabilityKind::GenerativeText,
            status: CapabilityStatus::Available,
            implementation: "llm-reranker".to_string(),
            model_id: Some(backend.model_id().to_string()),
            dimensions: None,
            requires_network,
            detail: Some("review refinement only; never authors filenames".to_string()),
        },
    )
}

fn vision_capability(
    backend: Option<&dyn tidyup_core::inference::VisionBackend>,
    requires_network: bool,
) -> tidyup_domain::CapabilityEntry {
    use tidyup_domain::{CapabilityEntry, CapabilityKind, CapabilityStatus};

    backend.map_or_else(
        || CapabilityEntry {
            kind: CapabilityKind::VisionCaptioning,
            status: CapabilityStatus::Unavailable,
            implementation: "vision-captioner".to_string(),
            model_id: None,
            dimensions: None,
            requires_network: false,
            detail: Some("no captioning backend wired".to_string()),
        },
        |backend| CapabilityEntry {
            kind: CapabilityKind::VisionCaptioning,
            status: CapabilityStatus::Available,
            implementation: "vision-captioner".to_string(),
            model_id: Some(backend.model_id().to_string()),
            dimensions: None,
            requires_network,
            detail: None,
        },
    )
}

fn ocr_capability(enabled: bool, available: bool) -> tidyup_domain::CapabilityEntry {
    use tidyup_domain::{CapabilityEntry, CapabilityKind, CapabilityStatus};

    CapabilityEntry {
        kind: CapabilityKind::Ocr,
        status: if !enabled {
            CapabilityStatus::Disabled
        } else if available {
            CapabilityStatus::Available
        } else {
            CapabilityStatus::Unavailable
        },
        implementation: "macos-vision-ocr".to_string(),
        model_id: None,
        dimensions: None,
        requires_network: false,
        detail: if enabled && !available {
            Some("OCR was enabled in config but the native helper is unavailable".to_string())
        } else {
            None
        },
    }
}

/// Materialise the pipeline's [`ClassifierConfig`](tidyup_domain::ClassifierConfig)
/// from the loaded [`TidyupConfig`](config::TidyupConfig) for a run.
///
/// - The `[rename]` TOML section drives the rename gate thresholds
///   (`min_classification_confidence`, `min_mismatch_score`).
/// - Optional LLM reranking (`enable_llm_fallback`) is enabled **iff** a text backend was
///   activated (`text_present`): [`ServiceContext::text`] is `Some` only when
///   the three-gate model passed (cargo feature + config bool + per-invocation
///   flag), so this ties reranking to activation while keeping it off by default.
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
            min_grounded_mismatch: cfg.rename.min_grounded_mismatch,
            min_grounding_confidence: cfg.rename.min_grounding_confidence,
            min_grounding_gap: cfg.rename.min_grounding_gap,
        },
        directory_envelopes: tidyup_domain::DirectoryEnvelopeConfig {
            enabled: cfg.directory_envelopes.enabled,
            cohesive_threshold: cfg.directory_envelopes.cohesive_threshold,
            container_threshold: cfg.directory_envelopes.container_threshold,
            sample_cap: cfg.directory_envelopes.sample_cap,
            min_move_improvement: cfg.directory_envelopes.min_move_improvement,
            backup_warn_bytes: cfg.directory_envelopes.backup_warn_bytes,
            backup_hard_limit_bytes: cfg.directory_envelopes.backup_hard_limit_bytes,
        },
        ..tidyup_domain::ClassifierConfig::default()
    }
}

/// Bundle of backend handles a service needs. Constructed once per process.
///
/// Using `Arc<dyn Trait>` everywhere keeps the services object-safe and lets
/// the same instance be shared across CLI command handlers or UI components.
///
/// Optional cross-modal embedding backends
/// ([`image_embeddings`](Self::image_embeddings) and
/// [`audio_embeddings`](Self::audio_embeddings)). Both are `Option<Arc<dyn …>>`
/// because the model artifacts ship out-of-band — the default install path
/// (text-only) leaves them `None` and the pipeline falls back to general text
/// classification for image/audio files.
///
/// [`text`](Self::text) is `Option` for the same reason on the LLM side: the
/// default build is LLM-silent (no `tidyup-inference-mistralrs` linked) and
/// the optional reranker only runs when the context builder both wires a real
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
    /// Runtime capabilities persisted with each run and surfaced before work.
    pub capabilities: tidyup_domain::CapabilityManifest,
    /// Classifier config materialised from the loaded `TidyupConfig` (rename
    /// thresholds, optional-reranker activation). Built once at context construction via
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
                min_grounded_mismatch: 0.73,
                min_grounding_confidence: 0.31,
                min_grounding_gap: 0.04,
            },
            ..TidyupConfig::default()
        };
        let cc = super::classifier_config_for(&cfg, false);
        assert!((cc.rename.min_classification_confidence - 0.42).abs() < f32::EPSILON);
        assert!((cc.rename.min_mismatch_score - 0.99).abs() < f32::EPSILON);
        assert!((cc.rename.min_grounded_mismatch - 0.73).abs() < f32::EPSILON);
        assert!((cc.rename.min_grounding_confidence - 0.31).abs() < f32::EPSILON);
        assert!((cc.rename.min_grounding_gap - 0.04).abs() < f32::EPSILON);
    }
}
