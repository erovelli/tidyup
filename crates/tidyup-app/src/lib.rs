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
pub mod migration;
pub mod rollback;
pub mod scan;

pub use migration::{MigrationReport, MigrationService};
pub use rollback::RollbackService;
pub use scan::{ScanReport, ScanService};

/// Bundle of backend handles a service needs. Constructed once per process.
///
/// Using `Arc<dyn Trait>` everywhere keeps the services object-safe and lets
/// the same instance be shared across CLI command handlers or UI components.
#[allow(missing_debug_implementations)] // trait objects don't implement Debug
pub struct ServiceContext {
    pub file_index: std::sync::Arc<dyn tidyup_core::storage::FileIndex>,
    pub change_log: std::sync::Arc<dyn tidyup_core::storage::ChangeLog>,
    pub backup_store: std::sync::Arc<dyn tidyup_core::storage::BackupStore>,
    pub text: std::sync::Arc<dyn tidyup_core::inference::TextBackend>,
    pub embeddings: std::sync::Arc<dyn tidyup_core::inference::EmbeddingBackend>,
    pub vision: Option<std::sync::Arc<dyn tidyup_core::inference::VisionBackend>>,
    pub extractors: Vec<std::sync::Arc<dyn tidyup_core::extractor::ContentExtractor>>,
}
