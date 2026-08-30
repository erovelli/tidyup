//! Run aggregate — a single invocation of a scan or migration.
//!
//! Every [`ChangeProposal`](crate::change::ChangeProposal) and
//! [`BundleProposal`](crate::bundle::BundleProposal) produced by a service
//! invocation points back at a `Run` via its `run_id`. The `Run` record is the
//! rollback handle: given a `run_id`, the storage layer can enumerate every
//! applied change and drive the shelf-store to restore originals.
//!
//! Runs are persisted in the default `tidyup-storage-sqlite` backend via the
//! [`RunLog`](crate::run::RunLog_ref) port.
//!
//! [`RunLog_ref`]: crate::run
//! (Trait lives in `tidyup-core::storage`; this module defines only data shapes.)

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::change::ParseError;

/// Semantic or extraction capability recorded with a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityKind {
    TextEmbedding,
    ImageEmbedding,
    AudioEmbedding,
    GenerativeText,
    VisionCaptioning,
    ContentExtractor,
    Ocr,
    /// Capability introduced by a newer writer and not understood by this
    /// binary. Retained so diagnostic provenance never blocks run recovery.
    #[serde(other)]
    Unknown,
}

impl CapabilityKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextEmbedding => "text-embedding",
            Self::ImageEmbedding => "image-embedding",
            Self::AudioEmbedding => "audio-embedding",
            Self::GenerativeText => "generative-text",
            Self::VisionCaptioning => "vision-captioning",
            Self::ContentExtractor => "content-extractor",
            Self::Ocr => "ocr",
            Self::Unknown => "unknown",
        }
    }
}

/// Whether a capability could participate in this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityStatus {
    Available,
    Unavailable,
    Disabled,
}

impl CapabilityStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Unavailable => "unavailable",
            Self::Disabled => "disabled",
        }
    }
}

/// One inspectable runtime capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEntry {
    pub kind: CapabilityKind,
    pub status: CapabilityStatus,
    pub implementation: String,
    pub model_id: Option<String>,
    pub dimensions: Option<usize>,
    pub requires_network: bool,
    pub detail: Option<String>,
}

/// Versioned manifest of the semantic facilities available to a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityManifest {
    pub schema_version: u32,
    pub entries: Vec<CapabilityEntry>,
}

impl Default for CapabilityManifest {
    fn default() -> Self {
        Self {
            schema_version: 1,
            entries: Vec::new(),
        }
    }
}

impl CapabilityManifest {
    /// Compact user-facing summary of primary semantic channels.
    #[must_use]
    pub fn summary(&self) -> String {
        let primary = [
            CapabilityKind::TextEmbedding,
            CapabilityKind::ImageEmbedding,
            CapabilityKind::AudioEmbedding,
            CapabilityKind::GenerativeText,
            CapabilityKind::Ocr,
        ];
        primary
            .iter()
            .filter_map(|kind| {
                self.entries
                    .iter()
                    .find(|entry| entry.kind == *kind)
                    .map(|entry| {
                        let identity = entry
                            .model_id
                            .as_deref()
                            .unwrap_or(entry.implementation.as_str());
                        format!("{}={} ({})", kind.as_str(), identity, entry.status.as_str())
                    })
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Which service produced this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunMode {
    Scan,
    Migrate,
}

impl RunMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "Scan",
            Self::Migrate => "Migrate",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "Scan" => Ok(Self::Scan),
            "Migrate" => Ok(Self::Migrate),
            other => Err(ParseError::UnknownRunMode(other.to_string())),
        }
    }
}

/// Terminal status of a run. `InProgress` is written at start and rewritten at
/// the end of the service call; `RolledBack` is set by the rollback service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunState {
    InProgress,
    Completed,
    RolledBack,
    Failed,
}

impl RunState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "InProgress",
            Self::Completed => "Completed",
            Self::RolledBack => "RolledBack",
            Self::Failed => "Failed",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "InProgress" => Ok(Self::InProgress),
            "Completed" => Ok(Self::Completed),
            "RolledBack" => Ok(Self::RolledBack),
            "Failed" => Ok(Self::Failed),
            other => Err(ParseError::UnknownRunState(other.to_string())),
        }
    }
}

/// One scan or migration invocation. Every applied change ties back to a run
/// record via `run_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: Uuid,
    pub mode: RunMode,
    /// Source root of the run. For scan mode this is the single directory
    /// being organised; for migration it is the source of the move.
    pub source_root: PathBuf,
    /// Target root for migration runs. `None` for scan mode.
    pub target_root: Option<PathBuf>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub state: RunState,
    /// Exact semantic/extraction capabilities available when this run began.
    #[serde(default)]
    pub capabilities: CapabilityManifest,
}

impl RunRecord {
    /// Start a new run record. The returned struct is ready to be persisted
    /// via the run log port; service code should update `state` + `completed_at`
    /// at the end of the call.
    pub fn begin(mode: RunMode, source_root: PathBuf, target_root: Option<PathBuf>) -> Self {
        Self {
            id: Uuid::new_v4(),
            mode,
            source_root,
            target_root,
            started_at: Utc::now(),
            completed_at: None,
            state: RunState::InProgress,
            capabilities: CapabilityManifest::default(),
        }
    }

    /// Attach the runtime capability manifest captured before work starts.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: CapabilityManifest) -> Self {
        self.capabilities = capabilities;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn run_mode_roundtrip() {
        for m in [RunMode::Scan, RunMode::Migrate] {
            assert_eq!(RunMode::parse(m.as_str()).unwrap(), m);
        }
        assert!(RunMode::parse("Nope").is_err());
    }

    #[test]
    fn run_state_roundtrip() {
        for s in [
            RunState::InProgress,
            RunState::Completed,
            RunState::RolledBack,
            RunState::Failed,
        ] {
            assert_eq!(RunState::parse(s.as_str()).unwrap(), s);
        }
    }

    #[test]
    fn begin_initialises_in_progress() {
        let r = RunRecord::begin(RunMode::Scan, PathBuf::from("/tmp/src"), None);
        assert_eq!(r.state, RunState::InProgress);
        assert!(r.completed_at.is_none());
        assert!(r.target_root.is_none());
    }

    #[test]
    fn run_record_serde_roundtrip() {
        let r = RunRecord::begin(
            RunMode::Migrate,
            PathBuf::from("/src"),
            Some(PathBuf::from("/target")),
        );
        let json = serde_json::to_string(&r).unwrap();
        let back: RunRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
