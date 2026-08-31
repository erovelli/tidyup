//! Bundle aggregate — a group of files that must move atomically.
//!
//! Coding projects, photo bursts, music albums, and similar structures lose meaning when
//! fragmented. Bundle detection (in the pipeline) marks a subtree as opaque and emits a
//! `BundleProposal` that flows through the same review/apply flow as loose `ChangeProposal`s.
//!
//! Invariants (enforced by the constructor):
//! - A bundle has at least one member.
//! - Every member carries `bundle_id == Some(bundle.id)` (stamped on construction).
//! - Structural bundle members preserve their names. Semantic collections may carry
//!   evidence-backed `RenameAndMove` proposals for loose members.
//! - Rename scores are accepted only for semantic collections.
//!
//! Individual member proposals are never approved, applied, or rolled back independently of
//! their bundle. See `CLAUDE.md` → "Bundle detection and atomicity".

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::change::{ChangeProposal, ChangeStatus, ChangeType, ParseError};

/// Kind of bundle detected. The pattern-bearing variant (`DocumentSeries`) carries the regex
/// or glob that clustered the members.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BundleKind {
    /// A generalized, non-overlapping directory boundary selected from the
    /// source hierarchy. Marker files may appear in its provenance, but they
    /// never determine this kind or its destination.
    DirectoryEnvelope,
    GitRepository,
    NodeProject,
    RustCrate,
    PythonProject,
    XcodeProject,
    AndroidStudioProject,
    JupyterNotebookSet,
    PhotoBurst,
    MusicAlbum,
    DocumentSeries {
        pattern: String,
    },
    /// Loose, cross-format artifacts linked by filename/content/session evidence.
    SemanticCollection {
        label: String,
    },
    Generic,
}

impl BundleKind {
    /// Stable discriminator string for persistence. Parameterised variants drop their payload.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::DirectoryEnvelope => "DirectoryEnvelope",
            Self::GitRepository => "GitRepository",
            Self::NodeProject => "NodeProject",
            Self::RustCrate => "RustCrate",
            Self::PythonProject => "PythonProject",
            Self::XcodeProject => "XcodeProject",
            Self::AndroidStudioProject => "AndroidStudioProject",
            Self::JupyterNotebookSet => "JupyterNotebookSet",
            Self::PhotoBurst => "PhotoBurst",
            Self::MusicAlbum => "MusicAlbum",
            Self::DocumentSeries { .. } => "DocumentSeries",
            Self::SemanticCollection { .. } => "SemanticCollection",
            Self::Generic => "Generic",
        }
    }

    /// Parse the discriminator produced by `as_str`. Payload-carrying variants use an empty
    /// payload because only the stable discriminator is on the wire; callers may rehydrate it.
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "DirectoryEnvelope" => Ok(Self::DirectoryEnvelope),
            "GitRepository" => Ok(Self::GitRepository),
            "NodeProject" => Ok(Self::NodeProject),
            "RustCrate" => Ok(Self::RustCrate),
            "PythonProject" => Ok(Self::PythonProject),
            "XcodeProject" => Ok(Self::XcodeProject),
            "AndroidStudioProject" => Ok(Self::AndroidStudioProject),
            "JupyterNotebookSet" => Ok(Self::JupyterNotebookSet),
            "PhotoBurst" => Ok(Self::PhotoBurst),
            "MusicAlbum" => Ok(Self::MusicAlbum),
            "SemanticCollection" => Ok(Self::SemanticCollection {
                label: String::new(),
            }),
            "Generic" => Ok(Self::Generic),
            other => Err(ParseError::UnknownBundleKind(other.to_string())),
        }
    }

    /// Whether this kind moves as a **set of individual files** (clustered loose
    /// siblings) rather than by renaming a single root directory.
    ///
    /// Photo bursts, music albums, and document series have no shared directory
    /// to relocate — their members move individually, but still all-or-nothing.
    /// The executor and rollback consult this to pick the atomic-apply strategy:
    /// `false` => rename the bundle root directory; `true` => move each member
    /// to its own `proposed_path`, rolling all back on any failure.
    #[must_use]
    pub const fn moves_as_file_set(&self) -> bool {
        matches!(
            self,
            Self::PhotoBurst
                | Self::MusicAlbum
                | Self::DocumentSeries { .. }
                | Self::SemanticCollection { .. }
        )
    }

    /// Whether loose members may be renamed while the collection is still
    /// reviewed and applied atomically.
    #[must_use]
    pub const fn allows_member_renames(&self) -> bool {
        matches!(self, Self::SemanticCollection { .. })
    }
}

/// Boundary confidence for a generalized directory envelope.
///
/// A cohesive directory is eligible for normal, confidence-gated application.
/// An uncertain directory is kept intact for safety, but must remain in review
/// until the boundary model has stronger evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectoryBoundary {
    Cohesive,
    Uncertain,
}

/// A deterministic snapshot of an envelope at planning time.
///
/// The executor recomputes this before applying a directory-root operation.
/// It intentionally includes directories and symlinks, not only regular files,
/// so empty folders and link topology participate in integrity checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectorySnapshot {
    /// Stable BLAKE3 digest over every relative entry and its identity token.
    pub digest: String,
    pub regular_files: u64,
    pub directories: u64,
    pub symlinks: u64,
    pub total_bytes: u64,
    /// `false` means discovery encountered an unreadable or unsupported entry.
    /// Such an envelope must not be applied automatically.
    pub complete: bool,
}

/// Audit and safety metadata for a [`BundleKind::DirectoryEnvelope`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectoryEnvelope {
    pub boundary: DirectoryBoundary,
    /// Aggregate affinity in the source hierarchy. This is deliberately
    /// independent of destination-routing confidence.
    pub cohesion: f32,
    pub snapshot: DirectorySnapshot,
    /// Machine-readable/plain-language evidence retained for review/history.
    pub evidence: Vec<String>,
    /// Descriptive structural observations, including recognized markers.
    pub provenance: Vec<String>,
    /// Whether this envelope is held for interactive review even when a caller
    /// requested bulk approval.
    pub requires_review: bool,
}

/// Atomic move proposal for a detected bundle. Either every member applies, or none do.
///
/// Status transitions mirror `ChangeStatus`: `Pending → Approved → Applied`, with `Rejected`
/// and `Unshelved` (rollback) as terminal states. Member status is never diverged from the
/// bundle's — treating members independently is a bug.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleProposal {
    pub id: Uuid,
    /// Source bundle root (e.g. the directory containing `.git/`, `Cargo.toml`, etc.).
    pub root: PathBuf,
    pub kind: BundleKind,
    /// Destination *parent* directory. The bundle root is moved under this path.
    pub target_parent: PathBuf,
    /// Per-file moves within the bundle. Each carries `bundle_id == Some(self.id)`.
    pub members: Vec<ChangeProposal>,
    pub confidence: f32,
    pub reasoning: String,
    pub status: ChangeStatus,
    pub created_at: DateTime<Utc>,
    pub applied_at: Option<DateTime<Utc>>,
    /// Present only for generalized directory envelopes. Kept optional so
    /// persisted marker and file-set bundles remain backwards compatible.
    #[serde(default)]
    pub envelope: Option<DirectoryEnvelope>,
}

/// Construction-time invariant violations.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("bundle must have at least one member")]
    Empty,
    #[error("directory-envelope metadata may only be attached to a DirectoryEnvelope bundle")]
    EnvelopeKind,
    #[error("bundle member has change_type {actual}; this bundle kind preserves member names")]
    MemberNotMove { actual: &'static str },
    #[error(
        "bundle member carries a rename_mismatch_score; this bundle kind preserves member names"
    )]
    MemberHasRenameScore,
}

impl BundleProposal {
    /// Build a bundle, stamping each member's `bundle_id` to the new bundle's id.
    ///
    /// # Errors
    /// Returns [`BundleError`] if members are empty, any member has a non-Move `change_type`,
    /// or any member carries a `rename_mismatch_score`.
    pub fn new(
        root: PathBuf,
        kind: BundleKind,
        target_parent: PathBuf,
        members: Vec<ChangeProposal>,
        confidence: f32,
        reasoning: String,
    ) -> Result<Self, BundleError> {
        if members.is_empty() && !matches!(kind, BundleKind::DirectoryEnvelope) {
            return Err(BundleError::Empty);
        }
        let id = Uuid::new_v4();
        let mut stamped = Vec::with_capacity(members.len());
        for mut member in members {
            let valid_change = member.change_type == ChangeType::Move
                || (kind.allows_member_renames()
                    && member.change_type == ChangeType::RenameAndMove);
            if !valid_change {
                return Err(BundleError::MemberNotMove {
                    actual: member.change_type.as_str(),
                });
            }
            if member.rename_mismatch_score.is_some() && !kind.allows_member_renames() {
                return Err(BundleError::MemberHasRenameScore);
            }
            member.bundle_id = Some(id);
            stamped.push(member);
        }
        Ok(Self {
            id,
            root,
            kind,
            target_parent,
            members: stamped,
            confidence,
            reasoning,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            envelope: None,
        })
    }

    /// Attach the immutable boundary snapshot and evidence for a generalized
    /// directory envelope.
    ///
    /// # Errors
    /// Returns [`BundleError::EnvelopeKind`] when called for any other bundle
    /// shape, preventing marker/file-set proposals from masquerading as a
    /// hierarchy-selected envelope.
    pub fn with_envelope(mut self, envelope: DirectoryEnvelope) -> Result<Self, BundleError> {
        if !matches!(self.kind, BundleKind::DirectoryEnvelope) {
            return Err(BundleError::EnvelopeKind);
        }
        self.envelope = Some(envelope);
        Ok(self)
    }

    /// Whether this proposal is intentionally held from non-interactive bulk
    /// approval because its boundary evidence is incomplete or uncertain.
    #[must_use]
    pub fn requires_explicit_review(&self) -> bool {
        self.envelope
            .as_ref()
            .is_some_and(|envelope| envelope.requires_review || !envelope.snapshot.complete)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::change::ChangeStatus;
    use crate::file::FileId;

    fn sample_member(name: &str) -> ChangeProposal {
        ChangeProposal {
            id: Uuid::new_v4(),
            file_id: Some(FileId(Uuid::new_v4())),
            change_type: ChangeType::Move,
            original_path: PathBuf::from(format!("/src/proj/{name}")),
            proposed_path: PathBuf::from(format!("/code/proj/{name}")),
            proposed_name: name.to_string(),
            confidence: 0.92,
            reasoning: "bundle member".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: None,
            rename_mismatch_score: None,
            content_hash: None,
        }
    }

    #[test]
    fn bundle_kind_roundtrip() {
        let kinds = [
            BundleKind::GitRepository,
            BundleKind::NodeProject,
            BundleKind::RustCrate,
            BundleKind::PythonProject,
            BundleKind::XcodeProject,
            BundleKind::AndroidStudioProject,
            BundleKind::JupyterNotebookSet,
            BundleKind::DirectoryEnvelope,
            BundleKind::PhotoBurst,
            BundleKind::MusicAlbum,
            BundleKind::SemanticCollection {
                label: String::new(),
            },
            BundleKind::Generic,
        ];
        for k in kinds {
            let s = k.as_str();
            let back = BundleKind::parse(s).unwrap();
            assert_eq!(k, back);
        }
    }

    #[test]
    fn moves_as_file_set_only_for_content_clusters() {
        assert!(BundleKind::PhotoBurst.moves_as_file_set());
        assert!(BundleKind::MusicAlbum.moves_as_file_set());
        assert!(BundleKind::DocumentSeries {
            pattern: "invoice".into()
        }
        .moves_as_file_set());
        assert!(BundleKind::SemanticCollection {
            label: "atoms not electrons".into()
        }
        .moves_as_file_set());
        // Marker (directory) bundles move by renaming their root.
        assert!(!BundleKind::RustCrate.moves_as_file_set());
        assert!(!BundleKind::GitRepository.moves_as_file_set());
        assert!(!BundleKind::JupyterNotebookSet.moves_as_file_set());
        assert!(!BundleKind::DirectoryEnvelope.moves_as_file_set());
        assert!(!BundleKind::Generic.moves_as_file_set());
    }

    #[test]
    fn bundle_kind_serde_roundtrip_with_pattern() {
        let k = BundleKind::DocumentSeries {
            pattern: r"invoice-\d{4}-\d{2}\.pdf".to_string(),
        };
        let json = serde_json::to_string(&k).unwrap();
        let back: BundleKind = serde_json::from_str(&json).unwrap();
        assert_eq!(k, back);
    }

    #[test]
    fn bundle_proposal_serde_roundtrip() {
        let bundle = BundleProposal::new(
            PathBuf::from("/src/proj"),
            BundleKind::RustCrate,
            PathBuf::from("/code"),
            vec![sample_member("main.rs"), sample_member("lib.rs")],
            0.88,
            "Detected Cargo.toml at root".to_string(),
        )
        .unwrap();
        let json = serde_json::to_string(&bundle).unwrap();
        let back: BundleProposal = serde_json::from_str(&json).unwrap();
        assert_eq!(bundle, back);
    }

    #[test]
    fn constructor_stamps_bundle_id_on_members() {
        let bundle = BundleProposal::new(
            PathBuf::from("/src/proj"),
            BundleKind::RustCrate,
            PathBuf::from("/code"),
            vec![
                sample_member("a.rs"),
                sample_member("b.rs"),
                sample_member("c.rs"),
            ],
            0.9,
            "bundle".to_string(),
        )
        .unwrap();
        for member in &bundle.members {
            assert_eq!(member.bundle_id, Some(bundle.id));
        }
    }

    #[test]
    fn constructor_rejects_empty_members() {
        let err = BundleProposal::new(
            PathBuf::from("/src/proj"),
            BundleKind::Generic,
            PathBuf::from("/code"),
            vec![],
            0.5,
            "empty".to_string(),
        )
        .unwrap_err();
        assert!(matches!(err, BundleError::Empty));
    }

    #[test]
    fn directory_envelope_allows_empty_directory() {
        let bundle = BundleProposal::new(
            PathBuf::from("/src/empty"),
            BundleKind::DirectoryEnvelope,
            PathBuf::from("/target"),
            vec![],
            0.6,
            "hierarchy boundary".to_string(),
        )
        .unwrap();
        assert!(bundle.members.is_empty());
    }

    #[test]
    fn constructor_rejects_non_move_member() {
        let mut bad = sample_member("renamed.rs");
        bad.change_type = ChangeType::RenameAndMove;
        let err = BundleProposal::new(
            PathBuf::from("/src/proj"),
            BundleKind::RustCrate,
            PathBuf::from("/code"),
            vec![sample_member("ok.rs"), bad],
            0.9,
            "bundle".to_string(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            BundleError::MemberNotMove {
                actual: "RenameAndMove"
            }
        ));
    }

    #[test]
    fn semantic_collection_accepts_member_rename() {
        let mut renamed = sample_member("screenshot.png");
        renamed.change_type = ChangeType::RenameAndMove;
        renamed.proposed_name = "submission-confirmation.png".to_string();
        renamed.proposed_path = PathBuf::from("/projects/atoms/submission-confirmation.png");
        renamed.rename_mismatch_score = Some(0.9);
        let bundle = BundleProposal::new(
            PathBuf::from("/src"),
            BundleKind::SemanticCollection {
                label: "atoms".to_string(),
            },
            PathBuf::from("/projects"),
            vec![renamed],
            0.9,
            "shared entity stem".to_string(),
        )
        .unwrap();
        assert_eq!(bundle.members[0].change_type, ChangeType::RenameAndMove);
    }

    #[test]
    fn constructor_rejects_member_with_rename_score() {
        let mut bad = sample_member("a.rs");
        bad.rename_mismatch_score = Some(0.7);
        let err = BundleProposal::new(
            PathBuf::from("/src/proj"),
            BundleKind::RustCrate,
            PathBuf::from("/code"),
            vec![bad],
            0.9,
            "bundle".to_string(),
        )
        .unwrap_err();
        assert!(matches!(err, BundleError::MemberHasRenameScore));
    }
}
