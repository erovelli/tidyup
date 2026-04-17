//! Bundle aggregate — a group of files that must move atomically.
//!
//! Coding projects, photo bursts, music albums, and similar structures lose meaning when
//! fragmented. Bundle detection (in the pipeline) marks a subtree as opaque and emits a
//! `BundleProposal` that flows through the same review/apply flow as loose `ChangeProposal`s.
//!
//! Invariants (enforced by the constructor):
//! - A bundle has at least one member.
//! - Every member carries `bundle_id == Some(bundle.id)` (stamped on construction).
//! - Every member has `change_type == ChangeType::Move`; bundle members never receive rename
//!   proposals.
//! - No member carries a `rename_mismatch_score` — rename signals are meaningless for members.
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
    GitRepository,
    NodeProject,
    RustCrate,
    PythonProject,
    XcodeProject,
    AndroidStudioProject,
    JupyterNotebookSet,
    PhotoBurst,
    MusicAlbum,
    DocumentSeries { pattern: String },
    Generic,
}

impl BundleKind {
    /// Stable discriminator string for persistence. Parameterised variants drop their payload.
    pub const fn as_str(&self) -> &'static str {
        match self {
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
            Self::Generic => "Generic",
        }
    }

    /// Parse the discriminator produced by `as_str`. Payload-carrying variants require explicit
    /// rehydration by the caller (only the discriminator is on the wire).
    pub fn parse(s: &str) -> Result<Self, ParseError> {
        match s {
            "GitRepository" => Ok(Self::GitRepository),
            "NodeProject" => Ok(Self::NodeProject),
            "RustCrate" => Ok(Self::RustCrate),
            "PythonProject" => Ok(Self::PythonProject),
            "XcodeProject" => Ok(Self::XcodeProject),
            "AndroidStudioProject" => Ok(Self::AndroidStudioProject),
            "JupyterNotebookSet" => Ok(Self::JupyterNotebookSet),
            "PhotoBurst" => Ok(Self::PhotoBurst),
            "MusicAlbum" => Ok(Self::MusicAlbum),
            "Generic" => Ok(Self::Generic),
            other => Err(ParseError::UnknownBundleKind(other.to_string())),
        }
    }
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
}

/// Construction-time invariant violations.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("bundle must have at least one member")]
    Empty,
    #[error(
        "bundle member has change_type {actual}; bundles only allow Move (rename suggestions \
         are never generated for bundle members)"
    )]
    MemberNotMove { actual: &'static str },
    #[error(
        "bundle member carries a rename_mismatch_score; rename signals are meaningless for \
         bundle members"
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
        if members.is_empty() {
            return Err(BundleError::Empty);
        }
        let id = Uuid::new_v4();
        let mut stamped = Vec::with_capacity(members.len());
        for mut member in members {
            if member.change_type != ChangeType::Move {
                return Err(BundleError::MemberNotMove {
                    actual: member.change_type.as_str(),
                });
            }
            if member.rename_mismatch_score.is_some() {
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
        })
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
            BundleKind::PhotoBurst,
            BundleKind::MusicAlbum,
            BundleKind::Generic,
        ];
        for k in kinds {
            let s = k.as_str();
            let back = BundleKind::parse(s).unwrap();
            assert_eq!(k, back);
        }
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
