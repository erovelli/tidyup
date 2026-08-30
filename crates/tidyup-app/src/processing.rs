//! Run-scoped processing-state reconciliation shared by scan and migration.

// These helpers intentionally remain crate-visible through a private sibling
// module; exposing the bookkeeping surface as application API would be wider.
#![allow(clippy::redundant_pub_crate)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use chrono::Utc;
use tidyup_core::frontend::{ProgressItem, ProgressReporter};
use tidyup_core::storage::FileIndex;
use tidyup_domain::{
    BundleProposal, ChangeProposal, FileId, FileProcessingRecord, FileProcessingRole,
    FileProcessingStage, FileProcessingState, IndexedFile,
};
use uuid::Uuid;

/// Emit the completed identity-index pass through the shared progress port.
/// The dedicated UI indexed counter consumes only `Phase::Indexing` items, so
/// later classification progress cannot overwrite the durable count.
pub(crate) async fn report_indexing(
    summary: &tidyup_pipeline::indexing::IndexSummary,
    progress: &dyn ProgressReporter,
) {
    let total = u64::try_from(summary.indexed.len().saturating_add(summary.failed)).ok();
    progress
        .phase_started(tidyup_domain::Phase::Indexing, total)
        .await;
    for (position, file) in summary.indexed.iter().enumerate() {
        progress
            .item_completed(
                tidyup_domain::Phase::Indexing,
                ProgressItem {
                    label: file.path.display().to_string(),
                    current: u64::try_from(position.saturating_add(1)).unwrap_or(u64::MAX),
                    total,
                },
            )
            .await;
    }
    progress
        .phase_finished(tidyup_domain::Phase::Indexing)
        .await;
}

/// Attach canonical index identities to proposals before they cross the
/// review or persistence boundary.
pub(crate) fn attach_indexed_identities(
    indexed: &[IndexedFile],
    proposals: &mut [ChangeProposal],
    bundles: &mut [BundleProposal],
) {
    let ids: HashMap<&std::path::Path, &FileId> = indexed
        .iter()
        .map(|file| (file.path.as_path(), &file.id))
        .collect();
    for proposal in proposals.iter_mut().chain(
        bundles
            .iter_mut()
            .flat_map(|bundle| bundle.members.iter_mut()),
    ) {
        if let Some(id) = ids.get(proposal.original_path.as_path()) {
            proposal.file_id = Some((*id).clone());
        }
    }
}

/// Persist the terminal planning result for every successfully indexed source.
pub(crate) async fn record_source_outcomes(
    index: &dyn FileIndex,
    run_id: Uuid,
    indexed: &[IndexedFile],
    proposals: &[ChangeProposal],
    bundles: &[BundleProposal],
    unclassified: &[PathBuf],
    already_in_place: &[PathBuf],
) -> tidyup_core::Result<()> {
    let classified: HashSet<&std::path::Path> = proposals
        .iter()
        .map(|proposal| proposal.original_path.as_path())
        .chain(
            bundles
                .iter()
                .flat_map(|bundle| bundle.members.iter())
                .map(|member| member.original_path.as_path()),
        )
        .collect();
    let unclassified: HashSet<&std::path::Path> =
        unclassified.iter().map(PathBuf::as_path).collect();
    let already_in_place: HashSet<&std::path::Path> =
        already_in_place.iter().map(PathBuf::as_path).collect();

    let records = indexed
        .iter()
        .map(|file| {
            let (lifecycle_step, lifecycle_status, reason) =
                if classified.contains(file.path.as_path()) {
                    (
                        FileProcessingStage::Planning,
                        FileProcessingState::Classified,
                        Some("included in the reviewable organization plan".to_string()),
                    )
                } else if already_in_place.contains(file.path.as_path()) {
                    // Classified successfully; the answer was "stay put". This
                    // must not fall through to `Indexed`, which would report a
                    // correctly filed file as excluded from planning.
                    (
                        FileProcessingStage::Planning,
                        FileProcessingState::Classified,
                        Some("already at its classified destination; no move needed".to_string()),
                    )
                } else if unclassified.contains(file.path.as_path()) {
                    (
                        FileProcessingStage::Classification,
                        FileProcessingState::Unclassified,
                        Some("no destination cleared the semantic decision gates".to_string()),
                    )
                } else {
                    (
                        FileProcessingStage::Indexing,
                        FileProcessingState::Indexed,
                        Some("identity indexed; excluded from semantic planning".to_string()),
                    )
                };
            FileProcessingRecord {
                run_id,
                path: file.path.clone(),
                file_id: Some(file.id.clone()),
                role: FileProcessingRole::Source,
                stage: lifecycle_step,
                state: lifecycle_status,
                reason,
                updated_at: Utc::now(),
            }
        })
        .collect::<Vec<_>>();
    index.put_processing_records(&records).await
}

/// Mark target identities as successfully admitted to hierarchy profiling.
pub(crate) async fn record_target_profiled(
    index: &dyn FileIndex,
    run_id: Uuid,
    indexed: &[IndexedFile],
) -> tidyup_core::Result<()> {
    let records = indexed
        .iter()
        .map(|file| FileProcessingRecord {
            run_id,
            path: file.path.clone(),
            file_id: Some(file.id.clone()),
            role: FileProcessingRole::TargetProfile,
            stage: FileProcessingStage::Profiling,
            state: FileProcessingState::Indexed,
            reason: Some("available as evidence for the learned target hierarchy".to_string()),
            updated_at: Utc::now(),
        })
        .collect::<Vec<_>>();
    index.put_processing_records(&records).await
}
