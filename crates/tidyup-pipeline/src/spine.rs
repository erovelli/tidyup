//! Shared building blocks for the scan and migration classification spines.

use std::path::Path;

use anyhow::Result;
use chrono::Utc;
use tidyup_core::inference::{EmbeddingBackend, FileModality, TextBackend};
use tidyup_domain::change::{ChangeProposal, ChangeStatus, ChangeType};
use tidyup_domain::ClassifierConfig;
use uuid::Uuid;

use crate::indexing::SourceIdentities;
use crate::naming::{propose_rename, RenameProposal};
use crate::yake;

pub use crate::semantic::cosine;

pub fn normalize_semantic_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect()
}

/// Classify through the optional LLM stage and return its dense re-embedding query.
pub async fn tier3_rerank(
    text_backend: &dyn TextBackend,
    text: &str,
    filename: &str,
) -> Result<Option<(String, String)>> {
    let classification = text_backend
        .classify_text(crate::text_util::char_prefix(text, 12_000), filename)
        .await?;
    let model_id = text_backend.model_id().to_string();
    let query = build_llm_query(&classification);
    Ok((!query.is_empty()).then_some((query, model_id)))
}

pub fn build_llm_query(c: &tidyup_core::inference::ContentClassification) -> String {
    let mut parts = Vec::with_capacity(3);
    if !c.category.is_empty() {
        parts.push(c.category.clone());
    }
    if !c.tags.is_empty() {
        parts.push(c.tags.join(" "));
    }
    if !c.summary.is_empty() {
        parts.push(c.summary.clone());
    }
    parts.join(" ")
}

pub fn file_modality(mime: Option<&str>) -> FileModality {
    if let Some(mime) = mime {
        if mime.starts_with("image/") {
            return FileModality::Image;
        }
        if mime.starts_with("audio/") {
            return FileModality::Audio;
        }
        if mime.starts_with("video/") {
            return FileModality::Video;
        }
        if mime.starts_with("text/") || mime == "application/pdf" {
            return FileModality::Text;
        }
    }
    FileModality::Text
}

#[derive(Debug)]
pub struct GatedRename {
    pub proposal: RenameProposal,
    pub mismatch_score: Option<f32>,
}

#[allow(clippy::too_many_arguments)]
pub async fn gate_rename(
    path: &Path,
    metadata: &serde_json::Value,
    keywords: &[yake::Keyword],
    year: Option<i32>,
    classification_confidence: f32,
    embeddings: &dyn EmbeddingBackend,
    content_text: Option<&str>,
    filename: &str,
    config: &ClassifierConfig,
) -> Result<GatedRename> {
    let proposal = propose_rename(path, metadata, keywords, year);
    if matches!(proposal, RenameProposal::Keep)
        || classification_confidence < config.rename.min_classification_confidence
    {
        return Ok(GatedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: None,
        });
    }
    let Some(content_text) = content_text else {
        return Ok(GatedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: None,
        });
    };
    let mismatch = 1.0
        - cosine(
            &embeddings.embed_text(filename).await?,
            &embeddings.embed_text(content_text).await?,
        );
    if mismatch < config.rename.min_mismatch_score {
        return Ok(GatedRename {
            proposal: RenameProposal::Keep,
            mismatch_score: Some(mismatch),
        });
    }
    Ok(GatedRename {
        proposal,
        mismatch_score: Some(mismatch),
    })
}

pub fn find_year(value: &str) -> Option<i32> {
    crate::text_util::find_year(value)
}

#[allow(clippy::too_many_arguments)]
pub fn build_proposal(
    source: &Path,
    destination_dir: &Path,
    rename: &RenameProposal,
    confidence: f32,
    reasoning: String,
    needs_review: bool,
    classification_confidence: Option<f32>,
    rename_mismatch_score: Option<f32>,
    identities: &SourceIdentities,
) -> ChangeProposal {
    let proposed_name = match rename {
        RenameProposal::Rename { name, .. } => name.clone(),
        RenameProposal::Keep => source
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string(),
    };
    ChangeProposal {
        id: Uuid::new_v4(),
        file_id: identities.file_id(source),
        change_type: if matches!(rename, RenameProposal::Rename { .. }) {
            ChangeType::RenameAndMove
        } else {
            ChangeType::Move
        },
        original_path: source.to_path_buf(),
        proposed_path: destination_dir.join(&proposed_name),
        proposed_name,
        confidence,
        reasoning,
        needs_review,
        status: ChangeStatus::Pending,
        created_at: Utc::now(),
        applied_at: None,
        bundle_id: None,
        classification_confidence,
        rename_mismatch_score,
        content_hash: identities.content_hash(source),
    }
}

#[cfg(test)]
mod tests {
    use super::cosine;

    #[test]
    fn cosine_rejects_mismatched_vector_lengths() {
        assert!(cosine(&[1.0, 1.0], &[1.0]).abs() < f32::EPSILON);
    }
}
