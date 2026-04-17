//! `ChangeLog` implementation on `SqliteStore`.
//!
//! Bundle writes are transactional: the bundle row and every member proposal land in a single
//! transaction. `pending` returns only loose proposals (`bundle_id IS NULL`); bundle members
//! are exposed exclusively through `pending_bundles`.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{params, Row, Transaction};
use tidyup_core::storage::ChangeLog;
use tidyup_domain::{BundleKind, BundleProposal, ChangeProposal, ChangeStatus, ChangeType, FileId};
use uuid::Uuid;

use crate::SqliteStore;

const CHANGE_COLS: &str = "id, file_id, change_type, original_path, proposed_path, \
     proposed_name, confidence, reasoning, needs_review, status, created_at, applied_at, \
     bundle_id, classification_confidence, rename_mismatch_score";

const BUNDLE_COLS: &str = "id, root, kind, target_parent, status, reasoning, confidence, \
     created_at, applied_at";

fn parse_uuid(s: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn parse_domain<T, E: std::error::Error + Send + Sync + 'static>(
    parsed: std::result::Result<T, E>,
) -> rusqlite::Result<T> {
    parsed.map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

#[allow(clippy::cast_possible_truncation)]
const fn f64_to_f32(v: f64) -> f32 {
    v as f32
}

fn row_to_proposal(row: &Row<'_>) -> rusqlite::Result<ChangeProposal> {
    let id = parse_uuid(&row.get::<_, String>("id")?)?;
    let file_id = row
        .get::<_, Option<String>>("file_id")?
        .map(|s| parse_uuid(&s))
        .transpose()?
        .map(FileId);
    let change_type = parse_domain(ChangeType::parse(&row.get::<_, String>("change_type")?))?;
    let status = parse_domain(ChangeStatus::parse(&row.get::<_, String>("status")?))?;
    let bundle_id = row
        .get::<_, Option<String>>("bundle_id")?
        .map(|s| parse_uuid(&s))
        .transpose()?;
    let confidence = f64_to_f32(row.get::<_, f64>("confidence")?);
    let classification_confidence = row
        .get::<_, Option<f64>>("classification_confidence")?
        .map(f64_to_f32);
    let rename_mismatch_score = row
        .get::<_, Option<f64>>("rename_mismatch_score")?
        .map(f64_to_f32);
    let needs_review: i64 = row.get("needs_review")?;
    Ok(ChangeProposal {
        id,
        file_id,
        change_type,
        original_path: PathBuf::from(row.get::<_, String>("original_path")?),
        proposed_path: PathBuf::from(row.get::<_, String>("proposed_path")?),
        proposed_name: row.get("proposed_name")?,
        confidence,
        reasoning: row.get("reasoning")?,
        needs_review: needs_review != 0,
        status,
        created_at: row.get::<_, DateTime<Utc>>("created_at")?,
        applied_at: row.get::<_, Option<DateTime<Utc>>>("applied_at")?,
        bundle_id,
        classification_confidence,
        rename_mismatch_score,
    })
}

fn row_to_bundle(row: &Row<'_>) -> rusqlite::Result<BundleProposal> {
    let id = parse_uuid(&row.get::<_, String>("id")?)?;
    let kind: BundleKind = parse_domain(serde_json::from_str(&row.get::<_, String>("kind")?))?;
    let status = parse_domain(ChangeStatus::parse(&row.get::<_, String>("status")?))?;
    let confidence = f64_to_f32(row.get::<_, f64>("confidence")?);
    Ok(BundleProposal {
        id,
        root: PathBuf::from(row.get::<_, String>("root")?),
        kind,
        target_parent: PathBuf::from(row.get::<_, String>("target_parent")?),
        members: Vec::new(),
        confidence,
        reasoning: row.get("reasoning")?,
        status,
        created_at: row.get::<_, DateTime<Utc>>("created_at")?,
        applied_at: row.get::<_, Option<DateTime<Utc>>>("applied_at")?,
    })
}

fn path_str(p: &std::path::Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", p.display()))
}

fn insert_proposal(tx: &Transaction<'_>, p: &ChangeProposal) -> Result<()> {
    tx.execute(
        &format!(
            "INSERT INTO change_proposals ({CHANGE_COLS}) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)"
        ),
        params![
            p.id.to_string(),
            p.file_id.as_ref().map(|f| f.0.to_string()),
            p.change_type.as_str(),
            path_str(&p.original_path)?,
            path_str(&p.proposed_path)?,
            p.proposed_name,
            f64::from(p.confidence),
            p.reasoning,
            i64::from(p.needs_review),
            p.status.as_str(),
            p.created_at,
            p.applied_at,
            p.bundle_id.map(|b| b.to_string()),
            p.classification_confidence.map(f64::from),
            p.rename_mismatch_score.map(f64::from),
        ],
    )
    .context("inserting change proposal")?;
    Ok(())
}

#[async_trait]
impl ChangeLog for SqliteStore {
    async fn record_proposal(&self, proposal: &ChangeProposal) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let proposal = proposal.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            if proposal.bundle_id.is_some() {
                return Err(anyhow!(
                    "record_proposal rejects bundle members; use record_bundle instead"
                ));
            }
            {
                let mut guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let tx = guard.transaction()?;
                insert_proposal(&tx, &proposal)?;
                tx.commit()?;
            }
            Ok(())
        })
        .await
        .context("join record_proposal task")??;
        Ok(())
    }

    async fn mark_applied(&self, proposal_id: Uuid) -> tidyup_core::Result<()> {
        let conn = self.conn();
        tokio::task::spawn_blocking(move || -> Result<()> {
            {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        "UPDATE change_proposals SET status = ?1, applied_at = ?2 WHERE id = ?3",
                        params![
                            ChangeStatus::Applied.as_str(),
                            Utc::now(),
                            proposal_id.to_string()
                        ],
                    )
                    .context("marking proposal applied")?;
            }
            Ok(())
        })
        .await
        .context("join mark_applied task")??;
        Ok(())
    }

    async fn pending(&self) -> tidyup_core::Result<Vec<ChangeProposal>> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<ChangeProposal>> {
            let rows = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt = guard.prepare(&format!(
                    "SELECT {CHANGE_COLS} FROM change_proposals \
                     WHERE bundle_id IS NULL AND status = ?1 ORDER BY created_at"
                ))?;
                let fetched: Vec<ChangeProposal> = stmt
                    .query_map(params![ChangeStatus::Pending.as_str()], row_to_proposal)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                fetched
            };
            Ok(rows)
        })
        .await
        .context("join pending task")??;
        Ok(result)
    }

    async fn record_bundle(&self, bundle: &BundleProposal) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let bundle = bundle.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let kind_json =
                serde_json::to_string(&bundle.kind).context("encoding BundleKind to JSON")?;
            {
                let mut guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let tx = guard.transaction()?;
                tx.execute(
                    &format!(
                        "INSERT INTO bundles ({BUNDLE_COLS}) VALUES \
                         (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
                    ),
                    params![
                        bundle.id.to_string(),
                        path_str(&bundle.root)?,
                        kind_json,
                        path_str(&bundle.target_parent)?,
                        bundle.status.as_str(),
                        bundle.reasoning,
                        f64::from(bundle.confidence),
                        bundle.created_at,
                        bundle.applied_at,
                    ],
                )
                .context("inserting bundle")?;
                for member in &bundle.members {
                    insert_proposal(&tx, member)?;
                }
                tx.commit()?;
            }
            Ok(())
        })
        .await
        .context("join record_bundle task")??;
        Ok(())
    }

    async fn mark_bundle_applied(&self, bundle_id: Uuid) -> tidyup_core::Result<()> {
        let conn = self.conn();
        tokio::task::spawn_blocking(move || -> Result<()> {
            {
                let mut guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let tx = guard.transaction()?;
                let now = Utc::now();
                let applied = ChangeStatus::Applied.as_str();
                let id_str = bundle_id.to_string();
                tx.execute(
                    "UPDATE bundles SET status = ?1, applied_at = ?2 WHERE id = ?3",
                    params![applied, now, id_str],
                )
                .context("marking bundle applied")?;
                tx.execute(
                    "UPDATE change_proposals SET status = ?1, applied_at = ?2 WHERE bundle_id = ?3",
                    params![applied, now, id_str],
                )
                .context("marking bundle members applied")?;
                tx.commit()?;
            }
            Ok(())
        })
        .await
        .context("join mark_bundle_applied task")??;
        Ok(())
    }

    async fn pending_bundles(&self) -> tidyup_core::Result<Vec<BundleProposal>> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<BundleProposal>> {
            let bundles = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt = guard.prepare(&format!(
                    "SELECT {BUNDLE_COLS} FROM bundles WHERE status = ?1 ORDER BY created_at"
                ))?;
                let mut bundles: Vec<BundleProposal> = stmt
                    .query_map(params![ChangeStatus::Pending.as_str()], row_to_bundle)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;

                let mut member_stmt = guard.prepare(&format!(
                    "SELECT {CHANGE_COLS} FROM change_proposals \
                     WHERE bundle_id = ?1 ORDER BY original_path"
                ))?;
                for bundle in &mut bundles {
                    let members = member_stmt
                        .query_map(params![bundle.id.to_string()], row_to_proposal)?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    bundle.members = members;
                }
                bundles
            };
            Ok(bundles)
        })
        .await
        .context("join pending_bundles task")??;
        Ok(result)
    }
}
