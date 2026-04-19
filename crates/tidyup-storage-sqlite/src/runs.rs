//! `RunLog` implementation — persists scan/migration run records.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{params, Row};
use tidyup_core::storage::RunLog;
use tidyup_domain::{RunMode, RunRecord, RunState};
use uuid::Uuid;

use crate::SqliteStore;

const RUN_COLS: &str = "id, mode, source_root, target_root, started_at, completed_at, state";

fn parse_uuid(s: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn from_text<T, E: std::error::Error + Send + Sync + 'static>(
    v: std::result::Result<T, E>,
) -> rusqlite::Result<T> {
    v.map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })
}

fn row_to_run(row: &Row<'_>) -> rusqlite::Result<RunRecord> {
    let id = parse_uuid(&row.get::<_, String>("id")?)?;
    let mode = from_text(RunMode::parse(&row.get::<_, String>("mode")?))?;
    let state = from_text(RunState::parse(&row.get::<_, String>("state")?))?;
    let target_root = row
        .get::<_, Option<String>>("target_root")?
        .map(PathBuf::from);
    Ok(RunRecord {
        id,
        mode,
        source_root: PathBuf::from(row.get::<_, String>("source_root")?),
        target_root,
        started_at: row.get::<_, DateTime<Utc>>("started_at")?,
        completed_at: row.get::<_, Option<DateTime<Utc>>>("completed_at")?,
        state,
    })
}

fn path_opt(p: Option<&PathBuf>) -> Result<Option<String>> {
    p.map_or_else(
        || Ok(None),
        |path| {
            path.to_str()
                .map(|s| Some(s.to_owned()))
                .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))
        },
    )
}

#[async_trait]
impl RunLog for SqliteStore {
    async fn record_run(&self, run: &RunRecord) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let run = run.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let src = run
                .source_root
                .to_str()
                .ok_or_else(|| anyhow!("source_root not valid UTF-8"))?
                .to_owned();
            let tgt = path_opt(run.target_root.as_ref())?;
            {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        &format!(
                            "INSERT INTO runs ({RUN_COLS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
                        ),
                        params![
                            run.id.to_string(),
                            run.mode.as_str(),
                            src,
                            tgt,
                            run.started_at,
                            run.completed_at,
                            run.state.as_str(),
                        ],
                    )
                    .context("inserting run")?;
            }
            Ok(())
        })
        .await
        .context("join record_run task")??;
        Ok(())
    }

    async fn finish_run(&self, run_id: Uuid, state: RunState) -> tidyup_core::Result<()> {
        let conn = self.conn();
        tokio::task::spawn_blocking(move || -> Result<()> {
            {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        "UPDATE runs SET state = ?1, completed_at = ?2 WHERE id = ?3",
                        params![state.as_str(), Utc::now(), run_id.to_string()],
                    )
                    .context("updating run state")?;
            }
            Ok(())
        })
        .await
        .context("join finish_run task")??;
        Ok(())
    }

    async fn get_run(&self, run_id: Uuid) -> tidyup_core::Result<Option<RunRecord>> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<RunRecord>> {
            let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
            let mut stmt = guard.prepare(&format!("SELECT {RUN_COLS} FROM runs WHERE id = ?1"))?;
            let fetched = stmt.query_row(params![run_id.to_string()], row_to_run).ok();
            Ok(fetched)
        })
        .await
        .context("join get_run task")??;
        Ok(result)
    }

    async fn list_runs(&self) -> tidyup_core::Result<Vec<RunRecord>> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<RunRecord>> {
            let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
            let mut stmt = guard.prepare(&format!(
                "SELECT {RUN_COLS} FROM runs ORDER BY started_at DESC"
            ))?;
            let rows = stmt
                .query_map([], row_to_run)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
        .await
        .context("join list_runs task")??;
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn record_and_get_run_roundtrips() {
        let store = SqliteStore::open_in_memory().unwrap();
        let run = RunRecord::begin(
            RunMode::Migrate,
            PathBuf::from("/src"),
            Some(PathBuf::from("/target")),
        );
        store.record_run(&run).await.unwrap();
        let back = store.get_run(run.id).await.unwrap().unwrap();
        assert_eq!(back.id, run.id);
        assert_eq!(back.mode, RunMode::Migrate);
        assert_eq!(back.state, RunState::InProgress);
    }

    #[tokio::test]
    async fn finish_run_sets_state_and_completed_at() {
        let store = SqliteStore::open_in_memory().unwrap();
        let run = RunRecord::begin(RunMode::Scan, PathBuf::from("/src"), None);
        store.record_run(&run).await.unwrap();
        store.finish_run(run.id, RunState::Completed).await.unwrap();
        let back = store.get_run(run.id).await.unwrap().unwrap();
        assert_eq!(back.state, RunState::Completed);
        assert!(back.completed_at.is_some());
    }

    #[tokio::test]
    async fn list_runs_orders_recent_first() {
        let store = SqliteStore::open_in_memory().unwrap();
        let first = RunRecord::begin(RunMode::Scan, PathBuf::from("/a"), None);
        store.record_run(&first).await.unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = RunRecord::begin(RunMode::Scan, PathBuf::from("/b"), None);
        store.record_run(&second).await.unwrap();

        let runs = store.list_runs().await.unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].id, second.id);
        assert_eq!(runs[1].id, first.id);
    }

    #[tokio::test]
    async fn get_run_returns_none_for_unknown_id() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert!(store.get_run(Uuid::new_v4()).await.unwrap().is_none());
    }
}
