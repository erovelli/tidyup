//! `FileIndex` implementation on `SqliteStore`.
//!
//! `upsert` preserves the existing `FileId` on path conflict via `SQLite`'s
//! `ON CONFLICT(path) DO UPDATE` — the caller-supplied `id` is used only for brand-new rows.
//! This is the load-bearing invariant for cross-scan stability.

use std::mem::size_of;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use rusqlite::{params, Row};
use tidyup_core::storage::FileIndex;
use tidyup_domain::{
    ContentHash, FileId, FileProcessingRecord, FileProcessingRole, FileProcessingStage,
    FileProcessingState, IndexedFile, SemanticArtifact,
};
use uuid::Uuid;

use crate::SqliteStore;

const COLS: &str = "id, path, name, extension, mime_type, size_bytes, content_hash, indexed_at";

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", p.display()))
}

/// Escape SQL LIKE metacharacters (`%`, `_`, `\`) so prefix matching is literal.
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if matches!(ch, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn row_to_file(row: &Row<'_>) -> rusqlite::Result<IndexedFile> {
    let id_str: String = row.get("id")?;
    let path_str: String = row.get("path")?;
    let hash: String = row.get("content_hash")?;
    let size: i64 = row.get("size_bytes")?;
    let id = Uuid::parse_str(&id_str).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let size_bytes = u64::try_from(size).unwrap_or(0);
    Ok(IndexedFile {
        id: FileId(id),
        path: PathBuf::from(path_str),
        name: row.get("name")?,
        extension: row.get("extension")?,
        mime_type: row.get("mime_type")?,
        size_bytes,
        content_hash: ContentHash(hash),
        indexed_at: row.get("indexed_at")?,
    })
}

fn parse_uuid(value: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
    })
}

fn row_to_processing(row: &Row<'_>) -> rusqlite::Result<FileProcessingRecord> {
    let run_id = parse_uuid(&row.get::<_, String>("run_id")?)?;
    let file_id = row
        .get::<_, Option<String>>("file_id")?
        .map(|value| parse_uuid(&value).map(FileId))
        .transpose()?;
    let role = FileProcessingRole::parse(&row.get::<_, String>("role")?).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
    })?;
    let lifecycle_step =
        FileProcessingStage::parse(&row.get::<_, String>("stage")?).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    let lifecycle_status =
        FileProcessingState::parse(&row.get::<_, String>("state")?).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    Ok(FileProcessingRecord {
        run_id,
        path: PathBuf::from(row.get::<_, String>("path")?),
        file_id,
        role,
        stage: lifecycle_step,
        state: lifecycle_status,
        reason: row.get("reason")?,
        updated_at: row.get("updated_at")?,
    })
}

fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len().saturating_mul(size_of::<f32>()));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_embedding(bytes: &[u8]) -> std::result::Result<Vec<f32>, std::io::Error> {
    if !bytes.len().is_multiple_of(size_of::<f32>()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "semantic embedding blob length is not divisible by four",
        ));
    }
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            <[u8; 4]>::try_from(chunk)
                .map(f32::from_le_bytes)
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "semantic embedding chunk has invalid width",
                    )
                })
        })
        .collect()
}

#[async_trait]
impl FileIndex for SqliteStore {
    async fn upsert(&self, record: &IndexedFile) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let record = record.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let size_bytes =
                i64::try_from(record.size_bytes).context("file size_bytes exceeds i64 range")?;
            let path = path_str(&record.path)?;
            {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                guard
                    .execute(
                        "INSERT INTO files (id, path, name, extension, mime_type, size_bytes, \
                         content_hash, indexed_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                         ON CONFLICT(path) DO UPDATE SET \
                            name = excluded.name, \
                            extension = excluded.extension, \
                            mime_type = excluded.mime_type, \
                            size_bytes = excluded.size_bytes, \
                            content_hash = excluded.content_hash, \
                            indexed_at = excluded.indexed_at",
                        params![
                            record.id.0.to_string(),
                            path,
                            record.name,
                            record.extension,
                            record.mime_type,
                            size_bytes,
                            record.content_hash.0,
                            record.indexed_at,
                        ],
                    )
                    .context("upserting indexed file")?;
            }
            Ok(())
        })
        .await
        .context("join upsert task")??;
        Ok(())
    }

    async fn get(&self, id: &FileId) -> tidyup_core::Result<Option<IndexedFile>> {
        let conn = self.conn();
        let id_str = id.0.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<IndexedFile>> {
            let row = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt = guard.prepare(&format!("SELECT {COLS} FROM files WHERE id = ?1"))?;
                stmt.query_row(params![id_str], row_to_file)
                    .map(Some)
                    .or_else(|e| match e {
                        rusqlite::Error::QueryReturnedNoRows => Ok(None),
                        other => Err(other),
                    })?
            };
            Ok(row)
        })
        .await
        .context("join get task")??;
        Ok(result)
    }

    async fn by_path(&self, path: &Path) -> tidyup_core::Result<Option<IndexedFile>> {
        let conn = self.conn();
        let path = path_str(path)?.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<IndexedFile>> {
            let row = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let mut stmt =
                    guard.prepare(&format!("SELECT {COLS} FROM files WHERE path = ?1"))?;
                stmt.query_row(params![path], row_to_file)
                    .map(Some)
                    .or_else(|e| match e {
                        rusqlite::Error::QueryReturnedNoRows => Ok(None),
                        other => Err(other),
                    })?
            };
            Ok(row)
        })
        .await
        .context("join by_path task")??;
        Ok(result)
    }

    async fn list_under(&self, root: &Path) -> tidyup_core::Result<Vec<IndexedFile>> {
        let conn = self.conn();
        let root_str = path_str(root)?.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<IndexedFile>> {
            let rows = {
                let guard = conn.lock().map_err(|e| anyhow!("lock poisoned: {e}"))?;
                let prefix = format!("{}/%", escape_like(&root_str));
                let mut stmt = guard.prepare(&format!(
                    "SELECT {COLS} FROM files WHERE path = ?1 OR path LIKE ?2 ESCAPE '\\' \
                     ORDER BY path"
                ))?;
                let fetched: Vec<IndexedFile> = stmt
                    .query_map(params![root_str, prefix], row_to_file)?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                fetched
            };
            Ok(rows)
        })
        .await
        .context("join list_under task")??;
        Ok(result)
    }

    async fn put_processing_record(
        &self,
        record: &FileProcessingRecord,
    ) -> tidyup_core::Result<()> {
        self.put_processing_records(std::slice::from_ref(record))
            .await
    }

    async fn put_processing_records(
        &self,
        records: &[FileProcessingRecord],
    ) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let records = records.to_vec();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut guard = conn
                .lock()
                .map_err(|error| anyhow!("lock poisoned: {error}"))?;
            let transaction = guard.transaction()?;
            for record in &records {
                let path = path_str(&record.path)?;
                let file_id = record.file_id.as_ref().map(|id| id.0.to_string());
                transaction.execute(
                    "INSERT INTO file_processing \
                     (run_id, path, role, file_id, stage, state, reason, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                     ON CONFLICT(run_id, path, role) DO UPDATE SET \
                        file_id = excluded.file_id, stage = excluded.stage, \
                        state = excluded.state, reason = excluded.reason, \
                        updated_at = excluded.updated_at",
                    params![
                        record.run_id.to_string(),
                        path,
                        record.role.as_str(),
                        file_id,
                        record.stage.as_str(),
                        record.state.as_str(),
                        record.reason.as_deref(),
                        record.updated_at,
                    ],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })
        .await
        .context("join put_processing_records task")??;
        Ok(())
    }

    async fn processing_for_run(
        &self,
        run_id: Uuid,
    ) -> tidyup_core::Result<Vec<FileProcessingRecord>> {
        let conn = self.conn();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<FileProcessingRecord>> {
            let guard = conn
                .lock()
                .map_err(|error| anyhow!("lock poisoned: {error}"))?;
            let mut stmt = guard.prepare(
                "SELECT run_id, path, role, file_id, stage, state, reason, updated_at \
                 FROM file_processing WHERE run_id = ?1 ORDER BY role, path",
            )?;
            let rows = stmt
                .query_map(params![run_id.to_string()], row_to_processing)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
        .await
        .context("join processing_for_run task")??;
        Ok(result)
    }

    async fn fail_incomplete_processing(
        &self,
        run_id: Uuid,
        role: FileProcessingRole,
        stage: FileProcessingStage,
        reason: &str,
    ) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let reason = reason.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let guard = conn
                .lock()
                .map_err(|error| anyhow!("lock poisoned: {error}"))?;
            guard.execute(
                "UPDATE file_processing SET stage = ?1, state = ?2, reason = ?3, updated_at = ?4 \
                 WHERE run_id = ?5 AND role = ?6 AND state IN ('Discovered', 'Indexed')",
                params![
                    stage.as_str(),
                    FileProcessingState::Failed.as_str(),
                    reason,
                    chrono::Utc::now(),
                    run_id.to_string(),
                    role.as_str(),
                ],
            )?;
            Ok(())
        })
        .await
        .context("join fail_incomplete_processing task")??;
        Ok(())
    }

    async fn semantic_artifact(
        &self,
        content_hash: &ContentHash,
        model_id: &str,
        preprocessing_version: &str,
        latent_space: &str,
    ) -> tidyup_core::Result<Option<SemanticArtifact>> {
        let conn = self.conn();
        let content_hash = content_hash.clone();
        let model_id = model_id.to_string();
        let preprocessing_version = preprocessing_version.to_string();
        let latent_space = latent_space.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Option<SemanticArtifact>> {
            let row = {
                let guard = conn
                    .lock()
                    .map_err(|error| anyhow!("lock poisoned: {error}"))?;
                guard
                    .query_row(
                        "SELECT embedding_json, embedding_blob, updated_at FROM semantic_artifacts \
                         WHERE content_hash = ?1 AND model_id = ?2 \
                         AND preprocessing_version = ?3 AND latent_space = ?4",
                        params![
                            content_hash.0,
                            model_id,
                            preprocessing_version,
                            latent_space,
                        ],
                        |row| {
                            let embedding_blob: Option<Vec<u8>> = row.get(1)?;
                            let embedding = if let Some(bytes) = embedding_blob {
                                decode_embedding(&bytes).map_err(|error| {
                                    rusqlite::Error::FromSqlConversionFailure(
                                        1,
                                        rusqlite::types::Type::Blob,
                                        Box::new(error),
                                    )
                                })?
                            } else {
                                let embedding_json: String = row.get(0)?;
                                serde_json::from_str(&embedding_json).map_err(|error| {
                                    rusqlite::Error::FromSqlConversionFailure(
                                        0,
                                        rusqlite::types::Type::Text,
                                        Box::new(error),
                                    )
                                })?
                            };
                            Ok(SemanticArtifact {
                                content_hash: content_hash.clone(),
                                model_id: model_id.clone(),
                                preprocessing_version: preprocessing_version.clone(),
                                latent_space: latent_space.clone(),
                                embedding,
                                updated_at: row.get(2)?,
                            })
                        },
                    )
                    .map(Some)
                    .or_else(|error| match error {
                        rusqlite::Error::QueryReturnedNoRows => Ok(None),
                        other => Err(other),
                    })?
            };
            Ok(row)
        })
        .await
        .context("join semantic_artifact task")??;
        Ok(result)
    }

    async fn put_semantic_artifact(&self, artifact: &SemanticArtifact) -> tidyup_core::Result<()> {
        let conn = self.conn();
        let artifact = artifact.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let embedding_blob = encode_embedding(&artifact.embedding);
            let guard = conn.lock().map_err(|error| anyhow!("lock poisoned: {error}"))?;
            guard.execute(
                "INSERT INTO semantic_artifacts \
                 (content_hash, model_id, preprocessing_version, latent_space, embedding_json, embedding_blob, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, '', ?5, ?6) \
                 ON CONFLICT(content_hash, model_id, preprocessing_version, latent_space) \
                 DO UPDATE SET embedding_json = '', embedding_blob = excluded.embedding_blob, \
                    updated_at = excluded.updated_at",
                params![
                    artifact.content_hash.0,
                    artifact.model_id,
                    artifact.preprocessing_version,
                    artifact.latent_space,
                    embedding_blob,
                    artifact.updated_at,
                ],
            )?;
            Ok(())
        })
        .await
        .context("join put_semantic_artifact task")??;
        Ok(())
    }

    async fn prune_semantic_artifacts_older_than_days(
        &self,
        days: u32,
    ) -> tidyup_core::Result<usize> {
        let conn = self.conn();
        let deleted = tokio::task::spawn_blocking(move || -> Result<usize> {
            let cutoff = chrono::Utc::now()
                .checked_sub_signed(chrono::Duration::days(i64::from(days)))
                .context("semantic artifact retention cutoff is out of range")?;
            let guard = conn
                .lock()
                .map_err(|error| anyhow!("lock poisoned: {error}"))?;
            let deleted = guard.execute(
                "DELETE FROM semantic_artifacts WHERE updated_at < ?1",
                params![cutoff],
            )?;
            Ok(deleted)
        })
        .await
        .context("join prune_semantic_artifacts task")??;
        Ok(deleted)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tidyup_core::storage::RunLog;
    use tidyup_domain::{RunMode, RunRecord};

    fn sample_file(path: &str, id: Uuid) -> IndexedFile {
        IndexedFile {
            id: FileId(id),
            path: PathBuf::from(path),
            name: path.rsplit('/').next().unwrap_or("").to_string(),
            extension: "rs".to_string(),
            mime_type: "text/x-rust".to_string(),
            size_bytes: 42,
            content_hash: ContentHash("deadbeef".to_string()),
            indexed_at: Utc::now(),
        }
    }

    fn sample_artifact(model_id: &str) -> SemanticArtifact {
        SemanticArtifact {
            content_hash: ContentHash("artifact-hash".to_string()),
            model_id: model_id.to_string(),
            preprocessing_version: "image-v1".to_string(),
            latent_space: "image".to_string(),
            embedding: vec![0.6, 0.8],
            updated_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn upsert_preserves_file_id_across_rescans() {
        let store = SqliteStore::open_in_memory().unwrap();
        let first_id = Uuid::new_v4();
        store
            .upsert(&sample_file("/code/main.rs", first_id))
            .await
            .unwrap();

        let second_id = Uuid::new_v4();
        assert_ne!(first_id, second_id);
        store
            .upsert(&sample_file("/code/main.rs", second_id))
            .await
            .unwrap();

        let found = store
            .by_path(Path::new("/code/main.rs"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(found.id.0, first_id, "upsert must preserve original FileId");
    }

    #[tokio::test]
    async fn semantic_artifact_roundtrips_by_exact_versioned_key() {
        let store = SqliteStore::open_in_memory().unwrap();
        let artifact = sample_artifact("siglip-test");
        store.put_semantic_artifact(&artifact).await.unwrap();

        let loaded = store
            .semantic_artifact(&artifact.content_hash, "siglip-test", "image-v1", "image")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded, artifact);

        let stale_model = store
            .semantic_artifact(
                &artifact.content_hash,
                "different-model",
                "image-v1",
                "image",
            )
            .await
            .unwrap();
        assert!(stale_model.is_none());
    }

    #[tokio::test]
    async fn semantic_artifact_reads_legacy_json_and_writes_compact_blob() {
        let store = SqliteStore::open_in_memory().unwrap();
        let updated_at = Utc::now();
        {
            let conn = store.conn();
            let guard = conn.lock().unwrap();
            guard
                .execute(
                    "INSERT INTO semantic_artifacts \
                     (content_hash, model_id, preprocessing_version, latent_space, embedding_json, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        "legacy-hash",
                        "legacy-model",
                        "image-v1",
                        "image",
                        "[0.6,0.8]",
                        updated_at,
                    ],
                )
                .unwrap();
        }
        let loaded = store
            .semantic_artifact(
                &ContentHash("legacy-hash".to_string()),
                "legacy-model",
                "image-v1",
                "image",
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.embedding, vec![0.6, 0.8]);

        let current = sample_artifact("blob-model");
        store.put_semantic_artifact(&current).await.unwrap();
        let (storage_type, bytes): (String, i64) = {
            let conn = store.conn();
            let guard = conn.lock().unwrap();
            guard
                .query_row(
                    "SELECT typeof(embedding_blob), length(embedding_blob) \
                     FROM semantic_artifacts WHERE model_id = ?1",
                    params![&current.model_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap()
        };
        assert_eq!(storage_type, "blob");
        assert_eq!(
            usize::try_from(bytes).unwrap(),
            current.embedding.len() * size_of::<f32>()
        );
    }

    #[tokio::test]
    async fn semantic_artifact_upsert_replaces_embedding() {
        let store = SqliteStore::open_in_memory().unwrap();
        let mut artifact = sample_artifact("siglip-test");
        store.put_semantic_artifact(&artifact).await.unwrap();
        artifact.embedding = vec![1.0, 0.0];
        artifact.updated_at = Utc::now();
        store.put_semantic_artifact(&artifact).await.unwrap();

        let loaded = store
            .semantic_artifact(&artifact.content_hash, "siglip-test", "image-v1", "image")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.embedding, vec![1.0, 0.0]);
    }

    #[tokio::test]
    async fn semantic_artifact_prune_removes_only_expired_rows() {
        let store = SqliteStore::open_in_memory().unwrap();
        let mut expired = sample_artifact("expired-model");
        expired.updated_at = Utc::now() - chrono::Duration::days(60);
        let current = sample_artifact("current-model");
        store.put_semantic_artifact(&expired).await.unwrap();
        store.put_semantic_artifact(&current).await.unwrap();

        let deleted = store
            .prune_semantic_artifacts_older_than_days(30)
            .await
            .unwrap();
        assert_eq!(deleted, 1);
        assert!(store
            .semantic_artifact(
                &expired.content_hash,
                &expired.model_id,
                &expired.preprocessing_version,
                &expired.latent_space,
            )
            .await
            .unwrap()
            .is_none());
        assert!(store
            .semantic_artifact(
                &current.content_hash,
                &current.model_id,
                &current.preprocessing_version,
                &current.latent_space,
            )
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn processing_record_batches_upsert_and_incomplete_states_fail() {
        let store = SqliteStore::open_in_memory().unwrap();
        let run = RunRecord::begin(RunMode::Scan, PathBuf::from("/source"), None);
        store.record_run(&run).await.unwrap();
        let first = sample_file("/source/first.rs", Uuid::new_v4());
        let second = sample_file("/source/second.rs", Uuid::new_v4());
        store.upsert(&first).await.unwrap();
        store.upsert(&second).await.unwrap();
        store
            .put_processing_records(&[
                FileProcessingRecord {
                    run_id: run.id,
                    path: first.path.clone(),
                    file_id: Some(first.id.clone()),
                    role: FileProcessingRole::Source,
                    stage: FileProcessingStage::Indexing,
                    state: FileProcessingState::Indexed,
                    reason: None,
                    updated_at: Utc::now(),
                },
                FileProcessingRecord {
                    run_id: run.id,
                    path: second.path.clone(),
                    file_id: Some(second.id.clone()),
                    role: FileProcessingRole::Source,
                    stage: FileProcessingStage::Indexing,
                    state: FileProcessingState::Indexed,
                    reason: None,
                    updated_at: Utc::now(),
                },
            ])
            .await
            .unwrap();

        store
            .fail_incomplete_processing(
                run.id,
                FileProcessingRole::Source,
                FileProcessingStage::Planning,
                "pipeline stopped",
            )
            .await
            .unwrap();
        let rows = store.processing_for_run(run.id).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| {
            row.state == FileProcessingState::Failed
                && row.stage == FileProcessingStage::Planning
                && row.reason.as_deref() == Some("pipeline stopped")
        }));
    }

    #[tokio::test]
    async fn by_path_and_get_return_none_when_absent() {
        let store = SqliteStore::open_in_memory().unwrap();
        assert!(store.by_path(Path::new("/nope")).await.unwrap().is_none());
        assert!(store.get(&FileId(Uuid::new_v4())).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_under_returns_recursive_subtree() {
        let store = SqliteStore::open_in_memory().unwrap();
        store
            .upsert(&sample_file("/code/a.rs", Uuid::new_v4()))
            .await
            .unwrap();
        store
            .upsert(&sample_file("/code/sub/b.rs", Uuid::new_v4()))
            .await
            .unwrap();
        store
            .upsert(&sample_file("/other/c.rs", Uuid::new_v4()))
            .await
            .unwrap();

        let listed = store.list_under(Path::new("/code")).await.unwrap();
        let paths: Vec<_> = listed
            .iter()
            .map(|f| f.path.display().to_string())
            .collect();
        assert_eq!(paths, ["/code/a.rs", "/code/sub/b.rs"]);
    }
}
