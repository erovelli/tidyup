//! End-to-end integration tests for `ScanService` + `MigrationService`.
//!
//! Uses a real in-memory `SQLite` [`ChangeLog`] per the project's
//! no-mocking-at-module-boundaries convention. The embedding backend is a
//! small deterministic stub (7-bucket hash + L2-normalize) so tests stay
//! hermetic — no ONNX models, no network.
//!
//! [`ChangeLog`]: tidyup_core::storage::ChangeLog

#![allow(clippy::unwrap_used, clippy::missing_panics_doc)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tempfile::TempDir;
use tidyup_app::{MigrationService, ScanService, ServiceContext};
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::inference::{EmbeddingBackend, TextBackend};
use tidyup_core::storage::ChangeLog;
use tidyup_core::{Result as CoreResult, ReviewHandler};
use tidyup_domain::{ChangeProposal, Phase, ReviewDecision};
use tidyup_storage_sqlite::SqliteStore;

struct NullProgress;
#[async_trait]
impl ProgressReporter for NullProgress {
    async fn phase_started(&self, _p: Phase, _t: Option<u64>) {}
    async fn item_completed(&self, _p: Phase, _i: ProgressItem) {}
    async fn phase_finished(&self, _p: Phase) {}
    async fn message(&self, _l: Level, _m: &str) {}
}

/// Review handler that auto-approves every proposal it sees and records them.
struct AutoApprove {
    seen: Mutex<Vec<ChangeProposal>>,
}

impl AutoApprove {
    const fn new() -> Self {
        Self {
            seen: Mutex::new(Vec::new()),
        }
    }
    fn seen_ids(&self) -> Vec<uuid::Uuid> {
        self.seen.lock().unwrap().iter().map(|p| p.id).collect()
    }
}

#[async_trait]
impl ReviewHandler for AutoApprove {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> CoreResult<Vec<ReviewDecision>> {
        let mut out = Vec::with_capacity(proposals.len());
        for p in &proposals {
            out.push(ReviewDecision::Approve(p.id));
        }
        self.seen.lock().unwrap().extend(proposals);
        Ok(out)
    }
}

/// Deterministic 7-bucket embedder. Not semantically meaningful but stable.
struct BucketEmbeddings;
#[async_trait]
impl EmbeddingBackend for BucketEmbeddings {
    async fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut v = vec![0.0_f32; 7];
        for (i, b) in text.bytes().enumerate() {
            v[i % 7] += f32::from(b);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    }
    async fn embed_texts(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed_text(t).await?);
        }
        Ok(out)
    }
    fn dimensions(&self) -> usize {
        7
    }
    fn model_id(&self) -> &'static str {
        "bucket"
    }
}

struct StubText;
#[async_trait]
impl TextBackend for StubText {
    async fn classify_text(
        &self,
        _text: &str,
        _filename: &str,
    ) -> CoreResult<tidyup_core::inference::ContentClassification> {
        anyhow::bail!("not used in Phase 4")
    }
    async fn classify_audio(
        &self,
        _filename: &str,
        _metadata: &str,
    ) -> CoreResult<tidyup_core::inference::ContentClassification> {
        anyhow::bail!("not used in Phase 4")
    }
    async fn classify_video(
        &self,
        _filename: &str,
        _frame_captions: &[String],
    ) -> CoreResult<tidyup_core::inference::ContentClassification> {
        anyhow::bail!("not used in Phase 4")
    }
    async fn classify_image_description(
        &self,
        _filename: &str,
        _description: &str,
    ) -> CoreResult<tidyup_core::inference::ContentClassification> {
        anyhow::bail!("not used in Phase 4")
    }
    async fn complete(
        &self,
        _prompt: &str,
        _opts: &tidyup_core::inference::GenerationOptions,
    ) -> CoreResult<String> {
        anyhow::bail!("not used in Phase 4")
    }
    fn model_id(&self) -> &'static str {
        "stub-text"
    }
}

struct PlainExtractor;
#[async_trait]
impl ContentExtractor for PlainExtractor {
    fn supports(&self, _path: &Path, _mime: Option<&str>) -> bool {
        true
    }
    async fn extract(&self, path: &Path) -> CoreResult<ExtractedContent> {
        let bytes = tokio::fs::read(path).await?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        Ok(ExtractedContent {
            text: Some(text),
            mime: "text/plain".to_string(),
            metadata: serde_json::json!({}),
        })
    }
}

fn make_ctx() -> (Arc<ServiceContext>, SqliteStore) {
    let store = SqliteStore::open_in_memory().unwrap();
    let ctx = Arc::new(ServiceContext {
        file_index: Arc::new(store.clone()),
        change_log: Arc::new(store.clone()),
        backup_store: Arc::new(store.clone()),
        text: Arc::new(StubText),
        embeddings: Arc::new(BucketEmbeddings),
        vision: None,
        extractors: vec![Arc::new(PlainExtractor)],
    });
    (ctx, store)
}

async fn sample_scan_candidates() -> Vec<tidyup_pipeline::scan::ScanCandidate> {
    let eb = BucketEmbeddings;
    let specs = [
        ("Finance/Taxes/", "tax return W-2 1099 1040 IRS", true),
        (
            "Code/",
            "source code rust python javascript compile function",
            false,
        ),
    ];
    let mut out = Vec::new();
    for (p, d, t) in &specs {
        let emb = eb.embed_text(d).await.unwrap();
        out.push(tidyup_pipeline::scan::ScanCandidate {
            folder_path: (*p).to_string(),
            description: (*d).to_string(),
            temporal: *t,
            embedding: emb,
        });
    }
    out
}

#[tokio::test]
async fn scan_service_persists_proposals_and_auto_approves() {
    let src = TempDir::new().unwrap();
    std::fs::write(src.path().join("helpers.rs"), b"fn main() {}").unwrap();

    let (ctx, store) = make_ctx();
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    let reviewer = AutoApprove::new();

    let report = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src.path().to_path_buf(),
                taxonomy_path: None,
                dry_run: true,
            },
            &candidates,
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 1, "expected 1 loose proposal");
    assert_eq!(report.bundles, 0);
    assert_eq!(report.approved, 1, "auto-approve should count 1");
    assert_eq!(report.applied, 0, "apply is Phase 5 — should stay 0");

    // Proposal should have been persisted to the change log.
    let pending = store.pending().await.unwrap();
    assert_eq!(pending.len(), 1);
    let seen = reviewer.seen_ids();
    assert!(seen.contains(&pending[0].id));
}

#[tokio::test]
async fn scan_service_records_bundles_separately() {
    let src = TempDir::new().unwrap();
    std::fs::create_dir_all(src.path().join("proj/src")).unwrap();
    std::fs::write(src.path().join("proj/Cargo.toml"), b"[package]\nname='x'\n").unwrap();
    std::fs::write(src.path().join("proj/src/main.rs"), b"fn main() {}").unwrap();

    let (ctx, store) = make_ctx();
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    let reviewer = AutoApprove::new();

    let report = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src.path().to_path_buf(),
                taxonomy_path: None,
                dry_run: true,
            },
            &candidates,
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.bundles, 1);
    assert_eq!(report.proposed, 0, "bundles consume all descendants");

    let bundles = store.pending_bundles().await.unwrap();
    assert_eq!(bundles.len(), 1);
    assert_eq!(bundles[0].members.len(), 2);
}

#[tokio::test]
async fn migration_service_builds_profiles_and_classifies() {
    let src = TempDir::new().unwrap();
    let tgt = TempDir::new().unwrap();
    std::fs::create_dir_all(tgt.path().join("Code")).unwrap();
    std::fs::create_dir_all(tgt.path().join("Finance")).unwrap();
    std::fs::write(src.path().join("snippet.rs"), b"fn main() {}").unwrap();

    let (ctx, store) = make_ctx();
    let service = MigrationService::new(Arc::clone(&ctx));
    let reviewer = AutoApprove::new();

    let report = service
        .run(
            tidyup_app::migration::MigrationRequest {
                source: src.path().to_path_buf(),
                target: tgt.path().to_path_buf(),
                dry_run: true,
            },
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 1, "one loose proposal");
    assert_eq!(report.bundles, 0);
    assert_eq!(report.approved, 1);
    assert_eq!(report.applied, 0, "apply is Phase 5");

    let pending = store.pending().await.unwrap();
    assert_eq!(pending.len(), 1);
    // Destination must be inside the target tree.
    assert!(pending[0].proposed_path.starts_with(tgt.path()));
}

#[tokio::test]
async fn migration_service_skips_review_when_no_loose_proposals() {
    let src = TempDir::new().unwrap();
    let tgt = TempDir::new().unwrap();
    std::fs::create_dir_all(tgt.path().join("Code")).unwrap();

    let (ctx, _store) = make_ctx();
    let service = MigrationService::new(Arc::clone(&ctx));
    let reviewer = AutoApprove::new();

    let report = service
        .run(
            tidyup_app::migration::MigrationRequest {
                source: src.path().to_path_buf(),
                target: tgt.path().to_path_buf(),
                dry_run: true,
            },
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 0);
    assert_eq!(report.approved, 0);
    // Reviewer was never called.
    assert!(reviewer.seen_ids().is_empty());
}
