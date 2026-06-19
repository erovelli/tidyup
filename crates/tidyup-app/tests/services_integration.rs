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
use tidyup_domain::{BundleProposal, ChangeProposal, Phase, ReviewDecision};
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

/// Interactive-style handler that approves every loose proposal *and* every
/// bundle it is shown. Exercises the non-`--yes` bundle-review path end to end.
struct ApproveEverything {
    bundles_seen: Mutex<Vec<uuid::Uuid>>,
}

impl ApproveEverything {
    const fn new() -> Self {
        Self {
            bundles_seen: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ReviewHandler for ApproveEverything {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> CoreResult<Vec<ReviewDecision>> {
        Ok(proposals
            .into_iter()
            .map(|p| ReviewDecision::Approve(p.id))
            .collect())
    }
    async fn review_bundles(&self, bundles: Vec<BundleProposal>) -> CoreResult<Vec<uuid::Uuid>> {
        let ids: Vec<_> = bundles.iter().map(|b| b.id).collect();
        self.bundles_seen
            .lock()
            .unwrap()
            .extend(ids.iter().copied());
        Ok(ids)
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
    make_ctx_with_shelf(None)
}

fn make_ctx_with_shelf(shelf: Option<std::path::PathBuf>) -> (Arc<ServiceContext>, SqliteStore) {
    let store = SqliteStore::open_in_memory().unwrap();
    let store = if let Some(s) = shelf {
        store.with_backup_root(s)
    } else {
        store
    };
    let ctx = Arc::new(ServiceContext {
        file_index: Arc::new(store.clone()),
        change_log: Arc::new(store.clone()),
        backup_store: Arc::new(store.clone()),
        run_log: Arc::new(store.clone()),
        text: Some(Arc::new(StubText) as Arc<dyn TextBackend>),
        embeddings: Arc::new(BucketEmbeddings),
        vision: None,
        image_embeddings: None,
        audio_embeddings: None,
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
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 1, "expected 1 loose proposal");
    assert_eq!(report.bundles, 0);
    assert_eq!(report.approved, 1, "auto-approve should count 1");
    assert_eq!(report.applied, 1, "dry-run apply counts approved proposals");

    // Dry-run leaves proposals Pending in the change log.
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
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
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
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 1, "one loose proposal");
    assert_eq!(report.bundles, 0);
    assert_eq!(report.approved, 1);
    assert_eq!(report.applied, 1, "dry-run apply counts approved");

    let pending = store.pending().await.unwrap();
    assert_eq!(pending.len(), 1);
    // Destination must be inside the target tree.
    assert!(pending[0].proposed_path.starts_with(tgt.path()));
}

#[tokio::test]
async fn scan_service_applies_moves_and_rollback_restores_them() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    std::fs::create_dir_all(&src_root).unwrap();
    let src_file = src_root.join("helpers.rs");
    std::fs::write(&src_file, b"fn main() {}").unwrap();

    let (ctx, _store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    let reviewer = AutoApprove::new();

    let report = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src_root.clone(),
                taxonomy_path: None,
                dry_run: false,
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 1);
    assert_eq!(report.applied, 1, "real apply should move the file");
    assert_eq!(report.failed, 0);
    assert!(!src_file.exists(), "original must be gone after apply");

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let report = rollback
        .rollback_run(report.run_id, &NullProgress)
        .await
        .unwrap();

    assert_eq!(report.restored, 1);
    assert_eq!(report.failures, 0);
    assert!(src_file.exists(), "rollback restores the original");
    assert_eq!(std::fs::read(&src_file).unwrap(), b"fn main() {}");
}

#[tokio::test]
async fn rollback_service_lists_runs_most_recent_first() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    std::fs::create_dir_all(&src_root).unwrap();

    let (ctx, _store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));

    // Run once with no source files — just to create a run record.
    let _ = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src_root.clone(),
                taxonomy_path: None,
                dry_run: true,
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
            &NullProgress,
            &AutoApprove::new(),
        )
        .await
        .unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let runs = rollback.list_runs().await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].mode, tidyup_domain::RunMode::Scan);
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
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
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

#[tokio::test]
async fn interactive_bundle_review_applies_and_rollback_restores_it() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    let proj = src_root.join("myproj");
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("Cargo.toml"), b"[package]\nname='x'\n").unwrap();
    std::fs::write(proj.join("src/main.rs"), b"fn main() {}").unwrap();

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    // NOT --yes: auto_approve_bundles = false. The bundle applies only because
    // the interactive handler approves it via review_bundles.
    let reviewer = ApproveEverything::new();

    let report = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src_root.clone(),
                taxonomy_path: None,
                dry_run: false,
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.bundles, 1, "the Cargo project is one bundle");
    assert_eq!(report.bundles_applied, 1, "interactive review approved it");
    assert_eq!(report.bundles_skipped, 0);
    assert_eq!(report.bundles_failed, 0);
    assert_eq!(
        reviewer.bundles_seen.lock().unwrap().len(),
        1,
        "the handler was shown the bundle",
    );
    assert!(!proj.exists(), "bundle root must be gone after atomic move");

    // The whole subtree must be restorable atomically.
    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback
        .rollback_run(report.run_id, &NullProgress)
        .await
        .unwrap();
    assert_eq!(rb.bundles_restored, 1);
    assert_eq!(rb.failures, 0);
    assert!(
        proj.join("Cargo.toml").exists(),
        "rollback restores the bundle"
    );
    assert!(proj.join("src/main.rs").exists());

    // No loose proposals were produced (bundle consumed all descendants).
    let pending = store.pending().await.unwrap();
    assert!(pending.is_empty());
}

#[tokio::test]
async fn file_set_bundle_applies_atomically_and_rollback_restores_it() {
    use tidyup_app::RollbackService;

    // Three sibling files forming a filename family become a DocumentSeries —
    // a *file-set* bundle (no shared directory; members move individually,
    // atomically). Exercises apply_file_set_bundle + rollback_file_set_bundle
    // through the real SqliteStore (shelve + restore keyed by member id).
    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    std::fs::create_dir_all(&src_root).unwrap();
    for n in ["invoice-01.pdf", "invoice-02.pdf", "invoice-03.pdf"] {
        std::fs::write(src_root.join(n), format!("contents of {n}").as_bytes()).unwrap();
    }

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    let reviewer = ApproveEverything::new();

    let report = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src_root.clone(),
                taxonomy_path: None,
                dry_run: false,
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(
        report.bundles, 1,
        "the invoice family is one file-set bundle"
    );
    assert_eq!(report.proposed, 0, "all three files were clustered");
    assert_eq!(report.bundles_applied, 1);
    assert_eq!(report.bundles_failed, 0);
    // Originals moved; the destination is under the source root's taxonomy.
    for n in ["invoice-01.pdf", "invoice-02.pdf", "invoice-03.pdf"] {
        assert!(!src_root.join(n).exists(), "{n} original must be moved");
    }
    let moved = src_root.join("Documents/Series/invoice/invoice-01.pdf");
    assert!(
        moved.exists(),
        "members land flat under the cluster subfolder"
    );

    // Atomic restore — every member comes back to its original path.
    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback
        .rollback_run(report.run_id, &NullProgress)
        .await
        .unwrap();
    assert_eq!(rb.bundles_restored, 1);
    assert_eq!(rb.failures, 0);
    for n in ["invoice-01.pdf", "invoice-02.pdf", "invoice-03.pdf"] {
        assert!(src_root.join(n).exists(), "{n} must be restored");
    }
    assert_eq!(
        std::fs::read(src_root.join("invoice-02.pdf")).unwrap(),
        b"contents of invoice-02.pdf",
    );
    assert!(store.pending().await.unwrap().is_empty());
}

#[tokio::test]
async fn bundle_held_when_interactive_review_rejects_it() {
    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    let proj = src_root.join("myproj");
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("Cargo.toml"), b"[package]\nname='x'\n").unwrap();
    std::fs::write(proj.join("src/main.rs"), b"fn main() {}").unwrap();

    let (ctx, _store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    // AutoApprove uses the default review_bundles (approve nothing) → bundle held.
    let reviewer = AutoApprove::new();

    let report = service
        .run(
            tidyup_app::scan::ScanRequest {
                root: src_root.clone(),
                taxonomy_path: None,
                dry_run: false,
                auto_approve_bundles: false,
                bundle_min_confidence: 0.85,
            },
            &candidates,
            &[],
            &[],
            &NullProgress,
            &reviewer,
        )
        .await
        .unwrap();

    assert_eq!(report.bundles, 1);
    assert_eq!(report.bundles_applied, 0, "rejected bundle must not move");
    assert_eq!(report.bundles_skipped, 1);
    assert!(proj.exists(), "rejected bundle stays in place");
}
