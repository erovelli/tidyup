//! End-to-end integration tests for `ScanService` + `MigrationService`.
//!
//! Uses a real in-memory `SQLite` [`ChangeLog`] per the project's
//! no-mocking-at-module-boundaries convention. The embedding backend is a
//! small deterministic stub (7-bucket hash + L2-normalize) so tests stay
//! hermetic — no ONNX models, no network.
//!
//! [`ChangeLog`]: tidyup_core::storage::ChangeLog

#![allow(
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::missing_panics_doc,
    clippy::unwrap_used
)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tempfile::TempDir;
use tidyup_app::{MigrationService, ScanService, ServiceContext};
use tidyup_core::extractor::{ContentExtractor, ExtractedContent};
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_core::inference::{EmbeddingBackend, TextBackend};
use tidyup_core::storage::{BackupStore, ChangeLog, RunLog};
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

/// Review handler that rejects every loose proposal it sees.
struct RejectAll;

#[async_trait]
impl ReviewHandler for RejectAll {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> CoreResult<Vec<ReviewDecision>> {
        Ok(proposals
            .into_iter()
            .map(|p| ReviewDecision::Reject(p.id))
            .collect())
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
        classifier: tidyup_app::classifier_config_for(
            &tidyup_app::config::TidyupConfig::default(),
            true,
        ),
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
async fn rejected_proposal_leaves_the_pending_set() {
    // A real (non-dry-run) scan where review rejects the only proposal must
    // persist that outcome: the proposal is recorded, then marked Rejected, so
    // it no longer appears in pending() (which previously grew forever because
    // no rejection status was ever written). The file itself stays put.
    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    std::fs::create_dir_all(&src_root).unwrap();
    let src_file = src_root.join("helpers.rs");
    std::fs::write(&src_file, b"fn main() {}").unwrap();

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));

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
            &RejectAll,
        )
        .await
        .unwrap();

    assert_eq!(report.proposed, 1);
    assert_eq!(report.skipped, 1, "the proposal was rejected");
    assert_eq!(report.applied, 0);
    // The rejection is persisted: the proposal is no longer pending, and it was
    // never applied. The file stays where it was.
    assert!(
        store.pending().await.unwrap().is_empty(),
        "a rejected proposal must not remain in the pending set"
    );
    assert!(store
        .applied_proposals_for_run(report.run_id)
        .await
        .unwrap()
        .is_empty(),);
    assert!(src_file.exists(), "a rejected file is left in place");
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
async fn migration_refuses_source_target_overlap() {
    // Target nested inside source, source nested inside target, and equal roots
    // must all be refused before any run is recorded.
    let root = TempDir::new().unwrap();
    let outer = root.path().join("outer");
    let inner = outer.join("inner");
    std::fs::create_dir_all(&inner).unwrap();

    let (ctx, store) = make_ctx();
    let service = MigrationService::new(Arc::clone(&ctx));
    let reviewer = AutoApprove::new();

    let cases = [
        (outer.clone(), inner.clone()), // target inside source
        (inner.clone(), outer.clone()), // source inside target
        (outer.clone(), outer.clone()), // equal
    ];
    for (source, target) in cases {
        let err = service
            .run(
                tidyup_app::migration::MigrationRequest {
                    source: source.clone(),
                    target: target.clone(),
                    dry_run: false,
                    auto_approve_bundles: false,
                    bundle_min_confidence: 0.85,
                },
                &NullProgress,
                &reviewer,
            )
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("inside") || msg.contains("same directory"),
            "expected overlap error for {source:?} -> {target:?}, got: {msg}"
        );
    }
    // No run should have been recorded for a rejected overlap.
    assert!(
        store.list_runs().await.unwrap().is_empty(),
        "overlap must be refused before a run is recorded"
    );
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
async fn rollback_service_prunes_shelved_backups() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    std::fs::create_dir_all(&src_root).unwrap();
    std::fs::write(src_root.join("helpers.rs"), b"fn main() {}").unwrap();

    let (ctx, _store) = make_ctx_with_shelf(Some(shelf));
    let candidates = sample_scan_candidates().await;
    let service = ScanService::new(Arc::clone(&ctx));
    let rollback = RollbackService::new(Arc::clone(&ctx));

    // Empty shelf → nothing to prune.
    assert_eq!(rollback.prune_backups(0).await.unwrap(), 0);

    // A real apply shelves the original before moving it.
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
            &AutoApprove::new(),
        )
        .await
        .unwrap();
    assert_eq!(report.applied, 1);

    // Prune everything shelved before now → the just-shelved original expires.
    assert_eq!(
        rollback.prune_backups(0).await.unwrap(),
        1,
        "the shelved original should be pruned",
    );
    // Idempotent: a second prune finds nothing.
    assert_eq!(rollback.prune_backups(0).await.unwrap(), 0);
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

/// Apply a single loose scan move and return the context, store, run id, and
/// the path the file was moved to. Shared setup for the WP-0 rollback-integrity
/// tests below.
async fn apply_one_scan_move(
    workdir: &TempDir,
) -> (
    Arc<ServiceContext>,
    SqliteStore,
    uuid::Uuid,
    std::path::PathBuf,
) {
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    std::fs::create_dir_all(&src_root).unwrap();
    std::fs::write(src_root.join("helpers.rs"), b"fn main() {}").unwrap();

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
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
    assert_eq!(report.applied, 1);

    let applied = store
        .applied_proposals_for_run(report.run_id)
        .await
        .unwrap();
    assert_eq!(applied.len(), 1);
    let dest = applied[0].proposed_path.clone();
    assert!(dest.exists(), "move landed at destination");
    (ctx, store, report.run_id, dest)
}

#[tokio::test]
async fn rollback_preserves_a_destination_edited_after_apply() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, dest) = apply_one_scan_move(&workdir).await;

    // The user edits the moved file after apply.
    let edited = b"fn main() { /* user edit after apply */ }";
    std::fs::write(&dest, edited).unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();

    assert_eq!(rb.conflicts, 1, "edited destination must be a conflict");
    assert_eq!(rb.restored, 0, "nothing is restored on conflict");
    assert_eq!(rb.failures, 0);
    // The edit survives — the destination was never deleted.
    assert!(dest.exists());
    assert_eq!(std::fs::read(&dest).unwrap(), edited);
    // The item stays applied so it can be retried after the user resolves it.
    assert_eq!(
        store.applied_proposals_for_run(run_id).await.unwrap().len(),
        1,
    );
    // A conflicted run is NOT flipped to RolledBack.
    let run = store.get_run(run_id).await.unwrap().unwrap();
    assert_ne!(run.state, tidyup_domain::RunState::RolledBack);
}

#[tokio::test]
async fn rollback_with_missing_shelf_leaves_destination_untouched() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, dest) = apply_one_scan_move(&workdir).await;

    // Simulate a lost/corrupted shelf: delete the shelved copy on disk.
    let proposal = &store.applied_proposals_for_run(run_id).await.unwrap()[0];
    let record = store.find_by_change_id(proposal.id).await.unwrap().unwrap();
    std::fs::remove_file(&record.backup_path).unwrap();
    let dest_before = std::fs::read(&dest).unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();

    assert_eq!(rb.failures, 1, "missing shelf copy is a failure");
    assert_eq!(rb.restored, 0);
    // Destination is left exactly as it was — never deleted on a failed restore.
    assert!(dest.exists());
    assert_eq!(std::fs::read(&dest).unwrap(), dest_before);
    // Run is NOT marked RolledBack when every restore failed (idempotence).
    let run = store.get_run(run_id).await.unwrap().unwrap();
    assert_ne!(run.state, tidyup_domain::RunState::RolledBack);

    // Re-running is stable: still a failure, destination still untouched.
    let rb2 = rollback.rollback_run(run_id, &NullProgress).await.unwrap();
    assert_eq!(rb2.failures, 1);
    assert!(dest.exists());
}

#[tokio::test]
async fn re_rollback_after_success_is_a_noop() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, dest) = apply_one_scan_move(&workdir).await;
    let src_file = workdir.path().join("src/helpers.rs");

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();
    assert_eq!(rb.restored, 1);
    assert_eq!(rb.failures, 0);
    assert_eq!(rb.conflicts, 0);
    assert!(src_file.exists(), "original restored");
    assert!(!dest.exists(), "destination removed");
    let run = store.get_run(run_id).await.unwrap().unwrap();
    assert_eq!(run.state, tidyup_domain::RunState::RolledBack);

    // Second rollback: nothing left to do, no error, no double-restore.
    let rb2 = rollback.rollback_run(run_id, &NullProgress).await.unwrap();
    assert_eq!(rb2.restored, 0);
    assert_eq!(rb2.failures, 0);
    assert_eq!(rb2.conflicts, 0);
    assert!(src_file.exists());
}

#[tokio::test]
async fn rollback_never_overwrites_new_file_in_original_slot() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, dest) = apply_one_scan_move(&workdir).await;
    let src_file = workdir.path().join("src/helpers.rs");

    // The user saves a brand-new file into the vacated original slot.
    let new_content = b"totally new file, not the moved one";
    std::fs::write(&src_file, new_content).unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();

    assert_eq!(rb.conflicts, 1, "occupied original slot must be a conflict");
    assert_eq!(rb.restored, 0);
    assert_eq!(rb.failures, 0);
    // The new file survives untouched; the moved copy stays at the destination.
    assert_eq!(std::fs::read(&src_file).unwrap(), new_content);
    assert!(dest.exists());
    let run = store.get_run(run_id).await.unwrap().unwrap();
    assert_ne!(run.state, tidyup_domain::RunState::RolledBack);
}

#[tokio::test]
async fn rollback_restores_when_destination_already_gone() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, dest) = apply_one_scan_move(&workdir).await;
    let src_file = workdir.path().join("src/helpers.rs");

    // The user deleted (or re-moved) the destination after apply.
    std::fs::remove_file(&dest).unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();

    assert_eq!(rb.restored, 1, "missing destination is still restorable");
    assert_eq!(rb.failures, 0);
    assert_eq!(rb.conflicts, 0);
    assert!(src_file.exists(), "original restored from the shelf");
    assert_eq!(std::fs::read(&src_file).unwrap(), b"fn main() {}");
    let run = store.get_run(run_id).await.unwrap().unwrap();
    assert_eq!(run.state, tidyup_domain::RunState::RolledBack);
}

#[tokio::test]
async fn rollback_restore_failure_preserves_destination() {
    use tidyup_app::executor::{apply_loose_decisions, ExecutorDeps};
    use tidyup_app::RollbackService;

    // A move whose destination is NOT under the original's parent, so we can
    // sabotage the original side without touching the destination.
    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_dir = workdir.path().join("a");
    std::fs::create_dir_all(&src_dir).unwrap();
    let src = src_dir.join("orig.txt");
    std::fs::write(&src, b"payload").unwrap();
    let dest = workdir.path().join("b/moved.txt");

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
    let run = tidyup_domain::RunRecord::begin(tidyup_domain::RunMode::Scan, src_dir.clone(), None);
    store.record_run(&run).await.unwrap();
    let proposal = ChangeProposal {
        id: uuid::Uuid::new_v4(),
        file_id: None,
        change_type: tidyup_domain::ChangeType::Move,
        original_path: src.clone(),
        proposed_path: dest.clone(),
        proposed_name: "moved.txt".to_string(),
        confidence: 0.95,
        reasoning: "t".to_string(),
        needs_review: false,
        status: tidyup_domain::ChangeStatus::Pending,
        created_at: chrono::Utc::now(),
        applied_at: None,
        bundle_id: None,
        classification_confidence: Some(0.95),
        rename_mismatch_score: None,
        content_hash: None,
    };
    store
        .record_proposal(&proposal, Some(run.id))
        .await
        .unwrap();
    let deps = ExecutorDeps {
        change_log: ctx.change_log.as_ref(),
        backup_store: ctx.backup_store.as_ref(),
        progress: &NullProgress,
    };
    let ar = apply_loose_decisions(
        std::slice::from_ref(&proposal),
        &[ReviewDecision::Approve(proposal.id)],
        &deps,
        false,
    )
    .await
    .unwrap();
    assert_eq!(ar.applied, 1);
    assert!(dest.exists());

    // Make the restore copy fail after the destination has been displaced:
    // replace the (now-empty) original directory with a regular FILE of the
    // same name, so restore()'s create_dir_all errors. (Permission tricks
    // don't work here — tests may run as root.)
    std::fs::remove_dir_all(&src_dir).unwrap();
    std::fs::write(&src_dir, b"a file where the dir used to be").unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run.id, &NullProgress).await.unwrap();

    assert_eq!(rb.failures, 1, "blocked original parent is a failure");
    assert_eq!(rb.restored, 0);
    // The displaced destination was renamed back — the data is NOT stranded
    // solely on the shelf.
    assert!(
        dest.exists(),
        "failed restore must leave the destination in place"
    );
    assert_eq!(std::fs::read(&dest).unwrap(), b"payload");
    let run_row = store.get_run(run.id).await.unwrap().unwrap();
    assert_ne!(run_row.state, tidyup_domain::RunState::RolledBack);

    // With the obstruction removed, a retry completes.
    std::fs::remove_file(&src_dir).unwrap();
    let rb2 = rollback.rollback_run(run.id, &NullProgress).await.unwrap();
    assert_eq!(rb2.restored, 1);
    assert!(src.exists(), "original restored on retry");
    assert!(!dest.exists(), "destination cleaned up after retry");
}

/// Apply a 3-member file-set (document-series) bundle and return the context,
/// store, run id, and the members' (original, destination) path pairs.
async fn apply_invoice_bundle(
    workdir: &TempDir,
) -> (
    Arc<ServiceContext>,
    SqliteStore,
    uuid::Uuid,
    Vec<(std::path::PathBuf, std::path::PathBuf)>,
) {
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
    assert_eq!(report.bundles_applied, 1);

    let bundles = store.applied_bundles_for_run(report.run_id).await.unwrap();
    assert_eq!(bundles.len(), 1);
    let members = bundles[0]
        .members
        .iter()
        .map(|m| (m.original_path.clone(), m.proposed_path.clone()))
        .collect::<Vec<_>>();
    assert_eq!(members.len(), 3);
    for (_, dst) in &members {
        assert!(dst.exists());
    }
    (ctx, store, report.run_id, members)
}

#[tokio::test]
async fn file_set_rollback_conflicts_whole_bundle_when_one_member_edited() {
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, members) = apply_invoice_bundle(&workdir).await;

    // The user edits ONE member's destination after apply.
    let edited = b"user-edited invoice, do not destroy";
    std::fs::write(&members[1].1, edited).unwrap();

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();

    assert_eq!(rb.conflicts, 1, "edited member conflicts the whole bundle");
    assert_eq!(rb.bundles_restored, 0);
    assert_eq!(rb.failures, 0);
    // All-or-nothing: NO member was restored, the edit survives.
    for (orig, dst) in &members {
        assert!(!orig.exists(), "no member may be restored on conflict");
        assert!(dst.exists(), "every destination stays in place");
    }
    assert_eq!(std::fs::read(&members[1].1).unwrap(), edited);
    // The bundle stays applied so the rollback can be retried after resolution.
    assert_eq!(
        store.applied_bundles_for_run(run_id).await.unwrap().len(),
        1
    );
}

#[tokio::test]
async fn file_set_crash_mid_apply_is_recovered_by_rollback() {
    // Build, by hand, the exact on-disk + DB state a crash leaves mid-file-set
    // under write-ahead journaling: the bundle is marked applied first, then
    // member 0 is shelved+moved, then the process dies before members 1-2 are
    // touched. `rollback_run` must restore member 0 and skip the never-moved
    // members, converging to all-at-origin with no data loss.
    use tidyup_app::RollbackService;
    use tidyup_core::storage::{BackupStore as _, ChangeLog as _, RunLog as _};

    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let src_root = workdir.path().join("src");
    let tgt = workdir.path().join("Photos/Bursts/x");
    std::fs::create_dir_all(&src_root).unwrap();
    std::fs::create_dir_all(&tgt).unwrap();

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
    let run = tidyup_domain::RunRecord::begin(tidyup_domain::RunMode::Scan, src_root.clone(), None);
    store.record_run(&run).await.unwrap();

    // Three members forming a photo burst (a file-set bundle).
    let mut members = Vec::new();
    let mut paths = Vec::new();
    for n in ["IMG_001.jpg", "IMG_002.jpg", "IMG_003.jpg"] {
        let orig = src_root.join(n);
        std::fs::write(&orig, format!("bytes of {n}").as_bytes()).unwrap();
        let dst = tgt.join(n);
        paths.push((orig.clone(), dst.clone()));
        members.push(ChangeProposal {
            id: uuid::Uuid::new_v4(),
            file_id: None,
            change_type: tidyup_domain::ChangeType::Move,
            original_path: orig,
            proposed_path: dst,
            proposed_name: n.to_string(),
            confidence: 0.9,
            reasoning: "burst".to_string(),
            needs_review: false,
            status: tidyup_domain::ChangeStatus::Pending,
            created_at: chrono::Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: None,
            rename_mismatch_score: None,
            content_hash: None,
        });
    }
    let bundle = BundleProposal::new(
        src_root.join("burst"),
        tidyup_domain::BundleKind::PhotoBurst,
        tgt.clone(),
        members.clone(),
        0.9,
        "photo burst".to_string(),
    )
    .unwrap();
    store.record_bundle(&bundle, Some(run.id)).await.unwrap();

    // Write-ahead journal fires first...
    store.mark_bundle_applied(bundle.id).await.unwrap();
    // ...then member 0 is shelved + moved, and the process "crashes".
    let m0 = &bundle.members[0];
    let indexed = tidyup_domain::IndexedFile {
        id: tidyup_domain::FileId::new(),
        path: m0.original_path.clone(),
        name: "IMG_001.jpg".to_string(),
        extension: "jpg".to_string(),
        mime_type: "image/jpeg".to_string(),
        size_bytes: 0,
        content_hash: tidyup_domain::ContentHash(String::new()),
        indexed_at: chrono::Utc::now(),
    };
    store.shelve(&indexed, m0.id).await.unwrap();
    std::fs::rename(&paths[0].0, &paths[0].1).unwrap();

    // The run never finished — `status`-style detection surfaces it as an
    // interrupted (InProgress) run that rollback can reconcile.
    assert_eq!(
        store.get_run(run.id).await.unwrap().unwrap().state,
        tidyup_domain::RunState::InProgress,
    );
    // Sanity: member 0 moved+shelved; members 1-2 at origin, never shelved.
    assert!(!paths[0].0.exists() && paths[0].1.exists());
    assert!(store.find_by_change_id(m0.id).await.unwrap().is_some());
    assert!(store
        .find_by_change_id(bundle.members[1].id)
        .await
        .unwrap()
        .is_none());

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run.id, &NullProgress).await.unwrap();

    assert_eq!(rb.bundles_restored, 1, "partial bundle recovered: {rb:?}");
    assert_eq!(rb.failures, 0);
    assert_eq!(rb.conflicts, 0);
    // Everything ends at its origin; nothing stranded at a destination.
    for (orig, dst) in &paths {
        assert!(orig.exists(), "restored: {}", orig.display());
        assert!(!dst.exists(), "cleared: {}", dst.display());
    }
    assert_eq!(std::fs::read(&paths[0].0).unwrap(), b"bytes of IMG_001.jpg");
}

#[tokio::test]
async fn file_set_rollback_retry_completes_after_partial_restore() {
    use tidyup_app::RollbackService;
    use tidyup_core::storage::BackupStore as _;

    let workdir = TempDir::new().unwrap();
    let (ctx, store, run_id, members) = apply_invoice_bundle(&workdir).await;
    let bundle = &store.applied_bundles_for_run(run_id).await.unwrap()[0];

    // Simulate an interrupted earlier rollback: member 0 was already restored
    // (shelf copy back at the original, backup row flipped Unshelved,
    // destination removed) but the bundle never got marked unshelved.
    let first_member = &bundle.members[0];
    let first_record = store
        .find_by_change_id(first_member.id)
        .await
        .unwrap()
        .unwrap();
    store.restore(&first_record).await.unwrap();
    std::fs::remove_file(&members[0].1).unwrap();

    // Retry must recognise member 0 as done and complete the remaining two.
    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run_id, &NullProgress).await.unwrap();

    assert_eq!(rb.bundles_restored, 1, "retry converges: {rb:?}");
    assert_eq!(rb.failures, 0);
    assert_eq!(rb.conflicts, 0);
    for (orig, dst) in &members {
        assert!(orig.exists(), "every member restored: {}", orig.display());
        assert!(
            !dst.exists(),
            "every destination cleared: {}",
            dst.display()
        );
    }
    assert!(store
        .applied_bundles_for_run(run_id)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn rollback_of_a_rename_restores_the_original_name() {
    use tidyup_app::executor::{apply_loose_decisions, ExecutorDeps};
    use tidyup_app::RollbackService;

    let workdir = TempDir::new().unwrap();
    let shelf = workdir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    let dir = workdir.path().join("docs");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("scan001.pdf");
    std::fs::write(&src, b"tax return 2023").unwrap();
    // Rename-and-move: different basename in a different folder.
    let dst = dir.join("Finance/tax-return-2023.pdf");

    let (ctx, store) = make_ctx_with_shelf(Some(shelf));
    let run = tidyup_domain::RunRecord::begin(tidyup_domain::RunMode::Scan, dir.clone(), None);
    store.record_run(&run).await.unwrap();

    let proposal = ChangeProposal {
        id: uuid::Uuid::new_v4(),
        // No file_id: the change_proposals.file_id FK requires a persisted files
        // row, and the real pipeline records loose proposals with `None`.
        file_id: None,
        change_type: tidyup_domain::ChangeType::RenameAndMove,
        original_path: src.clone(),
        proposed_path: dst.clone(),
        proposed_name: "tax-return-2023.pdf".to_string(),
        confidence: 0.95,
        reasoning: "rename".to_string(),
        needs_review: false,
        status: tidyup_domain::ChangeStatus::Pending,
        created_at: chrono::Utc::now(),
        applied_at: None,
        bundle_id: None,
        classification_confidence: Some(0.95),
        rename_mismatch_score: Some(0.7),
        content_hash: None,
    };
    store
        .record_proposal(&proposal, Some(run.id))
        .await
        .unwrap();

    let deps = ExecutorDeps {
        change_log: ctx.change_log.as_ref(),
        backup_store: ctx.backup_store.as_ref(),
        progress: &NullProgress,
    };
    let ar = apply_loose_decisions(
        std::slice::from_ref(&proposal),
        &[ReviewDecision::Approve(proposal.id)],
        &deps,
        false,
    )
    .await
    .unwrap();
    assert_eq!(ar.applied, 1);
    assert!(!src.exists() && dst.exists(), "rename applied");

    let rollback = RollbackService::new(Arc::clone(&ctx));
    let rb = rollback.rollback_run(run.id, &NullProgress).await.unwrap();
    assert_eq!(rb.restored, 1);
    assert_eq!(rb.failures, 0);
    assert!(src.exists(), "rename rollback restores the original name");
    assert!(!dst.exists(), "renamed destination removed");
    assert_eq!(std::fs::read(&src).unwrap(), b"tax return 2023");
}
