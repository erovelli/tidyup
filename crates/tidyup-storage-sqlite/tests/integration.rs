//! Integration tests against a real on-disk `SQLite` database.
//!
//! Per `CLAUDE.md`: "No mocking at module boundaries. Tests use real `SQLite`, real fixtures,
//! real filesystem (`tempfile`)."

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use chrono::Utc;
use tempfile::TempDir;
use tidyup_core::storage::{BackupStore, ChangeLog, FileIndex};
use tidyup_domain::{
    BackupStatus, BundleKind, BundleProposal, ChangeProposal, ChangeStatus, ChangeType,
    ContentHash, FileId, IndexedFile,
};
use tidyup_storage_sqlite::SqliteStore;
use uuid::Uuid;

fn new_store(dir: &TempDir) -> SqliteStore {
    SqliteStore::open(&dir.path().join("tidyup.db")).unwrap()
}

fn new_store_with_shelf(dir: &TempDir) -> SqliteStore {
    let shelf = dir.path().join("shelf");
    std::fs::create_dir_all(&shelf).unwrap();
    new_store(dir).with_backup_root(shelf)
}

fn sample_file(path: &str, id: Uuid) -> IndexedFile {
    IndexedFile {
        id: FileId(id),
        path: PathBuf::from(path),
        name: path.rsplit('/').next().unwrap().to_string(),
        extension: "rs".to_string(),
        mime_type: "text/x-rust".to_string(),
        size_bytes: 128,
        content_hash: ContentHash("cafef00d".to_string()),
        indexed_at: Utc::now(),
    }
}

fn sample_proposal(src: &str, dst: &str, file: Option<FileId>) -> ChangeProposal {
    ChangeProposal {
        id: Uuid::new_v4(),
        file_id: file,
        change_type: ChangeType::Move,
        original_path: PathBuf::from(src),
        proposed_path: PathBuf::from(dst),
        proposed_name: dst.rsplit('/').next().unwrap().to_string(),
        confidence: 0.9,
        reasoning: "integration test".to_string(),
        needs_review: false,
        status: ChangeStatus::Pending,
        created_at: Utc::now(),
        applied_at: None,
        bundle_id: None,
        classification_confidence: Some(0.88),
        rename_mismatch_score: None,
    }
}

#[tokio::test]
async fn upsert_and_roundtrip_indexed_file() {
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let id = Uuid::new_v4();
    let f = sample_file("/code/main.rs", id);
    store.upsert(&f).await.unwrap();

    let got = store.by_path(&f.path).await.unwrap().unwrap();
    assert_eq!(got.id.0, id);
    assert_eq!(got.name, "main.rs");
    assert_eq!(got.content_hash.0, "cafef00d");
}

#[tokio::test]
async fn upsert_preserves_id_across_rescan() {
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let original = Uuid::new_v4();
    store
        .upsert(&sample_file("/code/lib.rs", original))
        .await
        .unwrap();

    // Simulate a rescan generating a fresh FileId for the same path.
    store
        .upsert(&sample_file("/code/lib.rs", Uuid::new_v4()))
        .await
        .unwrap();

    let got = store
        .by_path(&PathBuf::from("/code/lib.rs"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got.id.0, original, "FileId must be stable across re-scans");
}

#[tokio::test]
async fn record_bundle_persists_members_and_filters_from_pending() {
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let loose = sample_proposal("/src/loose.pdf", "/docs/loose.pdf", None);
    store.record_proposal(&loose, None).await.unwrap();

    let bundle = BundleProposal::new(
        PathBuf::from("/src/proj"),
        BundleKind::RustCrate,
        PathBuf::from("/code"),
        vec![
            sample_proposal("/src/proj/src/main.rs", "/code/proj/src/main.rs", None),
            sample_proposal("/src/proj/Cargo.toml", "/code/proj/Cargo.toml", None),
        ],
        0.93,
        "Cargo.toml at root".to_string(),
    )
    .unwrap();
    let bundle_id = bundle.id;
    store.record_bundle(&bundle, None).await.unwrap();

    let pending_loose = store.pending().await.unwrap();
    assert_eq!(
        pending_loose.len(),
        1,
        "bundle members must not appear in pending()"
    );
    assert_eq!(pending_loose[0].id, loose.id);

    let pending_bundles = store.pending_bundles().await.unwrap();
    assert_eq!(pending_bundles.len(), 1);
    let fetched = &pending_bundles[0];
    assert_eq!(fetched.id, bundle_id);
    assert_eq!(fetched.members.len(), 2);
    for m in &fetched.members {
        assert_eq!(m.bundle_id, Some(bundle_id));
    }
    let kinds = std::mem::discriminant(&fetched.kind);
    assert_eq!(kinds, std::mem::discriminant(&BundleKind::RustCrate));
}

#[tokio::test]
async fn record_bundle_rejects_bundle_member_via_record_proposal() {
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let mut p = sample_proposal("/a", "/b", None);
    p.bundle_id = Some(Uuid::new_v4());
    let err = store.record_proposal(&p, None).await.unwrap_err();
    assert!(err.to_string().contains("record_bundle"), "actual: {err}");
}

#[tokio::test]
async fn mark_bundle_applied_flips_bundle_and_members() {
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let bundle = BundleProposal::new(
        PathBuf::from("/src/proj"),
        BundleKind::NodeProject,
        PathBuf::from("/code"),
        vec![
            sample_proposal("/src/proj/package.json", "/code/proj/package.json", None),
            sample_proposal("/src/proj/index.js", "/code/proj/index.js", None),
        ],
        0.9,
        "package.json at root".to_string(),
    )
    .unwrap();
    let bundle_id = bundle.id;
    store.record_bundle(&bundle, None).await.unwrap();

    store.mark_bundle_applied(bundle_id).await.unwrap();

    let still_pending = store.pending_bundles().await.unwrap();
    assert!(
        still_pending.is_empty(),
        "applied bundle should not be pending"
    );
}

#[tokio::test]
async fn bundle_kind_with_pattern_roundtrips_through_sql() {
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let pattern = r"invoice-\d{4}-\d{2}\.pdf".to_string();
    let bundle = BundleProposal::new(
        PathBuf::from("/src/invoices"),
        BundleKind::DocumentSeries {
            pattern: pattern.clone(),
        },
        PathBuf::from("/docs"),
        vec![sample_proposal("/src/invoices/a.pdf", "/docs/a.pdf", None)],
        0.8,
        "monthly invoice series".to_string(),
    )
    .unwrap();
    store.record_bundle(&bundle, None).await.unwrap();

    let got = store.pending_bundles().await.unwrap();
    assert_eq!(got.len(), 1);
    match &got[0].kind {
        BundleKind::DocumentSeries { pattern: p } => assert_eq!(p, &pattern),
        other => panic!("wrong kind: {other:?}"),
    }
}

#[tokio::test]
async fn backup_shelve_restore_full_cycle_on_disk() {
    let dir = TempDir::new().unwrap();
    let store = new_store_with_shelf(&dir);

    let src = dir.path().join("report.txt");
    std::fs::write(&src, b"report contents").unwrap();
    let file = IndexedFile {
        id: FileId(Uuid::new_v4()),
        path: src.clone(),
        name: "report.txt".to_string(),
        extension: "txt".to_string(),
        mime_type: "text/plain".to_string(),
        size_bytes: 15,
        content_hash: ContentHash("feedface".to_string()),
        indexed_at: Utc::now(),
    };
    let change_id = Uuid::new_v4();

    let record = store.shelve(&file, change_id).await.unwrap();
    assert_eq!(record.change_id, change_id);
    assert_eq!(record.status, BackupStatus::Shelved);
    assert!(record.backup_path.starts_with(dir.path().join("shelf")));
    assert_eq!(
        std::fs::read(&record.backup_path).unwrap(),
        b"report contents"
    );

    // Simulate the move by removing the original, then restore it.
    std::fs::remove_file(&src).unwrap();
    store.restore(&record).await.unwrap();
    assert_eq!(std::fs::read(&src).unwrap(), b"report contents");
}

#[tokio::test]
async fn backup_prune_marks_expired_and_clears_disk() {
    let dir = TempDir::new().unwrap();
    let store = new_store_with_shelf(&dir);

    let src = dir.path().join("old.txt");
    std::fs::write(&src, b"stale").unwrap();
    let file = IndexedFile {
        id: FileId(Uuid::new_v4()),
        path: src,
        name: "old.txt".to_string(),
        extension: "txt".to_string(),
        mime_type: "text/plain".to_string(),
        size_bytes: 5,
        content_hash: ContentHash("abc".to_string()),
        indexed_at: Utc::now(),
    };
    let record = store.shelve(&file, Uuid::new_v4()).await.unwrap();

    // Backdate the shelf so the 30-day window catches it.
    let db_path = dir.path().join("tidyup.db");
    let id_str = record.id.to_string();
    let backdated = Utc::now() - chrono::Duration::days(90);
    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(db_path).unwrap();
        conn.execute(
            "UPDATE backups SET shelved_at = ?1 WHERE id = ?2",
            rusqlite::params![backdated, id_str],
        )
        .unwrap();
    })
    .await
    .unwrap();

    let pruned = store.prune_older_than_days(30).await.unwrap();
    assert_eq!(pruned, 1);
    assert!(
        !record.backup_path.exists(),
        "pruned shelf file must be gone"
    );
}

#[tokio::test]
async fn file_fk_cascade_on_bundle_delete() {
    // Verifies ON DELETE CASCADE on change_proposals.bundle_id: deleting a bundle removes its
    // member proposals. We drive this via raw SQL since the trait doesn't expose delete.
    let dir = TempDir::new().unwrap();
    let store = new_store(&dir);

    let bundle = BundleProposal::new(
        PathBuf::from("/src/proj"),
        BundleKind::Generic,
        PathBuf::from("/code"),
        vec![sample_proposal("/src/proj/a", "/code/proj/a", None)],
        0.7,
        "generic".to_string(),
    )
    .unwrap();
    let bundle_id = bundle.id;
    store.record_bundle(&bundle, None).await.unwrap();

    // Confirm one member exists.
    let before = store.pending_bundles().await.unwrap();
    assert_eq!(before[0].members.len(), 1);

    // Now delete via a second store handle to exercise FK cascade on a persistent DB.
    let db_path = dir.path().join("tidyup.db");
    tokio::task::spawn_blocking(move || {
        let conn = rusqlite::Connection::open(db_path).unwrap();
        conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
        conn.execute(
            "DELETE FROM bundles WHERE id = ?1",
            rusqlite::params![bundle_id.to_string()],
        )
        .unwrap();
        let remaining: i64 = conn
            .query_row(
                "SELECT count(*) FROM change_proposals WHERE bundle_id = ?1",
                rusqlite::params![bundle_id.to_string()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 0, "FK cascade should remove bundle members");
    })
    .await
    .unwrap();
}
