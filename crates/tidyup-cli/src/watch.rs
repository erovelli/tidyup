//! `tidyup watch` — advisory filesystem watch.
//!
//! Watches a source directory and, on each debounced batch of changes, re-runs
//! scan classification in **dry-run** and reports what it *would* propose. It
//! never moves anything — upholding the "nothing moves without approval"
//! promise. To actually organize, run `tidyup scan`.
//!
//! The embedding model + taxonomy candidates load **once** and are reused across
//! rescans, so a watch session pays the model-load cost a single time.
//!
//! # Threading
//!
//! `notify` delivers events on its own thread via a synchronous channel. A
//! dedicated blocking thread owns the watcher + receiver, coalesces bursts
//! within the debounce window (ignoring editor/temp churn), and signals the
//! async loop through a tokio channel. This keeps the blocking `recv` off the
//! async runtime.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};
use tidyup_app::config::TidyupConfig;
use tidyup_app::{scan::ScanRequest, ScanService};
use tidyup_core::frontend::{Level, ProgressReporter};
use tidyup_pipeline::scan::ScanCandidate;

use crate::context::{
    build, build_audio_scan_candidates, build_custom_scan_candidates,
    build_default_scan_candidates, build_image_scan_candidates, InferenceActivation,
};
use crate::reporter::CliReporter;
use crate::review::AutoApproveHandler;

/// Loose-proposal confidence above which a watch rescan counts a file as one it
/// *would* organize. Matches the CLI's `--yes` threshold.
const WATCH_MIN_CONFIDENCE: f32 = 0.75;
/// Bundle confidence threshold for the same "would organize" tally.
const WATCH_BUNDLE_MIN_CONFIDENCE: f32 = 0.85;

/// Run the advisory watch loop. Builds the context + scan candidates once,
/// performs an initial dry-run scan, then re-scans on each debounced batch of
/// relevant filesystem events until interrupted (Ctrl-C).
///
/// `yes` is intentionally ignored: watch is always dry-run, so it can never
/// move a file (which also sidesteps the apply→event→rescan feedback loop).
///
/// # Errors
/// Propagates context/model build, initial-scan, and watcher-setup errors.
pub(crate) async fn run_watch(
    json: bool,
    activation: InferenceActivation,
    cfg: &TidyupConfig,
    root: PathBuf,
    taxonomy: Option<PathBuf>,
    debounce_ms: u64,
) -> Result<()> {
    let ctx = build(cfg, true, activation).await?;
    let candidates = match taxonomy.as_deref() {
        Some(path) => build_custom_scan_candidates(path, ctx.embeddings.as_ref()).await?,
        None => build_default_scan_candidates(ctx.embeddings.as_ref()).await?,
    };
    let image_candidates = build_image_scan_candidates(ctx.image_embeddings.as_deref()).await?;
    let audio_candidates = build_audio_scan_candidates(ctx.audio_embeddings.as_deref()).await?;

    let service = ScanService::new(ctx);
    let reporter = CliReporter::new(json);
    let reviewer = AutoApproveHandler {
        min_confidence: WATCH_MIN_CONFIDENCE,
    };
    let candidate_set = CandidateSet {
        text: candidates,
        image: image_candidates,
        audio: audio_candidates,
    };

    let debounce = Duration::from_millis(debounce_ms);
    reporter
        .message(
            Level::Info,
            &format!(
                "watching {} (dry-run, debounce {debounce_ms}ms) — Ctrl-C to stop. \
                 Nothing will be moved; run `tidyup scan` to apply.",
                root.display(),
            ),
        )
        .await;

    // Initial pass so the user sees the current state immediately. A failure
    // here (e.g. unreadable root) is fatal.
    rescan(&service, &root, &candidate_set, &reporter, &reviewer).await?;

    // The watcher thread owns the notify watcher + its receiver and signals here.
    let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<()>(8);
    let watch_root = root.clone();
    let watcher_thread = std::thread::spawn(move || watch_thread(&watch_root, debounce, &batch_tx));

    while batch_rx.recv().await.is_some() {
        reporter
            .message(Level::Info, "change detected — re-scanning")
            .await;
        // A transient rescan error shouldn't tear down the watch; report + keep going.
        if let Err(e) = rescan(&service, &root, &candidate_set, &reporter, &reviewer).await {
            reporter
                .message(Level::Warn, &format!("rescan failed: {e}"))
                .await;
        }
    }

    // The channel closed → the watcher thread exited. Surface any setup error.
    match watcher_thread.join() {
        Ok(res) => res,
        Err(_) => anyhow::bail!("watch thread panicked"),
    }
}

/// The three per-modality candidate lists, built once and borrowed each rescan.
struct CandidateSet {
    text: Vec<ScanCandidate>,
    image: Vec<ScanCandidate>,
    audio: Vec<ScanCandidate>,
}

/// Run one dry-run scan and report a one-line summary.
async fn rescan(
    service: &ScanService,
    root: &Path,
    candidates: &CandidateSet,
    reporter: &CliReporter,
    reviewer: &AutoApproveHandler,
) -> Result<()> {
    let report = service
        .run(
            ScanRequest {
                root: root.to_path_buf(),
                // Candidates are pre-built; the service doesn't re-read taxonomy.
                taxonomy_path: None,
                dry_run: true,
                auto_approve_bundles: false,
                bundle_min_confidence: WATCH_BUNDLE_MIN_CONFIDENCE,
            },
            &candidates.text,
            &candidates.image,
            &candidates.audio,
            reporter,
            reviewer,
        )
        .await?;
    reporter
        .message(
            Level::Info,
            &format!(
                "{} proposal(s), {} would be organized (dry-run)",
                report.proposed, report.approved,
            ),
        )
        .await;
    Ok(())
}

/// Blocking watcher loop: set up `notify`, then coalesce event bursts and signal
/// `batch_tx` once per debounced batch that contains at least one relevant path.
/// Returns when the watcher errors or the async side hangs up.
fn watch_thread(
    root: &Path,
    debounce: Duration,
    batch_tx: &tokio::sync::mpsc::Sender<()>,
) -> Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        // If the receiver is gone the loop has exited; dropping the event is fine.
        let _ = tx.send(res);
    })
    .context("create filesystem watcher")?;
    watcher
        .watch(root, RecursiveMode::Recursive)
        .with_context(|| format!("watch {}", root.display()))?;

    loop {
        // Block until the first event of a burst (or the watcher is dropped).
        let Ok(first) = rx.recv() else {
            return Ok(());
        };
        if drain_relevant(&first, &rx, debounce) == 0 {
            continue; // burst was only editor/temp churn
        }
        if batch_tx.blocking_send(()).is_err() {
            return Ok(()); // async loop gone
        }
    }
}

/// Count relevant paths in `first` plus every event arriving within `window`,
/// draining the channel so a burst collapses into one batch.
fn drain_relevant(
    first: &notify::Result<notify::Event>,
    rx: &Receiver<notify::Result<notify::Event>>,
    window: Duration,
) -> usize {
    let mut count = count_relevant(first);
    let deadline = Instant::now() + window;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(remaining) {
            Ok(event) => count += count_relevant(&event),
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        }
    }
    count
}

fn count_relevant(event: &notify::Result<notify::Event>) -> usize {
    event.as_ref().map_or(0, |ev| {
        ev.paths.iter().filter(|p| is_watch_relevant(p)).count()
    })
}

/// Common editor/temp/partial-download suffixes that shouldn't trigger a rescan.
const NOISE_SUFFIXES: &[&str] = &["~", ".tmp", ".swp", ".swx", ".part", ".crdownload", ".lock"];

/// Whether a changed path should trigger a rescan. Filters dotfiles and common
/// editor/temp/partial-download churn so a single save doesn't storm rescans.
fn is_watch_relevant(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    if name.starts_with('.') {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    !NOISE_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::unnecessary_wraps)]
mod tests {
    use super::*;
    use notify::{Event, EventKind};

    #[test]
    fn relevant_accepts_ordinary_files() {
        assert!(is_watch_relevant(Path::new("/src/report.pdf")));
        assert!(is_watch_relevant(Path::new("/src/sub dir/photo.jpg")));
    }

    #[test]
    fn relevant_rejects_dotfiles_and_temp_churn() {
        assert!(!is_watch_relevant(Path::new("/src/.hidden")));
        assert!(!is_watch_relevant(Path::new("/src/.DS_Store")));
        assert!(!is_watch_relevant(Path::new("/src/draft.txt~")));
        assert!(!is_watch_relevant(Path::new("/src/.report.pdf.swp")));
        assert!(!is_watch_relevant(Path::new("/src/movie.mkv.part")));
        assert!(!is_watch_relevant(Path::new("/src/file.CRDOWNLOAD")));
        // A path with no file name (filesystem root) is never relevant.
        assert!(!is_watch_relevant(Path::new("/")));
    }

    fn event(path: &str) -> notify::Result<Event> {
        Ok(Event::new(EventKind::Any).add_path(PathBuf::from(path)))
    }

    #[test]
    fn drain_counts_only_relevant_paths_in_a_burst() {
        let (tx, rx) = std::sync::mpsc::channel();
        // Pre-fill a burst: two real files + one editor swap.
        tx.send(event("/src/b.txt")).unwrap();
        tx.send(event("/src/.c.txt.swp")).unwrap();
        let first = event("/src/a.txt");

        // Events are already queued, so only the final (empty) recv waits out the
        // short window — keeps the test fast and deterministic.
        let count = drain_relevant(&first, &rx, Duration::from_millis(40));
        assert_eq!(count, 2, "two relevant files, one swap ignored");
    }

    #[test]
    fn drain_zero_when_burst_is_all_noise() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(event("/src/.x.swp")).unwrap();
        let first = event("/src/.hidden");
        assert_eq!(drain_relevant(&first, &rx, Duration::from_millis(40)), 0);
    }
}
