//! `cargo xtask eval-routing` — held-out migration-routing accuracy with baselines.
//!
//! # What this measures (and why it's the falsifiable test)
//!
//! tidyup's premise is "route files by their **contents**, not their filename."
//! That is a falsifiable claim, and this harness is the experiment that can
//! refute it. It uses **already-organized directories as ground truth**: the
//! folder a human filed a document in *is* its correct label — free, real
//! labels at scale, no manual annotation. (Off-the-shelf corpora like 20
//! Newsgroups, BBC-News-by-category, or arXiv-by-subject drop straight in.)
//!
//! Protocol (stratified held-out):
//!   1. Each immediate subdirectory of `--corpus` is a label; files under it are
//!      its members.
//!   2. Per label, deterministically split members into a **train** set (used to
//!      build the folder's content centroid) and a **test** set (routed + scored).
//!   3. Route each test file and compare the predicted folder to its true folder.
//!
//! The routing rule replicated here is migration mode's embedding core: a
//! folder's centroid is the mean of its train documents' (L2-normalized)
//! content embeddings, and a file routes to the folder whose centroid is most
//! cosine-similar. It uses the **real** [`OrtEmbeddings`] backend and the
//! **real** [`cosine_similarity`]/[`l2_normalize`] — not a re-implementation.
//!
//! # The baselines are the point
//!
//! A bare accuracy number means nothing. We report, on the same split:
//!   * **content** — route on the embedded file *content* (the thing under test);
//!   * **filename** — identical routing on the embedded *filename* only (the
//!     baseline tidyup claims to beat — if content doesn't beat this, the
//!     premise is wrong);
//!   * **most-frequent** — always predict the largest folder (chance floor);
//!   * **extension** — predict the folder where the file's extension is most
//!     common in train.
//!
//! The headline is the **delta** (content − filename, content − most-frequent),
//! with a bootstrap 95% CI on each method's top-1. `--fail-under <margin>` turns
//! the content-vs-filename delta into a CI gate so a regression fails the lane.
//!
//! # Honest limits
//!
//! Public corpora are cleanly separable, so they are an **upper bound** ("if it
//! can't sort 20NG it can't sort your Downloads" — necessary, not sufficient);
//! the real test is a consented personal corpus. Reported accuracy is bounded by
//! inter-annotator agreement on the labels. And like the rest of `eval`, this
//! needs the embedding model installed — it is **not** part of model-free CI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tidyup_core::inference::EmbeddingBackend;
use tidyup_embeddings_ort::{
    cosine_similarity, installation_instructions, l2_normalize, verify_default_model, OrtEmbeddings,
};

/// Max characters of file content embedded per file (bounds memory; the encoder
/// truncates to ~512 tokens regardless).
const CONTENT_CAP: usize = 10_000;
/// Bootstrap resamples for the top-1 confidence interval.
const BOOTSTRAP_ITERS: usize = 2_000;

/// Knobs for one evaluation run.
#[derive(Debug, Clone, Copy)]
struct Config {
    /// Fraction of each label's files used to build the centroid (rest are tested).
    train_frac: f64,
    /// Seed for the deterministic split + bootstrap.
    seed: u64,
    /// If set, fail the process when (content − filename) top-1 is below this.
    fail_under: Option<f64>,
}

/// A corpus file with its ground-truth label (its top-level folder name).
#[derive(Debug, Clone)]
struct LabeledFile {
    label: String,
    path: PathBuf,
}

/// One routing method's accuracy, with a bootstrap CI on top-1.
#[derive(Debug, Clone, Serialize)]
struct MethodReport {
    method: String,
    top1: f64,
    top3: f64,
    top1_ci_low: f64,
    top1_ci_high: f64,
    n: usize,
}

/// Full report for one run.
#[derive(Debug, Clone, Serialize)]
struct RoutingReport {
    labels: usize,
    train_files: usize,
    test_files: usize,
    skipped_unreadable: usize,
    dims: usize,
    methods: Vec<MethodReport>,
    /// Per-label top-1 for the content method (where it struggles).
    content_top1_by_label: BTreeMap<String, f64>,
    content_vs_filename_top1: f64,
    content_vs_most_frequent_top1: f64,
    /// `true` iff content beats *both* baselines on top-1.
    premise_supported: bool,
}

/// Per-test-file ranked predictions (labels, best first) under one method.
struct Prediction {
    truth: String,
    ranked: Vec<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the held-out routing eval over `corpus` and print the report.
///
/// # Errors
/// Fails if the model is missing, the corpus is malformed (fewer than two
/// labels, or a label with no train/test files), or — when `--fail-under` is
/// set — the content-vs-filename top-1 delta is below the gate.
#[allow(unreachable_pub)] // bin-crate entry point; mirrors `eval::run`
pub fn run(
    corpus: &Path,
    train_frac: f64,
    seed: u64,
    fail_under: Option<f64>,
    json: bool,
) -> Result<()> {
    let cfg = Config {
        train_frac,
        seed,
        fail_under,
    };
    if let Err(e) = verify_default_model() {
        bail!(
            "eval-routing needs the embedding model installed.\n{e}\n\n{}",
            installation_instructions()
        );
    }
    let files = load_corpus(corpus)?;
    let runtime = tokio::runtime::Runtime::new().context("build tokio runtime")?;
    let report = runtime.block_on(async {
        let embeddings = OrtEmbeddings::load_default().context("load bge-small embedding model")?;
        evaluate(&files, cfg, &embeddings).await
    })?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("serialize routing report")?
        );
    } else {
        print_report(corpus, &report);
    }

    if let Some(margin) = cfg.fail_under {
        if report.content_vs_filename_top1 < margin {
            bail!(
                "gate failed: content beat filename on top-1 by {:.3}, below the required {:.3}",
                report.content_vs_filename_top1,
                margin,
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Evaluation (model-dependent orchestration; pure helpers below)
// ---------------------------------------------------------------------------

async fn evaluate(
    corpus: &[LabeledFile],
    cfg: Config,
    embeddings: &dyn EmbeddingBackend,
) -> Result<RoutingReport> {
    let labels = distinct_labels(corpus);
    if labels.len() < 2 {
        bail!("need at least 2 labelled folders; found {}", labels.len());
    }
    let (train_idx, test_idx) = stratified_split(corpus, cfg.train_frac, cfg.seed);
    if train_idx.is_empty() || test_idx.is_empty() {
        bail!(
            "split left {} train / {} test files; add more files per folder or adjust --train-frac",
            train_idx.len(),
            test_idx.len(),
        );
    }

    // Read content + derive filenames for both splits, dropping unreadable files.
    let (train_docs, _train_skipped) = read_docs(corpus, &train_idx);
    let (test_docs, test_skipped) = read_docs(corpus, &test_idx);
    if train_docs.is_empty() || test_docs.is_empty() {
        bail!("no readable text files after filtering (corpus must be UTF-8 text)");
    }

    // Embed everything (batched per split/field).
    let train_content_emb =
        embed(embeddings, train_docs.iter().map(|d| d.content.as_str())).await?;
    let train_name_emb = embed(embeddings, train_docs.iter().map(|d| d.name.as_str())).await?;
    let test_content_emb = embed(embeddings, test_docs.iter().map(|d| d.content.as_str())).await?;
    let test_name_emb = embed(embeddings, test_docs.iter().map(|d| d.name.as_str())).await?;
    let dims = train_content_emb.first().map_or(0, Vec::len);

    // Build per-label centroids (content and filename) + count tables.
    let content_centroids = centroids_by_label(&train_docs, &train_content_emb);
    let name_centroids = centroids_by_label(&train_docs, &train_name_emb);
    let label_counts = count_by_label(&train_docs);
    let ext_counts = count_by_ext_label(&train_docs);
    let most_frequent_ranking = ranking_by_count(&label_counts);

    // Route every test file under each method.
    let mut content = Vec::with_capacity(test_docs.len());
    let mut filename = Vec::with_capacity(test_docs.len());
    let mut most_frequent = Vec::with_capacity(test_docs.len());
    let mut extension = Vec::with_capacity(test_docs.len());
    for (i, doc) in test_docs.iter().enumerate() {
        content.push(Prediction {
            truth: doc.label.clone(),
            ranked: rank_against(&test_content_emb[i], &content_centroids),
        });
        filename.push(Prediction {
            truth: doc.label.clone(),
            ranked: rank_against(&test_name_emb[i], &name_centroids),
        });
        most_frequent.push(Prediction {
            truth: doc.label.clone(),
            ranked: most_frequent_ranking.clone(),
        });
        extension.push(Prediction {
            truth: doc.label.clone(),
            ranked: extension_ranking(&doc.ext, &ext_counts, &most_frequent_ranking),
        });
    }

    let content_report = method_metrics("content", &content, cfg.seed);
    let filename_report = method_metrics("filename", &filename, cfg.seed);
    let freq_report = method_metrics("most-frequent", &most_frequent, cfg.seed);
    let ext_report = method_metrics("extension", &extension, cfg.seed);

    let content_vs_filename = content_report.top1 - filename_report.top1;
    let content_vs_freq = content_report.top1 - freq_report.top1;

    Ok(RoutingReport {
        labels: labels.len(),
        train_files: train_docs.len(),
        test_files: test_docs.len(),
        skipped_unreadable: test_skipped,
        dims,
        content_top1_by_label: per_label_top1(&content),
        content_vs_filename_top1: content_vs_filename,
        content_vs_most_frequent_top1: content_vs_freq,
        premise_supported: content_vs_filename > 0.0 && content_vs_freq > 0.0,
        methods: vec![content_report, filename_report, freq_report, ext_report],
    })
}

/// Embed a batch of texts, preserving order.
async fn embed<'a>(
    backend: &dyn EmbeddingBackend,
    texts: impl Iterator<Item = &'a str>,
) -> Result<Vec<Vec<f32>>> {
    let owned: Vec<&str> = texts.collect();
    let mut out = backend.embed_texts(&owned).await.context("embed batch")?;
    for v in &mut out {
        l2_normalize(v);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Corpus loading
// ---------------------------------------------------------------------------

fn load_corpus(dir: &Path) -> Result<Vec<LabeledFile>> {
    if !dir.is_dir() {
        bail!("corpus path is not a directory: {}", dir.display());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || !path.is_dir() {
            continue;
        }
        collect_files(&path, &name, &mut out)?;
    }
    if out.is_empty() {
        bail!(
            "no files found under {} (expected <corpus>/<label>/<files>)",
            dir.display()
        );
    }
    Ok(out)
}

/// Recursively collect non-hidden files under `dir`, all attributed to `label`.
fn collect_files(dir: &Path, label: &str, out: &mut Vec<LabeledFile>) -> Result<()> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).with_context(|| format!("read {}", d.display()))? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(LabeledFile {
                    label: label.to_string(),
                    path,
                });
            }
        }
    }
    Ok(())
}

/// A read + featurized document.
struct Doc {
    label: String,
    content: String,
    name: String,
    ext: String,
}

/// Read + featurize the files at `indices`, skipping unreadable (non-UTF-8) ones.
/// Returns the docs and the skip count.
fn read_docs(corpus: &[LabeledFile], indices: &[usize]) -> (Vec<Doc>, usize) {
    let mut docs = Vec::with_capacity(indices.len());
    let mut skipped = 0;
    for &i in indices {
        let f = &corpus[i];
        match read_text_capped(&f.path, CONTENT_CAP) {
            Some(content) => docs.push(Doc {
                label: f.label.clone(),
                content,
                name: filename_text(&f.path),
                ext: extension_of(&f.path),
            }),
            None => skipped += 1,
        }
    }
    (docs, skipped)
}

fn read_text_capped(path: &Path, cap: usize) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    Some(s.chars().take(cap).collect())
}

/// Filename stem as embeddable text: non-alphanumerics become spaces, so
/// `invoice_2024_q1.pdf` → `invoice 2024 q1`.
fn filename_text(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    stem.chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect()
}

fn extension_of(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested without a model)
// ---------------------------------------------------------------------------

fn distinct_labels(corpus: &[LabeledFile]) -> Vec<String> {
    let mut v: Vec<String> = corpus.iter().map(|f| f.label.clone()).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Stratified split: within each label, deterministically shuffle and take the
/// first `train_frac` as train (at least 1), the rest as test.
fn stratified_split(
    corpus: &[LabeledFile],
    train_frac: f64,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let mut by_label: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in corpus.iter().enumerate() {
        by_label.entry(f.label.as_str()).or_default().push(i);
    }
    let mut rng = Rng::new(seed);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (_, mut idxs) in by_label {
        rng.shuffle(&mut idxs);
        let n = idxs.len();
        // At least one train; if a label has a single file it becomes train-only.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let n_train = ((n as f64 * train_frac).round() as usize).clamp(1, n);
        train.extend_from_slice(&idxs[..n_train]);
        test.extend_from_slice(&idxs[n_train..]);
    }
    train.sort_unstable();
    test.sort_unstable();
    (train, test)
}

/// Centroid = mean of the (already L2-normalized) vectors, re-normalized.
fn centroid(vectors: &[&[f32]]) -> Vec<f32> {
    let Some(dim) = vectors.first().map(|v| v.len()) else {
        return Vec::new();
    };
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n = vectors.len() as f32;
    for a in &mut acc {
        *a /= n;
    }
    l2_normalize(&mut acc);
    acc
}

fn centroids_by_label(docs: &[Doc], embeddings: &[Vec<f32>]) -> BTreeMap<String, Vec<f32>> {
    let mut groups: BTreeMap<String, Vec<&[f32]>> = BTreeMap::new();
    for (doc, emb) in docs.iter().zip(embeddings) {
        groups
            .entry(doc.label.clone())
            .or_default()
            .push(emb.as_slice());
    }
    groups
        .into_iter()
        .map(|(label, vecs)| (label, centroid(&vecs)))
        .collect()
}

/// Rank labels by descending cosine of `query` to each centroid.
fn rank_against(query: &[f32], centroids: &BTreeMap<String, Vec<f32>>) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = centroids
        .iter()
        .map(|(label, c)| (label.clone(), cosine_similarity(query, c)))
        .collect();
    // Descending score; tie-break on label name for determinism.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().map(|(label, _)| label).collect()
}

fn count_by_label(docs: &[Doc]) -> BTreeMap<String, usize> {
    let mut m: BTreeMap<String, usize> = BTreeMap::new();
    for d in docs {
        *m.entry(d.label.clone()).or_default() += 1;
    }
    m
}

fn count_by_ext_label(docs: &[Doc]) -> BTreeMap<String, BTreeMap<String, usize>> {
    let mut m: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for d in docs {
        *m.entry(d.ext.clone())
            .or_default()
            .entry(d.label.clone())
            .or_default() += 1;
    }
    m
}

/// Labels ordered by descending count (tie-break on name). Used as the
/// most-frequent ranking and as the extension fallback.
fn ranking_by_count(counts: &BTreeMap<String, usize>) -> Vec<String> {
    let mut v: Vec<(&String, &usize)> = counts.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    v.into_iter().map(|(label, _)| label.clone()).collect()
}

fn extension_ranking(
    ext: &str,
    ext_counts: &BTreeMap<String, BTreeMap<String, usize>>,
    fallback: &[String],
) -> Vec<String> {
    ext_counts.get(ext).map_or_else(
        || fallback.to_vec(),
        |by_label| {
            let mut ranked = ranking_by_count(by_label);
            // Append any labels unseen for this extension so the ranking is total.
            for label in fallback {
                if !ranked.contains(label) {
                    ranked.push(label.clone());
                }
            }
            ranked
        },
    )
}

fn top_k_hit(truth: &str, ranked: &[String], k: usize) -> bool {
    ranked.iter().take(k).any(|l| l == truth)
}

#[allow(clippy::cast_precision_loss)]
fn method_metrics(name: &str, preds: &[Prediction], seed: u64) -> MethodReport {
    let n = preds.len();
    let hits1: Vec<bool> = preds
        .iter()
        .map(|p| top_k_hit(&p.truth, &p.ranked, 1))
        .collect();
    let top1 = mean_bool(&hits1);
    let top3 = preds
        .iter()
        .filter(|p| top_k_hit(&p.truth, &p.ranked, 3))
        .count() as f64
        / n.max(1) as f64;
    let (lo, hi) = bootstrap_ci(&hits1, seed);
    MethodReport {
        method: name.to_string(),
        top1,
        top3,
        top1_ci_low: lo,
        top1_ci_high: hi,
        n,
    }
}

fn per_label_top1(preds: &[Prediction]) -> BTreeMap<String, f64> {
    let mut hits: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for p in preds {
        let e = hits.entry(p.truth.clone()).or_insert((0, 0));
        e.1 += 1;
        if top_k_hit(&p.truth, &p.ranked, 1) {
            e.0 += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    hits.into_iter()
        .map(|(label, (correct, total))| (label, correct as f64 / total.max(1) as f64))
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn mean_bool(xs: &[bool]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().filter(|b| **b).count() as f64 / xs.len() as f64
}

/// Percentile bootstrap 95% CI for the mean of a 0/1 sample. Deterministic in
/// `seed`. Returns `(top1, top1)` for empty/degenerate input.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn bootstrap_ci(hits: &[bool], seed: u64) -> (f64, f64) {
    let n = hits.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = Rng::new(seed ^ 0xB529_7A4D);
    let mut means: Vec<f64> = Vec::with_capacity(BOOTSTRAP_ITERS);
    for _ in 0..BOOTSTRAP_ITERS {
        let mut c = 0usize;
        for _ in 0..n {
            if hits[rng.below(n)] {
                c += 1;
            }
        }
        means.push(c as f64 / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[(BOOTSTRAP_ITERS as f64 * 0.025) as usize];
    let hi = means[((BOOTSTRAP_ITERS as f64 * 0.975) as usize).min(BOOTSTRAP_ITERS - 1)];
    (lo, hi)
}

/// Minimal deterministic PRNG (splitmix64) — avoids a `rand` dependency for the
/// split shuffle + bootstrap resampling.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    const fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        #[allow(clippy::cast_possible_truncation)]
        ((self.next_u64() % n as u64) as usize)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_report(corpus: &Path, r: &RoutingReport) {
    println!("tidyup eval-routing — held-out folder-as-label routing\n");
    println!("  corpus:   {}", corpus.display());
    println!(
        "  labels:   {}   train: {}   test: {}   (skipped unreadable: {})   dims: {}",
        r.labels, r.train_files, r.test_files, r.skipped_unreadable, r.dims,
    );
    println!("\n  method          top-1            (95% CI)        top-3");
    println!("  ------------------------------------------------------------");
    for m in &r.methods {
        println!(
            "  {:<14}  {:>5.1}%          [{:>4.1}, {:>4.1}]    {:>5.1}%",
            m.method,
            m.top1 * 100.0,
            m.top1_ci_low * 100.0,
            m.top1_ci_high * 100.0,
            m.top3 * 100.0,
        );
    }
    println!(
        "\n  content vs filename (top-1):      {:+.1} pts",
        r.content_vs_filename_top1 * 100.0
    );
    println!(
        "  content vs most-frequent (top-1): {:+.1} pts",
        r.content_vs_most_frequent_top1 * 100.0
    );
    println!(
        "\n  verdict: {}",
        if r.premise_supported {
            "content routing beats both baselines — premise supported on this corpus"
        } else {
            "content routing does NOT beat both baselines — PREMISE NOT SUPPORTED here"
        },
    );

    // Surface the weakest labels — where to look next.
    let mut weak: Vec<(&String, &f64)> = r.content_top1_by_label.iter().collect();
    weak.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
    if !weak.is_empty() {
        println!("\n  weakest labels (content top-1):");
        for (label, acc) in weak.iter().take(5) {
            println!("    {:>5.1}%  {label}", **acc * 100.0);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use super::*;

    fn lf(label: &str, path: &str) -> LabeledFile {
        LabeledFile {
            label: label.to_string(),
            path: PathBuf::from(path),
        }
    }

    #[test]
    fn split_is_deterministic_and_stratified() {
        let corpus: Vec<LabeledFile> = (0..10)
            .map(|i| lf(if i < 5 { "a" } else { "b" }, &format!("/c/f{i}")))
            .collect();
        let (train_a, test_a) = stratified_split(&corpus, 0.6, 7);
        let (train_b, test_b) = stratified_split(&corpus, 0.6, 7);
        assert_eq!(train_a, train_b, "same seed => same split");
        assert_eq!(test_a, test_b);
        assert_eq!(train_a.len() + test_a.len(), 10, "every file placed once");
        // Each label (5 files, 0.6) => 3 train, 2 test.
        assert_eq!(train_a.len(), 6);
        assert_eq!(test_a.len(), 4);
    }

    #[test]
    fn single_file_label_goes_to_train() {
        let corpus = vec![lf("solo", "/c/x"), lf("b", "/c/y"), lf("b", "/c/z")];
        let (train, test) = stratified_split(&corpus, 0.7, 1);
        // "solo" must be in train (clamped to >=1), never test.
        assert!(train.contains(&0));
        assert!(!test.contains(&0));
    }

    #[test]
    fn centroid_is_mean_then_normalized() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        let c = centroid(&[&a[..], &b[..]]);
        // mean = (0.5, 0.5) -> normalized to (1/√2, 1/√2).
        let inv = 1.0 / 2.0f32.sqrt();
        assert!((c[0] - inv).abs() < 1e-6);
        assert!((c[1] - inv).abs() < 1e-6);
    }

    #[test]
    fn rank_orders_by_cosine() {
        let mut centroids = BTreeMap::new();
        centroids.insert("x".to_string(), vec![1.0, 0.0]);
        centroids.insert("y".to_string(), vec![0.0, 1.0]);
        let ranked = rank_against(&[0.9, 0.1], &centroids);
        assert_eq!(ranked, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn extension_ranking_falls_back_and_is_total() {
        let mut ext_counts: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
        let mut pdf = BTreeMap::new();
        pdf.insert("finance".to_string(), 3);
        ext_counts.insert("pdf".to_string(), pdf);
        let fallback = vec!["finance".to_string(), "code".to_string()];

        let pdf_rank = extension_ranking("pdf", &ext_counts, &fallback);
        assert_eq!(pdf_rank[0], "finance");
        assert!(
            pdf_rank.contains(&"code".to_string()),
            "ranking must be total"
        );

        // Unknown extension uses the fallback verbatim.
        assert_eq!(extension_ranking("zip", &ext_counts, &fallback), fallback);
    }

    #[test]
    fn metrics_count_top1_and_top3() {
        let preds = vec![
            Prediction {
                truth: "a".into(),
                ranked: vec!["a".into(), "b".into(), "c".into()],
            }, // top1 hit
            Prediction {
                truth: "a".into(),
                ranked: vec!["b".into(), "a".into(), "c".into()],
            }, // top3 only
            Prediction {
                truth: "a".into(),
                ranked: vec!["b".into(), "c".into(), "d".into()],
            }, // miss
        ];
        let m = method_metrics("t", &preds, 1);
        assert!((m.top1 - 1.0 / 3.0).abs() < 1e-9);
        assert!((m.top3 - 2.0 / 3.0).abs() < 1e-9);
        assert!(m.top1_ci_low <= m.top1 && m.top1 <= m.top1_ci_high);
    }

    #[test]
    fn bootstrap_ci_brackets_extremes() {
        assert_eq!(bootstrap_ci(&[true, true, true], 1), (1.0, 1.0));
        assert_eq!(bootstrap_ci(&[false, false], 1), (0.0, 0.0));
        assert_eq!(bootstrap_ci(&[], 1), (0.0, 0.0));
    }

    // -------------------------------------------------------------------
    // Full-harness test with a deterministic stub backend: proves the
    // instrument measures the content-vs-filename delta correctly, without
    // the real model.
    // -------------------------------------------------------------------

    /// Embeds text into a 3-bucket space by category keyword. Content carries the
    /// signal; uninformative filenames collapse to the same vector.
    struct BucketBackend;

    #[async_trait::async_trait]
    impl EmbeddingBackend for BucketBackend {
        async fn embed_text(&self, text: &str) -> tidyup_core::Result<Vec<f32>> {
            let t = text.to_lowercase();
            let mut v = vec![0.01f32; 3];
            if t.contains("invoice") || t.contains("tax") || t.contains("payment") {
                v[0] += 1.0;
            }
            if t.contains("struct") || t.contains("impl") || t.contains("function") {
                v[1] += 1.0;
            }
            if t.contains("flight") || t.contains("hotel") || t.contains("itinerary") {
                v[2] += 1.0;
            }
            Ok(v)
        }
        async fn embed_texts(&self, texts: &[&str]) -> tidyup_core::Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed_text(t).await?);
            }
            Ok(out)
        }
        fn dimensions(&self) -> usize {
            3
        }
        fn model_id(&self) -> &str {
            "bucket-test"
        }
    }

    fn doc(label: &str, content: &str, name: &str) -> Doc {
        Doc {
            label: label.to_string(),
            content: content.to_string(),
            name: name.to_string(),
            ext: "txt".to_string(),
        }
    }

    #[test]
    fn harness_detects_content_beating_filename() {
        // Build train/test docs directly (bypassing disk) and exercise the same
        // centroid + routing + metrics the binary uses.
        let train = vec![
            doc("finance", "invoice tax payment due", "a"),
            doc("finance", "tax invoice payment", "b"),
            doc("code", "struct impl function body", "c"),
            doc("code", "function struct return", "d"),
            doc("travel", "flight hotel itinerary booking", "e"),
            doc("travel", "hotel flight itinerary", "f"),
        ];
        let test = [
            doc("finance", "tax invoice", "x1"),
            doc("code", "impl struct", "x2"),
            doc("travel", "flight itinerary", "x3"),
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let (content, filename) = rt.block_on(async {
            let backend = BucketBackend;
            let train_content = embed(&backend, train.iter().map(|d| d.content.as_str()))
                .await
                .unwrap();
            let train_name = embed(&backend, train.iter().map(|d| d.name.as_str()))
                .await
                .unwrap();
            let test_content = embed(&backend, test.iter().map(|d| d.content.as_str()))
                .await
                .unwrap();
            let test_name = embed(&backend, test.iter().map(|d| d.name.as_str()))
                .await
                .unwrap();
            let content_centroids = centroids_by_label(&train, &train_content);
            let name_centroids = centroids_by_label(&train, &train_name);
            let content: Vec<Prediction> = test
                .iter()
                .enumerate()
                .map(|(i, d)| Prediction {
                    truth: d.label.clone(),
                    ranked: rank_against(&test_content[i], &content_centroids),
                })
                .collect();
            let filename: Vec<Prediction> = test
                .iter()
                .enumerate()
                .map(|(i, d)| Prediction {
                    truth: d.label.clone(),
                    ranked: rank_against(&test_name[i], &name_centroids),
                })
                .collect();
            (content, filename)
        });

        let cm = method_metrics("content", &content, 1);
        let fm = method_metrics("filename", &filename, 1);
        assert!(
            cm.top1 > 0.9,
            "content should route near-perfectly, got {}",
            cm.top1
        );
        assert!(
            cm.top1 - fm.top1 > 0.3,
            "content ({}) must clearly beat uninformative filenames ({})",
            cm.top1,
            fm.top1,
        );
    }

    #[test]
    fn corpus_loading_walks_label_folders() {
        let dir = tempfile::tempdir().unwrap();
        for (label, files) in [("finance", ["a.txt", "b.txt"]), ("code", ["c.rs", "d.rs"])] {
            let sub = dir.path().join(label);
            std::fs::create_dir_all(&sub).unwrap();
            for f in files {
                std::fs::write(sub.join(f), "x").unwrap();
            }
        }
        // A hidden dir is ignored.
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git").join("HEAD"), "ref").unwrap();

        let files = load_corpus(dir.path()).unwrap();
        assert_eq!(files.len(), 4);
        assert_eq!(
            distinct_labels(&files),
            vec!["code".to_string(), "finance".to_string()]
        );
    }
}
