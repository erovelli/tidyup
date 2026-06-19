//! `cargo xtask eval` — classification-accuracy harness over a labeled corpus.
//!
//! Runs the deterministic cascade against the golden corpus under
//! `xtask/corpus/` and reports overall accuracy, per-label precision / recall /
//! F1, tier coverage, and the top confusions.
//!
//! # Why two tiers behave differently
//!
//! Tier-1 heuristics need no model, so they always run — this harness is
//! meaningful even on a machine (or CI host) without the embedding bundle. The
//! Tier-2 embedding path is gated on `bge-small-en-v1.5` being installed
//! (`verify_default_model`); when it is absent, content-dependent entries are
//! reported as *deferred* rather than failed, and `--json`/text output says so.
//!
//! This is a developer + calibration tool (it feeds the Stage-5 confidence
//! calibration work). It is intentionally **not** wired into `cargo xtask ci`,
//! which must stay model-free.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use tidyup_embeddings_ort::{verify_default_model, EmbeddingClassifier, OrtEmbeddings};
use tidyup_pipeline::heuristics;

/// The cascade tier that produced a prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Tier {
    Heuristic,
    Embedding,
    Unresolved,
}

/// One row of the corpus manifest (`corpus.toml`).
#[derive(Debug, Clone, Deserialize)]
struct CorpusEntry {
    /// Path to the fixture, relative to the corpus directory.
    file: String,
    /// Taxonomy leaf the cascade is expected to route the file to.
    expected: String,
    /// Whether Tier-1 heuristics alone should resolve this entry.
    #[serde(default)]
    tier1: bool,
}

/// Top-level shape of `corpus.toml` (`[[entry]]` tables).
#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(rename = "entry")]
    entries: Vec<CorpusEntry>,
}

/// The outcome of classifying a single corpus entry.
#[derive(Debug, Clone)]
struct Outcome {
    expected: String,
    /// `None` when no evaluated tier produced a prediction (unresolved).
    predicted: Option<String>,
    tier: Tier,
    /// Whether the corpus declared this entry Tier-1-resolvable. Used to flag
    /// heuristic regressions (a `tier1 = true` entry that did not land at
    /// Tier 1).
    expected_tier1: bool,
}

impl Outcome {
    fn is_correct(&self) -> bool {
        self.predicted
            .as_deref()
            .is_some_and(|p| label_matches(&self.expected, p))
    }
}

/// Per-label confusion-matrix tallies.
#[derive(Debug, Default, Clone, Copy)]
struct LabelCounts {
    /// Number of corpus entries whose ground-truth label is this one.
    support: usize,
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
}

/// Precision / recall / F1 for a single taxonomy label.
#[derive(Debug, Clone, Serialize)]
struct LabelMetric {
    support: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

/// The full evaluation report. Serializable for `--json`.
#[derive(Debug, Clone, Serialize)]
struct Report {
    total: usize,
    resolved: usize,
    correct: usize,
    /// `correct / total` — credits unresolved entries as misses.
    accuracy: f64,
    /// `correct / resolved` — accuracy among entries that got a prediction.
    resolved_accuracy: f64,
    heuristic_count: usize,
    embedding_count: usize,
    unresolved_count: usize,
    /// `tier1 = true` corpus entries that did NOT resolve at Tier 1 — a
    /// heuristic regression. Should be zero on a healthy build.
    tier1_regressions: usize,
    macro_precision: f64,
    macro_recall: f64,
    macro_f1: f64,
    per_label: BTreeMap<String, LabelMetric>,
    /// `"Expected -> Predicted"` => count, for the mispredictions.
    confusions: BTreeMap<String, usize>,
}

/// Entry point for `cargo xtask eval`.
///
/// `json` switches to machine-readable output. `no_model` forces the
/// embedding tier off even when the bundle is present (useful for fast,
/// deterministic, model-free runs).
///
/// # Errors
/// Propagates corpus-loading, model-loading, or classification failures.
#[allow(unreachable_pub)]
pub fn run(json: bool, no_model: bool) -> Result<()> {
    let dir = corpus_dir();
    let entries = load_manifest(&dir)?;
    let use_model = !no_model && verify_default_model().is_ok();

    let outcomes = classify_corpus(&dir, &entries, use_model)?;
    let report = summarize(&outcomes);

    if json {
        let text = serde_json::to_string_pretty(&report).context("serialize report to JSON")?;
        println!("{text}");
    } else {
        print_report(&report, use_model);
    }
    Ok(())
}

/// The corpus directory (`xtask/corpus/`), resolved relative to this crate so
/// the command works regardless of the caller's working directory.
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus")
}

fn load_manifest(dir: &Path) -> Result<Vec<CorpusEntry>> {
    let path = dir.join("corpus.toml");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read corpus {}", path.display()))?;
    let manifest: Manifest =
        toml::from_str(&text).with_context(|| format!("parse corpus {}", path.display()))?;
    Ok(manifest.entries)
}

/// Run the cascade over every corpus entry. Tier 1 (heuristics) runs for all
/// entries; entries it does not resolve fall to the Tier-2 embedding pass when
/// `use_model` is true.
fn classify_corpus(dir: &Path, entries: &[CorpusEntry], use_model: bool) -> Result<Vec<Outcome>> {
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(entries.len());
    let mut pending: Vec<usize> = Vec::new();

    for (idx, entry) in entries.iter().enumerate() {
        let path = dir.join(&entry.file);
        if let Some(hit) = heuristics::classify(&path, None) {
            outcomes.push(Outcome {
                expected: entry.expected.clone(),
                predicted: Some(hit.taxonomy_path.to_string()),
                tier: Tier::Heuristic,
                expected_tier1: entry.tier1,
            });
        } else {
            outcomes.push(Outcome {
                expected: entry.expected.clone(),
                predicted: None,
                tier: Tier::Unresolved,
                expected_tier1: entry.tier1,
            });
            pending.push(idx);
        }
    }

    if use_model && !pending.is_empty() {
        run_embedding_pass(dir, entries, &pending, &mut outcomes)?;
    }
    Ok(outcomes)
}

/// Classify the heuristic-unresolved entries with the Tier-2 embedding
/// classifier. Only ever called when the model bundle verified present.
fn run_embedding_pass(
    dir: &Path,
    entries: &[CorpusEntry],
    pending: &[usize],
    outcomes: &mut [Outcome],
) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new().context("build tokio runtime for eval")?;
    runtime.block_on(async {
        let embeddings = OrtEmbeddings::load_default().context("load bge-small embedding model")?;
        let classifier = EmbeddingClassifier::with_default_taxonomy(Arc::new(embeddings))
            .await
            .context("build embedding classifier")?;

        for &idx in pending {
            let path = dir.join(&entries[idx].file);
            // Binary fixtures (placeholder media) are never pending — they
            // resolve at Tier 1 — so an unreadable file just stays unresolved.
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let filename = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let result = classifier
                .classify(&content, filename)
                .await
                .with_context(|| format!("classify {}", path.display()))?;
            outcomes[idx] = Outcome {
                expected: entries[idx].expected.clone(),
                predicted: Some(result.folder),
                tier: Tier::Embedding,
                expected_tier1: entries[idx].tier1,
            };
        }
        Ok::<(), anyhow::Error>(())
    })
}

/// A prediction is correct if it equals the expected leaf, or extends it with a
/// trailing year segment (temporal categories append `"<year>/"` at runtime).
fn label_matches(expected: &str, predicted: &str) -> bool {
    if predicted == expected {
        return true;
    }
    predicted.strip_prefix(expected).is_some_and(|rest| {
        let digits = rest.trim_end_matches('/');
        !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
    })
}

/// Canonicalize a predicted folder to its taxonomy leaf by stripping a trailing
/// year segment, so confusion accounting groups `"Finance/Taxes/2023/"` with
/// `"Finance/Taxes/"`.
fn base_label(folder: &str) -> String {
    if let Some((head, tail)) = folder.trim_end_matches('/').rsplit_once('/') {
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            return format!("{head}/");
        }
    }
    folder.to_string()
}

/// Aggregate per-entry outcomes into a full [`Report`]. Pure over its input so
/// it is unit-testable without a model.
#[allow(clippy::cast_precision_loss)]
fn summarize(outcomes: &[Outcome]) -> Report {
    let total = outcomes.len();
    let resolved = outcomes.iter().filter(|o| o.predicted.is_some()).count();
    let correct = outcomes.iter().filter(|o| o.is_correct()).count();
    let heuristic_count = outcomes
        .iter()
        .filter(|o| o.tier == Tier::Heuristic)
        .count();
    let embedding_count = outcomes
        .iter()
        .filter(|o| o.tier == Tier::Embedding)
        .count();
    let unresolved_count = total - heuristic_count - embedding_count;
    let tier1_regressions = outcomes
        .iter()
        .filter(|o| o.expected_tier1 && o.tier != Tier::Heuristic)
        .count();

    let mut labels: BTreeMap<String, LabelCounts> = BTreeMap::new();
    let mut confusions: BTreeMap<String, usize> = BTreeMap::new();

    for outcome in outcomes {
        labels.entry(outcome.expected.clone()).or_default().support += 1;
        match &outcome.predicted {
            Some(_) if outcome.is_correct() => {
                labels
                    .entry(outcome.expected.clone())
                    .or_default()
                    .true_positive += 1;
            }
            Some(predicted) => {
                labels
                    .entry(outcome.expected.clone())
                    .or_default()
                    .false_negative += 1;
                labels
                    .entry(base_label(predicted))
                    .or_default()
                    .false_positive += 1;
                *confusions
                    .entry(format!("{} -> {}", outcome.expected, base_label(predicted)))
                    .or_default() += 1;
            }
            None => {
                labels
                    .entry(outcome.expected.clone())
                    .or_default()
                    .false_negative += 1;
            }
        }
    }

    let mut per_label: BTreeMap<String, LabelMetric> = BTreeMap::new();
    let (mut sum_p, mut sum_r, mut sum_f, mut counted) = (0.0_f64, 0.0_f64, 0.0_f64, 0_usize);
    for (label, counts) in &labels {
        let metric = label_metric(*counts);
        // Macro-average only over labels that are ground truth in the corpus,
        // so a stray false-positive-only label doesn't dilute the average.
        if counts.support > 0 {
            sum_p += metric.precision;
            sum_r += metric.recall;
            sum_f += metric.f1;
            counted += 1;
        }
        per_label.insert(label.clone(), metric);
    }
    let denom = counted.max(1) as f64;

    Report {
        total,
        resolved,
        correct,
        accuracy: ratio(correct, total),
        resolved_accuracy: ratio(correct, resolved),
        heuristic_count,
        embedding_count,
        unresolved_count,
        tier1_regressions,
        macro_precision: sum_p / denom,
        macro_recall: sum_r / denom,
        macro_f1: sum_f / denom,
        per_label,
        confusions,
    }
}

#[allow(clippy::cast_precision_loss)]
fn label_metric(counts: LabelCounts) -> LabelMetric {
    let tp = counts.true_positive as f64;
    let fp = counts.false_positive as f64;
    let fn_ = counts.false_negative as f64;
    let precision = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
    let recall = if tp + fn_ == 0.0 {
        0.0
    } else {
        tp / (tp + fn_)
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    LabelMetric {
        support: counts.support,
        precision,
        recall,
        f1,
    }
}

#[allow(clippy::cast_precision_loss)]
fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn print_report(report: &Report, use_model: bool) {
    println!("tidyup eval — classification accuracy over the golden corpus\n");
    if use_model {
        println!("mode: Tier 1 (heuristics) + Tier 2 (embeddings)");
    } else {
        println!(
            "mode: Tier 1 (heuristics) only — embedding bundle absent; \
             content-dependent entries are deferred.\n      \
             Install it with `cargo xtask download-models` for the full run."
        );
    }
    println!();
    println!("  entries:       {}", report.total);
    println!(
        "  resolved:      {} ({} heuristic, {} embedding, {} unresolved)",
        report.resolved, report.heuristic_count, report.embedding_count, report.unresolved_count,
    );
    println!(
        "  correct:       {} / {}  (accuracy {:.1}%)",
        report.correct,
        report.total,
        report.accuracy * 100.0,
    );
    println!(
        "  resolved acc.: {:.1}%   (of the {} that got a prediction)",
        report.resolved_accuracy * 100.0,
        report.resolved,
    );
    println!(
        "  macro P/R/F1:  {:.3} / {:.3} / {:.3}",
        report.macro_precision, report.macro_recall, report.macro_f1,
    );
    if report.tier1_regressions > 0 {
        println!(
            "  WARNING: {} tier1 entr{} did not resolve at Tier 1 (heuristic regression)",
            report.tier1_regressions,
            if report.tier1_regressions == 1 {
                "y"
            } else {
                "ies"
            },
        );
    }

    if !report.confusions.is_empty() {
        println!("\n  confusions (expected -> predicted):");
        for (pair, count) in &report.confusions {
            println!("    {count:>3}x  {pair}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn outcome(expected: &str, predicted: Option<&str>, tier: Tier) -> Outcome {
        Outcome {
            expected: expected.to_string(),
            predicted: predicted.map(ToString::to_string),
            tier,
            expected_tier1: false,
        }
    }

    #[test]
    fn label_matches_exact_and_temporal() {
        assert!(label_matches("Finance/Taxes/", "Finance/Taxes/"));
        assert!(label_matches("Finance/Taxes/", "Finance/Taxes/2023/"));
        // A subfolder that is not a bare year must NOT count as a match.
        assert!(!label_matches("Code/", "Code/Config/"));
        assert!(!label_matches("Finance/Taxes/", "Finance/Invoices/"));
    }

    #[test]
    fn base_label_strips_trailing_year() {
        assert_eq!(base_label("Finance/Taxes/2023/"), "Finance/Taxes/");
        assert_eq!(base_label("Code/Config/"), "Code/Config/");
        assert_eq!(base_label("Photos/"), "Photos/");
    }

    #[test]
    fn summarize_counts_accuracy_and_coverage() {
        let outcomes = vec![
            outcome("Code/", Some("Code/"), Tier::Heuristic),
            outcome("Music/", Some("Music/"), Tier::Heuristic),
            outcome(
                "Finance/Taxes/",
                Some("Finance/Taxes/2023/"),
                Tier::Embedding,
            ),
            outcome("Recipes/", Some("Work/Career/"), Tier::Embedding), // wrong
            outcome("School/Notes/", None, Tier::Unresolved),           // deferred
        ];
        let report = summarize(&outcomes);
        assert_eq!(report.total, 5);
        assert_eq!(report.resolved, 4);
        assert_eq!(report.correct, 3);
        assert_eq!(report.heuristic_count, 2);
        assert_eq!(report.embedding_count, 2);
        assert_eq!(report.unresolved_count, 1);
        assert!((report.accuracy - 0.6).abs() < 1e-9);
        assert!((report.resolved_accuracy - 0.75).abs() < 1e-9);
    }

    #[test]
    fn summarize_records_confusions() {
        let outcomes = vec![outcome(
            "Recipes/",
            Some("Finance/Taxes/2023/"),
            Tier::Embedding,
        )];
        let report = summarize(&outcomes);
        // The predicted label is canonicalized (year stripped) in the confusion key.
        assert_eq!(
            report.confusions.get("Recipes/ -> Finance/Taxes/").copied(),
            Some(1),
        );
    }

    #[test]
    fn perfect_label_has_unit_metrics() {
        let outcomes = vec![
            outcome("Code/", Some("Code/"), Tier::Heuristic),
            outcome("Code/", Some("Code/"), Tier::Heuristic),
        ];
        let report = summarize(&outcomes);
        let code = report.per_label.get("Code/").unwrap();
        assert_eq!(code.support, 2);
        assert!((code.precision - 1.0).abs() < 1e-9);
        assert!((code.recall - 1.0).abs() < 1e-9);
        assert!((code.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_flags_tier1_regressions() {
        // A tier1-expected entry that fell through to the embedding tier is a
        // heuristic regression; one that landed at Tier 1 is fine.
        let regressed = Outcome {
            expected: "Code/".to_string(),
            predicted: Some("Code/".to_string()),
            tier: Tier::Embedding,
            expected_tier1: true,
        };
        let healthy = Outcome {
            expected: "Music/".to_string(),
            predicted: Some("Music/".to_string()),
            tier: Tier::Heuristic,
            expected_tier1: true,
        };
        let report = summarize(&[regressed, healthy]);
        assert_eq!(report.tier1_regressions, 1);
    }

    // ---- corpus integrity (model-free, runs in CI) -------------------------

    #[test]
    fn corpus_manifest_loads_and_files_exist() {
        let dir = corpus_dir();
        let entries = load_manifest(&dir).unwrap();
        assert!(entries.len() >= 15, "corpus unexpectedly small");
        for entry in &entries {
            assert!(
                entry.expected.ends_with('/'),
                "expected label must end with '/': {}",
                entry.expected,
            );
            let path = dir.join(&entry.file);
            assert!(path.exists(), "missing corpus fixture: {}", path.display());
        }
    }

    #[test]
    fn tier1_entries_match_heuristics() {
        // The deterministic Tier-1 path must route every `tier1 = true` entry to
        // its declared `expected` label. This pins the corpus to the real
        // heuristics so drift in either is caught with no model required.
        let dir = corpus_dir();
        let entries = load_manifest(&dir).unwrap();
        for entry in entries.iter().filter(|e| e.tier1) {
            let path = dir.join(&entry.file);
            let hit = heuristics::classify(&path, None).unwrap_or_else(|| {
                panic!("tier1 entry produced no heuristic match: {}", entry.file)
            });
            assert_eq!(
                hit.taxonomy_path, entry.expected,
                "heuristic mismatch for {}",
                entry.file,
            );
        }
    }
}
