//! `cargo xtask eval` — classification-accuracy harness over a labeled corpus.
//!
//! Runs semantic classification against the golden corpus under
//! `xtask/corpus/` and reports overall accuracy, per-label precision / recall /
//! F1, semantic coverage, and the top confusions.
//!
//! Classification is semantic and therefore gated on the local embedding model
//! being installed. Without it, entries are reported as deferred.
//!
//! This is a developer + calibration tool (it feeds the Stage-5 confidence
//! calibration work). It is intentionally **not** wired into `cargo xtask ci`,
//! which must stay model-free.

// This developer-only evaluator computes aggregate metrics and indexes vectors
// whose dimensions are established by the loaded corpus.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use tidyup_domain::Calibration;
use tidyup_embeddings_ort::{verify_default_model, EmbeddingClassifier, OrtEmbeddings};
use tidyup_pipeline::calibration::{expected_calibration_error, fit_platt};

/// The resolver that produced a prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Tier {
    Embedding,
    Unresolved,
}

/// One row of the corpus manifest (`corpus.toml`).
#[derive(Debug, Clone, Deserialize)]
struct CorpusEntry {
    /// Path to the fixture, relative to the corpus directory.
    file: String,
    /// Taxonomy leaf the semantic resolver is expected to route the file to.
    expected: String,
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
    /// `None` when no evaluated resolver produced a prediction (unresolved).
    predicted: Option<String>,
    tier: Tier,
    /// Raw confidence the resolver assigned (`None` when unresolved).
    /// Feeds `--calibrate`.
    confidence: Option<f32>,
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

/// Fitted confidence calibration over the corpus (`--calibrate`).
#[derive(Debug, Clone, Serialize)]
struct CalibrationReport {
    /// Resolved samples the calibrator was fit over.
    samples: usize,
    /// `"identity"` or `"platt"`.
    fit: &'static str,
    a: Option<f64>,
    b: Option<f64>,
    /// Expected Calibration Error of the raw scores (identity).
    ece_raw: f32,
    /// ECE after applying the fitted calibrator.
    ece_calibrated: f32,
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
    embedding_count: usize,
    unresolved_count: usize,
    macro_precision: f64,
    macro_recall: f64,
    macro_f1: f64,
    per_label: BTreeMap<String, LabelMetric>,
    /// `"Expected -> Predicted"` => count, for the mispredictions.
    confusions: BTreeMap<String, usize>,
    /// Fitted calibration, present only with `--calibrate`.
    #[serde(skip_serializing_if = "Option::is_none")]
    calibration: Option<CalibrationReport>,
}

/// Entry point for `cargo xtask eval`.
///
/// `json` switches to machine-readable output. `no_model` forces the
/// embedding resolver off even when the bundle is present (useful for fast,
/// deterministic, model-free runs).
///
/// # Errors
/// Propagates corpus-loading, model-loading, or classification failures.
#[allow(unreachable_pub)]
pub fn run(json: bool, no_model: bool, calibrate: bool) -> Result<()> {
    let started = std::time::Instant::now();
    let dir = corpus_dir();
    let entries = load_manifest(&dir)?;
    let use_model = !no_model && verify_default_model().is_ok();
    // Per-phase timing to stderr (keeps --json stdout clean) so a slow or hung
    // run is diagnosable instead of just timing out opaquely.
    eprintln!(
        "[eval] {} corpus entries; model={} ({:?} elapsed)",
        entries.len(),
        if use_model { "on" } else { "off" },
        started.elapsed(),
    );

    let classify_start = std::time::Instant::now();
    let outcomes = classify_corpus(&dir, &entries, use_model)?;
    eprintln!(
        "[eval] classification done in {:?} (total {:?})",
        classify_start.elapsed(),
        started.elapsed(),
    );
    let mut report = summarize(&outcomes);
    if calibrate {
        report.calibration = Some(compute_calibration(&outcomes));
    }

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

/// Run semantic classification over every corpus entry when the local model is
/// available.
fn classify_corpus(dir: &Path, entries: &[CorpusEntry], use_model: bool) -> Result<Vec<Outcome>> {
    let mut outcomes: Vec<Outcome> = entries
        .iter()
        .map(|entry| Outcome {
            expected: entry.expected.clone(),
            predicted: None,
            tier: Tier::Unresolved,
            confidence: None,
        })
        .collect();
    let pending: Vec<usize> = (0..entries.len()).collect();

    if use_model && !pending.is_empty() {
        run_embedding_pass(dir, entries, &pending, &mut outcomes)?;
    }
    Ok(outcomes)
}

/// Classify entries with the embedding classifier.
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
            // Placeholder binary fixtures remain unresolved when this text-only
            // evaluator cannot extract semantic content.
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
                confidence: Some(result.confidence),
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
    let embedding_count = outcomes
        .iter()
        .filter(|o| o.tier == Tier::Embedding)
        .count();
    let unresolved_count = total - embedding_count;

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
        embedding_count,
        unresolved_count,
        macro_precision: sum_p / denom,
        macro_recall: sum_r / denom,
        macro_f1: sum_f / denom,
        per_label,
        confusions,
        calibration: None,
    }
}

/// Fit a Platt calibrator over the resolved `(confidence, correct)` samples and
/// measure ECE before/after. Filled only under `--calibrate`.
fn compute_calibration(outcomes: &[Outcome]) -> CalibrationReport {
    let samples: Vec<(f32, bool)> = outcomes
        .iter()
        .filter_map(|o| o.confidence.map(|c| (c, o.is_correct())))
        .collect();
    let ece_raw = expected_calibration_error(&samples, &Calibration::Identity, 10);
    let fitted = fit_platt(&samples);
    let ece_calibrated = expected_calibration_error(&samples, &fitted, 10);
    let (fit, a, b) = match fitted {
        Calibration::Identity => ("identity", None, None),
        Calibration::Platt { a, b } => ("platt", Some(a), Some(b)),
    };
    CalibrationReport {
        samples: samples.len(),
        fit,
        a,
        b,
        ece_raw,
        ece_calibrated,
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
        println!("mode: semantic embeddings");
    } else {
        println!(
            "mode: embedding bundle absent; entries are deferred.\n      \
             Install it with `cargo xtask download-models` for the full run."
        );
    }
    println!();
    println!("  entries:       {}", report.total);
    println!(
        "  resolved:      {} ({} semantic embedding, {} unresolved)",
        report.resolved, report.embedding_count, report.unresolved_count,
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
    if !report.confusions.is_empty() {
        println!("\n  confusions (expected -> predicted):");
        for (pair, count) in &report.confusions {
            println!("    {count:>3}x  {pair}");
        }
    }

    if let Some(cal) = &report.calibration {
        println!(
            "\n  calibration (Platt scaling over {} samples):",
            cal.samples
        );
        match (cal.a, cal.b) {
            (Some(a), Some(b)) => println!("    fit: sigmoid({a:.4} * score + {b:.4})"),
            _ => println!("    fit: identity (degenerate — single class or too few samples)"),
        }
        println!(
            "    ECE: {:.4} (raw) -> {:.4} (calibrated)",
            cal.ece_raw, cal.ece_calibrated,
        );
        if !use_model {
            println!("    note: model absent — no semantic samples are available for calibration");
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
            confidence: None,
        }
    }

    fn outcome_conf(expected: &str, predicted: Option<&str>, confidence: f32) -> Outcome {
        Outcome {
            expected: expected.to_string(),
            predicted: predicted.map(ToString::to_string),
            tier: Tier::Embedding,
            confidence: Some(confidence),
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
            outcome("Code/", Some("Code/"), Tier::Embedding),
            outcome("Music/", Some("Music/"), Tier::Embedding),
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
        assert_eq!(report.embedding_count, 4);
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
            outcome("Code/", Some("Code/"), Tier::Embedding),
            outcome("Code/", Some("Code/"), Tier::Embedding),
        ];
        let report = summarize(&outcomes);
        let code = report.per_label.get("Code/").unwrap();
        assert_eq!(code.support, 2);
        assert!((code.precision - 1.0).abs() < 1e-9);
        assert!((code.recall - 1.0).abs() < 1e-9);
        assert!((code.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compute_calibration_reports_ece_and_fit() {
        // Mixed outcomes -> a fittable two-class set.
        let outcomes = vec![
            outcome_conf("Code/", Some("Code/"), 0.9),
            outcome_conf("Music/", Some("Music/"), 0.85),
            outcome_conf("Recipes/", Some("Finance/Taxes/"), 0.4), // wrong
            outcome_conf("School/Notes/", Some("Work/Career/"), 0.3), // wrong
        ];
        let cal = compute_calibration(&outcomes);
        assert_eq!(cal.samples, 4);
        assert!((0.0..=1.0).contains(&cal.ece_raw));
        assert!((0.0..=1.0).contains(&cal.ece_calibrated));
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
}
