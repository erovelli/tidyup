//! Inlined YAKE — unsupervised single-document keyword extraction.
//!
//! YAKE (Campos et al., 2020) ranks terms by a combination of five statistical
//! features computed from a single document:
//!
//! - **`T_Case`**: proper-noun / acronym signal — how often the term appears
//!   capitalized vs. lowercased.
//! - **`T_Position`**: terms earlier in the document score better.
//! - **`T_FreqT`**: frequency normalized by the mean + std of all term
//!   frequencies — damps both stopwords and hapaxes.
//! - **`T_Rel`**: contextual distinctiveness — terms flanked by many different
//!   unique words score better.
//! - **`T_DifSent`**: fraction of unique sentences containing the term.
//!
//! Per-term score = `T_Rel * T_Position / (T_Case + T_FreqT/T_Rel + T_DifSent/T_Rel)`.
//! **Lower is better.**
//!
//! # Rationale for inlining
//!
//! Mainstream pure-Rust YAKE crates (`keyword-extraction-rs`, ~20k DL/mo)
//! sit below the 100k-DL/mo dependency threshold set in `CLAUDE.md`. ~150
//! `LoC` of well-scoped statistics is a better trade than taking a
//! below-threshold dep — see `CLASSIFICATION.md` for the decision record.
//!
//! # Scope
//!
//! Candidates are n-grams up to [`DEFAULT_MAX_NGRAM`] words, built from runs of
//! consecutive content tokens (a phrase never spans a stopword, numeric, or
//! punctuation boundary). Each candidate is scored by the YAKE keyphrase rule
//! `∏ S(t) / (TF(kw) · (1 + ∑ S(t)))` over its constituent per-term scores —
//! for a unigram this reduces to `S(t) / (TF · (1 + S(t)))`. **Lower is better.**
//! English-only stopwords — multilingual support lands with `bge-m3`.

use std::collections::{HashMap, HashSet};

/// Longest keyphrase (in words) the extractor will consider.
pub const DEFAULT_MAX_NGRAM: usize = 3;

/// A keyword candidate and its YAKE score. Lower score = stronger keyword.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyword {
    pub term: String,
    pub score: f32,
}

/// Extract the top `k` keywords from `text`.
///
/// Returns candidates sorted by ascending YAKE score (best keywords first).
/// Skips stopwords, numerics, and tokens shorter than 2 chars. Empty or
/// all-stopword input returns an empty vector.
#[must_use]
pub fn extract_keywords(text: &str, k: usize) -> Vec<Keyword> {
    if text.trim().is_empty() || k == 0 {
        return Vec::new();
    }
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    // Per-term statistics keyed by lowercased form.
    let mut stats: HashMap<String, TermStats> = HashMap::new();
    for (sent_idx, sent) in sentences.iter().enumerate() {
        let tokens: Vec<&str> = tokenize(sent).collect();
        for (pos, tok) in tokens.iter().enumerate() {
            if !is_candidate(tok) {
                continue;
            }
            let key = tok.to_lowercase();
            let entry = stats.entry(key).or_default();
            entry.count += 1;
            if is_upper_initial(tok) {
                entry.upper_count += 1;
            }
            if is_all_upper(tok) {
                entry.acronym_count += 1;
            }
            entry.sentences.insert(sent_idx);
            if pos > 0 {
                entry.left_context.insert(tokens[pos - 1].to_lowercase());
            }
            if pos + 1 < tokens.len() {
                entry.right_context.insert(tokens[pos + 1].to_lowercase());
            }
        }
    }
    if stats.is_empty() {
        return Vec::new();
    }

    // Mean and std of frequency, for T_FreqT normalization.
    #[allow(clippy::cast_precision_loss)]
    let counts: Vec<f32> = stats.values().map(|s| s.count as f32).collect();
    #[allow(clippy::cast_precision_loss)]
    let n = counts.len() as f32;
    let mean = counts.iter().copied().sum::<f32>() / n;
    let variance = counts.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt().max(1.0);

    #[allow(clippy::cast_precision_loss)]
    let total_sentences = sentences.len() as f32;

    // Per-term YAKE score S(t); lower is stronger.
    let term_scores: HashMap<String, f32> = stats
        .into_iter()
        .map(|(term, s)| {
            #[allow(clippy::cast_precision_loss)]
            let freq = s.count as f32;
            let t_case = f32::from(s.upper_count.max(s.acronym_count)) / (1.0 + freq.ln());
            #[allow(clippy::cast_precision_loss)]
            let first_sent = s.sentences.iter().min().copied().unwrap_or(0) as f32;
            let t_position = (3.0 + first_sent).ln().ln();
            let t_freq = freq / (mean + std_dev);
            #[allow(clippy::cast_precision_loss)]
            let t_rel =
                1.0 + (s.left_context.len() as f32 + s.right_context.len() as f32) / (2.0 * freq);
            #[allow(clippy::cast_precision_loss)]
            let t_difsent = s.sentences.len() as f32 / total_sentences;

            let denom = t_case + (t_freq / t_rel) + (t_difsent / t_rel);
            let score = (t_rel * t_position) / denom.max(f32::EPSILON);
            (term, score)
        })
        .collect();

    // Assemble n-gram candidates and score each by the YAKE keyphrase rule.
    let mut scored: Vec<Keyword> = ngram_candidates(&sentences, DEFAULT_MAX_NGRAM)
        .into_iter()
        .map(|(phrase, (terms, tf))| {
            let prod: f32 = terms
                .iter()
                .map(|t| term_scores.get(t).copied().unwrap_or(1.0))
                .product();
            let sum: f32 = terms
                .iter()
                .map(|t| term_scores.get(t).copied().unwrap_or(1.0))
                .sum();
            #[allow(clippy::cast_precision_loss)]
            let tf_f = tf as f32;
            let score = prod / (tf_f * (1.0 + sum)).max(f32::EPSILON);
            Keyword {
                term: phrase,
                score,
            }
        })
        .collect();

    // Ascending by score; tie-break on the phrase text for deterministic output
    // (HashMap iteration order is otherwise unspecified).
    scored.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.term.cmp(&b.term))
    });
    scored.truncate(k);
    scored
}

/// Build n-gram candidates (length `1..=max_n`) from runs of consecutive
/// candidate tokens within each sentence — a phrase never spans a stopword,
/// numeric, or punctuation boundary. Returns a map from the lowercased phrase to
/// its constituent terms and its document frequency.
fn ngram_candidates(sentences: &[&str], max_n: usize) -> HashMap<String, (Vec<String>, u32)> {
    let mut grams: HashMap<String, (Vec<String>, u32)> = HashMap::new();
    for sent in sentences {
        // Split the sentence into runs of consecutive candidate tokens.
        let mut runs: Vec<Vec<String>> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        for tok in tokenize(sent) {
            if is_candidate(tok) {
                current.push(tok.to_lowercase());
            } else if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            runs.push(current);
        }
        for run in &runs {
            for n in 1..=max_n.min(run.len()) {
                for window in run.windows(n) {
                    let phrase = window.join(" ");
                    let entry = grams.entry(phrase).or_insert_with(|| (window.to_vec(), 0));
                    entry.1 += 1;
                }
            }
        }
    }
    grams
}

#[derive(Default)]
struct TermStats {
    count: u32,
    upper_count: u8,
    acronym_count: u8,
    sentences: HashSet<usize>,
    left_context: HashSet<String>,
    right_context: HashSet<String>,
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(['.', '!', '?', '\n', '\r'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn tokenize(sentence: &str) -> impl Iterator<Item = &str> {
    sentence
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|s| !s.is_empty())
}

fn is_candidate(tok: &str) -> bool {
    if tok.len() < 2 {
        return false;
    }
    if tok
        .chars()
        .all(|c| c.is_ascii_digit() || c == '-' || c == '_')
    {
        return false;
    }
    !is_stopword(&tok.to_lowercase())
}

fn is_upper_initial(tok: &str) -> bool {
    tok.chars().next().is_some_and(char::is_uppercase)
        && tok.chars().skip(1).any(char::is_lowercase)
}

fn is_all_upper(tok: &str) -> bool {
    tok.len() >= 2
        && tok
            .chars()
            .all(|c| c.is_uppercase() || c == '-' || c == '_')
}

// Minimal English stopword list. Tuned for filename/prose balance, not
// exhaustive NLP coverage — the YAKE score already penalizes most common
// words via `T_FreqT`.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "he",
    "her", "his", "i", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "them", "there", "they", "this", "to", "was", "we", "were", "will", "with", "you", "your",
    "can", "could", "do", "does", "done", "had", "if", "into", "no", "not", "one", "other", "our",
    "out", "over", "should", "so", "some", "such", "than", "then", "these", "those", "too", "up",
    "what", "when", "which", "who", "why", "would", "about", "above", "after", "again", "all",
    "also", "any", "because", "been", "before", "being", "below", "between", "both", "down",
    "during", "each", "few", "further", "here", "how", "just", "me", "more", "most", "my", "now",
    "off", "once", "only", "own", "same", "very", "where", "while", "am",
];

fn is_stopword(tok: &str) -> bool {
    // List is small enough (~100 entries) that linear search beats the
    // overhead of maintaining a lazily-built `HashSet` across calls.
    STOPWORDS.contains(&tok)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty() {
        assert!(extract_keywords("", 5).is_empty());
        assert!(extract_keywords("   ", 5).is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        assert!(extract_keywords("some text here", 0).is_empty());
    }

    #[test]
    fn respects_k_limit() {
        let text = "Apple banana cherry date elderberry fig grape honeydew imbe jackfruit.";
        let out = extract_keywords(text, 3);
        assert!(out.len() <= 3);
    }

    #[test]
    fn stopwords_are_excluded() {
        let out = extract_keywords("The quick brown fox jumps over the lazy dog.", 10);
        let terms: Vec<&str> = out.iter().map(|k| k.term.as_str()).collect();
        assert!(!terms.contains(&"the"));
        assert!(!terms.contains(&"over"));
    }

    #[test]
    fn numeric_only_tokens_excluded() {
        let out = extract_keywords("Invoice 2024 total 4500 dollars due.", 10);
        let terms: Vec<&str> = out.iter().map(|k| k.term.as_str()).collect();
        assert!(!terms.contains(&"2024"));
        assert!(!terms.contains(&"4500"));
    }

    #[test]
    fn surfaces_repeated_domain_term() {
        let text = "Quarterly tax return filing. The tax form is 1040. \
                    Include tax receipts for all deductions. Tax advisor signed.";
        let out = extract_keywords(text, 5);
        // With n-grams the domain term may surface as "tax" or as a phrase like
        // "tax return"/"tax form" — either is the right signal for a rename.
        assert!(
            out.iter().any(|k| k.term.contains("tax")),
            "expected a tax-related term in top 5; got {out:?}",
        );
    }

    #[test]
    fn extracts_multiword_phrases() {
        let text = "Quarterly tax return filing. The tax return is mailed. \
                    Prepare the tax return early.";
        let out = extract_keywords(text, 6);
        assert!(
            out.iter().any(|k| k.term.contains(' ')),
            "expected at least one multi-word phrase; got {out:?}",
        );
        // "tax return" recurs three times and should surface as a phrase.
        assert!(
            out.iter().any(|k| k.term == "tax return"),
            "expected 'tax return' phrase; got {out:?}",
        );
    }

    #[test]
    fn ngrams_do_not_span_stopwords() {
        // "cat" and "dog" are split by the stopword "and" → no "cat dog" phrase.
        let out = extract_keywords("The cat and the dog ran.", 10);
        assert!(
            out.iter().all(|k| k.term != "cat dog"),
            "phrases must not span a stopword; got {out:?}",
        );
    }

    #[test]
    fn respects_max_ngram_length() {
        let text = "alpha beta gamma delta epsilon zeta eta theta.";
        let out = extract_keywords(text, 50);
        for k in &out {
            let words = k.term.split(' ').count();
            assert!(words <= DEFAULT_MAX_NGRAM, "phrase too long: {}", k.term);
        }
    }

    #[test]
    fn sorted_ascending_by_score() {
        let text = "Rust systems programming language concurrent memory safe.";
        let out = extract_keywords(text, 5);
        for w in out.windows(2) {
            assert!(w[0].score <= w[1].score);
        }
    }

    #[test]
    fn is_stopword_lookup() {
        assert!(is_stopword("the"));
        assert!(is_stopword("and"));
        assert!(!is_stopword("foo"));
    }
}
