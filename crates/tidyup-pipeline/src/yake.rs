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
//!
//! Stopwords are **language-aware**: [`detect_language`] picks the document's
//! language by stopword overlap (English by default) so renames for non-English
//! content stay clean. This is independent of classification, which still uses
//! the English `bge-small` model — full multilingual *classification* awaits a
//! multilingual embedding model (`bge-m3`).

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

    // Pick the stopword set from the detected language (English by default).
    let stopwords = stopwords_for(detect_language(text));

    // Per-term statistics keyed by lowercased form.
    let mut stats: HashMap<String, TermStats> = HashMap::new();
    for (sent_idx, sent) in sentences.iter().enumerate() {
        let tokens: Vec<&str> = tokenize(sent).collect();
        for (pos, tok) in tokens.iter().enumerate() {
            if !is_candidate(tok, stopwords) {
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
    let mut scored: Vec<Keyword> = ngram_candidates(&sentences, DEFAULT_MAX_NGRAM, stopwords)
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
fn ngram_candidates(
    sentences: &[&str],
    max_n: usize,
    stopwords: &[&str],
) -> HashMap<String, (Vec<String>, u32)> {
    let mut grams: HashMap<String, (Vec<String>, u32)> = HashMap::new();
    for sent in sentences {
        // Split the sentence into runs of consecutive candidate tokens.
        let mut runs: Vec<Vec<String>> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        for tok in tokenize(sent) {
            if is_candidate(tok, stopwords) {
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

fn is_candidate(tok: &str, stopwords: &[&str]) -> bool {
    if tok.len() < 2 {
        return false;
    }
    if tok
        .chars()
        .all(|c| c.is_ascii_digit() || c == '-' || c == '_')
    {
        return false;
    }
    !is_stopword(&tok.to_lowercase(), stopwords)
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

/// Languages with stopword coverage.
///
/// This only governs which stopword set the keyword extractor uses, so renames
/// for non-English content stay clean. Full multilingual *classification* still
/// needs a multilingual embedding model (`bge-m3`) and is tracked separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    Spanish,
    French,
    German,
}

/// Candidate languages, English first (the default).
const LANGUAGES: &[Language] = &[
    Language::English,
    Language::Spanish,
    Language::French,
    Language::German,
];

/// Minimum non-English stopword hits before detection will switch away from the
/// English default — guards against flipping on incidental overlap.
const MIN_FOREIGN_HITS: usize = 3;

/// Detect the document language by stopword overlap.
///
/// English is the default and only loses to a language whose stopwords
/// *strictly* dominate (and clear [`MIN_FOREIGN_HITS`]), so English documents —
/// the common case — always stay on the existing path. Cheap enough to run
/// inline (a few `contains` scans).
#[must_use]
pub fn detect_language(text: &str) -> Language {
    let mut counts = [0usize; LANGUAGES.len()];
    for sentence in split_sentences(text) {
        for tok in tokenize(sentence) {
            let lower = tok.to_lowercase();
            for (i, lang) in LANGUAGES.iter().enumerate() {
                if stopwords_for(*lang).contains(&lower.as_str()) {
                    counts[i] += 1;
                }
            }
        }
    }
    let english = counts[0];
    let (best_idx, best) = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| **c)
        .map_or((0, 0), |(i, c)| (i, *c));
    if best_idx != 0 && best > english && best >= MIN_FOREIGN_HITS {
        LANGUAGES[best_idx]
    } else {
        Language::English
    }
}

const fn stopwords_for(lang: Language) -> &'static [&'static str] {
    match lang {
        Language::English => STOPWORDS_EN,
        Language::Spanish => STOPWORDS_ES,
        Language::French => STOPWORDS_FR,
        Language::German => STOPWORDS_DE,
    }
}

fn is_stopword(tok: &str, stopwords: &[&str]) -> bool {
    // Lists are small (~50-100 entries); linear search beats maintaining a
    // lazily-built `HashSet` across calls.
    stopwords.contains(&tok)
}

// Minimal stopword lists per language. Tuned for filename/prose balance, not
// exhaustive NLP coverage — the YAKE score already penalizes common words via
// `T_FreqT`. These also drive language detection above.
const STOPWORDS_EN: &[&str] = &[
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

const STOPWORDS_ES: &[&str] = &[
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "ante", "con",
    "en", "por", "para", "sin", "sobre", "entre", "hasta", "desde", "y", "e", "o", "u", "que",
    "como", "más", "pero", "porque", "cuando", "donde", "su", "sus", "se", "lo", "le", "les", "mi",
    "tu", "es", "son", "ser", "está", "están", "este", "esta", "esto", "ese", "esa", "eso", "no",
    "sí", "ya", "muy", "también", "hay", "fue", "han", "ha",
];

const STOPWORDS_FR: &[&str] = &[
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "que", "qui", "quoi", "dont",
    "où", "dans", "en", "au", "aux", "pour", "par", "sur", "sous", "sans", "avec", "ce", "cet",
    "cette", "ces", "son", "sa", "ses", "mon", "ma", "mes", "je", "tu", "il", "elle", "nous",
    "vous", "ils", "elles", "est", "sont", "été", "être", "ne", "pas", "plus", "mais", "comme",
    "ainsi", "donc", "aussi",
];

const STOPWORDS_DE: &[&str] = &[
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "und",
    "oder", "aber", "dass", "weil", "wenn", "als", "wie", "von", "zu", "mit", "nach", "bei", "aus",
    "auf", "im", "in", "an", "am", "für", "ist", "sind", "war", "waren", "sein", "hat", "haben",
    "wird", "werden", "nicht", "kein", "keine", "auch", "noch", "nur", "schon", "sehr", "man",
    "sich", "es", "ich", "wir", "ihr",
];

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
        assert!(is_stopword("the", STOPWORDS_EN));
        assert!(is_stopword("and", STOPWORDS_EN));
        assert!(!is_stopword("foo", STOPWORDS_EN));
    }

    #[test]
    fn detect_language_defaults_to_english() {
        assert_eq!(detect_language(""), Language::English);
        assert_eq!(
            detect_language("The quarterly report is on the shared drive."),
            Language::English,
        );
        // Too little foreign signal to switch away from English.
        assert_eq!(detect_language("le rapport"), Language::English);
    }

    #[test]
    fn detect_language_recognizes_romance_and_germanic() {
        assert_eq!(
            detect_language("le rapport trimestriel est sur le disque et dans le dossier"),
            Language::French,
        );
        assert_eq!(
            detect_language("el informe trimestral está en el disco y en la carpeta de la empresa"),
            Language::Spanish,
        );
        assert_eq!(
            detect_language(
                "der vierteljährliche Bericht ist auf der Festplatte und in dem Ordner"
            ),
            Language::German,
        );
    }

    #[test]
    fn french_stopwords_are_filtered_in_french_text() {
        let text = "Le contrat de location de l'appartement. \
                    Le contrat précise le loyer mensuel et la caution.";
        let out = extract_keywords(text, 8);
        let terms: Vec<&str> = out.iter().map(|k| k.term.as_str()).collect();
        // French stopwords must not surface as standalone keywords…
        assert!(!terms.contains(&"le"), "French stopword leaked: {terms:?}");
        assert!(!terms.contains(&"la"), "French stopword leaked: {terms:?}");
        assert!(!terms.contains(&"de"), "French stopword leaked: {terms:?}");
        // …while domain terms still do.
        assert!(
            out.iter().any(|k| k.term.contains("contrat")),
            "expected 'contrat'; got {terms:?}",
        );
    }
}
