//! Extractive rename cascade — produces proposed filenames without fabrication.
//!
//! The cascade runs highest-signal to lowest and returns the first hit:
//!
//! 1. **Embedded metadata.** ID3 `title` / `artist`, EXIF `image_description`
//!    or `make` + `model`, generic `title` keys. Any [`ExtractedContent::metadata`]
//!    that names the content directly.
//! 2. **Keyword-template fill.** Top-ranked YAKE terms from
//!    [`crate::yake::extract_keywords`] assembled into a `year_topic` style
//!    name. Year comes from filename or content when available.
//! 3. **No signal → no rename.** Returns `None`; the caller should keep the
//!    original filename.
//!
//! # Why extractive-only
//!
//! Rename proposals are capped at extracted evidence per the policy in
//! `CLAUDE.md` — no LLM-fabricated names even when `--features llm-fallback`
//! is enabled. This module is structurally incapable of producing a name
//! without either metadata or keywords from the file itself.
//!
//! # Gate
//!
//! Generating a proposal is distinct from *surfacing* one. The rename gate in
//! the pipeline ([`tidyup_domain::RenameConfig`]) requires both
//! classification confidence and filename-content mismatch to clear
//! thresholds; this module only produces candidates. A rejected gate leaves
//! the file with its original name but an approved move.

use std::path::Path;

use serde_json::Value;

use crate::yake::Keyword;

/// Maximum length of a generated filename stem (without extension). Keeps
/// proposals readable and avoids OS path-length issues on Windows.
pub const MAX_STEM_LEN: usize = 80;

/// Outcome of the rename cascade for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenameProposal {
    /// A new filename is proposed (extension preserved).
    Rename { name: String, source: RenameSource },
    /// No signal justified a rename; keep the original name.
    Keep,
}

/// Which tier of the cascade produced the rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenameSource {
    /// Pulled from embedded metadata (ID3, EXIF, PDF title, etc.).
    Metadata,
    /// Synthesized from YAKE top-k keywords plus optional year prefix.
    Keywords,
}

impl RenameSource {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::Keywords => "keywords",
        }
    }
}

/// Run the rename cascade and produce a [`RenameProposal`].
///
/// `metadata` is the `ExtractedContent::metadata` value returned by the
/// extractor. `keywords` is the (possibly empty) YAKE output; empty input
/// triggers fallthrough to `Keep`. `year` seeds the year prefix when the
/// keyword tier fires — `None` drops the prefix.
#[must_use]
pub fn propose_rename(
    original: &Path,
    metadata: &Value,
    keywords: &[Keyword],
    year: Option<i32>,
) -> RenameProposal {
    let ext = original
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    let original_stem = original
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();

    if let Some(stem) = stem_from_metadata(metadata) {
        if !is_trivial_rename(&stem, original_stem) {
            let name = finalize(&stem, ext.as_deref());
            return RenameProposal::Rename {
                name,
                source: RenameSource::Metadata,
            };
        }
    }

    if let Some(stem) = stem_from_keywords(keywords, year) {
        if !is_trivial_rename(&stem, original_stem) {
            let name = finalize(&stem, ext.as_deref());
            return RenameProposal::Rename {
                name,
                source: RenameSource::Keywords,
            };
        }
    }

    RenameProposal::Keep
}

// ---------------------------------------------------------------------------
// Tier 1 — embedded metadata
// ---------------------------------------------------------------------------

/// Look up a rename candidate stem in the extractor metadata.
///
/// Inspects in priority order:
/// - `tags.artist` + `tags.title` (audio)
/// - `tags.title` (audio without artist)
/// - `exif.image_description` (image)
/// - `exif.make` + `exif.model` (image fallback)
/// - `title` at the top level (PDF / generic)
fn stem_from_metadata(metadata: &Value) -> Option<String> {
    let tags = metadata.get("tags").and_then(Value::as_object);
    if let Some(tags) = tags {
        let artist = tags.get("artist").and_then(Value::as_str);
        let title = tags.get("title").and_then(Value::as_str);
        if let (Some(a), Some(t)) = (artist, title) {
            return Some(format!("{a} {t}"));
        }
        if let Some(t) = title {
            return Some(t.to_string());
        }
    }

    let exif = metadata.get("exif").and_then(Value::as_object);
    if let Some(exif) = exif {
        if let Some(desc) = exif.get("image_description").and_then(Value::as_str) {
            if !desc.trim().is_empty() {
                return Some(desc.to_string());
            }
        }
        let make = exif.get("make").and_then(Value::as_str);
        let model = exif.get("model").and_then(Value::as_str);
        if let (Some(make), Some(model)) = (make, model) {
            return Some(format!("{make} {model}"));
        }
    }

    if let Some(title) = metadata.get("title").and_then(Value::as_str) {
        if !title.trim().is_empty() {
            return Some(title.to_string());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tier 2 — keyword-template fill
// ---------------------------------------------------------------------------

/// Compose a stem from the top YAKE keywords, optionally prefixed with a year.
///
/// Takes up to the first four keywords in rank order (YAKE returns best-first).
/// Returns `None` when no usable keyword survives sanitization.
fn stem_from_keywords(keywords: &[Keyword], year: Option<i32>) -> Option<String> {
    let mut picked: Vec<String> = Vec::new();
    for kw in keywords {
        let token = sanitize_token(&kw.term);
        if token.is_empty() {
            continue;
        }
        if picked.iter().any(|p| p == &token) {
            continue;
        }
        picked.push(token);
        if picked.len() >= 4 {
            break;
        }
    }
    if picked.is_empty() {
        return None;
    }
    let body = picked.join("_");
    Some(year.map_or_else(|| body.clone(), |y| format!("{y}_{body}")))
}

// ---------------------------------------------------------------------------
// Sanitization helpers
// ---------------------------------------------------------------------------

/// Sanitize a free-form string into a filesystem-safe stem.
///
/// - Lowercases.
/// - Replaces any non-alphanumeric run with a single underscore.
/// - Collapses repeated underscores, trims leading/trailing ones.
/// - Truncates to [`MAX_STEM_LEN`] at a word boundary where possible.
#[must_use]
pub fn sanitize_filename(raw: &str) -> String {
    let lower = raw.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut last_was_sep = true;
    for ch in lower.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }

    if out.len() > MAX_STEM_LEN {
        truncate_at_boundary(&out, MAX_STEM_LEN)
    } else {
        out
    }
}

fn sanitize_token(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        }
    }
    out
}

fn truncate_at_boundary(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let window = &s[..max];
    if let Some(pos) = window.rfind('_') {
        // Only back off to the underscore if it leaves at least half the budget.
        if pos >= max / 2 {
            return s[..pos].to_string();
        }
    }
    s[..max].to_string()
}

/// Stitch sanitized stem + original extension back into a filename.
fn finalize(stem: &str, ext: Option<&str>) -> String {
    let sanitized = sanitize_filename(stem);
    match ext {
        Some(e) if !e.is_empty() => format!("{sanitized}.{e}"),
        _ => sanitized,
    }
}

/// A rename that only re-orders whitespace or punctuation isn't interesting —
/// suppress it so the proposal stream stays focused on meaningful changes.
fn is_trivial_rename(candidate_raw: &str, original_stem: &str) -> bool {
    let c = sanitize_filename(candidate_raw);
    let o = sanitize_filename(original_stem);
    c.is_empty() || c == o
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    use crate::yake::Keyword;

    fn kw(term: &str, score: f32) -> Keyword {
        Keyword {
            term: term.to_string(),
            score,
        }
    }

    #[test]
    fn sanitize_lowercases_and_underscores() {
        assert_eq!(sanitize_filename("Hello World!"), "hello_world");
        assert_eq!(sanitize_filename("  a--b__c  "), "a_b_c");
    }

    #[test]
    fn sanitize_strips_non_ascii() {
        assert_eq!(sanitize_filename("café résumé"), "caf_r_sum");
    }

    #[test]
    fn sanitize_truncates_at_boundary() {
        let long =
            "alpha_beta_gamma_delta_epsilon_zeta_eta_theta_iota_kappa_lambda_mu_nu_xi_omicron";
        let out = sanitize_filename(long);
        assert!(out.len() <= MAX_STEM_LEN);
        assert!(!out.ends_with('_'));
    }

    #[test]
    fn metadata_tier_audio_artist_plus_title() {
        let meta = json!({"tags": {"artist": "Radiohead", "title": "Idioteque"}});
        let p = propose_rename(&PathBuf::from("/m/01 - track.mp3"), &meta, &[], None);
        match p {
            RenameProposal::Rename { name, source } => {
                assert_eq!(name, "radiohead_idioteque.mp3");
                assert_eq!(source, RenameSource::Metadata);
            }
            RenameProposal::Keep => panic!("expected metadata rename"),
        }
    }

    #[test]
    fn metadata_tier_audio_title_only() {
        let meta = json!({"tags": {"title": "Nocturne"}});
        let p = propose_rename(&PathBuf::from("/m/x.flac"), &meta, &[], None);
        match p {
            RenameProposal::Rename { name, .. } => assert_eq!(name, "nocturne.flac"),
            RenameProposal::Keep => panic!("expected rename"),
        }
    }

    #[test]
    fn metadata_tier_exif_description() {
        let meta = json!({"exif": {"image_description": "Yosemite sunset from Glacier Point"}});
        let p = propose_rename(&PathBuf::from("/p/IMG_1234.jpg"), &meta, &[], None);
        match p {
            RenameProposal::Rename { name, source } => {
                assert_eq!(source, RenameSource::Metadata);
                assert!(name.starts_with("yosemite_sunset"));
                assert!(Path::new(&name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg")));
            }
            RenameProposal::Keep => panic!("expected rename"),
        }
    }

    #[test]
    fn metadata_tier_exif_make_model_fallback() {
        let meta = json!({"exif": {"make": "Canon", "model": "EOS R5"}});
        let p = propose_rename(&PathBuf::from("/p/IMG_1234.jpg"), &meta, &[], None);
        match p {
            RenameProposal::Rename { name, .. } => assert_eq!(name, "canon_eos_r5.jpg"),
            RenameProposal::Keep => panic!("expected rename"),
        }
    }

    #[test]
    fn metadata_tier_generic_title() {
        let meta = json!({"title": "Quarterly Report Q3 2024"});
        let p = propose_rename(&PathBuf::from("/d/scan.pdf"), &meta, &[], None);
        match p {
            RenameProposal::Rename { name, source } => {
                assert_eq!(source, RenameSource::Metadata);
                assert!(name.starts_with("quarterly_report"));
                assert!(Path::new(&name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf")));
            }
            RenameProposal::Keep => panic!("expected rename"),
        }
    }

    #[test]
    fn keyword_tier_composes_stem_with_year() {
        let kws = vec![kw("invoice", 0.1), kw("acme", 0.2), kw("march", 0.3)];
        let p = propose_rename(&PathBuf::from("/d/scan.pdf"), &json!({}), &kws, Some(2024));
        match p {
            RenameProposal::Rename { name, source } => {
                assert_eq!(source, RenameSource::Keywords);
                assert_eq!(name, "2024_invoice_acme_march.pdf");
            }
            RenameProposal::Keep => panic!("expected keyword rename"),
        }
    }

    #[test]
    fn keyword_tier_without_year() {
        let kws = vec![kw("mortgage", 0.1), kw("statement", 0.2)];
        let p = propose_rename(&PathBuf::from("/d/doc.pdf"), &json!({}), &kws, None);
        match p {
            RenameProposal::Rename { name, source } => {
                assert_eq!(source, RenameSource::Keywords);
                assert_eq!(name, "mortgage_statement.pdf");
            }
            RenameProposal::Keep => panic!("expected rename"),
        }
    }

    #[test]
    fn metadata_beats_keywords() {
        let meta = json!({"title": "Lease Agreement"});
        let kws = vec![kw("totallydifferent", 0.1)];
        let p = propose_rename(&PathBuf::from("/d/x.pdf"), &meta, &kws, Some(2024));
        match p {
            RenameProposal::Rename { source, name } => {
                assert_eq!(source, RenameSource::Metadata);
                assert!(name.contains("lease"));
            }
            RenameProposal::Keep => panic!(),
        }
    }

    #[test]
    fn no_signal_returns_keep() {
        let p = propose_rename(&PathBuf::from("/d/x.pdf"), &json!({}), &[], None);
        assert_eq!(p, RenameProposal::Keep);
    }

    #[test]
    fn trivial_rename_is_suppressed() {
        // Metadata title matches sanitized original stem.
        let meta = json!({"title": "Lease Agreement"});
        let p = propose_rename(&PathBuf::from("/d/Lease_Agreement.pdf"), &meta, &[], None);
        // Should return Keep, but with no keyword fallback, Keep is produced.
        assert_eq!(p, RenameProposal::Keep);
    }

    #[test]
    fn preserves_extension_case_folded() {
        let meta = json!({"title": "Trip Photos"});
        let p = propose_rename(&PathBuf::from("/p/IMG.JPG"), &meta, &[], None);
        match p {
            RenameProposal::Rename { name, .. } => assert!(Path::new(&name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))),
            RenameProposal::Keep => panic!(),
        }
    }

    #[test]
    fn empty_keywords_triggers_keep() {
        let p = propose_rename(&PathBuf::from("/d/x.pdf"), &json!({}), &[], Some(2024));
        assert_eq!(p, RenameProposal::Keep);
    }

    #[test]
    fn rename_source_label() {
        assert_eq!(RenameSource::Metadata.label(), "metadata");
        assert_eq!(RenameSource::Keywords.label(), "keywords");
    }

    #[test]
    fn keyword_tier_deduplicates_tokens() {
        // Same stem appearing twice — second should be skipped.
        let kws = vec![
            kw("invoice", 0.1),
            kw("Invoice", 0.2), // sanitizes to same
            kw("march", 0.3),
        ];
        let p = propose_rename(&PathBuf::from("/d/x.pdf"), &json!({}), &kws, None);
        match p {
            RenameProposal::Rename { name, .. } => {
                assert_eq!(name, "invoice_march.pdf");
            }
            RenameProposal::Keep => panic!(),
        }
    }
}
