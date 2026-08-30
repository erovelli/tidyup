//! Grounded rename cascade — produces proposed filenames from file evidence.
//!
//! The cascade runs highest-signal to lowest and returns the first hit:
//!
//! 1. **Embedded metadata.** ID3 `title` / `artist`, EXIF `image_description`
//!    or `make` + `model`, generic `title` keys. Any [`ExtractedContent::metadata`]
//!    that names the content directly.
//! 2. **Keyword composition.** Top-ranked YAKE terms from
//!    [`crate::yake::extract_keywords`] assembled into a `year_topic` style
//!    name. Year comes from filename or content when available.
//! 3. **Semantic concepts.** A caller with a contrastive multimodal embedding
//!    may provide scored visible/audible concepts through
//!    [`propose_grounded_rename`]. This path is retrieval, not generation.
//! 4. **No signal → no rename.** Returns `None`; the caller should keep the
//!    original filename.
//!
//! # Why grounded-only
//!
//! Rename proposals are capped at inspectable evidence: exact metadata,
//! extracted keywords, or concepts retrieved from a bounded contrastive-model
//! bank. No LLM-fabricated name is accepted even when an optional fallback is
//! enabled.
//!
//! # Gate
//!
//! Generating a proposal is distinct from *surfacing* one. The rename gate in
//! the pipeline ([`tidyup_domain::RenameConfig`]) requires both
//! classification confidence and filename-content mismatch to clear
//! thresholds; this module only produces candidates. A rejected gate leaves
//! the file with its original name but an approved move.

use std::collections::HashSet;
use std::path::Path;

use serde_json::Value;
use tidyup_domain::{ChangeProposal, ChangeType};

use crate::semantic::ConceptMatch;
use crate::yake::Keyword;

/// Maximum number of distinct words composing a synthesized keyword stem. Keeps
/// names readable now that keywords can be multi-word phrases.
const MAX_STEM_WORDS: usize = 4;

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

/// Which grounded evidence source produced the rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenameSource {
    /// Pulled from embedded metadata (for example ID3 or EXIF fields).
    Metadata,
    /// Derived from text visibly present in an image via local OCR.
    Ocr,
    /// Synthesized from YAKE top-k keywords plus optional year prefix.
    Keywords,
    /// Retrieved from non-textual content through a contrastive model.
    SemanticConcept,
}

impl RenameSource {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Metadata => "metadata",
            Self::Ocr => "local OCR",
            Self::Keywords => "keywords",
            Self::SemanticConcept => "grounded semantic concepts",
        }
    }
}

/// Run the rename cascade and produce a [`RenameProposal`].
///
/// `metadata` is the `ExtractedContent::metadata` value returned by the
/// extractor. `keywords` is the (possibly empty) YAKE output; empty input
/// triggers fallthrough to `Keep`. `year` seeds the year prefix when the
/// keyword-based rename path fires — `None` drops the prefix.
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
                source: if metadata.get("ocr_text").and_then(Value::as_str).is_some() {
                    RenameSource::Ocr
                } else {
                    RenameSource::Keywords
                },
            };
        }
    }

    RenameProposal::Keep
}

/// Compose a deterministic filename from strongly grounded semantic concepts.
///
/// The caller owns confidence and mismatch gating. This renderer only accepts
/// already-ranked concept labels and preserves the original extension.
#[must_use]
pub fn propose_grounded_rename(original: &Path, concepts: &[ConceptMatch]) -> RenameProposal {
    let stem = concepts
        .iter()
        .take(2)
        .map(|concept| sanitize_token(&concept.label))
        .filter(|label| !label.is_empty())
        .collect::<Vec<_>>()
        .join("_on_");
    let original_stem = original
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if stem.is_empty() || is_trivial_rename(&stem, original_stem) {
        return RenameProposal::Keep;
    }
    let extension = original
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    RenameProposal::Rename {
        name: finalize(&stem, extension.as_deref()),
        source: RenameSource::SemanticConcept,
    }
}

/// Make proposed member filenames unique within one atomic collection.
///
/// Existing move-only names are reserved first so a generated rename cannot
/// claim a sibling's unchanged basename. Colliding rename proposals receive a
/// deterministic numeric suffix while retaining their extracted stem and
/// extension. A move-only collision is rejected instead of fabricating an
/// ungrounded rename. Comparisons are case-insensitive to stay safe on the
/// default macOS and Windows filesystems.
pub(crate) fn uniquify_bundle_member_names(proposals: &mut [ChangeProposal]) -> anyhow::Result<()> {
    let mut claimed = HashSet::new();
    for proposal in proposals
        .iter()
        .filter(|proposal| proposal.change_type == ChangeType::Move)
    {
        if !claimed.insert(proposal.proposed_name.to_lowercase()) {
            return Err(anyhow::anyhow!(
                "move-only bundle members collide at basename {}",
                proposal.proposed_name
            ));
        }
    }

    for proposal in proposals.iter_mut().filter(|proposal| {
        matches!(
            proposal.change_type,
            ChangeType::Rename | ChangeType::RenameAndMove
        )
    }) {
        let original = proposal.proposed_name.clone();
        if claimed.insert(original.to_lowercase()) {
            continue;
        }

        for sequence in 2_u32.. {
            let candidate = filename_with_sequence(&original, sequence);
            if claimed.insert(candidate.to_lowercase()) {
                proposal.proposed_name.clone_from(&candidate);
                proposal.proposed_path.set_file_name(candidate);
                break;
            }
        }
    }
    Ok(())
}

fn filename_with_sequence(filename: &str, sequence: u32) -> String {
    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(filename);
    path.extension()
        .and_then(|value| value.to_str())
        .map_or_else(
            || format!("{stem}_{sequence}"),
            |extension| format!("{stem}_{sequence}.{extension}"),
        )
}

// ---------------------------------------------------------------------------
// Rename evidence 1 — embedded metadata
// ---------------------------------------------------------------------------

/// Look up a rename candidate stem in the extractor metadata.
///
/// Inspects in priority order:
/// - `tags.artist` + `tags.title` (audio)
/// - `tags.title` (audio without artist)
/// - `exif.image_description` (image)
/// - `exif.make` + `exif.model` (image fallback)
/// - `title` at the top level when supplied by an extractor
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
// Grounded keyword composition
// ---------------------------------------------------------------------------

/// Compose a stem from the top YAKE keywords, optionally prefixed with a year.
///
/// Keywords may be multi-word phrases (n-gram YAKE), so this flattens them, in
/// rank order, into a sequence of up to [`MAX_STEM_WORDS`] words and
/// de-duplicates at the **word** level — overlapping phrases like `"tax return"`
/// then `"tax form"` collapse to `tax_return_form` rather than repeating `tax`.
/// Returns `None` when no usable word survives sanitization.
fn stem_from_keywords(keywords: &[Keyword], year: Option<i32>) -> Option<String> {
    let mut words: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for kw in keywords {
        let token = sanitize_token(&kw.term);
        for word in token.split('_').filter(|w| !w.is_empty()) {
            if seen.insert(word.to_string()) {
                words.push(word.to_string());
            }
        }
        if words.len() >= MAX_STEM_WORDS {
            break;
        }
    }
    words.truncate(MAX_STEM_WORDS);
    if words.is_empty() {
        return None;
    }
    let body = words.join("_");
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

/// Sanitize a single keyword token, which may be a multi-word phrase. Lowercases,
/// maps any run of non-alphanumeric characters (notably the spaces inside a
/// phrase) to a single `_`, and trims leading/trailing separators — so the
/// caller can split the result back into words. Unlike [`sanitize_filename`] it
/// does not truncate; keyword tokens are short by construction.
fn sanitize_token(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_was_sep = true;
    for ch in raw.to_lowercase().chars() {
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
    out
}

fn truncate_at_boundary(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    // Clamp the budget to a char boundary first — `&s[..max]` panics if `max`
    // splits a multi-byte codepoint. (Today's caller passes ASCII, but the name
    // promises boundary-safety, so honour it.)
    let window = crate::text_util::char_prefix(s, max);
    if let Some(pos) = window.rfind('_') {
        // Only back off to the underscore if it leaves at least half the budget.
        if pos >= max / 2 {
            return window.get(..pos).unwrap_or(window).to_string();
        }
    }
    window.to_string()
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
    use chrono::Utc;
    use serde_json::json;
    use std::path::PathBuf;
    use tidyup_domain::ChangeStatus;
    use uuid::Uuid;

    use crate::yake::Keyword;

    fn kw(term: &str, score: f32) -> Keyword {
        Keyword {
            term: term.to_string(),
            score,
        }
    }

    fn renamed_member(original: &str, proposed_name: String) -> ChangeProposal {
        ChangeProposal {
            id: Uuid::new_v4(),
            file_id: None,
            change_type: ChangeType::RenameAndMove,
            original_path: PathBuf::from("/source").join(original),
            proposed_path: PathBuf::from("/target/collection").join(&proposed_name),
            proposed_name,
            confidence: 0.95,
            reasoning: "metadata rename".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: Some(0.95),
            rename_mismatch_score: Some(0.9),
            content_hash: None,
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
    fn bundle_member_names_suffix_colliding_metadata_renames() {
        let metadata = json!({"exif": {"make": "Canon", "model": "EOS R5"}});
        let proposed_names =
            ["IMG_0001.jpg", "IMG_0002.jpg"].map(|original| {
                match propose_rename(
                    &PathBuf::from("/source").join(original),
                    &metadata,
                    &[],
                    None,
                ) {
                    RenameProposal::Rename { name, .. } => name,
                    RenameProposal::Keep => panic!("camera metadata should propose a rename"),
                }
            });
        assert_eq!(proposed_names[0], proposed_names[1]);

        let mut members = vec![
            renamed_member("IMG_0001.jpg", proposed_names[0].clone()),
            renamed_member("IMG_0002.jpg", proposed_names[1].clone()),
        ];
        uniquify_bundle_member_names(&mut members).unwrap();

        assert_eq!(members[0].proposed_name, "canon_eos_r5.jpg");
        assert_eq!(members[1].proposed_name, "canon_eos_r5_2.jpg");
        assert!(members[1].proposed_path.ends_with("canon_eos_r5_2.jpg"));
    }

    #[test]
    fn move_only_collisions_are_rejected_instead_of_renamed_without_evidence() {
        let mut members = vec![
            renamed_member("A.txt", "A.txt".to_string()),
            renamed_member("a.txt", "a.txt".to_string()),
        ];
        for member in &mut members {
            member.change_type = ChangeType::Move;
            member.rename_mismatch_score = None;
        }

        let error = uniquify_bundle_member_names(&mut members).unwrap_err();
        assert!(error
            .to_string()
            .contains("move-only bundle members collide"));
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
    fn ocr_uses_general_keyword_evidence() {
        let meta = json!({"ocr_text": "Submitted quarterly expense report"});
        let keywords = vec![kw("quarterly expense", 0.1), kw("report", 0.2)];
        let p = propose_rename(
            &PathBuf::from("/d/Screenshot 2026-08-25 at 10.10.36 PM.png"),
            &meta,
            &keywords,
            None,
        );
        assert_eq!(
            p,
            RenameProposal::Rename {
                name: "quarterly_expense_report.png".to_string(),
                source: RenameSource::Ocr,
            }
        );
    }

    #[test]
    fn grounded_concepts_name_non_textual_image() {
        let concepts = vec![
            ConceptMatch {
                label: "cat".to_string(),
                score: 0.86,
                family: "animal".to_string(),
            },
            ConceptMatch {
                label: "beach".to_string(),
                score: 0.78,
                family: "scene".to_string(),
            },
        ];
        let p = propose_grounded_rename(&PathBuf::from("/d/image_2348985fg.png"), &concepts);
        assert_eq!(
            p,
            RenameProposal::Rename {
                name: "cat_on_beach.png".to_string(),
                source: RenameSource::SemanticConcept,
            }
        );
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
        assert_eq!(
            RenameSource::SemanticConcept.label(),
            "grounded semantic concepts"
        );
    }

    #[test]
    fn keyword_tier_flattens_phrases_word_deduplicated() {
        // n-gram YAKE can return phrases; overlapping words must not repeat.
        let kws = vec![kw("tax return", 0.05), kw("tax form", 0.06)];
        let p = propose_rename(&PathBuf::from("/d/scan.pdf"), &json!({}), &kws, None);
        match p {
            RenameProposal::Rename { name, source } => {
                assert_eq!(source, RenameSource::Keywords);
                assert_eq!(name, "tax_return_form.pdf");
            }
            RenameProposal::Keep => panic!("expected keyword rename"),
        }
    }

    #[test]
    fn keyword_tier_caps_total_words() {
        let kws = vec![kw("alpha beta gamma", 0.1), kw("delta epsilon", 0.2)];
        let p = propose_rename(&PathBuf::from("/d/x.pdf"), &json!({}), &kws, None);
        match p {
            RenameProposal::Rename { name, .. } => {
                // MAX_STEM_WORDS = 4 → alpha_beta_gamma_delta, epsilon dropped.
                assert_eq!(name, "alpha_beta_gamma_delta.pdf");
            }
            RenameProposal::Keep => panic!("expected rename"),
        }
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
