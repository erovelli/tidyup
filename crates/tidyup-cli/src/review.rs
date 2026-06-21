//! CLI [`ReviewHandler`](tidyup_core::frontend::ReviewHandler) impls:
//! - [`AutoApproveHandler`] — used under `--yes`; approve if confidence clears
//!   the threshold, otherwise reject. Bundle review (`review_bundles`) is never
//!   invoked under `--yes`: the service applies the threshold directly, so this
//!   handler relies on the trait's default (approve nothing) for bundles.
//! - [`InteractiveHandler`] — prompt-per-proposal via `console`. For each
//!   proposal, print a diff-like summary and read a single keystroke:
//!   `a` approve, `r` reject, `s`/`k` skip (same as reject), `q` reject all
//!   remaining, `Enter` = reject as default (safe choice). Renames that
//!   changed the filename are always surfaced explicitly even in `--yes`
//!   mode — but `--yes` doesn't auto-approve them; they stay rejected
//!   unless the user is in interactive mode (`CLAUDE.md` → "Don't auto-apply
//!   rename proposals"). Bundles get their own atomic approve/reject pass via
//!   [`InteractiveHandler::review_bundles`] after the loose-proposal pass.

use async_trait::async_trait;
use console::{style, Key, Term};
use tidyup_core::{frontend::ReviewHandler, Result};
use tidyup_domain::{BundleProposal, ChangeProposal, ChangeType, ReviewDecision};
use uuid::Uuid;

pub(crate) struct AutoApproveHandler {
    pub(crate) min_confidence: f32,
}

#[async_trait]
impl ReviewHandler for AutoApproveHandler {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
        Ok(proposals
            .into_iter()
            .map(|p| {
                // Renames never auto-apply, even under --yes.
                let is_rename = matches!(
                    p.change_type,
                    ChangeType::Rename | ChangeType::RenameAndMove
                );
                if !is_rename && p.confidence >= self.min_confidence {
                    ReviewDecision::Approve(p.id)
                } else {
                    ReviewDecision::Reject(p.id)
                }
            })
            .collect())
    }
}

pub(crate) struct InteractiveHandler;

#[async_trait]
impl ReviewHandler for InteractiveHandler {
    async fn review(&self, proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
        if proposals.is_empty() {
            return Ok(Vec::new());
        }
        // stdin reads are blocking; wrap in spawn_blocking.
        tokio::task::spawn_blocking(move || prompt_each(proposals))
            .await
            .map_err(|e| anyhow::anyhow!("interactive review task: {e}"))?
    }

    async fn review_bundles(&self, bundles: Vec<BundleProposal>) -> Result<Vec<Uuid>> {
        if bundles.is_empty() {
            return Ok(Vec::new());
        }
        tokio::task::spawn_blocking(move || prompt_each_bundle(bundles))
            .await
            .map_err(|e| anyhow::anyhow!("interactive bundle review task: {e}"))?
    }
}

fn prompt_each(proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
    let term = Term::stdout();
    let total = proposals.len();
    let _ = term.write_line(&format!(
        "\n{}",
        style(format!("Review — {total} proposal(s)")).bold().cyan()
    ));
    let _ = term.write_line("a=approve  r=reject  q=reject-all-remaining  ENTER=reject (default)");
    let _ = term.write_line("");

    let mut decisions = Vec::with_capacity(total);
    let mut reject_rest = false;
    for (i, p) in proposals.into_iter().enumerate() {
        if reject_rest {
            decisions.push(ReviewDecision::Reject(p.id));
            continue;
        }
        render_proposal(&term, i + 1, total, &p);
        loop {
            let key = match term.read_key() {
                Ok(k) => k,
                Err(e) => {
                    return Err(anyhow::anyhow!("reading stdin: {e}"));
                }
            };
            match key {
                Key::Char('a' | 'A') => {
                    decisions.push(ReviewDecision::Approve(p.id));
                    let _ = term.write_line(&style(" → approved").green().to_string());
                    break;
                }
                Key::Char('r' | 'R') => {
                    decisions.push(ReviewDecision::Reject(p.id));
                    let _ = term.write_line(&style(" → rejected").dim().to_string());
                    break;
                }
                Key::Char('q' | 'Q') => {
                    decisions.push(ReviewDecision::Reject(p.id));
                    let _ =
                        term.write_line(&style(" → rejecting all remaining").yellow().to_string());
                    reject_rest = true;
                    break;
                }
                Key::Enter => {
                    decisions.push(ReviewDecision::Reject(p.id));
                    let _ = term.write_line(&style(" → rejected (default)").dim().to_string());
                    break;
                }
                _ => {
                    let _ = term.write_line(&style(" (a/r/q/enter)").red().to_string());
                }
            }
        }
        let _ = term.write_line("");
    }
    Ok(decisions)
}

fn render_proposal(term: &Term, idx: usize, total: usize, p: &ChangeProposal) {
    let header = format!(
        "[{idx}/{total}] {}  (conf {:.2})",
        p.change_type.label(),
        p.confidence
    );
    let _ = term.write_line(&style(header).bold().to_string());
    let _ = term.write_line(&format!("  from: {}", p.original_path.display()));
    let _ = term.write_line(&format!("  to:   {}", p.proposed_path.display()));
    if p.proposed_name
        != p.original_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
    {
        let _ = term.write_line(&format!("  rename -> {}", style(&p.proposed_name).italic()));
    }
    let _ = term.write_line(&format!("  why:  {}", p.reasoning));
    if p.needs_review {
        let _ = term.write_line(
            &style("  ⚠ low-confidence — review carefully")
                .yellow()
                .to_string(),
        );
    }
}

/// Prompt per bundle, returning the ids of the bundles the user approved.
///
/// Bundles are atomic: the only choices are approve (move the whole subtree) or
/// reject (leave it pending). There is no per-member decision and no override —
/// members carry their own paths and never receive rename proposals. `Enter`
/// defaults to reject, the safe choice, mirroring the loose-proposal prompt.
fn prompt_each_bundle(bundles: Vec<BundleProposal>) -> Result<Vec<Uuid>> {
    let term = Term::stdout();
    let total = bundles.len();
    let _ = term.write_line(&format!(
        "\n{}",
        style(format!("Review — {total} bundle(s)")).bold().cyan()
    ));
    let _ = term.write_line("Bundles move atomically — the whole group or nothing.");
    let _ = term.write_line("a=approve  r=reject  q=reject-all-remaining  ENTER=reject (default)");
    let _ = term.write_line("");

    let mut approved = Vec::new();
    let mut reject_rest = false;
    for (i, b) in bundles.into_iter().enumerate() {
        if reject_rest {
            continue;
        }
        render_bundle(&term, i + 1, total, &b);
        loop {
            let key = match term.read_key() {
                Ok(k) => k,
                Err(e) => return Err(anyhow::anyhow!("reading stdin: {e}")),
            };
            match key {
                Key::Char('a' | 'A') => {
                    approved.push(b.id);
                    let _ = term.write_line(&style(" → approved").green().to_string());
                    break;
                }
                Key::Char('r' | 'R') => {
                    let _ = term.write_line(&style(" → rejected").dim().to_string());
                    break;
                }
                Key::Char('q' | 'Q') => {
                    let _ =
                        term.write_line(&style(" → rejecting all remaining").yellow().to_string());
                    reject_rest = true;
                    break;
                }
                Key::Enter => {
                    let _ = term.write_line(&style(" → rejected (default)").dim().to_string());
                    break;
                }
                _ => {
                    let _ = term.write_line(&style(" (a/r/q/enter)").red().to_string());
                }
            }
        }
        let _ = term.write_line("");
    }
    Ok(approved)
}

fn render_bundle(term: &Term, idx: usize, total: usize, b: &BundleProposal) {
    let header = format!(
        "[{idx}/{total}] {} bundle  ({} member(s), conf {:.2})",
        b.kind.as_str(),
        b.members.len(),
        b.confidence,
    );
    let _ = term.write_line(&style(header).bold().to_string());
    let _ = term.write_line(&format!("  root: {}", b.root.display()));
    let _ = term.write_line(&format!("  to:   {}/", b.target_parent.display()));
    let _ = term.write_line(&format!("  why:  {}", b.reasoning));
}
