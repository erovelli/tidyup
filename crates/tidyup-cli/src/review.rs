//! CLI [`ReviewHandler`](tidyup_core::frontend::ReviewHandler) impls:
//! - [`AutoApproveHandler`] — used under `--yes`; approve if confidence clears
//!   the threshold, otherwise reject. Bundle review (`review_bundles`) is never
//!   invoked under `--yes`: the service applies the threshold directly, so this
//!   handler relies on the trait's default (approve nothing) for bundles.
//! - [`InteractiveHandler`] — prompt-per-proposal via `console`. For each
//!   proposal, print a diff-like summary and read a single keystroke:
//!   `a` approve, `A` approve all remaining, `r` reject, `q` reject all
//!   remaining, `Esc` abort (reject the rest), `Enter` = reject as default
//!   (safe choice). Rename proposals are surfaced explicitly here so the user
//!   can approve them interactively; under `--yes` the [`AutoApproveHandler`]
//!   auto-rejects every rename, so renames are never auto-applied (`CLAUDE.md`
//!   → "Don't auto-apply rename proposals"). Bulk approve (`A`) covers moves
//!   only — a pending rename still gets its own explicit prompt. If there is no
//!   TTY (piped/redirected/CI), the handler errors up front pointing at
//!   `--yes`, rather than spinning on a stream that never yields a keystroke.
//!   Bundles get their own atomic approve/reject pass via
//!   [`InteractiveHandler::review_bundles`] after the loose-proposal pass.

use std::io::IsTerminal;

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

/// Refuse interactive review when there is no controlling terminal.
///
/// `Term::read_key` returns `Key::Unknown` immediately and forever on a non-TTY
/// stdout, which would spin the review loop at 100% CPU (and, when stdout is
/// redirected, grow the file unboundedly). Erroring — instead of silently
/// rejecting everything, which looks like a successful no-op — tells the user
/// how to proceed.
fn ensure_interactive_terminal(term: &Term) -> Result<()> {
    ensure_attended(std::io::stdin().is_terminal(), term.is_term())
}

fn ensure_attended(stdin_tty: bool, stdout_tty: bool) -> Result<()> {
    if stdin_tty && stdout_tty {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "interactive review needs a terminal, but stdin/stdout is not a TTY \
         (piped, redirected, or non-interactive). Re-run in a terminal, or pass \
         --yes to auto-approve moves above the confidence threshold (renames are \
         never auto-applied)."
    ))
}

const LOOSE_LEGEND: &str =
    "a=approve  A=approve-all-remaining  r=reject  q=reject-all-remaining  Esc=abort  ENTER=reject (default)";

const fn is_rename(p: &ChangeProposal) -> bool {
    matches!(
        p.change_type,
        ChangeType::Rename | ChangeType::RenameAndMove
    )
}

fn prompt_each(proposals: Vec<ChangeProposal>) -> Result<Vec<ReviewDecision>> {
    let term = Term::stdout();
    ensure_interactive_terminal(&term)?;
    let total = proposals.len();
    let _ = term.write_line(&format!(
        "\n{}",
        style(format!("Review — {total} proposal(s)")).bold().cyan()
    ));
    let _ = term.write_line(LOOSE_LEGEND);
    let _ = term.write_line("");

    let mut decisions = Vec::with_capacity(total);
    let mut reject_rest = false;
    let mut approve_rest = false;
    for (i, p) in proposals.into_iter().enumerate() {
        if reject_rest {
            decisions.push(ReviewDecision::Reject(p.id));
            continue;
        }
        // Bulk-approve covers moves only; a rename still surfaces for an
        // explicit keystroke (renames are never bulk/auto-approved).
        if approve_rest && !is_rename(&p) {
            decisions.push(ReviewDecision::Approve(p.id));
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
                Key::Char('a') => {
                    decisions.push(ReviewDecision::Approve(p.id));
                    let _ = term.write_line(&style(" → approved").green().to_string());
                    break;
                }
                Key::Char('A') => {
                    decisions.push(ReviewDecision::Approve(p.id));
                    approve_rest = true;
                    let _ = term.write_line(
                        &style(" → approving all remaining moves")
                            .green()
                            .to_string(),
                    );
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
                // Esc / Ctrl-C / Ctrl-D abort: reject this and everything left.
                // (Raw mode delivers Ctrl-C/D as control chars, not signals.)
                Key::Escape | Key::Char('\u{3}' | '\u{4}') => {
                    decisions.push(ReviewDecision::Reject(p.id));
                    let _ = term.write_line(
                        &style(" → aborted; rejecting all remaining")
                            .yellow()
                            .to_string(),
                    );
                    reject_rest = true;
                    break;
                }
                Key::Enter => {
                    decisions.push(ReviewDecision::Reject(p.id));
                    let _ = term.write_line(&style(" → rejected (default)").dim().to_string());
                    break;
                }
                // Non-TTY stdin yields Unknown forever; abort rather than spin
                // (defense in depth behind `ensure_interactive_terminal`).
                Key::Unknown => {
                    return Err(anyhow::anyhow!(
                        "interactive review got no keyboard input (non-interactive \
                         terminal); re-run in a TTY or pass --yes"
                    ));
                }
                _ => {
                    let _ = term.write_line(&style(" (a/A/r/q/Esc/enter)").red().to_string());
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
    ensure_interactive_terminal(&term)?;
    let total = bundles.len();
    let _ = term.write_line(&format!(
        "\n{}",
        style(format!("Review — {total} bundle(s)")).bold().cyan()
    ));
    let _ = term.write_line("Bundles move atomically — the whole group or nothing.");
    let _ = term.write_line(LOOSE_LEGEND);
    let _ = term.write_line("");

    let mut approved = Vec::new();
    let mut reject_rest = false;
    let mut approve_rest = false;
    for (i, b) in bundles.into_iter().enumerate() {
        if reject_rest {
            continue;
        }
        // Bundle members never carry renames, so bulk-approve is unconditional.
        if approve_rest {
            approved.push(b.id);
            continue;
        }
        render_bundle(&term, i + 1, total, &b);
        loop {
            let key = match term.read_key() {
                Ok(k) => k,
                Err(e) => return Err(anyhow::anyhow!("reading stdin: {e}")),
            };
            match key {
                Key::Char('a') => {
                    approved.push(b.id);
                    let _ = term.write_line(&style(" → approved").green().to_string());
                    break;
                }
                Key::Char('A') => {
                    approved.push(b.id);
                    approve_rest = true;
                    let _ =
                        term.write_line(&style(" → approving all remaining").green().to_string());
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
                Key::Escape | Key::Char('\u{3}' | '\u{4}') => {
                    let _ = term.write_line(
                        &style(" → aborted; rejecting all remaining")
                            .yellow()
                            .to_string(),
                    );
                    reject_rest = true;
                    break;
                }
                Key::Enter => {
                    let _ = term.write_line(&style(" → rejected (default)").dim().to_string());
                    break;
                }
                Key::Unknown => {
                    return Err(anyhow::anyhow!(
                        "interactive review got no keyboard input (non-interactive \
                         terminal); re-run in a TTY or pass --yes"
                    ));
                }
                _ => {
                    let _ = term.write_line(&style(" (a/A/r/q/Esc/enter)").red().to_string());
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tidyup_domain::ChangeStatus;

    fn proposal(change_type: ChangeType) -> ChangeProposal {
        ChangeProposal {
            id: Uuid::new_v4(),
            file_id: None,
            change_type,
            original_path: "/s/x.txt".into(),
            proposed_path: "/d/x.txt".into(),
            proposed_name: "x.txt".to_string(),
            confidence: 0.9,
            reasoning: "t".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: chrono::Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: None,
            rename_mismatch_score: None,
            content_hash: None,
        }
    }

    #[test]
    fn non_interactive_terminal_is_refused() {
        // With a real TTY on both ends, review proceeds.
        assert!(ensure_attended(true, true).is_ok());
        // Any missing TTY end must fail fast with a message pointing at --yes,
        // never spin on a stream that yields no keystroke.
        for (stdin_tty, stdout_tty) in [(false, true), (true, false), (false, false)] {
            let err = ensure_attended(stdin_tty, stdout_tty).unwrap_err();
            assert!(
                err.to_string().contains("--yes"),
                "non-tty error should point at --yes: {err}"
            );
        }
    }

    #[test]
    fn is_rename_flags_rename_change_types() {
        // Bulk-approve skips renames; this predicate is what protects the
        // "renames are never bulk/auto-approved" invariant.
        assert!(is_rename(&proposal(ChangeType::Rename)));
        assert!(is_rename(&proposal(ChangeType::RenameAndMove)));
        assert!(!is_rename(&proposal(ChangeType::Move)));
    }
}
