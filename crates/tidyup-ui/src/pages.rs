// Dioxus' `rsx!` expansion triggers a spurious `unused_qualifications` on
// event-handler attribute names on stable Rust 1.90. Scope the allow to this
// module so other lints stay strict.
#![allow(unused_qualifications)]
// SignalBundle is 448 bytes of Copy signal handles — shape matters more than
// indirection, so we pass by value everywhere. Each field is a cheap
// generational pointer; taking a reference would just add one hop.
#![allow(clippy::large_types_passed_by_value)]
// Pages construct one cloned `SharedState` per event handler so each closure
// captures its own cheap Arc-backed handle. Clippy flags the last such clone
// in a sequence as "redundant" (the original `state` goes unused after it),
// but losing the symmetry for that one case makes the handlers brittle to
// re-order.
#![allow(clippy::redundant_clone)]

//! Page components for the desktop UI.
//!
//! Each page reads from the shared [`SharedState`](crate::state::SharedState)
//! context and drives the same `tidyup-app` services the CLI calls. Pages
//! never own a `ServiceContext` across renders — each service invocation
//! builds a fresh one inside a tokio task, matching the CLI's one-shot
//! construction pattern.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use tidyup_app::config;
use tidyup_app::{
    migration::MigrationRequest, scan::ScanRequest, MigrationService, RollbackService, ScanService,
};
use tidyup_core::frontend::Level;
use tidyup_domain::{
    BundleProposal, ChangeProposal, ChangeType, ReviewDecision, RunRecord, RunState,
};
use uuid::Uuid;

use crate::context::{build, build_default_scan_candidates, quick_model_check};
use crate::reporter::DioxusReporter;
use crate::review::DioxusReviewHandler;
use crate::state::{Busy, LastReport, SharedState, SignalBundle};

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

#[component]
pub(crate) fn Dashboard() -> Element {
    let state = use_context::<SharedState>();
    let signals = state.signals;

    // Cheap on first paint: verify the embedding model is installed once and
    // cache the result. Avoids gating every Scan button press on filesystem.
    use_hook(|| {
        let mut ready = signals.model_ready;
        if ready.peek().is_none() {
            match quick_model_check() {
                Ok(()) => ready.set(Some(true)),
                Err(msg) => {
                    let mut err = signals.error;
                    err.set(Some(msg));
                    ready.set(Some(false));
                }
            }
        }
    });

    let source = use_signal(String::new);
    let target = use_signal(String::new);

    let model_ok = signals.model_ready.read().unwrap_or(false);
    let busy = *signals.busy.read();
    let disabled = !model_ok || busy != Busy::Idle;

    let scan_disabled = disabled || source.read().trim().is_empty();
    let migrate_disabled =
        disabled || source.read().trim().is_empty() || target.read().trim().is_empty();

    let scan_state = state.clone();
    let on_scan = move |_| {
        let src = source.read().trim().to_string();
        if src.is_empty() {
            return;
        }
        launch_scan(&scan_state, PathBuf::from(src));
    };

    let migrate_state = state.clone();
    let on_migrate = move |_| {
        let src = source.read().trim().to_string();
        let tgt = target.read().trim().to_string();
        if src.is_empty() || tgt.is_empty() {
            return;
        }
        launch_migrate(&migrate_state, PathBuf::from(src), PathBuf::from(tgt));
    };

    rsx! {
        div {
            h1 { class: "page-title", "Dashboard" }
            p {
                class: "page-subtitle",
                "tidyup runs entirely on-device. Point it at a source folder — and, for migration, the target hierarchy it should learn. Every move is proposed and reversible."
            }

            ModelBanner { signals }
            ErrorBanner { signals }
            PhaseBanner { signals }

            div {
                class: "card",
                h2 { class: "card-title", "Scan" }
                p { class: "card-subtitle", "Classify files in a directory against tidyup's built-in taxonomy." }
                PathField { label: "Source directory", value: source, placeholder: "/Users/you/Downloads" }
                div {
                    class: "button-row",
                    button {
                        class: "button button-primary",
                        disabled: scan_disabled,
                        onclick: on_scan,
                        "Start scan"
                    }
                }
            }

            div {
                class: "card",
                h2 { class: "card-title", "Migrate" }
                p { class: "card-subtitle", "Sort files from a source tree into the structure of an existing target hierarchy." }
                PathField { label: "Source directory", value: source, placeholder: "/Users/you/Downloads/incoming" }
                PathField { label: "Target hierarchy", value: target, placeholder: "/Users/you/Documents" }
                div {
                    class: "button-row",
                    button {
                        class: "button button-primary",
                        disabled: migrate_disabled,
                        onclick: on_migrate,
                        "Start migration"
                    }
                }
            }

            LogPane { signals }
            LastReportCard { signals }
        }
    }
}

// ---------------------------------------------------------------------------
// Review page
// ---------------------------------------------------------------------------

#[component]
pub(crate) fn Review() -> Element {
    let state = use_context::<SharedState>();
    let signals = state.signals;

    let proposals = signals.proposals.read().clone();
    let bundles = signals.bundles.read().clone();
    let pending = *signals.review_pending.read();

    let threshold = use_signal(|| 75_u32);
    let filter = use_signal(|| FilterMode::All);
    // Interactive diff state. Hover highlights both endpoints + the curve;
    // click on a row or curve "selects" the proposal and surfaces inline
    // approve/reject controls. Non-sync because only UI events write to them.
    let hovered = use_signal(|| Option::<Uuid>::None);
    let selected = use_signal(|| Option::<Uuid>::None);

    if !pending && proposals.is_empty() && bundles.is_empty() {
        let last = signals.last_report.read().clone();
        return rsx! {
            div {
                h1 { class: "page-title", "Review" }
                div {
                    class: "empty",
                    p { class: "empty-headline", "Nothing to review" }
                    p { "Start a scan or migration from the Dashboard. Proposals will appear here when classification finishes." }
                }
                if last.is_some() {
                    LastReportCard { signals }
                }
            }
        };
    }

    let model = build_diff_model(&proposals);
    let applied_count = signals.last_report.read().as_ref().map_or(0, |r| match r {
        LastReport::Scan(s) => s.applied,
        LastReport::Migration(m) => m.applied,
        LastReport::Rollback(_) => 0,
    });
    let folder_count = count_folders(&model.left_rows);

    let filtered: Vec<ChangeProposal> = filter_proposals(&proposals, *filter.read());

    rsx! {
        div {
            h1 { class: "page-title", "Review" }
            PhaseBanner { signals }

            SummaryCards {
                indexed: proposals.len(),
                pending: if pending { proposals.len() } else { 0 },
                applied: applied_count,
            }

            DiffHeader {
                proposals_count: proposals.len(),
                folder_count,
                threshold,
                pending,
                state: state.clone(),
            }

            DiffView { model, hovered, selected, signals }

            DiffLegend {}

            if !bundles.is_empty() {
                div {
                    class: "card",
                    style: "margin-top: 24px;",
                    h2 { class: "card-title", "{bundles.len()} bundle(s) held" }
                    p { class: "card-subtitle muted", "Bundles move as atomic units. Full review UX lands in Phase 7+." }
                    div {
                        for b in bundles.iter().cloned() {
                            BundleCard { key: "{b.id}", bundle: b }
                        }
                    }
                }
            }

            div {
                class: "section-heading",
                style: "margin-top: 32px;",
                "DETAILED CHANGES"
            }
            FilterTabs { filter, proposals: proposals.clone() }

            div {
                class: "card-stack",
                style: "margin-top: 16px;",
                for p in filtered {
                    ProposalCard { key: "{p.id}", proposal: p, signals }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Diff view — tree (proposed) + flat list (current) + bezier-curve overlay.
//
// All rows use the same fixed height so Y positions for the SVG paths can be
// derived from row indices without measuring the DOM.
// ---------------------------------------------------------------------------

const ROW_HEIGHT: u32 = 42;
const DIFF_TOP_PAD: u32 = 12;

/// Pixel height for `n` rows including top/bottom padding.
fn rows_to_height(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX / ROW_HEIGHT) * ROW_HEIGHT + DIFF_TOP_PAD * 2
}

/// Vertical center of the row at `idx` inside the diff body.
fn row_center_y(idx: usize) -> u32 {
    DIFF_TOP_PAD + u32::try_from(idx).unwrap_or(0) * ROW_HEIGHT + ROW_HEIGHT / 2
}

#[derive(Clone, PartialEq)]
struct DiffModel {
    left_rows: Vec<TreeRow>,
    /// Proposal id → row index in `left_rows`.
    file_row_by_id: HashMap<Uuid, usize>,
    /// Entries for the right column, pre-ordered to minimise curve crossings.
    right_entries: Vec<RightEntry>,
    /// Displayed label for the root (derived from the common parent of all proposed paths).
    root_label: String,
}

#[derive(Clone, PartialEq)]
enum TreeRow {
    Folder {
        name: String,
        depth: usize,
        file_count: usize,
    },
    File {
        name: String,
        depth: usize,
        confidence: f32,
        change_type: ChangeType,
        proposal_id: Uuid,
    },
}

#[derive(Clone, PartialEq)]
struct RightEntry {
    proposal_id: Uuid,
    display_name: String,
    confidence: f32,
}

fn build_diff_model(proposals: &[ChangeProposal]) -> DiffModel {
    let parents: Vec<PathBuf> = proposals
        .iter()
        .map(|p| {
            p.proposed_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default()
        })
        .collect();
    let root = common_ancestor(&parents);
    let root_label = root
        .file_name()
        .and_then(|s| s.to_str())
        .map_or_else(|| root.display().to_string(), ToString::to_string);

    // Build tree keyed by relative path components.
    let mut tree = TreeBuilder::default();
    for (idx, p) in proposals.iter().enumerate() {
        let parent = p.proposed_path.parent().unwrap_or_else(|| Path::new(""));
        let rel: Vec<String> = parent
            .strip_prefix(&root)
            .unwrap_or(parent)
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(ToString::to_string))
            .collect();
        let filename = p
            .proposed_path
            .file_name()
            .and_then(|s| s.to_str())
            .map_or_else(|| p.proposed_name.clone(), ToString::to_string);
        tree.insert(&rel, filename, idx);
    }

    let mut left_rows = Vec::new();
    let mut file_row_by_id = HashMap::new();
    tree.flatten(0, proposals, &mut left_rows, &mut file_row_by_id);

    // Order right entries by corresponding left row index — the key to keeping
    // bezier curves from spaghetti-ing across each other.
    let mut right_entries: Vec<RightEntry> = proposals
        .iter()
        .map(|p| RightEntry {
            proposal_id: p.id,
            display_name: p
                .original_path
                .file_name()
                .and_then(|s| s.to_str())
                .map_or_else(
                    || p.original_path.display().to_string(),
                    ToString::to_string,
                ),
            confidence: p.confidence,
        })
        .collect();
    right_entries.sort_by_key(|e| {
        file_row_by_id
            .get(&e.proposal_id)
            .copied()
            .unwrap_or(usize::MAX)
    });

    DiffModel {
        left_rows,
        file_row_by_id,
        right_entries,
        root_label,
    }
}

fn common_ancestor(paths: &[PathBuf]) -> PathBuf {
    let mut iter = paths.iter();
    let Some(first) = iter.next() else {
        return PathBuf::new();
    };
    let mut acc: Vec<std::ffi::OsString> = first
        .components()
        .map(|c| c.as_os_str().to_owned())
        .collect();
    for p in iter {
        let comps: Vec<std::ffi::OsString> =
            p.components().map(|c| c.as_os_str().to_owned()).collect();
        let n = acc
            .iter()
            .zip(comps.iter())
            .take_while(|(a, b)| a == b)
            .count();
        acc.truncate(n);
        if acc.is_empty() {
            break;
        }
    }
    acc.into_iter().collect()
}

#[derive(Default)]
struct TreeBuilder {
    folders: BTreeMap<String, TreeBuilder>,
    files: Vec<(String, usize)>, // (filename, proposal index)
}

impl TreeBuilder {
    fn insert(&mut self, folders: &[String], filename: String, idx: usize) {
        let mut node = self;
        for folder in folders {
            node = node.folders.entry(folder.clone()).or_default();
        }
        node.files.push((filename, idx));
    }

    fn file_count(&self) -> usize {
        self.files.len() + self.folders.values().map(Self::file_count).sum::<usize>()
    }

    fn flatten(
        &self,
        depth: usize,
        proposals: &[ChangeProposal],
        rows: &mut Vec<TreeRow>,
        file_row_by_id: &mut HashMap<Uuid, usize>,
    ) {
        for (name, child) in &self.folders {
            rows.push(TreeRow::Folder {
                name: name.clone(),
                depth,
                file_count: child.file_count(),
            });
            child.flatten(depth + 1, proposals, rows, file_row_by_id);
        }
        let mut files = self.files.clone();
        files.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, idx) in files {
            let p = &proposals[idx];
            let row_idx = rows.len();
            rows.push(TreeRow::File {
                name,
                depth,
                confidence: p.confidence,
                change_type: p.change_type.clone(),
                proposal_id: p.id,
            });
            file_row_by_id.insert(p.id, row_idx);
        }
    }
}

const fn count_folders(rows: &[TreeRow]) -> usize {
    let mut i = 0;
    let mut count = 0;
    while i < rows.len() {
        if matches!(rows[i], TreeRow::Folder { .. }) {
            count += 1;
        }
        i += 1;
    }
    count
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FilterMode {
    All,
    High,
    Medium,
    NeedsReview,
}

fn filter_proposals(proposals: &[ChangeProposal], mode: FilterMode) -> Vec<ChangeProposal> {
    proposals
        .iter()
        .filter(|p| match mode {
            FilterMode::All => true,
            FilterMode::High => p.confidence >= 0.80,
            FilterMode::Medium => (0.60..0.80).contains(&p.confidence),
            FilterMode::NeedsReview => p.needs_review || p.confidence < 0.60,
        })
        .cloned()
        .collect()
}

#[component]
fn SummaryCards(indexed: usize, pending: usize, applied: usize) -> Element {
    rsx! {
        div {
            class: "summary-cards",
            SummaryCard { value: "{indexed}", label: "Files indexed" }
            SummaryCard { value: "{pending}", label: "Pending review" }
            SummaryCard { value: "{applied}", label: "Applied" }
        }
    }
}

#[component]
fn SummaryCard(value: String, label: &'static str) -> Element {
    rsx! {
        div {
            class: "summary-card",
            div { class: "summary-value", "{value}" }
            div { class: "summary-label", "{label}" }
        }
    }
}

#[component]
fn DiffHeader(
    proposals_count: usize,
    folder_count: usize,
    threshold: Signal<u32>,
    pending: bool,
    state: SharedState,
) -> Element {
    let threshold_val = *threshold.read();
    let on_input = move |ev: Event<FormData>| {
        let mut t = threshold;
        if let Ok(n) = ev.value().parse::<u32>() {
            t.set(n.min(100));
        }
    };

    let execute_state = state.clone();
    let on_execute = move |_| execute_with_threshold(&execute_state, threshold_val);
    let reject_state = state.clone();
    let on_reject = move |_| reject_all_and_submit(&reject_state);

    rsx! {
        div {
            class: "diff-header",
            div {
                class: "diff-header-title",
                div { class: "diff-marker" }
                div {
                    h2 { class: "diff-title", "Proposed Migration" }
                    div {
                        class: "diff-subtitle",
                        "{proposals_count} proposals across {folder_count} folders"
                    }
                }
            }
            div {
                class: "diff-header-actions",
                div {
                    class: "threshold-input",
                    span { class: "threshold-label", "Approve all ≥" }
                    input {
                        r#type: "number",
                        min: "0",
                        max: "100",
                        step: "1",
                        value: "{threshold_val}",
                        oninput: on_input,
                        class: "threshold-number",
                    }
                    span { class: "threshold-unit", "%" }
                }
                button {
                    r#type: "button",
                    class: "button button-primary",
                    disabled: !pending,
                    onclick: on_execute,
                    "Execute"
                }
                button {
                    r#type: "button",
                    class: "button button-secondary",
                    disabled: !pending,
                    onclick: on_reject,
                    "Reject all"
                }
            }
        }
    }
}

fn execute_with_threshold(state: &SharedState, threshold_pct: u32) {
    let signals = state.signals;
    let proposals = signals.proposals.read().clone();
    let threshold = f32::from(u16::try_from(threshold_pct).unwrap_or(100)) / 100.0;
    let mut decisions = signals.decisions;
    decisions.with_mut(|map| {
        for p in &proposals {
            // Preserve any decision the user has already made — only fill in
            // untouched proposals. Otherwise Execute wipes manual Approve/Reject
            // clicks and everyone wonders why nothing moved.
            if map.contains_key(&p.id) {
                continue;
            }
            let is_rename = matches!(
                p.change_type,
                ChangeType::Rename | ChangeType::RenameAndMove
            );
            // Renames never auto-apply, even above threshold — per CLAUDE.md
            // "Rename policy". User still sees them in Detailed Changes.
            let approve = !is_rename && p.confidence >= threshold;
            let decision = if approve {
                ReviewDecision::Approve(p.id)
            } else {
                ReviewDecision::Reject(p.id)
            };
            map.insert(p.id, decision);
        }
    });
    submit_review(state);
}

/// Pre-computed render data for a single connector between the two columns.
#[derive(Clone, PartialEq, Eq)]
struct ConnectorRender {
    id: Uuid,
    d: String,
    strong: bool,
    dashed: bool,
}

/// Derived review state for a single proposal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecisionState {
    Undecided,
    Approved,
    Rejected,
}

fn decision_state_of(
    decisions: &std::collections::HashMap<Uuid, ReviewDecision>,
    id: Uuid,
) -> DecisionState {
    match decisions.get(&id) {
        Some(ReviewDecision::Approve(_) | ReviewDecision::Override { .. }) => {
            DecisionState::Approved
        }
        Some(ReviewDecision::Reject(_)) => DecisionState::Rejected,
        None => DecisionState::Undecided,
    }
}

const fn row_state_class(state: DecisionState) -> &'static str {
    match state {
        DecisionState::Approved => " row-approved",
        DecisionState::Rejected => " row-rejected",
        DecisionState::Undecided => "",
    }
}

#[component]
fn DiffView(
    model: DiffModel,
    hovered: Signal<Option<Uuid>>,
    selected: Signal<Option<Uuid>>,
    signals: SignalBundle,
) -> Element {
    let left_height = rows_to_height(model.left_rows.len());
    let right_height = rows_to_height(model.right_entries.len());
    let svg_height = left_height.max(right_height);

    let connectors: Vec<ConnectorRender> = model
        .right_entries
        .iter()
        .enumerate()
        .filter_map(|(right_idx, entry)| {
            let &left_idx = model.file_row_by_id.get(&entry.proposal_id)?;
            let ly = row_center_y(left_idx);
            let ry = row_center_y(right_idx);
            let d = format!("M 0 {ly} C 50 {ly}, 50 {ry}, 100 {ry}");
            Some(ConnectorRender {
                id: entry.proposal_id,
                d,
                strong: entry.confidence >= 0.80,
                dashed: false,
            })
        })
        .collect();

    let hovered_id = *hovered.read();
    let selected_id = *selected.read();

    // Map proposal id → left row index for the key prop on TreeRowView. Using
    // the proposal id as key (not the row index) so dioxus re-uses the same
    // element across renders even if ordering shifts.
    let left_rows = model.left_rows.clone();

    rsx! {
        div {
            class: "diff-view",
            div {
                class: "diff-col diff-proposed",
                div { class: "diff-col-header", "PROPOSED STRUCTURE" }
                div {
                    class: "diff-col-body",
                    style: "height: {left_height}px;",
                    for (i, row) in left_rows.iter().enumerate() {
                        TreeRowView {
                            key: "{tree_row_key(i, row)}",
                            row: row.clone(),
                            hovered,
                            selected,
                            signals,
                        }
                    }
                }
            }
            div {
                class: "diff-gap",
                style: "height: {svg_height}px;",
                svg {
                    class: "diff-overlay",
                    width: "100%",
                    height: "{svg_height}",
                    view_box: "0 0 100 {svg_height}",
                    preserve_aspect_ratio: "none",
                    for c in connectors.iter().cloned() {
                        ConnectorPath {
                            key: "{c.id}",
                            connector: c,
                            hovered,
                            selected,
                            hovered_id,
                            selected_id,
                            signals,
                        }
                    }
                }
            }
            div {
                class: "diff-col diff-current",
                div { class: "diff-col-header", "CURRENT STORAGE" }
                div {
                    class: "diff-col-body",
                    style: "height: {right_height}px;",
                    for entry in model.right_entries.iter().cloned() {
                        CurrentRow {
                            key: "{entry.proposal_id}",
                            entry,
                            hovered,
                            selected,
                            signals,
                        }
                    }
                }
            }
        }
    }
}

fn tree_row_key(i: usize, row: &TreeRow) -> String {
    match row {
        TreeRow::File { proposal_id, .. } => format!("file-{proposal_id}"),
        TreeRow::Folder { name, depth, .. } => format!("folder-{i}-{depth}-{name}"),
    }
}

/// Toggles `signal` to `id`, or clears it if already set to `id`.
fn toggle_selection(mut signal: Signal<Option<Uuid>>, id: Uuid) {
    let current = *signal.peek();
    if current == Some(id) {
        signal.set(None);
    } else {
        signal.set(Some(id));
    }
}

#[component]
fn ConnectorPath(
    connector: ConnectorRender,
    hovered: Signal<Option<Uuid>>,
    selected: Signal<Option<Uuid>>,
    hovered_id: Option<Uuid>,
    selected_id: Option<Uuid>,
    signals: SignalBundle,
) -> Element {
    let is_hovered = hovered_id == Some(connector.id);
    let is_selected = selected_id == Some(connector.id);
    let active = is_hovered || is_selected;
    let state = decision_state_of(&signals.decisions.read(), connector.id);

    // Decision state drives color; interactive state drives width. Rejected
    // connectors also get a thinner base so they visually recede.
    let stroke = match state {
        DecisionState::Approved => "var(--connector-approved)",
        DecisionState::Rejected => "var(--connector-rejected)",
        DecisionState::Undecided if active => "var(--connector-active)",
        DecisionState::Undecided if connector.strong => "var(--connector-strong)",
        DecisionState::Undecided => "var(--connector)",
    };
    let stroke_width = if active {
        "3.5"
    } else if matches!(state, DecisionState::Rejected) {
        "1"
    } else {
        "1.5"
    };
    let dash = if connector.dashed || matches!(state, DecisionState::Rejected) {
        "4 4"
    } else {
        "0"
    };

    let conn_id = connector.id;
    let on_enter = move |_| {
        let mut h = hovered;
        h.set(Some(conn_id));
    };
    let on_leave = move |_| {
        let mut h = hovered;
        h.set(None);
    };
    let on_click = move |_| toggle_selection(selected, conn_id);

    rsx! {
        // Invisible fat "hit area" so the thin visible stroke is easier to
        // point at. `pointer-events: stroke` restricts hits to the curve.
        path {
            class: "diff-hit",
            d: "{connector.d}",
            fill: "none",
            stroke: "transparent",
            stroke_width: "14",
            vector_effect: "non-scaling-stroke",
            onmouseenter: on_enter,
            onmouseleave: on_leave,
            onclick: on_click,
        }
        path {
            class: "diff-stroke",
            d: "{connector.d}",
            fill: "none",
            stroke: stroke,
            stroke_width: stroke_width,
            stroke_dasharray: dash,
            vector_effect: "non-scaling-stroke",
        }
    }
}

#[component]
fn TreeRowView(
    row: TreeRow,
    hovered: Signal<Option<Uuid>>,
    selected: Signal<Option<Uuid>>,
    signals: SignalBundle,
) -> Element {
    match row {
        TreeRow::Folder {
            name,
            depth,
            file_count,
        } => {
            let pad = 12 + depth * 24;
            rsx! {
                div {
                    class: "tree-row tree-folder",
                    style: "padding-left: {pad}px;",
                    span { class: "tree-caret", "▸" }
                    span { class: "tree-folder-name", "{name}/" }
                    span {
                        class: "tree-filecount",
                        if file_count == 1 { "1 file" } else { "{file_count} files" }
                    }
                }
            }
        }
        TreeRow::File {
            name,
            depth,
            confidence,
            change_type,
            proposal_id,
        } => {
            let pad = 12 + depth * 24;
            let chip = confidence_chip(confidence);
            let is_rename = matches!(change_type, ChangeType::Rename | ChangeType::RenameAndMove);

            let hovered_id = *hovered.read();
            let selected_id = *selected.read();
            let is_hovered = hovered_id == Some(proposal_id);
            let is_selected = selected_id == Some(proposal_id);
            let state = decision_state_of(&signals.decisions.read(), proposal_id);

            let mut row_class = String::from("tree-row tree-file");
            row_class.push_str(row_state_class(state));
            if is_hovered {
                row_class.push_str(" row-hovered");
            }
            if is_selected {
                row_class.push_str(" row-selected");
            }

            let on_enter = move |_| {
                let mut h = hovered;
                h.set(Some(proposal_id));
            };
            let on_leave = move |_| {
                let mut h = hovered;
                h.set(None);
            };
            let on_click = move |_| toggle_selection(selected, proposal_id);

            rsx! {
                div {
                    class: "{row_class}",
                    style: "padding-left: {pad}px;",
                    onmouseenter: on_enter,
                    onmouseleave: on_leave,
                    onclick: on_click,
                    span { class: "tree-file-name", "{name}" }
                    span {
                        class: "tree-row-meta",
                        if is_rename {
                            span { class: "chip chip-neutral", "rename" }
                        }
                        span { class: "chip {chip.0} tree-confidence", "{chip.1}" }
                    }
                }
            }
        }
    }
}

#[component]
fn CurrentRow(
    entry: RightEntry,
    hovered: Signal<Option<Uuid>>,
    selected: Signal<Option<Uuid>>,
    signals: SignalBundle,
) -> Element {
    let hovered_id = *hovered.read();
    let selected_id = *selected.read();
    let is_hovered = hovered_id == Some(entry.proposal_id);
    let is_selected = selected_id == Some(entry.proposal_id);
    let state = decision_state_of(&signals.decisions.read(), entry.proposal_id);

    let mut row_class = String::from("current-row");
    row_class.push_str(row_state_class(state));
    if is_hovered {
        row_class.push_str(" row-hovered");
    }
    if is_selected {
        row_class.push_str(" row-selected");
    }

    let pid = entry.proposal_id;
    let on_enter = move |_| {
        let mut h = hovered;
        h.set(Some(pid));
    };
    let on_leave = move |_| {
        let mut h = hovered;
        h.set(None);
    };
    let on_click = move |_| toggle_selection(selected, pid);

    let on_approve = move |ev: MouseEvent| {
        ev.stop_propagation(); // don't also toggle the row selection
        let mut d = signals.decisions;
        d.with_mut(|map| {
            map.insert(pid, ReviewDecision::Approve(pid));
        });
    };
    let on_reject = move |ev: MouseEvent| {
        ev.stop_propagation();
        let mut d = signals.decisions;
        d.with_mut(|map| {
            map.insert(pid, ReviewDecision::Reject(pid));
        });
    };

    // Three-state button "active" visual: undecided → neither active.
    let approve_class = if state == DecisionState::Approved {
        "mini-button mini-button-approve active"
    } else {
        "mini-button mini-button-approve"
    };
    let reject_class = if state == DecisionState::Rejected {
        "mini-button mini-button-reject active"
    } else {
        "mini-button mini-button-reject"
    };

    rsx! {
        div {
            class: "{row_class}",
            onmouseenter: on_enter,
            onmouseleave: on_leave,
            onclick: on_click,
            span { class: "current-name", "{entry.display_name}" }
            if is_selected {
                span {
                    class: "current-actions",
                    button {
                        r#type: "button",
                        class: "{approve_class}",
                        onclick: on_approve,
                        "Approve"
                    }
                    button {
                        r#type: "button",
                        class: "{reject_class}",
                        onclick: on_reject,
                        "Reject"
                    }
                }
            }
        }
    }
}

#[component]
fn DiffLegend() -> Element {
    rsx! {
        div {
            class: "diff-legend",
            span { class: "legend-item",
                span { class: "legend-line legend-solid" }
                "MOVE"
            }
            span { class: "legend-item",
                span { class: "legend-line legend-dashed" }
                "RENAME"
            }
        }
    }
}

#[component]
fn FilterTabs(filter: Signal<FilterMode>, proposals: Vec<ChangeProposal>) -> Element {
    let current = *filter.read();
    let all_n = proposals.len();
    let high_n = proposals.iter().filter(|p| p.confidence >= 0.80).count();
    let mid_n = proposals
        .iter()
        .filter(|p| (0.60..0.80).contains(&p.confidence))
        .count();
    let review_n = proposals
        .iter()
        .filter(|p| p.needs_review || p.confidence < 0.60)
        .count();

    let set = move |mode: FilterMode| {
        move |_| {
            let mut f = filter;
            f.set(mode);
        }
    };

    rsx! {
        div {
            class: "filter-tabs",
            FilterPill { label: "All",           count: all_n,    active: current == FilterMode::All,         onclick: set(FilterMode::All) }
            FilterPill { label: "High ≥80%",     count: high_n,   active: current == FilterMode::High,        onclick: set(FilterMode::High) }
            FilterPill { label: "Medium",        count: mid_n,    active: current == FilterMode::Medium,      onclick: set(FilterMode::Medium) }
            FilterPill { label: "Needs review",  count: review_n, active: current == FilterMode::NeedsReview, onclick: set(FilterMode::NeedsReview) }
        }
    }
}

#[component]
fn FilterPill(
    label: &'static str,
    count: usize,
    active: bool,
    onclick: EventHandler<MouseEvent>,
) -> Element {
    let class = if active {
        "filter-pill filter-pill-active"
    } else {
        "filter-pill"
    };
    rsx! {
        button {
            r#type: "button",
            class: "{class}",
            onclick: move |ev| onclick.call(ev),
            span { class: "filter-pill-label", "{label}" }
            span { class: "filter-pill-count", "{count}" }
        }
    }
}

#[component]
fn ProposalCard(proposal: ChangeProposal, signals: SignalBundle) -> Element {
    let decisions = signals.decisions;
    let state = decision_state_of(&decisions.read(), proposal.id);

    let from = proposal.original_path.display().to_string();
    let to = proposal.proposed_path.display().to_string();
    let conf = proposal.confidence;
    let chip = confidence_chip(conf);

    let is_rename = matches!(
        proposal.change_type,
        ChangeType::Rename | ChangeType::RenameAndMove
    );

    let card_class = match state {
        DecisionState::Approved => "proposal proposal-approved",
        DecisionState::Rejected => "proposal proposal-rejected",
        DecisionState::Undecided => "proposal",
    };

    let proposal_id = proposal.id;
    let change_label = proposal.change_type.label();

    let on_approve = move |_| {
        let mut d = decisions;
        d.with_mut(|map| {
            map.insert(
                proposal_id,
                tidyup_domain::ReviewDecision::Approve(proposal_id),
            );
        });
    };
    let on_reject = move |_| {
        let mut d = decisions;
        d.with_mut(|map| {
            map.insert(
                proposal_id,
                tidyup_domain::ReviewDecision::Reject(proposal_id),
            );
        });
    };

    let approve_class = if state == DecisionState::Approved {
        "mini-button mini-button-approve active"
    } else {
        "mini-button mini-button-approve"
    };
    let reject_class = if state == DecisionState::Rejected {
        "mini-button mini-button-reject active"
    } else {
        "mini-button mini-button-reject"
    };

    rsx! {
        div {
            class: "{card_class}",
            div {
                class: "proposal-meta",
                div {
                    class: "proposal-target",
                    "{proposal.proposed_name}"
                }
                div {
                    class: "proposal-path",
                    "{from}"
                    span { class: "proposal-arrow", " → " }
                    "{to}"
                }
                div {
                    class: "proposal-reason",
                    "{proposal.reasoning}"
                }
                div {
                    class: "button-row small",
                    style: "margin-top: 6px;",
                    span { class: "chip {chip.0}", "{chip.1}" }
                    span { class: "chip chip-neutral", "{change_label}" }
                    if is_rename {
                        span { class: "chip chip-neutral", "rename" }
                    }
                    if proposal.needs_review {
                        span { class: "chip chip-low", "needs review" }
                    }
                }
            }
            div {
                class: "proposal-actions",
                button {
                    class: "{approve_class}",
                    onclick: on_approve,
                    "Approve"
                }
                button {
                    class: "{reject_class}",
                    onclick: on_reject,
                    "Reject"
                }
            }
        }
    }
}

#[component]
fn BundleCard(bundle: BundleProposal) -> Element {
    let root = bundle.root.display().to_string();
    let target = bundle.target_parent.display().to_string();
    let kind = bundle.kind.as_str();
    let chip = confidence_chip(bundle.confidence);
    let member_count = bundle.members.len();

    rsx! {
        div {
            class: "proposal",
            div {
                class: "proposal-meta",
                div {
                    class: "proposal-target",
                    "{root}"
                }
                div {
                    class: "proposal-path",
                    "{member_count} member(s) → {target}"
                }
                div {
                    class: "proposal-reason",
                    "{bundle.reasoning}"
                }
                div {
                    class: "button-row small",
                    style: "margin-top: 6px;",
                    span { class: "chip {chip.0}", "{chip.1}" }
                    span { class: "chip chip-neutral", "{kind}" }
                }
            }
            div {
                class: "proposal-actions",
                span {
                    class: "muted small",
                    "held for review"
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Runs page
// ---------------------------------------------------------------------------

#[component]
pub(crate) fn Runs() -> Element {
    let state = use_context::<SharedState>();
    let signals = state.signals;

    // Load runs once when the page mounts.
    {
        let state = state.clone();
        use_hook(move || refresh_runs(&state));
    }

    let runs = signals.runs.read().clone();
    let busy = *signals.busy.read();

    let refresh_state = state.clone();
    let on_refresh = move |_| refresh_runs(&refresh_state);

    rsx! {
        div {
            h1 { class: "page-title", "Runs" }
            p {
                class: "page-subtitle",
                "Every scan and migration is recorded. Rollback restores originals from the backup shelf in reverse order."
            }

            ErrorBanner { signals }
            PhaseBanner { signals }

            div {
                class: "button-row",
                style: "margin-bottom: 16px;",
                button {
                    class: "button button-secondary",
                    onclick: on_refresh,
                    "Refresh"
                }
            }

            if runs.is_empty() {
                div {
                    class: "empty",
                    p { class: "empty-headline", "No recorded runs" }
                    p { "Once you run a scan or migration, it will be listed here with a rollback button." }
                }
            } else {
                div {
                    class: "runs",
                    for run in runs.iter().cloned() {
                        RunRow { key: "{run.id}", run, busy }
                    }
                }
            }
        }
    }
}

#[component]
fn RunRow(run: RunRecord, busy: Busy) -> Element {
    let state = use_context::<SharedState>();
    let mode = run.mode.as_str();
    let state_label = run.state.as_str();
    let source = run.source_root.display().to_string();
    let target = run
        .target_root
        .as_ref()
        .map_or_else(String::new, |p| p.display().to_string());
    let can_rollback = matches!(run.state, RunState::Completed) && busy == Busy::Idle;
    let run_id = run.id;

    let rollback_state = state.clone();
    let on_rollback = move |_| launch_rollback(&rollback_state, run_id);

    rsx! {
        div {
            class: "run-row",
            span { class: "run-mode", "{mode}" }
            span { class: "chip chip-neutral", "{state_label}" }
            div {
                class: "run-paths",
                "{source}"
                if !target.is_empty() {
                    span { " → {target}" }
                }
            }
            button {
                class: "button button-danger",
                disabled: !can_rollback,
                onclick: on_rollback,
                "Rollback"
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[component]
pub(crate) fn Settings() -> Element {
    let cfg_result = config::load();
    let config_path = config::platform_config_path()
        .map_or_else(|_| "<unresolved>".into(), |p| p.display().to_string());

    match cfg_result {
        Ok(cfg) => {
            let data_dir = config::resolve_data_dir(&cfg.storage)
                .map_or_else(|_| "<unresolved>".into(), |p| p.display().to_string());
            let toml_text =
                toml::to_string_pretty(&cfg).unwrap_or_else(|e| format!("<error: {e}>"));
            rsx! {
                div {
                    h1 { class: "page-title", "Settings" }
                    p {
                        class: "page-subtitle",
                        "Read-only view of the loaded TOML config. Edit the file directly; changes apply on next launch."
                    }

                    div {
                        class: "card",
                        h2 { class: "card-title", "Paths" }
                        div {
                            class: "kv",
                            div { class: "kv-key", "config file" }
                            div { class: "kv-value", "{config_path}" }
                            div { class: "kv-key", "data dir" }
                            div { class: "kv-value", "{data_dir}" }
                        }
                    }

                    div {
                        class: "card",
                        h2 { class: "card-title", "Current config" }
                        pre {
                            style: "background: var(--surface-container-low); padding: 16px; border-radius: 8px; font-size: 12px; overflow-x: auto; margin: 0;",
                            "{toml_text}"
                        }
                    }
                }
            }
        }
        Err(e) => rsx! {
            div {
                h1 { class: "page-title", "Settings" }
                div {
                    class: "banner banner-error",
                    strong { "Could not load config." }
                    pre { "{e}" }
                }
            }
        },
    }
}

// ---------------------------------------------------------------------------
// Shared UI bits
// ---------------------------------------------------------------------------

#[component]
fn PathField(label: &'static str, value: Signal<String>, placeholder: &'static str) -> Element {
    let current = value.read().clone();
    let on_input = move |ev: Event<FormData>| {
        let mut v = value;
        v.set(ev.value());
    };
    // Native folder picker. `AsyncFileDialog` routes the call to the platform
    // main thread on macOS, so it plays nicely with dioxus-desktop's winit
    // event loop. The sync variant would deadlock there.
    let starting_dir = current.clone();
    let on_browse = move |_| {
        let mut v = value;
        let start = starting_dir.clone();
        spawn_forever(async move {
            let mut dialog = rfd::AsyncFileDialog::new().set_title("Choose a directory");
            let start_path = std::path::PathBuf::from(&start);
            if start_path.is_dir() {
                dialog = dialog.set_directory(&start_path);
            }
            if let Some(handle) = dialog.pick_folder().await {
                v.set(handle.path().display().to_string());
            }
        });
    };
    rsx! {
        div {
            class: "form-group",
            span { class: "form-label", "{label}" }
            div {
                class: "path-row",
                input {
                    class: "form-input path-input",
                    r#type: "text",
                    placeholder: "{placeholder}",
                    value: "{current}",
                    oninput: on_input,
                }
                button {
                    r#type: "button",
                    class: "button button-tertiary path-browse",
                    onclick: on_browse,
                    "Browse…"
                }
            }
        }
    }
}

#[component]
fn ModelBanner(signals: SignalBundle) -> Element {
    let ready = *signals.model_ready.read();
    match ready {
        Some(true) | None => rsx! { span {} },
        Some(false) => {
            let err = signals
                .error
                .read()
                .clone()
                .unwrap_or_else(|| "embedding model not installed".to_string());
            rsx! {
                div {
                    class: "banner banner-warn",
                    strong { "Model not installed." }
                    p { class: "small muted",
                        "Run "
                        code { "cargo xtask download-models" }
                        " to fetch the default bge-small-en-v1.5 embedding bundle (~35 MB). tidyup never downloads models itself."
                    }
                    pre { "{err}" }
                }
            }
        }
    }
}

#[component]
fn ErrorBanner(signals: SignalBundle) -> Element {
    let err = signals.error.read().clone();
    err.map_or_else(
        || rsx! { span {} },
        |msg| {
            let on_dismiss = move |_| {
                let mut e = signals.error;
                e.set(None);
            };
            rsx! {
                div {
                    class: "banner banner-error",
                    strong { "Error" }
                    pre { "{msg}" }
                    div {
                        class: "button-row",
                        style: "margin-top: 8px;",
                        button {
                            class: "button button-secondary",
                            onclick: on_dismiss,
                            "Dismiss"
                        }
                    }
                }
            }
        },
    )
}

#[component]
fn PhaseBanner(signals: SignalBundle) -> Element {
    let phase = *signals.phase.read();
    let busy = *signals.busy.read();

    let Some(phase) = phase else {
        return rsx! { span {} };
    };
    if busy == Busy::Idle {
        return rsx! { span {} };
    }

    let label = phase_label(phase);
    let current = *signals.progress_current.read();
    let total = *signals.progress_total.read();
    let item_label = signals.progress_label.read().clone();

    let percent = percent_u32(current, total);

    rsx! {
        div {
            class: "banner banner-info",
            div {
                style: "display: flex; align-items: center; gap: 12px;",
                span { class: "spinner" }
                strong { "{label}" }
                if let Some(t) = total {
                    span { class: "small", "{current} / {t}" }
                }
            }
            if !item_label.is_empty() {
                div { class: "small muted", style: "margin-top: 4px;", "{item_label}" }
            }
            if let Some(p) = percent {
                div {
                    class: "progress-bar",
                    div { class: "progress-bar-fill", style: "width: {p}%;" }
                }
            }
        }
    }
}

#[component]
fn LogPane(signals: SignalBundle) -> Element {
    let messages = signals.messages.read().clone();
    if messages.is_empty() {
        return rsx! { span {} };
    }
    rsx! {
        div {
            class: "card",
            h2 { class: "card-title", "Log" }
            div {
                class: "log",
                for (i, m) in messages.iter().enumerate() {
                    p {
                        key: "{i}",
                        class: log_class(m.level),
                        "{m.text}"
                    }
                }
            }
        }
    }
}

const fn log_class(level: Level) -> &'static str {
    match level {
        Level::Error => "log-line log-error",
        Level::Warn => "log-line log-warn",
        _ => "log-line",
    }
}

#[component]
fn LastReportCard(signals: SignalBundle) -> Element {
    let report = signals.last_report.read().clone();
    let Some(report) = report else {
        return rsx! { span {} };
    };

    match report {
        LastReport::Scan(r) => rsx! {
            ReportSummary {
                title: "Scan complete",
                run_id: r.run_id,
                proposed: r.proposed,
                applied: r.applied,
                bundles: r.bundles,
                bundles_applied: r.bundles_applied,
                skipped: r.skipped,
                failed: r.failed,
            }
        },
        LastReport::Migration(r) => rsx! {
            ReportSummary {
                title: "Migration complete",
                run_id: r.run_id,
                proposed: r.proposed,
                applied: r.applied,
                bundles: r.bundles,
                bundles_applied: r.bundles_applied,
                skipped: r.skipped,
                failed: r.failed,
            }
        },
        LastReport::Rollback(r) => rsx! {
            div {
                class: "card",
                h2 { class: "card-title", "Rollback complete" }
                div { class: "small muted", "Run {r.run_id}" }
                div {
                    class: "button-row",
                    style: "margin-top: 8px;",
                    span { class: "chip chip-high", "restored {r.restored} file(s)" }
                    span { class: "chip chip-high", "restored {r.bundles_restored} bundle(s)" }
                    if r.failures > 0 {
                        span { class: "chip chip-low", "{r.failures} failure(s)" }
                    }
                }
            }
        },
    }
}

#[component]
#[allow(clippy::too_many_arguments)]
fn ReportSummary(
    title: &'static str,
    run_id: Uuid,
    proposed: usize,
    applied: usize,
    bundles: usize,
    bundles_applied: usize,
    skipped: usize,
    failed: usize,
) -> Element {
    rsx! {
        div {
            class: "card",
            h2 { class: "card-title", "{title}" }
            div { class: "small muted", "Run {run_id}" }
            div {
                class: "button-row",
                style: "margin-top: 8px;",
                span { class: "chip chip-neutral", "{proposed} proposed" }
                span { class: "chip chip-high",    "{applied} applied" }
                if skipped > 0 { span { class: "chip chip-medium", "{skipped} skipped" } }
                if failed  > 0 { span { class: "chip chip-low",    "{failed} failed" } }
                if bundles > 0 {
                    span { class: "chip chip-neutral", "{bundles} bundle(s)" }
                    span { class: "chip chip-high",    "{bundles_applied} bundle(s) applied" }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Async actions. Each spawns a task that drives a service to completion.
// ---------------------------------------------------------------------------

fn launch_scan(state: &SharedState, source: PathBuf) {
    let signals = state.signals;
    let slot = state.review_slot.clone();

    reset_run_state(signals);
    set_busy(signals, Busy::Scanning);

    // `spawn_forever` (vs `spawn`): the reviewer flips `review_pending` mid-run,
    // which routes the app to `/review` and unmounts the calling page. A
    // scope-bound `spawn` would have its task cancelled there, which looks
    // exactly like "the migration never commences".
    spawn_forever(async move {
        let result = async {
            let cfg = config::load()?;
            let ctx = build(&cfg, true).await?;
            let candidates = build_default_scan_candidates(ctx.embeddings.as_ref()).await?;
            let reporter = DioxusReporter::new(signals);
            let reviewer = DioxusReviewHandler::new(signals, slot);

            let service = ScanService::new(Arc::clone(&ctx));
            let report = service
                .run(
                    ScanRequest {
                        root: source,
                        taxonomy_path: None,
                        dry_run: false,
                        auto_approve_bundles: false,
                        bundle_min_confidence: 0.85,
                    },
                    &candidates,
                    &reporter,
                    &reviewer,
                )
                .await?;
            anyhow::Ok(report)
        }
        .await;

        match result {
            Ok(r) => {
                let mut last = signals.last_report;
                last.set(Some(LastReport::Scan(r)));
            }
            Err(e) => {
                let mut err = signals.error;
                err.set(Some(format!("{e}")));
            }
        }
        set_busy(signals, Busy::Idle);
        let mut phase = signals.phase;
        phase.set(None);
        refresh_runs_inner(signals).await;
    });
}

fn launch_migrate(state: &SharedState, source: PathBuf, target: PathBuf) {
    let signals = state.signals;
    let slot = state.review_slot.clone();

    reset_run_state(signals);
    set_busy(signals, Busy::Migrating);

    spawn_forever(async move {
        let result = async {
            let cfg = config::load()?;
            let ctx = build(&cfg, true).await?;
            let reporter = DioxusReporter::new(signals);
            let reviewer = DioxusReviewHandler::new(signals, slot);

            let service = MigrationService::new(Arc::clone(&ctx));
            let report = service
                .run(
                    MigrationRequest {
                        source,
                        target,
                        dry_run: false,
                        auto_approve_bundles: false,
                        bundle_min_confidence: 0.85,
                    },
                    &reporter,
                    &reviewer,
                )
                .await?;
            anyhow::Ok(report)
        }
        .await;

        match result {
            Ok(r) => {
                let mut last = signals.last_report;
                last.set(Some(LastReport::Migration(r)));
            }
            Err(e) => {
                let mut err = signals.error;
                err.set(Some(format!("{e}")));
            }
        }
        set_busy(signals, Busy::Idle);
        let mut phase = signals.phase;
        phase.set(None);
        refresh_runs_inner(signals).await;
    });
}

fn launch_rollback(state: &SharedState, run_id: Uuid) {
    let signals = state.signals;

    reset_run_state(signals);
    set_busy(signals, Busy::RollingBack);

    spawn_forever(async move {
        let result = async {
            let cfg = config::load()?;
            let ctx = build(&cfg, false).await?;
            let reporter = DioxusReporter::new(signals);
            let service = RollbackService::new(Arc::clone(&ctx));
            let report = service.rollback_run(run_id, &reporter).await?;
            anyhow::Ok(report)
        }
        .await;

        match result {
            Ok(r) => {
                let mut last = signals.last_report;
                last.set(Some(LastReport::Rollback(r)));
            }
            Err(e) => {
                let mut err = signals.error;
                err.set(Some(format!("{e}")));
            }
        }
        set_busy(signals, Busy::Idle);
        let mut phase = signals.phase;
        phase.set(None);
        refresh_runs_inner(signals).await;
    });
}

fn submit_review(state: &SharedState) {
    let signals = state.signals;
    let slot = state.review_slot.clone();
    // Root scope so this still fires even if the Review page unmounts
    // between the click and the task running.
    spawn_forever(async move {
        let tx_opt = {
            let mut guard = slot.lock().await;
            guard.take()
        };
        let Some(tx) = tx_opt else {
            tracing::warn!("submit_review: no pending oneshot sender");
            return;
        };
        let decisions: Vec<_> = signals.decisions.read().values().cloned().collect();
        if tx.send(decisions).is_err() {
            tracing::warn!("submit_review: service receiver dropped before send");
        }
    });
}

fn reject_all_and_submit(state: &SharedState) {
    let signals = state.signals;
    let proposals = signals.proposals.read().clone();
    let mut decisions = signals.decisions;
    decisions.with_mut(|map| {
        map.clear();
        for p in &proposals {
            map.insert(p.id, tidyup_domain::ReviewDecision::Reject(p.id));
        }
    });
    submit_review(state);
}

fn refresh_runs(state: &SharedState) {
    let signals = state.signals;
    spawn_forever(async move {
        refresh_runs_inner(signals).await;
    });
}

async fn refresh_runs_inner(signals: SignalBundle) {
    let loaded = async {
        let cfg = config::load()?;
        let ctx = build(&cfg, false).await?;
        let service = RollbackService::new(Arc::clone(&ctx));
        service.list_runs().await
    }
    .await;

    match loaded {
        Ok(list) => {
            let mut runs = signals.runs;
            runs.set(list);
        }
        Err(e) => {
            let mut err = signals.error;
            err.set(Some(format!("{e}")));
        }
    }
}

fn set_busy(signals: SignalBundle, busy: Busy) {
    let mut b = signals.busy;
    b.set(busy);
}

fn reset_run_state(signals: SignalBundle) {
    let mut error = signals.error;
    let mut last = signals.last_report;
    let mut phase = signals.phase;
    let mut messages = signals.messages;
    error.set(None);
    last.set(None);
    phase.set(None);
    messages.set(Vec::new());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fn phase_label(phase: tidyup_domain::Phase) -> &'static str {
    match phase {
        tidyup_domain::Phase::Indexing => "Indexing",
        tidyup_domain::Phase::Extracting => "Extracting content",
        tidyup_domain::Phase::ProfilingTarget => "Profiling target hierarchy",
        tidyup_domain::Phase::Classifying => "Classifying",
        tidyup_domain::Phase::AwaitingReview => "Awaiting review",
        tidyup_domain::Phase::Applying => "Applying approved changes",
        tidyup_domain::Phase::Rollback => "Rolling back",
    }
}

fn confidence_chip(c: f32) -> (&'static str, String) {
    let pct = format!("{:.0}% confidence", c * 100.0);
    let cls = if c >= 0.85 {
        "chip-high"
    } else if c >= 0.6 {
        "chip-medium"
    } else {
        "chip-low"
    };
    (cls, pct)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn percent_u32(current: u64, total: Option<u64>) -> Option<u32> {
    let total = total.filter(|t| *t > 0)?;
    let pct = (current.min(total) as f64 / total as f64 * 100.0).round();
    Some(pct as u32)
}
