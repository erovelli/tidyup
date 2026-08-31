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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dioxus::core::spawn_forever;
use dioxus::prelude::*;
use tidyup_app::config;
use tidyup_app::{
    migration::MigrationRequest, scan::ScanRequest, MigrationService, RollbackService, ScanService,
};
use tidyup_core::frontend::Level;
use tidyup_core::ReviewOutcome;
use tidyup_domain::{
    BundleProposal, ChangeProposal, ChangeType, ReviewDecision, RunRecord, RunState,
};
use uuid::Uuid;

use crate::context::{
    build, build_audio_scan_candidates, build_default_scan_candidates, build_image_scan_candidates,
    quick_model_check, InferenceActivation,
};
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
    let scan_dry_run = use_signal(|| true);
    let migrate_dry_run = use_signal(|| true);

    let model_ok = signals.model_ready.read().unwrap_or(false);
    let busy = *signals.busy.read();
    let disabled = !model_ok || busy != Busy::Idle;

    let scan_disabled = disabled || source.read().trim().is_empty();
    let migrate_disabled =
        disabled || source.read().trim().is_empty() || target.read().trim().is_empty();

    let scan_state = state.clone();
    let on_scan = move |_| {
        let src = source.read().trim().to_string();
        let dry_run = *scan_dry_run.read();
        if src.is_empty() {
            return;
        }
        launch_scan(&scan_state, PathBuf::from(src), dry_run);
    };

    let migrate_state = state.clone();
    let on_migrate = move |_| {
        let src = source.read().trim().to_string();
        let tgt = target.read().trim().to_string();
        let dry_run = *migrate_dry_run.read();
        if src.is_empty() || tgt.is_empty() {
            return;
        }
        launch_migrate(
            &migrate_state,
            PathBuf::from(src),
            PathBuf::from(tgt),
            dry_run,
        );
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
                DryRunToggle { value: scan_dry_run }
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
                DryRunToggle { value: migrate_dry_run }
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

    let threshold = use_signal(|| 0.75_f32);
    let filter = use_signal(|| FilterMode::All);
    // Interactive diff state. Hover highlights both endpoints + the curve;
    // click on a row or curve "selects" the proposal and surfaces inline
    // approve/reject controls. Non-sync because only UI events write to them.
    let hovered = use_signal(|| Option::<Uuid>::None);
    let selected = use_signal(|| Option::<Uuid>::None);

    // The desktop reviews the complete plan in one pass so semantic collections
    // are never hidden behind an earlier loose-file review step.
    if !bundles.is_empty() {
        return rsx! {
            CombinedReview { state: state.clone() }
        };
    }

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

    let source_root = signals.review_source_root.read().clone();
    let target_root = signals.review_target_root.read().clone();
    let model = build_diff_model(
        &proposals,
        &[],
        source_root.as_deref(),
        target_root.as_deref(),
    );
    let applied_count = signals.last_report.read().as_ref().map_or(0, |r| match r {
        LastReport::Scan { report, .. } => report.applied,
        LastReport::Migration { report, .. } => report.applied,
        LastReport::Rollback(_) => 0,
    });
    let folder_count = count_folders(&model.left_rows);

    let filtered: Vec<ChangeProposal> = filter_proposals(&proposals, *filter.read());
    let indexed_count = usize::try_from(*signals.indexed_count.read()).unwrap_or(usize::MAX);

    rsx! {
        div {
            h1 { class: "page-title", "Review" }
            PhaseBanner { signals }

            SummaryCards {
                indexed: indexed_count,
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

            DiffView { model, hovered, selected, signals, locked_ids: Vec::new() }

            DiffLegend {}

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
    u32::try_from(n)
        .unwrap_or(u32::MAX / ROW_HEIGHT)
        .saturating_mul(ROW_HEIGHT)
        .saturating_add(DIFF_TOP_PAD.saturating_mul(2))
}

/// Vertical center of the row at `idx` inside the diff body.
fn row_center_y(idx: usize) -> u32 {
    DIFF_TOP_PAD
        .saturating_add(u32::try_from(idx).unwrap_or(0).saturating_mul(ROW_HEIGHT))
        .saturating_add(ROW_HEIGHT / 2)
}

#[derive(Clone, PartialEq)]
struct DiffModel {
    left_rows: Vec<TreeRow>,
    /// Proposal id → row index in `left_rows`.
    file_row_by_id: HashMap<Uuid, usize>,
    /// Source hierarchy shown in the right column.
    right_rows: Vec<CurrentTreeRow>,
    /// Proposal id → row index in `right_rows`.
    current_file_row_by_id: HashMap<Uuid, usize>,
    /// Per-file metadata used to draw connectors between both trees.
    connector_entries: Vec<RightEntry>,
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
        bundle_kind: Option<OverviewBundleKind>,
        file_count: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OverviewBundleKind {
    Directory,
    Collection,
}

#[derive(Clone)]
struct OverviewEntry {
    id: Uuid,
    proposed_path: PathBuf,
    proposed_name: String,
    confidence: f32,
    change_type: ChangeType,
    bundle_kind: Option<OverviewBundleKind>,
    file_count: usize,
    sources: Vec<OverviewSource>,
}

#[derive(Clone)]
struct OverviewSource {
    id: Uuid,
    original_path: PathBuf,
    confidence: f32,
    rename: bool,
    bundle_kind: Option<OverviewBundleKind>,
    file_count: usize,
}

#[derive(Clone, PartialEq)]
struct RightEntry {
    proposal_id: Uuid,
    original_path: PathBuf,
    display_name: String,
    confidence: f32,
    rename: bool,
    bundle_kind: Option<OverviewBundleKind>,
    file_count: usize,
}

#[derive(Clone, PartialEq)]
enum CurrentTreeRow {
    Folder {
        name: String,
        depth: usize,
        file_count: usize,
    },
    File {
        entry: RightEntry,
        depth: usize,
    },
}

fn build_diff_model(
    proposals: &[ChangeProposal],
    bundles: &[BundleProposal],
    explicit_source_root: Option<&Path>,
    explicit_target_root: Option<&Path>,
) -> DiffModel {
    let entries = overview_entries(proposals, bundles);
    let parents: Vec<PathBuf> = entries
        .iter()
        .map(|p| {
            p.proposed_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default()
        })
        .collect();
    let original_parents: Vec<PathBuf> = entries
        .iter()
        .flat_map(|entry| entry.sources.iter())
        .filter_map(|source| source.original_path.parent().map(Path::to_path_buf))
        .collect();
    let source_root =
        explicit_source_root.map_or_else(|| common_ancestor(&original_parents), Path::to_path_buf);
    let proposed_root = common_ancestor(&parents);
    // Scan mode reorganizes in place, so the common source root is the most
    // useful anchor: it keeps the complete destination (`Work/Career`) visible.
    // Migration can target an unrelated hierarchy; in that case retain the
    // proposed common ancestor, backing up one level when every item lands in
    // exactly the same directory so the destination never disappears.
    let root = explicit_target_root.map_or_else(
        || {
            if explicit_source_root.is_some()
                || (!source_root.as_os_str().is_empty()
                    && parents
                        .iter()
                        .all(|parent| parent.starts_with(&source_root)))
            {
                source_root.clone()
            } else if !proposed_root.as_os_str().is_empty()
                && parents.iter().all(|parent| parent == &proposed_root)
            {
                proposed_root
                    .parent()
                    .map_or_else(|| proposed_root.clone(), Path::to_path_buf)
            } else {
                proposed_root
            }
        },
        Path::to_path_buf,
    );
    // Build tree keyed by relative path components.
    let mut tree = TreeBuilder::default();
    for (idx, p) in entries.iter().enumerate() {
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
        tree.insert(&rel, filename, idx, p.file_count);
    }

    let total_file_count = overview_file_count(proposals, bundles);
    let root_label = path_display_name(&root);
    let mut left_rows = vec![TreeRow::Folder {
        name: root_label,
        depth: 0,
        file_count: total_file_count,
    }];
    let mut file_row_by_id = HashMap::new();
    tree.flatten(1, &entries, &mut left_rows, &mut file_row_by_id);
    map_source_rows_to_targets(&entries, &mut file_row_by_id);
    let connector_entries = build_connector_entries(&entries);

    let (right_rows, current_file_row_by_id) = build_current_tree(
        &source_root,
        &connector_entries,
        &file_row_by_id,
        total_file_count,
    );

    DiffModel {
        left_rows,
        file_row_by_id,
        right_rows,
        current_file_row_by_id,
        connector_entries,
    }
}

fn map_source_rows_to_targets(
    entries: &[OverviewEntry],
    destination_row_by_id: &mut HashMap<Uuid, usize>,
) {
    for entry in entries {
        let Some(&target_row) = destination_row_by_id.get(&entry.id) else {
            continue;
        };
        for source in &entry.sources {
            destination_row_by_id.insert(source.id, target_row);
        }
    }
}

fn build_connector_entries(entries: &[OverviewEntry]) -> Vec<RightEntry> {
    entries
        .iter()
        .flat_map(|entry| entry.sources.iter())
        .map(|source| RightEntry {
            proposal_id: source.id,
            original_path: source.original_path.clone(),
            display_name: source
                .original_path
                .file_name()
                .and_then(|name| name.to_str())
                .map_or_else(
                    || source.original_path.display().to_string(),
                    ToString::to_string,
                ),
            confidence: source.confidence,
            rename: source.rename,
            bundle_kind: source.bundle_kind,
            file_count: source.file_count,
        })
        .collect()
}

/// Preserve the original source hierarchy instead of flattening every file
/// into a filename-only list. The explicit root row makes it clear which files
/// sit directly in the scan root and which live inside containers.
fn build_current_tree(
    source_root: &Path,
    entries: &[RightEntry],
    destination_row_by_id: &HashMap<Uuid, usize>,
    total_file_count: usize,
) -> (Vec<CurrentTreeRow>, HashMap<Uuid, usize>) {
    let mut current_tree = TreeBuilder::default();
    for (idx, entry) in entries.iter().enumerate() {
        let parent = entry
            .original_path
            .parent()
            .unwrap_or_else(|| Path::new(""));
        let rel: Vec<String> = parent
            .strip_prefix(source_root)
            .unwrap_or(parent)
            .components()
            .filter_map(|component| component.as_os_str().to_str().map(ToString::to_string))
            .collect();
        let filename = entry
            .original_path
            .file_name()
            .and_then(|name| name.to_str())
            .map_or_else(
                || entry.original_path.display().to_string(),
                ToString::to_string,
            );
        current_tree.insert(&rel, filename, idx, entry.file_count);
    }
    let source_root_label = path_display_name(source_root);
    let mut right_rows = vec![CurrentTreeRow::Folder {
        name: source_root_label,
        depth: 0,
        file_count: total_file_count,
    }];
    let mut current_file_row_by_id = HashMap::new();
    current_tree.flatten_current(
        1,
        entries,
        destination_row_by_id,
        &mut right_rows,
        &mut current_file_row_by_id,
    );
    (right_rows, current_file_row_by_id)
}

fn overview_entries(
    proposals: &[ChangeProposal],
    bundles: &[BundleProposal],
) -> Vec<OverviewEntry> {
    let mut entries = proposals
        .iter()
        .map(|proposal| OverviewEntry {
            id: proposal.id,
            proposed_path: proposal.proposed_path.clone(),
            proposed_name: proposal.proposed_name.clone(),
            confidence: proposal.confidence,
            change_type: proposal.change_type.clone(),
            bundle_kind: None,
            file_count: 1,
            sources: vec![OverviewSource {
                id: proposal.id,
                original_path: proposal.original_path.clone(),
                confidence: proposal.confidence,
                rename: matches!(
                    proposal.change_type,
                    ChangeType::Rename | ChangeType::RenameAndMove
                ),
                bundle_kind: None,
                file_count: 1,
            }],
        })
        .collect::<Vec<_>>();
    entries.extend(bundles.iter().filter_map(collapsed_bundle_entry));
    entries
}

fn collapsed_bundle_entry(bundle: &BundleProposal) -> Option<OverviewEntry> {
    let file_count = bundle.members.len();
    let label = match &bundle.kind {
        tidyup_domain::BundleKind::SemanticCollection { label } => label.clone(),
        _ => path_display_name(&bundle.root),
    };
    let (proposed_path, bundle_kind) = if bundle.kind.moves_as_file_set() {
        let proposed_parents = bundle
            .members
            .iter()
            .filter_map(|member| member.proposed_path.parent().map(Path::to_path_buf))
            .collect::<Vec<_>>();
        let proposed_path = common_ancestor(&proposed_parents);
        (proposed_path, OverviewBundleKind::Collection)
    } else {
        let leaf = bundle.root.file_name()?;
        (
            bundle.target_parent.join(leaf),
            OverviewBundleKind::Directory,
        )
    };

    let moves = if bundle.kind.moves_as_file_set() {
        bundle
            .members
            .iter()
            .any(|member| member.original_path != member.proposed_path)
    } else {
        bundle.root != proposed_path
    };
    let sources = if bundle.kind.moves_as_file_set() {
        bundle
            .members
            .iter()
            .map(|member| OverviewSource {
                id: member.id,
                original_path: member.original_path.clone(),
                confidence: member.confidence,
                rename: matches!(
                    member.change_type,
                    ChangeType::Rename | ChangeType::RenameAndMove
                ),
                bundle_kind: None,
                file_count: 1,
            })
            .collect()
    } else {
        vec![OverviewSource {
            id: bundle.id,
            original_path: bundle.root.clone(),
            confidence: bundle.confidence,
            rename: false,
            bundle_kind: Some(OverviewBundleKind::Directory),
            file_count,
        }]
    };
    moves.then_some(OverviewEntry {
        id: bundle.id,
        proposed_path,
        proposed_name: label,
        confidence: bundle.confidence,
        change_type: ChangeType::Move,
        bundle_kind: Some(bundle_kind),
        file_count,
        sources,
    })
}

fn overview_file_count(proposals: &[ChangeProposal], bundles: &[BundleProposal]) -> usize {
    bundles.iter().fold(proposals.len(), |count, bundle| {
        count.saturating_add(bundle.members.len())
    })
}

fn overview_locked_ids(bundles: &[BundleProposal]) -> Vec<Uuid> {
    bundles
        .iter()
        .flat_map(|bundle| {
            if bundle.kind.moves_as_file_set() {
                bundle
                    .members
                    .iter()
                    .map(|member| member.id)
                    .collect::<Vec<_>>()
            } else {
                vec![bundle.id]
            }
        })
        .collect()
}

fn path_display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| path.display().to_string(), ToString::to_string)
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
    folders: BTreeMap<String, Self>,
    files: Vec<(String, usize, usize)>, // (name, overview-entry index, represented files)
}

enum CurrentChild<'a> {
    Folder(&'a str, &'a TreeBuilder),
    File(&'a str, usize),
}

impl TreeBuilder {
    fn insert(&mut self, folders: &[String], filename: String, idx: usize, file_count: usize) {
        let mut node = self;
        for folder in folders {
            node = node.folders.entry(folder.clone()).or_default();
        }
        node.files.push((filename, idx, file_count));
    }

    fn file_count(&self) -> usize {
        self.folders.values().fold(
            self.files
                .iter()
                .fold(0_usize, |count, (_, _, files)| count.saturating_add(*files)),
            |count, folder| count.saturating_add(folder.file_count()),
        )
    }

    fn flatten(
        &self,
        depth: usize,
        proposals: &[OverviewEntry],
        rows: &mut Vec<TreeRow>,
        file_row_by_id: &mut HashMap<Uuid, usize>,
    ) {
        for (name, child) in &self.folders {
            rows.push(TreeRow::Folder {
                name: name.clone(),
                depth,
                file_count: child.file_count(),
            });
            child.flatten(depth.saturating_add(1), proposals, rows, file_row_by_id);
        }
        let mut files = self.files.clone();
        files.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, idx, _) in files {
            let Some(p) = proposals.get(idx) else {
                continue;
            };
            let row_idx = rows.len();
            rows.push(TreeRow::File {
                name,
                depth,
                confidence: p.confidence,
                change_type: p.change_type.clone(),
                proposal_id: p.id,
                bundle_kind: p.bundle_kind,
                file_count: p.file_count,
            });
            file_row_by_id.insert(p.id, row_idx);
        }
    }

    fn flatten_current(
        &self,
        depth: usize,
        entries: &[RightEntry],
        destination_row_by_id: &HashMap<Uuid, usize>,
        rows: &mut Vec<CurrentTreeRow>,
        file_row_by_id: &mut HashMap<Uuid, usize>,
    ) {
        let mut children: Vec<CurrentChild<'_>> = self
            .folders
            .iter()
            .map(|(name, child)| CurrentChild::Folder(name, child))
            .chain(
                self.files
                    .iter()
                    .map(|(name, idx, _)| CurrentChild::File(name, *idx)),
            )
            .collect();
        children.sort_by(|a, b| {
            a.destination_rank(entries, destination_row_by_id)
                .cmp(&b.destination_rank(entries, destination_row_by_id))
                .then_with(|| a.name().cmp(b.name()))
        });

        for child in children {
            match child {
                CurrentChild::Folder(name, child) => {
                    rows.push(CurrentTreeRow::Folder {
                        name: name.to_string(),
                        depth,
                        file_count: child.file_count(),
                    });
                    child.flatten_current(
                        depth.saturating_add(1),
                        entries,
                        destination_row_by_id,
                        rows,
                        file_row_by_id,
                    );
                }
                CurrentChild::File(_, idx) => {
                    let Some(entry) = entries.get(idx).cloned() else {
                        continue;
                    };
                    file_row_by_id.insert(entry.proposal_id, rows.len());
                    rows.push(CurrentTreeRow::File { entry, depth });
                }
            }
        }
    }

    fn minimum_destination_rank(
        &self,
        entries: &[RightEntry],
        destination_row_by_id: &HashMap<Uuid, usize>,
    ) -> usize {
        self.files
            .iter()
            .filter_map(|(_, idx, _)| entries.get(*idx))
            .filter_map(|entry| destination_row_by_id.get(&entry.proposal_id).copied())
            .chain(
                self.folders
                    .values()
                    .map(|folder| folder.minimum_destination_rank(entries, destination_row_by_id)),
            )
            .min()
            .unwrap_or(usize::MAX)
    }
}

impl CurrentChild<'_> {
    const fn name(&self) -> &str {
        match self {
            Self::Folder(name, _) | Self::File(name, _) => name,
        }
    }

    fn destination_rank(
        &self,
        entries: &[RightEntry],
        destination_row_by_id: &HashMap<Uuid, usize>,
    ) -> usize {
        match self {
            Self::Folder(_, folder) => {
                folder.minimum_destination_rank(entries, destination_row_by_id)
            }
            Self::File(_, idx) => entries
                .get(*idx)
                .and_then(|entry| destination_row_by_id.get(&entry.proposal_id))
                .copied()
                .unwrap_or(usize::MAX),
        }
    }
}

fn count_folders(rows: &[TreeRow]) -> usize {
    rows.iter()
        .filter(|row| matches!(row, TreeRow::Folder { .. }))
        .count()
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
    threshold: Signal<f32>,
    pending: bool,
    state: SharedState,
) -> Element {
    let threshold_val = *threshold.read();
    let threshold_label = format!("{threshold_val:.2}");
    let on_input = move |ev: Event<FormData>| {
        let mut t = threshold;
        if let Ok(n) = ev.value().parse::<f32>() {
            t.set(n.clamp(0.0, 1.0));
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
                        max: "1",
                        step: "0.01",
                        value: "{threshold_label}",
                        oninput: on_input,
                        class: "threshold-number",
                    }
                    span { class: "threshold-unit", "similarity" }
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

fn execute_with_threshold(state: &SharedState, threshold: f32) {
    let signals = state.signals;
    let proposals = signals.proposals.read().clone();
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

fn execute_combined_with_threshold(state: &SharedState, threshold: f32) {
    let signals = state.signals;
    let proposals = signals.proposals.read().clone();
    let mut decisions = signals.decisions;
    decisions.with_mut(|map| {
        for proposal in &proposals {
            if map.contains_key(&proposal.id) {
                continue;
            }
            let is_rename = matches!(
                proposal.change_type,
                ChangeType::Rename | ChangeType::RenameAndMove
            );
            let decision = if !is_rename && proposal.confidence >= threshold {
                ReviewDecision::Approve(proposal.id)
            } else {
                ReviewDecision::Reject(proposal.id)
            };
            map.insert(proposal.id, decision);
        }
    });
    submit_combined_review(state);
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

fn approve_proposal(signals: SignalBundle, proposal_id: Uuid) {
    let decision = signals
        .proposals
        .read()
        .iter()
        .find(|proposal| proposal.id == proposal_id)
        .map_or(ReviewDecision::Approve(proposal_id), approval_decision);
    let mut decisions = signals.decisions;
    decisions.with_mut(|items| {
        items.insert(proposal_id, decision);
    });
}

fn approval_decision(proposal: &ChangeProposal) -> ReviewDecision {
    if matches!(
        proposal.change_type,
        ChangeType::Rename | ChangeType::RenameAndMove
    ) {
        ReviewDecision::Override {
            proposal_id: proposal.id,
            new_target: proposal.proposed_path.clone(),
        }
    } else {
        ReviewDecision::Approve(proposal.id)
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
    locked_ids: Vec<Uuid>,
) -> Element {
    let locked_ids: HashSet<Uuid> = locked_ids.into_iter().collect();
    let left_height = rows_to_height(model.left_rows.len());
    let right_height = rows_to_height(model.right_rows.len());
    let svg_height = left_height.max(right_height);

    let connectors: Vec<ConnectorRender> = model
        .connector_entries
        .iter()
        .filter_map(|entry| {
            let &left_idx = model.file_row_by_id.get(&entry.proposal_id)?;
            let &right_idx = model.current_file_row_by_id.get(&entry.proposal_id)?;
            let ly = row_center_y(left_idx);
            let ry = row_center_y(right_idx);
            let d = format!("M 0 {ly} C 50 {ly}, 50 {ry}, 100 {ry}");
            Some(ConnectorRender {
                id: entry.proposal_id,
                d,
                strong: entry.confidence >= 0.80,
                dashed: entry.rename,
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
                    for (i, row) in model.right_rows.iter().cloned().enumerate() {
                        CurrentTreeRowView {
                            key: "{current_tree_row_key(i, &row)}",
                            row,
                            locked_ids: locked_ids.clone(),
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

fn current_tree_row_key(i: usize, row: &CurrentTreeRow) -> String {
    match row {
        CurrentTreeRow::File { entry, .. } => format!("current-file-{}", entry.proposal_id),
        CurrentTreeRow::Folder { name, depth, .. } => {
            format!("current-folder-{i}-{depth}-{name}")
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
            let pad = depth.saturating_mul(24).saturating_add(12);
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
            bundle_kind,
            file_count,
        } => {
            let pad = depth.saturating_mul(24).saturating_add(12);
            let chip = confidence_chip(confidence);
            let is_rename = matches!(change_type, ChangeType::Rename | ChangeType::RenameAndMove);

            let hovered_id = *hovered.read();
            let selected_id = *selected.read();
            let is_hovered = hovered_id == Some(proposal_id);
            let is_selected = selected_id == Some(proposal_id);
            let state = decision_state_of(&signals.decisions.read(), proposal_id);

            let mut row_class = if bundle_kind.is_some() {
                String::from("tree-row tree-folder tree-bundle")
            } else {
                String::from("tree-row tree-file")
            };
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
                    if bundle_kind.is_some() {
                        span { class: "tree-caret", "▸" }
                        span { class: "tree-folder-name", title: "{name}", "{name}/" }
                        span {
                            class: "tree-filecount",
                            if file_count == 1 { "1 file" } else { "{file_count} files" }
                        }
                    } else {
                        span { class: "tree-file-name", title: "{name}", "{name}" }
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
}

#[component]
fn CurrentRow(
    entry: RightEntry,
    depth: usize,
    hovered: Signal<Option<Uuid>>,
    selected: Signal<Option<Uuid>>,
    signals: SignalBundle,
    locked: bool,
) -> Element {
    let hovered_id = *hovered.read();
    let selected_id = *selected.read();
    let is_hovered = hovered_id == Some(entry.proposal_id);
    let is_selected = selected_id == Some(entry.proposal_id);
    let state = decision_state_of(&signals.decisions.read(), entry.proposal_id);

    let mut row_class = if entry.bundle_kind.is_some() {
        String::from("current-row current-bundle")
    } else {
        String::from("current-row")
    };
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
        approve_proposal(signals, pid);
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
            style: "padding-left: {depth.saturating_mul(24).saturating_add(12)}px;",
            onmouseenter: on_enter,
            onmouseleave: on_leave,
            onclick: on_click,
            if let Some(bundle_kind) = entry.bundle_kind {
                span { class: "tree-caret", "▸" }
                span {
                    class: "current-name",
                    title: "{entry.display_name}",
                    if bundle_kind == OverviewBundleKind::Directory {
                        "{entry.display_name}/"
                    } else {
                        "{entry.display_name} collection"
                    }
                }
                span {
                    class: "tree-filecount",
                    if entry.file_count == 1 { "1 file" } else { "{entry.file_count} files" }
                }
            } else {
                span { class: "current-name", title: "{entry.display_name}", "{entry.display_name}" }
            }
            if is_selected && !locked {
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
            if is_selected && locked {
                span { class: "chip chip-neutral", "approve with collection" }
            }
        }
    }
}

#[component]
fn CurrentTreeRowView(
    row: CurrentTreeRow,
    hovered: Signal<Option<Uuid>>,
    selected: Signal<Option<Uuid>>,
    signals: SignalBundle,
    locked_ids: HashSet<Uuid>,
) -> Element {
    match row {
        CurrentTreeRow::Folder {
            name,
            depth,
            file_count,
        } => {
            let pad = depth.saturating_mul(24).saturating_add(12);
            rsx! {
                div {
                    class: "tree-row tree-folder current-folder",
                    style: "padding-left: {pad}px;",
                    span { class: "tree-caret", "▸" }
                    span { class: "tree-folder-name", title: "{name}", "{name}/" }
                    span {
                        class: "tree-filecount",
                        if file_count == 1 { "1 file" } else { "{file_count} files" }
                    }
                }
            }
        }
        CurrentTreeRow::File { entry, depth } => {
            let locked = locked_ids.contains(&entry.proposal_id);
            rsx! {
                CurrentRow { entry, depth, hovered, selected, signals, locked }
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
            FilterPill { label: "High ≥0.80",    count: high_n,   active: current == FilterMode::High,        onclick: set(FilterMode::High) }
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
    let destination = proposal
        .proposed_path
        .parent()
        .map_or_else(String::new, |path| path.display().to_string());
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
        approve_proposal(signals, proposal_id);
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
                    title: "{from}",
                    "{proposal.original_path.file_name().and_then(|name| name.to_str()).unwrap_or(\"File\")}"
                }
                div {
                    class: "proposal-path",
                    "{from}"
                    span { class: "proposal-arrow", " → " }
                    "{destination}/ (1 file)"
                }
                div {
                    class: "proposal-reason",
                    "{proposal.reasoning}"
                }
                if is_rename {
                    ProposalRenameEditor { proposal: proposal.clone(), signals }
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
fn ProposalRenameEditor(proposal: ChangeProposal, signals: SignalBundle) -> Element {
    let mut validation_error = use_signal(|| None::<String>);
    let proposal_id = proposal.id;
    let proposed_name = proposal.proposed_name.clone();
    let original_name = proposal
        .original_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    let on_input = move |event: Event<FormData>| {
        validation_error.set(
            update_loose_proposal_name(signals, proposal_id, &event.value())
                .err()
                .map(str::to_string),
        );
    };
    let validation_message = validation_error.read().clone();
    rsx! {
        div {
            class: "semantic-members",
            style: "margin-top: 12px; display: grid; gap: 8px;",
            FilenameRenameEditor {
                original: original_name.clone(),
                proposed: proposed_name,
                validation_message,
                aria_label: format!("Proposed filename for {original_name}"),
                oninput: on_input,
            }
        }
    }
}

fn update_loose_proposal_name(
    signals: SignalBundle,
    proposal_id: Uuid,
    raw_name: &str,
) -> Result<(), &'static str> {
    let proposals = signals.proposals.read();
    let name = validate_loose_proposal_name(&proposals, proposal_id, raw_name)?;
    drop(proposals);

    let mut updated_target = None;
    let mut proposals = signals.proposals;
    proposals.with_mut(|items| {
        let Some(proposal) = items.iter_mut().find(|proposal| proposal.id == proposal_id) else {
            return;
        };
        proposal.proposed_name.clone_from(&name);
        proposal.proposed_path.set_file_name(&name);
        proposal.change_type =
            change_type_for_paths(&proposal.original_path, &proposal.proposed_path);
        updated_target = Some(proposal.proposed_path.clone());
    });

    if let Some(new_target) = updated_target {
        let mut decisions = signals.decisions;
        decisions.with_mut(|items| {
            if matches!(
                items.get(&proposal_id),
                Some(ReviewDecision::Approve(_) | ReviewDecision::Override { .. })
            ) {
                items.insert(
                    proposal_id,
                    ReviewDecision::Override {
                        proposal_id,
                        new_target,
                    },
                );
            }
        });
    }
    Ok(())
}

fn validate_loose_proposal_name(
    proposals: &[ChangeProposal],
    proposal_id: Uuid,
    raw_name: &str,
) -> Result<String, &'static str> {
    let proposal = proposals
        .iter()
        .find(|proposal| proposal.id == proposal_id)
        .ok_or("This proposal is no longer available.")?;
    let name = raw_name.trim();
    if name.is_empty() {
        return Err("Filename cannot be empty.");
    }
    if name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return Err("Enter a filename, not a path.");
    }

    let expected_extension = Path::new(&proposal.proposed_name)
        .extension()
        .and_then(|extension| extension.to_str());
    let supplied_extension = Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str());
    if !extensions_match(expected_extension, supplied_extension) {
        return Err("Filename extension must stay the same.");
    }

    let destination_parent = proposal.proposed_path.parent();
    if proposals.iter().any(|other| {
        other.id != proposal_id
            && other.proposed_path.parent() == destination_parent
            && other.proposed_name.eq_ignore_ascii_case(name)
    }) {
        return Err("Another proposal in this folder already uses that filename.");
    }
    Ok(name.to_string())
}

const fn extensions_match(expected: Option<&str>, supplied: Option<&str>) -> bool {
    match (expected, supplied) {
        (Some(expected), Some(supplied)) => expected.eq_ignore_ascii_case(supplied),
        (None, None) => true,
        _ => false,
    }
}

fn change_type_for_paths(original: &Path, proposed: &Path) -> ChangeType {
    let renamed = original.file_name() != proposed.file_name();
    let moved = original.parent() != proposed.parent();
    match (renamed, moved) {
        (true, true) => ChangeType::RenameAndMove,
        (true, false) => ChangeType::Rename,
        _ => ChangeType::Move,
    }
}

/// Unified review surface for individual changes and atomic bundles.
#[component]
fn CombinedReview(state: SharedState) -> Element {
    let signals = state.signals;
    let proposals = signals.proposals.read().clone();
    let bundles = signals.bundles.read().clone();
    let approvals = signals.bundle_approvals.read().clone();
    let threshold = use_signal(|| 0.75_f32);
    let threshold_val = *threshold.read();
    let threshold_label = format!("{threshold_val:.2}");
    let loose_count = proposals.len();
    let collection_count = bundles.len();
    let envelope_count = bundles
        .iter()
        .filter(|bundle| matches!(&bundle.kind, tidyup_domain::BundleKind::DirectoryEnvelope))
        .count();
    let file_set_count = collection_count.saturating_sub(envelope_count);
    let collection_files = bundles
        .iter()
        .map(|bundle| bundle.members.len())
        .sum::<usize>();
    let total_files = loose_count.saturating_add(collection_files);
    let indexed_count = usize::try_from(*signals.indexed_count.read()).unwrap_or(usize::MAX);
    let approved_n = approvals.values().filter(|v| **v).count();
    let locked_ids = overview_locked_ids(&bundles);
    let source_root = signals.review_source_root.read().clone();
    let target_root = signals.review_target_root.read().clone();
    let model = build_diff_model(
        &proposals,
        &bundles,
        source_root.as_deref(),
        target_root.as_deref(),
    );
    let hovered = use_signal(|| Option::<Uuid>::None);
    let selected = use_signal(|| Option::<Uuid>::None);

    let on_apply = {
        let state = state.clone();
        move |_| execute_combined_with_threshold(&state, threshold_val)
    };
    let on_reject_all = {
        let state = state.clone();
        move |_| reject_all_combined_and_submit(&state)
    };
    let on_threshold = move |event: Event<FormData>| {
        let mut value = threshold;
        if let Ok(number) = event.value().parse::<f32>() {
            value.set(number.clamp(0.0, 1.0));
        }
    };

    rsx! {
        div {
            h1 { class: "page-title", "Review" }
            PhaseBanner { signals }
            SummaryCards { indexed: indexed_count, pending: total_files, applied: 0 }
            div {
                class: "card",
                h2 { class: "card-title", "Complete organization plan" }
                p {
                    class: "card-subtitle muted",
                    "{loose_count} individual change(s), {envelope_count} preserved directory envelope(s), and {file_set_count} file-set collection(s), covering {total_files} files. Existing folders move as one unit; anything undecided is held."
                }
                div {
                    class: "button-row",
                    style: "margin-top: 12px;",
                    div {
                        class: "threshold-input",
                        span { class: "threshold-label", "Approve individual moves ≥" }
                        input {
                            r#type: "number",
                            min: "0",
                            max: "1",
                            step: "0.01",
                            value: "{threshold_label}",
                            oninput: on_threshold,
                            class: "threshold-number",
                        }
                        span { class: "threshold-unit", "similarity" }
                    }
                    button {
                        class: "button button-primary",
                        onclick: on_apply,
                        "Execute ({approved_n}/{collection_count} collections approved)"
                    }
                    button {
                        class: "button button-secondary",
                        onclick: on_reject_all,
                        "Reject all"
                    }
                }
            }
            div { class: "section-heading", style: "margin-top: 24px;", "PLAN OVERVIEW" }
            DiffView { model, hovered, selected, signals, locked_ids }
            DiffLegend {}
            div { class: "section-heading", style: "margin-top: 24px;", "CHANGES" }
            div {
                class: "card-stack",
                style: "margin-top: 16px;",
                for b in bundles.iter().cloned() {
                    BundleReviewCard { key: "{b.id}", bundle: b, signals }
                }
                for proposal in proposals.iter().cloned() {
                    ProposalCard { key: "{proposal.id}", proposal, signals }
                }
            }
        }
    }
}

/// One reviewable bundle: atomic approve/reject, no per-member decision.
#[component]
fn BundleReviewCard(bundle: BundleProposal, signals: SignalBundle) -> Element {
    let approvals = signals.bundle_approvals;
    let decision = approvals.read().get(&bundle.id).copied();

    let root = bundle.root.display().to_string();
    let target = if matches!(&bundle.kind, tidyup_domain::BundleKind::DirectoryEnvelope) {
        bundle.target_parent.display().to_string()
    } else {
        let target_parents: Vec<PathBuf> = bundle
            .members
            .iter()
            .filter_map(|member| member.proposed_path.parent().map(Path::to_path_buf))
            .collect();
        common_ancestor(&target_parents).display().to_string()
    };
    let kind = bundle.kind.as_str();
    let title = match &bundle.kind {
        tidyup_domain::BundleKind::SemanticCollection { label } => label.clone(),
        _ => bundle
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Bundle")
            .to_string(),
    };
    let chip = confidence_chip(bundle.confidence);
    let member_count = bundle.members.len();
    let envelope_detail = bundle.envelope.as_ref().map(|envelope| {
        let state = match envelope.boundary {
            tidyup_domain::DirectoryBoundary::Cohesive => "cohesive",
            tidyup_domain::DirectoryBoundary::Uncertain => "uncertain — review required",
        };
        let provenance = if envelope.provenance.is_empty() {
            String::new()
        } else {
            format!("; provenance: {}", envelope.provenance.join(", "))
        };
        format!(
            "Existing folder preserved as one unit ({state}); {} files, {} directories, {} symlinks; cohesion {:.2}{provenance}",
            envelope.snapshot.regular_files,
            envelope.snapshot.directories,
            envelope.snapshot.symlinks,
            envelope.cohesion,
        )
    });

    let card_class = match decision {
        Some(true) => "proposal proposal-approved",
        Some(false) => "proposal proposal-rejected",
        None => "proposal",
    };

    let bundle_id = bundle.id;
    let on_approve = move |_| {
        let mut a = approvals;
        a.with_mut(|map| {
            map.insert(bundle_id, true);
        });
    };
    let on_reject = move |_| {
        let mut a = approvals;
        a.with_mut(|map| {
            map.insert(bundle_id, false);
        });
    };

    let approve_class = if decision == Some(true) {
        "mini-button mini-button-approve active"
    } else {
        "mini-button mini-button-approve"
    };
    let reject_class = if decision == Some(false) {
        "mini-button mini-button-reject active"
    } else {
        "mini-button mini-button-reject"
    };

    rsx! {
        div {
            class: "{card_class}",
            div {
                class: "proposal-meta",
                div { class: "proposal-target", "{title}" }
                div { class: "proposal-path", "{root} → {target}/ ({member_count} files)" }
                div { class: "proposal-reason", "{bundle.reasoning}" }
                if let Some(detail) = envelope_detail {
                    div { class: "proposal-reason", "{detail}" }
                }
                if bundle.envelope.is_some() && !bundle.members.is_empty() {
                    details {
                        class: "semantic-members",
                        style: "margin-top: 10px;",
                        summary { "Show preserved descendants (read-only)" }
                        ul {
                            for member in bundle.members.iter().take(64) {
                                li { "{member.original_path.display()}" }
                            }
                        }
                    }
                }
                if bundle.kind.allows_member_renames() {
                    div {
                        class: "semantic-members",
                        style: "margin-top: 12px; display: grid; gap: 8px;",
                        for member in bundle.members.iter().cloned() {
                            SemanticMemberEditor {
                                key: "{member.id}",
                                bundle_id,
                                member,
                                signals,
                            }
                        }
                    }
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
                button { class: "{approve_class}", onclick: on_approve, "Approve" }
                button { class: "{reject_class}", onclick: on_reject, "Reject" }
            }
        }
    }
}

#[component]
fn SemanticMemberEditor(bundle_id: Uuid, member: ChangeProposal, signals: SignalBundle) -> Element {
    let mut validation_error = use_signal(|| None::<String>);
    let member_id = member.id;
    let original = member
        .original_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    let proposed = member.proposed_name.clone();
    let on_input = move |event: Event<FormData>| {
        validation_error.set(
            update_semantic_member_name(signals, bundle_id, member_id, &event.value())
                .err()
                .map(str::to_string),
        );
    };
    let validation_message = validation_error.read().clone();
    rsx! {
        FilenameRenameEditor {
            original: original.clone(),
            proposed,
            validation_message,
            aria_label: format!("Proposed filename for {original}"),
            oninput: on_input,
        }
    }
}

#[component]
fn FilenameRenameEditor(
    original: String,
    proposed: String,
    validation_message: Option<String>,
    aria_label: String,
    oninput: EventHandler<Event<FormData>>,
) -> Element {
    let invalid = validation_message.is_some();
    let input_class = if invalid {
        "path-input input-error"
    } else {
        "path-input"
    };
    rsx! {
        label {
            class: "small muted",
            style: "display: grid; grid-template-columns: minmax(180px, 1fr) 18px minmax(220px, 1fr); align-items: center; gap: 8px;",
            span { title: "{original}", "{original}" }
            span { "→" }
            input {
                r#type: "text",
                class: "{input_class}",
                value: "{proposed}",
                title: "{proposed}",
                oninput: move |event| oninput.call(event),
                aria_label,
                aria_invalid: invalid,
            }
            if let Some(ref message) = validation_message {
                span {
                    class: "field-error",
                    style: "grid-column: 3;",
                    "{message}"
                }
            }
        }
    }
}

fn update_semantic_member_name(
    signals: SignalBundle,
    bundle_id: Uuid,
    member_id: Uuid,
    raw_name: &str,
) -> Result<(), &'static str> {
    let bundles = signals.bundles;
    let items = bundles.read();
    let name = {
        let bundle = items
            .iter()
            .find(|bundle| bundle.id == bundle_id)
            .ok_or("This collection is no longer available.")?;
        validate_semantic_member_name(bundle, member_id, raw_name)?
    };
    drop(items);
    let mut bundles = bundles;
    bundles.with_mut(|items| {
        let Some(bundle) = items.iter_mut().find(|bundle| bundle.id == bundle_id) else {
            return;
        };
        let Some(member) = bundle
            .members
            .iter_mut()
            .find(|member| member.id == member_id)
        else {
            return;
        };
        member.proposed_name.clone_from(&name);
        member.proposed_path.set_file_name(&name);
        let original_name = member
            .original_path
            .file_name()
            .and_then(|original| original.to_str())
            .unwrap_or_default();
        member.change_type = if original_name == name {
            ChangeType::Move
        } else {
            ChangeType::RenameAndMove
        };
    });
    Ok(())
}

fn validate_semantic_member_name(
    bundle: &BundleProposal,
    member_id: Uuid,
    raw_name: &str,
) -> Result<String, &'static str> {
    let name = Path::new(raw_name)
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
        .ok_or("Filename cannot be empty.")?;
    if bundle
        .members
        .iter()
        .any(|member| member.id != member_id && member.proposed_name.eq_ignore_ascii_case(&name))
    {
        return Err("Another file in this collection already uses that name.");
    }
    Ok(name)
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
    let capability_summary = run.capabilities.summary();
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
                if !capability_summary.is_empty() {
                    span { class: "small muted", "{capability_summary}" }
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

/// Build provenance for bug reports.
///
/// The README tells users not to point tidyup at files they care about, which
/// invites reports from a pre-alpha. A report is only actionable if it says
/// which build produced it: the version alone does not distinguish two commits
/// on `0.1.0`, and the compiled feature set decides whether an optional
/// reranker could have touched the result at all.
///
/// Everything here is either a compile-time constant or a single filesystem
/// probe — nothing is inferred. In particular the embedding-model row reports
/// what `quick_model_check` actually found on disk, never a guess from config.
#[component]
fn About() -> Element {
    let state = use_context::<SharedState>();
    let mut model_ready = state.signals.model_ready;

    // Reuse the Dashboard's cached probe rather than hitting the filesystem
    // again — `model_ready` exists precisely so `verify_default_model` runs once
    // per session. It is only `None` if this card renders before the Dashboard
    // ever mounted, in which case populate it here so the Dashboard inherits
    // the answer instead of repeating the work.
    let ready = use_hook(move || {
        if model_ready.peek().is_none() {
            model_ready.set(Some(quick_model_check().is_ok()));
        }
        *model_ready.peek()
    });

    let model_status = match ready {
        Some(true) => "present",
        Some(false) => "not installed",
        None => "unknown",
    };

    let version = env!("CARGO_PKG_VERSION");
    // Absent in any checkout without a `.git` directory — a release tarball or
    // a vendored source drop. That is a normal build, not a broken one.
    let build = option_env!("TIDYUP_GIT_SHA").unwrap_or("unknown");

    // Compiled-in optional backends. `cfg!` reports what the binary can do at
    // all, which is the first of the three privacy gates; the LLM card below
    // reports the remaining two. The UI deliberately never links the remote
    // backend, so there is no row for it.
    let llm_feature = if cfg!(feature = "llm-fallback") {
        "compiled in"
    } else {
        "not compiled in"
    };

    rsx! {
        div {
            class: "card",
            h2 { class: "card-title", "About" }
            div {
                class: "kv",
                div { class: "kv-key", "version" }
                div { class: "kv-value", "{version}" }
                div { class: "kv-key", "build" }
                div { class: "kv-value", "{build}" }
                div { class: "kv-key", "embedding model" }
                div { class: "kv-value", "{model_status}" }
                div { class: "kv-key", "llm reranker" }
                div { class: "kv-value", "{llm_feature}" }
            }
            p {
                class: "small muted",
                style: "margin: 12px 0 0;",
                "tidyup is pre-alpha and confidence thresholds are not calibrated. Please include the version and build above in any bug report."
            }
        }
    }
}

#[component]
pub(crate) fn Settings() -> Element {
    // Hook: must be called unconditionally (before the fallible config load) so
    // the hook order is stable across renders regardless of load success.
    let state = use_context::<SharedState>();
    let cfg_result = config::load();
    let config_path = config::platform_config_path()
        .map_or_else(|_| "<unresolved>".into(), |p| p.display().to_string());

    match cfg_result {
        Ok(cfg) => {
            let data_dir = config::resolve_data_dir(&cfg.storage)
                .map_or_else(|_| "<unresolved>".into(), |p| p.display().to_string());
            let toml_text =
                toml::to_string_pretty(&cfg).unwrap_or_else(|e| format!("<error: {e}>"));

            // Optional LLM reranker — the three privacy gates, surfaced.
            let llm_active = state.signals.llm_fallback_active;
            let feature_compiled = cfg!(feature = "llm-fallback");
            let config_enabled = cfg.inference.llm_fallback;
            let can_activate = feature_compiled && config_enabled;
            let is_active = *llm_active.read();
            let toggle_class = if is_active {
                "button button-primary"
            } else {
                "button button-secondary"
            };
            let toggle_label = if is_active {
                "Disable for this session"
            } else {
                "Enable for this session"
            };
            let tier3_status = if !feature_compiled {
                "This desktop build was compiled without the `llm-fallback` feature, so the optional reranker is unavailable. Rebuild with `--features llm-fallback` to enable it.".to_string()
            } else if !config_enabled {
                "Set `[inference] llm_fallback = true` in the config file to allow optional LLM reranking. It stays off until you do.".to_string()
            } else if is_active {
                "The optional LLM reranker is active for this session. Ambiguous embedding results may be reranked on-device.".to_string()
            } else {
                "The optional LLM reranker is available. Enable it for this session to refine low-confidence classifications.".to_string()
            };

            rsx! {
                div {
                    h1 { class: "page-title", "Settings" }
                    p {
                        class: "page-subtitle",
                        "Read-only view of the loaded TOML config. Edit the file directly; changes apply on next launch."
                    }

                    About {}

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
                        h2 { class: "card-title", "Optional LLM reranker" }
                        p {
                            class: "small muted",
                            style: "margin: 0 0 12px;",
                            "{tier3_status}"
                        }
                        button {
                            r#type: "button",
                            class: "{toggle_class}",
                            disabled: !can_activate,
                            onclick: move |_| {
                                if can_activate {
                                    let mut active = llm_active;
                                    let now = *active.read();
                                    active.set(!now);
                                }
                            },
                            "{toggle_label}"
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
        // The About card renders here too. A failed config load is exactly the
        // situation someone files a report about, so the build identifiers must
        // not disappear along with the rest of the page.
        Err(e) => rsx! {
            div {
                h1 { class: "page-title", "Settings" }
                div {
                    class: "banner banner-error",
                    strong { "Could not load config." }
                    pre { "{e}" }
                }
                About {}
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
        spawn(async move {
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
fn DryRunToggle(value: Signal<bool>) -> Element {
    let enabled = *value.read();
    let on_change = move |event: Event<FormData>| value.set(event.checked());
    rsx! {
        label { class: "form-hint", style: "display: flex; gap: 8px; margin-bottom: var(--space-md); cursor: pointer;",
            input { r#type: "checkbox", checked: enabled, onchange: on_change }
            span { strong { "Preview only (dry run)" } " — reports what would apply; files, shelves, and proposal state stay unchanged." }
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
        LastReport::Scan { report: r, dry_run } => rsx! {
            ReportSummary {
                title: if dry_run { "Scan preview complete" } else { "Scan complete" },
                dry_run,
                run_id: r.run_id,
                indexed: r.indexed,
                indexing_failed: r.indexing_failed,
                capabilities: r.capabilities.summary(),
                proposed: r.proposed,
                applied: r.applied,
                bundles: r.bundles,
                bundles_applied: r.bundles_applied,
                skipped: r.skipped,
                failed: r.failed,
                already_in_place: r.already_in_place,
                visual_candidates_over_cap: r.visual_candidates_over_cap,
            }
        },
        LastReport::Migration { report: r, dry_run } => rsx! {
            ReportSummary {
                title: if dry_run { "Migration preview complete" } else { "Migration complete" },
                dry_run,
                run_id: r.run_id,
                indexed: r.source_indexed + r.target_indexed,
                indexing_failed: r.indexing_failed,
                capabilities: r.capabilities.summary(),
                proposed: r.proposed,
                applied: r.applied,
                bundles: r.bundles,
                bundles_applied: r.bundles_applied,
                skipped: r.skipped,
                failed: r.failed,
                // Migration always moves out of the source tree.
                already_in_place: 0,
                visual_candidates_over_cap: r.visual_candidates_over_cap,
            }
        },
        LastReport::Rollback(r) => rsx! {
            div {
                class: "card",
                h2 {
                    class: "card-title",
                    if r.failures > 0 || r.conflicts > 0 { "Rollback incomplete" } else { "Rollback complete" }
                }
                div { class: "small muted", "Run {r.run_id}" }
                div {
                    class: "button-row",
                    style: "margin-top: 8px;",
                    span { class: "chip chip-high", "restored {r.restored} file(s)" }
                    span { class: "chip chip-high", "restored {r.bundles_restored} bundle(s)" }
                    if r.failures > 0 {
                        span { class: "chip chip-low", "{r.failures} failure(s)" }
                    }
                    if r.conflicts > 0 {
                        span { class: "chip chip-low", "{r.conflicts} conflict(s)" }
                    }
                }
                if r.conflicts > 0 {
                    p {
                        class: "small muted",
                        "Conflicted items were modified after apply and were left in place \
                         to preserve your edits. Resolve them by hand, then roll back again."
                    }
                }
            }
        },
    }
}

#[component]
#[allow(clippy::too_many_arguments)]
fn ReportSummary(
    title: String,
    dry_run: bool,
    run_id: Uuid,
    indexed: usize,
    indexing_failed: usize,
    capabilities: String,
    proposed: usize,
    applied: usize,
    bundles: usize,
    bundles_applied: usize,
    skipped: usize,
    failed: usize,
    /// Scan only: classified, but already where it belongs.
    already_in_place: usize,
    /// Images the per-directory clustering cap excluded from collection
    /// discovery. Surfaced because they are otherwise indistinguishable from
    /// images the run simply found nothing to group with.
    visual_candidates_over_cap: usize,
) -> Element {
    rsx! {
        div {
            class: "card",
            h2 { class: "card-title", "{title}" }
            div { class: "small muted", "Run {run_id}" }
            div { class: "small muted", "{capabilities}" }
            div {
                class: "button-row",
                style: "margin-top: 8px;",
                span { class: "chip chip-neutral", "{proposed} proposed" }
                span { class: "chip chip-neutral", "{indexed} indexed" }
                if indexing_failed > 0 {
                    span { class: "chip chip-low", "{indexing_failed} indexing failure(s)" }
                }
                span { class: "chip chip-high",    if dry_run { "{applied} would apply" } else { "{applied} applied" } }
                if skipped > 0 { span { class: "chip chip-medium", "{skipped} skipped" } }
                if failed  > 0 { span { class: "chip chip-low",    "{failed} failed" } }
                if bundles > 0 {
                    span { class: "chip chip-neutral", "{bundles} bundle(s)" }
                    span { class: "chip chip-high",    if dry_run { "{bundles_applied} bundle(s) would apply" } else { "{bundles_applied} bundle(s) applied" } }
                }
                if already_in_place > 0 {
                    span { class: "chip chip-neutral", "{already_in_place} already in place" }
                }
            }
            if visual_candidates_over_cap > 0 {
                p {
                    class: "small muted",
                    "{visual_candidates_over_cap} image(s) exceeded the per-directory clustering \
                     limit and were classified individually. They were never compared for \
                     collection grouping, so a large folder may group some images and not others."
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Async actions. Each spawns a task that drives a service to completion.
// ---------------------------------------------------------------------------

/// Read optional LLM-reranker activation from the session toggle.
/// The Settings surface only lets the toggle reach `true` when the cargo feature
/// and the config gate are both satisfied, so this read is the third privacy
/// gate. Read in component scope (not inside the spawned task) to keep the
/// root-anchored signal access on its owning scope.
fn current_activation(signals: SignalBundle) -> InferenceActivation {
    InferenceActivation {
        llm_fallback: signals.llm_fallback_active.cloned(),
    }
}

fn launch_scan(state: &SharedState, source: PathBuf, dry_run: bool) {
    let signals = state.signals;
    let slot = state.review_slot.clone();
    let bundle_slot = state.bundle_review_slot.clone();
    let combined_slot = state.combined_review_slot.clone();
    let activation = current_activation(signals);

    reset_run_state(signals);
    let mut review_source_root = signals.review_source_root;
    review_source_root.set(Some(source.clone()));
    let mut review_target_root = signals.review_target_root;
    review_target_root.set(None);
    set_busy(signals, Busy::Scanning);
    // Config and model loading happen below, before any service call and so
    // before any phase event. Without this the banner renders nothing at all
    // for the whole model load and the window looks inert right after the
    // click that started the run.
    set_phase(signals, tidyup_domain::Phase::Preparing);

    // `spawn_forever` (vs `spawn`): the reviewer flips `review_pending` mid-run,
    // which routes the app to `/review` and unmounts the calling page. A
    // scope-bound `spawn` would have its task cancelled there, which looks
    // exactly like "the migration never commences".
    spawn_forever(async move {
        let result = async {
            let cfg = config::load()?;
            let ctx = build(&cfg, true, activation).await?;
            let candidates = build_default_scan_candidates(ctx.embeddings.as_ref()).await?;
            let image_candidates =
                build_image_scan_candidates(ctx.image_embeddings.as_deref()).await?;
            let audio_candidates =
                build_audio_scan_candidates(ctx.audio_embeddings.as_deref()).await?;
            let reporter = DioxusReporter::new(signals);
            let reviewer = DioxusReviewHandler::new(signals, slot, bundle_slot, combined_slot);

            let service = ScanService::new(Arc::clone(&ctx));
            let report = service
                .run(
                    ScanRequest {
                        root: source,
                        taxonomy_path: None,
                        dry_run,
                        auto_approve_bundles: false,
                        bundle_min_confidence: 0.85,
                    },
                    &candidates,
                    &image_candidates,
                    &audio_candidates,
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
                last.set(Some(LastReport::Scan { report: r, dry_run }));
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

fn launch_migrate(state: &SharedState, source: PathBuf, target: PathBuf, dry_run: bool) {
    let signals = state.signals;
    let slot = state.review_slot.clone();
    let bundle_slot = state.bundle_review_slot.clone();
    let combined_slot = state.combined_review_slot.clone();
    let activation = current_activation(signals);

    reset_run_state(signals);
    let mut review_source_root = signals.review_source_root;
    review_source_root.set(Some(source.clone()));
    let mut review_target_root = signals.review_target_root;
    review_target_root.set(Some(target.clone()));
    set_busy(signals, Busy::Migrating);
    set_phase(signals, tidyup_domain::Phase::Preparing);

    spawn_forever(async move {
        let result = async {
            let cfg = config::load()?;
            let ctx = build(&cfg, true, activation).await?;
            let reporter = DioxusReporter::new(signals);
            let reviewer = DioxusReviewHandler::new(signals, slot, bundle_slot, combined_slot);

            let service = MigrationService::new(Arc::clone(&ctx));
            let report = service
                .run(
                    MigrationRequest {
                        source,
                        target,
                        dry_run,
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
                last.set(Some(LastReport::Migration { report: r, dry_run }));
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
            // Rollback never classifies, so no optional reranker is needed.
            let ctx = build(&cfg, false, InferenceActivation::default()).await?;
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

fn submit_combined_review(state: &SharedState) {
    let signals = state.signals;
    let slot = state.combined_review_slot.clone();
    spawn_forever(async move {
        let tx_opt = {
            let mut guard = slot.lock().await;
            guard.take()
        };
        let Some(tx) = tx_opt else {
            tracing::warn!("submit_combined_review: no pending oneshot sender");
            return;
        };
        let decisions = signals.decisions.read().values().cloned().collect();
        let approvals = signals.bundle_approvals.read().clone();
        let approved_bundles = signals
            .bundles
            .read()
            .iter()
            .filter(|bundle| approvals.get(&bundle.id) == Some(&true))
            .cloned()
            .collect();
        if tx
            .send(ReviewOutcome {
                decisions,
                approved_bundles,
            })
            .is_err()
        {
            tracing::warn!("submit_combined_review: service receiver dropped before send");
        }
    });
}

fn reject_all_combined_and_submit(state: &SharedState) {
    let signals = state.signals;
    let proposals = signals.proposals.read().clone();
    let bundles = signals.bundles.read().clone();
    let mut decisions = signals.decisions;
    decisions.with_mut(|map| {
        map.clear();
        for proposal in &proposals {
            map.insert(proposal.id, ReviewDecision::Reject(proposal.id));
        }
    });
    let mut approvals = signals.bundle_approvals;
    approvals.with_mut(|map| {
        map.clear();
        for bundle in &bundles {
            map.insert(bundle.id, false);
        }
    });
    submit_combined_review(state);
}

/// Send the approved bundle ids to the parked `review_bundles` oneshot. Only
/// bundles explicitly toggled to approve are sent — undecided and rejected
/// bundles are held, the safe default (mirrors `submit_review`).
fn refresh_runs(state: &SharedState) {
    let signals = state.signals;
    spawn_forever(async move {
        refresh_runs_inner(signals).await;
    });
}

async fn refresh_runs_inner(signals: SignalBundle) {
    let loaded = async {
        let cfg = config::load()?;
        let ctx = build(&cfg, false, InferenceActivation::default()).await?;
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

fn set_phase(signals: SignalBundle, phase: tidyup_domain::Phase) {
    let mut p = signals.phase;
    p.set(Some(phase));
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
        tidyup_domain::Phase::Preparing => "Loading models",
        tidyup_domain::Phase::Indexing => "Indexing",
        tidyup_domain::Phase::Clustering => "Grouping related files",
        tidyup_domain::Phase::Extracting => "Extracting content",
        tidyup_domain::Phase::ProfilingTarget => "Profiling target hierarchy",
        tidyup_domain::Phase::Classifying => "Classifying",
        tidyup_domain::Phase::AwaitingReview => "Awaiting review",
        tidyup_domain::Phase::Applying => "Applying approved changes",
        tidyup_domain::Phase::Rollback => "Rolling back",
    }
}

fn confidence_chip(c: f32) -> (&'static str, String) {
    // The shipped classifier uses `Calibration::Identity`, so this is raw
    // semantic evidence, not a probability. Keep the relative tier styling
    // but never dress a cosine score up as a percentage.
    let similarity = format!("similarity {c:.2}");
    let cls = if c >= 0.85 {
        "chip-high"
    } else if c >= 0.6 {
        "chip-medium"
    } else {
        "chip-low"
    };
    (cls, similarity)
}

fn percent_u32(current: u64, total: Option<u64>) -> Option<u32> {
    let total = total.filter(|t| *t > 0)?;
    let pct = current
        .min(total)
        .saturating_mul(100)
        .saturating_add(total / 2)
        .checked_div(total)?;
    u32::try_from(pct).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tidyup_domain::ChangeStatus;

    fn move_proposal(name: &str) -> ChangeProposal {
        ChangeProposal {
            id: Uuid::new_v4(),
            file_id: None,
            change_type: ChangeType::Move,
            original_path: PathBuf::from("/Users/example/Desktop").join(name),
            proposed_path: PathBuf::from("/Users/example/Desktop/Work/Career").join(name),
            proposed_name: name.to_string(),
            confidence: 0.95,
            reasoning: "career-document filename".to_string(),
            needs_review: false,
            status: ChangeStatus::Pending,
            created_at: Utc::now(),
            applied_at: None,
            bundle_id: None,
            classification_confidence: Some(0.95),
            rename_mismatch_score: None,
            content_hash: None,
        }
    }

    #[test]
    fn diff_model_keeps_shared_scan_destination_visible() {
        let proposals = vec![
            move_proposal("Resume_Evan_Rovelli.pdf"),
            move_proposal("Resume_Evan_Rovelli_Audible.pdf"),
        ];
        let model = build_diff_model(
            &proposals,
            &[],
            Some(Path::new("/Users/example/Desktop")),
            None,
        );
        assert!(model.left_rows.iter().any(|row| {
            matches!(row, TreeRow::Folder { name, depth: 0, .. } if name == "Desktop")
        }));
        assert!(model.left_rows.iter().any(|row| {
            matches!(row, TreeRow::Folder { name, depth: 1, .. } if name == "Work")
        }));
        assert!(model.left_rows.iter().any(|row| {
            matches!(row, TreeRow::Folder { name, depth: 2, .. } if name == "Career")
        }));
        assert_eq!(count_folders(&model.left_rows), 3);
    }

    #[test]
    fn diff_model_preserves_current_source_directories_and_root_files() {
        let root_file = move_proposal("semantic.md");
        let root_file_id = root_file.id;
        let mut nested_file = move_proposal("Back Piece.stl");
        nested_file.original_path = PathBuf::from("/Users/example/Desktop/zprint/Back Piece.stl");
        let nested_file_id = nested_file.id;

        let model = build_diff_model(
            &[root_file, nested_file],
            &[],
            Some(Path::new("/Users/example/Desktop")),
            None,
        );

        assert!(matches!(
            model.right_rows.first(),
            Some(CurrentTreeRow::Folder {
                name,
                depth: 0,
                file_count: 2,
            }) if name == "Desktop"
        ));
        assert!(model.right_rows.iter().any(|row| {
            matches!(
                row,
                CurrentTreeRow::Folder {
                    name,
                    depth: 1,
                    file_count: 1,
                } if name == "zprint"
            )
        }));
        assert!(model.right_rows.iter().any(|row| {
            matches!(
                row,
                CurrentTreeRow::File { entry, depth: 1 }
                    if entry.proposal_id == root_file_id
            )
        }));
        assert!(model.right_rows.iter().any(|row| {
            matches!(
                row,
                CurrentTreeRow::File { entry, depth: 2 }
                    if entry.proposal_id == nested_file_id
            )
        }));
    }

    #[test]
    fn diff_model_pins_migration_roots_on_both_sides() {
        let mut proposal = move_proposal("resume.pdf");
        proposal.original_path = PathBuf::from("/Users/example/Incoming/resume.pdf");
        proposal.proposed_path = PathBuf::from("/Users/example/Documents/Career/resume.pdf");

        let model = build_diff_model(
            &[proposal],
            &[],
            Some(Path::new("/Users/example/Incoming")),
            Some(Path::new("/Users/example/Documents")),
        );

        assert!(matches!(
            model.left_rows.first(),
            Some(TreeRow::Folder { name, depth: 0, .. }) if name == "Documents"
        ));
        assert!(matches!(
            model.right_rows.first(),
            Some(CurrentTreeRow::Folder { name, depth: 0, .. }) if name == "Incoming"
        ));
    }

    #[test]
    fn current_siblings_follow_destination_order_to_avoid_crossings() {
        let mut first_source_alphabetically = move_proposal("a.txt");
        first_source_alphabetically.proposed_path = PathBuf::from("/Users/example/Desktop/Z/a.txt");
        let first_id = first_source_alphabetically.id;
        let mut last_source_alphabetically = move_proposal("z.txt");
        last_source_alphabetically.proposed_path = PathBuf::from("/Users/example/Desktop/A/z.txt");
        let last_id = last_source_alphabetically.id;

        let model = build_diff_model(
            &[first_source_alphabetically, last_source_alphabetically],
            &[],
            Some(Path::new("/Users/example/Desktop")),
            None,
        );

        assert!(model.current_file_row_by_id[&last_id] < model.current_file_row_by_id[&first_id]);
    }

    #[test]
    fn plan_overview_collapses_moved_bundles_and_omits_stationary_bundles() {
        let mut moved_member = move_proposal("Back Piece.stl");
        moved_member.original_path = PathBuf::from("/Users/example/Desktop/zprint/Back Piece.stl");
        moved_member.proposed_path =
            PathBuf::from("/Users/example/Desktop/3D Models/zprint/Back Piece.stl");
        let moved_member_id = moved_member.id;
        let moved = BundleProposal::new(
            PathBuf::from("/Users/example/Desktop/zprint"),
            tidyup_domain::BundleKind::DirectoryEnvelope,
            PathBuf::from("/Users/example/Desktop/3D Models"),
            vec![moved_member],
            0.7,
            "move directory as one unit".to_string(),
        )
        .unwrap();
        let moved_id = moved.id;

        let mut stationary_member = move_proposal("keep.txt");
        stationary_member.original_path = PathBuf::from("/Users/example/Desktop/keep/keep.txt");
        stationary_member.proposed_path = stationary_member.original_path.clone();
        let stationary = BundleProposal::new(
            PathBuf::from("/Users/example/Desktop/keep"),
            tidyup_domain::BundleKind::DirectoryEnvelope,
            PathBuf::from("/Users/example/Desktop"),
            vec![stationary_member],
            0.9,
            "already in place".to_string(),
        )
        .unwrap();
        let stationary_id = stationary.id;

        let model = build_diff_model(
            &[],
            &[moved, stationary],
            Some(Path::new("/Users/example/Desktop")),
            None,
        );

        assert_eq!(model.connector_entries.len(), 1);
        assert_eq!(model.connector_entries[0].proposal_id, moved_id);
        assert_eq!(model.connector_entries[0].file_count, 1);
        assert!(matches!(
            model.left_rows.first(),
            Some(TreeRow::Folder { file_count: 2, .. })
        ));
        assert!(matches!(
            model.right_rows.first(),
            Some(CurrentTreeRow::Folder { file_count: 2, .. })
        ));
        assert!(!model.file_row_by_id.contains_key(&moved_member_id));
        assert!(!model.file_row_by_id.contains_key(&stationary_id));
        assert!(model.left_rows.iter().any(|row| {
            matches!(
                row,
                TreeRow::File {
                    proposal_id,
                    bundle_kind: Some(OverviewBundleKind::Directory),
                    file_count: 1,
                    ..
                } if *proposal_id == moved_id
            )
        }));
    }

    #[test]
    fn virtual_collections_keep_raw_source_files_visible() {
        let mut first = move_proposal("first.txt");
        first.proposed_path = PathBuf::from("/Users/example/Desktop/project/first.txt");
        let first_id = first.id;
        let mut second = move_proposal("second.txt");
        second.proposed_path = PathBuf::from("/Users/example/Desktop/project/second.txt");
        let second_id = second.id;
        let bundle = BundleProposal::new(
            PathBuf::from("/Users/example/Desktop"),
            tidyup_domain::BundleKind::SemanticCollection {
                label: "project".to_string(),
            },
            PathBuf::from("/Users/example/Desktop"),
            vec![first, second],
            0.8,
            "related loose files".to_string(),
        )
        .unwrap();
        let bundle_id = bundle.id;

        let model = build_diff_model(
            &[],
            std::slice::from_ref(&bundle),
            Some(Path::new("/Users/example/Desktop")),
            None,
        );

        assert_eq!(model.connector_entries.len(), 2);
        assert!(model
            .connector_entries
            .iter()
            .all(|entry| entry.bundle_kind.is_none() && entry.file_count == 1));
        assert_eq!(
            model.file_row_by_id[&first_id],
            model.file_row_by_id[&second_id]
        );
        assert_eq!(
            model.file_row_by_id[&first_id],
            model.file_row_by_id[&bundle_id]
        );
        assert!(model.right_rows.iter().all(|row| {
            !matches!(
                row,
                CurrentTreeRow::File {
                    entry: RightEntry {
                        bundle_kind: Some(_),
                        ..
                    },
                    ..
                }
            )
        }));
        assert_eq!(overview_locked_ids(&[bundle]), vec![first_id, second_id]);
    }

    #[test]
    fn raw_similarity_is_never_formatted_as_a_percentage() {
        let (_, label) = confidence_chip(0.81);
        assert_eq!(label, "similarity 0.81");
        assert!(!label.contains('%'));
        assert!(!label.contains("confidence"));
    }

    #[test]
    fn semantic_member_editor_rejects_duplicate_sibling_name() {
        let first = move_proposal("first.png");
        let first_id = first.id;
        let second = move_proposal("second.png");
        let bundle = BundleProposal::new(
            PathBuf::from("/Users/example/Desktop"),
            tidyup_domain::BundleKind::SemanticCollection {
                label: "project".to_string(),
            },
            PathBuf::from("/Users/example/Desktop/Work"),
            vec![first, second],
            0.8,
            "shared project evidence".to_string(),
        )
        .unwrap();

        assert_eq!(
            validate_semantic_member_name(&bundle, first_id, "SECOND.PNG"),
            Err("Another file in this collection already uses that name."),
        );
        assert_eq!(
            validate_semantic_member_name(&bundle, first_id, "renamed.png").unwrap(),
            "renamed.png",
        );
    }

    #[test]
    fn loose_rename_editor_validates_filename_extension_and_collision() {
        let mut first = move_proposal("first.png");
        first.change_type = ChangeType::RenameAndMove;
        first.proposed_name = "project_dashboard.png".to_string();
        first.proposed_path.set_file_name(&first.proposed_name);
        let first_id = first.id;
        let second = move_proposal("existing.png");
        let proposals = vec![first, second];

        assert_eq!(
            validate_loose_proposal_name(&proposals, first_id, "robot_fleet.png").unwrap(),
            "robot_fleet.png",
        );
        assert_eq!(
            validate_loose_proposal_name(&proposals, first_id, "robot_fleet.jpg"),
            Err("Filename extension must stay the same."),
        );
        assert_eq!(
            validate_loose_proposal_name(&proposals, first_id, "EXISTING.PNG"),
            Err("Another proposal in this folder already uses that filename."),
        );
        assert_eq!(
            validate_loose_proposal_name(&proposals, first_id, "nested/robot_fleet.png"),
            Err("Enter a filename, not a path."),
        );
    }

    #[test]
    fn approving_a_loose_rename_carries_the_edited_target() {
        let mut proposal = move_proposal("Screenshot.png");
        proposal.change_type = ChangeType::RenameAndMove;
        proposal.proposed_name = "robot_fleet_dashboard.png".to_string();
        proposal
            .proposed_path
            .set_file_name(&proposal.proposed_name);

        assert_eq!(
            approval_decision(&proposal),
            ReviewDecision::Override {
                proposal_id: proposal.id,
                new_target: proposal.proposed_path.clone(),
            },
        );
    }
}
