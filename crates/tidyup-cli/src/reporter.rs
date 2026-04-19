//! CLI [`ProgressReporter`] impl — wraps `indicatif` `MultiProgress` for human mode,
//! or emits line-delimited JSON events for `--json` mode.
//!
//! The reporter is fully async-safe: indicatif's progress bars are thread-safe,
//! and the JSON path writes atomically to stdout with a mutex guard so events
//! don't interleave mid-line under parallel classification.

use std::sync::Mutex;

use async_trait::async_trait;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use tidyup_core::frontend::{Level, ProgressItem, ProgressReporter};
use tidyup_domain::Phase;

#[allow(missing_debug_implementations)]
pub(crate) struct CliReporter {
    pub(crate) json: bool,
    multi: MultiProgress,
    bars: Mutex<Bars>,
}

#[derive(Default)]
struct Bars {
    indexing: Option<ProgressBar>,
    extracting: Option<ProgressBar>,
    profiling: Option<ProgressBar>,
    classifying: Option<ProgressBar>,
    applying: Option<ProgressBar>,
    rollback: Option<ProgressBar>,
}

impl Bars {
    const fn slot_for(&mut self, phase: Phase) -> &mut Option<ProgressBar> {
        match phase {
            Phase::Indexing => &mut self.indexing,
            Phase::Extracting => &mut self.extracting,
            Phase::ProfilingTarget => &mut self.profiling,
            // AwaitingReview shares the classifying slot — it's the phase
            // right after classifying and we don't draw a separate bar.
            Phase::Classifying | Phase::AwaitingReview => &mut self.classifying,
            Phase::Applying => &mut self.applying,
            Phase::Rollback => &mut self.rollback,
        }
    }
}

impl CliReporter {
    pub(crate) fn new(json: bool) -> Self {
        Self {
            json,
            multi: MultiProgress::new(),
            bars: Mutex::new(Bars::default()),
        }
    }

    const fn label(phase: Phase) -> &'static str {
        match phase {
            Phase::Indexing => "indexing",
            Phase::Extracting => "extracting",
            Phase::ProfilingTarget => "profiling target",
            Phase::Classifying => "classifying",
            Phase::AwaitingReview => "awaiting review",
            Phase::Applying => "applying",
            Phase::Rollback => "rolling back",
        }
    }

    fn emit_json<T: Serialize>(event: &T) {
        if let Ok(line) = serde_json::to_string(event) {
            // Single-threaded stdout write so lines don't interleave.
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            let _ = writeln!(lock, "{line}");
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum JsonEvent<'a> {
    PhaseStarted {
        phase: &'a str,
        total: Option<u64>,
    },
    Progress {
        phase: &'a str,
        current: u64,
        total: Option<u64>,
        label: &'a str,
    },
    PhaseFinished {
        phase: &'a str,
    },
    Message {
        level: &'a str,
        message: &'a str,
    },
}

#[async_trait]
#[allow(clippy::significant_drop_tightening)]
impl ProgressReporter for CliReporter {
    async fn phase_started(&self, phase: Phase, total: Option<u64>) {
        let label = Self::label(phase);
        if self.json {
            Self::emit_json(&JsonEvent::PhaseStarted {
                phase: label,
                total,
            });
            return;
        }
        let bar = total.map_or_else(
            || {
                let pb = self.multi.add(ProgressBar::new_spinner());
                if let Ok(style) =
                    ProgressStyle::default_spinner().template("{prefix:>14.cyan} {spinner} {msg}")
                {
                    pb.set_style(style);
                }
                pb.enable_steady_tick(std::time::Duration::from_millis(100));
                pb
            },
            |t| {
                let pb = self.multi.add(ProgressBar::new(t));
                if let Ok(style) = ProgressStyle::default_bar()
                    .template("{prefix:>14.cyan} [{bar:30.green/blue}] {pos}/{len} {msg}")
                {
                    pb.set_style(style.progress_chars("#>-"));
                }
                pb
            },
        );
        bar.set_prefix(label.to_string());
        #[allow(clippy::unwrap_used)]
        let mut bars = self.bars.lock().unwrap();
        let slot = bars.slot_for(phase);
        if slot.is_none() {
            *slot = Some(bar);
        }
    }

    async fn item_completed(&self, phase: Phase, item: ProgressItem) {
        let label = Self::label(phase);
        if self.json {
            Self::emit_json(&JsonEvent::Progress {
                phase: label,
                current: item.current,
                total: item.total,
                label: &item.label,
            });
            return;
        }
        // Clone the progress bar so the guard can drop before the update calls.
        let pb_opt = {
            #[allow(clippy::unwrap_used)]
            let mut bars = self.bars.lock().unwrap();
            bars.slot_for(phase).clone()
        };
        if let Some(pb) = pb_opt {
            pb.set_position(item.current);
            pb.set_message(shorten(&item.label));
        }
    }

    async fn phase_finished(&self, phase: Phase) {
        let label = Self::label(phase);
        if self.json {
            Self::emit_json(&JsonEvent::PhaseFinished { phase: label });
            return;
        }
        #[allow(clippy::unwrap_used)]
        let mut bars = self.bars.lock().unwrap();
        if let Some(pb) = bars.slot_for(phase).take() {
            pb.finish_and_clear();
        }
    }

    async fn message(&self, level: Level, msg: &str) {
        let level_label = match level {
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        };
        if self.json {
            Self::emit_json(&JsonEvent::Message {
                level: level_label,
                message: msg,
            });
        } else {
            let painted = match level {
                Level::Error => console::style(format!("[error] {msg}")).red().to_string(),
                Level::Warn => console::style(format!("[warn]  {msg}"))
                    .yellow()
                    .to_string(),
                Level::Info => console::style(format!("[info]  {msg}")).dim().to_string(),
                Level::Debug => format!("[debug] {msg}"),
            };
            // println to stderr would break alignment with indicatif bars; print via multi.
            let _ = self.multi.println(painted);
        }
    }
}

fn shorten(s: &str) -> String {
    const MAX: usize = 60;
    if s.len() > MAX {
        let tail_len = MAX.saturating_sub(3);
        format!("…{}", &s[s.len() - tail_len..])
    } else {
        s.to_string()
    }
}
