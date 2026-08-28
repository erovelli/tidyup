//! Command dispatch — each branch builds a `ServiceContext`, a `CliReporter`,
//! and either an `AutoApproveHandler` or `InteractiveHandler`, then calls the
//! matching application service.

use anyhow::{Context, Result};
use tidyup_app::config;
use tidyup_app::{
    migration::MigrationRequest, scan::ScanRequest, MigrationService, RollbackService, ScanService,
};
use tidyup_core::frontend::{Level, ProgressReporter};

use crate::context::{
    build, build_audio_scan_candidates, build_custom_scan_candidates,
    build_default_scan_candidates, build_image_scan_candidates, describe_data_dir,
    InferenceActivation,
};
use crate::reporter::CliReporter;
use crate::review::{AutoApproveHandler, InteractiveHandler};
use crate::{Cli, Command};

/// Confidence threshold for auto-applying move-only bundles under `--yes`.
const YES_BUNDLE_MIN_CONFIDENCE: f32 = tidyup_app::executor::DEFAULT_BUNDLE_MIN_CONFIDENCE;

/// Interpret an environment variable as a boolean activation gate.
///
/// The documented activation forms are `TIDYUP_LLM_FALLBACK=1` /
/// `TIDYUP_REMOTE=1`. clap's derived `bool`+`env` only accepts `true`/`false`,
/// so we parse the env var here instead: any of `1`/`true`/`yes`/`on`
/// (case-insensitive) activates; unset or anything else does not. An explicitly
/// falsey value (`0`/`false`/`no`/`off`) is honoured as "off".
fn env_activates(var: &str) -> bool {
    parse_boolish(std::env::var(var).ok().as_deref())
}

/// Pure boolish parse (extracted from [`env_activates`] so it's testable
/// without mutating the process environment). `None`/unrecognised → `false`.
fn parse_boolish(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub(crate) async fn dispatch(cli: Cli) -> Result<()> {
    let cfg = config::load().context("loading tidyup config")?;
    let yes = cli.yes;
    let json = cli.json;
    // The flag OR its boolish env var activates. clap's `env` on a `bool` would
    // reject the documented `=1` form, so the env vars are read here instead.
    let llm_fallback = cli.llm_fallback || env_activates("TIDYUP_LLM_FALLBACK");
    let remote = cli.remote || env_activates("TIDYUP_REMOTE");
    if llm_fallback && remote {
        anyhow::bail!(
            "--llm-fallback and --remote are mutually exclusive; pick one Tier 3 backend"
        );
    }
    let activation = InferenceActivation {
        llm_fallback,
        remote,
    };
    match cli.command {
        Command::Migrate {
            source,
            target,
            dry_run,
        } => run_migrate(yes, json, activation, &cfg, source, target, dry_run).await,
        Command::Scan {
            root,
            taxonomy,
            dry_run,
        } => run_scan(yes, json, activation, &cfg, root, taxonomy, dry_run).await,
        Command::Watch {
            root,
            taxonomy,
            debounce_ms,
        } => crate::watch::run_watch(json, activation, &cfg, root, taxonomy, debounce_ms).await,
        Command::Rollback { run_id, list } => {
            if list {
                run_list_runs(json, &cfg).await
            } else if let Some(id) = run_id {
                run_rollback(json, &cfg, id).await
            } else {
                anyhow::bail!("rollback requires a run ID (or pass --list)")
            }
        }
        Command::Prune { days } => run_prune(json, &cfg, days).await,
        Command::Status => run_status(json, &cfg).await,
        Command::Config => run_config(&cfg),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_migrate(
    yes: bool,
    json: bool,
    activation: InferenceActivation,
    cfg: &config::TidyupConfig,
    source: std::path::PathBuf,
    target: std::path::PathBuf,
    dry_run: bool,
) -> Result<()> {
    let ctx = build(cfg, true, activation).await?;
    let reporter = CliReporter::new(json);
    let reviewer = reviewer_for(yes, cfg);

    let service = MigrationService::new(ctx);
    let report = service
        .run(
            MigrationRequest {
                source,
                target,
                dry_run,
                auto_approve_bundles: yes,
                bundle_min_confidence: YES_BUNDLE_MIN_CONFIDENCE,
            },
            &reporter,
            reviewer.as_ref(),
        )
        .await?;

    emit_summary(
        json,
        "migrate",
        report.run_id,
        report.proposed,
        report.bundles,
        report.unclassified,
        report.approved,
        report.applied,
        report.skipped,
        report.failed,
        report.bundles_applied,
        report.bundles_skipped,
        report.bundles_failed,
        dry_run,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_scan(
    yes: bool,
    json: bool,
    activation: InferenceActivation,
    cfg: &config::TidyupConfig,
    root: std::path::PathBuf,
    taxonomy: Option<std::path::PathBuf>,
    dry_run: bool,
) -> Result<()> {
    let ctx = build(cfg, true, activation).await?;
    let reporter = CliReporter::new(json);
    let reviewer = reviewer_for(yes, cfg);

    // A `--taxonomy <file>` overrides the built-in taxonomy for the text tier.
    // Image/audio candidates always use the default per-modality taxonomies.
    let candidates = match taxonomy.as_deref() {
        Some(path) => {
            reporter
                .message(
                    Level::Info,
                    &format!("using custom taxonomy from {}", path.display()),
                )
                .await;
            build_custom_scan_candidates(path, ctx.embeddings.as_ref()).await?
        }
        None => build_default_scan_candidates(ctx.embeddings.as_ref()).await?,
    };
    let image_candidates = build_image_scan_candidates(ctx.image_embeddings.as_deref()).await?;
    let audio_candidates = build_audio_scan_candidates(ctx.audio_embeddings.as_deref()).await?;

    let service = ScanService::new(ctx);
    let report = service
        .run(
            ScanRequest {
                root,
                taxonomy_path: taxonomy,
                dry_run,
                auto_approve_bundles: yes,
                bundle_min_confidence: YES_BUNDLE_MIN_CONFIDENCE,
            },
            &candidates,
            &image_candidates,
            &audio_candidates,
            &reporter,
            reviewer.as_ref(),
        )
        .await?;

    emit_summary(
        json,
        "scan",
        report.run_id,
        report.proposed,
        report.bundles,
        report.unclassified,
        report.approved,
        report.applied,
        report.skipped,
        report.failed,
        report.bundles_applied,
        report.bundles_skipped,
        report.bundles_failed,
        dry_run,
    );
    Ok(())
}

async fn run_list_runs(json: bool, cfg: &config::TidyupConfig) -> Result<()> {
    // Rollback never invokes the classifier, so Tier 3 activation is irrelevant.
    let ctx = build(cfg, false, InferenceActivation::default()).await?;
    let service = RollbackService::new(ctx);
    let runs = service.list_runs().await?;

    if json {
        let rows: Vec<_> = runs
            .iter()
            .map(|r| {
                serde_json::json!({
                    "run_id": r.id,
                    "mode": r.mode.as_str(),
                    "state": r.state.as_str(),
                    "source_root": r.source_root,
                    "target_root": r.target_root,
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                })
            })
            .collect();
        println!("{}", serde_json::json!({"event": "runs", "runs": rows}));
        return Ok(());
    }

    if runs.is_empty() {
        println!("No recorded runs.");
        return Ok(());
    }
    println!("Recorded runs (most recent first):");
    for r in &runs {
        let target = r
            .target_root
            .as_ref()
            .map(|p| format!(" -> {}", p.display()))
            .unwrap_or_default();
        println!(
            "  {}  {:<8}  {:<12}  {}{}",
            r.id,
            r.mode.as_str(),
            r.state.as_str(),
            r.source_root.display(),
            target,
        );
    }
    Ok(())
}

async fn run_prune(json: bool, cfg: &config::TidyupConfig, days: Option<u32>) -> Result<()> {
    // Pruning never invokes the classifier — the null embedding fallback is fine.
    let ctx = build(cfg, false, InferenceActivation::default()).await?;
    let service = RollbackService::new(ctx);
    let days = days.unwrap_or(cfg.storage.backup_retention_days);
    let pruned = service.prune_backups(days).await?;

    if json {
        println!(
            "{}",
            serde_json::json!({"event": "prune", "pruned": pruned, "older_than_days": days}),
        );
    } else {
        println!("Pruned {pruned} shelved backup(s) older than {days} day(s).");
    }
    Ok(())
}

async fn run_status(json: bool, cfg: &config::TidyupConfig) -> Result<()> {
    let data = describe_data_dir(cfg).unwrap_or_else(|| "<unresolved>".into());
    let model_ready = tidyup_embeddings_ort::verify_default_model().is_ok();
    // Status never classifies; absence of the model is reported, not fatal.
    let ctx = build(cfg, false, InferenceActivation::default()).await?;
    let service = RollbackService::new(ctx);
    let runs = service.list_runs().await?;
    let recent: Vec<_> = runs.iter().take(10).collect();
    // A run left `InProgress` is one whose process died mid-apply (a crash or
    // Ctrl-C): the applied subset is journaled per-move in the change log, so
    // `tidyup rollback <id>` reverses whatever landed. Surface these so the user
    // can reconcile.
    let interrupted: Vec<_> = runs
        .iter()
        .filter(|r| r.state == tidyup_domain::RunState::InProgress)
        .collect();

    if json {
        let rows: Vec<_> = recent
            .iter()
            .map(|r| {
                serde_json::json!({
                    "run_id": r.id,
                    "mode": r.mode.as_str(),
                    "state": r.state.as_str(),
                    "started_at": r.started_at,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "event": "status",
                "data_dir": data,
                "model_ready": model_ready,
                "backup_retention_days": cfg.storage.backup_retention_days,
                "total_runs": runs.len(),
                "interrupted_runs": interrupted.iter().map(|r| r.id).collect::<Vec<_>>(),
                "recent_runs": rows,
            }),
        );
        return Ok(());
    }

    println!("tidyup status");
    println!("  data dir:          {data}");
    println!(
        "  embedding model:   {}",
        if model_ready {
            "installed"
        } else {
            "missing (run `cargo xtask download-models`)"
        },
    );
    println!(
        "  backup retention:  {} day(s)",
        cfg.storage.backup_retention_days,
    );
    println!("  recorded runs:     {}", runs.len());
    if interrupted.is_empty() {
        println!("  interrupted runs:  none");
    } else {
        println!(
            "  interrupted runs:  {} (process died mid-apply)",
            interrupted.len()
        );
        for r in &interrupted {
            println!(
                "    {}  {:<8}  started {}  — reverse with `tidyup rollback {}`",
                r.id,
                r.mode.as_str(),
                r.started_at,
                r.id,
            );
        }
    }
    if recent.is_empty() {
        println!("  recent runs:       none");
    } else {
        println!("  recent runs:");
        for r in recent {
            println!(
                "    {}  {:<8}  {:<12}  {}",
                r.id,
                r.mode.as_str(),
                r.state.as_str(),
                r.started_at,
            );
        }
    }
    Ok(())
}

async fn run_rollback(json: bool, cfg: &config::TidyupConfig, run_id: uuid::Uuid) -> Result<()> {
    // Rollback never invokes the classifier, so Tier 3 activation is irrelevant.
    let ctx = build(cfg, false, InferenceActivation::default()).await?;
    let reporter = CliReporter::new(json);
    let service = RollbackService::new(ctx);
    let report = service.rollback_run(run_id, &reporter).await?;

    if json {
        let v = serde_json::json!({
            "event": "rollback_summary",
            "run_id": report.run_id,
            "restored": report.restored,
            "bundles_restored": report.bundles_restored,
            "failures": report.failures,
            "conflicts": report.conflicts,
        });
        println!("{v}");
    } else {
        println!(
            "Rollback {}: restored {} loose change(s), {} bundle(s); {} failure(s), {} conflict(s).",
            report.run_id,
            report.restored,
            report.bundles_restored,
            report.failures,
            report.conflicts,
        );
        if report.conflicts > 0 {
            println!(
                "Conflicted item(s) were modified after apply and were left in place to \
                 preserve your edits. Resolve them by hand, then re-run rollback."
            );
        }
    }
    Ok(())
}

fn run_config(cfg: &config::TidyupConfig) -> Result<()> {
    let data = describe_data_dir(cfg).unwrap_or_else(|| "<unresolved>".into());
    // Show the path `load` actually reads (honors `TIDYUP_CONFIG_PATH`), not
    // the bare platform path — otherwise `tidyup config` prints one file while
    // loading another under an env override.
    let config_path = config::resolved_config_path()
        .map_or_else(|_| "<unresolved>".into(), |p| p.display().to_string());
    println!("tidyup config");
    println!("  config file: {config_path}");
    println!("  data dir:    {data}");
    println!();
    let toml_text =
        toml::to_string_pretty(cfg).context("serialising config to TOML for display")?;
    print!("{toml_text}");
    Ok(())
}

/// Build the review handler. Under `--yes`, the auto-approve confidence
/// threshold comes from `[classifier] min_confidence` in config (default 0.75),
/// so users can tune how aggressively moves auto-apply without a rebuild.
fn reviewer_for(yes: bool, cfg: &config::TidyupConfig) -> Box<dyn tidyup_core::ReviewHandler> {
    if yes {
        Box::new(AutoApproveHandler {
            min_confidence: cfg.classifier.min_confidence,
        })
    } else {
        Box::new(InteractiveHandler)
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_summary(
    json: bool,
    mode: &str,
    run_id: uuid::Uuid,
    proposed: usize,
    bundles: usize,
    unclassified: usize,
    approved: usize,
    applied: usize,
    skipped: usize,
    failed: usize,
    bundles_applied: usize,
    bundles_skipped: usize,
    bundles_failed: usize,
    dry_run: bool,
) {
    if json {
        let v = serde_json::json!({
            "event": format!("{mode}_summary"),
            "run_id": run_id,
            "dry_run": dry_run,
            "proposed": proposed,
            "bundles": bundles,
            "unclassified": unclassified,
            "approved": approved,
            "applied": applied,
            "skipped": skipped,
            "failed": failed,
            "bundles_applied": bundles_applied,
            "bundles_skipped": bundles_skipped,
            "bundles_failed": bundles_failed,
        });
        println!("{v}");
        return;
    }
    let tag = if dry_run { " [dry-run]" } else { "" };
    println!();
    println!("{mode} complete{tag} (run {run_id}):");
    println!(
        "  proposals: {proposed} (approved {approved}, applied {applied}, skipped {skipped}, failed {failed})"
    );
    println!(
        "  bundles:   {bundles} (applied {bundles_applied}, skipped {bundles_skipped}, failed {bundles_failed})"
    );
    if unclassified > 0 {
        println!("  unclassified: {unclassified}");
    }
    if applied > 0 {
        println!("Undo with: tidyup rollback {run_id}");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_boolish_accepts_documented_truthy_values() {
        for v in ["1", "true", "TRUE", "yes", "on", " 1 ", "On"] {
            assert!(parse_boolish(Some(v)), "{v:?} should activate");
        }
    }

    #[test]
    fn parse_boolish_rejects_falsey_and_unset() {
        for v in ["0", "false", "no", "off", "", "maybe", "2"] {
            assert!(!parse_boolish(Some(v)), "{v:?} should not activate");
        }
        assert!(!parse_boolish(None), "unset must not activate");
    }

    #[tokio::test]
    async fn yes_threshold_comes_from_classifier_min_confidence() {
        // The --yes auto-approve threshold is [classifier] min_confidence, not a
        // hardcoded constant — a tuned config changes how aggressively --yes
        // auto-applies. Build the handler via reviewer_for and observe behavior.
        use tidyup_domain::{ChangeProposal, ChangeStatus, ChangeType, ReviewDecision};

        fn proposal_with_conf(conf: f32) -> ChangeProposal {
            ChangeProposal {
                id: uuid::Uuid::new_v4(),
                file_id: None,
                change_type: ChangeType::Move,
                original_path: "/s/x.txt".into(),
                proposed_path: "/d/x.txt".into(),
                proposed_name: "x.txt".to_string(),
                confidence: conf,
                reasoning: "t".to_string(),
                needs_review: false,
                status: ChangeStatus::Pending,
                created_at: chrono::Utc::now(),
                applied_at: None,
                bundle_id: None,
                classification_confidence: Some(conf),
                rename_mismatch_score: None,
                content_hash: None,
            }
        }

        let p = proposal_with_conf(0.80);

        let mut cfg = config::TidyupConfig::default();
        cfg.classifier.min_confidence = 0.90; // strict
        let strict = reviewer_for(true, &cfg);
        let decisions = strict.review(vec![p.clone()]).await.unwrap();
        assert!(
            matches!(decisions[0], ReviewDecision::Reject(_)),
            "0.80 must be rejected under a 0.90 --yes threshold"
        );

        cfg.classifier.min_confidence = 0.50; // lenient
        let lenient = reviewer_for(true, &cfg);
        let decisions = lenient.review(vec![p]).await.unwrap();
        assert!(
            matches!(decisions[0], ReviewDecision::Approve(_)),
            "0.80 must be approved under a 0.50 --yes threshold"
        );
    }
}
