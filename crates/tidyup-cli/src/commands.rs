//! Command dispatch — each branch builds a `ServiceContext`, a `CliReporter`,
//! and either an `AutoApproveHandler` or `InteractiveHandler`, then calls the
//! matching application service.

use anyhow::{Context, Result};
use tidyup_app::config;
use tidyup_app::{
    migration::MigrationRequest, scan::ScanRequest, MigrationService, RollbackService, ScanService,
};

use crate::context::{
    build, build_audio_scan_candidates, build_default_scan_candidates, build_image_scan_candidates,
    describe_data_dir, InferenceActivation,
};
use crate::reporter::CliReporter;
use crate::review::{AutoApproveHandler, InteractiveHandler};
use crate::{Cli, Command};

/// Confidence threshold for auto-approving loose proposals when the user
/// passes `--yes`. Tuned to bias toward safe auto-application of Tier-1
/// matches while surfacing Tier-2 ambiguity to review in interactive mode.
const YES_MIN_CONFIDENCE: f32 = 0.75;
/// Confidence threshold for auto-applying bundles under `--yes`.
const YES_BUNDLE_MIN_CONFIDENCE: f32 = 0.85;

pub(crate) async fn dispatch(cli: Cli) -> Result<()> {
    let cfg = config::load().context("loading tidyup config")?;
    let yes = cli.yes;
    let json = cli.json;
    if cli.llm_fallback && cli.remote {
        anyhow::bail!(
            "--llm-fallback and --remote are mutually exclusive; pick one Tier 3 backend"
        );
    }
    let activation = InferenceActivation {
        llm_fallback: cli.llm_fallback,
        remote: cli.remote,
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
        Command::Rollback { run_id, list } => {
            if list {
                run_list_runs(json, &cfg).await
            } else if let Some(id) = run_id {
                run_rollback(json, &cfg, id).await
            } else {
                anyhow::bail!("rollback requires a run ID (or pass --list)")
            }
        }
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
    let reviewer = reviewer_for(yes);

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
    let reviewer = reviewer_for(yes);

    let candidates = build_default_scan_candidates(ctx.embeddings.as_ref()).await?;
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
        });
        println!("{v}");
    } else {
        println!(
            "Rollback {}: restored {} loose change(s), {} bundle(s); {} failure(s).",
            report.run_id, report.restored, report.bundles_restored, report.failures,
        );
    }
    Ok(())
}

fn run_config(cfg: &config::TidyupConfig) -> Result<()> {
    let data = describe_data_dir(cfg).unwrap_or_else(|| "<unresolved>".into());
    let config_path = config::platform_config_path()
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

fn reviewer_for(yes: bool) -> Box<dyn tidyup_core::ReviewHandler> {
    if yes {
        Box::new(AutoApproveHandler {
            min_confidence: YES_MIN_CONFIDENCE,
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
