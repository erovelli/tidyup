//! Command dispatch — each branch builds a `ServiceContext`, a `CliReporter`,
//! and either an `AutoApproveHandler` or `InteractiveHandler`, then calls the
//! matching application service.

use crate::reporter::CliReporter;
use crate::review::{AutoApproveHandler, InteractiveHandler};
use crate::{Cli, Command};

pub(crate) async fn dispatch(cli: Cli) -> anyhow::Result<()> {
    let _config = tidyup_app::config::load()?;

    // TODO: build ServiceContext via backend registry:
    //   - pick TextBackend per config.inference.backends order
    //   - construct storage (sqlite), embeddings (ort), extractors
    // let ctx = Arc::new(tidyup_app::ServiceContext { ... });

    let reporter = CliReporter { json: cli.json };
    let _reviewer: Box<dyn tidyup_core::ReviewHandler> = if cli.yes {
        Box::new(AutoApproveHandler {
            min_confidence: 0.66,
        })
    } else {
        Box::new(InteractiveHandler)
    };

    match cli.command {
        Command::Migrate {
            source,
            target,
            dry_run,
        } => {
            // let svc = MigrationService::new(ctx);
            // svc.run(MigrationRequest { source, target, dry_run }, &reporter, &*reviewer).await?;
            let _ = (source, target, dry_run, &reporter);
        }
        Command::Scan {
            root,
            taxonomy,
            dry_run,
        } => {
            let _ = (root, taxonomy, dry_run);
        }
        Command::Rollback { run_id } => {
            let _ = run_id;
        }
        Command::Config => {}
    }
    Ok(())
}
