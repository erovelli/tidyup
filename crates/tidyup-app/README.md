# tidyup-app

Application services for [tidyup](https://github.com/erovelli/tidyup) — the plug-and-play handles shared by the CLI and desktop UI. Both frontends construct identical `MigrationService`/`ScanService`/`RollbackService` instances and differ only in their `ProgressReporter` and `ReviewHandler` implementations.

Also hosts layered configuration (`tidyup_app::config`) since config has no heavy deps of its own.
