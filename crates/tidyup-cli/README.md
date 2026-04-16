# tidyup-cli

Headless CLI binary for [tidyup](https://github.com/erovelli/tidyup) — the on-device AI file organizer. Installs as `tidyup`.

```bash
tidyup migrate ~/Downloads ~/Documents --dry-run
tidyup scan ~/Documents
tidyup rollback <run-id>
```

The CLI is a thin adapter: it implements the `ProgressReporter` and `ReviewHandler` ports from `tidyup-core` (indicatif progress + interactive prompts or `--yes` auto-approval) and delegates all logic to `tidyup-app`.
