# tidyup-cli

Headless CLI binary for [tidyup](https://github.com/erovelli/tidyup) — the on-device AI file organizer. Installs as `tidyup`.

```bash
tidyup migrate ~/Downloads ~/Documents --dry-run
tidyup scan ~/Documents
tidyup rollback <run-id>
```

The CLI is a thin adapter: it implements the `ProgressReporter` and `ReviewHandler` ports from `tidyup-core` (indicatif progress + interactive prompts or `--yes` auto-approval) and delegates all logic to `tidyup-app`.

## Optional inference backends

Default builds are LLM-silent and network-silent. Two power-user cargo features bring in optional Tier 3 backends; activation is **three-gated** (cargo feature + config + per-invocation flag):

- `--features llm-fallback` — local `mistralrs` Tier 3. Activate with `--llm-fallback` (or `TIDYUP_LLM_FALLBACK=1`) plus `[inference] llm_fallback = true` in config.
- `--features remote` — HTTP Tier 3. Activate with `--remote` (or `TIDYUP_REMOTE=1`) plus an `[inference.remote]` section in config. Only the `OpenAI`-compatible endpoint is wired from config today; Anthropic/Ollama adapters exist in `tidyup-inference-remote` but aren't yet selectable (an Ollama server works via its OpenAI-compatible `/v1` endpoint).

The two activation flags are mutually exclusive. The CLI fails fast with a rebuild hint if you pass an activation flag without the matching feature compiled in.
