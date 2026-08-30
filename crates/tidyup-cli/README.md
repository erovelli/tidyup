# tidyup-cli

Headless CLI binary for [tidyup](https://github.com/erovelli/tidyup) — the on-device AI file organizer. Installs as `tidyup`.

```bash
tidyup migrate ~/Downloads ~/Documents --dry-run
tidyup scan ~/Documents
tidyup rollback <run-id>
```

The CLI is a thin adapter: it implements the `ProgressReporter` and `ReviewHandler` ports from `tidyup-core` (indicatif progress + interactive prompts or restricted `--yes` auto-approval) and delegates all logic to `tidyup-app`. Text summaries and `--json` include identity-index counts, indexing failures, and the run's actual capability manifest.

`--yes` applies only loose move-only proposals above `[classifier] min_confidence` and recognized opaque structural bundles above the application's separate `0.50` raw-cosine floor. Semantic/file-set collections and every rename remain pending for explicit review. Dry-run uses the same non-zero dispositions as a real run but leaves files, shelves, and applied/rejected state untouched; diagnostic run/proposal records remain local. `tidyup prune` expires both shelved backups and reconstructible semantic artifacts.

## Optional inference backends

Default builds are LLM-silent and network-silent. Two power-user cargo features bring in optional rerank backends; activation is **three-gated** (cargo feature + config + per-invocation flag):

- `--features llm-fallback` — local `mistralrs` reranker. Activate with `--llm-fallback` (or `TIDYUP_LLM_FALLBACK=1`) plus `[inference] llm_fallback = true` in config.
- `--features remote` — HTTP reranker. Activate with `--remote` (or `TIDYUP_REMOTE=1`) plus an `[inference.remote]` section in config. Only the OpenAI-compatible endpoint is wired from config today; Anthropic/Ollama adapters exist in `tidyup-inference-remote` but aren't yet selectable (an Ollama server works via its OpenAI-compatible `/v1` endpoint).

The two activation flags are mutually exclusive. The CLI fails fast with a rebuild hint if you pass an activation flag without the matching feature compiled in.
