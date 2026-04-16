# Architecture

Tidyup is organised as a Cargo workspace with hexagonal (ports-and-adapters) structure. The shape is designed so that the CLI and the desktop UI share **100% of business logic** and differ only in how they report progress and collect user decisions.

## Layers

```
                      ┌──────────────────────────┐
                      │   tidyup-domain          │  pure types, no deps
                      └────────────┬─────────────┘
                                   │
                      ┌────────────▼─────────────┐
                      │   tidyup-core            │  port traits
                      └────────────┬─────────────┘
              ┌────────────────────┼────────────────────┐
              │                    │                    │
      ┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
      │  storage-      │  │  inference-     │  │  embeddings-    │
      │  sqlite        │  │  mistralrs      │  │  ort            │
      └───────┬────────┘  │  inference-     │  └────────┬────────┘
              │           │  remote         │           │
      ┌───────▼───────────┴─────────────────┴───────────▼─────┐
      │                 tidyup-pipeline                        │
      │         scan-mode + migration-mode classifiers         │
      └──────────────────────────┬─────────────────────────────┘
                                 │
                      ┌──────────▼───────────┐
                      │   tidyup-app         │  ScanService, MigrationService,
                      │                      │  RollbackService, config
                      └──────────┬───────────┘
                    ┌────────────┴────────────┐
                    │                         │
             ┌──────▼──────┐           ┌──────▼──────┐
             │ tidyup-cli  │           │ tidyup-ui   │
             │ (binary)    │           │ (binary)    │
             └─────────────┘           └─────────────┘
```

## The plug-and-play seam

Application services in `tidyup-app` accept `&dyn ProgressReporter` and `&dyn ReviewHandler` — the frontend ports from `tidyup-core::frontend`. Both the CLI (indicatif progress + interactive/auto approval) and the UI (Dioxus signals + diff-view page) implement these same traits.

Consequence: adding a new frontend (web, TUI, MCP server) means implementing two traits, not refactoring business logic.

## Model interchange

Inference backends register by capability, not by cargo feature flag. `tidyup-app::config::InferenceConfig.backends` is an ordered list of backend IDs (`"mistralrs"`, `"remote-openai"`, `"ollama"`). Users swap providers in config — no rebuild.

New backends are added by (a) creating a new `tidyup-inference-*` crate that implements `TextBackend` / `VisionBackend` / `EmbeddingBackend`, and (b) registering it in the registry on startup. No changes to `pipeline/` or `app/` required.

## Storage interchange

`FileIndex`, `ChangeLog`, `BackupStore` are traits. `tidyup-storage-sqlite` is the default implementation. Alternatives (sled, redb, an in-memory test double) slot in without touching the pipeline.

## Why these crate boundaries

| Crate | Reason to be separate |
|---|---|
| `tidyup-domain` | Zero-dep change-stability firewall. Breaking change = intentional |
| `tidyup-core` | Port traits; impl crates depend on this, never on each other |
| `tidyup-app` | Service layer. Holds config because config has no heavy deps |
| `tidyup-pipeline` | Classification logic — heavy enough to deserve isolation |
| `tidyup-extract` | Per-format deps are heavyweight (pdf, excel, image, audio) |
| `tidyup-storage-sqlite` | Default impl; future alternates want a peer slot |
| `tidyup-inference-*` | Each backend has disjoint heavy deps (local ML vs. HTTP) |
| `tidyup-embeddings-ort` | ONNX runtime is a very heavy dep; opt-in |
| `tidyup-cli` / `tidyup-ui` | Distinct binaries with different dep trees |
| `xtask` | Cross-platform workspace automation |

We deliberately **do not** split: config (folded into `app`), domain subtypes, pipeline tiers.
