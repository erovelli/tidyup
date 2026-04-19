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

Inference backends register by capability at runtime. Users **select** among available backends via `tidyup-app::config::InferenceConfig.backends` — an ordered list of backend IDs (`"mistralrs"`, `"remote-openai"`, `"ollama"`) — no rebuild needed to switch between backends that are compiled in.

New backends are added by (a) creating a new `tidyup-inference-*` crate that implements `TextBackend` / `VisionBackend` / `EmbeddingBackend`, and (b) registering it in the registry on startup. No changes to `pipeline/` or `app/` required.

**Inclusion is separate from selection.** Whether a backend is *compiled into the binary at all* is governed by cargo features. Network-capable backends (`tidyup-inference-remote`) are gated behind `--features remote`; LLM backends (`tidyup-inference-mistralrs`) are gated behind `--features llm-fallback`. **Both are excluded from the default binary** — the default classifier is `bge-small-en-v1.5` via `tidyup-embeddings-ort`, which is always compiled in. See [Privacy model](#privacy-model).

## Privacy model

tidyup's default binary is both **network-silent and LLM-silent** for inference. This is an architectural guarantee, not a runtime toggle.

- The default `cargo build -p tidyup-cli` produces a binary with **no HTTP client** (`reqwest` / `hyper` / `rustls` are not linked) AND **no LLM inference** (`mistralrs` / `candle` / `hf-hub` / heavy tokenizer tree are not linked). Classification on the default path is deterministic embedding similarity only (Tier 1 heuristics + Tier 2 `bge-small-en-v1.5` via `ort`). See `CLASSIFICATION.md`.
- Two symmetric opt-in inference features exist, each behind the same two-gate pattern:

  **`tidyup-inference-remote`** — optional workspace member, gated by `--features remote`. Compiling without the feature removes the crate from the dependency graph entirely. Runtime activation additionally requires `[inference] backends = ["remote-..."]` in the config TOML plus a `--remote` flag or env var per invocation.

  **`tidyup-inference-mistralrs`** — optional workspace member, gated by `--features llm-fallback`. Compiling without the feature removes the crate from the dependency graph entirely. Runtime activation additionally requires `[inference] llm_fallback = true` in config plus a `--llm-fallback` flag or env var per invocation.

- Both features are positioned as power-user escape hatches. First-run UX and default docs never recommend either; the tool is designed to be excellent offline with embedding-based classification.
- `cargo-deny` enforces the guarantee on both axes: `reqwest` and transitive network deps banned outside the `remote` feature flag; `mistralrs` / `candle` / `hf-hub` banned outside the `llm-fallback` feature flag.

This is the single most load-bearing decision in the project. Every other architectural choice (embedding-default classification, local-only models, first-run model download, SQLite-only storage, extractive-only renames) flows from it.

## Bundle detection and atomicity

Files are not always independent. A coding project, a photo burst, a music album, a document series — each is a group that loses meaning when fragmented. tidyup treats these as **bundles** and moves them as atomic units.

### Detection

Bundle detection happens during the initial walk, before per-file classification. Markers are checked at each directory level; when a marker is found, the entire subtree is marked as an opaque bundle and its contents are never classified individually.

| Bundle kind | Marker |
|---|---|
| `GitRepository` | `.git/` directory |
| `NodeProject` | `package.json` + `node_modules/` (latter never descended) |
| `RustCrate` | `Cargo.toml` (climbs to `[workspace]` root if present) |
| `PythonProject` | `pyproject.toml` / `setup.py` / `__init__.py` |
| `XcodeProject` | `*.xcodeproj` |
| `AndroidStudioProject` | `settings.gradle` / `build.gradle` |
| `JupyterNotebookSet` | `.ipynb` neighbours |
| `PhotoBurst` | EXIF timestamps within N seconds, same camera |
| `MusicAlbum` | consistent ID3 `album` tag |
| `DocumentSeries` | filename regex family (e.g., `invoice_*.pdf`) |
| `Generic` | user-defined marker (future) |

### Atomic apply

Bundle moves are all-or-nothing. Partial bundle state is never allowed to persist — a half-moved coding project is worse than no move at all (broken builds, dangling git state, invalid IDE indexes).

Two apply paths:

1. **Same-volume bundle moves** use a single `std::fs::rename()` on the bundle root. POSIX `rename(2)` on the same filesystem is atomic; NTFS `MoveFile` is atomic. One syscall, no intermediate state.
2. **Cross-volume bundle moves** use copy-verify-delete: copy entire subtree to shelf-staged destination, verify by content hash, delete original, then commit the final rename. Any failure at any step rolls back everything — the shelf staging area is discarded, originals remain untouched.

The domain models this as a `BundleProposal` aggregate: `{ root, kind, members: Vec<ChangeProposal>, target_parent, confidence, status }`. Individual member proposals are never approved, applied, or rolled back independently. Member proposals cannot carry rename suggestions — bundle identity depends on internal structure.

The SQL schema reflects this: a `bundles` table plus a `bundle_id` foreign key on `change_proposals`. Review, approval, and rollback operate on bundles, not members.

## Rename policy

Rename proposals are driven by two independent signals. Both must clear configured thresholds before a rename is proposed.

1. **Classification confidence** — how confident is the pipeline about what this file is? Tier 2 embedding cosine score or Tier 3 LLM structured confidence.
2. **Filename-content mismatch** — how misleading is the current filename given the content? Computed as `1.0 - cosine(embed(filename_as_text), content_embedding)`. A file named `taxes_2023.pdf` with tax-return content scores low (names matches); `DSC_0481.jpg` of a wedding scores high (name is content-free).

Both thresholds are config-tunable (`[rename] min_classification_confidence`, `[rename] min_mismatch_score`) with conservative defaults (0.85 / 0.60).

**Renames never auto-apply.** `--yes` auto-approves *moves* above a threshold; rename decisions always surface in the review flow explicitly. The risk of silently breaking external references (symlinks, docs pointing at paths, git history) is too high for auto-apply.

**Bundle members never receive rename proposals.** The internal structure of a project is load-bearing; renaming files inside a crate or photo burst breaks meaning.

## Storage interchange

`FileIndex`, `ChangeLog`, `BackupStore`, `RunLog` are traits. `tidyup-storage-sqlite` is the default implementation. Alternatives (sled, redb, an in-memory test double) slot in without touching the pipeline. `RunLog` records every scan/migration invocation so `tidyup rollback <run_id>` can enumerate applied changes and drive `BackupStore::restore` for each.

Content-addressed dedup is a first-class concern: `FileIndex` is keyed by `ContentHash` (BLAKE3), and `IndexedFile` is a many-to-one presence record. Classification happens once per unique hash; a single result applies to every path sharing that content.

## Why these crate boundaries

| Crate | Reason to be separate | Default binary? |
|---|---|---|
| `tidyup-domain` | Zero-dep change-stability firewall. Breaking change = intentional | yes |
| `tidyup-core` | Port traits; impl crates depend on this, never on each other | yes |
| `tidyup-app` | Service layer. Holds config because config has no heavy deps | yes |
| `tidyup-pipeline` | Classification logic — heavy enough to deserve isolation | yes |
| `tidyup-extract` | Per-format deps are heavyweight (pdf, excel, image, audio) | yes |
| `tidyup-storage-sqlite` | Default impl; future alternates want a peer slot | yes |
| `tidyup-inference-mistralrs` | Local LLM — pure-Rust via candle; optional Tier 3 fallback (cold-start + pathological extraction) | **no — `--features llm-fallback`** |
| `tidyup-inference-remote` | HTTP backend; network-capable | **no — `--features remote`** |
| `tidyup-embeddings-ort` | ONNX runtime; hosts `bge-small-en-v1.5` — the default Tier 2 classifier | yes |
| `tidyup-cli` / `tidyup-ui` | Distinct binaries with different dep trees | yes |
| `xtask` | Cross-platform workspace automation | n/a |

We deliberately **do not** split: config (folded into `app`), domain subtypes, pipeline tiers.

## Dependency policy

External deps must clear: widely used (first-party from a major maintainer, or ~100k+ monthly downloads), actively maintained, pure-Rust where feasible. FFI to battle-tested C/C++ is acceptable only when no mature Rust-native alternative exists (none currently required).

Content hashing uses **BLAKE3**, not SHA-256. BLAKE3 is ~2–3× faster, cryptographically strong, and maintained by the BLAKE3 team — aligns with perf, Rust-native, and stable-maintainership criteria.

License policy (see `deny.toml`): Apache-2.0 / MIT / BSD / ISC / Unicode / Zlib / CC0 / MPL only. GPL-family and unknown licenses fail CI.
