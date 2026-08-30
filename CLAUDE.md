# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

`tidyup` is a CLI-first, open-source Rust tool that auto-sorts a source directory recursively into a preexisting target hierarchy. It classifies files by their **contents** (not just filename/extension) using deterministic embedding similarity running entirely on-device, preserves logical file groupings (coding projects, photo bursts, music albums) as atomic bundles, and proposes rename-and-move operations for review. **Nothing moves without explicit approval.**

The workspace is past its scaffolding stage: every crate (`domain` through `cli`/`ui`) is substantively implemented and the default CLI runs end-to-end (`migrate`/`scan`/`watch`/`rollback`/`prune`/`status`). Remaining work includes video content embeddings, calibrated-confidence-by-default, embedding-verified soft bundles, and packaging; see the roadmap in `README.md`. Status is still **pre-alpha**: confidence thresholds are not calibrated against a real corpus by default.

**Core product promises** (design constraints — never violate):

1. **Local-first, LLM-optional classification.** The default binary has no network code path AND no LLM inference. Semantic embeddings are the sole destination-routing spine; both remote backends and LLM fallback are compile-time + runtime opt-ins for power users only. See `CLASSIFICATION.md`.
2. **Per-bundle atomicity.** Bundles move all-or-nothing. Partial bundle state is never allowed to persist.
3. **Grounded content-based renames with tuned thresholds.** Textual renames combine classification confidence and filename-content mismatch; non-textual semantic renames additionally require grounded contrastive-model concepts with sufficient score and separation. Never auto-applied. Free-form generation and ungrounded fabrication are forbidden.
4. **Every move is reversible.** Originals are shelved, never deleted.

## Commands

```bash
# Full CI locally (fmt + clippy-with-warnings-as-errors + tests). This is the gate.
cargo xtask ci

# Individual steps
cargo xtask check-privacy   # assert default CLI/UI graphs are network+LLM-silent (runs first inside `ci`)
cargo xtask fmt
cargo xtask lint            # clippy --all-features -D warnings
cargo xtask deny            # requires: cargo install cargo-deny
cargo xtask feature-matrix  # requires: cargo install cargo-hack

# Classification accuracy over the labeled golden corpus (xtask/corpus/).
# Classification entries need the bge-small bundle; without it they defer.
# The eval/calibration tool is NOT part of `ci`, which stays model-free.
cargo xtask eval
cargo xtask eval --json
cargo xtask eval --calibrate  # fit Platt calibration over the corpus (shipped default stays uncalibrated)

# The falsifiable "route by contents, not filename" test: held-out
# migration-routing accuracy over an already-organized corpus (folder = label),
# with content-vs-filename/most-frequent/extension baselines + bootstrap CIs.
# `--fail-under` gates on content beating the filename baseline. Needs the model;
# runs in the `model-eval` lane (nightly), never in model-free `ci`.
cargo xtask eval-routing <corpus-dir> [--json] [--fail-under 0.05]

# Fetch model bundles (download verifies each file: pinned BLAKE3 enforced,
# unpinned digests printed to pin). Then check an install on demand.
cargo xtask download-models [--siglip --clap | --multimodal]
cargo xtask verify-models   [--siglip --clap | --multimodal]

# Single crate / single test
cargo check -p tidyup-domain
cargo test  -p tidyup-pipeline
cargo test  -p tidyup-domain file_id_roundtrip

# Binaries
cargo run -p tidyup-cli -- --help
cargo run -p tidyup-cli -- migrate <src> <tgt> --dry-run
```

Toolchain is pinned to **1.95.0** via `rust-toolchain.toml` — needed for the std `cfg_select!` macro that `libsqlite3-sys` 0.38 (pulled in by `rusqlite` 0.40) uses in its build script. (Also satisfies `lofty` 0.24's `Seek for Take`, which only needed 1.90.)

## Architecture — strict layering

Dependency direction is enforced by the crate graph. See `ARCHITECTURE.md` for the full picture; the non-negotiable rules:

```
domain → core → { storage-sqlite, inference-*, embeddings-ort, extract } → pipeline → app → { cli, ui }
```

- `tidyup-domain` is the **lowest-layer stability firewall**. It has no I/O, no async, and no references to other tidyup crates; its small external dependency set supports serialization, ids, timestamps, and typed errors. Breaking change here = intentional. Use `thiserror` for typed errors (see `change::ParseError`).
- `tidyup-core` holds **port traits only** — `FileIndex`, `ChangeLog`, `BackupStore`, `RunLog` (storage), `TextBackend`/`VisionBackend`/`EmbeddingBackend` (inference), `ContentExtractor` (extract), `ProgressReporter`/`ReviewHandler` (frontend). No implementations.
- Impl crates (`storage-sqlite`, `inference-mistralrs`, `inference-remote`, `embeddings-ort`, `extract`) depend on `core` — **never on each other**. This keeps disjoint heavy deps (ONNX runtime, mistralrs, rusqlite) from leaking across the graph.
- `tidyup-pipeline` consumes trait objects from `core`, not concrete types.
- `tidyup-app` wires services (`ScanService`, `MigrationService`, `RollbackService`) with an Arc<dyn Trait> `ServiceContext`. Config lives here — no separate `tidyup-config` crate.
- `tidyup-cli` / `tidyup-ui` are thin adapters: each implements the frontend port pair.

**Before adding a crate**, ask: disjoint heavy deps? feature-gating needed? binary/lib split? If no to all three, add a module.

## Coding standards

**Workspace lints (root `Cargo.toml`) are the source of truth.** Everything below follows from them.

- `unsafe_code = forbid` — workspace-wide. No exceptions.
- `clippy::pedantic + nursery + cargo` at warn. CI treats warnings as errors (`-D warnings`).
- `unwrap_used` / `expect_used` / `dbg_macro` / `todo` / `unimplemented` = warn.
  - In production code: don't use them. Return `Result` and propagate.
  - In tests: add `#[allow(clippy::unwrap_used)]` at the `mod tests` level (not per-call).
- `missing_const_for_fn` (nursery) fires aggressively — prefer `pub const fn` on accessors/matches.
- `derive_partial_eq_without_eq` fires on any struct where `Eq` is possible. Add it.
- Derive `Eq` + `Hash` on ID newtypes (`FileId`, `ContentHash`). Don't derive `Eq` on records holding `f32`.
- Public APIs need doc comments (`missing_debug_implementations = warn`).

**Dependency policy:**

- Minimize external deps. Every added crate must clear: widely used (first-party from a major maintainer, or ~100k+ monthly downloads), actively maintained, pure-Rust where feasible.
- FFI to battle-tested C/C++ is acceptable only when it is an explicit architectural tradeoff. The workspace already uses ONNX Runtime through `ort` and bundled SQLite through `rusqlite`; `mistralrs` via `candle` remains the pure-Rust preference over `llama-cpp-2` for optional local LLM inference.
- Content hashing uses **BLAKE3**, not SHA-256. ~2–3× faster, cryptographically strong, maintained by the BLAKE3 team.
- Network deps (`reqwest`, `hyper`, `rustls`) are absent from the default build. They may enter under `--features remote` (direct use, by design) or `--features llm-fallback` (transitively through `hf-hub` for model download). Default-build verification lives in `cargo xtask ci`.
- New direct deps MUST be added to `[workspace.dependencies]` in root `Cargo.toml` first; crates reference via `{ workspace = true }`. No per-crate version pins.
- License policy (see `deny.toml`): Apache-2.0 / MIT / BSD / ISC / Unicode / Zlib / CC0 / MPL only. GPL-family fails CI.

## The plug-and-play seams

Two patterns are architectural contracts, not suggestions:

1. **Frontend seam.** `tidyup-app` services take `&dyn ProgressReporter` and `&dyn ReviewHandler`. Never embed a frontend impl in a service. Two live implementations already exercise this seam — `tidyup-cli` (indicatif + interactive prompts) and `tidyup-ui` (Dioxus signal-backed progress + oneshot-channel review). Adding another frontend (web, TUI, MCP) = implementing two traits; it must not require a service-layer refactor. Note that `SyncStorage`-backed signals are the UI-side requirement to satisfy `Send + Sync` on those trait objects.

2. **Inference backend registry.** Backends are *intended* to register by capability at runtime, driven by `InferenceConfig.backends` (ordered list of IDs: `"embeddings-ort"` (default), `"mistralrs"`, `"remote-openai"`, `"remote-anthropic"`, `"remote-ollama"`). **Reserved, not yet wired:** the `backends` list is read by serde for forward-compat but the context builder does not consult it — it selects the optional reranker directly from the `llm_fallback` bool / `[inference.remote]` section (see `ARCHITECTURE.md`). Runtime *selection* is config-driven — not a cargo feature flag. Adding a backend: new `tidyup-inference-*` crate + implement `TextBackend`/`VisionBackend`/`EmbeddingBackend` + register. No pipeline/app changes.

Storage follows the same shape (`FileIndex`/`ChangeLog`/`BackupStore`/`RunLog` are traits, sqlite is the default impl) but we don't expect alternates pre-v0.1.

**Backend *inclusion* is separate from *selection*.** Network-capable backends (`tidyup-inference-remote`) are compiled in only with `--features remote`. Default builds have no HTTP client linked. See the privacy model below.

## Privacy model — the load-bearing promise

Default `cargo build -p tidyup-cli` produces a **network-silent, LLM-silent** binary. No HTTP client (no `reqwest`, `hyper`, `rustls`) AND no LLM inference (no `mistralrs`, `candle`, `hf-hub`, heavy tokenizer tree) — not linked, not present, not reachable.

Verification: `cargo tree -p tidyup-cli -e normal | grep -E 'reqwest|hyper|rustls|mistralrs|candle|hf-hub'` returns empty. This is a CI-checked invariant for the default binary — break it and CI fails. `cargo xtask check-privacy` additionally asserts the default **`tidyup-ui`** graph is LLM-silent; the desktop UI surfaces optional LLM reranking behind the same triple gate (cargo feature → config → session toggle) but deliberately does **not** wire the remote backend. `TIDYUP_LLM_FALLBACK=1` may pre-arm the session toggle only after an asynchronous config load confirms the other gates. Network-silence is enforced only for the CLI (a webview app is not an airplane-mode promise).

**Two symmetric power-user opt-in features**, each gated identically. The shape is the same; the bans are different.

**Remote inference (`--features remote`)**

1. *Compile-time:* build with `--features remote` to include `tidyup-inference-remote` in the dependency graph.
2. *Runtime:* an `[inference.remote]` section in config TOML, plus an explicit `--remote` flag or `TIDYUP_REMOTE=1` env var per invocation. (Only the OpenAI-compatible endpoint is wired from config today; the `Anthropic`/`Ollama` `RemoteEndpoint` variants exist in `tidyup-inference-remote` but aren't yet selectable — see `README.md`.)

**LLM fallback (`--features llm-fallback`)**

1. *Compile-time:* build with `--features llm-fallback` to include `tidyup-inference-mistralrs` in the dependency graph.
2. *Runtime:* `[inference] llm_fallback = true` in config TOML, plus an explicit `--llm-fallback` flag or `TIDYUP_LLM_FALLBACK=1` env var per invocation.

**Network surface of `--features llm-fallback`**: this feature transitively links `reqwest` through `hf-hub` (mistralrs's mandatory model-download dep). That network surface exists *only to fetch models from Hugging Face* — there is no classifier-time phone-home, no telemetry, no analytics. The default path (neither feature) remains fully network-silent. Treat "llm-fallback implies HTTP for model download" as a documented consequence, not a leak; users who want zero network code should not enable `--features llm-fallback`.

First-run UX, onboarding, and default documentation never recommend either. The tool is designed to be excellent offline with **deterministic embedding classification** via `bge-small-en-v1.5` and ONNX Runtime. LLM fallback exists for sparse learned hierarchies and pathological extraction failures; a truly empty target has no candidates to rank, so bundles fall back to the target root at zero confidence for explicit review. Remote exists for users with a specific deployment need. Both are power-user features explicitly opted into at build + config + invocation.

## Bundle detection and atomicity

Files are not always independent. A coding project, photo burst, or music album loses meaning when fragmented. These are **bundles** and move as atomic units.

**Two bundle shapes.** *Directory bundles* are found during the initial walk (`scanner`), before per-file classification. The scanner descends transparently through unmarked directories, so ordinary nested files remain loose semantic inputs and markers at any depth (`.git/`, `Cargo.toml`, `package.json`, `pyproject.toml`, `*.xcodeproj`, Gradle, ≥2 sibling `.ipynb`) can establish an opaque bundle at their own root. Once a marker root is detected, that subtree is not descended into. Preserving an unmarked relative directory is a placement-policy question, never a reason to hide its contents from discovery. *File-set bundles* are content clusters of loose sibling files, found by a second pass (`clustering`) over the scanner's loose files: **photo bursts** (EXIF capture times within a window), **music albums** (shared ID3 album tag), **document series** (filename families like `invoice-01`, `-02`), and **semantic collections** found either from cross-format descriptive entity evidence or conservative complete-link image-embedding similarity. Explicit EXIF burst evidence runs before visual-neighbor clustering, so a burst remains a `PhotoBurst` and does not inherit semantic-collection rename behavior. Visual collections require at least three members, every member pair above the similarity floor, and a grounded collection label. A filename semantic key must contain non-numeric evidence; bare date/counter prefixes and generic screenshot names (`Screenshot`, `Screen Shot`, across space/dash/underscore variants) never create an atomic group. Generic screenshots may join an already-anchored collection only when locally extracted text overlaps its evidence. Per-run extraction and modality embeddings are shared by clustering, placement, and naming. File-set clustering is directory-local; a move-only basename collision is rejected before review, while grounded rename collisions receive deterministic suffixes. File-set bundles have no shared directory to rename — `BundleKind::moves_as_file_set()` is the discriminator the executor and rollback branch on.

**Atomic apply** (in `tidyup-app::executor`, routed by `BundleKind::moves_as_file_set()`):

- *Directory bundles* — same-volume: a single `std::fs::rename()` on the bundle root (POSIX `rename(2)` / NTFS `MoveFile`, atomic on one volume, no intermediate state). Cross-volume: copy-verify-delete the whole subtree (verify by content hash), then delete the original; any failure discards staged data, originals untouched.
- *File-set bundles* — pre-flight (every source present, no target occupied), then shelve + move each member individually, keyed by the member's **own** proposal id. **Any member failure reverses all completed moves** (LIFO), so the cluster relocates whole or not at all. Rollback restores each member from its shelf record by the same id.

**Domain shape.** Bundles are a first-class aggregate: `BundleProposal { root, kind, members: Vec<ChangeProposal>, target_parent, confidence, status }`. Individual member proposals are never approved, applied, or rolled back independently. Structural bundle members preserve their names; `SemanticCollection` members may carry grounded `RenameAndMove` proposals while remaining atomic. The SQL schema: a `bundles` table plus a `bundle_id` foreign key on `change_proposals`.

**Bundle review is per-bundle, never per-member.** `ReviewHandler::review_bundles` returns the approved `BundleProposal`s so a frontend can edit a semantic-collection label or member filename without adding a per-member approval path. The default impl approves nothing, so a frontend without a bundle surface holds every bundle. `--yes` skips the handler and applies the raw-cosine confidence threshold only to recognized opaque structural bundles. Soft/file-set collections remain pending until action-specific calibration exists; collections containing renames remain pending for explicit review. The legacy `Generic` kind remains review-only if read from persisted data, but current scanning does not create generic envelopes.

**The executor owns review-boundary integrity.** Frontend-returned bundles are untrusted. `apply_bundles` reconciles each one against the original by bundle/member id and rejects additions, removals, duplicate ids, changed roots/source paths/content hashes, a changed destination parent, or any member target outside the semantic collection's `target_parent/label/filename` shape. Only a `SemanticCollection` label and member basenames are editable; the executor rebuilds the approved aggregate from original immutable fields before applying it.

**Do not** introduce partial-bundle apply paths or any code that lets some members move while others don't. (File-set bundles necessarily move members one at a time, but the executor reverses every completed move on any failure and rollback restores every member — still strictly all-or-nothing.) This invariant has no exceptions.

## Rename policy

Text/metadata rename proposals require two signals, both above config thresholds:

1. deterministic embedding `classification_confidence ≥ min_classification_confidence` (default raw score 0.85; an optional LLM rerank never supplies this gate)
2. `filename_content_mismatch ≥ min_mismatch_score` (default 0.60) — computed as `1.0 - cosine(embed(filename_as_text), content_embedding)`

Both thresholds are user-tunable via `[rename]` config section. Log the sub-scores in the proposal's `reasoning` field for post-hoc calibration.

Non-textual semantic renames (for example, `image_2348985fg.png` → `cat_on_beach.png`) use a modality's contrastive embedding and require three gates: destination confidence clears the embedding threshold; the fraction of selected concept labels absent from literal filename tokens clears `[rename] min_grounded_mismatch`; and the selected concepts clear `min_grounding_confidence` plus `min_grounding_gap`. This discrete grounded metric deliberately has a separate threshold from text's continuous cosine mismatch. Concepts come from a bounded data asset plus dynamic target-folder labels, are scored in the file's own latent space, and are retained in proposal reasoning. This is retrieval, not caption generation.

Rename provenance never bypasses either gate. In particular, `RenameSource::Ocr` means the candidate came from locally recognized visible text; it does not lower or disable the configured classification-confidence or filename-mismatch thresholds.

**Renames never auto-apply.** `--yes` auto-approves loose move-only proposals above `[classifier] min_confidence` (`0.75` default) and recognized opaque structural bundles above the separate internal raw-cosine floor (`0.50`); rename decisions always surface in review explicitly. Soft/file-set collections remain review-only until action-specific bundle calibration exists. This includes semantic collections: if any member carries a rename, the whole collection is held for explicit atomic review.

**OCR is bounded and optional.** Image extraction must not launch OCR for an entire photo library. On supported macOS builds, Vision OCR is limited to plausible screenshot names, `[extraction] ocr_enabled`, and `ocr_max_bytes` (20 MiB default). A missing/incompatible Swift/Xcode toolchain degrades to a warning and compiles OCR out; it must never make the workspace unbuildable. The embedded helper is materialized as a private, automatically cleaned temporary file rather than written through a predictable path.

## Two operational modes

Both produce `ChangeProposal`s and `BundleProposal`s that flow through the same review flow. Both run bundle detection first; only loose (non-bundle) files enter per-file classification.

1. **Scan mode** (`tidyup-pipeline::scan`) — semantic embeddings rank each loose file against a fixed taxonomy using filename/path/MIME context plus extracted content. When SigLIP is present, the same cached image embedding also supports grounded visual naming and conservative visual-neighbor collections. There is no extension/keyword destination router. An **optional**, hardware-dependent LLM fallback—outside the one-second core semantic budget and only when compiled and triple-gated at runtime—may rerank uncertain embedding results. Default builds exclude it and surface low-confidence files directly to review.

2. **Migration mode** (`tidyup-pipeline::migration`) — classify against an *existing* target hierarchy. Embeddings rank pre-built `FolderProfile`s, with optional LLM reranking under the same feature gate. Every folder gets modality-specific text prototypes from its path/label; populated folders additionally get bounded `content_centroid`, `image_centroid`, and `audio_centroid` signals. This lets an empty but named `Photos/Cats` folder attract matching images without comparing across latent spaces. Review is the primary safety net for low-confidence cases. In a truly empty target, loose files are reported unclassified while atomic bundles are preserved at the target root with zero confidence and mandatory review; no file is silently omitted.

**Semantic artifact caching is live for image/audio embeddings.** `FileIndex` persists `SemanticArtifact`s under `(content_hash, model_id, preprocessing_version, latent_space)` as compact little-endian `f32` blobs; the SQLite reader remains compatible with legacy JSON rows. `tidyup prune` removes artifacts older than the selected retention window. A run also keeps an in-memory path/model cache shared by clustering, placement, and naming, so unchanged media is inferred once and reused across runs. Full text-extraction/result fan-out for duplicate documents is still planned: the pipeline does not yet classify one text result and fan it out to every path with the same hash.

**Processing accounting and capability provenance are durable.** Every operational run persists a versioned `CapabilityManifest` describing the backends/extractors actually wired, including unavailable or disabled multimodal/OCR/LLM channels. Unknown future capability kinds deserialize as `Unknown`, and an invalid diagnostic manifest falls back to an empty manifest so status and rollback listing remain available. `FileIndex` stores run-scoped `FileProcessingRecord`s for source and target-profile roles, batching each stage into one storage transaction. Discovery/indexing failures retain their path/stage/reason without requiring a `FileId`; successful source records end as `Classified`, `Unclassified`, or identity-only `Indexed`. On a run-level failure, incomplete source records become `Failed`. Frontend “indexed” counts come from indexing progress/report state, never proposal count.

## Safety model — invariants

- Every applied move is preceded by a backup. If backup fails, the move is aborted.
- Originals are **shelved, never deleted**. `RollbackService` restores them.
- **Rollback never destroys data.** Shelf records carry a content hash (BLAKE3 for files, a canonical tree digest for bundle subtrees). Before deleting any rollback destination, `BackupStore::precheck_restore` verifies the shelf copy is intact and hash-compares the live destination (and the original location) against it. A destination edited after apply — or a new file occupying the original slot — is reported as a **conflict** and left untouched; a missing/corrupt shelf is a failure that leaves the destination in place. A run flips to `RolledBack` only when every item restored cleanly, so partial rollbacks stay retryable. Don't add any rollback path that deletes unverified.
- Default backup TTL is 30 days (configurable).
- `FileIndex::upsert` preserves `FileId` UUIDs across re-scans (upsert-on-path).
- Every regular file the walker can enumerate is represented in the run processing ledger; a path that cannot be read or hashed is a persisted failure, not a missing count. If directory enumeration itself is denied, record the walk-error path when one is available—do not claim identities for children the OS never exposed.
- The capability manifest attached to a run is immutable provenance. Frontends may summarize it but must not infer a capability that the manifest marks unavailable/disabled.
- No file is moved without an approved `ChangeProposal` or `BundleProposal`. `--yes` uses `[classifier] min_confidence` for loose move-only proposals and `DEFAULT_BUNDLE_MIN_CONFIDENCE` for recognized opaque structural bundles; soft/file-set bundles remain review-only, and rename decisions never auto-apply. Dry-run reports count successful validations as “would apply” while leaving source/destination files, shelves, and applied/rejected state untouched; diagnostic run and proposal provenance may still be persisted.
- Every scanned file is accounted for. If proposal construction fails after clustering removed members from the loose-file pass, every affected member is returned in the outcome's `unclassified` bucket and included in report counts. A scan file already at its classified destination is a *successful* classification with no proposal — it belongs in `already_in_place`, never silently in the `Indexed` residue.
- **Hash and identify once per run.** `index_directory` runs before semantic work and produces `SourceIdentities`; `run_scan`/`run_migration` take it and every `ChangeProposal` draws its `file_id` and `content_hash` from it. Don't reintroduce a second `content_hash_of` at proposal time, and don't leave `file_id: None` — that is what left the processing ledger and the proposal ledger joinable only by path, the one field a move invalidates. An unindexed path falls back to hashing on demand so the apply-time TOCTOU guard still holds.
- **A work cap that changes results must be reported.** `max_semantic_candidates` bounds the quadratic visual-collection pass per directory. Capped files are still classified, so nothing looks wrong — which is exactly why the count is returned in the outcome and surfaced by both frontends. Any future bound that silently changes clustering, naming, or routing quality owes the user the same disclosure; a `tracing::debug!` is not disclosure.
- **Apply re-verifies content (TOCTOU).** Each proposal carries the source's BLAKE3 captured at scan/migration time (`ChangeProposal.content_hash`). The executor re-hashes the source before moving and aborts that item if it changed since review — a file edited or replaced in the review gap is never silently moved under a stale classification. `None` (unhashable at scan time) skips the guard.
- **No destination is applied twice.** The pipeline case-insensitively uniquifies colliding grounded member renames before constructing a semantic collection, and the UI rejects duplicate sibling edits inline. The executor remains the final guard: it refuses an apply where two approved changes resolve to the same target path, and `move_path` never overwrites an existing destination. Migration refuses a source/target that overlap (either nested in the other, or equal) before recording a run. The one exemption is a target that exists *because it is the source*: scan sorts in place, so a self-target is a no-op, not a collision, and must not fail the whole plan. The exemption is narrow — a *different* source targeting an occupied path is still refused.
- **Abstain rather than file under the least-wrong folder.** A semantic collection is placed only when a folder clears both `embedding_threshold` and `ambiguity_gap`. Otherwise migration routes it to the target root at confidence `0.0` with every member marked for review — migration must move everything, so abstention means "I could not place this", not "leave it". Unanimous member agreement short-circuits the ranking only when at least one member cleared its own gate; unanimity among members that all failed it is agreement without confidence. Don't compute an uncertainty signal and then use it only to gate the optional LLM: the default build has no LLM, so that path is the ordinary one.
- **Interrupted runs are recoverable (write-ahead journaling).** The executor shelves the original, marks the change/bundle `Applied` in the change log, *then* moves — the mark is a write-ahead journal written **before** the filesystem mutation. A crash between the mark and the move leaves a recoverable over-approximation (marked applied but not yet moved); rollback's precheck sees the destination absent and the original in place and treats the restore as a safe no-op. A file-set bundle is marked applied before any member moves, so a crash mid-bundle still enumerates in rollback: moved members restore from their shelf, never-shelved members resolve to `NeverMoved` and are skipped. `tidyup status` surfaces `InProgress` runs and points at `tidyup rollback <id>`. Never reorder to move-before-mark — that strands moved-but-unjournaled files rollback can't reach.
- **`tidyup watch` is advisory only.** It re-scans in dry-run on each debounced change and reports proposals; it never moves files. This both upholds the approval promise and avoids an apply→filesystem-event→rescan feedback loop. A future auto-apply-on-change mode would need explicit feedback-loop handling and its own opt-in — don't make `watch` mutate the filesystem by default.
- **Bundles move atomically or not at all.** Same-volume: single atomic `rename()`. Cross-volume: copy-verify-delete with full rollback on any failure.

## Testing conventions

- **No mocking at module boundaries.** Tests use real SQLite, real fixtures, real filesystem (`tempfile`). Mocked tests at these seams drift from production behaviour.
- Unit tests: in-file `#[cfg(test)] mod tests` — fast, no models, no DB.
- Integration tests: `crates/<crate>/tests/*.rs` — real impls, slower.
- Fixture files live under `crates/<crate>/tests/fixtures/`.
- CI runs a layered test matrix — `cargo test --workspace` (default features), `cargo test -p tidyup-cli --features llm-fallback,remote`, and `cargo test -p tidyup-extract --all-features`. `--all-features` is deliberately *not* used workspace-wide (the CLI's `llm-metal`/`llm-cuda` accelerator pass-throughs need platform toolchains). Keep feature combos green.

## Operational rules

- **Models must load sequentially.** Qwen3 + embedding model concurrent load OOMs on 8GB machines. Onboarding flows await models in a single-threaded chain, not `tokio::join!`.
- **Dev profile runs deps at `opt-level=2`, main crate at 0.** Debug-mode inference is otherwise unusable.
- **Taxonomy embedding cache invalidates by hash of taxonomy text**, not by version number.
- **`ProfileCache` carries a `target_root` field** but is currently rebuilt in full on every migration run — not persisted to disk, not looked up by root, not incrementally rebuilt. `ScanDiff` / `diff_scans` (comparing BLAKE3 `FolderMetadata.content_hash`, not timestamps) exist and are unit-tested but are not yet called in production. Incremental rebuild is future work — don't claim it's wired.

## Optional LLM fallback

The pipeline accepts `Option<&dyn TextBackend>` and only consults it when
embedding routing lands in the review zone (`needs_review = true`) AND
`config.enable_llm_fallback` is true. Three invariants future work must
preserve:

- **Three-gate activation.** The privacy model requires (a) the cargo
  feature compiled in (`--features llm-fallback` / `--features remote`),
  (b) the matching config bool (`[inference] llm_fallback = true`) or
  `[inference.remote]` section, and (c) the per-invocation flag
  (`--llm-fallback` / `--remote` or `TIDYUP_LLM_FALLBACK=1` /
  `TIDYUP_REMOTE=1`). The CLI rejects activation without the cargo
  feature; default builds and default invocations stay LLM-silent and
  network-silent. Don't shortcut the gate — don't auto-enable the fallback in
  default invocations, don't read `[inference] llm_fallback = true` from
  config alone, don't infer activation from environment heuristics.
- **`ServiceContext.text` is `Option`.** `None` is the privacy-preserving
  default. The pipeline's optional `text_backend` parameter is fed from
  `ctx.text.as_deref()`. A `NullTextBackend` stand-in is deliberately
  not used — absence is the signal, not a no-op trait object.
- **The LLM fallback never produces renames.** The pipeline calls
  `text_backend.classify_text(content, filename)` and re-embeds
  `category + tags + summary` for re-ranking, but the LLM's
  `suggested_name` field is deliberately dropped. The rename gate is
  driven by the original embedding confidence, not the post-rerank score, so
  LLM reroutes never produce renames — by design.

## Multimodal embeddings

Image and audio classification are cross-modal contrastive lookups (SigLIP /
CLAP) — `tidyup-embeddings-ort::siglip` and `…::clap`. Three invariants
beyond the text-embedding rules:

- **Latent-space isolation.** `EmbeddingBackend`, `ImageEmbeddingBackend`,
  and `AudioEmbeddingBackend` produce vectors in disjoint latent spaces.
  Never cosine-compare across them. In scan mode the pipeline keeps each
  modality's candidate list (`MultimodalContext.image.candidates`,
  `…audio.candidates`) separate from the text candidates. In migration mode
  the same rule applies to folder profiles: `FolderProfile` carries separate
  `image_centroid` (SigLIP space) and `audio_centroid` (CLAP space) alongside
  the text `name_embedding` / `content_centroid`, and an image source file is
  ranked only against `image_name_embedding` / `image_centroid`, and audio only
  against `audio_name_embedding` / `audio_centroid`. The name prototypes come
  from the corresponding modality's own text tower, so empty named folders are
  rankable without a cross-space cosine.
- **Optional inclusion, automatic detection.** SigLIP and CLAP backends live
  inside `tidyup-embeddings-ort` (no separate crate or feature gate — both
  use modality-specific Rust preprocessing and ONNX Runtime inference). They
  are loaded only when
  their bundles exist on disk, via `verify_siglip_model` /
  `verify_clap_model`. Missing bundles are NOT an error — image/audio files
  fall back to the general text-embedding path; uncertain results remain for
  review. Don't add a
  `--multimodal` runtime flag; presence of the artifacts is the gate.
- **Per-modality natural-language prototypes.** The text taxonomy in
  `default_taxonomy()` is keyword-soup tuned for `bge-small`. Image/audio scan
  taxonomies and migration folder-label prototypes use natural-language
  captions because cross-modal contrastive encoders need that phrasing. The
  image concept bank is a versioned data asset, augmented in migration by
  dynamic target-folder labels; concepts ground names but never map directly
  to destinations.

## What NOT to do

- **Don't violate the privacy model.** No HTTP clients, LLM deps, or phone-home code paths in the default binary. Network-capable code lives only in `tidyup-inference-remote` behind `--features remote`. LLM code lives only in `tidyup-inference-mistralrs` behind `--features llm-fallback`.
- **Don't make LLM or remote inference a default.** Both `tidyup-inference-mistralrs` and `tidyup-inference-remote` are feature-gated off by default. Never change CLI defaults, config defaults, or build defaults to turn them on. First-run UX and default docs never recommend either.
- **Don't add generative rename paths.** Rename proposals must be grounded in exact metadata/text or scored concepts retrieved from a bounded contrastive-model bank. No LLM/VLM-fabricated or free-form caption names, even under `--features llm-fallback`.
- **Don't introduce partial-bundle apply paths.** Bundles are atomic. No code that allows some members to move while others don't.
- **Don't auto-apply rename proposals.** Even under `--yes`, renames always surface in review.
- **Don't propose renames for structural bundle members.** Their internal structure is load-bearing. `SemanticCollection` is the narrow exception for formerly loose files, and remains atomic.
- **Don't descend into detected bundles for per-file classification.** Bundle subtrees are opaque to the classifier.
- **Don't claim calibrated confidence by default.** The default `Calibration::Identity` reports raw weighted-cosine. The Platt-scaling mechanism (`tidyup_domain::Calibration`, applied in the scan/migration pipelines) and the fitting tool (`cargo xtask eval --calibrate` over the golden corpus, `tidyup_pipeline::calibration`) now exist — but shipping a *fitted* non-Identity default needs a held-out corpus that doesn't yet exist. Don't change the default or claim calibrated probabilities until it does.
- **Don't add a `tidyup-config` crate** (folded into `app` deliberately).
- **Don't gate backend *selection* by cargo feature** (runtime registry). Backend *inclusion* is feature-gated only for network-capable and LLM backends.
- **Don't add cross-impl-crate deps** (e.g., `storage-sqlite` depending on `inference-mistralrs`). Port traits in `core` are the only shared vocabulary.
- **Don't compare embeddings across modality backends.** `EmbeddingBackend`, `ImageEmbeddingBackend`, and `AudioEmbeddingBackend` produce vectors in disjoint latent spaces. The pipeline keeps each modality's candidate list separate so a cross-space cosine is structurally impossible — don't try to cleverly route around it.
- **Don't reuse the keyword-soup text taxonomy for image/audio scan candidates.** Cross-modal encoders need natural-language captions (`default_image_taxonomy()` / `default_audio_taxonomy()`). Mixing them silently wrecks classification.
- **Don't add `sha2` or other slower hash crates** — use `blake3`.
- **Don't pin dep versions inside a crate's `Cargo.toml`** — use `{ workspace = true }`.
- **Don't skip hooks** (`--no-verify`) or bypass `cargo xtask ci`.
- **Don't commit without running `cargo xtask ci` locally first.**

## Keep docs in sync

Treat docs as part of the change. When a code change lands, update the affected doc(s) in the **same commit** — not as a follow-up, not "later."

- **`README.md`** is the roadmap of record and also makes user-facing claims (default behaviour, feature gates, CLI surface, privacy guarantees, supported modalities, install story). When a Phase item ships, tick the checkbox in the roadmap table and update the "What currently works" / "What does not yet work" lists. When a change alters user-facing behaviour, update the relevant section. If the change is purely internal, leave it alone.
- **`ARCHITECTURE.md`** — update when a crate boundary, seam, port trait, or layering rule changes. Not for implementation-only changes.
- **`CLASSIFICATION.md`** — update when semantic routing, optional reranking, default thresholds, rename behavior, or per-modality coverage changes.
- **`DESIGN.md`** — update when UI/UX tokens, surface rules, component specs, or do/don't guidance change. Frontend work (`tidyup-ui` and any future frontend) should conform to it; deviations get reflected here.
- **`CLAUDE.md`** — update when a change establishes a new invariant or "don't do this" rule future work must obey. It's guidance, not a spec; don't mirror implementation details.

Rule of thumb: if someone reading a doc *today* would be misled by *yesterday's* code change, the doc is out of date. Don't wait to be asked.

## Reference docs in repo

- `ARCHITECTURE.md` — layer diagram, seam rationale, crate boundary justifications.
- `CLASSIFICATION.md` — semantic embedding routing, optional LLM reranking, per-modality roadmap, and rename policy.
- `DESIGN.md` — UI/UX design system ("The Verdant Archive"): colors, typography, elevation, component specs, do/don't rules.
- `README.md` — user-facing overview plus the phase-by-phase roadmap.
- `CONTRIBUTING.md` — PR checklist, commit style.
- `deny.toml` — license + source policy.
- `SECURITY.md` — reporting policy.
