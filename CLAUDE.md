# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

`tidyup` is a CLI-first, open-source Rust tool that auto-sorts a source directory recursively into a preexisting target hierarchy. It classifies files by their **contents** (not just filename/extension) using deterministic embedding similarity running entirely on-device, preserves logical file groupings (coding projects, photo bursts, music albums) as atomic bundles, and proposes rename-and-move operations for review. **Nothing moves without explicit approval.**

The workspace is under active construction — many modules are scaffolded with typed stubs ahead of full implementation. Check which crates are wired up before assuming behaviour.

**Core product promises** (design constraints — never violate):

1. **Local-first, LLM-optional classification.** The default binary has no network code path AND no LLM inference. Both remote backends and LLM fallback are compile-time + runtime opt-ins for power users only. See `CLASSIFICATION.md` for the tier cascade.
2. **Per-bundle atomicity.** Bundles move all-or-nothing. Partial bundle state is never allowed to persist.
3. **Content-based renames with tuned thresholds.** Rename proposals combine classification confidence and filename-content mismatch; both must clear config thresholds. Never auto-applied. Rename generation is extractive only — no fabrication.
4. **Every move is reversible.** Originals are shelved, never deleted.

## Commands

```bash
# Full CI locally (fmt + clippy-with-warnings-as-errors + tests). This is the gate.
cargo xtask ci

# Individual steps
cargo xtask fmt
cargo xtask lint            # clippy --all-features -D warnings
cargo xtask deny            # requires: cargo install cargo-deny
cargo xtask feature-matrix  # requires: cargo install cargo-hack

# Single crate / single test
cargo check -p tidyup-domain
cargo test  -p tidyup-pipeline
cargo test  -p tidyup-domain file_id_roundtrip

# Binaries
cargo run -p tidyup-cli -- --help
cargo run -p tidyup-cli -- migrate <src> <tgt> --dry-run
```

Toolchain is pinned to **1.90.0** via `rust-toolchain.toml` — needed for `Seek for Take` used by `lofty` 0.24.

## Architecture — strict layering

Dependency direction is enforced by the crate graph. See `ARCHITECTURE.md` for the full picture; the non-negotiable rules:

```
domain → core → { storage-sqlite, inference-*, embeddings-ort, extract } → pipeline → app → { cli, ui }
```

- `tidyup-domain` is a **zero-dep stability firewall**. No `anyhow`, no I/O, no async, no references to other tidyup crates. Breaking change here = intentional. Use `thiserror` for typed errors (see `change::ParseError`).
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
- FFI to battle-tested C/C++ is acceptable only when no mature Rust-native alternative exists (none currently required — `mistralrs` via `candle` is pure Rust and preferred over `llama-cpp-2`).
- Content hashing uses **BLAKE3**, not SHA-256. ~2–3× faster, cryptographically strong, maintained by the BLAKE3 team.
- Network deps (`reqwest`, `hyper`, `rustls`) are absent from the default build. They may enter under `--features remote` (direct use, by design) or `--features llm-fallback` (transitively through `hf-hub` for model download). Default-build verification lives in `cargo xtask ci`.
- New direct deps MUST be added to `[workspace.dependencies]` in root `Cargo.toml` first; crates reference via `{ workspace = true }`. No per-crate version pins.
- License policy (see `deny.toml`): Apache-2.0 / MIT / BSD / ISC / Unicode / Zlib / CC0 / MPL only. GPL-family fails CI.

## The plug-and-play seams

Two patterns are architectural contracts, not suggestions:

1. **Frontend seam.** `tidyup-app` services take `&dyn ProgressReporter` and `&dyn ReviewHandler`. Never embed a frontend impl in a service. Two live implementations already exercise this seam — `tidyup-cli` (indicatif + interactive prompts) and `tidyup-ui` (Dioxus signal-backed progress + oneshot-channel review). Adding another frontend (web, TUI, MCP) = implementing two traits; it must not require a service-layer refactor. Note that `SyncStorage`-backed signals are the UI-side requirement to satisfy `Send + Sync` on those trait objects.

2. **Inference backend registry.** Backends register by capability at runtime, driven by `InferenceConfig.backends` (ordered list of IDs: `"mistralrs"`, `"remote-openai"`, `"ollama"`). Runtime *selection* is config-driven — not a cargo feature flag. Adding a backend: new `tidyup-inference-*` crate + implement `TextBackend`/`VisionBackend`/`EmbeddingBackend` + register. No pipeline/app changes.

Storage follows the same shape (`FileIndex`/`ChangeLog`/`BackupStore`/`RunLog` are traits, sqlite is the default impl) but we don't expect alternates pre-v0.1.

**Backend *inclusion* is separate from *selection*.** Network-capable backends (`tidyup-inference-remote`) are compiled in only with `--features remote`. Default builds have no HTTP client linked. See the privacy model below.

## Privacy model — the load-bearing promise

Default `cargo build -p tidyup-cli` produces a **network-silent, LLM-silent** binary. No HTTP client (no `reqwest`, `hyper`, `rustls`) AND no LLM inference (no `mistralrs`, `candle`, `hf-hub`, heavy tokenizer tree) — not linked, not present, not reachable.

Verification: `cargo tree -p tidyup-cli -e normal | grep -E 'reqwest|hyper|rustls|mistralrs|candle|hf-hub'` returns empty. This is a CI-checked invariant for the default binary — break it and CI fails.

**Two symmetric power-user opt-in features**, each gated identically. The shape is the same; the bans are different.

**Remote inference (`--features remote`)**

1. *Compile-time:* build with `--features remote` to include `tidyup-inference-remote` in the dependency graph.
2. *Runtime:* `[inference] backends = ["remote-..."]` in config TOML, plus an explicit `--remote` flag or `TIDYUP_REMOTE=1` env var per invocation.

**LLM fallback (`--features llm-fallback`)**

1. *Compile-time:* build with `--features llm-fallback` to include `tidyup-inference-mistralrs` in the dependency graph.
2. *Runtime:* `[inference] llm_fallback = true` in config TOML, plus an explicit `--llm-fallback` flag or `TIDYUP_LLM_FALLBACK=1` env var per invocation.

**Network surface of `--features llm-fallback`**: this feature transitively links `reqwest` through `hf-hub` (mistralrs's mandatory model-download dep). That network surface exists *only to fetch models from Hugging Face* — there is no classifier-time phone-home, no telemetry, no analytics. The default path (neither feature) remains fully network-silent. Treat "llm-fallback implies HTTP for model download" as a documented consequence, not a leak; users who want zero network code should not enable `--features llm-fallback`.

First-run UX, onboarding, and default documentation never recommend either. The tool is designed to be excellent offline with **deterministic embedding classification** — `bge-small-en-v1.5` via ONNX Runtime handles Tier 2 on the default path (see `CLASSIFICATION.md`). LLM-fallback exists for cold-start cases (empty target hierarchy, pathological extraction failures); remote exists for users with a specific deployment need. Both are power-user features explicitly opted into at build + config + invocation.

## Bundle detection and atomicity

Files are not always independent. A coding project, photo burst, or music album loses meaning when fragmented. These are **bundles** and move as atomic units.

**Detection happens during the initial walk**, before per-file classification. Markers (`.git/`, `Cargo.toml`, `package.json`, `pyproject.toml`, `*.xcodeproj`, consistent EXIF timestamps, consistent ID3 album tags) mark a subtree as an opaque bundle. **The pipeline never descends into a detected bundle for individual classification.**

**Atomic apply:**

- Same-volume bundle moves use a single `std::fs::rename()` on the bundle root. POSIX `rename(2)` and NTFS `MoveFile` are atomic on the same volume. One syscall, no intermediate state.
- Cross-volume bundle moves use copy-verify-delete: stage the entire subtree on the target volume, verify by content hash, delete original, commit the final rename. Any failure at any step rolls back the entire bundle — staged data is discarded, originals untouched.

**Domain shape.** Bundles are a first-class aggregate: `BundleProposal { root, kind, members: Vec<ChangeProposal>, target_parent, confidence, status }`. Individual member proposals are never approved, applied, or rolled back independently. Member proposals cannot carry rename suggestions. The SQL schema: a `bundles` table plus a `bundle_id` foreign key on `change_proposals`.

**Do not** introduce partial-bundle apply paths, per-member rollback, or any code that allows some members to move while others don't. This invariant has no exceptions.

## Rename policy

Rename proposals require two signals, both above config thresholds:

1. `classification_confidence ≥ min_classification_confidence` (default 0.85)
2. `filename_content_mismatch ≥ min_mismatch_score` (default 0.60) — computed as `1.0 - cosine(embed(filename_as_text), content_embedding)`

Both thresholds are user-tunable via `[rename]` config section. Log the sub-scores in the proposal's `reasoning` field for post-hoc calibration.

**Renames never auto-apply.** `--yes` auto-approves moves above a threshold; rename decisions always surface in review explicitly. Bundle members never receive rename proposals.

## Two operational modes

Both produce `ChangeProposal`s and `BundleProposal`s that flow through the same review flow. Both run bundle detection first; only loose (non-bundle) files enter per-file classification.

1. **Scan mode** (`tidyup-pipeline::scan`) — classify against a fixed taxonomy with a 3-tier cascade: Tier 1 heuristics (~1ms) → Tier 2 embeddings (~50ms, `bge-small-en-v1.5`) → **optional** Tier 3 LLM fallback (1–10s, only when `--features llm-fallback` is compiled and enabled at runtime). Each tier short-circuits above its confidence threshold. On default builds Tier 3 is absent entirely and low-confidence files surface directly to review.

2. **Migration mode** (`tidyup-pipeline::migration`) — classify against an *existing* target hierarchy. Same tier cascade: heuristic+date routing, then batch embeddings against pre-built `FolderProfile`s, with optional Tier 3 LLM fallback under the same feature gate. Review is the primary safety net for low-confidence cases; `--llm-fallback` is the remedy for cold-start (empty target hierarchy) and pathological-extraction workflows.

**Hash-based dedup** is a first-class pipeline concept, not an optimization. `FileIndex` is keyed by `ContentHash`; classification happens once per unique content hash and applies to every path sharing it. Real-world dedup on a home directory is 15–30% — worth enforcing in the schema, not bolting on later.

## Safety model — invariants

- Every applied move is preceded by a backup. If backup fails, the move is aborted.
- Originals are **shelved, never deleted**. `RollbackService` restores them.
- Default backup TTL is 30 days (configurable).
- `FileIndex::upsert` preserves `FileId` UUIDs across re-scans (upsert-on-path).
- No file is moved without an approved `ChangeProposal` or `BundleProposal`. `--yes` auto-approves *moves* above a confidence threshold; rename decisions never auto-apply.
- **Bundles move atomically or not at all.** Same-volume: single atomic `rename()`. Cross-volume: copy-verify-delete with full rollback on any failure.

## Testing conventions

- **No mocking at module boundaries.** Tests use real SQLite, real fixtures, real filesystem (`tempfile`). Mocked tests at these seams drift from production behaviour.
- Unit tests: in-file `#[cfg(test)] mod tests` — fast, no models, no DB.
- Integration tests: `crates/<crate>/tests/*.rs` — real impls, slower.
- Fixture files live under `crates/<crate>/tests/fixtures/`.
- `cargo test --all-features` in CI — keep feature combos green.

## Operational rules

- **Models must load sequentially.** Qwen3 + embedding model concurrent load OOMs on 8GB machines. Onboarding flows await models in a single-threaded chain, not `tokio::join!`.
- **Dev profile runs deps at `opt-level=2`, main crate at 0.** Debug-mode inference is otherwise unusable.
- **Taxonomy embedding cache invalidates by hash of taxonomy text**, not by version number.
- **`ProfileCache` is keyed by target root** and rebuilds incrementally via `ScanDiff` against `FolderMetadata.content_hash`. Not by timestamp.

## Tier 3 LLM fallback

The pipeline accepts `Option<&dyn TextBackend>` and only consults it when
Tier 2 lands in the review zone (`needs_review = true`) AND
`config.enable_llm_fallback` is true. Three invariants future work must
preserve:

- **Three-gate activation.** The privacy model requires (a) the cargo
  feature compiled in (`--features llm-fallback` / `--features remote`),
  (b) the matching config bool (`[inference] llm_fallback = true`) or
  `[inference.remote]` section, and (c) the per-invocation flag
  (`--llm-fallback` / `--remote` or `TIDYUP_LLM_FALLBACK=1` /
  `TIDYUP_REMOTE=1`). The CLI rejects activation without the cargo
  feature; default builds and default invocations stay LLM-silent and
  network-silent. Don't shortcut the gate — don't auto-enable Tier 3 in
  default invocations, don't read `[inference] llm_fallback = true` from
  config alone, don't infer activation from environment heuristics.
- **`ServiceContext.text` is `Option`.** `None` is the privacy-preserving
  default. The pipeline's optional `text_backend` parameter is fed from
  `ctx.text.as_deref()`. A `NullTextBackend` stand-in is deliberately
  not used — absence is the signal, not a no-op trait object.
- **Tier 3 never produces renames.** The cascade calls
  `text_backend.classify_text(content, filename)` and re-embeds
  `category + tags + summary` for re-ranking, but the LLM's
  `suggested_name` field is deliberately dropped. The rename gate is
  driven by Tier 2's confidence, not the post-Tier-3 rerank score, so
  Tier 3 reroutes never produce renames — by design.

## Multimodal Tier 2 (Phase 7)

Image and audio classification are cross-modal contrastive lookups (SigLIP /
CLAP) — `tidyup-embeddings-ort::siglip` and `…::clap`. Three invariants
beyond the text Tier 2 rules:

- **Latent-space isolation.** `EmbeddingBackend`, `ImageEmbeddingBackend`,
  and `AudioEmbeddingBackend` produce vectors in disjoint latent spaces.
  Never cosine-compare across them. The pipeline keeps each modality's
  candidate list (`MultimodalContext.image.candidates`, `…audio.candidates`)
  separate from the text candidates so a misconfigured caller can't compute
  a cross-space cosine.
- **Optional inclusion, automatic detection.** SigLIP and CLAP backends live
  inside `tidyup-embeddings-ort` (no separate crate or feature gate — both
  are pure-Rust ONNX with disjoint preprocessing). They are loaded only when
  their bundles exist on disk, via `verify_siglip_model` /
  `verify_clap_model`. Missing bundles are NOT an error — image/audio files
  fall back to the text Tier 2 path, which falls back to Tier 1. Don't add a
  `--multimodal` runtime flag; presence of the artifacts is the gate.
- **Per-modality natural-language taxonomies.** The text taxonomy in
  `default_taxonomy()` is keyword-soup tuned for `bge-small`. The image and
  audio taxonomies in `default_image_taxonomy()` / `default_audio_taxonomy()`
  are natural-language captions ("a photograph of a person", "a podcast
  episode") because cross-modal contrastive encoders need that phrasing to
  compare image/audio embeddings against text embeddings. Don't reuse text
  taxonomy descriptions for image/audio.

## What NOT to do

- **Don't violate the privacy model.** No HTTP clients, LLM deps, or phone-home code paths in the default binary. Network-capable code lives only in `tidyup-inference-remote` behind `--features remote`. LLM code lives only in `tidyup-inference-mistralrs` behind `--features llm-fallback`.
- **Don't make LLM or remote inference a default.** Both `tidyup-inference-mistralrs` and `tidyup-inference-remote` are feature-gated off by default. Never change CLI defaults, config defaults, or build defaults to turn them on. First-run UX and default docs never recommend either.
- **Don't add generative rename paths.** Rename proposals are extractive only (embedded metadata → keyword-template fill → nearest-neighbor adapt → no-rename). No LLM-fabricated names, even under `--features llm-fallback`.
- **Don't introduce partial-bundle apply paths.** Bundles are atomic. No code that allows some members to move while others don't.
- **Don't auto-apply rename proposals.** Even under `--yes`, renames always surface in review.
- **Don't propose renames for bundle members.** Their internal structure is load-bearing.
- **Don't descend into detected bundles for per-file classification.** Bundle subtrees are opaque to the classifier.
- **Don't claim calibrated confidence in v0.1.** Confidence is raw weighted-cosine; calibration is a v0.2 story dependent on a held-out corpus that doesn't yet exist.
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
- **`CLASSIFICATION.md`** — update when the tier cascade, default thresholds, rename cascade, or per-modality coverage changes.
- **`DESIGN.md`** — update when UI/UX tokens, surface rules, component specs, or do/don't guidance change. Frontend work (`tidyup-ui` and any future frontend) should conform to it; deviations get reflected here.
- **`CLAUDE.md`** — update when a change establishes a new invariant or "don't do this" rule future work must obey. It's guidance, not a spec; don't mirror implementation details.

Rule of thumb: if someone reading a doc *today* would be misled by *yesterday's* code change, the doc is out of date. Don't wait to be asked.

## Reference docs in repo

- `ARCHITECTURE.md` — layer diagram, seam rationale, crate boundary justifications.
- `CLASSIFICATION.md` — three-tier cascade, embedding-default rationale, per-modality roadmap, rename extractive cascade.
- `DESIGN.md` — UI/UX design system ("The Verdant Archive"): colors, typography, elevation, component specs, do/don't rules.
- `README.md` — user-facing overview plus the phase-by-phase roadmap.
- `CONTRIBUTING.md` — PR checklist, commit style.
- `deny.toml` — license + source policy.
- `SECURITY.md` — reporting policy.
