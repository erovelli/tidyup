# tidyup

> A local-first file organizer that never phones home.

**tidyup** watches a directory, understands what's in your files using small language models running _entirely on your machine_, and proposes a tidier structure. You review every change before anything moves. Nothing is uploaded. Nothing is logged. No account. No cloud. No telemetry.

Your files stay where they belong — with you.

---

## Status: pre-alpha, not yet functional

> **Warning — tidyup is under active construction and not yet usable as a tool.**
>
> The foundational layers (domain types, SQLite storage, BLAKE3 indexer, layered config) are in place. The classification pipeline, content extractors, and inference backends are being built out now. The CLI compiles and dispatches, but **no command yet runs end-to-end** — `tidyup migrate`, `tidyup scan`, and `tidyup rollback` are stubs.
>
> This repository is being developed in the open. What you see reflects the target design, not current capability. Everything below describes what tidyup is being built to do — check the [roadmap](#roadmap) for what actually works today.
>
> Until v0.1 ships, please do not point tidyup at files you care about.

---

## Why this exists

Most "smart" file organizers are thin wrappers around someone else's API. You hand them your tax returns, your medical bills, your draft manuscripts, your half-written love letters — and trust that a privacy policy somewhere protects you.

tidyup takes a different stance:

- **Deterministic, offline classification.** The default binary classifies by embedding similarity via `bge-small-en-v1.5` on ONNX Runtime — entirely local, fully reproducible, no LLM generation. Classification is bit-for-bit stable across runs.
- **LLMs are an optional escape hatch, not a default.** The default binary contains no LLM inference — `mistralrs`/`candle`/`hf-hub` are not loaded, not linked, not reachable. A local LLM fallback is available as a compile-time + runtime opt-in (`--features llm-fallback`) for cold-start cases (empty target hierarchy, pathological extraction).
- **No network calls by default.** The default binary has no HTTP client — not loaded, not linked, not reachable. Airplane-mode your machine and it still works. A remote backend is available as a symmetric compile-time + runtime opt-in (`--features remote`) for power users.
- **Groupings stay grouped.** A coding project, photo burst, or music album moves as an atomic bundle or not at all — tidyup will never fragment one.
- **Content-aware renames, but never silent.** When a filename clearly disagrees with the contents, tidyup proposes a rename — driven by tunable confidence thresholds. Renames always require explicit approval.
- **No telemetry.** No analytics, no crash reporting, no "anonymous usage data."
- **Human-in-the-loop, always.** Every rename and move is a _proposal_. Nothing touches your filesystem until you approve it.
- **Every change is reversible.** Originals are copied to a backup shelf before any move. Restore anything within 30 days (configurable).

This is a portfolio project and a personal tool. It is also a statement: useful AI does not require surrendering your data.

---

## What it does

- **Indexes a directory** into a local SQLite database with BLAKE3 content hashing and dedup. Identical contents are classified once, regardless of how many copies exist.
- **Detects logical groupings first.** Coding projects, photo bursts, music albums, Jupyter notebook sets, document series — tidyup recognizes these as bundles via structural markers (`.git/`, `Cargo.toml`, `package.json`, consistent EXIF timestamps, matching ID3 album tags, etc.) and moves them as atomic units. A coding project is never shredded; either the whole tree relocates or nothing does.
- **Classifies each loose file by its contents** via a three-tier cascade, cheapest first:
  1. **Heuristics** (~1ms) — extension, MIME, keyword rules.
  2. **Embeddings** (~50ms, default) — cosine similarity against learned target-folder profiles via `bge-small-en-v1.5` on ONNX Runtime. Deterministic, auditable, offline.
  3. **Local LLM fallback** (1–10s, optional) — available only with `--features llm-fallback`, off by default. Provides a second opinion on files that fall below Tier 2 confidence thresholds. Default builds exclude this tier entirely; low-confidence files surface directly to review.
- **Proposes a destination folder** — with a plain-English reason.
- **Proposes a rename** when filename and contents disagree — using two tunable signals (classification confidence × filename-content mismatch). Renames never auto-apply even with `--yes`; they always go through explicit review.
- **Shows you a diff-style review UI** — approve, edit, or reject per file or per bundle.
- **Backs up originals before moving** — restore anything, anytime.

Two modes:

- **Scan mode** — organize a messy directory against a built-in taxonomy.
- **Migration mode** — sort a source directory into an _existing_ target hierarchy whose structure tidyup learns: it profiles each target folder (name + content centroid + organizational type: semantic / date-based / project-based / status-based) and routes new files to where they semantically fit.

---

## Privacy guarantees

| Guarantee             | How it's enforced                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| No network inference (default) | Default binary is built without `tidyup-inference-remote`. No HTTP client, no `reqwest`/`hyper`/`rustls` linked. `cargo-deny` bans network deps outside the `remote` feature. |
| No LLM inference (default) | Default binary is built without `tidyup-inference-mistralrs`. No `mistralrs`/`candle`/`hf-hub`/heavy tokenizer tree linked. `cargo-deny` bans these deps outside the `llm-fallback` feature. Classification is deterministic embedding similarity only. |
| No telemetry          | No analytics crate in `Cargo.toml`. CI check blocks additions.                                        |
| No background uploads | App has no cloud sync feature, by design.                                                             |
| Local-only storage    | SQLite DB + config + backups all under platform data dir.                                             |
| Reversible by default | Every move is preceded by a copy to the backup shelf.                                                 |
| Atomic bundle moves   | Coding projects, photo albums, and other groupings move as a single unit or not at all.                |
| Extractive renames only | Rename proposals come from embedded metadata, keyword-template fill, or nearest-neighbor adaptation — never fabricated. Structurally incapable of generating a name without evidence. |

**Remote inference and LLM fallback are symmetric power-user opt-ins**, not defaults. Each requires the same two-gate opt-in: (a) compile with the feature flag (`--features remote` or `--features llm-fallback`), (b) enable in config (`[inference] backends = ["remote-..."]` or `[inference] llm_fallback = true`), and (c) pass the per-invocation flag (`--remote` or `--llm-fallback`). First-run and onboarding never recommend either; the tool is designed to be excellent offline with embedding-based classification as the spine.

If you find a privacy claim here that doesn't match the code, that's a bug — please open an issue.

---

## Building

```bash
# Default release build — fully offline, embedding-only classification.
# No LLM inference, no HTTP client compiled in.
cargo build --release -p tidyup-cli
cargo run   --release -p tidyup-cli -- --help

# Power user: include the optional local LLM fallback backend.
# Enables mistralrs/candle/hf-hub deps. Still requires runtime config + --llm-fallback flag.
cargo build --release -p tidyup-cli --features llm-fallback

# With hardware acceleration for the optional LLM fallback
cargo build --release -p tidyup-cli --features "llm-fallback metal"   # macOS
cargo build --release -p tidyup-cli --features "llm-fallback cuda"    # NVIDIA

# Power user: include the optional remote inference backend.
# Enables HTTP client deps (reqwest, rustls). Still requires runtime config + --remote flag.
cargo build --release -p tidyup-cli --features remote
```

Rust pinned to 1.90 via `rust-toolchain.toml`. The default-binary embedding model (~35 MB) is downloaded on first launch and cached under the platform data dir; subsequent runs are fully offline.

---

## Roadmap

tidyup is being built in phases. Each phase lands an independently compilable slice of the hexagonal architecture. The CLI binary exists from Phase 0, but doesn't become end-to-end runnable until Phase 5.

| Phase | Scope                                                                                       | Status         |
| ----- | ------------------------------------------------------------------------------------------- | -------------- |
| 0     | Workspace scaffold, port traits, CI, lints, `deny.toml`, `xtask`                            | [x] Complete   |
| 1     | Domain types, SQLite storage, BLAKE3 indexer, layered config, `BundleProposal` aggregate    | [x] Complete   |
| 2     | Content extractors: router + MIME detection, plain text, PDF, Excel, image, audio           | [~] In progress |
| 3     | Inference: `bge-small-en-v1.5` via ONNX Runtime (default); optional LLM + remote backends   | [ ] Not started |
| 4     | Pipeline: heuristics, bundle detection, scan + migration classifiers, rename cascade        | [ ] Not started |
| 5     | CLI wiring, first-run model download, end-to-end flows, v0.1 ship                           | [ ] Not started |
| 6+    | Multimodal encoders (image/audio/video), Dioxus UI, code signing, `brew`/`winget`, plugins  | [ ] Backlog    |

**What currently works:**

- `cargo build -p tidyup-cli` produces a binary with fully stubbed subcommands
- `cargo xtask ci` is green: `fmt` + `clippy --all-features -D warnings` + workspace tests
- SQLite storage: `FileIndex`, `ChangeLog`, `BackupStore` with bundle-atomic shelving
- Layered TOML config with platform-aware paths
- `tidyup-extract`: MIME detection + router + `PlainTextExtractor`

**What does not yet work:**

- Classification of any file (requires Phase 3 embeddings)
- Move/rollback execution (requires Phase 4 pipeline + Phase 5 CLI wiring)
- Bundle detection and atomic apply
- Any content extraction for PDF, Excel, image, or audio files
- Any LLM or remote inference (both feature-gated off and not yet implemented)

The invariants the finished tool will uphold — human-in-the-loop review, reversible moves, bundle atomicity, no-network-by-default, extractive-only renames — are enforced in the design today, but the code paths that would violate them don't exist yet.

---

## Contributing

This is primarily a personal project, but issues, discussions, and PRs are welcome. If you're considering a substantial change, please open an issue first so we can talk about fit.

See `CONTRIBUTING.md` for setup and style guidelines.

---

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).

Apache-2.0 is a permissive license: you can use, modify, and redistribute tidyup — including in commercial and closed-source projects — provided you preserve the copyright notice and the `NOTICE` file. It also includes an explicit patent grant from contributors, protecting you and downstream users.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in tidyup shall be licensed as above, without any additional terms or conditions.
