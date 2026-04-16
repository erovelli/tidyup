# tidyup

> A local-first file organizer that never phones home.

**tidyup** watches a directory, understands what's in your files using small language models running _entirely on your machine_, and proposes a tidier structure. You review every change before anything moves. Nothing is uploaded. Nothing is logged. No account. No cloud. No telemetry.

Your files stay where they belong — with you.

---

## Why this exists

Most "smart" file organizers are thin wrappers around someone else's API. You hand them your tax returns, your medical bills, your draft manuscripts, your half-written love letters — and trust that a privacy policy somewhere protects you.

tidyup takes a different stance:

- **Inference runs locally.** Quantized LLMs (Qwen3-0.6B for text, SmolVLM-256M for images) execute on your CPU or local GPU.
- **No network calls.** The binary has no HTTP client for inference. You can airplane-mode your machine and it still works.
- **No telemetry.** No analytics, no crash reporting, no "anonymous usage data."
- **Human-in-the-loop, always.** Every rename and move is a _proposal_. Nothing touches your filesystem until you approve it in the UI.
- **Every change is reversible.** Originals are copied to a backup shelf before any move. Restore anything within 30 days (configurable).

This is a portfolio project and a personal tool. It is also a statement: useful AI does not require surrendering your data.

---

## What it does

- **Indexes a directory** into a local SQLite database with SHA256 dedup.
- **Classifies each file** via a three-tier cascade, cheapest first:
  1. **Heuristics** (~1ms) — extension, MIME, keyword rules.
  2. **Embeddings** (~50ms) — cosine similarity against a pre-computed taxonomy.
  3. **Local LLM** (1–10s) — Qwen3 for text, SmolVLM for images/video, lofty for audio metadata.
- **Proposes a rename and a destination folder** — with a plain-English reason.
- **Shows you a diff-style review UI** — approve, edit, or reject per file.
- **Backs up originals before moving** — restore anything, anytime.

Two modes:

- **Scan mode** — organize a messy directory from scratch.
- **Migration mode** — sort new files into an _existing_ folder hierarchy whose structure it learns from your layout.

---

## Privacy guarantees

| Guarantee             | How it's enforced                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------- |
| No network inference  | Binary ships without an HTTP client for model calls. `cargo-deny` rule to block adding one. |
| No telemetry          | No analytics crate in `Cargo.toml`. CI check blocks additions.                              |
| No background uploads | App has no cloud sync feature, by design.                                                   |
| Local-only storage    | SQLite DB + config + backups all under platform data dir.                                   |
| Reversible by default | Every move is preceded by a copy to the backup shelf.                                       |

If you find a privacy claim here that doesn't match the code, that's a bug — please open an issue.

---

## Building

```bash
# Release build (recommended — local inference is slow in debug)
cargo build --release
cargo run --release

# With hardware acceleration
cargo run --release --features metal   # macOS
cargo run --release --features cuda    # NVIDIA
```

Rust stable, 1.XX+. No other runtime dependencies — models are downloaded on first launch and cached locally.

---

## Contributing

This is primarily a personal project, but issues, discussions, and PRs are welcome. If you're considering a substantial change, please open an issue first so we can talk about fit.

See `CONTRIBUTING.md` for setup and style guidelines.

---

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).

Apache-2.0 is a permissive license: you can use, modify, and redistribute tidyup — including in commercial and closed-source projects — provided you preserve the copyright notice and the `NOTICE` file. It also includes an explicit patent grant from contributors, protecting you and downstream users.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in tidyup shall be licensed as above, without any additional terms or conditions.
