# Contributing to tidyup

Thanks for your interest! A few ground rules.

## Quickstart

```bash
git clone https://github.com/erovelli/tidyup
cd tidyup
cargo xtask ci        # privacy check + fmt + clippy (-D warnings) + tests
cargo run -p tidyup-cli -- --help
```

## Prerequisites

- **Rust 1.95** — selected automatically via `rust-toolchain.toml`; no manual setup needed.
- **`cargo install cargo-deny cargo-hack`** — required by the supplemental gates `cargo xtask deny` (license/source policy) and `cargo xtask feature-matrix`.
- **ONNX Runtime (`libonnxruntime`)** — needed to build/run the embedding path (`tidyup-embeddings-ort`) and the model-dependent tests/eval. Fetch the model bundles with `cargo xtask download-models` before running them.

Two hard gates a PR must clear, both documented in `CLAUDE.md`:

- **The privacy model** — the default `tidyup-cli` (and `tidyup-ui`) build must stay network- and LLM-silent. `cargo xtask check-privacy` enforces it (it runs first inside `cargo xtask ci`).
- **The license allowlist** — only the licenses in `deny.toml` are permitted; GPL-family fails CI.

See `ARCHITECTURE.md`, `CLASSIFICATION.md`, and `DESIGN.md` before changing a crate boundary, the classifier, or the UI.

## Architecture

Hexagonal. See `ARCHITECTURE.md`. The short version:

- `tidyup-domain` — pure types, no tidyup-crate deps
- `tidyup-core` — port traits (no impls)
- Impl crates depend on `tidyup-core`, never on each other
- `tidyup-app` — application services; the CLI/UI seam
- `tidyup-cli` / `tidyup-ui` — thin adapters implementing frontend ports

Before adding a new crate, ask: does it have (a) heavy disjoint deps, (b) feature-flag hygiene, or (c) binary-vs-lib separation? If none of those, add a module instead.

## PR checklist

- [ ] `cargo xtask ci` passes locally
- [ ] New public items are documented
- [ ] No new `unwrap()` / `expect()` / `todo!()` outside tests (tests opt out with `#[allow(clippy::unwrap_used)]` at the `mod tests` level)
- [ ] New deps added to root `[workspace.dependencies]` (no per-crate version pins)
- [ ] Affected docs (`README.md` / `ARCHITECTURE.md` / `CLASSIFICATION.md` / `DESIGN.md`) updated in the same commit

This mirrors `.github/pull_request_template.md`; keep the two in sync.

## Commit style

Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`). One logical change per commit.

## Licensing

All contributions are licensed under Apache-2.0 (see `LICENSE`). Unless you explicitly state otherwise, any contribution you submit shall be licensed under those terms.
