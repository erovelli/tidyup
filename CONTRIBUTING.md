# Contributing to tidyup

Thanks for your interest! A few ground rules.

## Quickstart

```bash
git clone https://github.com/erovelli/tidyup
cd tidyup
cargo xtask ci        # fmt + clippy + tests
cargo run -p tidyup-cli -- --help
```

## Architecture

Hexagonal. See `ARCHITECTURE.md`. The short version:

- `tidyup-domain` — pure types, zero deps
- `tidyup-core` — port traits (no impls)
- Impl crates depend on `tidyup-core`, never on each other
- `tidyup-app` — application services; the CLI/UI seam
- `tidyup-cli` / `tidyup-ui` — thin adapters implementing frontend ports

Before adding a new crate, ask: does it have (a) heavy disjoint deps, (b) feature-flag hygiene, or (c) binary-vs-lib separation? If none of those, add a module instead.

## PR checklist

- [ ] `cargo xtask ci` passes locally
- [ ] New public APIs documented
- [ ] `unwrap()` / `expect()` only in tests or with a `// SAFETY:`-style rationale
- [ ] No new direct deps without workspace-level entry in root `Cargo.toml`

## Commit style

Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`). One logical change per commit.

## Licensing

All contributions are dual-licensed under Apache-2.0 (see `LICENSE`). By submitting a PR you agree to license your work under those terms.
