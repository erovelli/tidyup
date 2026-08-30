# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue. Email **evan.rovelli@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any mitigations you've identified

You'll receive an acknowledgement within 72 hours. We aim to ship a fix and coordinated disclosure within 30 days of confirmation, depending on severity.

**Privacy and local-state regressions are in scope.** tidyup's load-bearing promise is that the default build is network- and LLM-silent. The application does intentionally persist a local SQLite audit/index database, capability manifests, semantic artifacts, and rollback shelves under its data directory. Unexpected network activity, data leaving the machine, unsafe overwrite/delete behavior, or sensitive material appearing outside those documented local stores should be reported privately here rather than in a public issue.

## Supported versions

The repository contains tagged-release automation for checksummed CLI archives, but the project remains pre-alpha. During pre-1.0 development, only the newest published `0.x` line receives security fixes; source builds from older commits are unsupported.

## Supply chain

- Dependencies are audited via `cargo-deny` on every PR (see `deny.toml`).
- Binary releases are built in GitHub Actions; each artifact ships with a SHA-256 checksum (`taiki-e/upload-rust-binary-action`).
- Workspace crates compile with `unsafe_code = "forbid"`. Third-party crates may contain unsafe internals (notably runtime/database/system adapters); `cargo-deny`, pinned versions, source policy, and CI cover their accepted supply-chain surface.
