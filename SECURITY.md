# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue. Email **evan.rovelli@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any mitigations you've identified

You'll receive an acknowledgement within 72 hours. We aim to ship a fix and coordinated disclosure within 30 days of confirmation, depending on severity.

**Privacy regressions are in scope.** tidyup's load-bearing promise is that the default build is network- and LLM-silent. If you find the default binary making an unexpected network call or otherwise leaking data, please report it privately here rather than opening a public issue.

## Supported versions

No releases have shipped yet. Once they do, only the most recent `0.x` line receives security fixes during the pre-1.0 phase.

## Supply chain

- Dependencies are audited via `cargo-deny` on every PR (see `deny.toml`).
- Binary releases are built in GitHub Actions; each artifact ships with a SHA-256 checksum (`taiki-e/upload-rust-binary-action`).
- No dependencies introduce `unsafe` code that we haven't reviewed.
