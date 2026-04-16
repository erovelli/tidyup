# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue. Email **evan.rovelli@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any mitigations you've identified

You'll receive an acknowledgement within 72 hours. We aim to ship a fix and coordinated disclosure within 30 days of confirmation, depending on severity.

## Supported versions

During the pre-1.0 phase, only the latest minor release receives security updates.

## Supply chain

- Dependencies are audited via `cargo-deny` on every PR (see `deny.toml`).
- Binary releases are built in GitHub Actions with SLSA provenance where supported.
- No dependencies introduce `unsafe` code that we haven't reviewed.
