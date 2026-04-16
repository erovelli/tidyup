# xtask

Internal workspace automation for [tidyup](https://github.com/erovelli/tidyup). Run via `cargo xtask <task>`. Replaces shell scripts with cross-platform Rust. Not published.

```bash
cargo xtask ci             # fmt + clippy + test
cargo xtask fmt
cargo xtask lint
cargo xtask deny
cargo xtask feature-matrix
```
