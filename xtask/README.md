# xtask

Internal workspace automation for [tidyup](https://github.com/erovelli/tidyup). Run via `cargo xtask <task>`. Replaces shell scripts with cross-platform Rust. Not published.

```bash
cargo xtask ci             # privacy-check + fmt + clippy + test
cargo xtask fmt
cargo xtask lint
cargo xtask deny
cargo xtask feature-matrix
cargo xtask check-privacy  # asserts default tidyup-cli graph has no banned crates
cargo xtask download-models [--multimodal | --siglip | --clap]
```
