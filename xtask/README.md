# xtask

Internal workspace automation for [tidyup](https://github.com/erovelli/tidyup). Run via `cargo xtask <task>`. Replaces shell scripts with cross-platform Rust. Not published.

```bash
cargo xtask ci             # privacy + fmt check + layered clippy/test matrix
cargo xtask fmt
cargo xtask lint
cargo xtask deny
cargo xtask feature-matrix
cargo xtask check-privacy  # asserts default tidyup-cli graph has no banned crates
cargo xtask download-models [--multimodal | --siglip | --clap]
cargo xtask verify-models [--multimodal | --siglip | --clap]
cargo xtask eval [--json] [--no-model] [--calibrate]
cargo xtask eval-routing <organized-corpus> [--json] [--fail-under 0.05]
cargo xtask bench-semantic path/to/image.png [--iterations 20] [--fail-over-ms 1000]
```

`bench-semantic` reports cold SigLIP load separately and enforces the warm
read + hash + image embedding + concept-ranking + grounded-name p95 latency budget. It covers that representative path only; production per-stage spans/deadlines remain planned.

`eval` measures the labeled taxonomy corpus and can fit a development Platt calibrator; without the required BGE bundle, semantic entries are deferred. `eval-routing` is the model-backed falsification harness for “content beats filename”: it learns folder centroids from an already-organized corpus and compares held-out content routing with filename, most-frequent, and extension baselines using bootstrap confidence intervals. Neither model-backed command is part of `cargo xtask ci`.
