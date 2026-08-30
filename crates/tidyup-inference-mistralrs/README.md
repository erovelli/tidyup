# tidyup-inference-mistralrs

Optional local inference backend for [tidyup](https://github.com/erovelli/tidyup) via [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Implements `TextBackend` and `VisionBackend` with runtime-selected CPU/Metal/CUDA support. The shipped CLI/UI wire only the text backend as a low-confidence reranker behind cargo-feature, config, and per-invocation gates; the vision port exists but is not part of the default semantic or rename path. LLM output never authors filenames.
