# tidyup-embeddings-ort

ONNX Runtime embedding backends for [tidyup](https://github.com/erovelli/tidyup):

- **`bge-small-en-v1.5`** (384-dim) — text-only, the default Tier 2 classifier. Always loaded. Implements `EmbeddingBackend` from `tidyup-core`.
- **SigLIP-base** (768-dim, optional) — cross-modal image classifier. Implements `ImageEmbeddingBackend`. Loaded only when its ONNX bundle is present in the platform model cache; missing bundle is not an error.
- **CLAP-htsat-unfused** (512-dim, optional) — cross-modal audio classifier. Implements `AudioEmbeddingBackend`. Same load-when-present semantics as SigLIP.

All three are pure-Rust (no FFI beyond `ort`/`image`/`symphonia`) and emit L2-normalized vectors. Latent spaces are disjoint and not interchangeable — the pipeline keeps each modality's candidate list separate.

Also hosts the bundled scan-mode taxonomies (`default_taxonomy`, `default_image_taxonomy`, `default_audio_taxonomy`), inline YAKE keyword extraction, the BLAKE3-keyed taxonomy embedding cache, and the `verify_*_model` first-run installation checks.
