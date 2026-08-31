# tidyup-embeddings-ort

ONNX Runtime embedding backends for [tidyup](https://github.com/erovelli/tidyup):

- **`bge-small-en-v1.5`** (384-dim) — text-only, the required default semantic router. Implements `EmbeddingBackend` from `tidyup-core` and fails startup with installation guidance when its verified bundle is absent.
- **SigLIP-base INT8** (768-dim, optional, ~205 MB) — cross-modal image classifier. Implements `ImageEmbeddingBackend`. Loaded only when its checksum-verified ONNX bundle is present in the platform model cache; missing or invalid bundles degrade explicitly to general routing.
- **CLAP-htsat-unfused** (512-dim, optional) — cross-modal audio classifier. Implements `AudioEmbeddingBackend`. Same load-when-present semantics as SigLIP.

All three have Rust preprocessing/integration code backed by ONNX Runtime through `ort`, and emit L2-normalized vectors. Latent spaces are disjoint and not interchangeable — the pipeline keeps each modality's candidate list separate.

Also hosts the bundled scan-mode taxonomies (`default_taxonomy`, `default_image_taxonomy`, `default_audio_taxonomy`), natural-language folder-description helpers for cross-modal prototypes, the BLAKE3-keyed taxonomy embedding cache, and shared bundle specifications used by download-time and runtime verification. YAKE naming lives in `tidyup-pipeline`, not in this backend crate.

All remote model references and runtime preprocessing metadata live in the
checked-in [`models.toml`](models.toml) manifest. Artifact URLs use immutable
upstream revisions; production artifacts can additionally pin byte sizes and
BLAKE3 digests.
