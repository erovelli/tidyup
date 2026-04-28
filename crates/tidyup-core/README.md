# tidyup-core

Port traits for the [tidyup](https://github.com/erovelli/tidyup) hexagonal architecture.

- **Frontend ports:** `ProgressReporter`, `ReviewHandler`, `ConfigProvider`.
- **Inference ports:** `TextBackend`, `VisionBackend`, `EmbeddingBackend`, plus the cross-modal `ImageEmbeddingBackend` / `AudioEmbeddingBackend` (Phase 7) — each in its own latent space.
- **Storage ports:** `FileIndex`, `ChangeLog`, `BackupStore`, `RunLog`.
- **Extractor port:** `ContentExtractor`.

Also exposes the `ContentClassification` shape every `TextBackend` returns, the tolerant `parse_content_classification` JSON decoder, and shared system-prompt constants used by Tier 3 backends.

Implementation crates depend on `tidyup-core` — they never depend on each other.
