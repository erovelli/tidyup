# tidyup-core

Port traits for the [tidyup](https://github.com/erovelli/tidyup) hexagonal architecture.

- **Frontend ports:** `ProgressReporter`, `ReviewHandler`, `ConfigProvider`.
- **Inference ports:** `TextBackend`, `VisionBackend`, `EmbeddingBackend`, plus the cross-modal `ImageEmbeddingBackend` / `AudioEmbeddingBackend` — each embedding family stays in its own latent space.
- **Storage ports:** `FileIndex`, `ChangeLog`, `BackupStore`, `RunLog`.
- **Extractor port:** `ContentExtractor`.

`ReviewHandler::review_all` is the complete-plan seam; its default composes loose and bundle review sequentially, while richer frontends may override it. `review_bundles` returns approved proposals so semantic labels/member basenames can be edited, but the application executor—not the frontend—owns aggregate integrity.

The storage port includes single and batched run-scoped processing-state operations, versioned semantic-artifact lookup, and cache pruning. `ContentExtractor::capability_id` gives the application an inspectable implementation identity for each run manifest. The crate also exposes the `ContentClassification` shape returned by optional `TextBackend`s, its tolerant JSON decoder, and shared reranker prompts.

Implementation crates depend on `tidyup-core` — they never depend on each other.
