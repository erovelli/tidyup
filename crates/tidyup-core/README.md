# tidyup-core

Port traits for the [tidyup](https://github.com/erovelli/tidyup) hexagonal architecture. Defines `ProgressReporter`, `ReviewHandler`, `ConfigProvider` (frontend ports) and `TextBackend`, `VisionBackend`, `EmbeddingBackend`, `FileIndex`, `ChangeLog`, `BackupStore`, `ContentExtractor` (backend ports).

Implementation crates depend on `tidyup-core` — they never depend on each other.
