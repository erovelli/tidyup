# tidyup-ui

Dioxus desktop UI for [tidyup](https://github.com/erovelli/tidyup). Consumes the same `tidyup-app` services and extractor/model context as the CLI binary, with signal-backed `ProgressReporter` and oneshot-channel `ReviewHandler` adapters. Dashboard Scan and Migrate start in **Preview only (dry run)** mode; the independent toggles must be disabled deliberately before the UI can apply approved proposals. Preview summaries say “would apply” and do not move files or write shelf/applied-state records. Raw semantic routing values are shown as similarity scores (for example, `similarity 0.81`), not probabilities; high/medium/low chips remain relative guidance.

The Review page overrides `review_all` to present loose changes and bundles as one diff-style organization plan. Semantic collection labels and member basenames are editable, duplicate sibling names fail inline, and approval remains atomic. “Files indexed” comes from identity-index progress rather than proposal count. Run summaries/history expose indexing failures and the persisted capability manifest. Settings surfaces the optional local LLM reranker as a session-only third gate after asynchronous config loading; remote inference is not wired into the desktop UI.

Not published to crates.io.
