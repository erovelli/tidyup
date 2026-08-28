# tidyup-pipeline

Classification pipelines for [tidyup](https://github.com/erovelli/tidyup):

- **scan-mode** — semantic embeddings with optional LLM refinement against a fixed taxonomy.
- **migration-mode** — target-aware: learns folder profiles from an existing user hierarchy.

Built atop the port traits from `tidyup-core`. Has no knowledge of which concrete backends are wired in.
