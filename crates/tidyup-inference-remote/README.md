# tidyup-inference-remote

Feature-gated HTTP `TextBackend` adapters for [tidyup](https://github.com/erovelli/tidyup): OpenAI-compatible APIs, Anthropic, and Ollama. They plug into the same low-confidence rerank seam as the optional local backend and never author filenames. All three `RemoteEndpoint` adapters exist here, but only the OpenAI-compatible path is currently selectable from CLI config (`RemoteBackendConfig` has no provider discriminator); an Ollama server works through its OpenAI-compatible `/v1` endpoint. This crate and its HTTP dependencies are absent from the default CLI graph.
