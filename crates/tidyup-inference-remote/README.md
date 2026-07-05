# tidyup-inference-remote

HTTP-based inference backends for [tidyup](https://github.com/erovelli/tidyup): OpenAI-compatible APIs, Anthropic, Ollama. Useful for contributors without local accelerators or users who prefer frontier models. Implements `TextBackend` from `tidyup-core`. Note: all three `RemoteEndpoint` adapters exist in this crate, but only the OpenAI-compatible path is currently selectable from CLI config (`RemoteBackendConfig` has no provider discriminator yet); an Ollama server works via its OpenAI-compatible `/v1` endpoint.
