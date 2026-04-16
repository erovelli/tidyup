//! HTTP-based inference backends.
//!
//! Useful for:
//! - Contributors without local accelerators (CI, low-end laptops).
//! - Users who prefer frontier models.
//! - Proving the backend registry works — the first *new* backend that wasn't in docorg.

// TODO: OpenAICompatibleText, AnthropicText, OllamaText — all behind one `RemoteText` enum
//       selected by `config.inference.remote.endpoint`.
