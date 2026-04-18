//! HTTP-based [`TextBackend`](tidyup_core::inference::TextBackend) —
//! OpenAI-compatible, Anthropic, Ollama.
//!
//! # Privacy model
//!
//! This crate is **excluded from default builds** via `optional = true` on the
//! `tidyup-cli` dep. The default release binary has no HTTP client
//! (`reqwest` / `hyper` / `rustls`) linked. Inclusion requires `--features
//! remote` at build time AND `[inference] backends = ["remote-..."]` in
//! config AND `--remote` or `TIDYUP_REMOTE=1` at runtime. See
//! `CLAUDE.md#privacy-model`.
//!
//! # Shape
//!
//! One [`RemoteText`] struct dispatched by [`RemoteEndpoint`]. Each variant
//! translates the shared `TextBackend` surface to the provider's native HTTP
//! shape and feeds the result through
//! [`parse_content_classification`](tidyup_core::inference::parse_content_classification)
//! for tolerant JSON decoding.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tidyup_core::inference::{
    parse_content_classification, prompts, ContentClassification, GenerationOptions, TextBackend,
};
use tidyup_core::Result;

const USER_AGENT: &str = concat!("tidyup/", env!("CARGO_PKG_VERSION"));
const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 256;
const CLASSIFY_MAX_TOKENS: u32 = 300;
const CLASSIFY_TEMPERATURE: f32 = 0.1;

// ---------------------------------------------------------------------------
// Endpoint config
// ---------------------------------------------------------------------------

/// HTTP inference endpoint. One variant per supported provider shape.
///
/// The endpoint carries its own credentials (api key, base url) so a
/// [`RemoteText`] instance is self-contained — no ambient config lookups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RemoteEndpoint {
    /// `OpenAI` or any `OpenAI`-compatible `/v1/chat/completions` endpoint
    /// (`OpenRouter`, `Together`, `vLLM`, `LM Studio`, …). `url` is the API
    /// base, e.g. `https://api.openai.com/v1`.
    #[serde(rename = "openai")]
    OpenAi {
        url: String,
        api_key: String,
        model: String,
    },
    /// Anthropic Messages API. `api_base` defaults to
    /// `https://api.anthropic.com/v1/messages`.
    Anthropic {
        api_key: String,
        model: String,
        #[serde(default)]
        api_base: Option<String>,
    },
    /// Ollama `/api/chat` endpoint. `url` is the Ollama server root,
    /// e.g. `http://localhost:11434`.
    Ollama { url: String, model: String },
}

impl RemoteEndpoint {
    const fn model_id_prefix(&self) -> &'static str {
        match self {
            Self::OpenAi { .. } => "remote-openai",
            Self::Anthropic { .. } => "remote-anthropic",
            Self::Ollama { .. } => "remote-ollama",
        }
    }

    fn model(&self) -> &str {
        match self {
            Self::OpenAi { model, .. }
            | Self::Anthropic { model, .. }
            | Self::Ollama { model, .. } => model,
        }
    }
}

// ---------------------------------------------------------------------------
// RemoteText — one TextBackend across all providers
// ---------------------------------------------------------------------------

/// `TextBackend` implementation dispatched across HTTP providers.
///
/// Always routes responses through
/// [`parse_content_classification`](tidyup_core::inference::parse_content_classification)
/// so provider-specific pre/postamble (markdown fences, `<think>` blocks) is
/// normalized before returning.
pub struct RemoteText {
    endpoint: RemoteEndpoint,
    client: reqwest::Client,
    model_id: String,
}

impl std::fmt::Debug for RemoteText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteText")
            .field("endpoint", &self.endpoint.model_id_prefix())
            .field("model", &self.endpoint.model())
            .finish_non_exhaustive()
    }
}

impl RemoteText {
    /// Construct a `RemoteText` from an endpoint config.
    ///
    /// The inner `reqwest::Client` is built once and reused across calls for
    /// connection-pool efficiency.
    pub fn new(endpoint: RemoteEndpoint) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;
        let model_id = format!("{}/{}", endpoint.model_id_prefix(), endpoint.model(),);
        Ok(Self {
            endpoint,
            client,
            model_id,
        })
    }

    async fn chat(&self, system: Option<&str>, user: &str, opts: &ChatOpts) -> Result<String> {
        match &self.endpoint {
            RemoteEndpoint::OpenAi {
                url,
                api_key,
                model,
            } => openai::chat(&self.client, url, api_key, model, system, user, opts).await,
            RemoteEndpoint::Anthropic {
                api_key,
                model,
                api_base,
            } => {
                let base = api_base.as_deref().unwrap_or(ANTHROPIC_API_BASE);
                anthropic::chat(&self.client, base, api_key, model, system, user, opts).await
            }
            RemoteEndpoint::Ollama { url, model } => {
                ollama::chat(&self.client, url, model, system, user, opts).await
            }
        }
    }

    async fn classify(&self, system: &str, user: &str) -> Result<ContentClassification> {
        let opts = ChatOpts {
            max_tokens: CLASSIFY_MAX_TOKENS,
            temperature: CLASSIFY_TEMPERATURE,
        };
        let content = self.chat(Some(system), user, &opts).await?;
        parse_content_classification(&content)
    }
}

struct ChatOpts {
    max_tokens: u32,
    temperature: f32,
}

// ---------------------------------------------------------------------------
// TextBackend impl
// ---------------------------------------------------------------------------

#[async_trait]
impl TextBackend for RemoteText {
    async fn classify_text(&self, text: &str, filename: &str) -> Result<ContentClassification> {
        let user = format!("Filename: {filename}\n\nContent:\n{text}");
        self.classify(prompts::TEXT_CLASSIFY_SYSTEM, &user).await
    }

    async fn classify_audio(
        &self,
        filename: &str,
        metadata: &str,
    ) -> Result<ContentClassification> {
        let user = format!("Filename: {filename}\nMetadata:\n{metadata}");
        self.classify(prompts::AUDIO_CLASSIFY_SYSTEM, &user).await
    }

    async fn classify_video(
        &self,
        filename: &str,
        frame_captions: &[String],
    ) -> Result<ContentClassification> {
        let captions = frame_captions
            .iter()
            .enumerate()
            .map(|(i, c)| format!("Frame {}: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!("Filename: {filename}\n\nFrame descriptions:\n{captions}");
        self.classify(prompts::VIDEO_CLASSIFY_SYSTEM, &user).await
    }

    async fn classify_image_description(
        &self,
        filename: &str,
        description: &str,
    ) -> Result<ContentClassification> {
        let user = format!("Filename: {filename}\nImage description: {description}");
        self.classify(prompts::IMAGE_CLASSIFY_SYSTEM, &user).await
    }

    async fn complete(&self, prompt: &str, opts: &GenerationOptions) -> Result<String> {
        let chat_opts = ChatOpts {
            max_tokens: if opts.max_tokens == 0 {
                DEFAULT_MAX_TOKENS
            } else {
                opts.max_tokens
            },
            temperature: opts.temperature,
        };
        self.chat(None, prompt, &chat_opts).await
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible adapter
// ---------------------------------------------------------------------------

#[allow(unreachable_pub)]
mod openai {
    use super::{ChatOpts, Result};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize)]
    struct ChatRequest<'a> {
        model: &'a str,
        messages: Vec<Message<'a>>,
        max_tokens: u32,
        temperature: f32,
    }

    #[derive(Serialize)]
    struct Message<'a> {
        role: &'a str,
        content: &'a str,
    }

    #[derive(Deserialize)]
    struct ChatResponse {
        choices: Vec<Choice>,
    }

    #[derive(Deserialize)]
    struct Choice {
        message: ChoiceMessage,
    }

    #[derive(Deserialize)]
    struct ChoiceMessage {
        content: Option<String>,
    }

    pub async fn chat(
        client: &reqwest::Client,
        url: &str,
        api_key: &str,
        model: &str,
        system: Option<&str>,
        user: &str,
        opts: &ChatOpts,
    ) -> Result<String> {
        let mut messages = Vec::with_capacity(2);
        if let Some(sys) = system {
            messages.push(Message {
                role: "system",
                content: sys,
            });
        }
        messages.push(Message {
            role: "user",
            content: user,
        });

        let body = ChatRequest {
            model,
            messages,
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
        };

        let endpoint = format!("{}/chat/completions", url.trim_end_matches('/'));
        let response = client
            .post(&endpoint)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("POST {endpoint}: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let snippet = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI-compatible endpoint returned {status}: {snippet}",
            ));
        }

        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("decode OpenAI response: {e}"))?;
        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("empty OpenAI response"))
    }
}

// ---------------------------------------------------------------------------
// Anthropic adapter
// ---------------------------------------------------------------------------

#[allow(unreachable_pub)]
mod anthropic {
    use super::{ChatOpts, Result, ANTHROPIC_VERSION};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize)]
    struct MessagesRequest<'a> {
        model: &'a str,
        max_tokens: u32,
        temperature: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        system: Option<&'a str>,
        messages: Vec<Message<'a>>,
    }

    #[derive(Serialize)]
    struct Message<'a> {
        role: &'a str,
        content: &'a str,
    }

    #[derive(Deserialize)]
    struct MessagesResponse {
        content: Vec<ContentBlock>,
    }

    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum ContentBlock {
        Text {
            text: String,
        },
        #[serde(other)]
        Other,
    }

    pub async fn chat(
        client: &reqwest::Client,
        api_base: &str,
        api_key: &str,
        model: &str,
        system: Option<&str>,
        user: &str,
        opts: &ChatOpts,
    ) -> Result<String> {
        let body = MessagesRequest {
            model,
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
            system,
            messages: vec![Message {
                role: "user",
                content: user,
            }],
        };

        let response = client
            .post(api_base)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("POST {api_base}: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let snippet = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Anthropic endpoint returned {status}: {snippet}",
            ));
        }

        let parsed: MessagesResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("decode Anthropic response: {e}"))?;
        parsed
            .content
            .into_iter()
            .find_map(|b| match b {
                ContentBlock::Text { text } => Some(text),
                ContentBlock::Other => None,
            })
            .ok_or_else(|| anyhow::anyhow!("empty Anthropic response"))
    }
}

// ---------------------------------------------------------------------------
// Ollama adapter
// ---------------------------------------------------------------------------

#[allow(unreachable_pub)]
mod ollama {
    use super::{ChatOpts, Result};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize)]
    struct ChatRequest<'a> {
        model: &'a str,
        messages: Vec<Message<'a>>,
        stream: bool,
        options: Options,
    }

    #[derive(Serialize)]
    struct Options {
        num_predict: u32,
        temperature: f32,
    }

    #[derive(Serialize)]
    struct Message<'a> {
        role: &'a str,
        content: &'a str,
    }

    #[derive(Deserialize)]
    struct ChatResponse {
        message: ResponseMessage,
    }

    #[derive(Deserialize)]
    struct ResponseMessage {
        content: String,
    }

    pub async fn chat(
        client: &reqwest::Client,
        url: &str,
        model: &str,
        system: Option<&str>,
        user: &str,
        opts: &ChatOpts,
    ) -> Result<String> {
        let mut messages = Vec::with_capacity(2);
        if let Some(sys) = system {
            messages.push(Message {
                role: "system",
                content: sys,
            });
        }
        messages.push(Message {
            role: "user",
            content: user,
        });

        let body = ChatRequest {
            model,
            messages,
            stream: false,
            options: Options {
                num_predict: opts.max_tokens,
                temperature: opts.temperature,
            },
        };

        let endpoint = format!("{}/api/chat", url.trim_end_matches('/'));
        let response = client
            .post(&endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("POST {endpoint}: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let snippet = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Ollama endpoint returned {status}: {snippet}"
            ));
        }

        let parsed: ChatResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("decode Ollama response: {e}"))?;
        Ok(parsed.message.content)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_model_id_prefix() {
        let ep = RemoteEndpoint::OpenAi {
            url: "https://api.openai.com/v1".into(),
            api_key: "sk-x".into(),
            model: "gpt-4o".into(),
        };
        assert_eq!(ep.model_id_prefix(), "remote-openai");
        assert_eq!(ep.model(), "gpt-4o");
    }

    #[test]
    fn endpoint_serde_roundtrip_openai() {
        let ep = RemoteEndpoint::OpenAi {
            url: "https://api.openai.com/v1".into(),
            api_key: "sk-x".into(),
            model: "gpt-4o".into(),
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("\"kind\":\"openai\""));
        let back: RemoteEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ep);
    }

    #[test]
    fn endpoint_serde_roundtrip_anthropic() {
        let ep = RemoteEndpoint::Anthropic {
            api_key: "key".into(),
            model: "claude-sonnet-4-6".into(),
            api_base: None,
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("\"kind\":\"anthropic\""));
        let back: RemoteEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ep);
    }

    #[test]
    fn endpoint_serde_roundtrip_ollama() {
        let ep = RemoteEndpoint::Ollama {
            url: "http://localhost:11434".into(),
            model: "llama3.2".into(),
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("\"kind\":\"ollama\""));
        let back: RemoteEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ep);
    }

    #[test]
    fn model_id_includes_prefix_and_model() {
        let backend = RemoteText::new(RemoteEndpoint::Ollama {
            url: "http://localhost:11434".into(),
            model: "llama3.2".into(),
        })
        .unwrap();
        assert_eq!(backend.model_id(), "remote-ollama/llama3.2");
    }
}
