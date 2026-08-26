//! HTTP-based [`TextBackend`](tidyup_core::inference::TextBackend) —
//! OpenAI-compatible, Anthropic, Ollama.
//!
//! # Privacy model
//!
//! This crate is **excluded from default builds** via `optional = true` on the
//! `tidyup-cli` dep. The default release binary has no HTTP client
//! (`reqwest` / `hyper` / `rustls`) linked. Activation is triple-gated:
//! `--features remote` at build time AND an `[inference.remote]` section in
//! config AND `--remote` or `TIDYUP_REMOTE=1` at runtime. (The `[inference]
//! backends` list is parsed for forward-compat but not consulted for
//! activation.) See `CLAUDE.md#privacy-model`.
//!
//! # Shape
//!
//! One [`RemoteText`] struct dispatched by [`RemoteEndpoint`]. Each variant
//! translates the shared `TextBackend` surface to the provider's native HTTP
//! shape and feeds the result through
//! [`parse_content_classification`](tidyup_core::inference::parse_content_classification)
//! for tolerant JSON decoding.

use std::time::Duration;

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
/// Give up establishing a TCP/TLS connection after this long.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Give up on a whole request/response after this long, so a stalled server
/// can never hang classification indefinitely (Tier 3 then falls back to the
/// Tier-2 verdict).
const REQUEST_TIMEOUT: Duration = Duration::from_mins(1);

/// Redacting wrapper for an API key.
///
/// Keeps the secret out of `Debug`, `Display`, and any serialized output — the
/// raw value is reachable only via [`Secret::expose`]. This upholds the "the
/// API key is never written to disk" promise even if a [`RemoteEndpoint`] is
/// accidentally logged or serialized: the derived `Debug`/`Serialize` on the
/// enum inherit this masking.
#[derive(Clone, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    /// The raw secret. Call only at the HTTP boundary (the auth header).
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl From<String> for Secret {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Secret {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl Serialize for Secret {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Never emit the real key — masks it in any serialized form. This is
        // deliberately asymmetric with the transparent `Deserialize`: a
        // `RemoteEndpoint` is always built fresh from the `api_key_env` var at
        // invocation and is never persisted+reloaded, so a live key never
        // round-trips through this masking. Don't add a serialize-then-reload
        // path for a populated endpoint — the key would come back as "***".
        serializer.serialize_str("***")
    }
}

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
        api_key: Secret,
        model: String,
    },
    /// Anthropic Messages API. `api_base` defaults to
    /// `https://api.anthropic.com/v1/messages`.
    Anthropic {
        api_key: Secret,
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

/// Reject an endpoint whose URL is malformed or would send an API key over
/// cleartext `http`. Key-carrying providers (`OpenAI`, `Anthropic`) require
/// `https` unless the host is loopback; Ollama (no key) may use plain `http`.
fn validate_endpoint(endpoint: &RemoteEndpoint) -> Result<()> {
    match endpoint {
        RemoteEndpoint::OpenAi { url, .. } => require_valid_url(url, true),
        // `None` uses the hard-coded https `ANTHROPIC_API_BASE`, so it's safe.
        RemoteEndpoint::Anthropic { api_base, .. } => api_base
            .as_ref()
            .map_or_else(|| Ok(()), |base| require_valid_url(base, true)),
        RemoteEndpoint::Ollama { url, .. } => require_valid_url(url, false),
    }
}

fn require_valid_url(raw: &str, carries_secret: bool) -> Result<()> {
    let url = reqwest::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid remote endpoint URL {raw:?}: {e}"))?;
    match url.scheme() {
        "https" => Ok(()),
        "http" if !carries_secret || is_loopback_host(&url) => Ok(()),
        "http" => Err(anyhow::anyhow!(
            "refusing to send the API key over cleartext http to {raw:?}; use https \
             (plain http is allowed only for a loopback host)"
        )),
        other => Err(anyhow::anyhow!(
            "unsupported URL scheme {other:?} for {raw:?}; use https"
        )),
    }
}

fn is_loopback_host(url: &reqwest::Url) -> bool {
    // `Url::host_str` returns the *bracketed* form for IPv6, so loopback `::1`
    // arrives as "[::1]".
    matches!(
        url.host_str(),
        Some("localhost" | "127.0.0.1" | "::1" | "[::1]")
    )
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
        validate_endpoint(&endpoint)?;
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            // No redirects: `validate_endpoint` only vetted the configured URL,
            // and reqwest does NOT strip a custom `x-api-key` header (Anthropic)
            // on a cross-host redirect the way it strips `Authorization`. A
            // compromised endpoint could otherwise 302 the key to cleartext
            // http. Chat APIs don't redirect, so refuse to follow any.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| anyhow::anyhow!("build reqwest client: {e}"))?;
        let model_id = format!("{}/{}", endpoint.model_id_prefix(), endpoint.model());
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
            } => {
                openai::chat(
                    &self.client,
                    url,
                    api_key.expose(),
                    model,
                    system,
                    user,
                    opts,
                )
                .await
            }
            RemoteEndpoint::Anthropic {
                api_key,
                model,
                api_base,
            } => {
                let base = api_base.as_deref().unwrap_or(ANTHROPIC_API_BASE);
                anthropic::chat(
                    &self.client,
                    base,
                    api_key.expose(),
                    model,
                    system,
                    user,
                    opts,
                )
                .await
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
            .map(|(i, c)| format!("Frame {}: {c}", i.saturating_add(1)))
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
    fn serialize_redacts_openai_api_key() {
        let ep = RemoteEndpoint::OpenAi {
            url: "https://api.openai.com/v1".into(),
            api_key: "sk-supersecret".into(),
            model: "gpt-4o".into(),
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("\"kind\":\"openai\""));
        assert!(
            !json.contains("sk-supersecret"),
            "raw key must never be serialized: {json}"
        );
        assert!(json.contains("***"), "key field should be masked: {json}");
    }

    #[test]
    fn serialize_redacts_anthropic_api_key() {
        let ep = RemoteEndpoint::Anthropic {
            api_key: "secret-key-value".into(),
            model: "claude-sonnet-4-6".into(),
            api_base: None,
        };
        let json = serde_json::to_string(&ep).unwrap();
        assert!(json.contains("\"kind\":\"anthropic\""));
        assert!(!json.contains("secret-key-value"), "raw key leaked: {json}");
        assert!(json.contains("***"), "key field should be masked: {json}");
    }

    #[test]
    fn rejects_cleartext_http_anthropic_api_base() {
        // The Anthropic `api_base` override must be vetted too — its key rides
        // in an `x-api-key` header, so cleartext http to a non-loopback host
        // is refused.
        let err = RemoteText::new(RemoteEndpoint::Anthropic {
            api_key: "k".into(),
            model: "m".into(),
            api_base: Some("http://proxy.example.com/v1/messages".into()),
        })
        .unwrap_err();
        assert!(err.to_string().contains("cleartext http"), "got: {err}");

        // Loopback IPv6 over http is allowed (fixture for the "[::1]" arm).
        assert!(RemoteText::new(RemoteEndpoint::Anthropic {
            api_key: "k".into(),
            model: "m".into(),
            api_base: Some("http://[::1]:8080/v1/messages".into()),
        })
        .is_ok());
    }

    #[test]
    fn debug_redacts_api_key() {
        let ep = RemoteEndpoint::OpenAi {
            url: "https://api.openai.com/v1".into(),
            api_key: "sk-supersecret".into(),
            model: "gpt-4o".into(),
        };
        let rendered = format!("{ep:?}");
        assert!(
            !rendered.contains("sk-supersecret"),
            "Debug must not leak the key: {rendered}"
        );
        assert!(rendered.contains("***"));
    }

    #[test]
    fn endpoint_serde_roundtrip_ollama() {
        // Ollama has no secret field, so it roundtrips cleanly.
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

    #[test]
    fn rejects_cleartext_http_for_key_carrying_endpoint() {
        let err = RemoteText::new(RemoteEndpoint::OpenAi {
            url: "http://api.example.com/v1".into(),
            api_key: "sk-x".into(),
            model: "m".into(),
        })
        .unwrap_err();
        assert!(err.to_string().contains("cleartext http"), "got: {err}");
    }

    #[test]
    fn allows_https_and_loopback_http_with_key() {
        assert!(RemoteText::new(RemoteEndpoint::OpenAi {
            url: "https://api.openai.com/v1".into(),
            api_key: "sk-x".into(),
            model: "m".into(),
        })
        .is_ok());
        // Loopback http is allowed even with a key (local proxy / dev server).
        assert!(RemoteText::new(RemoteEndpoint::OpenAi {
            url: "http://localhost:8080/v1".into(),
            api_key: "sk-x".into(),
            model: "m".into(),
        })
        .is_ok());
        // Ollama over plain http (carries no key) is fine.
        assert!(RemoteText::new(RemoteEndpoint::Ollama {
            url: "http://localhost:11434".into(),
            model: "llama3.2".into(),
        })
        .is_ok());
    }

    #[test]
    fn rejects_malformed_endpoint_url() {
        let err = RemoteText::new(RemoteEndpoint::OpenAi {
            url: "not a url".into(),
            api_key: "sk-x".into(),
            model: "m".into(),
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("invalid remote endpoint URL"),
            "got: {err}"
        );
    }
}
