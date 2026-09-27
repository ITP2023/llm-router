//! MockLLM upstream server: a fake OpenAI-compatible endpoint for end-to-end
//! testing without network access or API keys. This is the *upstream* half
//! of the MockLLM provider; the router-side half is
//! [`super::MockLLMTranslator`]. Pointing a registry built with
//! [`crate::state::ModelRegistry::mockllm`] at a spawned server exercises the
//! full request path: router handler -> reqwest -> HTTP -> mock upstream ->
//! SSE/JSON bytes -> translator decoder -> canonical response.
//!
//! # Wire behavior
//!
//! - `POST /chat/completions` accepts the canonical request shape and answers
//!   deterministically: the assistant content echoes the last user message
//!   text prefixed with `mockllm: `. Token counts derive from the request.
//! - `stream: true` yields an SSE body: role delta, content deltas of
//!   [`Config::chunk_size`] chars, `finish_reason: "stop"` on the last delta,
//!   a usage chunk when `stream_options.include_usage` is set, and a
//!   `data: [DONE]` terminator — the canonical OpenAI streaming shape.
//! - Error injection: a last user message of the form `mock-error-<status>`
//!   makes the upstream answer with that HTTP status and an OpenAI-style
//!   error body, exercising the router's non-2xx mapping end to end.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::json;

use crate::models::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice,
    ChunkChoice, ContentPart, MessageContent, MessageDelta, Usage,
};

/// Deterministic `created` timestamp for all mock responses.
pub const MOCK_CREATED: u64 = 1_700_000_000;

/// Prefix of the last-user-message trigger for injected upstream errors.
pub const ERROR_TRIGGER_PREFIX: &str = "mock-error-";

/// Behavior knobs for the mock upstream.
#[derive(Debug, Clone)]
pub struct Config {
    /// Characters per content delta in streaming responses.
    pub chunk_size: usize,
    /// Delay between streamed deltas; simulates upstream generation latency
    /// (and makes "first chunk arrives before stream end" observable e2e).
    pub chunk_delay: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_size: 8,
            chunk_delay: Duration::ZERO,
        }
    }
}

/// Handle to a spawned mock upstream. Stopping is graceful: the shutdown
/// signal is sent, the server task is awaited briefly, then aborted.
pub struct MockUpstream {
    base_url: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl MockUpstream {
    /// Base URL the router should use as the provider base (includes scheme).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Signal shutdown and reap the server task.
    pub async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if tokio::time::timeout(Duration::from_secs(2), &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
        }
    }
}

/// Bind an ephemeral loopback port and serve the mock upstream.
pub async fn spawn(config: Config) -> std::io::Result<MockUpstream> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let router = Router::new()
        .route("/chat/completions", post(chat_completions))
        .with_state(Arc::new(config));
    let task = tokio::spawn(async move {
        let shutdown = async {
            let _ = rx.await;
        };
        if let Err(err) = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown)
            .await
        {
            tracing::error!(error = %err, "mockllm upstream server error");
        }
    });
    Ok(MockUpstream {
        base_url: format!("http://{addr}"),
        shutdown: Some(tx),
        task,
    })
}

async fn chat_completions(
    State(config): State<Arc<Config>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(parsed) => parsed,
        Err(rejection) => {
            // A real upstream answers bad JSON with a 400; the router's
            // non-2xx mapping path is exercised the same way.
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {"message": rejection.body_text(), "type": "invalid_request_error"}
                })),
            )
                .into_response();
        }
    };

    if let Some(status) = injected_error_status(&request) {
        let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        return (
            status,
            Json(json!({
                "error": {"message": format!("mock upstream error {status}"), "type": "server_error"}
            })),
        )
            .into_response();
    }

    let content = canned_content(&request);
    let prompt_tokens = prompt_chars(&request) as u64;
    let completion_tokens = content.chars().count() as u64;
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

    if request.stream {
        let include_usage = request
            .stream_options
            .as_ref()
            .map(|opts| opts.include_usage)
            .unwrap_or(false);
        sse_response(&request.model, &content, usage, include_usage, &config)
    } else {
        let response = ChatCompletionResponse {
            id: "chatcmpl-mockllm".to_string(),
            object: "chat.completion".to_string(),
            created: MOCK_CREATED,
            model: request.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(MessageContent::Text(content)),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    extra: Default::default(),
                },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage,
        };
        Json(response).into_response()
    }
}

/// Split content into delta strings of at most `size` chars (`size >= 1`).
fn content_chunks(content: &str, size: usize) -> Vec<String> {
    let size = size.max(1);
    content
        .chars()
        .collect::<Vec<_>>()
        .chunks(size)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn sse_response(
    model: &str,
    content: &str,
    usage: Usage,
    include_usage: bool,
    config: &Config,
) -> Response {
    let chunks = content_chunks(content, config.chunk_size);
    let delay = config.chunk_delay;
    let model = model.to_string();
    let stream = async_stream::stream! {
        let count = chunks.len();
        for (index, chunk) in chunks.iter().enumerate() {
            if !delay.is_zero() && index > 0 {
                tokio::time::sleep(delay).await;
            }
            let last = index + 1 == count;
            let frame = ChatCompletionChunk {
                id: "chatcmpl-mockllm".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: MOCK_CREATED,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: MessageDelta {
                        role: (index == 0).then(|| "assistant".to_string()),
                        content: Some(chunk.clone()),
                        tool_calls: None,
                        extra: Default::default(),
                    },
                    finish_reason: last.then(|| "stop".to_string()),
                    logprobs: None,
                }],
                usage: None,
            };
            match serde_json::to_string(&frame) {
                Ok(json) => yield Ok::<_, std::io::Error>(format!("data: {json}\n\n")),
                Err(err) => {
                    yield Err(std::io::Error::new(std::io::ErrorKind::InvalidData, err));
                    return;
                }
            }
        }
        if include_usage {
            let usage_frame = ChatCompletionChunk {
                id: "chatcmpl-mockllm".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: MOCK_CREATED,
                model: model.clone(),
                choices: Vec::new(),
                usage: Some(usage),
            };
            match serde_json::to_string(&usage_frame) {
                Ok(json) => yield Ok(format!("data: {json}\n\n")),
                Err(err) => {
                    yield Err(std::io::Error::new(std::io::ErrorKind::InvalidData, err));
                    return;
                }
            }
        }
        yield Ok("data: [DONE]\n\n".to_string());
    };
    match Response::builder()
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from_stream(stream))
    {
        Ok(response) => response,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("mockllm: failed to build SSE response: {err}"),
        )
            .into_response(),
    }
}

/// Text of the last user message, concatenating text parts when the content
/// is an array of parts.
fn last_user_text(request: &ChatCompletionRequest) -> Option<String> {
    let message = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")?;
    match message.content.as_ref()? {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => Some(
            parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        ),
    }
}

/// Deterministic assistant content: echo the last user text.
fn canned_content(request: &ChatCompletionRequest) -> String {
    match last_user_text(request) {
        Some(text) => format!("mockllm: {}", text.chars().take(2000).collect::<String>()),
        None => "mockllm: hello".to_string(),
    }
}

/// `mock-error-<status>` in the last user message triggers that status.
fn injected_error_status(request: &ChatCompletionRequest) -> Option<u16> {
    let text = last_user_text(request)?;
    let code = text.trim().strip_prefix(ERROR_TRIGGER_PREFIX)?;
    code.parse::<u16>()
        .ok()
        .filter(|status| (400..=599).contains(status))
}

/// Deterministic prompt accounting: total message text length in chars.
fn prompt_chars(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .filter_map(|message| match message.content.as_ref() {
            Some(MessageContent::Text(text)) => Some(text.chars().count()),
            Some(MessageContent::Parts(parts)) => Some(
                parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text { text } => Some(text.chars().count()),
                        ContentPart::ImageUrl { .. } => None,
                    })
                    .sum(),
            ),
            None => None,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request_with(text: &str) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "mock-chat",
            "messages": [{"role": "user", "content": text}]
        }))
        .expect("fixture deserializes")
    }

    #[test]
    fn content_echoes_last_user_text() {
        let request = request_with("hello e2e");
        assert_eq!(canned_content(&request), "mockllm: hello e2e");
    }

    #[test]
    fn content_without_user_message_is_fixed() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "system", "content": "only a system message"}]
        }))
        .expect("fixture deserializes");
        assert_eq!(canned_content(&request), "mockllm: hello");
    }

    #[test]
    fn error_trigger_parses_valid_status() {
        assert_eq!(
            injected_error_status(&request_with("mock-error-500")),
            Some(500)
        );
        assert_eq!(
            injected_error_status(&request_with("  mock-error-429  ")),
            Some(429)
        );
    }

    #[test]
    fn error_trigger_ignores_garbage() {
        assert_eq!(injected_error_status(&request_with("mock-error-abc")), None);
        assert_eq!(injected_error_status(&request_with("mock-error-99")), None);
        assert_eq!(injected_error_status(&request_with("mock-error-700")), None);
        assert_eq!(
            injected_error_status(&request_with("just a question")),
            None
        );
    }

    #[test]
    fn chunking_splits_on_char_boundaries() {
        assert_eq!(content_chunks("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        assert_eq!(content_chunks("hi", 8), vec!["hi"]);
        assert_eq!(content_chunks("", 8), Vec::<String>::new());
        assert_eq!(
            content_chunks("abc", 0),
            vec!["a", "b", "c"],
            "size 0 clamps to 1"
        );
    }

    #[tokio::test]
    async fn spawned_server_serves_non_streaming_round_trip() {
        let upstream = spawn(Config::default()).await.expect("spawns");
        let base = upstream.base_url().to_string();

        let http = reqwest::Client::new();
        let response = http
            .post(format!("{base}/chat/completions"))
            .json(&json!({
                "model": "mock-chat",
                "messages": [{"role": "user", "content": "ping"}]
            }))
            .send()
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = response.json().await.expect("JSON body");
        assert_eq!(body["choices"][0]["message"]["content"], "mockllm: ping");
        assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);
        upstream.stop().await;
    }

    #[tokio::test]
    async fn spawned_server_streams_sse_with_usage_and_done() {
        let upstream = spawn(Config {
            chunk_size: 4,
            ..Config::default()
        })
        .await
        .expect("spawns");
        let base = upstream.base_url().to_string();

        let http = reqwest::Client::new();
        let response = http
            .post(format!("{base}/chat/completions"))
            .json(&json!({
                "model": "mock-chat",
                "messages": [{"role": "user", "content": "abcdefgh"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }))
            .send()
            .await
            .expect("request succeeds");

        assert_eq!(response.status(), StatusCode::OK);
        let text = response.text().await.expect("body reads");
        let data_frames: Vec<&str> = text
            .split("\n\n")
            .filter_map(|block| block.strip_prefix("data: "))
            .collect();
        assert_eq!(data_frames.last(), Some(&"[DONE]"));

        let mut reassembled = String::new();
        let mut usage_frames = 0;
        for frame in &data_frames[..data_frames.len() - 1] {
            let chunk: ChatCompletionChunk = serde_json::from_str(frame).expect("frame parses");
            if let Some(usage) = chunk.usage {
                usage_frames += 1;
                assert_eq!(
                    usage.completion_tokens,
                    "mockllm: abcdefgh".chars().count() as u64
                );
            } else {
                if let Some(content) = chunk.choices[0].delta.content.as_ref() {
                    reassembled.push_str(content);
                }
            }
        }
        assert_eq!(reassembled, "mockllm: abcdefgh");
        assert_eq!(usage_frames, 1);
        upstream.stop().await;
    }

    #[tokio::test]
    async fn spawned_server_injects_error_status() {
        let upstream = spawn(Config::default()).await.expect("spawns");
        let base = upstream.base_url().to_string();

        let http = reqwest::Client::new();
        let response = http
            .post(format!("{base}/chat/completions"))
            .json(&json!({
                "model": "mock-chat",
                "messages": [{"role": "user", "content": "mock-error-503"}]
            }))
            .send()
            .await
            .expect("request succeeds");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        upstream.stop().await;
    }
}
