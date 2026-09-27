//! Canonical OpenAI-compatible chat completion DTOs.
//!
//! Requests deserialize leniently: unknown fields are captured in `extra`
//! maps and re-serialized, so clients can send the same payload regardless
//! of the target provider (spec: "Canonical request schema"). Responses
//! serialize strictly to the canonical shape.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Canonical chat completion request (OpenAI `chat.completions` shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Lenient ingress: unknown fields are tolerated and re-serialized.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// A chat message. `reasoning_content` and any other unknown fields are
/// tolerated via `extra` (design D2 deferred thinking-field decision).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Message content: a plain string or an array of typed content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A typed content part: `text` or `image_url`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// `stop` as a single string or an array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    String(String),
    Array(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}

/// Static tool declaration (request `tools[].function`) and live tool call
/// payload (response `tool_calls[].function`) share the `name`/`arguments`
/// wire shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    /// Live call arguments (empty for request-side declarations).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A live tool call emitted by the model (response `tool_calls[]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: ToolCallFunction,
}

/// `tool_choice`: a mode string or a named tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named {
        #[serde(rename = "type")]
        tool_type: ToolType,
        function: NamedTool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTool {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
}

/// Canonical non-streaming chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Canonical streaming chunk (`object: "chat.completion.chunk"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Present only when the client requested `stream_options.include_usage`,
    /// on the final chunk (spec: "Usage attribution").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: MessageDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const REQUEST_FIXTURE: &str = r#"{
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png", "detail": "low"}}
                ]
            },
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }
                ],
                "reasoning_content": "Let me check."
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_completion_tokens": 256,
        "stop": ["STOP", "END"],
        "stream": true,
        "stream_options": {"include_usage": true},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object"}
                }
            }
        ],
        "tool_choice": "auto",
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "logprobs": true,
        "top_logprobs": 5,
        "seed": 42,
        "n": 1,
        "response_format": {"type": "json_object"},
        "user": "user-123",
        "top_k": 40
    }"#;

    const RESPONSE_FIXTURE: &str = r#"{
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there.",
                    "reasoning_content": "thinking..."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18
        }
    }"#;

    #[test]
    fn request_fixture_round_trips() {
        let request: ChatCompletionRequest =
            serde_json::from_str(REQUEST_FIXTURE).expect("fixture should deserialize");
        assert_eq!(request.model, "deepseek-chat");
        assert_eq!(request.messages.len(), 4);
        assert!(request.stream);
        assert_eq!(
            request
                .stream_options
                .as_ref()
                .map(|opts| opts.include_usage),
            Some(true)
        );
        assert!(request.logprobs.expect("logprobs"));
        assert_eq!(request.top_logprobs, Some(5));
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.n, Some(1));
        assert_eq!(request.user.as_deref(), Some("user-123"));

        // Array `stop` sequence.
        match request.stop.as_ref().expect("stop") {
            StopSequence::Array(values) => assert_eq!(values, &["STOP", "END"]),
            StopSequence::String(_) => panic!("stop should be an array"),
        }

        // Array content parts: text + image_url.
        let user = &request.messages[1];
        match user.content.as_ref().expect("content") {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    ContentPart::Text { text } => assert_eq!(text, "What is in this image?"),
                    ContentPart::ImageUrl { .. } => panic!("part 0 should be text"),
                }
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "https://example.com/x.png");
                        assert_eq!(image_url.detail.as_deref(), Some("low"));
                    }
                    ContentPart::Text { .. } => panic!("part 1 should be an image"),
                }
            }
            MessageContent::Text(_) => panic!("content should be parts"),
        }

        // Tool calls on the assistant message.
        let assistant = &request.messages[2];
        let tool_calls = assistant.tool_calls.as_ref().expect("tool_calls");
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(
            tool_calls[0].function.arguments.as_deref(),
            Some("{\"city\":\"Paris\"}")
        );

        // Unknown top-level field is tolerated (lenient ingress).
        assert_eq!(request.extra.get("top_k").and_then(Value::as_u64), Some(40));

        // Unknown message-level field (`reasoning_content`) is tolerated.
        assert_eq!(
            assistant
                .extra
                .get("reasoning_content")
                .and_then(Value::as_str),
            Some("Let me check.")
        );

        // Tool declaration.
        let tool = &request.tools.as_ref().expect("tools")[0];
        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(
            tool.function.description.as_deref(),
            Some("Get the weather")
        );
        match request.tool_choice.as_ref().expect("tool_choice") {
            ToolChoice::Mode(mode) => assert_eq!(mode, "auto"),
            ToolChoice::Named { .. } => panic!("tool_choice should be a mode"),
        }

        // Re-serialize: unknown fields must survive the round-trip.
        let serialized = serde_json::to_value(&request).expect("request should serialize");
        assert_eq!(serialized["top_k"], json!(40));
        assert_eq!(
            serialized["messages"][2]["reasoning_content"],
            "Let me check."
        );
        assert_eq!(serialized["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(serialized["stream_options"]["include_usage"], true);
    }

    #[test]
    fn minimal_request_applies_defaults() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "m", "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .expect("minimal request should deserialize");
        assert!(!request.stream, "stream defaults to false");
        assert!(request.stream_options.is_none());
        assert!(request.extra.is_empty());
        match request.messages[0].content.as_ref().expect("content") {
            MessageContent::Text(text) => assert_eq!(text, "hi"),
            MessageContent::Parts(_) => panic!("content should be text"),
        }
    }

    #[test]
    fn stream_options_include_usage_defaults_to_false() {
        let opts: StreamOptions = serde_json::from_str("{}").expect("empty options");
        assert!(!opts.include_usage);
    }

    #[test]
    fn response_fixture_round_trips() {
        let response: ChatCompletionResponse =
            serde_json::from_str(RESPONSE_FIXTURE).expect("fixture should deserialize");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1_700_000_000);
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.role, "assistant");
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
        assert_eq!(response.usage.prompt_tokens, 11);
        assert_eq!(response.usage.completion_tokens, 7);
        assert_eq!(response.usage.total_tokens, 18);
        assert_eq!(
            choice
                .message
                .extra
                .get("reasoning_content")
                .and_then(Value::as_str),
            Some("thinking...")
        );

        let serialized = serde_json::to_value(&response).expect("response should serialize");
        assert_eq!(serialized["object"], "chat.completion");
        assert_eq!(serialized["usage"]["total_tokens"], 18);
        // `reasoning_content` tolerated on the message survives round-trip.
        assert_eq!(
            serialized["choices"][0]["message"]["reasoning_content"],
            "thinking..."
        );
        // Strict egress: no unknown envelope fields appear.
        assert_eq!(serialized["id"], "chatcmpl-abc");
    }

    #[test]
    fn chunk_fixture_round_trips() {
        let chunk: ChatCompletionChunk = serde_json::from_str(
            r#"{
                "id": "chatcmpl-abc",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "deepseek-chat",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": null}
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            }"#,
        )
        .expect("chunk should deserialize");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices.len(), 1);
        let delta = &chunk.choices[0].delta;
        assert_eq!(delta.role.as_deref(), Some("assistant"));
        assert_eq!(delta.content.as_deref(), Some("Hel"));
        assert!(chunk.choices[0].finish_reason.is_none());
        assert_eq!(chunk.usage.as_ref().map(|u| u.total_tokens), Some(18));

        // Without usage the field is omitted on egress.
        let mut chunk = chunk;
        chunk.usage = None;
        let serialized = serde_json::to_value(&chunk).expect("chunk should serialize");
        assert!(serialized.get("usage").is_none());
    }

    #[test]
    fn stop_sequence_accepts_string() {
        let request: ChatCompletionRequest =
            serde_json::from_str(r#"{"model": "m", "messages": [], "stop": "HALT"}"#)
                .expect("request should deserialize");
        match request.stop.as_ref().expect("stop") {
            StopSequence::String(value) => assert_eq!(value, "HALT"),
            StopSequence::Array(_) => panic!("stop should be a string"),
        }
    }

    #[test]
    fn tool_choice_accepts_named_tool() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model": "m",
                "messages": [],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
            }"#,
        )
        .expect("request should deserialize");
        match request.tool_choice.as_ref().expect("tool_choice") {
            ToolChoice::Named {
                tool_type,
                function,
            } => {
                assert_eq!(*tool_type, ToolType::Function);
                assert_eq!(function.name, "get_weather");
            }
            ToolChoice::Mode(_) => panic!("tool_choice should be named"),
        }
    }
}
