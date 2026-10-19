//! DeepSeek adapter. DeepSeek speaks native OpenAI-compatible wire format,
//! so this is a near-passthrough translator (design D1): encoding is serde
//! serialization of the canonical request with the capability diff applied,
//! and decoding deserializes provider JSON straight into the canonical DTOs.
//! `reasoning_content` on messages/deltas survives via the DTOs' lenient
//! `extra` flatten (design D2).

use serde_json::Value;

use crate::error::ProxyError;
use crate::models::chat::{ChatCompletionRequest, ChatCompletionResponse};
use crate::providers::capabilities::{
    ByteStream, CapabilitySet, ChunkStream, DropAction, FieldDrop, Translator, diff_request,
};

/// Near-passthrough translator for DeepSeek's OpenAI-compatible API.
#[derive(Clone)]
pub struct DeepSeekTranslator {
    caps: CapabilitySet,
    /// Upstream model name to send (may differ from the client-facing alias).
    model: String,
    /// Upstream base URL (e.g. `https://api.deepseek.com`).
    base_url: String,
}

impl DeepSeekTranslator {
    /// Build a translator for `model` at `base_url`.
    ///
    /// Capability values below are a placeholder config for this change —
    /// they should move to provider configuration when the registry lands
    /// (Stage C).
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::with_capabilities(
            model,
            base_url,
            CapabilitySet {
                seed: false,
                top_k: false,
                logprobs: true,
                tools: true,
                vision: false,
                parallel_tool_calls: true,
                // Placeholder ceiling; real per-model limits come from config.
                max_tokens_ceiling: Some(8192),
            },
        )
    }

    /// Build a translator with an explicit capability set. Used by the
    /// registry's test constructors so wiremock tests can exercise
    /// drop/clamp paths (e.g. a `logprobs: false` provider) without env vars.
    pub fn with_capabilities(
        model: impl Into<String>,
        base_url: impl Into<String>,
        caps: CapabilitySet,
    ) -> Self {
        Self {
            caps,
            model: model.into(),
            base_url: base_url.into(),
        }
    }

    /// Upstream model name sent on the wire.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Upstream base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Serialize the canonical request, then apply the drop/clamp diff:
    /// remove dropped fields and overwrite clamped fields with the applied
    /// value. The `model` field is rewritten to the bare upstream model
    /// name — the client-facing namespaced id (`deepseek/<model>`) must not
    /// leak onto the provider wire. Returns the possibly-mutated wire JSON
    /// plus the drops.
    fn encode(&self, req: &ChatCompletionRequest) -> Result<(Value, Vec<FieldDrop>), ProxyError> {
        let drops = diff_request(&self.caps, req);
        let mut wire = serde_json::to_value(req)
            .map_err(|e| ProxyError::Translation(format!("deepseek request encode: {e}")))?;
        let obj = wire.as_object_mut().ok_or_else(|| {
            ProxyError::Internal("encoded request was not a JSON object".to_string())
        })?;
        obj.insert("model".to_string(), Value::from(self.model.clone()));
        for drop in &drops {
            match drop.action {
                DropAction::Dropped => {
                    obj.remove(&drop.field);
                }
                DropAction::Clamped { applied } => {
                    if let Some(slot) = obj.get_mut(&drop.field) {
                        *slot = Value::from(applied);
                    }
                }
            }
        }
        Ok((wire, drops))
    }
}

impl Translator for DeepSeekTranslator {
    fn capabilities(&self) -> &CapabilitySet {
        &self.caps
    }

    fn encode_request(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<(Value, Vec<FieldDrop>), ProxyError> {
        self.encode(req)
    }

    fn decode_response(&self, body: &Value) -> Result<ChatCompletionResponse, ProxyError> {
        serde_json::from_value(body.clone())
            .map_err(|e| ProxyError::Translation(format!("deepseek response decode: {e}")))
    }

    fn decode_stream(&self, include_usage: bool, byte_stream: ByteStream) -> ChunkStream {
        stream::decode(include_usage, byte_stream)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::providers::capabilities::DropAction;

    fn translator() -> DeepSeekTranslator {
        DeepSeekTranslator::new("deepseek-chat", "https://api.deepseek.com")
    }

    fn request_from(value: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("fixture should deserialize")
    }

    #[test]
    fn exposes_config() {
        let t = translator();
        assert_eq!(t.model(), "deepseek-chat");
        assert_eq!(t.base_url(), "https://api.deepseek.com");
        assert!(!t.capabilities().seed);
        assert_eq!(t.capabilities().max_tokens_ceiling, Some(8192));
    }

    #[test]
    fn unsupported_seed_is_removed_and_budget_is_clamped() {
        let req = request_from(json!({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 42,
            "max_completion_tokens": 100_000
        }));
        let (wire, drops) = translator().encode_request(&req).expect("encode");

        assert!(wire.get("seed").is_none(), "seed must be dropped");
        assert_eq!(
            wire["max_completion_tokens"],
            json!(8192),
            "budget must be clamped to the ceiling"
        );
        assert_eq!(
            drops,
            vec![
                FieldDrop::dropped("seed"),
                FieldDrop::clamped("max_completion_tokens", 8192),
            ]
        );
    }

    #[test]
    fn supported_request_round_trips_unchanged() {
        let req = request_from(json!({
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "hi"}
            ],
            "temperature": 0.7,
            "max_completion_tokens": 256,
            "stream": false,
            "logprobs": true,
            "top_logprobs": 5
        }));
        let (wire, drops) = translator().encode_request(&req).expect("encode");

        assert!(drops.is_empty(), "no drops for a fully supported request");
        // Everything is passthrough EXCEPT the model: the namespaced
        // client-facing id must be rewritten to the bare upstream model name.
        let mut expected = serde_json::to_value(&req).expect("canonical request should serialize");
        expected["model"] = json!("deepseek-chat");
        assert_eq!(wire, expected);
        assert_eq!(wire["model"], json!("deepseek-chat"));
    }

    #[test]
    fn message_level_reasoning_content_survives_encoding() {
        let req = request_from(json!({
            "model": "deepseek-chat",
            "messages": [{
                "role": "assistant",
                "content": null,
                "reasoning_content": "thinking...",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"}
                }]
            }]
        }));
        let (wire, drops) = translator().encode_request(&req).expect("encode");
        assert!(drops.is_empty());
        assert_eq!(
            wire["messages"][0]["reasoning_content"], "thinking...",
            "lenient extra fields must pass through to the wire"
        );
    }

    #[test]
    fn decode_response_parses_canned_deepseek_body() {
        let body = json!({
            "id": "chatcmpl-deepseek-1",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there.",
                    "reasoning_content": "thinking..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18
            }
        });
        let response = translator().decode_response(&body).expect("decode");
        assert_eq!(response.id, "chatcmpl-deepseek-1");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(response.usage.total_tokens, 18);
        assert_eq!(
            response.choices[0]
                .message
                .extra
                .get("reasoning_content")
                .and_then(serde_json::Value::as_str),
            Some("thinking...")
        );
    }

    #[test]
    fn decode_response_maps_serde_errors_to_translation() {
        let body = json!({"unexpected": "shape"});
        let err = translator().decode_response(&body).expect_err("must fail");
        match err {
            ProxyError::Translation(msg) => assert!(msg.contains("deepseek response decode")),
            other => panic!("expected Translation, got {other}"),
        }
    }

    #[test]
    fn encode_request_always_succeeds_on_dto_shaped_requests() {
        // Every field of the canonical DTO is `serde_json`-representable
        // (non-finite floats degrade to null), so the Translation arm of
        // `encode` is defensive; assert the happy path stays infallible.
        let mut req = request_from(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        req.temperature = Some(f64::NAN);
        let (_wire, drops) = translator().encode_request(&req).expect("encodes");
        assert!(drops.is_empty());
    }

    #[test]
    fn clamp_drops_only_affect_reported_fields() {
        // A clamp on `max_completion_tokens` must not disturb neighboring
        // fields on the wire.
        let req = request_from(json!({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "max_completion_tokens": 999_999
        }));
        let (wire, drops) = translator().encode_request(&req).expect("encode");
        assert_eq!(
            drops,
            vec![FieldDrop {
                field: "max_completion_tokens".to_string(),
                action: DropAction::Clamped { applied: 8192 },
            }]
        );
        assert_eq!(wire["temperature"], json!(0.5));
        assert_eq!(wire["max_completion_tokens"], json!(8192));
    }
}

pub mod stream;
