//! Mock LLM adapter. this just sends garbage for e2e testing.
//! @human

use serde_json::Value;

use crate::error::ProxyError;
use crate::models::chat::{ChatCompletionRequest, ChatCompletionResponse};
use crate::providers::capabilities::{
    ByteStream, CapabilitySet, ChunkStream, DropAction, FieldDrop, Translator, diff_request,
};
pub mod server;
mod stream;

#[derive(Clone)]
pub struct MockLLMTranslator {
    caps: CapabilitySet,
    model: String,
    base_url: String,
}

impl MockLLMTranslator {
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::with_capabilites(
            model,
            base_url,
            CapabilitySet {
                seed: false,
                top_k: false,
                logprobs: true,
                tools: true,
                vision: false,
                parallel_tool_calls: false,
                max_tokens_ceiling: Some(8192),
            },
        )
    }

    pub fn with_capabilites(
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
    /// name — the client-facing namespaced id (`mockllm/<model>`) must not
    /// leak onto the wire. Returns the possibly-mutated wire JSON plus the
    /// drops.
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

impl Translator for MockLLMTranslator {
    fn capabilities(&self) -> &CapabilitySet {
        &self.caps
    }

    fn encode_request(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<(serde_json::Value, Vec<FieldDrop>), ProxyError> {
        self.encode(req)
    }

    fn decode_response(
        &self,
        body: &serde_json::Value,
    ) -> Result<ChatCompletionResponse, ProxyError> {
        serde_json::from_value(body.clone())
            .map_err(|e| ProxyError::Translation(format!("mockllm response decode: {e}")))
    }

    fn decode_stream(&self, include_usage: bool, byte_stream: ByteStream) -> ChunkStream {
        stream::decode(include_usage, byte_stream)
    }
}
