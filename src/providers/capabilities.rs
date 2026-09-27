//! Translator seam: capability declarations, drop/clamp bookkeeping, and the
//! `Translator` trait (design D3, spec: "Provider capability declaration").
//!
//! The request-time drop set is computed by diffing the canonical request
//! against the selected adapter's [`CapabilitySet`]. The resulting
//! [`FieldDrop`]s feed the `x-dropped-request-fields` header and, later,
//! routing decisions (design D3: "this request needs vision").

use std::fmt;

use futures_util::stream::BoxStream;
use serde::Serialize;

use crate::error::ProxyError;
use crate::models::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};

/// Raw upstream response body bytes as streamed by reqwest.
pub type ByteStream = BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>;

/// Canonical chunk stream produced by a [`Translator`]'s streaming decoder.
pub type ChunkStream = BoxStream<'static, Result<ChatCompletionChunk, ProxyError>>;

/// Typed per-adapter capability declaration. `false` means the provider does
/// not support the feature; request fields for it are dropped or clamped per
/// the drop/clamp-and-advertise policy. `max_tokens_ceiling: None` means
/// unbounded/unenforced.
///
/// Struct-update syntax with `..CapabilitySet::default()` leaves
/// not-yet-relevant features at their safe default (`false`), so new fields
/// are room to grow without touching every adapter.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CapabilitySet {
    /// The provider honors the `seed` request field.
    pub seed: bool,
    /// The provider honors `top_k`. (Note: the canonical DTO currently admits
    /// `top_k` only via its lenient `extra` map, so `diff_request` does not
    /// inspect it yet; the flag exists for routing and future DTO promotion.)
    pub top_k: bool,
    /// The provider returns `logprobs` in responses (a response feature;
    /// reported at response time, never in the request diff).
    pub logprobs: bool,
    /// The provider honors tool declarations and emits tool calls.
    pub tools: bool,
    /// The provider accepts image content parts.
    pub vision: bool,
    /// The provider honors `parallel_tool_calls`.
    pub parallel_tool_calls: bool,
    /// Maximum accepted completion-token budget; requests above it are
    /// clamped. `None` = unlimited/unenforced.
    pub max_tokens_ceiling: Option<u64>,
}

/// What happened to a request field the provider could not honor as sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DropAction {
    /// The field was removed from the upstream request.
    Dropped,
    /// A numeric field was lowered to the provider's ceiling; `applied` is
    /// the value actually sent upstream.
    Clamped {
        /// The clamped value sent to the provider.
        applied: u64,
    },
}

/// One field affected by the capability diff, advertised to the client via
/// the `x-dropped-request-fields` header (spec: "Drop/clamp-and-advertise
/// loss policy").
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldDrop {
    /// Canonical field name as the client knows it (e.g. `seed`,
    /// `max_completion_tokens`).
    pub field: String,
    /// What was done to the field.
    pub action: DropAction,
}

impl FieldDrop {
    /// Construct a [`DropAction::Dropped`] entry.
    pub fn dropped(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            action: DropAction::Dropped,
        }
    }

    /// Construct a [`DropAction::Clamped`] entry.
    pub fn clamped(field: impl Into<String>, applied: u64) -> Self {
        Self {
            field: field.into(),
            action: DropAction::Clamped { applied },
        }
    }

    /// Header-friendly rendering: the bare field name when dropped, or
    /// `field=applied` when clamped.
    pub fn to_header_value(&self) -> String {
        match self.action {
            DropAction::Dropped => self.field.clone(),
            DropAction::Clamped { applied } => format!("{}={applied}", self.field),
        }
    }
}

impl fmt::Display for FieldDrop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_header_value())
    }
}

/// Diff a canonical request against an adapter's capabilities, returning one
/// [`FieldDrop`] per field that must be dropped or clamped before the
/// upstream request is built.
///
/// Only fields that exist on the canonical DTO are diffed:
/// - `seed` is dropped when the adapter does not declare `seed` support.
/// - `max_completion_tokens` is clamped to `max_tokens_ceiling` when it
///   exceeds it.
///
/// Deliberately not diffed here: `logprobs`/`top_logprobs` are response
/// features whose drops are reported at decode/response time (Stage C header
/// and SSE-comment wiring); `top_k` and any other unknown fields arrive via
/// the DTO's lenient `extra` map and are passed through untouched (out of
/// scope for the request diff).
pub fn diff_request(caps: &CapabilitySet, req: &ChatCompletionRequest) -> Vec<FieldDrop> {
    let mut drops = Vec::new();
    if req.seed.is_some() && !caps.seed {
        drops.push(FieldDrop::dropped("seed"));
    }
    if let (Some(ceiling), Some(requested)) = (caps.max_tokens_ceiling, req.max_completion_tokens)
        && u64::from(requested) > ceiling
    {
        drops.push(FieldDrop::clamped("max_completion_tokens", ceiling));
    }
    drops
}

/// Provider adapter seam (design D3). Each adapter declares its capabilities
/// once and implements the three translation directions; the API and
/// streaming layers (Stage C) consume this trait without knowing providers.
pub trait Translator: Send + Sync {
    /// This adapter's capability declaration.
    fn capabilities(&self) -> &CapabilitySet;

    /// Encode a canonical request into a provider-native JSON body, applying
    /// the drop/clamp diff. Returns the wire body plus the drops that were
    /// applied (for the `x-dropped-request-fields` header).
    fn encode_request(
        &self,
        req: &ChatCompletionRequest,
    ) -> Result<(serde_json::Value, Vec<FieldDrop>), ProxyError>;

    /// Decode a non-streaming provider response body into the canonical
    /// response shape.
    fn decode_response(
        &self,
        body: &serde_json::Value,
    ) -> Result<ChatCompletionResponse, ProxyError>;

    /// Wrap a provider SSE byte stream, translating it into a stream of
    /// canonical chunks. The decoder must hold O(1) state (current frame
    /// buffer and bookkeeping only — never accumulated content) and emit
    /// usage exactly once at the end when `include_usage` is set.
    fn decode_stream(&self, include_usage: bool, byte_stream: ByteStream) -> ChunkStream;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps(seed: bool, ceiling: Option<u64>) -> CapabilitySet {
        CapabilitySet {
            seed,
            max_tokens_ceiling: ceiling,
            ..CapabilitySet::default()
        }
    }

    fn request_with(
        seed: Option<u64>,
        max_completion_tokens: Option<u32>,
    ) -> ChatCompletionRequest {
        let mut req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("minimal request should deserialize");
        req.seed = seed;
        req.max_completion_tokens = max_completion_tokens;
        req
    }

    #[test]
    fn unsupported_seed_is_dropped() {
        let drops = diff_request(&caps(false, None), &request_with(Some(42), None));
        assert_eq!(drops, vec![FieldDrop::dropped("seed")]);
    }

    #[test]
    fn supported_seed_emits_nothing() {
        let drops = diff_request(&caps(true, None), &request_with(Some(42), None));
        assert!(drops.is_empty());
    }

    #[test]
    fn token_budget_above_ceiling_is_clamped() {
        let drops = diff_request(&caps(true, Some(8192)), &request_with(None, Some(100_000)));
        assert_eq!(
            drops,
            vec![FieldDrop::clamped("max_completion_tokens", 8192)]
        );
    }

    #[test]
    fn token_budget_at_or_below_ceiling_passes_through() {
        for budget in [Some(8192), Some(100), Some(1)] {
            let drops = diff_request(&caps(true, Some(8192)), &request_with(None, budget));
            assert!(drops.is_empty(), "budget {budget:?} should pass");
        }
    }

    #[test]
    fn unlimited_capability_never_clamps() {
        let drops = diff_request(&caps(true, None), &request_with(None, Some(u32::MAX)));
        assert!(drops.is_empty());
    }

    #[test]
    fn absent_fields_emit_nothing() {
        let drops = diff_request(&caps(false, Some(8192)), &request_with(None, None));
        assert!(drops.is_empty());
    }

    #[test]
    fn field_drop_header_rendering() {
        assert_eq!(FieldDrop::dropped("seed").to_header_value(), "seed");
        assert_eq!(
            FieldDrop::clamped("max_completion_tokens", 8192).to_header_value(),
            "max_completion_tokens=8192"
        );
    }

    #[test]
    fn field_drop_serialization_is_header_friendly() {
        let value = serde_json::to_value(FieldDrop::clamped("max_completion_tokens", 8192))
            .expect("should serialize");
        assert_eq!(
            value,
            json!({
                "field": "max_completion_tokens",
                "action": {"kind": "clamped", "applied": 8192}
            })
        );
    }
}
