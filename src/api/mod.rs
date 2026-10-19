//! HTTP route handlers: `POST /v1/chat/completions` and `GET /v1/models`.
//!
//! Handlers stay thin (spec: translation lives in `src/providers/`, SSE
//! framing in `src/streaming/`): resolve the alias, diff capabilities,
//! encode, forward on the shared client, decode, and attach the
//! `x-dropped-*` advertising headers.

use axum::extract::rejection::JsonRejection;
use axum::extract::{Json, State};
use axum::http::header::HeaderName;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json as JsonExtractor, Router};
use futures_util::StreamExt;

use crate::error::ProxyError;
use crate::models::chat::ChatCompletionRequest;
use crate::models::list::{ModelEntry, ModelList};
use crate::providers::capabilities::FieldDrop;
use crate::state::AppState;
use crate::streaming;

/// `created` timestamp reported for every alias on `/v1/models`. Fixed (not
/// wall-clock) so the listing is deterministic.
const MODEL_LIST_CREATED: u64 = 1_700_000_000;

/// Routes with [`AppState`] as the router state, before `.with_state(...)`.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
}

/// `GET /v1/models` — list every configured model alias (spec: "Models
/// endpoint"). No upstream call is involved.
async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    tracing::debug!("model list request");
    let data = state
        .registry
        .aliases()
        .map(|alias| ModelEntry {
            id: alias.to_string(),
            object: "model".to_string(),
            created: MODEL_LIST_CREATED,
            owned_by: state
                .registry
                .resolve(alias)
                .map(|target| target.provider_name())
                .unwrap_or("unknown")
                .to_string(),
        })
        .collect();
    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}

/// `POST /v1/chat/completions` — canonical proxy endpoint (spec: "Chat
/// completions endpoint"). Non-streaming returns a canonical completion
/// object; `stream: true` returns an SSE stream of canonical chunks.
async fn chat_completions(
    State(state): State<AppState>,
    body: Result<JsonExtractor<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ProxyError> {
    // Malformed JSON (or a missing/JSON content type) must yield the
    // canonical 400 error shape, not axum's default rejection body.
    let JsonExtractor(req) =
        body.map_err(|rejection| ProxyError::MalformedRequest(rejection.body_text()))?;

    tracing::info!(model = %req.model, stream = req.stream, "chat completion request");

    // Resolve before any I/O: unknown aliases fail here and never touch the
    // upstream (spec: "Unknown model alias is rejected").
    let target = state
        .registry
        .resolve(&req.model)
        .ok_or_else(|| ProxyError::UnknownModel(req.model.clone()))?;

    let translator = target.translator();
    let (provider_json, request_drops) = translator.encode_request(&req)?;

    // Response-time drop: the client asked for logprobs the provider does
    // not return. Reported in a header for non-streaming, as an SSE comment
    // sideband for streaming (headers are committed before the body streams,
    // spec: "Streaming warning sideband").
    let response_drops: Vec<FieldDrop> =
        if req.logprobs == Some(true) && !translator.capabilities().logprobs {
            vec![FieldDrop::dropped("logprobs")]
        } else {
            Vec::new()
        };

    let url = target.chat_completions_url();
    tracing::debug!(url = %url, "dispatching upstream request");
    let upstream = state
        .http
        .post(&url)
        .bearer_auth(target.api_key())
        .json(&provider_json)
        .send()
        .await
        .map_err(|err| ProxyError::UpstreamUnavailable(err.to_string()))?;

    let status = upstream.status();
    if !status.is_success() {
        let body = upstream
            .text()
            .await
            .unwrap_or_else(|err| format!("(failed to read upstream body: {err})"));
        return Err(ProxyError::UpstreamStatus(status.as_u16(), body));
    }

    let mut headers = HeaderMap::new();
    set_dropped_headers(&mut headers, &request_drops, &response_drops, req.stream)?;

    if req.stream {
        let include_usage = req
            .stream_options
            .as_ref()
            .map(|opts| opts.include_usage)
            .unwrap_or(false);
        let byte_stream = upstream.bytes_stream().boxed();
        let client_model = req.model.clone();
        let chunk_stream: crate::providers::capabilities::ChunkStream = Box::pin(
            translator
                .decode_stream(include_usage, byte_stream)
                .map(move |result| {
                    // The upstream answers with the bare wire model name;
                    // the client must see the namespaced id it requested.
                    result.map(|mut chunk| {
                        chunk.model = client_model.clone();
                        chunk
                    })
                }),
        );
        let warnings = response_drops
            .iter()
            .map(|drop| format!("{} dropped (unsupported by {})", drop.field, req.model))
            .collect();
        Ok(streaming::sse_response(chunk_stream, warnings, headers))
    } else {
        // Non-streaming: read the full body, decode to canonical, respond.
        let body: serde_json::Value = upstream
            .json()
            .await
            .map_err(|err| ProxyError::Translation(format!("upstream body decode: {err}")))?;
        let mut response = translator.decode_response(&body)?;
        // Echo the namespaced client-facing id, not the bare wire model name.
        response.model = req.model.clone();
        Ok((StatusCode::OK, headers, JsonExtractor(response)).into_response())
    }
}

/// Attach `x-dropped-request-fields` (always when non-empty) and
/// `x-dropped-response-fields` (non-streaming only) to the response headers,
/// which are committed before the body starts.
fn set_dropped_headers(
    headers: &mut HeaderMap,
    request_drops: &[FieldDrop],
    response_drops: &[FieldDrop],
    is_streaming: bool,
) -> Result<(), ProxyError> {
    if !request_drops.is_empty() {
        let value = request_drops
            .iter()
            .map(FieldDrop::to_header_value)
            .collect::<Vec<_>>()
            .join(",");
        insert_header(headers, "x-dropped-request-fields", &value)?;
    }
    if !is_streaming && !response_drops.is_empty() {
        let value = response_drops
            .iter()
            .map(FieldDrop::to_header_value)
            .collect::<Vec<_>>()
            .join(",");
        insert_header(headers, "x-dropped-response-fields", &value)?;
    }
    Ok(())
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<(), ProxyError> {
    let header_value = HeaderValue::from_str(value)
        .map_err(|err| ProxyError::Internal(format!("invalid header value for {name}: {err}")))?;
    headers.insert(HeaderName::from_static(name), header_value);
    Ok(())
}
