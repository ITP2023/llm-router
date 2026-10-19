//! Integration tests for the canonical proxy core: wiremock stands in for
//! the DeepSeek upstream so the suite needs no network or API keys.
//!
//! Coverage maps to the `core-proxy` spec scenarios: non-streaming happy
//! path with drop headers, streaming SSE shape with the warning-comment
//! sideband, model listing, unknown-alias rejection (zero upstream calls),
//! malformed JSON, and upstream error-status mapping.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use llm_router::providers::{CapabilitySet, DeepSeekTranslator, Translator};
use llm_router::{AppState, DeepSeekTarget, ModelRegistry, ProviderTarget, app};
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ALIAS: &str = "deepseek/deepseek-chat";
/// Bare upstream model name: what the wire must carry, and what the
/// upstream fixture echoes in its response body.
const WIRE_MODEL: &str = "deepseek-chat";

/// Build an app whose single alias points at the wiremock server. The
/// registry is keyed by the namespaced client-facing id; the translator
/// carries the bare upstream model name for the wire. The `logprobs`
/// capability is overridable so response-drop advertising can be exercised
/// against a provider that does not return logprobs.
fn test_app(server: &MockServer, logprobs: bool) -> axum::Router {
    let translator = DeepSeekTranslator::with_capabilities(
        WIRE_MODEL,
        server.uri(),
        CapabilitySet {
            logprobs,
            ..DeepSeekTranslator::new(WIRE_MODEL, server.uri())
                .capabilities()
                .clone()
        },
    );
    let mut registry = ModelRegistry::empty();
    registry.insert(
        ALIAS.to_string(),
        ProviderTarget::DeepSeek(DeepSeekTarget::new(translator, "test-key")),
    );
    app(AppState {
        http: reqwest::Client::new(),
        registry,
    })
}

fn chat_request(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds")
}

fn upstream_response() -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": WIRE_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    })
}

async fn response_parts(response: axum::response::Response) -> (StatusCode, Value) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let json = serde_json::from_slice(&body).expect("body is JSON");
    (status, json)
}

#[tokio::test]
async fn non_streaming_happy_path_with_no_drops() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
        .mount(&server)
        .await;

    let response = test_app(&server, true)
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}]
        })))
        .await
        .expect("request succeeds");
    let (status, body) = response_parts(response).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["model"], ALIAS,
        "response echoes the namespaced client-facing id"
    );
    assert_eq!(body["choices"][0]["message"]["content"], "Hello there.");
    assert_eq!(body["usage"]["total_tokens"], 18);
    assert!(
        body.get("x-dropped-request-fields").is_none()
            && body.get("x-dropped-response-fields").is_none(),
        "JSON body must not carry advertising headers"
    );
}

#[tokio::test]
async fn request_drops_are_advertised_in_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
        .expect(1)
        .mount(&server)
        .await;

    let app = test_app(&server, true);
    let response = app
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 42,
            "max_completion_tokens": 100_000
        })))
        .await
        .expect("request succeeds");

    assert_eq!(response.status(), StatusCode::OK);
    let header = response
        .headers()
        .get("x-dropped-request-fields")
        .and_then(|v| v.to_str().ok())
        .expect("drop header present");
    assert!(header.contains("seed"), "header: {header}");
    assert!(
        header.contains("max_completion_tokens=8192"),
        "header: {header}"
    );

    // The upstream must have received the clamped/dropped wire form, with
    // the bare upstream model name (not the namespaced client id).
    let requests = server.received_requests().await.expect("requests recorded");
    let wire: Value = serde_json::from_slice(&requests[0].body).expect("upstream body is JSON");
    assert_eq!(wire["model"], json!(WIRE_MODEL));
    assert!(wire.get("seed").is_none(), "seed must not reach upstream");
    assert_eq!(wire["max_completion_tokens"], json!(8192));
}

#[tokio::test]
async fn unsupported_logprobs_advertised_in_response_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
        .mount(&server)
        .await;

    let response = test_app(&server, /* logprobs unsupported */ false)
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true
        })))
        .await
        .expect("request succeeds");

    assert_eq!(response.status(), StatusCode::OK);
    let header = response
        .headers()
        .get("x-dropped-response-fields")
        .and_then(|v| v.to_str().ok())
        .expect("response-drop header present");
    assert!(header.contains("logprobs"), "header: {header}");
}

#[tokio::test]
async fn models_endpoint_lists_configured_aliases() {
    let server = MockServer::start().await;
    let response = test_app(&server, true)
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("request succeeds");
    let (status, body) = response_parts(response).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|entry| entry["id"].as_str().expect("id string"))
        .collect();
    assert_eq!(ids, vec![ALIAS]);
    assert_eq!(body["data"][0]["object"], "model");
    assert_eq!(body["data"][0]["owned_by"], "deepseek");
}

#[tokio::test]
async fn unknown_alias_rejected_without_upstream_call() {
    let server = MockServer::start().await;
    // No mock mounted: any upstream request would be unmatched (404), which
    // the assertion on received_requests catches independently.
    let response = test_app(&server, true)
        .oneshot(chat_request(json!({
            "model": "does-not-exist",
            "messages": [{"role": "user", "content": "hi"}]
        })))
        .await
        .expect("request completes");
    let (status, body) = response_parts(response).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message string")
            .contains("does-not-exist")
    );
    let requests = server.received_requests().await.expect("requests recorded");
    assert!(requests.is_empty(), "no upstream request may be made");
}

#[tokio::test]
async fn malformed_json_is_canonical_400() {
    let server = MockServer::start().await;
    let response = test_app(&server, true)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{not json"))
                .expect("request builds"),
        )
        .await
        .expect("request completes");
    let (status, body) = response_parts(response).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].is_string());
}

#[tokio::test]
async fn upstream_500_surfaces_as_canonical_502() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("deep internal provider error"))
        .mount(&server)
        .await;

    let response = test_app(&server, true)
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}]
        })))
        .await
        .expect("request completes");
    let (status, body) = response_parts(response).await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["type"], "upstream_error");
    assert!(
        !body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("deep internal provider error"),
        "provider body must not leak unmodified"
    );
}

#[tokio::test]
async fn upstream_429_is_preserved() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let response = test_app(&server, true)
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}]
        })))
        .await
        .expect("request completes");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

/// SSE body the fake upstream streams: role delta, content deltas, a
/// usage-only chunk, then [DONE] — the canonical OpenAI streaming shape.
fn upstream_sse() -> String {
    let frames = [
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "created": 1_700_000_000, "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "created": 1_700_000_000, "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": null}]
        }),
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "created": 1_700_000_000, "model": WIRE_MODEL,
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]
        }),
        json!({
            "id": "chatcmpl-1", "object": "chat.completion.chunk",
            "created": 1_700_000_000, "model": WIRE_MODEL,
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }),
    ];
    let mut body = String::new();
    for frame in frames {
        body.push_str("data: ");
        body.push_str(&frame.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn streaming_sse_shape_warning_sideband_and_single_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(upstream_sse()),
        )
        .mount(&server)
        .await;

    let response = test_app(&server, /* logprobs unsupported */ false)
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "logprobs": true,
            "seed": 1
        })))
        .await
        .expect("request succeeds");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream"),
        "must be an SSE response"
    );
    let drop_header = response
        .headers()
        .get("x-dropped-request-fields")
        .and_then(|v| v.to_str().ok())
        .expect("request-drop header present");
    assert!(drop_header.contains("seed"), "header: {drop_header}");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let text = String::from_utf8(body.to_vec()).expect("utf8");

    // Warning sideband: an SSE comment line advertising the response drop.
    assert!(
        text.contains(": router-warning: logprobs dropped"),
        "stream must carry the warning comment:\n{text}"
    );

    // Frame structure: every data frame parses as a canonical chunk.
    let mut usage_frames = 0;
    let mut content_seen = String::new();
    let mut saw_done = false;
    for block in text.split("\n\n") {
        if block.is_empty() {
            continue;
        }
        if let Some(payload) = block.strip_prefix("data: ") {
            if payload == "[DONE]" {
                saw_done = true;
                continue;
            }
            let chunk: Value = serde_json::from_str(payload).expect("chunk is JSON");
            assert_eq!(chunk["object"], "chat.completion.chunk");
            assert_eq!(
                chunk["model"], ALIAS,
                "streamed chunks echo the namespaced client-facing id, not {WIRE_MODEL}"
            );
            if let Some(usage) = chunk.get("usage") {
                usage_frames += 1;
                assert_eq!(usage["total_tokens"], 7);
            }
            if let Some(content) = chunk["choices"][0]["delta"].get("content") {
                content_seen.push_str(content.as_str().expect("content string"));
            }
        }
    }
    assert_eq!(
        usage_frames, 1,
        "usage must be reported exactly once:\n{text}"
    );
    assert_eq!(content_seen, "Hello");
    assert!(saw_done, "stream must terminate with [DONE]:\n{text}");
    // The [DONE] marker must be the last frame.
    assert!(
        text.ends_with("data: [DONE]\n\n"),
        "tail: {:?}",
        &text[text.len().saturating_sub(40)..]
    );
}
