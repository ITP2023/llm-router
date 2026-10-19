//! End-to-end tests through the MockLLM provider: the real axum handler,
//! the real reqwest client, and real HTTP to the in-process mock upstream —
//! only the LLM is fake. Complements the wiremock suite (which stubs a
//! DeepSeek-shaped upstream) by exercising the MockLLM adapter and its
//! upstream server together.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use llm_router::providers::CapabilitySet;
use llm_router::providers::mockllm::server::{self, MockUpstream};
use llm_router::providers::{MockLLMTranslator, Translator};
use llm_router::{AppState, MockLLMTarget, ModelRegistry, ProviderTarget, app};
use serde_json::{Value, json};
use tower::ServiceExt;

/// Client-facing namespaced model id, as listed by `/v1/models` and sent in
/// the `model` request field.
const ALIAS: &str = "mockllm/mock-chat";
/// Bare upstream model name the wire carries.
const WIRE_MODEL: &str = "mock-chat";

/// Spawn the mock upstream and build the full router pointing at it.
/// `configure` overrides translator capabilities (e.g. logprobs: false).
async fn setup<F>(configure: F) -> (MockUpstream, axum::Router)
where
    F: FnOnce(&mut CapabilitySet),
{
    let upstream = server::spawn(server::Config::default())
        .await
        .expect("mock upstream spawns");
    let mut caps = MockLLMTranslator::new(WIRE_MODEL, upstream.base_url())
        .capabilities()
        .clone();
    configure(&mut caps);
    let translator = MockLLMTranslator::with_capabilites(WIRE_MODEL, upstream.base_url(), caps);

    let mut registry = ModelRegistry::empty();
    registry.insert(
        ALIAS.to_string(),
        ProviderTarget::MockLLM(MockLLMTarget::new(translator)),
    );
    let router = app(AppState {
        http: reqwest::Client::new(),
        registry,
    });
    (upstream, router)
}

fn chat_request(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds")
}

async fn json_body(response: axum::response::Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let value = serde_json::from_slice(&bytes).expect("body is JSON");
    (status, value)
}

#[tokio::test]
async fn e2e_non_streaming_echo_round_trip() {
    let (upstream, router) = setup(|_| {}).await;

    let response = router
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "ping"}]
        })))
        .await
        .expect("request succeeds");
    let (status, body) = json_body(response).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["model"], ALIAS,
        "response echoes the namespaced client-facing id, not the bare wire name"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"], "mockllm: ping",
        "the full request->upstream->response path must carry the user text"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);

    upstream.stop().await;
}

#[tokio::test]
async fn e2e_streaming_reassembles_echo_and_reports_usage_once() {
    let (upstream, router) = setup(|_| {}).await;

    let response = router
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "stream this text through"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        })))
        .await
        .expect("request succeeds");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("utf8");

    let mut reassembled = String::new();
    let mut usage_frames = 0;
    let mut saw_done = false;
    for block in text.split("\n\n") {
        if block.is_empty() {
            continue;
        }
        let payload = block.strip_prefix("data: ").expect("data frame or [DONE]");
        if payload == "[DONE]" {
            saw_done = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(payload).expect("chunk is JSON");
        assert_eq!(chunk["object"], "chat.completion.chunk");
        if let Some(usage) = chunk.get("usage") {
            usage_frames += 1;
            assert!(usage["total_tokens"].as_u64().unwrap() > 0);
        } else if let Some(content) = chunk["choices"][0]["delta"].get("content") {
            reassembled.push_str(content.as_str().expect("content string"));
        }
    }
    assert_eq!(reassembled, "mockllm: stream this text through");
    assert_eq!(usage_frames, 1, "usage exactly once:\n{text}");
    assert!(saw_done && text.ends_with("data: [DONE]\n\n"));
    // No warning comments: nothing was dropped.
    assert!(!text.contains(": router-warning:"), "no warnings:\n{text}");

    upstream.stop().await;
}

#[tokio::test]
async fn e2e_drops_advertised_and_upstream_sees_clamped_wire() {
    let (upstream, router) = setup(|_| {}).await;

    let response = router
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

    upstream.stop().await;
}

#[tokio::test]
async fn e2e_streaming_logprobs_drop_uses_sse_comment_sideband() {
    // Configure the translator as if the provider cannot return logprobs;
    // the request asks for them, so the warning must arrive in-band.
    let (upstream, router) = setup(|caps| caps.logprobs = false).await;

    let response = router
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "logprobs": true
        })))
        .await
        .expect("request succeeds");
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("utf8");
    assert!(
        text.contains(": router-warning: logprobs dropped"),
        "warning sideband missing:\n{text}"
    );

    upstream.stop().await;
}

#[tokio::test]
async fn e2e_injected_upstream_error_maps_to_canonical_502() {
    let (upstream, router) = setup(|_| {}).await;

    let response = router
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": "mock-error-503"}]
        })))
        .await
        .expect("request completes");
    let (status, body) = json_body(response).await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["type"], "upstream_error");
    assert!(
        !body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("mock upstream error"),
        "provider body must not leak unmodified"
    );

    upstream.stop().await;
}

#[tokio::test]
async fn e2e_models_listing_marks_mockllm_ownership() {
    let (upstream, router) = setup(|_| {}).await;

    let response = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("request succeeds");
    let (status, body) = json_body(response).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["id"], ALIAS);
    assert_eq!(body["data"][0]["owned_by"], "mockllm");

    upstream.stop().await;
}

/// Delayed-chunk config makes "first frame before stream end" observable
/// end to end (the unit-level pin lives in src/streaming tests; wiremock
/// cannot deliver delayed chunks, the mock upstream can).
#[tokio::test]
async fn e2e_streaming_first_chunk_arrives_before_generation_completes() {
    use futures_util::StreamExt;

    let upstream = server::spawn(server::Config {
        chunk_size: 4,
        chunk_delay: std::time::Duration::from_millis(30),
    })
    .await
    .expect("mock upstream spawns");
    // The constructor namespaces bare model names into `mockllm/<model>`.
    let registry =
        ModelRegistry::mockllm(upstream.base_url(), &[WIRE_MODEL]).expect("registry builds");
    let router = app(AppState {
        http: reqwest::Client::new(),
        registry,
    });

    // Content long enough for many delayed chunks: total generation time
    // far exceeds the time to deliver the first chunk.
    let content = "abcdefghijklmnopqrstuvwxyz0123456789";
    let response = router
        .oneshot(chat_request(json!({
            "model": ALIAS,
            "messages": [{"role": "user", "content": content}],
            "stream": true
        })))
        .await
        .expect("request succeeds");
    assert_eq!(response.status(), StatusCode::OK);

    let started = std::time::Instant::now();
    let mut stream = response.into_body().into_data_stream();
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("first chunk arrives promptly")
        .expect("stream yields")
        .expect("chunk bytes");
    let first_latency = started.elapsed();

    let text_head = String::from_utf8(first.to_vec()).expect("utf8");
    assert!(text_head.starts_with("data: "), "got: {text_head}");

    // Full generation: ~len/4 chunks * 30ms each. If the proxy buffered the
    // whole upstream body before forwarding, the first chunk could not have
    // arrived earlier than that total. It demonstrably did.
    let chunk_count = (content.len() as f64 / 4.0).ceil() as u32;
    let min_total = std::time::Duration::from_millis(30 * u64::from(chunk_count.saturating_sub(1)));
    assert!(
        first_latency < min_total,
        "first chunk in {first_latency:?}, but full generation needs >= {min_total:?} — \
         the proxy must not buffer the stream"
    );

    upstream.stop().await;
}
