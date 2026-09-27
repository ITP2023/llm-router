//! Binary smoke test (task 7.1): boot the real `llm-router` executable with
//! env config pointing at a wiremock upstream, verify it binds, serves
//! `/v1/models` and a chat completion, and terminates without leaving the
//! port bound.
//!
//! Graceful-drain semantics (in-flight requests completing before exit) are
//! provided by `axum::serve(...).with_graceful_shutdown`, exercised by the
//! SIGTERM/SIGINT wiring in `src/main.rs`; this test pins the observable
//! contract: the process starts from env alone and stops releasing the
//! listener on termination.

use serde_json::json;
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Uncommon loopback port for the smoke test instance.
const LISTEN_ADDR: &str = "127.0.0.1:18723";

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_until_accepting(addr: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        if TcpStream::connect(addr).is_ok() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "server did not start accepting connections on {addr} within {timeout:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[tokio::test]
async fn binary_boots_serves_and_terminates() {
    let upstream = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&upstream)
        .await;

    let binary = env!("CARGO_BIN_EXE_llm-router");
    let child = Command::new(binary)
        .env("LLM_ROUTER_LISTEN", LISTEN_ADDR)
        .env("LLM_ROUTER_DEEPSEEK_API_KEY", "smoke-key")
        .env("LLM_ROUTER_DEEPSEEK_BASE_URL", upstream.uri())
        .env("LLM_ROUTER_MODELS", "smoke-model")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary spawns");
    let guard = ChildGuard(child);

    wait_until_accepting(LISTEN_ADDR, Duration::from_secs(15));

    let http = reqwest::Client::new();

    // /v1/models lists the env-configured alias, namespaced by provider.
    let response = http
        .get(format!("http://{LISTEN_ADDR}/v1/models"))
        .send()
        .await
        .expect("models request succeeds");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("models body is JSON");
    let ids: Vec<&str> = body["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|entry| entry["id"].as_str().expect("id string"))
        .collect();
    assert_eq!(ids, vec!["deepseek/smoke-model"]);

    // A chat completion round-trips through the wiremock upstream. DeepSeek's
    // default caps clamp max_completion_tokens, so include a drop trigger to
    // prove the full pipeline runs in the real binary.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-smoke",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "smoke-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "smoke ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        })))
        .mount(&upstream)
        .await;

    let response = http
        .post(format!("http://{LISTEN_ADDR}/v1/chat/completions"))
        .json(&json!({
            "model": "deepseek/smoke-model",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 7
        }))
        .send()
        .await
        .expect("chat request succeeds");
    assert_eq!(response.status(), 200);
    let drop_header = response
        .headers()
        .get("x-dropped-request-fields")
        .and_then(|v| v.to_str().ok())
        .expect("drop header present (seed is unsupported)");
    assert!(drop_header.contains("seed"), "header: {drop_header}");
    let body: serde_json::Value = response.json().await.expect("chat body is JSON");
    assert_eq!(body["choices"][0]["message"]["content"], "smoke ok");

    // Terminate: the guard kills the child and reaps it. Afterwards the port
    // must be free again — i.e. the process released its listener.
    drop(guard);

    let released = wait_for_release(LISTEN_ADDR, Duration::from_secs(5));
    assert!(released, "listener port must be released after termination");
}

/// Second loopback port for the mockllm-mode smoke test (tests in this file
/// may run concurrently).
const MOCK_LISTEN_ADDR: &str = "127.0.0.1:18724";

#[tokio::test]
async fn binary_boots_in_mockllm_mode_without_credentials() {
    let binary = env!("CARGO_BIN_EXE_llm-router");
    let child = Command::new(binary)
        .env("LLM_ROUTER_LISTEN", MOCK_LISTEN_ADDR)
        .env("LLM_ROUTER_PROVIDER", "mockllm")
        .env("LLM_ROUTER_MODELS", "smoke-mock")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("binary spawns");
    let guard = ChildGuard(child);

    wait_until_accepting(MOCK_LISTEN_ADDR, Duration::from_secs(15));

    let http = reqwest::Client::new();

    // No LLM_ROUTER_DEEPSEEK_API_KEY was set: the in-process mock upstream
    // answers anyway.
    let response = http
        .get(format!("http://{MOCK_LISTEN_ADDR}/v1/models"))
        .send()
        .await
        .expect("models request succeeds");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("models body is JSON");
    assert_eq!(body["data"][0]["id"], "mockllm/smoke-mock");
    assert_eq!(body["data"][0]["owned_by"], "mockllm");

    let response = http
        .post(format!("http://{MOCK_LISTEN_ADDR}/v1/chat/completions"))
        .json(&json!({
            "model": "mockllm/smoke-mock",
            "messages": [{"role": "user", "content": "binary smoke"}]
        }))
        .send()
        .await
        .expect("chat request succeeds");
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.expect("chat body is JSON");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "mockllm: binary smoke"
    );

    drop(guard);
    let released = wait_for_release(MOCK_LISTEN_ADDR, Duration::from_secs(5));
    assert!(released, "listener port must be released after termination");
}

fn wait_for_release(addr: &str, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if TcpStream::connect(addr).is_err() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
