//! SSE response framing for streaming chat completions (design D4).
//!
//! [`sse_response`] turns a [`ChunkStream`] into an axum `Response`:
//! warning comments first (in-band sideband for response-time drops — spec:
//! "Streaming warning sideband"), then `data: {<chunk>}` frames, then a
//! terminating `data: [DONE]` frame. The returned `Response` is fully built
//! (headers included) before the chunk stream is polled, so response
//! headers are committed before the first upstream byte streams (design D4).

use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, HeaderValue};
use axum::http::{HeaderMap, Response};
use futures_util::StreamExt;
use futures_util::stream;

use crate::error::ProxyError;
use crate::providers::capabilities::ChunkStream;

/// Content type for SSE streams.
const SSE_CONTENT_TYPE: &str = "text/event-stream";

/// Build the SSE `Response` for a streaming chat completion.
///
/// `warnings` are rendered as SSE comment lines (`: router-warning: ...`)
/// before the first data frame; conformant SSE clients (including OpenAI
/// SDKs) ignore comment lines. `headers` are attached to the response and
/// committed before the body streams.
pub fn sse_response(
    chunks: ChunkStream,
    warnings: Vec<String>,
    headers: HeaderMap,
) -> Response<Body> {
    let warning_frames = stream::iter(
        warnings
            .into_iter()
            .map(|warning| Ok::<_, ProxyError>(bytes(format!(": router-warning: {warning}\n\n")))),
    );

    let chunk_frames = chunks.map(|result| {
        result.and_then(|chunk| {
            let json = serde_json::to_string(&chunk)
                .map_err(|err| ProxyError::Translation(format!("chunk encode: {err}")))?;
            Ok(bytes(format!("data: {json}\n\n")))
        })
    });

    let done_frame =
        stream::once(async { Ok::<_, ProxyError>(bytes("data: [DONE]\n\n".to_string())) });

    let body_stream = warning_frames.chain(chunk_frames).chain(done_frame);

    let mut response = Response::new(Body::from_stream(body_stream));
    let response_headers = response.headers_mut();
    response_headers.insert(CONTENT_TYPE, HeaderValue::from_static(SSE_CONTENT_TYPE));
    for (name, value) in &headers {
        response_headers.insert(name, value.clone());
    }
    response
}

fn bytes(text: String) -> bytes::Bytes {
    bytes::Bytes::from(text)
}

#[cfg(test)]
mod tests {
    use std::task::{Context, Poll};
    use std::time::Duration;

    use axum::body::to_bytes;
    use futures_util::Stream;
    use futures_util::stream;
    use serde_json::json;

    use super::*;
    use crate::providers::DeepSeekTranslator;
    use crate::providers::capabilities::Translator;

    fn chunk_json(content: &str) -> String {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "m",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": null
            }]
        })
        .to_string()
    }

    fn sse_bytes(frames: &[String]) -> bytes::Bytes {
        let mut text = String::new();
        for frame in frames {
            text.push_str("data: ");
            text.push_str(frame);
            text.push_str("\n\n");
        }
        bytes::Bytes::from(text)
    }

    /// Assemble a chunk stream by piping canned SSE frames through the
    /// DeepSeek decoder, exactly as the handler does.
    fn decode_frames(frames: &[String], include_usage: bool) -> ChunkStream {
        let byte_stream = stream::iter(vec![Ok(sse_bytes(frames))]).boxed();
        DeepSeekTranslator::new("m", "https://example.test")
            .decode_stream(include_usage, byte_stream)
    }

    #[tokio::test]
    async fn frames_warning_then_chunks_then_done() {
        let frames = vec![chunk_json("Hello")];
        let response = sse_response(
            decode_frames(&frames, false),
            vec!["logprobs dropped (unsupported by m)".to_string()],
            HeaderMap::new(),
        );

        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        // Compare framing structurally: comment line, one data frame, [DONE].
        // The data frame JSON is compared as parsed values because the decoder
        // re-serializes through the DTO (key order is not the fixture's).
        let expected_prefix = ": router-warning: logprobs dropped (unsupported by m)\n\n";
        let expected_suffix = "\n\ndata: [DONE]\n\n";
        assert!(text.starts_with(expected_prefix), "got {text}");
        assert!(text.ends_with(expected_suffix), "got {text}");
        let data_frame = text
            .strip_prefix(expected_prefix)
            .and_then(|rest| rest.strip_suffix(expected_suffix))
            .expect("single data frame")
            .strip_prefix("data: ")
            .expect("data frame prefix");
        let emitted: serde_json::Value =
            serde_json::from_str(data_frame).expect("data frame is JSON");
        let expected =
            strip_nulls(serde_json::from_str(&chunk_json("Hello")).expect("fixture is JSON"));
        assert_eq!(emitted, expected);
    }

    /// The DTO serializer skips `None` fields, so strip explicit nulls from
    /// fixtures before comparing against emitted frames.
    fn strip_nulls(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => serde_json::Value::Object(
                map.into_iter()
                    .filter(|(_, v)| !v.is_null())
                    .map(|(k, v)| (k, strip_nulls(v)))
                    .collect(),
            ),
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.into_iter().map(strip_nulls).collect())
            }
            other => other,
        }
    }

    #[tokio::test]
    async fn no_warnings_when_list_empty() {
        let frames = vec![chunk_json("Hi")];
        let response = sse_response(decode_frames(&frames, false), Vec::new(), HeaderMap::new());
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(!text.starts_with(':'), "no comment lines expected: {text}");
        assert!(text.ends_with("data: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn caller_headers_are_attached() {
        let frames = vec![chunk_json("Hi")];
        let mut headers = HeaderMap::new();
        headers.insert("x-dropped-request-fields", HeaderValue::from_static("seed"));
        let response = sse_response(decode_frames(&frames, false), Vec::new(), headers);
        assert_eq!(
            response
                .headers()
                .get("x-dropped-request-fields")
                .and_then(|v| v.to_str().ok()),
            Some("seed")
        );
    }

    /// A stream that yields its first item, then never completes: the
    /// strongest deterministic stand-in for "the upstream is still
    /// streaming" when asserting the decoder does not buffer (spec:
    /// "Constant memory for long streams"). Wiremock 0.6 cannot deliver
    /// delayed chunks, so this unit-level test pins the guarantee instead.
    struct FirstThenPending {
        first: Option<Result<bytes::Bytes, reqwest::Error>>,
    }

    impl Stream for FirstThenPending {
        type Item = Result<bytes::Bytes, reqwest::Error>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            match self.first.take() {
                Some(item) => Poll::Ready(Some(item)),
                None => Poll::Pending,
            }
        }
    }

    #[tokio::test]
    async fn first_chunk_emitted_without_waiting_for_upstream_end() {
        let first_frame = sse_bytes(&[chunk_json("partial")]);
        let upstream = FirstThenPending {
            first: Some(Ok(first_frame)),
        };
        let byte_stream: crate::providers::capabilities::ByteStream = Box::pin(upstream);
        let mut chunks =
            DeepSeekTranslator::new("m", "https://example.test").decode_stream(false, byte_stream);

        let timeout = tokio::time::timeout(Duration::from_millis(200), chunks.next());
        let first = timeout
            .await
            .expect("first chunk must arrive without waiting for stream end")
            .expect("stream yields one item")
            .expect("chunk decodes");
        let content = first.choices[0]
            .delta
            .content
            .as_deref()
            .expect("content delta");
        assert!(content.contains("partial"), "got {content}");

        // The upstream never completes; a buffering decoder would never
        // yield, a non-buffering one simply has nothing more right now.
        let second = tokio::time::timeout(Duration::from_millis(100), chunks.next()).await;
        assert!(second.is_err(), "decoder must not wait for upstream end");
    }
}
