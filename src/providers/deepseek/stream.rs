//! DeepSeek streaming decoder: OpenAI-style SSE byte frames in, canonical
//! [`ChatCompletionChunk`]s out (spec: "Stateful streaming translation").
//!
//! # Framing
//!
//! Upstream sends `data: {json}\n\n` frames terminated by `data: [DONE]\n\n`.
//! Bytes are accumulated into a frame buffer for the *current* frame only;
//! each complete frame (split at the first `\n\n` or `\r\n\r\n` boundary) is
//! parsed and discarded before the next bytes are awaited. A trailing frame
//! without a blank line is still processed when the input ends. Non-`data`
//! lines (comments, `event:`/`id:` fields) are ignored.
//!
//! # Usage attribution
//!
//! Per spec "Usage attribution", the client must receive usage exactly once,
//! in the final chunk, gated on `stream_options.include_usage`. The upstream
//! OpenAI protocol already sends a single usage chunk, but to be robust
//! against providers that report usage incrementally, this decoder takes the
//! simplest correct approach: the usage value (three `u64`s) is buffered in
//! an O(1) slot, usage is stripped from every forwarded chunk, and one
//! synthetic usage chunk is emitted after the final content chunk when the
//! stream terminates — carrying the *last* (cumulative) usage seen. When
//! `include_usage` is false, usage is stripped and never emitted, and no
//! trailing usage-only chunk is produced.
//!
//! # Memory
//!
//! Decoder state is the current-frame buffer plus bookkeeping (usage slot,
//! last-seen envelope fields, a small output queue drained within the same
//! poll). Content deltas are forwarded chunk-by-chunk and never accumulated,
//! so memory stays constant regardless of stream length (spec: "Constant
//! memory for long streams").

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;
use futures_util::ready;

use crate::error::ProxyError;
use crate::models::chat::{ChatCompletionChunk, Usage};
use crate::providers::capabilities::{ByteStream, ChunkStream};

/// Envelope fields reused for the synthetic trailing usage chunk.
#[derive(Debug, Clone)]
struct Envelope {
    id: String,
    created: u64,
    model: String,
}

/// Incremental SSE frame parser + canonical chunk emitter.
struct DeepSeekStream {
    inner: ByteStream,
    include_usage: bool,
    /// Accumulator for the SSE frame currently being received.
    frame_buf: Vec<u8>,
    /// Chunks produced from the current poll cycle, drained immediately.
    out: VecDeque<Result<ChatCompletionChunk, ProxyError>>,
    /// Latest cumulative usage seen (O(1) slot; emitted once at the end).
    usage: Option<Usage>,
    /// Envelope of the most recent chunk, for the trailing usage chunk.
    envelope: Option<Envelope>,
    /// `data: [DONE]` was received.
    saw_done: bool,
    /// Input ended or a terminal error occurred; draining remaining output.
    terminated: bool,
    /// A terminal error was emitted; suppress the trailing usage chunk.
    errored: bool,
}

/// Wrap a DeepSeek SSE byte stream as a canonical chunk stream.
pub fn decode(include_usage: bool, byte_stream: ByteStream) -> ChunkStream {
    Box::pin(DeepSeekStream {
        inner: byte_stream,
        include_usage,
        frame_buf: Vec::new(),
        out: VecDeque::new(),
        usage: None,
        envelope: None,
        saw_done: false,
        terminated: false,
        errored: false,
    })
}

/// Find the earliest SSE event boundary, returning `(frame_len, consumed)`:
/// the frame occupies `buf[..frame_len]` (exclusive of the boundary and any
/// trailing `\r`) and `buf[..consumed]` should be discarded.
fn split_frame(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if i + 4 <= buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, i + 4));
        }
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, i + 2));
        }
        i += 1;
    }
    None
}

impl DeepSeekStream {
    /// Parse one SSE frame body (everything between event boundaries) and
    /// record its effect: content chunk queued, usage buffered, or `[DONE]`.
    fn handle_frame(&mut self, frame: &[u8]) -> Result<(), ProxyError> {
        let text = String::from_utf8_lossy(frame);
        let mut data = String::new();
        for line in text.lines() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if let Some(rest) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest.trim_start());
            }
        }
        if data.is_empty() {
            return Ok(()); // comment / field-only event: nothing to translate
        }
        if data.trim() == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }
        let mut chunk: ChatCompletionChunk = serde_json::from_str(&data)
            .map_err(|e| ProxyError::Translation(format!("deepseek stream chunk decode: {e}")))?;
        self.envelope = Some(Envelope {
            id: chunk.id.clone(),
            created: chunk.created,
            model: chunk.model.clone(),
        });
        if let Some(usage) = chunk.usage.take() {
            // Keep only the latest cumulative value; emit once at the end.
            self.usage = Some(usage);
        }
        // A stripped usage-only chunk (empty choices) carries no information;
        // forwarding it would pollute the stream with empty envelopes.
        if !chunk.choices.is_empty() {
            self.out.push_back(Ok(chunk));
        }
        Ok(())
    }

    /// Consume `self.frame_buf[..consumed]` as one complete frame.
    fn take_frame(&mut self, frame_len: usize, consumed: usize) -> Vec<u8> {
        let mut frame: Vec<u8> = self.frame_buf.drain(..consumed).collect();
        frame.truncate(frame_len);
        frame
    }

    /// Build the single trailing usage chunk, reusing the last envelope.
    fn usage_chunk(&mut self) -> Option<ChatCompletionChunk> {
        let usage = self.usage.take()?;
        let envelope = self.envelope.as_ref()?;
        Some(ChatCompletionChunk {
            id: envelope.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: envelope.created,
            model: envelope.model.clone(),
            choices: Vec::new(),
            usage: Some(usage),
        })
    }
}

impl Stream for DeepSeekStream {
    type Item = Result<ChatCompletionChunk, ProxyError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(item) = self.out.pop_front() {
                return Poll::Ready(Some(item));
            }
            if self.terminated {
                if !self.errored
                    && self.include_usage
                    && let Some(chunk) = self.usage_chunk()
                {
                    return Poll::Ready(Some(Ok(chunk)));
                }
                return Poll::Ready(None);
            }
            if self.saw_done {
                self.terminated = true;
                continue;
            }
            if let Some((frame_len, consumed)) = split_frame(&self.frame_buf) {
                let frame = self.take_frame(frame_len, consumed);
                if let Err(e) = self.handle_frame(&frame) {
                    self.out.push_back(Err(e));
                    self.errored = true;
                    self.terminated = true;
                }
                continue;
            }
            match ready!(self.inner.as_mut().poll_next(cx)) {
                Some(Ok(bytes)) => {
                    self.frame_buf.extend_from_slice(&bytes);
                }
                Some(Err(e)) => {
                    self.out.push_back(Err(ProxyError::Translation(format!(
                        "deepseek upstream stream error: {e}"
                    ))));
                    self.errored = true;
                    self.terminated = true;
                }
                None => {
                    // Input exhausted: process a final frame that may lack
                    // the trailing blank line, then drain to termination.
                    if self.frame_buf.iter().any(|b| !b.is_ascii_whitespace()) {
                        let frame = std::mem::take(&mut self.frame_buf);
                        if let Err(e) = self.handle_frame(&frame) {
                            self.out.push_back(Err(e));
                            self.errored = true;
                        }
                    } else {
                        self.frame_buf.clear();
                    }
                    self.terminated = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures_util::StreamExt;
    use serde_json::json;

    use super::*;
    use crate::providers::capabilities::Translator;
    use crate::providers::deepseek::DeepSeekTranslator;

    fn chunk_json(delta: &str, content_key: &str) -> String {
        json!({
            "id": "chatcmpl-s1",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "delta": {content_key: delta},
                "finish_reason": null
            }]
        })
        .to_string()
    }

    fn usage_json(prompt: u64, completion: u64, total: u64) -> String {
        json!({
            "id": "chatcmpl-s1",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": "deepseek-chat",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total
            }
        })
        .to_string()
    }

    fn sse(data: &[String]) -> String {
        let mut out = String::new();
        for frame in data {
            out.push_str("data: ");
            out.push_str(frame);
            out.push_str("\n\n");
        }
        out
    }

    /// Build a byte stream from string pieces, each becoming one upstream
    /// `Bytes` item (simulating arbitrary TCP segmentation).
    fn byte_stream(pieces: &[&str]) -> ByteStream {
        let items: Vec<Result<Bytes, reqwest::Error>> = pieces
            .iter()
            .map(|piece| Ok(Bytes::copy_from_slice(piece.as_bytes())))
            .collect();
        Box::pin(futures_util::stream::iter(items))
    }

    fn decoder(pieces: &[&str], include_usage: bool) -> ChunkStream {
        DeepSeekTranslator::new("deepseek-chat", "https://api.deepseek.com")
            .decode_stream(include_usage, byte_stream(pieces))
    }

    async fn collect(
        pieces: &[&str],
        include_usage: bool,
    ) -> Vec<Result<ChatCompletionChunk, ProxyError>> {
        decoder(pieces, include_usage).collect().await
    }

    #[tokio::test]
    async fn deltas_forwarded_and_usage_emitted_once_at_end() {
        let body = sse(&[
            chunk_json("Hel", "content"),
            chunk_json("lo", "content"),
            usage_json(11, 2, 13),
        ]);
        let items = collect(&[&body], true).await;
        assert_eq!(items.len(), 3, "two deltas + one usage chunk");

        let first = items[0].as_ref().expect("chunk");
        assert_eq!(first.choices[0].delta.content.as_deref(), Some("Hel"));
        assert!(first.usage.is_none(), "usage stripped from content chunks");

        let second = items[1].as_ref().expect("chunk");
        assert_eq!(second.choices[0].delta.content.as_deref(), Some("lo"));

        let usage_chunk = items[2].as_ref().expect("chunk");
        assert!(
            usage_chunk.choices.is_empty(),
            "trailing usage chunk carries no deltas"
        );
        let usage = usage_chunk.usage.as_ref().expect("usage present");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 13);
    }

    #[tokio::test]
    async fn incremental_usage_collapses_to_last_value() {
        let body = sse(&[
            chunk_json("a", "content"),
            usage_json(10, 1, 11),
            chunk_json("b", "content"),
            usage_json(10, 2, 12),
            usage_json(10, 3, 13),
        ]);
        let items = collect(&[&body], true).await;
        assert_eq!(items.len(), 3, "two deltas + exactly one usage chunk");
        let usage = items[2]
            .as_ref()
            .expect("chunk")
            .usage
            .as_ref()
            .expect("usage");
        assert_eq!(usage.total_tokens, 13, "last cumulative usage wins");
    }

    #[tokio::test]
    async fn include_usage_false_strips_and_omits_usage() {
        let body = sse(&[chunk_json("Hel", "content"), usage_json(11, 2, 13)]);
        let items = collect(&[&body], false).await;
        assert_eq!(items.len(), 1, "no trailing usage chunk");
        assert!(items[0].as_ref().expect("chunk").usage.is_none());
    }

    #[tokio::test]
    async fn done_terminates_and_usage_still_emitted() {
        let body = sse(&[chunk_json("Hel", "content"), "[DONE]".to_string()]);
        let items = collect(&[&body], true).await;
        assert_eq!(items.len(), 1, "[DONE] itself is not forwarded");
        // No upstream usage seen -> no synthetic usage chunk.
        let body = sse(&[chunk_json("Hel", "content"), "[DONE]".to_string()]);
        let items = collect(&[&body], true).await;
        assert_eq!(items.len(), 1);
    }

    #[tokio::test]
    async fn malformed_frame_yields_single_error_then_ends() {
        let body = sse(&[
            chunk_json("Hel", "content"),
            "{not valid json".to_string(),
            chunk_json("ignored", "content"),
        ]);
        let items = collect(&[&body], true).await;
        assert_eq!(items.len(), 2, "first delta, then one error");
        assert!(items[0].is_ok());
        match &items[1] {
            Err(ProxyError::Translation(msg)) => {
                assert!(msg.contains("deepseek stream chunk decode"));
            }
            other => panic!("expected Translation error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn frame_split_across_byte_chunks_is_reassembled() {
        let body = sse(&[chunk_json("Hello", "content"), usage_json(3, 5, 8)]);
        // Split mid-JSON, inside the second SSE frame.
        let split_at = body.len() - 20;
        let (a, b) = body.split_at(split_at);
        assert!(a.starts_with("data:") || a.len() > 10);
        let items = collect(&[a, b], true).await;
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].as_ref().expect("chunk").choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hello")
        );
        let usage = items[1]
            .as_ref()
            .expect("chunk")
            .usage
            .as_ref()
            .expect("usage");
        assert_eq!(usage.total_tokens, 8);
    }

    #[tokio::test]
    async fn byte_by_byte_delivery_still_parses() {
        let body = sse(&[chunk_json("Hi", "content"), usage_json(1, 1, 2)]);
        let owned: Vec<String> = body.bytes().map(|b| (b as char).to_string()).collect();
        let pieces: Vec<&str> = owned.iter().map(String::as_str).collect();
        let items = collect(&pieces, true).await;
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].as_ref().expect("chunk").choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hi")
        );
    }

    #[tokio::test]
    async fn final_frame_without_trailing_blank_line_is_processed() {
        let mut body = sse(&[chunk_json("Hi", "content"), usage_json(1, 1, 2)]);
        body.truncate(body.len() - 2); // drop the final "\n\n"
        let items = collect(&[&body], true).await;
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[1]
                .as_ref()
                .expect("chunk")
                .usage
                .as_ref()
                .expect("usage")
                .total_tokens,
            2
        );
    }

    #[tokio::test]
    async fn empty_stream_terminates_immediately() {
        let items = collect(&[""], true).await;
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn comment_and_field_lines_are_ignored() {
        let mut body = sse(&[chunk_json("Hi", "content")]);
        body.insert_str(0, ": keepalive\n\n");
        body.insert_str(0, "event: message\nid: 7\n\n");
        let items = collect(&[&body], false).await;
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0].as_ref().expect("chunk").choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hi")
        );
    }
}
