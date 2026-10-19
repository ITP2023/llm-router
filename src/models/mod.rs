//! Canonical OpenAI-compatible DTOs shared by all providers.

pub mod chat;
pub mod list;

pub use chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice,
    ChunkChoice, ContentPart, ImageUrl, MessageContent, MessageDelta, ResponseFormat, StopSequence,
    StreamOptions, Tool, ToolCall, ToolCallFunction, ToolChoice, ToolType, Usage,
};
pub use list::{ModelEntry, ModelList};
