//! Upstream provider translation adapters. Provider-native payloads never
//! escape this module boundary: everything translates canonical -> provider
//! -> canonical through the [`Translator`] seam (design D3).

pub mod capabilities;
pub mod deepseek;
// @human
pub mod mockllm;

pub use capabilities::{
    ByteStream, CapabilitySet, ChunkStream, DropAction, FieldDrop, Translator, diff_request,
};
pub use deepseek::DeepSeekTranslator;
// @human
pub use mockllm::MockLLMTranslator;
