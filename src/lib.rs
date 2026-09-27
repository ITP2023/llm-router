pub mod api;
pub mod error;
pub mod models;
pub mod providers;
pub(crate) mod state;
pub mod streaming;

pub use state::{AppState, DeepSeekTarget, MockLLMTarget, ModelRegistry, ProviderTarget};

use axum::Router;

/// Assemble the full router with shared state (design D6). Integration
/// tests build the app in-process via this function — no port binding.
pub fn app(state: AppState) -> Router {
    api::routes().with_state(state)
}
