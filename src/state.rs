//! Shared application state: the long-lived upstream HTTP client and the
//! static model registry (design D6).
//!
//! [`AppState`] is built once at startup and cheap to `Clone` (the
//! `reqwest::Client` is an `Arc` handle over a shared connection pool).
//!
//! [`ModelRegistry`] maps client-facing model aliases to a [`ProviderTarget`].
//! This change ships exactly one provider, but the enum (plus the
//! [`Translator`] seam in `src/providers/`) is the extension point: adding a
//! second provider means adding a variant with its own request-URL builder and
//! API-key source — no handler code changes.

use std::collections::HashMap;

use crate::error::ProxyError;
use crate::providers::capabilities::Translator;
use crate::providers::deepseek::DeepSeekTranslator;
use crate::providers::mockllm::MockLLMTranslator;

/// Shared state for all handlers, attached to the axum router via
/// `.with_state(...)`.
#[derive(Clone)]
pub struct AppState {
    /// Long-lived upstream client; the connection pool must never be rebuilt
    /// per request (spec: "Server startup and graceful shutdown").
    pub http: reqwest::Client,
    /// Static alias -> provider map used for model resolution.
    pub registry: ModelRegistry,
}

/// A configured upstream a model alias routes to.
#[derive(Clone)]
pub enum ProviderTarget {
    /// DeepSeek's OpenAI-compatible API.
    DeepSeek(DeepSeekTarget),
    /// In-process MockLLM upstream for end-to-end testing (no credentials).
    MockLLM(MockLLMTarget),
}

/// DeepSeek-specific target configuration: the translator plus the
/// credential used to call it.
#[derive(Clone)]
pub struct DeepSeekTarget {
    translator: DeepSeekTranslator,
    api_key: String,
}

impl DeepSeekTarget {
    pub fn new(translator: DeepSeekTranslator, api_key: impl Into<String>) -> Self {
        Self {
            translator,
            api_key: api_key.into(),
        }
    }
}

/// MockLLM target configuration: the in-process test upstream needs no
/// credential, so this holds only the translator.
#[derive(Clone)]
pub struct MockLLMTarget {
    translator: MockLLMTranslator,
}

impl MockLLMTarget {
    pub fn new(translator: MockLLMTranslator) -> Self {
        Self { translator }
    }
}

impl ProviderTarget {
    /// The translation seam every handler goes through (design D3).
    pub fn translator(&self) -> &dyn Translator {
        match self {
            Self::DeepSeek(target) => &target.translator,
            Self::MockLLM(target) => &target.translator,
        }
    }

    /// Provider-native chat-completions endpoint for this target.
    pub fn chat_completions_url(&self) -> String {
        match self {
            Self::DeepSeek(target) => {
                format!(
                    "{}/chat/completions",
                    target.translator.base_url().trim_end_matches('/')
                )
            }
            Self::MockLLM(target) => {
                format!(
                    "{}/chat/completions",
                    target.translator.base_url().trim_end_matches('/')
                )
            }
        }
    }

    /// Bearer credential for upstream calls.
    pub fn api_key(&self) -> &str {
        match self {
            Self::DeepSeek(target) => &target.api_key,
            // The in-process mock upstream ignores authorization.
            Self::MockLLM(_) => "",
        }
    }

    /// Provider name surfaced as `owned_by` on `/v1/models`.
    pub fn provider_name(&self) -> &'static str {
        match self {
            Self::DeepSeek(_) => "deepseek",
            Self::MockLLM(_) => "mockllm",
        }
    }
}

/// Static model-alias map (design D6). For this change every alias owns its
/// own translator instance (the upstream model name is baked into the
/// translator); a later optimization could share one DeepSeek target across
/// aliases that differ only in upstream model.
#[derive(Clone, Default)]
pub struct ModelRegistry {
    entries: HashMap<String, ProviderTarget>,
}

impl ModelRegistry {
    /// Build the registry from the environment:
    /// - `LLM_ROUTER_DEEPSEEK_API_KEY` (required): bearer credential.
    /// - `LLM_ROUTER_DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`).
    /// - `LLM_ROUTER_MODELS`: comma-separated bare model names (default
    ///   `deepseek-chat,deepseek-reasoner`), each registered under the
    ///   namespaced id `deepseek/<model>`.
    pub fn from_env() -> Result<Self, ProxyError> {
        let api_key = std::env::var("LLM_ROUTER_DEEPSEEK_API_KEY").map_err(|_| {
            ProxyError::Internal("LLM_ROUTER_DEEPSEEK_API_KEY is not set".to_string())
        })?;
        let base_url = std::env::var("LLM_ROUTER_DEEPSEEK_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
        let models = std::env::var("LLM_ROUTER_MODELS")
            .unwrap_or_else(|_| "deepseek-chat,deepseek-reasoner".to_string());

        let mut registry = Self::default();
        for alias in models
            .split(',')
            .map(str::trim)
            .filter(|alias| !alias.is_empty())
        {
            // The registry key is the namespaced model id (`deepseek/<model>`);
            // the translator keeps the bare upstream model name for the wire.
            registry.insert(
                format!("deepseek/{alias}"),
                ProviderTarget::DeepSeek(DeepSeekTarget::new(
                    DeepSeekTranslator::new(alias, &base_url),
                    &api_key,
                )),
            );
        }
        if registry.entries.is_empty() {
            return Err(ProxyError::Internal(
                "LLM_ROUTER_MODELS did not contain any model aliases".to_string(),
            ));
        }
        Ok(registry)
    }

    /// Empty registry; combine with [`ModelRegistry::insert`] in tests (or a
    /// future file-based config path) to avoid depending on the environment.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a registry whose namespaced ids (`mockllm/<model>`) all route
    /// to a MockLLM upstream at `base_url` (as served by
    /// [`crate::providers::mockllm::server::spawn`]). Used for end-to-end
    /// testing without network access or API keys; `main` selects this via
    /// `LLM_ROUTER_PROVIDER=mockllm`.
    pub fn mockllm(base_url: &str, aliases: &[&str]) -> Result<Self, ProxyError> {
        if aliases.is_empty() {
            return Err(ProxyError::Internal(
                "mockllm registry requires at least one model alias".to_string(),
            ));
        }
        let mut registry = Self::default();
        for alias in aliases {
            // Registry key is the namespaced model id (`mockllm/<model>`);
            // the translator keeps the bare upstream model name for the wire.
            registry.insert(
                format!("mockllm/{alias}"),
                ProviderTarget::MockLLM(MockLLMTarget::new(MockLLMTranslator::new(
                    *alias, base_url,
                ))),
            );
        }
        Ok(registry)
    }

    /// Register an alias -> target mapping.
    pub fn insert(&mut self, alias: String, target: ProviderTarget) {
        self.entries.insert(alias, target);
    }

    /// Resolve an alias to its target.
    pub fn resolve(&self, alias: &str) -> Option<&ProviderTarget> {
        self.entries.get(alias)
    }

    /// Every configured alias, unordered.
    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::capabilities::CapabilitySet;

    #[test]
    fn insert_and_resolve_round_trip() {
        let mut registry = ModelRegistry::empty();
        registry.insert(
            "deepseek/deepseek-chat".to_string(),
            ProviderTarget::DeepSeek(DeepSeekTarget::new(
                DeepSeekTranslator::new("deepseek-chat", "https://api.deepseek.com"),
                "key",
            )),
        );
        let target = registry
            .resolve("deepseek/deepseek-chat")
            .expect("alias resolves");
        assert_eq!(target.provider_name(), "deepseek");
        assert_eq!(
            target.chat_completions_url(),
            "https://api.deepseek.com/chat/completions"
        );
        assert_eq!(target.api_key(), "key");
        assert!(registry.resolve("nope").is_none());
        // A bare upstream model name without the provider namespace must not
        // resolve: the namespaced id is the canonical form.
        assert!(registry.resolve("deepseek-chat").is_none());
    }

    #[test]
    fn mockllm_registry_namespaces_ids() {
        let registry = ModelRegistry::mockllm("http://127.0.0.1:9", &["chat", "reasoner"])
            .expect("registry builds");
        assert!(registry.resolve("mockllm/chat").is_some());
        assert!(registry.resolve("mockllm/reasoner").is_some());
        assert!(registry.resolve("chat").is_none());
        let mut aliases = registry.aliases().collect::<Vec<_>>();
        aliases.sort_unstable();
        assert_eq!(aliases, vec!["mockllm/chat", "mockllm/reasoner"]);
    }

    #[test]
    fn trailing_slash_in_base_url_is_normalized() {
        let registry_entry = ProviderTarget::DeepSeek(DeepSeekTarget::new(
            DeepSeekTranslator::new("m", "http://127.0.0.1:1234/"),
            "key",
        ));
        assert_eq!(
            registry_entry.chat_completions_url(),
            "http://127.0.0.1:1234/chat/completions"
        );
    }

    #[test]
    fn test_translator_can_override_capabilities() {
        // The /v1 chat-completions wiremock tests configure a logprobs=false
        // translator to exercise x-dropped-response-fields; make sure the
        // override constructor reaches the trait.
        let translator = DeepSeekTranslator::with_capabilities(
            "m",
            "https://api.deepseek.com",
            CapabilitySet {
                logprobs: false,
                ..CapabilitySet::default()
            },
        );
        assert!(!translator.capabilities().logprobs);
    }
}
