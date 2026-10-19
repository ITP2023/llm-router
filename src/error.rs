//! Central error type for the proxy. Every failure class the router produces
//! directly maps to a canonical OpenAI-shaped error response
//! (`{"error": {"message", "type", "code"}}`) with a mapped HTTP status
//! (spec: "OpenAI-shaped error responses", design D5).

use std::fmt;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug)]
pub enum ProxyError {
    /// The client's request body was not valid canonical request JSON.
    MalformedRequest(String),
    /// The requested model alias is not in the router's model map.
    UnknownModel(String),
    /// The upstream endpoint could not be reached (connect error, timeout).
    UpstreamUnavailable(String),
    /// The upstream returned an error status; the body is mapped, never
    /// passed through unmodified.
    UpstreamStatus(u16, String),
    /// A provider adapter failed to encode/decode a payload.
    Translation(String),
    /// Any other internal failure.
    Internal(String),
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedRequest(msg) => write!(f, "malformed request: {msg}"),
            Self::UnknownModel(alias) => write!(f, "unknown model alias: {alias}"),
            Self::UpstreamUnavailable(msg) => write!(f, "upstream unavailable: {msg}"),
            Self::UpstreamStatus(status, msg) => {
                write!(f, "upstream returned status {status}: {msg}")
            }
            Self::Translation(msg) => write!(f, "translation error: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for ProxyError {}

impl ProxyError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::MalformedRequest(_) => StatusCode::BAD_REQUEST,
            Self::UnknownModel(_) => StatusCode::NOT_FOUND,
            Self::UpstreamUnavailable(_) => StatusCode::BAD_GATEWAY,
            Self::UpstreamStatus(status, _) => {
                if *status == 429 {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    // 5xx and any other upstream status surface as 502.
                    StatusCode::BAD_GATEWAY
                }
            }
            Self::Translation(_) | Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::MalformedRequest(_) | Self::UnknownModel(_) => "invalid_request_error",
            Self::UpstreamStatus(_, _) => "upstream_error",
            Self::UpstreamUnavailable(_) | Self::Translation(_) | Self::Internal(_) => {
                "server_error"
            }
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = self.error_type();
        // Client-visible message: the upstream body is deliberately not
        // embedded (spec: "Upstream failure surfaced canonically" — provider
        // bodies must not leak unmodified); it stays available for logs via
        // Display.
        let message = match &self {
            Self::UpstreamStatus(status, _) => {
                format!("upstream returned status {status}")
            }
            _ => self.to_string(),
        };
        let body = json!({
            "error": {
                "message": message,
                "type": error_type,
                "code": null,
            }
        });
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::response::IntoResponse;

    use super::*;

    async fn render(err: ProxyError) -> (StatusCode, serde_json::Value) {
        let response = err.into_response();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error body should collect");
        let json: serde_json::Value =
            serde_json::from_slice(&body).expect("error body should be JSON");
        (status, json)
    }

    fn assert_error_shape(json: &serde_json::Value, error_type: &str) {
        let error = &json["error"];
        assert!(error["message"].is_string());
        assert_eq!(error["type"], error_type);
        assert!(
            error
                .as_object()
                .expect("error object")
                .contains_key("code")
        );
    }

    #[tokio::test]
    async fn malformed_request_is_400_invalid_request() {
        let (status, json) = render(ProxyError::MalformedRequest("bad json".into())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_error_shape(&json, "invalid_request_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .expect("message")
                .contains("bad json")
        );
    }

    #[tokio::test]
    async fn unknown_model_is_404_invalid_request() {
        let (status, json) = render(ProxyError::UnknownModel("gpt-99".into())).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_error_shape(&json, "invalid_request_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .expect("message")
                .contains("gpt-99")
        );
    }

    #[tokio::test]
    async fn upstream_unavailable_is_502_server_error() {
        let (status, json) =
            render(ProxyError::UpstreamUnavailable("connection refused".into())).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_error_shape(&json, "server_error");
    }

    #[tokio::test]
    async fn upstream_status_429_is_preserved() {
        let (status, json) = render(ProxyError::UpstreamStatus(429, "rate limited".into())).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_error_shape(&json, "upstream_error");
    }

    #[tokio::test]
    async fn upstream_status_5xx_maps_to_502() {
        for upstream in [500u16, 502, 503, 599] {
            let (status, json) = render(ProxyError::UpstreamStatus(upstream, "boom".into())).await;
            assert_eq!(status, StatusCode::BAD_GATEWAY, "upstream {upstream}");
            assert_error_shape(&json, "upstream_error");
        }
    }

    #[tokio::test]
    async fn upstream_status_other_maps_to_502() {
        let (status, json) = render(ProxyError::UpstreamStatus(400, "bad upstream".into())).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_error_shape(&json, "upstream_error");
    }

    #[tokio::test]
    async fn upstream_status_body_not_leaked_to_client() {
        let (status, json) = render(ProxyError::UpstreamStatus(
            500,
            "secret provider internals".into(),
        ))
        .await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(
            !json["error"]["message"]
                .as_str()
                .expect("message")
                .contains("secret provider internals"),
            "provider body must not leak unmodified"
        );
        // Display still carries the body for logs.
        let logged = ProxyError::UpstreamStatus(500, "secret provider internals".into());
        assert!(logged.to_string().contains("secret provider internals"));
    }

    #[tokio::test]
    async fn translation_is_500_server_error() {
        let (status, json) = render(ProxyError::Translation("bad payload".into())).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_error_shape(&json, "server_error");
    }

    #[tokio::test]
    async fn internal_is_500_server_error() {
        let (status, json) = render(ProxyError::Internal("oops".into())).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_error_shape(&json, "server_error");
    }
}
