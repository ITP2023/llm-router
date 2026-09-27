//! Server entrypoint: tracing init, env config, app assembly, and the
//! listener with graceful shutdown (spec: "Server startup and graceful
//! shutdown", design D6).

use llm_router::providers::mockllm;
use llm_router::{AppState, ModelRegistry, app};

#[tokio::main]
async fn main() {
    // Load .env into the process environment (no-op if the file is absent);
    // real environment variables always win over the file.
    let _ = dotenvy::dotenv();

    init_tracing();

    let listen_addr = std::env::var("LLM_ROUTER_LISTEN").unwrap_or_else(|_| "0.0.0.0:8080".into());

    // Provider selection: `deepseek` (default, needs an API key) or
    // `mockllm`, which spawns the in-process test upstream so the whole
    // proxy can be exercised with `cargo run` and no credentials.
    let provider = std::env::var("LLM_ROUTER_PROVIDER").unwrap_or_else(|_| "deepseek".into());

    let (registry, mut mock_upstream) = if provider == "mockllm" {
        let aliases = std::env::var("LLM_ROUTER_MODELS").unwrap_or_else(|_| "mockllm".into());
        let aliases: Vec<&str> = aliases.split(',').map(str::trim).collect();
        match mockllm::server::spawn(mockllm::server::Config::default()).await {
            Ok(upstream) => {
                let base_url = upstream.base_url().to_string();
                match ModelRegistry::mockllm(&base_url, &aliases) {
                    Ok(registry) => {
                        tracing::info!(upstream = %base_url, "mockllm upstream spawned in-process");
                        (registry, Some(upstream))
                    }
                    Err(err) => {
                        eprintln!("llm-router: {err}");
                        std::process::exit(1);
                    }
                }
            }
            Err(err) => {
                eprintln!("llm-router: cannot spawn mockllm upstream: {err}");
                std::process::exit(1);
            }
        }
    } else {
        let registry = match ModelRegistry::from_env() {
            Ok(registry) => registry,
            Err(err) => {
                tracing::error!(error = %err, "failed to build model registry");
                eprintln!("llm-router: {err}");
                std::process::exit(1);
            }
        };
        (registry, None)
    };

    let state = AppState {
        http: reqwest::Client::new(),
        registry,
    };
    let router = app(state);

    let listener = match tokio::net::TcpListener::bind(&listen_addr).await {
        Ok(listener) => listener,
        Err(err) => {
            tracing::error!(addr = %listen_addr, error = %err, "failed to bind listener");
            eprintln!("llm-router: cannot bind {listen_addr}: {err}");
            std::process::exit(1);
        }
    };
    tracing::info!(addr = %listen_addr, "llm-router listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap_or_else(|err| {
            tracing::error!(error = %err, "server exited with error");
            eprintln!("llm-router: serve error: {err}");
            std::process::exit(1);
        });

    if let Some(upstream) = mock_upstream.take() {
        upstream.stop().await;
    }
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("llm_router=info,tower_http=info"));
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
}

/// Stop accepting new connections on SIGTERM/SIGINT; in-flight requests
/// complete before exit (spec: "Shutdown drains in-flight requests").
async fn shutdown_signal() {
    use tokio::signal;

    #[cfg(unix)]
    async fn terminate() {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    }

    #[cfg(not(unix))]
    async fn terminate() {
        std::future::pending::<()>().await;
    }

    tokio::select! {
        _ = signal::ctrl_c() => {},
        () = terminate() => {},
    }
    tracing::info!("shutdown signal received, draining");
}
