//! # Pensieve HTTP API Server
//!
//! Binary entry point for the MoA-Lite debate HTTP server.
//!
//! ## Usage
//!
//! ```bash
//! # Start with default settings (connects to llama-server at localhost:8080)
//! pensieve-server
//!
//! # Start with custom port
//! pensieve-server --port 8000
//!
//! # Start in mock mode (for testing without llama-server)
//! pensieve-server --mock
//!
//! # Start with custom llama-server URL
//! pensieve-server --llm-url http://localhost:9000
//! ```
//!
//! ## API Endpoints
//!
//! - `POST /v1/chat/completions` - OpenAI-compatible chat completions
//! - `GET /health` - Health check

use std::process::ExitCode;

use clap::Parser;
use pensieve_http_api_server::{ServerConfig, start_server_with_config};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Pensieve MoA-Lite Debate Server
#[derive(Parser, Debug)]
#[command(name = "pensieve-server")]
#[command(author, version, about = "Multi-Agent Debate AI Assistant HTTP Server")]
struct CliArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 3000)]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// llama-server URL
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    llm_url: String,

    /// Use mock LLM backend (for testing)
    #[arg(long, default_value_t = false)]
    mock: bool,
}

fn setup_tracing_subscriber_global() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pensieve_http_api_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[tokio::main]
async fn main() -> ExitCode {
    setup_tracing_subscriber_global();

    let args = CliArgs::parse();

    info!(
        port = args.port,
        host = %args.host,
        llm_url = %args.llm_url,
        mock = args.mock,
        "Starting Pensieve server"
    );

    let config = if args.mock {
        let mock_response = r#"
Based on the multi-agent debate among our proposers, here is the synthesized answer:

The current Home Minister of India is **Amit Shah**. He has been serving in this role since May 30, 2019, when he was appointed following the BJP's victory in the 2019 general elections.

Key facts about Amit Shah as Home Minister:
- Full name: Amit Anil Chandra Shah
- Political party: Bharatiya Janata Party (BJP)
- Previous role: President of BJP (2014-2020)
- Notable for: Implementation of Article 370 abrogation in Jammu & Kashmir
- Also serves as: Minister of Cooperation (since 2021)

The Ministry of Home Affairs under his leadership handles internal security, border management, and administration of Union Territories.
"#;
        ServerConfig {
            port: args.port,
            host: args.host,
            llm_backend_url: "mock://localhost".to_string(),
            use_mock_llm: true,
            mock_response: mock_response.to_string(),
        }
    } else {
        ServerConfig {
            port: args.port,
            host: args.host,
            llm_backend_url: args.llm_url,
            use_mock_llm: false,
            mock_response: String::new(),
        }
    };

    match start_server_with_config(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Server error: {e}");
            ExitCode::FAILURE
        }
    }
}
