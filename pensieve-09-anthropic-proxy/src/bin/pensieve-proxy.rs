//! Pensieve Anthropic Proxy Server Binary
//!
//! Simple binary to start the Anthropic-compatible proxy server.
//!
//! Usage:
//!   cargo run --bin pensieve-proxy
//!   cargo run --bin pensieve-proxy --release

use pensieve_09_anthropic_proxy::{AnthropicProxyServer, ServerConfig};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    // Use default config (127.0.0.1:7777)
    let config = ServerConfig::default();

    println!("🚀 Starting Pensieve Anthropic Proxy Server");
    println!("   Host: {}", config.host);
    println!("   Port: {}", config.port);
    println!("   Model: {}", config.model_path);
    println!();
    println!("📝 To configure Claude Code, run:");
    println!("   ./scripts/setup-claude-code.sh");
    println!();
    println!("🧪 To test with curl:");
    println!("   curl -X POST http://127.0.0.1:7777/v1/messages \\");
    println!("     -H 'Authorization: Bearer pensieve-local-token' \\");
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{\"model\":\"claude-3-sonnet-20240229\",\"max_tokens\":100,\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'");
    println!();

    // Create and start server
    let server = AnthropicProxyServer::new(config);
    server.start().await?;

    // Wait for Ctrl+C
    println!("✅ Server running. Press Ctrl+C to shutdown.");
    tokio::signal::ctrl_c().await?;

    // Shutdown gracefully
    println!("\n🛑 Shutting down server...");
    server.shutdown().await?;

    println!("👋 Server stopped.");
    Ok(())
}
