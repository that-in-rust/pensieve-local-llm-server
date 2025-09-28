use anyhow::Result;
use code_ingest::cli::Cli;
use code_ingest::logging::{LoggingConfig, init_logging};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with default configuration
    // This can be overridden by environment variables or config files
    let logging_config = LoggingConfig::default();
    if let Err(e) = init_logging(&logging_config) {
        eprintln!("Failed to initialize logging: {}", e);
        // Continue without structured logging
    }

    let cli = Cli::new();
    cli.run().await
}
