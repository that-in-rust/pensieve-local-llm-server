use anyhow::Result;
use code_ingest::cli::Cli;
use code_ingest::logging::{LoggingConfig, init_logging};
use std::sync::Once;

static INIT_LOGGING: Once = Once::new();

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with default configuration only once
    INIT_LOGGING.call_once(|| {
        let logging_config = LoggingConfig::default();
        if let Err(e) = init_logging(&logging_config) {
            eprintln!("Failed to initialize logging: {}", e);
        }
    });

    let cli = Cli::new();
    cli.run().await
}
