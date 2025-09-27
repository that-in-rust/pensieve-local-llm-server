use anyhow::Result;
use code_ingest::cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::new();
    cli.run().await
}
