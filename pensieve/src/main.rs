use pensieve::cli::Cli;
use pensieve::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.run().await
}