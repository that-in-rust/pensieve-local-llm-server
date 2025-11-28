//! Pensieve CLI Binary Entry Point
//!
//! This is the main entry point for the Pensieve CLI application.

use clap::Parser;
use pensieve_01::{CliArgs, PensieveCli, traits::CliApp};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = CliArgs::parse();

    // Create CLI application
    let cli = PensieveCli::new(args)?;

    // Run the application
    let exit_code = cli.run().await?;

    // Exit with the appropriate code
    std::process::exit(exit_code);
}