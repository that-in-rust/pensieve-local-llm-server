//! Command-line interface for the Pensieve tool

use crate::prelude::*;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Pensieve - A CLI tool for ingesting text files into a deduplicated database for LLM processing
#[derive(Parser, Debug)]
#[command(name = "pensieve")]
#[command(about = "A CLI tool for ingesting text files into a deduplicated database for LLM processing")]
#[command(version)]
pub struct Cli {
    /// Input directory to scan for files
    #[arg(short, long, value_name = "DIR")]
    pub input: Option<PathBuf>,

    /// Database file path (SQLite)
    #[arg(short, long, value_name = "FILE")]
    pub database: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Dry run - don't modify database
    #[arg(long)]
    pub dry_run: bool,

    /// Force reprocess all files (ignore delta checks)
    #[arg(long)]
    pub force_reprocess: bool,

    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Subcommands
    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Available subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Initialize a new database with schema
    Init {
        /// Database file path
        #[arg(short, long, value_name = "FILE")]
        database: PathBuf,
    },
    /// Check external tool dependencies
    CheckDeps,
    /// Generate default configuration file
    Config {
        /// Output path for configuration file
        #[arg(short, long, value_name = "FILE", default_value = "pensieve.toml")]
        output: PathBuf,
    },
    /// Show statistics about processed data
    Stats {
        /// Database file path
        #[arg(short, long, value_name = "FILE")]
        database: PathBuf,
    },
}

impl Cli {
    /// Parse CLI arguments from command line
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }

    /// Run the CLI application
    pub async fn run(self) -> Result<()> {
        // Handle subcommands first
        if let Some(ref command) = self.command {
            return self.run_subcommand(command.clone()).await;
        }

        // Validate required arguments for main operation
        self.validate_args()?;

        // Main ingestion workflow
        self.run_ingestion().await
    }

    /// Validate CLI arguments for main ingestion operation
    fn validate_args(&self) -> Result<()> {
        // Check required arguments for main operation
        let input = self.input.as_ref().ok_or_else(|| {
            PensieveError::CliArgument("Input directory is required for ingestion".to_string())
        })?;

        let database = self.database.as_ref().ok_or_else(|| {
            PensieveError::CliArgument("Database path is required for ingestion".to_string())
        })?;

        // Check input directory exists
        if !input.exists() {
            return Err(PensieveError::CliArgument(format!(
                "Input directory does not exist: {}",
                input.display()
            )));
        }

        if !input.is_dir() {
            return Err(PensieveError::CliArgument(format!(
                "Input path is not a directory: {}",
                input.display()
            )));
        }

        // Check database directory is writable
        let db_parent = if let Some(parent) = database.parent() {
            if parent.as_os_str().is_empty() {
                // Empty parent means current directory
                std::env::current_dir().map_err(|e| {
                    PensieveError::CliArgument(format!("Cannot access current directory: {}", e))
                })?
            } else {
                parent.to_path_buf()
            }
        } else {
            // No parent means current directory
            std::env::current_dir().map_err(|e| {
                PensieveError::CliArgument(format!("Cannot access current directory: {}", e))
            })?
        };

        if !db_parent.exists() {
            return Err(PensieveError::CliArgument(format!(
                "Database directory does not exist: {}",
                db_parent.display()
            )));
        }

        Ok(())
    }

    /// Run a subcommand
    async fn run_subcommand(&self, command: Commands) -> Result<()> {
        match command {
            Commands::Init { database } => {
                println!("Initializing database at: {}", database.display());
                // TODO: Implement database initialization
                Ok(())
            }
            Commands::CheckDeps => {
                println!("Checking external tool dependencies...");
                // TODO: Implement dependency checking
                Ok(())
            }
            Commands::Config { output } => {
                println!("Generating configuration file at: {}", output.display());
                // TODO: Implement config generation
                Ok(())
            }
            Commands::Stats { database } => {
                println!("Showing statistics for database: {}", database.display());
                // TODO: Implement statistics display
                Ok(())
            }
        }
    }

    /// Run the main ingestion workflow
    async fn run_ingestion(&self) -> Result<()> {
        let input = self.input.as_ref().unwrap(); // Safe because validate_args() checks this
        let database = self.database.as_ref().unwrap(); // Safe because validate_args() checks this

        if self.verbose {
            println!("Starting Pensieve ingestion...");
            println!("Input directory: {}", input.display());
            println!("Database: {}", database.display());
            if self.dry_run {
                println!("DRY RUN MODE - No changes will be made");
            }
        }

        // TODO: Implement main ingestion workflow
        // 1. Initialize database connection
        // 2. Run metadata scanning phase
        // 3. Run content processing phase
        // 4. Generate final report

        println!("Ingestion workflow not yet implemented");
        Ok(())
    }
}