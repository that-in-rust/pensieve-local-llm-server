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
                
                // Check if database exists
                if !database.exists() {
                    return Err(PensieveError::CliArgument(format!(
                        "Database file does not exist: {}",
                        database.display()
                    )));
                }
                
                // Connect to database and get statistics
                let db = crate::database::Database::new(&database).await?;
                let stats = db.get_statistics().await?;
                let dup_stats = db.get_duplicate_statistics().await?;
                
                // Display comprehensive statistics
                println!("\n=== Pensieve Database Statistics ===");
                
                println!("\nFile Statistics:");
                println!("  Total files: {}", stats.total_files);
                println!("  Unique files: {}", stats.unique_files);
                println!("  Canonical files: {}", dup_stats.canonical_files);
                println!("  Duplicate files: {}", stats.duplicate_files);
                println!("  Duplicate groups: {}", dup_stats.duplicate_groups);
                
                if stats.total_files > 0 {
                    let dedup_rate = (stats.duplicate_files as f64 / stats.total_files as f64) * 100.0;
                    println!("  Deduplication rate: {:.1}%", dedup_rate);
                }
                
                println!("\nStorage Statistics:");
                println!("  Total size: {:.2} MB", dup_stats.total_size as f64 / 1_048_576.0);
                println!("  Duplicate size: {:.2} MB", dup_stats.duplicate_size as f64 / 1_048_576.0);
                println!("  Space savings: {:.2} MB", dup_stats.space_savings as f64 / 1_048_576.0);
                
                if dup_stats.total_size > 0 {
                    let savings_rate = (dup_stats.space_savings as f64 / dup_stats.total_size as f64) * 100.0;
                    println!("  Space savings rate: {:.1}%", savings_rate);
                }
                
                println!("\nContent Statistics:");
                println!("  Total paragraphs: {}", stats.total_paragraphs);
                println!("  Estimated tokens: {}", stats.total_tokens);
                
                if stats.total_tokens > 0 && stats.total_files > 0 {
                    println!("  Average tokens per file: {:.0}", stats.total_tokens as f64 / stats.total_files as f64);
                }
                
                println!("\nProcessing Status:");
                for (status, count) in &stats.files_by_status {
                    println!("  {}: {}", status, count);
                }
                
                if stats.error_count > 0 {
                    println!("\nErrors:");
                    println!("  Processing errors: {}", stats.error_count);
                }
                
                // Show top duplicate groups if any exist
                if dup_stats.duplicate_groups > 0 {
                    println!("\n=== Top Duplicate Groups ===");
                    let dedup_service = crate::deduplication::DeduplicationService::new(db);
                    let groups = dedup_service.list_duplicate_groups().await?;
                    
                    for (i, group) in groups.iter().take(10).enumerate() {
                        println!("{}. Group {} ({} files, {:.2} MB)", 
                            i + 1,
                            group.group_id,
                            group.file_count,
                            group.total_size as f64 / 1_048_576.0
                        );
                        println!("   Canonical: {}", group.canonical_path.display());
                        println!("   Hash: {}", &group.hash[..16]);
                    }
                    
                    if groups.len() > 10 {
                        println!("   ... and {} more groups", groups.len() - 10);
                    }
                }
                
                Ok(())
            }
        }
    }

    /// Run the main ingestion workflow
    async fn run_ingestion(&self) -> Result<()> {
        use crate::database::Database;
        use crate::scanner::FileScanner;
        
        let input = self.input.as_ref().unwrap(); // Safe because validate_args() checks this
        let database_path = self.database.as_ref().unwrap(); // Safe because validate_args() checks this

        if self.verbose {
            println!("Starting Pensieve ingestion...");
            println!("Input directory: {}", input.display());
            println!("Database: {}", database_path.display());
            if self.dry_run {
                println!("DRY RUN MODE - No changes will be made");
            }
        }

        // Phase 1: Initialize database connection
        println!("Phase 1: Initializing database...");
        let database = Database::new(database_path).await?;
        database.initialize_schema().await?;
        
        if self.verbose {
            println!("Database initialized successfully");
            let stats = database.pool_stats();
            println!("Connection pool: {} connections ({} idle)", stats.size, stats.idle);
        }

        // Phase 2: Run metadata scanning phase
        println!("Phase 2: Scanning files and extracting metadata...");
        let scanner = FileScanner::new(input);
        let file_metadata = scanner.scan().await?;
        
        println!("Metadata scanning complete: {} files discovered", file_metadata.len());
        
        // Show summary statistics
        let unique_files = file_metadata.iter()
            .filter(|f| f.duplicate_status == crate::types::DuplicateStatus::Unique || 
                       f.duplicate_status == crate::types::DuplicateStatus::Canonical)
            .count();
        let duplicate_files = file_metadata.len() - unique_files;
        let total_size: u64 = file_metadata.iter().map(|f| f.size).sum();
        
        println!("Summary:");
        println!("  Total files: {}", file_metadata.len());
        println!("  Unique files: {}", unique_files);
        println!("  Duplicate files: {}", duplicate_files);
        println!("  Total size: {:.2} MB", total_size as f64 / 1_048_576.0);
        
        if duplicate_files > 0 {
            let dedup_percentage = (duplicate_files as f64 / file_metadata.len() as f64) * 100.0;
            println!("  Deduplication rate: {:.1}%", dedup_percentage);
        }

        if !self.dry_run {
            // Phase 3: Store metadata in database with enhanced deduplication
            println!("Phase 3: Processing deduplication and storing metadata...");
            
            let dedup_service = crate::deduplication::DeduplicationService::new(database.clone());
            let processed_files = dedup_service.process_duplicates(file_metadata).await?;
            
            // Store files in database using batch operations
            println!("Storing {} files in database...", processed_files.len());
            database.insert_files_batch(&processed_files).await?;
            
            // Get final statistics from database
            let db_stats = database.get_statistics().await?;
            let dup_stats = database.get_duplicate_statistics().await?;
            
            println!("\nDatabase Storage Complete:");
            println!("  Files stored: {}", db_stats.total_files);
            println!("  Unique files: {}", db_stats.unique_files);
            println!("  Duplicate files: {}", db_stats.duplicate_files);
            println!("  Duplicate groups: {}", dup_stats.duplicate_groups);
            
            if dup_stats.space_savings > 0 {
                println!("  Space savings: {:.2} MB", dup_stats.space_savings as f64 / 1_048_576.0);
            }
            
            // TODO: Phase 4: Content processing
            println!("\nPhase 4: Content processing...");
            println!("Content processing will be implemented in the next task");
        } else {
            println!("DRY RUN: Skipping database operations");
        }

        println!("Ingestion workflow completed successfully!");
        Ok(())
    }
}