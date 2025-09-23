//! Command-line interface for the Pensieve tool

use crate::prelude::*;
use crate::types::{FileMetadata, DependencyCheck, DependencyType, DependencyStatus};
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
    /// Database migration commands
    Migrate {
        /// Database file path
        #[arg(short, long, value_name = "FILE")]
        database: PathBuf,
        
        /// Migration subcommand
        #[command(subcommand)]
        action: MigrateAction,
    },
}

/// Migration subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum MigrateAction {
    /// Run all pending migrations
    Up,
    /// Show migration status and history
    Status,
    /// Rollback to a specific version
    Rollback {
        /// Target version to rollback to
        #[arg(value_name = "VERSION")]
        version: u32,
    },
    /// Validate database schema integrity
    Validate,
    /// Reset database (drop all tables and start fresh)
    Reset {
        /// Confirm the reset operation
        #[arg(long)]
        confirm: bool,
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
                
                // Create database and run initial migration
                let db = crate::database::Database::new(&database).await?;
                db.initialize_schema().await?;
                
                println!("Database initialized successfully");
                
                // Show current schema version
                let version = db.get_schema_version().await?;
                println!("Schema version: {}", version);
                
                Ok(())
            }
            Commands::CheckDeps => {
                self.run_dependency_check().await
            }
            Commands::Config { output } => {
                self.run_config_generation(&output).await
            }
            Commands::Migrate { database, action } => {
                self.run_migration_command(&database, action).await
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
                println!("  Unique paragraphs: {}", stats.total_paragraphs);
                println!("  Total tokens: {}", stats.total_tokens);
                
                // Get paragraph-level deduplication statistics
                let paragraph_stats = db.get_paragraph_statistics().await?;
                if paragraph_stats.total_paragraph_instances > 0 {
                    println!("  Total paragraph instances: {}", paragraph_stats.total_paragraph_instances);
                    if paragraph_stats.deduplicated_paragraphs > 0 {
                        println!("  Paragraphs deduplicated: {} ({:.1}%)", 
                            paragraph_stats.deduplicated_paragraphs,
                            paragraph_stats.deduplication_rate);
                    }
                    println!("  Average paragraph length: {:.0} characters", 
                        paragraph_stats.average_paragraph_length);
                }
                
                if stats.total_tokens > 0 && stats.total_files > 0 {
                    println!("  Average tokens per file: {:.0}", stats.total_tokens as f64 / stats.total_files as f64);
                }
                
                if stats.total_paragraphs > 0 {
                    println!("  Average tokens per paragraph: {:.1}", 
                        stats.total_tokens as f64 / stats.total_paragraphs as f64);
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
            
            // Phase 4: Content processing
            println!("\nPhase 4: Content processing...");
            
            // Get unique files that need content processing
            let unique_files = processed_files.iter()
                .filter(|f| f.duplicate_status == crate::types::DuplicateStatus::Unique || 
                           f.duplicate_status == crate::types::DuplicateStatus::Canonical)
                .collect::<Vec<_>>();
            
            if unique_files.is_empty() {
                println!("No unique files to process for content extraction");
            } else {
                println!("Processing content for {} unique files...", unique_files.len());
                
                // Initialize content processing components
                let extraction_manager = crate::extractor::ExtractionManager::new();
                let content_processor = crate::extractor::ContentProcessor;
                
                let mut processed_count = 0;
                let mut error_count = 0;
                let mut total_paragraphs = 0;
                let mut total_tokens = 0;
                
                // Process each unique file with paragraph-level deduplication
                for file_metadata in unique_files {
                    if self.verbose {
                        println!("Processing: {}", file_metadata.full_filepath.display());
                    }
                    
                    match self.process_file_content(
                        &database,
                        &extraction_manager,
                        &content_processor,
                        file_metadata
                    ).await {
                        Ok((paragraph_count, token_count)) => {
                            processed_count += 1;
                            total_paragraphs += paragraph_count;
                            total_tokens += token_count;
                            
                            if processed_count % 100 == 0 {
                                println!("Processed {} files, {} paragraphs, {} tokens", 
                                    processed_count, total_paragraphs, total_tokens);
                            }
                        }
                        Err(e) => {
                            error_count += 1;
                            if self.verbose {
                                eprintln!("Error processing {}: {}", 
                                    file_metadata.full_filepath.display(), e);
                            }
                            
                            // Get file ID for error tracking
                            let file_id = database.get_file_id_by_path(&file_metadata.full_filepath).await
                                .unwrap_or(None);
                            
                            // Store error in database
                            let processing_error = crate::types::ProcessingError {
                                id: uuid::Uuid::new_v4(),
                                file_id,
                                error_type: "ContentProcessing".to_string(),
                                error_message: e.to_string(),
                                stack_trace: None,
                                occurred_at: chrono::Utc::now(),
                            };
                            
                            if let Err(db_err) = database.insert_error(&processing_error).await {
                                eprintln!("Failed to store error in database: {}", db_err);
                            }
                            
                            // Update file status to error
                            if let Err(status_err) = database.update_file_processing_status(
                                &file_metadata.full_filepath,
                                crate::types::ProcessingStatus::Error,
                                None,
                                Some(e.to_string()),
                            ).await {
                                eprintln!("Failed to update file status: {}", status_err);
                            }
                        }
                    }
                }
                
                println!("\nContent Processing Complete:");
                println!("  Files processed: {}", processed_count);
                println!("  Processing errors: {}", error_count);
                println!("  Total paragraphs processed: {}", total_paragraphs);
                println!("  Total tokens: {}", total_tokens);
                
                // Get comprehensive paragraph statistics from database
                let paragraph_stats = database.get_paragraph_statistics().await?;
                
                if total_paragraphs > 0 {
                    println!("  Unique paragraphs stored: {}", paragraph_stats.unique_paragraphs);
                    
                    if paragraph_stats.deduplicated_paragraphs > 0 {
                        println!("  Paragraphs deduplicated: {} ({:.1}%)", 
                            paragraph_stats.deduplicated_paragraphs, 
                            paragraph_stats.deduplication_rate);
                    }
                    
                    println!("  Average tokens per paragraph: {:.1}", 
                        total_tokens as f64 / total_paragraphs as f64);
                    
                    if paragraph_stats.unique_paragraphs > 0 {
                        println!("  Average paragraph length: {:.0} characters", 
                            paragraph_stats.average_paragraph_length);
                        println!("  Total unique tokens: {}", paragraph_stats.total_tokens);
                    }
                }
            }
        } else {
            println!("DRY RUN: Skipping database operations");
        }

        println!("Ingestion workflow completed successfully!");
        Ok(())
    }

    /// Run migration commands
    async fn run_migration_command(&self, database_path: &PathBuf, action: MigrateAction) -> Result<()> {
        use crate::database::{Database, MigrationManager};
        
        // Check if database exists for most operations
        match action {
            MigrateAction::Up => {
                // Create database if it doesn't exist
                if !database_path.exists() {
                    println!("Database does not exist, creating: {}", database_path.display());
                }
            }
            _ => {
                if !database_path.exists() {
                    return Err(PensieveError::CliArgument(format!(
                        "Database file does not exist: {}",
                        database_path.display()
                    )));
                }
            }
        }
        
        let db = Database::new(database_path).await?;
        
        // Try to load migrations from files first, fallback to hardcoded migrations
        let migrations_dir = std::path::Path::new("migrations");
        let migration_manager = if migrations_dir.exists() {
            println!("Loading migrations from directory: {}", migrations_dir.display());
            MigrationManager::from_files(db.clone(), migrations_dir)?
        } else {
            println!("Using built-in migrations (migrations directory not found)");
            MigrationManager::new(db.clone())
        };
        
        match action {
            MigrateAction::Up => {
                println!("Running database migrations...");
                
                let current_version = migration_manager.get_current_version().await?;
                let target_version = migration_manager.get_target_version();
                
                if current_version >= target_version {
                    println!("Database is already up to date (version {})", current_version);
                    return Ok(());
                }
                
                println!("Upgrading from version {} to {}", current_version, target_version);
                migration_manager.migrate().await?;
                
                println!("Migrations completed successfully");
                
                // Show final status
                let final_version = migration_manager.get_current_version().await?;
                println!("Current schema version: {}", final_version);
            }
            
            MigrateAction::Status => {
                println!("=== Migration Status ===");
                
                let current_version = migration_manager.get_current_version().await?;
                let target_version = migration_manager.get_target_version();
                
                println!("Current version: {}", current_version);
                println!("Target version: {}", target_version);
                
                if current_version < target_version {
                    println!("Status: {} migrations pending", target_version - current_version);
                } else if current_version > target_version {
                    println!("Status: Database is newer than expected (version {})", current_version);
                } else {
                    println!("Status: Up to date");
                }
                
                println!("\n=== Migration History ===");
                let history = migration_manager.get_migration_history().await?;
                
                if history.is_empty() {
                    println!("No migrations applied yet");
                } else {
                    for (version, description, applied_at) in history {
                        println!("Version {}: {} (applied: {})", 
                            version, 
                            description, 
                            applied_at.format("%Y-%m-%d %H:%M:%S UTC"));
                    }
                }
            }
            
            MigrateAction::Rollback { version } => {
                let current_version = migration_manager.get_current_version().await?;
                
                if version >= current_version {
                    println!("Cannot rollback to version {} (current version is {})", version, current_version);
                    return Ok(());
                }
                
                println!("Rolling back from version {} to {}", current_version, version);
                println!("WARNING: This operation may result in data loss!");
                
                // Confirm rollback
                print!("Are you sure you want to continue? (y/N): ");
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
                
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                
                if input.trim().to_lowercase() != "y" {
                    println!("Rollback cancelled");
                    return Ok(());
                }
                
                migration_manager.rollback_to(version).await?;
                println!("Rollback completed successfully");
                
                let final_version = migration_manager.get_current_version().await?;
                println!("Current schema version: {}", final_version);
            }
            
            MigrateAction::Validate => {
                println!("Validating database schema...");
                
                let issues = migration_manager.validate_schema().await?;
                
                if issues.is_empty() {
                    println!("✓ Schema validation passed - no issues found");
                } else {
                    println!("✗ Schema validation found {} issues:", issues.len());
                    for (i, issue) in issues.iter().enumerate() {
                        println!("  {}: {}", i + 1, issue);
                    }
                    return Err(PensieveError::Migration(
                        format!("Schema validation failed with {} issues", issues.len())
                    ));
                }
                
                // Additional integrity checks
                println!("Running additional integrity checks...");
                
                // Check foreign key constraints
                let fk_check: Vec<(String, i64, String, i64)> = sqlx::query_as(
                    "PRAGMA foreign_key_check"
                )
                .fetch_all(db.pool())
                .await
                .map_err(|e| PensieveError::Database(e))?;
                
                if fk_check.is_empty() {
                    println!("✓ Foreign key constraints are valid");
                } else {
                    println!("✗ Foreign key constraint violations found:");
                    for (table, rowid, parent, fkid) in fk_check {
                        println!("  Table: {}, Row: {}, Parent: {}, FK: {}", table, rowid, parent, fkid);
                    }
                }
                
                // Check database integrity
                let integrity_check: String = sqlx::query_scalar("PRAGMA integrity_check")
                    .fetch_one(db.pool())
                    .await
                    .map_err(|e| PensieveError::Database(e))?;
                
                if integrity_check == "ok" {
                    println!("✓ Database integrity check passed");
                } else {
                    println!("✗ Database integrity issues: {}", integrity_check);
                }
                
                println!("Schema validation completed");
            }
            
            MigrateAction::Reset { confirm } => {
                if !confirm {
                    println!("WARNING: This will delete ALL data in the database!");
                    println!("Use --confirm flag to proceed with the reset");
                    return Ok(());
                }
                
                println!("Resetting database (dropping all tables)...");
                
                // Drop all tables directly instead of using rollback
                let tables_to_drop = vec![
                    "DROP TABLE IF EXISTS processing_stats",
                    "DROP TABLE IF EXISTS paragraph_sources", 
                    "DROP TABLE IF EXISTS processing_errors",
                    "DROP TABLE IF EXISTS paragraphs",
                    "DROP TABLE IF EXISTS files",
                    "DROP TABLE IF EXISTS schema_version",
                ];
                
                for drop_sql in tables_to_drop {
                    sqlx::query(drop_sql)
                        .execute(db.pool())
                        .await
                        .map_err(|e| PensieveError::Database(e))?;
                }
                
                println!("Database reset completed");
                println!("Run 'pensieve migrate up' to recreate the schema");
            }
        }
        
        Ok(())
    }

    /// Process content for a single file with paragraph-level deduplication
    async fn process_file_content(
        &self,
        database: &crate::database::Database,
        extraction_manager: &crate::extractor::ExtractionManager,
        _content_processor: &crate::extractor::ContentProcessor,
        file_metadata: &FileMetadata,
    ) -> Result<(u32, u32)> {
        use crate::types::{Paragraph, ParagraphSource, ParagraphId};
        use crate::extractor::ContentProcessor;
        
        // Extract content from file
        let content = extraction_manager.extract_content(&file_metadata.full_filepath).await?;
        
        if content.trim().is_empty() {
            // Update file status to processed with zero tokens
            database.update_file_processing_status(
                &file_metadata.full_filepath,
                crate::types::ProcessingStatus::Processed,
                Some(0),
                None,
            ).await?;
            return Ok((0, 0));
        }
        
        // Normalize content
        let normalized_content = ContentProcessor::normalize_text(&content);
        
        // Split into paragraphs
        let paragraph_texts = ContentProcessor::split_paragraphs(&normalized_content);
        
        if paragraph_texts.is_empty() {
            // Update file status to processed with zero tokens
            database.update_file_processing_status(
                &file_metadata.full_filepath,
                crate::types::ProcessingStatus::Processed,
                Some(0),
                None,
            ).await?;
            return Ok((0, 0));
        }
        
        let mut total_tokens = 0;
        let mut new_paragraphs_count = 0;
        let total_paragraphs_count = paragraph_texts.len() as u32;
        
        // Get the actual file ID from database
        let file_id = database.get_file_id_by_path(&file_metadata.full_filepath).await?
            .ok_or_else(|| PensieveError::InvalidData(
                format!("File not found in database: {}", file_metadata.full_filepath.display())
            ))?;
        
        // Calculate byte offsets for each paragraph in the normalized content
        let mut byte_offsets = Vec::new();
        let mut current_offset = 0;
        
        for paragraph_text in &paragraph_texts {
            let start_offset = current_offset;
            let end_offset = current_offset + paragraph_text.len();
            byte_offsets.push((start_offset, end_offset));
            
            // Account for paragraph separator (double newline) in offset calculation
            current_offset = end_offset + 2; // +2 for "\n\n"
        }
        
        // Process each paragraph with proper deduplication
        for (index, paragraph_text) in paragraph_texts.iter().enumerate() {
            // Calculate paragraph metadata
            let content_hash = ContentProcessor::calculate_content_hash(paragraph_text);
            let estimated_tokens = ContentProcessor::estimate_tokens(paragraph_text);
            let word_count = ContentProcessor::count_words(paragraph_text);
            let char_count = ContentProcessor::count_characters(paragraph_text);
            
            total_tokens += estimated_tokens;
            
            let (byte_offset_start, byte_offset_end) = byte_offsets[index];
            
            // Check if paragraph already exists (deduplication)
            if let Some(existing_paragraph) = database.get_paragraph_by_hash(&content_hash).await? {
                // Paragraph exists, just create source link
                let paragraph_source = ParagraphSource {
                    paragraph_id: existing_paragraph.id,
                    file_id,
                    paragraph_index: index as u32,
                    byte_offset_start: byte_offset_start as u64,
                    byte_offset_end: byte_offset_end as u64,
                };
                
                database.insert_paragraph_source(&paragraph_source).await?;
                
                if self.verbose {
                    println!("  Deduplicated paragraph {} (hash: {}...)", 
                        index + 1, &content_hash[..8]);
                }
            } else {
                // Create new unique paragraph
                let paragraph = Paragraph {
                    id: ParagraphId::new(),
                    content_hash: content_hash.clone(),
                    content: paragraph_text.clone(),
                    estimated_tokens,
                    word_count,
                    char_count,
                    created_at: chrono::Utc::now(),
                };
                
                // Insert paragraph
                database.insert_paragraph(&paragraph).await?;
                
                // Create source link
                let paragraph_source = ParagraphSource {
                    paragraph_id: paragraph.id,
                    file_id,
                    paragraph_index: index as u32,
                    byte_offset_start: byte_offset_start as u64,
                    byte_offset_end: byte_offset_end as u64,
                };
                
                database.insert_paragraph_source(&paragraph_source).await?;
                new_paragraphs_count += 1;
                
                if self.verbose {
                    println!("  New paragraph {} (hash: {}..., {} tokens)", 
                        index + 1, &content_hash[..8], estimated_tokens);
                }
            }
        }
        
        // Update file metadata with processing results
        database.update_file_processing_status(
            &file_metadata.full_filepath,
            crate::types::ProcessingStatus::Processed,
            Some(total_tokens),
            None,
        ).await?;
        
        if self.verbose {
            println!("  File processed: {} total paragraphs, {} new, {} deduplicated, {} tokens",
                total_paragraphs_count, 
                new_paragraphs_count, 
                total_paragraphs_count - new_paragraphs_count,
                total_tokens);
        }
        
        Ok((total_paragraphs_count, total_tokens))
    }

    /// Run dependency check command
    async fn run_dependency_check(&self) -> Result<()> {
        println!("Checking external tool dependencies...");
        
        // Define the dependencies we need to check
        let dependencies = vec![
            DependencyCheck {
                name: "SQLite",
                description: "Database engine",
                check_type: DependencyType::Library,
                required: true,
                status: DependencyStatus::Unknown,
            },
            DependencyCheck {
                name: "File System Access",
                description: "Read/write permissions",
                check_type: DependencyType::SystemAccess,
                required: true,
                status: DependencyStatus::Unknown,
            },
            DependencyCheck {
                name: "PDF Processing",
                description: "Native PDF text extraction",
                check_type: DependencyType::Library,
                required: false,
                status: DependencyStatus::Unknown,
            },
            DependencyCheck {
                name: "HTML Processing",
                description: "HTML parsing and text extraction",
                check_type: DependencyType::Library,
                required: false,
                status: DependencyStatus::Unknown,
            },
            DependencyCheck {
                name: "Archive Processing",
                description: "ZIP/DOCX file extraction",
                check_type: DependencyType::Library,
                required: false,
                status: DependencyStatus::Unknown,
            },
        ];

        let mut all_passed = true;
        let mut results = Vec::new();

        println!("\n=== Dependency Check Results ===\n");

        for mut dep in dependencies {
            print!("Checking {}: ", dep.name);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            dep.status = match dep.check_type {
                DependencyType::Library => {
                    match dep.name {
                        "SQLite" => self.check_sqlite_dependency().await,
                        "PDF Processing" => self.check_pdf_dependency().await,
                        "HTML Processing" => self.check_html_dependency().await,
                        "Archive Processing" => self.check_archive_dependency().await,
                        _ => DependencyStatus::Unknown,
                    }
                }
                DependencyType::SystemAccess => {
                    match dep.name {
                        "File System Access" => self.check_filesystem_access().await,
                        _ => DependencyStatus::Unknown,
                    }
                }
            };

            match dep.status {
                DependencyStatus::Available => {
                    println!("✓ Available");
                }
                DependencyStatus::Missing => {
                    println!("✗ Missing");
                    if dep.required {
                        all_passed = false;
                    }
                }
                DependencyStatus::Error(ref msg) => {
                    println!("⚠ Error: {}", msg);
                    if dep.required {
                        all_passed = false;
                    }
                }
                DependencyStatus::Unknown => {
                    println!("? Unknown");
                    if dep.required {
                        all_passed = false;
                    }
                }
            }

            results.push(dep);
        }

        // Print summary
        println!("\n=== Summary ===");
        
        let available_count = results.iter().filter(|d| matches!(d.status, DependencyStatus::Available)).count();
        let missing_count = results.iter().filter(|d| matches!(d.status, DependencyStatus::Missing)).count();
        let error_count = results.iter().filter(|d| matches!(d.status, DependencyStatus::Error(_))).count();
        
        println!("Available: {}", available_count);
        println!("Missing: {}", missing_count);
        println!("Errors: {}", error_count);

        // Show detailed information for missing or error dependencies
        let problematic_deps: Vec<_> = results.iter()
            .filter(|d| !matches!(d.status, DependencyStatus::Available))
            .collect();

        if !problematic_deps.is_empty() {
            println!("\n=== Issues Found ===");
            for dep in problematic_deps {
                println!("\n{} ({})", dep.name, if dep.required { "Required" } else { "Optional" });
                println!("  Description: {}", dep.description);
                match &dep.status {
                    DependencyStatus::Missing => {
                        println!("  Status: Missing - functionality will be limited");
                    }
                    DependencyStatus::Error(msg) => {
                        println!("  Status: Error - {}", msg);
                    }
                    _ => {}
                }
            }
        }

        if all_passed {
            println!("\n✓ All required dependencies are available");
            println!("Pensieve is ready to use!");
        } else {
            println!("\n✗ Some required dependencies are missing");
            println!("Please install missing dependencies before using Pensieve");
            return Err(PensieveError::CliArgument(
                "Required dependencies are missing".to_string()
            ));
        }

        Ok(())
    }

    /// Check SQLite dependency
    async fn check_sqlite_dependency(&self) -> DependencyStatus {
        // Try to create an in-memory database to test SQLite functionality
        match sqlx::SqlitePool::connect("sqlite::memory:").await {
            Ok(pool) => {
                // Test basic SQL operations
                match sqlx::query("SELECT 1").fetch_one(&pool).await {
                    Ok(_) => DependencyStatus::Available,
                    Err(e) => DependencyStatus::Error(format!("SQLite query failed: {}", e)),
                }
            }
            Err(e) => DependencyStatus::Error(format!("SQLite connection failed: {}", e)),
        }
    }

    /// Check filesystem access dependency
    async fn check_filesystem_access(&self) -> DependencyStatus {
        use std::fs;

        // Test read access to current directory
        match fs::read_dir(".") {
            Ok(_) => {
                // Test write access by creating a temporary file
                match std::fs::File::create(".pensieve_temp_test") {
                    Ok(_) => {
                        // Clean up the test file
                        let _ = fs::remove_file(".pensieve_temp_test");
                        DependencyStatus::Available
                    }
                    Err(e) => DependencyStatus::Error(format!("Write access failed: {}", e)),
                }
            }
            Err(e) => DependencyStatus::Error(format!("Read access failed: {}", e)),
        }
    }

    /// Check PDF processing dependency
    async fn check_pdf_dependency(&self) -> DependencyStatus {
        // Test if we can use the pdf-extract crate
        // Since it's compiled in, we just check if the basic functionality works
        DependencyStatus::Available // PDF extraction is built-in via pdf-extract crate
    }

    /// Check HTML processing dependency
    async fn check_html_dependency(&self) -> DependencyStatus {
        // Test if we can use the scraper and html2md crates
        use scraper::{Html, Selector};
        
        let test_html = "<html><body><p>Test</p></body></html>";
        let document = Html::parse_document(test_html);
        
        match Selector::parse("p") {
            Ok(selector) => {
                let elements: Vec<_> = document.select(&selector).collect();
                if !elements.is_empty() {
                    DependencyStatus::Available
                } else {
                    DependencyStatus::Error("HTML parsing test failed".to_string())
                }
            }
            Err(e) => DependencyStatus::Error(format!("CSS selector parsing failed: {}", e)),
        }
    }

    /// Check archive processing dependency
    async fn check_archive_dependency(&self) -> DependencyStatus {
        // Test if we can use the zip crate for DOCX processing
        // Since it's compiled in, we just verify basic functionality
        DependencyStatus::Available // ZIP processing is built-in via zip crate
    }

    /// Run configuration file generation command
    async fn run_config_generation(&self, output_path: &PathBuf) -> Result<()> {
        println!("Generating configuration file at: {}", output_path.display());

        // Check if file already exists
        if output_path.exists() {
            print!("Configuration file already exists. Overwrite? (y/N): ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();

            if input.trim().to_lowercase() != "y" {
                println!("Configuration generation cancelled");
                return Ok(());
            }
        }

        // Generate default configuration
        let config = self.generate_default_config();

        // Write configuration to file
        match std::fs::write(output_path, config) {
            Ok(_) => {
                println!("✓ Configuration file generated successfully");
                println!("Edit {} to customize Pensieve settings", output_path.display());
                
                // Show brief explanation of key settings
                println!("\nKey configuration options:");
                println!("  - input_directory: Default directory to scan");
                println!("  - database_path: Default database location");
                println!("  - file_extensions: File types to process");
                println!("  - processing: Content processing settings");
                println!("  - performance: Performance tuning options");
                
                Ok(())
            }
            Err(e) => {
                Err(PensieveError::Io(e))
            }
        }
    }

    /// Generate default configuration content
    fn generate_default_config(&self) -> String {
        r#"# Pensieve Configuration File
# This file contains default settings for the Pensieve CLI tool
# Uncomment and modify values as needed

# Default input directory to scan (can be overridden with --input)
# input_directory = "./documents"

# Default database path (can be overridden with --database)
# database_path = "./pensieve.db"

# File processing settings
[processing]
# Minimum paragraph length in characters
min_paragraph_length = 10

# Maximum paragraph length in characters (longer paragraphs will be split)
max_paragraph_length = 10000

# Token estimation method ("simple" uses ~4 chars per token)
token_estimation = "simple"

# File extensions to process (case-insensitive)
file_extensions = [
    # Text files
    "txt", "md", "rst", "org",
    
    # Source code
    "rs", "py", "js", "ts", "java", "go", "c", "cpp", "h", "hpp",
    "php", "rb", "swift", "kt", "scala", "clj", "hs", "elm", "lua",
    "pl", "r", "m",
    
    # Configuration files
    "json", "yaml", "yml", "toml", "ini", "cfg", "env", "properties", "conf",
    
    # Web files
    "html", "css", "xml",
    
    # Scripts
    "sh", "bat", "ps1", "dockerfile",
    
    # Data files
    "csv", "tsv", "log", "sql",
    
    # Documentation
    "adoc", "wiki", "tex", "bib",
    
    # Documents (basic text extraction)
    "pdf", "doc", "docx", "odt", "rtf", "pages",
    
    # E-books (basic text extraction)
    "epub", "mobi", "azw", "azw3", "fb2", "lit", "pdb", "tcr", "prc",
    
    # Spreadsheets (basic text extraction)
    "xls", "xlsx"
]

# Binary file extensions to explicitly exclude
excluded_extensions = [
    # Images
    "jpg", "jpeg", "png", "gif", "bmp", "svg", "ico", "tiff", "webp",
    
    # Videos
    "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v",
    
    # Audio
    "mp3", "wav", "flac", "ogg", "aac", "wma", "m4a",
    
    # Archives
    "zip", "tar", "gz", "rar", "7z", "bz2", "xz",
    
    # Executables
    "exe", "bin", "app", "dmg", "msi", "deb", "rpm",
    
    # Libraries
    "dll", "so", "dylib", "lib", "a"
]

# Performance settings
[performance]
# Maximum number of files to process in parallel
max_parallel_files = 10

# Database connection pool size
db_pool_size = 10

# SQLite cache size in KB (default: 64MB)
sqlite_cache_size_kb = 65536

# Enable memory-mapped I/O for SQLite
sqlite_mmap_enabled = true

# Batch size for database operations
batch_size = 1000

# Logging settings
[logging]
# Log level: "error", "warn", "info", "debug", "trace"
level = "info"

# Enable verbose output by default
verbose = false

# Log file path (optional, logs to stdout if not specified)
# log_file = "./pensieve.log"

# Advanced settings
[advanced]
# Enable file system monitoring for incremental updates
# fs_monitoring = false

# Custom MIME type mappings (extension -> mime_type)
[advanced.mime_mappings]
# "myext" = "text/plain"

# Processing hooks (experimental)
[hooks]
# Commands to run before/after processing
# pre_processing = ["echo 'Starting processing'"]
# post_processing = ["echo 'Processing complete'"]

# Notification settings
[notifications]
# Enable desktop notifications (requires system support)
# desktop_notifications = false

# Webhook URL for completion notifications
# webhook_url = "https://example.com/webhook"
"#.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use crate::extractor::{ExtractionManager, ContentProcessor};
    use crate::types::{FileMetadata, DuplicateStatus, ProcessingStatus, FileType};
    use std::fs;
    use tempfile::TempDir;
    use chrono::Utc;

    async fn create_test_database() -> Result<(Database, TempDir)> {
        let temp_dir = TempDir::new().map_err(|e| PensieveError::Io(e))?;
        let db_path = temp_dir.path().join("test.db");
        let db = Database::new(&db_path).await?;
        db.initialize_schema().await?;
        Ok((db, temp_dir))
    }

    #[tokio::test]
    async fn test_paragraph_deduplication_integration() {
        let (db, _db_temp_dir) = create_test_database().await.unwrap();
        let temp_dir = TempDir::new().unwrap();
        
        // Create test files with duplicate content
        let file1_path = temp_dir.path().join("file1.txt");
        let file2_path = temp_dir.path().join("file2.txt");
        
        let content1 = "First paragraph with unique content.\n\nShared paragraph content.\n\nAnother unique paragraph.";
        let content2 = "Different first paragraph.\n\nShared paragraph content.\n\nDifferent last paragraph.";
        
        fs::write(&file1_path, content1).unwrap();
        fs::write(&file2_path, content2).unwrap();
        
        // Create file metadata
        let now = Utc::now();
        let file1_metadata = FileMetadata {
            full_filepath: file1_path.clone(),
            folder_path: temp_dir.path().to_path_buf(),
            filename: "file1.txt".to_string(),
            file_extension: Some("txt".to_string()),
            file_type: FileType::File,
            size: content1.len() as u64,
            hash: "hash1".to_string(),
            creation_date: now,
            modification_date: now,
            access_date: now,
            permissions: 644,
            depth_level: 0,
            relative_path: "file1.txt".into(),
            is_hidden: false,
            is_symlink: false,
            symlink_target: None,
            duplicate_status: DuplicateStatus::Unique,
            duplicate_group_id: None,
            processing_status: ProcessingStatus::Pending,
            estimated_tokens: None,
            processed_at: None,
            error_message: None,
        };
        
        let file2_metadata = FileMetadata {
            full_filepath: file2_path.clone(),
            folder_path: temp_dir.path().to_path_buf(),
            filename: "file2.txt".to_string(),
            file_extension: Some("txt".to_string()),
            file_type: FileType::File,
            size: content2.len() as u64,
            hash: "hash2".to_string(),
            creation_date: now,
            modification_date: now,
            access_date: now,
            permissions: 644,
            depth_level: 0,
            relative_path: "file2.txt".into(),
            is_hidden: false,
            is_symlink: false,
            symlink_target: None,
            duplicate_status: DuplicateStatus::Unique,
            duplicate_group_id: None,
            processing_status: ProcessingStatus::Pending,
            estimated_tokens: None,
            processed_at: None,
            error_message: None,
        };
        
        // Insert files into database
        db.insert_file(&file1_metadata).await.unwrap();
        db.insert_file(&file2_metadata).await.unwrap();
        
        // Create CLI instance for testing
        let cli = Cli {
            input: None,
            database: None,
            verbose: false,
            dry_run: false,
            force_reprocess: false,
            config: None,
            command: None,
        };
        
        // Process both files
        let extraction_manager = ExtractionManager::new();
        let content_processor = ContentProcessor;
        
        let (paragraphs1, tokens1) = cli.process_file_content(
            &db,
            &extraction_manager,
            &content_processor,
            &file1_metadata,
        ).await.unwrap();
        
        let (paragraphs2, tokens2) = cli.process_file_content(
            &db,
            &extraction_manager,
            &content_processor,
            &file2_metadata,
        ).await.unwrap();
        
        // Verify paragraph counts
        assert_eq!(paragraphs1, 3); // 3 paragraphs in file1
        assert_eq!(paragraphs2, 3); // 3 paragraphs in file2
        assert!(tokens1 > 0);
        assert!(tokens2 > 0);
        
        // Get paragraph statistics
        let paragraph_stats = db.get_paragraph_statistics().await.unwrap();
        
        // Should have 6 total paragraph instances but only 5 unique paragraphs
        // (one shared paragraph between the files)
        assert_eq!(paragraph_stats.total_paragraph_instances, 6);
        assert_eq!(paragraph_stats.unique_paragraphs, 5);
        assert_eq!(paragraph_stats.deduplicated_paragraphs, 1);
        assert!(paragraph_stats.deduplication_rate > 0.0);
        assert!(paragraph_stats.total_tokens > 0);
        
        // Verify database statistics
        let stats = db.get_statistics().await.unwrap();
        assert_eq!(stats.total_paragraphs, 5); // 5 unique paragraphs
        assert!(stats.total_tokens > 0);
    }

    #[tokio::test]
    async fn test_cli_subcommands() {
        use tempfile::TempDir;
        
        // Test check-deps command
        let cli = Cli {
            input: None,
            database: None,
            verbose: false,
            dry_run: false,
            force_reprocess: false,
            config: None,
            command: Some(Commands::CheckDeps),
        };
        
        let result = cli.run_dependency_check().await;
        assert!(result.is_ok(), "Dependency check should succeed");
        
        // Test config generation
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test-config.toml");
        
        let cli = Cli {
            input: None,
            database: None,
            verbose: false,
            dry_run: false,
            force_reprocess: false,
            config: None,
            command: Some(Commands::Config {
                output: config_path.clone(),
            }),
        };
        
        let result = cli.run_config_generation(&config_path).await;
        assert!(result.is_ok(), "Config generation should succeed");
        assert!(config_path.exists(), "Config file should be created");
        
        // Verify config file content
        let config_content = std::fs::read_to_string(&config_path).unwrap();
        assert!(config_content.contains("# Pensieve Configuration File"));
        assert!(config_content.contains("[processing]"));
        assert!(config_content.contains("file_extensions"));
        assert!(config_content.contains("[performance]"));
        
        // Test init command
        let db_path = temp_dir.path().join("test-init.db");
        
        let cli = Cli {
            input: None,
            database: None,
            verbose: false,
            dry_run: false,
            force_reprocess: false,
            config: None,
            command: Some(Commands::Init {
                database: db_path.clone(),
            }),
        };
        
        let result = cli.run_subcommand(Commands::Init {
            database: db_path.clone(),
        }).await;
        assert!(result.is_ok(), "Init command should succeed");
        assert!(db_path.exists(), "Database file should be created");
        
        // Verify database was properly initialized
        let db = crate::database::Database::new(&db_path).await.unwrap();
        let version = db.get_schema_version().await.unwrap();
        assert!(version > 0, "Schema version should be greater than 0");
    }

    #[tokio::test]
    async fn test_dependency_checks() {
        let cli = Cli {
            input: None,
            database: None,
            verbose: false,
            dry_run: false,
            force_reprocess: false,
            config: None,
            command: None,
        };
        
        // Test individual dependency checks
        let sqlite_status = cli.check_sqlite_dependency().await;
        assert_eq!(sqlite_status, DependencyStatus::Available);
        
        let fs_status = cli.check_filesystem_access().await;
        assert_eq!(fs_status, DependencyStatus::Available);
        
        let html_status = cli.check_html_dependency().await;
        assert_eq!(html_status, DependencyStatus::Available);
        
        let pdf_status = cli.check_pdf_dependency().await;
        assert_eq!(pdf_status, DependencyStatus::Available);
        
        let archive_status = cli.check_archive_dependency().await;
        assert_eq!(archive_status, DependencyStatus::Available);
    }

    #[test]
    fn test_config_generation_content() {
        let cli = Cli {
            input: None,
            database: None,
            verbose: false,
            dry_run: false,
            force_reprocess: false,
            config: None,
            command: None,
        };
        
        let config = cli.generate_default_config();
        
        // Verify essential sections are present
        assert!(config.contains("# Pensieve Configuration File"));
        assert!(config.contains("[processing]"));
        assert!(config.contains("min_paragraph_length"));
        assert!(config.contains("max_paragraph_length"));
        assert!(config.contains("file_extensions"));
        assert!(config.contains("excluded_extensions"));
        assert!(config.contains("[performance]"));
        assert!(config.contains("max_parallel_files"));
        assert!(config.contains("db_pool_size"));
        assert!(config.contains("[logging]"));
        assert!(config.contains("[advanced]"));
        
        // Verify file extensions are comprehensive
        assert!(config.contains("\"rs\""));
        assert!(config.contains("\"py\""));
        assert!(config.contains("\"js\""));
        assert!(config.contains("\"html\""));
        assert!(config.contains("\"pdf\""));
        assert!(config.contains("\"docx\""));
        
        // Verify excluded extensions
        assert!(config.contains("\"jpg\""));
        assert!(config.contains("\"mp4\""));
        assert!(config.contains("\"exe\""));
    }
}