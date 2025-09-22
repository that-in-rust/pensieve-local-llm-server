//! Command-line interface for the Pensieve tool

use crate::prelude::*;
use crate::types::FileMetadata;
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

    async fn create_test_database() -> Result<Database> {
        let temp_dir = TempDir::new().map_err(|e| PensieveError::Io(e))?;
        let db_path = temp_dir.path().join("test.db");
        let db = Database::new(&db_path).await?;
        db.initialize_schema().await?;
        Ok(db)
    }

    #[tokio::test]
    async fn test_paragraph_deduplication_integration() {
        let db = create_test_database().await.unwrap();
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
}