use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use crate::help::HelpSystem;

use crate::config::{Config, ConfigManager, merge_config_with_cli};
use crate::logging::MonitoringContext;
use colored::*;
use tracing::{info, error, instrument};

#[derive(Parser)]
#[command(name = "code-ingest")]
#[command(about = "A high-performance Rust tool for ingesting GitHub repositories and local folders into PostgreSQL databases for systematic code analysis")]
#[command(version = "0.1.0")]
#[command(long_about = "Code Ingest is a powerful command-line tool that enables systematic analysis of codebases by ingesting GitHub repositories or local folders into PostgreSQL databases. It supports hierarchical task generation, chunked analysis for large files, and provides comprehensive SQL interfaces for code exploration.

EXAMPLES:
    # Ingest GitHub repository
    code-ingest ingest https://github.com/user/repo --db-path /path/to/db
    
    # Ingest local folder
    code-ingest ingest /absolute/path/to/folder --folder-flag --db-path /path/to/db
    
    # Generate content files and task lists (simplified approach)
    code-ingest chunk-level-task-generator TABLE_NAME --db-path /path/to/db
    
    # Generate with chunking for large files
    code-ingest chunk-level-task-generator TABLE_NAME 500 --db-path /path/to/db
    
    # Generate hierarchical tasks (legacy)
    code-ingest generate-hierarchical-tasks TABLE_NAME --levels 4 --groups 7 --output tasks.md")]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Database path for PostgreSQL connection
    #[arg(long, global = true, help = "Path to PostgreSQL database directory")]
    db_path: Option<PathBuf>,
    
    /// Configuration file path (default: ~/.config/code-ingest/config.toml)
    #[arg(long, global = true, help = "Path to configuration file")]
    config: Option<PathBuf>,
    
    /// Log level (trace, debug, info, warn, error)
    #[arg(long, global = true, help = "Set logging level")]
    log_level: Option<String>,
    
    /// Enable JSON formatted logs
    #[arg(long, global = true, help = "Output logs in JSON format")]
    json_logs: bool,
    
    /// Enable verbose logging (equivalent to --log-level debug)
    #[arg(short, long, global = true, help = "Enable verbose logging")]
    verbose: bool,
    
    /// Disable progress reporting
    #[arg(long, global = true, help = "Disable progress reporting")]
    no_progress: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Ingest a GitHub repository or local folder into PostgreSQL database
    /// 
    /// Examples:
    ///   # Ingest GitHub repository
    ///   code-ingest ingest https://github.com/user/repo --db-path /path/to/db
    ///   
    ///   # Ingest local folder
    ///   code-ingest ingest /path/to/folder --folder-flag --db-path /path/to/db
    Ingest {
        /// GitHub repository URL (https://github.com/user/repo) or absolute local folder path
        #[arg(help = "GitHub repository URL or absolute path to local folder")]
        source: String,
        
        /// Enable local folder ingestion mode (required for local paths)
        #[arg(long, help = "Enable local folder ingestion mode for processing local directories")]
        folder_flag: bool,
        
        /// Database path (can also be set globally)
        #[arg(long, help = "Path to PostgreSQL database directory")]
        db_path: Option<PathBuf>,
        
        /// Enable automatic chunking for large files that exceed tsvector limits
        #[arg(long, help = "Automatically chunk large files to avoid PostgreSQL tsvector size limits")]
        auto_chunk: bool,
        
        /// Chunk size in MB for large files (default: 0.8MB)
        #[arg(long, default_value = "0.8", help = "Size of each chunk in MB (0.1-10.0)")]
        chunk_size: f64,
        
        /// Validate chunks after creation (slower but safer)
        #[arg(long, help = "Validate chunks against original file using checksums")]
        validate_chunks: bool,
        
        /// Clean up chunk files after processing
        #[arg(long, help = "Remove chunk files after successful processing")]
        cleanup_chunks: bool,
    },

    /// Execute SQL query and output results
    Sql {
        /// SQL query to execute
        query: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Maximum number of rows to return (0 = no limit)
        #[arg(long, default_value = "0")]
        limit: usize,
        /// Number of rows to skip (for pagination)
        #[arg(long, default_value = "0")]
        offset: usize,
        /// Query timeout in seconds (0 = no timeout)
        #[arg(long, default_value = "30")]
        timeout: u64,
        /// Format output for LLM consumption
        #[arg(long)]
        llm_format: bool,
    },

    /// Prepare query results for IDE task processing
    QueryPrepare {
        /// SQL query to execute
        query: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Temporary file path for query results
        #[arg(long)]
        temp_path: PathBuf,
        /// Tasks file path for IDE integration
        #[arg(long)]
        tasks_file: PathBuf,
        /// Output table name for storing analysis results
        #[arg(long)]
        output_table: String,
    },

    /// Generate structured task markdown for systematic analysis
    GenerateTasks {
        /// SQL query to get data for task generation
        #[arg(long)]
        sql: String,
        /// Prompt file for analysis instructions
        #[arg(long)]
        prompt_file: PathBuf,
        /// Output table name for storing results
        #[arg(long)]
        output_table: String,
        /// Tasks file path for generated markdown
        #[arg(long)]
        tasks_file: PathBuf,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Store analysis results back to database
    StoreResult {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Output table name for storing results
        #[arg(long)]
        output_table: String,
        /// Result file containing analysis output
        #[arg(long)]
        result_file: PathBuf,
        /// Original SQL query that generated the data
        #[arg(long)]
        original_query: String,
    },

    /// Export query results as individual markdown files
    PrintToMd {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to query
        #[arg(long)]
        table: String,
        /// SQL query to execute
        #[arg(long)]
        sql: String,
        /// Prefix for generated markdown files
        #[arg(long)]
        prefix: String,
        /// Location directory for markdown files
        #[arg(long)]
        location: PathBuf,
    },

    /// List all tables in the database
    ListTables {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Show sample data from a table
    Sample {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to sample
        #[arg(long)]
        table: String,
        /// Number of rows to show
        #[arg(long, default_value = "5")]
        limit: usize,
    },

    /// Show table schema information
    Describe {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to describe
        #[arg(long)]
        table: String,
    },

    /// Show database connection info
    DbInfo {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Show PostgreSQL setup instructions
    PgStart,

    /// Show comprehensive setup guide
    Setup,

    /// Show command examples for common tasks
    Examples,

    /// Show troubleshooting guide
    Troubleshoot,

    /// Clean up old ingestion tables
    CleanupTables {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Number of recent tables to keep
        #[arg(long, default_value = "5")]
        keep: usize,
        /// Confirm deletion without prompting
        #[arg(long)]
        confirm: bool,
    },

    /// Drop a specific table
    DropTable {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to drop
        #[arg(long)]
        table: String,
        /// Confirm deletion without prompting
        #[arg(long)]
        confirm: bool,
    },

    /// Get database management recommendations
    Recommendations {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Optimize database tables (VACUUM ANALYZE)
    OptimizeTables {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Specific tables to optimize (default: all tables)
        #[arg(long)]
        tables: Option<Vec<String>>,
    },

    /// Count rows in a database table
    CountRows {
        /// Table name to count rows from
        table_name: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Extract content from database table to A/B/C files
    ExtractContent {
        /// Table name to extract content from
        table_name: String,
        /// Output directory for content files
        #[arg(long, default_value = ".raw_data_202509")]
        output_dir: PathBuf,
        /// Chunk size for filename generation (e.g., 500 for 500-line chunks)
        #[arg(long)]
        chunk_size: Option<usize>,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Configuration management commands
    /// 
    /// Examples:
    ///   # Show current configuration
    ///   code-ingest config show
    ///   
    ///   # Create default configuration file
    ///   code-ingest config init
    ///   
    ///   # Set default database path
    ///   code-ingest config set database.default_path /path/to/db
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// [EXPERIMENTAL] Generate simple task lists from database table
    /// 
    /// Creates simple checkbox markdown files for systematic analysis.
    /// Focuses on proven patterns over complex hierarchical structures.
    /// 
    /// Examples:
    ///   # Simple task generation (limited to 50 tasks)
    ///   code-ingest generate-hierarchical-tasks INGEST_20250928101039 \
    ///     --output tasks.md --db-path /path/to/db
    ///   
    ///   # With chunking for large files
    ///   code-ingest generate-hierarchical-tasks INGEST_20250928101039 \
    ///     --chunks 500 --max-tasks 20 --output tasks.md \
    ///     --prompt-file .kiro/steering/analysis.md --db-path /path/to/db
    GenerateHierarchicalTasks {
        /// Database table name containing ingested code data
        #[arg(help = "Name of the database table to generate tasks from (e.g., INGEST_20250928101039)")]
        table_name: String,
        
        /// Number of hierarchy levels in the task structure (1-8)
        #[arg(long, default_value = "4", help = "Number of hierarchical levels (1-8, default: 4)")]
        levels: usize,
        
        /// Number of groups per hierarchy level (1-20)
        #[arg(long, default_value = "7", help = "Number of groups per level (1-20, default: 7)")]
        groups: usize,
        
        /// Output markdown file path for generated tasks
        #[arg(long, help = "Path where the generated task markdown file will be saved")]
        output: PathBuf,
        
        /// Prompt file containing analysis instructions
        #[arg(long, default_value = ".kiro/steering/spec-S04-steering-doc-analysis.md",
              help = "Path to prompt file with analysis instructions")]
        prompt_file: PathBuf,
        
        /// Chunk size in lines of code (LOC) for processing large files
        #[arg(long, help = "Split large files into chunks of specified LOC (50-10000)")]
        chunks: Option<usize>,
        
        /// Maximum number of tasks to generate (prevents Kiro overload)
        #[arg(long, default_value = "50", help = "Maximum number of tasks to generate (default: 50, prevents Kiro overload)")]
        max_tasks: usize,
        
        /// Enable windowed task system for large task sets
        #[arg(long, help = "Create windowed task system with master list and current window")]
        windowed: bool,
        
        /// Database path (can also be set globally)
        #[arg(long, help = "Path to PostgreSQL database directory")]
        db_path: Option<PathBuf>,
    },

    /// Advance to the next window in a windowed task system
    /// 
    /// Moves the current window to the next set of tasks, archiving the completed window
    /// and updating progress tracking. Use this after completing all tasks in the current window.
    /// 
    /// Example:
    ///   code-ingest advance-window .kiro/tasks/INGEST_20250929042515/
    AdvanceWindow {
        /// Path to the windowed task directory
        #[arg(help = "Directory containing the windowed task system")]
        task_dir: PathBuf,
    },

    /// Show progress for a windowed task system
    /// 
    /// Displays current progress, completion percentage, and window information
    /// for a windowed task system.
    /// 
    /// Example:
    ///   code-ingest task-progress .kiro/tasks/INGEST_20250929042515/
    TaskProgress {
        /// Path to the windowed task directory
        #[arg(help = "Directory containing the windowed task system")]
        task_dir: PathBuf,
    },

    /// Reset windowed task system to a specific position
    /// 
    /// Resets the current window to start at a specific task number. Useful for
    /// resuming work from a different position or correcting progress tracking.
    /// 
    /// Example:
    ///   code-ingest reset-window .kiro/tasks/INGEST_20250929042515/ --to 101
    ResetWindow {
        /// Path to the windowed task directory
        #[arg(help = "Directory containing the windowed task system")]
        task_dir: PathBuf,
        /// Task number to reset to (1-based)
        #[arg(long, help = "Task number to reset the current window to")]
        to: usize,
    },

    /// Generate content files and task lists with simplified chunk-level processing
    /// 
    /// This command replaces the complex task generation system with two simple modes:
    /// - File-level mode (no chunk size): generates content files for each database row
    /// - Chunk-level mode (with chunk size): processes large files with chunking
    /// 
    /// Examples:
    ///   # File-level mode - generate content files for each row
    ///   code-ingest chunk-level-task-generator INGEST_20250928101039 --db-path /path/to/db
    ///   
    ///   # Chunk-level mode - process large files with 500-line chunks
    ///   code-ingest chunk-level-task-generator INGEST_20250928101039 500 --db-path /path/to/db
    ///   
    ///   # Custom output directory
    ///   code-ingest chunk-level-task-generator INGEST_20250928101039 --output-dir ./output --db-path /path/to/db
    ChunkLevelTaskGenerator {
        /// Database table name containing ingested code data
        #[arg(help = "Name of the database table to generate tasks from (e.g., INGEST_20250928101039)")]
        table_name: String,
        
        /// Optional chunk size for large file processing (lines of code)
        #[arg(help = "Optional chunk size in lines of code for processing large files (50-10000)")]
        chunk_size: Option<usize>,
        
        /// Database path (can also be set globally)
        #[arg(long, help = "Path to PostgreSQL database directory")]
        db_path: Option<PathBuf>,
        
        /// Output directory for content files and task list
        #[arg(long, default_value = ".", help = "Directory where content files and task list will be created")]
        output_dir: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    
    /// Initialize default configuration file
    Init {
        /// Force overwrite existing configuration
        #[arg(long)]
        force: bool,
    },
    
    /// Set a configuration value
    Set {
        /// Configuration key (e.g., database.default_path)
        key: String,
        /// Configuration value
        value: String,
    },
    
    /// Get a configuration value
    Get {
        /// Configuration key (e.g., database.default_path)
        key: String,
    },
    
    /// Show configuration file path
    Path,
}

impl Default for Cli {
    fn default() -> Self {
        Self::new()
    }
}

impl Cli {
    pub fn new() -> Self {
        Self::parse()
    }
    
    /// Load configuration from file
    fn load_config(&self) -> Result<Config> {
        let config_manager = if let Some(config_path) = &self.config {
            ConfigManager::with_path(config_path.clone())
        } else {
            ConfigManager::new()
        };
        
        config_manager.load_config()
    }

    /// Initialize logging based on CLI arguments
    fn init_logging(&self) -> Result<()> {
        use crate::logging::{LoggingConfig, init_logging};
        
        let mut logging_config = LoggingConfig::default();
        
        // Set log level based on CLI arguments
        if self.verbose {
            logging_config.level = "debug".to_string();
        } else if let Some(ref level) = self.log_level {
            logging_config.level = level.clone();
        }
        
        // Set JSON format if requested
        logging_config.json_format = self.json_logs;
        
        // Disable progress reporting if requested
        logging_config.progress_reporting = !self.no_progress;
        
        // Try to load additional logging config from file
        if let Ok(_config) = self.load_config() {
            // Merge with file-based logging configuration if available
            // This would require extending the Config struct to include logging settings
        }
        
        init_logging(&logging_config).map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;
        
        tracing::info!("Code Ingest starting with log level: {}", logging_config.level);
        Ok(())
    }

    pub async fn run(&self) -> Result<()> {
        // Logging is initialized in main.rs
        
        match &self.command {
            Some(command) => self.execute_command(command).await,
            None => {
                // If no subcommand is provided, try to parse as ingest command
                // This handles the case: code-ingest <url> --db-path <path>
                let args: Vec<String> = std::env::args().collect();
                if args.len() >= 2 && !args[1].starts_with('-') {
                    let source = args[1].clone();
                    let db_path = self.db_path.clone();
                    // Default to false for folder_flag in legacy mode
                    self.execute_ingest(source, false, db_path, false, 0.8, true, false).await
                } else {
                    println!("Code ingestion tool");
                    println!("Use --help for usage information");
                    Ok(())
                }
            }
        }
    }

    async fn execute_command(&self, command: &Commands) -> Result<()> {
        match command {
            Commands::Ingest { source, folder_flag, db_path, auto_chunk, chunk_size, validate_chunks, cleanup_chunks } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_ingest(
                    source.clone(), 
                    *folder_flag, 
                    db_path, 
                    *auto_chunk, 
                    *chunk_size, 
                    *validate_chunks, 
                    *cleanup_chunks
                ).await
            }
            Commands::Sql { query, db_path, limit, offset, timeout, llm_format } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_sql(query.clone(), db_path, *limit, *offset, *timeout, *llm_format).await
            }
            Commands::QueryPrepare {
                query,
                db_path,
                temp_path,
                tasks_file,
                output_table,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_query_prepare(
                    query.clone(),
                    db_path,
                    temp_path.clone(),
                    tasks_file.clone(),
                    output_table.clone(),
                )
                .await
            }
            Commands::GenerateTasks {
                sql,
                prompt_file,
                output_table,
                tasks_file,
                db_path,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_generate_tasks(
                    sql.clone(),
                    prompt_file.clone(),
                    output_table.clone(),
                    tasks_file.clone(),
                    db_path,
                )
                .await
            }
            Commands::StoreResult {
                db_path,
                output_table,
                result_file,
                original_query,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_store_result(
                    db_path,
                    output_table.clone(),
                    result_file.clone(),
                    original_query.clone(),
                )
                .await
            }
            Commands::PrintToMd {
                db_path,
                table,
                sql,
                prefix,
                location,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_print_to_md(
                    db_path,
                    table.clone(),
                    sql.clone(),
                    prefix.clone(),
                    location.clone(),
                )
                .await
            }
            Commands::ListTables { db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_list_tables(db_path).await
            }
            Commands::Sample {
                db_path,
                table,
                limit,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_sample(db_path, table.clone(), *limit).await
            }
            Commands::Describe { db_path, table } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_describe(db_path, table.clone()).await
            }
            Commands::DbInfo { db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_db_info(db_path).await
            }
            Commands::PgStart => self.execute_pg_start().await,
            Commands::Setup => self.execute_setup().await,
            Commands::Examples => self.execute_examples().await,
            Commands::Troubleshoot => self.execute_troubleshoot().await,
            Commands::CleanupTables { db_path, keep, confirm } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_cleanup_tables(db_path, *keep, *confirm).await
            }
            Commands::DropTable { db_path, table, confirm } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_drop_table(db_path, table.clone(), *confirm).await
            }
            Commands::Recommendations { db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_recommendations(db_path).await
            }
            Commands::OptimizeTables { db_path, tables } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_optimize_tables(db_path, tables.clone()).await
            }
            Commands::CountRows { table_name, db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_count_rows(table_name.clone(), db_path).await
            }
            Commands::ExtractContent { table_name, output_dir, chunk_size, db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_extract_content(table_name.clone(), output_dir.clone(), *chunk_size, db_path).await
            }
            Commands::Config { action } => {
                self.execute_config_command(action).await
            }
            Commands::GenerateHierarchicalTasks { 
                table_name, 
                levels: _, // Ignore for now - keep simple
                groups: _, // Ignore for now - keep simple
                output, 
                prompt_file, 
                chunks,
                max_tasks,
                windowed: _, // Ignore for now - keep simple
                db_path 
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_generate_simple_tasks(
                    table_name.clone(),
                    output.clone(),
                    prompt_file.clone(),
                    *chunks,
                    *max_tasks,
                    db_path,
                ).await
            }
            Commands::AdvanceWindow { task_dir: _ } => {
                eprintln!("AdvanceWindow command not yet implemented");
                Ok(())
            }
            Commands::TaskProgress { task_dir: _ } => {
                eprintln!("TaskProgress command not yet implemented");
                Ok(())
            }
            Commands::ResetWindow { task_dir: _, to: _ } => {
                eprintln!("ResetWindow command not yet implemented");
                Ok(())
            }
            Commands::ChunkLevelTaskGenerator { table_name, chunk_size, db_path, output_dir } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_chunk_level_task_generator(
                    table_name.clone(),
                    *chunk_size,
                    db_path,
                    output_dir.clone(),
                ).await
            }
        }
    }

    #[instrument(skip(self))]
    async fn execute_ingest(
        &self, 
        source: String, 
        folder_flag: bool, 
        db_path: Option<PathBuf>,
        auto_chunk: bool,
        chunk_size: f64,
        validate_chunks: bool,
        cleanup_chunks: bool,
    ) -> Result<()> {
        use crate::database::{Database, SchemaManager};
        use crate::ingestion::{IngestionEngine, IngestionConfig};
        use crate::processing::text_processor::TextProcessor;
        use crate::ingestion::batch_processor::BatchProgress;
        use indicatif::{ProgressBar, ProgressStyle};
        use std::sync::Arc;
        
        // Initialize monitoring context
        let monitoring = MonitoringContext::new("ingestion", None);
        monitoring.performance.checkpoint("validation_start").await;
        
        // Validate inputs based on source type and folder flag
        info!(source = %source, folder_flag = folder_flag, "Starting ingestion validation");
        
        if folder_flag {
            // Local folder ingestion mode
            if !source.starts_with('/') {
                error!("Invalid local path: must be absolute path starting with '/'");
                anyhow::bail!("When using --folder-flag, source must be an absolute path (starting with '/'). Got: {}", source);
            }
            
            let path = std::path::Path::new(&source);
            if !path.exists() {
                error!(path = %source, "Local folder does not exist");
                anyhow::bail!("Local folder does not exist: {}", source);
            }
            
            if !path.is_dir() {
                error!(path = %source, "Path is not a directory");
                anyhow::bail!("Path is not a directory: {}", source);
            }
            
            info!("Local folder ingestion mode enabled");
            println!("📁 Local folder ingestion mode enabled");
        } else {
            // GitHub repository ingestion mode
            if source.starts_with('/') || source.starts_with('.') {
                error!("Local paths require --folder-flag parameter");
                anyhow::bail!("Local paths require --folder-flag parameter. Use: --folder-flag for local directory ingestion");
            }
            
            if let Err(e) = HelpSystem::validate_repository_url(&source) {
                error!(error = %e, "Repository URL validation failed");
                eprintln!("{}", format!("❌ {}", e).red());
                println!();
                HelpSystem::suggest_next_steps(&e).iter().for_each(|suggestion| {
                    println!("💡 {}", suggestion.yellow());
                });
                return Err(e.into());
            }
            
            info!("GitHub repository ingestion mode enabled");
            println!("🌐 GitHub repository ingestion mode");
        }
        
        monitoring.performance.checkpoint("validation_complete").await;

        if let Some(ref path) = db_path {
            if let Err(e) = HelpSystem::validate_database_path(path) {
                error!(path = ?path, error = %e, "Database path validation failed");
                eprintln!("{}", format!("❌ {}", e).red());
                println!();
                HelpSystem::suggest_next_steps(&e).iter().for_each(|suggestion| {
                    println!("💡 {}", suggestion.yellow());
                });
                return Err(e.into());
            }
        }
        
        info!("Starting ingestion from: {}", source);
        println!("{}", format!("🚀 Starting ingestion from: {}", source).bright_green());
        println!();
        
        // Get database connection
        monitoring.performance.checkpoint("database_connection_start").await;
        let database = if let Some(path) = db_path {
            info!(db_path = ?path, "Connecting to database from path");
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            info!("Connecting to database from DATABASE_URL");
            Database::new(&database_url).await?
        } else {
            error!("No database path provided");
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };
        monitoring.performance.checkpoint("database_connection_complete").await;

        // Initialize database schema
        monitoring.performance.checkpoint("schema_init_start").await;
        info!("Initializing database schema");
        println!("📋 Initializing database schema...");
        let schema_manager = SchemaManager::new(database.pool().clone());
        schema_manager.initialize_schema().await?;
        monitoring.performance.checkpoint("schema_init_complete").await;
        
        // Create progress bar
        let progress = ProgressBar::new_spinner();
        progress.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        
        // Create ingestion configuration
        let config = IngestionConfig::default();
        
        // Create file processor with optional chunking
        let base_processor = Arc::new(TextProcessor::new());
        let file_processor: Arc<dyn crate::processing::FileProcessor> = if auto_chunk {
            // Validate chunk size
            if chunk_size < 0.1 || chunk_size > 10.0 {
                return Err(anyhow::anyhow!("Chunk size must be between 0.1 and 10.0 MB, got {}", chunk_size));
            }
            
            info!("Auto-chunking enabled: chunk_size={}MB, validate={}, cleanup={}", 
                  chunk_size, validate_chunks, cleanup_chunks);
            
            // Create chunking configuration
            let chunking_config = crate::chunking::ChunkingConfig {
                chunk_size_mb: chunk_size,
                validate_chunks,
                cleanup_on_failure: true,
                output_dir: None,
            };
            
            let processor_config = crate::processing::ChunkingProcessorConfig {
                chunking_config,
                validate_chunks,
                cleanup_chunks,
                include_chunk_metadata: true,
            };
            
            // Wrap the base processor with chunking capabilities
            Arc::new(crate::processing::ChunkingProcessorFactory::wrap_processor_with_config(
                base_processor,
                processor_config,
            ))
        } else {
            info!("Auto-chunking disabled, using standard text processor");
            base_processor
        };
        
        // Create ingestion engine
        let engine = IngestionEngine::new(config, Arc::new(database), file_processor);
        
        // Start ingestion with progress callback
        monitoring.performance.checkpoint("ingestion_start").await;
        progress.set_message("Starting ingestion...");
        info!("Starting file ingestion");
        
        let progress_callback = {
            let progress = progress.clone();
            Box::new(move |batch_progress: BatchProgress| {
                progress.set_message(format!(
                    "Processing files: {} processed, {} total", 
                    batch_progress.files_processed, 
                    batch_progress.total_files
                ));
                
                // Log progress periodically
                if batch_progress.files_processed % 100 == 0 {
                    info!(
                        files_processed = batch_progress.files_processed,
                        total_files = batch_progress.total_files,
                        "Ingestion progress update"
                    );
                }
            })
        };
        
        let result = monitoring.performance.time_async("total_ingestion", || {
            engine.ingest_source(&source, Some(progress_callback))
        }).await?;
        
        monitoring.performance.checkpoint("ingestion_complete").await;
        
        progress.finish_and_clear();
        
        // Complete monitoring and generate reports
        monitoring.memory.check_memory().await;
        monitoring.complete_and_report().await;
        
        // Log comprehensive results
        info!(
            repository = %source,
            table_name = %result.table_name,
            files_processed = result.files_processed,
            files_failed = result.files_failed,
            files_skipped = result.files_skipped,
            processing_time_seconds = result.processing_time.as_secs_f64(),
            batches_processed = result.batch_stats.batches_processed,
            avg_file_duration_ms = result.batch_stats.avg_file_duration.as_millis(),
            peak_memory_mb = result.batch_stats.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            "Ingestion completed successfully"
        );
        
        // Display results
        println!("✅ Ingestion completed successfully!");
        println!();
        println!("📊 Ingestion Summary:");
        println!("   Repository: {}", source);
        println!("   Table: {}", result.table_name);
        println!("   Files processed: {}", result.files_processed);
        println!("   Processing time: {:.2}s", result.processing_time.as_secs_f64());
        
        // Display batch statistics
        println!();
        println!("📈 Processing Statistics:");
        println!("   Files processed: {}", result.files_processed);
        println!("   Files failed: {}", result.files_failed);
        println!("   Files skipped: {}", result.files_skipped);
        println!("   Batches processed: {}", result.batch_stats.batches_processed);
        println!("   Average file time: {:.2}ms", result.batch_stats.avg_file_duration.as_millis());
        println!("   Peak memory: {:.2} MB", result.batch_stats.peak_memory_bytes as f64 / (1024.0 * 1024.0));
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Explore your data: ./target/release/code-ingest list-tables --db-path <DB_PATH>");
        println!("   2. Sample the data: ./target/release/code-ingest sample --table {} --db-path <DB_PATH>", result.table_name);
        println!("   3. Run queries: ./target/release/code-ingest sql 'SELECT filepath, filename FROM \"{}\" LIMIT 5' --db-path <DB_PATH>", result.table_name);
        println!("   4. Export files: ./target/release/code-ingest print-to-md --table {} --sql 'SELECT * FROM \"{}\" LIMIT 10' --prefix analysis --location ./exports --db-path <DB_PATH>", result.table_name, result.table_name);
        
        Ok(())
    }

    async fn execute_sql(
        &self, 
        query: String, 
        db_path: Option<PathBuf>, 
        limit: usize, 
        offset: usize, 
        timeout: u64, 
        llm_format: bool
    ) -> Result<()> {
        use crate::database::{Database, QueryExecutor, query_executor::QueryConfig};
        
        // Validate inputs
        if let Err(e) = HelpSystem::validate_sql_query(&query) {
            eprintln!("{}", format!("❌ {}", e).red());
            println!();
            HelpSystem::suggest_next_steps(&e).iter().for_each(|suggestion| {
                println!("💡 {}", suggestion.yellow());
            });
            return Err(e.into());
        }

        if let Some(ref path) = db_path {
            if let Err(e) = HelpSystem::validate_database_path(path) {
                eprintln!("{}", format!("❌ {}", e).red());
                println!();
                HelpSystem::suggest_next_steps(&e).iter().for_each(|suggestion| {
                    println!("💡 {}", suggestion.yellow());
                });
                return Err(e.into());
            }
        }
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create query executor
        let executor = QueryExecutor::new(database.pool().clone());
        
        // Create query configuration
        let config = QueryConfig {
            max_rows: limit,
            offset,
            timeout_seconds: timeout,
            llm_format,
            include_stats: !llm_format, // Don't show stats in LLM format
            show_helpful_errors: true,
            ..Default::default()
        };
        
        // Execute query with configuration
        let result = executor.execute_query_with_config(&query, &config).await?;
        
        // Print results
        print!("{}", result.content);
        
        if result.truncated {
            if !llm_format {
                println!("\nNote: Results were truncated to {} rows", limit);
                if offset == 0 {
                    println!("Use --limit and --offset for pagination");
                    println!("Example: --limit 100 --offset 100 (for next 100 rows)");
                }
            }
        }
        
        // Show pagination info if using pagination
        if !llm_format && (limit > 0 || offset > 0) {
            println!("\nPagination: showing {} rows starting from row {}", 
                    if limit > 0 { limit } else { result.row_count }, 
                    offset + 1);
        }
        
        Ok(())
    }

    async fn execute_query_prepare(
        &self,
        query: String,
        db_path: Option<PathBuf>,
        temp_path: PathBuf,
        tasks_file: PathBuf,
        output_table: String,
    ) -> Result<()> {
        use crate::database::{Database, QueryExecutor, TempFileManager, TempFileConfig, TempFileMetadata, ResultStorage, StorageConfig};
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🚀 Preparing query results for IDE analysis...");
        println!();
        
        // Create progress bar
        let progress = ProgressBar::new(100);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Get database connection
        progress.set_message("Connecting to database...");
        progress.set_position(10);
        
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create query executor and temp file manager
        let executor = QueryExecutor::new(database.pool().clone());
        let temp_manager = TempFileManager::new(executor);
        
        // Validate temp path is absolute
        if !temp_path.is_absolute() {
            anyhow::bail!("Temporary file path must be absolute: {}", temp_path.display());
        }
        
        // Validate tasks file path is absolute
        if !tasks_file.is_absolute() {
            anyhow::bail!("Tasks file path must be absolute: {}", tasks_file.display());
        }
        
        progress.set_message("Executing SQL query...");
        progress.set_position(30);
        
        // Create metadata for the temporary file
        let metadata = TempFileMetadata {
            original_query: query.clone(),
            output_table: output_table.clone(),
            prompt_file_path: None, // Will be set when tasks are generated
            description: Some("Query results prepared for IDE analysis".to_string()),
        };
        
        // Create temporary file with query results
        let config = TempFileConfig::default();
        let temp_result = temp_manager
            .create_structured_temp_file(&query, &temp_path, &metadata, &config)
            .await?;
        
        progress.set_message("Creating output table...");
        progress.set_position(60);
        
        // Ensure the output table exists for storing analysis results
        let result_storage = ResultStorage::new(database.pool().clone());
        let storage_config = StorageConfig::default();
        
        // Create the output table by attempting to store a placeholder result
        // This ensures the table schema is created
        let placeholder_metadata = crate::database::ResultMetadata {
            original_query: query.clone(),
            prompt_file_path: None,
            analysis_type: Some("query_prepare_placeholder".to_string()),
            original_file_path: None,
            chunk_number: None,
            created_by: Some("code-ingest-query-prepare".to_string()),
            tags: vec!["placeholder".to_string()],
        };
        
        // Store and immediately delete placeholder to create table
        let placeholder_result = result_storage
            .store_result_content(&output_table, "PLACEHOLDER - DELETE ME", &placeholder_metadata, &storage_config)
            .await?;
        
        // Delete the placeholder
        result_storage
            .delete_analysis_result(&output_table, placeholder_result.analysis_id)
            .await?;
        
        progress.set_message("Generating task structure...");
        progress.set_position(80);
        
        // Create a comprehensive task structure for IDE integration
        let task_content = self.create_comprehensive_task_structure(&query, &temp_path, &output_table, temp_result.row_count);
        
        // Write tasks file
        tokio::fs::write(&tasks_file, task_content).await?;
        
        progress.set_message("Complete!");
        progress.set_position(100);
        progress.finish_with_message("✅ Query preparation complete!");
        
        // Display results
        println!();
        println!("📊 Query Preparation Summary:");
        println!("   SQL Query: {}", query);
        println!("   Rows Retrieved: {}", temp_result.row_count);
        println!("   Execution Time: {}ms", temp_result.execution_time_ms);
        println!("   Temporary File: {} ({} bytes)", temp_path.display(), temp_result.bytes_written);
        println!("   Tasks File: {}", tasks_file.display());
        println!("   Output Table: {} (ready for results)", output_table);
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Open the tasks file in your IDE: {}", tasks_file.display());
        println!("   2. Review the generated task structure");
        println!("   3. Execute analysis tasks systematically");
        println!("   4. Store results using:");
        println!("      code-ingest store-result \\");
        println!("        --output-table {} \\", output_table);
        println!("        --result-file <your_analysis_result.txt> \\");
        println!("        --original-query \"{}\"", query);
        
        println!();
        println!("💡 Pro Tips:");
        println!("   • The temporary file contains structured data ready for LLM processing");
        println!("   • Tasks are organized for systematic analysis");
        println!("   • The output table is pre-created and ready to store your analysis results");
        println!("   • Use the FILE: format in the temp file for context-aware analysis");
        
        Ok(())
    }

    async fn execute_generate_tasks(
        &self,
        sql: String,
        prompt_file: PathBuf,
        output_table: String,
        tasks_file: PathBuf,
        db_path: Option<PathBuf>,
    ) -> Result<()> {
        use crate::database::{Database, DatabaseOperations};
        use crate::tasks::{TaskDivider, TaskConfig, MarkdownGenerator, TaskMetadata, TaskStructure, QueryResultRow};
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🚀 Generating structured analysis tasks...");
        println!();
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Validate prompt file exists
        if !prompt_file.exists() {
            anyhow::bail!("Prompt file not found: {}", prompt_file.display());
        }

        // Create progress bar
        let progress = ProgressBar::new(100);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Step 1: Execute SQL query to get data
        progress.set_message("Executing SQL query...");
        progress.set_position(10);
        
        let operations = DatabaseOperations::new(database.pool().clone());
        let query_result = operations.execute_query(&sql).await?;
        
        if query_result.rows.is_empty() {
            anyhow::bail!("Query returned no results. Cannot generate tasks from empty dataset.");
        }
        
        progress.set_message("Processing query results...");
        progress.set_position(30);
        
        // Convert query results to QueryResultRow format
        let query_rows: Vec<QueryResultRow> = query_result.rows.into_iter().map(|row| {
            QueryResultRow {
                file_id: row.get("file_id").and_then(|s| s.parse().ok()),
                filepath: row.get("filepath").cloned(),
                filename: row.get("filename").cloned(),
                extension: row.get("extension").cloned(),
                file_size_bytes: row.get("file_size_bytes").and_then(|s| s.parse().ok()),
                line_count: row.get("line_count").and_then(|s| s.parse().ok()),
                word_count: row.get("word_count").and_then(|s| s.parse().ok()),
                token_count: row.get("token_count").and_then(|s| s.parse().ok()),
                content_text: row.get("content_text").cloned(),
                file_type: row.get("file_type").cloned(),
                conversion_command: row.get("conversion_command").cloned(),
                relative_path: row.get("relative_path").cloned(),
                absolute_path: row.get("absolute_path").cloned(),
            }
        }).collect();
        
        // Step 2: Create task configuration
        let config = TaskConfig {
            sql_query: sql.clone(),
            output_table: output_table.clone(),
            tasks_file: tasks_file.to_string_lossy().to_string(),
            group_count: 7, // Default to 7 groups as per requirements
            chunk_size: Some(300), // Default chunk size for large files
            chunk_overlap: Some(50), // Default overlap
            prompt_file: Some(prompt_file.to_string_lossy().to_string()),
        };
        
        // Step 3: Create task divider and generate tasks
        progress.set_message("Creating task structure...");
        progress.set_position(50);
        
        let divider = TaskDivider::new(config.group_count)?;
        let tasks = divider.create_tasks_from_query_results(&query_rows, &config)?;
        
        if tasks.is_empty() {
            anyhow::bail!("No tasks could be generated from the query results.");
        }
        
        // Adjust group count if we have fewer tasks than groups
        let actual_group_count = std::cmp::min(config.group_count, tasks.len());
        let adjusted_divider = if actual_group_count != config.group_count {
            println!("ℹ️  Adjusting group count from {} to {} (limited by number of tasks)", 
                config.group_count, actual_group_count);
            TaskDivider::new(actual_group_count)?
        } else {
            divider
        };
        
        // Step 4: Divide tasks into groups
        progress.set_message("Dividing tasks into groups...");
        progress.set_position(70);
        
        let groups = adjusted_divider.divide_into_groups(tasks.clone())?;
        
        // Step 5: Create task structure with metadata
        let metadata = TaskMetadata {
            total_tasks: tasks.len(),
            group_count: actual_group_count,
            sql_query: sql.clone(),
            output_table: output_table.clone(),
            generated_at: chrono::Utc::now(),
            prompt_file: Some(prompt_file.to_string_lossy().to_string()),
        };
        
        let task_structure = TaskStructure {
            groups,
            total_tasks: tasks.len(),
            metadata,
        };
        
        // Step 6: Generate markdown
        progress.set_message("Generating markdown...");
        progress.set_position(90);
        
        let generator = MarkdownGenerator::new();
        generator.write_to_file(&task_structure, &tasks_file.to_string_lossy()).await?;
        
        progress.set_message("Complete!");
        progress.set_position(100);
        progress.finish_with_message("✅ Task generation complete!");
        
        // Display results
        println!();
        println!("📊 Task Generation Summary:");
        println!("   SQL Query: {}", sql);
        println!("   Data Rows: {}", query_rows.len());
        println!("   Generated Tasks: {}", tasks.len());
        println!("   Task Groups: {}", actual_group_count);
        println!("   Output Table: {}", output_table);
        println!("   Prompt File: {}", prompt_file.display());
        println!("   Tasks File: {}", tasks_file.display());
        
        // Show task distribution
        println!();
        println!("📋 Task Distribution:");
        for (i, group) in task_structure.groups.iter().enumerate() {
            println!("   Group {}: {} tasks", i + 1, group.tasks.len());
        }
        
        // Show file statistics if available
        if let Some(first_row) = query_rows.first() {
            if first_row.file_type.is_some() {
                let file_types: std::collections::HashMap<String, usize> = query_rows
                    .iter()
                    .filter_map(|row| row.file_type.as_ref())
                    .fold(std::collections::HashMap::new(), |mut acc, file_type| {
                        *acc.entry(file_type.clone()).or_insert(0) += 1;
                        acc
                    });
                
                if !file_types.is_empty() {
                    println!();
                    println!("📁 File Type Distribution:");
                    for (file_type, count) in file_types {
                        println!("   {}: {} files", file_type, count);
                    }
                }
            }
        }
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Open the tasks file in your IDE: {}", tasks_file.display());
        println!("   2. Review the generated task structure");
        println!("   3. Execute tasks systematically, starting with Group 1");
        println!("   4. Use the prompt file for analysis guidance: {}", prompt_file.display());
        println!("   5. Store results using the store-result command when complete");
        
        println!();
        println!("💡 Pro Tips:");
        println!("   • Tasks are numbered with Kiro-compatible format (1., 1.1, 1.2, etc.)");
        println!("   • Large files are automatically chunked for manageable analysis");
        println!("   • Each task includes file metadata and chunk information");
        println!("   • Complete all sub-tasks in a group before moving to the next group");
        
        Ok(())
    }

    async fn execute_store_result(
        &self,
        db_path: Option<PathBuf>,
        output_table: String,
        result_file: PathBuf,
        original_query: String,
    ) -> Result<()> {
        use crate::database::{Database, ResultStorage, StorageConfig, ResultMetadata};
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🚀 Storing analysis results...");
        println!();
        
        // Create progress bar
        let progress = ProgressBar::new(100);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Validate result file exists
        if !result_file.exists() {
            anyhow::bail!("Result file not found: {}", result_file.display());
        }
        
        progress.set_message("Connecting to database...");
        progress.set_position(10);
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        progress.set_message("Reading result file...");
        progress.set_position(30);
        
        // Read and validate result file
        let result_content = tokio::fs::read_to_string(&result_file).await?;
        let _result_size = result_content.len();
        
        if result_content.trim().is_empty() {
            anyhow::bail!("Result file is empty: {}", result_file.display());
        }
        
        progress.set_message("Preparing storage...");
        progress.set_position(50);
        
        // Create result storage manager
        let storage = ResultStorage::new(database.pool().clone());
        
        // Detect analysis type from content
        let analysis_type = self.detect_analysis_type_from_content(&result_content, &original_query);
        
        // Create metadata for the result
        let metadata = ResultMetadata {
            original_query: original_query.clone(),
            prompt_file_path: None, // Could be enhanced to detect this from task files
            analysis_type: Some(analysis_type.clone()),
            original_file_path: Some(result_file.to_string_lossy().to_string()),
            chunk_number: None,
            created_by: Some("code-ingest-cli".to_string()),
            tags: vec!["analysis".to_string(), "cli_stored".to_string()],
        };
        
        progress.set_message("Storing results...");
        progress.set_position(80);
        
        // Store the result
        let config = StorageConfig::default();
        let storage_result = storage
            .store_result_from_file(&output_table, &result_file, &metadata, &config)
            .await?;
        
        progress.set_message("Complete!");
        progress.set_position(100);
        progress.finish_with_message("✅ Analysis results stored!");
        
        // Display results
        println!();
        println!("📊 Storage Summary:");
        println!("   Analysis ID: {}", storage_result.analysis_id);
        println!("   Table: {}", storage_result.table_name);
        println!("   Result Size: {} bytes", storage_result.result_size_bytes);
        println!("   Analysis Type: {}", analysis_type);
        println!("   Original Query: {}", original_query);
        println!("   Source File: {}", result_file.display());
        
        // Get storage statistics
        match storage.get_storage_stats(&output_table).await {
            Ok(stats) => {
                println!();
                println!("📈 Table Statistics:");
                println!("   Total Results: {}", stats.total_results);
                println!("   Average Size: {} bytes", stats.avg_result_size_bytes);
                println!("   Largest Result: {} bytes", stats.max_result_size_bytes);
                if let Some(oldest) = stats.oldest_result {
                    println!("   Oldest Result: {}", oldest.format("%Y-%m-%d %H:%M:%S UTC"));
                }
                if let Some(newest) = stats.newest_result {
                    println!("   Newest Result: {}", newest.format("%Y-%m-%d %H:%M:%S UTC"));
                }
            }
            Err(e) => {
                println!("   (Could not retrieve table statistics: {})", e);
            }
        }
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Verify storage:");
        println!("      code-ingest sql \"SELECT analysis_id, analysis_type, LENGTH(llm_result) as size, created_at FROM {} ORDER BY created_at DESC LIMIT 5\"", output_table);
        println!("   2. Retrieve your result:");
        println!("      code-ingest sql \"SELECT llm_result FROM {} WHERE analysis_id = {}\"", output_table, storage_result.analysis_id);
        println!("   3. Continue analysis workflow or export results as needed");
        
        println!();
        println!("💡 Pro Tips:");
        println!("   • Results are now permanently stored and linked to the original query");
        println!("   • Use the analysis_id ({}) to reference this specific analysis", storage_result.analysis_id);
        println!("   • The analysis_type ({}) helps categorize and filter results", analysis_type);
        println!("   • All metadata is preserved for full traceability");
        
        Ok(())
    }

    // This method was moved to the end of the impl block to avoid conflicts


    async fn execute_print_to_md(
        &self,
        db_path: Option<PathBuf>,
        table: String,
        sql: String,
        prefix: String,
        location: PathBuf,
    ) -> Result<()> {
        use crate::database::{Database, DatabaseExporter, ExportConfig};
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🚀 Exporting query results to markdown files...");
        println!();
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create database exporter
        let exporter = DatabaseExporter::new(database.pool().clone());
        
        // Create export configuration
        let config = ExportConfig {
            prefix,
            location: location.clone(),
            max_files: Some(1000), // Safety limit
            overwrite_existing: false,
            markdown_template: None,
        };

        // Validate location
        if !location.exists() {
            println!("📁 Creating output directory: {}", location.display());
            tokio::fs::create_dir_all(&location).await?;
        }

        // Create progress bar
        let progress = ProgressBar::new_spinner();
        progress.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        progress.set_message("Executing query and exporting files...");

        // Execute export
        let result = exporter.export_to_markdown_files(&table, &sql, &config).await?;
        
        progress.finish_and_clear();

        // Display results
        let formatted = exporter.format_export_result(&result);
        println!("{}", formatted);
        
        if result.files_created > 0 {
            println!();
            println!("📂 Files exported to: {}", location.display());
            println!("🔍 Example files:");
            for (i, file_path) in result.created_files.iter().take(3).enumerate() {
                println!("   {}. {}", i + 1, file_path.display());
            }
            
            if result.created_files.len() > 3 {
                println!("   ... and {} more files", result.created_files.len() - 3);
            }
            
            println!();
            println!("💡 Pro Tips:");
            println!("   • Each file contains one row from your query results");
            println!("   • Files are numbered sequentially: {}-00001.md, {}-00002.md, etc.", config.prefix, config.prefix);
            println!("   • Use these files for individual analysis or processing");
            println!("   • File content is formatted as markdown with syntax highlighting");
        }
        
        Ok(())
    }

    async fn execute_list_tables(&self, db_path: Option<PathBuf>) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create database explorer
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // List all tables
        let tables = explorer.list_tables(None).await?;
        
        if tables.is_empty() {
            println!("No tables found in the database.");
            println!("\n💡 To get started:");
            println!("   1. Ingest a repository: code-ingest <repo_url> --db-path <path>");
            println!("   2. Or check database connection: code-ingest db-info --db-path <path>");
            return Ok(());
        }

        // Format and display tables
        let formatted = explorer.format_table_list(&tables, true);
        println!("{}", formatted);
        
        Ok(())
    }

    async fn execute_sample(
        &self,
        db_path: Option<PathBuf>,
        table: String,
        limit: usize,
    ) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        
        // Validate inputs
        if let Err(e) = HelpSystem::validate_table_name(&table) {
            eprintln!("{}", format!("❌ {}", e).red());
            println!();
            HelpSystem::suggest_next_steps(&e).iter().for_each(|suggestion| {
                println!("💡 {}", suggestion.yellow());
            });
            return Err(e.into());
        }

        if let Some(ref path) = db_path {
            if let Err(e) = HelpSystem::validate_database_path(path) {
                eprintln!("{}", format!("❌ {}", e).red());
                println!();
                HelpSystem::suggest_next_steps(&e).iter().for_each(|suggestion| {
                    println!("💡 {}", suggestion.yellow());
                });
                return Err(e.into());
            }
        }
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create database explorer
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Get sample data
        let sample = explorer.sample_table(&table, limit).await?;
        
        // Format and display sample
        let formatted = explorer.format_table_sample(&sample);
        println!("{}", formatted);
        
        if sample.total_rows > sample.sample_size as i64 {
            println!("\n💡 Showing {} of {} total rows. Use --limit to see more.", 
                    sample.sample_size, sample.total_rows);
        }
        
        Ok(())
    }

    async fn execute_describe(&self, db_path: Option<PathBuf>, table: String) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create database explorer
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Get table schema
        let schema = explorer.describe_table(&table).await?;
        
        // Format and display schema
        let formatted = explorer.format_table_schema(&schema);
        println!("{}", formatted);
        
        Ok(())
    }

    async fn execute_db_info(&self, db_path: Option<PathBuf>) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create database explorer
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Get database info
        let info = explorer.get_database_info().await?;
        
        // Format and display info
        let formatted = explorer.format_database_info(&info);
        println!("{}", formatted);
        
        Ok(())
    }

    async fn execute_pg_start(&self) -> Result<()> {
        use crate::database::PostgreSQLSetup;
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🐘 PostgreSQL Setup Assistant");
        println!("==============================\n");
        
        // Create setup manager
        let setup = PostgreSQLSetup::new();
        
        // Show progress while gathering system info
        let progress = ProgressBar::new_spinner();
        progress.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        progress.set_message("Analyzing your system...");
        
        // Get system information
        let system_info = setup.get_system_info().await;
        progress.finish_and_clear();
        
        // Generate setup instructions
        let instructions = setup.generate_setup_instructions().await;
        
        // Display formatted instructions
        let formatted_instructions = setup.format_setup_instructions(&instructions, &system_info);
        println!("{}", formatted_instructions);
        
        // Test connection if PostgreSQL appears to be installed
        if system_info.has_psql {
            println!("\n🔍 Testing PostgreSQL Connection");
            println!("=================================\n");
            
            let progress = ProgressBar::new_spinner();
            progress.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg}")
                    .unwrap()
            );
            progress.set_message("Testing database connection...");
            
            let connection_test = setup.test_connection(None).await;
            progress.finish_and_clear();
            
            let formatted_test = setup.format_connection_test(&connection_test);
            println!("{}", formatted_test);
            
            // If connection successful, show additional options
            if connection_test.success {
                println!("\n🚀 Ready to Get Started!");
                println!("========================\n");
                println!("Your PostgreSQL setup is working! Here are some next steps:\n");
                
                println!("**Try these commands:**");
                println!("```bash");
                println!("# Ingest a small repository");
                println!("code-ingest https://github.com/rust-lang/mdBook");
                println!();
                println!("# List your tables");
                println!("code-ingest list-tables");
                println!();
                println!("# Run a simple query");
                println!("code-ingest sql \"SELECT COUNT(*) FROM ingestion_meta\"");
                println!("```\n");
                
                println!("**Explore your data:**");
                println!("- `code-ingest db-info` - Show database information");
                println!("- `code-ingest sample --table <table_name>` - Preview table data");
                println!("- `code-ingest describe --table <table_name>` - Show table schema");
                println!();
                
                println!("📚 **Documentation**: https://github.com/your-repo/code-ingest");
                println!("🐛 **Issues**: https://github.com/your-repo/code-ingest/issues");
            } else {
                println!("\n🔧 Need Help?");
                println!("=============\n");
                println!("If you're still having trouble, try these resources:\n");
                
                println!("**Common Solutions:**");
                println!("1. Make sure PostgreSQL is installed and running");
                println!("2. Check your DATABASE_URL format");
                println!("3. Verify database and user permissions");
                println!("4. Review the troubleshooting tips above");
                println!();
                
                println!("**Get Support:**");
                println!("- Check the setup guide: https://github.com/your-repo/code-ingest/docs/setup");
                println!("- Open an issue: https://github.com/your-repo/code-ingest/issues");
                println!("- Join our community: https://discord.gg/your-discord");
            }
        } else {
            println!("\n📋 Next Steps");
            println!("=============\n");
            println!("1. Follow the installation instructions above");
            println!("2. Run `code-ingest pg-start` again to test your setup");
            println!("3. Once connected, try ingesting your first repository!");
        }
        
        Ok(())
    }



    async fn execute_cleanup_tables(&self, db_path: Option<PathBuf>, keep: usize, confirm: bool) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        use std::io::{self, Write};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Get current ingestion tables
        let tables = explorer.list_tables(Some(crate::database::TableType::Ingestion)).await?;
        
        if tables.len() <= keep {
            println!("✅ No cleanup needed. You have {} ingestion tables, keeping {}", tables.len(), keep);
            return Ok(());
        }
        
        let tables_to_remove = tables.len() - keep;
        
        if !confirm {
            println!("🗑️  Table Cleanup Preview");
            println!("   Current tables: {}", tables.len());
            println!("   Tables to keep: {}", keep);
            println!("   Tables to remove: {}", tables_to_remove);
            println!();
            print!("Are you sure you want to proceed? (y/N): ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if !input.trim().to_lowercase().starts_with('y') {
                println!("Cleanup cancelled.");
                return Ok(());
            }
        }
        
        println!("🚀 Starting table cleanup...");
        let result = explorer.cleanup_old_tables(keep).await?;
        
        println!("✅ Cleanup completed!");
        println!("   Tables removed: {}", result.tables_removed);
        println!("   Tables kept: {}", result.tables_kept);
        println!("   Space freed: {:.2} MB", result.space_freed_mb);
        
        if !result.errors.is_empty() {
            println!("\n⚠️  Errors occurred:");
            for error in &result.errors {
                println!("   • {}", error);
            }
        }
        
        Ok(())
    }

    async fn execute_drop_table(&self, db_path: Option<PathBuf>, table: String, confirm: bool) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        use std::io::{self, Write};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Check if table exists
        if !explorer.schema_manager.table_exists(&table).await? {
            anyhow::bail!("Table '{}' does not exist", table);
        }
        
        // Get table info for confirmation
        let sample = explorer.sample_table(&table, 0).await?;
        
        if !confirm {
            println!("🗑️  Drop Table Confirmation");
            println!("   Table: {}", table);
            println!("   Rows: {}", sample.total_rows);
            println!();
            println!("⚠️  This action cannot be undone!");
            print!("Are you sure you want to drop this table? (y/N): ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            
            if !input.trim().to_lowercase().starts_with('y') {
                println!("Drop cancelled.");
                return Ok(());
            }
        }
        
        println!("🗑️  Dropping table '{}'...", table);
        explorer.drop_table(&table).await?;
        
        println!("✅ Table '{}' dropped successfully", table);
        
        Ok(())
    }

    async fn execute_recommendations(&self, db_path: Option<PathBuf>) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        let explorer = DatabaseExplorer::new(database.pool().clone());
        let recommendations = explorer.get_management_recommendations().await?;
        
        if recommendations.is_empty() {
            println!("✅ No management recommendations at this time.");
            println!("Your database appears to be well-maintained!");
            return Ok(());
        }
        
        println!("📋 Database Management Recommendations");
        println!("=====================================\n");
        
        for (i, rec) in recommendations.iter().enumerate() {
            let priority_icon = match rec.priority {
                crate::database::exploration::Priority::High => "🔴",
                crate::database::exploration::Priority::Medium => "🟡",
                crate::database::exploration::Priority::Low => "🟢",
            };
            
            let type_icon = match rec.recommendation_type {
                crate::database::exploration::RecommendationType::Cleanup => "🗑️",
                crate::database::exploration::RecommendationType::Performance => "⚡",
                crate::database::exploration::RecommendationType::Security => "🔒",
                crate::database::exploration::RecommendationType::Storage => "💾",
            };
            
            println!("{}. {} {} {}", i + 1, priority_icon, type_icon, rec.title);
            println!("   {}", rec.description);
            println!("   Action: {}", rec.action);
            println!();
        }
        
        Ok(())
    }

    async fn execute_optimize_tables(&self, db_path: Option<PathBuf>, tables: Option<Vec<String>>) -> Result<()> {
        use crate::database::{Database, DatabaseExplorer};
        use indicatif::{ProgressBar, ProgressStyle};
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Determine which tables to optimize
        let table_list = if let Some(specific_tables) = &tables {
            specific_tables.clone()
        } else {
            explorer.schema_manager.list_tables(None).await?
        };
        
        if table_list.is_empty() {
            println!("No tables to optimize.");
            return Ok(());
        }
        
        println!("⚡ Optimizing {} tables...", table_list.len());
        
        // Create progress bar
        let progress = ProgressBar::new(table_list.len() as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        let result = explorer.optimize_tables(tables).await?;
        
        progress.finish_and_clear();
        
        println!("✅ Optimization completed!");
        println!("   Tables optimized: {}/{}", result.tables_optimized, result.total_tables);
        println!("   Duration: {}ms", result.duration_ms);
        
        if !result.errors.is_empty() {
            println!("\n⚠️  Errors occurred:");
            for error in &result.errors {
                println!("   • {}", error);
            }
        }
        
        Ok(())
    }



    /// Create a comprehensive task structure for query-prepare operations
    fn create_comprehensive_task_structure(&self, query: &str, temp_path: &PathBuf, output_table: &str, row_count: usize) -> String {
        let analysis_type = self.detect_analysis_type(query);
        let task_suggestions = self.generate_task_suggestions(&analysis_type, row_count);
        
        format!(
            r#"# IDE Analysis Tasks

## Query Preparation Metadata

- **Generated At**: {}
- **SQL Query**: `{}`
- **Temporary File**: `{}`
- **Output Table**: `{}`
- **Row Count**: {}
- **Analysis Type**: {}

## Task Overview

This document provides a structured approach to analyzing the query results. The tasks are organized to ensure systematic and thorough analysis.

## Analysis Tasks

### Phase 1: Data Exploration

- [ ] 1. Initial Data Review
  - [ ] 1.1 Open and examine the temporary file structure
    - **File**: `{}`
    - **Action**: Review the FILE: format and data organization
    - **Goal**: Understand the scope and structure of the data
  
  - [ ] 1.2 Identify data patterns and characteristics
    - **Action**: Look for common patterns, file types, and content themes
    - **Goal**: Establish analysis focus areas

### Phase 2: Systematic Analysis

{}

### Phase 3: Results and Documentation

- [ ] 3. Compile and Store Results
  - [ ] 3.1 Synthesize analysis findings
    - **Action**: Combine insights from all analysis tasks
    - **Goal**: Create comprehensive analysis summary
  
  - [ ] 3.2 Format results for storage
    - **Action**: Structure findings in clear, actionable format
    - **Goal**: Ensure results are useful for future reference
  
  - [ ] 3.3 Store results in database
    - **Command**: See storage commands below
    - **Goal**: Persist analysis for traceability and future use

## Storage Commands

```bash
# Store your analysis results (update paths as needed)
code-ingest store-result \
  --output-table {} \
  --result-file /path/to/your/analysis_result.txt \
  --original-query "{}"

# Verify storage
code-ingest sql "SELECT analysis_id, analysis_type, LENGTH(llm_result) as result_size, created_at FROM {} ORDER BY created_at DESC LIMIT 5"
```

## Analysis Guidelines

### Data Processing Tips
- Use the FILE: markers to identify individual files in the temporary data
- Focus on patterns that emerge across multiple files
- Document specific examples to support your findings

### Result Formatting
- Structure your analysis with clear headings and bullet points
- Include specific file references and line numbers when relevant
- Provide actionable recommendations based on your findings

### Quality Assurance
- Review your analysis for completeness and accuracy
- Ensure all significant patterns and issues are documented
- Verify that recommendations are specific and actionable

## Notes

- **Row Count**: {} files/records to analyze
- **Analysis Focus**: {}
- **Expected Output**: Structured analysis results stored in `{}`
- **Traceability**: Results linked to original query for future reference

Complete tasks systematically, starting with Phase 1 and progressing through each phase in order.
"#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            query,
            temp_path.display(),
            output_table,
            row_count,
            analysis_type,
            temp_path.display(),
            task_suggestions,
            output_table,
            query,
            output_table,
            row_count,
            analysis_type,
            output_table
        )
    }

    /// Detect the type of analysis based on the SQL query
    fn detect_analysis_type(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("security") || query_lower.contains("vulnerability") {
            "Security Analysis".to_string()
        } else if query_lower.contains("performance") || query_lower.contains("optimization") {
            "Performance Analysis".to_string()
        } else if query_lower.contains("architecture") || query_lower.contains("design") {
            "Architecture Review".to_string()
        } else if query_lower.contains("test") || query_lower.contains("coverage") {
            "Testing Analysis".to_string()
        } else if query_lower.contains("documentation") || query_lower.contains("comment") {
            "Documentation Review".to_string()
        } else if query_lower.contains("dependency") || query_lower.contains("import") {
            "Dependency Analysis".to_string()
        } else if query_lower.contains("error") || query_lower.contains("exception") {
            "Error Analysis".to_string()
        } else {
            "General Code Analysis".to_string()
        }
    }

    /// Generate task suggestions based on analysis type and data size
    fn generate_task_suggestions(&self, analysis_type: &str, row_count: usize) -> String {
        let batch_size = if row_count > 100 { 
            "Process in batches of 20-30 files for manageable analysis"
        } else if row_count > 50 {
            "Process in batches of 10-15 files for thorough analysis"
        } else {
            "Analyze all files systematically"
        };

        match analysis_type {
            "Security Analysis" => format!(
                r#"- [ ] 2. Security-Focused Analysis
  - [ ] 2.1 Identify potential security vulnerabilities
    - **Focus**: Look for SQL injection, XSS, authentication issues
    - **Method**: {}
    - **Goal**: Document security risks and recommendations
  
  - [ ] 2.2 Review access control and permissions
    - **Focus**: Authentication, authorization, privilege escalation
    - **Goal**: Ensure proper security boundaries
  
  - [ ] 2.3 Analyze data handling and validation
    - **Focus**: Input validation, data sanitization, encryption
    - **Goal**: Identify data security improvements"#,
                batch_size
            ),
            "Performance Analysis" => format!(
                r#"- [ ] 2. Performance-Focused Analysis
  - [ ] 2.1 Identify performance bottlenecks
    - **Focus**: Slow algorithms, inefficient queries, resource usage
    - **Method**: {}
    - **Goal**: Document performance issues and optimization opportunities
  
  - [ ] 2.2 Review resource utilization patterns
    - **Focus**: Memory usage, CPU intensive operations, I/O patterns
    - **Goal**: Identify resource optimization opportunities
  
  - [ ] 2.3 Analyze scalability considerations
    - **Focus**: Concurrent access, load handling, caching strategies
    - **Goal**: Recommend scalability improvements"#,
                batch_size
            ),
            "Architecture Review" => format!(
                r#"- [ ] 2. Architecture-Focused Analysis
  - [ ] 2.1 Review system design patterns
    - **Focus**: Design patterns, separation of concerns, modularity
    - **Method**: {}
    - **Goal**: Assess architectural quality and consistency
  
  - [ ] 2.2 Analyze component relationships
    - **Focus**: Dependencies, coupling, cohesion, interfaces
    - **Goal**: Identify architectural improvements
  
  - [ ] 2.3 Evaluate maintainability factors
    - **Focus**: Code organization, extensibility, testability
    - **Goal**: Recommend architectural enhancements"#,
                batch_size
            ),
            _ => format!(
                r#"- [ ] 2. Comprehensive Code Analysis
  - [ ] 2.1 Review code quality and patterns
    - **Focus**: Code style, best practices, common patterns
    - **Method**: {}
    - **Goal**: Identify quality improvements and consistency issues
  
  - [ ] 2.2 Analyze functionality and logic
    - **Focus**: Business logic, error handling, edge cases
    - **Goal**: Document functionality and potential improvements
  
  - [ ] 2.3 Assess maintainability and documentation
    - **Focus**: Code clarity, comments, documentation completeness
    - **Goal**: Recommend maintainability enhancements"#,
                batch_size
            )
        }
    }

    /// Detect analysis type from result content and original query
    fn detect_analysis_type_from_content(&self, content: &str, query: &str) -> String {
        let content_lower = content.to_lowercase();
        let _query_lower = query.to_lowercase();
        
        // Check content for analysis type indicators
        if content_lower.contains("security") || content_lower.contains("vulnerability") || content_lower.contains("exploit") {
            "Security Analysis".to_string()
        } else if content_lower.contains("performance") || content_lower.contains("optimization") || content_lower.contains("bottleneck") {
            "Performance Analysis".to_string()
        } else if content_lower.contains("architecture") || content_lower.contains("design pattern") || content_lower.contains("structure") {
            "Architecture Review".to_string()
        } else if content_lower.contains("test") || content_lower.contains("coverage") || content_lower.contains("assertion") {
            "Testing Analysis".to_string()
        } else if content_lower.contains("documentation") || content_lower.contains("comment") || content_lower.contains("readme") {
            "Documentation Review".to_string()
        } else if content_lower.contains("dependency") || content_lower.contains("import") || content_lower.contains("library") {
            "Dependency Analysis".to_string()
        } else if content_lower.contains("error") || content_lower.contains("exception") || content_lower.contains("bug") {
            "Error Analysis".to_string()
        } else if content_lower.contains("refactor") || content_lower.contains("cleanup") || content_lower.contains("improvement") {
            "Code Refactoring".to_string()
        } else {
            // Fall back to query-based detection
            self.detect_analysis_type(query)
        }
    }

    /// Show comprehensive setup guide
    async fn execute_setup(&self) -> Result<()> {
        HelpSystem::show_setup_guide();
        Ok(())
    }

    /// Show command examples for common tasks
    async fn execute_examples(&self) -> Result<()> {
        HelpSystem::show_examples();
        Ok(())
    }

    /// Show troubleshooting guide
    async fn execute_troubleshoot(&self) -> Result<()> {
        HelpSystem::show_troubleshooting_guide();
        Ok(())
    }



    /// Execute count-rows command
    async fn execute_count_rows(&self, table_name: String, db_path: Option<PathBuf>) -> Result<()> {
        use crate::tasks::DatabaseQueryEngine;
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🔢 Counting rows in table: {}", table_name.bright_cyan());
        println!();
        
        // Get database connection
        let database = if let Some(path) = db_path {
            crate::database::Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            crate::database::Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create progress bar
        let progress = ProgressBar::new_spinner();
        progress.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        
        progress.set_message("Connecting to database...");
        
        // Create database query engine
        let query_engine = DatabaseQueryEngine::new(database.pool().clone().into());
        
        progress.set_message("Validating table...");
        
        // Validate table exists
        let validation = query_engine.validate_table(&table_name).await?;
        if !validation.exists {
            progress.finish_and_clear();
            anyhow::bail!("Table '{}' does not exist in the database", table_name);
        }
        
        progress.set_message("Counting rows...");
        
        // Count rows
        let row_count = query_engine.count_rows(&table_name).await?;
        
        progress.finish_and_clear();
        
        // Display results
        println!("✅ Row count completed!");
        println!();
        println!("📊 Results:");
        println!("   Table: {}", table_name.bright_cyan());
        println!("   Row Count: {}", row_count.to_string().bright_green());
        
        if !validation.columns.is_empty() {
            println!("   Columns: {}", validation.columns.len());
            println!("   Schema: {}", validation.columns.join(", "));
        }
        
        println!();
        println!("🎯 Next Steps:");
        if row_count > 0 {
            println!("   1. Extract content: code-ingest extract-content {}", table_name);
            println!("   2. Generate tasks: code-ingest generate-hierarchical-tasks {} --output {}_tasks.md", table_name, table_name);
            println!("   3. Sample data: code-ingest sample --table {}", table_name);
        } else {
            println!("   • Table is empty - no content to process");
        }
        
        Ok(())
    }

    /// Execute extract-content command
    async fn execute_extract_content(&self, table_name: String, output_dir: PathBuf, chunk_size: Option<usize>, db_path: Option<PathBuf>) -> Result<()> {
        use crate::tasks::ContentExtractor;
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("📁 Extracting content from table: {}", table_name.bright_cyan());
        println!("📂 Output directory: {}", output_dir.display().to_string().bright_yellow());
        println!();
        
        // Get database connection
        let database = if let Some(path) = db_path {
            crate::database::Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            crate::database::Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create progress bar
        let progress = ProgressBar::new(100);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        progress.set_message("Initializing content extractor...");
        progress.set_position(10);
        
        // Create content extractor
        let extractor = ContentExtractor::new(database.pool().clone().into(), output_dir.clone());
        
        progress.set_message("Validating table and counting rows...");
        progress.set_position(20);
        
        // Extract all content
        let content_triples = extractor.extract_all_rows(&table_name, chunk_size).await?;
        
        progress.set_message("Content extraction complete!");
        progress.set_position(100);
        progress.finish_with_message("✅ Content extraction completed!");
        
        // Display results
        println!();
        println!("📊 Extraction Summary:");
        println!("   Table: {}", table_name);
        println!("   Files Created: {}", (content_triples.len() * 3).to_string().bright_green());
        println!("   Content Triples: {}", content_triples.len());
        println!("   Output Directory: {}", output_dir.display());
        
        println!();
        println!("📁 Generated Files:");
        for (i, triple) in content_triples.iter().take(5).enumerate() {
            println!("   Row {}: ", i + 1);
            println!("     A: {}", triple.content_a.display());
            println!("     B: {}", triple.content_b.display());
            println!("     C: {}", triple.content_c.display());
        }
        
        if content_triples.len() > 5 {
            println!("   ... and {} more content triples", content_triples.len() - 5);
        }
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Generate hierarchical tasks:");
        println!("      code-ingest generate-hierarchical-tasks {} --output {}_tasks.md", table_name, table_name);
        println!("   2. Review generated content files in: {}", output_dir.display());
        
        Ok(())
    }

    /// Execute configuration management commands
    async fn execute_config_command(&self, action: &ConfigAction) -> Result<()> {
        let config_manager = if let Some(config_path) = &self.config {
            ConfigManager::with_path(config_path.clone())
        } else {
            ConfigManager::new()
        };
        
        match action {
            ConfigAction::Show => {
                let config = config_manager.load_config()?;
                let toml_content = toml::to_string_pretty(&config)?;
                
                println!("📋 Current Configuration:");
                println!("   File: {}", config_manager.config_path().display());
                println!("   Exists: {}", if config_manager.config_exists() { "✅" } else { "❌" });
                println!();
                println!("{}", toml_content);
            }
            
            ConfigAction::Init { force } => {
                if config_manager.config_exists() && !force {
                    anyhow::bail!("Configuration file already exists: {}. Use --force to overwrite.", 
                                 config_manager.config_path().display());
                }
                
                config_manager.create_default_config()?;
                println!("✅ Created default configuration file: {}", config_manager.config_path().display());
                println!();
                println!("📝 Edit the file to customize your settings:");
                println!("   Database path, task generation defaults, etc.");
            }
            
            ConfigAction::Set { key, value } => {
                let mut config = config_manager.load_config()?;
                
                // Simple key-value setting for common configurations
                match key.as_str() {
                    "database.default_path" => {
                        let path = PathBuf::from(value);
                        if let Some(ref mut db_config) = config.database {
                            db_config.default_path = Some(path.clone());
                        } else {
                            config.database = Some(crate::config::DatabaseConfig {
                                default_path: Some(path.clone()),
                                timeout_seconds: Some(30),
                            });
                        }
                        println!("✅ Set database.default_path = {}", path.display());
                    }
                    "task_generation.default_levels" => {
                        let levels: usize = value.parse()?;
                        if !(1..=8).contains(&levels) {
                            anyhow::bail!("Levels must be between 1 and 8, got: {}", levels);
                        }
                        if let Some(ref mut task_config) = config.task_generation {
                            task_config.default_levels = Some(levels);
                        } else {
                            config.task_generation = Some(crate::config::TaskGenerationConfig {
                                default_levels: Some(levels),
                                ..Default::default()
                            });
                        }
                        println!("✅ Set task_generation.default_levels = {}", levels);
                    }
                    "task_generation.default_groups" => {
                        let groups: usize = value.parse()?;
                        if !(1..=20).contains(&groups) {
                            anyhow::bail!("Groups must be between 1 and 20, got: {}", groups);
                        }
                        if let Some(ref mut task_config) = config.task_generation {
                            task_config.default_groups = Some(groups);
                        } else {
                            config.task_generation = Some(crate::config::TaskGenerationConfig {
                                default_groups: Some(groups),
                                ..Default::default()
                            });
                        }
                        println!("✅ Set task_generation.default_groups = {}", groups);
                    }
                    "task_generation.default_prompt_file" => {
                        let path = PathBuf::from(value);
                        if let Some(ref mut task_config) = config.task_generation {
                            task_config.default_prompt_file = Some(path.clone());
                        } else {
                            config.task_generation = Some(crate::config::TaskGenerationConfig {
                                default_prompt_file: Some(path.clone()),
                                ..Default::default()
                            });
                        }
                        println!("✅ Set task_generation.default_prompt_file = {}", path.display());
                    }
                    _ => {
                        anyhow::bail!("Unknown configuration key: {}. Supported keys: database.default_path, task_generation.default_levels, task_generation.default_groups, task_generation.default_prompt_file", key);
                    }
                }
                
                config_manager.save_config(&config)?;
                println!("💾 Configuration saved to: {}", config_manager.config_path().display());
            }
            
            ConfigAction::Get { key } => {
                let config = config_manager.load_config()?;
                
                match key.as_str() {
                    "database.default_path" => {
                        if let Some(path) = config.database.as_ref().and_then(|db| db.default_path.as_ref()) {
                            println!("{}", path.display());
                        } else {
                            println!("(not set)");
                        }
                    }
                    "task_generation.default_levels" => {
                        if let Some(levels) = config.task_generation.as_ref().and_then(|tg| tg.default_levels) {
                            println!("{}", levels);
                        } else {
                            println!("4"); // Default value
                        }
                    }
                    "task_generation.default_groups" => {
                        if let Some(groups) = config.task_generation.as_ref().and_then(|tg| tg.default_groups) {
                            println!("{}", groups);
                        } else {
                            println!("7"); // Default value
                        }
                    }
                    "task_generation.default_prompt_file" => {
                        if let Some(path) = config.task_generation.as_ref().and_then(|tg| tg.default_prompt_file.as_ref()) {
                            println!("{}", path.display());
                        } else {
                            println!(".kiro/steering/spec-S04-steering-doc-analysis.md"); // Default value
                        }
                    }
                    _ => {
                        anyhow::bail!("Unknown configuration key: {}", key);
                    }
                }
            }
            
            ConfigAction::Path => {
                println!("{}", config_manager.config_path().display());
            }
        }
        
        Ok(())
    }

    /// Validate parameters for generate-hierarchical-tasks command
    fn validate_hierarchical_task_parameters(
        &self,
        table_name: &str,
        levels: usize,
        groups: usize,
        output: &PathBuf,
        prompt_file: &PathBuf,
        chunks: Option<usize>,
    ) -> Result<()> {
        // Validate table name format
        if table_name.is_empty() {
            anyhow::bail!("Table name cannot be empty");
        }
        
        if table_name.contains(' ') {
            anyhow::bail!("Table name cannot contain spaces: '{}'", table_name);
        }
        
        // Validate levels (already validated by clap, but double-check)
        if !(1..=8).contains(&levels) {
            anyhow::bail!("Levels must be between 1 and 8, got: {}", levels);
        }
        
        // Validate groups (already validated by clap, but double-check)
        if !(1..=20).contains(&groups) {
            anyhow::bail!("Groups must be between 1 and 20, got: {}", groups);
        }
        
        // Validate output file path
        if let Some(parent) = output.parent() {
            if !parent.exists() {
                println!("ℹ️  Output directory will be created: {}", parent.display());
            }
        }
        
        // Check if output file already exists
        if output.exists() {
            println!("⚠️  Warning: Output file already exists and will be overwritten: {}", output.display());
        }
        
        // Validate output file extension
        if let Some(extension) = output.extension() {
            if extension != "md" {
                println!("ℹ️  Note: Output file does not have .md extension: {}", output.display());
            }
        } else {
            println!("ℹ️  Note: Output file has no extension, consider using .md: {}", output.display());
        }
        
        // Validate prompt file exists and is readable
        if !prompt_file.exists() {
            anyhow::bail!("Prompt file not found: {}", prompt_file.display());
        }
        
        if !prompt_file.is_file() {
            anyhow::bail!("Prompt file path is not a file: {}", prompt_file.display());
        }
        
        // Try to read the prompt file to ensure it's accessible
        match std::fs::metadata(&prompt_file) {
            Ok(metadata) => {
                if metadata.len() == 0 {
                    println!("⚠️  Warning: Prompt file is empty: {}", prompt_file.display());
                }
            }
            Err(e) => {
                anyhow::bail!("Cannot access prompt file {}: {}", prompt_file.display(), e);
            }
        }
        
        // Validate chunks parameter if provided
        if let Some(chunk_size) = chunks {
            if !(50..=10000).contains(&chunk_size) {
                anyhow::bail!("Chunk size must be between 50 and 10000 LOC, got: {}", chunk_size);
            }
            
            println!("🔄 Chunking mode enabled: {} LOC per chunk", chunk_size);
        }
        
        Ok(())
    }

    /// Execute generate-hierarchical-tasks command
    async fn execute_generate_hierarchical_tasks(
        &self,
        table_name: String,
        levels: usize,
        groups: usize,
        output: PathBuf,
        prompt_file: PathBuf,
        chunks: Option<usize>,
        max_tasks: usize,
        windowed: bool,
        db_path: Option<PathBuf>,
    ) -> Result<()> {
        // Load configuration and merge with CLI arguments
        let config = self.load_config()?;
        let merged_config = merge_config_with_cli(
            &config,
            db_path.clone().or_else(|| self.db_path.clone()),
            Some(levels),
            Some(groups),
            Some(prompt_file.clone()),
            chunks,
        );
        
        // Validate parameters using merged configuration
        self.validate_hierarchical_task_parameters(&table_name, merged_config.levels, merged_config.groups, &output, &merged_config.prompt_file, merged_config.chunks)?;
        use crate::tasks::{DatabaseQueryEngine, ContentExtractor, HierarchicalTaskDivider, SimpleTaskGenerator};
        use indicatif::{ProgressBar, ProgressStyle};
        
        println!("🏗️  Generating hierarchical tasks from table: {}", table_name.bright_cyan());
        println!("📊 Configuration:");
        println!("   Levels: {}", merged_config.levels);
        println!("   Groups per level: {}", merged_config.groups);
        println!("   Output file: {}", output.display().to_string().bright_yellow());
        println!("   Prompt file: {}", merged_config.prompt_file.display().to_string().bright_blue());
        if let Some(chunk_size) = merged_config.chunks {
            println!("   Chunk size: {} LOC", chunk_size.to_string().bright_green());
        }
        println!("   Database timeout: {}s", merged_config.timeout_seconds);
        println!();
        
        // Validate prompt file exists (using merged config)
        if !merged_config.prompt_file.exists() {
            anyhow::bail!("Prompt file not found: {}", merged_config.prompt_file.display());
        }
        
        // Get database connection using merged configuration
        let database = if let Some(path) = merged_config.db_path {
            crate::database::Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            crate::database::Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path, set in config file, or set DATABASE_URL environment variable");
        };

        // Create progress bar
        let progress = ProgressBar::new(100);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Step 1: Count rows and validate table
        progress.set_message("Validating table and counting rows...");
        progress.set_position(10);
        
        let query_engine = DatabaseQueryEngine::new(database.pool().clone().into());
        let row_count = query_engine.count_rows(&table_name).await?;
        
        if row_count == 0 {
            anyhow::bail!("Table '{}' is empty. Cannot generate tasks from empty table.", table_name);
        }

        // Handle chunking if requested
        let working_table_name = if let Some(chunk_size) = merged_config.chunks {
            println!("🔄 Chunking enabled with size: {} LOC", chunk_size);
            
            // Step 1.5: Create chunked table
            progress.set_message("Creating chunked table...");
            progress.set_position(20);
            
            use crate::processing::ChunkDatabaseManager;
            let chunk_manager = ChunkDatabaseManager::with_pool(database.pool().clone());
            let chunk_result = chunk_manager.process_table_with_chunking(&table_name, chunk_size).await?;
            
            println!("✅ Chunking completed:");
            println!("   Chunked table: {}", chunk_result.chunked_table_name.bright_cyan());
            println!("   Base rows processed: {}", chunk_result.base_rows_processed);
            println!("   Chunks created: {}", chunk_result.chunks_inserted);
            println!("   Files chunked: {}", chunk_result.files_chunked);
            println!("   Files below threshold: {}", chunk_result.files_below_threshold);
            println!();
            
            chunk_result.chunked_table_name
        } else {
            table_name.clone()
        };
        
        // Step 2: Extract content to A/B/C files
        progress.set_message("Extracting content to A/B/C files...");
        progress.set_position(30);
        
        let output_dir = std::path::PathBuf::from(".raw_data_202509");
        let extractor = ContentExtractor::new(database.pool().clone().into(), output_dir);
        let content_triples = extractor.extract_all_rows(&working_table_name, None).await?;
        
        // Step 3: Create hierarchical task structure
        progress.set_message("Creating hierarchical task structure...");
        progress.set_position(60);
        
        let task_divider = HierarchicalTaskDivider::new(merged_config.levels, merged_config.groups)?;
        let hierarchy = task_divider.create_hierarchy(content_triples.clone())?;
        
        // Step 4: Generate markdown or windowed system
        progress.set_message("Generating task system...");
        progress.set_position(80);
        
        if windowed && hierarchy.total_tasks > max_tasks {
            // Create windowed task system
            use crate::tasks::{WindowedTaskManager, WindowedTaskConfig};
            
            let task_dir = if output.is_dir() {
                output.clone()
            } else {
                output.parent().unwrap_or_else(|| std::path::Path::new(".")).join(format!("{}_windowed", working_table_name))
            };
            
            let config = WindowedTaskConfig {
                task_dir: task_dir.clone(),
                window_size: max_tasks,
                table_name: working_table_name.clone(),
            };
            
            let mut windowed_manager = WindowedTaskManager::new(config, hierarchy.total_tasks).await?;
            windowed_manager.generate_from_hierarchy(&hierarchy).await?;
            
            progress.set_message("Windowed system created!");
            progress.set_position(100);
            progress.finish_with_message("✅ Windowed task system generated!");
            
            // Display windowed results
            println!();
            println!("📊 Windowed System Summary:");
            println!("   Table: {}", table_name);
            println!("   Total Tasks: {}", hierarchy.total_tasks);
            println!("   Window Size: {}", max_tasks);
            println!("   Total Windows: {}", (hierarchy.total_tasks + max_tasks - 1) / max_tasks);
            println!("   Task Directory: {}", task_dir.display());
            println!();
            println!("🎯 Next Steps:");
            println!("   1. Work on current window: kiro {}/current-window.md", task_dir.display());
            println!("   2. When done, advance window: code-ingest advance-window {}", task_dir.display());
            println!("   3. Check progress: code-ingest task-progress {}", task_dir.display());
            
        } else {
            // Create regular single-file system
            let markdown_generator = SimpleTaskGenerator::with_max_tasks(max_tasks);
            let markdown_content = markdown_generator.generate_simple_tasks(&working_table_name, chunks, &prompt_file.to_string_lossy(), hierarchy.total_tasks).await?;
            
            // Step 5: Write output file
            progress.set_message("Writing output file...");
            progress.set_position(90);
            
            // Ensure parent directory exists
            if let Some(parent) = output.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            
            tokio::fs::write(&output, markdown_content).await?;
            
            progress.set_message("Complete!");
            progress.set_position(100);
            progress.finish_with_message("✅ Hierarchical tasks generated!");
            
            // Display regular results
            println!();
            println!("📊 Generation Summary:");
            println!("   Table: {}", table_name);
            println!("   Total Tasks: {}", hierarchy.total_tasks);
            println!("   Tasks Generated: {}", std::cmp::min(hierarchy.total_tasks, max_tasks));
            println!("   Output File: {}", output.display());
            
            if hierarchy.total_tasks > max_tasks {
                println!("   ⚠️  Note: Limited to {} tasks (use --windowed for complete coverage)", max_tasks);
            }
        }
        
        progress.set_message("Complete!");
        progress.set_position(100);
        progress.finish_with_message("✅ Hierarchical tasks generated!");
        
        // Display results
        println!();
        println!("📊 Generation Summary:");
        println!("   Table: {}", table_name);
        println!("   Total Tasks: {}", hierarchy.total_tasks);
        println!("   Hierarchy Levels: {}", merged_config.levels);
        println!("   Groups per Level: {}", merged_config.groups);
        println!("   Content Files: {}", content_triples.len() * 3);
        println!("   Output File: {}", output.display());
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Open the tasks file: {}", output.display());
        println!("   2. Review the hierarchical task structure");
        println!("   3. Execute tasks systematically using Kiro");
        println!("   4. Content files are available in: .raw_data_202509/");
        println!("   5. Analysis outputs will be saved to: gringotts/WorkArea/");
        
        Ok(())
    }

    /// Execute simple task generation (experimental - simplified approach)
    /// 
    /// # Preconditions
    /// - table_name exists in database
    /// - output path is writable
    /// - prompt_file exists and is readable
    /// 
    /// # Postconditions
    /// - Creates simple checkbox markdown file
    /// - Each task follows format: "- [ ] N. Analyze TABLE row N"
    /// - Includes Content/Prompt/Output paths
    /// 
    /// # Error Conditions
    /// - DatabaseError if table doesn't exist
    /// - IoError if output path not writable
    /// - ValidationError if max_tasks > 1000
    #[instrument(skip(self))]
    async fn execute_generate_simple_tasks(
        &self,
        table_name: String,
        output: PathBuf,
        prompt_file: PathBuf,
        chunks: Option<usize>,
        max_tasks: usize,
        db_path: Option<PathBuf>,
    ) -> anyhow::Result<()> {
        use crate::database::Database;
        use crate::tasks::simple_task_generator::SimpleTaskGenerator;
        
        println!("🚀 [EXPERIMENTAL] Generating simple task list for table: {}", table_name);
        println!("📋 Simplified task generation following MVP-First Rigor pattern");
        println!();

        // Validate inputs (following steering principle: Parse, don't validate)
        if max_tasks > 1000 {
            anyhow::bail!("max_tasks cannot exceed 1000 (got {}). This prevents Kiro IDE overload.", max_tasks);
        }
        
        if !prompt_file.exists() {
            anyhow::bail!("Prompt file does not exist: {}", prompt_file.display());
        }

        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Count rows in the table with timeout (following steering: timeout all external calls)
        let row_count_query = format!("SELECT COUNT(*) FROM \"{}\"", table_name);
        let pool = database.pool();
        
        let start = std::time::Instant::now();
        let row: (i64,) = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            sqlx::query_as(&row_count_query).fetch_one(pool)
        )
        .await
        .with_context(|| format!("Query timeout after 30s for table {}", table_name))?
        .with_context(|| format!("Failed to count rows in table {}", table_name))?;
        
        let query_time = start.elapsed();
        let total_rows = row.0 as usize;
        
        println!("📊 Found {} rows in table {} (query: {:?})", total_rows, table_name, query_time);
        
        // Performance contract validation (following steering: performance claims must be test-validated)
        if query_time > std::time::Duration::from_secs(5) {
            println!("⚠️  Warning: Row count query took {:?} (>5s). Consider table optimization.", query_time);
        }

        // Create simple task generator with max_tasks limit
        let generator = SimpleTaskGenerator::with_max_tasks(max_tasks);
        
        // Generate simple tasks with performance tracking
        let prompt_file_str = prompt_file.to_string_lossy();
        
        println!("📝 Generating tasks...");
        let generation_start = std::time::Instant::now();
        
        generator.write_simple_tasks_to_file(
            &table_name,
            chunks,
            &prompt_file_str,
            total_rows,
            &output,
        ).await
        .with_context(|| format!("Failed to write tasks to {}", output.display()))?;
        
        let generation_time = generation_start.elapsed();
        
        // Performance contract: Task generation should complete in <1s for typical workloads
        if generation_time > std::time::Duration::from_secs(1) {
            println!("⚠️  Warning: Task generation took {:?} (>1s). Consider reducing max_tasks.", generation_time);
        }

        println!("✅ Simple task generation completed!");
        println!();
        println!("📋 Task Details:");
        println!("   Table: {}", table_name);
        println!("   Total Rows: {}", total_rows);
        println!("   Max Tasks: {}", max_tasks);
        if let Some(chunk_size) = chunks {
            println!("   Chunk Size: {} LOC", chunk_size);
        }
        println!("   Prompt File: {}", prompt_file.display());
        println!("   Output File: {}", output.display());
        
        println!();
        println!("🎯 Next Steps:");
        println!("   1. Open the tasks file: {}", output.display());
        println!("   2. Each task follows the simple format:");
        println!("      - [ ] N. Analyze {} row N", table_name);
        println!("      - **Content**: A/B/C files in .raw_data_202509/");
        println!("      - **Prompt**: {}", prompt_file.display());
        println!("      - **Output**: gringotts/WorkArea/");
        println!("   3. Execute tasks one by one using Kiro");
        
        Ok(())
    }

    /// Execute chunk-level task generator command
    /// 
    /// This method implements the simplified chunk-level task generation that replaces
    /// the complex existing task generation system with two modes:
    /// - File-level mode (no chunk size): generates content files for each database row
    /// - Chunk-level mode (with chunk size): processes large files with chunking
    /// 
    /// # Requirements
    /// Satisfies requirements 1.1 and 2.1 by providing CLI integration for the
    /// ChunkLevelTaskGenerator with proper argument parsing and user feedback.
    #[instrument(skip(self))]
    async fn execute_chunk_level_task_generator(
        &self,
        table_name: String,
        chunk_size: Option<usize>,
        db_path: Option<PathBuf>,
        output_dir: PathBuf,
    ) -> Result<()> {
        use crate::tasks::chunk_level_task_generator::ChunkLevelTaskGenerator;
        use crate::tasks::database_service::DatabaseService;
        use crate::tasks::content_file_writer::ContentFileWriter;
        use crate::tasks::task_list_generator::TaskListGenerator;
        use crate::tasks::chunking_service::ChunkingService;
        use crate::database::Database;
        use indicatif::{ProgressBar, ProgressStyle};
        use std::sync::Arc;

        // Display command information
        println!("🚀 {}", "Chunk-Level Task Generator".bright_green().bold());
        println!("📋 Table: {}", table_name.bright_cyan());
        if let Some(chunk_size) = chunk_size {
            println!("📏 Chunk Size: {} lines (chunk-level mode)", chunk_size.to_string().bright_yellow());
        } else {
            println!("📄 Mode: File-level (no chunking)");
        }
        println!("📂 Output Directory: {}", output_dir.display().to_string().bright_yellow());
        println!();

        // Get database connection
        info!("Connecting to database");
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await
                .with_context(|| format!("Failed to connect to database at path: {}", path.display()))?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await
                .with_context(|| "Failed to connect to database using DATABASE_URL")?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create progress bar
        let progress = ProgressBar::new(100);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        progress.set_message("Initializing services...");
        progress.set_position(10);

        // Initialize all required services
        let database_service = Arc::new(DatabaseService::new(database.pool().clone()));
        let content_writer = ContentFileWriter::new(output_dir.clone());
        let task_generator = TaskListGenerator::new();
        let chunking_service = ChunkingService::new(database_service.clone());

        progress.set_message("Creating task generator...");
        progress.set_position(20);

        // Create the chunk-level task generator
        let generator = ChunkLevelTaskGenerator::new(
            database_service,
            content_writer,
            task_generator,
            chunking_service,
        );

        progress.set_message("Validating table and inputs...");
        progress.set_position(30);

        // Execute the task generation
        info!("Starting task generation for table '{}' with chunk_size {:?}", table_name, chunk_size);
        let result = generator.execute(&table_name, chunk_size, None).await
            .with_context(|| format!("Failed to execute chunk-level task generation for table '{}'", table_name))?;

        progress.set_message("Task generation complete!");
        progress.set_position(100);
        progress.finish_with_message("✅ Task generation completed!");

        // Display results
        println!();
        println!("📊 {}", "Generation Summary:".bright_green().bold());
        println!("   Table Used: {}", result.table_used.bright_cyan());
        println!("   Rows Processed: {}", result.rows_processed.to_string().bright_green());
        println!("   Content Files Created: {}", result.content_files_created.to_string().bright_green());
        println!("   Task List: {}", result.task_list_path.display().to_string().bright_yellow());
        
        if let Some(chunked_table) = &result.chunked_table_created {
            println!("   Chunked Table Created: {}", chunked_table.bright_magenta());
        }

        // Display processing statistics
        let stats = &result.processing_stats;
        println!();
        println!("⚡ {}", "Processing Statistics:".bright_blue().bold());
        println!("   Processing Time: {}ms", stats.processing_time_ms.to_string().bright_yellow());
        println!("   Files Chunked: {}", stats.files_chunked.to_string().bright_green());
        println!("   Files Copied: {}", stats.files_copied.to_string().bright_green());
        println!("   Total Files Processed: {}", stats.total_files_processed().to_string().bright_cyan());
        
        if stats.total_chunks_created > 0 {
            println!("   Total Chunks Created: {}", stats.total_chunks_created.to_string().bright_magenta());
            println!("   Average Chunk Size: {:.1} lines", stats.average_chunk_size.to_string().bright_yellow());
        }
        
        if stats.processing_time_ms > 0 {
            println!("   Processing Rate: {:.1} files/sec", stats.processing_rate_fps().to_string().bright_green());
        }

        // Display next steps
        println!();
        println!("🎯 {}", "Next Steps:".bright_green().bold());
        println!("   1. Review generated content files in: {}", output_dir.display().to_string().bright_yellow());
        println!("   2. Open task list: {}", result.task_list_path.display().to_string().bright_yellow());
        println!("   3. Execute tasks using your preferred method");
        
        if result.used_chunking() {
            println!("   4. Chunked table '{}' is available for further analysis", 
                    result.chunked_table_created.as_ref().unwrap().bright_magenta());
        }

        // Cleanup resources
        generator.cleanup().await
            .with_context(|| "Failed to cleanup resources after task generation")?;

        info!("Chunk-level task generation completed successfully");
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_parsing_ingest_command() {
        let args = vec![
            "code-ingest",
            "ingest",
            "https://github.com/test/repo",
            "--db-path",
            "/tmp/test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Ingest { source, folder_flag: _, db_path }) => {
                assert_eq!(source, "https://github.com/test/repo");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected Ingest command"),
        }
    }

    #[test]
    fn test_cli_parsing_sql_command() {
        let args = vec![
            "code-ingest",
            "sql",
            "SELECT * FROM test",
            "--db-path",
            "/tmp/test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Sql { query, db_path, limit, offset, timeout, llm_format }) => {
                assert_eq!(query, "SELECT * FROM test");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(*limit, 0); // Default value
                assert_eq!(*offset, 0); // Default value
                assert_eq!(*timeout, 30); // Default value
                assert!(!llm_format); // Default value
            }
            _ => panic!("Expected Sql command"),
        }
    }

    #[test]
    fn test_cli_parsing_query_prepare_command() {
        let args = vec![
            "code-ingest",
            "query-prepare",
            "SELECT * FROM test",
            "--db-path",
            "/tmp/test",
            "--temp-path",
            "/tmp/temp.txt",
            "--tasks-file",
            "/tmp/tasks.md",
            "--output-table",
            "QUERYRESULT_test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::QueryPrepare {
                query,
                db_path,
                temp_path,
                tasks_file,
                output_table,
            }) => {
                assert_eq!(query, "SELECT * FROM test");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(temp_path.to_str().unwrap(), "/tmp/temp.txt");
                assert_eq!(tasks_file.to_str().unwrap(), "/tmp/tasks.md");
                assert_eq!(output_table, "QUERYRESULT_test");
            }
            _ => panic!("Expected QueryPrepare command"),
        }
    }

    #[test]
    fn test_cli_parsing_generate_tasks_command() {
        let args = vec![
            "code-ingest",
            "generate-tasks",
            "--sql",
            "SELECT * FROM test",
            "--prompt-file",
            "/tmp/prompt.md",
            "--output-table",
            "QUERYRESULT_test",
            "--tasks-file",
            "/tmp/tasks.md",
            "--db-path",
            "/tmp/test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::GenerateTasks {
                sql,
                prompt_file,
                output_table,
                tasks_file,
                db_path,
            }) => {
                assert_eq!(sql, "SELECT * FROM test");
                assert_eq!(prompt_file.to_str().unwrap(), "/tmp/prompt.md");
                assert_eq!(output_table, "QUERYRESULT_test");
                assert_eq!(tasks_file.to_str().unwrap(), "/tmp/tasks.md");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected GenerateTasks command"),
        }
    }

    #[test]
    fn test_cli_parsing_print_to_md_command() {
        let args = vec![
            "code-ingest",
            "print-to-md",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
            "--sql",
            "SELECT * FROM test_table",
            "--prefix",
            "analysis",
            "--location",
            "/tmp/md-files",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::PrintToMd {
                db_path,
                table,
                sql,
                prefix,
                location,
            }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(table, "test_table");
                assert_eq!(sql, "SELECT * FROM test_table");
                assert_eq!(prefix, "analysis");
                assert_eq!(location.to_str().unwrap(), "/tmp/md-files");
            }
            _ => panic!("Expected PrintToMd command"),
        }
    }

    #[test]
    fn test_cli_parsing_list_tables_command() {
        let args = vec!["code-ingest", "list-tables", "--db-path", "/tmp/test"];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::ListTables { db_path }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected ListTables command"),
        }
    }

    #[test]
    fn test_cli_parsing_sample_command() {
        let args = vec![
            "code-ingest",
            "sample",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
            "--limit",
            "10",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Sample {
                db_path,
                table,
                limit,
            }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(table, "test_table");
                assert_eq!(*limit, 10);
            }
            _ => panic!("Expected Sample command"),
        }
    }

    #[test]
    fn test_cli_parsing_describe_command() {
        let args = vec![
            "code-ingest",
            "describe",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Describe { db_path, table }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(table, "test_table");
            }
            _ => panic!("Expected Describe command"),
        }
    }

    #[test]
    fn test_cli_parsing_db_info_command() {
        let args = vec!["code-ingest", "db-info", "--db-path", "/tmp/test"];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::DbInfo { db_path }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected DbInfo command"),
        }
    }

    #[test]
    fn test_cli_parsing_pg_start_command() {
        let args = vec!["code-ingest", "pg-start"];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::PgStart) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected PgStart command"),
        }
    }

    #[test]
    fn test_cli_parsing_global_db_path() {
        let args = vec![
            "code-ingest",
            "--db-path",
            "/global/path",
            "list-tables",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        // Global db_path should be set
        assert_eq!(cli.db_path.as_ref().unwrap().to_str().unwrap(), "/global/path");
        
        // Command should be parsed correctly
        match &cli.command {
            Some(Commands::ListTables { .. }) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected ListTables command"),
        }
    }

    #[test]
    fn test_cli_parsing_store_result_command() {
        let args = vec![
            "code-ingest",
            "store-result",
            "--db-path",
            "/tmp/test",
            "--output-table",
            "QUERYRESULT_test",
            "--result-file",
            "/tmp/result.txt",
            "--original-query",
            "SELECT * FROM test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::StoreResult {
                db_path,
                output_table,
                result_file,
                original_query,
            }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(output_table, "QUERYRESULT_test");
                assert_eq!(result_file.to_str().unwrap(), "/tmp/result.txt");
                assert_eq!(original_query, "SELECT * FROM test");
            }
            _ => panic!("Expected StoreResult command"),
        }
    }

    #[test]
    fn test_cli_validation_missing_required_args() {
        // Test that missing required arguments cause parsing to fail
        let args = vec!["code-ingest", "ingest"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err()); // Missing SOURCE argument

        let args = vec!["code-ingest", "sql"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err()); // Missing QUERY argument

        // Note: --db-path is optional since it can be set globally or via environment
        // So we test missing required positional arguments instead
        let args = vec!["code-ingest", "sample"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err()); // Missing --table argument
    }

    #[test]
    fn test_cli_parsing_chunk_level_task_generator() {
        // Test chunk-level task generator command - file-level mode
        let args = vec![
            "code-ingest",
            "chunk-level-task-generator",
            "INGEST_20250928101039",
            "--db-path",
            "/tmp/test.db",
            "--output-dir",
            "./output",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::ChunkLevelTaskGenerator { 
                table_name, 
                chunk_size, 
                db_path, 
                output_dir 
            }) => {
                assert_eq!(table_name, "INGEST_20250928101039");
                assert_eq!(*chunk_size, None);
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test.db");
                assert_eq!(output_dir.to_str().unwrap(), "./output");
            }
            _ => panic!("Expected ChunkLevelTaskGenerator command"),
        }

        // Test chunk-level task generator command - chunk-level mode
        let args = vec![
            "code-ingest",
            "chunk-level-task-generator",
            "INGEST_20250928101039",
            "500",
            "--db-path",
            "/tmp/test.db",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::ChunkLevelTaskGenerator { 
                table_name, 
                chunk_size, 
                db_path, 
                output_dir 
            }) => {
                assert_eq!(table_name, "INGEST_20250928101039");
                assert_eq!(*chunk_size, Some(500));
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test.db");
                assert_eq!(output_dir.to_str().unwrap(), ".");
            }
            _ => panic!("Expected ChunkLevelTaskGenerator command"),
        }
    }

    #[test]
    fn test_cli_parsing_help_commands() {
        // Test setup command
        let cli = Cli::try_parse_from(&["code-ingest", "setup"]).unwrap();
        match &cli.command {
            Some(Commands::Setup) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected Setup command"),
        }

        // Test examples command
        let cli = Cli::try_parse_from(&["code-ingest", "examples"]).unwrap();
        match &cli.command {
            Some(Commands::Examples) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected Examples command"),
        }

        // Test troubleshoot command
        let cli = Cli::try_parse_from(&["code-ingest", "troubleshoot"]).unwrap();
        match &cli.command {
            Some(Commands::Troubleshoot) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected Troubleshoot command"),
        }
    }

    #[test]
    fn test_sample_command_default_limit() {
        let args = vec![
            "code-ingest",
            "sample",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Sample { limit, .. }) => {
                assert_eq!(*limit, 5); // Default value
            }
            _ => panic!("Expected Sample command"),
        }
    }
}