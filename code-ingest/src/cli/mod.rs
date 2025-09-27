use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "code-ingest")]
#[command(about = "A Rust-based code ingestion system for PostgreSQL")]
#[command(version = "0.1.0")]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Database path for PostgreSQL connection
    #[arg(long, global = true)]
    db_path: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Ingest a GitHub repository or local folder
    Ingest {
        /// GitHub repository URL or local folder path
        source: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Execute SQL query and output results
    Sql {
        /// SQL query to execute
        query: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
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

    pub async fn run(&self) -> Result<()> {
        match &self.command {
            Some(command) => self.execute_command(command).await,
            None => {
                // If no subcommand is provided, try to parse as ingest command
                // This handles the case: code-ingest <url> --db-path <path>
                let args: Vec<String> = std::env::args().collect();
                if args.len() >= 2 && !args[1].starts_with('-') {
                    let source = args[1].clone();
                    let db_path = self.db_path.clone();
                    self.execute_ingest(source, db_path).await
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
            Commands::Ingest { source, db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_ingest(source.clone(), db_path).await
            }
            Commands::Sql { query, db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_sql(query.clone(), db_path).await
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
        }
    }

    async fn execute_ingest(&self, source: String, db_path: Option<PathBuf>) -> Result<()> {
        use crate::database::{Database, SchemaManager};
        use crate::ingestion::{IngestionEngine, IngestionConfig};
        use crate::processing::text_processor::TextProcessor;
        use crate::ingestion::batch_processor::BatchProgress;
        use indicatif::{ProgressBar, ProgressStyle};
        use std::sync::Arc;
        
        println!("🚀 Starting ingestion from: {}", source);
        println!();
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Initialize database schema
        println!("📋 Initializing database schema...");
        let schema_manager = SchemaManager::new(database.pool().clone());
        schema_manager.initialize_schema().await?;
        
        // Create progress bar
        let progress = ProgressBar::new_spinner();
        progress.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        
        // Create ingestion configuration
        let config = IngestionConfig::default();
        
        // Create file processor
        let file_processor = Arc::new(TextProcessor::new());
        
        // Create ingestion engine
        let engine = IngestionEngine::new(config, Arc::new(database), file_processor);
        
        // Start ingestion with progress callback
        progress.set_message("Starting ingestion...");
        
        let progress_callback = {
            let progress = progress.clone();
            Box::new(move |batch_progress: BatchProgress| {
                progress.set_message(format!(
                    "Processing files: {} processed, {} total", 
                    batch_progress.files_processed, 
                    batch_progress.total_files
                ));
            })
        };
        
        let result = engine.ingest_source(&source, Some(progress_callback)).await?;
        
        progress.finish_and_clear();
        
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
        println!("   1. Explore your data: cargo run -- list-tables");
        println!("   2. Sample the data: cargo run -- sample --table {}", result.table_name);
        println!("   3. Run queries: cargo run -- sql \"SELECT COUNT(*) FROM {}\"", result.table_name);
        println!("   4. Export files: cargo run -- print-to-md --table {} --sql \"SELECT * FROM {} LIMIT 10\" --prefix tauri --location ./exports", result.table_name, result.table_name);
        
        Ok(())
    }

    async fn execute_sql(&self, query: String, db_path: Option<PathBuf>) -> Result<()> {
        use crate::database::{Database, QueryExecutor};
        
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
        
        // Execute query and format for terminal
        let result = executor.execute_query_terminal(&query).await?;
        
        // Print results
        print!("{}", result.content);
        
        if result.truncated {
            println!("Note: Results were truncated");
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
        use crate::database::{Database, QueryExecutor, TempFileManager, TempFileConfig, TempFileMetadata};
        
        // Get database connection
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
        
        // Create metadata for the temporary file
        let metadata = TempFileMetadata {
            original_query: query.clone(),
            output_table: output_table.clone(),
            prompt_file_path: None, // Will be set when tasks are generated
            description: Some("Query results prepared for LLM analysis".to_string()),
        };
        
        // Create temporary file with query results
        let config = TempFileConfig::default();
        let temp_result = temp_manager
            .create_structured_temp_file(&query, &temp_path, &metadata, &config)
            .await?;
        
        println!("✅ Query executed successfully");
        println!("   Rows: {}", temp_result.row_count);
        println!("   Execution time: {}ms", temp_result.execution_time_ms);
        println!("   Temporary file: {} ({} bytes)", temp_path.display(), temp_result.bytes_written);
        
        // Create a basic task structure for IDE integration
        let task_content = self.create_basic_task_structure(&query, &temp_path, &output_table);
        
        // Write tasks file
        tokio::fs::write(&tasks_file, task_content).await?;
        println!("   Tasks file: {}", tasks_file.display());
        
        println!("\n📋 Next steps:");
        println!("   1. Open the tasks file in your IDE: {}", tasks_file.display());
        println!("   2. Execute the analysis tasks");
        println!("   3. Store results using: code-ingest store-result --output-table {} --result-file <result_file> --original-query \"{}\"", output_table, query);
        
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
        
        // Get database connection
        let database = if let Some(path) = db_path {
            Database::from_path(&path).await?
        } else if let Ok(database_url) = std::env::var("DATABASE_URL") {
            Database::new(&database_url).await?
        } else {
            anyhow::bail!("No database path provided. Use --db-path or set DATABASE_URL environment variable");
        };

        // Create result storage manager
        let storage = ResultStorage::new(database.pool().clone());
        
        // Create metadata for the result
        let metadata = ResultMetadata {
            original_query: original_query.clone(),
            prompt_file_path: None, // Could be enhanced to detect this
            analysis_type: Some("llm_analysis".to_string()),
            original_file_path: None,
            chunk_number: None,
            created_by: Some("code-ingest-cli".to_string()),
            tags: vec!["analysis".to_string()],
        };
        
        // Store the result
        let config = StorageConfig::default();
        let storage_result = storage
            .store_result_from_file(&output_table, &result_file, &metadata, &config)
            .await?;
        
        println!("✅ Analysis result stored successfully");
        println!("   Analysis ID: {}", storage_result.analysis_id);
        println!("   Table: {}", storage_result.table_name);
        println!("   Size: {} bytes", storage_result.result_size_bytes);
        println!("   Result file: {}", result_file.display());
        
        // Show storage statistics
        match storage.get_storage_stats(&output_table).await {
            Ok(stats) => {
                println!("\n📊 Table statistics:");
                println!("   Total results: {}", stats.total_results);
                println!("   Average size: {} bytes", stats.avg_result_size_bytes);
                if let Some(newest) = stats.newest_result {
                    println!("   Latest result: {}", newest.format("%Y-%m-%d %H:%M:%S UTC"));
                }
            }
            Err(e) => {
                println!("Note: Could not retrieve table statistics: {}", e);
            }
        }
        
        Ok(())
    }

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

    /// Create a basic task structure for IDE integration
    fn create_basic_task_structure(&self, query: &str, temp_path: &PathBuf, output_table: &str) -> String {
        format!(
            r#"# Analysis Tasks

## Query Information
- **Query**: `{}`
- **Data Source**: {}
- **Output Table**: {}
- **Generated**: {}

## Tasks

- [ ] 1. Review query results
  - Open and examine the temporary file: {}
  - Understand the data structure and content
  - Identify key patterns or areas of interest

- [ ] 2. Perform analysis
  - Apply your analysis methodology to the data
  - Document findings and insights
  - Prepare structured results

- [ ] 3. Store results
  - Save analysis results to a file
  - Use: `code-ingest store-result --output-table {} --result-file <your_result_file> --original-query "{}"`

## Notes

This is a basic task structure. For more sophisticated task generation with automatic division and detailed prompts, use the `generate-tasks` command.
"#,
            query,
            temp_path.display(),
            output_table,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            temp_path.display(),
            output_table,
            query
        )
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
            Some(Commands::Ingest { source, db_path }) => {
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
            Some(Commands::Sql { query, db_path }) => {
                assert_eq!(query, "SELECT * FROM test");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
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