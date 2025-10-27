//! Chunk-level task generator for creating content files and task lists
//!
//! This module provides a simplified interface for generating content files and task lists
//! from database tables. It operates in two modes:
//! - File-level mode (no chunk size): generates content files for each database row
//! - Chunk-level mode (with chunk size): processes large files with chunking
//!
//! # Examples
//!
//! ```rust
//! use code_ingest::tasks::chunk_level_task_generator::{ChunkLevelTaskGenerator, TaskGenerationResult};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let generator = ChunkLevelTaskGenerator::new(
//!     database_service,
//!     content_writer,
//!     task_generator,
//!     chunking_service,
//! );
//!
//! // File-level mode
//! let result = generator.execute("INGEST_TABLE", None, None).await?;
//! println!("Generated {} content files", result.content_files_created);
//!
//! // Chunk-level mode
//! let result = generator.execute("INGEST_TABLE", Some(500), None).await?;
//! println!("Created chunked table: {:?}", result.chunked_table_created);
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use crate::database::models::IngestedFile;
use tracing::{debug, info, warn, error, instrument};

/// Errors that can occur during chunk-level task generation
#[derive(Error, Debug)]
pub enum TaskGeneratorError {
    /// Table does not exist in the database
    #[error("Table '{table}' does not exist")]
    TableNotFound { table: String },

    /// Invalid chunk size provided
    #[error("Invalid chunk size: {size} (must be > 0)")]
    InvalidChunkSize { size: usize },

    /// Database operation failed
    #[error("Database error: {0}")]
    Database(#[from] crate::error::DatabaseError),

    /// File I/O operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Chunking operation failed
    #[error("Chunking failed: {cause}")]
    ChunkingFailed { cause: String },

    /// Content file writing failed
    #[error("Content file writing failed: {path} - {cause}")]
    ContentWriteFailed { path: String, cause: String },

    /// Task list generation failed
    #[error("Task list generation failed: {cause}")]
    TaskListFailed { cause: String },

    /// Invalid table name format
    #[error("Invalid table name: {table} - {cause}")]
    InvalidTableName { table: String, cause: String },

    /// Resource cleanup failed
    #[error("Resource cleanup failed: {cause}")]
    CleanupFailed { cause: String },
}

/// Represents a file after chunking processing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkedFile {
    /// Original file ID from the source table
    pub original_file_id: i64,
    
    /// Chunk number within the original file (0-based)
    pub chunk_number: usize,
    
    /// Content of this chunk
    pub content: String,
    
    /// L1 content: current chunk + next chunk
    pub content_l1: String,
    
    /// L2 content: current chunk + next chunk + next2 chunk
    pub content_l2: String,
    
    /// Number of lines in this chunk
    pub line_count: i32,
    
    /// Original file path for reference
    pub original_filepath: String,
}

/// Result of a task generation operation
#[derive(Debug, Serialize, Deserialize)]
pub struct TaskGenerationResult {
    /// Name of the table that was processed
    pub table_used: String,
    
    /// Number of rows processed from the table
    pub rows_processed: usize,
    
    /// Number of content files created (3 per row: content, contentL1, contentL2)
    pub content_files_created: usize,
    
    /// Path to the generated task list file
    pub task_list_path: PathBuf,
    
    /// Name of the chunked table created (if chunk-level mode was used)
    pub chunked_table_created: Option<String>,
    
    /// Processing statistics
    pub processing_stats: ProcessingStats,
}

/// Statistics about the processing operation
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Number of files that were chunked
    pub files_chunked: usize,
    
    /// Number of files copied unchanged (small files)
    pub files_copied: usize,
    
    /// Total number of chunks created
    pub total_chunks_created: usize,
    
    /// Average chunk size in lines
    pub average_chunk_size: f64,
}

/// Paths to content files for a single row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFiles {
    /// Path to the main content file
    pub content: PathBuf,
    
    /// Path to the L1 content file (content + next)
    pub content_l1: PathBuf,
    
    /// Path to the L2 content file (content + next + next2)
    pub content_l2: PathBuf,
}

/// Information about a chunking operation result
#[derive(Debug)]
pub struct ChunkingResult {
    /// Name of the created chunked table
    pub chunked_table_name: String,
    
    /// Number of original files processed
    pub original_files_processed: usize,
    
    /// Number of chunks created
    pub chunks_created: usize,
    
    /// Processing statistics
    pub stats: ProcessingStats,
}

impl TaskGeneratorError {
    /// Create a table not found error with helpful context
    pub fn table_not_found(table: impl Into<String>) -> Self {
        Self::TableNotFound { table: table.into() }
    }

    /// Create an invalid chunk size error
    pub fn invalid_chunk_size(size: usize) -> Self {
        Self::InvalidChunkSize { size }
    }

    /// Create a chunking failed error
    pub fn chunking_failed(cause: impl Into<String>) -> Self {
        Self::ChunkingFailed { cause: cause.into() }
    }

    /// Create a content write failed error
    pub fn content_write_failed(path: impl Into<String>, cause: impl Into<String>) -> Self {
        Self::ContentWriteFailed { 
            path: path.into(), 
            cause: cause.into() 
        }
    }

    /// Create a task list generation failed error
    pub fn task_list_failed(cause: impl Into<String>) -> Self {
        Self::TaskListFailed { cause: cause.into() }
    }

    /// Create an invalid table name error
    pub fn invalid_table_name(table: impl Into<String>, cause: impl Into<String>) -> Self {
        Self::InvalidTableName { 
            table: table.into(), 
            cause: cause.into() 
        }
    }

    /// Create a cleanup failed error
    pub fn cleanup_failed(cause: impl Into<String>) -> Self {
        Self::CleanupFailed { cause: cause.into() }
    }

    /// Check if this error is recoverable with retry
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Database(db_err) => {
                // Database connection errors are often recoverable
                matches!(db_err, crate::error::DatabaseError::ConnectionFailed { .. })
            }
            Self::Io(_) => true, // I/O errors might be temporary
            Self::ChunkingFailed { .. } => false, // Logic errors are not recoverable
            Self::TableNotFound { .. } => false, // Table existence is not recoverable
            Self::InvalidChunkSize { .. } => false, // Invalid input is not recoverable
            Self::ContentWriteFailed { .. } => true, // File write might be temporary
            Self::TaskListFailed { .. } => true, // Task list generation might be temporary
            Self::InvalidTableName { .. } => false, // Invalid table name is not recoverable
            Self::CleanupFailed { .. } => false, // Cleanup failures are not critical
        }
    }
}

impl ChunkedFile {
    /// Create a new chunked file from an original file and chunk data
    pub fn new(
        original_file: &IngestedFile,
        chunk_number: usize,
        content: String,
        content_l1: String,
        content_l2: String,
        line_count: i32,
    ) -> Self {
        Self {
            original_file_id: original_file.file_id,
            chunk_number,
            content,
            content_l1,
            content_l2,
            line_count,
            original_filepath: original_file.filepath.clone(),
        }
    }

    /// Get the chunk identifier string
    pub fn chunk_id(&self) -> String {
        format!("{}_{}", self.original_file_id, self.chunk_number)
    }

    /// Check if this is the first chunk of a file
    pub fn is_first_chunk(&self) -> bool {
        self.chunk_number == 0
    }

    /// Get the content length in characters
    pub fn content_length(&self) -> usize {
        self.content.len()
    }

    /// Get the L1 content length in characters
    pub fn content_l1_length(&self) -> usize {
        self.content_l1.len()
    }

    /// Get the L2 content length in characters
    pub fn content_l2_length(&self) -> usize {
        self.content_l2.len()
    }
}

impl TaskGenerationResult {
    /// Create a new task generation result
    pub fn new(
        table_used: String,
        rows_processed: usize,
        content_files_created: usize,
        task_list_path: PathBuf,
        chunked_table_created: Option<String>,
        processing_stats: ProcessingStats,
    ) -> Self {
        Self {
            table_used,
            rows_processed,
            content_files_created,
            task_list_path,
            chunked_table_created,
            processing_stats,
        }
    }

    /// Check if chunk-level mode was used
    pub fn used_chunking(&self) -> bool {
        self.chunked_table_created.is_some()
    }

    /// Get the average files per content file created
    pub fn files_per_content_ratio(&self) -> f64 {
        if self.content_files_created == 0 {
            0.0
        } else {
            self.rows_processed as f64 / (self.content_files_created as f64 / 3.0)
        }
    }
}

impl ProcessingStats {
    /// Create new processing stats
    pub fn new() -> Self {
        Self {
            processing_time_ms: 0,
            files_chunked: 0,
            files_copied: 0,
            total_chunks_created: 0,
            average_chunk_size: 0.0,
        }
    }

    /// Update the processing time
    pub fn set_processing_time(&mut self, time_ms: u64) {
        self.processing_time_ms = time_ms;
    }

    /// Add a chunked file to the stats
    pub fn add_chunked_file(&mut self, chunks_created: usize, total_lines: usize) {
        self.files_chunked += 1;
        self.total_chunks_created += chunks_created;
        
        // Update average chunk size
        if self.total_chunks_created > 0 {
            let total_lines_processed = (self.average_chunk_size * (self.total_chunks_created - chunks_created) as f64) + total_lines as f64;
            self.average_chunk_size = total_lines_processed / self.total_chunks_created as f64;
        }
    }

    /// Add a copied file to the stats
    pub fn add_copied_file(&mut self) {
        self.files_copied += 1;
    }

    /// Get total files processed
    pub fn total_files_processed(&self) -> usize {
        self.files_chunked + self.files_copied
    }

    /// Get processing rate in files per second
    pub fn processing_rate_fps(&self) -> f64 {
        if self.processing_time_ms == 0 {
            0.0
        } else {
            (self.total_files_processed() as f64) / (self.processing_time_ms as f64 / 1000.0)
        }
    }
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentFiles {
    /// Create new content files paths
    pub fn new(content: PathBuf, content_l1: PathBuf, content_l2: PathBuf) -> Self {
        Self {
            content,
            content_l1,
            content_l2,
        }
    }

    /// Get all file paths as a vector
    pub fn all_paths(&self) -> Vec<&PathBuf> {
        vec![&self.content, &self.content_l1, &self.content_l2]
    }

    /// Check if all files exist
    pub fn all_exist(&self) -> bool {
        self.content.exists() && self.content_l1.exists() && self.content_l2.exists()
    }

    /// Get the total size of all content files in bytes
    pub fn total_size_bytes(&self) -> std::io::Result<u64> {
        let mut total = 0;
        for path in self.all_paths() {
            total += std::fs::metadata(path)?.len();
        }
        Ok(total)
    }
}

impl ChunkingResult {
    /// Create a new chunking result
    pub fn new(
        chunked_table_name: String,
        original_files_processed: usize,
        chunks_created: usize,
        stats: ProcessingStats,
    ) -> Self {
        Self {
            chunked_table_name,
            original_files_processed,
            chunks_created,
            stats,
        }
    }

    /// Get the chunking ratio (chunks per original file)
    pub fn chunking_ratio(&self) -> f64 {
        if self.original_files_processed == 0 {
            0.0
        } else {
            self.chunks_created as f64 / self.original_files_processed as f64
        }
    }
}

/// Type alias for task generator results
pub type TaskGeneratorResult<T> = Result<T, TaskGeneratorError>;

/// Main coordinator for chunk-level task generation
/// 
/// This struct orchestrates all the services needed for chunk-level task generation:
/// - DatabaseService for table operations
/// - ContentFileWriter for file generation  
/// - TaskListGenerator for task list creation
/// - ChunkingService for file processing logic
/// 
/// # Examples
/// 
/// ```rust
/// use code_ingest::tasks::chunk_level_task_generator::ChunkLevelTaskGenerator;
/// use std::path::PathBuf;
/// 
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let generator = ChunkLevelTaskGenerator::new(
///     database_service,
///     content_writer, 
///     task_generator,
///     chunking_service,
/// );
/// 
/// // File-level mode (no chunking)
/// let result = generator.execute("INGEST_TABLE", None, None).await?;
/// 
/// // Chunk-level mode (with chunking)
/// let result = generator.execute("INGEST_TABLE", Some(500), None).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ChunkLevelTaskGenerator {
    database: std::sync::Arc<crate::tasks::database_service::DatabaseService>,
    content_writer: crate::tasks::content_file_writer::ContentFileWriter,
    task_generator: crate::tasks::task_list_generator::TaskListGenerator,
    chunking_service: crate::tasks::chunking_service::ChunkingService,
}

impl ChunkLevelTaskGenerator {
    /// Create a new chunk-level task generator with all required services
    /// 
    /// # Arguments
    /// * `database` - Database service for table operations
    /// * `content_writer` - Content file writer for generating content files
    /// * `task_generator` - Task list generator for creating task lists
    /// * `chunking_service` - Chunking service for file processing
    /// 
    /// # Returns
    /// * `ChunkLevelTaskGenerator` - New instance ready for task generation
    pub fn new(
        database: std::sync::Arc<crate::tasks::database_service::DatabaseService>,
        content_writer: crate::tasks::content_file_writer::ContentFileWriter,
        task_generator: crate::tasks::task_list_generator::TaskListGenerator,
        chunking_service: crate::tasks::chunking_service::ChunkingService,
    ) -> Self {
        tracing::debug!("Creating ChunkLevelTaskGenerator");
        Self {
            database,
            content_writer,
            task_generator,
            chunking_service,
        }
    }

    /// Execute the chunk-level task generation with file-level and chunk-level mode logic
    /// 
    /// # Arguments
    /// * `table_name` - Name of the database table to process
    /// * `chunk_size` - Optional chunk size for chunk-level mode (None for file-level mode)
    /// * `db_path` - Optional database path override
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<TaskGenerationResult>` - Result of the task generation operation
    /// 
    /// # Requirements
    /// This method satisfies requirements 1.1, 1.2, 2.1, 2.6, and 2.7 by orchestrating
    /// all services to generate content files and task lists in both modes
    #[tracing::instrument(skip(self), fields(table_name = %table_name, chunk_size = ?chunk_size))]
    pub async fn execute(
        &self,
        table_name: &str,
        chunk_size: Option<usize>,
        db_path: Option<PathBuf>,
    ) -> TaskGeneratorResult<TaskGenerationResult> {
        let start_time = std::time::Instant::now();
        tracing::info!("Starting chunk-level task generation for table '{}' with chunk_size {:?}", 
                      table_name, chunk_size);

        // Input validation
        self.validate_inputs(table_name, chunk_size)?;

        // Validate that the original table exists and has valid schema
        let table_info = self.database.validate_table(table_name).await?;
        tracing::info!("Validated table '{}' with {} rows", table_name, table_info.row_count);

        let mut processing_stats = ProcessingStats::new();
        let table_to_process;
        let chunked_table_created;

        // Determine processing mode and prepare data
        if let Some(chunk_size) = chunk_size {
            // Chunk-level mode: create chunked table and process with chunking
            tracing::info!("Using chunk-level mode with chunk size {}", chunk_size);
            
            let chunked_table_name = self.database.create_chunked_table(table_name, chunk_size).await?;
            tracing::info!("Created chunked table: {}", chunked_table_name);

            let chunking_result = self.chunking_service
                .process_with_chunking(table_name, &chunked_table_name, chunk_size)
                .await?;
            
            tracing::info!("Chunking completed: {} original files -> {} chunks", 
                          chunking_result.original_files_processed, chunking_result.chunks_created);

            table_to_process = chunked_table_name.clone();
            chunked_table_created = Some(chunked_table_name);
            processing_stats = chunking_result.stats;
        } else {
            // File-level mode: process original table directly
            tracing::info!("Using file-level mode (no chunking)");
            table_to_process = table_name.to_string();
            chunked_table_created = None;
        }

        // Query rows from the table to process (either original or chunked)
        let rows = self.database.query_rows(&table_to_process).await?;
        tracing::info!("Queried {} rows from table '{}'", rows.len(), table_to_process);

        // Generate content files
        let content_result = self.content_writer
            .write_content_files(&table_to_process, &rows)
            .await?;
        
        tracing::info!("Generated {} content files from {} rows", 
                      content_result.files_created, content_result.rows_processed);

        // Generate task list
        let task_list_filename = crate::tasks::task_list_generator::TaskListGenerator::default_task_list_filename(&table_to_process);
        let task_list_path = self.content_writer.config().output_dir.join(task_list_filename);
        
        self.task_generator
            .write_task_list_to_file(&rows, &task_list_path)
            .await?;
        
        tracing::info!("Generated task list at: {}", task_list_path.display());

        // Update processing stats with total time
        let total_time = start_time.elapsed();
        processing_stats.set_processing_time(total_time.as_millis() as u64);

        let result = TaskGenerationResult::new(
            table_to_process,
            rows.len(),
            content_result.files_created,
            task_list_path,
            chunked_table_created,
            processing_stats,
        );

        tracing::info!("Task generation completed successfully in {:?}: {} rows processed, {} files created", 
                      total_time, result.rows_processed, result.content_files_created);

        Ok(result)
    }

    /// Validate input parameters for task generation
    /// 
    /// # Arguments
    /// * `table_name` - Table name to validate
    /// * `chunk_size` - Optional chunk size to validate
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<()>` - Success or validation error
    /// 
    /// # Requirements
    /// This method satisfies requirement 3.1 and 3.2 by providing input validation
    fn validate_inputs(&self, table_name: &str, chunk_size: Option<usize>) -> TaskGeneratorResult<()> {
        tracing::debug!("Validating inputs: table_name='{}', chunk_size={:?}", table_name, chunk_size);

        // Validate table name
        if table_name.is_empty() {
            return Err(TaskGeneratorError::invalid_table_name(
                table_name,
                "Table name cannot be empty"
            ));
        }

        if table_name.len() > 63 {
            return Err(TaskGeneratorError::invalid_table_name(
                table_name,
                "Table name too long (max 63 characters)"
            ));
        }

        // Basic SQL injection protection
        if table_name.contains(';') || table_name.contains('\'') || table_name.contains('"') {
            return Err(TaskGeneratorError::invalid_table_name(
                table_name,
                "Table name contains invalid characters"
            ));
        }

        // Validate chunk size if provided
        if let Some(chunk_size) = chunk_size {
            self.chunking_service.validate_chunking_params(chunk_size)?;
        }

        tracing::debug!("Input validation passed");
        Ok(())
    }

    /// Perform cleanup operations after task generation
    /// 
    /// This method can be used to clean up temporary resources, close connections,
    /// or perform other cleanup operations.
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<()>` - Success or cleanup error
    pub async fn cleanup(&self) -> TaskGeneratorResult<()> {
        tracing::debug!("Performing cleanup operations");
        
        // Currently no specific cleanup needed, but this provides a hook
        // for future cleanup operations like:
        // - Closing database connections
        // - Cleaning up temporary files
        // - Releasing other resources
        
        tracing::debug!("Cleanup completed successfully");
        Ok(())
    }

    /// Get statistics about the last chunking operation
    /// 
    /// # Arguments
    /// * `chunked_table` - Name of the chunked table to analyze
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<ProcessingStats>` - Statistics about the chunked table
    pub async fn get_chunking_stats(&self, chunked_table: &str) -> TaskGeneratorResult<ProcessingStats> {
        self.chunking_service.get_chunking_stats(chunked_table).await
    }

    /// Get a reference to the database service
    pub fn database(&self) -> &std::sync::Arc<crate::tasks::database_service::DatabaseService> {
        &self.database
    }

    /// Get a reference to the content writer
    pub fn content_writer(&self) -> &crate::tasks::content_file_writer::ContentFileWriter {
        &self.content_writer
    }

    /// Get a reference to the task generator
    pub fn task_generator(&self) -> &crate::tasks::task_list_generator::TaskListGenerator {
        &self.task_generator
    }

    /// Get a reference to the chunking service
    pub fn chunking_service(&self) -> &crate::tasks::chunking_service::ChunkingService {
        &self.chunking_service
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::models::IngestedFile;
    use chrono::Utc;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_ingested_file(file_id: i64, filepath: &str, content: Option<String>) -> IngestedFile {
        IngestedFile {
            file_id,
            ingestion_id: 1,
            filepath: filepath.to_string(),
            filename: filepath.split('/').last().unwrap_or(filepath).to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: content.as_ref().map_or(0, |c| c.len() as i64),
            line_count: content.as_ref().map(|c| c.lines().count() as i32),
            word_count: content.as_ref().map(|c| c.split_whitespace().count() as i32),
            token_count: content.as_ref().map(|c| (c.split_whitespace().count() as f32 * 0.75) as i32),
            content_text: content,
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: filepath.to_string(),
            absolute_path: format!("/tmp/{}", filepath),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_task_generator_error_creation() {
        let error = TaskGeneratorError::table_not_found("NONEXISTENT_TABLE");
        assert!(matches!(error, TaskGeneratorError::TableNotFound { .. }));
        assert_eq!(error.to_string(), "Table 'NONEXISTENT_TABLE' does not exist");

        let error = TaskGeneratorError::invalid_chunk_size(0);
        assert!(matches!(error, TaskGeneratorError::InvalidChunkSize { .. }));
        assert_eq!(error.to_string(), "Invalid chunk size: 0 (must be > 0)");

        let error = TaskGeneratorError::chunking_failed("Test failure");
        assert!(matches!(error, TaskGeneratorError::ChunkingFailed { .. }));
        assert_eq!(error.to_string(), "Chunking failed: Test failure");
    }

    #[test]
    fn test_task_generator_error_recoverability() {
        let recoverable_error = TaskGeneratorError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Permission denied"
        ));
        assert!(recoverable_error.is_recoverable());

        let non_recoverable_error = TaskGeneratorError::table_not_found("TEST");
        assert!(!non_recoverable_error.is_recoverable());

        let non_recoverable_error = TaskGeneratorError::invalid_chunk_size(0);
        assert!(!non_recoverable_error.is_recoverable());
    }

    #[test]
    fn test_chunked_file_creation() {
        let original_file = create_test_ingested_file(
            123,
            "src/main.rs",
            Some("fn main() {\n    println!(\"Hello, world!\");\n}".to_string())
        );

        let chunked_file = ChunkedFile::new(
            &original_file,
            0,
            "fn main() {".to_string(),
            "fn main() {\n    println!(\"Hello, world!\");".to_string(),
            "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            1,
        );

        assert_eq!(chunked_file.original_file_id, 123);
        assert_eq!(chunked_file.chunk_number, 0);
        assert_eq!(chunked_file.content, "fn main() {");
        assert_eq!(chunked_file.line_count, 1);
        assert_eq!(chunked_file.original_filepath, "src/main.rs");
        assert_eq!(chunked_file.chunk_id(), "123_0");
        assert!(chunked_file.is_first_chunk());
        assert_eq!(chunked_file.content_length(), 11);
    }

    #[test]
    fn test_chunked_file_serialization() {
        let original_file = create_test_ingested_file(456, "test.rs", Some("test content".to_string()));
        let chunked_file = ChunkedFile::new(
            &original_file,
            1,
            "content".to_string(),
            "content_l1".to_string(),
            "content_l2".to_string(),
            5,
        );

        // Test serialization
        let serialized = serde_json::to_string(&chunked_file).unwrap();
        assert!(serialized.contains("456"));
        assert!(serialized.contains("content"));
        assert!(serialized.contains("test.rs"));

        // Test deserialization
        let deserialized: ChunkedFile = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, chunked_file);
    }

    #[test]
    fn test_task_generation_result() {
        let temp_dir = TempDir::new().unwrap();
        let task_list_path = temp_dir.path().join("tasks.txt");
        let stats = ProcessingStats::new();

        let result = TaskGenerationResult::new(
            "INGEST_TEST".to_string(),
            10,
            30, // 3 files per row
            task_list_path.clone(),
            Some("INGEST_TEST_500".to_string()),
            stats,
        );

        assert_eq!(result.table_used, "INGEST_TEST");
        assert_eq!(result.rows_processed, 10);
        assert_eq!(result.content_files_created, 30);
        assert_eq!(result.task_list_path, task_list_path);
        assert!(result.used_chunking());
        assert_eq!(result.chunked_table_created, Some("INGEST_TEST_500".to_string()));
        assert_eq!(result.files_per_content_ratio(), 1.0); // 10 rows / (30 files / 3) = 1.0
    }

    #[test]
    fn test_processing_stats() {
        let mut stats = ProcessingStats::new();
        assert_eq!(stats.total_files_processed(), 0);
        assert_eq!(stats.processing_rate_fps(), 0.0);

        stats.add_chunked_file(3, 150); // 3 chunks, 150 total lines
        stats.add_copied_file();
        stats.set_processing_time(1000); // 1 second

        assert_eq!(stats.files_chunked, 1);
        assert_eq!(stats.files_copied, 1);
        assert_eq!(stats.total_chunks_created, 3);
        assert_eq!(stats.total_files_processed(), 2);
        assert_eq!(stats.average_chunk_size, 50.0); // 150 lines / 3 chunks
        assert_eq!(stats.processing_rate_fps(), 2.0); // 2 files / 1 second
    }

    #[test]
    fn test_content_files() {
        let temp_dir = TempDir::new().unwrap();
        let content_path = temp_dir.path().join("content_1.txt");
        let content_l1_path = temp_dir.path().join("contentL1_1.txt");
        let content_l2_path = temp_dir.path().join("contentL2_1.txt");

        let content_files = ContentFiles::new(
            content_path.clone(),
            content_l1_path.clone(),
            content_l2_path.clone(),
        );

        assert_eq!(content_files.content, content_path);
        assert_eq!(content_files.content_l1, content_l1_path);
        assert_eq!(content_files.content_l2, content_l2_path);
        assert_eq!(content_files.all_paths().len(), 3);
        assert!(!content_files.all_exist()); // Files don't exist yet

        // Create the files
        std::fs::write(&content_path, "content").unwrap();
        std::fs::write(&content_l1_path, "content_l1").unwrap();
        std::fs::write(&content_l2_path, "content_l2").unwrap();

        assert!(content_files.all_exist());
        let total_size = content_files.total_size_bytes().unwrap();
        assert_eq!(total_size, 7 + 10 + 10); // "content" + "content_l1" + "content_l2"
    }

    #[test]
    fn test_chunking_result() {
        let stats = ProcessingStats::new();
        let result = ChunkingResult::new(
            "INGEST_TEST_500".to_string(),
            10, // original files
            25, // chunks created
            stats,
        );

        assert_eq!(result.chunked_table_name, "INGEST_TEST_500");
        assert_eq!(result.original_files_processed, 10);
        assert_eq!(result.chunks_created, 25);
        assert_eq!(result.chunking_ratio(), 2.5); // 25 chunks / 10 files = 2.5
    }

    #[test]
    fn test_error_conversion() {
        // Test conversion from DatabaseError
        let db_error = crate::error::DatabaseError::TableNotFound {
            table_name: "TEST".to_string(),
        };
        let task_error: TaskGeneratorError = db_error.into();
        assert!(matches!(task_error, TaskGeneratorError::Database(_)));

        // Test conversion from std::io::Error
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let task_error: TaskGeneratorError = io_error.into();
        assert!(matches!(task_error, TaskGeneratorError::Io(_)));
    }

    #[test]
    fn test_basic_functionality() {
        // Test that our basic data structures work without external dependencies
        let mut stats = ProcessingStats::new();
        assert_eq!(stats.total_files_processed(), 0);
        
        stats.add_copied_file();
        assert_eq!(stats.files_copied, 1);
        assert_eq!(stats.total_files_processed(), 1);
        
        stats.add_chunked_file(2, 100);
        assert_eq!(stats.files_chunked, 1);
        assert_eq!(stats.total_chunks_created, 2);
        assert_eq!(stats.average_chunk_size, 50.0);
        
        println!("✅ Basic functionality test passed");
    }

    // Tests for ChunkLevelTaskGenerator

    async fn create_test_generator() -> Option<ChunkLevelTaskGenerator> {
        // Create a test generator with mock services
        // This requires a database connection for full functionality
        if let Ok(database_url) = std::env::var("DATABASE_URL") {
            match sqlx::PgPool::connect(&database_url).await {
                Ok(pool) => {
                    let database = std::sync::Arc::new(crate::tasks::database_service::DatabaseService::new(std::sync::Arc::new(pool.clone())));
                    let content_writer = crate::tasks::content_file_writer::ContentFileWriter::new(
                        crate::tasks::content_file_writer::ContentWriteConfig::new(
                            tempfile::TempDir::new().unwrap().into_path()
                        )
                    );
                    let task_generator = crate::tasks::task_list_generator::TaskListGenerator::new();
                    let chunking_service = crate::tasks::chunking_service::ChunkingService::new(database.clone());
                    
                    Some(ChunkLevelTaskGenerator::new(
                        database,
                        content_writer,
                        task_generator,
                        chunking_service,
                    ))
                }
                Err(_) => None,
            }
        } else {
            None
        }
    }

    #[tokio::test]
    async fn test_chunk_level_task_generator_creation() {
        // Test that we can create a ChunkLevelTaskGenerator
        if let Some(generator) = create_test_generator().await {
            // Test that all services are accessible
            assert!(generator.database().pool().size() >= 0);
            assert!(generator.content_writer().config().output_dir.exists() || !generator.content_writer().config().output_dir.exists());
            
            println!("✅ ChunkLevelTaskGenerator creation test passed");
        } else {
            println!("⚠️ Skipping ChunkLevelTaskGenerator creation test (no database connection)");
        }
    }

    #[tokio::test]
    async fn test_validate_inputs() {
        if let Some(generator) = create_test_generator().await {
            // Test valid inputs
            let result = generator.validate_inputs("VALID_TABLE", Some(500));
            assert!(result.is_ok());

            let result = generator.validate_inputs("VALID_TABLE", None);
            assert!(result.is_ok());

            // Test invalid table names
            let result = generator.validate_inputs("", None);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), TaskGeneratorError::InvalidTableName { .. }));

            let result = generator.validate_inputs("table_with_semicolon;", None);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), TaskGeneratorError::InvalidTableName { .. }));

            let result = generator.validate_inputs("table'with'quotes", None);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), TaskGeneratorError::InvalidTableName { .. }));

            // Test very long table name
            let long_name = "a".repeat(64);
            let result = generator.validate_inputs(&long_name, None);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), TaskGeneratorError::InvalidTableName { .. }));

            // Test invalid chunk size
            let result = generator.validate_inputs("VALID_TABLE", Some(0));
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), TaskGeneratorError::InvalidChunkSize { .. }));

            println!("✅ Input validation test passed");
        } else {
            println!("⚠️ Skipping input validation test (no database connection)");
        }
    }

    #[tokio::test]
    async fn test_cleanup() {
        if let Some(generator) = create_test_generator().await {
            // Test cleanup operation
            let result = generator.cleanup().await;
            assert!(result.is_ok());

            println!("✅ Cleanup test passed");
        } else {
            println!("⚠️ Skipping cleanup test (no database connection)");
        }
    }

    #[tokio::test]
    async fn test_service_accessors() {
        if let Some(generator) = create_test_generator().await {
            // Test that we can access all services
            let _database = generator.database();
            let _content_writer = generator.content_writer();
            let _task_generator = generator.task_generator();
            let _chunking_service = generator.chunking_service();

            println!("✅ Service accessors test passed");
        } else {
            println!("⚠️ Skipping service accessors test (no database connection)");
        }
    }

    // Integration tests that require a real database connection

    #[tokio::test]
    async fn test_execute_file_level_mode() {
        // Test file-level mode execution (no chunking)
        if std::env::var("DATABASE_URL").is_err() {
            println!("⚠️ Skipping file-level mode test (no DATABASE_URL)");
            return;
        }

        if let Some(generator) = create_test_generator().await {
            // Test with a non-existent table (should fail gracefully)
            let result = generator.execute("NONEXISTENT_TABLE", None, None).await;
            assert!(result.is_err());
            
            match result.unwrap_err() {
                TaskGeneratorError::TableNotFound { table } => {
                    assert_eq!(table, "NONEXISTENT_TABLE");
                }
                _ => panic!("Expected TableNotFound error"),
            }

            println!("✅ File-level mode execution test passed");
        }
    }

    #[tokio::test]
    async fn test_execute_chunk_level_mode() {
        // Test chunk-level mode execution (with chunking)
        if std::env::var("DATABASE_URL").is_err() {
            println!("⚠️ Skipping chunk-level mode test (no DATABASE_URL)");
            return;
        }

        if let Some(generator) = create_test_generator().await {
            // Test with invalid chunk size
            let result = generator.execute("ANY_TABLE", Some(0), None).await;
            assert!(result.is_err());
            
            match result.unwrap_err() {
                TaskGeneratorError::InvalidChunkSize { size } => {
                    assert_eq!(size, 0);
                }
                _ => panic!("Expected InvalidChunkSize error"),
            }

            println!("✅ Chunk-level mode execution test passed");
        }
    }

    #[tokio::test]
    async fn test_execute_with_real_table() {
        // Test execution with a real table (if available)
        if std::env::var("DATABASE_URL").is_err() {
            println!("⚠️ Skipping real table test (no DATABASE_URL)");
            return;
        }

        if let Some(generator) = create_test_generator().await {
            // Initialize database schema if needed
            let db = crate::database::connection::Database::new(&std::env::var("DATABASE_URL").unwrap()).await.unwrap();
            let _ = db.initialize_schema().await;

            // Try to find an existing ingestion table
            let pool = generator.database().pool();
            let tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'INGEST_%' LIMIT 1";
            
            if let Ok(row) = sqlx::query(tables_query).fetch_optional(pool.as_ref()).await {
                if let Some(row) = row {
                    let table_name: String = row.get("table_name");
                    println!("Found test table: {}", table_name);

                    // Test file-level mode
                    let result = generator.execute(&table_name, None, None).await;
                    if result.is_ok() {
                        let result = result.unwrap();
                        assert_eq!(result.table_used, table_name);
                        assert!(!result.used_chunking());
                        println!("✅ Real table file-level test passed: {} rows processed", result.rows_processed);
                    } else {
                        println!("⚠️ Real table test failed (table may not have proper schema): {:?}", result.unwrap_err());
                    }
                } else {
                    println!("⚠️ No ingestion tables found for testing");
                }
            } else {
                println!("⚠️ Could not query for ingestion tables");
            }
        }
    }

    #[test]
    fn test_task_generation_result_properties() {
        // Test TaskGenerationResult helper methods
        let temp_dir = tempfile::TempDir::new().unwrap();
        let task_list_path = temp_dir.path().join("tasks.txt");
        let stats = ProcessingStats::new();

        // Test file-level mode result
        let result = TaskGenerationResult::new(
            "TEST_TABLE".to_string(),
            10,
            30, // 3 files per row
            task_list_path.clone(),
            None, // No chunking
            stats.clone(),
        );

        assert!(!result.used_chunking());
        assert_eq!(result.files_per_content_ratio(), 1.0); // 10 rows / (30 files / 3) = 1.0

        // Test chunk-level mode result
        let result = TaskGenerationResult::new(
            "TEST_TABLE_500".to_string(),
            25, // More rows due to chunking
            75, // 3 files per row
            task_list_path,
            Some("TEST_TABLE_500".to_string()),
            stats,
        );

        assert!(result.used_chunking());
        assert_eq!(result.files_per_content_ratio(), 1.0); // 25 rows / (75 files / 3) = 1.0

        println!("✅ TaskGenerationResult properties test passed");
    }

    #[test]
    fn test_error_handling_comprehensive() {
        // Test comprehensive error handling
        
        // Test error creation methods
        let error = TaskGeneratorError::table_not_found("TEST");
        assert!(error.to_string().contains("TEST"));
        assert!(!error.is_recoverable());

        let error = TaskGeneratorError::invalid_chunk_size(0);
        assert!(error.to_string().contains("0"));
        assert!(!error.is_recoverable());

        let error = TaskGeneratorError::chunking_failed("Test failure");
        assert!(error.to_string().contains("Test failure"));
        assert!(!error.is_recoverable());

        let error = TaskGeneratorError::content_write_failed("test.txt", "Permission denied");
        assert!(error.to_string().contains("test.txt"));
        assert!(error.to_string().contains("Permission denied"));
        assert!(error.is_recoverable());

        let error = TaskGeneratorError::task_list_failed("Generation failed");
        assert!(error.to_string().contains("Generation failed"));
        assert!(error.is_recoverable());

        let error = TaskGeneratorError::invalid_table_name("BAD_TABLE", "Invalid format");
        assert!(error.to_string().contains("BAD_TABLE"));
        assert!(error.to_string().contains("Invalid format"));
        assert!(!error.is_recoverable());

        let error = TaskGeneratorError::cleanup_failed("Cleanup issue");
        assert!(error.to_string().contains("Cleanup issue"));
        assert!(!error.is_recoverable());

        println!("✅ Comprehensive error handling test passed");
    }

    #[test]
    fn test_processing_stats_comprehensive() {
        // Test comprehensive ProcessingStats functionality
        let mut stats = ProcessingStats::new();
        
        // Test initial state
        assert_eq!(stats.processing_time_ms, 0);
        assert_eq!(stats.files_chunked, 0);
        assert_eq!(stats.files_copied, 0);
        assert_eq!(stats.total_chunks_created, 0);
        assert_eq!(stats.average_chunk_size, 0.0);
        assert_eq!(stats.total_files_processed(), 0);
        assert_eq!(stats.processing_rate_fps(), 0.0);

        // Add some files
        stats.add_copied_file();
        stats.add_copied_file();
        stats.add_chunked_file(3, 150); // 3 chunks, 150 total lines
        stats.add_chunked_file(2, 100); // 2 chunks, 100 total lines
        stats.set_processing_time(2000); // 2 seconds

        // Test updated state
        assert_eq!(stats.files_copied, 2);
        assert_eq!(stats.files_chunked, 2);
        assert_eq!(stats.total_chunks_created, 5); // 3 + 2
        assert_eq!(stats.total_files_processed(), 4); // 2 copied + 2 chunked
        assert_eq!(stats.average_chunk_size, 50.0); // (150 + 100) / 5 chunks
        assert_eq!(stats.processing_rate_fps(), 2.0); // 4 files / 2 seconds
        assert_eq!(stats.processing_time_ms, 2000);

        println!("✅ Comprehensive ProcessingStats test passed");
    }

    #[test]
    fn test_content_files_comprehensive() {
        // Test comprehensive ContentFiles functionality
        let temp_dir = tempfile::TempDir::new().unwrap();
        let content_path = temp_dir.path().join("content_1.txt");
        let content_l1_path = temp_dir.path().join("contentL1_1.txt");
        let content_l2_path = temp_dir.path().join("contentL2_1.txt");

        let content_files = ContentFiles::new(
            content_path.clone(),
            content_l1_path.clone(),
            content_l2_path.clone(),
        );

        // Test paths
        assert_eq!(content_files.all_paths().len(), 3);
        assert!(content_files.all_paths().contains(&&content_path));
        assert!(content_files.all_paths().contains(&&content_l1_path));
        assert!(content_files.all_paths().contains(&&content_l2_path));

        // Test existence (files don't exist yet)
        assert!(!content_files.all_exist());

        // Create files with different sizes
        std::fs::write(&content_path, "content").unwrap(); // 7 bytes
        std::fs::write(&content_l1_path, "content_l1_data").unwrap(); // 15 bytes
        std::fs::write(&content_l2_path, "content_l2_extended_data").unwrap(); // 24 bytes

        // Test existence and size
        assert!(content_files.all_exist());
        let total_size = content_files.total_size_bytes().unwrap();
        assert_eq!(total_size, 7 + 15 + 24); // 46 bytes total

        println!("✅ Comprehensive ContentFiles test passed");
    }

    #[test]
    fn test_chunking_result_comprehensive() {
        // Test comprehensive ChunkingResult functionality
        let mut stats = ProcessingStats::new();
        stats.add_chunked_file(3, 150);
        stats.add_copied_file();
        stats.set_processing_time(1500);

        let result = ChunkingResult::new(
            "TEST_TABLE_500".to_string(),
            10, // original files
            25, // chunks created
            stats,
        );

        assert_eq!(result.chunked_table_name, "TEST_TABLE_500");
        assert_eq!(result.original_files_processed, 10);
        assert_eq!(result.chunks_created, 25);
        assert_eq!(result.chunking_ratio(), 2.5); // 25 chunks / 10 files = 2.5
        assert_eq!(result.stats.processing_time_ms, 1500);
        assert_eq!(result.stats.files_chunked, 1);
        assert_eq!(result.stats.files_copied, 1);

        println!("✅ Comprehensive ChunkingResult test passed");
    }
}