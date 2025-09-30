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
}