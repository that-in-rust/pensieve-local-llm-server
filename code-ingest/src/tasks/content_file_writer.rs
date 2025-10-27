//! Content file writer for generating content, contentL1, and contentL2 files
//!
//! This module provides async file I/O operations for creating content files from database rows.
//! It supports both file-level and chunk-level processing modes with proper error handling
//! and performance optimization.
//!
//! # Examples
//!
//! ```rust
//! use code_ingest::tasks::content_file_writer::{ContentFileWriter, ContentWriteConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ContentWriteConfig::new(PathBuf::from(".raw_data_202509"));
//! let writer = ContentFileWriter::new(config);
//!
//! // Write content files for database rows
//! let result = writer.write_content_files("INGEST_TABLE", &rows).await?;
//! println!("Created {} content files", result.files_created);
//! # Ok(())
//! # }
//! ```

use crate::database::models::IngestedFile;
use crate::tasks::chunk_level_task_generator::{TaskGeneratorError, TaskGeneratorResult, ChunkedFile, ContentFiles};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn, error, instrument};
use futures::future;

/// Configuration for content file writing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentWriteConfig {
    /// Base output directory for content files
    pub output_dir: PathBuf,
    
    /// Whether to create subdirectories for organization
    pub create_subdirs: bool,
    
    /// File naming pattern for content files
    pub naming_pattern: ContentNamingPattern,
    
    /// Whether to overwrite existing files
    pub overwrite_existing: bool,
    
    /// Buffer size for file operations (in bytes)
    pub buffer_size: usize,
    
    /// Maximum concurrent file operations
    pub max_concurrent_writes: usize,
}

/// Naming patterns for content files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentNamingPattern {
    /// Table-based naming: {table_name}_{row_number}_Content.txt
    TableBased,
    /// File-based naming: {filename}_{row_number}_Content.txt
    FileBased,
    /// Custom pattern with placeholders
    Custom(String),
}

/// Result of content file writing operations
#[derive(Debug, Serialize, Deserialize)]
pub struct ContentWriteResult {
    /// Number of content files created
    pub files_created: usize,
    
    /// Number of rows processed
    pub rows_processed: usize,
    
    /// Total bytes written
    pub bytes_written: u64,
    
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// List of created file paths
    pub created_files: Vec<PathBuf>,
    
    /// Any warnings encountered during processing
    pub warnings: Vec<String>,
}

/// Content file writer with async I/O operations
#[derive(Debug)]
pub struct ContentFileWriter {
    config: ContentWriteConfig,
}

impl ContentWriteConfig {
    /// Create a new content write configuration
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            create_subdirs: true,
            naming_pattern: ContentNamingPattern::TableBased,
            overwrite_existing: true,
            buffer_size: 64 * 1024, // 64KB buffer
            max_concurrent_writes: 10,
        }
    }

    /// Set whether to create subdirectories
    pub fn with_subdirs(mut self, create_subdirs: bool) -> Self {
        self.create_subdirs = create_subdirs;
        self
    }

    /// Set the naming pattern
    pub fn with_naming_pattern(mut self, pattern: ContentNamingPattern) -> Self {
        self.naming_pattern = pattern;
        self
    }

    /// Set whether to overwrite existing files
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite_existing = overwrite;
        self
    }

    /// Set buffer size for file operations
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set maximum concurrent writes
    pub fn with_max_concurrent_writes(mut self, max: usize) -> Self {
        self.max_concurrent_writes = max;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> TaskGeneratorResult<()> {
        if self.buffer_size == 0 {
            return Err(TaskGeneratorError::ContentWriteFailed {
                path: "config".to_string(),
                cause: "Buffer size must be greater than 0".to_string(),
            });
        }

        if self.max_concurrent_writes == 0 {
            return Err(TaskGeneratorError::ContentWriteFailed {
                path: "config".to_string(),
                cause: "Max concurrent writes must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

impl ContentFileWriter {
    /// Create a new content file writer
    pub fn new(config: ContentWriteConfig) -> Self {
        Self { config }
    }

    /// Write content files for a list of database rows
    /// 
    /// Creates content, contentL1, and contentL2 files for each row with valid content.
    /// 
    /// # Arguments
    /// * `table_name` - Name of the source table for file naming
    /// * `rows` - List of ingested file records to process
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<ContentWriteResult>` - Write operation results or error
    /// 
    /// # Requirements
    /// This method satisfies requirements 1.1 and 2.6 by creating content files from database rows
    #[instrument(skip(self, rows), fields(table_name = %table_name, row_count = rows.len()))]
    pub async fn write_content_files(
        &self,
        table_name: &str,
        rows: &[IngestedFile],
    ) -> TaskGeneratorResult<ContentWriteResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Starting content file writing for table '{}' with {} rows", table_name, rows.len());
        
        // Validate configuration
        self.config.validate()?;
        
        // Ensure output directory exists
        self.ensure_output_directory().await?;
        
        let mut result = ContentWriteResult {
            files_created: 0,
            rows_processed: 0,
            bytes_written: 0,
            processing_time_ms: 0,
            created_files: Vec::new(),
            warnings: Vec::new(),
        };

        // Process rows in batches to control concurrency
        let batch_size = self.config.max_concurrent_writes;
        let mut row_number = 1;

        for batch in rows.chunks(batch_size) {
            let mut batch_tasks = Vec::new();

            for row in batch {
                let task = self.write_row_files(table_name, row, row_number);
                batch_tasks.push(task);
                row_number += 1;
            }

            // Execute batch concurrently
            let batch_results = futures::future::join_all(batch_tasks).await;

            // Process batch results
            for batch_result in batch_results {
                match batch_result {
                    Ok(row_result) => {
                        result.files_created += row_result.files_created;
                        result.bytes_written += row_result.bytes_written;
                        result.created_files.extend(row_result.created_files);
                        result.rows_processed += 1;
                    }
                    Err(e) => {
                        warn!("Failed to write files for row: {}", e);
                        result.warnings.push(format!("Row processing failed: {}", e));
                    }
                }
            }
        }

        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        info!(
            "Content file writing completed: {} files created, {} rows processed, {} bytes written in {}ms",
            result.files_created, result.rows_processed, result.bytes_written, result.processing_time_ms
        );

        Ok(result)
    }

    /// Write content files for individual row processing
    /// 
    /// Creates the three content files (content, contentL1, contentL2) for a single database row.
    /// 
    /// # Arguments
    /// * `table_name` - Name of the source table for file naming
    /// * `row` - Single ingested file record to process
    /// * `row_number` - Sequential number for this row (1-based)
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<RowWriteResult>` - Individual row write results or error
    /// 
    /// # Requirements
    /// This method satisfies requirement 2.6 by handling individual row processing
    #[instrument(skip(self, row), fields(file_id = row.file_id, row_number = row_number))]
    pub async fn write_row_files(
        &self,
        table_name: &str,
        row: &IngestedFile,
        row_number: usize,
    ) -> TaskGeneratorResult<RowWriteResult> {
        debug!("Writing content files for row {} (file_id: {})", row_number, row.file_id);

        // Skip rows without content
        if !row.has_content() {
            debug!("Skipping row {} - no content available", row_number);
            return Ok(RowWriteResult {
                files_created: 0,
                bytes_written: 0,
                created_files: Vec::new(),
            });
        }

        let content = row.content_text.as_ref().unwrap();
        
        // Generate file paths
        let content_files = self.generate_file_paths(table_name, row, row_number)?;
        
        // Prepare content for L1 and L2 files
        let content_l1 = self.generate_l1_content(content);
        let content_l2 = self.generate_l2_content(content);

        let mut result = RowWriteResult {
            files_created: 0,
            bytes_written: 0,
            created_files: Vec::new(),
        };

        // Write content file
        let bytes_written = self.write_file(&content_files.content, content).await?;
        result.files_created += 1;
        result.bytes_written += bytes_written;
        result.created_files.push(content_files.content.clone());

        // Write L1 content file
        let bytes_written = self.write_file(&content_files.content_l1, &content_l1).await?;
        result.files_created += 1;
        result.bytes_written += bytes_written;
        result.created_files.push(content_files.content_l1.clone());

        // Write L2 content file
        let bytes_written = self.write_file(&content_files.content_l2, &content_l2).await?;
        result.files_created += 1;
        result.bytes_written += bytes_written;
        result.created_files.push(content_files.content_l2.clone());

        debug!(
            "Successfully wrote {} content files for row {} ({} bytes total)",
            result.files_created, row_number, result.bytes_written
        );

        Ok(result)
    }

    /// Write content files for chunked data
    /// 
    /// Creates content files for chunked file data with proper L1/L2 context.
    /// 
    /// # Arguments
    /// * `table_name` - Name of the chunked table
    /// * `chunked_files` - List of chunked file data
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<ContentWriteResult>` - Write operation results or error
    #[instrument(skip(self, chunked_files), fields(table_name = %table_name, chunk_count = chunked_files.len()))]
    pub async fn write_chunked_content_files(
        &self,
        table_name: &str,
        chunked_files: &[ChunkedFile],
    ) -> TaskGeneratorResult<ContentWriteResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Starting chunked content file writing for table '{}' with {} chunks", table_name, chunked_files.len());
        
        // Validate configuration
        self.config.validate()?;
        
        // Ensure output directory exists
        self.ensure_output_directory().await?;
        
        let mut result = ContentWriteResult {
            files_created: 0,
            rows_processed: 0,
            bytes_written: 0,
            processing_time_ms: 0,
            created_files: Vec::new(),
            warnings: Vec::new(),
        };

        // Process chunks in batches
        let batch_size = self.config.max_concurrent_writes;

        for batch in chunked_files.chunks(batch_size) {
            let mut batch_tasks = Vec::new();

            for chunk in batch {
                let task = self.write_chunk_files(table_name, chunk);
                batch_tasks.push(task);
            }

            // Execute batch concurrently
            let batch_results = futures::future::join_all(batch_tasks).await;

            // Process batch results
            for batch_result in batch_results {
                match batch_result {
                    Ok(chunk_result) => {
                        result.files_created += chunk_result.files_created;
                        result.bytes_written += chunk_result.bytes_written;
                        result.created_files.extend(chunk_result.created_files);
                        result.rows_processed += 1;
                    }
                    Err(e) => {
                        warn!("Failed to write files for chunk: {}", e);
                        result.warnings.push(format!("Chunk processing failed: {}", e));
                    }
                }
            }
        }

        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        info!(
            "Chunked content file writing completed: {} files created, {} chunks processed, {} bytes written in {}ms",
            result.files_created, result.rows_processed, result.bytes_written, result.processing_time_ms
        );

        Ok(result)
    }

    /// Write content files for a single chunk
    async fn write_chunk_files(
        &self,
        table_name: &str,
        chunk: &ChunkedFile,
    ) -> TaskGeneratorResult<RowWriteResult> {
        debug!("Writing content files for chunk {} of file {}", chunk.chunk_number, chunk.original_file_id);

        // Generate file paths for chunk
        let content_files = self.generate_chunk_file_paths(table_name, chunk)?;

        let mut result = RowWriteResult {
            files_created: 0,
            bytes_written: 0,
            created_files: Vec::new(),
        };

        // Write content file
        let bytes_written = self.write_file(&content_files.content, &chunk.content).await?;
        result.files_created += 1;
        result.bytes_written += bytes_written;
        result.created_files.push(content_files.content.clone());

        // Write L1 content file
        let bytes_written = self.write_file(&content_files.content_l1, &chunk.content_l1).await?;
        result.files_created += 1;
        result.bytes_written += bytes_written;
        result.created_files.push(content_files.content_l1.clone());

        // Write L2 content file
        let bytes_written = self.write_file(&content_files.content_l2, &chunk.content_l2).await?;
        result.files_created += 1;
        result.bytes_written += bytes_written;
        result.created_files.push(content_files.content_l2.clone());

        debug!(
            "Successfully wrote {} content files for chunk {} ({} bytes total)",
            result.files_created, chunk.chunk_number, result.bytes_written
        );

        Ok(result)
    }

    /// Ensure the output directory exists
    async fn ensure_output_directory(&self) -> TaskGeneratorResult<()> {
        if !self.config.output_dir.exists() {
            debug!("Creating output directory: {:?}", self.config.output_dir);
            fs::create_dir_all(&self.config.output_dir).await.map_err(|e| {
                TaskGeneratorError::ContentWriteFailed {
                    path: self.config.output_dir.display().to_string(),
                    cause: format!("Failed to create output directory: {}", e),
                }
            })?;
        }
        Ok(())
    }

    /// Write content to a file with proper error handling
    async fn write_file(&self, file_path: &Path, content: &str) -> TaskGeneratorResult<u64> {
        // Check if file exists and handle overwrite policy
        if file_path.exists() && !self.config.overwrite_existing {
            return Err(TaskGeneratorError::ContentWriteFailed {
                path: file_path.display().to_string(),
                cause: "File exists and overwrite is disabled".to_string(),
            });
        }

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    TaskGeneratorError::ContentWriteFailed {
                        path: parent.display().to_string(),
                        cause: format!("Failed to create parent directory: {}", e),
                    }
                })?;
            }
        }

        // Write file with buffered I/O
        let mut file = fs::File::create(file_path).await.map_err(|e| {
            TaskGeneratorError::ContentWriteFailed {
                path: file_path.display().to_string(),
                cause: format!("Failed to create file: {}", e),
            }
        })?;

        file.write_all(content.as_bytes()).await.map_err(|e| {
            TaskGeneratorError::ContentWriteFailed {
                path: file_path.display().to_string(),
                cause: format!("Failed to write content: {}", e),
            }
        })?;

        file.flush().await.map_err(|e| {
            TaskGeneratorError::ContentWriteFailed {
                path: file_path.display().to_string(),
                cause: format!("Failed to flush file: {}", e),
            }
        })?;

        Ok(content.len() as u64)
    }

    /// Generate file paths for content files
    fn generate_file_paths(
        &self,
        table_name: &str,
        row: &IngestedFile,
        row_number: usize,
    ) -> TaskGeneratorResult<ContentFiles> {
        let base_name = match &self.config.naming_pattern {
            ContentNamingPattern::TableBased => {
                format!("{}_{}", table_name, row_number)
            }
            ContentNamingPattern::FileBased => {
                let filename = Path::new(&row.filename)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                format!("{}_{}", filename, row_number)
            }
            ContentNamingPattern::Custom(pattern) => {
                pattern
                    .replace("{table_name}", table_name)
                    .replace("{row_number}", &row_number.to_string())
                    .replace("{filename}", &row.filename)
                    .replace("{file_id}", &row.file_id.to_string())
            }
        };

        let content_path = self.config.output_dir.join(format!("{}_Content.txt", base_name));
        let content_l1_path = self.config.output_dir.join(format!("{}_ContentL1.txt", base_name));
        let content_l2_path = self.config.output_dir.join(format!("{}_ContentL2.txt", base_name));

        Ok(ContentFiles::new(content_path, content_l1_path, content_l2_path))
    }

    /// Generate file paths for chunked content files
    fn generate_chunk_file_paths(
        &self,
        table_name: &str,
        chunk: &ChunkedFile,
    ) -> TaskGeneratorResult<ContentFiles> {
        let base_name = format!("{}_{}_{}", table_name, chunk.original_file_id, chunk.chunk_number);

        let content_path = self.config.output_dir.join(format!("{}_Content.txt", base_name));
        let content_l1_path = self.config.output_dir.join(format!("{}_ContentL1.txt", base_name));
        let content_l2_path = self.config.output_dir.join(format!("{}_ContentL2.txt", base_name));

        Ok(ContentFiles::new(content_path, content_l1_path, content_l2_path))
    }

    /// Generate L1 content (current content + context hint)
    fn generate_l1_content(&self, content: &str) -> String {
        format!("{}\n\n--- L1 Context ---\nThis is the primary content with L1 context level.", content)
    }

    /// Generate L2 content (current content + extended context hint)
    fn generate_l2_content(&self, content: &str) -> String {
        format!("{}\n\n--- L2 Context ---\nThis is the primary content with L2 extended context level.", content)
    }

    /// Get the configuration
    pub fn config(&self) -> &ContentWriteConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ContentWriteConfig) {
        self.config = config;
    }
}

/// Result of writing files for a single row or chunk
#[derive(Debug)]
struct RowWriteResult {
    /// Number of files created for this row
    files_created: usize,
    /// Total bytes written for this row
    bytes_written: u64,
    /// List of created file paths
    created_files: Vec<PathBuf>,
}

impl ContentWriteResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            files_created: 0,
            rows_processed: 0,
            bytes_written: 0,
            processing_time_ms: 0,
            created_files: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Check if any files were created
    pub fn has_files(&self) -> bool {
        self.files_created > 0
    }

    /// Get the average bytes per file
    pub fn average_bytes_per_file(&self) -> f64 {
        if self.files_created == 0 {
            0.0
        } else {
            self.bytes_written as f64 / self.files_created as f64
        }
    }

    /// Get the processing rate in files per second
    pub fn processing_rate_fps(&self) -> f64 {
        if self.processing_time_ms == 0 {
            0.0
        } else {
            (self.files_created as f64) / (self.processing_time_ms as f64 / 1000.0)
        }
    }

    /// Get the processing rate in rows per second
    pub fn processing_rate_rps(&self) -> f64 {
        if self.processing_time_ms == 0 {
            0.0
        } else {
            (self.rows_processed as f64) / (self.processing_time_ms as f64 / 1000.0)
        }
    }
}

impl Default for ContentWriteResult {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ContentWriteConfig {
    fn default() -> Self {
        Self::new(PathBuf::from(".raw_data_202509"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::models::IngestedFile;
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_test_ingested_file(file_id: i64, filename: &str, content: Option<String>) -> IngestedFile {
        IngestedFile {
            file_id,
            ingestion_id: 1,
            filepath: format!("src/{}", filename),
            filename: filename.to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: content.as_ref().map_or(0, |c| c.len() as i64),
            line_count: content.as_ref().map(|c| c.lines().count() as i32),
            word_count: content.as_ref().map(|c| c.split_whitespace().count() as i32),
            token_count: content.as_ref().map(|c| (c.split_whitespace().count() as f32 * 0.75) as i32),
            content_text: content,
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: format!("src/{}", filename),
            absolute_path: format!("/tmp/src/{}", filename),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_content_write_config_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());

        assert_eq!(config.output_dir, temp_dir.path());
        assert!(config.create_subdirs);
        assert!(matches!(config.naming_pattern, ContentNamingPattern::TableBased));
        assert!(config.overwrite_existing);
        assert_eq!(config.buffer_size, 64 * 1024);
        assert_eq!(config.max_concurrent_writes, 10);
    }

    #[test]
    fn test_content_write_config_builder() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf())
            .with_subdirs(false)
            .with_naming_pattern(ContentNamingPattern::FileBased)
            .with_overwrite(false)
            .with_buffer_size(32 * 1024)
            .with_max_concurrent_writes(5);

        assert!(!config.create_subdirs);
        assert!(matches!(config.naming_pattern, ContentNamingPattern::FileBased));
        assert!(!config.overwrite_existing);
        assert_eq!(config.buffer_size, 32 * 1024);
        assert_eq!(config.max_concurrent_writes, 5);
    }

    #[test]
    fn test_content_write_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        
        // Valid configuration
        let valid_config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        assert!(valid_config.validate().is_ok());

        // Invalid buffer size
        let invalid_config = ContentWriteConfig::new(temp_dir.path().to_path_buf())
            .with_buffer_size(0);
        assert!(invalid_config.validate().is_err());

        // Invalid max concurrent writes
        let invalid_config = ContentWriteConfig::new(temp_dir.path().to_path_buf())
            .with_max_concurrent_writes(0);
        assert!(invalid_config.validate().is_err());
    }

    #[tokio::test]
    async fn test_content_file_writer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        assert_eq!(writer.config().output_dir, temp_dir.path());
    }

    #[tokio::test]
    async fn test_write_content_files_empty_list() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        let result = writer.write_content_files("TEST_TABLE", &[]).await.unwrap();

        assert_eq!(result.files_created, 0);
        assert_eq!(result.rows_processed, 0);
        assert_eq!(result.bytes_written, 0);
        assert!(result.created_files.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_write_content_files_with_content() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        let rows = vec![
            create_test_ingested_file(1, "main.rs", Some("fn main() {}".to_string())),
            create_test_ingested_file(2, "lib.rs", Some("pub mod test;".to_string())),
        ];

        let result = writer.write_content_files("TEST_TABLE", &rows).await.unwrap();

        assert_eq!(result.files_created, 6); // 3 files per row * 2 rows
        assert_eq!(result.rows_processed, 2);
        assert!(result.bytes_written > 0);
        assert_eq!(result.created_files.len(), 6);
        assert!(result.warnings.is_empty());

        // Verify files were created
        for file_path in &result.created_files {
            assert!(file_path.exists(), "File should exist: {:?}", file_path);
        }
    }

    #[tokio::test]
    async fn test_write_content_files_skip_empty_content() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        let rows = vec![
            create_test_ingested_file(1, "main.rs", Some("fn main() {}".to_string())),
            create_test_ingested_file(2, "empty.rs", None), // No content
            create_test_ingested_file(3, "lib.rs", Some("pub mod test;".to_string())),
        ];

        let result = writer.write_content_files("TEST_TABLE", &rows).await.unwrap();

        assert_eq!(result.files_created, 6); // 3 files per row * 2 rows (skipping empty)
        assert_eq!(result.rows_processed, 3); // All rows processed, but empty one skipped
        assert!(result.bytes_written > 0);
        assert_eq!(result.created_files.len(), 6);
    }

    #[tokio::test]
    async fn test_write_row_files_individual() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        let row = create_test_ingested_file(1, "test.rs", Some("fn test() {}".to_string()));

        let result = writer.write_row_files("TEST_TABLE", &row, 1).await.unwrap();

        assert_eq!(result.files_created, 3);
        assert!(result.bytes_written > 0);
        assert_eq!(result.created_files.len(), 3);

        // Verify file names
        let expected_files = vec![
            "TEST_TABLE_1_Content.txt",
            "TEST_TABLE_1_ContentL1.txt", 
            "TEST_TABLE_1_ContentL2.txt",
        ];

        for expected_file in expected_files {
            let expected_path = temp_dir.path().join(expected_file);
            assert!(result.created_files.contains(&expected_path), 
                    "Should contain file: {:?}", expected_path);
            assert!(expected_path.exists(), "File should exist: {:?}", expected_path);
        }
    }

    #[tokio::test]
    async fn test_write_chunked_content_files() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        let chunked_files = vec![
            ChunkedFile {
                original_file_id: 1,
                chunk_number: 0,
                content: "chunk 0 content".to_string(),
                content_l1: "chunk 0 content with L1".to_string(),
                content_l2: "chunk 0 content with L2".to_string(),
                line_count: 5,
                original_filepath: "src/main.rs".to_string(),
            },
            ChunkedFile {
                original_file_id: 1,
                chunk_number: 1,
                content: "chunk 1 content".to_string(),
                content_l1: "chunk 1 content with L1".to_string(),
                content_l2: "chunk 1 content with L2".to_string(),
                line_count: 5,
                original_filepath: "src/main.rs".to_string(),
            },
        ];

        let result = writer.write_chunked_content_files("CHUNKED_TABLE", &chunked_files).await.unwrap();

        assert_eq!(result.files_created, 6); // 3 files per chunk * 2 chunks
        assert_eq!(result.rows_processed, 2);
        assert!(result.bytes_written > 0);
        assert_eq!(result.created_files.len(), 6);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_content_write_result_calculations() {
        let mut result = ContentWriteResult::new();
        result.files_created = 10;
        result.bytes_written = 1000;
        result.processing_time_ms = 2000; // 2 seconds

        assert!(result.has_files());
        assert_eq!(result.average_bytes_per_file(), 100.0);
        assert_eq!(result.processing_rate_fps(), 5.0); // 10 files / 2 seconds
    }

    #[test]
    fn test_naming_patterns() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test table-based naming
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf())
            .with_naming_pattern(ContentNamingPattern::TableBased);
        let writer = ContentFileWriter::new(config);
        
        let row = create_test_ingested_file(1, "test.rs", Some("content".to_string()));
        let files = writer.generate_file_paths("TABLE", &row, 5).unwrap();
        
        assert!(files.content.file_name().unwrap().to_str().unwrap().starts_with("TABLE_5"));

        // Test file-based naming
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf())
            .with_naming_pattern(ContentNamingPattern::FileBased);
        let writer = ContentFileWriter::new(config);
        
        let files = writer.generate_file_paths("TABLE", &row, 5).unwrap();
        assert!(files.content.file_name().unwrap().to_str().unwrap().starts_with("test_5"));

        // Test custom naming
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf())
            .with_naming_pattern(ContentNamingPattern::Custom("{file_id}_{filename}".to_string()));
        let writer = ContentFileWriter::new(config);
        
        let files = writer.generate_file_paths("TABLE", &row, 5).unwrap();
        assert!(files.content.file_name().unwrap().to_str().unwrap().starts_with("1_test.rs"));
    }

    #[test]
    fn test_l1_l2_content_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        let writer = ContentFileWriter::new(config);

        let content = "Original content";
        let l1_content = writer.generate_l1_content(content);
        let l2_content = writer.generate_l2_content(content);

        assert!(l1_content.contains("Original content"));
        assert!(l1_content.contains("L1 Context"));
        
        assert!(l2_content.contains("Original content"));
        assert!(l2_content.contains("L2 Context"));
        
        assert_ne!(l1_content, l2_content);
    }

    #[test]
    fn test_basic_functionality() {
        // Test basic data structures without external dependencies
        let temp_dir = TempDir::new().unwrap();
        let config = ContentWriteConfig::new(temp_dir.path().to_path_buf());
        
        assert!(config.validate().is_ok());
        
        let writer = ContentFileWriter::new(config);
        assert_eq!(writer.config().buffer_size, 64 * 1024);
        
        let mut result = ContentWriteResult::new();
        assert!(!result.has_files());
        assert_eq!(result.average_bytes_per_file(), 0.0);
        
        result.files_created = 5;
        result.bytes_written = 500;
        assert!(result.has_files());
        assert_eq!(result.average_bytes_per_file(), 100.0);
        
        println!("✅ Basic functionality test passed");
    }
}