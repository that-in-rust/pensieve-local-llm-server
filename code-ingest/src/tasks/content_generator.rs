//! Content File Generator for Task List Generation
//!
//! This module provides the ContentGenerator struct that handles creating
//! three-tier content files (A, L1, L2) for systematic analysis workflows.
//! It supports both basic file-level generation and chunked content generation.

use crate::error::{TaskError, TaskResult};
use crate::tasks::models::{
    ContentFileReference, ContentFileType, GenerationConfig,
};
use crate::tasks::output_directory_manager::{OutputDirectoryManager, ConflictResolution};
use sqlx::{PgPool, Row};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info, warn};

/// Content generator for creating A/B/C analysis files
#[derive(Clone, Debug)]
pub struct ContentGenerator {
    /// Database connection pool
    db_pool: Arc<PgPool>,
    /// Output directory for content files
    output_dir: PathBuf,
    /// Configuration for generation
    config: GenerationConfig,
    /// Output directory manager
    dir_manager: OutputDirectoryManager,
}

/// Set of three content files for analysis (A, L1, L2)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContentFileSet {
    /// Primary content file (A)
    pub content_a: ContentFileReference,
    /// L1 context file (B)
    pub content_l1: ContentFileReference,
    /// L2 context file (C)
    pub content_l2: ContentFileReference,
    /// Row number from database
    pub row_number: usize,
    /// Table name this content came from
    pub table_name: String,
}

/// Metadata extracted from a database row for content generation
#[derive(Debug, Clone)]
pub struct RowData {
    pub file_id: Option<i64>,
    pub filepath: Option<String>,
    pub filename: Option<String>,
    pub extension: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub line_count: Option<i32>,
    pub word_count: Option<i32>,
    pub content_text: Option<String>,
    pub file_type: Option<String>,
    pub relative_path: Option<String>,
    pub absolute_path: Option<String>,
    pub chunk_number: Option<i32>,
    pub chunk_start_line: Option<i32>,
    pub chunk_end_line: Option<i32>,
    pub parent_filepath: Option<String>,
}

impl ContentGenerator {
    /// Create a new ContentGenerator
    ///
    /// # Arguments
    /// * `db_pool` - PostgreSQL connection pool
    /// * `output_dir` - Directory where content files will be created
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// * `Self` - New ContentGenerator instance
    pub fn new(db_pool: Arc<PgPool>, output_dir: PathBuf, config: GenerationConfig) -> Self {
        let dir_manager = OutputDirectoryManager::from_generation_config(&config);
        Self {
            db_pool,
            output_dir,
            config,
            dir_manager,
        }
    }

    /// Generate content files for all rows in a table
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to process
    ///
    /// # Returns
    /// * `TaskResult<Vec<ContentFileSet>>` - List of created content file sets
    pub async fn generate_all_content_files(
        &self,
        table_name: &str,
    ) -> TaskResult<Vec<ContentFileSet>> {
        debug!("Generating content files for table: {}", table_name);

        // Validate table name
        self.validate_table_name(table_name)?;

        // Ensure output directory exists
        self.ensure_output_directory().await?;

        // Query all rows from the table
        let rows = self.query_table_rows(table_name).await?;
        info!("Retrieved {} rows from table '{}'", rows.len(), table_name);

        // Generate content files for each row
        let mut content_file_sets = Vec::new();
        for (row_idx, row_data) in rows.iter().enumerate() {
            let row_number = row_idx + 1; // 1-based numbering

            let content_set = if self.config.enable_chunking {
                self.generate_chunked_content_files(row_data, row_number, table_name)
                    .await?
            } else {
                self.generate_basic_content_files(row_data, row_number, table_name)
                    .await?
            };

            content_file_sets.push(content_set);
        }

        info!(
            "Generated {} content file sets for table '{}'",
            content_file_sets.len(),
            table_name
        );
        Ok(content_file_sets)
    }

    /// Generate basic content files (A, L1, L2) for a single row
    ///
    /// # Arguments
    /// * `row_data` - Database row data
    /// * `row_number` - Row number (1-based)
    /// * `table_name` - Source table name
    ///
    /// # Returns
    /// * `TaskResult<ContentFileSet>` - Created content file set
    pub async fn generate_basic_content_files(
        &self,
        row_data: &RowData,
        row_number: usize,
        table_name: &str,
    ) -> TaskResult<ContentFileSet> {
        debug!(
            "Generating basic content files for row {} from table '{}'",
            row_number, table_name
        );

        // Generate file paths using the naming convention and directory manager
        let base_name = format!("{}_{}_Content", table_name, row_number);
        let content_a_path = self.dir_manager.get_output_path(
            &format!("{}.txt", base_name), 
            Some(table_name)
        );
        let content_l1_path = self.dir_manager.get_output_path(
            &format!("{}_L1.txt", base_name), 
            Some(table_name)
        );
        let content_l2_path = self.dir_manager.get_output_path(
            &format!("{}_L2.txt", base_name), 
            Some(table_name)
        );

        // Get raw content
        let raw_content = row_data.content_text.as_deref().unwrap_or("");

        // Generate L1 context (immediate file context)
        let l1_context = self.generate_l1_context(raw_content, row_data);

        // Generate L2 context (architectural context)
        let l2_context = self.generate_l2_context(raw_content, row_data);

        // Write content files
        self.write_content_file(&content_a_path, raw_content).await?;
        self.write_content_file(&content_l1_path, &l1_context).await?;
        self.write_content_file(&content_l2_path, &l2_context).await?;

        // Create content file references
        let content_a = ContentFileReference::new(
            content_a_path,
            "A".to_string(),
            "Primary content file".to_string(),
            ContentFileType::Content,
        );

        let content_l1 = ContentFileReference::new(
            content_l1_path,
            "B".to_string(),
            "L1 context file - immediate file context".to_string(),
            ContentFileType::L1Context,
        );

        let content_l2 = ContentFileReference::new(
            content_l2_path,
            "C".to_string(),
            "L2 context file - architectural context".to_string(),
            ContentFileType::L2Context,
        );

        debug!(
            "Created basic content files: A={}, B={}, C={}",
            content_a.file_path.display(),
            content_l1.file_path.display(),
            content_l2.file_path.display()
        );

        Ok(ContentFileSet {
            content_a,
            content_l1,
            content_l2,
            row_number,
            table_name: table_name.to_string(),
        })
    }

    /// Generate chunked content files for a single row
    ///
    /// # Arguments
    /// * `row_data` - Database row data (from chunked table)
    /// * `row_number` - Row number (1-based)
    /// * `table_name` - Source table name
    ///
    /// # Returns
    /// * `TaskResult<ContentFileSet>` - Created content file set
    pub async fn generate_chunked_content_files(
        &self,
        row_data: &RowData,
        row_number: usize,
        table_name: &str,
    ) -> TaskResult<ContentFileSet> {
        debug!(
            "Generating chunked content files for row {} from table '{}'",
            row_number, table_name
        );

        // Validate chunk data
        self.validate_chunk_data(row_data)?;

        // For chunked tables, use the chunked table name in file paths
        let effective_table_name = self.config.effective_table_name();
        let base_name = format!("{}_{}_Content", effective_table_name, row_number);
        let content_a_path = self.dir_manager.get_output_path(
            &format!("{}.txt", base_name), 
            Some(&effective_table_name)
        );
        let content_l1_path = self.dir_manager.get_output_path(
            &format!("{}_L1.txt", base_name), 
            Some(&effective_table_name)
        );
        let content_l2_path = self.dir_manager.get_output_path(
            &format!("{}_L2.txt", base_name), 
            Some(&effective_table_name)
        );

        // Get chunk content
        let chunk_content = row_data.content_text.as_deref().unwrap_or("");

        // Generate L1 context using chunking engine output
        let l1_context = self.generate_chunked_l1_context(row_data).await?;

        // Generate L2 context using chunking engine output
        let l2_context = self.generate_chunked_l2_context(row_data).await?;

        // Write content files with validation
        self.write_content_file_with_validation(&content_a_path, chunk_content, "Primary chunk content").await?;
        self.write_content_file_with_validation(&content_l1_path, &l1_context, "L1 context content").await?;
        self.write_content_file_with_validation(&content_l2_path, &l2_context, "L2 context content").await?;

        // Create content file references with enhanced descriptions
        let chunk_num = row_data.chunk_number.unwrap_or(1);
        let start_line = row_data.chunk_start_line.unwrap_or(1);
        let end_line = row_data.chunk_end_line.unwrap_or(1);
        let line_count = end_line - start_line + 1;

        let content_a = ContentFileReference::new(
            content_a_path,
            "A".to_string(),
            format!(
                "Chunk {} content: {} lines ({}-{}) from {}",
                chunk_num, line_count, start_line, end_line,
                row_data.filepath.as_deref().unwrap_or("unknown")
            ),
            ContentFileType::Content,
        );

        let content_l1 = ContentFileReference::new(
            content_l1_path,
            "B".to_string(),
            format!(
                "L1 context for chunk {}: previous + current + next chunk concatenation",
                chunk_num
            ),
            ContentFileType::L1Context,
        );

        let content_l2 = ContentFileReference::new(
            content_l2_path,
            "C".to_string(),
            format!(
                "L2 context for chunk {}: ±2 chunks concatenation for architectural context",
                chunk_num
            ),
            ContentFileType::L2Context,
        );

        debug!(
            "Created chunked content files: A={}, B={}, C={}",
            content_a.file_path.display(),
            content_l1.file_path.display(),
            content_l2.file_path.display()
        );

        Ok(ContentFileSet {
            content_a,
            content_l1,
            content_l2,
            row_number,
            table_name: effective_table_name,
        })
    }

    /// Ensure the output directory exists
    async fn ensure_output_directory(&self) -> TaskResult<()> {
        self.dir_manager.ensure_directory_structure().await?;
        debug!("Ensured output directory exists: {}", self.output_dir.display());
        Ok(())
    }

    /// Query all rows from a table
    async fn query_table_rows(&self, table_name: &str) -> TaskResult<Vec<RowData>> {
        let query = if self.config.enable_chunking {
            // Query chunked table with additional chunk columns
            format!(
                "SELECT file_id, filepath, filename, extension, file_size_bytes, 
                        line_count, word_count, content_text, file_type, 
                        relative_path, absolute_path, chunk_number, 
                        chunk_start_line, chunk_end_line, parent_filepath
                 FROM \"{}\" 
                 ORDER BY file_id, chunk_number",
                self.config.effective_table_name()
            )
        } else {
            // Query regular table
            format!(
                "SELECT file_id, filepath, filename, extension, file_size_bytes, 
                        line_count, word_count, content_text, file_type, 
                        relative_path, absolute_path
                 FROM \"{}\" 
                 ORDER BY file_id",
                table_name
            )
        };

        debug!("Executing query: {}", query);

        let rows = sqlx::query(&query)
            .fetch_all(&*self.db_pool)
            .await
            .map_err(|e| TaskError::QueryResultProcessingFailed {
                cause: format!("Failed to fetch rows from table '{}': {}", table_name, e),
                suggestion: "Check database connection and table accessibility".to_string(),
                source: Some(Box::new(e)),
            })?;

        let mut row_data_list = Vec::new();
        for row in rows {
            let row_data = RowData {
                file_id: row.try_get("file_id").ok(),
                filepath: row.try_get("filepath").ok(),
                filename: row.try_get("filename").ok(),
                extension: row.try_get("extension").ok(),
                file_size_bytes: row.try_get("file_size_bytes").ok(),
                line_count: row.try_get("line_count").ok(),
                word_count: row.try_get("word_count").ok(),
                content_text: row.try_get("content_text").ok(),
                file_type: row.try_get("file_type").ok(),
                relative_path: row.try_get("relative_path").ok(),
                absolute_path: row.try_get("absolute_path").ok(),
                chunk_number: if self.config.enable_chunking {
                    row.try_get("chunk_number").ok()
                } else {
                    None
                },
                chunk_start_line: if self.config.enable_chunking {
                    row.try_get("chunk_start_line").ok()
                } else {
                    None
                },
                chunk_end_line: if self.config.enable_chunking {
                    row.try_get("chunk_end_line").ok()
                } else {
                    None
                },
                parent_filepath: if self.config.enable_chunking {
                    row.try_get("parent_filepath").ok()
                } else {
                    None
                },
            };
            row_data_list.push(row_data);
        }

        Ok(row_data_list)
    }

    /// Write content to a file
    async fn write_content_file(&self, file_path: &Path, content: &str) -> TaskResult<()> {
        fs::write(file_path, content).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: file_path.display().to_string(),
                cause: format!("Failed to write content file: {}", e),
                suggestion: "Check file permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        debug!("Wrote content file: {}", file_path.display());
        Ok(())
    }

    /// Write content to a file with validation and error handling
    async fn write_content_file_with_validation(
        &self,
        file_path: &Path,
        content: &str,
        content_type: &str,
    ) -> TaskResult<()> {
        // Validate content is not empty
        if content.trim().is_empty() {
            warn!(
                "Writing empty {} to file: {}",
                content_type,
                file_path.display()
            );
        }

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: parent.display().to_string(),
                    cause: format!("Failed to create parent directory: {}", e),
                    suggestion: "Check directory permissions and available disk space".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }

        // Write content with UTF-8 encoding
        fs::write(file_path, content.as_bytes()).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: file_path.display().to_string(),
                cause: format!("Failed to write {} file: {}", content_type, e),
                suggestion: "Check file permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        debug!(
            "Wrote {} file: {} ({} bytes)",
            content_type,
            file_path.display(),
            content.len()
        );
        Ok(())
    }

    /// Validate chunk data for chunked content generation
    fn validate_chunk_data(&self, row_data: &RowData) -> TaskResult<()> {
        if row_data.chunk_number.is_none() {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Chunk number is missing for chunked content generation".to_string(),
                suggestion: "Ensure the row data comes from a properly chunked table".to_string(),
            });
        }

        if row_data.chunk_start_line.is_none() || row_data.chunk_end_line.is_none() {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Chunk line range is missing for chunked content generation".to_string(),
                suggestion: "Ensure the chunked table has proper start_line and end_line columns".to_string(),
            });
        }

        let start_line = row_data.chunk_start_line.unwrap();
        let end_line = row_data.chunk_end_line.unwrap();

        if start_line > end_line {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: format!("Invalid chunk line range: start_line ({}) > end_line ({})", start_line, end_line),
                suggestion: "Check the chunking algorithm for proper line range calculation".to_string(),
            });
        }

        if start_line < 1 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: format!("Invalid start_line: {} (must be >= 1)", start_line),
                suggestion: "Line numbers should be 1-based".to_string(),
            });
        }

        Ok(())
    }

    /// Generate L1 context (immediate file context)
    fn generate_l1_context(&self, content: &str, row_data: &RowData) -> String {
        let mut context = String::new();

        // Add file metadata header
        context.push_str("# L1 Context: Immediate File Context\n\n");
        context.push_str("## File Information\n\n");

        if let Some(filepath) = &row_data.filepath {
            context.push_str(&format!("- **File Path**: `{}`\n", filepath));

            // Extract directory information
            if let Some(parent) = Path::new(filepath).parent() {
                context.push_str(&format!("- **Directory**: `{}`\n", parent.display()));
            }
        }

        if let Some(filename) = &row_data.filename {
            context.push_str(&format!("- **Filename**: `{}`\n", filename));
        }

        if let Some(extension) = &row_data.extension {
            context.push_str(&format!("- **Extension**: `{}`\n", extension));
        }

        if let Some(file_type) = &row_data.file_type {
            context.push_str(&format!("- **File Type**: `{}`\n", file_type));
        }

        if let Some(line_count) = row_data.line_count {
            context.push_str(&format!("- **Line Count**: {}\n", line_count));
        }

        if let Some(word_count) = row_data.word_count {
            context.push_str(&format!("- **Word Count**: {}\n", word_count));
        }

        if let Some(file_size) = row_data.file_size_bytes {
            context.push_str(&format!("- **File Size**: {} bytes\n", file_size));
        }

        context.push_str("\n## Import/Include Analysis\n\n");

        // Analyze imports/includes based on file extension
        let imports = self.extract_imports(content, row_data.extension.as_deref());
        if imports.is_empty() {
            context.push_str("- No imports/includes detected\n");
        } else {
            context.push_str(&format!("### Detected {} import(s)/include(s):\n\n", imports.len()));
            for import in imports {
                context.push_str(&format!("- `{}`\n", import));
            }
        }

        context.push_str("\n## Original File Content\n\n");
        context.push_str("```\n");
        context.push_str(content);
        context.push_str("\n```\n");

        context
    }

    /// Generate L2 context (architectural context)
    fn generate_l2_context(&self, content: &str, row_data: &RowData) -> String {
        let mut context = String::new();

        // Add architectural context header
        context.push_str("# L2 Context: Architectural Context\n\n");
        context.push_str("## Package/Module Structure\n\n");

        if let Some(filepath) = &row_data.filepath {
            // Analyze package/module structure from path
            let path_components: Vec<&str> = Path::new(filepath)
                .components()
                .filter_map(|c| c.as_os_str().to_str())
                .collect();

            context.push_str(&format!("- **Path Depth**: {} levels\n", path_components.len()));
            context.push_str("- **Path Components**:\n");
            for (i, component) in path_components.iter().enumerate() {
                context.push_str(&format!("  {}. `{}`\n", i + 1, component));
            }
        }

        context.push_str("\n## Architectural Patterns\n\n");

        // Detect architectural patterns based on content and path
        let patterns = self.detect_architectural_patterns(content, row_data);
        if patterns.is_empty() {
            context.push_str("- No specific architectural patterns detected\n");
        } else {
            for pattern in patterns {
                context.push_str(&format!("- {}\n", pattern));
            }
        }

        context.push_str("\n## Cross-Module Relationships\n\n");

        // Analyze potential cross-module relationships
        let relationships = self.analyze_cross_module_relationships(content, row_data);
        if relationships.is_empty() {
            context.push_str("- No cross-module relationships detected\n");
        } else {
            for relationship in relationships {
                context.push_str(&format!("- {}\n", relationship));
            }
        }

        context.push_str("\n## Original File Content\n\n");
        context.push_str("```\n");
        context.push_str(content);
        context.push_str("\n```\n");

        context
    }

    /// Generate L1 context for chunked content
    async fn generate_chunked_l1_context(&self, row_data: &RowData) -> TaskResult<String> {
        let mut context = String::new();

        context.push_str("# L1 Context: Chunk Context (Previous + Current + Next)\n\n");
        context.push_str("## Chunk Information\n\n");

        let chunk_number = row_data.chunk_number.unwrap_or(1);
        let start_line = row_data.chunk_start_line.unwrap_or(1);
        let end_line = row_data.chunk_end_line.unwrap_or(1);

        context.push_str(&format!("- **Current Chunk**: {}\n", chunk_number));
        context.push_str(&format!("- **Line Range**: {}-{} ({} lines)\n", 
            start_line, end_line, end_line - start_line + 1));

        if let Some(filepath) = &row_data.filepath {
            context.push_str(&format!("- **File Path**: `{}`\n", filepath));
        }

        if let Some(parent_filepath) = &row_data.parent_filepath {
            context.push_str(&format!("- **Original File**: `{}`\n", parent_filepath));
        }

        // Get chunk context information
        let context_info = self.get_chunk_context_info(row_data, 1).await?;
        context.push_str(&format!("- **L1 Context Range**: Chunks {}-{} ({} chunks)\n", 
            context_info.start_chunk, context_info.end_chunk, context_info.chunk_count));

        // Query for previous and next chunks to build L1 context
        let l1_content = self.query_chunk_context(row_data, 1).await?;
        
        context.push_str("\n## L1 Context Content\n\n");
        context.push_str("*This section contains the concatenated content of the previous chunk, current chunk, and next chunk to provide immediate context for analysis.*\n\n");
        
        if l1_content.trim().is_empty() {
            context.push_str("*No context content available - this may be a single chunk file.*\n");
        } else {
            context.push_str("```\n");
            context.push_str(&l1_content);
            context.push_str("\n```\n");
        }

        Ok(context)
    }

    /// Generate L2 context for chunked content
    async fn generate_chunked_l2_context(&self, row_data: &RowData) -> TaskResult<String> {
        let mut context = String::new();

        context.push_str("# L2 Context: Extended Chunk Context (±2 Chunks)\n\n");
        context.push_str("## Extended Chunk Information\n\n");

        let chunk_number = row_data.chunk_number.unwrap_or(1);
        let start_line = row_data.chunk_start_line.unwrap_or(1);
        let end_line = row_data.chunk_end_line.unwrap_or(1);

        context.push_str(&format!("- **Current Chunk**: {}\n", chunk_number));
        context.push_str(&format!("- **Line Range**: {}-{} ({} lines)\n", 
            start_line, end_line, end_line - start_line + 1));

        // Get chunk context information
        let context_info = self.get_chunk_context_info(row_data, 2).await?;
        context.push_str(&format!("- **L2 Context Range**: Chunks {}-{} ({} chunks)\n", 
            context_info.start_chunk, context_info.end_chunk, context_info.chunk_count));

        if let Some(filepath) = &row_data.filepath {
            context.push_str(&format!("- **File Path**: `{}`\n", filepath));
        }

        if let Some(parent_filepath) = &row_data.parent_filepath {
            context.push_str(&format!("- **Original File**: `{}`\n", parent_filepath));
        }

        // Query for ±2 chunks to build L2 context
        let l2_content = self.query_chunk_context(row_data, 2).await?;
        
        context.push_str("\n## L2 Context Content\n\n");
        context.push_str("*This section contains the concatenated content of ±2 chunks around the current chunk to provide broader architectural context for analysis.*\n\n");
        
        if l2_content.trim().is_empty() {
            context.push_str("*No extended context content available - this may be a small file with few chunks.*\n");
        } else {
            context.push_str("```\n");
            context.push_str(&l2_content);
            context.push_str("\n```\n");
        }

        Ok(context)
    }



    /// Get information about chunk context range
    async fn get_chunk_context_info(&self, row_data: &RowData, context_level: i32) -> TaskResult<ChunkContextInfo> {
        let chunk_number = row_data.chunk_number.unwrap_or(1);
        let file_id = row_data.file_id.unwrap_or(0);
        
        let start_chunk = std::cmp::max(1, chunk_number - context_level);
        let end_chunk = chunk_number + context_level;

        // Query to get the actual range of available chunks
        let query = format!(
            "SELECT MIN(chunk_number) as min_chunk, MAX(chunk_number) as max_chunk, COUNT(*) as chunk_count
             FROM \"{}\" 
             WHERE file_id = $1 AND chunk_number BETWEEN $2 AND $3",
            self.config.effective_table_name()
        );

        let row = sqlx::query(&query)
            .bind(file_id)
            .bind(start_chunk)
            .bind(end_chunk)
            .fetch_one(&*self.db_pool)
            .await
            .map_err(|e| TaskError::QueryResultProcessingFailed {
                cause: format!("Failed to fetch chunk context info: {}", e),
                suggestion: "Check database connection and chunked table accessibility".to_string(),
                source: Some(Box::new(e)),
            })?;

        let actual_start: Option<i32> = row.try_get("min_chunk").ok().flatten();
        let actual_end: Option<i32> = row.try_get("max_chunk").ok().flatten();
        let count: i64 = row.try_get("chunk_count").unwrap_or(0);

        Ok(ChunkContextInfo {
            start_chunk: actual_start.unwrap_or(start_chunk),
            end_chunk: actual_end.unwrap_or(end_chunk),
            chunk_count: count as usize,
        })
    }

    /// Query chunk context (±N chunks around current)
    async fn query_chunk_context(&self, row_data: &RowData, context_level: i32) -> TaskResult<String> {
        let chunk_number = row_data.chunk_number.unwrap_or(1);
        let file_id = row_data.file_id.unwrap_or(0);
        
        let start_chunk = std::cmp::max(1, chunk_number - context_level);
        let end_chunk = chunk_number + context_level;

        let query = format!(
            "SELECT chunk_number, content_text, chunk_start_line, chunk_end_line
             FROM \"{}\" 
             WHERE file_id = $1 AND chunk_number BETWEEN $2 AND $3
             ORDER BY chunk_number",
            self.config.effective_table_name()
        );

        let rows = sqlx::query(&query)
            .bind(file_id)
            .bind(start_chunk)
            .bind(end_chunk)
            .fetch_all(&*self.db_pool)
            .await
            .map_err(|e| TaskError::QueryResultProcessingFailed {
                cause: format!("Failed to fetch chunk context: {}", e),
                suggestion: "Check database connection and chunked table accessibility".to_string(),
                source: Some(Box::new(e)),
            })?;

        if rows.is_empty() {
            return Ok(String::new());
        }

        let mut context_content = String::new();
        for (i, row) in rows.iter().enumerate() {
            let chunk_num: i32 = row.try_get("chunk_number").unwrap_or(0);
            let content: Option<String> = row.try_get("content_text").ok();
            let start_line: Option<i32> = row.try_get("chunk_start_line").ok();
            let end_line: Option<i32> = row.try_get("chunk_end_line").ok();
            
            if let Some(content) = content {
                // Add separator between chunks
                if i > 0 {
                    context_content.push_str("\n");
                }

                // Add chunk header with line information
                if let (Some(start), Some(end)) = (start_line, end_line) {
                    context_content.push_str(&format!(
                        "=== Chunk {} (lines {}-{}) ===\n", 
                        chunk_num, start, end
                    ));
                } else {
                    context_content.push_str(&format!("=== Chunk {} ===\n", chunk_num));
                }

                // Mark current chunk
                if chunk_num == chunk_number {
                    context_content.push_str(">>> CURRENT CHUNK <<<\n");
                }

                context_content.push_str(&content);
                
                // Add trailing newline if content doesn't end with one
                if !content.ends_with('\n') {
                    context_content.push_str("\n");
                }
            }
        }

        Ok(context_content)
    }

    /// Extract imports/includes from content based on file extension
    fn extract_imports(&self, content: &str, extension: Option<&str>) -> Vec<String> {
        let mut imports = Vec::new();

        match extension {
            Some("rs") => {
                // Rust imports
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("use ") || trimmed.starts_with("extern crate ") {
                        imports.push(trimmed.to_string());
                    }
                }
            }
            Some("py") => {
                // Python imports
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                        imports.push(trimmed.to_string());
                    }
                }
            }
            Some("js") | Some("ts") => {
                // JavaScript/TypeScript imports
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("import ") || trimmed.starts_with("const ") && trimmed.contains("require(") {
                        imports.push(trimmed.to_string());
                    }
                }
            }
            Some("c") | Some("cpp") | Some("h") | Some("hpp") => {
                // C/C++ includes
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("#include ") {
                        imports.push(trimmed.to_string());
                    }
                }
            }
            _ => {
                // Generic pattern matching for other languages
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.contains("import") || trimmed.contains("include") || trimmed.contains("require") {
                        imports.push(trimmed.to_string());
                    }
                }
            }
        }

        imports
    }

    /// Detect architectural patterns from content and metadata
    fn detect_architectural_patterns(&self, content: &str, row_data: &RowData) -> Vec<String> {
        let mut patterns = Vec::new();

        // Pattern detection based on file path
        if let Some(filepath) = &row_data.filepath {
            let path_str = filepath.to_lowercase();
            
            if path_str.contains("/src/") {
                patterns.push("Source code organization pattern".to_string());
            }
            if path_str.contains("/lib/") || path_str.contains("/libs/") {
                patterns.push("Library organization pattern".to_string());
            }
            if path_str.contains("/test/") || path_str.contains("/tests/") {
                patterns.push("Test organization pattern".to_string());
            }
            if path_str.contains("/bin/") {
                patterns.push("Binary/executable pattern".to_string());
            }
            if path_str.contains("/mod.rs") || path_str.contains("/lib.rs") || path_str.contains("/main.rs") {
                patterns.push("Rust module system pattern".to_string());
            }
        }

        // Pattern detection based on content
        let content_lower = content.to_lowercase();
        
        if content_lower.contains("struct") && content_lower.contains("impl") {
            patterns.push("Object-oriented design pattern".to_string());
        }
        if content_lower.contains("trait") || content_lower.contains("interface") {
            patterns.push("Interface/trait abstraction pattern".to_string());
        }
        if content_lower.contains("async") || content_lower.contains("await") {
            patterns.push("Asynchronous programming pattern".to_string());
        }
        if content_lower.contains("error") && (content_lower.contains("result") || content_lower.contains("option")) {
            patterns.push("Error handling pattern".to_string());
        }

        patterns
    }

    /// Analyze cross-module relationships
    fn analyze_cross_module_relationships(&self, content: &str, row_data: &RowData) -> Vec<String> {
        let mut relationships = Vec::new();

        // Analyze imports for cross-module dependencies
        let imports = self.extract_imports(content, row_data.extension.as_deref());
        
        for import in imports {
            if import.contains("::") || import.contains(".") {
                relationships.push(format!("External dependency: {}", import));
            }
        }

        // Analyze function calls and type usage
        if let Some(extension) = &row_data.extension {
            match extension.as_str() {
                "rs" => {
                    // Look for module path usage (e.g., crate::module::function)
                    for line in content.lines() {
                        if line.contains("crate::") || line.contains("super::") || line.contains("self::") {
                            relationships.push(format!("Module path usage: {}", line.trim()));
                        }
                    }
                }
                _ => {
                    // Generic analysis for other languages
                    for line in content.lines() {
                        if line.contains("::") || (line.contains(".") && !line.trim_start().starts_with("//")) {
                            relationships.push(format!("Potential cross-module reference: {}", line.trim()));
                        }
                    }
                }
            }
        }

        relationships
    }

    /// Validate table name to prevent SQL injection
    fn validate_table_name(&self, table_name: &str) -> TaskResult<()> {
        if table_name.is_empty() {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Table name cannot be empty".to_string(),
                suggestion: "Provide a valid table name".to_string(),
            });
        }

        // Basic validation - table names should be alphanumeric with underscores
        if !table_name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: format!("Invalid table name: {}", table_name),
                suggestion: "Table names should contain only alphanumeric characters and underscores".to_string(),
            });
        }

        Ok(())
    }

    /// Validate generated content files
    pub async fn validate_content_files(&self, content_set: &ContentFileSet) -> TaskResult<()> {
        // Check that all files exist and are readable
        let files = [
            &content_set.content_a.file_path,
            &content_set.content_l1.file_path,
            &content_set.content_l2.file_path,
        ];

        for file_path in &files {
            if !file_path.exists() {
                return Err(TaskError::TaskFileCreationFailed {
                    path: file_path.display().to_string(),
                    cause: "Content file does not exist after creation".to_string(),
                    suggestion: "Check file system permissions and disk space".to_string(),
                    source: None,
                });
            }

            // Check file is readable and not empty
            let metadata = fs::metadata(file_path).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: file_path.display().to_string(),
                    cause: format!("Cannot read file metadata: {}", e),
                    suggestion: "Check file permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            if metadata.len() == 0 {
                warn!("Content file is empty: {}", file_path.display());
            }
        }

        debug!("Validated content files for row {}", content_set.row_number);
        Ok(())
    }

    /// Clean up content files (for error recovery)
    pub async fn cleanup_content_files(&self, content_set: &ContentFileSet) -> TaskResult<()> {
        let files = [
            &content_set.content_a.file_path,
            &content_set.content_l1.file_path,
            &content_set.content_l2.file_path,
        ];

        for file_path in &files {
            if file_path.exists() {
                if let Err(e) = fs::remove_file(file_path).await {
                    warn!("Failed to cleanup content file {}: {}", file_path.display(), e);
                } else {
                    debug!("Cleaned up content file: {}", file_path.display());
                }
            }
        }

        Ok(())
    }

    /// Get content file statistics
    pub async fn get_content_statistics(&self, content_set: &ContentFileSet) -> TaskResult<ContentStatistics> {
        let mut stats = ContentStatistics::default();

        let files = [
            (&content_set.content_a.file_path, "A"),
            (&content_set.content_l1.file_path, "L1"),
            (&content_set.content_l2.file_path, "L2"),
        ];

        for (file_path, file_type) in &files {
            if file_path.exists() {
                let content = fs::read_to_string(file_path).await.map_err(|e| {
                    TaskError::TaskFileCreationFailed {
                        path: file_path.display().to_string(),
                        cause: format!("Failed to read content file for statistics: {}", e),
                        suggestion: "Check file permissions".to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;

                let file_stats = FileStatistics {
                    file_type: file_type.to_string(),
                    size_bytes: content.len(),
                    line_count: content.lines().count(),
                    word_count: content.split_whitespace().count(),
                    char_count: content.chars().count(),
                };

                match *file_type {
                    "A" => stats.content_a = Some(file_stats),
                    "L1" => stats.content_l1 = Some(file_stats),
                    "L2" => stats.content_l2 = Some(file_stats),
                    _ => {}
                }
            }
        }

        Ok(stats)
    }

    /// Get directory statistics from the output directory manager
    pub async fn get_output_directory_statistics(&self) -> TaskResult<crate::tasks::output_directory_manager::DirectoryStatistics> {
        self.dir_manager.get_directory_statistics().await
    }

    /// Clean up old content files
    pub async fn cleanup_old_content_files(&self) -> TaskResult<crate::tasks::output_directory_manager::CleanupResult> {
        self.dir_manager.cleanup_old_files().await
    }

    /// Organize content files into subdirectories
    pub async fn organize_content_files(&self) -> TaskResult<crate::tasks::output_directory_manager::OrganizationResult> {
        self.dir_manager.organize_files().await
    }

    /// Clear all content files from the output directory
    pub async fn clear_all_content_files(&self) -> TaskResult<crate::tasks::output_directory_manager::ClearResult> {
        self.dir_manager.clear_directory().await
    }

    /// Create a backup of the output directory
    pub async fn backup_output_directory(&self, backup_path: &Path) -> TaskResult<crate::tasks::output_directory_manager::BackupResult> {
        self.dir_manager.create_backup(backup_path).await
    }

    /// Validate and resolve file path conflicts
    pub async fn validate_output_path(
        &self,
        file_path: &Path,
        content_size: usize,
    ) -> TaskResult<PathBuf> {
        self.dir_manager.validate_and_resolve_path(
            file_path,
            content_size,
            ConflictResolution::BackupAndOverwrite,
        ).await
    }
}

/// Statistics for content files
#[derive(Debug, Clone, Default)]
pub struct ContentStatistics {
    pub content_a: Option<FileStatistics>,
    pub content_l1: Option<FileStatistics>,
    pub content_l2: Option<FileStatistics>,
}

/// Statistics for a single file
#[derive(Debug, Clone)]
pub struct FileStatistics {
    pub file_type: String,
    pub size_bytes: usize,
    pub line_count: usize,
    pub word_count: usize,
    pub char_count: usize,
}

/// Information about chunk context range
#[derive(Debug)]
struct ChunkContextInfo {
    start_chunk: i32,
    end_chunk: i32,
    chunk_count: usize,
}

impl ContentStatistics {
    /// Get total size across all files
    pub fn total_size_bytes(&self) -> usize {
        let mut total = 0;
        if let Some(ref stats) = self.content_a {
            total += stats.size_bytes;
        }
        if let Some(ref stats) = self.content_l1 {
            total += stats.size_bytes;
        }
        if let Some(ref stats) = self.content_l2 {
            total += stats.size_bytes;
        }
        total
    }

    /// Get total line count across all files
    pub fn total_line_count(&self) -> usize {
        let mut total = 0;
        if let Some(ref stats) = self.content_a {
            total += stats.line_count;
        }
        if let Some(ref stats) = self.content_l1 {
            total += stats.line_count;
        }
        if let Some(ref stats) = self.content_l2 {
            total += stats.line_count;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_row_data() -> RowData {
        RowData {
            file_id: Some(1),
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(50),
            word_count: Some(200),
            content_text: Some("fn main() {\n    println!(\"Hello, world!\");\n}".to_string()),
            file_type: Some("direct_text".to_string()),
            relative_path: Some("src/main.rs".to_string()),
            absolute_path: Some("/project/src/main.rs".to_string()),
            chunk_number: None,
            chunk_start_line: None,
            chunk_end_line: None,
            parent_filepath: None,
        }
    }

    fn create_test_config() -> GenerationConfig {
        GenerationConfig::new(
            "INGEST_20250928101039".to_string(),
            4,
            7,
            PathBuf::from("tasks.md"),
        )
    }

    #[tokio::test]
    async fn test_extract_rust_imports() {
        let content = r#"
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
extern crate tokio;

fn main() {
    println!("Hello, world!");
}
"#;

        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        
        // Create a mock database pool (this won't be used in this test)
        let db_pool = Arc::new(
            sqlx::PgPool::connect("postgresql://localhost/test")
                .await
                .unwrap_or_else(|_| panic!("Failed to create mock pool"))
        );
        
        let generator = ContentGenerator::new(
            db_pool,
            temp_dir.path().to_path_buf(),
            config,
        );

        let imports = generator.extract_imports(content, Some("rs"));
        
        assert_eq!(imports.len(), 3);
        assert!(imports.contains(&"use std::collections::HashMap;".to_string()));
        assert!(imports.contains(&"use serde::{Serialize, Deserialize};".to_string()));
        assert!(imports.contains(&"extern crate tokio;".to_string()));
    }

    #[tokio::test]
    async fn test_detect_architectural_patterns() {
        let content = r#"
struct MyStruct {
    field: String,
}

impl MyStruct {
    async fn process(&self) -> Result<(), Error> {
        // Implementation
        Ok(())
    }
}

trait MyTrait {
    fn method(&self);
}
"#;

        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        
        // Create a mock database pool
        let db_pool = Arc::new(
            sqlx::PgPool::connect("postgresql://localhost/test")
                .await
                .unwrap_or_else(|_| panic!("Failed to create mock pool"))
        );
        
        let generator = ContentGenerator::new(
            db_pool,
            temp_dir.path().to_path_buf(),
            config,
        );

        let mut row_data = create_test_row_data();
        row_data.filepath = Some("src/lib.rs".to_string());

        let patterns = generator.detect_architectural_patterns(content, &row_data);
        
        assert!(patterns.iter().any(|p| p.contains("Object-oriented design")));
        assert!(patterns.iter().any(|p| p.contains("Interface/trait abstraction")));
        assert!(patterns.iter().any(|p| p.contains("Asynchronous programming")));
        assert!(patterns.iter().any(|p| p.contains("Error handling")));
        assert!(patterns.iter().any(|p| p.contains("Source code organization")));
    }

    #[tokio::test]
    async fn test_validate_table_name() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config();
        
        // Create a mock database pool
        let db_pool = Arc::new(
            sqlx::PgPool::connect("postgresql://localhost/test")
                .await
                .unwrap_or_else(|_| panic!("Failed to create mock pool"))
        );
        
        let generator = ContentGenerator::new(
            db_pool,
            temp_dir.path().to_path_buf(),
            config,
        );

        // Valid table names
        assert!(generator.validate_table_name("INGEST_20250928101039").is_ok());
        assert!(generator.validate_table_name("test_table_123").is_ok());

        // Invalid table names
        assert!(generator.validate_table_name("").is_err());
        assert!(generator.validate_table_name("table-name").is_err());
        assert!(generator.validate_table_name("table name").is_err());
        assert!(generator.validate_table_name("table;DROP TABLE").is_err());
    }
}