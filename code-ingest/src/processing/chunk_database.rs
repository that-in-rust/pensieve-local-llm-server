//! Database integration for chunking operations
//!
//! This module handles the database operations for chunked data, including
//! chunked table population, batch insertion, and size threshold checking.

use crate::database::schema::SchemaManager;
use crate::error::{ProcessingError, ProcessingResult};
use crate::processing::chunking::{ChunkData, ChunkContext, ChunkingEngine};
use sqlx::{PgPool, Row};

use tracing::{debug, info, warn};

/// Configuration for chunk database operations
#[derive(Debug, Clone)]
pub struct ChunkDatabaseConfig {
    /// Batch size for database insertions
    pub batch_size: usize,
    /// Maximum number of concurrent database operations
    pub max_concurrent_ops: usize,
    /// Whether to create indexes immediately after table creation
    pub create_indexes_immediately: bool,
}

impl Default for ChunkDatabaseConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_concurrent_ops: 10,
            create_indexes_immediately: true,
        }
    }
}

/// Database manager for chunked data operations
pub struct ChunkDatabaseManager {
    pool: PgPool,
    schema_manager: SchemaManager,
    config: ChunkDatabaseConfig,
}

/// Represents a row from the base table for chunking
#[derive(Debug, Clone)]
pub struct BaseTableRow {
    pub file_id: String,
    pub filepath: String,
    pub filename: String,
    pub extension: Option<String>,
    pub line_count: Option<i32>,
    pub content_text: Option<String>,
}

/// Result of chunking operation with database integration
#[derive(Debug)]
pub struct ChunkDatabaseResult {
    /// Name of the created chunked table
    pub chunked_table_name: String,
    /// Number of rows processed from base table
    pub base_rows_processed: usize,
    /// Number of chunks created and inserted
    pub chunks_inserted: usize,
    /// Number of files that were actually chunked (vs. single chunk)
    pub files_chunked: usize,
    /// Number of files below chunk size threshold
    pub files_below_threshold: usize,
}

impl ChunkDatabaseManager {
    /// Create a new chunk database manager
    pub fn new(pool: PgPool, config: ChunkDatabaseConfig) -> Self {
        let schema_manager = SchemaManager::new(pool.clone());
        Self {
            pool,
            schema_manager,
            config,
        }
    }

    /// Create a new chunk database manager with default configuration
    pub fn with_pool(pool: PgPool) -> Self {
        Self::new(pool, ChunkDatabaseConfig::default())
    }

    /// Process a base table and create a chunked version
    /// 
    /// This is the main entry point for chunking database operations.
    /// It reads from the base table, chunks files that exceed the threshold,
    /// and populates the chunked table.
    pub async fn process_table_with_chunking(
        &self,
        base_table_name: &str,
        chunk_size: usize,
    ) -> ProcessingResult<ChunkDatabaseResult> {
        info!("Starting chunking process for table '{}' with chunk size {}", base_table_name, chunk_size);

        // Check if base table exists
        if !self.schema_manager.table_exists(base_table_name).await.map_err(|e| {
            ProcessingError::ChunkingFailed {
                reason: format!("Failed to check if base table exists: {}", e),
            }
        })? {
            return Err(ProcessingError::ChunkingFailed {
                reason: format!("Base table '{}' does not exist", base_table_name),
            });
        }

        // Create chunked table
        let chunked_table_name = self.schema_manager
            .create_chunked_table(base_table_name, chunk_size)
            .await
            .map_err(|e| ProcessingError::ChunkingFailed {
                reason: format!("Failed to create chunked table: {}", e),
            })?;

        // Read data from base table
        let base_rows = self.read_base_table_data(base_table_name).await?;
        info!("Read {} rows from base table '{}'", base_rows.len(), base_table_name);

        // Process rows and create chunks
        let chunking_engine = ChunkingEngine::with_chunk_size(chunk_size);
        let mut all_chunks = Vec::new();
        let mut all_contexts = Vec::new();
        let mut files_chunked = 0;
        let mut files_below_threshold = 0;

        for row in &base_rows {
            if let Some(content) = &row.content_text {
                let line_count = row.line_count.unwrap_or(0) as usize;
                
                // Check size threshold
                if line_count < chunk_size {
                    files_below_threshold += 1;
                    // Create single chunk for small files
                    let single_chunk = self.create_single_chunk_for_small_file(row, content)?;
                    let single_context = self.create_single_chunk_context(content);
                    all_chunks.push(single_chunk);
                    all_contexts.push(single_context);
                } else {
                    files_chunked += 1;
                    // Chunk the file
                    let (chunking_result, contexts) = chunking_engine.chunk_content_with_context(
                        content,
                        row.file_id.clone(),
                        row.filepath.clone(),
                        row.filename.clone(),
                        row.extension.clone(),
                    )?;
                    
                    all_chunks.extend(chunking_result.chunks);
                    all_contexts.extend(contexts);
                }
            } else {
                warn!("Row with file_id '{}' has no content, skipping", row.file_id);
            }
        }

        info!("Generated {} chunks from {} files ({} chunked, {} below threshold)", 
              all_chunks.len(), base_rows.len(), files_chunked, files_below_threshold);

        // Insert chunks into database in batches
        let chunks_inserted = self.insert_chunks_batch(&chunked_table_name, &all_chunks, &all_contexts).await?;

        Ok(ChunkDatabaseResult {
            chunked_table_name,
            base_rows_processed: base_rows.len(),
            chunks_inserted,
            files_chunked,
            files_below_threshold,
        })
    }

    /// Read data from the base table
    async fn read_base_table_data(&self, table_name: &str) -> ProcessingResult<Vec<BaseTableRow>> {
        let query = format!(
            r#"
            SELECT 
                file_id::text as file_id,
                filepath,
                filename,
                extension,
                line_count,
                content_text
            FROM "{}"
            WHERE content_text IS NOT NULL
            ORDER BY file_id
            "#,
            table_name
        );

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ProcessingError::ChunkingFailed {
                reason: format!("Failed to read base table data: {}", e),
            })?;

        let mut base_rows = Vec::new();
        for row in rows {
            let base_row = BaseTableRow {
                file_id: row.get("file_id"),
                filepath: row.get("filepath"),
                filename: row.get("filename"),
                extension: row.get("extension"),
                line_count: row.get("line_count"),
                content_text: row.get("content_text"),
            };
            base_rows.push(base_row);
        }

        Ok(base_rows)
    }

    /// Create a single chunk for files below the chunk size threshold
    fn create_single_chunk_for_small_file(&self, row: &BaseTableRow, content: &str) -> ProcessingResult<ChunkData> {
        let line_count = content.lines().count() as u32;
        
        Ok(ChunkData {
            metadata: crate::processing::chunking::ChunkMetadata {
                chunk_number: 1,
                start_line: 1,
                end_line: line_count,
                line_count,
                structure_adjusted: false,
            },
            content: content.to_string(),
            file_id: row.file_id.clone(),
            filepath: row.filepath.clone(),
            filename: row.filename.clone(),
            extension: row.extension.clone(),
        })
    }

    /// Create context for a single chunk (small file)
    fn create_single_chunk_context(&self, content: &str) -> ChunkContext {
        ChunkContext {
            l1_content: content.to_string(),
            l2_content: content.to_string(),
            is_boundary: true, // Single chunk is always a boundary
            l1_chunk_count: 1,
            l2_chunk_count: 1,
        }
    }

    /// Insert chunks into the database in batches
    async fn insert_chunks_batch(
        &self,
        table_name: &str,
        chunks: &[ChunkData],
        contexts: &[ChunkContext],
    ) -> ProcessingResult<usize> {
        if chunks.len() != contexts.len() {
            return Err(ProcessingError::ChunkingFailed {
                reason: format!("Chunks and contexts length mismatch: {} vs {}", chunks.len(), contexts.len()),
            });
        }

        let mut total_inserted = 0;
        let batch_size = self.config.batch_size;

        for (batch_start, chunk_batch) in chunks.chunks(batch_size).enumerate() {
            let context_batch = &contexts[batch_start * batch_size..std::cmp::min((batch_start + 1) * batch_size, contexts.len())];
            
            debug!("Inserting batch {} with {} chunks", batch_start, chunk_batch.len());
            
            let inserted = self.insert_chunk_batch(table_name, chunk_batch, context_batch).await?;
            total_inserted += inserted;
        }

        info!("Successfully inserted {} chunks into table '{}'", total_inserted, table_name);
        Ok(total_inserted)
    }

    /// Insert a single batch of chunks
    async fn insert_chunk_batch(
        &self,
        table_name: &str,
        chunks: &[ChunkData],
        contexts: &[ChunkContext],
    ) -> ProcessingResult<usize> {
        let mut query_builder = sqlx::QueryBuilder::new(
            format!(
                r#"INSERT INTO "{}" (
                    file_id, filepath, parent_filepath, filename, extension,
                    chunk_number, chunk_start_line, chunk_end_line, line_count,
                    content, content_l1, content_l2
                ) "#,
                table_name
            )
        );

        query_builder.push_values(chunks.iter().zip(contexts.iter()), |mut b, (chunk, context)| {
            b.push_bind(&chunk.file_id)
                .push_bind(&chunk.filepath)
                .push_bind(&chunk.filepath) // parent_filepath same as filepath for now
                .push_bind(&chunk.filename)
                .push_bind(&chunk.extension)
                .push_bind(chunk.metadata.chunk_number as i32)
                .push_bind(chunk.metadata.start_line as i32)
                .push_bind(chunk.metadata.end_line as i32)
                .push_bind(chunk.metadata.line_count as i32)
                .push_bind(&chunk.content)
                .push_bind(&context.l1_content)
                .push_bind(&context.l2_content);
        });

        let query = query_builder.build();
        let result = query.execute(&self.pool).await.map_err(|e| {
            ProcessingError::ChunkingFailed {
                reason: format!("Failed to insert chunk batch: {}", e),
            }
        })?;

        Ok(result.rows_affected() as usize)
    }

    /// Check if a file should be chunked based on its line count
    pub fn should_chunk_file(&self, line_count: Option<i32>, chunk_size: usize) -> bool {
        match line_count {
            Some(count) => count as usize >= chunk_size,
            None => false, // If we don't know the line count, don't chunk
        }
    }

    /// Get statistics about a chunked table
    pub async fn get_chunked_table_stats(&self, table_name: &str) -> ProcessingResult<ChunkedTableStats> {
        let query = format!(
            r#"
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT file_id) as unique_files,
                AVG(line_count) as avg_chunk_size,
                MIN(line_count) as min_chunk_size,
                MAX(line_count) as max_chunk_size,
                SUM(CASE WHEN chunk_number = 1 THEN 1 ELSE 0 END) as files_with_single_chunk
            FROM "{}"
            "#,
            table_name
        );

        let row = sqlx::query(&query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| ProcessingError::ChunkingFailed {
                reason: format!("Failed to get chunked table stats: {}", e),
            })?;

        Ok(ChunkedTableStats {
            total_chunks: row.get::<i64, _>("total_chunks") as usize,
            unique_files: row.get::<i64, _>("unique_files") as usize,
            avg_chunk_size: row.get::<Option<f64>, _>("avg_chunk_size").unwrap_or(0.0),
            min_chunk_size: row.get::<Option<i32>, _>("min_chunk_size").unwrap_or(0) as usize,
            max_chunk_size: row.get::<Option<i32>, _>("max_chunk_size").unwrap_or(0) as usize,
            files_with_single_chunk: row.get::<i64, _>("files_with_single_chunk") as usize,
        })
    }

    /// Validate that a chunked table has consistent data
    pub async fn validate_chunked_table(&self, table_name: &str) -> ProcessingResult<ValidationResult> {
        let mut issues = Vec::new();

        // Check for gaps in chunk numbering
        let gap_query = format!(
            r#"
            SELECT file_id, chunk_number, 
                   LAG(chunk_number) OVER (PARTITION BY file_id ORDER BY chunk_number) as prev_chunk
            FROM "{}"
            ORDER BY file_id, chunk_number
            "#,
            table_name
        );

        let gap_rows = sqlx::query(&gap_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ProcessingError::ChunkingFailed {
                reason: format!("Failed to validate chunk numbering: {}", e),
            })?;

        for row in gap_rows {
            let file_id: String = row.get("file_id");
            let chunk_number: i32 = row.get("chunk_number");
            let prev_chunk: Option<i32> = row.get("prev_chunk");

            if let Some(prev) = prev_chunk {
                if chunk_number != prev + 1 {
                    issues.push(format!("Gap in chunk numbering for file '{}': {} -> {}", file_id, prev, chunk_number));
                }
            }
        }

        // Check for overlapping line ranges
        let overlap_query = format!(
            r#"
            SELECT file_id, chunk_number, chunk_start_line, chunk_end_line,
                   LAG(chunk_end_line) OVER (PARTITION BY file_id ORDER BY chunk_number) as prev_end_line
            FROM "{}"
            ORDER BY file_id, chunk_number
            "#,
            table_name
        );

        let overlap_rows = sqlx::query(&overlap_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| ProcessingError::ChunkingFailed {
                reason: format!("Failed to validate line ranges: {}", e),
            })?;

        for row in overlap_rows {
            let file_id: String = row.get("file_id");
            let chunk_number: i32 = row.get("chunk_number");
            let start_line: i32 = row.get("chunk_start_line");
            let prev_end_line: Option<i32> = row.get("prev_end_line");

            if let Some(prev_end) = prev_end_line {
                if start_line != prev_end + 1 {
                    issues.push(format!("Line range gap for file '{}' chunk {}: expected start {}, got {}", 
                                      file_id, chunk_number, prev_end + 1, start_line));
                }
            }
        }

        Ok(ValidationResult {
            is_valid: issues.is_empty(),
            issues,
        })
    }
}

/// Statistics about a chunked table
#[derive(Debug, Clone)]
pub struct ChunkedTableStats {
    pub total_chunks: usize,
    pub unique_files: usize,
    pub avg_chunk_size: f64,
    pub min_chunk_size: usize,
    pub max_chunk_size: usize,
    pub files_with_single_chunk: usize,
}

/// Result of table validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub issues: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::chunking::ChunkingConfig;

    // Note: These tests require a PostgreSQL database connection
    // They are integration tests and should be run with DATABASE_URL set

    fn create_test_pool() -> Option<PgPool> {
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    #[tokio::test]
    async fn test_chunk_database_manager_creation() {
        if let Some(pool) = create_test_pool() {
            let _manager = ChunkDatabaseManager::with_pool(pool);
            // Just test that we can create the manager
            assert!(true);
        }
    }

    #[tokio::test]
    async fn test_should_chunk_file() {
        if let Some(pool) = create_test_pool() {
            let manager = ChunkDatabaseManager::with_pool(pool);
            
            assert!(manager.should_chunk_file(Some(1000), 500));  // Should chunk
            assert!(!manager.should_chunk_file(Some(100), 500));  // Should not chunk
            assert!(!manager.should_chunk_file(None, 500));       // No line count, don't chunk
        }
    }

    #[test]
    fn test_base_table_row_structure() {
        let content = "line 1\nline 2\nline 3";
        let row = BaseTableRow {
            file_id: "test_file".to_string(),
            filepath: "test.txt".to_string(),
            filename: "test.txt".to_string(),
            extension: Some("txt".to_string()),
            line_count: Some(3),
            content_text: Some(content.to_string()),
        };

        // Test the structure is correct
        assert_eq!(row.file_id, "test_file");
        assert_eq!(row.filepath, "test.txt");
        assert_eq!(row.filename, "test.txt");
        assert_eq!(row.extension, Some("txt".to_string()));
        assert_eq!(row.line_count, Some(3));
        assert!(row.content_text.is_some());
    }

    #[test]
    fn test_chunked_table_stats_structure() {
        let stats = ChunkedTableStats {
            total_chunks: 100,
            unique_files: 25,
            avg_chunk_size: 45.5,
            min_chunk_size: 10,
            max_chunk_size: 100,
            files_with_single_chunk: 5,
        };

        assert_eq!(stats.total_chunks, 100);
        assert_eq!(stats.unique_files, 25);
        assert_eq!(stats.avg_chunk_size, 45.5);
        assert_eq!(stats.files_with_single_chunk, 5);
    }

    #[test]
    fn test_validation_result_structure() {
        let valid_result = ValidationResult {
            is_valid: true,
            issues: Vec::new(),
        };

        let invalid_result = ValidationResult {
            is_valid: false,
            issues: vec!["Gap in chunk numbering".to_string()],
        };

        assert!(valid_result.is_valid);
        assert!(valid_result.issues.is_empty());
        
        assert!(!invalid_result.is_valid);
        assert_eq!(invalid_result.issues.len(), 1);
    }
}