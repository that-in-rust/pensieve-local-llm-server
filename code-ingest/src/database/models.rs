//! Core data models for database operations
//!
//! This module defines the primary data structures used for database operations,
//! including ingestion metadata, file records, and query results with full
//! sqlx::FromRow support and serde serialization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, Row};
use crate::processing::FileType;

/// Ingestion metadata record from the ingestion_meta table
/// 
/// This struct represents a complete ingestion operation with timing,
/// source information, and processing statistics.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct IngestionMeta {
    /// Unique identifier for this ingestion operation
    pub ingestion_id: i64,
    
    /// GitHub repository URL (None for local folder ingestion)
    pub repo_url: Option<String>,
    
    /// Local path where the source was processed
    pub local_path: String,
    
    /// Unix timestamp when ingestion started
    pub start_timestamp_unix: i64,
    
    /// Unix timestamp when ingestion completed (None if still in progress)
    pub end_timestamp_unix: Option<i64>,
    
    /// Name of the timestamped table created for this ingestion
    pub table_name: String,
    
    /// Total number of files processed (None if still in progress)
    pub total_files_processed: Option<i32>,
    
    /// Timestamp when this record was created
    pub created_at: DateTime<Utc>,
}

/// Individual file record from INGEST_YYYYMMDDHHMMSS tables
/// 
/// This struct represents a single processed file with all its metadata,
/// content, and processing information.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct IngestedFile {
    /// Unique identifier for this file record
    pub file_id: i64,
    
    /// Foreign key to ingestion_meta.ingestion_id
    pub ingestion_id: i64,
    
    /// Full file path as discovered during ingestion
    pub filepath: String,
    
    /// Just the filename portion
    pub filename: String,
    
    /// File extension (without the dot)
    pub extension: Option<String>,
    
    /// File size in bytes
    pub file_size_bytes: i64,
    
    /// Number of lines in the file (None for binary files)
    pub line_count: Option<i32>,
    
    /// Number of words in the file (None for binary files)
    pub word_count: Option<i32>,
    
    /// Estimated token count for LLM processing (None for binary files)
    pub token_count: Option<i32>,
    
    /// Full text content of the file (None for binary files)
    pub content_text: Option<String>,
    
    /// File type classification as string for database storage
    #[sqlx(rename = "file_type")]
    pub file_type_str: String,
    
    /// Command used for conversion (None for direct text files)
    pub conversion_command: Option<String>,
    
    /// Relative path from the repository root
    pub relative_path: String,
    
    /// Absolute path on the local filesystem
    pub absolute_path: String,
    
    /// Timestamp when this record was created
    pub created_at: DateTime<Utc>,
}

/// Query result record from QUERYRESULT_* tables
/// 
/// This struct represents the result of an LLM analysis operation,
/// storing both the original query and the analysis results.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct QueryResult {
    /// Unique identifier for this analysis result
    pub analysis_id: i64,
    
    /// Original SQL query that generated the data for analysis
    pub sql_query: String,
    
    /// Path to the prompt file used for analysis (optional)
    pub prompt_file_path: Option<String>,
    
    /// LLM analysis result text
    pub llm_result: String,
    
    /// Original file path that was analyzed (optional)
    pub original_file_path: Option<String>,
    
    /// Chunk number if the analysis was done in chunks (optional)
    pub chunk_number: Option<i32>,
    
    /// Type of analysis performed (optional)
    pub analysis_type: Option<String>,
    
    /// Timestamp when this analysis was created
    pub created_at: DateTime<Utc>,
}

impl IngestionMeta {
    /// Create a new ingestion metadata record for starting an ingestion
    pub fn new_starting(
        repo_url: Option<String>,
        local_path: String,
        start_timestamp_unix: i64,
        table_name: String,
    ) -> Self {
        Self {
            ingestion_id: 0, // Will be set by database
            repo_url,
            local_path,
            start_timestamp_unix,
            end_timestamp_unix: None,
            table_name,
            total_files_processed: None,
            created_at: Utc::now(),
        }
    }

    /// Mark this ingestion as completed
    pub fn complete(&mut self, end_timestamp_unix: i64, total_files_processed: i32) {
        self.end_timestamp_unix = Some(end_timestamp_unix);
        self.total_files_processed = Some(total_files_processed);
    }

    /// Check if this ingestion is completed
    pub fn is_completed(&self) -> bool {
        self.end_timestamp_unix.is_some()
    }

    /// Get the duration of this ingestion in seconds (None if not completed)
    pub fn duration_seconds(&self) -> Option<i64> {
        self.end_timestamp_unix.map(|end| end - self.start_timestamp_unix)
    }
}

impl IngestedFile {
    /// Create a new ingested file record from processed file data
    pub fn from_processed_file(
        processed_file: &crate::processing::ProcessedFile,
        ingestion_id: i64,
    ) -> Self {
        Self {
            file_id: 0, // Will be set by database
            ingestion_id,
            filepath: processed_file.filepath.clone(),
            filename: processed_file.filename.clone(),
            extension: if processed_file.extension.is_empty() {
                None
            } else {
                Some(processed_file.extension.clone())
            },
            file_size_bytes: processed_file.file_size_bytes,
            line_count: processed_file.line_count,
            word_count: processed_file.word_count,
            token_count: processed_file.token_count,
            content_text: processed_file.content_text.clone(),
            file_type_str: processed_file.file_type.as_str().to_string(),
            conversion_command: processed_file.conversion_command.clone(),
            relative_path: processed_file.relative_path.clone(),
            absolute_path: processed_file.absolute_path.clone(),
            created_at: Utc::now(),
        }
    }

    /// Get the file type as an enum
    pub fn file_type(&self) -> Option<FileType> {
        FileType::from_str(&self.file_type_str)
    }

    /// Check if this file has text content
    pub fn has_content(&self) -> bool {
        self.content_text.is_some() && !self.content_text.as_ref().unwrap().is_empty()
    }

    /// Get content length in characters
    pub fn content_length(&self) -> usize {
        self.content_text.as_ref().map_or(0, |content| content.len())
    }

    /// Check if this file was converted from another format
    pub fn was_converted(&self) -> bool {
        self.conversion_command.is_some()
    }
}

impl QueryResult {
    /// Create a new query result record
    pub fn new(
        sql_query: String,
        llm_result: String,
        prompt_file_path: Option<String>,
        original_file_path: Option<String>,
        chunk_number: Option<i32>,
        analysis_type: Option<String>,
    ) -> Self {
        Self {
            analysis_id: 0, // Will be set by database
            sql_query,
            prompt_file_path,
            llm_result,
            original_file_path,
            chunk_number,
            analysis_type,
            created_at: Utc::now(),
        }
    }

    /// Check if this result is part of a chunked analysis
    pub fn is_chunked(&self) -> bool {
        self.chunk_number.is_some()
    }

    /// Get the result length in characters
    pub fn result_length(&self) -> usize {
        self.llm_result.len()
    }
}

/// Database operations for ingestion metadata
impl IngestionMeta {
    /// Insert a new ingestion record and return the assigned ID
    pub async fn insert(&self, pool: &sqlx::PgPool) -> crate::error::DatabaseResult<i64> {
        let record = sqlx::query(
            r#"
            INSERT INTO ingestion_meta (
                repo_url, local_path, start_timestamp_unix, 
                end_timestamp_unix, table_name, total_files_processed
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING ingestion_id
            "#,
        )
        .bind(&self.repo_url)
        .bind(&self.local_path)
        .bind(self.start_timestamp_unix)
        .bind(self.end_timestamp_unix)
        .bind(&self.table_name)
        .bind(self.total_files_processed)
        .fetch_one(pool)
        .await
        .map_err(|e| crate::error::DatabaseError::QueryFailed {
            query: "INSERT INTO ingestion_meta".to_string(),
            cause: e.to_string(),
        })?;

        Ok(record.get("ingestion_id"))
    }

    /// Update an existing ingestion record
    pub async fn update(&self, pool: &sqlx::PgPool) -> crate::error::DatabaseResult<()> {
        sqlx::query(
            r#"
            UPDATE ingestion_meta 
            SET end_timestamp_unix = $1, total_files_processed = $2
            WHERE ingestion_id = $3
            "#,
        )
        .bind(self.end_timestamp_unix)
        .bind(self.total_files_processed)
        .bind(self.ingestion_id)
        .execute(pool)
        .await
        .map_err(|e| crate::error::DatabaseError::QueryFailed {
            query: "UPDATE ingestion_meta".to_string(),
            cause: e.to_string(),
        })?;

        Ok(())
    }

    /// Find an ingestion record by ID
    pub async fn find_by_id(
        pool: &sqlx::PgPool,
        ingestion_id: i64,
    ) -> crate::error::DatabaseResult<Option<Self>> {
        let record = sqlx::query_as::<_, IngestionMeta>(
            "SELECT * FROM ingestion_meta WHERE ingestion_id = $1"
        )
        .bind(ingestion_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| crate::error::DatabaseError::QueryFailed {
            query: "SELECT FROM ingestion_meta BY ID".to_string(),
            cause: e.to_string(),
        })?;

        Ok(record)
    }

    /// List all ingestion records, ordered by creation time
    pub async fn list_all(pool: &sqlx::PgPool) -> crate::error::DatabaseResult<Vec<Self>> {
        let records = sqlx::query_as::<_, IngestionMeta>(
            "SELECT * FROM ingestion_meta ORDER BY created_at DESC"
        )
        .fetch_all(pool)
        .await
        .map_err(|e| crate::error::DatabaseError::QueryFailed {
            query: "SELECT FROM ingestion_meta".to_string(),
            cause: e.to_string(),
        })?;

        Ok(records)
    }
}

/// Database operations for ingested files
impl IngestedFile {
    /// Insert a batch of ingested files into the specified table
    pub async fn insert_batch(
        pool: &sqlx::PgPool,
        table_name: &str,
        files: &[Self],
    ) -> crate::error::DatabaseResult<u64> {
        if files.is_empty() {
            return Ok(0);
        }

        let mut query_builder = sqlx::QueryBuilder::new(format!(
            r#"INSERT INTO "{}" (
                ingestion_id, filepath, filename, extension, file_size_bytes,
                line_count, word_count, token_count, content_text, file_type,
                conversion_command, relative_path, absolute_path
            ) "#,
            table_name
        ));

        query_builder.push_values(files, |mut b, file| {
            b.push_bind(file.ingestion_id)
                .push_bind(&file.filepath)
                .push_bind(&file.filename)
                .push_bind(&file.extension)
                .push_bind(file.file_size_bytes)
                .push_bind(file.line_count)
                .push_bind(file.word_count)
                .push_bind(file.token_count)
                .push_bind(&file.content_text)
                .push_bind(&file.file_type_str)
                .push_bind(&file.conversion_command)
                .push_bind(&file.relative_path)
                .push_bind(&file.absolute_path);
        });

        let result = query_builder
            .build()
            .execute(pool)
            .await
            .map_err(|e| crate::error::DatabaseError::BatchInsertionFailed {
                cause: format!("Failed to insert {} files into {}: {}", files.len(), table_name, e),
            })?;

        Ok(result.rows_affected())
    }

    /// Query files from a specific ingestion table
    pub async fn query_from_table(
        pool: &sqlx::PgPool,
        table_name: &str,
        where_clause: Option<&str>,
        limit: Option<i64>,
    ) -> crate::error::DatabaseResult<Vec<Self>> {
        let mut query = format!("SELECT * FROM \"{}\"", table_name);
        
        if let Some(where_clause) = where_clause {
            query.push_str(" WHERE ");
            query.push_str(where_clause);
        }
        
        query.push_str(" ORDER BY file_id");
        
        if let Some(limit) = limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        let records = sqlx::query_as::<_, IngestedFile>(&query)
            .fetch_all(pool)
            .await
            .map_err(|e| crate::error::DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;

        Ok(records)
    }
}

/// Database operations for query results
impl QueryResult {
    /// Insert a new query result record
    pub async fn insert(
        &self,
        pool: &sqlx::PgPool,
        table_name: &str,
    ) -> crate::error::DatabaseResult<i64> {
        let record = sqlx::query(&format!(
            r#"
            INSERT INTO "{}" (
                sql_query, prompt_file_path, llm_result, 
                original_file_path, chunk_number, analysis_type
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING analysis_id
            "#,
            table_name
        ))
        .bind(&self.sql_query)
        .bind(&self.prompt_file_path)
        .bind(&self.llm_result)
        .bind(&self.original_file_path)
        .bind(self.chunk_number)
        .bind(&self.analysis_type)
        .fetch_one(pool)
        .await
        .map_err(|e| crate::error::DatabaseError::QueryFailed {
            query: format!("INSERT INTO {}", table_name),
            cause: e.to_string(),
        })?;

        let analysis_id: i64 = record.get("analysis_id");
        Ok(analysis_id)
    }

    /// Query results from a specific query result table
    pub async fn query_from_table(
        pool: &sqlx::PgPool,
        table_name: &str,
        where_clause: Option<&str>,
        limit: Option<i64>,
    ) -> crate::error::DatabaseResult<Vec<Self>> {
        let mut query = format!("SELECT * FROM \"{}\"", table_name);
        
        if let Some(where_clause) = where_clause {
            query.push_str(" WHERE ");
            query.push_str(where_clause);
        }
        
        query.push_str(" ORDER BY created_at DESC");
        
        if let Some(limit) = limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        let records = sqlx::query_as::<_, QueryResult>(&query)
            .fetch_all(pool)
            .await
            .map_err(|e| crate::error::DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::{ProcessedFile, FileType};

    #[test]
    fn test_ingestion_meta_creation() {
        let meta = IngestionMeta::new_starting(
            Some("https://github.com/user/repo".to_string()),
            "/tmp/clone".to_string(),
            1695825022,
            "INGEST_20250927143022".to_string(),
        );

        assert_eq!(meta.repo_url, Some("https://github.com/user/repo".to_string()));
        assert_eq!(meta.local_path, "/tmp/clone");
        assert_eq!(meta.start_timestamp_unix, 1695825022);
        assert_eq!(meta.table_name, "INGEST_20250927143022");
        assert!(!meta.is_completed());
        assert_eq!(meta.duration_seconds(), None);
    }

    #[test]
    fn test_ingestion_meta_completion() {
        let mut meta = IngestionMeta::new_starting(
            None,
            "/local/path".to_string(),
            1695825022,
            "INGEST_20250927143022".to_string(),
        );

        meta.complete(1695825122, 150);

        assert!(meta.is_completed());
        assert_eq!(meta.total_files_processed, Some(150));
        assert_eq!(meta.duration_seconds(), Some(100));
    }

    #[test]
    fn test_ingested_file_from_processed_file() {
        let processed_file = ProcessedFile {
            filepath: "src/main.rs".to_string(),
            filename: "main.rs".to_string(),
            extension: "rs".to_string(),
            file_size_bytes: 1024,
            line_count: Some(50),
            word_count: Some(200),
            token_count: Some(180),
            content_text: Some("fn main() {}".to_string()),
            file_type: FileType::DirectText,
            conversion_command: None,
            relative_path: "src/main.rs".to_string(),
            absolute_path: "/home/user/project/src/main.rs".to_string(),
            skipped: false,
            skip_reason: None,
        };

        let ingested_file = IngestedFile::from_processed_file(&processed_file, 123);

        assert_eq!(ingested_file.ingestion_id, 123);
        assert_eq!(ingested_file.filepath, "src/main.rs");
        assert_eq!(ingested_file.filename, "main.rs");
        assert_eq!(ingested_file.extension, Some("rs".to_string()));
        assert_eq!(ingested_file.file_size_bytes, 1024);
        assert_eq!(ingested_file.line_count, Some(50));
        assert_eq!(ingested_file.word_count, Some(200));
        assert_eq!(ingested_file.token_count, Some(180));
        assert_eq!(ingested_file.content_text, Some("fn main() {}".to_string()));
        assert_eq!(ingested_file.file_type_str, "direct_text");
        assert_eq!(ingested_file.conversion_command, None);
        assert_eq!(ingested_file.relative_path, "src/main.rs");
        assert_eq!(ingested_file.absolute_path, "/home/user/project/src/main.rs");
        
        assert!(ingested_file.has_content());
        assert_eq!(ingested_file.content_length(), 12);
        assert!(!ingested_file.was_converted());
        assert_eq!(ingested_file.file_type(), Some(FileType::DirectText));
    }

    #[test]
    fn test_ingested_file_empty_extension() {
        let processed_file = ProcessedFile {
            filepath: "README".to_string(),
            filename: "README".to_string(),
            extension: "".to_string(), // Empty extension
            file_size_bytes: 512,
            line_count: Some(20),
            word_count: Some(100),
            token_count: Some(90),
            content_text: Some("# Project README".to_string()),
            file_type: FileType::DirectText,
            conversion_command: None,
            relative_path: "README".to_string(),
            absolute_path: "/home/user/project/README".to_string(),
            skipped: false,
            skip_reason: None,
        };

        let ingested_file = IngestedFile::from_processed_file(&processed_file, 456);

        assert_eq!(ingested_file.extension, None); // Should be None for empty extension
    }

    #[test]
    fn test_query_result_creation() {
        let result = QueryResult::new(
            "SELECT * FROM INGEST_20250927143022 WHERE extension = 'rs'".to_string(),
            "Analysis of Rust files shows...".to_string(),
            Some("/path/to/prompt.md".to_string()),
            Some("/path/to/file.rs".to_string()),
            Some(1),
            Some("code_analysis".to_string()),
        );

        assert_eq!(result.sql_query, "SELECT * FROM INGEST_20250927143022 WHERE extension = 'rs'");
        assert_eq!(result.llm_result, "Analysis of Rust files shows...");
        assert_eq!(result.prompt_file_path, Some("/path/to/prompt.md".to_string()));
        assert_eq!(result.original_file_path, Some("/path/to/file.rs".to_string()));
        assert_eq!(result.chunk_number, Some(1));
        assert_eq!(result.analysis_type, Some("code_analysis".to_string()));
        
        assert!(result.is_chunked());
        assert_eq!(result.result_length(), 31);
    }

    #[test]
    fn test_query_result_not_chunked() {
        let result = QueryResult::new(
            "SELECT COUNT(*) FROM INGEST_20250927143022".to_string(),
            "Total files: 150".to_string(),
            None,
            None,
            None,
            Some("statistics".to_string()),
        );

        assert!(!result.is_chunked());
        assert_eq!(result.result_length(), 16);
    }

    #[test]
    fn test_serde_serialization() {
        let meta = IngestionMeta::new_starting(
            Some("https://github.com/user/repo".to_string()),
            "/tmp/clone".to_string(),
            1695825022,
            "INGEST_20250927143022".to_string(),
        );

        // Test serialization
        let serialized = serde_json::to_string(&meta).unwrap();
        assert!(serialized.contains("https://github.com/user/repo"));
        assert!(serialized.contains("INGEST_20250927143022"));

        // Test deserialization
        let deserialized: IngestionMeta = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.repo_url, meta.repo_url);
        assert_eq!(deserialized.table_name, meta.table_name);
    }

    #[tokio::test]
    async fn test_database_operations_require_real_db() {
        // These tests would require a real database connection
        // They are included to show the interface but will be skipped
        // unless DATABASE_URL is set in the environment
        
        if std::env::var("DATABASE_URL").is_err() {
            return; // Skip database tests
        }

        // Example of how the database operations would be tested:
        // let pool = sqlx::PgPool::connect(&std::env::var("DATABASE_URL").unwrap()).await.unwrap();
        // let meta = IngestionMeta::new_starting(...);
        // let id = meta.insert(&pool).await.unwrap();
        // assert!(id > 0);
    }
}