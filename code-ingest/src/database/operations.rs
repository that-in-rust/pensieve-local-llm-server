//! Core database operations
//!
//! This module provides batch insertion operations, query execution with result formatting,
//! transaction management, and performance optimization for the code ingestion system.

use crate::error::{DatabaseError, DatabaseResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row, Transaction, Postgres, Column};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Represents a processed file ready for database insertion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedFile {
    pub filepath: String,
    pub filename: String,
    pub extension: Option<String>,
    pub file_size_bytes: i64,
    pub line_count: Option<i32>,
    pub word_count: Option<i32>,
    pub token_count: Option<i32>,
    pub content_text: Option<String>,
    pub file_type: FileType,
    pub conversion_command: Option<String>,
    pub relative_path: String,
    pub absolute_path: String,
}

/// File type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType {
    DirectText,
    Convertible,
    NonText,
}

impl FileType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::DirectText => "direct_text",
            FileType::Convertible => "convertible",
            FileType::NonText => "non_text",
        }
    }
}

/// Result of a SQL query execution
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, String>>,
    pub row_count: usize,
    pub execution_time_ms: u64,
}

/// Result of a batch insertion operation
#[derive(Debug, Clone)]
pub struct BatchInsertResult {
    pub inserted_count: usize,
    pub failed_count: usize,
    pub execution_time_ms: u64,
    pub errors: Vec<String>,
}

/// Analysis result for storage in QUERYRESULT_* tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub sql_query: String,
    pub prompt_file_path: Option<String>,
    pub llm_result: String,
    pub original_file_path: Option<String>,
    pub chunk_number: Option<i32>,
    pub analysis_type: Option<String>,
}

/// Database operations manager
pub struct DatabaseOperations {
    pool: PgPool,
}

impl DatabaseOperations {
    /// Create a new database operations manager
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert a batch of processed files into an ingestion table
    pub async fn batch_insert_files(
        &self,
        table_name: &str,
        ingestion_id: i64,
        files: Vec<ProcessedFile>,
    ) -> DatabaseResult<BatchInsertResult> {
        let start_time = std::time::Instant::now();
        
        if files.is_empty() {
            return Ok(BatchInsertResult {
                inserted_count: 0,
                failed_count: 0,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                errors: vec![],
            });
        }

        debug!("Batch inserting {} files into table {}", files.len(), table_name);

        let mut transaction = self.pool.begin().await.map_err(|e| DatabaseError::TransactionFailed {
            cause: format!("Failed to start transaction: {}", e),
        })?;

        let mut inserted_count = 0;
        let mut failed_count = 0;
        let mut errors = Vec::new();

        // Use batch insertion for better performance
        const BATCH_SIZE: usize = 1000;
        
        for chunk in files.chunks(BATCH_SIZE) {
            match self.insert_file_chunk(&mut transaction, table_name, ingestion_id, chunk).await {
                Ok(count) => inserted_count += count,
                Err(e) => {
                    failed_count += chunk.len();
                    errors.push(e.to_string());
                    warn!("Failed to insert batch of {} files: {}", chunk.len(), e);
                }
            }
        }

        // Populate window content after all files are inserted
        if inserted_count > 0 {
            info!("Populating multi-scale context windows for {} files", inserted_count);
            self.populate_window_content(&mut transaction, table_name).await?;
        }

        // Commit transaction
        transaction.commit().await.map_err(|e| DatabaseError::TransactionFailed {
            cause: format!("Failed to commit transaction: {}", e),
        })?;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        
        info!(
            "Batch insert completed: {} inserted, {} failed in {}ms",
            inserted_count, failed_count, execution_time_ms
        );

        Ok(BatchInsertResult {
            inserted_count,
            failed_count,
            execution_time_ms,
            errors,
        })
    }

    /// Execute a SQL query and return formatted results
    pub async fn execute_query(&self, sql: &str) -> DatabaseResult<QueryResult> {
        let start_time = std::time::Instant::now();
        
        debug!("Executing query: {}", sql);

        let rows = sqlx::query(sql)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: e.to_string(),
            })?;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        if rows.is_empty() {
            return Ok(QueryResult {
                columns: vec![],
                rows: vec![],
                row_count: 0,
                execution_time_ms,
            });
        }

        // Extract column names from the first row
        let columns: Vec<String> = rows[0]
            .columns()
            .iter()
            .map(|col| col.name().to_string())
            .collect();

        // Convert rows to HashMap format
        let mut result_rows = Vec::new();
        for row in &rows {
            let mut row_map = HashMap::new();
            for (i, column) in columns.iter().enumerate() {
                let value = self.extract_column_value(&row, i)?;
                row_map.insert(column.clone(), value);
            }
            result_rows.push(row_map);
        }

        debug!("Query executed successfully: {} rows in {}ms", rows.len(), execution_time_ms);

        Ok(QueryResult {
            columns,
            rows: result_rows,
            row_count: rows.len(),
            execution_time_ms,
        })
    }

    /// Execute a query and format results for LLM consumption
    pub async fn execute_query_for_llm(&self, sql: &str) -> DatabaseResult<String> {
        let result = self.execute_query(sql).await?;
        Ok(self.format_query_result_for_llm(&result))
    }

    /// Store analysis results in a QUERYRESULT_* table
    pub async fn store_analysis_result(
        &self,
        table_name: &str,
        result: AnalysisResult,
    ) -> DatabaseResult<i64> {
        debug!("Storing analysis result in table: {}", table_name);

        let insert_sql = format!(
            r#"
            INSERT INTO "{}" (sql_query, prompt_file_path, llm_result, original_file_path, chunk_number, analysis_type)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING analysis_id
            "#,
            table_name
        );

        let row = sqlx::query(&insert_sql)
            .bind(&result.sql_query)
            .bind(&result.prompt_file_path)
            .bind(&result.llm_result)
            .bind(&result.original_file_path)
            .bind(result.chunk_number)
            .bind(&result.analysis_type)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: insert_sql,
                cause: e.to_string(),
            })?;

        let analysis_id: i64 = row.get(0);
        debug!("Stored analysis result with ID: {}", analysis_id);
        
        Ok(analysis_id)
    }

    /// Create and record an ingestion metadata entry
    pub async fn create_ingestion_record(
        &self,
        repo_url: Option<&str>,
        local_path: &str,
        start_timestamp: u64,
        table_name: &str,
    ) -> DatabaseResult<i64> {
        let insert_sql = r#"
            INSERT INTO ingestion_meta (repo_url, local_path, start_timestamp_unix, table_name)
            VALUES ($1, $2, $3, $4)
            RETURNING ingestion_id
        "#;

        let row = sqlx::query(insert_sql)
            .bind(repo_url)
            .bind(local_path)
            .bind(start_timestamp as i64)
            .bind(table_name)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: insert_sql.to_string(),
                cause: e.to_string(),
            })?;

        let ingestion_id: i64 = row.get(0);
        info!("Created ingestion record with ID: {}", ingestion_id);
        
        Ok(ingestion_id)
    }

    /// Update an ingestion record with completion information
    pub async fn complete_ingestion_record(
        &self,
        ingestion_id: i64,
        end_timestamp: u64,
        total_files_processed: i32,
    ) -> DatabaseResult<()> {
        let update_sql = r#"
            UPDATE ingestion_meta 
            SET end_timestamp_unix = $1, total_files_processed = $2
            WHERE ingestion_id = $3
        "#;

        sqlx::query(update_sql)
            .bind(end_timestamp as i64)
            .bind(total_files_processed)
            .bind(ingestion_id)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: update_sql.to_string(),
                cause: e.to_string(),
            })?;

        info!("Completed ingestion record {} with {} files", ingestion_id, total_files_processed);
        Ok(())
    }

    /// Get ingestion statistics
    pub async fn get_ingestion_stats(&self, ingestion_id: i64) -> DatabaseResult<IngestionStats> {
        let stats_sql = r#"
            SELECT 
                repo_url,
                local_path,
                start_timestamp_unix,
                end_timestamp_unix,
                table_name,
                total_files_processed,
                created_at
            FROM ingestion_meta 
            WHERE ingestion_id = $1
        "#;

        let row = sqlx::query(stats_sql)
            .bind(ingestion_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: stats_sql.to_string(),
                cause: e.to_string(),
            })?;

        Ok(IngestionStats {
            ingestion_id,
            repo_url: row.get("repo_url"),
            local_path: row.get("local_path"),
            start_timestamp_unix: row.get("start_timestamp_unix"),
            end_timestamp_unix: row.get("end_timestamp_unix"),
            table_name: row.get("table_name"),
            total_files_processed: row.get("total_files_processed"),
            created_at: row.get("created_at"),
        })
    }

    /// Execute a transaction with multiple operations
    pub async fn execute_transaction<F, R>(&self, operations: F) -> DatabaseResult<R>
    where
        F: for<'c> FnOnce(&'c mut Transaction<'_, Postgres>) -> std::pin::Pin<Box<dyn std::future::Future<Output = DatabaseResult<R>> + Send + 'c>>,
    {
        let mut transaction = self.pool.begin().await.map_err(|e| DatabaseError::TransactionFailed {
            cause: format!("Failed to start transaction: {}", e),
        })?;

        let result = operations(&mut transaction).await?;

        transaction.commit().await.map_err(|e| DatabaseError::TransactionFailed {
            cause: format!("Failed to commit transaction: {}", e),
        })?;

        Ok(result)
    }

    // Private helper methods

    fn calculate_parent_filepath(&self, filepath: &str) -> String {
        if let Some(last_slash_pos) = filepath.rfind('/') {
            filepath[..last_slash_pos].to_string()
        } else {
            filepath.to_string()
        }
    }

    async fn populate_window_content(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        table_name: &str,
    ) -> DatabaseResult<()> {
        // Update l1_window_content (directory level) using GROUP BY instead of window functions
        let l1_update_sql = format!(
            r#"
            UPDATE "{}" SET l1_window_content = subquery.l1_content
            FROM (
                SELECT 
                    parent_filepath,
                    STRING_AGG(content_text, E'\n--- FILE SEPARATOR ---\n' ORDER BY filepath) as l1_content
                FROM "{}"
                WHERE content_text IS NOT NULL
                GROUP BY parent_filepath
            ) AS subquery
            WHERE "{}".parent_filepath = subquery.parent_filepath
            "#,
            table_name, table_name, table_name
        );

        sqlx::query(&l1_update_sql)
            .execute(&mut **transaction)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: "populate l1_window_content".to_string(),
                cause: e.to_string(),
            })?;

        // Update l2_window_content (system level - grandfather directory)
        let l2_update_sql = format!(
            r#"
            UPDATE "{}" SET l2_window_content = subquery.l2_content
            FROM (
                SELECT 
                    CASE 
                        WHEN parent_filepath LIKE '%/%' THEN 
                            LEFT(parent_filepath, LENGTH(parent_filepath) - POSITION('/' IN REVERSE(parent_filepath)))
                        ELSE parent_filepath 
                    END as grandfather_filepath,
                    STRING_AGG(content_text, E'\n--- MODULE SEPARATOR ---\n' ORDER BY parent_filepath, filepath) as l2_content
                FROM "{}"
                WHERE content_text IS NOT NULL
                GROUP BY grandfather_filepath
            ) AS subquery
            WHERE CASE 
                WHEN "{}".parent_filepath LIKE '%/%' THEN 
                    LEFT("{}".parent_filepath, LENGTH("{}".parent_filepath) - POSITION('/' IN REVERSE("{}".parent_filepath)))
                ELSE "{}".parent_filepath 
            END = subquery.grandfather_filepath
            "#,
            table_name, table_name, table_name, table_name, table_name, table_name, table_name
        );

        sqlx::query(&l2_update_sql)
            .execute(&mut **transaction)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: "populate l2_window_content".to_string(),
                cause: e.to_string(),
            })?;

        Ok(())
    }

    async fn insert_file_chunk(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        table_name: &str,
        ingestion_id: i64,
        files: &[ProcessedFile],
    ) -> DatabaseResult<usize> {
        // First, insert all files with basic data and calculated parent_filepath
        let insert_sql = format!(
            r#"
            INSERT INTO "{}" (
                ingestion_id, filepath, filename, extension, file_size_bytes,
                line_count, word_count, token_count, content_text, file_type,
                conversion_command, relative_path, absolute_path, parent_filepath
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            "#,
            table_name
        );

        let mut inserted = 0;
        for file in files {
            let parent_filepath = self.calculate_parent_filepath(&file.filepath);
            
            sqlx::query(&insert_sql)
                .bind(ingestion_id)
                .bind(&file.filepath)
                .bind(&file.filename)
                .bind(&file.extension)
                .bind(file.file_size_bytes)
                .bind(file.line_count)
                .bind(file.word_count)
                .bind(file.token_count)
                .bind(&file.content_text)
                .bind(file.file_type.as_str())
                .bind(&file.conversion_command)
                .bind(&file.relative_path)
                .bind(&file.absolute_path)
                .bind(&parent_filepath)
                .execute(&mut **transaction)
                .await
                .map_err(|e| DatabaseError::BatchInsertionFailed {
                    cause: format!("Failed to insert file {}: {}", file.filepath, e),
                })?;
            
            inserted += 1;
        }

        Ok(inserted)
    }

    fn extract_column_value(&self, row: &sqlx::postgres::PgRow, column_index: usize) -> DatabaseResult<String> {
        let column = &row.columns()[column_index];
        let column_name = column.name();

        // Handle different PostgreSQL types using the actual type info
        use sqlx::TypeInfo;
        
        let type_info = column.type_info();
        let type_name = type_info.name();
        
        let value = match type_name {
            "TEXT" | "VARCHAR" | "CHAR" => {
                row.try_get::<Option<String>, _>(column_index)
                    .map_err(|e| DatabaseError::QueryFailed {
                        query: format!("extract column {}", column_name),
                        cause: e.to_string(),
                    })?
                    .unwrap_or_default()
            }
            "INT4" => {
                row.try_get::<Option<i32>, _>(column_index)
                    .map_err(|e| DatabaseError::QueryFailed {
                        query: format!("extract column {}", column_name),
                        cause: e.to_string(),
                    })?
                    .map(|v| v.to_string())
                    .unwrap_or_default()
            }
            "INT8" => {
                row.try_get::<Option<i64>, _>(column_index)
                    .map_err(|e| DatabaseError::QueryFailed {
                        query: format!("extract column {}", column_name),
                        cause: e.to_string(),
                    })?
                    .map(|v| v.to_string())
                    .unwrap_or_default()
            }
            "TIMESTAMPTZ" | "TIMESTAMP" => {
                row.try_get::<Option<DateTime<Utc>>, _>(column_index)
                    .map_err(|e| DatabaseError::QueryFailed {
                        query: format!("extract column {}", column_name),
                        cause: e.to_string(),
                    })?
                    .map(|v| v.to_rfc3339())
                    .unwrap_or_default()
            }
            "BOOL" => {
                row.try_get::<Option<bool>, _>(column_index)
                    .map_err(|e| DatabaseError::QueryFailed {
                        query: format!("extract column {}", column_name),
                        cause: e.to_string(),
                    })?
                    .map(|v| v.to_string())
                    .unwrap_or_default()
            }
            _ => {
                // For unknown types, try to get as string first, then fallback to type name
                match row.try_get::<Option<String>, _>(column_index) {
                    Ok(Some(s)) => s,
                    Ok(None) => String::new(),
                    Err(_) => format!("[{}]", type_name),
                }
            }
        };

        Ok(value)
    }

    fn format_query_result_for_llm(&self, result: &QueryResult) -> String {
        let mut output = String::new();
        
        for row in &result.rows {
            if let Some(filepath) = row.get("filepath") {
                output.push_str(&format!("FILE: {}\n\n", filepath));
                
                if let Some(content) = row.get("content_text") {
                    output.push_str(content);
                } else {
                    output.push_str("[No content available]");
                }
                
                output.push_str("\n\n---\n\n");
            } else {
                // For non-file queries, format as key-value pairs
                for (key, value) in row {
                    output.push_str(&format!("{}: {}\n", key, value));
                }
                output.push_str("\n---\n\n");
            }
        }
        
        output
    }

    /// Insert processed files into the database (wrapper for batch_insert_files)
    pub async fn insert_processed_files(
        &self,
        table_name: &str,
        files: &[crate::processing::ProcessedFile],
        ingestion_id: i64,
    ) -> DatabaseResult<()> {
        // Convert from processing::ProcessedFile to operations::ProcessedFile
        let converted_files: Vec<ProcessedFile> = files
            .iter()
            .filter(|f| !f.skipped) // Only insert non-skipped files
            .map(|f| ProcessedFile {
                filepath: f.filepath.clone(),
                filename: f.filename.clone(),
                extension: Some(f.extension.clone()),
                file_size_bytes: f.file_size_bytes,
                line_count: f.line_count,
                word_count: f.word_count,
                token_count: f.token_count,
                content_text: f.content_text.clone(),
                file_type: match f.file_type {
                    crate::processing::FileType::DirectText => FileType::DirectText,
                    crate::processing::FileType::Convertible => FileType::Convertible,
                    crate::processing::FileType::NonText => FileType::NonText,
                },
                conversion_command: f.conversion_command.clone(),
                relative_path: f.relative_path.clone(),
                absolute_path: f.absolute_path.clone(),
            })
            .collect();

        let result = self.batch_insert_files(table_name, ingestion_id, converted_files).await?;
        
        if result.failed_count > 0 {
            warn!("Some files failed to insert: {} failures", result.failed_count);
        }
        
        Ok(())
    }

    /// Get ingestion statistics (wrapper for get_ingestion_stats)
    pub async fn get_ingestion_statistics(
        &self,
        ingestion_id: i64,
    ) -> DatabaseResult<crate::ingestion::IngestionStatistics> {
        let stats = self.get_ingestion_stats(ingestion_id).await?;
        
        // Get additional statistics from the ingestion table
        let table_stats_sql = format!(
            r#"
            SELECT 
                COUNT(*) as total_files,
                SUM(file_size_bytes) as total_size_bytes,
                file_type,
                COUNT(*) as type_count
            FROM "{}"
            WHERE ingestion_id = $1
            GROUP BY file_type
            "#,
            stats.table_name
        );

        let type_rows = sqlx::query(&table_stats_sql)
            .bind(ingestion_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: table_stats_sql,
                cause: e.to_string(),
            })?;

        let mut file_type_counts = std::collections::HashMap::new();
        for row in type_rows {
            let file_type: String = row.get("file_type");
            let count: i64 = row.get("type_count");
            file_type_counts.insert(file_type, count as i32);
        }

        // Get extension statistics
        let extension_stats_sql = format!(
            r#"
            SELECT 
                extension,
                COUNT(*) as ext_count
            FROM "{}"
            WHERE ingestion_id = $1
            GROUP BY extension
            ORDER BY ext_count DESC
            LIMIT 20
            "#,
            stats.table_name
        );

        let ext_rows = sqlx::query(&extension_stats_sql)
            .bind(ingestion_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: extension_stats_sql,
                cause: e.to_string(),
            })?;

        let mut extension_counts = std::collections::HashMap::new();
        for row in ext_rows {
            let extension: Option<String> = row.get("extension");
            let count: i64 = row.get("ext_count");
            let ext_key = extension.unwrap_or_else(|| "no_extension".to_string());
            extension_counts.insert(ext_key, count as i32);
        }

        let processing_duration = if let Some(end_time) = stats.end_timestamp_unix {
            std::time::Duration::from_secs((end_time - stats.start_timestamp_unix) as u64)
        } else {
            std::time::Duration::ZERO
        };

        Ok(crate::ingestion::IngestionStatistics {
            ingestion_id: stats.ingestion_id,
            table_name: stats.table_name,
            total_files: stats.total_files_processed.unwrap_or(0),
            total_size_bytes: 0, // Would need to calculate from table
            file_type_counts,
            extension_counts,
            processing_duration,
        })
    }

    /// List all ingestion records
    pub async fn list_ingestion_records(&self) -> DatabaseResult<Vec<crate::ingestion::IngestionRecord>> {
        let list_sql = r#"
            SELECT 
                ingestion_id,
                repo_url,
                local_path,
                table_name,
                start_timestamp_unix,
                end_timestamp_unix,
                total_files_processed,
                created_at
            FROM ingestion_meta
            ORDER BY created_at DESC
        "#;

        let rows = sqlx::query(list_sql)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: list_sql.to_string(),
                cause: e.to_string(),
            })?;

        let mut records = Vec::new();
        for row in rows {
            let start_timestamp: i64 = row.get("start_timestamp_unix");
            let end_timestamp: Option<i64> = row.get("end_timestamp_unix");
            let total_files: Option<i32> = row.get("total_files_processed");

            let start_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(start_timestamp as u64);
            let end_time = end_timestamp.map(|ts| std::time::UNIX_EPOCH + std::time::Duration::from_secs(ts as u64));

            let status = if end_time.is_some() {
                if total_files.unwrap_or(0) > 0 {
                    crate::ingestion::IngestionStatus::Completed
                } else {
                    crate::ingestion::IngestionStatus::Failed
                }
            } else {
                crate::ingestion::IngestionStatus::InProgress
            };

            records.push(crate::ingestion::IngestionRecord {
                ingestion_id: row.get("ingestion_id"),
                repo_url: row.get("repo_url"),
                local_path: row.get("local_path"),
                table_name: row.get("table_name"),
                start_time,
                end_time,
                total_files_processed: total_files,
                status,
            });
        }

        Ok(records)
    }
}

/// Statistics for an ingestion operation
#[derive(Debug, Clone)]
pub struct IngestionStats {
    pub ingestion_id: i64,
    pub repo_url: Option<String>,
    pub local_path: String,
    pub start_timestamp_unix: i64,
    pub end_timestamp_unix: Option<i64>,
    pub table_name: String,
    pub total_files_processed: Option<i32>,
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool() -> Option<PgPool> {
        // Only run tests if DATABASE_URL is set
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    fn create_test_file() -> ProcessedFile {
        ProcessedFile {
            filepath: "/test/file.rs".to_string(),
            filename: "file.rs".to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: 1024,
            line_count: Some(50),
            word_count: Some(200),
            token_count: Some(180),
            content_text: Some("fn main() { println!(\"Hello, world!\"); }".to_string()),
            file_type: FileType::DirectText,
            conversion_command: None,
            relative_path: "file.rs".to_string(),
            absolute_path: "/test/file.rs".to_string(),
        }
    }

    #[test]
    fn test_file_type_as_str() {
        assert_eq!(FileType::DirectText.as_str(), "direct_text");
        assert_eq!(FileType::Convertible.as_str(), "convertible");
        assert_eq!(FileType::NonText.as_str(), "non_text");
    }

    #[test]
    fn test_processed_file_creation() {
        let file = create_test_file();
        assert_eq!(file.filename, "file.rs");
        assert_eq!(file.file_type.as_str(), "direct_text");
        assert!(file.content_text.is_some());
    }

    #[tokio::test]
    async fn test_database_operations_creation() {
        if let Some(pool) = create_test_pool() {
            let _ops = DatabaseOperations::new(pool);
            // Just test that we can create the operations manager
            assert!(true);
        }
    }

    #[tokio::test]
    async fn test_empty_batch_insert() {
        if let Some(pool) = create_test_pool() {
            let ops = DatabaseOperations::new(pool);
            
            let result = ops.batch_insert_files("test_table", 1, vec![]).await.unwrap();
            assert_eq!(result.inserted_count, 0);
            assert_eq!(result.failed_count, 0);
            assert!(result.errors.is_empty());
        }
    }

    #[tokio::test]
    async fn test_query_result_formatting() {
        if let Some(pool) = create_test_pool() {
            let ops = DatabaseOperations::new(pool);
            
            // Create a mock query result
            let mut row1 = HashMap::new();
            row1.insert("filepath".to_string(), "/test/file1.rs".to_string());
            row1.insert("content_text".to_string(), "fn test() {}".to_string());
            
            let mut row2 = HashMap::new();
            row2.insert("filepath".to_string(), "/test/file2.rs".to_string());
            row2.insert("content_text".to_string(), "struct Test;".to_string());
            
            let result = QueryResult {
                columns: vec!["filepath".to_string(), "content_text".to_string()],
                rows: vec![row1, row2],
                row_count: 2,
                execution_time_ms: 10,
            };
            
            let formatted = ops.format_query_result_for_llm(&result);
            assert!(formatted.contains("FILE: /test/file1.rs"));
            assert!(formatted.contains("fn test() {}"));
            assert!(formatted.contains("FILE: /test/file2.rs"));
            assert!(formatted.contains("struct Test;"));
            assert!(formatted.contains("---"));
        }
    }

    #[tokio::test]
    async fn test_analysis_result_creation() {
        let result = AnalysisResult {
            sql_query: "SELECT * FROM test".to_string(),
            prompt_file_path: Some("/prompts/test.md".to_string()),
            llm_result: "Analysis complete".to_string(),
            original_file_path: Some("/test/file.rs".to_string()),
            chunk_number: Some(1),
            analysis_type: Some("code_review".to_string()),
        };
        
        assert_eq!(result.sql_query, "SELECT * FROM test");
        assert_eq!(result.llm_result, "Analysis complete");
        assert_eq!(result.chunk_number, Some(1));
    }

    #[tokio::test]
    async fn test_ingestion_stats_structure() {
        let stats = IngestionStats {
            ingestion_id: 123,
            repo_url: Some("https://github.com/user/repo".to_string()),
            local_path: "/local/path".to_string(),
            start_timestamp_unix: 1695825022,
            end_timestamp_unix: Some(1695825122),
            table_name: "INGEST_20250927143022".to_string(),
            total_files_processed: Some(100),
            created_at: Utc::now(),
        };
        
        assert_eq!(stats.ingestion_id, 123);
        assert_eq!(stats.total_files_processed, Some(100));
        assert!(stats.repo_url.is_some());
    }

    #[test]
    fn test_batch_insert_result() {
        let result = BatchInsertResult {
            inserted_count: 95,
            failed_count: 5,
            execution_time_ms: 1500,
            errors: vec!["Error 1".to_string(), "Error 2".to_string()],
        };
        
        assert_eq!(result.inserted_count, 95);
        assert_eq!(result.failed_count, 5);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_query_result_structure() {
        let mut row = HashMap::new();
        row.insert("id".to_string(), "1".to_string());
        row.insert("name".to_string(), "test".to_string());
        
        let result = QueryResult {
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![row],
            row_count: 1,
            execution_time_ms: 50,
        };
        
        assert_eq!(result.columns.len(), 2);
        assert_eq!(result.row_count, 1);
        assert_eq!(result.execution_time_ms, 50);
    }
}