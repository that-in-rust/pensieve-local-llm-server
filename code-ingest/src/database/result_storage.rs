//! Analysis result storage functionality
//!
//! This module handles storing LLM analysis results in QUERYRESULT_* tables,
//! creating tables with proper schema, metadata tracking, and result validation.

use crate::database::operations::{DatabaseOperations, AnalysisResult};
use crate::error::{DatabaseError, DatabaseResult};
use sqlx::{PgPool, Row};
use std::path::Path;
use tokio::fs;
use tracing::{debug, info, warn};

/// Result storage manager for analysis results
pub struct ResultStorage {
    operations: DatabaseOperations,
    pool: PgPool,
}

/// Configuration for result storage
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Whether to create table if it doesn't exist
    pub create_table_if_missing: bool,
    /// Whether to validate result content
    pub validate_content: bool,
    /// Maximum result size in bytes
    pub max_result_size: usize,
    /// Whether to compress large results
    pub compress_large_results: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            create_table_if_missing: true,
            validate_content: true,
            max_result_size: 10 * 1024 * 1024, // 10MB
            compress_large_results: false, // Keep simple for now
        }
    }
}

/// Result of storing analysis results
#[derive(Debug, Clone)]
pub struct StorageResult {
    pub analysis_id: i64,
    pub table_name: String,
    pub result_size_bytes: usize,
    pub compressed: bool,
}

/// Metadata for analysis results
#[derive(Debug, Clone)]
pub struct ResultMetadata {
    pub original_query: String,
    pub prompt_file_path: Option<String>,
    pub analysis_type: Option<String>,
    pub original_file_path: Option<String>,
    pub chunk_number: Option<i32>,
    pub created_by: Option<String>,
    pub tags: Vec<String>,
}

impl ResultStorage {
    /// Create a new result storage manager
    pub fn new(pool: PgPool) -> Self {
        Self {
            operations: DatabaseOperations::new(pool.clone()),
            pool,
        }
    }

    /// Store analysis result from file
    pub async fn store_result_from_file(
        &self,
        table_name: &str,
        result_file_path: &Path,
        metadata: &ResultMetadata,
        config: &StorageConfig,
    ) -> DatabaseResult<StorageResult> {
        debug!("Storing analysis result from file: {}", result_file_path.display());
        
        // Read result content from file
        let result_content = self.read_result_file(result_file_path).await?;
        
        // Validate content if configured
        if config.validate_content {
            self.validate_result_content(&result_content, config)?;
        }
        
        // Store the result
        self.store_result_content(table_name, &result_content, metadata, config).await
    }

    /// Store analysis result from string content
    pub async fn store_result_content(
        &self,
        table_name: &str,
        result_content: &str,
        metadata: &ResultMetadata,
        config: &StorageConfig,
    ) -> DatabaseResult<StorageResult> {
        debug!("Storing analysis result to table: {}", table_name);
        
        // Ensure table exists
        if config.create_table_if_missing {
            self.ensure_result_table_exists(table_name).await?;
        }
        
        // Prepare analysis result
        let analysis_result = AnalysisResult {
            sql_query: metadata.original_query.clone(),
            prompt_file_path: metadata.prompt_file_path.clone(),
            llm_result: result_content.to_string(),
            original_file_path: metadata.original_file_path.clone(),
            chunk_number: metadata.chunk_number,
            analysis_type: metadata.analysis_type.clone(),
        };
        
        // Store in database
        let analysis_id = self.operations.store_analysis_result(table_name, analysis_result).await?;
        
        info!(
            "Stored analysis result: ID {} in table {} ({} bytes)",
            analysis_id,
            table_name,
            result_content.len()
        );
        
        Ok(StorageResult {
            analysis_id,
            table_name: table_name.to_string(),
            result_size_bytes: result_content.len(),
            compressed: false, // Not implemented yet
        })
    }

    /// Retrieve analysis result by ID
    pub async fn get_analysis_result(
        &self,
        table_name: &str,
        analysis_id: i64,
    ) -> DatabaseResult<AnalysisResultWithMetadata> {
        debug!("Retrieving analysis result: ID {} from table {}", analysis_id, table_name);
        
        let query = format!(
            r#"
            SELECT 
                analysis_id,
                sql_query,
                prompt_file_path,
                llm_result,
                original_file_path,
                chunk_number,
                analysis_type,
                created_at
            FROM "{}"
            WHERE analysis_id = $1
            "#,
            table_name
        );
        
        let row = sqlx::query(&query)
            .bind(analysis_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;
        
        Ok(AnalysisResultWithMetadata {
            analysis_id: row.get("analysis_id"),
            sql_query: row.get("sql_query"),
            prompt_file_path: row.get("prompt_file_path"),
            llm_result: row.get("llm_result"),
            original_file_path: row.get("original_file_path"),
            chunk_number: row.get("chunk_number"),
            analysis_type: row.get("analysis_type"),
            created_at: row.get("created_at"),
        })
    }

    /// List all analysis results in a table
    pub async fn list_analysis_results(
        &self,
        table_name: &str,
        limit: Option<usize>,
    ) -> DatabaseResult<Vec<AnalysisResultSummary>> {
        debug!("Listing analysis results from table: {}", table_name);
        
        let limit_clause = if let Some(limit) = limit {
            format!("LIMIT {}", limit)
        } else {
            String::new()
        };
        
        let query = format!(
            r#"
            SELECT 
                analysis_id,
                sql_query,
                prompt_file_path,
                analysis_type,
                LENGTH(llm_result) as result_size,
                created_at
            FROM "{}"
            ORDER BY created_at DESC
            {}
            "#,
            table_name, limit_clause
        );
        
        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(AnalysisResultSummary {
                analysis_id: row.get("analysis_id"),
                sql_query: row.get("sql_query"),
                prompt_file_path: row.get("prompt_file_path"),
                analysis_type: row.get("analysis_type"),
                result_size_bytes: row.get::<i32, _>("result_size") as usize,
                created_at: row.get("created_at"),
            });
        }
        
        Ok(results)
    }

    /// Delete analysis result by ID
    pub async fn delete_analysis_result(
        &self,
        table_name: &str,
        analysis_id: i64,
    ) -> DatabaseResult<bool> {
        debug!("Deleting analysis result: ID {} from table {}", analysis_id, table_name);
        
        let query = format!("DELETE FROM \"{}\" WHERE analysis_id = $1", table_name);
        
        let result = sqlx::query(&query)
            .bind(analysis_id)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;
        
        let deleted = result.rows_affected() > 0;
        if deleted {
            info!("Deleted analysis result: ID {} from table {}", analysis_id, table_name);
        } else {
            warn!("Analysis result not found: ID {} in table {}", analysis_id, table_name);
        }
        
        Ok(deleted)
    }

    /// Get storage statistics for a result table
    pub async fn get_storage_stats(&self, table_name: &str) -> DatabaseResult<StorageStats> {
        debug!("Getting storage statistics for table: {}", table_name);
        
        let query = format!(
            r#"
            SELECT 
                COUNT(*) as total_results,
                AVG(LENGTH(llm_result)) as avg_result_size,
                MAX(LENGTH(llm_result)) as max_result_size,
                MIN(created_at) as oldest_result,
                MAX(created_at) as newest_result
            FROM "{}"
            "#,
            table_name
        );
        
        let row = sqlx::query(&query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;
        
        Ok(StorageStats {
            table_name: table_name.to_string(),
            total_results: row.get::<i64, _>("total_results") as usize,
            avg_result_size_bytes: row.get::<Option<f64>, _>("avg_result_size").unwrap_or(0.0) as usize,
            max_result_size_bytes: row.get::<Option<i32>, _>("max_result_size").unwrap_or(0) as usize,
            oldest_result: row.get("oldest_result"),
            newest_result: row.get("newest_result"),
        })
    }

    // Private helper methods

    async fn read_result_file(&self, file_path: &Path) -> DatabaseResult<String> {
        fs::read_to_string(file_path)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: format!("read_file: {}", file_path.display()),
                cause: format!("Failed to read result file: {}", e),
            })
    }

    fn validate_result_content(&self, content: &str, config: &StorageConfig) -> DatabaseResult<()> {
        // Check size limits
        if content.len() > config.max_result_size {
            return Err(DatabaseError::QueryFailed {
                query: "validate_content".to_string(),
                cause: format!(
                    "Result content too large: {} bytes (max: {})",
                    content.len(),
                    config.max_result_size
                ),
            });
        }
        
        // Check for empty content
        if content.trim().is_empty() {
            warn!("Storing empty analysis result");
        }
        
        // Basic content validation
        if content.len() < 10 {
            warn!("Analysis result is very short: {} characters", content.len());
        }
        
        Ok(())
    }

    async fn ensure_result_table_exists(&self, table_name: &str) -> DatabaseResult<()> {
        debug!("Ensuring result table exists: {}", table_name);
        
        // Check if table exists
        let exists_query = r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        "#;
        
        let exists: bool = sqlx::query_scalar(exists_query)
            .bind(table_name)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: exists_query.to_string(),
                cause: e.to_string(),
            })?;
        
        if !exists {
            self.create_result_table(table_name).await?;
        }
        
        Ok(())
    }

    async fn create_result_table(&self, table_name: &str) -> DatabaseResult<()> {
        info!("Creating result table: {}", table_name);
        
        let create_query = format!(
            r#"
            CREATE TABLE "{}" (
                analysis_id BIGSERIAL PRIMARY KEY,
                sql_query TEXT NOT NULL,
                prompt_file_path VARCHAR,
                llm_result TEXT NOT NULL,
                original_file_path VARCHAR,
                chunk_number INTEGER,
                analysis_type VARCHAR,
                created_at TIMESTAMP DEFAULT NOW()
            )
            "#,
            table_name
        );
        
        sqlx::query(&create_query)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::TableCreationFailed {
                table_name: table_name.to_string(),
                cause: e.to_string(),
            })?;
        
        // Create indexes for better performance
        let index_queries = vec![
            format!("CREATE INDEX idx_{}_created_at ON \"{}\"(created_at)", table_name, table_name),
            format!("CREATE INDEX idx_{}_analysis_type ON \"{}\"(analysis_type)", table_name, table_name),
        ];
        
        for index_query in index_queries {
            sqlx::query(&index_query)
                .execute(&self.pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: index_query.clone(),
                    cause: e.to_string(),
                })?;
        }
        
        info!("Created result table with indexes: {}", table_name);
        Ok(())
    }
}

/// Analysis result with full metadata
#[derive(Debug, Clone)]
pub struct AnalysisResultWithMetadata {
    pub analysis_id: i64,
    pub sql_query: String,
    pub prompt_file_path: Option<String>,
    pub llm_result: String,
    pub original_file_path: Option<String>,
    pub chunk_number: Option<i32>,
    pub analysis_type: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Summary of analysis result (without full content)
#[derive(Debug, Clone)]
pub struct AnalysisResultSummary {
    pub analysis_id: i64,
    pub sql_query: String,
    pub prompt_file_path: Option<String>,
    pub analysis_type: Option<String>,
    pub result_size_bytes: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Storage statistics for a result table
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub table_name: String,
    pub total_results: usize,
    pub avg_result_size_bytes: usize,
    pub max_result_size_bytes: usize,
    pub oldest_result: Option<chrono::DateTime<chrono::Utc>>,
    pub newest_result: Option<chrono::DateTime<chrono::Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool() -> Option<PgPool> {
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert!(config.create_table_if_missing);
        assert!(config.validate_content);
        assert_eq!(config.max_result_size, 10 * 1024 * 1024);
        assert!(!config.compress_large_results);
    }

    #[test]
    fn test_result_metadata() {
        let metadata = ResultMetadata {
            original_query: "SELECT * FROM test".to_string(),
            prompt_file_path: Some("/tmp/prompt.md".to_string()),
            analysis_type: Some("code_review".to_string()),
            original_file_path: Some("/tmp/source.rs".to_string()),
            chunk_number: Some(1),
            created_by: Some("user123".to_string()),
            tags: vec!["rust".to_string(), "analysis".to_string()],
        };
        
        assert_eq!(metadata.original_query, "SELECT * FROM test");
        assert_eq!(metadata.analysis_type, Some("code_review".to_string()));
        assert_eq!(metadata.chunk_number, Some(1));
        assert_eq!(metadata.tags.len(), 2);
    }

    #[tokio::test]
    async fn test_result_storage_creation() {
        if let Some(pool) = create_test_pool() {
            let _storage = ResultStorage::new(pool);
            // Just test that we can create the storage manager
            assert!(true);
        }
    }

    #[test]
    fn test_content_validation() {
        if let Some(pool) = create_test_pool() {
            let storage = ResultStorage::new(pool);
            let config = StorageConfig::default();
            
            // Valid content
            let valid_content = "This is a valid analysis result with sufficient content.";
            assert!(storage.validate_result_content(valid_content, &config).is_ok());
            
            // Empty content (should warn but not error)
            let empty_content = "";
            assert!(storage.validate_result_content(empty_content, &config).is_ok());
            
            // Very short content (should warn but not error)
            let short_content = "Short";
            assert!(storage.validate_result_content(short_content, &config).is_ok());
        }
    }

    #[test]
    fn test_content_size_validation() {
        if let Some(pool) = create_test_pool() {
            let storage = ResultStorage::new(pool);
            let config = StorageConfig {
                max_result_size: 100, // Very small limit for testing
                ..Default::default()
            };
            
            // Content within limit
            let small_content = "Small content";
            assert!(storage.validate_result_content(small_content, &config).is_ok());
            
            // Content exceeding limit
            let large_content = "x".repeat(200);
            assert!(storage.validate_result_content(&large_content, &config).is_err());
        }
    }

    #[test]
    fn test_storage_result() {
        let result = StorageResult {
            analysis_id: 123,
            table_name: "QUERYRESULT_test".to_string(),
            result_size_bytes: 1024,
            compressed: false,
        };
        
        assert_eq!(result.analysis_id, 123);
        assert_eq!(result.table_name, "QUERYRESULT_test");
        assert_eq!(result.result_size_bytes, 1024);
        assert!(!result.compressed);
    }

    #[test]
    fn test_analysis_result_summary() {
        let summary = AnalysisResultSummary {
            analysis_id: 456,
            sql_query: "SELECT * FROM files".to_string(),
            prompt_file_path: Some("/tmp/prompt.md".to_string()),
            analysis_type: Some("security_audit".to_string()),
            result_size_bytes: 2048,
            created_at: chrono::Utc::now(),
        };
        
        assert_eq!(summary.analysis_id, 456);
        assert_eq!(summary.sql_query, "SELECT * FROM files");
        assert_eq!(summary.analysis_type, Some("security_audit".to_string()));
        assert_eq!(summary.result_size_bytes, 2048);
    }

    #[test]
    fn test_storage_stats() {
        let stats = StorageStats {
            table_name: "QUERYRESULT_analysis".to_string(),
            total_results: 50,
            avg_result_size_bytes: 1500,
            max_result_size_bytes: 5000,
            oldest_result: Some(chrono::Utc::now() - chrono::Duration::days(30)),
            newest_result: Some(chrono::Utc::now()),
        };
        
        assert_eq!(stats.table_name, "QUERYRESULT_analysis");
        assert_eq!(stats.total_results, 50);
        assert_eq!(stats.avg_result_size_bytes, 1500);
        assert_eq!(stats.max_result_size_bytes, 5000);
        assert!(stats.oldest_result.is_some());
        assert!(stats.newest_result.is_some());
    }

    #[test]
    fn test_analysis_result_with_metadata() {
        let result = AnalysisResultWithMetadata {
            analysis_id: 789,
            sql_query: "SELECT filepath, content_text FROM files".to_string(),
            prompt_file_path: Some("/prompts/analyze.md".to_string()),
            llm_result: "Analysis complete: Found 5 security issues".to_string(),
            original_file_path: Some("/src/main.rs".to_string()),
            chunk_number: Some(2),
            analysis_type: Some("security".to_string()),
            created_at: chrono::Utc::now(),
        };
        
        assert_eq!(result.analysis_id, 789);
        assert!(result.llm_result.contains("security issues"));
        assert_eq!(result.chunk_number, Some(2));
        assert_eq!(result.analysis_type, Some("security".to_string()));
    }
}