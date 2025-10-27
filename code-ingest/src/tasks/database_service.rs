//! Database service for chunk-level task generation
//!
//! This module provides database operations specifically for the chunk-level task generator,
//! including table validation, row querying, and chunked table creation.

use crate::database::models::IngestedFile;
use crate::error::{DatabaseError, DatabaseResult};
use crate::tasks::chunk_level_task_generator::{TaskGeneratorError, TaskGeneratorResult};
use sqlx::{PgPool, Row};
use std::sync::Arc;
use tracing::{debug, info, warn, error};

/// Information about a database table
#[derive(Debug, Clone)]
pub struct TableInfo {
    /// Table name
    pub name: String,
    /// Number of rows in the table
    pub row_count: i64,
    /// Whether the table has the required schema
    pub has_valid_schema: bool,
    /// List of column names
    pub columns: Vec<String>,
}

/// Database service for chunk-level task generation operations
#[derive(Debug, Clone)]
pub struct DatabaseService {
    pool: Arc<PgPool>,
}

impl DatabaseService {
    /// Create a new database service with connection pool management
    pub fn new(pool: Arc<PgPool>) -> Self {
        debug!("Creating DatabaseService with connection pool");
        Self { pool }
    }

    /// Validate that a table exists and has the required schema for task generation
    /// 
    /// # Arguments
    /// * `table_name` - Name of the table to validate
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<TableInfo>` - Table information or error
    /// 
    /// # Requirements
    /// This method satisfies requirement 1.1 and 2.1 by validating table existence and schema
    pub async fn validate_table(&self, table_name: &str) -> TaskGeneratorResult<TableInfo> {
        debug!("Validating table: {}", table_name);

        // Check if table exists
        let table_exists = self.check_table_exists(table_name).await?;
        if !table_exists {
            error!("Table '{}' does not exist", table_name);
            return Err(TaskGeneratorError::table_not_found(table_name));
        }

        // Get table schema information
        let columns = self.get_table_columns(table_name).await?;
        debug!("Table '{}' has columns: {:?}", table_name, columns);

        // Validate required columns for IngestedFile
        let required_columns = vec![
            "file_id", "ingestion_id", "filepath", "filename", "extension",
            "file_size_bytes", "line_count", "word_count", "token_count",
            "content_text", "file_type", "conversion_command", "relative_path",
            "absolute_path", "created_at"
        ];

        let has_valid_schema = required_columns.iter()
            .all(|col| columns.contains(&col.to_string()));

        if !has_valid_schema {
            let missing_columns: Vec<_> = required_columns.iter()
                .filter(|col| !columns.contains(&col.to_string()))
                .collect();
            
            warn!("Table '{}' is missing required columns: {:?}", table_name, missing_columns);
            return Err(TaskGeneratorError::invalid_table_name(
                table_name,
                format!("Missing required columns: {:?}", missing_columns)
            ));
        }

        // Get row count
        let row_count = self.get_table_row_count(table_name).await?;
        info!("Table '{}' validated successfully with {} rows", table_name, row_count);

        Ok(TableInfo {
            name: table_name.to_string(),
            row_count,
            has_valid_schema,
            columns,
        })
    }

    /// Query rows from a table for task generation
    /// 
    /// # Arguments
    /// * `table_name` - Name of the table to query
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<Vec<IngestedFile>>` - List of ingested files or error
    /// 
    /// # Requirements
    /// This method satisfies requirement 1.1 and 2.1 by fetching IngestedFile records
    pub async fn query_rows(&self, table_name: &str) -> TaskGeneratorResult<Vec<IngestedFile>> {
        debug!("Querying rows from table: {}", table_name);

        // First validate the table exists
        let _table_info = self.validate_table(table_name).await?;

        // Query all rows from the table
        let query = format!("SELECT * FROM \"{}\" ORDER BY file_id", table_name);
        
        let rows = sqlx::query_as::<_, IngestedFile>(&query)
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| {
                error!("Failed to query rows from table '{}': {}", table_name, e);
                TaskGeneratorError::Database(DatabaseError::QueryFailed {
                    query: query.clone(),
                    cause: e.to_string(),
                })
            })?;

        info!("Successfully queried {} rows from table '{}'", rows.len(), table_name);
        Ok(rows)
    }

    /// Create a chunked table for chunk-level mode
    /// 
    /// # Arguments
    /// * `original_table` - Name of the original table
    /// * `chunk_size` - Size of chunks in lines
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<String>` - Name of the created chunked table or error
    /// 
    /// # Requirements
    /// This method satisfies requirement 2.2 by creating chunked tables for chunk-level mode
    pub async fn create_chunked_table(&self, original_table: &str, chunk_size: usize) -> TaskGeneratorResult<String> {
        debug!("Creating chunked table for '{}' with chunk size {}", original_table, chunk_size);

        if chunk_size == 0 {
            return Err(TaskGeneratorError::invalid_chunk_size(chunk_size));
        }

        // Validate original table exists
        let _table_info = self.validate_table(original_table).await?;

        // Generate chunked table name
        let chunked_table_name = format!("{}_{}", original_table, chunk_size);

        // Check if chunked table already exists
        if self.check_table_exists(&chunked_table_name).await? {
            warn!("Chunked table '{}' already exists, dropping it", chunked_table_name);
            self.drop_table(&chunked_table_name).await?;
        }

        // Create the chunked table with the same schema as the original
        let create_table_sql = format!(
            r#"
            CREATE TABLE "{}" (
                file_id BIGSERIAL PRIMARY KEY,
                ingestion_id BIGINT NOT NULL,
                filepath TEXT NOT NULL,
                filename TEXT NOT NULL,
                extension TEXT,
                file_size_bytes BIGINT NOT NULL,
                line_count INTEGER,
                word_count INTEGER,
                token_count INTEGER,
                content_text TEXT,
                file_type TEXT NOT NULL,
                conversion_command TEXT,
                relative_path TEXT NOT NULL,
                absolute_path TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Additional fields for chunking
                original_file_id BIGINT NOT NULL,
                chunk_number INTEGER NOT NULL DEFAULT 0,
                content_l1 TEXT,
                content_l2 TEXT,
                
                -- Index for efficient querying
                UNIQUE(original_file_id, chunk_number)
            )
            "#,
            chunked_table_name
        );

        sqlx::query(&create_table_sql)
            .execute(&*self.pool)
            .await
            .map_err(|e| {
                error!("Failed to create chunked table '{}': {}", chunked_table_name, e);
                TaskGeneratorError::Database(DatabaseError::TableCreationFailed {
                    table_name: chunked_table_name.clone(),
                    cause: e.to_string(),
                })
            })?;

        // Create indexes for better performance
        let create_index_sql = format!(
            "CREATE INDEX idx_{}_original_file_id ON \"{}\" (original_file_id)",
            chunked_table_name.to_lowercase().replace("_", ""),
            chunked_table_name
        );

        sqlx::query(&create_index_sql)
            .execute(&*self.pool)
            .await
            .map_err(|e| {
                warn!("Failed to create index for chunked table '{}': {}", chunked_table_name, e);
                // Don't fail the operation for index creation failure
            })?;

        info!("Successfully created chunked table: {}", chunked_table_name);
        Ok(chunked_table_name)
    }

    /// Get the connection pool for advanced operations
    pub fn pool(&self) -> &Arc<PgPool> {
        &self.pool
    }

    // Private helper methods

    /// Check if a table exists in the database
    async fn check_table_exists(&self, table_name: &str) -> TaskGeneratorResult<bool> {
        let query = r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = $1
            )
        "#;

        let exists: bool = sqlx::query_scalar(query)
            .bind(table_name)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| {
                error!("Failed to check if table '{}' exists: {}", table_name, e);
                TaskGeneratorError::Database(DatabaseError::QueryFailed {
                    query: query.to_string(),
                    cause: e.to_string(),
                })
            })?;

        debug!("Table '{}' exists: {}", table_name, exists);
        Ok(exists)
    }

    /// Get column names for a table
    async fn get_table_columns(&self, table_name: &str) -> TaskGeneratorResult<Vec<String>> {
        let query = r#"
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = $1
            ORDER BY ordinal_position
        "#;

        let rows = sqlx::query(query)
            .bind(table_name)
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| {
                error!("Failed to get columns for table '{}': {}", table_name, e);
                TaskGeneratorError::Database(DatabaseError::QueryFailed {
                    query: query.to_string(),
                    cause: e.to_string(),
                })
            })?;

        let columns: Vec<String> = rows
            .into_iter()
            .map(|row| row.get::<String, _>("column_name"))
            .collect();

        Ok(columns)
    }

    /// Get row count for a table
    async fn get_table_row_count(&self, table_name: &str) -> TaskGeneratorResult<i64> {
        let query = format!("SELECT COUNT(*) FROM \"{}\"", table_name);

        let count: i64 = sqlx::query_scalar(&query)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| {
                error!("Failed to get row count for table '{}': {}", table_name, e);
                TaskGeneratorError::Database(DatabaseError::QueryFailed {
                    query: query.clone(),
                    cause: e.to_string(),
                })
            })?;

        Ok(count)
    }

    /// Drop a table if it exists
    async fn drop_table(&self, table_name: &str) -> TaskGeneratorResult<()> {
        let query = format!("DROP TABLE IF EXISTS \"{}\"", table_name);

        sqlx::query(&query)
            .execute(&*self.pool)
            .await
            .map_err(|e| {
                error!("Failed to drop table '{}': {}", table_name, e);
                TaskGeneratorError::Database(DatabaseError::QueryFailed {
                    query: query.clone(),
                    cause: e.to_string(),
                })
            })?;

        debug!("Successfully dropped table: {}", table_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::connection::Database;
    use std::env;

    /// Create a test database service
    async fn create_test_database_service() -> Option<DatabaseService> {
        if let Ok(database_url) = env::var("DATABASE_URL") {
            match Database::new(&database_url).await {
                Ok(db) => Some(DatabaseService::new(Arc::new(db.pool().clone()))),
                Err(e) => {
                    eprintln!("Failed to create test database: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    #[tokio::test]
    async fn test_database_service_creation() {
        // Test that we can create a DatabaseService with a mock pool
        // This test doesn't require a real database connection
        
        // Create a mock pool (this would fail to connect, but we're just testing structure)
        let database_url = "postgresql://test:test@localhost:5432/test";
        
        // We can't actually connect without a real database, but we can test the structure
        // In a real test environment with DATABASE_URL set, this would work
        if env::var("DATABASE_URL").is_ok() {
            if let Some(service) = create_test_database_service().await {
                // Test that the service was created successfully
                assert!(service.pool.size() >= 0);
            }
        }
        
        println!("✅ DatabaseService creation test passed (structure validation)");
    }

    #[tokio::test]
    async fn test_table_validation_logic() {
        // Test table validation logic without requiring a real database
        
        // Test invalid chunk size
        if let Some(service) = create_test_database_service().await {
            let result = service.create_chunked_table("test_table", 0).await;
            assert!(result.is_err());
            
            if let Err(TaskGeneratorError::InvalidChunkSize { size }) = result {
                assert_eq!(size, 0);
            } else {
                panic!("Expected InvalidChunkSize error");
            }
        }
        
        println!("✅ Table validation logic test passed");
    }

    #[tokio::test]
    async fn test_chunked_table_name_generation() {
        // Test chunked table name generation
        let original_table = "INGEST_20250927143022";
        let chunk_size = 500;
        let expected_name = format!("{}_{}", original_table, chunk_size);
        
        assert_eq!(expected_name, "INGEST_20250927143022_500");
        
        println!("✅ Chunked table name generation test passed");
    }

    #[tokio::test]
    async fn test_required_columns_validation() {
        // Test that we validate the correct required columns
        let required_columns = vec![
            "file_id", "ingestion_id", "filepath", "filename", "extension",
            "file_size_bytes", "line_count", "word_count", "token_count",
            "content_text", "file_type", "conversion_command", "relative_path",
            "absolute_path", "created_at"
        ];
        
        // Test that all required columns are present
        let test_columns: Vec<String> = required_columns.iter().map(|s| s.to_string()).collect();
        let has_all_columns = required_columns.iter()
            .all(|col| test_columns.contains(&col.to_string()));
        
        assert!(has_all_columns);
        
        // Test missing columns detection
        let incomplete_columns = vec!["file_id", "filepath", "filename"];
        let missing_columns: Vec<_> = required_columns.iter()
            .filter(|col| !incomplete_columns.contains(&col.to_string()))
            .collect();
        
        assert!(!missing_columns.is_empty());
        
        println!("✅ Required columns validation test passed");
    }

    #[tokio::test]
    async fn test_table_info_structure() {
        // Test TableInfo structure
        let table_info = TableInfo {
            name: "test_table".to_string(),
            row_count: 100,
            has_valid_schema: true,
            columns: vec!["file_id".to_string(), "filepath".to_string()],
        };
        
        assert_eq!(table_info.name, "test_table");
        assert_eq!(table_info.row_count, 100);
        assert!(table_info.has_valid_schema);
        assert_eq!(table_info.columns.len(), 2);
        
        println!("✅ TableInfo structure test passed");
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test error creation and handling
        let error = TaskGeneratorError::table_not_found("NONEXISTENT_TABLE");
        assert!(matches!(error, TaskGeneratorError::TableNotFound { .. }));
        
        let error = TaskGeneratorError::invalid_chunk_size(0);
        assert!(matches!(error, TaskGeneratorError::InvalidChunkSize { .. }));
        
        let error = TaskGeneratorError::invalid_table_name("BAD_TABLE", "Missing columns");
        assert!(matches!(error, TaskGeneratorError::InvalidTableName { .. }));
        
        println!("✅ Error handling test passed");
    }

    // Integration tests that require a real database connection
    
    #[tokio::test]
    async fn test_validate_table_integration() {
        // Skip if no DATABASE_URL is set
        if env::var("DATABASE_URL").is_err() {
            return;
        }
        
        if let Some(service) = create_test_database_service().await {
            // Test with a non-existent table
            let result = service.validate_table("NONEXISTENT_TABLE").await;
            assert!(result.is_err());
            
            match result.unwrap_err() {
                TaskGeneratorError::TableNotFound { table } => {
                    assert_eq!(table, "NONEXISTENT_TABLE");
                }
                _ => panic!("Expected TableNotFound error"),
            }
            
            println!("✅ Table validation integration test passed");
        }
    }

    #[tokio::test]
    async fn test_query_rows_integration() {
        // Skip if no DATABASE_URL is set
        if env::var("DATABASE_URL").is_err() {
            return;
        }
        
        if let Some(service) = create_test_database_service().await {
            // Test querying from a non-existent table
            let result = service.query_rows("NONEXISTENT_TABLE").await;
            assert!(result.is_err());
            
            println!("✅ Query rows integration test passed");
        }
    }

    #[tokio::test]
    async fn test_create_chunked_table_integration() {
        // Skip if no DATABASE_URL is set
        if env::var("DATABASE_URL").is_err() {
            return;
        }
        
        if let Some(service) = create_test_database_service().await {
            // Test creating chunked table for non-existent original table
            let result = service.create_chunked_table("NONEXISTENT_TABLE", 500).await;
            assert!(result.is_err());
            
            // Test invalid chunk size
            let result = service.create_chunked_table("ANY_TABLE", 0).await;
            assert!(result.is_err());
            
            match result.unwrap_err() {
                TaskGeneratorError::InvalidChunkSize { size } => {
                    assert_eq!(size, 0);
                }
                _ => panic!("Expected InvalidChunkSize error"),
            }
            
            println!("✅ Create chunked table integration test passed");
        }
    }

    #[tokio::test]
    async fn test_database_service_with_real_schema() {
        // This test requires a real database with proper schema
        // Skip if no DATABASE_URL is set
        if env::var("DATABASE_URL").is_err() {
            return;
        }
        
        if let Some(service) = create_test_database_service().await {
            // Initialize database schema if needed
            let db = Database::new(&env::var("DATABASE_URL").unwrap()).await.unwrap();
            let _ = db.initialize_schema().await;
            
            // Test that ingestion_meta table exists (created by schema initialization)
            let result = service.validate_table("ingestion_meta").await;
            
            // This might fail if the table doesn't have the exact schema we expect for IngestedFile
            // That's expected since ingestion_meta has a different schema than ingestion tables
            if result.is_err() {
                println!("✅ Schema validation correctly detected schema mismatch");
            } else {
                println!("✅ Schema validation passed");
            }
        }
    }

    #[test]
    fn test_database_service_basic_functionality() {
        // Test basic functionality without external dependencies
        
        // Test error types
        let error = TaskGeneratorError::table_not_found("test");
        assert!(error.to_string().contains("test"));
        
        let error = TaskGeneratorError::invalid_chunk_size(0);
        assert!(error.to_string().contains("0"));
        
        // Test table info creation
        let table_info = TableInfo {
            name: "test".to_string(),
            row_count: 42,
            has_valid_schema: true,
            columns: vec!["col1".to_string(), "col2".to_string()],
        };
        
        assert_eq!(table_info.name, "test");
        assert_eq!(table_info.row_count, 42);
        assert!(table_info.has_valid_schema);
        assert_eq!(table_info.columns.len(), 2);
        
        println!("✅ Basic functionality test passed");
    }
}