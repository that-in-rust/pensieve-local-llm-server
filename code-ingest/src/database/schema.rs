//! Database schema management
//!
//! This module handles the creation and management of database schemas,
//! including timestamped ingestion tables, query result tables, and metadata tables.

use crate::error::{DatabaseError, DatabaseResult};
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use tracing::{debug, info, warn};

/// Schema manager for database table operations
pub struct SchemaManager {
    pool: PgPool,
}

/// Types of tables in the system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableType {
    /// INGEST_YYYYMMDDHHMMSS tables for storing ingested code files
    Ingestion,
    /// QUERYRESULT_* tables for storing LLM analysis results
    QueryResult,
    /// ingestion_meta table for tracking ingestion metadata
    Meta,
}

/// Information about a created table
#[derive(Debug, Clone)]
pub struct TableInfo {
    pub name: String,
    pub table_type: TableType,
    pub created_at: DateTime<Utc>,
    pub column_count: usize,
}

impl SchemaManager {
    /// Create a new schema manager
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Initialize the database schema by creating the ingestion_meta table
    pub async fn initialize_schema(&self) -> DatabaseResult<()> {
        info!("Initializing database schema");
        
        self.create_ingestion_meta_table().await?;
        self.create_indexes().await?;
        
        info!("Database schema initialization complete");
        Ok(())
    }

    /// Create a new timestamped ingestion table
    /// Returns the table name in format INGEST_YYYYMMDDHHMMSS
    pub async fn create_ingestion_table(&self, timestamp: Option<DateTime<Utc>>) -> DatabaseResult<String> {
        let timestamp = timestamp.unwrap_or_else(Utc::now);
        let table_name = format!("INGEST_{}", timestamp.format("%Y%m%d%H%M%S"));
        
        debug!("Creating ingestion table: {}", table_name);
        
        let create_sql = format!(
            r#"
            CREATE TABLE "{}" (
                file_id BIGSERIAL PRIMARY KEY,
                ingestion_id BIGINT NOT NULL,
                filepath VARCHAR NOT NULL,
                filename VARCHAR NOT NULL,
                extension VARCHAR,
                file_size_bytes BIGINT NOT NULL,
                line_count INTEGER,
                word_count INTEGER,
                token_count INTEGER,
                content_text TEXT,
                file_type VARCHAR NOT NULL CHECK (file_type IN ('direct_text', 'convertible', 'non_text')),
                conversion_command VARCHAR,
                relative_path VARCHAR NOT NULL,
                absolute_path VARCHAR NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Multi-scale context columns for knowledge arbitrage
                parent_filepath VARCHAR,
                l1_window_content TEXT,
                l2_window_content TEXT,
                ast_patterns JSONB
            )
            "#,
            table_name
        );

        sqlx::query(&create_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::TableCreationFailed {
                table_name: table_name.clone(),
                cause: e.to_string(),
            })?;

        // Create indexes for the new table
        self.create_ingestion_table_indexes(&table_name).await?;

        info!("Created ingestion table: {}", table_name);
        Ok(table_name)
    }

    /// Create a query result table with dynamic naming
    pub async fn create_query_result_table(&self, table_suffix: &str) -> DatabaseResult<String> {
        let table_name = format!("QUERYRESULT_{}", table_suffix.to_uppercase());
        
        debug!("Creating query result table: {}", table_name);
        
        let create_sql = format!(
            r#"
            CREATE TABLE "{}" (
                analysis_id BIGSERIAL PRIMARY KEY,
                sql_query TEXT NOT NULL,
                prompt_file_path VARCHAR,
                llm_result TEXT NOT NULL,
                original_file_path VARCHAR,
                chunk_number INTEGER,
                analysis_type VARCHAR,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            "#,
            table_name
        );

        sqlx::query(&create_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::TableCreationFailed {
                table_name: table_name.clone(),
                cause: e.to_string(),
            })?;

        // Create indexes for the new table
        self.create_query_result_table_indexes(&table_name).await?;

        info!("Created query result table: {}", table_name);
        Ok(table_name)
    }

    /// Check if a table exists
    pub async fn table_exists(&self, table_name: &str) -> DatabaseResult<bool> {
        let exists_query = r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = $1
            )
        "#;

        let exists: (bool,) = sqlx::query_as(exists_query)
            .bind(table_name)  // Don't convert to lowercase - PostgreSQL table names are case-sensitive
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: exists_query.to_string(),
                cause: e.to_string(),
            })?;

        Ok(exists.0)
    }

    /// List all tables of a specific type
    pub async fn list_tables(&self, table_type: Option<TableType>) -> DatabaseResult<Vec<String>> {
        let query = r#"
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        "#;

        let rows: Vec<(String,)> = sqlx::query_as(query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.to_string(),
                cause: e.to_string(),
            })?;

        let mut tables: Vec<String> = rows.into_iter().map(|(name,)| name).collect();

        // Filter by table type if specified
        if let Some(filter_type) = table_type {
            tables = tables
                .into_iter()
                .filter(|name| self.classify_table_type(name) == filter_type)
                .collect();
        }

        Ok(tables)
    }

    /// Get information about a table
    pub async fn get_table_info(&self, table_name: &str) -> DatabaseResult<TableInfo> {
        // Check if table exists
        if !self.table_exists(table_name).await? {
            return Err(DatabaseError::QueryFailed {
                query: format!("Table info for {}", table_name),
                cause: "Table does not exist".to_string(),
            });
        }

        // Get column count
        let column_query = r#"
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = $1
        "#;

        let (column_count,): (i64,) = sqlx::query_as(column_query)
            .bind(table_name.to_lowercase())
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: column_query.to_string(),
                cause: e.to_string(),
            })?;

        Ok(TableInfo {
            name: table_name.to_string(),
            table_type: self.classify_table_type(table_name),
            created_at: Utc::now(), // We could get this from pg_class if needed
            column_count: column_count as usize,
        })
    }

    /// Drop a table (use with caution)
    pub async fn drop_table(&self, table_name: &str) -> DatabaseResult<()> {
        warn!("Dropping table: {}", table_name);
        
        let drop_sql = format!("DROP TABLE IF EXISTS \"{}\" CASCADE", table_name);
        
        sqlx::query(&drop_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: drop_sql,
                cause: e.to_string(),
            })?;

        info!("Dropped table: {}", table_name);
        Ok(())
    }

    /// Validate table schema matches expected structure
    pub async fn validate_table_schema(&self, table_name: &str, expected_type: TableType) -> DatabaseResult<bool> {
        let columns_query = r#"
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = $1
            ORDER BY ordinal_position
        "#;

        let columns: Vec<(String, String, String)> = sqlx::query_as(columns_query)
            .bind(table_name.to_lowercase())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: columns_query.to_string(),
                cause: e.to_string(),
            })?;

        // Validate based on expected type
        match expected_type {
            TableType::Ingestion => self.validate_ingestion_table_schema(&columns),
            TableType::QueryResult => self.validate_query_result_table_schema(&columns),
            TableType::Meta => self.validate_meta_table_schema(&columns),
        }
    }

    // Private helper methods

    async fn create_ingestion_meta_table(&self) -> DatabaseResult<()> {
        let create_sql = r#"
            CREATE TABLE IF NOT EXISTS ingestion_meta (
                ingestion_id BIGSERIAL PRIMARY KEY,
                repo_url VARCHAR,
                local_path VARCHAR NOT NULL,
                start_timestamp_unix BIGINT NOT NULL,
                end_timestamp_unix BIGINT,
                table_name VARCHAR NOT NULL,
                total_files_processed INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        "#;

        sqlx::query(create_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::TableCreationFailed {
                table_name: "ingestion_meta".to_string(),
                cause: e.to_string(),
            })?;

        debug!("Created or verified ingestion_meta table");
        Ok(())
    }

    async fn create_indexes(&self) -> DatabaseResult<()> {
        let indexes = vec![
            "CREATE INDEX IF NOT EXISTS idx_ingestion_meta_table_name ON ingestion_meta(table_name)",
            "CREATE INDEX IF NOT EXISTS idx_ingestion_meta_repo_url ON ingestion_meta(repo_url)",
            "CREATE INDEX IF NOT EXISTS idx_ingestion_meta_created_at ON ingestion_meta(created_at)",
        ];

        for index_sql in indexes {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: index_sql.to_string(),
                    cause: e.to_string(),
                })?;
        }

        debug!("Created base indexes");
        Ok(())
    }

    async fn create_ingestion_table_indexes(&self, table_name: &str) -> DatabaseResult<()> {
        let indexes = vec![
            format!("CREATE INDEX IF NOT EXISTS idx_{}_filepath ON \"{}\"(filepath)", 
                   table_name.to_lowercase(), table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_extension ON \"{}\"(extension)", 
                   table_name.to_lowercase(), table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_file_type ON \"{}\"(file_type)", 
                   table_name.to_lowercase(), table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_created_at ON \"{}\"(created_at)", 
                   table_name.to_lowercase(), table_name),
            // New indexes for multi-scale context columns
            format!("CREATE INDEX IF NOT EXISTS idx_{}_parent_filepath ON \"{}\"(parent_filepath)", 
                   table_name.to_lowercase(), table_name),
        ];

        // Create full-text search index on content
        let fts_index = format!(
            "CREATE INDEX IF NOT EXISTS idx_{}_content_fts ON \"{}\" USING gin(to_tsvector('english', content_text))",
            table_name.to_lowercase(), table_name
        );
        
        // Create JSONB index for ast_patterns
        let ast_patterns_index = format!(
            "CREATE INDEX IF NOT EXISTS idx_{}_ast_patterns ON \"{}\" USING gin(ast_patterns)",
            table_name.to_lowercase(), table_name
        );
        
        for index_sql in indexes.into_iter().chain(std::iter::once(fts_index)).chain(std::iter::once(ast_patterns_index)) {
            sqlx::query(&index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: index_sql.clone(),
                    cause: e.to_string(),
                })?;
        }

        debug!("Created indexes for table: {}", table_name);
        Ok(())
    }

    async fn create_query_result_table_indexes(&self, table_name: &str) -> DatabaseResult<()> {
        let indexes = vec![
            format!("CREATE INDEX IF NOT EXISTS idx_{}_created_at ON \"{}\"(created_at)", 
                   table_name.to_lowercase(), table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_prompt_file ON \"{}\"(prompt_file_path)", 
                   table_name.to_lowercase(), table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_analysis_type ON \"{}\"(analysis_type)", 
                   table_name.to_lowercase(), table_name),
        ];

        for index_sql in indexes {
            sqlx::query(&index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: index_sql.clone(),
                    cause: e.to_string(),
                })?;
        }

        debug!("Created indexes for query result table: {}", table_name);
        Ok(())
    }

    fn classify_table_type(&self, table_name: &str) -> TableType {
        let name_upper = table_name.to_uppercase();
        if name_upper.starts_with("INGEST_") {
            TableType::Ingestion
        } else if name_upper.starts_with("QUERYRESULT_") {
            TableType::QueryResult
        } else if name_upper == "INGESTION_META" {
            TableType::Meta
        } else {
            // Default to ingestion for unknown types
            TableType::Ingestion
        }
    }

    fn validate_ingestion_table_schema(&self, columns: &[(String, String, String)]) -> DatabaseResult<bool> {
        let required_columns = vec![
            "file_id", "ingestion_id", "filepath", "filename", "extension",
            "file_size_bytes", "line_count", "word_count", "token_count",
            "content_text", "file_type", "conversion_command", "relative_path",
            "absolute_path", "created_at", 
            // New multi-scale context columns
            "parent_filepath", "l1_window_content", "l2_window_content", "ast_patterns"
        ];

        let column_names: Vec<&str> = columns.iter().map(|(name, _, _)| name.as_str()).collect();
        
        for required in &required_columns {
            if !column_names.contains(required) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn validate_query_result_table_schema(&self, columns: &[(String, String, String)]) -> DatabaseResult<bool> {
        let required_columns = vec![
            "analysis_id", "sql_query", "prompt_file_path", "llm_result",
            "original_file_path", "chunk_number", "analysis_type", "created_at"
        ];

        let column_names: Vec<&str> = columns.iter().map(|(name, _, _)| name.as_str()).collect();
        
        for required in &required_columns {
            if !column_names.contains(required) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn validate_meta_table_schema(&self, columns: &[(String, String, String)]) -> DatabaseResult<bool> {
        let required_columns = vec![
            "ingestion_id", "repo_url", "local_path", "start_timestamp_unix",
            "end_timestamp_unix", "table_name", "total_files_processed", "created_at"
        ];

        let column_names: Vec<&str> = columns.iter().map(|(name, _, _)| name.as_str()).collect();
        
        for required in &required_columns {
            if !column_names.contains(required) {
                return Ok(false);
            }
        }

        Ok(true)
    }
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

    #[tokio::test]
    async fn test_schema_manager_creation() {
        if let Some(pool) = create_test_pool() {
            let _schema_manager = SchemaManager::new(pool);
            // Just test that we can create the manager
            assert!(true);
        }
    }

    #[tokio::test]
    async fn test_table_type_classification() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            assert_eq!(schema_manager.classify_table_type("INGEST_20250927143022"), TableType::Ingestion);
            assert_eq!(schema_manager.classify_table_type("QUERYRESULT_auth_analysis"), TableType::QueryResult);
            assert_eq!(schema_manager.classify_table_type("ingestion_meta"), TableType::Meta);
            assert_eq!(schema_manager.classify_table_type("unknown_table"), TableType::Ingestion);
        }
    }

    #[tokio::test]
    async fn test_ingestion_table_creation() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Initialize schema first
            schema_manager.initialize_schema().await.unwrap();
            
            // Create a test ingestion table
            let timestamp = chrono::Utc::now();
            let table_name = schema_manager.create_ingestion_table(Some(timestamp)).await.unwrap();
            
            assert!(table_name.starts_with("INGEST_"));
            assert_eq!(table_name.len(), 21); // INGEST_ + 14 digit timestamp
            
            // Verify table exists
            let exists = schema_manager.table_exists(&table_name).await.unwrap();
            assert!(exists);
            
            // Clean up
            schema_manager.drop_table(&table_name).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_query_result_table_creation() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            let table_name = schema_manager.create_query_result_table("test_analysis").await.unwrap();
            assert_eq!(table_name, "QUERYRESULT_TEST_ANALYSIS");
            
            // Verify table exists
            let exists = schema_manager.table_exists(&table_name).await.unwrap();
            assert!(exists);
            
            // Clean up
            schema_manager.drop_table(&table_name).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_list_tables() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Initialize schema
            schema_manager.initialize_schema().await.unwrap();
            
            // Create test tables
            let ingestion_table = schema_manager.create_ingestion_table(None).await.unwrap();
            let query_table = schema_manager.create_query_result_table("test").await.unwrap();
            
            // List all tables
            let all_tables = schema_manager.list_tables(None).await.unwrap();
            assert!(all_tables.contains(&ingestion_table));
            assert!(all_tables.contains(&query_table));
            assert!(all_tables.contains(&"ingestion_meta".to_string()));
            
            // List only ingestion tables
            let ingestion_tables = schema_manager.list_tables(Some(TableType::Ingestion)).await.unwrap();
            assert!(ingestion_tables.contains(&ingestion_table));
            assert!(!ingestion_tables.contains(&query_table));
            
            // Clean up
            schema_manager.drop_table(&ingestion_table).await.unwrap();
            schema_manager.drop_table(&query_table).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_table_info() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            let table_name = schema_manager.create_ingestion_table(None).await.unwrap();
            
            let info = schema_manager.get_table_info(&table_name).await.unwrap();
            assert_eq!(info.name, table_name);
            assert_eq!(info.table_type, TableType::Ingestion);
            assert!(info.column_count > 0);
            
            // Clean up
            schema_manager.drop_table(&table_name).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_schema_validation() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Initialize schema
            schema_manager.initialize_schema().await.unwrap();
            
            // Test meta table validation
            let is_valid = schema_manager.validate_table_schema("ingestion_meta", TableType::Meta).await.unwrap();
            assert!(is_valid);
            
            // Create and test ingestion table validation
            let table_name = schema_manager.create_ingestion_table(None).await.unwrap();
            let is_valid = schema_manager.validate_table_schema(&table_name, TableType::Ingestion).await.unwrap();
            assert!(is_valid);
            
            // Clean up
            schema_manager.drop_table(&table_name).await.unwrap();
        }
    }

    #[test]
    fn test_table_name_generation() {
        let timestamp = chrono::DateTime::parse_from_rfc3339("2025-09-27T14:30:22Z").unwrap().with_timezone(&Utc);
        let expected = "INGEST_20250927143022";
        let actual = format!("INGEST_{}", timestamp.format("%Y%m%d%H%M%S"));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_query_result_table_naming() {
        let table_name = format!("QUERYRESULT_{}", "auth_analysis".to_uppercase());
        assert_eq!(table_name, "QUERYRESULT_AUTH_ANALYSIS");
        
        let table_name = format!("QUERYRESULT_{}", "test-with-dashes".to_uppercase());
        assert_eq!(table_name, "QUERYRESULT_TEST-WITH-DASHES");
    }

    #[tokio::test]
    async fn test_nonexistent_table_operations() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Test operations on nonexistent table
            let exists = schema_manager.table_exists("nonexistent_table").await.unwrap();
            assert!(!exists);
            
            let info_result = schema_manager.get_table_info("nonexistent_table").await;
            assert!(info_result.is_err());
        }
    }
}