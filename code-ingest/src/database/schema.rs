//! Database schema management
//!
//! This module handles the creation and management of database schemas,
//! including timestamped ingestion tables, query result tables, and metadata tables.

use crate::error::{DatabaseError, DatabaseResult};
use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
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

    /// Create a chunked table with naming pattern <TableName_ChunkSize>
    /// Returns the chunked table name
    pub async fn create_chunked_table(&self, base_table_name: &str, chunk_size: usize) -> DatabaseResult<String> {
        let chunked_table_name = format!("{}_{}", base_table_name, chunk_size);
        
        debug!("Creating chunked table: {}", chunked_table_name);
        
        let create_sql = format!(
            r#"
            CREATE TABLE "{}" (
                id BIGSERIAL PRIMARY KEY,
                file_id TEXT NOT NULL,
                filepath TEXT NOT NULL,
                parent_filepath TEXT NOT NULL,
                filename TEXT NOT NULL,
                extension TEXT,
                chunk_number INTEGER NOT NULL,
                chunk_start_line INTEGER NOT NULL,
                chunk_end_line INTEGER NOT NULL,
                line_count INTEGER,
                content TEXT,
                content_l1 TEXT,  -- Context with ±1 chunk
                content_l2 TEXT,  -- Context with ±2 chunks
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- Constraints
                CONSTRAINT check_{}_chunk_number_positive CHECK (chunk_number > 0),
                CONSTRAINT check_{}_line_numbers CHECK (chunk_end_line >= chunk_start_line)
            )
            "#,
            chunked_table_name,
            chunked_table_name.to_lowercase().replace("-", "_"),
            chunked_table_name.to_lowercase().replace("-", "_")
        );

        sqlx::query(&create_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::TableCreationFailed {
                table_name: chunked_table_name.clone(),
                cause: e.to_string(),
            })?;

        // Create indexes for the chunked table
        self.create_chunked_table_indexes(&chunked_table_name).await?;

        info!("Created chunked table: {}", chunked_table_name);
        Ok(chunked_table_name)
    }

    /// Validate that a table name follows the chunked table naming pattern
    pub fn is_chunked_table_name(&self, table_name: &str) -> bool {
        // Pattern: TABLENAME_CHUNKSIZE where CHUNKSIZE is a number
        if let Some(last_underscore) = table_name.rfind('_') {
            let potential_chunk_size = &table_name[last_underscore + 1..];
            potential_chunk_size.parse::<usize>().is_ok()
        } else {
            false
        }
    }

    /// Extract base table name and chunk size from a chunked table name
    pub fn parse_chunked_table_name(&self, table_name: &str) -> Option<(String, usize)> {
        if !self.is_chunked_table_name(table_name) {
            return None;
        }
        
        if let Some(last_underscore) = table_name.rfind('_') {
            let base_name = table_name[..last_underscore].to_string();
            let chunk_size_str = &table_name[last_underscore + 1..];
            
            if let Ok(chunk_size) = chunk_size_str.parse::<usize>() {
                return Some((base_name, chunk_size));
            }
        }
        
        None
    }

    /// Check if a chunked table exists for the given base table and chunk size
    pub async fn chunked_table_exists(&self, base_table_name: &str, chunk_size: usize) -> DatabaseResult<bool> {
        let chunked_table_name = format!("{}_{}", base_table_name, chunk_size);
        self.table_exists(&chunked_table_name).await
    }

    /// List all chunked tables for a given base table
    pub async fn list_chunked_tables(&self, base_table_name: &str) -> DatabaseResult<Vec<(String, usize)>> {
        let all_tables = self.list_tables(None).await?;
        let mut chunked_tables = Vec::new();
        
        for table_name in all_tables {
            if let Some((base_name, chunk_size)) = self.parse_chunked_table_name(&table_name) {
                if base_name == base_table_name {
                    chunked_tables.push((table_name, chunk_size));
                }
            }
        }
        
        // Sort by chunk size
        chunked_tables.sort_by_key(|(_, chunk_size)| *chunk_size);
        Ok(chunked_tables)
    }

    /// Get the optimal chunk size for a table based on its content
    pub async fn suggest_chunk_size(&self, table_name: &str) -> DatabaseResult<usize> {
        debug!("Analyzing table {} to suggest optimal chunk size", table_name);
        
        let analysis_sql = format!(
            r#"
            SELECT 
                AVG(line_count) as avg_lines,
                MAX(line_count) as max_lines,
                MIN(line_count) as min_lines,
                COUNT(*) as total_files
            FROM "{}"
            WHERE line_count IS NOT NULL
            "#,
            table_name
        );

        let row = sqlx::query(&analysis_sql)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: analysis_sql,
                cause: e.to_string(),
            })?;

        let avg_lines: Option<f64> = row.get("avg_lines");
        let max_lines: Option<i32> = row.get("max_lines");
        let total_files: i64 = row.get("total_files");

        // Suggest chunk size based on analysis
        let suggested_size = match (avg_lines, max_lines) {
            (Some(avg), Some(max)) => {
                if max > 10000 {
                    // Very large files - use smaller chunks
                    1000
                } else if avg > 1000.0 {
                    // Large files - use medium chunks
                    500
                } else if avg > 100.0 {
                    // Medium files - use larger chunks
                    200
                } else {
                    // Small files - use very large chunks or no chunking
                    100
                }
            }
            _ => {
                // Default chunk size if no line count data
                if total_files > 1000 {
                    500
                } else {
                    200
                }
            }
        };

        debug!("Suggested chunk size for table {}: {}", table_name, suggested_size);
        Ok(suggested_size)
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

    async fn create_chunked_table_indexes(&self, table_name: &str) -> DatabaseResult<()> {
        let safe_name = table_name.to_lowercase().replace("-", "_");
        let indexes = vec![
            format!("CREATE INDEX IF NOT EXISTS idx_{}_filepath ON \"{}\"(filepath)", 
                   safe_name, table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_parent_filepath ON \"{}\"(parent_filepath)", 
                   safe_name, table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_chunk_number ON \"{}\"(chunk_number)", 
                   safe_name, table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_file_id ON \"{}\"(file_id)", 
                   safe_name, table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_extension ON \"{}\"(extension)", 
                   safe_name, table_name),
            format!("CREATE INDEX IF NOT EXISTS idx_{}_created_at ON \"{}\"(created_at)", 
                   safe_name, table_name),
            // Composite index for efficient chunk ordering
            format!("CREATE INDEX IF NOT EXISTS idx_{}_file_chunk ON \"{}\"(file_id, chunk_number)", 
                   safe_name, table_name),
        ];

        // Create full-text search index on content
        let fts_index = format!(
            "CREATE INDEX IF NOT EXISTS idx_{}_content_fts ON \"{}\" USING gin(to_tsvector('english', content))",
            safe_name, table_name
        );
        
        for index_sql in indexes.into_iter().chain(std::iter::once(fts_index)) {
            sqlx::query(&index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: index_sql.clone(),
                    cause: e.to_string(),
                })?;
        }

        debug!("Created indexes for chunked table: {}", table_name);
        Ok(())
    }

    fn classify_table_type(&self, table_name: &str) -> TableType {
        let name_upper = table_name.to_uppercase();
        if name_upper.starts_with("INGEST_") {
            // Check if it's a chunked table (has _NUMBER at the end)
            if self.is_chunked_table_name(table_name) {
                // For chunked tables, we still classify as Ingestion but could add a new type
                TableType::Ingestion
            } else {
                TableType::Ingestion
            }
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
        let column_names: Vec<&str> = columns.iter().map(|(name, _, _)| name.as_str()).collect();
        
        // Check if this might be a chunked table by looking for chunk-specific columns
        let has_chunk_columns = column_names.contains(&"chunk_number") && 
                               column_names.contains(&"chunk_start_line") && 
                               column_names.contains(&"chunk_end_line");
        
        if has_chunk_columns {
            // Validate chunked table schema
            let required_chunked_columns = vec![
                "id", "file_id", "filepath", "parent_filepath", "filename", "extension",
                "chunk_number", "chunk_start_line", "chunk_end_line", "line_count",
                "content", "content_l1", "content_l2", "created_at"
            ];
            
            for required in &required_chunked_columns {
                if !column_names.contains(required) {
                    return Ok(false);
                }
            }
        } else {
            // Validate regular ingestion table schema
            let required_columns = vec![
                "file_id", "ingestion_id", "filepath", "filename", "extension",
                "file_size_bytes", "line_count", "word_count", "token_count",
                "content_text", "file_type", "conversion_command", "relative_path",
                "absolute_path", "created_at", 
                // New multi-scale context columns
                "parent_filepath", "l1_window_content", "l2_window_content", "ast_patterns"
            ];
            
            for required in &required_columns {
                if !column_names.contains(required) {
                    return Ok(false);
                }
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

    #[tokio::test]
    async fn test_chunked_table_creation() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Create a chunked table
            let base_table = "INGEST_20250927143022";
            let chunk_size = 500;
            let chunked_table_name = schema_manager.create_chunked_table(base_table, chunk_size).await.unwrap();
            
            assert_eq!(chunked_table_name, format!("{}_{}", base_table, chunk_size));
            
            // Verify table exists
            let exists = schema_manager.table_exists(&chunked_table_name).await.unwrap();
            assert!(exists);
            
            // Verify it's recognized as a chunked table
            assert!(schema_manager.is_chunked_table_name(&chunked_table_name));
            
            // Clean up
            schema_manager.drop_table(&chunked_table_name).await.unwrap();
        }
    }

    #[test]
    fn test_chunked_table_name_parsing() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Test valid chunked table names
            assert!(schema_manager.is_chunked_table_name("INGEST_20250927143022_500"));
            assert!(schema_manager.is_chunked_table_name("TABLE_NAME_1000"));
            
            // Test invalid chunked table names
            assert!(!schema_manager.is_chunked_table_name("INGEST_20250927143022"));
            assert!(!schema_manager.is_chunked_table_name("TABLE_NAME_abc"));
            assert!(!schema_manager.is_chunked_table_name("SIMPLE_TABLE"));
            
            // Test parsing
            let (base_name, chunk_size) = schema_manager.parse_chunked_table_name("INGEST_20250927143022_500").unwrap();
            assert_eq!(base_name, "INGEST_20250927143022");
            assert_eq!(chunk_size, 500);
            
            // Test parsing failure
            assert!(schema_manager.parse_chunked_table_name("INVALID_TABLE").is_none());
        }
    }

    #[tokio::test]
    async fn test_chunked_table_existence_check() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            let base_table = "INGEST_TEST";
            let chunk_size = 200;
            
            // Should not exist initially
            let exists_before = schema_manager.chunked_table_exists(base_table, chunk_size).await.unwrap();
            assert!(!exists_before);
            
            // Create the chunked table
            let chunked_table_name = schema_manager.create_chunked_table(base_table, chunk_size).await.unwrap();
            
            // Should exist now
            let exists_after = schema_manager.chunked_table_exists(base_table, chunk_size).await.unwrap();
            assert!(exists_after);
            
            // Clean up
            schema_manager.drop_table(&chunked_table_name).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_list_chunked_tables() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            let base_table = "INGEST_TEST_LIST";
            let chunk_sizes = vec![100, 500, 1000];
            let mut created_tables = Vec::new();
            
            // Create multiple chunked tables
            for chunk_size in &chunk_sizes {
                let table_name = schema_manager.create_chunked_table(base_table, *chunk_size).await.unwrap();
                created_tables.push(table_name);
            }
            
            // List chunked tables
            let chunked_tables = schema_manager.list_chunked_tables(base_table).await.unwrap();
            
            // Should find all created tables
            assert_eq!(chunked_tables.len(), chunk_sizes.len());
            
            // Verify chunk sizes are sorted
            let found_sizes: Vec<usize> = chunked_tables.iter().map(|(_, size)| *size).collect();
            let mut expected_sizes = chunk_sizes.clone();
            expected_sizes.sort();
            assert_eq!(found_sizes, expected_sizes);
            
            // Clean up
            for table_name in created_tables {
                schema_manager.drop_table(&table_name).await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn test_chunk_size_suggestion() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // This test would require a real table with data
            // For now, we test the error handling for non-existent tables
            let result = schema_manager.suggest_chunk_size("nonexistent_table").await;
            assert!(result.is_err(), "Should fail for non-existent table");
        }
    }

    #[tokio::test]
    async fn test_chunked_table_schema_validation() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Create a chunked table
            let base_table = "INGEST_VALIDATION_TEST";
            let chunk_size = 300;
            let chunked_table_name = schema_manager.create_chunked_table(base_table, chunk_size).await.unwrap();
            
            // Validate the schema
            let is_valid = schema_manager.validate_table_schema(&chunked_table_name, TableType::Ingestion).await.unwrap();
            assert!(is_valid, "Chunked table should have valid schema");
            
            // Clean up
            schema_manager.drop_table(&chunked_table_name).await.unwrap();
        }
    }

    #[test]
    fn test_chunked_table_type_classification() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            // Test classification of chunked table names
            assert_eq!(schema_manager.classify_table_type("INGEST_20250927143022_500"), TableType::Ingestion);
            assert_eq!(schema_manager.classify_table_type("INGEST_TEST_1000"), TableType::Ingestion);
            assert_eq!(schema_manager.classify_table_type("QUERYRESULT_ANALYSIS_200"), TableType::QueryResult);
            
            // Test non-chunked tables
            assert_eq!(schema_manager.classify_table_type("INGEST_20250927143022"), TableType::Ingestion);
            assert_eq!(schema_manager.classify_table_type("ingestion_meta"), TableType::Meta);
        }
    }

    #[tokio::test]
    async fn test_chunked_table_comprehensive_workflow() {
        if let Some(pool) = create_test_pool() {
            let schema_manager = SchemaManager::new(pool);
            
            let base_table = "INGEST_WORKFLOW_TEST";
            let chunk_size = 250;
            
            // 1. Check table doesn't exist
            let exists_before = schema_manager.chunked_table_exists(base_table, chunk_size).await.unwrap();
            assert!(!exists_before);
            
            // 2. Create chunked table
            let chunked_table_name = schema_manager.create_chunked_table(base_table, chunk_size).await.unwrap();
            assert_eq!(chunked_table_name, format!("{}_{}", base_table, chunk_size));
            
            // 3. Verify existence
            let exists_after = schema_manager.chunked_table_exists(base_table, chunk_size).await.unwrap();
            assert!(exists_after);
            
            // 4. Verify it appears in table listings
            let all_tables = schema_manager.list_tables(Some(TableType::Ingestion)).await.unwrap();
            assert!(all_tables.contains(&chunked_table_name));
            
            // 5. Verify chunked table listing
            let chunked_tables = schema_manager.list_chunked_tables(base_table).await.unwrap();
            assert_eq!(chunked_tables.len(), 1);
            assert_eq!(chunked_tables[0].0, chunked_table_name);
            assert_eq!(chunked_tables[0].1, chunk_size);
            
            // 6. Validate schema
            let is_valid = schema_manager.validate_table_schema(&chunked_table_name, TableType::Ingestion).await.unwrap();
            assert!(is_valid);
            
            // 7. Get table info
            let info = schema_manager.get_table_info(&chunked_table_name).await.unwrap();
            assert_eq!(info.name, chunked_table_name);
            assert_eq!(info.table_type, TableType::Ingestion);
            
            // 8. Clean up
            schema_manager.drop_table(&chunked_table_name).await.unwrap();
            
            // 9. Verify deletion
            let exists_final = schema_manager.chunked_table_exists(base_table, chunk_size).await.unwrap();
            assert!(!exists_final);
        }
    }
}