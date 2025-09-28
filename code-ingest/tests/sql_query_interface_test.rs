//! Integration tests for SQL query interface functionality
//!
//! Tests the enhanced SQL execution with pagination, timeout, and error handling.

use code_ingest::database::{Database, QueryExecutor, query_executor::QueryConfig};
use code_ingest::error::DatabaseResult;
use std::time::Duration;
use tokio::time::timeout;

/// Helper function to create a test database connection
async fn create_test_database() -> DatabaseResult<Database> {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/code_ingest_test".to_string());
    
    Database::new(&database_url).await
}

/// Helper function to setup test data
async fn setup_test_data(database: &Database) -> DatabaseResult<String> {
    use code_ingest::database::SchemaManager;
    
    let schema_manager = SchemaManager::new(database.pool().clone());
    schema_manager.initialize_schema().await?;
    
    // Create a test ingestion table with sample data
    let table_name = format!("INGEST_TEST_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"));
    
    let create_sql = format!(
        r#"
        CREATE TABLE "{}" (
            file_id BIGSERIAL PRIMARY KEY,
            ingestion_id BIGINT DEFAULT 1,
            filepath VARCHAR NOT NULL,
            filename VARCHAR NOT NULL,
            extension VARCHAR,
            file_size_bytes BIGINT NOT NULL,
            line_count INTEGER,
            word_count INTEGER,
            token_count INTEGER,
            content_text TEXT,
            file_type VARCHAR NOT NULL,
            conversion_command VARCHAR,
            relative_path VARCHAR NOT NULL,
            absolute_path VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
        "#,
        table_name
    );
    
    sqlx::query(&create_sql).execute(database.pool()).await.unwrap();
    
    // Insert test data
    for i in 1..=100 {
        let insert_sql = format!(
            r#"
            INSERT INTO "{}" (filepath, filename, extension, file_size_bytes, line_count, 
                           word_count, token_count, content_text, file_type, relative_path, absolute_path)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
            table_name
        );
        
        sqlx::query(&insert_sql)
            .bind(format!("/test/file{}.rs", i))
            .bind(format!("file{}.rs", i))
            .bind("rs")
            .bind(1024_i64)
            .bind(50_i32)
            .bind(200_i32)
            .bind(180_i32)
            .bind(format!("fn test_function_{}() {{ println!(\"Hello {}\"); }}", i, i))
            .bind("direct_text")
            .bind(format!("file{}.rs", i))
            .bind(format!("/test/file{}.rs", i))
            .execute(database.pool())
            .await
            .unwrap();
    }
    
    Ok(table_name)
}

#[tokio::test]
async fn test_sql_query_basic_execution() {
    if let Ok(database) = create_test_database().await {
        let table_name = setup_test_data(&database).await.unwrap();
        let executor = QueryExecutor::new(database.pool().clone());
        
        let query = format!("SELECT COUNT(*) as count FROM \"{}\"", table_name);
        let result = executor.execute_query_terminal(&query).await.unwrap();
        
        assert!(result.content.contains("100"));
        assert_eq!(result.row_count, 1);
        assert!(!result.truncated);
        
        // Cleanup
        let drop_sql = format!("DROP TABLE \"{}\"", table_name);
        sqlx::query(&drop_sql).execute(database.pool()).await.unwrap();
    }
}

#[tokio::test]
async fn test_sql_query_pagination() {
    if let Ok(database) = create_test_database().await {
        let table_name = setup_test_data(&database).await.unwrap();
        let executor = QueryExecutor::new(database.pool().clone());
        
        // Test with limit
        let config = QueryConfig {
            max_rows: 10,
            offset: 0,
            ..Default::default()
        };
        
        let query = format!("SELECT * FROM \"{}\" ORDER BY file_id", table_name);
        let result = executor.execute_query_with_config(&query, &config).await.unwrap();
        
        assert_eq!(result.row_count, 10);
        assert!(!result.truncated); // Not truncated because we applied pagination
        
        // Test with offset
        let config = QueryConfig {
            max_rows: 10,
            offset: 20,
            ..Default::default()
        };
        
        let result = executor.execute_query_with_config(&query, &config).await.unwrap();
        assert_eq!(result.row_count, 10);
        
        // Cleanup
        let drop_sql = format!("DROP TABLE \"{}\"", table_name);
        sqlx::query(&drop_sql).execute(database.pool()).await.unwrap();
    }
}

#[tokio::test]
async fn test_sql_query_timeout() {
    if let Ok(database) = create_test_database().await {
        let executor = QueryExecutor::new(database.pool().clone());
        
        // Test with very short timeout on a slow query
        let config = QueryConfig {
            timeout_seconds: 1, // 1 second timeout
            ..Default::default()
        };
        
        // This query should timeout (simulated slow query)
        let slow_query = "SELECT pg_sleep(2)"; // Sleep for 2 seconds
        let result = executor.execute_query_with_config(slow_query, &config).await;
        
        assert!(result.is_err());
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("timed out") || error_msg.contains("timeout"));
    }
}

#[tokio::test]
async fn test_sql_query_error_enhancement() {
    if let Ok(database) = create_test_database().await {
        let executor = QueryExecutor::new(database.pool().clone());
        
        let config = QueryConfig {
            show_helpful_errors: true,
            ..Default::default()
        };
        
        // Test table not found error
        let result = executor.execute_query_with_config("SELECT * FROM nonexistent_table", &config).await;
        assert!(result.is_err());
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("Helpful suggestions"));
        assert!(error_msg.contains("list-tables"));
        
        // Test syntax error
        let result = executor.execute_query_with_config("SELCT * FROM test", &config).await;
        assert!(result.is_err());
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("syntax error") || error_msg.contains("Helpful suggestions"));
    }
}

#[tokio::test]
async fn test_sql_query_llm_format() {
    if let Ok(database) = create_test_database().await {
        let table_name = setup_test_data(&database).await.unwrap();
        let executor = QueryExecutor::new(database.pool().clone());
        
        let config = QueryConfig {
            llm_format: true,
            max_rows: 2,
            ..Default::default()
        };
        
        let query = format!("SELECT filepath, content_text FROM \"{}\" LIMIT 2", table_name);
        let result = executor.execute_query_with_config(&query, &config).await.unwrap();
        
        // Should contain FILE: markers for LLM format
        assert!(result.content.contains("FILE: /test/file"));
        assert!(result.content.contains("fn test_function_"));
        assert!(result.content.contains("---"));
        
        // Cleanup
        let drop_sql = format!("DROP TABLE \"{}\"", table_name);
        sqlx::query(&drop_sql).execute(database.pool()).await.unwrap();
    }
}

#[tokio::test]
async fn test_sql_query_validation() {
    if let Ok(database) = create_test_database().await {
        let executor = QueryExecutor::new(database.pool().clone());
        
        // Test empty query
        let result = executor.execute_query_terminal("").await;
        assert!(result.is_err());
        
        // Test unbalanced quotes
        let result = executor.execute_query_terminal("SELECT 'unclosed quote FROM test").await;
        assert!(result.is_err());
        
        // Test valid query
        let result = executor.execute_query_terminal("SELECT 1 as test").await;
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_sql_query_performance_contract() {
    if let Ok(database) = create_test_database().await {
        let table_name = setup_test_data(&database).await.unwrap();
        let executor = QueryExecutor::new(database.pool().clone());
        
        // Simple query should execute quickly
        let start = std::time::Instant::now();
        let query = format!("SELECT COUNT(*) FROM \"{}\"", table_name);
        let result = executor.execute_query_terminal(&query).await.unwrap();
        let duration = start.elapsed();
        
        // Should complete within reasonable time (5 seconds for test environment)
        assert!(duration < Duration::from_secs(5));
        assert!(result.execution_time_ms < 5000);
        
        // Cleanup
        let drop_sql = format!("DROP TABLE \"{}\"", table_name);
        sqlx::query(&drop_sql).execute(database.pool()).await.unwrap();
    }
}

#[tokio::test]
async fn test_sql_query_large_result_handling() {
    if let Ok(database) = create_test_database().await {
        let table_name = setup_test_data(&database).await.unwrap();
        let executor = QueryExecutor::new(database.pool().clone());
        
        // Test with row limit smaller than result set
        let config = QueryConfig {
            max_rows: 5,
            ..Default::default()
        };
        
        let query = format!("SELECT * FROM \"{}\"", table_name);
        let result = executor.execute_query_with_config(&query, &config).await.unwrap();
        
        // Should be truncated to 5 rows
        assert_eq!(result.row_count, 5);
        assert!(result.truncated);
        
        // Cleanup
        let drop_sql = format!("DROP TABLE \"{}\"", table_name);
        sqlx::query(&drop_sql).execute(database.pool()).await.unwrap();
    }
}