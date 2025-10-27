//! Integration tests for table management functionality
//!
//! Tests table cleanup, optimization, and management recommendations.

use code_ingest::database::{Database, DatabaseExplorer, TableType};
use code_ingest::error::DatabaseResult;
use std::time::Duration;

/// Helper function to create a test database connection
async fn create_test_database() -> DatabaseResult<Database> {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/code_ingest_test".to_string());
    
    Database::new(&database_url).await
}

/// Helper function to create test ingestion tables
async fn create_test_tables(database: &Database, count: usize) -> DatabaseResult<Vec<String>> {
    use code_ingest::database::SchemaManager;
    
    let schema_manager = SchemaManager::new(database.pool().clone());
    schema_manager.initialize_schema().await?;
    
    let mut table_names = Vec::new();
    
    for i in 0..count {
        let table_name = format!("INGEST_TEST_{}_{}", i, chrono::Utc::now().format("%Y%m%d%H%M%S"));
        
        let create_sql = format!(
            r#"
            CREATE TABLE "{}" (
                file_id BIGSERIAL PRIMARY KEY,
                ingestion_id BIGINT DEFAULT {},
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
            table_name, i + 1
        );
        
        sqlx::query(&create_sql).execute(database.pool()).await.unwrap();
        
        // Add some test data to make tables non-empty
        let insert_sql = format!(
            r#"
            INSERT INTO "{}" (filepath, filename, extension, file_size_bytes, line_count, 
                           word_count, token_count, content_text, file_type, relative_path, absolute_path)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
            table_name
        );
        
        for j in 1..=10 {
            sqlx::query(&insert_sql)
                .bind(format!("/test/file{}.rs", j))
                .bind(format!("file{}.rs", j))
                .bind("rs")
                .bind(1024_i64)
                .bind(50_i32)
                .bind(200_i32)
                .bind(180_i32)
                .bind(format!("fn test_function_{}() {{ println!(\"Hello {}\"); }}", j, j))
                .bind("direct_text")
                .bind(format!("file{}.rs", j))
                .bind(format!("/test/file{}.rs", j))
                .execute(database.pool())
                .await
                .unwrap();
        }
        
        table_names.push(table_name);
        
        // Add small delay to ensure different timestamps
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    Ok(table_names)
}

/// Cleanup test tables
async fn cleanup_test_tables(database: &Database, table_names: &[String]) {
    for table_name in table_names {
        let drop_sql = format!("DROP TABLE IF EXISTS \"{}\"", table_name);
        let _ = sqlx::query(&drop_sql).execute(database.pool()).await;
    }
}

#[tokio::test]
async fn test_database_explorer_creation() {
    if let Ok(database) = create_test_database().await {
        let _explorer = DatabaseExplorer::new(database.pool().clone());
        // Just test that we can create the explorer
        assert!(true);
    }
}

#[tokio::test]
async fn test_table_cleanup_functionality() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create 5 test tables
        let table_names = create_test_tables(&database, 5).await.unwrap();
        
        // Verify tables were created
        let initial_tables = explorer.list_tables(Some(TableType::Ingestion)).await.unwrap();
        let test_tables: Vec<_> = initial_tables.iter()
            .filter(|t| t.name.starts_with("INGEST_TEST_"))
            .collect();
        assert_eq!(test_tables.len(), 5);
        
        // Cleanup keeping only 2 tables
        let result = explorer.cleanup_old_tables(2).await.unwrap();
        
        // Should have removed 3 tables, kept 2
        assert_eq!(result.tables_removed, 3);
        assert_eq!(result.tables_kept, 2);
        assert!(result.space_freed_mb >= 0.0);
        
        // Verify cleanup worked
        let remaining_tables = explorer.list_tables(Some(TableType::Ingestion)).await.unwrap();
        let remaining_test_tables: Vec<_> = remaining_tables.iter()
            .filter(|t| t.name.starts_with("INGEST_TEST_"))
            .collect();
        assert_eq!(remaining_test_tables.len(), 2);
        
        // Cleanup remaining tables
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_table_cleanup_no_action_needed() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create 2 test tables
        let table_names = create_test_tables(&database, 2).await.unwrap();
        
        // Try to cleanup keeping 5 tables (more than we have)
        let result = explorer.cleanup_old_tables(5).await.unwrap();
        
        // Should not remove any tables
        assert_eq!(result.tables_removed, 0);
        assert_eq!(result.tables_kept, 2);
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_drop_table_functionality() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create a test table
        let table_names = create_test_tables(&database, 1).await.unwrap();
        let table_name = &table_names[0];
        
        // Verify table exists
        assert!(explorer.schema_manager.table_exists(table_name).await.unwrap());
        
        // Drop the table
        explorer.drop_table(table_name).await.unwrap();
        
        // Verify table no longer exists
        assert!(!explorer.schema_manager.table_exists(table_name).await.unwrap());
    }
}

#[tokio::test]
async fn test_drop_nonexistent_table() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Try to drop a table that doesn't exist
        let result = explorer.drop_table("nonexistent_table").await;
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn test_management_recommendations() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create many test tables to trigger recommendations
        let table_names = create_test_tables(&database, 12).await.unwrap();
        
        let recommendations = explorer.get_management_recommendations().await.unwrap();
        
        // Should have at least one recommendation about too many tables
        assert!(!recommendations.is_empty());
        
        let cleanup_rec = recommendations.iter()
            .find(|r| r.title.contains("ingestion tables"));
        assert!(cleanup_rec.is_some());
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_table_optimization() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create test tables
        let table_names = create_test_tables(&database, 2).await.unwrap();
        
        // Optimize specific tables
        let result = explorer.optimize_tables(Some(table_names.clone())).await.unwrap();
        
        assert_eq!(result.tables_optimized, 2);
        assert_eq!(result.total_tables, 2);
        assert!(result.duration_ms > 0);
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_table_optimization_all_tables() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create test tables
        let table_names = create_test_tables(&database, 1).await.unwrap();
        
        // Optimize all tables (None parameter)
        let result = explorer.optimize_tables(None).await.unwrap();
        
        // Should optimize at least our test table plus system tables
        assert!(result.tables_optimized >= 1);
        assert!(result.total_tables >= 1);
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_table_list_with_metadata() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create test tables
        let table_names = create_test_tables(&database, 2).await.unwrap();
        
        // List ingestion tables
        let tables = explorer.list_tables(Some(TableType::Ingestion)).await.unwrap();
        
        // Should find our test tables
        let test_tables: Vec<_> = tables.iter()
            .filter(|t| t.name.starts_with("INGEST_TEST_"))
            .collect();
        assert_eq!(test_tables.len(), 2);
        
        // Check metadata
        for table in &test_tables {
            assert!(table.row_count.is_some());
            assert!(table.row_count.unwrap() > 0); // Should have data
            assert!(table.size_mb.is_some());
            assert!(table.created_at.is_some());
        }
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_table_sample_functionality() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create test table
        let table_names = create_test_tables(&database, 1).await.unwrap();
        let table_name = &table_names[0];
        
        // Sample the table
        let sample = explorer.sample_table(table_name, 5).await.unwrap();
        
        assert_eq!(sample.table_name, *table_name);
        assert_eq!(sample.sample_size, 5);
        assert_eq!(sample.total_rows, 10); // We inserted 10 rows
        assert!(!sample.columns.is_empty());
        assert_eq!(sample.rows.len(), 5);
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_table_describe_functionality() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create test table
        let table_names = create_test_tables(&database, 1).await.unwrap();
        let table_name = &table_names[0];
        
        // Describe the table
        let schema = explorer.describe_table(table_name).await.unwrap();
        
        assert_eq!(schema.table_name, *table_name);
        assert!(!schema.columns.is_empty());
        
        // Should have primary key column
        let pk_column = schema.columns.iter().find(|c| c.is_primary_key);
        assert!(pk_column.is_some());
        assert_eq!(pk_column.unwrap().name, "file_id");
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}

#[tokio::test]
async fn test_database_info_functionality() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        let info = explorer.get_database_info().await.unwrap();
        
        assert_eq!(info.connection_status, "Connected");
        assert!(!info.database_name.is_empty());
        assert!(!info.server_version.is_empty());
        assert!(info.total_tables > 0);
        assert!(info.total_size_mb >= 0.0);
        assert!(info.connection_time_ms > 0);
    }
}

#[tokio::test]
async fn test_table_management_performance_contract() {
    if let Ok(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Create test tables
        let table_names = create_test_tables(&database, 3).await.unwrap();
        
        // Test that operations complete within reasonable time
        let start = std::time::Instant::now();
        
        // List tables
        let _tables = explorer.list_tables(Some(TableType::Ingestion)).await.unwrap();
        let list_duration = start.elapsed();
        
        // Sample table
        let start = std::time::Instant::now();
        let _sample = explorer.sample_table(&table_names[0], 5).await.unwrap();
        let sample_duration = start.elapsed();
        
        // Describe table
        let start = std::time::Instant::now();
        let _schema = explorer.describe_table(&table_names[0]).await.unwrap();
        let describe_duration = start.elapsed();
        
        // All operations should complete quickly (within 5 seconds each)
        assert!(list_duration < Duration::from_secs(5));
        assert!(sample_duration < Duration::from_secs(5));
        assert!(describe_duration < Duration::from_secs(5));
        
        // Cleanup
        cleanup_test_tables(&database, &table_names).await;
    }
}