//! Integration tests for database exploration functionality
//!
//! These tests verify that the database exploration commands work correctly
//! with real PostgreSQL databases.

use code_ingest::database::{Database, DatabaseExplorer, SchemaManager, TableType};
use std::env;
use tokio;

/// Helper function to create a test database connection
async fn create_test_database() -> Option<Database> {
    if let Ok(database_url) = env::var("DATABASE_URL") {
        Database::new(&database_url).await.ok()
    } else {
        None
    }
}

/// Helper function to set up test data
async fn setup_test_data(database: &Database) -> anyhow::Result<String> {
    let schema_manager = SchemaManager::new(database.pool().clone());
    
    // Initialize schema
    schema_manager.initialize_schema().await?;
    
    // Create a test ingestion table
    let table_name = schema_manager.create_ingestion_table(None).await?;
    
    // Insert some test data
    let insert_sql = format!(
        r#"
        INSERT INTO "{}" (
            ingestion_id, filepath, filename, extension, file_size_bytes,
            line_count, word_count, token_count, content_text, file_type,
            relative_path, absolute_path
        ) VALUES 
        (1, '/test/file1.rs', 'file1.rs', 'rs', 1024, 50, 200, 180, 'fn main() {{}}', 'direct_text', 'file1.rs', '/test/file1.rs'),
        (1, '/test/file2.py', 'file2.py', 'py', 2048, 100, 400, 360, 'print("hello")', 'direct_text', 'file2.py', '/test/file2.py'),
        (1, '/test/doc.pdf', 'doc.pdf', 'pdf', 4096, NULL, NULL, NULL, NULL, 'non_text', 'doc.pdf', '/test/doc.pdf')
        "#,
        table_name
    );
    
    sqlx::query(&insert_sql)
        .execute(database.pool())
        .await?;
    
    Ok(table_name)
}

#[tokio::test]
async fn test_database_info_retrieval() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        let info = explorer.get_database_info().await.unwrap();
        
        // Verify basic info structure
        assert_eq!(info.connection_status, "Connected");
        assert!(!info.database_name.is_empty());
        assert!(!info.server_version.is_empty());
        assert!(info.connection_time_ms > 0);
        assert!(info.total_size_mb >= 0.0);
        
        // Should have at least some tables after initialization
        assert!(info.total_tables >= 0);
        
        println!("✅ Database info retrieval test passed");
        println!("   Database: {}", info.database_name);
        println!("   Version: {}", info.server_version);
        println!("   Tables: {}", info.total_tables);
    } else {
        println!("⏭️  Skipping database info test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_table_listing() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Set up test data
        let test_table = setup_test_data(&database).await.unwrap();
        
        // Test listing all tables
        let all_tables = explorer.list_tables(None).await.unwrap();
        assert!(!all_tables.is_empty());
        
        // Should include our test table and ingestion_meta
        let test_table_found = all_tables.iter().any(|t| t.name == test_table);
        let meta_table_found = all_tables.iter().any(|t| t.name == "ingestion_meta");
        
        assert!(test_table_found, "Test ingestion table should be in the list");
        assert!(meta_table_found, "ingestion_meta table should be in the list");
        
        // Test filtering by table type
        let ingestion_tables = explorer.list_tables(Some(TableType::Ingestion)).await.unwrap();
        let meta_tables = explorer.list_tables(Some(TableType::Meta)).await.unwrap();
        
        assert!(ingestion_tables.iter().any(|t| t.name == test_table));
        assert!(meta_tables.iter().any(|t| t.name == "ingestion_meta"));
        
        // Clean up
        let schema_manager = SchemaManager::new(database.pool().clone());
        schema_manager.drop_table(&test_table).await.unwrap();
        
        println!("✅ Table listing test passed");
        println!("   Total tables: {}", all_tables.len());
        println!("   Ingestion tables: {}", ingestion_tables.len());
        println!("   Meta tables: {}", meta_tables.len());
    } else {
        println!("⏭️  Skipping table listing test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_table_sampling() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Set up test data
        let test_table = setup_test_data(&database).await.unwrap();
        
        // Test sampling the table
        let sample = explorer.sample_table(&test_table, 5).await.unwrap();
        
        assert_eq!(sample.table_name, test_table);
        assert_eq!(sample.total_rows, 3); // We inserted 3 test rows
        assert_eq!(sample.sample_size, 3); // Should get all 3 rows
        assert!(!sample.columns.is_empty());
        assert_eq!(sample.rows.len(), 3);
        
        // Verify we have the expected columns
        let expected_columns = vec![
            "file_id", "ingestion_id", "filepath", "filename", "extension",
            "file_size_bytes", "line_count", "word_count", "token_count",
            "content_text", "file_type", "conversion_command", "relative_path",
            "absolute_path", "created_at"
        ];
        
        for expected_col in expected_columns {
            assert!(sample.columns.contains(&expected_col.to_string()),
                   "Missing expected column: {}", expected_col);
        }
        
        // Verify sample data content
        let rust_file = sample.rows.iter().find(|row| 
            row.get("filename").map(|f| f == "file1.rs").unwrap_or(false)
        );
        assert!(rust_file.is_some(), "Should find the Rust file in sample");
        
        let rust_file = rust_file.unwrap();
        assert_eq!(rust_file.get("extension").unwrap(), "rs");
        assert_eq!(rust_file.get("file_type").unwrap(), "direct_text");
        
        // Test sampling with limit smaller than total rows
        let limited_sample = explorer.sample_table(&test_table, 2).await.unwrap();
        assert_eq!(limited_sample.sample_size, 2);
        assert_eq!(limited_sample.total_rows, 3);
        
        // Clean up
        let schema_manager = SchemaManager::new(database.pool().clone());
        schema_manager.drop_table(&test_table).await.unwrap();
        
        println!("✅ Table sampling test passed");
        println!("   Sample size: {}/{}", sample.sample_size, sample.total_rows);
        println!("   Columns: {}", sample.columns.len());
        println!("   Query time: {}ms", sample.execution_time_ms);
    } else {
        println!("⏭️  Skipping table sampling test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_table_schema_description() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Set up test data
        let test_table = setup_test_data(&database).await.unwrap();
        
        // Test describing the table schema
        let schema = explorer.describe_table(&test_table).await.unwrap();
        
        assert_eq!(schema.table_name, test_table);
        assert_eq!(schema.table_type, "Ingestion");
        assert!(!schema.columns.is_empty());
        
        // Verify primary key column
        let pk_column = schema.columns.iter().find(|c| c.is_primary_key);
        assert!(pk_column.is_some(), "Should have a primary key column");
        assert_eq!(pk_column.unwrap().name, "file_id");
        
        // Verify some expected columns
        let filepath_col = schema.columns.iter().find(|c| c.name == "filepath");
        assert!(filepath_col.is_some());
        assert!(!filepath_col.unwrap().is_nullable);
        
        let content_col = schema.columns.iter().find(|c| c.name == "content_text");
        assert!(content_col.is_some());
        assert!(content_col.unwrap().is_nullable); // Content can be NULL for non-text files
        
        // Should have indexes
        assert!(!schema.indexes.is_empty(), "Should have indexes created");
        
        // Should have constraints (at least primary key)
        assert!(!schema.constraints.is_empty(), "Should have constraints");
        
        // Test describing the meta table
        let meta_schema = explorer.describe_table("ingestion_meta").await.unwrap();
        assert_eq!(meta_schema.table_name, "ingestion_meta");
        assert_eq!(meta_schema.table_type, "Meta");
        
        // Clean up
        let schema_manager = SchemaManager::new(database.pool().clone());
        schema_manager.drop_table(&test_table).await.unwrap();
        
        println!("✅ Table schema description test passed");
        println!("   Columns: {}", schema.columns.len());
        println!("   Indexes: {}", schema.indexes.len());
        println!("   Constraints: {}", schema.constraints.len());
    } else {
        println!("⏭️  Skipping table schema test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_nonexistent_table_handling() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Test operations on nonexistent table
        let sample_result = explorer.sample_table("nonexistent_table", 5).await;
        assert!(sample_result.is_err(), "Should fail for nonexistent table");
        
        let describe_result = explorer.describe_table("nonexistent_table").await;
        assert!(describe_result.is_err(), "Should fail for nonexistent table");
        
        println!("✅ Nonexistent table handling test passed");
    } else {
        println!("⏭️  Skipping nonexistent table test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_formatting_functions() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Test database info formatting
        let info = explorer.get_database_info().await.unwrap();
        let formatted_info = explorer.format_database_info(&info);
        
        assert!(formatted_info.contains("Database Information"));
        assert!(formatted_info.contains(&info.database_name));
        assert!(formatted_info.contains("Connected"));
        
        // Test table list formatting
        let tables = explorer.list_tables(None).await.unwrap();
        if !tables.is_empty() {
            let formatted_list = explorer.format_table_list(&tables, true);
            assert!(formatted_list.contains("Database Tables"));
            assert!(formatted_list.contains(&format!("Total: {} tables", tables.len())));
        }
        
        println!("✅ Formatting functions test passed");
        println!("   Database info formatted: {} chars", formatted_info.len());
    } else {
        println!("⏭️  Skipping formatting test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_performance_characteristics() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Test that operations complete within reasonable time
        let start = std::time::Instant::now();
        let _info = explorer.get_database_info().await.unwrap();
        let info_time = start.elapsed();
        
        let start = std::time::Instant::now();
        let _tables = explorer.list_tables(None).await.unwrap();
        let list_time = start.elapsed();
        
        // Operations should complete quickly
        assert!(info_time.as_millis() < 5000, "Database info should complete within 5 seconds");
        assert!(list_time.as_millis() < 5000, "Table listing should complete within 5 seconds");
        
        println!("✅ Performance characteristics test passed");
        println!("   Database info: {}ms", info_time.as_millis());
        println!("   Table listing: {}ms", list_time.as_millis());
    } else {
        println!("⏭️  Skipping performance test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_empty_database_handling() {
    if let Some(database) = create_test_database().await {
        let explorer = DatabaseExplorer::new(database.pool().clone());
        
        // Initialize schema but don't add any ingestion data
        let schema_manager = SchemaManager::new(database.pool().clone());
        schema_manager.initialize_schema().await.unwrap();
        
        // Should still work with empty database
        let info = explorer.get_database_info().await.unwrap();
        assert_eq!(info.connection_status, "Connected");
        
        let tables = explorer.list_tables(None).await.unwrap();
        // Should have at least the ingestion_meta table
        assert!(tables.iter().any(|t| t.name == "ingestion_meta"));
        
        // Sample empty table should work
        let sample = explorer.sample_table("ingestion_meta", 5).await.unwrap();
        assert_eq!(sample.total_rows, 0);
        assert_eq!(sample.sample_size, 0);
        assert!(!sample.columns.is_empty()); // Should still have column definitions
        
        println!("✅ Empty database handling test passed");
    } else {
        println!("⏭️  Skipping empty database test (no DATABASE_URL)");
    }
}