//! Unit tests for database operations
//! 
//! Tests Requirements 3.1, 3.2, 3.3 - PostgreSQL schema and operations

use code_ingest::database::{
    Database, DatabaseConfig, IngestionMeta, IngestedFile,
    models::{ProcessedFile, FileType},
    operations::DatabaseOperations,
};
use sqlx::PgPool;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use uuid::Uuid;

/// Helper to create a test database
async fn create_test_database() -> anyhow::Result<Database> {
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/code_ingest_test".to_string());
    
    let config = DatabaseConfig {
        database_url,
        max_connections: 5,
        connection_timeout_seconds: 10,
    };
    
    Database::new(config).await
}

/// Helper to create test processed file
fn create_test_processed_file(filename: &str, content: &str) -> ProcessedFile {
    ProcessedFile {
        filepath: format!("/test/path/{}", filename),
        filename: filename.to_string(),
        extension: filename.split('.').last().unwrap_or("").to_string(),
        file_size_bytes: content.len() as i64,
        line_count: Some(content.lines().count() as i32),
        word_count: Some(content.split_whitespace().count() as i32),
        token_count: Some((content.split_whitespace().count() as f32 * 1.3) as i32),
        content_text: Some(content.to_string()),
        file_type: FileType::DirectText,
        conversion_command: None,
        relative_path: filename.to_string(),
        absolute_path: format!("/absolute/test/path/{}", filename),
        skipped: false,
        skip_reason: None,
    }
}

#[tokio::test]
async fn test_database_connection_and_setup() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Test that we can get a connection
    let pool = db.get_pool();
    assert!(pool.is_closed() == false);
    
    // Test that schema exists
    let tables = sqlx::query_scalar::<_, String>(
        "SELECT table_name FROM information_schema.tables 
         WHERE table_schema = 'public' AND table_name = 'ingestion_meta'"
    )
    .fetch_all(pool)
    .await?;
    
    assert!(tables.contains(&"ingestion_meta".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_ingestion_metadata_lifecycle() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Test starting an ingestion
    let repo_url = "https://github.com/test/repo";
    let local_path = "/tmp/test_repo";
    
    let ingestion_id = db.start_ingestion_record(repo_url, local_path).await?;
    assert!(ingestion_id > 0);
    
    // Verify the record was created
    let meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT * FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    assert_eq!(meta.repo_url, repo_url);
    assert_eq!(meta.local_path, local_path);
    assert!(meta.start_timestamp_unix > 0);
    assert!(meta.end_timestamp_unix.is_none());
    assert!(!meta.table_name.is_empty());
    assert!(meta.table_name.starts_with("INGEST_"));
    
    // Test completing the ingestion
    let files_processed = 42;
    db.complete_ingestion_record(ingestion_id, files_processed).await?;
    
    // Verify the record was updated
    let updated_meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT * FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    assert!(updated_meta.end_timestamp_unix.is_some());
    assert_eq!(updated_meta.total_files_processed, Some(files_processed));
    
    Ok(())
}

#[tokio::test]
async fn test_timestamped_table_creation() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Start an ingestion to create a timestamped table
    let ingestion_id = db.start_ingestion_record(
        "https://github.com/test/table_test", 
        "/tmp/table_test"
    ).await?;
    
    // Get the table name
    let meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT * FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    let table_name = &meta.table_name;
    
    // Verify table exists and has correct schema
    let columns = sqlx::query_scalar::<_, String>(
        "SELECT column_name FROM information_schema.columns 
         WHERE table_name = $1 ORDER BY ordinal_position"
    )
    .bind(table_name)
    .fetch_all(db.get_pool())
    .await?;
    
    let expected_columns = [
        "file_id", "ingestion_id", "filepath", "filename", "extension",
        "file_size_bytes", "line_count", "word_count", "token_count",
        "content_text", "file_type", "conversion_command", "relative_path",
        "absolute_path", "created_at"
    ];
    
    for expected_col in &expected_columns {
        assert!(
            columns.contains(&expected_col.to_string()),
            "Table {} should have column {}",
            table_name,
            expected_col
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn test_file_insertion_and_retrieval() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Start an ingestion
    let ingestion_id = db.start_ingestion_record(
        "https://github.com/test/files", 
        "/tmp/files_test"
    ).await?;
    
    // Get table name
    let meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    let table_name = &meta.table_name;
    
    // Create test files
    let test_files = vec![
        create_test_processed_file("main.rs", "fn main() { println!(\"Hello!\"); }"),
        create_test_processed_file("README.md", "# Test Project\n\nThis is a test."),
        create_test_processed_file("config.json", r#"{"key": "value"}"#),
    ];
    
    // Insert files
    for file in &test_files {
        db.insert_processed_file(table_name, ingestion_id, file).await?;
    }
    
    // Retrieve and verify files
    let query = format!(
        "SELECT * FROM {} WHERE ingestion_id = $1 ORDER BY filename",
        table_name
    );
    
    let stored_files = sqlx::query_as::<_, IngestedFile>(&query)
        .bind(ingestion_id)
        .fetch_all(db.get_pool())
        .await?;
    
    assert_eq!(stored_files.len(), 3);
    
    // Verify first file (config.json alphabetically first)
    let config_file = &stored_files[0];
    assert_eq!(config_file.filename, "config.json");
    assert_eq!(config_file.extension, Some("json".to_string()));
    assert_eq!(config_file.file_type, "direct_text");
    assert!(config_file.content_text.is_some());
    assert_eq!(config_file.content_text.as_ref().unwrap(), r#"{"key": "value"}"#);
    
    Ok(())
}

#[tokio::test]
async fn test_file_type_storage() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    let ingestion_id = db.start_ingestion_record(
        "https://github.com/test/types", 
        "/tmp/types_test"
    ).await?;
    
    let meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    // Test different file types
    let mut direct_text = create_test_processed_file("script.py", "print('hello')");
    direct_text.file_type = FileType::DirectText;
    
    let mut convertible = create_test_processed_file("document.pdf", "");
    convertible.file_type = FileType::Convertible;
    convertible.content_text = Some("Converted PDF content".to_string());
    convertible.conversion_command = Some("pdftotext document.pdf -".to_string());
    
    let mut binary = create_test_processed_file("image.jpg", "");
    binary.file_type = FileType::NonText;
    binary.content_text = None;
    binary.line_count = None;
    binary.word_count = None;
    binary.token_count = None;
    binary.skipped = true;
    binary.skip_reason = Some("Binary file".to_string());
    
    // Insert all types
    db.insert_processed_file(&meta.table_name, ingestion_id, &direct_text).await?;
    db.insert_processed_file(&meta.table_name, ingestion_id, &convertible).await?;
    db.insert_processed_file(&meta.table_name, ingestion_id, &binary).await?;
    
    // Verify storage
    let query = format!(
        "SELECT filename, file_type, content_text, conversion_command 
         FROM {} WHERE ingestion_id = $1 ORDER BY filename",
        meta.table_name
    );
    
    let results = sqlx::query(&query)
        .bind(ingestion_id)
        .fetch_all(db.get_pool())
        .await?;
    
    assert_eq!(results.len(), 3);
    
    // Check direct text file
    let direct_row = &results[0]; // document.pdf
    assert_eq!(direct_row.get::<String, _>("file_type"), "convertible");
    assert!(direct_row.get::<Option<String>, _>("content_text").is_some());
    assert!(direct_row.get::<Option<String>, _>("conversion_command").is_some());
    
    // Check binary file
    let binary_row = &results[1]; // image.jpg
    assert_eq!(binary_row.get::<String, _>("file_type"), "non_text");
    assert!(binary_row.get::<Option<String>, _>("content_text").is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_full_text_search_index() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    let ingestion_id = db.start_ingestion_record(
        "https://github.com/test/search", 
        "/tmp/search_test"
    ).await?;
    
    let meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    // Insert files with searchable content
    let files = vec![
        create_test_processed_file("auth.rs", "fn authenticate_user(token: &str) -> bool { true }"),
        create_test_processed_file("db.rs", "fn connect_database() -> Connection { todo!() }"),
        create_test_processed_file("main.rs", "fn main() { let user = authenticate_user(\"token\"); }"),
    ];
    
    for file in &files {
        db.insert_processed_file(&meta.table_name, ingestion_id, file).await?;
    }
    
    // Test full-text search
    let search_query = format!(
        "SELECT filename, content_text FROM {} 
         WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'authenticate')
         ORDER BY filename",
        meta.table_name
    );
    
    let search_results = sqlx::query(&search_query)
        .fetch_all(db.get_pool())
        .await?;
    
    assert_eq!(search_results.len(), 2); // auth.rs and main.rs
    
    let filenames: Vec<String> = search_results
        .iter()
        .map(|row| row.get::<String, _>("filename"))
        .collect();
    
    assert!(filenames.contains(&"auth.rs".to_string()));
    assert!(filenames.contains(&"main.rs".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_database_operations() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Test concurrent ingestion starts
    let mut tasks = Vec::new();
    
    for i in 0..5 {
        let db_clone = db.clone();
        let task = tokio::spawn(async move {
            let repo_url = format!("https://github.com/test/concurrent_{}", i);
            let local_path = format!("/tmp/concurrent_{}", i);
            
            let ingestion_id = db_clone.start_ingestion_record(&repo_url, &local_path).await?;
            
            // Insert some files
            let meta = sqlx::query_as::<_, IngestionMeta>(
                "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
            )
            .bind(ingestion_id)
            .fetch_one(db_clone.get_pool())
            .await?;
            
            let file = create_test_processed_file(
                &format!("file_{}.rs", i),
                &format!("fn function_{}() {{}}", i)
            );
            
            db_clone.insert_processed_file(&meta.table_name, ingestion_id, &file).await?;
            db_clone.complete_ingestion_record(ingestion_id, 1).await?;
            
            anyhow::Ok(ingestion_id)
        });
        
        tasks.push(task);
    }
    
    // Wait for all tasks to complete
    let mut ingestion_ids = Vec::new();
    for task in tasks {
        let ingestion_id = task.await??;
        ingestion_ids.push(ingestion_id);
    }
    
    // Verify all ingestions completed successfully
    assert_eq!(ingestion_ids.len(), 5);
    
    for ingestion_id in &ingestion_ids {
        let meta = sqlx::query_as::<_, IngestionMeta>(
            "SELECT * FROM ingestion_meta WHERE ingestion_id = $1"
        )
        .bind(ingestion_id)
        .fetch_one(db.get_pool())
        .await?;
        
        assert!(meta.end_timestamp_unix.is_some());
        assert_eq!(meta.total_files_processed, Some(1));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_database_error_handling() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Test inserting into non-existent table
    let file = create_test_processed_file("test.rs", "fn test() {}");
    let result = db.insert_processed_file("nonexistent_table", 1, &file).await;
    assert!(result.is_err());
    
    // Test invalid ingestion ID
    let result = db.complete_ingestion_record(99999, 10).await;
    assert!(result.is_err());
    
    // Test duplicate ingestion (if we had unique constraints)
    let ingestion_id1 = db.start_ingestion_record(
        "https://github.com/test/duplicate", 
        "/tmp/duplicate"
    ).await?;
    
    // This should succeed (no unique constraint on repo_url in basic implementation)
    let ingestion_id2 = db.start_ingestion_record(
        "https://github.com/test/duplicate", 
        "/tmp/duplicate"
    ).await?;
    
    assert_ne!(ingestion_id1, ingestion_id2);
    
    Ok(())
}

#[tokio::test]
async fn test_database_performance_contracts() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Test ingestion start performance
    let start = std::time::Instant::now();
    let ingestion_id = db.start_ingestion_record(
        "https://github.com/test/performance", 
        "/tmp/performance"
    ).await?;
    let ingestion_start_time = start.elapsed();
    
    // Should be very fast (< 100ms)
    assert!(
        ingestion_start_time.as_millis() < 100,
        "Ingestion start took too long: {:?}",
        ingestion_start_time
    );
    
    // Test batch file insertion performance
    let meta = sqlx::query_as::<_, IngestionMeta>(
        "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
    )
    .bind(ingestion_id)
    .fetch_one(db.get_pool())
    .await?;
    
    let start = std::time::Instant::now();
    
    // Insert 100 files
    for i in 0..100 {
        let file = create_test_processed_file(
            &format!("file_{}.rs", i),
            &format!("fn function_{}() {{ println!(\"File {}\"); }}", i, i)
        );
        db.insert_processed_file(&meta.table_name, ingestion_id, &file).await?;
    }
    
    let batch_insert_time = start.elapsed();
    
    // Should complete within reasonable time (< 5 seconds for 100 files)
    assert!(
        batch_insert_time.as_secs() < 5,
        "Batch insert took too long: {:?}",
        batch_insert_time
    );
    
    // Test query performance
    let start = std::time::Instant::now();
    
    let query = format!(
        "SELECT COUNT(*) FROM {} WHERE ingestion_id = $1",
        meta.table_name
    );
    
    let count: i64 = sqlx::query_scalar(&query)
        .bind(ingestion_id)
        .fetch_one(db.get_pool())
        .await?;
    
    let query_time = start.elapsed();
    
    assert_eq!(count, 100);
    
    // Query should be very fast (< 50ms)
    assert!(
        query_time.as_millis() < 50,
        "Query took too long: {:?}",
        query_time
    );
    
    Ok(())
}

#[tokio::test]
async fn test_database_cleanup() -> anyhow::Result<()> {
    let db = create_test_database().await?;
    
    // Create multiple ingestions
    let mut table_names = Vec::new();
    
    for i in 0..3 {
        let ingestion_id = db.start_ingestion_record(
            &format!("https://github.com/test/cleanup_{}", i),
            &format!("/tmp/cleanup_{}", i)
        ).await?;
        
        let meta = sqlx::query_as::<_, IngestionMeta>(
            "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
        )
        .bind(ingestion_id)
        .fetch_one(db.get_pool())
        .await?;
        
        table_names.push(meta.table_name);
        
        db.complete_ingestion_record(ingestion_id, 0).await?;
    }
    
    // Verify tables exist
    for table_name in &table_names {
        let exists = sqlx::query_scalar::<_, bool>(
            "SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )"
        )
        .bind(table_name)
        .fetch_one(db.get_pool())
        .await?;
        
        assert!(exists, "Table {} should exist", table_name);
    }
    
    // Test cleanup functionality (if implemented)
    // For now, just verify we can list all ingestion tables
    let all_tables = sqlx::query_scalar::<_, String>(
        "SELECT table_name FROM information_schema.tables 
         WHERE table_name LIKE 'INGEST_%' 
         ORDER BY table_name"
    )
    .fetch_all(db.get_pool())
    .await?;
    
    assert!(all_tables.len() >= 3);
    
    Ok(())
}

// Property-based tests for database operations
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_ingestion_metadata_roundtrip(
            repo_url in "[a-zA-Z0-9:/._-]{10,100}",
            local_path in "[a-zA-Z0-9/._-]{5,50}"
        ) {
            tokio_test::block_on(async {
                let db = create_test_database().await.unwrap();
                
                let ingestion_id = db.start_ingestion_record(&repo_url, &local_path).await.unwrap();
                
                let meta = sqlx::query_as::<_, IngestionMeta>(
                    "SELECT * FROM ingestion_meta WHERE ingestion_id = $1"
                )
                .bind(ingestion_id)
                .fetch_one(db.get_pool())
                .await.unwrap();
                
                assert_eq!(meta.repo_url, repo_url);
                assert_eq!(meta.local_path, local_path);
                assert!(meta.start_timestamp_unix > 0);
            });
        }
        
        #[test]
        fn test_file_content_storage_integrity(
            filename in "[a-zA-Z0-9._-]{1,50}",
            content in ".*{0,1000}"
        ) {
            tokio_test::block_on(async {
                let db = create_test_database().await.unwrap();
                
                let ingestion_id = db.start_ingestion_record(
                    "https://github.com/test/property",
                    "/tmp/property"
                ).await.unwrap();
                
                let meta = sqlx::query_as::<_, IngestionMeta>(
                    "SELECT table_name FROM ingestion_meta WHERE ingestion_id = $1"
                )
                .bind(ingestion_id)
                .fetch_one(db.get_pool())
                .await.unwrap();
                
                let file = create_test_processed_file(&filename, &content);
                db.insert_processed_file(&meta.table_name, ingestion_id, &file).await.unwrap();
                
                let query = format!(
                    "SELECT content_text FROM {} WHERE filename = $1",
                    meta.table_name
                );
                
                let stored_content: Option<String> = sqlx::query_scalar(&query)
                    .bind(&filename)
                    .fetch_one(db.get_pool())
                    .await.unwrap();
                
                assert_eq!(stored_content, Some(content));
            });
        }
    }
}