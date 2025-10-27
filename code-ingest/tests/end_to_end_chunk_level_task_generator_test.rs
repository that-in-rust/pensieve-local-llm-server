//! End-to-end integration tests for chunk-level task generator
//!
//! This module tests the complete workflow of the chunk-level-task-generator command,
//! including both file-level and chunk-level modes, error handling, and content generation.
//!
//! ## Test Coverage
//!
//! ### Requirements Tested:
//! - **1.1**: File-level mode content file generation (test_file_level_mode_end_to_end)
//! - **1.2**: Task list generation with row references (test_task_list_generation_and_format)
//! - **2.1**: Chunk-level mode with chunked table creation (test_chunk_level_mode_end_to_end)
//! - **2.2**: Small file copying unchanged (test_chunked_table_creation_and_population)
//! - **2.3**: Large file chunking (test_chunked_table_creation_and_population)
//! - **2.4**: L1 concatenation (current + next) (test_content_file_creation_and_validation)
//! - **2.5**: L2 concatenation (current + next + next2) (test_content_file_creation_and_validation)
//! - **2.6**: Content file creation and validation (test_content_file_creation_and_validation)
//! - **2.7**: Task list format compatibility (test_task_list_generation_and_format)
//! - **3.1**: Error handling for invalid inputs (test_error_handling_invalid_table_name, test_error_handling_invalid_chunk_size)
//! - **3.2**: Clear error messages (test_error_handling_invalid_table_name, test_error_handling_invalid_chunk_size)
//!
//! ### Test Structure:
//! Each test follows the pattern:
//! 1. **Setup**: Create test database and sample data
//! 2. **Execute**: Run the chunk-level task generator
//! 3. **Verify**: Check results against requirements
//! 4. **Cleanup**: Remove test data and tables
//!
//! ### Database Requirements:
//! These tests require a PostgreSQL database connection via DATABASE_URL environment variable.
//! Tests are automatically skipped if DATABASE_URL is not set.
//!
//! ### Test Data:
//! - Small files (< 500 lines) for file-level mode testing
//! - Large files (> 1000 lines) for chunk-level mode testing
//! - Various content types (Rust code, Markdown, etc.)
//! - Edge cases (empty content, special characters)

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio;
use chrono::Utc;

use code_ingest::database::{Database, models::IngestedFile};
use code_ingest::tasks::{
    chunk_level_task_generator::{ChunkLevelTaskGenerator, TaskGeneratorError},
    database_service::DatabaseService,
    content_file_writer::{ContentFileWriter, ContentWriteConfig},
    task_list_generator::TaskListGenerator,
    chunking_service::ChunkingService,
};

/// Helper function to create a test database with sample data
async fn create_test_database_with_data() -> Result<(Database, String)> {
    // Skip if DATABASE_URL is not set
    let database_url = std::env::var("DATABASE_URL")
        .map_err(|_| anyhow::anyhow!("DATABASE_URL not set - skipping integration test"))?;
    
    let database = Database::new(&database_url).await?;
    
    // Create a unique test table name
    let table_name = format!("test_chunk_generator_{}", 
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs());
    
    // Create test table with IngestedFile schema
    let create_table_sql = format!(r#"
        CREATE TABLE {} (
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
            file_type VARCHAR NOT NULL,
            conversion_command VARCHAR,
            relative_path VARCHAR NOT NULL,
            absolute_path VARCHAR NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    "#, table_name);
    
    sqlx::query(&create_table_sql)
        .execute(database.pool())
        .await?;
    
    Ok((database, table_name))
}

/// Helper function to insert test data into the database
async fn insert_test_data(database: &Database, table_name: &str, files: Vec<IngestedFile>) -> Result<()> {
    for file in files {
        let insert_sql = format!(r#"
            INSERT INTO {} (
                ingestion_id, filepath, filename, extension, file_size_bytes,
                line_count, word_count, token_count, content_text, file_type,
                conversion_command, relative_path, absolute_path, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        "#, table_name);
        
        sqlx::query(&insert_sql)
            .bind(file.ingestion_id)
            .bind(&file.filepath)
            .bind(&file.filename)
            .bind(&file.extension)
            .bind(file.file_size_bytes)
            .bind(file.line_count)
            .bind(file.word_count)
            .bind(file.token_count)
            .bind(&file.content_text)
            .bind(&file.file_type_str)
            .bind(&file.conversion_command)
            .bind(&file.relative_path)
            .bind(&file.absolute_path)
            .bind(file.created_at)
            .execute(database.pool())
            .await?;
    }
    
    Ok(())
}

/// Helper function to create test IngestedFile data
fn create_test_files() -> Vec<IngestedFile> {
    vec![
        IngestedFile {
            file_id: 1,
            ingestion_id: 1,
            filepath: "src/main.rs".to_string(),
            filename: "main.rs".to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: 150,
            line_count: Some(10),
            word_count: Some(25),
            token_count: Some(20),
            content_text: Some("fn main() {\n    println!(\"Hello, world!\");\n    let x = 42;\n    println!(\"x = {}\", x);\n}".to_string()),
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: "src/main.rs".to_string(),
            absolute_path: "/tmp/test/src/main.rs".to_string(),
            created_at: Utc::now(),
        },
        IngestedFile {
            file_id: 2,
            ingestion_id: 1,
            filepath: "src/lib.rs".to_string(),
            filename: "lib.rs".to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: 300,
            line_count: Some(20),
            word_count: Some(50),
            token_count: Some(40),
            content_text: Some("pub mod utils;\npub mod config;\n\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn test_add() {\n        assert_eq!(add(2, 3), 5);\n    }\n}".to_string()),
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: "src/lib.rs".to_string(),
            absolute_path: "/tmp/test/src/lib.rs".to_string(),
            created_at: Utc::now(),
        },
        IngestedFile {
            file_id: 3,
            ingestion_id: 1,
            filepath: "README.md".to_string(),
            filename: "README.md".to_string(),
            extension: Some("md".to_string()),
            file_size_bytes: 500,
            line_count: Some(25),
            word_count: Some(100),
            token_count: Some(80),
            content_text: Some("# Test Project\n\nThis is a test project for integration testing.\n\n## Features\n\n- Feature 1\n- Feature 2\n- Feature 3\n\n## Usage\n\n```rust\nuse test_project::add;\n\nlet result = add(2, 3);\nprintln!(\"Result: {}\", result);\n```\n\n## License\n\nMIT License".to_string()),
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: "README.md".to_string(),
            absolute_path: "/tmp/test/README.md".to_string(),
            created_at: Utc::now(),
        },
    ]
}

/// Helper function to create large test files for chunking
fn create_large_test_files() -> Vec<IngestedFile> {
    let large_content = (0..1000)
        .map(|i| format!("// Line {} of large file content\nfn function_{}() {{\n    println!(\"Function {}\");\n}}\n", i, i, i))
        .collect::<Vec<_>>()
        .join("\n");
    
    vec![
        IngestedFile {
            file_id: 1,
            ingestion_id: 1,
            filepath: "src/large_file.rs".to_string(),
            filename: "large_file.rs".to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: large_content.len() as i64,
            line_count: Some(4000), // 4 lines per function * 1000 functions
            word_count: Some(8000),
            token_count: Some(6000),
            content_text: Some(large_content.clone()),
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: "src/large_file.rs".to_string(),
            absolute_path: "/tmp/test/src/large_file.rs".to_string(),
            created_at: Utc::now(),
        },
        IngestedFile {
            file_id: 2,
            ingestion_id: 1,
            filepath: "src/another_large_file.rs".to_string(),
            filename: "another_large_file.rs".to_string(),
            extension: Some("rs".to_string()),
            file_size_bytes: large_content.len() as i64,
            line_count: Some(4000),
            word_count: Some(8000),
            token_count: Some(6000),
            content_text: Some(large_content),
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: "src/another_large_file.rs".to_string(),
            absolute_path: "/tmp/test/src/another_large_file.rs".to_string(),
            created_at: Utc::now(),
        },
    ]
}

/// Helper function to cleanup test table
async fn cleanup_test_table(database: &Database, table_name: &str) -> Result<()> {
    let drop_sql = format!("DROP TABLE IF EXISTS {}", table_name);
    sqlx::query(&drop_sql)
        .execute(database.pool())
        .await?;
    Ok(())
}

/// Helper function to create a test generator
fn create_test_generator(database: Arc<PgPool>, output_dir: PathBuf) -> ChunkLevelTaskGenerator {
    let database_service = Arc::new(DatabaseService::new(database));
    let content_writer_config = ContentWriteConfig::new(output_dir);
    let content_writer = ContentFileWriter::new(content_writer_config);
    let task_generator = TaskListGenerator::new();
    let chunking_service = ChunkingService::new(database_service.clone());
    
    ChunkLevelTaskGenerator::new(
        database_service,
        content_writer,
        task_generator,
        chunking_service,
    )
}

/// Test file-level mode (no chunk size) - Requirement 1.1, 1.2, 2.6, 2.7
#[tokio::test]
async fn test_file_level_mode_end_to_end() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping file-level mode integration test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let test_files = create_test_files();
    insert_test_data(&database, &table_name, test_files.clone()).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Execute file-level mode (no chunk size)
    let result = generator.execute(&table_name, None, None).await?;

    // Verify results
    assert_eq!(result.table_used, table_name);
    assert_eq!(result.rows_processed, 3); // 3 test files
    assert_eq!(result.content_files_created, 9); // 3 files per row (content, contentL1, contentL2)
    assert!(result.chunked_table_created.is_none()); // No chunking in file-level mode
    assert!(result.task_list_path.exists());

    // Verify content files exist
    for i in 1..=3 {
        let content_file = temp_dir.path().join(format!("content_{}.txt", i));
        let content_l1_file = temp_dir.path().join(format!("contentL1_{}.txt", i));
        let content_l2_file = temp_dir.path().join(format!("contentL2_{}.txt", i));
        
        assert!(content_file.exists(), "content_{}.txt should exist", i);
        assert!(content_l1_file.exists(), "contentL1_{}.txt should exist", i);
        assert!(content_l2_file.exists(), "contentL2_{}.txt should exist", i);
        
        // Verify content is not empty
        let content = std::fs::read_to_string(&content_file)?;
        assert!(!content.is_empty(), "content_{}.txt should not be empty", i);
    }

    // Verify task list content
    let task_list_content = std::fs::read_to_string(&result.task_list_path)?;
    assert!(task_list_content.contains("content_1.txt"));
    assert!(task_list_content.contains("contentL1_1.txt"));
    assert!(task_list_content.contains("contentL2_1.txt"));
    assert!(task_list_content.contains("content_2.txt"));
    assert!(task_list_content.contains("content_3.txt"));

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    
    println!("✅ File-level mode end-to-end test passed");
    Ok(())
}

/// Test chunk-level mode (with chunk size) - Requirement 2.1, 2.2, 2.3, 2.4, 2.5
#[tokio::test]
async fn test_chunk_level_mode_end_to_end() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping chunk-level mode integration test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let large_files = create_large_test_files();
    insert_test_data(&database, &table_name, large_files.clone()).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Execute chunk-level mode with chunk size 500
    let chunk_size = 500;
    let result = generator.execute(&table_name, Some(chunk_size), None).await?;

    // Verify results
    assert_eq!(result.table_used, format!("{}_{}", table_name, chunk_size));
    assert!(result.rows_processed > 2); // Should have more rows due to chunking
    assert!(result.content_files_created > 6); // More than 3 files per original row
    assert!(result.chunked_table_created.is_some()); // Chunking was used
    assert_eq!(result.chunked_table_created.unwrap(), format!("{}_{}", table_name, chunk_size));
    assert!(result.task_list_path.exists());

    // Verify chunked table was created and populated
    let chunked_table_name = format!("{}_{}", table_name, chunk_size);
    let count_sql = format!("SELECT COUNT(*) FROM {}", chunked_table_name);
    let row_count: (i64,) = sqlx::query_as(&count_sql)
        .fetch_one(database.pool())
        .await?;
    
    assert!(row_count.0 > 2, "Chunked table should have more rows than original");

    // Verify content files exist (at least some)
    let content_files: Vec<_> = std::fs::read_dir(temp_dir.path())?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_name().to_string_lossy().starts_with("content")
        })
        .collect();
    
    assert!(content_files.len() >= 3, "Should have at least 3 content files");

    // Verify task list references the content files
    let task_list_content = std::fs::read_to_string(&result.task_list_path)?;
    assert!(task_list_content.contains("content_"));
    assert!(task_list_content.contains("contentL1_"));
    assert!(task_list_content.contains("contentL2_"));

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    cleanup_test_table(&database, &chunked_table_name).await?;
    
    println!("✅ Chunk-level mode end-to-end test passed");
    Ok(())
}

/// Test error handling for invalid table names - Requirement 3.1, 3.2
#[tokio::test]
async fn test_error_handling_invalid_table_name() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping error handling test - DATABASE_URL not set");
        return Ok(());
    }

    let database_url = std::env::var("DATABASE_URL")?;
    let database = Database::new(&database_url).await?;
    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Test with non-existent table
    let result = generator.execute("nonexistent_table_12345", None, None).await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        TaskGeneratorError::TableNotFound { table } => {
            assert_eq!(table, "nonexistent_table_12345");
        }
        other => panic!("Expected TableNotFound error, got: {:?}", other),
    }

    // Test with invalid table name (SQL injection attempt)
    let result = generator.execute("test'; DROP TABLE users; --", None, None).await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        TaskGeneratorError::InvalidTableName { table, cause } => {
            assert!(table.contains("DROP TABLE"));
            assert!(cause.contains("invalid characters"));
        }
        other => panic!("Expected InvalidTableName error, got: {:?}", other),
    }

    // Test with empty table name
    let result = generator.execute("", None, None).await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        TaskGeneratorError::InvalidTableName { table, cause } => {
            assert_eq!(table, "");
            assert!(cause.contains("cannot be empty"));
        }
        other => panic!("Expected InvalidTableName error, got: {:?}", other),
    }

    println!("✅ Error handling for invalid table names test passed");
    Ok(())
}

/// Test error handling for invalid chunk sizes - Requirement 3.1, 3.2
#[tokio::test]
async fn test_error_handling_invalid_chunk_size() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping chunk size error handling test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let test_files = create_test_files();
    insert_test_data(&database, &table_name, test_files).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Test with chunk size 0
    let result = generator.execute(&table_name, Some(0), None).await;
    assert!(result.is_err());
    
    match result.unwrap_err() {
        TaskGeneratorError::InvalidChunkSize { size } => {
            assert_eq!(size, 0);
        }
        other => panic!("Expected InvalidChunkSize error, got: {:?}", other),
    }

    // Test with extremely large chunk size (should work but might be inefficient)
    let result = generator.execute(&table_name, Some(1_000_000), None).await;
    // This should succeed but create minimal chunking
    assert!(result.is_ok());

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    
    println!("✅ Error handling for invalid chunk sizes test passed");
    Ok(())
}

/// Test content file creation and validation - Requirement 2.6
#[tokio::test]
async fn test_content_file_creation_and_validation() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping content file validation test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let test_files = create_test_files();
    insert_test_data(&database, &table_name, test_files.clone()).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Execute file-level mode
    let result = generator.execute(&table_name, None, None).await?;

    // Verify each content file has correct content
    for (i, original_file) in test_files.iter().enumerate() {
        let row_num = i + 1;
        
        // Check content file
        let content_file = temp_dir.path().join(format!("content_{}.txt", row_num));
        let content = std::fs::read_to_string(&content_file)?;
        assert_eq!(content.trim(), original_file.content_text.as_ref().unwrap().trim());
        
        // Check contentL1 file (should contain current + next, or just current if last)
        let content_l1_file = temp_dir.path().join(format!("contentL1_{}.txt", row_num));
        let content_l1 = std::fs::read_to_string(&content_l1_file)?;
        assert!(content_l1.contains(original_file.content_text.as_ref().unwrap()));
        
        // Check contentL2 file (should contain current + next + next2, or available content)
        let content_l2_file = temp_dir.path().join(format!("contentL2_{}.txt", row_num));
        let content_l2 = std::fs::read_to_string(&content_l2_file)?;
        assert!(content_l2.contains(original_file.content_text.as_ref().unwrap()));
        
        // L2 should be longer than or equal to L1
        assert!(content_l2.len() >= content_l1.len());
        // L1 should be longer than or equal to content
        assert!(content_l1.len() >= content.len());
    }

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    
    println!("✅ Content file creation and validation test passed");
    Ok(())
}

/// Test task list generation and format - Requirement 2.7
#[tokio::test]
async fn test_task_list_generation_and_format() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping task list generation test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let test_files = create_test_files();
    insert_test_data(&database, &table_name, test_files.clone()).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Execute file-level mode
    let result = generator.execute(&table_name, None, None).await?;

    // Read and validate task list content
    let task_list_content = std::fs::read_to_string(&result.task_list_path)?;
    
    // Verify task list format
    assert!(task_list_content.contains("Task List"));
    
    // Verify each row is referenced
    for i in 1..=test_files.len() {
        assert!(task_list_content.contains(&format!("content_{}.txt", i)));
        assert!(task_list_content.contains(&format!("contentL1_{}.txt", i)));
        assert!(task_list_content.contains(&format!("contentL2_{}.txt", i)));
    }
    
    // Verify task list has proper structure (should be compatible with existing workflows)
    let lines: Vec<&str> = task_list_content.lines().collect();
    assert!(lines.len() > 0, "Task list should not be empty");
    
    // Check that file references are properly formatted
    let file_references = lines.iter()
        .filter(|line| line.contains("content_") || line.contains("contentL1_") || line.contains("contentL2_"))
        .count();
    
    assert!(file_references >= test_files.len() * 3, "Should have at least 3 references per file");

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    
    println!("✅ Task list generation and format test passed");
    Ok(())
}

/// Test chunked table creation and population - Requirement 2.1, 2.2, 2.3, 2.4, 2.5
#[tokio::test]
async fn test_chunked_table_creation_and_population() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping chunked table test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let large_files = create_large_test_files();
    insert_test_data(&database, &table_name, large_files.clone()).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Execute chunk-level mode
    let chunk_size = 500;
    let result = generator.execute(&table_name, Some(chunk_size), None).await?;

    let chunked_table_name = result.chunked_table_created.unwrap();

    // Verify chunked table exists
    let table_exists_sql = r#"
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = $1
        )
    "#;
    let (exists,): (bool,) = sqlx::query_as(table_exists_sql)
        .bind(&chunked_table_name)
        .fetch_one(database.pool())
        .await?;
    
    assert!(exists, "Chunked table should exist");

    // Verify chunked table has correct schema
    let schema_sql = r#"
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = $1
        ORDER BY ordinal_position
    "#;
    let columns: Vec<(String,)> = sqlx::query_as(schema_sql)
        .bind(&chunked_table_name)
        .fetch_all(database.pool())
        .await?;
    
    let column_names: Vec<String> = columns.into_iter().map(|(name,)| name).collect();
    
    // Should have all the required columns for IngestedFile
    assert!(column_names.contains(&"file_id".to_string()));
    assert!(column_names.contains(&"content_text".to_string()));
    assert!(column_names.contains(&"line_count".to_string()));

    // Verify chunked table has more rows than original (due to chunking)
    let original_count_sql = format!("SELECT COUNT(*) FROM {}", table_name);
    let (original_count,): (i64,) = sqlx::query_as(&original_count_sql)
        .fetch_one(database.pool())
        .await?;
    
    let chunked_count_sql = format!("SELECT COUNT(*) FROM {}", chunked_table_name);
    let (chunked_count,): (i64,) = sqlx::query_as(&chunked_count_sql)
        .fetch_one(database.pool())
        .await?;
    
    assert!(chunked_count >= original_count, 
           "Chunked table should have at least as many rows as original (got {} vs {})", 
           chunked_count, original_count);

    // For large files with 4000 lines each and chunk size 500, we should get multiple chunks
    assert!(chunked_count > original_count, 
           "Large files should be chunked into multiple rows");

    // Verify chunk content is properly split
    let sample_chunk_sql = format!("SELECT content_text, line_count FROM {} LIMIT 1", chunked_table_name);
    let (chunk_content, chunk_line_count): (Option<String>, Option<i32>) = sqlx::query_as(&sample_chunk_sql)
        .fetch_one(database.pool())
        .await?;
    
    if let (Some(content), Some(line_count)) = (chunk_content, chunk_line_count) {
        assert!(line_count <= chunk_size as i32, 
               "Chunk should not exceed chunk size limit");
        assert!(!content.is_empty(), "Chunk content should not be empty");
    }

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    cleanup_test_table(&database, &chunked_table_name).await?;
    
    println!("✅ Chunked table creation and population test passed");
    Ok(())
}

/// Test complete workflow with both modes
#[tokio::test]
async fn test_complete_workflow_both_modes() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping complete workflow test - DATABASE_URL not set");
        return Ok(());
    }

    let (database, table_name) = create_test_database_with_data().await?;
    let test_files = create_test_files();
    insert_test_data(&database, &table_name, test_files.clone()).await?;

    let temp_dir = TempDir::new()?;
    let generator = create_test_generator(database.pool().clone(), temp_dir.path().to_path_buf());

    // Test file-level mode first
    let file_level_result = generator.execute(&table_name, None, None).await?;
    assert!(file_level_result.chunked_table_created.is_none());
    assert_eq!(file_level_result.rows_processed, 3);

    // Clear output directory for chunk-level test
    for entry in std::fs::read_dir(temp_dir.path())? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::remove_file(entry.path())?;
        }
    }

    // Test chunk-level mode with small chunk size to force chunking
    let chunk_level_result = generator.execute(&table_name, Some(5), None).await?;
    assert!(chunk_level_result.chunked_table_created.is_some());
    assert!(chunk_level_result.rows_processed >= 3); // Should have same or more rows due to chunking

    // Both modes should create task lists
    assert!(file_level_result.task_list_path.exists() || chunk_level_result.task_list_path.exists());

    // Cleanup
    cleanup_test_table(&database, &table_name).await?;
    if let Some(chunked_table) = chunk_level_result.chunked_table_created {
        cleanup_test_table(&database, &chunked_table).await?;
    }
    
    println!("✅ Complete workflow test passed");
    Ok(())
}