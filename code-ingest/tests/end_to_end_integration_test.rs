//! End-to-end integration tests
//! 
//! Tests complete workflows from Requirements 1.1, 1.2, 1.4, 1.5

use code_ingest::{
    core::IngestionEngine,
    database::{Database, DatabaseConfig},
    cli::commands::{IngestCommand, SqlCommand, ListTablesCommand},
    ingestion::SourceType,
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use tempfile::TempDir;
use serial_test::serial;

/// Create a comprehensive test repository
fn create_comprehensive_test_repo(temp_dir: &TempDir) -> PathBuf {
    let repo_path = temp_dir.path().join("comprehensive_repo");
    std::fs::create_dir_all(&repo_path).unwrap();
    
    // Create Rust source files
    let src_dir = repo_path.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    std::fs::write(
        src_dir.join("main.rs"),
        r#"//! Main application entry point
use std::collections::HashMap;

fn main() {
    println!("Hello, world!");
    let mut map = HashMap::new();
    map.insert("key", "value");
    
    authenticate_user("admin");
    process_data(&map);
}

fn authenticate_user(username: &str) -> bool {
    // Authentication logic here
    username == "admin"
}

fn process_data(data: &HashMap<&str, &str>) {
    for (key, value) in data {
        println!("{}: {}", key, value);
    }
}
"#,
    ).unwrap();
    
    std::fs::write(
        src_dir.join("lib.rs"),
        r#"//! Library module
pub mod auth;
pub mod database;
pub mod utils;

pub use auth::*;
pub use database::*;
pub use utils::*;
"#,
    ).unwrap();
    
    std::fs::write(
        src_dir.join("auth.rs"),
        r#"//! Authentication module
use std::collections::HashMap;

pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
}

pub fn authenticate(username: &str, password: &str) -> Option<User> {
    // Mock authentication
    if username == "admin" && password == "secret" {
        Some(User {
            id: 1,
            username: username.to_string(),
            email: "admin@example.com".to_string(),
        })
    } else {
        None
    }
}

pub fn authorize_action(user: &User, action: &str) -> bool {
    // Mock authorization
    user.username == "admin"
}
"#,
    ).unwrap();
    
    std::fs::write(
        src_dir.join("database.rs"),
        r#"//! Database operations
use std::collections::HashMap;

pub struct Database {
    connection_string: String,
}

impl Database {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
    
    pub fn connect(&self) -> Result<Connection, DatabaseError> {
        // Mock connection
        Ok(Connection::new())
    }
}

pub struct Connection {
    // Mock connection
}

impl Connection {
    fn new() -> Self {
        Self {}
    }
    
    pub fn execute_query(&self, query: &str) -> Result<Vec<Row>, DatabaseError> {
        // Mock query execution
        Ok(vec![])
    }
}

pub struct Row {
    data: HashMap<String, String>,
}

#[derive(Debug)]
pub enum DatabaseError {
    ConnectionFailed,
    QueryFailed(String),
}
"#,
    ).unwrap();
    
    // Create configuration files
    std::fs::write(
        repo_path.join("Cargo.toml"),
        r#"[package]
name = "comprehensive-test"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-rustls"] }
anyhow = "1.0"
thiserror = "1.0"

[dev-dependencies]
tempfile = "3.0"
"#,
    ).unwrap();
    
    std::fs::write(
        repo_path.join("README.md"),
        r#"# Comprehensive Test Repository

This is a test repository for comprehensive integration testing.

## Features

- Authentication system
- Database operations
- User management
- Configuration management

## Usage

```bash
cargo run
```

## Testing

```bash
cargo test
```

## Authentication

The system supports user authentication with the following features:

- Username/password authentication
- Role-based authorization
- Session management

## Database

The database module provides:

- Connection management
- Query execution
- Error handling
- Transaction support
"#,
    ).unwrap();
    
    std::fs::write(
        repo_path.join("config.json"),
        r#"{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "testdb",
    "username": "testuser",
    "password": "testpass"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "logging": {
    "level": "info",
    "file": "app.log"
  },
  "features": {
    "authentication": true,
    "authorization": true,
    "database_logging": false
  }
}
"#,
    ).unwrap();
    
    // Create test files
    let tests_dir = repo_path.join("tests");
    std::fs::create_dir_all(&tests_dir).unwrap();
    
    std::fs::write(
        tests_dir.join("integration_test.rs"),
        r#"//! Integration tests
use comprehensive_test::*;

#[tokio::test]
async fn test_authentication_flow() {
    let user = authenticate("admin", "secret");
    assert!(user.is_some());
    
    let user = user.unwrap();
    assert_eq!(user.username, "admin");
    assert!(authorize_action(&user, "read"));
}

#[tokio::test]
async fn test_database_operations() {
    let db = Database::new("postgresql://localhost/test".to_string());
    let conn = db.connect().unwrap();
    
    let results = conn.execute_query("SELECT * FROM users").unwrap();
    assert!(results.is_empty()); // Mock returns empty
}
"#,
    ).unwrap();
    
    // Create documentation
    let docs_dir = repo_path.join("docs");
    std::fs::create_dir_all(&docs_dir).unwrap();
    
    std::fs::write(
        docs_dir.join("architecture.md"),
        r#"# Architecture Documentation

## Overview

This application follows a layered architecture pattern:

1. **Presentation Layer**: CLI interface
2. **Business Logic Layer**: Core application logic
3. **Data Access Layer**: Database operations
4. **Infrastructure Layer**: External services

## Authentication Flow

```
User Input -> Authentication Module -> Database -> Authorization -> Access Granted
```

## Database Schema

### Users Table

- id: Primary key
- username: Unique username
- email: User email
- created_at: Timestamp

### Sessions Table

- id: Primary key
- user_id: Foreign key to users
- token: Session token
- expires_at: Expiration timestamp
"#,
    ).unwrap();
    
    // Create some binary files
    std::fs::write(
        repo_path.join("image.jpg"),
        b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00"
    ).unwrap();
    
    // Create gitignore
    std::fs::write(
        repo_path.join(".gitignore"),
        r#"target/
*.log
*.tmp
.env
.DS_Store
"#,
    ).unwrap();
    
    repo_path
}

/// Helper to create test database
async fn create_test_database() -> anyhow::Result<Database> {
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/code_ingest_test".to_string());
    
    let config = DatabaseConfig {
        database_url,
        max_connections: 10,
        connection_timeout_seconds: 30,
    };
    
    Database::new(config).await
}

#[tokio::test]
#[serial]
async fn test_complete_local_folder_ingestion_workflow() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    // Create database
    let db = create_test_database().await?;
    
    // Create ingestion engine
    let engine = IngestionEngine::new(db).await?;
    
    // Test complete ingestion workflow
    let start_time = Instant::now();
    
    let result = engine.ingest_source(
        SourceType::LocalFolder(repo_path.clone()),
        None, // No GitHub token needed for local folder
    ).await?;
    
    let ingestion_duration = start_time.elapsed();
    
    // Verify ingestion completed successfully
    assert!(result.files_processed > 0, "Should process files");
    assert!(!result.table_name.is_empty(), "Should create table");
    assert!(result.table_name.starts_with("INGEST_"), "Table should have correct prefix");
    
    // Performance contract: Should complete within reasonable time
    assert!(
        ingestion_duration < Duration::from_secs(30),
        "Ingestion took too long: {:?}",
        ingestion_duration
    );
    
    println!(
        "Ingested {} files into {} in {:?}",
        result.files_processed,
        result.table_name,
        ingestion_duration
    );
    
    // Test querying the ingested data
    let query = format!(
        "SELECT COUNT(*) FROM {} WHERE file_type = 'direct_text'",
        result.table_name
    );
    
    let text_file_count: i64 = sqlx::query_scalar(&query)
        .fetch_one(engine.get_database().get_pool())
        .await?;
    
    assert!(text_file_count > 0, "Should have text files");
    
    // Test searching for specific content
    let auth_query = format!(
        "SELECT filepath, filename FROM {} WHERE content_text LIKE '%authenticate%'",
        result.table_name
    );
    
    let auth_files = sqlx::query(&auth_query)
        .fetch_all(engine.get_database().get_pool())
        .await?;
    
    assert!(auth_files.len() >= 2, "Should find authentication-related files");
    
    // Verify specific files were processed correctly
    let main_rs_query = format!(
        "SELECT content_text, line_count, word_count FROM {} WHERE filename = 'main.rs'",
        result.table_name
    );
    
    let main_rs_data = sqlx::query(&main_rs_query)
        .fetch_optional(engine.get_database().get_pool())
        .await?;
    
    assert!(main_rs_data.is_some(), "Should find main.rs");
    
    let main_rs_row = main_rs_data.unwrap();
    let content: Option<String> = main_rs_row.get("content_text");
    let line_count: Option<i32> = main_rs_row.get("line_count");
    let word_count: Option<i32> = main_rs_row.get("word_count");
    
    assert!(content.is_some(), "main.rs should have content");
    assert!(line_count.unwrap_or(0) > 10, "main.rs should have multiple lines");
    assert!(word_count.unwrap_or(0) > 20, "main.rs should have multiple words");
    
    let content = content.unwrap();
    assert!(content.contains("authenticate_user"), "Should contain function name");
    assert!(content.contains("HashMap"), "Should contain imports");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_sql_query_interface_workflow() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    let db = create_test_database().await?;
    let engine = IngestionEngine::new(db).await?;
    
    // Ingest the repository first
    let result = engine.ingest_source(
        SourceType::LocalFolder(repo_path),
        None,
    ).await?;
    
    // Test SQL query interface
    let sql_command = SqlCommand {
        query: format!(
            "SELECT filepath, filename, file_type, line_count FROM {} WHERE extension = 'rs' ORDER BY filename",
            result.table_name
        ),
        limit: Some(10),
        offset: None,
    };
    
    let query_start = Instant::now();
    let query_results = engine.execute_sql_query(&sql_command).await?;
    let query_duration = query_start.elapsed();
    
    // Performance contract: Query should be fast
    assert!(
        query_duration < Duration::from_millis(500),
        "Query took too long: {:?}",
        query_duration
    );
    
    assert!(query_results.len() > 0, "Should return Rust files");
    assert!(query_results.len() <= 10, "Should respect limit");
    
    // Verify query results structure
    for result in &query_results {
        assert!(result.contains_key("filepath"), "Should have filepath");
        assert!(result.contains_key("filename"), "Should have filename");
        assert!(result.contains_key("file_type"), "Should have file_type");
        
        let filename: &str = result.get("filename").unwrap().as_str().unwrap();
        assert!(filename.ends_with(".rs"), "Should be Rust file");
        
        let file_type: &str = result.get("file_type").unwrap().as_str().unwrap();
        assert_eq!(file_type, "direct_text", "Rust files should be direct_text");
    }
    
    // Test full-text search query
    let search_query = format!(
        "SELECT filepath, filename FROM {} 
         WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'authentication')
         ORDER BY filename",
        result.table_name
    );
    
    let search_command = SqlCommand {
        query: search_query,
        limit: Some(5),
        offset: None,
    };
    
    let search_results = engine.execute_sql_query(&search_command).await?;
    assert!(search_results.len() > 0, "Should find authentication-related files");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_table_management_workflow() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    let db = create_test_database().await?;
    let engine = IngestionEngine::new(db).await?;
    
    // Create multiple ingestions
    let mut table_names = Vec::new();
    
    for i in 0..3 {
        let result = engine.ingest_source(
            SourceType::LocalFolder(repo_path.clone()),
            None,
        ).await?;
        
        table_names.push(result.table_name);
        
        // Small delay to ensure different timestamps
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    // Test list tables command
    let list_command = ListTablesCommand {};
    let tables = engine.list_ingestion_tables(&list_command).await?;
    
    assert!(tables.len() >= 3, "Should have at least 3 tables");
    
    // Verify our tables are in the list
    for table_name in &table_names {
        assert!(
            tables.iter().any(|t| &t.table_name == table_name),
            "Table {} should be in the list",
            table_name
        );
    }
    
    // Test table sampling
    for table_name in &table_names {
        let sample_query = format!("SELECT * FROM {} LIMIT 3", table_name);
        let sample_command = SqlCommand {
            query: sample_query,
            limit: Some(3),
            offset: None,
        };
        
        let sample_results = engine.execute_sql_query(&sample_command).await?;
        assert!(sample_results.len() > 0, "Should have sample data");
        assert!(sample_results.len() <= 3, "Should respect limit");
    }
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_ide_integration_workflow() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    let db = create_test_database().await?;
    let engine = IngestionEngine::new(db).await?;
    
    // Ingest repository
    let result = engine.ingest_source(
        SourceType::LocalFolder(repo_path),
        None,
    ).await?;
    
    // Test query-prepare workflow
    let temp_file = temp_dir.path().join("query_results.txt");
    let tasks_file = temp_dir.path().join("analysis_tasks.md");
    let output_table = "QUERYRESULT_auth_analysis";
    
    let query = format!(
        "SELECT filepath, filename, content_text FROM {} WHERE content_text LIKE '%auth%'",
        result.table_name
    );
    
    let prepare_result = engine.prepare_query_for_ide(
        &query,
        &temp_file,
        &tasks_file,
        output_table,
    ).await?;
    
    // Verify temp file was created with results
    assert!(temp_file.exists(), "Temp file should be created");
    let temp_content = std::fs::read_to_string(&temp_file)?;
    assert!(!temp_content.is_empty(), "Temp file should have content");
    assert!(temp_content.contains("auth"), "Should contain auth-related content");
    
    // Verify tasks file was created
    assert!(tasks_file.exists(), "Tasks file should be created");
    let tasks_content = std::fs::read_to_string(&tasks_file)?;
    assert!(!tasks_content.is_empty(), "Tasks file should have content");
    assert!(tasks_content.contains("# Analysis Tasks"), "Should have task structure");
    assert!(tasks_content.contains("- [ ]"), "Should have checkboxes");
    
    // Verify output table was created
    let table_exists = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = $1
        )"
    )
    .bind(output_table)
    .fetch_one(engine.get_database().get_pool())
    .await?;
    
    assert!(table_exists, "Output table should be created");
    
    // Test storing results back
    let analysis_result = "Analysis complete: Found 3 authentication-related functions";
    let result_file = temp_dir.path().join("analysis_result.txt");
    std::fs::write(&result_file, analysis_result)?;
    
    engine.store_analysis_result(
        output_table,
        &result_file,
        &query,
    ).await?;
    
    // Verify result was stored
    let stored_result = sqlx::query_scalar::<_, String>(
        &format!("SELECT analysis_result FROM {} LIMIT 1", output_table)
    )
    .fetch_one(engine.get_database().get_pool())
    .await?;
    
    assert_eq!(stored_result, analysis_result);
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_error_handling_and_recovery() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db = create_test_database().await?;
    let engine = IngestionEngine::new(db).await?;
    
    // Test ingesting non-existent folder
    let non_existent = temp_dir.path().join("does_not_exist");
    let result = engine.ingest_source(
        SourceType::LocalFolder(non_existent),
        None,
    ).await;
    
    assert!(result.is_err(), "Should fail for non-existent folder");
    
    // Test invalid SQL query
    let invalid_query = SqlCommand {
        query: "SELECT * FROM non_existent_table".to_string(),
        limit: None,
        offset: None,
    };
    
    let query_result = engine.execute_sql_query(&invalid_query).await;
    assert!(query_result.is_err(), "Should fail for invalid query");
    
    // Test recovery after error - should still work for valid operations
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    let valid_result = engine.ingest_source(
        SourceType::LocalFolder(repo_path),
        None,
    ).await?;
    
    assert!(valid_result.files_processed > 0, "Should recover and work normally");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_performance_contracts_end_to_end() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    // Add more files to test performance
    for i in 0..50 {
        let file_path = repo_path.join(format!("generated_file_{}.rs", i));
        let content = format!(
            "// Generated file {}\nfn function_{}() {{\n    println!(\"Function {}\");\n}}\n",
            i, i, i
        );
        std::fs::write(file_path, content)?;
    }
    
    let db = create_test_database().await?;
    let engine = IngestionEngine::new(db).await?;
    
    // Test ingestion performance contract
    let ingestion_start = Instant::now();
    let result = engine.ingest_source(
        SourceType::LocalFolder(repo_path),
        None,
    ).await?;
    let ingestion_duration = ingestion_start.elapsed();
    
    // Performance contract: Should process >50 files in reasonable time
    assert!(result.files_processed >= 50, "Should process at least 50 files");
    
    let throughput = result.files_processed as f64 / ingestion_duration.as_secs_f64();
    assert!(
        throughput >= 10.0, // At least 10 files/second
        "Ingestion throughput was {:.2} files/sec, expected >= 10",
        throughput
    );
    
    // Test query performance contract
    let query_start = Instant::now();
    let query_command = SqlCommand {
        query: format!(
            "SELECT COUNT(*) as count FROM {} WHERE file_type = 'direct_text'",
            result.table_name
        ),
        limit: None,
        offset: None,
    };
    
    let query_results = engine.execute_sql_query(&query_command).await?;
    let query_duration = query_start.elapsed();
    
    // Performance contract: Queries should complete within 1 second
    assert!(
        query_duration < Duration::from_secs(1),
        "Query took {:?}, expected < 1 second",
        query_duration
    );
    
    assert_eq!(query_results.len(), 1, "Should return one result");
    
    // Test search performance
    let search_start = Instant::now();
    let search_command = SqlCommand {
        query: format!(
            "SELECT filepath FROM {} 
             WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'function')
             LIMIT 10",
            result.table_name
        ),
        limit: Some(10),
        offset: None,
    };
    
    let search_results = engine.execute_sql_query(&search_command).await?;
    let search_duration = search_start.elapsed();
    
    // Performance contract: Full-text search should be fast
    assert!(
        search_duration < Duration::from_millis(500),
        "Search took {:?}, expected < 500ms",
        search_duration
    );
    
    assert!(search_results.len() > 0, "Should find function-related files");
    
    println!(
        "Performance test results:\n- Ingestion: {:.2} files/sec\n- Query: {:?}\n- Search: {:?}",
        throughput, query_duration, search_duration
    );
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_concurrent_operations() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    let db = create_test_database().await?;
    
    // Test concurrent ingestions
    let mut tasks = Vec::new();
    
    for i in 0..3 {
        let repo_clone = repo_path.clone();
        let db_clone = db.clone();
        
        let task = tokio::spawn(async move {
            let engine = IngestionEngine::new(db_clone).await?;
            
            let result = engine.ingest_source(
                SourceType::LocalFolder(repo_clone),
                None,
            ).await?;
            
            anyhow::Ok((i, result))
        });
        
        tasks.push(task);
    }
    
    // Wait for all ingestions to complete
    let mut results = Vec::new();
    for task in tasks {
        let (i, result) = task.await??;
        results.push((i, result));
    }
    
    // Verify all ingestions succeeded
    assert_eq!(results.len(), 3, "All concurrent ingestions should succeed");
    
    for (i, result) in &results {
        assert!(result.files_processed > 0, "Ingestion {} should process files", i);
        assert!(!result.table_name.is_empty(), "Ingestion {} should create table", i);
    }
    
    // Verify tables are distinct
    let table_names: Vec<_> = results.iter().map(|(_, r)| &r.table_name).collect();
    let unique_tables: std::collections::HashSet<_> = table_names.iter().collect();
    assert_eq!(unique_tables.len(), 3, "Should create distinct tables");
    
    // Test concurrent queries
    let engine = IngestionEngine::new(db).await?;
    let mut query_tasks = Vec::new();
    
    for (i, result) in &results {
        let table_name = result.table_name.clone();
        let engine_clone = engine.clone();
        
        let task = tokio::spawn(async move {
            let query_command = SqlCommand {
                query: format!("SELECT COUNT(*) FROM {}", table_name),
                limit: None,
                offset: None,
            };
            
            let query_result = engine_clone.execute_sql_query(&query_command).await?;
            anyhow::Ok((*i, query_result))
        });
        
        query_tasks.push(task);
    }
    
    // Wait for all queries to complete
    for task in query_tasks {
        let (i, query_result) = task.await??;
        assert_eq!(query_result.len(), 1, "Query {} should return count", i);
    }
    
    Ok(())
}

/// Test memory usage during large operations
#[tokio::test]
#[serial]
async fn test_memory_usage_bounds() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let repo_path = create_comprehensive_test_repo(&temp_dir);
    
    // Create many files to test memory management
    for i in 0..200 {
        let file_path = repo_path.join(format!("large_file_{}.rs", i));
        let content = format!(
            "// Large file {}\n{}\n",
            i,
            "// This is a line of code\n".repeat(100) // 100 lines per file
        );
        std::fs::write(file_path, content)?;
    }
    
    let db = create_test_database().await?;
    let engine = IngestionEngine::new(db).await?;
    
    // Monitor memory usage during ingestion
    let initial_memory = get_memory_usage();
    
    let result = engine.ingest_source(
        SourceType::LocalFolder(repo_path),
        None,
    ).await?;
    
    let final_memory = get_memory_usage();
    let memory_increase = final_memory.saturating_sub(initial_memory);
    
    // Memory usage should be reasonable (not proportional to total file size)
    assert!(
        memory_increase < 100 * 1024 * 1024, // < 100MB increase
        "Memory usage increased by {} bytes, expected < 100MB",
        memory_increase
    );
    
    assert!(result.files_processed >= 200, "Should process all files");
    
    println!(
        "Memory usage test: processed {} files, memory increase: {} bytes",
        result.files_processed, memory_increase
    );
    
    Ok(())
}

// Helper function to get approximate memory usage
fn get_memory_usage() -> usize {
    // This is a simplified memory check
    std::process::Command::new("ps")
        .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()
        })
        .map(|kb| kb * 1024) // Convert KB to bytes
        .unwrap_or(0)
}