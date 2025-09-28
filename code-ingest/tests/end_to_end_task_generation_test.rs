//! End-to-end integration tests for task generation workflow
//! 
//! Tests the complete pipeline: count → extract → generate tasks
//! Validates output file structure and content accuracy
//! Tests with real INGEST_* table data
//! Verifies L1-L8 analysis methodology integration

use anyhow::Result;
use code_ingest::{
    database::{Database, DatabaseConfig},
    tasks::{
        DatabaseQueryEngine, ContentExtractor, HierarchicalTaskDivider, 
        L1L8MarkdownGenerator, ContentTriple, TaskHierarchy
    },
    error::TaskError,
};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tempfile::TempDir;
use serial_test::serial;
use tokio;

/// Helper to create test database with sample INGEST table
async fn create_test_database_with_sample_data() -> Result<(Database, String)> {
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/code_ingest_test".to_string());
    
    let config = DatabaseConfig {
        database_url,
        max_connections: 10,
        connection_timeout_seconds: 30,
    };
    
    let db = Database::new(config).await?;
    
    // Create a test table with sample data
    let table_name = format!("INGEST_TEST_{}", chrono::Utc::now().timestamp());
    
    let create_table_sql = format!(
        r#"
        CREATE TABLE "{}" (
            file_id SERIAL PRIMARY KEY,
            filepath TEXT NOT NULL,
            filename TEXT NOT NULL,
            extension TEXT,
            file_size_bytes BIGINT,
            line_count INTEGER,
            word_count INTEGER,
            content_text TEXT,
            file_type TEXT,
            relative_path TEXT,
            absolute_path TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
        "#,
        table_name
    );
    
    sqlx::query(&create_table_sql)
        .execute(db.pool())
        .await?;
    
    // Insert sample data representing a typical Rust project
    let sample_data = vec![
        (
            "src/main.rs", "main.rs", "rs", 1024, 45, 200,
            r#"//! Main application entry point
use std::collections::HashMap;
use crate::auth::User;
use crate::database::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting application...");
    
    let db = Database::connect("postgresql://localhost/myapp")?;
    let user = User::authenticate("admin", "password")?;
    
    println!("User {} authenticated successfully", user.username);
    Ok(())
}
"#,
            "direct_text"
        ),
        (
            "src/lib.rs", "lib.rs", "rs", 512, 25, 100,
            r#"//! Library root module
pub mod auth;
pub mod database;
pub mod utils;

pub use auth::*;
pub use database::*;
pub use utils::*;

/// Application configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub port: u16,
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_url: "postgresql://localhost/myapp".to_string(),
            port: 8080,
            debug: false,
        }
    }
}
"#,
            "direct_text"
        ),
        (
            "src/auth.rs", "auth.rs", "rs", 2048, 80, 350,
            r#"//! Authentication module
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: u64,
    pub username: String,
    pub email: String,
    pub roles: Vec<String>,
}

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,
    #[error("User not found: {username}")]
    UserNotFound { username: String },
    #[error("Permission denied")]
    PermissionDenied,
}

impl User {
    pub fn authenticate(username: &str, password: &str) -> Result<Self, AuthError> {
        // Mock authentication logic
        if username == "admin" && password == "password" {
            Ok(User {
                id: 1,
                username: username.to_string(),
                email: "admin@example.com".to_string(),
                roles: vec!["admin".to_string()],
            })
        } else {
            Err(AuthError::InvalidCredentials)
        }
    }
    
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(&role.to_string())
    }
}
"#,
            "direct_text"
        ),
        (
            "src/database.rs", "database.rs", "rs", 1536, 60, 280,
            r#"//! Database operations module
use sqlx::{PgPool, Row};
use anyhow::Result;
use crate::auth::User;

#[derive(Clone)]
pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        Ok(Self { pool })
    }
    
    pub async fn get_user_by_id(&self, user_id: u64) -> Result<Option<User>> {
        let row = sqlx::query("SELECT id, username, email FROM users WHERE id = $1")
            .bind(user_id as i64)
            .fetch_optional(&self.pool)
            .await?;
        
        if let Some(row) = row {
            Ok(Some(User {
                id: row.get::<i64, _>("id") as u64,
                username: row.get("username"),
                email: row.get("email"),
                roles: vec![], // Would be loaded separately
            }))
        } else {
            Ok(None)
        }
    }
    
    pub async fn create_user(&self, user: &User) -> Result<u64> {
        let row = sqlx::query(
            "INSERT INTO users (username, email) VALUES ($1, $2) RETURNING id"
        )
        .bind(&user.username)
        .bind(&user.email)
        .fetch_one(&self.pool)
        .await?;
        
        Ok(row.get::<i64, _>("id") as u64)
    }
}
"#,
            "direct_text"
        ),
        (
            "Cargo.toml", "Cargo.toml", "toml", 800, 35, 150,
            r#"[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-rustls"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tempfile = "3.0"
serial_test = "3.0"

[[bin]]
name = "main"
path = "src/main.rs"
"#,
            "direct_text"
        ),
        (
            "README.md", "README.md", "md", 1200, 50, 220,
            r#"# Test Project

A sample Rust project for testing the code ingestion and analysis system.

## Features

- User authentication system
- Database operations with PostgreSQL
- Modular architecture
- Error handling with `thiserror`
- Async operations with `tokio`

## Usage

```bash
cargo run
```

## Testing

```bash
cargo test
```

## Architecture

The project follows a layered architecture:

1. **Main**: Application entry point
2. **Auth**: Authentication and authorization
3. **Database**: Data persistence layer
4. **Utils**: Utility functions

## Dependencies

- `tokio`: Async runtime
- `sqlx`: Database operations
- `serde`: Serialization
- `anyhow`: Error handling
- `thiserror`: Structured errors
"#,
            "direct_text"
        ),
    ];
    
    for (filepath, filename, extension, file_size, line_count, word_count, content, file_type) in sample_data {
        let insert_sql = format!(
            r#"
            INSERT INTO "{}" (
                filepath, filename, extension, file_size_bytes, 
                line_count, word_count, content_text, file_type,
                relative_path, absolute_path
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            "#,
            table_name
        );
        
        sqlx::query(&insert_sql)
            .bind(filepath)
            .bind(filename)
            .bind(extension)
            .bind(file_size as i64)
            .bind(line_count)
            .bind(word_count)
            .bind(content)
            .bind(file_type)
            .bind(filepath) // relative_path same as filepath for test
            .bind(format!("/test/project/{}", filepath)) // absolute_path
            .execute(db.pool())
            .await?;
    }
    
    Ok((db, table_name))
}

#[tokio::test]
#[serial]
async fn test_complete_pipeline_count_extract_generate_tasks() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() && std::env::var("TEST_DATABASE_URL").is_err() {
        println!("Skipping end-to-end test - DATABASE_URL not set");
        return Ok(());
    }

    let temp_dir = TempDir::new()?;
    let (db, table_name) = create_test_database_with_sample_data().await?;
    
    println!("🧪 Testing complete pipeline with table: {}", table_name);
    
    // Step 1: Count rows
    println!("📊 Step 1: Counting rows...");
    let query_engine = DatabaseQueryEngine::new(Arc::new(db.pool().clone()));
    
    let start_time = Instant::now();
    let row_count = query_engine.count_rows(&table_name).await?;
    let count_duration = start_time.elapsed();
    
    assert_eq!(row_count, 6, "Should have 6 sample rows");
    assert!(count_duration < Duration::from_secs(1), "Count should be fast");
    
    println!("✅ Row count: {} (took {:?})", row_count, count_duration);
    
    // Step 2: Extract content to A/B/C files
    println!("📁 Step 2: Extracting content to A/B/C files...");
    let output_dir = temp_dir.path().join(".raw_data_202509");
    let content_extractor = ContentExtractor::new(Arc::new(db.pool().clone()), output_dir.clone());
    
    let start_time = Instant::now();
    let content_triples = content_extractor.extract_all_rows(&table_name).await?;
    let extract_duration = start_time.elapsed();
    
    assert_eq!(content_triples.len(), 6, "Should create 6 content triples");
    assert!(extract_duration < Duration::from_secs(10), "Extraction should be reasonably fast");
    
    println!("✅ Content extraction: {} triples (took {:?})", content_triples.len(), extract_duration);
    
    // Validate A/B/C files were created
    for (i, triple) in content_triples.iter().enumerate() {
        assert!(triple.content_a.exists(), "Content A file should exist for row {}", i + 1);
        assert!(triple.content_b.exists(), "Content B file should exist for row {}", i + 1);
        assert!(triple.content_c.exists(), "Content C file should exist for row {}", i + 1);
        
        // Validate file naming convention
        let expected_base = format!("{}_{}_Content", table_name, i + 1);
        assert!(triple.content_a.file_name().unwrap().to_str().unwrap().starts_with(&expected_base));
        assert!(triple.content_b.file_name().unwrap().to_str().unwrap().contains("_L1"));
        assert!(triple.content_c.file_name().unwrap().to_str().unwrap().contains("_L2"));
        
        // Validate file contents are not empty
        let content_a = tokio::fs::read_to_string(&triple.content_a).await?;
        let content_b = tokio::fs::read_to_string(&triple.content_b).await?;
        let content_c = tokio::fs::read_to_string(&triple.content_c).await?;
        
        assert!(!content_a.is_empty(), "Content A should not be empty for row {}", i + 1);
        assert!(!content_b.is_empty(), "Content B should not be empty for row {}", i + 1);
        assert!(!content_c.is_empty(), "Content C should not be empty for row {}", i + 1);
        
        // Validate L1 context contains expected sections
        assert!(content_b.contains("# L1 Context: Immediate File Context"), "L1 should have proper header");
        assert!(content_b.contains("## File Information"), "L1 should have file info section");
        assert!(content_b.contains("## Import/Include Analysis"), "L1 should have import analysis");
        
        // Validate L2 context contains expected sections
        assert!(content_c.contains("# L2 Context: Architectural Context"), "L2 should have proper header");
        assert!(content_c.contains("## Package/Module Structure"), "L2 should have package structure");
        assert!(content_c.contains("## Architectural Patterns"), "L2 should have pattern analysis");
    }
    
    // Step 3: Generate hierarchical tasks
    println!("📋 Step 3: Generating hierarchical tasks...");
    let task_divider = HierarchicalTaskDivider::new(4, 7); // 4 levels, 7 groups per level
    
    let start_time = Instant::now();
    let task_hierarchy = task_divider.create_hierarchy(content_triples.clone())?;
    let hierarchy_duration = start_time.elapsed();
    
    assert!(hierarchy_duration < Duration::from_secs(5), "Hierarchy creation should be fast");
    
    println!("✅ Task hierarchy created (took {:?})", hierarchy_duration);
    
    // Validate hierarchy structure
    assert_eq!(task_hierarchy.levels.len(), 4, "Should have 4 levels");
    assert!(task_hierarchy.total_tasks > 0, "Should have tasks");
    
    // Validate level structure
    for (level_idx, level) in task_hierarchy.levels.iter().enumerate() {
        assert_eq!(level.level, level_idx + 1, "Level number should match index + 1");
        assert!(level.groups.len() <= 7, "Should have at most 7 groups per level");
        
        // Validate group IDs are properly formatted
        for group in &level.groups {
            let id_parts: Vec<&str> = group.id.split('.').collect();
            assert_eq!(id_parts.len(), level_idx + 1, "Group ID should have correct depth");
            
            // Validate each part is a number
            for part in id_parts {
                assert!(part.parse::<usize>().is_ok(), "Group ID part should be numeric: {}", part);
            }
        }
    }
    
    // Step 4: Generate markdown with L1-L8 methodology
    println!("📝 Step 4: Generating L1-L8 analysis markdown...");
    let prompt_file = temp_dir.path().join("test_prompt.md");
    
    // Create a test prompt file
    let prompt_content = r#"# L1-L8 Analysis Methodology

## Analysis Stages

### L1: Idiomatic Patterns & Micro-Optimizations
Analyze efficiency, bug reduction, raw performance, mechanical sympathy.

### L2: Design Patterns & Composition
Examine abstraction boundaries, API ergonomics, RAII variants, advanced trait usage.

### L3: Micro-Library Opportunities
Identify high-utility components under ~2000 LOC.

### L4: Macro-Library & Platform Opportunities
Find high-PMF ideas offering ecosystem dominance.

### L5: LLD Architecture Decisions & Invariants
Study concurrency models, state management, internal modularity.

### L6: Domain-Specific Architecture & Hardware Interaction
Analyze kernel bypass, GPU pipelines, OS abstractions.

### L7: Language Capability & Evolution
Identify limitations of Rust itself.

### L8: The Meta-Context
Archaeology of intent - analyze commit history, bug trackers, historical constraints.
"#;
    
    tokio::fs::write(&prompt_file, prompt_content).await?;
    
    let output_file = temp_dir.path().join("generated_tasks.md");
    let markdown_generator = L1L8MarkdownGenerator::new(prompt_file.clone(), temp_dir.path().to_path_buf());
    
    let start_time = Instant::now();
    let markdown_content = markdown_generator.generate_hierarchical_markdown(&task_hierarchy, &table_name)?;
    let markdown_duration = start_time.elapsed();
    
    // Write the generated markdown
    tokio::fs::write(&output_file, &markdown_content).await?;
    
    assert!(markdown_duration < Duration::from_secs(5), "Markdown generation should be fast");
    assert!(!markdown_content.is_empty(), "Generated markdown should not be empty");
    
    println!("✅ Markdown generated: {} bytes (took {:?})", markdown_content.len(), markdown_duration);
    
    // Step 5: Validate generated markdown structure
    println!("🔍 Step 5: Validating generated markdown structure...");
    
    // Check for proper task structure
    assert!(markdown_content.contains("# Implementation Plan"), "Should have main header");
    assert!(markdown_content.contains("- [ ]"), "Should have checkboxes");
    
    // Check for hierarchical numbering (1.2.3.4 format)
    let hierarchical_pattern = regex::Regex::new(r"- \[ \] \d+(\.\d+)*\.").unwrap();
    let hierarchical_matches: Vec<_> = hierarchical_pattern.find_iter(&markdown_content).collect();
    assert!(!hierarchical_matches.is_empty(), "Should have hierarchical task numbering");
    
    // Check for content file references
    assert!(markdown_content.contains("Content.txt"), "Should reference A files");
    assert!(markdown_content.contains("Content_L1.txt"), "Should reference B files");
    assert!(markdown_content.contains("Content_L2.txt"), "Should reference C files");
    
    // Check for L1-L8 analysis references
    assert!(markdown_content.contains("L1-L8"), "Should reference L1-L8 methodology");
    
    // Check for proper output paths
    assert!(markdown_content.contains("gringotts/WorkArea"), "Should have proper output paths");
    
    // Validate task format matches requirements
    let task_pattern = regex::Regex::new(
        r"- \[ \] \d+(\.\d+)*\. Analyze .+ row \d+\s*- \*\*Content\*\*: .+Content\.txt.+ as A \+ .+Content_L1\.txt.+ as B \+ .+Content_L2\.txt.+ as C"
    ).unwrap();
    let task_matches: Vec<_> = task_pattern.find_iter(&markdown_content).collect();
    assert!(!task_matches.is_empty(), "Should have properly formatted analysis tasks");
    
    println!("✅ Markdown structure validation passed");
    
    // Step 6: Performance validation
    println!("⚡ Step 6: Performance validation...");
    
    let total_duration = count_duration + extract_duration + hierarchy_duration + markdown_duration;
    println!("📊 Performance Summary:");
    println!("   - Count rows: {:?}", count_duration);
    println!("   - Extract content: {:?}", extract_duration);
    println!("   - Create hierarchy: {:?}", hierarchy_duration);
    println!("   - Generate markdown: {:?}", markdown_duration);
    println!("   - Total pipeline: {:?}", total_duration);
    
    // Performance contracts
    assert!(total_duration < Duration::from_secs(30), "Complete pipeline should finish within 30 seconds");
    
    let throughput = content_triples.len() as f64 / total_duration.as_secs_f64();
    assert!(throughput >= 0.2, "Should process at least 0.2 files/second (got {:.2})", throughput);
    
    println!("✅ Performance validation passed: {:.2} files/second", throughput);
    
    // Cleanup test table
    let drop_sql = format!("DROP TABLE IF EXISTS \"{}\"", table_name);
    sqlx::query(&drop_sql).execute(db.pool()).await?;
    
    println!("🎉 Complete pipeline test passed successfully!");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_real_ingest_table_data_validation() -> Result<()> {
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() && std::env::var("TEST_DATABASE_URL").is_err() {
        println!("Skipping real INGEST table test - DATABASE_URL not set");
        return Ok(());
    }

    let database_url = std::env::var("DATABASE_URL")
        .or_else(|_| std::env::var("TEST_DATABASE_URL"))
        .unwrap();
    
    let config = DatabaseConfig {
        database_url,
        max_connections: 5,
        connection_timeout_seconds: 30,
    };
    
    let db = Database::new(config).await?;
    
    // Look for existing INGEST tables
    let tables_query = "SELECT table_name FROM information_schema.tables 
                       WHERE table_schema = 'public' 
                       AND table_name LIKE 'INGEST_%' 
                       ORDER BY table_name DESC 
                       LIMIT 1";
    
    let table_row = sqlx::query(tables_query)
        .fetch_optional(db.pool())
        .await?;
    
    if let Some(row) = table_row {
        let table_name: String = row.get("table_name");
        println!("🔍 Testing with real INGEST table: {}", table_name);
        
        let temp_dir = TempDir::new()?;
        
        // Test the pipeline with real data
        let query_engine = DatabaseQueryEngine::new(Arc::new(db.pool().clone()));
        let row_count = query_engine.count_rows(&table_name).await?;
        
        println!("📊 Real table has {} rows", row_count);
        
        if row_count > 0 && row_count <= 50 { // Only test with reasonable sized tables
            let output_dir = temp_dir.path().join(".raw_data_202509");
            let content_extractor = ContentExtractor::new(Arc::new(db.pool().clone()), output_dir);
            
            // Extract first few rows only for testing
            let limited_query = format!(
                "SELECT file_id, filepath, filename, extension, file_size_bytes, 
                        line_count, word_count, content_text, file_type, 
                        relative_path, absolute_path 
                 FROM \"{}\" 
                 ORDER BY file_id 
                 LIMIT 3",
                table_name
            );
            
            let rows = sqlx::query(&limited_query)
                .fetch_all(db.pool())
                .await?;
            
            assert!(!rows.is_empty(), "Should have at least one row");
            
            // Test content extraction on first row
            if let Some(row) = rows.first() {
                let metadata = content_extractor.extract_row_metadata(&row)?;
                let content_triple = content_extractor.create_content_files(&metadata, 1, &table_name).await?;
                
                // Validate files were created
                assert!(content_triple.content_a.exists(), "Content A should exist");
                assert!(content_triple.content_b.exists(), "Content B should exist");
                assert!(content_triple.content_c.exists(), "Content C should exist");
                
                // Validate content
                let content_a = tokio::fs::read_to_string(&content_triple.content_a).await?;
                let content_b = tokio::fs::read_to_string(&content_triple.content_b).await?;
                let content_c = tokio::fs::read_to_string(&content_triple.content_c).await?;
                
                assert!(!content_a.is_empty(), "Content A should not be empty");
                assert!(!content_b.is_empty(), "Content B should not be empty");
                assert!(!content_c.is_empty(), "Content C should not be empty");
                
                println!("✅ Real data validation passed");
            }
        } else {
            println!("⚠️  Skipping real data test - table too large ({} rows) or empty", row_count);
        }
    } else {
        println!("⚠️  No INGEST tables found - skipping real data test");
    }
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_l1_l8_analysis_methodology_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create test content triple
    let content_triple = ContentTriple {
        content_a: temp_dir.path().join("test_content.txt"),
        content_b: temp_dir.path().join("test_content_L1.txt"),
        content_c: temp_dir.path().join("test_content_L2.txt"),
        row_number: 1,
        table_name: "TEST_TABLE".to_string(),
    };
    
    // Create test content files
    tokio::fs::write(&content_triple.content_a, "fn main() { println!(\"Hello, world!\"); }").await?;
    tokio::fs::write(&content_triple.content_b, "# L1 Context\nImmediate file context...").await?;
    tokio::fs::write(&content_triple.content_c, "# L2 Context\nArchitectural context...").await?;
    
    // Create L1-L8 prompt file
    let prompt_file = temp_dir.path().join("l1_l8_prompt.md");
    let prompt_content = r#"# L1-L8 Analysis Methodology

## L1: Idiomatic Patterns & Micro-Optimizations
Focus on efficiency, bug reduction, raw performance.

## L2: Design Patterns & Composition  
Examine abstraction boundaries, API ergonomics.

## L3: Micro-Library Opportunities
Identify high-utility components.

## L4: Macro-Library & Platform Opportunities
Find ecosystem dominance opportunities.

## L5: LLD Architecture Decisions & Invariants
Study concurrency models, state management.

## L6: Domain-Specific Architecture & Hardware Interaction
Analyze system-level interactions.

## L7: Language Capability & Evolution
Identify language limitations.

## L8: The Meta-Context
Archaeology of intent and historical constraints.
"#;
    
    tokio::fs::write(&prompt_file, prompt_content).await?;
    
    // Test markdown generation with L1-L8 methodology
    let markdown_generator = L1L8MarkdownGenerator::new(prompt_file, temp_dir.path().to_path_buf());
    
    // Create simple hierarchy for testing
    let task_divider = HierarchicalTaskDivider::new(2, 3);
    let hierarchy = task_divider.create_hierarchy(vec![content_triple])?;
    
    let markdown = markdown_generator.generate_hierarchical_markdown(&hierarchy, "TEST_TABLE")?;
    
    // Validate L1-L8 methodology integration
    assert!(markdown.contains("L1-L8"), "Should reference L1-L8 methodology");
    assert!(markdown.contains("insights of A alone"), "Should have A alone analysis");
    assert!(markdown.contains("A in context of B"), "Should have A in context of B analysis");
    assert!(markdown.contains("B in context of C"), "Should have B in context of C analysis");
    assert!(markdown.contains("A in context B & C"), "Should have A in context of B & C analysis");
    
    // Validate proper analysis stages are referenced
    for level in 1..=8 {
        // L1-L8 levels should be mentioned in the methodology
        assert!(prompt_content.contains(&format!("L{}", level)), "Prompt should contain L{} level", level);
    }
    
    println!("✅ L1-L8 methodology integration validated");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_output_file_structure_and_content_accuracy() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let output_dir = temp_dir.path().join(".raw_data_202509");
    
    // Create mock database pool (this test focuses on file structure)
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/code_ingest_test".to_string());
    
    let config = DatabaseConfig {
        database_url,
        max_connections: 5,
        connection_timeout_seconds: 30,
    };
    
    // Skip if can't connect to database
    let db = match Database::new(config).await {
        Ok(db) => db,
        Err(_) => {
            println!("Skipping file structure test - database not available");
            return Ok(());
        }
    };
    
    let content_extractor = ContentExtractor::new(Arc::new(db.pool().clone()), output_dir.clone());
    
    // Test file naming convention
    let test_metadata = code_ingest::tasks::content_extractor::RowMetadata {
        file_id: Some(1),
        filepath: Some("src/main.rs".to_string()),
        filename: Some("main.rs".to_string()),
        extension: Some("rs".to_string()),
        file_size_bytes: Some(1024),
        line_count: Some(45),
        word_count: Some(200),
        content_text: Some("fn main() { println!(\"Hello, world!\"); }".to_string()),
        file_type: Some("direct_text".to_string()),
        relative_path: Some("src/main.rs".to_string()),
        absolute_path: Some("/project/src/main.rs".to_string()),
    };
    
    let table_name = "TEST_TABLE_20250928101039";
    let row_number = 35;
    
    let content_triple = content_extractor.create_content_files(&test_metadata, row_number, table_name).await?;
    
    // Validate file naming convention matches requirements
    let expected_base = format!("{}_{}_Content", table_name, row_number);
    
    assert_eq!(
        content_triple.content_a.file_name().unwrap().to_str().unwrap(),
        format!("{}.txt", expected_base),
        "Content A file should follow naming convention"
    );
    
    assert_eq!(
        content_triple.content_b.file_name().unwrap().to_str().unwrap(),
        format!("{}_L1.txt", expected_base),
        "Content B file should follow naming convention"
    );
    
    assert_eq!(
        content_triple.content_c.file_name().unwrap().to_str().unwrap(),
        format!("{}_L2.txt", expected_base),
        "Content C file should follow naming convention"
    );
    
    // Validate directory structure
    assert_eq!(
        content_triple.content_a.parent().unwrap(),
        output_dir,
        "Files should be in correct output directory"
    );
    
    // Validate file contents
    let content_a = tokio::fs::read_to_string(&content_triple.content_a).await?;
    let content_b = tokio::fs::read_to_string(&content_triple.content_b).await?;
    let content_c = tokio::fs::read_to_string(&content_triple.content_c).await?;
    
    // Content A should be raw content
    assert_eq!(content_a.trim(), "fn main() { println!(\"Hello, world!\"); }", "Content A should be raw content");
    
    // Content B should have L1 context structure
    assert!(content_b.contains("# L1 Context: Immediate File Context"), "Content B should have L1 header");
    assert!(content_b.contains("## File Information"), "Content B should have file info");
    assert!(content_b.contains("src/main.rs"), "Content B should contain filepath");
    assert!(content_b.contains("main.rs"), "Content B should contain filename");
    assert!(content_b.contains("## Original File Content"), "Content B should include original content");
    
    // Content C should have L2 context structure
    assert!(content_c.contains("# L2 Context: Architectural Context"), "Content C should have L2 header");
    assert!(content_c.contains("## Package/Module Structure"), "Content C should have package structure");
    assert!(content_c.contains("## Architectural Patterns"), "Content C should have pattern analysis");
    
    println!("✅ Output file structure and content accuracy validated");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_error_handling_and_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Test with invalid database connection
    let invalid_config = DatabaseConfig {
        database_url: "postgresql://invalid:invalid@localhost:9999/invalid".to_string(),
        max_connections: 5,
        connection_timeout_seconds: 1,
    };
    
    let db_result = Database::new(invalid_config).await;
    assert!(db_result.is_err(), "Should fail with invalid database connection");
    
    // Test with valid database but invalid table
    if let Ok(database_url) = std::env::var("TEST_DATABASE_URL") {
        let config = DatabaseConfig {
            database_url,
            max_connections: 5,
            connection_timeout_seconds: 30,
        };
        
        let db = Database::new(config).await?;
        let query_engine = DatabaseQueryEngine::new(Arc::new(db.pool().clone()));
        
        // Test with non-existent table
        let result = query_engine.count_rows("NON_EXISTENT_TABLE").await;
        assert!(result.is_err(), "Should fail with non-existent table");
        
        if let Err(e) = result {
            match e {
                TaskError::DatabaseError(db_err) => {
                    // Should be a table not found error
                    assert!(db_err.to_string().contains("not found") || db_err.to_string().contains("does not exist"));
                }
                _ => panic!("Expected DatabaseError for non-existent table"),
            }
        }
        
        // Test with invalid table name (SQL injection attempt)
        let result = query_engine.count_rows("'; DROP TABLE users; --").await;
        assert!(result.is_err(), "Should fail with invalid table name");
        
        println!("✅ Error handling validation passed");
    } else {
        println!("⚠️  Skipping error handling test - TEST_DATABASE_URL not set");
    }
    
    Ok(())
}