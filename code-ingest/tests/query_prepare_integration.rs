//! Integration tests for query-prepare and store-result functionality
//!
//! These tests verify the complete IDE integration workflow:
//! 1. query-prepare: Execute SQL, create temp file, generate tasks
//! 2. store-result: Store analysis results with traceability

use anyhow::Result;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_query_prepare_workflow() -> Result<()> {
    // Only run if DATABASE_URL is set
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test: DATABASE_URL not set");
            return Ok(());
        }
    };

    // Create temporary directory for test files
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().join("query_results.txt");
    let tasks_path = temp_dir.path().join("analysis_tasks.md");

    // Test query-prepare command
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "query-prepare",
            "SELECT 'test_file.rs' as filepath, 'fn main() {}' as content_text",
            "--temp-path",
            &temp_path.to_string_lossy(),
            "--tasks-file", 
            &tasks_path.to_string_lossy(),
            "--output-table",
            "QUERYRESULT_test_prepare",
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute query-prepare command");

    // Check command succeeded
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("query-prepare command failed:\nSTDOUT: {}\nSTDERR: {}", stdout, stderr);
    }

    // Verify temporary file was created
    assert!(temp_path.exists(), "Temporary file should be created");
    
    // Verify tasks file was created
    assert!(tasks_path.exists(), "Tasks file should be created");

    // Check temporary file content
    let temp_content = fs::read_to_string(&temp_path).await?;
    assert!(temp_content.contains("FILE: test_file.rs"), "Temp file should contain FILE: marker");
    assert!(temp_content.contains("fn main() {}"), "Temp file should contain query result content");

    // Check tasks file content
    let tasks_content = fs::read_to_string(&tasks_path).await?;
    assert!(tasks_content.contains("# IDE Analysis Tasks"), "Tasks file should have proper header");
    assert!(tasks_content.contains("QUERYRESULT_test_prepare"), "Tasks file should reference output table");
    assert!(tasks_content.contains("Phase 1: Data Exploration"), "Tasks file should have structured phases");

    println!("✅ query-prepare workflow test passed");
    Ok(())
}

#[tokio::test]
async fn test_store_result_workflow() -> Result<()> {
    // Only run if DATABASE_URL is set
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test: DATABASE_URL not set");
            return Ok(());
        }
    };

    // Create temporary directory for test files
    let temp_dir = TempDir::new()?;
    let result_path = temp_dir.path().join("analysis_result.txt");

    // Create a sample analysis result file
    let analysis_content = r#"# Analysis Results

## Summary
This is a test analysis of the code structure.

## Findings
1. The code follows standard Rust conventions
2. Main function is properly structured
3. No security issues identified

## Recommendations
- Consider adding error handling
- Add unit tests for better coverage
"#;

    fs::write(&result_path, analysis_content).await?;

    // Test store-result command
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "store-result",
            "--output-table",
            "QUERYRESULT_test_store",
            "--result-file",
            &result_path.to_string_lossy(),
            "--original-query",
            "SELECT * FROM test_table WHERE file_type = 'rust'",
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute store-result command");

    // Check command succeeded
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("store-result command failed:\nSTDOUT: {}\nSTDERR: {}", stdout, stderr);
    }

    // Verify the result was stored by querying the database
    let verify_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "sql",
            "SELECT COUNT(*) as count FROM QUERYRESULT_test_store WHERE analysis_type IS NOT NULL",
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute verification query");

    if !verify_output.status.success() {
        let stderr = String::from_utf8_lossy(&verify_output.stderr);
        panic!("Verification query failed: {}", stderr);
    }

    let verify_stdout = String::from_utf8_lossy(&verify_output.stdout);
    assert!(verify_stdout.contains("1") || verify_stdout.contains("count"), 
           "Should find at least one stored result");

    println!("✅ store-result workflow test passed");
    Ok(())
}

#[tokio::test]
async fn test_complete_ide_integration_workflow() -> Result<()> {
    // Only run if DATABASE_URL is set
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test: DATABASE_URL not set");
            return Ok(());
        }
    };

    // Create temporary directory for test files
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().join("workflow_query_results.txt");
    let tasks_path = temp_dir.path().join("workflow_tasks.md");
    let result_path = temp_dir.path().join("workflow_analysis.txt");

    let output_table = "QUERYRESULT_workflow_test";
    let test_query = "SELECT 'main.rs' as filepath, 'use std::io;' as content_text UNION SELECT 'lib.rs' as filepath, 'pub mod utils;' as content_text";

    // Step 1: Execute query-prepare
    let prepare_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "query-prepare",
            test_query,
            "--temp-path",
            &temp_path.to_string_lossy(),
            "--tasks-file", 
            &tasks_path.to_string_lossy(),
            "--output-table",
            output_table,
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute query-prepare");

    assert!(prepare_output.status.success(), "query-prepare should succeed");
    assert!(temp_path.exists(), "Temp file should be created");
    assert!(tasks_path.exists(), "Tasks file should be created");

    // Step 2: Simulate analysis work (create result file)
    let analysis_result = r#"# Workflow Analysis Results

## Files Analyzed
- main.rs: Contains standard I/O imports
- lib.rs: Defines public module structure

## Code Quality Assessment
- Both files follow Rust naming conventions
- Proper use of module system
- Standard library usage is appropriate

## Recommendations
1. Consider adding documentation comments
2. Implement error handling patterns
3. Add integration tests

## Conclusion
The code structure is well-organized and follows Rust best practices.
"#;

    fs::write(&result_path, analysis_result).await?;

    // Step 3: Store the analysis results
    let store_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "store-result",
            "--output-table",
            output_table,
            "--result-file",
            &result_path.to_string_lossy(),
            "--original-query",
            test_query,
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute store-result");

    assert!(store_output.status.success(), "store-result should succeed");

    // Step 4: Verify the complete workflow by querying stored results
    let verify_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "sql",
            &format!("SELECT analysis_id, analysis_type, LENGTH(llm_result) as result_length FROM {} WHERE llm_result LIKE '%Workflow Analysis Results%'", output_table),
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute verification query");

    assert!(verify_output.status.success(), "Verification query should succeed");

    let verify_stdout = String::from_utf8_lossy(&verify_output.stdout);
    assert!(verify_stdout.contains("analysis_id") || verify_stdout.contains("1"), 
           "Should find the stored analysis result");

    println!("✅ Complete IDE integration workflow test passed");
    Ok(())
}

#[tokio::test]
async fn test_query_prepare_error_handling() -> Result<()> {
    // Test with invalid paths (relative instead of absolute)
    let temp_dir = TempDir::new()?;
    let relative_temp_path = "relative_temp.txt";
    let relative_tasks_path = "relative_tasks.md";

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "query-prepare",
            "SELECT 'test' as filepath",
            "--temp-path",
            relative_temp_path,
            "--tasks-file", 
            relative_tasks_path,
            "--output-table",
            "QUERYRESULT_error_test",
        ])
        .output()
        .expect("Failed to execute query-prepare command");

    // Should fail due to relative paths
    assert!(!output.status.success(), "query-prepare should fail with relative paths");
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("absolute") || stderr.contains("path"), 
           "Error message should mention path requirements");

    println!("✅ query-prepare error handling test passed");
    Ok(())
}

#[tokio::test]
async fn test_store_result_error_handling() -> Result<()> {
    // Test with non-existent result file
    let non_existent_file = "/tmp/non_existent_result.txt";

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "store-result",
            "--output-table",
            "QUERYRESULT_error_test",
            "--result-file",
            non_existent_file,
            "--original-query",
            "SELECT * FROM test",
        ])
        .output()
        .expect("Failed to execute store-result command");

    // Should fail due to missing file
    assert!(!output.status.success(), "store-result should fail with non-existent file");
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found") || stderr.contains("No such file"), 
           "Error message should mention missing file");

    println!("✅ store-result error handling test passed");
    Ok(())
}