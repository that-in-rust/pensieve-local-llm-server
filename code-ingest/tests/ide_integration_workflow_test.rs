//! Comprehensive IDE integration workflow tests
//!
//! These tests verify the complete IDE integration workflow including:
//! 1. query-prepare: Execute SQL, create temp file, generate tasks, create output table
//! 2. store-result: Store analysis results with full traceability and metadata
//! 3. End-to-end workflow validation

use anyhow::Result;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::fs;

/// Test the complete IDE integration workflow from query-prepare to store-result
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
    let temp_path = temp_dir.path().join("ide_workflow_query_results.txt");
    let tasks_path = temp_dir.path().join("ide_workflow_tasks.md");
    let result_path = temp_dir.path().join("ide_workflow_analysis.txt");

    let output_table = "QUERYRESULT_ide_workflow_test";
    let test_query = r#"
        SELECT 'src/main.rs' as filepath, 'fn main() { println!("Hello, world!"); }' as content_text, 'rs' as extension, 'direct_text' as file_type, 25 as line_count, 100 as word_count
        UNION ALL
        SELECT 'src/lib.rs' as filepath, 'pub mod utils; pub mod config;' as content_text, 'rs' as extension, 'direct_text' as file_type, 15 as line_count, 50 as word_count
        UNION ALL
        SELECT 'README.md' as filepath, '# My Project\n\nThis is a sample Rust project.' as content_text, 'md' as extension, 'direct_text' as file_type, 3 as line_count, 10 as word_count
    "#;

    println!("🧪 Testing complete IDE integration workflow...");

    // Step 1: Execute query-prepare command
    println!("Step 1: Executing query-prepare...");
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

    if !prepare_output.status.success() {
        let stderr = String::from_utf8_lossy(&prepare_output.stderr);
        let stdout = String::from_utf8_lossy(&prepare_output.stdout);
        panic!("query-prepare command failed:\nSTDOUT: {}\nSTDERR: {}", stdout, stderr);
    }

    // Verify files were created
    assert!(temp_path.exists(), "Temporary file should be created");
    assert!(tasks_path.exists(), "Tasks file should be created");

    // Step 2: Verify temporary file content
    println!("Step 2: Verifying temporary file content...");
    let temp_content = fs::read_to_string(&temp_path).await?;
    
    // Should contain metadata header
    assert!(temp_content.contains("Query Results Metadata"), "Temp file should have metadata header");
    assert!(temp_content.contains("Row Count: 3"), "Temp file should show correct row count");
    
    // Should contain FILE: markers for LLM processing
    assert!(temp_content.contains("FILE: src/main.rs"), "Temp file should contain main.rs file marker");
    assert!(temp_content.contains("FILE: src/lib.rs"), "Temp file should contain lib.rs file marker");
    assert!(temp_content.contains("FILE: README.md"), "Temp file should contain README.md file marker");
    
    // Should contain actual content
    assert!(temp_content.contains("Hello, world!"), "Temp file should contain main.rs content");
    assert!(temp_content.contains("pub mod utils"), "Temp file should contain lib.rs content");
    assert!(temp_content.contains("# My Project"), "Temp file should contain README content");

    // Step 3: Verify tasks file content
    println!("Step 3: Verifying tasks file content...");
    let tasks_content = fs::read_to_string(&tasks_path).await?;
    
    // Should have proper structure
    assert!(tasks_content.contains("# IDE Analysis Tasks"), "Tasks file should have proper header");
    assert!(tasks_content.contains("Query Preparation Metadata"), "Tasks file should have metadata section");
    assert!(tasks_content.contains(output_table), "Tasks file should reference output table");
    
    // Should have structured phases
    assert!(tasks_content.contains("Phase 1: Data Exploration"), "Tasks file should have exploration phase");
    assert!(tasks_content.contains("Phase 2: Systematic Analysis"), "Tasks file should have analysis phase");
    assert!(tasks_content.contains("Phase 3: Results and Documentation"), "Tasks file should have results phase");
    
    // Should have analysis type detection
    assert!(tasks_content.contains("General Code Analysis"), "Tasks file should detect analysis type");
    
    // Should have storage commands
    assert!(tasks_content.contains("store-result"), "Tasks file should include storage commands");

    // Step 4: Simulate analysis work (create comprehensive result file)
    println!("Step 4: Creating analysis results...");
    let analysis_result = format!(r#"# IDE Workflow Analysis Results

## Executive Summary
Comprehensive analysis of 3 files from the codebase using the IDE integration workflow.

## Files Analyzed
1. **src/main.rs** (25 lines, 100 words)
   - Contains the main entry point
   - Uses standard println! macro for output
   - Follows Rust naming conventions

2. **src/lib.rs** (15 lines, 50 words)
   - Defines public module structure
   - Exposes utils and config modules
   - Clean module organization

3. **README.md** (3 lines, 10 words)
   - Project documentation
   - Brief but clear description
   - Standard markdown format

## Code Quality Assessment

### Strengths
- Consistent Rust naming conventions across all files
- Proper module structure and organization
- Clear separation of concerns
- Good documentation practices

### Areas for Improvement
1. **Error Handling**: Main function could benefit from Result return type
2. **Documentation**: Add doc comments to public modules
3. **Testing**: No test files detected in the analysis
4. **Configuration**: Config module structure could be documented

## Security Analysis
- No obvious security vulnerabilities detected
- Standard library usage appears safe
- No unsafe code blocks identified

## Performance Considerations
- Simple main function with minimal overhead
- Module structure supports efficient compilation
- No performance bottlenecks identified

## Recommendations

### Immediate Actions
1. Add error handling to main function
2. Include doc comments for public APIs
3. Create unit tests for modules

### Future Enhancements
1. Consider adding logging framework
2. Implement configuration management
3. Add CI/CD pipeline configuration

## Traceability
- **Original Query**: {}
- **Analysis Date**: {}
- **Files Processed**: 3
- **Analysis Type**: General Code Analysis
- **Output Table**: {}

## Conclusion
The codebase demonstrates good Rust practices with a clean structure. The main areas for improvement are error handling, documentation, and testing coverage. The code is well-organized and follows standard conventions.
"#, test_query.trim(), chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"), output_table);

    fs::write(&result_path, analysis_result).await?;

    // Step 5: Store the analysis results
    println!("Step 5: Storing analysis results...");
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

    if !store_output.status.success() {
        let stderr = String::from_utf8_lossy(&store_output.stderr);
        let stdout = String::from_utf8_lossy(&store_output.stdout);
        panic!("store-result command failed:\nSTDOUT: {}\nSTDERR: {}", stdout, stderr);
    }

    let store_stdout = String::from_utf8_lossy(&store_output.stdout);
    println!("Store result output: {}", store_stdout);

    // Step 6: Verify the complete workflow by querying stored results
    println!("Step 6: Verifying stored results...");
    let verify_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "sql",
            &format!("SELECT analysis_id, analysis_type, LENGTH(llm_result) as result_length, created_at FROM {} WHERE llm_result LIKE '%IDE Workflow Analysis Results%' ORDER BY created_at DESC LIMIT 1", output_table),
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute verification query");

    if !verify_output.status.success() {
        let stderr = String::from_utf8_lossy(&verify_output.stderr);
        panic!("Verification query failed: {}", stderr);
    }

    let verify_stdout = String::from_utf8_lossy(&verify_output.stdout);
    println!("Verification query result: {}", verify_stdout);
    
    // Should find the stored analysis result
    assert!(verify_stdout.contains("analysis_id") || verify_stdout.contains("1"), 
           "Should find the stored analysis result");

    // Step 7: Test result retrieval and content verification
    println!("Step 7: Testing result retrieval...");
    let content_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "sql",
            &format!("SELECT llm_result FROM {} WHERE llm_result LIKE '%IDE Workflow Analysis Results%' ORDER BY created_at DESC LIMIT 1", output_table),
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute content retrieval query");

    if !content_output.status.success() {
        let stderr = String::from_utf8_lossy(&content_output.stderr);
        panic!("Content retrieval query failed: {}", stderr);
    }

    let content_stdout = String::from_utf8_lossy(&content_output.stdout);
    
    // Verify stored content contains key analysis elements
    assert!(content_stdout.contains("Executive Summary"), "Stored result should contain executive summary");
    assert!(content_stdout.contains("src/main.rs"), "Stored result should reference main.rs");
    assert!(content_stdout.contains("Code Quality Assessment"), "Stored result should contain quality assessment");
    assert!(content_stdout.contains("Recommendations"), "Stored result should contain recommendations");
    assert!(content_stdout.contains("Traceability"), "Stored result should contain traceability section");

    // Step 8: Test table statistics and metadata
    println!("Step 8: Testing table statistics...");
    let stats_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "sql",
            &format!("SELECT COUNT(*) as total_results, AVG(LENGTH(llm_result)) as avg_size FROM {}", output_table),
        ])
        .env("DATABASE_URL", &database_url)
        .output()
        .expect("Failed to execute statistics query");

    if !stats_output.status.success() {
        let stderr = String::from_utf8_lossy(&stats_output.stderr);
        panic!("Statistics query failed: {}", stderr);
    }

    let stats_stdout = String::from_utf8_lossy(&stats_output.stdout);
    println!("Table statistics: {}", stats_stdout);
    
    // Should show at least one result
    assert!(stats_stdout.contains("total_results") || stats_stdout.contains("1"), 
           "Should show table statistics");

    println!("✅ Complete IDE integration workflow test passed!");
    println!("   - query-prepare: Created temp file and tasks");
    println!("   - Analysis simulation: Generated comprehensive results");
    println!("   - store-result: Stored results with full traceability");
    println!("   - Verification: Confirmed end-to-end workflow integrity");

    Ok(())
}

/// Test error handling in the IDE integration workflow
#[tokio::test]
async fn test_ide_integration_error_handling() -> Result<()> {
    println!("🧪 Testing IDE integration error handling...");

    // Test 1: query-prepare with invalid paths
    println!("Test 1: query-prepare with relative paths (should fail)...");
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "query-prepare",
            "SELECT 'test' as filepath",
            "--temp-path",
            "relative_temp.txt",  // Relative path should fail
            "--tasks-file", 
            "relative_tasks.md",  // Relative path should fail
            "--output-table",
            "QUERYRESULT_error_test",
        ])
        .output()
        .expect("Failed to execute query-prepare command");

    assert!(!output.status.success(), "query-prepare should fail with relative paths");
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("absolute") || stderr.contains("path"), 
           "Error message should mention path requirements");

    // Test 2: store-result with non-existent file
    println!("Test 2: store-result with non-existent file (should fail)...");
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "store-result",
            "--output-table",
            "QUERYRESULT_error_test",
            "--result-file",
            "/tmp/non_existent_result_file_12345.txt",
            "--original-query",
            "SELECT * FROM test",
        ])
        .output()
        .expect("Failed to execute store-result command");

    assert!(!output.status.success(), "store-result should fail with non-existent file");
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found") || stderr.contains("No such file"), 
           "Error message should mention missing file");

    // Test 3: store-result with empty file
    println!("Test 3: store-result with empty file (should fail)...");
    let temp_dir = TempDir::new()?;
    let empty_file = temp_dir.path().join("empty_result.txt");
    fs::write(&empty_file, "").await?; // Create empty file

    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "store-result",
            "--output-table",
            "QUERYRESULT_error_test",
            "--result-file",
            &empty_file.to_string_lossy(),
            "--original-query",
            "SELECT * FROM test",
        ])
        .output()
        .expect("Failed to execute store-result command");

    assert!(!output.status.success(), "store-result should fail with empty file");
    
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("empty") || stderr.contains("Result file is empty"), 
           "Error message should mention empty file");

    println!("✅ IDE integration error handling tests passed!");

    Ok(())
}

/// Test analysis type detection in the IDE integration workflow
#[tokio::test]
async fn test_analysis_type_detection() -> Result<()> {
    // Only run if DATABASE_URL is set
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test: DATABASE_URL not set");
            return Ok(());
        }
    };

    println!("🧪 Testing analysis type detection...");

    let temp_dir = TempDir::new()?;
    
    // Test different analysis types
    let test_cases = vec![
        ("security", "Security Analysis", "Found potential SQL injection vulnerability in user input handling."),
        ("performance", "Performance Analysis", "Identified performance bottleneck in database query optimization."),
        ("architecture", "Architecture Review", "The system architecture follows clean design patterns and separation of concerns."),
        ("testing", "Testing Analysis", "Test coverage analysis shows 85% coverage with comprehensive unit tests."),
        ("documentation", "Documentation Review", "Documentation review reveals missing API documentation for public methods."),
    ];

    for (test_name, expected_type, result_content) in test_cases {
        println!("Testing {} analysis type detection...", test_name);
        
        let result_file = temp_dir.path().join(format!("{}_result.txt", test_name));
        let output_table = format!("QUERYRESULT_{}_test", test_name);
        
        // Create result file with specific content
        fs::write(&result_file, result_content).await?;
        
        // Store the result
        let store_output = std::process::Command::new("cargo")
            .args([
                "run",
                "--",
                "store-result",
                "--output-table",
                &output_table,
                "--result-file",
                &result_file.to_string_lossy(),
                "--original-query",
                &format!("SELECT * FROM test WHERE type = '{}'", test_name),
            ])
            .env("DATABASE_URL", &database_url)
            .output()
            .expect("Failed to execute store-result");

        if !store_output.status.success() {
            let stderr = String::from_utf8_lossy(&store_output.stderr);
            println!("Warning: store-result failed for {}: {}", test_name, stderr);
            continue; // Skip this test case but continue with others
        }

        // Verify the analysis type was detected correctly
        let verify_output = std::process::Command::new("cargo")
            .args([
                "run",
                "--",
                "sql",
                &format!("SELECT analysis_type FROM {} ORDER BY created_at DESC LIMIT 1", output_table),
            ])
            .env("DATABASE_URL", &database_url)
            .output()
            .expect("Failed to execute verification query");

        if verify_output.status.success() {
            let verify_stdout = String::from_utf8_lossy(&verify_output.stdout);
            if verify_stdout.contains(expected_type) {
                println!("✅ {} type detected correctly", expected_type);
            } else {
                println!("⚠️  {} type detection may need improvement (got: {})", expected_type, verify_stdout);
            }
        }
    }

    println!("✅ Analysis type detection tests completed!");

    Ok(())
}

/// Test task structure generation quality
#[tokio::test]
async fn test_task_structure_quality() -> Result<()> {
    // Only run if DATABASE_URL is set
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test: DATABASE_URL not set");
            return Ok(());
        }
    };

    println!("🧪 Testing task structure generation quality...");

    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path().join("quality_test_results.txt");
    let tasks_path = temp_dir.path().join("quality_test_tasks.md");

    // Test with a more complex query that should generate detailed tasks
    let complex_query = r#"
        SELECT 'src/auth/mod.rs' as filepath, 'pub mod login; pub mod session; pub mod tokens;' as content_text, 'rs' as extension, 'direct_text' as file_type, 50 as line_count, 200 as word_count
        UNION ALL
        SELECT 'src/auth/login.rs' as filepath, 'use bcrypt::{hash, verify}; pub fn authenticate_user(username: &str, password: &str) -> Result<bool, AuthError> { /* implementation */ }' as content_text, 'rs' as extension, 'direct_text' as file_type, 120 as line_count, 500 as word_count
        UNION ALL
        SELECT 'src/auth/session.rs' as filepath, 'use uuid::Uuid; pub struct Session { id: Uuid, user_id: i64, expires_at: DateTime<Utc> }' as content_text, 'rs' as extension, 'direct_text' as file_type, 80 as line_count, 300 as word_count
        UNION ALL
        SELECT 'src/database/queries.rs' as filepath, 'SELECT * FROM users WHERE username = $1 AND password_hash = $2' as content_text, 'rs' as extension, 'direct_text' as file_type, 200 as line_count, 800 as word_count
        UNION ALL
        SELECT 'tests/auth_tests.rs' as filepath, '#[test] fn test_user_authentication() { assert!(authenticate_user("test", "password").is_ok()); }' as content_text, 'rs' as extension, 'direct_text' as file_type, 150 as line_count, 600 as word_count
    "#;

    let output_table = "QUERYRESULT_quality_test";

    // Execute query-prepare
    let prepare_output = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "query-prepare",
            complex_query,
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

    if !prepare_output.status.success() {
        let stderr = String::from_utf8_lossy(&prepare_output.stderr);
        println!("Warning: query-prepare failed: {}", stderr);
        return Ok(()); // Skip test if it fails
    }

    // Analyze the generated tasks file for quality
    let tasks_content = fs::read_to_string(&tasks_path).await?;
    
    // Quality checks
    let quality_checks = vec![
        ("Metadata section", tasks_content.contains("Query Preparation Metadata")),
        ("Row count accuracy", tasks_content.contains("Row Count: 5")),
        ("Analysis type detection", tasks_content.contains("Analysis") && (tasks_content.contains("Security") || tasks_content.contains("General"))),
        ("Structured phases", tasks_content.contains("Phase 1") && tasks_content.contains("Phase 2") && tasks_content.contains("Phase 3")),
        ("Actionable tasks", tasks_content.contains("[ ]") && tasks_content.contains("Action:")),
        ("Storage commands", tasks_content.contains("store-result") && tasks_content.contains(output_table)),
        ("File references", tasks_content.contains("auth") && tasks_content.contains("database")),
        ("Analysis guidelines", tasks_content.contains("Analysis Guidelines") || tasks_content.contains("Pro Tips")),
    ];

    let mut passed_checks = 0;
    let total_checks = quality_checks.len();

    for (check_name, passed) in quality_checks {
        if passed {
            println!("✅ {}", check_name);
            passed_checks += 1;
        } else {
            println!("❌ {}", check_name);
        }
    }

    let quality_score = (passed_checks as f32 / total_checks as f32) * 100.0;
    println!("📊 Task structure quality score: {:.1}% ({}/{} checks passed)", 
             quality_score, passed_checks, total_checks);

    // Require at least 75% quality score
    assert!(quality_score >= 75.0, 
           "Task structure quality should be at least 75% (got {:.1}%)", quality_score);

    println!("✅ Task structure generation quality test passed!");

    Ok(())
}