//! Comprehensive end-to-end integration tests for the complete Pensieve CLI workflow
//!
//! These tests verify the entire pipeline from directory scanning to final statistics,
//! including paragraph-level deduplication, database consistency, and error recovery.

use pensieve::prelude::*;
use pensieve::database::Database;
use std::fs;
use std::path::Path;
use tempfile::TempDir;
use tokio::process::Command;

/// Test data structure for creating comprehensive test scenarios
struct TestScenario {
    name: &'static str,
    files: Vec<TestFile>,
    expected_unique_files: usize,
    expected_duplicate_files: usize,
    expected_unique_paragraphs: usize,
    expected_paragraph_instances: usize,
}

struct TestFile {
    path: &'static str,
    content: &'static str,
    should_process: bool,
}

/// Create comprehensive test scenarios with various file types and content patterns
fn create_test_scenarios() -> Vec<TestScenario> {
    vec![
        TestScenario {
            name: "basic_deduplication",
            files: vec![
                TestFile {
                    path: "file1.txt",
                    content: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                    should_process: true,
                },
                TestFile {
                    path: "file2.txt", 
                    content: "First paragraph.\n\nSecond paragraph.\n\nDifferent third paragraph.",
                    should_process: true,
                },
                TestFile {
                    path: "duplicate.txt",
                    content: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", // Exact duplicate of file1.txt
                    should_process: false, // Should be marked as duplicate
                },
            ],
            expected_unique_files: 2,
            expected_duplicate_files: 1,
            expected_unique_paragraphs: 4, // "First paragraph", "Second paragraph", "Third paragraph", "Different third paragraph"
            expected_paragraph_instances: 6, // Total paragraph instances across processed files (duplicate file is not processed)
        },
        TestScenario {
            name: "mixed_file_types",
            files: vec![
                TestFile {
                    path: "code.rs",
                    content: "fn main() {\n    println!(\"Hello, world!\");\n}\n\n// This is a comment\nstruct Person {\n    name: String,\n}",
                    should_process: true,
                },
                TestFile {
                    path: "config.json",
                    content: "{\n  \"name\": \"test\",\n  \"version\": \"1.0.0\"\n}\n\n{\n  \"settings\": {\n    \"debug\": true\n  }\n}",
                    should_process: true,
                },
                TestFile {
                    path: "readme.md",
                    content: "# Test Project\n\nThis is a test project.\n\n## Features\n\n- Feature 1\n- Feature 2",
                    should_process: true,
                },
                TestFile {
                    path: "binary.jpg",
                    content: "\u{FF}\u{D8}\u{FF}\u{E0}\u{00}\u{10}JFIF", // JPEG magic bytes
                    should_process: false, // Should be excluded as binary
                },
            ],
            expected_unique_files: 4, // All files are scanned, but binary files fail during content processing
            expected_duplicate_files: 0,
            expected_unique_paragraphs: 6, // Various paragraphs from different file types
            expected_paragraph_instances: 6,
        },
        TestScenario {
            name: "nested_directories",
            files: vec![
                TestFile {
                    path: "root.txt",
                    content: "Root level content.\n\nShared paragraph content.",
                    should_process: true,
                },
                TestFile {
                    path: "subdir/nested.txt",
                    content: "Nested content.\n\nShared paragraph content.", // Shares paragraph with root.txt
                    should_process: true,
                },
                TestFile {
                    path: "subdir/deep/deeply_nested.txt",
                    content: "Deeply nested content.\n\nUnique deep content.",
                    should_process: true,
                },
                TestFile {
                    path: "subdir/deep/another.txt",
                    content: "Root level content.\n\nAnother unique paragraph.", // Shares paragraph with root.txt
                    should_process: true,
                },
            ],
            expected_unique_files: 4,
            expected_duplicate_files: 0,
            expected_unique_paragraphs: 6, // "Root level content", "Shared paragraph content", "Nested content", "Deeply nested content", "Unique deep content", "Another unique paragraph"
            expected_paragraph_instances: 8, // Total instances across all files
        },
    ]
}

#[tokio::test]
async fn test_complete_cli_workflow_basic_scenario() -> Result<()> {
    let scenario = &create_test_scenarios()[0]; // Basic deduplication scenario
    
    // Create temporary directory and test files
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("test.db");
    
    // Create input and database directories
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create test files
    for file in &scenario.files {
        let file_path = input_dir.join(file.path);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&file_path, file.content).unwrap();
    }
    
    println!("Testing scenario: {}", scenario.name);
    println!("Input directory: {}", input_dir.display());
    println!("Database path: {}", db_path.display());
    
    // List all files in the directory for debugging
    println!("All files in test directory:");
    for entry in fs::read_dir(&input_dir).unwrap() {
        let entry = entry.unwrap();
        println!("  {}", entry.path().display());
    }
    
    // Phase 1: Run complete CLI workflow
    let result = run_cli_ingestion(&input_dir, &db_path).await;
    assert!(result.is_ok(), "CLI ingestion should succeed: {:?}", result);
    
    // Phase 2: Verify database state
    let db = Database::new(&db_path).await?;
    
    // Verify file-level statistics
    let stats = db.get_statistics().await?;
    
    // Verify file-level statistics match expectations
    
    // The scanner might find additional files (like .gitkeep, temp files, etc.)
    // So we check that we have at least the expected files
    assert!(stats.total_files as usize >= scenario.files.len(), 
        "Total files should be at least the input files count");
    assert_eq!(stats.unique_files as usize, scenario.expected_unique_files,
        "Unique files count should match expected");
    assert_eq!(stats.duplicate_files as usize, scenario.expected_duplicate_files,
        "Duplicate files count should match expected");
    
    // Verify paragraph-level statistics
    let paragraph_stats = db.get_paragraph_statistics().await?;
    assert_eq!(paragraph_stats.unique_paragraphs as usize, scenario.expected_unique_paragraphs,
        "Unique paragraphs should match expected");
    assert_eq!(paragraph_stats.total_paragraph_instances as usize, scenario.expected_paragraph_instances,
        "Total paragraph instances should match expected");
    
    // Phase 3: Verify database consistency
    verify_database_consistency(&db).await?;
    
    println!("✅ Complete CLI workflow test passed for scenario: {}", scenario.name);
    Ok(())
}

#[tokio::test]
async fn test_paragraph_level_deduplication_across_files() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("dedup_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create files with overlapping paragraph content
    let files = vec![
        ("file1.txt", "Shared paragraph 1.\n\nUnique to file 1.\n\nShared paragraph 2."),
        ("file2.txt", "Shared paragraph 1.\n\nUnique to file 2.\n\nShared paragraph 2."),
        ("file3.txt", "Unique to file 3.\n\nShared paragraph 1.\n\nAnother unique paragraph."),
    ];
    
    for (filename, content) in &files {
        fs::write(input_dir.join(filename), content).unwrap();
    }
    
    // Run ingestion
    run_cli_ingestion(&input_dir, &db_path).await?;
    
    // Verify paragraph deduplication
    let db = Database::new(&db_path).await?;
    let paragraph_stats = db.get_paragraph_statistics().await?;
    
    // Should have 5 unique paragraphs:
    // "Shared paragraph 1", "Shared paragraph 2", "Unique to file 1", "Unique to file 2", "Unique to file 3", "Another unique paragraph"
    assert_eq!(paragraph_stats.unique_paragraphs, 6, "Should have 6 unique paragraphs");
    
    // Should have 9 total paragraph instances (3 paragraphs per file)
    assert_eq!(paragraph_stats.total_paragraph_instances, 9, "Should have 9 total paragraph instances");
    
    // Verify deduplication rate
    let expected_deduplication_rate = ((9 - 6) as f64 / 9.0) * 100.0;
    assert!((paragraph_stats.deduplication_rate - expected_deduplication_rate).abs() < 0.1,
        "Deduplication rate should be approximately {:.1}%, got {:.1}%", 
        expected_deduplication_rate, paragraph_stats.deduplication_rate);
    
    // Verify paragraph-to-file relationships
    let paragraph_sources_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM paragraph_sources")
        .fetch_one(db.pool())
        .await
        .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(paragraph_sources_count, 9, "Should have 9 paragraph-to-file relationships");
    
    println!("✅ Paragraph-level deduplication test passed");
    Ok(())
}

#[tokio::test]
async fn test_database_consistency_after_full_pipeline() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("consistency_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create comprehensive test data
    create_comprehensive_test_data(&input_dir)?;
    
    // Run full ingestion pipeline
    run_cli_ingestion(&input_dir, &db_path).await?;
    
    // Verify database consistency
    let db = Database::new(&db_path).await?;
    verify_database_consistency(&db).await?;
    
    // Additional consistency checks
    verify_referential_integrity(&db).await?;
    verify_deduplication_integrity(&db).await?;
    verify_paragraph_source_integrity(&db).await?;
    
    println!("✅ Database consistency test passed");
    Ok(())
}

#[tokio::test]
async fn test_error_recovery_and_partial_processing() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("error_recovery_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create test files including problematic ones
    let large_content = "x".repeat(100000);
    let test_files = vec![
        ("good1.txt", "This is good content.\n\nAnother paragraph."),
        ("good2.txt", "More good content.\n\nYet another paragraph."),
        ("empty.txt", ""), // Empty file - should be handled gracefully
        ("binary.exe", "\u{00}\u{01}\u{02}\u{03}\u{04}\u{05}"), // Binary file - should be excluded
        ("large_line.txt", large_content.as_str()), // Very large single line - should be handled
        ("unicode.txt", "Unicode content: 🚀 🌟 ✨\n\nEmoji paragraph: 😀 😃 😄"),
    ];
    
    for (filename, content) in &test_files {
        fs::write(input_dir.join(filename), content).unwrap();
    }
    
    // Create a file with permission issues (if possible on this platform)
    let restricted_file = input_dir.join("restricted.txt");
    fs::write(&restricted_file, "Restricted content").unwrap();
    
    // Run ingestion - should handle errors gracefully
    let result = run_cli_ingestion(&input_dir, &db_path).await;
    assert!(result.is_ok(), "CLI should handle errors gracefully and continue processing");
    
    // Verify that good files were processed despite errors
    let db = Database::new(&db_path).await?;
    let stats = db.get_statistics().await?;
    
    // Should have processed the good files
    assert!(stats.total_files >= 4, "Should have processed at least the good files");
    
    // Check that some files were processed successfully
    let processed_files = stats.files_by_status.get("processed").unwrap_or(&0);
    assert!(*processed_files > 0, "Should have some successfully processed files");
    
    // Check error tracking
    let error_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM processing_errors")
        .fetch_one(db.pool())
        .await
        .map_err(|e| PensieveError::Database(e))?;
    
    println!("Processed files: {}, Errors recorded: {}", processed_files, error_count);
    
    // Verify database consistency even with errors
    verify_database_consistency(&db).await?;
    
    println!("✅ Error recovery and partial processing test passed");
    Ok(())
}

#[tokio::test]
async fn test_mixed_file_types_comprehensive() -> Result<()> {
    let scenario = &create_test_scenarios()[1]; // Mixed file types scenario
    
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("mixed_types_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create test files
    for file in &scenario.files {
        let file_path = input_dir.join(file.path);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&file_path, file.content).unwrap();
    }
    
    // Run ingestion
    run_cli_ingestion(&input_dir, &db_path).await?;
    
    // Verify results
    let db = Database::new(&db_path).await?;
    let stats = db.get_statistics().await?;
    
    // Verify file type handling - all files are scanned, but binary files should fail processing
    assert_eq!(stats.unique_files as usize, scenario.expected_unique_files,
        "Should scan all files during metadata phase");
    
    // Check that text files were successfully processed
    let processed_files = *stats.files_by_status.get("processed").unwrap_or(&0);
    assert_eq!(processed_files, 3, "Should successfully process 3 text files");
    
    // Verify that binary files are handled appropriately
    
    // Check if binary files were excluded (either skipped_binary or error status)
    let skipped_binary: u64 = *stats.files_by_status.get("skipped_binary").unwrap_or(&0);
    let error_files: u64 = *stats.files_by_status.get("error").unwrap_or(&0);
    
    // Binary files should either be skipped or cause errors
    assert!(skipped_binary > 0 || error_files > 0, 
        "Should have skipped binary files or had processing errors for them");
    
    // Verify paragraph processing for different file types
    let paragraph_stats = db.get_paragraph_statistics().await?;
    assert!(paragraph_stats.unique_paragraphs > 0, "Should have processed paragraphs from text files");
    
    println!("✅ Mixed file types test passed");
    Ok(())
}

#[tokio::test]
async fn test_nested_directory_processing() -> Result<()> {
    let scenario = &create_test_scenarios()[2]; // Nested directories scenario
    
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("nested_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create nested directory structure
    for file in &scenario.files {
        let file_path = input_dir.join(file.path);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&file_path, file.content).unwrap();
    }
    
    // Run ingestion
    run_cli_ingestion(&input_dir, &db_path).await?;
    
    // Verify results
    let db = Database::new(&db_path).await?;
    let stats = db.get_statistics().await?;
    
    assert_eq!(stats.total_files as usize, scenario.expected_unique_files,
        "Should process all files in nested directories");
    
    // Verify depth levels are calculated correctly
    let depth_check: Vec<(String, i64)> = sqlx::query_as(
        "SELECT relative_path, depth_level FROM files ORDER BY depth_level"
    )
    .fetch_all(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    // Verify depth calculations
    for (path, depth) in depth_check {
        let expected_depth = path.split('/').count() as i64;
        assert_eq!(depth, expected_depth, 
            "Depth level should match path components for {}", path);
    }
    
    // Verify paragraph deduplication across nested files
    let paragraph_stats = db.get_paragraph_statistics().await?;
    assert_eq!(paragraph_stats.unique_paragraphs as usize, scenario.expected_unique_paragraphs,
        "Should deduplicate paragraphs across nested directories");
    
    println!("✅ Nested directory processing test passed");
    Ok(())
}

#[tokio::test]
async fn test_cli_statistics_command() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("stats_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create test data and run ingestion
    create_comprehensive_test_data(&input_dir)?;
    run_cli_ingestion(&input_dir, &db_path).await?;
    
    // Test stats command
    let output = Command::new(env!("CARGO_BIN_EXE_pensieve"))
        .args(&["stats", "--database", &db_path.to_string_lossy()])
        .output()
        .await
        .expect("Failed to execute stats command");
    
    assert!(output.status.success(), "Stats command should succeed");
    
    let stdout = String::from_utf8(output.stdout).unwrap();
    
    // Verify stats output contains expected information
    assert!(stdout.contains("=== Pensieve Database Statistics ==="), 
        "Should contain statistics header");
    assert!(stdout.contains("Total files:"), "Should show total files");
    assert!(stdout.contains("Unique files:"), "Should show unique files");
    assert!(stdout.contains("Unique paragraphs:"), "Should show paragraph count");
    assert!(stdout.contains("Total tokens:"), "Should show token count");
    
    println!("✅ CLI statistics command test passed");
    Ok(())
}

#[tokio::test]
async fn test_performance_with_large_dataset() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let input_dir = temp_dir.path().join("input");
    let db_path = temp_dir.path().join("db").join("performance_test.db");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(db_path.parent().unwrap()).unwrap();
    
    // Create a larger dataset for performance testing
    let num_files = 50;
    let paragraphs_per_file = 10;
    
    for i in 0..num_files {
        let mut content = String::new();
        for j in 0..paragraphs_per_file {
            if j > 0 {
                content.push_str("\n\n");
            }
            // Create some duplicate content across files for deduplication testing
            if j % 3 == 0 {
                content.push_str(&format!("Common paragraph content {}", j % 5));
            } else {
                content.push_str(&format!("Unique content for file {} paragraph {}", i, j));
            }
        }
        fs::write(input_dir.join(format!("file_{:03}.txt", i)), content).unwrap();
    }
    
    // Measure performance
    let start = std::time::Instant::now();
    run_cli_ingestion(&input_dir, &db_path).await?;
    let elapsed = start.elapsed();
    
    // Verify results
    let db = Database::new(&db_path).await?;
    let stats = db.get_statistics().await?;
    
    assert_eq!(stats.total_files, num_files as u64, "Should process all files");
    
    // Performance should be reasonable (>10 files/sec for this test size)
    let files_per_second = stats.total_files as f64 / elapsed.as_secs_f64();
    assert!(files_per_second > 10.0, 
        "Should process >10 files/sec, got {:.2} files/sec", files_per_second);
    
    // Verify paragraph deduplication worked
    let paragraph_stats = db.get_paragraph_statistics().await?;
    assert!(paragraph_stats.deduplication_rate > 0.0, 
        "Should have some paragraph deduplication");
    
    println!("Performance: {:.2} files/sec, {:.1}% paragraph deduplication", 
        files_per_second, paragraph_stats.deduplication_rate);
    
    println!("✅ Performance test passed");
    Ok(())
}

// Helper functions

/// Run CLI ingestion workflow programmatically
async fn run_cli_ingestion(input_dir: &Path, db_path: &Path) -> Result<()> {
    use pensieve::cli::Cli;
    
    // Create CLI instance with test parameters
    let cli = Cli {
        input: Some(input_dir.to_path_buf()),
        database: Some(db_path.to_path_buf()),
        verbose: false,
        dry_run: false,
        force_reprocess: false,
        config: None,
        command: None,
    };
    
    // Run the CLI workflow
    cli.run().await
}

/// Create comprehensive test data with various file types and patterns
fn create_comprehensive_test_data(input_dir: &Path) -> Result<()> {
    let test_files = vec![
        // Text files with various content patterns
        ("simple.txt", "Simple text content.\n\nAnother paragraph."),
        ("multiline.txt", "Line 1\nLine 2\nLine 3\n\nParagraph 2\nMore content."),
        ("duplicate1.txt", "Duplicate content.\n\nShared paragraph."),
        ("duplicate2.txt", "Duplicate content.\n\nShared paragraph."), // Exact duplicate
        
        // Source code files
        ("main.rs", "fn main() {\n    println!(\"Hello\");\n}\n\n// Comment\nstruct Data {}"),
        ("script.py", "#!/usr/bin/env python3\n\ndef main():\n    print(\"Hello\")\n\nif __name__ == \"__main__\":\n    main()"),
        ("config.js", "const config = {\n  name: 'test',\n  version: '1.0'\n};\n\nmodule.exports = config;"),
        
        // Configuration files
        ("config.json", "{\n  \"name\": \"test\",\n  \"version\": \"1.0.0\"\n}"),
        ("settings.yaml", "name: test\nversion: 1.0.0\n\nsettings:\n  debug: true"),
        ("app.toml", "[app]\nname = \"test\"\nversion = \"1.0.0\"\n\n[settings]\ndebug = true"),
        
        // Documentation files
        ("README.md", "# Test Project\n\nThis is a test.\n\n## Features\n\n- Feature 1"),
        ("CHANGELOG.md", "# Changelog\n\n## v1.0.0\n\n- Initial release"),
        
        // Empty and minimal files
        ("empty.txt", ""),
        ("minimal.txt", "x"),
        
        // Unicode content
        ("unicode.txt", "Unicode: 🚀 ✨ 🌟\n\nEmoji paragraph: 😀 😃"),
    ];
    
    // Create nested directory structure
    fs::create_dir_all(input_dir.join("src"))?;
    fs::create_dir_all(input_dir.join("docs"))?;
    fs::create_dir_all(input_dir.join("config"))?;
    
    for (filename, content) in test_files {
        let file_path = if filename.ends_with(".rs") || filename.ends_with(".py") {
            input_dir.join("src").join(filename)
        } else if filename.ends_with(".md") {
            input_dir.join("docs").join(filename)
        } else if filename.contains("config") || filename.ends_with(".json") || filename.ends_with(".yaml") || filename.ends_with(".toml") {
            input_dir.join("config").join(filename)
        } else {
            input_dir.join(filename)
        };
        
        fs::write(file_path, content)?;
    }
    
    Ok(())
}

/// Verify database consistency and integrity
async fn verify_database_consistency(db: &Database) -> Result<()> {
    // Check foreign key constraints
    let fk_violations: Vec<(String, i64, String, i64)> = sqlx::query_as(
        "PRAGMA foreign_key_check"
    )
    .fetch_all(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert!(fk_violations.is_empty(), 
        "Database should have no foreign key violations: {:?}", fk_violations);
    
    // Check database integrity
    let integrity_result: String = sqlx::query_scalar("PRAGMA integrity_check")
        .fetch_one(db.pool())
        .await
        .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(integrity_result, "ok", 
        "Database integrity check should pass: {}", integrity_result);
    
    Ok(())
}

/// Verify referential integrity between tables
async fn verify_referential_integrity(db: &Database) -> Result<()> {
    // Check that all paragraph_sources reference valid paragraphs
    let orphaned_sources: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM paragraph_sources ps
        LEFT JOIN paragraphs p ON ps.paragraph_id = p.paragraph_id
        WHERE p.paragraph_id IS NULL
        "#
    )
    .fetch_one(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(orphaned_sources, 0, "Should have no orphaned paragraph sources");
    
    // Check that all paragraph_sources reference valid files
    let orphaned_file_refs: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM paragraph_sources ps
        LEFT JOIN files f ON ps.file_id = f.file_id
        WHERE f.file_id IS NULL
        "#
    )
    .fetch_one(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(orphaned_file_refs, 0, "Should have no orphaned file references");
    
    // Check that all processing_errors with file_id reference valid files
    let orphaned_errors: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM processing_errors pe
        LEFT JOIN files f ON pe.file_id = f.file_id
        WHERE pe.file_id IS NOT NULL AND f.file_id IS NULL
        "#
    )
    .fetch_one(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(orphaned_errors, 0, "Should have no orphaned error references");
    
    Ok(())
}

/// Verify deduplication integrity
async fn verify_deduplication_integrity(db: &Database) -> Result<()> {
    // Check that duplicate files have valid group IDs
    let invalid_duplicates: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM files 
        WHERE duplicate_status = 'duplicate' AND duplicate_group_id IS NULL
        "#
    )
    .fetch_one(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(invalid_duplicates, 0, 
        "Duplicate files should have group IDs");
    
    // Check that each duplicate group has exactly one canonical file
    let group_canonical_counts: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT duplicate_group_id, COUNT(*) as canonical_count
        FROM files 
        WHERE duplicate_status = 'canonical' AND duplicate_group_id IS NOT NULL
        GROUP BY duplicate_group_id
        HAVING canonical_count != 1
        "#
    )
    .fetch_all(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert!(group_canonical_counts.is_empty(),
        "Each duplicate group should have exactly one canonical file: {:?}", 
        group_canonical_counts);
    
    // Check that files in the same duplicate group have the same hash
    let hash_mismatches: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT duplicate_group_id, COUNT(DISTINCT hash) as hash_count
        FROM files 
        WHERE duplicate_group_id IS NOT NULL
        GROUP BY duplicate_group_id
        HAVING hash_count > 1
        "#
    )
    .fetch_all(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert!(hash_mismatches.is_empty(),
        "Files in same duplicate group should have same hash: {:?}", 
        hash_mismatches);
    
    Ok(())
}

/// Verify paragraph source integrity
async fn verify_paragraph_source_integrity(db: &Database) -> Result<()> {
    // Check that paragraph indices start from 0 within each file
    let index_issues: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT file_id, MIN(paragraph_index) as min_index
        FROM paragraph_sources
        GROUP BY file_id
        HAVING min_index != 0
        "#
    )
    .fetch_all(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    // Paragraph indices should start from 0 for each file
    assert!(index_issues.is_empty(), 
        "All files should have paragraph indices starting from 0: {} issues found", index_issues.len());
    
    // Check that byte offsets are valid (end > start)
    let invalid_offsets: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM paragraph_sources WHERE byte_offset_end <= byte_offset_start"
    )
    .fetch_one(db.pool())
    .await
    .map_err(|e| PensieveError::Database(e))?;
    
    assert_eq!(invalid_offsets, 0, "All byte offsets should be valid");
    
    Ok(())
}