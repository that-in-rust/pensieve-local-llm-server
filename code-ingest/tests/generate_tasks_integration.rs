use code_ingest::tasks::{TaskDivider, TaskConfig, MarkdownGenerator, TaskMetadata, TaskStructure, QueryResultRow};

use tempfile::TempDir;

/// Integration test for the complete generate-tasks workflow
/// This test simulates the entire process that the CLI command would execute
#[tokio::test]
async fn test_generate_tasks_end_to_end() {
    let temp_dir = TempDir::new().unwrap();
    let tasks_file = temp_dir.path().join("generated_tasks.md");
    let prompt_file = temp_dir.path().join("analysis_prompt.md");
    
    // Create a mock prompt file
    tokio::fs::write(&prompt_file, "# Analysis Prompt\n\nAnalyze the provided code files for patterns and structure.").await.unwrap();
    
    // Simulate the workflow that execute_generate_tasks would perform
    
    // Step 1: Simulate query results from database
    let query_rows = vec![
        QueryResultRow {
            file_id: Some(1),
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(150),
            word_count: Some(500),
            token_count: Some(750),
            content_text: Some("fn main() { println!(\"Hello, world!\"); }".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/main.rs".to_string()),
            absolute_path: Some("/project/src/main.rs".to_string()),
        },
        QueryResultRow {
            file_id: Some(2),
            filepath: Some("src/lib.rs".to_string()),
            filename: Some("lib.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(2048),
            line_count: Some(300),
            word_count: Some(1000),
            token_count: Some(1500),
            content_text: Some("pub mod utils;\npub mod config;".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/lib.rs".to_string()),
            absolute_path: Some("/project/src/lib.rs".to_string()),
        },
        QueryResultRow {
            file_id: Some(3),
            filepath: Some("src/large_module.rs".to_string()),
            filename: Some("large_module.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(10240),
            line_count: Some(1000), // Large file that will be chunked
            word_count: Some(5000),
            token_count: Some(7500),
            content_text: Some("// Large module with many functions...".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/large_module.rs".to_string()),
            absolute_path: Some("/project/src/large_module.rs".to_string()),
        },
    ];
    
    // Step 2: Create task configuration
    let config = TaskConfig {
        sql_query: "SELECT * FROM INGEST_20250927143022 WHERE extension = 'rs'".to_string(),
        output_table: "QUERYRESULT_rust_analysis".to_string(),
        tasks_file: tasks_file.to_string_lossy().to_string(),
        group_count: 7,
        chunk_size: Some(300), // Will chunk the large file
        chunk_overlap: Some(50),
        prompt_file: Some(prompt_file.to_string_lossy().to_string()),
    };
    
    // Step 3: Create task divider and generate tasks
    let divider = TaskDivider::new(config.group_count).unwrap();
    let tasks = divider.create_tasks_from_query_results(&query_rows, &config).unwrap();
    
    // Should have more than 3 tasks due to chunking of the large file
    assert!(tasks.len() > 3);
    
    // Verify chunking worked for the large file
    let chunked_tasks: Vec<_> = tasks.iter().filter(|t| t.chunk_info.is_some()).collect();
    assert!(!chunked_tasks.is_empty(), "Large file should have been chunked");
    
    // Step 4: Adjust group count and divide tasks
    let actual_group_count = std::cmp::min(config.group_count, tasks.len());
    let adjusted_divider = TaskDivider::new(actual_group_count).unwrap();
    let groups = adjusted_divider.divide_into_groups(tasks.clone()).unwrap();
    
    // Step 5: Create task structure with metadata
    let metadata = TaskMetadata {
        total_tasks: tasks.len(),
        group_count: actual_group_count,
        sql_query: config.sql_query.clone(),
        output_table: config.output_table.clone(),
        generated_at: chrono::Utc::now(),
        prompt_file: config.prompt_file.clone(),
    };
    
    let task_structure = TaskStructure {
        groups,
        total_tasks: tasks.len(),
        metadata,
    };
    
    // Step 6: Generate and write markdown
    let generator = MarkdownGenerator::new();
    generator.write_to_file(&task_structure, &tasks_file.to_string_lossy()).await.unwrap();
    
    // Step 7: Verify the generated file
    assert!(tasks_file.exists());
    let content = tokio::fs::read_to_string(&tasks_file).await.unwrap();
    
    // Verify structure
    assert!(content.contains("# Implementation Tasks"));
    assert!(content.contains("## Task Generation Metadata"));
    assert!(content.contains("## Task Overview"));
    assert!(content.contains("## Processing Instructions"));
    
    // Verify metadata
    assert!(content.contains("QUERYRESULT_rust_analysis"));
    assert!(content.contains(&prompt_file.to_string_lossy().to_string()));
    assert!(content.contains("SELECT * FROM INGEST_20250927143022"));
    
    // Verify Kiro-compatible numbering
    assert!(content.contains("- [ ] 1."));
    assert!(content.contains("- [ ] 1.1"));
    
    // Verify file information
    assert!(content.contains("src/main.rs"));
    assert!(content.contains("src/lib.rs"));
    assert!(content.contains("src/large_module.rs"));
    
    // Verify chunking information is present
    assert!(content.contains("**Chunk**:"));
    
    // Verify task details
    assert!(content.contains("**File**:"));
    assert!(content.contains("**Type**: rs"));
    assert!(content.contains("**Lines**:"));
    assert!(content.contains("**Task ID**:"));
    
    // Verify processing instructions
    assert!(content.contains("code-ingest query-prepare"));
    assert!(content.contains("code-ingest store-result"));
    
    println!("✅ End-to-end generate-tasks test completed successfully");
    println!("   Generated {} tasks in {} groups", tasks.len(), actual_group_count);
    println!("   Tasks file: {} ({} bytes)", tasks_file.display(), content.len());
}

#[test]
fn test_task_generation_with_various_file_types() {
    // Test task generation with different file types and sizes
    let divider = TaskDivider::new(7).unwrap();
    
    let query_rows = vec![
        // Small text file
        QueryResultRow {
            file_id: Some(1),
            filepath: Some("README.md".to_string()),
            filename: Some("README.md".to_string()),
            extension: Some("md".to_string()),
            file_size_bytes: Some(512),
            line_count: Some(50),
            word_count: Some(200),
            token_count: Some(300),
            content_text: Some("# Project README".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("README.md".to_string()),
            absolute_path: Some("/project/README.md".to_string()),
        },
        // Convertible file
        QueryResultRow {
            file_id: Some(2),
            filepath: Some("docs/manual.pdf".to_string()),
            filename: Some("manual.pdf".to_string()),
            extension: Some("pdf".to_string()),
            file_size_bytes: Some(1048576),
            line_count: Some(500),
            word_count: Some(2000),
            token_count: Some(3000),
            content_text: Some("Converted PDF content...".to_string()),
            file_type: Some("convertible".to_string()),
            conversion_command: Some("pdftotext".to_string()),
            relative_path: Some("docs/manual.pdf".to_string()),
            absolute_path: Some("/project/docs/manual.pdf".to_string()),
        },
        // Non-text file
        QueryResultRow {
            file_id: Some(3),
            filepath: Some("assets/logo.png".to_string()),
            filename: Some("logo.png".to_string()),
            extension: Some("png".to_string()),
            file_size_bytes: Some(204800),
            line_count: None,
            word_count: None,
            token_count: None,
            content_text: None,
            file_type: Some("non_text".to_string()),
            conversion_command: None,
            relative_path: Some("assets/logo.png".to_string()),
            absolute_path: Some("/project/assets/logo.png".to_string()),
        },
    ];
    
    let config = TaskConfig {
        sql_query: "SELECT * FROM INGEST_20250927143022".to_string(),
        output_table: "QUERYRESULT_mixed_analysis".to_string(),
        tasks_file: "/tmp/mixed_tasks.md".to_string(),
        group_count: 7,
        chunk_size: None, // No chunking for this test
        chunk_overlap: None,
        prompt_file: Some("mixed_analysis.md".to_string()),
    };
    
    let tasks = divider.create_tasks_from_query_results(&query_rows, &config).unwrap();
    
    // Should have 3 tasks (one for each file)
    assert_eq!(tasks.len(), 3);
    
    // Verify task descriptions
    assert!(tasks.iter().any(|t| t.description.contains("README.md")));
    assert!(tasks.iter().any(|t| t.description.contains("manual.pdf")));
    assert!(tasks.iter().any(|t| t.description.contains("logo.png")));
    
    // Verify metadata includes file types
    let readme_task = tasks.iter().find(|t| t.file_path.as_ref().unwrap().contains("README.md")).unwrap();
    assert_eq!(readme_task.metadata.get("file_type").unwrap(), "direct_text");
    
    let pdf_task = tasks.iter().find(|t| t.file_path.as_ref().unwrap().contains("manual.pdf")).unwrap();
    assert_eq!(pdf_task.metadata.get("file_type").unwrap(), "convertible");
    
    let png_task = tasks.iter().find(|t| t.file_path.as_ref().unwrap().contains("logo.png")).unwrap();
    assert_eq!(png_task.metadata.get("file_type").unwrap(), "non_text");
}

#[test]
fn test_task_generation_error_handling() {
    let divider = TaskDivider::new(7).unwrap();
    
    // Test with empty query results
    let empty_results: Vec<QueryResultRow> = vec![];
    let config = TaskConfig::default();
    
    let tasks = divider.create_tasks_from_query_results(&empty_results, &config).unwrap();
    assert!(tasks.is_empty());
    
    // Test with invalid chunk configuration
    let query_rows = vec![
        QueryResultRow {
            file_id: Some(1),
            filepath: Some("test.rs".to_string()),
            filename: Some("test.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(1000),
            word_count: Some(5000),
            token_count: Some(7500),
            content_text: Some("test content".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("test.rs".to_string()),
            absolute_path: Some("/project/test.rs".to_string()),
        },
    ];
    
    let invalid_config = TaskConfig {
        sql_query: "SELECT * FROM test".to_string(),
        output_table: "QUERYRESULT_test".to_string(),
        tasks_file: "/tmp/test.md".to_string(),
        group_count: 7,
        chunk_size: Some(0), // Invalid chunk size
        chunk_overlap: None,
        prompt_file: None,
    };
    
    // Should handle invalid chunk size gracefully
    let result = divider.create_tasks_from_query_results(&query_rows, &invalid_config);
    assert!(result.is_err());
}