use code_ingest::tasks::{TaskDivider, Task, TaskConfig, QueryResultRow, MarkdownGenerator, TaskStructure, TaskGroup, TaskMetadata, ChunkInfo};
use std::collections::HashMap;

#[test]
fn test_task_division_basic() {
    let divider = TaskDivider::new(7).unwrap();
    
    let tasks: Vec<Task> = (1..=35)
        .map(|i| Task {
            id: format!("{}", i),
            description: format!("Test task {}", i),
            file_path: Some(format!("test_file_{}.rs", i)),
            chunk_info: None,
            metadata: HashMap::new(),
        })
        .collect();

    let groups = divider.divide_into_groups(tasks).unwrap();

    assert_eq!(groups.len(), 7);
    for group in &groups {
        assert_eq!(group.tasks.len(), 5); // 35 ÷ 7 = 5 tasks per group
    }

    // Verify all tasks are included
    let total_tasks: usize = groups.iter().map(|g| g.tasks.len()).sum();
    assert_eq!(total_tasks, 35);
}

#[test]
fn test_task_creation_from_query_results() {
    let divider = TaskDivider::new(7).unwrap();
    
    let query_results = vec![
        QueryResultRow {
            file_id: Some(1),
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(100),
            word_count: Some(500),
            token_count: Some(750),
            content_text: Some("fn main() {}".to_string()),
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
            line_count: Some(200),
            word_count: Some(1000),
            token_count: Some(1500),
            content_text: Some("pub mod test;".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/lib.rs".to_string()),
            absolute_path: Some("/project/src/lib.rs".to_string()),
        },
    ];
    
    let config = TaskConfig {
        sql_query: "SELECT * FROM test_table".to_string(),
        output_table: "QUERYRESULT_test".to_string(),
        tasks_file: "/tmp/tasks.md".to_string(),
        group_count: 7,
        chunk_size: None,
        chunk_overlap: None,
        prompt_file: Some("analyze.md".to_string()),
    };

    let tasks = divider.create_tasks_from_query_results(&query_results, &config).unwrap();

    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].description, "Analyze src/main.rs");
    assert_eq!(tasks[1].description, "Analyze src/lib.rs");
    
    // Check metadata
    assert_eq!(tasks[0].metadata.get("output_table").unwrap(), "QUERYRESULT_test");
    assert_eq!(tasks[0].metadata.get("prompt_file").unwrap(), "analyze.md");
}

#[test]
fn test_markdown_generation_integration() {
    let generator = MarkdownGenerator::new();
    
    // Create test task structure
    let tasks = vec![
        Task {
            id: "1".to_string(),
            description: "Analyze main.rs".to_string(),
            file_path: Some("src/main.rs".to_string()),
            chunk_info: None,
            metadata: {
                let mut map = HashMap::new();
                map.insert("extension".to_string(), "rs".to_string());
                map.insert("file_type".to_string(), "direct_text".to_string());
                map.insert("line_count".to_string(), "100".to_string());
                map
            },
        },
        Task {
            id: "2".to_string(),
            description: "Analyze lib.rs".to_string(),
            file_path: Some("src/lib.rs".to_string()),
            chunk_info: None,
            metadata: {
                let mut map = HashMap::new();
                map.insert("extension".to_string(), "rs".to_string());
                map.insert("file_type".to_string(), "direct_text".to_string());
                map.insert("line_count".to_string(), "200".to_string());
                map
            },
        },
    ];
    
    let divider = TaskDivider::new(2).unwrap();
    let groups = divider.divide_into_groups(tasks).unwrap();
    
    let task_structure = TaskStructure {
        groups,
        total_tasks: 2,
        metadata: TaskMetadata {
            total_tasks: 2,
            group_count: 2,
            sql_query: "SELECT * FROM test_table".to_string(),
            output_table: "QUERYRESULT_test".to_string(),
            generated_at: chrono::Utc::now(),
            prompt_file: Some("analyze.md".to_string()),
        },
    };
    
    let markdown = generator.generate_markdown(&task_structure).unwrap();
    
    // Check that markdown contains expected Kiro-compatible elements
    assert!(markdown.contains("# Implementation Tasks"));
    assert!(markdown.contains("- [ ] 1. Task Group 1"));
    assert!(markdown.contains("- [ ] 1.1 Analyze main.rs"));
    assert!(markdown.contains("- [ ] 2. Task Group 2"));
    assert!(markdown.contains("- [ ] 2.1 Analyze lib.rs"));
    assert!(markdown.contains("**File**: `src/main.rs`"));
    assert!(markdown.contains("**Task ID**: 1.1"));
    assert!(markdown.contains("**Task ID**: 2.1"));
}

#[tokio::test]
async fn test_complete_task_generation_workflow() {
    use tempfile::TempDir;
    
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("complete_tasks.md");
    
    // Create complete workflow: query results -> tasks -> groups -> markdown
    let divider = TaskDivider::new(7).unwrap();
    let generator = MarkdownGenerator::new();
    
    let query_results = vec![
        QueryResultRow {
            file_id: Some(1),
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(100),
            word_count: Some(500),
            token_count: Some(750),
            content_text: Some("fn main() {}".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/main.rs".to_string()),
            absolute_path: Some("/project/src/main.rs".to_string()),
        },
        QueryResultRow {
            file_id: Some(2),
            filepath: Some("src/large_file.rs".to_string()),
            filename: Some("large_file.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(10240),
            line_count: Some(1000), // Large file that will be chunked
            word_count: Some(5000),
            token_count: Some(7500),
            content_text: Some("// Large file content".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/large_file.rs".to_string()),
            absolute_path: Some("/project/src/large_file.rs".to_string()),
        },
    ];
    
    let config = TaskConfig {
        sql_query: "SELECT * FROM INGEST_20250927143022 WHERE extension = 'rs'".to_string(),
        output_table: "QUERYRESULT_rust_analysis".to_string(),
        tasks_file: file_path.to_str().unwrap().to_string(),
        group_count: 7,
        chunk_size: Some(300), // Will chunk the large file
        chunk_overlap: Some(50),
        prompt_file: Some("rust_analysis.md".to_string()),
    };

    // Create tasks from query results
    let tasks = divider.create_tasks_from_query_results(&query_results, &config).unwrap();
    
    // Should have more than 2 tasks due to chunking
    assert!(tasks.len() > 2);
    
    // Divide into groups (but we only have a few tasks, so adjust group count)
    let adjusted_divider = TaskDivider::new(std::cmp::min(7, tasks.len())).unwrap();
    let groups = adjusted_divider.divide_into_groups(tasks.clone()).unwrap();
    
    let task_structure = TaskStructure {
        groups,
        total_tasks: tasks.len(),
        metadata: TaskMetadata {
            total_tasks: tasks.len(),
            group_count: std::cmp::min(7, tasks.len()),
            sql_query: config.sql_query.clone(),
            output_table: config.output_table.clone(),
            generated_at: chrono::Utc::now(),
            prompt_file: config.prompt_file.clone(),
        },
    };
    
    // Generate and write markdown
    generator.write_to_file(&task_structure, file_path.to_str().unwrap()).await.unwrap();
    
    // Verify file was created and contains expected content
    assert!(file_path.exists());
    let content = tokio::fs::read_to_string(&file_path).await.unwrap();
    
    // Check structure
    assert!(content.contains("# Implementation Tasks"));
    assert!(content.contains("## Task Generation Metadata"));
    assert!(content.contains("## Task Overview"));
    assert!(content.contains("## Processing Instructions"));
    
    // Check Kiro-compatible numbering
    assert!(content.contains("- [ ] 1."));
    
    // Check that chunked file information is included
    assert!(content.contains("**Chunk**:"));
    assert!(content.contains("src/large_file.rs"));
    
    // Check metadata
    assert!(content.contains("QUERYRESULT_rust_analysis"));
    assert!(content.contains("rust_analysis.md"));
    assert!(content.contains("SELECT * FROM INGEST_20250927143022"));
}
#[test]
fn test_generate_tasks_command_integration() {
    use code_ingest::tasks::{TaskDivider, TaskConfig, MarkdownGenerator, TaskMetadata, TaskStructure};
    
    // Test the complete workflow that the CLI command would use
    let divider = TaskDivider::new(7).unwrap();
    let generator = MarkdownGenerator::new();
    
    // Simulate query results from database
    let query_results = vec![
        QueryResultRow {
            file_id: Some(1),
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(100),
            word_count: Some(500),
            token_count: Some(750),
            content_text: Some("fn main() {}".to_string()),
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
            line_count: Some(200),
            word_count: Some(1000),
            token_count: Some(1500),
            content_text: Some("pub mod test;".to_string()),
            file_type: Some("direct_text".to_string()),
            conversion_command: None,
            relative_path: Some("src/lib.rs".to_string()),
            absolute_path: Some("/project/src/lib.rs".to_string()),
        },
    ];
    
    // Create task configuration (simulating CLI arguments)
    let config = TaskConfig {
        sql_query: "SELECT * FROM INGEST_20250927143022 WHERE extension = 'rs'".to_string(),
        output_table: "QUERYRESULT_rust_analysis".to_string(),
        tasks_file: "/tmp/tasks.md".to_string(),
        group_count: 7,
        chunk_size: None,
        chunk_overlap: None,
        prompt_file: Some("rust_analysis.md".to_string()),
    };

    // Execute the workflow steps
    let tasks = divider.create_tasks_from_query_results(&query_results, &config).unwrap();
    assert_eq!(tasks.len(), 2);
    
    // Adjust group count for small number of tasks
    let actual_group_count = std::cmp::min(config.group_count, tasks.len());
    let adjusted_divider = TaskDivider::new(actual_group_count).unwrap();
    let groups = adjusted_divider.divide_into_groups(tasks.clone()).unwrap();
    
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
    
    // Generate markdown
    let markdown = generator.generate_markdown(&task_structure).unwrap();
    
    // Verify the generated markdown has the expected structure
    assert!(markdown.contains("# Implementation Tasks"));
    assert!(markdown.contains("## Task Generation Metadata"));
    assert!(markdown.contains("QUERYRESULT_rust_analysis"));
    assert!(markdown.contains("rust_analysis.md"));
    assert!(markdown.contains("- [ ] 1."));
    assert!(markdown.contains("src/main.rs"));
    assert!(markdown.contains("src/lib.rs"));
}