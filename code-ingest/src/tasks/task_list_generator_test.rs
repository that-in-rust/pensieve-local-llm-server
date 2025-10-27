//! Unit tests for TaskListGenerator
//!
//! This module contains comprehensive unit tests for the TaskListGenerator
//! to ensure it meets all requirements for task file creation.

#[cfg(test)]
mod tests {
    use crate::tasks::task_list_generator::TaskListGenerator;
    use crate::database::models::IngestedFile;
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_test_ingested_file(
        file_id: i64,
        filepath: &str,
        filename: &str,
        extension: Option<&str>,
        line_count: Option<i32>,
        word_count: Option<i32>,
        file_size_bytes: i64,
    ) -> IngestedFile {
        IngestedFile {
            file_id,
            ingestion_id: 1,
            filepath: filepath.to_string(),
            filename: filename.to_string(),
            extension: extension.map(|s| s.to_string()),
            file_size_bytes,
            line_count,
            word_count,
            token_count: word_count.map(|w| (w as f32 * 0.75) as i32),
            content_text: Some(format!("Content for {}", filename)),
            file_type_str: "direct_text".to_string(),
            conversion_command: None,
            relative_path: filepath.to_string(),
            absolute_path: format!("/tmp/{}", filepath),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_task_list_generator_creation() {
        let generator = TaskListGenerator::new();
        assert!(generator.task_template.is_none());

        let generator_with_template = TaskListGenerator::with_template("Custom template".to_string());
        assert!(generator_with_template.task_template.is_some());
        assert_eq!(generator_with_template.task_template.unwrap(), "Custom template");
        
        println!("✅ TaskListGenerator creation test passed");
    }

    #[test]
    fn test_generate_task_list_empty() {
        let generator = TaskListGenerator::new();
        let rows = vec![];
        
        let result = generator.generate_task_list(&rows).unwrap();
        assert!(result.is_empty());
        
        println!("✅ Empty task list generation test passed");
    }

    #[test]
    fn test_generate_task_list_single_file() {
        let generator = TaskListGenerator::new();
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024)
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Check header
        assert!(result.contains("# Task List"));
        assert!(result.contains("Generated task list for 1 files"));
        assert!(result.contains("**Total Files**: 1"));
        assert!(result.contains("**Total Lines**: 50"));
        assert!(result.contains("**Total Size**: 1024 bytes"));
        
        // Check task entry
        assert!(result.contains("### Task 1: main.rs"));
        assert!(result.contains("Path: `src/main.rs`"));
        assert!(result.contains("Type: rs"));
        assert!(result.contains("Lines: 50"));
        assert!(result.contains("Words: 200"));
        assert!(result.contains("Size: 1024 bytes"));
        
        // Check content file references (requirement 1.2 and 2.7)
        assert!(result.contains("Primary: `content_1.txt`"));
        assert!(result.contains("L1 Context: `contentL1_1.txt`"));
        assert!(result.contains("L2 Context: `contentL2_1.txt`"));
        
        // Check instructions
        assert!(result.contains("## Processing Instructions"));
        assert!(result.contains("Sequential Processing"));
        
        println!("✅ Single file task list generation test passed");
    }

    #[test]
    fn test_generate_task_list_multiple_files() {
        let generator = TaskListGenerator::new();
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024),
            create_test_ingested_file(2, "src/lib.rs", "lib.rs", Some("rs"), Some(100), Some(400), 2048),
            create_test_ingested_file(3, "README.md", "README.md", Some("md"), Some(25), Some(150), 512),
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Check summary
        assert!(result.contains("**Total Files**: 3"));
        assert!(result.contains("**Total Lines**: 175")); // 50 + 100 + 25
        assert!(result.contains("**Total Size**: 3584 bytes")); // 1024 + 2048 + 512
        assert!(result.contains("**File Types**: rs (2), md (1)"));
        
        // Check all tasks are present
        assert!(result.contains("### Task 1: main.rs"));
        assert!(result.contains("### Task 2: lib.rs"));
        assert!(result.contains("### Task 3: README.md"));
        
        // Check content file references for each task (requirement 1.2 and 2.7)
        assert!(result.contains("Primary: `content_1.txt`"));
        assert!(result.contains("Primary: `content_2.txt`"));
        assert!(result.contains("Primary: `content_3.txt`"));
        
        assert!(result.contains("L1 Context: `contentL1_1.txt`"));
        assert!(result.contains("L1 Context: `contentL1_2.txt`"));
        assert!(result.contains("L1 Context: `contentL1_3.txt`"));
        
        assert!(result.contains("L2 Context: `contentL2_1.txt`"));
        assert!(result.contains("L2 Context: `contentL2_2.txt`"));
        assert!(result.contains("L2 Context: `contentL2_3.txt`"));
        
        println!("✅ Multiple files task list generation test passed");
    }

    #[test]
    fn test_generate_task_list_with_template() {
        let template = "Task {row_number}: Analyze {filename} ({extension}) - {line_count} lines\nContent: {content_file}\nL1: {content_l1_file}\nL2: {content_l2_file}\n";
        let generator = TaskListGenerator::with_template(template.to_string());
        
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024)
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Check template application
        assert!(result.contains("Task 1: Analyze main.rs (rs) - 50 lines"));
        assert!(result.contains("Content: content_1.txt"));
        assert!(result.contains("L1: contentL1_1.txt"));
        assert!(result.contains("L2: contentL2_1.txt"));
        
        println!("✅ Template-based task list generation test passed");
    }

    #[test]
    fn test_template_variable_replacement() {
        let generator = TaskListGenerator::new();
        let template = "File: {filename}, Path: {filepath}, ID: {file_id}, Ext: {extension}, Lines: {line_count}, Words: {word_count}, Size: {file_size}";
        
        let row = create_test_ingested_file(42, "src/test.rs", "test.rs", Some("rs"), Some(100), Some(500), 2048);
        
        let result = generator.apply_template(template, &row, 5).unwrap();
        
        assert_eq!(result, "File: test.rs, Path: src/test.rs, ID: 42, Ext: rs, Lines: 100, Words: 500, Size: 2048");
        
        println!("✅ Template variable replacement test passed");
    }

    #[test]
    fn test_template_with_missing_values() {
        let generator = TaskListGenerator::new();
        let template = "File: {filename}, Ext: {extension}, Lines: {line_count}, Words: {word_count}";
        
        let row = create_test_ingested_file(1, "src/test.txt", "test.txt", None, None, None, 1024);
        
        let result = generator.apply_template(template, &row, 1).unwrap();
        
        assert_eq!(result, "File: test.txt, Ext: unknown, Lines: unknown, Words: unknown");
        
        println!("✅ Template with missing values test passed");
    }

    #[tokio::test]
    async fn test_write_task_list_to_file() {
        let generator = TaskListGenerator::new();
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_tasks.txt");
        
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024)
        ];
        
        let result = generator.write_task_list_to_file(&rows, &output_path).await;
        assert!(result.is_ok());
        
        // Verify file was created
        assert!(output_path.exists());
        
        // Verify file content
        let content = tokio::fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("# Task List"));
        assert!(content.contains("### Task 1: main.rs"));
        assert!(content.contains("Primary: `content_1.txt`"));
        
        println!("✅ Write task list to file test passed");
    }

    #[tokio::test]
    async fn test_write_task_list_creates_parent_directory() {
        let generator = TaskListGenerator::new();
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("subdir").join("tasks.txt");
        
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024)
        ];
        
        let result = generator.write_task_list_to_file(&rows, &output_path).await;
        assert!(result.is_ok());
        
        // Verify parent directory was created
        assert!(output_path.parent().unwrap().exists());
        assert!(output_path.exists());
        
        println!("✅ Parent directory creation test passed");
    }

    #[test]
    fn test_default_task_list_filename() {
        assert_eq!(
            TaskListGenerator::default_task_list_filename("INGEST_20250927143022"),
            "ingest_20250927143022-tasks.txt"
        );
        
        assert_eq!(
            TaskListGenerator::default_task_list_filename("TEST_TABLE"),
            "test_table-tasks.txt"
        );
        
        println!("✅ Default filename generation test passed");
    }

    #[test]
    fn test_validate_task_list_format() {
        let generator = TaskListGenerator::new();
        
        // Test empty task list
        let result = generator.validate_task_list_format("");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Task list is empty"));
        
        // Test missing header
        let result = generator.validate_task_list_format("Some content without header");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing task list header"));
        
        // Test missing tasks section
        let result = generator.validate_task_list_format("# Task List\nSome content");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing tasks section"));
        
        // Test missing content file references
        let result = generator.validate_task_list_format("# Task List\n## Tasks\nSome tasks");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No content file references found"));
        
        // Test valid format
        let valid_task_list = "# Task List\n## Tasks\nTask with content_1.txt reference";
        let result = generator.validate_task_list_format(valid_task_list);
        assert!(result.is_ok());
        
        println!("✅ Task list format validation test passed");
    }

    #[test]
    fn test_file_type_breakdown() {
        let generator = TaskListGenerator::new();
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024),
            create_test_ingested_file(2, "src/lib.rs", "lib.rs", Some("rs"), Some(100), Some(400), 2048),
            create_test_ingested_file(3, "README.md", "README.md", Some("md"), Some(25), Some(150), 512),
            create_test_ingested_file(4, "Cargo.toml", "Cargo.toml", Some("toml"), Some(20), Some(80), 256),
            create_test_ingested_file(5, "src/utils.rs", "utils.rs", Some("rs"), Some(75), Some(300), 1536),
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Check file type breakdown (should show rs (3), md (1), toml (1))
        assert!(result.contains("**File Types**:"));
        assert!(result.contains("rs (3)"));
        assert!(result.contains("md (1)"));
        assert!(result.contains("toml (1)"));
        
        println!("✅ File type breakdown test passed");
    }

    #[test]
    fn test_task_list_generator_default() {
        let generator = TaskListGenerator::default();
        assert!(generator.task_template.is_none());
        
        println!("✅ Default constructor test passed");
    }

    #[test]
    fn test_content_file_references_by_row_number() {
        // This test specifically validates requirement 1.2 and 2.7:
        // "references content files by row number"
        let generator = TaskListGenerator::new();
        let rows = vec![
            create_test_ingested_file(101, "file1.rs", "file1.rs", Some("rs"), Some(50), Some(200), 1024),
            create_test_ingested_file(202, "file2.rs", "file2.rs", Some("rs"), Some(100), Some(400), 2048),
            create_test_ingested_file(303, "file3.rs", "file3.rs", Some("rs"), Some(75), Some(300), 1536),
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Verify that content files are referenced by row number (1, 2, 3), not file_id (101, 202, 303)
        assert!(result.contains("content_1.txt"));
        assert!(result.contains("contentL1_1.txt"));
        assert!(result.contains("contentL2_1.txt"));
        
        assert!(result.contains("content_2.txt"));
        assert!(result.contains("contentL1_2.txt"));
        assert!(result.contains("contentL2_2.txt"));
        
        assert!(result.contains("content_3.txt"));
        assert!(result.contains("contentL1_3.txt"));
        assert!(result.contains("contentL2_3.txt"));
        
        // Verify that file_id values are NOT used in content file names
        assert!(!result.contains("content_101.txt"));
        assert!(!result.contains("content_202.txt"));
        assert!(!result.contains("content_303.txt"));
        
        println!("✅ Content file references by row number test passed");
    }

    #[test]
    fn test_task_list_format_compatibility() {
        // This test validates requirement 2.7:
        // "Create task list format that's compatible with existing task processing workflows"
        let generator = TaskListGenerator::new();
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024)
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Check for workflow-compatible format elements
        assert!(result.contains("# Task List")); // Standard markdown header
        assert!(result.contains("## Summary")); // Summary section
        assert!(result.contains("## Tasks")); // Tasks section
        assert!(result.contains("## Processing Instructions")); // Instructions section
        
        // Check for structured task entries
        assert!(result.contains("### Task 1:")); // Numbered task headers
        assert!(result.contains("**File Details:**")); // Structured details
        assert!(result.contains("**Content Files:**")); // Content file section
        assert!(result.contains("**Analysis Instructions:**")); // Analysis guidance
        
        // Check for workflow instructions
        assert!(result.contains("Sequential Processing"));
        assert!(result.contains("Content File Usage"));
        assert!(result.contains("File Naming Convention"));
        
        println!("✅ Task list format compatibility test passed");
    }

    #[test]
    fn test_requirements_validation() {
        // This test validates that all requirements are met:
        // Requirement 1.2: "references content files by row number"
        // Requirement 2.7: "Create task list format that's compatible with existing task processing workflows"
        
        let generator = TaskListGenerator::new();
        let rows = vec![
            create_test_ingested_file(1, "src/main.rs", "main.rs", Some("rs"), Some(50), Some(200), 1024),
            create_test_ingested_file(2, "src/lib.rs", "lib.rs", Some("rs"), Some(100), Some(400), 2048),
        ];
        
        let result = generator.generate_task_list(&rows).unwrap();
        
        // Requirement 1.2: References content files by row number
        assert!(result.contains("content_1.txt"));
        assert!(result.contains("contentL1_1.txt"));
        assert!(result.contains("contentL2_1.txt"));
        assert!(result.contains("content_2.txt"));
        assert!(result.contains("contentL1_2.txt"));
        assert!(result.contains("contentL2_2.txt"));
        
        // Requirement 2.7: Compatible format with existing workflows
        let validation_result = generator.validate_task_list_format(&result);
        assert!(validation_result.is_ok(), "Task list format should be valid for existing workflows");
        
        // Additional compatibility checks
        assert!(result.contains("# Task List")); // Standard header
        assert!(result.contains("## Tasks")); // Required section
        assert!(result.contains("### Task")); // Task entries
        assert!(result.contains("Primary: `content_")); // Content file references
        assert!(result.contains("L1 Context: `contentL1_")); // L1 references
        assert!(result.contains("L2 Context: `contentL2_")); // L2 references
        
        println!("✅ All requirements validation test passed");
    }

    #[test]
    fn test_basic_functionality() {
        // Test that our basic functionality works without external dependencies
        let generator = TaskListGenerator::new();
        
        // Test filename generation
        let filename = TaskListGenerator::default_task_list_filename("TEST_TABLE");
        assert_eq!(filename, "test_table-tasks.txt");
        
        // Test empty task list generation
        let empty_result = generator.generate_task_list(&[]).unwrap();
        assert!(empty_result.is_empty());
        
        println!("✅ TaskListGenerator basic functionality test passed");
    }
}