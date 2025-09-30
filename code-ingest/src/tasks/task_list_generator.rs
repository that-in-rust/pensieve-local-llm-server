//! Task list generator for creating task lists in txt format
//!
//! This module provides functionality to generate task lists that reference content files
//! by row number, creating a format compatible with existing task processing workflows.
//!
//! # Examples
//!
//! ```rust
//! use code_ingest::tasks::task_list_generator::TaskListGenerator;
//! use code_ingest::database::models::IngestedFile;
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let generator = TaskListGenerator::new();
//! let rows = vec![/* IngestedFile instances */];
//! let task_list = generator.generate_task_list(&rows)?;
//! 
//! // Write to file
//! generator.write_task_list_to_file(&rows, &PathBuf::from("tasks.txt")).await?;
//! # Ok(())
//! # }
//! ```

use crate::database::models::IngestedFile;
use crate::tasks::chunk_level_task_generator::{TaskGeneratorError, TaskGeneratorResult};
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Generator for creating task lists in txt format
#[derive(Debug, Clone)]
pub struct TaskListGenerator {
    /// Template for task descriptions (optional)
    task_template: Option<String>,
}

impl TaskListGenerator {
    /// Create a new task list generator
    pub fn new() -> Self {
        debug!("Creating new TaskListGenerator");
        Self {
            task_template: None,
        }
    }

    /// Create a task list generator with a custom template
    pub fn with_template(template: String) -> Self {
        debug!("Creating TaskListGenerator with custom template");
        Self {
            task_template: Some(template),
        }
    }

    /// Generate task list content that references content files by row number
    /// 
    /// # Arguments
    /// * `rows` - Vector of IngestedFile records to generate tasks for
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<String>` - Generated task list content or error
    /// 
    /// # Requirements
    /// This method satisfies requirement 1.2 and 2.7 by generating task lists that reference
    /// content files by row number in a format compatible with existing workflows
    pub fn generate_task_list(&self, rows: &[IngestedFile]) -> TaskGeneratorResult<String> {
        debug!("Generating task list for {} rows", rows.len());

        if rows.is_empty() {
            warn!("No rows provided for task list generation");
            return Ok(String::new());
        }

        let mut task_list = String::new();
        
        // Add header
        self.add_header(&mut task_list, rows)?;
        
        // Add task entries
        for (index, row) in rows.iter().enumerate() {
            let row_number = index + 1;
            self.add_task_entry(&mut task_list, row, row_number)?;
        }
        
        // Add footer with instructions
        self.add_footer(&mut task_list, rows)?;
        
        info!("Successfully generated task list with {} entries", rows.len());
        Ok(task_list)
    }

    /// Write task list to a file
    /// 
    /// # Arguments
    /// * `rows` - Vector of IngestedFile records to generate tasks for
    /// * `output_path` - Path where the task list file should be written
    /// 
    /// # Returns
    /// * `TaskGeneratorResult<()>` - Success or error
    pub async fn write_task_list_to_file(
        &self,
        rows: &[IngestedFile],
        output_path: &PathBuf,
    ) -> TaskGeneratorResult<()> {
        debug!("Writing task list to file: {}", output_path.display());

        let task_list_content = self.generate_task_list(rows)?;

        // Ensure parent directory exists
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                TaskGeneratorError::task_list_failed(format!(
                    "Failed to create parent directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        // Write the task list to file
        tokio::fs::write(output_path, task_list_content).await.map_err(|e| {
            TaskGeneratorError::task_list_failed(format!(
                "Failed to write task list to {}: {}",
                output_path.display(),
                e
            ))
        })?;

        info!("Successfully wrote task list to: {}", output_path.display());
        Ok(())
    }

    /// Get the default task list filename for a table
    pub fn default_task_list_filename(table_name: &str) -> String {
        format!("{}-tasks.txt", table_name.to_lowercase())
    }

    /// Validate task list format for compatibility with existing workflows
    pub fn validate_task_list_format(&self, task_list: &str) -> TaskGeneratorResult<()> {
        debug!("Validating task list format");

        if task_list.is_empty() {
            return Err(TaskGeneratorError::task_list_failed("Task list is empty"));
        }

        // Check for required sections
        if !task_list.contains("# Task List") {
            return Err(TaskGeneratorError::task_list_failed("Missing task list header"));
        }

        if !task_list.contains("## Tasks") {
            return Err(TaskGeneratorError::task_list_failed("Missing tasks section"));
        }

        // Check for content file references
        if !task_list.contains("content_") {
            return Err(TaskGeneratorError::task_list_failed("No content file references found"));
        }

        debug!("Task list format validation passed");
        Ok(())
    }

    // Private helper methods

    /// Add header section to the task list
    fn add_header(&self, task_list: &mut String, rows: &[IngestedFile]) -> TaskGeneratorResult<()> {
        task_list.push_str("# Task List\n\n");
        task_list.push_str(&format!("Generated task list for {} files.\n\n", rows.len()));
        
        // Add metadata about the files
        let total_lines: i32 = rows.iter()
            .filter_map(|row| row.line_count)
            .sum();
        
        let total_size: i64 = rows.iter()
            .map(|row| row.file_size_bytes)
            .sum();

        task_list.push_str("## Summary\n\n");
        task_list.push_str(&format!("- **Total Files**: {}\n", rows.len()));
        task_list.push_str(&format!("- **Total Lines**: {}\n", total_lines));
        task_list.push_str(&format!("- **Total Size**: {} bytes\n", total_size));
        
        // Add file type breakdown
        let mut file_types = std::collections::HashMap::new();
        for row in rows {
            if let Some(ext) = &row.extension {
                *file_types.entry(ext.clone()).or_insert(0) += 1;
            }
        }
        
        if !file_types.is_empty() {
            task_list.push_str("- **File Types**: ");
            let type_summary: Vec<String> = file_types.iter()
                .map(|(ext, count)| format!("{} ({})", ext, count))
                .collect();
            task_list.push_str(&type_summary.join(", "));
            task_list.push('\n');
        }
        
        task_list.push_str("\n## Tasks\n\n");
        task_list.push_str("Each task references three content files:\n");
        task_list.push_str("- `content_N.txt`: Primary content for analysis\n");
        task_list.push_str("- `contentL1_N.txt`: Content with L1 context (current + next)\n");
        task_list.push_str("- `contentL2_N.txt`: Content with L2 context (current + next + next2)\n\n");
        
        Ok(())
    }

    /// Add a task entry for a specific row
    fn add_task_entry(
        &self,
        task_list: &mut String,
        row: &IngestedFile,
        row_number: usize,
    ) -> TaskGeneratorResult<()> {
        // Use custom template if provided, otherwise use default format
        if let Some(template) = &self.task_template {
            let task_entry = self.apply_template(template, row, row_number)?;
            task_list.push_str(&task_entry);
        } else {
            self.add_default_task_entry(task_list, row, row_number)?;
        }
        
        task_list.push('\n');
        Ok(())
    }

    /// Add a default task entry format
    fn add_default_task_entry(
        &self,
        task_list: &mut String,
        row: &IngestedFile,
        row_number: usize,
    ) -> TaskGeneratorResult<()> {
        // Task header with file information
        task_list.push_str(&format!("### Task {}: {}\n\n", row_number, row.filename));
        
        // File details
        task_list.push_str("**File Details:**\n");
        task_list.push_str(&format!("- Path: `{}`\n", row.filepath));
        
        if let Some(extension) = &row.extension {
            task_list.push_str(&format!("- Type: {}\n", extension));
        }
        
        if let Some(line_count) = row.line_count {
            task_list.push_str(&format!("- Lines: {}\n", line_count));
        }
        
        if let Some(word_count) = row.word_count {
            task_list.push_str(&format!("- Words: {}\n", word_count));
        }
        
        task_list.push_str(&format!("- Size: {} bytes\n", row.file_size_bytes));
        task_list.push('\n');
        
        // Content file references
        task_list.push_str("**Content Files:**\n");
        task_list.push_str(&format!("- Primary: `content_{}.txt`\n", row_number));
        task_list.push_str(&format!("- L1 Context: `contentL1_{}.txt`\n", row_number));
        task_list.push_str(&format!("- L2 Context: `contentL2_{}.txt`\n", row_number));
        task_list.push('\n');
        
        // Analysis instructions
        task_list.push_str("**Analysis Instructions:**\n");
        task_list.push_str("1. Review the primary content file for initial understanding\n");
        task_list.push_str("2. Use L1 context file for immediate context (current + next)\n");
        task_list.push_str("3. Use L2 context file for broader context (current + next + next2)\n");
        task_list.push_str("4. Document findings and analysis results\n");
        
        Ok(())
    }

    /// Apply a custom template to generate task entry
    fn apply_template(
        &self,
        template: &str,
        row: &IngestedFile,
        row_number: usize,
    ) -> TaskGeneratorResult<String> {
        let mut result = template.to_string();
        
        // Replace template variables
        result = result.replace("{row_number}", &row_number.to_string());
        result = result.replace("{filename}", &row.filename);
        result = result.replace("{filepath}", &row.filepath);
        result = result.replace("{file_id}", &row.file_id.to_string());
        
        if let Some(extension) = &row.extension {
            result = result.replace("{extension}", extension);
        } else {
            result = result.replace("{extension}", "unknown");
        }
        
        if let Some(line_count) = row.line_count {
            result = result.replace("{line_count}", &line_count.to_string());
        } else {
            result = result.replace("{line_count}", "unknown");
        }
        
        if let Some(word_count) = row.word_count {
            result = result.replace("{word_count}", &word_count.to_string());
        } else {
            result = result.replace("{word_count}", "unknown");
        }
        
        result = result.replace("{file_size}", &row.file_size_bytes.to_string());
        result = result.replace("{content_file}", &format!("content_{}.txt", row_number));
        result = result.replace("{content_l1_file}", &format!("contentL1_{}.txt", row_number));
        result = result.replace("{content_l2_file}", &format!("contentL2_{}.txt", row_number));
        
        Ok(result)
    }

    /// Add footer with processing instructions
    fn add_footer(&self, task_list: &mut String, rows: &[IngestedFile]) -> TaskGeneratorResult<()> {
        task_list.push_str("## Processing Instructions\n\n");
        task_list.push_str("### Workflow\n\n");
        task_list.push_str("1. **Sequential Processing**: Process tasks in order from 1 to {}\n", rows.len());
        task_list.push_str("2. **Content File Usage**:\n");
        task_list.push_str("   - Start with `content_N.txt` for focused analysis\n");
        task_list.push_str("   - Use `contentL1_N.txt` when you need immediate context\n");
        task_list.push_str("   - Use `contentL2_N.txt` for comprehensive context analysis\n");
        task_list.push_str("3. **Documentation**: Record analysis results and findings\n");
        task_list.push_str("4. **Quality Check**: Verify analysis completeness before moving to next task\n\n");
        
        task_list.push_str("### File Naming Convention\n\n");
        task_list.push_str("- `content_N.txt`: Primary content for task N\n");
        task_list.push_str("- `contentL1_N.txt`: L1 context (current + next) for task N\n");
        task_list.push_str("- `contentL2_N.txt`: L2 context (current + next + next2) for task N\n\n");
        
        task_list.push_str("### Notes\n\n");
        task_list.push_str(&format!("- Total tasks: {}\n", rows.len()));
        task_list.push_str("- Each task corresponds to one source file\n");
        task_list.push_str("- Content files provide different levels of context for analysis\n");
        task_list.push_str("- Process systematically for consistent results\n");
        
        Ok(())
    }
}

impl Default for TaskListGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    }

    #[test]
    fn test_generate_task_list_empty() {
        let generator = TaskListGenerator::new();
        let rows = vec![];
        
        let result = generator.generate_task_list(&rows).unwrap();
        assert!(result.is_empty());
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
        
        // Check content file references
        assert!(result.contains("Primary: `content_1.txt`"));
        assert!(result.contains("L1 Context: `contentL1_1.txt`"));
        assert!(result.contains("L2 Context: `contentL2_1.txt`"));
        
        // Check instructions
        assert!(result.contains("## Processing Instructions"));
        assert!(result.contains("Sequential Processing"));
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
        
        // Check content file references for each task
        assert!(result.contains("Primary: `content_1.txt`"));
        assert!(result.contains("Primary: `content_2.txt`"));
        assert!(result.contains("Primary: `content_3.txt`"));
        
        assert!(result.contains("L1 Context: `contentL1_1.txt`"));
        assert!(result.contains("L1 Context: `contentL1_2.txt`"));
        assert!(result.contains("L1 Context: `contentL1_3.txt`"));
        
        assert!(result.contains("L2 Context: `contentL2_1.txt`"));
        assert!(result.contains("L2 Context: `contentL2_2.txt`"));
        assert!(result.contains("L2 Context: `contentL2_3.txt`"));
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
    }

    #[test]
    fn test_template_variable_replacement() {
        let generator = TaskListGenerator::new();
        let template = "File: {filename}, Path: {filepath}, ID: {file_id}, Ext: {extension}, Lines: {line_count}, Words: {word_count}, Size: {file_size}";
        
        let row = create_test_ingested_file(42, "src/test.rs", "test.rs", Some("rs"), Some(100), Some(500), 2048);
        
        let result = generator.apply_template(template, &row, 5).unwrap();
        
        assert_eq!(result, "File: test.rs, Path: src/test.rs, ID: 42, Ext: rs, Lines: 100, Words: 500, Size: 2048");
    }

    #[test]
    fn test_template_with_missing_values() {
        let generator = TaskListGenerator::new();
        let template = "File: {filename}, Ext: {extension}, Lines: {line_count}, Words: {word_count}";
        
        let row = create_test_ingested_file(1, "src/test.txt", "test.txt", None, None, None, 1024);
        
        let result = generator.apply_template(template, &row, 1).unwrap();
        
        assert_eq!(result, "File: test.txt, Ext: unknown, Lines: unknown, Words: unknown");
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
    }

    #[test]
    fn test_task_list_generator_default() {
        let generator = TaskListGenerator::default();
        assert!(generator.task_template.is_none());
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