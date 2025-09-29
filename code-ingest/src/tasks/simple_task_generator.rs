//! Simple Task Generator for Kiro-compatible task files
//!
//! This module provides a simple task generator that creates clean checkbox markdown
//! files that Kiro can parse and execute. Unlike the complex L1L8MarkdownGenerator,
//! this generator focuses on producing minimal, parseable task lists.

use crate::error::{TaskError, TaskResult};
use crate::tasks::hierarchical_task_divider::{TaskHierarchy, HierarchicalTaskGroup, AnalysisTask};
use std::path::Path;
use tracing::{debug, info};

/// Simple task generator for creating Kiro-compatible task files
#[derive(Clone, Debug)]
pub struct SimpleTaskGenerator {
    /// Maximum number of tasks to generate (prevents Kiro overload)
    max_tasks: Option<usize>,
    /// Number of tasks to skip (for pagination)
    offset: usize,
}

impl SimpleTaskGenerator {
    /// Create a new SimpleTaskGenerator with default settings
    pub fn new() -> Self {
        Self {
            max_tasks: Some(50), // Default limit to prevent Kiro overload
            offset: 0,
        }
    }

    /// Create a SimpleTaskGenerator with no task limit
    pub fn unlimited() -> Self {
        Self {
            max_tasks: None,
            offset: 0,
        }
    }

    /// Create a SimpleTaskGenerator with a specific task limit
    pub fn with_max_tasks(max_tasks: usize) -> Self {
        Self {
            max_tasks: Some(max_tasks),
            offset: 0,
        }
    }

    /// Create a SimpleTaskGenerator with offset and limit for pagination
    pub fn with_offset_and_limit(offset: usize, max_tasks: usize) -> Self {
        Self {
            max_tasks: Some(max_tasks),
            offset,
        }
    }

    /// Generate simple checkbox markdown for a complete task structure
    ///
    /// # Arguments
    /// * `hierarchy` - Task hierarchy to generate markdown for
    /// * `table_name` - Name of the source table
    ///
    /// # Returns
    /// * `TaskResult<String>` - Generated simple markdown content
    pub async fn generate_simple_markdown(
        &self,
        hierarchy: &TaskHierarchy,
        _table_name: &str,
    ) -> TaskResult<String> {
        debug!("Generating simple checkbox markdown");

        let mut markdown = String::new();

        // Process each level to create hierarchical tasks
        self.add_hierarchy_tasks(&mut markdown, hierarchy, 0).await?;

        info!("Generated simple markdown with {} total tasks", hierarchy.total_tasks);
        Ok(markdown)
    }

    /// Write simple markdown to a file
    ///
    /// # Arguments
    /// * `hierarchy` - Task hierarchy to write
    /// * `table_name` - Name of the source table
    /// * `output_file` - Path to the output file
    ///
    /// # Returns
    /// * `TaskResult<()>` - Success or error
    pub async fn write_simple_markdown_to_file(
        &self,
        hierarchy: &TaskHierarchy,
        table_name: &str,
        output_file: &Path,
    ) -> TaskResult<()> {
        debug!("Writing simple markdown to file: {}", output_file.display());

        // Generate markdown content
        let markdown = self.generate_simple_markdown(hierarchy, table_name).await?;

        // Ensure parent directory exists
        if let Some(parent) = output_file.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: parent.display().to_string(),
                    cause: format!("Failed to create parent directory: {}", e),
                    suggestion: "Check directory permissions and available disk space".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }

        // Write markdown to file
        tokio::fs::write(output_file, markdown).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: output_file.display().to_string(),
                cause: e.to_string(),
                suggestion: "Check file permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        info!("Successfully wrote simple markdown to: {}", output_file.display());
        Ok(())
    }

    /// Add hierarchy tasks to markdown recursively
    async fn add_hierarchy_tasks(
        &self,
        markdown: &mut String,
        hierarchy: &TaskHierarchy,
        _depth: usize,
    ) -> TaskResult<()> {
        let mut task_count = 0;
        let mut tasks_processed = 0;
        
        // Add pagination info if using offset
        if self.offset > 0 {
            let end_task = self.offset + self.max_tasks.unwrap_or(50);
            markdown.push_str(&format!("<!-- Batch: Tasks {}-{} of {} total -->\n\n", 
                self.offset + 1, 
                std::cmp::min(end_task, hierarchy.total_tasks),
                hierarchy.total_tasks
            ));
        }
        
        // Process each level, respecting offset and limits
        'outer: for level in &hierarchy.levels {
            for group in &level.groups {
                // Check if we've processed enough tasks to reach our offset
                if tasks_processed < self.offset {
                    let group_task_count = self.count_tasks_in_group(group);
                    if tasks_processed + group_task_count <= self.offset {
                        tasks_processed += group_task_count;
                        continue; // Skip this entire group
                    }
                }
                
                // Check if we've hit the max limit
                if let Some(max) = self.max_tasks {
                    if task_count >= max {
                        // Add a note about pagination
                        let next_offset = self.offset + max;
                        if next_offset < hierarchy.total_tasks {
                            markdown.push_str(&format!("\n<!-- Next batch starts at task {} -->\n", next_offset + 1));
                        }
                        break 'outer;
                    }
                }
                
                let tasks_added = self.add_group_tasks_with_offset_and_limit(
                    markdown, group, 0, &mut task_count, &mut tasks_processed
                ).await?;
                
                if tasks_added == 0 && self.max_tasks.is_some() {
                    break 'outer; // Hit the limit
                }
            }
        }
        Ok(())
    }

    /// Add a hierarchical group to markdown recursively with task counting
    fn add_group_tasks_with_limit<'a>(
        &'a self,
        markdown: &'a mut String,
        group: &'a HierarchicalTaskGroup,
        depth: usize,
        task_count: &'a mut usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<usize>> + 'a>> {
        Box::pin(async move {
            let indent = "  ".repeat(depth);
            let mut tasks_added = 0;

            // Check if we've hit the limit
            if let Some(max) = self.max_tasks {
                if *task_count >= max {
                    return Ok(0);
                }
            }

            // Add group as a task using its actual title (cleaned up)
            let clean_title = self.clean_group_title(&group.title);
            if !clean_title.is_empty() {
                markdown.push_str(&format!("{}- [ ] {}\n", indent, clean_title));
                *task_count += 1;
                tasks_added += 1;
                
                // Check limit after adding group task
                if let Some(max) = self.max_tasks {
                    if *task_count >= max {
                        return Ok(tasks_added);
                    }
                }
            }

            // Add individual tasks with their actual IDs and content
            for task in &group.tasks {
                if let Some(max) = self.max_tasks {
                    if *task_count >= max {
                        break;
                    }
                }
                
                let task_indent = "  ".repeat(depth + 1);
                let task_description = self.format_task_description(task);
                markdown.push_str(&format!("{}- [ ] {}\n", task_indent, task_description));
                *task_count += 1;
                tasks_added += 1;
            }

            // Add sub-groups recursively
            for sub_group in &group.sub_groups {
                if let Some(max) = self.max_tasks {
                    if *task_count >= max {
                        break;
                    }
                }
                
                let sub_tasks_added = self.add_group_tasks_with_limit(markdown, sub_group, depth + 1, task_count).await?;
                tasks_added += sub_tasks_added;
            }

            // Add spacing after root-level groups
            if depth == 0 {
                markdown.push('\n');
            }

            Ok(tasks_added)
        })
    }

    /// Clean up group title to be more readable
    fn clean_group_title(&self, title: &str) -> String {
        // Remove common prefixes and suffixes, extract meaningful content
        let cleaned = title
            .replace("Task Group ", "")
            .replace("Analysis Group ", "")
            .replace(" (Level ", " - Level ")
            .replace(")", "");
        
        // If it's just a number or number pattern, format it nicely
        if cleaned.chars().all(|c| c.is_numeric() || c == '.' || c == ' ' || c == '-') {
            let parts: Vec<&str> = cleaned.split(" - ").collect();
            if let Some(number) = parts.first() {
                if number.trim().chars().all(|c| c.is_numeric() || c == '.') {
                    return format!("{}. Task Group {}", number.trim(), number.trim());
                }
            }
        }
        
        // Return cleaned title or original if cleaning didn't help
        if cleaned.trim().is_empty() {
            title.to_string()
        } else {
            cleaned
        }
    }

    /// Format task description using actual task data
    fn format_task_description(&self, task: &AnalysisTask) -> String {
        // Use the actual task ID and any meaningful content
        let base_description = if task.id.is_empty() {
            format!("Task {}", task.row_number)
        } else {
            task.id.clone()
        };
        
        // Add context if available (but keep it concise for Kiro)
        if !task.table_name.is_empty() && task.row_number > 0 {
            format!("{}. Analyze {} row {}", base_description, task.table_name, task.row_number)
        } else {
            base_description
        }
    }
}

impl Default for SimpleTaskGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::content_extractor::ContentTriple;
    use crate::tasks::hierarchical_task_divider::HierarchicalTaskDivider;
    use std::path::PathBuf;

    fn create_test_hierarchy() -> TaskHierarchy {
        let divider = HierarchicalTaskDivider::new(2, 2).unwrap();
        let content_triples = vec![
            ContentTriple {
                content_a: PathBuf::from(".raw_data_202509/TEST_1_Content.txt"),
                content_b: PathBuf::from(".raw_data_202509/TEST_1_Content_L1.txt"),
                content_c: PathBuf::from(".raw_data_202509/TEST_1_Content_L2.txt"),
                row_number: 1,
                table_name: "TEST_TABLE".to_string(),
            },
            ContentTriple {
                content_a: PathBuf::from(".raw_data_202509/TEST_2_Content.txt"),
                content_b: PathBuf::from(".raw_data_202509/TEST_2_Content_L1.txt"),
                content_c: PathBuf::from(".raw_data_202509/TEST_2_Content_L2.txt"),
                row_number: 2,
                table_name: "TEST_TABLE".to_string(),
            },
        ];
        divider.create_hierarchy(content_triples).unwrap()
    }

    #[test]
    fn test_simple_task_generator_creation() {
        let generator = SimpleTaskGenerator::new();
        // Just verify it can be created
        assert_eq!(format!("{:?}", generator), "SimpleTaskGenerator");
    }

    #[tokio::test]
    async fn test_generate_simple_markdown() {
        let generator = SimpleTaskGenerator::new();
        let hierarchy = create_test_hierarchy();
        let markdown = generator.generate_simple_markdown(&hierarchy, "TEST_TABLE").await.unwrap();
        
        // Verify simple checkbox format
        assert!(markdown.contains("- [ ]"));
        assert!(!markdown.contains("# L1-L8 Analysis Tasks")); // Should not contain complex headers
        assert!(!markdown.contains("## Task Generation Metadata")); // Should not contain metadata
        assert!(!markdown.contains("**Content**:")); // Should not contain detailed content info
        
        // Should be simple and clean
        let lines: Vec<&str> = markdown.lines().collect();
        for line in lines {
            if !line.trim().is_empty() {
                // Each non-empty line should be a checkbox or indented checkbox
                assert!(line.contains("- [ ]") || line.starts_with("  "));
            }
        }
    }

    #[test]
    fn test_clean_group_title() {
        let generator = SimpleTaskGenerator::new();
        
        // Test various title formats
        assert_eq!(generator.clean_group_title("Task Group 1 (Level 1)"), "1. Task Group 1");
        assert_eq!(generator.clean_group_title("Analysis Group 1.1.1 (Level 3)"), "1.1.1. Task Group 1.1.1");
        assert_eq!(generator.clean_group_title("Custom Analysis Phase"), "Custom Analysis Phase");
        
        // Test edge cases
        assert_eq!(generator.clean_group_title(""), "");
        assert_eq!(generator.clean_group_title("Random Title"), "Random Title");
    }

    #[test]
    fn test_format_task_description() {
        use crate::tasks::hierarchical_task_divider::AnalysisStage;
        use std::path::PathBuf;
        
        let generator = SimpleTaskGenerator::new();
        
        // Create a test task
        let task = AnalysisTask {
            id: "1.2.3.4".to_string(),
            table_name: "INGEST_TEST".to_string(),
            row_number: 42,
            content_files: crate::tasks::content_extractor::ContentTriple {
                content_a: PathBuf::from("test_a.txt"),
                content_b: PathBuf::from("test_b.txt"),
                content_c: PathBuf::from("test_c.txt"),
                row_number: 42,
                table_name: "INGEST_TEST".to_string(),
            },
            prompt_file: PathBuf::from("prompt.md"),
            output_file: PathBuf::from("output.md"),
            analysis_stages: vec![AnalysisStage::AnalyzeA],
        };
        
        let description = generator.format_task_description(&task);
        assert_eq!(description, "1.2.3.4. Analyze INGEST_TEST row 42");
        
        // Test with empty ID
        let task_no_id = AnalysisTask {
            id: "".to_string(),
            table_name: "TEST_TABLE".to_string(),
            row_number: 5,
            content_files: crate::tasks::content_extractor::ContentTriple {
                content_a: PathBuf::from("test_a.txt"),
                content_b: PathBuf::from("test_b.txt"),
                content_c: PathBuf::from("test_c.txt"),
                row_number: 5,
                table_name: "TEST_TABLE".to_string(),
            },
            prompt_file: PathBuf::from("prompt.md"),
            output_file: PathBuf::from("output.md"),
            analysis_stages: vec![AnalysisStage::AnalyzeA],
        };
        
        let description_no_id = generator.format_task_description(&task_no_id);
        assert_eq!(description_no_id, "Task 5. Analyze TEST_TABLE row 5");
    }

    #[tokio::test]
    async fn test_write_simple_markdown_to_file() {
        let generator = SimpleTaskGenerator::new();
        let hierarchy = create_test_hierarchy();
        
        // Create a temporary file path
        let temp_dir = tempfile::tempdir().unwrap();
        let output_file = temp_dir.path().join("simple_tasks.md");
        
        // Write the task structure
        generator.write_simple_markdown_to_file(&hierarchy, "TEST_TABLE", &output_file).await.unwrap();
        
        // Verify the file was created and contains expected content
        assert!(output_file.exists());
        let content = tokio::fs::read_to_string(&output_file).await.unwrap();
        
        // Should contain simple checkbox format
        assert!(content.contains("- [ ]"));
        assert!(!content.contains("# L1-L8")); // Should not contain complex headers
        
        // Should be parseable by simple markdown parsers
        let lines: Vec<&str> = content.lines().collect();
        for line in lines {
            if !line.trim().is_empty() {
                assert!(line.contains("- [ ]") || line.starts_with("  "));
            }
        }
    }

    #[tokio::test]
    async fn test_format_matches_reference() {
        let generator = SimpleTaskGenerator::new();
        let hierarchy = create_test_hierarchy();
        let markdown = generator.generate_simple_markdown(&hierarchy, "TEST_TABLE").await.unwrap();
        
        // The format should match the reference file pattern:
        // - [ ] 1. Task 1
        //   - [ ] 1.1 Task 1.1
        //     - [ ] 1.1.1 Task 1.1.1
        
        // Check for proper indentation and checkbox format
        let lines: Vec<&str> = markdown.lines().filter(|line| !line.trim().is_empty()).collect();
        
        for line in lines {
            // Each line should be a checkbox with proper indentation
            let trimmed = line.trim_start();
            assert!(trimmed.starts_with("- [ ]"), "Line should start with checkbox: {}", line);
            
            // Indentation should be multiples of 2 spaces
            let indent_count = line.len() - line.trim_start().len();
            assert_eq!(indent_count % 2, 0, "Indentation should be multiples of 2 spaces: {}", line);
        }
    }
}