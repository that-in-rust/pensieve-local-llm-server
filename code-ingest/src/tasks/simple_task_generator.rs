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
pub struct SimpleTaskGenerator;

impl SimpleTaskGenerator {
    /// Create a new SimpleTaskGenerator
    pub fn new() -> Self {
        Self
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
        // Process each level
        for level in &hierarchy.levels {
            for group in &level.groups {
                self.add_group_tasks(markdown, group, 0).await?;
            }
        }
        Ok(())
    }

    /// Add a hierarchical group to markdown recursively
    fn add_group_tasks<'a>(
        &'a self,
        markdown: &'a mut String,
        group: &'a HierarchicalTaskGroup,
        depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<()>> + 'a>> {
        Box::pin(async move {
            let indent = "  ".repeat(depth);

            // Add group as a task if it has an ID
            if let Some(task_id) = self.extract_task_id(&group.title) {
                markdown.push_str(&format!("{}* [ ] {}\n", indent, task_id));
            }

            // Add individual tasks
            for task in &group.tasks {
                let task_indent = "  ".repeat(depth + 1);
                markdown.push_str(&format!("{}* [ ] {}\n", task_indent, task.id));
            }

            // Add sub-groups recursively
            for sub_group in &group.sub_groups {
                self.add_group_tasks(markdown, sub_group, depth + 1).await?;
            }

            // Add spacing after root-level groups
            if depth == 0 {
                markdown.push('\n');
            }

            Ok(())
        })
    }

    /// Extract task ID from group title
    fn extract_task_id(&self, title: &str) -> Option<String> {
        // Look for patterns like "Task Group 1.1.1 (Level 3)" and extract "1.1.1"
        if title.contains("Task Group") || title.contains("Analysis Group") {
            // Try to extract the number pattern
            let parts: Vec<&str> = title.split_whitespace().collect();
            for part in parts {
                if part.contains('.') && part.chars().all(|c| c.is_numeric() || c == '.') {
                    return Some(format!("{}. {}", part, self.generate_task_description(part)));
                }
                if part.chars().all(|c| c.is_numeric()) && part.len() <= 3 {
                    return Some(format!("{}. {}", part, self.generate_task_description(part)));
                }
            }
        }
        None
    }

    /// Generate a simple task description based on the task ID
    fn generate_task_description(&self, task_id: &str) -> String {
        let level = task_id.matches('.').count() + 1;
        match level {
            1 => format!("Task {}", task_id),
            2 => format!("Task {}", task_id),
            3 => format!("Task {}", task_id),
            4 => format!("Task {}", task_id),
            _ => format!("Task {}", task_id),
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
        assert!(markdown.contains("* [ ]"));
        assert!(!markdown.contains("# L1-L8 Analysis Tasks")); // Should not contain complex headers
        assert!(!markdown.contains("## Task Generation Metadata")); // Should not contain metadata
        assert!(!markdown.contains("**Content**:")); // Should not contain detailed content info
        
        // Should be simple and clean
        let lines: Vec<&str> = markdown.lines().collect();
        for line in lines {
            if !line.trim().is_empty() {
                // Each non-empty line should be a checkbox or indented checkbox
                assert!(line.contains("* [ ]") || line.starts_with("  "));
            }
        }
    }

    #[test]
    fn test_extract_task_id() {
        let generator = SimpleTaskGenerator::new();
        
        // Test various title formats
        assert_eq!(generator.extract_task_id("Task Group 1 (Level 1)"), Some("1. Task 1".to_string()));
        assert_eq!(generator.extract_task_id("Analysis Group 1.1.1 (Level 3)"), Some("1.1.1. Task 1.1.1".to_string()));
        assert_eq!(generator.extract_task_id("Task Group 2.3 (Level 2)"), Some("2.3. Task 2.3".to_string()));
        
        // Test titles without extractable IDs
        assert_eq!(generator.extract_task_id("Random Title"), None);
        assert_eq!(generator.extract_task_id(""), None);
    }

    #[test]
    fn test_generate_task_description() {
        let generator = SimpleTaskGenerator::new();
        
        // Test different task ID formats
        assert_eq!(generator.generate_task_description("1"), "Task 1");
        assert_eq!(generator.generate_task_description("1.1"), "Task 1.1");
        assert_eq!(generator.generate_task_description("1.1.1"), "Task 1.1.1");
        assert_eq!(generator.generate_task_description("1.1.1.1"), "Task 1.1.1.1");
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
        assert!(content.contains("* [ ]"));
        assert!(!content.contains("# L1-L8")); // Should not contain complex headers
        
        // Should be parseable by simple markdown parsers
        let lines: Vec<&str> = content.lines().collect();
        for line in lines {
            if !line.trim().is_empty() {
                assert!(line.contains("* [ ]") || line.starts_with("  "));
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
            assert!(trimmed.starts_with("* [ ]"), "Line should start with checkbox: {}", line);
            
            // Indentation should be multiples of 2 spaces
            let indent_count = line.len() - line.trim_start().len();
            assert_eq!(indent_count % 2, 0, "Indentation should be multiples of 2 spaces: {}", line);
        }
    }
}