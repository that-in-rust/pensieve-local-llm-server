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

    /// Generate simple checkbox task list following the exact format specified
    ///
    /// # Arguments
    /// * `table_name` - Name of the source table
    /// * `chunk_size` - Chunk size in LOC for file naming
    /// * `prompt_file` - Path to the prompt file
    /// * `total_rows` - Total number of rows to process
    ///
    /// # Returns
    /// * `TaskResult<String>` - Generated simple task list content
    pub async fn generate_simple_tasks(
        &self,
        table_name: &str,
        chunk_size: Option<usize>,
        prompt_file: &str,
        total_rows: usize,
    ) -> TaskResult<String> {
        debug!("Generating simple checkbox task list for {} with {} rows", table_name, total_rows);

        let mut output = String::new();
        let chunk_suffix = if let Some(size) = chunk_size {
            format!("_{}", size)
        } else {
            String::new()
        };

        // Process each row up to max_tasks limit
        let max_rows = if let Some(max) = self.max_tasks {
            std::cmp::min(max, total_rows)
        } else {
            total_rows
        };

        for row_num in 1..=max_rows {
            // Skip rows based on offset
            if row_num <= self.offset {
                continue;
            }

            output.push_str(&format!(
                "- [ ] {}. Analyze {} row {}\n",
                row_num, table_name, row_num
            ));
            
            output.push_str(&format!(
                "  - **Content**: `.raw_data_202509/{}{}_Content.txt` as A + `.raw_data_202509/{}_Content_L1.txt` as B + `.raw_data_202509/{}_Content_L2.txt` as C\n",
                table_name, chunk_suffix, table_name, table_name
            ));
            
            output.push_str(&format!(
                "  - **Prompt**: `{}` where you try to find insights of A alone ; A in context of B ; B in context of C ; A in context B & C\n",
                prompt_file
            ));
            
            output.push_str(&format!(
                "  - **Output**: `gringotts/WorkArea/{}{}_{}.md`\n\n",
                table_name, chunk_suffix, row_num
            ));
        }

        // Add note if there are more rows
        if total_rows > max_rows {
            output.push_str(&format!(
                "<!-- Note: {} more rows available. Use --offset {} to continue -->\n",
                total_rows - max_rows,
                max_rows
            ));
        }

        info!("Generated simple task list with {} tasks", max_rows);
        Ok(output)
    }

    /// Write simple task list to a file
    ///
    /// # Arguments
    /// * `table_name` - Name of the source table
    /// * `chunk_size` - Chunk size in LOC for file naming
    /// * `prompt_file` - Path to the prompt file
    /// * `total_rows` - Total number of rows to process
    /// * `output_file` - Path to the output file
    ///
    /// # Returns
    /// * `TaskResult<()>` - Success or error
    pub async fn write_simple_tasks_to_file(
        &self,
        table_name: &str,
        chunk_size: Option<usize>,
        prompt_file: &str,
        total_rows: usize,
        output_file: &Path,
    ) -> TaskResult<()> {
        debug!("Writing simple task list to file: {}", output_file.display());

        // Generate task content
        let tasks = self.generate_simple_tasks(table_name, chunk_size, prompt_file, total_rows).await?;

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

        // Write tasks to file
        tokio::fs::write(output_file, tasks).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: output_file.display().to_string(),
                cause: e.to_string(),
                suggestion: "Check file permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        info!("Successfully wrote simple task list to: {}", output_file.display());
        Ok(())
    }



    /// Add hierarchy tasks to markdown recursively (legacy method)
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

    /// Count total tasks in a group (including sub-groups)
    fn count_tasks_in_group(&self, group: &HierarchicalTaskGroup) -> usize {
        let mut count = 0;
        
        // Count the group itself as a task if it has a meaningful title
        let clean_title = self.clean_group_title(&group.title);
        if !clean_title.is_empty() {
            count += 1;
        }
        
        // Count individual tasks
        count += group.tasks.len();
        
        // Count tasks in sub-groups recursively
        for sub_group in &group.sub_groups {
            count += self.count_tasks_in_group(sub_group);
        }
        
        count
    }

    /// Add group tasks with offset and limit support
    fn add_group_tasks_with_offset_and_limit<'a>(
        &'a self,
        markdown: &'a mut String,
        group: &'a HierarchicalTaskGroup,
        depth: usize,
        task_count: &'a mut usize,
        tasks_processed: &'a mut usize,
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

        // Handle group as a task
        let clean_title = self.clean_group_title(&group.title);
        if !clean_title.is_empty() {
            if *tasks_processed >= self.offset {
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
            *tasks_processed += 1;
        }

        // Handle individual tasks
        for task in &group.tasks {
            if let Some(max) = self.max_tasks {
                if *task_count >= max {
                    break;
                }
            }
            
            if *tasks_processed >= self.offset {
                let task_indent = "  ".repeat(depth + 1);
                let task_description = self.format_task_description(task);
                markdown.push_str(&format!("{}- [ ] {}\n", task_indent, task_description));
                *task_count += 1;
                tasks_added += 1;
            }
            *tasks_processed += 1;
        }

        // Handle sub-groups recursively
        for sub_group in &group.sub_groups {
            if let Some(max) = self.max_tasks {
                if *task_count >= max {
                    break;
                }
            }
            
            let sub_tasks_added = self.add_group_tasks_with_offset_and_limit(
                markdown, sub_group, depth + 1, task_count, tasks_processed
            ).await?;
            tasks_added += sub_tasks_added;
        }

        // Add spacing after root-level groups
        if depth == 0 && tasks_added > 0 {
            markdown.push('\n');
        }

        Ok(tasks_added)
        })
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
    async fn test_generate_simple_tasks() {
        let generator = SimpleTaskGenerator::new();
        let tasks = generator.generate_simple_tasks("TEST_TABLE", Some(500), ".kiro/steering/test.md", 3).await.unwrap();
        
        // Verify simple checkbox format
        assert!(tasks.contains("- [ ] 1. Analyze TEST_TABLE row 1"));
        assert!(tasks.contains("- [ ] 2. Analyze TEST_TABLE row 2"));
        assert!(tasks.contains("- [ ] 3. Analyze TEST_TABLE row 3"));
        assert!(tasks.contains("**Content**:"));
        assert!(tasks.contains("**Prompt**:"));
        assert!(tasks.contains("**Output**:"));
        assert!(tasks.contains(".raw_data_202509/TEST_TABLE_500_Content.txt"));
        assert!(tasks.contains("gringotts/WorkArea/TEST_TABLE_500_"));
        
        // Should be simple checkbox list with proper structure
        let lines: Vec<&str> = tasks.lines().collect();
        let task_lines: Vec<&str> = lines.iter()
            .filter(|line| line.trim().starts_with("- [ ]"))
            .cloned()
            .collect();
        
        // Should have 3 main task lines
        assert_eq!(task_lines.len(), 3);
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
    async fn test_write_simple_tasks_to_file() {
        let generator = SimpleTaskGenerator::new();
        
        // Create a temporary file path
        let temp_dir = tempfile::tempdir().unwrap();
        let output_file = temp_dir.path().join("simple_tasks.md");
        
        // Write the task structure
        generator.write_simple_tasks_to_file("TEST_TABLE", Some(500), ".kiro/steering/test.md", 3, &output_file).await.unwrap();
        
        // Verify the file was created and contains expected content
        assert!(output_file.exists());
        let content = tokio::fs::read_to_string(&output_file).await.unwrap();
        
        // Should contain simple checkbox format
        assert!(content.contains("- [ ] 1. Analyze TEST_TABLE row 1"));
        assert!(content.contains("**Content**:"));
        assert!(content.contains("**Prompt**:"));
        assert!(content.contains("**Output**:"));
        assert!(content.contains(".raw_data_202509/TEST_TABLE_500_Content.txt"));
        
        // Should be simple checkbox list
        let lines: Vec<&str> = content.lines().collect();
        let task_lines: Vec<&str> = lines.iter()
            .filter(|line| line.trim().starts_with("- [ ]"))
            .cloned()
            .collect();
        
        assert!(!task_lines.is_empty(), "Should have checkbox task lines");
    }

    #[tokio::test]
    async fn test_format_matches_simple_reference() {
        let generator = SimpleTaskGenerator::new();
        let tasks = generator.generate_simple_tasks("TEST_TABLE", Some(500), ".kiro/steering/test.md", 2).await.unwrap();
        
        // The format should match the exact specification:
        // - [ ] 1. Analyze TEST_TABLE row 1
        //   - **Content**: .raw_data_202509/TEST_TABLE_500_Content.txt as A + ...
        //   - **Prompt**: .kiro/steering/test.md where you try to find insights...
        //   - **Output**: gringotts/WorkArea/TEST_TABLE_500_1.md
        
        // Check for proper checkbox format
        let lines: Vec<&str> = tasks.lines().filter(|line| !line.trim().is_empty()).collect();
        let mut found_checkbox_tasks = false;
        let mut found_content_lines = false;
        let mut found_prompt_lines = false;
        let mut found_output_lines = false;
        
        for line in lines {
            if line.trim().starts_with("- [ ]") && line.contains("Analyze TEST_TABLE row") {
                found_checkbox_tasks = true;
            }
            if line.trim().starts_with("- **Content**:") {
                found_content_lines = true;
            }
            if line.trim().starts_with("- **Prompt**:") {
                found_prompt_lines = true;
            }
            if line.trim().starts_with("- **Output**:") {
                found_output_lines = true;
            }
        }
        
        assert!(found_checkbox_tasks, "Should have found checkbox task lines");
        assert!(found_content_lines, "Should have found content lines");
        assert!(found_prompt_lines, "Should have found prompt lines");
        assert!(found_output_lines, "Should have found output lines");
    }
}