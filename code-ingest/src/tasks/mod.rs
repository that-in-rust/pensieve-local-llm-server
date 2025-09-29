//! Task generation module for creating structured markdown task files
//! 
//! This module implements the 7-part division algorithm for distributing analysis tasks
//! across multiple groups with Kiro-compatible numbering and markdown generation.

pub mod database_query_engine;
pub mod content_extractor;
pub mod content_generator;
pub mod hierarchical_task_divider;
pub mod hierarchical_generator;
pub mod l1l8_markdown_generator;
pub mod markdown_writer;
pub mod models;
pub mod output_directory_manager;
pub mod simple_task_generator;
pub mod task_structure_builder;

pub use database_query_engine::{DatabaseQueryEngine, TableValidation, DatabaseConnectionStats};
pub use content_extractor::{ContentExtractor, ContentTriple, RowMetadata};
pub use content_generator::{ContentGenerator, ContentFileSet, RowData, ContentStatistics, FileStatistics};
pub use output_directory_manager::{
    OutputDirectoryManager, OutputDirectoryConfig, DirectoryStatistics, FileConflict, 
    ConflictResolution, CleanupResult, OrganizationResult, ClearResult, BackupResult
};
pub use hierarchical_task_divider::{
    HierarchicalTaskDivider, TaskHierarchy, TaskLevel, HierarchicalTaskGroup, 
    AnalysisTask, AnalysisStage
};
pub use hierarchical_generator::{
    HierarchicalTaskGenerator, TaskDistribution, TaskNumbering, DatabaseRow
};
pub use l1l8_markdown_generator::L1L8MarkdownGenerator;
pub use simple_task_generator::SimpleTaskGenerator;
pub use task_structure_builder::{
    TaskStructureBuilder, EnhancedTask, PromptReference, TaskRelationship, 
    RelationshipType, TaskStructure as EnhancedTaskStructure, TaskStructureMetadata, TaskStructureStatistics
};
pub use markdown_writer::{MarkdownWriter, MarkdownConfig, MarkdownUtils};

// Re-export the new core data models
pub use models::{
    IngestionSource, ChunkMetadata, TaskHierarchy as NewTaskHierarchy, 
    Task as NewTask, TaskMetadata as NewTaskMetadata, ContentFileReference, 
    ContentFileType, GenerationConfig, GenerationConfigMetadata,
    TaskHierarchyMetadata, TaskHierarchyStatistics
};

use crate::error::{TaskError, TaskResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for task generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// SQL query to execute for generating tasks
    pub sql_query: String,
    /// Name of the output table for storing analysis results
    pub output_table: String,
    /// Path to the tasks markdown file to create
    pub tasks_file: String,
    /// Number of groups to divide tasks into (default: 7)
    pub group_count: usize,
    /// Optional chunk size for large files
    pub chunk_size: Option<usize>,
    /// Optional overlap between chunks
    pub chunk_overlap: Option<usize>,
    /// Optional prompt file path for analysis
    pub prompt_file: Option<String>,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            sql_query: String::new(),
            output_table: String::new(),
            tasks_file: String::new(),
            group_count: 7,
            chunk_size: None,
            chunk_overlap: None,
            prompt_file: None,
        }
    }
}

/// Metadata for task generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Total number of individual tasks
    pub total_tasks: usize,
    /// Number of groups created
    pub group_count: usize,
    /// SQL query used to generate tasks
    pub sql_query: String,
    /// Output table name
    pub output_table: String,
    /// Timestamp when tasks were generated
    pub generated_at: chrono::DateTime<chrono::Utc>,
    /// Optional prompt file used
    pub prompt_file: Option<String>,
}

/// Information about file chunks for large files
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkInfo {
    /// Starting line number (1-based)
    pub start_line: i32,
    /// Ending line number (inclusive)
    pub end_line: i32,
    /// Chunk number (1-based)
    pub chunk_number: i32,
    /// Total number of chunks for this file
    pub total_chunks: i32,
}

/// Individual task within a task group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier (e.g., "1.1", "2.3")
    pub id: String,
    /// Human-readable task description
    pub description: String,
    /// Optional file path this task relates to
    pub file_path: Option<String>,
    /// Optional chunk information for large files
    pub chunk_info: Option<ChunkInfo>,
    /// Additional metadata for the task
    pub metadata: HashMap<String, String>,
}

/// Group of related tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGroup {
    /// Group identifier (1-based)
    pub id: usize,
    /// Group title/description
    pub title: String,
    /// List of tasks in this group
    pub tasks: Vec<Task>,
}

/// Complete task structure with groups and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStructure {
    /// All task groups
    pub groups: Vec<TaskGroup>,
    /// Total number of individual tasks across all groups
    pub total_tasks: usize,
    /// Metadata about task generation
    pub metadata: TaskMetadata,
}

/// Task division algorithm implementation
#[derive(Debug)]
pub struct TaskDivider {
    group_count: usize,
}

impl TaskDivider {
    /// Create a new task divider with specified group count
    pub fn new(group_count: usize) -> TaskResult<Self> {
        if group_count == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Group count must be greater than 0".to_string(),
                suggestion: "Provide a group count of 1 or more".to_string(),
            });
        }
        
        Ok(Self { group_count })
    }

    /// Divide tasks into groups using mathematical distribution
    /// 
    /// Uses simple division with remainder distribution to ensure even task allocation.
    /// For example: 35 tasks ÷ 7 groups = 5 tasks per group
    /// If there's a remainder, it's distributed across the first few groups.
    pub fn divide_into_groups(&self, tasks: Vec<Task>) -> TaskResult<Vec<TaskGroup>> {
        let total_tasks = tasks.len();
        
        if total_tasks == 0 {
            return Ok(vec![]);
        }

        if total_tasks < self.group_count {
            return Err(TaskError::TaskDivisionFailed {
                total_tasks,
                groups: self.group_count,
                suggestion: format!("Reduce group count to {} or fewer, or increase the number of tasks", total_tasks),
            });
        }

        let base_tasks_per_group = total_tasks / self.group_count;
        let remainder = total_tasks % self.group_count;
        
        let mut groups = Vec::with_capacity(self.group_count);
        let mut task_index = 0;

        for group_id in 1..=self.group_count {
            // First 'remainder' groups get one extra task
            let tasks_in_this_group = if group_id <= remainder {
                base_tasks_per_group + 1
            } else {
                base_tasks_per_group
            };

            let group_tasks: Vec<Task> = tasks
                .iter()
                .skip(task_index)
                .take(tasks_in_this_group)
                .cloned()
                .collect();

            let group = TaskGroup {
                id: group_id,
                title: format!("Task Group {}", group_id),
                tasks: group_tasks,
            };

            groups.push(group);
            task_index += tasks_in_this_group;
        }

        Ok(groups)
    }

    /// Create tasks from query results
    /// 
    /// This method processes database query results and creates individual tasks
    /// for each file or chunk that needs to be analyzed.
    pub fn create_tasks_from_query_results(
        &self,
        query_results: &[QueryResultRow],
        config: &TaskConfig,
    ) -> TaskResult<Vec<Task>> {
        let mut tasks = Vec::new();
        let mut task_counter = 1;

        for row in query_results {
            // Determine if we need to chunk this file
            if let Some(chunk_size) = config.chunk_size {
                if let Some(line_count) = row.line_count {
                    if line_count > chunk_size as i32 {
                        // Create chunked tasks
                        let chunks = self.create_chunks(line_count, chunk_size, config.chunk_overlap)?;
                        
                        for (_chunk_idx, chunk_info) in chunks.into_iter().enumerate() {
                            let task = Task {
                                id: format!("{}", task_counter),
                                description: format!(
                                    "Analyze {} (chunk {}/{})",
                                    row.filepath.as_deref().unwrap_or("unknown"),
                                    chunk_info.chunk_number,
                                    chunk_info.total_chunks
                                ),
                                file_path: row.filepath.clone(),
                                chunk_info: Some(chunk_info),
                                metadata: self.create_task_metadata(row, config),
                            };
                            tasks.push(task);
                            task_counter += 1;
                        }
                        continue;
                    }
                }
            }

            // Create single task for this file
            let task = Task {
                id: format!("{}", task_counter),
                description: format!(
                    "Analyze {}",
                    row.filepath.as_deref().unwrap_or("unknown file")
                ),
                file_path: row.filepath.clone(),
                chunk_info: None,
                metadata: self.create_task_metadata(row, config),
            };
            tasks.push(task);
            task_counter += 1;
        }

        Ok(tasks)
    }

    /// Create chunk information for large files
    fn create_chunks(
        &self,
        total_lines: i32,
        chunk_size: usize,
        overlap: Option<usize>,
    ) -> TaskResult<Vec<ChunkInfo>> {
        let overlap = overlap.unwrap_or(0);
        let chunk_size = chunk_size as i32;
        let overlap = overlap as i32;

        if chunk_size <= 0 {
            return Err(TaskError::ChunkAnalysisFailed {
                cause: "Chunk size must be positive".to_string(),
                suggestion: "Use a chunk size greater than 0".to_string(),
            });
        }

        if overlap >= chunk_size {
            return Err(TaskError::ChunkAnalysisFailed {
                cause: "Overlap must be less than chunk size".to_string(),
                suggestion: "Reduce overlap to be less than chunk size".to_string(),
            });
        }

        let mut chunks = Vec::new();
        let mut current_line = 1;
        let mut chunk_number = 1;

        while current_line <= total_lines {
            let end_line = std::cmp::min(current_line + chunk_size - 1, total_lines);
            
            let chunk = ChunkInfo {
                start_line: current_line,
                end_line,
                chunk_number,
                total_chunks: 0, // Will be updated after all chunks are created
            };
            
            chunks.push(chunk);
            
            // Move to next chunk, accounting for overlap
            current_line = end_line + 1 - overlap;
            chunk_number += 1;

            // Prevent infinite loop if overlap is too large
            if current_line <= chunks.last().unwrap().start_line {
                break;
            }
        }

        // Update total_chunks for all chunks
        let total_chunks = chunks.len() as i32;
        for chunk in &mut chunks {
            chunk.total_chunks = total_chunks;
        }

        Ok(chunks)
    }

    /// Create metadata for a task based on query result row
    fn create_task_metadata(&self, row: &QueryResultRow, config: &TaskConfig) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        if let Some(filepath) = &row.filepath {
            metadata.insert("filepath".to_string(), filepath.clone());
        }
        
        if let Some(extension) = &row.extension {
            metadata.insert("extension".to_string(), extension.clone());
        }
        
        if let Some(file_type) = &row.file_type {
            metadata.insert("file_type".to_string(), file_type.clone());
        }
        
        if let Some(line_count) = row.line_count {
            metadata.insert("line_count".to_string(), line_count.to_string());
        }
        
        if let Some(word_count) = row.word_count {
            metadata.insert("word_count".to_string(), word_count.to_string());
        }
        
        metadata.insert("output_table".to_string(), config.output_table.clone());
        
        if let Some(prompt_file) = &config.prompt_file {
            metadata.insert("prompt_file".to_string(), prompt_file.clone());
        }
        
        metadata
    }
}

/// Represents a row from a database query result
/// This structure matches the typical columns from INGEST_* tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResultRow {
    pub file_id: Option<i64>,
    pub filepath: Option<String>,
    pub filename: Option<String>,
    pub extension: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub line_count: Option<i32>,
    pub word_count: Option<i32>,
    pub token_count: Option<i32>,
    pub content_text: Option<String>,
    pub file_type: Option<String>,
    pub conversion_command: Option<String>,
    pub relative_path: Option<String>,
    pub absolute_path: Option<String>,
}

/// Markdown generator for creating Kiro-compatible task files
pub struct MarkdownGenerator {
    /// Template for task descriptions
    #[allow(dead_code)]
    task_template: Option<String>,
}

impl MarkdownGenerator {
    /// Create a new markdown generator
    pub fn new() -> Self {
        Self {
            task_template: None,
        }
    }

    /// Create a markdown generator with a custom task template
    pub fn with_template(template: String) -> Self {
        Self {
            task_template: Some(template),
        }
    }

    /// Generate markdown content for a task structure
    /// 
    /// Creates Kiro-compatible markdown with proper numbering:
    /// - [ ] 1. Task Group 1
    /// - [ ] 1.1 First sub-task
    /// - [ ] 1.2 Second sub-task
    /// - [ ] 2. Task Group 2
    /// - [ ] 2.1 First sub-task
    pub fn generate_markdown(&self, task_structure: &TaskStructure) -> TaskResult<String> {
        let mut markdown = String::new();
        
        // Add header
        markdown.push_str("# Implementation Tasks\n\n");
        
        // Add metadata section
        self.add_metadata_section(&mut markdown, &task_structure.metadata)?;
        
        // Add task overview
        self.add_task_overview(&mut markdown, task_structure)?;
        
        // Add task groups
        for group in &task_structure.groups {
            self.add_task_group(&mut markdown, group)?;
        }
        
        // Add footer with instructions
        self.add_footer(&mut markdown, task_structure)?;
        
        Ok(markdown)
    }

    /// Write markdown content to a file
    pub async fn write_to_file(&self, task_structure: &TaskStructure, file_path: &str) -> TaskResult<()> {
        let markdown = self.generate_markdown(task_structure)?;
        
        // Ensure parent directory exists
        if let Some(parent) = Path::new(file_path).parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: file_path.to_string(),
                    cause: format!("Failed to create parent directory: {}", e),
                    suggestion: "Check directory permissions and available disk space".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }
        
        tokio::fs::write(file_path, markdown).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: file_path.to_string(),
                cause: e.to_string(),
                suggestion: "Check file permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;
        
        Ok(())
    }

    /// Add metadata section to markdown
    fn add_metadata_section(&self, markdown: &mut String, metadata: &TaskMetadata) -> TaskResult<()> {
        markdown.push_str("## Task Generation Metadata\n\n");
        markdown.push_str(&format!("- **Generated At**: {}\n", metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        markdown.push_str(&format!("- **Total Tasks**: {}\n", metadata.total_tasks));
        markdown.push_str(&format!("- **Task Groups**: {}\n", metadata.group_count));
        markdown.push_str(&format!("- **Output Table**: `{}`\n", metadata.output_table));
        
        if let Some(prompt_file) = &metadata.prompt_file {
            markdown.push_str(&format!("- **Prompt File**: `{}`\n", prompt_file));
        }
        
        markdown.push_str(&format!("- **SQL Query**: \n```sql\n{}\n```\n\n", metadata.sql_query));
        
        Ok(())
    }

    /// Add task overview section
    fn add_task_overview(&self, markdown: &mut String, task_structure: &TaskStructure) -> TaskResult<()> {
        markdown.push_str("## Task Overview\n\n");
        markdown.push_str(&format!("This document contains {} analysis tasks divided into {} groups for systematic processing.\n\n", 
            task_structure.total_tasks, task_structure.groups.len()));
        
        markdown.push_str("### Task Distribution\n\n");
        for (i, group) in task_structure.groups.iter().enumerate() {
            markdown.push_str(&format!("- **Group {}**: {} tasks\n", i + 1, group.tasks.len()));
        }
        markdown.push_str("\n");
        
        Ok(())
    }

    /// Add a task group to markdown with Kiro-compatible numbering
    fn add_task_group(&self, markdown: &mut String, group: &TaskGroup) -> TaskResult<()> {
        // Main group task (e.g., "- [ ] 1. Task Group 1")
        markdown.push_str(&format!("- [ ] {}. {}\n", group.id, group.title));
        
        // Add group description if tasks have common metadata
        if let Some(description) = self.generate_group_description(group) {
            markdown.push_str(&format!("  - {}\n", description));
        }
        
        // Sub-tasks (e.g., "- [ ] 1.1", "- [ ] 1.2")
        for (task_idx, task) in group.tasks.iter().enumerate() {
            let task_number = format!("{}.{}", group.id, task_idx + 1);
            markdown.push_str(&format!("- [ ] {} {}\n", task_number, task.description));
            
            // Add task details as sub-bullets
            self.add_task_details(markdown, task, &task_number)?;
        }
        
        markdown.push_str("\n");
        Ok(())
    }

    /// Generate a description for a task group based on common metadata
    fn generate_group_description(&self, group: &TaskGroup) -> Option<String> {
        if group.tasks.is_empty() {
            return None;
        }

        // Check if all tasks in the group have the same file type
        let first_file_type = group.tasks[0].metadata.get("file_type");
        let all_same_type = group.tasks.iter()
            .all(|task| task.metadata.get("file_type") == first_file_type);

        if all_same_type {
            if let Some(file_type) = first_file_type {
                return Some(format!("Analyze {} files in this group", file_type));
            }
        }

        // Check if all tasks are for the same directory
        let first_dir = group.tasks[0].file_path.as_ref()
            .and_then(|path| Path::new(path).parent())
            .map(|p| p.to_string_lossy().to_string());
        
        if let Some(dir) = first_dir {
            let all_same_dir = group.tasks.iter()
                .all(|task| {
                    task.file_path.as_ref()
                        .and_then(|path| Path::new(path).parent())
                        .map(|p| p.to_string_lossy().to_string())
                        .as_ref() == Some(&dir)
                });
            
            if all_same_dir {
                return Some(format!("Process files in directory: {}", dir));
            }
        }

        Some(format!("Process {} files", group.tasks.len()))
    }

    /// Add task details as sub-bullets
    fn add_task_details(&self, markdown: &mut String, task: &Task, task_number: &str) -> TaskResult<()> {
        // Add file path if available
        if let Some(file_path) = &task.file_path {
            markdown.push_str(&format!("  - **File**: `{}`\n", file_path));
        }

        // Add chunk information if available
        if let Some(chunk_info) = &task.chunk_info {
            markdown.push_str(&format!("  - **Chunk**: {}/{} (lines {}-{})\n", 
                chunk_info.chunk_number, 
                chunk_info.total_chunks,
                chunk_info.start_line,
                chunk_info.end_line
            ));
        }

        // Add relevant metadata
        if let Some(extension) = task.metadata.get("extension") {
            markdown.push_str(&format!("  - **Type**: {}\n", extension));
        }

        if let Some(line_count) = task.metadata.get("line_count") {
            markdown.push_str(&format!("  - **Lines**: {}\n", line_count));
        }

        // Add task ID for reference
        markdown.push_str(&format!("  - **Task ID**: {}\n", task_number));

        Ok(())
    }

    /// Add footer with processing instructions
    fn add_footer(&self, markdown: &mut String, task_structure: &TaskStructure) -> TaskResult<()> {
        markdown.push_str("## Processing Instructions\n\n");
        markdown.push_str("### How to Execute These Tasks\n\n");
        markdown.push_str("1. **Start with Group 1**: Begin with the first task group and work through each sub-task sequentially\n");
        markdown.push_str("2. **Complete Sub-tasks First**: For each main task (e.g., \"1. Task Group 1\"), complete all sub-tasks (1.1, 1.2, etc.) before moving to the next group\n");
        markdown.push_str("3. **Use Provided Context**: Each task references specific files and metadata to guide your analysis\n");
        markdown.push_str("4. **Store Results**: Analysis results should be stored in the specified output table\n\n");
        
        markdown.push_str("### Task Execution Commands\n\n");
        markdown.push_str("```bash\n");
        markdown.push_str("# Query data for analysis\n");
        markdown.push_str(&format!("code-ingest query-prepare \"{}\" --output-table {} --temp-path /tmp/analysis.txt\n", 
            task_structure.metadata.sql_query, task_structure.metadata.output_table));
        markdown.push_str("\n# Store analysis results\n");
        markdown.push_str(&format!("code-ingest store-result --output-table {} --result-file /tmp/result.txt\n", 
            task_structure.metadata.output_table));
        markdown.push_str("```\n\n");
        
        markdown.push_str("### Notes\n\n");
        markdown.push_str(&format!("- Total files to analyze: {}\n", task_structure.total_tasks));
        markdown.push_str(&format!("- Organized into {} groups for systematic processing\n", task_structure.groups.len()));
        markdown.push_str("- Each task includes file path, type, and relevant metadata\n");
        markdown.push_str("- Chunk information provided for large files\n\n");
        
        Ok(())
    }
}

impl Default for MarkdownGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tasks(count: usize) -> Vec<Task> {
        (1..=count)
            .map(|i| Task {
                id: format!("{}", i),
                description: format!("Test task {}", i),
                file_path: Some(format!("test_file_{}.rs", i)),
                chunk_info: None,
                metadata: HashMap::new(),
            })
            .collect()
    }

    fn create_test_query_results(count: usize) -> Vec<QueryResultRow> {
        (1..=count)
            .map(|i| QueryResultRow {
                file_id: Some(i as i64),
                filepath: Some(format!("src/test_{}.rs", i)),
                filename: Some(format!("test_{}.rs", i)),
                extension: Some("rs".to_string()),
                file_size_bytes: Some(1024 * i as i64),
                line_count: Some(100 * i as i32),
                word_count: Some(500 * i as i32),
                token_count: Some(750 * i as i32),
                content_text: Some(format!("Test content for file {}", i)),
                file_type: Some("direct_text".to_string()),
                conversion_command: None,
                relative_path: Some(format!("src/test_{}.rs", i)),
                absolute_path: Some(format!("/project/src/test_{}.rs", i)),
            })
            .collect()
    }

    #[test]
    fn test_task_divider_creation() {
        let divider = TaskDivider::new(7).unwrap();
        assert_eq!(divider.group_count, 7);

        // Test invalid group count
        let result = TaskDivider::new(0);
        assert!(result.is_err());
        match result.unwrap_err() {
            TaskError::InvalidTaskConfiguration { cause, suggestion: _ } => {
                assert!(cause.contains("Group count must be greater than 0"));
            }
            _ => panic!("Expected InvalidTaskConfiguration error"),
        }
    }

    #[test]
    fn test_even_task_division() {
        let divider = TaskDivider::new(7).unwrap();
        let tasks = create_test_tasks(35); // 35 ÷ 7 = 5 tasks per group

        let groups = divider.divide_into_groups(tasks).unwrap();

        assert_eq!(groups.len(), 7);
        for group in &groups {
            assert_eq!(group.tasks.len(), 5);
            assert!(group.title.contains(&format!("Task Group {}", group.id)));
        }

        // Verify all tasks are included
        let total_tasks: usize = groups.iter().map(|g| g.tasks.len()).sum();
        assert_eq!(total_tasks, 35);
    }

    #[test]
    fn test_uneven_task_division() {
        let divider = TaskDivider::new(7).unwrap();
        let tasks = create_test_tasks(37); // 37 ÷ 7 = 5 remainder 2

        let groups = divider.divide_into_groups(tasks).unwrap();

        assert_eq!(groups.len(), 7);
        
        // First 2 groups should have 6 tasks (5 + 1 from remainder)
        assert_eq!(groups[0].tasks.len(), 6);
        assert_eq!(groups[1].tasks.len(), 6);
        
        // Remaining 5 groups should have 5 tasks each
        for i in 2..7 {
            assert_eq!(groups[i].tasks.len(), 5);
        }

        // Verify all tasks are included
        let total_tasks: usize = groups.iter().map(|g| g.tasks.len()).sum();
        assert_eq!(total_tasks, 37);
    }

    #[test]
    fn test_insufficient_tasks_for_groups() {
        let divider = TaskDivider::new(7).unwrap();
        let tasks = create_test_tasks(5); // Only 5 tasks for 7 groups

        let result = divider.divide_into_groups(tasks);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TaskError::TaskDivisionFailed { total_tasks, groups, suggestion: _ } => {
                assert_eq!(total_tasks, 5);
                assert_eq!(groups, 7);
            }
            _ => panic!("Expected TaskDivisionFailed error"),
        }
    }

    #[test]
    fn test_empty_task_list() {
        let divider = TaskDivider::new(7).unwrap();
        let tasks = Vec::new();

        let groups = divider.divide_into_groups(tasks).unwrap();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_create_tasks_from_query_results() {
        let divider = TaskDivider::new(7).unwrap();
        let query_results = create_test_query_results(10);
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

        assert_eq!(tasks.len(), 10);
        
        for (i, task) in tasks.iter().enumerate() {
            assert_eq!(task.id, format!("{}", i + 1));
            assert!(task.description.contains(&format!("src/test_{}.rs", i + 1)));
            assert_eq!(task.file_path, Some(format!("src/test_{}.rs", i + 1)));
            assert!(task.chunk_info.is_none());
            
            // Check metadata
            assert_eq!(task.metadata.get("output_table").unwrap(), "QUERYRESULT_test");
            assert_eq!(task.metadata.get("prompt_file").unwrap(), "analyze.md");
            assert_eq!(task.metadata.get("extension").unwrap(), "rs");
            assert_eq!(task.metadata.get("file_type").unwrap(), "direct_text");
        }
    }

    #[test]
    fn test_create_tasks_with_chunking() {
        let divider = TaskDivider::new(7).unwrap();
        
        // Create a query result with a large file (1000 lines)
        let mut query_results = create_test_query_results(1);
        query_results[0].line_count = Some(1000);
        query_results[0].filepath = Some("large_file.rs".to_string());
        
        let config = TaskConfig {
            sql_query: "SELECT * FROM test_table".to_string(),
            output_table: "QUERYRESULT_test".to_string(),
            tasks_file: "/tmp/tasks.md".to_string(),
            group_count: 7,
            chunk_size: Some(300), // Chunk size of 300 lines
            chunk_overlap: Some(50), // 50 line overlap
            prompt_file: None,
        };

        let tasks = divider.create_tasks_from_query_results(&query_results, &config).unwrap();

        // Should create multiple tasks for the chunked file
        assert!(tasks.len() > 1);
        
        // Check that all tasks have chunk info
        for task in &tasks {
            assert!(task.chunk_info.is_some());
            let chunk_info = task.chunk_info.as_ref().unwrap();
            assert!(chunk_info.start_line >= 1);
            assert!(chunk_info.end_line <= 1000);
            assert!(chunk_info.chunk_number >= 1);
            assert!(chunk_info.total_chunks > 1);
            assert!(task.description.contains("chunk"));
        }
    }

    #[test]
    fn test_create_chunks() {
        let divider = TaskDivider::new(7).unwrap();
        
        // Test chunking 1000 lines with 300 line chunks and 50 line overlap
        let chunks = divider.create_chunks(1000, 300, Some(50)).unwrap();
        
        assert!(chunks.len() > 1);
        
        // Check first chunk
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 300);
        assert_eq!(chunks[0].chunk_number, 1);
        
        // Check second chunk (should start at 251 due to 50 line overlap)
        assert_eq!(chunks[1].start_line, 251); // 300 + 1 - 50 = 251
        assert_eq!(chunks[1].chunk_number, 2);
        
        // All chunks should have the same total_chunks value
        let total_chunks = chunks[0].total_chunks;
        for chunk in &chunks {
            assert_eq!(chunk.total_chunks, total_chunks);
        }
        
        // Last chunk should end at or before line 1000
        let last_chunk = chunks.last().unwrap();
        assert!(last_chunk.end_line <= 1000);
    }

    #[test]
    fn test_create_chunks_edge_cases() {
        let divider = TaskDivider::new(7).unwrap();
        
        // Test invalid chunk size
        let result = divider.create_chunks(1000, 0, None);
        assert!(result.is_err());
        
        // Test overlap >= chunk size
        let result = divider.create_chunks(1000, 100, Some(100));
        assert!(result.is_err());
        
        // Test small file that doesn't need chunking
        let chunks = divider.create_chunks(50, 100, None).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 50);
        assert_eq!(chunks[0].total_chunks, 1);
    }

    #[test]
    fn test_task_config_default() {
        let config = TaskConfig::default();
        assert_eq!(config.group_count, 7);
        assert!(config.sql_query.is_empty());
        assert!(config.output_table.is_empty());
        assert!(config.tasks_file.is_empty());
        assert!(config.chunk_size.is_none());
        assert!(config.chunk_overlap.is_none());
        assert!(config.prompt_file.is_none());
    }

    #[test]
    fn test_chunk_info_equality() {
        let chunk1 = ChunkInfo {
            start_line: 1,
            end_line: 100,
            chunk_number: 1,
            total_chunks: 3,
        };
        
        let chunk2 = ChunkInfo {
            start_line: 1,
            end_line: 100,
            chunk_number: 1,
            total_chunks: 3,
        };
        
        let chunk3 = ChunkInfo {
            start_line: 1,
            end_line: 100,
            chunk_number: 2,
            total_chunks: 3,
        };
        
        assert_eq!(chunk1, chunk2);
        assert_ne!(chunk1, chunk3);
    }

    #[test]
    fn test_markdown_generation() {
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
        
        // Check that markdown contains expected elements
        assert!(markdown.contains("# Implementation Tasks"));
        assert!(markdown.contains("## Task Generation Metadata"));
        assert!(markdown.contains("## Task Overview"));
        assert!(markdown.contains("- [ ] 1. Task Group 1"));
        assert!(markdown.contains("- [ ] 1.1 Analyze main.rs"));
        assert!(markdown.contains("- [ ] 2. Task Group 2"));
        assert!(markdown.contains("- [ ] 2.1 Analyze lib.rs"));
        assert!(markdown.contains("**File**: `src/main.rs`"));
        assert!(markdown.contains("**File**: `src/lib.rs`"));
        assert!(markdown.contains("**Type**: rs"));
        assert!(markdown.contains("**Lines**: 100"));
        assert!(markdown.contains("**Lines**: 200"));
        assert!(markdown.contains("## Processing Instructions"));
        assert!(markdown.contains("QUERYRESULT_test"));
    }

    #[test]
    fn test_markdown_generation_with_chunks() {
        let generator = MarkdownGenerator::new();
        
        // Create test task with chunk info
        let task = Task {
            id: "1".to_string(),
            description: "Analyze large_file.rs (chunk 1/3)".to_string(),
            file_path: Some("src/large_file.rs".to_string()),
            chunk_info: Some(ChunkInfo {
                start_line: 1,
                end_line: 300,
                chunk_number: 1,
                total_chunks: 3,
            }),
            metadata: {
                let mut map = HashMap::new();
                map.insert("extension".to_string(), "rs".to_string());
                map.insert("file_type".to_string(), "direct_text".to_string());
                map.insert("line_count".to_string(), "1000".to_string());
                map
            },
        };
        
        let group = TaskGroup {
            id: 1,
            title: "Task Group 1".to_string(),
            tasks: vec![task],
        };
        
        let task_structure = TaskStructure {
            groups: vec![group],
            total_tasks: 1,
            metadata: TaskMetadata {
                total_tasks: 1,
                group_count: 1,
                sql_query: "SELECT * FROM test_table".to_string(),
                output_table: "QUERYRESULT_test".to_string(),
                generated_at: chrono::Utc::now(),
                prompt_file: None,
            },
        };
        
        let markdown = generator.generate_markdown(&task_structure).unwrap();
        
        // Check chunk information is included
        assert!(markdown.contains("**Chunk**: 1/3 (lines 1-300)"));
        assert!(markdown.contains("**File**: `src/large_file.rs`"));
        assert!(markdown.contains("- [ ] 1.1 Analyze large_file.rs (chunk 1/3)"));
    }

    #[test]
    fn test_group_description_generation() {
        let generator = MarkdownGenerator::new();
        
        // Test group with same file type
        let tasks = vec![
            Task {
                id: "1".to_string(),
                description: "Analyze file1.rs".to_string(),
                file_path: Some("src/file1.rs".to_string()),
                chunk_info: None,
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("file_type".to_string(), "direct_text".to_string());
                    map
                },
            },
            Task {
                id: "2".to_string(),
                description: "Analyze file2.rs".to_string(),
                file_path: Some("src/file2.rs".to_string()),
                chunk_info: None,
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("file_type".to_string(), "direct_text".to_string());
                    map
                },
            },
        ];
        
        let group = TaskGroup {
            id: 1,
            title: "Task Group 1".to_string(),
            tasks,
        };
        
        let description = generator.generate_group_description(&group);
        assert_eq!(description, Some("Analyze direct_text files in this group".to_string()));
        
        // Test group with same directory
        let tasks = vec![
            Task {
                id: "1".to_string(),
                description: "Analyze file1.rs".to_string(),
                file_path: Some("src/module/file1.rs".to_string()),
                chunk_info: None,
                metadata: HashMap::new(),
            },
            Task {
                id: "2".to_string(),
                description: "Analyze file2.rs".to_string(),
                file_path: Some("src/module/file2.rs".to_string()),
                chunk_info: None,
                metadata: HashMap::new(),
            },
        ];
        
        let group = TaskGroup {
            id: 1,
            title: "Task Group 1".to_string(),
            tasks,
        };
        
        let description = generator.generate_group_description(&group);
        assert_eq!(description, Some("Process files in directory: src/module".to_string()));
    }

    #[tokio::test]
    async fn test_write_to_file() {
        use tempfile::TempDir;
        
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_tasks.md");
        
        let generator = MarkdownGenerator::new();
        
        let task_structure = TaskStructure {
            groups: vec![TaskGroup {
                id: 1,
                title: "Test Group".to_string(),
                tasks: vec![Task {
                    id: "1".to_string(),
                    description: "Test task".to_string(),
                    file_path: Some("test.rs".to_string()),
                    chunk_info: None,
                    metadata: HashMap::new(),
                }],
            }],
            total_tasks: 1,
            metadata: TaskMetadata {
                total_tasks: 1,
                group_count: 1,
                sql_query: "SELECT * FROM test".to_string(),
                output_table: "QUERYRESULT_test".to_string(),
                generated_at: chrono::Utc::now(),
                prompt_file: None,
            },
        };
        
        generator.write_to_file(&task_structure, file_path.to_str().unwrap()).await.unwrap();
        
        // Verify file was created and contains expected content
        assert!(file_path.exists());
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert!(content.contains("# Implementation Tasks"));
        assert!(content.contains("- [ ] 1. Test Group"));
        assert!(content.contains("- [ ] 1.1 Test task"));
    }

    #[test]
    fn test_markdown_generator_with_template() {
        let template = "Custom template: {description}".to_string();
        let generator = MarkdownGenerator::with_template(template);
        
        // The template is stored but not used in current implementation
        // This test verifies the constructor works
        assert!(generator.task_template.is_some());
    }

    #[test]
    fn test_task_metadata_creation() {
        let divider = TaskDivider::new(7).unwrap();
        let query_results = create_test_query_results(1);
        let config = TaskConfig {
            sql_query: "SELECT * FROM test".to_string(),
            output_table: "QUERYRESULT_test".to_string(),
            tasks_file: "/tmp/tasks.md".to_string(),
            group_count: 7,
            chunk_size: None,
            chunk_overlap: None,
            prompt_file: Some("prompt.md".to_string()),
        };
        
        let metadata = divider.create_task_metadata(&query_results[0], &config);
        
        assert_eq!(metadata.get("filepath").unwrap(), "src/test_1.rs");
        assert_eq!(metadata.get("extension").unwrap(), "rs");
        assert_eq!(metadata.get("file_type").unwrap(), "direct_text");
        assert_eq!(metadata.get("line_count").unwrap(), "100");
        assert_eq!(metadata.get("word_count").unwrap(), "500");
        assert_eq!(metadata.get("output_table").unwrap(), "QUERYRESULT_test");
        assert_eq!(metadata.get("prompt_file").unwrap(), "prompt.md");
    }
}
