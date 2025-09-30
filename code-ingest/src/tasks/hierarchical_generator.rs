//! [DEPRECATED] Hierarchical task generation and distribution algorithms
//! 
//! ⚠️  DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
//! Please use `chunk_level_task_generator` instead for simpler, more maintainable task generation.
//! 
//! This module implements the core algorithms for distributing database rows
//! across hierarchical task structures with proper numbering and balancing.
//! 
//! Migration path: Use `ChunkLevelTaskGenerator` for file-level or chunk-level task generation.

use crate::error::{TaskError, TaskResult};
use crate::tasks::models::{
    Task, TaskHierarchy, GenerationConfig, ContentFileReference, ContentFileType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Database row data for task generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseRow {
    /// Row ID from database
    pub row_id: i64,
    /// File path
    pub filepath: Option<String>,
    /// Filename
    pub filename: Option<String>,
    /// File extension
    pub extension: Option<String>,
    /// Line count
    pub line_count: Option<i32>,
    /// Content text
    pub content: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Task distribution statistics
#[derive(Debug, Clone, PartialEq)]
pub struct TaskDistribution {
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of levels in hierarchy
    pub levels: u8,
    /// Number of groups per level
    pub groups_per_level: u8,
    /// Tasks per group at each level
    pub tasks_per_level: Vec<usize>,
    /// Remainder tasks distributed across first groups
    pub remainder_distribution: Vec<usize>,
}

impl TaskDistribution {
    /// Calculate optimal task distribution
    pub fn calculate(total_tasks: usize, levels: u8, groups_per_level: u8) -> TaskResult<Self> {
        if total_tasks == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Total tasks must be greater than 0".to_string(),
                suggestion: "Provide at least 1 task to distribute".to_string(),
            });
        }

        if levels == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Levels must be greater than 0".to_string(),
                suggestion: "Provide at least 1 level".to_string(),
            });
        }

        if groups_per_level == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Groups per level must be greater than 0".to_string(),
                suggestion: "Provide at least 1 group per level".to_string(),
            });
        }

        let mut tasks_per_level = Vec::new();
        let mut remainder_distribution = Vec::new();
        let mut remaining_tasks = total_tasks;

        // Calculate distribution for each level
        for level in 0..levels {
            let groups_at_level = if level == 0 {
                // Root level: distribute all tasks across groups
                std::cmp::min(groups_per_level as usize, remaining_tasks)
            } else {
                // Subsequent levels: each group from previous level can have up to groups_per_level children
                std::cmp::min(groups_per_level as usize, remaining_tasks)
            };

            if groups_at_level == 0 {
                tasks_per_level.push(0);
                remainder_distribution.push(0);
                continue;
            }

            let base_tasks = remaining_tasks / groups_at_level;
            let remainder = remaining_tasks % groups_at_level;

            tasks_per_level.push(base_tasks);
            remainder_distribution.push(remainder);

            // For hierarchical distribution, we need to account for the fact that
            // each task at this level will be a parent for the next level
            if level < levels - 1 {
                remaining_tasks = groups_at_level;
            }
        }

        Ok(Self {
            total_tasks,
            levels,
            groups_per_level,
            tasks_per_level,
            remainder_distribution,
        })
    }

    /// Get the number of tasks for a specific group at a specific level
    pub fn tasks_for_group(&self, level: usize, group_index: usize) -> usize {
        if level >= self.tasks_per_level.len() {
            return 0;
        }

        let base_tasks = self.tasks_per_level[level];
        let remainder = self.remainder_distribution[level];

        // First 'remainder' groups get one extra task
        if group_index < remainder {
            base_tasks + 1
        } else {
            base_tasks
        }
    }

    /// Get the total number of groups at a specific level
    pub fn groups_at_level(&self, level: usize) -> usize {
        if level >= self.tasks_per_level.len() {
            return 0;
        }

        let total_tasks_at_level = if level == 0 {
            self.total_tasks
        } else {
            // Each group from previous level becomes a task at this level
            std::cmp::min(self.groups_per_level as usize, self.total_tasks)
        };

        std::cmp::min(self.groups_per_level as usize, total_tasks_at_level)
    }
}

/// Task numbering system for hierarchical structures
#[derive(Debug, Clone)]
pub struct TaskNumbering {
    /// Current numbering state for each level
    level_counters: Vec<usize>,
    /// Maximum levels supported
    max_levels: u8,
}

impl TaskNumbering {
    /// Create a new task numbering system
    pub fn new(max_levels: u8) -> Self {
        Self {
            level_counters: vec![0; max_levels as usize],
            max_levels,
        }
    }

    /// Generate the next task number at the specified level
    pub fn next_number(&mut self, level: u8) -> TaskResult<String> {
        if level >= self.max_levels {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: format!("Level {} exceeds maximum levels {}", level, self.max_levels),
                suggestion: "Use a level within the configured range".to_string(),
            });
        }

        let level_idx = level as usize;
        
        // Increment counter at this level
        self.level_counters[level_idx] += 1;
        
        // Reset all deeper level counters
        for i in (level_idx + 1)..self.level_counters.len() {
            self.level_counters[i] = 0;
        }

        // Generate hierarchical number (e.g., "1.2.3")
        let number_parts: Vec<String> = self.level_counters[0..=level_idx]
            .iter()
            .map(|&count| count.to_string())
            .collect();

        Ok(number_parts.join("."))
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        for counter in &mut self.level_counters {
            *counter = 0;
        }
    }

    /// Get current state as a string for debugging
    pub fn current_state(&self) -> String {
        format!("Counters: {:?}", self.level_counters)
    }
}

/// Hierarchical task generator
pub struct HierarchicalTaskGenerator {
    /// Generation configuration
    config: GenerationConfig,
    /// Task numbering system
    numbering: TaskNumbering,
    /// Task distribution calculator
    distribution: TaskDistribution,
}

impl HierarchicalTaskGenerator {
    /// Create a new hierarchical task generator
    pub fn new(config: GenerationConfig) -> TaskResult<Self> {
        config.validate().map_err(|e| TaskError::InvalidTaskConfiguration {
            cause: e,
            suggestion: "Fix the configuration validation errors".to_string(),
        })?;

        // Create a placeholder distribution that will be recalculated when we have actual rows
        let distribution = TaskDistribution {
            total_tasks: 0,
            levels: config.levels,
            groups_per_level: config.groups,
            tasks_per_level: vec![0; config.levels as usize],
            remainder_distribution: vec![0; config.levels as usize],
        };

        let numbering = TaskNumbering::new(config.levels);

        Ok(Self {
            config,
            numbering,
            distribution,
        })
    }

    /// Generate hierarchical task structure from database rows
    pub fn generate_hierarchy(&mut self, rows: Vec<DatabaseRow>) -> TaskResult<TaskHierarchy> {
        let start_time = Instant::now();
        let total_tasks = rows.len();

        if total_tasks == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "No database rows provided for task generation".to_string(),
                suggestion: "Ensure the database query returns at least one row".to_string(),
            });
        }

        // Apply max_tasks limit if configured
        let effective_rows = if let Some(max_tasks) = self.config.max_tasks {
            if total_tasks > max_tasks {
                rows.into_iter().take(max_tasks).collect()
            } else {
                rows
            }
        } else {
            rows
        };

        let effective_total = effective_rows.len();

        // Recalculate distribution with actual task count
        self.distribution = TaskDistribution::calculate(
            effective_total,
            self.config.levels,
            self.config.groups,
        )?;

        // Reset numbering system
        self.numbering.reset();

        // Create hierarchy
        let mut hierarchy = TaskHierarchy::new(
            self.config.levels,
            self.config.groups,
            effective_total,
        );

        // Generate root level tasks
        let root_tasks = self.generate_level_tasks(0, &effective_rows)?;
        
        for task in root_tasks {
            hierarchy.add_root_task(task);
        }

        // Set generation duration
        let duration = start_time.elapsed();
        hierarchy.metadata.set_generation_duration(duration);

        // Validate the generated hierarchy
        hierarchy.validate().map_err(|e| TaskError::InvalidTaskConfiguration {
            cause: e,
            suggestion: "Check the hierarchy generation logic".to_string(),
        })?;

        Ok(hierarchy)
    }

    /// Generate tasks for a specific level
    fn generate_level_tasks(&mut self, level: u8, rows: &[DatabaseRow]) -> TaskResult<Vec<Task>> {
        let level_idx = level as usize;
        let groups_at_level = self.distribution.groups_at_level(level_idx);
        
        if groups_at_level == 0 {
            return Ok(Vec::new());
        }

        let mut tasks = Vec::new();
        let mut row_index = 0;

        for group_idx in 0..groups_at_level {
            let tasks_in_group = self.distribution.tasks_for_group(level_idx, group_idx);
            
            if tasks_in_group == 0 {
                continue;
            }

            // Generate task number
            let task_number = self.numbering.next_number(level)?;
            
            // Create group task
            let mut group_task = Task::new(
                format!("{}_{}", self.config.table_name, task_number.replace('.', "_")),
                self.generate_task_description(level, group_idx, tasks_in_group),
                task_number.clone(),
                level,
            );

            // Add content files and subtasks based on level
            if level == self.config.levels - 1 {
                // Leaf level: add actual content files for each row
                for _ in 0..tasks_in_group {
                    if row_index < rows.len() {
                        let row = &rows[row_index];
                        self.add_content_files_for_row(&mut group_task, row, row_index + 1)?;
                        row_index += 1;
                    }
                }
            } else {
                // Branch level: create subtasks
                let subtask_rows: Vec<DatabaseRow> = rows
                    .iter()
                    .skip(row_index)
                    .take(tasks_in_group)
                    .cloned()
                    .collect();

                if !subtask_rows.is_empty() {
                    let subtasks = self.generate_level_tasks(level + 1, &subtask_rows)?;
                    for subtask in subtasks {
                        group_task.add_subtask(subtask);
                    }
                    row_index += subtask_rows.len();
                }
            }

            // Set prompt file if configured
            if let Some(prompt_file) = &self.config.prompt_file {
                group_task.set_prompt_file(prompt_file.clone());
            }

            // Set output file
            let output_file = self.config.work_area_dir.join(format!(
                "{}_{}.md",
                self.config.effective_table_name(),
                task_number.replace('.', "_")
            ));
            group_task.set_output_file(output_file);

            tasks.push(group_task);
        }

        Ok(tasks)
    }

    /// Generate task description based on level and context
    fn generate_task_description(&self, level: u8, group_idx: usize, task_count: usize) -> String {
        let table_name = self.config.effective_table_name();
        
        match level {
            0 => {
                if self.config.levels == 1 {
                    format!("Analyze {} rows from {}", task_count, table_name)
                } else {
                    format!("Process {} group {} ({} items)", table_name, group_idx + 1, task_count)
                }
            }
            level if level == self.config.levels - 1 => {
                // Leaf level
                format!("Analyze {} files from group {}", task_count, group_idx + 1)
            }
            _ => {
                // Intermediate level
                format!("Process subgroup {} ({} items)", group_idx + 1, task_count)
            }
        }
    }

    /// Add content files for a specific database row
    fn add_content_files_for_row(
        &self,
        task: &mut Task,
        row: &DatabaseRow,
        row_number: usize,
    ) -> TaskResult<()> {
        let table_name = self.config.effective_table_name();
        
        // Primary content file (A)
        let content_file = ContentFileReference::new(
            self.config.output_dir.join(format!("{}_{}_Content.txt", table_name, row_number)),
            "A".to_string(),
            "Primary content file".to_string(),
            ContentFileType::Content,
        );
        task.add_content_file(content_file);

        // L1 context file (B)
        let l1_file = ContentFileReference::new(
            self.config.output_dir.join(format!("{}_{}_Content_L1.txt", table_name, row_number)),
            "B".to_string(),
            "L1 context file".to_string(),
            ContentFileType::L1Context,
        );
        task.add_content_file(l1_file);

        // L2 context file (C)
        let l2_file = ContentFileReference::new(
            self.config.output_dir.join(format!("{}_{}_Content_L2.txt", table_name, row_number)),
            "C".to_string(),
            "L2 context file".to_string(),
            ContentFileType::L2Context,
        );
        task.add_content_file(l2_file);

        // Add row metadata to task
        if let Some(filepath) = &row.filepath {
            task.metadata.add_tag(format!("file:{}", filepath));
        }
        if let Some(extension) = &row.extension {
            task.metadata.add_tag(format!("ext:{}", extension));
        }
        if let Some(line_count) = row.line_count {
            task.metadata.add_tag(format!("lines:{}", line_count));
        }

        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }

    /// Get the current task distribution
    pub fn distribution(&self) -> &TaskDistribution {
        &self.distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_config() -> GenerationConfig {
        GenerationConfig::new(
            "INGEST_20250928101039".to_string(),
            4,
            7,
            PathBuf::from("tasks.md"),
        )
    }

    fn create_test_rows(count: usize) -> Vec<DatabaseRow> {
        (1..=count)
            .map(|i| DatabaseRow {
                row_id: i as i64,
                filepath: Some(format!("src/file_{}.rs", i)),
                filename: Some(format!("file_{}.rs", i)),
                extension: Some("rs".to_string()),
                line_count: Some(100 * i as i32),
                content: Some(format!("Content for file {}", i)),
                metadata: HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_task_distribution_calculation() {
        // Test even distribution: 35 tasks, 4 levels, 7 groups
        let dist = TaskDistribution::calculate(35, 4, 7).unwrap();
        
        assert_eq!(dist.total_tasks, 35);
        assert_eq!(dist.levels, 4);
        assert_eq!(dist.groups_per_level, 7);
        
        // At level 0: 35 tasks ÷ 7 groups = 5 tasks per group
        assert_eq!(dist.tasks_for_group(0, 0), 5);
        assert_eq!(dist.tasks_for_group(0, 6), 5);
        
        // Check groups at each level
        assert_eq!(dist.groups_at_level(0), 7);
    }

    #[test]
    fn test_task_distribution_with_remainder() {
        // Test uneven distribution: 37 tasks, 4 levels, 7 groups
        let dist = TaskDistribution::calculate(37, 4, 7).unwrap();
        
        // At level 0: 37 tasks ÷ 7 groups = 5 base + 2 remainder
        // First 2 groups get 6 tasks, remaining 5 groups get 5 tasks
        assert_eq!(dist.tasks_for_group(0, 0), 6); // First group gets extra
        assert_eq!(dist.tasks_for_group(0, 1), 6); // Second group gets extra
        assert_eq!(dist.tasks_for_group(0, 2), 5); // Third group gets base
        assert_eq!(dist.tasks_for_group(0, 6), 5); // Last group gets base
    }

    #[test]
    fn test_task_distribution_edge_cases() {
        // Test with 0 tasks (should fail)
        let result = TaskDistribution::calculate(0, 4, 7);
        assert!(result.is_err());

        // Test with 0 levels (should fail)
        let result = TaskDistribution::calculate(35, 0, 7);
        assert!(result.is_err());

        // Test with 0 groups (should fail)
        let result = TaskDistribution::calculate(35, 4, 0);
        assert!(result.is_err());

        // Test with fewer tasks than groups
        let dist = TaskDistribution::calculate(5, 4, 7).unwrap();
        assert_eq!(dist.groups_at_level(0), 5); // Only 5 groups needed
    }

    #[test]
    fn test_task_numbering() {
        let mut numbering = TaskNumbering::new(4);
        
        // Test level 0 numbering
        assert_eq!(numbering.next_number(0).unwrap(), "1");
        assert_eq!(numbering.next_number(0).unwrap(), "2");
        
        // Test level 1 numbering
        assert_eq!(numbering.next_number(1).unwrap(), "2.1");
        assert_eq!(numbering.next_number(1).unwrap(), "2.2");
        
        // Test level 2 numbering
        assert_eq!(numbering.next_number(2).unwrap(), "2.2.1");
        
        // Test going back to level 0 (should reset deeper levels)
        assert_eq!(numbering.next_number(0).unwrap(), "3");
        assert_eq!(numbering.next_number(1).unwrap(), "3.1");
    }

    #[test]
    fn test_task_numbering_edge_cases() {
        let mut numbering = TaskNumbering::new(3);
        
        // Test exceeding max levels
        let result = numbering.next_number(5);
        assert!(result.is_err());
        
        // Test reset functionality
        numbering.next_number(0).unwrap();
        numbering.next_number(1).unwrap();
        numbering.reset();
        assert_eq!(numbering.next_number(0).unwrap(), "1");
    }

    #[test]
    fn test_hierarchical_task_generator_creation() {
        let config = create_test_config();
        let generator = HierarchicalTaskGenerator::new(config);
        assert!(generator.is_ok());
        
        // Test with invalid config
        let invalid_config = GenerationConfig::new(
            String::new(), // Empty table name
            4,
            7,
            PathBuf::from("tasks.md"),
        );
        let result = HierarchicalTaskGenerator::new(invalid_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_hierarchy_basic() {
        let config = create_test_config();
        let mut generator = HierarchicalTaskGenerator::new(config).unwrap();
        let rows = create_test_rows(35);
        
        let hierarchy = generator.generate_hierarchy(rows).unwrap();
        
        assert_eq!(hierarchy.levels, 4);
        assert_eq!(hierarchy.groups_per_level, 7);
        assert_eq!(hierarchy.total_tasks, 35);
        assert!(!hierarchy.root_tasks.is_empty());
        
        // Validate hierarchy structure
        assert!(hierarchy.validate().is_ok());
        
        // Check that generation duration was recorded
        assert!(hierarchy.metadata.generation_duration.is_some());
    }

    #[test]
    fn test_generate_hierarchy_with_max_tasks() {
        let config = create_test_config().with_max_tasks(20);
        let mut generator = HierarchicalTaskGenerator::new(config).unwrap();
        let rows = create_test_rows(35); // More rows than max_tasks
        
        let hierarchy = generator.generate_hierarchy(rows).unwrap();
        
        // Should only process 20 tasks due to max_tasks limit
        assert_eq!(hierarchy.total_tasks, 20);
    }

    #[test]
    fn test_generate_hierarchy_empty_rows() {
        let config = create_test_config();
        let mut generator = HierarchicalTaskGenerator::new(config).unwrap();
        let rows = Vec::new();
        
        let result = generator.generate_hierarchy(rows);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            TaskError::InvalidTaskConfiguration { cause, .. } => {
                assert!(cause.contains("No database rows provided"));
            }
            _ => panic!("Expected InvalidTaskConfiguration error"),
        }
    }

    #[test]
    fn test_task_description_generation() {
        let config = create_test_config();
        let mut generator = HierarchicalTaskGenerator::new(config).unwrap();
        
        // Test different level descriptions
        let desc_0 = generator.generate_task_description(0, 0, 5);
        assert!(desc_0.contains("INGEST_20250928101039"));
        assert!(desc_0.contains("group 1"));
        
        let desc_leaf = generator.generate_task_description(3, 2, 3); // Level 3 is leaf for 4 levels
        assert!(desc_leaf.contains("Analyze"));
        assert!(desc_leaf.contains("files"));
    }

    #[test]
    fn test_content_files_generation() {
        let config = create_test_config();
        let generator = HierarchicalTaskGenerator::new(config).unwrap();
        let mut task = Task::new(
            "test_task".to_string(),
            "Test task".to_string(),
            "1.1".to_string(),
            0,
        );
        
        let row = DatabaseRow {
            row_id: 1,
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            line_count: Some(100),
            content: Some("fn main() {}".to_string()),
            metadata: HashMap::new(),
        };
        
        generator.add_content_files_for_row(&mut task, &row, 1).unwrap();
        
        // Should have 3 content files (A, B, C)
        assert_eq!(task.content_files.len(), 3);
        
        // Check file types
        let content_types: Vec<_> = task.content_files
            .iter()
            .map(|f| &f.file_type)
            .collect();
        
        assert!(content_types.contains(&&ContentFileType::Content));
        assert!(content_types.contains(&&ContentFileType::L1Context));
        assert!(content_types.contains(&&ContentFileType::L2Context));
        
        // Check that metadata tags were added
        assert!(task.metadata.tags.iter().any(|tag| tag.starts_with("file:")));
        assert!(task.metadata.tags.iter().any(|tag| tag.starts_with("ext:")));
        assert!(task.metadata.tags.iter().any(|tag| tag.starts_with("lines:")));
    }
}