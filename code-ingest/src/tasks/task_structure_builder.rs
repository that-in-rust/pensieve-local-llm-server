//! [DEPRECATED] Task structure generation and management
//! 
//! ⚠️  DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
//! Please use `chunk_level_task_generator` instead for simpler, more maintainable task generation.
//! 
//! This module provides functionality for building complex task structures
//! with proper relationships, content file references, and prompt integration.
//! 
//! Migration path: Use `ChunkLevelTaskGenerator` for file-level or chunk-level task generation.

use crate::error::{TaskError, TaskResult};
use crate::tasks::models::{
    Task, TaskHierarchy, GenerationConfig, ContentFileReference, ContentFileType,
};
use crate::tasks::hierarchical_generator::DatabaseRow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Task structure builder for creating complex hierarchical task structures
pub struct TaskStructureBuilder {
    /// Configuration for task generation
    config: GenerationConfig,
    /// Template for task descriptions
    description_template: Option<String>,
    /// Custom content file naming patterns
    content_file_patterns: HashMap<ContentFileType, String>,
}

impl TaskStructureBuilder {
    /// Create a new task structure builder
    pub fn new(config: GenerationConfig) -> Self {
        let mut content_file_patterns = HashMap::new();
        
        // Default content file patterns
        content_file_patterns.insert(
            ContentFileType::Content,
            "{table_name}_{row_number}_Content.txt".to_string(),
        );
        content_file_patterns.insert(
            ContentFileType::L1Context,
            "{table_name}_{row_number}_Content_L1.txt".to_string(),
        );
        content_file_patterns.insert(
            ContentFileType::L2Context,
            "{table_name}_{row_number}_Content_L2.txt".to_string(),
        );

        Self {
            config,
            description_template: None,
            content_file_patterns,
        }
    }

    /// Set a custom description template
    /// 
    /// Template variables:
    /// - {table_name}: The table name
    /// - {task_number}: The hierarchical task number (e.g., "1.2.3")
    /// - {level}: The task level (0-based)
    /// - {group_index}: The group index within the level
    /// - {task_count}: Number of tasks in this group
    /// - {file_path}: File path (for leaf tasks)
    /// - {file_extension}: File extension (for leaf tasks)
    pub fn with_description_template(mut self, template: String) -> Self {
        self.description_template = Some(template);
        self
    }

    /// Set custom content file naming patterns
    pub fn with_content_file_pattern(mut self, file_type: ContentFileType, pattern: String) -> Self {
        self.content_file_patterns.insert(file_type, pattern);
        self
    }

    /// Build a complete task structure from database rows
    pub fn build_task_structure(
        &self,
        rows: Vec<DatabaseRow>,
        hierarchy: &TaskHierarchy,
    ) -> TaskResult<TaskStructure> {
        let mut task_structure = TaskStructure::new(
            self.config.table_name.clone(),
            hierarchy.levels,
            hierarchy.groups_per_level,
        );

        // Build tasks from hierarchy
        for root_task in &hierarchy.root_tasks {
            let enhanced_task = self.enhance_task_with_structure(root_task, &rows)?;
            task_structure.add_root_task(enhanced_task);
        }

        // Set metadata
        task_structure.set_generation_config(self.config.clone());
        task_structure.validate()?;

        Ok(task_structure)
    }

    /// Enhance a task with proper structure, content files, and relationships
    fn enhance_task_with_structure(
        &self,
        task: &Task,
        rows: &[DatabaseRow],
    ) -> TaskResult<EnhancedTask> {
        let mut enhanced_task = EnhancedTask::from_task(task.clone());

        // Add content files if this is a leaf task
        if task.is_leaf() {
            self.add_content_files_to_task(&mut enhanced_task, rows)?;
        }

        // Add prompt file reference
        if let Some(prompt_file) = &self.config.prompt_file {
            enhanced_task.set_prompt_reference(PromptReference::new(
                prompt_file.clone(),
                self.generate_prompt_description(&enhanced_task),
            ));
        }

        // Set output file
        let output_file = self.generate_output_file_path(&enhanced_task);
        enhanced_task.set_output_file(output_file);

        // Enhance subtasks recursively
        for subtask in &task.subtasks {
            let enhanced_subtask = self.enhance_task_with_structure(subtask, rows)?;
            enhanced_task.add_enhanced_subtask(enhanced_subtask);
        }

        // Add task relationships
        self.add_task_relationships(&mut enhanced_task)?;

        Ok(enhanced_task)
    }

    /// Add content files to a task based on associated database rows
    fn add_content_files_to_task(
        &self,
        task: &mut EnhancedTask,
        rows: &[DatabaseRow],
    ) -> TaskResult<()> {
        // Find rows associated with this task based on task metadata
        let associated_rows = self.find_associated_rows(task, rows)?;

        for (index, row) in associated_rows.iter().enumerate() {
            let row_number = index + 1;
            
            // Add primary content file (A)
            let content_file = self.create_content_file_reference(
                ContentFileType::Content,
                row,
                row_number,
                "A",
                "Primary content file containing the main file content",
            )?;
            task.add_content_file(content_file);

            // Add L1 context file (B)
            let l1_file = self.create_content_file_reference(
                ContentFileType::L1Context,
                row,
                row_number,
                "B",
                "L1 context file with surrounding context (±1 chunk)",
            )?;
            task.add_content_file(l1_file);

            // Add L2 context file (C)
            let l2_file = self.create_content_file_reference(
                ContentFileType::L2Context,
                row,
                row_number,
                "C",
                "L2 context file with extended context (±2 chunks)",
            )?;
            task.add_content_file(l2_file);

            // Add row metadata to task
            self.add_row_metadata_to_task(task, row)?;
        }

        Ok(())
    }

    /// Create a content file reference for a specific row and file type
    fn create_content_file_reference(
        &self,
        file_type: ContentFileType,
        row: &DatabaseRow,
        row_number: usize,
        role: &str,
        description: &str,
    ) -> TaskResult<ContentFileReference> {
        let pattern = self.content_file_patterns.get(&file_type)
            .ok_or_else(|| TaskError::InvalidTaskConfiguration {
                cause: format!("No pattern defined for content file type: {:?}", file_type),
                suggestion: "Define a pattern for this content file type".to_string(),
            })?;

        let filename = self.apply_content_file_pattern(pattern, row, row_number)?;
        let file_path = self.config.output_dir.join(&filename);

        Ok(ContentFileReference::new(
            file_path,
            role.to_string(),
            description.to_string(),
            file_type,
        ))
    }

    /// Apply content file naming pattern with variable substitution
    fn apply_content_file_pattern(
        &self,
        pattern: &str,
        row: &DatabaseRow,
        row_number: usize,
    ) -> TaskResult<String> {
        let mut result = pattern.to_string();
        
        // Replace template variables
        result = result.replace("{table_name}", &self.config.effective_table_name());
        result = result.replace("{row_number}", &row_number.to_string());
        
        if let Some(filepath) = &row.filepath {
            result = result.replace("{file_path}", filepath);
            
            if let Some(extension) = &row.extension {
                result = result.replace("{file_extension}", extension);
            }
        }

        Ok(result)
    }

    /// Find database rows associated with a specific task
    fn find_associated_rows<'a>(
        &self,
        task: &EnhancedTask,
        rows: &'a [DatabaseRow],
    ) -> TaskResult<Vec<&'a DatabaseRow>> {
        // For now, we'll use a simple approach based on task metadata
        // In a more sophisticated implementation, this could use task-specific logic
        
        let mut associated_rows = Vec::new();
        
        // Look for file path tags in task metadata
        for tag in &task.base_task.metadata.tags {
            if let Some(filepath) = tag.strip_prefix("file:") {
                if let Some(row) = rows.iter().find(|r| {
                    r.filepath.as_ref().map(|p| p == filepath).unwrap_or(false)
                }) {
                    associated_rows.push(row);
                }
            }
        }

        // If no specific associations found, return empty (branch tasks don't have content files)
        Ok(associated_rows)
    }

    /// Add metadata from a database row to a task
    fn add_row_metadata_to_task(
        &self,
        task: &mut EnhancedTask,
        row: &DatabaseRow,
    ) -> TaskResult<()> {
        // Add file-specific metadata
        if let Some(filepath) = &row.filepath {
            task.add_metadata("source_file", filepath.clone());
        }
        
        if let Some(extension) = &row.extension {
            task.add_metadata("file_extension", extension.clone());
        }
        
        if let Some(line_count) = row.line_count {
            task.add_metadata("line_count", line_count.to_string());
        }

        // Add custom metadata from the row
        for (key, value) in &row.metadata {
            task.add_metadata(&format!("row_{}", key), value.clone());
        }

        Ok(())
    }

    /// Generate a prompt description for a task
    fn generate_prompt_description(&self, task: &EnhancedTask) -> String {
        if let Some(template) = &self.description_template {
            self.apply_description_template(template, task)
        } else {
            format!(
                "Analysis prompt for task {} - {}",
                task.base_task.task_number,
                task.base_task.description
            )
        }
    }

    /// Apply description template with variable substitution
    fn apply_description_template(&self, template: &str, task: &EnhancedTask) -> String {
        let mut result = template.to_string();
        
        result = result.replace("{table_name}", &self.config.effective_table_name());
        result = result.replace("{task_number}", &task.base_task.task_number);
        result = result.replace("{level}", &task.base_task.level.to_string());
        result = result.replace("{description}", &task.base_task.description);
        
        // Add more template variables as needed
        result
    }

    /// Generate output file path for a task
    fn generate_output_file_path(&self, task: &EnhancedTask) -> PathBuf {
        let filename = format!(
            "{}_{}.md",
            self.config.effective_table_name(),
            task.base_task.task_number.replace('.', "_")
        );
        
        self.config.work_area_dir.join(filename)
    }

    /// Add task relationships (dependencies, prerequisites, etc.)
    fn add_task_relationships(&self, task: &mut EnhancedTask) -> TaskResult<()> {
        // For now, we'll add basic parent-child relationships
        // This can be extended to include more complex dependency management
        
        let relationships: Vec<TaskRelationship> = task.enhanced_subtasks
            .iter()
            .enumerate()
            .map(|(index, subtask)| {
                TaskRelationship::new(
                    RelationshipType::ParentChild,
                    task.base_task.id.clone(),
                    subtask.base_task.id.clone(),
                    format!("Subtask {} of parent task", index + 1),
                )
            })
            .collect();
        
        for relationship in relationships {
            task.add_relationship(relationship);
        }

        Ok(())
    }
}

/// Enhanced task with additional structure and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTask {
    /// Base task information
    pub base_task: Task,
    /// Content file references
    pub content_files: Vec<ContentFileReference>,
    /// Prompt reference
    pub prompt_reference: Option<PromptReference>,
    /// Output file path
    pub output_file: Option<PathBuf>,
    /// Enhanced subtasks
    pub enhanced_subtasks: Vec<EnhancedTask>,
    /// Task relationships
    pub relationships: Vec<TaskRelationship>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl EnhancedTask {
    /// Create an enhanced task from a base task
    pub fn from_task(task: Task) -> Self {
        Self {
            base_task: task,
            content_files: Vec::new(),
            prompt_reference: None,
            output_file: None,
            enhanced_subtasks: Vec::new(),
            relationships: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a content file reference
    pub fn add_content_file(&mut self, content_file: ContentFileReference) {
        self.content_files.push(content_file);
    }

    /// Set prompt reference
    pub fn set_prompt_reference(&mut self, prompt_reference: PromptReference) {
        self.prompt_reference = Some(prompt_reference);
    }

    /// Set output file
    pub fn set_output_file(&mut self, output_file: PathBuf) {
        self.output_file = Some(output_file);
    }

    /// Add an enhanced subtask
    pub fn add_enhanced_subtask(&mut self, subtask: EnhancedTask) {
        self.enhanced_subtasks.push(subtask);
    }

    /// Add a task relationship
    pub fn add_relationship(&mut self, relationship: TaskRelationship) {
        self.relationships.push(relationship);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: &str, value: String) {
        self.metadata.insert(key.to_string(), value);
    }

    /// Check if this is a leaf task (no subtasks)
    pub fn is_leaf(&self) -> bool {
        self.enhanced_subtasks.is_empty()
    }

    /// Get all content files of a specific type
    pub fn get_content_files_by_type(&self, file_type: &ContentFileType) -> Vec<&ContentFileReference> {
        self.content_files
            .iter()
            .filter(|cf| &cf.file_type == file_type)
            .collect()
    }
}

/// Reference to a prompt file with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptReference {
    /// Path to the prompt file
    pub file_path: PathBuf,
    /// Description of how to use this prompt
    pub description: String,
    /// Prompt variables or parameters
    pub variables: HashMap<String, String>,
}

impl PromptReference {
    /// Create a new prompt reference
    pub fn new(file_path: PathBuf, description: String) -> Self {
        Self {
            file_path,
            description,
            variables: HashMap::new(),
        }
    }

    /// Add a prompt variable
    pub fn add_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }
}

/// Task relationship definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRelationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Source task ID
    pub source_task_id: String,
    /// Target task ID
    pub target_task_id: String,
    /// Description of the relationship
    pub description: String,
}

impl TaskRelationship {
    /// Create a new task relationship
    pub fn new(
        relationship_type: RelationshipType,
        source_task_id: String,
        target_task_id: String,
        description: String,
    ) -> Self {
        Self {
            relationship_type,
            source_task_id,
            target_task_id,
            description,
        }
    }
}

/// Types of task relationships
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Parent-child relationship
    ParentChild,
    /// Dependency relationship (source depends on target)
    DependsOn,
    /// Prerequisite relationship (source is prerequisite for target)
    Prerequisite,
    /// Sequential relationship (source comes before target)
    Sequential,
    /// Custom relationship type
    Custom(String),
}

/// Complete task structure with enhanced tasks and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStructure {
    /// Table name this structure was generated from
    pub table_name: String,
    /// Number of hierarchy levels
    pub levels: u8,
    /// Number of groups per level
    pub groups_per_level: u8,
    /// Root level enhanced tasks
    pub root_tasks: Vec<EnhancedTask>,
    /// Generation configuration
    pub generation_config: Option<GenerationConfig>,
    /// Structure metadata
    pub metadata: TaskStructureMetadata,
}

impl TaskStructure {
    /// Create a new task structure
    pub fn new(table_name: String, levels: u8, groups_per_level: u8) -> Self {
        Self {
            table_name: table_name.clone(),
            levels,
            groups_per_level,
            root_tasks: Vec::new(),
            generation_config: None,
            metadata: TaskStructureMetadata::new(table_name),
        }
    }

    /// Add a root task
    pub fn add_root_task(&mut self, task: EnhancedTask) {
        self.root_tasks.push(task);
    }

    /// Set generation configuration
    pub fn set_generation_config(&mut self, config: GenerationConfig) {
        self.generation_config = Some(config);
    }

    /// Validate the task structure
    pub fn validate(&self) -> TaskResult<()> {
        if self.table_name.is_empty() {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Table name cannot be empty".to_string(),
                suggestion: "Provide a valid table name".to_string(),
            });
        }

        if self.levels == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Levels must be greater than 0".to_string(),
                suggestion: "Provide at least 1 level".to_string(),
            });
        }

        if self.groups_per_level == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Groups per level must be greater than 0".to_string(),
                suggestion: "Provide at least 1 group per level".to_string(),
            });
        }

        // Validate each root task
        for task in &self.root_tasks {
            self.validate_enhanced_task(task)?;
        }

        Ok(())
    }

    /// Validate an enhanced task recursively
    fn validate_enhanced_task(&self, task: &EnhancedTask) -> TaskResult<()> {
        if task.base_task.id.is_empty() {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Task ID cannot be empty".to_string(),
                suggestion: "Ensure all tasks have valid IDs".to_string(),
            });
        }

        if task.base_task.description.is_empty() {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Task description cannot be empty".to_string(),
                suggestion: "Ensure all tasks have descriptions".to_string(),
            });
        }

        // Validate subtasks recursively
        for subtask in &task.enhanced_subtasks {
            self.validate_enhanced_task(subtask)?;
        }

        Ok(())
    }

    /// Get all tasks flattened into a single list
    pub fn flatten_tasks(&self) -> Vec<&EnhancedTask> {
        let mut tasks = Vec::new();
        for root_task in &self.root_tasks {
            tasks.push(root_task);
            self.collect_subtasks(root_task, &mut tasks);
        }
        tasks
    }

    /// Recursively collect all subtasks
    fn collect_subtasks<'a>(&self, task: &'a EnhancedTask, tasks: &mut Vec<&'a EnhancedTask>) {
        for subtask in &task.enhanced_subtasks {
            tasks.push(subtask);
            self.collect_subtasks(subtask, tasks);
        }
    }

    /// Get statistics about the task structure
    pub fn statistics(&self) -> TaskStructureStatistics {
        let flattened = self.flatten_tasks();
        let total_tasks = flattened.len();
        let leaf_tasks = flattened.iter().filter(|t| t.is_leaf()).count();
        let branch_tasks = total_tasks - leaf_tasks;
        let total_content_files = flattened.iter().map(|t| t.content_files.len()).sum();

        TaskStructureStatistics {
            total_tasks,
            leaf_tasks,
            branch_tasks,
            total_content_files,
            tasks_with_prompts: flattened.iter().filter(|t| t.prompt_reference.is_some()).count(),
            tasks_with_output_files: flattened.iter().filter(|t| t.output_file.is_some()).count(),
        }
    }
}

/// Metadata for task structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStructureMetadata {
    /// Structure ID
    pub structure_id: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Version of the structure format
    pub version: String,
}

impl TaskStructureMetadata {
    /// Create new metadata
    pub fn new(structure_id: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            structure_id,
            created_at: now,
            modified_at: now,
            version: "1.0.0".to_string(),
        }
    }

    /// Update the modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = chrono::Utc::now();
    }
}

/// Statistics about a task structure
#[derive(Debug, Clone, PartialEq)]
pub struct TaskStructureStatistics {
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of leaf tasks
    pub leaf_tasks: usize,
    /// Number of branch tasks
    pub branch_tasks: usize,
    /// Total number of content files
    pub total_content_files: usize,
    /// Number of tasks with prompt references
    pub tasks_with_prompts: usize,
    /// Number of tasks with output files
    pub tasks_with_output_files: usize,
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
        .with_prompt_file(PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"))
    }

    fn create_test_rows() -> Vec<DatabaseRow> {
        vec![
            DatabaseRow {
                row_id: 1,
                filepath: Some("src/main.rs".to_string()),
                filename: Some("main.rs".to_string()),
                extension: Some("rs".to_string()),
                line_count: Some(100),
                content: Some("fn main() {}".to_string()),
                metadata: HashMap::new(),
            },
            DatabaseRow {
                row_id: 2,
                filepath: Some("src/lib.rs".to_string()),
                filename: Some("lib.rs".to_string()),
                extension: Some("rs".to_string()),
                line_count: Some(200),
                content: Some("pub mod test;".to_string()),
                metadata: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_task_structure_builder_creation() {
        let config = create_test_config();
        let builder = TaskStructureBuilder::new(config);
        
        assert!(builder.description_template.is_none());
        assert_eq!(builder.content_file_patterns.len(), 3);
    }

    #[test]
    fn test_task_structure_builder_with_template() {
        let config = create_test_config();
        let template = "Analyze {table_name} task {task_number} at level {level}".to_string();
        let builder = TaskStructureBuilder::new(config)
            .with_description_template(template.clone());
        
        assert_eq!(builder.description_template, Some(template));
    }

    #[test]
    fn test_content_file_pattern_application() {
        let config = create_test_config();
        let builder = TaskStructureBuilder::new(config);
        let row = &create_test_rows()[0];
        
        let pattern = "{table_name}_{row_number}_Content.txt";
        let result = builder.apply_content_file_pattern(pattern, row, 1).unwrap();
        
        assert_eq!(result, "INGEST_20250928101039_1_Content.txt");
    }

    #[test]
    fn test_enhanced_task_creation() {
        let base_task = Task::new(
            "task_1".to_string(),
            "Test task".to_string(),
            "1".to_string(),
            0,
        );
        
        let mut enhanced_task = EnhancedTask::from_task(base_task);
        
        assert_eq!(enhanced_task.base_task.id, "task_1");
        assert!(enhanced_task.content_files.is_empty());
        assert!(enhanced_task.prompt_reference.is_none());
        assert!(enhanced_task.is_leaf());
        
        // Add content file
        let content_file = ContentFileReference::new(
            PathBuf::from("content.txt"),
            "A".to_string(),
            "Test content".to_string(),
            ContentFileType::Content,
        );
        enhanced_task.add_content_file(content_file);
        
        assert_eq!(enhanced_task.content_files.len(), 1);
    }

    #[test]
    fn test_prompt_reference() {
        let mut prompt_ref = PromptReference::new(
            PathBuf::from("prompt.md"),
            "Test prompt".to_string(),
        );
        
        prompt_ref.add_variable("table_name".to_string(), "TEST_TABLE".to_string());
        
        assert_eq!(prompt_ref.variables.len(), 1);
        assert_eq!(prompt_ref.variables.get("table_name"), Some(&"TEST_TABLE".to_string()));
    }

    #[test]
    fn test_task_relationship() {
        let relationship = TaskRelationship::new(
            RelationshipType::ParentChild,
            "parent_task".to_string(),
            "child_task".to_string(),
            "Parent-child relationship".to_string(),
        );
        
        assert_eq!(relationship.relationship_type, RelationshipType::ParentChild);
        assert_eq!(relationship.source_task_id, "parent_task");
        assert_eq!(relationship.target_task_id, "child_task");
    }

    #[test]
    fn test_task_structure_creation() {
        let mut structure = TaskStructure::new(
            "TEST_TABLE".to_string(),
            3,
            5,
        );
        
        assert_eq!(structure.table_name, "TEST_TABLE");
        assert_eq!(structure.levels, 3);
        assert_eq!(structure.groups_per_level, 5);
        assert!(structure.root_tasks.is_empty());
        
        // Validation should pass for valid structure
        assert!(structure.validate().is_ok());
    }

    #[test]
    fn test_task_structure_validation() {
        // Test empty table name
        let mut structure = TaskStructure::new(
            String::new(),
            3,
            5,
        );
        assert!(structure.validate().is_err());
        
        // Test zero levels
        structure = TaskStructure::new(
            "TEST_TABLE".to_string(),
            0,
            5,
        );
        assert!(structure.validate().is_err());
        
        // Test zero groups
        structure = TaskStructure::new(
            "TEST_TABLE".to_string(),
            3,
            0,
        );
        assert!(structure.validate().is_err());
    }

    #[test]
    fn test_task_structure_statistics() {
        let mut structure = TaskStructure::new(
            "TEST_TABLE".to_string(),
            2,
            3,
        );
        
        // Add a root task with subtasks
        let mut root_task = EnhancedTask::from_task(Task::new(
            "root".to_string(),
            "Root task".to_string(),
            "1".to_string(),
            0,
        ));
        
        let subtask = EnhancedTask::from_task(Task::new(
            "sub".to_string(),
            "Subtask".to_string(),
            "1.1".to_string(),
            1,
        ));
        
        root_task.add_enhanced_subtask(subtask);
        structure.add_root_task(root_task);
        
        let stats = structure.statistics();
        assert_eq!(stats.total_tasks, 2); // Root + 1 subtask
        assert_eq!(stats.leaf_tasks, 1); // Only subtask is leaf
        assert_eq!(stats.branch_tasks, 1); // Only root is branch
    }

    #[test]
    fn test_content_file_reference_creation() {
        let config = create_test_config();
        let builder = TaskStructureBuilder::new(config);
        let row = &create_test_rows()[0];
        
        let content_file = builder.create_content_file_reference(
            ContentFileType::Content,
            row,
            1,
            "A",
            "Primary content file",
        ).unwrap();
        
        assert_eq!(content_file.role, "A");
        assert_eq!(content_file.description, "Primary content file");
        assert_eq!(content_file.file_type, ContentFileType::Content);
        assert!(content_file.file_path.to_string_lossy().contains("INGEST_20250928101039_1_Content.txt"));
    }
}