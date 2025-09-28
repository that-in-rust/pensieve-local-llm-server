//! Core data models for task generation and hierarchical analysis
//! 
//! This module defines the fundamental data structures used throughout the
//! task generation system, including ingestion sources, chunk metadata,
//! task hierarchies, and generation configurations.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

/// Source for code ingestion - either a Git repository or local folder
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestionSource {
    /// Git repository with URL and optional authentication
    GitRepository {
        /// Repository URL (https, ssh, or git protocol)
        url: String,
        /// Optional branch or tag to clone
        branch: Option<String>,
        /// Optional authentication token
        token: Option<String>,
    },
    /// Local folder with absolute path and processing options
    LocalFolder {
        /// Absolute path to the folder
        path: PathBuf,
        /// Whether to process recursively
        recursive: bool,
        /// Whether to follow symbolic links
        follow_symlinks: bool,
    },
}

impl IngestionSource {
    /// Create a new Git repository source
    pub fn git_repository(url: impl Into<String>) -> Self {
        Self::GitRepository {
            url: url.into(),
            branch: None,
            token: None,
        }
    }

    /// Create a new Git repository source with branch
    pub fn git_repository_with_branch(url: impl Into<String>, branch: impl Into<String>) -> Self {
        Self::GitRepository {
            url: url.into(),
            branch: Some(branch.into()),
            token: None,
        }
    }

    /// Create a new local folder source
    pub fn local_folder(path: impl Into<PathBuf>) -> Self {
        Self::LocalFolder {
            path: path.into(),
            recursive: true,
            follow_symlinks: false,
        }
    }

    /// Create a new local folder source with options
    pub fn local_folder_with_options(
        path: impl Into<PathBuf>,
        recursive: bool,
        follow_symlinks: bool,
    ) -> Self {
        Self::LocalFolder {
            path: path.into(),
            recursive,
            follow_symlinks,
        }
    }

    /// Get a display string for the source
    pub fn display_name(&self) -> String {
        match self {
            Self::GitRepository { url, branch, .. } => {
                if let Some(branch) = branch {
                    format!("{}@{}", url, branch)
                } else {
                    url.clone()
                }
            }
            Self::LocalFolder { path, .. } => path.display().to_string(),
        }
    }

    /// Check if this is a Git repository source
    pub fn is_git_repository(&self) -> bool {
        matches!(self, Self::GitRepository { .. })
    }

    /// Check if this is a local folder source
    pub fn is_local_folder(&self) -> bool {
        matches!(self, Self::LocalFolder { .. })
    }

    /// Validate the source configuration
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::GitRepository { url, .. } => {
                if url.is_empty() {
                    return Err("Git repository URL cannot be empty".to_string());
                }
                // Basic URL validation
                if !url.starts_with("http") && !url.starts_with("git") && !url.contains('@') {
                    return Err("Invalid Git repository URL format".to_string());
                }
                Ok(())
            }
            Self::LocalFolder { path, .. } => {
                if !path.exists() {
                    return Err(format!("Local folder does not exist: {}", path.display()));
                }
                if !path.is_dir() {
                    return Err(format!("Path is not a directory: {}", path.display()));
                }
                Ok(())
            }
        }
    }
}

/// Metadata for a chunk of code content with context information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Unique identifier for this chunk
    pub chunk_id: Uuid,
    /// Original file path
    pub file_path: PathBuf,
    /// Parent file path (for chunked files)
    pub parent_file_path: Option<PathBuf>,
    /// Filename without path
    pub filename: String,
    /// File extension
    pub extension: Option<String>,
    /// Chunk number within the file (1-based)
    pub chunk_number: u32,
    /// Starting line number in the original file (1-based)
    pub start_line: u32,
    /// Ending line number in the original file (1-based)
    pub end_line: u32,
    /// Total lines in this chunk
    pub line_count: u32,
    /// Size of the chunk in bytes
    pub size_bytes: u64,
    /// Content of the chunk
    pub content: String,
    /// L1 context (previous + current + next chunk)
    pub l1_context: Option<String>,
    /// L2 context (±2 chunks around current)
    pub l2_context: Option<String>,
    /// Whether this chunk has previous chunks
    pub has_previous: bool,
    /// Whether this chunk has next chunks
    pub has_next: bool,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ChunkMetadata {
    /// Create a new chunk metadata
    pub fn new(
        file_path: PathBuf,
        filename: String,
        chunk_number: u32,
        start_line: u32,
        end_line: u32,
        content: String,
    ) -> Self {
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_string());

        Self {
            chunk_id: Uuid::new_v4(),
            file_path: file_path.clone(),
            parent_file_path: None,
            filename,
            extension,
            chunk_number,
            start_line,
            end_line,
            line_count: end_line - start_line + 1,
            size_bytes: content.len() as u64,
            content,
            l1_context: None,
            l2_context: None,
            has_previous: chunk_number > 1,
            has_next: false, // Will be set during processing
            created_at: chrono::Utc::now(),
        }
    }

    /// Set the L1 context (previous + current + next chunk)
    pub fn set_l1_context(&mut self, context: String) {
        self.l1_context = Some(context);
    }

    /// Set the L2 context (±2 chunks around current)
    pub fn set_l2_context(&mut self, context: String) {
        self.l2_context = Some(context);
    }

    /// Mark whether this chunk has a next chunk
    pub fn set_has_next(&mut self, has_next: bool) {
        self.has_next = has_next;
    }

    /// Get the chunk identifier string for file naming
    pub fn chunk_identifier(&self) -> String {
        format!("{}_{}", self.filename, self.chunk_number)
    }

    /// Get the content file path for this chunk
    pub fn content_file_path(&self, base_dir: &PathBuf, table_name: &str) -> PathBuf {
        base_dir.join(format!("{}_{}_Content.txt", table_name, self.chunk_number))
    }

    /// Get the L1 content file path for this chunk
    pub fn l1_content_file_path(&self, base_dir: &PathBuf, table_name: &str) -> PathBuf {
        base_dir.join(format!("{}_{}_Content_L1.txt", table_name, self.chunk_number))
    }

    /// Get the L2 content file path for this chunk
    pub fn l2_content_file_path(&self, base_dir: &PathBuf, table_name: &str) -> PathBuf {
        base_dir.join(format!("{}_{}_Content_L2.txt", table_name, self.chunk_number))
    }
}

/// Hierarchical task structure with configurable levels and grouping
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskHierarchy {
    /// Total number of hierarchy levels
    pub levels: u8,
    /// Number of groups per level
    pub groups_per_level: u8,
    /// Total number of tasks to distribute
    pub total_tasks: usize,
    /// Root level tasks
    pub root_tasks: Vec<Task>,
    /// Metadata about the hierarchy
    pub metadata: TaskHierarchyMetadata,
}

impl TaskHierarchy {
    /// Create a new task hierarchy
    pub fn new(levels: u8, groups_per_level: u8, total_tasks: usize) -> Self {
        Self {
            levels,
            groups_per_level,
            total_tasks,
            root_tasks: Vec::new(),
            metadata: TaskHierarchyMetadata::new(levels, groups_per_level, total_tasks),
        }
    }

    /// Add a root level task
    pub fn add_root_task(&mut self, task: Task) {
        self.root_tasks.push(task);
    }

    /// Get all tasks flattened into a single list
    pub fn flatten_tasks(&self) -> Vec<&Task> {
        let mut tasks = Vec::new();
        for root_task in &self.root_tasks {
            tasks.push(root_task);
            self.collect_subtasks(root_task, &mut tasks);
        }
        tasks
    }

    /// Recursively collect all subtasks
    fn collect_subtasks<'a>(&self, task: &'a Task, tasks: &mut Vec<&'a Task>) {
        for subtask in &task.subtasks {
            tasks.push(subtask);
            self.collect_subtasks(subtask, tasks);
        }
    }

    /// Validate the hierarchy structure
    pub fn validate(&self) -> Result<(), String> {
        if self.levels == 0 {
            return Err("Hierarchy must have at least 1 level".to_string());
        }
        if self.groups_per_level == 0 {
            return Err("Groups per level must be at least 1".to_string());
        }
        if self.total_tasks == 0 {
            return Err("Total tasks must be at least 1".to_string());
        }

        // Validate that we have the expected number of root tasks
        let expected_root_tasks = std::cmp::min(self.groups_per_level as usize, self.total_tasks);
        if self.root_tasks.len() != expected_root_tasks {
            return Err(format!(
                "Expected {} root tasks, found {}",
                expected_root_tasks,
                self.root_tasks.len()
            ));
        }

        Ok(())
    }

    /// Get statistics about the hierarchy
    pub fn statistics(&self) -> TaskHierarchyStatistics {
        let flattened = self.flatten_tasks();
        let total_tasks = flattened.len();
        let leaf_tasks = flattened.iter().filter(|t| t.subtasks.is_empty()).count();
        let branch_tasks = total_tasks - leaf_tasks;

        TaskHierarchyStatistics {
            total_tasks,
            leaf_tasks,
            branch_tasks,
            max_depth: self.calculate_max_depth(),
            avg_children_per_branch: if branch_tasks > 0 {
                flattened
                    .iter()
                    .filter(|t| !t.subtasks.is_empty())
                    .map(|t| t.subtasks.len())
                    .sum::<usize>() as f64
                    / branch_tasks as f64
            } else {
                0.0
            },
        }
    }

    /// Calculate the maximum depth of the hierarchy
    fn calculate_max_depth(&self) -> usize {
        self.root_tasks
            .iter()
            .map(|task| self.calculate_task_depth(task, 1))
            .max()
            .unwrap_or(0)
    }

    /// Calculate the depth of a specific task
    fn calculate_task_depth(&self, task: &Task, current_depth: usize) -> usize {
        if task.subtasks.is_empty() {
            current_depth
        } else {
            task.subtasks
                .iter()
                .map(|subtask| self.calculate_task_depth(subtask, current_depth + 1))
                .max()
                .unwrap_or(current_depth)
        }
    }
}

/// Individual task within the hierarchy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Task description/title
    pub description: String,
    /// Task number in hierarchy (e.g., "1.2.3")
    pub task_number: String,
    /// Level in the hierarchy (0-based)
    pub level: u8,
    /// Content files associated with this task
    pub content_files: Vec<ContentFileReference>,
    /// Prompt file reference
    pub prompt_file: Option<PathBuf>,
    /// Output file path
    pub output_file: Option<PathBuf>,
    /// Subtasks
    pub subtasks: Vec<Task>,
    /// Whether this task is completed
    pub completed: bool,
    /// Task metadata
    pub metadata: TaskMetadata,
}

impl Task {
    /// Create a new task
    pub fn new(
        id: String,
        description: String,
        task_number: String,
        level: u8,
    ) -> Self {
        Self {
            id: id.clone(),
            description,
            task_number,
            level,
            content_files: Vec::new(),
            prompt_file: None,
            output_file: None,
            subtasks: Vec::new(),
            completed: false,
            metadata: TaskMetadata::new(id),
        }
    }

    /// Add a content file reference
    pub fn add_content_file(&mut self, content_file: ContentFileReference) {
        self.content_files.push(content_file);
    }

    /// Set the prompt file
    pub fn set_prompt_file(&mut self, prompt_file: PathBuf) {
        self.prompt_file = Some(prompt_file);
    }

    /// Set the output file
    pub fn set_output_file(&mut self, output_file: PathBuf) {
        self.output_file = Some(output_file);
    }

    /// Add a subtask
    pub fn add_subtask(&mut self, subtask: Task) {
        self.subtasks.push(subtask);
    }

    /// Mark task as completed
    pub fn mark_completed(&mut self) {
        self.completed = true;
        self.metadata.completed_at = Some(chrono::Utc::now());
    }

    /// Check if this is a leaf task (no subtasks)
    pub fn is_leaf(&self) -> bool {
        self.subtasks.is_empty()
    }

    /// Get the total number of subtasks (recursive)
    pub fn total_subtasks(&self) -> usize {
        self.subtasks.len() + self.subtasks.iter().map(|t| t.total_subtasks()).sum::<usize>()
    }
}

/// Reference to a content file with its role in the analysis
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentFileReference {
    /// File path
    pub file_path: PathBuf,
    /// Role of this file in the analysis (A, B, C, etc.)
    pub role: String,
    /// Description of the file's purpose
    pub description: String,
    /// File type (Content, L1, L2, etc.)
    pub file_type: ContentFileType,
}

impl ContentFileReference {
    /// Create a new content file reference
    pub fn new(
        file_path: PathBuf,
        role: String,
        description: String,
        file_type: ContentFileType,
    ) -> Self {
        Self {
            file_path,
            role,
            description,
            file_type,
        }
    }
}

/// Type of content file
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentFileType {
    /// Primary content file (A)
    Content,
    /// L1 context file (B)
    L1Context,
    /// L2 context file (C)
    L2Context,
    /// Custom content type
    Custom(String),
}

impl ContentFileType {
    /// Get the display name for this file type
    pub fn display_name(&self) -> &str {
        match self {
            Self::Content => "Content",
            Self::L1Context => "L1 Context",
            Self::L2Context => "L2 Context",
            Self::Custom(name) => name,
        }
    }
}

/// Configuration for task generation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Database table name to process
    pub table_name: String,
    /// Number of hierarchy levels
    pub levels: u8,
    /// Number of groups per level
    pub groups: u8,
    /// Output file path for the task list
    pub output_file: PathBuf,
    /// Optional prompt file path
    pub prompt_file: Option<PathBuf>,
    /// Optional chunk size for chunked analysis
    pub chunk_size: Option<usize>,
    /// Output directory for content files
    pub output_dir: PathBuf,
    /// Work area directory for analysis outputs
    pub work_area_dir: PathBuf,
    /// Whether to enable chunked processing
    pub enable_chunking: bool,
    /// Maximum number of tasks to generate
    pub max_tasks: Option<usize>,
    /// Configuration metadata
    pub metadata: GenerationConfigMetadata,
}

impl GenerationConfig {
    /// Create a new generation configuration
    pub fn new(
        table_name: String,
        levels: u8,
        groups: u8,
        output_file: PathBuf,
    ) -> Self {
        let output_dir = PathBuf::from(".raw_data_202509");
        let work_area_dir = PathBuf::from("gringotts/WorkArea");

        Self {
            table_name: table_name.clone(),
            levels,
            groups,
            output_file,
            prompt_file: None,
            chunk_size: None,
            output_dir,
            work_area_dir,
            enable_chunking: false,
            max_tasks: None,
            metadata: GenerationConfigMetadata::new(table_name),
        }
    }

    /// Enable chunked processing with the specified chunk size
    pub fn with_chunking(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self.enable_chunking = true;
        self
    }

    /// Set the prompt file
    pub fn with_prompt_file(mut self, prompt_file: PathBuf) -> Self {
        self.prompt_file = Some(prompt_file);
        self
    }

    /// Set custom output directory
    pub fn with_output_dir(mut self, output_dir: PathBuf) -> Self {
        self.output_dir = output_dir;
        self
    }

    /// Set custom work area directory
    pub fn with_work_area_dir(mut self, work_area_dir: PathBuf) -> Self {
        self.work_area_dir = work_area_dir;
        self
    }

    /// Set maximum number of tasks
    pub fn with_max_tasks(mut self, max_tasks: usize) -> Self {
        self.max_tasks = Some(max_tasks);
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.table_name.is_empty() {
            return Err("Table name cannot be empty".to_string());
        }
        if self.levels == 0 {
            return Err("Levels must be at least 1".to_string());
        }
        if self.groups == 0 {
            return Err("Groups must be at least 1".to_string());
        }
        if let Some(chunk_size) = self.chunk_size {
            if chunk_size == 0 {
                return Err("Chunk size must be greater than 0".to_string());
            }
        }
        if let Some(max_tasks) = self.max_tasks {
            if max_tasks == 0 {
                return Err("Max tasks must be greater than 0".to_string());
            }
        }
        Ok(())
    }

    /// Get the chunked table name if chunking is enabled
    pub fn chunked_table_name(&self) -> Option<String> {
        if let Some(chunk_size) = self.chunk_size {
            Some(format!("{}_{}", self.table_name, chunk_size))
        } else {
            None
        }
    }

    /// Get the effective table name (chunked or original)
    pub fn effective_table_name(&self) -> String {
        self.chunked_table_name().unwrap_or_else(|| self.table_name.clone())
    }
}

/// Metadata for task hierarchy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskHierarchyMetadata {
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Configuration used to create this hierarchy
    pub levels: u8,
    pub groups_per_level: u8,
    pub total_tasks: usize,
    /// Generation statistics
    pub generation_duration: Option<Duration>,
}

impl TaskHierarchyMetadata {
    /// Create new metadata
    pub fn new(levels: u8, groups_per_level: u8, total_tasks: usize) -> Self {
        Self {
            created_at: chrono::Utc::now(),
            levels,
            groups_per_level,
            total_tasks,
            generation_duration: None,
        }
    }

    /// Set the generation duration
    pub fn set_generation_duration(&mut self, duration: Duration) {
        self.generation_duration = Some(duration);
    }
}

/// Statistics about a task hierarchy
#[derive(Debug, Clone, PartialEq)]
pub struct TaskHierarchyStatistics {
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of leaf tasks (no subtasks)
    pub leaf_tasks: usize,
    /// Number of branch tasks (have subtasks)
    pub branch_tasks: usize,
    /// Maximum depth of the hierarchy
    pub max_depth: usize,
    /// Average number of children per branch task
    pub avg_children_per_branch: f64,
}

/// Metadata for individual tasks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Task ID
    pub task_id: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Completion timestamp
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Estimated effort (in minutes)
    pub estimated_effort_minutes: Option<u32>,
    /// Actual effort (in minutes)
    pub actual_effort_minutes: Option<u32>,
    /// Task priority (1-5, 5 being highest)
    pub priority: Option<u8>,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TaskMetadata {
    /// Create new task metadata
    pub fn new(task_id: String) -> Self {
        Self {
            task_id,
            created_at: chrono::Utc::now(),
            completed_at: None,
            estimated_effort_minutes: None,
            actual_effort_minutes: None,
            priority: None,
            tags: Vec::new(),
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Set estimated effort
    pub fn set_estimated_effort(&mut self, minutes: u32) {
        self.estimated_effort_minutes = Some(minutes);
    }

    /// Set actual effort
    pub fn set_actual_effort(&mut self, minutes: u32) {
        self.actual_effort_minutes = Some(minutes);
    }

    /// Set priority
    pub fn set_priority(&mut self, priority: u8) {
        self.priority = Some(priority.clamp(1, 5));
    }
}

/// Metadata for generation configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationConfigMetadata {
    /// Configuration ID
    pub config_id: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Version of the configuration format
    pub version: String,
    /// Description of this configuration
    pub description: Option<String>,
}

impl GenerationConfigMetadata {
    /// Create new configuration metadata
    pub fn new(config_id: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            config_id,
            created_at: now,
            modified_at: now,
            version: "1.0.0".to_string(),
            description: None,
        }
    }

    /// Update the modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = chrono::Utc::now();
    }

    /// Set description
    pub fn set_description(&mut self, description: String) {
        self.description = Some(description);
        self.touch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_ingestion_source_creation() {
        // Test Git repository source
        let git_source = IngestionSource::git_repository("https://github.com/user/repo.git");
        assert!(git_source.is_git_repository());
        assert!(!git_source.is_local_folder());
        assert_eq!(git_source.display_name(), "https://github.com/user/repo.git");

        // Test local folder source
        let temp_dir = std::env::temp_dir();
        let folder_source = IngestionSource::local_folder(&temp_dir);
        assert!(!folder_source.is_git_repository());
        assert!(folder_source.is_local_folder());
        assert_eq!(folder_source.display_name(), temp_dir.display().to_string());
    }

    #[test]
    fn test_ingestion_source_validation() {
        // Valid Git repository
        let git_source = IngestionSource::git_repository("https://github.com/user/repo.git");
        assert!(git_source.validate().is_ok());

        // Invalid Git repository (empty URL)
        let invalid_git = IngestionSource::GitRepository {
            url: String::new(),
            branch: None,
            token: None,
        };
        assert!(invalid_git.validate().is_err());

        // Valid local folder (temp dir should exist)
        let temp_dir = std::env::temp_dir();
        let folder_source = IngestionSource::local_folder(&temp_dir);
        assert!(folder_source.validate().is_ok());

        // Invalid local folder (non-existent path)
        let invalid_folder = IngestionSource::local_folder("/non/existent/path");
        assert!(invalid_folder.validate().is_err());
    }

    #[test]
    fn test_chunk_metadata_creation() {
        let file_path = PathBuf::from("src/main.rs");
        let chunk = ChunkMetadata::new(
            file_path.clone(),
            "main.rs".to_string(),
            1,
            1,
            50,
            "fn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
        );

        assert_eq!(chunk.file_path, file_path);
        assert_eq!(chunk.filename, "main.rs");
        assert_eq!(chunk.extension, Some("rs".to_string()));
        assert_eq!(chunk.chunk_number, 1);
        assert_eq!(chunk.start_line, 1);
        assert_eq!(chunk.end_line, 50);
        assert_eq!(chunk.line_count, 50);
        assert!(chunk.has_previous == false); // First chunk
        assert_eq!(chunk.chunk_identifier(), "main.rs_1");
    }

    #[test]
    fn test_task_hierarchy_creation() {
        let mut hierarchy = TaskHierarchy::new(3, 5, 25);
        
        // Add a root task
        let root_task = Task::new(
            "task_1".to_string(),
            "Analyze main components".to_string(),
            "1".to_string(),
            0,
        );
        hierarchy.add_root_task(root_task);

        assert_eq!(hierarchy.levels, 3);
        assert_eq!(hierarchy.groups_per_level, 5);
        assert_eq!(hierarchy.total_tasks, 25);
        assert_eq!(hierarchy.root_tasks.len(), 1);
    }

    #[test]
    fn test_generation_config_creation() {
        let config = GenerationConfig::new(
            "INGEST_20250928101039".to_string(),
            4,
            7,
            PathBuf::from("tasks.md"),
        );

        assert_eq!(config.table_name, "INGEST_20250928101039");
        assert_eq!(config.levels, 4);
        assert_eq!(config.groups, 7);
        assert_eq!(config.output_file, PathBuf::from("tasks.md"));
        assert!(!config.enable_chunking);
        assert_eq!(config.effective_table_name(), "INGEST_20250928101039");
    }

    #[test]
    fn test_generation_config_with_chunking() {
        let config = GenerationConfig::new(
            "INGEST_20250928101039".to_string(),
            4,
            7,
            PathBuf::from("tasks.md"),
        )
        .with_chunking(300);

        assert!(config.enable_chunking);
        assert_eq!(config.chunk_size, Some(300));
        assert_eq!(config.chunked_table_name(), Some("INGEST_20250928101039_300".to_string()));
        assert_eq!(config.effective_table_name(), "INGEST_20250928101039_300");
    }

    #[test]
    fn test_generation_config_validation() {
        // Valid configuration
        let valid_config = GenerationConfig::new(
            "INGEST_20250928101039".to_string(),
            4,
            7,
            PathBuf::from("tasks.md"),
        );
        assert!(valid_config.validate().is_ok());

        // Invalid configuration (empty table name)
        let invalid_config = GenerationConfig::new(
            String::new(),
            4,
            7,
            PathBuf::from("tasks.md"),
        );
        assert!(invalid_config.validate().is_err());

        // Invalid configuration (zero levels)
        let invalid_config = GenerationConfig::new(
            "INGEST_20250928101039".to_string(),
            0,
            7,
            PathBuf::from("tasks.md"),
        );
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_task_operations() {
        let mut task = Task::new(
            "task_1_1".to_string(),
            "Analyze core module".to_string(),
            "1.1".to_string(),
            1,
        );

        // Add content file
        let content_file = ContentFileReference::new(
            PathBuf::from("content.txt"),
            "A".to_string(),
            "Primary content".to_string(),
            ContentFileType::Content,
        );
        task.add_content_file(content_file);

        // Add subtask
        let subtask = Task::new(
            "task_1_1_1".to_string(),
            "Analyze functions".to_string(),
            "1.1.1".to_string(),
            2,
        );
        task.add_subtask(subtask);

        assert_eq!(task.content_files.len(), 1);
        assert_eq!(task.subtasks.len(), 1);
        assert!(!task.is_leaf());
        assert_eq!(task.total_subtasks(), 1);
        assert!(!task.completed);

        // Mark as completed
        task.mark_completed();
        assert!(task.completed);
        assert!(task.metadata.completed_at.is_some());
    }

    #[test]
    fn test_content_file_type_display() {
        assert_eq!(ContentFileType::Content.display_name(), "Content");
        assert_eq!(ContentFileType::L1Context.display_name(), "L1 Context");
        assert_eq!(ContentFileType::L2Context.display_name(), "L2 Context");
        assert_eq!(ContentFileType::Custom("Custom".to_string()).display_name(), "Custom");
    }
}