//! Windowed Task Manager for handling large task sets in manageable chunks
//!
//! This module provides a windowed approach to task management, where large task sets
//! are stored in a master list and presented to users in manageable windows (e.g., 50 tasks).
//! Users can advance through windows as they complete work, with automatic progress tracking.

use crate::error::{TaskError, TaskResult};
use crate::tasks::hierarchical_task_divider::{TaskHierarchy, AnalysisTask};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Configuration for windowed task management
#[derive(Debug, Clone)]
pub struct WindowedTaskConfig {
    /// Directory to store all windowed task files
    pub task_dir: PathBuf,
    /// Number of tasks per window
    pub window_size: usize,
    /// Table name for identification
    pub table_name: String,
}

/// Progress information for windowed tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProgress {
    /// Source table name
    pub table_name: String,
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: usize,
    /// Current window information
    pub current_window: WindowInfo,
    /// Completion percentage (0.0 to 100.0)
    pub completion_percentage: f64,
    /// When the task system was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Information about a specific window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowInfo {
    /// Starting task number (1-based)
    pub start: usize,
    /// Ending task number (inclusive)
    pub end: usize,
    /// Current status of this window
    pub status: WindowStatus,
    /// Window number (1-based)
    pub window_number: usize,
}

/// Status of a task window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowStatus {
    /// Currently active window
    Active,
    /// Completed window
    Completed,
    /// Future window (not yet active)
    Pending,
}

/// Individual task entry in the master list
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterTaskEntry {
    /// Unique task ID
    pub task_id: String,
    /// Hierarchical task number (e.g., "1.2.3.4")
    pub task_number: String,
    /// Task description
    pub description: String,
    /// Current status
    pub status: TaskStatus,
    /// When this task was completed (if applicable)
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Status of an individual task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending
    Pending,
    /// Task is currently in active window
    Active,
    /// Task has been completed
    Completed,
    /// Task was skipped
    Skipped,
}

/// Windowed task manager for handling large task sets
pub struct WindowedTaskManager {
    /// Configuration
    config: WindowedTaskConfig,
    /// Current progress state
    progress: TaskProgress,
}

impl WindowedTaskManager {
    /// Create a new windowed task manager
    pub async fn new(config: WindowedTaskConfig, total_tasks: usize) -> TaskResult<Self> {
        // Ensure task directory exists
        fs::create_dir_all(&config.task_dir).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: config.task_dir.display().to_string(),
                cause: format!("Failed to create task directory: {}", e),
                suggestion: "Check directory permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        // Create initial progress state
        let progress = TaskProgress {
            table_name: config.table_name.clone(),
            total_tasks,
            completed_tasks: 0,
            current_window: WindowInfo {
                start: 1,
                end: std::cmp::min(config.window_size, total_tasks),
                status: WindowStatus::Active,
                window_number: 1,
            },
            completion_percentage: 0.0,
            created_at: chrono::Utc::now(),
            last_updated: chrono::Utc::now(),
        };

        Ok(Self { config, progress })
    }

    /// Load existing windowed task manager from disk
    pub async fn load(task_dir: PathBuf) -> TaskResult<Self> {
        let progress_file = task_dir.join("progress.json");
        
        if !progress_file.exists() {
            return Err(TaskError::TaskFileCreationFailed {
                path: progress_file.display().to_string(),
                cause: "Progress file not found".to_string(),
                suggestion: "Initialize the windowed task system first".to_string(),
                source: None,
            });
        }

        let progress_content = fs::read_to_string(&progress_file).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: progress_file.display().to_string(),
                cause: format!("Failed to read progress file: {}", e),
                suggestion: "Check file permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        let progress: TaskProgress = serde_json::from_str(&progress_content).map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: progress_file.display().to_string(),
                cause: format!("Failed to parse progress file: {}", e),
                suggestion: "Progress file may be corrupted".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        let config = WindowedTaskConfig {
            task_dir: task_dir.clone(),
            window_size: progress.current_window.end - progress.current_window.start + 1,
            table_name: progress.table_name.clone(),
        };

        Ok(Self { config, progress })
    }

    /// Generate the complete windowed task system from a hierarchy
    pub async fn generate_from_hierarchy(&mut self, hierarchy: &TaskHierarchy) -> TaskResult<()> {
        info!("Generating windowed task system for {} tasks", hierarchy.total_tasks);

        // Create master task list
        let master_tasks = self.extract_tasks_from_hierarchy(hierarchy)?;
        self.save_master_task_list(&master_tasks).await?;

        // Create initial current window
        self.create_current_window(&master_tasks).await?;

        // Save progress
        self.save_progress().await?;

        // Create completed directory
        let completed_dir = self.config.task_dir.join("completed");
        fs::create_dir_all(&completed_dir).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: completed_dir.display().to_string(),
                cause: format!("Failed to create completed directory: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        info!("Windowed task system generated successfully");
        Ok(())
    }

    /// Advance to the next window
    pub async fn advance_window(&mut self) -> TaskResult<bool> {
        debug!("Advancing window from {}-{}", 
               self.progress.current_window.start, 
               self.progress.current_window.end);

        // Archive current window
        self.archive_current_window().await?;

        // Update completed count
        let window_size = self.progress.current_window.end - self.progress.current_window.start + 1;
        self.progress.completed_tasks += window_size;

        // Check if we're done
        if self.progress.completed_tasks >= self.progress.total_tasks {
            info!("All tasks completed!");
            return Ok(false); // No more windows
        }

        // Calculate next window
        let next_start = self.progress.current_window.end + 1;
        let next_end = std::cmp::min(
            next_start + self.config.window_size - 1,
            self.progress.total_tasks
        );

        self.progress.current_window = WindowInfo {
            start: next_start,
            end: next_end,
            status: WindowStatus::Active,
            window_number: self.progress.current_window.window_number + 1,
        };

        // Update completion percentage
        self.progress.completion_percentage = 
            (self.progress.completed_tasks as f64 / self.progress.total_tasks as f64) * 100.0;
        self.progress.last_updated = chrono::Utc::now();

        // Load master tasks and create new current window
        let master_tasks = self.load_master_task_list().await?;
        self.create_current_window(&master_tasks).await?;

        // Save updated progress
        self.save_progress().await?;

        info!("Advanced to window {} (tasks {}-{})", 
              self.progress.current_window.window_number,
              self.progress.current_window.start,
              self.progress.current_window.end);

        Ok(true) // More windows available
    }

    /// Get current progress information
    pub fn get_progress(&self) -> &TaskProgress {
        &self.progress
    }

    /// Extract tasks from hierarchy into master task entries
    fn extract_tasks_from_hierarchy(&self, hierarchy: &TaskHierarchy) -> TaskResult<Vec<MasterTaskEntry>> {
        let mut master_tasks = Vec::new();
        let mut task_counter = 1;

        // Recursively extract all tasks
        for level in &hierarchy.levels {
            for group in &level.groups {
                self.extract_tasks_from_group(group, &mut master_tasks, &mut task_counter)?;
            }
        }

        Ok(master_tasks)
    }

    /// Extract tasks from a hierarchical group recursively
    fn extract_tasks_from_group(
        &self,
        group: &crate::tasks::hierarchical_task_divider::HierarchicalTaskGroup,
        master_tasks: &mut Vec<MasterTaskEntry>,
        task_counter: &mut usize,
    ) -> TaskResult<()> {
        // Add individual tasks
        for task in &group.tasks {
            let entry = MasterTaskEntry {
                task_id: format!("TASK_{:04}", task_counter),
                task_number: task.id.clone(),
                description: format!("Analyze {} row {}", task.table_name, task.row_number),
                status: TaskStatus::Pending,
                completed_at: None,
            };
            master_tasks.push(entry);
            *task_counter += 1;
        }

        // Process sub-groups recursively
        for sub_group in &group.sub_groups {
            self.extract_tasks_from_group(sub_group, master_tasks, task_counter)?;
        }

        Ok(())
    }

    /// Save master task list to disk
    async fn save_master_task_list(&self, master_tasks: &[MasterTaskEntry]) -> TaskResult<()> {
        let master_file = self.config.task_dir.join("master-tasks.json");
        let content = serde_json::to_string_pretty(master_tasks).map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: master_file.display().to_string(),
                cause: format!("Failed to serialize master tasks: {}", e),
                suggestion: "Check task data integrity".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        fs::write(&master_file, content).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: master_file.display().to_string(),
                cause: format!("Failed to write master tasks: {}", e),
                suggestion: "Check directory permissions and disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        debug!("Saved master task list with {} tasks", master_tasks.len());
        Ok(())
    }

    /// Load master task list from disk
    async fn load_master_task_list(&self) -> TaskResult<Vec<MasterTaskEntry>> {
        let master_file = self.config.task_dir.join("master-tasks.json");
        let content = fs::read_to_string(&master_file).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: master_file.display().to_string(),
                cause: format!("Failed to read master tasks: {}", e),
                suggestion: "Ensure master task list exists".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        let master_tasks: Vec<MasterTaskEntry> = serde_json::from_str(&content).map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: master_file.display().to_string(),
                cause: format!("Failed to parse master tasks: {}", e),
                suggestion: "Master task file may be corrupted".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        Ok(master_tasks)
    }

    /// Create current window markdown file
    async fn create_current_window(&self, master_tasks: &[MasterTaskEntry]) -> TaskResult<()> {
        let window_file = self.config.task_dir.join("current-window.md");
        let mut markdown = String::new();

        // Add window header
        markdown.push_str(&format!(
            "<!-- Window: Tasks {}-{} of {} total ({:.1}% complete) -->\n\n",
            self.progress.current_window.start,
            self.progress.current_window.end,
            self.progress.total_tasks,
            self.progress.completion_percentage
        ));

        // Add tasks for current window
        let start_idx = self.progress.current_window.start - 1; // Convert to 0-based
        let end_idx = std::cmp::min(self.progress.current_window.end, master_tasks.len());

        for i in start_idx..end_idx {
            if let Some(task) = master_tasks.get(i) {
                markdown.push_str(&format!("- [ ] {}. {}\n", task.task_number, task.description));
            }
        }

        // Add footer with next window info
        if self.progress.current_window.end < self.progress.total_tasks {
            let next_start = self.progress.current_window.end + 1;
            let next_end = std::cmp::min(
                next_start + self.config.window_size - 1,
                self.progress.total_tasks
            );
            markdown.push_str(&format!("\n<!-- Next window: Tasks {}-{} -->\n", next_start, next_end));
        } else {
            markdown.push_str("\n<!-- This is the final window -->\n");
        }

        fs::write(&window_file, markdown).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: window_file.display().to_string(),
                cause: format!("Failed to write current window: {}", e),
                suggestion: "Check directory permissions and disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        debug!("Created current window: tasks {}-{}", 
               self.progress.current_window.start, 
               self.progress.current_window.end);
        Ok(())
    }

    /// Archive the current window to completed directory
    async fn archive_current_window(&self) -> TaskResult<()> {
        let current_file = self.config.task_dir.join("current-window.md");
        let completed_dir = self.config.task_dir.join("completed");
        let archive_file = completed_dir.join(format!(
            "batch-{:03}.md", 
            self.progress.current_window.window_number
        ));

        if current_file.exists() {
            fs::copy(&current_file, &archive_file).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: archive_file.display().to_string(),
                    cause: format!("Failed to archive current window: {}", e),
                    suggestion: "Check directory permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            debug!("Archived window {} to {}", 
                   self.progress.current_window.window_number,
                   archive_file.display());
        }

        Ok(())
    }

    /// Save current progress to disk
    async fn save_progress(&self) -> TaskResult<()> {
        let progress_file = self.config.task_dir.join("progress.json");
        let content = serde_json::to_string_pretty(&self.progress).map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: progress_file.display().to_string(),
                cause: format!("Failed to serialize progress: {}", e),
                suggestion: "Check progress data integrity".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        fs::write(&progress_file, content).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: progress_file.display().to_string(),
                cause: format!("Failed to write progress: {}", e),
                suggestion: "Check directory permissions and disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        debug!("Saved progress: {:.1}% complete", self.progress.completion_percentage);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_windowed_task_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WindowedTaskConfig {
            task_dir: temp_dir.path().to_path_buf(),
            window_size: 10,
            table_name: "TEST_TABLE".to_string(),
        };

        let manager = WindowedTaskManager::new(config, 100).await.unwrap();
        assert_eq!(manager.progress.total_tasks, 100);
        assert_eq!(manager.progress.current_window.start, 1);
        assert_eq!(manager.progress.current_window.end, 10);
    }

    #[tokio::test]
    async fn test_window_advancement() {
        let temp_dir = TempDir::new().unwrap();
        let config = WindowedTaskConfig {
            task_dir: temp_dir.path().to_path_buf(),
            window_size: 10,
            table_name: "TEST_TABLE".to_string(),
        };

        let mut manager = WindowedTaskManager::new(config, 25).await.unwrap();
        
        // Create dummy master tasks
        let master_tasks: Vec<MasterTaskEntry> = (1..=25)
            .map(|i| MasterTaskEntry {
                task_id: format!("TASK_{:04}", i),
                task_number: format!("{}", i),
                description: format!("Task {}", i),
                status: TaskStatus::Pending,
                completed_at: None,
            })
            .collect();
        
        manager.save_master_task_list(&master_tasks).await.unwrap();
        manager.create_current_window(&master_tasks).await.unwrap();

        // Advance window
        let has_more = manager.advance_window().await.unwrap();
        assert!(has_more);
        assert_eq!(manager.progress.current_window.start, 11);
        assert_eq!(manager.progress.current_window.end, 20);
        assert_eq!(manager.progress.completed_tasks, 10);

        // Advance again
        let has_more = manager.advance_window().await.unwrap();
        assert!(has_more);
        assert_eq!(manager.progress.current_window.start, 21);
        assert_eq!(manager.progress.current_window.end, 25);
        assert_eq!(manager.progress.completed_tasks, 20);

        // Final advance
        let has_more = manager.advance_window().await.unwrap();
        assert!(!has_more); // No more windows
        assert_eq!(manager.progress.completed_tasks, 25);
    }
}