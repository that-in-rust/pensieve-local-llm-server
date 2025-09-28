//! Output Directory Manager for Content File Organization
//!
//! This module provides the OutputDirectoryManager that handles creation and management
//! of the `.raw_data_202509/` directory structure, file cleanup, organization utilities,
//! and output path validation with conflict resolution.

use crate::error::{TaskError, TaskResult};
use crate::tasks::models::GenerationConfig;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Configuration for output directory management
#[derive(Debug, Clone)]
pub struct OutputDirectoryConfig {
    /// Base output directory (e.g., ".raw_data_202509")
    pub base_dir: PathBuf,
    /// Whether to create subdirectories by table name
    pub use_table_subdirs: bool,
    /// Whether to create subdirectories by date
    pub use_date_subdirs: bool,
    /// Maximum number of files per directory before creating subdirectories
    pub max_files_per_dir: Option<usize>,
    /// Whether to enable automatic cleanup of old files
    pub enable_auto_cleanup: bool,
    /// Maximum age of files to keep (in days)
    pub max_file_age_days: Option<u32>,
    /// Whether to compress old files
    pub compress_old_files: bool,
}

impl Default for OutputDirectoryConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from(".raw_data_202509"),
            use_table_subdirs: false,
            use_date_subdirs: false,
            max_files_per_dir: Some(1000),
            enable_auto_cleanup: false,
            max_file_age_days: Some(30),
            compress_old_files: false,
        }
    }
}

/// Statistics about the output directory
#[derive(Debug, Clone, Default)]
pub struct DirectoryStatistics {
    /// Total number of files
    pub total_files: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Number of subdirectories
    pub subdirectory_count: usize,
    /// Files by extension
    pub files_by_extension: HashMap<String, usize>,
    /// Files by table name
    pub files_by_table: HashMap<String, usize>,
    /// Oldest file timestamp
    pub oldest_file: Option<chrono::DateTime<chrono::Utc>>,
    /// Newest file timestamp
    pub newest_file: Option<chrono::DateTime<chrono::Utc>>,
}

/// Information about a file conflict
#[derive(Debug, Clone)]
pub struct FileConflict {
    /// Path of the conflicting file
    pub file_path: PathBuf,
    /// Size of existing file
    pub existing_size: u64,
    /// Size of new file
    pub new_size: u64,
    /// Modification time of existing file
    pub existing_modified: chrono::DateTime<chrono::Utc>,
    /// Suggested resolution strategy
    pub suggested_resolution: ConflictResolution,
}

/// Strategies for resolving file conflicts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Overwrite the existing file
    Overwrite,
    /// Create a backup and overwrite
    BackupAndOverwrite,
    /// Skip writing the new file
    Skip,
    /// Create a new file with a different name
    Rename,
}

/// Output directory manager
#[derive(Debug, Clone)]
pub struct OutputDirectoryManager {
    config: OutputDirectoryConfig,
}

impl OutputDirectoryManager {
    /// Create a new output directory manager
    pub fn new(config: OutputDirectoryConfig) -> Self {
        Self { config }
    }

    /// Create a manager with default configuration
    pub fn default() -> Self {
        Self::new(OutputDirectoryConfig::default())
    }

    /// Create a manager for a specific base directory
    pub fn with_base_dir(base_dir: PathBuf) -> Self {
        Self::new(OutputDirectoryConfig {
            base_dir,
            ..OutputDirectoryConfig::default()
        })
    }

    /// Create a manager from generation config
    pub fn from_generation_config(gen_config: &GenerationConfig) -> Self {
        Self::new(OutputDirectoryConfig {
            base_dir: gen_config.output_dir.clone(),
            use_table_subdirs: gen_config.enable_chunking, // Use subdirs for chunked tables
            ..OutputDirectoryConfig::default()
        })
    }

    /// Ensure the output directory structure exists
    pub async fn ensure_directory_structure(&self) -> TaskResult<()> {
        debug!("Ensuring output directory structure: {}", self.config.base_dir.display());

        // Create base directory
        fs::create_dir_all(&self.config.base_dir).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: self.config.base_dir.display().to_string(),
                cause: format!("Failed to create base output directory: {}", e),
                suggestion: "Check directory permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        // Create subdirectories if configured
        if self.config.use_date_subdirs {
            let date_subdir = self.get_date_subdirectory();
            let full_date_path = self.config.base_dir.join(date_subdir);
            fs::create_dir_all(&full_date_path).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: full_date_path.display().to_string(),
                    cause: format!("Failed to create date subdirectory: {}", e),
                    suggestion: "Check directory permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }

        info!("Output directory structure ensured: {}", self.config.base_dir.display());
        Ok(())
    }

    /// Get the appropriate output path for a file
    pub fn get_output_path(&self, filename: &str, table_name: Option<&str>) -> PathBuf {
        let mut path = self.config.base_dir.clone();

        // Add date subdirectory if configured
        if self.config.use_date_subdirs {
            path = path.join(self.get_date_subdirectory());
        }

        // Add table subdirectory if configured
        if self.config.use_table_subdirs {
            if let Some(table) = table_name {
                path = path.join(table);
            }
        }

        path.join(filename)
    }

    /// Validate output path and resolve conflicts
    pub async fn validate_and_resolve_path(
        &self,
        file_path: &Path,
        content_size: usize,
        resolution_strategy: ConflictResolution,
    ) -> TaskResult<PathBuf> {
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: parent.display().to_string(),
                    cause: format!("Failed to create parent directory: {}", e),
                    suggestion: "Check directory permissions and available disk space".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }

        // Check for conflicts
        if file_path.exists() {
            let conflict = self.analyze_file_conflict(file_path, content_size).await?;
            return self.resolve_file_conflict(conflict, resolution_strategy).await;
        }

        Ok(file_path.to_path_buf())
    }

    /// Clean up old files based on configuration
    pub async fn cleanup_old_files(&self) -> TaskResult<CleanupResult> {
        if !self.config.enable_auto_cleanup {
            return Ok(CleanupResult::default());
        }

        debug!("Starting cleanup of old files in: {}", self.config.base_dir.display());

        let mut cleanup_result = CleanupResult::default();
        let cutoff_date = if let Some(max_age) = self.config.max_file_age_days {
            chrono::Utc::now() - chrono::Duration::days(max_age as i64)
        } else {
            return Ok(cleanup_result);
        };

        self.cleanup_directory_recursive(&self.config.base_dir, cutoff_date, &mut cleanup_result).await?;

        info!(
            "Cleanup completed: {} files removed, {} files compressed, {} bytes freed",
            cleanup_result.files_removed,
            cleanup_result.files_compressed,
            cleanup_result.bytes_freed
        );

        Ok(cleanup_result)
    }

    /// Organize files into subdirectories if needed
    pub async fn organize_files(&self) -> TaskResult<OrganizationResult> {
        debug!("Organizing files in: {}", self.config.base_dir.display());

        let mut org_result = OrganizationResult::default();

        // Check if we need to organize based on file count
        if let Some(max_files) = self.config.max_files_per_dir {
            let file_count = self.count_files_in_directory(&self.config.base_dir).await?;
            
            if file_count > max_files {
                info!("Directory has {} files (max: {}), organizing into subdirectories", 
                      file_count, max_files);
                
                self.organize_by_table_name(&mut org_result).await?;
            }
        }

        Ok(org_result)
    }

    /// Get statistics about the output directory
    pub async fn get_directory_statistics(&self) -> TaskResult<DirectoryStatistics> {
        debug!("Collecting directory statistics for: {}", self.config.base_dir.display());

        let mut stats = DirectoryStatistics::default();
        self.collect_statistics_recursive(&self.config.base_dir, &mut stats).await?;

        debug!("Directory statistics: {} files, {} bytes, {} subdirs", 
               stats.total_files, stats.total_size_bytes, stats.subdirectory_count);

        Ok(stats)
    }

    /// Remove all files in the output directory
    pub async fn clear_directory(&self) -> TaskResult<ClearResult> {
        warn!("Clearing all files in directory: {}", self.config.base_dir.display());

        let mut clear_result = ClearResult::default();
        
        if !self.config.base_dir.exists() {
            return Ok(clear_result);
        }

        self.clear_directory_recursive(&self.config.base_dir, &mut clear_result).await?;

        info!("Directory cleared: {} files removed, {} bytes freed", 
              clear_result.files_removed, clear_result.bytes_freed);

        Ok(clear_result)
    }

    /// Create a backup of the output directory
    pub async fn create_backup(&self, backup_path: &Path) -> TaskResult<BackupResult> {
        info!("Creating backup of {} to {}", 
              self.config.base_dir.display(), backup_path.display());

        let mut backup_result = BackupResult::default();

        // Ensure backup parent directory exists
        if let Some(parent) = backup_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: parent.display().to_string(),
                    cause: format!("Failed to create backup parent directory: {}", e),
                    suggestion: "Check directory permissions and available disk space".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }

        // Copy directory recursively
        self.copy_directory_recursive(&self.config.base_dir, backup_path, &mut backup_result).await?;

        info!("Backup completed: {} files copied, {} bytes", 
              backup_result.files_copied, backup_result.bytes_copied);

        Ok(backup_result)
    }

    // Private helper methods

    /// Get date subdirectory name (YYYY-MM-DD format)
    fn get_date_subdirectory(&self) -> String {
        chrono::Utc::now().format("%Y-%m-%d").to_string()
    }

    /// Analyze file conflict
    async fn analyze_file_conflict(&self, file_path: &Path, new_size: usize) -> TaskResult<FileConflict> {
        let metadata = fs::metadata(file_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: file_path.display().to_string(),
                cause: format!("Failed to read existing file metadata: {}", e),
                suggestion: "Check file permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        let existing_size = metadata.len();
        let existing_modified = metadata.modified()
            .map(|time| chrono::DateTime::<chrono::Utc>::from(time))
            .unwrap_or_else(|_| chrono::Utc::now());

        // Suggest resolution strategy based on file characteristics
        let suggested_resolution = if new_size as u64 == existing_size {
            ConflictResolution::Skip // Same size, probably same content
        } else if existing_modified < chrono::Utc::now() - chrono::Duration::hours(1) {
            ConflictResolution::BackupAndOverwrite // Old file, backup and overwrite
        } else {
            ConflictResolution::Rename // Recent file, create new name
        };

        Ok(FileConflict {
            file_path: file_path.to_path_buf(),
            existing_size,
            new_size: new_size as u64,
            existing_modified,
            suggested_resolution,
        })
    }

    /// Resolve file conflict
    async fn resolve_file_conflict(
        &self,
        conflict: FileConflict,
        strategy: ConflictResolution,
    ) -> TaskResult<PathBuf> {
        match strategy {
            ConflictResolution::Overwrite => {
                debug!("Resolving conflict by overwriting: {}", conflict.file_path.display());
                Ok(conflict.file_path)
            }
            ConflictResolution::BackupAndOverwrite => {
                debug!("Resolving conflict by backing up and overwriting: {}", conflict.file_path.display());
                let backup_path = self.create_backup_filename(&conflict.file_path);
                fs::rename(&conflict.file_path, &backup_path).await.map_err(|e| {
                    TaskError::TaskFileCreationFailed {
                        path: conflict.file_path.display().to_string(),
                        cause: format!("Failed to create backup: {}", e),
                        suggestion: "Check file permissions and available disk space".to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;
                debug!("Created backup: {}", backup_path.display());
                Ok(conflict.file_path)
            }
            ConflictResolution::Skip => {
                debug!("Resolving conflict by skipping: {}", conflict.file_path.display());
                Err(TaskError::TaskFileCreationFailed {
                    path: conflict.file_path.display().to_string(),
                    cause: "File already exists and skip strategy was chosen".to_string(),
                    suggestion: "Use a different conflict resolution strategy".to_string(),
                    source: None,
                })
            }
            ConflictResolution::Rename => {
                debug!("Resolving conflict by renaming: {}", conflict.file_path.display());
                let new_path = self.create_unique_filename(&conflict.file_path).await?;
                debug!("Created unique filename: {}", new_path.display());
                Ok(new_path)
            }
        }
    }

    /// Create a backup filename
    fn create_backup_filename(&self, original_path: &Path) -> PathBuf {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        
        if let Some(parent) = original_path.parent() {
            if let Some(stem) = original_path.file_stem() {
                if let Some(extension) = original_path.extension() {
                    return parent.join(format!("{}.backup_{}.{}", 
                        stem.to_string_lossy(), timestamp, extension.to_string_lossy()));
                } else {
                    return parent.join(format!("{}.backup_{}", 
                        stem.to_string_lossy(), timestamp));
                }
            }
        }
        
        // Fallback
        PathBuf::from(format!("{}.backup_{}", 
            original_path.display(), timestamp))
    }

    /// Create a unique filename by adding a counter
    async fn create_unique_filename(&self, original_path: &Path) -> TaskResult<PathBuf> {
        let parent = original_path.parent().unwrap_or(Path::new("."));
        let stem = original_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("file");
        let extension = original_path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        for counter in 1..=9999 {
            let new_filename = if extension.is_empty() {
                format!("{}_{:04}", stem, counter)
            } else {
                format!("{}_{:04}.{}", stem, counter, extension)
            };
            
            let new_path = parent.join(new_filename);
            if !new_path.exists() {
                return Ok(new_path);
            }
        }

        Err(TaskError::TaskFileCreationFailed {
            path: original_path.display().to_string(),
            cause: "Could not create unique filename after 9999 attempts".to_string(),
            suggestion: "Clean up existing files or use a different naming strategy".to_string(),
            source: None,
        })
    }

    /// Count files in a directory (non-recursive)
    async fn count_files_in_directory(&self, dir_path: &Path) -> TaskResult<usize> {
        if !dir_path.exists() {
            return Ok(0);
        }

        let mut count = 0;
        let mut entries = fs::read_dir(dir_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory entry: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })? {
            let metadata = entry.metadata().await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: entry.path().display().to_string(),
                    cause: format!("Failed to read file metadata: {}", e),
                    suggestion: "Check file permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            if metadata.is_file() {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Organize files by table name
    async fn organize_by_table_name(&self, org_result: &mut OrganizationResult) -> TaskResult<()> {
        // This is a placeholder for table-based organization
        // In a real implementation, you would scan files and move them to table-specific subdirectories
        org_result.files_moved = 0;
        org_result.directories_created = 0;
        Ok(())
    }

    /// Cleanup directory recursively
    fn cleanup_directory_recursive<'a>(
        &'a self,
        dir_path: &'a Path,
        cutoff_date: chrono::DateTime<chrono::Utc>,
        cleanup_result: &'a mut CleanupResult,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<()>> + 'a>> {
        Box::pin(async move {
        if !dir_path.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(dir_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory for cleanup: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory entry for cleanup: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })? {
            let path = entry.path();
            let metadata = entry.metadata().await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: path.display().to_string(),
                    cause: format!("Failed to read metadata for cleanup: {}", e),
                    suggestion: "Check file permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            if metadata.is_file() {
                let modified = metadata.modified()
                    .map(|time| chrono::DateTime::<chrono::Utc>::from(time))
                    .unwrap_or_else(|_| chrono::Utc::now());

                if modified < cutoff_date {
                    let file_size = metadata.len();
                    
                    if self.config.compress_old_files {
                        // Placeholder for compression logic
                        cleanup_result.files_compressed += 1;
                    } else {
                        fs::remove_file(&path).await.map_err(|e| {
                            TaskError::TaskFileCreationFailed {
                                path: path.display().to_string(),
                                cause: format!("Failed to remove old file: {}", e),
                                suggestion: "Check file permissions".to_string(),
                                source: Some(Box::new(e)),
                            }
                        })?;
                        
                        cleanup_result.files_removed += 1;
                        cleanup_result.bytes_freed += file_size;
                    }
                }
            } else if metadata.is_dir() {
                self.cleanup_directory_recursive(&path, cutoff_date, cleanup_result).await?;
            }
        }

        Ok(())
        })
    }

    /// Collect statistics recursively
    fn collect_statistics_recursive<'a>(
        &'a self,
        dir_path: &'a Path,
        stats: &'a mut DirectoryStatistics,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<()>> + 'a>> {
        Box::pin(async move {
        if !dir_path.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(dir_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory for statistics: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory entry for statistics: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })? {
            let path = entry.path();
            let metadata = entry.metadata().await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: path.display().to_string(),
                    cause: format!("Failed to read metadata for statistics: {}", e),
                    suggestion: "Check file permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            if metadata.is_file() {
                stats.total_files += 1;
                stats.total_size_bytes += metadata.len();

                // Track by extension
                if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
                    *stats.files_by_extension.entry(extension.to_string()).or_insert(0) += 1;
                }

                // Track file timestamps
                if let Ok(modified) = metadata.modified() {
                    let modified_utc = chrono::DateTime::<chrono::Utc>::from(modified);
                    
                    if stats.oldest_file.is_none() || Some(modified_utc) < stats.oldest_file {
                        stats.oldest_file = Some(modified_utc);
                    }
                    
                    if stats.newest_file.is_none() || Some(modified_utc) > stats.newest_file {
                        stats.newest_file = Some(modified_utc);
                    }
                }
            } else if metadata.is_dir() {
                stats.subdirectory_count += 1;
                self.collect_statistics_recursive(&path, stats).await?;
            }
        }

        Ok(())
        })
    }

    /// Clear directory recursively
    fn clear_directory_recursive<'a>(
        &'a self,
        dir_path: &'a Path,
        clear_result: &'a mut ClearResult,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<()>> + 'a>> {
        Box::pin(async move {
        if !dir_path.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(dir_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory for clearing: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dir_path.display().to_string(),
                cause: format!("Failed to read directory entry for clearing: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })? {
            let path = entry.path();
            let metadata = entry.metadata().await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: path.display().to_string(),
                    cause: format!("Failed to read metadata for clearing: {}", e),
                    suggestion: "Check file permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            if metadata.is_file() {
                let file_size = metadata.len();
                fs::remove_file(&path).await.map_err(|e| {
                    TaskError::TaskFileCreationFailed {
                        path: path.display().to_string(),
                        cause: format!("Failed to remove file: {}", e),
                        suggestion: "Check file permissions".to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;
                
                clear_result.files_removed += 1;
                clear_result.bytes_freed += file_size;
            } else if metadata.is_dir() {
                self.clear_directory_recursive(&path, clear_result).await?;
                fs::remove_dir(&path).await.map_err(|e| {
                    TaskError::TaskFileCreationFailed {
                        path: path.display().to_string(),
                        cause: format!("Failed to remove directory: {}", e),
                        suggestion: "Check directory permissions".to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;
            }
        }

        Ok(())
        })
    }

    /// Copy directory recursively
    fn copy_directory_recursive<'a>(
        &'a self,
        src_path: &'a Path,
        dst_path: &'a Path,
        backup_result: &'a mut BackupResult,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<()>> + 'a>> {
        Box::pin(async move {
        if !src_path.exists() {
            return Ok(());
        }

        fs::create_dir_all(dst_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: dst_path.display().to_string(),
                cause: format!("Failed to create backup directory: {}", e),
                suggestion: "Check directory permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        let mut entries = fs::read_dir(src_path).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: src_path.display().to_string(),
                cause: format!("Failed to read source directory for backup: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: src_path.display().to_string(),
                cause: format!("Failed to read directory entry for backup: {}", e),
                suggestion: "Check directory permissions".to_string(),
                source: Some(Box::new(e)),
            }
        })? {
            let src_file = entry.path();
            let dst_file = dst_path.join(entry.file_name());
            let metadata = entry.metadata().await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: src_file.display().to_string(),
                    cause: format!("Failed to read metadata for backup: {}", e),
                    suggestion: "Check file permissions".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;

            if metadata.is_file() {
                fs::copy(&src_file, &dst_file).await.map_err(|e| {
                    TaskError::TaskFileCreationFailed {
                        path: src_file.display().to_string(),
                        cause: format!("Failed to copy file for backup: {}", e),
                        suggestion: "Check file permissions and available disk space".to_string(),
                        source: Some(Box::new(e)),
                    }
                })?;
                
                backup_result.files_copied += 1;
                backup_result.bytes_copied += metadata.len();
            } else if metadata.is_dir() {
                self.copy_directory_recursive(&src_file, &dst_file, backup_result).await?;
            }
        }

        Ok(())
        })
    }
}

/// Result of cleanup operation
#[derive(Debug, Clone, Default)]
pub struct CleanupResult {
    pub files_removed: usize,
    pub files_compressed: usize,
    pub bytes_freed: u64,
}

/// Result of organization operation
#[derive(Debug, Clone, Default)]
pub struct OrganizationResult {
    pub files_moved: usize,
    pub directories_created: usize,
}

/// Result of clear operation
#[derive(Debug, Clone, Default)]
pub struct ClearResult {
    pub files_removed: usize,
    pub bytes_freed: u64,
}

/// Result of backup operation
#[derive(Debug, Clone, Default)]
pub struct BackupResult {
    pub files_copied: usize,
    pub bytes_copied: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    #[tokio::test]
    async fn test_ensure_directory_structure() {
        let temp_dir = TempDir::new().unwrap();
        let config = OutputDirectoryConfig {
            base_dir: temp_dir.path().join("test_output"),
            ..OutputDirectoryConfig::default()
        };
        
        let manager = OutputDirectoryManager::new(config);
        
        // Directory should not exist initially
        assert!(!manager.config.base_dir.exists());
        
        // Ensure directory structure
        manager.ensure_directory_structure().await.unwrap();
        
        // Directory should now exist
        assert!(manager.config.base_dir.exists());
        assert!(manager.config.base_dir.is_dir());
    }

    #[tokio::test]
    async fn test_get_output_path() {
        let temp_dir = TempDir::new().unwrap();
        let config = OutputDirectoryConfig {
            base_dir: temp_dir.path().join("test_output"),
            use_table_subdirs: true,
            use_date_subdirs: false,
            ..OutputDirectoryConfig::default()
        };
        
        let manager = OutputDirectoryManager::new(config);
        
        let path = manager.get_output_path("test_file.txt", Some("INGEST_20250928101039"));
        
        assert!(path.to_string_lossy().contains("test_output"));
        assert!(path.to_string_lossy().contains("INGEST_20250928101039"));
        assert!(path.to_string_lossy().contains("test_file.txt"));
    }

    #[tokio::test]
    async fn test_conflict_resolution() {
        let temp_dir = TempDir::new().unwrap();
        let config = OutputDirectoryConfig {
            base_dir: temp_dir.path().to_path_buf(),
            ..OutputDirectoryConfig::default()
        };
        
        let manager = OutputDirectoryManager::new(config);
        
        // Create a test file
        let test_file = temp_dir.path().join("test_file.txt");
        fs::write(&test_file, "existing content").await.unwrap();
        
        // Test conflict resolution
        let resolved_path = manager.validate_and_resolve_path(
            &test_file,
            20, // new content size
            ConflictResolution::Rename,
        ).await.unwrap();
        
        // Should get a different path
        assert_ne!(resolved_path, test_file);
        assert!(resolved_path.to_string_lossy().contains("test_file"));
    }

    #[tokio::test]
    async fn test_directory_statistics() {
        let temp_dir = TempDir::new().unwrap();
        let config = OutputDirectoryConfig {
            base_dir: temp_dir.path().to_path_buf(),
            ..OutputDirectoryConfig::default()
        };
        
        let manager = OutputDirectoryManager::new(config);
        
        // Create some test files
        fs::write(temp_dir.path().join("file1.txt"), "content1").await.unwrap();
        fs::write(temp_dir.path().join("file2.rs"), "content2").await.unwrap();
        fs::create_dir(temp_dir.path().join("subdir")).await.unwrap();
        fs::write(temp_dir.path().join("subdir/file3.py"), "content3").await.unwrap();
        
        let stats = manager.get_directory_statistics().await.unwrap();
        
        assert_eq!(stats.total_files, 3);
        assert_eq!(stats.subdirectory_count, 1);
        assert!(stats.files_by_extension.contains_key("txt"));
        assert!(stats.files_by_extension.contains_key("rs"));
        assert!(stats.files_by_extension.contains_key("py"));
    }

    #[tokio::test]
    async fn test_clear_directory() {
        let temp_dir = TempDir::new().unwrap();
        let config = OutputDirectoryConfig {
            base_dir: temp_dir.path().to_path_buf(),
            ..OutputDirectoryConfig::default()
        };
        
        let manager = OutputDirectoryManager::new(config);
        
        // Create some test files
        fs::write(temp_dir.path().join("file1.txt"), "content1").await.unwrap();
        fs::write(temp_dir.path().join("file2.txt"), "content2").await.unwrap();
        
        // Verify files exist
        assert!(temp_dir.path().join("file1.txt").exists());
        assert!(temp_dir.path().join("file2.txt").exists());
        
        // Clear directory
        let result = manager.clear_directory().await.unwrap();
        
        assert_eq!(result.files_removed, 2);
        assert!(result.bytes_freed > 0);
        
        // Verify files are gone
        assert!(!temp_dir.path().join("file1.txt").exists());
        assert!(!temp_dir.path().join("file2.txt").exists());
    }
}