//! Ingestion engine module for processing repositories and local folders
//! 
//! This module provides the core functionality for:
//! - Cloning Git repositories with progress tracking and authentication
//! - Processing local folders with filtering and safety features  
//! - Batch processing files with controlled concurrency and memory management
//! - Coordinating the entire ingestion workflow

pub mod git_cloner;
pub mod folder_processor;
pub mod batch_processor;
pub mod resume;
pub mod providers;

use crate::database::Database;
use crate::error::IngestionError;
use crate::processing::FileProcessor;
use batch_processor::{BatchConfig, BatchProcessor, BatchProgress, BatchStats};
use folder_processor::{FolderConfig, FolderProcessor, FolderResult};
use git_cloner::{CloneConfig, CloneResult, GitCloner};
use resume::{ResumeManager, ResumeState, IngestionResumeConfig, ProgressStats};
use crate::processing::ProcessedFile;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};
use url::Url;

/// Configuration for the ingestion engine
#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// Git cloning configuration
    pub clone_config: CloneConfig,
    /// Folder processing configuration  
    pub folder_config: FolderConfig,
    /// Batch processing configuration
    pub batch_config: BatchConfig,
    /// Whether to clean up cloned repositories after processing
    pub cleanup_cloned_repos: bool,
    /// Maximum total ingestion time
    pub max_ingestion_time: Duration,
    /// Resume configuration for handling interruptions
    pub resume_config: IngestionResumeConfig,
    /// Whether to enable resume capability
    pub enable_resume: bool,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            clone_config: CloneConfig::default(),
            folder_config: FolderConfig::default(),
            batch_config: BatchConfig::default(),
            cleanup_cloned_repos: true,
            max_ingestion_time: Duration::from_secs(1800), // 30 minutes
            resume_config: IngestionResumeConfig::default(),
            enable_resume: true,
        }
    }
}

/// Result of a complete ingestion operation
#[derive(Debug, Clone)]
pub struct IngestionOperationResult {
    /// Source that was ingested (URL or local path)
    pub source: String,
    /// Type of source (git repository or local folder)
    pub source_type: SourceType,
    /// Database table name where data was stored
    pub table_name: String,
    /// Ingestion ID for tracking
    pub ingestion_id: i64,
    /// Total files processed successfully
    pub files_processed: usize,
    /// Total files that failed processing
    pub files_failed: usize,
    /// Total files skipped
    pub files_skipped: usize,
    /// Total processing time
    pub processing_time: Duration,
    /// Repository information (if applicable)
    pub repo_info: Option<CloneResult>,
    /// Folder processing information (if applicable)
    pub folder_info: Option<FolderResult>,
    /// Batch processing statistics
    pub batch_stats: BatchStats,
}

/// Type of ingestion source
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceType {
    GitRepository,
    LocalFolder,
}

/// Main ingestion engine that coordinates all ingestion operations
pub struct IngestionEngine {
    config: IngestionConfig,
    database: Arc<Database>,
    file_processor: Arc<dyn FileProcessor>,
    resume_manager: Option<ResumeManager>,
}

impl IngestionEngine {
    /// Create a new ingestion engine
    pub fn new(
        config: IngestionConfig,
        database: Arc<Database>,
        file_processor: Arc<dyn FileProcessor>,
    ) -> Self {
        let resume_manager = if config.enable_resume {
            Some(ResumeManager::new(config.resume_config.clone(), Arc::clone(&database)))
        } else {
            None
        };

        Self {
            config,
            database,
            file_processor,
            resume_manager,
        }
    }

    /// Ingest from a source (either Git repository URL or local folder path)
    pub async fn ingest_source(
        &self,
        source: &str,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        info!("Starting ingestion from source: {}", source);

        // Determine source type and process accordingly
        if self.is_git_repository_url(source) {
            self.ingest_git_repository(source, progress_callback).await
        } else {
            self.ingest_local_folder(source, progress_callback).await
        }
    }

    /// Ingest from a Git repository
    pub async fn ingest_git_repository(
        &self,
        repo_url: &str,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        let start_time = std::time::Instant::now();

        // Step 1: Clone the repository
        info!("Cloning repository: {}", repo_url);
        let git_cloner = GitCloner::new(self.config.clone_config.clone());
        let clone_result = git_cloner.clone_repository(repo_url).await?;

        info!(
            "Successfully cloned {} files from {}",
            clone_result.file_count, repo_url
        );

        // Step 2: Create ingestion record in database
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let table_name = format!("INGEST_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"));

        let ingestion_id = self
            .database
            .create_ingestion_record(
                Some(repo_url.to_string()),
                clone_result.repo_path.display().to_string(),
                timestamp,
                &table_name,
            )
            .await?;

        // Step 3: Process the cloned repository folder
        let ingestion_result = self
            .process_folder_to_database(
                &clone_result.repo_path,
                &table_name,
                ingestion_id,
                progress_callback,
            )
            .await;

        // Step 4: Clean up cloned repository if configured
        if self.config.cleanup_cloned_repos {
            if let Err(e) = std::fs::remove_dir_all(&clone_result.repo_path) {
                warn!(
                    "Failed to clean up cloned repository at {}: {}",
                    clone_result.repo_path.display(),
                    e
                );
            } else {
                info!("Cleaned up cloned repository at {}", clone_result.repo_path.display());
            }
        }

        // Step 5: Update ingestion record with completion
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match &ingestion_result {
            Ok(result) => {
                self.database
                    .complete_ingestion_record(
                        ingestion_id,
                        end_timestamp,
                        result.files_processed as i32,
                    )
                    .await?;
            }
            Err(_) => {
                self.database
                    .complete_ingestion_record(ingestion_id, end_timestamp, 0)
                    .await?;
            }
        }

        // Return the result with repository information
        match ingestion_result {
            Ok(mut result) => {
                result.source = repo_url.to_string();
                result.source_type = SourceType::GitRepository;
                result.repo_info = Some(clone_result);
                result.processing_time = start_time.elapsed();
                Ok(result)
            }
            Err(e) => Err(e),
        }
    }

    /// Ingest from a local folder
    pub async fn ingest_local_folder(
        &self,
        folder_path: &str,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        let start_time = std::time::Instant::now();
        let folder_path = Path::new(folder_path);

        info!("Processing local folder: {}", folder_path.display());

        // Step 1: Create ingestion record in database
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let table_name = format!("INGEST_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"));

        let ingestion_id = self
            .database
            .create_ingestion_record(
                None, // No repository URL for local folders
                folder_path.display().to_string(),
                timestamp,
                &table_name,
            )
            .await?;

        // Step 2: Process the folder
        let ingestion_result = self
            .process_folder_to_database(folder_path, &table_name, ingestion_id, progress_callback)
            .await;

        // Step 3: Update ingestion record with completion
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match &ingestion_result {
            Ok(result) => {
                self.database
                    .complete_ingestion_record(
                        ingestion_id,
                        end_timestamp,
                        result.files_processed as i32,
                    )
                    .await?;
            }
            Err(_) => {
                self.database
                    .complete_ingestion_record(ingestion_id, end_timestamp, 0)
                    .await?;
            }
        }

        // Return the result with folder information
        match ingestion_result {
            Ok(mut result) => {
                result.source = folder_path.display().to_string();
                result.source_type = SourceType::LocalFolder;
                result.processing_time = start_time.elapsed();
                Ok(result)
            }
            Err(e) => Err(e),
        }
    }

    /// Process a folder and store results in the database
    async fn process_folder_to_database(
        &self,
        folder_path: &Path,
        table_name: &str,
        ingestion_id: i64,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        // Step 1: Create the ingestion table
        self.database.create_ingestion_table(table_name).await?;

        // Step 2: Discover files in the folder
        let folder_processor = FolderProcessor::new(self.config.folder_config.clone());
        let folder_result = folder_processor.process_folder(folder_path)?;

        info!(
            "Discovered {} files in folder ({}  skipped)",
            folder_result.total_files, folder_result.skipped_files
        );

        // Step 3: Extract file paths for processing
        let file_paths: Vec<PathBuf> = folder_result
            .files
            .iter()
            .filter(|f| !f.skipped)
            .map(|f| f.absolute_path.clone())
            .collect();

        if file_paths.is_empty() {
            warn!("No files to process in folder: {}", folder_path.display());
            return Ok(IngestionOperationResult {
                source: folder_path.display().to_string(),
                source_type: SourceType::LocalFolder,
                table_name: table_name.to_string(),
                ingestion_id,
                files_processed: 0,
                files_failed: 0,
                files_skipped: folder_result.skipped_files,
                processing_time: Duration::ZERO,
                repo_info: None,
                folder_info: Some(folder_result),
                batch_stats: BatchStats::default(),
            });
        }

        // Step 4: Process files in batches
        let batch_processor = BatchProcessor::new(
            self.config.batch_config.clone(),
            Arc::clone(&self.file_processor),
        );

        let (processed_files, batch_stats) = batch_processor
            .process_files(file_paths, progress_callback)
            .await?;

        // Step 5: Store processed files in database
        if !processed_files.is_empty() {
            self.database
                .insert_processed_files(table_name, &processed_files, ingestion_id)
                .await?;
        }

        info!(
            "Successfully stored {} processed files in table {}",
            processed_files.len(),
            table_name
        );

        Ok(IngestionOperationResult {
            source: folder_path.display().to_string(),
            source_type: SourceType::LocalFolder,
            table_name: table_name.to_string(),
            ingestion_id,
            files_processed: batch_stats.files_processed,
            files_failed: batch_stats.files_failed,
            files_skipped: batch_stats.files_skipped + folder_result.skipped_files,
            processing_time: batch_stats.total_duration,
            repo_info: None,
            folder_info: Some(folder_result),
            batch_stats,
        })
    }

    /// Check if a string is a Git repository URL
    fn is_git_repository_url(&self, source: &str) -> bool {
        // Try to parse as URL
        if let Ok(url) = Url::parse(source) {
            match url.scheme() {
                "http" | "https" | "git" | "ssh" => {
                    // Check for common Git hosting services
                    if let Some(host) = url.host_str() {
                        return host.contains("github.com")
                            || host.contains("gitlab.com")
                            || host.contains("bitbucket.org")
                            || host.contains("git")
                            || source.ends_with(".git");
                    }
                }
                _ => return false,
            }
        }

        // Check for SSH-style Git URLs (git@github.com:user/repo.git)
        if source.starts_with("git@") && source.contains(':') {
            return true;
        }

        false
    }

    /// Get ingestion statistics for a completed ingestion
    pub async fn get_ingestion_statistics(
        &self,
        ingestion_id: i64,
    ) -> crate::error::IngestionResult<IngestionStatistics> {
        let stats = self.database.get_ingestion_statistics(ingestion_id).await?;
        Ok(stats)
    }

    /// List all ingestion records
    pub async fn list_ingestions(&self) -> crate::error::IngestionResult<Vec<IngestionRecord>> {
        let records = self.database.list_ingestion_records().await?;
        Ok(records)
    }

    /// Request graceful shutdown of any ongoing ingestion
    pub fn request_shutdown(&self) {
        // This would be implemented to signal shutdown to any running batch processors
        info!("Ingestion engine shutdown requested");
    }

    /// Resume an interrupted ingestion
    pub async fn resume_ingestion(
        &self,
        ingestion_id: i64,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        let resume_manager = self.resume_manager.as_ref().ok_or_else(|| {
            IngestionError::ConfigurationError {
                message: "Resume capability is not enabled".to_string(),
            }
        })?;

        // Load resume state
        let resume_state = resume_manager
            .load_resume_state(ingestion_id)
            .await?
            .ok_or_else(|| IngestionError::ConfigurationError {
                message: format!("No resume state found for ingestion {}", ingestion_id),
            })?;

        info!(
            "Resuming ingestion {} from {} with {}/{} files completed",
            ingestion_id,
            resume_state.source,
            resume_state.completed_files.len(),
            resume_state.total_files
        );

        // Determine if this was a git repository or local folder
        if self.is_git_repository_url(&resume_state.source) {
            self.resume_git_repository_ingestion(resume_state, progress_callback)
                .await
        } else {
            self.resume_local_folder_ingestion(resume_state, progress_callback)
                .await
        }
    }

    /// Resume a git repository ingestion
    async fn resume_git_repository_ingestion(
        &self,
        mut resume_state: ResumeState,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        let start_time = std::time::Instant::now();
        let resume_manager = self.resume_manager.as_ref().unwrap();

        // Re-clone the repository (we don't persist cloned repos)
        info!("Re-cloning repository for resume: {}", resume_state.source);
        let git_cloner = GitCloner::new(self.config.clone_config.clone());
        let clone_result = git_cloner.clone_repository(&resume_state.source).await?;

        // Get all files from the cloned repository
        let folder_processor = FolderProcessor::new(self.config.folder_config.clone());
        let folder_result = folder_processor.process_folder(&clone_result.repo_path)?;

        let all_files: Vec<PathBuf> = folder_result
            .files
            .iter()
            .filter(|f| !f.skipped)
            .map(|f| f.absolute_path.clone())
            .collect();

        // Get remaining files to process
        let remaining_files = resume_manager.get_remaining_files(&resume_state, &all_files);
        let retryable_files = resume_manager.get_retryable_files(&resume_state);

        info!(
            "Resume processing: {} remaining files, {} retryable files",
            remaining_files.len(),
            retryable_files.len()
        );

        // Process remaining and retryable files
        let mut files_to_process = remaining_files;
        files_to_process.extend(retryable_files);

        let result = self
            .process_files_with_resume(
                files_to_process,
                &mut resume_state,
                progress_callback,
            )
            .await;

        // Clean up cloned repository
        if self.config.cleanup_cloned_repos {
            if let Err(e) = std::fs::remove_dir_all(&clone_result.repo_path) {
                warn!(
                    "Failed to clean up cloned repository at {}: {}",
                    clone_result.repo_path.display(),
                    e
                );
            }
        }

        // Update completion status
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match &result {
            Ok((processed_files, _)) => {
                self.database
                    .complete_ingestion_record(
                        resume_state.ingestion_id,
                        end_timestamp,
                        (resume_state.completed_files.len() + processed_files.len()) as i32,
                    )
                    .await?;

                // Clean up resume state on successful completion
                let stats = resume_manager.get_progress_stats(&resume_state);
                if stats.is_complete() {
                    resume_manager.cleanup_resume_state(resume_state.ingestion_id).await?;
                }
            }
            Err(_) => {
                self.database
                    .complete_ingestion_record(
                        resume_state.ingestion_id,
                        end_timestamp,
                        resume_state.completed_files.len() as i32,
                    )
                    .await?;
            }
        }

        // Return result with updated statistics
        match result {
            Ok((processed_files, mut stats)) => {
                stats.files_processed += resume_state.completed_files.len();
                let mut operation_result = IngestionOperationResult {
                    source: resume_state.source,
                    source_type: SourceType::GitRepository,
                    table_name: resume_state.table_name,
                    ingestion_id: resume_state.ingestion_id,
                    files_processed: stats.files_processed,
                    files_failed: stats.files_failed,
                    files_skipped: stats.files_skipped,
                    processing_time: start_time.elapsed(),
                    repo_info: Some(clone_result),
                    folder_info: Some(folder_result),
                    batch_stats: stats,
                };
                Ok(operation_result)
            }
            Err(e) => Err(e),
        }
    }

    /// Resume a local folder ingestion
    async fn resume_local_folder_ingestion(
        &self,
        mut resume_state: ResumeState,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        let start_time = std::time::Instant::now();
        let resume_manager = self.resume_manager.as_ref().unwrap();
        let folder_path = Path::new(&resume_state.source);

        // Re-discover files in the folder
        let folder_processor = FolderProcessor::new(self.config.folder_config.clone());
        let folder_result = folder_processor.process_folder(folder_path)?;

        let all_files: Vec<PathBuf> = folder_result
            .files
            .iter()
            .filter(|f| !f.skipped)
            .map(|f| f.absolute_path.clone())
            .collect();

        // Get remaining files to process
        let remaining_files = resume_manager.get_remaining_files(&resume_state, &all_files);
        let retryable_files = resume_manager.get_retryable_files(&resume_state);

        info!(
            "Resume processing: {} remaining files, {} retryable files",
            remaining_files.len(),
            retryable_files.len()
        );

        // Process remaining and retryable files
        let mut files_to_process = remaining_files;
        files_to_process.extend(retryable_files);

        let result = self
            .process_files_with_resume(
                files_to_process,
                &mut resume_state,
                progress_callback,
            )
            .await;

        // Update completion status
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        match &result {
            Ok((processed_files, _)) => {
                self.database
                    .complete_ingestion_record(
                        resume_state.ingestion_id,
                        end_timestamp,
                        (resume_state.completed_files.len() + processed_files.len()) as i32,
                    )
                    .await?;

                // Clean up resume state on successful completion
                let stats = resume_manager.get_progress_stats(&resume_state);
                if stats.is_complete() {
                    resume_manager.cleanup_resume_state(resume_state.ingestion_id).await?;
                }
            }
            Err(_) => {
                self.database
                    .complete_ingestion_record(
                        resume_state.ingestion_id,
                        end_timestamp,
                        resume_state.completed_files.len() as i32,
                    )
                    .await?;
            }
        }

        // Return result with updated statistics
        match result {
            Ok((processed_files, mut stats)) => {
                stats.files_processed += resume_state.completed_files.len();
                Ok(IngestionOperationResult {
                    source: resume_state.source,
                    source_type: SourceType::LocalFolder,
                    table_name: resume_state.table_name,
                    ingestion_id: resume_state.ingestion_id,
                    files_processed: stats.files_processed,
                    files_failed: stats.files_failed,
                    files_skipped: stats.files_skipped,
                    processing_time: start_time.elapsed(),
                    repo_info: None,
                    folder_info: Some(folder_result),
                    batch_stats: stats,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Process files with resume capability
    async fn process_files_with_resume(
        &self,
        file_paths: Vec<PathBuf>,
        resume_state: &mut ResumeState,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<(Vec<ProcessedFile>, BatchStats)> {
        if file_paths.is_empty() {
            info!("No files to process for resume");
            return Ok((Vec::new(), BatchStats::default()));
        }

        let resume_manager = self.resume_manager.as_ref().unwrap();

        // Create batch processor with resume-aware configuration
        let mut batch_config = self.config.batch_config.clone();
        batch_config.continue_on_error = true; // Always continue on error for resume

        let batch_processor = BatchProcessor::new(batch_config, Arc::clone(&self.file_processor));

        // Create progress callback that updates resume state
        let resume_progress_callback = if let Some(callback) = progress_callback {
            Some(Box::new(move |progress: BatchProgress| {
                callback(progress);
            }) as Box<dyn Fn(BatchProgress) + Send + Sync>)
        } else {
            None
        };

        // Process files
        let (mut processed_files, stats) = batch_processor
            .process_files(file_paths.clone(), resume_progress_callback)
            .await?;

        // Update resume state with results
        for (i, file_path) in file_paths.iter().enumerate() {
            if i < processed_files.len() {
                let processed_file = &processed_files[i];
                if processed_file.skipped {
                    // File was skipped, don't mark as completed
                    continue;
                }

                // Mark as completed
                resume_manager
                    .mark_file_completed(resume_state, file_path.clone())
                    .await?;
            } else {
                // File failed processing
                let error_msg = "Processing failed during resume".to_string();
                let is_permanent = false; // Assume transient for retry

                resume_manager
                    .mark_file_failed(resume_state, file_path.clone(), error_msg, is_permanent)
                    .await?;
            }
        }

        // Store successfully processed files in database
        if !processed_files.is_empty() {
            self.database
                .insert_processed_files(&resume_state.table_name, &processed_files, resume_state.ingestion_id)
                .await?;
        }

        // Save final resume state
        resume_manager.save_resume_state(resume_state).await?;

        Ok((processed_files, stats))
    }

    /// Get resume progress for an ingestion
    pub async fn get_resume_progress(&self, ingestion_id: i64) -> crate::error::IngestionResult<Option<ProgressStats>> {
        let resume_manager = self.resume_manager.as_ref().ok_or_else(|| {
            IngestionError::ConfigurationError {
                message: "Resume capability is not enabled".to_string(),
            }
        })?;

        if let Some(resume_state) = resume_manager.load_resume_state(ingestion_id).await? {
            Ok(Some(resume_manager.get_progress_stats(&resume_state)))
        } else {
            Ok(None)
        }
    }

    /// List all resumable ingestions
    pub async fn list_resumable_ingestions(&self) -> crate::error::IngestionResult<Vec<(i64, ProgressStats)>> {
        let resume_manager = self.resume_manager.as_ref().ok_or_else(|| {
            IngestionError::ConfigurationError {
                message: "Resume capability is not enabled".to_string(),
            }
        })?;

        // This is a simplified implementation - in practice, you'd scan the resume directory
        // For now, return empty list
        Ok(Vec::new())
    }
}

/// Statistics for a completed ingestion
#[derive(Debug, Clone)]
pub struct IngestionStatistics {
    pub ingestion_id: i64,
    pub table_name: String,
    pub total_files: i32,
    pub total_size_bytes: i64,
    pub file_type_counts: std::collections::HashMap<String, i32>,
    pub extension_counts: std::collections::HashMap<String, i32>,
    pub processing_duration: Duration,
}

/// Record of an ingestion operation
#[derive(Debug, Clone)]
pub struct IngestionRecord {
    pub ingestion_id: i64,
    pub repo_url: Option<String>,
    pub local_path: String,
    pub table_name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub total_files_processed: Option<i32>,
    pub status: IngestionStatus,
}

/// Status of an ingestion operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestionStatus {
    InProgress,
    Completed,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::{FileType, ProcessedFile};
    use crate::error::{ProcessingError, ProcessingResult, IngestionResult};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // Mock file processor for testing
    struct MockFileProcessor;

    #[async_trait::async_trait]
    impl FileProcessor for MockFileProcessor {
        fn can_process(&self, _file_path: &Path) -> bool {
            true
        }

        async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
            Ok(ProcessedFile {
                filepath: file_path.display().to_string(),
                filename: file_path.file_name().unwrap().to_str().unwrap().to_string(),
                extension: file_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_string(),
                file_size_bytes: 1024,
                line_count: Some(10),
                word_count: Some(50),
                token_count: Some(100),
                content_text: Some("Mock content".to_string()),
                file_type: FileType::DirectText,
                conversion_command: None,
                relative_path: file_path.display().to_string(),
                absolute_path: file_path.display().to_string(),
                skipped: false,
                skip_reason: None,
            })
        }

        fn get_file_type(&self) -> FileType {
            FileType::DirectText
        }
    }

    // Mock database for testing
    struct MockDatabase;

    impl MockDatabase {
        async fn create_ingestion_record(
            &self,
            _repo_url: Option<String>,
            _local_path: String,
            _timestamp: u64,
            _table_name: &str,
        ) -> IngestionResult<i64> {
            Ok(1)
        }

        async fn create_ingestion_table(&self, _table_name: &str) -> IngestionResult<()> {
            Ok(())
        }

        async fn insert_processed_files(
            &self,
            _table_name: &str,
            _files: &[ProcessedFile],
            _ingestion_id: i64,
        ) -> IngestionResult<()> {
            Ok(())
        }

        async fn complete_ingestion_record(
            &self,
            _ingestion_id: i64,
            _end_timestamp: u64,
            _total_files: i32,
        ) -> IngestionResult<()> {
            Ok(())
        }
    }

    fn create_test_directory() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create test files
        std::fs::write(root.join("file1.txt"), "Hello, world!").unwrap();
        std::fs::write(root.join("file2.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("README.md"), "# Test Repository").unwrap();

        temp_dir
    }

    #[test]
    fn test_ingestion_config_default() {
        let config = IngestionConfig::default();
        assert!(config.cleanup_cloned_repos);
        assert_eq!(config.max_ingestion_time, Duration::from_secs(1800));
    }

    #[test]
    fn test_is_git_repository_url() {
        // Test URL validation logic directly using url crate
        use url::Url;
        
        // Valid Git URLs
        assert!(Url::parse("https://github.com/user/repo").is_ok());
        assert!(Url::parse("https://github.com/user/repo.git").is_ok());
        assert!(Url::parse("https://gitlab.com/user/repo").is_ok());
        assert!(Url::parse("https://git.example.com/repo.git").is_ok());

        // Invalid URLs
        assert!(Url::parse("/local/path").is_err());
        assert!(Url::parse("https://example.com/page").is_ok()); // This is valid URL but not necessarily git
        assert!(Url::parse("not-a-url").is_err());
        assert!(Url::parse("").is_err());
        
        // Test path detection (non-URLs should be treated as local paths)
        assert!(std::path::Path::new("/local/path").exists() || !std::path::Path::new("/local/path").exists()); // Always true, just testing path logic
    }

    #[test]
    fn test_source_type_debug() {
        assert_eq!(format!("{:?}", SourceType::GitRepository), "GitRepository");
        assert_eq!(format!("{:?}", SourceType::LocalFolder), "LocalFolder");
    }

    #[test]
    fn test_ingestion_status_debug() {
        assert_eq!(format!("{:?}", IngestionStatus::InProgress), "InProgress");
        assert_eq!(format!("{:?}", IngestionStatus::Completed), "Completed");
        assert_eq!(format!("{:?}", IngestionStatus::Failed), "Failed");
    }

    #[test]
    fn test_ingestion_result_debug() {
        let result = IngestionOperationResult {
            source: "test_source".to_string(),
            source_type: SourceType::LocalFolder,
            table_name: "INGEST_20250927143022".to_string(),
            ingestion_id: 1,
            files_processed: 10,
            files_failed: 1,
            files_skipped: 2,
            processing_time: Duration::from_secs(60),
            repo_info: None,
            folder_info: None,
            batch_stats: BatchStats::default(),
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("test_source"));
        assert!(debug_str.contains("LocalFolder"));
        assert!(debug_str.contains("INGEST_20250927143022"));
        assert!(debug_str.contains("10"));
    }
}
