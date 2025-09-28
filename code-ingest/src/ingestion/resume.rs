//! Resume capability and error recovery for ingestion operations
//! 
//! This module provides functionality to resume interrupted ingestions,
//! handle partial ingestion cleanup, and implement comprehensive error
//! recovery mechanisms with retry logic for transient failures.

use crate::database::Database;
use crate::error::{IngestionError, IngestionResult};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;
use tracing::{debug, info, warn};

/// Resume state for an interrupted ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeState {
    /// Ingestion ID from the database
    pub ingestion_id: i64,
    /// Table name for this ingestion
    pub table_name: String,
    /// Source path or URL
    pub source: String,
    /// Files that have been successfully processed
    pub completed_files: HashSet<PathBuf>,
    /// Files that failed processing (with error details)
    pub failed_files: Vec<FailedFile>,
    /// Total files discovered for processing
    pub total_files: usize,
    /// Timestamp when the ingestion was started
    pub start_timestamp: u64,
    /// Timestamp of the last successful file processing
    pub last_progress_timestamp: u64,
    /// Configuration used for the ingestion
    pub ingestion_config: IngestionResumeConfig,
}

/// Information about a failed file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedFile {
    /// Path to the file that failed
    pub file_path: PathBuf,
    /// Error message
    pub error_message: String,
    /// Number of retry attempts made
    pub retry_count: usize,
    /// Timestamp of the last failure
    pub last_failure_timestamp: u64,
    /// Whether this failure is considered permanent
    pub permanent_failure: bool,
}

/// Configuration for resume and retry behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionResumeConfig {
    /// Maximum number of retry attempts for transient failures
    pub max_retry_attempts: usize,
    /// Base delay between retry attempts
    pub retry_base_delay: Duration,
    /// Maximum delay between retry attempts (for exponential backoff)
    pub retry_max_delay: Duration,
    /// Whether to use exponential backoff for retries
    pub exponential_backoff: bool,
    /// Timeout for individual file processing
    pub file_processing_timeout: Duration,
    /// Whether to save resume state periodically during processing
    pub save_progress_periodically: bool,
    /// Interval for saving progress
    pub progress_save_interval: Duration,
    /// Directory to store resume state files
    pub resume_state_dir: PathBuf,
}

impl Default for IngestionResumeConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            retry_base_delay: Duration::from_secs(1),
            retry_max_delay: Duration::from_secs(60),
            exponential_backoff: true,
            file_processing_timeout: Duration::from_secs(300), // 5 minutes
            save_progress_periodically: true,
            progress_save_interval: Duration::from_secs(30),
            resume_state_dir: PathBuf::from(".code-ingest-resume"),
        }
    }
}

/// Resume manager for handling ingestion interruptions and recovery
pub struct ResumeManager {
    config: IngestionResumeConfig,
    database: Arc<Database>,
}

impl ResumeManager {
    /// Create a new resume manager
    pub fn new(config: IngestionResumeConfig, database: Arc<Database>) -> Self {
        Self { config, database }
    }

    /// Create a new resume state for an ingestion
    pub async fn create_resume_state(
        &self,
        ingestion_id: i64,
        table_name: String,
        source: String,
        total_files: usize,
    ) -> IngestionResult<ResumeState> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let resume_state = ResumeState {
            ingestion_id,
            table_name,
            source,
            completed_files: HashSet::new(),
            failed_files: Vec::new(),
            total_files,
            start_timestamp: now,
            last_progress_timestamp: now,
            ingestion_config: self.config.clone(),
        };

        // Save initial state
        self.save_resume_state(&resume_state).await?;

        info!(
            "Created resume state for ingestion {} with {} total files",
            ingestion_id, total_files
        );

        Ok(resume_state)
    }

    /// Load existing resume state for an ingestion
    pub async fn load_resume_state(&self, ingestion_id: i64) -> IngestionResult<Option<ResumeState>> {
        let state_file = self.get_resume_state_path(ingestion_id);
        
        if !state_file.exists() {
            return Ok(None);
        }

        match fs::read_to_string(&state_file).await {
            Ok(content) => {
                match serde_json::from_str::<ResumeState>(&content) {
                    Ok(state) => {
                        info!(
                            "Loaded resume state for ingestion {}: {}/{} files completed",
                            ingestion_id,
                            state.completed_files.len(),
                            state.total_files
                        );
                        Ok(Some(state))
                    }
                    Err(e) => {
                        warn!(
                            "Failed to parse resume state file {}: {}",
                            state_file.display(),
                            e
                        );
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                warn!(
                    "Failed to read resume state file {}: {}",
                    state_file.display(),
                    e
                );
                Ok(None)
            }
        }
    }

    /// Save resume state to disk
    pub async fn save_resume_state(&self, state: &ResumeState) -> IngestionResult<()> {
        // Ensure resume state directory exists
        if let Err(e) = fs::create_dir_all(&self.config.resume_state_dir).await {
            return Err(IngestionError::FileSystemError {
                path: self.config.resume_state_dir.display().to_string(),
                cause: format!("Failed to create resume state directory: {}", e),
            });
        }

        let state_file = self.get_resume_state_path(state.ingestion_id);
        let content = serde_json::to_string_pretty(state).map_err(|e| {
            IngestionError::FileSystemError {
                path: state_file.display().to_string(),
                cause: format!("Failed to serialize resume state: {}", e),
            }
        })?;

        fs::write(&state_file, content).await.map_err(|e| {
            IngestionError::FileSystemError {
                path: state_file.display().to_string(),
                cause: format!("Failed to write resume state: {}", e),
            }
        })?;

        debug!("Saved resume state for ingestion {}", state.ingestion_id);
        Ok(())
    }

    /// Update resume state with a successfully processed file
    pub async fn mark_file_completed(
        &self,
        state: &mut ResumeState,
        file_path: PathBuf,
    ) -> IngestionResult<()> {
        state.completed_files.insert(file_path.clone());
        state.last_progress_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Remove from failed files if it was there
        state.failed_files.retain(|f| f.file_path != file_path);

        debug!("Marked file as completed: {}", file_path.display());

        // Save progress periodically
        if self.config.save_progress_periodically {
            self.save_resume_state(state).await?;
        }

        Ok(())
    }

    /// Update resume state with a failed file
    pub async fn mark_file_failed(
        &self,
        state: &mut ResumeState,
        file_path: PathBuf,
        error_message: String,
        is_permanent: bool,
    ) -> IngestionResult<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Find existing failed file entry or create new one
        if let Some(failed_file) = state.failed_files.iter_mut().find(|f| f.file_path == file_path) {
            failed_file.retry_count += 1;
            failed_file.error_message = error_message;
            failed_file.last_failure_timestamp = now;
            failed_file.permanent_failure = is_permanent || failed_file.retry_count >= self.config.max_retry_attempts;
        } else {
            state.failed_files.push(FailedFile {
                file_path: file_path.clone(),
                error_message,
                retry_count: 1,
                last_failure_timestamp: now,
                permanent_failure: is_permanent,
            });
        }

        warn!("Marked file as failed: {}", file_path.display());

        // Save progress periodically
        if self.config.save_progress_periodically {
            self.save_resume_state(state).await?;
        }

        Ok(())
    }

    /// Get files that need to be processed (not completed and not permanently failed)
    pub fn get_remaining_files(
        &self,
        state: &ResumeState,
        all_files: &[PathBuf],
    ) -> Vec<PathBuf> {
        let permanent_failures: HashSet<_> = state
            .failed_files
            .iter()
            .filter(|f| f.permanent_failure)
            .map(|f| &f.file_path)
            .collect();

        all_files
            .iter()
            .filter(|path| {
                !state.completed_files.contains(*path) && !permanent_failures.contains(path)
            })
            .cloned()
            .collect()
    }

    /// Get files that can be retried (failed but not permanently)
    pub fn get_retryable_files(&self, state: &ResumeState) -> Vec<PathBuf> {
        state
            .failed_files
            .iter()
            .filter(|f| !f.permanent_failure && f.retry_count < self.config.max_retry_attempts)
            .map(|f| f.file_path.clone())
            .collect()
    }

    /// Calculate retry delay for a failed file
    pub fn calculate_retry_delay(&self, retry_count: usize) -> Duration {
        if !self.config.exponential_backoff {
            return self.config.retry_base_delay;
        }

        let delay_secs = self.config.retry_base_delay.as_secs() * (2_u64.pow(retry_count as u32));
        let delay = Duration::from_secs(delay_secs.min(self.config.retry_max_delay.as_secs()));
        
        debug!("Calculated retry delay for attempt {}: {:?}", retry_count, delay);
        delay
    }

    /// Check if an error is considered transient (retryable)
    pub fn is_transient_error(&self, error: &IngestionError) -> bool {
        match error {
            IngestionError::NetworkError { .. } => true,
            IngestionError::DatabaseError { .. } => true,
            IngestionError::FileSystemError { .. } => false, // Usually permanent
            IngestionError::GitError { .. } => true,
            IngestionError::ProcessingError { .. } => false, // Usually permanent
            IngestionError::ConfigurationError { .. } => false, // Permanent
            IngestionError::AuthenticationError { .. } => false, // Usually permanent
            IngestionError::GitCloneFailed { .. } => true, // Usually transient
            IngestionError::LocalPathNotFound { .. } => false, // Permanent
            IngestionError::PermissionDenied { .. } => false, // Usually permanent
            IngestionError::RepositoryTooLarge { .. } => false, // Permanent
            IngestionError::InvalidRepositoryUrl { .. } => false, // Permanent
            IngestionError::AuthenticationFailed { .. } => false, // Usually permanent
        }
    }

    /// Clean up resume state after successful completion
    pub async fn cleanup_resume_state(&self, ingestion_id: i64) -> IngestionResult<()> {
        let state_file = self.get_resume_state_path(ingestion_id);
        
        if state_file.exists() {
            if let Err(e) = fs::remove_file(&state_file).await {
                warn!(
                    "Failed to remove resume state file {}: {}",
                    state_file.display(),
                    e
                );
            } else {
                info!("Cleaned up resume state for ingestion {}", ingestion_id);
            }
        }

        Ok(())
    }

    /// Clean up old resume state files
    pub async fn cleanup_old_resume_states(&self, max_age: Duration) -> IngestionResult<()> {
        let cutoff_time = SystemTime::now() - max_age;
        
        if !self.config.resume_state_dir.exists() {
            return Ok(());
        }

        let mut dir_entries = fs::read_dir(&self.config.resume_state_dir).await.map_err(|e| {
            IngestionError::FileSystemError {
                path: self.config.resume_state_dir.display().to_string(),
                cause: format!("Failed to read resume state directory: {}", e),
            }
        })?;

        let mut cleaned_count = 0;

        while let Some(entry) = dir_entries.next_entry().await.map_err(|e| {
            IngestionError::FileSystemError {
                path: self.config.resume_state_dir.display().to_string(),
                cause: format!("Failed to read directory entry: {}", e),
            }
        })? {
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(metadata) = entry.metadata().await {
                    if let Ok(modified) = metadata.modified() {
                        if modified < cutoff_time {
                            if let Err(e) = fs::remove_file(&path).await {
                                warn!("Failed to remove old resume state file {}: {}", path.display(), e);
                            } else {
                                cleaned_count += 1;
                                debug!("Removed old resume state file: {}", path.display());
                            }
                        }
                    }
                }
            }
        }

        if cleaned_count > 0 {
            info!("Cleaned up {} old resume state files", cleaned_count);
        }

        Ok(())
    }

    /// Get the file path for storing resume state
    fn get_resume_state_path(&self, ingestion_id: i64) -> PathBuf {
        self.config
            .resume_state_dir
            .join(format!("ingestion_{}.json", ingestion_id))
    }

    /// Get progress statistics for a resume state
    pub fn get_progress_stats(&self, state: &ResumeState) -> ProgressStats {
        let completed = state.completed_files.len();
        let failed_permanent = state.failed_files.iter().filter(|f| f.permanent_failure).count();
        let failed_retryable = state.failed_files.iter().filter(|f| !f.permanent_failure).count();
        let remaining = state.total_files.saturating_sub(completed + failed_permanent);

        ProgressStats {
            total_files: state.total_files,
            completed_files: completed,
            failed_permanent_files: failed_permanent,
            failed_retryable_files: failed_retryable,
            remaining_files: remaining,
            completion_percentage: if state.total_files > 0 {
                (completed as f64 / state.total_files as f64) * 100.0
            } else {
                0.0
            },
        }
    }
}

/// Progress statistics for resume operations
#[derive(Debug, Clone)]
pub struct ProgressStats {
    pub total_files: usize,
    pub completed_files: usize,
    pub failed_permanent_files: usize,
    pub failed_retryable_files: usize,
    pub remaining_files: usize,
    pub completion_percentage: f64,
}

impl ProgressStats {
    /// Check if the ingestion is complete
    pub fn is_complete(&self) -> bool {
        self.remaining_files == 0 && self.failed_retryable_files == 0
    }

    /// Get a human-readable status string
    pub fn status_string(&self) -> String {
        format!(
            "{}/{} files completed ({:.1}%), {} failed permanently, {} retryable",
            self.completed_files,
            self.total_files,
            self.completion_percentage,
            self.failed_permanent_files,
            self.failed_retryable_files
        )
    }
}