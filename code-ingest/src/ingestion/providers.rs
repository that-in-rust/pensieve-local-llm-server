//! Ingestion provider implementations for different source types
//!
//! This module provides the trait-based ingestion system that allows
//! unified handling of different source types (git repositories, local folders)
//! through a common interface.

use crate::database::Database;
use crate::error::{IngestionError, IngestionResult};
use crate::ingestion::{
    git_cloner::{CloneConfig, GitCloner},
    folder_processor::{FolderConfig, FolderProcessor},
    batch_processor::{BatchConfig, BatchProcessor},
};
use crate::processing::FileProcessor;
use async_trait::async_trait;
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn, debug};
use url::Url;

/// Source types for ingestion operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestionSource {
    /// Git repository with URL
    GitRepository(String),
    /// Local folder with path and recursive flag
    LocalFolder { path: PathBuf, recursive: bool },
}

/// Validation error for ingestion sources
#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("Invalid Git repository URL: {url}")]
    InvalidGitUrl { url: String },
    
    #[error("Local path not found: {path}")]
    LocalPathNotFound { path: String },
    
    #[error("Permission denied: {path}")]
    PermissionDenied { path: String },
    
    #[error("Invalid source configuration: {message}")]
    InvalidConfiguration { message: String },
}

/// Common trait for all ingestion providers
#[async_trait]
pub trait IngestionProvider: Send + Sync {
    /// Ingest from a source and return the table name where data was stored
    async fn ingest(&self, source: IngestionSource, db_path: &Path) -> IngestionResult<String>;
    
    /// Validate that the source is accessible and properly configured
    async fn validate_source(&self, source: &IngestionSource) -> Result<(), ValidationError>;
    
    /// Get the provider name for logging and identification
    fn provider_name(&self) -> &'static str;
}

/// Git repository ingestion provider
pub struct GitIngestionProvider {
    clone_config: CloneConfig,
    folder_config: FolderConfig,
    batch_config: BatchConfig,
    file_processor: Arc<dyn FileProcessor>,
    cleanup_cloned_repos: bool,
}

impl GitIngestionProvider {
    /// Create a new GitIngestionProvider
    pub fn new(
        clone_config: CloneConfig,
        folder_config: FolderConfig,
        batch_config: BatchConfig,
        file_processor: Arc<dyn FileProcessor>,
        cleanup_cloned_repos: bool,
    ) -> Self {
        Self {
            clone_config,
            folder_config,
            batch_config,
            file_processor,
            cleanup_cloned_repos,
        }
    }

    /// Clone repository and extract file metadata
    async fn clone_and_extract_metadata(&self, repo_url: &str) -> IngestionResult<(PathBuf, Vec<FileMetadata>)> {
        info!("Cloning repository: {}", repo_url);
        
        // Clone the repository
        let git_cloner = GitCloner::new(self.clone_config.clone());
        let clone_result = git_cloner.clone_repository(repo_url).await?;
        
        info!(
            "Successfully cloned {} files from {}",
            clone_result.file_count, repo_url
        );

        // Process the cloned repository folder to extract file metadata
        let folder_processor = FolderProcessor::new(self.folder_config.clone());
        let folder_result = folder_processor.process_folder(&clone_result.repo_path)
            .map_err(|e| IngestionError::ProcessingError { 
                cause: format!("Failed to process cloned repository: {}", e) 
            })?;

        // Convert folder result to file metadata
        let file_metadata: Vec<FileMetadata> = folder_result
            .files
            .into_iter()
            .filter(|f| !f.skipped)
            .map(|f| FileMetadata {
                absolute_path: f.absolute_path,
                relative_path: f.relative_path,
                filename: f.filename,
                extension: f.extension,
                size_bytes: f.size_bytes,
                modified_time: f.modified_time,
            })
            .collect();

        Ok((clone_result.repo_path, file_metadata))
    }

    /// Read file content and create processed file entries
    async fn read_and_process_files(&self, file_metadata: Vec<FileMetadata>) -> IngestionResult<Vec<crate::processing::ProcessedFile>> {
        if file_metadata.is_empty() {
            return Ok(Vec::new());
        }

        let file_paths: Vec<PathBuf> = file_metadata
            .iter()
            .map(|f| f.absolute_path.clone())
            .collect();

        // Process files using the batch processor
        let batch_processor = BatchProcessor::new(
            self.batch_config.clone(),
            Arc::clone(&self.file_processor),
        );

        let (processed_files, _stats) = batch_processor
            .process_files(file_paths, None)
            .await
            .map_err(|e| IngestionError::ProcessingError { 
                cause: format!("Failed to process files: {}", e) 
            })?;

        Ok(processed_files)
    }

    /// Insert processed files into database
    async fn insert_into_database(
        &self,
        processed_files: Vec<crate::processing::ProcessedFile>,
        table_name: &str,
        ingestion_id: i64,
        database: &Database,
    ) -> IngestionResult<()> {
        if processed_files.is_empty() {
            info!("No files to insert into database");
            return Ok(());
        }

        // Create the ingestion table
        database.create_ingestion_table(table_name).await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to create table {}: {}", table_name, e) 
            })?;

        // Insert processed files
        database.insert_processed_files(table_name, &processed_files, ingestion_id).await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to insert files into {}: {}", table_name, e) 
            })?;

        info!(
            "Successfully inserted {} files into table {}",
            processed_files.len(),
            table_name
        );

        Ok(())
    }
}

#[async_trait]
impl IngestionProvider for GitIngestionProvider {
    async fn ingest(&self, source: IngestionSource, db_path: &Path) -> IngestionResult<String> {
        let repo_url = match source {
            IngestionSource::GitRepository(url) => url,
            _ => return Err(IngestionError::ConfigurationError {
                message: "GitIngestionProvider can only handle GitRepository sources".to_string(),
            }),
        };

        // Validate the source first
        self.validate_source(&IngestionSource::GitRepository(repo_url.clone())).await
            .map_err(|e| IngestionError::ConfigurationError { 
                message: format!("Source validation failed: {}", e) 
            })?;

        // Create database connection
        let db_url = format!("sqlite://{}", db_path.display());
        let database = Arc::new(Database::new(&db_url).await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to connect to database: {}", e) 
            })?);

        // Generate table name and create ingestion record
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let table_name = format!("INGEST_{}", Utc::now().format("%Y%m%d%H%M%S"));

        let ingestion_id = database
            .create_ingestion_record(
                Some(repo_url.clone()),
                repo_url.clone(),
                timestamp,
                &table_name,
            )
            .await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to create ingestion record: {}", e) 
            })?;

        // Clone repository and extract metadata
        let (repo_path, file_metadata) = self.clone_and_extract_metadata(&repo_url).await?;

        // Process files and read content
        let processed_files = self.read_and_process_files(file_metadata).await?;

        // Insert into database
        self.insert_into_database(processed_files.clone(), &table_name, ingestion_id, &database).await?;

        // Clean up cloned repository if configured
        if self.cleanup_cloned_repos {
            if let Err(e) = std::fs::remove_dir_all(&repo_path) {
                warn!(
                    "Failed to clean up cloned repository at {}: {}",
                    repo_path.display(),
                    e
                );
            } else {
                info!("Cleaned up cloned repository at {}", repo_path.display());
            }
        }

        // Update ingestion record with completion
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        database
            .complete_ingestion_record(
                ingestion_id,
                end_timestamp,
                processed_files.len() as i32,
            )
            .await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to complete ingestion record: {}", e) 
            })?;

        info!(
            "Git ingestion completed successfully: {} files ingested into table {}",
            processed_files.len(),
            table_name
        );

        Ok(table_name)
    }

    async fn validate_source(&self, source: &IngestionSource) -> Result<(), ValidationError> {
        match source {
            IngestionSource::GitRepository(url) => {
                // Validate URL format
                let parsed_url = Url::parse(url).map_err(|_| ValidationError::InvalidGitUrl {
                    url: url.clone(),
                })?;

                // Check if it's a supported Git hosting service
                match parsed_url.host_str() {
                    Some("github.com") | Some("gitlab.com") | Some("bitbucket.org") => {
                        debug!("Validated Git repository URL: {}", url);
                        Ok(())
                    }
                    Some(host) if host.contains("gitlab") || host.contains("git") => {
                        // Allow custom Git hosting services
                        debug!("Validated custom Git host: {}", host);
                        Ok(())
                    }
                    _ => Err(ValidationError::InvalidGitUrl {
                        url: url.clone(),
                    }),
                }
            }
            _ => Err(ValidationError::InvalidConfiguration {
                message: "GitIngestionProvider can only validate GitRepository sources".to_string(),
            }),
        }
    }

    fn provider_name(&self) -> &'static str {
        "GitIngestionProvider"
    }
}

/// Local folder ingestion provider
pub struct FolderIngestionProvider {
    folder_config: FolderConfig,
    batch_config: BatchConfig,
    file_processor: Arc<dyn FileProcessor>,
}

impl FolderIngestionProvider {
    /// Create a new FolderIngestionProvider
    pub fn new(
        folder_config: FolderConfig,
        batch_config: BatchConfig,
        file_processor: Arc<dyn FileProcessor>,
    ) -> Self {
        Self {
            folder_config,
            batch_config,
            file_processor,
        }
    }

    /// Process local folder and extract file metadata
    async fn process_folder_and_extract_metadata(&self, folder_path: &Path) -> IngestionResult<Vec<FileMetadata>> {
        info!("Processing local folder: {}", folder_path.display());

        // Process the folder to extract file metadata
        let folder_processor = FolderProcessor::new(self.folder_config.clone());
        let folder_result = folder_processor.process_folder(folder_path)
            .map_err(|e| IngestionError::ProcessingError { 
                cause: format!("Failed to process local folder: {}", e) 
            })?;

        // Convert folder result to file metadata
        let file_metadata: Vec<FileMetadata> = folder_result
            .files
            .into_iter()
            .filter(|f| !f.skipped)
            .map(|f| FileMetadata {
                absolute_path: f.absolute_path,
                relative_path: f.relative_path,
                filename: f.filename,
                extension: f.extension,
                size_bytes: f.size_bytes,
                modified_time: f.modified_time,
            })
            .collect();

        info!(
            "Discovered {} files in folder ({}  total)",
            file_metadata.len(), folder_result.total_files
        );

        Ok(file_metadata)
    }
}

#[async_trait]
impl IngestionProvider for FolderIngestionProvider {
    async fn ingest(&self, source: IngestionSource, db_path: &Path) -> IngestionResult<String> {
        let folder_path = match source {
            IngestionSource::LocalFolder { path, .. } => path,
            _ => return Err(IngestionError::ConfigurationError {
                message: "FolderIngestionProvider can only handle LocalFolder sources".to_string(),
            }),
        };

        // Validate the source first
        self.validate_source(&IngestionSource::LocalFolder { 
            path: folder_path.clone(), 
            recursive: true 
        }).await
            .map_err(|e| IngestionError::ConfigurationError { 
                message: format!("Source validation failed: {}", e) 
            })?;

        // Create database connection
        let db_url = format!("sqlite://{}", db_path.display());
        let database = Arc::new(Database::new(&db_url).await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to connect to database: {}", e) 
            })?);

        // Generate table name and create ingestion record
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let table_name = format!("INGEST_{}", Utc::now().format("%Y%m%d%H%M%S"));

        let ingestion_id = database
            .create_ingestion_record(
                None, // No repository URL for local folders
                folder_path.display().to_string(),
                timestamp,
                &table_name,
            )
            .await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to create ingestion record: {}", e) 
            })?;

        // Process folder and extract metadata
        let file_metadata = self.process_folder_and_extract_metadata(&folder_path).await?;

        // Process files and read content
        if file_metadata.is_empty() {
            warn!("No files to process in folder: {}", folder_path.display());
            
            // Complete ingestion record with 0 files
            let end_timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            database
                .complete_ingestion_record(ingestion_id, end_timestamp, 0)
                .await
                .map_err(|e| IngestionError::DatabaseError { 
                    cause: format!("Failed to complete ingestion record: {}", e) 
                })?;

            return Ok(table_name);
        }

        let file_paths: Vec<PathBuf> = file_metadata
            .iter()
            .map(|f| f.absolute_path.clone())
            .collect();

        // Process files using the batch processor
        let batch_processor = BatchProcessor::new(
            self.batch_config.clone(),
            Arc::clone(&self.file_processor),
        );

        let (processed_files, _stats) = batch_processor
            .process_files(file_paths, None)
            .await
            .map_err(|e| IngestionError::ProcessingError { 
                cause: format!("Failed to process files: {}", e) 
            })?;

        // Create the ingestion table
        database.create_ingestion_table(&table_name).await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to create table {}: {}", table_name, e) 
            })?;

        // Insert processed files into database
        if !processed_files.is_empty() {
            database.insert_processed_files(&table_name, &processed_files, ingestion_id).await
                .map_err(|e| IngestionError::DatabaseError { 
                    cause: format!("Failed to insert files into {}: {}", table_name, e) 
                })?;
        }

        // Update ingestion record with completion
        let end_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        database
            .complete_ingestion_record(
                ingestion_id,
                end_timestamp,
                processed_files.len() as i32,
            )
            .await
            .map_err(|e| IngestionError::DatabaseError { 
                cause: format!("Failed to complete ingestion record: {}", e) 
            })?;

        info!(
            "Folder ingestion completed successfully: {} files ingested into table {}",
            processed_files.len(),
            table_name
        );

        Ok(table_name)
    }

    async fn validate_source(&self, source: &IngestionSource) -> Result<(), ValidationError> {
        match source {
            IngestionSource::LocalFolder { path, .. } => {
                // Check if path exists
                if !path.exists() {
                    return Err(ValidationError::LocalPathNotFound {
                        path: path.display().to_string(),
                    });
                }

                // Check if it's a directory
                if !path.is_dir() {
                    return Err(ValidationError::LocalPathNotFound {
                        path: format!("{} is not a directory", path.display()),
                    });
                }

                // Test read access
                std::fs::read_dir(path).map_err(|_| ValidationError::PermissionDenied {
                    path: path.display().to_string(),
                })?;

                debug!("Validated local folder path: {}", path.display());
                Ok(())
            }
            _ => Err(ValidationError::InvalidConfiguration {
                message: "FolderIngestionProvider can only validate LocalFolder sources".to_string(),
            }),
        }
    }

    fn provider_name(&self) -> &'static str {
        "FolderIngestionProvider"
    }
}

/// Unified ingestion engine that coordinates git and folder providers
pub struct IngestionEngine {
    git_provider: GitIngestionProvider,
    folder_provider: FolderIngestionProvider,
}

impl IngestionEngine {
    /// Create a new IngestionEngine with the specified providers
    pub fn new(
        git_provider: GitIngestionProvider,
        folder_provider: FolderIngestionProvider,
    ) -> Self {
        Self {
            git_provider,
            folder_provider,
        }
    }

    /// Create a new IngestionEngine with default configuration
    pub fn with_default_config(file_processor: Arc<dyn FileProcessor>) -> Self {
        let git_provider = GitIngestionProvider::new(
            CloneConfig::default(),
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::clone(&file_processor),
            true, // cleanup cloned repos by default
        );

        let folder_provider = FolderIngestionProvider::new(
            FolderConfig::default(),
            BatchConfig::default(),
            file_processor,
        );

        Self::new(git_provider, folder_provider)
    }

    /// Ingest from any source type, automatically selecting the appropriate provider
    pub async fn ingest_source(&self, source: IngestionSource, db_path: &Path) -> IngestionResult<String> {
        info!("Starting ingestion from source: {:?}", source);

        // Validate source first
        self.validate_source(&source).await
            .map_err(|e| IngestionError::ConfigurationError { 
                message: format!("Source validation failed: {}", e) 
            })?;

        // Select appropriate provider and perform ingestion
        let table_name = match &source {
            IngestionSource::GitRepository(_) => {
                info!("Using GitIngestionProvider for repository ingestion");
                self.git_provider.ingest(source, db_path).await?
            }
            IngestionSource::LocalFolder { .. } => {
                info!("Using FolderIngestionProvider for local folder ingestion");
                self.folder_provider.ingest(source, db_path).await?
            }
        };

        info!("Ingestion completed successfully, data stored in table: {}", table_name);
        Ok(table_name)
    }

    /// Validate any source type using the appropriate provider
    pub async fn validate_source(&self, source: &IngestionSource) -> Result<(), ValidationError> {
        match source {
            IngestionSource::GitRepository(_) => {
                self.git_provider.validate_source(source).await
            }
            IngestionSource::LocalFolder { .. } => {
                self.folder_provider.validate_source(source).await
            }
        }
    }

    /// Get the appropriate provider for a source type
    pub fn get_provider(&self, source: &IngestionSource) -> &dyn IngestionProvider {
        match source {
            IngestionSource::GitRepository(_) => &self.git_provider,
            IngestionSource::LocalFolder { .. } => &self.folder_provider,
        }
    }

    /// Check if a string looks like a Git repository URL
    pub fn is_git_repository_url(source: &str) -> bool {
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

    /// Create an IngestionSource from a string, automatically detecting the type
    pub fn create_source_from_string(source: &str) -> Result<IngestionSource, ValidationError> {
        if Self::is_git_repository_url(source) {
            Ok(IngestionSource::GitRepository(source.to_string()))
        } else {
            // Assume it's a local path
            let path = PathBuf::from(source);
            
            // Validate that it exists and is a directory
            if !path.exists() {
                return Err(ValidationError::LocalPathNotFound {
                    path: source.to_string(),
                });
            }

            if !path.is_dir() {
                return Err(ValidationError::LocalPathNotFound {
                    path: format!("{} is not a directory", source),
                });
            }

            Ok(IngestionSource::LocalFolder {
                path,
                recursive: true,
            })
        }
    }

    /// Handle ingestion failures with recovery strategies
    pub async fn ingest_with_recovery(
        &self,
        source: IngestionSource,
        db_path: &Path,
        max_retries: u32,
    ) -> IngestionResult<String> {
        let mut last_error = None;
        
        for attempt in 0..=max_retries {
            if attempt > 0 {
                info!("Retry attempt {} for ingestion", attempt);
                
                // Wait before retry (exponential backoff)
                let delay = Duration::from_secs(2_u64.pow(attempt.min(5)));
                tokio::time::sleep(delay).await;
            }

            match self.ingest_source(source.clone(), db_path).await {
                Ok(table_name) => {
                    if attempt > 0 {
                        info!("Ingestion succeeded on retry attempt {}", attempt);
                    }
                    return Ok(table_name);
                }
                Err(e) => {
                    warn!("Ingestion attempt {} failed: {}", attempt + 1, e);
                    
                    // Check if this error is recoverable
                    let is_recoverable = match &e {
                        IngestionError::NetworkError { .. } => true,
                        IngestionError::DatabaseError { cause } if cause.contains("connection") => true,
                        IngestionError::GitCloneFailed { cause, .. } if cause.contains("timeout") => true,
                        _ => false,
                    };

                    if !is_recoverable {
                        return Err(e);
                    }

                    last_error = Some(e);
                }
            }
        }

        // All retries exhausted
        Err(last_error.unwrap_or_else(|| IngestionError::ConfigurationError {
            message: "All retry attempts failed".to_string(),
        }))
    }

    /// Get ingestion statistics and provider information
    pub fn get_engine_info(&self) -> IngestionEngineInfo {
        IngestionEngineInfo {
            git_provider_name: self.git_provider.provider_name().to_string(),
            folder_provider_name: self.folder_provider.provider_name().to_string(),
            supported_sources: vec![
                "Git repositories (https://github.com/user/repo)".to_string(),
                "Git repositories (https://gitlab.com/user/repo.git)".to_string(),
                "Git repositories (git@github.com:user/repo.git)".to_string(),
                "Local folders (/path/to/folder)".to_string(),
            ],
        }
    }
}

/// Information about the ingestion engine configuration
#[derive(Debug, Clone)]
pub struct IngestionEngineInfo {
    pub git_provider_name: String,
    pub folder_provider_name: String,
    pub supported_sources: Vec<String>,
}

/// File metadata extracted during folder processing
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// Absolute path to the file
    pub absolute_path: PathBuf,
    /// Path relative to the root directory
    pub relative_path: PathBuf,
    /// File name
    pub filename: String,
    /// File extension (without the dot)
    pub extension: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Last modified time
    pub modified_time: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::{FileType, ProcessedFile};
    use crate::error::ProcessingResult;
    use tempfile::TempDir;
    use std::fs;

    // Mock file processor for testing
    struct MockFileProcessor;

    #[async_trait::async_trait]
    impl FileProcessor for MockFileProcessor {
        fn can_process(&self, _file_path: &Path) -> bool {
            true
        }

        async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
            let content = std::fs::read_to_string(file_path)
                .unwrap_or_else(|_| "mock content".to_string());
            
            Ok(ProcessedFile {
                filepath: file_path.display().to_string(),
                filename: file_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                extension: file_path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_string(),
                file_size_bytes: content.len() as i64,
                line_count: Some(content.lines().count() as i32),
                word_count: Some(content.split_whitespace().count() as i32),
                token_count: None,
                content_text: Some(content),
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

    fn create_test_directory() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();

        // Create test files
        fs::write(root.join("file1.txt"), "Hello, world!").unwrap();
        fs::write(root.join("file2.rs"), "fn main() {}").unwrap();
        fs::write(root.join("README.md"), "# Test Repository").unwrap();

        // Create subdirectory with files
        let subdir = root.join("src");
        fs::create_dir(&subdir).unwrap();
        fs::write(subdir.join("lib.rs"), "pub mod test;").unwrap();

        temp_dir
    }

    #[test]
    fn test_ingestion_source_debug() {
        let git_source = IngestionSource::GitRepository("https://github.com/user/repo".to_string());
        let folder_source = IngestionSource::LocalFolder {
            path: PathBuf::from("/tmp/test"),
            recursive: true,
        };

        let git_debug = format!("{:?}", git_source);
        let folder_debug = format!("{:?}", folder_source);

        assert!(git_debug.contains("GitRepository"));
        assert!(git_debug.contains("github.com"));
        assert!(folder_debug.contains("LocalFolder"));
        assert!(folder_debug.contains("/tmp/test"));
    }

    #[tokio::test]
    async fn test_git_provider_validate_source() {
        let provider = GitIngestionProvider::new(
            CloneConfig::default(),
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
            true,
        );

        // Valid Git URLs
        let valid_sources = vec![
            IngestionSource::GitRepository("https://github.com/user/repo".to_string()),
            IngestionSource::GitRepository("https://gitlab.com/user/repo.git".to_string()),
            IngestionSource::GitRepository("https://bitbucket.org/user/repo".to_string()),
        ];

        for source in valid_sources {
            assert!(provider.validate_source(&source).await.is_ok());
        }

        // Invalid Git URLs
        let invalid_sources = vec![
            IngestionSource::GitRepository("not-a-url".to_string()),
            IngestionSource::GitRepository("https://example.com/repo".to_string()),
        ];

        for source in invalid_sources {
            assert!(provider.validate_source(&source).await.is_err());
        }

        // Wrong source type
        let folder_source = IngestionSource::LocalFolder {
            path: PathBuf::from("/tmp"),
            recursive: true,
        };
        assert!(provider.validate_source(&folder_source).await.is_err());
    }

    #[tokio::test]
    async fn test_folder_provider_validate_source() {
        let provider = FolderIngestionProvider::new(
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
        );

        let temp_dir = create_test_directory();

        // Valid folder source
        let valid_source = IngestionSource::LocalFolder {
            path: temp_dir.path().to_path_buf(),
            recursive: true,
        };
        assert!(provider.validate_source(&valid_source).await.is_ok());

        // Non-existent folder
        let invalid_source = IngestionSource::LocalFolder {
            path: PathBuf::from("/nonexistent/path"),
            recursive: true,
        };
        assert!(provider.validate_source(&invalid_source).await.is_err());

        // Wrong source type
        let git_source = IngestionSource::GitRepository("https://github.com/user/repo".to_string());
        assert!(provider.validate_source(&git_source).await.is_err());
    }

    #[tokio::test]
    async fn test_folder_provider_process_folder_metadata() {
        let provider = FolderIngestionProvider::new(
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
        );

        let temp_dir = create_test_directory();
        let metadata = provider.process_folder_and_extract_metadata(temp_dir.path()).await.unwrap();

        assert!(!metadata.is_empty());
        
        // Check that we have expected files
        let filenames: std::collections::HashSet<String> = metadata
            .iter()
            .map(|f| f.filename.clone())
            .collect();
        
        assert!(filenames.contains("file1.txt"));
        assert!(filenames.contains("file2.rs"));
        assert!(filenames.contains("README.md"));
        assert!(filenames.contains("lib.rs"));

        // Check metadata fields
        for file_meta in &metadata {
            assert!(!file_meta.filename.is_empty());
            assert!(file_meta.absolute_path.exists());
            assert!(file_meta.size_bytes > 0);
        }
    }

    #[test]
    fn test_provider_names() {
        let git_provider = GitIngestionProvider::new(
            CloneConfig::default(),
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
            true,
        );

        let folder_provider = FolderIngestionProvider::new(
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
        );

        assert_eq!(git_provider.provider_name(), "GitIngestionProvider");
        assert_eq!(folder_provider.provider_name(), "FolderIngestionProvider");
    }

    #[test]
    fn test_validation_error_display() {
        let errors = vec![
            ValidationError::InvalidGitUrl { url: "invalid".to_string() },
            ValidationError::LocalPathNotFound { path: "/nonexistent".to_string() },
            ValidationError::PermissionDenied { path: "/restricted".to_string() },
            ValidationError::InvalidConfiguration { message: "test error".to_string() },
        ];

        for error in errors {
            let error_str = error.to_string();
            assert!(!error_str.is_empty());
        }
    }

    #[test]
    fn test_file_metadata_debug() {
        let metadata = FileMetadata {
            absolute_path: PathBuf::from("/tmp/test.txt"),
            relative_path: PathBuf::from("test.txt"),
            filename: "test.txt".to_string(),
            extension: "txt".to_string(),
            size_bytes: 1024,
            modified_time: SystemTime::UNIX_EPOCH,
        };

        let debug_str = format!("{:?}", metadata);
        assert!(debug_str.contains("test.txt"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn test_ingestion_engine_creation() {
        let git_provider = GitIngestionProvider::new(
            CloneConfig::default(),
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
            true,
        );

        let folder_provider = FolderIngestionProvider::new(
            FolderConfig::default(),
            BatchConfig::default(),
            Arc::new(MockFileProcessor),
        );

        let engine = IngestionEngine::new(git_provider, folder_provider);
        let info = engine.get_engine_info();

        assert_eq!(info.git_provider_name, "GitIngestionProvider");
        assert_eq!(info.folder_provider_name, "FolderIngestionProvider");
        assert!(!info.supported_sources.is_empty());
    }

    #[test]
    fn test_ingestion_engine_with_default_config() {
        let engine = IngestionEngine::with_default_config(Arc::new(MockFileProcessor));
        let info = engine.get_engine_info();

        assert_eq!(info.git_provider_name, "GitIngestionProvider");
        assert_eq!(info.folder_provider_name, "FolderIngestionProvider");
    }

    #[test]
    fn test_is_git_repository_url() {
        // Valid Git URLs
        assert!(IngestionEngine::is_git_repository_url("https://github.com/user/repo"));
        assert!(IngestionEngine::is_git_repository_url("https://gitlab.com/user/repo.git"));
        assert!(IngestionEngine::is_git_repository_url("https://bitbucket.org/user/repo"));
        assert!(IngestionEngine::is_git_repository_url("git@github.com:user/repo.git"));
        assert!(IngestionEngine::is_git_repository_url("https://git.example.com/repo"));

        // Invalid Git URLs
        assert!(!IngestionEngine::is_git_repository_url("/local/path"));
        assert!(!IngestionEngine::is_git_repository_url("not-a-url"));
        assert!(!IngestionEngine::is_git_repository_url("https://example.com/page"));
    }

    #[test]
    fn test_create_source_from_string() {
        // Git repository URL
        let git_source = IngestionEngine::create_source_from_string("https://github.com/user/repo").unwrap();
        match git_source {
            IngestionSource::GitRepository(url) => {
                assert_eq!(url, "https://github.com/user/repo");
            }
            _ => panic!("Expected GitRepository source"),
        }

        // Local folder (using temp directory)
        let temp_dir = create_test_directory();
        let folder_source = IngestionEngine::create_source_from_string(
            temp_dir.path().to_str().unwrap()
        ).unwrap();
        
        match folder_source {
            IngestionSource::LocalFolder { path, recursive } => {
                assert_eq!(path, temp_dir.path());
                assert!(recursive);
            }
            _ => panic!("Expected LocalFolder source"),
        }

        // Non-existent path
        let invalid_result = IngestionEngine::create_source_from_string("/nonexistent/path");
        assert!(invalid_result.is_err());
    }

    #[tokio::test]
    async fn test_ingestion_engine_validate_source() {
        let engine = IngestionEngine::with_default_config(Arc::new(MockFileProcessor));

        // Valid Git repository
        let git_source = IngestionSource::GitRepository("https://github.com/user/repo".to_string());
        assert!(engine.validate_source(&git_source).await.is_ok());

        // Valid local folder
        let temp_dir = create_test_directory();
        let folder_source = IngestionSource::LocalFolder {
            path: temp_dir.path().to_path_buf(),
            recursive: true,
        };
        assert!(engine.validate_source(&folder_source).await.is_ok());

        // Invalid Git repository
        let invalid_git = IngestionSource::GitRepository("not-a-url".to_string());
        assert!(engine.validate_source(&invalid_git).await.is_err());

        // Invalid local folder
        let invalid_folder = IngestionSource::LocalFolder {
            path: PathBuf::from("/nonexistent"),
            recursive: true,
        };
        assert!(engine.validate_source(&invalid_folder).await.is_err());
    }

    #[test]
    fn test_ingestion_engine_get_provider() {
        let engine = IngestionEngine::with_default_config(Arc::new(MockFileProcessor));

        let git_source = IngestionSource::GitRepository("https://github.com/user/repo".to_string());
        let git_provider = engine.get_provider(&git_source);
        assert_eq!(git_provider.provider_name(), "GitIngestionProvider");

        let temp_dir = create_test_directory();
        let folder_source = IngestionSource::LocalFolder {
            path: temp_dir.path().to_path_buf(),
            recursive: true,
        };
        let folder_provider = engine.get_provider(&folder_source);
        assert_eq!(folder_provider.provider_name(), "FolderIngestionProvider");
    }

    #[test]
    fn test_ingestion_engine_info_debug() {
        let info = IngestionEngineInfo {
            git_provider_name: "GitIngestionProvider".to_string(),
            folder_provider_name: "FolderIngestionProvider".to_string(),
            supported_sources: vec!["test".to_string()],
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("GitIngestionProvider"));
        assert!(debug_str.contains("FolderIngestionProvider"));
    }
}