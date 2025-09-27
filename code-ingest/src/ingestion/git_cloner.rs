use crate::error::{IngestionError, IngestionResult};
use git2::{
    build::RepoBuilder, Cred, CredentialType, FetchOptions, Progress, RemoteCallbacks, Repository,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::task;
use tracing::{debug, info, warn};
use url::Url;

/// Configuration for Git repository cloning
#[derive(Debug, Clone)]
pub struct CloneConfig {
    /// Target directory for cloning
    pub target_dir: PathBuf,
    /// Optional branch, tag, or commit hash to checkout
    pub reference: Option<String>,
    /// Authentication credentials (username, token/password)
    pub credentials: Option<(String, String)>,
    /// Timeout for clone operations
    pub timeout: Duration,
    /// Whether to show progress
    pub show_progress: bool,
    /// Maximum repository size in MB (0 = no limit)
    pub max_size_mb: u64,
}

impl Default for CloneConfig {
    fn default() -> Self {
        Self {
            target_dir: PathBuf::from("."),
            reference: None,
            credentials: None,
            timeout: Duration::from_secs(300), // 5 minutes
            show_progress: true,
            max_size_mb: 1000, // 1GB default limit
        }
    }
}

/// Git repository cloner with progress tracking and authentication support
pub struct GitCloner {
    config: CloneConfig,
}

/// Result of a successful clone operation
#[derive(Debug, Clone)]
pub struct CloneResult {
    /// Path to the cloned repository
    pub repo_path: PathBuf,
    /// Repository URL that was cloned
    pub repo_url: String,
    /// Commit hash that was checked out
    pub commit_hash: String,
    /// Branch or reference that was checked out
    pub reference: Option<String>,
    /// Total size of cloned repository in bytes
    pub size_bytes: u64,
    /// Number of files in the repository
    pub file_count: usize,
}

impl GitCloner {
    /// Create a new GitCloner with the specified configuration
    pub fn new(config: CloneConfig) -> Self {
        Self { config }
    }

    /// Clone a repository from the given URL
    pub async fn clone_repository(&self, repo_url: &str) -> IngestionResult<CloneResult> {
        // Validate the repository URL
        let parsed_url = self.validate_repository_url(repo_url)?;
        
        // Create target directory if it doesn't exist
        let target_path = self.prepare_target_directory(&parsed_url)?;
        
        info!("Starting clone of {} to {}", repo_url, target_path.display());
        
        // Perform the clone operation in a blocking task
        let config = self.config.clone();
        let repo_url = repo_url.to_string();
        let target_path_clone = target_path.clone();
        
        let clone_result = task::spawn_blocking(move || {
            Self::perform_clone(&config, &repo_url, &target_path_clone)
        })
        .await
        .map_err(|e| IngestionError::GitCloneFailed {
            repo_url: repo_url.clone(),
            cause: format!("Task join error: {}", e),
        })??;
        
        // Validate the cloned repository
        self.validate_cloned_repository(&clone_result)?;
        
        info!("Successfully cloned {} files from {}", 
              clone_result.file_count, repo_url);
        
        Ok(clone_result)
    }

    /// Validate that the repository URL is well-formed and supported
    fn validate_repository_url(&self, repo_url: &str) -> IngestionResult<Url> {
        let parsed_url = Url::parse(repo_url).map_err(|e| IngestionError::InvalidRepositoryUrl {
            url: repo_url.to_string(),
        })?;

        // Check if it's a supported Git hosting service
        match parsed_url.host_str() {
            Some("github.com") | Some("gitlab.com") | Some("bitbucket.org") => {
                debug!("Validated repository URL: {}", repo_url);
                Ok(parsed_url)
            }
            Some(host) if host.contains("gitlab") || host.contains("git") => {
                // Allow custom Git hosting services
                debug!("Validated custom Git host: {}", host);
                Ok(parsed_url)
            }
            _ => Err(IngestionError::InvalidRepositoryUrl {
                url: repo_url.to_string(),
            }),
        }
    }

    /// Prepare the target directory for cloning
    fn prepare_target_directory(&self, parsed_url: &Url) -> IngestionResult<PathBuf> {
        // Extract repository name from URL
        let repo_name = parsed_url
            .path_segments()
            .and_then(|segments| segments.last())
            .map(|name| name.trim_end_matches(".git"))
            .ok_or_else(|| IngestionError::InvalidRepositoryUrl {
                url: parsed_url.to_string(),
            })?;

        let target_path = self.config.target_dir.join(repo_name);

        // Remove existing directory if it exists
        if target_path.exists() {
            std::fs::remove_dir_all(&target_path).map_err(|e| {
                IngestionError::PermissionDenied {
                    path: target_path.display().to_string(),
                }
            })?;
        }

        // Create parent directories
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| IngestionError::PermissionDenied {
                path: parent.display().to_string(),
            })?;
        }

        Ok(target_path)
    }

    /// Perform the actual clone operation (blocking)
    fn perform_clone(
        config: &CloneConfig,
        repo_url: &str,
        target_path: &Path,
    ) -> IngestionResult<CloneResult> {
        let progress_bar = if config.show_progress {
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message("Cloning repository...");
            Some(pb)
        } else {
            None
        };

        // Setup progress callback
        let progress_bar_ref = progress_bar.as_ref().map(|pb| Arc::new(RefCell::new(pb)));
        let mut callbacks = RemoteCallbacks::new();
        
        if let Some(pb_ref) = progress_bar_ref {
            callbacks.transfer_progress(move |progress: Progress| {
                if let Ok(pb) = pb_ref.try_borrow() {
                    let received = progress.received_objects();
                    let total = progress.total_objects();
                    if total > 0 {
                        pb.set_length(total as u64);
                        pb.set_position(received as u64);
                        pb.set_message(format!("Receiving objects: {}/{}", received, total));
                    }
                }
                true
            });
        }

        // Setup authentication if provided
        if let Some((username, password)) = &config.credentials {
            let username = username.clone();
            let password = password.clone();
            callbacks.credentials(move |_url, username_from_url, _allowed_types| {
                Cred::userpass_plaintext(&username, &password)
            });
        } else {
            // Try default Git credential helpers
            callbacks.credentials(|_url, username_from_url, allowed_types| {
                if allowed_types.contains(CredentialType::SSH_KEY) {
                    Cred::ssh_key_from_agent(username_from_url.unwrap_or("git"))
                } else if allowed_types.contains(CredentialType::USER_PASS_PLAINTEXT) {
                    Cred::credential_helper(_url, username_from_url)
                } else {
                    Cred::default()
                }
            });
        }

        let mut fetch_options = FetchOptions::new();
        fetch_options.remote_callbacks(callbacks);

        // Build and execute the clone
        let mut builder = RepoBuilder::new();
        builder.fetch_options(fetch_options);

        if let Some(reference) = &config.reference {
            builder.branch(reference);
        }

        let repository = builder.clone(repo_url, target_path).map_err(|e| {
            if let Some(pb) = &progress_bar {
                pb.finish_with_message("Clone failed");
            }
            IngestionError::GitCloneFailed {
                repo_url: repo_url.to_string(),
                cause: e.message().to_string(),
            }
        })?;

        if let Some(pb) = &progress_bar {
            pb.finish_with_message("Clone completed");
        }

        // Get commit information
        let head = repository.head().map_err(|e| IngestionError::GitCloneFailed {
            repo_url: repo_url.to_string(),
            cause: format!("Failed to get HEAD: {}", e.message()),
        })?;

        let commit_hash = head
            .target()
            .ok_or_else(|| IngestionError::GitCloneFailed {
                repo_url: repo_url.to_string(),
                cause: "No commit found at HEAD".to_string(),
            })?
            .to_string();

        let reference = head.shorthand().map(|s| s.to_string());

        // Calculate repository size and file count
        let (size_bytes, file_count) = Self::calculate_repo_stats(target_path)?;

        // Check size limit
        let size_mb = size_bytes / (1024 * 1024);
        if config.max_size_mb > 0 && size_mb > config.max_size_mb {
            // Clean up the cloned repository
            let _ = std::fs::remove_dir_all(target_path);
            return Err(IngestionError::RepositoryTooLarge { size_mb });
        }

        Ok(CloneResult {
            repo_path: target_path.to_path_buf(),
            repo_url: repo_url.to_string(),
            commit_hash,
            reference,
            size_bytes,
            file_count,
        })
    }

    /// Calculate repository statistics (size and file count)
    fn calculate_repo_stats(repo_path: &Path) -> IngestionResult<(u64, usize)> {
        let mut total_size = 0u64;
        let mut file_count = 0usize;

        fn visit_dir(dir: &Path, total_size: &mut u64, file_count: &mut usize) -> std::io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                // Skip .git directory for size calculation
                if path.file_name().and_then(|n| n.to_str()) == Some(".git") {
                    continue;
                }

                if path.is_dir() {
                    visit_dir(&path, total_size, file_count)?;
                } else {
                    *total_size += entry.metadata()?.len();
                    *file_count += 1;
                }
            }
            Ok(())
        }

        visit_dir(repo_path, &mut total_size, &mut file_count).map_err(|e| {
            IngestionError::GitCloneFailed {
                repo_url: "unknown".to_string(),
                cause: format!("Failed to calculate repository stats: {}", e),
            }
        })?;

        Ok((total_size, file_count))
    }

    /// Validate the cloned repository
    fn validate_cloned_repository(&self, result: &CloneResult) -> IngestionResult<()> {
        // Check that the repository directory exists
        if !result.repo_path.exists() {
            return Err(IngestionError::GitCloneFailed {
                repo_url: result.repo_url.clone(),
                cause: "Cloned repository directory does not exist".to_string(),
            });
        }

        // Check that it's a valid Git repository
        Repository::open(&result.repo_path).map_err(|e| IngestionError::GitCloneFailed {
            repo_url: result.repo_url.clone(),
            cause: format!("Invalid Git repository: {}", e.message()),
        })?;

        // Check that we have at least some files
        if result.file_count == 0 {
            warn!("Cloned repository contains no files: {}", result.repo_url);
        }

        debug!(
            "Validated cloned repository: {} files, {} bytes",
            result.file_count, result.size_bytes
        );

        Ok(())
    }

    /// Check if a path contains a Git repository
    pub fn is_git_repository(path: &Path) -> bool {
        Repository::open(path).is_ok()
    }

    /// Get information about an existing Git repository
    pub fn get_repository_info(path: &Path) -> IngestionResult<CloneResult> {
        let repository = Repository::open(path).map_err(|e| IngestionError::GitCloneFailed {
            repo_url: path.display().to_string(),
            cause: format!("Failed to open repository: {}", e.message()),
        })?;

        // Get remote URL
        let remote = repository
            .find_remote("origin")
            .or_else(|_| repository.remotes().and_then(|remotes| {
                remotes.get(0).and_then(|name| repository.find_remote(name))
            }))
            .map_err(|e| IngestionError::GitCloneFailed {
                repo_url: path.display().to_string(),
                cause: format!("No remote found: {}", e.message()),
            })?;

        let repo_url = remote.url().unwrap_or("unknown").to_string();

        // Get commit information
        let head = repository.head().map_err(|e| IngestionError::GitCloneFailed {
            repo_url: repo_url.clone(),
            cause: format!("Failed to get HEAD: {}", e.message()),
        })?;

        let commit_hash = head
            .target()
            .ok_or_else(|| IngestionError::GitCloneFailed {
                repo_url: repo_url.clone(),
                cause: "No commit found at HEAD".to_string(),
            })?
            .to_string();

        let reference = head.shorthand().map(|s| s.to_string());

        // Calculate repository size and file count
        let (size_bytes, file_count) = Self::calculate_repo_stats(path)?;

        Ok(CloneResult {
            repo_path: path.to_path_buf(),
            repo_url,
            commit_hash,
            reference,
            size_bytes,
            file_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio_test;

    #[test]
    fn test_clone_config_default() {
        let config = CloneConfig::default();
        assert_eq!(config.target_dir, PathBuf::from("."));
        assert_eq!(config.reference, None);
        assert_eq!(config.credentials, None);
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert!(config.show_progress);
        assert_eq!(config.max_size_mb, 1000);
    }

    #[test]
    fn test_validate_repository_url() {
        let config = CloneConfig::default();
        let cloner = GitCloner::new(config);

        // Valid URLs
        assert!(cloner.validate_repository_url("https://github.com/user/repo").is_ok());
        assert!(cloner.validate_repository_url("https://gitlab.com/user/repo.git").is_ok());
        assert!(cloner.validate_repository_url("https://bitbucket.org/user/repo").is_ok());
        assert!(cloner.validate_repository_url("https://git.example.com/user/repo").is_ok());

        // Invalid URLs
        assert!(cloner.validate_repository_url("not-a-url").is_err());
        assert!(cloner.validate_repository_url("https://example.com/repo").is_err());
        assert!(cloner.validate_repository_url("").is_err());
    }

    #[test]
    fn test_prepare_target_directory() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloneConfig {
            target_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        let cloner = GitCloner::new(config);

        let url = Url::parse("https://github.com/user/test-repo.git").unwrap();
        let target_path = cloner.prepare_target_directory(&url).unwrap();

        assert_eq!(target_path.file_name().unwrap(), "test-repo");
        assert!(target_path.parent().unwrap().exists());
    }

    #[test]
    fn test_calculate_repo_stats() {
        let temp_dir = TempDir::new().unwrap();
        let repo_path = temp_dir.path();

        // Create some test files
        std::fs::write(repo_path.join("file1.txt"), "Hello, world!").unwrap();
        std::fs::write(repo_path.join("file2.txt"), "Another file").unwrap();
        
        // Create a subdirectory with a file
        let subdir = repo_path.join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("file3.txt"), "Subdirectory file").unwrap();

        // Create a .git directory (should be ignored)
        let git_dir = repo_path.join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        std::fs::write(git_dir.join("config"), "git config").unwrap();

        let (size_bytes, file_count) = GitCloner::calculate_repo_stats(repo_path).unwrap();

        assert_eq!(file_count, 3); // Should not count .git/config
        assert!(size_bytes > 0);
        
        // Verify the size calculation
        let expected_size = "Hello, world!".len() + "Another file".len() + "Subdirectory file".len();
        assert_eq!(size_bytes, expected_size as u64);
    }

    #[test]
    fn test_is_git_repository() {
        let temp_dir = TempDir::new().unwrap();
        
        // Not a Git repository
        assert!(!GitCloner::is_git_repository(temp_dir.path()));
        
        // Create a fake .git directory
        let git_dir = temp_dir.path().join(".git");
        std::fs::create_dir(&git_dir).unwrap();
        
        // Still not a valid Git repository (missing required files)
        assert!(!GitCloner::is_git_repository(temp_dir.path()));
    }

    #[tokio::test]
    async fn test_clone_invalid_url() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloneConfig {
            target_dir: temp_dir.path().to_path_buf(),
            show_progress: false,
            ..Default::default()
        };
        let cloner = GitCloner::new(config);

        let result = cloner.clone_repository("invalid-url").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IngestionError::InvalidRepositoryUrl { url } => {
                assert_eq!(url, "invalid-url");
            }
            _ => panic!("Expected InvalidRepositoryUrl error"),
        }
    }

    #[tokio::test]
    async fn test_clone_nonexistent_repository() {
        let temp_dir = TempDir::new().unwrap();
        let config = CloneConfig {
            target_dir: temp_dir.path().to_path_buf(),
            show_progress: false,
            timeout: Duration::from_secs(10),
            ..Default::default()
        };
        let cloner = GitCloner::new(config);

        let result = cloner.clone_repository("https://github.com/nonexistent/repository").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            IngestionError::GitCloneFailed { repo_url, cause: _ } => {
                assert_eq!(repo_url, "https://github.com/nonexistent/repository");
            }
            _ => panic!("Expected GitCloneFailed error"),
        }
    }

    #[test]
    fn test_clone_result_debug() {
        let result = CloneResult {
            repo_path: PathBuf::from("/tmp/test-repo"),
            repo_url: "https://github.com/user/repo".to_string(),
            commit_hash: "abc123".to_string(),
            reference: Some("main".to_string()),
            size_bytes: 1024,
            file_count: 10,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("test-repo"));
        assert!(debug_str.contains("abc123"));
        assert!(debug_str.contains("main"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_error_conversions() {
        let git_error = git2::Error::from_str("Test error");
        let ingestion_error: IngestionError = git_error.into();
        
        match ingestion_error {
            IngestionError::GitCloneFailed { repo_url: _, cause } => {
                assert!(cause.contains("Test error"));
            }
            _ => panic!("Expected GitCloneFailed error"),
        }
    }
}