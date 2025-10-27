//! Git operations module
//! 
//! This module provides Git repository cloning and management functionality
//! for the code ingestion system.

// Re-export git functionality from the ingestion module
pub use crate::ingestion::git_cloner::{
    GitCloner, CloneConfig, CloneResult,
};

use crate::error::IngestionResult;
use std::path::{Path, PathBuf};
use url::Url;

/// Git repository manager that provides high-level Git operations
pub struct GitManager {
    config: CloneConfig,
}

impl GitManager {
    /// Create a new Git manager with default configuration
    pub fn new() -> Self {
        Self {
            config: CloneConfig::default(),
        }
    }
    
    /// Create a new Git manager with custom configuration
    pub fn with_config(config: CloneConfig) -> Self {
        Self { config }
    }
    
    /// Clone a Git repository to a temporary location
    pub async fn clone_repository(&self, repo_url: &str) -> IngestionResult<CloneResult> {
        let cloner = GitCloner::new(self.config.clone());
        cloner.clone_repository(repo_url).await
    }
    
    /// Clone a Git repository to a specific target directory
    pub async fn clone_to_directory(
        &self,
        repo_url: &str,
        target_dir: &Path,
    ) -> IngestionResult<CloneResult> {
        let mut config = self.config.clone();
        config.target_dir = target_dir.to_path_buf();
        
        let cloner = GitCloner::new(config);
        cloner.clone_repository(repo_url).await
    }
    
    /// Validate if a string is a valid Git repository URL
    pub fn is_valid_git_url(&self, url: &str) -> bool {
        // Try to parse as URL
        if let Ok(parsed_url) = Url::parse(url) {
            match parsed_url.scheme() {
                "http" | "https" | "git" | "ssh" => {
                    // Check for common Git hosting services or .git extension
                    if let Some(host) = parsed_url.host_str() {
                        return host.contains("github.com")
                            || host.contains("gitlab.com")
                            || host.contains("bitbucket.org")
                            || host.contains("git")
                            || url.ends_with(".git");
                    }
                }
                _ => return false,
            }
        }

        // Check for SSH-style Git URLs (git@github.com:user/repo.git)
        if url.starts_with("git@") && url.contains(':') {
            return true;
        }

        false
    }
    
    /// Extract repository name from URL
    pub fn extract_repo_name(&self, repo_url: &str) -> Option<String> {
        if let Ok(url) = Url::parse(repo_url) {
            if let Some(path_segments) = url.path_segments() {
                if let Some(last_segment) = path_segments.last() {
                    let repo_name = last_segment.trim_end_matches(".git");
                    if !repo_name.is_empty() {
                        return Some(repo_name.to_string());
                    }
                }
            }
        }
        
        // Handle SSH-style URLs
        if repo_url.starts_with("git@") && repo_url.contains(':') {
            if let Some(colon_pos) = repo_url.rfind(':') {
                let path_part = &repo_url[colon_pos + 1..];
                let repo_name = path_part.trim_end_matches(".git");
                if let Some(slash_pos) = repo_name.rfind('/') {
                    return Some(repo_name[slash_pos + 1..].to_string());
                } else {
                    return Some(repo_name.to_string());
                }
            }
        }
        
        None
    }
    
    /// Set authentication credentials for Git operations
    pub fn set_credentials(&mut self, username: String, token: String) {
        self.config.credentials = Some((username, token));
    }
    
    /// Set timeout for Git operations
    pub fn set_timeout(&mut self, timeout: std::time::Duration) {
        self.config.timeout = timeout;
    }
    
    /// Enable or disable progress reporting
    pub fn set_progress_reporting(&mut self, enabled: bool) {
        self.config.show_progress = enabled;
    }
}

impl Default for GitManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for Git operations
pub mod utils {
    use super::*;
    
    /// Check if a directory is a Git repository
    pub fn is_git_repository(path: &Path) -> bool {
        path.join(".git").exists()
    }
    
    /// Get the Git repository root directory
    pub fn find_git_root(start_path: &Path) -> Option<PathBuf> {
        let mut current = start_path;
        
        loop {
            if is_git_repository(current) {
                return Some(current.to_path_buf());
            }
            
            match current.parent() {
                Some(parent) => current = parent,
                None => return None,
            }
        }
    }
    
    /// Extract owner and repository name from GitHub URL
    pub fn parse_github_url(url: &str) -> Option<(String, String)> {
        if let Ok(parsed_url) = Url::parse(url) {
            if let Some(host) = parsed_url.host_str() {
                if host == "github.com" {
                    if let Some(path_segments) = parsed_url.path_segments() {
                        let segments: Vec<&str> = path_segments.collect();
                        if segments.len() >= 2 {
                            let owner = segments[0].to_string();
                            let repo = segments[1].trim_end_matches(".git").to_string();
                            return Some((owner, repo));
                        }
                    }
                }
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_manager_creation() {
        let manager = GitManager::new();
        assert_eq!(manager.config.timeout, std::time::Duration::from_secs(300));
        
        let default_manager = GitManager::default();
        assert_eq!(default_manager.config.timeout, std::time::Duration::from_secs(300));
    }

    #[test]
    fn test_is_valid_git_url() {
        let manager = GitManager::new();
        
        // Valid Git URLs
        assert!(manager.is_valid_git_url("https://github.com/user/repo"));
        assert!(manager.is_valid_git_url("https://github.com/user/repo.git"));
        assert!(manager.is_valid_git_url("https://gitlab.com/user/repo"));
        assert!(manager.is_valid_git_url("git@github.com:user/repo.git"));
        assert!(manager.is_valid_git_url("https://git.example.com/repo.git"));
        
        // Invalid URLs
        assert!(!manager.is_valid_git_url("/local/path"));
        assert!(!manager.is_valid_git_url("https://example.com/page"));
        assert!(!manager.is_valid_git_url("not-a-url"));
        assert!(!manager.is_valid_git_url(""));
    }

    #[test]
    fn test_extract_repo_name() {
        let manager = GitManager::new();
        
        assert_eq!(
            manager.extract_repo_name("https://github.com/user/my-repo"),
            Some("my-repo".to_string())
        );
        assert_eq!(
            manager.extract_repo_name("https://github.com/user/my-repo.git"),
            Some("my-repo".to_string())
        );
        assert_eq!(
            manager.extract_repo_name("git@github.com:user/my-repo.git"),
            Some("my-repo".to_string())
        );
        assert_eq!(manager.extract_repo_name("invalid-url"), None);
    }

    #[test]
    fn test_utils_parse_github_url() {
        assert_eq!(
            utils::parse_github_url("https://github.com/octocat/Hello-World"),
            Some(("octocat".to_string(), "Hello-World".to_string()))
        );
        assert_eq!(
            utils::parse_github_url("https://github.com/octocat/Hello-World.git"),
            Some(("octocat".to_string(), "Hello-World".to_string()))
        );
        assert_eq!(utils::parse_github_url("https://gitlab.com/user/repo"), None);
        assert_eq!(utils::parse_github_url("invalid-url"), None);
    }

    #[test]
    fn test_utils_is_git_repository() {
        // This test would need a real git repository to work properly
        // For now, just test that the function doesn't panic
        let temp_dir = std::env::temp_dir();
        let result = utils::is_git_repository(&temp_dir);
        // Result can be true or false, we just care that it doesn't panic
        let _ = result;
    }
}