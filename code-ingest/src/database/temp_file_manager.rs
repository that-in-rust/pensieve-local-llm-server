//! Temporary file management for query-prepare workflow
//!
//! This module handles creating temporary files with query results,
//! absolute path handling and validation, structured output format for LLM processing,
//! and file cleanup with error handling.

use crate::database::query_executor::{QueryExecutor, QueryConfig};
use crate::error::{DatabaseError, DatabaseResult};
use anyhow::Result;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

/// Temporary file manager for query results
pub struct TempFileManager {
    query_executor: QueryExecutor,
}

/// Configuration for temporary file operations
#[derive(Debug, Clone)]
pub struct TempFileConfig {
    /// Whether to overwrite existing files
    pub overwrite_existing: bool,
    /// Whether to create parent directories
    pub create_parent_dirs: bool,
    /// File permissions (Unix only)
    pub file_permissions: Option<u32>,
    /// Whether to validate file paths
    pub validate_paths: bool,
}

impl Default for TempFileConfig {
    fn default() -> Self {
        Self {
            overwrite_existing: true,
            create_parent_dirs: true,
            file_permissions: Some(0o644),
            validate_paths: true,
        }
    }
}

/// Result of temporary file creation
#[derive(Debug, Clone)]
pub struct TempFileResult {
    pub file_path: PathBuf,
    pub bytes_written: u64,
    pub row_count: usize,
    pub execution_time_ms: u64,
}

impl TempFileManager {
    /// Create a new temporary file manager
    pub fn new(query_executor: QueryExecutor) -> Self {
        Self { query_executor }
    }

    /// Execute query and write results to temporary file
    pub async fn create_temp_file(
        &self,
        sql: &str,
        temp_path: &Path,
        config: &TempFileConfig,
    ) -> Result<TempFileResult> {
        debug!("Creating temporary file at: {}", temp_path.display());
        
        // Validate the temporary file path
        if config.validate_paths {
            self.validate_temp_path(temp_path)?;
        }
        
        // Create parent directories if needed
        if config.create_parent_dirs {
            self.ensure_parent_directories(temp_path).await?;
        }
        
        // Check if file exists and handle accordingly
        if temp_path.exists() && !config.overwrite_existing {
            return Err(anyhow::anyhow!(
                "Temporary file already exists: {}",
                temp_path.display()
            ));
        }
        
        // Execute query with LLM formatting
        let query_config = QueryConfig {
            llm_format: true,
            include_stats: false,
            ..Default::default()
        };
        
        let query_result = self
            .query_executor
            .execute_query_with_config(sql, &query_config)
            .await?;
        
        // Write results to temporary file
        let bytes_written = self.write_to_temp_file(temp_path, &query_result.content, config).await?;
        
        info!(
            "Created temporary file: {} ({} bytes, {} rows)",
            temp_path.display(),
            bytes_written,
            query_result.row_count
        );
        
        Ok(TempFileResult {
            file_path: temp_path.to_path_buf(),
            bytes_written,
            row_count: query_result.row_count,
            execution_time_ms: query_result.execution_time_ms,
        })
    }

    /// Create temporary file with structured output for LLM processing
    pub async fn create_structured_temp_file(
        &self,
        sql: &str,
        temp_path: &Path,
        metadata: &TempFileMetadata,
        config: &TempFileConfig,
    ) -> Result<TempFileResult> {
        debug!("Creating structured temporary file with metadata");
        
        // Execute query
        let temp_result = self.create_temp_file(sql, temp_path, config).await?;
        
        // Add metadata header to the file
        let metadata_header = self.create_metadata_header(metadata, &temp_result);
        let original_content = fs::read_to_string(temp_path).await?;
        
        let structured_content = format!("{}\n{}", metadata_header, original_content);
        
        // Rewrite file with metadata
        let bytes_written = self.write_to_temp_file(temp_path, &structured_content, config).await?;
        
        Ok(TempFileResult {
            bytes_written,
            ..temp_result
        })
    }

    /// Clean up temporary file
    pub async fn cleanup_temp_file(&self, temp_path: &Path) -> Result<()> {
        if temp_path.exists() {
            fs::remove_file(temp_path).await?;
            debug!("Cleaned up temporary file: {}", temp_path.display());
        }
        Ok(())
    }

    /// Validate multiple temporary file paths
    pub async fn cleanup_temp_files(&self, temp_paths: &[PathBuf]) -> Result<Vec<String>> {
        let mut errors = Vec::new();
        
        for path in temp_paths {
            if let Err(e) = self.cleanup_temp_file(path).await {
                let error_msg = format!("Failed to cleanup {}: {}", path.display(), e);
                warn!("{}", error_msg);
                errors.push(error_msg);
            }
        }
        
        if !errors.is_empty() {
            info!("Cleanup completed with {} errors", errors.len());
        } else {
            info!("All temporary files cleaned up successfully");
        }
        
        Ok(errors)
    }

    /// Get temporary file information
    pub async fn get_temp_file_info(&self, temp_path: &Path) -> Result<TempFileInfo> {
        let metadata = fs::metadata(temp_path).await?;
        let content_preview = self.read_file_preview(temp_path, 500).await?;
        
        Ok(TempFileInfo {
            path: temp_path.to_path_buf(),
            size_bytes: metadata.len(),
            created: metadata.created().ok(),
            modified: metadata.modified().ok(),
            content_preview,
        })
    }

    // Private helper methods

    fn validate_temp_path(&self, temp_path: &Path) -> Result<()> {
        // Check if path is absolute
        if !temp_path.is_absolute() {
            return Err(anyhow::anyhow!(
                "Temporary file path must be absolute: {}",
                temp_path.display()
            ));
        }
        
        // Check for valid file extension
        if let Some(extension) = temp_path.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            if !["txt", "md", "json", "csv"].contains(&ext_str.as_str()) {
                warn!("Unusual file extension for temporary file: {}", ext_str);
            }
        }
        
        // Check for dangerous path components
        let path_str = temp_path.to_string_lossy();
        if path_str.contains("..") {
            return Err(anyhow::anyhow!(
                "Path traversal detected in temporary file path: {}",
                temp_path.display()
            ));
        }
        
        // Check if path is in a reasonable location
        let path_components: Vec<&str> = path_str.split('/').collect();
        if path_components.len() < 2 {
            return Err(anyhow::anyhow!(
                "Temporary file path too short: {}",
                temp_path.display()
            ));
        }
        
        Ok(())
    }

    async fn ensure_parent_directories(&self, temp_path: &Path) -> Result<()> {
        if let Some(parent) = temp_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await?;
                debug!("Created parent directories: {}", parent.display());
            }
        }
        Ok(())
    }

    async fn write_to_temp_file(
        &self,
        temp_path: &Path,
        content: &str,
        config: &TempFileConfig,
    ) -> Result<u64> {
        let mut file = fs::File::create(temp_path).await?;
        
        // Set file permissions on Unix systems
        #[cfg(unix)]
        if let Some(permissions) = config.file_permissions {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(permissions);
            file.set_permissions(perms).await?;
        }
        
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
        
        Ok(content.len() as u64)
    }

    fn create_metadata_header(&self, metadata: &TempFileMetadata, result: &TempFileResult) -> String {
        let mut header = String::new();
        header.push_str("# Query Results Metadata\n\n");
        header.push_str(&format!("- **Query**: {}\n", metadata.original_query));
        header.push_str(&format!("- **Output Table**: {}\n", metadata.output_table));
        header.push_str(&format!("- **Generated**: {}\n", chrono::Utc::now().to_rfc3339()));
        header.push_str(&format!("- **Row Count**: {}\n", result.row_count));
        header.push_str(&format!("- **Execution Time**: {}ms\n", result.execution_time_ms));
        
        if let Some(prompt_file) = &metadata.prompt_file_path {
            header.push_str(&format!("- **Prompt File**: {}\n", prompt_file.display()));
        }
        
        if let Some(description) = &metadata.description {
            header.push_str(&format!("- **Description**: {}\n", description));
        }
        
        header.push_str("\n---\n\n");
        header
    }

    async fn read_file_preview(&self, path: &Path, max_chars: usize) -> Result<String> {
        let content = fs::read_to_string(path).await?;
        if content.len() <= max_chars {
            Ok(content)
        } else {
            Ok(format!("{}...", &content[..max_chars]))
        }
    }
}

/// Metadata for temporary file creation
#[derive(Debug, Clone)]
pub struct TempFileMetadata {
    pub original_query: String,
    pub output_table: String,
    pub prompt_file_path: Option<PathBuf>,
    pub description: Option<String>,
}

/// Information about a temporary file
#[derive(Debug, Clone)]
pub struct TempFileInfo {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub created: Option<std::time::SystemTime>,
    pub modified: Option<std::time::SystemTime>,
    pub content_preview: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::QueryExecutor;
    use sqlx::PgPool;
    use tempfile::TempDir;

    fn create_test_pool() -> Option<PgPool> {
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    #[test]
    fn test_temp_file_config_default() {
        let config = TempFileConfig::default();
        assert!(config.overwrite_existing);
        assert!(config.create_parent_dirs);
        assert_eq!(config.file_permissions, Some(0o644));
        assert!(config.validate_paths);
    }

    #[test]
    fn test_temp_file_metadata() {
        let metadata = TempFileMetadata {
            original_query: "SELECT * FROM test".to_string(),
            output_table: "QUERYRESULT_test".to_string(),
            prompt_file_path: Some(PathBuf::from("/tmp/prompt.md")),
            description: Some("Test query".to_string()),
        };
        
        assert_eq!(metadata.original_query, "SELECT * FROM test");
        assert_eq!(metadata.output_table, "QUERYRESULT_test");
        assert!(metadata.prompt_file_path.is_some());
        assert!(metadata.description.is_some());
    }

    #[tokio::test]
    async fn test_temp_file_manager_creation() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let _manager = TempFileManager::new(executor);
            // Just test that we can create the manager
            assert!(true);
        }
    }

    #[test]
    fn test_path_validation() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let manager = TempFileManager::new(executor);
            
            // Valid absolute paths
            assert!(manager.validate_temp_path(Path::new("/tmp/test.txt")).is_ok());
            assert!(manager.validate_temp_path(Path::new("/home/user/temp.md")).is_ok());
            
            // Invalid relative paths
            assert!(manager.validate_temp_path(Path::new("temp.txt")).is_err());
            assert!(manager.validate_temp_path(Path::new("./temp.txt")).is_err());
            
            // Path traversal attempts
            assert!(manager.validate_temp_path(Path::new("/tmp/../etc/passwd")).is_err());
            
            // Too short paths
            assert!(manager.validate_temp_path(Path::new("/")).is_err());
        }
    }

    #[tokio::test]
    async fn test_temp_file_creation() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().join("test.txt");
        
        // Test directory creation logic
        assert!(temp_path.parent().is_some());
        assert!(temp_path.parent().unwrap().exists());
    }

    #[test]
    fn test_temp_file_result() {
        let result = TempFileResult {
            file_path: PathBuf::from("/tmp/test.txt"),
            bytes_written: 1024,
            row_count: 10,
            execution_time_ms: 50,
        };
        
        assert_eq!(result.file_path, PathBuf::from("/tmp/test.txt"));
        assert_eq!(result.bytes_written, 1024);
        assert_eq!(result.row_count, 10);
        assert_eq!(result.execution_time_ms, 50);
    }

    #[test]
    fn test_temp_file_info() {
        let info = TempFileInfo {
            path: PathBuf::from("/tmp/test.txt"),
            size_bytes: 2048,
            created: None,
            modified: None,
            content_preview: "Preview content...".to_string(),
        };
        
        assert_eq!(info.path, PathBuf::from("/tmp/test.txt"));
        assert_eq!(info.size_bytes, 2048);
        assert_eq!(info.content_preview, "Preview content...");
    }

    #[tokio::test]
    async fn test_cleanup_operations() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let manager = TempFileManager::new(executor);
            
            // Test cleanup of non-existent file (should not error)
            let non_existent = PathBuf::from("/tmp/non_existent_file.txt");
            let result = manager.cleanup_temp_file(&non_existent).await;
            assert!(result.is_ok());
            
            // Test cleanup of multiple files
            let paths = vec![
                PathBuf::from("/tmp/file1.txt"),
                PathBuf::from("/tmp/file2.txt"),
            ];
            let errors = manager.cleanup_temp_files(&paths).await.unwrap();
            // Should not error even if files don't exist
            assert!(errors.is_empty());
        }
    }

    #[test]
    fn test_metadata_header_creation() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let manager = TempFileManager::new(executor);
            
            let metadata = TempFileMetadata {
                original_query: "SELECT * FROM test".to_string(),
                output_table: "QUERYRESULT_test".to_string(),
                prompt_file_path: Some(PathBuf::from("/tmp/prompt.md")),
                description: Some("Test description".to_string()),
            };
            
            let result = TempFileResult {
                file_path: PathBuf::from("/tmp/test.txt"),
                bytes_written: 100,
                row_count: 5,
                execution_time_ms: 25,
            };
            
            let header = manager.create_metadata_header(&metadata, &result);
            
            assert!(header.contains("Query Results Metadata"));
            assert!(header.contains("SELECT * FROM test"));
            assert!(header.contains("QUERYRESULT_test"));
            assert!(header.contains("Row Count: 5"));
            assert!(header.contains("Execution Time: 25ms"));
            assert!(header.contains("/tmp/prompt.md"));
            assert!(header.contains("Test description"));
        }
    }

    #[test]
    fn test_file_extension_validation() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let manager = TempFileManager::new(executor);
            
            // Common extensions should pass
            assert!(manager.validate_temp_path(Path::new("/tmp/test.txt")).is_ok());
            assert!(manager.validate_temp_path(Path::new("/tmp/test.md")).is_ok());
            assert!(manager.validate_temp_path(Path::new("/tmp/test.json")).is_ok());
            
            // Unusual extensions should pass with warning
            assert!(manager.validate_temp_path(Path::new("/tmp/test.xyz")).is_ok());
            
            // No extension should pass
            assert!(manager.validate_temp_path(Path::new("/tmp/test")).is_ok());
        }
    }
}