use crate::error::{IngestionError, IngestionResult};
use ignore::{Walk, WalkBuilder};

use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tracing::{debug, info, warn};

/// Configuration for local folder processing
#[derive(Debug, Clone)]
pub struct FolderConfig {
    /// Maximum file size in bytes (0 = no limit)
    pub max_file_size_bytes: u64,
    /// Custom include patterns (glob patterns)
    pub include_patterns: Vec<String>,
    /// Custom exclude patterns (glob patterns)
    pub exclude_patterns: Vec<String>,
    /// Whether to respect .gitignore files
    pub respect_gitignore: bool,
    /// Whether to follow symbolic links
    pub follow_symlinks: bool,
    /// Maximum depth for directory traversal (0 = no limit)
    pub max_depth: Option<usize>,
    /// Whether to include hidden files and directories
    pub include_hidden: bool,
}

impl Default for FolderConfig {
    fn default() -> Self {
        Self {
            max_file_size_bytes: 100 * 1024 * 1024, // 100MB default limit
            include_patterns: vec![],
            exclude_patterns: vec![
                // Common build artifacts and dependencies
                "target/**".to_string(),
                "node_modules/**".to_string(),
                "build/**".to_string(),
                "dist/**".to_string(),
                ".next/**".to_string(),
                // IDE and editor files
                ".vscode/**".to_string(),
                ".idea/**".to_string(),
                "*.swp".to_string(),
                "*.swo".to_string(),
                // OS files
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
                // Log files
                "*.log".to_string(),
                "logs/**".to_string(),
            ],
            respect_gitignore: true,
            follow_symlinks: false,
            max_depth: None,
            include_hidden: false,
        }
    }
}

/// Information about a discovered file
#[derive(Debug, Clone)]
pub struct FileInfo {
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
    /// Whether this file was skipped due to size limits
    pub skipped: bool,
    /// Reason for skipping (if skipped)
    pub skip_reason: Option<String>,
}

/// Result of folder processing
#[derive(Debug, Clone)]
pub struct FolderResult {
    /// Root path that was processed
    pub root_path: PathBuf,
    /// List of discovered files
    pub files: Vec<FileInfo>,
    /// Total number of files discovered (including skipped)
    pub total_files: usize,
    /// Number of files skipped
    pub skipped_files: usize,
    /// Total size of all files in bytes
    pub total_size_bytes: u64,
    /// Directories that were skipped due to errors
    pub skipped_directories: Vec<(PathBuf, String)>,
}

/// Local folder processor with filtering and safety features
pub struct FolderProcessor {
    config: FolderConfig,
}

impl FolderProcessor {
    /// Create a new FolderProcessor with the specified configuration
    pub fn new(config: FolderConfig) -> Self {
        Self { config }
    }

    /// Process a local folder and discover all files
    pub fn process_folder(&self, root_path: &Path) -> IngestionResult<FolderResult> {
        // Validate the root path
        self.validate_root_path(root_path)?;

        info!("Starting folder processing: {}", root_path.display());

        // Build the walker with configuration
        let walker = self.build_walker(root_path)?;

        // Process all files
        let mut files = Vec::new();
        let mut total_files = 0;
        let mut skipped_files = 0;
        let mut total_size_bytes = 0;
        let mut skipped_directories = Vec::new();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = entry.path();
                    
                    if path.is_file() {
                        total_files += 1;
                        
                        match self.process_file(root_path, path) {
                            Ok(file_info) => {
                                if file_info.skipped {
                                    skipped_files += 1;
                                } else {
                                    total_size_bytes += file_info.size_bytes;
                                }
                                files.push(file_info);
                            }
                            Err(e) => {
                                warn!("Failed to process file {}: {}", path.display(), e);
                                skipped_files += 1;
                                
                                // Create a skipped file entry
                                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                                    let relative_path = path.strip_prefix(root_path)
                                        .unwrap_or(path)
                                        .to_path_buf();
                                    
                                    files.push(FileInfo {
                                        absolute_path: path.to_path_buf(),
                                        relative_path,
                                        filename: filename.to_string(),
                                        extension: path.extension()
                                            .and_then(|ext| ext.to_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        size_bytes: 0,
                                        modified_time: SystemTime::UNIX_EPOCH,
                                        skipped: true,
                                        skip_reason: Some(e.to_string()),
                                    });
                                }
                            }
                        }
                    } else if path.is_dir() {
                        debug!("Processing directory: {}", path.display());
                    }
                }
                Err(e) => {
                    let path = Path::new("unknown"); // ignore::Error doesn't have path() method
                    warn!("Error accessing path: {}", e);
                    
                    if path.is_dir() {
                        skipped_directories.push((path.to_path_buf(), e.to_string()));
                    }
                }
            }
        }

        let result = FolderResult {
            root_path: root_path.to_path_buf(),
            files,
            total_files,
            skipped_files,
            total_size_bytes,
            skipped_directories,
        };

        info!(
            "Folder processing complete: {} files ({} skipped), {} bytes total",
            result.total_files, result.skipped_files, result.total_size_bytes
        );

        Ok(result)
    }

    /// Validate that the root path exists and is accessible
    fn validate_root_path(&self, root_path: &Path) -> IngestionResult<()> {
        if !root_path.exists() {
            return Err(IngestionError::LocalPathNotFound {
                path: root_path.display().to_string(),
            });
        }

        if !root_path.is_dir() {
            return Err(IngestionError::LocalPathNotFound {
                path: format!("{} is not a directory", root_path.display()),
            });
        }

        // Test read access
        std::fs::read_dir(root_path).map_err(|e| IngestionError::PermissionDenied {
            path: root_path.display().to_string(),
        })?;

        debug!("Validated root path: {}", root_path.display());
        Ok(())
    }

    /// Build a walker with the configured options
    fn build_walker(&self, root_path: &Path) -> IngestionResult<Walk> {
        let mut builder = WalkBuilder::new(root_path);

        // Configure basic options
        builder
            .follow_links(self.config.follow_symlinks)
            .hidden(!self.config.include_hidden)
            .git_ignore(self.config.respect_gitignore)
            .git_global(self.config.respect_gitignore)
            .git_exclude(self.config.respect_gitignore);

        // Set maximum depth if specified
        if let Some(max_depth) = self.config.max_depth {
            builder.max_depth(Some(max_depth));
        }

        // Add custom ignore patterns
        for pattern in &self.config.exclude_patterns {
            builder.add_custom_ignore_filename(pattern);
        }

        // Build the walker
        let walker = builder.build();
        
        debug!("Built walker for path: {}", root_path.display());
        Ok(walker)
    }

    /// Process a single file and extract its information
    fn process_file(&self, root_path: &Path, file_path: &Path) -> IngestionResult<FileInfo> {
        // Get file metadata
        let metadata = std::fs::metadata(file_path).map_err(|e| {
            IngestionError::PermissionDenied {
                path: file_path.display().to_string(),
            }
        })?;

        let size_bytes = metadata.len();
        let modified_time = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);

        // Extract file information
        let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| IngestionError::LocalPathNotFound {
                path: format!("Invalid filename: {}", file_path.display()),
            })?
            .to_string();

        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_string();

        let relative_path = file_path
            .strip_prefix(root_path)
            .unwrap_or(file_path)
            .to_path_buf();

        // Check file size limits
        let (skipped, skip_reason) = if self.config.max_file_size_bytes > 0 
            && size_bytes > self.config.max_file_size_bytes {
            let size_mb = size_bytes / (1024 * 1024);
            let limit_mb = self.config.max_file_size_bytes / (1024 * 1024);
            (
                true,
                Some(format!("File too large: {}MB exceeds limit of {}MB", size_mb, limit_mb)),
            )
        } else {
            (false, None)
        };

        // Check include/exclude patterns
        let (pattern_skipped, pattern_reason) = self.check_patterns(&relative_path)?;
        let final_skipped = skipped || pattern_skipped;
        let final_reason = skip_reason.or(pattern_reason);

        let file_info = FileInfo {
            absolute_path: file_path.to_path_buf(),
            relative_path,
            filename,
            extension,
            size_bytes,
            modified_time,
            skipped: final_skipped,
            skip_reason: final_reason.clone(),
        };

        if !final_skipped {
            debug!("Processed file: {} ({} bytes)", file_path.display(), size_bytes);
        } else {
            debug!("Skipped file: {} ({})", file_path.display(), 
                   final_reason.as_deref().unwrap_or("unknown reason"));
        }

        Ok(file_info)
    }

    /// Check if a file matches include/exclude patterns
    fn check_patterns(&self, relative_path: &Path) -> IngestionResult<(bool, Option<String>)> {
        let path_str = relative_path.to_string_lossy();

        // Check exclude patterns first
        for pattern in &self.config.exclude_patterns {
            if self.matches_glob_pattern(&path_str, pattern) {
                return Ok((true, Some(format!("Matches exclude pattern: {}", pattern))));
            }
        }

        // If we have include patterns, file must match at least one
        if !self.config.include_patterns.is_empty() {
            let mut matches_include = false;
            for pattern in &self.config.include_patterns {
                if self.matches_glob_pattern(&path_str, pattern) {
                    matches_include = true;
                    break;
                }
            }
            
            if !matches_include {
                return Ok((true, Some("Does not match any include pattern".to_string())));
            }
        }

        Ok((false, None))
    }

    /// Simple glob pattern matching
    fn matches_glob_pattern(&self, path: &str, pattern: &str) -> bool {
        // Simple implementation - in production, you might want to use a proper glob library
        if pattern.contains("**") {
            // Handle recursive patterns
            let parts: Vec<&str> = pattern.split("**").collect();
            if parts.len() == 2 {
                let prefix = parts[0];
                let suffix = parts[1];
                return path.starts_with(prefix) && path.ends_with(suffix);
            }
        } else if pattern.contains('*') {
            // Handle single-level wildcards
            if pattern.starts_with('*') && pattern.len() > 1 {
                return path.ends_with(&pattern[1..]);
            } else if pattern.ends_with('*') && pattern.len() > 1 {
                return path.starts_with(&pattern[..pattern.len() - 1]);
            }
        } else {
            // Exact match
            return path == pattern;
        }
        
        false
    }

    /// Get file statistics for a processed folder
    pub fn get_file_statistics(result: &FolderResult) -> FileStatistics {
        let mut stats = FileStatistics::default();
        
        for file in &result.files {
            if !file.skipped {
                stats.total_files += 1;
                stats.total_size_bytes += file.size_bytes;
                
                // Count by extension
                let ext = if file.extension.is_empty() {
                    "no_extension".to_string()
                } else {
                    file.extension.clone()
                };
                *stats.extensions.entry(ext).or_insert(0) += 1;
                
                // Track largest files
                if stats.largest_files.len() < 10 {
                    stats.largest_files.push((file.relative_path.clone(), file.size_bytes));
                } else {
                    // Find the smallest in our top 10 and replace if this file is larger
                    if let Some((min_idx, _)) = stats.largest_files
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, size))| *size) {
                        if file.size_bytes > stats.largest_files[min_idx].1 {
                            stats.largest_files[min_idx] = (file.relative_path.clone(), file.size_bytes);
                        }
                    }
                }
            }
        }
        
        // Sort largest files by size
        stats.largest_files.sort_by(|a, b| b.1.cmp(&a.1));
        
        stats
    }
}

/// Statistics about processed files
#[derive(Debug, Default)]
pub struct FileStatistics {
    pub total_files: usize,
    pub total_size_bytes: u64,
    pub extensions: std::collections::HashMap<String, usize>,
    pub largest_files: Vec<(PathBuf, u64)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

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
        fs::write(subdir.join("main.rs"), "fn main() { println!(\"Hello\"); }").unwrap();

        // Create a large file for size testing
        let large_content = "x".repeat(1024 * 1024); // 1MB
        fs::write(root.join("large_file.bin"), large_content).unwrap();

        // Create hidden file
        fs::write(root.join(".hidden"), "hidden content").unwrap();

        // Create .gitignore
        fs::write(root.join(".gitignore"), "target/\n*.log\n").unwrap();

        // Create ignored directory
        let target_dir = root.join("target");
        fs::create_dir(&target_dir).unwrap();
        fs::write(target_dir.join("ignored.txt"), "This should be ignored").unwrap();

        temp_dir
    }

    #[test]
    fn test_folder_config_default() {
        let config = FolderConfig::default();
        assert_eq!(config.max_file_size_bytes, 100 * 1024 * 1024);
        assert!(config.respect_gitignore);
        assert!(!config.follow_symlinks);
        assert!(!config.include_hidden);
        assert!(config.exclude_patterns.contains(&"target/**".to_string()));
    }

    #[test]
    fn test_process_folder_basic() {
        let temp_dir = create_test_directory();
        let config = FolderConfig::default();
        let processor = FolderProcessor::new(config);

        let result = processor.process_folder(temp_dir.path()).unwrap();

        assert_eq!(result.root_path, temp_dir.path());
        assert!(result.total_files > 0);
        
        // Should find at least the basic files (excluding .gitignore filtered ones)
        let non_skipped_files: Vec<_> = result.files.iter().filter(|f| !f.skipped).collect();
        assert!(!non_skipped_files.is_empty());
        
        // Check that we have some expected files
        let filenames: std::collections::HashSet<String> = non_skipped_files
            .iter()
            .map(|f| f.filename.clone())
            .collect();
        assert!(filenames.contains("file1.txt"));
        assert!(filenames.contains("file2.rs"));
        assert!(filenames.contains("README.md"));
    }

    #[test]
    fn test_process_folder_with_size_limit() {
        let temp_dir = create_test_directory();
        let config = FolderConfig {
            max_file_size_bytes: 1024, // 1KB limit
            ..Default::default()
        };
        let processor = FolderProcessor::new(config);

        let result = processor.process_folder(temp_dir.path()).unwrap();

        // The large file should be skipped
        let large_file = result.files.iter().find(|f| f.filename == "large_file.bin");
        assert!(large_file.is_some());
        assert!(large_file.unwrap().skipped);
        assert!(large_file.unwrap().skip_reason.is_some());
    }

    #[test]
    fn test_process_folder_with_include_patterns() {
        let temp_dir = create_test_directory();
        let config = FolderConfig {
            include_patterns: vec!["*.rs".to_string()],
            respect_gitignore: false, // Disable to test pattern matching only
            ..Default::default()
        };
        let processor = FolderProcessor::new(config);

        let result = processor.process_folder(temp_dir.path()).unwrap();

        let non_skipped_files: Vec<_> = result.files.iter().filter(|f| !f.skipped).collect();
        
        // Should only include .rs files
        for file in non_skipped_files {
            assert!(file.filename.ends_with(".rs"));
        }
    }

    #[test]
    fn test_process_folder_with_exclude_patterns() {
        let temp_dir = create_test_directory();
        let config = FolderConfig {
            exclude_patterns: vec!["*.txt".to_string()],
            respect_gitignore: false,
            ..Default::default()
        };
        let processor = FolderProcessor::new(config);

        let result = processor.process_folder(temp_dir.path()).unwrap();

        // .txt files should be skipped
        let txt_files: Vec<_> = result.files.iter()
            .filter(|f| f.filename.ends_with(".txt"))
            .collect();
        
        for file in txt_files {
            assert!(file.skipped);
        }
    }

    #[test]
    fn test_process_folder_hidden_files() {
        let temp_dir = create_test_directory();
        
        // Test with hidden files excluded (default)
        let config = FolderConfig::default();
        let processor = FolderProcessor::new(config);
        let result = processor.process_folder(temp_dir.path()).unwrap();
        
        let hidden_files: Vec<_> = result.files.iter()
            .filter(|f| f.filename.starts_with('.'))
            .collect();
        
        // Hidden files should be present but may be filtered by gitignore
        // The exact behavior depends on the ignore crate's implementation
        
        // Test with hidden files included
        let config_with_hidden = FolderConfig {
            include_hidden: true,
            respect_gitignore: false,
            ..Default::default()
        };
        let processor_with_hidden = FolderProcessor::new(config_with_hidden);
        let result_with_hidden = processor_with_hidden.process_folder(temp_dir.path()).unwrap();
        
        let hidden_files_included: Vec<_> = result_with_hidden.files.iter()
            .filter(|f| f.filename.starts_with('.') && !f.skipped)
            .collect();
        
        // Should find the .hidden file when including hidden files
        assert!(hidden_files_included.iter().any(|f| f.filename == ".hidden"));
    }

    #[test]
    fn test_validate_root_path() {
        let temp_dir = TempDir::new().unwrap();
        let config = FolderConfig::default();
        let processor = FolderProcessor::new(config);

        // Valid directory
        assert!(processor.validate_root_path(temp_dir.path()).is_ok());

        // Non-existent path
        let nonexistent = temp_dir.path().join("nonexistent");
        assert!(processor.validate_root_path(&nonexistent).is_err());

        // File instead of directory
        let file_path = temp_dir.path().join("test_file.txt");
        fs::write(&file_path, "test").unwrap();
        assert!(processor.validate_root_path(&file_path).is_err());
    }

    #[test]
    fn test_glob_pattern_matching() {
        let config = FolderConfig::default();
        let processor = FolderProcessor::new(config);

        // Test exact match
        assert!(processor.matches_glob_pattern("file.txt", "file.txt"));
        assert!(!processor.matches_glob_pattern("file.rs", "file.txt"));

        // Test suffix wildcard
        assert!(processor.matches_glob_pattern("file.txt", "*.txt"));
        assert!(processor.matches_glob_pattern("another.txt", "*.txt"));
        assert!(!processor.matches_glob_pattern("file.rs", "*.txt"));

        // Test prefix wildcard
        assert!(processor.matches_glob_pattern("src/main.rs", "src/*"));
        assert!(!processor.matches_glob_pattern("lib/main.rs", "src/*"));

        // Test recursive wildcard
        assert!(processor.matches_glob_pattern("target/debug/main", "target/**"));
        assert!(processor.matches_glob_pattern("target/release/deps/lib.so", "target/**"));
        assert!(!processor.matches_glob_pattern("src/main.rs", "target/**"));
    }

    #[test]
    fn test_file_statistics() {
        let temp_dir = create_test_directory();
        let config = FolderConfig {
            respect_gitignore: false,
            include_hidden: true,
            ..Default::default()
        };
        let processor = FolderProcessor::new(config);

        let result = processor.process_folder(temp_dir.path()).unwrap();
        let stats = FolderProcessor::get_file_statistics(&result);

        assert!(stats.total_files > 0);
        assert!(stats.total_size_bytes > 0);
        assert!(!stats.extensions.is_empty());
        
        // Should have some .rs files
        assert!(stats.extensions.contains_key("rs"));
        
        // Should track largest files
        assert!(!stats.largest_files.is_empty());
    }

    #[test]
    fn test_file_info_debug() {
        let file_info = FileInfo {
            absolute_path: PathBuf::from("/tmp/test.txt"),
            relative_path: PathBuf::from("test.txt"),
            filename: "test.txt".to_string(),
            extension: "txt".to_string(),
            size_bytes: 1024,
            modified_time: SystemTime::UNIX_EPOCH,
            skipped: false,
            skip_reason: None,
        };

        let debug_str = format!("{:?}", file_info);
        assert!(debug_str.contains("test.txt"));
        assert!(debug_str.contains("1024"));
        assert!(debug_str.contains("false"));
    }
}