use super::{FileProcessor, FileType, ProcessedFile};
use crate::error::{ProcessingError, ProcessingResult};
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokio::fs as async_fs;
use encoding_rs::{Encoding, UTF_8};
use ignore::{gitignore::Gitignore, Match};

/// Configuration for text file processing
#[derive(Debug, Clone)]
pub struct TextProcessorConfig {
    /// Maximum file size in bytes (default: 100MB)
    pub max_file_size_bytes: u64,
    /// Maximum number of lines to process (default: 1M lines)
    pub max_lines: usize,
    /// Whether to respect .gitignore patterns
    pub respect_gitignore: bool,
    /// Buffer size for streaming large files (default: 64KB)
    pub buffer_size: usize,
    /// Whether to detect encoding automatically
    pub auto_detect_encoding: bool,
}

impl Default for TextProcessorConfig {
    fn default() -> Self {
        Self {
            max_file_size_bytes: 100 * 1024 * 1024, // 100MB
            max_lines: 1_000_000,                    // 1M lines
            respect_gitignore: true,
            buffer_size: 64 * 1024,                  // 64KB
            auto_detect_encoding: true,
        }
    }
}

/// Processor for direct text files (Type 1)
#[derive(Debug, Clone)]
pub struct TextProcessor {
    config: TextProcessorConfig,
    gitignore: Option<Gitignore>,
}

impl TextProcessor {
    /// Create a new text processor with default configuration
    pub fn new() -> Self {
        Self::with_config(TextProcessorConfig::default())
    }

    /// Create a new text processor with custom configuration
    pub fn with_config(config: TextProcessorConfig) -> Self {
        Self {
            config,
            gitignore: None,
        }
    }

    /// Set up gitignore patterns from a directory
    pub fn with_gitignore<P: AsRef<Path>>(mut self, repo_root: P) -> ProcessingResult<Self> {
        if self.config.respect_gitignore {
            let gitignore_path = repo_root.as_ref().join(".gitignore");
            if gitignore_path.exists() {
                let mut builder = ignore::gitignore::GitignoreBuilder::new(repo_root);
                builder.add(&gitignore_path);
                
                match builder.build() {
                    Ok(gitignore) => {
                        self.gitignore = Some(gitignore);
                    }
                    Err(e) => {
                        return Err(ProcessingError::ContentAnalysisFailed {
                            path: gitignore_path.display().to_string(),
                            cause: format!("Failed to parse .gitignore: {}", e),
                        });
                    }
                }
            }
        }
        Ok(self)
    }

    /// Check if a file should be ignored based on .gitignore patterns
    pub fn should_ignore<P: AsRef<Path>>(&self, file_path: P) -> bool {
        if let Some(ref gitignore) = self.gitignore {
            match gitignore.matched(file_path.as_ref(), false) {
                Match::Ignore(_) => true,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Detect the encoding of a file
    fn detect_encoding(&self, content: &[u8]) -> &'static Encoding {
        if !self.config.auto_detect_encoding {
            return UTF_8;
        }

        // Try to detect encoding using encoding_rs
        let (encoding, _) = encoding_rs::Encoding::for_bom(content)
            .unwrap_or((UTF_8, 0));

        // If no BOM, try to detect based on content
        if encoding == UTF_8 {
            // Simple heuristic: if it's valid UTF-8, use UTF-8
            if std::str::from_utf8(content).is_ok() {
                UTF_8
            } else {
                // Try common encodings
                for encoding in &[encoding_rs::WINDOWS_1252, encoding_rs::ISO_8859_2] {
                    let (_, _, had_errors) = encoding.decode(content);
                    if !had_errors {
                        return encoding;
                    }
                }
                // Fallback to UTF-8 with replacement characters
                UTF_8
            }
        } else {
            encoding
        }
    }

    /// Read file content with encoding detection
    async fn read_file_content<P: AsRef<Path>>(&self, file_path: P) -> ProcessingResult<String> {
        let path = file_path.as_ref();
        
        // Check file size first
        let metadata = async_fs::metadata(path).await.map_err(|e| {
            ProcessingError::FileReadFailed {
                path: path.display().to_string(),
                cause: e.to_string(),
            }
        })?;

        if metadata.len() > self.config.max_file_size_bytes {
            return Err(ProcessingError::FileTooLarge {
                path: path.display().to_string(),
                size_mb: metadata.len() / (1024 * 1024),
            });
        }

        // Read file content
        let content_bytes = async_fs::read(path).await.map_err(|e| {
            ProcessingError::FileReadFailed {
                path: path.display().to_string(),
                cause: e.to_string(),
            }
        })?;

        // Detect encoding and decode
        let encoding = self.detect_encoding(&content_bytes);
        let (decoded, _, had_errors) = encoding.decode(&content_bytes);

        if had_errors {
            return Err(ProcessingError::EncodingDetectionFailed {
                path: path.display().to_string(),
            });
        }

        Ok(decoded.into_owned())
    }

    /// Analyze text content for metrics
    fn analyze_content(&self, content: &str) -> ProcessingResult<ContentAnalysis> {
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();

        // Check line limit
        if line_count > self.config.max_lines {
            return Err(ProcessingError::ContentAnalysisFailed {
                path: "content".to_string(),
                cause: format!("File has {} lines, exceeds limit of {}", line_count, self.config.max_lines),
            });
        }

        // Count words (split by whitespace)
        let word_count = content
            .split_whitespace()
            .count();

        // Estimate tokens (rough approximation: words * 1.3 for subword tokenization)
        let token_count = (word_count as f64 * 1.3) as usize;

        Ok(ContentAnalysis {
            line_count: line_count as i32,
            word_count: word_count as i32,
            token_count: token_count as i32,
        })
    }

    /// Process a file with streaming for large files
    async fn process_large_file<P: AsRef<Path>>(&self, file_path: P) -> ProcessingResult<ProcessedFile> {
        let path = file_path.as_ref();
        
        // For very large files, we'll read in chunks and analyze incrementally
        let file = std::fs::File::open(path).map_err(|e| {
            ProcessingError::FileReadFailed {
                path: path.display().to_string(),
                cause: e.to_string(),
            }
        })?;

        let mut reader = BufReader::with_capacity(self.config.buffer_size, file);
        let mut content = String::new();
        let mut line_count = 0i32;
        let mut word_count = 0i32;
        let mut total_bytes = 0u64;

        // Read line by line to avoid loading entire file into memory
        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line).map_err(|e| {
                ProcessingError::FileReadFailed {
                    path: path.display().to_string(),
                    cause: e.to_string(),
                }
            })?;

            if bytes_read == 0 {
                break; // EOF
            }

            total_bytes += bytes_read as u64;
            if total_bytes > self.config.max_file_size_bytes {
                return Err(ProcessingError::FileTooLarge {
                    path: path.display().to_string(),
                    size_mb: total_bytes / (1024 * 1024),
                });
            }

            line_count += 1;
            if line_count > self.config.max_lines as i32 {
                return Err(ProcessingError::ContentAnalysisFailed {
                    path: path.display().to_string(),
                    cause: format!("File has more than {} lines", self.config.max_lines),
                });
            }

            word_count += line.split_whitespace().count() as i32;
            content.push_str(&line);
        }

        let token_count = (word_count as f64 * 1.3) as i32;

        self.create_processed_file(path, content, line_count, word_count, token_count, total_bytes as i64)
    }

    /// Create a ProcessedFile from analyzed content
    fn create_processed_file<P: AsRef<Path>>(
        &self,
        file_path: P,
        content: String,
        line_count: i32,
        word_count: i32,
        token_count: i32,
        file_size_bytes: i64,
    ) -> ProcessingResult<ProcessedFile> {
        let path = file_path.as_ref();
        
        let filename = path
            .file_name()
            .ok_or_else(|| ProcessingError::FileReadFailed {
                path: path.display().to_string(),
                cause: "Invalid filename".to_string(),
            })?
            .to_string_lossy()
            .to_string();

        let extension = path
            .extension()
            .map(|ext| ext.to_string_lossy().to_string())
            .unwrap_or_default();

        let filepath = path.to_string_lossy().to_string();
        let absolute_path = path
            .canonicalize()
            .map_err(|e| ProcessingError::FileReadFailed {
                path: path.display().to_string(),
                cause: e.to_string(),
            })?
            .to_string_lossy()
            .to_string();

        // For relative path, we'll use the provided path as-is
        // In a real implementation, this would be relative to the repo root
        let relative_path = filepath.clone();

        Ok(ProcessedFile {
            filepath,
            filename,
            extension,
            file_size_bytes,
            line_count: Some(line_count),
            word_count: Some(word_count),
            token_count: Some(token_count),
            content_text: Some(content),
            file_type: FileType::DirectText,
            conversion_command: None,
            relative_path,
            absolute_path,
            skipped: false,
            skip_reason: None,
        })
    }
}

#[async_trait::async_trait]
impl FileProcessor for TextProcessor {
    fn can_process(&self, file_path: &Path) -> bool {
        use super::classifier::FileClassifier;
        
        // Check if file should be ignored
        if self.should_ignore(file_path) {
            return false;
        }

        // Check if it's a direct text file
        let classifier = FileClassifier::new();
        classifier.classify_file(file_path) == FileType::DirectText
    }

    async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
        // Check if we can process this file
        if !self.can_process(file_path) {
            return Err(ProcessingError::UnsupportedFileType {
                path: file_path.display().to_string(),
                extension: file_path
                    .extension()
                    .map(|ext| ext.to_string_lossy().to_string())
                    .unwrap_or_default(),
            });
        }

        // Get file metadata
        let metadata = async_fs::metadata(file_path).await.map_err(|e| {
            ProcessingError::FileReadFailed {
                path: file_path.display().to_string(),
                cause: e.to_string(),
            }
        })?;

        let file_size = metadata.len();

        // Choose processing strategy based on file size
        if file_size > self.config.buffer_size as u64 * 10 {
            // Use streaming for large files
            self.process_large_file(file_path).await
        } else {
            // Read entire file for smaller files
            let content = self.read_file_content(file_path).await?;
            let analysis = self.analyze_content(&content)?;
            
            self.create_processed_file(
                file_path,
                content,
                analysis.line_count,
                analysis.word_count,
                analysis.token_count,
                file_size as i64,
            )
        }
    }

    fn get_file_type(&self) -> FileType {
        FileType::DirectText
    }
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Content analysis results
#[derive(Debug, Clone)]
struct ContentAnalysis {
    line_count: i32,
    word_count: i32,
    token_count: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn create_test_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".rs").unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[tokio::test]
    async fn test_simple_text_processing() {
        let processor = TextProcessor::new();
        let content = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let file = create_test_file(content);

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::DirectText);
        assert_eq!(result.line_count, Some(3));
        assert_eq!(result.word_count, Some(6)); // fn, main, println, Hello, world, (split by whitespace)
        assert!(result.token_count.unwrap() > 0);
        assert_eq!(result.content_text.as_ref().unwrap(), content);
        assert!(result.conversion_command.is_none());
    }

    #[tokio::test]
    async fn test_empty_file_processing() {
        let processor = TextProcessor::new();
        let file = create_test_file("");

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::DirectText);
        assert_eq!(result.line_count, Some(0));
        assert_eq!(result.word_count, Some(0));
        assert_eq!(result.token_count, Some(0));
        assert_eq!(result.content_text.as_ref().unwrap(), "");
    }

    #[tokio::test]
    async fn test_multiline_text_processing() {
        let processor = TextProcessor::new();
        let content = "Line 1\nLine 2 with more words\nLine 3\n\nLine 5 after empty line";
        let file = create_test_file(content);

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::DirectText);
        assert_eq!(result.line_count, Some(5));
        assert_eq!(result.word_count, Some(14)); // Count all words split by whitespace
        assert!(result.token_count.unwrap() >= result.word_count.unwrap());
    }

    #[tokio::test]
    async fn test_file_size_limit() {
        let mut config = TextProcessorConfig::default();
        config.max_file_size_bytes = 10; // Very small limit
        let processor = TextProcessor::with_config(config);

        let content = "This content is longer than 10 bytes";
        let file = create_test_file(content);

        let result = processor.process(file.path()).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::FileTooLarge { .. } => {
                // Expected error
            }
            other => panic!("Expected FileTooLarge error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_line_count_limit() {
        let mut config = TextProcessorConfig::default();
        config.max_lines = 2; // Very small limit
        let processor = TextProcessor::with_config(config);

        let content = "Line 1\nLine 2\nLine 3\nLine 4";
        let file = create_test_file(content);

        let result = processor.process(file.path()).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::ContentAnalysisFailed { .. } => {
                // Expected error
            }
            other => panic!("Expected ContentAnalysisFailed error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_can_process_text_files() {
        let processor = TextProcessor::new();

        // Create temporary files with different extensions
        let temp_dir = TempDir::new().unwrap();
        
        let rust_file = temp_dir.path().join("test.rs");
        std::fs::write(&rust_file, "fn main() {}").unwrap();
        
        let python_file = temp_dir.path().join("test.py");
        std::fs::write(&python_file, "print('hello')").unwrap();
        
        let image_file = temp_dir.path().join("test.jpg");
        std::fs::write(&image_file, b"\xFF\xD8\xFF").unwrap(); // JPEG header

        assert!(processor.can_process(&rust_file));
        assert!(processor.can_process(&python_file));
        assert!(!processor.can_process(&image_file));
    }

    #[tokio::test]
    async fn test_gitignore_integration() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create .gitignore file
        let gitignore_path = temp_dir.path().join(".gitignore");
        std::fs::write(&gitignore_path, "*.log\ntarget/\n").unwrap();
        
        // Create test files
        let normal_file = temp_dir.path().join("src.rs");
        std::fs::write(&normal_file, "fn main() {}").unwrap();
        
        let ignored_file = temp_dir.path().join("debug.log");
        std::fs::write(&ignored_file, "log content").unwrap();

        let processor = TextProcessor::new()
            .with_gitignore(temp_dir.path())
            .unwrap();

        assert!(processor.can_process(&normal_file));
        assert!(!processor.can_process(&ignored_file));
    }

    #[tokio::test]
    async fn test_encoding_detection() {
        let processor = TextProcessor::new();
        
        // Test UTF-8 content
        let utf8_content = "Hello, 世界! 🦀";
        let file = create_test_file(utf8_content);

        let result = processor.process(file.path()).await.unwrap();
        assert_eq!(result.content_text.as_ref().unwrap(), utf8_content);
    }

    #[tokio::test]
    async fn test_word_and_token_counting() {
        let processor = TextProcessor::new();
        
        let test_cases = [
            ("hello world", 2, 2), // Simple case
            ("hello,world", 1, 1),  // Punctuation attached
            ("hello, world!", 2, 2), // Punctuation separated
            ("fn main() { println!(\"hello\"); }", 5, 6), // Code with punctuation
            ("", 0, 0), // Empty
            ("   \n  \t  ", 0, 0), // Whitespace only
        ];

        for (content, expected_words, expected_min_tokens) in test_cases {
            let file = create_test_file(content);
            let result = processor.process(file.path()).await.unwrap();
            
            assert_eq!(
                result.word_count.unwrap(),
                expected_words,
                "Word count mismatch for content: '{}'",
                content
            );
            
            assert!(
                result.token_count.unwrap() >= expected_min_tokens,
                "Token count {} should be >= {} for content: '{}'",
                result.token_count.unwrap(),
                expected_min_tokens,
                content
            );
        }
    }

    #[tokio::test]
    async fn test_file_metadata_extraction() {
        let processor = TextProcessor::new();
        let content = "fn main() {}";
        let file = create_test_file(content);

        let result = processor.process(file.path()).await.unwrap();

        // Check that all metadata fields are populated
        assert!(!result.filepath.is_empty());
        assert!(!result.filename.is_empty());
        assert!(!result.absolute_path.is_empty());
        assert!(!result.relative_path.is_empty());
        assert!(result.file_size_bytes > 0);
        
        // Check that the filename matches the actual file
        let actual_filename = file.path().file_name().unwrap().to_string_lossy();
        assert_eq!(result.filename, actual_filename);
    }

    #[tokio::test]
    async fn test_large_file_streaming() {
        let mut config = TextProcessorConfig::default();
        config.buffer_size = 64; // Small buffer to trigger streaming
        let processor = TextProcessor::with_config(config);

        // Create content larger than buffer size
        let line = "This is a test line with some content.\n";
        let content = line.repeat(10); // Should be > 64 bytes
        let file = create_test_file(&content);

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::DirectText);
        assert_eq!(result.line_count, Some(10));
        assert!(result.word_count.unwrap() > 0);
        assert_eq!(result.content_text.as_ref().unwrap(), &content);
    }

    #[test]
    fn test_processor_configuration() {
        let config = TextProcessorConfig {
            max_file_size_bytes: 1024,
            max_lines: 100,
            respect_gitignore: false,
            buffer_size: 512,
            auto_detect_encoding: false,
        };

        let processor = TextProcessor::with_config(config.clone());
        assert_eq!(processor.config.max_file_size_bytes, 1024);
        assert_eq!(processor.config.max_lines, 100);
        assert!(!processor.config.respect_gitignore);
        assert_eq!(processor.config.buffer_size, 512);
        assert!(!processor.config.auto_detect_encoding);
    }

    #[test]
    fn test_get_file_type() {
        let processor = TextProcessor::new();
        assert_eq!(processor.get_file_type(), FileType::DirectText);
    }
}