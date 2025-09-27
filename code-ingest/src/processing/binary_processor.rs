use super::{FileProcessor, FileType, ProcessedFile};
use crate::error::{ProcessingError, ProcessingResult};
use std::path::Path;
use tokio::fs as async_fs;

/// Configuration for binary file processing
#[derive(Debug, Clone)]
pub struct BinaryProcessorConfig {
    /// Maximum file size in bytes to process metadata (default: 1GB)
    pub max_file_size_bytes: u64,
    /// Whether to include file hash for integrity checking
    pub include_file_hash: bool,
}

impl Default for BinaryProcessorConfig {
    fn default() -> Self {
        Self {
            max_file_size_bytes: 1024 * 1024 * 1024, // 1GB
            include_file_hash: false, // Disabled by default for performance
        }
    }
}

/// Processor for binary files (Type 3) - metadata only
#[derive(Debug, Clone)]
pub struct BinaryProcessor {
    config: BinaryProcessorConfig,
}

impl BinaryProcessor {
    /// Create a new binary processor with default configuration
    pub fn new() -> Self {
        Self::with_config(BinaryProcessorConfig::default())
    }

    /// Create a new binary processor with custom configuration
    pub fn with_config(config: BinaryProcessorConfig) -> Self {
        Self { config }
    }

    /// Calculate file hash if enabled
    async fn calculate_file_hash<P: AsRef<Path>>(&self, file_path: P) -> ProcessingResult<Option<String>> {
        if !self.config.include_file_hash {
            return Ok(None);
        }

        use sha2::{Sha256, Digest};
        
        let content = async_fs::read(file_path.as_ref()).await.map_err(|e| {
            ProcessingError::FileReadFailed {
                path: file_path.as_ref().display().to_string(),
                cause: e.to_string(),
            }
        })?;

        let mut hasher = Sha256::new();
        hasher.update(&content);
        let hash = hasher.finalize();
        
        Ok(Some(format!("{:x}", hash)))
    }

    /// Create a ProcessedFile with metadata only
    fn create_processed_file<P: AsRef<Path>>(
        &self,
        file_path: P,
        file_size_bytes: i64,
        file_hash: Option<String>,
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
        let relative_path = filepath.clone();

        // Create skip reason for binary files
        let skip_reason = Some(format!(
            "Binary file ({}): content not extracted, metadata only",
            extension.to_uppercase()
        ));

        Ok(ProcessedFile {
            filepath,
            filename,
            extension,
            file_size_bytes,
            line_count: None,
            word_count: None,
            token_count: None,
            content_text: file_hash, // Store hash in content_text if available
            file_type: FileType::NonText,
            conversion_command: None,
            relative_path,
            absolute_path,
            skipped: true, // Mark as skipped since we don't extract content
            skip_reason,
        })
    }
}

#[async_trait::async_trait]
impl FileProcessor for BinaryProcessor {
    fn can_process(&self, file_path: &Path) -> bool {
        use super::classifier::FileClassifier;
        
        // Check if it's a binary file
        let classifier = FileClassifier::new();
        classifier.classify_file(file_path) == FileType::NonText
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

        // Check file size
        let metadata = async_fs::metadata(file_path).await.map_err(|e| {
            ProcessingError::FileReadFailed {
                path: file_path.display().to_string(),
                cause: e.to_string(),
            }
        })?;

        if metadata.len() > self.config.max_file_size_bytes {
            return Err(ProcessingError::FileTooLarge {
                path: file_path.display().to_string(),
                size_mb: metadata.len() / (1024 * 1024),
            });
        }

        // Calculate hash if enabled
        let file_hash = self.calculate_file_hash(file_path).await?;

        // Create processed file with metadata only
        self.create_processed_file(file_path, metadata.len() as i64, file_hash)
    }

    fn get_file_type(&self) -> FileType {
        FileType::NonText
    }
}

impl Default for BinaryProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn create_binary_file(content: &[u8], extension: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(&format!(".{}", extension)).unwrap();
        file.write_all(content).unwrap();
        file.flush().unwrap();
        file
    }

    #[tokio::test]
    async fn test_binary_file_processing() {
        let processor = BinaryProcessor::new();
        
        // Create a fake binary file (JPEG header)
        let jpeg_content = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01fake jpeg content";
        let file = create_binary_file(jpeg_content, "jpg");

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::NonText);
        assert_eq!(result.filename, file.path().file_name().unwrap().to_string_lossy());
        assert_eq!(result.extension, "jpg");
        assert_eq!(result.file_size_bytes, jpeg_content.len() as i64);
        assert!(result.line_count.is_none());
        assert!(result.word_count.is_none());
        assert!(result.token_count.is_none());
        assert!(result.conversion_command.is_none());
        assert!(result.skipped);
        assert!(result.skip_reason.is_some());
        assert!(result.skip_reason.as_ref().unwrap().contains("Binary file"));
    }

    #[tokio::test]
    async fn test_binary_file_with_hash() {
        let mut config = BinaryProcessorConfig::default();
        config.include_file_hash = true;
        let processor = BinaryProcessor::with_config(config);
        
        let content = b"test binary content";
        let file = create_binary_file(content, "bin");

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::NonText);
        assert!(result.content_text.is_some()); // Should contain hash
        assert_eq!(result.content_text.as_ref().unwrap().len(), 64); // SHA256 hex length
    }

    #[tokio::test]
    async fn test_binary_file_without_hash() {
        let processor = BinaryProcessor::new(); // Default config has hash disabled
        
        let content = b"test binary content";
        let file = create_binary_file(content, "bin");

        let result = processor.process(file.path()).await.unwrap();

        assert_eq!(result.file_type, FileType::NonText);
        assert!(result.content_text.is_none()); // Should not contain hash
    }

    #[tokio::test]
    async fn test_file_size_limit() {
        let mut config = BinaryProcessorConfig::default();
        config.max_file_size_bytes = 10; // Very small limit
        let processor = BinaryProcessor::with_config(config);

        let large_content = b"This content is longer than 10 bytes for sure";
        let file = create_binary_file(large_content, "bin");

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
    async fn test_can_process_binary_files() {
        let processor = BinaryProcessor::new();
        let temp_dir = TempDir::new().unwrap();
        
        // Create test files
        let jpg_file = temp_dir.path().join("image.jpg");
        std::fs::write(&jpg_file, b"\xFF\xD8\xFF fake jpeg").unwrap();
        
        let exe_file = temp_dir.path().join("program.exe");
        std::fs::write(&exe_file, b"MZ fake executable").unwrap();
        
        let txt_file = temp_dir.path().join("text.txt");
        std::fs::write(&txt_file, b"text content").unwrap();
        
        let pdf_file = temp_dir.path().join("doc.pdf");
        std::fs::write(&pdf_file, b"%PDF fake pdf").unwrap();

        assert!(processor.can_process(&jpg_file));
        assert!(processor.can_process(&exe_file));
        assert!(!processor.can_process(&txt_file)); // Text files are DirectText
        assert!(!processor.can_process(&pdf_file)); // PDF files are Convertible
    }

    #[tokio::test]
    async fn test_various_binary_extensions() {
        let processor = BinaryProcessor::new();
        
        let binary_extensions = ["jpg", "png", "gif", "mp4", "exe", "dll", "so", "bin"];
        
        for ext in &binary_extensions {
            let content = format!("fake {} content", ext).into_bytes();
            let file = create_binary_file(&content, ext);
            
            assert!(processor.can_process(file.path()), "Should process .{} files", ext);
            
            let result = processor.process(file.path()).await.unwrap();
            assert_eq!(result.file_type, FileType::NonText);
            assert_eq!(result.extension, *ext);
            assert!(result.skipped);
        }
    }

    #[tokio::test]
    async fn test_unsupported_file_type() {
        let processor = BinaryProcessor::new();
        let temp_dir = TempDir::new().unwrap();
        
        // Create a text file (should not be processed by binary processor)
        let text_file = temp_dir.path().join("test.rs");
        std::fs::write(&text_file, "fn main() {}").unwrap();
        
        let result = processor.process(&text_file).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::UnsupportedFileType { .. } => {
                // Expected error
            }
            other => panic!("Expected UnsupportedFileType, got: {:?}", other),
        }
    }

    #[test]
    fn test_get_file_type() {
        let processor = BinaryProcessor::new();
        assert_eq!(processor.get_file_type(), FileType::NonText);
    }

    #[test]
    fn test_processor_configuration() {
        let config = BinaryProcessorConfig {
            max_file_size_bytes: 1024,
            include_file_hash: true,
        };

        let processor = BinaryProcessor::with_config(config.clone());
        assert_eq!(processor.config.max_file_size_bytes, 1024);
        assert!(processor.config.include_file_hash);
    }

    #[tokio::test]
    async fn test_file_metadata_extraction() {
        let processor = BinaryProcessor::new();
        let content = b"binary content";
        let file = create_binary_file(content, "bin");

        let result = processor.process(file.path()).await.unwrap();

        // Check that all metadata fields are populated
        assert!(!result.filepath.is_empty());
        assert!(!result.filename.is_empty());
        assert!(!result.absolute_path.is_empty());
        assert!(!result.relative_path.is_empty());
        assert_eq!(result.file_size_bytes, content.len() as i64);
        
        // Check that the filename matches the actual file
        let actual_filename = file.path().file_name().unwrap().to_string_lossy();
        assert_eq!(result.filename, actual_filename);
    }
}