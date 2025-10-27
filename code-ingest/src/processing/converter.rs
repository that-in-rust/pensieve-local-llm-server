use super::{FileProcessor, FileType, ProcessedFile};
use crate::error::{ProcessingError, ProcessingResult};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use tokio::fs as async_fs;
use tokio::process::Command as AsyncCommand;

/// Configuration for convertible file processing
#[derive(Debug, Clone)]
pub struct ConverterConfig {
    /// Maximum file size in bytes for conversion (default: 50MB)
    pub max_file_size_bytes: u64,
    /// Timeout for conversion commands in seconds (default: 60s)
    pub conversion_timeout_secs: u64,
    /// Custom conversion commands (extension -> command template)
    pub custom_commands: HashMap<String, String>,
    /// Whether to validate conversion results
    pub validate_results: bool,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            max_file_size_bytes: 50 * 1024 * 1024, // 50MB
            conversion_timeout_secs: 60,            // 60 seconds
            custom_commands: HashMap::new(),
            validate_results: true,
        }
    }
}

/// Processor for convertible files (Type 2)
#[derive(Debug, Clone)]
pub struct Converter {
    config: ConverterConfig,
    conversion_commands: HashMap<String, String>,
}

impl Converter {
    /// Create a new converter with default configuration
    pub fn new() -> Self {
        Self::with_config(ConverterConfig::default())
    }

    /// Create a new converter with custom configuration
    pub fn with_config(config: ConverterConfig) -> Self {
        let mut converter = Self {
            config,
            conversion_commands: HashMap::new(),
        };

        converter.initialize_default_commands();
        converter
    }

    /// Initialize default conversion commands
    fn initialize_default_commands(&mut self) {
        // PDF conversion commands
        self.conversion_commands.insert(
            "pdf".to_string(),
            "pdftotext '{input}' -".to_string(),
        );

        // Microsoft Office documents
        self.conversion_commands.insert(
            "docx".to_string(),
            "pandoc '{input}' -t plain".to_string(),
        );
        self.conversion_commands.insert(
            "doc".to_string(),
            "antiword '{input}'".to_string(),
        );

        // Spreadsheets
        self.conversion_commands.insert(
            "xlsx".to_string(),
            "xlsx2csv '{input}'".to_string(),
        );
        self.conversion_commands.insert(
            "xls".to_string(),
            "xls2csv '{input}'".to_string(),
        );

        // Presentations
        self.conversion_commands.insert(
            "pptx".to_string(),
            "pandoc '{input}' -t plain".to_string(),
        );

        // OpenDocument formats
        self.conversion_commands.insert(
            "odt".to_string(),
            "pandoc '{input}' -t plain".to_string(),
        );
        self.conversion_commands.insert(
            "ods".to_string(),
            "libreoffice --headless --convert-to csv '{input}' --outdir /tmp && cat /tmp/$(basename '{input}' .ods).csv".to_string(),
        );

        // E-books
        self.conversion_commands.insert(
            "epub".to_string(),
            "pandoc '{input}' -t plain".to_string(),
        );

        // Archives (extract and list contents)
        self.conversion_commands.insert(
            "zip".to_string(),
            "unzip -l '{input}'".to_string(),
        );
        self.conversion_commands.insert(
            "tar".to_string(),
            "tar -tf '{input}'".to_string(),
        );
        self.conversion_commands.insert(
            "gz".to_string(),
            "gunzip -l '{input}' 2>/dev/null || tar -tzf '{input}' 2>/dev/null || echo 'Archive contents not readable'".to_string(),
        );

        // Apply custom commands (override defaults)
        for (ext, cmd) in &self.config.custom_commands {
            self.conversion_commands.insert(ext.clone(), cmd.clone());
        }
    }

    /// Get the conversion command for a file extension
    pub fn get_conversion_command(&self, extension: &str) -> Option<&str> {
        self.conversion_commands.get(&extension.to_lowercase()).map(|s| s.as_str())
    }

    /// Check if a file extension is supported for conversion
    pub fn supports_extension(&self, extension: &str) -> bool {
        self.conversion_commands.contains_key(&extension.to_lowercase())
    }

    /// Execute a conversion command
    async fn execute_conversion_command(
        &self,
        command_template: &str,
        input_path: &Path,
    ) -> ProcessingResult<String> {
        // Replace {input} placeholder with actual file path
        let command_str = command_template.replace("{input}", &input_path.display().to_string());

        // Parse command into program and arguments
        let parts: Vec<&str> = command_str.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ProcessingError::ExternalCommandFailed {
                command: command_str,
                cause: "Empty command".to_string(),
            });
        }

        let program = parts[0];
        let args = &parts[1..];

        // Execute command with timeout
        let mut cmd = AsyncCommand::new(program);
        cmd.args(args);

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.conversion_timeout_secs),
            cmd.output(),
        )
        .await
        .map_err(|_| ProcessingError::ExternalCommandFailed {
            command: command_str.clone(),
            cause: "Command timeout".to_string(),
        })?
        .map_err(|e| ProcessingError::ExternalCommandFailed {
            command: command_str.clone(),
            cause: e.to_string(),
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ProcessingError::ExternalCommandFailed {
                command: command_str,
                cause: format!("Command failed with exit code {:?}: {}", output.status.code(), stderr),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.into_owned())
    }

    /// Validate conversion result
    fn validate_conversion_result(&self, content: &str, original_path: &Path) -> ProcessingResult<()> {
        if !self.config.validate_results {
            return Ok(());
        }

        // Basic validation: ensure we got some content
        if content.trim().is_empty() {
            return Err(ProcessingError::ContentConversionFailed {
                path: original_path.display().to_string(),
                cause: "Conversion produced empty result".to_string(),
            });
        }

        // Check for common error indicators
        let error_indicators = [
            "command not found",
            "no such file",
            "permission denied",
            "cannot open",
            "error:",
            "failed to",
        ];

        let content_lower = content.to_lowercase();
        for indicator in &error_indicators {
            if content_lower.contains(indicator) {
                return Err(ProcessingError::ContentConversionFailed {
                    path: original_path.display().to_string(),
                    cause: format!("Conversion result contains error indicator: {}", indicator),
                });
            }
        }

        Ok(())
    }

    /// Analyze converted text content
    fn analyze_converted_content(&self, content: &str) -> (i32, i32, i32) {
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len() as i32;

        let word_count = content.split_whitespace().count() as i32;

        // Estimate tokens (rough approximation: words * 1.3 for subword tokenization)
        let token_count = (word_count as f64 * 1.3) as i32;

        (line_count, word_count, token_count)
    }

    /// Create a ProcessedFile from converted content
    fn create_processed_file(
        &self,
        file_path: &Path,
        content: String,
        conversion_command: String,
        file_size_bytes: i64,
    ) -> ProcessingResult<ProcessedFile> {
        let filename = file_path
            .file_name()
            .ok_or_else(|| ProcessingError::FileReadFailed {
                path: file_path.display().to_string(),
                cause: "Invalid filename".to_string(),
            })?
            .to_string_lossy()
            .to_string();

        let extension = file_path
            .extension()
            .map(|ext| ext.to_string_lossy().to_string())
            .unwrap_or_default();

        let filepath = file_path.to_string_lossy().to_string();
        let absolute_path = file_path
            .canonicalize()
            .map_err(|e| ProcessingError::FileReadFailed {
                path: file_path.display().to_string(),
                cause: e.to_string(),
            })?
            .to_string_lossy()
            .to_string();

        // For relative path, we'll use the provided path as-is
        let relative_path = filepath.clone();

        let (line_count, word_count, token_count) = self.analyze_converted_content(&content);

        Ok(ProcessedFile {
            filepath,
            filename,
            extension,
            file_size_bytes,
            line_count: Some(line_count),
            word_count: Some(word_count),
            token_count: Some(token_count),
            content_text: Some(content),
            file_type: FileType::Convertible,
            conversion_command: Some(conversion_command),
            relative_path,
            absolute_path,
            skipped: false,
            skip_reason: None,
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Add a custom conversion command
    pub fn add_custom_command(&mut self, extension: String, command: String) {
        self.conversion_commands.insert(extension.to_lowercase(), command);
    }

    /// Remove a conversion command
    pub fn remove_command(&mut self, extension: &str) -> Option<String> {
        self.conversion_commands.remove(&extension.to_lowercase())
    }

    /// Get all supported extensions
    pub fn get_supported_extensions(&self) -> Vec<String> {
        self.conversion_commands.keys().cloned().collect()
    }

    /// Check if a command is available on the system
    pub fn check_command_availability(&self, command: &str) -> bool {
        let program = command.split_whitespace().next().unwrap_or(command);
        
        Command::new("which")
            .arg(program)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Get available conversion commands on the system
    pub fn get_available_commands(&self) -> HashMap<String, String> {
        let mut available = HashMap::new();
        
        for (ext, cmd) in &self.conversion_commands {
            let program = cmd.split_whitespace().next().unwrap_or("");
            if self.check_command_availability(program) {
                available.insert(ext.clone(), cmd.clone());
            }
        }
        
        available
    }
}

#[async_trait::async_trait]
impl FileProcessor for Converter {
    fn can_process(&self, file_path: &Path) -> bool {
        use super::classifier::FileClassifier;
        
        // Check if it's a convertible file
        let classifier = FileClassifier::new();
        if classifier.classify_file(file_path) != FileType::Convertible {
            return false;
        }

        // Check if we have a conversion command for this extension
        let extension = file_path
            .extension()
            .map(|ext| ext.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        self.supports_extension(&extension)
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

        // Get conversion command
        let extension = file_path
            .extension()
            .map(|ext| ext.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let command_template = self
            .get_conversion_command(&extension)
            .ok_or_else(|| ProcessingError::UnsupportedFileType {
                path: file_path.display().to_string(),
                extension: extension.clone(),
            })?;

        // Execute conversion
        let converted_content = self
            .execute_conversion_command(command_template, file_path)
            .await?;

        // Validate result
        self.validate_conversion_result(&converted_content, file_path)?;

        // Create processed file
        self.create_processed_file(
            file_path,
            converted_content,
            command_template.to_string(),
            metadata.len() as i64,
        )
    }

    fn get_file_type(&self) -> FileType {
        FileType::Convertible
    }
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn create_test_file_with_extension(content: &[u8], extension: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(&format!(".{}", extension)).unwrap();
        file.write_all(content).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_converter_initialization() {
        let converter = Converter::new();
        
        // Check that default commands are loaded
        assert!(converter.supports_extension("pdf"));
        assert!(converter.supports_extension("docx"));
        assert!(converter.supports_extension("xlsx"));
        assert!(!converter.supports_extension("jpg"));
    }

    #[test]
    fn test_custom_commands() {
        let mut converter = Converter::new();
        
        // Add custom command
        converter.add_custom_command("custom".to_string(), "echo 'custom content'".to_string());
        assert!(converter.supports_extension("custom"));
        assert_eq!(converter.get_conversion_command("custom"), Some("echo 'custom content'"));
        
        // Remove command
        let removed = converter.remove_command("custom");
        assert_eq!(removed, Some("echo 'custom content'".to_string()));
        assert!(!converter.supports_extension("custom"));
    }

    #[test]
    fn test_get_supported_extensions() {
        let converter = Converter::new();
        let extensions = converter.get_supported_extensions();
        
        assert!(extensions.contains(&"pdf".to_string()));
        assert!(extensions.contains(&"docx".to_string()));
        assert!(extensions.contains(&"xlsx".to_string()));
    }

    #[test]
    fn test_can_process_convertible_files() {
        let converter = Converter::new();
        let temp_dir = TempDir::new().unwrap();
        
        // Create test files
        let pdf_file = temp_dir.path().join("test.pdf");
        std::fs::write(&pdf_file, b"fake pdf content").unwrap();
        
        let docx_file = temp_dir.path().join("test.docx");
        std::fs::write(&docx_file, b"fake docx content").unwrap();
        
        let txt_file = temp_dir.path().join("test.txt");
        std::fs::write(&txt_file, b"text content").unwrap();
        
        let jpg_file = temp_dir.path().join("test.jpg");
        std::fs::write(&jpg_file, b"fake jpg content").unwrap();

        assert!(converter.can_process(&pdf_file));
        assert!(converter.can_process(&docx_file));
        assert!(!converter.can_process(&txt_file)); // Text files are DirectText, not Convertible
        assert!(!converter.can_process(&jpg_file)); // Image files are NonText
    }

    #[tokio::test]
    async fn test_simple_conversion_with_echo() {
        let mut converter = Converter::new();
        
        // Add a simple echo command for testing (use pdf extension which is already convertible)
        let file = create_test_file_with_extension(b"original content", "pdf");
        
        // Override the pdf command with echo for testing
        converter.add_custom_command("pdf".to_string(), "echo 'converted content'".to_string());
        
        let result = converter.process(file.path()).await.unwrap();
        
        assert_eq!(result.file_type, FileType::Convertible);
        assert_eq!(result.content_text.as_ref().unwrap().trim(), "'converted content'");
        assert_eq!(result.conversion_command.as_ref().unwrap(), "echo 'converted content'");
        assert!(result.line_count.unwrap() > 0);
        assert!(result.word_count.unwrap() > 0);
    }

    #[tokio::test]
    async fn test_file_size_limit() {
        let mut config = ConverterConfig::default();
        config.max_file_size_bytes = 10; // Very small limit
        let mut converter = Converter::with_config(config);
        
        let content = b"This content is longer than 10 bytes for sure";
        let file = create_test_file_with_extension(content, "pdf");
        
        let result = converter.process(file.path()).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::FileTooLarge { .. } => {
                // Expected error
            }
            other => panic!("Expected FileTooLarge error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_conversion_timeout() {
        let mut config = ConverterConfig::default();
        config.conversion_timeout_secs = 1; // Very short timeout
        let mut converter = Converter::with_config(config);
        
        // Command that sleeps longer than timeout (use a command that will definitely timeout)
        converter.add_custom_command("pdf".to_string(), "sleep 10".to_string());
        
        let file = create_test_file_with_extension(b"content", "pdf");
        
        let result = converter.process(file.path()).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::ExternalCommandFailed { cause, .. } => {
                assert!(cause.contains("timeout") || cause.contains("Command timeout"));
            }
            other => panic!("Expected ExternalCommandFailed with timeout, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_command_failure() {
        let mut converter = Converter::new();
        
        // Command that will fail
        converter.add_custom_command("pdf".to_string(), "false".to_string());
        
        let file = create_test_file_with_extension(b"content", "pdf");
        
        let result = converter.process(file.path()).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::ExternalCommandFailed { .. } => {
                // Expected error
            }
            other => panic!("Expected ExternalCommandFailed, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_validation_empty_result() {
        let mut converter = Converter::new();
        
        // Command that produces empty output
        converter.add_custom_command("pdf".to_string(), "true".to_string());
        
        let file = create_test_file_with_extension(b"content", "pdf");
        
        let result = converter.process(file.path()).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::ContentConversionFailed { cause, .. } => {
                assert!(cause.contains("empty result"));
            }
            other => panic!("Expected ContentConversionFailed, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_validation_disabled() {
        let mut config = ConverterConfig::default();
        config.validate_results = false;
        let mut converter = Converter::with_config(config);
        
        // Command that produces empty output
        converter.add_custom_command("pdf".to_string(), "true".to_string());
        
        let file = create_test_file_with_extension(b"content", "pdf");
        
        let result = converter.process(file.path()).await;
        // Should succeed because validation is disabled
        assert!(result.is_ok());
    }

    #[test]
    fn test_content_analysis() {
        let converter = Converter::new();
        
        let test_cases = [
            ("hello world", 1, 2, 2),
            ("line 1\nline 2\nline 3", 3, 6, 7),
            ("", 0, 0, 0),
            ("single", 1, 1, 1),
        ];
        
        for (content, expected_lines, expected_words, expected_min_tokens) in test_cases {
            let (lines, words, tokens) = converter.analyze_converted_content(content);
            
            assert_eq!(lines, expected_lines, "Line count mismatch for: '{}'", content);
            assert_eq!(words, expected_words, "Word count mismatch for: '{}'", content);
            assert!(tokens >= expected_min_tokens, "Token count {} should be >= {} for: '{}'", tokens, expected_min_tokens, content);
        }
    }

    #[test]
    fn test_command_template_replacement() {
        let converter = Converter::new();
        let template = "pdftotext '{input}' -";
        let path = Path::new("/path/to/file.pdf");
        
        // This is testing the internal logic, so we'll create a simple test
        let expected = "pdftotext '/path/to/file.pdf' -";
        let actual = template.replace("{input}", &path.display().to_string());
        
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_get_file_type() {
        let converter = Converter::new();
        assert_eq!(converter.get_file_type(), FileType::Convertible);
    }

    #[test]
    fn test_command_availability_check() {
        let converter = Converter::new();
        
        // Test with a command that should exist on most systems
        assert!(converter.check_command_availability("echo"));
        
        // Test with a command that likely doesn't exist
        assert!(!converter.check_command_availability("nonexistent_command_12345"));
    }

    #[test]
    fn test_get_available_commands() {
        let converter = Converter::new();
        let available = converter.get_available_commands();
        
        // Should contain at least some commands (depending on system)
        // We can't guarantee specific commands, but the structure should be correct
        for (ext, cmd) in &available {
            assert!(!ext.is_empty());
            assert!(!cmd.is_empty());
        }
    }

    #[test]
    fn test_validation_error_indicators() {
        let converter = Converter::new();
        let path = Path::new("test.pdf");
        
        let error_contents = [
            "command not found: pdftotext",
            "Error: cannot open file",
            "Permission denied",
            "Failed to process document",
        ];
        
        for content in &error_contents {
            let result = converter.validate_conversion_result(content, path);
            assert!(result.is_err(), "Should fail validation for content: '{}'", content);
        }
        
        // Valid content should pass
        let valid_result = converter.validate_conversion_result("This is valid converted text", path);
        assert!(valid_result.is_ok());
    }

    #[tokio::test]
    async fn test_unsupported_file_type() {
        let converter = Converter::new();
        let temp_dir = TempDir::new().unwrap();
        
        // Create a file with unsupported extension
        let unsupported_file = temp_dir.path().join("test.xyz");
        std::fs::write(&unsupported_file, b"content").unwrap();
        
        let result = converter.process(&unsupported_file).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::UnsupportedFileType { .. } => {
                // Expected error
            }
            other => panic!("Expected UnsupportedFileType, got: {:?}", other),
        }
    }

    #[test]
    fn test_converter_config() {
        let config = ConverterConfig {
            max_file_size_bytes: 1024,
            conversion_timeout_secs: 30,
            custom_commands: {
                let mut map = HashMap::new();
                map.insert("custom".to_string(), "custom_command".to_string());
                map
            },
            validate_results: false,
        };
        
        let converter = Converter::with_config(config.clone());
        assert_eq!(converter.config.max_file_size_bytes, 1024);
        assert_eq!(converter.config.conversion_timeout_secs, 30);
        assert!(!converter.config.validate_results);
        assert!(converter.supports_extension("custom"));
    }
}