use super::{
    classifier::FileClassifier,
    text_processor::{TextProcessor, TextProcessorConfig},
    converter::{Converter, ConverterConfig},
    binary_processor::{BinaryProcessor, BinaryProcessorConfig},
    FileProcessor, FileType, ProcessedFile,
};
use crate::error::{ProcessingError, ProcessingResult};
use std::path::Path;


/// Configuration for the content extraction pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Configuration for text processing
    pub text_config: TextProcessorConfig,
    /// Configuration for file conversion
    pub converter_config: ConverterConfig,
    /// Configuration for binary file processing
    pub binary_config: BinaryProcessorConfig,
    /// Whether to respect .gitignore patterns
    pub respect_gitignore: bool,
    /// Repository root path for gitignore resolution
    pub repo_root: Option<std::path::PathBuf>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            text_config: TextProcessorConfig::default(),
            converter_config: ConverterConfig::default(),
            binary_config: BinaryProcessorConfig::default(),
            respect_gitignore: true,
            repo_root: None,
        }
    }
}

/// Content extraction pipeline that coordinates all file processors
#[derive(Debug)]
pub struct ContentExtractionPipeline {
    classifier: FileClassifier,
    text_processor: TextProcessor,
    converter: Converter,
    binary_processor: BinaryProcessor,
    config: PipelineConfig,
}

impl ContentExtractionPipeline {
    /// Create a new content extraction pipeline with default configuration
    pub fn new() -> ProcessingResult<Self> {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a new content extraction pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> ProcessingResult<Self> {
        let classifier = FileClassifier::new();
        
        // Set up text processor with gitignore if configured
        let text_processor = if config.respect_gitignore {
            if let Some(ref repo_root) = config.repo_root {
                TextProcessor::with_config(config.text_config.clone())
                    .with_gitignore(repo_root)?
            } else {
                TextProcessor::with_config(config.text_config.clone())
            }
        } else {
            TextProcessor::with_config(config.text_config.clone())
        };

        let converter = Converter::with_config(config.converter_config.clone());
        let binary_processor = BinaryProcessor::with_config(config.binary_config.clone());

        Ok(Self {
            classifier,
            text_processor,
            converter,
            binary_processor,
            config,
        })
    }

    /// Set the repository root for gitignore processing
    pub fn with_repo_root<P: AsRef<Path>>(mut self, repo_root: P) -> ProcessingResult<Self> {
        let repo_root = repo_root.as_ref().to_path_buf();
        self.config.repo_root = Some(repo_root.clone());

        // Recreate text processor with gitignore
        if self.config.respect_gitignore {
            self.text_processor = TextProcessor::with_config(self.config.text_config.clone())
                .with_gitignore(&repo_root)?;
        }

        Ok(self)
    }

    /// Classify a file and return its type
    pub fn classify_file<P: AsRef<Path>>(&self, file_path: P) -> FileType {
        self.classifier.classify_file(file_path)
    }

    /// Check if a file should be processed (not ignored by gitignore)
    pub fn should_process<P: AsRef<Path>>(&self, file_path: P) -> bool {
        let file_type = self.classify_file(&file_path);
        
        match file_type {
            FileType::DirectText => self.text_processor.can_process(file_path.as_ref()),
            FileType::Convertible => self.converter.can_process(file_path.as_ref()),
            FileType::NonText => self.binary_processor.can_process(file_path.as_ref()),
        }
    }

    /// Process a single file through the appropriate processor
    pub async fn process_file<P: AsRef<Path>>(&self, file_path: P) -> ProcessingResult<ProcessedFile> {
        let path = file_path.as_ref();
        let file_type = self.classify_file(path);

        match file_type {
            FileType::DirectText => {
                self.text_processor.process(path).await
            }
            FileType::Convertible => {
                self.converter.process(path).await
            }
            FileType::NonText => {
                self.binary_processor.process(path).await
            }
        }
    }

    /// Process multiple files in parallel
    pub async fn process_files<P: AsRef<Path> + Send + Sync + Clone + 'static>(
        &self,
        file_paths: Vec<P>,
    ) -> Vec<ProcessingResult<ProcessedFile>> {
        use tokio::task::JoinSet;

        let mut tasks = JoinSet::new();
        
        // Clone the processors for parallel use
        let classifier = self.classifier.clone();
        let text_processor = self.text_processor.clone();
        let converter = self.converter.clone();
        let binary_processor = self.binary_processor.clone();

        for file_path in file_paths {
            let classifier_clone = classifier.clone();
            let text_processor_clone = text_processor.clone();
            let converter_clone = converter.clone();
            let binary_processor_clone = binary_processor.clone();
            
            tasks.spawn(async move {
                let path = file_path.as_ref();
                let file_type = classifier_clone.classify_file(path);

                match file_type {
                    FileType::DirectText => text_processor_clone.process(path).await,
                    FileType::Convertible => converter_clone.process(path).await,
                    FileType::NonText => binary_processor_clone.process(path).await,
                }
            });
        }

        let mut results = Vec::new();
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(processing_result) => results.push(processing_result),
                Err(join_error) => {
                    results.push(Err(ProcessingError::ContentAnalysisFailed {
                        path: "unknown".to_string(),
                        cause: format!("Task join error: {}", join_error),
                    }));
                }
            }
        }

        results
    }

    /// Get processing statistics for a set of files
    pub fn get_processing_stats<P: AsRef<Path>>(&self, file_paths: &[P]) -> ProcessingStats {
        let mut stats = ProcessingStats::default();

        for file_path in file_paths {
            let file_type = self.classify_file(file_path);
            match file_type {
                FileType::DirectText => stats.direct_text_count += 1,
                FileType::Convertible => stats.convertible_count += 1,
                FileType::NonText => stats.binary_count += 1,
            }

            if self.should_process(file_path) {
                stats.processable_count += 1;
            } else {
                stats.ignored_count += 1;
            }
        }

        stats.total_count = file_paths.len();
        stats
    }

    /// Get available conversion commands on the system
    pub fn get_available_conversion_commands(&self) -> std::collections::HashMap<String, String> {
        self.converter.get_available_commands()
    }

    /// Check if external conversion tools are available
    pub fn check_conversion_tools(&self) -> ConversionToolStatus {
        let available_commands = self.get_available_conversion_commands();
        
        ConversionToolStatus {
            pdftotext_available: available_commands.contains_key("pdf"),
            pandoc_available: available_commands.iter()
                .any(|(_, cmd)| cmd.contains("pandoc")),
            libreoffice_available: available_commands.iter()
                .any(|(_, cmd)| cmd.contains("libreoffice")),
            total_supported: self.converter.get_supported_extensions().len(),
            total_available: available_commands.len(),
        }
    }
}

impl Default for ContentExtractionPipeline {
    fn default() -> Self {
        Self::new().expect("Failed to create default pipeline")
    }
}

/// Statistics about file processing
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    pub total_count: usize,
    pub direct_text_count: usize,
    pub convertible_count: usize,
    pub binary_count: usize,
    pub processable_count: usize,
    pub ignored_count: usize,
}

impl ProcessingStats {
    /// Calculate the percentage of files that are processable
    pub fn processable_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.processable_count as f64 / self.total_count as f64) * 100.0
        }
    }

    /// Calculate the percentage of files that contain extractable text
    pub fn text_extractable_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            let text_extractable = self.direct_text_count + self.convertible_count;
            (text_extractable as f64 / self.total_count as f64) * 100.0
        }
    }
}

/// Status of external conversion tools
#[derive(Debug, Clone)]
pub struct ConversionToolStatus {
    pub pdftotext_available: bool,
    pub pandoc_available: bool,
    pub libreoffice_available: bool,
    pub total_supported: usize,
    pub total_available: usize,
}

impl ConversionToolStatus {
    /// Check if all essential tools are available
    pub fn all_essential_available(&self) -> bool {
        self.pdftotext_available && self.pandoc_available
    }

    /// Get availability percentage
    pub fn availability_percentage(&self) -> f64 {
        if self.total_supported == 0 {
            100.0
        } else {
            (self.total_available as f64 / self.total_supported as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;


    async fn create_test_files(temp_dir: &TempDir) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();

        // Create DirectText files
        let rust_file = temp_dir.path().join("main.rs");
        std::fs::write(&rust_file, "fn main() { println!(\"Hello!\"); }").unwrap();
        files.push(rust_file);

        let python_file = temp_dir.path().join("script.py");
        std::fs::write(&python_file, "print('Hello, World!')").unwrap();
        files.push(python_file);

        let json_file = temp_dir.path().join("config.json");
        std::fs::write(&json_file, r#"{"key": "value"}"#).unwrap();
        files.push(json_file);

        // Create Convertible files
        let pdf_file = temp_dir.path().join("document.pdf");
        std::fs::write(&pdf_file, b"%PDF-1.4 fake pdf content").unwrap();
        files.push(pdf_file);

        // Create NonText files
        let jpg_file = temp_dir.path().join("image.jpg");
        std::fs::write(&jpg_file, b"\xFF\xD8\xFF fake jpeg").unwrap();
        files.push(jpg_file);

        let exe_file = temp_dir.path().join("program.exe");
        std::fs::write(&exe_file, b"MZ fake executable").unwrap();
        files.push(exe_file);

        files
    }

    #[tokio::test]
    async fn test_pipeline_file_classification() {
        let pipeline = ContentExtractionPipeline::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir).await;

        // Test classification
        assert_eq!(pipeline.classify_file(&files[0]), FileType::DirectText); // main.rs
        assert_eq!(pipeline.classify_file(&files[1]), FileType::DirectText); // script.py
        assert_eq!(pipeline.classify_file(&files[2]), FileType::DirectText); // config.json
        assert_eq!(pipeline.classify_file(&files[3]), FileType::Convertible); // document.pdf
        assert_eq!(pipeline.classify_file(&files[4]), FileType::NonText); // image.jpg
        assert_eq!(pipeline.classify_file(&files[5]), FileType::NonText); // program.exe
    }

    #[tokio::test]
    async fn test_pipeline_single_file_processing() {
        let pipeline = ContentExtractionPipeline::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir).await;

        // Test processing DirectText file
        let result = pipeline.process_file(&files[0]).await.unwrap(); // main.rs
        assert_eq!(result.file_type, FileType::DirectText);
        assert!(result.content_text.is_some());
        assert!(result.line_count.is_some());
        assert!(!result.skipped);

        // Test processing NonText file
        let result = pipeline.process_file(&files[4]).await.unwrap(); // image.jpg
        assert_eq!(result.file_type, FileType::NonText);
        assert!(result.line_count.is_none());
        assert!(result.skipped);
    }

    #[tokio::test]
    async fn test_pipeline_parallel_processing() {
        let pipeline = ContentExtractionPipeline::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir).await;

        // Process all files in parallel
        let results = pipeline.process_files(files.clone()).await;

        assert_eq!(results.len(), files.len());
        
        // Check results - some may fail if conversion tools are not available
        let mut successful_results = Vec::new();
        for result in results {
            match result {
                Ok(processed) => successful_results.push(processed),
                Err(ProcessingError::ExternalCommandFailed { .. }) => {
                    // Expected if conversion tools are not available
                    continue;
                }
                Err(other) => panic!("Unexpected processing error: {:?}", other),
            }
        }

        // Check that we got the expected file types for successful results
        let direct_text_count = successful_results.iter().filter(|p| p.file_type == FileType::DirectText).count();
        let convertible_count = successful_results.iter().filter(|p| p.file_type == FileType::Convertible).count();
        let non_text_count = successful_results.iter().filter(|p| p.file_type == FileType::NonText).count();

        // We should have at least the text files and binary files
        assert!(direct_text_count >= 3); // main.rs, script.py, config.json
        assert!(non_text_count >= 2); // image.jpg, program.exe
        // Convertible count may be 0 if conversion tools are not available
    }

    #[tokio::test]
    async fn test_processing_stats() {
        let pipeline = ContentExtractionPipeline::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir).await;

        let stats = pipeline.get_processing_stats(&files);

        assert_eq!(stats.total_count, 6);
        assert_eq!(stats.direct_text_count, 3);
        assert_eq!(stats.convertible_count, 1);
        assert_eq!(stats.binary_count, 2);
        assert_eq!(stats.processable_count, 6); // All files should be processable
        assert_eq!(stats.ignored_count, 0);

        // Test percentage calculations
        assert!(stats.processable_percentage() > 99.0);
        assert!(stats.text_extractable_percentage() > 65.0); // 4 out of 6 files
    }

    #[tokio::test]
    async fn test_gitignore_integration() {
        let temp_dir = TempDir::new().unwrap();

        // Create .gitignore file
        let gitignore_path = temp_dir.path().join(".gitignore");
        std::fs::write(&gitignore_path, "*.log\n*.tmp\ntarget/\n").unwrap();

        // Create test files
        let normal_file = temp_dir.path().join("main.rs");
        std::fs::write(&normal_file, "fn main() {}").unwrap();

        let ignored_file = temp_dir.path().join("debug.log");
        std::fs::write(&ignored_file, "log content").unwrap();

        let config = PipelineConfig {
            respect_gitignore: true,
            repo_root: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };

        let pipeline = ContentExtractionPipeline::with_config(config).unwrap();

        // Normal file should be processable
        assert!(pipeline.should_process(&normal_file));

        // Ignored file should not be processable
        assert!(!pipeline.should_process(&ignored_file));
    }

    #[test]
    fn test_conversion_tool_status() {
        let pipeline = ContentExtractionPipeline::new().unwrap();
        let status = pipeline.check_conversion_tools();

        // Basic structure checks
        assert!(status.total_supported > 0);
        assert!(status.availability_percentage() >= 0.0);
        assert!(status.availability_percentage() <= 100.0);
    }

    #[test]
    fn test_pipeline_configuration() {
        let mut config = PipelineConfig::default();
        config.text_config.max_file_size_bytes = 1024;
        config.converter_config.max_file_size_bytes = 2048;
        config.binary_config.max_file_size_bytes = 4096;

        let pipeline = ContentExtractionPipeline::with_config(config.clone()).unwrap();
        
        assert_eq!(pipeline.config.text_config.max_file_size_bytes, 1024);
        assert_eq!(pipeline.config.converter_config.max_file_size_bytes, 2048);
        assert_eq!(pipeline.config.binary_config.max_file_size_bytes, 4096);
    }

    #[tokio::test]
    async fn test_should_process_filtering() {
        let pipeline = ContentExtractionPipeline::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir).await;

        // All test files should be processable by default
        for file in &files {
            assert!(pipeline.should_process(file), "File should be processable: {}", file.display());
        }
    }

    #[test]
    fn test_processing_stats_edge_cases() {
        let stats = ProcessingStats::default();
        
        // Test with zero files
        assert_eq!(stats.processable_percentage(), 0.0);
        assert_eq!(stats.text_extractable_percentage(), 0.0);

        // Test with some files
        let stats = ProcessingStats {
            total_count: 10,
            direct_text_count: 6,
            convertible_count: 2,
            binary_count: 2,
            processable_count: 8,
            ignored_count: 2,
        };

        assert_eq!(stats.processable_percentage(), 80.0);
        assert_eq!(stats.text_extractable_percentage(), 80.0); // 6 + 2 = 8 out of 10
    }

    #[test]
    fn test_conversion_tool_status_methods() {
        let status = ConversionToolStatus {
            pdftotext_available: true,
            pandoc_available: true,
            libreoffice_available: false,
            total_supported: 10,
            total_available: 7,
        };

        assert!(status.all_essential_available());
        assert_eq!(status.availability_percentage(), 70.0);

        let status_incomplete = ConversionToolStatus {
            pdftotext_available: false,
            pandoc_available: true,
            libreoffice_available: true,
            total_supported: 10,
            total_available: 5,
        };

        assert!(!status_incomplete.all_essential_available());
        assert_eq!(status_incomplete.availability_percentage(), 50.0);
    }
}