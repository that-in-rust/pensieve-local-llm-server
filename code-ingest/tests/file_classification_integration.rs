use code_ingest::processing::{
    classifier::FileClassifier,
    text_processor::TextProcessor,
    converter::Converter,
    binary_processor::BinaryProcessor,
    pipeline::ContentExtractionPipeline,
    FileProcessor, FileType, ProcessedFile,
};
use std::path::Path;
use tempfile::{TempDir, NamedTempFile};
use std::io::Write;

/// Integration test for the complete file classification system
/// Tests Requirements 2.1, 2.2, 2.3 from the spec

#[tokio::test]
async fn test_complete_file_classification_pipeline() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create test files of different types
    let test_files = create_test_files(&temp_dir).await;
    
    // Initialize processors
    let classifier = FileClassifier::new();
    let text_processor = TextProcessor::new();
    let converter = Converter::new();
    let binary_processor = BinaryProcessor::new();
    
    // Test each file type
    for (file_path, expected_type, expected_processable) in test_files {
        // Test classification
        let classified_type = classifier.classify_file(&file_path);
        assert_eq!(
            classified_type, expected_type,
            "Classification failed for file: {}",
            file_path.display()
        );
        
        // Test processor selection and processing
        match classified_type {
            FileType::DirectText => {
                assert!(text_processor.can_process(&file_path));
                assert!(!converter.can_process(&file_path));
                
                if expected_processable {
                    let result = text_processor.process(&file_path).await;
                    assert!(result.is_ok(), "Text processing failed for: {}", file_path.display());
                    
                    let processed = result.unwrap();
                    validate_processed_file(&processed, &file_path, FileType::DirectText);
                }
            }
            FileType::Convertible => {
                assert!(!text_processor.can_process(&file_path));
                assert!(converter.can_process(&file_path));
                
                if expected_processable {
                    // Note: Converter tests use mock commands, so we'll just test the interface
                    let can_process = converter.can_process(&file_path);
                    assert!(can_process, "Converter should be able to process: {}", file_path.display());
                }
            }
            FileType::NonText => {
                assert!(!text_processor.can_process(&file_path));
                assert!(!converter.can_process(&file_path));
                assert!(binary_processor.can_process(&file_path));
                
                if expected_processable {
                    let result = binary_processor.process(&file_path).await;
                    assert!(result.is_ok(), "Binary processing failed for: {}", file_path.display());
                    
                    let processed = result.unwrap();
                    validate_processed_file(&processed, &file_path, FileType::NonText);
                }
            }
        }
    }
}

#[tokio::test]
async fn test_file_type_coverage_requirements() {
    let classifier = FileClassifier::new();
    
    // Test Type 1 (DirectText) extensions from requirements
    let direct_text_extensions = [
        "rs", "py", "js", "ts", "md", "txt", "json", "yaml", "toml",
        "sql", "sh", "c", "cpp", "java", "go", "rb", "php", "html", "css", "xml"
    ];
    
    for ext in &direct_text_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, FileType::DirectText,
            "Extension '{}' should be classified as DirectText", ext
        );
    }
    
    // Test Type 2 (Convertible) extensions from requirements
    let convertible_extensions = ["pdf", "docx", "xlsx"];
    
    for ext in &convertible_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, FileType::Convertible,
            "Extension '{}' should be classified as Convertible", ext
        );
    }
    
    // Test Type 3 (NonText) extensions from requirements
    let non_text_extensions = ["jpg", "png", "gif", "mp4", "exe", "bin"];
    
    for ext in &non_text_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, FileType::NonText,
            "Extension '{}' should be classified as NonText", ext
        );
    }
    
    // Note: zip is actually classified as Convertible because it can extract file listings
    assert_eq!(classifier.classify_by_extension("zip"), FileType::Convertible);
}

#[tokio::test]
async fn test_text_processing_metrics_accuracy() {
    let processor = TextProcessor::new();
    
    // Test content with known metrics
    let test_content = "fn main() {\n    println!(\"Hello, world!\");\n    let x = 42;\n}";
    let temp_file = create_temp_file_with_content(test_content, "rs");
    
    let result = processor.process(temp_file.path()).await.unwrap();
    
    // Validate metrics
    assert_eq!(result.line_count, Some(4)); // 4 lines
    assert!(result.word_count.unwrap() > 0); // Should have words
    assert!(result.token_count.unwrap() >= result.word_count.unwrap()); // Tokens >= words
    assert_eq!(result.content_text.as_ref().unwrap(), test_content);
    assert_eq!(result.file_type, FileType::DirectText);
    assert!(result.conversion_command.is_none());
}

#[tokio::test]
async fn test_file_size_and_line_limits() {
    use code_ingest::processing::text_processor::TextProcessorConfig;
    
    // Test file size limit
    let mut config = TextProcessorConfig::default();
    config.max_file_size_bytes = 50; // Very small limit
    let processor = TextProcessor::with_config(config);
    
    let large_content = "x".repeat(100); // Exceeds 50 bytes
    let temp_file = create_temp_file_with_content(&large_content, "txt");
    
    let result = processor.process(temp_file.path()).await;
    assert!(result.is_err(), "Should fail for files exceeding size limit");
    
    // Test line limit
    let mut config = TextProcessorConfig::default();
    config.max_lines = 2; // Very small limit
    let processor = TextProcessor::with_config(config);
    
    let multi_line_content = "line1\nline2\nline3\nline4";
    let temp_file = create_temp_file_with_content(multi_line_content, "txt");
    
    let result = processor.process(temp_file.path()).await;
    assert!(result.is_err(), "Should fail for files exceeding line limit");
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
    
    let processor = TextProcessor::new()
        .with_gitignore(temp_dir.path())
        .unwrap();
    
    // Normal file should be processable
    assert!(processor.can_process(&normal_file));
    
    // Ignored file should not be processable
    assert!(!processor.can_process(&ignored_file));
}

#[test]
fn test_classifier_custom_mappings() {
    let mut classifier = FileClassifier::new();
    
    // Test adding custom mapping
    classifier.add_custom_mapping("custom".to_string(), FileType::DirectText);
    assert_eq!(classifier.classify_by_extension("custom"), FileType::DirectText);
    
    // Test overriding existing mapping
    classifier.add_custom_mapping("pdf".to_string(), FileType::NonText);
    assert_eq!(classifier.classify_by_extension("pdf"), FileType::NonText);
    
    // Test removing mapping
    let removed = classifier.remove_custom_mapping("custom");
    assert_eq!(removed, Some(FileType::DirectText));
    assert_eq!(classifier.classify_by_extension("custom"), FileType::NonText); // Falls back to default
}

#[test]
fn test_file_type_string_conversion() {
    // Test database storage format
    assert_eq!(FileType::DirectText.as_str(), "direct_text");
    assert_eq!(FileType::Convertible.as_str(), "convertible");
    assert_eq!(FileType::NonText.as_str(), "non_text");
    
    // Test parsing from database
    assert_eq!(FileType::from_str("direct_text"), Some(FileType::DirectText));
    assert_eq!(FileType::from_str("convertible"), Some(FileType::Convertible));
    assert_eq!(FileType::from_str("non_text"), Some(FileType::NonText));
    assert_eq!(FileType::from_str("invalid"), None);
}

#[tokio::test]
async fn test_error_handling_robustness() {
    let processor = TextProcessor::new();
    
    // Test non-existent file
    let non_existent = Path::new("/non/existent/file.rs");
    let result = processor.process(non_existent).await;
    assert!(result.is_err());
    
    // Test directory instead of file
    let temp_dir = TempDir::new().unwrap();
    let result = processor.process(temp_dir.path()).await;
    assert!(result.is_err());
}

// Helper functions

async fn create_test_files(temp_dir: &TempDir) -> Vec<(std::path::PathBuf, FileType, bool)> {
    let mut files = Vec::new();
    
    // DirectText files
    let rust_file = temp_dir.path().join("main.rs");
    std::fs::write(&rust_file, "fn main() { println!(\"Hello!\"); }").unwrap();
    files.push((rust_file, FileType::DirectText, true));
    
    let python_file = temp_dir.path().join("script.py");
    std::fs::write(&python_file, "print('Hello, World!')").unwrap();
    files.push((python_file, FileType::DirectText, true));
    
    let markdown_file = temp_dir.path().join("README.md");
    std::fs::write(&markdown_file, "# Title\n\nContent here.").unwrap();
    files.push((markdown_file, FileType::DirectText, true));
    
    let json_file = temp_dir.path().join("config.json");
    std::fs::write(&json_file, r#"{"key": "value"}"#).unwrap();
    files.push((json_file, FileType::DirectText, true));
    
    // Convertible files (we'll create fake ones for testing)
    let pdf_file = temp_dir.path().join("document.pdf");
    std::fs::write(&pdf_file, b"%PDF-1.4 fake pdf content").unwrap();
    files.push((pdf_file, FileType::Convertible, false)); // Don't test processing without real tools
    
    let docx_file = temp_dir.path().join("document.docx");
    std::fs::write(&docx_file, b"PK fake docx content").unwrap();
    files.push((docx_file, FileType::Convertible, false));
    
    // NonText files
    let jpg_file = temp_dir.path().join("image.jpg");
    std::fs::write(&jpg_file, b"\xFF\xD8\xFF fake jpeg").unwrap();
    files.push((jpg_file, FileType::NonText, false));
    
    let exe_file = temp_dir.path().join("program.exe");
    std::fs::write(&exe_file, b"MZ fake executable").unwrap();
    files.push((exe_file, FileType::NonText, false));
    
    files
}

fn create_temp_file_with_content(content: &str, extension: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(&format!(".{}", extension)).unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();
    file
}

fn validate_processed_file(processed: &ProcessedFile, original_path: &Path, expected_type: FileType) {
    assert_eq!(processed.file_type, expected_type);
    assert!(!processed.filepath.is_empty());
    assert!(!processed.filename.is_empty());
    assert!(!processed.absolute_path.is_empty());
    assert!(processed.file_size_bytes > 0);
    
    // Validate filename matches
    let expected_filename = original_path.file_name().unwrap().to_string_lossy();
    assert_eq!(processed.filename, expected_filename);
    
    // Validate extension
    let expected_extension = original_path
        .extension()
        .map(|ext| ext.to_string_lossy().to_string())
        .unwrap_or_default();
    assert_eq!(processed.extension, expected_extension);
    
    // For text files, should have content and metrics
    if expected_type == FileType::DirectText {
        assert!(processed.content_text.is_some());
        assert!(processed.line_count.is_some());
        assert!(processed.word_count.is_some());
        assert!(processed.token_count.is_some());
        assert!(processed.conversion_command.is_none());
        assert!(!processed.skipped);
    }
    
    // For binary files, should be marked as skipped
    if expected_type == FileType::NonText {
        assert!(processed.skipped);
        assert!(processed.skip_reason.is_some());
        assert!(processed.line_count.is_none());
        assert!(processed.word_count.is_none());
        assert!(processed.token_count.is_none());
    }
}

#[tokio::test]
async fn test_complete_pipeline_integration() {
    let temp_dir = TempDir::new().unwrap();
    let test_files = create_test_files(&temp_dir).await;
    
    // Create pipeline
    let pipeline = ContentExtractionPipeline::new().unwrap();
    
    // Test single file processing
    for (file_path, expected_type, _) in &test_files {
        let classified_type = pipeline.classify_file(file_path);
        assert_eq!(classified_type, *expected_type);
        
        let should_process = pipeline.should_process(file_path);
        assert!(should_process, "Pipeline should be able to process: {}", file_path.display());
        
        let result = pipeline.process_file(file_path).await;
        match result {
            Ok(processed) => {
                assert_eq!(processed.file_type, *expected_type);
            }
            Err(code_ingest::error::ProcessingError::ExternalCommandFailed { .. }) => {
                // Expected if conversion tools are not available
                if *expected_type == FileType::Convertible {
                    continue; // Skip convertible files if tools are missing
                } else {
                    panic!("Unexpected conversion failure for non-convertible file: {}", file_path.display());
                }
            }
            Err(other) => {
                panic!("Pipeline processing failed for {}: {:?}", file_path.display(), other);
            }
        }
    }
    
    // Test parallel processing
    let file_paths: Vec<_> = test_files.iter().map(|(path, _, _)| path.clone()).collect();
    let results = pipeline.process_files(file_paths.clone()).await;
    
    // Count successful results (some may fail if conversion tools are missing)
    let successful_count = results.iter().filter(|r| r.is_ok()).count();
    assert!(successful_count >= 5, "Should process at least 5 files successfully"); // All except possibly PDF
    
    // Test processing statistics
    let stats = pipeline.get_processing_stats(&file_paths);
    assert_eq!(stats.total_count, test_files.len());
    assert!(stats.processable_count > 0);
}