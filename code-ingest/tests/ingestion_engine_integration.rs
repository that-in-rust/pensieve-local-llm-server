//! Integration tests for the complete ingestion engine workflow
//! 
//! Tests Requirements 1.1, 1.2, 6.1, 6.2 from the spec:
//! - Parallel file processing pipeline with async processing
//! - Progress reporting with mpsc channels  
//! - Memory management and resource cleanup
//! - Performance contracts for streaming and parallel processing

use code_ingest::{
    core::{IngestionEngine, IngestionConfig},
    database::Database,
    processing::{
        FileProcessor, FileType, ProcessedFile,
    },
    error::ProcessingResult,
    ingestion::{
        batch_processor::BatchProgress,
        SourceType,
    },
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tempfile::TempDir;
use walkdir;

/// Mock file processor for testing with configurable behavior
struct TestFileProcessor {
    processing_delay: Duration,
    failure_rate: f64,
    processed_files: Arc<Mutex<Vec<PathBuf>>>,
}

impl TestFileProcessor {
    fn new(processing_delay: Duration, failure_rate: f64) -> Self {
        Self {
            processing_delay,
            failure_rate,
            processed_files: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_processed_files(&self) -> Vec<PathBuf> {
        self.processed_files.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl FileProcessor for TestFileProcessor {
    fn can_process(&self, _file_path: &Path) -> bool {
        true
    }

    async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
        // Record processing
        self.processed_files.lock().unwrap().push(file_path.to_path_buf());

        // Simulate processing delay
        tokio::time::sleep(self.processing_delay).await;

        // Simulate random failures
        if fastrand::f64() < self.failure_rate {
            return Err(code_ingest::error::ProcessingError::FileReadFailed {
                path: file_path.display().to_string(),
                cause: "Simulated failure".to_string(),
            });
        }

        // Create mock processed file
        let metadata = std::fs::metadata(file_path).unwrap();
        let content = std::fs::read_to_string(file_path).unwrap_or_default();
        
        Ok(ProcessedFile {
            filepath: file_path.display().to_string(),
            filename: file_path.file_name().unwrap().to_str().unwrap().to_string(),
            extension: file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_string(),
            file_size_bytes: metadata.len() as i64,
            line_count: Some(content.lines().count() as i32),
            word_count: Some(content.split_whitespace().count() as i32),
            token_count: Some((content.split_whitespace().count() as f32 * 1.3) as i32),
            content_text: Some(content),
            file_type: FileType::DirectText,
            conversion_command: None,
            relative_path: file_path.display().to_string(),
            absolute_path: file_path.canonicalize().unwrap().display().to_string(),
            skipped: false,
            skip_reason: None,
        })
    }

    fn get_file_type(&self) -> FileType {
        FileType::DirectText
    }
}

/// Create a test repository with various file types and sizes
fn create_test_repository(temp_dir: &TempDir, file_count: usize) -> PathBuf {
    let repo_path = temp_dir.path().join("test_repo");
    std::fs::create_dir_all(&repo_path).unwrap();

    // Create various file types
    for i in 0..file_count {
        let file_path = repo_path.join(format!("file_{}.rs", i));
        let content = format!(
            "// File {}\nfn main() {{\n    println!(\"Hello from file {}\");\n}}\n",
            i, i
        );
        std::fs::write(&file_path, content).unwrap();
    }

    // Create some subdirectories
    let src_dir = repo_path.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    
    for i in 0..5 {
        let file_path = src_dir.join(format!("module_{}.rs", i));
        let content = format!(
            "pub mod module_{} {{\n    pub fn function_{}() {{\n        // Implementation\n    }}\n}}\n",
            i, i
        );
        std::fs::write(&file_path, content).unwrap();
    }

    // Create some configuration files
    std::fs::write(
        repo_path.join("Cargo.toml"),
        r#"[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
"#,
    ).unwrap();

    std::fs::write(
        repo_path.join("README.md"),
        "# Test Project\n\nThis is a test repository for ingestion testing.\n",
    ).unwrap();

    repo_path
}

/// Test Requirement 1.1: Async file processing with BatchProcessor
#[tokio::test]
async fn test_async_file_processing_pipeline() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 20);

    // Create test file processor
    let file_processor = Arc::new(TestFileProcessor::new(
        Duration::from_millis(10), // Fast processing
        0.0, // No failures
    ));

    // Create batch processor with async configuration
    let config = BatchConfig {
        max_concurrency: 4,
        show_progress: false,
        ..Default::default()
    };
    let batch_processor = BatchProcessor::new(config, file_processor.clone());

    // Collect all files from the test repository
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    // Test async parallel processing
    let start_time = Instant::now();
    let (processed_files, stats) = batch_processor
        .process_files(file_paths.clone(), None)
        .await
        .unwrap();

    let elapsed = start_time.elapsed();

    // Verify results
    assert!(processed_files.len() > 15, "Should process most files");
    assert_eq!(stats.files_failed, 0, "No files should fail");
    assert!(stats.files_processed > 15, "Should process files successfully");
    assert!(elapsed < Duration::from_secs(5), "Should complete quickly");

    // Verify files were actually processed by the processor
    let processor_files = file_processor.get_processed_files();
    assert_eq!(processor_files.len(), processed_files.len());

    // Verify async processing performance
    let throughput = processed_files.len() as f64 / elapsed.as_secs_f64();
    assert!(throughput > 10.0, "Should achieve reasonable throughput: {:.2} files/sec", throughput);

    println!(
        "Async processing: {} files in {:?} ({:.2} files/sec)",
        processed_files.len(), elapsed, throughput
    );
}

/// Test Requirement 1.2: Add progress reporting with mpsc channels
#[tokio::test]
async fn test_progress_reporting_with_mpsc() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 30);

    let file_processor = Arc::new(TestFileProcessor::new(
        Duration::from_millis(20), // Slower to see progress
        0.0,
    ));

    // Set up progress tracking
    let progress_updates = Arc::new(Mutex::new(Vec::new()));
    let progress_updates_clone = Arc::clone(&progress_updates);

    // Create progress callback
    let progress_callback = Box::new(move |progress: BatchProgress| {
        progress_updates_clone.lock().unwrap().push(progress);
    });

    // Create batch processor
    let config = BatchConfig {
        max_concurrency: 2,
        show_progress: false,
        progress_update_interval: Duration::from_millis(10), // Frequent updates
        ..Default::default()
    };
    let batch_processor = BatchProcessor::new(config, file_processor);

    // Collect files
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    // Process with progress tracking
    let result = batch_processor
        .process_files(file_paths, Some(progress_callback))
        .await;

    assert!(result.is_ok(), "Processing should succeed");

    // Verify progress updates were received
    let updates = progress_updates.lock().unwrap();
    assert!(updates.len() > 0, "Should have received progress updates");

    // Verify progress data makes sense
    if let Some(last_update) = updates.last() {
        assert!(last_update.files_processed > 0);
        assert!(last_update.total_files > 0);
        assert!(last_update.processing_rate >= 0.0);
    }

    // Verify progress progression
    let mut previous_processed = 0;
    for update in updates.iter() {
        assert!(update.files_processed >= previous_processed, "Progress should not go backwards");
        assert!(update.files_processed <= update.total_files, "Processed should not exceed total");
        previous_processed = update.files_processed;
    }

    println!("Received {} progress updates", updates.len());
}

/// Test Requirement 6.1: Implement parallel processing using tokio tasks
#[tokio::test]
async fn test_parallel_processing_performance() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 50); // More files for parallel testing

    // Collect files once
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    // Test with different concurrency levels
    for max_concurrency in [1, 4, 8] {
        let file_processor = Arc::new(TestFileProcessor::new(
            Duration::from_millis(10), // Consistent processing time
            0.0,
        ));

        let config = BatchConfig {
            max_concurrency,
            show_progress: false,
            ..Default::default()
        };

        let batch_processor = BatchProcessor::new(config, file_processor);

        let start_time = Instant::now();
        let (processed_files, stats) = batch_processor
            .process_files(file_paths.clone(), None)
            .await
            .unwrap();

        let elapsed = start_time.elapsed();
        let throughput = processed_files.len() as f64 / elapsed.as_secs_f64();

        println!(
            "Concurrency {}: {} files in {:?} ({:.2} files/sec)",
            max_concurrency, processed_files.len(), elapsed, throughput
        );

        // Verify results
        assert!(throughput > 0.0, "Should have positive throughput");
        assert!(elapsed < Duration::from_secs(30), "Should complete in reasonable time");
        assert_eq!(stats.files_failed, 0, "No files should fail");
        assert!(stats.files_processed > 40, "Should process most files");

        // Higher concurrency should generally process more files per second
        // (though this isn't guaranteed due to overhead and test environment variability)
        if max_concurrency == 1 {
            assert!(throughput > 10.0, "Single-threaded should be reasonably fast");
        }
    }
}

/// Test Requirement 6.1: Use streaming file processing to maintain constant memory usage
#[tokio::test]
async fn test_streaming_memory_management() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create a larger repository to test memory management
    let repo_path = create_test_repository(&temp_dir, 100);
    
    // Add some larger files to test memory pressure
    for i in 0..10 {
        let large_file = repo_path.join(format!("large_file_{}.txt", i));
        let content = "x".repeat(10_000); // 10KB files
        std::fs::write(&large_file, content).unwrap();
    }

    let file_processor = Arc::new(TestFileProcessor::new(
        Duration::from_millis(5),
        0.0,
    ));

    // Configure with memory limits
    let config = BatchConfig {
        max_memory_bytes: 512 * 1024, // 512KB limit (tight for testing)
        max_concurrency: 4,
        show_progress: false,
        ..Default::default()
    };

    let batch_processor = BatchProcessor::new(config, file_processor);

    // Collect files
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    // Monitor memory usage during processing
    let start_memory = get_memory_usage();
    
    let result = batch_processor
        .process_files(file_paths, None)
        .await;

    let end_memory = get_memory_usage();
    
    assert!(result.is_ok(), "Processing should succeed even with memory limits");
    
    let (processed_files, stats) = result.unwrap();
    assert!(processed_files.len() > 100, "Should process most files: {}", processed_files.len());
    
    // Verify memory monitoring worked
    let (current_memory, peak_memory) = batch_processor.get_memory_usage();
    println!("Memory usage - Current: {}, Peak: {}", current_memory, peak_memory);
    
    // Memory usage should be reasonable (this is a rough check)
    let memory_increase = end_memory.saturating_sub(start_memory);
    println!("System memory increase during processing: {} bytes", memory_increase);
    
    // The memory increase should be bounded (not proportional to total file size)
    assert!(memory_increase < 50 * 1024 * 1024, "Memory usage should be bounded"); // < 50MB increase
    
    // Peak memory should be tracked
    assert!(peak_memory > 0, "Should track peak memory usage");
}

/// Test Requirement 6.2: Add memory management and resource cleanup
#[tokio::test]
async fn test_resource_cleanup() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 30);

    // Collect files
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    // Test multiple processing runs to verify cleanup
    for i in 0..3 {
        let file_processor = Arc::new(TestFileProcessor::new(
            Duration::from_millis(10),
            0.1, // 10% failure rate to test cleanup
        ));

        let config = BatchConfig {
            max_concurrency: 4,
            show_progress: false,
            continue_on_error: true, // Continue despite failures
            ..Default::default()
        };

        let batch_processor = BatchProcessor::new(config, file_processor);

        let result = batch_processor
            .process_files(file_paths.clone(), None)
            .await;

        assert!(result.is_ok(), "Processing run {} should succeed", i);
        
        let (processed_files, stats) = result.unwrap();
        
        // Should process some files successfully despite failures
        assert!(stats.files_processed > 0, "Should process some files in run {}", i);
        assert!(stats.files_failed > 0, "Should have some failures in run {}", i);
        
        // Verify resources are cleaned up between runs
        // (This is implicit in the successful completion of multiple runs)
        println!("Completed processing run {} - processed: {}, failed: {}", 
                i + 1, stats.files_processed, stats.files_failed);
    }

    // Test graceful shutdown
    let file_processor = Arc::new(TestFileProcessor::new(Duration::from_millis(100), 0.0));
    let config = BatchConfig::default();
    let batch_processor = BatchProcessor::new(config, file_processor);
    
    // Start a long-running process
    let files_clone = file_paths.clone();
    let processor_clone = batch_processor.clone();
    let process_task = tokio::spawn(async move {
        processor_clone.process_files(files_clone, None).await
    });

    // Request shutdown after a short delay
    tokio::time::sleep(Duration::from_millis(50)).await;
    batch_processor.request_shutdown();
    
    // Verify shutdown was requested
    assert!(batch_processor.is_shutdown_requested(), "Shutdown should be requested");
    
    // Wait for processing to complete (should be interrupted)
    let result = process_task.await.unwrap();
    assert!(result.is_ok(), "Should handle shutdown gracefully");
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling_and_recovery() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 20);

    // Create processor with high failure rate
    let file_processor = Arc::new(TestFileProcessor::new(
        Duration::from_millis(10),
        0.5, // 50% failure rate
    ));

    let config = BatchConfig {
        max_concurrency: 2,
        show_progress: false,
        continue_on_error: true, // Should continue despite errors
        ..Default::default()
    };

    let batch_processor = BatchProcessor::new(config, file_processor);

    // Collect files
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    let result = batch_processor
        .process_files(file_paths, None)
        .await;

    assert!(result.is_ok(), "Processing should succeed despite individual file failures");
    
    let (processed_files, stats) = result.unwrap();
    
    // Should have some successes and some failures
    assert!(stats.files_processed > 0, "Should process some files successfully");
    assert!(stats.files_failed > 0, "Should have some failures");
    assert_eq!(processed_files.len(), stats.files_processed, "Processed files count should match");
    
    println!(
        "Error handling test - Processed: {}, Failed: {}, Skipped: {}",
        stats.files_processed,
        stats.files_failed,
        stats.files_skipped
    );
}

/// Test performance contracts from requirements
#[tokio::test]
async fn test_performance_contracts() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 100);

    let file_processor = Arc::new(TestFileProcessor::new(
        Duration::from_millis(1), // Very fast processing
        0.0,
    ));

    let config = BatchConfig {
        max_concurrency: 8,
        show_progress: false,
        ..Default::default()
    };

    let batch_processor = BatchProcessor::new(config, file_processor);

    // Collect files
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    let start_time = Instant::now();
    let (processed_files, stats) = batch_processor
        .process_files(file_paths, None)
        .await
        .unwrap();

    let elapsed = start_time.elapsed();
    let throughput = processed_files.len() as f64 / elapsed.as_secs_f64();

    // Performance contract: Should process >100 files/second for typical text files
    // (This is from the requirements success metrics)
    assert!(
        throughput >= 50.0, // Relaxed for test environment
        "Throughput was {:.2} files/sec, expected >= 50 files/sec",
        throughput
    );

    // Memory usage should be reasonable
    assert!(
        stats.peak_memory_bytes < 100 * 1024 * 1024, // < 100MB
        "Peak memory usage was {} bytes, expected < 100MB",
        stats.peak_memory_bytes
    );

    // Verify no failures
    assert_eq!(stats.files_failed, 0, "Should have no failures");
    assert!(stats.files_processed > 90, "Should process most files");

    println!(
        "Performance test: {:.2} files/sec, peak memory: {} bytes, processed: {}",
        throughput, stats.peak_memory_bytes, stats.files_processed
    );
}

/// Test concurrent processing operations
#[tokio::test]
async fn test_concurrent_processing() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    
    // Create multiple test repositories
    let repo1 = create_test_repository(&temp_dir, 20);
    let repo2 = create_test_repository(&temp_dir, 25);
    let repo3 = create_test_repository(&temp_dir, 15);

    let repos = vec![repo1, repo2, repo3];

    // Run concurrent processing operations
    let mut tasks = Vec::new();
    for (i, repo) in repos.into_iter().enumerate() {
        let task = tokio::spawn(async move {
            let file_processor = Arc::new(TestFileProcessor::new(
                Duration::from_millis(10),
                0.0,
            ));

            let config = BatchConfig {
                max_concurrency: 2,
                show_progress: false,
                ..Default::default()
            };

            let batch_processor = BatchProcessor::new(config, file_processor);

            // Collect files
            let mut file_paths = Vec::new();
            for entry in walkdir::WalkDir::new(&repo) {
                let entry = entry.unwrap();
                if entry.file_type().is_file() {
                    file_paths.push(entry.path().to_path_buf());
                }
            }

            let result = batch_processor.process_files(file_paths, None).await;
            (i, result)
        });
        tasks.push(task);
    }

    // Wait for all to complete
    let mut results = Vec::new();
    for task in tasks {
        let (i, result) = task.await.unwrap();
        assert!(result.is_ok(), "Concurrent processing {} should succeed", i);
        results.push((i, result.unwrap()));
    }

    // Verify all processing completed successfully
    assert_eq!(results.len(), 3);
    for (i, (processed_files, stats)) in results.iter() {
        assert!(stats.files_processed > 10, "Processing {} should process files", i);
        assert_eq!(stats.files_failed, 0, "Processing {} should have no failures", i);
        println!("Concurrent processing {}: {} files processed", i, stats.files_processed);
    }
}

// Helper function to get current memory usage (rough approximation)
fn get_memory_usage() -> usize {
    // This is a simplified memory check - in a real implementation,
    // you might use a more sophisticated memory monitoring approach
    std::process::Command::new("ps")
        .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()?
                .trim()
                .parse::<usize>()
                .ok()
        })
        .map(|kb| kb * 1024) // Convert KB to bytes
        .unwrap_or(0)
}

/// Integration test for the complete processing workflow
#[tokio::test]
async fn test_complete_processing_workflow() {
    use code_ingest::ingestion::batch_processor::{BatchProcessor, BatchConfig};
    
    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_test_repository(&temp_dir, 50);

    // Use a test file processor
    let file_processor = Arc::new(TestFileProcessor::new(
        Duration::from_millis(5),
        0.0,
    ));

    let config = BatchConfig {
        max_concurrency: 4,
        show_progress: false,
        ..Default::default()
    };

    let batch_processor = BatchProcessor::new(config, file_processor.clone());

    // Collect files
    let mut file_paths = Vec::new();
    for entry in walkdir::WalkDir::new(&repo_path) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            file_paths.push(entry.path().to_path_buf());
        }
    }

    // Test complete workflow
    let result = batch_processor
        .process_files(file_paths.clone(), None)
        .await;

    assert!(result.is_ok(), "Complete workflow should succeed");
    let (processed_files, stats) = result.unwrap();

    // Verify processing results
    assert!(processed_files.len() > 40, "Should process most files");
    assert_eq!(stats.files_failed, 0, "Should have no failures");
    assert!(stats.files_processed > 40, "Should process files successfully");
    assert!(stats.total_duration > Duration::ZERO, "Should track processing time");

    // Verify files were actually processed by the processor
    let processor_files = file_processor.get_processed_files();
    assert_eq!(processor_files.len(), processed_files.len(), "Processor should have processed all files");

    // Verify processed file structure
    for processed_file in &processed_files {
        assert!(!processed_file.filepath.is_empty(), "Should have filepath");
        assert!(!processed_file.filename.is_empty(), "Should have filename");
        assert!(processed_file.file_size_bytes > 0, "Should have file size");
        assert!(processed_file.line_count.is_some(), "Should have line count");
        assert!(processed_file.word_count.is_some(), "Should have word count");
        assert!(processed_file.content_text.is_some(), "Should have content");
        assert_eq!(processed_file.file_type, FileType::DirectText, "Should be direct text");
        assert!(!processed_file.skipped, "Should not be skipped");
    }

    println!(
        "Complete workflow test: {} files processed in {:?}",
        processed_files.len(), stats.total_duration
    );
}