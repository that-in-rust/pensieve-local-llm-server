//! Integration tests for ingestion tracking and metadata management
//! 
//! Tests Requirements 3.5, 3.6, 1.5 from the spec:
//! - start_ingestion_record and complete_ingestion_record methods
//! - Timestamp tracking and file count statistics
//! - Error recovery and partial ingestion handling
//! - Ingestion result reporting with table names and metrics

use code_ingest::{
    database::Database,
    ingestion::{IngestionEngine, IngestionConfig, IngestionStatus},
    processing::{FileProcessor, FileType, ProcessedFile},
    error::ProcessingResult,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tempfile::TempDir;

/// Mock file processor for testing ingestion metadata
struct MetadataTestProcessor {
    processed_files: Arc<Mutex<Vec<PathBuf>>>,
    should_fail: bool,
}

impl MetadataTestProcessor {
    fn new(should_fail: bool) -> Self {
        Self {
            processed_files: Arc::new(Mutex::new(Vec::new())),
            should_fail,
        }
    }

    fn get_processed_count(&self) -> usize {
        self.processed_files.lock().unwrap().len()
    }
}

#[async_trait::async_trait]
impl FileProcessor for MetadataTestProcessor {
    fn can_process(&self, _file_path: &Path) -> bool {
        true
    }

    async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
        // Record processing
        self.processed_files.lock().unwrap().push(file_path.to_path_buf());

        if self.should_fail {
            return Err(code_ingest::error::ProcessingError::FileReadFailed {
                path: file_path.display().to_string(),
                cause: "Simulated failure for metadata testing".to_string(),
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

/// Create a test repository with known files
fn create_metadata_test_repository(temp_dir: &TempDir, file_count: usize) -> PathBuf {
    let repo_path = temp_dir.path().join("metadata_test_repo");
    std::fs::create_dir_all(&repo_path).unwrap();

    for i in 0..file_count {
        let file_path = repo_path.join(format!("test_file_{}.rs", i));
        let content = format!(
            "// Test file {}\nfn test_function_{}() {{\n    println!(\"Test {}\");\n}}\n",
            i, i, i
        );
        std::fs::write(&file_path, content).unwrap();
    }

    repo_path
}

/// Test Requirements 3.5, 3.6: start_ingestion_record and complete_ingestion_record methods
#[tokio::test]
async fn test_ingestion_record_lifecycle() {
    // Skip if no database URL
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test - no DATABASE_URL environment variable set");
            return;
        }
    };

    let database = match Database::new(&database_url).await {
        Ok(db) => Arc::new(db),
        Err(_) => {
            println!("Skipping test - could not connect to database");
            return;
        }
    };

    // Test creating an ingestion record
    let repo_url = Some("https://github.com/test/repo".to_string());
    let local_path = "/tmp/test/repo".to_string();
    let start_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let table_name = format!("INGEST_{}", chrono::Utc::now().format("%Y%m%d%H%M%S"));

    let ingestion_id = database
        .create_ingestion_record(repo_url.clone(), local_path.clone(), start_timestamp, &table_name)
        .await
        .unwrap();

    assert!(ingestion_id > 0, "Should return a valid ingestion ID");

    // Test completing the ingestion record
    let end_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let total_files = 42;

    database
        .complete_ingestion_record(ingestion_id, end_timestamp, total_files)
        .await
        .unwrap();

    // Verify the record was created and completed correctly
    let stats = database.get_ingestion_statistics(ingestion_id).await.unwrap();
    
    assert_eq!(stats.ingestion_id, ingestion_id);
    assert_eq!(stats.table_name, table_name);
    assert_eq!(stats.total_files, total_files);
    assert!(stats.processing_duration >= Duration::ZERO, "Should track processing duration");

    println!(
        "Successfully created and completed ingestion record {} with {} files",
        ingestion_id, total_files
    );
}

/// Test Requirements 3.6, 1.5: Timestamp tracking and file count statistics
#[tokio::test]
async fn test_timestamp_tracking_and_statistics() {
    // Skip if no database URL
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test - no DATABASE_URL environment variable set");
            return;
        }
    };

    let database = match Database::new(&database_url).await {
        Ok(db) => Arc::new(db),
        Err(_) => {
            println!("Skipping test - could not connect to database");
            return;
        }
    };

    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_metadata_test_repository(&temp_dir, 15);

    let file_processor = Arc::new(MetadataTestProcessor::new(false));

    let config = IngestionConfig::default();
    let engine = IngestionEngine::new(config, database.clone(), file_processor.clone());

    // Record start time
    let start_time = Instant::now();

    // Perform ingestion
    let result = engine
        .ingest_local_folder(repo_path.to_str().unwrap(), None)
        .await
        .unwrap();

    let elapsed = start_time.elapsed();

    // Verify timestamp tracking
    assert!(result.processing_time > Duration::ZERO, "Should track processing time");
    assert!(result.processing_time <= elapsed + Duration::from_millis(100), "Processing time should be reasonable");

    // Verify file count statistics
    assert_eq!(result.files_processed, 15, "Should process all 15 files");
    assert_eq!(result.files_failed, 0, "Should have no failures");
    assert_eq!(result.files_skipped, 0, "Should have no skipped files");

    // Verify ingestion metadata
    assert!(result.ingestion_id > 0, "Should have valid ingestion ID");
    assert!(!result.table_name.is_empty(), "Should have table name");
    assert!(result.table_name.starts_with("INGEST_"), "Table name should follow pattern");

    // Get detailed statistics
    let stats = engine.get_ingestion_statistics(result.ingestion_id).await.unwrap();
    
    assert_eq!(stats.ingestion_id, result.ingestion_id);
    assert_eq!(stats.total_files, 15);
    assert!(stats.processing_duration > Duration::ZERO, "Should track processing duration");

    println!(
        "Timestamp tracking test: {} files in {:?} (recorded: {:?})",
        result.files_processed, elapsed, stats.processing_duration
    );
}

/// Test Requirements 1.5: Error recovery and partial ingestion handling
#[tokio::test]
async fn test_error_recovery_and_partial_ingestion() {
    // Skip if no database URL
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test - no DATABASE_URL environment variable set");
            return;
        }
    };

    let database = match Database::new(&database_url).await {
        Ok(db) => Arc::new(db),
        Err(_) => {
            println!("Skipping test - could not connect to database");
            return;
        }
    };

    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_metadata_test_repository(&temp_dir, 20);

    // Create processor that fails on some files
    let file_processor = Arc::new(MetadataTestProcessor::new(true)); // Will fail on all files

    let mut config = IngestionConfig::default();
    config.batch_config.continue_on_error = true; // Continue despite errors

    let engine = IngestionEngine::new(config, database.clone(), file_processor.clone());

    // Perform ingestion with expected failures
    let result = engine
        .ingest_local_folder(repo_path.to_str().unwrap(), None)
        .await
        .unwrap();

    // Verify error recovery behavior
    assert_eq!(result.files_processed, 0, "Should have no successful processing");
    assert!(result.files_failed > 0, "Should have failures");
    assert!(result.ingestion_id > 0, "Should still create ingestion record");

    // Verify that ingestion metadata was still recorded
    let stats = engine.get_ingestion_statistics(result.ingestion_id).await.unwrap();
    
    assert_eq!(stats.ingestion_id, result.ingestion_id);
    assert_eq!(stats.total_files, 0); // No files successfully processed
    assert!(stats.processing_duration > Duration::ZERO, "Should still track time");

    // Verify the processor was called (attempted processing)
    assert!(file_processor.get_processed_count() > 0, "Should have attempted processing");

    println!(
        "Error recovery test: {} failed, {} processed, ingestion ID: {}",
        result.files_failed, result.files_processed, result.ingestion_id
    );
}

/// Test ingestion result reporting with table names and metrics
#[tokio::test]
async fn test_ingestion_result_reporting() {
    // Skip if no database URL
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test - no DATABASE_URL environment variable set");
            return;
        }
    };

    let database = match Database::new(&database_url).await {
        Ok(db) => Arc::new(db),
        Err(_) => {
            println!("Skipping test - could not connect to database");
            return;
        }
    };

    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_metadata_test_repository(&temp_dir, 10);

    let file_processor = Arc::new(MetadataTestProcessor::new(false));

    let config = IngestionConfig::default();
    let engine = IngestionEngine::new(config, database.clone(), file_processor);

    // Perform multiple ingestions to test reporting
    let mut results = Vec::new();
    for i in 0..3 {
        let result = engine
            .ingest_local_folder(repo_path.to_str().unwrap(), None)
            .await
            .unwrap();
        
        results.push(result);
        
        // Small delay between ingestions to ensure different timestamps
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Test listing all ingestion records
    let records = engine.list_ingestions().await.unwrap();
    
    assert!(records.len() >= 3, "Should have at least 3 ingestion records");

    // Verify each result has proper reporting
    for (i, result) in results.iter().enumerate() {
        // Verify table name format
        assert!(result.table_name.starts_with("INGEST_"), "Table name should follow pattern");
        assert!(result.table_name.len() >= 15, "Table name should include timestamp");
        
        // Verify metrics
        assert_eq!(result.files_processed, 10, "Should process all files in iteration {}", i);
        assert_eq!(result.files_failed, 0, "Should have no failures in iteration {}", i);
        assert!(result.processing_time > Duration::ZERO, "Should track processing time in iteration {}", i);
        
        // Verify ingestion ID is unique
        assert!(result.ingestion_id > 0, "Should have valid ingestion ID in iteration {}", i);
        
        // Find corresponding record in the list
        let record = records.iter().find(|r| r.ingestion_id == result.ingestion_id);
        assert!(record.is_some(), "Should find record for ingestion {} in list", result.ingestion_id);
        
        let record = record.unwrap();
        assert_eq!(record.table_name, result.table_name, "Table names should match");
        assert_eq!(record.status, IngestionStatus::Completed, "Status should be completed");
        assert_eq!(record.total_files_processed, Some(10), "File count should match");
    }

    println!(
        "Result reporting test: {} ingestions completed, {} total records found",
        results.len(), records.len()
    );
}

/// Test ingestion metadata lifecycle with batch statistics
#[tokio::test]
async fn test_batch_statistics_integration() {
    // Skip if no database URL
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping test - no DATABASE_URL environment variable set");
            return;
        }
    };

    let database = match Database::new(&database_url).await {
        Ok(db) => Arc::new(db),
        Err(_) => {
            println!("Skipping test - could not connect to database");
            return;
        }
    };

    let temp_dir = TempDir::new().unwrap();
    let repo_path = create_metadata_test_repository(&temp_dir, 25);

    let file_processor = Arc::new(MetadataTestProcessor::new(false));

    let mut config = IngestionConfig::default();
    config.batch_config.max_concurrency = 4; // Test with parallel processing

    let engine = IngestionEngine::new(config, database.clone(), file_processor);

    let result = engine
        .ingest_local_folder(repo_path.to_str().unwrap(), None)
        .await
        .unwrap();

    // Verify batch statistics are integrated with ingestion metadata
    assert!(result.batch_stats.files_processed > 0, "Batch stats should show processed files");
    assert_eq!(result.batch_stats.files_failed, 0, "Batch stats should show no failures");
    assert!(result.batch_stats.total_duration > Duration::ZERO, "Batch stats should track duration");
    assert!(result.batch_stats.batches_processed > 0, "Should process at least one batch");

    // Verify consistency between result and batch stats
    assert_eq!(result.files_processed, result.batch_stats.files_processed, "File counts should match");
    assert_eq!(result.files_failed, result.batch_stats.files_failed, "Failure counts should match");

    // Verify ingestion statistics include batch information
    let stats = engine.get_ingestion_statistics(result.ingestion_id).await.unwrap();
    assert_eq!(stats.total_files, result.files_processed as i32, "Statistics should match result");

    println!(
        "Batch statistics integration: {} files, {} batches, {:?} duration",
        result.batch_stats.files_processed,
        result.batch_stats.batches_processed,
        result.batch_stats.total_duration
    );
}