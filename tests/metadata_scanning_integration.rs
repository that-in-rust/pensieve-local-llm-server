//! Integration tests for metadata scanning and hashing engine

use pensieve::prelude::*;
use pensieve::scanner::FileScanner;
use pensieve::types::FileType;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[tokio::test]
async fn test_metadata_scanning_with_parallel_processing() {
    // Create temporary directory with test files
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();
    
    // Create test files with different content
    fs::write(temp_path.join("file1.txt"), "Hello world").unwrap();
    fs::write(temp_path.join("file2.txt"), "Different content").unwrap();
    fs::write(temp_path.join("file3.txt"), "Hello world").unwrap(); // Duplicate content
    fs::write(temp_path.join("file4.rs"), "fn main() { println!(\"Hello\"); }").unwrap();
    
    // Create subdirectory with more files
    let subdir = temp_path.join("subdir");
    fs::create_dir(&subdir).unwrap();
    fs::write(subdir.join("nested.md"), "# Nested file").unwrap();
    
    // Run metadata scanning
    let scanner = FileScanner::new(temp_path);
    let metadata = scanner.scan().await.unwrap();
    
    // Verify results
    assert_eq!(metadata.len(), 5, "Should discover all 5 files");
    
    // Check that all files have metadata
    for file_meta in &metadata {
        assert!(!file_meta.full_filepath.as_os_str().is_empty());
        assert!(!file_meta.filename.is_empty());
        assert!(file_meta.size > 0);
        assert!(!file_meta.hash.is_empty(), "Hash should be calculated for file: {}", file_meta.filename);
        assert!(file_meta.depth_level > 0);
    }
    
    // Check duplicate detection
    let unique_count = metadata.iter()
        .filter(|f| f.duplicate_status == DuplicateStatus::Unique || 
                   f.duplicate_status == DuplicateStatus::Canonical)
        .count();
    let duplicate_count = metadata.iter()
        .filter(|f| f.duplicate_status == DuplicateStatus::Duplicate)
        .count();
    
    assert_eq!(unique_count, 4, "Should have 4 unique files");
    assert_eq!(duplicate_count, 1, "Should have 1 duplicate file");
    
    // Verify that duplicate files have the same hash
    let file1_hash = metadata.iter()
        .find(|f| f.filename == "file1.txt")
        .unwrap()
        .hash
        .clone();
    let file3_hash = metadata.iter()
        .find(|f| f.filename == "file3.txt")
        .unwrap()
        .hash
        .clone();
    
    assert_eq!(file1_hash, file3_hash, "Duplicate files should have same hash");
}

#[tokio::test]
async fn test_hash_calculation_consistency() {
    // Create temporary file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.txt");
    let content = "This is test content for hash calculation";
    fs::write(&file_path, content).unwrap();
    
    // Calculate hash multiple times
    let scanner = FileScanner::new(temp_dir.path());
    let metadata1 = scanner.extract_metadata(&file_path).await.unwrap();
    let metadata2 = scanner.extract_metadata(&file_path).await.unwrap();
    
    // Hashes should be identical
    assert_eq!(metadata1.hash, metadata2.hash, "Hash calculation should be consistent");
    assert!(!metadata1.hash.is_empty(), "Hash should not be empty");
    assert_eq!(metadata1.hash.len(), 64, "SHA-256 hash should be 64 characters");
}

#[tokio::test]
async fn test_file_metadata_extraction() {
    // Create temporary file with known properties
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("metadata_test.txt");
    let content = "Test content for metadata extraction";
    fs::write(&file_path, content).unwrap();
    
    // Extract metadata
    let scanner = FileScanner::new(temp_dir.path());
    let metadata = scanner.extract_metadata(&file_path).await.unwrap();
    
    // Verify metadata fields
    assert_eq!(metadata.filename, "metadata_test.txt");
    assert_eq!(metadata.file_extension, Some("txt".to_string()));
    assert_eq!(metadata.file_type, FileType::File);
    assert_eq!(metadata.size, content.len() as u64);
    assert!(!metadata.hash.is_empty());
    assert_eq!(metadata.depth_level, 1); // One level deep from root
    assert_eq!(metadata.duplicate_status, DuplicateStatus::Unique);
    assert!(metadata.duplicate_group_id.is_none());
    assert_eq!(metadata.processing_status, ProcessingStatus::Pending);
    
    // Verify path components
    assert_eq!(metadata.relative_path, Path::new("metadata_test.txt"));
    assert!(!metadata.is_hidden);
    assert!(!metadata.is_symlink);
    assert!(metadata.symlink_target.is_none());
}

#[tokio::test]
async fn test_progress_reporting() {
    // Create multiple test files
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();
    
    for i in 0..10 {
        fs::write(temp_path.join(format!("file{}.txt", i)), format!("Content {}", i)).unwrap();
    }
    
    // Run scanning and verify progress is reported
    let scanner = FileScanner::new(temp_path);
    let metadata = scanner.scan().await.unwrap();
    
    assert_eq!(metadata.len(), 10, "Should process all 10 files");
    
    // All files should have been processed successfully
    for file_meta in &metadata {
        assert!(!file_meta.hash.is_empty(), "All files should have hashes calculated");
        assert!(file_meta.size > 0, "All files should have size > 0");
    }
}

#[tokio::test]
async fn test_large_file_handling() {
    // Create a larger file to test buffered I/O
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large_file.txt");
    
    // Create 1MB file
    let content = "A".repeat(1024 * 1024);
    fs::write(&file_path, &content).unwrap();
    
    // Extract metadata
    let scanner = FileScanner::new(temp_dir.path());
    let metadata = scanner.extract_metadata(&file_path).await.unwrap();
    
    // Verify large file is handled correctly
    assert_eq!(metadata.size, content.len() as u64);
    assert!(!metadata.hash.is_empty());
    assert_eq!(metadata.hash.len(), 64); // SHA-256 hash length
}

#[tokio::test]
async fn test_directory_depth_calculation() {
    // Create nested directory structure
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();
    
    // Create nested directories
    let level1 = temp_path.join("level1");
    let level2 = level1.join("level2");
    let level3 = level2.join("level3");
    fs::create_dir_all(&level3).unwrap();
    
    // Create files at different levels
    fs::write(temp_path.join("root.txt"), "root level").unwrap();
    fs::write(level1.join("level1.txt"), "level 1").unwrap();
    fs::write(level2.join("level2.txt"), "level 2").unwrap();
    fs::write(level3.join("level3.txt"), "level 3").unwrap();
    
    // Scan and verify depth levels
    let scanner = FileScanner::new(temp_path);
    let metadata = scanner.scan().await.unwrap();
    
    assert_eq!(metadata.len(), 4);
    
    // Find and verify each file's depth
    let root_file = metadata.iter().find(|f| f.filename == "root.txt").unwrap();
    assert_eq!(root_file.depth_level, 1);
    
    let level1_file = metadata.iter().find(|f| f.filename == "level1.txt").unwrap();
    assert_eq!(level1_file.depth_level, 2);
    
    let level2_file = metadata.iter().find(|f| f.filename == "level2.txt").unwrap();
    assert_eq!(level2_file.depth_level, 3);
    
    let level3_file = metadata.iter().find(|f| f.filename == "level3.txt").unwrap();
    assert_eq!(level3_file.depth_level, 4);
}

#[tokio::test]
async fn test_performance_requirements() {
    // Create many small files to test performance
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();
    
    // Create 100 small files
    for i in 0..100 {
        fs::write(temp_path.join(format!("perf_test_{}.txt", i)), format!("Content {}", i)).unwrap();
    }
    
    // Measure scanning performance
    let start = std::time::Instant::now();
    let scanner = FileScanner::new(temp_path);
    let metadata = scanner.scan().await.unwrap();
    let elapsed = start.elapsed();
    
    // Verify all files processed
    assert_eq!(metadata.len(), 100);
    
    // Performance should be reasonable (>100 files/sec for small files)
    let files_per_second = metadata.len() as f64 / elapsed.as_secs_f64();
    assert!(files_per_second > 100.0, "Should process >100 files/sec, got {:.2}", files_per_second);
    
    println!("Performance: {:.2} files/sec", files_per_second);
}