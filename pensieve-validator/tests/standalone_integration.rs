//! Standalone integration tests for the validation pipeline
//! 
//! This module provides integration tests that work independently
//! of the main library to verify core validation concepts.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Test that we can create and analyze a basic directory structure
#[tokio::test]
async fn test_basic_directory_creation_and_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing basic directory creation and analysis...");
    
    // Create a test directory
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create various types of files
    fs::write(base_path.join("README.md"), "# Test Project\n\nThis is a test.")?;
    fs::write(base_path.join("config.json"), r#"{"version": "1.0", "debug": true}"#)?;
    fs::write(base_path.join("data.txt"), "Sample text data")?;
    fs::write(base_path.join("script.py"), "print('Hello, World!')")?;
    fs::write(base_path.join("binary.dat"), &[0x00, 0x01, 0x02, 0x03, 0xFF])?;
    
    // Create subdirectory
    let subdir = base_path.join("subdir");
    fs::create_dir_all(&subdir)?;
    fs::write(subdir.join("nested.txt"), "Nested file content")?;
    
    // Analyze the directory
    let start_time = Instant::now();
    let analysis_results = analyze_directory_simple(base_path)?;
    let analysis_time = start_time.elapsed();
    
    // Verify results
    assert!(analysis_results.total_files >= 6, "Should find at least 6 files");
    assert!(analysis_results.total_directories >= 1, "Should find at least 1 subdirectory");
    assert!(analysis_results.total_size_bytes > 0, "Should have some content");
    assert!(analysis_time < Duration::from_secs(5), "Analysis should be fast");
    
    println!("✅ Basic directory analysis completed successfully");
    println!("   Files: {}, Directories: {}, Size: {} bytes", 
             analysis_results.total_files, analysis_results.total_directories, analysis_results.total_size_bytes);
    println!("   Analysis time: {:?}", analysis_time);
    
    Ok(())
}

/// Test chaos detection with problematic files
#[tokio::test]
async fn test_chaos_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌪️ Testing chaos detection...");
    
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create problematic files
    fs::write(base_path.join("no_extension"), "File without extension")?;
    fs::write(base_path.join("empty.txt"), "")?; // Zero-byte file
    fs::write(base_path.join("测试.txt"), "Unicode filename")?;
    fs::write(base_path.join("café.md"), "Accented filename")?;
    
    // Create a large file
    let large_content = "x".repeat(1_000_000); // 1MB
    fs::write(base_path.join("large.txt"), large_content)?;
    
    // Create deep nesting
    let mut deep_path = base_path.to_path_buf();
    for level in 1..=8 {
        deep_path = deep_path.join(format!("level_{}", level));
        fs::create_dir_all(&deep_path)?;
    }
    fs::write(deep_path.join("deep_file.txt"), "Deeply nested file")?;
    
    // Analyze for chaos
    let chaos_results = detect_chaos_simple(base_path)?;
    
    // Verify chaos detection
    assert!(chaos_results.files_without_extensions > 0, "Should detect files without extensions");
    assert!(chaos_results.zero_byte_files > 0, "Should detect zero-byte files");
    assert!(chaos_results.unicode_filenames > 0, "Should detect unicode filenames");
    assert!(chaos_results.large_files > 0, "Should detect large files");
    assert!(chaos_results.max_depth >= 8, "Should detect deep nesting");
    
    let chaos_score = calculate_chaos_score(&chaos_results);
    assert!(chaos_score > 0.0, "Should have some chaos score");
    assert!(chaos_score <= 1.0, "Chaos score should be normalized");
    
    println!("✅ Chaos detection completed successfully");
    println!("   Files without extensions: {}", chaos_results.files_without_extensions);
    println!("   Zero-byte files: {}", chaos_results.zero_byte_files);
    println!("   Unicode filenames: {}", chaos_results.unicode_filenames);
    println!("   Large files: {}", chaos_results.large_files);
    println!("   Max depth: {}", chaos_results.max_depth);
    println!("   Chaos score: {:.2}", chaos_score);
    
    Ok(())
}

/// Test performance measurement
#[tokio::test]
async fn test_performance_measurement() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Testing performance measurement...");
    
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create many files for performance testing
    for i in 0..100 {
        fs::write(base_path.join(format!("file_{:03}.txt", i)), format!("Content of file {}", i))?;
    }
    
    // Measure performance multiple times
    let mut times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _results = analyze_directory_simple(base_path)?;
        times.push(start.elapsed());
    }
    
    // Check performance consistency
    let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
    let max_time = *times.iter().max().unwrap();
    let min_time = *times.iter().min().unwrap();
    
    assert!(avg_time < Duration::from_secs(1), "Average time should be under 1 second");
    assert!(max_time.as_secs_f64() / min_time.as_secs_f64() < 3.0, "Performance should be consistent");
    
    println!("✅ Performance measurement completed successfully");
    println!("   Average time: {:?}", avg_time);
    println!("   Min time: {:?}, Max time: {:?}", min_time, max_time);
    println!("   Performance variance: {:.2}x", max_time.as_secs_f64() / min_time.as_secs_f64());
    
    Ok(())
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚨 Testing error handling...");
    
    // Test with non-existent directory
    let non_existent = Path::new("/non/existent/directory");
    let result = analyze_directory_simple(non_existent);
    assert!(result.is_err(), "Should return error for non-existent directory");
    
    // Test with empty directory
    let empty_dir = TempDir::new()?;
    let results = analyze_directory_simple(empty_dir.path())?;
    assert_eq!(results.total_files, 0, "Empty directory should have 0 files");
    assert_eq!(results.total_directories, 0, "Empty directory should have 0 subdirectories");
    
    println!("✅ Error handling test completed successfully");
    
    Ok(())
}

/// Test complete validation pipeline simulation
#[tokio::test]
async fn test_complete_validation_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Testing complete validation pipeline...");
    
    let temp_dir = TempDir::new()?;
    let base_path = temp_dir.path();
    
    // Create a mixed dataset
    fs::write(base_path.join("README.md"), "# Project\n\nDescription")?;
    fs::write(base_path.join("config.json"), r#"{"setting": "value"}"#)?;
    fs::write(base_path.join("no_ext"), "No extension file")?;
    fs::write(base_path.join("empty.txt"), "")?;
    fs::write(base_path.join("测试.txt"), "Unicode test")?;
    
    let subdir = base_path.join("src");
    fs::create_dir_all(&subdir)?;
    fs::write(subdir.join("main.rs"), "fn main() { println!(\"Hello\"); }")?;
    
    // Run complete pipeline
    let start_time = Instant::now();
    
    // Phase 1: Directory Analysis
    let analysis_results = analyze_directory_simple(base_path)?;
    
    // Phase 2: Chaos Detection
    let chaos_results = detect_chaos_simple(base_path)?;
    let chaos_score = calculate_chaos_score(&chaos_results);
    
    // Phase 3: Performance Assessment
    let performance_score = calculate_performance_score(&analysis_results, start_time.elapsed());
    
    // Phase 4: Production Readiness Assessment
    let is_production_ready = assess_production_readiness(chaos_score, performance_score);
    
    let total_time = start_time.elapsed();
    
    // Verify pipeline results
    assert!(analysis_results.total_files > 0, "Should analyze files");
    assert!(chaos_score >= 0.0 && chaos_score <= 1.0, "Chaos score should be normalized");
    assert!(performance_score >= 0.0 && performance_score <= 1.0, "Performance score should be normalized");
    assert!(total_time < Duration::from_secs(10), "Pipeline should complete quickly");
    
    println!("✅ Complete validation pipeline completed successfully");
    println!("   Total files: {}", analysis_results.total_files);
    println!("   Chaos score: {:.2}", chaos_score);
    println!("   Performance score: {:.2}", performance_score);
    println!("   Production ready: {}", is_production_ready);
    println!("   Total pipeline time: {:?}", total_time);
    
    Ok(())
}

/// Run all integration tests
#[tokio::test]
async fn test_integration_suite() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Running complete integration test suite...");
    
    let start_time = Instant::now();
    
    // Run all tests
    let test_results = vec![
        ("Basic Directory Analysis", test_basic_directory_creation_and_analysis().await),
        ("Chaos Detection", test_chaos_detection().await),
        ("Performance Measurement", test_performance_measurement().await),
        ("Error Handling", test_error_handling().await),
        ("Complete Validation Pipeline", test_complete_validation_pipeline().await),
    ];
    
    let total_time = start_time.elapsed();
    
    // Summarize results
    let mut passed = 0;
    let mut failed = 0;
    
    for (test_name, result) in test_results {
        match result {
            Ok(_) => {
                println!("✅ {}", test_name);
                passed += 1;
            }
            Err(e) => {
                println!("❌ {}: {:?}", test_name, e);
                failed += 1;
            }
        }
    }
    
    println!("\n📊 Integration Test Suite Summary:");
    println!("   Total time: {:?}", total_time);
    println!("   Tests passed: {}", passed);
    println!("   Tests failed: {}", failed);
    println!("   Success rate: {:.1}%", (passed as f64 / (passed + failed) as f64) * 100.0);
    
    if failed > 0 {
        return Err(format!("{} tests failed", failed).into());
    }
    
    println!("🎉 All integration tests passed!");
    Ok(())
}

// Helper functions for the integration tests

#[derive(Debug)]
struct SimpleAnalysisResults {
    total_files: usize,
    total_directories: usize,
    total_size_bytes: u64,
}

#[derive(Debug)]
struct SimpleChaosResults {
    files_without_extensions: usize,
    zero_byte_files: usize,
    unicode_filenames: usize,
    large_files: usize,
    max_depth: usize,
}

fn analyze_directory_simple(directory: &Path) -> Result<SimpleAnalysisResults, Box<dyn std::error::Error>> {
    let mut total_files = 0;
    let mut total_directories = 0;
    let mut total_size_bytes = 0;
    
    analyze_directory_recursive(directory, &mut total_files, &mut total_directories, &mut total_size_bytes)?;
    
    Ok(SimpleAnalysisResults {
        total_files,
        total_directories,
        total_size_bytes,
    })
}

fn analyze_directory_recursive(
    directory: &Path,
    total_files: &mut usize,
    total_directories: &mut usize,
    total_size_bytes: &mut u64,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            *total_directories += 1;
            analyze_directory_recursive(&path, total_files, total_directories, total_size_bytes)?;
        } else {
            *total_files += 1;
            if let Ok(metadata) = entry.metadata() {
                *total_size_bytes += metadata.len();
            }
        }
    }
    
    Ok(())
}

fn detect_chaos_simple(directory: &Path) -> Result<SimpleChaosResults, Box<dyn std::error::Error>> {
    let mut files_without_extensions = 0;
    let mut zero_byte_files = 0;
    let mut unicode_filenames = 0;
    let mut large_files = 0;
    let mut max_depth = 0;
    
    detect_chaos_recursive(
        directory,
        0,
        &mut files_without_extensions,
        &mut zero_byte_files,
        &mut unicode_filenames,
        &mut large_files,
        &mut max_depth,
    )?;
    
    Ok(SimpleChaosResults {
        files_without_extensions,
        zero_byte_files,
        unicode_filenames,
        large_files,
        max_depth,
    })
}

fn detect_chaos_recursive(
    directory: &Path,
    current_depth: usize,
    files_without_extensions: &mut usize,
    zero_byte_files: &mut usize,
    unicode_filenames: &mut usize,
    large_files: &mut usize,
    max_depth: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
    *max_depth = (*max_depth).max(current_depth);
    
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            detect_chaos_recursive(
                &path,
                current_depth + 1,
                files_without_extensions,
                zero_byte_files,
                unicode_filenames,
                large_files,
                max_depth,
            )?;
        } else {
            // Check for files without extensions
            if path.extension().is_none() {
                *files_without_extensions += 1;
            }
            
            // Check file size
            if let Ok(metadata) = entry.metadata() {
                let size = metadata.len();
                if size == 0 {
                    *zero_byte_files += 1;
                }
                if size > 10_000_000 { // > 10MB
                    *large_files += 1;
                }
            }
            
            // Check for unicode filenames
            if let Some(filename) = path.file_name() {
                let filename_str = filename.to_string_lossy();
                if filename_str.chars().any(|c| !c.is_ascii()) {
                    *unicode_filenames += 1;
                }
            }
        }
    }
    
    Ok(())
}

fn calculate_chaos_score(chaos_results: &SimpleChaosResults) -> f64 {
    let total_issues = chaos_results.files_without_extensions
        + chaos_results.zero_byte_files
        + chaos_results.unicode_filenames
        + chaos_results.large_files;
    
    let base_score = (total_issues as f64) * 0.1;
    let depth_penalty = if chaos_results.max_depth > 5 { 0.2 } else { 0.0 };
    
    (base_score + depth_penalty).min(1.0)
}

fn calculate_performance_score(analysis_results: &SimpleAnalysisResults, analysis_time: Duration) -> f64 {
    if analysis_results.total_files == 0 {
        return 1.0;
    }
    
    let files_per_second = analysis_results.total_files as f64 / analysis_time.as_secs_f64();
    let throughput_score = (files_per_second / 1000.0).min(1.0);
    
    // Penalty for very large datasets
    let size_penalty = if analysis_results.total_files > 10000 { 0.2 } else { 0.0 };
    
    (throughput_score - size_penalty).max(0.0)
}

fn assess_production_readiness(chaos_score: f64, performance_score: f64) -> bool {
    chaos_score < 0.5 && performance_score > 0.7
}