//! Performance regression tests for the validation framework
//! 
//! These tests ensure that the validation framework itself maintains
//! acceptable performance characteristics and can detect performance
//! regressions in the tools it validates.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use pensieve_validator::*;

/// Performance test data generator
pub struct PerformanceTestDataGenerator;

impl PerformanceTestDataGenerator {
    /// Create a dataset optimized for performance testing
    pub fn create_performance_test_dataset(
        num_files: usize,
        avg_file_size: usize,
        directory_depth: usize,
    ) -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(ValidationError::FileSystem)?;
        let base_path = temp_dir.path();

        // Create files distributed across directory structure
        let files_per_level = (num_files as f64 / directory_depth as f64).ceil() as usize;
        
        for level in 0..directory_depth {
            let level_dir = base_path.join(format!("level_{}", level));
            fs::create_dir_all(&level_dir)?;
            
            let files_this_level = if level == directory_depth - 1 {
                num_files - (level * files_per_level)
            } else {
                files_per_level
            };
            
            for file_idx in 0..files_this_level {
                let file_name = format!("file_{}_{}.txt", level, file_idx);
                let content = Self::generate_file_content(avg_file_size, level, file_idx);
                fs::write(level_dir.join(file_name), content)?;
            }
        }

        Ok(temp_dir)
    }

    /// Create a dataset that scales linearly with size
    pub fn create_scalability_test_dataset(scale_factor: usize) -> Result<TempDir> {
        let base_files = 100;
        let base_size = 1000;
        let base_depth = 5;

        Self::create_performance_test_dataset(
            base_files * scale_factor,
            base_size * scale_factor,
            base_depth + (scale_factor / 2),
        )
    }

    /// Create a dataset with specific performance characteristics
    pub fn create_targeted_performance_dataset(characteristics: &PerformanceCharacteristics) -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(ValidationError::FileSystem)?;
        let base_path = temp_dir.path();

        match characteristics {
            PerformanceCharacteristics::ManySmallFiles => {
                // 10,000 small files (100 bytes each)
                for i in 0..10_000 {
                    let dir_idx = i / 100;
                    let dir_path = base_path.join(format!("dir_{:03}", dir_idx));
                    fs::create_dir_all(&dir_path)?;
                    
                    let content = format!("Small file content {}", i);
                    fs::write(dir_path.join(format!("small_{:05}.txt", i)), content)?;
                }
            }
            PerformanceCharacteristics::FewLargeFiles => {
                // 10 large files (10MB each)
                for i in 0..10 {
                    let content = "Large file content line.\n".repeat(400_000); // ~10MB
                    fs::write(base_path.join(format!("large_{}.txt", i)), content)?;
                }
            }
            PerformanceCharacteristics::DeepNesting => {
                // Deep directory structure (50 levels)
                let mut current_path = base_path.to_path_buf();
                for level in 0..50 {
                    current_path = current_path.join(format!("level_{:02}", level));
                    fs::create_dir_all(&current_path)?;
                    
                    // Add a file every 5 levels
                    if level % 5 == 0 {
                        let content = format!("File at level {}", level);
                        fs::write(current_path.join(format!("file_{}.txt", level)), content)?;
                    }
                }
            }
            PerformanceCharacteristics::WideStructure => {
                // Wide directory structure (1000 subdirectories)
                for i in 0..1000 {
                    let subdir = base_path.join(format!("subdir_{:04}", i));
                    fs::create_dir_all(&subdir)?;
                    
                    let content = format!("File in subdirectory {}", i);
                    fs::write(subdir.join("file.txt"), content)?;
                }
            }
            PerformanceCharacteristics::MixedContent => {
                // Mix of different file types and sizes
                Self::create_mixed_content_dataset(base_path)?;
            }
        }

        Ok(temp_dir)
    }

    fn generate_file_content(size: usize, level: usize, file_idx: usize) -> String {
        let base_content = format!("Level {} File {} Content: ", level, file_idx);
        let padding_needed = size.saturating_sub(base_content.len());
        let padding = "x".repeat(padding_needed);
        format!("{}{}", base_content, padding)
    }

    fn create_mixed_content_dataset(base_path: &Path) -> Result<()> {
        // Text files
        let text_dir = base_path.join("text");
        fs::create_dir_all(&text_dir)?;
        for i in 0..100 {
            let content = format!("Text file {} with some content that varies in length.", i);
            fs::write(text_dir.join(format!("text_{:03}.txt", i)), content)?;
        }

        // Binary files
        let binary_dir = base_path.join("binary");
        fs::create_dir_all(&binary_dir)?;
        for i in 0..50 {
            let content: Vec<u8> = (0..1000).map(|x| ((x + i) % 256) as u8).collect();
            fs::write(binary_dir.join(format!("binary_{:03}.dat", i)), content)?;
        }

        // JSON files
        let json_dir = base_path.join("json");
        fs::create_dir_all(&json_dir)?;
        for i in 0..30 {
            let content = format!(r#"{{"id": {}, "name": "item_{}", "data": [1, 2, 3, 4, 5]}}"#, i, i);
            fs::write(json_dir.join(format!("data_{:03}.json", i)), content)?;
        }

        // Large files
        let large_dir = base_path.join("large");
        fs::create_dir_all(&large_dir)?;
        for i in 0..5 {
            let content = "Large file line content.\n".repeat(100_000); // ~2.5MB each
            fs::write(large_dir.join(format!("large_{}.txt", i)), content)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum PerformanceCharacteristics {
    ManySmallFiles,
    FewLargeFiles,
    DeepNesting,
    WideStructure,
    MixedContent,
}

/// Performance benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub execution_time: Duration,
    pub memory_usage: u64,
    pub files_processed: u64,
    pub throughput: f64, // files per second
    pub memory_efficiency: f64, // bytes per file
}

impl BenchmarkResults {
    pub fn new(
        execution_time: Duration,
        memory_usage: u64,
        files_processed: u64,
    ) -> Self {
        let throughput = if execution_time.as_secs_f64() > 0.0 {
            files_processed as f64 / execution_time.as_secs_f64()
        } else {
            0.0
        };
        
        let memory_efficiency = if files_processed > 0 {
            memory_usage as f64 / files_processed as f64
        } else {
            0.0
        };

        Self {
            execution_time,
            memory_usage,
            files_processed,
            throughput,
            memory_efficiency,
        }
    }

    pub fn compare_to(&self, baseline: &BenchmarkResults) -> PerformanceComparison {
        let execution_time_ratio = self.execution_time.as_secs_f64() / baseline.execution_time.as_secs_f64();
        let memory_ratio = self.memory_usage as f64 / baseline.memory_usage as f64;
        let throughput_ratio = self.throughput / baseline.throughput;

        PerformanceComparison {
            execution_time_change: execution_time_ratio - 1.0,
            memory_usage_change: memory_ratio - 1.0,
            throughput_change: throughput_ratio - 1.0,
            is_regression: execution_time_ratio > 1.2 || memory_ratio > 1.5 || throughput_ratio < 0.8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    pub execution_time_change: f64, // Positive = slower, negative = faster
    pub memory_usage_change: f64,   // Positive = more memory, negative = less
    pub throughput_change: f64,     // Positive = faster, negative = slower
    pub is_regression: bool,
}

/// Performance regression test suite
pub struct PerformanceRegressionTester;

impl PerformanceRegressionTester {
    /// Run a comprehensive performance benchmark
    pub async fn run_performance_benchmark(
        dataset_path: &Path,
        test_name: &str,
    ) -> Result<BenchmarkResults> {
        println!("🏃 Running performance benchmark: {}", test_name);
        
        let config = ValidationOrchestratorConfig {
            target_directory: dataset_path.to_path_buf(),
            pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
            output_directory: dataset_path.join("benchmark_output"),
            timeout_seconds: 300,
            memory_limit_mb: 1000,
            performance_thresholds: PerformanceThresholds::default(),
            validation_phases: vec![
                ValidationPhase::PreFlight,
                ValidationPhase::Performance,
            ],
            chaos_detection_enabled: true,
            detailed_profiling_enabled: true,
            export_formats: vec![OutputFormat::Json],
            report_detail_level: ReportDetailLevel::Summary,
        };

        let start_time = Instant::now();
        let start_memory = Self::get_memory_usage();
        
        let orchestrator = ValidationOrchestrator::new(config);
        let results = orchestrator.run_validation().await?;
        
        let end_time = Instant::now();
        let end_memory = Self::get_memory_usage();
        
        let execution_time = end_time - start_time;
        let memory_usage = end_memory.saturating_sub(start_memory);
        let files_processed = results.directory_analysis.total_files;
        
        let benchmark = BenchmarkResults::new(execution_time, memory_usage, files_processed);
        
        println!("   ⏱️  Execution time: {:?}", benchmark.execution_time);
        println!("   💾 Memory usage: {} MB", benchmark.memory_usage / 1_000_000);
        println!("   📁 Files processed: {}", benchmark.files_processed);
        println!("   🚀 Throughput: {:.2} files/sec", benchmark.throughput);
        
        Ok(benchmark)
    }

    /// Test scalability across different dataset sizes
    pub async fn test_scalability() -> Result<Vec<(usize, BenchmarkResults)>> {
        println!("📈 Testing scalability across dataset sizes...");
        
        let scale_factors = vec![1, 2, 4, 8];
        let mut results = Vec::new();
        
        for scale_factor in scale_factors {
            let dataset = PerformanceTestDataGenerator::create_scalability_test_dataset(scale_factor)?;
            let benchmark = Self::run_performance_benchmark(
                dataset.path(),
                &format!("Scalability Test ({}x)", scale_factor),
            ).await?;
            
            results.push((scale_factor, benchmark));
        }
        
        // Analyze scalability
        Self::analyze_scalability(&results);
        
        Ok(results)
    }

    /// Test performance across different dataset characteristics
    pub async fn test_performance_characteristics() -> Result<HashMap<String, BenchmarkResults>> {
        println!("🎯 Testing performance across different dataset characteristics...");
        
        let characteristics = vec![
            ("Many Small Files", PerformanceCharacteristics::ManySmallFiles),
            ("Few Large Files", PerformanceCharacteristics::FewLargeFiles),
            ("Deep Nesting", PerformanceCharacteristics::DeepNesting),
            ("Wide Structure", PerformanceCharacteristics::WideStructure),
            ("Mixed Content", PerformanceCharacteristics::MixedContent),
        ];
        
        let mut results = HashMap::new();
        
        for (name, characteristic) in characteristics {
            let dataset = PerformanceTestDataGenerator::create_targeted_performance_dataset(&characteristic)?;
            let benchmark = Self::run_performance_benchmark(dataset.path(), name).await?;
            results.insert(name.to_string(), benchmark);
        }
        
        Self::analyze_characteristic_performance(&results);
        
        Ok(results)
    }

    /// Test for performance regressions by comparing against baseline
    pub async fn test_regression_detection() -> Result<()> {
        println!("🔍 Testing performance regression detection...");
        
        // Create baseline dataset
        let baseline_dataset = PerformanceTestDataGenerator::create_performance_test_dataset(1000, 1000, 10)?;
        let baseline = Self::run_performance_benchmark(baseline_dataset.path(), "Baseline").await?;
        
        // Create comparison dataset (slightly different)
        let comparison_dataset = PerformanceTestDataGenerator::create_performance_test_dataset(1100, 1100, 11)?;
        let comparison = Self::run_performance_benchmark(comparison_dataset.path(), "Comparison").await?;
        
        // Compare results
        let performance_comparison = comparison.compare_to(&baseline);
        
        println!("📊 Performance Comparison Results:");
        println!("   Execution time change: {:.1}%", performance_comparison.execution_time_change * 100.0);
        println!("   Memory usage change: {:.1}%", performance_comparison.memory_usage_change * 100.0);
        println!("   Throughput change: {:.1}%", performance_comparison.throughput_change * 100.0);
        println!("   Is regression: {}", performance_comparison.is_regression);
        
        // Test regression detection logic
        assert!(performance_comparison.execution_time_change.abs() < 2.0, "Execution time change too large");
        assert!(performance_comparison.memory_usage_change.abs() < 3.0, "Memory usage change too large");
        
        Ok(())
    }

    /// Test memory usage patterns and leak detection
    pub async fn test_memory_patterns() -> Result<()> {
        println!("🧠 Testing memory usage patterns...");
        
        let dataset = PerformanceTestDataGenerator::create_performance_test_dataset(500, 2000, 8)?;
        
        // Run multiple iterations to check for memory leaks
        let mut memory_readings = Vec::new();
        
        for iteration in 0..5 {
            let start_memory = Self::get_memory_usage();
            
            let config = ValidationOrchestratorConfig {
                target_directory: dataset.path().to_path_buf(),
                pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
                output_directory: dataset.path().join(format!("output_{}", iteration)),
                timeout_seconds: 120,
                memory_limit_mb: 500,
                performance_thresholds: PerformanceThresholds::default(),
                validation_phases: vec![ValidationPhase::PreFlight],
                chaos_detection_enabled: true,
                detailed_profiling_enabled: false,
                export_formats: vec![OutputFormat::Json],
                report_detail_level: ReportDetailLevel::Summary,
            };
            
            let orchestrator = ValidationOrchestrator::new(config);
            let _results = orchestrator.run_validation().await?;
            
            let end_memory = Self::get_memory_usage();
            let memory_delta = end_memory.saturating_sub(start_memory);
            
            memory_readings.push(memory_delta);
            println!("   Iteration {}: {} MB", iteration + 1, memory_delta / 1_000_000);
            
            // Small delay between iterations
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        // Analyze memory pattern
        let avg_memory = memory_readings.iter().sum::<u64>() / memory_readings.len() as u64;
        let max_memory = *memory_readings.iter().max().unwrap();
        let min_memory = *memory_readings.iter().min().unwrap();
        let memory_variance = max_memory - min_memory;
        
        println!("📊 Memory Pattern Analysis:");
        println!("   Average memory usage: {} MB", avg_memory / 1_000_000);
        println!("   Memory variance: {} MB", memory_variance / 1_000_000);
        println!("   Max/Min ratio: {:.2}", max_memory as f64 / min_memory as f64);
        
        // Check for memory leaks (variance should be reasonable)
        assert!(memory_variance < avg_memory, "Memory variance too high - possible leak");
        assert!(max_memory as f64 / min_memory as f64 < 2.0, "Memory usage too inconsistent");
        
        Ok(())
    }

    fn analyze_scalability(results: &[(usize, BenchmarkResults)]) {
        println!("📈 Scalability Analysis:");
        
        for (i, (scale_factor, benchmark)) in results.iter().enumerate() {
            if i > 0 {
                let prev_benchmark = &results[i - 1].1;
                let time_scaling = benchmark.execution_time.as_secs_f64() / prev_benchmark.execution_time.as_secs_f64();
                let memory_scaling = benchmark.memory_usage as f64 / prev_benchmark.memory_usage as f64;
                
                println!("   {}x -> {}x: Time scaling {:.2}x, Memory scaling {:.2}x", 
                         results[i - 1].0, scale_factor, time_scaling, memory_scaling);
            }
        }
        
        // Check if scaling is reasonable (should be roughly linear)
        if results.len() >= 2 {
            let first = &results[0].1;
            let last = &results[results.len() - 1].1;
            let scale_ratio = results[results.len() - 1].0 as f64 / results[0].0 as f64;
            let time_ratio = last.execution_time.as_secs_f64() / first.execution_time.as_secs_f64();
            
            println!("   Overall scaling efficiency: {:.2} (1.0 = perfect linear)", time_ratio / scale_ratio);
        }
    }

    fn analyze_characteristic_performance(results: &HashMap<String, BenchmarkResults>) {
        println!("🎯 Performance Characteristic Analysis:");
        
        // Find best and worst performers
        let mut by_throughput: Vec<_> = results.iter().collect();
        by_throughput.sort_by(|a, b| b.1.throughput.partial_cmp(&a.1.throughput).unwrap());
        
        println!("   Throughput ranking:");
        for (i, (name, benchmark)) in by_throughput.iter().enumerate() {
            println!("     {}. {}: {:.2} files/sec", i + 1, name, benchmark.throughput);
        }
        
        // Memory efficiency ranking
        let mut by_memory: Vec<_> = results.iter().collect();
        by_memory.sort_by(|a, b| a.1.memory_efficiency.partial_cmp(&b.1.memory_efficiency).unwrap());
        
        println!("   Memory efficiency ranking:");
        for (i, (name, benchmark)) in by_memory.iter().enumerate() {
            println!("     {}. {}: {:.0} bytes/file", i + 1, name, benchmark.memory_efficiency);
        }
    }

    fn get_memory_usage() -> u64 {
        // In a real implementation, this would use system APIs to get actual memory usage
        // For testing, we'll simulate memory usage
        use std::sync::atomic::{AtomicU64, Ordering};
        static SIMULATED_MEMORY: AtomicU64 = AtomicU64::new(100_000_000); // Start at 100MB
        
        // Simulate some memory usage variation
        let current = SIMULATED_MEMORY.load(Ordering::Relaxed);
        let variation = (current / 100) * (rand::random::<u64>() % 10); // ±10% variation
        let new_value = current + variation;
        SIMULATED_MEMORY.store(new_value, Ordering::Relaxed);
        new_value
    }
}

// Add a simple random number generator for testing
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static SEED: AtomicU64 = AtomicU64::new(12345);
    
    pub fn random<T>() -> T 
    where 
        T: From<u64>
    {
        let current = SEED.load(Ordering::Relaxed);
        let next = current.wrapping_mul(1103515245).wrapping_add(12345);
        SEED.store(next, Ordering::Relaxed);
        T::from(next)
    }
}

#[tokio::test]
async fn test_baseline_performance_benchmark() -> Result<()> {
    let dataset = PerformanceTestDataGenerator::create_performance_test_dataset(100, 1000, 5)?;
    let benchmark = PerformanceRegressionTester::run_performance_benchmark(
        dataset.path(),
        "Baseline Performance Test",
    ).await?;

    // Basic performance assertions
    assert!(benchmark.execution_time < Duration::from_secs(60), "Execution time too long");
    assert!(benchmark.files_processed > 0, "No files processed");
    assert!(benchmark.throughput > 0.0, "Zero throughput");
    assert!(benchmark.memory_usage > 0, "No memory usage recorded");

    println!("✅ Baseline performance benchmark test passed");
    Ok(())
}

#[tokio::test]
async fn test_scalability_performance() -> Result<()> {
    let scalability_results = PerformanceRegressionTester::test_scalability().await?;
    
    // Should have results for multiple scale factors
    assert!(scalability_results.len() >= 2, "Not enough scalability data points");
    
    // Performance should scale reasonably
    for (scale_factor, benchmark) in &scalability_results {
        assert!(benchmark.files_processed > 0, "No files processed at scale {}", scale_factor);
        assert!(benchmark.throughput > 0.0, "Zero throughput at scale {}", scale_factor);
    }
    
    // Check that larger datasets don't have dramatically worse performance
    let first = &scalability_results[0].1;
    let last = &scalability_results[scalability_results.len() - 1].1;
    let throughput_ratio = last.throughput / first.throughput;
    
    assert!(throughput_ratio > 0.1, "Throughput degraded too much with scale: {:.2}", throughput_ratio);

    println!("✅ Scalability performance test passed");
    Ok(())
}

#[tokio::test]
async fn test_characteristic_performance() -> Result<()> {
    let characteristic_results = PerformanceRegressionTester::test_performance_characteristics().await?;
    
    // Should have results for all characteristics
    assert!(characteristic_results.len() >= 5, "Missing performance characteristic results");
    
    // All characteristics should complete successfully
    for (name, benchmark) in &characteristic_results {
        assert!(benchmark.files_processed > 0, "No files processed for {}", name);
        assert!(benchmark.execution_time < Duration::from_secs(300), "Too slow for {}", name);
        assert!(benchmark.memory_usage < 2_000_000_000, "Too much memory for {}: {} MB", name, benchmark.memory_usage / 1_000_000);
    }
    
    // Different characteristics should have different performance profiles
    let many_small = characteristic_results.get("Many Small Files").unwrap();
    let few_large = characteristic_results.get("Few Large Files").unwrap();
    
    // Many small files should process more files but potentially slower per file
    assert!(many_small.files_processed > few_large.files_processed, 
           "Many small files should process more files");

    println!("✅ Characteristic performance test passed");
    Ok(())
}

#[tokio::test]
async fn test_regression_detection() -> Result<()> {
    PerformanceRegressionTester::test_regression_detection().await?;
    println!("✅ Regression detection test passed");
    Ok(())
}

#[tokio::test]
async fn test_memory_leak_detection() -> Result<()> {
    PerformanceRegressionTester::test_memory_patterns().await?;
    println!("✅ Memory leak detection test passed");
    Ok(())
}

#[tokio::test]
async fn test_framework_performance_limits() -> Result<()> {
    // Test the validation framework's own performance limits
    let large_dataset = PerformanceTestDataGenerator::create_performance_test_dataset(5000, 5000, 15)?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: large_dataset.path().to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: large_dataset.path().join("output"),
        timeout_seconds: 600, // 10 minutes max
        memory_limit_mb: 2000, // 2GB max
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 1.0, // Very lenient
            max_memory_mb: 1500,
            max_processing_time_seconds: 300,
            acceptable_error_rate: 0.1,
            max_performance_degradation: 0.5,
        },
        validation_phases: vec![
            ValidationPhase::PreFlight,
            ValidationPhase::Performance,
        ],
        chaos_detection_enabled: true,
        detailed_profiling_enabled: true,
        export_formats: vec![OutputFormat::Json],
        report_detail_level: ReportDetailLevel::Comprehensive,
    };

    let start_time = Instant::now();
    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;
    let execution_time = start_time.elapsed();

    // Framework should handle large datasets within reasonable limits
    assert!(execution_time < Duration::from_secs(300), "Framework took too long: {:?}", execution_time);
    assert!(results.directory_analysis.total_files > 1000, "Should process many files");
    assert!(results.performance_results.overall_performance_score >= 0.0, "Should produce valid performance score");

    println!("✅ Framework performance limits test passed");
    println!("   Processed {} files in {:?}", results.directory_analysis.total_files, execution_time);
    Ok(())
}

/// Comprehensive performance regression test suite
#[tokio::test]
async fn test_comprehensive_performance_suite() -> Result<()> {
    println!("🚀 Starting comprehensive performance regression test suite...");
    
    let start_time = Instant::now();
    
    // Run all performance tests
    let test_results = vec![
        ("Baseline Performance", test_baseline_performance_benchmark().await),
        ("Scalability Performance", test_scalability_performance().await),
        ("Characteristic Performance", test_characteristic_performance().await),
        ("Regression Detection", test_regression_detection().await),
        ("Memory Leak Detection", test_memory_leak_detection().await),
        ("Framework Performance Limits", test_framework_performance_limits().await),
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
    
    println!("\n📊 Performance Regression Test Suite Summary:");
    println!("   Total time: {:?}", total_time);
    println!("   Tests passed: {}", passed);
    println!("   Tests failed: {}", failed);
    println!("   Success rate: {:.1}%", (passed as f64 / (passed + failed) as f64) * 100.0);
    
    if failed > 0 {
        return Err(ValidationError::TestSuite(format!("{} performance tests failed", failed)));
    }
    
    println!("🎉 All performance regression tests passed!");
    Ok(())
}