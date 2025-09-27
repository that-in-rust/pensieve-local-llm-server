//! Pensieve compatibility and configuration tests
//! 
//! Tests the validation framework against different pensieve versions
//! and configurations to ensure broad compatibility and proper handling
//! of various tool behaviors.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;
use pensieve_validator::*;

/// Mock pensieve configurations for testing different behaviors
#[derive(Debug, Clone)]
pub struct MockPensieveConfig {
    pub version: String,
    pub behavior: PensieveBehavior,
    pub performance_profile: PerformanceProfile,
    pub error_patterns: Vec<ErrorPattern>,
    pub output_format: OutputFormat,
}

#[derive(Debug, Clone)]
pub enum PensieveBehavior {
    Normal,
    Slow,
    MemoryHeavy,
    ErrorProne,
    Crashy,
    InconsistentOutput,
    VerboseLogging,
    SilentMode,
}

#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub files_per_second: f64,
    pub memory_usage_mb: u64,
    pub startup_time_ms: u64,
    pub shutdown_time_ms: u64,
}

#[derive(Debug, Clone)]
pub enum ErrorPattern {
    RandomErrors(f64), // Error rate (0.0 to 1.0)
    SpecificFileTypes(Vec<String>), // File extensions that cause errors
    LargeFileErrors(u64), // Files larger than this size cause errors
    PermissionErrors,
    TimeoutErrors,
    MemoryErrors,
}

/// Pensieve version simulator for testing compatibility
pub struct PensieveVersionSimulator;

impl PensieveVersionSimulator {
    /// Create configurations for different pensieve versions
    pub fn create_version_configs() -> HashMap<String, MockPensieveConfig> {
        let mut configs = HashMap::new();

        // Version 1.0.0 - Original stable version
        configs.insert("1.0.0".to_string(), MockPensieveConfig {
            version: "1.0.0".to_string(),
            behavior: PensieveBehavior::Normal,
            performance_profile: PerformanceProfile {
                files_per_second: 50.0,
                memory_usage_mb: 100,
                startup_time_ms: 500,
                shutdown_time_ms: 200,
            },
            error_patterns: vec![
                ErrorPattern::RandomErrors(0.01), // 1% error rate
            ],
            output_format: OutputFormat::Json,
        });

        // Version 1.1.0 - Performance improvements
        configs.insert("1.1.0".to_string(), MockPensieveConfig {
            version: "1.1.0".to_string(),
            behavior: PensieveBehavior::Normal,
            performance_profile: PerformanceProfile {
                files_per_second: 75.0, // Faster
                memory_usage_mb: 80,    // More efficient
                startup_time_ms: 300,   // Faster startup
                shutdown_time_ms: 150,
            },
            error_patterns: vec![
                ErrorPattern::RandomErrors(0.005), // Lower error rate
            ],
            output_format: OutputFormat::Json,
        });

        // Version 1.2.0 - Added features but slower
        configs.insert("1.2.0".to_string(), MockPensieveConfig {
            version: "1.2.0".to_string(),
            behavior: PensieveBehavior::Slow,
            performance_profile: PerformanceProfile {
                files_per_second: 40.0, // Slower due to new features
                memory_usage_mb: 150,   // More memory for features
                startup_time_ms: 800,   // Slower startup
                shutdown_time_ms: 300,
            },
            error_patterns: vec![
                ErrorPattern::RandomErrors(0.02), // Higher error rate due to complexity
                ErrorPattern::SpecificFileTypes(vec!["bin".to_string(), "exe".to_string()]),
            ],
            output_format: OutputFormat::Json,
        });

        // Version 2.0.0-beta - Major rewrite, unstable
        configs.insert("2.0.0-beta".to_string(), MockPensieveConfig {
            version: "2.0.0-beta".to_string(),
            behavior: PensieveBehavior::ErrorProne,
            performance_profile: PerformanceProfile {
                files_per_second: 100.0, // Much faster when it works
                memory_usage_mb: 200,    // Higher memory usage
                startup_time_ms: 1000,   // Slow startup
                shutdown_time_ms: 100,   // Fast shutdown
            },
            error_patterns: vec![
                ErrorPattern::RandomErrors(0.05), // 5% error rate
                ErrorPattern::LargeFileErrors(10_000_000), // 10MB+ files cause issues
                ErrorPattern::MemoryErrors,
            ],
            output_format: OutputFormat::Json,
        });

        // Version 0.9.0 - Legacy version
        configs.insert("0.9.0".to_string(), MockPensieveConfig {
            version: "0.9.0".to_string(),
            behavior: PensieveBehavior::VerboseLogging,
            performance_profile: PerformanceProfile {
                files_per_second: 30.0, // Slow
                memory_usage_mb: 60,    // Low memory
                startup_time_ms: 200,   // Fast startup
                shutdown_time_ms: 500,  // Slow shutdown
            },
            error_patterns: vec![
                ErrorPattern::RandomErrors(0.03),
                ErrorPattern::PermissionErrors,
            ],
            output_format: OutputFormat::Json,
        });

        // Development version - Latest features, potentially unstable
        configs.insert("dev".to_string(), MockPensieveConfig {
            version: "dev".to_string(),
            behavior: PensieveBehavior::InconsistentOutput,
            performance_profile: PerformanceProfile {
                files_per_second: 80.0,
                memory_usage_mb: 120,
                startup_time_ms: 400,
                shutdown_time_ms: 200,
            },
            error_patterns: vec![
                ErrorPattern::RandomErrors(0.08), // High error rate
                ErrorPattern::TimeoutErrors,
            ],
            output_format: OutputFormat::Json,
        });

        configs
    }

    /// Create different configuration scenarios
    pub fn create_configuration_scenarios() -> HashMap<String, ValidationOrchestratorConfig> {
        let mut scenarios = HashMap::new();
        let base_dir = PathBuf::from("/tmp/test"); // Will be replaced with actual test dirs

        // Minimal configuration
        scenarios.insert("minimal".to_string(), ValidationOrchestratorConfig {
            target_directory: base_dir.clone(),
            pensieve_binary_path: PathBuf::from("mock_pensieve"),
            output_directory: base_dir.join("output_minimal"),
            timeout_seconds: 30,
            memory_limit_mb: 100,
            performance_thresholds: PerformanceThresholds {
                min_files_per_second: 10.0,
                max_memory_mb: 50,
                max_processing_time_seconds: 20,
                acceptable_error_rate: 0.01,
                max_performance_degradation: 0.1,
            },
            validation_phases: vec![ValidationPhase::PreFlight],
            chaos_detection_enabled: false,
            detailed_profiling_enabled: false,
            export_formats: vec![OutputFormat::Json],
            report_detail_level: ReportDetailLevel::Summary,
        });

        // Standard configuration
        scenarios.insert("standard".to_string(), ValidationOrchestratorConfig {
            target_directory: base_dir.clone(),
            pensieve_binary_path: PathBuf::from("mock_pensieve"),
            output_directory: base_dir.join("output_standard"),
            timeout_seconds: 120,
            memory_limit_mb: 500,
            performance_thresholds: PerformanceThresholds {
                min_files_per_second: 20.0,
                max_memory_mb: 300,
                max_processing_time_seconds: 60,
                acceptable_error_rate: 0.05,
                max_performance_degradation: 0.2,
            },
            validation_phases: vec![
                ValidationPhase::PreFlight,
                ValidationPhase::Reliability,
                ValidationPhase::Performance,
            ],
            chaos_detection_enabled: true,
            detailed_profiling_enabled: false,
            export_formats: vec![OutputFormat::Json, OutputFormat::Html],
            report_detail_level: ReportDetailLevel::Detailed,
        });

        // Comprehensive configuration
        scenarios.insert("comprehensive".to_string(), ValidationOrchestratorConfig {
            target_directory: base_dir.clone(),
            pensieve_binary_path: PathBuf::from("mock_pensieve"),
            output_directory: base_dir.join("output_comprehensive"),
            timeout_seconds: 600,
            memory_limit_mb: 2000,
            performance_thresholds: PerformanceThresholds {
                min_files_per_second: 5.0,
                max_memory_mb: 1500,
                max_processing_time_seconds: 300,
                acceptable_error_rate: 0.1,
                max_performance_degradation: 0.5,
            },
            validation_phases: vec![
                ValidationPhase::PreFlight,
                ValidationPhase::Reliability,
                ValidationPhase::Performance,
                ValidationPhase::UserExperience,
                ValidationPhase::ProductionIntelligence,
            ],
            chaos_detection_enabled: true,
            detailed_profiling_enabled: true,
            export_formats: vec![OutputFormat::Json, OutputFormat::Html, OutputFormat::Csv],
            report_detail_level: ReportDetailLevel::Comprehensive,
        });

        // High-performance configuration
        scenarios.insert("high_performance".to_string(), ValidationOrchestratorConfig {
            target_directory: base_dir.clone(),
            pensieve_binary_path: PathBuf::from("mock_pensieve"),
            output_directory: base_dir.join("output_high_perf"),
            timeout_seconds: 60,
            memory_limit_mb: 200,
            performance_thresholds: PerformanceThresholds {
                min_files_per_second: 100.0,
                max_memory_mb: 100,
                max_processing_time_seconds: 30,
                acceptable_error_rate: 0.001,
                max_performance_degradation: 0.05,
            },
            validation_phases: vec![ValidationPhase::Performance],
            chaos_detection_enabled: false,
            detailed_profiling_enabled: true,
            export_formats: vec![OutputFormat::Json],
            report_detail_level: ReportDetailLevel::Summary,
        });

        // Fault-tolerant configuration
        scenarios.insert("fault_tolerant".to_string(), ValidationOrchestratorConfig {
            target_directory: base_dir.clone(),
            pensieve_binary_path: PathBuf::from("mock_pensieve"),
            output_directory: base_dir.join("output_fault_tolerant"),
            timeout_seconds: 300,
            memory_limit_mb: 1000,
            performance_thresholds: PerformanceThresholds {
                min_files_per_second: 1.0,
                max_memory_mb: 800,
                max_processing_time_seconds: 240,
                acceptable_error_rate: 0.3,
                max_performance_degradation: 0.8,
            },
            validation_phases: vec![
                ValidationPhase::PreFlight,
                ValidationPhase::Reliability,
            ],
            chaos_detection_enabled: true,
            detailed_profiling_enabled: false,
            export_formats: vec![OutputFormat::Json],
            report_detail_level: ReportDetailLevel::Detailed,
        });

        scenarios
    }

    /// Simulate pensieve execution with specific behavior
    pub async fn simulate_pensieve_execution(
        config: &MockPensieveConfig,
        target_dir: &Path,
    ) -> Result<PensieveExecutionResults> {
        // Simulate startup time
        tokio::time::sleep(Duration::from_millis(config.performance_profile.startup_time_ms)).await;

        // Count files in target directory
        let file_count = Self::count_files_recursive(target_dir)?;
        
        // Calculate execution time based on performance profile
        let processing_time = Duration::from_secs_f64(file_count as f64 / config.performance_profile.files_per_second);
        
        // Simulate processing time
        tokio::time::sleep(processing_time.min(Duration::from_millis(100))).await; // Cap simulation time

        // Check for errors based on error patterns
        let should_error = Self::should_generate_error(&config.error_patterns, target_dir)?;
        
        if should_error {
            return Err(ValidationError::PensieveExecution {
                exit_code: 1,
                stderr: format!("Simulated error for pensieve version {}", config.version),
            });
        }

        // Generate output based on behavior
        let (stdout, stderr) = Self::generate_output(&config.behavior, file_count, &config.version);

        // Simulate shutdown time
        tokio::time::sleep(Duration::from_millis(config.performance_profile.shutdown_time_ms)).await;

        Ok(PensieveExecutionResults {
            exit_code: 0,
            stdout,
            stderr,
            execution_time: processing_time,
            peak_memory_usage: config.performance_profile.memory_usage_mb * 1_000_000,
            files_processed: file_count,
            database_size: file_count * 1000, // Simulate 1KB per file in database
        })
    }

    fn count_files_recursive(dir: &Path) -> Result<u64> {
        let mut count = 0;
        if dir.is_dir() {
            for entry in fs::read_dir(dir).map_err(ValidationError::FileSystem)? {
                let entry = entry.map_err(ValidationError::FileSystem)?;
                let path = entry.path();
                if path.is_dir() {
                    count += Self::count_files_recursive(&path)?;
                } else {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    fn should_generate_error(error_patterns: &[ErrorPattern], _target_dir: &Path) -> Result<bool> {
        for pattern in error_patterns {
            match pattern {
                ErrorPattern::RandomErrors(rate) => {
                    if rand::random::<f64>() < *rate {
                        return Ok(true);
                    }
                }
                ErrorPattern::MemoryErrors => {
                    // Simulate occasional memory errors
                    if rand::random::<f64>() < 0.02 {
                        return Ok(true);
                    }
                }
                ErrorPattern::TimeoutErrors => {
                    // Simulate occasional timeout errors
                    if rand::random::<f64>() < 0.01 {
                        return Ok(true);
                    }
                }
                _ => {
                    // Other error patterns would be implemented based on file analysis
                    // For now, just simulate low probability
                    if rand::random::<f64>() < 0.005 {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    fn generate_output(behavior: &PensieveBehavior, file_count: u64, version: &str) -> (String, String) {
        let stdout = match behavior {
            PensieveBehavior::Normal => {
                format!("Pensieve {} processed {} files successfully", version, file_count)
            }
            PensieveBehavior::VerboseLogging => {
                format!("Pensieve {} starting...\nScanning directory...\nProcessing files...\nProcessed {} files\nGenerating database...\nComplete!", version, file_count)
            }
            PensieveBehavior::SilentMode => {
                String::new() // No output
            }
            PensieveBehavior::InconsistentOutput => {
                if rand::random::<bool>() {
                    format!("Files processed: {}", file_count)
                } else {
                    format!("Completed processing {} items", file_count)
                }
            }
            _ => {
                format!("Pensieve {} completed with {} files", version, file_count)
            }
        };

        let stderr = match behavior {
            PensieveBehavior::VerboseLogging => {
                "DEBUG: Starting file scan\nDEBUG: Processing file types\nDEBUG: Building database".to_string()
            }
            PensieveBehavior::ErrorProne => {
                "WARNING: Some files could not be processed\nWARNING: Performance may be degraded".to_string()
            }
            _ => String::new(),
        };

        (stdout, stderr)
    }
}

// Simple random number generator for testing
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    static SEED: AtomicU64 = AtomicU64::new(54321);
    
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

/// Test data generator for compatibility tests
pub struct CompatibilityTestDataGenerator;

impl CompatibilityTestDataGenerator {
    pub fn create_standard_test_dataset() -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(ValidationError::FileSystem)?;
        let base_path = temp_dir.path();

        // Create a standard set of files for compatibility testing
        fs::write(base_path.join("README.md"), "# Test Project\n\nThis is a test.")?;
        fs::write(base_path.join("config.json"), r#"{"version": "1.0"}"#)?;
        fs::write(base_path.join("data.txt"), "Sample data file")?;
        fs::write(base_path.join("script.py"), "print('hello world')")?;
        fs::write(base_path.join("binary.dat"), &[0x00, 0x01, 0x02, 0x03])?;

        // Create subdirectory
        let subdir = base_path.join("subdir");
        fs::create_dir_all(&subdir)?;
        fs::write(subdir.join("nested.txt"), "Nested file")?;

        Ok(temp_dir)
    }
}

#[tokio::test]
async fn test_pensieve_version_compatibility() -> Result<()> {
    let test_dataset = CompatibilityTestDataGenerator::create_standard_test_dataset()?;
    let version_configs = PensieveVersionSimulator::create_version_configs();

    println!("🔄 Testing compatibility across pensieve versions...");

    for (version, mock_config) in &version_configs {
        println!("  Testing version: {}", version);

        // Simulate pensieve execution
        let execution_result = PensieveVersionSimulator::simulate_pensieve_execution(
            mock_config,
            test_dataset.path(),
        ).await;

        match execution_result {
            Ok(results) => {
                assert!(results.files_processed > 0, "Version {} processed no files", version);
                assert!(results.execution_time > Duration::ZERO, "Version {} had zero execution time", version);
                println!("    ✅ Version {} - {} files in {:?}", version, results.files_processed, results.execution_time);
            }
            Err(e) => {
                // Some versions are expected to have errors (like beta versions)
                if version.contains("beta") || version == "dev" {
                    println!("    ⚠️  Version {} - Expected error: {:?}", version, e);
                } else {
                    return Err(e);
                }
            }
        }
    }

    println!("✅ Pensieve version compatibility test passed");
    Ok(())
}

#[tokio::test]
async fn test_configuration_scenarios() -> Result<()> {
    let test_dataset = CompatibilityTestDataGenerator::create_standard_test_dataset()?;
    let mut config_scenarios = PensieveVersionSimulator::create_configuration_scenarios();

    println!("⚙️ Testing different configuration scenarios...");

    for (scenario_name, config) in &mut config_scenarios {
        println!("  Testing scenario: {}", scenario_name);

        // Update the target directory to use our test dataset
        config.target_directory = test_dataset.path().to_path_buf();
        config.output_directory = test_dataset.path().join(format!("output_{}", scenario_name));

        let orchestrator = ValidationOrchestrator::new(config.clone());
        let results = orchestrator.run_validation().await?;

        // Basic validation that each scenario produces results
        assert!(results.directory_analysis.total_files > 0, "Scenario {} found no files", scenario_name);
        
        // Check that the scenario-specific settings are reflected in results
        match scenario_name.as_str() {
            "minimal" => {
                // Minimal should have basic analysis only
                assert_eq!(results.validation_phases_completed.len(), 1);
                assert!(results.validation_phases_completed.contains(&ValidationPhase::PreFlight));
            }
            "comprehensive" => {
                // Comprehensive should have all phases
                assert!(results.validation_phases_completed.len() >= 4);
                assert!(results.validation_phases_completed.contains(&ValidationPhase::PreFlight));
                assert!(results.validation_phases_completed.contains(&ValidationPhase::Performance));
            }
            "high_performance" => {
                // High performance should focus on performance
                assert!(results.validation_phases_completed.contains(&ValidationPhase::Performance));
                assert!(results.performance_results.overall_performance_score >= 0.0);
            }
            _ => {
                // Other scenarios should at least complete successfully
                assert!(!results.validation_phases_completed.is_empty());
            }
        }

        println!("    ✅ Scenario {} - {} phases completed", scenario_name, results.validation_phases_completed.len());
    }

    println!("✅ Configuration scenarios test passed");
    Ok(())
}

#[tokio::test]
async fn test_version_performance_comparison() -> Result<()> {
    let test_dataset = CompatibilityTestDataGenerator::create_standard_test_dataset()?;
    let version_configs = PensieveVersionSimulator::create_version_configs();

    println!("📊 Testing performance comparison across versions...");

    let mut version_results = HashMap::new();

    // Test each version
    for (version, mock_config) in &version_configs {
        let execution_result = PensieveVersionSimulator::simulate_pensieve_execution(
            mock_config,
            test_dataset.path(),
        ).await;

        if let Ok(results) = execution_result {
            version_results.insert(version.clone(), results);
        }
    }

    // Compare performance across versions
    if version_results.len() >= 2 {
        let mut versions: Vec<_> = version_results.keys().collect();
        versions.sort();

        println!("  Performance comparison:");
        for version in &versions {
            let results = &version_results[*version];
            let throughput = results.files_processed as f64 / results.execution_time.as_secs_f64();
            println!("    {} - {:.1} files/sec, {} MB memory", 
                     version, throughput, results.peak_memory_usage / 1_000_000);
        }

        // Check for reasonable performance differences
        let throughputs: Vec<f64> = version_results.values()
            .map(|r| r.files_processed as f64 / r.execution_time.as_secs_f64())
            .collect();
        
        let max_throughput = throughputs.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_throughput = throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        // Performance shouldn't vary too wildly between versions
        assert!(max_throughput / min_throughput < 10.0, 
               "Performance varies too much between versions: {:.1}x difference", 
               max_throughput / min_throughput);
    }

    println!("✅ Version performance comparison test passed");
    Ok(())
}

#[tokio::test]
async fn test_error_handling_across_versions() -> Result<()> {
    let test_dataset = CompatibilityTestDataGenerator::create_standard_test_dataset()?;
    let version_configs = PensieveVersionSimulator::create_version_configs();

    println!("🚨 Testing error handling across versions...");

    let mut error_counts = HashMap::new();

    // Run multiple iterations to test error patterns
    for iteration in 0..10 {
        for (version, mock_config) in &version_configs {
            let execution_result = PensieveVersionSimulator::simulate_pensieve_execution(
                mock_config,
                test_dataset.path(),
            ).await;

            let error_count = error_counts.entry(version.clone()).or_insert(0);
            if execution_result.is_err() {
                *error_count += 1;
            }
        }
    }

    // Analyze error patterns
    println!("  Error rates by version:");
    for (version, error_count) in &error_counts {
        let error_rate = *error_count as f64 / 10.0 * 100.0;
        println!("    {}: {:.1}% error rate", version, error_rate);
        
        // Check that error rates are reasonable
        if version.contains("beta") || version == "dev" {
            // Beta/dev versions can have higher error rates
            assert!(error_rate <= 80.0, "Version {} has too high error rate: {:.1}%", version, error_rate);
        } else {
            // Stable versions should have low error rates
            assert!(error_rate <= 30.0, "Version {} has too high error rate: {:.1}%", version, error_rate);
        }
    }

    println!("✅ Error handling across versions test passed");
    Ok(())
}

#[tokio::test]
async fn test_output_format_compatibility() -> Result<()> {
    let test_dataset = CompatibilityTestDataGenerator::create_standard_test_dataset()?;
    
    println!("📄 Testing output format compatibility...");

    // Test different output formats
    let output_formats = vec![
        OutputFormat::Json,
        OutputFormat::Html,
        OutputFormat::Csv,
    ];

    for format in output_formats {
        let config = ValidationOrchestratorConfig {
            target_directory: test_dataset.path().to_path_buf(),
            pensieve_binary_path: PathBuf::from("mock_pensieve"),
            output_directory: test_dataset.path().join(format!("output_{:?}", format)),
            timeout_seconds: 60,
            memory_limit_mb: 200,
            performance_thresholds: PerformanceThresholds::default(),
            validation_phases: vec![ValidationPhase::PreFlight],
            chaos_detection_enabled: true,
            detailed_profiling_enabled: false,
            export_formats: vec![format.clone()],
            report_detail_level: ReportDetailLevel::Summary,
        };

        let orchestrator = ValidationOrchestrator::new(config);
        let results = orchestrator.run_validation().await?;

        // Verify that results are generated regardless of output format
        assert!(results.directory_analysis.total_files > 0, "No files processed for format {:?}", format);
        
        println!("    ✅ Format {:?} - {} files processed", format, results.directory_analysis.total_files);
    }

    println!("✅ Output format compatibility test passed");
    Ok(())
}

#[tokio::test]
async fn test_backward_compatibility() -> Result<()> {
    let test_dataset = CompatibilityTestDataGenerator::create_standard_test_dataset()?;
    
    println!("⏪ Testing backward compatibility...");

    // Test with legacy configuration (simulating older validation framework)
    let legacy_config = ValidationOrchestratorConfig {
        target_directory: test_dataset.path().to_path_buf(),
        pensieve_binary_path: PathBuf::from("mock_pensieve"),
        output_directory: test_dataset.path().join("output_legacy"),
        timeout_seconds: 30, // Shorter timeout like older versions
        memory_limit_mb: 100, // Lower memory limit
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 5.0, // Lower expectations
            max_memory_mb: 50,
            max_processing_time_seconds: 20,
            acceptable_error_rate: 0.1, // Higher tolerance
            max_performance_degradation: 0.5,
        },
        validation_phases: vec![ValidationPhase::PreFlight], // Only basic phase
        chaos_detection_enabled: false, // Feature didn't exist
        detailed_profiling_enabled: false, // Feature didn't exist
        export_formats: vec![OutputFormat::Json], // Only JSON supported
        report_detail_level: ReportDetailLevel::Summary, // Only summary available
    };

    let orchestrator = ValidationOrchestrator::new(legacy_config);
    let results = orchestrator.run_validation().await?;

    // Should still work with legacy configuration
    assert!(results.directory_analysis.total_files > 0);
    assert!(results.validation_phases_completed.contains(&ValidationPhase::PreFlight));
    
    // Should have basic results even with limited configuration
    assert!(results.performance_results.overall_performance_score >= 0.0);

    println!("✅ Backward compatibility test passed");
    Ok(())
}

/// Comprehensive pensieve compatibility test suite
#[tokio::test]
async fn test_comprehensive_pensieve_compatibility() -> Result<()> {
    println!("🔧 Starting comprehensive pensieve compatibility test suite...");
    
    let start_time = std::time::Instant::now();
    
    // Run all compatibility tests
    let test_results = vec![
        ("Version Compatibility", test_pensieve_version_compatibility().await),
        ("Configuration Scenarios", test_configuration_scenarios().await),
        ("Performance Comparison", test_version_performance_comparison().await),
        ("Error Handling", test_error_handling_across_versions().await),
        ("Output Format Compatibility", test_output_format_compatibility().await),
        ("Backward Compatibility", test_backward_compatibility().await),
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
    
    println!("\n📊 Pensieve Compatibility Test Suite Summary:");
    println!("   Total time: {:?}", total_time);
    println!("   Tests passed: {}", passed);
    println!("   Tests failed: {}", failed);
    println!("   Success rate: {:.1}%", (passed as f64 / (passed + failed) as f64) * 100.0);
    
    if failed > 0 {
        return Err(ValidationError::TestSuite(format!("{} compatibility tests failed", failed)));
    }
    
    println!("🎉 All pensieve compatibility tests passed!");
    Ok(())
}