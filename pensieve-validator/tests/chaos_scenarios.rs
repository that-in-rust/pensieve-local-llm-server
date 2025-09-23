//! Chaos scenario integration tests
//! 
//! Tests the validation framework against various chaotic directory structures
//! that represent real-world edge cases and problematic file patterns.

use std::fs;
use std::path::Path;
use tempfile::TempDir;
use pensieve_validator::*;

/// Chaos scenario generator for specific edge cases
pub struct ChaosScenarioGenerator;

impl ChaosScenarioGenerator {
    /// Create a directory with maximum chaos - every possible edge case
    pub fn create_maximum_chaos_directory() -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(ValidationError::FileSystem)?;
        let base_path = temp_dir.path();

        // Unicode nightmare
        Self::create_unicode_chaos(base_path)?;
        
        // Size extremes
        Self::create_size_chaos(base_path)?;
        
        // Extension confusion
        Self::create_extension_chaos(base_path)?;
        
        // Nesting madness
        Self::create_nesting_chaos(base_path)?;
        
        // Content chaos
        Self::create_content_chaos(base_path)?;
        
        #[cfg(unix)]
        Self::create_permission_chaos(base_path)?;

        Ok(temp_dir)
    }

    /// Create a directory that mimics a real messy developer workspace
    pub fn create_developer_workspace_chaos() -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(ValidationError::FileSystem)?;
        let base_path = temp_dir.path();

        // Typical developer mess
        fs::write(base_path.join("README"), "# My Project\n\nThis is my awesome project.")?;
        fs::write(base_path.join("TODO.txt"), "- Fix bug\n- Add feature\n- Write tests")?;
        fs::write(base_path.join("notes"), "Random notes about the project")?;
        fs::write(base_path.join("temp_file"), "Temporary file I forgot to delete")?;
        fs::write(base_path.join("backup_old"), "Old backup file")?;
        fs::write(base_path.join("test_output.log"), "Test output that should be ignored")?;

        // Mixed case and spaces
        fs::write(base_path.join("My Important File.txt"), "Important content")?;
        fs::write(base_path.join("SCREAMING_CASE_FILE.LOG"), "LOUD CONTENT")?;
        fs::write(base_path.join("file-with-dashes.md"), "# Dashed File")?;
        fs::write(base_path.join("file_with_underscores.py"), "print('hello')")?;

        // Build artifacts and cache
        let build_dir = base_path.join("target");
        fs::create_dir_all(&build_dir)?;
        for i in 0..50 {
            fs::write(build_dir.join(format!("artifact_{}.o", i)), vec![0u8; 1000])?;
        }

        let cache_dir = base_path.join(".cache");
        fs::create_dir_all(&cache_dir)?;
        for i in 0..100 {
            fs::write(cache_dir.join(format!("cache_{}.tmp", i)), vec![0u8; 500])?;
        }

        // Version control artifacts
        let git_dir = base_path.join(".git");
        fs::create_dir_all(&git_dir)?;
        fs::write(git_dir.join("config"), "[core]\n\trepositoryformatversion = 0")?;
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/main")?;

        // IDE files
        fs::write(base_path.join(".vscode/settings.json"), r#"{"editor.tabSize": 4}"#)?;
        fs::create_dir_all(base_path.join(".vscode"))?;
        fs::write(base_path.join(".idea/workspace.xml"), "<?xml version=\"1.0\"?><project></project>")?;
        fs::create_dir_all(base_path.join(".idea"))?;

        Ok(temp_dir)
    }

    /// Create a directory that simulates a corrupted filesystem
    pub fn create_corrupted_filesystem_scenario() -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(ValidationError::FileSystem)?;
        let base_path = temp_dir.path();

        // Files with corrupted headers
        fs::write(base_path.join("corrupted.jpg"), &[0xFF, 0xD8, 0xFF, 0x00, 0x00, 0x00])?; // Broken JPEG
        fs::write(base_path.join("corrupted.png"), &[0x89, 0x50, 0x4E, 0x00, 0x00, 0x00])?; // Broken PNG
        fs::write(base_path.join("corrupted.pdf"), &[0x25, 0x50, 0x44, 0x00, 0x00, 0x00])?; // Broken PDF

        // Files with mixed content
        let mut mixed_content = Vec::new();
        mixed_content.extend_from_slice(b"This looks like text but then ");
        mixed_content.extend_from_slice(&[0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD]);
        mixed_content.extend_from_slice(b" and then text again");
        fs::write(base_path.join("mixed_content.txt"), mixed_content)?;

        // Files with invalid UTF-8
        let invalid_utf8 = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA];
        fs::write(base_path.join("invalid_utf8.txt"), invalid_utf8)?;

        // Files with control characters
        let control_chars = "Line 1\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0B\x0C\x0E\x0F\x10Line 2";
        fs::write(base_path.join("control_chars.log"), control_chars)?;

        // Extremely long lines
        let long_line = "x".repeat(10_000_000); // 10MB single line
        fs::write(base_path.join("long_line.txt"), long_line)?;

        // Files with only whitespace variations
        fs::write(base_path.join("spaces.txt"), "     ")?;
        fs::write(base_path.join("tabs.txt"), "\t\t\t\t")?;
        fs::write(base_path.join("newlines.txt"), "\n\n\n\n")?;
        fs::write(base_path.join("mixed_whitespace.txt"), " \t\n\r \t\n\r")?;

        Ok(temp_dir)
    }

    fn create_unicode_chaos(base_path: &Path) -> Result<()> {
        // Every possible Unicode category
        fs::write(base_path.join("العربية.txt"), "Arabic filename")?;
        fs::write(base_path.join("中文测试.md"), "Chinese filename")?;
        fs::write(base_path.join("русский.log"), "Russian filename")?;
        fs::write(base_path.join("日本語.dat"), "Japanese filename")?;
        fs::write(base_path.join("한국어.json"), "Korean filename")?;
        fs::write(base_path.join("ελληνικά.xml"), "Greek filename")?;
        fs::write(base_path.join("עברית.csv"), "Hebrew filename")?;
        fs::write(base_path.join("हिन्दी.py"), "Hindi filename")?;
        fs::write(base_path.join("ไทย.js"), "Thai filename")?;
        
        // Emoji chaos
        fs::write(base_path.join("🚀🎉🔥.txt"), "Emoji filename")?;
        fs::write(base_path.join("👨‍💻👩‍💻.md"), "Complex emoji filename")?;
        fs::write(base_path.join("🏳️‍🌈🏳️‍⚧️.log"), "Flag emoji filename")?;
        
        // Combining characters
        fs::write(base_path.join("café́́́́.txt"), "Multiple combining accents")?;
        fs::write(base_path.join("naïve̊̊̊.md"), "Stacked diacritics")?;
        
        // Right-to-left text
        fs::write(base_path.join("مرحبا‎.txt"), "RTL filename")?;
        fs::write(base_path.join("שלום‎.log"), "RTL Hebrew filename")?;
        
        // Zero-width characters
        fs::write(base_path.join("zero​width.txt"), "Zero-width space in filename")?;
        fs::write(base_path.join("invisible‌chars.md"), "Zero-width non-joiner")?;

        Ok(())
    }

    fn create_size_chaos(base_path: &Path) -> Result<()> {
        // Zero-byte files
        for i in 0..10 {
            fs::write(base_path.join(format!("empty_{}.txt", i)), "")?;
        }

        // Tiny files (1 byte each)
        for i in 0..20 {
            fs::write(base_path.join(format!("tiny_{}.dat", i)), "x")?;
        }

        // Medium files (1MB each)
        let medium_content = "x".repeat(1_000_000);
        for i in 0..5 {
            fs::write(base_path.join(format!("medium_{}.txt", i)), &medium_content)?;
        }

        // Large files (50MB each)
        let large_content = vec![0xAB; 50_000_000];
        fs::write(base_path.join("large_1.bin"), &large_content)?;
        fs::write(base_path.join("large_2.dat"), &large_content)?;

        // Extremely large file (200MB)
        let huge_content = vec![0xCD; 200_000_000];
        fs::write(base_path.join("huge.bin"), huge_content)?;

        Ok(())
    }

    fn create_extension_chaos(base_path: &Path) -> Result<()> {
        // Files with wrong extensions
        fs::write(base_path.join("image.txt"), &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])?; // PNG as TXT
        fs::write(base_path.join("text.jpg"), "This is actually text content")?; // Text as JPG
        fs::write(base_path.join("binary.json"), &[0x00, 0x01, 0x02, 0x03, 0xFF])?; // Binary as JSON
        fs::write(base_path.join("executable.md"), &[0x7F, 0x45, 0x4C, 0x46])?; // ELF as Markdown

        // Multiple extensions
        fs::write(base_path.join("file.tar.gz.bak.old"), "Multiple extensions")?;
        fs::write(base_path.join("data.json.backup.2024"), "Backup with date")?;
        fs::write(base_path.join("script.py.orig.save"), "Original save file")?;

        // Unusual extensions
        fs::write(base_path.join("file.xyz123"), "Unusual extension")?;
        fs::write(base_path.join("data.qqq"), "Made-up extension")?;
        fs::write(base_path.join("test."), "Extension with just dot")?;

        // Case variations
        fs::write(base_path.join("FILE.TXT"), "Uppercase extension")?;
        fs::write(base_path.join("Data.JSON"), "Mixed case extension")?;
        fs::write(base_path.join("script.PY"), "Uppercase Python")?;

        Ok(())
    }

    fn create_nesting_chaos(base_path: &Path) -> Result<()> {
        // Extremely deep nesting (30 levels)
        let mut current_path = base_path.to_path_buf();
        for level in 1..=30 {
            current_path = current_path.join(format!("level_{}", level));
            fs::create_dir_all(&current_path)?;
            
            // Add files at various levels
            if level % 5 == 0 {
                fs::write(current_path.join(format!("file_at_{}.txt", level)), 
                         format!("File at level {}", level))?;
            }
        }

        // Wide structure (100 subdirectories)
        let wide_dir = base_path.join("wide");
        fs::create_dir_all(&wide_dir)?;
        for i in 0..100 {
            let subdir = wide_dir.join(format!("sub_{:03}", i));
            fs::create_dir_all(&subdir)?;
            fs::write(subdir.join("file.txt"), format!("File in sub {}", i))?;
        }

        // Mixed deep and wide
        let mixed_dir = base_path.join("mixed");
        fs::create_dir_all(&mixed_dir)?;
        for i in 0..10 {
            let mut deep_path = mixed_dir.join(format!("branch_{}", i));
            for j in 0..10 {
                deep_path = deep_path.join(format!("level_{}", j));
                fs::create_dir_all(&deep_path)?;
                fs::write(deep_path.join("nested.txt"), format!("Branch {} Level {}", i, j))?;
            }
        }

        Ok(())
    }

    fn create_content_chaos(base_path: &Path) -> Result<()> {
        // Files with problematic content patterns
        
        // Null bytes throughout
        let null_content = vec![0x00; 1000];
        fs::write(base_path.join("null_bytes.dat"), null_content)?;

        // Mixed line endings
        fs::write(base_path.join("mixed_endings.txt"), "Unix\nWindows\r\nMac\rMixed\r\n\r\r\n\n")?;

        // Very long lines
        let long_line = "This is an extremely long line that goes on and on and on ".repeat(10000);
        fs::write(base_path.join("long_lines.txt"), long_line)?;

        // Binary data disguised as text
        let mut fake_text = Vec::new();
        fake_text.extend_from_slice(b"This looks like text\n");
        fake_text.extend_from_slice(&[0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD]);
        fake_text.extend_from_slice(b"\nBut contains binary data\n");
        fs::write(base_path.join("fake_text.txt"), fake_text)?;

        // Extremely repetitive content
        let repetitive = "AAAAAAAAAA".repeat(100000);
        fs::write(base_path.join("repetitive.txt"), repetitive)?;

        // Random binary data
        let mut random_data = Vec::new();
        for i in 0..10000 {
            random_data.push((i % 256) as u8);
        }
        fs::write(base_path.join("random.bin"), random_data)?;

        // Files with only special characters
        fs::write(base_path.join("special_chars.txt"), "!@#$%^&*()_+-=[]{}|;':\",./<>?")?;
        fs::write(base_path.join("brackets.txt"), "(((((((((((((((((((((((((((((((((((((((((((")?;
        fs::write(base_path.join("quotes.txt"), "\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"")?;

        Ok(())
    }

    #[cfg(unix)]
    fn create_permission_chaos(base_path: &Path) -> Result<()> {
        use std::os::unix::fs::{symlink, PermissionsExt};

        // Files with various permission combinations
        let perms_dir = base_path.join("permissions");
        fs::create_dir_all(&perms_dir)?;

        // No permissions
        let no_perms = perms_dir.join("no_permissions.txt");
        fs::write(&no_perms, "No permissions")?;
        let mut perms = fs::metadata(&no_perms)?.permissions();
        perms.set_mode(0o000);
        fs::set_permissions(&no_perms, perms)?;

        // Read-only
        let read_only = perms_dir.join("read_only.txt");
        fs::write(&read_only, "Read only")?;
        let mut perms = fs::metadata(&read_only)?.permissions();
        perms.set_mode(0o444);
        fs::set_permissions(&read_only, perms)?;

        // Execute-only
        let exec_only = perms_dir.join("execute_only.txt");
        fs::write(&exec_only, "Execute only")?;
        let mut perms = fs::metadata(&exec_only)?.permissions();
        perms.set_mode(0o111);
        fs::set_permissions(&exec_only, perms)?;

        // Symlink chaos
        let symlink_dir = base_path.join("symlinks");
        fs::create_dir_all(&symlink_dir)?;

        // Broken symlinks
        symlink("nonexistent.txt", symlink_dir.join("broken_link"))?;
        symlink("../../../nonexistent/path.txt", symlink_dir.join("deeply_broken"))?;

        // Circular symlinks
        symlink("circular_b", symlink_dir.join("circular_a"))?;
        symlink("circular_c", symlink_dir.join("circular_b"))?;
        symlink("circular_a", symlink_dir.join("circular_c"))?;

        // Long symlink chains
        fs::write(symlink_dir.join("target.txt"), "Final target")?;
        for i in 1..=10 {
            let source = if i == 1 { "target.txt".to_string() } else { format!("link_{}.txt", i - 1) };
            symlink(source, symlink_dir.join(format!("link_{}.txt", i)))?;
        }

        Ok(())
    }
}

#[tokio::test]
async fn test_maximum_chaos_scenario() -> Result<()> {
    let chaos_dir = ChaosScenarioGenerator::create_maximum_chaos_directory()?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: chaos_dir.path().to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: chaos_dir.path().join("output"),
        timeout_seconds: 300,
        memory_limit_mb: 1000,
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 1.0, // Very lenient for chaos
            max_memory_mb: 800,
            max_processing_time_seconds: 240,
            acceptable_error_rate: 0.3, // High error rate expected
            max_performance_degradation: 0.8,
        },
        validation_phases: vec![
            ValidationPhase::PreFlight,
            ValidationPhase::Reliability,
            ValidationPhase::Performance,
        ],
        chaos_detection_enabled: true,
        detailed_profiling_enabled: true,
        export_formats: vec![OutputFormat::Json],
        report_detail_level: ReportDetailLevel::Comprehensive,
    };

    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;

    // Maximum chaos should be detected
    assert!(results.chaos_report.total_chaos_files() > 50);
    assert!(results.directory_analysis.chaos_indicators.chaos_score > 0.8);
    assert!(results.directory_analysis.chaos_indicators.chaos_percentage > 70.0);

    // Should definitely not be production ready
    assert!(matches!(
        results.production_readiness_assessment.overall_readiness,
        ProductionReadiness::NotReady
    ));

    // Should have many critical issues
    assert!(results.improvement_roadmap.critical_blockers.len() > 5);

    println!("✅ Maximum chaos scenario test passed");
    println!("   Chaos score: {:.2}", results.directory_analysis.chaos_indicators.chaos_score);
    println!("   Chaos files: {}", results.chaos_report.total_chaos_files());
    println!("   Critical blockers: {}", results.improvement_roadmap.critical_blockers.len());

    Ok(())
}

#[tokio::test]
async fn test_developer_workspace_scenario() -> Result<()> {
    let workspace_dir = ChaosScenarioGenerator::create_developer_workspace_chaos()?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: workspace_dir.path().to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: workspace_dir.path().join("output"),
        timeout_seconds: 120,
        memory_limit_mb: 500,
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 10.0,
            max_memory_mb: 300,
            max_processing_time_seconds: 60,
            acceptable_error_rate: 0.1,
            max_performance_degradation: 0.3,
        },
        validation_phases: vec![
            ValidationPhase::PreFlight,
            ValidationPhase::Performance,
            ValidationPhase::UserExperience,
        ],
        chaos_detection_enabled: true,
        detailed_profiling_enabled: false,
        export_formats: vec![OutputFormat::Json],
        report_detail_level: ReportDetailLevel::Detailed,
    };

    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;

    // Developer workspace should have moderate chaos
    assert!(results.chaos_report.total_chaos_files() > 10);
    assert!(results.chaos_report.total_chaos_files() < 100);
    assert!(results.directory_analysis.chaos_indicators.chaos_score > 0.2);
    assert!(results.directory_analysis.chaos_indicators.chaos_score < 0.7);

    // Should detect build artifacts and cache files
    assert!(results.directory_analysis.total_files > 100); // Many small files
    assert!(results.directory_analysis.file_type_distribution.len() > 5); // Various file types

    // Performance should be reasonable
    assert!(results.performance_results.overall_performance_score > 0.3);

    println!("✅ Developer workspace scenario test passed");
    println!("   Total files: {}", results.directory_analysis.total_files);
    println!("   File types: {}", results.directory_analysis.file_type_distribution.len());
    println!("   Chaos score: {:.2}", results.directory_analysis.chaos_indicators.chaos_score);

    Ok(())
}

#[tokio::test]
async fn test_corrupted_filesystem_scenario() -> Result<()> {
    let corrupted_dir = ChaosScenarioGenerator::create_corrupted_filesystem_scenario()?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: corrupted_dir.path().to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: corrupted_dir.path().join("output"),
        timeout_seconds: 180,
        memory_limit_mb: 800,
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 5.0, // Slower due to corruption handling
            max_memory_mb: 600,
            max_processing_time_seconds: 120,
            acceptable_error_rate: 0.5, // High error rate expected
            max_performance_degradation: 0.6,
        },
        validation_phases: vec![
            ValidationPhase::PreFlight,
            ValidationPhase::Reliability,
        ],
        chaos_detection_enabled: true,
        detailed_profiling_enabled: true,
        export_formats: vec![OutputFormat::Json],
        report_detail_level: ReportDetailLevel::Comprehensive,
    };

    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;

    // Should detect many problematic files
    assert!(results.chaos_report.misleading_extensions.len() > 0);
    assert!(results.chaos_report.corrupted_files.len() > 0);
    assert!(results.directory_analysis.chaos_indicators.chaos_score > 0.6);

    // Reliability should be a major concern
    assert!(results.reliability_results.overall_reliability_score < 0.7);

    // Should have specific recommendations for handling corruption
    assert!(!results.improvement_roadmap.high_priority_improvements.is_empty());

    println!("✅ Corrupted filesystem scenario test passed");
    println!("   Corrupted files: {}", results.chaos_report.corrupted_files.len());
    println!("   Misleading extensions: {}", results.chaos_report.misleading_extensions.len());
    println!("   Reliability score: {:.2}", results.reliability_results.overall_reliability_score);

    Ok(())
}

#[tokio::test]
async fn test_unicode_handling_robustness() -> Result<()> {
    let temp_dir = tempfile::TempDir::new().map_err(ValidationError::FileSystem)?;
    let base_path = temp_dir.path();
    
    // Create files with every Unicode category we can think of
    ChaosScenarioGenerator::create_unicode_chaos(base_path)?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: base_path.to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: base_path.join("output"),
        timeout_seconds: 60,
        memory_limit_mb: 300,
        performance_thresholds: PerformanceThresholds::default(),
        validation_phases: vec![ValidationPhase::PreFlight],
        chaos_detection_enabled: true,
        detailed_profiling_enabled: false,
        export_formats: vec![OutputFormat::Json],
        report_detail_level: ReportDetailLevel::Summary,
    };

    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;

    // Should detect Unicode filenames
    assert!(results.chaos_report.unicode_filenames.len() > 10);
    
    // Should handle all files without crashing
    assert!(results.directory_analysis.total_files > 15);
    
    // Unicode should contribute to chaos score
    assert!(results.directory_analysis.chaos_indicators.chaos_score > 0.3);

    println!("✅ Unicode handling robustness test passed");
    println!("   Unicode files detected: {}", results.chaos_report.unicode_filenames.len());
    println!("   Total files processed: {}", results.directory_analysis.total_files);

    Ok(())
}

#[tokio::test]
async fn test_size_extremes_handling() -> Result<()> {
    let temp_dir = tempfile::TempDir::new().map_err(ValidationError::FileSystem)?;
    let base_path = temp_dir.path();
    
    // Create files with extreme sizes
    ChaosScenarioGenerator::create_size_chaos(base_path)?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: base_path.to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: base_path.join("output"),
        timeout_seconds: 180,
        memory_limit_mb: 1000,
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 1.0, // Very slow due to large files
            max_memory_mb: 800,
            max_processing_time_seconds: 150,
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
        report_detail_level: ReportDetailLevel::Detailed,
    };

    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;

    // Should detect size extremes
    assert!(results.chaos_report.zero_byte_files.len() >= 10);
    assert!(results.chaos_report.extremely_large_files.len() >= 3);
    
    // Size distribution should show extremes
    assert!(results.directory_analysis.size_distribution.zero_byte_files >= 10);
    assert!(results.directory_analysis.size_distribution.very_large_files >= 3);
    assert!(results.directory_analysis.size_distribution.largest_file_size >= 200_000_000);

    // Performance should be impacted by large files
    assert!(results.performance_results.overall_performance_score < 0.8);

    println!("✅ Size extremes handling test passed");
    println!("   Zero-byte files: {}", results.chaos_report.zero_byte_files.len());
    println!("   Large files: {}", results.chaos_report.extremely_large_files.len());
    println!("   Largest file: {} MB", results.directory_analysis.size_distribution.largest_file_size / 1_000_000);

    Ok(())
}

#[tokio::test]
async fn test_nesting_extremes_handling() -> Result<()> {
    let temp_dir = tempfile::TempDir::new().map_err(ValidationError::FileSystem)?;
    let base_path = temp_dir.path();
    
    // Create extreme nesting scenarios
    ChaosScenarioGenerator::create_nesting_chaos(base_path)?;
    
    let config = ValidationOrchestratorConfig {
        target_directory: base_path.to_path_buf(),
        pensieve_binary_path: std::path::PathBuf::from("mock_pensieve"),
        output_directory: base_path.join("output"),
        timeout_seconds: 120,
        memory_limit_mb: 400,
        performance_thresholds: PerformanceThresholds {
            min_files_per_second: 5.0,
            max_memory_mb: 300,
            max_processing_time_seconds: 90,
            acceptable_error_rate: 0.1,
            max_performance_degradation: 0.4,
        },
        validation_phases: vec![
            ValidationPhase::PreFlight,
            ValidationPhase::Performance,
        ],
        chaos_detection_enabled: true,
        detailed_profiling_enabled: false,
        export_formats: vec![OutputFormat::Json],
        report_detail_level: ReportDetailLevel::Summary,
    };

    let orchestrator = ValidationOrchestrator::new(config);
    let results = orchestrator.run_validation().await?;

    // Should detect deep nesting
    assert!(results.chaos_report.deep_nesting.len() > 0);
    assert!(results.directory_analysis.depth_analysis.max_depth >= 30);
    
    // Should handle many directories
    assert!(results.directory_analysis.total_directories > 100);
    assert!(results.directory_analysis.total_files > 100);

    // Nesting should contribute to chaos
    assert!(results.directory_analysis.chaos_indicators.chaos_score > 0.4);

    println!("✅ Nesting extremes handling test passed");
    println!("   Max depth: {}", results.directory_analysis.depth_analysis.max_depth);
    println!("   Total directories: {}", results.directory_analysis.total_directories);
    println!("   Deep nesting files: {}", results.chaos_report.deep_nesting.len());

    Ok(())
}

/// Comprehensive chaos scenario test suite
#[tokio::test]
async fn test_all_chaos_scenarios() -> Result<()> {
    println!("🌪️ Starting comprehensive chaos scenario test suite...");
    
    let start_time = std::time::Instant::now();
    
    // Run all chaos scenario tests
    let test_results = vec![
        ("Maximum Chaos", test_maximum_chaos_scenario().await),
        ("Developer Workspace", test_developer_workspace_scenario().await),
        ("Corrupted Filesystem", test_corrupted_filesystem_scenario().await),
        ("Unicode Handling", test_unicode_handling_robustness().await),
        ("Size Extremes", test_size_extremes_handling().await),
        ("Nesting Extremes", test_nesting_extremes_handling().await),
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
    
    println!("\n📊 Chaos Scenario Test Suite Summary:");
    println!("   Total time: {:?}", total_time);
    println!("   Tests passed: {}", passed);
    println!("   Tests failed: {}", failed);
    println!("   Success rate: {:.1}%", (passed as f64 / (passed + failed) as f64) * 100.0);
    
    if failed > 0 {
        return Err(ValidationError::TestSuite(format!("{} chaos scenario tests failed", failed)));
    }
    
    println!("🎉 All chaos scenario tests passed!");
    Ok(())
}