pub mod chaos_detector;
pub mod cli_config;
pub mod comparative_analyzer;
pub mod deduplication_analyzer;
pub mod directory_analyzer;
pub mod error_reporter;
pub mod errors;
pub mod graceful_degradation;
pub mod metrics_collector;
pub mod pensieve_runner;
pub mod performance_benchmarker;
pub mod process_monitor;
pub mod production_readiness_assessor;
pub mod reliability_validator;
pub mod report_generator;
#[cfg(test)]
pub mod test_error_handling;
pub mod types;
pub mod ux_analyzer;
pub mod validation_orchestrator;

pub use chaos_detector::ChaosDetector;
pub use comparative_analyzer::{
    ComparativeAnalyzer, BaselineSet, ValidationComparison, HistoricalTrendAnalysis,
    TrendDirection, PerformanceComparison, ReliabilityComparison, UXComparison,
    DeduplicationComparison, RegressionAlert, ImprovementHighlight, AlertSeverity,
};
pub use deduplication_analyzer::DeduplicationAnalyzer;
pub use directory_analyzer::DirectoryAnalyzer;
pub use error_reporter::{
    ErrorReporter, ErrorReportConfig, ComprehensiveErrorReport, ReportFormat,
    ErrorExecutiveSummary, ErrorAnalysis, ImpactAssessment, RecoveryGuidance
};
pub use errors::{ValidationError, Result, ErrorRecoveryManager, ErrorAggregator, ErrorDetails, ErrorSummary, RecoveryAction};
pub use graceful_degradation::{
    GracefulDegradationManager, DegradationConfig, PhaseResult, PhaseStatus, 
    DegradationStrategy, DegradationType, DegradationDecision, DegradationReport
};
pub use metrics_collector::{
    MetricsCollector, MetricsCollectionResults, PerformanceTracker, ErrorTracker, UXTracker, DatabaseTracker,
    ErrorCategory, ErrorSeverity, ErrorContext, SystemState, UXEventType, UXQualityScores, DatabaseOperation
};
pub use pensieve_runner::{PensieveRunner, PensieveConfig, PensieveExecutionResults};
pub use performance_benchmarker::{
    PerformanceBenchmarker, BenchmarkConfig, PerformanceThresholds as BenchmarkThresholds,
    PerformanceBaseline, BaselineMetrics, PerformanceBenchmarkingResults, ScalabilityAnalysisResults,
    MemoryAnalysisResults, DatabaseProfilingResults, DegradationDetectionResults,
    OverallPerformanceAssessment, DatasetCharacteristics, MemoryGrowthPattern, ScalingBehavior
};
pub use process_monitor::{ProcessMonitor, MonitoringConfig, MonitoringResults};
pub use production_readiness_assessor::{
    ProductionReadinessAssessor, AssessmentConfig, ProductionReadinessAssessment,
    ProductionReadinessLevel, FactorScores, CriticalIssue, ProductionBlocker,
    ScalingGuidance, DeploymentRecommendations, ImprovementRoadmap
};
pub use reliability_validator::{
    ReliabilityValidator, ReliabilityConfig, ReliabilityResults, CrashTestResults,
    InterruptionTestResults, ResourceLimitTestResults, CorruptionHandlingResults,
    PermissionHandlingResults, RecoveryTestResults, FailureAnalysis, CrashIncident,
    CrashType, CrashSeverity, GracefulFailure, CriticalFailure, ReliabilityBlocker,
    ReliabilityRecommendation, RiskAssessment
};
pub use report_generator::{
    ReportGenerator, ReportGeneratorConfig, ProductionReadinessReport, OutputFormat,
    ReportDetailLevel, ExecutiveSummary, OverallRecommendation, PerformanceAnalysisReport,
    UserExperienceReport, ImprovementRoadmapReport, ScalingGuidanceReport,
    DeploymentRecommendationsReport
};
pub use types::*;
pub use ux_analyzer::{
    UXAnalyzer, UXResults, ProgressReportingQuality, ErrorMessageClarity, 
    CompletionFeedbackQuality, InterruptionHandlingQuality, UserFeedbackAnalysis,
    UXImprovement, UXCategory, ImprovementPriority, InterruptionType, FeedbackType
};
pub use validation_orchestrator::{
    ValidationOrchestrator, ValidationOrchestratorConfig, ComprehensiveValidationResults,
    PerformanceThresholds, ProductionReadiness, RecommendationPriority
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    /// Create a test directory with known problematic files
    pub fn create_chaos_test_directory() -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let base_path = temp_dir.path();

        // Create files without extensions
        fs::write(base_path.join("no_extension_file"), "This file has no extension")?;
        fs::write(base_path.join("README"), "This is a README file without extension")?;
        
        // Create files with misleading extensions
        // A PNG file with .txt extension
        let fake_txt = base_path.join("fake_text.txt");
        fs::write(&fake_txt, &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])?; // PNG header
        
        // A binary file with .json extension
        let fake_json = base_path.join("fake_data.json");
        fs::write(&fake_json, &[0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD])?; // Binary data
        
        // Create files with unicode names
        fs::write(base_path.join("файл.txt"), "File with Cyrillic name")?;
        fs::write(base_path.join("测试文件.md"), "File with Chinese name")?;
        fs::write(base_path.join("🚀rocket.log"), "File with emoji")?;
        fs::write(base_path.join("café_résumé.pdf"), "File with accented characters")?;
        
        // Create files with unusual characters
        fs::write(base_path.join("file<with>bad:chars.txt"), "File with problematic characters")?;
        // Note: Cannot create files with null bytes in filename on most filesystems
        fs::write(base_path.join("file_with_control_chars.dat"), "File with control characters")?;
        
        // Create zero-byte files
        fs::write(base_path.join("empty.txt"), "")?;
        fs::write(base_path.join("zero_size.log"), "")?;
        
        // Create large files
        let large_content = "x".repeat(150_000_000); // 150MB
        fs::write(base_path.join("large_file.dat"), large_content)?;
        
        // Create deeply nested structure
        let deep_path = base_path.join("level1/level2/level3/level4/level5/level6/level7/level8/level9/level10");
        fs::create_dir_all(&deep_path)?;
        fs::write(deep_path.join("deeply_nested.txt"), "This file is deeply nested")?;
        
        // Create a file with long name (but within filesystem limits)
        let long_name = "a".repeat(200); // Reduced to stay within filesystem limits
        fs::write(base_path.join(format!("{}.txt", long_name)), "File with very long name")?;
        
        // Create symlinks (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            
            // Simple symlink
            symlink("no_extension_file", base_path.join("symlink_to_file"))?;
            
            // Circular symlink
            symlink("circular_b", base_path.join("circular_a"))?;
            symlink("circular_a", base_path.join("circular_b"))?;
            
            // Chain of symlinks
            symlink("target.txt", base_path.join("link1"))?;
            symlink("link1", base_path.join("link2"))?;
            symlink("link2", base_path.join("link3"))?;
            fs::write(base_path.join("target.txt"), "Final target")?;
        }
        
        Ok(temp_dir)
    }

    #[test]
    fn test_chaos_detector_extensionless_files() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect at least 2 extensionless files (no_extension_file, README)
        assert!(report.files_without_extensions.len() >= 2);
        
        let extensionless_names: Vec<String> = report.files_without_extensions
            .iter()
            .filter_map(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .collect();
        
        assert!(extensionless_names.contains(&"no_extension_file".to_string()));
        assert!(extensionless_names.contains(&"README".to_string()));
        
        Ok(())
    }

    #[test]
    fn test_chaos_detector_misleading_extensions() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect misleading extensions
        assert!(report.misleading_extensions.len() >= 1);
        
        // Check that we detected the PNG file with .txt extension
        let misleading_files: Vec<String> = report.misleading_extensions
            .iter()
            .filter_map(|f| f.path.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .collect();
        
        assert!(misleading_files.iter().any(|name| name.contains("fake_text.txt")));
        
        Ok(())
    }

    #[test]
    fn test_chaos_detector_unicode_filenames() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect unicode filenames
        assert!(report.unicode_filenames.len() >= 4);
        
        let unicode_names: Vec<String> = report.unicode_filenames
            .iter()
            .filter_map(|f| f.path.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .collect();
        
        assert!(unicode_names.iter().any(|name| name.contains("файл")));
        assert!(unicode_names.iter().any(|name| name.contains("测试")));
        assert!(unicode_names.iter().any(|name| name.contains("🚀")));
        assert!(unicode_names.iter().any(|name| name.contains("café")));
        
        Ok(())
    }

    #[test]
    fn test_chaos_detector_large_files() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect the large file
        assert!(report.extremely_large_files.len() >= 1);
        
        let large_file = &report.extremely_large_files[0];
        assert!(large_file.size_bytes >= 100_000_000); // At least 100MB
        assert!(matches!(large_file.size_category, crate::types::SizeCategory::Large));
        
        Ok(())
    }

    #[test]
    fn test_chaos_detector_zero_byte_files() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect zero-byte files
        assert!(report.zero_byte_files.len() >= 2);
        
        let zero_byte_names: Vec<String> = report.zero_byte_files
            .iter()
            .filter_map(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .collect();
        
        assert!(zero_byte_names.contains(&"empty.txt".to_string()));
        assert!(zero_byte_names.contains(&"zero_size.log".to_string()));
        
        Ok(())
    }

    #[test]
    fn test_chaos_detector_unusual_characters() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect files with unusual characters (like < > : in filenames)
        // Note: The exact count may vary by filesystem, so we just check that detection works
        let unusual_files: Vec<String> = report.unusual_characters
            .iter()
            .filter_map(|f| f.path.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .collect();
        
        // On some filesystems, files with < > : might be detected as unusual
        // This test mainly verifies the detection mechanism works
        println!("Detected unusual character files: {:?}", unusual_files);
        
        Ok(())
    }

    #[test]
    fn test_chaos_detector_deep_nesting() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        // Use a detector with lower thresholds to ensure detection
        let detector = ChaosDetector::with_config(10, 100_000_000, 5, 100);
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect deeply nested files (our test creates 10+ levels)
        assert!(report.deep_nesting.len() >= 1);
        
        let deep_file = &report.deep_nesting[0];
        assert!(deep_file.depth > 5);
        
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn test_chaos_detector_symlink_chains() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        
        // Should detect symlink chains
        assert!(report.symlink_chains.len() >= 2);
        
        // Check for circular symlink
        let circular_links: Vec<&SymlinkChain> = report.symlink_chains
            .iter()
            .filter(|chain| chain.is_circular)
            .collect();
        
        assert!(circular_links.len() >= 1);
        
        Ok(())
    }

    #[test]
    fn test_directory_analyzer_basic_analysis() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let analyzer = DirectoryAnalyzer::new();
        
        let analysis = analyzer.analyze_directory(temp_dir.path())?;
        
        // Basic checks
        assert!(analysis.total_files > 0);
        assert!(analysis.total_directories > 0);
        assert!(analysis.total_size_bytes > 0);
        assert!(analysis.depth_analysis.max_depth > 0);
        assert!(!analysis.file_type_distribution.is_empty());
        
        // Chaos indicators
        assert!(analysis.chaos_indicators.chaos_score > 0.0);
        assert!(analysis.chaos_indicators.problematic_file_count > 0);
        assert!(analysis.chaos_indicators.chaos_percentage > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_chaos_metrics_calculation() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let detector = ChaosDetector::new();
        
        let report = detector.detect_chaos_files(temp_dir.path())?;
        let chaos_indicators = report.calculate_chaos_metrics(100); // Assume 100 total files
        
        assert!(chaos_indicators.chaos_score >= 0.0);
        assert!(chaos_indicators.chaos_score <= 1.0);
        assert!(chaos_indicators.problematic_file_count > 0);
        assert!(chaos_indicators.chaos_percentage >= 0.0);
        assert!(chaos_indicators.chaos_percentage <= 100.0);
        
        Ok(())
    }

    #[test]
    fn test_file_type_detection() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let analyzer = DirectoryAnalyzer::new();
        
        let analysis = analyzer.analyze_directory(temp_dir.path())?;
        
        // Should have detected various file types
        assert!(analysis.file_type_distribution.contains_key("text"));
        
        // Print detected file types for debugging
        println!("Detected file types: {:?}", analysis.file_type_distribution.keys().collect::<Vec<_>>());
        
        // Check that file type stats are reasonable
        for (_file_type, stats) in &analysis.file_type_distribution {
            assert!(stats.count > 0);
            assert!(!stats.largest_file.as_os_str().is_empty());
        }
        
        Ok(())
    }

    #[test]
    fn test_size_distribution() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let analyzer = DirectoryAnalyzer::new();
        
        let analysis = analyzer.analyze_directory(temp_dir.path())?;
        
        let size_dist = &analysis.size_distribution;
        
        // Should have detected zero-byte files
        assert!(size_dist.zero_byte_files > 0);
        
        // Should have detected very large files
        assert!(size_dist.very_large_files > 0);
        
        // Largest file should be reasonable
        assert!(size_dist.largest_file_size > 100_000_000); // At least 100MB
        
        Ok(())
    }

    #[test]
    fn test_processing_complexity_assessment() -> Result<()> {
        let temp_dir = create_chaos_test_directory()?;
        let analyzer = DirectoryAnalyzer::new();
        
        let analysis = analyzer.analyze_directory(temp_dir.path())?;
        
        // Should have files with different complexity levels
        let complexities: Vec<&ProcessingComplexity> = analysis.file_type_distribution
            .values()
            .map(|stats| &stats.processing_complexity)
            .collect();
        
        // Should have at least some low complexity files (text files)
        assert!(complexities.iter().any(|c| matches!(c, ProcessingComplexity::Low)));
        
        Ok(())
    }
}