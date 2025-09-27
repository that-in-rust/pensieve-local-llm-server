use crate::errors::{ValidationError, ErrorRecoveryManager, ErrorAggregator, ValidationContext};
use crate::graceful_degradation::{GracefulDegradationManager, DegradationConfig};
use crate::error_reporter::{ErrorReporter, ErrorReportConfig};
use std::path::PathBuf;
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation_and_categorization() {
        let error = ValidationError::pensieve_binary_not_found(PathBuf::from("/usr/bin/pensieve"));
        
        assert_eq!(error.category(), crate::errors::ErrorCategory::Configuration);
        assert_eq!(error.impact(), crate::errors::ErrorImpact::Blocker);
        
        let reproduction_steps = error.reproduction_steps();
        assert!(!reproduction_steps.is_empty());
        assert!(reproduction_steps[0].contains("pensieve binary"));
        
        let suggested_fixes = error.suggested_fixes();
        assert!(!suggested_fixes.is_empty());
        assert!(suggested_fixes.iter().any(|fix| fix.contains("Install")));
    }

    #[test]
    fn test_error_recovery_manager() {
        let manager = ErrorRecoveryManager::new();
        let context = ValidationContext {
            current_phase: "test".to_string(),
            target_directory: PathBuf::from("/test"),
            pensieve_binary: PathBuf::from("/usr/bin/pensieve"),
            config_file: None,
            elapsed_time: Duration::from_secs(10),
            processed_files: 5,
            current_file: None,
        };
        
        let error = ValidationError::validation_timeout(300);
        let error_details = manager.create_error_details(error, context);
        
        assert_eq!(error_details.category, crate::errors::ErrorCategory::ResourceConstraints);
        assert_eq!(error_details.impact, crate::errors::ErrorImpact::High);
        assert!(!error_details.reproduction_steps.is_empty());
        assert!(!error_details.suggested_fixes.is_empty());
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new();
        let recovery_manager = ErrorRecoveryManager::new();
        
        let context = ValidationContext {
            current_phase: "test".to_string(),
            target_directory: PathBuf::from("/test"),
            pensieve_binary: PathBuf::from("/usr/bin/pensieve"),
            config_file: None,
            elapsed_time: Duration::from_secs(10),
            processed_files: 5,
            current_file: None,
        };
        
        // Add multiple errors
        let error1 = ValidationError::pensieve_binary_not_found(PathBuf::from("/usr/bin/pensieve"));
        let error2 = ValidationError::validation_timeout(300);
        let error3 = ValidationError::resource_limit_exceeded("memory".to_string(), "8GB".to_string());
        
        let details1 = recovery_manager.create_error_details(error1, context.clone());
        let details2 = recovery_manager.create_error_details(error2, context.clone());
        let details3 = recovery_manager.create_error_details(error3, context);
        
        aggregator.add_error(details1);
        aggregator.add_error(details2);
        aggregator.add_error(details3);
        
        let summary = aggregator.get_error_summary();
        assert_eq!(summary.total_errors, 3);
        assert_eq!(summary.blocker_count, 1); // pensieve_binary_not_found
        assert_eq!(summary.high_impact_count, 2); // timeout and resource limit
        assert!(summary.most_common_category.is_some());
    }

    #[test]
    fn test_graceful_degradation_config() {
        let config = DegradationConfig {
            allow_partial_validation: true,
            minimum_successful_phases: 3,
            max_phase_retries: 2,
            continue_after_critical_errors: false,
            essential_phases: vec![
                crate::types::ValidationPhase::DirectoryAnalysis,
                crate::types::ValidationPhase::ReliabilityTesting,
            ],
        };
        
        let manager = GracefulDegradationManager::new(config.clone());
        assert_eq!(manager.degradation_config.minimum_successful_phases, 3);
        assert!(manager.degradation_config.allow_partial_validation);
        assert_eq!(manager.degradation_config.essential_phases.len(), 2);
    }

    #[test]
    fn test_error_reporter_creation() {
        let config = ErrorReportConfig::default();
        let reporter = ErrorReporter::new(config);
        
        // Should create successfully with default config
        assert!(reporter.report_config.include_stack_traces);
        assert!(reporter.report_config.include_system_info);
        assert!(reporter.report_config.include_reproduction_steps);
        assert!(reporter.report_config.include_suggested_fixes);
    }

    #[test]
    fn test_error_conversion_from_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let validation_error: ValidationError = io_error.into();
        
        match validation_error {
            ValidationError::FileSystem { cause, path, recovery_strategy } => {
                assert!(cause.contains("File not found"));
                assert!(path.is_none());
                assert!(matches!(recovery_strategy, crate::errors::RecoveryStrategy::SkipAndContinue { .. }));
            },
            _ => panic!("Expected FileSystem error"),
        }
    }

    #[test]
    fn test_error_conversion_from_serde_error() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_error.is_err());
        
        let validation_error: ValidationError = json_error.unwrap_err().into();
        
        match validation_error {
            ValidationError::Serialization { cause, recovery_strategy } => {
                assert!(cause.contains("expected"));
                assert!(matches!(recovery_strategy, crate::errors::RecoveryStrategy::SkipAndContinue { .. }));
            },
            _ => panic!("Expected Serialization error"),
        }
    }

    #[test]
    fn test_comprehensive_error_report_generation() {
        let config = ErrorReportConfig::default();
        let mut reporter = ErrorReporter::new(config);
        
        // Add some test errors
        let context = ValidationContext {
            current_phase: "test".to_string(),
            target_directory: PathBuf::from("/test"),
            pensieve_binary: PathBuf::from("/usr/bin/pensieve"),
            config_file: None,
            elapsed_time: Duration::from_secs(10),
            processed_files: 5,
            current_file: None,
        };
        
        let recovery_manager = ErrorRecoveryManager::new();
        let error = ValidationError::pensieve_binary_not_found(PathBuf::from("/usr/bin/pensieve"));
        let error_details = recovery_manager.create_error_details(error, context);
        
        reporter.add_error(error_details);
        
        let report = reporter.generate_comprehensive_report(None);
        
        // Verify report structure
        assert_eq!(report.executive_summary.total_errors, 1);
        assert_eq!(report.executive_summary.critical_errors, 1);
        assert!(!report.executive_summary.validation_success);
        assert!(!report.detailed_errors.is_empty());
        assert!(!report.error_analysis.error_patterns.is_empty() || report.error_analysis.root_cause_analysis.len() > 0);
        assert!(!report.recovery_guidance.immediate_actions.is_empty());
    }
}