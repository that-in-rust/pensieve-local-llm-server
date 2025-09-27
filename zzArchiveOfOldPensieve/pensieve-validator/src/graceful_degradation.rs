use crate::errors::{ValidationError, ErrorDetails, ErrorAggregator, RecoveryAction, ErrorRecoveryManager};
use crate::types::{ValidationResults, ValidationPhase};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Manages graceful degradation when validation components fail
pub struct GracefulDegradationManager {
    recovery_manager: ErrorRecoveryManager,
    error_aggregator: ErrorAggregator,
    phase_results: HashMap<ValidationPhase, PhaseResult>,
    degradation_config: DegradationConfig,
}

/// Configuration for graceful degradation behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    /// Allow partial validation when some phases fail
    pub allow_partial_validation: bool,
    /// Minimum number of phases that must succeed for validation to be considered successful
    pub minimum_successful_phases: usize,
    /// Maximum number of retries per phase
    pub max_phase_retries: u32,
    /// Whether to continue validation after critical errors
    pub continue_after_critical_errors: bool,
    /// Phases that are considered essential (validation fails if these fail)
    pub essential_phases: Vec<ValidationPhase>,
}

/// Result of a validation phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub phase: ValidationPhase,
    pub status: PhaseStatus,
    pub error_details: Option<ErrorDetails>,
    pub partial_data: Option<serde_json::Value>,
    pub retry_count: u32,
    pub degradation_applied: Option<String>,
}

/// Status of a validation phase
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhaseStatus {
    /// Phase completed successfully
    Success,
    /// Phase completed with warnings but usable results
    SuccessWithWarnings,
    /// Phase failed but validation can continue
    FailedNonCritical,
    /// Phase failed and is critical for validation
    FailedCritical,
    /// Phase was skipped due to dependencies
    Skipped,
    /// Phase is currently running
    InProgress,
    /// Phase has not started yet
    NotStarted,
}

/// Degradation strategy applied to handle failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationStrategy {
    pub strategy_type: DegradationType,
    pub description: String,
    pub impact_assessment: String,
    pub alternative_approach: Option<String>,
}

/// Types of degradation that can be applied
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DegradationType {
    /// Skip the failed component entirely
    Skip,
    /// Use a simpler/faster alternative implementation
    SimplifiedImplementation,
    /// Reduce the scope of analysis
    ReducedScope,
    /// Use cached or default values
    FallbackData,
    /// Continue with reduced accuracy
    ReducedAccuracy,
}

impl GracefulDegradationManager {
    pub fn new(config: DegradationConfig) -> Self {
        Self {
            recovery_manager: ErrorRecoveryManager::new(),
            error_aggregator: ErrorAggregator::new(),
            phase_results: HashMap::new(),
            degradation_config: config,
        }
    }
    
    /// Handle a phase failure and determine how to proceed
    pub async fn handle_phase_failure(
        &mut self,
        phase: ValidationPhase,
        error: ValidationError,
        context: crate::errors::ValidationContext,
    ) -> DegradationDecision {
        // Create detailed error information
        let error_details = self.recovery_manager.create_error_details(error.clone(), context);
        self.error_aggregator.add_error(error_details.clone());
        
        // Check if this is an essential phase
        let is_essential = self.degradation_config.essential_phases.contains(&phase);
        
        // Get current retry count for this phase
        let retry_count = self.phase_results
            .get(&phase)
            .map(|r| r.retry_count)
            .unwrap_or(0);
        
        // Attempt recovery
        let recovery_action = self.recovery_manager.attempt_recovery(&error).await;
        
        let decision = match recovery_action {
            RecoveryAction::Abort => {
                if is_essential {
                    DegradationDecision::AbortValidation
                } else {
                    DegradationDecision::SkipPhase {
                        strategy: DegradationStrategy {
                            strategy_type: DegradationType::Skip,
                            description: format!("Skipping {} due to critical error", phase),
                            impact_assessment: "Phase results will not be available".to_string(),
                            alternative_approach: None,
                        }
                    }
                }
            },
            
            RecoveryAction::Retry => {
                if retry_count < self.degradation_config.max_phase_retries {
                    DegradationDecision::RetryPhase
                } else {
                    self.decide_degradation_strategy(phase, &error, is_essential)
                }
            },
            
            RecoveryAction::RetryWithChanges(suggestion) => {
                DegradationDecision::RetryWithModification { suggestion }
            },
            
            RecoveryAction::SkipComponent(impact) => {
                DegradationDecision::SkipPhase {
                    strategy: DegradationStrategy {
                        strategy_type: DegradationType::Skip,
                        description: format!("Skipping component in {}", phase),
                        impact_assessment: impact,
                        alternative_approach: None,
                    }
                }
            },
            
            RecoveryAction::ContinueWithReducedFunctionality(description) => {
                DegradationDecision::ApplyDegradation {
                    strategy: DegradationStrategy {
                        strategy_type: DegradationType::ReducedAccuracy,
                        description,
                        impact_assessment: "Results may be less comprehensive".to_string(),
                        alternative_approach: Some("Use simplified analysis".to_string()),
                    }
                }
            },
            
            RecoveryAction::RequireManualIntervention(steps) => {
                DegradationDecision::RequireIntervention { steps }
            },
        };
        
        // Update phase result
        let status = match &decision {
            DegradationDecision::AbortValidation => PhaseStatus::FailedCritical,
            DegradationDecision::SkipPhase { .. } => PhaseStatus::FailedNonCritical,
            DegradationDecision::RetryPhase => PhaseStatus::InProgress,
            DegradationDecision::RetryWithModification { .. } => PhaseStatus::InProgress,
            DegradationDecision::ApplyDegradation { .. } => PhaseStatus::SuccessWithWarnings,
            DegradationDecision::RequireIntervention { .. } => PhaseStatus::FailedCritical,
        };
        
        self.phase_results.insert(phase.clone(), PhaseResult {
            phase: phase.clone(),
            status,
            error_details: Some(error_details),
            partial_data: None,
            retry_count: retry_count + 1,
            degradation_applied: match &decision {
                DegradationDecision::ApplyDegradation { strategy } => Some(strategy.description.clone()),
                DegradationDecision::SkipPhase { strategy } => Some(strategy.description.clone()),
                _ => None,
            },
        });
        
        decision
    }
    
    /// Decide on a degradation strategy for a failed phase
    fn decide_degradation_strategy(
        &self,
        phase: ValidationPhase,
        error: &ValidationError,
        is_essential: bool,
    ) -> DegradationDecision {
        if is_essential && !self.degradation_config.continue_after_critical_errors {
            return DegradationDecision::AbortValidation;
        }
        
        // Choose degradation strategy based on phase and error type
        let strategy = match phase {
            ValidationPhase::DirectoryAnalysis => {
                DegradationStrategy {
                    strategy_type: DegradationType::ReducedScope,
                    description: "Use basic file listing instead of detailed analysis".to_string(),
                    impact_assessment: "File chaos detection may be incomplete".to_string(),
                    alternative_approach: Some("Use simple file enumeration".to_string()),
                }
            },
            
            ValidationPhase::ChaosDetection => {
                DegradationStrategy {
                    strategy_type: DegradationType::SimplifiedImplementation,
                    description: "Use basic file type detection only".to_string(),
                    impact_assessment: "Advanced chaos patterns may not be detected".to_string(),
                    alternative_approach: Some("Check file extensions only".to_string()),
                }
            },
            
            ValidationPhase::PerformanceBenchmarking => {
                DegradationStrategy {
                    strategy_type: DegradationType::FallbackData,
                    description: "Use estimated performance metrics".to_string(),
                    impact_assessment: "Performance analysis will be less accurate".to_string(),
                    alternative_approach: Some("Use default performance assumptions".to_string()),
                }
            },
            
            ValidationPhase::ReliabilityTesting => {
                if is_essential {
                    return DegradationDecision::AbortValidation;
                }
                DegradationStrategy {
                    strategy_type: DegradationType::ReducedScope,
                    description: "Skip stress testing, perform basic reliability checks only".to_string(),
                    impact_assessment: "Reliability assessment will be limited".to_string(),
                    alternative_approach: Some("Basic error handling verification".to_string()),
                }
            },
            
            ValidationPhase::ReportGeneration => {
                DegradationStrategy {
                    strategy_type: DegradationType::SimplifiedImplementation,
                    description: "Generate basic text report instead of full HTML/JSON".to_string(),
                    impact_assessment: "Report will have reduced formatting and features".to_string(),
                    alternative_approach: Some("Plain text summary report".to_string()),
                }
            },
        };
        
        DegradationDecision::ApplyDegradation { strategy }
    }
    
    /// Check if validation can continue with current phase results
    pub fn can_continue_validation(&self) -> bool {
        if !self.degradation_config.allow_partial_validation {
            return self.phase_results.values().all(|r| matches!(r.status, PhaseStatus::Success | PhaseStatus::SuccessWithWarnings));
        }
        
        let successful_phases = self.phase_results.values()
            .filter(|r| matches!(r.status, PhaseStatus::Success | PhaseStatus::SuccessWithWarnings))
            .count();
        
        let critical_failures = self.phase_results.values()
            .filter(|r| matches!(r.status, PhaseStatus::FailedCritical))
            .count();
        
        successful_phases >= self.degradation_config.minimum_successful_phases && critical_failures == 0
    }
    
    /// Generate a degradation report showing what compromises were made
    pub fn generate_degradation_report(&self) -> DegradationReport {
        let total_phases = self.phase_results.len();
        let successful_phases = self.phase_results.values()
            .filter(|r| matches!(r.status, PhaseStatus::Success))
            .count();
        let degraded_phases = self.phase_results.values()
            .filter(|r| r.degradation_applied.is_some())
            .count();
        let failed_phases = self.phase_results.values()
            .filter(|r| matches!(r.status, PhaseStatus::FailedCritical | PhaseStatus::FailedNonCritical))
            .count();
        
        let degradation_strategies: Vec<_> = self.phase_results.values()
            .filter_map(|r| {
                r.degradation_applied.as_ref().map(|desc| {
                    (r.phase.clone(), desc.clone())
                })
            })
            .collect();
        
        let error_summary = self.error_aggregator.get_error_summary();
        
        DegradationReport {
            total_phases,
            successful_phases,
            degraded_phases,
            failed_phases,
            degradation_strategies,
            error_summary,
            overall_impact: self.assess_overall_impact(),
            recommendations: self.generate_recommendations(),
        }
    }
    
    fn assess_overall_impact(&self) -> String {
        let degraded_count = self.phase_results.values()
            .filter(|r| r.degradation_applied.is_some())
            .count();
        
        let total_count = self.phase_results.len();
        
        if degraded_count == 0 {
            "No degradation applied - full validation completed".to_string()
        } else if degraded_count < total_count / 3 {
            "Minor degradation - validation results are mostly complete".to_string()
        } else if degraded_count < total_count * 2 / 3 {
            "Moderate degradation - some validation results may be incomplete".to_string()
        } else {
            "Significant degradation - validation results are limited".to_string()
        }
    }
    
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let critical_failures = self.phase_results.values()
            .filter(|r| matches!(r.status, PhaseStatus::FailedCritical))
            .count();
        
        if critical_failures > 0 {
            recommendations.push("Address critical failures before relying on validation results".to_string());
        }
        
        let degraded_phases = self.phase_results.values()
            .filter(|r| r.degradation_applied.is_some())
            .count();
        
        if degraded_phases > 0 {
            recommendations.push("Review degraded phases and consider re-running with fixes".to_string());
        }
        
        if self.error_aggregator.get_error_summary().blocker_count > 0 {
            recommendations.push("Resolve blocker issues for complete validation".to_string());
        }
        
        recommendations.push("Check error details for specific remediation steps".to_string());
        
        recommendations
    }
    
    pub fn get_phase_results(&self) -> &HashMap<ValidationPhase, PhaseResult> {
        &self.phase_results
    }
    
    pub fn get_error_summary(&self) -> crate::errors::ErrorSummary {
        self.error_aggregator.get_error_summary()
    }
}

/// Decision on how to handle a phase failure
#[derive(Debug, Clone)]
pub enum DegradationDecision {
    /// Abort the entire validation process
    AbortValidation,
    /// Skip this phase and continue
    SkipPhase { strategy: DegradationStrategy },
    /// Retry the phase
    RetryPhase,
    /// Retry with suggested modifications
    RetryWithModification { suggestion: String },
    /// Apply degradation strategy and continue
    ApplyDegradation { strategy: DegradationStrategy },
    /// Require manual intervention
    RequireIntervention { steps: Vec<String> },
}

/// Report on degradation applied during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationReport {
    pub total_phases: usize,
    pub successful_phases: usize,
    pub degraded_phases: usize,
    pub failed_phases: usize,
    pub degradation_strategies: Vec<(ValidationPhase, String)>,
    pub error_summary: crate::errors::ErrorSummary,
    pub overall_impact: String,
    pub recommendations: Vec<String>,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        Self {
            allow_partial_validation: true,
            minimum_successful_phases: 3,
            max_phase_retries: 2,
            continue_after_critical_errors: false,
            essential_phases: vec![
                ValidationPhase::DirectoryAnalysis,
                ValidationPhase::ReliabilityTesting,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ValidationError;
    use std::path::PathBuf;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_graceful_degradation_manager() {
        let config = DegradationConfig::default();
        let mut manager = GracefulDegradationManager::new(config);
        
        let context = crate::errors::ValidationContext {
            current_phase: "test".to_string(),
            target_directory: PathBuf::from("/test"),
            pensieve_binary: PathBuf::from("/usr/bin/pensieve"),
            config_file: None,
            elapsed_time: Duration::from_secs(10),
            processed_files: 5,
            current_file: None,
        };
        
        let error = ValidationError::analysis_error(
            "chaos_detection".to_string(),
            "File type detection failed".to_string()
        );
        
        let decision = manager.handle_phase_failure(
            ValidationPhase::ChaosDetection,
            error,
            context
        ).await;
        
        match decision {
            DegradationDecision::ApplyDegradation { strategy } => {
                assert_eq!(strategy.strategy_type, DegradationType::SimplifiedImplementation);
            },
            _ => panic!("Expected ApplyDegradation decision"),
        }
        
        assert!(manager.can_continue_validation());
        
        let report = manager.generate_degradation_report();
        assert_eq!(report.total_phases, 1);
        assert_eq!(report.degraded_phases, 1);
    }
    
    #[test]
    fn test_degradation_config() {
        let config = DegradationConfig {
            allow_partial_validation: false,
            minimum_successful_phases: 5,
            max_phase_retries: 1,
            continue_after_critical_errors: true,
            essential_phases: vec![ValidationPhase::ReliabilityTesting],
        };
        
        let manager = GracefulDegradationManager::new(config.clone());
        assert_eq!(manager.degradation_config.minimum_successful_phases, 5);
        assert!(!manager.degradation_config.allow_partial_validation);
    }
}