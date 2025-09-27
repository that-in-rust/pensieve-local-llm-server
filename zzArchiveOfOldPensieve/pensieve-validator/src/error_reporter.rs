use crate::errors::{ErrorDetails, ErrorSummary, ErrorAggregator, ValidationError, ErrorCategory, ErrorImpact};
use crate::graceful_degradation::{DegradationReport, GracefulDegradationManager};
use crate::types::ValidationPhase;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::{DateTime, Utc};

/// Comprehensive error reporting system for validation failures
pub struct ErrorReporter {
    aggregator: ErrorAggregator,
    report_config: ErrorReportConfig,
}

/// Configuration for error reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReportConfig {
    /// Include full stack traces in reports
    pub include_stack_traces: bool,
    /// Include system information in reports
    pub include_system_info: bool,
    /// Include environment variables in reports
    pub include_environment: bool,
    /// Include file system state in reports
    pub include_filesystem_state: bool,
    /// Maximum number of similar errors to include
    pub max_similar_errors: usize,
    /// Include reproduction steps
    pub include_reproduction_steps: bool,
    /// Include suggested fixes
    pub include_suggested_fixes: bool,
    /// Generate reports in multiple formats
    pub output_formats: Vec<ReportFormat>,
}

/// Available report formats
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Human-readable HTML report
    Html,
    /// Machine-readable JSON report
    Json,
    /// Plain text report for logs
    Text,
    /// Markdown report for documentation
    Markdown,
    /// CSV report for analysis
    Csv,
}

/// Comprehensive error report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveErrorReport {
    pub metadata: ErrorReportMetadata,
    pub executive_summary: ErrorExecutiveSummary,
    pub detailed_errors: Vec<ErrorDetails>,
    pub error_analysis: ErrorAnalysis,
    pub impact_assessment: ImpactAssessment,
    pub recovery_guidance: RecoveryGuidance,
    pub debugging_information: DebuggingInformation,
    pub degradation_report: Option<DegradationReport>,
}

/// Metadata about the error report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReportMetadata {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub validator_version: String,
    pub validation_target: PathBuf,
    pub validation_duration: Option<std::time::Duration>,
    pub report_format: ReportFormat,
}

/// Executive summary of errors for quick assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorExecutiveSummary {
    pub total_errors: usize,
    pub critical_errors: usize,
    pub validation_success: bool,
    pub primary_failure_reason: Option<String>,
    pub overall_impact: String,
    pub immediate_actions_required: Vec<String>,
    pub estimated_fix_time: String,
}

/// Detailed analysis of error patterns and trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub error_patterns: Vec<ErrorPattern>,
    pub root_cause_analysis: Vec<RootCause>,
    pub error_correlation: Vec<ErrorCorrelation>,
    pub temporal_analysis: TemporalAnalysis,
    pub phase_failure_analysis: PhaseFailureAnalysis,
}

/// Pattern of similar errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub pattern_id: String,
    pub description: String,
    pub occurrence_count: usize,
    pub affected_phases: Vec<ValidationPhase>,
    pub common_characteristics: Vec<String>,
    pub suggested_fix: String,
}

/// Root cause analysis for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCause {
    pub cause_id: String,
    pub description: String,
    pub contributing_factors: Vec<String>,
    pub affected_errors: Vec<String>,
    pub likelihood: f64,
    pub fix_complexity: FixComplexity,
}

/// Complexity of fixing an issue
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FixComplexity {
    Simple,    // Configuration change or simple fix
    Moderate,  // Code changes or system updates
    Complex,   // Architectural changes or major updates
    Unknown,   // Cannot determine complexity
}

/// Correlation between different errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrelation {
    pub primary_error: String,
    pub related_errors: Vec<String>,
    pub correlation_strength: f64,
    pub relationship_type: CorrelationType,
}

/// Type of relationship between errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CorrelationType {
    CausedBy,      // One error caused another
    Concurrent,    // Errors occurred at the same time
    Sequential,    // Errors occurred in sequence
    Similar,       // Errors have similar characteristics
}

/// Analysis of error timing and patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub error_timeline: Vec<ErrorTimelineEvent>,
    pub peak_error_periods: Vec<ErrorPeriod>,
    pub error_frequency_analysis: ErrorFrequencyAnalysis,
}

/// Event in the error timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTimelineEvent {
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
    pub phase: ValidationPhase,
    pub severity: ErrorImpact,
    pub description: String,
}

/// Period of high error activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPeriod {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub error_count: usize,
    pub dominant_error_types: Vec<String>,
}

/// Analysis of error frequency patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorFrequencyAnalysis {
    pub errors_per_minute: f64,
    pub error_burst_detection: Vec<ErrorBurst>,
    pub steady_state_error_rate: f64,
}

/// Detected burst of errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBurst {
    pub start_time: DateTime<Utc>,
    pub duration_seconds: f64,
    pub error_count: usize,
    pub trigger_hypothesis: String,
}

/// Analysis of failures by validation phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseFailureAnalysis {
    pub phase_success_rates: HashMap<ValidationPhase, f64>,
    pub phase_error_counts: HashMap<ValidationPhase, usize>,
    pub critical_phase_failures: Vec<ValidationPhase>,
    pub phase_dependencies: Vec<PhaseDependency>,
}

/// Dependency between validation phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDependency {
    pub dependent_phase: ValidationPhase,
    pub required_phase: ValidationPhase,
    pub dependency_strength: f64,
}

/// Assessment of error impact on validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub validation_completeness: f64,
    pub data_quality_impact: DataQualityImpact,
    pub user_experience_impact: UserExperienceImpact,
    pub production_readiness_impact: ProductionReadinessImpact,
    pub business_impact: BusinessImpact,
}

/// Impact on data quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityImpact {
    pub missing_data_percentage: f64,
    pub unreliable_data_percentage: f64,
    pub affected_metrics: Vec<String>,
    pub confidence_level: f64,
}

/// Impact on user experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserExperienceImpact {
    pub confusion_likelihood: f64,
    pub frustration_factors: Vec<String>,
    pub workflow_disruption: f64,
    pub support_burden_increase: f64,
}

/// Impact on production readiness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionReadinessImpact {
    pub readiness_score_reduction: f64,
    pub blocked_deployment_scenarios: Vec<String>,
    pub increased_risk_factors: Vec<String>,
    pub additional_testing_required: Vec<String>,
}

/// Business impact of validation errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    pub decision_confidence_impact: f64,
    pub timeline_impact: String,
    pub resource_impact: String,
    pub risk_exposure: String,
}

/// Recovery guidance for addressing errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryGuidance {
    pub immediate_actions: Vec<ImmediateAction>,
    pub short_term_fixes: Vec<ShortTermFix>,
    pub long_term_improvements: Vec<LongTermImprovement>,
    pub prevention_strategies: Vec<PreventionStrategy>,
}

/// Immediate action to address critical errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmediateAction {
    pub action_id: String,
    pub description: String,
    pub priority: ActionPriority,
    pub estimated_time: String,
    pub required_skills: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// Priority level for actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionPriority {
    Critical,  // Must be done immediately
    High,      // Should be done within hours
    Medium,    // Should be done within days
    Low,       // Can be done when convenient
}

/// Short-term fix for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortTermFix {
    pub fix_id: String,
    pub description: String,
    pub affected_errors: Vec<String>,
    pub implementation_steps: Vec<String>,
    pub testing_requirements: Vec<String>,
    pub rollback_plan: String,
}

/// Long-term improvement to prevent errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermImprovement {
    pub improvement_id: String,
    pub description: String,
    pub benefits: Vec<String>,
    pub implementation_complexity: FixComplexity,
    pub estimated_effort: String,
    pub success_metrics: Vec<String>,
}

/// Strategy to prevent similar errors in the future
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreventionStrategy {
    pub strategy_id: String,
    pub description: String,
    pub target_error_types: Vec<ErrorCategory>,
    pub implementation_approach: String,
    pub monitoring_requirements: Vec<String>,
}

/// Debugging information for technical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggingInformation {
    pub log_analysis: LogAnalysis,
    pub system_state_analysis: SystemStateAnalysis,
    pub configuration_analysis: ConfigurationAnalysis,
    pub dependency_analysis: DependencyAnalysis,
}

/// Analysis of log files and messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogAnalysis {
    pub log_patterns: Vec<LogPattern>,
    pub error_message_analysis: Vec<ErrorMessageAnalysis>,
    pub warning_indicators: Vec<String>,
    pub performance_indicators: Vec<String>,
}

/// Pattern found in log files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogPattern {
    pub pattern: String,
    pub frequency: usize,
    pub severity: String,
    pub context: String,
}

/// Analysis of error messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessageAnalysis {
    pub message: String,
    pub clarity_score: f64,
    pub actionability_score: f64,
    pub technical_level: String,
    pub improvement_suggestions: Vec<String>,
}

/// Analysis of system state during errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStateAnalysis {
    pub resource_utilization: ResourceUtilization,
    pub process_state: ProcessState,
    pub network_state: Option<NetworkState>,
    pub filesystem_state: FilesystemState,
}

/// Resource utilization during errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub io_wait_percent: f64,
}

/// Process state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessState {
    pub process_count: usize,
    pub zombie_processes: usize,
    pub high_cpu_processes: Vec<String>,
    pub high_memory_processes: Vec<String>,
}

/// Network state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    pub active_connections: usize,
    pub network_errors: usize,
    pub bandwidth_utilization: f64,
}

/// Filesystem state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemState {
    pub available_space_gb: f64,
    pub inode_usage_percent: f64,
    pub mount_points: Vec<String>,
    pub filesystem_errors: Vec<String>,
}

/// Configuration analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationAnalysis {
    pub configuration_issues: Vec<ConfigurationIssue>,
    pub missing_configurations: Vec<String>,
    pub conflicting_configurations: Vec<ConfigurationConflict>,
    pub optimization_opportunities: Vec<String>,
}

/// Configuration issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationIssue {
    pub parameter: String,
    pub current_value: String,
    pub issue_description: String,
    pub recommended_value: String,
    pub impact: String,
}

/// Conflicting configuration values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationConflict {
    pub parameter1: String,
    pub parameter2: String,
    pub conflict_description: String,
    pub resolution: String,
}

/// Dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    pub missing_dependencies: Vec<String>,
    pub version_conflicts: Vec<VersionConflict>,
    pub dependency_health: Vec<DependencyHealth>,
}

/// Version conflict between dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionConflict {
    pub dependency: String,
    pub required_version: String,
    pub actual_version: String,
    pub conflict_severity: String,
}

/// Health status of a dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyHealth {
    pub dependency: String,
    pub status: String,
    pub last_check: DateTime<Utc>,
    pub issues: Vec<String>,
}

impl ErrorReporter {
    pub fn new(config: ErrorReportConfig) -> Self {
        Self {
            aggregator: ErrorAggregator::new(),
            report_config: config,
        }
    }
    
    pub fn add_error(&mut self, error_details: ErrorDetails) {
        self.aggregator.add_error(error_details);
    }
    
    pub fn generate_comprehensive_report(
        &self,
        degradation_manager: Option<&GracefulDegradationManager>,
    ) -> ComprehensiveErrorReport {
        let error_summary = self.aggregator.get_error_summary();
        let errors = self.aggregator.get_errors();
        
        ComprehensiveErrorReport {
            metadata: self.generate_metadata(),
            executive_summary: self.generate_executive_summary(&error_summary),
            detailed_errors: errors.to_vec(),
            error_analysis: self.analyze_errors(errors),
            impact_assessment: self.assess_impact(errors, &error_summary),
            recovery_guidance: self.generate_recovery_guidance(errors),
            debugging_information: self.collect_debugging_information(),
            degradation_report: degradation_manager.map(|dm| dm.generate_degradation_report()),
        }
    }
    
    fn generate_metadata(&self) -> ErrorReportMetadata {
        ErrorReportMetadata {
            report_id: uuid::Uuid::new_v4().to_string(),
            generated_at: Utc::now(),
            validator_version: env!("CARGO_PKG_VERSION").to_string(),
            validation_target: std::env::current_dir().unwrap_or_default(),
            validation_duration: None, // Would be populated by caller
            report_format: ReportFormat::Json, // Default format
        }
    }
    
    fn generate_executive_summary(&self, summary: &ErrorSummary) -> ErrorExecutiveSummary {
        let validation_success = summary.blocker_count == 0 && summary.high_impact_count < 3;
        
        let primary_failure_reason = if summary.blocker_count > 0 {
            Some("Critical errors prevent validation completion".to_string())
        } else if summary.high_impact_count > 0 {
            Some("High-impact errors affect validation quality".to_string())
        } else {
            None
        };
        
        let overall_impact = if summary.blocker_count > 0 {
            "Severe - Validation cannot be completed"
        } else if summary.high_impact_count > 2 {
            "High - Validation results are significantly compromised"
        } else if summary.medium_impact_count > 5 {
            "Medium - Some validation results may be incomplete"
        } else {
            "Low - Minor issues with minimal impact"
        }.to_string();
        
        let immediate_actions_required = if summary.blocker_count > 0 {
            vec!["Resolve critical errors before proceeding".to_string()]
        } else if summary.high_impact_count > 0 {
            vec!["Review high-impact errors and apply fixes".to_string()]
        } else {
            vec!["Review error details for optimization opportunities".to_string()]
        };
        
        let estimated_fix_time = if summary.blocker_count > 0 {
            "Hours to days"
        } else if summary.high_impact_count > 0 {
            "Minutes to hours"
        } else {
            "Minutes"
        }.to_string();
        
        ErrorExecutiveSummary {
            total_errors: summary.total_errors,
            critical_errors: summary.blocker_count + summary.high_impact_count,
            validation_success,
            primary_failure_reason,
            overall_impact,
            immediate_actions_required,
            estimated_fix_time,
        }
    }
    
    fn analyze_errors(&self, errors: &[ErrorDetails]) -> ErrorAnalysis {
        ErrorAnalysis {
            error_patterns: self.identify_error_patterns(errors),
            root_cause_analysis: self.perform_root_cause_analysis(errors),
            error_correlation: self.analyze_error_correlations(errors),
            temporal_analysis: self.analyze_temporal_patterns(errors),
            phase_failure_analysis: self.analyze_phase_failures(errors),
        }
    }
    
    fn identify_error_patterns(&self, errors: &[ErrorDetails]) -> Vec<ErrorPattern> {
        let mut patterns = Vec::new();
        let mut error_groups: HashMap<String, Vec<&ErrorDetails>> = HashMap::new();
        
        // Group errors by type
        for error in errors {
            let error_type = format!("{:?}", error.error);
            error_groups.entry(error_type).or_default().push(error);
        }
        
        // Create patterns for groups with multiple occurrences
        for (error_type, group_errors) in error_groups {
            if group_errors.len() > 1 {
                let affected_phases: Vec<ValidationPhase> = group_errors
                    .iter()
                    .map(|e| ValidationPhase::DirectoryAnalysis) // Would extract from context
                    .collect();
                
                patterns.push(ErrorPattern {
                    pattern_id: uuid::Uuid::new_v4().to_string(),
                    description: format!("Repeated {} errors", error_type),
                    occurrence_count: group_errors.len(),
                    affected_phases,
                    common_characteristics: vec!["Similar error conditions".to_string()],
                    suggested_fix: "Address common root cause".to_string(),
                });
            }
        }
        
        patterns
    }
    
    fn perform_root_cause_analysis(&self, errors: &[ErrorDetails]) -> Vec<RootCause> {
        let mut root_causes = Vec::new();
        
        // Analyze configuration-related errors
        let config_errors: Vec<_> = errors
            .iter()
            .filter(|e| matches!(e.category, ErrorCategory::Configuration))
            .collect();
        
        if !config_errors.is_empty() {
            root_causes.push(RootCause {
                cause_id: "config_issues".to_string(),
                description: "Configuration problems preventing proper operation".to_string(),
                contributing_factors: vec![
                    "Missing configuration files".to_string(),
                    "Invalid configuration values".to_string(),
                    "Configuration conflicts".to_string(),
                ],
                affected_errors: config_errors.iter().map(|e| e.error.to_string()).collect(),
                likelihood: 0.9,
                fix_complexity: FixComplexity::Simple,
            });
        }
        
        // Analyze filesystem-related errors
        let fs_errors: Vec<_> = errors
            .iter()
            .filter(|e| matches!(e.category, ErrorCategory::FileSystem))
            .collect();
        
        if !fs_errors.is_empty() {
            root_causes.push(RootCause {
                cause_id: "filesystem_issues".to_string(),
                description: "File system access or permission problems".to_string(),
                contributing_factors: vec![
                    "Insufficient permissions".to_string(),
                    "Missing files or directories".to_string(),
                    "Filesystem corruption".to_string(),
                ],
                affected_errors: fs_errors.iter().map(|e| e.error.to_string()).collect(),
                likelihood: 0.8,
                fix_complexity: FixComplexity::Moderate,
            });
        }
        
        root_causes
    }
    
    fn analyze_error_correlations(&self, errors: &[ErrorDetails]) -> Vec<ErrorCorrelation> {
        let mut correlations = Vec::new();
        
        // Simple temporal correlation analysis
        for (i, error1) in errors.iter().enumerate() {
            for error2 in errors.iter().skip(i + 1) {
                let time_diff = (error2.timestamp - error1.timestamp).num_seconds().abs();
                
                if time_diff < 60 { // Errors within 1 minute
                    correlations.push(ErrorCorrelation {
                        primary_error: error1.error.to_string(),
                        related_errors: vec![error2.error.to_string()],
                        correlation_strength: 1.0 - (time_diff as f64 / 60.0),
                        relationship_type: CorrelationType::Concurrent,
                    });
                }
            }
        }
        
        correlations
    }
    
    fn analyze_temporal_patterns(&self, errors: &[ErrorDetails]) -> TemporalAnalysis {
        let timeline_events: Vec<ErrorTimelineEvent> = errors
            .iter()
            .map(|e| ErrorTimelineEvent {
                timestamp: e.timestamp,
                error_type: format!("{:?}", e.error),
                phase: ValidationPhase::DirectoryAnalysis, // Would extract from context
                severity: e.impact.clone(),
                description: e.error.to_string(),
            })
            .collect();
        
        TemporalAnalysis {
            error_timeline: timeline_events,
            peak_error_periods: Vec::new(), // Would analyze for peaks
            error_frequency_analysis: ErrorFrequencyAnalysis {
                errors_per_minute: errors.len() as f64 / 60.0, // Simplified
                error_burst_detection: Vec::new(),
                steady_state_error_rate: 0.1,
            },
        }
    }
    
    fn analyze_phase_failures(&self, errors: &[ErrorDetails]) -> PhaseFailureAnalysis {
        let mut phase_error_counts = HashMap::new();
        
        // Count errors by phase (simplified - would extract from context)
        for _error in errors {
            *phase_error_counts.entry(ValidationPhase::DirectoryAnalysis).or_insert(0) += 1;
        }
        
        PhaseFailureAnalysis {
            phase_success_rates: HashMap::new(),
            phase_error_counts,
            critical_phase_failures: Vec::new(),
            phase_dependencies: Vec::new(),
        }
    }
    
    fn assess_impact(&self, errors: &[ErrorDetails], summary: &ErrorSummary) -> ImpactAssessment {
        let validation_completeness = if summary.blocker_count > 0 {
            0.0
        } else {
            1.0 - (summary.high_impact_count as f64 * 0.2 + summary.medium_impact_count as f64 * 0.1)
        }.max(0.0f64);
        
        ImpactAssessment {
            validation_completeness,
            data_quality_impact: DataQualityImpact {
                missing_data_percentage: (1.0 - validation_completeness) * 100.0,
                unreliable_data_percentage: summary.medium_impact_count as f64 * 5.0,
                affected_metrics: vec!["Performance metrics".to_string(), "Reliability scores".to_string()],
                confidence_level: validation_completeness,
            },
            user_experience_impact: UserExperienceImpact {
                confusion_likelihood: summary.high_impact_count as f64 * 0.3,
                frustration_factors: vec!["Unclear error messages".to_string()],
                workflow_disruption: summary.blocker_count as f64 * 0.8,
                support_burden_increase: summary.total_errors as f64 * 0.1,
            },
            production_readiness_impact: ProductionReadinessImpact {
                readiness_score_reduction: summary.blocker_count as f64 * 0.5,
                blocked_deployment_scenarios: if summary.blocker_count > 0 {
                    vec!["Production deployment".to_string()]
                } else {
                    Vec::new()
                },
                increased_risk_factors: vec!["Validation uncertainty".to_string()],
                additional_testing_required: vec!["Manual verification".to_string()],
            },
            business_impact: BusinessImpact {
                decision_confidence_impact: 1.0 - validation_completeness,
                timeline_impact: if summary.blocker_count > 0 { "Delayed" } else { "Minimal" }.to_string(),
                resource_impact: "Additional investigation required".to_string(),
                risk_exposure: if summary.blocker_count > 0 { "High" } else { "Low" }.to_string(),
            },
        }
    }
    
    fn generate_recovery_guidance(&self, errors: &[ErrorDetails]) -> RecoveryGuidance {
        let mut immediate_actions = Vec::new();
        let mut short_term_fixes = Vec::new();
        let mut long_term_improvements = Vec::new();
        let mut prevention_strategies = Vec::new();
        
        // Generate immediate actions for critical errors
        for error in errors.iter().filter(|e| matches!(e.impact, ErrorImpact::Blocker)) {
            immediate_actions.push(ImmediateAction {
                action_id: uuid::Uuid::new_v4().to_string(),
                description: format!("Address critical error: {}", error.error),
                priority: ActionPriority::Critical,
                estimated_time: "1-2 hours".to_string(),
                required_skills: vec!["System administration".to_string()],
                success_criteria: vec!["Error no longer occurs".to_string()],
            });
        }
        
        // Generate short-term fixes
        short_term_fixes.push(ShortTermFix {
            fix_id: "config_validation".to_string(),
            description: "Implement configuration validation".to_string(),
            affected_errors: vec!["Configuration errors".to_string()],
            implementation_steps: vec![
                "Add configuration schema validation".to_string(),
                "Implement configuration file checks".to_string(),
            ],
            testing_requirements: vec!["Test with invalid configurations".to_string()],
            rollback_plan: "Revert to previous configuration handling".to_string(),
        });
        
        // Generate long-term improvements
        long_term_improvements.push(LongTermImprovement {
            improvement_id: "error_prevention".to_string(),
            description: "Implement comprehensive error prevention system".to_string(),
            benefits: vec!["Reduced error rates".to_string(), "Better user experience".to_string()],
            implementation_complexity: FixComplexity::Complex,
            estimated_effort: "2-4 weeks".to_string(),
            success_metrics: vec!["50% reduction in error rates".to_string()],
        });
        
        // Generate prevention strategies
        prevention_strategies.push(PreventionStrategy {
            strategy_id: "proactive_validation".to_string(),
            description: "Implement proactive validation checks".to_string(),
            target_error_types: vec![ErrorCategory::Configuration, ErrorCategory::FileSystem],
            implementation_approach: "Add pre-flight checks before validation".to_string(),
            monitoring_requirements: vec!["Monitor validation success rates".to_string()],
        });
        
        RecoveryGuidance {
            immediate_actions,
            short_term_fixes,
            long_term_improvements,
            prevention_strategies,
        }
    }
    
    fn collect_debugging_information(&self) -> DebuggingInformation {
        DebuggingInformation {
            log_analysis: LogAnalysis {
                log_patterns: Vec::new(),
                error_message_analysis: Vec::new(),
                warning_indicators: Vec::new(),
                performance_indicators: Vec::new(),
            },
            system_state_analysis: SystemStateAnalysis {
                resource_utilization: ResourceUtilization {
                    cpu_usage_percent: 0.0,
                    memory_usage_percent: 0.0,
                    disk_usage_percent: 0.0,
                    io_wait_percent: 0.0,
                },
                process_state: ProcessState {
                    process_count: 0,
                    zombie_processes: 0,
                    high_cpu_processes: Vec::new(),
                    high_memory_processes: Vec::new(),
                },
                network_state: None,
                filesystem_state: FilesystemState {
                    available_space_gb: 0.0,
                    inode_usage_percent: 0.0,
                    mount_points: Vec::new(),
                    filesystem_errors: Vec::new(),
                },
            },
            configuration_analysis: ConfigurationAnalysis {
                configuration_issues: Vec::new(),
                missing_configurations: Vec::new(),
                conflicting_configurations: Vec::new(),
                optimization_opportunities: Vec::new(),
            },
            dependency_analysis: DependencyAnalysis {
                missing_dependencies: Vec::new(),
                version_conflicts: Vec::new(),
                dependency_health: Vec::new(),
            },
        }
    }
    
    /// Export report in specified format
    pub fn export_report(
        &self,
        report: &ComprehensiveErrorReport,
        format: ReportFormat,
        output_path: &std::path::Path,
    ) -> Result<(), ValidationError> {
        match format {
            ReportFormat::Json => {
                let json = serde_json::to_string_pretty(report)
                    .map_err(|e| ValidationError::Serialization { 
                        cause: e.to_string(),
                        recovery_strategy: crate::errors::RecoveryStrategy::FailFast,
                    })?;
                std::fs::write(output_path, json)
                    .map_err(|e| ValidationError::file_system_error(e.to_string(), Some(output_path.to_path_buf())))?;
            },
            ReportFormat::Html => {
                let html = self.generate_html_report(report);
                std::fs::write(output_path, html)
                    .map_err(|e| ValidationError::file_system_error(e.to_string(), Some(output_path.to_path_buf())))?;
            },
            ReportFormat::Text => {
                let text = self.generate_text_report(report);
                std::fs::write(output_path, text)
                    .map_err(|e| ValidationError::file_system_error(e.to_string(), Some(output_path.to_path_buf())))?;
            },
            ReportFormat::Markdown => {
                let markdown = self.generate_markdown_report(report);
                std::fs::write(output_path, markdown)
                    .map_err(|e| ValidationError::file_system_error(e.to_string(), Some(output_path.to_path_buf())))?;
            },
            ReportFormat::Csv => {
                let csv = self.generate_csv_report(report);
                std::fs::write(output_path, csv)
                    .map_err(|e| ValidationError::file_system_error(e.to_string(), Some(output_path.to_path_buf())))?;
            },
        }
        Ok(())
    }
    
    fn generate_html_report(&self, report: &ComprehensiveErrorReport) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Validation Error Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .error {{ background-color: #ffebee; padding: 10px; margin: 10px 0; border-left: 4px solid #f44336; }}
        .warning {{ background-color: #fff3e0; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }}
        .info {{ background-color: #e3f2fd; padding: 10px; margin: 10px 0; border-left: 4px solid #2196f3; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Validation Error Report</h1>
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Total Errors:</strong> {}</p>
        <p><strong>Critical Errors:</strong> {}</p>
        <p><strong>Validation Success:</strong> {}</p>
        <p><strong>Overall Impact:</strong> {}</p>
    </div>
    <h2>Detailed Errors</h2>
    {}
</body>
</html>"#,
            report.executive_summary.total_errors,
            report.executive_summary.critical_errors,
            report.executive_summary.validation_success,
            report.executive_summary.overall_impact,
            report.detailed_errors.iter()
                .map(|e| format!("<div class=\"error\"><h3>{}</h3><p>{}</p></div>", e.error, e.error))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
    
    fn generate_text_report(&self, report: &ComprehensiveErrorReport) -> String {
        format!(
            "VALIDATION ERROR REPORT\n\
             ======================\n\n\
             Generated: {}\n\
             Report ID: {}\n\n\
             EXECUTIVE SUMMARY\n\
             -----------------\n\
             Total Errors: {}\n\
             Critical Errors: {}\n\
             Validation Success: {}\n\
             Overall Impact: {}\n\n\
             DETAILED ERRORS\n\
             ---------------\n\
             {}",
            report.metadata.generated_at,
            report.metadata.report_id,
            report.executive_summary.total_errors,
            report.executive_summary.critical_errors,
            report.executive_summary.validation_success,
            report.executive_summary.overall_impact,
            report.detailed_errors.iter()
                .enumerate()
                .map(|(i, e)| format!("{}. {}\n   Category: {:?}\n   Impact: {:?}\n", i + 1, e.error, e.category, e.impact))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
    
    fn generate_markdown_report(&self, report: &ComprehensiveErrorReport) -> String {
        format!(
            "# Validation Error Report\n\n\
             **Generated:** {}\n\
             **Report ID:** {}\n\n\
             ## Executive Summary\n\n\
             - **Total Errors:** {}\n\
             - **Critical Errors:** {}\n\
             - **Validation Success:** {}\n\
             - **Overall Impact:** {}\n\n\
             ## Detailed Errors\n\n\
             {}",
            report.metadata.generated_at,
            report.metadata.report_id,
            report.executive_summary.total_errors,
            report.executive_summary.critical_errors,
            report.executive_summary.validation_success,
            report.executive_summary.overall_impact,
            report.detailed_errors.iter()
                .enumerate()
                .map(|(i, e)| format!("### {}. {}\n\n- **Category:** {:?}\n- **Impact:** {:?}\n- **Timestamp:** {}\n", i + 1, e.error, e.category, e.impact, e.timestamp))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
    
    fn generate_csv_report(&self, report: &ComprehensiveErrorReport) -> String {
        let mut csv = "Timestamp,Error,Category,Impact,Phase\n".to_string();
        for error in &report.detailed_errors {
            csv.push_str(&format!(
                "{},{:?},{:?},{:?},Unknown\n",
                error.timestamp,
                error.error,
                error.category,
                error.impact
            ));
        }
        csv
    }
}

impl Default for ErrorReportConfig {
    fn default() -> Self {
        Self {
            include_stack_traces: true,
            include_system_info: true,
            include_environment: false,
            include_filesystem_state: true,
            max_similar_errors: 10,
            include_reproduction_steps: true,
            include_suggested_fixes: true,
            output_formats: vec![ReportFormat::Json, ReportFormat::Html],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{ValidationError, ValidationContext};
    use std::time::Duration;
    
    #[test]
    fn test_error_reporter_creation() {
        let config = ErrorReportConfig::default();
        let reporter = ErrorReporter::new(config);
        
        // Should create successfully
        assert_eq!(reporter.aggregator.get_error_summary().total_errors, 0);
    }
    
    #[test]
    fn test_comprehensive_report_generation() {
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
        
        let recovery_manager = crate::errors::ErrorRecoveryManager::new();
        let error = ValidationError::pensieve_binary_not_found(PathBuf::from("/usr/bin/pensieve"));
        let error_details = recovery_manager.create_error_details(error, context);
        
        reporter.add_error(error_details);
        
        let report = reporter.generate_comprehensive_report(None);
        
        assert_eq!(report.executive_summary.total_errors, 1);
        assert_eq!(report.executive_summary.critical_errors, 1);
        assert!(!report.executive_summary.validation_success);
        assert!(!report.detailed_errors.is_empty());
    }
    
    #[test]
    fn test_report_format_generation() {
        let config = ErrorReportConfig::default();
        let reporter = ErrorReporter::new(config);
        
        let report = reporter.generate_comprehensive_report(None);
        
        // Test different format generation
        let html = reporter.generate_html_report(&report);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Validation Error Report"));
        
        let text = reporter.generate_text_report(&report);
        assert!(text.contains("VALIDATION ERROR REPORT"));
        
        let markdown = reporter.generate_markdown_report(&report);
        assert!(markdown.contains("# Validation Error Report"));
        
        let csv = reporter.generate_csv_report(&report);
        assert!(csv.contains("Timestamp,Error,Category,Impact,Phase"));
    }
}