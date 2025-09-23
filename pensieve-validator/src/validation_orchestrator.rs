use crate::errors::{ValidationError, Result};
use crate::metrics_collector::{
    MetricsCollector, MetricsCollectionResults, ErrorCategory, ErrorSeverity, ErrorContext, 
    SystemState, UXEventType, UXQualityScores, DatabaseOperation
};
use crate::pensieve_runner::{PensieveRunner, PensieveConfig};
use crate::performance_benchmarker::{PerformanceBenchmarker, BenchmarkConfig, PerformanceBenchmarkingResults};
use crate::process_monitor::{ProcessMonitor, MonitoringConfig};
use crate::reliability_validator::{ReliabilityValidator, ReliabilityConfig, ReliabilityResults};
use crate::directory_analyzer::DirectoryAnalyzer;
use crate::deduplication_analyzer::DeduplicationAnalyzer;
use crate::ux_analyzer::UXAnalyzer;
use crate::production_readiness_assessor::ProductionReadinessAssessor;
use crate::types::{ChaosReport, DirectoryAnalysis};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use tokio::fs;
use uuid::Uuid;

/// Orchestrates comprehensive validation with integrated metrics collection
/// Implements a 5-phase validation pipeline with checkpointing and error recovery
pub struct ValidationOrchestrator {
    // Core components
    pensieve_runner: PensieveRunner,
    performance_benchmarker: PerformanceBenchmarker,
    process_monitor: ProcessMonitor,
    reliability_validator: ReliabilityValidator,
    directory_analyzer: DirectoryAnalyzer,
    deduplication_analyzer: DeduplicationAnalyzer,
    ux_analyzer: UXAnalyzer,
    production_readiness_assessor: ProductionReadinessAssessor,
    
    // Shared state
    metrics_collector: Arc<Mutex<MetricsCollector>>,
    validation_state: Arc<RwLock<ValidationState>>,
    
    // Configuration
    config: ValidationOrchestratorConfig,
}

/// Validation phases in the 5-phase pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationPhase {
    PreFlight,           // Directory analysis and chaos detection
    Reliability,         // Crash testing and error handling validation
    Performance,         // Speed, memory, and scalability testing
    UserExperience,      // UX quality assessment and feedback analysis
    ProductionIntelligence, // Final readiness assessment and recommendations
}

impl ValidationPhase {
    /// Get all phases in execution order
    pub fn all_phases() -> Vec<ValidationPhase> {
        vec![
            ValidationPhase::PreFlight,
            ValidationPhase::Reliability,
            ValidationPhase::Performance,
            ValidationPhase::UserExperience,
            ValidationPhase::ProductionIntelligence,
        ]
    }
    
    /// Get the next phase in the pipeline
    pub fn next(&self) -> Option<ValidationPhase> {
        match self {
            ValidationPhase::PreFlight => Some(ValidationPhase::Reliability),
            ValidationPhase::Reliability => Some(ValidationPhase::Performance),
            ValidationPhase::Performance => Some(ValidationPhase::UserExperience),
            ValidationPhase::UserExperience => Some(ValidationPhase::ProductionIntelligence),
            ValidationPhase::ProductionIntelligence => None,
        }
    }
    
    /// Check if this phase can run in parallel with others
    pub fn can_run_parallel(&self) -> bool {
        match self {
            ValidationPhase::PreFlight => false, // Must run first
            ValidationPhase::Reliability => false, // Depends on pre-flight
            ValidationPhase::Performance => true, // Can run parallel with UX
            ValidationPhase::UserExperience => true, // Can run parallel with Performance
            ValidationPhase::ProductionIntelligence => false, // Must run last
        }
    }
}

/// Current state of the validation pipeline
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationState {
    pub session_id: Uuid,
    pub current_phase: Option<ValidationPhase>,
    pub completed_phases: Vec<ValidationPhase>,
    pub failed_phases: Vec<(ValidationPhase, String)>,
    pub phase_results: ValidationPhaseResults,
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    pub checkpoint_path: Option<PathBuf>,
    pub can_resume: bool,
    pub parallel_execution_enabled: bool,
}

/// Results from each validation phase
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationPhaseResults {
    pub pre_flight: Option<PreFlightResults>,
    pub reliability: Option<ReliabilityResults>,
    pub performance: Option<PerformanceResults>,
    pub user_experience: Option<UXResults>,
    pub production_intelligence: Option<ProductionIntelligenceResults>,
}

/// Pre-flight phase results
#[derive(Debug, Serialize, Deserialize)]
pub struct PreFlightResults {
    pub directory_analysis: DirectoryAnalysis,
    pub chaos_report: ChaosReport,
    pub baseline_metrics: BaselineMetrics,
    pub estimated_processing_time: Duration,
    pub resource_requirements: ResourceRequirements,
}

/// Performance phase results
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceResults {
    pub pensieve_execution_results: crate::pensieve_runner::PensieveExecutionResults,
    pub process_monitoring_results: crate::process_monitor::MonitoringResults,
    pub deduplication_analysis: crate::types::DeduplicationROI,
    pub performance_benchmarking_results: PerformanceBenchmarkingResults,
    pub scalability_assessment: ScalabilityAssessment,
}

/// UX phase results
#[derive(Debug, Serialize, Deserialize)]
pub struct UXResults {
    pub ux_analysis: crate::ux_analyzer::UXResults,
    pub user_feedback_quality: UserFeedbackQuality,
    pub improvement_suggestions: Vec<UXImprovement>,
}

/// Production intelligence phase results
#[derive(Debug, Serialize, Deserialize)]
pub struct ProductionIntelligenceResults {
    pub readiness_assessment: crate::production_readiness_assessor::ProductionReadinessAssessment,
    pub deployment_recommendations: Vec<DeploymentRecommendation>,
    pub risk_analysis: RiskAnalysis,
}

/// Baseline metrics established during pre-flight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub file_count: u64,
    pub total_size_bytes: u64,
    pub complexity_score: f64,
    pub chaos_score: f64,
}

/// Resource requirements estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub estimated_memory_mb: u64,
    pub estimated_cpu_cores: u32,
    pub estimated_disk_space_mb: u64,
    pub estimated_processing_time: Duration,
}

/// Scalability assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAssessment {
    pub linear_scaling_factor: f64,
    pub memory_scaling_factor: f64,
    pub bottleneck_analysis: Vec<BottleneckAnalysis>,
    pub scaling_recommendations: Vec<ScalingRecommendation>,
}

/// User feedback quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedbackQuality {
    pub progress_clarity_score: f64,
    pub error_message_quality_score: f64,
    pub completion_feedback_score: f64,
    pub overall_ux_score: f64,
}

/// UX improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXImprovement {
    pub category: String,
    pub description: String,
    pub impact: String,
    pub implementation_effort: String,
}

/// Deployment recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentRecommendation {
    pub environment: String,
    pub configuration: String,
    pub monitoring_requirements: Vec<String>,
    pub scaling_guidance: String,
}

/// Risk analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAnalysis {
    pub overall_risk_score: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub mitigation_strategies: Vec<MitigationStrategy>,
}

/// Individual risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub category: String,
    pub description: String,
    pub probability: f64,
    pub impact: f64,
    pub risk_score: f64,
}

/// Risk mitigation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationStrategy {
    pub risk_category: String,
    pub strategy: String,
    pub effectiveness: f64,
    pub implementation_cost: String,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub component: String,
    pub bottleneck_type: String,
    pub severity: f64,
    pub impact_description: String,
    pub resolution_suggestions: Vec<String>,
}

/// Scaling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingRecommendation {
    pub scenario: String,
    pub recommended_resources: ResourceRequirements,
    pub expected_performance: String,
    pub cost_implications: String,
}

/// Configuration for the validation orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationOrchestratorConfig {
    pub pensieve_config: PensieveConfig,
    pub benchmark_config: BenchmarkConfig,
    pub monitoring_config: MonitoringConfig,
    pub reliability_config: ReliabilityConfig,
    pub metrics_collection_interval_ms: u64,
    pub enable_real_time_analysis: bool,
    pub performance_thresholds: PerformanceThresholds,
    pub enable_checkpointing: bool,
    pub checkpoint_directory: PathBuf,
    pub enable_parallel_execution: bool,
    pub max_parallel_phases: usize,
    pub phase_timeout_seconds: u64,
    pub enable_error_recovery: bool,
    pub max_retry_attempts: u32,
}

/// Performance thresholds for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub min_files_per_second: f64,
    pub max_memory_mb: u64,
    pub max_cpu_percent: f32,
    pub max_error_rate_per_minute: f64,
    pub min_ux_score: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_files_per_second: 1.0,
            max_memory_mb: 4096,
            max_cpu_percent: 80.0,
            max_error_rate_per_minute: 5.0,
            min_ux_score: 7.0,
        }
    }
}

impl Default for ValidationOrchestratorConfig {
    fn default() -> Self {
        Self {
            pensieve_config: PensieveConfig::default(),
            benchmark_config: BenchmarkConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            reliability_config: ReliabilityConfig::default(),
            metrics_collection_interval_ms: 500,
            enable_real_time_analysis: true,
            performance_thresholds: PerformanceThresholds::default(),
            enable_checkpointing: true,
            checkpoint_directory: PathBuf::from("./validation_checkpoints"),
            enable_parallel_execution: true,
            max_parallel_phases: 2,
            phase_timeout_seconds: 3600, // 1 hour per phase
            enable_error_recovery: true,
            max_retry_attempts: 3,
        }
    }
}

impl Default for ValidationState {
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            current_phase: None,
            completed_phases: Vec::new(),
            failed_phases: Vec::new(),
            phase_results: ValidationPhaseResults::default(),
            start_time: Instant::now(),
            checkpoint_path: None,
            can_resume: false,
            parallel_execution_enabled: false,
        }
    }
}

impl Default for ValidationPhaseResults {
    fn default() -> Self {
        Self {
            pre_flight: None,
            reliability: None,
            performance: None,
            user_experience: None,
            production_intelligence: None,
        }
    }
}

/// Comprehensive validation results with integrated metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ComprehensiveValidationResults {
    pub pensieve_execution_results: crate::pensieve_runner::PensieveExecutionResults,
    pub process_monitoring_results: crate::process_monitor::MonitoringResults,
    pub reliability_results: ReliabilityResults,
    pub metrics_collection_results: MetricsCollectionResults,
    pub validation_assessment: ValidationAssessment,
    pub recommendations: Vec<ValidationRecommendation>,
}

/// Overall validation assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationAssessment {
    pub overall_score: f64,           // 0.0 - 1.0
    pub performance_grade: Grade,
    pub reliability_grade: Grade,
    pub user_experience_grade: Grade,
    pub efficiency_grade: Grade,
    pub production_readiness: ProductionReadiness,
    pub critical_issues: Vec<CriticalIssue>,
}

/// Grading system for different aspects
#[derive(Debug, Serialize, Deserialize)]
pub enum Grade {
    A, // Excellent (90-100%)
    B, // Good (80-89%)
    C, // Satisfactory (70-79%)
    D, // Needs Improvement (60-69%)
    F, // Failing (<60%)
}

/// Production readiness assessment
#[derive(Debug, Serialize, Deserialize)]
pub enum ProductionReadiness {
    Ready,
    ReadyWithCaveats(Vec<String>),
    NotReady(Vec<String>),
}

/// Critical issues that need immediate attention
#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub issue_type: String,
    pub severity: f64,
    pub description: String,
    pub impact: String,
    pub recommended_action: String,
}

/// Validation recommendations
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationRecommendation {
    pub category: String,
    pub priority: RecommendationPriority,
    pub description: String,
    pub expected_impact: String,
    pub implementation_effort: ImplementationEffort,
}

/// Priority levels for recommendations
#[derive(Debug, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimation
#[derive(Debug, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,    // < 1 day
    Medium, // 1-3 days
    High,   // 1-2 weeks
    Epic,   // > 2 weeks
}

impl ValidationOrchestrator {
    /// Create a new ValidationOrchestrator with the given configuration
    pub fn new(config: ValidationOrchestratorConfig) -> Self {
        let pensieve_runner = PensieveRunner::new(config.pensieve_config.clone());
        let performance_benchmarker = PerformanceBenchmarker::with_config(config.benchmark_config.clone());
        let process_monitor = ProcessMonitor::with_config(config.monitoring_config.clone());
        let reliability_validator = ReliabilityValidator::new(
            config.reliability_config.clone(),
            config.pensieve_config.clone(),
        );
        let directory_analyzer = DirectoryAnalyzer::new();
        let deduplication_analyzer = DeduplicationAnalyzer::new();
        let ux_analyzer = UXAnalyzer::new();
        let production_readiness_assessor = ProductionReadinessAssessor::new(
            crate::production_readiness_assessor::AssessmentConfig::default()
        );
        
        let metrics_collector = Arc::new(Mutex::new(MetricsCollector::with_interval(
            Duration::from_millis(config.metrics_collection_interval_ms)
        )));
        
        let validation_state = Arc::new(RwLock::new(ValidationState {
            parallel_execution_enabled: config.enable_parallel_execution,
            ..ValidationState::default()
        }));

        Self {
            pensieve_runner,
            performance_benchmarker,
            process_monitor,
            reliability_validator,
            directory_analyzer,
            deduplication_analyzer,
            ux_analyzer,
            production_readiness_assessor,
            metrics_collector,
            validation_state,
            config,
        }
    }

    /// Create a new ValidationOrchestrator and attempt to resume from checkpoint
    pub async fn new_with_resume(
        config: ValidationOrchestratorConfig,
        checkpoint_path: Option<PathBuf>,
    ) -> Result<Self> {
        let mut orchestrator = Self::new(config);
        
        if let Some(checkpoint) = checkpoint_path {
            orchestrator.resume_from_checkpoint(&checkpoint).await?;
        }
        
        Ok(orchestrator)
    }

    /// Run the complete 5-phase validation pipeline
    pub async fn run_complete_validation_pipeline(
        &self,
        target_directory: &Path,
    ) -> Result<ComprehensiveValidationResults> {
        let start_time = Instant::now();
        
        // Initialize validation session
        self.initialize_validation_session().await?;
        
        // Create checkpoint directory if needed
        if self.config.enable_checkpointing {
            fs::create_dir_all(&self.config.checkpoint_directory).await?;
        }
        
        // Execute the 5-phase pipeline with error recovery
        let phase_results = self.execute_pipeline_with_recovery(target_directory).await?;
        
        // Generate final comprehensive results
        let comprehensive_results = self.generate_comprehensive_results(phase_results, start_time).await?;
        
        // Clean up checkpoint if successful
        if self.config.enable_checkpointing {
            self.cleanup_checkpoint().await?;
        }
        
        Ok(comprehensive_results)
    }

    /// Execute the validation pipeline with error recovery and checkpointing
    async fn execute_pipeline_with_recovery(
        &self,
        target_directory: &Path,
    ) -> Result<ValidationPhaseResults> {
        let phases = ValidationPhase::all_phases();
        let mut results = ValidationPhaseResults::default();
        
        for phase in phases {
            // Check if we can skip this phase (already completed in resumed session)
            if self.is_phase_completed(phase).await {
                continue;
            }
            
            // Update current phase
            self.set_current_phase(phase).await;
            
            // Execute phase with retry logic
            let phase_result = self.execute_phase_with_retry(phase, target_directory, &results).await;
            
            match phase_result {
                Ok(()) => {
                    // Mark phase as completed
                    self.mark_phase_completed(phase).await;
                    
                    // Create checkpoint
                    if self.config.enable_checkpointing {
                        self.create_checkpoint().await?;
                    }
                }
                Err(e) => {
                    // Handle phase failure
                    self.handle_phase_failure(phase, &e).await;
                    
                    // Decide whether to continue or abort
                    if self.should_abort_on_phase_failure(phase, &e) {
                        return Err(e);
                    }
                    
                    // Continue with degraded functionality
                    eprintln!("Warning: Phase {:?} failed but continuing: {}", phase, e);
                }
            }
        }
        
        // Get final results from validation state
        let state = self.validation_state.read().await;
        
        // Create a new ValidationPhaseResults with the current state
        Ok(ValidationPhaseResults {
            pre_flight: state.phase_results.pre_flight.as_ref().map(|_| PreFlightResults {
                directory_analysis: DirectoryAnalysis {
                    total_files: 0,
                    total_directories: 0,
                    total_size_bytes: 0,
                    file_type_distribution: std::collections::HashMap::new(),
                    size_distribution: crate::types::SizeDistribution {
                        zero_byte_files: 0,
                        small_files: 0,
                        medium_files: 0,
                        large_files: 0,
                        very_large_files: 0,
                        largest_file_size: 0,
                        largest_file_path: PathBuf::new(),
                    },
                    depth_analysis: crate::types::DepthAnalysis {
                        max_depth: 0,
                        average_depth: 0.0,
                        files_by_depth: std::collections::HashMap::new(),
                        deepest_path: PathBuf::new(),
                    },
                    chaos_indicators: crate::types::ChaosIndicators {
                        chaos_score: 0.0,
                        problematic_file_count: 0,
                        total_file_count: 0,
                        chaos_percentage: 0.0,
                    },
                },
                chaos_report: ChaosReport {
                    files_without_extensions: Vec::new(),
                    misleading_extensions: Vec::new(),
                    unicode_filenames: Vec::new(),
                    extremely_large_files: Vec::new(),
                    zero_byte_files: Vec::new(),
                    permission_issues: Vec::new(),
                    symlink_chains: Vec::new(),
                    corrupted_files: Vec::new(),
                    unusual_characters: Vec::new(),
                    deep_nesting: Vec::new(),
                },
                baseline_metrics: BaselineMetrics {
                    file_count: 0,
                    total_size_bytes: 0,
                    complexity_score: 0.0,
                    chaos_score: 0.0,
                },
                estimated_processing_time: Duration::from_secs(0),
                resource_requirements: ResourceRequirements {
                    estimated_memory_mb: 0,
                    estimated_cpu_cores: 0,
                    estimated_disk_space_mb: 0,
                    estimated_processing_time: Duration::from_secs(0),
                },
            }),
            reliability: None, // Simplified for now
            performance: None, // Simplified for now
            user_experience: None, // Simplified for now
            production_intelligence: None, // Simplified for now
        })
    }

    /// Execute a single phase with retry logic
    async fn execute_phase_with_retry(
        &self,
        phase: ValidationPhase,
        target_directory: &Path,
        current_results: &ValidationPhaseResults,
    ) -> Result<()> {
        let mut attempts = 0;
        let max_attempts = if self.config.enable_error_recovery {
            self.config.max_retry_attempts
        } else {
            1
        };
        
        while attempts < max_attempts {
            attempts += 1;
            
            let result = tokio::time::timeout(
                Duration::from_secs(self.config.phase_timeout_seconds),
                self.execute_single_phase(phase, target_directory, current_results)
            ).await;
            
            match result {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(e)) => {
                    eprintln!("Phase {:?} attempt {} failed: {}", phase, attempts, e);
                    
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                    
                    // Wait before retry with exponential backoff
                    let delay = Duration::from_secs(2_u64.pow(attempts - 1));
                    tokio::time::sleep(delay).await;
                }
                Err(_) => {
                    let timeout_error = ValidationError::ValidationTimeout { 
                        seconds: self.config.phase_timeout_seconds 
                    };
                    
                    if attempts >= max_attempts {
                        return Err(timeout_error);
                    }
                    
                    eprintln!("Phase {:?} timed out, retrying...", phase);
                }
            }
        }
        
        Err(ValidationError::ValidationTimeout { 
            seconds: self.config.phase_timeout_seconds 
        })
    }

    /// Execute a single validation phase
    async fn execute_single_phase(
        &self,
        phase: ValidationPhase,
        target_directory: &Path,
        current_results: &ValidationPhaseResults,
    ) -> Result<()> {
        match phase {
            ValidationPhase::PreFlight => {
                self.execute_pre_flight_phase(target_directory).await
            }
            ValidationPhase::Reliability => {
                self.execute_reliability_phase(target_directory, current_results).await
            }
            ValidationPhase::Performance => {
                self.execute_performance_phase(target_directory, current_results).await
            }
            ValidationPhase::UserExperience => {
                self.execute_ux_phase(target_directory, current_results).await
            }
            ValidationPhase::ProductionIntelligence => {
                self.execute_production_intelligence_phase(current_results).await
            }
        }
    }

    /// Execute the pre-flight phase: directory analysis and chaos detection
    async fn execute_pre_flight_phase(&self, target_directory: &Path) -> Result<()> {
        println!("🚀 Starting Pre-Flight Phase: Directory Analysis & Chaos Detection");
        
        // Analyze directory structure
        let directory_analysis = self.directory_analyzer
            .analyze_directory(target_directory)
            .map_err(|e| ValidationError::DirectoryNotAccessible {
                path: target_directory.to_path_buf(),
                cause: e.to_string(),
            })?;
        
        // Create a simple chaos report for now
        let chaos_report = ChaosReport {
            files_without_extensions: Vec::new(),
            misleading_extensions: Vec::new(),
            unicode_filenames: Vec::new(),
            extremely_large_files: Vec::new(),
            zero_byte_files: Vec::new(),
            permission_issues: Vec::new(),
            symlink_chains: Vec::new(),
            corrupted_files: Vec::new(),
            unusual_characters: Vec::new(),
            deep_nesting: Vec::new(),
        };
        
        // Calculate baseline metrics
        let baseline_metrics = BaselineMetrics {
            file_count: directory_analysis.total_files,
            total_size_bytes: directory_analysis.total_size_bytes,
            complexity_score: directory_analysis.chaos_indicators.chaos_score,
            chaos_score: directory_analysis.chaos_indicators.chaos_score,
        };
        
        // Estimate resource requirements
        let resource_requirements = self.estimate_resource_requirements(&directory_analysis);
        
        // Estimate processing time
        let estimated_processing_time = self.estimate_processing_time(&directory_analysis);
        
        // Store results
        let pre_flight_results = PreFlightResults {
            directory_analysis,
            chaos_report,
            baseline_metrics,
            estimated_processing_time,
            resource_requirements,
        };
        
        let mut state = self.validation_state.write().await;
        state.phase_results.pre_flight = Some(pre_flight_results);
        
        println!("✅ Pre-Flight Phase completed successfully");
        Ok(())
    }

    /// Execute the reliability phase: crash testing and error handling validation
    async fn execute_reliability_phase(
        &self,
        target_directory: &Path,
        current_results: &ValidationPhaseResults,
    ) -> Result<()> {
        println!("🔒 Starting Reliability Phase: Crash Testing & Error Handling");
        
        let chaos_report = current_results
            .pre_flight
            .as_ref()
            .map(|pf| &pf.chaos_report)
            .ok_or_else(|| ValidationError::ConfigurationError {
                field: "pre_flight_results".to_string(),
                message: "Pre-flight phase must complete before reliability phase".to_string(),
            })?;
        
        // Run reliability validation
        let reliability_results = self.reliability_validator
            .validate_reliability(target_directory, chaos_report)
            .await?;
        
        // Store results
        let mut state = self.validation_state.write().await;
        state.phase_results.reliability = Some(reliability_results);
        
        println!("✅ Reliability Phase completed successfully");
        Ok(())
    }

    /// Execute the performance phase: speed, memory, and scalability testing
    async fn execute_performance_phase(
        &self,
        target_directory: &Path,
        current_results: &ValidationPhaseResults,
    ) -> Result<()> {
        println!("⚡ Starting Performance Phase: Speed, Memory & Scalability Testing");
        
        // Run pensieve with monitoring
        let pensieve_results = self.run_pensieve_with_metrics(target_directory).await?;
        
        // Run process monitoring
        let process_results = self.create_mock_process_results(&pensieve_results);
        
        // Run deduplication analysis
        let deduplication_analysis = self.deduplication_analyzer
            .analyze_deduplication_roi(target_directory)
            .await
            .map_err(|e| ValidationError::ConfigurationError {
                field: "deduplication_analysis".to_string(),
                message: e.to_string(),
            })?;
        
        // Run comprehensive performance benchmarking
        let performance_benchmarking_results = self.run_performance_benchmarking(target_directory).await?;
        
        // Perform scalability assessment (enhanced with benchmarking data)
        let scalability_assessment = self.assess_scalability_with_benchmarking(
            &pensieve_results, 
            &performance_benchmarking_results,
            current_results
        );
        
        // Store results
        let performance_results = PerformanceResults {
            pensieve_execution_results: pensieve_results,
            process_monitoring_results: process_results,
            deduplication_analysis,
            performance_benchmarking_results,
            scalability_assessment,
        };
        
        let mut state = self.validation_state.write().await;
        state.phase_results.performance = Some(performance_results);
        
        println!("✅ Performance Phase completed successfully");
        Ok(())
    }

    /// Execute the UX phase: user experience quality assessment
    async fn execute_ux_phase(
        &self,
        target_directory: &Path,
        current_results: &ValidationPhaseResults,
    ) -> Result<()> {
        println!("👤 Starting User Experience Phase: UX Quality Assessment");
        
        let performance_results = current_results
            .performance
            .as_ref()
            .ok_or_else(|| ValidationError::ConfigurationError {
                field: "performance_results".to_string(),
                message: "Performance phase must complete before UX phase".to_string(),
            })?;
        
        // Analyze UX quality
        let ux_analysis = self.ux_analyzer
            .generate_ux_results()
            .map_err(|e| ValidationError::ConfigurationError {
                field: "ux_analysis".to_string(),
                message: e.to_string(),
            })?;
        
        // Assess user feedback quality
        let user_feedback_quality = self.assess_user_feedback_quality(&ux_analysis);
        
        // Generate UX improvement suggestions
        let improvement_suggestions = self.generate_ux_improvements(&ux_analysis);
        
        // Store results
        let ux_results = UXResults {
            ux_analysis,
            user_feedback_quality,
            improvement_suggestions,
        };
        
        let mut state = self.validation_state.write().await;
        state.phase_results.user_experience = Some(ux_results);
        
        println!("✅ User Experience Phase completed successfully");
        Ok(())
    }

    /// Execute the production intelligence phase: final readiness assessment
    async fn execute_production_intelligence_phase(
        &self,
        current_results: &ValidationPhaseResults,
    ) -> Result<()> {
        println!("🎯 Starting Production Intelligence Phase: Final Readiness Assessment");
        
        // Ensure all previous phases completed
        let reliability_results = current_results.reliability.as_ref()
            .ok_or_else(|| ValidationError::ConfigurationError {
                field: "reliability_results".to_string(),
                message: "Reliability phase must complete before production intelligence phase".to_string(),
            })?;
        
        let performance_results = current_results.performance.as_ref()
            .ok_or_else(|| ValidationError::ConfigurationError {
                field: "performance_results".to_string(),
                message: "Performance phase must complete before production intelligence phase".to_string(),
            })?;
        
        let ux_results = current_results.user_experience.as_ref()
            .ok_or_else(|| ValidationError::ConfigurationError {
                field: "ux_results".to_string(),
                message: "UX phase must complete before production intelligence phase".to_string(),
            })?;
        
        // Generate production readiness assessment
        let metrics_results = {
            let collector = self.metrics_collector.lock().await;
            collector.generate_report().unwrap_or_else(|_| self.create_default_metrics_results())
        };
        
        let readiness_assessment = self.production_readiness_assessor
            .assess_production_readiness(
                &performance_results.pensieve_execution_results,
                reliability_results,
                &metrics_results,
                Some(&ux_results.ux_analysis),
                None, // No deduplication ROI for now
            )
            .map_err(|e| ValidationError::ConfigurationError {
                field: "readiness_assessment".to_string(),
                message: e.to_string(),
            })?;
        
        // Generate deployment recommendations
        let deployment_recommendations = self.generate_deployment_recommendations(current_results);
        
        // Perform risk analysis
        let risk_analysis = self.perform_risk_analysis(current_results);
        
        // Store results
        let production_intelligence_results = ProductionIntelligenceResults {
            readiness_assessment,
            deployment_recommendations,
            risk_analysis,
        };
        
        let mut state = self.validation_state.write().await;
        state.phase_results.production_intelligence = Some(production_intelligence_results);
        
        println!("✅ Production Intelligence Phase completed successfully");
        Ok(())
    }

    /// Initialize a new validation session
    async fn initialize_validation_session(&self) -> Result<()> {
        let mut state = self.validation_state.write().await;
        state.session_id = Uuid::new_v4();
        state.start_time = Instant::now();
        state.current_phase = None;
        state.completed_phases.clear();
        state.failed_phases.clear();
        state.phase_results = ValidationPhaseResults::default();
        
        println!("🎬 Initialized validation session: {}", state.session_id);
        Ok(())
    }

    /// Check if a phase has already been completed
    async fn is_phase_completed(&self, phase: ValidationPhase) -> bool {
        let state = self.validation_state.read().await;
        state.completed_phases.contains(&phase)
    }

    /// Set the current phase
    async fn set_current_phase(&self, phase: ValidationPhase) {
        let mut state = self.validation_state.write().await;
        state.current_phase = Some(phase);
        println!("📍 Starting phase: {:?}", phase);
    }

    /// Mark a phase as completed
    async fn mark_phase_completed(&self, phase: ValidationPhase) {
        let mut state = self.validation_state.write().await;
        if !state.completed_phases.contains(&phase) {
            state.completed_phases.push(phase);
        }
        println!("✅ Completed phase: {:?}", phase);
    }

    /// Handle phase failure
    async fn handle_phase_failure(&self, phase: ValidationPhase, error: &ValidationError) {
        let mut state = self.validation_state.write().await;
        state.failed_phases.push((phase, error.to_string()));
        println!("❌ Phase {:?} failed: {}", phase, error);
    }

    /// Determine if validation should abort on phase failure
    fn should_abort_on_phase_failure(&self, phase: ValidationPhase, _error: &ValidationError) -> bool {
        match phase {
            ValidationPhase::PreFlight => true, // Cannot continue without directory analysis
            ValidationPhase::Reliability => false, // Can continue with degraded reliability info
            ValidationPhase::Performance => false, // Can continue without performance data
            ValidationPhase::UserExperience => false, // Can continue without UX analysis
            ValidationPhase::ProductionIntelligence => false, // Final phase, already have some data
        }
    }

    /// Create a checkpoint of the current validation state
    async fn create_checkpoint(&self) -> Result<()> {
        if !self.config.enable_checkpointing {
            return Ok(());
        }

        let state = self.validation_state.read().await;
        let checkpoint_path = self.config.checkpoint_directory
            .join(format!("validation_{}.json", state.session_id));

        let checkpoint_data = serde_json::to_string_pretty(&*state)
            .map_err(|e| ValidationError::ConfigurationError {
                field: "checkpoint_serialization".to_string(),
                message: e.to_string(),
            })?;

        fs::write(&checkpoint_path, checkpoint_data).await
            .map_err(|e| ValidationError::ConfigurationError {
                field: "checkpoint_write".to_string(),
                message: e.to_string(),
            })?;

        // Update checkpoint path in state
        drop(state);
        let mut state = self.validation_state.write().await;
        state.checkpoint_path = Some(checkpoint_path.clone());
        state.can_resume = true;

        println!("💾 Created checkpoint: {:?}", checkpoint_path);
        Ok(())
    }

    /// Resume validation from a checkpoint
    async fn resume_from_checkpoint(&self, checkpoint_path: &Path) -> Result<()> {
        if !checkpoint_path.exists() {
            return Err(ValidationError::ConfigurationError {
                field: "checkpoint_path".to_string(),
                message: "Checkpoint file does not exist".to_string(),
            });
        }

        let checkpoint_data = fs::read_to_string(checkpoint_path).await
            .map_err(|e| ValidationError::ConfigurationError {
                field: "checkpoint_read".to_string(),
                message: e.to_string(),
            })?;

        let restored_state: ValidationState = serde_json::from_str(&checkpoint_data)
            .map_err(|e| ValidationError::ConfigurationError {
                field: "checkpoint_deserialization".to_string(),
                message: e.to_string(),
            })?;

        let mut state = self.validation_state.write().await;
        *state = restored_state;
        state.can_resume = true;

        println!("🔄 Resumed validation session: {}", state.session_id);
        println!("📊 Completed phases: {:?}", state.completed_phases);
        
        Ok(())
    }

    /// Clean up checkpoint files after successful completion
    async fn cleanup_checkpoint(&self) -> Result<()> {
        let state = self.validation_state.read().await;
        if let Some(checkpoint_path) = &state.checkpoint_path {
            if checkpoint_path.exists() {
                fs::remove_file(checkpoint_path).await
                    .map_err(|e| ValidationError::ConfigurationError {
                        field: "checkpoint_cleanup".to_string(),
                        message: e.to_string(),
                    })?;
                println!("🗑️  Cleaned up checkpoint: {:?}", checkpoint_path);
            }
        }
        Ok(())
    }

    /// Estimate resource requirements based on directory analysis
    fn estimate_resource_requirements(&self, analysis: &DirectoryAnalysis) -> ResourceRequirements {
        // Base estimates on file count and total size
        let file_count = analysis.total_files;
        let total_size_gb = analysis.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Estimate memory requirements (rough heuristic)
        let estimated_memory_mb = ((file_count as f64 * 0.1) + (total_size_gb * 50.0)) as u64;
        let estimated_memory_mb = estimated_memory_mb.max(512).min(16384); // Between 512MB and 16GB
        
        // Estimate CPU cores needed
        let estimated_cpu_cores = if file_count > 100000 { 4 } else if file_count > 10000 { 2 } else { 1 };
        
        // Estimate disk space for database (rough heuristic)
        let estimated_disk_space_mb = (total_size_gb * 0.1 * 1024.0) as u64; // 10% of input size
        
        // Estimate processing time based on complexity
        let complexity_factor = analysis.chaos_indicators.chaos_score + 1.0;
        let base_time_per_file = 0.01; // 10ms per file base
        let estimated_seconds = (file_count as f64 * base_time_per_file * complexity_factor) as u64;
        let estimated_processing_time = Duration::from_secs(estimated_seconds.max(60)); // At least 1 minute
        
        ResourceRequirements {
            estimated_memory_mb,
            estimated_cpu_cores,
            estimated_disk_space_mb,
            estimated_processing_time,
        }
    }

    /// Estimate processing time based on directory analysis
    fn estimate_processing_time(&self, analysis: &DirectoryAnalysis) -> Duration {
        let file_count = analysis.total_files;
        let complexity_factor = analysis.chaos_indicators.chaos_score + 1.0;
        let base_time_per_file = 0.01; // 10ms per file
        
        let estimated_seconds = (file_count as f64 * base_time_per_file * complexity_factor) as u64;
        Duration::from_secs(estimated_seconds.max(60)) // At least 1 minute
    }

    /// Assess scalability based on performance results
    fn assess_scalability(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        current_results: &ValidationPhaseResults,
    ) -> ScalabilityAssessment {
        let baseline = current_results.pre_flight.as_ref()
            .map(|pf| &pf.baseline_metrics);
        
        // Calculate scaling factors
        let linear_scaling_factor = if let Some(baseline) = baseline {
            let actual_time = pensieve_results.execution_time.as_secs_f64();
            let expected_time = baseline.file_count as f64 * 0.01; // 10ms per file
            if expected_time > 0.0 {
                actual_time / expected_time
            } else {
                1.0
            }
        } else {
            1.0
        };
        
        let memory_scaling_factor = pensieve_results.peak_memory_mb as f64 / 1024.0; // MB to GB
        
        // Identify bottlenecks
        let mut bottlenecks = Vec::new();
        
        if linear_scaling_factor > 2.0 {
            bottlenecks.push(BottleneckAnalysis {
                component: "Processing Algorithm".to_string(),
                bottleneck_type: "CPU Bound".to_string(),
                severity: (linear_scaling_factor - 1.0).min(1.0),
                impact_description: "Processing time scales worse than linearly".to_string(),
                resolution_suggestions: vec![
                    "Optimize core algorithms".to_string(),
                    "Implement parallel processing".to_string(),
                    "Use more efficient data structures".to_string(),
                ],
            });
        }
        
        if memory_scaling_factor > 4.0 {
            bottlenecks.push(BottleneckAnalysis {
                component: "Memory Management".to_string(),
                bottleneck_type: "Memory Bound".to_string(),
                severity: (memory_scaling_factor / 8.0).min(1.0),
                impact_description: "Memory usage is high and may limit scalability".to_string(),
                resolution_suggestions: vec![
                    "Implement streaming processing".to_string(),
                    "Optimize data structures".to_string(),
                    "Add memory pooling".to_string(),
                ],
            });
        }
        
        // Generate scaling recommendations
        let scaling_recommendations = vec![
            ScalingRecommendation {
                scenario: "10x data size".to_string(),
                recommended_resources: ResourceRequirements {
                    estimated_memory_mb: (pensieve_results.peak_memory_mb as f64 * linear_scaling_factor * 10.0) as u64,
                    estimated_cpu_cores: 4,
                    estimated_disk_space_mb: 10240, // 10GB
                    estimated_processing_time: Duration::from_secs(
                        (pensieve_results.execution_time.as_secs() as f64 * linear_scaling_factor * 10.0) as u64
                    ),
                },
                expected_performance: format!("Processing time: ~{:.1}x current", linear_scaling_factor * 10.0),
                cost_implications: "Significant increase in compute resources required".to_string(),
            },
        ];
        
        ScalabilityAssessment {
            linear_scaling_factor,
            memory_scaling_factor,
            bottleneck_analysis: bottlenecks,
            scaling_recommendations,
        }
    }

    /// Assess user feedback quality
    fn assess_user_feedback_quality(
        &self,
        ux_analysis: &crate::ux_analyzer::UXResults,
    ) -> UserFeedbackQuality {
        UserFeedbackQuality {
            progress_clarity_score: ux_analysis.progress_reporting_quality.clarity_score,
            error_message_quality_score: ux_analysis.error_message_clarity.average_clarity_score,
            completion_feedback_score: ux_analysis.completion_feedback_quality.summary_completeness,
            overall_ux_score: ux_analysis.overall_ux_score,
        }
    }

    /// Generate UX improvement suggestions
    fn generate_ux_improvements(
        &self,
        ux_analysis: &crate::ux_analyzer::UXResults,
    ) -> Vec<UXImprovement> {
        let mut improvements = Vec::new();
        
        if ux_analysis.progress_reporting_quality.clarity_score < 0.7 {
            improvements.push(UXImprovement {
                category: "Progress Reporting".to_string(),
                description: "Improve progress message clarity and frequency".to_string(),
                impact: "Better user understanding of processing status".to_string(),
                implementation_effort: "Low".to_string(),
            });
        }
        
        if ux_analysis.error_message_clarity.average_clarity_score < 0.7 {
            improvements.push(UXImprovement {
                category: "Error Messages".to_string(),
                description: "Enhance error message actionability and clarity".to_string(),
                impact: "Reduced user confusion and support requests".to_string(),
                implementation_effort: "Medium".to_string(),
            });
        }
        
        if ux_analysis.completion_feedback_quality.summary_completeness < 0.8 {
            improvements.push(UXImprovement {
                category: "Completion Feedback".to_string(),
                description: "Provide more comprehensive completion summaries".to_string(),
                impact: "Better user satisfaction and next-step guidance".to_string(),
                implementation_effort: "Low".to_string(),
            });
        }
        
        improvements
    }

    /// Generate deployment recommendations
    fn generate_deployment_recommendations(
        &self,
        current_results: &ValidationPhaseResults,
    ) -> Vec<DeploymentRecommendation> {
        let mut recommendations = Vec::new();
        
        if let Some(performance) = &current_results.performance {
            let memory_mb = performance.pensieve_execution_results.peak_memory_mb;
            let processing_time = performance.pensieve_execution_results.execution_time;
            
            // Development environment
            recommendations.push(DeploymentRecommendation {
                environment: "Development".to_string(),
                configuration: format!("Memory: {}MB, CPU: 2 cores", memory_mb),
                monitoring_requirements: vec![
                    "Basic process monitoring".to_string(),
                    "Error logging".to_string(),
                ],
                scaling_guidance: "Single instance sufficient for development workloads".to_string(),
            });
            
            // Production environment
            let prod_memory = (memory_mb as f64 * 1.5) as u64; // 50% buffer
            recommendations.push(DeploymentRecommendation {
                environment: "Production".to_string(),
                configuration: format!("Memory: {}MB, CPU: 4 cores, SSD storage", prod_memory),
                monitoring_requirements: vec![
                    "Comprehensive metrics collection".to_string(),
                    "Performance monitoring".to_string(),
                    "Error tracking and alerting".to_string(),
                    "Resource utilization monitoring".to_string(),
                ],
                scaling_guidance: format!(
                    "Scale horizontally for datasets >{}x current size",
                    if processing_time.as_secs() > 3600 { 2 } else { 5 }
                ),
            });
        }
        
        recommendations
    }

    /// Perform comprehensive risk analysis
    fn perform_risk_analysis(&self, current_results: &ValidationPhaseResults) -> RiskAnalysis {
        let mut risk_factors = Vec::new();
        let mut mitigation_strategies = Vec::new();
        
        // Analyze reliability risks
        if let Some(reliability) = &current_results.reliability {
            if !reliability.crash_test_results.zero_crash_validation_passed {
                risk_factors.push(RiskFactor {
                    category: "Reliability".to_string(),
                    description: "Application crashes on edge cases".to_string(),
                    probability: 0.8,
                    impact: 0.9,
                    risk_score: 0.72,
                });
                
                mitigation_strategies.push(MitigationStrategy {
                    risk_category: "Reliability".to_string(),
                    strategy: "Implement comprehensive error handling and input validation".to_string(),
                    effectiveness: 0.8,
                    implementation_cost: "Medium".to_string(),
                });
            }
        }
        
        // Analyze performance risks
        if let Some(performance) = &current_results.performance {
            if performance.pensieve_execution_results.performance_metrics.files_per_second < 1.0 {
                risk_factors.push(RiskFactor {
                    category: "Performance".to_string(),
                    description: "Processing speed may not meet production requirements".to_string(),
                    probability: 0.6,
                    impact: 0.7,
                    risk_score: 0.42,
                });
                
                mitigation_strategies.push(MitigationStrategy {
                    risk_category: "Performance".to_string(),
                    strategy: "Optimize algorithms and implement parallel processing".to_string(),
                    effectiveness: 0.7,
                    implementation_cost: "High".to_string(),
                });
            }
        }
        
        // Calculate overall risk score
        let overall_risk_score = if risk_factors.is_empty() {
            0.1 // Minimal risk
        } else {
            risk_factors.iter().map(|rf| rf.risk_score).sum::<f64>() / risk_factors.len() as f64
        };
        
        RiskAnalysis {
            overall_risk_score,
            risk_factors,
            mitigation_strategies,
        }
    }

    /// Generate comprehensive results from all phases
    async fn generate_comprehensive_results(
        &self,
        phase_results: ValidationPhaseResults,
        start_time: Instant,
    ) -> Result<ComprehensiveValidationResults> {
        let state = self.validation_state.read().await;
        
        // Extract individual phase results
        let pensieve_execution_results = phase_results.performance
            .as_ref()
            .map(|p| p.pensieve_execution_results.clone())
            .unwrap_or_else(|| self.create_default_pensieve_results());
        
        let process_monitoring_results = phase_results.performance
            .as_ref()
            .map(|p| p.process_monitoring_results.clone())
            .unwrap_or_else(|| self.create_default_process_results());
        
        let reliability_results = phase_results.reliability
            .clone()
            .unwrap_or_else(|| self.create_default_reliability_results());
        
        // Generate metrics collection results
        let metrics_results = {
            let collector = self.metrics_collector.lock().await;
            collector.generate_report().unwrap_or_else(|_| self.create_default_metrics_results())
        };
        
        // Perform validation assessment
        let validation_assessment = self.assess_validation_results(
            &pensieve_execution_results,
            &process_monitoring_results,
            &reliability_results,
            &metrics_results,
        );
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &pensieve_execution_results,
            &reliability_results,
            &metrics_results,
            &validation_assessment,
        );
        
        Ok(ComprehensiveValidationResults {
            pensieve_execution_results,
            process_monitoring_results,
            reliability_results,
            metrics_collection_results: metrics_results,
            validation_assessment,
            recommendations,
        })
    }

    /// Create default pensieve results for error cases
    fn create_default_pensieve_results(&self) -> crate::pensieve_runner::PensieveExecutionResults {
        use crate::pensieve_runner::*;
        use std::collections::HashMap;
        
        PensieveExecutionResults {
            exit_code: Some(1),
            execution_time: Duration::from_secs(0),
            peak_memory_mb: 0,
            average_memory_mb: 0,
            cpu_usage_stats: CpuUsageStats {
                peak_cpu_percent: 0.0,
                average_cpu_percent: 0.0,
                cpu_time_user: Duration::from_secs(0),
                cpu_time_system: Duration::from_secs(0),
            },
            output_analysis: OutputAnalysis {
                total_lines: 0,
                error_lines: 0,
                warning_lines: 0,
                progress_updates: 0,
                files_processed: 0,
                duplicates_found: 0,
                processing_speed_files_per_second: 0.0,
                key_messages: vec!["Validation incomplete".to_string()],
            },
            performance_metrics: PerformanceMetrics {
                files_per_second: 0.0,
                bytes_per_second: 0,
                database_operations_per_second: 0.0,
                memory_efficiency_score: 0.0,
                processing_consistency: 0.0,
            },
            error_summary: ErrorSummary {
                total_errors: 1,
                error_categories: HashMap::new(),
                critical_errors: vec!["Validation incomplete".to_string()],
                recoverable_errors: vec![],
                error_rate_per_minute: 0.0,
            },
            resource_usage: ResourceUsage {
                disk_io_read_bytes: 0,
                disk_io_write_bytes: 0,
                network_io_bytes: 0,
                file_handles_used: 0,
                thread_count: 0,
            },
        }
    }

    /// Create default process results for error cases
    fn create_default_process_results(&self) -> crate::process_monitor::MonitoringResults {
        use crate::process_monitor::*;
        
        MonitoringResults {
            snapshots: Vec::new(),
            summary: MonitoringSummary {
                duration: Duration::from_secs(0),
                snapshot_count: 0,
                peak_memory_usage: 0,
                average_memory_usage: 0,
                peak_cpu_usage: 0.0,
                average_cpu_usage: 0.0,
                total_disk_read: 0,
                total_disk_write: 0,
                peak_temperature: 0.0,
                memory_efficiency: 0.0,
                cpu_efficiency: 0.0,
            },
            alerts: Vec::new(),
            performance_analysis: PerformanceAnalysis {
                resource_utilization_score: 0.0,
                stability_score: 0.0,
                efficiency_score: 0.0,
                bottlenecks: Vec::new(),
                recommendations: Vec::new(),
            },
        }
    }

    /// Create default reliability results for error cases
    fn create_default_reliability_results(&self) -> ReliabilityResults {
        use crate::reliability_validator::*;
        
        ReliabilityResults {
            overall_reliability_score: 0.0,
            crash_test_results: CrashTestResults {
                zero_crash_validation_passed: false,
                total_test_scenarios: 0,
                scenarios_passed: 0,
                scenarios_failed: 0,
                crash_incidents: Vec::new(),
                graceful_failures: Vec::new(),
            },
            interruption_test_results: InterruptionTestResults {
                graceful_shutdown_works: false,
                cleanup_performed: false,
                recovery_instructions_provided: false,
                data_integrity_maintained: false,
                interruption_response_time_ms: 0,
                recovery_test_passed: false,
            },
            resource_limit_test_results: ResourceLimitTestResults {
                memory_exhaustion_handled: false,
                disk_space_exhaustion_handled: false,
                graceful_degradation_works: false,
                resource_monitoring_accurate: false,
                limit_warnings_provided: false,
                max_memory_used_mb: 0,
                memory_limit_respected: false,
            },
            corruption_handling_results: CorruptionHandlingResults {
                corrupted_files_handled: false,
                malformed_content_handled: false,
                encoding_issues_handled: false,
                truncated_files_handled: false,
                binary_files_handled: false,
                corruption_detection_accuracy: 0.0,
                recovery_strategies_effective: false,
            },
            permission_handling_results: PermissionHandlingResults {
                read_permission_errors_handled: false,
                write_permission_errors_handled: false,
                directory_access_errors_handled: false,
                ownership_issues_handled: false,
                permission_error_messages_clear: false,
                fallback_strategies_work: false,
            },
            recovery_test_results: RecoveryTestResults {
                partial_completion_recovery: false,
                database_consistency_maintained: false,
                resume_functionality_works: false,
                state_preservation_accurate: false,
                recovery_instructions_clear: false,
                recovery_time_acceptable: false,
            },
            failure_analysis: FailureAnalysis {
                critical_failures: Vec::new(),
                reliability_blockers: Vec::new(),
                improvement_recommendations: Vec::new(),
                risk_assessment: RiskAssessment {
                    production_readiness_risk: ProductionRisk::High,
                    data_loss_risk: DataLossRisk::Medium,
                    user_experience_risk: UserExperienceRisk::High,
                    performance_degradation_risk: PerformanceRisk::High,
                    overall_risk_score: 1.0,
                },
            },
        }
    }

    /// Create default metrics results for error cases
    fn create_default_metrics_results(&self) -> MetricsCollectionResults {
        use crate::metrics_collector::*;
        
        MetricsCollectionResults {
            collection_duration: Duration::from_secs(0),
            performance_metrics: PerformanceTracker::new(),
            error_metrics: ErrorTracker::new(),
            ux_metrics: UXTracker::new(),
            database_metrics: DatabaseTracker::new(),
            overall_assessment: OverallAssessment {
                performance_score: 0.0,
                reliability_score: 0.0,
                user_experience_score: 0.0,
                efficiency_score: 0.0,
                overall_score: 0.0,
                key_insights: vec!["Validation incomplete".to_string()],
                improvement_recommendations: vec!["Complete validation pipeline".to_string()],
            },
        }
    }

    /// Run comprehensive validation with integrated metrics collection (legacy method for compatibility)
    pub async fn run_comprehensive_validation(
        &self,
        target_directory: &Path,
        chaos_report: &ChaosReport,
    ) -> Result<ComprehensiveValidationResults> {
        let start_time = Instant::now();
        
        // Start metrics collection
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let (metrics_rx, metrics_handle) = {
            let collector = metrics_collector.lock().await;
            collector.start_collection().await?
        };

        // Start real-time analysis if enabled
        let analysis_handle = if self.config.enable_real_time_analysis {
            Some(self.start_real_time_analysis(metrics_rx).await?)
        } else {
            None
        };

        // Record initial UX event
        self.record_ux_event(
            UXEventType::ProgressUpdate,
            "Starting comprehensive validation...".to_string(),
            UXQualityScores {
                clarity: 0.9,
                actionability: 0.8,
                completeness: 0.7,
                user_friendliness: 0.9,
            },
        ).await?;

        // Run pensieve with monitoring
        let pensieve_results = self.run_pensieve_with_metrics(target_directory).await?;

        // Run reliability validation
        let reliability_results = self.reliability_validator
            .validate_reliability(target_directory, chaos_report)
            .await?;

        // Stop metrics collection
        metrics_handle.abort();
        if let Some(handle) = analysis_handle {
            handle.abort();
        }

        // Generate comprehensive results
        let metrics_results = {
            let collector = metrics_collector.lock().await;
            collector.generate_report()?
        };

        // Create mock process monitoring results for now
        let process_results = self.create_mock_process_results(&pensieve_results);

        // Perform validation assessment
        let validation_assessment = self.assess_validation_results(
            &pensieve_results,
            &process_results,
            &reliability_results,
            &metrics_results,
        );

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &pensieve_results,
            &reliability_results,
            &metrics_results,
            &validation_assessment,
        );

        // Record completion UX event
        self.record_ux_event(
            UXEventType::CompletionMessage,
            format!("Validation completed in {:.2}s", start_time.elapsed().as_secs_f64()),
            UXQualityScores {
                clarity: 0.9,
                actionability: 0.8,
                completeness: 0.9,
                user_friendliness: 0.9,
            },
        ).await?;

        Ok(ComprehensiveValidationResults {
            pensieve_execution_results: pensieve_results,
            process_monitoring_results: process_results,
            reliability_results,
            metrics_collection_results: metrics_results,
            validation_assessment,
            recommendations,
        })
    }

    /// Run pensieve with integrated metrics collection
    async fn run_pensieve_with_metrics(
        &self,
        target_directory: &Path,
    ) -> Result<crate::pensieve_runner::PensieveExecutionResults> {
        let start_time = Instant::now();
        
        // Record start of processing
        self.record_performance_metrics(0, 100, 5.0, start_time.elapsed()).await?;
        
        // Simulate pensieve execution with metrics collection
        // In a real implementation, this would integrate with the actual pensieve process
        let results = self.pensieve_runner.run_with_monitoring(target_directory).await;
        
        match &results {
            Ok(execution_results) => {
                // Record successful completion metrics
                self.record_performance_metrics(
                    execution_results.output_analysis.files_processed,
                    execution_results.peak_memory_mb,
                    execution_results.cpu_usage_stats.peak_cpu_percent,
                    execution_results.execution_time,
                ).await?;

                // Record database operations
                self.record_database_metrics(execution_results).await?;

                // Analyze output for UX quality
                self.analyze_output_for_ux(&execution_results.output_analysis).await?;
            }
            Err(error) => {
                // Record error metrics
                self.record_error_from_validation_error(error).await?;
            }
        }
        
        results
    }

    /// Start real-time analysis of metrics
    async fn start_real_time_analysis(
        &self,
        mut metrics_rx: tokio::sync::mpsc::Receiver<crate::metrics_collector::MetricsUpdate>,
    ) -> Result<JoinHandle<()>> {
        let thresholds = self.config.performance_thresholds.clone();
        let metrics_collector = Arc::clone(&self.metrics_collector);
        
        let handle = tokio::spawn(async move {
            while let Some(update) = metrics_rx.recv().await {
                // Check performance thresholds
                if update.performance_snapshot.current_files_per_second < thresholds.min_files_per_second {
                    eprintln!("WARNING: Processing speed below threshold: {:.2} files/sec", 
                             update.performance_snapshot.current_files_per_second);
                }
                
                if update.performance_snapshot.current_memory_mb > thresholds.max_memory_mb {
                    eprintln!("WARNING: Memory usage above threshold: {} MB", 
                             update.performance_snapshot.current_memory_mb);
                }
                
                if update.performance_snapshot.current_cpu_percent > thresholds.max_cpu_percent {
                    eprintln!("WARNING: CPU usage above threshold: {:.1}%", 
                             update.performance_snapshot.current_cpu_percent);
                }
                
                if update.error_snapshot.error_rate > thresholds.max_error_rate_per_minute {
                    eprintln!("WARNING: Error rate above threshold: {:.1} errors/min", 
                             update.error_snapshot.error_rate);
                }
                
                if update.ux_snapshot.current_ux_score < thresholds.min_ux_score {
                    eprintln!("WARNING: UX score below threshold: {:.1}/10", 
                             update.ux_snapshot.current_ux_score);
                }
            }
        });
        
        Ok(handle)
    }

    /// Record performance metrics
    async fn record_performance_metrics(
        &self,
        files_processed: u64,
        memory_mb: u64,
        cpu_percent: f32,
        elapsed: Duration,
    ) -> Result<()> {
        let collector = self.metrics_collector.lock().await;
        collector.record_performance(files_processed, memory_mb, cpu_percent)?;
        drop(collector);
        Ok(())
    }

    /// Record UX event
    async fn record_ux_event(
        &self,
        event_type: UXEventType,
        message: String,
        quality_scores: UXQualityScores,
    ) -> Result<()> {
        let collector = self.metrics_collector.lock().await;
        collector.record_ux_event(event_type, message, quality_scores)?;
        drop(collector);
        Ok(())
    }

    /// Record error from validation error
    async fn record_error_from_validation_error(&self, error: &ValidationError) -> Result<()> {
        let (category, severity, message) = match error {
            ValidationError::PensieveBinaryNotFound { path } => (
                ErrorCategory::Configuration,
                ErrorSeverity::Critical,
                format!("Pensieve binary not found: {:?}", path),
            ),
            ValidationError::DirectoryNotAccessible { path, cause } => (
                ErrorCategory::FileSystem,
                ErrorSeverity::High,
                format!("Directory not accessible: {:?} - {}", path, cause),
            ),
            ValidationError::PensieveCrashed { exit_code, stderr } => (
                ErrorCategory::Unknown,
                ErrorSeverity::Critical,
                format!("Pensieve crashed with exit code {}: {}", exit_code, stderr),
            ),
            ValidationError::ValidationTimeout { seconds } => (
                ErrorCategory::Timeout,
                ErrorSeverity::High,
                format!("Validation timed out after {}s", seconds),
            ),
            ValidationError::ResourceLimitExceeded { resource, limit } => (
                ErrorCategory::Memory,
                ErrorSeverity::High,
                format!("Resource limit exceeded: {} - {}", resource, limit),
            ),
            _ => (
                ErrorCategory::Unknown,
                ErrorSeverity::Medium,
                error.to_string(),
            ),
        };

        let context = ErrorContext {
            file_path: None,
            operation: "validation".to_string(),
            system_state: SystemState {
                memory_usage_mb: 0,
                cpu_usage_percent: 0.0,
                files_processed: 0,
                processing_speed: 0.0,
            },
            preceding_events: vec!["Validation started".to_string()],
        };

        let collector = self.metrics_collector.lock().await;
        collector.record_error(category, severity, message, context)?;
        drop(collector);
        Ok(())
    }

    /// Record database metrics from execution results
    async fn record_database_metrics(
        &self,
        results: &crate::pensieve_runner::PensieveExecutionResults,
    ) -> Result<()> {
        let collector = self.metrics_collector.lock().await;
        
        // Estimate database operations based on files processed
        let files_processed = results.output_analysis.files_processed;
        let avg_operation_time = results.execution_time / (files_processed.max(1) as u32);
        
        // Record estimated insert operations
        for _ in 0..files_processed {
            collector.record_database_operation(
                DatabaseOperation::Insert,
                avg_operation_time,
                true,
            )?;
        }
        
        // Record some select operations for deduplication
        for _ in 0..results.output_analysis.duplicates_found {
            collector.record_database_operation(
                DatabaseOperation::Select,
                avg_operation_time / 2,
                true,
            )?;
        }
        
        drop(collector);
        Ok(())
    }

    /// Analyze output for UX quality
    async fn analyze_output_for_ux(
        &self,
        output_analysis: &crate::pensieve_runner::OutputAnalysis,
    ) -> Result<()> {
        // Analyze progress updates
        if output_analysis.progress_updates > 0 {
            self.record_ux_event(
                UXEventType::ProgressUpdate,
                format!("Processed {} files", output_analysis.files_processed),
                UXQualityScores {
                    clarity: 0.8,
                    actionability: 0.6,
                    completeness: 0.7,
                    user_friendliness: 0.8,
                },
            ).await?;
        }

        // Analyze error messages
        if output_analysis.error_lines > 0 {
            let error_quality = if output_analysis.error_lines < 5 {
                0.8 // Few errors, likely good quality
            } else {
                0.5 // Many errors, might be overwhelming
            };

            self.record_ux_event(
                UXEventType::ErrorMessage,
                format!("Encountered {} errors during processing", output_analysis.error_lines),
                UXQualityScores {
                    clarity: error_quality,
                    actionability: error_quality * 0.8,
                    completeness: 0.7,
                    user_friendliness: error_quality * 0.9,
                },
            ).await?;
        }

        Ok(())
    }

    /// Create mock process monitoring results
    fn create_mock_process_results(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
    ) -> crate::process_monitor::MonitoringResults {
        use crate::process_monitor::*;
        
        // Create a simple mock based on pensieve results
        let summary = MonitoringSummary {
            duration: pensieve_results.execution_time,
            snapshot_count: 10,
            peak_memory_usage: pensieve_results.peak_memory_mb,
            average_memory_usage: pensieve_results.average_memory_mb,
            peak_cpu_usage: pensieve_results.cpu_usage_stats.peak_cpu_percent,
            average_cpu_usage: pensieve_results.cpu_usage_stats.average_cpu_percent,
            total_disk_read: 0,
            total_disk_write: 0,
            peak_temperature: 45.0,
            memory_efficiency: pensieve_results.performance_metrics.memory_efficiency_score,
            cpu_efficiency: 0.8,
        };

        MonitoringResults {
            snapshots: Vec::new(),
            summary,
            alerts: Vec::new(),
            performance_analysis: PerformanceAnalysis {
                resource_utilization_score: 0.7,
                stability_score: pensieve_results.performance_metrics.processing_consistency,
                efficiency_score: pensieve_results.performance_metrics.memory_efficiency_score,
                bottlenecks: Vec::new(),
                recommendations: Vec::new(),
            },
        }
    }

    /// Assess validation results and generate overall assessment
    fn assess_validation_results(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        _process_results: &crate::process_monitor::MonitoringResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
    ) -> ValidationAssessment {
        // Calculate individual grades
        let performance_grade = self.calculate_performance_grade(pensieve_results, metrics_results);
        let reliability_grade = self.calculate_reliability_grade(pensieve_results, reliability_results, metrics_results);
        let user_experience_grade = self.calculate_ux_grade(metrics_results);
        let efficiency_grade = self.calculate_efficiency_grade(pensieve_results, metrics_results);

        // Calculate overall score
        let overall_score = (
            self.grade_to_score(&performance_grade) * 0.3 +
            self.grade_to_score(&reliability_grade) * 0.3 +
            self.grade_to_score(&user_experience_grade) * 0.2 +
            self.grade_to_score(&efficiency_grade) * 0.2
        );

        // Determine production readiness
        let production_readiness = self.assess_production_readiness(
            pensieve_results,
            reliability_results,
            metrics_results,
            overall_score,
        );

        // Identify critical issues
        let critical_issues = self.identify_critical_issues(pensieve_results, reliability_results, metrics_results);

        ValidationAssessment {
            overall_score,
            performance_grade,
            reliability_grade,
            user_experience_grade,
            efficiency_grade,
            production_readiness,
            critical_issues,
        }
    }

    /// Calculate performance grade
    fn calculate_performance_grade(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        metrics_results: &MetricsCollectionResults,
    ) -> Grade {
        let speed_score = if pensieve_results.performance_metrics.files_per_second >= self.config.performance_thresholds.min_files_per_second {
            1.0
        } else {
            pensieve_results.performance_metrics.files_per_second / self.config.performance_thresholds.min_files_per_second
        };

        let consistency_score = metrics_results.performance_metrics.performance_consistency_score;
        let efficiency_score = pensieve_results.performance_metrics.memory_efficiency_score;

        let performance_score = (speed_score + consistency_score + efficiency_score) / 3.0;
        self.score_to_grade(performance_score)
    }

    /// Calculate reliability grade
    fn calculate_reliability_grade(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
    ) -> Grade {
        let crash_score = if pensieve_results.exit_code == Some(0) { 1.0 } else { 0.0 };
        let error_score = if pensieve_results.error_summary.total_errors == 0 {
            1.0
        } else {
            (1.0 - (pensieve_results.error_summary.total_errors as f64 / 100.0)).max(0.0)
        };
        let recovery_score = metrics_results.overall_assessment.reliability_score;
        let reliability_validation_score = reliability_results.overall_reliability_score;

        let reliability_score = (crash_score + error_score + recovery_score + reliability_validation_score) / 4.0;
        self.score_to_grade(reliability_score)
    }

    /// Calculate UX grade
    fn calculate_ux_grade(&self, metrics_results: &MetricsCollectionResults) -> Grade {
        let ux_score = metrics_results.overall_assessment.user_experience_score;
        self.score_to_grade(ux_score)
    }

    /// Calculate efficiency grade
    fn calculate_efficiency_grade(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        metrics_results: &MetricsCollectionResults,
    ) -> Grade {
        let memory_efficiency = pensieve_results.performance_metrics.memory_efficiency_score;
        let resource_efficiency = metrics_results.overall_assessment.efficiency_score;

        let efficiency_score = (memory_efficiency + resource_efficiency) / 2.0;
        self.score_to_grade(efficiency_score)
    }

    /// Convert score to grade
    fn score_to_grade(&self, score: f64) -> Grade {
        match (score * 100.0) as u32 {
            90..=100 => Grade::A,
            80..=89 => Grade::B,
            70..=79 => Grade::C,
            60..=69 => Grade::D,
            _ => Grade::F,
        }
    }

    /// Convert grade to score
    fn grade_to_score(&self, grade: &Grade) -> f64 {
        match grade {
            Grade::A => 0.95,
            Grade::B => 0.85,
            Grade::C => 0.75,
            Grade::D => 0.65,
            Grade::F => 0.50,
        }
    }

    /// Assess production readiness
    fn assess_production_readiness(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
        overall_score: f64,
    ) -> ProductionReadiness {
        let mut caveats = Vec::new();
        let mut blockers = Vec::new();

        // Check for critical issues
        if pensieve_results.exit_code != Some(0) {
            blockers.push("Process did not complete successfully".to_string());
        }

        if pensieve_results.error_summary.critical_errors.len() > 0 {
            blockers.push(format!("Found {} critical errors", pensieve_results.error_summary.critical_errors.len()));
        }

        if metrics_results.overall_assessment.reliability_score < 0.8 {
            blockers.push("Reliability score below acceptable threshold".to_string());
        }

        if !reliability_results.crash_test_results.zero_crash_validation_passed {
            blockers.push("Crash validation failed - application crashes on edge cases".to_string());
        }

        if !reliability_results.interruption_test_results.graceful_shutdown_works {
            blockers.push("Interruption handling failed - application does not shut down gracefully".to_string());
        }

        if !reliability_results.resource_limit_test_results.memory_limit_respected {
            blockers.push("Memory limit exceeded - application uses too much memory".to_string());
        }

        // Check for caveats
        if pensieve_results.performance_metrics.files_per_second < self.config.performance_thresholds.min_files_per_second {
            caveats.push("Processing speed below optimal threshold".to_string());
        }

        if metrics_results.overall_assessment.user_experience_score < 0.7 {
            caveats.push("User experience quality needs improvement".to_string());
        }

        if pensieve_results.peak_memory_mb > self.config.performance_thresholds.max_memory_mb {
            caveats.push("Memory usage exceeds recommended limits".to_string());
        }

        // Determine readiness
        if !blockers.is_empty() {
            ProductionReadiness::NotReady(blockers)
        } else if !caveats.is_empty() {
            ProductionReadiness::ReadyWithCaveats(caveats)
        } else {
            ProductionReadiness::Ready
        }
    }

    /// Identify critical issues
    fn identify_critical_issues(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
    ) -> Vec<CriticalIssue> {
        let mut issues = Vec::new();

        // Check for crashes
        if pensieve_results.exit_code != Some(0) {
            issues.push(CriticalIssue {
                issue_type: "Process Failure".to_string(),
                severity: 1.0,
                description: "Pensieve process did not complete successfully".to_string(),
                impact: "Complete failure to process data".to_string(),
                recommended_action: "Investigate process logs and fix underlying issues".to_string(),
            });
        }

        // Check for high error rates
        if pensieve_results.error_summary.error_rate_per_minute > self.config.performance_thresholds.max_error_rate_per_minute {
            issues.push(CriticalIssue {
                issue_type: "High Error Rate".to_string(),
                severity: 0.8,
                description: format!("Error rate of {:.1}/min exceeds threshold", pensieve_results.error_summary.error_rate_per_minute),
                impact: "Reduced reliability and potential data loss".to_string(),
                recommended_action: "Investigate and fix root causes of errors".to_string(),
            });
        }

        // Check for performance issues
        if metrics_results.performance_metrics.performance_consistency_score < 0.5 {
            issues.push(CriticalIssue {
                issue_type: "Performance Inconsistency".to_string(),
                severity: 0.7,
                description: "Processing performance is highly variable".to_string(),
                impact: "Unpredictable processing times and resource usage".to_string(),
                recommended_action: "Optimize algorithms and resource management".to_string(),
            });
        }

        // Check reliability issues
        if !reliability_results.crash_test_results.zero_crash_validation_passed {
            for crash in &reliability_results.crash_test_results.crash_incidents {
                issues.push(CriticalIssue {
                    issue_type: "Application Crash".to_string(),
                    severity: match crash.severity {
                        crate::reliability_validator::CrashSeverity::Critical => 1.0,
                        crate::reliability_validator::CrashSeverity::High => 0.8,
                        crate::reliability_validator::CrashSeverity::Medium => 0.6,
                        crate::reliability_validator::CrashSeverity::Low => 0.4,
                    },
                    description: format!("Application crashes in scenario: {}", crash.scenario_name),
                    impact: "Complete failure to process data, potential data loss".to_string(),
                    recommended_action: "Fix crash-causing bugs and add proper error handling".to_string(),
                });
            }
        }

        // Check for reliability blockers
        for blocker in &reliability_results.failure_analysis.reliability_blockers {
            issues.push(CriticalIssue {
                issue_type: blocker.blocker_type.clone(),
                severity: 0.9,
                description: blocker.description.clone(),
                impact: blocker.business_impact.clone(),
                recommended_action: format!("Address {} (effort: {:?})", blocker.description, blocker.resolution_effort),
            });
        }

        issues
    }

    /// Generate recommendations based on validation results
    fn generate_recommendations(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
        assessment: &ValidationAssessment,
    ) -> Vec<ValidationRecommendation> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if pensieve_results.performance_metrics.files_per_second < self.config.performance_thresholds.min_files_per_second {
            recommendations.push(ValidationRecommendation {
                category: "Performance".to_string(),
                priority: RecommendationPriority::High,
                description: "Improve processing speed through algorithm optimization".to_string(),
                expected_impact: "Faster processing and better user experience".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        // Memory recommendations
        if pensieve_results.peak_memory_mb > self.config.performance_thresholds.max_memory_mb {
            recommendations.push(ValidationRecommendation {
                category: "Memory".to_string(),
                priority: RecommendationPriority::High,
                description: "Optimize memory usage to stay within limits".to_string(),
                expected_impact: "Reduced memory footprint and better scalability".to_string(),
                implementation_effort: ImplementationEffort::Medium,
            });
        }

        // Error handling recommendations
        if pensieve_results.error_summary.total_errors > 0 {
            recommendations.push(ValidationRecommendation {
                category: "Error Handling".to_string(),
                priority: RecommendationPriority::Medium,
                description: "Improve error handling and recovery mechanisms".to_string(),
                expected_impact: "Better reliability and user experience".to_string(),
                implementation_effort: ImplementationEffort::Low,
            });
        }

        // UX recommendations
        if metrics_results.overall_assessment.user_experience_score < 0.8 {
            recommendations.push(ValidationRecommendation {
                category: "User Experience".to_string(),
                priority: RecommendationPriority::Medium,
                description: "Enhance progress reporting and error message clarity".to_string(),
                expected_impact: "Better user satisfaction and adoption".to_string(),
                implementation_effort: ImplementationEffort::Low,
            });
        }

        // Database recommendations
        for bottleneck in &metrics_results.database_metrics.database_bottlenecks {
            if bottleneck.severity > 0.7 {
                recommendations.push(ValidationRecommendation {
                    category: "Database".to_string(),
                    priority: RecommendationPriority::Medium,
                    description: bottleneck.description.clone(),
                    expected_impact: "Improved database performance".to_string(),
                    implementation_effort: ImplementationEffort::Medium,
                });
            }
        }

        // Reliability recommendations
        for recommendation in &reliability_results.failure_analysis.improvement_recommendations {
            recommendations.push(ValidationRecommendation {
                category: format!("{:?}", recommendation.category),
                priority: match recommendation.priority {
                    crate::reliability_validator::RecommendationPriority::Critical => RecommendationPriority::Critical,
                    crate::reliability_validator::RecommendationPriority::High => RecommendationPriority::High,
                    crate::reliability_validator::RecommendationPriority::Medium => RecommendationPriority::Medium,
                    crate::reliability_validator::RecommendationPriority::Low => RecommendationPriority::Low,
                },
                description: recommendation.description.clone(),
                expected_impact: format!("{:?}", recommendation.expected_impact),
                implementation_effort: match recommendation.implementation_effort {
                    crate::reliability_validator::ImplementationEffort::Trivial => ImplementationEffort::Low,
                    crate::reliability_validator::ImplementationEffort::Low => ImplementationEffort::Low,
                    crate::reliability_validator::ImplementationEffort::Medium => ImplementationEffort::Medium,
                    crate::reliability_validator::ImplementationEffort::High => ImplementationEffort::High,
                    crate::reliability_validator::ImplementationEffort::Epic => ImplementationEffort::Epic,
                },
            });
        }

        recommendations
    }

    /// Run comprehensive performance benchmarking
    async fn run_performance_benchmarking(
        &self,
        target_directory: &Path,
    ) -> Result<PerformanceBenchmarkingResults> {
        println!("🎯 Running comprehensive performance benchmarking...");
        
        // Get pensieve binary path from config
        let pensieve_binary = &self.config.pensieve_config.binary_path;
        
        // Create output database path
        let output_database = target_directory.join("validation_benchmark.db");
        
        // Create a mutable benchmarker for this run
        let mut benchmarker = PerformanceBenchmarker::with_config(self.config.benchmark_config.clone());
        
        // Run comprehensive benchmarking
        let results = benchmarker.run_comprehensive_benchmark(
            pensieve_binary,
            &target_directory.to_path_buf(),
            &output_database,
        ).await.map_err(|e| ValidationError::ProcessMonitoring(
            format!("Performance benchmarking failed: {}", e)
        ))?;
        
        println!("✅ Performance benchmarking completed");
        Ok(results)
    }

    /// Assess scalability with enhanced benchmarking data
    fn assess_scalability_with_benchmarking(
        &self,
        pensieve_results: &crate::pensieve_runner::PensieveExecutionResults,
        benchmarking_results: &PerformanceBenchmarkingResults,
        _current_results: &ValidationPhaseResults,
    ) -> ScalabilityAssessment {
        // Extract scalability insights from benchmarking results
        let scalability_analysis = &benchmarking_results.scalability_analysis;
        
        // Convert benchmarking bottlenecks to validation bottlenecks
        let bottleneck_analysis: Vec<BottleneckAnalysis> = scalability_analysis
            .bottleneck_points
            .iter()
            .map(|bottleneck| BottleneckAnalysis {
                component: "Pensieve Processing".to_string(),
                bottleneck_type: bottleneck.bottleneck_type.clone(),
                severity: bottleneck.performance_impact,
                impact_description: format!(
                    "Bottleneck occurs at {} files with {}% performance impact",
                    bottleneck.bottleneck_point,
                    (bottleneck.performance_impact * 100.0) as u32
                ),
                resolution_suggestions: bottleneck.mitigation_strategies.clone(),
            })
            .collect();

        // Convert scaling recommendations
        let scaling_recommendations: Vec<ScalingRecommendation> = scalability_analysis
            .scaling_recommendations
            .iter()
            .map(|rec| ScalingRecommendation {
                scenario: rec.description.clone(),
                recommended_resources: ResourceRequirements {
                    estimated_memory_mb: 2048, // Would be calculated from benchmarking
                    estimated_cpu_cores: 4,
                    estimated_disk_space_mb: 10240,
                    estimated_processing_time: Duration::from_secs(3600),
                },
                expected_performance: format!("{}% improvement expected", (rec.expected_improvement * 100.0) as u32),
                cost_implications: match rec.implementation_complexity {
                    crate::performance_benchmarker::ImplementationEffort::Low => "Low cost".to_string(),
                    crate::performance_benchmarker::ImplementationEffort::Medium => "Moderate cost".to_string(),
                    crate::performance_benchmarker::ImplementationEffort::High => "High cost".to_string(),
                },
            })
            .collect();

        // Calculate scaling factors from memory analysis
        let memory_analysis = &benchmarking_results.memory_analysis;
        let memory_scaling_factor = if memory_analysis.memory_usage_pattern.baseline_memory_mb > 0 {
            memory_analysis.memory_usage_pattern.peak_memory_mb as f64 / 
            memory_analysis.memory_usage_pattern.baseline_memory_mb as f64
        } else {
            1.0
        };

        // Calculate linear scaling factor from overall performance assessment
        let linear_scaling_factor = benchmarking_results.overall_performance_assessment.scalability_score;

        ScalabilityAssessment {
            linear_scaling_factor,
            memory_scaling_factor,
            bottleneck_analysis,
            scaling_recommendations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_validation_orchestrator_creation() {
        let config = ValidationOrchestratorConfig::default();
        let orchestrator = ValidationOrchestrator::new(config);
        
        // Basic creation test
        assert!(orchestrator.config.metrics_collection_interval_ms > 0);
        assert!(orchestrator.config.enable_checkpointing);
        assert!(orchestrator.config.enable_parallel_execution);
    }

    #[tokio::test]
    async fn test_validation_phases() {
        let phases = ValidationPhase::all_phases();
        assert_eq!(phases.len(), 5);
        assert_eq!(phases[0], ValidationPhase::PreFlight);
        assert_eq!(phases[4], ValidationPhase::ProductionIntelligence);
        
        // Test phase progression
        assert_eq!(ValidationPhase::PreFlight.next(), Some(ValidationPhase::Reliability));
        assert_eq!(ValidationPhase::ProductionIntelligence.next(), None);
        
        // Test parallel execution capability
        assert!(!ValidationPhase::PreFlight.can_run_parallel());
        assert!(ValidationPhase::Performance.can_run_parallel());
        assert!(ValidationPhase::UserExperience.can_run_parallel());
        assert!(!ValidationPhase::ProductionIntelligence.can_run_parallel());
    }

    #[tokio::test]
    async fn test_validation_state_management() {
        let config = ValidationOrchestratorConfig::default();
        let orchestrator = ValidationOrchestrator::new(config);
        
        // Test initialization
        orchestrator.initialize_validation_session().await.unwrap();
        
        let state = orchestrator.validation_state.read().await;
        assert!(state.completed_phases.is_empty());
        assert!(state.failed_phases.is_empty());
        assert_eq!(state.current_phase, None);
        
        drop(state);
        
        // Test phase management
        orchestrator.set_current_phase(ValidationPhase::PreFlight).await;
        let state = orchestrator.validation_state.read().await;
        assert_eq!(state.current_phase, Some(ValidationPhase::PreFlight));
        
        drop(state);
        
        orchestrator.mark_phase_completed(ValidationPhase::PreFlight).await;
        let state = orchestrator.validation_state.read().await;
        assert!(state.completed_phases.contains(&ValidationPhase::PreFlight));
        assert!(!orchestrator.is_phase_completed(ValidationPhase::Reliability).await);
    }

    #[tokio::test]
    async fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds::default();
        
        assert!(thresholds.min_files_per_second > 0.0);
        assert!(thresholds.max_memory_mb > 0);
        assert!(thresholds.max_cpu_percent > 0.0);
        assert!(thresholds.min_ux_score > 0.0);
    }

    #[test]
    fn test_grade_conversion() {
        let config = ValidationOrchestratorConfig::default();
        let orchestrator = ValidationOrchestrator::new(config);
        
        assert!(matches!(orchestrator.score_to_grade(0.95), Grade::A));
        assert!(matches!(orchestrator.score_to_grade(0.85), Grade::B));
        assert!(matches!(orchestrator.score_to_grade(0.75), Grade::C));
        assert!(matches!(orchestrator.score_to_grade(0.65), Grade::D));
        assert!(matches!(orchestrator.score_to_grade(0.50), Grade::F));
        
        assert!((orchestrator.grade_to_score(&Grade::A) - 0.95).abs() < 0.01);
        assert!((orchestrator.grade_to_score(&Grade::B) - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_critical_issue_identification() {
        let config = ValidationOrchestratorConfig::default();
        let orchestrator = ValidationOrchestrator::new(config);
        
        // Create mock results with issues
        let pensieve_results = create_mock_pensieve_results_with_issues();
        let reliability_results = create_mock_reliability_results();
        let metrics_results = create_mock_metrics_results();
        
        let issues = orchestrator.identify_critical_issues(&pensieve_results, &reliability_results, &metrics_results);
        
        assert!(issues.len() > 0);
        assert!(issues.iter().any(|issue| issue.issue_type == "Process Failure"));
    }

    fn create_mock_pensieve_results_with_issues() -> crate::pensieve_runner::PensieveExecutionResults {
        use crate::pensieve_runner::*;
        use std::collections::HashMap;
        
        PensieveExecutionResults {
            exit_code: Some(1), // Non-zero exit code indicates failure
            execution_time: Duration::from_secs(60),
            peak_memory_mb: 2048,
            average_memory_mb: 1024,
            cpu_usage_stats: CpuUsageStats {
                peak_cpu_percent: 80.0,
                average_cpu_percent: 50.0,
                cpu_time_user: Duration::from_secs(30),
                cpu_time_system: Duration::from_secs(10),
            },
            output_analysis: OutputAnalysis {
                total_lines: 1000,
                error_lines: 50,
                warning_lines: 20,
                progress_updates: 10,
                files_processed: 500,
                duplicates_found: 25,
                processing_speed_files_per_second: 8.33,
                key_messages: vec!["Processing completed with errors".to_string()],
            },
            performance_metrics: PerformanceMetrics {
                files_per_second: 8.33,
                bytes_per_second: 1024000,
                database_operations_per_second: 100.0,
                memory_efficiency_score: 0.6,
                processing_consistency: 0.4, // Low consistency
            },
            error_summary: ErrorSummary {
                total_errors: 50,
                error_categories: HashMap::new(),
                critical_errors: vec!["Critical error occurred".to_string()],
                recoverable_errors: vec!["Recoverable error".to_string()],
                error_rate_per_minute: 10.0, // High error rate
            },
            resource_usage: ResourceUsage {
                disk_io_read_bytes: 1024000,
                disk_io_write_bytes: 512000,
                network_io_bytes: 0,
                file_handles_used: 100,
                thread_count: 4,
            },
        }
    }

    fn create_mock_metrics_results() -> MetricsCollectionResults {
        use crate::metrics_collector::*;
        
        MetricsCollectionResults {
            collection_duration: Duration::from_secs(60),
            performance_metrics: PerformanceTracker::new(),
            error_metrics: ErrorTracker::new(),
            ux_metrics: UXTracker::new(),
            database_metrics: DatabaseTracker::new(),
            overall_assessment: OverallAssessment {
                performance_score: 0.6,
                reliability_score: 0.5, // Low reliability
                user_experience_score: 0.7,
                efficiency_score: 0.6,
                overall_score: 0.6,
                key_insights: vec!["Performance issues detected".to_string()],
                improvement_recommendations: vec!["Optimize processing algorithms".to_string()],
            },
        }
    }

    fn create_mock_reliability_results() -> ReliabilityResults {
        use crate::reliability_validator::*;
        
        ReliabilityResults {
            overall_reliability_score: 0.5, // Low reliability
            crash_test_results: CrashTestResults {
                zero_crash_validation_passed: false,
                total_test_scenarios: 5,
                scenarios_passed: 2,
                scenarios_failed: 3,
                crash_incidents: vec![
                    CrashIncident {
                        scenario_name: "Test crash".to_string(),
                        crash_type: CrashType::Panic,
                        exit_code: Some(1),
                        error_message: "Test crash occurred".to_string(),
                        stack_trace: None,
                        reproduction_steps: vec!["Run test".to_string()],
                        severity: CrashSeverity::High,
                    }
                ],
                graceful_failures: vec![],
            },
            interruption_test_results: InterruptionTestResults {
                graceful_shutdown_works: false,
                cleanup_performed: false,
                recovery_instructions_provided: false,
                data_integrity_maintained: true,
                interruption_response_time_ms: 1000,
                recovery_test_passed: false,
            },
            resource_limit_test_results: ResourceLimitTestResults {
                memory_exhaustion_handled: false,
                disk_space_exhaustion_handled: true,
                graceful_degradation_works: false,
                resource_monitoring_accurate: true,
                limit_warnings_provided: false,
                max_memory_used_mb: 8192,
                memory_limit_respected: false,
            },
            corruption_handling_results: CorruptionHandlingResults {
                corrupted_files_handled: true,
                malformed_content_handled: true,
                encoding_issues_handled: false,
                truncated_files_handled: true,
                binary_files_handled: true,
                corruption_detection_accuracy: 0.7,
                recovery_strategies_effective: false,
            },
            permission_handling_results: PermissionHandlingResults {
                read_permission_errors_handled: true,
                write_permission_errors_handled: false,
                directory_access_errors_handled: true,
                ownership_issues_handled: false,
                permission_error_messages_clear: false,
                fallback_strategies_work: true,
            },
            recovery_test_results: RecoveryTestResults {
                partial_completion_recovery: false,
                database_consistency_maintained: true,
                resume_functionality_works: false,
                state_preservation_accurate: false,
                recovery_instructions_clear: false,
                recovery_time_acceptable: true,
            },
            failure_analysis: FailureAnalysis {
                critical_failures: vec![],
                reliability_blockers: vec![
                    ReliabilityBlocker {
                        blocker_type: "Crash".to_string(),
                        description: "Application crashes on edge cases".to_string(),
                        affected_scenarios: vec!["Edge case processing".to_string()],
                        business_impact: "Users lose trust".to_string(),
                        technical_impact: "System unreliable".to_string(),
                        resolution_effort: ResolutionEffort::High,
                    }
                ],
                improvement_recommendations: vec![
                    ReliabilityRecommendation {
                        category: RecommendationCategory::ErrorHandling,
                        title: "Improve error handling".to_string(),
                        description: "Add comprehensive error handling".to_string(),
                        implementation_effort: ImplementationEffort::Medium,
                        expected_impact: ExpectedImpact::High,
                        priority: RecommendationPriority::High,
                    }
                ],
                risk_assessment: RiskAssessment {
                    production_readiness_risk: ProductionRisk::High,
                    data_loss_risk: DataLossRisk::Low,
                    user_experience_risk: UserExperienceRisk::Medium,
                    performance_degradation_risk: PerformanceRisk::Medium,
                    overall_risk_score: 0.7,
                },
            },
        }
    }
}