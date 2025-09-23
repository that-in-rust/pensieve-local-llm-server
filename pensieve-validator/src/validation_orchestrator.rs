use crate::errors::{ValidationError, Result};
use crate::metrics_collector::{
    MetricsCollector, MetricsCollectionResults, ErrorCategory, ErrorSeverity, ErrorContext, 
    SystemState, UXEventType, UXQualityScores, DatabaseOperation
};
use crate::pensieve_runner::{PensieveRunner, PensieveConfig};
use crate::process_monitor::{ProcessMonitor, MonitoringConfig};
use crate::reliability_validator::{ReliabilityValidator, ReliabilityConfig, ReliabilityResults};
use crate::types::ChaosReport;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Orchestrates comprehensive validation with integrated metrics collection
pub struct ValidationOrchestrator {
    pensieve_runner: PensieveRunner,
    process_monitor: ProcessMonitor,
    reliability_validator: ReliabilityValidator,
    metrics_collector: Arc<Mutex<MetricsCollector>>,
    config: ValidationOrchestratorConfig,
}

/// Configuration for the validation orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationOrchestratorConfig {
    pub pensieve_config: PensieveConfig,
    pub monitoring_config: MonitoringConfig,
    pub reliability_config: ReliabilityConfig,
    pub metrics_collection_interval_ms: u64,
    pub enable_real_time_analysis: bool,
    pub performance_thresholds: PerformanceThresholds,
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
            monitoring_config: MonitoringConfig::default(),
            reliability_config: ReliabilityConfig::default(),
            metrics_collection_interval_ms: 500,
            enable_real_time_analysis: true,
            performance_thresholds: PerformanceThresholds::default(),
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
        let process_monitor = ProcessMonitor::with_config(config.monitoring_config.clone());
        let reliability_validator = ReliabilityValidator::new(
            config.reliability_config.clone(),
            config.pensieve_config.clone(),
        );
        let metrics_collector = Arc::new(Mutex::new(MetricsCollector::with_interval(
            Duration::from_millis(config.metrics_collection_interval_ms)
        )));

        Self {
            pensieve_runner,
            process_monitor,
            reliability_validator,
            metrics_collector,
            config,
        }
    }

    /// Run comprehensive validation with integrated metrics collection
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