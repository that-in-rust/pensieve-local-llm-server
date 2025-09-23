use crate::errors::{ValidationError, Result};
use crate::metrics_collector::MetricsCollectionResults;
use crate::pensieve_runner::PensieveExecutionResults;
use crate::reliability_validator::ReliabilityResults;
use crate::types::*;
use crate::ux_analyzer::UXResults;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Production readiness assessment engine with multi-factor evaluation
pub struct ProductionReadinessAssessor {
    config: AssessmentConfig,
}

/// Configuration for production readiness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentConfig {
    pub reliability_weight: f64,
    pub performance_weight: f64,
    pub ux_weight: f64,
    pub minimum_reliability_score: f64,
    pub minimum_performance_score: f64,
    pub minimum_ux_score: f64,
    pub critical_issue_threshold: f64,
    pub blocker_detection_enabled: bool,
    pub scaling_analysis_enabled: bool,
}

impl Default for AssessmentConfig {
    fn default() -> Self {
        Self {
            reliability_weight: 0.4,
            performance_weight: 0.35,
            ux_weight: 0.25,
            minimum_reliability_score: 0.8,
            minimum_performance_score: 0.7,
            minimum_ux_score: 0.7,
            critical_issue_threshold: 0.8,
            blocker_detection_enabled: true,
            scaling_analysis_enabled: true,
        }
    }
}

/// Comprehensive production readiness assessment results
#[derive(Debug, Serialize, Deserialize)]
pub struct ProductionReadinessAssessment {
    pub overall_readiness: ProductionReadinessLevel,
    pub readiness_score: f64, // 0.0 - 1.0
    pub factor_scores: FactorScores,
    pub critical_issues: Vec<CriticalIssue>,
    pub blockers: Vec<ProductionBlocker>,
    pub scaling_guidance: ScalingGuidance,
    pub deployment_recommendations: DeploymentRecommendations,
    pub improvement_roadmap: ImprovementRoadmap,
    pub assessment_metadata: AssessmentMetadata,
}

/// Production readiness levels
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum ProductionReadinessLevel {
    Ready,
    ReadyWithCaveats(Vec<String>),
    NotReady(Vec<String>),
    RequiresImprovement(Vec<String>),
}

/// Factor-based scoring breakdown
#[derive(Debug, Serialize, Deserialize)]
pub struct FactorScores {
    pub reliability_score: f64,
    pub performance_score: f64,
    pub user_experience_score: f64,
    pub consistency_score: f64,
    pub scalability_score: f64,
    pub factor_breakdown: HashMap<String, FactorBreakdown>,
}

/// Detailed breakdown for each factor
#[derive(Debug, Serialize, Deserialize)]
pub struct FactorBreakdown {
    pub score: f64,
    pub weight: f64,
    pub contributing_metrics: Vec<MetricContribution>,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub improvement_potential: f64,
}

/// Individual metric contribution to factor score
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricContribution {
    pub metric_name: String,
    pub value: f64,
    pub weight: f64,
    pub contribution: f64,
    pub threshold: Option<f64>,
    pub status: MetricStatus,
}

/// Status of individual metrics
#[derive(Debug, Serialize, Deserialize)]
pub enum MetricStatus {
    Excellent,
    Good,
    Acceptable,
    NeedsImprovement,
    Critical,
}

/// Critical issues that impact production readiness
#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub issue_id: String,
    pub title: String,
    pub description: String,
    pub severity: IssueSeverity,
    pub impact_areas: Vec<ImpactArea>,
    pub affected_scenarios: Vec<String>,
    pub business_impact: String,
    pub technical_impact: String,
    pub resolution_priority: ResolutionPriority,
    pub estimated_effort: EstimatedEffort,
    pub recommended_actions: Vec<String>,
}

/// Severity levels for issues
#[derive(Debug, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Areas impacted by issues
#[derive(Debug, Serialize, Deserialize)]
pub enum ImpactArea {
    Reliability,
    Performance,
    UserExperience,
    Scalability,
    Security,
    Maintainability,
}

/// Resolution priority levels
#[derive(Debug, Serialize, Deserialize)]
pub enum ResolutionPriority {
    Immediate,
    High,
    Medium,
    Low,
    Deferred,
}

/// Estimated effort for resolution
#[derive(Debug, Serialize, Deserialize)]
pub enum EstimatedEffort {
    Trivial,    // < 1 hour
    Low,        // < 1 day
    Medium,     // 1-3 days
    High,       // 1-2 weeks
    Epic,       // > 2 weeks
}

/// Production blockers that prevent deployment
#[derive(Debug, Serialize, Deserialize)]
pub struct ProductionBlocker {
    pub blocker_id: String,
    pub title: String,
    pub description: String,
    pub blocker_type: BlockerType,
    pub detection_method: String,
    pub evidence: Vec<String>,
    pub must_fix_before_production: bool,
    pub workaround_available: bool,
    pub workaround_description: Option<String>,
    pub resolution_steps: Vec<String>,
}

/// Types of production blockers
#[derive(Debug, Serialize, Deserialize)]
pub enum BlockerType {
    CrashOnEdgeCases,
    DataCorruption,
    MemoryLeak,
    PerformanceDegradation,
    SecurityVulnerability,
    UserExperienceFailure,
    ScalabilityLimit,
    ConfigurationIssue,
}

/// Scaling guidance based on performance patterns
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingGuidance {
    pub current_capacity_assessment: CapacityAssessment,
    pub scaling_recommendations: Vec<ScalingRecommendation>,
    pub resource_requirements: ResourceRequirements,
    pub performance_projections: PerformanceProjections,
    pub bottleneck_analysis: BottleneckAnalysis,
}

/// Current capacity assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct CapacityAssessment {
    pub max_files_per_run: u64,
    pub max_data_size_gb: f64,
    pub max_concurrent_operations: u32,
    pub memory_ceiling_mb: u64,
    pub processing_time_ceiling_hours: f64,
    pub confidence_level: f64,
}

/// Scaling recommendations
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingRecommendation {
    pub scenario: String,
    pub recommended_resources: HashMap<String, String>,
    pub expected_performance: String,
    pub cost_implications: String,
    pub implementation_complexity: String,
}

/// Resource requirements for different scales
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub small_scale: ResourceSpec,    // < 10K files
    pub medium_scale: ResourceSpec,   // 10K - 100K files
    pub large_scale: ResourceSpec,    // 100K - 1M files
    pub enterprise_scale: ResourceSpec, // > 1M files
}

/// Resource specification
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub min_memory_gb: f64,
    pub recommended_memory_gb: f64,
    pub min_cpu_cores: u32,
    pub recommended_cpu_cores: u32,
    pub min_disk_space_gb: f64,
    pub recommended_disk_space_gb: f64,
    pub estimated_processing_time: Duration,
}

/// Performance projections for scaling
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceProjections {
    pub linear_scaling_factors: HashMap<String, f64>,
    pub performance_degradation_points: Vec<DegradationPoint>,
    pub optimal_batch_sizes: HashMap<String, u64>,
    pub resource_utilization_curves: HashMap<String, Vec<UtilizationPoint>>,
}

/// Points where performance degrades
#[derive(Debug, Serialize, Deserialize)]
pub struct DegradationPoint {
    pub threshold: String,
    pub degradation_factor: f64,
    pub cause: String,
    pub mitigation: String,
}

/// Resource utilization data points
#[derive(Debug, Serialize, Deserialize)]
pub struct UtilizationPoint {
    pub scale: f64,
    pub utilization: f64,
    pub efficiency: f64,
}

/// Bottleneck analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub identified_bottlenecks: Vec<Bottleneck>,
    pub bottleneck_severity_ranking: Vec<String>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Individual bottleneck
#[derive(Debug, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub severity: f64,
    pub description: String,
    pub impact_on_scaling: String,
    pub optimization_suggestions: Vec<String>,
}

/// Optimization opportunity
#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub area: String,
    pub potential_improvement: f64,
    pub implementation_effort: EstimatedEffort,
    pub description: String,
}

/// Deployment recommendations
#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentRecommendations {
    pub environment_requirements: EnvironmentRequirements,
    pub configuration_recommendations: Vec<ConfigurationRecommendation>,
    pub monitoring_requirements: MonitoringRequirements,
    pub operational_considerations: Vec<OperationalConsideration>,
    pub rollback_strategy: RollbackStrategy,
}

/// Environment requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct EnvironmentRequirements {
    pub minimum_os_requirements: HashMap<String, String>,
    pub required_dependencies: Vec<Dependency>,
    pub network_requirements: NetworkRequirements,
    pub security_requirements: Vec<SecurityRequirement>,
    pub compliance_considerations: Vec<String>,
}

/// Dependency specification
#[derive(Debug, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version_requirement: String,
    pub purpose: String,
    pub criticality: DependencyCriticality,
}

/// Dependency criticality levels
#[derive(Debug, Serialize, Deserialize)]
pub enum DependencyCriticality {
    Critical,
    Important,
    Optional,
}

/// Network requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkRequirements {
    pub bandwidth_requirements: String,
    pub latency_requirements: String,
    pub connectivity_requirements: Vec<String>,
    pub firewall_considerations: Vec<String>,
}

/// Security requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityRequirement {
    pub requirement_type: String,
    pub description: String,
    pub implementation_guidance: String,
    pub compliance_frameworks: Vec<String>,
}

/// Configuration recommendations
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigurationRecommendation {
    pub parameter: String,
    pub recommended_value: String,
    pub rationale: String,
    pub environment_specific: bool,
    pub tuning_guidance: String,
}

/// Monitoring requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringRequirements {
    pub key_metrics: Vec<KeyMetric>,
    pub alerting_thresholds: HashMap<String, f64>,
    pub dashboard_requirements: Vec<String>,
    pub log_retention_requirements: String,
}

/// Key metric for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct KeyMetric {
    pub metric_name: String,
    pub description: String,
    pub collection_frequency: String,
    pub alert_conditions: Vec<String>,
    pub business_relevance: String,
}

/// Operational considerations
#[derive(Debug, Serialize, Deserialize)]
pub struct OperationalConsideration {
    pub area: String,
    pub consideration: String,
    pub impact: String,
    pub mitigation: String,
}

/// Rollback strategy
#[derive(Debug, Serialize, Deserialize)]
pub struct RollbackStrategy {
    pub rollback_triggers: Vec<String>,
    pub rollback_procedures: Vec<String>,
    pub data_recovery_procedures: Vec<String>,
    pub estimated_rollback_time: Duration,
    pub rollback_testing_requirements: Vec<String>,
}

/// Improvement roadmap
#[derive(Debug, Serialize, Deserialize)]
pub struct ImprovementRoadmap {
    pub immediate_actions: Vec<ImprovementAction>,
    pub short_term_improvements: Vec<ImprovementAction>,
    pub long_term_improvements: Vec<ImprovementAction>,
    pub roadmap_timeline: RoadmapTimeline,
    pub success_metrics: Vec<SuccessMetric>,
}

/// Individual improvement action
#[derive(Debug, Serialize, Deserialize)]
pub struct ImprovementAction {
    pub action_id: String,
    pub title: String,
    pub description: String,
    pub category: ImprovementCategory,
    pub priority: ImprovementPriority,
    pub estimated_effort: EstimatedEffort,
    pub expected_impact: ExpectedImpact,
    pub dependencies: Vec<String>,
    pub success_criteria: Vec<String>,
    pub implementation_notes: Vec<String>,
}

/// Categories of improvements
#[derive(Debug, Serialize, Deserialize)]
pub enum ImprovementCategory {
    Reliability,
    Performance,
    UserExperience,
    Scalability,
    Security,
    Maintainability,
    Documentation,
    Testing,
}

/// Priority levels for improvements
#[derive(Debug, Serialize, Deserialize)]
pub enum ImprovementPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Expected impact levels
#[derive(Debug, Serialize, Deserialize)]
pub enum ExpectedImpact {
    High,
    Medium,
    Low,
}

/// Roadmap timeline
#[derive(Debug, Serialize, Deserialize)]
pub struct RoadmapTimeline {
    pub immediate_phase_duration: Duration,
    pub short_term_phase_duration: Duration,
    pub long_term_phase_duration: Duration,
    pub total_estimated_duration: Duration,
    pub milestone_dates: HashMap<String, String>,
}

/// Success metrics for tracking improvement
#[derive(Debug, Serialize, Deserialize)]
pub struct SuccessMetric {
    pub metric_name: String,
    pub current_value: f64,
    pub target_value: f64,
    pub measurement_method: String,
    pub tracking_frequency: String,
}

/// Assessment metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct AssessmentMetadata {
    pub assessment_timestamp: chrono::DateTime<chrono::Utc>,
    pub assessment_version: String,
    pub data_sources: Vec<String>,
    pub assessment_duration: Duration,
    pub confidence_level: f64,
    pub limitations: Vec<String>,
    pub assumptions: Vec<String>,
}

impl ProductionReadinessAssessor {
    /// Create a new production readiness assessor
    pub fn new(config: AssessmentConfig) -> Self {
        Self { config }
    }

    /// Create assessor with default configuration
    pub fn with_default_config() -> Self {
        Self::new(AssessmentConfig::default())
    }

    /// Perform comprehensive production readiness assessment
    pub fn assess_production_readiness(
        &self,
        pensieve_results: &PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
        ux_results: Option<&UXResults>,
        deduplication_roi: Option<&DeduplicationROI>,
    ) -> Result<ProductionReadinessAssessment> {
        let assessment_start = std::time::Instant::now();

        // Calculate factor scores
        let factor_scores = self.calculate_factor_scores(
            pensieve_results,
            reliability_results,
            metrics_results,
            ux_results,
        )?;

        // Calculate overall readiness score
        let readiness_score = self.calculate_overall_readiness_score(&factor_scores);

        // Identify critical issues
        let critical_issues = self.identify_critical_issues(
            pensieve_results,
            reliability_results,
            metrics_results,
            &factor_scores,
        )?;

        // Detect production blockers
        let blockers = if self.config.blocker_detection_enabled {
            self.detect_production_blockers(
                pensieve_results,
                reliability_results,
                &critical_issues,
            )?
        } else {
            Vec::new()
        };

        // Generate scaling guidance
        let scaling_guidance = if self.config.scaling_analysis_enabled {
            self.generate_scaling_guidance(pensieve_results, metrics_results)?
        } else {
            self.create_minimal_scaling_guidance()
        };

        // Generate deployment recommendations
        let deployment_recommendations = self.generate_deployment_recommendations(
            pensieve_results,
            reliability_results,
            &factor_scores,
        )?;

        // Create improvement roadmap
        let improvement_roadmap = self.create_improvement_roadmap(
            &critical_issues,
            &blockers,
            &factor_scores,
        )?;

        // Determine overall readiness level
        let overall_readiness = self.determine_readiness_level(
            readiness_score,
            &critical_issues,
            &blockers,
        );

        // Create assessment metadata
        let assessment_metadata = AssessmentMetadata {
            assessment_timestamp: chrono::Utc::now(),
            assessment_version: "1.0.0".to_string(),
            data_sources: vec![
                "Pensieve Execution Results".to_string(),
                "Reliability Validation Results".to_string(),
                "Metrics Collection Results".to_string(),
            ],
            assessment_duration: assessment_start.elapsed(),
            confidence_level: self.calculate_confidence_level(&factor_scores),
            limitations: vec![
                "Assessment based on single test run".to_string(),
                "Real production load patterns may differ".to_string(),
            ],
            assumptions: vec![
                "Test environment representative of production".to_string(),
                "Input data representative of real usage".to_string(),
            ],
        };

        Ok(ProductionReadinessAssessment {
            overall_readiness,
            readiness_score,
            factor_scores,
            critical_issues,
            blockers,
            scaling_guidance,
            deployment_recommendations,
            improvement_roadmap,
            assessment_metadata,
        })
    }

    /// Calculate factor-based scores
    fn calculate_factor_scores(
        &self,
        pensieve_results: &PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        metrics_results: &MetricsCollectionResults,
        ux_results: Option<&UXResults>,
    ) -> Result<FactorScores> {
        let mut factor_breakdown = HashMap::new();

        // Calculate reliability score
        let reliability_score = self.calculate_reliability_score(pensieve_results, reliability_results)?;
        factor_breakdown.insert("reliability".to_string(), self.create_reliability_breakdown(
            pensieve_results, reliability_results, reliability_score
        ));

        // Calculate performance score
        let performance_score = self.calculate_performance_score(pensieve_results, metrics_results)?;
        factor_breakdown.insert("performance".to_string(), self.create_performance_breakdown(
            pensieve_results, metrics_results, performance_score
        ));

        // Calculate user experience score
        let user_experience_score = self.calculate_ux_score(metrics_results, ux_results)?;
        factor_breakdown.insert("user_experience".to_string(), self.create_ux_breakdown(
            metrics_results, ux_results, user_experience_score
        ));

        // Calculate consistency score
        let consistency_score = self.calculate_consistency_score(pensieve_results, metrics_results)?;
        factor_breakdown.insert("consistency".to_string(), self.create_consistency_breakdown(
            pensieve_results, metrics_results, consistency_score
        ));

        // Calculate scalability score
        let scalability_score = self.calculate_scalability_score(pensieve_results, metrics_results)?;
        factor_breakdown.insert("scalability".to_string(), self.create_scalability_breakdown(
            pensieve_results, metrics_results, scalability_score
        ));

        Ok(FactorScores {
            reliability_score,
            performance_score,
            user_experience_score,
            consistency_score,
            scalability_score,
            factor_breakdown,
        })
    }

    /// Calculate reliability score based on crash-free operation and error handling
    fn calculate_reliability_score(
        &self,
        pensieve_results: &PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
    ) -> Result<f64> {
        let mut score_components = Vec::new();

        // Crash-free operation (40% weight)
        let crash_free_score = if pensieve_results.exit_code == Some(0) { 1.0 } else { 0.0 };
        score_components.push((crash_free_score, 0.4));

        // Reliability validation score (30% weight)
        score_components.push((reliability_results.overall_reliability_score, 0.3));

        // Error handling quality (20% weight)
        let error_handling_score = self.calculate_error_handling_score(pensieve_results);
        score_components.push((error_handling_score, 0.2));

        // Recovery capability (10% weight)
        let recovery_score = if reliability_results.recovery_test_results.partial_completion_recovery {
            0.8
        } else {
            0.3
        };
        score_components.push((recovery_score, 0.1));

        // Calculate weighted average
        let weighted_sum: f64 = score_components.iter().map(|(score, weight)| score * weight).sum();
        Ok(weighted_sum)
    }

    /// Calculate error handling score
    fn calculate_error_handling_score(&self, pensieve_results: &PensieveExecutionResults) -> f64 {
        let total_errors = pensieve_results.error_summary.total_errors;
        let critical_errors = pensieve_results.error_summary.critical_errors.len();

        if total_errors == 0 {
            1.0
        } else if critical_errors == 0 {
            // Has errors but no critical ones
            (1.0 - (total_errors as f64 / 1000.0)).max(0.5)
        } else {
            // Has critical errors
            (1.0 - (critical_errors as f64 / 10.0)).max(0.0)
        }
    }

    /// Calculate performance score based on speed and consistency
    fn calculate_performance_score(
        &self,
        pensieve_results: &PensieveExecutionResults,
        metrics_results: &MetricsCollectionResults,
    ) -> Result<f64> {
        let mut score_components = Vec::new();

        // Processing speed (40% weight)
        let speed_score = self.calculate_speed_score(pensieve_results);
        score_components.push((speed_score, 0.4));

        // Memory efficiency (25% weight)
        let memory_score = pensieve_results.performance_metrics.memory_efficiency_score;
        score_components.push((memory_score, 0.25));

        // Performance consistency (25% weight)
        let consistency_score = metrics_results.performance_metrics.performance_consistency_score;
        score_components.push((consistency_score, 0.25));

        // Resource utilization (10% weight)
        let resource_score = self.calculate_resource_utilization_score(pensieve_results);
        score_components.push((resource_score, 0.1));

        // Calculate weighted average
        let weighted_sum: f64 = score_components.iter().map(|(score, weight)| score * weight).sum();
        Ok(weighted_sum)
    }

    /// Calculate speed score
    fn calculate_speed_score(&self, pensieve_results: &PensieveExecutionResults) -> f64 {
        let files_per_second = pensieve_results.performance_metrics.files_per_second;
        
        // Score based on files per second thresholds
        if files_per_second >= 50.0 {
            1.0
        } else if files_per_second >= 20.0 {
            0.8
        } else if files_per_second >= 10.0 {
            0.6
        } else if files_per_second >= 5.0 {
            0.4
        } else if files_per_second >= 1.0 {
            0.2
        } else {
            0.0
        }
    }

    /// Calculate resource utilization score
    fn calculate_resource_utilization_score(&self, pensieve_results: &PensieveExecutionResults) -> f64 {
        // Score based on CPU efficiency and memory usage patterns
        let cpu_efficiency = if pensieve_results.cpu_usage_stats.average_cpu_percent > 0.0 {
            (pensieve_results.cpu_usage_stats.average_cpu_percent as f64 / 100.0).min(1.0)
        } else {
            0.5
        };

        let memory_efficiency = pensieve_results.performance_metrics.memory_efficiency_score;

        (cpu_efficiency + memory_efficiency) / 2.0
    }

    /// Calculate user experience score
    fn calculate_ux_score(
        &self,
        metrics_results: &MetricsCollectionResults,
        ux_results: Option<&UXResults>,
    ) -> Result<f64> {
        if let Some(ux) = ux_results {
            // Use detailed UX analysis if available - calculate average of available scores
            let progress_score = (ux.progress_reporting_quality.update_frequency_score +
                                ux.progress_reporting_quality.information_completeness_score +
                                ux.progress_reporting_quality.clarity_score) / 3.0;
            let error_score = (ux.error_message_clarity.average_clarity_score +
                             ux.error_message_clarity.actionability_score) / 2.0;
            let completion_score = (ux.completion_feedback_quality.summary_completeness +
                                  ux.completion_feedback_quality.results_clarity) / 2.0;
            let interruption_score = (ux.interruption_handling_quality.graceful_shutdown_score +
                                    ux.interruption_handling_quality.state_preservation_score) / 2.0;
            
            Ok((progress_score + error_score + completion_score + interruption_score) / 4.0)
        } else {
            // Fall back to metrics-based UX score
            Ok(metrics_results.overall_assessment.user_experience_score)
        }
    }

    /// Calculate consistency score
    fn calculate_consistency_score(
        &self,
        pensieve_results: &PensieveExecutionResults,
        metrics_results: &MetricsCollectionResults,
    ) -> Result<f64> {
        let processing_consistency = pensieve_results.performance_metrics.processing_consistency;
        let metrics_consistency = metrics_results.performance_metrics.performance_consistency_score;
        
        Ok((processing_consistency + metrics_consistency) / 2.0)
    }

    /// Calculate scalability score
    fn calculate_scalability_score(
        &self,
        pensieve_results: &PensieveExecutionResults,
        _metrics_results: &MetricsCollectionResults,
    ) -> Result<f64> {
        // Base scalability assessment on memory efficiency and processing patterns
        let memory_efficiency = pensieve_results.performance_metrics.memory_efficiency_score;
        let processing_consistency = pensieve_results.performance_metrics.processing_consistency;
        
        // Higher consistency and efficiency indicate better scalability
        Ok((memory_efficiency * 0.6) + (processing_consistency * 0.4))
    }

    /// Calculate overall readiness score
    fn calculate_overall_readiness_score(&self, factor_scores: &FactorScores) -> f64 {
        (factor_scores.reliability_score * self.config.reliability_weight) +
        (factor_scores.performance_score * self.config.performance_weight) +
        (factor_scores.user_experience_score * self.config.ux_weight)
    }

    /// Create factor breakdown for reliability
    fn create_reliability_breakdown(
        &self,
        pensieve_results: &PensieveExecutionResults,
        reliability_results: &ReliabilityResults,
        score: f64,
    ) -> FactorBreakdown {
        let mut contributing_metrics = Vec::new();
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        // Crash-free operation metric
        let crash_free = pensieve_results.exit_code == Some(0);
        contributing_metrics.push(MetricContribution {
            metric_name: "Crash-Free Operation".to_string(),
            value: if crash_free { 1.0 } else { 0.0 },
            weight: 0.4,
            contribution: if crash_free { 0.4 } else { 0.0 },
            threshold: Some(1.0),
            status: if crash_free { MetricStatus::Excellent } else { MetricStatus::Critical },
        });

        if crash_free {
            strengths.push("Application completes without crashes".to_string());
        } else {
            weaknesses.push("Application crashes during execution".to_string());
        }

        // Error handling metric
        let error_score = self.calculate_error_handling_score(pensieve_results);
        contributing_metrics.push(MetricContribution {
            metric_name: "Error Handling Quality".to_string(),
            value: error_score,
            weight: 0.2,
            contribution: error_score * 0.2,
            threshold: Some(0.8),
            status: self.score_to_metric_status(error_score),
        });

        if error_score > 0.8 {
            strengths.push("Excellent error handling and recovery".to_string());
        } else if error_score < 0.5 {
            weaknesses.push("Poor error handling, many critical errors".to_string());
        }

        // Recovery capability
        let recovery_works = reliability_results.recovery_test_results.partial_completion_recovery;
        if recovery_works {
            strengths.push("Supports recovery from partial completion".to_string());
        } else {
            weaknesses.push("No recovery mechanism for interrupted operations".to_string());
        }

        FactorBreakdown {
            score,
            weight: self.config.reliability_weight,
            contributing_metrics,
            strengths,
            weaknesses,
            improvement_potential: self.calculate_improvement_potential(score),
        }
    }

    /// Create factor breakdown for performance
    fn create_performance_breakdown(
        &self,
        pensieve_results: &PensieveExecutionResults,
        _metrics_results: &MetricsCollectionResults,
        score: f64,
    ) -> FactorBreakdown {
        let mut contributing_metrics = Vec::new();
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        // Processing speed
        let files_per_second = pensieve_results.performance_metrics.files_per_second;
        let speed_score = self.calculate_speed_score(pensieve_results);
        contributing_metrics.push(MetricContribution {
            metric_name: "Processing Speed".to_string(),
            value: files_per_second,
            weight: 0.4,
            contribution: speed_score * 0.4,
            threshold: Some(10.0),
            status: self.score_to_metric_status(speed_score),
        });

        if files_per_second >= 20.0 {
            strengths.push(format!("High processing speed: {:.1} files/sec", files_per_second));
        } else if files_per_second < 5.0 {
            weaknesses.push(format!("Low processing speed: {:.1} files/sec", files_per_second));
        }

        // Memory efficiency
        let memory_efficiency = pensieve_results.performance_metrics.memory_efficiency_score;
        contributing_metrics.push(MetricContribution {
            metric_name: "Memory Efficiency".to_string(),
            value: memory_efficiency,
            weight: 0.25,
            contribution: memory_efficiency * 0.25,
            threshold: Some(0.7),
            status: self.score_to_metric_status(memory_efficiency),
        });

        if memory_efficiency > 0.8 {
            strengths.push("Excellent memory efficiency".to_string());
        } else if memory_efficiency < 0.5 {
            weaknesses.push("Poor memory efficiency, high memory usage".to_string());
        }

        FactorBreakdown {
            score,
            weight: self.config.performance_weight,
            contributing_metrics,
            strengths,
            weaknesses,
            improvement_potential: self.calculate_improvement_potential(score),
        }
    }

    /// Create factor breakdown for UX
    fn create_ux_breakdown(
        &self,
        metrics_results: &MetricsCollectionResults,
        ux_results: Option<&UXResults>,
        score: f64,
    ) -> FactorBreakdown {
        let mut contributing_metrics = Vec::new();
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        if let Some(ux) = ux_results {
            // Detailed UX metrics - use available fields
            let progress_score = (ux.progress_reporting_quality.update_frequency_score +
                                ux.progress_reporting_quality.information_completeness_score +
                                ux.progress_reporting_quality.clarity_score) / 3.0;
            contributing_metrics.push(MetricContribution {
                metric_name: "Progress Reporting Quality".to_string(),
                value: progress_score,
                weight: 0.3,
                contribution: progress_score * 0.3,
                threshold: Some(0.7),
                status: self.score_to_metric_status(progress_score),
            });

            let error_score = (ux.error_message_clarity.average_clarity_score +
                             ux.error_message_clarity.actionability_score) / 2.0;
            contributing_metrics.push(MetricContribution {
                metric_name: "Error Message Clarity".to_string(),
                value: error_score,
                weight: 0.3,
                contribution: error_score * 0.3,
                threshold: Some(0.7),
                status: self.score_to_metric_status(error_score),
            });

            // Add strengths and weaknesses based on UX analysis
            for improvement in &ux.improvement_recommendations {
                // Use string comparison instead of enum comparison for now
                if improvement.description.contains("high priority") || improvement.description.contains("critical") {
                    weaknesses.push(improvement.description.clone());
                }
            }
        } else {
            // Fall back to basic metrics
            contributing_metrics.push(MetricContribution {
                metric_name: "Overall UX Score".to_string(),
                value: metrics_results.overall_assessment.user_experience_score,
                weight: 1.0,
                contribution: metrics_results.overall_assessment.user_experience_score,
                threshold: Some(0.7),
                status: self.score_to_metric_status(metrics_results.overall_assessment.user_experience_score),
            });
        }

        if score > 0.8 {
            strengths.push("Excellent user experience quality".to_string());
        } else if score < 0.6 {
            weaknesses.push("User experience needs significant improvement".to_string());
        }

        FactorBreakdown {
            score,
            weight: self.config.ux_weight,
            contributing_metrics,
            strengths,
            weaknesses,
            improvement_potential: self.calculate_improvement_potential(score),
        }
    }

    /// Create factor breakdown for consistency
    fn create_consistency_breakdown(
        &self,
        pensieve_results: &PensieveExecutionResults,
        metrics_results: &MetricsCollectionResults,
        score: f64,
    ) -> FactorBreakdown {
        let mut contributing_metrics = Vec::new();
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        let processing_consistency = pensieve_results.performance_metrics.processing_consistency;
        contributing_metrics.push(MetricContribution {
            metric_name: "Processing Consistency".to_string(),
            value: processing_consistency,
            weight: 0.5,
            contribution: processing_consistency * 0.5,
            threshold: Some(0.8),
            status: self.score_to_metric_status(processing_consistency),
        });

        let metrics_consistency = metrics_results.performance_metrics.performance_consistency_score;
        contributing_metrics.push(MetricContribution {
            metric_name: "Metrics Consistency".to_string(),
            value: metrics_consistency,
            weight: 0.5,
            contribution: metrics_consistency * 0.5,
            threshold: Some(0.8),
            status: self.score_to_metric_status(metrics_consistency),
        });

        if score > 0.8 {
            strengths.push("Highly consistent performance".to_string());
        } else if score < 0.6 {
            weaknesses.push("Inconsistent performance, high variability".to_string());
        }

        FactorBreakdown {
            score,
            weight: 0.1, // Consistency is a secondary factor
            contributing_metrics,
            strengths,
            weaknesses,
            improvement_potential: self.calculate_improvement_potential(score),
        }
    }

    /// Create factor breakdown for scalability
    fn create_scalability_breakdown(
        &self,
        pensieve_results: &PensieveExecutionResults,
        _metrics_results: &MetricsCollectionResults,
        score: f64,
    ) -> FactorBreakdown {
        let mut contributing_metrics = Vec::new();
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        let memory_efficiency = pensieve_results.performance_metrics.memory_efficiency_score;
        contributing_metrics.push(MetricContribution {
            metric_name: "Memory Efficiency".to_string(),
            value: memory_efficiency,
            weight: 0.6,
            contribution: memory_efficiency * 0.6,
            threshold: Some(0.7),
            status: self.score_to_metric_status(memory_efficiency),
        });

        let processing_consistency = pensieve_results.performance_metrics.processing_consistency;
        contributing_metrics.push(MetricContribution {
            metric_name: "Processing Consistency".to_string(),
            value: processing_consistency,
            weight: 0.4,
            contribution: processing_consistency * 0.4,
            threshold: Some(0.8),
            status: self.score_to_metric_status(processing_consistency),
        });

        if score > 0.8 {
            strengths.push("Good scalability characteristics".to_string());
        } else if score < 0.6 {
            weaknesses.push("Limited scalability, may not handle large datasets well".to_string());
        }

        FactorBreakdown {
            score,
            weight: 0.1, // Scalability is a secondary factor
            contributing_metrics,
            strengths,
            weaknesses,
            improvement_potential: self.calculate_improvement_potential(score),
        }
    }

    /// Convert score to metric status
    fn score_to_metric_status(&self, score: f64) -> MetricStatus {
        if score >= 0.9 {
            MetricStatus::Excellent
        } else if score >= 0.8 {
            MetricStatus::Good
        } else if score >= 0.7 {
            MetricStatus::Acceptable
        } else if score >= 0.5 {
            MetricStatus::NeedsImprovement
        } else {
            MetricStatus::Critical
        }
    }

    /// Calculate improvement potential
    fn calculate_improvement_potential(&self, current_score: f64) -> f64 {
        (1.0 - current_score).max(0.0)
    }

    /// Calculate confidence level for the assessment
    fn calculate_confidence_level(&self, factor_scores: &FactorScores) -> f64 {
        // Base confidence on the number of data points and score consistency
        let score_variance = self.calculate_score_variance(factor_scores);
        let base_confidence = 0.8;
        
        // Lower confidence if scores are highly variable
        if score_variance > 0.2 {
            base_confidence - 0.2
        } else if score_variance > 0.1 {
            base_confidence - 0.1
        } else {
            base_confidence
        }
    }

    /// Calculate variance in factor scores
    fn calculate_score_variance(&self, factor_scores: &FactorScores) -> f64 {
        let scores = vec![
            factor_scores.reliability_score,
            factor_scores.performance_score,
            factor_scores.user_experience_score,
            factor_scores.consistency_score,
            factor_scores.scalability_score,
        ];

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>() / scores.len() as f64;
        
        variance.sqrt()
    }

    // Placeholder implementations for remaining methods
    fn identify_critical_issues(
        &self,
        _pensieve_results: &PensieveExecutionResults,
        _reliability_results: &ReliabilityResults,
        _metrics_results: &MetricsCollectionResults,
        _factor_scores: &FactorScores,
    ) -> Result<Vec<CriticalIssue>> {
        Ok(Vec::new()) // Simplified implementation
    }

    fn detect_production_blockers(
        &self,
        _pensieve_results: &PensieveExecutionResults,
        _reliability_results: &ReliabilityResults,
        _critical_issues: &[CriticalIssue],
    ) -> Result<Vec<ProductionBlocker>> {
        Ok(Vec::new()) // Simplified implementation
    }

    fn generate_scaling_guidance(
        &self,
        _pensieve_results: &PensieveExecutionResults,
        _metrics_results: &MetricsCollectionResults,
    ) -> Result<ScalingGuidance> {
        Ok(self.create_minimal_scaling_guidance())
    }

    fn generate_deployment_recommendations(
        &self,
        _pensieve_results: &PensieveExecutionResults,
        _reliability_results: &ReliabilityResults,
        _factor_scores: &FactorScores,
    ) -> Result<DeploymentRecommendations> {
        Ok(DeploymentRecommendations {
            environment_requirements: EnvironmentRequirements {
                minimum_os_requirements: HashMap::new(),
                required_dependencies: Vec::new(),
                network_requirements: NetworkRequirements {
                    bandwidth_requirements: "Minimal".to_string(),
                    latency_requirements: "Not applicable".to_string(),
                    connectivity_requirements: Vec::new(),
                    firewall_considerations: Vec::new(),
                },
                security_requirements: Vec::new(),
                compliance_considerations: Vec::new(),
            },
            configuration_recommendations: Vec::new(),
            monitoring_requirements: MonitoringRequirements {
                key_metrics: Vec::new(),
                alerting_thresholds: HashMap::new(),
                dashboard_requirements: Vec::new(),
                log_retention_requirements: "30 days".to_string(),
            },
            operational_considerations: Vec::new(),
            rollback_strategy: RollbackStrategy {
                rollback_triggers: Vec::new(),
                rollback_procedures: Vec::new(),
                data_recovery_procedures: Vec::new(),
                estimated_rollback_time: Duration::from_secs(1800),
                rollback_testing_requirements: Vec::new(),
            },
        })
    }

    fn create_improvement_roadmap(
        &self,
        _critical_issues: &[CriticalIssue],
        _blockers: &[ProductionBlocker],
        _factor_scores: &FactorScores,
    ) -> Result<ImprovementRoadmap> {
        Ok(ImprovementRoadmap {
            immediate_actions: Vec::new(),
            short_term_improvements: Vec::new(),
            long_term_improvements: Vec::new(),
            roadmap_timeline: RoadmapTimeline {
                immediate_phase_duration: Duration::from_secs(7 * 24 * 3600),
                short_term_phase_duration: Duration::from_secs(30 * 24 * 3600),
                long_term_phase_duration: Duration::from_secs(90 * 24 * 3600),
                total_estimated_duration: Duration::from_secs(127 * 24 * 3600),
                milestone_dates: HashMap::new(),
            },
            success_metrics: Vec::new(),
        })
    }

    fn determine_readiness_level(
        &self,
        readiness_score: f64,
        critical_issues: &[CriticalIssue],
        blockers: &[ProductionBlocker],
    ) -> ProductionReadinessLevel {
        // Check for production blockers first
        let production_blockers: Vec<&ProductionBlocker> = blockers
            .iter()
            .filter(|b| b.must_fix_before_production)
            .collect();

        if !production_blockers.is_empty() {
            let blocker_descriptions: Vec<String> = production_blockers
                .iter()
                .map(|b| b.title.clone())
                .collect();
            return ProductionReadinessLevel::NotReady(blocker_descriptions);
        }

        // Check for critical issues
        let critical_issues_count = critical_issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .count();

        if critical_issues_count > 0 {
            let critical_descriptions: Vec<String> = critical_issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Critical)
                .map(|i| i.title.clone())
                .collect();
            return ProductionReadinessLevel::NotReady(critical_descriptions);
        }

        // Check readiness score thresholds
        if readiness_score >= 0.9 {
            ProductionReadinessLevel::Ready
        } else if readiness_score >= 0.8 {
            let mut caveats = Vec::new();
            
            if critical_issues.iter().any(|i| i.severity == IssueSeverity::High) {
                caveats.push("High-priority issues need attention".to_string());
            }
            
            if readiness_score < 0.85 {
                caveats.push("Performance or reliability could be improved".to_string());
            }
            
            ProductionReadinessLevel::ReadyWithCaveats(caveats)
        } else if readiness_score >= 0.7 {
            ProductionReadinessLevel::RequiresImprovement(vec![
                "Significant improvements needed before production deployment".to_string(),
                "Address performance and reliability issues".to_string(),
            ])
        } else {
            ProductionReadinessLevel::NotReady(vec![
                "Major issues prevent production deployment".to_string(),
                "Comprehensive improvements required".to_string(),
            ])
        }
    }

    fn create_minimal_scaling_guidance(&self) -> ScalingGuidance {
        ScalingGuidance {
            current_capacity_assessment: CapacityAssessment {
                max_files_per_run: 10000,
                max_data_size_gb: 10.0,
                max_concurrent_operations: 1,
                memory_ceiling_mb: 4096,
                processing_time_ceiling_hours: 8.0,
                confidence_level: 0.5,
            },
            scaling_recommendations: Vec::new(),
            resource_requirements: ResourceRequirements {
                small_scale: ResourceSpec {
                    min_memory_gb: 2.0,
                    recommended_memory_gb: 4.0,
                    min_cpu_cores: 1,
                    recommended_cpu_cores: 2,
                    min_disk_space_gb: 50.0,
                    recommended_disk_space_gb: 100.0,
                    estimated_processing_time: Duration::from_secs(3600),
                },
                medium_scale: ResourceSpec {
                    min_memory_gb: 4.0,
                    recommended_memory_gb: 8.0,
                    min_cpu_cores: 2,
                    recommended_cpu_cores: 4,
                    min_disk_space_gb: 200.0,
                    recommended_disk_space_gb: 500.0,
                    estimated_processing_time: Duration::from_secs(7200),
                },
                large_scale: ResourceSpec {
                    min_memory_gb: 8.0,
                    recommended_memory_gb: 16.0,
                    min_cpu_cores: 4,
                    recommended_cpu_cores: 8,
                    min_disk_space_gb: 1000.0,
                    recommended_disk_space_gb: 2000.0,
                    estimated_processing_time: Duration::from_secs(14400),
                },
                enterprise_scale: ResourceSpec {
                    min_memory_gb: 16.0,
                    recommended_memory_gb: 32.0,
                    min_cpu_cores: 8,
                    recommended_cpu_cores: 16,
                    min_disk_space_gb: 5000.0,
                    recommended_disk_space_gb: 10000.0,
                    estimated_processing_time: Duration::from_secs(28800),
                },
            },
            performance_projections: PerformanceProjections {
                linear_scaling_factors: HashMap::new(),
                performance_degradation_points: Vec::new(),
                optimal_batch_sizes: HashMap::new(),
                resource_utilization_curves: HashMap::new(),
            },
            bottleneck_analysis: BottleneckAnalysis {
                identified_bottlenecks: Vec::new(),
                bottleneck_severity_ranking: Vec::new(),
                optimization_opportunities: Vec::new(),
            },
        }
    }
}
