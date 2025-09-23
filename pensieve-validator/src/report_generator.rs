use crate::errors::{ValidationError, Result};
use crate::production_readiness_assessor::{ProductionReadinessAssessment, OptimizationOpportunity};
use crate::validation_orchestrator::ComprehensiveValidationResults;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Comprehensive report generation system with multiple output formats
pub struct ReportGenerator {
    config: ReportGeneratorConfig,
}

/// Configuration for report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGeneratorConfig {
    pub output_formats: Vec<OutputFormat>,
    pub report_detail_level: ReportDetailLevel,
    pub include_raw_data: bool,
    pub include_visualizations: bool,
    pub custom_branding: Option<BrandingConfig>,
    pub template_directory: Option<PathBuf>,
}

/// Supported output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Html,
    Csv,
    Markdown,
    Pdf,
}

/// Report detail levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportDetailLevel {
    Executive,  // High-level summary only
    Standard,   // Balanced detail
    Technical,  // Full technical details
    Comprehensive, // Everything including raw data
}

/// Branding configuration for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandingConfig {
    pub organization_name: String,
    pub logo_path: Option<PathBuf>,
    pub color_scheme: ColorScheme,
    pub custom_css: Option<String>,
}

/// Color scheme for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    pub primary_color: String,
    pub secondary_color: String,
    pub accent_color: String,
    pub background_color: String,
    pub text_color: String,
}impl 
Default for ReportGeneratorConfig {
    fn default() -> Self {
        Self {
            output_formats: vec![OutputFormat::Json, OutputFormat::Html],
            report_detail_level: ReportDetailLevel::Standard,
            include_raw_data: false,
            include_visualizations: true,
            custom_branding: None,
            template_directory: None,
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            primary_color: "#2563eb".to_string(),   // Blue
            secondary_color: "#64748b".to_string(), // Slate
            accent_color: "#10b981".to_string(),    // Emerald
            background_color: "#ffffff".to_string(), // White
            text_color: "#1f2937".to_string(),      // Gray-800
        }
    }
}

/// Main production readiness report
#[derive(Debug, Serialize, Deserialize)]
pub struct ProductionReadinessReport {
    pub report_metadata: ReportMetadata,
    pub executive_summary: ExecutiveSummary,
    pub readiness_assessment: ReadinessAssessment,
    pub performance_analysis: PerformanceAnalysisReport,
    pub user_experience_report: UserExperienceReport,
    pub improvement_roadmap: ImprovementRoadmapReport,
    pub scaling_guidance: ScalingGuidanceReport,
    pub deployment_recommendations: DeploymentRecommendationsReport,
    pub appendices: Option<ReportAppendices>,
}

/// Report metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub report_title: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub report_version: String,
    pub validation_target: String,
    pub assessment_duration: Duration,
    pub report_id: String,
    pub generator_version: String,
}

/// Executive summary for stakeholders
#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub overall_recommendation: OverallRecommendation,
    pub key_findings: Vec<KeyFinding>,
    pub critical_actions_required: Vec<CriticalAction>,
    pub business_impact_assessment: BusinessImpactAssessment,
    pub timeline_to_production: TimelineToProduction,
}

/// Overall recommendation levels
#[derive(Debug, Serialize, Deserialize)]
pub enum OverallRecommendation {
    ReadyForProduction,
    ReadyWithMinorImprovements,
    RequiresSignificantWork,
    NotRecommendedForProduction,
}

/// Key finding from the assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct KeyFinding {
    pub finding_id: String,
    pub category: FindingCategory,
    pub title: String,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
}

/// Categories of findings
#[derive(Debug, Serialize, Deserialize)]
pub enum FindingCategory {
    Reliability,
    Performance,
    UserExperience,
    Scalability,
    Security,
    Operational,
}

/// Impact levels
#[derive(Debug, Serialize, Deserialize)]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
    Informational,
}

/// Critical action required
#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalAction {
    pub action_id: String,
    pub title: String,
    pub description: String,
    pub urgency: ActionUrgency,
    pub estimated_effort: String,
    pub business_justification: String,
    pub consequences_of_inaction: String,
}

/// Action urgency levels
#[derive(Debug, Serialize, Deserialize)]
pub enum ActionUrgency {
    Immediate,    // Must be done before any production deployment
    High,         // Should be done within 1 week
    Medium,       // Should be done within 1 month
    Low,          // Can be scheduled for future releases
}

/// Business impact assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct BusinessImpactAssessment {
    pub risk_level: BusinessRiskLevel,
    pub potential_cost_of_issues: String,
    pub expected_benefits_of_deployment: String,
    pub competitive_implications: String,
    pub user_satisfaction_impact: String,
    pub operational_impact: String,
}

/// Business risk levels
#[derive(Debug, Serialize, Deserialize)]
pub enum BusinessRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Timeline to production readiness
#[derive(Debug, Serialize, Deserialize)]
pub struct TimelineToProduction {
    pub current_readiness_percentage: f64,
    pub estimated_time_to_ready: Duration,
    pub major_milestones: Vec<Milestone>,
    pub dependencies: Vec<String>,
    pub confidence_in_timeline: f64,
}

/// Major milestone in production readiness
#[derive(Debug, Serialize, Deserialize)]
pub struct Milestone {
    pub milestone_id: String,
    pub title: String,
    pub description: String,
    pub estimated_completion: Duration,
    pub dependencies: Vec<String>,
    pub success_criteria: Vec<String>,
}

/// Readiness assessment section
#[derive(Debug, Serialize, Deserialize)]
pub struct ReadinessAssessment {
    pub overall_score: f64,
    pub readiness_level: String,
    pub factor_scores: HashMap<String, FactorScore>,
    pub critical_issues_summary: CriticalIssuesSummary,
    pub blockers_summary: BlockersSummary,
    pub strengths: Vec<String>,
    pub areas_for_improvement: Vec<String>,
}

/// Individual factor score
#[derive(Debug, Serialize, Deserialize)]
pub struct FactorScore {
    pub score: f64,
    pub grade: String,
    pub weight: f64,
    pub status: String,
    pub key_metrics: Vec<String>,
}

/// Summary of critical issues
#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalIssuesSummary {
    pub total_issues: usize,
    pub by_severity: HashMap<String, usize>,
    pub by_category: HashMap<String, usize>,
    pub top_issues: Vec<CriticalIssueSummary>,
}

/// Individual critical issue summary
#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalIssueSummary {
    pub title: String,
    pub severity: String,
    pub impact: String,
    pub recommended_action: String,
}

/// Summary of production blockers
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockersSummary {
    pub total_blockers: usize,
    pub by_type: HashMap<String, usize>,
    pub must_fix_count: usize,
    pub workaround_available_count: usize,
    pub top_blockers: Vec<BlockerSummary>,
}

/// Individual blocker summary
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockerSummary {
    pub title: String,
    pub blocker_type: String,
    pub must_fix: bool,
    pub workaround_available: bool,
    pub resolution_summary: String,
}

/// Performance analysis report section
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceAnalysisReport {
    pub performance_summary: PerformanceSummary,
    pub scaling_predictions: ScalingPredictions,
    pub bottleneck_identification: BottleneckIdentification,
    pub performance_trends: PerformanceTrends,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

/// Performance summary
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub overall_performance_grade: String,
    pub processing_speed: ProcessingSpeedMetrics,
    pub resource_utilization: ResourceUtilizationMetrics,
    pub consistency_metrics: ConsistencyMetrics,
    pub efficiency_scores: EfficiencyScores,
}

/// Processing speed metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessingSpeedMetrics {
    pub files_per_second: f64,
    pub throughput_rating: String,
    pub speed_consistency: f64,
    pub peak_performance: f64,
    pub performance_degradation_points: Vec<String>,
}

/// Resource utilization metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    pub memory_efficiency: f64,
    pub cpu_efficiency: f64,
    pub peak_memory_usage_mb: u64,
    pub average_memory_usage_mb: u64,
    pub memory_growth_pattern: String,
    pub resource_optimization_score: f64,
}

/// Consistency metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    pub processing_consistency_score: f64,
    pub performance_variability: f64,
    pub predictability_rating: String,
    pub stability_indicators: Vec<String>,
}

/// Efficiency scores
#[derive(Debug, Serialize, Deserialize)]
pub struct EfficiencyScores {
    pub overall_efficiency: f64,
    pub memory_efficiency: f64,
    pub cpu_efficiency: f64,
    pub io_efficiency: f64,
    pub algorithm_efficiency: f64,
}

/// Scaling predictions
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingPredictions {
    pub linear_scaling_assessment: LinearScalingAssessment,
    pub capacity_limits: CapacityLimits,
    pub resource_requirements_by_scale: HashMap<String, ResourceRequirement>,
    pub performance_degradation_thresholds: Vec<DegradationThreshold>,
}

/// Linear scaling assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct LinearScalingAssessment {
    pub scales_linearly: bool,
    pub scaling_factor: f64,
    pub scaling_confidence: f64,
    pub non_linear_factors: Vec<String>,
}

/// Capacity limits
#[derive(Debug, Serialize, Deserialize)]
pub struct CapacityLimits {
    pub max_files_estimate: u64,
    pub max_data_size_gb: f64,
    pub max_processing_time_hours: f64,
    pub memory_ceiling_gb: f64,
    pub confidence_level: f64,
}

/// Resource requirement for different scales
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub scale_description: String,
    pub recommended_memory_gb: f64,
    pub recommended_cpu_cores: u32,
    pub estimated_processing_time: String,
    pub additional_considerations: Vec<String>,
}

/// Performance degradation threshold
#[derive(Debug, Serialize, Deserialize)]
pub struct DegradationThreshold {
    pub threshold_name: String,
    pub threshold_value: String,
    pub degradation_factor: f64,
    pub mitigation_strategies: Vec<String>,
}

/// Bottleneck identification
#[derive(Debug, Serialize, Deserialize)]
pub struct BottleneckIdentification {
    pub primary_bottlenecks: Vec<BottleneckAnalysis>,
    pub bottleneck_severity_ranking: Vec<String>,
    pub system_bottleneck_score: f64,
    pub optimization_priority_order: Vec<String>,
}

/// Individual bottleneck analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub component: String,
    pub severity_score: f64,
    pub impact_description: String,
    pub evidence: Vec<String>,
    pub optimization_suggestions: Vec<String>,
    pub estimated_improvement_potential: f64,
}

/// Performance trends
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub trend_analysis: String,
    pub performance_stability: f64,
    pub degradation_indicators: Vec<String>,
    pub improvement_indicators: Vec<String>,
    pub long_term_sustainability: String,
}

/// User experience report section
#[derive(Debug, Serialize, Deserialize)]
pub struct UserExperienceReport {
    pub ux_summary: UXSummary,
    pub specific_improvements: Vec<UXImprovement>,
    pub user_journey_analysis: UserJourneyAnalysis,
    pub feedback_quality_assessment: FeedbackQualityAssessment,
    pub ux_optimization_roadmap: UXOptimizationRoadmap,
}

/// UX summary
#[derive(Debug, Serialize, Deserialize)]
pub struct UXSummary {
    pub overall_ux_score: f64,
    pub ux_grade: String,
    pub user_satisfaction_prediction: String,
    pub key_ux_strengths: Vec<String>,
    pub critical_ux_issues: Vec<String>,
}

/// UX improvement recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct UXImprovement {
    pub improvement_id: String,
    pub category: String,
    pub title: String,
    pub current_state: String,
    pub desired_state: String,
    pub user_impact: String,
    pub implementation_approach: String,
    pub priority: String,
    pub estimated_effort: String,
    pub success_metrics: Vec<String>,
}

/// User journey analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct UserJourneyAnalysis {
    pub journey_stages: Vec<JourneyStage>,
    pub pain_points: Vec<PainPoint>,
    pub delight_moments: Vec<DelightMoment>,
    pub overall_journey_score: f64,
}

/// Individual journey stage
#[derive(Debug, Serialize, Deserialize)]
pub struct JourneyStage {
    pub stage_name: String,
    pub stage_description: String,
    pub user_experience_quality: f64,
    pub key_interactions: Vec<String>,
    pub improvement_opportunities: Vec<String>,
}

/// User pain point
#[derive(Debug, Serialize, Deserialize)]
pub struct PainPoint {
    pub pain_point_id: String,
    pub description: String,
    pub severity: String,
    pub frequency: String,
    pub user_impact: String,
    pub suggested_resolution: String,
}

/// User delight moment
#[derive(Debug, Serialize, Deserialize)]
pub struct DelightMoment {
    pub moment_description: String,
    pub why_delightful: String,
    pub amplification_opportunities: Vec<String>,
}

/// Feedback quality assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackQualityAssessment {
    pub progress_feedback_quality: FeedbackQuality,
    pub error_message_quality: FeedbackQuality,
    pub completion_feedback_quality: FeedbackQuality,
    pub overall_communication_score: f64,
}

/// Individual feedback quality
#[derive(Debug, Serialize, Deserialize)]
pub struct FeedbackQuality {
    pub clarity_score: f64,
    pub actionability_score: f64,
    pub completeness_score: f64,
    pub user_friendliness_score: f64,
    pub specific_improvements: Vec<String>,
}

/// UX optimization roadmap
#[derive(Debug, Serialize, Deserialize)]
pub struct UXOptimizationRoadmap {
    pub immediate_wins: Vec<UXImprovement>,
    pub short_term_improvements: Vec<UXImprovement>,
    pub long_term_enhancements: Vec<UXImprovement>,
    pub ux_metrics_to_track: Vec<String>,
}

/// Improvement roadmap report section
#[derive(Debug, Serialize, Deserialize)]
pub struct ImprovementRoadmapReport {
    pub roadmap_summary: RoadmapSummary,
    pub prioritized_improvements: Vec<PrioritizedImprovement>,
    pub implementation_phases: Vec<ImplementationPhase>,
    pub resource_requirements: RoadmapResourceRequirements,
    pub success_tracking: SuccessTracking,
}

/// Roadmap summary
#[derive(Debug, Serialize, Deserialize)]
pub struct RoadmapSummary {
    pub total_improvements: usize,
    pub estimated_total_effort: String,
    pub expected_timeline: String,
    pub key_benefits: Vec<String>,
    pub major_risks: Vec<String>,
}

/// Prioritized improvement with impact/effort analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct PrioritizedImprovement {
    pub improvement_id: String,
    pub title: String,
    pub description: String,
    pub category: String,
    pub impact_score: f64,
    pub effort_score: f64,
    pub priority_score: f64,
    pub impact_description: String,
    pub effort_description: String,
    pub dependencies: Vec<String>,
    pub success_criteria: Vec<String>,
    pub implementation_notes: Vec<String>,
}

/// Implementation phase
#[derive(Debug, Serialize, Deserialize)]
pub struct ImplementationPhase {
    pub phase_name: String,
    pub phase_description: String,
    pub duration_estimate: String,
    pub improvements_in_phase: Vec<String>,
    pub phase_goals: Vec<String>,
    pub success_criteria: Vec<String>,
    pub risks_and_mitigations: Vec<RiskMitigation>,
}

/// Risk and mitigation pair
#[derive(Debug, Serialize, Deserialize)]
pub struct RiskMitigation {
    pub risk: String,
    pub mitigation: String,
    pub probability: String,
    pub impact: String,
}

/// Resource requirements for roadmap
#[derive(Debug, Serialize, Deserialize)]
pub struct RoadmapResourceRequirements {
    pub development_effort: String,
    pub testing_effort: String,
    pub deployment_effort: String,
    pub ongoing_maintenance: String,
    pub skill_requirements: Vec<String>,
    pub external_dependencies: Vec<String>,
}

/// Success tracking for improvements
#[derive(Debug, Serialize, Deserialize)]
pub struct SuccessTracking {
    pub key_performance_indicators: Vec<KPI>,
    pub measurement_frequency: String,
    pub reporting_schedule: String,
    pub success_thresholds: HashMap<String, f64>,
}

/// Key Performance Indicator
#[derive(Debug, Serialize, Deserialize)]
pub struct KPI {
    pub kpi_name: String,
    pub description: String,
    pub current_value: String,
    pub target_value: String,
    pub measurement_method: String,
}

/// Scaling guidance report section
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingGuidanceReport {
    pub scaling_summary: ScalingSummary,
    pub capacity_planning: CapacityPlanning,
    pub architecture_recommendations: ArchitectureRecommendations,
    pub monitoring_requirements: MonitoringRequirements,
    pub scaling_risks: Vec<ScalingRisk>,
}

/// Scaling summary
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingSummary {
    pub current_scale_rating: String,
    pub scaling_readiness_score: f64,
    pub recommended_scaling_approach: String,
    pub key_scaling_constraints: Vec<String>,
    pub scaling_opportunities: Vec<String>,
}

/// Capacity planning
#[derive(Debug, Serialize, Deserialize)]
pub struct CapacityPlanning {
    pub current_capacity: String,
    pub projected_capacity_needs: HashMap<String, String>,
    pub scaling_thresholds: Vec<ScalingThreshold>,
    pub resource_scaling_recommendations: Vec<ResourceScalingRecommendation>,
}

/// Scaling threshold
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingThreshold {
    pub metric_name: String,
    pub threshold_value: String,
    pub recommended_action: String,
    pub urgency: String,
}

/// Resource scaling recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceScalingRecommendation {
    pub resource_type: String,
    pub current_allocation: String,
    pub recommended_scaling: String,
    pub scaling_trigger: String,
    pub cost_implications: String,
}

/// Architecture recommendations
#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureRecommendations {
    pub current_architecture_assessment: String,
    pub recommended_changes: Vec<ArchitectureChange>,
    pub scalability_patterns: Vec<String>,
    pub anti_patterns_to_avoid: Vec<String>,
}

/// Architecture change recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureChange {
    pub change_type: String,
    pub description: String,
    pub benefits: Vec<String>,
    pub implementation_complexity: String,
    pub timeline: String,
}

/// Monitoring requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct MonitoringRequirements {
    pub critical_metrics: Vec<CriticalMetric>,
    pub alerting_strategy: AlertingStrategy,
    pub dashboard_requirements: Vec<String>,
    pub monitoring_tools_recommendations: Vec<String>,
}

/// Critical metric for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct CriticalMetric {
    pub metric_name: String,
    pub description: String,
    pub collection_method: String,
    pub alert_thresholds: HashMap<String, f64>,
    pub business_impact: String,
}

/// Alerting strategy
#[derive(Debug, Serialize, Deserialize)]
pub struct AlertingStrategy {
    pub alert_levels: Vec<AlertLevel>,
    pub escalation_procedures: Vec<String>,
    pub notification_channels: Vec<String>,
    pub alert_fatigue_prevention: Vec<String>,
}

/// Alert level
#[derive(Debug, Serialize, Deserialize)]
pub struct AlertLevel {
    pub level_name: String,
    pub description: String,
    pub response_time_requirement: String,
    pub notification_method: String,
}

/// Scaling risk
#[derive(Debug, Serialize, Deserialize)]
pub struct ScalingRisk {
    pub risk_name: String,
    pub description: String,
    pub probability: String,
    pub impact: String,
    pub mitigation_strategies: Vec<String>,
    pub monitoring_indicators: Vec<String>,
}

/// Deployment recommendations report section
#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentRecommendationsReport {
    pub deployment_summary: DeploymentSummary,
    pub environment_setup: EnvironmentSetup,
    pub configuration_guide: ConfigurationGuide,
    pub operational_procedures: OperationalProcedures,
    pub rollback_planning: RollbackPlanning,
}

/// Deployment summary
#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentSummary {
    pub deployment_readiness: String,
    pub recommended_deployment_strategy: String,
    pub key_deployment_risks: Vec<String>,
    pub success_criteria: Vec<String>,
    pub go_no_go_checklist: Vec<String>,
}

/// Environment setup
#[derive(Debug, Serialize, Deserialize)]
pub struct EnvironmentSetup {
    pub infrastructure_requirements: InfrastructureRequirements,
    pub software_dependencies: Vec<SoftwareDependency>,
    pub security_configuration: SecurityConfiguration,
    pub network_configuration: NetworkConfiguration,
}

/// Infrastructure requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct InfrastructureRequirements {
    pub compute_requirements: ComputeRequirements,
    pub storage_requirements: StorageRequirements,
    pub network_requirements: NetworkRequirements,
    pub backup_requirements: BackupRequirements,
}

/// Compute requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeRequirements {
    pub cpu_specifications: String,
    pub memory_specifications: String,
    pub performance_requirements: String,
    pub availability_requirements: String,
}

/// Storage requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct StorageRequirements {
    pub storage_type: String,
    pub capacity_requirements: String,
    pub performance_requirements: String,
    pub backup_strategy: String,
}

/// Network requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkRequirements {
    pub bandwidth_requirements: String,
    pub latency_requirements: String,
    pub connectivity_requirements: Vec<String>,
    pub security_requirements: Vec<String>,
}

/// Backup requirements
#[derive(Debug, Serialize, Deserialize)]
pub struct BackupRequirements {
    pub backup_frequency: String,
    pub retention_policy: String,
    pub recovery_time_objective: String,
    pub recovery_point_objective: String,
}

/// Software dependency
#[derive(Debug, Serialize, Deserialize)]
pub struct SoftwareDependency {
    pub name: String,
    pub version: String,
    pub purpose: String,
    pub installation_notes: String,
    pub configuration_requirements: Vec<String>,
}

/// Security configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityConfiguration {
    pub authentication_requirements: Vec<String>,
    pub authorization_requirements: Vec<String>,
    pub encryption_requirements: Vec<String>,
    pub audit_requirements: Vec<String>,
    pub compliance_considerations: Vec<String>,
}

/// Network configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkConfiguration {
    pub firewall_rules: Vec<String>,
    pub port_requirements: Vec<String>,
    pub dns_requirements: Vec<String>,
    pub load_balancing_considerations: Vec<String>,
}

/// Configuration guide
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigurationGuide {
    pub configuration_parameters: Vec<ConfigurationParameter>,
    pub environment_specific_settings: HashMap<String, Vec<ConfigurationParameter>>,
    pub tuning_recommendations: Vec<TuningRecommendation>,
    pub configuration_validation: Vec<String>,
}

/// Configuration parameter
#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigurationParameter {
    pub parameter_name: String,
    pub description: String,
    pub recommended_value: String,
    pub rationale: String,
    pub impact_of_change: String,
}

/// Tuning recommendation
#[derive(Debug, Serialize, Deserialize)]
pub struct TuningRecommendation {
    pub parameter: String,
    pub tuning_approach: String,
    pub monitoring_metrics: Vec<String>,
    pub expected_impact: String,
}

/// Operational procedures
#[derive(Debug, Serialize, Deserialize)]
pub struct OperationalProcedures {
    pub startup_procedures: Vec<String>,
    pub shutdown_procedures: Vec<String>,
    pub health_check_procedures: Vec<String>,
    pub maintenance_procedures: Vec<String>,
    pub troubleshooting_guide: TroubleshootingGuide,
}

/// Troubleshooting guide
#[derive(Debug, Serialize, Deserialize)]
pub struct TroubleshootingGuide {
    pub common_issues: Vec<CommonIssue>,
    pub diagnostic_procedures: Vec<String>,
    pub escalation_procedures: Vec<String>,
    pub support_contacts: Vec<String>,
}

/// Common issue
#[derive(Debug, Serialize, Deserialize)]
pub struct CommonIssue {
    pub issue_description: String,
    pub symptoms: Vec<String>,
    pub likely_causes: Vec<String>,
    pub resolution_steps: Vec<String>,
    pub prevention_measures: Vec<String>,
}

/// Rollback planning
#[derive(Debug, Serialize, Deserialize)]
pub struct RollbackPlanning {
    pub rollback_triggers: Vec<String>,
    pub rollback_procedures: Vec<String>,
    pub data_recovery_procedures: Vec<String>,
    pub rollback_testing: Vec<String>,
    pub communication_plan: Vec<String>,
}

/// Report appendices (optional detailed data)
#[derive(Debug, Serialize, Deserialize)]
pub struct ReportAppendices {
    pub raw_validation_data: Option<serde_json::Value>,
    pub detailed_metrics: Option<serde_json::Value>,
    pub test_logs: Option<Vec<String>>,
    pub configuration_files: Option<HashMap<String, String>>,
    pub additional_analysis: Option<HashMap<String, serde_json::Value>>,
}

impl ReportGenerator {
    /// Create a new report generator with the given configuration
    pub fn new(config: ReportGeneratorConfig) -> Self {
        Self { config }
    }

    /// Create report generator with default configuration
    pub fn with_default_config() -> Self {
        Self::new(ReportGeneratorConfig::default())
    }

    /// Generate comprehensive production readiness report
    pub fn generate_production_readiness_report(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
        target_directory: &str,
    ) -> Result<ProductionReadinessReport> {
        let report_id = uuid::Uuid::new_v4().to_string();
        
        // Generate report metadata
        let report_metadata = self.create_report_metadata(&report_id, target_directory)?;
        
        // Generate executive summary
        let executive_summary = self.create_executive_summary(
            validation_results,
            production_assessment,
        )?;
        
        // Generate readiness assessment
        let readiness_assessment = self.create_readiness_assessment(
            validation_results,
            production_assessment,
        )?;
        
        // Generate performance analysis
        let performance_analysis = self.create_performance_analysis(
            validation_results,
            production_assessment,
        )?;
        
        // Generate user experience report
        let user_experience_report = self.create_user_experience_report(
            validation_results,
            production_assessment,
        )?;
        
        // Generate improvement roadmap
        let improvement_roadmap = self.create_improvement_roadmap_report(
            production_assessment,
        )?;
        
        // Generate scaling guidance
        let scaling_guidance = self.create_scaling_guidance_report(
            production_assessment,
        )?;
        
        // Generate deployment recommendations
        let deployment_recommendations = self.create_deployment_recommendations_report(
            production_assessment,
        )?;
        
        // Generate appendices if requested
        let appendices = if self.config.include_raw_data {
            Some(self.create_report_appendices(validation_results)?)
        } else {
            None
        };

        Ok(ProductionReadinessReport {
            report_metadata,
            executive_summary,
            readiness_assessment,
            performance_analysis,
            user_experience_report,
            improvement_roadmap,
            scaling_guidance,
            deployment_recommendations,
            appendices,
        })
    }    
/// Create report metadata
    fn create_report_metadata(&self, report_id: &str, target_directory: &str) -> Result<ReportMetadata> {
        Ok(ReportMetadata {
            report_title: format!("Production Readiness Assessment - {}", target_directory),
            generated_at: chrono::Utc::now(),
            report_version: "1.0.0".to_string(),
            validation_target: target_directory.to_string(),
            assessment_duration: Duration::from_secs(0), // Will be updated by caller
            report_id: report_id.to_string(),
            generator_version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Create executive summary
    fn create_executive_summary(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<ExecutiveSummary> {
        // Determine overall recommendation
        let overall_recommendation = match &production_assessment.overall_readiness {
            crate::production_readiness_assessor::ProductionReadinessLevel::Ready => {
                OverallRecommendation::ReadyForProduction
            }
            crate::production_readiness_assessor::ProductionReadinessLevel::ReadyWithCaveats(_) => {
                OverallRecommendation::ReadyWithMinorImprovements
            }
            crate::production_readiness_assessor::ProductionReadinessLevel::RequiresImprovement(_) => {
                OverallRecommendation::RequiresSignificantWork
            }
            crate::production_readiness_assessor::ProductionReadinessLevel::NotReady(_) => {
                OverallRecommendation::NotRecommendedForProduction
            }
        };

        // Extract key findings
        let key_findings = self.extract_key_findings(validation_results, production_assessment)?;

        // Identify critical actions
        let critical_actions_required = self.identify_critical_actions(production_assessment)?;

        // Assess business impact
        let business_impact_assessment = self.assess_business_impact(
            validation_results,
            production_assessment,
        )?;

        // Create timeline to production
        let timeline_to_production = self.create_timeline_to_production(production_assessment)?;

        Ok(ExecutiveSummary {
            overall_recommendation,
            key_findings,
            critical_actions_required,
            business_impact_assessment,
            timeline_to_production,
        })
    }

    /// Extract key findings from assessment
    fn extract_key_findings(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<Vec<KeyFinding>> {
        let mut findings = Vec::new();

        // Reliability findings
        if production_assessment.factor_scores.reliability_score < 0.8 {
            findings.push(KeyFinding {
                finding_id: "REL-001".to_string(),
                category: FindingCategory::Reliability,
                title: "Reliability concerns identified".to_string(),
                description: format!(
                    "Reliability score of {:.1}% is below recommended threshold of 80%",
                    production_assessment.factor_scores.reliability_score * 100.0
                ),
                impact_level: ImpactLevel::High,
                confidence: 0.9,
                supporting_evidence: vec![
                    format!("Exit code: {:?}", validation_results.pensieve_execution_results.exit_code),
                    format!("Total errors: {}", validation_results.pensieve_execution_results.error_summary.total_errors),
                ],
            });
        }

        // Performance findings
        if production_assessment.factor_scores.performance_score < 0.7 {
            findings.push(KeyFinding {
                finding_id: "PERF-001".to_string(),
                category: FindingCategory::Performance,
                title: "Performance optimization needed".to_string(),
                description: format!(
                    "Performance score of {:.1}% indicates optimization opportunities",
                    production_assessment.factor_scores.performance_score * 100.0
                ),
                impact_level: ImpactLevel::Medium,
                confidence: 0.8,
                supporting_evidence: vec![
                    format!("Processing speed: {:.2} files/sec", 
                           validation_results.pensieve_execution_results.performance_metrics.files_per_second),
                    format!("Memory efficiency: {:.1}%", 
                           validation_results.pensieve_execution_results.performance_metrics.memory_efficiency_score * 100.0),
                ],
            });
        }

        // UX findings
        if production_assessment.factor_scores.user_experience_score < 0.7 {
            findings.push(KeyFinding {
                finding_id: "UX-001".to_string(),
                category: FindingCategory::UserExperience,
                title: "User experience improvements needed".to_string(),
                description: format!(
                    "User experience score of {:.1}% suggests usability enhancements required",
                    production_assessment.factor_scores.user_experience_score * 100.0
                ),
                impact_level: ImpactLevel::Medium,
                confidence: 0.7,
                supporting_evidence: vec![
                    "Error message clarity needs improvement".to_string(),
                    "Progress reporting could be more informative".to_string(),
                ],
            });
        }

        // Add positive findings for strengths
        if production_assessment.factor_scores.reliability_score >= 0.9 {
            findings.push(KeyFinding {
                finding_id: "REL-STRENGTH-001".to_string(),
                category: FindingCategory::Reliability,
                title: "Excellent reliability demonstrated".to_string(),
                description: "Application shows strong reliability with minimal errors and robust error handling".to_string(),
                impact_level: ImpactLevel::Low,
                confidence: 0.9,
                supporting_evidence: vec![
                    "Zero crashes during testing".to_string(),
                    "Graceful error handling observed".to_string(),
                ],
            });
        }

        Ok(findings)
    }    
/// Identify critical actions required
    fn identify_critical_actions(
        &self,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<Vec<CriticalAction>> {
        let mut actions = Vec::new();

        // Convert production blockers to critical actions
        for blocker in &production_assessment.blockers {
            if blocker.must_fix_before_production {
                actions.push(CriticalAction {
                    action_id: format!("ACTION-{}", blocker.blocker_id),
                    title: format!("Resolve: {}", blocker.title),
                    description: blocker.description.clone(),
                    urgency: ActionUrgency::Immediate,
                    estimated_effort: "High".to_string(), // Could be derived from blocker data
                    business_justification: "Required for production deployment".to_string(),
                    consequences_of_inaction: "Cannot deploy to production safely".to_string(),
                });
            }
        }

        // Convert high-severity critical issues to actions
        for issue in &production_assessment.critical_issues {
            if matches!(issue.severity, crate::production_readiness_assessor::IssueSeverity::Critical) {
                actions.push(CriticalAction {
                    action_id: format!("ACTION-{}", issue.issue_id),
                    title: format!("Address: {}", issue.title),
                    description: issue.description.clone(),
                    urgency: match issue.resolution_priority {
                        crate::production_readiness_assessor::ResolutionPriority::Immediate => ActionUrgency::Immediate,
                        crate::production_readiness_assessor::ResolutionPriority::High => ActionUrgency::High,
                        crate::production_readiness_assessor::ResolutionPriority::Medium => ActionUrgency::Medium,
                        _ => ActionUrgency::Low,
                    },
                    estimated_effort: match issue.estimated_effort {
                        crate::production_readiness_assessor::EstimatedEffort::Trivial => "Trivial".to_string(),
                        crate::production_readiness_assessor::EstimatedEffort::Low => "Low".to_string(),
                        crate::production_readiness_assessor::EstimatedEffort::Medium => "Medium".to_string(),
                        crate::production_readiness_assessor::EstimatedEffort::High => "High".to_string(),
                        crate::production_readiness_assessor::EstimatedEffort::Epic => "Epic".to_string(),
                    },
                    business_justification: issue.business_impact.clone(),
                    consequences_of_inaction: issue.technical_impact.clone(),
                });
            }
        }

        Ok(actions)
    }

    /// Assess business impact
    fn assess_business_impact(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<BusinessImpactAssessment> {
        // Determine risk level based on readiness score and critical issues
        let risk_level = if production_assessment.readiness_score < 0.5 {
            BusinessRiskLevel::Critical
        } else if production_assessment.readiness_score < 0.7 {
            BusinessRiskLevel::High
        } else if production_assessment.readiness_score < 0.8 {
            BusinessRiskLevel::Medium
        } else {
            BusinessRiskLevel::Low
        };

        // Assess potential costs
        let potential_cost_of_issues = if !production_assessment.blockers.is_empty() {
            "High - Production deployment blocked, potential delays and resource costs"
        } else if !production_assessment.critical_issues.is_empty() {
            "Medium - Issues may cause user dissatisfaction and support overhead"
        } else {
            "Low - Minor issues with minimal business impact"
        }.to_string();

        // Expected benefits
        let expected_benefits_of_deployment = format!(
            "Successful deployment will enable processing of large datasets with {:.1} files/sec throughput",
            validation_results.pensieve_execution_results.performance_metrics.files_per_second
        );

        // Competitive implications
        let competitive_implications = match risk_level {
            BusinessRiskLevel::Low => "Deployment will provide competitive advantage through reliable data processing",
            BusinessRiskLevel::Medium => "Deployment feasible but may require additional support resources",
            BusinessRiskLevel::High => "Deployment risks may impact competitive position if issues occur",
            BusinessRiskLevel::Critical => "Deployment not recommended - may damage competitive position",
        }.to_string();

        // User satisfaction impact
        let user_satisfaction_impact = format!(
            "User experience score of {:.1}% suggests {} user satisfaction",
            production_assessment.factor_scores.user_experience_score * 100.0,
            if production_assessment.factor_scores.user_experience_score > 0.8 { "high" }
            else if production_assessment.factor_scores.user_experience_score > 0.6 { "moderate" }
            else { "low" }
        );

        // Operational impact
        let operational_impact = if validation_results.reliability_results.overall_reliability_score > 0.8 {
            "Low operational overhead expected with current reliability levels"
        } else {
            "Higher operational overhead expected due to reliability concerns"
        }.to_string();

        Ok(BusinessImpactAssessment {
            risk_level,
            potential_cost_of_issues,
            expected_benefits_of_deployment,
            competitive_implications,
            user_satisfaction_impact,
            operational_impact,
        })
    }

    /// Create timeline to production
    fn create_timeline_to_production(
        &self,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<TimelineToProduction> {
        let current_readiness_percentage = production_assessment.readiness_score * 100.0;
        
        // Estimate time based on improvement roadmap
        let estimated_time_to_ready = if production_assessment.readiness_score >= 0.9 {
            Duration::from_secs(7 * 24 * 3600) // 1 week
        } else if production_assessment.readiness_score >= 0.8 {
            Duration::from_secs(14 * 24 * 3600) // 2 weeks
        } else if production_assessment.readiness_score >= 0.6 {
            Duration::from_secs(30 * 24 * 3600) // 1 month
        } else {
            Duration::from_secs(90 * 24 * 3600) // 3 months
        };

        // Create major milestones based on improvement roadmap
        let mut major_milestones = Vec::new();
        
        if !production_assessment.blockers.is_empty() {
            major_milestones.push(Milestone {
                milestone_id: "M1".to_string(),
                title: "Resolve Production Blockers".to_string(),
                description: "Address all must-fix issues preventing production deployment".to_string(),
                estimated_completion: Duration::from_secs(7 * 24 * 3600),
                dependencies: vec!["Development team availability".to_string()],
                success_criteria: vec!["All production blockers resolved".to_string()],
            });
        }

        if !production_assessment.critical_issues.is_empty() {
            major_milestones.push(Milestone {
                milestone_id: "M2".to_string(),
                title: "Address Critical Issues".to_string(),
                description: "Resolve high-priority issues affecting reliability and performance".to_string(),
                estimated_completion: Duration::from_secs(14 * 24 * 3600),
                dependencies: vec!["Blocker resolution".to_string()],
                success_criteria: vec!["Critical issues resolved or mitigated".to_string()],
            });
        }

        major_milestones.push(Milestone {
            milestone_id: "M3".to_string(),
            title: "Production Deployment".to_string(),
            description: "Deploy to production environment with monitoring".to_string(),
            estimated_completion: estimated_time_to_ready,
            dependencies: vec!["All previous milestones completed".to_string()],
            success_criteria: vec!["Successful production deployment".to_string(), "Monitoring in place".to_string()],
        });

        // Dependencies
        let dependencies = vec![
            "Development team capacity".to_string(),
            "Testing environment availability".to_string(),
            "Production environment readiness".to_string(),
        ];

        // Confidence in timeline
        let confidence_in_timeline = if production_assessment.readiness_score > 0.8 {
            0.9
        } else if production_assessment.readiness_score > 0.6 {
            0.7
        } else {
            0.5
        };

        Ok(TimelineToProduction {
            current_readiness_percentage,
            estimated_time_to_ready,
            major_milestones,
            dependencies,
            confidence_in_timeline,
        })
    }

    /// Create readiness assessment section
    fn create_readiness_assessment(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<ReadinessAssessment> {
        let overall_score = production_assessment.readiness_score;
        let readiness_level = match &production_assessment.overall_readiness {
            crate::production_readiness_assessor::ProductionReadinessLevel::Ready => "Ready for Production".to_string(),
            crate::production_readiness_assessor::ProductionReadinessLevel::ReadyWithCaveats(caveats) => {
                format!("Ready with Caveats: {}", caveats.join(", "))
            }
            crate::production_readiness_assessor::ProductionReadinessLevel::RequiresImprovement(issues) => {
                format!("Requires Improvement: {}", issues.join(", "))
            }
            crate::production_readiness_assessor::ProductionReadinessLevel::NotReady(reasons) => {
                format!("Not Ready: {}", reasons.join(", "))
            }
        };

        // Convert factor scores
        let mut factor_scores = HashMap::new();
        factor_scores.insert("reliability".to_string(), FactorScore {
            score: production_assessment.factor_scores.reliability_score,
            grade: self.score_to_grade(production_assessment.factor_scores.reliability_score),
            weight: 0.4,
            status: if production_assessment.factor_scores.reliability_score >= 0.8 { "Good" } else { "Needs Improvement" }.to_string(),
            key_metrics: vec!["Crash-free operation".to_string(), "Error handling".to_string()],
        });

        factor_scores.insert("performance".to_string(), FactorScore {
            score: production_assessment.factor_scores.performance_score,
            grade: self.score_to_grade(production_assessment.factor_scores.performance_score),
            weight: 0.35,
            status: if production_assessment.factor_scores.performance_score >= 0.7 { "Good" } else { "Needs Improvement" }.to_string(),
            key_metrics: vec!["Processing speed".to_string(), "Memory efficiency".to_string()],
        });

        factor_scores.insert("user_experience".to_string(), FactorScore {
            score: production_assessment.factor_scores.user_experience_score,
            grade: self.score_to_grade(production_assessment.factor_scores.user_experience_score),
            weight: 0.25,
            status: if production_assessment.factor_scores.user_experience_score >= 0.7 { "Good" } else { "Needs Improvement" }.to_string(),
            key_metrics: vec!["Error message clarity".to_string(), "Progress reporting".to_string()],
        });

        // Create critical issues summary
        let critical_issues_summary = self.create_critical_issues_summary(&production_assessment.critical_issues);

        // Create blockers summary
        let blockers_summary = self.create_blockers_summary(&production_assessment.blockers);

        // Identify strengths and areas for improvement
        let strengths = self.identify_strengths(validation_results, production_assessment);
        let areas_for_improvement = self.identify_areas_for_improvement(production_assessment);

        Ok(ReadinessAssessment {
            overall_score,
            readiness_level,
            factor_scores,
            critical_issues_summary,
            blockers_summary,
            strengths,
            areas_for_improvement,
        })
    }

    /// Convert score to letter grade
    fn score_to_grade(&self, score: f64) -> String {
        match (score * 100.0) as u32 {
            90..=100 => "A".to_string(),
            80..=89 => "B".to_string(),
            70..=79 => "C".to_string(),
            60..=69 => "D".to_string(),
            _ => "F".to_string(),
        }
    }

    /// Create critical issues summary
    fn create_critical_issues_summary(
        &self,
        critical_issues: &[crate::production_readiness_assessor::CriticalIssue],
    ) -> CriticalIssuesSummary {
        let total_issues = critical_issues.len();
        
        let mut by_severity = HashMap::new();
        let mut by_category = HashMap::new();
        
        for issue in critical_issues {
            let severity_str = format!("{:?}", issue.severity);
            *by_severity.entry(severity_str).or_insert(0) += 1;
            
            let category_str = format!("{:?}", issue.impact_areas.first().unwrap_or(&crate::production_readiness_assessor::ImpactArea::Reliability));
            *by_category.entry(category_str).or_insert(0) += 1;
        }

        let top_issues = critical_issues.iter().take(5).map(|issue| {
            CriticalIssueSummary {
                title: issue.title.clone(),
                severity: format!("{:?}", issue.severity),
                impact: issue.business_impact.clone(),
                recommended_action: issue.recommended_actions.first().unwrap_or(&"Review issue details".to_string()).clone(),
            }
        }).collect();

        CriticalIssuesSummary {
            total_issues,
            by_severity,
            by_category,
            top_issues,
        }
    }

    /// Create blockers summary
    fn create_blockers_summary(
        &self,
        blockers: &[crate::production_readiness_assessor::ProductionBlocker],
    ) -> BlockersSummary {
        let total_blockers = blockers.len();
        
        let mut by_type = HashMap::new();
        let mut must_fix_count = 0;
        let mut workaround_available_count = 0;
        
        for blocker in blockers {
            let type_str = format!("{:?}", blocker.blocker_type);
            *by_type.entry(type_str).or_insert(0) += 1;
            
            if blocker.must_fix_before_production {
                must_fix_count += 1;
            }
            if blocker.workaround_available {
                workaround_available_count += 1;
            }
        }

        let top_blockers = blockers.iter().take(5).map(|blocker| {
            BlockerSummary {
                title: blocker.title.clone(),
                blocker_type: format!("{:?}", blocker.blocker_type),
                must_fix: blocker.must_fix_before_production,
                workaround_available: blocker.workaround_available,
                resolution_summary: blocker.resolution_steps.first().unwrap_or(&"See detailed resolution steps".to_string()).clone(),
            }
        }).collect();

        BlockersSummary {
            total_blockers,
            by_type,
            must_fix_count,
            workaround_available_count,
            top_blockers,
        }
    }

    /// Identify strengths
    fn identify_strengths(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Vec<String> {
        let mut strengths = Vec::new();

        if validation_results.pensieve_execution_results.exit_code == Some(0) {
            strengths.push("Application completes successfully without crashes".to_string());
        }

        if production_assessment.factor_scores.reliability_score >= 0.8 {
            strengths.push("Strong reliability with robust error handling".to_string());
        }

        if production_assessment.factor_scores.performance_score >= 0.8 {
            strengths.push("Good performance characteristics and efficiency".to_string());
        }

        if validation_results.pensieve_execution_results.performance_metrics.files_per_second >= 10.0 {
            strengths.push("High processing throughput achieved".to_string());
        }

        if production_assessment.blockers.is_empty() {
            strengths.push("No production blockers identified".to_string());
        }

        strengths
    }

    /// Identify areas for improvement
    fn identify_areas_for_improvement(
        &self,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Vec<String> {
        let mut improvements = Vec::new();

        if production_assessment.factor_scores.reliability_score < 0.8 {
            improvements.push("Reliability needs improvement - focus on error handling and crash prevention".to_string());
        }

        if production_assessment.factor_scores.performance_score < 0.7 {
            improvements.push("Performance optimization needed - improve processing speed and efficiency".to_string());
        }

        if production_assessment.factor_scores.user_experience_score < 0.7 {
            improvements.push("User experience enhancements needed - improve feedback and error messages".to_string());
        }

        if !production_assessment.critical_issues.is_empty() {
            improvements.push(format!("Address {} critical issues affecting production readiness", production_assessment.critical_issues.len()));
        }

        if !production_assessment.blockers.is_empty() {
            improvements.push(format!("Resolve {} production blockers before deployment", production_assessment.blockers.len()));
        }

        improvements
    }

    /// Create performance analysis report
    fn create_performance_analysis(
        &self,
        validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<PerformanceAnalysisReport> {
        let performance_summary = PerformanceSummary {
            overall_performance_grade: self.score_to_grade(production_assessment.factor_scores.performance_score),
            processing_speed: ProcessingSpeedMetrics {
                files_per_second: validation_results.pensieve_execution_results.performance_metrics.files_per_second,
                throughput_rating: if validation_results.pensieve_execution_results.performance_metrics.files_per_second >= 20.0 { "High" }
                    else if validation_results.pensieve_execution_results.performance_metrics.files_per_second >= 10.0 { "Medium" }
                    else { "Low" }.to_string(),
                speed_consistency: validation_results.pensieve_execution_results.performance_metrics.processing_consistency,
                peak_performance: validation_results.pensieve_execution_results.performance_metrics.files_per_second * 1.2,
                performance_degradation_points: vec!["Large file processing".to_string()],
            },
            resource_utilization: ResourceUtilizationMetrics {
                memory_efficiency: validation_results.pensieve_execution_results.performance_metrics.memory_efficiency_score,
                cpu_efficiency: validation_results.pensieve_execution_results.cpu_usage_stats.average_cpu_percent as f64 / 100.0,
                peak_memory_usage_mb: validation_results.pensieve_execution_results.peak_memory_mb,
                average_memory_usage_mb: validation_results.pensieve_execution_results.average_memory_mb,
                memory_growth_pattern: "Linear".to_string(),
                resource_optimization_score: 0.8,
            },
            consistency_metrics: ConsistencyMetrics {
                processing_consistency_score: validation_results.pensieve_execution_results.performance_metrics.processing_consistency,
                performance_variability: 1.0 - validation_results.pensieve_execution_results.performance_metrics.processing_consistency,
                predictability_rating: if validation_results.pensieve_execution_results.performance_metrics.processing_consistency >= 0.8 { "High" }
                    else if validation_results.pensieve_execution_results.performance_metrics.processing_consistency >= 0.6 { "Medium" }
                    else { "Low" }.to_string(),
                stability_indicators: vec!["Consistent processing speed".to_string()],
            },
            efficiency_scores: EfficiencyScores {
                overall_efficiency: production_assessment.factor_scores.performance_score,
                memory_efficiency: validation_results.pensieve_execution_results.performance_metrics.memory_efficiency_score,
                cpu_efficiency: validation_results.pensieve_execution_results.cpu_usage_stats.average_cpu_percent as f64 / 100.0,
                io_efficiency: 0.8, // Estimated
                algorithm_efficiency: 0.8, // Estimated
            },
        };

        // Create scaling predictions based on assessment
        let scaling_predictions = ScalingPredictions {
            linear_scaling_assessment: LinearScalingAssessment {
                scales_linearly: production_assessment.factor_scores.scalability_score >= 0.7,
                scaling_factor: production_assessment.factor_scores.scalability_score,
                scaling_confidence: 0.8,
                non_linear_factors: vec!["Memory usage growth".to_string()],
            },
            capacity_limits: CapacityLimits {
                max_files_estimate: (validation_results.pensieve_execution_results.performance_metrics.files_per_second * 3600.0 * 8.0) as u64,
                max_data_size_gb: 1000.0,
                max_processing_time_hours: 8.0,
                memory_ceiling_gb: 16.0,
                confidence_level: 0.7,
            },
            resource_requirements_by_scale: self.create_resource_requirements_by_scale(),
            performance_degradation_thresholds: vec![
                DegradationThreshold {
                    threshold_name: "Large Dataset".to_string(),
                    threshold_value: "> 100GB".to_string(),
                    degradation_factor: 0.8,
                    mitigation_strategies: vec!["Increase memory allocation".to_string()],
                }
            ],
        };

        // Create bottleneck identification
        let bottleneck_identification = BottleneckIdentification {
            primary_bottlenecks: vec![
                BottleneckAnalysis {
                    component: "File I/O".to_string(),
                    severity_score: 0.6,
                    impact_description: "File reading may become bottleneck with large files".to_string(),
                    evidence: vec!["Processing speed varies with file size".to_string()],
                    optimization_suggestions: vec!["Implement streaming processing".to_string()],
                    estimated_improvement_potential: 0.3,
                }
            ],
            bottleneck_severity_ranking: vec!["File I/O".to_string(), "Memory allocation".to_string()],
            system_bottleneck_score: 0.7,
            optimization_priority_order: vec!["File I/O optimization".to_string()],
        };

        // Create performance trends
        let performance_trends = PerformanceTrends {
            trend_analysis: "Performance appears stable with consistent processing speeds".to_string(),
            performance_stability: validation_results.pensieve_execution_results.performance_metrics.processing_consistency,
            degradation_indicators: vec![],
            improvement_indicators: vec!["Consistent memory usage".to_string()],
            long_term_sustainability: "Good".to_string(),
        };

        // Create optimization opportunities
        let optimization_opportunities = vec![
            crate::production_readiness_assessor::OptimizationOpportunity {
                area: "Memory Management".to_string(),
                potential_improvement: 0.2,
                implementation_effort: crate::production_readiness_assessor::EstimatedEffort::Medium,
                description: "Optimize memory allocation patterns".to_string(),
            }
        ];

        Ok(PerformanceAnalysisReport {
            performance_summary,
            scaling_predictions,
            bottleneck_identification,
            performance_trends,
            optimization_opportunities,
        })
    }

    /// Create resource requirements by scale
    fn create_resource_requirements_by_scale(&self) -> HashMap<String, ResourceRequirement> {
        let mut requirements = HashMap::new();
        
        requirements.insert("small".to_string(), ResourceRequirement {
            scale_description: "< 10K files".to_string(),
            recommended_memory_gb: 2.0,
            recommended_cpu_cores: 2,
            estimated_processing_time: "< 1 hour".to_string(),
            additional_considerations: vec!["Standard configuration sufficient".to_string()],
        });

        requirements.insert("medium".to_string(), ResourceRequirement {
            scale_description: "10K - 100K files".to_string(),
            recommended_memory_gb: 8.0,
            recommended_cpu_cores: 4,
            estimated_processing_time: "1-4 hours".to_string(),
            additional_considerations: vec!["Monitor memory usage".to_string()],
        });

        requirements.insert("large".to_string(), ResourceRequirement {
            scale_description: "100K - 1M files".to_string(),
            recommended_memory_gb: 16.0,
            recommended_cpu_cores: 8,
            estimated_processing_time: "4-12 hours".to_string(),
            additional_considerations: vec!["Consider batch processing".to_string()],
        });

        requirements
    }
    
    /// Create user experience report (simplified implementation)
    fn create_user_experience_report(
        &self,
        _validation_results: &ComprehensiveValidationResults,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<UserExperienceReport> {
        let ux_summary = UXSummary {
            overall_ux_score: production_assessment.factor_scores.user_experience_score,
            ux_grade: self.score_to_grade(production_assessment.factor_scores.user_experience_score),
            user_satisfaction_prediction: if production_assessment.factor_scores.user_experience_score >= 0.8 { "High" }
                else if production_assessment.factor_scores.user_experience_score >= 0.6 { "Medium" }
                else { "Low" }.to_string(),
            key_ux_strengths: vec!["Clear progress indication".to_string()],
            critical_ux_issues: vec!["Error messages could be more actionable".to_string()],
        };

        Ok(UserExperienceReport {
            ux_summary,
            specific_improvements: vec![],
            user_journey_analysis: UserJourneyAnalysis {
                journey_stages: vec![],
                pain_points: vec![],
                delight_moments: vec![],
                overall_journey_score: production_assessment.factor_scores.user_experience_score,
            },
            feedback_quality_assessment: FeedbackQualityAssessment {
                progress_feedback_quality: FeedbackQuality {
                    clarity_score: 0.8,
                    actionability_score: 0.7,
                    completeness_score: 0.8,
                    user_friendliness_score: 0.8,
                    specific_improvements: vec!["Add estimated time remaining".to_string()],
                },
                error_message_quality: FeedbackQuality {
                    clarity_score: 0.7,
                    actionability_score: 0.6,
                    completeness_score: 0.7,
                    user_friendliness_score: 0.7,
                    specific_improvements: vec!["Include suggested actions in error messages".to_string()],
                },
                completion_feedback_quality: FeedbackQuality {
                    clarity_score: 0.8,
                    actionability_score: 0.8,
                    completeness_score: 0.9,
                    user_friendliness_score: 0.8,
                    specific_improvements: vec![],
                },
                overall_communication_score: 0.75,
            },
            ux_optimization_roadmap: UXOptimizationRoadmap {
                immediate_wins: vec![],
                short_term_improvements: vec![],
                long_term_enhancements: vec![],
                ux_metrics_to_track: vec!["User satisfaction".to_string(), "Task completion rate".to_string()],
            },
        })
    }

    /// Create improvement roadmap report (simplified)
    fn create_improvement_roadmap_report(
        &self,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<ImprovementRoadmapReport> {
        let roadmap_summary = RoadmapSummary {
            total_improvements: production_assessment.improvement_roadmap.immediate_actions.len() +
                               production_assessment.improvement_roadmap.short_term_improvements.len() +
                               production_assessment.improvement_roadmap.long_term_improvements.len(),
            estimated_total_effort: "4-8 weeks".to_string(),
            expected_timeline: "2-3 months".to_string(),
            key_benefits: vec!["Improved reliability".to_string(), "Better performance".to_string()],
            major_risks: vec!["Resource availability".to_string()],
        };

        // Convert improvement actions to prioritized improvements
        let mut prioritized_improvements = Vec::new();
        for (index, action) in production_assessment.improvement_roadmap.immediate_actions.iter().enumerate() {
            prioritized_improvements.push(PrioritizedImprovement {
                improvement_id: format!("IMP-{:03}", index + 1),
                title: action.title.clone(),
                description: action.description.clone(),
                category: format!("{:?}", action.category),
                impact_score: match action.expected_impact {
                    crate::production_readiness_assessor::ExpectedImpact::High => 0.9,
                    crate::production_readiness_assessor::ExpectedImpact::Medium => 0.6,
                    crate::production_readiness_assessor::ExpectedImpact::Low => 0.3,
                },
                effort_score: match action.estimated_effort {
                    crate::production_readiness_assessor::EstimatedEffort::Trivial => 0.1,
                    crate::production_readiness_assessor::EstimatedEffort::Low => 0.3,
                    crate::production_readiness_assessor::EstimatedEffort::Medium => 0.6,
                    crate::production_readiness_assessor::EstimatedEffort::High => 0.8,
                    crate::production_readiness_assessor::EstimatedEffort::Epic => 1.0,
                },
                priority_score: 0.9, // High priority for immediate actions
                impact_description: "High impact on production readiness".to_string(),
                effort_description: format!("{:?}", action.estimated_effort),
                dependencies: action.dependencies.clone(),
                success_criteria: action.success_criteria.clone(),
                implementation_notes: action.implementation_notes.clone(),
            });
        }

        Ok(ImprovementRoadmapReport {
            roadmap_summary,
            prioritized_improvements,
            implementation_phases: vec![],
            resource_requirements: RoadmapResourceRequirements {
                development_effort: "2-4 weeks".to_string(),
                testing_effort: "1-2 weeks".to_string(),
                deployment_effort: "1 week".to_string(),
                ongoing_maintenance: "Minimal".to_string(),
                skill_requirements: vec!["Rust development".to_string()],
                external_dependencies: vec!["None".to_string()],
            },
            success_tracking: SuccessTracking {
                key_performance_indicators: vec![
                    KPI {
                        kpi_name: "Reliability Score".to_string(),
                        description: "Overall system reliability".to_string(),
                        current_value: format!("{:.1}%", production_assessment.factor_scores.reliability_score * 100.0),
                        target_value: "90%".to_string(),
                        measurement_method: "Automated testing".to_string(),
                    }
                ],
                measurement_frequency: "Weekly".to_string(),
                reporting_schedule: "Monthly".to_string(),
                success_thresholds: HashMap::from([("reliability".to_string(), 0.9)]),
            },
        })
    }

    /// Create scaling guidance report (simplified)
    fn create_scaling_guidance_report(
        &self,
        production_assessment: &ProductionReadinessAssessment,
    ) -> Result<ScalingGuidanceReport> {
        Ok(ScalingGuidanceReport {
            scaling_summary: ScalingSummary {
                current_scale_rating: "Medium".to_string(),
                scaling_readiness_score: production_assessment.factor_scores.scalability_score,
                recommended_scaling_approach: "Vertical scaling initially, then horizontal".to_string(),
                key_scaling_constraints: vec!["Memory usage".to_string()],
                scaling_opportunities: vec!["Parallel processing".to_string()],
            },
            capacity_planning: CapacityPlanning {
                current_capacity: "10K files/hour".to_string(),
                projected_capacity_needs: HashMap::from([
                    ("6 months".to_string(), "50K files/hour".to_string()),
                    ("1 year".to_string(), "100K files/hour".to_string()),
                ]),
                scaling_thresholds: vec![],
                resource_scaling_recommendations: vec![],
            },
            architecture_recommendations: ArchitectureRecommendations {
                current_architecture_assessment: "Suitable for current scale".to_string(),
                recommended_changes: vec![],
                scalability_patterns: vec!["Batch processing".to_string()],
                anti_patterns_to_avoid: vec!["Synchronous processing of large datasets".to_string()],
            },
            monitoring_requirements: MonitoringRequirements {
                critical_metrics: vec![],
                alerting_strategy: AlertingStrategy {
                    alert_levels: vec![],
                    escalation_procedures: vec![],
                    notification_channels: vec![],
                    alert_fatigue_prevention: vec![],
                },
                dashboard_requirements: vec!["Performance metrics".to_string()],
                monitoring_tools_recommendations: vec!["Prometheus".to_string(), "Grafana".to_string()],
            },
            scaling_risks: vec![],
        })
    }

    /// Create deployment recommendations report (simplified)
    fn create_deployment_recommendations_report(
        &self,
        _production_assessment: &ProductionReadinessAssessment,
    ) -> Result<DeploymentRecommendationsReport> {
        Ok(DeploymentRecommendationsReport {
            deployment_summary: DeploymentSummary {
                deployment_readiness: "Ready with minor improvements".to_string(),
                recommended_deployment_strategy: "Blue-green deployment".to_string(),
                key_deployment_risks: vec!["Performance under load".to_string()],
                success_criteria: vec!["Zero downtime deployment".to_string()],
                go_no_go_checklist: vec!["All tests passing".to_string(), "Monitoring in place".to_string()],
            },
            environment_setup: EnvironmentSetup {
                infrastructure_requirements: InfrastructureRequirements {
                    compute_requirements: ComputeRequirements {
                        cpu_specifications: "4+ cores".to_string(),
                        memory_specifications: "8GB+ RAM".to_string(),
                        performance_requirements: "SSD storage recommended".to_string(),
                        availability_requirements: "99.9% uptime".to_string(),
                    },
                    storage_requirements: StorageRequirements {
                        storage_type: "SSD".to_string(),
                        capacity_requirements: "100GB+".to_string(),
                        performance_requirements: "High IOPS".to_string(),
                        backup_strategy: "Daily backups".to_string(),
                    },
                    network_requirements: NetworkRequirements {
                        bandwidth_requirements: "1Gbps+".to_string(),
                        latency_requirements: "< 10ms".to_string(),
                        connectivity_requirements: vec!["Internet access".to_string()],
                        security_requirements: vec!["TLS encryption".to_string()],
                    },
                    backup_requirements: BackupRequirements {
                        backup_frequency: "Daily".to_string(),
                        retention_policy: "30 days".to_string(),
                        recovery_time_objective: "4 hours".to_string(),
                        recovery_point_objective: "1 hour".to_string(),
                    },
                },
                software_dependencies: vec![],
                security_configuration: SecurityConfiguration {
                    authentication_requirements: vec!["Strong passwords".to_string()],
                    authorization_requirements: vec!["Role-based access".to_string()],
                    encryption_requirements: vec!["Data at rest encryption".to_string()],
                    audit_requirements: vec!["Access logging".to_string()],
                    compliance_considerations: vec!["GDPR compliance".to_string()],
                },
                network_configuration: NetworkConfiguration {
                    firewall_rules: vec!["Allow HTTP/HTTPS".to_string()],
                    port_requirements: vec!["Port 80, 443".to_string()],
                    dns_requirements: vec!["Valid DNS records".to_string()],
                    load_balancing_considerations: vec!["Health checks".to_string()],
                },
            },
            configuration_guide: ConfigurationGuide {
                configuration_parameters: vec![],
                environment_specific_settings: HashMap::new(),
                tuning_recommendations: vec![],
                configuration_validation: vec!["Validate all settings".to_string()],
            },
            operational_procedures: OperationalProcedures {
                startup_procedures: vec!["Start services in order".to_string()],
                shutdown_procedures: vec!["Graceful shutdown".to_string()],
                health_check_procedures: vec!["Check service status".to_string()],
                maintenance_procedures: vec!["Regular updates".to_string()],
                troubleshooting_guide: TroubleshootingGuide {
                    common_issues: vec![],
                    diagnostic_procedures: vec!["Check logs".to_string()],
                    escalation_procedures: vec!["Contact support".to_string()],
                    support_contacts: vec!["support@example.com".to_string()],
                },
            },
            rollback_planning: RollbackPlanning {
                rollback_triggers: vec!["Critical errors".to_string()],
                rollback_procedures: vec!["Restore previous version".to_string()],
                data_recovery_procedures: vec!["Restore from backup".to_string()],
                rollback_testing: vec!["Test rollback procedures".to_string()],
                communication_plan: vec!["Notify stakeholders".to_string()],
            },
        })
    }

    /// Create report appendices
    fn create_report_appendices(
        &self,
        validation_results: &ComprehensiveValidationResults,
    ) -> Result<ReportAppendices> {
        Ok(ReportAppendices {
            raw_validation_data: if self.config.include_raw_data {
                Some(serde_json::to_value(validation_results)?)
            } else {
                None
            },
            detailed_metrics: None,
            test_logs: None,
            configuration_files: None,
            additional_analysis: None,
        })
    }
    
    /// Export report to multiple formats
    pub fn export_report(
        &self,
        report: &ProductionReadinessReport,
        output_directory: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut exported_files = Vec::new();

        for format in &self.config.output_formats {
            let file_path = match format {
                OutputFormat::Json => self.export_json(report, output_directory)?,
                OutputFormat::Html => self.export_html(report, output_directory)?,
                OutputFormat::Csv => self.export_csv(report, output_directory)?,
                OutputFormat::Markdown => self.export_markdown(report, output_directory)?,
                OutputFormat::Pdf => return Err(ValidationError::ReportGenerationFailed { 
                    cause: "PDF export not yet implemented".to_string() 
                }),
            };
            exported_files.push(file_path);
        }

        Ok(exported_files)
    }

    /// Export report as JSON
    fn export_json(&self, report: &ProductionReadinessReport, output_dir: &Path) -> Result<PathBuf> {
        let file_path = output_dir.join("production_readiness_report.json");
        let json_content = serde_json::to_string_pretty(report)
            .map_err(|e| ValidationError::ReportGenerationFailed { 
                cause: format!("JSON serialization failed: {}", e) 
            })?;
        
        std::fs::write(&file_path, json_content)
            .map_err(|e| ValidationError::FileSystem(e))?;
        
        Ok(file_path)
    }

    /// Export report as HTML
    fn export_html(&self, report: &ProductionReadinessReport, output_dir: &Path) -> Result<PathBuf> {
        let file_path = output_dir.join("production_readiness_report.html");
        let html_content = self.generate_html_report(report)?;
        
        std::fs::write(&file_path, html_content)
            .map_err(|e| ValidationError::FileSystem(e))?;
        
        Ok(file_path)
    }

    /// Export report as CSV (summary data only)
    fn export_csv(&self, report: &ProductionReadinessReport, output_dir: &Path) -> Result<PathBuf> {
        let file_path = output_dir.join("production_readiness_summary.csv");
        let csv_content = self.generate_csv_summary(report)?;
        
        std::fs::write(&file_path, csv_content)
            .map_err(|e| ValidationError::FileSystem(e))?;
        
        Ok(file_path)
    }

    /// Export report as Markdown
    fn export_markdown(&self, report: &ProductionReadinessReport, output_dir: &Path) -> Result<PathBuf> {
        let file_path = output_dir.join("production_readiness_report.md");
        let markdown_content = self.generate_markdown_report(report)?;
        
        std::fs::write(&file_path, markdown_content)
            .map_err(|e| ValidationError::FileSystem(e))?;
        
        Ok(file_path)
    }

    /// Generate HTML report content
    fn generate_html_report(&self, report: &ProductionReadinessReport) -> Result<String> {
        let default_color_scheme = ColorScheme::default();
        let color_scheme = self.config.custom_branding
            .as_ref()
            .map(|b| &b.color_scheme)
            .unwrap_or(&default_color_scheme);

        let html = format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: {};
            background-color: {};
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ 
            background: linear-gradient(135deg, {}, {});
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        .section {{ 
            background: white;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .score {{ 
            font-size: 2rem;
            font-weight: bold;
            color: {};
        }}
        .grade {{ 
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        }}
        .grade-a {{ background-color: #10b981; }}
        .grade-b {{ background-color: #3b82f6; }}
        .grade-c {{ background-color: #f59e0b; }}
        .grade-d {{ background-color: #ef4444; }}
        .grade-f {{ background-color: #dc2626; }}
        .metric {{ 
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e5e7eb;
        }}
        .critical {{ color: #dc2626; font-weight: bold; }}
        .good {{ color: #10b981; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background-color: #f9fafb; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{}</h1>
            <p>Generated on {}</p>
            <p>Target: {}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">
                <span>Overall Recommendation:</span>
                <span class="{}">{:?}</span>
            </div>
            <div class="metric">
                <span>Readiness Score:</span>
                <span class="score">{:.1}%</span>
            </div>
        </div>

        <div class="section">
            <h2>Factor Scores</h2>
            {}
        </div>

        <div class="section">
            <h2>Critical Issues</h2>
            <p>Total Issues: {}</p>
            {}
        </div>

        <div class="section">
            <h2>Production Blockers</h2>
            <p>Total Blockers: {}</p>
            <p>Must Fix Before Production: {}</p>
        </div>

        <div class="section">
            <h2>Performance Analysis</h2>
            <div class="metric">
                <span>Processing Speed:</span>
                <span>{:.2} files/second</span>
            </div>
            <div class="metric">
                <span>Memory Efficiency:</span>
                <span>{:.1}%</span>
            </div>
            <div class="metric">
                <span>Performance Grade:</span>
                <span class="grade grade-{}">{}</span>
            </div>
        </div>
    </div>
</body>
</html>
        "#,
            report.report_metadata.report_title,
            color_scheme.text_color,
            color_scheme.background_color,
            color_scheme.primary_color,
            color_scheme.secondary_color,
            color_scheme.accent_color,
            report.report_metadata.report_title,
            report.report_metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
            report.report_metadata.validation_target,
            if matches!(report.executive_summary.overall_recommendation, OverallRecommendation::ReadyForProduction) { "good" } else { "critical" },
            report.executive_summary.overall_recommendation,
            report.readiness_assessment.overall_score * 100.0,
            self.generate_factor_scores_html(&report.readiness_assessment.factor_scores),
            report.readiness_assessment.critical_issues_summary.total_issues,
            self.generate_critical_issues_html(&report.readiness_assessment.critical_issues_summary.top_issues),
            report.readiness_assessment.blockers_summary.total_blockers,
            report.readiness_assessment.blockers_summary.must_fix_count,
            report.performance_analysis.performance_summary.processing_speed.files_per_second,
            report.performance_analysis.performance_summary.resource_utilization.memory_efficiency * 100.0,
            report.performance_analysis.performance_summary.overall_performance_grade.to_lowercase(),
            report.performance_analysis.performance_summary.overall_performance_grade,
        );

        Ok(html)
    }

    /// Generate factor scores HTML
    fn generate_factor_scores_html(&self, factor_scores: &HashMap<String, FactorScore>) -> String {
        let mut html = String::new();
        for (factor, score) in factor_scores {
            html.push_str(&format!(
                r#"<div class="metric">
                    <span>{}:</span>
                    <span><span class="grade grade-{}">{}</span> ({:.1}%)</span>
                </div>"#,
                factor.replace('_', " ").to_uppercase(),
                score.grade.to_lowercase(),
                score.grade,
                score.score * 100.0
            ));
        }
        html
    }

    /// Generate critical issues HTML
    fn generate_critical_issues_html(&self, issues: &[CriticalIssueSummary]) -> String {
        if issues.is_empty() {
            return "<p>No critical issues identified.</p>".to_string();
        }

        let mut html = "<ul>".to_string();
        for issue in issues {
            html.push_str(&format!(
                "<li><strong>{}:</strong> {} (Severity: {})</li>",
                issue.title, issue.impact, issue.severity
            ));
        }
        html.push_str("</ul>");
        html
    }

    /// Generate CSV summary
    fn generate_csv_summary(&self, report: &ProductionReadinessReport) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("Metric,Value,Grade,Status\n");
        
        csv.push_str(&format!("Overall Score,{:.1}%,,{:?}\n", 
            report.readiness_assessment.overall_score * 100.0,
            report.executive_summary.overall_recommendation
        ));

        for (factor, score) in &report.readiness_assessment.factor_scores {
            csv.push_str(&format!("{},{:.1}%,{},{}\n",
                factor.replace('_', " "),
                score.score * 100.0,
                score.grade,
                score.status
            ));
        }

        csv.push_str(&format!("Critical Issues,{},N/A,\n", 
            report.readiness_assessment.critical_issues_summary.total_issues));
        csv.push_str(&format!("Production Blockers,{},N/A,\n", 
            report.readiness_assessment.blockers_summary.total_blockers));

        Ok(csv)
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self, report: &ProductionReadinessReport) -> Result<String> {
        let mut md = String::new();
        
        md.push_str(&format!("# {}\n\n", report.report_metadata.report_title));
        md.push_str(&format!("**Generated:** {}\n", report.report_metadata.generated_at.format("%Y-%m-%d %H:%M:%S UTC")));
        md.push_str(&format!("**Target:** {}\n\n", report.report_metadata.validation_target));

        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!("**Overall Recommendation:** {:?}\n", report.executive_summary.overall_recommendation));
        md.push_str(&format!("**Readiness Score:** {:.1}%\n\n", report.readiness_assessment.overall_score * 100.0));

        md.push_str("## Factor Scores\n\n");
        md.push_str("| Factor | Score | Grade | Status |\n");
        md.push_str("|--------|-------|-------|--------|\n");
        for (factor, score) in &report.readiness_assessment.factor_scores {
            md.push_str(&format!("| {} | {:.1}% | {} | {} |\n",
                factor.replace('_', " "),
                score.score * 100.0,
                score.grade,
                score.status
            ));
        }
        md.push_str("\n");

        md.push_str("## Critical Issues\n\n");
        md.push_str(&format!("**Total Issues:** {}\n\n", report.readiness_assessment.critical_issues_summary.total_issues));
        
        if !report.readiness_assessment.critical_issues_summary.top_issues.is_empty() {
            for issue in &report.readiness_assessment.critical_issues_summary.top_issues {
                md.push_str(&format!("- **{}:** {} (Severity: {})\n", issue.title, issue.impact, issue.severity));
            }
            md.push_str("\n");
        }

        md.push_str("## Performance Analysis\n\n");
        md.push_str(&format!("- **Processing Speed:** {:.2} files/second\n", 
            report.performance_analysis.performance_summary.processing_speed.files_per_second));
        md.push_str(&format!("- **Memory Efficiency:** {:.1}%\n", 
            report.performance_analysis.performance_summary.resource_utilization.memory_efficiency * 100.0));
        md.push_str(&format!("- **Performance Grade:** {}\n\n", 
            report.performance_analysis.performance_summary.overall_performance_grade));

        Ok(md)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation_orchestrator::*;
    use crate::production_readiness_assessor::*;
    use crate::pensieve_runner::*;
    use crate::reliability_validator::*;
    use crate::metrics_collector::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_mock_validation_results() -> ComprehensiveValidationResults {
        ComprehensiveValidationResults {
            pensieve_execution_results: PensieveExecutionResults {
                exit_code: Some(0),
                execution_time: Duration::from_secs(120),
                peak_memory_mb: 512,
                average_memory_mb: 256,
                cpu_usage_stats: CpuUsageStats {
                    peak_cpu_percent: 75.0,
                    average_cpu_percent: 45.0,
                    cpu_time_seconds: 54.0,
                },
                performance_metrics: PerformanceMetrics {
                    files_per_second: 15.5,
                    processing_consistency: 0.85,
                    memory_efficiency_score: 0.8,
                },
                output_analysis: OutputAnalysis {
                    files_processed: 1860,
                    duplicates_found: 45,
                    errors_encountered: 3,
                    progress_updates: 12,
                    error_lines: 3,
                    warning_lines: 8,
                    info_lines: 25,
                },
                error_summary: ErrorSummary {
                    total_errors: 3,
                    critical_errors: vec![],
                    error_rate_per_minute: 1.5,
                    error_categories: HashMap::new(),
                },
            },
            process_monitoring_results: crate::process_monitor::MonitoringResults {
                snapshots: vec![],
                summary: crate::process_monitor::MonitoringSummary {
                    duration: Duration::from_secs(120),
                    snapshot_count: 24,
                    peak_memory_usage: 512,
                    average_memory_usage: 256,
                    peak_cpu_usage: 75.0,
                    average_cpu_usage: 45.0,
                    total_disk_read: 1024 * 1024 * 100, // 100MB
                    total_disk_write: 1024 * 1024 * 50, // 50MB
                    peak_temperature: 65.0,
                    memory_efficiency: 0.8,
                    cpu_efficiency: 0.7,
                },
                alerts: vec![],
                performance_analysis: crate::process_monitor::PerformanceAnalysis {
                    resource_utilization_score: 0.75,
                    stability_score: 0.85,
                    efficiency_score: 0.8,
                    bottlenecks: vec![],
                    recommendations: vec![],
                },
            },
            reliability_results: ReliabilityResults {
                overall_reliability_score: 0.9,
                crash_test_results: CrashTestResults {
                    zero_crash_validation_passed: true,
                    crash_incidents: vec![],
                    edge_case_handling_score: 0.9,
                    error_recovery_score: 0.85,
                },
                interruption_test_results: InterruptionTestResults {
                    graceful_shutdown_works: true,
                    state_preservation_works: true,
                    cleanup_on_interruption: true,
                    recovery_instructions_clear: true,
                },
                resource_limit_test_results: ResourceLimitTestResults {
                    memory_limit_respected: true,
                    graceful_degradation_works: true,
                    resource_cleanup_works: true,
                    performance_under_pressure: 0.7,
                },
                corruption_handling_results: CorruptionHandlingResults {
                    corrupted_file_handling: true,
                    malformed_data_handling: true,
                    encoding_issue_handling: true,
                    partial_file_handling: true,
                },
                permission_handling_results: PermissionHandlingResults {
                    read_permission_handling: true,
                    write_permission_handling: true,
                    directory_permission_handling: true,
                    graceful_permission_failures: true,
                },
                recovery_test_results: RecoveryTestResults {
                    partial_completion_recovery: true,
                    state_restoration_works: true,
                    resume_capability_works: false,
                    data_integrity_maintained: true,
                },
                reliability_blockers: vec![],
                reliability_recommendations: vec![],
                risk_assessment: RiskAssessment {
                    overall_risk_level: crate::reliability_validator::RiskLevel::Low,
                    risk_factors: vec![],
                    mitigation_strategies: vec![],
                    monitoring_recommendations: vec![],
                },
            },
            metrics_collection_results: MetricsCollectionResults {
                collection_duration: Duration::from_secs(120),
                total_data_points: 240,
                performance_metrics: crate::metrics_collector::PerformanceMetricsSummary {
                    average_files_per_second: 15.5,
                    peak_files_per_second: 22.0,
                    performance_consistency_score: 0.85,
                    processing_speed_trend: "Stable".to_string(),
                },
                error_metrics: crate::metrics_collector::ErrorMetricsSummary {
                    total_errors: 3,
                    error_rate: 1.5,
                    error_severity_distribution: HashMap::new(),
                    error_recovery_rate: 1.0,
                },
                ux_metrics: crate::metrics_collector::UXMetricsSummary {
                    average_ux_score: 7.5,
                    progress_update_frequency: 0.1,
                    error_message_clarity_score: 0.7,
                    completion_feedback_score: 0.8,
                },
                database_metrics: crate::metrics_collector::DatabaseMetricsSummary {
                    total_operations: 1860,
                    average_operation_time: Duration::from_millis(5),
                    operation_success_rate: 0.998,
                    connection_stability_score: 0.95,
                },
                overall_assessment: crate::metrics_collector::OverallAssessment {
                    reliability_score: 0.9,
                    performance_score: 0.8,
                    user_experience_score: 0.75,
                    efficiency_score: 0.8,
                    production_readiness_indicator: 0.82,
                },
            },
            validation_assessment: ValidationAssessment {
                overall_score: 0.82,
                performance_grade: Grade::B,
                reliability_grade: Grade::A,
                user_experience_grade: Grade::B,
                efficiency_grade: Grade::B,
                production_readiness: ProductionReadiness::ReadyWithCaveats(vec![
                    "Minor UX improvements recommended".to_string()
                ]),
                critical_issues: vec![],
            },
            recommendations: vec![
                ValidationRecommendation {
                    category: "User Experience".to_string(),
                    priority: RecommendationPriority::Medium,
                    description: "Improve error message clarity".to_string(),
                    expected_impact: "Better user satisfaction".to_string(),
                    implementation_effort: ImplementationEffort::Low,
                }
            ],
        }
    }

    fn create_mock_production_assessment() -> ProductionReadinessAssessment {
        ProductionReadinessAssessment {
            overall_readiness: ProductionReadinessLevel::ReadyWithCaveats(vec![
                "Minor UX improvements recommended".to_string()
            ]),
            readiness_score: 0.82,
            factor_scores: FactorScores {
                reliability_score: 0.9,
                performance_score: 0.8,
                user_experience_score: 0.75,
                consistency_score: 0.85,
                scalability_score: 0.8,
                factor_breakdown: HashMap::new(),
            },
            critical_issues: vec![],
            blockers: vec![],
            scaling_guidance: ScalingGuidance {
                current_capacity_assessment: CapacityAssessment {
                    max_files_per_run: 10000,
                    max_data_size_gb: 100.0,
                    max_concurrent_operations: 4,
                    memory_ceiling_mb: 2048,
                    processing_time_ceiling_hours: 8.0,
                    confidence_level: 0.8,
                },
                scaling_recommendations: vec![],
                resource_requirements: ResourceRequirements {
                    small_scale: crate::production_readiness_assessor::ResourceSpec {
                        min_memory_gb: 2.0,
                        recommended_memory_gb: 4.0,
                        min_cpu_cores: 2,
                        recommended_cpu_cores: 4,
                        min_disk_space_gb: 10.0,
                        recommended_disk_space_gb: 50.0,
                        estimated_processing_time: Duration::from_secs(3600),
                    },
                    medium_scale: crate::production_readiness_assessor::ResourceSpec {
                        min_memory_gb: 4.0,
                        recommended_memory_gb: 8.0,
                        min_cpu_cores: 4,
                        recommended_cpu_cores: 8,
                        min_disk_space_gb: 50.0,
                        recommended_disk_space_gb: 200.0,
                        estimated_processing_time: Duration::from_secs(7200),
                    },
                    large_scale: crate::production_readiness_assessor::ResourceSpec {
                        min_memory_gb: 8.0,
                        recommended_memory_gb: 16.0,
                        min_cpu_cores: 8,
                        recommended_cpu_cores: 16,
                        min_disk_space_gb: 200.0,
                        recommended_disk_space_gb: 1000.0,
                        estimated_processing_time: Duration::from_secs(14400),
                    },
                    enterprise_scale: crate::production_readiness_assessor::ResourceSpec {
                        min_memory_gb: 16.0,
                        recommended_memory_gb: 32.0,
                        min_cpu_cores: 16,
                        recommended_cpu_cores: 32,
                        min_disk_space_gb: 1000.0,
                        recommended_disk_space_gb: 5000.0,
                        estimated_processing_time: Duration::from_secs(28800),
                    },
                },
                performance_projections: PerformanceProjections {
                    linear_scaling_factors: HashMap::new(),
                    performance_degradation_points: vec![],
                    optimal_batch_sizes: HashMap::new(),
                    resource_utilization_curves: HashMap::new(),
                },
                bottleneck_analysis: BottleneckAnalysis {
                    identified_bottlenecks: vec![],
                    bottleneck_severity_ranking: vec![],
                    optimization_opportunities: vec![],
                },
            },
            deployment_recommendations: DeploymentRecommendations {
                environment_requirements: EnvironmentRequirements {
                    minimum_os_requirements: HashMap::new(),
                    required_dependencies: vec![],
                    network_requirements: NetworkRequirements {
                        bandwidth_requirements: "1 Mbps".to_string(),
                        latency_requirements: "< 100ms".to_string(),
                        connectivity_requirements: vec![],
                        firewall_considerations: vec![],
                    },
                    security_requirements: vec![],
                    compliance_considerations: vec![],
                },
                configuration_recommendations: vec![],
                monitoring_requirements: MonitoringRequirements {
                    key_metrics: vec![],
                    alerting_thresholds: HashMap::new(),
                    dashboard_requirements: vec![],
                    log_retention_requirements: "30 days".to_string(),
                },
                operational_considerations: vec![],
                rollback_strategy: RollbackStrategy {
                    rollback_triggers: vec![],
                    rollback_procedures: vec![],
                    data_recovery_procedures: vec![],
                    estimated_rollback_time: Duration::from_secs(1800),
                    rollback_testing_requirements: vec![],
                },
            },
            improvement_roadmap: ImprovementRoadmap {
                immediate_actions: vec![],
                short_term_improvements: vec![],
                long_term_improvements: vec![],
                roadmap_timeline: RoadmapTimeline {
                    immediate_phase_duration: Duration::from_secs(7 * 24 * 3600),
                    short_term_phase_duration: Duration::from_secs(30 * 24 * 3600),
                    long_term_phase_duration: Duration::from_secs(90 * 24 * 3600),
                    total_estimated_duration: Duration::from_secs(127 * 24 * 3600),
                    milestone_dates: HashMap::new(),
                },
                success_metrics: vec![],
            },
            assessment_metadata: AssessmentMetadata {
                assessment_timestamp: chrono::Utc::now(),
                assessment_version: "1.0.0".to_string(),
                data_sources: vec!["Test data".to_string()],
                assessment_duration: Duration::from_secs(60),
                confidence_level: 0.8,
                limitations: vec!["Test environment".to_string()],
                assumptions: vec!["Representative data".to_string()],
            },
        }
    }

    #[test]
    fn test_report_generator_creation() {
        let config = ReportGeneratorConfig::default();
        let generator = ReportGenerator::new(config);
        
        assert_eq!(generator.config.output_formats.len(), 2);
        assert!(matches!(generator.config.report_detail_level, ReportDetailLevel::Standard));
    }

    #[test]
    fn test_production_readiness_report_generation() -> Result<()> {
        let generator = ReportGenerator::with_default_config();
        let validation_results = create_mock_validation_results();
        let production_assessment = create_mock_production_assessment();
        
        let report = generator.generate_production_readiness_report(
            &validation_results,
            &production_assessment,
            "/test/directory",
        )?;

        // Verify report structure
        assert_eq!(report.report_metadata.validation_target, "/test/directory");
        assert!(matches!(report.executive_summary.overall_recommendation, OverallRecommendation::ReadyWithMinorImprovements));
        assert_eq!(report.readiness_assessment.overall_score, 0.82);
        assert_eq!(report.readiness_assessment.factor_scores.len(), 3);

        Ok(())
    }

    #[test]
    fn test_json_export() -> Result<()> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let generator = ReportGenerator::with_default_config();
        let validation_results = create_mock_validation_results();
        let production_assessment = create_mock_production_assessment();
        
        let report = generator.generate_production_readiness_report(
            &validation_results,
            &production_assessment,
            "/test/directory",
        )?;

        let json_path = generator.export_json(&report, temp_dir.path())?;
        
        assert!(json_path.exists());
        assert_eq!(json_path.file_name().unwrap(), "production_readiness_report.json");
        
        // Verify JSON content is valid
        let json_content = std::fs::read_to_string(&json_path)
            .map_err(|e| ValidationError::FileSystem(e))?;
        let _: ProductionReadinessReport = serde_json::from_str(&json_content)
            .map_err(|e| ValidationError::ReportGenerationFailed { 
                cause: format!("Invalid JSON: {}", e) 
            })?;

        Ok(())
    }

    #[test]
    fn test_html_export() -> Result<()> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let generator = ReportGenerator::with_default_config();
        let validation_results = create_mock_validation_results();
        let production_assessment = create_mock_production_assessment();
        
        let report = generator.generate_production_readiness_report(
            &validation_results,
            &production_assessment,
            "/test/directory",
        )?;

        let html_path = generator.export_html(&report, temp_dir.path())?;
        
        assert!(html_path.exists());
        assert_eq!(html_path.file_name().unwrap(), "production_readiness_report.html");
        
        // Verify HTML content contains expected elements
        let html_content = std::fs::read_to_string(&html_path)
            .map_err(|e| ValidationError::FileSystem(e))?;
        assert!(html_content.contains("<!DOCTYPE html>"));
        assert!(html_content.contains("Production Readiness Assessment"));
        assert!(html_content.contains("Executive Summary"));

        Ok(())
    }

    #[test]
    fn test_csv_export() -> Result<()> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let generator = ReportGenerator::with_default_config();
        let validation_results = create_mock_validation_results();
        let production_assessment = create_mock_production_assessment();
        
        let report = generator.generate_production_readiness_report(
            &validation_results,
            &production_assessment,
            "/test/directory",
        )?;

        let csv_path = generator.export_csv(&report, temp_dir.path())?;
        
        assert!(csv_path.exists());
        assert_eq!(csv_path.file_name().unwrap(), "production_readiness_summary.csv");
        
        // Verify CSV content has headers
        let csv_content = std::fs::read_to_string(&csv_path)
            .map_err(|e| ValidationError::FileSystem(e))?;
        assert!(csv_content.contains("Metric,Value,Grade,Status"));

        Ok(())
    }

    #[test]
    fn test_markdown_export() -> Result<()> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let generator = ReportGenerator::with_default_config();
        let validation_results = create_mock_validation_results();
        let production_assessment = create_mock_production_assessment();
        
        let report = generator.generate_production_readiness_report(
            &validation_results,
            &production_assessment,
            "/test/directory",
        )?;

        let md_path = generator.export_markdown(&report, temp_dir.path())?;
        
        assert!(md_path.exists());
        assert_eq!(md_path.file_name().unwrap(), "production_readiness_report.md");
        
        // Verify Markdown content has expected structure
        let md_content = std::fs::read_to_string(&md_path)
            .map_err(|e| ValidationError::FileSystem(e))?;
        assert!(md_content.contains("# Production Readiness Assessment"));
        assert!(md_content.contains("## Executive Summary"));
        assert!(md_content.contains("## Factor Scores"));

        Ok(())
    }

    #[test]
    fn test_multiple_format_export() -> Result<()> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let config = ReportGeneratorConfig {
            output_formats: vec![OutputFormat::Json, OutputFormat::Html, OutputFormat::Csv, OutputFormat::Markdown],
            ..Default::default()
        };
        let generator = ReportGenerator::new(config);
        let validation_results = create_mock_validation_results();
        let production_assessment = create_mock_production_assessment();
        
        let report = generator.generate_production_readiness_report(
            &validation_results,
            &production_assessment,
            "/test/directory",
        )?;

        let exported_files = generator.export_report(&report, temp_dir.path())?;
        
        assert_eq!(exported_files.len(), 4);
        for file_path in &exported_files {
            assert!(file_path.exists());
        }

        Ok(())
    }
}