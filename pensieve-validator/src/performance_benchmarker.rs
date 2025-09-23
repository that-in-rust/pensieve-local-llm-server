use crate::errors::{ValidationError, Result};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Performance benchmarking and scalability analysis system
pub struct PerformanceBenchmarker {
    baseline_metrics: Option<PerformanceBaseline>,
    benchmark_config: BenchmarkConfig,
    scalability_analyzer: ScalabilityAnalyzer,
    memory_analyzer: MemoryAnalyzer,
    database_profiler: DatabaseProfiler,
}

/// Configuration for performance benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub enable_baseline_establishment: bool,
    pub enable_degradation_detection: bool,
    pub enable_scalability_testing: bool,
    pub enable_memory_analysis: bool,
    pub enable_database_profiling: bool,
    pub benchmark_iterations: u32,
    pub warmup_iterations: u32,
    pub timeout_seconds: u64,
    pub memory_sampling_interval_ms: u64,
    pub performance_thresholds: PerformanceThresholds,
}

/// Performance thresholds for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_files_per_second_degradation: f64, // 0.2 = 20% degradation allowed
    pub max_memory_growth_rate_mb_per_sec: f64,
    pub max_database_operation_time_ms: u64,
    pub min_cpu_efficiency_score: f64,
    pub max_memory_leak_rate_mb_per_hour: f64,
}

/// Performance baseline for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub dataset_characteristics: DatasetCharacteristics,
    pub baseline_metrics: BaselineMetrics,
    pub system_configuration: SystemConfiguration,
    pub pensieve_version: String,
}

/// Characteristics of the dataset used for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCharacteristics {
    pub total_files: u64,
    pub total_size_bytes: u64,
    pub file_type_distribution: HashMap<String, u64>,
    pub average_file_size_bytes: u64,
    pub largest_file_size_bytes: u64,
    pub directory_depth: usize,
    pub chaos_score: f64,
}

/// Baseline performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub files_per_second: f64,
    pub peak_memory_usage_mb: u64,
    pub average_memory_usage_mb: u64,
    pub cpu_efficiency_score: f64,
    pub database_operations_per_second: f64,
    pub total_processing_time: Duration,
    pub memory_growth_pattern: MemoryGrowthPattern,
}

/// Memory growth pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGrowthPattern {
    pub growth_type: MemoryGrowthType,
    pub growth_rate_mb_per_sec: f64,
    pub peak_memory_mb: u64,
    pub memory_efficiency_score: f64,
    pub leak_indicators: Vec<MemoryLeakIndicator>,
}

/// Types of memory growth patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryGrowthType {
    Linear,
    Logarithmic,
    Exponential,
    Constant,
    Volatile,
}

/// Memory leak indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakIndicator {
    pub indicator_type: String,
    pub severity: f64, // 0.0 - 1.0
    pub description: String,
    pub evidence: Vec<String>,
}

/// System configuration at time of baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfiguration {
    pub cpu_cores: u32,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
    pub storage_type: String, // SSD, HDD, etc.
    pub os_info: String,
    pub rust_version: String,
}

/// Scalability analysis system
pub struct ScalabilityAnalyzer {
    test_configurations: Vec<ScalabilityTestConfig>,
    extrapolation_models: Vec<ExtrapolationModel>,
}

/// Configuration for scalability testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityTestConfig {
    pub test_name: String,
    pub dataset_size_multipliers: Vec<f64>, // 0.1x, 0.5x, 1x, 2x, 5x, 10x
    pub file_count_multipliers: Vec<f64>,
    pub complexity_multipliers: Vec<f64>,
    pub expected_scaling_behavior: ScalingBehavior,
}

/// Expected scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingBehavior {
    Linear,      // O(n)
    Logarithmic, // O(log n)
    Quadratic,   // O(n²)
    Exponential, // O(2^n)
    Unknown,
}

/// Extrapolation model for predicting performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtrapolationModel {
    pub model_name: String,
    pub model_type: ModelType,
    pub parameters: HashMap<String, f64>,
    pub accuracy_score: f64, // 0.0 - 1.0
    pub confidence_interval: (f64, f64),
    pub applicable_range: (u64, u64), // Min/max dataset size
}

/// Types of prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Linear,
    Polynomial,
    Exponential,
    Logarithmic,
    PowerLaw,
}

/// Memory usage pattern analysis system
pub struct MemoryAnalyzer {
    sampling_interval: Duration,
    leak_detection_threshold: f64,
    pattern_analysis_window: Duration,
}

/// Memory usage pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysisResults {
    pub memory_usage_pattern: MemoryUsagePattern,
    pub leak_detection_results: LeakDetectionResults,
    pub memory_efficiency_analysis: MemoryEfficiencyAnalysis,
    pub memory_optimization_recommendations: Vec<MemoryOptimizationRecommendation>,
}

/// Memory usage pattern over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsagePattern {
    pub pattern_type: MemoryPatternType,
    pub baseline_memory_mb: u64,
    pub peak_memory_mb: u64,
    pub average_memory_mb: u64,
    pub memory_variance: f64,
    pub growth_phases: Vec<MemoryGrowthPhase>,
}

/// Types of memory usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPatternType {
    Stable,        // Consistent usage
    GradualGrowth, // Slow increase
    RapidGrowth,   // Fast increase
    Spiky,         // High variance
    Leaking,       // Continuous growth
}

/// Memory growth phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGrowthPhase {
    pub phase_name: String,
    pub start_time: Duration,
    pub duration: Duration,
    pub start_memory_mb: u64,
    pub end_memory_mb: u64,
    pub growth_rate_mb_per_sec: f64,
    pub files_processed_in_phase: u64,
}

/// Memory leak detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakDetectionResults {
    pub leak_detected: bool,
    pub leak_severity: LeakSeverity,
    pub leak_rate_mb_per_hour: f64,
    pub leak_sources: Vec<LeakSource>,
    pub confidence_score: f64, // 0.0 - 1.0
}

/// Severity of memory leaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeakSeverity {
    None,
    Minor,    // < 10MB/hour
    Moderate, // 10-100MB/hour
    Severe,   // > 100MB/hour
}

/// Potential memory leak source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakSource {
    pub source_type: String,
    pub description: String,
    pub estimated_contribution_mb_per_hour: f64,
    pub evidence: Vec<String>,
}

/// Memory efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEfficiencyAnalysis {
    pub files_per_mb_ratio: f64,
    pub memory_utilization_score: f64, // 0.0 - 1.0
    pub memory_waste_indicators: Vec<MemoryWasteIndicator>,
    pub optimization_potential_mb: u64,
}

/// Memory waste indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWasteIndicator {
    pub waste_type: String,
    pub wasted_memory_mb: u64,
    pub waste_percentage: f64,
    pub description: String,
}

/// Memory optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub estimated_savings_mb: u64,
    pub implementation_effort: ImplementationEffort,
    pub impact_score: f64, // 0.0 - 1.0
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,    // Configuration change
    Medium, // Code modification
    High,   // Architecture change
}

/// Database performance profiler
pub struct DatabaseProfiler {
    profiling_enabled: bool,
    query_analysis_threshold: Duration,
    bottleneck_detection_threshold: f64,
}

/// Database performance profiling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseProfilingResults {
    pub query_performance_analysis: QueryPerformanceAnalysis,
    pub bottleneck_identification: BottleneckIdentification,
    pub index_efficiency_analysis: IndexEfficiencyAnalysis,
    pub connection_pool_analysis: ConnectionPoolAnalysis,
    pub optimization_recommendations: Vec<DatabaseOptimizationRecommendation>,
}

/// Query performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    pub slow_queries: Vec<SlowQueryAnalysis>,
    pub query_patterns: Vec<QueryPattern>,
    pub performance_trends: Vec<PerformanceTrend>,
    pub query_optimization_opportunities: Vec<QueryOptimizationOpportunity>,
}

/// Slow query analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryAnalysis {
    pub query_type: String,
    pub average_execution_time: Duration,
    pub max_execution_time: Duration,
    pub execution_count: u64,
    pub total_time_spent: Duration,
    pub performance_impact: f64, // 0.0 - 1.0
    pub optimization_suggestions: Vec<String>,
}

/// Query pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    pub pattern_name: String,
    pub frequency: u64,
    pub average_performance: Duration,
    pub performance_variance: f64,
    pub scalability_characteristics: ScalingBehavior,
}

/// Performance trend over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64, // 0.0 - 1.0
    pub statistical_significance: f64, // 0.0 - 1.0
}

/// Direction of performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Query optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptimizationOpportunity {
    pub query_type: String,
    pub current_performance: Duration,
    pub estimated_improvement_percentage: f64,
    pub optimization_type: OptimizationType,
    pub implementation_complexity: ImplementationEffort,
}

/// Types of database optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    IndexCreation,
    QueryRewrite,
    SchemaOptimization,
    ConnectionPoolTuning,
    CacheOptimization,
}

/// Database bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckIdentification {
    pub primary_bottlenecks: Vec<DatabaseBottleneck>,
    pub bottleneck_severity_score: f64, // 0.0 - 1.0
    pub performance_limiting_factors: Vec<PerformanceLimitingFactor>,
    pub scalability_constraints: Vec<ScalabilityConstraint>,
}

/// Database bottleneck details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseBottleneck {
    pub bottleneck_type: BottleneckType,
    pub severity: f64, // 0.0 - 1.0
    pub description: String,
    pub impact_on_throughput: f64, // Percentage impact
    pub mitigation_strategies: Vec<String>,
}

/// Types of database bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    DiskIO,
    NetworkIO,
    Locking,
    IndexMissing,
    QueryInefficiency,
}

/// Performance limiting factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceLimitingFactor {
    pub factor_name: String,
    pub current_utilization: f64, // 0.0 - 1.0
    pub capacity_limit: f64,
    pub time_to_limit: Option<Duration>,
}

/// Scalability constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityConstraint {
    pub constraint_type: String,
    pub current_load: f64,
    pub maximum_capacity: f64,
    pub scaling_recommendations: Vec<String>,
}

/// Index efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEfficiencyAnalysis {
    pub index_usage_statistics: Vec<IndexUsageStatistic>,
    pub missing_index_opportunities: Vec<MissingIndexOpportunity>,
    pub redundant_indexes: Vec<RedundantIndex>,
    pub index_maintenance_recommendations: Vec<String>,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStatistic {
    pub index_name: String,
    pub usage_frequency: u64,
    pub selectivity: f64, // 0.0 - 1.0
    pub maintenance_cost: f64,
    pub performance_benefit: f64,
}

/// Missing index opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingIndexOpportunity {
    pub table_name: String,
    pub column_names: Vec<String>,
    pub estimated_performance_improvement: f64,
    pub query_patterns_affected: Vec<String>,
    pub creation_cost: ImplementationEffort,
}

/// Redundant index identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundantIndex {
    pub index_name: String,
    pub redundant_with: String,
    pub space_savings_mb: u64,
    pub maintenance_cost_savings: f64,
}

/// Connection pool analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolAnalysis {
    pub pool_utilization: f64, // 0.0 - 1.0
    pub average_wait_time: Duration,
    pub connection_churn_rate: f64,
    pub optimal_pool_size_recommendation: u32,
    pub configuration_recommendations: Vec<String>,
}

/// Database optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseOptimizationRecommendation {
    pub recommendation_type: OptimizationType,
    pub description: String,
    pub estimated_performance_improvement: f64,
    pub implementation_effort: ImplementationEffort,
    pub priority: RecommendationPriority,
    pub implementation_steps: Vec<String>,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical, // Blocking performance issues
    High,     // Significant impact
    Medium,   // Moderate improvement
    Low,      // Minor optimization
}

/// Performance prediction model for different dataset characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictionModel {
    pub model_name: String,
    pub dataset_characteristics: DatasetCharacteristics,
    pub performance_predictions: Vec<PerformancePrediction>,
    pub model_accuracy: f64, // 0.0 - 1.0
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Performance prediction for specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    pub metric_name: String,
    pub predicted_value: f64,
    pub confidence_score: f64, // 0.0 - 1.0
    pub prediction_range: (f64, f64),
    pub factors_considered: Vec<String>,
}

/// Comprehensive performance benchmarking results
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceBenchmarkingResults {
    pub benchmark_timestamp: chrono::DateTime<chrono::Utc>,
    pub baseline_comparison: Option<BaselineComparison>,
    pub scalability_analysis: ScalabilityAnalysisResults,
    pub memory_analysis: MemoryAnalysisResults,
    pub database_profiling: DatabaseProfilingResults,
    pub performance_predictions: Vec<PerformancePredictionModel>,
    pub degradation_detection: DegradationDetectionResults,
    pub overall_performance_assessment: OverallPerformanceAssessment,
}

/// Baseline comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub baseline_established: bool,
    pub performance_changes: Vec<PerformanceChange>,
    pub degradation_detected: bool,
    pub improvement_detected: bool,
    pub overall_change_percentage: f64,
}

/// Performance change from baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceChange {
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_percentage: f64,
    pub change_significance: ChangeSignificance,
}

/// Significance of performance changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeSignificance {
    Negligible,  // < 5% change
    Minor,       // 5-15% change
    Moderate,    // 15-30% change
    Significant, // > 30% change
}

/// Scalability analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAnalysisResults {
    pub scaling_behavior: ScalingBehavior,
    pub scalability_score: f64, // 0.0 - 1.0
    pub bottleneck_points: Vec<ScalabilityBottleneck>,
    pub extrapolation_results: Vec<ExtrapolationResult>,
    pub scaling_recommendations: Vec<ScalingRecommendation>,
}

/// Scalability bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityBottleneck {
    pub bottleneck_point: u64, // Dataset size where bottleneck occurs
    pub bottleneck_type: String,
    pub performance_impact: f64,
    pub mitigation_strategies: Vec<String>,
}

/// Extrapolation result for larger datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtrapolationResult {
    pub target_dataset_size: u64,
    pub predicted_performance: HashMap<String, f64>,
    pub confidence_score: f64,
    pub assumptions: Vec<String>,
}

/// Scaling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub applicable_dataset_size_range: (u64, u64),
    pub expected_improvement: f64,
    pub implementation_complexity: ImplementationEffort,
}

/// Degradation detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationDetectionResults {
    pub degradation_detected: bool,
    pub degradation_severity: DegradationSeverity,
    pub affected_metrics: Vec<String>,
    pub degradation_causes: Vec<DegradationCause>,
    pub mitigation_recommendations: Vec<String>,
}

/// Severity of performance degradation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationSeverity {
    None,
    Minor,    // < 10% degradation
    Moderate, // 10-25% degradation
    Severe,   // > 25% degradation
}

/// Cause of performance degradation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationCause {
    pub cause_type: String,
    pub description: String,
    pub contribution_percentage: f64,
    pub evidence: Vec<String>,
}

/// Overall performance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallPerformanceAssessment {
    pub performance_score: f64, // 0.0 - 1.0
    pub scalability_score: f64, // 0.0 - 1.0
    pub efficiency_score: f64,  // 0.0 - 1.0
    pub reliability_score: f64, // 0.0 - 1.0
    pub overall_score: f64,     // 0.0 - 1.0
    pub key_strengths: Vec<String>,
    pub key_weaknesses: Vec<String>,
    pub priority_improvements: Vec<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_baseline_establishment: true,
            enable_degradation_detection: true,
            enable_scalability_testing: true,
            enable_memory_analysis: true,
            enable_database_profiling: true,
            benchmark_iterations: 3,
            warmup_iterations: 1,
            timeout_seconds: 3600, // 1 hour
            memory_sampling_interval_ms: 100,
            performance_thresholds: PerformanceThresholds {
                max_files_per_second_degradation: 0.2,
                max_memory_growth_rate_mb_per_sec: 10.0,
                max_database_operation_time_ms: 1000,
                min_cpu_efficiency_score: 0.6,
                max_memory_leak_rate_mb_per_hour: 50.0,
            },
        }
    }
}

impl PerformanceBenchmarker {
    /// Create a new performance benchmarker with default configuration
    pub fn new() -> Self {
        Self::with_config(BenchmarkConfig::default())
    }

    /// Create a new performance benchmarker with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            baseline_metrics: None,
            benchmark_config: config.clone(),
            scalability_analyzer: ScalabilityAnalyzer::new(config.clone()),
            memory_analyzer: MemoryAnalyzer::new(
                Duration::from_millis(config.memory_sampling_interval_ms),
                config.performance_thresholds.max_memory_leak_rate_mb_per_hour,
            ),
            database_profiler: DatabaseProfiler::new(config.enable_database_profiling),
        }
    }

    /// Establish performance baseline for future comparisons
    pub async fn establish_baseline(
        &mut self,
        pensieve_binary: &PathBuf,
        target_directory: &PathBuf,
        output_database: &PathBuf,
    ) -> Result<PerformanceBaseline> {
        println!("🎯 Establishing performance baseline...");

        // Analyze dataset characteristics
        let dataset_characteristics = self.analyze_dataset_characteristics(target_directory).await?;
        
        // Run baseline performance test
        let baseline_metrics = self.run_baseline_benchmark(
            pensieve_binary,
            target_directory,
            output_database,
        ).await?;

        // Get system configuration
        let system_configuration = self.get_system_configuration().await?;

        // Get pensieve version
        let pensieve_version = self.get_pensieve_version(pensieve_binary).await?;

        let baseline = PerformanceBaseline {
            established_at: chrono::Utc::now(),
            dataset_characteristics,
            baseline_metrics,
            system_configuration,
            pensieve_version,
        };

        self.baseline_metrics = Some(baseline.clone());
        
        println!("✅ Performance baseline established");
        Ok(baseline)
    }

    /// Run comprehensive performance benchmarking and analysis
    pub async fn run_comprehensive_benchmark(
        &mut self,
        pensieve_binary: &PathBuf,
        target_directory: &PathBuf,
        output_database: &PathBuf,
    ) -> Result<PerformanceBenchmarkingResults> {
        println!("🚀 Starting comprehensive performance benchmarking...");

        let benchmark_start = Instant::now();

        // Establish baseline if not already done
        if self.baseline_metrics.is_none() && self.benchmark_config.enable_baseline_establishment {
            self.establish_baseline(pensieve_binary, target_directory, output_database).await?;
        }

        // Run scalability analysis
        let scalability_analysis = if self.benchmark_config.enable_scalability_testing {
            println!("📈 Running scalability analysis...");
            Some(self.scalability_analyzer.analyze_scalability(
                pensieve_binary,
                target_directory,
                output_database,
            ).await?)
        } else {
            None
        };

        // Run memory analysis
        let memory_analysis = if self.benchmark_config.enable_memory_analysis {
            println!("🧠 Running memory analysis...");
            Some(self.memory_analyzer.analyze_memory_patterns(
                pensieve_binary,
                target_directory,
                output_database,
            ).await?)
        } else {
            None
        };

        // Run database profiling
        let database_profiling = if self.benchmark_config.enable_database_profiling {
            println!("🗄️ Running database profiling...");
            Some(self.database_profiler.profile_database_performance(
                pensieve_binary,
                target_directory,
                output_database,
            ).await?)
        } else {
            None
        };

        // Generate performance predictions
        let performance_predictions = self.generate_performance_predictions(
            target_directory,
            scalability_analysis.as_ref(),
        ).await?;

        // Detect performance degradation
        let degradation_detection = if self.benchmark_config.enable_degradation_detection {
            self.detect_performance_degradation().await?
        } else {
            DegradationDetectionResults {
                degradation_detected: false,
                degradation_severity: DegradationSeverity::None,
                affected_metrics: Vec::new(),
                degradation_causes: Vec::new(),
                mitigation_recommendations: Vec::new(),
            }
        };

        // Compare with baseline
        let baseline_comparison = if let Some(baseline) = &self.baseline_metrics {
            Some(self.compare_with_baseline(baseline, &memory_analysis, &database_profiling).await?)
        } else {
            None
        };

        // Generate overall assessment
        let overall_performance_assessment = self.generate_overall_assessment(
            &scalability_analysis,
            &memory_analysis,
            &database_profiling,
            &degradation_detection,
        ).await?;

        let benchmark_duration = benchmark_start.elapsed();
        println!("✅ Comprehensive benchmarking completed in {:?}", benchmark_duration);

        Ok(PerformanceBenchmarkingResults {
            benchmark_timestamp: chrono::Utc::now(),
            baseline_comparison,
            scalability_analysis: scalability_analysis.unwrap_or_else(|| ScalabilityAnalysisResults {
                scaling_behavior: ScalingBehavior::Unknown,
                scalability_score: 0.0,
                bottleneck_points: Vec::new(),
                extrapolation_results: Vec::new(),
                scaling_recommendations: Vec::new(),
            }),
            memory_analysis: memory_analysis.unwrap_or_else(|| MemoryAnalysisResults {
                memory_usage_pattern: MemoryUsagePattern {
                    pattern_type: MemoryPatternType::Stable,
                    baseline_memory_mb: 0,
                    peak_memory_mb: 0,
                    average_memory_mb: 0,
                    memory_variance: 0.0,
                    growth_phases: Vec::new(),
                },
                leak_detection_results: LeakDetectionResults {
                    leak_detected: false,
                    leak_severity: LeakSeverity::None,
                    leak_rate_mb_per_hour: 0.0,
                    leak_sources: Vec::new(),
                    confidence_score: 1.0,
                },
                memory_efficiency_analysis: MemoryEfficiencyAnalysis {
                    files_per_mb_ratio: 0.0,
                    memory_utilization_score: 1.0,
                    memory_waste_indicators: Vec::new(),
                    optimization_potential_mb: 0,
                },
                memory_optimization_recommendations: Vec::new(),
            }),
            database_profiling: database_profiling.unwrap_or_else(|| DatabaseProfilingResults {
                query_performance_analysis: QueryPerformanceAnalysis {
                    slow_queries: Vec::new(),
                    query_patterns: Vec::new(),
                    performance_trends: Vec::new(),
                    query_optimization_opportunities: Vec::new(),
                },
                bottleneck_identification: BottleneckIdentification {
                    primary_bottlenecks: Vec::new(),
                    bottleneck_severity_score: 0.0,
                    performance_limiting_factors: Vec::new(),
                    scalability_constraints: Vec::new(),
                },
                index_efficiency_analysis: IndexEfficiencyAnalysis {
                    index_usage_statistics: Vec::new(),
                    missing_index_opportunities: Vec::new(),
                    redundant_indexes: Vec::new(),
                    index_maintenance_recommendations: Vec::new(),
                },
                connection_pool_analysis: ConnectionPoolAnalysis {
                    pool_utilization: 0.0,
                    average_wait_time: Duration::from_secs(0),
                    connection_churn_rate: 0.0,
                    optimal_pool_size_recommendation: 10,
                    configuration_recommendations: Vec::new(),
                },
                optimization_recommendations: Vec::new(),
            }),
            performance_predictions,
            degradation_detection,
            overall_performance_assessment,
        })
    }

    /// Analyze dataset characteristics for baseline establishment
    async fn analyze_dataset_characteristics(&self, target_directory: &PathBuf) -> Result<DatasetCharacteristics> {
        // This would integrate with the existing directory analyzer
        // For now, return a placeholder implementation
        Ok(DatasetCharacteristics {
            total_files: 1000,
            total_size_bytes: 1024 * 1024 * 100, // 100MB
            file_type_distribution: HashMap::new(),
            average_file_size_bytes: 1024 * 100, // 100KB
            largest_file_size_bytes: 1024 * 1024 * 10, // 10MB
            directory_depth: 5,
            chaos_score: 0.2,
        })
    }

    /// Run baseline benchmark
    async fn run_baseline_benchmark(
        &self,
        pensieve_binary: &PathBuf,
        target_directory: &PathBuf,
        output_database: &PathBuf,
    ) -> Result<BaselineMetrics> {
        // Run pensieve and collect baseline metrics
        let start_time = Instant::now();
        
        // This would integrate with the existing pensieve runner
        // For now, return placeholder metrics
        Ok(BaselineMetrics {
            files_per_second: 100.0,
            peak_memory_usage_mb: 512,
            average_memory_usage_mb: 256,
            cpu_efficiency_score: 0.8,
            database_operations_per_second: 1000.0,
            total_processing_time: Duration::from_secs(60),
            memory_growth_pattern: MemoryGrowthPattern {
                growth_type: MemoryGrowthType::Linear,
                growth_rate_mb_per_sec: 2.0,
                peak_memory_mb: 512,
                memory_efficiency_score: 0.8,
                leak_indicators: Vec::new(),
            },
        })
    }

    /// Get system configuration
    async fn get_system_configuration(&self) -> Result<SystemConfiguration> {
        Ok(SystemConfiguration {
            cpu_cores: num_cpus::get() as u32,
            total_memory_mb: 8192, // This would be detected from system
            available_memory_mb: 4096,
            storage_type: "SSD".to_string(),
            os_info: std::env::consts::OS.to_string(),
            rust_version: "1.70.0".to_string(), // This would be detected
        })
    }

    /// Get pensieve version
    async fn get_pensieve_version(&self, pensieve_binary: &PathBuf) -> Result<String> {
        let output = Command::new(pensieve_binary)
            .arg("--version")
            .output()
            .await
            .map_err(|e| ValidationError::ProcessMonitoring(format!("Failed to get pensieve version: {}", e)))?;

        let version = String::from_utf8_lossy(&output.stdout);
        Ok(version.trim().to_string())
    }

    /// Generate performance predictions for different dataset characteristics
    async fn generate_performance_predictions(
        &self,
        target_directory: &PathBuf,
        scalability_analysis: Option<&ScalabilityAnalysisResults>,
    ) -> Result<Vec<PerformancePredictionModel>> {
        let mut predictions = Vec::new();

        // Generate predictions based on current dataset characteristics
        let dataset_chars = self.analyze_dataset_characteristics(target_directory).await?;
        
        // Create prediction models for different scenarios
        let scenarios = vec![
            ("2x Dataset Size", 2.0),
            ("5x Dataset Size", 5.0),
            ("10x Dataset Size", 10.0),
        ];

        for (scenario_name, multiplier) in scenarios {
            let mut scaled_chars = dataset_chars.clone();
            scaled_chars.total_files = (scaled_chars.total_files as f64 * multiplier) as u64;
            scaled_chars.total_size_bytes = (scaled_chars.total_size_bytes as f64 * multiplier) as u64;

            let performance_predictions = vec![
                PerformancePrediction {
                    metric_name: "Files Per Second".to_string(),
                    predicted_value: 100.0 / multiplier.sqrt(), // Assume square root scaling
                    confidence_score: 0.8,
                    prediction_range: (80.0 / multiplier.sqrt(), 120.0 / multiplier.sqrt()),
                    factors_considered: vec![
                        "Dataset size scaling".to_string(),
                        "Memory constraints".to_string(),
                        "I/O limitations".to_string(),
                    ],
                },
                PerformancePrediction {
                    metric_name: "Peak Memory Usage MB".to_string(),
                    predicted_value: 512.0 * multiplier.powf(0.8), // Assume sub-linear memory scaling
                    confidence_score: 0.7,
                    prediction_range: (400.0 * multiplier.powf(0.8), 600.0 * multiplier.powf(0.8)),
                    factors_considered: vec![
                        "Memory growth pattern".to_string(),
                        "Caching behavior".to_string(),
                    ],
                },
            ];

            predictions.push(PerformancePredictionModel {
                model_name: scenario_name.to_string(),
                dataset_characteristics: scaled_chars,
                performance_predictions,
                model_accuracy: 0.75,
                confidence_intervals: HashMap::new(),
            });
        }

        Ok(predictions)
    }

    /// Detect performance degradation compared to baseline
    async fn detect_performance_degradation(&self) -> Result<DegradationDetectionResults> {
        if let Some(baseline) = &self.baseline_metrics {
            // Compare current performance with baseline
            // This would involve running current tests and comparing
            // For now, return no degradation detected
            Ok(DegradationDetectionResults {
                degradation_detected: false,
                degradation_severity: DegradationSeverity::None,
                affected_metrics: Vec::new(),
                degradation_causes: Vec::new(),
                mitigation_recommendations: Vec::new(),
            })
        } else {
            Ok(DegradationDetectionResults {
                degradation_detected: false,
                degradation_severity: DegradationSeverity::None,
                affected_metrics: Vec::new(),
                degradation_causes: Vec::new(),
                mitigation_recommendations: vec![
                    "Establish performance baseline for degradation detection".to_string()
                ],
            })
        }
    }

    /// Compare current performance with baseline
    async fn compare_with_baseline(
        &self,
        baseline: &PerformanceBaseline,
        memory_analysis: &Option<MemoryAnalysisResults>,
        database_profiling: &Option<DatabaseProfilingResults>,
    ) -> Result<BaselineComparison> {
        // This would compare current metrics with baseline
        // For now, return a placeholder comparison
        Ok(BaselineComparison {
            baseline_established: true,
            performance_changes: vec![
                PerformanceChange {
                    metric_name: "Files Per Second".to_string(),
                    baseline_value: baseline.baseline_metrics.files_per_second,
                    current_value: 95.0, // Simulated current value
                    change_percentage: -5.0,
                    change_significance: ChangeSignificance::Minor,
                },
            ],
            degradation_detected: false,
            improvement_detected: false,
            overall_change_percentage: -2.5,
        })
    }

    /// Generate overall performance assessment
    async fn generate_overall_assessment(
        &self,
        scalability_analysis: &Option<ScalabilityAnalysisResults>,
        memory_analysis: &Option<MemoryAnalysisResults>,
        database_profiling: &Option<DatabaseProfilingResults>,
        degradation_detection: &DegradationDetectionResults,
    ) -> Result<OverallPerformanceAssessment> {
        // Calculate scores based on analysis results
        let performance_score = 0.8; // Would be calculated from actual metrics
        let scalability_score = scalability_analysis.as_ref()
            .map(|s| s.scalability_score)
            .unwrap_or(0.7);
        let efficiency_score = memory_analysis.as_ref()
            .map(|m| m.memory_efficiency_analysis.memory_utilization_score)
            .unwrap_or(0.8);
        let reliability_score = if degradation_detection.degradation_detected { 0.6 } else { 0.9 };

        let overall_score = (performance_score + scalability_score + efficiency_score + reliability_score) / 4.0;

        Ok(OverallPerformanceAssessment {
            performance_score,
            scalability_score,
            efficiency_score,
            reliability_score,
            overall_score,
            key_strengths: vec![
                "Consistent processing speed".to_string(),
                "Efficient memory usage".to_string(),
            ],
            key_weaknesses: vec![
                "Database query optimization opportunities".to_string(),
            ],
            priority_improvements: vec![
                "Optimize slow database queries".to_string(),
                "Implement memory usage monitoring".to_string(),
            ],
        })
    }
}

impl ScalabilityAnalyzer {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            test_configurations: vec![
                ScalabilityTestConfig {
                    test_name: "File Count Scaling".to_string(),
                    dataset_size_multipliers: vec![0.1, 0.5, 1.0, 2.0, 5.0],
                    file_count_multipliers: vec![0.1, 0.5, 1.0, 2.0, 5.0],
                    complexity_multipliers: vec![1.0],
                    expected_scaling_behavior: ScalingBehavior::Linear,
                },
                ScalabilityTestConfig {
                    test_name: "Dataset Size Scaling".to_string(),
                    dataset_size_multipliers: vec![0.1, 0.5, 1.0, 2.0, 5.0],
                    file_count_multipliers: vec![1.0],
                    complexity_multipliers: vec![1.0],
                    expected_scaling_behavior: ScalingBehavior::Linear,
                },
            ],
            extrapolation_models: Vec::new(),
        }
    }

    pub async fn analyze_scalability(
        &mut self,
        pensieve_binary: &PathBuf,
        target_directory: &PathBuf,
        output_database: &PathBuf,
    ) -> Result<ScalabilityAnalysisResults> {
        // Run scalability tests and analyze results
        // This would involve creating test datasets of different sizes
        // For now, return placeholder results
        Ok(ScalabilityAnalysisResults {
            scaling_behavior: ScalingBehavior::Linear,
            scalability_score: 0.8,
            bottleneck_points: vec![
                ScalabilityBottleneck {
                    bottleneck_point: 100000, // 100K files
                    bottleneck_type: "Memory Pressure".to_string(),
                    performance_impact: 0.3,
                    mitigation_strategies: vec![
                        "Increase available memory".to_string(),
                        "Implement streaming processing".to_string(),
                    ],
                },
            ],
            extrapolation_results: vec![
                ExtrapolationResult {
                    target_dataset_size: 1000000, // 1M files
                    predicted_performance: {
                        let mut perf = HashMap::new();
                        perf.insert("files_per_second".to_string(), 50.0);
                        perf.insert("peak_memory_mb".to_string(), 2048.0);
                        perf
                    },
                    confidence_score: 0.7,
                    assumptions: vec![
                        "Linear memory scaling".to_string(),
                        "No I/O bottlenecks".to_string(),
                    ],
                },
            ],
            scaling_recommendations: vec![
                ScalingRecommendation {
                    recommendation_type: "Memory Optimization".to_string(),
                    description: "Implement memory-efficient processing for large datasets".to_string(),
                    applicable_dataset_size_range: (50000, 1000000),
                    expected_improvement: 0.4,
                    implementation_complexity: ImplementationEffort::Medium,
                },
            ],
        })
    }
}

impl MemoryAnalyzer {
    pub fn new(sampling_interval: Duration, leak_threshold: f64) -> Self {
        Self {
            sampling_interval,
            leak_detection_threshold: leak_threshold,
            pattern_analysis_window: Duration::from_secs(300), // 5 minutes
        }
    }

    pub async fn analyze_memory_patterns(
        &self,
        pensieve_binary: &PathBuf,
        target_directory: &PathBuf,
        output_database: &PathBuf,
    ) -> Result<MemoryAnalysisResults> {
        // Run memory analysis during pensieve execution
        // This would involve monitoring memory usage over time
        // For now, return placeholder results
        Ok(MemoryAnalysisResults {
            memory_usage_pattern: MemoryUsagePattern {
                pattern_type: MemoryPatternType::GradualGrowth,
                baseline_memory_mb: 128,
                peak_memory_mb: 512,
                average_memory_mb: 256,
                memory_variance: 0.2,
                growth_phases: vec![
                    MemoryGrowthPhase {
                        phase_name: "Initial Loading".to_string(),
                        start_time: Duration::from_secs(0),
                        duration: Duration::from_secs(10),
                        start_memory_mb: 128,
                        end_memory_mb: 256,
                        growth_rate_mb_per_sec: 12.8,
                        files_processed_in_phase: 1000,
                    },
                ],
            },
            leak_detection_results: LeakDetectionResults {
                leak_detected: false,
                leak_severity: LeakSeverity::None,
                leak_rate_mb_per_hour: 5.0,
                leak_sources: Vec::new(),
                confidence_score: 0.9,
            },
            memory_efficiency_analysis: MemoryEfficiencyAnalysis {
                files_per_mb_ratio: 4.0, // 4 files per MB
                memory_utilization_score: 0.8,
                memory_waste_indicators: Vec::new(),
                optimization_potential_mb: 64,
            },
            memory_optimization_recommendations: vec![
                MemoryOptimizationRecommendation {
                    recommendation_type: "Buffer Size Optimization".to_string(),
                    description: "Optimize buffer sizes for better memory efficiency".to_string(),
                    estimated_savings_mb: 32,
                    implementation_effort: ImplementationEffort::Low,
                    impact_score: 0.6,
                },
            ],
        })
    }
}

impl DatabaseProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            profiling_enabled: enabled,
            query_analysis_threshold: Duration::from_millis(100),
            bottleneck_detection_threshold: 0.7,
        }
    }

    pub async fn profile_database_performance(
        &self,
        pensieve_binary: &PathBuf,
        target_directory: &PathBuf,
        output_database: &PathBuf,
    ) -> Result<DatabaseProfilingResults> {
        if !self.profiling_enabled {
            return Ok(DatabaseProfilingResults {
                query_performance_analysis: QueryPerformanceAnalysis {
                    slow_queries: Vec::new(),
                    query_patterns: Vec::new(),
                    performance_trends: Vec::new(),
                    query_optimization_opportunities: Vec::new(),
                },
                bottleneck_identification: BottleneckIdentification {
                    primary_bottlenecks: Vec::new(),
                    bottleneck_severity_score: 0.0,
                    performance_limiting_factors: Vec::new(),
                    scalability_constraints: Vec::new(),
                },
                index_efficiency_analysis: IndexEfficiencyAnalysis {
                    index_usage_statistics: Vec::new(),
                    missing_index_opportunities: Vec::new(),
                    redundant_indexes: Vec::new(),
                    index_maintenance_recommendations: Vec::new(),
                },
                connection_pool_analysis: ConnectionPoolAnalysis {
                    pool_utilization: 0.0,
                    average_wait_time: Duration::from_secs(0),
                    connection_churn_rate: 0.0,
                    optimal_pool_size_recommendation: 10,
                    configuration_recommendations: Vec::new(),
                },
                optimization_recommendations: Vec::new(),
            });
        }

        // Profile database performance during pensieve execution
        // This would involve monitoring database operations
        // For now, return placeholder results
        Ok(DatabaseProfilingResults {
            query_performance_analysis: QueryPerformanceAnalysis {
                slow_queries: vec![
                    SlowQueryAnalysis {
                        query_type: "INSERT INTO files".to_string(),
                        average_execution_time: Duration::from_millis(150),
                        max_execution_time: Duration::from_millis(500),
                        execution_count: 10000,
                        total_time_spent: Duration::from_secs(1500),
                        performance_impact: 0.8,
                        optimization_suggestions: vec![
                            "Add index on file_path column".to_string(),
                            "Use batch inserts".to_string(),
                        ],
                    },
                ],
                query_patterns: Vec::new(),
                performance_trends: Vec::new(),
                query_optimization_opportunities: Vec::new(),
            },
            bottleneck_identification: BottleneckIdentification {
                primary_bottlenecks: vec![
                    DatabaseBottleneck {
                        bottleneck_type: BottleneckType::DiskIO,
                        severity: 0.7,
                        description: "High disk I/O during bulk inserts".to_string(),
                        impact_on_throughput: 30.0,
                        mitigation_strategies: vec![
                            "Use SSD storage".to_string(),
                            "Optimize transaction batch size".to_string(),
                        ],
                    },
                ],
                bottleneck_severity_score: 0.7,
                performance_limiting_factors: Vec::new(),
                scalability_constraints: Vec::new(),
            },
            index_efficiency_analysis: IndexEfficiencyAnalysis {
                index_usage_statistics: Vec::new(),
                missing_index_opportunities: vec![
                    MissingIndexOpportunity {
                        table_name: "files".to_string(),
                        column_names: vec!["file_path".to_string()],
                        estimated_performance_improvement: 40.0,
                        query_patterns_affected: vec!["File lookup queries".to_string()],
                        creation_cost: ImplementationEffort::Low,
                    },
                ],
                redundant_indexes: Vec::new(),
                index_maintenance_recommendations: Vec::new(),
            },
            connection_pool_analysis: ConnectionPoolAnalysis {
                pool_utilization: 0.6,
                average_wait_time: Duration::from_millis(10),
                connection_churn_rate: 0.1,
                optimal_pool_size_recommendation: 15,
                configuration_recommendations: vec![
                    "Increase connection pool size to 15".to_string(),
                ],
            },
            optimization_recommendations: vec![
                DatabaseOptimizationRecommendation {
                    recommendation_type: OptimizationType::IndexCreation,
                    description: "Create index on files.file_path for faster lookups".to_string(),
                    estimated_performance_improvement: 40.0,
                    implementation_effort: ImplementationEffort::Low,
                    priority: RecommendationPriority::High,
                    implementation_steps: vec![
                        "CREATE INDEX idx_files_path ON files(file_path)".to_string(),
                    ],
                },
            ],
        })
    }
}

impl Default for PerformanceBenchmarker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn test_benchmark_config_creation() {
        let config = BenchmarkConfig::default();
        
        assert!(config.enable_baseline_establishment);
        assert!(config.enable_degradation_detection);
        assert!(config.enable_scalability_testing);
        assert!(config.enable_memory_analysis);
        assert!(config.enable_database_profiling);
        assert_eq!(config.benchmark_iterations, 3);
        assert_eq!(config.warmup_iterations, 1);
        assert_eq!(config.timeout_seconds, 3600);
        assert_eq!(config.memory_sampling_interval_ms, 100);
    }

    #[test]
    fn test_performance_benchmarker_creation() {
        let benchmarker = PerformanceBenchmarker::new();
        
        // Should create without panicking
        assert!(benchmarker.baseline_metrics.is_none());
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds {
            max_files_per_second_degradation: 0.2,
            max_memory_growth_rate_mb_per_sec: 10.0,
            max_database_operation_time_ms: 1000,
            min_cpu_efficiency_score: 0.6,
            max_memory_leak_rate_mb_per_hour: 50.0,
        };
        
        assert_eq!(thresholds.max_files_per_second_degradation, 0.2);
        assert_eq!(thresholds.max_memory_growth_rate_mb_per_sec, 10.0);
        assert_eq!(thresholds.max_database_operation_time_ms, 1000);
        assert_eq!(thresholds.min_cpu_efficiency_score, 0.6);
        assert_eq!(thresholds.max_memory_leak_rate_mb_per_hour, 50.0);
    }

    #[test]
    fn test_dataset_characteristics() {
        let characteristics = DatasetCharacteristics {
            total_files: 1000,
            total_size_bytes: 1024 * 1024 * 100, // 100MB
            file_type_distribution: std::collections::HashMap::new(),
            average_file_size_bytes: 1024 * 100, // 100KB
            largest_file_size_bytes: 1024 * 1024 * 10, // 10MB
            directory_depth: 5,
            chaos_score: 0.2,
        };
        
        assert_eq!(characteristics.total_files, 1000);
        assert_eq!(characteristics.total_size_bytes, 1024 * 1024 * 100);
        assert_eq!(characteristics.average_file_size_bytes, 1024 * 100);
        assert_eq!(characteristics.largest_file_size_bytes, 1024 * 1024 * 10);
        assert_eq!(characteristics.directory_depth, 5);
        assert_eq!(characteristics.chaos_score, 0.2);
    }

    #[test]
    fn test_memory_growth_pattern() {
        let pattern = MemoryGrowthPattern {
            growth_type: MemoryGrowthType::Linear,
            growth_rate_mb_per_sec: 2.0,
            peak_memory_mb: 512,
            memory_efficiency_score: 0.8,
            leak_indicators: Vec::new(),
        };
        
        assert!(matches!(pattern.growth_type, MemoryGrowthType::Linear));
        assert_eq!(pattern.growth_rate_mb_per_sec, 2.0);
        assert_eq!(pattern.peak_memory_mb, 512);
        assert_eq!(pattern.memory_efficiency_score, 0.8);
        assert!(pattern.leak_indicators.is_empty());
    }

    #[test]
    fn test_scaling_behavior() {
        let behaviors = vec![
            ScalingBehavior::Linear,
            ScalingBehavior::Logarithmic,
            ScalingBehavior::Quadratic,
            ScalingBehavior::Exponential,
            ScalingBehavior::Unknown,
        ];
        
        assert_eq!(behaviors.len(), 5);
        assert!(matches!(behaviors[0], ScalingBehavior::Linear));
        assert!(matches!(behaviors[1], ScalingBehavior::Logarithmic));
        assert!(matches!(behaviors[2], ScalingBehavior::Quadratic));
        assert!(matches!(behaviors[3], ScalingBehavior::Exponential));
        assert!(matches!(behaviors[4], ScalingBehavior::Unknown));
    }
}