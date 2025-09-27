use crate::errors::{ValidationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Comprehensive metrics collection system for real-time performance tracking
pub struct MetricsCollector {
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
    error_tracker: Arc<Mutex<ErrorTracker>>,
    ux_tracker: Arc<Mutex<UXTracker>>,
    database_tracker: Arc<Mutex<DatabaseTracker>>,
    start_time: Instant,
    collection_interval: Duration,
}

/// Real-time performance tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTracker {
    pub files_processed_per_second: Vec<f64>,
    pub memory_usage_over_time: Vec<MemoryDataPoint>,
    pub cpu_usage_over_time: Vec<CpuDataPoint>,
    pub processing_speed_analysis: ProcessingSpeedAnalysis,
    pub resource_efficiency_metrics: ResourceEfficiencyMetrics,
    pub performance_consistency_score: f64,
}

/// Memory usage data point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDataPoint {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64,
    pub memory_mb: u64,
    pub virtual_memory_mb: u64,
    pub memory_growth_rate: f64, // MB per second
}

impl MemoryDataPoint {
    pub fn new(memory_mb: u64, virtual_memory_mb: u64, memory_growth_rate: f64) -> Self {
        let now = Instant::now();
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            timestamp: now,
            timestamp_secs,
            memory_mb,
            virtual_memory_mb,
            memory_growth_rate,
        }
    }
}

/// CPU usage data point with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuDataPoint {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64,
    pub cpu_percent: f32,
    pub cpu_efficiency_score: f64, // 0.0 - 1.0
}

impl CpuDataPoint {
    pub fn new(cpu_percent: f32, cpu_efficiency_score: f64) -> Self {
        let now = Instant::now();
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            timestamp: now,
            timestamp_secs,
            cpu_percent,
            cpu_efficiency_score,
        }
    }
}

/// Processing speed analysis and trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSpeedAnalysis {
    pub current_files_per_second: f64,
    pub peak_files_per_second: f64,
    pub average_files_per_second: f64,
    pub speed_trend: SpeedTrend,
    pub speed_consistency_score: f64, // 0.0 - 1.0 (higher = more consistent)
    pub processing_phases: Vec<ProcessingPhase>,
}

/// Speed trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpeedTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Processing phase with distinct performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingPhase {
    pub phase_name: String,
    pub start_time: Duration,
    pub duration: Duration,
    pub files_per_second: f64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
}

/// Resource efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEfficiencyMetrics {
    pub memory_efficiency_score: f64,    // Files processed per MB
    pub cpu_efficiency_score: f64,       // Files processed per CPU percent
    pub overall_efficiency_score: f64,   // Combined efficiency metric
    pub resource_waste_indicators: Vec<ResourceWasteIndicator>,
}

/// Indicator of resource waste or inefficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceWasteIndicator {
    pub resource_type: String,
    pub waste_percentage: f64,
    pub description: String,
    pub impact_assessment: String,
}

/// Error tracking and categorization system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTracker {
    pub error_categories: HashMap<ErrorCategory, ErrorCategoryStats>,
    pub error_timeline: Vec<ErrorEvent>,
    pub recovery_patterns: Vec<RecoveryPattern>,
    pub error_rate_analysis: ErrorRateAnalysis,
    pub critical_error_threshold_breaches: u64,
}

/// Error category classification
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCategory {
    FileSystem,
    Permission,
    Memory,
    Database,
    Network,
    Timeout,
    Corruption,
    Configuration,
    Unknown,
}

/// Statistics for each error category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCategoryStats {
    pub total_count: u64,
    pub critical_count: u64,
    pub recoverable_count: u64,
    pub average_recovery_time: Duration,
    pub success_rate_after_error: f64,
}

/// Individual error event with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64,
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub context: ErrorContext,
    pub recovery_attempted: bool,
    pub recovery_successful: bool,
    pub recovery_time: Option<Duration>,
}

impl ErrorEvent {
    pub fn new(
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: String,
        context: ErrorContext,
    ) -> Self {
        let now = Instant::now();
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            timestamp: now,
            timestamp_secs,
            category,
            severity,
            message,
            context,
            recovery_attempted: false,
            recovery_successful: false,
            recovery_time: None,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Context information for errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub file_path: Option<String>,
    pub operation: String,
    pub system_state: SystemState,
    pub preceding_events: Vec<String>,
}

/// System state at time of error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f32,
    pub files_processed: u64,
    pub processing_speed: f64,
}

/// Recovery pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPattern {
    pub error_category: ErrorCategory,
    pub recovery_strategy: String,
    pub success_rate: f64,
    pub average_recovery_time: Duration,
    pub conditions_for_success: Vec<String>,
}

/// Error rate analysis over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRateAnalysis {
    pub errors_per_minute: f64,
    pub errors_per_thousand_files: f64,
    pub error_rate_trend: ErrorRateTrend,
    pub peak_error_periods: Vec<ErrorPeriod>,
}

/// Error rate trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorRateTrend {
    Improving,
    Worsening,
    Stable,
    Volatile,
}

/// Period of high error rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPeriod {
    pub start_time: Duration,
    pub duration: Duration,
    pub error_count: u64,
    pub dominant_error_category: ErrorCategory,
}

/// User experience tracking and evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXTracker {
    pub progress_reporting_quality: ProgressReportingQuality,
    pub error_message_clarity: ErrorMessageClarity,
    pub user_feedback_analysis: UserFeedbackAnalysis,
    pub completion_feedback_quality: CompletionFeedbackQuality,
    pub interruption_handling_quality: InterruptionHandlingQuality,
    pub overall_ux_score: f64, // 0.0 - 10.0
}

/// Progress reporting quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressReportingQuality {
    pub update_frequency_score: f64,        // 0.0 - 1.0
    pub information_completeness_score: f64, // 0.0 - 1.0
    pub clarity_score: f64,                 // 0.0 - 1.0
    pub eta_accuracy_score: f64,            // 0.0 - 1.0
    pub progress_updates: Vec<ProgressUpdate>,
}

/// Individual progress update analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64,
    pub message: String,
    pub information_density: f64,    // Information per character
    pub clarity_score: f64,          // 0.0 - 1.0
    pub actionability_score: f64,    // 0.0 - 1.0
}

impl ProgressUpdate {
    pub fn new(message: String, information_density: f64, clarity_score: f64, actionability_score: f64) -> Self {
        let now = Instant::now();
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            timestamp: now,
            timestamp_secs,
            message,
            information_density,
            clarity_score,
            actionability_score,
        }
    }
}

/// Error message clarity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessageClarity {
    pub average_clarity_score: f64,      // 0.0 - 1.0
    pub actionability_score: f64,        // 0.0 - 1.0
    pub technical_jargon_score: f64,     // 0.0 - 1.0 (lower = less jargon)
    pub solution_guidance_score: f64,    // 0.0 - 1.0
    pub error_messages: Vec<ErrorMessageAnalysis>,
}

/// Analysis of individual error messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessageAnalysis {
    pub message: String,
    pub clarity_score: f64,
    pub actionability_score: f64,
    pub contains_solution: bool,
    pub technical_complexity: TechnicalComplexity,
    pub improvement_suggestions: Vec<String>,
}

/// Technical complexity of error messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TechnicalComplexity {
    UserFriendly,
    Moderate,
    Technical,
    ExpertLevel,
}

/// User feedback analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedbackAnalysis {
    pub feedback_frequency: f64,         // Updates per minute
    pub feedback_usefulness_score: f64,  // 0.0 - 1.0
    pub next_steps_clarity: f64,         // 0.0 - 1.0
    pub user_confidence_indicators: Vec<ConfidenceIndicator>,
}

/// Indicators of user confidence in the process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIndicator {
    pub indicator_type: String,
    pub confidence_level: f64, // 0.0 - 1.0
    pub evidence: String,
}

/// Completion feedback quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionFeedbackQuality {
    pub summary_completeness: f64,       // 0.0 - 1.0
    pub results_clarity: f64,            // 0.0 - 1.0
    pub next_steps_guidance: f64,        // 0.0 - 1.0
    pub actionable_insights: f64,        // 0.0 - 1.0
}

/// Interruption handling quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionHandlingQuality {
    pub graceful_shutdown_score: f64,    // 0.0 - 1.0
    pub state_preservation_score: f64,   // 0.0 - 1.0
    pub recovery_instructions_score: f64, // 0.0 - 1.0
    pub cleanup_completeness_score: f64, // 0.0 - 1.0
}

/// Database operation timing and efficiency tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseTracker {
    pub operation_timings: HashMap<DatabaseOperation, OperationTimingStats>,
    pub query_performance_analysis: QueryPerformanceAnalysis,
    pub connection_efficiency: ConnectionEfficiency,
    pub transaction_patterns: Vec<TransactionPattern>,
    pub database_bottlenecks: Vec<DatabaseBottleneck>,
}

/// Database operation types
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseOperation {
    Insert,
    Update,
    Select,
    Delete,
    CreateIndex,
    Transaction,
    Vacuum,
    Analyze,
}

/// Timing statistics for database operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTimingStats {
    pub total_operations: u64,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub percentile_95: Duration,
    pub operations_per_second: f64,
}

/// Query performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    pub slow_queries: Vec<SlowQuery>,
    pub query_optimization_opportunities: Vec<OptimizationOpportunity>,
    pub index_usage_efficiency: f64, // 0.0 - 1.0
    pub cache_hit_ratio: f64,        // 0.0 - 1.0
}

/// Slow query identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQuery {
    pub query_pattern: String,
    pub execution_time: Duration,
    pub frequency: u64,
    pub impact_score: f64, // 0.0 - 1.0
}

/// Query optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub query_pattern: String,
    pub current_performance: Duration,
    pub estimated_improvement: f64, // Percentage improvement
    pub optimization_suggestion: String,
}

/// Database connection efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionEfficiency {
    pub connection_pool_utilization: f64, // 0.0 - 1.0
    pub connection_wait_time: Duration,
    pub connection_reuse_rate: f64,       // 0.0 - 1.0
    pub idle_connection_percentage: f64,
}

/// Transaction pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionPattern {
    pub pattern_name: String,
    pub frequency: u64,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub rollback_rate: f64,
}

/// Database bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseBottleneck {
    pub bottleneck_type: String,
    pub severity: f64, // 0.0 - 1.0
    pub description: String,
    pub impact_on_performance: f64, // 0.0 - 1.0
    pub mitigation_suggestions: Vec<String>,
}

/// Comprehensive metrics collection results
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsCollectionResults {
    pub collection_duration: Duration,
    pub performance_metrics: PerformanceTracker,
    pub error_metrics: ErrorTracker,
    pub ux_metrics: UXTracker,
    pub database_metrics: DatabaseTracker,
    pub overall_assessment: OverallAssessment,
}

/// Overall assessment based on all metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct OverallAssessment {
    pub performance_score: f64,    // 0.0 - 1.0
    pub reliability_score: f64,    // 0.0 - 1.0
    pub user_experience_score: f64, // 0.0 - 1.0
    pub efficiency_score: f64,     // 0.0 - 1.0
    pub overall_score: f64,        // 0.0 - 1.0
    pub key_insights: Vec<String>,
    pub improvement_recommendations: Vec<String>,
}

impl MetricsCollector {
    /// Create a new MetricsCollector with default configuration
    pub fn new() -> Self {
        Self::with_interval(Duration::from_millis(500))
    }

    /// Create a new MetricsCollector with custom collection interval
    pub fn with_interval(interval: Duration) -> Self {
        Self {
            performance_tracker: Arc::new(Mutex::new(PerformanceTracker::new())),
            error_tracker: Arc::new(Mutex::new(ErrorTracker::new())),
            ux_tracker: Arc::new(Mutex::new(UXTracker::new())),
            database_tracker: Arc::new(Mutex::new(DatabaseTracker::new())),
            start_time: Instant::now(),
            collection_interval: interval,
        }
    }

    /// Start comprehensive metrics collection
    pub async fn start_collection(&self) -> Result<(
        mpsc::Receiver<MetricsUpdate>,
        JoinHandle<()>
    )> {
        let (tx, rx) = mpsc::channel(1000);
        
        let performance_tracker = Arc::clone(&self.performance_tracker);
        let error_tracker = Arc::clone(&self.error_tracker);
        let ux_tracker = Arc::clone(&self.ux_tracker);
        let database_tracker = Arc::clone(&self.database_tracker);
        let interval = self.collection_interval;
        
        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                // Collect metrics from all trackers
                let update = MetricsUpdate {
                    timestamp: Instant::now(),
                    performance_snapshot: {
                        let tracker = performance_tracker.lock().unwrap();
                        tracker.create_snapshot()
                    },
                    error_snapshot: {
                        let tracker = error_tracker.lock().unwrap();
                        tracker.create_snapshot()
                    },
                    ux_snapshot: {
                        let tracker = ux_tracker.lock().unwrap();
                        tracker.create_snapshot()
                    },
                    database_snapshot: {
                        let tracker = database_tracker.lock().unwrap();
                        tracker.create_snapshot()
                    },
                };
                
                if tx.send(update).await.is_err() {
                    break; // Receiver dropped
                }
            }
        });
        
        Ok((rx, handle))
    }

    /// Record performance data point
    pub fn record_performance(&self, files_processed: u64, memory_mb: u64, cpu_percent: f32) -> Result<()> {
        let mut tracker = self.performance_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire performance tracker lock".to_string()))?;
        
        tracker.record_data_point(files_processed, memory_mb, cpu_percent, self.start_time.elapsed());
        Ok(())
    }

    /// Record error event
    pub fn record_error(&self, category: ErrorCategory, severity: ErrorSeverity, message: String, context: ErrorContext) -> Result<()> {
        let mut tracker = self.error_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire error tracker lock".to_string()))?;
        
        let error_event = ErrorEvent::new(category, severity, message, context);
        tracker.record_error(error_event);
        Ok(())
    }

    /// Record UX event
    pub fn record_ux_event(&self, event_type: UXEventType, message: String, quality_scores: UXQualityScores) -> Result<()> {
        let mut tracker = self.ux_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire UX tracker lock".to_string()))?;
        
        tracker.record_event(event_type, message, quality_scores);
        Ok(())
    }

    /// Record database operation
    pub fn record_database_operation(&self, operation: DatabaseOperation, duration: Duration, success: bool) -> Result<()> {
        let mut tracker = self.database_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire database tracker lock".to_string()))?;
        
        tracker.record_operation(operation, duration, success);
        Ok(())
    }

    /// Generate comprehensive metrics report
    pub fn generate_report(&self) -> Result<MetricsCollectionResults> {
        let collection_duration = self.start_time.elapsed();
        
        let performance_metrics = self.performance_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire performance tracker lock".to_string()))?
            .clone();
        
        let error_metrics = self.error_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire error tracker lock".to_string()))?
            .clone();
        
        let ux_metrics = self.ux_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire UX tracker lock".to_string()))?
            .clone();
        
        let database_metrics = self.database_tracker.lock()
            .map_err(|_| ValidationError::ProcessMonitoring("Failed to acquire database tracker lock".to_string()))?
            .clone();
        
        let overall_assessment = self.calculate_overall_assessment(
            &performance_metrics,
            &error_metrics,
            &ux_metrics,
            &database_metrics,
        );
        
        Ok(MetricsCollectionResults {
            collection_duration,
            performance_metrics,
            error_metrics,
            ux_metrics,
            database_metrics,
            overall_assessment,
        })
    }

    /// Calculate overall assessment from all metrics
    fn calculate_overall_assessment(
        &self,
        performance: &PerformanceTracker,
        errors: &ErrorTracker,
        ux: &UXTracker,
        database: &DatabaseTracker,
    ) -> OverallAssessment {
        // Calculate individual scores
        let performance_score = performance.performance_consistency_score;
        let reliability_score = self.calculate_reliability_score(errors);
        let user_experience_score = ux.overall_ux_score / 10.0; // Normalize to 0-1
        let efficiency_score = performance.resource_efficiency_metrics.overall_efficiency_score;
        
        // Calculate weighted overall score
        let overall_score = (performance_score * 0.3) + 
                           (reliability_score * 0.3) + 
                           (user_experience_score * 0.2) + 
                           (efficiency_score * 0.2);
        
        // Generate insights and recommendations
        let key_insights = self.generate_key_insights(performance, errors, ux, database);
        let improvement_recommendations = self.generate_improvement_recommendations(performance, errors, ux, database);
        
        OverallAssessment {
            performance_score,
            reliability_score,
            user_experience_score,
            efficiency_score,
            overall_score,
            key_insights,
            improvement_recommendations,
        }
    }

    /// Calculate reliability score from error metrics
    fn calculate_reliability_score(&self, errors: &ErrorTracker) -> f64 {
        let total_errors: u64 = errors.error_categories.values().map(|stats| stats.total_count).sum();
        let critical_errors: u64 = errors.error_categories.values().map(|stats| stats.critical_count).sum();
        
        if total_errors == 0 {
            return 1.0;
        }
        
        // Score based on error rate and severity
        let error_impact = (critical_errors as f64 * 2.0 + total_errors as f64) / (total_errors as f64 + 1.0);
        (1.0 - (error_impact / 10.0)).max(0.0)
    }

    /// Generate key insights from metrics
    fn generate_key_insights(
        &self,
        performance: &PerformanceTracker,
        errors: &ErrorTracker,
        ux: &UXTracker,
        _database: &DatabaseTracker,
    ) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Performance insights
        if performance.processing_speed_analysis.speed_consistency_score < 0.7 {
            insights.push("Processing speed shows high variability - investigate resource bottlenecks".to_string());
        }
        
        if performance.resource_efficiency_metrics.overall_efficiency_score < 0.6 {
            insights.push("Resource utilization efficiency is below optimal - consider optimization".to_string());
        }
        
        // Error insights
        let total_errors: u64 = errors.error_categories.values().map(|stats| stats.total_count).sum();
        if total_errors > 0 {
            insights.push(format!("Detected {} errors across {} categories", total_errors, errors.error_categories.len()));
        }
        
        // UX insights
        if ux.overall_ux_score < 7.0 {
            insights.push("User experience quality is below good threshold - focus on clarity and feedback".to_string());
        }
        
        insights
    }

    /// Generate improvement recommendations
    fn generate_improvement_recommendations(
        &self,
        performance: &PerformanceTracker,
        errors: &ErrorTracker,
        ux: &UXTracker,
        database: &DatabaseTracker,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Performance recommendations
        for waste_indicator in &performance.resource_efficiency_metrics.resource_waste_indicators {
            if waste_indicator.waste_percentage > 20.0 {
                recommendations.push(format!("Address {} waste: {}", waste_indicator.resource_type, waste_indicator.description));
            }
        }
        
        // Error recommendations
        for (category, stats) in &errors.error_categories {
            if stats.total_count > 10 {
                recommendations.push(format!("High {:?} error count ({}) - implement better error handling", category, stats.total_count));
            }
        }
        
        // UX recommendations
        if ux.progress_reporting_quality.update_frequency_score < 0.7 {
            recommendations.push("Improve progress update frequency for better user feedback".to_string());
        }
        
        if ux.error_message_clarity.actionability_score < 0.7 {
            recommendations.push("Make error messages more actionable with specific solutions".to_string());
        }
        
        // Database recommendations
        for bottleneck in &database.database_bottlenecks {
            if bottleneck.severity > 0.7 {
                recommendations.extend(bottleneck.mitigation_suggestions.clone());
            }
        }
        
        recommendations
    }
}

/// Metrics update message
#[derive(Debug)]
pub struct MetricsUpdate {
    pub timestamp: Instant,
    pub performance_snapshot: PerformanceSnapshot,
    pub error_snapshot: ErrorSnapshot,
    pub ux_snapshot: UXSnapshot,
    pub database_snapshot: DatabaseSnapshot,
}

/// Performance snapshot for real-time monitoring
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub current_files_per_second: f64,
    pub current_memory_mb: u64,
    pub current_cpu_percent: f32,
    pub efficiency_score: f64,
}

/// Error snapshot for real-time monitoring
#[derive(Debug, Clone)]
pub struct ErrorSnapshot {
    pub recent_errors: Vec<ErrorEvent>,
    pub error_rate: f64,
    pub critical_error_count: u64,
}

/// UX snapshot for real-time monitoring
#[derive(Debug, Clone)]
pub struct UXSnapshot {
    pub recent_progress_updates: Vec<ProgressUpdate>,
    pub current_ux_score: f64,
    pub feedback_quality: f64,
}

/// Database snapshot for real-time monitoring
#[derive(Debug, Clone)]
pub struct DatabaseSnapshot {
    pub recent_operations: Vec<(DatabaseOperation, Duration)>,
    pub current_performance: f64,
    pub bottleneck_severity: f64,
}

/// UX event types
#[derive(Debug, Clone)]
pub enum UXEventType {
    ProgressUpdate,
    ErrorMessage,
    CompletionMessage,
    UserFeedback,
    InterruptionHandling,
}

/// UX quality scores for events
#[derive(Debug, Clone)]
pub struct UXQualityScores {
    pub clarity: f64,
    pub actionability: f64,
    pub completeness: f64,
    pub user_friendliness: f64,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation details for individual trackers

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            files_processed_per_second: Vec::new(),
            memory_usage_over_time: Vec::new(),
            cpu_usage_over_time: Vec::new(),
            processing_speed_analysis: ProcessingSpeedAnalysis::new(),
            resource_efficiency_metrics: ResourceEfficiencyMetrics::new(),
            performance_consistency_score: 1.0,
        }
    }

    pub fn record_data_point(&mut self, files_processed: u64, memory_mb: u64, cpu_percent: f32, elapsed: Duration) {
        // Calculate files per second
        let files_per_second = if elapsed.as_secs() > 0 {
            files_processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        
        self.files_processed_per_second.push(files_per_second);
        
        // Calculate memory growth rate
        let memory_growth_rate = if let Some(last_memory) = self.memory_usage_over_time.last() {
            let time_diff = Instant::now().duration_since(last_memory.timestamp).as_secs_f64();
            if time_diff > 0.0 {
                (memory_mb as f64 - last_memory.memory_mb as f64) / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        self.memory_usage_over_time.push(MemoryDataPoint::new(memory_mb, memory_mb * 2, memory_growth_rate));
        
        // Calculate CPU efficiency score
        let cpu_efficiency = if cpu_percent > 0.0 {
            (files_per_second / cpu_percent as f64).min(1.0)
        } else {
            1.0
        };
        
        self.cpu_usage_over_time.push(CpuDataPoint::new(cpu_percent, cpu_efficiency));
        
        // Update processing speed analysis
        self.update_processing_speed_analysis();
        
        // Update resource efficiency metrics
        self.update_resource_efficiency_metrics();
        
        // Update performance consistency score
        self.update_performance_consistency_score();
    }

    fn update_processing_speed_analysis(&mut self) {
        if self.files_processed_per_second.is_empty() {
            return;
        }
        
        let current_speed = self.files_processed_per_second.last().copied().unwrap_or(0.0);
        let peak_speed = self.files_processed_per_second.iter().fold(0.0f64, |a, &b| a.max(b));
        let average_speed = self.files_processed_per_second.iter().sum::<f64>() / self.files_processed_per_second.len() as f64;
        
        // Determine speed trend
        let speed_trend = if self.files_processed_per_second.len() >= 10 {
            let recent_avg = self.files_processed_per_second.iter().rev().take(5).sum::<f64>() / 5.0;
            let older_avg = self.files_processed_per_second.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;
            
            let change_ratio = (recent_avg - older_avg) / older_avg.max(0.1);
            
            if change_ratio > 0.1 {
                SpeedTrend::Increasing
            } else if change_ratio < -0.1 {
                SpeedTrend::Decreasing
            } else if self.calculate_coefficient_of_variation(&self.files_processed_per_second) > 0.3 {
                SpeedTrend::Volatile
            } else {
                SpeedTrend::Stable
            }
        } else {
            SpeedTrend::Stable
        };
        
        // Calculate speed consistency
        let speed_consistency = 1.0 - self.calculate_coefficient_of_variation(&self.files_processed_per_second).min(1.0);
        
        self.processing_speed_analysis = ProcessingSpeedAnalysis {
            current_files_per_second: current_speed,
            peak_files_per_second: peak_speed,
            average_files_per_second: average_speed,
            speed_trend,
            speed_consistency_score: speed_consistency,
            processing_phases: self.identify_processing_phases(),
        };
    }

    fn identify_processing_phases(&self) -> Vec<ProcessingPhase> {
        let mut phases = Vec::new();
        
        if self.files_processed_per_second.len() < 10 {
            return phases;
        }
        
        // Simple phase detection based on speed changes
        let mut current_phase_start = 0;
        let mut current_phase_speed = self.files_processed_per_second[0];
        
        for (i, &speed) in self.files_processed_per_second.iter().enumerate().skip(1) {
            let speed_change = (speed - current_phase_speed).abs() / current_phase_speed.max(0.1);
            
            if speed_change > 0.5 || i == self.files_processed_per_second.len() - 1 {
                // End current phase
                let phase_duration = Duration::from_secs((i - current_phase_start) as u64);
                let phase_avg_speed = self.files_processed_per_second[current_phase_start..i]
                    .iter().sum::<f64>() / (i - current_phase_start) as f64;
                
                phases.push(ProcessingPhase {
                    phase_name: format!("Phase {}", phases.len() + 1),
                    start_time: Duration::from_secs(current_phase_start as u64),
                    duration: phase_duration,
                    files_per_second: phase_avg_speed,
                    memory_usage_mb: self.memory_usage_over_time.get(i).map(|m| m.memory_mb).unwrap_or(0),
                    cpu_usage_percent: self.cpu_usage_over_time.get(i).map(|c| c.cpu_percent).unwrap_or(0.0),
                });
                
                current_phase_start = i;
                current_phase_speed = speed;
            }
        }
        
        phases
    }

    fn update_resource_efficiency_metrics(&mut self) {
        if self.memory_usage_over_time.is_empty() || self.files_processed_per_second.is_empty() {
            return;
        }
        
        // Calculate memory efficiency (files per MB)
        let avg_memory = self.memory_usage_over_time.iter().map(|m| m.memory_mb).sum::<u64>() as f64 / self.memory_usage_over_time.len() as f64;
        let avg_files_per_second = self.files_processed_per_second.iter().sum::<f64>() / self.files_processed_per_second.len() as f64;
        let memory_efficiency = if avg_memory > 0.0 {
            (avg_files_per_second / avg_memory * 100.0).min(1.0)
        } else {
            0.0
        };
        
        // Calculate CPU efficiency
        let avg_cpu = self.cpu_usage_over_time.iter().map(|c| c.cpu_percent).sum::<f32>() as f64 / self.cpu_usage_over_time.len() as f64;
        let cpu_efficiency = if avg_cpu > 0.0 {
            (avg_files_per_second / avg_cpu * 10.0).min(1.0)
        } else {
            0.0
        };
        
        let overall_efficiency = (memory_efficiency + cpu_efficiency) / 2.0;
        
        // Identify resource waste
        let mut waste_indicators = Vec::new();
        
        if memory_efficiency < 0.3 {
            waste_indicators.push(ResourceWasteIndicator {
                resource_type: "Memory".to_string(),
                waste_percentage: (1.0 - memory_efficiency) * 100.0,
                description: "High memory usage relative to processing throughput".to_string(),
                impact_assessment: "May indicate memory leaks or inefficient data structures".to_string(),
            });
        }
        
        if cpu_efficiency < 0.3 {
            waste_indicators.push(ResourceWasteIndicator {
                resource_type: "CPU".to_string(),
                waste_percentage: (1.0 - cpu_efficiency) * 100.0,
                description: "High CPU usage relative to processing throughput".to_string(),
                impact_assessment: "May indicate algorithmic inefficiencies or blocking operations".to_string(),
            });
        }
        
        self.resource_efficiency_metrics = ResourceEfficiencyMetrics {
            memory_efficiency_score: memory_efficiency,
            cpu_efficiency_score: cpu_efficiency,
            overall_efficiency_score: overall_efficiency,
            resource_waste_indicators: waste_indicators,
        };
    }

    fn update_performance_consistency_score(&mut self) {
        if self.files_processed_per_second.len() < 2 {
            self.performance_consistency_score = 1.0;
            return;
        }
        
        let speed_consistency = 1.0 - self.calculate_coefficient_of_variation(&self.files_processed_per_second).min(1.0);
        
        let memory_values: Vec<f64> = self.memory_usage_over_time.iter().map(|m| m.memory_mb as f64).collect();
        let memory_consistency = if memory_values.len() > 1 {
            1.0 - self.calculate_coefficient_of_variation(&memory_values).min(1.0)
        } else {
            1.0
        };
        
        let cpu_values: Vec<f64> = self.cpu_usage_over_time.iter().map(|c| c.cpu_percent as f64).collect();
        let cpu_consistency = if cpu_values.len() > 1 {
            1.0 - self.calculate_coefficient_of_variation(&cpu_values).min(1.0)
        } else {
            1.0
        };
        
        self.performance_consistency_score = (speed_consistency + memory_consistency + cpu_consistency) / 3.0;
    }

    fn calculate_coefficient_of_variation(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }
        
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        std_dev / mean
    }

    pub fn create_snapshot(&self) -> PerformanceSnapshot {
        PerformanceSnapshot {
            current_files_per_second: self.processing_speed_analysis.current_files_per_second,
            current_memory_mb: self.memory_usage_over_time.last().map(|m| m.memory_mb).unwrap_or(0),
            current_cpu_percent: self.cpu_usage_over_time.last().map(|c| c.cpu_percent).unwrap_or(0.0),
            efficiency_score: self.resource_efficiency_metrics.overall_efficiency_score,
        }
    }
}

impl ProcessingSpeedAnalysis {
    pub fn new() -> Self {
        Self {
            current_files_per_second: 0.0,
            peak_files_per_second: 0.0,
            average_files_per_second: 0.0,
            speed_trend: SpeedTrend::Stable,
            speed_consistency_score: 1.0,
            processing_phases: Vec::new(),
        }
    }
}

impl ResourceEfficiencyMetrics {
    pub fn new() -> Self {
        Self {
            memory_efficiency_score: 0.0,
            cpu_efficiency_score: 0.0,
            overall_efficiency_score: 0.0,
            resource_waste_indicators: Vec::new(),
        }
    }
}

impl ErrorTracker {
    pub fn new() -> Self {
        Self {
            error_categories: HashMap::new(),
            error_timeline: Vec::new(),
            recovery_patterns: Vec::new(),
            error_rate_analysis: ErrorRateAnalysis::new(),
            critical_error_threshold_breaches: 0,
        }
    }

    pub fn record_error(&mut self, mut error_event: ErrorEvent) {
        // Update category statistics
        let stats = self.error_categories.entry(error_event.category.clone()).or_insert_with(|| ErrorCategoryStats {
            total_count: 0,
            critical_count: 0,
            recoverable_count: 0,
            average_recovery_time: Duration::from_secs(0),
            success_rate_after_error: 0.0,
        });
        
        stats.total_count += 1;
        
        match error_event.severity {
            ErrorSeverity::Critical => {
                stats.critical_count += 1;
                self.critical_error_threshold_breaches += 1;
            }
            _ => {
                stats.recoverable_count += 1;
            }
        }
        
        // Attempt recovery analysis
        self.analyze_recovery_potential(&mut error_event);
        
        // Add to timeline
        self.error_timeline.push(error_event);
        
        // Update error rate analysis
        self.update_error_rate_analysis();
        
        // Update recovery patterns
        self.update_recovery_patterns();
    }

    fn analyze_recovery_potential(&self, error_event: &mut ErrorEvent) {
        // Simple recovery analysis based on error category and context
        match error_event.category {
            ErrorCategory::FileSystem | ErrorCategory::Permission => {
                error_event.recovery_attempted = true;
                error_event.recovery_successful = error_event.context.system_state.memory_usage_mb < 1000; // Simple heuristic
                if error_event.recovery_successful {
                    error_event.recovery_time = Some(Duration::from_millis(100));
                }
            }
            ErrorCategory::Memory => {
                error_event.recovery_attempted = true;
                error_event.recovery_successful = false; // Memory errors are typically not recoverable
            }
            ErrorCategory::Timeout => {
                error_event.recovery_attempted = true;
                error_event.recovery_successful = true; // Timeouts can often be retried
                error_event.recovery_time = Some(Duration::from_secs(1));
            }
            _ => {
                error_event.recovery_attempted = false;
            }
        }
    }

    fn update_error_rate_analysis(&mut self) {
        if self.error_timeline.is_empty() {
            return;
        }
        
        let now = Instant::now();
        let recent_errors = self.error_timeline.iter()
            .filter(|e| now.duration_since(e.timestamp) < Duration::from_secs(60))
            .count();
        
        let errors_per_minute = recent_errors as f64;
        
        // Calculate errors per thousand files (rough estimate)
        let total_files_processed = self.error_timeline.iter()
            .map(|e| e.context.system_state.files_processed)
            .max()
            .unwrap_or(1);
        
        let errors_per_thousand_files = (self.error_timeline.len() as f64 / total_files_processed as f64) * 1000.0;
        
        // Determine trend
        let error_rate_trend = if self.error_timeline.len() >= 10 {
            let recent_rate = self.error_timeline.iter().rev().take(5).count() as f64;
            let older_rate = self.error_timeline.iter().rev().skip(5).take(5).count() as f64;
            
            if recent_rate > older_rate * 1.2 {
                ErrorRateTrend::Worsening
            } else if recent_rate < older_rate * 0.8 {
                ErrorRateTrend::Improving
            } else {
                ErrorRateTrend::Stable
            }
        } else {
            ErrorRateTrend::Stable
        };
        
        // Identify peak error periods
        let peak_error_periods = self.identify_peak_error_periods();
        
        self.error_rate_analysis = ErrorRateAnalysis {
            errors_per_minute,
            errors_per_thousand_files,
            error_rate_trend,
            peak_error_periods,
        };
    }

    fn identify_peak_error_periods(&self) -> Vec<ErrorPeriod> {
        let mut periods = Vec::new();
        
        if self.error_timeline.len() < 5 {
            return periods;
        }
        
        // Group errors by time windows
        let window_size = Duration::from_secs(60); // 1-minute windows
        let mut current_window_start = self.error_timeline[0].timestamp;
        let mut current_window_errors = Vec::new();
        
        for error in &self.error_timeline {
            if error.timestamp.duration_since(current_window_start) > window_size {
                // Process current window
                if current_window_errors.len() > 3 { // Threshold for "peak"
                    let dominant_category = self.find_dominant_error_category(&current_window_errors);
                    periods.push(ErrorPeriod {
                        start_time: current_window_start.duration_since(self.error_timeline[0].timestamp),
                        duration: window_size,
                        error_count: current_window_errors.len() as u64,
                        dominant_error_category: dominant_category,
                    });
                }
                
                // Start new window
                current_window_start = error.timestamp;
                current_window_errors.clear();
            }
            
            current_window_errors.push(error);
        }
        
        periods
    }

    fn find_dominant_error_category(&self, errors: &[&ErrorEvent]) -> ErrorCategory {
        let mut category_counts = HashMap::new();
        
        for error in errors {
            *category_counts.entry(error.category.clone()).or_insert(0) += 1;
        }
        
        category_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(category, _)| category)
            .unwrap_or(ErrorCategory::Unknown)
    }

    fn update_recovery_patterns(&mut self) {
        let mut pattern_stats: HashMap<ErrorCategory, (u64, u64, Vec<Duration>)> = HashMap::new();
        
        for error in &self.error_timeline {
            if error.recovery_attempted {
                let (total, successful, times) = pattern_stats.entry(error.category.clone()).or_insert((0, 0, Vec::new()));
                *total += 1;
                
                if error.recovery_successful {
                    *successful += 1;
                    if let Some(recovery_time) = error.recovery_time {
                        times.push(recovery_time);
                    }
                }
            }
        }
        
        self.recovery_patterns = pattern_stats.into_iter().map(|(category, (total, successful, times))| {
            let success_rate = if total > 0 { successful as f64 / total as f64 } else { 0.0 };
            let average_recovery_time = if !times.is_empty() {
                times.iter().sum::<Duration>() / times.len() as u32
            } else {
                Duration::from_secs(0)
            };
            
            RecoveryPattern {
                error_category: category.clone(),
                recovery_strategy: format!("Auto-recovery for {:?} errors", category),
                success_rate,
                average_recovery_time,
                conditions_for_success: vec![
                    "System memory usage < 80%".to_string(),
                    "No concurrent critical errors".to_string(),
                ],
            }
        }).collect();
    }

    pub fn create_snapshot(&self) -> ErrorSnapshot {
        let now = Instant::now();
        let recent_errors = self.error_timeline.iter()
            .filter(|e| now.duration_since(e.timestamp) < Duration::from_secs(60))
            .cloned()
            .collect();
        
        let critical_error_count = self.error_timeline.iter()
            .filter(|e| matches!(e.severity, ErrorSeverity::Critical))
            .count() as u64;
        
        ErrorSnapshot {
            recent_errors,
            error_rate: self.error_rate_analysis.errors_per_minute,
            critical_error_count,
        }
    }
}

impl ErrorRateAnalysis {
    pub fn new() -> Self {
        Self {
            errors_per_minute: 0.0,
            errors_per_thousand_files: 0.0,
            error_rate_trend: ErrorRateTrend::Stable,
            peak_error_periods: Vec::new(),
        }
    }
}

impl UXTracker {
    pub fn new() -> Self {
        Self {
            progress_reporting_quality: ProgressReportingQuality::new(),
            error_message_clarity: ErrorMessageClarity::new(),
            user_feedback_analysis: UserFeedbackAnalysis::new(),
            completion_feedback_quality: CompletionFeedbackQuality::new(),
            interruption_handling_quality: InterruptionHandlingQuality::new(),
            overall_ux_score: 7.0, // Start with neutral score
        }
    }

    pub fn record_event(&mut self, event_type: UXEventType, message: String, quality_scores: UXQualityScores) {
        match event_type {
            UXEventType::ProgressUpdate => {
                self.record_progress_update(message, quality_scores);
            }
            UXEventType::ErrorMessage => {
                self.record_error_message(message, quality_scores);
            }
            UXEventType::CompletionMessage => {
                self.record_completion_message(quality_scores);
            }
            UXEventType::UserFeedback => {
                self.record_user_feedback(message, quality_scores);
            }
            UXEventType::InterruptionHandling => {
                self.record_interruption_handling(quality_scores);
            }
        }
        
        self.update_overall_ux_score();
    }

    fn record_progress_update(&mut self, message: String, quality_scores: UXQualityScores) {
        let information_density = self.calculate_information_density(&message);
        
        let progress_update = ProgressUpdate::new(
            message,
            information_density,
            quality_scores.clarity,
            quality_scores.actionability,
        );
        
        self.progress_reporting_quality.progress_updates.push(progress_update);
        self.update_progress_reporting_quality();
    }

    fn record_error_message(&mut self, message: String, quality_scores: UXQualityScores) {
        let technical_complexity = self.assess_technical_complexity(&message);
        let contains_solution = self.contains_solution_guidance(&message);
        let improvement_suggestions = self.generate_error_message_improvements(&message, &quality_scores);
        
        let error_analysis = ErrorMessageAnalysis {
            message,
            clarity_score: quality_scores.clarity,
            actionability_score: quality_scores.actionability,
            contains_solution,
            technical_complexity,
            improvement_suggestions,
        };
        
        self.error_message_clarity.error_messages.push(error_analysis);
        self.update_error_message_clarity();
    }

    fn record_completion_message(&mut self, quality_scores: UXQualityScores) {
        self.completion_feedback_quality = CompletionFeedbackQuality {
            summary_completeness: quality_scores.completeness,
            results_clarity: quality_scores.clarity,
            next_steps_guidance: quality_scores.actionability,
            actionable_insights: quality_scores.user_friendliness,
        };
    }

    fn record_user_feedback(&mut self, _message: String, quality_scores: UXQualityScores) {
        // Update user feedback analysis
        self.user_feedback_analysis.feedback_usefulness_score = quality_scores.completeness;
        self.user_feedback_analysis.next_steps_clarity = quality_scores.actionability;
        
        // Add confidence indicator
        let confidence_indicator = ConfidenceIndicator {
            indicator_type: "User Feedback Quality".to_string(),
            confidence_level: quality_scores.user_friendliness,
            evidence: "Based on feedback clarity and actionability scores".to_string(),
        };
        
        self.user_feedback_analysis.user_confidence_indicators.push(confidence_indicator);
    }

    fn record_interruption_handling(&mut self, quality_scores: UXQualityScores) {
        self.interruption_handling_quality = InterruptionHandlingQuality {
            graceful_shutdown_score: quality_scores.user_friendliness,
            state_preservation_score: quality_scores.completeness,
            recovery_instructions_score: quality_scores.actionability,
            cleanup_completeness_score: quality_scores.clarity,
        };
    }

    fn calculate_information_density(&self, message: &str) -> f64 {
        // Simple heuristic: information words per total words
        let total_words = message.split_whitespace().count();
        if total_words == 0 {
            return 0.0;
        }
        
        let information_words = message.split_whitespace()
            .filter(|word| {
                // Count words that likely contain information
                word.len() > 3 && 
                !["the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "man", "men", "put", "say", "she", "too", "use"].contains(&word.to_lowercase().as_str())
            })
            .count();
        
        information_words as f64 / total_words as f64
    }

    fn assess_technical_complexity(&self, message: &str) -> TechnicalComplexity {
        let technical_terms = ["error", "exception", "stack", "trace", "null", "pointer", "memory", "allocation", "thread", "mutex", "deadlock", "race", "condition"];
        let message_lower = message.to_lowercase();
        
        let technical_word_count = technical_terms.iter()
            .filter(|term| message_lower.contains(*term))
            .count();
        
        match technical_word_count {
            0 => TechnicalComplexity::UserFriendly,
            1..=2 => TechnicalComplexity::Moderate,
            3..=5 => TechnicalComplexity::Technical,
            _ => TechnicalComplexity::ExpertLevel,
        }
    }

    fn contains_solution_guidance(&self, message: &str) -> bool {
        let solution_indicators = ["try", "check", "ensure", "verify", "run", "install", "update", "configure", "set", "use"];
        let message_lower = message.to_lowercase();
        
        solution_indicators.iter().any(|indicator| message_lower.contains(indicator))
    }

    fn generate_error_message_improvements(&self, message: &str, quality_scores: &UXQualityScores) -> Vec<String> {
        let mut improvements = Vec::new();
        
        if quality_scores.clarity < 0.7 {
            improvements.push("Make the error message more specific and clear".to_string());
        }
        
        if quality_scores.actionability < 0.7 {
            improvements.push("Add specific steps the user can take to resolve the issue".to_string());
        }
        
        if !self.contains_solution_guidance(message) {
            improvements.push("Include suggested solutions or next steps".to_string());
        }
        
        if matches!(self.assess_technical_complexity(message), TechnicalComplexity::Technical | TechnicalComplexity::ExpertLevel) {
            improvements.push("Reduce technical jargon and use more user-friendly language".to_string());
        }
        
        improvements
    }

    fn update_progress_reporting_quality(&mut self) {
        if self.progress_reporting_quality.progress_updates.is_empty() {
            return;
        }
        
        let updates = &self.progress_reporting_quality.progress_updates;
        
        // Calculate update frequency score
        let update_frequency_score = if updates.len() > 1 {
            let total_time = updates.last().unwrap().timestamp.duration_since(updates.first().unwrap().timestamp);
            let expected_updates = total_time.as_secs() / 5; // Expect update every 5 seconds
            (updates.len() as f64 / expected_updates.max(1) as f64).min(1.0)
        } else {
            0.5
        };
        
        // Calculate average scores
        let information_completeness_score = updates.iter().map(|u| u.information_density).sum::<f64>() / updates.len() as f64;
        let clarity_score = updates.iter().map(|u| u.clarity_score).sum::<f64>() / updates.len() as f64;
        let actionability_score = updates.iter().map(|u| u.actionability_score).sum::<f64>() / updates.len() as f64;
        
        self.progress_reporting_quality.update_frequency_score = update_frequency_score;
        self.progress_reporting_quality.information_completeness_score = information_completeness_score;
        self.progress_reporting_quality.clarity_score = clarity_score;
        self.progress_reporting_quality.eta_accuracy_score = 0.8; // Would need actual ETA tracking
    }

    fn update_error_message_clarity(&mut self) {
        if self.error_message_clarity.error_messages.is_empty() {
            return;
        }
        
        let messages = &self.error_message_clarity.error_messages;
        
        let average_clarity = messages.iter().map(|m| m.clarity_score).sum::<f64>() / messages.len() as f64;
        let average_actionability = messages.iter().map(|m| m.actionability_score).sum::<f64>() / messages.len() as f64;
        
        let technical_jargon_score = messages.iter()
            .map(|m| match m.technical_complexity {
                TechnicalComplexity::UserFriendly => 1.0,
                TechnicalComplexity::Moderate => 0.7,
                TechnicalComplexity::Technical => 0.4,
                TechnicalComplexity::ExpertLevel => 0.1,
            })
            .sum::<f64>() / messages.len() as f64;
        
        let solution_guidance_score = messages.iter()
            .map(|m| if m.contains_solution { 1.0 } else { 0.0 })
            .sum::<f64>() / messages.len() as f64;
        
        self.error_message_clarity.average_clarity_score = average_clarity;
        self.error_message_clarity.actionability_score = average_actionability;
        self.error_message_clarity.technical_jargon_score = technical_jargon_score;
        self.error_message_clarity.solution_guidance_score = solution_guidance_score;
    }

    fn update_overall_ux_score(&mut self) {
        let progress_score = (self.progress_reporting_quality.update_frequency_score + 
                             self.progress_reporting_quality.information_completeness_score + 
                             self.progress_reporting_quality.clarity_score) / 3.0;
        
        let error_message_score = (self.error_message_clarity.average_clarity_score + 
                                  self.error_message_clarity.actionability_score + 
                                  self.error_message_clarity.solution_guidance_score) / 3.0;
        
        let completion_score = (self.completion_feedback_quality.summary_completeness + 
                               self.completion_feedback_quality.results_clarity + 
                               self.completion_feedback_quality.next_steps_guidance + 
                               self.completion_feedback_quality.actionable_insights) / 4.0;
        
        let interruption_score = (self.interruption_handling_quality.graceful_shutdown_score + 
                                 self.interruption_handling_quality.state_preservation_score + 
                                 self.interruption_handling_quality.recovery_instructions_score + 
                                 self.interruption_handling_quality.cleanup_completeness_score) / 4.0;
        
        // Weighted average (0-1 scale, then convert to 0-10)
        self.overall_ux_score = ((progress_score * 0.3) + 
                                (error_message_score * 0.3) + 
                                (completion_score * 0.2) + 
                                (interruption_score * 0.2)) * 10.0;
    }

    pub fn create_snapshot(&self) -> UXSnapshot {
        let recent_updates = self.progress_reporting_quality.progress_updates.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        
        UXSnapshot {
            recent_progress_updates: recent_updates,
            current_ux_score: self.overall_ux_score,
            feedback_quality: self.user_feedback_analysis.feedback_usefulness_score,
        }
    }
}

impl ProgressReportingQuality {
    pub fn new() -> Self {
        Self {
            update_frequency_score: 0.0,
            information_completeness_score: 0.0,
            clarity_score: 0.0,
            eta_accuracy_score: 0.0,
            progress_updates: Vec::new(),
        }
    }
}

impl ErrorMessageClarity {
    pub fn new() -> Self {
        Self {
            average_clarity_score: 0.0,
            actionability_score: 0.0,
            technical_jargon_score: 1.0,
            solution_guidance_score: 0.0,
            error_messages: Vec::new(),
        }
    }
}

impl UserFeedbackAnalysis {
    pub fn new() -> Self {
        Self {
            feedback_frequency: 0.0,
            feedback_usefulness_score: 0.0,
            next_steps_clarity: 0.0,
            user_confidence_indicators: Vec::new(),
        }
    }
}

impl CompletionFeedbackQuality {
    pub fn new() -> Self {
        Self {
            summary_completeness: 0.0,
            results_clarity: 0.0,
            next_steps_guidance: 0.0,
            actionable_insights: 0.0,
        }
    }
}

impl InterruptionHandlingQuality {
    pub fn new() -> Self {
        Self {
            graceful_shutdown_score: 0.0,
            state_preservation_score: 0.0,
            recovery_instructions_score: 0.0,
            cleanup_completeness_score: 0.0,
        }
    }
}

impl DatabaseTracker {
    pub fn new() -> Self {
        Self {
            operation_timings: HashMap::new(),
            query_performance_analysis: QueryPerformanceAnalysis::new(),
            connection_efficiency: ConnectionEfficiency::new(),
            transaction_patterns: Vec::new(),
            database_bottlenecks: Vec::new(),
        }
    }

    pub fn record_operation(&mut self, operation: DatabaseOperation, duration: Duration, success: bool) {
        let stats = self.operation_timings.entry(operation.clone()).or_insert_with(|| OperationTimingStats {
            total_operations: 0,
            total_time: Duration::from_secs(0),
            average_time: Duration::from_secs(0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::from_secs(0),
            percentile_95: Duration::from_secs(0),
            operations_per_second: 0.0,
        });
        
        stats.total_operations += 1;
        stats.total_time += duration;
        stats.average_time = stats.total_time / stats.total_operations as u32;
        stats.min_time = stats.min_time.min(duration);
        stats.max_time = stats.max_time.max(duration);
        
        // Simple operations per second calculation
        if stats.total_time.as_secs() > 0 {
            stats.operations_per_second = stats.total_operations as f64 / stats.total_time.as_secs_f64();
        }
        
        // Identify slow queries
        if duration > Duration::from_millis(100) {
            self.identify_slow_query(operation, duration);
        }
        
        // Update bottleneck analysis
        self.update_bottleneck_analysis();
    }

    fn identify_slow_query(&mut self, operation: DatabaseOperation, duration: Duration) {
        let query_pattern = format!("{:?} operation", operation);
        
        // Check if this pattern already exists
        if let Some(existing) = self.query_performance_analysis.slow_queries.iter_mut()
            .find(|q| q.query_pattern == query_pattern) {
            existing.frequency += 1;
            existing.execution_time = existing.execution_time.max(duration);
        } else {
            self.query_performance_analysis.slow_queries.push(SlowQuery {
                query_pattern,
                execution_time: duration,
                frequency: 1,
                impact_score: (duration.as_millis() as f64 / 1000.0).min(1.0),
            });
        }
    }

    fn update_bottleneck_analysis(&mut self) {
        self.database_bottlenecks.clear();
        
        // Analyze operation timings for bottlenecks
        for (operation, stats) in &self.operation_timings {
            if stats.average_time > Duration::from_millis(50) {
                let severity = (stats.average_time.as_millis() as f64 / 1000.0).min(1.0);
                
                self.database_bottlenecks.push(DatabaseBottleneck {
                    bottleneck_type: format!("{:?} operations", operation),
                    severity,
                    description: format!("Average {:?} operation time is {}ms", operation, stats.average_time.as_millis()),
                    impact_on_performance: severity,
                    mitigation_suggestions: self.generate_mitigation_suggestions(operation, stats),
                });
            }
        }
    }

    fn generate_mitigation_suggestions(&self, operation: &DatabaseOperation, _stats: &OperationTimingStats) -> Vec<String> {
        match operation {
            DatabaseOperation::Insert => vec![
                "Consider batch inserts for better performance".to_string(),
                "Ensure proper indexing on frequently queried columns".to_string(),
            ],
            DatabaseOperation::Select => vec![
                "Add indexes on WHERE clause columns".to_string(),
                "Consider query optimization or result caching".to_string(),
            ],
            DatabaseOperation::Update => vec![
                "Ensure WHERE clause uses indexed columns".to_string(),
                "Consider batch updates for multiple rows".to_string(),
            ],
            DatabaseOperation::Delete => vec![
                "Ensure WHERE clause uses indexed columns".to_string(),
                "Consider soft deletes for frequently accessed data".to_string(),
            ],
            _ => vec![
                "Monitor and optimize database configuration".to_string(),
            ],
        }
    }

    pub fn create_snapshot(&self) -> DatabaseSnapshot {
        let recent_operations = self.operation_timings.iter()
            .take(10)
            .map(|(op, stats)| (op.clone(), stats.average_time))
            .collect();
        
        let current_performance = if !self.operation_timings.is_empty() {
            let avg_ops_per_sec: f64 = self.operation_timings.values()
                .map(|stats| stats.operations_per_second)
                .sum::<f64>() / self.operation_timings.len() as f64;
            (avg_ops_per_sec / 100.0).min(1.0) // Normalize to 0-1
        } else {
            0.0
        };
        
        let bottleneck_severity = self.database_bottlenecks.iter()
            .map(|b| b.severity)
            .fold(0.0f64, f64::max);
        
        DatabaseSnapshot {
            recent_operations,
            current_performance,
            bottleneck_severity,
        }
    }
}

impl QueryPerformanceAnalysis {
    pub fn new() -> Self {
        Self {
            slow_queries: Vec::new(),
            query_optimization_opportunities: Vec::new(),
            index_usage_efficiency: 0.8, // Default assumption
            cache_hit_ratio: 0.9,        // Default assumption
        }
    }
}

impl ConnectionEfficiency {
    pub fn new() -> Self {
        Self {
            connection_pool_utilization: 0.5,
            connection_wait_time: Duration::from_millis(10),
            connection_reuse_rate: 0.8,
            idle_connection_percentage: 20.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert!(collector.collection_interval.as_millis() > 0);
    }

    #[tokio::test]
    async fn test_performance_tracking() {
        let collector = MetricsCollector::new();
        
        // Record some performance data
        collector.record_performance(100, 512, 25.0).unwrap();
        collector.record_performance(200, 600, 30.0).unwrap();
        collector.record_performance(300, 550, 28.0).unwrap();
        
        let report = collector.generate_report().unwrap();
        
        assert!(report.performance_metrics.files_processed_per_second.len() > 0);
        assert!(report.performance_metrics.memory_usage_over_time.len() > 0);
        assert!(report.performance_metrics.cpu_usage_over_time.len() > 0);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let collector = MetricsCollector::new();
        
        let context = ErrorContext {
            file_path: Some("/test/file.txt".to_string()),
            operation: "file_read".to_string(),
            system_state: SystemState {
                memory_usage_mb: 512,
                cpu_usage_percent: 25.0,
                files_processed: 100,
                processing_speed: 10.0,
            },
            preceding_events: vec!["Started file processing".to_string()],
        };
        
        collector.record_error(
            ErrorCategory::FileSystem,
            ErrorSeverity::Medium,
            "File not found".to_string(),
            context,
        ).unwrap();
        
        let report = collector.generate_report().unwrap();
        
        assert_eq!(report.error_metrics.error_timeline.len(), 1);
        assert!(report.error_metrics.error_categories.contains_key(&ErrorCategory::FileSystem));
    }

    #[tokio::test]
    async fn test_ux_tracking() {
        let collector = MetricsCollector::new();
        
        let quality_scores = UXQualityScores {
            clarity: 0.8,
            actionability: 0.7,
            completeness: 0.9,
            user_friendliness: 0.8,
        };
        
        collector.record_ux_event(
            UXEventType::ProgressUpdate,
            "Processing file 100 of 1000...".to_string(),
            quality_scores,
        ).unwrap();
        
        let report = collector.generate_report().unwrap();
        
        assert!(report.ux_metrics.progress_reporting_quality.progress_updates.len() > 0);
        assert!(report.ux_metrics.overall_ux_score > 0.0);
    }

    #[tokio::test]
    async fn test_database_tracking() {
        let collector = MetricsCollector::new();
        
        collector.record_database_operation(
            DatabaseOperation::Insert,
            Duration::from_millis(25),
            true,
        ).unwrap();
        
        collector.record_database_operation(
            DatabaseOperation::Select,
            Duration::from_millis(150), // Slow query
            true,
        ).unwrap();
        
        let report = collector.generate_report().unwrap();
        
        assert!(report.database_metrics.operation_timings.contains_key(&DatabaseOperation::Insert));
        assert!(report.database_metrics.operation_timings.contains_key(&DatabaseOperation::Select));
        assert!(report.database_metrics.query_performance_analysis.slow_queries.len() > 0);
    }

    #[test]
    fn test_coefficient_of_variation_calculation() {
        let tracker = PerformanceTracker::new();
        
        // Test with consistent values (low variation)
        let consistent_values = vec![10.0, 10.1, 9.9, 10.0, 10.2];
        let cv = tracker.calculate_coefficient_of_variation(&consistent_values);
        assert!(cv < 0.1);
        
        // Test with highly variable values
        let variable_values = vec![1.0, 10.0, 5.0, 15.0, 2.0];
        let cv = tracker.calculate_coefficient_of_variation(&variable_values);
        assert!(cv > 0.5);
    }

    #[test]
    fn test_error_category_stats() {
        let mut tracker = ErrorTracker::new();
        
        let context = ErrorContext {
            file_path: None,
            operation: "test".to_string(),
            system_state: SystemState {
                memory_usage_mb: 100,
                cpu_usage_percent: 10.0,
                files_processed: 50,
                processing_speed: 5.0,
            },
            preceding_events: Vec::new(),
        };
        
        // Record multiple errors of same category
        for i in 0..5 {
            tracker.record_error(ErrorEvent::new(
                ErrorCategory::FileSystem,
                if i < 2 { ErrorSeverity::Critical } else { ErrorSeverity::Medium },
                format!("Error {}", i),
                context.clone(),
            ));
        }
        
        let stats = tracker.error_categories.get(&ErrorCategory::FileSystem).unwrap();
        assert_eq!(stats.total_count, 5);
        assert_eq!(stats.critical_count, 2);
        assert_eq!(stats.recoverable_count, 3);
    }
}