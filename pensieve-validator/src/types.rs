use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::{DateTime, Utc};

/// Comprehensive directory analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryAnalysis {
    pub total_files: u64,
    pub total_directories: u64,
    pub total_size_bytes: u64,
    pub file_type_distribution: HashMap<String, FileTypeStats>,
    pub size_distribution: SizeDistribution,
    pub depth_analysis: DepthAnalysis,
    pub chaos_indicators: ChaosIndicators,
}

/// File type processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTypeStats {
    pub count: u64,
    pub total_size_bytes: u64,
    pub average_size_bytes: u64,
    pub largest_file: PathBuf,
    pub processing_complexity: ProcessingComplexity,
}

/// Processing complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingComplexity {
    Low,    // Plain text, simple formats
    Medium, // Structured data, common formats
    High,   // Binary, compressed, complex formats
}

/// Technical complexity of messages for UX analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TechnicalComplexity {
    UserFriendly,
    Moderate,
    Technical,
    ExpertLevel,
}

/// File size distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeDistribution {
    pub zero_byte_files: u64,
    pub small_files: u64,      // < 1KB
    pub medium_files: u64,     // 1KB - 1MB
    pub large_files: u64,      // 1MB - 100MB
    pub very_large_files: u64, // > 100MB
    pub largest_file_size: u64,
    pub largest_file_path: PathBuf,
}

/// Directory depth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthAnalysis {
    pub max_depth: usize,
    pub average_depth: f64,
    pub files_by_depth: HashMap<usize, u64>,
    pub deepest_path: PathBuf,
}

/// High-level chaos indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosIndicators {
    pub chaos_score: f64, // 0.0 = clean, 1.0 = maximum chaos
    pub problematic_file_count: u64,
    pub total_file_count: u64,
    pub chaos_percentage: f64,
}

/// Comprehensive directory analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosReport {
    pub files_without_extensions: Vec<PathBuf>,
    pub misleading_extensions: Vec<MisleadingFile>,
    pub unicode_filenames: Vec<UnicodeFile>,
    pub extremely_large_files: Vec<LargeFile>,
    pub zero_byte_files: Vec<PathBuf>,
    pub permission_issues: Vec<PermissionIssue>,
    pub symlink_chains: Vec<SymlinkChain>,
    pub corrupted_files: Vec<CorruptedFile>,
    pub unusual_characters: Vec<UnusualCharacterFile>,
    pub deep_nesting: Vec<DeepNestedFile>,
}

/// File with misleading extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisleadingFile {
    pub path: PathBuf,
    pub claimed_type: String,    // Based on extension
    pub actual_type: String,     // Based on content analysis
    pub confidence: f64,         // 0.0 - 1.0
}

/// File with unicode characters in name
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnicodeFile {
    pub path: PathBuf,
    pub unicode_categories: Vec<String>,
    pub problematic_chars: Vec<char>,
}

/// Large file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeFile {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub size_category: SizeCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SizeCategory {
    Large,      // 100MB - 1GB
    VeryLarge,  // 1GB - 10GB
    Enormous,   // > 10GB
}

/// Permission-related issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionIssue {
    pub path: PathBuf,
    pub issue_type: PermissionIssueType,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PermissionIssueType {
    ReadDenied,
    WriteDenied,
    ExecuteDenied,
    OwnershipIssue,
}

/// Symlink chain information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymlinkChain {
    pub start_path: PathBuf,
    pub chain: Vec<PathBuf>,
    pub chain_length: usize,
    pub is_circular: bool,
    pub final_target: Option<PathBuf>,
}

/// Corrupted or problematic file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorruptedFile {
    pub path: PathBuf,
    pub corruption_type: CorruptionType,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorruptionType {
    UnreadableContent,
    InvalidEncoding,
    TruncatedFile,
    MalformedStructure,
}

/// File with unusual characters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnusualCharacterFile {
    pub path: PathBuf,
    pub unusual_chars: Vec<char>,
    pub char_categories: Vec<String>,
}

/// Deeply nested file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepNestedFile {
    pub path: PathBuf,
    pub depth: usize,
    pub path_length: usize,
}

/// Deduplication return on investment analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct DeduplicationROI {
    pub file_level_duplicates: u64,
    pub storage_saved_bytes: u64,
    pub storage_saved_percentage: f64,
    pub processing_time_saved_seconds: f64,
    pub deduplication_overhead_seconds: f64,
    pub net_benefit_seconds: f64,
    pub paragraph_level_savings: ParagraphDeduplicationSavings,
    pub duplicate_groups: Vec<DuplicateGroup>,
    pub roi_recommendation: ROIRecommendation,
    pub canonical_selection_logic: CanonicalSelectionLogic,
}

/// Paragraph-level deduplication savings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParagraphDeduplicationSavings {
    pub total_paragraphs: u64,
    pub unique_paragraphs: u64,
    pub duplicate_paragraphs: u64,
    pub token_savings: u64,
    pub token_savings_percentage: f64,
    pub processing_time_saved_seconds: f64,
}

/// Group of duplicate files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub canonical_file: PathBuf,
    pub duplicate_files: Vec<PathBuf>,
    pub file_size_bytes: u64,
    pub total_savings_bytes: u64,
    pub selection_reason: String,
}

/// ROI recommendation levels
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum ROIRecommendation {
    HighValue,    // >50% savings
    ModerateValue, // 20-50% savings
    LowValue,     // 5-20% savings
    Negative,     // Overhead exceeds savings
}

/// Logic used for canonical file selection
#[derive(Debug, Serialize, Deserialize)]
pub struct CanonicalSelectionLogic {
    pub primary_criteria: String,
    pub secondary_criteria: Vec<String>,
    pub explanation: String,
}

/// Deduplication analysis configuration
#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    pub enable_file_deduplication: bool,
    pub enable_paragraph_deduplication: bool,
    pub min_file_size_for_analysis: u64,
    pub paragraph_similarity_threshold: f64,
    pub max_files_per_group: usize,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            enable_file_deduplication: true,
            enable_paragraph_deduplication: true,
            min_file_size_for_analysis: 1024, // 1KB minimum
            paragraph_similarity_threshold: 0.95, // 95% similarity
            max_files_per_group: 1000,
        }
    }
}

impl ChaosReport {
    /// Calculate overall chaos metrics
    pub fn calculate_chaos_metrics(&self, total_files: u64) -> ChaosIndicators {
        let problematic_count = self.files_without_extensions.len() as u64
            + self.misleading_extensions.len() as u64
            + self.unicode_filenames.len() as u64
            + self.extremely_large_files.len() as u64
            + self.zero_byte_files.len() as u64
            + self.permission_issues.len() as u64
            + self.symlink_chains.len() as u64
            + self.corrupted_files.len() as u64
            + self.unusual_characters.len() as u64
            + self.deep_nesting.len() as u64;

        let chaos_percentage = if total_files > 0 {
            (problematic_count as f64 / total_files as f64) * 100.0
        } else {
            0.0
        };

        // Chaos score calculation (weighted by severity)
        let chaos_score = self.calculate_weighted_chaos_score(total_files);

        ChaosIndicators {
            chaos_score,
            problematic_file_count: problematic_count,
            total_file_count: total_files,
            chaos_percentage,
        }
    }

    fn calculate_weighted_chaos_score(&self, total_files: u64) -> f64 {
        if total_files == 0 {
            return 0.0;
        }

        let total_files_f64 = total_files as f64;
        
        // Weight different types of chaos by severity
        let weighted_score = 
            (self.corrupted_files.len() as f64 * 1.0) +           // Highest weight
            (self.permission_issues.len() as f64 * 0.9) +
            (self.symlink_chains.len() as f64 * 0.8) +
            (self.misleading_extensions.len() as f64 * 0.7) +
            (self.extremely_large_files.len() as f64 * 0.6) +
            (self.files_without_extensions.len() as f64 * 0.5) +
            (self.unicode_filenames.len() as f64 * 0.4) +
            (self.unusual_characters.len() as f64 * 0.3) +
            (self.deep_nesting.len() as f64 * 0.2) +
            (self.zero_byte_files.len() as f64 * 0.1);            // Lowest weight

        // Normalize to 0.0 - 1.0 range
        (weighted_score / total_files_f64).min(1.0)
    }
}/
// Validation phases for the pensieve validation framework
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationPhase {
    /// Pre-flight directory analysis and chaos detection
    PreFlight,
    /// Reliability testing and crash detection
    Reliability,
    /// Performance benchmarking and scalability analysis
    Performance,
    /// User experience analysis and feedback quality assessment
    UserExperience,
    /// Production intelligence and readiness assessment
    ProductionIntelligence,
}

impl std::fmt::Display for ValidationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreFlight => write!(f, "Pre-Flight Analysis"),
            Self::Reliability => write!(f, "Reliability Testing"),
            Self::Performance => write!(f, "Performance Benchmarking"),
            Self::UserExperience => write!(f, "User Experience Analysis"),
            Self::ProductionIntelligence => write!(f, "Production Intelligence"),
        }
    }
}

/// Complete validation results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub directory_analysis: DirectoryAnalysis,
    pub chaos_report: ChaosReport,
    pub reliability_results: ReliabilityResults,
    pub performance_results: PerformanceResults,
    pub user_experience_results: UXResults,
    pub deduplication_roi: DeduplicationROI,
    pub production_readiness: ProductionReadinessReport,
    pub validation_metadata: ValidationMetadata,
}

/// Reliability testing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityResults {
    pub crash_count: u32,
    pub critical_errors: Vec<String>,
    pub error_recovery_success_rate: f64,
    pub graceful_interruption_handling: bool,
    pub resource_limit_handling: ResourceLimitHandling,
    pub overall_reliability_score: f64,
}

/// Resource limit handling assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimitHandling {
    pub memory_limit_respected: bool,
    pub disk_space_limit_respected: bool,
    pub timeout_handling: bool,
    pub graceful_degradation: bool,
}

/// Performance testing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceResults {
    pub files_per_second: f64,
    pub memory_usage_mb: f64,
    pub peak_memory_mb: f64,
    pub processing_time_seconds: f64,
    pub performance_consistency: f64,
    pub scalability_assessment: ScalabilityAssessment,
    pub bottleneck_analysis: BottleneckAnalysis,
}

/// Scalability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAssessment {
    pub linear_scaling: bool,
    pub performance_degradation_point: Option<u64>,
    pub recommended_max_files: u64,
    pub scaling_guidance: String,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: String,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub io_utilization: f64,
    pub optimization_suggestions: Vec<String>,
}

/// User experience analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXResults {
    pub progress_reporting_quality: f64,
    pub error_message_clarity: f64,
    pub completion_feedback_quality: f64,
    pub interruption_handling_quality: f64,
    pub overall_ux_score: f64,
    pub improvement_recommendations: Vec<UXImprovement>,
}

/// UX improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXImprovement {
    pub category: String,
    pub current_issue: String,
    pub suggested_improvement: String,
    pub impact_level: String,
}

/// Production readiness report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionReadinessReport {
    pub overall_assessment: ProductionReadiness,
    pub reliability_score: f64,
    pub performance_score: f64,
    pub ux_score: f64,
    pub critical_issues: Vec<CriticalIssue>,
    pub improvement_roadmap: Vec<ImprovementItem>,
    pub scaling_guidance: String,
    pub deployment_recommendations: Vec<String>,
}

/// Production readiness assessment levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProductionReadiness {
    Ready,
    ReadyWithCaveats,
    NotReady,
}

/// Critical issue that blocks production readiness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub title: String,
    pub description: String,
    pub impact: String,
    pub affected_scenarios: Vec<String>,
    pub suggested_fix: String,
}

/// Improvement roadmap item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementItem {
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub estimated_effort: String,
    pub expected_impact: String,
}

/// Priority levels for improvements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    pub validation_start_time: chrono::DateTime<chrono::Utc>,
    pub validation_end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub total_duration_seconds: Option<f64>,
    pub pensieve_version: Option<String>,
    pub validator_version: String,
    pub target_directory: PathBuf,
    pub configuration_used: String,
    pub partial_validation: bool,
    pub completed_phases: Vec<ValidationPhase>,
    pub failed_phases: Vec<ValidationPhase>,
}

/// Additional types needed for integration tests

impl ChaosReport {
    /// Get total count of chaos files
    pub fn total_chaos_files(&self) -> usize {
        self.files_without_extensions.len()
            + self.misleading_extensions.len()
            + self.unicode_filenames.len()
            + self.extremely_large_files.len()
            + self.zero_byte_files.len()
            + self.permission_issues.len()
            + self.symlink_chains.len()
            + self.corrupted_files.len()
            + self.unusual_characters.len()
            + self.deep_nesting.len()
    }
}

impl PerformanceResults {
    /// Calculate overall performance score (0.0 to 1.0)
    pub fn overall_performance_score(&self) -> f64 {
        // Simple scoring based on multiple factors
        let speed_score = (self.files_per_second / 100.0).min(1.0);
        let consistency_score = self.performance_consistency;
        let memory_score = (200.0 / self.peak_memory_mb.max(1.0)).min(1.0);
        
        (speed_score + consistency_score + memory_score) / 3.0
    }
}

impl ReliabilityResults {
    /// Calculate overall reliability score (0.0 to 1.0)
    pub fn overall_reliability_score(&self) -> f64 {
        if self.crash_count > 0 {
            return 0.0;
        }
        
        let error_score = if self.critical_errors.is_empty() { 1.0 } else { 0.5 };
        let recovery_score = self.error_recovery_success_rate;
        let interruption_score = if self.graceful_interruption_handling { 1.0 } else { 0.0 };
        
        (error_score + recovery_score + interruption_score) / 3.0
    }
}

impl UXResults {
    /// Calculate overall UX score (0.0 to 10.0)
    pub fn overall_ux_score(&self) -> f64 {
        (self.progress_reporting_quality + 
         self.error_message_clarity + 
         self.completion_feedback_quality + 
         self.interruption_handling_quality) / 4.0
    }
}