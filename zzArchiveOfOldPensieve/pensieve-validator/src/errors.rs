use thiserror::Error;
use std::path::PathBuf;
use std::time::Duration;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Comprehensive error handling for validation framework with structured hierarchy
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    // Critical errors that prevent validation from starting
    #[error("Pensieve binary not found at path: {path}")]
    PensieveBinaryNotFound { 
        path: PathBuf,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Target directory not accessible: {path} - {cause}")]
    DirectoryNotAccessible { 
        path: PathBuf, 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Configuration error: {field} - {message}")]
    ConfigurationError { 
        field: String, 
        message: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    // Runtime errors during validation
    #[error("Pensieve process crashed: {exit_code} - {stderr}")]
    PensieveCrashed { 
        exit_code: i32, 
        stderr: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Validation timeout after {seconds}s")]
    ValidationTimeout { 
        seconds: u64,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Resource limit exceeded: {resource} - {limit}")]
    ResourceLimitExceeded { 
        resource: String, 
        limit: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Process monitoring error: {cause}")]
    ProcessMonitoring { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    // File system and I/O errors
    #[error("File system error: {cause} - {path:?}")]
    FileSystem { 
        cause: String, 
        path: Option<PathBuf>,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Permission denied accessing: {path}")]
    PermissionDenied { 
        path: PathBuf,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Symlink chain too deep: {path} (max depth: {max_depth})")]
    SymlinkChainTooDeep { 
        path: PathBuf, 
        max_depth: usize,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Invalid file path: {path}")]
    InvalidPath { 
        path: PathBuf,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("File type detection failed: {path} - {cause}")]
    FileTypeDetectionFailed { 
        path: PathBuf, 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    // Analysis and processing errors
    #[error("Analysis error: {phase} - {cause}")]
    Analysis { 
        phase: String, 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Chaos detection failed: {cause}")]
    ChaosDetection { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Performance benchmarking failed: {cause}")]
    PerformanceBenchmarking { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Deduplication analysis failed: {cause}")]
    DeduplicationAnalysis { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("UX analysis failed: {cause}")]
    UXAnalysis { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    // Report generation errors
    #[error("Report generation failed: {format} - {cause}")]
    ReportGenerationFailed { 
        format: String, 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Serialization error: {cause}")]
    Serialization { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    // Partial validation errors (allow graceful degradation)
    #[error("Partial validation failure: {completed_phases} of {total_phases} phases completed")]
    PartialValidation { 
        completed_phases: usize, 
        total_phases: usize, 
        failed_phases: Vec<String>,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    // Interruption and cleanup errors
    #[error("Validation interrupted: {reason}")]
    ValidationInterrupted { 
        reason: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
    
    #[error("Cleanup failed: {cause}")]
    CleanupFailed { 
        cause: String,
        #[serde(skip)]
        recovery_strategy: RecoveryStrategy,
    },
}

/// Recovery strategies for different error types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecoveryStrategy {
    /// Error is fatal, cannot recover
    FailFast,
    /// Retry the operation with the same parameters
    Retry { max_attempts: u32, delay: Duration },
    /// Retry with modified parameters
    RetryWithModification { suggestion: String },
    /// Skip this component and continue with partial validation
    SkipAndContinue { impact: String },
    /// Provide manual intervention steps
    ManualIntervention { steps: Vec<String> },
    /// Graceful degradation with reduced functionality
    GracefulDegradation { reduced_functionality: String },
}

/// Error impact assessment for prioritization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorImpact {
    /// Blocks all validation functionality
    Blocker,
    /// Significantly affects validation quality or user experience
    High,
    /// Noticeable but doesn't prevent core functionality
    Medium,
    /// Minor issue with minimal impact
    Low,
}

/// Error category for grouping and analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Configuration or setup issues
    Configuration,
    /// File system access or I/O problems
    FileSystem,
    /// Process execution or monitoring issues
    ProcessExecution,
    /// Analysis or computation failures
    Analysis,
    /// Resource constraints or limits
    ResourceConstraints,
    /// Report generation or output issues
    Reporting,
    /// User interruption or cleanup issues
    Interruption,
}

/// Detailed error information for debugging and reproduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub error: ValidationError,
    pub timestamp: DateTime<Utc>,
    pub category: ErrorCategory,
    pub impact: ErrorImpact,
    pub reproduction_steps: Vec<String>,
    pub debugging_info: DebugInfo,
    pub suggested_fixes: Vec<String>,
    pub related_errors: Vec<String>,
}

/// Debugging information for error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    pub system_info: SystemInfo,
    pub validation_context: ValidationContext,
    pub stack_trace: Option<String>,
    pub environment_variables: std::collections::HashMap<String, String>,
    pub file_system_state: Option<FileSystemState>,
}

/// System information for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub available_memory: u64,
    pub available_disk_space: u64,
    pub cpu_count: usize,
}

/// Validation context when error occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    pub current_phase: String,
    pub target_directory: PathBuf,
    pub pensieve_binary: PathBuf,
    pub config_file: Option<PathBuf>,
    pub elapsed_time: Duration,
    pub processed_files: usize,
    pub current_file: Option<PathBuf>,
}

/// File system state for debugging file-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemState {
    pub current_directory: PathBuf,
    pub target_exists: bool,
    pub target_permissions: Option<String>,
    pub target_size: Option<u64>,
    pub available_space: u64,
}

pub type Result<T> = std::result::Result<T, ValidationError>;
impl 
ValidationError {
    /// Get the recovery strategy for this error
    pub fn recovery_strategy(&self) -> &RecoveryStrategy {
        match self {
            Self::PensieveBinaryNotFound { recovery_strategy, .. } => recovery_strategy,
            Self::DirectoryNotAccessible { recovery_strategy, .. } => recovery_strategy,
            Self::ConfigurationError { recovery_strategy, .. } => recovery_strategy,
            Self::PensieveCrashed { recovery_strategy, .. } => recovery_strategy,
            Self::ValidationTimeout { recovery_strategy, .. } => recovery_strategy,
            Self::ResourceLimitExceeded { recovery_strategy, .. } => recovery_strategy,
            Self::ProcessMonitoring { recovery_strategy, .. } => recovery_strategy,
            Self::FileSystem { recovery_strategy, .. } => recovery_strategy,
            Self::PermissionDenied { recovery_strategy, .. } => recovery_strategy,
            Self::SymlinkChainTooDeep { recovery_strategy, .. } => recovery_strategy,
            Self::InvalidPath { recovery_strategy, .. } => recovery_strategy,
            Self::FileTypeDetectionFailed { recovery_strategy, .. } => recovery_strategy,
            Self::Analysis { recovery_strategy, .. } => recovery_strategy,
            Self::ChaosDetection { recovery_strategy, .. } => recovery_strategy,
            Self::PerformanceBenchmarking { recovery_strategy, .. } => recovery_strategy,
            Self::DeduplicationAnalysis { recovery_strategy, .. } => recovery_strategy,
            Self::UXAnalysis { recovery_strategy, .. } => recovery_strategy,
            Self::ReportGenerationFailed { recovery_strategy, .. } => recovery_strategy,
            Self::Serialization { recovery_strategy, .. } => recovery_strategy,
            Self::PartialValidation { recovery_strategy, .. } => recovery_strategy,
            Self::ValidationInterrupted { recovery_strategy, .. } => recovery_strategy,
            Self::CleanupFailed { recovery_strategy, .. } => recovery_strategy,
        }
    }
    
    /// Get the error category for grouping and analysis
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::PensieveBinaryNotFound { .. } | 
            Self::ConfigurationError { .. } => ErrorCategory::Configuration,
            
            Self::DirectoryNotAccessible { .. } |
            Self::FileSystem { .. } |
            Self::PermissionDenied { .. } |
            Self::SymlinkChainTooDeep { .. } |
            Self::InvalidPath { .. } |
            Self::FileTypeDetectionFailed { .. } => ErrorCategory::FileSystem,
            
            Self::PensieveCrashed { .. } |
            Self::ProcessMonitoring { .. } => ErrorCategory::ProcessExecution,
            
            Self::Analysis { .. } |
            Self::ChaosDetection { .. } |
            Self::PerformanceBenchmarking { .. } |
            Self::DeduplicationAnalysis { .. } |
            Self::UXAnalysis { .. } => ErrorCategory::Analysis,
            
            Self::ValidationTimeout { .. } |
            Self::ResourceLimitExceeded { .. } => ErrorCategory::ResourceConstraints,
            
            Self::ReportGenerationFailed { .. } |
            Self::Serialization { .. } => ErrorCategory::Reporting,
            
            Self::PartialValidation { .. } |
            Self::ValidationInterrupted { .. } |
            Self::CleanupFailed { .. } => ErrorCategory::Interruption,
        }
    }
    
    /// Assess the impact of this error
    pub fn impact(&self) -> ErrorImpact {
        match self {
            Self::PensieveBinaryNotFound { .. } |
            Self::DirectoryNotAccessible { .. } |
            Self::ConfigurationError { .. } |
            Self::PensieveCrashed { .. } => ErrorImpact::Blocker,
            
            Self::ValidationTimeout { .. } |
            Self::ResourceLimitExceeded { .. } |
            Self::ProcessMonitoring { .. } => ErrorImpact::High,
            
            Self::FileSystem { .. } |
            Self::Analysis { .. } |
            Self::ChaosDetection { .. } |
            Self::PerformanceBenchmarking { .. } |
            Self::DeduplicationAnalysis { .. } |
            Self::UXAnalysis { .. } |
            Self::PartialValidation { .. } => ErrorImpact::Medium,
            
            Self::PermissionDenied { .. } |
            Self::SymlinkChainTooDeep { .. } |
            Self::InvalidPath { .. } |
            Self::FileTypeDetectionFailed { .. } |
            Self::ReportGenerationFailed { .. } |
            Self::Serialization { .. } |
            Self::ValidationInterrupted { .. } |
            Self::CleanupFailed { .. } => ErrorImpact::Low,
        }
    }
    
    /// Generate reproduction steps for this error
    pub fn reproduction_steps(&self) -> Vec<String> {
        match self {
            Self::PensieveBinaryNotFound { path, .. } => vec![
                format!("1. Attempt to run validation with pensieve binary at: {}", path.display()),
                "2. Verify the binary exists and has execute permissions".to_string(),
                "3. Check PATH environment variable if using relative path".to_string(),
            ],
            
            Self::DirectoryNotAccessible { path, cause, .. } => vec![
                format!("1. Attempt to access directory: {}", path.display()),
                format!("2. Error encountered: {}", cause),
                "3. Check directory permissions and existence".to_string(),
                "4. Verify parent directory permissions".to_string(),
            ],
            
            Self::PensieveCrashed { exit_code, stderr, .. } => vec![
                "1. Run pensieve with the same parameters".to_string(),
                format!("2. Process exits with code: {}", exit_code),
                format!("3. Error output: {}", stderr),
                "4. Check pensieve logs for additional details".to_string(),
            ],
            
            Self::ValidationTimeout { seconds, .. } => vec![
                format!("1. Start validation process"),
                format!("2. Wait for {} seconds", seconds),
                "3. Process times out without completion".to_string(),
                "4. Check system resources and pensieve responsiveness".to_string(),
            ],
            
            Self::ResourceLimitExceeded { resource, limit, .. } => vec![
                format!("1. Monitor {} usage during validation", resource),
                format!("2. Usage exceeds limit: {}", limit),
                "3. Validation fails or degrades performance".to_string(),
                "4. Check system resource availability".to_string(),
            ],
            
            _ => vec![
                "1. Reproduce the exact validation configuration".to_string(),
                "2. Run validation with verbose logging enabled".to_string(),
                "3. Monitor system resources during execution".to_string(),
                "4. Check validation logs for detailed error information".to_string(),
            ],
        }
    }
    
    /// Generate suggested fixes for this error
    pub fn suggested_fixes(&self) -> Vec<String> {
        match self {
            Self::PensieveBinaryNotFound { path, .. } => vec![
                "Install pensieve binary in the system PATH".to_string(),
                format!("Provide correct path to pensieve binary (currently: {})", path.display()),
                "Build pensieve from source if binary is not available".to_string(),
                "Check pensieve installation documentation".to_string(),
            ],
            
            Self::DirectoryNotAccessible { path, .. } => vec![
                format!("Grant read permissions to directory: {}", path.display()),
                "Run validation with appropriate user privileges".to_string(),
                "Check if directory exists and is not corrupted".to_string(),
                "Verify network connectivity if directory is on remote filesystem".to_string(),
            ],
            
            Self::PensieveCrashed { .. } => vec![
                "Update pensieve to the latest version".to_string(),
                "Check pensieve issue tracker for known bugs".to_string(),
                "Run pensieve with smaller dataset to isolate the issue".to_string(),
                "Enable pensieve debug logging for more details".to_string(),
                "Report crash to pensieve maintainers with reproduction steps".to_string(),
            ],
            
            Self::ValidationTimeout { .. } => vec![
                "Increase validation timeout in configuration".to_string(),
                "Run validation on smaller dataset first".to_string(),
                "Check system resources (CPU, memory, disk I/O)".to_string(),
                "Consider running validation in multiple phases".to_string(),
            ],
            
            Self::ResourceLimitExceeded { resource, .. } => vec![
                format!("Increase {} limit in system configuration", resource),
                "Close other resource-intensive applications".to_string(),
                "Consider running validation on more powerful hardware".to_string(),
                "Enable resource optimization options in pensieve".to_string(),
            ],
            
            Self::PartialValidation { failed_phases, .. } => vec![
                "Review failed phases and address specific issues".to_string(),
                format!("Failed phases: {}", failed_phases.join(", ")),
                "Consider running validation in smaller chunks".to_string(),
                "Check logs for specific errors in each failed phase".to_string(),
            ],
            
            _ => vec![
                "Check validation logs for detailed error information".to_string(),
                "Verify system requirements and dependencies".to_string(),
                "Try running validation with minimal configuration".to_string(),
                "Contact support with error details and system information".to_string(),
            ],
        }
    }
}

/// Error recovery manager for handling validation failures
pub struct ErrorRecoveryManager {
    max_retry_attempts: u32,
    retry_delay: Duration,
    enable_graceful_degradation: bool,
}

impl ErrorRecoveryManager {
    pub fn new() -> Self {
        Self {
            max_retry_attempts: 3,
            retry_delay: Duration::from_secs(1),
            enable_graceful_degradation: true,
        }
    }
    
    pub fn with_retry_config(mut self, max_attempts: u32, delay: Duration) -> Self {
        self.max_retry_attempts = max_attempts;
        self.retry_delay = delay;
        self
    }
    
    pub fn with_graceful_degradation(mut self, enable: bool) -> Self {
        self.enable_graceful_degradation = enable;
        self
    }
    
    /// Attempt to recover from a validation error
    pub async fn attempt_recovery(&self, error: &ValidationError) -> RecoveryAction {
        match error.recovery_strategy() {
            RecoveryStrategy::FailFast => RecoveryAction::Abort,
            
            RecoveryStrategy::Retry { max_attempts, delay } => {
                if *max_attempts > 0 {
                    tokio::time::sleep(*delay).await;
                    RecoveryAction::Retry
                } else {
                    RecoveryAction::Abort
                }
            },
            
            RecoveryStrategy::RetryWithModification { suggestion } => {
                RecoveryAction::RetryWithChanges(suggestion.clone())
            },
            
            RecoveryStrategy::SkipAndContinue { impact } => {
                if self.enable_graceful_degradation {
                    RecoveryAction::SkipComponent(impact.clone())
                } else {
                    RecoveryAction::Abort
                }
            },
            
            RecoveryStrategy::ManualIntervention { steps } => {
                RecoveryAction::RequireManualIntervention(steps.clone())
            },
            
            RecoveryStrategy::GracefulDegradation { reduced_functionality } => {
                if self.enable_graceful_degradation {
                    RecoveryAction::ContinueWithReducedFunctionality(reduced_functionality.clone())
                } else {
                    RecoveryAction::Abort
                }
            },
        }
    }
    
    /// Create a detailed error report for debugging and analysis
    pub fn create_error_details(
        &self, 
        error: ValidationError, 
        context: ValidationContext
    ) -> ErrorDetails {
        ErrorDetails {
            timestamp: Utc::now(),
            category: error.category(),
            impact: error.impact(),
            reproduction_steps: error.reproduction_steps(),
            debugging_info: DebugInfo {
                system_info: self.collect_system_info(),
                validation_context: context,
                stack_trace: None, // Could be populated with backtrace
                environment_variables: std::env::vars().collect(),
                file_system_state: self.collect_file_system_state(&error),
            },
            suggested_fixes: error.suggested_fixes(),
            related_errors: Vec::new(), // Could be populated by error correlation
            error,
        }
    }
    
    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            available_memory: 0, // Would use sysinfo crate in real implementation
            available_disk_space: 0, // Would use sysinfo crate in real implementation
            cpu_count: num_cpus::get(),
        }
    }
    
    fn collect_file_system_state(&self, error: &ValidationError) -> Option<FileSystemState> {
        match error {
            ValidationError::DirectoryNotAccessible { path, .. } |
            ValidationError::PermissionDenied { path, .. } |
            ValidationError::InvalidPath { path, .. } => {
                Some(FileSystemState {
                    current_directory: std::env::current_dir().unwrap_or_default(),
                    target_exists: path.exists(),
                    target_permissions: None, // Would collect actual permissions
                    target_size: path.metadata().ok().map(|m| m.len()),
                    available_space: 0, // Would use statvfs or similar
                })
            },
            _ => None,
        }
    }
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Actions that can be taken in response to errors
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Abort the validation process
    Abort,
    /// Retry the failed operation
    Retry,
    /// Retry with suggested modifications
    RetryWithChanges(String),
    /// Skip the failed component and continue
    SkipComponent(String),
    /// Continue with reduced functionality
    ContinueWithReducedFunctionality(String),
    /// Require manual intervention before continuing
    RequireManualIntervention(Vec<String>),
}

/// Error aggregator for collecting and analyzing multiple errors
#[derive(Debug, Default)]
pub struct ErrorAggregator {
    errors: Vec<ErrorDetails>,
    error_counts: std::collections::HashMap<ErrorCategory, usize>,
    impact_counts: std::collections::HashMap<ErrorImpact, usize>,
}

impl ErrorAggregator {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_error(&mut self, error_details: ErrorDetails) {
        *self.error_counts.entry(error_details.category.clone()).or_insert(0) += 1;
        *self.impact_counts.entry(error_details.impact.clone()).or_insert(0) += 1;
        self.errors.push(error_details);
    }
    
    pub fn get_errors(&self) -> &[ErrorDetails] {
        &self.errors
    }
    
    pub fn get_error_summary(&self) -> ErrorSummary {
        ErrorSummary {
            total_errors: self.errors.len(),
            blocker_count: *self.impact_counts.get(&ErrorImpact::Blocker).unwrap_or(&0),
            high_impact_count: *self.impact_counts.get(&ErrorImpact::High).unwrap_or(&0),
            medium_impact_count: *self.impact_counts.get(&ErrorImpact::Medium).unwrap_or(&0),
            low_impact_count: *self.impact_counts.get(&ErrorImpact::Low).unwrap_or(&0),
            category_breakdown: self.error_counts.clone(),
            most_common_category: self.get_most_common_category(),
            critical_errors: self.get_critical_errors(),
        }
    }
    
    fn get_most_common_category(&self) -> Option<ErrorCategory> {
        self.error_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(category, _)| category.clone())
    }
    
    fn get_critical_errors(&self) -> Vec<String> {
        self.errors
            .iter()
            .filter(|e| matches!(e.impact, ErrorImpact::Blocker | ErrorImpact::High))
            .map(|e| e.error.to_string())
            .collect()
    }
}

/// Convenience helpers for constructing common ValidationErrors
impl ValidationError {
    pub fn fs_io<E: std::fmt::Display>(err: E, path: Option<PathBuf>) -> Self {
        ValidationError::FileSystem {
            cause: err.to_string(),
            path,
            recovery_strategy: RecoveryStrategy::SkipAndContinue {
                impact: "File skipped".into(),
            },
        }
    }
    pub fn cfg(field: &str, msg: &str) -> Self {
        ValidationError::ConfigurationError {
            field: field.into(),
            message: msg.into(),
            recovery_strategy: RecoveryStrategy::ManualIntervention {
                steps: vec!["Fix configuration".into()],
            },
        }
    }
    pub fn proc_mon(msg: &str) -> Self {
        ValidationError::ProcessMonitoring {
            cause: msg.into(),
            recovery_strategy: RecoveryStrategy::Retry {
                max_attempts: 1,
                delay: Duration::from_secs(5),
            },
        }
    }
}

/// Summary of errors for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    pub total_errors: usize,
    pub blocker_count: usize,
    pub high_impact_count: usize,
    pub medium_impact_count: usize,
    pub low_impact_count: usize,
    pub category_breakdown: std::collections::HashMap<ErrorCategory, usize>,
    pub most_common_category: Option<ErrorCategory>,
    pub critical_errors: Vec<String>,
}
/// Helper functions for creating errors with appropriate recovery strategies
impl ValidationError {
    pub fn pensieve_binary_not_found(path: PathBuf) -> Self {
        Self::PensieveBinaryNotFound {
            path,
            recovery_strategy: RecoveryStrategy::ManualIntervention {
                steps: vec![
                    "Install pensieve binary".to_string(),
                    "Add pensieve to PATH".to_string(),
                    "Verify binary has execute permissions".to_string(),
                ],
            },
        }
    }
    
    pub fn directory_not_accessible(path: PathBuf, cause: String) -> Self {
        Self::DirectoryNotAccessible {
            path,
            cause,
            recovery_strategy: RecoveryStrategy::ManualIntervention {
                steps: vec![
                    "Check directory permissions".to_string(),
                    "Verify directory exists".to_string(),
                    "Run with appropriate user privileges".to_string(),
                ],
            },
        }
    }
    
    pub fn pensieve_crashed(exit_code: i32, stderr: String) -> Self {
        Self::PensieveCrashed {
            exit_code,
            stderr,
            recovery_strategy: RecoveryStrategy::Retry {
                max_attempts: 2,
                delay: Duration::from_secs(5),
            },
        }
    }
    
    pub fn validation_timeout(seconds: u64) -> Self {
        Self::ValidationTimeout {
            seconds,
            recovery_strategy: RecoveryStrategy::RetryWithModification {
                suggestion: "Increase timeout or reduce dataset size".to_string(),
            },
        }
    }
    
    pub fn resource_limit_exceeded(resource: String, limit: String) -> Self {
        Self::ResourceLimitExceeded {
            resource,
            limit,
            recovery_strategy: RecoveryStrategy::GracefulDegradation {
                reduced_functionality: "Continue with reduced resource usage".to_string(),
            },
        }
    }
    
    pub fn file_system_error(cause: String, path: Option<PathBuf>) -> Self {
        Self::FileSystem {
            cause,
            path,
            recovery_strategy: RecoveryStrategy::SkipAndContinue {
                impact: "File will be excluded from analysis".to_string(),
            },
        }
    }
    
    pub fn permission_denied(path: PathBuf) -> Self {
        Self::PermissionDenied {
            path,
            recovery_strategy: RecoveryStrategy::SkipAndContinue {
                impact: "File will be excluded from analysis".to_string(),
            },
        }
    }
    
    pub fn analysis_error(phase: String, cause: String) -> Self {
        Self::Analysis {
            phase,
            cause,
            recovery_strategy: RecoveryStrategy::SkipAndContinue {
                impact: "Analysis phase will be marked as incomplete".to_string(),
            },
        }
    }
    
    pub fn partial_validation(completed_phases: usize, total_phases: usize, failed_phases: Vec<String>) -> Self {
        Self::PartialValidation {
            completed_phases,
            total_phases,
            failed_phases,
            recovery_strategy: RecoveryStrategy::GracefulDegradation {
                reduced_functionality: "Generate report with available data".to_string(),
            },
        }
    }
    
    pub fn validation_interrupted(reason: String) -> Self {
        Self::ValidationInterrupted {
            reason,
            recovery_strategy: RecoveryStrategy::ManualIntervention {
                steps: vec![
                    "Review partial results".to_string(),
                    "Restart validation if needed".to_string(),
                    "Check for cleanup requirements".to_string(),
                ],
            },
        }
    }
}

/// Convert standard I/O errors to ValidationError with appropriate recovery strategies
impl From<std::io::Error> for ValidationError {
    fn from(error: std::io::Error) -> Self {
        let recovery_strategy = match error.kind() {
            std::io::ErrorKind::NotFound => RecoveryStrategy::SkipAndContinue {
                impact: "File will be excluded from analysis".to_string(),
            },
            std::io::ErrorKind::PermissionDenied => RecoveryStrategy::SkipAndContinue {
                impact: "File will be excluded from analysis".to_string(),
            },
            std::io::ErrorKind::TimedOut => RecoveryStrategy::Retry {
                max_attempts: 3,
                delay: Duration::from_secs(1),
            },
            _ => RecoveryStrategy::FailFast,
        };
        
        Self::FileSystem {
            cause: error.to_string(),
            path: None,
            recovery_strategy,
        }
    }
}

/// Convert serde JSON errors to ValidationError
impl From<serde_json::Error> for ValidationError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization {
            cause: error.to_string(),
            recovery_strategy: RecoveryStrategy::SkipAndContinue {
                impact: "Data will be excluded from serialized output".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_error_categorization() {
        let error = ValidationError::pensieve_binary_not_found(PathBuf::from("/usr/bin/pensieve"));
        assert_eq!(error.category(), ErrorCategory::Configuration);
        assert_eq!(error.impact(), ErrorImpact::Blocker);
    }
    
    #[test]
    fn test_recovery_strategy() {
        let error = ValidationError::validation_timeout(300);
        match error.recovery_strategy() {
            RecoveryStrategy::RetryWithModification { suggestion } => {
                assert!(suggestion.contains("timeout"));
            },
            _ => panic!("Expected RetryWithModification strategy"),
        }
    }
    
    #[test]
    fn test_reproduction_steps() {
        let error = ValidationError::pensieve_crashed(1, "Segmentation fault".to_string());
        let steps = error.reproduction_steps();
        assert!(!steps.is_empty());
        assert!(steps[0].contains("Run pensieve"));
    }
    
    #[test]
    fn test_suggested_fixes() {
        let error = ValidationError::resource_limit_exceeded(
            "memory".to_string(), 
            "8GB".to_string()
        );
        let fixes = error.suggested_fixes();
        assert!(!fixes.is_empty());
        assert!(fixes.iter().any(|fix| fix.contains("memory")));
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
        
        let error1 = ValidationError::pensieve_binary_not_found(PathBuf::from("/usr/bin/pensieve"));
        let error2 = ValidationError::validation_timeout(300);
        
        let details1 = recovery_manager.create_error_details(error1, context.clone());
        let details2 = recovery_manager.create_error_details(error2, context);
        
        aggregator.add_error(details1);
        aggregator.add_error(details2);
        
        let summary = aggregator.get_error_summary();
        assert_eq!(summary.total_errors, 2);
        assert_eq!(summary.blocker_count, 1);
        assert_eq!(summary.high_impact_count, 1);
    }
    
    #[tokio::test]
    async fn test_recovery_manager() {
        let manager = ErrorRecoveryManager::new();
        let error = ValidationError::validation_timeout(300);
        
        let action = manager.attempt_recovery(&error).await;
        match action {
            RecoveryAction::RetryWithChanges(suggestion) => {
                assert!(suggestion.contains("timeout"));
            },
            _ => panic!("Expected RetryWithChanges action"),
        }
    }
}