use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use crate::validation_orchestrator::{ValidationOrchestratorConfig, PerformanceThresholds};
use crate::pensieve_runner::PensieveConfig;
use crate::process_monitor::MonitoringConfig;
use crate::reliability_validator::ReliabilityConfig;

/// Complete CLI configuration that can be loaded from TOML files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Validation configuration
    pub validation: ValidationConfig,
    
    /// Pensieve-specific configuration
    pub pensieve: PensieveConfigToml,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfigToml,
    
    /// Performance thresholds
    pub performance: PerformanceThresholdsToml,
    
    /// Output configuration
    pub output: OutputConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Validation-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable real-time analysis during validation
    pub enable_real_time_analysis: bool,
    
    /// Metrics collection interval in milliseconds
    pub metrics_collection_interval_ms: u64,
    
    /// Enable chaos detection
    pub enable_chaos_detection: bool,
    
    /// Enable deduplication analysis
    pub enable_deduplication_analysis: bool,
    
    /// Enable UX analysis
    pub enable_ux_analysis: bool,
    
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    
    /// Performance benchmarking configuration
    pub benchmarking: BenchmarkingConfig,
}

/// Performance benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    /// Enable baseline establishment
    pub enable_baseline_establishment: bool,
    
    /// Enable degradation detection
    pub enable_degradation_detection: bool,
    
    /// Enable scalability testing
    pub enable_scalability_testing: bool,
    
    /// Enable memory analysis
    pub enable_memory_analysis: bool,
    
    /// Enable database profiling
    pub enable_database_profiling: bool,
    
    /// Number of benchmark iterations
    pub benchmark_iterations: u32,
    
    /// Number of warmup iterations
    pub warmup_iterations: u32,
    
    /// Timeout in seconds
    pub timeout_seconds: u64,
    
    /// Memory sampling interval in milliseconds
    pub memory_sampling_interval_ms: u64,
    
    /// Performance degradation thresholds
    pub degradation_thresholds: DegradationThresholds,
}

/// Performance degradation thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationThresholds {
    /// Maximum allowed files per second degradation (0.2 = 20%)
    pub max_files_per_second_degradation: f64,
    
    /// Maximum memory growth rate in MB per second
    pub max_memory_growth_rate_mb_per_sec: f64,
    
    /// Maximum database operation time in milliseconds
    pub max_database_operation_time_ms: u64,
    
    /// Minimum CPU efficiency score (0.0 - 1.0)
    pub min_cpu_efficiency_score: f64,
    
    /// Maximum memory leak rate in MB per hour
    pub max_memory_leak_rate_mb_per_hour: f64,
}

/// Pensieve configuration for TOML serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PensieveConfigToml {
    /// Path to pensieve binary
    pub binary_path: Option<PathBuf>,
    
    /// Timeout in seconds
    pub timeout_seconds: Option<u64>,
    
    /// Memory limit in MB
    pub memory_limit_mb: Option<u64>,
    
    /// Enable deduplication in pensieve
    pub enable_deduplication: Option<bool>,
    
    /// Enable verbose output from pensieve
    pub verbose_output: Option<bool>,
    
    /// Additional pensieve arguments
    pub additional_args: Option<Vec<String>>,
}

/// Monitoring configuration for TOML serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfigToml {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: Option<u64>,
    
    /// Memory threshold percentage
    pub memory_threshold_percent: Option<f64>,
    
    /// CPU threshold percentage
    pub cpu_threshold_percent: Option<f64>,
    
    /// Disk threshold percentage
    pub disk_threshold_percent: Option<f64>,
    
    /// Temperature threshold in Celsius
    pub temperature_threshold_celsius: Option<f32>,
    
    /// Enable detailed disk monitoring
    pub enable_detailed_monitoring: Option<bool>,
    
    /// Enable network monitoring
    pub enable_network_monitoring: Option<bool>,
    
    /// Enable thermal monitoring
    pub enable_thermal_monitoring: Option<bool>,
}

/// Performance thresholds for TOML serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholdsToml {
    /// Minimum files per second
    pub min_files_per_second: Option<f64>,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<u64>,
    
    /// Maximum CPU usage percentage
    pub max_cpu_percent: Option<f32>,
    
    /// Maximum error rate per minute
    pub max_error_rate_per_minute: Option<f64>,
    
    /// Minimum UX score (0.0 - 10.0)
    pub min_ux_score: Option<f64>,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory for reports
    pub output_directory: Option<PathBuf>,
    
    /// Enable JSON output
    pub enable_json_output: bool,
    
    /// Enable HTML report generation
    pub enable_html_reports: bool,
    
    /// Enable CSV export
    pub enable_csv_export: bool,
    
    /// Include detailed metrics in output
    pub include_detailed_metrics: bool,
    
    /// Include raw data in output
    pub include_raw_data: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (error, warn, info, debug, trace)
    pub level: String,
    
    /// Enable file logging
    pub enable_file_logging: bool,
    
    /// Log file path
    pub log_file_path: Option<PathBuf>,
    
    /// Enable structured logging (JSON format)
    pub enable_structured_logging: bool,
    
    /// Enable progress reporting
    pub enable_progress_reporting: bool,
    
    /// Progress update interval in milliseconds
    pub progress_update_interval_ms: u64,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            pensieve: PensieveConfigToml::default(),
            monitoring: MonitoringConfigToml::default(),
            performance: PerformanceThresholdsToml::default(),
            output: OutputConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_real_time_analysis: true,
            metrics_collection_interval_ms: 500,
            enable_chaos_detection: true,
            enable_deduplication_analysis: true,
            enable_ux_analysis: true,
            enable_performance_profiling: true,
            benchmarking: BenchmarkingConfig::default(),
        }
    }
}

impl Default for BenchmarkingConfig {
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
            degradation_thresholds: DegradationThresholds::default(),
        }
    }
}

impl Default for DegradationThresholds {
    fn default() -> Self {
        Self {
            max_files_per_second_degradation: 0.2, // 20%
            max_memory_growth_rate_mb_per_sec: 10.0,
            max_database_operation_time_ms: 1000,
            min_cpu_efficiency_score: 0.6,
            max_memory_leak_rate_mb_per_hour: 50.0,
        }
    }
}

impl Default for PensieveConfigToml {
    fn default() -> Self {
        Self {
            binary_path: None,
            timeout_seconds: None,
            memory_limit_mb: None,
            enable_deduplication: None,
            verbose_output: None,
            additional_args: None,
        }
    }
}

impl Default for MonitoringConfigToml {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: None,
            memory_threshold_percent: None,
            cpu_threshold_percent: None,
            disk_threshold_percent: None,
            temperature_threshold_celsius: None,
            enable_detailed_monitoring: None,
            enable_network_monitoring: None,
            enable_thermal_monitoring: None,
        }
    }
}

impl Default for PerformanceThresholdsToml {
    fn default() -> Self {
        Self {
            min_files_per_second: None,
            max_memory_mb: None,
            max_cpu_percent: None,
            max_error_rate_per_minute: None,
            min_ux_score: None,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_directory: None,
            enable_json_output: true,
            enable_html_reports: true,
            enable_csv_export: false,
            include_detailed_metrics: true,
            include_raw_data: false,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            enable_file_logging: false,
            log_file_path: None,
            enable_structured_logging: false,
            enable_progress_reporting: true,
            progress_update_interval_ms: 1000,
        }
    }
}

impl CliConfig {
    /// Load configuration from a TOML file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: CliConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Generate a default configuration file
    pub fn generate_default_config_file(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let default_config = Self::default();
        default_config.save_to_file(path)?;
        Ok(())
    }
    
    /// Convert to ValidationOrchestratorConfig
    pub fn to_validation_orchestrator_config(&self) -> ValidationOrchestratorConfig {
        ValidationOrchestratorConfig {
            pensieve_config: self.to_pensieve_config(),
            benchmark_config: self.to_benchmark_config(),
            monitoring_config: self.to_monitoring_config(),
            reliability_config: ReliabilityConfig::default(),
            metrics_collection_interval_ms: self.validation.metrics_collection_interval_ms,
            enable_real_time_analysis: self.validation.enable_real_time_analysis,
            performance_thresholds: self.to_performance_thresholds(),
            enable_checkpointing: true,
            checkpoint_directory: PathBuf::from("./validation_checkpoints"),
            enable_parallel_execution: true,
            max_parallel_phases: 2,
            phase_timeout_seconds: 3600,
            enable_error_recovery: true,
            max_retry_attempts: 3,
        }
    }
    
    /// Convert to PensieveConfig
    pub fn to_pensieve_config(&self) -> PensieveConfig {
        PensieveConfig {
            binary_path: self.pensieve.binary_path.clone()
                .unwrap_or_else(|| PathBuf::from("pensieve")),
            timeout_seconds: self.pensieve.timeout_seconds.unwrap_or(3600),
            memory_limit_mb: self.pensieve.memory_limit_mb.unwrap_or(8192),
            output_database_path: PathBuf::from("validation_results.db"),
            enable_deduplication: self.pensieve.enable_deduplication.unwrap_or(true),
            verbose_output: self.pensieve.verbose_output.unwrap_or(false),
        }
    }
    
    /// Convert to MonitoringConfig
    pub fn to_monitoring_config(&self) -> MonitoringConfig {
        MonitoringConfig {
            interval_ms: self.monitoring.monitoring_interval_ms.unwrap_or(500),
            memory_threshold_percent: self.monitoring.memory_threshold_percent.unwrap_or(80.0),
            cpu_threshold_percent: self.monitoring.cpu_threshold_percent.unwrap_or(90.0),
            disk_threshold_percent: self.monitoring.disk_threshold_percent.unwrap_or(85.0),
            temperature_threshold_celsius: self.monitoring.temperature_threshold_celsius.unwrap_or(80.0),
            enable_detailed_disk_monitoring: self.monitoring.enable_detailed_monitoring.unwrap_or(true),
            enable_network_monitoring: self.monitoring.enable_network_monitoring.unwrap_or(true),
            enable_thermal_monitoring: self.monitoring.enable_thermal_monitoring.unwrap_or(true),
        }
    }
    
    /// Convert to PerformanceThresholds
    pub fn to_performance_thresholds(&self) -> PerformanceThresholds {
        PerformanceThresholds {
            min_files_per_second: self.performance.min_files_per_second.unwrap_or(1.0),
            max_memory_mb: self.performance.max_memory_mb.unwrap_or(8192),
            max_cpu_percent: self.performance.max_cpu_percent.unwrap_or(80.0),
            max_error_rate_per_minute: self.performance.max_error_rate_per_minute.unwrap_or(5.0),
            min_ux_score: self.performance.min_ux_score.unwrap_or(7.0),
        }
    }

    /// Convert to BenchmarkConfig
    pub fn to_benchmark_config(&self) -> crate::performance_benchmarker::BenchmarkConfig {
        use crate::performance_benchmarker::{BenchmarkConfig, PerformanceThresholds as BenchmarkThresholds};
        
        BenchmarkConfig {
            enable_baseline_establishment: self.validation.benchmarking.enable_baseline_establishment,
            enable_degradation_detection: self.validation.benchmarking.enable_degradation_detection,
            enable_scalability_testing: self.validation.benchmarking.enable_scalability_testing,
            enable_memory_analysis: self.validation.benchmarking.enable_memory_analysis,
            enable_database_profiling: self.validation.benchmarking.enable_database_profiling,
            benchmark_iterations: self.validation.benchmarking.benchmark_iterations,
            warmup_iterations: self.validation.benchmarking.warmup_iterations,
            timeout_seconds: self.validation.benchmarking.timeout_seconds,
            memory_sampling_interval_ms: self.validation.benchmarking.memory_sampling_interval_ms,
            performance_thresholds: BenchmarkThresholds {
                max_files_per_second_degradation: self.validation.benchmarking.degradation_thresholds.max_files_per_second_degradation,
                max_memory_growth_rate_mb_per_sec: self.validation.benchmarking.degradation_thresholds.max_memory_growth_rate_mb_per_sec,
                max_database_operation_time_ms: self.validation.benchmarking.degradation_thresholds.max_database_operation_time_ms,
                min_cpu_efficiency_score: self.validation.benchmarking.degradation_thresholds.min_cpu_efficiency_score,
                max_memory_leak_rate_mb_per_hour: self.validation.benchmarking.degradation_thresholds.max_memory_leak_rate_mb_per_hour,
            },
        }
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate pensieve binary path
        if let Some(binary_path) = &self.pensieve.binary_path {
            if !binary_path.exists() && binary_path != &PathBuf::from("pensieve") {
                return Err(format!("Pensieve binary not found at: {:?}", binary_path));
            }
        }
        
        // Validate performance thresholds
        if let Some(min_files_per_second) = self.performance.min_files_per_second {
            if min_files_per_second <= 0.0 {
                return Err("min_files_per_second must be positive".to_string());
            }
        }
        
        if let Some(max_memory_mb) = self.performance.max_memory_mb {
            if max_memory_mb == 0 {
                return Err("max_memory_mb must be positive".to_string());
            }
        }
        
        if let Some(max_cpu_percent) = self.performance.max_cpu_percent {
            if max_cpu_percent <= 0.0 || max_cpu_percent > 100.0 {
                return Err("max_cpu_percent must be between 0 and 100".to_string());
            }
        }
        
        // Validate logging level
        match self.logging.level.to_lowercase().as_str() {
            "error" | "warn" | "info" | "debug" | "trace" => {},
            _ => return Err(format!("Invalid log level: {}", self.logging.level)),
        }
        
        // Validate intervals
        if self.validation.metrics_collection_interval_ms == 0 {
            return Err("metrics_collection_interval_ms must be positive".to_string());
        }
        
        if self.logging.progress_update_interval_ms == 0 {
            return Err("progress_update_interval_ms must be positive".to_string());
        }
        
        Ok(())
    }
}

/// Configuration validation errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },
    
    #[error("Invalid configuration format: {message}")]
    InvalidFormat { message: String },
    
    #[error("Configuration validation failed: {message}")]
    ValidationFailed { message: String },
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("TOML parsing error: {0}")]
    TomlError(#[from] toml::de::Error),
    
    #[error("TOML serialization error: {0}")]
    TomlSerError(#[from] toml::ser::Error),
}