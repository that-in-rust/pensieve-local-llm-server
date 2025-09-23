use crate::errors::{ValidationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, System};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command as TokioCommand;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::timeout;

/// Configuration for pensieve execution and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PensieveConfig {
    pub binary_path: PathBuf,
    pub timeout_seconds: u64,
    pub memory_limit_mb: u64,
    pub output_database_path: PathBuf,
    pub enable_deduplication: bool,
    pub verbose_output: bool,
}

impl Default for PensieveConfig {
    fn default() -> Self {
        Self {
            binary_path: PathBuf::from("pensieve"),
            timeout_seconds: 3600, // 1 hour default
            memory_limit_mb: 8192,  // 8GB default
            output_database_path: PathBuf::from("pensieve_validation.db"),
            enable_deduplication: true,
            verbose_output: false,
        }
    }
}

/// Real-time memory usage reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReading {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub timestamp_secs: u64, // Unix timestamp for serialization
    pub memory_mb: u64,
    pub virtual_memory_mb: u64,
    pub cpu_usage_percent: f32,
}

impl MemoryReading {
    pub fn new(memory_mb: u64, virtual_memory_mb: u64, cpu_usage_percent: f32) -> Self {
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
            cpu_usage_percent,
        }
    }
}

/// Process execution results with comprehensive monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PensieveExecutionResults {
    pub exit_code: Option<i32>,
    pub execution_time: Duration,
    pub peak_memory_mb: u64,
    pub average_memory_mb: u64,
    pub cpu_usage_stats: CpuUsageStats,
    pub output_analysis: OutputAnalysis,
    pub performance_metrics: PerformanceMetrics,
    pub error_summary: ErrorSummary,
    pub resource_usage: ResourceUsage,
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuUsageStats {
    pub peak_cpu_percent: f32,
    pub average_cpu_percent: f32,
    pub cpu_time_user: Duration,
    pub cpu_time_system: Duration,
}

/// Analysis of pensieve's stdout/stderr output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputAnalysis {
    pub total_lines: u64,
    pub error_lines: u64,
    pub warning_lines: u64,
    pub progress_updates: u64,
    pub files_processed: u64,
    pub duplicates_found: u64,
    pub processing_speed_files_per_second: f64,
    pub key_messages: Vec<String>,
}

/// Performance metrics extracted from pensieve output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub files_per_second: f64,
    pub bytes_per_second: u64,
    pub database_operations_per_second: f64,
    pub memory_efficiency_score: f64, // 0.0 - 1.0
    pub processing_consistency: f64,   // 0.0 - 1.0 (lower variance = higher consistency)
}

/// Summary of errors encountered during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    pub total_errors: u64,
    pub error_categories: HashMap<String, u64>,
    pub critical_errors: Vec<String>,
    pub recoverable_errors: Vec<String>,
    pub error_rate_per_minute: f64,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub disk_io_read_bytes: u64,
    pub disk_io_write_bytes: u64,
    pub network_io_bytes: u64,
    pub file_handles_used: u64,
    pub thread_count: u64,
}

/// Wrapper for running pensieve with comprehensive monitoring
pub struct PensieveRunner {
    config: PensieveConfig,
    system: Arc<Mutex<System>>,
}

impl PensieveRunner {
    /// Create a new PensieveRunner with the given configuration
    pub fn new(config: PensieveConfig) -> Self {
        Self {
            config,
            system: Arc::new(Mutex::new(System::new_all())),
        }
    }

    /// Run pensieve with full monitoring and intelligence collection
    pub async fn run_with_monitoring(
        &self,
        target_dir: &Path,
    ) -> Result<PensieveExecutionResults> {
        let start_time = Instant::now();
        
        // Validate inputs
        self.validate_inputs(target_dir)?;
        
        // Spawn pensieve process
        let mut child = self.spawn_pensieve_process(target_dir).await?;
        let process_id = child.id().ok_or_else(|| ValidationError::ConfigurationError {
            field: "process_id".to_string(),
            message: "Failed to get process ID".to_string(),
        })?;

        // Start monitoring tasks
        let (memory_tx, memory_rx) = mpsc::channel(1000);
        let (output_tx, output_rx) = mpsc::channel(1000);
        
        let memory_monitor = self.start_memory_monitoring(process_id, memory_tx);
        let output_monitor = self.start_output_monitoring(&mut child, output_tx).await?;
        
        // Wait for completion with timeout
        let execution_result = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            child.wait()
        ).await;

        // Stop monitoring and collect results
        memory_monitor.abort();
        output_monitor.abort();
        
        let execution_time = start_time.elapsed();
        
        // Process results
        let exit_status = match execution_result {
            Ok(Ok(status)) => status.code(),
            Ok(Err(e)) => return Err(ValidationError::FileSystem(e)),
            Err(_) => {
                // Timeout occurred - kill the process
                let _ = child.kill().await;
                return Err(ValidationError::ConfigurationError {
                    field: "timeout".to_string(),
                    message: format!("Process timed out after {} seconds", self.config.timeout_seconds),
                });
            }
        };

        // Collect monitoring data
        let memory_readings = self.collect_memory_readings(memory_rx).await;
        let output_lines = self.collect_output_lines(output_rx).await;
        
        // Analyze results
        let results = self.analyze_execution_results(
            exit_status,
            execution_time,
            memory_readings,
            output_lines,
        ).await?;

        Ok(results)
    }

    /// Validate input parameters before execution
    fn validate_inputs(&self, target_dir: &Path) -> Result<()> {
        // Check if pensieve binary exists
        if !self.config.binary_path.exists() {
            return Err(ValidationError::ConfigurationError {
                field: "binary_path".to_string(),
                message: format!("Pensieve binary not found at: {:?}", self.config.binary_path),
            });
        }

        // Check if target directory exists and is accessible
        if !target_dir.exists() {
            return Err(ValidationError::DirectoryNotAccessible {
                path: target_dir.to_path_buf(),
                cause: "Directory does not exist".to_string(),
            });
        }

        if !target_dir.is_dir() {
            return Err(ValidationError::DirectoryNotAccessible {
                path: target_dir.to_path_buf(),
                cause: "Path is not a directory".to_string(),
            });
        }

        // Test read access
        match std::fs::read_dir(target_dir) {
            Ok(_) => Ok(()),
            Err(e) => Err(ValidationError::DirectoryNotAccessible {
                path: target_dir.to_path_buf(),
                cause: e.to_string(),
            }),
        }
    }

    /// Spawn pensieve process with appropriate arguments
    async fn spawn_pensieve_process(&self, target_dir: &Path) -> Result<tokio::process::Child> {
        let mut cmd = TokioCommand::new(&self.config.binary_path);
        
        // Build command arguments
        cmd.arg("scan")
            .arg(target_dir)
            .arg("--database")
            .arg(&self.config.output_database_path);

        if self.config.enable_deduplication {
            cmd.arg("--deduplicate");
        }

        if self.config.verbose_output {
            cmd.arg("--verbose");
        }

        // Configure stdio for monitoring
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        // Spawn the process
        cmd.spawn().map_err(|e| ValidationError::FileSystem(e))
    }

    /// Start memory and CPU monitoring in a background task
    fn start_memory_monitoring(
        &self,
        process_id: u32,
        tx: mpsc::Sender<MemoryReading>,
    ) -> JoinHandle<()> {
        let system = Arc::clone(&self.system);
        let memory_limit = self.config.memory_limit_mb;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            let pid = Pid::from(process_id as usize);
            
            loop {
                interval.tick().await;
                
                // Update system information
                {
                    let mut sys = system.lock().unwrap();
                    sys.refresh_process(pid);
                }
                
                // Get process information
                let reading = {
                    let sys = system.lock().unwrap();
                    if let Some(process) = sys.process(pid) {
                        let memory_kb = process.memory();
                        let virtual_memory_kb = process.virtual_memory();
                        let cpu_usage = process.cpu_usage();
                        
                        Some(MemoryReading::new(
                            memory_kb / 1024,
                            virtual_memory_kb / 1024,
                            cpu_usage,
                        ))
                    } else {
                        None // Process ended
                    }
                };
                
                match reading {
                    Some(reading) => {
                        // Check memory limit
                        if reading.memory_mb > memory_limit {
                            eprintln!("WARNING: Process memory usage ({} MB) exceeds limit ({} MB)", 
                                     reading.memory_mb, memory_limit);
                        }
                        
                        if tx.send(reading).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    None => break, // Process ended
                }
            }
        })
    }

    /// Start output monitoring for stdout and stderr
    async fn start_output_monitoring(
        &self,
        child: &mut tokio::process::Child,
        tx: mpsc::Sender<String>,
    ) -> Result<JoinHandle<()>> {
        let stdout = child.stdout.take().ok_or_else(|| ValidationError::ConfigurationError {
            field: "stdout".to_string(),
            message: "Failed to capture stdout".to_string(),
        })?;
        
        let stderr = child.stderr.take().ok_or_else(|| ValidationError::ConfigurationError {
            field: "stderr".to_string(),
            message: "Failed to capture stderr".to_string(),
        })?;

        let tx_clone = tx.clone();
        
        // Monitor stdout
        let stdout_task = tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send(format!("STDOUT: {}", line)).await.is_err() {
                    break;
                }
            }
        });

        // Monitor stderr
        let stderr_task = tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            
            while let Ok(Some(line)) = lines.next_line().await {
                if tx_clone.send(format!("STDERR: {}", line)).await.is_err() {
                    break;
                }
            }
        });

        // Return a combined task handle
        Ok(tokio::spawn(async move {
            let _ = tokio::join!(stdout_task, stderr_task);
        }))
    }

    /// Collect memory readings from the monitoring channel
    async fn collect_memory_readings(
        &self,
        mut rx: mpsc::Receiver<MemoryReading>,
    ) -> Vec<MemoryReading> {
        let mut readings = Vec::new();
        
        // Give a short time for any remaining readings
        let timeout_duration = Duration::from_millis(100);
        
        while let Ok(Some(reading)) = timeout(timeout_duration, rx.recv()).await {
            readings.push(reading);
        }
        
        readings
    }

    /// Collect output lines from the monitoring channel
    async fn collect_output_lines(&self, mut rx: mpsc::Receiver<String>) -> Vec<String> {
        let mut lines = Vec::new();
        
        // Give a short time for any remaining output
        let timeout_duration = Duration::from_millis(100);
        
        while let Ok(Some(line)) = timeout(timeout_duration, rx.recv()).await {
            lines.push(line);
        }
        
        lines
    }

    /// Analyze execution results and generate comprehensive report
    async fn analyze_execution_results(
        &self,
        exit_code: Option<i32>,
        execution_time: Duration,
        memory_readings: Vec<MemoryReading>,
        output_lines: Vec<String>,
    ) -> Result<PensieveExecutionResults> {
        // Analyze memory usage
        let (peak_memory_mb, average_memory_mb, cpu_stats) = self.analyze_memory_usage(&memory_readings);
        
        // Analyze output
        let output_analysis = self.analyze_output(&output_lines);
        
        // Calculate performance metrics
        let performance_metrics = self.calculate_performance_metrics(&memory_readings, &output_analysis, execution_time);
        
        // Analyze errors
        let error_summary = self.analyze_errors(&output_lines);
        
        // Calculate resource usage (simplified for now)
        let resource_usage = ResourceUsage {
            disk_io_read_bytes: 0,  // Would need more sophisticated monitoring
            disk_io_write_bytes: 0,
            network_io_bytes: 0,
            file_handles_used: 0,
            thread_count: 1, // Single process for now
        };

        Ok(PensieveExecutionResults {
            exit_code,
            execution_time,
            peak_memory_mb,
            average_memory_mb,
            cpu_usage_stats: cpu_stats,
            output_analysis,
            performance_metrics,
            error_summary,
            resource_usage,
        })
    }

    /// Analyze memory usage patterns
    fn analyze_memory_usage(&self, readings: &[MemoryReading]) -> (u64, u64, CpuUsageStats) {
        if readings.is_empty() {
            return (0, 0, CpuUsageStats {
                peak_cpu_percent: 0.0,
                average_cpu_percent: 0.0,
                cpu_time_user: Duration::from_secs(0),
                cpu_time_system: Duration::from_secs(0),
            });
        }

        let peak_memory = readings.iter().map(|r| r.memory_mb).max().unwrap_or(0);
        let average_memory = readings.iter().map(|r| r.memory_mb).sum::<u64>() / readings.len() as u64;
        
        let peak_cpu = readings.iter().map(|r| r.cpu_usage_percent).fold(0.0f32, f32::max);
        let average_cpu = readings.iter().map(|r| r.cpu_usage_percent).sum::<f32>() / readings.len() as f32;

        let cpu_stats = CpuUsageStats {
            peak_cpu_percent: peak_cpu,
            average_cpu_percent: average_cpu,
            cpu_time_user: Duration::from_secs(0), // Would need more detailed process info
            cpu_time_system: Duration::from_secs(0),
        };

        (peak_memory, average_memory, cpu_stats)
    }

    /// Analyze pensieve output for insights
    fn analyze_output(&self, lines: &[String]) -> OutputAnalysis {
        let mut analysis = OutputAnalysis {
            total_lines: lines.len() as u64,
            error_lines: 0,
            warning_lines: 0,
            progress_updates: 0,
            files_processed: 0,
            duplicates_found: 0,
            processing_speed_files_per_second: 0.0,
            key_messages: Vec::new(),
        };

        for line in lines {
            let line_lower = line.to_lowercase();
            
            // Count different types of messages
            if line_lower.contains("error") {
                analysis.error_lines += 1;
            }
            if line_lower.contains("warning") || line_lower.contains("warn") {
                analysis.warning_lines += 1;
            }
            if line_lower.contains("processed") || line_lower.contains("progress") {
                analysis.progress_updates += 1;
            }
            
            // Extract specific metrics
            if let Some(files) = self.extract_files_processed(line) {
                analysis.files_processed = files;
            }
            if let Some(duplicates) = self.extract_duplicates_found(line) {
                analysis.duplicates_found = duplicates;
            }
            if let Some(speed) = self.extract_processing_speed(line) {
                analysis.processing_speed_files_per_second = speed;
            }
            
            // Collect key messages
            if line_lower.contains("completed") || 
               line_lower.contains("finished") || 
               line_lower.contains("summary") ||
               line_lower.contains("total") {
                analysis.key_messages.push(line.clone());
            }
        }

        analysis
    }

    /// Extract number of files processed from output line
    fn extract_files_processed(&self, line: &str) -> Option<u64> {
        // Look for patterns like "Processed 1234 files"
        let line_lower = line.to_lowercase();
        if let Some(start) = line_lower.find("processed") {
            let after_processed = &line[start..];
            // Simple regex-like extraction
            for word in after_processed.split_whitespace() {
                if let Ok(num) = word.parse::<u64>() {
                    return Some(num);
                }
            }
        }
        None
    }

    /// Extract number of duplicates found from output line
    fn extract_duplicates_found(&self, line: &str) -> Option<u64> {
        // Look for patterns like "Found 567 duplicates"
        if line.to_lowercase().contains("duplicate") {
            for word in line.split_whitespace() {
                if let Ok(num) = word.parse::<u64>() {
                    return Some(num);
                }
            }
        }
        None
    }

    /// Extract processing speed from output line
    fn extract_processing_speed(&self, line: &str) -> Option<f64> {
        // Look for patterns like "123.45 files/sec"
        if line.contains("files/sec") || line.contains("files per second") {
            for word in line.split_whitespace() {
                if let Ok(speed) = word.parse::<f64>() {
                    return Some(speed);
                }
            }
        }
        None
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(
        &self,
        memory_readings: &[MemoryReading],
        output_analysis: &OutputAnalysis,
        execution_time: Duration,
    ) -> PerformanceMetrics {
        let files_per_second = if execution_time.as_secs() > 0 {
            output_analysis.files_processed as f64 / execution_time.as_secs_f64()
        } else {
            output_analysis.processing_speed_files_per_second
        };

        // Calculate memory efficiency (lower memory usage relative to work done = higher efficiency)
        let memory_efficiency = if !memory_readings.is_empty() && output_analysis.files_processed > 0 {
            let avg_memory = memory_readings.iter().map(|r| r.memory_mb).sum::<u64>() as f64 / memory_readings.len() as f64;
            let files_per_mb = output_analysis.files_processed as f64 / avg_memory.max(1.0);
            (files_per_mb / 100.0).min(1.0) // Normalize to 0-1 scale
        } else {
            0.0
        };

        // Calculate processing consistency (lower variance in memory/CPU = higher consistency)
        let consistency = if memory_readings.len() > 1 {
            let memory_values: Vec<f64> = memory_readings.iter().map(|r| r.memory_mb as f64).collect();
            let mean = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
            let variance = memory_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / memory_values.len() as f64;
            let coefficient_of_variation = variance.sqrt() / mean.max(1.0);
            (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
        } else {
            1.0
        };

        PerformanceMetrics {
            files_per_second,
            bytes_per_second: 0, // Would need more detailed I/O monitoring
            database_operations_per_second: 0.0, // Would need pensieve to report this
            memory_efficiency_score: memory_efficiency,
            processing_consistency: consistency,
        }
    }

    /// Analyze errors in the output
    fn analyze_errors(&self, lines: &[String]) -> ErrorSummary {
        let mut error_categories = HashMap::new();
        let mut critical_errors = Vec::new();
        let mut recoverable_errors = Vec::new();
        let mut total_errors = 0;

        for line in lines {
            let line_lower = line.to_lowercase();
            
            if line_lower.contains("error") {
                total_errors += 1;
                
                // Determine if error is critical or recoverable first
                let is_critical = line_lower.contains("fatal") || 
                                 line_lower.contains("critical") || 
                                 line_lower.contains("panic") ||
                                 line_lower.contains("out of memory");
                
                if is_critical {
                    critical_errors.push(line.clone());
                } else {
                    recoverable_errors.push(line.clone());
                }
                
                // Categorize errors
                if line_lower.contains("permission") || line_lower.contains("access denied") {
                    *error_categories.entry("Permission".to_string()).or_insert(0) += 1;
                } else if line_lower.contains("file not found") || line_lower.contains("no such file") {
                    *error_categories.entry("FileNotFound".to_string()).or_insert(0) += 1;
                } else if line_lower.contains("database") || line_lower.contains("sql") {
                    *error_categories.entry("Database".to_string()).or_insert(0) += 1;
                } else if line_lower.contains("memory") || line_lower.contains("out of memory") {
                    *error_categories.entry("Memory".to_string()).or_insert(0) += 1;
                } else if line_lower.contains("timeout") {
                    *error_categories.entry("Timeout".to_string()).or_insert(0) += 1;
                } else {
                    *error_categories.entry("Other".to_string()).or_insert(0) += 1;
                }
            }
        }

        let error_rate_per_minute = if total_errors > 0 && !lines.is_empty() {
            // Rough estimate based on output volume
            (total_errors as f64 / lines.len() as f64) * 60.0
        } else {
            0.0
        };

        ErrorSummary {
            total_errors,
            error_categories,
            critical_errors,
            recoverable_errors,
            error_rate_per_minute,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_pensieve_runner_creation() {
        let config = PensieveConfig::default();
        let runner = PensieveRunner::new(config);
        
        // Basic creation test
        assert!(runner.config.timeout_seconds > 0);
    }

    #[tokio::test]
    async fn test_input_validation() {
        let config = PensieveConfig {
            binary_path: PathBuf::from("/nonexistent/binary"),
            ..Default::default()
        };
        let runner = PensieveRunner::new(config);
        
        let temp_dir = TempDir::new().unwrap();
        let result = runner.validate_inputs(temp_dir.path());
        
        // Should fail because binary doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_output_analysis() {
        let config = PensieveConfig::default();
        let runner = PensieveRunner::new(config);
        
        let test_lines = vec![
            "STDOUT: Starting pensieve scan...".to_string(),
            "STDOUT: Processed 1000 files".to_string(),
            "STDOUT: Found 50 duplicates".to_string(),
            "STDERR: Warning: Large file detected".to_string(),
            "STDERR: Error: Permission denied for file.txt".to_string(),
            "STDOUT: Processing speed: 123.45 files/sec".to_string(),
            "STDOUT: Scan completed successfully".to_string(),
        ];
        
        let analysis = runner.analyze_output(&test_lines);
        
        assert_eq!(analysis.total_lines, 7);
        assert_eq!(analysis.error_lines, 1);
        assert_eq!(analysis.warning_lines, 1);
        assert_eq!(analysis.files_processed, 1000);
        assert_eq!(analysis.duplicates_found, 50);
        assert!((analysis.processing_speed_files_per_second - 123.45).abs() < 0.01);
        assert!(analysis.key_messages.len() > 0);
    }

    #[test]
    fn test_memory_analysis() {
        let config = PensieveConfig::default();
        let runner = PensieveRunner::new(config);
        
        let readings = vec![
            MemoryReading::new(100, 200, 25.0),
            MemoryReading::new(150, 250, 50.0),
            MemoryReading::new(120, 220, 30.0),
        ];
        
        let (peak_memory, avg_memory, cpu_stats) = runner.analyze_memory_usage(&readings);
        
        assert_eq!(peak_memory, 150);
        assert_eq!(avg_memory, 123); // (100 + 150 + 120) / 3 = 123.33 -> 123
        assert!((cpu_stats.peak_cpu_percent - 50.0).abs() < 0.01);
        assert!((cpu_stats.average_cpu_percent - 35.0).abs() < 0.01);
    }

    #[test]
    fn test_error_analysis() {
        let config = PensieveConfig::default();
        let runner = PensieveRunner::new(config);
        
        let test_lines = vec![
            "STDOUT: Processing file1.txt".to_string(),
            "STDERR: Error: Permission denied accessing file2.txt".to_string(),
            "STDERR: Error: File not found: missing.txt".to_string(),
            "STDERR: Fatal error: Out of memory".to_string(),
            "STDOUT: Warning: Large file detected".to_string(),
            "STDERR: Error: Database connection failed".to_string(),
        ];
        
        let error_summary = runner.analyze_errors(&test_lines);
        
        assert_eq!(error_summary.total_errors, 4);
        assert!(error_summary.error_categories.contains_key("Permission"));
        assert!(error_summary.error_categories.contains_key("FileNotFound"));
        assert!(error_summary.error_categories.contains_key("Database"));
        assert_eq!(error_summary.critical_errors.len(), 1);
        assert_eq!(error_summary.recoverable_errors.len(), 3);
    }
}