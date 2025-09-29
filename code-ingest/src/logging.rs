
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Logging configuration for the application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable JSON formatting for structured logs
    pub json_format: bool,
    /// Enable file logging
    pub file_logging: bool,
    /// Log file path (if file logging is enabled)
    pub log_file: Option<PathBuf>,
    /// Maximum log file size in MB
    pub max_file_size_mb: u64,
    /// Number of log files to keep in rotation
    pub max_files: u32,
    /// Enable progress reporting
    pub progress_reporting: bool,
    /// Enable performance metrics collection
    pub performance_metrics: bool,
    /// Enable span tracing for detailed operation tracking
    pub span_tracing: bool,
    /// Custom log targets and their levels
    pub targets: HashMap<String, String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json_format: false,
            file_logging: false,
            log_file: None,
            max_file_size_mb: 100,
            max_files: 5,
            progress_reporting: true,
            performance_metrics: true,
            span_tracing: false,
            targets: HashMap::new(),
        }
    }
}

/// Initialize logging system with configuration
pub fn init_logging(config: &LoggingConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&config.level))?;

    // Add custom target filters
    for (target, level) in &config.targets {
        filter = filter.add_directive(format!("{}={}", target, level).parse()?);
    }

    let registry = Registry::default().with(filter);

    if config.json_format {
        // JSON structured logging
        let json_layer = fmt::layer()
            .json()
            .with_current_span(config.span_tracing)
            .with_span_list(config.span_tracing)
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true);

        if config.file_logging {
            if let Some(log_file) = &config.log_file {
                let file_appender = tracing_appender::rolling::daily(
                    log_file.parent().unwrap_or_else(|| std::path::Path::new(".")),
                    log_file.file_name().unwrap_or_else(|| std::ffi::OsStr::new("app.log"))
                );
                let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
                
                let stdout_layer = fmt::layer()
                    .json()
                    .with_current_span(config.span_tracing)
                    .with_span_list(config.span_tracing)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .with_writer(io::stdout);
                
                let file_layer = fmt::layer()
                    .json()
                    .with_current_span(config.span_tracing)
                    .with_span_list(config.span_tracing)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .with_writer(non_blocking);
                
                registry
                    .with(stdout_layer)
                    .with(file_layer)
                    .init();
            } else {
                registry.with(json_layer).init();
            }
        } else {
            registry.with(json_layer).init();
        }
    } else {
        // Human-readable logging
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_thread_names(false)
            .with_file(true)
            .with_line_number(true)
            .with_span_events(if config.span_tracing { FmtSpan::CLOSE } else { FmtSpan::NONE })
            .compact();

        if config.file_logging {
            if let Some(log_file) = &config.log_file {
                let file_appender = tracing_appender::rolling::daily(
                    log_file.parent().unwrap_or_else(|| std::path::Path::new(".")),
                    log_file.file_name().unwrap_or_else(|| std::ffi::OsStr::new("app.log"))
                );
                let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
                
                let stdout_layer = fmt::layer()
                    .with_target(true)
                    .with_thread_ids(false)
                    .with_thread_names(false)
                    .with_file(true)
                    .with_line_number(true)
                    .with_span_events(if config.span_tracing { FmtSpan::CLOSE } else { FmtSpan::NONE })
                    .compact()
                    .with_writer(io::stdout);
                
                let file_layer = fmt::layer()
                    .with_target(true)
                    .with_thread_ids(false)
                    .with_thread_names(false)
                    .with_file(true)
                    .with_line_number(true)
                    .with_span_events(if config.span_tracing { FmtSpan::CLOSE } else { FmtSpan::NONE })
                    .compact()
                    .with_writer(non_blocking);
                
                registry
                    .with(stdout_layer)
                    .with(file_layer)
                    .init();
            } else {
                registry.with(fmt_layer).init();
            }
        } else {
            registry.with(fmt_layer).init();
        }
    }

    tracing::info!("Logging initialized with level: {}", config.level);
    if config.performance_metrics {
        tracing::info!("Performance metrics collection enabled");
    }
    if config.progress_reporting {
        tracing::info!("Progress reporting enabled");
    }

    Ok(())
}

/// Progress reporter for long-running operations
#[derive(Debug)]
pub struct ProgressReporter {
    operation_name: String,
    total_items: Option<u64>,
    processed_items: Arc<RwLock<u64>>,
    start_time: Instant,
    last_report_time: Arc<RwLock<Instant>>,
    report_interval: Duration,
    enabled: bool,
}

impl ProgressReporter {
    pub fn new(operation_name: impl Into<String>, total_items: Option<u64>) -> Self {
        Self {
            operation_name: operation_name.into(),
            total_items,
            processed_items: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
            last_report_time: Arc::new(RwLock::new(Instant::now())),
            report_interval: Duration::from_secs(5),
            enabled: true,
        }
    }

    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.report_interval = interval;
        self
    }

    pub fn disabled() -> Self {
        Self {
            operation_name: "disabled".to_string(),
            total_items: None,
            processed_items: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
            last_report_time: Arc::new(RwLock::new(Instant::now())),
            report_interval: Duration::from_secs(5),
            enabled: false,
        }
    }

    /// Increment processed items count
    pub async fn increment(&self) {
        if !self.enabled {
            return;
        }

        let mut processed = self.processed_items.write().await;
        *processed += 1;
        
        let current_time = Instant::now();
        let mut last_report = self.last_report_time.write().await;
        
        if current_time.duration_since(*last_report) >= self.report_interval {
            self.report_progress_internal(*processed, current_time).await;
            *last_report = current_time;
        }
    }

    /// Set processed items count
    pub async fn set_progress(&self, processed: u64) {
        if !self.enabled {
            return;
        }

        {
            let mut current_processed = self.processed_items.write().await;
            *current_processed = processed;
        }
        
        let current_time = Instant::now();
        let mut last_report = self.last_report_time.write().await;
        
        if current_time.duration_since(*last_report) >= self.report_interval {
            self.report_progress_internal(processed, current_time).await;
            *last_report = current_time;
        }
    }

    /// Force progress report
    pub async fn report_progress(&self) {
        if !self.enabled {
            return;
        }

        let processed = *self.processed_items.read().await;
        self.report_progress_internal(processed, Instant::now()).await;
    }

    async fn report_progress_internal(&self, processed: u64, current_time: Instant) {
        let elapsed = current_time.duration_since(self.start_time);
        let rate = if elapsed.as_secs() > 0 {
            processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        if let Some(total) = self.total_items {
            let percentage = (processed as f64 / total as f64) * 100.0;
            let eta = if rate > 0.0 {
                let remaining = total.saturating_sub(processed);
                Duration::from_secs_f64(remaining as f64 / rate)
            } else {
                Duration::from_secs(0)
            };

            tracing::info!(
                operation = %self.operation_name,
                processed = processed,
                total = total,
                percentage = format!("{:.1}%", percentage),
                rate = format!("{:.1}/s", rate),
                elapsed = format!("{:.1}s", elapsed.as_secs_f64()),
                eta = format!("{:.1}s", eta.as_secs_f64()),
                "Progress update"
            );
        } else {
            tracing::info!(
                operation = %self.operation_name,
                processed = processed,
                rate = format!("{:.1}/s", rate),
                elapsed = format!("{:.1}s", elapsed.as_secs_f64()),
                "Progress update"
            );
        }
    }

    /// Complete the operation and report final statistics
    pub async fn complete(&self) {
        if !self.enabled {
            return;
        }

        let processed = *self.processed_items.read().await;
        let elapsed = self.start_time.elapsed();
        let rate = if elapsed.as_secs() > 0 {
            processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        tracing::info!(
            operation = %self.operation_name,
            processed = processed,
            total = self.total_items,
            rate = format!("{:.1}/s", rate),
            elapsed = format!("{:.1}s", elapsed.as_secs_f64()),
            "Operation completed"
        );
    }
}

/// Performance metrics collector
#[derive(Debug)]
pub struct PerformanceMetrics {
    operation_name: String,
    start_time: Instant,
    checkpoints: Arc<RwLock<Vec<(String, Instant, Duration)>>>,
    counters: Arc<RwLock<HashMap<String, u64>>>,
    timers: Arc<RwLock<HashMap<String, Duration>>>,
    enabled: bool,
}

impl PerformanceMetrics {
    pub fn new(operation_name: impl Into<String>) -> Self {
        Self {
            operation_name: operation_name.into(),
            start_time: Instant::now(),
            checkpoints: Arc::new(RwLock::new(Vec::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
            enabled: true,
        }
    }

    pub fn disabled() -> Self {
        Self {
            operation_name: "disabled".to_string(),
            start_time: Instant::now(),
            checkpoints: Arc::new(RwLock::new(Vec::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
            enabled: false,
        }
    }

    /// Add a checkpoint with elapsed time
    pub async fn checkpoint(&self, name: impl Into<String>) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.start_time);
        let checkpoint_name = name.into();
        
        {
            let mut checkpoints = self.checkpoints.write().await;
            checkpoints.push((checkpoint_name.clone(), now, elapsed));
        }

        tracing::debug!(
            operation = %self.operation_name,
            checkpoint = %checkpoint_name,
            elapsed = format!("{:.3}s", elapsed.as_secs_f64()),
            "Performance checkpoint"
        );
    }

    /// Increment a counter
    pub async fn increment_counter(&self, name: impl Into<String>) {
        if !self.enabled {
            return;
        }

        let counter_name = name.into();
        let mut counters = self.counters.write().await;
        *counters.entry(counter_name).or_insert(0) += 1;
    }

    /// Add to a counter
    pub async fn add_to_counter(&self, name: impl Into<String>, value: u64) {
        if !self.enabled {
            return;
        }

        let counter_name = name.into();
        let mut counters = self.counters.write().await;
        *counters.entry(counter_name).or_insert(0) += value;
    }

    /// Record timing for an operation
    pub async fn record_timing(&self, name: impl Into<String>, duration: Duration) {
        if !self.enabled {
            return;
        }

        let timer_name = name.into();
        let mut timers = self.timers.write().await;
        
        // Store the latest timing (could be enhanced to store statistics)
        timers.insert(timer_name.clone(), duration);
        
        tracing::debug!(
            operation = %self.operation_name,
            timer = %timer_name,
            duration = format!("{:.3}ms", duration.as_millis()),
            "Performance timing"
        );
    }

    /// Time an async operation
    pub async fn time_async<F, Fut, T>(&self, name: impl Into<String>, operation: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let timer_name = name.into();
        let start = Instant::now();
        let result = operation().await;
        let duration = start.elapsed();
        
        self.record_timing(timer_name, duration).await;
        result
    }

    /// Generate performance report
    pub async fn report(&self) {
        if !self.enabled {
            return;
        }

        let total_elapsed = self.start_time.elapsed();
        
        tracing::info!(
            operation = %self.operation_name,
            total_elapsed = format!("{:.3}s", total_elapsed.as_secs_f64()),
            "Performance report start"
        );

        // Report checkpoints
        let checkpoints = self.checkpoints.read().await;
        for (name, _time, elapsed) in checkpoints.iter() {
            tracing::info!(
                operation = %self.operation_name,
                checkpoint = %name,
                elapsed = format!("{:.3}s", elapsed.as_secs_f64()),
                "Checkpoint timing"
            );
        }

        // Report counters
        let counters = self.counters.read().await;
        for (name, count) in counters.iter() {
            tracing::info!(
                operation = %self.operation_name,
                counter = %name,
                count = count,
                "Counter value"
            );
        }

        // Report timers
        let timers = self.timers.read().await;
        for (name, duration) in timers.iter() {
            tracing::info!(
                operation = %self.operation_name,
                timer = %name,
                duration = format!("{:.3}ms", duration.as_millis()),
                "Timer value"
            );
        }
    }
}

/// Memory usage monitor
#[derive(Debug)]
pub struct MemoryMonitor {
    operation_name: String,
    initial_memory: u64,
    peak_memory: Arc<RwLock<u64>>,
    enabled: bool,
}

impl MemoryMonitor {
    pub fn new(operation_name: impl Into<String>) -> Self {
        let initial_memory = Self::get_current_memory();
        Self {
            operation_name: operation_name.into(),
            initial_memory,
            peak_memory: Arc::new(RwLock::new(initial_memory)),
            enabled: true,
        }
    }

    pub fn disabled() -> Self {
        Self {
            operation_name: "disabled".to_string(),
            initial_memory: 0,
            peak_memory: Arc::new(RwLock::new(0)),
            enabled: false,
        }
    }

    /// Check current memory usage and update peak
    pub async fn check_memory(&self) {
        if !self.enabled {
            return;
        }

        let current_memory = Self::get_current_memory();
        let mut peak = self.peak_memory.write().await;
        
        if current_memory > *peak {
            *peak = current_memory;
            
            tracing::debug!(
                operation = %self.operation_name,
                current_memory_mb = current_memory / (1024 * 1024),
                peak_memory_mb = *peak / (1024 * 1024),
                "Memory usage update"
            );
        }
    }

    /// Report memory statistics
    pub async fn report(&self) {
        if !self.enabled {
            return;
        }

        let current_memory = Self::get_current_memory();
        let peak_memory = *self.peak_memory.read().await;
        let memory_increase = current_memory.saturating_sub(self.initial_memory);

        tracing::info!(
            operation = %self.operation_name,
            initial_memory_mb = self.initial_memory / (1024 * 1024),
            current_memory_mb = current_memory / (1024 * 1024),
            peak_memory_mb = peak_memory / (1024 * 1024),
            memory_increase_mb = memory_increase / (1024 * 1024),
            "Memory usage report"
        );
    }

    fn get_current_memory() -> u64 {
        use sysinfo::{System, Pid};
        
        let mut system = System::new_all();
        system.refresh_all();
        
        if let Some(process) = system.process(Pid::from(std::process::id() as usize)) {
            process.memory() * 1024 // Convert from KB to bytes
        } else {
            0
        }
    }
}

/// Comprehensive monitoring context that combines all monitoring tools
#[derive(Debug)]
pub struct MonitoringContext {
    pub progress: ProgressReporter,
    pub performance: PerformanceMetrics,
    pub memory: MemoryMonitor,
}

impl MonitoringContext {
    pub fn new(operation_name: impl Into<String>, total_items: Option<u64>) -> Self {
        let op_name = operation_name.into();
        Self {
            progress: ProgressReporter::new(op_name.clone(), total_items),
            performance: PerformanceMetrics::new(op_name.clone()),
            memory: MemoryMonitor::new(op_name),
        }
    }

    pub fn disabled() -> Self {
        Self {
            progress: ProgressReporter::disabled(),
            performance: PerformanceMetrics::disabled(),
            memory: MemoryMonitor::disabled(),
        }
    }

    /// Complete monitoring and generate comprehensive report
    pub async fn complete_and_report(&self) {
        self.progress.complete().await;
        self.performance.report().await;
        self.memory.report().await;
    }
}

/// Structured logging macros for consistent log formatting
#[macro_export]
macro_rules! log_operation_start {
    ($operation:expr, $($field:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            $($field = $value,)*
            "Operation started"
        );
    };
}

#[macro_export]
macro_rules! log_operation_complete {
    ($operation:expr, $duration:expr, $($field:ident = $value:expr),*) => {
        tracing::info!(
            operation = $operation,
            duration = format!("{:.3}s", $duration.as_secs_f64()),
            $($field = $value,)*
            "Operation completed"
        );
    };
}

#[macro_export]
macro_rules! log_operation_error {
    ($operation:expr, $error:expr, $($field:ident = $value:expr),*) => {
        tracing::error!(
            operation = $operation,
            error = %$error,
            $($field = $value,)*
            "Operation failed"
        );
    };
}

/// Batch operation logger for tracking batch processing
pub struct BatchLogger {
    operation_name: String,
    batch_size: usize,
    #[allow(dead_code)]
    total_items: Option<usize>,
    processed_batches: Arc<RwLock<usize>>,
    failed_batches: Arc<RwLock<usize>>,
    start_time: Instant,
}

impl BatchLogger {
    pub fn new(operation_name: impl Into<String>, batch_size: usize, total_items: Option<usize>) -> Self {
        let op_name = operation_name.into();
        tracing::info!(
            operation = %op_name,
            batch_size = batch_size,
            total_items = total_items,
            "Batch operation started"
        );

        Self {
            operation_name: op_name,
            batch_size,
            total_items,
            processed_batches: Arc::new(RwLock::new(0)),
            failed_batches: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
        }
    }

    pub async fn log_batch_complete(&self, batch_number: usize, items_in_batch: usize) {
        let mut processed = self.processed_batches.write().await;
        *processed += 1;

        let total_processed_items = (*processed - 1) * self.batch_size + items_in_batch;
        
        tracing::debug!(
            operation = %self.operation_name,
            batch_number = batch_number,
            items_in_batch = items_in_batch,
            total_processed_items = total_processed_items,
            "Batch completed"
        );
    }

    pub async fn log_batch_failed(&self, batch_number: usize, error: &str) {
        let mut failed = self.failed_batches.write().await;
        *failed += 1;

        tracing::error!(
            operation = %self.operation_name,
            batch_number = batch_number,
            error = error,
            "Batch failed"
        );
    }

    pub async fn complete(&self) {
        let processed = *self.processed_batches.read().await;
        let failed = *self.failed_batches.read().await;
        let elapsed = self.start_time.elapsed();
        let success_rate = if processed + failed > 0 {
            (processed as f64 / (processed + failed) as f64) * 100.0
        } else {
            0.0
        };

        tracing::info!(
            operation = %self.operation_name,
            processed_batches = processed,
            failed_batches = failed,
            success_rate = format!("{:.1}%", success_rate),
            elapsed = format!("{:.3}s", elapsed.as_secs_f64()),
            "Batch operation completed"
        );
    }
}