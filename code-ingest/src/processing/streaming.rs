//! Streaming file processing module for constant memory usage
//! 
//! This module provides streaming file processing capabilities that maintain
//! constant memory usage regardless of repository size, with parallel processing
//! using all available CPU cores and comprehensive progress tracking.

use crate::error::{ProcessingError, ProcessingResult};
use crate::processing::{FileProcessor, ProcessedFile};
use futures::stream::Stream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn};

/// Configuration for streaming file processing
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of concurrent processing tasks
    pub max_concurrency: usize,
    /// Size of the streaming buffer (number of files to buffer)
    pub buffer_size: usize,
    /// Maximum memory usage per file processing task (bytes)
    pub max_memory_per_task: u64,
    /// Global memory limit for all processing (bytes, 0 = no limit)
    pub global_memory_limit: u64,
    /// Timeout for individual file processing
    pub file_timeout: Duration,
    /// Interval for progress updates
    pub progress_interval: Duration,
    /// Whether to use adaptive concurrency based on system load
    pub adaptive_concurrency: bool,
    /// Target CPU utilization percentage (0-100)
    pub target_cpu_utilization: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            max_concurrency: cpu_count * 2, // Use all available cores with hyperthreading
            buffer_size: 1000,
            max_memory_per_task: 64 * 1024 * 1024, // 64MB per task
            global_memory_limit: 1024 * 1024 * 1024, // 1GB global limit
            file_timeout: Duration::from_secs(60),
            progress_interval: Duration::from_millis(500),
            adaptive_concurrency: true,
            target_cpu_utilization: 80.0,
        }
    }
}

/// Real-time processing statistics
#[derive(Debug)]
pub struct StreamingStats {
    /// Files processed successfully
    pub files_processed: AtomicUsize,
    /// Files that failed processing
    pub files_failed: AtomicUsize,
    /// Files skipped
    pub files_skipped: AtomicUsize,
    /// Total bytes processed
    pub bytes_processed: AtomicU64,
    /// Current memory usage (bytes)
    pub current_memory_usage: AtomicU64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: AtomicU64,
    /// Processing start time
    pub start_time: Instant,
    /// Current processing rate (files per second)
    pub current_rate: AtomicU64, // Stored as files per second * 1000 for precision
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for StreamingStats {
    fn clone(&self) -> Self {
        Self {
            files_processed: AtomicUsize::new(self.files_processed.load(Ordering::Relaxed)),
            files_failed: AtomicUsize::new(self.files_failed.load(Ordering::Relaxed)),
            files_skipped: AtomicUsize::new(self.files_skipped.load(Ordering::Relaxed)),
            bytes_processed: AtomicU64::new(self.bytes_processed.load(Ordering::Relaxed)),
            current_memory_usage: AtomicU64::new(self.current_memory_usage.load(Ordering::Relaxed)),
            peak_memory_usage: AtomicU64::new(self.peak_memory_usage.load(Ordering::Relaxed)),
            start_time: self.start_time,
            current_rate: AtomicU64::new(self.current_rate.load(Ordering::Relaxed)),
        }
    }
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            files_processed: AtomicUsize::new(0),
            files_failed: AtomicUsize::new(0),
            files_skipped: AtomicUsize::new(0),
            bytes_processed: AtomicU64::new(0),
            current_memory_usage: AtomicU64::new(0),
            peak_memory_usage: AtomicU64::new(0),
            start_time: Instant::now(),
            current_rate: AtomicU64::new(0),
        }
    }

    pub fn files_processed(&self) -> usize {
        self.files_processed.load(Ordering::Relaxed)
    }

    pub fn files_failed(&self) -> usize {
        self.files_failed.load(Ordering::Relaxed)
    }

    pub fn files_skipped(&self) -> usize {
        self.files_skipped.load(Ordering::Relaxed)
    }

    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed.load(Ordering::Relaxed)
    }

    pub fn current_memory_usage(&self) -> u64 {
        self.current_memory_usage.load(Ordering::Relaxed)
    }

    pub fn peak_memory_usage(&self) -> u64 {
        self.peak_memory_usage.load(Ordering::Relaxed)
    }

    pub fn processing_rate(&self) -> f64 {
        let rate_millis = self.current_rate.load(Ordering::Relaxed);
        rate_millis as f64 / 1000.0
    }

    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn total_files(&self) -> usize {
        self.files_processed() + self.files_failed() + self.files_skipped()
    }

    pub fn estimate_completion_time(&self, total_files: usize) -> Option<Duration> {
        let processed = self.total_files();
        if processed == 0 {
            return None;
        }

        let rate = self.processing_rate();
        if rate <= 0.0 {
            return None;
        }

        let remaining = total_files.saturating_sub(processed);
        let seconds_remaining = remaining as f64 / rate;
        Some(Duration::from_secs_f64(seconds_remaining))
    }
}

/// Progress information for streaming processing
#[derive(Debug, Clone)]
pub struct StreamingProgress {
    /// Current file being processed (if available)
    pub current_file: Option<String>,
    /// Files processed so far
    pub files_processed: usize,
    /// Files failed so far
    pub files_failed: usize,
    /// Files skipped so far
    pub files_skipped: usize,
    /// Total files to process (if known)
    pub total_files: Option<usize>,
    /// Current processing rate (files per second)
    pub processing_rate: f64,
    /// Current memory usage (bytes)
    pub memory_usage: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,
    /// Estimated time to completion
    pub eta: Option<Duration>,
    /// Elapsed processing time
    pub elapsed: Duration,
    /// Current CPU utilization percentage
    pub cpu_utilization: Option<f64>,
    /// Active concurrent tasks
    pub active_tasks: usize,
}

/// Memory tracker for streaming processing
#[derive(Debug)]
struct MemoryTracker {
    current_usage: AtomicU64,
    peak_usage: AtomicU64,
    global_limit: u64,
    per_task_limit: u64,
}

impl MemoryTracker {
    fn new(global_limit: u64, per_task_limit: u64) -> Self {
        Self {
            current_usage: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            global_limit,
            per_task_limit,
        }
    }

    fn try_allocate(&self, bytes: u64) -> bool {
        // Check per-task limit
        if self.per_task_limit > 0 && bytes > self.per_task_limit {
            return false;
        }

        // Check global limit
        if self.global_limit > 0 {
            let current = self.current_usage.load(Ordering::Relaxed);
            if current + bytes > self.global_limit {
                return false;
            }
        }

        // Atomically add the usage
        let new_usage = self.current_usage.fetch_add(bytes, Ordering::Relaxed) + bytes;
        
        // Update peak usage
        self.peak_usage.fetch_max(new_usage, Ordering::Relaxed);
        
        true
    }

    fn deallocate(&self, bytes: u64) {
        self.current_usage.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn current_usage(&self) -> u64 {
        self.current_usage.load(Ordering::Relaxed)
    }

    fn peak_usage(&self) -> u64 {
        self.peak_usage.load(Ordering::Relaxed)
    }
}

/// CPU utilization monitor
#[derive(Debug)]
struct CpuMonitor {
    #[allow(dead_code)]
    last_measurement: std::sync::Mutex<Option<(Instant, f64)>>,
}

impl CpuMonitor {
    fn new() -> Self {
        Self {
            last_measurement: std::sync::Mutex::new(None),
        }
    }

    /// Get current CPU utilization percentage (0-100)
    /// This is a simplified implementation - in production you'd use system APIs
    fn get_utilization(&self) -> Option<f64> {
        // For now, return None to indicate CPU monitoring is not implemented
        // In a real implementation, you would use system APIs like:
        // - /proc/stat on Linux
        // - GetSystemTimes on Windows
        // - host_processor_info on macOS
        None
    }
}

/// Streaming file processor with constant memory usage
pub struct StreamingProcessor {
    config: StreamingConfig,
    file_processor: Arc<dyn FileProcessor>,
    memory_tracker: Arc<MemoryTracker>,
    cpu_monitor: Arc<CpuMonitor>,
    stats: Arc<StreamingStats>,
}

impl StreamingProcessor {
    /// Create a new streaming processor
    pub fn new(config: StreamingConfig, file_processor: Arc<dyn FileProcessor>) -> Self {
        let memory_tracker = Arc::new(MemoryTracker::new(
            config.global_memory_limit,
            config.max_memory_per_task,
        ));
        let cpu_monitor = Arc::new(CpuMonitor::new());
        let stats = Arc::new(StreamingStats::new());

        Self {
            config,
            file_processor,
            memory_tracker,
            cpu_monitor,
            stats,
        }
    }

    /// Process files as a stream with constant memory usage
    pub async fn process_files_streaming<I>(
        &self,
        file_paths: I,
        progress_callback: Option<Box<dyn Fn(StreamingProgress) + Send + Sync>>,
    ) -> ProcessingResult<impl Stream<Item = ProcessingResult<ProcessedFile>>>
    where
        I: IntoIterator<Item = PathBuf> + Send,
        I::IntoIter: Send,
    {
        let (tx, rx) = mpsc::channel(self.config.buffer_size);
        let file_paths: Vec<PathBuf> = file_paths.into_iter().collect();
        let total_files = file_paths.len();

        info!("Starting streaming processing of {} files", total_files);

        // Spawn the processing task
        let processor = self.clone();
        let progress_callback_clone = progress_callback;
        tokio::spawn(async move {
            let result = processor
                .process_files_internal(file_paths, total_files, tx.clone(), progress_callback_clone)
                .await;

            if let Err(e) = result {
                warn!("Streaming processing error: {}", e);
                let _ = tx.send(Err(e)).await;
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    /// Internal processing implementation
    async fn process_files_internal(
        &self,
        file_paths: Vec<PathBuf>,
        total_files: usize,
        result_tx: mpsc::Sender<ProcessingResult<ProcessedFile>>,
        progress_callback: Option<Box<dyn Fn(StreamingProgress) + Send + Sync>>,
    ) -> ProcessingResult<()> {
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));
        let mut join_set = JoinSet::new();
        let active_tasks = Arc::new(AtomicUsize::new(0));

        // Start progress reporting task
        let progress_task = if progress_callback.is_some() {
            let stats = Arc::clone(&self.stats);
            let memory_tracker = Arc::clone(&self.memory_tracker);
            let cpu_monitor = Arc::clone(&self.cpu_monitor);
            let active_tasks_clone = Arc::clone(&active_tasks);
            let callback = progress_callback.unwrap();
            let interval = self.config.progress_interval;

            Some(tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(interval);
                loop {
                    interval_timer.tick().await;
                    
                    let progress = StreamingProgress {
                        current_file: None, // Could be enhanced to track current files
                        files_processed: stats.files_processed(),
                        files_failed: stats.files_failed(),
                        files_skipped: stats.files_skipped(),
                        total_files: Some(total_files),
                        processing_rate: stats.processing_rate(),
                        memory_usage: memory_tracker.current_usage(),
                        peak_memory_usage: memory_tracker.peak_usage(),
                        eta: stats.estimate_completion_time(total_files),
                        elapsed: stats.elapsed_time(),
                        cpu_utilization: cpu_monitor.get_utilization(),
                        active_tasks: active_tasks_clone.load(Ordering::Relaxed),
                    };

                    callback(progress);

                    // Break if processing is complete
                    if stats.total_files() >= total_files {
                        break;
                    }
                }
            }))
        } else {
            None
        };

        // Process files with adaptive concurrency
        for (index, file_path) in file_paths.into_iter().enumerate() {
            // Adaptive concurrency adjustment
            if self.config.adaptive_concurrency {
                self.adjust_concurrency(&semaphore, &active_tasks).await;
            }

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let file_processor = Arc::clone(&self.file_processor);
            let memory_tracker = Arc::clone(&self.memory_tracker);
            let stats = Arc::clone(&self.stats);
            let result_tx = result_tx.clone();
            let active_tasks_clone = Arc::clone(&active_tasks);
            let timeout = self.config.file_timeout;

            active_tasks.fetch_add(1, Ordering::Relaxed);

            join_set.spawn(async move {
                let _permit = permit; // Keep permit alive
                
                let result = tokio::time::timeout(
                    timeout,
                    Self::process_single_file_streaming(
                        file_processor,
                        file_path,
                        memory_tracker,
                        stats,
                        index,
                    ),
                )
                .await;

                let final_result = match result {
                    Ok(Ok(processed_file)) => Ok(processed_file),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(ProcessingError::ContentAnalysisFailed {
                        path: "unknown".to_string(),
                        cause: "Processing timeout".to_string(),
                    }),
                };

                active_tasks_clone.fetch_sub(1, Ordering::Relaxed);
                let _ = result_tx.send(final_result).await;
            });
        }

        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                warn!("Task join error: {}", e);
            }
        }

        // Stop progress reporting
        if let Some(task) = progress_task {
            task.abort();
        }

        info!(
            "Streaming processing completed: {} processed, {} failed, {} skipped",
            self.stats.files_processed(),
            self.stats.files_failed(),
            self.stats.files_skipped()
        );

        Ok(())
    }

    /// Process a single file with streaming memory management
    async fn process_single_file_streaming(
        file_processor: Arc<dyn FileProcessor>,
        file_path: PathBuf,
        memory_tracker: Arc<MemoryTracker>,
        stats: Arc<StreamingStats>,
        _index: usize,
    ) -> ProcessingResult<ProcessedFile> {
        let start_time = Instant::now();

        // Estimate memory usage
        let file_size = tokio::fs::metadata(&file_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        // Estimate processing memory (file size + overhead)
        let estimated_memory = file_size + (file_size / 2).max(1024 * 1024); // At least 1MB overhead

        // Try to allocate memory
        if !memory_tracker.try_allocate(estimated_memory) {
            stats.files_failed.fetch_add(1, Ordering::Relaxed);
            return Err(ProcessingError::ContentAnalysisFailed {
                path: file_path.display().to_string(),
                cause: "Memory limit exceeded".to_string(),
            });
        }

        // Update memory usage in stats
        stats.current_memory_usage.store(memory_tracker.current_usage(), Ordering::Relaxed);
        stats.peak_memory_usage.store(memory_tracker.peak_usage(), Ordering::Relaxed);

        // Process the file
        let result = file_processor.process(&file_path).await;

        // Release memory
        memory_tracker.deallocate(estimated_memory);
        stats.current_memory_usage.store(memory_tracker.current_usage(), Ordering::Relaxed);

        // Update statistics
        match &result {
            Ok(processed_file) => {
                if processed_file.skipped {
                    stats.files_skipped.fetch_add(1, Ordering::Relaxed);
                } else {
                    stats.files_processed.fetch_add(1, Ordering::Relaxed);
                    stats.bytes_processed.fetch_add(file_size, Ordering::Relaxed);
                }
            }
            Err(_) => {
                stats.files_failed.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update processing rate
        let elapsed = stats.elapsed_time();
        if elapsed.as_secs_f64() > 0.0 {
            let total_processed = stats.total_files();
            let rate = total_processed as f64 / elapsed.as_secs_f64();
            stats.current_rate.store((rate * 1000.0) as u64, Ordering::Relaxed);
        }

        debug!(
            "Processed file {} in {:?}: {}",
            file_path.display(),
            start_time.elapsed(),
            if result.is_ok() { "success" } else { "failed" }
        );

        result
    }

    /// Adjust concurrency based on system load
    async fn adjust_concurrency(
        &self,
        _semaphore: &Arc<Semaphore>,
        _active_tasks: &Arc<AtomicUsize>,
    ) {
        // This is a placeholder for adaptive concurrency adjustment
        // In a real implementation, you would:
        // 1. Monitor CPU utilization
        // 2. Monitor memory pressure
        // 3. Adjust the semaphore permits dynamically
        // 4. Consider I/O wait times
        
        // For now, we use static concurrency based on configuration
    }

    /// Get current processing statistics
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.as_ref().clone()
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> (u64, u64) {
        (
            self.memory_tracker.current_usage(),
            self.memory_tracker.peak_usage(),
        )
    }
}

impl Clone for StreamingProcessor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            file_processor: Arc::clone(&self.file_processor),
            memory_tracker: Arc::clone(&self.memory_tracker),
            cpu_monitor: Arc::clone(&self.cpu_monitor),
            stats: Arc::clone(&self.stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::{FileType, ProcessedFile, FileProcessor};
    use crate::error::{ProcessingError, ProcessingResult};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use tempfile::TempDir;
    use tokio_stream::StreamExt;

    // Mock file processor for testing
    struct MockStreamingProcessor {
        delay: Duration,
        should_fail: Arc<Mutex<bool>>,
        processed_files: Arc<Mutex<Vec<PathBuf>>>,
    }

    impl MockStreamingProcessor {
        fn new(delay: Duration) -> Self {
            Self {
                delay,
                should_fail: Arc::new(Mutex::new(false)),
                processed_files: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn set_should_fail(&self, should_fail: bool) {
            *self.should_fail.lock().unwrap() = should_fail;
        }

        fn get_processed_files(&self) -> Vec<PathBuf> {
            self.processed_files.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl FileProcessor for MockStreamingProcessor {
        fn can_process(&self, _file_path: &Path) -> bool {
            true
        }

        async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
            self.processed_files.lock().unwrap().push(file_path.to_path_buf());
            tokio::time::sleep(self.delay).await;

            if *self.should_fail.lock().unwrap() {
                return Err(ProcessingError::FileReadFailed {
                    path: file_path.display().to_string(),
                    cause: "Mock failure".to_string(),
                });
            }

            Ok(ProcessedFile {
                filepath: file_path.display().to_string(),
                filename: file_path.file_name().unwrap().to_str().unwrap().to_string(),
                extension: file_path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_string(),
                file_size_bytes: 1024,
                line_count: Some(10),
                word_count: Some(50),
                token_count: Some(100),
                content_text: Some("Mock content".to_string()),
                file_type: FileType::DirectText,
                conversion_command: None,
                relative_path: file_path.display().to_string(),
                absolute_path: file_path.display().to_string(),
                skipped: false,
                skip_reason: None,
            })
        }

        fn get_file_type(&self) -> FileType {
            FileType::DirectText
        }
    }

    fn create_test_files(temp_dir: &TempDir, count: usize) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for i in 0..count {
            let file_path = temp_dir.path().join(format!("test_file_{}.txt", i));
            std::fs::write(&file_path, format!("Content of file {}", i)).unwrap();
            files.push(file_path);
        }
        files
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert!(config.max_concurrency > 0);
        assert_eq!(config.buffer_size, 1000);
        assert!(config.global_memory_limit > 0);
        assert!(config.adaptive_concurrency);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new(1000, 500);

        // Test successful allocation within limits
        assert!(tracker.try_allocate(300));
        assert_eq!(tracker.current_usage(), 300);

        // Test allocation exceeding per-task limit
        assert!(!tracker.try_allocate(600));
        assert_eq!(tracker.current_usage(), 300);

        // Test allocation exceeding global limit
        assert!(!tracker.try_allocate(800));
        assert_eq!(tracker.current_usage(), 300);

        // Test successful allocation within remaining space
        assert!(tracker.try_allocate(200));
        assert_eq!(tracker.current_usage(), 500);

        // Test deallocation
        tracker.deallocate(100);
        assert_eq!(tracker.current_usage(), 400);

        // Test peak usage tracking
        assert_eq!(tracker.peak_usage(), 500);
    }

    #[test]
    fn test_streaming_stats() {
        let stats = StreamingStats::new();

        // Test initial state
        assert_eq!(stats.files_processed(), 0);
        assert_eq!(stats.files_failed(), 0);
        assert_eq!(stats.files_skipped(), 0);
        assert_eq!(stats.processing_rate(), 0.0);

        // Test updates
        stats.files_processed.store(10, Ordering::Relaxed);
        stats.files_failed.store(2, Ordering::Relaxed);
        stats.files_skipped.store(1, Ordering::Relaxed);
        stats.current_rate.store(2500, Ordering::Relaxed); // 2.5 files/sec

        assert_eq!(stats.files_processed(), 10);
        assert_eq!(stats.files_failed(), 2);
        assert_eq!(stats.files_skipped(), 1);
        assert_eq!(stats.total_files(), 13);
        assert_eq!(stats.processing_rate(), 2.5);

        // Test ETA calculation
        let eta = stats.estimate_completion_time(20);
        assert!(eta.is_some());
        let eta_duration = eta.unwrap();
        // Should be approximately (20 - 13) / 2.5 = 2.8 seconds
        assert!(eta_duration.as_secs_f64() > 2.0 && eta_duration.as_secs_f64() < 4.0);
    }

    #[tokio::test]
    async fn test_streaming_processor_basic() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 5);

        let config = StreamingConfig {
            max_concurrency: 2,
            buffer_size: 10,
            adaptive_concurrency: false,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockStreamingProcessor::new(Duration::from_millis(10)));
        let streaming_processor = StreamingProcessor::new(config, mock_processor.clone());

        let mut stream = streaming_processor
            .process_files_streaming(files.clone(), None)
            .await
            .unwrap();

        let mut results: Vec<ProcessingResult<ProcessedFile>> = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_ok()));

        // Verify all files were processed
        let processed_paths = mock_processor.get_processed_files();
        assert_eq!(processed_paths.len(), 5);
        for file in &files {
            assert!(processed_paths.contains(file));
        }
    }

    #[tokio::test]
    async fn test_streaming_processor_with_failures() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 3);

        let config = StreamingConfig {
            max_concurrency: 1,
            adaptive_concurrency: false,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockStreamingProcessor::new(Duration::from_millis(10)));
        mock_processor.set_should_fail(true);

        let streaming_processor = StreamingProcessor::new(config, mock_processor);

        let mut stream = streaming_processor
            .process_files_streaming(files, None)
            .await
            .unwrap();

        let mut results: Vec<ProcessingResult<ProcessedFile>> = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_err()));
    }

    #[tokio::test]
    async fn test_streaming_processor_memory_limits() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 3);

        let config = StreamingConfig {
            max_concurrency: 1,
            global_memory_limit: 100, // Very low limit
            max_memory_per_task: 50,
            adaptive_concurrency: false,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockStreamingProcessor::new(Duration::from_millis(10)));
        let streaming_processor = StreamingProcessor::new(config, mock_processor);

        let mut stream = streaming_processor
            .process_files_streaming(files, None)
            .await
            .unwrap();

        let mut results: Vec<ProcessingResult<ProcessedFile>> = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        // Some files should fail due to memory limits
        let failed_count = results.iter().filter(|r| r.is_err()).count();
        assert!(failed_count > 0);
    }

    #[tokio::test]
    async fn test_streaming_processor_progress_callback() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 5);

        let config = StreamingConfig {
            max_concurrency: 2,
            progress_interval: Duration::from_millis(50),
            adaptive_concurrency: false,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockStreamingProcessor::new(Duration::from_millis(20)));
        let streaming_processor = StreamingProcessor::new(config, mock_processor);

        let progress_updates = Arc::new(Mutex::new(Vec::new()));
        let progress_updates_clone = Arc::clone(&progress_updates);

        let progress_callback = Box::new(move |progress: StreamingProgress| {
            progress_updates_clone.lock().unwrap().push(progress);
        });

        let mut stream = streaming_processor
            .process_files_streaming(files, Some(progress_callback))
            .await
            .unwrap();

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        // Give progress task time to complete
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert_eq!(results.len(), 5);
        let updates = progress_updates.lock().unwrap();
        assert!(!updates.is_empty());

        // Check that progress updates show increasing completion
        let first_update = &updates[0];
        let last_update = &updates[updates.len() - 1];
        assert!(last_update.files_processed >= first_update.files_processed);
    }

    #[test]
    fn test_cpu_monitor() {
        let monitor = CpuMonitor::new();
        // CPU monitoring is not implemented in the mock, so it should return None
        assert!(monitor.get_utilization().is_none());
    }
}