use crate::error::{IngestionError, IngestionResult};
use crate::processing::{FileProcessor, ProcessedFile, StreamingProcessor, StreamingConfig, StreamingProgress, PerformanceMonitor, PerformanceThresholds, OptimizationRecommendation};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use tracing::{debug, error, info, warn};

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of concurrent file processing tasks
    pub max_concurrency: usize,
    /// Timeout for individual file processing
    pub file_timeout: Duration,
    /// Batch size for database operations
    pub batch_size: usize,
    /// Whether to show progress bars
    pub show_progress: bool,
    /// Maximum memory usage in bytes (0 = no limit)
    pub max_memory_bytes: u64,
    /// Whether to continue processing on individual file errors
    pub continue_on_error: bool,
    /// Interval for progress updates
    pub progress_update_interval: Duration,
    /// Whether to use streaming processing for constant memory usage
    pub use_streaming: bool,
    /// Whether to enable performance monitoring
    pub enable_performance_monitoring: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrency: num_cpus::get().min(8), // Reasonable default
            file_timeout: Duration::from_secs(30),
            batch_size: 100,
            show_progress: true,
            max_memory_bytes: 512 * 1024 * 1024, // 512MB default
            continue_on_error: true,
            progress_update_interval: Duration::from_millis(100),
            use_streaming: true, // Enable streaming by default for better memory management
            enable_performance_monitoring: true,
        }
    }
}

/// Statistics about batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total files processed successfully
    pub files_processed: usize,
    /// Total files that failed processing
    pub files_failed: usize,
    /// Total files skipped
    pub files_skipped: usize,
    /// Total processing time
    pub total_duration: Duration,
    /// Average processing time per file
    pub avg_file_duration: Duration,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Number of batches processed
    pub batches_processed: usize,
}

/// Progress information for batch processing
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Current file being processed
    pub current_file: Option<String>,
    /// Files processed so far
    pub files_processed: usize,
    /// Total files to process
    pub total_files: usize,
    /// Current batch being processed
    pub current_batch: usize,
    /// Total batches to process
    pub total_batches: usize,
    /// Processing rate (files per second)
    pub processing_rate: f64,
    /// Estimated time remaining
    pub eta: Option<Duration>,
}

/// Batch processor for parallel file processing with controlled concurrency
#[derive(Clone)]
pub struct BatchProcessor {
    config: BatchConfig,
    file_processor: Arc<dyn FileProcessor>,
    shutdown_signal: Arc<AtomicBool>,
    memory_monitor: Arc<MemoryMonitor>,
    streaming_processor: Option<StreamingProcessor>,
    performance_monitor: Option<Arc<PerformanceMonitor>>,
}

/// Memory usage monitor
struct MemoryMonitor {
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    max_allowed: u64,
}

impl MemoryMonitor {
    fn new(max_allowed: u64) -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            max_allowed,
        }
    }

    fn add_usage(&self, bytes: usize) -> bool {
        let new_usage = self.current_usage.fetch_add(bytes, Ordering::Relaxed) + bytes;
        
        // Update peak usage
        self.peak_usage.fetch_max(new_usage, Ordering::Relaxed);
        
        // Check if we're within limits
        if self.max_allowed > 0 && new_usage as u64 > self.max_allowed {
            self.current_usage.fetch_sub(bytes, Ordering::Relaxed);
            false
        } else {
            true
        }
    }

    fn remove_usage(&self, bytes: usize) {
        self.current_usage.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn get_current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    fn get_peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }
}

impl BatchProcessor {
    /// Create a new BatchProcessor
    pub fn new(config: BatchConfig, file_processor: Arc<dyn FileProcessor>) -> Self {
        let memory_monitor = Arc::new(MemoryMonitor::new(config.max_memory_bytes));
        
        // Create streaming processor if enabled
        let streaming_processor = if config.use_streaming {
            let streaming_config = StreamingConfig {
                max_concurrency: config.max_concurrency,
                buffer_size: config.batch_size * 2,
                global_memory_limit: config.max_memory_bytes,
                file_timeout: config.file_timeout,
                progress_interval: config.progress_update_interval,
                adaptive_concurrency: true,
                ..Default::default()
            };
            Some(StreamingProcessor::new(streaming_config, Arc::clone(&file_processor)))
        } else {
            None
        };

        // Create performance monitor if enabled
        let performance_monitor = if config.enable_performance_monitoring {
            let perf_thresholds = PerformanceThresholds {
                max_cpu_usage: 90.0,
                max_memory_usage: 85.0,
                max_error_rate: 5.0,
                target_processing_rate: 100.0,
                max_latency_ms: 1000.0,
            };
            match PerformanceMonitor::new(perf_thresholds) {
                Ok(monitor) => Some(Arc::new(monitor)),
                Err(_) => None,
            }
        } else {
            None
        };
        
        Self {
            config,
            file_processor,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            memory_monitor,
            streaming_processor,
            performance_monitor,
        }
    }

    /// Process a batch of files with parallel processing and progress tracking
    pub async fn process_files<P>(
        &self,
        file_paths: Vec<P>,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> IngestionResult<(Vec<ProcessedFile>, BatchStats)>
    where
        P: AsRef<Path> + Send + 'static,
    {
        // Use streaming processor if available for better memory management
        if let Some(ref _streaming_processor) = self.streaming_processor {
            return self.process_files_streaming(file_paths, progress_callback).await;
        }

        // Fall back to original batch processing
        self.process_files_legacy(file_paths, progress_callback).await
    }

    /// Process files using streaming for constant memory usage
    async fn process_files_streaming<P>(
        &self,
        file_paths: Vec<P>,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> IngestionResult<(Vec<ProcessedFile>, BatchStats)>
    where
        P: AsRef<Path> + Send + 'static,
    {
        let start_time = Instant::now();
        let total_files = file_paths.len();
        
        info!("Starting streaming batch processing of {} files", total_files);

        // Start performance monitoring if enabled
        let _perf_handle = if let Some(ref perf_monitor) = self.performance_monitor {
            Some(perf_monitor.start_monitoring().await)
        } else {
            None
        };

        // Convert paths to PathBuf for streaming processor
        let path_bufs: Vec<PathBuf> = file_paths.into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();

        let streaming_processor = self.streaming_processor.as_ref().unwrap();

        // Create progress adapter for streaming
        let batch_size = self.config.batch_size;
        let streaming_progress_callback = progress_callback.map(move |callback| {
            Box::new(move |streaming_progress: StreamingProgress| {
                let batch_progress = BatchProgress {
                    current_file: streaming_progress.current_file,
                    files_processed: streaming_progress.files_processed,
                    total_files: streaming_progress.total_files.unwrap_or(total_files),
                    current_batch: (streaming_progress.files_processed / batch_size) + 1,
                    total_batches: (total_files + batch_size - 1) / batch_size,
                    processing_rate: streaming_progress.processing_rate,
                    eta: streaming_progress.eta,
                };
                callback(batch_progress);
            }) as Box<dyn Fn(StreamingProgress) + Send + Sync>
        });

        // Process files as a stream
        let mut stream = streaming_processor
            .process_files_streaming(path_bufs, streaming_progress_callback)
            .await
            .map_err(|e| IngestionError::NetworkError {
                cause: format!("Streaming processing failed: {}", e),
            })?;

        // Collect results
        let mut processed_files = Vec::new();
        let mut files_processed = 0;
        let mut files_failed = 0;
        let mut files_skipped = 0;

        while let Some(result) = stream.next().await {
            match result {
                Ok(processed_file) => {
                    if processed_file.skipped {
                        files_skipped += 1;
                    } else {
                        files_processed += 1;
                        processed_files.push(processed_file);
                    }
                }
                Err(e) => {
                    files_failed += 1;
                    if !self.config.continue_on_error {
                        error!("Streaming processing failed, stopping: {}", e);
                        break;
                    } else {
                        warn!("File processing failed in stream, continuing: {}", e);
                    }
                }
            }

            // Check shutdown signal
            if self.shutdown_signal.load(Ordering::Relaxed) {
                warn!("Shutdown requested, stopping streaming processing");
                break;
            }
        }

        let total_duration = start_time.elapsed();
        let avg_file_duration = if files_processed > 0 {
            total_duration / files_processed as u32
        } else {
            Duration::ZERO
        };

        // Get memory usage from streaming processor
        let (_, peak_memory) = streaming_processor.get_memory_usage();

        let stats = BatchStats {
            files_processed,
            files_failed,
            files_skipped,
            total_duration,
            avg_file_duration,
            peak_memory_bytes: peak_memory,
            batches_processed: (files_processed + self.config.batch_size - 1) / self.config.batch_size,
        };

        info!(
            "Streaming batch processing completed: {} files in {:?} ({:.2} files/sec)",
            files_processed,
            total_duration,
            files_processed as f64 / total_duration.as_secs_f64()
        );

        Ok((processed_files, stats))
    }

    /// Legacy batch processing method (non-streaming)
    async fn process_files_legacy<P>(
        &self,
        file_paths: Vec<P>,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> IngestionResult<(Vec<ProcessedFile>, BatchStats)>
    where
        P: AsRef<Path> + Send + 'static,
    {
        let start_time = Instant::now();
        let total_files = file_paths.len();
        
        info!("Starting batch processing of {} files", total_files);

        // Calculate number of batches
        let total_batches = (total_files + self.config.batch_size - 1) / self.config.batch_size;

        // Setup progress tracking
        let multi_progress = if self.config.show_progress {
            Some(MultiProgress::new())
        } else {
            None
        };

        let main_progress = if let Some(ref mp) = multi_progress {
            let pb = mp.add(ProgressBar::new(total_files as u64));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} files ({per_sec}) {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message("Processing files...");
            Some(pb)
        } else {
            None
        };

        // Setup concurrency control
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency));
        let (result_tx, mut result_rx) = mpsc::channel(self.config.batch_size * 2);

        // Statistics tracking
        let stats = Arc::new(std::sync::Mutex::new(BatchStats::default()));
        let processed_count = Arc::new(AtomicUsize::new(0));

        // Spawn file processing tasks
        let mut join_set = JoinSet::new();
        
        for (file_index, file_path) in file_paths.into_iter().enumerate() {
            let semaphore = Arc::clone(&semaphore);
            let file_processor = Arc::clone(&self.file_processor);
            let result_tx = result_tx.clone();
            let shutdown_signal = Arc::clone(&self.shutdown_signal);
            let memory_monitor = Arc::clone(&self.memory_monitor);
            let config = self.config.clone();

            join_set.spawn(async move {
                // Acquire semaphore permit
                let _permit = semaphore.acquire().await.unwrap();

                // Check shutdown signal
                if shutdown_signal.load(Ordering::Relaxed) {
                    return;
                }

                // Process the file with timeout
                let file_path_ref = file_path.as_ref();
                let process_result = timeout(
                    config.file_timeout,
                    Self::process_single_file(
                        file_processor,
                        file_path_ref,
                        memory_monitor,
                        file_index,
                    ),
                )
                .await;

                let result = match process_result {
                    Ok(Ok(processed_file)) => Ok(processed_file),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(IngestionError::NetworkError {
                        cause: format!("File processing timeout: {}", file_path_ref.display()),
                    }),
                };

                // Send result
                let _ = result_tx.send((file_index, result)).await;
            });
        }

        // Drop the sender to signal completion
        drop(result_tx);

        // Collect results
        let mut processed_files = Vec::new();
        let mut file_results = vec![None; total_files];
        let mut files_processed = 0;
        let mut files_failed = 0;
        let mut files_skipped = 0;

        // Progress tracking
        let mut last_progress_update = Instant::now();

        while let Some((file_index, result)) = result_rx.recv().await {
            match result {
                Ok(processed_file) => {
                    if processed_file.skipped {
                        files_skipped += 1;
                    } else {
                        files_processed += 1;
                        processed_files.push(processed_file.clone());
                    }
                    file_results[file_index] = Some(Ok(processed_file));
                }
                Err(e) => {
                    files_failed += 1;
                    file_results[file_index] = Some(Err(e.clone()));
                    
                    if !self.config.continue_on_error {
                        error!("File processing failed, stopping batch: {}", e);
                        self.shutdown_signal.store(true, Ordering::Relaxed);
                        break;
                    } else {
                        warn!("File processing failed, continuing: {}", e);
                    }
                }
            }

            let total_completed = files_processed + files_failed + files_skipped;
            processed_count.store(total_completed, Ordering::Relaxed);

            // Update progress
            if let Some(ref pb) = main_progress {
                pb.set_position(total_completed as u64);
                pb.set_message(format!(
                    "Processed: {}, Failed: {}, Skipped: {}",
                    files_processed, files_failed, files_skipped
                ));
            }

            // Call progress callback
            if let Some(ref callback) = progress_callback {
                let now = Instant::now();
                if now.duration_since(last_progress_update) >= self.config.progress_update_interval {
                    let elapsed = now.duration_since(start_time);
                    let processing_rate = if elapsed.as_secs_f64() > 0.0 {
                        total_completed as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };

                    let eta = if processing_rate > 0.0 {
                        let remaining_files = total_files - total_completed;
                        Some(Duration::from_secs_f64(remaining_files as f64 / processing_rate))
                    } else {
                        None
                    };

                    let progress = BatchProgress {
                        current_file: None, // Could be enhanced to track current file
                        files_processed: total_completed,
                        total_files,
                        current_batch: total_completed / self.config.batch_size + 1,
                        total_batches,
                        processing_rate,
                        eta,
                    };

                    callback(progress);
                    last_progress_update = now;
                }
            }
        }

        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                warn!("Task join error: {}", e);
            }
        }

        // Finalize progress
        if let Some(ref pb) = main_progress {
            pb.finish_with_message(format!(
                "Completed: {} processed, {} failed, {} skipped",
                files_processed, files_failed, files_skipped
            ));
        }

        let total_duration = start_time.elapsed();
        let avg_file_duration = if files_processed > 0 {
            total_duration / files_processed as u32
        } else {
            Duration::ZERO
        };

        let final_stats = BatchStats {
            files_processed,
            files_failed,
            files_skipped,
            total_duration,
            avg_file_duration,
            peak_memory_bytes: self.memory_monitor.get_peak_usage() as u64,
            batches_processed: total_batches,
        };

        info!(
            "Batch processing completed: {} files in {:?} ({:.2} files/sec)",
            files_processed,
            total_duration,
            files_processed as f64 / total_duration.as_secs_f64()
        );

        Ok((processed_files, final_stats))
    }

    /// Process a single file with memory management
    async fn process_single_file(
        file_processor: Arc<dyn FileProcessor>,
        file_path: &Path,
        memory_monitor: Arc<MemoryMonitor>,
        file_index: usize,
    ) -> IngestionResult<ProcessedFile> {
        debug!("Processing file {}: {}", file_index, file_path.display());

        // Estimate memory usage (rough approximation)
        let file_size = std::fs::metadata(file_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        
        // Reserve memory (estimate 2x file size for processing overhead)
        let estimated_memory = file_size * 2;
        if !memory_monitor.add_usage(estimated_memory) {
            return Err(IngestionError::NetworkError {
                cause: format!("Memory limit exceeded for file: {}", file_path.display()),
            });
        }

        // Process the file
        let result = file_processor.process(file_path).await;

        // Release memory
        memory_monitor.remove_usage(estimated_memory);

        match result {
            Ok(processed_file) => {
                debug!("Successfully processed file {}: {}", file_index, file_path.display());
                Ok(processed_file)
            }
            Err(e) => {
                debug!("Failed to process file {}: {} - {}", file_index, file_path.display(), e);
                Err(IngestionError::NetworkError {
                    cause: format!("File processing failed: {}", e),
                })
            }
        }
    }

    /// Request graceful shutdown of batch processing
    pub fn request_shutdown(&self) {
        info!("Requesting batch processor shutdown");
        self.shutdown_signal.store(true, Ordering::Relaxed);
    }

    /// Check if shutdown has been requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_signal.load(Ordering::Relaxed)
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> (usize, usize) {
        (
            self.memory_monitor.get_current_usage(),
            self.memory_monitor.get_peak_usage(),
        )
    }

    /// Create batches from a list of items
    pub fn create_batches<T: Clone>(items: Vec<T>, batch_size: usize) -> Vec<Vec<T>> {
        items
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Estimate processing time based on file sizes and previous performance
    pub fn estimate_processing_time(
        file_sizes: &[u64],
        previous_stats: Option<&BatchStats>,
    ) -> Duration {
        let total_size: u64 = file_sizes.iter().sum();
        
        if let Some(stats) = previous_stats {
            if stats.files_processed > 0 {
                // Use previous performance data
                let avg_time_per_file = stats.avg_file_duration;
                return avg_time_per_file * file_sizes.len() as u32;
            }
        }

        // Fallback estimation: assume 1MB/second processing rate
        let estimated_seconds = total_size / (1024 * 1024); // 1MB/sec
        Duration::from_secs(estimated_seconds.max(1))
    }

    /// Get performance monitor if available
    pub fn get_performance_monitor(&self) -> Option<Arc<PerformanceMonitor>> {
        self.performance_monitor.as_ref().map(Arc::clone)
    }

    /// Get current system resource utilization
    pub async fn get_resource_utilization(&self) -> Option<(f64, f64)> {
        if let Some(monitor) = &self.performance_monitor {
            if let Ok(util) = monitor.get_current_utilization().await {
                Some((util, util))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if the system is under resource pressure
    pub fn is_under_resource_pressure(&self) -> bool {
        self.performance_monitor
            .as_ref()
            .map_or(false, |monitor| {
                // For now, return false since we can't await in this context
                // In a real implementation, this would need to be async
                false
            })
    }

    /// Get optimization recommendations based on current performance
    pub async fn get_optimization_recommendations(&self) -> Vec<crate::processing::OptimizationRecommendation> {
        if let Some(monitor) = &self.performance_monitor {
            monitor.get_optimization_recommendations().await
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::FileType;
    use crate::error::{ProcessingError, ProcessingResult};
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tempfile::TempDir;
    use tokio_test;

    // Mock file processor for testing
    struct MockFileProcessor {
        delay: Duration,
        should_fail: Arc<Mutex<bool>>,
        processed_files: Arc<Mutex<Vec<PathBuf>>>,
    }

    impl MockFileProcessor {
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
    impl FileProcessor for MockFileProcessor {
        fn can_process(&self, _file_path: &Path) -> bool {
            true
        }

        async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
            // Record that we processed this file
            self.processed_files.lock().unwrap().push(file_path.to_path_buf());

            // Simulate processing delay
            tokio::time::sleep(self.delay).await;

            // Check if we should fail
            if *self.should_fail.lock().unwrap() {
                return Err(ProcessingError::FileReadFailed {
                    path: file_path.display().to_string(),
                    cause: "Mock failure".to_string(),
                });
            }

            // Create a mock processed file
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
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert!(config.max_concurrency > 0);
        assert_eq!(config.file_timeout, Duration::from_secs(30));
        assert_eq!(config.batch_size, 100);
        assert!(config.show_progress);
        assert!(config.continue_on_error);
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1000);

        // Test adding usage within limits
        assert!(monitor.add_usage(500));
        assert_eq!(monitor.get_current_usage(), 500);

        // Test adding more usage within limits
        assert!(monitor.add_usage(300));
        assert_eq!(monitor.get_current_usage(), 800);

        // Test exceeding limits
        assert!(!monitor.add_usage(300)); // Would exceed 1000
        assert_eq!(monitor.get_current_usage(), 800); // Should remain unchanged

        // Test removing usage
        monitor.remove_usage(200);
        assert_eq!(monitor.get_current_usage(), 600);

        // Test peak usage tracking
        assert_eq!(monitor.get_peak_usage(), 800);
    }

    #[tokio::test]
    async fn test_batch_processor_basic() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 5);

        let config = BatchConfig {
            max_concurrency: 2,
            show_progress: false,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockFileProcessor::new(Duration::from_millis(10)));
        let batch_processor = BatchProcessor::new(config, mock_processor.clone());

        let (processed_files, stats) = batch_processor
            .process_files(files.clone(), None)
            .await
            .unwrap();

        assert_eq!(processed_files.len(), 5);
        assert_eq!(stats.files_processed, 5);
        assert_eq!(stats.files_failed, 0);
        assert_eq!(stats.files_skipped, 0);

        // Verify all files were processed
        let processed_paths = mock_processor.get_processed_files();
        assert_eq!(processed_paths.len(), 5);
        for file in &files {
            assert!(processed_paths.contains(file));
        }
    }

    #[tokio::test]
    async fn test_batch_processor_with_failures() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 3);

        let config = BatchConfig {
            max_concurrency: 1,
            show_progress: false,
            continue_on_error: true,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockFileProcessor::new(Duration::from_millis(10)));
        
        // Make the processor fail
        mock_processor.set_should_fail(true);
        
        let batch_processor = BatchProcessor::new(config, mock_processor.clone());

        let (processed_files, stats) = batch_processor
            .process_files(files, None)
            .await
            .unwrap();

        assert_eq!(processed_files.len(), 0); // No successful processing
        assert_eq!(stats.files_processed, 0);
        assert_eq!(stats.files_failed, 3);
        assert_eq!(stats.files_skipped, 0);
    }

    #[tokio::test]
    async fn test_batch_processor_timeout() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 2);

        let config = BatchConfig {
            max_concurrency: 1,
            file_timeout: Duration::from_millis(5), // Very short timeout
            show_progress: false,
            continue_on_error: true,
            ..Default::default()
        };

        // Create a processor with longer delay than timeout
        let mock_processor = Arc::new(MockFileProcessor::new(Duration::from_millis(100)));
        let batch_processor = BatchProcessor::new(config, mock_processor);

        let (processed_files, stats) = batch_processor
            .process_files(files, None)
            .await
            .unwrap();

        assert_eq!(processed_files.len(), 0);
        assert_eq!(stats.files_processed, 0);
        assert_eq!(stats.files_failed, 2); // Both should timeout
    }

    #[tokio::test]
    async fn test_batch_processor_memory_limit() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 3);

        let config = BatchConfig {
            max_concurrency: 1,
            max_memory_bytes: 100, // Very low memory limit
            show_progress: false,
            continue_on_error: true,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockFileProcessor::new(Duration::from_millis(10)));
        let batch_processor = BatchProcessor::new(config, mock_processor);

        let (processed_files, stats) = batch_processor
            .process_files(files, None)
            .await
            .unwrap();

        // Some files should fail due to memory limits
        assert!(stats.files_failed > 0);
    }

    #[tokio::test]
    async fn test_batch_processor_shutdown() {
        let temp_dir = TempDir::new().unwrap();
        let files = create_test_files(&temp_dir, 10);

        let config = BatchConfig {
            max_concurrency: 1,
            show_progress: false,
            ..Default::default()
        };

        let mock_processor = Arc::new(MockFileProcessor::new(Duration::from_millis(100)));
        let batch_processor = BatchProcessor::new(config, mock_processor);

        // Start processing in background
        let batch_processor_clone = batch_processor.clone();
        let files_clone = files.clone();
        let process_task = tokio::spawn(async move {
            batch_processor_clone.process_files(files_clone, None).await
        });

        // Wait a bit then request shutdown
        tokio::time::sleep(Duration::from_millis(50)).await;
        batch_processor.request_shutdown();

        // Wait for processing to complete
        let result = process_task.await.unwrap();
        assert!(result.is_ok());

        // Should have processed fewer files due to shutdown
        let (_, stats) = result.unwrap();
        assert!(stats.files_processed < 10);
    }

    #[test]
    fn test_create_batches() {
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let batches = BatchProcessor::create_batches(items, 3);

        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], vec![1, 2, 3]);
        assert_eq!(batches[1], vec![4, 5, 6]);
        assert_eq!(batches[2], vec![7, 8, 9]);
        assert_eq!(batches[3], vec![10]);
    }

    #[test]
    fn test_estimate_processing_time() {
        let file_sizes = vec![1024, 2048, 4096]; // 7KB total

        // Without previous stats
        let estimated = BatchProcessor::estimate_processing_time(&file_sizes, None);
        assert!(estimated > Duration::ZERO);

        // With previous stats
        let previous_stats = BatchStats {
            files_processed: 10,
            avg_file_duration: Duration::from_millis(100),
            ..Default::default()
        };

        let estimated_with_stats = BatchProcessor::estimate_processing_time(&file_sizes, Some(&previous_stats));
        assert_eq!(estimated_with_stats, Duration::from_millis(300)); // 3 files * 100ms
    }

    #[test]
    fn test_batch_stats_debug() {
        let stats = BatchStats {
            files_processed: 100,
            files_failed: 5,
            files_skipped: 2,
            total_duration: Duration::from_secs(60),
            avg_file_duration: Duration::from_millis(600),
            peak_memory_bytes: 1024 * 1024,
            batches_processed: 10,
        };

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("60s"));
        assert!(debug_str.contains("1048576"));
    }

    #[test]
    fn test_batch_progress_debug() {
        let progress = BatchProgress {
            current_file: Some("test.txt".to_string()),
            files_processed: 50,
            total_files: 100,
            current_batch: 5,
            total_batches: 10,
            processing_rate: 2.5,
            eta: Some(Duration::from_secs(20)),
        };

        let debug_str = format!("{:?}", progress);
        assert!(debug_str.contains("test.txt"));
        assert!(debug_str.contains("50"));
        assert!(debug_str.contains("2.5"));
    }
}