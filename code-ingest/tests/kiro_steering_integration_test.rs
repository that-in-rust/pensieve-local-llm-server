//! Integration tests for .kiro/steering workflow compatibility
//!
//! This test suite validates integration with the existing Kiro steering documents
//! and ensures the generated tasks work correctly with the analysis framework.

use code_ingest::database::Database;
use code_ingest::ingestion::{IngestionEngine, IngestionConfig, IngestionSource};
use code_ingest::tasks::{TaskGenerator, TaskGeneratorConfig, HierarchicalTaskGenerator};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs;

/// Test integration with existing .kiro/steering/spec-S04-steering-doc-analysis.md
#[tokio::test]
async fn test_spec_s04_steering_integration() {
    // Skip if no database URL is provided
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping steering integration test: DATABASE_URL not set");
            return;
        }
    };

    // Create test workspace that mimics the project structure
    let workspace = TempDir::new().unwrap();
    let workspace_path = workspace.path();

    // Create .kiro/steering directory structure
    let kiro_dir = workspace_path.join(".kiro");
    let steering_dir = kiro_dir.join("steering");
    fs::create_dir_all(&steering_dir).await.unwrap();

    // Create the spec-S04-steering-doc-analysis.md file with the actual content
    let steering_doc = steering_dir.join("spec-S04-steering-doc-analysis.md");
    fs::write(
        &steering_doc,
        include_str!("../../.kiro/steering/spec-S04-steering-doc-analysis.md")
    ).await.expect("Should create steering document");

    // Create gringotts/WorkArea directory
    let gringotts_dir = workspace_path.join("gringotts");
    let work_area_dir = gringotts_dir.join("WorkArea");
    fs::create_dir_all(&work_area_dir).await.unwrap();

    // Create .raw_data_202509 directory
    let raw_data_dir = workspace_path.join(".raw_data_202509");
    fs::create_dir_all(&raw_data_dir).await.unwrap();

    // Create a test repository with realistic Rust code
    let test_repo = create_realistic_rust_project(workspace_path).await;

    // Set up database and ingest the test repository
    let db = Database::new(&database_url).await
        .expect("Failed to connect to test database");
    
    db.initialize_schema().await
        .expect("Failed to initialize database schema");

    let ingestion_config = IngestionConfig {
        batch_size: 50,
        max_concurrency: 2,
        include_patterns: vec!["*.rs".to_string(), "*.md".to_string(), "*.toml".to_string()],
        exclude_patterns: vec!["target/*".to_string()],
        ..Default::default()
    };

    let ingestion_engine = IngestionEngine::new(Arc::new(db.clone()), ingestion_config);
    
    let ingestion_result = ingestion_engine
        .ingest_source(IngestionSource::LocalFolder {
            path: test_repo,
            recursive: true,
        })
        .await
        .expect("Ingestion should succeed");

    println!("✓ Ingested {} files into table {}", 
             ingestion_result.total_files, 
             ingestion_result.table_name);

    // Test basic task generation with steering document
    let task_config = TaskGeneratorConfig {
        levels: 4,
        groups_per_level: 7,
        output_directory: workspace_path.to_path_buf(),
        content_directory: raw_data_dir.clone(),
        prompt_file: Some(steering_doc.clone()),
    };

    let task_generator = HierarchicalTaskGenerator::new(Arc::new(db.clone()), task_config);
    
    let task_result = task_generator
        .generate_tasks(&ingestion_result.table_name)
        .await
        .expect("Task generation should succeed");

    println!("✓ Generated {} tasks with steering document integration", 
             task_result.total_tasks);

    // Validate generated task file structure
    let task_files: Vec<_> = fs::read_dir(workspace_path)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();

    let task_file = task_files
        .iter()
        .find(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "md")
                .unwrap_or(false) &&
            entry.file_name().to_str()
                .map(|name| name.contains("tasks"))
                .unwrap_or(false)
        })
        .expect("Should find generated task file");

    // Validate task file content matches expected format
    let task_content = fs::read_to_string(task_file.path()).await
        .expect("Should read task file");

    // Check for proper task structure
    assert!(task_content.contains("- [ ]"), "Should contain task checkboxes");
    assert!(task_content.contains(&format!("Analyze {} row", ingestion_result.table_name)), 
            "Should reference correct table name");
    assert!(task_content.contains("**Content**:"), "Should contain content references");
    assert!(task_content.contains("**Prompt**:"), "Should contain prompt references");
    assert!(task_content.contains("**Output**:"), "Should contain output references");
    assert!(task_content.contains(".raw_data_202509/"), "Should reference correct content directory");
    assert!(task_content.contains("gringotts/WorkArea/"), "Should reference correct output directory");
    assert!(task_content.contains("spec-S04-steering-doc-analysis.md"), 
            "Should reference steering document");

    // Validate content files were created
    let content_files: Vec<_> = fs::read_dir(&raw_data_dir)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();

    assert!(!content_files.is_empty(), "Should create content files");

    // Check for expected content file naming pattern
    let content_file_names: Vec<String> = content_files
        .iter()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();

    let has_content_files = content_file_names
        .iter()
        .any(|name| name.contains("_Content.txt"));
    
    assert!(has_content_files, "Should create _Content.txt files");

    println!("✓ Validated task file structure and content file creation");

    // Test chunked analysis with steering document
    let chunked_config = TaskGeneratorConfig {
        levels: 3,
        groups_per_level: 5,
        chunk_size: Some(300),
        output_directory: workspace_path.to_path_buf(),
        content_directory: raw_data_dir.clone(),
        prompt_file: Some(steering_doc),
    };

    let chunked_generator = HierarchicalTaskGenerator::new(Arc::new(db.clone()), chunked_config);
    
    let chunked_result = chunked_generator
        .generate_tasks(&ingestion_result.table_name)
        .await
        .expect("Chunked task generation should succeed");

    println!("✓ Generated {} chunked tasks with L1/L2 context", 
             chunked_result.total_tasks);

    // Validate chunked table was created
    let chunked_table_name = format!("{}_300", ingestion_result.table_name);
    let chunked_table_exists = db.execute_raw(&format!(
        "SELECT COUNT(*) FROM \"{}\"", 
        chunked_table_name
    )).await.is_ok();

    assert!(chunked_table_exists, "Should create chunked table");

    // Validate L1/L2 content files were created
    let l1_files: Vec<String> = content_file_names
        .iter()
        .filter(|name| name.contains("_L1.txt"))
        .cloned()
        .collect();

    let l2_files: Vec<String> = content_file_names
        .iter()
        .filter(|name| name.contains("_L2.txt"))
        .cloned()
        .collect();

    // Note: L1/L2 files are only created for chunked analysis
    // So we need to check the updated content directory
    let updated_content_files: Vec<_> = fs::read_dir(&raw_data_dir)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();

    let has_l1_files = updated_content_files
        .iter()
        .any(|entry| entry.file_name().to_string_lossy().contains("_L1.txt"));

    let has_l2_files = updated_content_files
        .iter()
        .any(|entry| entry.file_name().to_string_lossy().contains("_L2.txt"));

    if chunked_result.total_tasks > 0 {
        assert!(has_l1_files, "Should create L1 context files for chunked analysis");
        assert!(has_l2_files, "Should create L2 context files for chunked analysis");
    }

    println!("✓ Validated L1/L2 context file creation for chunked analysis");
}

/// Create a realistic Rust project for testing
async fn create_realistic_rust_project(base_path: &std::path::Path) -> PathBuf {
    let project_path = base_path.join("realistic_rust_project");
    fs::create_dir_all(&project_path).await.unwrap();

    // Create src directory
    let src_path = project_path.join("src");
    fs::create_dir_all(&src_path).await.unwrap();

    // Create a complex main.rs that would benefit from L1-L8 analysis
    fs::write(
        src_path.join("main.rs"),
        r#"//! Advanced Rust application demonstrating various patterns
//!
//! This application showcases multiple Rust idioms and patterns that would
//! benefit from systematic analysis using the L1-L8 extraction hierarchy.

use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use std::error::Error;
use std::fmt;

/// Custom error type demonstrating structured error handling
#[derive(Debug)]
pub enum AppError {
    ConfigurationError(String),
    ProcessingError { 
        operation: String, 
        cause: Box<dyn Error + Send + Sync> 
    },
    ResourceExhausted,
    InvalidState(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AppError::ProcessingError { operation, cause } => {
                write!(f, "Processing error in {}: {}", operation, cause)
            }
            AppError::ResourceExhausted => write!(f, "System resources exhausted"),
            AppError::InvalidState(state) => write!(f, "Invalid application state: {}", state),
        }
    }
}

impl Error for AppError {}

/// Configuration management with builder pattern
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub worker_threads: usize,
    pub cache_size: usize,
    pub timeout: Duration,
    pub debug_mode: bool,
    pub features: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            cache_size: 1000,
            timeout: Duration::from_secs(30),
            debug_mode: false,
            features: vec!["basic".to_string()],
        }
    }
}

impl AppConfig {
    pub fn builder() -> AppConfigBuilder {
        AppConfigBuilder::default()
    }
}

/// Builder pattern implementation for configuration
#[derive(Default)]
pub struct AppConfigBuilder {
    worker_threads: Option<usize>,
    cache_size: Option<usize>,
    timeout: Option<Duration>,
    debug_mode: Option<bool>,
    features: Vec<String>,
}

impl AppConfigBuilder {
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = Some(threads);
        self
    }
    
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }
    
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    pub fn debug_mode(mut self, debug: bool) -> Self {
        self.debug_mode = Some(debug);
        self
    }
    
    pub fn feature(mut self, feature: String) -> Self {
        self.features.push(feature);
        self
    }
    
    pub fn build(self) -> AppConfig {
        let mut config = AppConfig::default();
        
        if let Some(threads) = self.worker_threads {
            config.worker_threads = threads;
        }
        if let Some(size) = self.cache_size {
            config.cache_size = size;
        }
        if let Some(timeout) = self.timeout {
            config.timeout = timeout;
        }
        if let Some(debug) = self.debug_mode {
            config.debug_mode = debug;
        }
        if !self.features.is_empty() {
            config.features = self.features;
        }
        
        config
    }
}

/// Thread-safe cache implementation with LRU eviction
pub struct LruCache<K, V> {
    data: Arc<RwLock<HashMap<K, V>>>,
    access_order: Arc<Mutex<BTreeMap<Instant, K>>>,
    max_size: usize,
}

impl<K, V> LruCache<K, V> 
where 
    K: Clone + Eq + std::hash::Hash + Ord,
    V: Clone,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(BTreeMap::new())),
            max_size,
        }
    }
    
    pub fn get(&self, key: &K) -> Option<V> {
        let data = self.data.read().unwrap();
        if let Some(value) = data.get(key) {
            // Update access time
            let mut access_order = self.access_order.lock().unwrap();
            access_order.insert(Instant::now(), key.clone());
            Some(value.clone())
        } else {
            None
        }
    }
    
    pub fn insert(&self, key: K, value: V) -> Result<(), AppError> {
        let mut data = self.data.write().unwrap();
        let mut access_order = self.access_order.lock().unwrap();
        
        // Check if we need to evict
        if data.len() >= self.max_size && !data.contains_key(&key) {
            if let Some((_, oldest_key)) = access_order.iter().next() {
                let oldest_key = oldest_key.clone();
                data.remove(&oldest_key);
                access_order.retain(|_, k| k != &oldest_key);
            }
        }
        
        data.insert(key.clone(), value);
        access_order.insert(Instant::now(), key);
        
        Ok(())
    }
    
    pub fn size(&self) -> usize {
        self.data.read().unwrap().len()
    }
}

/// Worker pool for concurrent task processing
pub struct WorkerPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<WorkItem>,
}

/// Work item for the worker pool
pub struct WorkItem {
    pub id: u64,
    pub data: Vec<u8>,
    pub callback: Box<dyn FnOnce(Vec<u8>) -> Result<Vec<u8>, AppError> + Send>,
}

impl WorkerPool {
    pub fn new(num_workers: usize) -> Result<Self, AppError> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::new();
        
        for id in 0..num_workers {
            let receiver = Arc::clone(&receiver);
            let handle = thread::spawn(move || {
                loop {
                    let work_item = {
                        let receiver = receiver.lock().unwrap();
                        match receiver.recv() {
                            Ok(item) => item,
                            Err(_) => break, // Channel closed
                        }
                    };
                    
                    // Process work item
                    let result = (work_item.callback)(work_item.data);
                    match result {
                        Ok(_) => println!("Worker {} completed task {}", id, work_item.id),
                        Err(e) => eprintln!("Worker {} failed task {}: {}", id, work_item.id, e),
                    }
                }
            });
            
            workers.push(handle);
        }
        
        Ok(Self { workers, sender })
    }
    
    pub fn submit(&self, work_item: WorkItem) -> Result<(), AppError> {
        self.sender.send(work_item)
            .map_err(|_| AppError::InvalidState("Worker pool is shut down".to_string()))
    }
    
    pub fn shutdown(self) -> Result<(), AppError> {
        drop(self.sender); // Close the channel
        
        for worker in self.workers {
            worker.join()
                .map_err(|_| AppError::ProcessingError {
                    operation: "worker_shutdown".to_string(),
                    cause: "Worker thread panicked".into(),
                })?;
        }
        
        Ok(())
    }
}

/// Application state management
pub struct Application {
    config: AppConfig,
    cache: LruCache<String, String>,
    worker_pool: WorkerPool,
    metrics: Arc<Mutex<AppMetrics>>,
}

/// Application metrics for monitoring
#[derive(Debug, Default)]
pub struct AppMetrics {
    pub tasks_processed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub errors: u64,
    pub uptime: Duration,
    pub start_time: Option<Instant>,
}

impl Application {
    pub fn new(config: AppConfig) -> Result<Self, AppError> {
        let cache = LruCache::new(config.cache_size);
        let worker_pool = WorkerPool::new(config.worker_threads)?;
        let mut metrics = AppMetrics::default();
        metrics.start_time = Some(Instant::now());
        
        Ok(Self {
            config,
            cache,
            worker_pool,
            metrics: Arc::new(Mutex::new(metrics)),
        })
    }
    
    pub fn process_data(&self, key: String, data: Vec<u8>) -> Result<Vec<u8>, AppError> {
        // Check cache first
        if let Some(cached_result) = self.cache.get(&key) {
            self.increment_metric("cache_hits");
            return Ok(cached_result.into_bytes());
        }
        
        self.increment_metric("cache_misses");
        
        // Process data (simulate complex processing)
        let processed_data = data.iter()
            .map(|&b| b.wrapping_add(1))
            .collect::<Vec<u8>>();
        
        // Cache the result
        let result_string = String::from_utf8_lossy(&processed_data).to_string();
        self.cache.insert(key, result_string.clone())?;
        
        self.increment_metric("tasks_processed");
        Ok(processed_data)
    }
    
    pub fn submit_async_task<F>(&self, id: u64, data: Vec<u8>, processor: F) -> Result<(), AppError>
    where
        F: FnOnce(Vec<u8>) -> Result<Vec<u8>, AppError> + Send + 'static,
    {
        let work_item = WorkItem {
            id,
            data,
            callback: Box::new(processor),
        };
        
        self.worker_pool.submit(work_item)
    }
    
    pub fn get_metrics(&self) -> AppMetrics {
        let mut metrics = self.metrics.lock().unwrap();
        if let Some(start_time) = metrics.start_time {
            metrics.uptime = start_time.elapsed();
        }
        metrics.clone()
    }
    
    fn increment_metric(&self, metric: &str) {
        let mut metrics = self.metrics.lock().unwrap();
        match metric {
            "cache_hits" => metrics.cache_hits += 1,
            "cache_misses" => metrics.cache_misses += 1,
            "tasks_processed" => metrics.tasks_processed += 1,
            "errors" => metrics.errors += 1,
            _ => {}
        }
    }
    
    pub fn shutdown(self) -> Result<(), AppError> {
        self.worker_pool.shutdown()
    }
}

impl Clone for AppMetrics {
    fn clone(&self) -> Self {
        Self {
            tasks_processed: self.tasks_processed,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            errors: self.errors,
            uptime: self.uptime,
            start_time: self.start_time,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting advanced Rust application...");
    
    // Build configuration using builder pattern
    let config = AppConfig::builder()
        .worker_threads(4)
        .cache_size(500)
        .timeout(Duration::from_secs(10))
        .debug_mode(true)
        .feature("advanced_processing".to_string())
        .feature("metrics".to_string())
        .build();
    
    println!("Configuration: {:?}", config);
    
    // Create application instance
    let app = Application::new(config)?;
    
    // Process some data synchronously
    let test_data = b"Hello, World!".to_vec();
    let result = app.process_data("test_key".to_string(), test_data)?;
    println!("Processed result: {:?}", String::from_utf8_lossy(&result));
    
    // Submit asynchronous tasks
    for i in 0..10 {
        let data = format!("Task data {}", i).into_bytes();
        app.submit_async_task(i, data, |input| {
            // Simulate processing
            thread::sleep(Duration::from_millis(100));
            Ok(input.iter().map(|&b| b.wrapping_mul(2)).collect())
        })?;
    }
    
    // Wait a bit for async tasks to complete
    thread::sleep(Duration::from_secs(2));
    
    // Display metrics
    let metrics = app.get_metrics();
    println!("Application metrics: {:?}", metrics);
    
    // Shutdown gracefully
    app.shutdown()?;
    
    println!("Application shutdown complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = AppConfig::builder()
            .worker_threads(8)
            .cache_size(2000)
            .debug_mode(true)
            .build();
        
        assert_eq!(config.worker_threads, 8);
        assert_eq!(config.cache_size, 2000);
        assert!(config.debug_mode);
    }
    
    #[test]
    fn test_lru_cache() {
        let cache = LruCache::new(2);
        
        cache.insert("key1".to_string(), "value1".to_string()).unwrap();
        cache.insert("key2".to_string(), "value2".to_string()).unwrap();
        
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(cache.size(), 2);
        
        // This should evict key1
        cache.insert("key3".to_string(), "value3".to_string()).unwrap();
        assert_eq!(cache.size(), 2);
        assert_eq!(cache.get(&"key1".to_string()), None);
        assert_eq!(cache.get(&"key3".to_string()), Some("value3".to_string()));
    }
    
    #[test]
    fn test_error_handling() {
        let error = AppError::ConfigurationError("Test error".to_string());
        assert!(error.to_string().contains("Configuration error"));
        
        let processing_error = AppError::ProcessingError {
            operation: "test_op".to_string(),
            cause: "Inner error".into(),
        };
        assert!(processing_error.to_string().contains("Processing error in test_op"));
    }
    
    #[test]
    fn test_application_creation() {
        let config = AppConfig::default();
        let app = Application::new(config);
        assert!(app.is_ok());
    }
}
"#,
    ).await.unwrap();

    // Create lib.rs with additional complexity
    fs::write(
        src_path.join("lib.rs"),
        r#"//! Library module demonstrating advanced Rust patterns
//!
//! This module showcases various Rust idioms that benefit from L1-L8 analysis:
//! - Zero-cost abstractions
//! - Trait-based polymorphism  
//! - Advanced lifetime management
//! - Unsafe code patterns
//! - Performance optimizations

pub mod algorithms;
pub mod data_structures;
pub mod networking;
pub mod async_processing;

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::alloc::{alloc, dealloc, Layout};

/// Zero-cost abstraction for type-safe identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id<T> {
    value: u64,
    _phantom: PhantomData<T>,
}

impl<T> Id<T> {
    pub fn new(value: u64) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
    
    pub fn value(&self) -> u64 {
        self.value
    }
}

/// Trait for objects that can be processed
pub trait Processable {
    type Output;
    type Error;
    
    fn process(&self) -> Result<Self::Output, Self::Error>;
    fn validate(&self) -> bool;
}

/// Generic processor with compile-time optimization
pub struct Processor<T, P> 
where 
    T: Processable,
    P: ProcessingStrategy<T>,
{
    strategy: P,
    _phantom: PhantomData<T>,
}

impl<T, P> Processor<T, P>
where
    T: Processable,
    P: ProcessingStrategy<T>,
{
    pub fn new(strategy: P) -> Self {
        Self {
            strategy,
            _phantom: PhantomData,
        }
    }
    
    pub fn process_item(&self, item: &T) -> Result<T::Output, T::Error> {
        if !item.validate() {
            return self.strategy.handle_invalid(item);
        }
        
        self.strategy.process_valid(item)
    }
    
    pub fn process_batch(&self, items: &[T]) -> Vec<Result<T::Output, T::Error>> {
        items.iter().map(|item| self.process_item(item)).collect()
    }
}

/// Strategy pattern for different processing approaches
pub trait ProcessingStrategy<T: Processable> {
    fn process_valid(&self, item: &T) -> Result<T::Output, T::Error>;
    fn handle_invalid(&self, item: &T) -> Result<T::Output, T::Error>;
}

/// Conservative processing strategy
pub struct ConservativeStrategy;

impl<T: Processable> ProcessingStrategy<T> for ConservativeStrategy {
    fn process_valid(&self, item: &T) -> Result<T::Output, T::Error> {
        item.process()
    }
    
    fn handle_invalid(&self, _item: &T) -> Result<T::Output, T::Error> {
        // Conservative approach: always fail on invalid items
        panic!("Invalid item encountered in conservative mode");
    }
}

/// Optimistic processing strategy
pub struct OptimisticStrategy;

impl<T: Processable> ProcessingStrategy<T> for OptimisticStrategy {
    fn process_valid(&self, item: &T) -> Result<T::Output, T::Error> {
        item.process()
    }
    
    fn handle_invalid(&self, item: &T) -> Result<T::Output, T::Error> {
        // Optimistic approach: try to process anyway
        item.process()
    }
}

/// Custom allocator for performance-critical scenarios
pub struct CustomAllocator {
    pool: NonNull<u8>,
    size: usize,
    offset: usize,
}

impl CustomAllocator {
    /// Create a new custom allocator with pre-allocated memory pool
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it allocates raw memory that must be
    /// properly managed to avoid memory leaks and use-after-free bugs.
    pub unsafe fn new(size: usize) -> Option<Self> {
        let layout = Layout::from_size_align(size, 8).ok()?;
        let ptr = alloc(layout);
        
        if ptr.is_null() {
            return None;
        }
        
        Some(Self {
            pool: NonNull::new_unchecked(ptr),
            size,
            offset: 0,
        })
    }
    
    /// Allocate memory from the pool
    /// 
    /// # Safety
    /// 
    /// The returned pointer is valid for the requested size but the caller
    /// must ensure proper alignment and lifetime management.
    pub unsafe fn allocate(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Align the offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);
        
        if aligned_offset + size > self.size {
            return None; // Out of memory
        }
        
        let ptr = self.pool.as_ptr().add(aligned_offset);
        self.offset = aligned_offset + size;
        
        NonNull::new(ptr)
    }
    
    /// Reset the allocator (doesn't actually free memory, just resets offset)
    pub fn reset(&mut self) {
        self.offset = 0;
    }
    
    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.size.saturating_sub(self.offset)
    }
}

impl Drop for CustomAllocator {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, 8);
            dealloc(self.pool.as_ptr(), layout);
        }
    }
}

/// SIMD-optimized vector operations (when available)
#[cfg(target_arch = "x86_64")]
pub mod simd_ops {
    use std::arch::x86_64::*;
    
    /// SIMD-accelerated vector addition
    /// 
    /// # Safety
    /// 
    /// This function uses unsafe SIMD intrinsics and requires that the input
    /// slices have compatible lengths and alignment.
    pub unsafe fn add_vectors_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        assert!(a.len() % 8 == 0, "Vector length must be multiple of 8");
        
        let chunks = a.len() / 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let vr = _mm256_add_ps(va, vb);
            
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), vr);
        }
    }
    
    /// Check if AVX2 is available at runtime
    pub fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }
}

/// Lock-free data structure for high-performance scenarios
pub struct LockFreeQueue<T> {
    head: std::sync::atomic::AtomicPtr<Node<T>>,
    tail: std::sync::atomic::AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: Option<T>,
    next: std::sync::atomic::AtomicPtr<Node<T>>,
}

impl<T> LockFreeQueue<T> {
    pub fn new() -> Self {
        let dummy = Box::into_raw(Box::new(Node {
            data: None,
            next: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        }));
        
        Self {
            head: std::sync::atomic::AtomicPtr::new(dummy),
            tail: std::sync::atomic::AtomicPtr::new(dummy),
        }
    }
    
    pub fn enqueue(&self, item: T) {
        let new_node = Box::into_raw(Box::new(Node {
            data: Some(item),
            next: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        }));
        
        loop {
            let tail = self.tail.load(std::sync::atomic::Ordering::Acquire);
            let next = unsafe { (*tail).next.load(std::sync::atomic::Ordering::Acquire) };
            
            if next.is_null() {
                if unsafe { (*tail).next.compare_exchange_weak(
                    std::ptr::null_mut(),
                    new_node,
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Relaxed,
                ).is_ok() } {
                    break;
                }
            } else {
                let _ = self.tail.compare_exchange_weak(
                    tail,
                    next,
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }
        
        let _ = self.tail.compare_exchange_weak(
            self.tail.load(std::sync::atomic::Ordering::Acquire),
            new_node,
            std::sync::atomic::Ordering::Release,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    
    pub fn dequeue(&self) -> Option<T> {
        loop {
            let head = self.head.load(std::sync::atomic::Ordering::Acquire);
            let tail = self.tail.load(std::sync::atomic::Ordering::Acquire);
            let next = unsafe { (*head).next.load(std::sync::atomic::Ordering::Acquire) };
            
            if head == tail {
                if next.is_null() {
                    return None; // Queue is empty
                }
                
                let _ = self.tail.compare_exchange_weak(
                    tail,
                    next,
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Relaxed,
                );
            } else {
                if next.is_null() {
                    continue;
                }
                
                let data = unsafe { (*next).data.take() };
                
                if self.head.compare_exchange_weak(
                    head,
                    next,
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Relaxed,
                ).is_ok() {
                    unsafe { Box::from_raw(head) }; // Free old head
                    return data;
                }
            }
        }
    }
}

unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        while self.dequeue().is_some() {}
        
        // Clean up remaining dummy node
        let head = self.head.load(std::sync::atomic::Ordering::Relaxed);
        if !head.is_null() {
            unsafe { Box::from_raw(head) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_id_type_safety() {
        struct User;
        struct Product;
        
        let user_id: Id<User> = Id::new(123);
        let product_id: Id<Product> = Id::new(456);
        
        assert_eq!(user_id.value(), 123);
        assert_eq!(product_id.value(), 456);
        
        // This would not compile due to type safety:
        // assert_eq!(user_id, product_id);
    }
    
    #[test]
    fn test_custom_allocator() {
        unsafe {
            let mut allocator = CustomAllocator::new(1024).unwrap();
            
            let ptr1 = allocator.allocate(64, 8).unwrap();
            let ptr2 = allocator.allocate(128, 8).unwrap();
            
            assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
            assert!(allocator.remaining_capacity() < 1024);
            
            allocator.reset();
            assert_eq!(allocator.remaining_capacity(), 1024);
        }
    }
    
    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::new();
        
        queue.enqueue(1);
        queue.enqueue(2);
        queue.enqueue(3);
        
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);
    }
    
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_operations() {
        if !simd_ops::has_avx2() {
            return; // Skip test if AVX2 not available
        }
        
        let a = vec![1.0f32; 8];
        let b = vec![2.0f32; 8];
        let mut result = vec![0.0f32; 8];
        
        unsafe {
            simd_ops::add_vectors_f32(&a, &b, &mut result);
        }
        
        for &val in &result {
            assert_eq!(val, 3.0);
        }
    }
}
"#,
    ).await.unwrap();

    // Create additional modules to increase complexity
    fs::create_dir_all(src_path.join("algorithms")).await.unwrap();
    fs::write(
        src_path.join("algorithms").join("mod.rs"),
        r#"//! Algorithm implementations showcasing various optimization techniques

pub mod sorting;
pub mod searching;
pub mod graph;

/// Generic trait for algorithms with complexity analysis
pub trait Algorithm<Input, Output> {
    /// Time complexity of the algorithm
    fn time_complexity(&self) -> &'static str;
    
    /// Space complexity of the algorithm  
    fn space_complexity(&self) -> &'static str;
    
    /// Execute the algorithm
    fn execute(&self, input: Input) -> Output;
    
    /// Validate input constraints
    fn validate_input(&self, input: &Input) -> bool;
}
"#,
    ).await.unwrap();

    // Create Cargo.toml
    fs::write(
        project_path.join("Cargo.toml"),
        r#"[package]
name = "realistic-rust-project"
version = "0.2.0"
edition = "2021"
authors = ["Test Author <test@example.com>"]
description = "A realistic Rust project for testing code analysis workflows"
license = "MIT OR Apache-2.0"
repository = "https://github.com/example/realistic-rust-project"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
num_cpus = "1.0"

[dev-dependencies]
criterion = "0.5"
proptest = "1.0"
tempfile = "3.0"

[features]
default = ["std"]
std = []
simd = []
unsafe-optimizations = []

[[bench]]
name = "algorithm_benchmarks"
harness = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.bench]
opt-level = 3
lto = true
codegen-units = 1
debug = false
"#,
    ).await.unwrap();

    // Create comprehensive README
    fs::write(
        project_path.join("README.md"),
        r#"# Realistic Rust Project

A comprehensive Rust application demonstrating advanced patterns and idioms suitable for L1-L8 analysis.

## Architecture Overview

This project showcases multiple layers of Rust complexity:

### L1: Idiomatic Patterns & Micro-Optimizations
- Zero-cost abstractions with `PhantomData`
- RAII resource management
- Efficient error handling with `thiserror`
- SIMD optimizations for performance-critical code

### L2: Design Patterns & Composition  
- Builder pattern for configuration
- Strategy pattern for pluggable algorithms
- Trait-based polymorphism
- Generic programming with type constraints

### L3: Advanced System Programming
- Custom memory allocators
- Lock-free data structures
- Unsafe code with proper safety documentation
- SIMD intrinsics for vectorized operations

### L4: Concurrent & Async Programming
- Thread-safe caching with RwLock
- Worker pool implementation
- Atomic operations for lock-free programming
- Async/await patterns (in separate modules)

## Key Components

### Configuration Management
```rust
let config = AppConfig::builder()
    .worker_threads(8)
    .cache_size(2000)
    .timeout(Duration::from_secs(30))
    .debug_mode(true)
    .build();
```

### Custom Allocator
```rust
unsafe {
    let mut allocator = CustomAllocator::new(1024)?;
    let ptr = allocator.allocate(64, 8)?;
    // Use allocated memory...
}
```

### Lock-Free Queue
```rust
let queue = LockFreeQueue::new();
queue.enqueue(item);
let item = queue.dequeue();
```

## Performance Characteristics

- **Memory Usage**: Bounded by configuration
- **Concurrency**: Scales with CPU cores
- **Allocation**: Custom allocator for hot paths
- **SIMD**: Vectorized operations where supported

## Safety Considerations

This project contains `unsafe` code in several areas:
- Custom memory allocation
- SIMD intrinsics
- Lock-free data structures

All unsafe code is documented with safety requirements and invariants.

## Testing Strategy

- Unit tests for individual components
- Property-based tests with `proptest`
- Benchmarks with `criterion`
- Integration tests for end-to-end workflows

## Build Instructions

```bash
# Standard build
cargo build --release

# With SIMD optimizations
cargo build --release --features simd

# With unsafe optimizations
cargo build --release --features unsafe-optimizations

# Run benchmarks
cargo bench

# Run tests
cargo test
```

## Analysis Opportunities

This codebase is designed to showcase patterns that benefit from systematic analysis:

1. **Performance Optimization**: SIMD usage, allocation patterns, lock contention
2. **Safety Analysis**: Unsafe code blocks, memory management, concurrency
3. **Architecture Review**: Design patterns, abstraction boundaries, modularity
4. **Code Quality**: Error handling, testing coverage, documentation

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
"#,
    ).await.unwrap();

    project_path
}

/// Test that validates the complete workflow produces expected task structure
#[tokio::test]
async fn test_task_structure_validation() {
    // Skip if no database URL is provided
    let database_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("Skipping task structure validation: DATABASE_URL not set");
            return;
        }
    };

    let workspace = TempDir::new().unwrap();
    let workspace_path = workspace.path();

    // Create minimal test setup
    let test_repo = workspace_path.join("test_repo");
    fs::create_dir_all(&test_repo).await.unwrap();
    
    let src_dir = test_repo.join("src");
    fs::create_dir_all(&src_dir).await.unwrap();
    
    fs::write(
        src_dir.join("main.rs"),
        "fn main() { println!(\"Hello, world!\"); }"
    ).await.unwrap();

    // Set up database and ingest
    let db = Database::new(&database_url).await.unwrap();
    db.initialize_schema().await.unwrap();

    let ingestion_config = IngestionConfig::default();
    let ingestion_engine = IngestionEngine::new(Arc::new(db.clone()), ingestion_config);
    
    let ingestion_result = ingestion_engine
        .ingest_source(IngestionSource::LocalFolder {
            path: test_repo,
            recursive: true,
        })
        .await
        .unwrap();

    // Generate tasks with specific structure
    let task_config = TaskGeneratorConfig {
        levels: 4,
        groups_per_level: 7,
        output_directory: workspace_path.to_path_buf(),
        content_directory: workspace_path.join(".raw_data_202509"),
        prompt_file: None,
    };

    let task_generator = HierarchicalTaskGenerator::new(Arc::new(db), task_config);
    let task_result = task_generator
        .generate_tasks(&ingestion_result.table_name)
        .await
        .unwrap();

    // Validate task structure matches expected format
    let task_files: Vec<_> = fs::read_dir(workspace_path)
        .await
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .await
        .unwrap();

    let task_file = task_files
        .iter()
        .find(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "md")
                .unwrap_or(false)
        })
        .expect("Should find task file");

    let task_content = fs::read_to_string(task_file.path()).await.unwrap();

    // Validate hierarchical structure
    assert!(task_content.contains("- [ ] 1."), "Should have level 1 tasks");
    assert!(task_content.contains("- [ ] 1.1."), "Should have level 2 tasks");
    assert!(task_content.contains("- [ ] 1.1.1."), "Should have level 3 tasks");
    assert!(task_content.contains("- [ ] 1.1.1.1."), "Should have level 4 tasks");

    // Validate content references
    assert!(task_content.contains("**Content**:"), "Should have content references");
    assert!(task_content.contains("_Content.txt"), "Should reference content files");
    assert!(task_content.contains("**Output**:"), "Should have output references");
    assert!(task_content.contains("gringotts/WorkArea/"), "Should reference output directory");

    println!("✓ Task structure validation completed successfully");
}