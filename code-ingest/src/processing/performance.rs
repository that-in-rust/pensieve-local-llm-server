//! Performance monitoring and optimization module
//! 
//! This module provides comprehensive performance monitoring, resource tracking,
//! and adaptive optimization for the code ingestion system.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{System, Pid};
use tracing::{debug, info, warn};

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Interval for collecting performance metrics
    pub collection_interval: Duration,
    /// Number of historical samples to keep
    pub history_size: usize,
    /// CPU utilization threshold for warnings (0-100)
    pub cpu_warning_threshold: f64,
    /// Memory utilization threshold for warnings (0-100)
    pub memory_warning_threshold: f64,
    /// Whether to enable detailed process monitoring
    pub detailed_monitoring: bool,
    /// Whether to enable adaptive optimization
    pub adaptive_optimization: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(1),
            history_size: 300, // 5 minutes at 1-second intervals
            cpu_warning_threshold: 90.0,
            memory_warning_threshold: 85.0,
            detailed_monitoring: true,
            adaptive_optimization: true,
        }
    }
}

/// System performance metrics at a point in time
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp of the measurement
    pub timestamp: SystemTime,
    /// CPU utilization percentage (0-100)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Total system memory in bytes
    pub total_memory: u64,
    /// Memory utilization percentage (0-100)
    pub memory_percentage: f64,
    /// Number of active processing tasks
    pub active_tasks: usize,
    /// Current processing rate (files per second)
    pub processing_rate: f64,
    /// Disk I/O read bytes per second
    pub disk_read_bps: u64,
    /// Disk I/O write bytes per second
    pub disk_write_bps: u64,
    /// Network I/O bytes per second (if applicable)
    pub network_bps: u64,
}

impl PerformanceSnapshot {
    /// Check if this snapshot indicates resource pressure
    pub fn has_resource_pressure(&self, config: &PerformanceConfig) -> bool {
        self.cpu_usage > config.cpu_warning_threshold
            || self.memory_percentage > config.memory_warning_threshold
    }

    /// Get a resource pressure score (0.0 = no pressure, 1.0 = maximum pressure)
    pub fn resource_pressure_score(&self) -> f64 {
        let cpu_pressure = (self.cpu_usage / 100.0).min(1.0);
        let memory_pressure = (self.memory_percentage / 100.0).min(1.0);
        cpu_pressure.max(memory_pressure)
    }
}

/// Performance statistics over a time period
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Average CPU utilization
    pub avg_cpu_usage: f64,
    /// Peak CPU utilization
    pub peak_cpu_usage: f64,
    /// Average memory usage in bytes
    pub avg_memory_usage: u64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Average processing rate (files per second)
    pub avg_processing_rate: f64,
    /// Peak processing rate (files per second)
    pub peak_processing_rate: f64,
    /// Total files processed during the period
    pub total_files_processed: usize,
    /// Time period covered by these statistics
    pub time_period: Duration,
    /// Number of resource pressure events
    pub pressure_events: usize,
}

/// Adaptive optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommended concurrency level
    pub recommended_concurrency: usize,
    /// Recommended batch size
    pub recommended_batch_size: usize,
    /// Recommended memory limit per task
    pub recommended_memory_per_task: u64,
    /// Reason for the recommendation
    pub reason: String,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
}

/// Performance monitor that tracks system resources and processing metrics
pub struct PerformanceMonitor {
    config: PerformanceConfig,
    system: Arc<Mutex<System>>,
    process_id: Pid,
    history: Arc<Mutex<VecDeque<PerformanceSnapshot>>>,
    active_tasks: Arc<AtomicUsize>,
    files_processed: Arc<AtomicUsize>,
    start_time: Instant,
    last_disk_stats: Arc<Mutex<Option<(u64, u64)>>>, // (read_bytes, write_bytes)
    monitoring_active: Arc<AtomicUsize>, // 0 = stopped, 1 = running
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: PerformanceConfig) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let process_id = sysinfo::get_current_pid().unwrap_or(Pid::from(0));
        let history_size = config.history_size;

        Self {
            config,
            system: Arc::new(Mutex::new(system)),
            process_id,
            history: Arc::new(Mutex::new(VecDeque::with_capacity(history_size))),
            active_tasks: Arc::new(AtomicUsize::new(0)),
            files_processed: Arc::new(AtomicUsize::new(0)),
            start_time: Instant::now(),
            last_disk_stats: Arc::new(Mutex::new(None)),
            monitoring_active: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Start performance monitoring in the background
    pub async fn start_monitoring(&self) -> tokio::task::JoinHandle<()> {
        if self.monitoring_active.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            panic!("Performance monitoring is already active");
        }

        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.monitoring_loop().await;
        })
    }

    /// Stop performance monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(0, Ordering::Relaxed);
    }

    /// Main monitoring loop
    async fn monitoring_loop(&self) {
        let mut interval = tokio::time::interval(self.config.collection_interval);
        
        info!("Performance monitoring started");

        while self.monitoring_active.load(Ordering::Relaxed) == 1 {
            interval.tick().await;
            
            if let Ok(snapshot) = self.collect_snapshot() {
                self.add_snapshot(snapshot.clone());
                
                if snapshot.has_resource_pressure(&self.config) {
                    warn!(
                        "Resource pressure detected: CPU {:.1}%, Memory {:.1}%",
                        snapshot.cpu_usage, snapshot.memory_percentage
                    );
                }

                debug!(
                    "Performance: CPU {:.1}%, Memory {:.1}%, Tasks {}, Rate {:.1} files/sec",
                    snapshot.cpu_usage,
                    snapshot.memory_percentage,
                    snapshot.active_tasks,
                    snapshot.processing_rate
                );
            }
        }

        info!("Performance monitoring stopped");
    }

    /// Collect a performance snapshot
    pub fn collect_snapshot(&self) -> Result<PerformanceSnapshot, Box<dyn std::error::Error>> {
        let mut system = self.system.lock().unwrap();
        system.refresh_all();

        // Get CPU usage
        let cpu_usage = system.global_cpu_info().cpu_usage() as f64;

        // Get memory usage
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let memory_percentage = (used_memory as f64 / total_memory as f64) * 100.0;

        // Get process-specific memory if detailed monitoring is enabled
        let process_memory = if self.config.detailed_monitoring {
            system.process(self.process_id)
                .map(|p| p.memory())
                .unwrap_or(0)
        } else {
            used_memory
        };

        // Calculate processing rate
        let elapsed = self.start_time.elapsed();
        let total_processed = self.files_processed.load(Ordering::Relaxed);
        let processing_rate = if elapsed.as_secs_f64() > 0.0 {
            total_processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Get disk I/O stats (simplified - would need platform-specific implementation)
        let (disk_read_bps, disk_write_bps) = self.calculate_disk_io_rate();

        let snapshot = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            cpu_usage,
            memory_usage: process_memory,
            total_memory,
            memory_percentage,
            active_tasks: self.active_tasks.load(Ordering::Relaxed),
            processing_rate,
            disk_read_bps,
            disk_write_bps,
            network_bps: 0, // Not implemented in this version
        };

        Ok(snapshot)
    }

    /// Calculate disk I/O rate based on previous measurements
    fn calculate_disk_io_rate(&self) -> (u64, u64) {
        // This is a simplified implementation
        // In a real system, you would track actual disk I/O statistics
        let mut last_stats = self.last_disk_stats.lock().unwrap();
        
        // For now, return zeros as we don't have actual disk I/O tracking
        // In production, you would use platform-specific APIs:
        // - /proc/diskstats on Linux
        // - Performance counters on Windows
        // - iostat on macOS
        
        *last_stats = Some((0, 0));
        (0, 0)
    }

    /// Add a snapshot to the history
    fn add_snapshot(&self, snapshot: PerformanceSnapshot) {
        let mut history = self.history.lock().unwrap();
        
        if history.len() >= self.config.history_size {
            history.pop_front();
        }
        
        history.push_back(snapshot);
    }

    /// Get performance statistics for a time period
    pub fn get_stats(&self, duration: Option<Duration>) -> PerformanceStats {
        let history = self.history.lock().unwrap();
        
        let cutoff_time = duration.map(|d| SystemTime::now() - d);
        
        let relevant_snapshots: Vec<&PerformanceSnapshot> = history
            .iter()
            .filter(|s| {
                cutoff_time.map_or(true, |cutoff| s.timestamp >= cutoff)
            })
            .collect();

        if relevant_snapshots.is_empty() {
            return PerformanceStats {
                avg_cpu_usage: 0.0,
                peak_cpu_usage: 0.0,
                avg_memory_usage: 0,
                peak_memory_usage: 0,
                avg_processing_rate: 0.0,
                peak_processing_rate: 0.0,
                total_files_processed: 0,
                time_period: Duration::ZERO,
                pressure_events: 0,
            };
        }

        let count = relevant_snapshots.len() as f64;
        
        let avg_cpu_usage = relevant_snapshots.iter().map(|s| s.cpu_usage).sum::<f64>() / count;
        let peak_cpu_usage = relevant_snapshots.iter().map(|s| s.cpu_usage).fold(0.0, f64::max);
        
        let avg_memory_usage = (relevant_snapshots.iter().map(|s| s.memory_usage).sum::<u64>() as f64 / count) as u64;
        let peak_memory_usage = relevant_snapshots.iter().map(|s| s.memory_usage).max().unwrap_or(0);
        
        let avg_processing_rate = relevant_snapshots.iter().map(|s| s.processing_rate).sum::<f64>() / count;
        let peak_processing_rate = relevant_snapshots.iter().map(|s| s.processing_rate).fold(0.0, f64::max);
        
        let pressure_events = relevant_snapshots
            .iter()
            .filter(|s| s.has_resource_pressure(&self.config))
            .count();

        let time_period = if relevant_snapshots.len() > 1 {
            relevant_snapshots.last().unwrap().timestamp
                .duration_since(relevant_snapshots.first().unwrap().timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        };

        PerformanceStats {
            avg_cpu_usage,
            peak_cpu_usage,
            avg_memory_usage,
            peak_memory_usage,
            avg_processing_rate,
            peak_processing_rate,
            total_files_processed: self.files_processed.load(Ordering::Relaxed),
            time_period,
            pressure_events,
        }
    }

    /// Get optimization recommendations based on performance history
    pub fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        if !self.config.adaptive_optimization {
            return Vec::new();
        }

        let stats = self.get_stats(Some(Duration::from_secs(60))); // Last minute
        let mut recommendations = Vec::new();

        // CPU-based recommendations
        if stats.avg_cpu_usage > 90.0 {
            recommendations.push(OptimizationRecommendation {
                recommended_concurrency: (num_cpus::get() / 2).max(1),
                recommended_batch_size: 50,
                recommended_memory_per_task: 32 * 1024 * 1024, // 32MB
                reason: "High CPU usage detected - reducing concurrency".to_string(),
                confidence: 0.8,
            });
        } else if stats.avg_cpu_usage < 50.0 && stats.pressure_events == 0 {
            recommendations.push(OptimizationRecommendation {
                recommended_concurrency: num_cpus::get() * 2,
                recommended_batch_size: 200,
                recommended_memory_per_task: 64 * 1024 * 1024, // 64MB
                reason: "Low CPU usage detected - increasing concurrency".to_string(),
                confidence: 0.7,
            });
        }

        // Memory-based recommendations
        if stats.peak_memory_usage > (stats.avg_memory_usage * 2) {
            recommendations.push(OptimizationRecommendation {
                recommended_concurrency: num_cpus::get(),
                recommended_batch_size: 25,
                recommended_memory_per_task: 16 * 1024 * 1024, // 16MB
                reason: "High memory variance detected - reducing memory per task".to_string(),
                confidence: 0.6,
            });
        }

        // Processing rate recommendations
        if stats.avg_processing_rate < 1.0 && stats.pressure_events > 0 {
            recommendations.push(OptimizationRecommendation {
                recommended_concurrency: (num_cpus::get() / 2).max(1),
                recommended_batch_size: 10,
                recommended_memory_per_task: 8 * 1024 * 1024, // 8MB
                reason: "Low processing rate with resource pressure - conservative settings".to_string(),
                confidence: 0.9,
            });
        }

        recommendations
    }

    /// Update active task count
    pub fn set_active_tasks(&self, count: usize) {
        self.active_tasks.store(count, Ordering::Relaxed);
    }

    /// Increment files processed counter
    pub fn increment_files_processed(&self) {
        self.files_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current resource utilization
    pub fn get_current_utilization(&self) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let snapshot = self.collect_snapshot()?;
        Ok((snapshot.cpu_usage, snapshot.memory_percentage))
    }

    /// Check if system is under resource pressure
    pub fn is_under_pressure(&self) -> bool {
        if let Ok(snapshot) = self.collect_snapshot() {
            snapshot.has_resource_pressure(&self.config)
        } else {
            false
        }
    }

    /// Get estimated completion time based on current performance
    pub fn estimate_completion_time(&self, remaining_files: usize) -> Option<Duration> {
        let stats = self.get_stats(Some(Duration::from_secs(30))); // Last 30 seconds
        
        if stats.avg_processing_rate > 0.0 {
            let seconds = remaining_files as f64 / stats.avg_processing_rate;
            Some(Duration::from_secs_f64(seconds))
        } else {
            None
        }
    }
}

impl Clone for PerformanceMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            system: Arc::clone(&self.system),
            process_id: self.process_id,
            history: Arc::clone(&self.history),
            active_tasks: Arc::clone(&self.active_tasks),
            files_processed: Arc::clone(&self.files_processed),
            start_time: self.start_time,
            last_disk_stats: Arc::clone(&self.last_disk_stats),
            monitoring_active: Arc::clone(&self.monitoring_active),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert_eq!(config.collection_interval, Duration::from_secs(1));
        assert_eq!(config.history_size, 300);
        assert_eq!(config.cpu_warning_threshold, 90.0);
        assert_eq!(config.memory_warning_threshold, 85.0);
        assert!(config.detailed_monitoring);
        assert!(config.adaptive_optimization);
    }

    #[test]
    fn test_performance_snapshot_pressure_detection() {
        let snapshot = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            cpu_usage: 95.0,
            memory_usage: 1024 * 1024 * 1024, // 1GB
            total_memory: 2 * 1024 * 1024 * 1024, // 2GB
            memory_percentage: 50.0,
            active_tasks: 4,
            processing_rate: 2.5,
            disk_read_bps: 0,
            disk_write_bps: 0,
            network_bps: 0,
        };

        let config = PerformanceConfig::default();
        assert!(snapshot.has_resource_pressure(&config)); // CPU > 90%

        let pressure_score = snapshot.resource_pressure_score();
        assert!(pressure_score > 0.9); // High CPU usage
    }

    #[test]
    fn test_performance_monitor_creation() {
        let config = PerformanceConfig::default();
        let monitor = PerformanceMonitor::new(config);
        
        assert_eq!(monitor.active_tasks.load(Ordering::Relaxed), 0);
        assert_eq!(monitor.files_processed.load(Ordering::Relaxed), 0);
        assert_eq!(monitor.monitoring_active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_performance_monitor_snapshot_collection() {
        let config = PerformanceConfig::default();
        let monitor = PerformanceMonitor::new(config);
        
        let snapshot = monitor.collect_snapshot();
        assert!(snapshot.is_ok());
        
        let snapshot = snapshot.unwrap();
        assert!(snapshot.cpu_usage >= 0.0);
        assert!(snapshot.memory_percentage >= 0.0);
        assert!(snapshot.total_memory > 0);
    }

    #[test]
    fn test_performance_stats_empty_history() {
        let config = PerformanceConfig::default();
        let monitor = PerformanceMonitor::new(config);
        
        let stats = monitor.get_stats(None);
        assert_eq!(stats.avg_cpu_usage, 0.0);
        assert_eq!(stats.peak_cpu_usage, 0.0);
        assert_eq!(stats.total_files_processed, 0);
        assert_eq!(stats.pressure_events, 0);
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = PerformanceConfig {
            adaptive_optimization: true,
            ..Default::default()
        };
        let monitor = PerformanceMonitor::new(config);
        
        // Add some mock snapshots to history
        let high_cpu_snapshot = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            cpu_usage: 95.0,
            memory_usage: 512 * 1024 * 1024,
            total_memory: 2 * 1024 * 1024 * 1024,
            memory_percentage: 25.0,
            active_tasks: 8,
            processing_rate: 0.5,
            disk_read_bps: 0,
            disk_write_bps: 0,
            network_bps: 0,
        };
        
        monitor.add_snapshot(high_cpu_snapshot);
        
        let recommendations = monitor.get_optimization_recommendations();
        assert!(!recommendations.is_empty());
        
        // Should recommend reducing concurrency due to high CPU
        let cpu_rec = recommendations.iter().find(|r| r.reason.contains("CPU"));
        assert!(cpu_rec.is_some());
        assert!(cpu_rec.unwrap().recommended_concurrency < num_cpus::get() * 2);
    }

    #[test]
    fn test_performance_monitor_task_tracking() {
        let config = PerformanceConfig::default();
        let monitor = PerformanceMonitor::new(config);
        
        monitor.set_active_tasks(5);
        assert_eq!(monitor.active_tasks.load(Ordering::Relaxed), 5);
        
        monitor.increment_files_processed();
        monitor.increment_files_processed();
        assert_eq!(monitor.files_processed.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_completion_time_estimation() {
        let config = PerformanceConfig::default();
        let monitor = PerformanceMonitor::new(config);
        
        // With no processing history, should return None
        let eta = monitor.estimate_completion_time(100);
        assert!(eta.is_none());
        
        // Add a snapshot with processing rate
        let snapshot = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            cpu_usage: 50.0,
            memory_usage: 512 * 1024 * 1024,
            total_memory: 2 * 1024 * 1024 * 1024,
            memory_percentage: 25.0,
            active_tasks: 4,
            processing_rate: 10.0, // 10 files per second
            disk_read_bps: 0,
            disk_write_bps: 0,
            network_bps: 0,
        };
        
        monitor.add_snapshot(snapshot);
        
        let eta = monitor.estimate_completion_time(100);
        assert!(eta.is_some());
        
        let eta_duration = eta.unwrap();
        // Should be approximately 100 / 10 = 10 seconds
        assert!(eta_duration.as_secs() >= 9 && eta_duration.as_secs() <= 11);
    }

    #[tokio::test]
    async fn test_performance_monitor_lifecycle() {
        let config = PerformanceConfig {
            collection_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let monitor = PerformanceMonitor::new(config);
        
        // Start monitoring
        let handle = monitor.start_monitoring().await;
        assert_eq!(monitor.monitoring_active.load(Ordering::Relaxed), 1);
        
        // Let it run for a short time
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        // Stop monitoring
        monitor.stop_monitoring();
        
        // Wait for the task to complete
        tokio::time::sleep(Duration::from_millis(20)).await;
        handle.abort(); // Ensure the task is stopped
        
        assert_eq!(monitor.monitoring_active.load(Ordering::Relaxed), 0);
    }
}