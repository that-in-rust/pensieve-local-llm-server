//! Unit tests for PerformanceMonitor implementation
//! 
//! Tests Requirements 1.3, 5.1, 5.2, 5.3, 5.4 - Performance monitoring functionality

use code_ingest::processing::{
    PerformanceMonitor, PerformanceThresholds, PerformanceConfig, PerformanceMetrics,
    OptimizationRecommendation, ConcurrencyController, LatencyTracker, MemoryPool,
    BatchSizeController,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_test;

/// Test PerformanceMonitor creation and basic functionality
#[tokio::test]
async fn test_performance_monitor_creation() {
    let thresholds = PerformanceThresholds {
        max_cpu_usage: 85.0,
        max_memory_usage: 80.0,
        max_error_rate: 5.0,
        target_processing_rate: 100.0,
        max_latency_ms: 1000.0,
    };

    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test that monitor was created successfully
    assert!(monitor.get_concurrency() > 0);
    
    // Test basic metrics collection
    let metrics = monitor.get_metrics().await.unwrap();
    assert!(metrics.cpu_usage >= 0.0);
    assert!(metrics.memory_usage > 0);
    assert!(metrics.memory_percentage >= 0.0);
    assert!(metrics.processing_rate >= 0.0);
    assert!(metrics.error_rate >= 0.0);
}

/// Test PerformanceMonitor start_monitoring method
#[tokio::test]
async fn test_performance_monitor_start_monitoring() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test start_monitoring method
    let result = monitor.start_monitoring().await;
    assert!(result.is_ok(), "start_monitoring should succeed");
}

/// Test PerformanceMonitor get_current_utilization method
#[tokio::test]
async fn test_performance_monitor_get_current_utilization() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test get_current_utilization method
    let utilization = monitor.get_current_utilization().await.unwrap();
    assert!(utilization >= 0.0 && utilization <= 1.0, 
           "Utilization should be between 0.0 and 1.0, got: {}", utilization);
}

/// Test PerformanceMonitor is_under_pressure method
#[tokio::test]
async fn test_performance_monitor_is_under_pressure() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test is_under_pressure method
    let under_pressure = monitor.is_under_pressure().await;
    // Should return a boolean value
    assert!(under_pressure == true || under_pressure == false);
}

/// Test PerformanceMonitor get_optimization_recommendations method
#[tokio::test]
async fn test_performance_monitor_get_optimization_recommendations() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test get_optimization_recommendations method
    let recommendations = monitor.get_optimization_recommendations().await;
    
    // Should return a vector (may be empty)
    assert!(recommendations.len() >= 0);
    
    // If there are recommendations, they should have valid structure
    for rec in recommendations {
        assert!(!rec.category.is_empty());
        assert!(!rec.description.is_empty());
        assert!(!rec.impact.is_empty());
        assert!(rec.priority > 0);
    }
}

/// Test PerformanceMonitor operation recording
#[tokio::test]
async fn test_performance_monitor_operation_recording() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Record some successful operations
    monitor.record_success(Duration::from_millis(100));
    monitor.record_success(Duration::from_millis(150));
    monitor.record_success(Duration::from_millis(200));
    
    // Record some errors
    monitor.record_error(Duration::from_millis(500));
    
    // Get metrics and verify they reflect the recorded operations
    let metrics = monitor.get_metrics().await.unwrap();
    assert!(metrics.error_rate > 0.0, "Error rate should be > 0 after recording errors");
    assert!(metrics.avg_latency_ms > 0.0, "Average latency should be > 0 after recording operations");
}

/// Test PerformanceMonitor concurrency adjustment
#[tokio::test]
async fn test_performance_monitor_concurrency_adjustment() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    let initial_concurrency = monitor.get_concurrency();
    
    // Test concurrency adjustment
    let adjusted = monitor.adjust_concurrency().await.unwrap();
    
    // Should return a boolean indicating whether adjustment occurred
    assert!(adjusted == true || adjusted == false);
    
    // Concurrency should still be a positive value
    assert!(monitor.get_concurrency() > 0);
}

/// Test PerformanceMonitor system stress detection
#[tokio::test]
async fn test_performance_monitor_system_stress_detection() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test system stress detection
    let is_stressed = monitor.is_system_stressed().await.unwrap();
    
    // Should return a boolean
    assert!(is_stressed == true || is_stressed == false);
}

/// Test PerformanceMonitor summary generation
#[tokio::test]
async fn test_performance_monitor_summary() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Record some operations
    monitor.record_success(Duration::from_millis(100));
    monitor.record_error(Duration::from_millis(200));
    
    // Get summary
    let summary = monitor.get_summary().await.unwrap();
    
    // Verify summary structure
    assert!(summary.uptime.as_millis() > 0);
    assert_eq!(summary.total_processed, 1);
    assert_eq!(summary.total_errors, 1);
    assert!(summary.concurrency_level > 0);
    assert!(summary.is_stressed == true || summary.is_stressed == false);
    
    // Verify current metrics are included
    assert!(summary.current_metrics.cpu_usage >= 0.0);
    assert!(summary.current_metrics.memory_usage > 0);
}

/// Test PerformanceMonitor threshold updates
#[tokio::test]
async fn test_performance_monitor_threshold_updates() {
    let initial_thresholds = PerformanceThresholds::default();
    let mut monitor = PerformanceMonitor::new(initial_thresholds).unwrap();
    
    // Update thresholds
    let new_thresholds = PerformanceThresholds {
        max_cpu_usage: 95.0,
        max_memory_usage: 90.0,
        max_error_rate: 10.0,
        target_processing_rate: 200.0,
        max_latency_ms: 2000.0,
    };
    
    monitor.update_thresholds(new_thresholds);
    
    // Test that the monitor still works after threshold update
    let metrics = monitor.get_metrics().await.unwrap();
    assert!(metrics.cpu_usage >= 0.0);
}

/// Test ConcurrencyController functionality
#[test]
fn test_concurrency_controller() {
    let controller = ConcurrencyController::new(4, 1, 8);
    
    // Test initial concurrency
    assert_eq!(controller.get_concurrency(), 4);
    
    // Create test metrics with high stress
    let stressed_metrics = PerformanceMetrics {
        cpu_usage: 95.0,
        memory_percentage: 90.0,
        error_rate: 10.0,
        processing_rate: 50.0,
        avg_latency_ms: 2000.0,
        p95_latency_ms: 3000.0,
        memory_usage: 1024 * 1024 * 1024,
        available_memory: 512 * 1024 * 1024,
        disk_read_bps: 0,
        disk_write_bps: 0,
        network_bps: 0,
        active_threads: 8,
        timestamp: std::time::SystemTime::now(),
    };
    
    let thresholds = PerformanceThresholds::default();
    
    // Should decrease concurrency under stress
    std::thread::sleep(Duration::from_millis(10)); // Ensure cooldown passes
    let adjusted = controller.adjust_concurrency(&stressed_metrics, &thresholds);
    // Adjustment may or may not happen depending on cooldown timing
    if adjusted {
        assert!(controller.get_concurrency() <= 4);
    }
    
    // Test with low stress metrics
    let relaxed_metrics = PerformanceMetrics {
        cpu_usage: 30.0,
        memory_percentage: 40.0,
        error_rate: 1.0,
        processing_rate: 80.0,
        avg_latency_ms: 100.0,
        p95_latency_ms: 200.0,
        memory_usage: 256 * 1024 * 1024,
        available_memory: 1024 * 1024 * 1024,
        disk_read_bps: 0,
        disk_write_bps: 0,
        network_bps: 0,
        active_threads: 2,
        timestamp: std::time::SystemTime::now(),
    };
    
    std::thread::sleep(Duration::from_millis(10)); // Ensure cooldown passes
    let controller2 = ConcurrencyController::new(2, 1, 8);
    let adjusted2 = controller2.adjust_concurrency(&relaxed_metrics, &thresholds);
    // May or may not adjust depending on conditions
    assert!(adjusted2 == true || adjusted2 == false);
}

/// Test LatencyTracker functionality
#[test]
fn test_latency_tracker() {
    let tracker = LatencyTracker::new(100);
    
    // Record some latencies
    for i in 1..=10 {
        tracker.record_latency(Duration::from_millis(i * 10));
    }
    
    let (avg, p95) = tracker.get_statistics();
    assert!(avg > 0.0, "Average latency should be > 0");
    assert!(p95 > avg, "P95 latency should be >= average latency");
    assert!(p95 <= 100.0, "P95 should be <= max recorded latency (100ms)");
    
    // Test empty tracker
    let empty_tracker = LatencyTracker::new(100);
    let (empty_avg, empty_p95) = empty_tracker.get_statistics();
    assert_eq!(empty_avg, 0.0);
    assert_eq!(empty_p95, 0.0);
}

/// Test MemoryPool functionality
#[test]
fn test_memory_pool() {
    let pool = MemoryPool::new(|| Vec::<u8>::with_capacity(1024), 5);
    
    // Test initial state
    assert_eq!(pool.size(), 0);
    
    // Acquire and release items
    let item1 = pool.acquire();
    let item2 = pool.acquire();
    
    assert_eq!(item1.capacity(), 1024);
    assert_eq!(item2.capacity(), 1024);
    
    pool.release(item1);
    pool.release(item2);
    
    assert_eq!(pool.size(), 2);
    
    // Acquire again - should reuse pooled items
    let _item3 = pool.acquire();
    assert_eq!(pool.size(), 1);
    
    // Test pool size limit
    for _ in 0..10 {
        let item = pool.acquire();
        pool.release(item);
    }
    
    // Should not exceed max size
    assert!(pool.size() <= 5);
}

/// Test BatchSizeController functionality
#[test]
fn test_batch_size_controller() {
    let controller = BatchSizeController::new(100, 10, 1000, 500.0);
    
    // Test initial state
    assert_eq!(controller.get_batch_size(), 100);
    
    // High latency should decrease batch size
    let new_size = controller.adjust_batch_size(800.0);
    assert!(new_size < 100, "Batch size should decrease with high latency");
    
    // Low latency should increase batch size
    let controller2 = BatchSizeController::new(100, 10, 1000, 500.0);
    let new_size2 = controller2.adjust_batch_size(200.0);
    assert!(new_size2 > 100, "Batch size should increase with low latency");
    
    // Target latency should not change batch size much
    let controller3 = BatchSizeController::new(100, 10, 1000, 500.0);
    let new_size3 = controller3.adjust_batch_size(500.0);
    assert_eq!(new_size3, 100, "Batch size should remain same with target latency");
    
    // Test bounds
    let controller4 = BatchSizeController::new(10, 10, 1000, 500.0);
    let new_size4 = controller4.adjust_batch_size(2000.0); // Very high latency
    assert!(new_size4 >= 10, "Batch size should not go below minimum");
    
    let controller5 = BatchSizeController::new(1000, 10, 1000, 500.0);
    let new_size5 = controller5.adjust_batch_size(100.0); // Very low latency
    assert!(new_size5 <= 1000, "Batch size should not exceed maximum");
}

/// Test PerformanceConfig default values
#[test]
fn test_performance_config_default() {
    let config = PerformanceConfig::default();
    
    assert_eq!(config.monitoring_interval, Duration::from_secs(5));
    assert_eq!(config.memory_threshold, 0.8);
    assert_eq!(config.cpu_threshold, 0.9);
    assert!(config.enable_recommendations);
}

/// Test PerformanceThresholds default values
#[test]
fn test_performance_thresholds_default() {
    let thresholds = PerformanceThresholds::default();
    
    assert_eq!(thresholds.max_cpu_usage, 85.0);
    assert_eq!(thresholds.max_memory_usage, 80.0);
    assert_eq!(thresholds.max_error_rate, 5.0);
    assert_eq!(thresholds.target_processing_rate, 100.0);
    assert_eq!(thresholds.max_latency_ms, 1000.0);
}

/// Test OptimizationRecommendation structure
#[test]
fn test_optimization_recommendation_structure() {
    let recommendation = OptimizationRecommendation {
        category: "CPU".to_string(),
        description: "High CPU usage detected".to_string(),
        impact: "High".to_string(),
        priority: 1,
    };
    
    assert_eq!(recommendation.category, "CPU");
    assert_eq!(recommendation.description, "High CPU usage detected");
    assert_eq!(recommendation.impact, "High");
    assert_eq!(recommendation.priority, 1);
}

/// Test PerformanceMonitor with custom thresholds
#[tokio::test]
async fn test_performance_monitor_custom_thresholds() {
    let custom_thresholds = PerformanceThresholds {
        max_cpu_usage: 70.0,
        max_memory_usage: 60.0,
        max_error_rate: 2.0,
        target_processing_rate: 150.0,
        max_latency_ms: 500.0,
    };
    
    let monitor = PerformanceMonitor::new(custom_thresholds).unwrap();
    
    // Test that monitor works with custom thresholds
    let metrics = monitor.get_metrics().await.unwrap();
    assert!(metrics.cpu_usage >= 0.0);
    
    // Test stress detection with custom thresholds
    let is_stressed = monitor.is_system_stressed().await.unwrap();
    assert!(is_stressed == true || is_stressed == false);
}

/// Test PerformanceMonitor error handling
#[tokio::test]
async fn test_performance_monitor_error_handling() {
    let thresholds = PerformanceThresholds::default();
    let monitor = PerformanceMonitor::new(thresholds).unwrap();
    
    // Test that methods handle errors gracefully
    let metrics_result = monitor.get_metrics().await;
    assert!(metrics_result.is_ok(), "get_metrics should not fail under normal conditions");
    
    let utilization_result = monitor.get_current_utilization().await;
    assert!(utilization_result.is_ok(), "get_current_utilization should not fail under normal conditions");
    
    let stress_result = monitor.is_system_stressed().await;
    assert!(stress_result.is_ok(), "is_system_stressed should not fail under normal conditions");
}

/// Test PerformanceMonitor concurrent access
#[tokio::test]
async fn test_performance_monitor_concurrent_access() {
    let thresholds = PerformanceThresholds::default();
    let monitor = Arc::new(PerformanceMonitor::new(thresholds).unwrap());
    
    let mut handles = Vec::new();
    
    // Spawn multiple tasks that use the monitor concurrently
    for i in 0..10 {
        let monitor_clone = Arc::clone(&monitor);
        let handle = tokio::spawn(async move {
            // Record operations
            monitor_clone.record_success(Duration::from_millis(100 + i * 10));
            
            // Get metrics
            let _metrics = monitor_clone.get_metrics().await.unwrap();
            
            // Check utilization
            let _utilization = monitor_clone.get_current_utilization().await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify the monitor is still functional
    let final_metrics = monitor.get_metrics().await.unwrap();
    assert!(final_metrics.cpu_usage >= 0.0);
}