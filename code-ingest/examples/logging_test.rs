use code_ingest::logging::{LoggingConfig, init_logging, ProgressReporter, PerformanceMetrics, MonitoringContext};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let logging_config = LoggingConfig {
        level: "debug".to_string(),
        json_format: false,
        progress_reporting: true,
        performance_metrics: true,
        ..Default::default()
    };
    
    init_logging(&logging_config)?;
    
    tracing::info!("Starting logging test");
    
    // Test progress reporting
    let progress = ProgressReporter::new("test_operation", Some(100));
    
    for i in 0..=100 {
        progress.set_progress(i).await;
        if i % 20 == 0 {
            sleep(Duration::from_millis(100)).await;
        }
    }
    
    progress.complete().await;
    
    // Test performance metrics
    let metrics = PerformanceMetrics::new("test_metrics");
    
    metrics.checkpoint("start").await;
    
    // Simulate some work
    let result = metrics.time_async("async_operation", || async {
        sleep(Duration::from_millis(200)).await;
        "operation_result"
    }).await;
    
    metrics.increment_counter("operations_completed").await;
    metrics.checkpoint("end").await;
    metrics.report().await;
    
    // Test monitoring context
    let monitoring = MonitoringContext::new("comprehensive_test", Some(50));
    
    for i in 0..50 {
        monitoring.progress.increment().await;
        monitoring.performance.increment_counter("items_processed").await;
        monitoring.memory.check_memory().await;
        
        if i % 10 == 0 {
            sleep(Duration::from_millis(50)).await;
        }
    }
    
    monitoring.complete_and_report().await;
    
    tracing::info!("Logging test completed successfully");
    
    Ok(())
}