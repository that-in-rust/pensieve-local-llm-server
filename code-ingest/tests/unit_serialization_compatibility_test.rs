//! Unit tests for serialization compatibility
//! 
//! Tests Requirements 4.4, 5.4 - Serialization of system types and metrics

use code_ingest::processing::{PerformanceMetrics, PerformanceConfig, OptimizationRecommendation};
use code_ingest::utils::timestamp::{
    system_time_serde, optional_system_time_serde, duration_millis_serde, duration_secs_serde,
    instant_to_system_time, system_time_to_instant,
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Test structure for SystemTime serialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestSystemTimeStruct {
    #[serde(with = "system_time_serde")]
    timestamp: SystemTime,
    name: String,
}

/// Test structure for optional SystemTime serialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestOptionalSystemTimeStruct {
    #[serde(with = "optional_system_time_serde")]
    timestamp: Option<SystemTime>,
    name: String,
}

/// Test structure for Duration serialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestDurationStruct {
    #[serde(with = "duration_millis_serde")]
    duration_millis: Duration,
    #[serde(with = "duration_secs_serde")]
    duration_secs: Duration,
    name: String,
}

/// Test SystemTime serialization and deserialization
#[test]
fn test_system_time_serialization() {
    let test_data = TestSystemTimeStruct {
        timestamp: SystemTime::now(),
        name: "test".to_string(),
    };
    
    // Test JSON serialization
    let json = serde_json::to_string(&test_data).unwrap();
    assert!(json.contains("\"name\":\"test\""));
    assert!(json.contains("\"timestamp\":"));
    
    // Test deserialization
    let deserialized: TestSystemTimeStruct = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, test_data.name);
    
    // Timestamps should be close (within a reasonable margin due to serialization precision)
    let original_secs = test_data.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs();
    let deserialized_secs = deserialized.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs();
    assert!(
        (original_secs as i64 - deserialized_secs as i64).abs() <= 1,
        "Timestamps should be within 1 second: {} vs {}",
        original_secs,
        deserialized_secs
    );
}

/// Test optional SystemTime serialization
#[test]
fn test_optional_system_time_serialization() {
    // Test with Some value
    let test_data_some = TestOptionalSystemTimeStruct {
        timestamp: Some(SystemTime::now()),
        name: "test_some".to_string(),
    };
    
    let json_some = serde_json::to_string(&test_data_some).unwrap();
    let deserialized_some: TestOptionalSystemTimeStruct = serde_json::from_str(&json_some).unwrap();
    
    assert_eq!(deserialized_some.name, test_data_some.name);
    assert!(deserialized_some.timestamp.is_some());
    
    // Test with None value
    let test_data_none = TestOptionalSystemTimeStruct {
        timestamp: None,
        name: "test_none".to_string(),
    };
    
    let json_none = serde_json::to_string(&test_data_none).unwrap();
    let deserialized_none: TestOptionalSystemTimeStruct = serde_json::from_str(&json_none).unwrap();
    
    assert_eq!(deserialized_none.name, test_data_none.name);
    assert!(deserialized_none.timestamp.is_none());
}

/// Test Duration serialization in different formats
#[test]
fn test_duration_serialization() {
    let test_data = TestDurationStruct {
        duration_millis: Duration::from_millis(1500),
        duration_secs: Duration::from_secs_f64(2.5),
        name: "test_duration".to_string(),
    };
    
    // Test JSON serialization
    let json = serde_json::to_string(&test_data).unwrap();
    assert!(json.contains("\"name\":\"test_duration\""));
    assert!(json.contains("\"duration_millis\":1500"));
    assert!(json.contains("\"duration_secs\":2.5"));
    
    // Test deserialization
    let deserialized: TestDurationStruct = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, test_data.name);
    assert_eq!(deserialized.duration_millis, Duration::from_millis(1500));
    assert_eq!(deserialized.duration_secs, Duration::from_secs_f64(2.5));
}

/// Test PerformanceMetrics serialization
#[test]
fn test_performance_metrics_serialization() {
    let metrics = PerformanceMetrics {
        cpu_usage: 75.5,
        memory_usage: 1024 * 1024 * 512, // 512MB
        available_memory: 1024 * 1024 * 1024, // 1GB
        memory_percentage: 50.0,
        disk_read_bps: 1000000,
        disk_write_bps: 500000,
        network_bps: 100000,
        active_threads: 8,
        processing_rate: 125.7,
        avg_latency_ms: 45.2,
        p95_latency_ms: 89.1,
        error_rate: 2.3,
        timestamp: SystemTime::now(),
    };
    
    // Test JSON serialization
    let json = serde_json::to_string(&metrics).unwrap();
    assert!(json.contains("\"cpu_usage\":75.5"));
    assert!(json.contains("\"memory_usage\":536870912"));
    assert!(json.contains("\"processing_rate\":125.7"));
    
    // Test deserialization
    let deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.cpu_usage, metrics.cpu_usage);
    assert_eq!(deserialized.memory_usage, metrics.memory_usage);
    assert_eq!(deserialized.processing_rate, metrics.processing_rate);
    assert_eq!(deserialized.error_rate, metrics.error_rate);
    
    // Test that timestamp was serialized/deserialized correctly
    let original_secs = metrics.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs();
    let deserialized_secs = deserialized.timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs();
    assert!(
        (original_secs as i64 - deserialized_secs as i64).abs() <= 1,
        "Timestamps should be within 1 second"
    );
}

/// Test PerformanceConfig serialization
#[test]
fn test_performance_config_serialization() {
    let config = PerformanceConfig {
        monitoring_interval: Duration::from_secs(10),
        memory_threshold: 0.85,
        cpu_threshold: 0.95,
        enable_recommendations: true,
    };
    
    // Test JSON serialization
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("\"memory_threshold\":0.85"));
    assert!(json.contains("\"cpu_threshold\":0.95"));
    assert!(json.contains("\"enable_recommendations\":true"));
    
    // Test deserialization
    let deserialized: PerformanceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.memory_threshold, config.memory_threshold);
    assert_eq!(deserialized.cpu_threshold, config.cpu_threshold);
    assert_eq!(deserialized.enable_recommendations, config.enable_recommendations);
    assert_eq!(deserialized.monitoring_interval, config.monitoring_interval);
}

/// Test OptimizationRecommendation serialization
#[test]
fn test_optimization_recommendation_serialization() {
    let recommendation = OptimizationRecommendation {
        category: "Memory".to_string(),
        description: "High memory usage detected. Consider reducing batch sizes.".to_string(),
        impact: "High".to_string(),
        priority: 1,
    };
    
    // Test JSON serialization
    let json = serde_json::to_string(&recommendation).unwrap();
    assert!(json.contains("\"category\":\"Memory\""));
    assert!(json.contains("\"impact\":\"High\""));
    assert!(json.contains("\"priority\":1"));
    
    // Test deserialization
    let deserialized: OptimizationRecommendation = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.category, recommendation.category);
    assert_eq!(deserialized.description, recommendation.description);
    assert_eq!(deserialized.impact, recommendation.impact);
    assert_eq!(deserialized.priority, recommendation.priority);
}

/// Test Instant to SystemTime conversion
#[test]
fn test_instant_to_system_time_conversion() {
    let instant = Instant::now();
    let system_time = instant_to_system_time(instant);
    
    // Should be able to convert to Unix timestamp
    let unix_timestamp = system_time.duration_since(UNIX_EPOCH);
    assert!(unix_timestamp.is_ok(), "Should be able to convert to Unix timestamp");
    
    // Should be reasonably close to current time
    let now = SystemTime::now();
    let diff = if system_time >= now {
        system_time.duration_since(now).unwrap()
    } else {
        now.duration_since(system_time).unwrap()
    };
    
    assert!(diff < Duration::from_secs(1), "Converted time should be close to current time");
}

/// Test SystemTime to Instant conversion
#[test]
fn test_system_time_to_instant_conversion() {
    let system_time = SystemTime::now();
    let instant = system_time_to_instant(system_time);
    
    // Should be able to measure elapsed time
    std::thread::sleep(Duration::from_millis(1));
    let elapsed = instant.elapsed();
    assert!(elapsed >= Duration::from_millis(1), "Should measure elapsed time correctly");
    assert!(elapsed < Duration::from_millis(100), "Elapsed time should be reasonable");
}

/// Test round-trip conversion between Instant and SystemTime
#[test]
fn test_instant_system_time_round_trip() {
    let original_instant = Instant::now();
    let system_time = instant_to_system_time(original_instant);
    let back_to_instant = system_time_to_instant(system_time);
    
    // The round-trip conversion should be approximately correct
    let diff = if back_to_instant >= original_instant {
        back_to_instant.duration_since(original_instant)
    } else {
        original_instant.duration_since(back_to_instant)
    };
    
    // Should be very close (within 10ms due to timing differences and conversion precision)
    assert!(diff < Duration::from_millis(10), 
           "Round-trip conversion should be accurate within 10ms, got: {:?}", diff);
}

/// Test serialization with different timestamp formats
#[test]
fn test_different_timestamp_formats() {
    // Test with Unix epoch
    let epoch_time = UNIX_EPOCH;
    let epoch_struct = TestSystemTimeStruct {
        timestamp: epoch_time,
        name: "epoch".to_string(),
    };
    
    let json = serde_json::to_string(&epoch_struct).unwrap();
    let deserialized: TestSystemTimeStruct = serde_json::from_str(&json).unwrap();
    
    assert_eq!(deserialized.timestamp, epoch_time);
    assert_eq!(deserialized.name, "epoch");
    
    // Test with future time
    let future_time = UNIX_EPOCH + Duration::from_secs(2000000000); // Year 2033
    let future_struct = TestSystemTimeStruct {
        timestamp: future_time,
        name: "future".to_string(),
    };
    
    let json = serde_json::to_string(&future_struct).unwrap();
    let deserialized: TestSystemTimeStruct = serde_json::from_str(&json).unwrap();
    
    assert_eq!(deserialized.timestamp, future_time);
    assert_eq!(deserialized.name, "future");
}

/// Test serialization error handling
#[test]
fn test_serialization_error_handling() {
    // Test invalid JSON for SystemTime
    let invalid_json = r#"{"timestamp": "not_a_number", "name": "test"}"#;
    let result: Result<TestSystemTimeStruct, _> = serde_json::from_str(invalid_json);
    assert!(result.is_err(), "Should fail to deserialize invalid timestamp");
    
    // Test invalid JSON for Duration
    let invalid_duration_json = r#"{"duration_millis": "not_a_number", "duration_secs": 1.0, "name": "test"}"#;
    let result: Result<TestDurationStruct, _> = serde_json::from_str(invalid_duration_json);
    assert!(result.is_err(), "Should fail to deserialize invalid duration");
}

/// Test serialization with extreme values
#[test]
fn test_serialization_extreme_values() {
    // Test with very large duration (but reasonable for serialization)
    let large_duration = Duration::from_secs(1_000_000); // Large but reasonable
    let test_struct = TestDurationStruct {
        duration_millis: large_duration,
        duration_secs: large_duration,
        name: "large".to_string(),
    };
    
    let json = serde_json::to_string(&test_struct).unwrap();
    let deserialized: TestDurationStruct = serde_json::from_str(&json).unwrap();
    
    assert_eq!(deserialized.duration_millis, large_duration);
    assert_eq!(deserialized.duration_secs, large_duration);
    
    // Test with zero duration
    let zero_duration = Duration::ZERO;
    let zero_struct = TestDurationStruct {
        duration_millis: zero_duration,
        duration_secs: zero_duration,
        name: "zero".to_string(),
    };
    
    let json = serde_json::to_string(&zero_struct).unwrap();
    let deserialized: TestDurationStruct = serde_json::from_str(&json).unwrap();
    
    assert_eq!(deserialized.duration_millis, zero_duration);
    assert_eq!(deserialized.duration_secs, zero_duration);
}

/// Test serialization compatibility across different formats
#[test]
fn test_serialization_format_compatibility() {
    let metrics = PerformanceMetrics {
        cpu_usage: 50.0,
        memory_usage: 1024,
        available_memory: 2048,
        memory_percentage: 50.0,
        disk_read_bps: 100,
        disk_write_bps: 200,
        network_bps: 300,
        active_threads: 4,
        processing_rate: 10.5,
        avg_latency_ms: 25.0,
        p95_latency_ms: 50.0,
        error_rate: 1.0,
        timestamp: UNIX_EPOCH + Duration::from_secs(1000000000),
    };
    
    // Test JSON format
    let json = serde_json::to_string(&metrics).unwrap();
    let from_json: PerformanceMetrics = serde_json::from_str(&json).unwrap();
    assert_eq!(from_json.cpu_usage, metrics.cpu_usage);
    
    // Test TOML format (if available)
    #[cfg(feature = "toml")]
    {
        let toml = toml::to_string(&metrics).unwrap();
        let from_toml: PerformanceMetrics = toml::from_str(&toml).unwrap();
        assert_eq!(from_toml.cpu_usage, metrics.cpu_usage);
    }
    
    // Test bincode format (if available)
    #[cfg(feature = "bincode")]
    {
        let binary = bincode::serialize(&metrics).unwrap();
        let from_binary: PerformanceMetrics = bincode::deserialize(&binary).unwrap();
        assert_eq!(from_binary.cpu_usage, metrics.cpu_usage);
    }
}

/// Test serialization performance
#[test]
fn test_serialization_performance() {
    let metrics = PerformanceMetrics {
        cpu_usage: 75.0,
        memory_usage: 1024 * 1024,
        available_memory: 2048 * 1024,
        memory_percentage: 50.0,
        disk_read_bps: 1000,
        disk_write_bps: 2000,
        network_bps: 3000,
        active_threads: 8,
        processing_rate: 100.0,
        avg_latency_ms: 50.0,
        p95_latency_ms: 100.0,
        error_rate: 2.0,
        timestamp: SystemTime::now(),
    };
    
    let start = Instant::now();
    
    // Perform many serialization/deserialization cycles
    for _ in 0..1000 {
        let json = serde_json::to_string(&metrics).unwrap();
        let _deserialized: PerformanceMetrics = serde_json::from_str(&json).unwrap();
    }
    
    let elapsed = start.elapsed();
    
    // Should complete 1000 cycles in reasonable time (less than 1 second)
    assert!(elapsed < Duration::from_secs(1), 
           "1000 serialization cycles should complete in <1s, took: {:?}", elapsed);
    
    // Calculate average time per cycle
    let avg_per_cycle = elapsed / 1000;
    println!("Average serialization time per cycle: {:?}", avg_per_cycle);
    
    // Should be very fast (less than 1ms per cycle)
    assert!(avg_per_cycle < Duration::from_millis(1), 
           "Average serialization time should be <1ms, got: {:?}", avg_per_cycle);
}

/// Test serialization with nested structures
#[test]
fn test_nested_structure_serialization() {
    #[derive(Serialize, Deserialize, Debug)]
    struct NestedStruct {
        metrics: PerformanceMetrics,
        config: PerformanceConfig,
        recommendations: Vec<OptimizationRecommendation>,
    }
    
    let nested = NestedStruct {
        metrics: PerformanceMetrics {
            cpu_usage: 80.0,
            memory_usage: 1024,
            available_memory: 2048,
            memory_percentage: 50.0,
            disk_read_bps: 100,
            disk_write_bps: 200,
            network_bps: 300,
            active_threads: 4,
            processing_rate: 50.0,
            avg_latency_ms: 100.0,
            p95_latency_ms: 200.0,
            error_rate: 5.0,
            timestamp: SystemTime::now(),
        },
        config: PerformanceConfig::default(),
        recommendations: vec![
            OptimizationRecommendation {
                category: "CPU".to_string(),
                description: "High CPU usage".to_string(),
                impact: "High".to_string(),
                priority: 1,
            },
        ],
    };
    
    // Test serialization of nested structure
    let json = serde_json::to_string(&nested).unwrap();
    let deserialized: NestedStruct = serde_json::from_str(&json).unwrap();
    
    assert_eq!(deserialized.metrics.cpu_usage, nested.metrics.cpu_usage);
    assert_eq!(deserialized.config.memory_threshold, nested.config.memory_threshold);
    assert_eq!(deserialized.recommendations.len(), nested.recommendations.len());
    assert_eq!(deserialized.recommendations[0].category, nested.recommendations[0].category);
}