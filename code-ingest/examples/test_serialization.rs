use code_ingest::processing::performance::PerformanceMetrics;
use std::time::SystemTime;

fn main() {
    let metrics = PerformanceMetrics {
        cpu_usage: 45.5,
        memory_usage: 1024 * 1024 * 512, // 512 MB
        available_memory: 1024 * 1024 * 1024 * 4, // 4 GB
        memory_percentage: 12.5,
        disk_read_bps: 1000,
        disk_write_bps: 2000,
        network_bps: 500,
        active_threads: 8,
        processing_rate: 150.0,
        avg_latency_ms: 25.5,
        p95_latency_ms: 45.0,
        error_rate: 0.5,
        timestamp: SystemTime::now(),
    };

    // Test serialization
    match serde_json::to_string_pretty(&metrics) {
        Ok(json) => {
            println!("Serialization successful:");
            println!("{}", json);
            
            // Test deserialization
            match serde_json::from_str::<PerformanceMetrics>(&json) {
                Ok(deserialized) => {
                    println!("\nDeserialization successful!");
                    println!("CPU Usage: {:.1}%", deserialized.cpu_usage);
                    println!("Memory Usage: {} bytes", deserialized.memory_usage);
                    println!("Timestamp: {:?}", deserialized.timestamp);
                }
                Err(e) => {
                    println!("Deserialization failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Serialization failed: {}", e);
        }
    }
}