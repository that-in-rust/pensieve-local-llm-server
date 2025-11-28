//! Server Memory Integration Tests
//!
//! RED Phase: These tests should fail initially.
//! Tests the integration of memory monitoring into the HTTP server.

use pensieve_09_anthropic_proxy::memory::{MemoryMonitor, MemoryStatus};
use pensieve_09_anthropic_proxy::server::{AnthropicProxyServer, ServerConfig};
use std::sync::Arc;

/// Mock memory monitor for testing server behavior
struct MockMemoryMonitor {
    available_gb: f64,
}

impl MockMemoryMonitor {
    fn new(available_gb: f64) -> Self {
        Self { available_gb }
    }
}

impl MemoryMonitor for MockMemoryMonitor {
    fn check_status(&self) -> MemoryStatus {
        match self.available_gb {
            x if x > 3.0 => MemoryStatus::Safe,
            x if x > 2.0 => MemoryStatus::Caution,
            x if x > 1.0 => MemoryStatus::Warning,
            x if x > 0.5 => MemoryStatus::Critical,
            _ => MemoryStatus::Emergency,
        }
    }

    fn available_gb(&self) -> f64 {
        self.available_gb
    }
}

#[test]
fn test_server_accepts_request_with_safe_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(4.0)) as Arc<dyn MemoryMonitor>;
    let config = ServerConfig::default();

    // Server should accept creation with safe memory
    let server = AnthropicProxyServer::with_memory_monitor(config, monitor);

    // Server should be created successfully
    assert!(!server.is_running());
}

#[test]
fn test_server_accepts_request_with_caution_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(2.5)) as Arc<dyn MemoryMonitor>;
    let config = ServerConfig::default();

    let server = AnthropicProxyServer::with_memory_monitor(config, monitor);
    assert!(!server.is_running());
}

#[test]
fn test_server_accepts_request_with_warning_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(1.5)) as Arc<dyn MemoryMonitor>;
    let config = ServerConfig::default();

    let server = AnthropicProxyServer::with_memory_monitor(config, monitor);
    assert!(!server.is_running());
}

#[test]
fn test_memory_monitor_can_be_queried() {
    let monitor = Arc::new(MockMemoryMonitor::new(2.5)) as Arc<dyn MemoryMonitor>;

    assert_eq!(monitor.check_status(), MemoryStatus::Caution);
    assert_eq!(monitor.available_gb(), 2.5);
}

#[test]
fn test_server_config_default_includes_memory() {
    let config = ServerConfig::default();
    let monitor = Arc::new(MockMemoryMonitor::new(4.0)) as Arc<dyn MemoryMonitor>;

    let server = AnthropicProxyServer::with_memory_monitor(config, monitor);
    assert!(!server.is_running());
}

// Integration tests that verify memory status affects routing
// These will fail initially until we implement the integration

#[tokio::test]
async fn test_health_endpoint_includes_memory_status() {
    let monitor = Arc::new(MockMemoryMonitor::new(2.5)) as Arc<dyn MemoryMonitor>;
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 7778, // Different port for testing
        python_bridge_path: "python_bridge/mlx_inference.py".to_string(),
        model_path: "models/Phi-3-mini-128k-instruct-4bit".to_string(),
    };

    let server = AnthropicProxyServer::with_memory_monitor(config.clone(), monitor.clone());

    // Start server
    server.start().await.expect("Failed to start server");

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Query health endpoint
    let client = reqwest::Client::new();
    let response = client
        .get(&format!("http://{}:{}/health", config.host, config.port))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.expect("Failed to parse JSON");

    // Should include memory information
    assert!(body.get("memory").is_some(), "Health response should include memory field");

    let memory = body.get("memory").unwrap();
    assert!(memory.get("status").is_some());
    assert!(memory.get("available_gb").is_some());

    // Cleanup
    server.shutdown().await.expect("Failed to shutdown server");
}

#[tokio::test]
async fn test_messages_endpoint_rejects_critical_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(0.7)) as Arc<dyn MemoryMonitor>; // Critical
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 7779, // Different port
        python_bridge_path: "python_bridge/mlx_inference.py".to_string(),
        model_path: "models/Phi-3-mini-128k-instruct-4bit".to_string(),
    };

    let server = AnthropicProxyServer::with_memory_monitor(config.clone(), monitor.clone());
    server.start().await.expect("Failed to start server");
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Send request to messages endpoint
    let client = reqwest::Client::new();
    let response = client
        .post(&format!("http://{}:{}/v1/messages", config.host, config.port))
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer test-token")
        .json(&serde_json::json!({
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "test"}]
        }))
        .send()
        .await
        .expect("Failed to send request");

    // Should return 503 Service Unavailable
    assert_eq!(response.status(), 503, "Critical memory should return 503");

    // Should include memory headers
    assert!(response.headers().contains_key("x-memory-status"));

    server.shutdown().await.expect("Failed to shutdown");
}

#[tokio::test]
async fn test_messages_endpoint_accepts_safe_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(4.0)) as Arc<dyn MemoryMonitor>; // Safe
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 7780,
        python_bridge_path: "python_bridge/mlx_inference.py".to_string(),
        model_path: "models/Phi-3-mini-128k-instruct-4bit".to_string(),
    };

    let server = AnthropicProxyServer::with_memory_monitor(config.clone(), monitor.clone());
    server.start().await.expect("Failed to start server");
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(&format!("http://{}:{}/v1/messages", config.host, config.port))
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer test-token")
        .json(&serde_json::json!({
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "test"}]
        }))
        .send()
        .await
        .expect("Failed to send request");

    // With safe memory, should not return 503
    // (May return 500 if Python bridge not running, but that's OK - we're testing memory check)
    assert_ne!(response.status(), 503, "Safe memory should not return 503");

    server.shutdown().await.expect("Failed to shutdown");
}
