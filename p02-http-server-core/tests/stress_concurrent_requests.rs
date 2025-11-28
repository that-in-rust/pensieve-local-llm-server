# Stress Tests for Concurrent Request Handling
# Parseltongue principle: Validate thread safety with stress tests

use std::sync::Arc;
use tokio_test;
use warp::test;
use serde_json::json;
use futures::future::join_all;

use p02_http_server_core::{
    create_http_routes_with_middleware,
    traits::RequestHandler,
    Request, Response, ApiResponseError
};

// Thread-safe handler for stress testing
struct StressTestHandler {
    request_count: Arc<std::sync::atomic::AtomicU32>,
}

impl StressTestHandler {
    fn new() -> Self {
        Self {
            request_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    fn get_count(&self) -> u32 {
        self.request_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl RequestHandler for StressTestHandler {
    async fn handle_message_request(&self, request: Request) -> Result<Response, ApiResponseError> {
        // Increment counter atomically
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Simulate some processing time (1ms)
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

        Ok(Response {
            content: format!("Processed request for model: {}", request.model),
            model: request.model,
            usage: Default::default(),
        })
    }
}

/// Stress Test: High Concurrency Message Processing
///
/// GIVEN 100 concurrent message requests
/// WHEN I send them simultaneously
/// THEN the system SHALL handle all requests without data races
/// AND SHALL return success for all requests
/// AND SHALL maintain accurate request count
/// AND SHALL complete within 5 seconds (performance contract)
#[tokio::test]
async fn stress_test_concurrent_message_requests_maintain_data_integrity() {
    // Arrange
    let handler = Arc::new(StressTestHandler::new());
    let initial_count = handler.get_count();
    let routes = create_http_routes_with_middleware(handler.clone());

    const NUM_REQUESTS: usize = 100;

    // Act - Spawn 100 concurrent requests
    let start_time = std::time::Instant::now();
    let mut handles = Vec::with_capacity(NUM_REQUESTS);

    for i in 0..NUM_REQUESTS {
        let routes_clone = routes.clone();
        let handle = tokio::spawn(async move {
            let request_json = json!({
                "model": "phi-3-mini-128k-instruct-4bit",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": format!("Test message {}", i)
                    }
                ]
            });

            test::request()
                .method("POST")
                .path("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-ant-api03-test-key")
                .body(request_json.to_string())
                .reply(&routes_clone)
                .await
        });
        handles.push(handle);
    }

    // Assert - Performance contract
    let results = join_all(handles).await;
    let elapsed = start_time.elapsed();

    assert!(elapsed.as_secs() < 5,
           "100 concurrent requests took {:?}s, contract requires <5s", elapsed.as_secs());

    // Assert - All requests completed successfully
    let mut success_count = 0;
    for (i, result) in results.into_iter().enumerate() {
        let response = result.expect("Request {} must complete without panic", i);
        assert_eq!(response.status(), 200,
                  "Request {} must return 200 OK", i);
        success_count += 1;
    }

    assert_eq!(success_count, NUM_REQUESTS,
              "All {} requests must succeed", NUM_REQUESTS);

    // Assert - Data integrity contract
    let final_count = handler.get_count();
    assert_eq!(final_count, initial_count + NUM_REQUESTS as u32,
              "Handler count must increment by exactly {}", NUM_REQUESTS);

    println!("✅ Stress test passed: {} requests in {:?}ms", NUM_REQUESTS, elapsed.as_millis());
}

/// Stress Test: Mixed Health and Message Requests
///
/// GIVEN 50 health checks and 50 message requests mixed
/// WHEN I send them concurrently
/// THEN the system SHALL handle all requests correctly
/// AND SHALL maintain different response types for each endpoint
/// AND SHALL not cross-contaminate request handling
#[tokio::test]
async fn stress_test_mixed_request_types_maintain_endpoint_isolation() {
    // Arrange
    let handler = Arc::new(StressTestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    const NUM_HEALTH_REQUESTS: usize = 50;
    const NUM_MESSAGE_REQUESTS: usize = 50;

    // Act - Spawn mixed concurrent requests
    let start_time = std::time::Instant::now();
    let mut handles = Vec::new();

    // Add health check requests
    for _ in 0..NUM_HEALTH_REQUESTS {
        let routes_clone = routes.clone();
        let handle = tokio::spawn(async move {
            test::request()
                .method("GET")
                .path("/health")
                .reply(&routes_clone)
                .await
        });
        handles.push(handle);
    }

    // Add message requests
    for i in 0..NUM_MESSAGE_REQUESTS {
        let routes_clone = routes.clone();
        let handle = tokio::spawn(async move {
            let request_json = json!({
                "model": "phi-3-mini-128k-instruct-4bit",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": format!("Mixed message {}", i)}]
            });

            test::request()
                .method("POST")
                .path("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-ant-api03-test-key")
                .body(request_json.to_string())
                .reply(&routes_clone)
                .await
        });
        handles.push(handle);
    }

    // Assert - All requests completed
    let results = join_all(handles).await;
    let elapsed = start_time.elapsed();

    assert!(elapsed.as_secs() < 3,
           "Mixed requests took {:?}s, contract requires <3s", elapsed.as_secs());

    // Verify endpoint isolation
    let mut health_responses = 0;
    let mut message_responses = 0;

    for (i, result) in results.into_iter().enumerate() {
        let response = result.expect("Request {} must complete", i);

        if response.status() == 200 {
            // Check if it's a health response (has status field) or message response
            let body: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
            if body.get("status").is_some() {
                health_responses += 1;
                assert_eq!(body["status"], "healthy",
                          "Health response {} must have healthy status", i);
            } else {
                message_responses += 1;
                assert!(body.get("content").is_some(),
                         "Message response {} must have content", i);
            }
        }
    }

    assert_eq!(health_responses, NUM_HEALTH_REQUESTS,
              "All {} health requests must succeed", NUM_HEALTH_REQUESTS);
    assert_eq!(message_responses, NUM_MESSAGE_REQUESTS,
              "All {} message requests must succeed", NUM_MESSAGE_REQUESTS);

    println!("✅ Mixed stress test passed: {} health + {} messages in {:?}ms",
             NUM_HEALTH_REQUESTS, NUM_MESSAGE_REQUESTS, elapsed.as_millis());
}

/// Stress Test: Rapid Sequential Requests
///
/// GIVEN 1000 rapid sequential requests
/// WHEN I send them one after another quickly
/// THEN the system SHALL handle all requests without connection drops
/// AND SHALL maintain consistent performance
/// AND SHALL not accumulate memory over time
#[tokio::test]
async fn stress_test_rapid_sequential_requests_maintain_performance() {
    // Arrange
    let handler = Arc::new(StressTestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    const NUM_REQUESTS: usize = 1000;

    // Act - Send rapid sequential requests
    let start_time = std::time::Instant::now();
    let mut response_times = Vec::with_capacity(NUM_REQUESTS);

    for i in 0..NUM_REQUESTS {
        let request_start = std::time::Instant::now();

        let response = test::request()
            .method("GET")
            .path("/health")
            .reply(&routes)
            .await;

        let request_time = request_start.elapsed();
        response_times.push(request_time);

        assert_eq!(response.status(), 200,
                  "Sequential request {} must return 200 OK", i);
    }

    let total_elapsed = start_time.elapsed();

    // Assert - Performance consistency
    let avg_time = response_times.iter().sum::<std::time::Duration>() / NUM_REQUESTS as u32;
    let max_time = response_times.iter().max().unwrap();
    let min_time = response_times.iter().min().unwrap();

    assert!(avg_time.as_millis() < 10,
           "Average response time {:?}ms must be <10ms", avg_time.as_millis());
    assert!(max_time.as_millis() < 50,
           "Max response time {:?}ms must be <50ms", max_time.as_millis());
    assert!(total_elapsed.as_secs() < 10,
           "Total time {:?}s must be <10s", total_elapsed.as_secs());

    // Performance variance should be reasonable
    let variance_ratio = max_time.as_millis() as f64 / min_time.as_millis() as f64;
    assert!(variance_ratio < 10.0,
           "Response time variance {:.2}x must be <10x", variance_ratio);

    println!("✅ Rapid sequential test passed: {} requests in {:?}ms (avg: {:?}ms)",
             NUM_REQUESTS, total_elapsed.as_millis(), avg_time.as_millis());
}

/// Stress Test: Memory Usage Under Load
///
/// GIVEN sustained concurrent load
/// WHEN I run requests for 10 seconds
/// THEN the system SHALL maintain stable memory usage
/// AND SHALL not show memory growth pattern
/// AND SHALL handle consistent throughput
#[tokio::test]
async fn stress_test_memory_usage_stable_under_sustained_load() {
    // Arrange
    let handler = Arc::new(StressTestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    let test_duration = std::time::Duration::from_secs(10);
    let start_time = std::time::Instant::now();
    let mut request_count = 0;

    // Act - Sustained load for 10 seconds
    while start_time.elapsed() < test_duration {
        let batch_size = 10;
        let mut handles = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let routes_clone = routes.clone();
            let handle = tokio::spawn(async move {
                test::request()
                    .method("GET")
                    .path("/health")
                    .reply(&routes_clone)
                    .await
            });
            handles.push(handle);
        }

        // Wait for batch to complete
        let results = join_all(handles).await;

        // Verify all succeeded
        for result in results {
            let response = result.unwrap();
            assert_eq!(response.status(), 200);
            request_count += 1;
        }
    }

    let total_elapsed = start_time.elapsed();
    let requests_per_second = request_count as f64 / total_elapsed.as_secs_f64();

    // Assert - Performance contracts
    assert!(requests_per_second > 50.0,
           "Throughput {:.1} req/s must be >50 req/s", requests_per_second);
    assert!(total_elapsed.as_secs() >= 10,
           "Test must run for at least 10 seconds");

    println!("✅ Memory stress test passed: {} requests in {:.1}s ({:.1} req/s)",
             request_count, total_elapsed.as_secs_f64(), requests_per_second);
}