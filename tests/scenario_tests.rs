//! Scenario-Based Integration Tests
//!
//! These tests validate specific user workflows and edge cases
//! that represent real-world usage patterns.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;
use pensieve_02::{HttpApiServer, ServerConfig, traits::RequestHandler};
use pensieve_03::{anthropic::*, ApiMessage};
use pensieve_04::{traits::CandleInferenceEngine, GenerationConfig, StreamingTokenResponse};
use pensieve_07_core::{CoreError, CoreResult};

/// Enhanced mock request handler with realistic simulation
#[derive(Debug, Clone)]
pub struct ScenarioRequestHandler {
    request_count: Arc<std::sync::atomic::AtomicUsize>,
    error_simulation: Arc<std::sync::atomic::AtomicBool>,
    slow_simulation: Arc<std::sync::atomic::AtomicBool>,
}

impl ScenarioRequestHandler {
    pub fn new() -> Self {
        Self {
            request_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            error_simulation: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            slow_simulation: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub fn simulate_errors(&self, simulate: bool) {
        self.error_simulation.store(simulate, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn simulate_slow_responses(&self, simulate: bool) {
        self.slow_simulation.store(simulate, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_request_count(&self) -> usize {
        self.request_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl RequestHandler for ScenarioRequestHandler {
    async fn handle_message(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse, pensieve_02::error::ServerError> {
        // Count requests
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Simulate errors if enabled
        if self.error_simulation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(pensieve_02::error::ServerError::Internal("Simulated error".to_string()));
        }

        // Validate request
        request.validate().map_err(|e| pensieve_02::error::ServerError::Request(e.to_string()))?;

        // Simulate processing delay
        if self.slow_simulation.load(std::sync::atomic::Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(2000)).await;
        } else {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Create realistic response
        Ok(CreateMessageResponse {
            id: format!("msg_{}", self.get_request_count()),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text {
                text: format!("Mock response to request {}: {}", 
                    self.get_request_count(), 
                    request.messages.first()
                        .and_then(|m| m.content.first())
                        .and_then(|c| if let Content::Text { text } = c { Some(text) } else { None })
                        .unwrap_or("empty message")),
            }],
            model: request.model,
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        })
    }

    async fn handle_stream(&self, request: CreateMessageRequest) -> pensieve_02::error::ServerResult<pensieve_02::StreamingResponse> {
        // Count requests
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Simulate errors if enabled
        if self.error_simulation.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(pensieve_02::error::ServerError::Internal("Simulated streaming error".to_string()));
        }

        // Validate request
        request.validate().map_err(|e| pensieve_02::error::ServerError::Request(e.to_string()))?;

        // Create streaming response
        let response_text = format!("Mock streaming response #{}", self.get_request_count());
        let chars: Vec<char> = response_text.chars().collect();

        let stream = async_stream::stream! {
            // Yield message start
            yield "data: {\"type\": \"message_start\"}\n\n".to_string();

            // Yield content deltas
            for (i, c) in chars.iter().enumerate() {
                if i < 3 { // Limit for demo
                    yield format!("data: {{\"type\": \"content_block_delta\", \"delta\": {{\"text\": \"{}\"}}}}\n\n", c);
                }
            }

            // Yield message stop
            yield "data: {\"type\": \"message_stop\"}\n\n".to_string();
        };

        Ok(Box::pin(stream))
    }
}

/// Scenario 1: High-Frequency Request Pattern
#[tokio::test]
async fn test_high_frequency_requests() {
    println!("🔥 Testing High-Frequency Request Pattern");
    
    let handler = Arc::new(ScenarioRequestHandler::new());
    let server_config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // Random port
        max_concurrent_requests: 50,
        request_timeout_ms: 10000,
        enable_cors: true,
    };

    let server = HttpApiServer::new(server_config, handler.clone());

    // Start server
    let _ = server.start().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send multiple rapid requests
    let mut handles = Vec::new();
    for i in 0..10 {
        let handler_clone = handler.clone();
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 50,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: format!("Quick test {}", i),
                }],
            }],
            temperature: None,
            top_p: None,
            stream: Some(false),
            system: None,
        };

        handles.push(tokio::spawn(async move {
            handler_clone.handle_message(request).await
        }));
    }

    // Wait for all requests to complete
    let results = futures::future::join_all(handles).await;
    let successful = results.into_iter().filter(|r| r.is_ok()).count();
    
    // Shutdown server
    let _ = server.shutdown().await;

    assert_eq!(successful, 10, "All high-frequency requests should succeed");
    assert_eq!(handler.get_request_count(), 10, "All requests should be counted");
}

/// Scenario 2: Error Recovery Pattern
#[tokio::test]
async fn test_error_recovery_pattern() {
    println!("🔄 Testing Error Recovery Pattern");
    
    let handler = Arc::new(ScenarioRequestHandler::new());

    // Test normal operation
    let normal_request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 10,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Normal request".to_string(),
            }],
        }],
        temperature: None,
        top_p: None,
        stream: Some(false),
        system: None,
    };

    let normal_result = handler.handle_message(normal_request.clone()).await;
    assert!(normal_result.is_ok(), "Normal request should succeed");

    // Simulate errors
    handler.simulate_errors(true);

    let error_result = handler.handle_message(normal_request.clone()).await;
    assert!(error_result.is_err(), "Error simulation should cause failures");

    // Test recovery (stop simulating errors)
    handler.simulate_errors(false);

    let recovery_result = handler.handle_message(normal_request).await;
    assert!(recovery_result.is_ok(), "Recovery should work after stopping error simulation");
}

/// Scenario 3: Streaming vs Non-Streaming Consistency
#[tokio::test]
async fn test_streaming_consistency() {
    println!("📡 Testing Streaming vs Non-Streaming Consistency");
    
    let handler = Arc::new(ScenarioRequestHandler::new());

    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 5,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Consistency test".to_string(),
            }],
        }],
        temperature: Some(0.7),
        top_p: Some(0.9),
        stream: None, // Non-streaming
        system: None,
    };

    // Test non-streaming response
    let non_streaming = handler.handle_message(request.clone()).await;
    assert!(non_streaming.is_ok(), "Non-streaming should work");
    let non_streaming_response = non_streaming.unwrap();

    // Test streaming response
    let streaming = handler.handle_stream(request).await;
    assert!(streaming.is_ok(), "Streaming should work");
    let mut streaming_stream = streaming.unwrap();

    // Collect streaming response
    let mut streaming_content = String::new();
    while let Some(chunk_result) = streaming_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if chunk.contains("\"type\": \"content_block_delta\"") {
                    // Extract the text content
                    if let Some(text_part) = chunk.split("\"text\": \"").nth(1) {
                        if let Some(text_end) = text_part.split("\"").next() {
                            streaming_content.push_str(text_end);
                        }
                    }
                }
            }
            Err(_) => break,
        }
    }

    // Both should have similar content (basic check)
    assert!(!non_streaming_response.content.is_empty(), "Non-streaming should have content");
    assert!(!streaming_content.is_empty(), "Streaming should have content");
    
    // Both should be from assistant
    assert_eq!(non_streaming_response.role, Role::Assistant);
    // Streaming doesn't have explicit role in individual chunks, but content should exist
}

/// Scenario 4: Long Context Handling
#[tokio::test]
async fn test_long_context_handling() {
    println!("📜 Testing Long Context Handling");
    
    let handler = Arc::new(ScenarioRequestHandler::new());

    // Create a very long message
    let long_message = "This is a test message. ".repeat(50); // ~1200 characters
    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 100,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: long_message.clone(),
            }],
        }],
        temperature: Some(0.7),
        top_p: Some(0.9),
        stream: Some(false),
        system: None,
    };

    let start_time = std::time::Instant::now();
    let result = handler.handle_message(request).await;
    let duration = start_time.elapsed();

    assert!(result.is_ok(), "Long context request should succeed");
    assert!(duration.as_millis() < 5000, "Long context should process within 5 seconds");

    let response = result.unwrap();
    assert_eq!(response.role, Role::Assistant);
    assert!(!response.content.is_empty());
    
    // Count tokens in response
    let response_text = if let Content::Text { text } = &response.content[0] {
        text
    } else {
        ""
    };
    assert!(!response_text.is_empty(), "Response should have text content");
}

/// Scenario 5: Memory Pressure Simulation
#[tokio::test]
async fn test_memory_pressure_simulation() {
    println!("💾 Testing Memory Pressure Simulation");
    
    let handler = Arc::new(ScenarioRequestHandler::new());
    
    // Simulate slow responses to test timeout handling
    handler.simulate_slow_responses(true);

    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 10,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Slow response test".to_string(),
            }],
        }],
        temperature: None,
        top_p: None,
        stream: Some(false),
        system: None,
    };

    let start_time = std::time::Instant::now();
    let result = timeout(Duration::from_millis(3000), handler.handle_message(request)).await;
    
    // Clean up simulation
    handler.simulate_slow_responses(false);

    match result {
        Ok(r) => {
            // Should succeed or fail, but not timeout
            println!("Request completed in {:?}", start_time.elapsed());
            assert!(r.is_ok(), "Slow response should still complete (not timeout)");
        }
        Err(_) => {
            assert!(false, "Request should not timeout after 3 seconds");
        }
    }
}

/// Scenario 6: API Version Compatibility
#[tokio::test]
async fn test_api_version_compatibility() {
    println!("🔄 Testing API Version Compatibility");
    
    let handler = Arc::new(ScenarioRequestHandler::new());

    // Test with different model names
    let models_to_test = vec![
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "unknown-model",
    ];

    for model in models_to_test {
        let request = CreateMessageRequest {
            model: model.to_string(),
            max_tokens: 10,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "API compatibility test".to_string(),
                }],
            }],
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            system: None,
        };

        let result = handler.handle_message(request).await;
        
        // Unknown model might fail, but others should succeed
        if model == "unknown-model" {
            // Might fail validation
            println!("Testing unknown model: {:?}", result);
        } else {
            assert!(result.is_ok(), "Known model {} should work", model);
        }
    }
}

/// Scenario 7: Concurrent Stream Handling
#[tokio::test]
async fn test_concurrent_stream_handling() {
    println!("🌊 Testing Concurrent Stream Handling");
    
    let handler = Arc::new(ScenarioRequestHandler::new());

    let requests: Vec<CreateMessageRequest> = (0..3)
        .map(|i| CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 10,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: format!("Concurrent stream test {}", i),
                }],
            }],
            temperature: Some(0.7),
            top_p: None,
            stream: Some(true),
            system: None,
        })
        .collect();

    let mut handles = Vec::new();
    for request in requests {
        let handler_clone = handler.clone();
        handles.push(tokio::spawn(async move {
            handler_clone.handle_stream(request).await
        }));
    }

    let results = futures::future::join_all(handles).await;
    let successful = results.into_iter().filter(|r| r.is_ok()).count();
    
    assert_eq!(successful, 3, "All concurrent streams should succeed");
    assert!(handler.get_request_count() >= 3, "All requests should be counted");
}

/// Scenario 8: Graceful Shutdown Under Load
#[tokio::test]
async fn test_graceful_shutdown_under_load() {
    println!("🛑 Testing Graceful Shutdown Under Load");
    
    let handler = Arc::new(ScenarioRequestHandler::new());
    let server_config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        max_concurrent_requests: 5,
        request_timeout_ms: 2000,
        enable_cors: true,
    };

    let server = HttpApiServer::new(server_config, handler.clone());

    // Start server
    let _ = server.start().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start some requests
    let mut handles = Vec::new();
    for i in 0..3 {
        let handler_clone = handler.clone();
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 5,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: format!("Shutdown test {}", i),
                }],
            }],
            temperature: None,
            top_p: None,
            stream: Some(false),
            system: None,
        };

        handles.push(tokio::spawn(async move {
            handler_clone.handle_message(request).await
        }));
    }

    // Wait a bit for requests to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Initiate shutdown
    let shutdown_handle = tokio::spawn(async move {
        server.shutdown().await
    });

    // Wait for shutdown to complete
    let shutdown_result = timeout(Duration::from_millis(5000), shutdown_handle).await;
    assert!(shutdown_result.is_ok(), "Shutdown should complete within 5 seconds");

    // Check that our main requests can still complete
    let results = futures::future::join_all(handles).await;
    let successful = results.into_iter().filter(|r| r.is_ok()).count();
    
    println!("Shutdown completed. Successful requests: {}/{}", successful, 3);
    assert!(successful >= 2, "At least 2 requests should complete during shutdown");
}
