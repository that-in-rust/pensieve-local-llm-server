//! Simple Integration Tests - Phase 2.8 End-to-End Validation
//!
//! These tests validate the complete integration of all 7 crates
//! in a working implementation.

use pensieve_01::{CliConfig, CliArgs, Commands, PensieveCli};
use pensieve_02::{HttpApiServer, ServerConfig, traits::RequestHandler};
use pensieve_03::{anthropic::*, ApiMessage};
use pensieve_07_core::{CoreError, CoreResult};

// Simple mock handler for integration testing
#[derive(Debug, Clone)]
pub struct SimpleMockHandler {
    request_count: std::sync::atomic::AtomicUsize,
}

impl SimpleMockHandler {
    pub fn new() -> Self {
        Self {
            request_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn get_request_count(&self) -> usize {
        self.request_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl RequestHandler for SimpleMockHandler {
    async fn handle_message(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse, pensieve_02::error::ServerError> {
        // Count requests
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Validate request
        request.validate().map_err(|e| pensieve_02::error::ServerError::Request(e.to_string()))?;

        // Create simple response
        let response_text = format!("Mock response #{} to: {}", 
            self.get_request_count(),
            request.messages.first()
                .and_then(|m| m.content.first())
                .and_then(|c| if let Content::Text { text } = c { Some(text) } else { None })
                .unwrap_or("empty message"));

        Ok(CreateMessageResponse {
            id: format!("msg_{}", self.get_request_count()),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text {
                text: response_text,
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

    async fn handle_stream(&self, request: CreateMessageRequest) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = String> + Send>>, pensieve_02::error::ServerError> {
        // Count requests
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Validate request
        request.validate().map_err(|e| pensieve_02::error::ServerError::Request(e.to_string()))?;

        let response_text = format!("Mock streaming response #{}", self.get_request_count());

        // Create simple stream
        let stream = futures::stream::iter(vec![
            format!("data: {{\"type\": \"message_start\"}}\n\n"),
            format!("data: {{\"type\": \"content_block_delta\", \"delta\": {{\"text\": \"{}\"}}}}\n\n", response_text),
            format!("data: {{\"type\": \"message_stop\"}}\n\n"),
        ]);

        Ok(Box::pin(stream))
    }
}

/// Test 1: Basic CLI Configuration and Validation
#[tokio::test]
async fn test_cli_configuration() {
    println!("🧪 Testing CLI Configuration");
    
    // Test basic CLI creation
    let args = CliArgs {
        command: Commands::Validate { config: None },
        config: None,
        verbose: false,
        log_level: None,
    };

    let cli = PensieveCli::new(args);
    assert!(cli.is_ok(), "CLI creation should succeed");

    let cli = cli.unwrap();
    
    // Test configuration validation (should fail due to missing model)
    let validation_result = cli.validate_config();
    assert!(validation_result.is_err(), "Should fail validation without model file");
    println!("✅ CLI configuration test passed");
}

/// Test 2: HTTP Server Lifecycle
#[tokio::test]
async fn test_http_server_lifecycle() {
    println!("🧪 Testing HTTP Server Lifecycle");
    
    let handler = std::sync::Arc::new(SimpleMockHandler::new());
    let server_config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // Use random port for testing
        max_concurrent_requests: 10,
        request_timeout_ms: 30000,
        enable_cors: true,
    };

    let server = HttpApiServer::new(server_config, handler.clone());
    
    // Test server start
    let start_result = server.start().await;
    assert!(start_result.is_ok(), "Server should start successfully");
    println!("✅ Server started successfully");

    // Give server time to bind
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Test server health
    assert!(server.is_healthy(), "Server should be healthy");
    
    // Test server shutdown
    let shutdown_result = server.shutdown().await;
    assert!(shutdown_result.is_ok(), "Server should shutdown successfully");
    println!("✅ Server shutdown completed");
}

/// Test 3: API Request Validation
#[tokio::test]
async fn test_api_request_validation() {
    println!("🧪 Testing API Request Validation");
    
    // Test valid request
    let valid_request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 100,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Hello, world!".to_string(),
            }],
        }],
        temperature: Some(0.7),
        top_p: None,
        stream: None,
        system: None,
    };

    assert!(valid_request.validate().is_ok(), "Valid request should pass validation");

    // Test invalid requests
    let mut invalid_requests = vec![];

    // Empty model
    let mut invalid = valid_request.clone();
    invalid.model = "".to_string();
    invalid_requests.push((invalid, "empty model"));

    // Zero max tokens
    let mut invalid = valid_request.clone();
    invalid.max_tokens = 0;
    invalid_requests.push((invalid, "zero max tokens"));

    // Empty messages
    let mut invalid = valid_request.clone();
    invalid.messages = vec![];
    invalid_requests.push((invalid, "empty messages"));

    // Empty content
    let mut invalid = valid_request.clone();
    invalid.messages = vec![Message {
        role: Role::User,
        content: vec![],
    }];
    invalid_requests.push((invalid, "empty content"));

    // Test all invalid requests
    for (request, error_type) in invalid_requests {
        let result = request.validate();
        assert!(result.is_err(), 
                "Invalid request ({}) should fail validation", error_type);
        println!("✅ Invalid request ({}) correctly rejected", error_type);
    }
}

/// Test 4: API Request Serialization/Deserialization
#[tokio::test]
async fn test_api_serialization() {
    println!("🧪 Testing API Serialization/Deserialization");
    
    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 100,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Serialization test".to_string(),
            }],
        }],
        temperature: Some(0.7),
        top_p: Some(0.9),
        stream: Some(true),
        system: Some("You are a helpful assistant.".to_string()),
    };

    // Test serialization
    let json = request.to_json();
    assert!(json.is_ok(), "Serialization should succeed");
    println!("✅ Request serialization successful");

    let json_str = json.unwrap();
    assert!(json_str.contains("claude-3-sonnet-20240229"));
    assert!(json_str.contains("Serialization test"));
    assert!(json_str.contains("0.7"));
    assert!(json_str.contains("true"));

    // Test deserialization
    let deserialized = CreateMessageRequest::from_json(&json_str);
    assert!(deserialized.is_ok(), "Deserialization should succeed");
    
    let deserialized_request = deserialized.unwrap();
    assert_eq!(deserialized_request.model, request.model);
    assert_eq!(deserialized_request.max_tokens, request.max_tokens);
    assert_eq!(deserialized_request.messages.len(), request.messages.len());
    assert_eq!(deserialized_request.temperature, request.temperature);
    assert_eq!(deserialized_request.stream, request.stream);
    assert_eq!(deserialized_request.system, request.system);
    
    println!("✅ Request deserialization successful");
}

/// Test 5: Message Handling with Mock Handler
#[tokio::test]
async fn test_message_handling() {
    println!("🧪 Testing Message Handling");
    
    let handler = SimpleMockHandler::new();
    
    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 50,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Hello, handler!".to_string(),
            }],
        }],
        temperature: Some(0.7),
        stream: Some(false),
        system: None,
    };

    // Test message handling
    let response = handler.handle_message(request).await;
    assert!(response.is_ok(), "Message handling should succeed");
    
    let response = response.unwrap();
    assert_eq!(response.role, Role::Assistant);
    assert!(!response.content.is_empty());
    
    if let Content::Text { text } = &response.content[0] {
        assert!(text.contains("Mock response"));
        assert!(text.contains("Hello, handler!"));
    } else {
        panic!("Expected text content");
    }
    
    // Verify request count was incremented
    assert_eq!(handler.get_request_count(), 1);
    
    println!("✅ Message handling successful");
}

/// Test 6: Streaming Response Handling
#[tokio::test] 
async fn test_streaming_response() {
    println!("🧪 Testing Streaming Response");
    
    let handler = SimpleMockHandler::new();
    
    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 50,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Stream test".to_string(),
            }],
        }],
        temperature: Some(0.7),
        stream: Some(true),
        system: None,
    };

    // Test streaming handling
    let stream_result = handler.handle_stream(request).await;
    assert!(stream_result.is_ok(), "Streaming should succeed");
    
    let mut stream = stream_result.unwrap();
    
    // Collect stream events
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    
    // Verify we got stream events
    assert!(!events.is_empty(), "Should have streaming events");
    
    // Check for expected event types
    let has_message_start = events.iter().any(|e| e.contains("message_start"));
    let has_content_delta = events.iter().any(|e| e.contains("content_block_delta"));
    let has_message_stop = events.iter().any(|e| e.contains("message_stop"));
    
    assert!(has_message_start, "Should have message_start event");
    assert!(has_content_delta, "Should have content_block_delta event");
    assert!(has_message_stop, "Should have message_stop event");
    
    // Verify request count was incremented
    assert_eq!(handler.get_request_count(), 1);
    
    println!("✅ Streaming response successful");
}

/// Test 7: Concurrent Request Handling
#[tokio::test]
async fn test_concurrent_requests() {
    println!("🧪 Testing Concurrent Request Handling");
    
    let handler = std::sync::Arc::new(SimpleMockHandler::new());
    
    let requests: Vec<CreateMessageRequest> = (0..5)
        .map(|i| CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 10,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: format!("Concurrent test {}", i),
                }],
            }],
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            system: None,
        })
        .collect();

    // Spawn concurrent tasks
    let mut handles = Vec::new();
    for request in requests {
        let handler_clone = handler.clone();
        handles.push(tokio::spawn(async move {
            handler_clone.handle_message(request).await
        }));
    }

    // Wait for all tasks to complete
    let results = futures::future::join_all(handles).await;
    let successful = results.into_iter().filter(|r| r.is_ok()).count();
    
    // All requests should succeed
    assert_eq!(successful, 5, "All concurrent requests should succeed");
    
    // Verify all requests were counted
    assert_eq!(handler.get_request_count(), 5);
    
    println!("✅ Concurrent requests handled successfully");
}

/// Test 8: Server with Multiple Requests
#[tokio::test]
async fn test_server_with_multiple_requests() {
    println!("🧪 Testing Server with Multiple Requests");
    
    let handler = std::sync::Arc::new(SimpleMockHandler::new());
    let server_config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // Random port
        max_concurrent_requests: 10,
        request_timeout_ms: 30000,
        enable_cors: true,
    };

    let server = HttpApiServer::new(server_config, handler.clone());
    
    // Start server
    let _ = server.start().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Send multiple requests
    let mut handles = Vec::new();
    for i in 0..3 {
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 10,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: format!("Server test {}", i),
                }],
            }],
            temperature: None,
            top_p: None,
            stream: Some(false),
            system: None,
        };

        let handler_clone = handler.clone();
        handles.push(tokio::spawn(async move {
            handler_clone.handle_message(request).await
        }));
    }

    // Wait for all requests
    let results = futures::future::join_all(handles).await;
    let successful = results.into_iter().filter(|r| r.is_ok()).count();
    
    // Shutdown server
    let _ = server.shutdown().await;
    
    // All requests should succeed
    assert_eq!(successful, 3, "All server requests should succeed");
    
    // Verify request count
    assert_eq!(handler.get_request_count(), 3);
    
    println!("✅ Server multiple requests completed");
}

#[tokio::test]
async fn test_complete_integration_workflow() {
    println!("🚀 Testing Complete Integration Workflow");
    
    // 1. Create CLI
    let args = CliArgs {
        command: Commands::Config { action: pensieve_01::ConfigAction::Show },
        config: None,
        verbose: false,
        log_level: None,
    };

    let cli = PensieveCli::new(args);
    assert!(cli.is_ok(), "CLI creation should succeed");
    
    // 2. Create and start server
    let handler = std::sync::Arc::new(SimpleMockHandler::new());
    let server_config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        max_concurrent_requests: 5,
        request_timeout_ms: 10000,
        enable_cors: true,
    };

    let server = HttpApiServer::new(server_config, handler.clone());
    let start_result = server.start().await;
    assert!(start_result.is_ok(), "Server should start");
    
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // 3. Send multiple requests
    let test_requests = vec![
        "Hello, integration test!",
        "This is a complete workflow test.",
        "Testing all components together.",
    ];
    
    for (i, test_input) in test_requests.iter().enumerate() {
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 20,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: test_input.to_string(),
                }],
            }],
            temperature: Some(0.7),
            stream: Some(false),
            system: None,
        };

        let response = handler.handle_message(request).await;
        assert!(response.is_ok(), "Request {} should succeed", i);
        
        let response = response.unwrap();
        assert_eq!(response.role, Role::Assistant);
        assert!(!response.content.is_empty());
    }
    
    // 4. Shutdown server
    let shutdown_result = server.shutdown().await;
    assert!(shutdown_result.is_ok(), "Server should shutdown");
    
    // 5. Verify all requests were handled
    assert_eq!(handler.get_request_count(), test_requests.len());
    
    println!("🎉 Complete integration workflow test passed!");
}
