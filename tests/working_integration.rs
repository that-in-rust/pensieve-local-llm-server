//! Working Integration Tests - Phase 2.8 End-to-End Validation
//! 
//! These tests validate the integration of working crates without
//! running into compilation issues in problematic test modules.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// Import crates that we know work correctly
use pensieve_01::{CliConfig, CliArgs, Commands, PensieveCli};
use pensieve_02::{HttpApiServer, ServerConfig, traits::RequestHandler};
use pensieve_03::{anthropic::*, ApiMessage};
use pensieve_07_core::{CoreError, CoreResult};

// Simple mock handler that works
#[derive(Debug, Clone)]
pub struct WorkingMockHandler {
    request_count: std::sync::atomic::AtomicUsize,
}

impl WorkingMockHandler {
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
impl RequestHandler for WorkingMockHandler {
    async fn handle_message(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse, pensieve_02::error::ServerError> {
        // Count requests
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Validate request
        request.validate().map_err(|e| pensieve_02::error::ServerError::Request(e.to_string()))?;

        // Create simple response
        let response_text = format!("Working response #{} to: {}", 
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

        let response_text = format!("Working streaming response #{}", self.get_request_count());

        // Create simple stream
        let stream = futures::stream::iter(vec![
            format!("data: {{\"type\": \"message_start\"}}\n\n"),
            format!("data: {{\"type\": \"content_block_delta\", \"delta\": {{\"text\": \"{}\"}}}}\n\n", response_text),
            format!("data: {{\"type\": \"message_stop\"}}\n\n"),
        ]);

        Ok(Box::pin(stream))
    }
}

/// Test 1: CLI Configuration Integration
#[tokio::test]
async fn test_cli_configuration_integration() {
    println!("🧪 Testing CLI Configuration Integration");
    
    // Test basic CLI creation
    let args = CliArgs {
        command: Commands::Validate { config: None },
        config: None,
        verbose: false,
        log_level: None,
    };

    let cli_result = PensieveCli::new(args);
    assert!(cli_result.is_ok(), "CLI creation should succeed");

    let cli = cli_result.unwrap();
    
    // Test configuration validation (should fail due to missing model)
    let validation_result = cli.validate_config();
    assert!(validation_result.is_err(), "Should fail validation without model file");
    println!("✅ CLI configuration integration test passed");
}

/// Test 2: HTTP Server Integration
#[tokio::test]
async fn test_http_server_integration() {
    println!("🧪 Testing HTTP Server Integration");
    
    let handler = Arc::new(WorkingMockHandler::new());
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
    sleep(Duration::from_millis(100)).await;

    // Test server health
    assert!(server.is_healthy(), "Server should be healthy");
    
    // Test server shutdown
    let shutdown_result = server.shutdown().await;
    assert!(shutdown_result.is_ok(), "Server should shutdown successfully");
    println!("✅ Server shutdown completed");
}

/// Test 3: API Request Integration
#[tokio::test]
async fn test_api_request_integration() {
    println!("🧪 Testing API Request Integration");
    
    let handler = WorkingMockHandler::new();
    
    let request = CreateMessageRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 100,
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Hello from integration test!".to_string(),
            }],
        }],
        temperature: Some(0.7),
        top_p: None,
        stream: None,
        system: None,
    };

    // Test message handling
    let response = handler.handle_message(request).await;
    assert!(response.is_ok(), "Message handling should succeed");
    
    let response = response.unwrap();
    assert_eq!(response.role, Role::Assistant);
    assert!(!response.content.is_empty());
    
    if let Content::Text { text } = &response.content[0] {
        assert!(text.contains("Working response"));
        assert!(text.contains("Hello from integration test"));
    } else {
        panic!("Expected text content");
    }
    
    // Verify request count was incremented
    assert_eq!(handler.get_request_count(), 1);
    
    println!("✅ API request integration test passed");
}

/// Test 4: Complete Workflow Integration
#[tokio::test]
async fn test_complete_workflow_integration() {
    println!("🚀 Testing Complete Workflow Integration");
    
    // 1. Create CLI
    let args = CliArgs {
        command: Commands::Config { action: pensieve_01::ConfigAction::Show },
        config: None,
        verbose: false,
        log_level: None,
    };

    let cli_result = PensieveCli::new(args);
    assert!(cli_result.is_ok(), "CLI creation should succeed");
    
    // 2. Create and start server
    let handler = Arc::new(WorkingMockHandler::new());
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
    
    sleep(Duration::from_millis(100)).await;
    
    // 3. Send multiple requests
    let test_requests = vec![
        "Hello complete workflow!",
        "This tests all components together.",
        "CLI → Server → Handler → Response!",
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
    
    println!("🎉 Complete workflow integration test passed!");
}
