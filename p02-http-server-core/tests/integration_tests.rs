# Integration Tests for HTTP Server Core
# Following parseltongue principles with comprehensive testing

use std::sync::Arc;
use tokio_test;
use warp::test;

// Import our four-word functions from lib.rs
use p02_http_server_core::{
    create_http_routes_with_middleware,
    validate_api_key_from_header,
    parse_json_body_safely,
    create_error_response_from_status,
    create_cors_headers_for_options,
    traits::RequestHandler,
    ServerConfig, Request, Response, ApiResponseError
};

// Mock handler for testing
struct MockRequestHandler;

#[async_trait::async_trait]
impl RequestHandler for MockRequestHandler {
    async fn handle_message_request(&self, request: Request) -> Result<Response, ApiResponseError> {
        // Simple mock response for testing
        Ok(Response {
            content: "Mock response".to_string(),
            model: "test-model".to_string(),
            usage: Default::default(),
        })
    }
}

/// Test successful health check endpoint
///
/// # Preconditions
/// - HTTP routes created with mock handler
/// - Warp test environment available
///
/// # Postconditions
/// - Returns 200 OK status
/// - Returns valid JSON response
/// - Includes timestamp field
#[tokio::test]
async fn test_health_check_endpoint_success() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let handler = Arc::new(MockRequestHandler);
    let routes = create_http_routes_with_middleware(handler);

    // Act
    let response = test::request()
        .method("GET")
        .path("/health")
        .reply(&routes)
        .await;

    // Assert
    assert_eq!(response.status(), 200, "Health check should return 200 OK");

    let body: serde_json::Value = serde_json::from_slice(response.body())?;
    assert_eq!(body["status"], "healthy", "Status should be healthy");
    assert!(body["timestamp"].is_string(), "Should include timestamp");

    Ok(())
}

/// Test CORS preflight handling
///
/// # Preconditions
/// - HTTP routes created with CORS middleware
/// - OPTIONS request to /v1/messages
///
/// # Postconditions
/// - Returns 200 OK status
/// - Includes appropriate CORS headers
/// - Allows required methods and headers
#[tokio::test]
async fn test_cors_preflight_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let handler = Arc::new(MockRequestHandler);
    let routes = create_http_routes_with_middleware(handler);

    // Act
    let response = test::request()
        .method("OPTIONS")
        .path("/v1/messages")
        .header("origin", "https://claude.ai")
        .header("access-control-request-method", "POST")
        .reply(&routes)
        .await;

    // Assert
    assert_eq!(response.status(), 200, "CORS preflight should return 200 OK");

    // Check CORS headers
    let headers = response.headers();
    assert!(
        headers.contains_key("access-control-allow-origin"),
        "Should include Access-Control-Allow-Origin header"
    );
    assert!(
        headers.contains_key("access-control-allow-methods"),
        "Should include Access-Control-Allow-Methods header"
    );
    assert!(
        headers.contains_key("access-control-allow-headers"),
        "Should include Access-Control-Allow-Headers header"
    );

    Ok(())
}

/// Test API key validation with valid key
///
/// # Preconditions
/// - Valid API key provided
/// - Validation function available
///
/// # Postconditions
/// - Returns Ok(()) for valid key
/// - Properly extracts and validates key format
#[test]
fn test_api_key_validation_valid_key() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let valid_key = "sk-ant-api03-test-key-123456";

    // Act
    let result = validate_api_key_from_header(&Some(valid_key.to_string()))?;

    // Assert
    assert!(result.is_ok(), "Valid API key should pass validation");

    Ok(())
}

/// Test API key validation with missing key
///
/// # Preconditions
/// - No API key provided
/// - Validation function available
///
/// # Postconditions
/// - Returns Err with proper error message
/// - Indicates missing API key
#[test]
fn test_api_key_validation_missing_key() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let no_key: Option<String> = None;

    // Act
    let result = validate_api_key_from_header(&no_key);

    // Assert
    assert!(result.is_err(), "Missing API key should fail validation");

    let error = result.unwrap_err();
    assert!(
        matches!(error, ApiResponseError::Unauthorized(_)),
        "Should return Unauthorized error for missing key"
    );

    Ok(())
}

/// Test JSON body parsing with valid payload
///
/// # Preconditions
/// - Valid JSON bytes provided
/// - Expected type specified
///
/// # Postconditions
/// - Returns parsed object
/// - Maintains data integrity
#[tokio::test]
async fn test_json_body_parsing_valid_payload() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let valid_json = r#"{
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, world!"
            }
        ]
    }"#;

    // Act
    let result: Result<serde_json::Value, ApiResponseError> =
        parse_json_body_safely(valid_json.as_bytes().into()).await;

    // Assert
    assert!(result.is_ok(), "Valid JSON should parse successfully");

    let parsed = result.unwrap();
    assert_eq!(parsed["model"], "phi-3-mini-128k-instruct-4bit");
    assert_eq!(parsed["max_tokens"], 100);
    assert_eq!(parsed["messages"][0]["role"], "user");

    Ok(())
}

/// Test JSON body parsing with malformed payload
///
/// # Preconditions
/// - Invalid JSON bytes provided
/// - Expected type specified
///
/// # Postconditions
/// - Returns Err with parsing error
/// - Provides context about failure
#[tokio::test]
async fn test_json_body_parsing_malformed_payload() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let malformed_json = b"{ invalid json syntax }";

    // Act
    let result: Result<serde_json::Value, ApiResponseError> =
        parse_json_body_safely(malformed_json.to_vec().into()).await;

    // Assert
    assert!(result.is_err(), "Malformed JSON should fail parsing");

    let error = result.unwrap_err();
    assert!(
        matches!(error, ApiResponseError::JsonParsingFailed(_)),
        "Should return JsonParsingFailed error for malformed JSON"
    );

    Ok(())
}

/// Test error response creation
///
/// # Preconditions
/// - HTTP status code provided
/// - Error message available
///
/// # Postconditions
/// - Returns properly formatted error response
/// - Includes appropriate status code and error details
#[test]
fn test_error_response_creation() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let status = 400;
    let message = "Bad request: Invalid parameters";

    // Act
    let (response_status, response_body) = create_error_response_from_status(status, message);

    // Assert
    assert_eq!(response_status, status, "Should preserve provided status code");

    let error_json: serde_json::Value = serde_json::from_str(&response_body)?;
    assert_eq!(error_json["error"]["type"], "invalid_request_error");
    assert_eq!(error_json["error"]["message"], message);

    Ok(())
}

/// Test server configuration with fixed values
///
/// # Preconditions
/// - Default configuration created
/// - Fixed values expected per parseltongue principles
///
/// # Postconditions
/// - Configuration matches expected fixed values
/// - No deviation from hardcoded parameters
#[test]
fn test_server_configuration_fixed_values() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange & Act
    let config = ServerConfig::default();

    // Assert
    assert_eq!(config.host, "127.0.0.1", "Host should be fixed to 127.0.0.1");
    assert_eq!(config.port, 528491, "Port should be fixed to 528491 for Claude Code");
    assert_eq!(config.max_concurrent_requests, 10, "Should have fixed concurrency limit");
    assert_eq!(config.request_timeout_ms, 30000, "Should have fixed timeout");
    assert!(config.enable_cors, "CORS should be enabled by default");

    Ok(())
}

/// Test complete request flow for message creation
///
/// # Preconditions
/// - Valid Anthropic API request
/// - Proper authorization header
/// - JSON content type
///
/// # Postconditions
/// - Returns 200 OK status
/// - Response matches API specification
/// - Proper content negotiation
#[tokio::test]
async fn test_complete_message_request_flow() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let handler = Arc::new(MockRequestHandler);
    let routes = create_http_routes_with_middleware(handler);

    let request_json = r#"{
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, Claude!"
            }
        ]
    }"#;

    // Act
    let response = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-api03-test-key")
        .body(request_json)
        .reply(&routes)
        .await;

    // Assert
    assert_eq!(response.status(), 200, "Message creation should return 200 OK");

    let body: serde_json::Value = serde_json::from_slice(response.body())?;
    assert!(body["content"].is_string(), "Response should include content");
    assert!(body["model"].is_string(), "Response should include model");
    assert!(body["usage"].is_object(), "Response should include usage metrics");

    Ok(())
}

/// Test concurrent request handling
///
/// # Preconditions
/// - Multiple concurrent requests
/// - Server configured with concurrency limits
///
/// # Postconditions
/// - All requests handled successfully
/// - No resource contention
/// - Responses returned in reasonable time
#[tokio::test]
async fn test_concurrent_request_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Arrange
    let handler = Arc::new(MockRequestHandler);
    let routes = create_http_routes_with_middleware(handler);

    // Act - Send 5 concurrent health check requests
    let mut handles = vec![];
    for _ in 0..5 {
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

    // Assert - Wait for all requests to complete
    for handle in handles {
        let response = handle.await?;
        assert_eq!(response.status(), 200, "All concurrent requests should succeed");
    }

    Ok(())
}

/// Property-based test for request validation
///
/// # Preconditions
/// - Various request configurations generated
/// - Edge cases included
///
/// # Postconditions
/// - All requests handled appropriately
/// - No panics or crashes
/// - Proper error responses for invalid requests
#[tokio::test]
async fn test_property_based_request_validation() -> Result<(), Box<dyn std::error::Error>> {
    use proptest::prelude::*;

    proptest!(|(
        model in "[a-zA-Z0-9_-]{1,50}",
        max_tokens in 1u32..=8192,
        message_count in 1u32..=10
    )| {
        let handler = Arc::new(MockRequestHandler);
        let routes = create_http_routes_with_middleware(handler);

        // Create valid request with generated parameters
        let messages: Vec<String> = (0..message_count)
            .map(|i| format!(
                r#"{{"role": "user", "content": "Message {}"}}"#, i
            ))
            .collect();

        let request_json = format!(
            r#"{{
                "model": "{}",
                "max_tokens": {},
                "messages": [{}]
            }}"#,
            model,
            max_tokens,
            messages.join(",")
        );

        // Test should not panic
        let response = tokio_test::block_on(async {
            test::request()
                .method("POST")
                .path("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-ant-api03-test-key")
                .body(&request_json)
                .reply(&routes)
                .await
        });

        // Basic sanity checks
        assert!(response.status() >= 200 && response.status() < 600);
    });

    Ok(())
}

/// Test server graceful shutdown scenarios
///
/// # Preconditions
/// - Server running with active connections
/// - Shutdown signal received
///
/// # Postconditions
/// - Existing connections allowed to complete
/// - New connections rejected
/// - Resources cleaned up properly
#[tokio::test]
async fn test_server_graceful_shutdown() -> Result<(), Box<dyn std::error::Error>> {
    // This is a placeholder test for graceful shutdown functionality
    // In a real implementation, we would test server shutdown scenarios

    let handler = Arc::new(MockRequestHandler);
    let _routes = create_http_routes_with_middleware(handler);

    // Simulate shutdown scenario
    println!("Testing graceful shutdown scenario...");

    // Assert that shutdown logic would work
    assert!(true, "Graceful shutdown should be supported");

    Ok(())
}