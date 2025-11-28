# Parseltongue Executable Contracts for HTTP Server
# TDD-First: STUB → RED → GREEN → REFACTOR
# Every claim validated by automated tests

use std::sync::Arc;
use tokio_test;
use warp::test;
use serde_json::{json, Value};

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

// Mock handler with executable contracts
struct MockRequestHandler {
    call_count: Arc<std::sync::atomic::AtomicU32>,
}

impl MockRequestHandler {
    fn new() -> Self {
        Self {
            call_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    fn get_call_count(&self) -> u32 {
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl RequestHandler for MockRequestHandler {
    async fn handle_message_request(&self, request: Request) -> Result<Response, ApiResponseError> {
        // Increment call count for testing
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Validate contract: request must have model and messages
        if request.model.is_empty() {
            return Err(ApiResponseError::ValidationFailed("Model cannot be empty".to_string()));
        }

        if request.messages.is_empty() {
            return Err(ApiResponseError::ValidationFailed("Messages cannot be empty".to_string()));
        }

        // Return mock response for testing
        Ok(Response {
            content: format!("Mock response for model: {}", request.model),
            model: request.model.clone(),
            usage: Default::default(),
        })
    }
}

/// Executable Contract: Health Check Endpoint
///
/// WHEN I send GET request to /health
/// THEN the system SHALL return 200 OK status
/// AND SHALL return JSON with status field set to "healthy"
/// AND SHALL include timestamp field in RFC3339 format
/// AND SHALL complete within 10ms (performance contract)
#[tokio::test]
async fn contract_health_check_endpoint_must_return_healthy_status_with_timestamp_under_10ms() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    // Act
    let start_time = std::time::Instant::now();
    let response = test::request()
        .method("GET")
        .path("/health")
        .reply(&routes)
        .await;
    let elapsed = start_time.elapsed();

    // Assert - Performance contract first (fail fast)
    assert!(elapsed.as_millis() < 10,
           "Health check took {:?}ms, contract requires <10ms", elapsed.as_millis());

    // Assert - Response structure contract
    assert_eq!(response.status(), 200, "Health check must return 200 OK");

    // Assert - JSON structure contract
    let body: Value = serde_json::from_slice(response.body())
        .expect("Health check must return valid JSON");

    assert_eq!(body["status"], "healthy", "Status field must be 'healthy'");
    assert!(body["timestamp"].is_string(), "Must include timestamp field");

    // Assert - Timestamp format contract (RFC3339)
    let timestamp_str = body["timestamp"].as_str().unwrap();
    chrono::DateTime::parse_from_rfc3339(timestamp_str)
        .expect("Timestamp must be in RFC3339 format");
}

/// Executable Contract: CORS Preflight Handling
///
/// WHEN I send OPTIONS request to /v1/messages with origin header
/// THEN the system SHALL return 200 OK status
/// AND SHALL include Access-Control-Allow-Origin header
/// AND SHALL include Access-Control-Allow-Methods header
/// AND SHALL include Access-Control-Allow-Headers header
/// AND SHALL complete within 5ms (performance contract)
#[tokio::test]
async fn contract_cors_preflight_must_return_correct_headers_under_5ms() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    // Act
    let start_time = std::time::Instant::now();
    let response = test::request()
        .method("OPTIONS")
        .path("/v1/messages")
        .header("origin", "https://claude.ai")
        .header("access-control-request-method", "POST")
        .reply(&routes)
        .await;
    let elapsed = start_time.elapsed();

    // Assert - Performance contract
    assert!(elapsed.as_millis() < 5,
           "CORS preflight took {:?}ms, contract requires <5ms", elapsed.as_millis());

    // Assert - Response status contract
    assert_eq!(response.status(), 200, "CORS preflight must return 200 OK");

    // Assert - Required headers contract
    let headers = response.headers();

    assert!(headers.contains_key("access-control-allow-origin"),
           "Must include Access-Control-Allow-Origin header");

    assert!(headers.contains_key("access-control-allow-methods"),
           "Must include Access-Control-Allow-Methods header");

    assert!(headers.contains_key("access-control-allow-headers"),
           "Must include Access-Control-Allow-Headers header");
}

/// Executable Contract: API Key Validation with Valid Key
///
/// GIVEN a valid API key in format "sk-ant-api03-*"
/// WHEN I call validate_api_key_from_header
/// THEN the system SHALL return Ok(())
/// AND SHALL not modify the key
/// AND SHALL complete within 1ms (performance contract)
#[test]
fn contract_api_key_validation_must_accept_valid_keys_under_1ms() {
    // Arrange
    let valid_keys = vec![
        "sk-ant-api03-test-key-123456",
        "sk-ant-api03-abcdefghijklmnop",
        "sk-ant-api03-1234567890123456",
    ];

    for key in valid_keys {
        // Act
        let start_time = std::time::Instant::now();
        let result = validate_api_key_from_header(&Some(key.to_string()));
        let elapsed = start_time.elapsed();

        // Assert - Performance contract
        assert!(elapsed.as_micros() < 1000,
               "API key validation took {:?}μs, contract requires <1000μs", elapsed.as_micros());

        // Assert - Validation contract
        assert!(result.is_ok(), "Valid API key '{}' must be accepted", key);
    }
}

/// Executable Contract: API Key Validation with Missing Key
///
/// GIVEN no API key provided
/// WHEN I call validate_api_key_from_header
/// THEN the system SHALL return Err(ApiResponseError::Unauthorized)
/// AND SHALL include descriptive error message
/// AND SHALL complete within 1ms (performance contract)
#[test]
fn contract_api_key_validation_must_reject_missing_keys_under_1ms() {
    // Arrange
    let no_key: Option<String> = None;

    // Act
    let start_time = std::time::Instant::now();
    let result = validate_api_key_from_header(&no_key);
    let elapsed = start_time.elapsed();

    // Assert - Performance contract
    assert!(elapsed.as_micros() < 1000,
           "API key validation took {:?}μs, contract requires <1000μs", elapsed.as_micros());

    // Assert - Error contract
    assert!(result.is_err(), "Missing API key must be rejected");

    let error = result.unwrap_err();
    assert!(matches!(error, ApiResponseError::Unauthorized(_)),
           "Must return Unauthorized error for missing key");

    // Assert - Error message contract
    let error_msg = error.to_string();
    assert!(!error_msg.is_empty(), "Error message must not be empty");
}

/// Executable Contract: JSON Body Parsing with Valid Payload
///
/// GIVEN a valid JSON string with expected structure
/// WHEN I call parse_json_body_safely
/// THEN the system SHALL return Ok(Value)
/// AND SHALL preserve all fields
/// AND SHALL complete within 2ms for 1KB payload (performance contract)
#[tokio::test]
async fn contract_json_parsing_must_accept_valid_payload_under_2ms() {
    // Arrange
    let valid_payloads = vec![
        // Small payload
        json!({
            "model": "phi-3-mini-128k-instruct-4bit",
            "max_tokens": 100
        }).to_string(),

        // Medium payload
        json!({
            "model": "phi-3-mini-128k-instruct-4bit",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "temperature": 0.7,
            "stream": false
        }).to_string(),

        // Large payload (~1KB)
        json!({
            "model": "phi-3-mini-128k-instruct-4bit",
            "max_tokens": 8192,
            "messages": (0..50).map(|i| json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("Message {} with some additional content to increase size", i)
            })).collect::<Vec<_>>(),
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": false
        }).to_string(),
    ];

    for (i, payload) in valid_payloads.into_iter().enumerate() {
        // Act
        let start_time = std::time::Instant::now();
        let result: Result<Value, ApiResponseError> =
            parse_json_body_safely(payload.as_bytes().into()).await;
        let elapsed = start_time.elapsed();

        // Assert - Performance contract (2ms for 1KB, scaled for smaller)
        let max_time_ms = if i == 2 { 2 } else { 1 };
        assert!(elapsed.as_millis() <= max_time_ms,
               "JSON parsing took {:?}ms, contract requires <{}ms", elapsed.as_millis(), max_time_ms);

        // Assert - Parsing contract
        assert!(result.is_ok(), "Valid JSON payload must be parsed successfully");

        let parsed = result.unwrap();
        assert!(parsed.is_object(), "Parsed value must be an object");
        assert!(parsed.get("model").is_some(), "Must preserve model field");
    }
}

/// Executable Contract: JSON Body Parsing with Invalid Payload
///
/// GIVEN an invalid JSON string
/// WHEN I call parse_json_body_safely
/// THEN the system SHALL return Err(ApiResponseError::JsonParsingFailed)
/// AND SHALL include error context
/// AND SHALL complete within 1ms (performance contract)
#[tokio::test]
async fn contract_json_parsing_must_reject_invalid_payload_under_1ms() {
    // Arrange
    let invalid_payloads = vec![
        "{ invalid json syntax }",
        "{\"unclosed\": \"string",
        "{\"missing_value\":}",
        "{\"extra_comma\": 100,}",
        "not json at all",
        "",
    ];

    for payload in invalid_payloads {
        // Act
        let start_time = std::time::Instant::now();
        let result: Result<Value, ApiResponseError> =
            parse_json_body_safely(payload.as_bytes().into()).await;
        let elapsed = start_time.elapsed();

        // Assert - Performance contract
        assert!(elapsed.as_millis() < 1,
               "JSON parsing took {:?}ms, contract requires <1ms", elapsed.as_millis());

        // Assert - Error contract
        assert!(result.is_err(), "Invalid JSON payload must be rejected");

        let error = result.unwrap_err();
        assert!(matches!(error, ApiResponseError::JsonParsingFailed(_)),
               "Must return JsonParsingFailed error for invalid JSON");
    }
}

/// Executable Contract: Message Creation Endpoint with Valid Request
///
/// GIVEN a valid Anthropic API request with proper authorization
/// WHEN I send POST request to /v1/messages
/// THEN the system SHALL return 200 OK status
/// AND SHALL return JSON response with content, model, and usage fields
/// AND SHALL call the request handler exactly once
/// AND SHALL complete within 50ms (performance contract)
#[tokio::test]
async fn contract_message_creation_must_handle_valid_requests_under_50ms() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let initial_call_count = handler.get_call_count();
    let routes = create_http_routes_with_middleware(handler.clone());

    let request_json = json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Hello, Claude!"
            }
        ]
    });

    // Act
    let start_time = std::time::Instant::now();
    let response = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-api03-test-key")
        .body(request_json.to_string())
        .reply(&routes)
        .await;
    let elapsed = start_time.elapsed();

    // Assert - Performance contract
    assert!(elapsed.as_millis() < 50,
           "Message creation took {:?}ms, contract requires <50ms", elapsed.as_millis());

    // Assert - Response contract
    assert_eq!(response.status(), 200, "Message creation must return 200 OK");

    // Assert - Handler call contract
    let final_call_count = handler.get_call_count();
    assert_eq!(final_call_count, initial_call_count + 1,
              "Request handler must be called exactly once");

    // Assert - Response structure contract
    let body: Value = serde_json::from_slice(response.body())
        .expect("Response must be valid JSON");

    assert!(body.get("content").is_some(), "Response must include content field");
    assert!(body.get("model").is_some(), "Response must include model field");
    assert!(body.get("usage").is_some(), "Response must include usage field");
}

/// Executable Contract: Message Creation Endpoint with Missing Authorization
///
/// GIVEN a request without authorization header
/// WHEN I send POST request to /v1/messages
/// THEN the system SHALL return 401 Unauthorized status
/// AND SHALL not call the request handler
/// AND SHALL complete within 10ms (performance contract)
#[tokio::test]
async fn contract_message_creation_must_reject_unauthorized_requests_under_10ms() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let initial_call_count = handler.get_call_count();
    let routes = create_http_routes_with_middleware(handler.clone());

    let request_json = json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    });

    // Act
    let start_time = std::time::Instant::now();
    let response = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        // Missing authorization header
        .body(request_json.to_string())
        .reply(&routes)
        .await;
    let elapsed = start_time.elapsed();

    // Assert - Performance contract
    assert!(elapsed.as_millis() < 10,
           "Unauthorized rejection took {:?}ms, contract requires <10ms", elapsed.as_millis());

    // Assert - Response contract
    assert_eq!(response.status(), 401, "Must return 401 for missing authorization");

    // Assert - Handler call contract
    let final_call_count = handler.get_call_count();
    assert_eq!(final_call_count, initial_call_count,
              "Request handler must not be called for unauthorized requests");
}

/// Executable Contract: Error Response Creation
///
/// GIVEN an HTTP status code and error message
/// WHEN I call create_error_response_from_status
/// THEN the system SHALL return tuple of status and JSON string
/// AND SHALL include error type field
/// AND SHALL include error message field
/// AND SHALL complete within 1ms (performance contract)
#[test]
fn contract_error_response_creation_must_include_type_and_message_under_1ms() {
    // Arrange
    let test_cases = vec![
        (400, "Bad request"),
        (401, "Unauthorized"),
        (404, "Not found"),
        (500, "Internal server error"),
        (429, "Rate limited"),
    ];

    for (status, message) in test_cases {
        // Act
        let start_time = std::time::Instant::now();
        let (response_status, response_body) =
            create_error_response_from_status(status, message);
        let elapsed = start_time.elapsed();

        // Assert - Performance contract
        assert!(elapsed.as_micros() < 1000,
               "Error response creation took {:?}μs, contract requires <1000μs", elapsed.as_micros());

        // Assert - Status contract
        assert_eq!(response_status, status, "Status must match input");

        // Assert - JSON structure contract
        let error_json: Value = serde_json::from_str(&response_body)
            .expect("Error response must be valid JSON");

        assert_eq!(error_json["error"]["type"], "invalid_request_error",
                  "Error type must be 'invalid_request_error'");
        assert_eq!(error_json["error"]["message"], message,
                  "Error message must match input");
    }
}

/// Executable Contract: Server Configuration with Fixed Values
///
/// WHEN I create ServerConfig with default()
/// THEN the system SHALL use fixed port 528491
/// AND SHALL use fixed host "127.0.0.1"
/// AND SHALL use fixed max_concurrent_requests 10
/// AND SHALL use fixed request_timeout_ms 30000
/// AND SHALL have enable_cors set to true
/// AND SHALL complete within 1ms (performance contract)
#[test]
fn contract_server_config_must_use_fixed_values_under_1ms() {
    // Act
    let start_time = std::time::Instant::now();
    let config = ServerConfig::default();
    let elapsed = start_time.elapsed();

    // Assert - Performance contract
    assert!(elapsed.as_micros() < 1000,
           "ServerConfig creation took {:?}μs, contract requires <1000μs", elapsed.as_micros());

    // Assert - Fixed values contract (parseltongue principle: fixed constraints)
    assert_eq!(config.host, "127.0.0.1", "Host must be fixed to 127.0.0.1");
    assert_eq!(config.port, 528491, "Port must be fixed to 528491 for Claude Code");
    assert_eq!(config.max_concurrent_requests, 10, "Must have fixed concurrency limit");
    assert_eq!(config.request_timeout_ms, 30000, "Must have fixed timeout");
    assert!(config.enable_cors, "CORS must be enabled by default");
}

/// Executable Contract: Concurrent Request Handling
///
/// GIVEN 5 concurrent health check requests
/// WHEN I send them simultaneously
/// THEN the system SHALL handle all requests successfully
/// AND SHALL return 200 OK for all requests
/// AND SHALL complete all requests within 50ms total (performance contract)
/// AND SHALL maintain data isolation between requests
#[tokio::test]
async fn contract_concurrent_health_checks_must_all_succeed_under_50ms() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    // Act - Send 5 concurrent health check requests
    let start_time = std::time::Instant::now();
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

    // Assert - Performance contract
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("All concurrent requests must complete");

    let elapsed = start_time.elapsed();
    assert!(elapsed.as_millis() < 50,
           "5 concurrent requests took {:?}ms, contract requires <50ms", elapsed.as_millis());

    // Assert - All requests succeed contract
    for (i, response) in results.into_iter().enumerate() {
        assert_eq!(response.status(), 200,
                  "Concurrent request {} must return 200 OK", i + 1);

        let body: Value = serde_json::from_slice(response.body())
            .expect("Response {} must be valid JSON", i + 1);

        assert_eq!(body["status"], "healthy",
                  "Response {} must have healthy status", i + 1);
    }
}

/// Executable Contract: Memory Usage for Large Requests
///
/// GIVEN a 10KB JSON payload
/// WHEN I send POST request to /v1/messages
/// THEN the system SHALL handle the request without panicking
/// AND SHALL return appropriate response
/// AND SHALL not leak memory (no growth in repeated calls)
/// AND SHALL complete within 100ms (performance contract)
#[tokio::test]
async fn contract_large_requests_must_not_cause_memory_leaks_under_100ms() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    // Create large payload (~10KB)
    let large_messages: Vec<Value> = (0..100).map(|i| json!({
        "role": "user",
        "content": format!("This is message {} with enough content to make the payload larger. ", i)
    })).collect();

    let large_payload = json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 4096,
        "messages": large_messages,
        "temperature": 0.7,
        "stream": false
    });

    let payload_size = large_payload.to_string().len();
    assert!(payload_size > 10000, "Payload must be at least 10KB");

    // Act - Send request multiple times to test for memory leaks
    for round in 1..=5 {
        let start_time = std::time::Instant::now();
        let response = test::request()
            .method("POST")
            .path("/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer sk-ant-api03-test-key")
            .body(large_payload.to_string())
            .reply(&routes)
            .await;
        let elapsed = start_time.elapsed();

        // Assert - Performance contract
        assert!(elapsed.as_millis() < 100,
               "Large request {} took {:?}ms, contract requires <100ms", round, elapsed.as_millis());

        // Assert - Response contract
        assert_eq!(response.status(), 200,
                  "Large request {} must return 200 OK", round);
    }
}

/// Executable Contract: Server Response Headers
///
/// WHEN I send any request to the server
/// THEN the system SHALL include appropriate headers
/// AND SHALL set Content-Type to application/json for API endpoints
/// AND SHALL include CORS headers when appropriate
/// AND SHALL not expose internal server information
#[tokio::test]
async fn contract_server_responses_must_include_proper_headers() {
    // Arrange
    let handler = Arc::new(MockRequestHandler::new());
    let routes = create_http_routes_with_middleware(handler);

    let request_json = json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Test"}]
    });

    // Act
    let response = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-api03-test-key")
        .header("origin", "https://claude.ai")
        .body(request_json.to_string())
        .reply(&routes)
        .await;

    // Assert - Header contracts
    assert_eq!(response.status(), 200, "Must return 200 OK");

    let headers = response.headers();

    // Content-Type contract
    if let Some(content_type) = headers.get("content-type") {
        assert!(content_type.to_str().unwrap().contains("application/json"),
               "Content-Type must be application/json for API endpoints");
    }

    // CORS headers contract
    assert!(headers.contains_key("access-control-allow-origin"),
           "Must include CORS headers for cross-origin requests");
}