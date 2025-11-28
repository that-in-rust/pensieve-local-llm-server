# Property-Based Tests for HTTP Server
# Parseltongue principle: Invariants must hold across large input spaces

use std::sync::Arc;
use warp::test;
use proptest::prelude::*;
use serde_json::{json, Value};

use p02_http_server_core::{
    create_http_routes_with_middleware,
    validate_api_key_from_header,
    parse_json_body_safely,
    traits::RequestHandler,
    Request, Response, ApiResponseError
};

// Simple handler for property testing
struct PropertyTestHandler;

#[async_trait::async_trait]
impl RequestHandler for PropertyTestHandler {
    async fn handle_message_request(&self, request: Request) -> Result<Response, ApiResponseError> {
        // For property testing, always succeed
        Ok(Response {
            content: format!("Property test response for: {}", request.model),
            model: request.model,
            usage: Default::default(),
        })
    }
}

/// Property-based test: API Key Validation
///
/// PROPERTY: For all valid API keys following the pattern "sk-ant-api03-*",
/// validation should succeed. For all invalid keys, validation should fail.
#[proptest!]
fn property_api_key_validation_is_deterministic(
    #[filter(#|key: &str| key.starts_with("sk-ant-api03-"))] valid_key: String,
    invalid_key_prefixes: Vec<String>
) {
    // Test valid key property
    let result = validate_api_key_from_header(&Some(valid_key));
    prop_assert!(result.is_ok(), "Valid API key should always pass validation");

    // Test invalid key property
    for prefix in invalid_key_prefixes {
        let invalid_key = format!("{}invalid-key", prefix);
        let result = validate_api_key_from_header(&Some(invalid_key));
        prop_assert!(result.is_err(), "Invalid API key with prefix '{}' should fail", prefix);
    }
}

/// Property-based test: JSON Parsing Roundtrip
///
/// PROPERTY: For all valid JSON structures, parsing should succeed
/// and the parsed structure should preserve all original fields.
#[proptest!]
fn property_json_parsing_preserves_structure(
    #[regex(r"[a-zA-Z0-9_-]{1,20}")] model: String,
    max_tokens: u32,
    message_count: usize,
    #[filter(#|msgs: &Vec<_>| !msgs.is_empty() && msgs.len() <= 10)] messages: Vec<String>
) {
    use proptest::collection::vec;

    // Create valid request structure
    let message_objects: Vec<Value> = messages.into_iter()
        .enumerate()
        .map(|(i, msg)| json!({
            "role": if i % 2 == 0 { "user" } else { "assistant" },
            "content": msg
        }))
        .collect();

    let request_json = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": message_objects
    });

    // Test parsing property
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result: Result<Value, ApiResponseError> = rt.block_on(
        parse_json_body_safely(request_json.to_string().as_bytes().into())
    );

    prop_assert!(result.is_ok(), "Valid JSON structure should always parse");

    let parsed = result.unwrap();

    // Verify roundtrip property
    prop_assert_eq!(parsed["model"], request_json["model"], "Model field should be preserved");
    prop_assert_eq!(parsed["max_tokens"], request_json["max_tokens"], "Max tokens should be preserved");
    prop_assert_eq!(parsed["messages"].as_array().unwrap().len(),
                   request_json["messages"].as_array().unwrap().len(),
                   "Message count should be preserved");
}

/// Property-based test: Request Handler Idempotency
///
/// PROPERTY: For identical requests, the handler should return identical responses
/// (excluding timestamps or other dynamic fields).
#[proptest!]
async fn property_request_handler_idempotency(
    #[regex(r"[a-zA-Z0-9_-]{1,30}")] model: String,
    max_tokens: u32
) {
    // Create two identical handlers
    let handler1 = Arc::new(PropertyTestHandler);
    let handler2 = Arc::new(PropertyTestHandler);

    let routes1 = create_http_routes_with_middleware(handler1);
    let routes2 = create_http_routes_with_middleware(handler2);

    let request_json = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "test message"}]
    });

    // Send identical requests
    let response1 = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-api03-test-key")
        .body(request_json.to_string())
        .reply(&routes1)
        .await;

    let response2 = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-api03-test-key")
        .body(request_json.to_string())
        .reply(&routes2)
        .await;

    // Property: responses should be identical
    prop_assert_eq!(response1.status(), response2.status(),
                   "Identical requests should return same status");

    let body1: Value = serde_json::from_slice(response1.body()).unwrap();
    let body2: Value = serde_json::from_slice(response2.body()).unwrap();

    prop_assert_eq!(body1["model"], body2["model"],
                   "Identical requests should return same model");
    prop_assert_eq!(body1["content"], body2["content"],
                   "Identical requests should return same content");
}

/// Property-based test: HTTP Status Code Determinism
///
/// PROPERTY: For the same request conditions, HTTP status codes should be consistent
#[proptest!]
async fn property_http_status_codes_are_deterministic(
    has_auth: bool,
    has_content_type: bool,
    has_valid_json: bool
) {
    let handler = Arc::new(PropertyTestHandler);
    let routes = create_http_routes_with_middleware(handler);

    let request_json = if has_valid_json {
        json!({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "test"}]
        }).to_string()
    } else {
        "{ invalid json }".to_string()
    };

    let mut request = test::request()
        .method("POST")
        .path("/v1/messages")
        .body(request_json);

    if has_auth {
        request = request.header("authorization", "Bearer sk-ant-api03-test-key");
    }
    if has_content_type {
        request = request.header("content-type", "application/json");
    }

    // Send request multiple times
    let mut status_codes = Vec::new();
    for _ in 0..3 {
        let response = request.clone().reply(&routes).await;
        status_codes.push(response.status());
    }

    // Property: All status codes should be identical
    let first_status = status_codes[0];
    for (i, &status) in status_codes.iter().enumerate() {
        prop_assert_eq!(status, first_status,
                       "Request {} should have consistent status code", i);
    }

    // Property: Expected status based on conditions
    let expected_status = if !has_auth {
        401 // Unauthorized
    } else if !has_content_type {
        400 // Bad Request (missing content-type)
    } else if !has_valid_json {
        400 // Bad Request (invalid JSON)
    } else {
        200 // OK
    };

    prop_assert_eq!(first_status, expected_status,
                   "Status code {} should match expected {} for conditions: auth={}, content_type={}, valid_json={}",
                   first_status, expected_status, has_auth, has_content_type, has_valid_json);
}

/// Property-based test: Response Structure Consistency
///
/// PROPERTY: All successful responses should have consistent structure
#[proptest!]
async fn property_successful_response_structure_consistency(
    #[regex(r"[a-zA-Z0-9_-]{1,20}")] model: String
) {
    let handler = Arc::new(PropertyTestHandler);
    let routes = create_http_routes_with_middleware(handler);

    let request_json = json!({
        "model": model,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "test"}]
    });

    let response = test::request()
        .method("POST")
        .path("/v1/messages")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-ant-api03-test-key")
        .body(request_json.to_string())
        .reply(&routes)
        .await;

    // Property: Successful response structure
    prop_assert_eq!(response.status(), 200, "Valid request should return 200");

    let body: Value = serde_json::from_slice(response.body()).unwrap();

    // Required fields property
    prop_assert!(body.get("content").is_some(), "Response must have content field");
    prop_assert!(body.get("model").is_some(), "Response must have model field");
    prop_assert!(body.get("usage").is_some(), "Response must have usage field");

    // Type consistency property
    prop_assert!(body["content"].is_string(), "Content must be string");
    prop_assert!(body["model"].is_string(), "Model must be string");
    prop_assert!(body["usage"].is_object(), "Usage must be object");
}

/// Property-based test: Health Check Always Returns Same Structure
///
/// PROPERTY: Health check endpoint should always return consistent structure
#[proptest!]
async fn property_health_check_response_always_consistent() {
    let handler = Arc::new(PropertyTestHandler);
    let routes = create_http_routes_with_middleware(handler);

    // Test multiple health check calls
    for _ in 0..10 {
        let response = test::request()
            .method("GET")
            .path("/health")
            .reply(&routes)
            .await;

        // Property: Always returns 200
        prop_assert_eq!(response.status(), 200, "Health check must always return 200");

        let body: Value = serde_json::from_slice(response.body()).unwrap();

        // Property: Always has required fields
        prop_assert!(body.get("status").is_some(), "Health check must have status field");
        prop_assert!(body.get("timestamp").is_some(), "Health check must have timestamp field");

        // Property: Always has correct values
        prop_assert_eq!(body["status"], "healthy", "Status must always be 'healthy'");
        prop_assert!(body["timestamp"].is_string(), "Timestamp must be string");
    }
}

/// Property-based test: CORS Headers Consistency
///
/// PROPERTY: CORS preflight requests should always return consistent headers
#[proptest!]
async fn property_cors_headers_always_consistent(
    #[regex(r"[a-zA-Z0-9.-]{5,20}")] origin: String
) {
    let handler = Arc::new(PropertyTestHandler);
    let routes = create_http_routes_with_middleware(handler);

    let response = test::request()
        .method("OPTIONS")
        .path("/v1/messages")
        .header("origin", &origin)
        .header("access-control-request-method", "POST")
        .reply(&routes)
        .await;

    // Property: Always returns 200
    prop_assert_eq!(response.status(), 200, "CORS preflight must always return 200");

    let headers = response.headers();

    // Property: Always has required CORS headers
    prop_assert!(headers.contains_key("access-control-allow-origin"),
                 "Must always include Access-Control-Allow-Origin");
    prop_assert!(headers.contains_key("access-control-allow-methods"),
                 "Must always include Access-Control-Allow-Methods");
    prop_assert!(headers.contains_key("access-control-allow-headers"),
                 "Must always include Access-Control-Allow-Headers");
}

/// Property-based test: Error Response Structure
///
/// PROPERTY: All error responses should follow consistent structure
#[proptest!]
async fn property_error_response_structure_consistency(
    error_conditions: Vec<bool> // [no_auth, no_content_type, invalid_json]
) {
    let handler = Arc::new(PropertyTestHandler);
    let routes = create_http_routes_with_middleware(handler);

    let (no_auth, no_content_type, invalid_json) = match error_conditions.len() {
        0 => (false, false, false),
        1 => (error_conditions[0], false, false),
        2 => (error_conditions[0], error_conditions[1], false),
        _ => (error_conditions[0], error_conditions[1], error_conditions[2]),
    };

    let request_json = if invalid_json {
        "{ invalid json }".to_string()
    } else {
        json!({
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "test"}]
        }).to_string()
    };

    let mut request = test::request()
        .method("POST")
        .path("/v1/messages")
        .body(request_json);

    if !no_auth {
        request = request.header("authorization", "Bearer sk-ant-api03-test-key");
    }
    if !no_content_type {
        request = request.header("content-type", "application/json");
    }

    let response = request.reply(&routes).await;

    // If this is an error response (not 200), check structure
    if response.status() != 200 {
        let body: Value = serde_json::from_slice(response.body()).unwrap();

        // Property: Error responses must have error object
        prop_assert!(body.get("error").is_some(), "Error response must have error object");

        let error_obj = &body["error"];

        // Property: Error object must have type and message
        prop_assert!(error_obj.get("type").is_some(), "Error must have type field");
        prop_assert!(error_obj.get("message").is_some(), "Error must have message field");

        // Property: Type should be string
        prop_assert!(error_obj["type"].is_string(), "Error type must be string");
        prop_assert!(error_obj["message"].is_string(), "Error message must be string");
    }
}