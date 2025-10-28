//! Pensieve API Server - HTTP API server with streaming support
//!
//! This is the Layer 3 (L3) HTTP API server crate that provides:
//! - Warp-based HTTP server
//! - Anthropic API compatibility
//! - Streaming response support
//! - Concurrent request handling
//!
//! Depends on all L1 and L2 crates, plus pensieve-03 for API models.

use pensieve_07_core::CoreError;
use pensieve_03::{anthropic::*, ApiError, ApiMessage, StreamingResponse as ApiStreamingResponse};
use std::sync::Arc;
use std::pin::Pin;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use warp::Filter;
use futures::{Stream, StreamExt};
use bytes::Bytes;
use uuid::Uuid;

/// Type alias for streaming responses
pub type StreamingResponse = Pin<Box<dyn Stream<Item = String> + Send>>;

/// Server-specific error types
pub mod error {
    use super::*;
    use thiserror::Error;

    /// Server result type
    pub type ServerResult<T> = std::result::Result<T, ServerError>;

    /// Server-specific errors
    #[derive(Error, Debug)]
    pub enum ServerError {
        #[error("HTTP error: {0}")]
        Http(String),

        #[error("Request error: {0}")]
        Request(String),

        #[error("Response error: {0}")]
        Response(String),

        #[error("Configuration error: {0}")]
        Configuration(String),

        #[error("Internal server error: {0}")]
        Internal(String),

        #[error("API error: {0}")]
        Api(#[from] ApiError),

        #[error("Core error: {0}")]
        Core(#[from] CoreError),

        #[error("Warp error: {0}")]
        Warp(#[from] warp::Error),
    }
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub request_timeout_ms: u64,
    pub enable_cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent_requests: 100,
            request_timeout_ms: 30000,
            enable_cors: true,
        }
    }
}

/// Core API traits
pub mod traits {
    use super::*;

    /// Trait for API servers
    #[async_trait::async_trait]
    pub trait ApiServer: Send + Sync {
        /// Start the server
        async fn start(&self) -> error::ServerResult<()>;

        /// Shutdown the server
        async fn shutdown(&self) -> error::ServerResult<()>;

        /// Check if server is healthy
        fn is_healthy(&self) -> bool;

        /// Get server address
        fn address(&self) -> String;
    }

    /// Trait for request handlers
    #[async_trait::async_trait]
    pub trait RequestHandler: Send + Sync {
        /// Handle a message request
        async fn handle_message(&self, request: CreateMessageRequest) -> error::ServerResult<CreateMessageResponse>;

        /// Handle a streaming request
        async fn handle_stream(&self, request: CreateMessageRequest) -> error::ServerResult<StreamingResponse>;
    }
}

/// Mock request handler for testing
#[derive(Debug, Clone)]
pub struct MockRequestHandler {
    response_delay_ms: u64,
}

impl MockRequestHandler {
    pub fn new(response_delay_ms: u64) -> Self {
        Self { response_delay_ms }
    }
}

#[async_trait::async_trait]
impl traits::RequestHandler for MockRequestHandler {
    async fn handle_message(&self, request: CreateMessageRequest) -> error::ServerResult<CreateMessageResponse> {
        // Validate request
        request.validate()?;

        // Simulate processing delay
        tokio::time::sleep(tokio::time::Duration::from_millis(self.response_delay_ms)).await;

        // Create mock response
        Ok(CreateMessageResponse {
            id: Uuid::new_v4().to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text {
                text: format!("Mock response to: {}",
                    request.messages.first()
                        .and_then(|m| m.content.first())
                        .and_then(|c| match c {
                            Content::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .unwrap_or("no message")
                ),
            }],
            model: request.model.clone(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        })
    }

    async fn handle_stream(&self, request: CreateMessageRequest) -> error::ServerResult<StreamingResponse> {
        // Validate request
        request.validate()?;

        let delay = self.response_delay_ms;
        let response_text = "Mock streaming response";
        let chars: Vec<char> = response_text.chars().collect();

        // Create a simple mock stream for now
        let stream = futures::stream::iter(vec![
            "data: {\"type\": \"message_start\"}\n\n".to_string(),
            "data: {\"type\": \"content_block_delta\", \"delta\": {\"text\": \"Mock\"}}\n\n".to_string(),
            "data: {\"type\": \"message_stop\"}\n\n".to_string(),
        ]);

        Ok(Box::pin(stream))
    }
}

/// HTTP API server implementation
pub struct HttpApiServer {
    config: ServerConfig,
    handler: Arc<dyn traits::RequestHandler>,
    shutdown_signal: Arc<RwLock<bool>>,
    server_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl std::fmt::Debug for HttpApiServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpApiServer")
            .field("config", &self.config)
            .field("shutdown_signal", &self.shutdown_signal)
            .field("server_handle", &self.server_handle)
            .finish()
    }
}

impl HttpApiServer {
    pub fn new(config: ServerConfig, handler: Arc<dyn traits::RequestHandler>) -> Self {
        Self {
            config,
            handler,
            shutdown_signal: Arc::new(RwLock::new(false)),
            server_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Create routes for the API server
    fn routes(&self) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        let health = warp::path("health")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "status": "healthy",
                    "timestamp": "2024-01-01T00:00:00Z" // Simplified for now
                }))
            });

        let messages = warp::path("v1")
            .and(warp::path("messages"))
            .and(warp::post())
            .and(warp::body::json())
            .and(warp::header::optional::<String>("x-stream"))
            .and(self.with_handler())
            .and_then(handle_create_message);

        // Simple routes without CORS for now
        health.or(messages)
    }

    /// Add handler to warp filter
    fn with_handler(&self) -> impl Filter<Extract = (Arc<dyn traits::RequestHandler>,), Error = std::convert::Infallible> + Clone {
        let handler = self.handler.clone();
        warp::any().map(move || handler.clone())
    }
}

#[async_trait::async_trait]
impl traits::ApiServer for HttpApiServer {
    async fn start(&self) -> error::ServerResult<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        info!("Starting server on {}", addr);

        let routes = self.routes();
        let shutdown_signal = self.shutdown_signal.clone();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let server_handle = tokio::spawn(async move {
            let addr: std::net::SocketAddr = addr.parse().expect("Invalid address");
            let (_, fut) = warp::serve(routes).bind_with_graceful_shutdown(addr, async {
                rx.await.ok();
                info!("Server shutdown signal received");
            });
            fut.await;
        });

        *self.server_handle.write().await = Some(server_handle);

        info!("Server started successfully on {}", self.address());
        Ok(())
    }

    async fn shutdown(&self) -> error::ServerResult<()> {
        info!("Shutting down server");
        *self.shutdown_signal.write().await = true;

        if let Some(handle) = self.server_handle.write().await.take() {
            if let Err(e) = handle.await {
                warn!("Error waiting for server shutdown: {}", e);
            }
        }

        info!("Server shutdown complete");
        Ok(())
    }

    fn is_healthy(&self) -> bool {
        true // For now, always return healthy
    }

    fn address(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }
}

/// HTTP request handler
async fn handle_create_message(
    request: CreateMessageRequest,
    stream_header: Option<String>,
    handler: Arc<dyn traits::RequestHandler>,
) -> std::result::Result<impl warp::Reply, warp::Rejection> {
    // Validate request
    if let Err(e) = request.validate() {
        error!("Invalid request: {}", e);
        return Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": e.to_string()
                }
            })),
            warp::http::StatusCode::BAD_REQUEST,
        ));
    }

    // Check if streaming is requested
    let is_streaming = stream_header
        .map(|s| s.to_lowercase() == "true")
        .unwrap_or(request.stream.unwrap_or(false));

    if is_streaming {
        // For now, treat streaming as regular requests
        match handler.handle_message(request).await {
            Ok(response) => {
                Ok(warp::reply::with_status(
                    warp::reply::json(&response),
                    warp::http::StatusCode::OK,
                ))
            }
            Err(e) => {
                error!("Message handling error: {}", e);
                Ok(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "error": {
                            "type": "internal_error",
                            "message": e.to_string()
                        }
                    })),
                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                ))
            }
        }
    } else {
        match handler.handle_message(request).await {
            Ok(response) => {
                Ok(warp::reply::with_status(
                    warp::reply::json(&response),
                    warp::http::StatusCode::OK,
                ))
            }
            Err(e) => {
                error!("Message handling error: {}", e);
                Ok(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "error": {
                            "type": "internal_error",
                            "message": e.to_string()
                        }
                    })),
                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                ))
            }
        }
    }
}

/// Re-export commonly used items
pub use error::{ServerError, ServerResult};
pub use traits::{ApiServer, RequestHandler};

/// Result type alias for convenience
pub type Result<T> = ServerResult<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_concurrent_requests, 100);
        assert_eq!(config.request_timeout_ms, 30000);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_mock_request_handler_creation() {
        let handler = MockRequestHandler::new(100);
        assert_eq!(handler.response_delay_ms, 100);
    }

    #[tokio::test]
    async fn test_mock_message_handling() {
        let handler = MockRequestHandler::new(10);
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "Hello, world!".to_string(),
                }],
            }],
            temperature: None,
            top_p: None,
            stream: None,
            system: None,
        };

        let result = handler.handle_message(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.role, Role::Assistant);
        assert!(!response.content.is_empty());
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn test_mock_stream_handling() {
        let handler = MockRequestHandler::new(10);
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "Hello, world!".to_string(),
                }],
            }],
            temperature: None,
            top_p: None,
            stream: Some(true),
            system: None,
        };

        let result = handler.handle_stream(request).await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let mut events = Vec::new();

        // Collect up to 10 events or timeout
        let timeout_result = timeout(Duration::from_secs(5), async {
            while let Some(event) = stream.next().await {
                events.push(event);
                if events.len() >= 10 {
                    break;
                }
            }
        }).await;

        assert!(timeout_result.is_ok());
        assert!(!events.is_empty());

        // Check that we have start and stop events
        let has_start = events.iter().any(|e| e.contains("message_start"));
        let has_stop = events.iter().any(|e| e.contains("message_stop"));
        assert!(has_start);
        assert!(has_stop);
    }

    #[tokio::test]
    async fn test_http_api_server_creation() {
        let config = ServerConfig::default();
        let handler = Arc::new(MockRequestHandler::new(10));
        let server = HttpApiServer::new(config.clone(), handler);

        assert_eq!(server.address(), format!("{}:{}", config.host, config.port));
        assert!(server.is_healthy());
    }

    #[tokio::test]
    async fn test_invalid_request_handling() {
        let handler = MockRequestHandler::new(10);
        let invalid_request = CreateMessageRequest {
            model: "".to_string(), // Empty model should fail validation
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "Hello".to_string(),
                }],
            }],
            temperature: None,
            top_p: None,
            stream: None,
            system: None,
        };

        let result = handler.handle_message(invalid_request).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let error = ServerError::Configuration("Invalid port".to_string());
        assert_eq!(error.to_string(), "Configuration error: Invalid port");

        let api_error = ApiError::Validation("Test error".to_string());
        let server_error = ServerError::Api(api_error);
        assert!(server_error.to_string().contains("API error"));
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        let config = ServerConfig {
            port: 0, // Use random port for testing
            ..Default::default()
        };
        let handler = Arc::new(MockRequestHandler::new(10));
        let server = HttpApiServer::new(config, handler);

        // Test that we can start and shutdown
        let start_result = server.start().await;
        assert!(start_result.is_ok());

        // Give it a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        let shutdown_result = server.shutdown().await;
        assert!(shutdown_result.is_ok());
    }

    // Property-based test for server configuration
    #[test]
    fn test_proptest_server_config() {
        use proptest::prelude::*;

        proptest!(|(
            port in 1024u16..=65535,
            max_requests in 1usize..=1000,
            timeout_ms in 1000u64..=60000
        )| {
            let config = ServerConfig {
                host: "127.0.0.1".to_string(),
                port,
                max_concurrent_requests: max_requests,
                request_timeout_ms: timeout_ms,
                enable_cors: true,
            };

            prop_assert!(config.port >= 1024);
            prop_assert!(config.port <= 65535);
            prop_assert!(config.max_concurrent_requests > 0);
            prop_assert!(config.request_timeout_ms > 0);
        });
    }
}