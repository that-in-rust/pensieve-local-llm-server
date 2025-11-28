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
use pensieve_03::{anthropic::*, ApiError, ApiMessage};
use std::sync::Arc;
use std::pin::Pin;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use warp::Filter;
use futures::Stream;
use uuid::Uuid;
use serde_json;

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
            port: 7777,
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

/// MLX request handler that calls persistent Python MLX HTTP server
#[derive(Clone)]
pub struct MlxRequestHandler {
    model_path: String,
    mlx_server_url: String,
    http_client: reqwest::Client,
    /// Semaphore to limit concurrent inference requests
    /// This prevents memory spikes from multiple simultaneous model activations
    inference_semaphore: Arc<tokio::sync::Semaphore>,
    /// Memory monitor for request gating
    memory_monitor: Arc<pensieve_09_anthropic_proxy::memory::SystemMemoryMonitor>,
}

impl std::fmt::Debug for MlxRequestHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxRequestHandler")
            .field("model_path", &self.model_path)
            .field("mlx_server_url", &self.mlx_server_url)
            .finish()
    }
}

impl MlxRequestHandler {
    /// Create a new MLX request handler
    ///
    /// Args:
    ///   - model_path: Path to the MLX model (for reference only, server loads it)
    ///   - mlx_server_url: URL of the persistent MLX server (default: http://127.0.0.1:8765)
    ///   - max_concurrent: Maximum concurrent inference requests (default: 2)
    pub fn new(model_path: String) -> Self {
        Self::with_config(
            model_path,
            "http://127.0.0.1:8765".to_string(),
            2, // Safe default for Apple Silicon
        )
    }

    pub fn with_config(
        model_path: String,
        mlx_server_url: String,
        max_concurrent: usize,
    ) -> Self {
        info!("Initializing MLX handler - Server: {}, Max concurrent: {}", mlx_server_url, max_concurrent);

        Self {
            model_path,
            mlx_server_url,
            http_client: reqwest::Client::new(),
            inference_semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            memory_monitor: Arc::new(pensieve_09_anthropic_proxy::memory::SystemMemoryMonitor::new()),
        }
    }

    /// Extract the conversation text from messages
    fn extract_prompt(&self, messages: &[pensieve_03::anthropic::Message]) -> String {
        let mut prompt_parts = Vec::new();

        for message in messages {
            match message.role {
                pensieve_03::anthropic::Role::User => {
                    match &message.content {
                        pensieve_03::anthropic::MessageContent::String(text) => {
                            prompt_parts.push(format!("User: {}", text));
                        }
                        pensieve_03::anthropic::MessageContent::Blocks(blocks) => {
                            for content in blocks {
                                if let pensieve_03::anthropic::Content::Text { text } = content {
                                    prompt_parts.push(format!("User: {}", text));
                                }
                            }
                        }
                    }
                }
                pensieve_03::anthropic::Role::Assistant => {
                    match &message.content {
                        pensieve_03::anthropic::MessageContent::String(text) => {
                            prompt_parts.push(format!("Assistant: {}", text));
                        }
                        pensieve_03::anthropic::MessageContent::Blocks(blocks) => {
                            for content in blocks {
                                if let pensieve_03::anthropic::Content::Text { text } = content {
                                    prompt_parts.push(format!("Assistant: {}", text));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Add a final assistant prompt
        prompt_parts.push("Assistant:".to_string());
        prompt_parts.join("\n")
    }

    /// Call persistent Python MLX HTTP server for non-streaming generation
    ///
    /// This replaces the old process-per-request approach with HTTP calls to a
    /// persistent server. The model stays loaded in the server's memory, avoiding
    /// repeated multi-GB model loads.
    async fn call_mlx_server(&self, prompt: &str, max_tokens: u32, temperature: f32) -> error::ServerResult<String> {
        #[derive(serde::Serialize)]
        struct GenerateRequest {
            prompt: String,
            max_tokens: u32,
            temperature: f32,
            stream: bool,
        }

        #[derive(serde::Deserialize)]
        struct GenerateResponse {
            text: String,
        }

        let request_body = GenerateRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            stream: false,
        };

        let url = format!("{}/generate", self.mlx_server_url);

        let response = self.http_client
            .post(&url)
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| error::ServerError::Internal(format!("MLX server request failed: {}. Is the server running? Start with: python3 python_bridge/mlx_server.py --model-path {}", e, self.model_path)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(error::ServerError::Internal(format!("MLX server error ({}): {}", status, error_text)));
        }

        let result: GenerateResponse = response.json().await
            .map_err(|e| error::ServerError::Internal(format!("Failed to parse MLX server response: {}", e)))?;

        Ok(result.text)
    }

    /// Call persistent Python MLX HTTP server for streaming generation
    ///
    /// Returns a byte stream from the server as Vec<u8> chunks.
    async fn call_mlx_server_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> error::ServerResult<reqwest::Response> {
        #[derive(serde::Serialize)]
        struct GenerateRequest {
            prompt: String,
            max_tokens: u32,
            temperature: f32,
            stream: bool,
        }

        let request_body = GenerateRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            stream: true,
        };

        let url = format!("{}/generate", self.mlx_server_url);

        let response = self.http_client
            .post(&url)
            .json(&request_body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| error::ServerError::Internal(format!("MLX streaming request failed: {}. Is the server running?", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(error::ServerError::Internal(format!("MLX streaming error ({}): {}", status, error_text)));
        }

        Ok(response)
    }

    /// Parse JSON response from Python bridge
    fn parse_mlx_response(&self, response: &str) -> error::ServerResult<String> {
        use serde_json::Value;

        // Parse each line of JSON output
        for line in response.lines() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<Value>(line) {
                Ok(json) => {
                    if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                        return Ok(text.to_string());
                    } else if let Some(error) = json.get("error").and_then(|v| v.as_str()) {
                        return Err(error::ServerError::Internal(format!("MLX inference error: {}", error)));
                    }
                }
                Err(e) => {
                    // Log parsing error but continue trying other lines
                    warn!("Failed to parse MLX response line: {} - {}", line, e);
                }
            }
        }

        Err(error::ServerError::Internal("No valid response from MLX bridge".to_string()))
    }
}

#[async_trait::async_trait]
impl traits::RequestHandler for MlxRequestHandler {
    async fn handle_message(&self, request: CreateMessageRequest) -> error::ServerResult<CreateMessageResponse> {
        use pensieve_09_anthropic_proxy::memory::MemoryMonitor;

        // Validate request
        request.validate()?;

        // MEMORY GATING: Check memory BEFORE acquiring semaphore
        let mem_status = self.memory_monitor.check_status();
        if !mem_status.accepts_requests() {
            let available = self.memory_monitor.available_gb();
            warn!("Request rejected - memory status: {:?}, available: {:.2}GB", mem_status, available);
            return Err(error::ServerError::Internal(format!(
                "Insufficient memory for inference. Status: {:?}, Available: {:.2}GB",
                mem_status, available
            )));
        }

        // CONCURRENCY LIMITING: Acquire semaphore permit
        // This ensures only N requests can run simultaneously, preventing memory spikes
        let _permit = self.inference_semaphore.acquire().await
            .map_err(|e| error::ServerError::Internal(format!("Semaphore error: {}", e)))?;

        info!("Processing inference request (memory status: {:?})", mem_status);

        // Extract prompt from messages
        let prompt = self.extract_prompt(&request.messages);

        // Get generation parameters
        let max_tokens = request.max_tokens;
        let temperature = request.temperature.unwrap_or(0.7);

        // Call persistent MLX HTTP server (model already loaded!)
        let generated_text = self.call_mlx_server(&prompt, max_tokens, temperature).await?;

        // Calculate token counts
        let input_tokens = prompt.split_whitespace().count() as u32;
        let output_tokens = generated_text.split_whitespace().count() as u32;

        // Create response
        Ok(CreateMessageResponse {
            id: Uuid::new_v4().to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![Content::Text {
                text: generated_text,
            }],
            model: request.model.clone(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens,
                output_tokens,
            },
        })
    }

    async fn handle_stream(&self, request: CreateMessageRequest) -> error::ServerResult<StreamingResponse> {
        use pensieve_09_anthropic_proxy::memory::MemoryMonitor;
        use futures::StreamExt;
        use tokio_util::codec::{FramedRead, LinesCodec};

        // Validate request
        request.validate()?;

        // MEMORY GATING: Check memory BEFORE acquiring semaphore
        let mem_status = self.memory_monitor.check_status();
        if !mem_status.accepts_requests() {
            let available = self.memory_monitor.available_gb();
            warn!("Streaming request rejected - memory status: {:?}, available: {:.2}GB", mem_status, available);
            return Err(error::ServerError::Internal(format!(
                "Insufficient memory for streaming. Status: {:?}, Available: {:.2}GB",
                mem_status, available
            )));
        }

        // CONCURRENCY LIMITING: Acquire semaphore permit
        // Keep it alive for the duration of streaming
        let permit = self.inference_semaphore.clone().acquire_owned().await
            .map_err(|e| error::ServerError::Internal(format!("Semaphore error: {}", e)))?;

        info!("Processing streaming request (memory status: {:?})", mem_status);

        // Extract prompt from messages
        let prompt = self.extract_prompt(&request.messages);

        // Get generation parameters
        let max_tokens = request.max_tokens;
        let temperature = request.temperature.unwrap_or(0.7);

        // Call persistent MLX HTTP server for streaming
        let response = self.call_mlx_server_streaming(&prompt, max_tokens, temperature).await?;

        // Convert byte stream to SSE events
        // This is TRUE STREAMING - no buffering!
        let stream = async_stream::stream! {
            // Send message start event
            yield "data: {\"type\": \"message_start\"}\n\n".to_string();

            // Process chunks as they arrive
            let mut byte_stream = response.bytes_stream();
            let mut buffer = bytes::BytesMut::new();

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        buffer.extend_from_slice(&chunk);

                        // Process complete lines from buffer
                        while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                            let line_bytes = buffer.split_to(newline_pos + 1);
                            let line = String::from_utf8_lossy(&line_bytes);

                            if line.trim().is_empty() {
                                continue;
                            }

                            // Parse JSON chunk from MLX server
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                                    // Convert to Anthropic SSE format
                                    let event = format!(
                                        "data: {{\"type\": \"content_block_delta\", \"delta\": {{\"text\": \"{}\"}}}}\n\n",
                                        text.escape_default()
                                    );
                                    yield event;
                                } else if json.get("type").and_then(|v| v.as_str()) == Some("complete") {
                                    // Streaming complete
                                    break;
                                } else if let Some(error) = json.get("error").and_then(|v| v.as_str()) {
                                    error!("Streaming error from MLX server: {}", error);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Stream error: {}", e);
                        break;
                    }
                }
            }

            // Send message stop event
            yield "data: {\"type\": \"message_stop\"}\n\n".to_string();

            // Permit is dropped here, releasing semaphore
            drop(permit);
        };

        Ok(Box::pin(stream))
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
                        .and_then(|m| {
                            match &m.content {
                                pensieve_03::anthropic::MessageContent::String(s) => Some(s.as_str()),
                                pensieve_03::anthropic::MessageContent::Blocks(blocks) => {
                                    blocks.first().and_then(|c| match c {
                                        Content::Text { text } => Some(text.as_str()),
                                        _ => None,
                                    })
                                }
                            }
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

        let _delay = self.response_delay_ms; // Simulate delay for realism

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
    shutdown_tx: Arc<RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
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
            shutdown_tx: Arc::new(RwLock::new(None)),
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

        let models = warp::path("v1")
            .and(warp::path("models"))
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "object": "list",
                    "data": [
                        {
                            "id": "phi-3-mini",
                            "object": "model",
                            "created": 1677610602,
                            "owned_by": "microsoft"
                        }
                    ]
                }))
            });

        let messages = warp::path("v1")
            .and(warp::path("messages"))
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and(warp::body::json())
            .and(warp::header::optional::<String>("x-stream"))
            .and(self.with_handler())
            .and_then(handle_create_message);

        // Simple routes without CORS for now
        health.or(models).or(messages)
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
        let _shutdown_signal = self.shutdown_signal.clone();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        *self.shutdown_tx.write().await = Some(tx);
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

        // Signal shutdown using stored sender
        if let Some(tx) = self.shutdown_tx.write().await.take() {
            let _ = tx.send(());
        }

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

/// Validate API key from Authorization header
fn validate_api_key(auth_header: &str) -> bool {
    // Extract Bearer token from "Bearer <token>" format
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        // For now, accept a simple test token
        // In production, this should validate against environment variable or secure storage
        token == "test-api-key-12345"
            || token == "pensieve-local-token"
            || token.starts_with("sk-ant-api")
    } else {
        false
    }
}

/// HTTP request handler
async fn handle_create_message(
    auth_header: Option<String>,
    request: CreateMessageRequest,
    stream_header: Option<String>,
    handler: Arc<dyn traits::RequestHandler>,
) -> std::result::Result<Box<dyn warp::Reply + Send>, warp::Rejection> {
    // Validate authentication if provided
    // For local development, allow requests without auth header
    if let Some(auth) = auth_header {
        if !validate_api_key(&auth) {
            error!("Invalid authentication");
            return Ok(Box::new(warp::reply::with_status(
                warp::reply::json(&serde_json::json!({
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid API key"
                    }
                })),
                warp::http::StatusCode::UNAUTHORIZED,
            )) as Box<dyn warp::Reply + Send>);
        }
    }
    // If no auth header provided, allow (for local development)

    // Validate request
    if let Err(e) = request.validate() {
        error!("Invalid request: {}", e);
        return Ok(Box::new(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": e.to_string()
                }
            })),
            warp::http::StatusCode::BAD_REQUEST,
        )) as Box<dyn warp::Reply + Send>);
    }

    // Check if streaming is requested
    let is_streaming = stream_header
        .map(|s| s.to_lowercase() == "true")
        .unwrap_or(request.stream.unwrap_or(false));

    if is_streaming {
        // FIXED: Use proper streaming handler with SSE response
        match handler.handle_stream(request).await {
            Ok(stream) => {
                // Convert Stream<String> to SSE response with proper headers
                use bytes::Bytes;
                use futures::StreamExt;

                let byte_stream = stream.map(|s| {
                    Ok::<Bytes, std::convert::Infallible>(Bytes::from(s))
                });

                let body = warp::hyper::Body::wrap_stream(byte_stream);
                let mut response = warp::http::Response::new(body);

                // Set proper SSE headers
                response.headers_mut().insert(
                    warp::http::header::CONTENT_TYPE,
                    warp::http::HeaderValue::from_static("text/event-stream"),
                );
                response.headers_mut().insert(
                    warp::http::header::CACHE_CONTROL,
                    warp::http::HeaderValue::from_static("no-cache"),
                );
                response.headers_mut().insert(
                    warp::http::header::CONNECTION,
                    warp::http::HeaderValue::from_static("keep-alive"),
                );
                response.headers_mut().insert(
                    warp::http::header::ACCESS_CONTROL_ALLOW_ORIGIN,
                    warp::http::HeaderValue::from_static("*"),
                );

                // Return the custom response directly - warp::http::Response implements warp::Reply
                Ok(Box::new(response) as Box<dyn warp::Reply + Send>)
            }
            Err(e) => {
                error!("Streaming error: {}", e);
                Ok(Box::new(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "error": {
                            "type": "internal_error",
                            "message": e.to_string()
                        }
                    })),
                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                )) as Box<dyn warp::Reply + Send>)
            }
        }
    } else {
        match handler.handle_message(request).await {
            Ok(response) => {
                Ok(Box::new(warp::reply::with_status(
                    warp::reply::json(&response),
                    warp::http::StatusCode::OK,
                )) as Box<dyn warp::Reply + Send>)
            }
            Err(e) => {
                error!("Message handling error: {}", e);
                Ok(Box::new(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "error": {
                            "type": "internal_error",
                            "message": e.to_string()
                        }
                    })),
                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                )) as Box<dyn warp::Reply + Send>)
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
        assert_eq!(config.port, 7777);
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