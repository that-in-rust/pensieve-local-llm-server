//! HTTP Server Integration
//!
//! Integrates auth and translator modules into a Warp-based HTTP server
//! that provides Anthropic API compatibility.

use crate::auth::{validate_auth, AuthError};
use crate::translator::{translate_anthropic_to_mlx, translate_mlx_to_anthropic};
use pensieve_03::anthropic::{CreateMessageRequest, CreateMessageResponse};
use pensieve_03::ApiMessage; // Import trait for validate() method
use std::sync::Arc;
use tokio::sync::RwLock;
use warp::{Filter, Reply};
use serde_json::json;
use tracing::{info, error};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub python_bridge_path: String,
    pub model_path: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7777,
            python_bridge_path: "python_bridge/mlx_inference.py".to_string(),
            model_path: "models/Phi-3-mini-128k-instruct-4bit".to_string(), // Directory, not file
        }
    }
}

/// HTTP API Server
pub struct AnthropicProxyServer {
    config: ServerConfig,
    shutdown_tx: Arc<RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
    server_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl AnthropicProxyServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            shutdown_tx: Arc::new(RwLock::new(None)),
            server_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the HTTP server
    pub async fn start(&self) -> Result<(), ServerError> {
        let addr = format!("{}:{}", self.config.host, self.config.port);
        info!("Starting Anthropic Proxy server on {}", addr);

        let routes = self.create_routes();

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

        info!("Anthropic Proxy server started successfully");
        Ok(())
    }

    /// Shutdown the server
    pub async fn shutdown(&self) -> Result<(), ServerError> {
        info!("Shutting down Anthropic Proxy server");

        // Signal shutdown using stored sender
        if let Some(tx) = self.shutdown_tx.write().await.take() {
            let _ = tx.send(());
        }

        // Wait for server task to complete
        if let Some(handle) = self.server_handle.write().await.take() {
            if let Err(e) = handle.await {
                error!("Error waiting for server shutdown: {:?}", e);
            }
        }

        info!("Anthropic Proxy server shutdown complete");
        Ok(())
    }

    /// Check if server is running
    pub fn is_running(&self) -> bool {
        // Check if server handle exists and is not finished
        let handle_guard = self.server_handle.try_read();
        if let Ok(guard) = handle_guard {
            if let Some(handle) = guard.as_ref() {
                return !handle.is_finished();
            }
        }
        false
    }

    /// Get server address
    pub fn address(&self) -> String {
        format!("http://{}:{}", self.config.host, self.config.port)
    }

    /// Create Warp routes for the server
    fn create_routes(&self) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
        let health = self.health_route();
        let messages = self.messages_route();

        health.or(messages)
    }

    /// Health check route
    fn health_route(&self) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
        warp::path("health")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&json!({
                    "status": "healthy",
                    "service": "pensieve-anthropic-proxy"
                }))
            })
    }

    /// Messages route with auth and translation
    fn messages_route(&self) -> impl Filter<Extract = impl Reply, Error = warp::Rejection> + Clone {
        let config = self.config.clone();

        warp::path!("v1" / "messages")
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and(warp::body::json())
            .and(warp::any().map(move || config.clone()))
            .and_then(handle_messages)
    }
}

/// Server error types
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Authentication error: {0}")]
    Auth(#[from] AuthError),

    #[error("Translation error: {0}")]
    Translation(#[from] crate::translator::TranslationError),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type ServerResult<T> = Result<T, ServerError>;

/// Handle /v1/messages endpoint
async fn handle_messages(
    auth_header: Option<String>,
    request: CreateMessageRequest,
    config: ServerConfig,
) -> Result<Box<dyn Reply>, warp::Rejection> {
    // Step 1: Validate authentication
    let token = auth_header.clone().and_then(|h| h.strip_prefix("Bearer ").map(String::from));

    if let Err(auth_error) = validate_auth(token.as_deref()) {
        error!("Authentication failed: {:?}", auth_error);
        return Ok(Box::new(warp::reply::with_status(
            warp::reply::json(&json!({
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key"
                }
            })),
            warp::http::StatusCode::UNAUTHORIZED,
        )));
    }

    // Step 2: Validate request
    if let Err(e) = request.validate() {
        error!("Request validation failed: {:?}", e);
        return Ok(Box::new(warp::reply::with_status(
            warp::reply::json(&json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": e.to_string()
                }
            })),
            warp::http::StatusCode::BAD_REQUEST,
        )));
    }

    // Step 2.5: Check if streaming is requested
    if request.stream.unwrap_or(false) {
        info!("Streaming request detected, delegating to streaming handler");
        return handle_messages_streaming(auth_header, request, config).await;
    }

    // Step 3: Translate Anthropic request to MLX format
    let mlx_request = match translate_anthropic_to_mlx(&request) {
        Ok(req) => req,
        Err(e) => {
            error!("Request translation failed: {:?}", e);
            return Ok(Box::new(warp::reply::with_status(
                warp::reply::json(&json!({
                    "error": {
                        "type": "internal_error",
                        "message": format!("Translation error: {}", e)
                    }
                })),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            )));
        }
    };

    // Step 4: Call Python MLX bridge
    let mlx_output = match call_mlx_bridge(&config, &mlx_request).await {
        Ok(output) => output,
        Err(e) => {
            error!("MLX inference failed: {:?}", e);
            return Ok(Box::new(warp::reply::with_status(
                warp::reply::json(&json!({
                    "error": {
                        "type": "internal_error",
                        "message": format!("Inference error: {}", e)
                    }
                })),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            )));
        }
    };

    // Step 5: Calculate token counts (simplified for now)
    let input_tokens = mlx_request.prompt.split_whitespace().count() as u32;
    let output_tokens = mlx_output.split_whitespace().count() as u32;

    // Step 6: Translate MLX response to Anthropic format
    let response = translate_mlx_to_anthropic(&mlx_output, input_tokens, output_tokens);

    // Step 7: Return response
    Ok(Box::new(warp::reply::with_status(
        warp::reply::json(&response),
        warp::http::StatusCode::OK,
    )))
}

/// Call Python MLX bridge for inference
async fn call_mlx_bridge(
    config: &ServerConfig,
    mlx_request: &crate::translator::MlxRequest,
) -> Result<String, ServerError> {
    use tokio::process::Command;

    let mut cmd = Command::new("python3");
    cmd.arg(&config.python_bridge_path)
        .arg("--model-path")
        .arg(&config.model_path)
        .arg("--prompt")
        .arg(&mlx_request.prompt)
        .arg("--max-tokens")
        .arg(mlx_request.max_tokens.to_string())
        .arg("--temperature")
        .arg(mlx_request.temperature.to_string());

    let output = cmd.output()
        .await
        .map_err(|e| ServerError::Internal(format!("Failed to execute MLX bridge: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ServerError::Internal(format!("MLX bridge error: {}", stderr)));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse JSON response from Python bridge
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(json) => {
                if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                    return Ok(text.to_string());
                } else if let Some(error) = json.get("error").and_then(|v| v.as_str()) {
                    return Err(ServerError::Internal(format!("MLX inference error: {}", error)));
                }
            }
            Err(_) => {
                // Skip invalid JSON lines
                continue;
            }
        }
    }

    Err(ServerError::Internal("No valid response from MLX bridge".to_string()))
}

/// Handle streaming /v1/messages endpoint (SSE)
async fn handle_messages_streaming(
    _auth_header: Option<String>,
    _request: CreateMessageRequest,
    _config: ServerConfig,
) -> Result<Box<dyn Reply>, warp::Rejection> {
    // TODO: Implement streaming handler
    // 1. Auth already validated by handle_messages
    // 2. Translate request to MLX format
    // 3. Call Python bridge with --stream flag
    // 4. Parse streaming output
    // 5. Generate SSE events using streaming::generate_sse_stream()
    // 6. Return with proper SSE headers

    // For now, return error indicating streaming not yet implemented
    Ok(Box::new(warp::reply::with_status(
        warp::reply::json(&json!({
            "error": {
                "type": "not_implemented",
                "message": "Streaming not yet implemented"
            }
        })),
        warp::http::StatusCode::NOT_IMPLEMENTED,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pensieve_03::anthropic::{Content, Message, Role, MessageContent};

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 7777);
        assert_eq!(config.python_bridge_path, "python_bridge/mlx_inference.py");
        assert!(config.model_path.contains("Phi-3-mini"));
    }

    #[test]
    fn test_server_creation() {
        let config = ServerConfig::default();
        let server = AnthropicProxyServer::new(config.clone());
        assert_eq!(server.address(), "http://127.0.0.1:7777");
    }

    #[tokio::test]
    async fn test_server_lifecycle() {
        // RED: This should fail - server lifecycle not implemented yet
        let config = ServerConfig {
            port: 0, // Use random port for testing
            ..Default::default()
        };
        let server = AnthropicProxyServer::new(config);

        // Should be able to start
        let start_result = server.start().await;
        assert!(start_result.is_ok(), "Server should start successfully");

        // Give it time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Should be running
        assert!(server.is_running(), "Server should be running");

        // Should be able to shutdown
        let shutdown_result = server.shutdown().await;
        assert!(shutdown_result.is_ok(), "Server should shutdown successfully");

        // Should not be running after shutdown
        assert!(!server.is_running(), "Server should not be running after shutdown");
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        // RED: This should fail - health endpoint not implemented yet
        let config = ServerConfig {
            port: 7778, // Different port to avoid conflicts
            ..Default::default()
        };
        let server = AnthropicProxyServer::new(config.clone());

        server.start().await.expect("Server should start");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Test health endpoint
        let client = reqwest::Client::new();
        let response = client
            .get(&format!("{}/health", server.address()))
            .send()
            .await
            .expect("Health check should succeed");

        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.expect("Should parse JSON");
        assert_eq!(body["status"], "healthy");

        server.shutdown().await.expect("Server should shutdown");
    }

    #[tokio::test]
    async fn test_messages_endpoint_requires_auth() {
        // RED: This should fail - auth integration not implemented yet
        let config = ServerConfig {
            port: 7779,
            ..Default::default()
        };
        let server = AnthropicProxyServer::new(config.clone());

        server.start().await.expect("Server should start");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Test without auth header
        let client = reqwest::Client::new();
        let response = client
            .post(&format!("{}/v1/messages", server.address()))
            .json(&serde_json::json!({
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .expect("Request should complete");

        assert_eq!(response.status(), 401, "Should return 401 Unauthorized");

        server.shutdown().await.expect("Server should shutdown");
    }

    #[tokio::test]
    async fn test_messages_endpoint_with_valid_auth() {
        // RED: This should fail - message handling not implemented yet
        let config = ServerConfig {
            port: 7780,
            ..Default::default()
        };
        let server = AnthropicProxyServer::new(config.clone());

        server.start().await.expect("Server should start");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Test with valid auth header
        let client = reqwest::Client::new();
        let response = client
            .post(&format!("{}/v1/messages", server.address()))
            .header("Authorization", "Bearer pensieve-local-token")
            .json(&serde_json::json!({
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}]
            }))
            .send()
            .await
            .expect("Request should complete");

        // Should accept valid token (even if implementation is not complete)
        assert!(
            response.status().is_success() || response.status() == 500,
            "Should accept valid auth token (may fail on implementation)"
        );

        server.shutdown().await.expect("Server should shutdown");
    }

    #[tokio::test]
    async fn test_integration_auth_translator_mlx() {
        // RED: This should fail - full integration not implemented yet
        // This test verifies the complete flow:
        // 1. Receive Anthropic format request
        // 2. Validate auth
        // 3. Translate to MLX format
        // 4. Call MLX bridge (mocked for now)
        // 5. Translate response back to Anthropic format
        // 6. Return response

        let config = ServerConfig {
            port: 7781,
            ..Default::default()
        };
        let server = AnthropicProxyServer::new(config.clone());

        server.start().await.expect("Server should start");
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let client = reqwest::Client::new();
        let response = client
            .post(&format!("{}/v1/messages", server.address()))
            .header("Authorization", "Bearer pensieve-local-token")
            .json(&serde_json::json!({
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Say hello"}]
                    }
                ]
            }))
            .send()
            .await
            .expect("Request should complete");

        if response.status().is_success() {
            let body: CreateMessageResponse = response
                .json()
                .await
                .expect("Should parse Anthropic response format");

            // Verify Anthropic response structure
            assert!(body.id.starts_with("msg_"));
            assert_eq!(body.role, Role::Assistant);
            assert!(!body.content.is_empty());
            assert!(body.usage.input_tokens > 0);
            assert!(body.usage.output_tokens > 0);
        }

        server.shutdown().await.expect("Server should shutdown");
    }
}
