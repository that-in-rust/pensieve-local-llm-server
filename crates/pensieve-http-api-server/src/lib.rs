//! # pensieve-http-api-server
//!
//! L3 Application: Axum HTTP API server for MoA-Lite debate system.
//!
//! ## Executable Specification
//!
//! ### Preconditions
//! - llama-server running at configured URL (or use mock mode)
//! - Port available for binding
//!
//! ### Postconditions
//! - HTTP server listening on configured port
//! - `/v1/chat/completions` endpoint accepts OpenAI-compatible requests
//! - `/health` endpoint returns server status
//!
//! ### API Endpoints
//! - `POST /v1/chat/completions` - Chat completion with debate
//! - `GET /health` - Health check
//! - `GET /metrics` - Performance metrics

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info, instrument};
use uuid::Uuid;

// Re-export dependencies
pub use debate_orchestrator_state_machine::{
    create_test_orchestrator_mock, DebateOrchestrator, DebateResult, HeuristicRouter,
    InMemoryBlackboard, MoaLiteOrchestrator, MockLlmClient, OrchestratorError, ResponseSource,
};
pub use llama_server_client_streaming::{LlamaServerClient, LlmClient, LlmClientConfig};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default server port
pub const DEFAULT_SERVER_PORT: u16 = 3000;

/// Default llama-server URL
pub const DEFAULT_LLAMA_SERVER_URL: &str = "http://127.0.0.1:8080";

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Server errors.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Orchestrator error
    #[error("Orchestrator error: {0}")]
    OrchestratorError(#[from] OrchestratorError),

    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Internal server error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            Self::InvalidRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            Self::OrchestratorError(e) => {
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            }
            Self::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: "server_error".to_string(),
            },
        });

        (status, body).into_response()
    }
}

// ============================================================================
// REQUEST/RESPONSE TYPES (OpenAI-Compatible)
// ============================================================================

/// Chat completion request (OpenAI-compatible).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model to use (ignored - always uses MoA-Lite)
    #[serde(default)]
    pub model: String,
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// Temperature (optional, uses role defaults)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Max tokens (optional)
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Whether to stream (not yet supported)
    #[serde(default)]
    pub stream: bool,
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: system, user, or assistant
    pub role: String,
    /// Message content
    pub content: String,
}

/// Chat completion response (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Response choices
    pub choices: Vec<ChatChoice>,
    /// Usage statistics
    pub usage: UsageStats,
    /// Pensieve-specific metadata
    pub pensieve_metadata: PensieveMetadata,
}

/// Chat choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: usize,
    /// Message content
    pub message: ChatMessage,
    /// Finish reason
    pub finish_reason: String,
}

/// Usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Prompt tokens (estimated)
    pub prompt_tokens: usize,
    /// Completion tokens (estimated)
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// Pensieve-specific metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PensieveMetadata {
    /// Response source (LocalDebate, PartialDebate, CloudHandoff)
    pub source: String,
    /// Number of proposals used
    pub proposal_count: u8,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Conversation ID
    pub conversation_id: String,
}

/// Error response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetail,
}

/// Error detail.
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
}

/// Health check response.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Server status
    pub status: String,
    /// Server version
    pub version: String,
    /// LLM backend status
    pub llm_backend: String,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Total requests processed
    pub total_requests: u64,
}

// ============================================================================
// SERVER STATE
// ============================================================================

/// Shared server state.
pub struct ServerState<O>
where
    O: DebateOrchestrator + 'static,
{
    /// The debate orchestrator
    orchestrator: Arc<O>,
    /// Start time for uptime calculation
    start_time: Instant,
    /// Request counter
    request_count: RwLock<u64>,
    /// LLM backend URL
    llm_backend_url: String,
}

impl<O> ServerState<O>
where
    O: DebateOrchestrator + 'static,
{
    /// Create server state with orchestrator.
    #[must_use]
    pub fn create_state_with_orchestrator(orchestrator: Arc<O>, llm_backend_url: String) -> Self {
        Self {
            orchestrator,
            start_time: Instant::now(),
            request_count: RwLock::new(0),
            llm_backend_url,
        }
    }

    /// Increment request counter.
    async fn increment_request_counter_async(&self) {
        let mut count = self.request_count.write().await;
        *count += 1;
    }

    /// Get current request count.
    async fn get_request_count_current(&self) -> u64 {
        *self.request_count.read().await
    }
}

// ============================================================================
// SERVER CONFIGURATION
// ============================================================================

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// llama-server URL
    pub llm_backend_url: String,
    /// Use mock LLM (for testing)
    pub use_mock_llm: bool,
    /// Mock response (if using mock)
    pub mock_response: String,
}

impl ServerConfig {
    /// Create config with default values.
    #[must_use]
    pub fn create_config_with_defaults() -> Self {
        Self {
            port: DEFAULT_SERVER_PORT,
            host: "0.0.0.0".to_string(),
            llm_backend_url: DEFAULT_LLAMA_SERVER_URL.to_string(),
            use_mock_llm: false,
            mock_response: String::new(),
        }
    }

    /// Create config for mock mode.
    #[must_use]
    pub fn create_config_for_mock(mock_response: String) -> Self {
        Self {
            port: DEFAULT_SERVER_PORT,
            host: "0.0.0.0".to_string(),
            llm_backend_url: "mock://localhost".to_string(),
            use_mock_llm: true,
            mock_response,
        }
    }

    /// Create config with port.
    #[must_use]
    pub fn create_config_with_port(port: u16) -> Self {
        Self {
            port,
            ..Self::create_config_with_defaults()
        }
    }

    /// Get socket address.
    #[must_use]
    pub fn get_socket_address_parsed(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("Invalid socket address")
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::create_config_with_defaults()
    }
}

// ============================================================================
// ROUTE HANDLERS
// ============================================================================

/// Health check handler.
#[instrument(skip(state))]
async fn handle_health_check_request<O>(
    State(state): State<Arc<ServerState<O>>>,
) -> Json<HealthResponse>
where
    O: DebateOrchestrator + 'static,
{
    let uptime = state.start_time.elapsed().as_secs();
    let requests = state.get_request_count_current().await;

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        llm_backend: state.llm_backend_url.clone(),
        uptime_seconds: uptime,
        total_requests: requests,
    })
}

/// Chat completions handler.
#[instrument(skip(state, request), fields(messages_count = request.messages.len()))]
async fn handle_chat_completion_request<O>(
    State(state): State<Arc<ServerState<O>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ServerError>
where
    O: DebateOrchestrator + 'static,
{
    state.increment_request_counter_async().await;

    // Validate request
    if request.messages.is_empty() {
        return Err(ServerError::InvalidRequest("Messages cannot be empty".into()));
    }

    // Extract the last user message as the query
    let query = request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .ok_or_else(|| ServerError::InvalidRequest("No user message found".into()))?;

    info!(query_len = query.len(), "Processing chat completion");

    // Process through debate orchestrator
    let result = state
        .orchestrator
        .process_query_through_debate(&query)
        .await?;

    // Build response
    let response = build_completion_response_from_result(&result, &query);

    info!(
        latency_ms = result.latency_ms,
        source = ?result.source,
        "Chat completion finished"
    );

    Ok(Json(response))
}

/// Build completion response from debate result.
fn build_completion_response_from_result(result: &DebateResult, query: &str) -> ChatCompletionResponse {
    let prompt_tokens = query.len() / 4;
    let completion_tokens = result.response.len() / 4;

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: "pensieve-moa-lite".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: result.response.clone(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        pensieve_metadata: PensieveMetadata {
            source: format!("{:?}", result.source),
            proposal_count: result.proposal_count,
            latency_ms: result.latency_ms,
            conversation_id: result.conversation_id.to_string(),
        },
    }
}

// ============================================================================
// SERVER BUILDER
// ============================================================================

/// Build router with state.
pub fn build_router_with_state<O>(state: Arc<ServerState<O>>) -> Router
where
    O: DebateOrchestrator + Send + Sync + 'static,
{
    Router::new()
        .route("/health", get(handle_health_check_request::<O>))
        .route("/v1/chat/completions", post(handle_chat_completion_request::<O>))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Create mock server for testing.
#[must_use]
pub fn create_mock_server_for_testing(
    mock_response: &str,
) -> Router {
    let llm_client = Arc::new(MockLlmClient::create_mock_for_testing(mock_response));
    let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
    let router = Arc::new(HeuristicRouter::create_router_with_defaults());

    let orchestrator = Arc::new(MoaLiteOrchestrator::create_orchestrator_with_components(
        llm_client,
        blackboard,
        router,
    ));

    let state = Arc::new(ServerState::create_state_with_orchestrator(
        orchestrator,
        "mock://localhost".to_string(),
    ));

    build_router_with_state(state)
}

/// Start server with configuration.
///
/// # Errors
/// Returns error if server fails to start
pub async fn start_server_with_config(config: ServerConfig) -> Result<(), anyhow::Error> {
    let addr = config.get_socket_address_parsed();

    info!(%addr, "Starting Pensieve HTTP API server");

    let router = if config.use_mock_llm {
        info!("Using mock LLM backend");
        create_mock_server_for_testing(&config.mock_response)
    } else {
        // Create real LLM client
        let llm_config = LlmClientConfig::create_config_with_url(config.llm_backend_url.clone());
        let llm_client = Arc::new(
            LlamaServerClient::create_client_with_config(llm_config)
                .map_err(|e| anyhow::anyhow!("Failed to create LLM client: {}", e))?,
        );
        let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
        let router_component = Arc::new(HeuristicRouter::create_router_with_defaults());

        let orchestrator = Arc::new(MoaLiteOrchestrator::create_orchestrator_with_components(
            llm_client,
            blackboard,
            router_component,
        ));

        let state = Arc::new(ServerState::create_state_with_orchestrator(
            orchestrator,
            config.llm_backend_url,
        ));

        build_router_with_state(state)
    };

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Pensieve server listening on http://{}", addr);

    axum::serve(listener, router).await?;

    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn create_test_router_with_mock() -> Router {
        create_mock_server_for_testing("Test response from MoA-Lite debate system.")
    }

    #[tokio::test]
    async fn test_health_endpoint_returns_ok() {
        let app = create_test_router_with_mock();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(health.status, "healthy");
    }

    #[tokio::test]
    async fn test_chat_completion_success() {
        let app = create_test_router_with_mock();

        let request_body = serde_json::json!({
            "model": "pensieve",
            "messages": [
                {"role": "user", "content": "Who is the Home Minister of India?"}
            ]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let completion: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(completion.object, "chat.completion");
        assert!(!completion.choices.is_empty());
        assert!(!completion.choices[0].message.content.is_empty());
        assert!(completion.pensieve_metadata.proposal_count >= 2);
    }

    #[tokio::test]
    async fn test_chat_completion_empty_messages() {
        let app = create_test_router_with_mock();

        let request_body = serde_json::json!({
            "messages": []
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_chat_completion_no_user_message() {
        let app = create_test_router_with_mock();

        let request_body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful"}
            ]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::create_config_with_defaults();

        assert_eq!(config.port, DEFAULT_SERVER_PORT);
        assert_eq!(config.llm_backend_url, DEFAULT_LLAMA_SERVER_URL);
        assert!(!config.use_mock_llm);
    }

    #[test]
    fn test_server_config_mock() {
        let config = ServerConfig::create_config_for_mock("test response".to_string());

        assert!(config.use_mock_llm);
        assert_eq!(config.mock_response, "test response");
    }

    #[test]
    fn test_build_completion_response() {
        let result = DebateResult::create_result_from_local(
            "Test response".to_string(),
            3,
            15000,
            Uuid::new_v4(),
        );

        let response = build_completion_response_from_result(&result, "Test query");

        assert_eq!(response.model, "pensieve-moa-lite");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.role, "assistant");
        assert_eq!(response.choices[0].message.content, "Test response");
        assert_eq!(response.pensieve_metadata.proposal_count, 3);
    }
}
