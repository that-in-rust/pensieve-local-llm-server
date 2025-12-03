//! # llama-server-client-streaming
//!
//! L2 Engine: HTTP client with SSE streaming for llama.cpp server.
//!
//! ## Executable Specification
//!
//! ### Preconditions
//! - llama-server is running at configured URL
//! - Model is loaded in llama-server
//!
//! ### Postconditions
//! - Streaming responses via Server-Sent Events (SSE)
//! - Configurable temperature and max_tokens per request
//! - Thread-safe client (Send + Sync)
//!
//! ### Performance Contract
//! - First token latency: <500ms (warm model)
//! - Throughput: 35-45 tok/s on Mac Mini M4 (Q4_K_M quantization)

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

pub use agent_role_definition_types::{
    AgentRole, AggregatorConfig, ProposerConfig, AGGREGATOR_MAX_TOKENS, PROPOSER_MAX_TOKENS,
};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default llama-server URL
pub const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8080";

/// Default request timeout
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Default max retries
pub const DEFAULT_MAX_RETRIES: u8 = 2;

/// SSE data prefix
const SSE_DATA_PREFIX: &str = "data: ";

/// SSE done marker
const SSE_DONE_MARKER: &str = "[DONE]";

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors from LLM client operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LlmClientError {
    /// Connection to server failed
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Request timed out
    #[error("Request timed out after {0} seconds")]
    Timeout(u64),

    /// Server returned error
    #[error("Server error: {status} - {message}")]
    ServerError {
        /// HTTP status code
        status: u16,
        /// Error message
        message: String,
    },

    /// Response parsing failed
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid configuration
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Streaming error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Max retries exceeded
    #[error("Max retries exceeded after {0} attempts")]
    MaxRetriesExceeded(u8),
}

// ============================================================================
// REQUEST/RESPONSE TYPES
// ============================================================================

/// Request to llama-server completion endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The prompt to complete
    pub prompt: String,
    /// Maximum tokens to generate
    pub n_predict: usize,
    /// Temperature for sampling (0.0 - 2.0)
    pub temperature: f32,
    /// Whether to stream response
    pub stream: bool,
    /// Stop sequences (optional)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
}

impl CompletionRequest {
    /// Create request for proposer generation.
    #[must_use]
    pub fn create_request_for_proposer(prompt: String, config: &ProposerConfig) -> Self {
        Self {
            prompt,
            n_predict: config.max_tokens(),
            temperature: config.temperature(),
            stream: true,
            stop: vec![],
        }
    }

    /// Create request for aggregator generation.
    #[must_use]
    pub fn create_request_for_aggregator(prompt: String, config: &AggregatorConfig) -> Self {
        Self {
            prompt,
            n_predict: config.max_tokens(),
            temperature: config.temperature(),
            stream: true,
            stop: vec![],
        }
    }

    /// Create request with explicit parameters.
    #[must_use]
    pub fn create_request_with_params(
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        stream: bool,
    ) -> Self {
        Self {
            prompt,
            n_predict: max_tokens,
            temperature: temperature.clamp(0.0, 2.0),
            stream,
            stop: vec![],
        }
    }
}

/// Streaming response chunk from llama-server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    /// Content of this chunk
    pub content: String,
    /// Whether this is the final chunk
    pub stop: bool,
    /// Number of tokens generated so far (optional)
    #[serde(default)]
    pub tokens_predicted: usize,
    /// Model name (optional)
    #[serde(default)]
    pub model: String,
}

/// Complete response from llama-server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Full generated content
    pub content: String,
    /// Total tokens generated
    pub tokens_predicted: usize,
    /// Total tokens in prompt
    pub tokens_evaluated: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f32,
}

impl CompletionResponse {
    /// Create response from content and stats.
    #[must_use]
    pub fn create_response_from_content(
        content: String,
        tokens_predicted: usize,
        generation_time_ms: u64,
    ) -> Self {
        let tokens_per_second = if generation_time_ms > 0 {
            tokens_predicted as f32 / (generation_time_ms as f32 / 1000.0)
        } else {
            0.0
        };

        Self {
            content,
            tokens_predicted,
            tokens_evaluated: 0,
            generation_time_ms,
            tokens_per_second,
        }
    }
}

// ============================================================================
// CLIENT CONFIGURATION
// ============================================================================

/// Configuration for LLM client.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmClientConfig {
    /// Server base URL
    pub server_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on failure
    pub max_retries: u8,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
}

impl LlmClientConfig {
    /// Create config with default values.
    #[must_use]
    pub fn create_config_with_defaults() -> Self {
        Self {
            server_url: DEFAULT_SERVER_URL.to_string(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            max_retries: DEFAULT_MAX_RETRIES,
            retry_delay_ms: 500,
        }
    }

    /// Create config with custom URL.
    #[must_use]
    pub fn create_config_with_url(server_url: String) -> Self {
        Self {
            server_url,
            ..Self::create_config_with_defaults()
        }
    }

    /// Get completion endpoint URL.
    #[must_use]
    pub fn get_completion_endpoint_url(&self) -> String {
        format!("{}/completion", self.server_url.trim_end_matches('/'))
    }

    /// Get health endpoint URL.
    #[must_use]
    pub fn get_health_endpoint_url(&self) -> String {
        format!("{}/health", self.server_url.trim_end_matches('/'))
    }
}

impl Default for LlmClientConfig {
    fn default() -> Self {
        Self::create_config_with_defaults()
    }
}

// ============================================================================
// LLM CLIENT TRAIT
// ============================================================================

/// Type alias for streaming response.
pub type StreamResult = Pin<Box<dyn Stream<Item = Result<StreamingChunk, LlmClientError>> + Send>>;

/// Trait for LLM client implementations.
///
/// All implementations must be thread-safe (Send + Sync).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Generate completion (blocking/full response).
    ///
    /// # Preconditions
    /// - `prompt` is non-empty
    /// - `max_tokens` > 0
    ///
    /// # Postconditions
    /// - Returns complete generated text
    async fn generate_complete_response_blocking(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, LlmClientError>;

    /// Generate completion with streaming.
    ///
    /// # Postconditions
    /// - Returns stream of chunks
    async fn generate_streaming_response_chunks(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<StreamResult, LlmClientError>;

    /// Check server health.
    ///
    /// # Postconditions
    /// - Returns `true` if server is healthy
    async fn check_server_health_status(&self) -> Result<bool, LlmClientError>;

    /// Get client configuration.
    fn get_client_configuration_current(&self) -> &LlmClientConfig;
}

// ============================================================================
// LLAMA SERVER CLIENT IMPLEMENTATION
// ============================================================================

/// HTTP client for llama.cpp server.
///
/// Uses reqwest for HTTP requests and supports SSE streaming.
#[derive(Debug, Clone)]
pub struct LlamaServerClient {
    config: LlmClientConfig,
    http_client: reqwest::Client,
}

impl LlamaServerClient {
    /// Create client with default configuration.
    ///
    /// # Errors
    /// Returns `LlmClientError::InvalidConfig` if client creation fails
    pub fn create_client_with_defaults() -> Result<Self, LlmClientError> {
        Self::create_client_with_config(LlmClientConfig::create_config_with_defaults())
    }

    /// Create client with custom configuration.
    ///
    /// # Errors
    /// Returns `LlmClientError::InvalidConfig` if configuration is invalid
    pub fn create_client_with_config(config: LlmClientConfig) -> Result<Self, LlmClientError> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LlmClientError::InvalidConfig(e.to_string()))?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Create client with custom URL.
    ///
    /// # Errors
    /// Returns `LlmClientError::InvalidConfig` if URL is invalid
    pub fn create_client_with_url(server_url: &str) -> Result<Self, LlmClientError> {
        Self::create_client_with_config(LlmClientConfig::create_config_with_url(
            server_url.to_string(),
        ))
    }

    /// Send completion request and collect full response.
    #[instrument(skip(self, request), fields(prompt_len = request.prompt.len()))]
    async fn send_completion_request_full(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, LlmClientError> {
        let start = std::time::Instant::now();

        // For blocking mode, disable streaming
        let mut blocking_request = request.clone();
        blocking_request.stream = false;

        let url = self.config.get_completion_endpoint_url();
        debug!(%url, "Sending completion request");

        let response = self
            .http_client
            .post(&url)
            .json(&blocking_request)
            .send()
            .await
            .map_err(|e| {
                error!(%e, "HTTP request failed");
                if e.is_timeout() {
                    LlmClientError::Timeout(self.config.timeout_secs)
                } else if e.is_connect() {
                    LlmClientError::ConnectionFailed(e.to_string())
                } else {
                    LlmClientError::ConnectionFailed(e.to_string())
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            error!(%status, body = %error_body, "Server returned error");
            return Err(LlmClientError::ServerError {
                status: status.as_u16(),
                message: error_body,
            });
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmClientError::ParseError(e.to_string()))?;

        let content = body["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();

        let tokens_predicted = body["tokens_predicted"]
            .as_u64()
            .unwrap_or(0) as usize;

        let generation_time_ms = start.elapsed().as_millis() as u64;

        info!(
            tokens = tokens_predicted,
            time_ms = generation_time_ms,
            "Completion finished"
        );

        Ok(CompletionResponse::create_response_from_content(
            content,
            tokens_predicted,
            generation_time_ms,
        ))
    }

    /// Execute request with retry logic.
    async fn execute_with_retry_logic<T, F, Fut>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> Result<T, LlmClientError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, LlmClientError>>,
    {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                warn!(
                    attempt,
                    max = self.config.max_retries,
                    operation = operation_name,
                    "Retrying operation"
                );
                tokio::time::sleep(Duration::from_millis(
                    self.config.retry_delay_ms * (attempt as u64),
                ))
                .await;
            }

            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Don't retry on certain errors
                    if matches!(e, LlmClientError::InvalidConfig(_)) {
                        return Err(e);
                    }
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(LlmClientError::MaxRetriesExceeded(self.config.max_retries)))
    }
}

#[async_trait]
impl LlmClient for LlamaServerClient {
    #[instrument(skip(self), fields(prompt_len = prompt.len(), max_tokens, temperature))]
    async fn generate_complete_response_blocking(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<String, LlmClientError> {
        if prompt.is_empty() {
            return Err(LlmClientError::InvalidConfig("Prompt cannot be empty".into()));
        }

        let request = CompletionRequest::create_request_with_params(
            prompt.to_string(),
            max_tokens,
            temperature,
            false,
        );

        let response = self
            .execute_with_retry_logic("completion", || {
                self.send_completion_request_full(&request)
            })
            .await?;

        Ok(response.content)
    }

    async fn generate_streaming_response_chunks(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<StreamResult, LlmClientError> {
        if prompt.is_empty() {
            return Err(LlmClientError::InvalidConfig("Prompt cannot be empty".into()));
        }

        let request = CompletionRequest::create_request_with_params(
            prompt.to_string(),
            max_tokens,
            temperature,
            true,
        );

        let url = self.config.get_completion_endpoint_url();

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    LlmClientError::Timeout(self.config.timeout_secs)
                } else {
                    LlmClientError::ConnectionFailed(e.to_string())
                }
            })?;

        if !response.status().is_success() {
            return Err(LlmClientError::ServerError {
                status: response.status().as_u16(),
                message: "Streaming request failed".to_string(),
            });
        }

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut byte_stream = response.bytes_stream();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result.map_err(|e| LlmClientError::StreamError(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                // Parse SSE lines
                for line in text.lines() {
                    if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
                        let data = data.trim();
                        if data == SSE_DONE_MARKER || data.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<StreamingChunk>(data) {
                            Ok(chunk) => yield chunk,
                            Err(e) => {
                                warn!(%e, data, "Failed to parse SSE chunk");
                                // Try to extract content manually
                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                                    if let Some(content) = json["content"].as_str() {
                                        yield StreamingChunk {
                                            content: content.to_string(),
                                            stop: json["stop"].as_bool().unwrap_or(false),
                                            tokens_predicted: 0,
                                            model: String::new(),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn check_server_health_status(&self) -> Result<bool, LlmClientError> {
        let url = self.config.get_health_endpoint_url();

        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| LlmClientError::ConnectionFailed(e.to_string()))?;

        Ok(response.status().is_success())
    }

    fn get_client_configuration_current(&self) -> &LlmClientConfig {
        &self.config
    }
}

// ============================================================================
// MOCK CLIENT FOR TESTING
// ============================================================================

/// Mock LLM client for testing.
#[derive(Debug, Clone, Default)]
pub struct MockLlmClient {
    config: LlmClientConfig,
    /// Response to return
    pub mock_response: String,
    /// Whether to simulate failure
    pub should_fail: bool,
    /// Simulated latency in ms
    pub latency_ms: u64,
}

impl MockLlmClient {
    /// Create mock client for testing.
    #[must_use]
    pub fn create_mock_for_testing(response: &str) -> Self {
        Self {
            config: LlmClientConfig::create_config_with_defaults(),
            mock_response: response.to_string(),
            should_fail: false,
            latency_ms: 0,
        }
    }

    /// Create mock client that fails.
    #[must_use]
    pub fn create_mock_that_fails() -> Self {
        Self {
            config: LlmClientConfig::create_config_with_defaults(),
            mock_response: String::new(),
            should_fail: true,
            latency_ms: 0,
        }
    }

    /// Set simulated latency.
    #[must_use]
    pub fn with_latency_milliseconds(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn generate_complete_response_blocking(
        &self,
        _prompt: &str,
        _max_tokens: usize,
        _temperature: f32,
    ) -> Result<String, LlmClientError> {
        if self.latency_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
        }

        if self.should_fail {
            return Err(LlmClientError::ConnectionFailed("Mock failure".into()));
        }

        Ok(self.mock_response.clone())
    }

    async fn generate_streaming_response_chunks(
        &self,
        _prompt: &str,
        _max_tokens: usize,
        _temperature: f32,
    ) -> Result<StreamResult, LlmClientError> {
        if self.should_fail {
            return Err(LlmClientError::ConnectionFailed("Mock failure".into()));
        }

        let response = self.mock_response.clone();
        let latency = self.latency_ms;

        let stream = async_stream::try_stream! {
            if latency > 0 {
                tokio::time::sleep(Duration::from_millis(latency)).await;
            }

            // Split response into chunks
            for (i, word) in response.split_whitespace().enumerate() {
                yield StreamingChunk {
                    content: format!("{} ", word),
                    stop: false,
                    tokens_predicted: i + 1,
                    model: "mock".to_string(),
                };
            }

            // Final chunk
            yield StreamingChunk {
                content: String::new(),
                stop: true,
                tokens_predicted: response.split_whitespace().count(),
                model: "mock".to_string(),
            };
        };

        Ok(Box::pin(stream))
    }

    async fn check_server_health_status(&self) -> Result<bool, LlmClientError> {
        if self.should_fail {
            return Err(LlmClientError::ConnectionFailed("Mock failure".into()));
        }
        Ok(true)
    }

    fn get_client_configuration_current(&self) -> &LlmClientConfig {
        &self.config
    }
}

// ============================================================================
// TESTS (TDD-First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    // -------------------------------------------------------------------------
    // Configuration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let config = LlmClientConfig::create_config_with_defaults();

        assert_eq!(config.server_url, DEFAULT_SERVER_URL);
        assert_eq!(config.timeout_secs, DEFAULT_TIMEOUT_SECS);
        assert_eq!(config.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_config_custom_url() {
        let config = LlmClientConfig::create_config_with_url("http://localhost:9000".to_string());

        assert_eq!(config.server_url, "http://localhost:9000");
    }

    #[test]
    fn test_config_endpoint_urls() {
        let config = LlmClientConfig::create_config_with_defaults();

        assert_eq!(
            config.get_completion_endpoint_url(),
            "http://127.0.0.1:8080/completion"
        );
        assert_eq!(
            config.get_health_endpoint_url(),
            "http://127.0.0.1:8080/health"
        );
    }

    #[test]
    fn test_config_endpoint_url_trailing_slash() {
        let config = LlmClientConfig::create_config_with_url("http://localhost:8080/".to_string());

        assert_eq!(
            config.get_completion_endpoint_url(),
            "http://localhost:8080/completion"
        );
    }

    // -------------------------------------------------------------------------
    // CompletionRequest Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_completion_request_for_proposer() {
        let config = ProposerConfig::create_config_with_index(0);
        let request =
            CompletionRequest::create_request_for_proposer("Test prompt".to_string(), &config);

        assert_eq!(request.prompt, "Test prompt");
        assert_eq!(request.n_predict, config.max_tokens());
        assert!((request.temperature - config.temperature()).abs() < f32::EPSILON);
        assert!(request.stream);
    }

    #[test]
    fn test_completion_request_for_aggregator() {
        let config = AggregatorConfig::create_config_default_aggregator();
        let request =
            CompletionRequest::create_request_for_aggregator("Test prompt".to_string(), &config);

        assert_eq!(request.n_predict, config.max_tokens());
        assert!((request.temperature - config.temperature()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_completion_request_with_params() {
        let request = CompletionRequest::create_request_with_params(
            "Test".to_string(),
            100,
            0.7,
            false,
        );

        assert_eq!(request.n_predict, 100);
        assert!((request.temperature - 0.7).abs() < f32::EPSILON);
        assert!(!request.stream);
    }

    #[test]
    fn test_completion_request_clamps_temperature() {
        let request1 = CompletionRequest::create_request_with_params(
            "Test".to_string(),
            100,
            5.0, // Above max
            false,
        );
        assert!((request1.temperature - 2.0).abs() < f32::EPSILON);

        let request2 = CompletionRequest::create_request_with_params(
            "Test".to_string(),
            100,
            -1.0, // Below min
            false,
        );
        assert!((request2.temperature - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_completion_request_serialization() {
        let request = CompletionRequest::create_request_with_params(
            "Test".to_string(),
            100,
            0.7,
            true,
        );

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"prompt\":\"Test\""));
        assert!(json.contains("\"n_predict\":100"));
        assert!(json.contains("\"stream\":true"));
    }

    // -------------------------------------------------------------------------
    // CompletionResponse Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_completion_response_creation() {
        let response = CompletionResponse::create_response_from_content(
            "Generated text".to_string(),
            50,
            1000,
        );

        assert_eq!(response.content, "Generated text");
        assert_eq!(response.tokens_predicted, 50);
        assert_eq!(response.generation_time_ms, 1000);
        assert!((response.tokens_per_second - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_completion_response_zero_time() {
        let response = CompletionResponse::create_response_from_content(
            "Text".to_string(),
            10,
            0, // Zero time
        );

        assert!((response.tokens_per_second - 0.0).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // StreamingChunk Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_streaming_chunk_serialization() {
        let chunk = StreamingChunk {
            content: "Hello ".to_string(),
            stop: false,
            tokens_predicted: 1,
            model: "test".to_string(),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: StreamingChunk = serde_json::from_str(&json).unwrap();

        assert_eq!(chunk.content, deserialized.content);
        assert_eq!(chunk.stop, deserialized.stop);
    }

    // -------------------------------------------------------------------------
    // MockLlmClient Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_mock_client_blocking() {
        let client = MockLlmClient::create_mock_for_testing("Test response");

        let result = client
            .generate_complete_response_blocking("prompt", 100, 0.7)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Test response");
    }

    #[tokio::test]
    async fn test_mock_client_failure() {
        let client = MockLlmClient::create_mock_that_fails();

        let result = client
            .generate_complete_response_blocking("prompt", 100, 0.7)
            .await;

        assert!(result.is_err());
        assert!(matches!(result, Err(LlmClientError::ConnectionFailed(_))));
    }

    #[tokio::test]
    async fn test_mock_client_streaming() {
        let client = MockLlmClient::create_mock_for_testing("Hello world test");

        let stream = client
            .generate_streaming_response_chunks("prompt", 100, 0.7)
            .await
            .unwrap();

        let chunks: Vec<_> = stream.collect().await;

        // Should have chunks for each word + final chunk
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.is_ok()));

        // Last chunk should have stop=true
        let last = chunks.last().unwrap().as_ref().unwrap();
        assert!(last.stop);
    }

    #[tokio::test]
    async fn test_mock_client_health() {
        let client = MockLlmClient::create_mock_for_testing("");

        let health = client.check_server_health_status().await;
        assert!(health.is_ok());
        assert!(health.unwrap());
    }

    #[tokio::test]
    async fn test_mock_client_latency() {
        let client = MockLlmClient::create_mock_for_testing("Test")
            .with_latency_milliseconds(100);

        let start = std::time::Instant::now();
        let _ = client
            .generate_complete_response_blocking("prompt", 100, 0.7)
            .await;
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() >= 100);
    }

    // -------------------------------------------------------------------------
    // Client Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_client_creation_defaults() {
        let result = LlamaServerClient::create_client_with_defaults();
        assert!(result.is_ok());

        let client = result.unwrap();
        assert_eq!(
            client.get_client_configuration_current().server_url,
            DEFAULT_SERVER_URL
        );
    }

    #[test]
    fn test_client_creation_custom_url() {
        let result = LlamaServerClient::create_client_with_url("http://custom:9000");
        assert!(result.is_ok());

        let client = result.unwrap();
        assert_eq!(
            client.get_client_configuration_current().server_url,
            "http://custom:9000"
        );
    }

    // -------------------------------------------------------------------------
    // Error Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let errors = vec![
            LlmClientError::ConnectionFailed("test".into()),
            LlmClientError::Timeout(30),
            LlmClientError::ServerError {
                status: 500,
                message: "Internal error".into(),
            },
            LlmClientError::ParseError("Invalid JSON".into()),
            LlmClientError::InvalidConfig("Bad config".into()),
            LlmClientError::StreamError("Stream closed".into()),
            LlmClientError::MaxRetriesExceeded(3),
        ];

        for error in errors {
            let display = format!("{error}");
            assert!(!display.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_match_architecture() {
        assert_eq!(DEFAULT_SERVER_URL, "http://127.0.0.1:8080");
        assert_eq!(DEFAULT_TIMEOUT_SECS, 60);
        assert_eq!(DEFAULT_MAX_RETRIES, 2);
    }
}
