//! Pensieve Claude Core - Claude-specific traits and dependency injection
//!
//! This crate provides the core abstractions for Claude Code Local LLM:
//! - Claude-specific inference traits
//! - Dependency injection containers
//! - Test specification framework
//! - Performance measurement interfaces
//!
//! # Architecture Principles
//!
//! ## Layer 2 (L2) - Standard Library Dependencies
//! - Uses std library features
//! - Depends on pensieve-07_core (L1)
//! - Provides testable abstractions for L3
//!
//! ## Dependency Injection
//! Every component depends on traits, not concrete types.
//! This enables comprehensive testing and modularity.
//!
//! ## Test-First Design
//! All traits are designed with executable specifications.
//! Tests are written first (STUB → RED → GREEN → REFACTOR).

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

// Core re-exports from pensieve-07
pub use pensieve_07_core::{CoreError, CoreResult, Resource, Reset, Validate};

/// Claude-specific error types
pub mod error {
    use super::CoreError;
    use thiserror::Error;

    /// Claude result type
    pub type ClaudeResult<T> = core::result::Result<T, ClaudeError>;

    /// Comprehensive error types for Claude operations
    #[derive(Error, Debug)]
    pub enum ClaudeError {
        #[error("Model loading failed: {source}")]
        ModelLoadError {
            #[from]
            source: ModelLoadError,
        },

        #[error("Inference failed: {source}")]
        InferenceError {
            #[from]
            source: InferenceError,
        },

        #[error("Resource management failed: {source}")]
        ResourceError {
            #[from]
            source: ResourceError,
        },

        #[error("Configuration error: {message}")]
        Configuration { message: String },

        #[error("Performance constraint violated: {constraint}")]
        PerformanceConstraint { constraint: String },

        #[error("Test assertion failed: {assertion}")]
        TestAssertion { assertion: String },

        #[error("Core error: {0}")]
        Core(#[from] CoreError),
    }

    /// Model loading specific errors
    #[derive(Error, Debug)]
    pub enum ModelLoadError {
        #[error("Model file not found: {path}")]
        FileNotFound { path: String },

        #[error("Invalid model format: {format}")]
        InvalidFormat { format: String },

        #[error("Model size exceeds limit: {size_mb}MB > {limit_mb}MB")]
        SizeExceeded { size_mb: u64, limit_mb: u64 },

        #[error("Model validation failed: {reason}")]
        ValidationFailed { reason: String },

        #[error("Model architecture not supported: {architecture}")]
        UnsupportedArchitecture { architecture: String },
    }

    /// Inference specific errors
    #[derive(Error, Debug)]
    pub enum InferenceError {
        #[error("Model not loaded")]
        ModelNotLoaded,

        #[error("Invalid input: {reason}")]
        InvalidInput { reason: String },

        #[error("Context length exceeded: {tokens} > {max_tokens}")]
        ContextExceeded { tokens: usize, max_tokens: usize },

        #[error("Generation failed: {reason}")]
        GenerationFailed { reason: String },

        #[error("Streaming interrupted")]
        StreamingInterrupted,

        #[error("Timeout occurred: {timeout_ms}ms")]
        Timeout { timeout_ms: u64 },
    }

    /// Resource management specific errors
    #[derive(Error, Debug)]
    pub enum ResourceError {
        #[error("Memory allocation failed: {requested_mb}MB")]
        MemoryAllocationFailed { requested_mb: u64 },

        #[error("Resource pool exhausted: {resource_type}")]
        PoolExhausted { resource_type: String },

        #[error("Resource not available: {resource_type}")]
        ResourceUnavailable { resource_type: String },

        #[error("Resource cleanup failed: {resource_type}")]
        CleanupFailed { resource_type: String },
    }
}

/// Core Claude traits for dependency injection
pub mod traits {
    use super::error::ClaudeResult;
    use super::types::{
        GenerationConfig, HealthStatus, MeasurementSession, ModelInfo, PerformanceConstraints,
        ResourceLimits, StreamingToken, ValidationResult,
    };
    use async_trait::async_trait;
    use futures::{Stream, StreamExt};
    use serde::{Deserialize, Serialize};

    /// Trait for model loading and management
    #[async_trait]
    pub trait ModelManager: Send + Sync {
        /// Model type identifier
        type Model: Send + Sync;

        /// Load a model from the given path
        async fn load_model(&mut self, path: &str) -> ClaudeResult<Self::Model>;

        /// Unload current model
        async fn unload_model(&mut self) -> ClaudeResult<()>;

        /// Check if model is loaded
        fn is_model_loaded(&self) -> bool;

        /// Get model information
        fn model_info(&self) -> Option<ModelInfo>;
    }

    /// Trait for streaming inference
    #[async_trait]
    pub trait InferenceEngine: Send + Sync {
        /// Token type for streaming
        type Token: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

        /// Stream of tokens
        type TokenStream: Stream<Item = ClaudeResult<StreamingToken<Self::Token>>> + Send + Unpin;

        /// Generate tokens with streaming
        async fn generate_stream(
            &self,
            input: &str,
            config: GenerationConfig,
        ) -> ClaudeResult<Self::TokenStream>;

        /// Generate complete response (non-streaming)
        async fn generate(
            &self,
            input: &str,
            config: GenerationConfig,
        ) -> ClaudeResult<Vec<StreamingToken<Self::Token>>> {
            let mut stream = self.generate_stream(input, config).await?;
            let mut tokens = Vec::new();

            use futures::StreamExt;
            while let Some(token_result) = stream.next().await {
                tokens.push(token_result?);
            }

            Ok(tokens)
        }
    }

    /// Trait for resource management with performance monitoring
    #[async_trait]
    pub trait ResourceManager: Send + Sync + super::Resource {
        /// Resource statistics
        type Stats: Clone + Send + Sync + serde::Serialize;

        /// Get current resource statistics
        async fn get_stats(&self) -> ClaudeResult<Self::Stats>;

        /// Apply resource limits
        async fn apply_limits(&mut self, limits: ResourceLimits) -> ClaudeResult<()>;

        /// Check resource health
        async fn health_check(&self) -> ClaudeResult<HealthStatus>;
    }

    /// Trait for performance measurement and validation
    #[async_trait]
    pub trait PerformanceMonitor: Send + Sync {
        /// Measurement type
        type Measurement: Clone + Send + Sync + serde::Serialize;

        /// Start measuring an operation
        async fn start_measurement(&self, operation: &str) -> ClaudeResult<MeasurementSession>;

        /// End measurement and get results
        async fn end_measurement(
            &self,
            session: MeasurementSession,
        ) -> ClaudeResult<Self::Measurement>;

        /// Validate performance against constraints
        async fn validate_performance(
            &self,
            measurement: &Self::Measurement,
            constraints: &PerformanceConstraints,
        ) -> ClaudeResult<ValidationResult>;
    }

    /// Trait for dependency injection container
    pub trait DependencyContainer: Send + Sync {
        /// Get a service by type
        fn get<T: 'static + Send + Sync>(&self) -> ClaudeResult<&T>;

        /// Register a service
        fn register<T: 'static + Send + Sync>(&mut self, service: T) -> ClaudeResult<()>;

        /// Check if a service is registered
        fn has<T: 'static + Send + Sync>(&self) -> bool;
    }
}

/// Data structures for Claude operations
pub mod types {
    use serde::{Deserialize, Serialize};

    /// Model information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelInfo {
        pub name: String,
        pub architecture: String,
        pub parameters: u64,
        pub context_size: usize,
        pub quantization: String,
        pub memory_mb: u64,
    }

    /// Generation configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GenerationConfig {
        pub max_tokens: Option<usize>,
        pub temperature: Option<f32>,
        pub top_p: Option<f32>,
        pub top_k: Option<usize>,
        pub stop_sequences: Vec<String>,
        pub stream: bool,
        pub timeout_ms: Option<u64>,
    }

    impl Default for GenerationConfig {
        fn default() -> Self {
            Self {
                max_tokens: Some(512),
                temperature: Some(0.7),
                top_p: Some(0.9),
                top_k: None,
                stop_sequences: vec![],
                stream: false,
                timeout_ms: Some(30000),
            }
        }
    }

    /// Streaming token response
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StreamingToken<T> {
        pub token: T,
        pub text: String,
        pub log_prob: Option<f32>,
        pub is_special: bool,
        pub timestamp_ms: u64,
    }

    /// Resource limits
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ResourceLimits {
        pub max_memory_mb: Option<u64>,
        pub max_concurrent_requests: Option<usize>,
        pub max_model_size_mb: Option<u64>,
    }

    /// Health status
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum HealthStatus {
        Healthy,
        Degraded { reasons: Vec<String> },
        Unhealthy { reasons: Vec<String> },
    }

    /// Performance measurement session
    #[derive(Debug, Clone)]
    pub struct MeasurementSession {
        pub id: uuid::Uuid,
        pub operation: String,
        pub start_time: std::time::Instant,
    }

    /// Performance constraints
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceConstraints {
        pub max_latency_ms: Option<u64>,
        pub min_tokens_per_second: Option<f32>,
        pub max_memory_mb: Option<u64>,
        pub max_cpu_usage_percent: Option<f32>,
    }

    /// Validation result
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ValidationResult {
        pub passed: bool,
        pub violated_constraints: Vec<String>,
        pub measurements: serde_json::Value,
    }
}

/// Re-export commonly used items
pub use error::{ClaudeError, ClaudeResult, InferenceError, ModelLoadError, ResourceError};
pub use traits::{
    DependencyContainer, InferenceEngine, ModelManager, PerformanceMonitor, ResourceManager,
};
pub use types::{
    GenerationConfig, HealthStatus, MeasurementSession, ModelInfo, PerformanceConstraints,
    ResourceLimits, StreamingToken, ValidationResult,
};

/// Test specification framework
#[cfg(feature = "test-utils")]
pub mod testing {
    use super::*;
    use proptest::prelude::*;

    /// Property-based testing utilities
    pub mod proptest_helpers {
        use super::super::GenerationConfig;
        use proptest::prelude::*;

        /// Strategy for generating valid generation configs
        pub fn generation_config_strategy() -> impl Strategy<Value = GenerationConfig> {
            (any::<Option<usize>>(), any::<Option<f32>>(), any::<Option<f32>>()).prop_map(
                |(max_tokens, temperature, top_p)| GenerationConfig {
                    max_tokens,
                    temperature,
                    top_p,
                    top_k: None,
                    stop_sequences: vec![],
                    stream: false,
                    timeout_ms: Some(30000),
                },
            )
        }

        /// Strategy for generating valid text inputs
        pub fn text_input_strategy() -> impl Strategy<Value = String> {
            prop::string::string_regex("[a-zA-Z0-9\\s\\.,!?;:]{1,1000}").unwrap()
        }
    }

    /// Performance testing utilities
    pub mod performance {
        use super::super::{ClaudeResult, PerformanceConstraints, ValidationResult};
        use std::time::{Duration, Instant};

        /// Measure execution time of a function
        pub fn measure_time<F, T>(f: F) -> ClaudeResult<(T, Duration)>
        where
            F: FnOnce() -> ClaudeResult<T>,
        {
            let start = Instant::now();
            let result = f()?;
            let elapsed = start.elapsed();
            Ok((result, elapsed))
        }

        /// Validate performance constraints
        pub fn validate_constraints(
            elapsed: Duration,
            constraints: &PerformanceConstraints,
        ) -> ClaudeResult<ValidationResult> {
            let mut violated = Vec::new();

            if let Some(max_latency_ms) = constraints.max_latency_ms {
                if elapsed.as_millis() > max_latency_ms as u128 {
                    violated.push(format!(
                        "Latency: {}ms > {}ms",
                        elapsed.as_millis(),
                        max_latency_ms
                    ));
                }
            }

            Ok(ValidationResult {
                passed: violated.is_empty(),
                violated_constraints: violated,
                measurements: serde_json::json!({
                    "elapsed_ms": elapsed.as_millis()
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, Some(512));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.top_p, Some(0.9));
        assert!(!config.stream);
    }

    #[test]
    fn test_claude_error_chain() {
        let model_error = ModelLoadError::FileNotFound {
            path: "model.gguf".to_string(),
        };
        let claude_error: ClaudeError = model_error.into();

        assert!(matches!(claude_error, ClaudeError::ModelLoadError { .. }));
        assert!(claude_error.to_string().contains("Model loading failed"));
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ValidationResult {
            passed: true,
            violated_constraints: vec![],
            measurements: serde_json::json!({"latency_ms": 100}),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: ValidationResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(result.passed, deserialized.passed);
        assert_eq!(result.violated_constraints, deserialized.violated_constraints);
    }

    #[cfg(feature = "test-utils")]
  proptest! {
        #[test]
        fn test_generation_config_roundtrip(config in testing::proptest_helpers::generation_config_strategy()) {
            let serialized = serde_json::to_string(&config).unwrap();
            let deserialized: GenerationConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(config.max_tokens, deserialized.max_tokens);
            assert_eq!(config.temperature, deserialized.temperature);
            assert_eq!(config.top_p, deserialized.top_p);
        }

        #[test]
        fn test_text_input_validity(input in testing::proptest_helpers::text_input_strategy()) {
            // Test that generated inputs are reasonable
            assert!(!input.is_empty());
            assert!(input.len() <= 1000);
        }
    }
}