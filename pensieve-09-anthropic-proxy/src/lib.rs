//! Pensieve Anthropic Proxy (Layer 3)
//!
//! This crate provides an Anthropic API-compatible proxy layer that:
//! - Validates authentication (Bearer tokens)
//! - Translates requests from Anthropic format to MLX format
//! - Translates responses from MLX format to Anthropic format
//! - Handles SSE streaming for real-time responses
//!
//! Follows TDD principles with tests written first.

/// Authentication handling
pub mod auth;

/// Request/Response translation
pub mod translator;

/// HTTP server integration
pub mod server;

/// SSE streaming support
pub mod streaming;

/// Re-export commonly used types
pub use auth::{AuthError, AuthResult, validate_auth};
pub use translator::{
    MlxRequest, TranslationError, TranslationResult,
    translate_anthropic_to_mlx, translate_mlx_to_anthropic,
};
pub use server::{AnthropicProxyServer, ServerConfig, ServerError, ServerResult};
