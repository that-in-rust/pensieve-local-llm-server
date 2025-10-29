//! Authentication handler for Anthropic API compatibility
//!
//! Validates Bearer tokens from Authorization headers.
//! Accepts both local tokens and Anthropic-format tokens.

use thiserror::Error;

/// Authentication errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AuthError {
    #[error("Missing authorization header")]
    MissingHeader,

    #[error("Invalid token format")]
    InvalidFormat,

    #[error("Token not authorized")]
    Unauthorized,
}

impl AuthError {
    /// HTTP status code for this error
    pub fn status(&self) -> u16 {
        match self {
            AuthError::MissingHeader => 401,
            AuthError::InvalidFormat => 401,
            AuthError::Unauthorized => 401,
        }
    }
}

/// Result type for authentication operations
pub type AuthResult<T> = Result<T, AuthError>;

/// Validate an authentication token
///
/// Accepts:
/// - Local development tokens: "pensieve-local-token"
/// - Anthropic format: "sk-ant-*"
/// - Test tokens: "test-api-key-12345"
///
/// # Arguments
/// * `token` - Optional token string (without "Bearer " prefix)
///
/// # Returns
/// * `Ok(())` if token is valid
/// * `Err(AuthError)` if token is missing or invalid
pub fn validate_auth(token: Option<&str>) -> AuthResult<()> {
    // Check if token is provided
    let token = token.ok_or(AuthError::MissingHeader)?;

    // Check for empty string
    if token.is_empty() {
        return Err(AuthError::InvalidFormat);
    }

    // Accept valid token formats:
    // 1. Local development token
    // 2. Anthropic format (sk-ant-*)
    // 3. Test token
    if token == "pensieve-local-token"
        || token.starts_with("sk-ant-")
        || token == "test-api-key-12345"
    {
        return Ok(());
    }

    // Reject invalid tokens
    Err(AuthError::Unauthorized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_auth_header_fails() {
        let result = validate_auth(None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AuthError::MissingHeader);
        assert_eq!(AuthError::MissingHeader.status(), 401);
    }

    #[test]
    fn test_local_token_succeeds() {
        let result = validate_auth(Some("pensieve-local-token"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_anthropic_format_succeeds() {
        let result = validate_auth(Some("sk-ant-abc123def456"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_test_token_succeeds() {
        let result = validate_auth(Some("test-api-key-12345"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_token_fails() {
        let result = validate_auth(Some("invalid-token-xyz"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AuthError::Unauthorized);
    }

    #[test]
    fn test_empty_string_fails() {
        let result = validate_auth(Some(""));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AuthError::InvalidFormat);
    }
}
