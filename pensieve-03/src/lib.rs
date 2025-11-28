//! Pensieve API Models - External model serialization and API compatibility
//!
//! This is the Layer 3 (L3) API compatibility crate that provides:
//! - Anthropic API message formats
//! - JSON serialization/deserialization
//! - Streaming response formats
//! - API error handling
//!
//! Depends on L1 (pensieve-07) and L2 (pensieve-05) crates.

use pensieve_07_core::CoreError;
use serde::{Deserialize, Serialize};

/// API-specific error types
pub mod error {
    use super::*;
    use thiserror::Error;

    /// API result type
    pub type ApiResult<T> = std::result::Result<T, ApiError>;

    /// API-specific errors
    #[derive(Error, Debug, Clone)]
    pub enum ApiError {
        #[error("Serialization error: {0}")]
        Serialization(String),

        #[error("Validation error: {0}")]
        Validation(String),

        #[error("Invalid message format: {0}")]
        InvalidMessage(String),

        #[error("Missing required field: {0}")]
        MissingField(String),

        #[error("Stream error: {0}")]
        Stream(String),

        #[error("Core error: {0}")]
        Core(#[from] CoreError),
    }

    impl From<serde_json::Error> for ApiError {
        fn from(err: serde_json::Error) -> Self {
            ApiError::Serialization(err.to_string())
        }
    }
}

/// Core API traits
pub mod traits {
    use super::*;

    /// Trait for API messages that can be validated and serialized
    pub trait ApiMessage: Serialize + for<'de> Deserialize<'de> + Send + Sync {
        /// Validate the message content
        fn validate(&self) -> error::ApiResult<()>;

        /// Convert to JSON string
        fn to_json(&self) -> error::ApiResult<String> {
            serde_json::to_string(self).map_err(Into::into)
        }

        /// Parse from JSON string
        fn from_json(json: &str) -> error::ApiResult<Self>
        where
            Self: Sized;
    }

    /// Trait for streaming responses
    pub trait StreamingResponse: Send + Sync {
        /// Convert to streaming format
        fn to_stream_format(&self) -> error::ApiResult<String>;

        /// Check if stream is complete
        fn is_complete(&self) -> bool;

        /// Get event type for streaming
        fn event_type(&self) -> &'static str;
    }
}

/// Anthropic API compatibility models
pub mod anthropic {
    use super::*;

    /// Anthropic message role
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Assistant,
    }

    /// Anthropic content type
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Content {
        Text { text: String },
        Image { source: ImageSource },
    }

    /// System prompt can be either a string or an array of content blocks
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum SystemPrompt {
        String(String),
        Blocks(Vec<Content>),
    }

    /// Image source for content
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct ImageSource {
        #[serde(rename = "type")]
        pub source_type: String,
        pub media_type: String,
        pub data: String, // base64 encoded
    }

    /// Message content can be either a string or an array of content blocks
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum MessageContent {
        String(String),
        Blocks(Vec<Content>),
    }

    /// Anthropic message format
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Message {
        pub role: Role,
        pub content: MessageContent,
    }

    /// Anthropic API request
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CreateMessageRequest {
        pub model: String,
        #[serde(default = "default_max_tokens")]
        pub max_tokens: u32,
        pub messages: Vec<Message>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system: Option<SystemPrompt>,
    }

    fn default_max_tokens() -> u32 {
        4096
    }

    impl ApiMessage for CreateMessageRequest {
        fn validate(&self) -> error::ApiResult<()> {
            if self.model.is_empty() {
                return Err(ApiError::Validation("Model cannot be empty".to_string()));
            }
            if self.max_tokens == 0 {
                return Err(ApiError::Validation("Max tokens must be positive".to_string()));
            }
            if self.messages.is_empty() {
                return Err(ApiError::Validation("Messages cannot be empty".to_string()));
            }
            for message in &self.messages {
                match &message.content {
                    MessageContent::String(s) if s.is_empty() => {
                        return Err(ApiError::Validation("Message content cannot be empty".to_string()));
                    }
                    MessageContent::Blocks(blocks) if blocks.is_empty() => {
                        return Err(ApiError::Validation("Message content cannot be empty".to_string()));
                    }
                    _ => {}
                }
            }
            Ok(())
        }

        fn from_json(json: &str) -> error::ApiResult<Self>
        where
            Self: Sized,
        {
            serde_json::from_str(json).map_err(Into::into)
        }
    }

    /// Anthropic API response
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CreateMessageResponse {
        pub id: String,
        pub r#type: String,
        pub role: Role,
        pub content: Vec<Content>,
        pub model: String,
        pub stop_reason: Option<String>,
        pub stop_sequence: Option<String>,
        pub usage: Usage,
    }

    /// Token usage information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Usage {
        pub input_tokens: u32,
        pub output_tokens: u32,
    }

    /// Streaming response event
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StreamingEvent {
        #[serde(rename = "type")]
        pub event_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub delta: Option<Content>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub message: Option<CreateMessageResponse>,
    }

    impl StreamingResponse for StreamingEvent {
        fn to_stream_format(&self) -> error::ApiResult<String> {
            let json = serde_json::to_string(self)?;
            Ok(format!("data: {}\n\n", json))
        }

        fn is_complete(&self) -> bool {
            self.event_type == "message_stop"
        }

        fn event_type(&self) -> &'static str {
            match self.event_type.as_str() {
                "message_start" => "message_start",
                "content_block_start" => "content_block_start",
                "content_block_delta" => "content_block_delta",
                "content_block_stop" => "content_block_stop",
                "message_delta" => "message_delta",
                "message_stop" => "message_stop",
                _ => "unknown",
            }
        }
    }
}

/// Re-export commonly used items
pub use error::{ApiError, ApiResult};
pub use traits::{ApiMessage, StreamingResponse};

/// Result type alias for convenience
pub type Result<T> = ApiResult<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use anthropic::*;

    #[test]
    fn test_basic_message_creation() {
        let message = Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "Hello, world!".to_string(),
            }],
        };

        assert_eq!(message.role, Role::User);
        assert_eq!(message.content.len(), 1);
        if let Content::Text { text } = &message.content[0] {
            assert_eq!(text, "Hello, world!");
        }
    }

    #[test]
    fn test_request_validation() {
        let valid_request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "Test message".to_string(),
                }],
            }],
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            system: None,
        };

        assert!(valid_request.validate().is_ok());

        // Test empty model validation
        let invalid_request = CreateMessageRequest {
            model: "".to_string(),
            ..valid_request.clone()
        };
        assert!(invalid_request.validate().is_err());

        // Test zero max_tokens validation
        let invalid_request = CreateMessageRequest {
            max_tokens: 0,
            ..valid_request.clone()
        };
        assert!(invalid_request.validate().is_err());

        // Test empty messages validation
        let invalid_request = CreateMessageRequest {
            messages: vec![],
            ..valid_request.clone()
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_request_serialization() {
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "Test message".to_string(),
                }],
            }],
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            system: None,
        };

        let json = request.to_json().unwrap();
        assert!(json.contains("claude-3-sonnet-20240229"));
        assert!(json.contains("Test message"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"
        {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        "#;

        let response: CreateMessageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "msg_123");
        assert_eq!(response.role, Role::Assistant);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn test_streaming_response() {
        let event = StreamingEvent {
            event_type: "content_block_delta".to_string(),
            index: Some(0),
            delta: Some(Content::Text {
                text: "Hello".to_string(),
            }),
            message: None,
        };

        assert!(!event.is_complete());
        assert_eq!(event.event_type(), "content_block_delta");

        let stream_format = event.to_stream_format().unwrap();
        assert!(stream_format.starts_with("data: "));
        assert!(stream_format.ends_with("\n\n"));
    }

    #[test]
    fn test_streaming_completion() {
        let event = StreamingEvent {
            event_type: "message_stop".to_string(),
            index: None,
            delta: None,
            message: None,
        };

        assert!(event.is_complete());
        assert_eq!(event.event_type(), "message_stop");
    }

    #[test]
    fn test_image_content_serialization() {
        let message = Message {
            role: Role::User,
            content: vec![Content::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/jpeg".to_string(),
                    data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg==".to_string(),
                },
            }],
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("image/jpeg"));
        assert!(json.contains("base64"));
    }

    #[test]
    fn test_error_handling() {
        let invalid_json = "{ invalid json }";
        let result: std::result::Result<CreateMessageRequest, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());

        let error = ApiError::Validation("test error".to_string());
        assert_eq!(error.to_string(), "Validation error: test error");
    }

    #[test]
    fn test_request_from_json() {
        let json = r#"
        {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                }
            ],
            "temperature": 0.7,
            "stream": false
        }
        "#;

        let request = CreateMessageRequest::from_json(json).unwrap();
        assert_eq!(request.model, "claude-3-sonnet-20240229");
        assert_eq!(request.max_tokens, 100);
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.stream, Some(false));
    }

    #[test]
    fn test_usage_tracking() {
        let usage = Usage {
            input_tokens: 25,
            output_tokens: 15,
        };

        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("25"));
        assert!(json.contains("15"));

        let deserialized: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.input_tokens, 25);
        assert_eq!(deserialized.output_tokens, 15);
    }

    #[test]
    fn test_complex_message_with_multiple_content() {
        let message = Message {
            role: Role::User,
            content: vec![
                Content::Text {
                    text: "Look at this image:".to_string(),
                },
                Content::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/png".to_string(),
                        data: "ABC123".to_string(),
                    },
                },
                Content::Text {
                    text: "What do you see?".to_string(),
                },
            ],
        };

        assert_eq!(message.content.len(), 3);
        match &message.content[0] {
            Content::Text { text } => assert_eq!(text, "Look at this image:"),
            _ => panic!("Expected text content"),
        }
        match &message.content[1] {
            Content::Image { source } => assert_eq!(source.media_type, "image/png"),
            _ => panic!("Expected image content"),
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original_request = CreateMessageRequest {
            model: "claude-3-opus-20240229".to_string(),
            max_tokens: 4096,
            messages: vec![
                Message {
                    role: Role::User,
                    content: vec![Content::Text {
                        text: "Explain quantum computing".to_string(),
                    }],
                },
                Message {
                    role: Role::Assistant,
                    content: vec![Content::Text {
                        text: "Quantum computing is...".to_string(),
                    }],
                },
            ],
            temperature: Some(0.5),
            top_p: Some(0.9),
            stream: Some(true),
            system: Some("You are a helpful assistant.".to_string()),
        };

        let json = original_request.to_json().unwrap();
        let deserialized_request = CreateMessageRequest::from_json(&json).unwrap();

        assert_eq!(original_request.model, deserialized_request.model);
        assert_eq!(original_request.max_tokens, deserialized_request.max_tokens);
        assert_eq!(original_request.messages.len(), deserialized_request.messages.len());
        assert_eq!(original_request.temperature, deserialized_request.temperature);
        assert_eq!(original_request.top_p, deserialized_request.top_p);
        assert_eq!(original_request.stream, deserialized_request.stream);
        assert_eq!(original_request.system, deserialized_request.system);
    }

    // Property-based test using proptest
    #[test]
    fn test_proptest_message_validation() {
        use proptest::prelude::*;

        proptest!(|(
            model in "[a-zA-Z0-9_-]{1,50}",
            max_tokens in 1u32..=8192,
            text in "[a-zA-Z0-9 ]{1,100}"
        )| {
            let request = CreateMessageRequest {
                model: model.clone(),
                max_tokens,
                messages: vec![Message {
                    role: Role::User,
                    content: vec![Content::Text { text }],
                }],
                temperature: None,
                top_p: None,
                stream: None,
                system: None,
            };

            prop_assert!(request.validate().is_ok());
        });
    }
}