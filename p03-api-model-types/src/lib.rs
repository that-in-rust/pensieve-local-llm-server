# API Model Types with Four-Word Naming
# Following parseltongue principles for API compatibility

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Four-word function names as per parseltongue principles

/// Validate API message contract with rules
///
/// # Preconditions
/// - Message structure follows Anthropic API format
/// - All required fields present
///
/// # Postconditions
/// - Returns Ok(()) if validation passes
/// - Returns Err(ApiError) if validation fails
/// - Provides detailed error context
///
/// # Error Conditions
/// - Empty model string
/// - Invalid token counts
/// - Missing required fields
/// - Malformed content structure
pub fn validate_api_message_contract<T: ApiMessage>(message: &T) -> Result<(), ApiError> {
    message.validate()
}

/// Serialize API message to JSON format
///
/// # Preconditions
/// - Valid API message structure
/// - All fields serializable
///
/// # Postconditions
/// - Returns JSON string representation
/// - Returns Err(ApiError) on serialization failure
/// - Format matches Anthropic API specification
///
/// # Error Conditions
/// - Circular references
/// - Non-serializable fields
/// - Invalid Unicode content
pub fn serialize_api_message_to_json<T: ApiMessage>(message: &T) -> Result<String, ApiError> {
    message.to_json()
}

/// Parse API message from JSON string
///
/// # Preconditions
/// - Valid JSON string
/// - Format matches Anthropic API specification
///
/// # Postconditions
/// - Returns parsed API message
/// - Returns Err(ApiError) on parsing failure
/// - Type safety guaranteed
///
/// # Error Conditions
/// - Invalid JSON syntax
/// - Missing required fields
/// - Type mismatches
/// - Unknown enum variants
pub fn parse_api_message_from_json<T: ApiMessage>(json: &str) -> Result<T, ApiError>
where
    T: Sized,
{
    T::from_json(json)
}

/// Convert streaming response to SSE format
///
/// # Preconditions
/// - Valid streaming response structure
/// - Implements StreamingResponse trait
///
/// # Postconditions
/// - Returns SSE formatted string
/// - Includes proper data prefix
/// - Terminates with double newlines
///
/// # Error Conditions
/// - Serialization failure
/// - Invalid event structure
/// - Missing required fields
pub fn convert_streaming_to_sse_format(response: &dyn StreamingResponse) -> Result<String, ApiError> {
    response.to_stream_format()
}

/// Check if streaming response is complete
///
/// # Preconditions
/// - Valid streaming response
/// - Implements StreamingResponse trait
///
/// # Postconditions
/// - Returns boolean completion status
/// - No side effects
///
/// # Error Conditions
/// - None (always returns Result)
pub fn check_streaming_response_complete(response: &dyn StreamingResponse) -> Result<bool, ApiError> {
    Ok(response.is_complete())
}

/// Get event type from streaming response
///
/// # Preconditions
/// - Valid streaming response
/// - Implements StreamingResponse trait
///
/// # Postconditions
/// - Returns string event type
/// - Maps to standard SSE events
/// - Returns "unknown" for invalid types
///
/// # Error Conditions
/// - None (always returns Result)
pub fn extract_streaming_event_type(response: &dyn StreamingResponse) -> Result<&str, ApiError> {
    Ok(response.event_type())
}

// Supporting types following parseltongue patterns

/// API error types with hierarchical structure
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Invalid message format: {0}")]
    InvalidMessageFormat(String),

    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("JSON parsing error: {0}")]
    JsonParsingError(String),
}

/// Core traits for API message handling
pub mod traits {
    use super::*;

    /// Trait for API messages with validation and serialization
    pub trait ApiMessage: Serialize + for<'de> Deserialize<'de> + Send + Sync {
        /// Validate message structure and content
        fn validate(&self) -> Result<(), ApiError>;

        /// Convert to JSON string with error handling
        fn to_json(&self) -> Result<String, ApiError> {
            serde_json::to_string(self).map_err(|e| ApiError::SerializationFailed(e.to_string()))
        }

        /// Parse from JSON string with validation
        fn from_json(json: &str) -> Result<Self, ApiError>
        where
            Self: Sized,
        {
            serde_json::from_str(json).map_err(|e| ApiError::JsonParsingError(e.to_string()))
        }
    }

    /// Trait for streaming responses with SSE formatting
    pub trait StreamingResponse: Send + Sync {
        /// Convert to Server-Sent Events format
        fn to_stream_format(&self) -> Result<String, ApiError>;

        /// Check if streaming is complete
        fn is_complete(&self) -> bool;

        /// Get event type identifier
        fn event_type(&self) -> &str;

        /// Get streaming timestamp
        fn timestamp(&self) -> Option<chrono::DateTime<chrono::Utc>>;
    }
}

/// Anthropic API compatibility models with proper naming
pub mod anthropic {
    use super::*;
    use chrono;

    /// Message role identifier
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Assistant,
    }

    /// Content block type identifier
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ContentType {
        Text { text: String },
        Image { source: ImageSource },
    }

    /// Image source for content blocks
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct ImageSource {
        #[serde(rename = "type")]
        pub source_type: String,
        pub media_type: String,
        pub data: String, // Base64 encoded image data
    }

    /// Message content can be string or content blocks
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum MessageContent {
        String(String),
        Blocks(Vec<ContentType>),
    }

    /// Complete message structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Message {
        pub role: Role,
        pub content: MessageContent,
    }

    /// Anthropic API creation request
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
        pub system: Option<String>,
    }

    impl Default for CreateMessageRequest {
        fn default() -> Self {
            Self {
                model: "phi-3-mini-128k-instruct-4bit".to_string(),
                max_tokens: 4096,
                messages: vec![],
                temperature: None,
                top_p: None,
                stream: None,
                system: None,
            }
        }
    }

    impl ApiMessage for CreateMessageRequest {
        fn validate(&self) -> Result<(), ApiError> {
            if self.model.is_empty() {
                return Err(ApiError::ValidationFailed(
                    "Model identifier cannot be empty".to_string()
                ));
            }
            if self.max_tokens == 0 {
                return Err(ApiError::ValidationFailed(
                    "Max tokens must be positive".to_string()
                ));
            }
            if self.messages.is_empty() {
                return Err(ApiError::ValidationFailed(
                    "Messages array cannot be empty".to_string()
                ));
            }

            for message in &self.messages {
                match &message.content {
                    MessageContent::String(text) if text.trim().is_empty() => {
                        return Err(ApiError::ValidationFailed(
                            "Message content cannot be empty".to_string()
                        ));
                    }
                    MessageContent::Blocks(blocks) if blocks.is_empty() => {
                        return Err(ApiError::ValidationFailed(
                            "Content blocks cannot be empty".to_string()
                        ));
                    }
                    _ => {} // Valid content
                }
            }

            Ok(())
        }
    }

    /// Anthropic API creation response
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CreateMessageResponse {
        pub id: String,
        pub r#type: String,
        pub role: Role,
        pub content: Vec<ContentType>,
        pub model: String,
        pub stop_reason: Option<String>,
        pub stop_sequence: Option<String>,
        pub usage: UsageMetrics,
    }

    /// Token usage tracking
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UsageMetrics {
        pub input_tokens: u32,
        pub output_tokens: u32,
    }

    /// Server-Sent Events streaming response
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StreamingEvent {
        #[serde(rename = "type")]
        pub event_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub index: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub delta: Option<ContentType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub message: Option<CreateMessageResponse>,
    }

    impl StreamingResponse for StreamingEvent {
        fn to_stream_format(&self) -> Result<String, ApiError> {
            let json = serde_json::to_string(self)
                .map_err(|e| ApiError::SerializationFailed(e.to_string()))?;
            Ok(format!("data: {}\n\n", json))
        }

        fn is_complete(&self) -> bool {
            self.event_type == "message_stop"
        }

        fn event_type(&self) -> &str {
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

        fn timestamp(&self) -> Option<chrono::DateTime<chrono::Utc>> {
            Some(chrono::Utc::now())
        }
    }
}

// Re-export commonly used items for API compatibility
pub use {
    ApiError,
    ApiMessage,
    StreamingResponse,
};

/// Result type alias for API operations
pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use anthropic::*;

    #[test]
    fn test_validate_message_contract_valid_request() -> Result<(), ApiError> {
        let request = CreateMessageRequest {
            model: "phi-3-mini-128k-instruct-4bit".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::String("Test message".to_string()),
            }],
            temperature: Some(0.7),
            ..Default::default()
        };

        validate_api_message_contract(&request)
    }

    #[test]
    fn test_validate_message_contract_empty_model() -> Result<(), ApiError> {
        let request = CreateMessageRequest {
            model: "".to_string(),
            ..Default::default()
        };

        assert!(validate_api_message_contract(&request).is_err());
        match validate_api_message_contract(&request).unwrap_err() {
            ApiError::ValidationFailed(msg) => assert!(msg.contains("Model identifier cannot be empty")),
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_serialize_api_message_to_json_success() -> Result<(), ApiError> {
        let request = CreateMessageRequest {
            model: "phi-3-mini-128k-instruct-4bit".to_string(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::String("Hello".to_string()),
            }],
            ..Default::default()
        };

        let json = serialize_api_message_to_json(&request)?;
        assert!(json.contains("phi-3-mini-128k-instruct-4bit"));
        assert!(json.contains("Hello"));
        Ok(())
    }

    #[test]
    fn test_parse_api_message_from_json_success() -> Result<(), ApiError> {
        let json = r#"
        {
            "model": "phi-3-mini-128k-instruct-4bit",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                }
            ]
        }
        "#;

        let request: CreateMessageRequest = parse_api_message_from_json(json)?;
        assert_eq!(request.model, "phi-3-mini-128k-instruct-4bit");
        assert_eq!(request.max_tokens, 100);
        assert_eq!(request.messages.len(), 1);
        Ok(())
    }

    #[test]
    fn test_parse_api_message_invalid_json() -> Result<(), ApiError> {
        let invalid_json = "{ invalid json }";
        let result: Result<CreateMessageRequest, ApiError> = parse_api_message_from_json(invalid_json);
        assert!(result.is_err());

        match result.unwrap_err() {
            ApiError::JsonParsingError(_) => {}, // Expected
            _ => panic!("Expected JsonParsingError"),
        }
        Ok(())
    }

    #[test]
    fn test_streaming_conversion_to_sse_format() -> Result<(), ApiError> {
        let event = StreamingEvent {
            event_type: "content_block_delta".to_string(),
            delta: Some(ContentType::Text {
                text: "Hello".to_string(),
            }),
            ..Default::default()
        };

        let sse = convert_streaming_to_sse_format(&event)?;
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
        Ok(())
    }

    #[test]
    fn test_streaming_completion_detection() -> Result<(), ApiError> {
        let incomplete = StreamingEvent {
            event_type: "content_block_delta".to_string(),
            ..Default::default()
        };

        let complete = StreamingEvent {
            event_type: "message_stop".to_string(),
            ..Default::default()
        };

        assert!(!check_streaming_response_complete(&incomplete)?);
        assert!(check_streaming_response_complete(&complete)?);
        Ok(())
    }

    #[test]
    fn test_event_type_extraction() -> Result<(), ApiError> {
        let start_event = StreamingEvent {
            event_type: "message_start".to_string(),
            ..Default::default()
        };

        let delta_event = StreamingEvent {
            event_type: "content_block_delta".to_string(),
            ..Default::default()
        };

        let stop_event = StreamingEvent {
            event_type: "message_stop".to_string(),
            ..Default::default()
        };

        assert_eq!(extract_streaming_event_type(&start_event)?, "message_start");
        assert_eq!(extract_streaming_event_type(&delta_event)?, "content_block_delta");
        assert_eq!(extract_streaming_event_type(&stop_event)?, "message_stop");
        Ok(())
    }

    #[test]
    fn test_complex_message_with_multiple_content() -> Result<(), ApiError> {
        let request = CreateMessageRequest {
            model: "phi-3-mini-128k-instruct-4bit".to_string(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![
                        ContentType::Text {
                            text: "Explain quantum computing".to_string(),
                        },
                        ContentType::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type: "image/png".to_string(),
                                data: "ABC123".to_string(),
                            },
                        },
                    ]),
                }
            ],
            ..Default::default()
        };

        validate_api_message_contract(&request)
    }

    // Property-based test for request validation
    #[test]
    fn test_property_based_message_validation() -> Result<(), ApiError> {
        use proptest::prelude::*;

        proptest!(|(
            model in "[a-zA-Z0-9_-]{1,50}",
            max_tokens in 1u32..=8192,
            message_text in "[a-zA-Z0-9 ]{1,200}"
        )| {
            let request = CreateMessageRequest {
                model: model.clone(),
                max_tokens,
                messages: vec![Message {
                    role: Role::User,
                    content: MessageContent::String(message_text.clone()),
                }],
                ..Default::default()
            };

            validate_api_message_contract(&request)
        });
    }

    #[test]
    fn test_serialization_roundtrip_preserves_data() -> Result<(), ApiError> {
        let original = CreateMessageRequest {
            model: "claude-3-opus-20240229".to_string(),
            max_tokens: 4096,
            messages: vec![
                Message {
                    role: Role::User,
                    content: MessageContent::String("Explain Rust ownership".to_string()),
                },
                Message {
                    role: Role::Assistant,
                    content: MessageContent::String("Rust uses ownership system".to_string()),
                },
            ],
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(true),
            system: Some("You are helpful".to_string()),
        };

        let json = serialize_api_message_to_json(&original)?;
        let deserialized = parse_api_message_from_json(&json)?;

        assert_eq!(original.model, deserialized.model);
        assert_eq!(original.max_tokens, deserialized.max_tokens);
        assert_eq!(original.messages.len(), deserialized.messages.len());
        assert_eq!(original.temperature, deserialized.temperature);
        assert_eq!(original.top_p, deserialized.top_p);
        assert_eq!(original.stream, deserialized.stream);
        Ok(())
    }

    #[test]
    fn test_usage_metrics_tracking() -> Result<(), ApiError> {
        let usage = UsageMetrics {
            input_tokens: 100,
            output_tokens: 200,
        };

        let response = CreateMessageResponse {
            id: "msg_123".to_string(),
            r#type: "message".to_string(),
            role: Role::Assistant,
            content: vec![ContentType::Text {
                text: "Response content".to_string(),
            }],
            model: "phi-3".to_string(),
            usage,
            ..Default::default()
        };

        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 200);
        Ok(())
    }

    #[test]
    fn test_error_message_context() -> Result<(), ApiError> {
        let error = ApiError::ValidationFailed("Model identifier cannot be empty".to_string());
        assert_eq!(error.to_string(), "Validation failed: Model identifier cannot be empty");
        Ok(())
    }
}