//! Request/Response translation between Anthropic and MLX formats
//!
//! Translates Anthropic Messages API format to MLX inference format and back.

use pensieve_03::anthropic::{
    CreateMessageRequest, CreateMessageResponse, Content, Message, MessageContent, Role, SystemPrompt, Usage
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Translation errors
#[derive(Error, Debug, Clone)]
pub enum TranslationError {
    #[error("Invalid message format: {0}")]
    InvalidFormat(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Unsupported content type")]
    UnsupportedContent,
}

/// Result type for translation operations
pub type TranslationResult<T> = Result<T, TranslationError>;

/// MLX-compatible request format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxRequest {
    /// Combined prompt string with role markers
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p for sampling
    pub top_p: Option<f32>,
}

/// Translate Anthropic request to MLX format
///
/// Converts the Anthropic Messages API format into a simple prompt string
/// that MLX can process, with role markers (System:, User:, Assistant:).
///
/// # Arguments
/// * `request` - Anthropic CreateMessageRequest
///
/// # Returns
/// * `Ok(MlxRequest)` - Translated request
/// * `Err(TranslationError)` - If translation fails
pub fn translate_anthropic_to_mlx(request: &CreateMessageRequest) -> TranslationResult<MlxRequest> {
    let mut prompt_parts = Vec::new();

    // Add system prompt if provided
    if let Some(system) = &request.system {
        match system {
            SystemPrompt::String(text) => {
                prompt_parts.push(format!("System: {}", text));
            }
            SystemPrompt::Blocks(blocks) => {
                for block in blocks {
                    if let Content::Text { text } = block {
                        prompt_parts.push(format!("System: {}", text));
                    }
                }
            }
        }
    }

    // Add conversation messages
    for message in &request.messages {
        let role_prefix = match message.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };

        // Extract text from message content
        match &message.content {
            MessageContent::String(text) => {
                prompt_parts.push(format!("{}: {}", role_prefix, text));
            }
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    if let Content::Text { text } = block {
                        prompt_parts.push(format!("{}: {}", role_prefix, text));
                    }
                }
            }
        }
    }

    // Add final assistant prompt for generation
    prompt_parts.push("Assistant:".to_string());

    // Combine into single prompt
    let prompt = prompt_parts.join("\n");

    // Get temperature with sensible default
    let temperature = request.temperature.unwrap_or(0.7);

    Ok(MlxRequest {
        prompt,
        max_tokens: request.max_tokens,
        temperature,
        top_p: request.top_p,
    })
}

/// Translate MLX output to Anthropic response format
///
/// Converts MLX text output into Anthropic Messages API response format
/// with proper structure, token counts, and metadata.
///
/// # Arguments
/// * `mlx_output` - Generated text from MLX
/// * `input_tokens` - Number of input tokens processed
/// * `output_tokens` - Number of output tokens generated
///
/// # Returns
/// * Anthropic CreateMessageResponse
pub fn translate_mlx_to_anthropic(
    mlx_output: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> CreateMessageResponse {
    use uuid::Uuid;

    CreateMessageResponse {
        id: format!("msg_{}", Uuid::new_v4()),
        r#type: "message".to_string(),
        role: Role::Assistant,
        content: vec![Content::Text {
            text: mlx_output.to_string(),
        }],
        model: "phi-3-mini".to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_user_message_translation() {
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".into(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![Content::Text {
                    text: "Hello, Claude!".into(),
                }]),
            }],
            system: None,
            temperature: Some(0.7),
            top_p: None,
            stream: None,
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // Postconditions: Verify prompt structure
        assert!(mlx_request.prompt.contains("User: Hello, Claude!"));
        assert!(mlx_request.prompt.contains("Assistant:"));
        assert_eq!(mlx_request.max_tokens, 100);
        assert_eq!(mlx_request.temperature, 0.7);
    }

    #[test]
    fn test_system_prompt_included() {
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".into(),
            max_tokens: 100,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::String("Hello".into()),
            }],
            system: Some(SystemPrompt::String("You are a helpful assistant.".into())),
            temperature: Some(1.0),
            top_p: None,
            stream: None,
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // System prompt should come first
        assert!(mlx_request.prompt.contains("System: You are a helpful assistant."));
        assert!(mlx_request.prompt.contains("User: Hello"));
    }

    #[test]
    fn test_multi_turn_conversation() {
        let request = CreateMessageRequest {
            model: "claude-3-sonnet-20240229".into(),
            max_tokens: 200,
            messages: vec![
                Message {
                    role: Role::User,
                    content: MessageContent::String("What is 2+2?".into()),
                },
                Message {
                    role: Role::Assistant,
                    content: MessageContent::String("4".into()),
                },
                Message {
                    role: Role::User,
                    content: MessageContent::String("Correct! What is 3+3?".into()),
                },
            ],
            system: None,
            temperature: Some(0.7),
            top_p: None,
            stream: None,
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // Verify conversation structure in order
        assert!(mlx_request.prompt.contains("User: What is 2+2?"));
        assert!(mlx_request.prompt.contains("Assistant: 4"));
        assert!(mlx_request.prompt.contains("User: Correct! What is 3+3?"));
        assert!(mlx_request.prompt.ends_with("Assistant:"));
    }

    #[test]
    fn test_mlx_output_to_anthropic_response() {
        let mlx_output = "Hello! How can I help you?";
        let input_tokens = 15;
        let output_tokens = 8;

        let response = translate_mlx_to_anthropic(mlx_output, input_tokens, output_tokens);

        // Postconditions: Verify response structure
        assert_eq!(response.role, Role::Assistant);
        assert_eq!(response.r#type, "message");
        assert_eq!(response.content.len(), 1);

        if let Content::Text { text } = &response.content[0] {
            assert_eq!(text, mlx_output);
        } else {
            panic!("Expected text content");
        }

        // Verify token counts
        assert_eq!(response.usage.input_tokens, input_tokens);
        assert_eq!(response.usage.output_tokens, output_tokens);
        assert!(response.stop_reason.is_some());
    }

    #[test]
    fn test_default_temperature() {
        let request = CreateMessageRequest {
            model: "phi-3-mini".into(),
            max_tokens: 50,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::String("Test".into()),
            }],
            system: None,
            temperature: None, // No temperature specified
            top_p: None,
            stream: None,
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // Should use a sensible default (0.7 is common)
        assert!(mlx_request.temperature > 0.0);
        assert!(mlx_request.temperature <= 1.0);
    }

    #[test]
    fn test_string_content_format() {
        let request = CreateMessageRequest {
            model: "phi-3-mini".into(),
            max_tokens: 50,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::String("Simple string message".into()),
            }],
            system: None,
            temperature: Some(0.8),
            top_p: None,
            stream: None,
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        assert!(mlx_request.prompt.contains("User: Simple string message"));
    }

    #[test]
    fn test_blocks_content_format() {
        let request = CreateMessageRequest {
            model: "phi-3-mini".into(),
            max_tokens: 50,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Blocks(vec![
                    Content::Text {
                        text: "Part 1".into(),
                    },
                    Content::Text {
                        text: "Part 2".into(),
                    },
                ]),
            }],
            system: None,
            temperature: Some(0.9),
            top_p: None,
            stream: None,
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // Both parts should be included
        assert!(mlx_request.prompt.contains("Part 1"));
        assert!(mlx_request.prompt.contains("Part 2"));
    }
}
