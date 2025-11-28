//! SSE (Server-Sent Events) Streaming Support
//!
//! Implements Anthropic-compatible streaming for real-time token generation.
//!
//! Event sequence (per Anthropic specification):
//! 1. message_start
//! 2. content_block_start
//! 3. content_block_delta (repeated for each token)
//! 4. content_block_stop
//! 5. message_delta (with usage)
//! 6. message_stop

use serde::{Deserialize, Serialize};
use serde_json::json;

/// SSE event types as defined by Anthropic API
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEventType {
    MessageStart,
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockStop,
    MessageDelta,
    MessageStop,
}

impl SseEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SseEventType::MessageStart => "message_start",
            SseEventType::ContentBlockStart => "content_block_start",
            SseEventType::ContentBlockDelta => "content_block_delta",
            SseEventType::ContentBlockStop => "content_block_stop",
            SseEventType::MessageDelta => "message_delta",
            SseEventType::MessageStop => "message_stop",
        }
    }
}

/// Format an SSE event with proper structure
/// Format: "event: {name}\ndata: {json}\n\n"
pub fn format_sse_event(event_type: SseEventType, data: serde_json::Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type.as_str(),
        serde_json::to_string(&data).unwrap()
    )
}

/// Generate message_start event
pub fn create_message_start_event(message_id: &str, model: &str) -> String {
    let data = json!({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        }
    });
    format_sse_event(SseEventType::MessageStart, data)
}

/// Generate content_block_start event
pub fn create_content_block_start_event(index: u32) -> String {
    let data = json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "text",
            "text": ""
        }
    });
    format_sse_event(SseEventType::ContentBlockStart, data)
}

/// Generate content_block_delta event for a single token
pub fn create_content_block_delta_event(index: u32, text: &str) -> String {
    let data = json!({
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "text_delta",
            "text": text
        }
    });
    format_sse_event(SseEventType::ContentBlockDelta, data)
}

/// Generate content_block_stop event
pub fn create_content_block_stop_event(index: u32) -> String {
    let data = json!({
        "type": "content_block_stop",
        "index": index
    });
    format_sse_event(SseEventType::ContentBlockStop, data)
}

/// Generate message_delta event with usage information
pub fn create_message_delta_event(stop_reason: &str, _input_tokens: u32, output_tokens: u32) -> String {
    let data = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason
        },
        "usage": {
            "output_tokens": output_tokens
        }
    });
    format_sse_event(SseEventType::MessageDelta, data)
}

/// Generate message_stop event
pub fn create_message_stop_event() -> String {
    let data = json!({
        "type": "message_stop"
    });
    format_sse_event(SseEventType::MessageStop, data)
}

/// Stream generator that produces SSE events from token stream
pub async fn generate_sse_stream(
    tokens: Vec<String>,
    message_id: String,
    model: String,
    input_tokens: u32,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut events = Vec::new();

    // 1. message_start
    events.push(create_message_start_event(&message_id, &model));

    // 2. content_block_start
    events.push(create_content_block_start_event(0));

    // 3. content_block_delta for each token
    for token in &tokens {
        events.push(create_content_block_delta_event(0, token));
    }

    // 4. content_block_stop
    events.push(create_content_block_stop_event(0));

    // 5. message_delta with usage (output_tokens = number of tokens)
    let output_tokens = tokens.len() as u32;
    events.push(create_message_delta_event("end_turn", input_tokens, output_tokens));

    // 6. message_stop
    events.push(create_message_stop_event());

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_event_type_as_str() {
        assert_eq!(SseEventType::MessageStart.as_str(), "message_start");
        assert_eq!(SseEventType::ContentBlockDelta.as_str(), "content_block_delta");
        assert_eq!(SseEventType::MessageStop.as_str(), "message_stop");
    }

    #[test]
    fn test_format_sse_event() {
        // RED: This should fail because format_sse_event is todo!()
        let data = json!({"type": "message_start", "message": {"id": "msg_123"}});
        let event = format_sse_event(SseEventType::MessageStart, data);

        // Verify SSE format: "event: name\ndata: json\n\n"
        assert!(event.starts_with("event: message_start\n"));
        assert!(event.contains("data: "));
        assert!(event.ends_with("\n\n"));
        assert!(event.contains("msg_123"));
    }

    #[test]
    fn test_sse_event_stream_sequence() {
        // RED: This should fail - testing the complete 6-event sequence
        let tokens = vec!["Hello".to_string(), " world".to_string(), "!".to_string()];
        let message_id = "msg_test123".to_string();
        let model = "claude-3-sonnet-20240229".to_string();
        let input_tokens = 10;

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let events = runtime.block_on(async {
            generate_sse_stream(tokens, message_id, model, input_tokens).await.unwrap()
        });

        // Should have exactly 6 base events + 3 token deltas = 9 total
        // 1. message_start
        // 2. content_block_start
        // 3. content_block_delta (Hello)
        // 4. content_block_delta ( world)
        // 5. content_block_delta (!)
        // 6. content_block_stop
        // 7. message_delta
        // 8. message_stop
        assert_eq!(events.len(), 8, "Should have 8 events total");

        // Verify first event is message_start
        assert!(events[0].contains("event: message_start"));
        assert!(events[0].contains("msg_test123"));

        // Verify second event is content_block_start
        assert!(events[1].contains("event: content_block_start"));
        assert!(events[1].contains("\"index\":0"));

        // Verify token delta events
        assert!(events[2].contains("event: content_block_delta"));
        assert!(events[2].contains("Hello"));

        assert!(events[3].contains("event: content_block_delta"));
        assert!(events[3].contains(" world"));

        assert!(events[4].contains("event: content_block_delta"));
        assert!(events[4].contains("!"));

        // Verify content_block_stop
        assert!(events[5].contains("event: content_block_stop"));

        // Verify message_delta with usage
        // Note: Per Anthropic spec, message_delta only includes output_tokens
        // input_tokens are in message_start
        assert!(events[6].contains("event: message_delta"));
        assert!(events[6].contains("\"stop_reason\":\"end_turn\""));
        assert!(events[6].contains("\"output_tokens\":3"));

        // Verify message_stop
        assert!(events[7].contains("event: message_stop"));
    }

    #[test]
    fn test_sse_event_json_validity() {
        // RED: This should fail - testing JSON validity
        let event = create_message_start_event("msg_abc", "claude-3-sonnet-20240229");

        // Extract data portion
        let lines: Vec<&str> = event.lines().collect();
        let data_line = lines.iter().find(|l| l.starts_with("data: ")).unwrap();
        let json_str = &data_line[6..]; // Skip "data: " prefix

        // Should parse as valid JSON
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed["type"], "message_start");
        assert_eq!(parsed["message"]["id"], "msg_abc");
    }

    #[test]
    fn test_content_block_delta_structure() {
        // RED: This should fail - testing delta event structure
        let event = create_content_block_delta_event(0, "Hello");

        // Parse the JSON data
        let lines: Vec<&str> = event.lines().collect();
        let data_line = lines.iter().find(|l| l.starts_with("data: ")).unwrap();
        let json_str = &data_line[6..];
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();

        // Verify structure matches Anthropic spec
        assert_eq!(parsed["type"], "content_block_delta");
        assert_eq!(parsed["index"], 0);
        assert_eq!(parsed["delta"]["type"], "text_delta");
        assert_eq!(parsed["delta"]["text"], "Hello");
    }

    #[test]
    fn test_message_delta_with_usage() {
        // RED: This should fail - testing message_delta with usage info
        let event = create_message_delta_event("end_turn", 10, 5);

        // Parse the JSON data
        let lines: Vec<&str> = event.lines().collect();
        let data_line = lines.iter().find(|l| l.starts_with("data: ")).unwrap();
        let json_str = &data_line[6..];
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();

        // Verify structure
        assert_eq!(parsed["type"], "message_delta");
        assert_eq!(parsed["delta"]["stop_reason"], "end_turn");
        assert_eq!(parsed["usage"]["output_tokens"], 5);

        // Note: input_tokens typically not included in message_delta
        // Only output_tokens for the final count
    }

    #[test]
    fn test_message_stop_event() {
        // RED: This should fail - testing message_stop
        let event = create_message_stop_event();

        // Parse the JSON data
        let lines: Vec<&str> = event.lines().collect();
        let data_line = lines.iter().find(|l| l.starts_with("data: ")).unwrap();
        let json_str = &data_line[6..];
        let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();

        assert_eq!(parsed["type"], "message_stop");
    }
}
