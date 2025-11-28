# D14: SSE Streaming Implementation Research

**Date**: 2025-10-30
**Status**: Research Complete - Ready for TDD Implementation
**Priority**: HIGH (Required for Step 5)
**Project**: Pensieve Anthropic Proxy - Server-Sent Events Streaming

---

## Executive Summary

This document comprehensively analyzes SSE (Server-Sent Events) streaming patterns from reference repositories (mlx-lm, vllm, text-generation-inference) to inform Step 5 TDD implementation of the Pensieve Anthropic Proxy. The research reveals exact Anthropic API SSE format, MLX streaming output patterns, and Rust/async integration strategies.

**Key Finding**: Anthropic SSE format uses a 6-event sequence with specific JSON structures. MLX streaming outputs tokens incrementally as JSON objects. Rust implementation requires Warp + Tokio for SSE response handling.

---

## 1. Anthropic SSE Streaming Format (Target Implementation)

### 1.1 Complete Event Sequence (from D12 and vllm source)

The Anthropic Messages API streaming response follows this exact sequence:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_xxx","type":"message","role":"assistant","content":[],"model":"phi-3-mini","stop_reason":null,"stop_sequence":null}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":42,"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

data: [DONE]
```

### 1.2 HTTP Headers for Streaming

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
Transfer-Encoding: chunked
```

**Critical Detail**: SSE requires `text/event-stream` MIME type (NOT `application/json`).

### 1.3 SSE Protocol Format

Each event follows this format:
```
event: <event_type>\n
data: <json_payload>\n
\n
```

**Rules**:
- `event:` line specifies event type
- `data:` line contains JSON payload
- Double newline (`\n\n`) terminates each event
- Multiple `data:` lines can be concatenated if JSON is multi-line
- Comments start with `:` and are ignored by clients

### 1.4 Event Types and Structures

#### `message_start`
Sent once at the beginning with initial message metadata:
```json
{
  "type": "message_start",
  "message": {
    "id": "msg_xxx",
    "type": "message",
    "role": "assistant",
    "content": [],
    "model": "phi-3-mini",
    "stop_reason": null,
    "stop_sequence": null,
    "usage": null
  }
}
```

#### `content_block_start`
Marks the start of content (index 0 for text):
```json
{
  "type": "content_block_start",
  "index": 0,
  "content_block": {
    "type": "text",
    "text": ""
  }
}
```

#### `content_block_delta`
Token-by-token text deltas (most frequent event):
```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "text_delta",
    "text": "Hello"
  }
}
```

#### `content_block_stop`
Marks the end of content block:
```json
{
  "type": "content_block_stop",
  "index": 0
}
```

#### `message_delta`
Final message metadata with stop reason and usage:
```json
{
  "type": "message_delta",
  "delta": {
    "stop_reason": "end_turn",
    "stop_sequence": null
  },
  "usage": {
    "input_tokens": 42,
    "output_tokens": 5,
    "cache_creation_input_tokens": null,
    "cache_read_input_tokens": null
  }
}
```

Stop reason values:
- `"end_turn"` - Model finished normally
- `"max_tokens"` - Hit max token limit
- `"stop_sequence"` - Matched stop sequence
- `"tool_use"` - Tool call initiated

#### `message_stop`
Final event signaling completion:
```json
{
  "type": "message_stop"
}
```

#### `[DONE]`
Final raw data marker: `data: [DONE]\n\n`

---

## 2. MLX Streaming Output Format

### 2.1 MLX-LM Server (Reference Implementation)

Location: `.doNotCommit/.refGitHubRepo/mlx-lm/mlx_lm/server.py` (lines 699-797)

The mlx-lm HTTP server streams using its own format (non-Anthropic):

```python
# From mlx_lm/server.py, handle_completion method
for gen_response in stream_generate(...):
    # Each iteration yields GenerationResponse:
    # - text: str (decoded text segment)
    # - token: int (token ID)
    # - logprobs: mx.array (log probabilities)
    # - from_draft: bool (speculative decoding flag)
    # - prompt_tokens: int
    # - prompt_tps: float
    # - generation_tokens: int
    # - generation_tps: float
    # - peak_memory: float (GB)
    # - finish_reason: Optional[str]
```

Key code from server.py lines 755-772:
```python
if self.stream and not in_tool_call:
    # SSE comment for keepalive during long processing
    self.wfile.write(f": keepalive {processed_tokens}/{total_tokens}\n\n".encode())
    self.wfile.flush()
    
    # Send OpenAI-style streaming chunks
    response = self.generate_response(segment, None, tool_calls=tool_calls)
    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
    self.wfile.flush()
```

**MLX Output Format**: OpenAI-style (not Anthropic), emits JSON objects with `delta` field containing text chunks.

### 2.2 Python Bridge MLX Streaming

Location: `python_bridge/mlx_inference.py` (lines 197-230)

The Pensieve MLX bridge uses mlx-lm's `stream_generate()` function:

```python
def real_mlx_generate(model_data, prompt, max_tokens, temperature, stream):
    if stream:
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            yield response.text  # Yields decoded text segment only
```

**Actual Output**: One text segment per iteration, e.g., `"Hello"`, `" there"`, etc.

### 2.3 MLX Stream Response Object

From `mlx_lm/generate.py`, lines 259-286:

```python
@dataclass
class GenerationResponse:
    text: str                      # Decoded text segment
    token: int                     # Token ID
    logprobs: mx.array            # Log probabilities
    from_draft: bool              # Speculative decoding flag
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float            # GB
    finish_reason: Optional[str]  # "stop" or "length"
```

**Key Feature**: Each response includes performance metrics (TPS, memory) that can be leveraged for monitoring.

---

## 3. SSE Streaming Patterns from Reference Repos

### 3.1 vllm Anthropic Implementation (BEST REFERENCE)

Location: `.doNotCommit/.refGitHubRepo/vllm/vllm/entrypoints/anthropic/serving_messages.py`

#### Helper Function (lines 43-44):
```python
def wrap_data_with_event(data: str, event: str):
    return f"event: {event}\ndata: {data}\n\n"
```

This is the canonical SSE wrapper for Anthropic events.

#### Stream Converter (lines 289-410):

The `message_stream_converter` method handles the transformation:

1. **First Item Handling** (lines 316-328):
   - Extract message ID from first chunk
   - Emit `message_start` event with empty content
   - Set `first_item = False` flag

2. **Content Block Start** (lines 364-374):
   - Track `content_block_started` flag
   - Emit `content_block_start` with empty text
   - Initialize `content_block_index = 0`

3. **Delta Streaming** (lines 376-388):
   - For each token, emit `content_block_delta`
   - Include text segment in delta
   - Skip empty content strings

4. **Content Block Stop** (lines 332-338):
   - When content ends, emit `content_block_stop`
   - Reset `content_block_started` flag

5. **Message Delta with Usage** (lines 342-355):
   - Emit `message_delta` with stop reason
   - Include token usage stats
   - Stop reason mapping:
     - `"stop"` → `"end_turn"`
     - `"length"` → `"max_tokens"`
     - `"tool_calls"` → `"tool_use"`

6. **Final Message Stop** (lines 303-310):
   - Emit `message_stop` event
   - Send raw `data: [DONE]\n\n`

#### State Machine Summary:
```
[First Chunk] → message_start
    ↓
[Content Start] → content_block_start
    ↓
[Each Token] → content_block_delta (repeated)
    ↓
[Content End] → content_block_stop
    ↓
[Final Chunk] → message_delta (with usage)
    ↓
[Completion] → message_stop
    ↓
[Done] → [DONE]
```

### 3.2 mlx-lm Server Streaming Pattern

From `server.py`, lines 683-797:

#### Keepalive Mechanism (lines 683-697):
```python
def keepalive_callback(processed_tokens, total_tokens):
    logging.info(f"Prompt processing progress: {processed_tokens}/{total_tokens}")
    if self.stream:
        try:
            # SSE comments are invisible to clients but keep connection alive
            self.wfile.write(f": keepalive {processed_tokens}/{total_tokens}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected
```

**Key Insight**: Use SSE comments (`:` prefix) for keepalive during long prompt processing without confusing clients.

#### Error Handling During Stream (lines 299-309):
```python
if self.stream:
    self._set_stream_headers(400)
    self.wfile.write(
        f"data: {json.dumps({'error': 'Invalid JSON'})}\n\n".encode()
    )
else:
    self._set_completion_headers(400)
    self.wfile.write(json.dumps({"error": "..."}))
```

**Pattern**: Send errors as SSE data events even in streaming mode, using proper event-stream headers.

---

## 4. Rust Integration Strategy

### 4.1 Current Pensieve Architecture

**Current Code** (pensieve-09-anthropic-proxy/src/server.rs, lines 252-299):

```rust
async fn call_mlx_bridge(config: &ServerConfig, mlx_request: &MlxRequest) -> Result<String, ServerError> {
    use tokio::process::Command;

    let mut cmd = Command::new("python3");
    cmd.arg(&config.python_bridge_path)
        .arg("--model-path").arg(&config.model_path)
        .arg("--prompt").arg(&mlx_request.prompt)
        .arg("--max-tokens").arg(mlx_request.max_tokens.to_string())
        .arg("--temperature").arg(mlx_request.temperature.to_string());

    let output = cmd.output().await?;  // BLOCKING - waits for full output

    // Parse JSON responses (currently only gets final output)
    for line in stdout.lines() {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = json.get("text") {
                return Ok(text.to_string());
            }
        }
    }
}
```

**Current Limitation**: Uses `.output()` which waits for process completion. Cannot stream in real-time.

### 4.2 Streaming Implementation Pattern

#### Option A: `tokio::process::Command` with Streaming stdout

```rust
async fn call_mlx_bridge_streaming(
    config: &ServerConfig,
    mlx_request: &MlxRequest,
) -> Result<impl Stream<Item = Result<String, ServerError>>, ServerError> {
    use tokio::process::Command;
    use tokio::io::{AsyncBufReadExt, BufReader};
    
    let mut child = Command::new("python3")
        .arg(&config.python_bridge_path)
        .arg("--model-path").arg(&config.model_path)
        .arg("--prompt").arg(&mlx_request.prompt)
        .arg("--max-tokens").arg(mlx_request.max_tokens.to_string())
        .arg("--temperature").arg(mlx_request.temperature.to_string())
        .arg("--stream")  // Add streaming flag
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take()
        .ok_or_else(|| ServerError::Internal("No stdout".to_string()))?;
    
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    // Returns stream of JSON objects from Python bridge
    Ok(async_stream::stream! {
        while let Some(line) = lines.next_line().await? {
            if !line.trim().is_empty() {
                match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(json) => yield Ok(line),
                    Err(e) => yield Err(ServerError::Internal(format!("JSON parse error: {}", e))),
                }
            }
        }
    })
}
```

**Key Points**:
- Use `Stdio::piped()` to capture stdout
- `BufReader` + `.lines()` reads line-by-line
- `async_stream` crate provides convenient stream macro
- Yields JSON objects as they arrive

#### Option B: Warp SSE Response

```rust
use warp::{Reply, reply, sse};

fn create_sse_response(
    stream: impl Stream<Item = Result<String, ServerError>> + Send + 'static
) -> impl Reply {
    let events = stream.then(|line_result| async move {
        match line_result {
            Ok(json_line) => {
                // Parse Python output
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&json_line) {
                    if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                        // Convert to Anthropic event
                        let event = AnthropicStreamEvent {
                            type_: "content_block_delta",
                            index: 0,
                            delta: AnthropicDelta {
                                type_: "text_delta",
                                text: text.to_string(),
                            },
                            ..Default::default()
                        };
                        
                        return Ok::<_, Infallible>(sse::Event::default()
                            .event("content_block_delta")
                            .json_data(event)
                            .unwrap());
                    }
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                // Send error event
                let event = AnthropicStreamEvent {
                    type_: "error",
                    error: Some(AnthropicError {
                        type_: "internal_error".to_string(),
                        message: e.to_string(),
                    }),
                    ..Default::default()
                };
                
                return Ok::<_, Infallible>(sse::Event::default()
                    .event("error")
                    .json_data(event)
                    .unwrap());
            }
        }
        
        // Fallback empty event
        Ok(sse::Event::default())
    });

    reply::with_status(
        sse::reply(events),
        warp::http::StatusCode::OK
    )
}
```

**Key Points**:
- `warp::sse` module handles SSE response formatting
- `Event::default().event("name").json_data(obj)` creates proper SSE events
- Stream must be `Send + 'static`
- Automatically handles `Content-Type: text/event-stream`

### 4.3 Complete Request Handler (Streaming)

```rust
async fn handle_messages_streaming(
    auth_header: Option<String>,
    request: CreateMessageRequest,
    config: ServerConfig,
) -> Result<impl Reply, warp::Rejection> {
    // 1. Auth validation
    if let Err(auth_error) = validate_auth(auth_header.as_deref()) {
        return Ok(error_response("authentication_error", "Invalid API key", 401));
    }

    // 2. Request validation
    if let Err(e) = request.validate() {
        return Ok(error_response("invalid_request_error", &e.to_string(), 400));
    }

    // 3. Translate to MLX
    let mlx_request = match translate_anthropic_to_mlx(&request) {
        Ok(req) => req,
        Err(e) => return Ok(error_response("internal_error", &e.to_string(), 500)),
    };

    // 4. Create streaming response
    let response = create_anthropic_streaming_response(config, mlx_request).await;
    
    Ok(response)
}

async fn create_anthropic_streaming_response(
    config: ServerConfig,
    mlx_request: MlxRequest,
) -> impl Reply {
    let stream = async move {
        // Create stream from Python bridge
        match call_mlx_bridge_streaming(&config, &mlx_request).await {
            Ok(python_stream) => {
                // Emit message_start
                yield create_event("message_start", json!({
                    "type": "message_start",
                    "message": {
                        "id": format!("msg_{}", uuid::Uuid::new_v4()),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "phi-3-mini",
                    }
                })).unwrap();

                // Emit content_block_start
                yield create_event("content_block_start", json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                })).unwrap();

                let mut content_started = true;
                let mut total_output_tokens = 0;

                // Stream tokens
                pin_mut!(python_stream);
                while let Some(Ok(line)) = python_stream.next().await {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
                            total_output_tokens += 1;
                            
                            // Emit content_block_delta
                            yield create_event("content_block_delta", json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {
                                    "type": "text_delta",
                                    "text": text
                                }
                            })).unwrap();
                        }
                    }
                }

                // Emit content_block_stop
                yield create_event("content_block_stop", json!({
                    "type": "content_block_stop",
                    "index": 0
                })).unwrap();

                // Emit message_delta with usage
                yield create_event("message_delta", json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn"
                    },
                    "usage": {
                        "input_tokens": estimate_tokens(&mlx_request.prompt),
                        "output_tokens": total_output_tokens
                    }
                })).unwrap();

                // Emit message_stop
                yield create_event("message_stop", json!({
                    "type": "message_stop"
                })).unwrap();

                // Final [DONE]
                yield sse::Event::default().data("[DONE]");
            }
            Err(e) => {
                // Error event
                yield create_event("error", json!({
                    "type": "error",
                    "error": {
                        "type": "internal_error",
                        "message": e.to_string()
                    }
                })).unwrap();
            }
        }
    };

    reply::with_status(
        sse::reply(stream),
        warp::http::StatusCode::OK
    )
}

fn create_event(event_type: &str, data: serde_json::Value) -> Result<sse::Event, String> {
    Ok(sse::Event::default()
        .event(event_type)
        .data(serde_json::to_string(&data).map_err(|e| e.to_string())?))
}
```

---

## 5. Token Counting Strategy

### 5.1 Input Token Estimation

MLX bridge doesn't return token counts directly, so we estimate:

```rust
fn estimate_tokens(text: &str) -> u32 {
    // Simple: split by whitespace
    // More accurate: use tokenizer from Phi-3
    (text.split_whitespace().count() as u32).max(1)
}

// For accurate counting (requires tokenizer access):
fn count_tokens_accurate(text: &str, tokenizer: &Tokenizer) -> u32 {
    tokenizer.encode(text, false).unwrap().len() as u32
}
```

### 5.2 Output Token Counting

Stream each token from Python bridge and count:

```rust
let mut output_token_count = 0;
while let Some(Ok(line)) = python_stream.next().await {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
        if let Some(text) = json.get("text").and_then(|v| v.as_str()) {
            output_token_count += 1;  // Each JSON output = 1 token from MLX
            // Emit delta with this token
        }
    }
}
```

**Note**: MLX's `stream_generate()` already handles tokenization, so each iteration = 1 token.

---

## 6. Error Handling in Streams

### 6.1 Mid-Stream Error Recovery

```rust
async fn handle_python_process_error(error: ServerError) -> sse::Event {
    eprintln!("Stream error: {:?}", error);
    
    sse::Event::default()
        .event("error")
        .data(json!({
            "type": "error",
            "error": {
                "type": "internal_error",
                "message": error.to_string()
            }
        }).to_string())
}
```

### 6.2 Client Disconnection Detection

```rust
async fn create_streaming_response(...) -> impl Reply {
    let stream = async move {
        // ... setup ...
        
        while let Some(result) = python_stream.next().await {
            match result {
                Ok(line) => {
                    // Process and yield event
                    yield process_line(line);
                }
                Err(e) => {
                    eprintln!("Python bridge error: {}", e);
                    yield create_error_event("Process error", &e.to_string());
                    break;  // Exit stream
                }
            }
        }
    };
    
    sse::reply(stream)
}
```

### 6.3 Timeout Handling

```rust
use tokio::time::{timeout, Duration};

let stream = async move {
    // Apply timeout to entire streaming operation
    match timeout(Duration::from_secs(300), stream_inner()).await {
        Ok(result) => result,
        Err(_) => {
            eprintln!("Streaming timeout");
            yield create_error_event("timeout", "Request timed out after 5 minutes");
        }
    }
};
```

---

## 7. Python Bridge Modifications for Streaming

### 7.1 Add Streaming Mode to Python Bridge

Current `mlx_inference.py` needs `--stream` flag integration:

```python
def main():
    parser = argparse.ArgumentParser(description="Pensieve MLX Inference Bridge")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    
    args = parser.parse_args()
    
    for response in generate_text(
        args.model_path,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.stream  # Pass streaming flag
    ):
        # Output JSON line per token
        json.dump(response, sys.stdout)
        print()  # Newline between JSON objects
        sys.stdout.flush()  # Ensure real-time output
```

### 7.2 Streaming Output Format

Each line is a JSON object (one per token):

```json
{"type": "text_chunk", "text": "Hello", "accumulated": "Hello", "tokens_per_second": 15.3, "elapsed_ms": 65}
{"type": "text_chunk", "text": " there", "accumulated": "Hello there", "tokens_per_second": 14.8, "elapsed_ms": 130}
...
```

The Rust handler extracts the `"text"` field from each JSON object.

---

## 8. Testing Strategy for Streaming

### 8.1 Unit Tests (Rust)

```rust
#[tokio::test]
async fn test_sse_event_formatting() {
    let event = create_event("content_block_delta", json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": {
            "type": "text_delta",
            "text": "Hello"
        }
    })).unwrap();
    
    let formatted = event.to_string();
    assert!(formatted.contains("event: content_block_delta"));
    assert!(formatted.contains("text_delta"));
    assert!(formatted.contains("Hello"));
}

#[tokio::test]
async fn test_stream_event_sequence() {
    // Mock Python process
    // Verify: message_start → content_block_start → content_block_delta* → 
    //         content_block_stop → message_delta → message_stop
}
```

### 8.2 Integration Tests

```rust
#[tokio::test]
async fn test_streaming_request_with_mock_mlx() {
    let server = AnthropicProxyServer::new(ServerConfig::default());
    server.start().await.unwrap();

    let request = CreateMessageRequest {
        model: "phi-3-mini".to_string(),
        max_tokens: 100,
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::String("Hello".to_string()),
        }],
        stream: Some(true),
        ..Default::default()
    };

    // Send request
    let response = reqwest::Client::new()
        .post("http://127.0.0.1:7777/v1/messages")
        .json(&request)
        .send()
        .await
        .unwrap();

    // Parse SSE events
    let body = response.text().await.unwrap();
    let events: Vec<&str> = body.split("event: ").collect();
    
    assert!(events.iter().any(|e| e.contains("message_start")));
    assert!(events.iter().any(|e| e.contains("content_block_delta")));
    assert!(events.iter().any(|e| e.contains("message_stop")));
}
```

### 8.3 Manual Testing

```bash
# Test streaming endpoint
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3-mini",
    "max_tokens": 100,
    "messages": [{"role":"user","content":"Hello"}],
    "stream": true
  }' -v
```

Expected output:
```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

...
```

---

## 9. Performance Considerations

### 9.1 Metrics from MLX

MLX's `stream_generate()` provides timing data per token:

```python
@dataclass
class GenerationResponse:
    prompt_tps: float        # Tokens per second for prompt processing
    generation_tps: float    # Tokens per second for generation
    peak_memory: float       # Peak memory in GB
```

Can include in SSE for monitoring:

```rust
// Optional: Include performance in delta if needed
json!({
    "type": "content_block_delta",
    "index": 0,
    "delta": {
        "type": "text_delta",
        "text": "Hello",
        "metrics": {  // Optional, non-standard
            "generation_tps": 15.3,
            "peak_memory_mb": 2100
        }
    }
})
```

### 9.2 Memory Efficiency

- **Chunked Streaming**: Don't buffer entire response
- **Line-by-Line Reading**: Use `BufReader` to avoid memory spikes
- **Drop Processed Events**: Don't keep emitted events in memory

### 9.3 Throughput

Measured performance from current implementation:
- **Current (non-streaming)**: ~16.85 TPS
- **Target**: 25+ TPS
- **Streaming Overhead**: Minimal (<5ms per event with Warp)

---

## 10. Implementation Checklist for Step 5

- [ ] Add `--stream` flag to Python bridge (`mlx_inference.py`)
- [ ] Modify `call_mlx_bridge` to use `tokio::process::Command` with streaming stdout
- [ ] Create `call_mlx_bridge_streaming()` function returning async stream
- [ ] Create SSE event formatters:
  - [ ] `create_message_start_event()`
  - [ ] `create_content_block_start_event()`
  - [ ] `create_content_block_delta_event()`
  - [ ] `create_content_block_stop_event()`
  - [ ] `create_message_delta_event()`
  - [ ] `create_message_stop_event()`
- [ ] Implement `handle_messages_streaming()` handler
- [ ] Create state machine for event sequence management
- [ ] Add token counting (input estimation + output counting)
- [ ] Add error event handling
- [ ] Add timeout handling (5 min default)
- [ ] Update route handler to detect `stream: true` parameter
- [ ] Write TDD tests:
  - [ ] Event format validation
  - [ ] Event sequence verification
  - [ ] Token counting accuracy
  - [ ] Error propagation
  - [ ] Client disconnection handling
- [ ] Integration test with mock MLX process
- [ ] Manual curl testing

---

## 11. Key File Locations

### Reference Implementations:
- **vllm Anthropic Protocol**: `/vllm/entrypoints/anthropic/protocol.py`
- **vllm Streaming Handler**: `/vllm/entrypoints/anthropic/serving_messages.py`
- **mlx-lm Server**: `/mlx-lm/mlx_lm/server.py`
- **mlx-lm Generate**: `/mlx-lm/mlx_lm/generate.py`

### Pensieve Files to Modify:
- **Python Bridge**: `python_bridge/mlx_inference.py`
- **Server Handler**: `pensieve-09-anthropic-proxy/src/server.rs`
- **API Models**: `pensieve-03/src/lib.rs` (may need streaming event types)
- **Tests**: `pensieve-09-anthropic-proxy/tests/` (new streaming test files)

---

## 12. Estimated Effort

- **Rust Implementation**: 6-8 hours
  - Streaming stdout handling: 1-2 hours
  - SSE event formatting: 1 hour
  - State machine + token counting: 2-3 hours
  - Error handling: 1 hour
  - Testing: 1-2 hours

- **Python Bridge Modifications**: 1-2 hours
  - Add `--stream` flag support: 30 min
  - JSON output per token: 30 min
  - Testing: 30 min

- **Total**: ~8-10 hours (1-1.5 days)

---

## 13. References and Further Reading

1. **Anthropic API Docs**: https://docs.anthropic.com/en/api/messages-streaming
2. **Server-Sent Events MDN**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
3. **Warp SSE Module**: https://github.com/seanmonstar/warp/blob/master/src/filters/sse.rs
4. **vllm Anthropic Adapter**: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/
5. **MLX-LM Server**: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py

---

## Appendix A: Complete SSE Wire Format Example

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Transfer-Encoding: chunked

event: message_start
data: {"type":"message_start","message":{"id":"msg_1730307600000","type":"message","role":"assistant","content":[],"model":"phi-3-mini"}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text","!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

data: [DONE]
```

---

**Document Created**: 2025-10-30  
**Ready for TDD Implementation**: YES  
**Next Step**: Create Step 5 TDD test file based on this research
