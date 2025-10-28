# Pensieve API Documentation

**Complete API reference for the Pensieve Local LLM Server**

## Overview

Pensieve provides **full Anthropic API v1 compatibility**, enabling seamless integration with existing tools and applications designed for Claude AI. The API supports both streaming and non-streaming responses with identical request/response formats.

## Base URL

```
http://127.0.0.1:8080
```

## Authentication

Currently, Pensieve does not require authentication for local development. In production deployments, you can add authentication through reverse proxies or API gateway configurations.

## Endpoints

### Messages API

#### `POST /v1/messages`

Create a message completion using the specified model.

**Request Format:**

```json
{
  "model": "claude-3-sonnet-20240229",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Your message here"
    }
  ],
  "temperature": 0.7,
  "top_p": null,
  "stream": false,
  "system": null
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | ✅ | Model identifier (for compatibility) |
| `max_tokens` | integer | ✅ | Maximum number of tokens to generate |
| `messages` | array | ✅ | Array of message objects |
| `temperature` | float | ❌ | Sampling temperature (0.0-1.0) |
| `top_p` | float | ❌ | Nucleus sampling parameter |
| `stream` | boolean | ❌ | Enable streaming responses |
| `system` | string | ❌ | System prompt for the conversation |

**Message Object Format:**

```json
{
  "role": "user|assistant",
  "content": [
    {
      "type": "text",
      "text": "Message content"
    }
  ]
}
```

**Non-Streaming Response:**

```json
{
  "id": "msg_123456789",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Generated response text"
    }
  ],
  "model": "claude-3-sonnet-20240229",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 25,
    "output_tokens": 150
  }
}
```

**Streaming Response:**

When `stream: true`, the response is sent as Server-Sent Events (SSE):

```
data: {"type": "message_start"}

data: {"type": "content_block_start", "index": 0}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there!"}}

data: {"type": "content_block_stop", "index": 0}

data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}}

data: {"type": "message_stop"}
```

### Health Check

#### `GET /health`

Check the health and status of the server.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-10-28T12:00:00Z",
  "uptime_seconds": 3600,
  "version": "0.1.0",
  "model_loaded": false,
  "requests_processed": 150,
  "memory_usage_mb": 512
}
```

## Error Handling

Pensieve returns appropriate HTTP status codes and detailed error information:

### Error Response Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Detailed error description",
    "code": "validation_error"
  }
}
```

### Common Error Types

| HTTP Status | Error Type | Description |
|-------------|------------|-------------|
| 400 | `invalid_request_error` | Invalid request parameters or format |
| 401 | `authentication_error` | Authentication failed (future feature) |
| 429 | `rate_limit_error` | Rate limit exceeded (future feature) |
| 500 | `server_error` | Internal server error |
| 503 | `overloaded_error` | Server overloaded, try again later |

### Validation Errors

Common validation errors include:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "messages: required field missing",
    "code": "validation_error"
  }
}
```

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "max_tokens: must be between 1 and 8192",
    "code": "validation_error"
  }
}
```

## Usage Examples

### Python Client

```python
import requests
import json

# Non-streaming request
response = requests.post("http://127.0.0.1:8080/v1/messages",
    json={
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello, Pensieve!"}
        ]
    }
)

result = response.json()
print(result["content"][0]["text"])

# Streaming request
response = requests.post("http://127.0.0.1:8080/v1/messages",
    json={
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 100,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Tell me a story"}
        ]
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            print(data)
```

### cURL Examples

```bash
# Basic message request
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'

# Streaming request
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Write a poem"
      }
    ]
  }'

# Health check
curl http://127.0.0.1:8080/health
```

### JavaScript Client

```javascript
// Non-streaming request
async function sendMessage(message) {
  const response = await fetch('http://127.0.0.1:8080/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-3-sonnet-20240229',
      max_tokens: 100,
      messages: [
        { role: 'user', content: message }
      ]
    })
  });

  const result = await response.json();
  return result.content[0].text;
}

// Streaming request
async function streamMessage(message) {
  const response = await fetch('http://127.0.0.1:8080/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-3-sonnet-20240229',
      max_tokens: 100,
      stream: true,
      messages: [
        { role: 'user', content: message }
      ]
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        console.log(data);
      }
    }
  }
}
```

## Rate Limiting

Currently, Pensieve does not enforce strict rate limiting for local development. However, the following limits are recommended:

- **Concurrent Requests**: 100 (configurable)
- **Request Timeout**: 30 seconds (configurable)
- **Memory Usage**: Optimized for 16GB M1 constraints

## Streaming Implementation

### Server-Sent Events Format

Pensieve follows the SSE specification for streaming responses:

1. **message_start**: Indicates the beginning of a message
2. **content_block_start**: Starts a new content block
3. **content_block_delta**: Partial content update
4. **content_block_stop**: Ends a content block
5. **message_delta**: Final message metadata
6. **message_stop**: Completes the message

### Error Handling in Streams

Errors during streaming are sent as special events:

```
data: {"type": "error", "error": {"type": "server_error", "message": "Model loading failed"}}
```

Clients should handle these events appropriately.

## Model Information

### Supported Model Formats

- **GGUF**: Primary format with quantization support
- **Future**: SafeTensors (planned)

### Model Configuration

Models are configured through the CLI or configuration file:

```json
{
  "model": {
    "model_path": "/path/to/model.gguf",
    "model_type": "llama",
    "context_size": 2048,
    "gpu_layers": 32
  }
}
```

### Hardware Acceleration

- **Apple Silicon**: Metal GPU acceleration
- **CPU Fallback**: Automatic CPU-only operation
- **Memory Management**: Optimized for 16GB constraints

## Integration with Claude Code

Pensieve is designed to work seamlessly with Claude Code:

```bash
# Configure Claude Code to use Pensieve
export ANTHROPIC_API_KEY="unused"
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"

# Start Pensieve server
cargo run --bin pensieve -- start --model your-model.gguf

# Use Claude Code normally
claude "Hello from Claude Code!"
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the server is running on the correct port
2. **Model Loading Errors**: Check model file path and format
3. **Memory Issues**: Reduce context size or GPU layers
4. **Slow Responses**: Check GPU acceleration status

### Debug Mode

Enable verbose logging for debugging:

```bash
cargo run --bin pensieve -- --log-level debug start
```

### Health Monitoring

Regularly check the health endpoint:

```bash
curl http://127.0.0.1:8080/health
```

## Version History

- **v0.1.0**: Initial production release with full Anthropic API compatibility
- **v0.2.0**: Planned - Advanced model management and monitoring

---

**API Version**: v1
**Last Updated**: October 28, 2025
**Compatible with**: Anthropic Claude API v1