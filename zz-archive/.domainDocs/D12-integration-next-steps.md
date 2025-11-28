# D12: Claude Code Integration - Next Steps & TDD Action Plan

**Date**: October 29, 2025
**Status**: Ready for Implementation
**Priority**: HIGH
**Confidence**: 95% (Validated by two production implementations)

---

## Executive Summary

### Current Status: Where We Are

**✅ What Works:**
- Pensieve local inference engine (phi-3-mini works)
- HTTP API endpoint responds correctly (curl tests pass)
- Basic request/response structure matches Anthropic format

**❌ What's Missing:**
- Anthropic SDK expects specific header handling
- Claude Code fails to authenticate properly with current setup
- No timeout configuration for long-running local inference
- No request/response translation layer
- No streaming SSE event support
- No intelligent routing

**🎯 Root Cause:**
From D10 & D11 research: **Claude Code isn't failing because of our API - it's failing because we lack a translation/proxy layer.** The Anthropic SDK has specific requirements that our direct endpoint doesn't satisfy.

---

## 1. What Claude Code Expects: API Specification

### 1.1 Core Requirements

**Authentication:**
- Expects header: `Authorization: Bearer {token}`
- Claude Code passes via: `ANTHROPIC_AUTH_TOKEN` environment variable
- **Key insight from D10 (line 99)**: Use `ANTHROPIC_AUTH_TOKEN` not `ANTHROPIC_API_KEY`
- Can accept ANY token for local development (D10, line 195-202)

**Base URL:**
- Environment variable: `ANTHROPIC_BASE_URL`
- Format: `http://127.0.0.1:7777` (no `/v1` suffix)
- Claude Code's SDK appends `/v1/messages` automatically

**Timeout:**
- Default: 60 seconds
- Local inference needs: 3,000,000ms (50 minutes) as per D10, line 101
- Set via: `API_TIMEOUT_MS` environment variable

### 1.2 Request Format

**POST /v1/messages**

```json
{
  "model": "claude-3-sonnet-20240229",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hello, Claude!"
        }
      ]
    }
  ],
  "stream": false,
  "temperature": 0.7,
  "system": "You are a helpful assistant."
}
```

**From D10 (lines 207-227):** Shows exact request/response structure expected.

### 1.3 Response Format (Non-Streaming)

```json
{
  "id": "msg_123abc",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "{response_text}"
    }
  ],
  "model": "claude-3-sonnet-20240229",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 42
  }
}
```

**From D10 (lines 243-262):** Shows exact response structure.

### 1.4 Streaming Response Format

**Server-Sent Events (SSE) with exact Anthropic format:**

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant"}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}

event: message_stop
data: {"type":"message_stop"}
```

**From D10 (lines 264-288):** Shows exact SSE event sequence.
**From D11 (lines 374-423):** Shows SSEParser implementation to parse this format.

### 1.5 Critical Settings File Configuration

**File**: `~/.claude/settings.json`

**Required Content:**
```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "pensieve-local-token",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:7777",
    "API_TIMEOUT_MS": "3000000"
  },
  "alwaysThinkingEnabled": true
}
```

**Key insight from D10 (lines 142-180):**
- Must use Node.js for safe JSON manipulation (not bash)
- Must NOT include `/v1` in base URL
- Timeout is critical for local inference

---

## 2. Specific Codebase References

### 2.1 Current HTTP Server Implementation

**File**: `/Users/amuldotexe/Projects/pensieve-local-llm-server/pensieve-02/src/lib.rs`
- **Status**: Accepts requests, but missing translation layer
- **Issue**: No proxy/translation of Anthropic format to MLX

### 2.2 API Models

**File**: `/Users/amuldotexe/Projects/pensieve-local-llm-server/pensieve-03/src/lib.rs`
- **Status**: May need enhancement for full Anthropic spec compliance
- **Action**: Verify all fields from Section 1.2-1.4 above are supported

### 2.3 MLX Python Bridge

**File**: `/Users/amuldotexe/Projects/pensieve-local-llm-server/python_bridge/mlx_inference.py`
- **Usage**: Will be called by proxy layer
- **Input Format**: Needs translation from Anthropic to MLX format
- **Output Format**: Will be translated back to Anthropic format

### 2.4 CLI Interface

**File**: `/Users/amuldotexe/Projects/pensieve-local-llm-server/pensieve-01/src/main.rs`
- **Enhancement needed**: Installation script for settings.json

### 2.5 Workspace Cargo.toml

**File**: `/Users/amuldotexe/Projects/pensieve-local-llm-server/Cargo.toml`
- **Action**: Will need to add new `pensieve-09-anthropic-proxy` crate

---

## 3. The Three Production Implementations

| Feature | z.ai (D10) | claude-code-router (D11) | Pensieve (Target) |
|---------|-----------|-------------------------|-------------------|
| **Integration Pattern** | Settings.json + timeout | Env vars + launcher | Env vars + launcher + proxy |
| **Authentication Handling** | ANTHROPIC_AUTH_TOKEN | ANTHROPIC_AUTH_TOKEN override | Same as z.ai |
| **Model Mapping** | GLM-4.6 ← claude-* | Provider selection | phi-3-mini ← claude-* |
| **Request Translation** | ✅ Proxy layer | ✅ Transformer pipeline | ⏳ Need to build |
| **Response Translation** | ✅ Proxy layer | ✅ Transformer pipeline | ⏳ Need to build |
| **SSE Streaming** | ✅ Fully supported | ✅ Fully supported | ⏳ Need to build |
| **Timeout Handling** | 50 minutes (D10, line 101) | 10 min default | Need to implement |
| **Intelligent Routing** | Basic | Advanced (D11, lines 199-235) | Will implement Phase 2 |
| **Testing** | Unknown | None ❌ (CRITICAL GAP) | ✅ TDD-first |

---

## 4. 🎯 Next Steps: TDD-Based Implementation

### Foundation: Testing First (Per S01 Principles)

**Core principle from S01:**
> "Test-First Development: Write tests FIRST, following the STUB → RED → GREEN → REFACTOR cycle"

Every step below includes tests BEFORE implementation.

---

### **STEP 1: Authentication Handler with TDD** (Days 1-2)

**Goal**: `claude --print "hello"` returns 401 for invalid auth, 200 for valid auth

**Test First (RED):**

```rust
// File: pensieve-09-anthropic-proxy/src/auth.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_auth_header_fails() {
        let result = validate_auth(None);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().status, 401);
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
    fn test_invalid_token_fails() {
        let result = validate_auth(Some("invalid-token-xyz"));
        assert!(result.is_err());
    }
}
```

**Implementation (GREEN):**
- Create new crate: `pensieve-09-anthropic-proxy`
- Implement `validate_auth()` function
- Return proper HTTP 401 responses
- Accept both local and Anthropic format tokens

**Reference Implementation**: D10, lines 193-202

**Files to create:**
- `/Users/amuldotexe/Projects/pensieve-local-llm-server/pensieve-09-anthropic-proxy/Cargo.toml`
- `/Users/amuldotexe/Projects/pensieve-local-llm-server/pensieve-09-anthropic-proxy/src/lib.rs`
- `/Users/amuldotexe/Projects/pensieve-local-llm-server/pensieve-09-anthropic-proxy/src/auth.rs`

**Success Criterion:**
```bash
# Invalid auth
curl -X POST http://127.0.0.1:7777/v1/messages \
  -d '{"model":"claude-3-sonnet-20240229","messages":[{"role":"user","content":"hi"}]}'
# Returns: 401 Unauthorized

# Valid auth
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Authorization: Bearer pensieve-local-token" \
  -d '{...}'
# Returns: 200 or processes request
```

---

### **STEP 2: Request Translation (Anthropic → MLX) with TDD** (Days 3-4)

**Goal**: Convert Anthropic format messages to MLX-compatible prompt format

**Test First (RED):**

```rust
// File: pensieve-09-anthropic-proxy/src/translator.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_user_message_translation() {
        let request = AnthropicRequest {
            model: "claude-3-sonnet-20240229".into(),
            messages: vec![Message {
                role: "user".into(),
                content: vec![ContentBlock::Text {
                    text: "Hello, Claude!".into(),
                }],
            }],
            system: None,
            max_tokens: 100,
            temperature: 0.7,
            stream: Some(false),
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // Postconditions
        assert!(mlx_request.prompt.contains("User: Hello, Claude!"));
        assert_eq!(mlx_request.max_tokens, 100);
        assert_eq!(mlx_request.temperature, 0.7);
    }

    #[test]
    fn test_system_prompt_included() {
        let request = AnthropicRequest {
            system: Some("You are a helpful assistant.".into()),
            messages: vec![Message {
                role: "user".into(),
                content: vec![ContentBlock::Text {
                    text: "Hello".into(),
                }],
            }],
            // ... rest
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        assert!(mlx_request.prompt.contains("System: You are a helpful assistant."));
    }

    #[test]
    fn test_multi_turn_conversation() {
        let request = AnthropicRequest {
            messages: vec![
                Message {
                    role: "user".into(),
                    content: vec![ContentBlock::Text {
                        text: "What is 2+2?".into(),
                    }],
                },
                Message {
                    role: "assistant".into(),
                    content: vec![ContentBlock::Text {
                        text: "4".into(),
                    }],
                },
                Message {
                    role: "user".into(),
                    content: vec![ContentBlock::Text {
                        text: "Correct! What is 3+3?".into(),
                    }],
                },
            ],
            // ... rest
        };

        let mlx_request = translate_anthropic_to_mlx(&request).unwrap();

        // Verify conversation structure
        assert!(mlx_request.prompt.contains("What is 2+2?"));
        assert!(mlx_request.prompt.contains("4"));
        assert!(mlx_request.prompt.contains("What is 3+3?"));
    }
}
```

**Implementation (GREEN):**
- Implement `translate_anthropic_to_mlx()` function
- Handle system prompts correctly
- Convert message role/content to MLX prompt format
- Preserve temperature and max_tokens settings

**Reference Implementation**: D10, lines 230-240

**Success Criterion:**
```bash
# Send Anthropic format request
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Authorization: Bearer pensieve-local-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"claude-3-sonnet-20240229",
    "max_tokens":100,
    "messages":[{"role":"user","content":"Say hello in 5 words"}]
  }'

# Returns: Some response (may not be perfect yet, but translation works)
```

---

### **STEP 3: Response Translation (MLX → Anthropic) with TDD** (Days 5-6)

**Goal**: Convert MLX output to Anthropic response format with correct tokens counting

**Test First (RED):**

```rust
// File: pensieve-09-anthropic-proxy/src/translator.rs (extend)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlx_output_to_anthropic_response() {
        let mlx_output = "Hello! How can I help you?";
        let input_tokens = 15;
        let output_tokens = 8;

        let response = translate_mlx_to_anthropic(
            mlx_output,
            input_tokens,
            output_tokens,
        );

        // Postconditions
        assert_eq!(response.role, "assistant");
        assert_eq!(response.type_, "message");
        assert_eq!(response.content.len(), 1);

        if let ContentBlock::Text { text, .. } = &response.content[0] {
            assert_eq!(text, mlx_output);
        } else {
            panic!("Expected text content");
        }

        assert_eq!(response.usage.input_tokens, 15);
        assert_eq!(response.usage.output_tokens, 8);
        assert_eq!(response.stop_reason, Some("end_turn".into()));
    }

    #[test]
    fn test_response_has_valid_message_id() {
        let response = translate_mlx_to_anthropic("test", 1, 1);

        assert!(response.id.starts_with("msg_"));
        assert!(response.id.len() > 4); // Should include UUID
    }

    #[test]
    fn test_response_model_mapping() {
        let response = translate_mlx_to_anthropic("test", 1, 1);

        // Should map to Claude model name
        assert!(response.model.contains("claude-3"));
    }
}
```

**Implementation (GREEN):**
- Implement `translate_mlx_to_anthropic()` function
- Generate unique message IDs (UUID-based)
- Map response to Anthropic model names (claude-3-sonnet-*)
- Implement token counting

**Reference Implementation**: D10, lines 243-262

**Success Criterion:**
```bash
# Full request/response cycle
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Authorization: Bearer pensieve-local-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"claude-3-sonnet-20240229",
    "max_tokens":100,
    "messages":[{"role":"user","content":"Say hello"}]
  }' | jq .

# Returns valid Anthropic format with id, role, content, usage
```

---

### **STEP 4: Integration Script with Settings File** (Days 7-8)

**Goal**: One-command setup that updates `~/.claude/settings.json` safely

**Test First (RED):**

```bash
#!/bin/bash
# File: scripts/test-setup.sh

# This is an integration test (runs actual script and verifies)
set -e

# 1. Back up existing settings.json if present
if [ -f "$HOME/.claude/settings.json" ]; then
    cp "$HOME/.claude/settings.json" "$HOME/.claude/settings.json.backup"
fi

# 2. Run setup script
./scripts/setup-claude-code.sh

# 3. Verify settings.json was created
[ -f "$HOME/.claude/settings.json" ] || {
    echo "FAIL: settings.json not created"
    exit 1
}

# 4. Verify required fields exist
node --eval "
const fs = require('fs');
const settings = JSON.parse(fs.readFileSync('$HOME/.claude/settings.json'));

// Postconditions
if (!settings.env.ANTHROPIC_AUTH_TOKEN) {
    console.error('FAIL: Missing ANTHROPIC_AUTH_TOKEN');
    process.exit(1);
}

if (!settings.env.ANTHROPIC_BASE_URL.includes('127.0.0.1:7777')) {
    console.error('FAIL: ANTHROPIC_BASE_URL incorrect');
    process.exit(1);
}

if (parseInt(settings.env.API_TIMEOUT_MS) < 3000000) {
    console.error('FAIL: API_TIMEOUT_MS too low');
    process.exit(1);
}

console.log('✅ All postconditions met');
"

# 5. Restore backup
if [ -f "$HOME/.claude/settings.json.backup" ]; then
    mv "$HOME/.claude/settings.json.backup" "$HOME/.claude/settings.json"
fi
```

**Implementation (GREEN):**

```bash
#!/bin/bash
# File: scripts/setup-claude-code.sh
# Safe setup script using Node.js (from D10, lines 142-180)

set -e

SETTINGS_FILE="$HOME/.claude/settings.json"

echo "🔧 Setting up Claude Code for Pensieve local server..."

# Step 1: Create onboarding bypass (from D10, lines 122-136)
echo "  → Creating onboarding bypass..."
cat > "$HOME/.claude.json" << 'EOF'
{
  "hasCompletedOnboarding": true
}
EOF

# Step 2: Update settings.json using Node.js (safe JSON manipulation)
echo "  → Updating settings.json..."
node --eval "
const fs = require('fs');
const path = '$SETTINGS_FILE';

// Load existing or create new
let content;
try {
    content = JSON.parse(fs.readFileSync(path, 'utf-8'));
} catch {
    content = {};
}

// Update with Pensieve config
const updated = {
    ...content,
    env: {
        ANTHROPIC_AUTH_TOKEN: 'pensieve-local-token',
        ANTHROPIC_BASE_URL: 'http://127.0.0.1:7777',
        API_TIMEOUT_MS: '3000000'
    },
    alwaysThinkingEnabled: true
};

// Write back safely
fs.writeFileSync(path, JSON.stringify(updated, null, 2), 'utf-8');
console.log('  ✅ settings.json updated');
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start Pensieve server: pensieve start"
echo "  2. Test Claude Code: claude --print 'Say hello'"
echo ""
```

**Files to create:**
- `/Users/amuldotexe/Projects/pensieve-local-llm-server/scripts/setup-claude-code.sh`
- `/Users/amuldotexe/Projects/pensieve-local-llm-server/scripts/test-setup.sh`

**Success Criterion:**
```bash
# Run setup
./scripts/setup-claude-code.sh
# Output: ✅ Setup complete!

# Verify settings
cat ~/.claude/settings.json | jq .env
# Output shows: ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, API_TIMEOUT_MS
```

---

### **STEP 5: Streaming SSE Support** (Days 9-10)

**Goal**: Real-time streaming with exact Anthropic SSE format (See Section 1.4)

**Test First (RED):**

```rust
// File: pensieve-09-anthropic-proxy/src/streaming.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_event_stream_sequence() {
        let tokens = vec!["Hello", " ", "there", "!"];
        let message_id = "msg_test123";

        let events = generate_sse_events(&tokens, message_id).collect::<Vec<_>>();

        // Postconditions: Verify event sequence
        assert!(events[0].contains("event: message_start"));
        assert!(events[1].contains("event: content_block_start"));

        // Should have content_block_delta for each token
        let delta_events: Vec<_> = events.iter()
            .filter(|e| e.contains("content_block_delta"))
            .collect();
        assert_eq!(delta_events.len(), tokens.len());

        // Last events should be stop events
        assert!(events.last().unwrap().contains("message_stop"));
    }

    #[test]
    fn test_sse_event_json_validity() {
        let events = generate_sse_events(&vec!["test"], "msg_123").collect::<Vec<_>>();

        // Verify each event's data is valid JSON
        for event in events {
            if let Some(data_line) = event.lines().find(|l| l.starts_with("data: ")) {
                let json_str = &data_line[6..]; // Remove "data: " prefix
                let parsed: serde_json::Result<serde_json::Value> =
                    serde_json::from_str(json_str);
                assert!(parsed.is_ok(), "Invalid JSON: {}", json_str);
            }
        }
    }

    #[test]
    fn test_streaming_request_response() {
        let request = AnthropicRequest {
            stream: Some(true),
            // ... rest of fields
        };

        // Should return SSE stream, not JSON
        let response = handle_streaming_request(&request);
        assert!(response.headers().get("content-type")
            .map(|v| v.to_str().unwrap_or(""))
            .unwrap_or("")
            .contains("text/event-stream"));
    }
}
```

**Implementation (GREEN):**
- Implement `generate_sse_events()` function
- Convert MLX token stream to exact Anthropic SSE format
- Handle streaming request detection
- Return proper `Content-Type: text/event-stream` header

**Reference Implementation**:
- D10, lines 264-288 (event format)
- D11, lines 374-423 (SSE parsing)
- D11, lines 425-454 (SSE serialization)

**Success Criterion:**
```bash
# Streaming request
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Authorization: Bearer pensieve-local-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"claude-3-sonnet-20240229",
    "max_tokens":100,
    "stream":true,
    "messages":[{"role":"user","content":"Tell a short story"}]
  }'

# Should output: event: message_start\ndata: {...}\n\nevent: content_block_start\n...
# With real tokens appearing in real-time
```

---

### **STEP 6: End-to-End Integration Test with Claude Code** (Days 11-12)

**Goal**: Verify `claude --print "hello"` works end-to-end

**Test First (RED):**

```bash
#!/bin/bash
# File: scripts/test-claude-integration.sh

set -e

echo "🧪 Testing Claude Code integration..."

# 1. Setup environment
export ANTHROPIC_AUTH_TOKEN="pensieve-local-token"
export ANTHROPIC_BASE_URL="http://127.0.0.1:7777"
export API_TIMEOUT_MS="3000000"

# 2. Start Pensieve if not running
if ! curl -s http://127.0.0.1:7777/health > /dev/null; then
    echo "  → Starting Pensieve server..."
    cargo run --release --manifest-path pensieve-02/Cargo.toml &
    sleep 2
fi

# 3. Test simple query
echo "  → Testing simple query..."
OUTPUT=$(claude --print "Say hello in 3 words" 2>&1)

if echo "$OUTPUT" | grep -q -i "hello\|hi"; then
    echo "✅ Simple query works: '$OUTPUT'"
else
    echo "❌ Simple query failed"
    echo "Output: $OUTPUT"
    exit 1
fi

# 4. Test streaming (interactive)
echo "  → Testing streaming mode..."
# TODO: Interactive test (requires pty)

echo ""
echo "✅ Integration test passed!"
```

**Implementation (GREEN):**
- Ensure all previous steps are complete
- Test with actual Claude Code CLI
- Verify auth, translation, and response handling

**Success Criterion:**
```bash
$ export ANTHROPIC_AUTH_TOKEN="pensieve-local-token"
$ export ANTHROPIC_BASE_URL="http://127.0.0.1:7777"
$ claude --print "Say hello"
Hello! How can I help you today?
# ^ Real response from Pensieve, not cloud API
```

---

## 5. Testing Strategy Following S01 Principles

### 5.1 Test Execution Order (STUB → RED → GREEN → REFACTOR)

**Week 1 Daily Cycle:**

```
Monday:
  1. Write tests (RED) - all fail
  2. Implement auth handler (GREEN)
  3. Refactor and verify

Tuesday-Wednesday:
  1. Write translation tests (RED)
  2. Implement Anthropic → MLX (GREEN)
  3. Implement MLX → Anthropic (GREEN)
  4. Refactor

Thursday-Friday:
  1. Write setup script tests (RED)
  2. Implement shell script (GREEN)
  3. Test on clean machine

Week 2:
  1. Write streaming tests (RED)
  2. Implement SSE events (GREEN)
  3. Integration test with Claude Code
```

### 5.2 Test Categories

**Unit Tests** (Fastest - run during development)
- Authentication validation
- Request translation accuracy
- Response structure validation
- Token counting correctness

**Integration Tests** (Medium speed - run per feature)
- Full request/response cycle
- Streaming event sequence
- Error handling
- Timeout behavior

**End-to-End Tests** (Slowest - run before release)
- Actual Claude Code CLI invocation
- Real model inference
- Performance benchmarks
- Multi-request scenarios

### 5.3 Performance Contracts

**From D11 (lines 1368-1399):**

```rust
#[tokio::test]
async fn test_proxy_overhead_within_contract() {
    // Direct MLX call latency
    let direct_start = Instant::now();
    mlx_client.generate("test").await.unwrap();
    let direct_time = direct_start.elapsed();

    // Via proxy latency
    let proxy_start = Instant::now();
    proxy_server.handle_request(...).await.unwrap();
    let proxy_time = proxy_start.elapsed();

    // PERFORMANCE CONTRACT: Overhead < 20ms
    let overhead = proxy_time.saturating_sub(direct_time);
    assert!(overhead < Duration::from_millis(20),
            "Proxy overhead {:?} exceeds contract", overhead);
}

#[tokio::test]
async fn test_throughput_meets_contract() {
    // CONTRACT: Minimum 16 tokens/second (current MLX capability)
    let start = Instant::now();
    let tokens = generate_n_tokens(160).await;
    let elapsed = start.elapsed();

    let tps = tokens as f64 / elapsed.as_secs_f64();
    assert!(tps >= 16.0, "TPS {:.2} below contract of 16", tps);
}
```

---

## 6. Critical Implementation Details

### 6.1 From z.ai Implementation (D10)

**Do's:**
- ✅ Use `ANTHROPIC_AUTH_TOKEN` (not `ANTHROPIC_API_KEY`)
- ✅ Use Node.js for JSON file manipulation
- ✅ Set `API_TIMEOUT_MS` to 3,000,000
- ✅ Create `~/.claude.json` with onboarding flag
- ✅ Implement model name mapping (claude-* → phi-3-mini-4bit)

**Don'ts:**
- ❌ Don't use bash heredocs for JSON (can corrupt)
- ❌ Don't use shell environment variables (process-local)
- ❌ Don't modify system-wide configs
- ❌ Don't require manual environment setup

### 6.2 From claude-code-router Implementation (D11)

**Adopt:**
- ✅ Environment variable override pattern (D11, lines 161-174)
- ✅ Token counting with tiktoken (D11, lines 290-327)
- ✅ Multi-factor routing logic (D11, lines 199-235)
- ✅ SSE stream processing (D11, lines 356-454)
- ✅ Configuration with env interpolation (D11, lines 690-718)

**Avoid:**
- ❌ Zero test coverage (D11, line 1182)
- ❌ Over-engineering early (stick to MVP)
- ❌ Complex agent system without routing first

---

## 7. Dependencies to Add

### Cargo.toml for pensieve-09-anthropic-proxy

```toml
[package]
name = "pensieve-09-anthropic-proxy"
version = "0.1.0"
edition = "2021"

[dependencies]
# HTTP server
warp = "0.3"
tokio = { version = "1", features = ["full"] }

# JSON/Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Token counting
tiktoken-rs = "0.5"  # For token counting

# UUID generation
uuid = { version = "1", features = ["v4", "serde"] }

# Error handling (from S01 principles)
thiserror = "1"
anyhow = "1"

# Async utilities
futures = "0.3"
bytes = "1"

[dev-dependencies]
tokio-test = "0.4"
reqwest = { version = "0.11", features = ["json"] }
```

---

## 8. Success Metrics & Milestones

### Phase 1: Basic Proxy (End of Week 1)

**Metrics:**
- ✅ `claude --print "say hello"` returns non-error response
- ✅ Authentication tests all pass
- ✅ Translation tests all pass
- ✅ 95%+ test coverage for proxy layer
- ✅ Setup script runs without errors

**Tests to pass:**
```bash
cargo test --package pensieve-09-anthropic-proxy auth:: --all
cargo test --package pensieve-09-anthropic-proxy translator:: --all
./scripts/test-setup.sh
```

### Phase 2: Intelligent Routing (Week 2)

**Metrics:**
- ✅ Token counting accurate within 5% of tiktoken reference
- ✅ Long-context detection working
- ✅ Background task detection working
- ✅ Routing logic tests pass
- ✅ Session tracking working

### Phase 3: Streaming (Week 2-3)

**Metrics:**
- ✅ SSE events match Anthropic format exactly
- ✅ Streaming integrates with Claude Code CLI
- ✅ No dropped tokens
- ✅ Streaming performance contract met (>16 TPS)

### Phase 4: Production Ready (Week 3-4)

**Metrics:**
- ✅ Full integration test with Claude Code passes
- ✅ Performance contracts met
- ✅ Error handling comprehensive
- ✅ Documentation complete
- ✅ Setup script works on fresh machine

---

## 9. Risk Mitigation

### Known Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| SSE format not exact | Medium | High | Compare byte-for-byte with curl output from Anthropic |
| Token counting inaccurate | Low | Medium | Validate against tiktoken reference implementation |
| Timeout too short | Low | High | Use 50-minute timeout per z.ai (D10, line 101) |
| JSON corruption in setup | Low | High | Use Node.js for file manipulation (D10, lines 484-490) |
| Auth header parsing | Medium | Medium | Test with actual Anthropic SDK |

### Mitigation Strategies

1. **Compare with Reference Implementations**
   - Always check z.ai setup script
   - Always check claude-code-router source
   - Use curl to validate manually

2. **Automated Testing**
   - Unit tests for every function
   - Integration tests with actual requests
   - Performance tests with benchmarks

3. **Manual Verification**
   - Test on fresh machine before release
   - Compare output with cloud Claude
   - Verify streaming format with hexdump

---

## 10. File Manifest & Creation Checklist

### New Crate: pensieve-09-anthropic-proxy

```
pensieve-09-anthropic-proxy/
├── Cargo.toml                          # [NEW] Workspace member
├── src/
│   ├── lib.rs                          # [NEW] Public API exports
│   ├── server.rs                       # [NEW] HTTP server setup
│   ├── auth.rs                         # [NEW] Authentication handler
│   ├── translator.rs                   # [NEW] Request/response translation
│   ├── streaming.rs                    # [NEW] SSE event generation
│   └── router.rs                       # [NEW] Intelligent routing (Phase 2)
└── tests/
    └── integration_tests.rs            # [NEW] Integration tests

scripts/
├── setup-claude-code.sh                # [NEW] User setup script
├── test-setup.sh                       # [NEW] Setup verification
└── test-claude-integration.sh          # [NEW] E2E integration test

.domainDocs/
└── D12-integration-next-steps.md       # [NEW] This document

Root/
└── Cargo.toml                          # [MODIFY] Add pensieve-09 member
```

---

## 11. References & Citations

### Domain Documentation
- **D09**: Claude Code integration challenges and solutions
  - Reference: Settings.json approach (line 135-156)
  - Reference: Multi-solution comparison (line 585-595)

- **D10**: z.ai research - production implementation
  - Reference: Authentication approach (line 99, 176)
  - Reference: Timeout configuration (line 101)
  - Reference: Settings.json manipulation (line 142-180)
  - Reference: Request/response format (line 207-262)

- **D11**: claude-code-router research - intelligent routing
  - Reference: Environment override pattern (line 161-174)
  - Reference: Routing logic (line 199-235)
  - Reference: Token counting (line 290-327)
  - Reference: SSE processing (line 356-454)
  - Reference: Critical gap: No tests (line 1182)

- **S01**: Steering principles - TDD architecture
  - Reference: Test-first development (line 6)
  - Reference: Executable specifications (line 14-19)
  - Reference: 8 architectural principles (line 23-49)

### External References
- Anthropic API Reference: https://docs.anthropic.com/claude/reference
- Messages API: https://docs.anthropic.com/claude/reference/messages
- claude-code-router GitHub: https://github.com/musistudio/claude-code-router
- tiktoken-rs: https://crates.io/crates/tiktoken-rs

---

## 12. Quick Start: Day 1 Checklist

```bash
# 1. Review this document and three research docs
[ ] Read D09-claude-code-integration-ultrathink.md
[ ] Read D10-claude-code-zai-integration-research.md
[ ] Read D11-claude-code-router-research.md
[ ] Review S01-README-MOSTIMP.md (TDD principles)

# 2. Create project structure
[ ] mkdir -p pensieve-09-anthropic-proxy/{src,tests}
[ ] Create Cargo.toml with dependencies from Section 7
[ ] Create pensieve-09-anthropic-proxy/src/lib.rs

# 3. Write tests (STEP 1: Authentication)
[ ] Create tests/auth_tests.rs with RED tests from STEP 1
[ ] Run: cargo test --package pensieve-09-anthropic-proxy (should fail)

# 4. Implement to pass tests
[ ] Create src/auth.rs with validate_auth() implementation
[ ] Run: cargo test --package pensieve-09-anthropic-proxy (should pass)

# 5. Setup verification
[ ] Create scripts/setup-claude-code.sh
[ ] Create scripts/test-setup.sh
[ ] Test on fresh shell: ./scripts/test-setup.sh
```

---

## 13. Next Handoff

**For Week 1 Implementation:**

1. **Start with STEP 1** (Authentication) - smallest, fastest feedback
2. **Write tests first** - follow RED → GREEN → REFACTOR
3. **One step per day** to maintain quality and debugging time
4. **Run all tests daily** to catch regressions early
5. **Verify with actual Claude Code** after each step

**Weekly Checkpoints:**
- **EOD Monday**: Auth handler complete + tests passing
- **EOD Tuesday**: Translation layer complete + tests passing
- **EOD Wednesday**: Setup script working + tested
- **EOD Thursday**: Streaming partial + being tested
- **EOD Friday**: Streaming complete + Claude Code integration test

**Definition of Done:**
```bash
$ ./scripts/test-claude-integration.sh
✅ Integration test passed!

$ claude --print "Say hello"
Hello! How can I help you today?
```

---

**Document Status**: ✅ COMPLETE - Ready for Implementation
**Confidence Level**: 95% (Validated by two production systems)
**Timeline**: 4 weeks to production-ready
**Next Action**: Begin STEP 1 (Authentication) implementation

*Synthesized from D09, D10, D11 research and S01 principles*
*Date: October 29, 2025*
