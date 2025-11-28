# D10: Claude Code Integration - z.ai Research & Implementation Strategy

**Date**: October 29, 2025
**Status**: Research Complete - Implementation Pending
**Priority**: HIGH - Claude Code integration currently non-functional

---

## Executive Summary

Research into z.ai's successful Claude Code integration reveals they use an **Anthropic SDK-compatible proxy endpoint** that translates requests between the Anthropic SDK format and their GLM model backend. This is a proven, production-grade approach that requires no Claude Code modifications.

**Key Insight**: We need to build a similar proxy/wrapper layer for Pensieve, not just expose an API endpoint.

---

## 1. The Problem: Why Our Current Approach Doesn't Work

### Current Pensieve Implementation
```rust
// pensieve-02/src/lib.rs
let messages = warp::path("v1")
    .and(warp::path("messages"))
    .and(warp::post())
    .and(warp::header::optional::<String>("authorization"))
    .and(warp::body::json())
```

**What's Wrong:**
- ✅ Correct API endpoint structure
- ✅ Correct request/response models
- ❌ Claude Code's Anthropic SDK doesn't send requests correctly
- ❌ No timeout configuration support
- ❌ Authentication header handling issues
- ❌ SDK expects specific base URL format

### Test Evidence
```bash
# Environment variables set correctly
export ANTHROPIC_API_KEY="test-api-key-12345"
export ANTHROPIC_BASE_URL="http://127.0.0.1:7777"

# Claude Code command
claude --print "Say hello"

# Result: API Error: 400 Missing request header "authorization"
# The SDK isn't passing through the auth correctly
```

---

## 2. How z.ai Solved It: The Proxy Pattern

### Architecture Overview
```
Claude Code (Anthropic SDK v3.x)
        ↓
ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
        ↓
[z.ai Translation Proxy Layer]
├── Request parsing (Anthropic format)
├── Authentication validation
├── Model name mapping (claude-3-* → GLM-4.6)
├── Request translation (Anthropic → GLM format)
├── GLM inference call
├── Response translation (GLM → Anthropic format)
└── Stream handling (SSE format)
        ↓
Claude Code receives Anthropic-formatted response
```

### z.ai's Installation Script Analysis

**File**: `1753683755292-30b3431f487b4cc1863e57a81d78e289.sh`

#### 1. Authentication Bypass (Lines 122-136)
```bash
# Create ~/.claude.json with onboarding flag
cat > "$CLAUDE_CONFIG_FILE" << 'EOF'
{
  "hasCompletedOnboarding": true
}
EOF
```

**Key Insight**: Bypasses Claude Code's login flow entirely.

#### 2. Settings Injection (Lines 142-183)
```bash
# Update ~/.claude/settings.json with env vars
node --eval '
    const content = fs.existsSync(filePath)
        ? JSON.parse(fs.readFileSync(filePath, "utf-8"))
        : {};

    fs.writeFileSync(filePath, JSON.stringify({
        ...content,
        env: {
            ANTHROPIC_AUTH_TOKEN: apiKey,
            ANTHROPIC_BASE_URL: "https://api.z.ai/api/anthropic",
            API_TIMEOUT_MS: "3000000",  // 50 minutes!
        }
    }, null, 2), "utf-8");
'
```

**Key Insights:**
- Uses `ANTHROPIC_AUTH_TOKEN` instead of `ANTHROPIC_API_KEY`
- Base URL points to proxy endpoint
- Extended timeout: 3,000,000ms (50 minutes) vs default 60 seconds
- Uses Node.js for safe JSON manipulation

#### 3. Model Name Mapping
```javascript
// z.ai proxy translates model names
"claude-3-opus-*"   → GLM-4.6
"claude-3-sonnet-*" → GLM-4.6
"claude-3-haiku-*"  → GLM-4.5-Air
```

**Key Insight**: Accepts any Claude model name, routes to appropriate backend.

---

## 3. Technical Requirements for Pensieve Proxy

### 3.1 Anthropic SDK Compatibility Layer

**What We Need to Build:**

```rust
// New crate: pensieve-09-anthropic-proxy
//
// Purpose: Translate between Anthropic SDK and Pensieve MLX backend

pub struct AnthropicProxyServer {
    mlx_handler: Arc<MlxInferenceHandler>,
    timeout_ms: u64,  // Configurable, default 3000000 (50 min)
}

impl AnthropicProxyServer {
    // Handle requests from Anthropic SDK
    async fn handle_messages(&self, req: AnthropicRequest) -> Result<AnthropicResponse> {
        // 1. Validate request format (must match Anthropic spec exactly)
        // 2. Extract authentication (ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY)
        // 3. Map model name (claude-* → phi-3-mini-4bit)
        // 4. Translate to MLX format
        // 5. Call MLX inference
        // 6. Translate response back to Anthropic format
        // 7. Return with correct headers and status codes
    }

    // Handle streaming requests
    async fn handle_messages_stream(&self, req: AnthropicRequest) -> Result<SseStream> {
        // Same translation but with SSE streaming
        // Must match Anthropic's exact SSE event format
    }
}
```

### 3.2 Configuration Requirements

**Settings File Format** (`~/.claude/settings.json`):
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

**Critical Details:**
- Must use `ANTHROPIC_AUTH_TOKEN` (not `ANTHROPIC_API_KEY`)
- Base URL should NOT include `/v1` suffix
- Timeout must be extended for local inference (50 minutes)
- JSON must be manipulated safely (use Node.js like z.ai)

### 3.3 Authentication Handling

**Current z.ai Approach:**
```typescript
// z.ai validates API key format
function validateApiKey(token: string): boolean {
  // Format: {id}.{secret}
  // Example: "1234567890abcdef.fedcba0987654321"
  return /^[a-f0-9]{16}\.[a-f0-9]{16}$/.test(token);
}
```

**Pensieve Approach (Simpler):**
```rust
fn validate_local_auth(token: Option<&str>) -> Result<(), ApiError> {
    match token {
        None => Ok(()), // Allow no auth for local dev
        Some(t) if t == "pensieve-local-token" => Ok(()),
        Some(t) if t.starts_with("sk-ant-") => Ok(()), // Anthropic format
        Some(_) => Err(ApiError::Unauthorized),
    }
}
```

### 3.4 Request/Response Translation

**Anthropic SDK Request Format:**
```json
POST /v1/messages
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

**Translation to MLX Python Bridge:**
```python
# python_bridge/mlx_inference.py call
{
  "model_path": "./models/Phi-3-mini-128k-instruct-4bit",
  "prompt": "System: You are a helpful assistant.\n\nUser: Hello, Claude!",
  "max_tokens": 1024,
  "temperature": 0.7,
  "stream": false
}
```

**Response Translation Back:**
```json
{
  "id": "msg_{uuid}",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "{mlx_generated_text}"
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

### 3.5 Streaming SSE Format

**Anthropic SDK Expects:**
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

**Current Pensieve Implementation:**
```rust
// pensieve-02/src/lib.rs - Needs update to match exact format
// Currently sends simplified events, need full Anthropic spec
```

---

## 4. Implementation Roadmap

### Phase 1: Basic Proxy Layer (Week 1)
**Goal**: Get Claude Code connecting without errors

#### Tasks:
1. **Create `pensieve-09-anthropic-proxy` crate**
   - Depends on: pensieve-02, pensieve-04, pensieve-07
   - Purpose: Anthropic SDK compatibility layer

2. **Implement Authentication Handler**
   ```rust
   // Allow local development tokens
   // Support ANTHROPIC_AUTH_TOKEN and ANTHROPIC_API_KEY
   // Return proper 401 responses for invalid auth
   ```

3. **Model Name Mapping**
   ```rust
   fn map_model_name(anthropic_model: &str) -> &str {
       match anthropic_model {
           s if s.starts_with("claude-3-opus") => "phi-3-mini-4bit",
           s if s.starts_with("claude-3-sonnet") => "phi-3-mini-4bit",
           s if s.starts_with("claude-3-haiku") => "phi-3-mini-4bit",
           _ => "phi-3-mini-4bit",
       }
   }
   ```

4. **Request Translation**
   - Parse Anthropic format
   - Convert to MLX Python bridge format
   - Handle system prompts, multi-turn conversations
   - Preserve temperature, max_tokens settings

5. **Response Translation**
   - Convert MLX output to Anthropic format
   - Generate proper message IDs
   - Calculate token usage correctly
   - Handle stop reasons

**Validation:**
```bash
# Should work after Phase 1
export ANTHROPIC_AUTH_TOKEN="pensieve-local-token"
export ANTHROPIC_BASE_URL="http://127.0.0.1:7777"
claude --print "Say hello in 5 words"
# Expected: "Hello! How can I help?"
```

---

### Phase 2: Streaming Support (Week 2)
**Goal**: Full SSE streaming with exact Anthropic format

#### Tasks:
1. **Implement SSE Event Generator**
   ```rust
   async fn stream_anthropic_events(
       mlx_stream: impl Stream<Item = String>,
       message_id: String,
   ) -> impl Stream<Item = ServerSentEvent> {
       // Convert MLX tokens to Anthropic SSE events
       // Follow exact event sequence: start → deltas → stop
   }
   ```

2. **Test Streaming with Claude Code**
   ```bash
   # Interactive mode (uses streaming)
   claude
   > Tell me a story
   # Should see tokens appearing in real-time
   ```

**Validation:**
- Streaming appears in real-time
- No dropped tokens
- Proper completion signals

---

### Phase 3: Timeout & Optimization (Week 3)
**Goal**: Production-grade reliability

#### Tasks:
1. **Extended Timeout Configuration**
   ```json
   {
     "env": {
       "API_TIMEOUT_MS": "3000000"
     }
   }
   ```

2. **Error Handling**
   - Proper Anthropic error format
   - Graceful timeout handling
   - Model loading error responses

3. **Performance Optimization**
   - Response time < 1s for first token
   - Sustained streaming without delays
   - Memory usage monitoring

**Validation:**
- Long conversations don't timeout
- Error messages are clear and actionable
- Performance meets user expectations

---

### Phase 4: Setup Script (Week 4)
**Goal**: One-command installation like z.ai

#### Tasks:
1. **Create `scripts/setup-claude-code.sh`**
   ```bash
   #!/bin/bash
   # Similar to z.ai's approach

   # 1. Create ~/.claude.json with onboarding flag
   # 2. Update ~/.claude/settings.json with env vars (using Node.js)
   # 3. Test connection to local server
   # 4. Display success message
   ```

2. **Test Installation Flow**
   ```bash
   cd pensieve-local-llm-server
   ./scripts/setup-claude-code.sh
   # Should complete in < 30 seconds
   # Should work immediately after
   ```

**Validation:**
- Fresh machine setup works
- No manual configuration needed
- Clear error messages if issues

---

## 5. Key Differences: z.ai vs Pensieve

| Feature | z.ai | Pensieve (Target) | Advantage |
|---------|------|-------------------|-----------|
| **Hosting** | Cloud (https://api.z.ai) | Local (127.0.0.1:7777) | **Pensieve** - Privacy |
| **Cost** | $3-15/month | Free | **Pensieve** - Zero cost |
| **Authentication** | Requires signup | Local tokens | **Pensieve** - No signup |
| **Model** | GLM-4.6 (cloud) | Phi-3 (local MLX) | Context-dependent |
| **Performance** | Cloud latency | Local inference | Context-dependent |
| **Privacy** | Data to z.ai | 100% local | **Pensieve** - Privacy |
| **Internet** | Required | Optional | **Pensieve** - Offline |
| **Setup** | 1 script | 1 script (planned) | Tie |
| **Timeout** | 50 minutes | Will add | Tie |
| **Image Support** | No (API) | Planned | **Pensieve** - Potential |

---

## 6. Technical Challenges & Solutions

### Challenge 1: Anthropic SDK Compatibility
**Problem**: SDK has hardcoded assumptions about API behavior

**Solution**:
- Implement exact Anthropic API spec
- Test with actual Anthropic SDK (not just curl)
- Use z.ai as reference for edge cases

### Challenge 2: Timeout Configuration
**Problem**: Default 60s timeout insufficient for local inference

**Solution**:
```json
{
  "env": {
    "API_TIMEOUT_MS": "3000000"  // 50 minutes like z.ai
  }
}
```

### Challenge 3: Settings File Management
**Problem**: Bash heredocs can corrupt JSON

**Solution**: Use Node.js like z.ai
```bash
node --eval '
    const fs = require("fs");
    const settings = JSON.parse(fs.readFileSync(path, "utf-8"));
    settings.env = { ...newEnv };
    fs.writeFileSync(path, JSON.stringify(settings, null, 2));
'
```

### Challenge 4: Authentication Flow
**Problem**: Claude Code expects specific auth patterns

**Solution**: Support both formats
```rust
// Accept both ANTHROPIC_AUTH_TOKEN and ANTHROPIC_API_KEY
// Allow local dev tokens like "pensieve-local-token"
// Accept Anthropic format "sk-ant-*" for compatibility
```

---

## 7. Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_anthropic_request_translation() {
    let anthropic_req = AnthropicRequest {
        model: "claude-3-sonnet-20240229".to_string(),
        max_tokens: 100,
        messages: vec![/* ... */],
    };

    let mlx_req = translate_to_mlx(&anthropic_req).unwrap();
    assert_eq!(mlx_req.model_path, "./models/Phi-3-mini-128k-instruct-4bit");
}

#[tokio::test]
async fn test_response_translation() {
    let mlx_output = "Hello! How can I help you today?";
    let anthropic_resp = translate_to_anthropic(mlx_output, "msg_123");

    assert_eq!(anthropic_resp.role, "assistant");
    assert_eq!(anthropic_resp.content[0].text, mlx_output);
}
```

### Integration Tests
```bash
# Test with actual Claude Code
export ANTHROPIC_AUTH_TOKEN="pensieve-local-token"
export ANTHROPIC_BASE_URL="http://127.0.0.1:7777"

# Test non-streaming
claude --print "Say hello"

# Test streaming
claude
> Tell me a short story
```

### Performance Tests
```bash
# Test timeout handling
claude --print "Generate a very long response with lots of details..."
# Should not timeout before 50 minutes

# Test concurrent requests
for i in {1..10}; do
  claude --print "Test $i" &
done
wait
# All should complete successfully
```

---

## 8. Success Metrics

### Phase 1 Complete When:
- ✅ Claude Code connects without errors
- ✅ Simple queries return responses
- ✅ Authentication works
- ✅ No 400/404 errors

### Phase 2 Complete When:
- ✅ Streaming works in real-time
- ✅ Interactive mode functional
- ✅ No dropped tokens
- ✅ Proper event formatting

### Phase 3 Complete When:
- ✅ Long conversations don't timeout
- ✅ Performance acceptable (< 1s first token)
- ✅ Error handling production-ready
- ✅ Memory usage stable

### Phase 4 Complete When:
- ✅ One-command setup works
- ✅ Fresh machine installation successful
- ✅ Documentation complete
- ✅ Ready for public release

---

## 9. Reference Materials

### z.ai Resources
- **Installation Script**: `1753683755292-30b3431f487b4cc1863e57a81d78e289.sh`
- **API Endpoint**: `https://api.z.ai/api/anthropic`
- **Documentation**: https://docs.z.ai/api-reference/introduction

### Anthropic SDK
- **TypeScript SDK**: https://github.com/anthropics/anthropic-sdk-typescript
- **API Reference**: https://docs.anthropic.com/claude/reference
- **Messages API**: https://docs.anthropic.com/claude/reference/messages

### Existing Pensieve Implementation
- **HTTP Server**: `pensieve-02/src/lib.rs`
- **API Models**: `pensieve-03/src/lib.rs`
- **MLX Bridge**: `python_bridge/mlx_inference.py`
- **CLI**: `pensieve-01/src/main.rs`

---

## 10. Next Actions

### Immediate (Today)
1. ✅ Document z.ai research (this file)
2. ⏳ Update `next-steps.md` with new roadmap
3. ⏳ Create stub for `pensieve-09-anthropic-proxy` crate

### This Week
1. Implement basic proxy layer
2. Get Claude Code connecting
3. Test simple queries

### Next 2 Weeks
1. Add streaming support
2. Optimize performance
3. Create setup script

### Within Month
1. Public release
2. Documentation
3. Community testing

---

## 11. Conclusion

### What We Learned from z.ai
1. **Proxy pattern works** in production
2. **Extended timeouts critical** for local inference
3. **Settings file injection** is reliable approach
4. **Node.js for JSON** safer than bash
5. **No Claude Code modifications** needed

### What Makes Pensieve Better
1. **100% local** - True privacy
2. **Zero cost** - No subscription
3. **Offline capable** - No internet required
4. **Open source** - Full transparency
5. **Apple Silicon optimized** - Metal GPU acceleration

### Reality Check
- Current API is close but not SDK-compatible
- Need translation/proxy layer, not just endpoint
- z.ai validates our architecture approach
- Implementation is straightforward, just needs execution

---

**Status**: Research complete, implementation roadmap defined
**Next**: Update next-steps.md and begin Phase 1 implementation
**Timeline**: 4 weeks to production-ready Claude Code integration
**Confidence**: HIGH (z.ai proves it works)

---

*Document created: October 29, 2025*
*Research source: z.ai installation script analysis*
*Author: Pensieve Development Team*
