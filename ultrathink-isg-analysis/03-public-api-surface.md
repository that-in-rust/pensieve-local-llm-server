# Public API Surface Analysis

## Overview

This document catalogs the public interfaces exposed by each Pensieve crate, based on ISG entity analysis.

**Note**: The ISG analysis shows `is_public: None` for most entities, suggesting visibility metadata wasn't fully extracted. This analysis infers public APIs based on crate purpose and architectural position.

## HTTP API (Primary External Interface)

### Endpoint: POST /v1/messages

**Crate**: pensieve-02 (HTTP Server)
**Compatibility**: Anthropic Messages API v1
**Authentication**: Optional (local development)

**Request Model** (pensieve-03):
```rust
// Inferred from 6 structs in pensieve-03
pub struct MessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
}

pub struct Message {
    pub role: MessageRole,  // enum: user, assistant, system
    pub content: Vec<ContentBlock>,
}

pub enum ContentBlock {
    Text { text: String },
    // Future: Image, Document, etc.
}
```

**Response Model**:
```rust
pub struct MessageResponse {
    pub id: String,
    pub model: String,
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<StopReason>,
    pub usage: TokenUsage,
}

pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
}
```

**Streaming Response** (SSE format):
```
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

data: {"type":"content_block_stop","index":0}

data: {"type":"message_stop"}

data: [DONE]
```

### Endpoint: GET /health

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

## CLI Interface (pensieve-01)

**Binary**: `pensieve`

### Commands

**Entity Breakdown**: 54 entities total
- 4 enums (likely Commands, SubCommands, etc.)
- 7 structs (config, args)
- 15 functions (command handlers)
- 16 methods (arg parsing, validation)

### Inferred CLI Structure

```bash
# Start server
pensieve start [OPTIONS]

Options:
  --model <PATH>          Path to model.safetensors (REQUIRED)
  --host <HOST>           Bind address [default: 127.0.0.1]
  --port <PORT>           Port number [default: 7777]
  --verbose               Enable verbose logging

# Stop server
pensieve stop

# Configuration
pensieve config [ACTION]

Actions:
  get <KEY>               Get configuration value
  set <KEY> <VALUE>       Set configuration value
  list                    List all configuration

# Health check
pensieve health

# Version info
pensieve version
```

### CLI Exit Codes

Inferred from `CliError` enum (4 enum variants):
```
0  - Success
1  - Invalid arguments / configuration error
2  - Server start failed
3  - I/O error
```

## Crate-Level Public APIs

### pensieve-07 (Core Traits)

**Purpose**: Foundation layer
**Entities**: 25 total (3 traits, 6 impl blocks)

**Public Traits** (inferred):
```rust
// Core error type - used by all crates
pub enum CoreError {
    ConfigError(String),
    InferenceError(String),
    IOError(std::io::Error),
    // ... (1 enum in stats)
}

pub type CoreResult<T> = Result<T, CoreError>;

// Core trait abstractions (3 traits identified)
pub trait InferenceProvider {
    fn load_model(&mut self, path: &Path) -> CoreResult<()>;
    fn generate(&self, input: &[TokenId]) -> CoreResult<Vec<TokenId>>;
}

pub trait TokenEncoder {
    fn encode(&self, text: &str) -> CoreResult<Vec<TokenId>>;
    fn decode(&self, tokens: &[TokenId]) -> CoreResult<String>;
}

pub trait ModelMetadata {
    fn model_type(&self) -> &str;
    fn context_length(&self) -> usize;
    fn vocab_size(&self) -> usize;
}
```

**Public Types**:
```rust
pub type TokenId = u32;
pub type LogProb = f32;
```

### pensieve-04 (Engine Layer)

**Purpose**: Inference abstraction
**Entities**: 217 total (9 traits, 39 impl blocks)

**Public Traits** (9 identified):
```rust
// Primary inference interface
pub trait InferenceEngine {
    fn generate_tokens(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        config: GenerationConfig,
    ) -> CoreResult<TokenStream>;
}

// Streaming interface
pub trait TokenStream: Iterator<Item = CoreResult<Token>> {
    fn cancel(&mut self);
}

// Configuration
pub trait EngineConfig {
    fn temperature(&self) -> f32;
    fn top_p(&self) -> f32;
    fn top_k(&self) -> Option<usize>;
}

// Additional 6 traits (inferred):
// - ModelLoader
// - ContextManager
// - CacheStrategy
// - SamplingStrategy
// - ErrorRecovery
// - PerformanceMonitor
```

### pensieve-05 (Model Handling)

**Purpose**: Model loading and management
**Entities**: 339 total (1 trait, 63 impl blocks)

**Public Structs** (inferred from 33 structs):
```rust
pub struct ModelHandle {
    pub path: PathBuf,
    pub metadata: ModelMetadata,
    pub config: ModelConfig,
}

pub struct ModelConfig {
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

pub struct SafetensorsLoader;
pub struct TokenizerLoader;
```

**Public Functions** (41 functions):
```rust
// Model loading
pub fn load_model(path: &Path) -> CoreResult<ModelHandle>;
pub fn load_safetensors(path: &Path) -> CoreResult<Weights>;
pub fn load_tokenizer(path: &Path) -> CoreResult<Tokenizer>;

// Validation
pub fn validate_model(handle: &ModelHandle) -> CoreResult<()>;
pub fn check_compatibility(config: &ModelConfig) -> CoreResult<()>;
```

### pensieve-06 (Metal GPU)

**Purpose**: Apple Silicon GPU acceleration
**Entities**: 167 total (4 traits, 35 impl blocks)

**Public Traits** (4 identified):
```rust
pub trait MetalDevice {
    fn name(&self) -> &str;
    fn max_buffer_size(&self) -> usize;
    fn supports_operation(&self, op: GpuOperation) -> bool;
}

pub trait MetalBuffer {
    fn allocate(size: usize) -> CoreResult<Self>;
    fn copy_from_host(&mut self, data: &[f32]) -> CoreResult<()>;
    fn copy_to_host(&self, data: &mut [f32]) -> CoreResult<()>;
}

pub trait MetalKernel {
    fn execute(&self, inputs: &[&MetalBuffer], outputs: &mut [&mut MetalBuffer]) -> CoreResult<()>;
}

pub trait MetalCommandQueue {
    fn submit(&mut self, kernel: &dyn MetalKernel) -> CoreResult<()>;
    fn wait(&mut self) -> CoreResult<()>;
}
```

### pensieve-03 (API Models)

**Purpose**: Anthropic API compatibility
**Entities**: 41 total (5 enums, 6 structs)

**Public Enums** (5 identified):
```rust
pub enum MessageRole {
    User,
    Assistant,
    System,
}

pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    Error,
}

pub enum ApiError {
    InvalidRequest(String),
    AuthenticationFailed,
    ModelNotFound,
    RateLimitExceeded,
    InternalError(CoreError),
}

// Additional 2 enums (likely ContentBlockType, StreamEventType)
```

**Public Structs** (6 identified):
```rust
pub struct MessageRequest { /* ... */ }
pub struct MessageResponse { /* ... */ }
pub struct Message { /* ... */ }
pub struct ContentBlock { /* ... */ }
pub struct TokenUsage { /* ... */ }
pub struct StreamEvent { /* ... */ }
```

### pensieve-02 (HTTP Server)

**Purpose**: HTTP API server
**Entities**: 60 total (1 enum, 7 structs)

**Public Functions** (17 functions):
```rust
// Server lifecycle
pub async fn start_server(config: ServerConfig) -> CoreResult<()>;
pub async fn shutdown_server() -> CoreResult<()>;

// Route handlers (internal, but part of public API semantics)
async fn handle_messages(req: MessageRequest) -> Result<Response, ApiError>;
async fn handle_health() -> Result<Response, ApiError>;

// Middleware
pub fn authentication_middleware() -> Middleware;
pub fn logging_middleware() -> Middleware;
```

**Server Config**:
```rust
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_path: PathBuf,
    pub require_auth: bool,
    pub api_key: Option<String>,
}
```

## Python Bridge Public API

**Module**: `python_bridge/mlx_inference.py`
**Entities**: 46 total (35 functions, 8 classes)

### CLI Interface

```bash
python3 python_bridge/mlx_inference.py \
  --model-path <PATH> \
  --prompt <TEXT> \
  --max-tokens <N> \
  [--stream] \
  [--metrics] \
  [--temperature <FLOAT>] \
  [--top-p <FLOAT>]
```

### Python API

```python
# Main inference class (1 of 8 classes)
class MLXInference:
    def __init__(self, model_path: str):
        self.model = mlx.load_model(model_path)
        self.tokenizer = load_tokenizer(model_path)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """Generate text from prompt."""
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        return {
            "tokens_per_second": self.tps,
            "total_tokens": self.total_tokens,
            "inference_time_ms": self.inference_time,
        }
```

## API Stability Analysis

### Stable APIs (v1.0 ready)

**HTTP Endpoints**:
- ✅ `POST /v1/messages` - Anthropic-compatible
- ✅ `GET /health` - Standard health check

**CLI Commands**:
- ✅ `pensieve start` - Core functionality
- ✅ `pensieve stop` - Server management

### Unstable APIs (internal use)

**Engine Traits** (pensieve-04):
- ⚠️ 9 traits - may change during MLX migration
- ⚠️ Internal abstractions not yet stabilized

**Model Loading** (pensieve-05):
- ⚠️ 339 entities suggest complex internal API
- ⚠️ May be refactored during architecture cleanup

### Deprecated APIs

Currently none (project is pre-1.0).

## Breaking Change Risk

### High Risk Areas

1. **pensieve-05 refactoring** (339 entities)
   - If split into multiple crates, internal APIs will change
   - Risk: High for internal callers, None for external users

2. **Engine trait redesign** (9 traits)
   - MLX migration may require trait changes
   - Risk: High for internal extensions, Low for CLI/HTTP users

3. **Python bridge replacement**
   - Native MLX-Rust bindings will change internal interface
   - Risk: None for external users (HTTP API unchanged)

### Low Risk Areas

1. **HTTP API** (Anthropic-compatible)
   - Follows external standard
   - Risk: Very low

2. **CLI interface**
   - Simple, focused command set
   - Risk: Low

## API Versioning Strategy

### Current State

- No explicit versioning in API
- HTTP endpoint uses `/v1/` prefix
- CLI has no version-specific commands

### Recommendations

1. **HTTP API**: Already versioned (`/v1/`), maintain compatibility
2. **CLI**: Add `--version` flag, semantic versioning
3. **Rust crates**: Use Cargo semantic versioning (0.x.y → 1.0.0)
4. **Internal APIs**: Allow breaking changes until 1.0.0

## Public API Surface Summary

| Layer | Crate | Public Entities | Stability | External Use |
|-------|-------|----------------|-----------|--------------|
| L3 | pensieve-01 (CLI) | ~20 commands/args | Medium | Direct (users) |
| L3 | pensieve-02 (HTTP) | 2 endpoints | High | Direct (clients) |
| L3 | pensieve-03 (API) | 15 types | High | Indirect (HTTP) |
| L2 | pensieve-04 (Engine) | 9 traits | Low | Internal only |
| L2 | pensieve-05 (Models) | ~10 types | Low | Internal only |
| L2 | pensieve-06 (Metal) | 4 traits | Low | Internal only |
| L1 | pensieve-07 (Core) | 3 traits + error | Medium | All crates |
| External | python_bridge | 1 class + CLI | Medium | Indirect (HTTP) |

**Total Public Surface**:
- **2 HTTP endpoints** (external)
- **~5 CLI commands** (external)
- **~15 API types** (indirectly exposed via HTTP)
- **~16 traits** (internal abstractions)

## Usage Examples

### HTTP API

```bash
# Non-streaming request
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
    ]
  }'

# Streaming request
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "stream": true,
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
    ]
  }'
```

### CLI

```bash
# Start server
./target/debug/pensieve start \
  --model ./models/Phi-3-mini-128k-instruct-4bit/model.safetensors \
  --host 127.0.0.1 \
  --port 7777

# Check health
./target/debug/pensieve health

# Stop server
./target/debug/pensieve stop
```

### Python Bridge (Direct)

```bash
# Direct inference
python3 python_bridge/mlx_inference.py \
  --model-path ./models/Phi-3-mini-128k-instruct-4bit \
  --prompt "Hello!" \
  --max-tokens 50 \
  --stream \
  --metrics
```

---

*API surface analysis based on 1,137 entities across 10 crates*
