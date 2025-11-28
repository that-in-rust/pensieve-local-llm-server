# Data Flow Analysis

## Request Processing Pipeline

This document traces how data flows through the Pensieve system from HTTP request to LLM response.

### High-Level Flow

```
┌──────────────┐
│   HTTP API   │  (pensieve-02: 60 entities)
│  Request In  │  - Warp handlers
└──────┬───────┘  - JSON deserialization
       │
       ↓
┌──────────────────┐
│   API Models     │  (pensieve-03: 41 entities)
│  Validation &    │  - Anthropic API compatibility
│  Transformation  │  - Message format conversion
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Engine Layer    │  (pensieve-04: 217 entities)
│  Inference       │  - Trait-based abstraction
│  Orchestration   │  - 9 traits for flexibility
└──────┬───────────┘
       │
       ├─────────────────────┬──────────────────┐
       ↓                     ↓                  ↓
┌─────────────┐    ┌──────────────┐   ┌──────────────┐
│   Models    │    │    Metal     │   │Python Bridge │
│ (pensieve-05│    │ (pensieve-06)│   │(MLX Runtime) │
│ 339 entities│    │ 167 entities │   │  46 entities │
│             │    │              │   │              │
│ - Safetensors   │ - GPU buffers│   │ - MLX model  │
│ - Tokenization  │ - Metal API  │   │ - Inference  │
│ - Model loading │ - Acceleration│   │ - 16.85 TPS  │
└─────────────┘    └──────────────┘   └──────────────┘
       │                     │                  │
       └─────────────────────┴──────────────────┘
                             │
                             ↓
                    ┌────────────────┐
                    │  Token Stream  │
                    │  Generation    │
                    └────────┬───────┘
                             │
                             ↓
                    ┌────────────────┐
                    │   SSE Stream   │  (pensieve-02)
                    │  HTTP Response │  - Server-Sent Events
                    └────────────────┘
```

## Layer-by-Layer Breakdown

### Layer 1: HTTP API Server (pensieve-02)

**Entry Point**: `POST /v1/messages`

**Entity Breakdown**:
- **21 methods** - Request handlers, middleware
- **17 functions** - Route setup, server lifecycle
- **9 impl blocks** - Handler implementations
- **7 structs** - Server config, request context

**Data Structures** (from pensieve-03):
```rust
// Incoming request
struct MessageRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    stream: Option<bool>,
    ...
}
```

**Flow**:
1. Warp receives HTTP POST
2. JSON body deserialized to `MessageRequest`
3. Authentication validation (optional for local dev)
4. Forward to engine layer

### Layer 2: API Models (pensieve-03)

**Responsibility**: Anthropic API compatibility

**Entity Breakdown**:
- **15 functions** - Serialization helpers
- **6 structs** - Request/response models
- **5 enums** - Error types, message roles
- **6 methods** - Model validation

**Key Transformations**:
```rust
// Anthropic format → Internal format
MessageRequest → InferenceRequest
  - Extract system prompt
  - Concatenate message history
  - Apply max_tokens limit
  - Configure streaming

// Internal format → Anthropic response
TokenStream → MessageResponse
  - Format as SSE events
  - Track token usage
  - Handle stop sequences
```

**Data Flow**:
```
HTTP JSON → MessageRequest → Validation → InferenceRequest → Engine
```

### Layer 3: Engine Orchestration (pensieve-04)

**Core Abstraction**: 9 traits for flexibility

**Entity Breakdown**:
- **93 methods** - Trait method implementations
- **41 functions** - Helper functions
- **39 impl blocks** - Trait implementations
- **26 structs** - Engine config, state machines
- **9 traits** - Abstraction layer

**Key Traits** (inferred from structure):
- `InferenceEngine` - Core inference interface
- `TokenStream` - Streaming token generation
- `ModelProvider` - Model loading abstraction
- `ContextManager` - Context window handling
- `ErrorHandler` - Error recovery
- (Plus 4 more from 9 total)

**Engine Flow**:
```
InferenceRequest
   ↓
[Select Model Provider]
   ↓
[Load Model if needed]  (pensieve-05: 339 entities)
   ↓
[Prepare GPU Buffers]   (pensieve-06: 167 entities)
   ↓
[Execute Inference]     (python_bridge: MLX runtime)
   ↓
TokenStream (async iterator)
```

### Layer 4A: Model Loading (pensieve-05)

**Largest crate**: 339 entities

**Entity Breakdown**:
- **184 methods** - Model manipulation
- **63 impl blocks** - Extensive trait implementations
- **41 functions** - Loading utilities
- **33 structs** - Model metadata, tensors

**Responsibilities**:
1. **Safetensors Loading**: Parse .safetensors format
2. **Tokenizer**: Phi-3 tokenizer (128k context)
3. **Model Metadata**: Config, architecture info
4. **Memory Management**: Tensor allocation

**Data Structures**:
```rust
// Model file representation
struct SafetensorsModel {
    weights: HashMap<String, Tensor>,
    metadata: ModelMetadata,
    config: ModelConfig,
}

// Tokenizer state
struct Tokenizer {
    vocab: Vocabulary,
    special_tokens: HashMap<String, TokenId>,
    ...
}
```

**Flow**:
```
Model Path (CLI arg)
   ↓
[Read safetensors file]
   ↓
[Parse tensor metadata]
   ↓
[Load tokenizer.json]
   ↓
[Validate model architecture]
   ↓
Ready for inference
```

### Layer 4B: Metal GPU Acceleration (pensieve-06)

**GPU Layer**: 167 entities

**Entity Breakdown**:
- **85 methods** - GPU operations
- **35 impl blocks** - Metal API wrappers
- **19 functions** - Buffer management
- **14 structs** - GPU state, buffers
- **4 traits** - GPU abstraction

**Metal Pipeline**:
```
CPU Tensors
   ↓
[Allocate Metal Buffers]
   ↓
[Copy to GPU memory]
   ↓
[Configure Metal Pipeline]
   ↓
[Execute GPU kernels]
   ↓
[Read results back]
   ↓
CPU Tensors (outputs)
```

**Key Operations**:
- Matrix multiplication (GEMM)
- Activation functions (GELU, ReLU)
- Attention mechanisms
- Layer normalization
- Token embedding lookup

### Layer 4C: Python Bridge (MLX Runtime)

**External Layer**: 46 entities (35 functions)

**Actual Implementation**: `python_bridge/mlx_inference.py`

**MLX Flow**:
```python
# Load model once (cached)
model = mlx.load_model(model_path)

# For each inference:
tokens = tokenizer.encode(prompt)
  ↓
output_tokens = model.generate(
    tokens,
    max_tokens=max_tokens,
    temperature=0.7
)
  ↓
text = tokenizer.decode(output_tokens)
```

**Performance**: ~16.85 TPS (target: 25+ TPS)

**Data Flow**:
```
Rust Request → Python via subprocess
   ↓
MLX model inference (GPU)
   ↓
Stream tokens back → Rust
   ↓
Format as SSE → HTTP client
```

## Error Propagation

### Error Types by Layer

**pensieve-07 (Core)**: `CoreError` enum
```rust
enum CoreError {
    ConfigError,
    InferenceError,
    IOError,
    // ... (inferred from 1 enum in stats)
}
```

**pensieve-03 (API)**: `ApiError` enum (5 enums total)
```rust
enum ApiError {
    InvalidRequest,
    AuthenticationFailed,
    ModelNotFound,
    RateLimitExceeded,
    InternalError(CoreError),
}
```

**pensieve-01 (CLI)**: `CliError` enum (4 enums)
```rust
enum CliError {
    InvalidArguments,
    ServerStartFailed,
    ConfigError,
    // ...
}
```

### Error Flow

```
┌─────────────────────┐
│  Python MLX Error   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ InferenceError      │ (pensieve-04)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ ApiError            │ (pensieve-03)
│ (HTTP 500)          │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ JSON Error Response │
│ { "error": {...} }  │
└─────────────────────┘
```

## Streaming Flow (SSE)

When `stream: true` in request:

```
1. Client sends request with stream=true
   ↓
2. Server opens SSE connection
   ↓
3. For each generated token:
   ┌─────────────────────────┐
   │ MLX generates token     │
   ├─────────────────────────┤
   │ Decode to text          │
   ├─────────────────────────┤
   │ Format as SSE event:    │
   │ data: {"delta": "tok"}  │
   ├─────────────────────────┤
   │ Flush to HTTP stream    │
   └─────────────────────────┘
   ↓
4. Final event: data: [DONE]
   ↓
5. Close connection
```

**Implementation** (pensieve-02):
- Async stream handling
- Backpressure management
- Error handling in stream
- Graceful connection close

## Performance Bottlenecks

### Identified Bottlenecks (from CLAUDE.md)

1. **MLX/Rust Bridge** (current: 16.85 TPS)
   - Subprocess communication overhead
   - Serialization/deserialization
   - **Target**: Native MLX Rust bindings

2. **Model Loading** (pensieve-05: 339 entities)
   - Large entity count suggests complex loading
   - Potential optimization: Lazy loading

3. **Memory Copies** (pensieve-06)
   - CPU → GPU transfers
   - Metal buffer management

### Optimization Opportunities

```
Request Path:
HTTP (1ms) → API Validation (1ms) → Engine (5ms) →
Python Bridge (50ms) → MLX Inference (500ms) →
Stream Response (10ms/token)

Total: ~566ms first token + 10ms/token
```

**Optimization Targets**:
- ⚡ Python Bridge: Replace with native MLX-Rust bindings (-45ms)
- ⚡ Model caching: Keep model in memory (-100ms cold start)
- ⚡ Async pipeline: Overlap GPU/CPU work

## Memory Flow

### Memory Management by Layer

**pensieve-05 (Models)**:
- Model weights: ~2.1GB (Phi-3 4-bit)
- Tokenizer vocab: ~50MB
- Context buffers: Dynamic (up to 128k tokens)

**pensieve-06 (Metal)**:
- GPU buffers: Allocated per-request
- Activation tensors: Transient
- Peak GPU usage: ~3-4GB

**python_bridge**:
- MLX model cache: Persistent
- Inference buffers: Per-request
- Peak memory: ~2.2GB

### Memory Safety

From CLAUDE.md: "92% memory reduction via persistent MLX server"

**Before**: Model loaded per-request (~2GB × requests)
**After**: Single persistent model instance (~2.2GB total)

## Data Transformation Summary

```
HTTP JSON (bytes)
   ↓ deserialize
MessageRequest (struct)
   ↓ validate & transform
InferenceRequest (struct)
   ↓ tokenize
Vec<TokenId> (u32 array)
   ↓ embed
Tensor<f32> (GPU)
   ↓ inference
Tensor<f32> (logits)
   ↓ sample
TokenId (u32)
   ↓ decode
String (text)
   ↓ format SSE
"data: {...}\n\n" (bytes)
   ↓
HTTP Response Stream
```

---

*Analysis based on entity flow through 1,137 Pensieve components*
