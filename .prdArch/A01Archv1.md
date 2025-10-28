# Pensieve Local LLM Server Architecture v1.0

## Executive Summary

### Architecture Vision
The Pensieve Local LLM Server is a high-performance, Apple M1/M2/M3-optimized local server that provides Anthropic-compatible API endpoints for seamless Claude Code integration. Built exclusively with the MLX framework and Apple Silicon acceleration, it delivers cloud-like performance with local privacy and reliability.

### Design Principles
1. **MLX-First Architecture**: Exclusive focus on MLX framework with Metal backend for optimal M1/M2/M3 performance
2. **Anthropic Compatibility**: Native API compatibility for drop-in Claude Code integration
3. **Memory Efficiency**: Quantized models and intelligent caching for 16GB+ Apple systems
4. **Production Ready**: Single binary deployment with comprehensive error handling
5. **Extensible Foundation**: Modular crate architecture for future enhancements

### Key Technical Decisions & Trade-offs

#### **MLX + Apple Silicon Stack** 
- **Pros**: Native Apple Silicon performance, single binary, optimized M-series GPU utilization
- **Cons**: Limited to Apple Silicon ecosystem, requires MLX framework
- **Rationale**: Optimal performance for target hardware with Apple-optimized deployment

#### **MLX Framework**
- **Pros**: Apple-optimized, excellent M1/M2/M3 performance, native Metal backend, official Apple support
- **Cons**: Apple ecosystem dependency only
- **Rationale**: Superior Apple Silicon performance with framework-level Metal optimization and future-proofing

#### **MLX Quantization**
- **Pros**: 4-bit models reduce memory usage by 75%, fast loading, Metal-optimized, MLX-native format
- **Cons**: Slight quality reduction vs full precision
- **Rationale**: Essential for fitting Phi-3 models in 16GB RAM while maintaining quality with MLX optimization

#### **Modular Crate Architecture**
- **Pros**: Separation of concerns, independent development, testability
- **Cons**: Complex coordination, potential duplication
- **Rationale**: Long-term maintainability and feature isolation

#### **Blocking Model Loading**
- **Pros**: Simplified architecture, deterministic resource allocation
- **Cons**: Cannot serve requests during model load
- **Rationale**: Trade-off complexity for reliability in single-user scenario

### Success Metrics & Performance Targets

#### **Performance Benchmarks**
- **First Token Time**: <300ms
- **Token Throughput**: 25-40 tokens/second
- **Memory Usage**: <12GB peak (16GB constraint)
- **Model Load Time**: <8 seconds (MLX-optimized Phi-3 4-bit)
- **Concurrent Requests**: 3-5 simultaneous users

#### **Quality Metrics**
- **Anthropic API Compatibility**: 100% endpoint compatibility
- **Response Accuracy**: >95% match with cloud Claude quality
- **Uptime**: >99.9% for continuous operation
- **Error Rate**: <0.1% inference failures

## System Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pensieve Local LLM Server                     │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface (pensieve-01)                                  │
│  - Argument parsing & validation                              │
│  - Configuration management                                   │
│  - Process lifecycle (background execution)                    │
│  - Logging & monitoring                                        │
├─────────────────────────────────────────────────────────────────┤
│  HTTP API Server (pensieve-02)                                │
│  - Warp web framework                                          │
│  - Anthropic-compatible endpoints (/v1/messages)              │
│  - OpenAI fallback (/v1/chat/completions)                     │
│  - Streaming (SSE) support                                    │
│  - Request/response validation                                │
├─────────────────────────────────────────────────────────────────┤
│  Request/Response Models (pensieve-03)                       │
│  - Anthropic message format structures                         │
│  - OpenAI compatibility layer                                 │
│  - Token streaming types                                       │
│  - Error response types                                        │
├─────────────────────────────────────────────────────────────────┤
│  MLX Inference Engine (pensieve-04)                          │
│  - MLX framework with Metal backend                           │
│  - Hugging Face model loading                                 │
│  - Token generation with streaming                           │
│  - Memory management                                           │
├─────────────────────────────────────────────────────────────────┤
│  Model Management (pensieve-05)                               │
│  - Quantized model loading (4-bit MLX)                       │
│  - KV-Cache optimization                                       │
│  - Context window management                                   │
│  - Model switching (future enhancement)                        │
├─────────────────────────────────────────────────────────────────┤
│  Metal GPU Optimization (pensieve-06)                         │
│  - MLX Metal backend optimization                             │
│  - Memory pooling                                              │
│  - Async computation streams                                   │
│  - Performance profiling                                       │
├─────────────────────────────────────────────────────────────────┤
│  Configuration & Utils (pensieve-07)                          │
│  - Environment variable management                             │
│  - File I/O operations                                         │
│  - Error handling utilities                                    │
│  - Common type definitions                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow and Interaction Patterns

#### **Request Flow**
1. **CLI Interface**: Parses command-line arguments and validates configuration
2. **HTTP Server**: Accepts requests on configurable port (default: 8000)
3. **Model Validation**: Checks model file existence and compatibility
4. **Inference Engine**: Loads model into Metal GPU memory if not cached
5. **Token Generation**: Processes input through MLX + Metal
6. **Streaming Response**: Returns tokens via Server-Sent Events
7. **Logging**: Records request metrics and performance data

#### **Memory Flow**
1. **Model Loading**: HuggingFace → MLX tensors → Metal GPU memory
2. **KV-Cache**: Allocation during inference, intelligent eviction
3. **Token Buffer**: Circular buffer for streaming responses
4. **Memory Pool**: Pre-allocated buffers for consistent performance
5. **Garbage Collection**: Automatic cleanup between requests

#### **Error Handling Flow**
1. **Request Validation**: Early rejection of malformed requests
2. **Model Loading Errors**: Fallback to CPU or error response
3. **Inference Errors**: Graceful degradation with error details
4. **Network Errors**: Automatic retry with exponential backoff
5. **System Errors**: Process restart with state preservation

### Core Technology Stack

#### **Rust Ecosystem**
- **Language**: Rust 1.75+ for memory safety and performance
- **Async Runtime**: Tokio for concurrent request handling
- **Web Framework**: Warp for high-performance HTTP API
- **Serialization**: Serde for JSON request/response handling
- **CLI Parsing**: Clap for command-line interface

#### **MLX Framework**
- **Core**: `mlx` for tensor operations and Metal backend
- **Neural Networks**: `mlx.nn` for model architecture
- **Transformers**: `mlx-examples` for pre-trained models
- **Quantization**: Native support with 4-bit precision
- **Model Loading**: Hugging Face Hub integration

#### **Apple Silicon Optimization**
- **MLX Metal**: Native Apple Silicon acceleration
- **Unified Memory**: Efficient RAM-GPU memory sharing
- **Async Compute**: Parallel processing streams
- **MLX Optimizations**: Framework-level optimization for M-series chips

## Detailed Component Design

### 1. CLI Interface (pensieve-01)

#### **Responsibilities**
- Parse command-line arguments with validation
- Manage server configuration and environment variables
- Handle process lifecycle (background execution, signal handling)
- Provide user feedback and progress indicators
- Manage logging levels and output destinations

#### **Key Interfaces**
```rust
pub struct CliConfig {
    pub model_filepath: PathBuf,
    pub server_address: String,
    pub auth_token: String,
    pub log_level: LogLevel,
    pub background: bool,
}

impl CliConfig {
    pub fn parse() -> Result<Self, CliError> {
        // Argument parsing and validation
    }
    
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check model file existence, network port availability
    }
}
```

#### **Background Execution**
- Fork process for true background operation
- PID file management for process tracking
- Signal handling (SIGTERM, SIGINT) for graceful shutdown
- Systemd service file generation (optional)

#### **Configuration Management**
- Environment variable precedence over CLI arguments
- Configuration file support (TOML format)
- Runtime configuration validation
- Hot-reload support for development

### 2. HTTP API Server (pensieve-02)

#### **Web Framework Selection**
- **Warp**: High-performance, async HTTP server
- **Features**: Built-in CORS, compression, streaming support
- **Architecture**: Middleware-based request processing

#### **Endpoint Design**

##### **Primary: Anthropic Messages API**
```rust
// POST /v1/messages
{
  "model": "pensieve-phi-3-mini",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user", 
      "content": "Your message here"
    }
  ],
  "stream": false  // optional
}
```

##### **Secondary: OpenAI Compatibility**
```rust
// POST /v1/chat/completions
{
  "model": "pensieve-phi-3-mini",
  "messages": [...],
  "stream": true,
  "max_tokens": 1024
}
```

##### **Health Check**
```rust
// GET /health
{
  "status": "healthy",
  "model": "pensieve-phi-3-mini",
  "memory_usage": "4.2GB/16GB",
  "uptime": "2h 34m"
}
```

#### **Streaming Implementation**
- **Server-Sent Events**: `text/event-stream` content type
- **Chunked Encoding**: Efficient large response handling
- **Backpressure**: Request throttling for slow clients
- **Connection Management**: Timeout and cleanup handling

#### **Request Processing Pipeline**
1. **Middleware Stack**
   - CORS handling
   - Authentication (API key validation)
   - Request size limiting
   - Rate limiting (future enhancement)

2. **Validation Layer**
   - JSON schema validation
   - Parameter bounds checking
   - Model availability verification

3. **Routing**
   - Endpoint dispatching
   - Handler selection based on content-type
   - Error response generation

### 3. Request/Response Models (pensieve-03)

#### **Anthropic Message Format**
```rust
#[derive(Serialize, Deserialize, Debug)]
pub struct MessageRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MessageResponse {
    pub id: String,
    pub type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}
```

#### **Streaming Types**
```rust
#[derive(Serialize, Debug)]
pub struct StreamChunk {
    pub type: String,
    pub delta: Delta,
    pub usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}
```

#### **Error Response Format**
```rust
#[derive(Serialize, Debug)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize, Debug)]
pub struct ErrorDetail {
    pub r#type: String,
    pub message: String,
    pub code: Option<String>,
}
```

#### **OpenAI Compatibility Layer**
```rust
#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}
```

### 4. MLX Inference Engine (pensieve-04)

#### **Core Architecture**
- **Model Loading**: HuggingFace model loading with MLX quantization support
- **Tokenization**: HuggingFace tokenizer integration
- **Inference**: MLX-accelerated forward pass with Metal backend
- **Streaming**: Async token generation with proper handling

#### **Model Interface**
```rust
pub struct ModelEngine {
    model: mlx::nn::Model,
    tokenizer: Tokenizer,
    device: mlx::Device,
    config: ModelConfig,
}

impl ModelEngine {
    pub async fn load(model_path: &Path, device: Device) -> Result<Self> {
        // Load model with MLX
    }
    
    pub async fn generate(
        &mut self,
        input: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<StreamingResponse> {
        // Tokenize and generate tokens
    }
}
```

#### **Token Streaming**
```rust
pub struct StreamingResponse {
    stream: Pin<Box<dyn Stream<Item = Result<Token>> + Send>>,
}

impl Stream for StreamingResponse {
    type Item = Result<Token>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        // Async token generation
    }
}
```

#### **Memory Management**
- **Tensor Allocation**: Pre-allocated buffers for consistent performance
- **GPU Memory**: MLX variable management with Metal acceleration
- **Garbage Collection**: Automatic cleanup of intermediate tensors
- **Memory Pressure Monitoring**: Adaptive behavior under memory constraints

#### **Error Handling**
- **Model Loading**: Fallback to CPU with detailed error messages
- **Inference Errors**: Graceful handling with error propagation
- **Tokenization**: Robust error handling for malformed input
- **Resource Management**: Automatic cleanup on errors

### 5. Model Management (pensieve-05)

#### **Model Loading Strategy**
- **Lazy Loading**: Model loaded on first request
- **Cache Management**: In-memory cache for active models
- **Pre-loading**: Option to load model at startup
- **Hot Swapping**: Future support for model switching

#### **MLX Quantization Support**
- **4-bit Quantization**: Significant memory reduction (75%)
- **Model Format**: HuggingFace models with MLX quantization
- **Memory Mapping**: Efficient large file handling
- **Validation**: Pre-load model structure validation

#### **KV-Cache Optimization**
```rust
pub struct KVCache {
    key_cache: mlx::core::Tensor,
    value_cache: mlx::core::Tensor,
    max_seq_len: usize,
    current_len: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, hidden_size: usize, n_heads: usize) -> Result<Self> {
        // Allocate GPU memory for key-value cache
    }
    
    pub fn update(&mut self, new_keys: &Tensor, new_values: &Tensor) -> Result<()> {
        // Update cache with new tokens
    }
    
    pub fn clear(&mut self) {
        // Reset cache for new conversation
    }
}
```

#### **Context Window Management**
- **Sliding Window**: Efficient handling of long contexts
- **Token Position Tracking**: Proper attention mechanism maintenance
- **Memory Efficiency**: Eviction of old tokens when necessary
- **Configuration**: Configurable context window sizes

#### **Model Selection**
```rust
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub path: PathBuf,
    pub quantization: QuantizationLevel,
    pub context_length: usize,
    pub memory_requirements: usize,
}

impl ModelConfig {
    pub fn detect_requirements(&self) -> Result<usize> {
        // Estimate memory needs based on model size and quantization
    }
}
```

### 6. Metal GPU Optimization (pensieve-06)

#### **MLX Metal Backend Integration**
- **Device Selection**: Optimal M-series GPU utilization
- **Variable Management**: Efficient memory layout for MLX variables
- **Compute Operations**: MLX-optimized Metal operations
- **Performance**: Framework-level Metal optimization

#### **Memory Optimization**
```rust
pub struct MemoryPool {
    variables: Vec<mlx::core::Variable>,
    allocated: usize,
    max_size: usize,
}

impl MemoryPool {
    pub fn allocate(&mut self, size: usize) -> Result<&mlx::core::Variable> {
        // Pool allocation with fragmentation avoidance
    }
    
    pub fn deallocate(&mut self, variable: &mlx::core::Variable) {
        // Return variable to pool
    }
}
```

#### **Async Compute Streams**
- **MLX Async**: Native async support in MLX framework
- **Dependency Management**: Proper synchronization of GPU operations
- **Load Balancing**: Optimal utilization of GPU cores
- **Profiling**: Performance metrics collection

#### **MLX Optimization**
- **Model Optimization**: MLX model compilation and optimization
- **Quantization**: MLX-native 4-bit quantization
- **Memory Layout**: Optimized variable allocation patterns
- **Kernel Optimization**: Framework-level kernel optimization

#### **Performance Profiling**
```rust
pub struct Profiler {
    measurements: HashMap<String, Vec<Duration>>,
    active_timers: HashMap<String, Instant>,
}

impl Profiler {
    pub fn start(&mut self, name: String) {
        // Start timing operation
    }
    
    pub fn end(&mut self, name: String) -> Duration {
        // Stop timing and record
    }
    
    pub fn report(&self) -> String {
        // Generate performance report
    }
}
```

### 7. Configuration & Utils (pensieve-07)

#### **Environment Variable Management**
```rust
pub struct EnvironmentConfig {
    pub anthropic_base_url: String,
    pub anthropic_auth_token: String,
    pub model_path: Option<PathBuf>,
    pub log_level: String,
    pub max_concurrent_requests: usize,
}

impl EnvironmentConfig {
    pub fn load() -> Result<Self> {
        // Load with precedence: CLI > ENV > defaults
    }
}
```

#### **File I/O Operations**
- **Model Loading**: Efficient large file handling with memory mapping
- **Configuration Files**: TOML configuration parsing
- **Log Files**: Structured logging with rotation
- **Temporary Files**: Safe temporary file management

#### **Error Handling Utilities**
```rust
pub trait ResultExt<T> {
    fn with_context<C>(self, context: C) -> Result<T, ContextError>
    where
        C: Display;
}

pub struct ContextError {
    pub original: Box<dyn StdError>,
    pub context: String,
}
```

#### **Common Type Definitions**
- **Result Types**: Unified error handling across crates
- **Data Types**: Common structures shared across components
- **Configuration Types**: Shared configuration structures
- **Metric Types**: Common performance metrics

## Performance Architecture

### Memory Management for 16GB Apple Systems

#### **Memory Allocation Strategy**
```rust
pub struct MemoryManager {
    total_ram: usize,           // 16GB+ available
    model_memory: usize,        // ~4-8GB for 7B model
    kv_cache_memory: usize,     // ~1-2GB for cache
    token_buffer_memory: usize,  // ~100MB for streaming
    overhead_memory: usize,     // ~1GB for system overhead
    available_memory: usize,    // Remaining for concurrent requests
}
```

#### **Model Memory Optimization**
- **4-bit MLX Quantization**: Reduces memory usage by 75%
  - Phi-3 Mini 4-bit: ~1.5GB (vs 6GB full precision)
  - Mistral 7B 4-bit: ~4GB (vs 14GB full precision)
  - Deepseek Coder 4-bit: ~4-5GB (vs 15GB full precision)

#### **KV-Cache Management**
- **Dynamic Allocation**: Cache size grows with context length
- **Eviction Strategy**: LRU eviction for old tokens
- **Memory Pressure**: Adaptive cache sizing based on available memory
- **Batch Processing**: Efficient cache utilization for concurrent requests

#### **Garbage Collection**
- **Reference Counting**: Automatic cleanup of unused MLX variables
- **Explicit Cleanup**: Manual cleanup of large temporary objects
- **Memory Monitoring**: Continuous tracking of memory usage
- **Emergency Procedures**: Graceful degradation under extreme memory pressure

### MLX GPU Utilization Strategies

#### **GPU Memory Architecture**
- **Unified Memory**: Shared CPU/GPU memory (no copying needed)
- **Variable Optimization**: Efficient memory layout for MLX variables
- **Buffer Pooling**: Reused GPU buffers to avoid allocation overhead
- **Async Transfers**: Background memory operations

#### **Compute Optimization**
```rust
pub struct ComputeScheduler {
    command_queue: metal::CommandQueue,
    in_flight_operations: usize,
    max_concurrent: usize,
}

impl ComputeScheduler {
    pub fn schedule<F>(&mut self, operation: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        // Schedule operation on optimal command queue
    }
}
```

#### **MLX Optimization**
- **Model Compilation**: MLX model compilation and optimization
- **Quantization**: Native 4-bit quantization with MLX
- **Memory Coalescing**: Optimized memory access patterns
- **Metal Backend**: Native Metal backend for Apple Silicon

#### **Parallel Processing**
- **Multi-stream Processing**: Parallel execution of independent operations
- **Pipeline Parallelism**: Overlapping computation and memory transfers
- **Batch Processing**: Efficient handling of multiple requests
- **Load Balancing**: Optimal utilization of GPU cores

### Quantization and Model Optimization Pipeline

#### **MLX Quantization Process**
1. **Model Conversion**: HuggingFace → MLX format using MLX tools
2. **Quantization**: 4-bit quantization with MLX optimization
3. **Validation**: Quality and performance verification
4. **Compression**: Optional additional compression for distribution

#### **Quantization Types**
- **4-bit Integer**: Optimal for memory and performance
- **5-bit Float**: Alternative for quality-sensitive applications
- **8-bit Float**: Fallback for compatibility

#### **Optimization Techniques**
- **Weight Pruning**: Remove redundant weights
- **Knowledge Distillation**: Smaller models with comparable performance
- **Layer Fusion**: Combine operations to reduce memory access
- **Quantization Awareness**: Hardware-aware quantization parameters

### Concurrent Request Handling Architecture

#### **Request Processing Model**
```rust
pub struct RequestDispatcher {
    active_requests: usize,
    max_concurrent: usize,
    model_engine: Arc<ModelEngine>,
    memory_manager: Arc<MemoryManager>,
}

impl RequestDispatcher {
    pub async fn dispatch(&mut self, request: Request) -> Result<Response> {
        // Check resource availability
        // Reserve memory and GPU resources
        // Execute request with proper error handling
        // Cleanup resources on completion
    }
}
```

#### **Resource Management**
- **Token Bucket Algorithm**: Request rate limiting
- **Memory Quotas**: Per-request memory allocation
- **GPU Time Slicing**: Fair sharing of GPU resources
- **Priority Queuing**: Higher priority for shorter requests

#### **Load Balancing**
- **Request Distribution**: Across multiple inference pipelines
- **Dynamic Scaling**: Adjust based on system load
- **Adaptive Timeouts**: Based on current system performance
- **Graceful Degradation**: Reduce quality under heavy load

## Integration Architecture

### Claude Code Integration Patterns

#### **Environment Variable Configuration**
```bash
# Claude Code configuration
export ANTHROPIC_API_KEY="pensieve-local-key"
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_MODEL="pensieve-phi-3-mini"
```

#### **API Compatibility**
- **Endpoint Compatibility**: 100% match with Anthropic API
- **Request Format**: Identical message structure
- **Response Format**: Same streaming and non-streaming responses
- **Error Handling**: Consistent error codes and messages

#### **Authentication**
- **Simple Token**: Fixed token for local development
- **Header Format**: `Authorization: Bearer pensieve-local-key`
- **No External Dependencies**: Local-only authentication

### API Endpoint Compatibility

#### **Anthropic API Format**
```rust
// Exact match with Anthropic v1/messages
POST /v1/messages
Content-Type: application/json

{
  "model": "pensieve-phi-3-mini",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "stream": false
}
```

#### **OpenAI Fallback Format**
```rust
// OpenAI-compatible for broader tooling
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "pensieve-phi-3-mini",
  "messages": [...],
  "stream": true,
  "max_tokens": 1024
}
```

#### **Response Format Compatibility**
- **Streaming**: Server-Sent Events with `data: {json}` format
- **Non-streaming**: JSON response with identical structure
- **Error Responses**: Same error codes and message formats
- **Metadata**: Consistent usage statistics and model information

### Error Handling and Fallback Mechanisms

#### **Error Categories**
```rust
pub enum ErrorType {
    // Client Errors (4xx)
    ValidationError,
    AuthenticationError,
    RateLimitError,
    
    // Server Errors (5xx)
    ModelLoadingError,
    InferenceError,
    ResourceExhaustedError,
    
    // System Errors
    MemoryError,
    GpuError,
    NetworkError,
}
```

#### **Graceful Degradation**
1. **Memory Pressure**: Reduce batch size, enable quantization
2. **GPU Errors**: Fallback to CPU inference
3. **Model Errors**: Try alternative model or return error
4. **Network Errors**: Retry with exponential backoff

#### **Error Recovery**
- **Automatic Retry**: For transient errors (network, temporary resource exhaustion)
- **Model Reload**: Automatic reload if model becomes corrupted
- **Process Restart**: For unrecoverable errors with state preservation
- **User Notification**: Clear error messages with suggested actions

### Configuration Management

#### **Configuration Hierarchy**
1. **Command Line Arguments**: Highest precedence
2. **Environment Variables**: Medium precedence
3. **Configuration File**: Lowest precedence
4. **Defaults**: Fallback values

#### **Configuration File Format**
```toml
# pensieve.toml
[model]
path = "/path/to/phi-3-mlx-model"
quantization = "4bit"
context_length = 4096

[server]
address = "localhost:8000"
max_concurrent_requests = 5
auth_token = "pensieve-local-key"

[performance]
memory_limit_gb = 12
gpu_utilization = "high"
stream_buffer_size = 1024
```

## Deployment Architecture

### Single Binary Deployment Strategy

#### **Build Configuration**
```toml
[package]
name = "pensieve-local-llm-server"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[dependencies]
# Core dependencies
tokio = { version = "1.0", features = ["full"] }
warp = "0.3"
mlx = { version = "0.2", features = ["metal"] }
mlx-examples = { version = "0.2" }

# Apple-specific
metal = { version = "0.27", features = ["mps"], optional = true }

[features]
default = ["metal"]
metal = ["dep:metal"]

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true
```

#### **Release Process**
1. **Apple Silicon Builds**: macOS aarch64 (M1/M2/M3)
2. **Static Linking**: No external runtime dependencies
3. **Binary Size Optimization**: Remove debug symbols, strip binaries
4. **Asset Bundling**: Include default configuration and documentation

#### **Distribution Methods**
- **Binary Release**: Single executable with embedded assets
- **Homebrew Formula**: Easy installation via `brew install`
- **Docker Image**: Containerized deployment
- **Source Distribution**: Cargo workspace for custom builds

### Configuration File Management

#### **Default Configuration**
- **Embedded Defaults**: Built-in configuration for quick startup
- **Configuration Templates**: Example configurations for different models
- **Auto-generation**: Generate configs based on detected hardware
- **Validation**: Automatic config validation with helpful error messages

#### **Environment Integration**
- **System Integration**: macOS system service configuration
- **Path Resolution**: Smart config file discovery
- **Hot Reload**: Runtime configuration updates
- **Environment Overrides**: Per-environment customization

### Logging and Monitoring Setup

#### **Structured Logging**
```rust
#[derive(Serialize)]
pub struct LogEntry {
    timestamp: chrono::DateTime<chrono::Utc>,
    level: LogLevel,
    component: String,
    message: String,
    metadata: Option<serde_json::Value>,
}
```

#### **Log Levels**
- **ERROR**: Critical errors requiring immediate attention
- **WARN**: Warning conditions that should be addressed
- **INFO**: General information about operation
- **DEBUG**: Detailed debugging information
- **TRACE**: Fine-grained operational details

#### **Log Rotation**
- **Size-based**: Rotate when files reach maximum size
- **Time-based**: Daily or weekly rotation
- **Retention**: Keep last N log files
- **Compression**: Compress old log files

#### **Metrics Collection**
- **Performance Metrics**: Token generation speed, memory usage
- **Request Metrics**: Request count, response times, error rates
- **System Metrics**: CPU usage, GPU utilization, memory pressure
- **Business Metrics**: Active users, model usage patterns

### Process Management (Background Execution)

#### **Background Process Architecture**
- **Daemon Mode**: True background operation with PID file
- **Systemd Service**: Proper Linux service integration
- **Launchd Configuration**: macOS launch agent/service
- **Process Supervision**: Automatic restart on failure

#### **Signal Handling**
- **SIGTERM**: Graceful shutdown with active request completion
- **SIGINT**: Immediate shutdown with cleanup
- **SIGHUP**: Reload configuration (graceful)
- **SIGUSR1**: Dump diagnostic information

#### **State Persistence**
- **PID Files**: Process identification and management
- **State Files**: Save/load operation state
- **Checkpoint Files**: Model loading progress tracking
- **Recovery Procedures**: Automatic restoration after crash

#### **Resource Management**
- **CPU Limits**: Set CPU affinity and priorities
- **Memory Limits**: Enforce memory usage constraints
- **File Descriptors**: Monitor and limit open file handles
- **Network Ports**: Manage port allocation and release

## Implementation Roadmap

### Phase-wise Development Approach

#### **Phase 1: Core Infrastructure (Week 1-2)**
**Goal**: Establish working MLX + Apple Silicon foundation

**Tasks**:
1. **Setup Development Environment**
   ```bash
   # Install Rust toolchain
   rustup default stable
   rustup target set aarch64-apple-darwin
   
   # Install Xcode command line tools
   xcode-select --install
   
   # Verify MLX support
   python3 -c "import mlx; print('MLX available')"
   ```

2. **Model Loading Implementation**
   - Load HuggingFace models with MLX
   - Basic Metal GPU acceleration
   - Tokenization with HuggingFace tokenizers
   - Simple inference loop

3. **Basic API Server**
   - Warp HTTP server setup
   - Simple hello-world endpoint
   - Basic request/response handling
   - Error handling foundation

4. **Memory Management**
   - Basic memory tracking
   - Model loading optimization
   - Simple cache implementation

**Success Criteria**:
- ✅ Load Phi-3 Mini 4-bit model successfully
- ✅ Generate first token in <2 seconds
- ✅ Basic HTTP server running on localhost
- ✅ No memory leaks during operation

#### **Phase 2: API Development (Week 3-4)**
**Goal**: Full Anthropic API compatibility

**Tasks**:
1. **API Endpoint Implementation**
   - `/v1/messages` endpoint
   - `/v1/chat/completions` fallback
   - Request validation and parsing
   - Response formatting

2. **Streaming Support**
   - Server-Sent Events implementation
   - Async token streaming
   - Proper backpressure handling
   - Connection management

3. **Authentication**
   - Simple token validation
   - Header parsing and verification
   - Error response generation

4. **Error Handling**
   - Comprehensive error types
   - Proper HTTP status codes
   - Error response formatting
   - User-friendly error messages

**Success Criteria**:
- ✅ 100% Anthropic API compatibility
- ✅ Streaming token generation working
- ✅ Proper error handling for all scenarios
- ✅ Pass Anthropic API conformance tests

#### **Phase 3: Optimization & Production (Week 5-6)**
**Goal**: Production-ready performance and reliability

**Tasks**:
1. **Metal GPU Optimization**
   - MLX Metal backend optimization
   - Memory pooling and optimization
   - Async compute streams
   - Performance profiling

2. **Memory Management**
   - Advanced KV-cache optimization
   - Memory pressure monitoring
   - Garbage collection optimization
   - Concurrent request handling

3. **Performance Tuning**
   - Token generation speed optimization
   - Memory usage reduction
   - Throughput improvement
   - Latency reduction

4. **Production Features**
   - Background process management
   - Configuration file support
   - Logging and monitoring
   - CLI interface completion

**Success Criteria**:
- ✅ First token <500ms
- ✅ 25+ tokens/second throughput
- ✅ <12GB memory usage peak
- ✅ Background process working
- ✅ Complete CLI interface

#### **Phase 4: Testing & Validation (Week 7-8)**
**Goal**: Comprehensive testing and validation

**Tasks**:
1. **Unit Testing**
   - Component-level testing
   - Error condition testing
   - Performance benchmarking
   - Memory leak detection

2. **Integration Testing**
   - API endpoint testing
   - Streaming functionality testing
   - Authentication testing
   - Configuration testing

3. **Performance Testing**
   - Load testing with concurrent requests
   - Memory usage under load
   - GPU utilization monitoring
   - Long-running stability tests

4. **User Acceptance Testing**
   - Claude Code integration testing
   - Real-world usage scenarios
   - Error recovery testing
   - Documentation validation

**Success Criteria**:
- ✅ 100% test coverage
- ✅ Pass all integration tests
- ✅ Stable under load testing
- ✅ Claude Code integration working
- ✅ Complete documentation

### Key Milestones and Dependencies

#### **Milestone 1: Working Prototype (End Week 2)**
- **Dependencies**: Rust toolchain, MLX framework, Apple Silicon hardware
- **Deliverables**: Basic model loading, simple API server
- **Risk Factors**: MLX compatibility, model format issues

#### **Milestone 2: API Compatibility (End Week 4)**
- **Dependencies**: Milestone 1, Warp framework, tokenizers
- **Deliverables**: Full Anthropic API, streaming support
- **Risk Factors**: API specification changes, streaming complexity

#### **Milestone 3: Production Ready (End Week 6)**
- **Dependencies**: Milestone 2, MLX optimization, memory management
- **Deliverables**: Performance targets, background execution, CLI
- **Risk Factors**: Performance optimization challenges, memory issues

#### **Milestone 4: Release Candidate (End Week 8)**
- **Dependencies**: Milestone 3, comprehensive testing, documentation
- **Deliverables**: Complete testing, user documentation, release build
- **Risk Factors**: Test failures, performance issues, user feedback

### Risk Mitigation Strategies

#### **Technical Risks**
1. **MLX Framework Compatibility**
   - **Risk**: MLX version changes, Apple Silicon compatibility
   - **Mitigation**: Test on multiple Apple generations, provide CPU fallback
   - **Contingency**: Monitor MLX updates and maintain compatibility layer

2. **Memory Constraints**
   - **Risk**: 16GB insufficient for larger models
   - **Mitigation**: Aggressive quantization, memory optimization
   - **Contingency**: Implement memory pressure handling and graceful degradation

3. **Performance Targets**
   - **Risk**: Unable to achieve 25+ tokens/second
   - **Mitigation**: MLX optimization, batch processing
   - **Contingency**: Adjust targets based on actual hardware performance

4. **API Compatibility**
   - **Risk**: Anthropic API changes break compatibility
   - **Mitigation**: Test against actual Anthropic API, versioned endpoints
   - **Contingency**: Maintain compatibility layer with version detection

#### **Project Risks**
1. **Timeline Delays**
   - **Risk**: Development takes longer than expected
   - **Mitigation**: Weekly milestone reviews, scope prioritization
   - **Contingency**: Phase-based delivery with minimum viable features

2. **Resource Constraints**
   - **Risk**: Hardware limitations testing
   - **Mitigation**: Cloud-based testing, simulator usage
   - **Contingency**: Partner with Apple Silicon hardware owners for testing

3. **Technical Debt**
   - **Risk**: Rapid development leads to code quality issues
   - **Mitigation**: Regular code reviews, comprehensive testing
   - **Contingency**: Refactoring sprints between phases

### Testing and Validation Framework

#### **Unit Testing Strategy**
- **Component Testing**: Individual component isolation
- **Mock Objects**: Mock external dependencies
- **Property Testing**: Random input testing
- **Performance Testing**: Component-level benchmarks

#### **Integration Testing Approach**
- **API Testing**: HTTP endpoint validation
- **Streaming Testing**: Connection and data flow validation
- **Authentication Testing**: Security validation
- **Error Handling Testing**: Error condition scenarios

#### **Performance Testing Methodology**
- **Benchmarking**: Standard performance metrics
- **Load Testing**: Concurrent user simulation
- **Stress Testing**: Extreme condition testing
- **Profiling**: Detailed performance analysis

#### **User Acceptance Testing**
- **Alpha Testing**: Early internal testing
- **Beta Testing**: Selected user feedback
- **Production Testing**: Real-world usage
- **Regression Testing**: Version compatibility

## Conclusion

The Pensieve Local LLM Server architecture provides a comprehensive, production-ready solution for running local LLM inference on Apple Silicon hardware. By leveraging the MLX framework with Metal GPU acceleration, we achieve optimal performance while maintaining memory efficiency for 16GB+ systems.

The modular crate architecture ensures maintainability and extensibility, while the Anthropic API compatibility guarantees seamless integration with Claude Code. The comprehensive performance optimization and error handling make this solution suitable for production deployment.

With the phased development approach and risk mitigation strategies, this architecture provides a clear path to delivering a high-performance, reliable local LLM server that enables users to enjoy cloud-like AI experiences with local privacy and performance benefits.

**Next Steps**:
1. Begin Phase 1 development with core infrastructure setup
2. Clone and analyze reference MLX implementations
3. Acquire and test MLX-optimized model files
4. Establish performance benchmarks and monitoring

The architecture is designed to be both immediately practical and extensible for future enhancements, ensuring long-term viability as the LLM ecosystem evolves.

---

**Document Version**: 1.0 (MLX Transition)
**Last Updated**: October 28, 2025
**Next Review**: Upon Phase 1 completion
**Framework**: MLX for Apple Silicon
**Target**: M1/M2/M3 16GB+ Systems
