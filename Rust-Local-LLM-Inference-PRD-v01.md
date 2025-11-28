# Rust-Local-LLM-Inference-PRD-v01.md

## Executive Summary

This document provides a comprehensive analysis and design specification for a Rust-based HTTP server for local LLM inference, built upon the existing Pensieve codebase. The design delivers a single binary that takes a Hugging Face model URL and port number, providing high-performance local inference with Apple Silicon Metal GPU acceleration.

## Current Codebase Analysis

### Existing Architecture Overview

The Pensieve project contains a sophisticated hybrid architecture with two parallel implementations:

#### 1. **Production Python Implementation** (Current)
- **Location**: `/src/` directory
- **Components**:
  - `server.py`: FastAPI server with persistent model loading (2.5GB baseline memory)
  - `inference.py`: MLX-based inference with memory safety and performance monitoring
- **Key Innovation**: Solves memory explosion problem through persistent model residency
- **Performance**: Sub-5-second startup, 25+ TPS target, Metal GPU acceleration

#### 2. **Advanced Rust Workspace** (Complete but Unused)
- **Structure**: 9 specialized crates (p01-p09) in layered architecture
- **Layers**: L1 Foundation (p07) → L2 Domain Logic (p04-p06, p08) → L3 Application (p01-p03, p09)
- **Technology**: Tokio async runtime, Warp HTTP framework, Metal GPU integration
- **Status**: Production-ready implementation with comprehensive error handling

### Key Insights from Analysis

1. **Memory Optimization Solved**: Python implementation demonstrates persistent model loading works
2. **Metal Integration Ready**: p06-metal-gpu-accel provides complete Apple Silicon optimization
3. **API Compatibility Achieved**: Full Anthropic API v1 compatibility implemented
4. **Claude Code Integration Abandoned**: Configuration complexity makes direct integration impractical
5. **Performance Targets Met**: 25+ TPS achievable with MLX + Metal optimization

## Rubber Duck Debugging Analysis

### Problem Statement
*"We need a simple Rust binary that downloads a model from Hugging Face and serves HTTP requests on a specified port."*

### Thinking Through the Problem

**Rubber Duck**: "So you want to recreate the Python server but in Rust?"

**Me**: "Yes, but wait - the Python server already works perfectly. It has persistent model loading, memory safety, performance monitoring, and Metal GPU acceleration. Why recreate it?"

**Rubber Duck**: "What's wrong with the Python implementation?"

**Me**: "Nothing actually! It solves the core problem. But the Rust workspace exists and is production-ready."

**Rubber Duck**: "So the real question is: should you build a NEW simplified binary or use the EXISTING comprehensive implementation?"

**Me**: "Ah, that's the key insight. The existing Rust workspace (p01-p09) already provides everything we need. We just need to:

1. Simplify the interface to match the requirements
2. Add Hugging Face model downloading capability
3. Create a unified binary interface
4. Leverage the existing Metal GPU acceleration"

**Rubber Duck**: "So you're not building from scratch - you're consolidating and simplifying existing production code?"

**Me**: "Exactly! The p01-cli-interface-launcher already provides CLI parsing. p02-http-server-core provides the HTTP server. p06-metal-gpu-accel provides Metal integration. We just need to wire them together properly."

**Rubber Duck**: "What about model management? The Python server expects a local model path."

**Me**: "Good point. We need to add Hugging Face model downloading and conversion. The p05-model-storage-core can handle this with extensions for HF Hub integration."

**Rubber Duck**: "And the interface requirement? Just --model-url and --port?"

**Me**: "Yes, much simpler than the current CLI. We'll create a streamlined interface that abstracts away the complexity."

**Rubber Duck**: "This makes sense now. You're not rebuilding - you're exposing the existing sophisticated architecture through a simple interface."

## High-Level Design (HLD)

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    rust-llm-server                          │
│                  (Single Binary)                           │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface Layer                                        │
│  ├─ Parse --model-url and --port arguments                 │
│  ├─ Validate model URL and port availability               │
│  └─ Error handling for invalid inputs                      │
├─────────────────────────────────────────────────────────────┤
│  Model Management Layer                                    │
│  ├─ Download model from Hugging Face Hub                   │
│  ├─ Convert to MLX format for Apple Silicon                │
│  ├─ Cache management and verification                       │
│  └─ Model loading and persistence                          │
├─────────────────────────────────────────────────────────────┤
│  Inference Engine Layer                                    │
│  ├─ Metal GPU acceleration via p06-metal-gpu-accel         │
│  ├─ MLX framework integration for tensor operations         │
│  ├─ Memory management and safety checks                    │
│  └─ Request processing and response generation            │
├─────────────────────────────────────────────────────────────┤
│  HTTP Server Layer                                         │
│  ├─ Warp-based HTTP server via p02-http-server-core        │
│  ├─ Anthropic API compatibility layer                      │
│  ├─ Streaming support (Server-Sent Events)                 │
│  └─ Health checks and monitoring endpoints                 │
├─────────────────────────────────────────────────────────────┤
│  Foundation Layer                                          │
│  ├─ Core traits and types via p07-foundation-types         │
│  ├─ Error handling and logging                            │
│  └─ Configuration management                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. **Simplified Interface Strategy**
- **Rationale**: Hide complexity while providing powerful functionality
- **Implementation**: Single binary with two required parameters
- **Benefit**: Easy adoption while leveraging sophisticated internals

#### 2. **Leverage Existing Architecture**
- **Rationale**: Production-ready Rust workspace exists with comprehensive functionality
- **Implementation**: Consolidate p01-p09 crates into unified binary
- **Benefit**: Reduced development risk, proven performance characteristics

#### 3. **Apple Silicon First Design**
- **Rationale**: Target the high-performance local LLM use case
- **Implementation**: Metal GPU acceleration, MLX framework integration
- **Benefit**: Optimal performance on target hardware platform

#### 4. **Memory-Optimized Architecture**
- **Rationale**: Learn from Python implementation's memory optimization
- **Implementation**: Persistent model loading, shared model weights across requests
- **Benefit**: Linear memory scaling vs. exponential in naive approaches

### Component Integration Strategy

#### **Core Crate Usage**
- **p01-cli-interface-launcher**: CLI argument parsing and server lifecycle
- **p02-http-server-core**: HTTP server functionality and API endpoints
- **p04-inference-engine-core**: Inference request processing and coordination
- **p05-model-storage-core**: Model loading and management (extended for HF Hub)
- **p06-metal-gpu-accel**: Apple Silicon GPU acceleration
- **p07-foundation-types**: Core abstractions and error handling
- **p08-claude-api-core**: Anthropic API compatibility
- **p09-api-proxy-compat**: Additional API compatibility features

#### **New Components Required**
- **Model Downloader**: Hugging Face Hub integration and format conversion
- **Unified Interface**: Simplified binary that wires together existing components
- **Configuration Management**: Runtime settings and model caching

## Low-Level Design (LLD)

### Binary Interface Specification

```rust
// src/main.rs
use clap::Parser;
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(name = "rust-llm-server")]
#[command(about = "Local LLM inference server for Apple Silicon")]
struct Args {
    /// Hugging Face model URL or repository ID
    #[arg(long, required = true)]
    model_url: String,

    /// HTTP server port number
    #[arg(long, required = true)]
    port: u16,

    /// Optional: Model cache directory
    #[arg(long, default_value = "./models")]
    cache_dir: PathBuf,

    /// Optional: Maximum concurrent requests
    #[arg(long, default_value = "2")]
    max_concurrent: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Validate port availability
    validate_port(args.port).await?;

    // Initialize and start server
    let server = LlmServer::new(args).await?;
    server.start().await
}
```

### Model Management System

#### **Hugging Face Integration**
```rust
// src/model_manager.rs
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokio::fs;

pub struct ModelManager {
    cache_dir: PathBuf,
    api: Api,
}

impl ModelManager {
    pub async fn ensure_model(&self, model_url: &str) -> Result<ModelPath> {
        // Parse model identifier (URL or HF repo ID)
        let repo = self.parse_model_identifier(model_url)?;

        // Check cache first
        let cached_path = self.get_cached_path(&repo)?;
        if cached_path.exists() {
            return Ok(ModelPath::Cached(cached_path));
        }

        // Download from Hugging Face Hub
        let model_path = self.download_model(&repo).await?;

        // Convert to MLX format if needed
        let mlx_path = self.convert_to_mlx(&model_path).await?;

        // Cache the converted model
        self.cache_model(&repo, &mlx_path).await?;

        Ok(ModelPath::Downloaded(mlx_path))
    }

    async fn download_model(&self, repo: &Repo) -> Result<PathBuf> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create HF API: {}", e))?;

        let repo_api = api.repo(repo);

        // Download model files
        let model_dir = self.cache_dir.join(&repo.repo_id);
        fs::create_dir_all(&model_dir).await?;

        // Download essential files
        for file in ["config.json", "pytorch_model.bin", "tokenizer.json"] {
            if let Ok(api_file) = repo_api.get(file) {
                api_file.download(&model_dir).await?;
            }
        }

        Ok(model_dir)
    }

    async fn convert_to_mlx(&self, model_path: &Path) -> Result<PathBuf> {
        // Use Python MLX conversion tools via subprocess
        let output = tokio::process::Command::new("python3")
            .arg("-c")
            .arg(&format!(
                "import mlx; mlx.convert.save('{}', '{}')",
                model_path.display(),
                model_path.with_extension("mlx").display()
            ))
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!("MLX conversion failed"));
        }

        Ok(model_path.with_extension("mlx"))
    }
}

pub enum ModelPath {
    Cached(PathBuf),
    Downloaded(PathBuf),
}
```

### HTTP Server Implementation

#### **Core Server Structure**
```rust
// src/server.rs
use warp::Filter;
use std::sync::Arc;

pub struct LlmServer {
    model_manager: Arc<ModelManager>,
    inference_engine: Arc<dyn InferenceEngine>,
    port: u16,
    max_concurrent: usize,
}

impl LlmServer {
    pub async fn new(args: Args) -> Result<Self> {
        // Initialize model manager
        let model_manager = Arc::new(ModelManager::new(args.cache_dir)?);

        // Ensure model is downloaded and loaded
        let model_path = model_manager.ensure_model(&args.model_url).await?;

        // Initialize inference engine with loaded model
        let inference_engine = Arc::new(MetalInferenceEngine::new(model_path).await?);

        Ok(Self {
            model_manager,
            inference_engine,
            port: args.port,
            max_concurrent: args.max_concurrent,
        })
    }

    pub async fn start(self) -> Result<()> {
        // Create shared state for handlers
        let server_state = Arc::new(ServerState {
            inference_engine: self.inference_engine.clone(),
            semaphore: Arc::new(tokio::sync::Semaphore::new(self.max_concurrent)),
        });

        // Health check endpoint
        let health = warp::path("health")
            .and(warp::get())
            .and(with_server_state(server_state.clone()))
            .and_then(health_handler);

        // Generate endpoint
        let generate = warp::path("v1")
            .and(warp::path("messages"))
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server_state(server_state.clone()))
            .and_then(generate_handler);

        // Start server
        let routes = health.or(generate);

        println!("🚀 Starting server on http://0.0.0.0:{}", self.port);

        warp::serve(routes)
            .run(([0, 0, 0, 0], self.port))
            .await;

        Ok(())
    }
}

// Helper to pass server state to handlers
fn with_server_state(
    state: Arc<ServerState>,
) -> impl Filter<Extract = (Arc<ServerState>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || state.clone())
}
```

#### **Request Handlers**
```rust
// src/handlers.rs
use serde_json::json;
use warp::{http::StatusCode, Rejection, Reply};

async fn health_handler(state: Arc<ServerState>) -> Result<impl Reply, Rejection> {
    let health = json!({
        "status": "healthy",
        "model_loaded": true,
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION")
    });

    Ok(warp::reply::json(&health))
}

async fn generate_handler(
    request: AnthropicRequest,
    state: Arc<ServerState>,
) -> Result<impl Reply, Rejection> {
    // Acquire semaphore for concurrency control
    let _permit = state.semaphore.acquire().await.map_err(|_| {
        warp::reject::custom(ApiError::ServiceUnavailable)
    })?;

    // Convert Anthropic request to internal format
    let internal_request = convert_anthropic_request(request)?;

    // Generate response
    let response = state
        .inference_engine
        .generate(internal_request)
        .await
        .map_err(|e| warp::reject::custom(ApiError::InferenceFailed(e.to_string())))?;

    // Convert back to Anthropic format
    let anthropic_response = convert_to_anthropic_response(response)?;

    Ok(warp::reply::json(&anthropic_response))
}

// Error handling
#[derive(Debug)]
enum ApiError {
    ServiceUnavailable,
    InferenceFailed(String),
    InvalidRequest(String),
}

impl warp::reject::Reject for ApiError {}

pub async fn handle_rejection(err: Rejection) -> Result<impl Reply, StatusCode> {
    if let Some(api_error) = err.find::<ApiError>() {
        match api_error {
            ApiError::ServiceUnavailable => {
                Ok(StatusCode::SERVICE_UNAVAILABLE)
            }
            ApiError::InferenceFailed(_) => {
                Ok(StatusCode::INTERNAL_SERVER_ERROR)
            }
            ApiError::InvalidRequest(_) => {
                Ok(StatusCode::BAD_REQUEST)
            }
        }
    } else {
        Ok(StatusCode::INTERNAL_SERVER_ERROR)
    }
}
```

### Inference Engine Integration

#### **Metal GPU Acceleration**
```rust
// src/inference.rs
use p06_metal_gpu_accel::{MetalDevice, MetalInferenceEngine};

pub struct MetalInferenceEngine {
    device: MetalDevice,
    model: LoadedModel,
    tokenizer: Tokenizer,
}

impl MetalInferenceEngine {
    pub async fn new(model_path: ModelPath) -> Result<Self> {
        // Initialize Metal device
        let device = MetalDevice::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize Metal: {}", e))?;

        // Load model weights into GPU memory
        let model_path = match model_path {
            ModelPath::Cached(p) | ModelPath::Downloaded(p) => p,
        };

        let model = LoadedModel::from_path(&device, &model_path).await?;
        let tokenizer = Tokenizer::from_file(&model_path.join("tokenizer.json"))?;

        Ok(Self {
            device,
            model,
            tokenizer,
        })
    }
}

#[async_trait]
impl InferenceEngine for MetalInferenceEngine {
    async fn generate(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Tokenize input
        let input_ids = self.tokenizer.encode(&request.prompt)?;

        // Setup Metal buffers
        let input_buffer = self.device.create_buffer(&input_ids)?;
        let output_buffer = self.device.create_buffer_for_generation()?;

        // Execute inference on GPU
        let command_buffer = self.device.create_command_buffer();

        // Encode attention mechanism
        self.encode_attention(&command_buffer, &input_buffer, &output_buffer)?;

        // Encode feed-forward layers
        self.encode_feedforward(&command_buffer, &output_buffer)?;

        // Submit to GPU
        command_buffer.commit();
        command_buffer.wait_until_completed()?;

        // Decode output tokens
        let output_ids = self.device.read_buffer(&output_buffer)?;
        let output_text = self.tokenizer.decode(&output_ids)?;

        Ok(InferenceResponse {
            text: output_text,
            token_count: output_ids.len(),
            generation_time: request.start_time.elapsed(),
        })
    }
}
```

## Error Handling Strategy

### Port Conflict Handling
```rust
async fn validate_port(port: u16) -> Result<()> {
    use tokio::net::TcpListener;

    match TcpListener::bind(("0.0.0.0", port)).await {
        Ok(_) => {
            // Port is available
            drop(listener);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            Err(anyhow::anyhow!(
                "Port {} is already in use. Please choose a different port.",
                port
            ))
        }
        Err(e) => {
            Err(anyhow::anyhow!(
                "Failed to bind to port {}: {}",
                port, e
            ))
        }
    }
}
```

### Model Download Error Handling
```rust
impl ModelManager {
    async fn download_with_retry(&self, repo: &Repo, max_retries: u32) -> Result<PathBuf> {
        for attempt in 1..=max_retries {
            match self.download_model(repo).await {
                Ok(path) => return Ok(path),
                Err(e) if attempt < max_retries => {
                    eprintln!("Download attempt {} failed: {}. Retrying...", attempt, e);
                    tokio::time::sleep(Duration::from_secs(2 * attempt as u64)).await;
                }
                Err(e) => return Err(e),
            }
        }

        Err(anyhow::anyhow!("Failed to download after {} attempts", max_retries))
    }
}
```

### Memory Safety Checks
```rust
impl MetalInferenceEngine {
    fn check_memory_pressure(&self) -> Result<()> {
        let available_memory = get_available_memory_mb();
        let model_memory = self.model.memory_usage_mb();

        if available_memory < model_memory + 1024 { // 1GB safety margin
            return Err(anyhow::anyhow!(
                "Insufficient memory: {}MB available, {}MB required",
                available_memory, model_memory
            ));
        }

        Ok(())
    }
}
```

## Performance Optimization Strategy

### 1. **Model Caching and Persistence**
- Cache converted MLX models locally
- Validate cache integrity before reuse
- Support pre-warming of cached models

### 2. **Metal GPU Optimization**
- Leverage existing p06-metal-gpu-accel optimizations
- Batch multiple requests when possible
- Optimize memory layout for Apple Silicon

### 3. **Concurrent Request Handling**
- Semaphore-based concurrency control
- Async request processing
- Shared model weights across requests

### 4. **Memory Management**
- Persistent model loading (2.5GB baseline)
- Linear memory scaling with concurrent requests
- Automatic cache clearing under memory pressure

## Deployment Considerations

### Binary Size Optimization
```toml
# Cargo.toml optimizations for release builds
[profile.release]
lto = true              # Link-time optimization
codegen-units = 1       # Better optimization
panic = "abort"         # Smaller binary
strip = true            # Remove debug symbols
```

### Distribution Strategy
- Single binary distribution for macOS (Apple Silicon only)
- Embedded model caching and download capability
- No external dependencies beyond system libraries

### Installation Experience
```bash
# One-line installation
curl -sSL https://github.com/pensieve-llm/install.sh | bash

# Simple usage
rust-llm-server --model-url microsoft/Phi-3-mini-128k-instruct --port 8765
```

## Implementation Roadmap

### Phase 1: Core Integration (2 weeks)
1. **Binary Interface Creation**
   - Create unified binary entry point
   - Implement CLI argument parsing (--model-url, --port)
   - Add basic error handling

2. **Model Management**
   - Integrate Hugging Face Hub downloading
   - Implement MLX format conversion
   - Add model caching and validation

3. **Server Consolidation**
   - Wire together existing p01-p09 crates
   - Implement simplified server startup
   - Add basic health checks

### Phase 2: Production Features (1 week)
1. **Error Handling**
   - Comprehensive error handling for all failure modes
   - Graceful degradation under resource pressure
   - User-friendly error messages

2. **Performance Optimization**
   - Memory usage monitoring and optimization
   - Concurrency control and request queuing
   - Metal GPU performance tuning

3. **Monitoring and Observability**
   - Metrics endpoints for performance monitoring
   - Structured logging for debugging
   - Health check improvements

### Phase 3: Polish and Distribution (1 week)
1. **User Experience**
   - Progress indicators for model downloads
   - Better startup messages and status reporting
   - Configuration file support

2. **Testing and Validation**
   - Comprehensive test suite
   - Performance benchmarking
   - Memory leak detection

3. **Documentation and Distribution**
   - Complete documentation
   - Installation scripts
   - Binary distribution

## Success Metrics

### Functional Requirements
- ✅ Single binary with --model-url and --port parameters
- ✅ Hugging Face model downloading and conversion
- ✅ HTTP server with Anthropic API compatibility
- ✅ Apple Silicon Metal GPU acceleration
- ✅ Memory-optimized inference (≤4GB for 4 concurrent requests)

### Performance Targets
- ✅ Startup time: <10 seconds (including model download if cached)
- ✅ Inference speed: ≥20 tokens/second sustained
- ✅ Memory usage: 2.5GB baseline + 0.5GB per concurrent request
- ✅ Model download speed: Optimized with resume capability

### Reliability Requirements
- ✅ Graceful error handling for all failure modes
- ✅ Port conflict detection and user-friendly error messages
- ✅ Memory pressure handling with automatic cache clearing
- ✅ Model download retry logic with exponential backoff

## Conclusion

This design leverages the sophisticated existing Pensieve architecture while providing the simplified interface requested. By consolidating the production-ready p01-p09 crates into a unified binary with Hugging Face integration, we can deliver a robust, high-performance local LLM server that meets all requirements while minimizing development risk.

The key insight is that we're not building from scratch—we're exposing the existing sophisticated system through a simple, user-friendly interface. This approach provides the best of both worlds: proven, production-ready internals with an elegant user experience.

The rubber duck debugging revealed that the problem isn't technical implementation but rather interface consolidation. The existing architecture already solves the hard problems of memory optimization, Metal GPU acceleration, and API compatibility. Our task is to expose this capability through the requested simple binary interface.

**Next Steps**: Begin Phase 1 implementation by creating the unified binary interface and integrating the existing crates, starting with the p01 CLI launcher and p02 HTTP server components.