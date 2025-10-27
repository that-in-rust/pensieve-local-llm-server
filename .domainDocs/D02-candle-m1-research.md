# D02: Candle-Only M1 Local LLM Server Research

**Project**: Pensieve Local LLM Server
**Framework**: Candle ONLY (no ONNX, no external engines)
**Target**: Apple M1 (16GB+ RAM) with Metal GPU
**Goal**: Local server with Anthropic API for Claude Code integration

## Core Strategy: Candle-First Architecture

Focus exclusively on **Candle framework** with **Metal GPU optimization** for Apple M1, creating a lightweight local server that provides Anthropic-compatible API endpoints.

## Essential Repositories (Candle Only)

### 🔥 **Foundation Libraries**

#### 1. **huggingface/candle** ⭐ 18.4k
- **Purpose**: Core ML framework for Rust
- **Why Essential**: Official Candle implementation with Metal support
- **M1 Features**:
  - Native Metal kernels for Apple silicon
  - PyTorch-like syntax with Rust performance
  - Model support: Mistral, LLaMA, Phi, Gemma
  - GGUF quantization support
- **Must Clone**: ✅ YES

#### 2. **GarthDB/metal-candle** ⭐
- **Purpose**: Production Metal backend for Apple Silicon
- **Why Critical**: Zero-overhead Rust-to-Metal calls
- **M1 Optimizations**:
  - Native M-series chip performance
  - KV-Cache: ~173 MB for 2048 tokens
  - Single binary, no Python runtime
  - Training overhead: 5-10% vs base model
- **Must Clone**: ✅ YES

### 🚀 **Server Implementation References**

#### 3. **EricLBuehler/candle-vllm** ⭐ 501
- **Purpose**: OpenAI compatible server using Candle
- **Why Essential**: Production-ready API server implementation
- **Key Features**:
  - OpenAI-compatible API endpoints
  - Mac/Metal and multi-GPU support
  - Continuous batching and PagedAttention
  - 4-bit quantization (GPTQ/Marlin)
  - Performance: 115 tks/s for 8B models
- **Must Clone**: ✅ YES

#### 4. **fcn94/llm_stream_endpoint**
- **Purpose**: Simple REST API with streaming
- **Why Useful**: Minimal implementation patterns
- **Architecture**:
  - Warp + Candle + Tokio
  - GGUF file support
  - Streaming endpoint at `/token_stream`
  - Supports Mistral 7B, Phi-2, LLaMA2
- **Must Clone**: ✅ YES

### 🛠️ **Application Examples**

#### 5. **shettysach/CandleMist**
- **Purpose**: Fullstack chatbot with Candle
- **Why Useful**: Complete production example
- **Stack**: Actix + Tokio + Leptos + TailwindCSS
- **Features**: Quantized Mistral 7B, Metal GPU support
- **Must Clone**: ✅ YES

#### 6. **tbogdala/ai_notepad**
- **Purpose**: Lightweight GGUF model runner
- **Why Useful**: Simple deployment patterns
- **Features**: Metal/CUDA acceleration, egui interface
- **Optimization**: Automatic model downloads
- **Must Clone**: ✅ YES

#### 7. **EricLBuehler/candle-lora**
- **Purpose**: Efficient LoRA fine-tuning for Candle
- **Why Future-Proof**: Custom model adaptation capabilities
- **Features**: Multiple transformer support, weight merging
- **Must Clone**: ✅ YES

## Technical Architecture

### Core Components

```rust
┌─────────────────────────────────────────────────┐
│              Pensieve LLM Server                │
├─────────────────────────────────────────────────┤
│  HTTP API Layer (Warp/Actix)                    │
│  - /v1/messages (Anthropic format)             │
│  - /v1/chat/completions (OpenAI fallback)      │
│  - Streaming support                           │
├─────────────────────────────────────────────────┤
│  Request/Response Models (Serde)               │
│  - Anthropic message format                    │
│  - OpenAI compatibility layer                 │
│  - Token streaming structures                 │
├─────────────────────────────────────────────────┤
│  Candle Inference Engine                       │
│  - Metal GPU backend                          │
│  - GGUF model loading                         │
│  - Memory management for 16GB RAM             │
│  - Token generation with streaming            │
├─────────────────────────────────────────────────┤
│  Model Management                              │
│  - Quantized model loading (4-bit GGUF)       │
│  - KV-Cache optimization                      │
│  - Context window management                  │
└─────────────────────────────────────────────────┘
```

### Apple M1 Optimization Strategy

#### Memory Management (16GB Constraint)
1. **Model Quantization**: 4-bit GGUF format
   - Mistral 7B: ~4GB RAM
   - Phi-2: ~1.5GB RAM
   - LLaMA 7B: ~4GB RAM

2. **KV-Cache Optimization**:
   - Efficient attention mechanisms
   - Context window: 2048-4096 tokens
   - Cache size: ~200-500MB

3. **Metal GPU Usage**:
   - Offload computation to GPU cores
   - Unified memory architecture
   - Async computation streams

#### Performance Targets
- **First Token**: <500ms
- **Throughput**: 15-30 tokens/second
- **Memory Usage**: <12GB peak
- **Concurrent Requests**: 3-5 simultaneous

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
```bash
# Clone essential repositories
git clone https://github.com/huggingface/candle.git
git clone https://github.com/GarthDB/metal-candle.git
git clone https://github.com/EricLBuehler/candle-vllm.git
```

**Tasks**:
1. Setup Candle with Metal backend
2. Implement basic model loading (GGUF)
3. Create simple inference API
4. Test with Mistral 7B quantized

### Phase 2: API Development (Week 2)
```bash
# Clone API reference implementations
git clone https://github.com/fcn94/llm_stream_endpoint.git
git clone https://github.com/shettysach/CandleMist.git
```

**Tasks**:
1. Implement Anthropic API format
2. Add streaming token generation
3. OpenAI compatibility layer
4. Error handling and validation

### Phase 3: Optimization & Production (Week 3-4)
```bash
# Clone optimization examples
git clone https://github.com/tbogdula/ai_notepad.git
git clone https://github.com/EricLBuehler/candle-lora.git
```

**Tasks**:
1. Memory optimization and profiling
2. Metal GPU kernel optimization
3. Production deployment setup
4. Performance tuning and benchmarking

## API Design for Claude Code Integration

### Anthropic-Compatible Endpoints

```rust
// Main message endpoint (Anthropic format)
POST /v1/messages
{
  "model": "pensieve-mistral-7b",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Your message here"
    }
  ]
}

// Streaming version
POST /v1/messages/stream
// Returns: Server-Sent Events (SSE)

// OpenAI compatibility (fallback)
POST /v1/chat/completions
{
  "model": "pensieve-mistral-7b",
  "messages": [...],
  "stream": true
}
```

### Local Configuration
```rust
// Server runs on localhost:8000
// API key: "pensieve-local-key"
// Model: "pensieve-mistral-7b"

// Claude Code configuration:
export ANTHROPIC_API_KEY="pensieve-local-key"
export ANTHROPIC_BASE_URL="http://localhost:8000"
```

## Model Selection Strategy

### Recommended Models for M1 16GB

#### Primary: Mistral 7B Instruct
- **Size**: ~4GB (4-bit GGUF)
- **Strengths**: Excellent reasoning, instruction following
- **M1 Performance**: Very good Metal optimization
- **Context**: 8k context window

#### Secondary: Phi-2 (Microsoft)
- **Size**: ~1.5GB (4-bit GGUF)
- **Strengths**: Surprisingly capable reasoning, very fast
- **M1 Performance**: Excellent, fits easily in RAM
- **Context**: 2k context window

#### Tertiary: LLaMA 7B Chat
- **Size**: ~4GB (4-bit GGUF)
- **Strengths**: Well-supported, good general performance
- **M1 Performance**: Good Metal support
- **Context**: 4k context window

## Implementation Benefits

### For Claude Code Users
1. **Privacy**: All processing stays local
2. **Speed**: No network latency after initial load
3. **Cost**: No API fees after setup
4. **Reliability**: Works offline, no rate limits
5. **Customization**: Can fine-tune for specific needs

### Technical Advantages
1. **Single Binary**: No Python dependencies
2. **Metal Optimization**: Native Apple Silicon performance
3. **Memory Efficient**: Quantized models for 16GB systems
4. **API Compatible**: Drop-in replacement for Claude API
5. **Extensible**: Rust ecosystem for future enhancements

## Next Actions

### Immediate (This Week)
1. **Clone repositories**: Execute clone script below
2. **Setup development**: Rust + Metal toolchain
3. **Model acquisition**: Download quantized Mistral 7B
4. **Prototype basic inference**: Test Candle + Metal setup

### Clone Script
```bash
#!/bin/bash
# Clone Candle-focused repositories for M1 LLM server

mkdir -p .refGitHubRepo
cd .refGitHubRepo

echo "Cloning core Candle repositories..."
git clone https://github.com/huggingface/candle.git
git clone https://github.com/GarthDB/metal-candle.git

echo "Cloning server implementation references..."
git clone https://github.com/EricLBuehler/candle-vllm.git
git clone https://github.com/fcn94/llm_stream_endpoint.git

echo "Cloning application examples..."
git clone https://github.com/shettysach/CandleMist.git
git clone https://github.com/tbogdala/ai_notepad.git
git clone https://github.com/EricLBuehler/candle-lora.git

echo "All Candle repositories cloned successfully!"
echo "Ready to build M1-optimized local LLM server."
```

### Success Criteria
1. ✅ Load Mistral 7B GGUF in <10 seconds
2. ✅ Generate first token in <500ms
3. ✅ Sustain 15+ tokens/second throughput
4. ✅ Serve Anthropic-compatible API
5. ✅ Work with Claude Code integration

## Conclusion

This Candle-only approach provides a clean, focused path to building a high-performance local LLM server specifically optimized for Apple M1 hardware. By leveraging the Metal backend and quantized models, we can achieve excellent performance while staying within the 16GB memory constraint.

The combination of official Candle libraries, production-ready server implementations, and application examples gives us a complete reference architecture for building a local server that seamlessly integrates with Claude Code through Anthropic-compatible APIs.

**Next Step**: Execute the clone script and begin Phase 1 core infrastructure development.