# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pensieve is a local LLM server for Apple Silicon that provides Anthropic API compatibility. It uses MLX framework for Metal GPU acceleration and exposes an HTTP API compatible with the Anthropic Messages API format.

**Key Feature**: Run LLM inference locally on your Mac with Metal GPU acceleration for privacy and zero API costs.

## Build Commands

```bash
# Build entire workspace (produces warnings, compiles successfully)
cargo build --workspace

# Build release version
cargo build --release --workspace

# Run tests (note: some test compilation issues exist, prefer manual testing)
cargo test --workspace

# Build specific crate
cargo build -p pensieve-01
```

## Running the Server

```bash
# Start server with Phi-3 model (standard method)
./target/debug/pensieve start --model ./models/Phi-3-mini-128k-instruct-4bit/model.safetensors

# Start on custom port
./target/debug/pensieve start --host 0.0.0.0 --port 8080 --model ./models/Phi-3-mini-128k-instruct-4bit/model.safetensors

# Using cargo run
cargo run -p pensieve-01 -- start --model ./models/Phi-3-mini-128k-instruct-4bit/model.safetensors

# Stop server
pkill -f pensieve

# Check server health
curl http://127.0.0.1:7777/health
```

## Testing

```bash
# Test Python MLX bridge directly (RECOMMENDED)
python3 python_bridge/mlx_inference.py \
  --model-path ./models/Phi-3-mini-128k-instruct-4bit \
  --prompt "Test prompt" \
  --max-tokens 20

# Performance test with metrics
python3 python_bridge/mlx_inference.py \
  --model-path ./models/Phi-3-mini-128k-instruct-4bit \
  --prompt "Performance test" \
  --max-tokens 50 \
  --metrics

# Streaming test
python3 python_bridge/mlx_inference.py \
  --model-path ./models/Phi-3-mini-128k-instruct-4bit \
  --prompt "Stream test" \
  --max-tokens 30 \
  --stream

# Test HTTP API (authentication optional for local development)
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 50, "messages": [{"role":"user","content":[{"type":"text","text":"Hello!"}]}]}'

# Health check
curl http://127.0.0.1:7777/health
```

## Architecture

Pensieve uses a **layered 8-crate architecture** with strict dependency hierarchy:

### Layer 1 (L1) - Core Foundation
- **pensieve-07_core**: Foundation traits, error types, no_std compatible
  - No external dependencies
  - Provides CoreError, CoreResult types
  - Base traits for all other crates

### Layer 2 (L2) - Engine Layer
- **pensieve-04_engine**: Inference engine (currently Candle-based, MLX planned)
  - Depends on: pensieve-07_core
  - Provides inference abstractions

- **pensieve-05_models**: Data models and GGUF interfaces
  - Depends on: pensieve-07_core
  - Model loading, safetensors support

- **pensieve-06_metal**: Metal GPU implementations (macOS only)
  - Depends on: pensieve-07_core, pensieve-04_engine
  - GPU acceleration via Metal framework

### Layer 3 (L3) - Application Layer
- **pensieve-01**: CLI interface (binary: `pensieve`)
  - Depends on: all L1, L2, and other L3 crates
  - Entry point: `src/main.rs`

- **pensieve-02**: HTTP API server with streaming
  - Depends on: pensieve-07_core, L2 crates, pensieve-03
  - Warp-based server, SSE streaming

- **pensieve-03**: API models and serialization
  - Depends on: pensieve-07_core, pensieve-05_models
  - Anthropic API compatibility types

### External Layer - Python Bridge
- **python_bridge/mlx_inference.py**: Real MLX implementation
  - Uses MLX framework for Apple Silicon
  - Performance: ~16.85 TPS (target: 25+ TPS)
  - Model caching, metrics tracking

### Dependency Rules
1. L1 crates have zero external dependencies (except core/alloc)
2. L2 crates depend only on L1 crates + framework libs (Candle, Metal)
3. L3 crates can depend on any L1/L2 crates and external libs
4. Never create circular dependencies between layers
5. Keep pensieve-07_core as minimal foundation

## Current Implementation Status

### Working Features
- ✅ MLX inference via Python bridge (16.85 TPS)
- ✅ Phi-3 Mini 4-bit model support
- ✅ HTTP API server with Anthropic compatibility
- ✅ CLI interface for server management
- ✅ Metal GPU acceleration
- ✅ Project builds successfully (warnings only)
- ✅ Optional authentication for local development

### Known Limitations
- ⚠️ Performance: 16.85 TPS (below 25+ TPS target)
- ⚠️ Rust crates still use Candle (not fully migrated to MLX)
- ⚠️ Mixed Candle/MLX architecture (Python bridge is MLX, Rust is Candle)
- ⚠️ Some test compilation issues (manual testing works)

## Model Information

**Current Model**: `mlx-community/Phi-3-mini-128k-instruct-4bit`
- Format: MLX-compatible safetensors
- Size: ~2.1GB
- Context Length: 128k tokens
- Quantization: 4-bit
- Location: `models/Phi-3-mini-128k-instruct-4bit/`

Required files:
```
models/Phi-3-mini-128k-instruct-4bit/
├── config.json
├── model.safetensors        # MUST pass this file path to CLI
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**Important**: When starting the server, pass the full path to `model.safetensors`, not the directory.

## API Compatibility

Pensieve implements the **Anthropic Messages API v1**:
- Endpoint: `POST /v1/messages`
- Streaming: Server-Sent Events (SSE) when `stream: true`
- Authentication: Optional for local development (accepts requests with or without Bearer tokens)
- Health: `GET /health`

### Example Usage

```bash
# Basic request (no auth required for local dev)
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "messages": [{"role":"user","content":[{"type":"text","text":"Hello!"}]}]
  }'

# With authentication (optional)
curl -X POST http://127.0.0.1:7777/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key-12345" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "messages": [{"role":"user","content":[{"type":"text","text":"Hello!"}]}]
  }'
```

See `docs/API.md` for complete API documentation.

## Development Workflow

1. **Make changes** to Rust code
2. **Build**: `cargo build --workspace`
3. **Test manually** using Python bridge or curl commands
4. **Check performance**: Run with `--metrics` flag
5. **Update documentation** with verified results

## Performance Optimization Notes

- Current bottleneck: MLX/Rust bridge performance
- Target: Migrate Rust crates from Candle to native MLX bindings
- Phi-3 4-bit quantization provides best speed/quality tradeoff for M1/M2/M3 Macs
- Memory usage peaks at ~2.2GB (well within 16GB constraints)

## Common Issues

**"Address already in use"**
```bash
pensieve-server stop
# or
pkill -f pensieve
```

**Model loading fails**
- Verify model path points to `model.safetensors` file, not directory
- Check all required model files exist
- Ensure MLX and mlx-lm are installed: `pip install mlx mlx-lm`

**Slow inference**
- Verify Metal GPU is active (check device in logs)
- Consider reducing max_tokens
- Current performance: ~16.85 TPS is expected with current implementation

## Project Goals

1. Provide privacy-first local LLM inference on Apple Silicon
2. Maintain full Anthropic API compatibility for easy integration
3. Optimize for Apple Silicon (M1/M2/M3) with Metal GPU acceleration
4. Enable local AI development without cloud dependencies
5. Achieve 25+ TPS performance with MLX optimization
