# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

# Project Overview
Pensieve is a local LLM inference server designed for **macOS with Apple Silicon**. It provides **Anthropic API compatibility**, allowing tools like Claude Code to run against a local model (specifically Phi-3) without API fees or data privacy concerns.

The project consists of:
- A **Rust workspace** (9 crates) implementing the core server, API handling, and application logic.
- A **Python bridge** (`python_bridge/`) using Apple's **MLX** framework for efficient model inference.
- **Scripts** (`scripts/`) for orchestration and terminal isolation.

# Architecture
The system follows a strict 3-layer architecture (L1 -> L2 -> L3):

## Layer 3: Application (User Facing)
- **pensieve-01 (CLI)**: Command-line interface and process management.
- **pensieve-02 (HTTP)**: Core HTTP server (Warp framework) and SSE streaming.
- **pensieve-09 (Proxy)**: Anthropic-compatible proxy handling auth and translation.

## Layer 2: Domain (Business Logic)
- **pensieve-04 (Engine)**: Inference engine abstractions.
- **pensieve-05 (Models)**: GGUF/SafeTensors parsing and tensor operations.
- **pensieve-06 (Metal)**: GPU acceleration and buffer management.
- **pensieve-08 (Claude Core)**: Claude-specific domain logic.

## Layer 1: Core (Foundation)
- **pensieve-07 (Core)**: `no_std` traits, error types, and primitives used by all layers.

## Python Bridge
The actual inference runs in a persistent Python process (`mlx_server.py`) to leverage MLX, communicating with the Rust server (or exposed directly).

# Development & Usage

## Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- Rust toolchain

## Common Commands

### Server Management
Start the MLX inference server (Python):
```bash
# Start with default settings
./scripts/start-mlx-server.sh

# Manual start
python3 python_bridge/mlx_server.py --model-path models/Phi-3-mini-128k-instruct-4bit --port 8765
```

Start the Rust Anthropic Proxy (Optional for direct MLX usage, required for full API compat):
```bash
cargo run -p pensieve-09 -- --port 7777
```

### Client Integration
Run Claude Code isolated to this local server:
```bash
./scripts/claude-local [arguments]
```

### Testing
Check server health:
```bash
curl http://127.0.0.1:8765/health
```

Run Python tests:
```bash
python3 python_bridge/test_mlx_inference.py
```

## Key Directories
- `python_bridge/`: MLX inference logic and Python server.
- `models/`: Storage for downloaded LLM weights (gitignored).
- `scripts/`: Helper scripts for setup and running.
- `.domainDocs/`: Detailed architectural documentation.

## Performance Verification (Nov 2025)
- **Model**: Phi-3-mini-128k-instruct-4bit
- **Memory**: ~2.45 GB RSS (Verified)
- **Start Time**: <5 seconds (Warm)

## Model Setup
Models must be downloaded to `models/` before running:
```bash
# Download Phi-3-mini-128k-instruct-4bit
huggingface-cli download mlx-community/Phi-3-mini-128k-instruct-4bit --local-dir models/Phi-3-mini-128k-instruct-4bit
```
