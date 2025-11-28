# Pensieve Local LLM Server

**Run Claude Code with a local LLM on Apple Silicon - no API fees, complete privacy.**

Pensieve is a local LLM inference server that provides **Anthropic API compatibility**, allowing you to use Claude Code (and other Anthropic-compatible tools) with a locally-running model.

---

## Quick Start (5 minutes)

### Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.9+**
- **Rust toolchain** (for the HTTP server)

### Step 1: Install Python Dependencies

```bash
cd pensieve-local-llm-server
pip3 install -r python_bridge/requirements.txt
```

### Step 2: Download the Phi-3 Model

We use **Phi-3-mini-128k-instruct** (4-bit quantized for memory efficiency):

```bash
# Create models directory
mkdir -p models

# Download using mlx-lm (automatically converts to MLX format)
python3 -m mlx_lm.convert \
  --hf-path microsoft/Phi-3-mini-128k-instruct \
  --mlx-path models/Phi-3-mini-128k-instruct-4bit \
  --quantize \
  --q-bits 4
```

**Alternative: Download pre-converted model from MLX Community**
```bash
# Faster - uses pre-quantized model
pip3 install huggingface_hub
huggingface-cli download mlx-community/Phi-3-mini-128k-instruct-4bit --local-dir models/Phi-3-mini-128k-instruct-4bit
```

### Step 3: Start the MLX Server

```bash
python3 python_bridge/mlx_server.py \
  --model-path models/Phi-3-mini-128k-instruct-4bit \
  --port 8765
```

You should see:
```
INFO:     Pensieve MLX Server starting...
INFO:     Loading model from models/Phi-3-mini-128k-instruct-4bit
INFO:     Model loaded successfully!
INFO:     Uvicorn running on http://127.0.0.1:8765
```

### Step 4: Test the Server

```bash
# Health check
curl http://127.0.0.1:8765/health

# Generate text
curl -X POST http://127.0.0.1:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

### Step 5: Use with Claude Code

```bash
# Option A: Use the wrapper script (recommended)
./scripts/claude-local

# Option B: Set environment variables manually
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
export ANTHROPIC_API_KEY=pensieve-local-token
claude
```

---

## Detailed Installation Guide

### 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Platform** | macOS with Apple Silicon | macOS 14+ |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 5GB free | 10GB free |
| **Python** | 3.9 | 3.11+ |
| **Rust** | 1.70+ | Latest stable |

### 2. Install Dependencies

#### Python Dependencies

```bash
# Core dependencies
pip3 install mlx>=0.0.10 mlx-lm>=0.0.12

# Server dependencies
pip3 install fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.0.0

# Memory monitoring
pip3 install psutil>=5.9.0

# Or install all at once:
pip3 install -r python_bridge/requirements.txt
```

#### Rust Toolchain (Optional - for full server)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
cargo --version
rustc --version
```

### 3. Download Models

#### Option A: Phi-3 Mini 128K (Recommended)

Best balance of capability and memory usage.

```bash
mkdir -p models

# Method 1: Convert from HuggingFace (slower, more control)
python3 -m mlx_lm.convert \
  --hf-path microsoft/Phi-3-mini-128k-instruct \
  --mlx-path models/Phi-3-mini-128k-instruct-4bit \
  --quantize \
  --q-bits 4

# Method 2: Download pre-converted (faster)
huggingface-cli download mlx-community/Phi-3-mini-128k-instruct-4bit \
  --local-dir models/Phi-3-mini-128k-instruct-4bit
```

**Memory Usage**: ~2.5GB loaded

#### Option B: Smaller Models (for testing)

```bash
# SmolLM 135M - Very fast, less capable
huggingface-cli download mlx-community/SmolLM-135M-Instruct-4bit \
  --local-dir models/SmolLM-135M-Instruct-4bit

# Qwen 0.5B - Good balance for testing
huggingface-cli download mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --local-dir models/Qwen2.5-0.5B-Instruct-4bit
```

### 4. Verify Installation

```bash
# Test MLX is working
python3 -c "import mlx; print('MLX version:', mlx.__version__)"

# Test model loading
python3 -c "
from mlx_lm import load, generate
model, tokenizer = load('models/Phi-3-mini-128k-instruct-4bit')
print('Model loaded successfully!')
print(generate(model, tokenizer, prompt='Hello', max_tokens=10))
"
```

---

## Running the Servers

### Python MLX Server (Required)

The MLX server handles actual model inference:

```bash
# Basic usage
python3 python_bridge/mlx_server.py \
  --model-path models/Phi-3-mini-128k-instruct-4bit

# With options
python3 python_bridge/mlx_server.py \
  --model-path models/Phi-3-mini-128k-instruct-4bit \
  --host 127.0.0.1 \
  --port 8765 \
  --max-concurrent 2
```

### Rust HTTP Server (Optional - for Anthropic API)

The Rust server provides full Anthropic API compatibility:

```bash
# Build and run
cargo run -p pensieve-01 -- start \
  --port 7777 \
  --model models/Phi-3-mini-128k-instruct-4bit
```

### Using Both Together

For full Anthropic API compatibility with Claude Code:

```bash
# Terminal 1: Start MLX server
python3 python_bridge/mlx_server.py \
  --model-path models/Phi-3-mini-128k-instruct-4bit \
  --port 8765

# Terminal 2: Start Anthropic proxy
cargo run -p pensieve-09 -- --port 7777

# Terminal 3: Use Claude Code
./scripts/claude-local
```

---

## API Reference

### MLX Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health and memory status |
| `/generate` | POST | Generate text completion |

#### POST /generate

```json
{
  "prompt": "Hello, how are you?",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

### Anthropic-Compatible Endpoints (Rust Server)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/v1/models` | GET | List available models |
| `/v1/messages` | POST | Create message (Anthropic format) |

---

## Configuration

### Environment Variables

```bash
# Server configuration
export PENSIEVE_PORT=7777
export PENSIEVE_HOST=127.0.0.1

# For Claude Code integration
export ANTHROPIC_BASE_URL=http://127.0.0.1:7777
export ANTHROPIC_API_KEY=pensieve-local-token

# Logging
export RUST_LOG=info
```

### Memory Thresholds

The server monitors system memory and protects against exhaustion:

| Status | Available RAM | Behavior |
|--------|---------------|----------|
| SAFE | >2GB | Normal operation |
| CAUTION | 1-2GB | Log warnings |
| WARNING | ~1GB | Throttle requests |
| CRITICAL | 0.5-1GB | Reject new requests |
| EMERGENCY | <500MB | Graceful shutdown |

---

## Troubleshooting

### "MLX not found"

```bash
# Ensure you're on Apple Silicon
uname -m  # Should show "arm64"

# Install MLX
pip3 install mlx mlx-lm
```

### "Model not found"

```bash
# Verify model exists
ls -la models/Phi-3-mini-128k-instruct-4bit/

# Should contain: config.json, model.safetensors, tokenizer files
```

### "Out of memory"

```bash
# Use a smaller model
python3 python_bridge/mlx_server.py \
  --model-path models/SmolLM-135M-Instruct-4bit

# Or reduce concurrent requests
python3 python_bridge/mlx_server.py \
  --model-path models/Phi-3-mini-128k-instruct-4bit \
  --max-concurrent 1
```

### "Server not responding"

```bash
# Check if server is running
curl http://127.0.0.1:8765/health

# Check port availability
lsof -i :8765
```

---

## Project Structure

```
pensieve-local-llm-server/
├── pensieve-01/          # CLI interface
├── pensieve-02/          # HTTP API server
├── pensieve-03/          # API models (Anthropic format)
├── pensieve-04/          # Inference engine
├── pensieve-05/          # Model loading (GGUF/SafeTensors)
├── pensieve-06/          # Metal GPU support
├── pensieve-07/          # Core traits and errors
├── pensieve-08/          # Claude Core integration
├── pensieve-09/          # Anthropic proxy
├── python_bridge/        # MLX inference server
│   ├── mlx_server.py     # FastAPI persistent server
│   ├── mlx_inference.py  # MLX model loading
│   └── requirements.txt  # Python dependencies
├── scripts/              # Helper scripts
│   └── claude-local      # Terminal-isolated Claude wrapper
├── models/               # Downloaded models (gitignored)
└── README.md             # This file
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 25-40 TPS | Tokens per second |
| Memory | <4GB | With Phi-3 4-bit |
| First Token | <500ms | Time to first token |
| Concurrent | 2 requests | Safe default |

---

## Contributing

See `.domainDocs/` for detailed research and architecture documentation.

## License

MIT OR Apache-2.0
