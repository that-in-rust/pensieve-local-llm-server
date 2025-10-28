# User Journey for Pensieve Local LLM Server

## Architecture Decision: MLX-Powered

**Framework**: Apple MLX (optimized for Apple Silicon)
**Default Model**: mlx-community/Phi-3-mini-128k-instruct-4bit
**Target Platform**: macOS with Apple Silicon (M1/M2/M3)

## Simplified User Journey

### One-Command Setup Experience

1. **User runs the following command**:
    ```bash
    cargo run -p pensieve-01 -- start
    ```
    - Pensieve automatically downloads and configures Phi-3-mini-128k-instruct-4bit model
    - Server starts on default port 8080 with MLX acceleration
    - Zero manual configuration required

2. **User configures Claude Code**:
    ```bash
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
    export ANTHROPIC_API_KEY=pensieve-local-key
    ```
    - Opens Claude Code with local LLM backend
    - Experience identical to cloud-based Claude
    - Full 128k context window support
    - Fast generation speeds on Apple Silicon

### Advanced Configuration (Optional)

For users who want custom models or configurations:
```bash
cargo run -p pensieve-01 -- start --model <custom-model> --port <custom-port>
```

## Key Benefits

- **Instant Setup**: No model downloading or manual configuration
- **Optimized Performance**: MLX provides superior Apple Silicon acceleration
- **Memory Efficient**: 4-bit quantization allows running on 16GB+ Macs
- **Transparent Integration**: Claude Code works unchanged
- **Local Privacy**: All processing happens locally on device

## Technical Implementation

- **MLX Framework**: Apple's machine learning framework for Silicon
- **Automatic Model Management**: Built-in Hugging Face integration
- **Memory Optimization**: Smart KV cache management for large context
- **Metal Acceleration**: Full GPU utilization for fast inference
- **Rust + Python/C++**: Mixed architecture for optimal performance