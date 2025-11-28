# D25: Memory Verification Test
**Date**: November 28, 2025
**Status**: Verified

## Test Context
- **Hardware**: Apple Silicon (M1/M2/M3/M4)
- **Model**: `Phi-3-mini-128k-instruct-4bit`
- **Method**: `ps -o rss` against Python process running `mlx_server.py`

## Results
| Metric | Value | Notes |
|--------|-------|-------|
| **RSS Memory** | **2.45 GB** | Total physical memory used |
| **Model Size (Disk)** | 2.0 GB | `model.safetensors` |
| **Runtime Overhead** | ~0.45 GB | Python + MLX + HTTP Server |

## Analysis
The memory usage is highly optimized and aligns with theoretical expectations for 4-bit quantization of a 3.8B parameter model.

- **Theoretical Min**: 3.8B params * 0.5 bytes = 1.9 GB
- **Observed**: 2.45 GB
- **Efficiency**: The server adds only ~450MB overhead for the entire HTTP stack and inference engine.

## Conclusion
The "Persistent Server" architecture successfully mitigates the 8GB memory spike observed in previous process-per-request architectures. The server remains stable at ~2.5GB baseline.
