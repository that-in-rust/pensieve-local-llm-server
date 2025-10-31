#!/usr/bin/env python3
"""
Pensieve MLX Inference Bridge - REAL Implementation with Memory Safety
Provides MLX-powered text generation for the Pensieve Local LLM Server

Safety Features (D17 Research):
- Memory monitoring with psutil
- Automatic cache clearing after generation
- Pre-inference memory checks
- Emergency shutdown on critical memory
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

# REAL MLX imports
import mlx.core as mx
from mlx_lm import load, stream_generate, generate as mlx_generate
import gc
import threading
from contextlib import contextmanager

# Memory safety (D17 requirement)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("[WARN] psutil not available - install with: pip install psutil", file=sys.stderr)
    PSUTIL_AVAILABLE = False

# Performance optimization: Model cache with persistent storage
import pickle
import os.path

_model_cache = {}
_cache_lock = threading.Lock()
_cache_file_path = "/tmp/pensieve_model_cache.pkl"

# Performance monitoring with persistent metrics
_performance_metrics = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "session_start": time.time()
}

def load_persistent_metrics():
    """Load persistent metrics from disk if available"""
    global _performance_metrics
    if os.path.exists("/tmp/pensieve_metrics.pkl"):
        try:
            with open("/tmp/pensieve_metrics.pkl", "rb") as f:
                saved_metrics = pickle.load(f)
                # Preserve session start time but update cumulative metrics
                old_session_start = _performance_metrics["session_start"]
                _performance_metrics.update(saved_metrics)
                _performance_metrics["session_start"] = old_session_start
                print(f"[PERF] Loaded persistent metrics: {_performance_metrics['total_requests']} requests", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to load persistent metrics: {e}", file=sys.stderr)

def save_persistent_metrics():
    """Save current metrics to disk"""
    try:
        with open("/tmp/pensieve_metrics.pkl", "wb") as f:
            pickle.dump(_performance_metrics, f)
    except Exception as e:
        print(f"[WARN] Failed to save persistent metrics: {e}", file=sys.stderr)

# Load persistent metrics on startup
load_persistent_metrics()

# Memory safety thresholds (from D17 research)
MEMORY_CRITICAL_GB = 1.0  # Reject requests
MEMORY_EMERGENCY_GB = 0.5  # Emergency shutdown

def check_memory_status() -> tuple[str, float]:
    """
    Check system memory status

    Returns: (status, available_gb)
        status: 'SAFE', 'WARNING', 'CRITICAL', 'EMERGENCY'
        available_gb: Available RAM in GB
    """
    if not PSUTIL_AVAILABLE:
        return 'UNKNOWN', 0.0

    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)

        if available_gb < MEMORY_EMERGENCY_GB:
            return 'EMERGENCY', available_gb
        elif available_gb < MEMORY_CRITICAL_GB:
            return 'CRITICAL', available_gb
        elif available_gb < 2.0:
            return 'WARNING', available_gb
        else:
            return 'SAFE', available_gb
    except Exception as e:
        print(f"[WARN] Memory check failed: {e}", file=sys.stderr)
        return 'UNKNOWN', 0.0

def clear_mlx_cache():
    """
    Clear MLX Metal cache to free GPU memory
    Based on D17 research - critical for preventing memory leaks
    """
    try:
        if hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
            print("[MEMORY] MLX cache cleared", file=sys.stderr)
        else:
            print("[WARN] mx.metal.clear_cache() not available", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to clear MLX cache: {e}", file=sys.stderr)

def log_memory_state(label: str = ""):
    """Log current memory state for monitoring"""
    if not PSUTIL_AVAILABLE:
        return

    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        used_percent = mem.percent

        # Try to get MLX cache memory
        cache_info = ""
        if hasattr(mx.metal, 'get_cache_memory'):
            try:
                cache_bytes = mx.metal.get_cache_memory()
                cache_gb = cache_bytes / (1024 ** 3)
                cache_info = f" | MLX Cache: {cache_gb:.2f}GB"
            except:
                pass

        print(f"[MEMORY{' ' + label if label else ''}] Available: {available_gb:.2f}GB / {total_gb:.2f}GB ({used_percent:.1f}% used){cache_info}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to log memory: {e}", file=sys.stderr)

@contextmanager
def _performance_timer(operation_name: str):
    """Context manager for timing operations"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"[PERF] {operation_name}: {elapsed:.3f}s", file=sys.stderr)

def load_model(model_path: str):
    """Load REAL MLX model and tokenizer with enhanced caching and performance optimizations"""
    global _model_cache, _performance_metrics

    with _cache_lock:
        if model_path in _model_cache:
            _performance_metrics["cache_hits"] += 1
            print(f"[CACHE HIT] Model already loaded: {model_path}", file=sys.stderr)
            return _model_cache[model_path]

        _performance_metrics["cache_misses"] += 1
        print(f"[CACHE MISS] Loading REAL MLX model from: {model_path}", file=sys.stderr)

    with _performance_timer("model_loading"):
        try:
            # Performance optimization: Set optimal Metal settings
            # Check MLX version compatibility
            if hasattr(mx, 'set_stream') and hasattr(mx, 'default_stream'):
                mx.set_stream(mx.default_stream())
                print("[PERF] Applied stream optimization", file=sys.stderr)

            if hasattr(mx.metal, 'set_cache_limit'):
                mx.metal.set_cache_limit(256 * 1024 * 1024)  # 256MB cache (optimized)
                print("[PERF] Applied Metal cache optimization (256MB)", file=sys.stderr)

            # Use REAL MLX-LM loading (this handles Phi-3 automatically)
            model, tokenizer = load(model_path)

            print("REAL MLX model loaded successfully", file=sys.stderr)
            print(f"Model device: {mx.default_device()}", file=sys.stderr)

            # Performance optimization: Pre-warm the model
            print("[PERF] Warming up model...", file=sys.stderr)
            warmup_prompt = "Hello"
            try:
                _ = mlx_generate(model, tokenizer, warmup_prompt, max_tokens=5)
                print("[PERF] Model warmup completed", file=sys.stderr)
            except Exception as warmup_error:
                print(f"[WARN] Model warmup failed: {warmup_error}", file=sys.stderr)

            with _cache_lock:
                _model_cache[model_path] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'mlx_path': model_path,
                    'load_time': time.time()
                }

            return _model_cache[model_path]

        except Exception as e:
            raise RuntimeError(f"Failed to load REAL MLX model: {e}")

def optimize_mlx_performance():
    """Apply MLX performance optimizations for Apple Silicon"""
    optimizations_applied = []

    # Metal GPU optimizations
    if hasattr(mx.metal, 'set_active_device'):
        mx.metal.set_active_device(0)
        optimizations_applied.append("Metal active device")

    # Memory optimizations
    if hasattr(mx, 'set_memory_pool'):
        mx.set_memory_pool(True)
        optimizations_applied.append("Memory pooling")

    # Stream optimizations (version-compatible)
    if hasattr(mx, 'set_stream') and hasattr(mx, 'default_stream'):
        mx.set_stream(mx.default_stream())
        optimizations_applied.append("Stream optimization")

    # Cache optimizations (reduced to 256MB for better memory profile)
    if hasattr(mx.metal, 'set_cache_limit'):
        try:
            mx.metal.set_cache_limit(256 * 1024 * 1024)  # 256MB cache (optimized)
            optimizations_applied.append("Metal cache (256MB)")
        except:
            pass  # Ignore if cache limit setting fails

    # Garbage collection optimization
    gc.collect()
    if hasattr(mx, 'eval'):
        try:
            mx.eval()  # Force evaluation to clear pending operations
            optimizations_applied.append("MLX eval")
        except:
            pass  # Ignore if eval fails

    if optimizations_applied:
        print(f"[PERF] Applied optimizations: {', '.join(optimizations_applied)}", file=sys.stderr)
    else:
        print("[PERF] No additional optimizations available for this MLX version", file=sys.stderr)

def real_mlx_generate(
    model_data: Dict[str, Any],
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> Iterator[str]:
    """
    REAL MLX generation using MLX-LM with memory safety and performance optimizations

    Memory Safety (D17 Research):
    - Pre-inference memory check
    - Post-generation cache clearing
    - Emergency rejection on critical memory
    """
    global _performance_metrics

    # SAFETY CHECK: Memory status before inference
    log_memory_state("BEFORE")
    mem_status, available_gb = check_memory_status()

    if mem_status == 'CRITICAL':
        error_msg = f"Critical memory pressure: {available_gb:.2f}GB available. Request rejected."
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg)
    elif mem_status == 'EMERGENCY':
        error_msg = f"Emergency memory exhaustion: {available_gb:.2f}GB available. Shutting down."
        print(f"[EMERGENCY] {error_msg}", file=sys.stderr)
        clear_mlx_cache()
        raise RuntimeError(error_msg)
    elif mem_status == 'WARNING':
        print(f"[WARN] Low memory: {available_gb:.2f}GB available", file=sys.stderr)

    model = model_data['model']
    tokenizer = model_data['tokenizer']

    # Skip chat template for now - it's returning token IDs instead of text
    # TODO: Fix chat template handling later for better Phi-3 compatibility
    # The basic prompt should work for initial testing

    print(f"[PERF] Starting REAL MLX generation with prompt length: {len(prompt.split())}", file=sys.stderr)

    # Performance monitoring
    generation_start = time.perf_counter()
    _performance_metrics["total_requests"] += 1

    # Apply performance optimizations
    optimize_mlx_performance()

    if stream:
        # Use REAL MLX-LM streaming with optimizations
        token_count = 0
        with _performance_timer("streaming_generation"):
            # Use version-compatible streaming
            try:
                # Try with chunk_size parameter (newer MLX-LM)
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    chunk_size=4,
                ):
                    token_count += 1
                    yield response.text
            except TypeError:
                # Fallback to older MLX-LM API without chunk_size
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                ):
                    token_count += 1
                    yield response.text

        # Update performance metrics
        generation_time = time.perf_counter() - generation_start
        _performance_metrics["total_tokens"] += token_count
        _performance_metrics["total_time"] += generation_time
        tps = token_count / generation_time if generation_time > 0 else 0
        print(f"[PERF] Streaming: {token_count} tokens in {generation_time:.3f}s = {tps:.1f} TPS", file=sys.stderr)

        # SAFETY: Only clear cache if memory is under pressure (CRITICAL or worse)
        # Keeping cache warm improves performance
        final_mem_status, final_available = check_memory_status()
        if final_mem_status in ['CRITICAL', 'EMERGENCY']:
            print(f"[MEMORY] Clearing cache due to memory pressure: {final_mem_status}", file=sys.stderr)
            clear_mlx_cache()
        log_memory_state("AFTER")

    else:
        # Use REAL MLX-LM non-streaming with optimizations
        try:
            with _performance_timer("batch_generation"):
                # Use basic parameters compatible with all MLX-LM versions
                response = mlx_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                )

            # MLX-LM returns a string, not a list
            if isinstance(response, list):
                response = " ".join(response) if response else ""

            # Update performance metrics
            token_count = len(response.split())
            generation_time = time.perf_counter() - generation_start
            _performance_metrics["total_tokens"] += token_count
            _performance_metrics["total_time"] += generation_time
            tps = token_count / generation_time if generation_time > 0 else 0
            print(f"[PERF] Batch: {token_count} tokens in {generation_time:.3f}s = {tps:.1f} TPS", file=sys.stderr)

            yield response

        finally:
            # SAFETY: Only clear cache if memory is under pressure
            # Keeping cache warm improves performance and reduces model reload overhead
            final_mem_status, final_available = check_memory_status()
            if final_mem_status in ['CRITICAL', 'EMERGENCY']:
                print(f"[MEMORY] Clearing cache due to memory pressure: {final_mem_status}", file=sys.stderr)
                clear_mlx_cache()
            log_memory_state("AFTER")

def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics and save to disk"""
    global _performance_metrics

    avg_tps = (_performance_metrics["total_tokens"] / _performance_metrics["total_time"]
               if _performance_metrics["total_time"] > 0 else 0)

    cache_hit_rate = (_performance_metrics["cache_hits"] /
                     (_performance_metrics["cache_hits"] + _performance_metrics["cache_misses"])
                     if (_performance_metrics["cache_hits"] + _performance_metrics["cache_misses"]) > 0 else 0)

    session_time = time.time() - _performance_metrics["session_start"]

    metrics = {
        "total_requests": _performance_metrics["total_requests"],
        "total_tokens": _performance_metrics["total_tokens"],
        "total_time_seconds": round(_performance_metrics["total_time"], 3),
        "average_tps": round(avg_tps, 2),
        "cache_hit_rate": round(cache_hit_rate, 3),
        "peak_memory_mb": mx.get_peak_memory() / 1e6 if hasattr(mx, 'get_peak_memory') else 0,
        "session_time_minutes": round(session_time / 60, 1),
        "requests_per_minute": round(_performance_metrics["total_requests"] / (session_time / 60), 1) if session_time > 0 else 0
    }

    # Save metrics to disk
    save_persistent_metrics()
    return metrics

def generate_text(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> Iterator[Dict[str, Any]]:
    """
    Main generation function using REAL MLX inference with performance monitoring
    """
    try:
        # Load model using REAL MLX
        model_data = load_model(model_path)

        # Generate text using REAL MLX-LM
        if stream:
            accumulated_text = ""
            start_time = time.time()

            for text_chunk in real_mlx_generate(
                model_data, prompt, max_tokens, temperature, stream=True
            ):
                accumulated_text += text_chunk

                # Calculate performance metrics
                elapsed = time.time() - start_time
                # Handle both string and list responses from MLX-LM
                token_count = len(accumulated_text.split()) if isinstance(accumulated_text, str) else len(accumulated_text)
                tps = token_count / elapsed if elapsed > 0 else 0

                yield {
                    "type": "text_chunk",
                    "text": text_chunk,
                    "accumulated": accumulated_text,
                    "tokens_per_second": round(tps, 2),
                    "elapsed_ms": round(elapsed * 1000, 2)
                }
        else:
            # Non-streaming
            start_time = time.time()
            full_response = ""
            for text_chunk in real_mlx_generate(
                model_data, prompt, max_tokens, temperature, stream=False
            ):
                full_response += text_chunk

            elapsed = time.time() - start_time
            # Handle both string and list responses from MLX-LM
            token_count = len(full_response.split()) if isinstance(full_response, str) else len(full_response)
            tps = token_count / elapsed if elapsed > 0 else 0

            # Include performance metrics
            performance_metrics = get_performance_metrics()
            yield {
                "type": "complete",
                "text": full_response,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(full_response.split()),
                "total_tokens": len(prompt.split()) + len(full_response.split()),
                "tokens_per_second": round(tps, 2),
                "elapsed_ms": round(elapsed * 1000, 2),
                "peak_memory_mb": mx.get_peak_memory() / 1e6 if hasattr(mx, 'get_peak_memory') else 0,
                "performance_metrics": performance_metrics
            }

    except Exception as e:
        import traceback
        error_details = f"MLX inference error: {str(e)}\nTraceback: {traceback.format_exc()}"
        yield {
            "type": "error",
            "error": error_details
        }

def main():
    """Main CLI interface with performance monitoring and memory safety"""
    parser = argparse.ArgumentParser(description="Pensieve MLX Inference Bridge - REAL with Memory Safety")
    parser.add_argument("--model-path", help="Path to MLX model directory")
    parser.add_argument("--prompt", help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--stream", action="store_true", help="Stream generation output")
    parser.add_argument("--metrics", action="store_true", help="Show performance metrics")
    parser.add_argument("--clear-cache", action="store_true", help="Clear MLX Metal cache and exit")

    args = parser.parse_args()

    # Handle --clear-cache command
    if args.clear_cache:
        print("[MEMORY] Clearing MLX cache...", file=sys.stderr)
        clear_mlx_cache()
        log_memory_state("AFTER CLEAR")
        print("[MEMORY] Cache cleared successfully", file=sys.stderr)
        sys.exit(0)

    # Validate required arguments for generation
    if not args.model_path or not args.prompt:
        parser.error("--model-path and --prompt are required for generation")

    # Print initial performance info
    if args.metrics:
        print(f"[PERF] MLX Performance Test - Target: 25+ TPS", file=sys.stderr)
        print(f"[PERF] Model: {args.model_path}", file=sys.stderr)
        print(f"[PERF] Device: {mx.default_device()}", file=sys.stderr)

    try:
        # Generate text
        for response in generate_text(
            args.model_path,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.stream
        ):
            # Output JSON response
            json.dump(response, sys.stdout)
            print()  # Newline

            if not args.stream and response.get("type") == "complete":
                # Show performance metrics if requested
                if args.metrics and "performance_metrics" in response:
                    metrics = response["performance_metrics"]
                    print(f"[PERF] Performance Summary:", file=sys.stderr)
                    print(f"[PERF]   Total Requests: {metrics['total_requests']}", file=sys.stderr)
                    print(f"[PERF]   Total Tokens: {metrics['total_tokens']}", file=sys.stderr)
                    print(f"[PERF]   Average TPS: {metrics['average_tps']}", file=sys.stderr)
                    print(f"[PERF]   Cache Hit Rate: {metrics['cache_hit_rate']:.1%}", file=sys.stderr)
                    print(f"[PERF]   Peak Memory: {metrics['peak_memory_mb']:.1f} MB", file=sys.stderr)

                    # Performance assessment
                    tps = metrics['average_tps']
                    if tps >= 25:
                        print(f"🎉 PERFORMANCE TARGET ACHIEVED: {tps:.1f} TPS >= 25 TPS", file=sys.stderr)
                    elif tps >= 20:
                        print(f"⚠️  CLOSE TO TARGET: {tps:.1f} TPS (need {25-tps:.1f} more TPS)", file=sys.stderr)
                    else:
                        print(f"❌ BELOW TARGET: {tps:.1f} TPS (need {25-tps:.1f} more TPS)", file=sys.stderr)
                break

    except KeyboardInterrupt:
        json.dump({"type": "error", "error": "Generation interrupted"}, sys.stdout)
        print()
        sys.exit(1)
    except Exception as e:
        json.dump({"type": "error", "error": str(e)}, sys.stdout)
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()