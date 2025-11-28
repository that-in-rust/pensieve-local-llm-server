#!/usr/bin/env python3
"""
Pensieve MLX Server - FastAPI HTTP Server with Persistent Model

This server solves the 8GB memory spike problem by:
1. Loading the MLX model ONCE on startup
2. Keeping it resident in memory
3. Reusing the same model for all requests
4. Supporting concurrent requests via async handlers

Memory Profile:
- Baseline: ~2.5-4GB (one loaded Phi-3 4-bit model)
- Per request: +0.5-1GB (activation memory only)
- Peak with 4 concurrent: <5GB (shared model, separate contexts)

Previous architecture (process-per-request):
- Each request spawned new Python process
- Each process loaded 2GB model
- 4 concurrent = 4 × 2GB = 8GB spike ❌

New architecture (persistent server):
- Model loaded once at startup
- All requests share the same model weights
- 4 concurrent = 2.5GB baseline + 2GB activations = 4.5GB ✅
"""

import sys
import json
import time
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our existing MLX inference code
from inference import (
    load_model,
    real_mlx_generate,
    check_memory_status,
    log_memory_state,
    clear_mlx_cache,
    get_performance_metrics,
    optimize_mlx_performance,
)

# Global state: Model loaded once at startup
_global_model = None
_model_path = None
_startup_time = None

# Concurrency control: Limit concurrent inference requests
_inference_semaphore = None
MAX_CONCURRENT_INFERENCES = 2  # Safe default for Apple Silicon

# Metal cache configuration (optimized for memory efficiency)
MLX_METAL_CACHE_MB = 256  # Reduced from 1GB to 256MB for better memory profile

import uuid
from typing import Dict, Any, Optional, List, Union

class GenerateRequest(BaseModel):
    """Request model for /generate endpoint"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    stream: bool = Field(default=False, description="Enable streaming response")

class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    system: Optional[str] = None

def convert_to_phi3_prompt(req: AnthropicRequest) -> str:
    """Convert Anthropic messages to Phi-3 prompt format"""
    prompt = ""
    
    # Handle system prompt if present
    if req.system:
        prompt += f"<|system|>\n{req.system}<|end|>\n"
        
    for msg in req.messages:
        content = msg.content
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
            content = "".join(text_parts)
            
        prompt += f"<|{msg.role}|>\n{content}<|end|>\n"
        
    prompt += "<|assistant|>\n"
    return prompt

async def stream_anthropic_generator(
    model_data: Dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float,
    msg_id: str
):
    """Yields SSE events in Anthropic format"""
    loop = asyncio.get_event_loop()
    chunk_queue = asyncio.Queue()

    def sync_stream():
        try:
            for chunk in real_mlx_generate(
                model_data, prompt, max_tokens, temperature, stream=True
            ):
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put({"type": "chunk", "text": chunk}), loop
                )
            asyncio.run_coroutine_threadsafe(chunk_queue.put({"type": "done"}), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put({"type": "error", "error": str(e)}), loop
            )

    loop.run_in_executor(None, sync_stream)

    # 1. message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'phi-3', 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    
    # 2. content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    while True:
        chunk = await chunk_queue.get()
        
        if chunk["type"] == "done":
            break
        elif chunk["type"] == "error":
            # Log error but finish stream gracefully to avoid client crash
            print(f"[ERROR] Streaming failed: {chunk['error']}", file=sys.stderr)
            break
        else:
            text = chunk["text"]
            if text:
                # 3. content_block_delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

    # 4. content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # 5. message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'output_tokens': 0}}})}\n\n"

    # 6. message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    model_loaded: bool
    model_path: Optional[str]
    uptime_seconds: float
    memory_status: str
    memory_available_gb: float
    concurrent_limit: int
    performance_metrics: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Load model on startup, cleanup on shutdown

    This is where we solve the memory problem:
    - Model loads ONCE here
    - Stays resident for the lifetime of the server
    - All requests reuse the same model
    """
    global _global_model, _startup_time, _inference_semaphore

    print("\n" + "="*60, file=sys.stderr)
    print("🚀 Pensieve MLX Server Starting", file=sys.stderr)
    print("="*60, file=sys.stderr)

    _startup_time = time.time()

    # Create concurrency semaphore
    _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    print(f"[CONCURRENCY] Semaphore created: max {MAX_CONCURRENT_INFERENCES} concurrent inferences", file=sys.stderr)

    # Load model once at startup
    if _model_path:
        print(f"[STARTUP] Loading model from: {_model_path}", file=sys.stderr)
        log_memory_state("BEFORE MODEL LOAD")

        try:
            _global_model = load_model(_model_path)
            print("✅ Model loaded successfully and ready for requests", file=sys.stderr)
            log_memory_state("AFTER MODEL LOAD")

            # Apply performance optimizations with reduced cache
            import mlx.core as mx
            if hasattr(mx.metal, 'set_cache_limit'):
                cache_bytes = MLX_METAL_CACHE_MB * 1024 * 1024
                mx.metal.set_cache_limit(cache_bytes)
                print(f"[MEMORY] Set Metal cache limit to {MLX_METAL_CACHE_MB}MB", file=sys.stderr)

            optimize_mlx_performance()

        except Exception as e:
            print(f"❌ FATAL: Failed to load model: {e}", file=sys.stderr)
            raise
    else:
        print("⚠️  WARNING: No model path provided, server running without model", file=sys.stderr)

    print("\n" + "="*60, file=sys.stderr)
    print("✅ Server Ready - Model Resident in Memory", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    yield  # Server runs here

    # Cleanup on shutdown
    print("\n[SHUTDOWN] Cleaning up...", file=sys.stderr)
    clear_mlx_cache()
    print("✅ Shutdown complete", file=sys.stderr)


# Create FastAPI app with lifespan
app = FastAPI(
    title="Pensieve MLX Server",
    description="Persistent MLX inference server for Pensieve Local LLM",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint with memory and performance information

    Returns:
        - Server status
        - Model load status
        - Memory status
        - Performance metrics
    """
    mem_status, mem_available = check_memory_status()
    uptime = time.time() - _startup_time if _startup_time else 0

    return HealthResponse(
        status="healthy" if _global_model else "no_model",
        model_loaded=_global_model is not None,
        model_path=_model_path,
        uptime_seconds=round(uptime, 2),
        memory_status=mem_status,
        memory_available_gb=round(mem_available, 2),
        concurrent_limit=MAX_CONCURRENT_INFERENCES,
        performance_metrics=get_performance_metrics()
    )


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate text using the persistent MLX model

    This endpoint uses the SAME model loaded at startup, avoiding memory spikes.

    Concurrency control:
    - Semaphore limits parallel inferences
    - Memory is checked before each request
    - Requests are rejected if memory is critical

    Args:
        request: GenerateRequest with prompt, max_tokens, temperature, stream

    Returns:
        - Non-streaming: JSON with generated text
        - Streaming: Server-Sent Events (SSE) with text chunks
    """
    # Check if model is loaded
    if not _global_model:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be starting up."
        )

    # Memory safety check BEFORE inference
    mem_status, mem_available = check_memory_status()

    if mem_status == 'CRITICAL':
        log_memory_state("REJECTED - CRITICAL")
        raise HTTPException(
            status_code=503,
            detail=f"Insufficient memory: {mem_available:.2f}GB available. Request rejected for safety."
        )

    if mem_status == 'EMERGENCY':
        log_memory_state("REJECTED - EMERGENCY")
        # Emergency: clear cache and reject
        clear_mlx_cache()
        raise HTTPException(
            status_code=503,
            detail=f"Memory exhaustion: {mem_available:.2f}GB available. Emergency cache cleared."
        )

    # Acquire semaphore for concurrency control
    async with _inference_semaphore:
        print(f"[REQUEST] Prompt: {request.prompt[:50]}... | Tokens: {request.max_tokens} | Stream: {request.stream}", file=sys.stderr)
        log_memory_state("START REQUEST")

        try:
            if request.stream:
                # Streaming response
                return StreamingResponse(
                    stream_generator(
                        _global_model,
                        request.prompt,
                        request.max_tokens,
                        request.temperature
                    ),
                    media_type="application/x-ndjson"  # Newline-delimited JSON
                )
            else:
                # Non-streaming response
                result = await generate_non_streaming(
                    _global_model,
                    request.prompt,
                    request.max_tokens,
                    request.temperature
                )
                log_memory_state("END REQUEST")
                return JSONResponse(content=result)

        except Exception as e:
            log_memory_state("ERROR")
            print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=str(e))


async def generate_non_streaming(
    model_data: Dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float
) -> Dict[str, Any]:
    """
    Generate text without streaming (returns complete response)

    This runs in an executor to avoid blocking the event loop,
    since MLX operations are CPU/GPU intensive.
    """
    loop = asyncio.get_event_loop()

    def sync_generate():
        start_time = time.time()
        full_text = ""

        # Use existing real_mlx_generate from mlx_inference.py
        for chunk in real_mlx_generate(
            model_data,
            prompt,
            max_tokens,
            temperature,
            stream=False
        ):
            full_text += chunk

        elapsed = time.time() - start_time
        token_count = len(full_text.split())
        tps = token_count / elapsed if elapsed > 0 else 0

        return {
            "text": full_text,
            "tokens": token_count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tps, 2),
            "memory_status": check_memory_status()[0]
        }

    # Run in thread pool to avoid blocking
    result = await loop.run_in_executor(None, sync_generate)
    return result


async def stream_generator(
    model_data: Dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float
):
    """
    Async generator for streaming responses

    Yields newline-delimited JSON chunks as they're generated.
    This is TRUE streaming - no buffering!
    """
    loop = asyncio.get_event_loop()

    # Create a queue for chunks
    chunk_queue = asyncio.Queue()

    def sync_stream():
        """Run MLX streaming in thread pool"""
        try:
            for chunk in real_mlx_generate(
                model_data,
                prompt,
                max_tokens,
                temperature,
                stream=True
            ):
                # Put chunk in queue (thread-safe)
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put({"type": "chunk", "text": chunk}),
                    loop
                )

            # Signal completion
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put({"type": "done"}),
                loop
            )
        except Exception as e:
            # Signal error
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put({"type": "error", "error": str(e)}),
                loop
            )

    # Start generation in background
    loop.run_in_executor(None, sync_stream)

    # Yield chunks as they arrive
    while True:
        chunk = await chunk_queue.get()

        if chunk["type"] == "done":
            yield json.dumps({"type": "complete"}) + "\n"
            break
        elif chunk["type"] == "error":
            yield json.dumps({"type": "error", "error": chunk["error"]}) + "\n"
            break
        else:
            yield json.dumps(chunk) + "\n"


@app.get("/metrics")
async def metrics():
    """Get detailed performance metrics"""
    return get_performance_metrics()


@app.post("/cache/clear")
async def clear_cache():
    """Manually trigger MLX cache clearing (emergency use)"""
    log_memory_state("BEFORE CLEAR")
    clear_mlx_cache()
    log_memory_state("AFTER CLEAR")

    return {
        "status": "cleared",
        "memory_status": check_memory_status()[0],
        "memory_available_gb": round(check_memory_status()[1], 2)
    }


@app.post("/v1/messages")
async def create_message(request: AnthropicRequest):
    """Anthropic-compatible message generation endpoint"""
    if not _global_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Memory checks
    mem_status, mem_available = check_memory_status()
    if mem_status in ['CRITICAL', 'EMERGENCY']:
        if mem_status == 'EMERGENCY': clear_mlx_cache()
        raise HTTPException(status_code=503, detail=f"Memory pressure: {mem_status}")

    prompt = convert_to_phi3_prompt(request)
    msg_id = f"msg_{uuid.uuid4()}"

    async with _inference_semaphore:
        print(f"[ANTHROPIC] Request: {prompt[:50].replace(chr(10), ' ')}... | Stream: {request.stream}", file=sys.stderr)
        
        if request.stream:
            return StreamingResponse(
                stream_anthropic_generator(
                    _global_model, prompt, request.max_tokens, request.temperature, msg_id
                ),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming
            result = await generate_non_streaming(
                _global_model, prompt, request.max_tokens, request.temperature
            )
            
            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": result["text"]}],
                "model": request.model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": len(prompt.split()), # Approx
                    "output_tokens": result["tokens"]
                }
            }

def main():
    """
    Main entry point for the server

    Usage:
        python3 python_bridge/mlx_server.py --model-path ./models/Phi-3-mini-128k-instruct-4bit
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pensieve MLX Server - Persistent Model")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to MLX model directory (loads once at startup)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent inference requests (default: 2)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn workers (default: 1, must be 1 for model persistence)"
    )

    args = parser.parse_args()

    # Set global model path
    global _model_path, MAX_CONCURRENT_INFERENCES
    _model_path = args.model_path
    MAX_CONCURRENT_INFERENCES = args.max_concurrent

    # Validate model path exists
    from pathlib import Path
    if not Path(args.model_path).exists():
        print(f"❌ ERROR: Model path does not exist: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Starting Pensieve MLX Server", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Model: {args.model_path}", file=sys.stderr)
    print(f"Address: http://{args.host}:{args.port}", file=sys.stderr)
    print(f"Max Concurrent: {args.max_concurrent}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    if args.workers > 1:
        print("⚠️  WARNING: workers > 1 will create multiple model copies!", file=sys.stderr)
        print("    For memory efficiency, keep workers=1", file=sys.stderr)

    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
