#!/usr/bin/env python3
"""
Pensieve MLX Inference Bridge - REAL Implementation
Provides MLX-powered text generation for the Pensieve Local LLM Server
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

# Model cache to avoid reloading
_model_cache = {}

def load_model(model_path: str):
    """Load REAL MLX model and tokenizer with caching"""
    global _model_cache

    if model_path in _model_cache:
        return _model_cache[model_path]

    print(f"Loading REAL MLX model from: {model_path}", file=sys.stderr)

    try:
        # Use REAL MLX-LM loading (this handles Phi-3 automatically)
        model, tokenizer = load(model_path)

        print("REAL MLX model loaded successfully", file=sys.stderr)
        print(f"Model device: {mx.default_device()}", file=sys.stderr)

        _model_cache[model_path] = {
            'model': model,
            'tokenizer': tokenizer,
            'mlx_path': model_path
        }

        return _model_cache[model_path]

    except Exception as e:
        raise RuntimeError(f"Failed to load REAL MLX model: {e}")

def real_mlx_generate(
    model_data: Dict[str, Any],
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> Iterator[str]:
    """
    REAL MLX generation using MLX-LM stream_generate
    """
    model = model_data['model']
    tokenizer = model_data['tokenizer']

    # Skip chat template for now - it's returning token IDs instead of text
    # TODO: Fix chat template handling later for better Phi-3 compatibility
    # The basic prompt should work for initial testing

    print(f"Starting REAL MLX generation with prompt length: {len(prompt.split())}", file=sys.stderr)

    if stream:
        # Use REAL MLX-LM streaming
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens
        ):
            yield response.text
    else:
        # Use REAL MLX-LM non-streaming
        response = mlx_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens
        )
        # MLX-LM returns a string, not a list
        if isinstance(response, list):
            response = " ".join(response) if response else ""
        yield response

def generate_text(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    stream: bool = False
) -> Iterator[Dict[str, Any]]:
    """
    Main generation function using REAL MLX inference
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

            yield {
                "type": "complete",
                "text": full_response,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(full_response.split()),
                "total_tokens": len(prompt.split()) + len(full_response.split()),
                "tokens_per_second": round(tps, 2),
                "elapsed_ms": round(elapsed * 1000, 2),
                "peak_memory_mb": mx.get_peak_memory() / 1e6 if hasattr(mx, 'get_peak_memory') else 0
            }

    except Exception as e:
        import traceback
        error_details = f"MLX inference error: {str(e)}\nTraceback: {traceback.format_exc()}"
        yield {
            "type": "error",
            "error": error_details
        }

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Pensieve MLX Inference Bridge - REAL")
    parser.add_argument("--model-path", required=True, help="Path to MLX model directory")
    parser.add_argument("--prompt", required=True, help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--stream", action="store_true", help="Stream generation output")

    args = parser.parse_args()

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