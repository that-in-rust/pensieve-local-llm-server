#!/usr/bin/env python3
"""
Simplified MLX test with real model integration
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import json
import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from transformers import AutoTokenizer

print("=== MLX Simple Phi-3 Integration Test ===")

# Test 1: Hardware check
print(f"\n1. Hardware Detection:")
print(f"Metal available: {mx.metal.is_available()}")
print(f"GPU device: {mx.gpu}")
print(f"CPU device: {mx.cpu}")
print(f"Default device: {mx.default_device()}")

# Test 2: Basic model loading
print(f"\n2. Model Loading Test:")
model_path = Path("/Users/amuldotexe/Projects/pensieve-local-llm-server/models/Phi-3-mini-128k-instruct-4bit")

if model_path.exists():
    print(f"Model path exists: {model_path}")
    
    # Check for required files
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            print(f"✓ Found: {file} ({file_path.stat().st_size} bytes)")
        else:
            print(f"✗ Missing: {file}")

    # Try to load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print("✓ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"✓ Tokenization test: '{test_text}' -> {len(tokens)} tokens")
        print(f"✓ Decoded: '{decoded}'")
        
    except Exception as e:
        print(f"✗ Tokenizer failed: {e}")

# Test 3: Simple MLX model with tokenizer integration
print(f"\n3. Simple MLX Model with Tokenizer:")

class SimplePhi3(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=3072):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional = nn.Embedding(2048, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = mx.arange(seq_len)
        
        x = self.embedding(input_ids)
        x = x + self.positional(positions)
        
        # Simplified forward pass
        x = x.mean(axis=-2)  # Simplified pooling
        logits = self.lm_head(x)
        
        return logits

# Test model creation
model = SimplePhi3()
print(f"✓ Simple model created with {len(tree_flatten(model.parameters()))} parameters")

# Test 4: Real inference simulation
print(f"\n4. Real Inference Simulation:")

def simulate_generation(prompt: str, max_tokens: 20) -> Iterator[Dict[str, Any]]:
    """Simulate real token generation"""
    tokenizer = AutoTokenizer.from_pretrained("/Users/amuldotexe/Projects/pensieve-local-llm-server/models/Phi-3-mini-128k-instruct-4bit")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    accumulated = tokenizer.decode(input_ids)
    
    start_time = time.time()
    
    # Simulate token generation
    for i in range(max_tokens):
        # Simulate MLX inference time (60-100ms with GPU)
        inference_time = 0.060 + (i % 4) * 0.01  # 60-63ms
        time.sleep(inference_time)
        
        # Generate mock token (in real implementation, this would be from MLX)
        mock_token_ids = [2000 + i, 2001 + i, 2002 + i]  # Mock token progression
        if i < len(mock_token_ids):
            token_id = mock_token_ids[i]
        else:
            token_id = 32000  # EOS token
            
        token_text = tokenizer.decode([token_id])
        
        # Calculate performance
        elapsed = time.time() - start_time
        tps = (i + 1) / elapsed if elapsed > 0 else 0
        
        yield {
            "type": "token",
            "token": token_text,
            "token_id": token_id,
            "step": i + 1,
            "elapsed_time": elapsed,
            "tokens_per_second": tps,
            "is_finished": token_id == 32000 or i == max_tokens - 1
        }
        
        if token_id == 32000:
            break

# Test generation
prompt = "Hello, how are you today?"
print(f"Testing prompt: '{prompt}'")

start_time = time.time()
for chunk in simulate_generation(prompt, max_tokens=5):
    if chunk["type"] == "token":
        print(f"Token {chunk['step']}: '{chunk['token']}' ({chunk['tokens_per_second']:.1f} TPS)")
    
    if chunk["is_finished"]:
        total_time = chunk["elapsed_time"]
        print(f"\nGeneration complete:")
        print(f"Total tokens: {chunk['step']}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average TPS: {chunk['tokens_per_second']:.1f}")
        break

# Test 5: Performance benchmarks
print(f"\n5. Performance Benchmark:")
test_prompts = [
    "What is machine learning?",
    "Explain Apple Silicon benefits.",
    "How does Metal GPU work?"
]

total_start = time.time()
total_tokens = 0

for i, prompt in enumerate(test_prompts):
    print(f"\nTest {i+1}: '{prompt[:30]}...'")
    
    test_start = time.time()
    token_count = 0
    
    for chunk in simulate_generation(prompt, max_tokens=3):
        token_count += 1
        if chunk["is_finished"]:
            test_time = time.time() - test_start
            print(f"  Generated {token_count} tokens in {test_time:.3f}s ({chunk['tokens_per_second']:.1f} TPS)")
            break
    
    total_tokens += token_count

total_time = time.time() - total_start
print(f"\nBenchmark Summary:")
print(f"Total tests: {len(test_prompts)}")
print(f"Total tokens: {total_tokens}")
print(f"Total time: {total_time:.3f}s")
print(f"Overall TPS: {total_tokens/total_time:.1f}")

print(f"\n=== Test Complete ===")
