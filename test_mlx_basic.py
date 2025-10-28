#!/usr/bin/env python3
"""
Basic MLX framework test to understand the API and capabilities
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

print("=== MLX Framework Test ===")

# Test 1: Basic MX operations
print("\n1. Basic MX Operations:")
a = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
b = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
c = mx.dot(a, b)
print(f"A:\n{a}")
print(f"B:\n{b}")
print(f"A @ B:\n{c}")

# Test 2: Neural Network Module
print("\n2. Neural Network Module:")
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
print(f"Model parameters: {len(tree_flatten(model.parameters()))}")
for name, param in tree_flatten(model.parameters()):
    print(f"  {name}: {param.shape} {param.dtype}")

# Test 3: GPU/CPU Detection
print("\n3. Hardware Detection:")
print(f"Default device: {mx.default_device()}")
print(f"Available devices: {mx.devices()}")
print(f"Metal available: {mx.metal.is_available()}")

# Test 4: Performance Test
print("\n4. Performance Test:")
import time

# Large matrix multiplication
size = 2048
x = mx.random.normal((size, size))
y = mx.random.normal((size, size))

print(f"Performing {size}x{size} matrix multiplication...")
start = time.time()
z = mx.dot(x, y)
end = time.time()

print(f"Time: {end - start:.3f} seconds")
print(f"Result shape: {z.shape}")

# Test 5: Quantization Support
print("\n5. Quantization Support:")
original = mx.random.normal((1000, 1000))
print(f"Original shape: {original.shape}")
print(f"Original dtype: {original.dtype}")

# Test quantization simulation (since MLX doesn't have explicit quantization APIs)
quantized = original.astype(mx.float16)
print(f"Quantized dtype: {quantized.dtype}")
print(f"Memory reduction: {original.nbytes / quantized.nbytes:.1f}x")

print("\n=== MLX Test Complete ===")
