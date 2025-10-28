#!/usr/bin/env python3
"""
Corrected MLX Implementation Example for Phi-3-mini-128k-instruct-4bit
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import time

print("=== MLX Framework Test ===")

# Test 1: Hardware Detection
print("\n1. Hardware Detection:")
print(f"Metal available: {mx.metal.is_available()}")
print(f"GPU device: {mx.gpu}")
print(f"CPU device: {mx.cpu}")
print(f"Default device: {mx.default_device()}")

# Test 2: Basic operations
print("\n2. Basic Operations:")
a = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
b = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
c = mx.matmul(a, b)
print(f"A:\n{a}")
print(f"B:\n{b}")
print(f"A @ B:\n{c}")

# Test 3: Neural Network
print("\n3. Neural Network:")
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
print(f"Model created with {len(tree_flatten(model.parameters()))} parameters")
for name, param in tree_flatten(model.parameters()):
    print(f"  {name}: {param.shape}")

# Test 4: Performance
print("\n4. Performance Test:")
size = 1024  # Reduced for quick test
x = mx.random.normal((size, size))
y = mx.random.normal((size, size))

print(f"Performing {size}x{size} matrix multiplication...")
start = time.time()
z = mx.matmul(x, y)
end = time.time()

print(f"Time: {end - start:.3f} seconds")
print(f"Result shape: {z.shape}")

# Test 5: Memory usage
print("\n5. Memory Usage:")
print(f"Input memory: {x.nbytes / 1024**2:.2f} MB")
print(f"Output memory: {z.nbytes / 1024**2:.2f} MB")

print("\n=== MLX Test Complete ===")
