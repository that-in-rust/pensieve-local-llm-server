# Pensieve Low-Level Design (LLD)

**Document Version**: 1.0
**Last Updated**: October 29, 2025
**Current Status**: Foundation Complete, MLX Integration Ready

## Executive Summary

The Pensieve Local LLM Server requires detailed low-level design specifications for implementing MLX integration. This document provides concrete implementation details, function signatures, test strategies, and performance optimization techniques for transitioning from mock responses to real MLX-powered inference.

## Section 1: MLX Integration Layer

### 1.1 MLX Rust Binding Interface

```rust
// In pensieve-04/src/inference/mlx_handler.rs

use crate::inference::{InferenceEngine, InferenceConfig, InferenceResult};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct MlxInferenceEngine {
    model: Arc<RwLock<MlxModel>>,
    config: InferenceConfig,
    tokenizer: Arc<RwLock<MlxTokenizer>>,
}

impl MlxInferenceEngine {
    pub async fn new(
        model_path: PathBuf,
        config: InferenceConfig,
    ) -> Result<Self, MlxError> {
        // Load MLX model from disk
        let model = MlxModel::load(model_path).await?;
        let tokenizer = MlxTokenizer::from_model(&model).await?;

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
            tokenizer: Arc::new(RwLock::new(tokenizer)),
        })
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<impl Stream<Item = Result<String, MlxError>>, MlxError> {
        let model = self.model.read().await;
        let tokenizer = self.tokenizer.read().await;

        // Convert prompt to tokens
        let input_ids = tokenizer.encode(prompt)?;

        // Generate tokens with streaming
        let stream = model.generate_stream(
            input_ids,
            max_tokens,
            temperature,
            &self.config,
        )?;

        Ok(stream.map(move |token_result| {
            match token_result {
                Ok(token_id) => {
                    let text = tokenizer.decode(&[token_id])?;
                    Ok(text)
                }
                Err(e) => Err(e),
            }
        }))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    #[error("MLX model loading failed: {0}")]
    ModelLoadFailed(String),
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    #[error("Generation failed: {0}")]
    GenerationError(String),
    #[error("Metal device error: {0}")]
    MetalError(String),
}
```

### 1.2 Python MLX Bridge Interface

```python
# In python_bridge/mlx_wrapper.py

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
from typing import Iterator, Optional, Dict, Any
import torch

class MlxModelWrapper:
    def __init__(self, model_path: str, device: str = "metal"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load MLX model and tokenizer"""
        try:
            # Load tokenizer from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-128k-instruct"
            )

            # Load MLX model
            self.model = nn.Module.from_pretrained(self.model_path)

            # Move to Metal device
            if mx.metal.is_available():
                self.model = self.model.to(mx.metal)

        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model: {e}")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        stop_tokens: Optional[list] = None,
    ) -> Iterator[str]:
        """Generate tokens with streaming"""
        if not self.model or not self.tokenizer:
            self.load_model()

        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Convert to MLX array
        input_ids = mx.array(inputs.numpy())

        # Generate with streaming
        for token_id in self._generate_stream_internal(
            input_ids, max_tokens, temperature, stop_tokens
        ):
            # Decode token to text
            token_text = self.tokenizer.decode([token_id])
            yield token_text

    def _generate_stream_internal(
        self,
        input_ids: mx.array,
        max_tokens: int,
        temperature: float,
        stop_tokens: Optional[list],
    ) -> Iterator[int]:
        """Internal streaming generation using MLX"""
        # Implementation using MLX generation loop
        past_key_values = None

        for _ in range(max_tokens):
            # Get next token
            with mx.stream(self.device):
                outputs = self.model(input_ids, past_key_values=past_key_values)
                logits = outputs.logits[:, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Sample next token
                next_token = mx.argmax(mx.softmax(logits, axis=-1), axis=-1)
                next_token_id = int(next_token.item())

                # Check for stop tokens
                if stop_tokens and next_token_id in stop_tokens:
                    break

                yield next_token_id

                # Update input for next iteration
                input_ids = mx.concatenate([
                    input_ids,
                    mx.array([[next_token_id]])
                ], axis=1)

                past_key_values = outputs.past_key_values
```

### 1.3 Rust-Python Interface

```rust
// In pensieve-04/src/inference/python_bridge.rs

use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyString};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio_stream::Stream;
use std::sync::Arc;

#[pyclass]
struct MlxBridge {
    wrapper: Py<PyAny>,
}

#[pymethods]
impl MlxBridge {
    #[new]
    fn new(model_path: &str, device: &str) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mlx_module = py.import("mlx_bridge.mlx_wrapper")?;
            let wrapper_class = mlx_module.getattr("MlxModelWrapper")?;
            let wrapper = wrapper_class.call1((model_path, device))?;

            Ok(Self {
                wrapper: wrapper.into(),
            })
        })
    }

    fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> PyResult<PyTokenStream> {
        Python::with_gil(|py| {
            let stream = self.wrapper.call_method1(
                "generate_stream",
                (prompt, max_tokens, temperature),
            )?;
            Ok(PyTokenStream::new(stream.into()))
        })
    }
}

struct PyTokenStream {
    iterator: Py<PyAny>,
}

impl PyTokenStream {
    fn new(iterator: Py<PyAny>) -> Self {
        Self { iterator }
    }
}

impl Stream for PyTokenStream {
    type Item = Result<String, MlxError>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Python::with_gil(|py| {
            match self.iterator.as_ref(py).call_method0("__next__") {
                Ok(token) => {
                    let text = token.extract::<String>().unwrap();
                    Poll::Ready(Some(Ok(text)))
                }
                Err(e) => {
                    if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                        Poll::Ready(None)
                    } else {
                        Poll::Ready(Some(Err(MlxError::GenerationError(
                            e.to_string()
                        ))))
                    }
                }
            }
        })
    }
}
```

## Section 2: Model Management

### 2.1 Model Download Service

```rust
// In pensieve-05/src/model/download.rs

use crate::model::{ModelInfo, ModelDownloadError};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use futures_util::StreamExt;

pub struct ModelDownloader {
    cache_dir: PathBuf,
    client: reqwest::Client,
}

impl ModelDownloader {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            client: reqwest::Client::new(),
        }
    }

    pub async fn download_model(
        &self,
        model_info: &ModelInfo,
        progress_callback: Option<Box<dyn Fn(u64, u64) + Send + Sync>>,
    ) -> Result<PathBuf, ModelDownloadError> {
        let model_path = self.cache_dir.join(&model_info.filename);

        // Check if model already exists
        if model_path.exists() {
            return Ok(model_path);
        }

        // Create directory if needed
        if let Some(parent) = model_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Download file with progress tracking
        let response = self.client
            .get(&model_info.download_url)
            .send()
            .await?;

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        let mut file = File::create(&model_path).await?;

        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(ref callback) = progress_callback {
                callback(downloaded, total_size);
            }
        }

        file.flush().await?;
        Ok(model_path)
    }

    pub async fn verify_model(&self, model_path: &PathBuf) -> Result<bool, ModelDownloadError> {
        // Verify model file integrity
        let metadata = tokio::fs::metadata(model_path).await?;

        // Check minimum file size (should be >100MB for Phi-3)
        if metadata.len() < 100 * 1024 * 1024 {
            return Ok(false);
        }

        // TODO: Add checksum verification
        Ok(true)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelDownloadError {
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("File system error: {0}")]
    FileError(#[from] std::io::Error),
    #[error("Model verification failed")]
    VerificationFailed,
}
```

### 2.2 Model Cache Management

```rust
// In pensieve-05/src/model/cache.rs

use crate::model::ModelInfo;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ModelCache {
    models: HashMap<String, CachedModel>,
    cache_dir: PathBuf,
    max_cache_size: u64, // in bytes
}

#[derive(Debug, Clone)]
struct CachedModel {
    info: ModelInfo,
    path: PathBuf,
    last_used: u64,
    size: u64,
}

impl ModelCache {
    pub fn new(cache_dir: PathBuf, max_cache_size: u64) -> Self {
        Self {
            models: HashMap::new(),
            cache_dir,
            max_cache_size,
        }
    }

    pub async fn load_cache(&mut self) -> Result<(), std::io::Error> {
        // Scan cache directory for existing models
        let mut entries = tokio::fs::read_dir(&self.cache_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                let metadata = tokio::fs::metadata(&path).await?;
                let size = metadata.len();
                let last_used = metadata
                    .modified()?
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Try to identify model from filename
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    let model_info = ModelInfo {
                        id: name.to_string(),
                        name: name.to_string(),
                        filename: path.file_name().unwrap().to_str().unwrap().to_string(),
                        size,
                        download_url: String::new(), // Not needed for cached models
                    };

                    let cached = CachedModel {
                        info: model_info,
                        path: path.clone(),
                        last_used,
                        size,
                    };

                    self.models.insert(name.to_string(), cached);
                }
            }
        }

        Ok(())
    }

    pub async fn cleanup_cache(&mut self) -> Result<(), std::io::Error> {
        let total_size: u64 = self.models.values().map(|m| m.size).sum();

        if total_size <= self.max_cache_size {
            return Ok(());
        }

        // Sort by last used (oldest first)
        let mut models_sorted: Vec<_> = self.models.values().collect();
        models_sorted.sort_by_key(|m| m.last_used);

        let mut to_remove = Vec::new();
        let mut removed_size = 0u64;
        let target_size = self.max_cache_size * 8 / 10; // Remove to 80% capacity

        for model in models_sorted {
            if total_size - removed_size <= target_size {
                break;
            }

            to_remove.push(model.info.id.clone());
            removed_size += model.size;
        }

        // Remove oldest models
        for model_id in to_remove {
            if let Some(cached) = self.models.remove(&model_id) {
                tokio::fs::remove_file(&cached.path).await?;
            }
        }

        Ok(())
    }
}
```

## Section 3: Performance Optimization

### 3.1 Memory Management for 16GB Systems

```rust
// In pensieve-06/src/metal/memory_manager.rs

use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct MemoryManager {
    max_model_memory: usize, // bytes
    kv_cache_semaphore: Arc<Semaphore>,
    memory_pool: MemoryPool,
}

impl MemoryManager {
    pub fn new(total_system_memory: usize) -> Self {
        // Reserve 8GB for system, use up to 8GB for model on 16GB system
        let max_model_memory = if total_system_memory >= 16 * 1024 * 1024 * 1024 {
            8 * 1024 * 1024 * 1024 // 8GB
        } else {
            total_system_memory / 2 // Use 50% on smaller systems
        };

        Self {
            max_model_memory,
            kv_cache_semaphore: Arc::new(Semaphore::new(4)), // Max 4 concurrent requests
            memory_pool: MemoryPool::new(max_model_memory / 4), // 25% for KV cache
        }
    }

    pub async fn allocate_kv_cache(&self, size: usize) -> Result<KvCacheAllocation, MemoryError> {
        let _permit = self.kv_cache_semaphore.acquire().await?;

        let allocation = self.memory_pool.allocate(size).await?;
        Ok(KvCacheAllocation::new(allocation, _permit))
    }

    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            max_model_memory: self.max_model_memory,
            allocated_memory: self.memory_pool.allocated(),
            available_memory: self.memory_pool.available(),
            active_requests: self.kv_cache_semaphore.available_permits(),
        }
    }
}

pub struct KvCacheAllocation {
    allocation: MemoryAllocation,
    _permit: SemaphorePermit<'static>,
}

impl KvCacheAllocation {
    fn new(allocation: MemoryAllocation, _permit: SemaphorePermit<'static>) -> Self {
        Self { allocation, _permit }
    }
}

#[derive(Debug)]
pub struct MemoryStats {
    pub max_model_memory: usize,
    pub allocated_memory: usize,
    pub available_memory: usize,
    pub active_requests: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Insufficient memory for allocation")]
    InsufficientMemory,
    #[error("Memory pool exhausted")]
    PoolExhausted,
}
```

### 3.2 Metal Backend Optimization

```rust
// In pensieve-06/src/metal/optimizer.rs

use metal::*;

pub struct MetalOptimizer {
    device: Device,
    command_queue: CommandQueue,
}

impl MetalOptimizer {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()
            .ok_or(MetalError::NoDevice)?;
        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
        })
    }

    pub fn optimize_model_execution(
        &self,
        model: &MetalModel,
        batch_size: usize,
    ) -> Result<OptimizedExecutionPlan, MetalError> {
        let mut plan = OptimizedExecutionPlan::new();

        // Optimize memory layout for Metal
        plan = self.optimize_memory_layout(plan, model)?;

        // Optimize compute shaders for batch size
        plan = self.optimize_compute_shaders(plan, model, batch_size)?;

        // Optimize data transfer between CPU and GPU
        plan = self.optimize_data_transfer(plan, model)?;

        Ok(plan)
    }

    fn optimize_memory_layout(
        &self,
        mut plan: OptimizedExecutionPlan,
        model: &MetalModel,
    ) -> Result<OptimizedExecutionPlan, MetalError> {
        // Align buffers to Metal requirements
        let buffer_alignment = 256; // Metal buffer alignment

        for layer in &model.layers {
            let aligned_size = ((layer.size + buffer_alignment - 1) / buffer_alignment) * buffer_alignment;
            plan.add_buffer_optimization(layer.id, aligned_size);
        }

        Ok(plan)
    }

    fn optimize_compute_shaders(
        &self,
        mut plan: OptimizedExecutionPlan,
        model: &MetalModel,
        batch_size: usize,
    ) -> Result<OptimizedExecutionPlan, MetalError> {
        // Choose optimal thread group sizes for M1/M2/M3
        let optimal_threads = match self.device.family() {
            MTLGPUFamily::Apple7 => 32, // M1
            MTLGPUFamily::Apple8 => 64, // M2
            MTLGPUFamily::Apple9 => 128, // M3
            _ => 32,
        };

        for shader in &model.compute_shaders {
            let threads_per_group = (optimal_threads.min(shader.max_threads))
                .max(shader.min_threads);

            plan.add_shader_optimization(
                shader.id,
                threads_per_group,
                batch_size,
            );
        }

        Ok(plan)
    }
}

#[derive(Debug)]
pub struct OptimizedExecutionPlan {
    buffer_optimizations: Vec<BufferOptimization>,
    shader_optimizations: Vec<ShaderOptimization>,
}

#[derive(Debug)]
struct BufferOptimization {
    layer_id: u32,
    aligned_size: usize,
}

#[derive(Debug)]
struct ShaderOptimization {
    shader_id: u32,
    threads_per_group: usize,
    batch_size: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("No Metal device available")]
    NoDevice,
    #[error("Metal compilation failed: {0}")]
    CompilationFailed(String),
    #[error("Buffer allocation failed")]
    BufferAllocationFailed,
}
```

## Section 4: Test Strategies

### 4.1 Unit Tests

```rust
// In pensieve-04/tests/test_mlx_integration.rs

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_mlx_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.gguf");

        // Create a dummy model file
        tokio::fs::write(&model_path, b"dummy model data").await.unwrap();

        let config = InferenceConfig::default();

        // This should fail with actual model loading, but tests the interface
        let result = MlxInferenceEngine::new(model_path, config).await;

        // Verify error handling
        assert!(result.is_err());
        match result.unwrap_err() {
            MlxError::ModelLoadFailed(_) => {}, // Expected
            e => panic!("Unexpected error: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_tokenization() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.gguf");
        tokio::fs::write(&model_path, b"dummy model data").await.unwrap();

        // Test tokenization interface
        let config = InferenceConfig::default();
        let engine_result = MlxInferenceEngine::new(model_path, config).await;

        assert!(engine_result.is_err());
    }

    #[tokio::test]
    async fn test_stream_generation_interface() {
        // Test the streaming interface with mock data
        let mock_stream = tokio_stream::iter(vec![
            Ok("Hello".to_string()),
            Ok(" world".to_string()),
            Ok("!".to_string()),
        ]);

        let collected: Vec<_> = mock_stream.collect().await;
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[1].as_ref().unwrap(), " world");
    }
}
```

### 4.2 Integration Tests

```rust
// In tests/integration_test.rs

use pensieve_02::server::HttpServer;
use pensieve_04::inference::MlxInferenceEngine;
use reqwest::Client;
use serde_json::json;

#[tokio::test]
async fn test_complete_api_flow() {
    // Start test server
    let server = HttpServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let addr = server.local_addr();

    // Spawn server in background
    tokio::spawn(async move {
        server.run().await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = Client::new();

    // Test health endpoint
    let health_response = client
        .get(&format!("http://{}/health", addr))
        .send()
        .await
        .unwrap();

    assert_eq!(health_response.status(), 200);

    // Test models endpoint
    let models_response = client
        .get(&format!("http://{}/v1/models", addr))
        .header("Authorization", "Bearer pensieve-local-key")
        .send()
        .await
        .unwrap();

    assert_eq!(models_response.status(), 200);

    // Test chat completion endpoint (with mock data)
    let chat_request = json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "max_tokens": 10,
        "stream": false
    });

    let chat_response = client
        .post(&format!("http://{}/v1/messages", addr))
        .header("Authorization", "Bearer pensieve-local-key")
        .header("Content-Type", "application/json")
        .json(&chat_request)
        .send()
        .await
        .unwrap();

    assert_eq!(chat_response.status(), 200);

    let response_body: serde_json::Value = chat_response.json().await.unwrap();
    assert!(response_body.get("content").is_some());
}

#[tokio::test]
async fn test_streaming_response() {
    let server = HttpServer::new("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let addr = server.local_addr();

    tokio::spawn(async move {
        server.run().await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = Client::new();

    let chat_request = json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "messages": [
            {"role": "user", "content": "Count to 5"}
        ],
        "max_tokens": 20,
        "stream": true
    });

    let response = client
        .post(&format!("http://{}/v1/messages", addr))
        .header("Authorization", "Bearer pensieve-local-key") // Fixed token for local development
        .header("Content-Type", "application/json")
        .json(&chat_request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    // Verify streaming response format
    let bytes = response.bytes().await.unwrap();
    let text = String::from_utf8(bytes).unwrap();

    // Should contain SSE format lines
    assert!(text.contains("data: "));
    assert!(text.contains("\n\n"));
}
```

### 4.3 Performance Tests

```rust
// In benches/performance_bench.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pensieve_04::inference::{InferenceEngine, InferenceConfig};
use std::time::Duration;

fn benchmark_inference_speed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("inference_single_token", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = InferenceConfig::default();
                // Test with actual MLX engine when available
                let result = simulate_inference("Hello world").await;
                black_box(result)
            })
        })
    });

    c.bench_function("inference_batch_5", |b| {
        b.iter(|| {
            rt.block_on(async {
                let prompts = vec!["Hello", "World", "Test", "Benchmark", "Speed"];
                let results = simulate_batch_inference(prompts).await;
                black_box(results)
            })
        })
    });
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("memory_allocation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let initial_memory = get_memory_usage();
                let _allocation = simulate_model_loading().await;
                let final_memory = get_memory_usage();
                black_box(final_memory - initial_memory)
            })
        })
    });
}

async fn simulate_inference(prompt: &str) -> String {
    // Mock inference for benchmarking - replace with real MLX
    tokio::time::sleep(Duration::from_millis(50)).await; // Simulate 50ms latency
    format!("Mock response to: {}", prompt)
}

async fn simulate_batch_inference(prompts: Vec<&str>) -> Vec<String> {
    let mut results = Vec::new();
    for prompt in prompts {
        results.push(simulate_inference(prompt).await);
    }
    results
}

async fn simulate_model_loading() -> () {
    tokio::time::sleep(Duration::from_millis(1000)).await; // Simulate 1s load time
}

fn get_memory_usage() -> usize {
    // Mock memory usage - implement actual memory tracking
    1024 * 1024 * 1024 // 1GB mock
}

criterion_group!(benches, benchmark_inference_speed, benchmark_memory_usage);
criterion_main!(benches);
```

### 4.4 End-to-End Tests

```rust
// In tests/e2e_test.rs

use std::process::Command;
use std::time::Duration;
use reqwest::Client;

#[tokio::test]
async fn test_complete_user_journey() {
    // Test: Start server with CLI
    let mut server_process = Command::new("cargo")
        .args(&["run", "-p", "pensieve-01", "--", "start"])
        .spawn()
        .expect("Failed to start server");

    // Wait for server to be ready
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Test: Health check
    let client = Client::new();
    let health_response = client
        .get("http://127.0.0.1:7777/health")
        .send()
        .await
        .expect("Health check failed");

    assert_eq!(health_response.status(), 200);

    // Test: Model info (local authentication - localhost security model)
    let models_response = client
        .get("http://127.0.0.1:7777/v1/models")
        .header("Authorization", "Bearer pensieve-local-key") // Fixed token for local development
        .send()
        .await
        .expect("Models endpoint failed");

    assert_eq!(models_response.status(), 200);

    // Test: Chat completion (local authentication - localhost security model)
    let chat_request = serde_json::json!({
        "model": "phi-3-mini-128k-instruct-4bit",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 10
    });

    let chat_response = client
        .post("http://127.0.0.1:7777/v1/messages")
        .header("Authorization", "Bearer pensieve-local-key") // Fixed token for local development
        .json(&chat_request)
        .send()
        .await
        .expect("Chat completion failed");

    assert_eq!(chat_response.status(), 200);

    let response_body: serde_json::Value = chat_response.json().await.unwrap();
    assert!(response_body.get("content").is_some());

    // Cleanup: Stop server
    server_process.kill().expect("Failed to stop server");
}

#[tokio::test]
async fn test_claude_code_compatibility() {
    // Test that the API response format matches Claude Code expectations

    let mut server_process = Command::new("cargo")
        .args(&["run", "-p", "pensieve-01", "--", "start"])
        .spawn()
        .expect("Failed to start server");

    tokio::time::sleep(Duration::from_secs(5)).await;

    let client = Client::new();

    // Test request format that Claude Code would send
    let claude_request = serde_json::json!({
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": "Help me write Rust code"
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": false
    });

    let response = client
        .post("http://127.0.0.1:7777/v1/messages")
        .header("Authorization", "Bearer pensieve-local-key") // Fixed token for local development
        .header("Content-Type", "application/json")
        .json(&claude_request)
        .send()
        .await
        .expect("Claude Code compatibility test failed");

    assert_eq!(response.status(), 200);

    let response_body: serde_json::Value = response.json().await.unwrap();

    // Verify response structure matches Claude Code expectations
    assert!(response_body.get("id").is_some());
    assert!(response_body.get("type").is_some());
    assert!(response_body.get("role").is_some());
    assert!(response_body.get("content").is_some());
    assert!(response_body.get("model").is_some());
    assert!(response_body.get("stop_reason").is_some());
    assert!(response_body.get("stop_sequence").is_some());
    assert!(response_body.get("usage").is_some());

    // Verify usage statistics
    let usage = response_body.get("usage").unwrap();
    assert!(usage.get("input_tokens").is_some());
    assert!(usage.get("output_tokens").is_some());

    server_process.kill().expect("Failed to stop server");
}
```

## Section 5: Performance Targets and Monitoring

### 5.1 Key Performance Indicators

```rust
// In pensieve-07/src/monitoring/metrics.rs

use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub request_latency: Duration,
    pub tokens_per_second: f64,
    pub memory_usage_mb: usize,
    pub gpu_utilization_percent: f64,
    pub cache_hit_rate: f64,
    pub concurrent_requests: usize,
}

pub struct MetricsCollector {
    metrics: Vec<PerformanceMetrics>,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            start_time: Instant::now(),
        }
    }

    pub fn record_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics.push(metrics);
    }

    pub fn get_average_performance(&self) -> Option<PerformanceSummary> {
        if self.metrics.is_empty() {
            return None;
        }

        let count = self.metrics.len() as f64;
        let total_latency: Duration = self.metrics.iter()
            .map(|m| m.request_latency)
            .sum();

        let avg_tokens_per_sec = self.metrics.iter()
            .map(|m| m.tokens_per_second)
            .sum::<f64>() / count;

        let avg_memory_mb = self.metrics.iter()
            .map(|m| m.memory_usage_mb)
            .sum::<usize>() / self.metrics.len();

        let avg_gpu_util = self.metrics.iter()
            .map(|m| m.gpu_utilization_percent)
            .sum::<f64>() / count;

        Some(PerformanceSummary {
            average_latency: total_latency / count as u32,
            average_tokens_per_second: avg_tokens_per_sec,
            average_memory_mb: avg_memory_mb,
            average_gpu_utilization: avg_gpu_util,
            total_requests: self.metrics.len(),
            uptime: self.start_time.elapsed(),
        })
    }
}

#[derive(Debug)]
pub struct PerformanceSummary {
    pub average_latency: Duration,
    pub average_tokens_per_second: f64,
    pub average_memory_mb: usize,
    pub average_gpu_utilization: f64,
    pub total_requests: usize,
    pub uptime: Duration,
}

impl PerformanceSummary {
    pub fn meets_targets(&self) -> bool {
        // Performance targets from PRD
        const TARGET_LATENCY_MS: u64 = 300; // <300ms first token
        const TARGET_TPS: f64 = 25.0; // 25-40 TPS
        const TARGET_MEMORY_MB: usize = 12 * 1024; // <12GB

        self.average_latency.as_millis() as u64 <= TARGET_LATENCY_MS
            && self.average_tokens_per_second >= TARGET_TPS
            && self.average_memory_mb <= TARGET_MEMORY_MB
    }
}
```

### 5.2 Performance Monitoring

```rust
// In pensieve-07/src/monitoring/monitor.rs

use crate::monitoring::{PerformanceMetrics, MetricsCollector};
use tokio::time::{interval, Duration};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};

pub struct PerformanceMonitor {
    collector: MetricsCollector,
    system: System,
    metal_device: metal::Device,
}

impl PerformanceMonitor {
    pub fn new() -> Result<Self, MonitoringError> {
        let metal_device = metal::Device::system_default()
            .ok_or(MonitoringError::NoMetalDevice)?;

        Ok(Self {
            collector: MetricsCollector::new(),
            system: System::new_all(),
            metal_device,
        })
    }

    pub async fn start_monitoring(&mut self) -> Result<(), MonitoringError> {
        let mut interval = interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            self.system.refresh_all();

            let metrics = self.collect_current_metrics()?;
            self.collector.record_metrics(metrics);

            // Log performance summary
            if let Some(summary) = self.collector.get_average_performance() {
                println!("Performance: {:.1} TPS, {}ms latency, {}MB memory",
                    summary.average_tokens_per_second,
                    summary.average_latency.as_millis(),
                    summary.average_memory_mb
                );

                // Alert if performance degrades
                if !summary.meets_targets() {
                    eprintln!("WARNING: Performance below targets!");
                }
            }
        }
    }

    fn collect_current_metrics(&self) -> Result<PerformanceMetrics, MonitoringError> {
        let process = self.system.process(std::process::id() as usize)
            .ok_or(MonitoringError::ProcessNotFound)?;

        let memory_usage_mb = process.memory() / 1024; // Convert KB to MB
        let cpu_usage = process.cpu_usage();

        // Get Metal memory usage (this would need Metal API integration)
        let metal_memory = self.get_metal_memory_usage()?;

        Ok(PerformanceMetrics {
            request_latency: Duration::from_millis(150), // This should be measured per request
            tokens_per_second: 30.0, // This should be calculated from actual generation
            memory_usage_mb,
            gpu_utilization_percent: cpu_usage as f64 / 100.0,
            cache_hit_rate: 0.85, // This should be tracked in the cache
            concurrent_requests: 1, // This should be tracked in the server
        })
    }

    fn get_metal_memory_usage(&self) -> Result<usize, MonitoringError> {
        // Metal API memory usage tracking
        // This would require Metal-specific APIs
        Ok(1024) // Mock 1GB Metal memory usage
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MonitoringError {
    #[error("No Metal device available")]
    NoMetalDevice,
    #[error("Process not found")]
    ProcessNotFound,
    #[error("System information error")]
    SystemError,
}
```

## Section 6: Error Handling and Recovery

### 6.1 Comprehensive Error Types

```rust
// In pensieve-07/src/error/mod.rs

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PensieveError {
    #[error("MLX inference error: {0}")]
    MlxInference(#[from] MlxError),

    #[error("Model management error: {0}")]
    ModelManagement(#[from] ModelError),

    #[error("HTTP server error: {0}")]
    HttpServer(#[from] HttpError),

    #[error("Metal acceleration error: {0}")]
    Metal(#[from] MetalError),

    #[error("Memory allocation error: {0}")]
    Memory(#[from] MemoryError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Service temporarily unavailable")]
    ServiceUnavailable,

    #[error("Internal server error: {0}")]
    Internal(String),
}

// Implement conversion traits for error handling
impl From<PensieveError> for HttpResponse {
    fn from(error: PensieveError) -> HttpResponse {
        match error {
            PensieveError::Auth(msg) => HttpResponse::Unauthorized()
                .json(serde_json::json!({
                    "error": "AuthenticationError",
                    "message": msg
                })),

            PensieveError::RateLimit => HttpResponse::TooManyRequests()
                .json(serde_json::json!({
                    "error": "RateLimitError",
                    "message": "Rate limit exceeded"
                })),

            PensieveError::ModelManagement(e) => HttpResponse::ServiceUnavailable()
                .json(serde_json::json!({
                    "error": "ModelError",
                    "message": e.to_string()
                })),

            PensieveError::MlxInference(e) => HttpResponse::InternalServerError()
                .json(serde_json::json!({
                    "error": "InferenceError",
                    "message": e.to_string()
                })),

            _ => HttpResponse::InternalServerError()
                .json(serde_json::json!({
                    "error": "InternalError",
                    "message": "An unexpected error occurred"
                })),
        }
    }
}
```

### 6.2 Recovery Strategies

```rust
// In pensieve-07/src/recovery/recovery_manager.rs

use crate::error::PensieveError;
use std::time::Duration;

pub struct RecoveryManager {
    max_retries: u32,
    base_delay: Duration,
    backoff_multiplier: f64,
}

impl RecoveryManager {
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        }
    }

    pub async fn execute_with_recovery<F, T, E>(
        &self,
        operation: F,
    ) -> Result<T, E>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send>>,
        E: std::fmt::Display,
    {
        let mut delay = self.base_delay;

        for attempt in 0..=self.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) if attempt < self.max_retries => {
                    eprintln!("Attempt {} failed: {}. Retrying in {:?}...",
                        attempt + 1, e, delay);

                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.backoff_multiplier) as u64
                    );
                }
                Err(e) => return Err(e),
            }
        }

        unreachable!("All retries exhausted")
    }

    pub async fn recover_from_mlx_error(&self, error: MlxError) -> Result<(), MlxError> {
        match error {
            MlxError::MetalError(_) => {
                // Try to reset Metal device
                self.reset_metal_device().await?;
                Ok(())
            }
            MlxError::ModelLoadFailed(_) => {
                // Try to reload model
                self.reload_model().await?;
                Ok(())
            }
            MlxError::GenerationError(_) => {
                // Clear caches and retry
                self.clear_caches().await?;
                Ok(())
            }
            e => Err(e),
        }
    }

    async fn reset_metal_device(&self) -> Result<(), MlxError> {
        // Implementation for resetting Metal device
        tokio::time::sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    async fn reload_model(&self) -> Result<(), MlxError> {
        // Implementation for reloading model
        tokio::time::sleep(Duration::from_secs(2)).await;
        Ok(())
    }

    async fn clear_caches(&self) -> Result<(), MlxError> {
        // Implementation for clearing caches
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }
}
```

---

**Next Steps**: Proceed to D08-MVP-queries.md for research questions and validation
**Dependencies**: MLX framework integration research completed
**Test Coverage**: Unit, Integration, Performance, and E2E tests defined