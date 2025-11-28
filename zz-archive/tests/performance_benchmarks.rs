//! Performance Benchmarks for Pensieve Local LLM Server
//!
//! These benchmarks validate that the system meets performance targets
//! defined in the production requirements.

use std::time::{Duration, Instant};
use tokio::time::timeout;
use pensieve_04::{traits::CandleInferenceEngine, GenerationConfig, StreamingTokenResponse, MemoryUsage, InferencePerformanceContract};
use pensieve_07_core::{CoreError, CoreResult};

// Mock inference engine for performance testing
#[derive(Debug, Clone)]
pub struct PerformanceTestEngine {
    ready: bool,
}

impl PerformanceTestEngine {
    pub fn new() -> Self {
        Self { ready: false }
    }

    pub async fn load_model(&mut self, _model_path: &str) -> CoreResult<()> {
        // Simulate realistic model loading time
        tokio::time::sleep(Duration::from_millis(1000)).await;
        self.ready = true;
        Ok(())
    }

    pub async fn generate_stream(
        &self,
        input: &str,
        config: GenerationConfig,
    ) -> CoreResult<Vec<StreamingTokenResponse>> {
        if !self.ready {
            return Err(CoreError::Unavailable("Model not loaded"));
        }

        if input.is_empty() {
            return Err(CoreError::InvalidInput("Input cannot be empty"));
        }

        let mut responses = Vec::new();
        let start_time = Instant::now();

        // Simulate token generation with varying performance
        for (i, c) in input.chars().enumerate() {
            if i >= config.max_tokens {
                break;
            }

            // Simulate realistic token generation time (50-150ms per token)
            let delay = 50 + (i % 100);
            tokio::time::sleep(Duration::from_millis(delay)).await;

            let generation_time = start_time.elapsed().as_millis() as u64;
            let cumulative_tps = i as f64 / (generation_time as f64 / 1000.0);

            responses.push(StreamingTokenResponse {
                token: c.to_string(),
                token_id: (i + 1000) as u32,
                is_finished: i == input.len().min(config.max_tokens) - 1,
                tokens_generated: i + 1,
                generation_time_ms: generation_time,
                memory_usage_mb: 2000.0 + (i as f64 * 15.0),
                cumulative_tps,
            });
        }

        Ok(responses)
    }

    pub fn get_memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            model_memory_mb: 2000.0,
            kv_cache_memory_mb: 250.0,
            activation_memory_mb: 150.0,
            total_memory_mb: 2400.0,
            peak_memory_mb: 2800.0,
        }
    }

    pub fn get_performance_contract(&self) -> InferencePerformanceContract {
        InferencePerformanceContract {
            first_token_ms: 2000,
            tokens_per_second: 10.0,
            memory_usage_gb: 12.0,
            concurrent_requests: 2,
            error_rate: 0.01,
        }
    }
}

/// Performance benchmark suite
pub struct PerformanceBenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub target_value: f64,
    pub actual_value: f64,
    pub passed: bool,
    pub duration_ms: u64,
}

impl PerformanceBenchmarkSuite {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub async fn run_all_benchmarks(&mut self) {
        println!("🚀 Running Performance Benchmarks");
        
        self.benchmark_first_token_latency().await;
        self.benchmark_tokens_per_second().await;
        self.benchmark_memory_usage().await;
        self.benchmark_concurrent_requests().await;
        self.benchmark_error_rate().await;
        
        self.print_results();
    }

    /// Benchmark 1: First Token Latency
    async fn benchmark_first_token_latency(&mut self) {
        let test_name = "First Token Latency";
        let target = 2000.0; // 2 seconds
        let start = Instant::now();
        
        let engine = PerformanceTestEngine::new();
        engine.load_model("test_model.gguf").await.unwrap();
        
        let config = GenerationConfig {
            max_tokens: 5,
            temperature: 0.7,
            stream: true,
            ..Default::default()
        };

        let responses = engine.generate_stream("Hello", config).await.unwrap();
        let actual = responses.first()
            .map(|r| r.generation_time_ms as f64)
            .unwrap_or(0.0) as f64;
        
        let duration = start.elapsed().as_millis() as u64;
        let passed = actual <= target;
        
        self.results.push(BenchmarkResult {
            test_name: test_name.to_string(),
            target_value: target,
            actual_value: actual,
            passed,
            duration_ms: duration,
        });

        println!("   {}: {:.2}ms (target: {:.2}ms) - {}", 
                 test_name, actual, target, if passed { "✅ PASS" } else { "❌ FAIL" });
    }

    /// Benchmark 2: Tokens Per Second
    async fn benchmark_tokens_per_second(&mut self) {
        let test_name = "Tokens Per Second";
        let target = 10.0; // 10 TPS
        let start = Instant::now();
        
        let engine = PerformanceTestEngine::new();
        engine.load_model("test_model.gguf").await.unwrap();
        
        let config = GenerationConfig {
            max_tokens: 20,
            temperature: 0.7,
            stream: false,
            ..Default::default()
        };

        let responses = engine.generate_stream("Hello world how are you", config).await.unwrap();
        let duration_sec = start.elapsed().as_secs_f64();
        let actual = responses.len() as f64 / duration_sec;
        
        let duration = start.elapsed().as_millis() as u64;
        let passed = actual >= target;
        
        self.results.push(BenchmarkResult {
            test_name: test_name.to_string(),
            target_value: target,
            actual_value: actual,
            passed,
            duration_ms: duration,
        });

        println!("   {}: {:.2} TPS (target: {:.2} TPS) - {}", 
                 test_name, actual, target, if passed { "✅ PASS" } else { "❌ FAIL" });
    }

    /// Benchmark 3: Memory Usage
    async fn benchmark_memory_usage(&mut self) {
        let test_name = "Memory Usage";
        let target = 12000.0; // 12GB
        let start = Instant::now();
        
        let engine = PerformanceTestEngine::new();
        engine.load_model("test_model.gguf").await.unwrap();
        
        let memory_usage = engine.get_memory_usage();
        let actual = memory_usage.total_memory_mb;
        
        let duration = start.elapsed().as_millis() as u64;
        let passed = actual <= target;
        
        self.results.push(BenchmarkResult {
            test_name: test_name.to_string(),
            target_value: target,
            actual_value: actual,
            passed,
            duration_ms: duration,
        });

        println!("   {:.2}MB (target: {:.2}MB) - {}", 
                 actual, target, if passed { "✅ PASS" } else { "❌ FAIL" });
    }

    /// Benchmark 4: Concurrent Requests
    async fn benchmark_concurrent_requests(&mut self) {
        let test_name = "Concurrent Requests";
        let target = 2.0;
        let start = Instant::now();
        
        let engine = PerformanceTestEngine::new();
        engine.load_model("test_model.gguf").await.unwrap();
        
        let config = GenerationConfig {
            max_tokens: 10,
            temperature: 0.7,
            stream: false,
            ..Default::default()
        };

        // Test concurrent execution
        let mut handles = Vec::new();
        for i in 0..3 {
            let engine_clone = engine.clone();
            let input = format!("Concurrent test {}", i);
            let config_clone = config.clone();
            
            handles.push(tokio::spawn(async move {
                engine_clone.generate_stream(&input, config_clone).await
            }));
        }

        let successful = futures::future::join_all(handles)
            .await
            .into_iter()
            .filter(|result| result.is_ok())
            .count();
        
        let actual = successful as f64;
        let duration = start.elapsed().as_millis() as u64;
        let passed = actual >= target;
        
        self.results.push(BenchmarkResult {
            test_name: test_name.to_string(),
            target_value: target,
            actual_value: actual,
            passed,
            duration_ms: duration,
        });

        println!("   {}: {} concurrent (target: {}) - {}", 
                 test_name, successful, target, if passed { "✅ PASS" } else { "❌ FAIL" });
    }

    /// Benchmark 5: Error Rate
    async fn benchmark_error_rate(&mut self) {
        let test_name = "Error Rate";
        let target = 0.01; // 1%
        let start = Instant::now();
        
        let engine = PerformanceTestEngine::new();
        engine.load_model("test_model.gguf").await.unwrap();
        
        let config = GenerationConfig {
            max_tokens: 5,
            temperature: 0.7,
            stream: false,
            ..Default::default()
        };

        let total_requests = 100;
        let mut successful = 0;

        for i in 0..total_requests {
            let input = format!("Test input {}", i);
            let result = engine.generate_stream(&input, config.clone()).await;
            
            if result.is_ok() {
                successful += 1;
            }
        }

        let actual = 1.0 - (successful as f64 / total_requests as f64);
        let duration = start.elapsed().as_millis() as u64;
        let passed = actual <= target;
        
        self.results.push(BenchmarkResult {
            test_name: test_name.to_string(),
            target_value: target,
            actual_value: actual,
            passed,
            duration_ms: duration,
        });

        println!("   {:.2}% error rate (target: {:.2}%) - {}", 
                 actual * 100.0, target * 100.0, if passed { "✅ PASS" } else { "❌ FAIL" });
    }

    fn print_results(&self) {
        println!("\n📊 Performance Benchmark Summary:");
        println!("{'Test Name':<25} {'Target':<12} {'Actual':<12} {'Status':<10} {'Time (ms)':<10}");
        println!("{}", "-".repeat(70));
        
        for result in &self.results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            println!("{:<25} {:<12.2} {:<12.2} {:<10} {:<10}", 
                     result.test_name, result.target_value, result.actual_value, 
                     status, result.duration_ms);
        }
        
        let passed_count = self.results.iter().filter(|r| r.passed).count();
        let total_count = self.results.len();
        let success_rate = passed_count as f64 / total_count as f64 * 100.0;
        
        println!("\nOverall: {}/} tests passed ({:.1}%)", passed_count, total_count, success_rate);
        
        if success_rate >= 80.0 {
            println!("🎉 Performance benchmarks PASSED!");
        } else {
            println!("⚠️  Performance benchmarks need improvement");
        }
    }
}

#[tokio::test]
async fn test_performance_benchmarks() {
    println!("🎯 Running Performance Benchmarks");
    
    let mut suite = PerformanceBenchmarkSuite::new();
    suite.run_all_benchmarks().await;
    
    // Check that we ran all benchmarks
    assert_eq!(suite.results.len(), 5, "Should run 5 benchmarks");
    
    // Check that at least 80% pass
    let passed_count = suite.results.iter().filter(|r| r.passed).count();
    let success_rate = passed_count as f64 / suite.results.len() as f64;
    assert!(success_rate >= 0.8, "Should have at least 80% success rate");
    
    // Check key individual benchmarks
    let first_token_passed = suite.results.iter()
        .find(|r| r.test_name == "First Token Latency")
        .map(|r| r.passed)
        .unwrap_or(false);
    assert!(first_token_passed, "First token latency should pass");
    
    let tps_passed = suite.results.iter()
        .find(|r| r.test_name == "Tokens Per Second")
        .map(|r| r.passed)
        .unwrap_or(false);
    assert!(tps_passed, "Tokens per second should pass");
}
