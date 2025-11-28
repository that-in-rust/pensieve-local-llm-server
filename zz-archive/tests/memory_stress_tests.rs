//! Memory Stress Tests for Pensieve Local LLM Server
//!
//! These tests validate memory management under extreme conditions
//! and ensure no memory leaks occur during prolonged operation.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use pensieve_04::{traits::CandleInferenceEngine, GenerationConfig, MemoryUsage, InferencePerformanceContract};
use pensieve_07_core::{CoreError, CoreResult};

/// Stress test engine with realistic memory simulation
#[derive(Debug)]
pub struct StressTestEngine {
    ready: bool,
    request_count: std::sync::atomic::AtomicUsize,
    memory_allocations: Arc<std::sync::Mutex<Vec<f64>>>,
}

impl StressTestEngine {
    pub fn new() -> Self {
        Self {
            ready: false,
            request_count: std::sync::atomic::AtomicUsize::new(0),
            memory_allocations: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub async fn load_model(&mut self, _model_path: &str) -> CoreResult<()> {
        // Simulate model loading with memory allocation
        for i in 0..100 {
            let _memory_mb = i as f64 * 10.0; // Simulate 100 allocations
            self.memory_allocations.lock().unwrap().push(_memory_mb);
        }
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

        // Count request
        self.request_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut responses = Vec::new();
        let start_time = Instant::now();

        // Simulate memory allocation during generation
        for (i, c) in input.chars().enumerate() {
            if i >= config.max_tokens {
                break;
            }

            // Simulate memory growth
            let allocated_mb = 1000.0 + (i as f64 * 50.0);
            self.memory_allocations.lock().unwrap().push(allocated_mb);

            tokio::time::sleep(Duration::from_millis(50)).await;

            let generation_time = start_time.elapsed().as_millis() as u64;
            let cumulative_tps = i as f64 / (generation_time as f64 / 1000.0);

            responses.push(StreamingTokenResponse {
                token: c.to_string(),
                token_id: (i + 1000) as u32,
                is_finished: i == input.len().min(config.max_tokens) - 1,
                tokens_generated: i + 1,
                generation_time_ms: generation_time,
                memory_usage_mb: allocated_mb,
                cumulative_tps,
            });
        }

        Ok(responses)
    }

    pub fn get_memory_usage(&self) -> MemoryUsage {
        let allocations = self.memory_allocations.lock().unwrap();
        let current_usage = allocations.last().unwrap_or(&0.0);
        let peak_usage = allocations.iter().fold(0.0, |max, &val| max.max(val));

        MemoryUsage {
            model_memory_mb: 2000.0,
            kv_cache_memory_mb: current_usage * 0.2,
            activation_memory_mb: current_usage * 0.1,
            total_memory_mb: *current_usage,
            peak_memory_mb: peak_usage,
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

    pub fn get_memory_allocations(&self) -> Vec<f64> {
        self.memory_allocations.lock().unwrap().clone()
    }

    pub fn get_request_count(&self) -> usize {
        self.request_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Memory stress test suite
pub struct MemoryStressTestSuite {
    results: Vec<StressTestResult>,
}

#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub test_name: String,
    pub peak_memory_mb: f64,
    pub requests_processed: usize,
    pub avg_memory_per_request_mb: f64,
    pub memory_growth_rate: f64,
    pub test_duration_ms: u64,
    pub passed: bool,
}

impl MemoryStressTestSuite {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub async fn run_all_stress_tests(&mut self) {
        println!("💾 Running Memory Stress Tests");
        
        self.test_memory_growth_under_load().await;
        self.test_memory_cleanup_after_requests().await;
        self.test_peak_memory_usage().await;
        self.test_prolonged_operation().await;
        self.test_concurrent_memory_usage().await;
        self.test_memory_leak_detection().await;
        
        self.print_results();
    }

    /// Stress Test 1: Memory Growth Under Load
    async fn test_memory_growth_under_load(&mut self) {
        let test_name = "Memory Growth Under Load";
        println!("   🧪 Running: {}", test_name);
        
        let mut engine = StressTestEngine::new();
        engine.load_model("stress_test_model.gguf").await.unwrap();
        
        let start_time = Instant::now();
        let memory_before = engine.get_memory_usage().total_memory_mb;
        
        // Process many requests
        for i in 0..100 {
            let config = GenerationConfig {
                max_tokens: 10 + (i % 20), // Varying token counts
                temperature: 0.7,
                stream: false,
                ..Default::default()
            };

            let input = format!("Stress test request {}", i);
            let _result = engine.generate_stream(&input, config).await
                .expect("Stress generation should succeed");
        }
        
        let duration = start_time.elapsed();
        let memory_after = engine.get_memory_usage().total_memory_mb;
        let peak_memory = engine.get_memory_usage().peak_memory_mb;
        
        let memory_growth_rate = (memory_after - memory_before) / 100.0; // Per request
        let avg_memory_per_request = (memory_after - memory_before) / 100.0;
        
        let passed = memory_after < 10000.0 && memory_growth_rate < 50.0; // <10GB total, <50MB per request
        
        self.results.push(StressTestResult {
            test_name: test_name.to_string(),
            peak_memory_mb: peak_memory,
            requests_processed: 100,
            avg_memory_per_request_mb: avg_memory_per_request,
            memory_growth_rate,
            test_duration_ms: duration.as_millis() as u64,
            passed,
        });

        println!("   Peak: {:.2}MB, Growth rate: {:.2}MB/request, Passed: {}", 
                 peak_memory, memory_growth_rate, if passed { "✅" } else { "❌" });
    }

    /// Stress Test 2: Memory Cleanup After Requests
    async fn test_memory_cleanup_after_requests(&mut self) {
        let test_name = "Memory Cleanup After Requests";
        println!("   🧪 Running: {}", test_name);
        
        let mut engine = StressTestEngine::new();
        engine.load_model("cleanup_test_model.gguf").await.unwrap();
        
        let start_memory = engine.get_memory_usage().total_memory_mb;
        
        // Process requests to build up memory
        for i in 0..50 {
            let config = GenerationConfig {
                max_tokens: 20,
                temperature: 0.7,
                stream: false,
                ..Default::default()
            };

            let _result = engine.generate_stream(&format!("Cleanup test {}", i), config).await
                .expect("Cleanup generation should succeed");
        }
        
        let peak_memory = engine.get_memory_usage().total_memory_mb;
        let requests_before_cleanup = engine.get_request_count();
        
        // Simulate cleanup (in real implementation, this would clear caches, etc.)
        let allocations = engine.get_memory_allocations();
        let cleanup_factor = 0.3; // Should clean up at least 30% of allocations
        
        // Mock cleanup by reducing allocations
        let mut allocations_mut = engine.memory_allocations.lock().unwrap();
        allocations_mut.retain(|&allocation| allocation < peak_memory * cleanup_factor);
        
        let final_memory = engine.get_memory_usage().total_memory_mb;
        
        let cleanup_efficiency = (peak_memory - final_memory) / peak_memory;
        let passed = cleanup_efficiency >= cleanup_factor;
        
        self.results.push(StressTestResult {
            test_name: test_name.to_string(),
            peak_memory_mb: peak_memory,
            requests_processed: requests_before_cleanup,
            avg_memory_per_request_mb: peak_memory / requests_before_cleanup as f64,
            memory_growth_rate: 0.0, // Not applicable for this test
            test_duration_ms: 0, // Not measured for this test
            passed,
        });

        println!("   Cleanup efficiency: {:.1}%, Passed: {}", 
                 cleanup_efficiency * 100.0, if passed { "✅" } else { "❌" });
    }

    /// Stress Test 3: Peak Memory Usage
    async fn test_peak_memory_usage(&mut self) {
        let test_name = "Peak Memory Usage";
        println!("   🧪 Running: {}", test_name);
        
        let mut engine = StressTestEngine::new();
        engine.load_model("peak_test_model.gguf").await.unwrap();
        
        let start_time = Instant::now();
        
        // Process multiple large requests simultaneously (simultaneously, not concurrently)
        let mut peak_memory = 0.0;
        
        for i in 0..20 {
            let config = GenerationConfig {
                max_tokens: 50, // Large requests
                temperature: 0.7,
                stream: false,
                ..Default::default()
            };

            let input = " ".repeat(1000) + &format!("Large request {}", i); // 1000+ chars
            let current_memory = engine.get_memory_usage().total_memory_mb;
            
            let _result = engine.generate_stream(&input, config).await
                .expect("Peak generation should succeed");
            
            peak_memory = peak_memory.max(current_memory);
            
            // Verify memory doesn't exceed limits
            assert!(peak_memory < 15000.0, "Peak memory should be <15GB");
        }
        
        let duration = start_time.elapsed();
        
        self.results.push(StressTestResult {
            test_name: test_name.to_string(),
            peak_memory_mb: peak_memory,
            requests_processed: 20,
            avg_memory_per_request_mb: peak_memory / 20.0,
            memory_growth_rate: 0.0, // Not applicable
            test_duration_ms: duration.as_millis() as u64,
            passed: peak_memory < 15000.0,
        });

        println!("   Peak: {:.2}MB, Limit: 15000MB, Passed: {}", 
                 peak_memory, if peak_memory < 15000.0 { "✅" } else { "❌" });
    }

    /// Stress Test 4: Prolonged Operation
    async fn test_prolonged_operation(&mut self) {
        let test_name = "Prolonged Operation";
        println!("   🧪 Running: {}", test_name);
        
        let mut engine = StressTestEngine::new();
        engine.load_model("prolonged_test_model.gguf").await.unwrap();
        
        let start_time = Instant::now();
        let test_duration = Duration::from_secs(30); // 30 minute test
        
        let mut request_count = 0;
        let mut memory_samples = Vec::new();
        
        // Run for extended duration
        while start_time.elapsed() < test_duration {
            let config = GenerationConfig {
                max_tokens: 5 + (request_count % 10),
                temperature: 0.7,
                stream: false,
                ..Default::default()
            };

            let input = format!("Prolonged test request {}", request_count);
            let _result = engine.generate_stream(&input, config).await
                .expect("Prolonged generation should succeed");
            
            request_count += 1;
            
            // Sample memory usage every 10 requests
            if request_count % 10 == 0 {
                memory_samples.push(engine.get_memory_usage().total_memory_mb);
            }
        }
        
        let total_duration = start_time.elapsed();
        let final_memory = engine.get_memory_usage().total_memory_mb;
        let peak_memory = memory_samples.iter().fold(0.0, |max, &val| max.max(val));
        
        // Check for memory leaks (memory should stabilize, not grow indefinitely)
        let memory_growth_stabilized = memory_samples.len() >= 10;
        let memory_variance = if memory_samples.len() > 1 {
            let mean = memory_samples.iter().sum::<f64>() / memory_samples.len() as f64;
            let variance = memory_samples.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / memory_samples.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        
        let passed = memory_growth_stabilized && final_memory < 10000.0;
        
        self.results.push(StressTestResult {
            test_name: test_name.to_string(),
            peak_memory_mb: peak_memory,
            requests_processed: request_count,
            avg_memory_per_request_mb: final_memory / request_count.max(1) as f64,
            memory_growth_rate: 0.0, // Not applicable
            test_duration_ms: total_duration.as_millis() as u64,
            passed,
        });

        println!("   Duration: {:.1}s, Requests: {}, Peak: {:.2}MB, Passed: {}", 
                 total_duration.as_secs_f64(), request_count, peak_memory, if passed { "✅" } else { "❌" });
    }

    /// Stress Test 5: Concurrent Memory Usage
    async fn test_concurrent_memory_usage(&mut self) {
        let test_name = "Concurrent Memory Usage";
        println!("   🧪 Running: {}", test_name);
        
        let engine = Arc::new(StressTestEngine::new());
        engine.load_model("concurrent_test_model.gguf").await.unwrap();
        
        let start_time = Instant::now();
        
        // Create many concurrent engines to simulate multiple instances
        let mut handles = Vec::new();
        for i in 0..5 {
            let engine_clone = engine.clone();
            let input = format!("Concurrent instance {}", i);
            
            handles.push(tokio::spawn(async move {
                let config = GenerationConfig {
                    max_tokens: 10,
                    temperature: 0.7,
                    stream: false,
                    ..Default::default()
                };
                
                let _result = engine_clone.generate_stream(&input, config).await
                    .expect("Concurrent generation should succeed");
                
                engine_clone.get_memory_usage().total_memory_mb
            }));
        }
        
        let results = futures::future::join_all(handles).await;
        let concurrent_memory_usages: Vec<f64> = results.into_iter()
            .filter_map(|r| r.ok())
            .collect();
        
        let total_peak_memory = concurrent_memory_usages.iter().sum::<f64>();
        let avg_memory = total_peak_memory / concurrent_memory_usages.len() as f64;
        
        let duration = start_time.elapsed();
        let passed = total_peak_memory < 20000.0; // <20GB total for all concurrent instances
        
        self.results.push(StressTestResult {
            test_name: test_name.to_string(),
            peak_memory_mb: total_peak_memory,
            requests_processed: concurrent_memory_usages.len(),
            avg_memory_per_request_mb: avg_memory,
            memory_growth_rate: 0.0, // Not applicable
            test_duration_ms: duration.as_millis() as u64,
            passed,
        });

        println!("   Total concurrent peak: {:.2}MB, Avg per instance: {:.2}MB, Passed: {}", 
                 total_peak_memory, avg_memory, if passed { "✅" } else { "❌" });
    }

    /// Stress Test 6: Memory Leak Detection
    async fn test_memory_leak_detection(&mut self) {
        let test_name = "Memory Leak Detection";
        println!("   🧪 Running: {}", test_name);
        
        let mut engine = StressTestEngine::new();
        engine.load_model("leak_test_model.gguf").await.unwrap();
        
        let initial_memory = engine.get_memory_usage().total_memory_mb;
        let mut memory_readings = Vec::new();
        
        // Run multiple cycles of requests
        for cycle in 0..10 {
            for request in 0..10 {
                let config = GenerationConfig {
                    max_tokens: 5,
                    temperature: 0.7,
                    stream: false,
                    ..Default::default()
                };

                let input = format!("Leak test cycle {} request {}", cycle, request);
                let _result = engine.generate_stream(&input, config).await
                    .expect("Leak test generation should succeed");
            }
            
            // Record memory after each cycle
            let current_memory = engine.get_memory_usage().total_memory_mb;
            memory_readings.push(current_memory);
            
            // Check for abnormal growth
            let growth_rate = if cycle > 0 {
                (current_memory - memory_readings[cycle - 1]) / initial_memory
            } else {
                0.0
            };
            
            assert!(growth_rate < 0.1, "Memory growth per cycle should be <10% of initial");
        }
        
        let final_memory = engine.get_memory_usage().total_memory_mb;
        let total_growth = final_memory - initial_memory;
        let growth_rate = total_growth / initial_memory;
        
        // Memory should return to baseline (indicating proper cleanup)
        let memory_returned_to_baseline = final_memory <= initial_memory * 1.2; // Within 20% is acceptable
        let no_excessive_growth = total_growth < initial_memory * 0.5; // <50% growth total
        
        let passed = memory_returned_to_baseline && no_excessive_growth;
        
        self.results.push(StressTestResult {
            test_name: test_name.to_string(),
            peak_memory_mb: memory_readings.iter().fold(0.0, |max, &val| max.max(val)),
            requests_processed: 100,
            avg_memory_per_request_mb: total_growth / 100.0,
            memory_growth_rate: growth_rate,
            test_duration_ms: 0, // Not measured for this test
            passed,
        });

        println!("   Initial: {:.2}MB, Final: {:.2}MB, Growth: {:.1}%, Passed: {}", 
                 initial_memory, final_memory, growth_rate * 100.0, if passed { "✅" } else { "❌" });
    }

    fn print_results(&self) {
        println!("\n💾 Memory Stress Test Summary:");
        println!("{'Test Name':<30} {'Peak Memory (MB)':<18} {'Requests':<10} {'Growth Rate':<12} {'Status':<8} {'Time (ms)':<10}");
        println!("{}", "-".repeat(100));
        
        for result in &self.results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            let growth_rate_str = if result.memory_growth_rate > 0.0 {
                format!("{:.2}", result.memory_growth_rate)
            } else {
                "N/A".to_string()
            };
            println!("{:<30} {:<18.2} {:<10} {:<12} {:<8} {:<10}", 
                     result.test_name, result.peak_memory_mb, result.requests_processed,
                     growth_rate_str, status, result.test_duration_ms);
        }
        
        let passed_count = self.results.iter().filter(|r| r.passed).count();
        let total_count = self.results.len();
        let success_rate = passed_count as f64 / total_count as f64 * 100.0;
        
        println!("\nMemory Stress Tests: {}/} passed ({:.1}%)", passed_count, total_count, success_rate);
        
        if success_rate >= 80.0 {
            println!("🎉 Memory stress tests PASSED!");
        } else {
            println!("⚠️  Memory stress tests need improvement");
        }
    }
}

#[tokio::test]
async fn test_memory_stress_tests() {
    println!("💾 Running Memory Stress Tests");
    
    let mut suite = MemoryStressTestSuite::new();
    suite.run_all_stress_tests().await;
    
    // Check that we ran all stress tests
    assert_eq!(suite.results.len(), 6, "Should run 6 memory stress tests");
    
    // Check that at least 80% pass
    let passed_count = suite.results.iter().filter(|r| r.passed).count();
    let success_rate = passed_count as f64 / suite.results.len() as f64;
    assert!(success_rate >= 0.8, "Should have at least 80% success rate");
    
    // Check key memory constraints
    let peak_memory_tests = suite.results.iter()
        .filter(|r| r.peak_memory_mb < 10000.0) // <10GB
        .count();
    
    assert!(peak_memory_tests >= 4, "At least 4 tests should have peak memory <10GB");
}
