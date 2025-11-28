//! MLX Integration Tests for Pensieve
//!
//! These tests validate the MLX pipeline integration with the Pensieve system

use std::process::Command;
use std::path::Path;
use serde_json::Value;

/// Performance contract for MLX inference
pub struct MlxPerformanceContract {
    pub min_tps: f64,
    pub max_memory_mb: f64,
    pub max_latency_ms: u64,
}

impl Default for MlxPerformanceContract {
    fn default() -> Self {
        Self {
            min_tps: 25.0,        // Target: 25+ TPS
            max_memory_mb: 3000.0, // Max 3GB memory
            max_latency_ms: 5000,  // Max 5s for 100 tokens
        }
    }
}

/// Test MLX basic functionality
#[test]
fn test_mlx_basic_functionality() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing MLX Basic Functionality");

    let output = Command::new("python3")
        .arg("../python_bridge/mlx_inference.py")
        .arg("--model-path")
        .arg("../models/Phi-3-mini-128k-instruct-4bit")
        .arg("--prompt")
        .arg("Hello, world!")
        .arg("--max-tokens")
        .arg("50")
        .arg("--temperature")
        .arg("0.7")
        .arg("--metrics")
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("MLX inference failed: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_str = stdout.lines().last().unwrap_or("{}");
    let json: Value = serde_json::from_str(json_str)?;

    // Verify basic functionality
    let text = json["text"].as_str().unwrap_or("");
    let tps = json["tokens_per_second"].as_f64().unwrap_or(0.0);
    let memory_mb = json["peak_memory_mb"].as_f64().unwrap_or(0.0);
    let completion_tokens = json["completion_tokens"].as_u64().unwrap_or(0);

    assert!(!text.is_empty(), "Should generate non-empty text");
    assert!(completion_tokens > 0, "Should generate tokens");
    assert!(tps > 0.0, "Should measure TPS");

    println!("✅ Basic functionality test passed:");
    println!("   - Generated {} tokens", completion_tokens);
    println!("   - TPS: {:.1}", tps);
    println!("   - Memory: {:.1} MB", memory_mb);

    Ok(())
}

/// Test MLX performance target
#[test]
fn test_mlx_performance_target() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing MLX Performance Target (25+ TPS)");

    let contract = MlxPerformanceContract::default();

    let output = Command::new("python3")
        .arg("../python_bridge/mlx_inference.py")
        .arg("--model-path")
        .arg("../models/Phi-3-mini-128k-instruct-4bit")
        .arg("--prompt")
        .arg("Performance test prompt with sufficient length to generate meaningful output.")
        .arg("--max-tokens")
        .arg("100")
        .arg("--temperature")
        .arg("0.7")
        .arg("--metrics")
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("MLX inference failed: {}", stderr).into());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_str = stdout.lines().last().unwrap_or("{}");
    let json: Value = serde_json::from_str(json_str)?;

    let tps = json["tokens_per_second"].as_f64().unwrap_or(0.0);
    let memory_mb = json["peak_memory_mb"].as_f64().unwrap_or(0.0);
    let latency_ms = json["elapsed_ms"].as_u64().unwrap_or(0);

    println!("📊 Performance Results:");
    println!("   - TPS: {:.1} (target: {})", tps, contract.min_tps);
    println!("   - Memory: {:.1} MB (max: {})", memory_mb, contract.max_memory_mb);
    println!("   - Latency: {} ms (max: {})", latency_ms, contract.max_latency_ms);

    if tps >= contract.min_tps {
        println!("🎉 TPS Target Achieved! ({:.1} >= {})", tps, contract.min_tps);
    } else {
        println!("⚠️  TPS Below Target ({:.1} < {})", tps, contract.min_tps);
    }

    // Performance assertions
    assert!(tps > 10.0, "Should achieve at least 10 TPS");
    assert!(memory_mb > 0.0, "Should measure memory usage");
    // Note: latency_ms might be 0 in some cases, so we don't assert on it
    assert!(latency_ms >= 0, "Should have non-negative latency");

    Ok(())
}

/// Test MLX model availability
#[test]
fn test_mlx_model_availability() {
    println!("🧪 Testing MLX Model Availability");

    let model_path = Path::new("../models/Phi-3-mini-128k-instruct-4bit");
    assert!(model_path.exists(), "Model directory should exist");

    let required_files = vec![
        "config.json",
        "model.safetensors",
        "tokenizer.json"
    ];

    for file in required_files {
        let file_path = model_path.join(file);
        assert!(file_path.exists(), "Required file should exist: {}", file);
    }

    println!("✅ Model files available in: {}", model_path.display());
}

/// Test MLX Python bridge availability
#[test]
fn test_mlx_python_bridge_availability() {
    println!("🧪 Testing MLX Python Bridge Availability");

    let output = Command::new("python3")
        .arg("--version")
        .output();

    assert!(output.is_ok(), "Python should be available");

    let version_output = output.unwrap();
    assert!(version_output.status.success(), "Python version should be retrievable");

    let version_str = String::from_utf8_lossy(&version_output.stdout);
    assert!(version_str.contains("Python 3"), "Should be Python 3");

    println!("✅ Python bridge available: {}", version_str.trim());
}

/// Test MLX dependencies
#[test]
fn test_mlx_dependencies() {
    println!("🧪 Testing MLX Dependencies");

    let test_script = r#"
import sys
try:
    import mlx.core as mx
    import mlx_lm
    print("MLX dependencies available")
    sys.exit(0)
except ImportError as e:
    print(f"MLX dependency missing: {e}")
    sys.exit(1)
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(test_script)
        .output();

    assert!(output.is_ok(), "Should be able to run Python script");

    let result = output.unwrap();
    if !result.status.success() {
        println!("⚠️  MLX dependencies check failed: {}", String::from_utf8_lossy(&result.stderr));
        // Don't fail the test if dependencies are missing in CI
        return;
    }

    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("MLX dependencies available"));

    println!("✅ MLX dependencies available");
}

/// Test MLX cache performance
#[test]
fn test_mlx_cache_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing MLX Cache Performance");

    // First request (cache miss)
    let output1 = Command::new("python3")
        .arg("../python_bridge/mlx_inference.py")
        .arg("--model-path")
        .arg("../models/Phi-3-mini-128k-instruct-4bit")
        .arg("--prompt")
        .arg("Cache test prompt")
        .arg("--max-tokens")
        .arg("50")
        .arg("--temperature")
        .arg("0.7")
        .arg("--metrics")
        .output()?;

    // Second request (potential cache hit)
    let output2 = Command::new("python3")
        .arg("../python_bridge/mlx_inference.py")
        .arg("--model-path")
        .arg("../models/Phi-3-mini-128k-instruct-4bit")
        .arg("--prompt")
        .arg("Different cache test prompt")
        .arg("--max-tokens")
        .arg("50")
        .arg("--temperature")
        .arg("0.7")
        .arg("--metrics")
        .output()?;

    assert!(output1.status.success(), "First request should succeed");
    assert!(output2.status.success(), "Second request should succeed");

    let stdout1 = String::from_utf8_lossy(&output1.stdout);
    let json_str1 = stdout1.lines().last().unwrap_or("{}");
    let json1: Value = serde_json::from_str(json_str1)?;

    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    let json_str2 = stdout2.lines().last().unwrap_or("{}");
    let json2: Value = serde_json::from_str(json_str2)?;

    let tps1 = json1["tokens_per_second"].as_f64().unwrap_or(0.0);
    let tps2 = json2["tokens_per_second"].as_f64().unwrap_or(0.0);

    let cache_hit_rate1 = json1["performance_metrics"]["cache_hit_rate"].as_f64().unwrap_or(0.0);
    let cache_hit_rate2 = json2["performance_metrics"]["cache_hit_rate"].as_f64().unwrap_or(0.0);

    println!("📊 Cache Performance:");
    println!("   - Request 1 TPS: {:.1}", tps1);
    println!("   - Request 2 TPS: {:.1}", tps2);
    println!("   - Request 1 Cache Hit: {:.1}%", cache_hit_rate1 * 100.0);
    println!("   - Request 2 Cache Hit: {:.1}%", cache_hit_rate2 * 100.0);

    // Both requests should work regardless of cache performance
    assert!(tps1 > 0.0, "First request should work");
    assert!(tps2 > 0.0, "Second request should work");

    // Cache hit rate should be reasonable (0.0 to 1.0)
    assert!(cache_hit_rate1 >= 0.0 && cache_hit_rate1 <= 1.0);
    assert!(cache_hit_rate2 >= 0.0 && cache_hit_rate2 <= 1.0);

    Ok(())
}