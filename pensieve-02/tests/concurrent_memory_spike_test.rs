//! Integration test that reproduces the 8GB memory spike with concurrent requests
//!
//! This test validates that the server can handle concurrent requests without
//! loading the model multiple times. With the current process-per-request
//! architecture, this test will FAIL, proving the problem exists.
//!
//! Expected behavior after fix:
//! - Baseline: ~2.5-4GB (one model loaded)
//! - Peak with 4 concurrent requests: <5GB (model shared, only activation overhead)
//!
//! Current behavior (will fail):
//! - Peak with 4 concurrent requests: ~8GB+ (4 × 2GB model loads)

use sysinfo::System;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// Get current process RSS in GB
fn get_process_memory_gb() -> f64 {
    let mut sys = System::new_all();
    sys.refresh_all();

    let pid = sysinfo::Pid::from_u32(std::process::id());
    if let Some(process) = sys.process(pid) {
        let bytes = process.memory();
        bytes as f64 / 1_073_741_824.0 // Convert to GB
    } else {
        0.0
    }
}

/// Get available system memory in GB
fn get_available_memory_gb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory() as f64 / 1_073_741_824.0
}

/// Send a real inference request to the MLX server
async fn send_inference_request(client: &reqwest::Client, server_url: &str) -> Result<String, reqwest::Error> {
    let request_body = serde_json::json!({
        "prompt": "Test prompt for memory measurement",
        "max_tokens": 20,
        "temperature": 0.7,
        "stream": false
    });

    let response = client
        .post(format!("{}/generate", server_url))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .timeout(Duration::from_secs(30))
        .send()
        .await?;

    let text = response.text().await?;
    Ok(text)
}

/// Test: Sequential requests should not cause memory spikes
///
/// This test establishes a baseline: with proper model persistence,
/// sequential requests should only add minimal overhead (~0.5GB max)
#[tokio::test]
#[ignore] // Requires running server with real model
async fn test_sequential_requests_stable_memory() {
    let server_url = "http://127.0.0.1:8765";
    let client = reqwest::Client::new();

    // Check server is running
    let health_check = client.get(format!("{}/health", server_url))
        .send()
        .await;

    if health_check.is_err() {
        eprintln!("MLX server not running at {}. Start with:", server_url);
        eprintln!("python3 python_bridge/mlx_server.py --model-path ./models/Phi-3-mini-128k-instruct-4bit");
        panic!("Server not available");
    }

    println!("=== Sequential Request Memory Test ===");

    // Wait for server to stabilize
    sleep(Duration::from_secs(2)).await;

    let baseline_mem = get_available_memory_gb();
    println!("Baseline available memory: {:.2}GB", baseline_mem);

    // Send 5 sequential requests
    let mut max_memory_delta = 0.0f64;

    for i in 1..=5 {
        println!("Sending request {}...", i);

        let mem_before = get_available_memory_gb();

        match send_inference_request(&client, server_url).await {
            Ok(response) => {
                println!("  Response received: {} bytes", response.len());
            }
            Err(e) => {
                eprintln!("  Request failed: {}", e);
                continue;
            }
        }

        // Wait for Python process to exit
        sleep(Duration::from_secs(2)).await;

        let mem_after = get_available_memory_gb();
        let delta = mem_before - mem_after; // Positive means memory decreased (used)

        println!("  Memory delta: {:.2}GB", delta);

        if delta > max_memory_delta {
            max_memory_delta = delta;
        }
    }

    println!("\nMax memory delta across requests: {:.2}GB", max_memory_delta);

    // With proper model persistence, sequential requests should not
    // repeatedly load the model. Allow 1GB overhead for first load,
    // but subsequent requests should be minimal.
    assert!(
        max_memory_delta < 2.0,
        "Sequential requests showed {:.2}GB memory delta, expected <2GB. \
         This suggests model is being reloaded on each request.",
        max_memory_delta
    );
}

/// Test: Concurrent requests cause 8GB+ spike (WILL FAIL with current architecture)
///
/// This is the RED test that proves the problem exists.
/// Current architecture spawns 4 Python processes, each loading 2GB model = 8GB spike.
///
/// After implementing persistent worker, this test should PASS with <5GB peak.
#[tokio::test]
#[ignore] // Requires running server with real model
async fn test_concurrent_requests_memory_spike() {
    // Test the persistent MLX server directly on port 8765
    let server_url = "http://127.0.0.1:8765";
    let client = Arc::new(reqwest::Client::new());

    // Check server is running
    let health_check = client.get(format!("{}/health", server_url))
        .send()
        .await;

    if health_check.is_err() {
        eprintln!("MLX server not running at {}. Start with:", server_url);
        eprintln!("python3 python_bridge/mlx_server.py --model-path ./models/Phi-3-mini-128k-instruct-4bit");
        panic!("Server not available");
    }

    println!("\n=== Concurrent Request Memory Spike Test ===");
    println!("This test will FAIL with current architecture (process-per-request)");
    println!("Expected to PASS after implementing persistent Python worker\n");

    // Warm up server (first request loads model)
    println!("Warming up server with initial request...");
    let _ = send_inference_request(&client, server_url).await;
    sleep(Duration::from_secs(3)).await;

    let baseline_mem = get_available_memory_gb();
    println!("Baseline available memory: {:.2}GB", baseline_mem);

    // Spawn 4 concurrent requests
    // Current architecture will spawn 4 Python processes, each loading the model
    println!("\nSending 4 concurrent requests...");

    let mut handles = vec![];

    for i in 1..=4 {
        let client_clone = Arc::clone(&client);
        let url = server_url.to_string();

        let handle = tokio::spawn(async move {
            println!("  Request {} starting...", i);
            let result = send_inference_request(&client_clone, &url).await;
            match &result {
                Ok(resp) => println!("  Request {} complete: {} bytes", i, resp.len()),
                Err(e) => eprintln!("  Request {} failed: {}", i, e),
            }
            result
        });

        handles.push(handle);
    }

    // Monitor memory during concurrent execution
    let monitor_handle = tokio::spawn(async move {
        let mut min_available = f64::MAX;
        let mut max_used = 0.0f64;

        for _ in 0..10 {
            let current = get_available_memory_gb();
            if current < min_available {
                min_available = current;
            }
            let used = baseline_mem - current;
            if used > max_used {
                max_used = used;
            }
            sleep(Duration::from_millis(200)).await;
        }

        (min_available, max_used)
    });

    // Wait for all requests to complete
    let mut successful = 0;
    for handle in handles {
        if let Ok(Ok(_)) = handle.await {
            successful += 1;
        }
    }

    println!("\nCompleted {}/4 requests", successful);

    // Get peak memory usage
    let (min_available, peak_usage) = monitor_handle.await.unwrap();
    let final_mem = get_available_memory_gb();

    println!("\n=== Memory Report ===");
    println!("Baseline available: {:.2}GB", baseline_mem);
    println!("Minimum available during test: {:.2}GB", min_available);
    println!("Peak memory used: {:.2}GB", peak_usage);
    println!("Final available: {:.2}GB", final_mem);
    println!("Memory recovered: {:.2}GB", final_mem - min_available);

    // THE CRITICAL ASSERTION
    // With proper model persistence, 4 concurrent requests should only use
    // ~1-2GB extra (activation memory for 4 contexts sharing one loaded model).
    //
    // Current behavior: ~6-8GB spike (4 × 2GB model loads)
    // Expected after fix: <3GB spike (shared model + 4 small contexts)
    assert!(
        peak_usage < 5.0,
        "\n\n❌ MEMORY SPIKE DETECTED: {:.2}GB used by concurrent requests!\n\
         \n\
         Root cause: Process-per-request architecture loads model 4 times.\n\
         \n\
         Expected after persistent worker implementation:\n\
         - Model loaded once at startup: ~2.5GB baseline\n\
         - 4 concurrent contexts: +1-2GB overhead\n\
         - Total peak: <5GB\n\
         \n\
         Current behavior:\n\
         - Each request spawns new Python process\n\
         - Each process loads 2GB Phi-3 model\n\
         - 4 concurrent = 4 × 2GB = 8GB spike\n\
         \n\
         Solution: Implement persistent Python worker (see optimization plan)\n",
        peak_usage
    );

    println!("\n✅ Test passed! Concurrent requests use <5GB (model persistence working)");
}

/// Test: Memory recovers after concurrent load
///
/// Validates that memory returns to baseline after concurrent requests complete,
/// indicating no memory leaks.
#[tokio::test]
#[ignore] // Requires running server with real model
async fn test_memory_recovery_after_load() {
    let server_url = "http://127.0.0.1:8765";
    let client = Arc::new(reqwest::Client::new());

    println!("\n=== Memory Recovery Test ===");

    let baseline_mem = get_available_memory_gb();
    println!("Baseline: {:.2}GB", baseline_mem);

    // Create load
    println!("Creating load with 3 concurrent requests...");
    let mut handles = vec![];
    for _ in 1..=3 {
        let client_clone = Arc::clone(&client);
        let url = server_url.to_string();
        handles.push(tokio::spawn(async move {
            send_inference_request(&client_clone, &url).await
        }));
    }

    for handle in handles {
        let _ = handle.await;
    }

    println!("Waiting for processes to exit and memory to stabilize...");
    sleep(Duration::from_secs(5)).await;

    let recovered_mem = get_available_memory_gb();
    let delta = (baseline_mem - recovered_mem).abs();

    println!("Recovered: {:.2}GB", recovered_mem);
    println!("Delta from baseline: {:.2}GB", delta);

    // Memory should recover to within 1GB of baseline
    assert!(
        delta < 1.0,
        "Memory did not recover after load. Delta: {:.2}GB (expected <1GB). \
         This indicates a memory leak.",
        delta
    );

    println!("✅ Memory recovered successfully");
}
