# HTTP Server Performance Benchmarks
# Following parseltongue principles with performance validation

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use warp::test;

// Import our four-word functions from lib.rs
use p02_http_server_core::{
    create_http_routes_with_middleware,
    validate_api_key_from_header,
    parse_json_body_safely,
    traits::RequestHandler,
    Request, Response, ApiResponseError
};

// Mock handler for benchmarking
struct BenchmarkRequestHandler;

#[async_trait::async_trait]
impl RequestHandler for BenchmarkRequestHandler {
    async fn handle_message_request(&self, _request: Request) -> Result<Response, ApiResponseError> {
        // Minimal overhead for benchmarking
        Ok(Response {
            content: "Benchmark response".to_string(),
            model: "benchmark-model".to_string(),
            usage: Default::default(),
        })
    }
}

/// Benchmark API key validation performance
///
/// Measures the overhead of API key validation under various conditions
fn benchmark_api_key_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_key_validation");

    // Benchmark with valid key
    group.bench_function("valid_key", |b| {
        let key = "sk-ant-api03-test-key-123456";
        b.iter(|| {
            let result = validate_api_key_from_header(&Some(key.to_string()));
            criterion::black_box(result)
        })
    });

    // Benchmark with missing key
    group.bench_function("missing_key", |b| {
        b.iter(|| {
            let result = validate_api_key_from_header(&None);
            criterion::black_box(result)
        })
    });

    group.finish();
}

/// Benchmark JSON parsing performance
///
/// Measures JSON body parsing overhead for different payload sizes
fn benchmark_json_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_parsing");

    // Small payload benchmark
    let small_json = r#"{"model": "test", "max_tokens": 100}"#;
    group.bench_function("small_payload", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result: Result<serde_json::Value, ApiResponseError> = rt.block_on(
                parse_json_body_safely(small_json.as_bytes().into())
            );
            criterion::black_box(result)
        })
    });

    // Medium payload benchmark
    let medium_json = r#"{
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
    }"#;
    group.bench_function("medium_payload", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result: Result<serde_json::Value, ApiResponseError> = rt.block_on(
                parse_json_body_safely(medium_json.as_bytes().into())
            );
            criterion::black_box(result)
        })
    });

    group.finish();
}

/// Benchmark HTTP request handling
///
/// Measures end-to-end request processing performance
fn benchmark_http_request_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("http_request_handling");

    let handler = Arc::new(BenchmarkRequestHandler);
    let routes = create_http_routes_with_middleware(handler);

    // Health check endpoint benchmark
    group.bench_function("health_check", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
         .iter(|| async {
            let response = test::request()
                .method("GET")
                .path("/health")
                .reply(&routes)
                .await;
            criterion::black_box(response.status())
        })
    });

    // Message creation endpoint benchmark
    let message_json = r#"{
        "model": "phi-3-mini-128k-instruct-4bit",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }"#;

    group.bench_function("message_creation", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
         .iter(|| async {
            let response = test::request()
                .method("POST")
                .path("/v1/messages")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-ant-api03-test-key")
                .body(message_json)
                .reply(&routes)
                .await;
            criterion::black_box(response.status())
        })
    });

    group.finish();
}

/// Benchmark concurrent request handling
///
/// Measures performance under concurrent load
fn benchmark_concurrent_requests(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_requests");

    for concurrent_count in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_health_checks", concurrent_count),
            concurrent_count,
            |b, &count| {
                let handler = Arc::new(BenchmarkRequestHandler);
                let routes = create_http_routes_with_middleware(handler);

                b.to_async(tokio::runtime::Runtime::new().unwrap())
                 .iter(|| async {
                    let mut handles = vec![];
                    for _ in 0..count {
                        let routes_clone = routes.clone();
                        let handle = tokio::spawn(async move {
                            test::request()
                                .method("GET")
                                .path("/health")
                                .reply(&routes_clone)
                                .await
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        let response = handle.await.unwrap();
                        criterion::black_box(response.status());
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
///
/// Checks for memory leaks and allocation efficiency
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    group.bench_function("repeated_route_creation", |b| {
        b.iter(|| {
            let handler = Arc::new(BenchmarkRequestHandler);
            let _routes = create_http_routes_with_middleware(handler);
            criterion::black_box(())
        })
    });

    group.finish();
}

/// Benchmark response serialization
///
/// Measures JSON serialization overhead for responses
fn benchmark_response_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_serialization");

    let response = Response {
        content: "Hello, world!".to_string(),
        model: "phi-3-mini-128k-instruct-4bit".to_string(),
        usage: Default::default(),
    };

    group.bench_function("serialize_response", |b| {
        b.iter(|| {
            let result = serde_json::to_string(&response);
            criterion::black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_api_key_validation,
    benchmark_json_parsing,
    benchmark_http_request_handling,
    benchmark_concurrent_requests,
    benchmark_memory_patterns,
    benchmark_response_serialization
);
criterion_main!(benches);