//! Memory Monitoring Performance Benchmark
//!
//! Validates S01 Principle #5: Performance Claims Must Be Test-Validated
//!
//! Executable Specification:
//! GIVEN: Memory monitoring is enabled in production
//! WHEN: check_status() is called before each request
//! THEN: Overhead must be <5ms per check
//!
//! This benchmark measures the actual performance impact of memory safety
//! to ensure it doesn't degrade request latency beyond acceptable limits.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pensieve_09_anthropic_proxy::memory::{MemoryMonitor, SystemMemoryMonitor};
use std::time::Duration;

/// Benchmark: Single memory check operation
///
/// Target: <1ms for single check (to allow <5ms total overhead with safety margin)
fn bench_single_memory_check(c: &mut Criterion) {
    let monitor = SystemMemoryMonitor::new();

    c.bench_function("memory_check_single", |b| {
        b.iter(|| {
            let status = monitor.check_status();
            black_box(status);
        });
    });
}

/// Benchmark: Memory check with GB calculation
///
/// Target: <2ms (includes both status and available_gb)
fn bench_memory_check_with_gb(c: &mut Criterion) {
    let monitor = SystemMemoryMonitor::new();

    c.bench_function("memory_check_with_gb", |b| {
        b.iter(|| {
            let status = monitor.check_status();
            let available = monitor.available_gb();
            black_box((status, available));
        });
    });
}

/// Benchmark: Realistic request overhead simulation
///
/// Simulates checking memory before processing each request
/// Target: <5ms total overhead
fn bench_request_overhead(c: &mut Criterion) {
    let monitor = SystemMemoryMonitor::new();

    c.bench_function("request_memory_overhead", |b| {
        b.iter(|| {
            // Simulate pre-request check (what we do in handle_messages)
            let status = monitor.check_status();
            let available_gb = monitor.available_gb();

            // Simulate decision logic
            let should_process = !matches!(
                status,
                pensieve_09_anthropic_proxy::memory::MemoryStatus::Critical
                    | pensieve_09_anthropic_proxy::memory::MemoryStatus::Emergency
            );

            black_box((status, available_gb, should_process));
        });
    });
}

/// Benchmark: Bulk request simulation (100 checks)
///
/// Validates that repeated checks don't accumulate overhead
fn bench_bulk_requests(c: &mut Criterion) {
    let monitor = SystemMemoryMonitor::new();

    c.bench_function("bulk_100_checks", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let status = monitor.check_status();
                black_box(status);
            }
        });
    });
}

/// Benchmark: Memory check scaling
///
/// Tests performance at different request rates
fn bench_scaling(c: &mut Criterion) {
    let monitor = SystemMemoryMonitor::new();
    let mut group = c.benchmark_group("memory_check_scaling");

    for count in [1, 10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| {
                for _ in 0..count {
                    let status = monitor.check_status();
                    black_box(status);
                }
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = memory_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(1000);
    targets =
        bench_single_memory_check,
        bench_memory_check_with_gb,
        bench_request_overhead,
        bench_bulk_requests,
        bench_scaling
}

criterion_main!(memory_benches);
