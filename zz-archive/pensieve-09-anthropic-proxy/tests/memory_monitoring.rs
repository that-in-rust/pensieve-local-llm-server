//! Memory Monitoring Tests
//!
//! RED Phase: These tests should fail initially.
//! GREEN Phase: Implement src/memory.rs to make tests pass.
//! REFACTOR Phase: Cleanup and optimize implementation.

use pensieve_09_anthropic_proxy::memory::{MemoryMonitor, MemoryStatus};

/// Mock memory monitor for testing
struct MockMemoryMonitor {
    available_gb: f64,
}

impl MockMemoryMonitor {
    fn new(available_gb: f64) -> Self {
        Self { available_gb }
    }
}

impl MemoryMonitor for MockMemoryMonitor {
    fn check_status(&self) -> MemoryStatus {
        match self.available_gb {
            x if x > 3.0 => MemoryStatus::Safe,
            x if x > 2.0 => MemoryStatus::Caution,
            x if x > 1.0 => MemoryStatus::Warning,
            x if x > 0.5 => MemoryStatus::Critical,
            _ => MemoryStatus::Emergency,
        }
    }

    fn available_gb(&self) -> f64 {
        self.available_gb
    }
}

#[test]
fn test_memory_status_safe() {
    let monitor = MockMemoryMonitor::new(4.0);
    assert_eq!(monitor.check_status(), MemoryStatus::Safe);
}

#[test]
fn test_memory_status_caution() {
    let monitor = MockMemoryMonitor::new(2.5);
    assert_eq!(monitor.check_status(), MemoryStatus::Caution);
}

#[test]
fn test_memory_status_warning() {
    let monitor = MockMemoryMonitor::new(1.5);
    assert_eq!(monitor.check_status(), MemoryStatus::Warning);
}

#[test]
fn test_memory_status_critical() {
    let monitor = MockMemoryMonitor::new(0.7);
    assert_eq!(monitor.check_status(), MemoryStatus::Critical);
}

#[test]
fn test_memory_status_emergency() {
    let monitor = MockMemoryMonitor::new(0.3);
    assert_eq!(monitor.check_status(), MemoryStatus::Emergency);
}

#[test]
fn test_available_gb_returns_positive() {
    let monitor = MockMemoryMonitor::new(2.5);
    let available = monitor.available_gb();
    assert!(available > 0.0);
    assert_eq!(available, 2.5);
}

#[test]
fn test_memory_status_boundaries() {
    // Test exact boundaries
    assert_eq!(MockMemoryMonitor::new(3.0).check_status(), MemoryStatus::Caution);
    assert_eq!(MockMemoryMonitor::new(2.0).check_status(), MemoryStatus::Warning);
    assert_eq!(MockMemoryMonitor::new(1.0).check_status(), MemoryStatus::Critical);
    assert_eq!(MockMemoryMonitor::new(0.5).check_status(), MemoryStatus::Emergency);
}

#[test]
fn test_memory_status_is_copy() {
    // MemoryStatus should be Copy
    let status = MemoryStatus::Safe;
    let status2 = status;
    assert_eq!(status, status2);
}

#[test]
fn test_memory_status_debug() {
    // MemoryStatus should implement Debug
    let status = MemoryStatus::Safe;
    let debug_str = format!("{:?}", status);
    assert!(debug_str.contains("Safe"));
}

// Integration test with real system monitor (ignored by default)
#[test]
#[ignore]
fn test_system_memory_monitor_real() {
    use pensieve_09_anthropic_proxy::memory::SystemMemoryMonitor;

    let monitor = SystemMemoryMonitor::new();
    let available = monitor.available_gb();

    // Should have some memory available
    assert!(available > 0.0);

    // Status should be valid
    let status = monitor.check_status();
    // Can't assert specific status as it depends on system state
    // But should not panic
    let _ = format!("{:?}", status);
}

#[test]
#[ignore]
fn test_system_memory_monitor_performance() {
    use pensieve_09_anthropic_proxy::memory::SystemMemoryMonitor;
    use std::time::Instant;

    let monitor = SystemMemoryMonitor::new();

    // Check should complete in <10ms
    let start = Instant::now();
    let _ = monitor.check_status();
    let duration = start.elapsed();

    assert!(duration.as_millis() < 10, "Memory check took {}ms, should be <10ms", duration.as_millis());
}
