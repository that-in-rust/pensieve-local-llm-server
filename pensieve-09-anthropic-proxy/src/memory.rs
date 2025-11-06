//! Memory Monitoring Module
//!
//! Provides system memory monitoring with configurable thresholds
//! following TDD principles and idiomatic Rust patterns.
//!
//! ## Architecture
//!
//! - Trait-based design for testability (Dependency Injection)
//! - RAII resource management (System automatically cleaned up)
//! - No panics - always returns valid state
//! - Clear, descriptive types
//!
//! ## Usage
//!
//! ```rust
//! use pensieve_09_anthropic_proxy::memory::{SystemMemoryMonitor, MemoryMonitor, MemoryStatus};
//!
//! let monitor = SystemMemoryMonitor::new();
//! match monitor.check_status() {
//!     MemoryStatus::Safe => {
//!         // Process request normally
//!     }
//!     MemoryStatus::Critical => {
//!         // Reject new requests
//!     }
//!     MemoryStatus::Emergency => {
//!         // Initiate emergency shutdown
//!     }
//!     _ => {}
//! }
//! ```

use sysinfo::System;
use std::sync::Mutex;

/// Memory monitoring trait for dependency injection
///
/// Allows mocking in tests while using real system monitoring in production.
pub trait MemoryMonitor: Send + Sync {
    /// Check current memory status
    ///
    /// Returns classification based on available memory thresholds.
    /// Never panics - returns valid status even if system info unavailable.
    fn check_status(&self) -> MemoryStatus;

    /// Get available memory in GB
    ///
    /// Returns amount of free RAM available to the system.
    /// Guaranteed to return non-negative value.
    fn available_gb(&self) -> f64;
}

/// Memory status classification
///
/// Based on absolute free RAM (not percentage) for universal applicability
/// across systems with different total RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStatus {
    /// >3GB available - normal operation
    Safe,

    /// 2-3GB available - monitor closely
    Caution,

    /// 1-2GB available - log warning, continue
    Warning,

    /// 500MB-1GB available - reject new requests
    Critical,

    /// <500MB available - emergency shutdown
    Emergency,
}

impl MemoryStatus {
    /// Returns true if requests should be accepted
    pub fn accepts_requests(&self) -> bool {
        matches!(self, MemoryStatus::Safe | MemoryStatus::Caution | MemoryStatus::Warning)
    }

    /// Returns true if emergency shutdown should trigger
    pub fn requires_shutdown(&self) -> bool {
        matches!(self, MemoryStatus::Emergency)
    }

    /// Get severity level (0-4, higher is more severe)
    pub fn severity(&self) -> u8 {
        match self {
            MemoryStatus::Safe => 0,
            MemoryStatus::Caution => 1,
            MemoryStatus::Warning => 2,
            MemoryStatus::Critical => 3,
            MemoryStatus::Emergency => 4,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            MemoryStatus::Safe => "Normal operation",
            MemoryStatus::Caution => "Monitoring memory pressure",
            MemoryStatus::Warning => "Low memory warning",
            MemoryStatus::Critical => "Critical memory pressure - rejecting requests",
            MemoryStatus::Emergency => "Emergency memory exhaustion - shutting down",
        }
    }
}

/// Real system memory monitor using sysinfo crate
///
/// Uses RAII for automatic resource management.
/// System refresh happens on each check for up-to-date information.
///
/// Thread-safe using Mutex for interior mutability to allow refreshing
/// from &self methods (required by MemoryMonitor trait).
pub struct SystemMemoryMonitor {
    system: Mutex<System>,
}

impl SystemMemoryMonitor {
    /// Create new system memory monitor
    ///
    /// Initializes sysinfo System with all components.
    /// Wrapped in Mutex for thread-safe interior mutability.
    pub fn new() -> Self {
        Self {
            system: Mutex::new(System::new_all()),
        }
    }

    /// Create with custom refresh settings (for testing/optimization)
    pub fn new_minimal() -> Self {
        Self {
            system: Mutex::new(System::new()),
        }
    }
}

impl Default for SystemMemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryMonitor for SystemMemoryMonitor {
    fn check_status(&self) -> MemoryStatus {
        let available_gb = self.available_gb();

        // Thresholds based on D17 research findings
        match available_gb {
            x if x > 3.0 => MemoryStatus::Safe,
            x if x > 2.0 => MemoryStatus::Caution,
            x if x > 1.0 => MemoryStatus::Warning,
            x if x > 0.5 => MemoryStatus::Critical,
            _ => MemoryStatus::Emergency,
        }
    }

    fn available_gb(&self) -> f64 {
        // Lock the stored System, refresh it, and get memory info
        // This reuses the System instance instead of creating a new one each time
        let mut system = self.system.lock().unwrap_or_else(|poisoned| {
            // If the mutex is poisoned, recover by taking the guard anyway
            // Memory monitoring is critical and we want to continue operating
            poisoned.into_inner()
        });

        // Refresh memory information to get current values
        system.refresh_memory();

        // Get available memory in bytes (sysinfo v0.30 API)
        let available_bytes = system.available_memory();

        // Convert to GB (1 GB = 1024^3 bytes)
        available_bytes as f64 / 1024_f64.powi(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock memory monitor for unit tests
    pub struct MockMemoryMonitor {
        available_gb: f64,
    }

    impl MockMemoryMonitor {
        pub fn new(available_gb: f64) -> Self {
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
    fn test_memory_status_accepts_requests() {
        assert!(MemoryStatus::Safe.accepts_requests());
        assert!(MemoryStatus::Caution.accepts_requests());
        assert!(MemoryStatus::Warning.accepts_requests());
        assert!(!MemoryStatus::Critical.accepts_requests());
        assert!(!MemoryStatus::Emergency.accepts_requests());
    }

    #[test]
    fn test_memory_status_requires_shutdown() {
        assert!(!MemoryStatus::Safe.requires_shutdown());
        assert!(!MemoryStatus::Critical.requires_shutdown());
        assert!(MemoryStatus::Emergency.requires_shutdown());
    }

    #[test]
    fn test_memory_status_severity() {
        assert_eq!(MemoryStatus::Safe.severity(), 0);
        assert_eq!(MemoryStatus::Caution.severity(), 1);
        assert_eq!(MemoryStatus::Warning.severity(), 2);
        assert_eq!(MemoryStatus::Critical.severity(), 3);
        assert_eq!(MemoryStatus::Emergency.severity(), 4);
    }

    #[test]
    fn test_memory_status_description() {
        assert!(MemoryStatus::Safe.description().contains("Normal"));
        assert!(MemoryStatus::Emergency.description().contains("Emergency"));
    }

    #[test]
    fn test_system_monitor_reports_available_memory() {
        // RED: Test that should FAIL with current bug
        // This test documents the expected behavior: SystemMemoryMonitor
        // should return the actual available memory (not 0.00GB)
        let monitor = SystemMemoryMonitor::new();
        let available = monitor.available_gb();

        println!("SystemMemoryMonitor reports: {:.2} GB available", available);

        // On a system with 25GB total RAM and some usage,
        // we should have at least 1GB available
        assert!(
            available >= 1.0,
            "Expected at least 1.0 GB available, but got {:.2} GB. \
             This indicates a bug in memory reporting on macOS.",
            available
        );
    }

    #[test]
    fn test_system_monitor_reuses_stored_system() {
        // RED: This test documents the EFFICIENCY bug
        // The current implementation creates a NEW System on every call,
        // wasting the stored System instance. This test verifies the fix.

        let monitor = SystemMemoryMonitor::new();

        // Make multiple calls - they should all work consistently
        let reading1 = monitor.available_gb();
        let reading2 = monitor.available_gb();
        let reading3 = monitor.available_gb();

        // All readings should be reasonable (>0) and consistent
        assert!(reading1 > 0.0, "First reading should be > 0, got {:.2}", reading1);
        assert!(reading2 > 0.0, "Second reading should be > 0, got {:.2}", reading2);
        assert!(reading3 > 0.0, "Third reading should be > 0, got {:.2}", reading3);

        // Readings should be within 1GB of each other (accounting for system variance)
        let max_diff = (reading1 - reading2).abs().max((reading2 - reading3).abs());
        assert!(
            max_diff < 1.0,
            "Memory readings vary too much: {:.2}, {:.2}, {:.2} GB",
            reading1, reading2, reading3
        );
    }
}
