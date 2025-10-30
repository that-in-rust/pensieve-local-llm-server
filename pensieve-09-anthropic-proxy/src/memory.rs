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
pub struct SystemMemoryMonitor {
    system: System,
}

impl SystemMemoryMonitor {
    /// Create new system memory monitor
    ///
    /// Initializes sysinfo System with all components.
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
        }
    }

    /// Create with custom refresh settings (for testing/optimization)
    pub fn new_minimal() -> Self {
        Self {
            system: System::new(),
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
        // Create new system and refresh memory info
        let mut system = System::new();
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
}
