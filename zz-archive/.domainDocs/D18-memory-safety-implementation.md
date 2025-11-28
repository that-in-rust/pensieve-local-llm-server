# D18: Memory Safety Implementation - TDD Approach

**Date**: 2025-10-30
**Status**: Implementation In Progress
**Methodology**: TDD (RED → GREEN → REFACTOR)
**Follows**: `.steeringDocs/S01-README-MOSTIMP.md` principles

---

## Executive Summary

This document outlines the implementation of memory safety mechanisms for Pensieve, following Test-Driven Development and idiomatic Rust patterns. Based on research in D17, we implement:

1. **Memory monitoring** with absolute free RAM thresholds
2. **Request rejection** at critical memory levels
3. **Emergency shutdown** when memory exhausted
4. **MLX cache management** after each inference
5. **Multi-instance isolation** via wrapper scripts

## Architecture Principles Applied

### 1. Executable Specifications Over Narratives
Every feature is specified as testable contracts with preconditions, postconditions, and error conditions.

### 2. Layered Rust Architecture (L3)
Memory monitoring lives in `pensieve-09-anthropic-proxy` (L3 - Application Layer).

### 3. Dependency Injection for Testability
`MemoryMonitor` trait allows mocking in tests.

### 4. RAII Resource Management
Emergency shutdown uses Drop trait for cleanup.

### 5. Structured Error Handling
`thiserror` for library errors, clear error types.

---

## Executable Specifications

### Specification 1: Memory Status Detection

**Contract**:
```rust
trait MemoryMonitor {
    /// Returns current memory status
    ///
    /// Postconditions:
    /// - Returns MemoryStatus enum
    /// - Never panics
    /// - Completes in <10ms
    fn check_status(&self) -> MemoryStatus;

    /// Returns available memory in GB
    ///
    /// Postconditions:
    /// - Returns f64 >= 0.0
    /// - Accurate within 100MB
    fn available_gb(&self) -> f64;
}

enum MemoryStatus {
    Safe,      // > 3GB free
    Caution,   // 2-3GB free
    Warning,   // 1-2GB free
    Critical,  // 500MB-1GB free
    Emergency, // < 500MB free
}
```

**Test Cases**:
- ✅ Returns `Safe` when >3GB available
- ✅ Returns `Warning` when 1-2GB available
- ✅ Returns `Critical` when 500MB-1GB available
- ✅ Returns `Emergency` when <500MB available
- ✅ Completes in <10ms
- ✅ Never panics even if sysinfo fails

### Specification 2: Request Rejection

**Contract**:
```rust
/// Request handler with memory checking
///
/// Preconditions:
/// - Valid CreateMessageRequest
/// - Server running
///
/// Postconditions:
/// - If memory >= Warning: Process request normally
/// - If memory == Critical: Return 503 with warning
/// - If memory == Emergency: Reject immediately with 503
/// - All responses include X-Memory-Status header
async fn handle_messages_with_memory_check(
    req: CreateMessageRequest,
    monitor: &dyn MemoryMonitor,
) -> Result<Response, ServerError>;
```

**Test Cases**:
- ✅ Accepts request when memory is Safe/Caution
- ✅ Logs warning but accepts when memory is Warning
- ✅ Rejects with 503 when memory is Critical
- ✅ Rejects immediately when memory is Emergency
- ✅ Response includes `X-Memory-Status` header
- ✅ Response includes `X-Available-Memory-GB` header

### Specification 3: Emergency Shutdown

**Contract**:
```rust
/// Emergency shutdown handler
///
/// Preconditions:
/// - Memory < 1GB available
/// - Server is running
///
/// Postconditions:
/// - Logs emergency state
/// - Clears MLX cache (via Python bridge)
/// - Waits max 5s for in-flight requests
/// - Saves state to emergency log
/// - Exits with code 1
struct EmergencyShutdown {
    timeout_secs: u64,
}

impl EmergencyShutdown {
    fn initiate(&self, reason: &str) -> !;
}
```

**Test Cases**:
- ✅ Triggers when available memory < 1GB
- ✅ Logs emergency reason
- ✅ Attempts cache clear (best effort)
- ✅ Times out after 5 seconds
- ✅ Exits with code 1
- ✅ Creates emergency log file

### Specification 4: MLX Cache Management

**Contract**:
```python
# In python_bridge/mlx_inference.py

def generate_with_cleanup(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    **kwargs
) -> dict:
    """
    Generate text with automatic cache cleanup.

    Postconditions:
    - Calls mx.metal.clear_cache() after generation
    - Returns generated text
    - Clears cache even if generation fails
    - Logs memory before/after
    """
```

**Test Cases**:
- ✅ Clears cache after successful generation
- ✅ Clears cache after failed generation
- ✅ Logs memory stats before/after
- ✅ Memory usage returns to baseline

### Specification 5: Multi-Instance Isolation

**Contract**:
```bash
#!/bin/bash
# claude-local wrapper script
#
# Preconditions:
# - Pensieve server running on specified port
# - Claude Code CLI installed
#
# Postconditions:
# - Sets ANTHROPIC_BASE_URL for this process only
# - Sets ANTHROPIC_API_KEY for this process only
# - Does not modify ~/.claude/settings.json
# - Other terminals unaffected
# - Exits with claude's exit code
```

**Test Cases**:
- ✅ Terminal A uses local server
- ✅ Terminal B uses Anthropic API
- ✅ No interference between terminals
- ✅ Settings file unchanged
- ✅ Exit codes propagated correctly

---

## Implementation Plan (TDD Cycle)

### Phase 1: Memory Monitoring (Week 1)

#### RED: Write Failing Tests
```rust
// pensieve-09-anthropic-proxy/tests/memory_monitoring.rs

#[test]
fn test_memory_status_safe() {
    let monitor = MockMemoryMonitor::new(4.0); // 4GB free
    assert_eq!(monitor.check_status(), MemoryStatus::Safe);
}

#[test]
fn test_memory_status_warning() {
    let monitor = MockMemoryMonitor::new(1.5); // 1.5GB free
    assert_eq!(monitor.check_status(), MemoryStatus::Warning);
}

#[test]
fn test_memory_status_critical() {
    let monitor = MockMemoryMonitor::new(0.7); // 700MB free
    assert_eq!(monitor.check_status(), MemoryStatus::Critical);
}

#[test]
fn test_request_rejected_at_critical() {
    let monitor = MockMemoryMonitor::new(0.7);
    let result = handle_request_with_memory(&monitor, test_request());
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().status_code(), 503);
}
```

#### GREEN: Implement to Pass Tests
```rust
// pensieve-09-anthropic-proxy/src/memory.rs

use sysinfo::{System, SystemExt};

pub trait MemoryMonitor: Send + Sync {
    fn check_status(&self) -> MemoryStatus;
    fn available_gb(&self) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStatus {
    Safe,      // > 3GB
    Caution,   // 2-3GB
    Warning,   // 1-2GB
    Critical,  // 0.5-1GB
    Emergency, // < 0.5GB
}

pub struct SystemMemoryMonitor {
    system: System,
}

impl SystemMemoryMonitor {
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
        }
    }
}

impl MemoryMonitor for SystemMemoryMonitor {
    fn check_status(&self) -> MemoryStatus {
        let available_gb = self.available_gb();

        match available_gb {
            x if x > 3.0 => MemoryStatus::Safe,
            x if x > 2.0 => MemoryStatus::Caution,
            x if x > 1.0 => MemoryStatus::Warning,
            x if x > 0.5 => MemoryStatus::Critical,
            _ => MemoryStatus::Emergency,
        }
    }

    fn available_gb(&self) -> f64 {
        self.system.refresh_memory();
        let available_bytes = self.system.available_memory();
        available_bytes as f64 / 1024_f64.powi(3)
    }
}

// Mock for testing
#[cfg(test)]
pub struct MockMemoryMonitor {
    available_gb: f64,
}

#[cfg(test)]
impl MockMemoryMonitor {
    pub fn new(available_gb: f64) -> Self {
        Self { available_gb }
    }
}

#[cfg(test)]
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
```

#### REFACTOR: Idiomatic Rust
- Use `sysinfo` crate (already in ecosystem)
- Trait-based design for testability
- Mock implementation for tests
- Clear, descriptive names
- No panics, always returns valid state

### Phase 2: Request Rejection (Week 1)

#### RED: Write Failing Tests
```rust
#[tokio::test]
async fn test_request_handling_with_safe_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(4.0));
    let config = ServerConfig::default();
    let result = handle_messages_with_memory_check(
        test_request(),
        monitor,
        config,
    ).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_request_rejected_at_critical_memory() {
    let monitor = Arc::new(MockMemoryMonitor::new(0.7));
    let config = ServerConfig::default();
    let result = handle_messages_with_memory_check(
        test_request(),
        monitor,
        config,
    ).await;

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert_eq!(error.status_code(), 503);
    assert!(error.message().contains("memory"));
}

#[tokio::test]
async fn test_response_includes_memory_headers() {
    let monitor = Arc::new(MockMemoryMonitor::new(2.5));
    let config = ServerConfig::default();
    let response = handle_messages_with_memory_check(
        test_request(),
        monitor,
        config,
    ).await.unwrap();

    let headers = response.headers();
    assert!(headers.contains_key("X-Memory-Status"));
    assert!(headers.contains_key("X-Available-Memory-GB"));
}
```

#### GREEN: Implement Request Checking
```rust
// In pensieve-09-anthropic-proxy/src/server.rs

use crate::memory::{MemoryMonitor, MemoryStatus};

pub async fn handle_messages_with_memory_check(
    req: CreateMessageRequest,
    monitor: Arc<dyn MemoryMonitor>,
    config: ServerConfig,
) -> Result<CreateMessageResponse, ServerError> {
    // Check memory before processing
    let status = monitor.check_status();
    let available_gb = monitor.available_gb();

    // Log status
    match status {
        MemoryStatus::Safe | MemoryStatus::Caution => {
            tracing::debug!("Memory status: {:?}, available: {:.2}GB", status, available_gb);
        }
        MemoryStatus::Warning => {
            tracing::warn!("Memory warning: {:.2}GB available", available_gb);
        }
        MemoryStatus::Critical => {
            tracing::error!("Critical memory: {:.2}GB available, rejecting request", available_gb);
            return Err(ServerError::MemoryCritical {
                available_gb,
            });
        }
        MemoryStatus::Emergency => {
            tracing::error!("Emergency memory: {:.2}GB available, initiating shutdown", available_gb);
            // Trigger emergency shutdown
            initiate_emergency_shutdown("Memory exhausted");
            return Err(ServerError::MemoryEmergency {
                available_gb,
            });
        }
    }

    // Process request normally
    let response = handle_messages_internal(req, config).await?;

    // Add memory headers to response
    // (handled by middleware)

    Ok(response)
}
```

#### REFACTOR: Error Types
```rust
// In pensieve-09-anthropic-proxy/src/error.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Critical memory pressure: {available_gb:.2}GB available")]
    MemoryCritical {
        available_gb: f64,
    },

    #[error("Emergency memory exhaustion: {available_gb:.2}GB available")]
    MemoryEmergency {
        available_gb: f64,
    },

    // ... existing errors
}

impl ServerError {
    pub fn status_code(&self) -> u16 {
        match self {
            ServerError::MemoryCritical { .. } => 503,
            ServerError::MemoryEmergency { .. } => 503,
            // ... other mappings
        }
    }
}
```

### Phase 3: Emergency Shutdown (Week 1)

#### RED: Write Failing Tests
```rust
#[tokio::test]
async fn test_emergency_shutdown_logs_state() {
    // Mock emergency scenario
    let monitor = MockMemoryMonitor::new(0.3); // 300MB
    let shutdown = EmergencyShutdown::new(5);

    // Trigger shutdown (in test, capture instead of exit)
    let result = shutdown.prepare("Test emergency");

    assert!(result.logged_state);
    assert!(result.attempted_cleanup);
}

#[tokio::test]
async fn test_emergency_shutdown_creates_log() {
    let shutdown = EmergencyShutdown::new(5);
    shutdown.prepare("Test emergency");

    // Check log file exists
    let log_path = format!("pensieve-emergency-{}.log", /* timestamp */);
    assert!(Path::new(&log_path).exists());
}
```

#### GREEN: Implement Emergency Shutdown
```rust
// pensieve-09-anthropic-proxy/src/emergency.rs

use std::process;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{error, warn};

pub struct EmergencyShutdown {
    timeout_secs: u64,
}

impl EmergencyShutdown {
    pub fn new(timeout_secs: u64) -> Self {
        Self { timeout_secs }
    }

    pub async fn initiate(&self, reason: &str) -> ! {
        error!("EMERGENCY SHUTDOWN: {}", reason);

        // Log memory state
        if let Ok(mem) = self.log_memory_state() {
            error!("Final memory: {:.2}GB available", mem);
        }

        // Attempt cache cleanup (best effort)
        let cleanup_result = timeout(
            Duration::from_secs(self.timeout_secs),
            self.cleanup_resources()
        ).await;

        match cleanup_result {
            Ok(Ok(())) => warn!("Cleanup completed"),
            Ok(Err(e)) => error!("Cleanup failed: {}", e),
            Err(_) => error!("Cleanup timed out"),
        }

        // Exit
        error!("Exiting with code 1");
        process::exit(1);
    }

    fn log_memory_state(&self) -> Result<f64, Box<dyn std::error::Error>> {
        use sysinfo::{System, SystemExt};
        let mut sys = System::new_all();
        sys.refresh_memory();
        let available_gb = sys.available_memory() as f64 / 1024_f64.powi(3);
        Ok(available_gb)
    }

    async fn cleanup_resources(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Call Python bridge to clear MLX cache
        // This is best-effort, may fail
        let output = tokio::process::Command::new("python3")
            .arg("python_bridge/mlx_inference.py")
            .arg("--clear-cache")
            .output()
            .await?;

        if !output.status.success() {
            warn!("Cache clear failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        Ok(())
    }
}
```

### Phase 4: MLX Cache Management (Week 1)

#### Update Python Bridge
```python
# python_bridge/mlx_inference.py

import mlx.core as mx
import psutil
import logging

logger = logging.getLogger(__name__)

def generate_with_cleanup(model, tokenizer, prompt, max_tokens, **kwargs):
    """
    Generate text with automatic memory management.

    Ensures:
    - Memory logged before/after
    - Cache cleared after generation
    - Cleanup happens even on error
    """
    try:
        # Log memory before
        mem_before = psutil.virtual_memory()
        logger.info(f"Memory before: {mem_before.available / 1024**3:.2f}GB free")

        # Generate
        result = generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs)

        return result

    finally:
        # Always cleanup
        try:
            mx.metal.clear_cache()
            mem_after = psutil.virtual_memory()
            logger.info(f"Memory after: {mem_after.available / 1024**3:.2f}GB free")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def initialize_model_with_limits(model_path, cache_limit_gb=5):
    """
    Initialize model with memory limits.
    """
    # Set cache limit
    cache_bytes = int(cache_limit_gb * 1024**3)
    mx.metal.set_cache_limit(cache_bytes)
    logger.info(f"Set MLX cache limit: {cache_limit_gb}GB")

    # Load model
    model, tokenizer = load(model_path)

    return model, tokenizer

# Add CLI command for cache clearing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        mx.metal.clear_cache()
        print("Cache cleared")
        sys.exit(0)
```

### Phase 5: Multi-Instance Isolation (Week 1)

#### Create Wrapper Script
```bash
#!/bin/bash
# scripts/claude-local
#
# Wrapper for using Claude Code with local Pensieve server.
# Does NOT modify global configuration.

set -e

# Default configuration
PENSIEVE_PORT=${PENSIEVE_PORT:-7777}
PENSIEVE_TOKEN=${PENSIEVE_TOKEN:-pensieve-local-token}

# Set environment variables for this process only
export ANTHROPIC_BASE_URL="http://127.0.0.1:${PENSIEVE_PORT}"
export ANTHROPIC_API_KEY="${PENSIEVE_TOKEN}"

# Check if server is running
if ! curl -s "http://127.0.0.1:${PENSIEVE_PORT}/health" > /dev/null 2>&1; then
    echo "⚠️  Warning: Pensieve server not responding on port ${PENSIEVE_PORT}"
    echo "   Start server with: cargo run --bin pensieve-proxy --release"
    exit 1
fi

# Show configuration
echo "🔧 Using local Pensieve server"
echo "   URL: http://127.0.0.1:${PENSIEVE_PORT}"
echo "   Token: ${PENSIEVE_TOKEN}"
echo ""

# Run claude with all arguments
exec claude "$@"
```

#### Test Isolation
```bash
# tests/test-isolation.sh

#!/bin/bash
# Test multi-instance isolation

echo "Testing multi-instance isolation..."

# Terminal 1 simulation (local)
ANTHROPIC_BASE_URL=http://127.0.0.1:7777 \
ANTHROPIC_API_KEY=pensieve-local-token \
claude --print "test local" > /tmp/test-local.txt &
PID1=$!

# Terminal 2 simulation (real API - will fail without real key, but tests isolation)
claude --print "test real" > /tmp/test-real.txt &
PID2=$!

# Wait for both
wait $PID1
wait $P2

# Check isolation
if grep -q "test local" /tmp/test-local.txt && \
   ! grep -q "pensieve" /tmp/test-real.txt; then
    echo "✅ Isolation test passed"
    exit 0
else
    echo "❌ Isolation test failed"
    exit 1
fi
```

---

## Dependency Changes

### Add to Cargo.toml
```toml
# pensieve-09-anthropic-proxy/Cargo.toml

[dependencies]
sysinfo = "0.30"  # System memory monitoring
tokio = { version = "1.40", features = ["full"] }
tracing = "0.1"
thiserror = "2.0"

[dev-dependencies]
mockall = "0.12"  # For mocking traits in tests
```

### Python Dependencies
```bash
pip install psutil  # Memory monitoring in Python
```

---

## Testing Strategy

### Unit Tests
```bash
# Test memory monitoring
cargo test -p pensieve-09-anthropic-proxy --test memory_monitoring

# Test request rejection
cargo test -p pensieve-09-anthropic-proxy --test request_handling

# Test emergency shutdown (mocked)
cargo test -p pensieve-09-anthropic-proxy --test emergency_shutdown
```

### Integration Tests
```bash
# Test with real memory monitoring
cargo test -p pensieve-09-anthropic-proxy --test integration -- --include-ignored

# Test Python bridge cleanup
python3 python_bridge/test_memory_cleanup.py

# Test wrapper script
bash tests/test-isolation.sh
```

### Manual Testing
```bash
# 1. Start server with monitoring
RUST_LOG=debug cargo run --bin pensieve-proxy --release

# 2. Monitor memory in Activity Monitor

# 3. Send requests and watch memory
for i in {1..10}; do
    curl -X POST http://127.0.0.1:7777/v1/messages \
        -H "Authorization: Bearer pensieve-local-token" \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-3-sonnet-20240229","max_tokens":50,"messages":[{"role":"user","content":"Test '$i'"}]}'
    sleep 2
done

# 4. Verify memory returns to baseline
```

---

## Success Criteria

### Phase 1 Success (Memory Monitoring)
- ✅ All unit tests pass (10/10)
- ✅ Memory status correctly detected
- ✅ Available memory accurate within 100MB
- ✅ Performance: <10ms per check

### Phase 2 Success (Request Rejection)
- ✅ All integration tests pass (8/8)
- ✅ Requests accepted when memory safe
- ✅ Requests rejected when memory critical
- ✅ Proper HTTP status codes (503)
- ✅ Headers include memory status

### Phase 3 Success (Emergency Shutdown)
- ✅ Shutdown triggers at <1GB
- ✅ Logs emergency state
- ✅ Attempts cleanup (best effort)
- ✅ Exits cleanly with code 1

### Phase 4 Success (MLX Cache Management)
- ✅ Cache cleared after each request
- ✅ Memory returns to baseline
- ✅ No memory leaks over 100 requests
- ✅ Cleanup happens even on error

### Phase 5 Success (Multi-Instance Isolation)
- ✅ Wrapper script works
- ✅ Terminal A uses local server
- ✅ Terminal B uses real API
- ✅ No interference verified
- ✅ Global config unchanged

---

## Risk Mitigation

### Risk 1: sysinfo crate accuracy
**Mitigation**: Compare with `psutil` in Python, validate accuracy within 100MB

### Risk 2: Emergency shutdown during active request
**Mitigation**: 5-second grace period, in-flight request tracking

### Risk 3: MLX cache clear fails
**Mitigation**: Best-effort cleanup, log failure, still exit safely

### Risk 4: Wrapper script conflicts
**Mitigation**: Thorough isolation testing, document caveats

---

## Timeline

### Week 1 (Days 1-3): Core Implementation
- Day 1: Memory monitoring (RED → GREEN → REFACTOR)
- Day 2: Request rejection (RED → GREEN → REFACTOR)
- Day 3: Emergency shutdown (RED → GREEN → REFACTOR)

### Week 1 (Days 4-5): Integration
- Day 4: MLX cache management, Python bridge updates
- Day 5: Wrapper script, isolation testing

### Week 1 (Days 6-7): Testing & Documentation
- Day 6: Integration tests, manual testing
- Day 7: Documentation, final validation

---

## References

- D17: Memory Safety Research
- S01: README-MOSTIMP (steering principles)
- Ollama: Model lifecycle management patterns
- llama.cpp: Memory monitoring approaches
- MLX issues: #724, #1124, #1076 (memory leaks)

---

**Status**: Ready for Implementation
**Next Step**: Begin RED phase - write failing tests for memory monitoring
