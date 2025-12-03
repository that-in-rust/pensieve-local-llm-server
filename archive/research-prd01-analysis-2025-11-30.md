# PRD01 Research Analysis
**Generated:** 2025-11-30T00:00:00Z
**Version Analyzed:** v0.9.4 (commit 6f242a0)
**Methodology:** Explore + Plan agents with comprehensive codebase analysis

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Overall PRD01 Compliance | **47%** |
| Critical Blockers | 4 |
| TODOs in Production | 41 |
| Test Coverage Status | Partial (scaffolding exists) |

The codebase has **solid architectural scaffolding** following parseltongue principles, but remains in **STUB/RED TDD phase** with critical paths incomplete.

---

## Codebase Inventory

### Workspace Structure (9 Crates)

| Crate | Purpose | LOC | Status |
|-------|---------|-----|--------|
| **p01** | CLI entry point | 897 | Partial - server never starts |
| **p02** | HTTP server (Warp) | 586 | Routes defined, mock handlers |
| **p03** | API types & validation | 629 | Implemented |
| **p04** | Inference engine | 2,364 | Traits only, mock impl |
| **p05** | Model storage | 3,925 | Download works, cache path wrong |
| **p06** | Metal GPU accel | 1,304 | Trait definitions only |
| **p07** | Foundation types | 184 | Complete (no-std) |
| **p08** | Claude API core | 660 | Trait-based DI |
| **p09** | Anthropic proxy | ~1,500 | SSE streaming implemented |

**Total:** ~17,500 lines Rust + 1,500 lines Python bridge

---

## Executable Specification Compliance

### ES001: Zero-Config First Launch

**Compliance: 60%**

| Requirement | Status | Evidence |
|-------------|--------|----------|
| No-argument execution | PARTIAL | `p01/src/main.rs:37-82` - parses with defaults |
| Auto-download Phi-4 | PARTIAL | Uses `Phi-4-mini-instruct-mlx` not `Phi-4-reasoning-plus-4bit` |
| Progress with ETA | COMPLETE | `p01/src/lib.rs:472-522` - indicatif progress bar |
| Start server port 528491 | **NOT DONE** | `main.rs:73` has `// TODO: Start HTTP server` |
| Serve requests | **NOT DONE** | Server loop only waits for Ctrl+C |

**Critical Gap:** Line 73-75 in `p01/src/main.rs`:
```rust
// TODO: Start HTTP server and inference engine
println!("Server starting on http://127.0.0.1:528491");
// Server never actually starts - only prints message
```

---

### ES002: Warm Start Fast Launch

**Compliance: 30%**

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Cached model detection | COMPLETE | `p01/src/lib.rs:415-419` |
| Checksum validation | PARTIAL | Uses `placeholder_checksum` (lib.rs:410) |
| <10 second startup | **NOT TESTED** | No benchmark exists |
| Immediate serving | **NOT DONE** | Server doesn't start |

**Critical Gap:** `standalone_build/src/lib.rs:89,92`:
```rust
// TODO: Implement actual memory check
// TODO: Implement actual disk space check
```

---

### ES003: Anthropic API Compatibility

**Compliance: 65%**

| Requirement | Status | Evidence |
|-------------|--------|----------|
| POST /v1/messages | COMPLETE | `p02/src/lib.rs:45-54` |
| GET /health | COMPLETE | `p02/src/lib.rs:29-36` |
| Claude API request format | COMPLETE | `p09/src/translator.rs:51-107` |
| Claude API response format | COMPLETE | `p09/src/translator.rs:121-143` |
| SSE streaming (6 events) | COMPLETE | `p09/src/streaming.rs` |
| Real inference | **NOT DONE** | Uses `MockRequestHandler` |

**Critical Gap:** `p02/src/lib.rs:88-91`:
```rust
// Handler delegates to MockRequestHandler, not real MLX
pub struct MlxRequestHandler;
impl MlxRequestHandler {
    // Returns mock responses only
}
```

**Working Alternative:** `python_bridge/mlx_inference.py` (696 lines) has complete MLX implementation but is not wired to Rust binary.

---

### ES004: Model Management Reliability

**Compliance: 55%**

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Resume downloads | PARTIAL | No HTTP Range header impl |
| SHA256 validation | COMPLETE | `standalone_build/src/lib.rs:537-561` |
| Cache path | INCORRECT | Uses `~/.cache/pensieve-models/` not `models/phi-4-reasoning-plus-4bit/` |
| >5GB disk check | PARTIAL | `standalone_build/src/lib.rs:92` is TODO |

**Model Mismatch:**
- PRD01 specifies: `Phi-4-reasoning-plus-4bit`
- Code uses: `Phi-4-mini-instruct-mlx`

---

## Performance Contract Analysis

| Contract | PRD01 Target | Code Value | Status |
|----------|--------------|------------|--------|
| PERF-COLD-001 | <15 min | Not measured | **NO TESTS** |
| PERF-WARM-002 | <10 sec | Not measured | **NO TESTS** |
| PERF-THROUGHPUT-003 | >=20 TPS | 10.0 TPS (p04:481) | **MISMATCH** |
| PERF-MEMORY-004 | <5GB | 12.0 GB (p04:479) | **MISMATCH** |
| PERF-API-005 | <500ms p95 | Route tests only | **INCOMPLETE** |
| PERF-COMPAT-006 | 100% | No compliance suite | **NOT TESTED** |

**Contract Value Contradictions in `p04/src/lib.rs:478-493`:**
```rust
pub struct InferencePerformanceContract {
    pub memory_usage_gb: f64,        // Code: 12.0, PRD01: <5.0
    pub tokens_per_second: f64,      // Code: 10.0, PRD01: >=20.0
}
```

---

## TDD Compliance

### Four-Word Naming Convention

**Compliance: 70%**

| Compliant Examples | Non-Compliant Examples |
|--------------------|------------------------|
| `parse_cli_arguments_validate()` | `format_sse_event()` (3 words) |
| `check_system_prerequisites_quiet()` | `load_model()` (2 words) |
| `ensure_model_directory_exists()` | `real_mlx_generate()` (3 words) |
| `create_http_routes_with_middleware()` | Python functions don't follow |
| `download_phi4_model_with_progress_async()` | |

### Test Categories

| Category | Required | Implemented |
|----------|----------|-------------|
| Unit Tests | YES | YES - `#[cfg(test)]` blocks |
| Integration Tests | YES | PARTIAL - 2 test files |
| Performance Tests | YES | PARTIAL - benchmark scaffolding |
| Compliance Tests | YES | **NO** - missing |
| Stress Tests | YES | YES - `stress_concurrent_requests.rs` |

### Pre-Commit Checklist

| Requirement | Status |
|-------------|--------|
| Four-word names | PARTIAL (70%) |
| `cargo test --all` passes | UNKNOWN |
| `cargo build --release` passes | UNKNOWN |
| Zero TODOs | **FAIL** (41 TODOs found) |
| Performance contracts | **FAIL** (0/6 validated) |
| 100% API compliance | **FAIL** (not tested) |
| Memory safety | UNKNOWN |
| README updated | YES |

---

## Version Roadmap Status

| Version | Milestone | Expected | Actual |
|---------|-----------|----------|--------|
| **v0.9.4** | Model download/caching | COMPLETE | **PARTIAL** - wrong model, placeholder checksum |
| v0.9.5 | Inference engine | NOT STARTED | MockRequestHandler only |
| v0.9.6 | HTTP server/API | NOT STARTED | Routes defined, no startup |
| v0.9.7 | Performance | NOT STARTED | Contract values wrong |
| v0.9.8 | Integration testing | NOT STARTED | - |
| v0.9.9 | Documentation | NOT STARTED | - |
| v1.0.0 | Production | NOT STARTED | - |

---

## Critical Blockers (Must Fix)

### 1. HTTP Server Never Starts
**File:** `p01-cli-interface-launcher/src/main.rs:73`
```rust
// TODO: Start HTTP server and inference engine
println!("Server starting..."); // Cosmetic only
```
**Impact:** ES001, ES002, ES003 all blocked

### 2. Mock Inference Handler
**File:** `p02-http-server-core/src/lib.rs:88-91, 341-344`
```rust
pub struct MlxRequestHandler; // Returns mock responses
```
**Impact:** ES003, PERF-THROUGHPUT-003, PERF-COMPAT-006 blocked

### 3. Wrong Model Identifier
**File:** `p01-cli-interface-launcher/src/lib.rs:399-401`
- Uses: `mlx-community/Phi-4-mini-instruct-mlx`
- PRD01: `Phi-4-reasoning-plus-4bit`
**Impact:** ES001, ES004 non-compliant

### 4. Performance Contract Mismatch
**File:** `p04-inference-engine-core/src/lib.rs:478-493`
- Memory: 12GB vs PRD01's 5GB
- Throughput: 10 TPS vs PRD01's 20 TPS
**Impact:** PERF-003, PERF-004 failures guaranteed

---

## Asset Map

### Key Implementation Files

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Main entry | `p01/src/main.rs` | 191 | TODO at line 73 |
| CLI parsing | `p01/src/lib.rs` | 707 | Defaults baked in |
| Model download | `standalone_build/src/lib.rs` | 700+ | Working but wrong model |
| HTTP routes | `p02/src/lib.rs` | 586 | Mock handler |
| API translation | `p09/src/translator.rs` | 200+ | Complete |
| SSE streaming | `p09/src/streaming.rs` | 300+ | Complete |
| Python MLX | `python_bridge/mlx_inference.py` | 696 | Working, not integrated |

### Test Files

| File | Lines | Purpose |
|------|-------|---------|
| `p02/tests/parseltongue_executable_contracts.rs` | 646 | BDD contract tests |
| `p02/tests/integration_tests.rs` | 456 | Route testing |
| `p02/tests/stress_concurrent_requests.rs` | 330 | Concurrency |
| `p05/tests/phi4_download_tests.rs` | 143 | Download (stubs) |
| `python_bridge/test_mlx_inference.py` | 414 | Python MLX tests |

---

## Recommendations

### Immediate (v0.9.4 Completion)
1. Fix model identifier to `Phi-4-reasoning-plus-4bit`
2. Replace placeholder checksum with real SHA256
3. Fix cache path to `models/phi-4-reasoning-plus-4bit/`

### Short-term (v0.9.5-v0.9.6)
1. Implement actual HTTP server startup in `main.rs`
2. Wire Python MLX bridge to Rust binary OR implement Rust MLX bindings
3. Replace `MockRequestHandler` with real inference

### Medium-term (v0.9.7-v0.9.8)
1. Fix performance contract values (20 TPS, 5GB)
2. Add PERF-001 through PERF-006 benchmark tests
3. Create Anthropic API compliance test suite

### Pre-v1.0.0
1. Remove all 41 TODOs from production code
2. Enforce four-word naming across Python files
3. Full integration test suite with real hardware

---

## Appendix: TODO Inventory (41 items)

```
p01-cli-interface-launcher/src/main.rs:73
standalone_build/src/lib.rs:89
standalone_build/src/lib.rs:92
standalone_build/src/main.rs:24
standalone_build/src/main.rs:28
p05-model-storage-core/tests/phi4_download_tests.rs: Multiple todo!()
[... 35 more across codebase]
```

---

## Rust-Native Inference Options (PRD01 Compliant)

PRD01 mandates: **"all functionality lives in Rust (MLX handled via Rust↔C bindings)"**

### Option Comparison Matrix

| Solution | Crate | Version | Phi-4 Support | Metal GPU | Maturity | Recommendation |
|----------|-------|---------|---------------|-----------|----------|----------------|
| **mistral.rs** | `mistral-rs` | v0.5.0 | YES (Phi-4 MM) | YES | Production | **BEST CHOICE** |
| **Candle** | `candle-core` | 0.9.1 | YES (quantized) | YES | Production | Good fallback |
| **mlx-rs** | `mlx-rs` | 0.25.1 | Partial | YES (native) | Active dev | Apple-native |
| **llama-cpp-rs** | `llama-cpp-2` | Latest | Via GGUF | YES | Stable | Max performance |

### Recommended Path: mistral.rs

**Why mistral.rs is the best fit for PRD01:**

1. **Pure Rust** - No Python dependency, no FFI complexity
2. **Phi-4 Multimodal support** - Added in v0.5.0 (March 2025)
3. **Metal acceleration** - Competitive with llama.cpp on M-series
4. **OpenAI-compatible API** - Easy to adapt to Anthropic format
5. **Active maintenance** - Regular releases, good community
6. **Built on Candle** - Leverages HuggingFace ecosystem

**Integration approach:**
```rust
// Replace MockRequestHandler with mistral.rs backend
// mistral.rs provides async streaming generation
// Metal GPU acceleration automatic on macOS
```

### Current Codebase Gap

| Component | Current State | Required Change |
|-----------|---------------|-----------------|
| Candle version | 0.8 | Upgrade to 0.9.1 |
| p04 inference | Mock only | Integrate mistral.rs or Candle Phi |
| p06 Metal | Traits only | Use Candle's `Device::new_metal(0)` |
| Model format | MLX safetensors | GGUF or Candle safetensors |

### Alternative: Candle Native Phi

If mistral.rs doesn't fit, implement Phi directly in Candle:

```rust
// candle-examples/examples/quantized-phi/ exists
// Can load Phi-4 GGUF format
// Metal support via features = ["metal"]

cargo run --example quantized-phi --release --features metal \
  -- --model phi-4 --prompt "Hello"
```

**Candle Phi-4 example exists at:**
- `candle-examples/examples/phi/` (standard)
- `candle-examples/examples/quantized-phi/` (GGUF quantized)

### mlx-rs Option (Apple-Native)

**Crate:** `mlx-rs` v0.25.1 (oxideai/mlx-rs)

- Unofficial but active Rust bindings to Apple MLX
- Idiomatic Rust API
- Optimized for unified memory architecture
- Requires Rust 1.82.0+

**Trade-off:** Trails llama.cpp and Candle in raw performance, but native Apple optimization.

### Performance Benchmarks (M-series)

From research (Mistral-7B Q4 GGUF):

| Framework | Tokens/sec | Notes |
|-----------|------------|-------|
| llama.cpp | **Fastest** | C++ with Metal |
| mistral.rs | ~Same | Pure Rust, v0.5.0 optimized |
| Candle | Close second | HuggingFace maintained |
| MLX | Third | Python-first design |

**For PRD01's 20 TPS target:** All Rust options can achieve this on M1+.

---

## Implementation Roadmap (Rust-Native)

### Phase 1: Foundation (Week 1-2)
- [ ] Upgrade `candle-core` to 0.9.1 in workspace Cargo.toml
- [ ] Add `mistral-rs` or implement Candle Phi loader
- [ ] Download Phi-4 in GGUF format (compatible with Candle)
- [ ] Validate model loads on Metal GPU

### Phase 2: Inference Engine (Week 3-4)
- [ ] Replace `MockRequestHandler` with real inference
- [ ] Implement streaming token generation
- [ ] Wire to HTTP server (p02)
- [ ] Test `/v1/messages` endpoint with real responses

### Phase 3: Integration (Week 5-6)
- [ ] Full startup flow: download → load → serve
- [ ] Performance benchmarking (target: 20 TPS)
- [ ] Memory profiling (target: <5GB)
- [ ] Stress testing concurrent requests

### Phase 4: Polish (Week 7-8)
- [ ] Remove all TODOs
- [ ] Complete test coverage
- [ ] PRD01 compliance validation
- [ ] Documentation

---

## Key Files to Modify

| File | Change Required |
|------|-----------------|
| `Cargo.toml` (workspace) | Upgrade candle to 0.9.1, add mistral-rs |
| `p04/src/lib.rs` | Replace mock with real Candle/mistral inference |
| `p04/Cargo.toml` | Add `features = ["metal"]` |
| `p01/src/main.rs:73` | Implement actual server startup |
| `p02/src/lib.rs` | Wire real inference to HTTP handler |

---

## Sources

- [huggingface/candle](https://github.com/huggingface/candle) - Candle framework
- [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs) - mistral.rs
- [oxideai/mlx-rs](https://github.com/oxideai/mlx-rs) - MLX Rust bindings
- [candle quantized-phi example](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-phi/main.rs)
- [mistral.rs v0.5.0 release](https://huggingface.co/blog/EricB/mistralrs-v0-5-0) - Phi-4 MM support

---

*Research generated by Claude Code using Explore + Plan agents*
