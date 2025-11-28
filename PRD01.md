# PRD01 – pensieve-local-llm-server (Ultra Minimal)

## Objective
Deliver a zero-config binary (`pensieve-local-llm-server`) that, when executed with no arguments, downloads the prebuilt Phi-4-reasoning-plus-4bit MLX bundle (if missing) and immediately starts an HTTP server on port **528491** ready to serve Anthropic-compatible requests on Apple Silicon hardware.

## Context Snapshot
- **Model**: Phi-4-reasoning-plus-4bit (advanced reasoning model, 3.8B params, ~2.4 GB) from mlx-community with cutting-edge reasoning capabilities and Apple Silicon MLX optimization.
- **Platform**: Apple Silicon w/ Metal GPU acceleration (p06-metal-gpu-accel).
- **Codebase**: Rust workspace with crates p01–p09; all functionality lives in Rust (MLX handled via Rust↔C bindings).
- **Existing Flow**: Previous CLIs required manual flags and optional conversions; this MVP hides everything behind one default action.

## Scope (MVP)
1. **Binary defaults**
   - Executable name: `pensieve-local-llm-server`.
   - Under the hood we still reuse the clap-based CLI stack, but every flag is hard-coded to the single supported configuration (no other inputs accepted).
   - Running without parameters triggers the full flow (no CLI flags, no config file).
2. **Model acquisition**
   - Check `models/phi-4-reasoning-plus-4bit/` for MLX cache.
   - If absent, auto-download the prebuilt bundle from mlx-community with resume + checksum verification.
   - No on-device conversion or format changes; bundle is used as-is.
3. **Server launch**
   - After verifying model assets, immediately boot Warp HTTP server on fixed port **528491**.
   - Bind to localhost; expose `/v1/messages` (Anthropic-compatible) and `/health`.
4. **Startup UX**
   - Console logs for: prerequisite checks, download progress, cache hit, server ready at `http://127.0.0.1:528491`.
   - Exit with descriptive error if hardware prerequisites fail (non-Apple Silicon, MLX missing, port already bound).

## Out of Scope
- CLI arguments, multi-model selection, or dynamic ports.
- Metrics/observability endpoints beyond `/health`.
- Additional download sources or conversion pipelines.
- Non-Apple-Silicon targets.

## Executable Specifications (WHEN...THEN...SHALL)

### ES001: Zero-Config First Launch
```
WHEN user executes `pensieve-local-llm-server` with no arguments
AND system has Apple Silicon with MLX framework
AND network connection is available
THEN system SHALL download Phi-4-reasoning-plus-4bit model automatically
AND SHALL display progress with ETA and checksum verification
AND SHALL start HTTP server on fixed port 528491
AND SHALL serve Anthropic-compatible requests within 15 minutes
```

### ES002: Warm Start Fast Launch
```
WHEN user executes `pensieve-local-llm-server` with cached model
AND model files pass integrity checksum validation
AND port 528491 is available
THEN system SHALL start server within 10 seconds
AND SHALL serve requests immediately
AND SHALL maintain full Anthropic API compatibility
```

### ES003: Anthropic API Compatibility
```
WHEN client sends valid Claude API v1 request to /v1/messages
AND includes proper authentication and content-type headers
AND request follows Claude API specification format
THEN system SHALL process request through Phi-4 reasoning model
AND SHALL return response in exact Claude API format
AND SHALL achieve ≥20 tokens/second streaming throughput
AND SHALL maintain 100% API specification compliance
```

### ES004: Model Management Reliability
```
WHEN system downloads Phi-4-reasoning-plus-4bit model from mlx-community
AND download is interrupted or resumed
AND sufficient disk space exists (>5GB)
THEN system SHALL resume interrupted downloads automatically
AND SHALL validate model integrity with SHA256 checksum
AND SHALL cache model for subsequent instant startups
```

## Performance Contracts (Test-Validated)

| Contract ID | Metric | Target | Validation Method |
|-------------|--------|--------|-------------------|
| PERF-COLD-001 | Fresh machine startup | <15 minutes | Integration test with timer |
| PERF-WARM-002 | Cached model startup | <10 seconds | Automated benchmark test |
| PERF-THROUGHPUT-003 | Token generation speed | ≥20 tokens/sec | Load test with token counting |
| PERF-MEMORY-004 | Total memory usage | <5GB under load | Memory monitoring test |
| PERF-API-005 | API response latency (p95) | <500ms | Latency percentile test |
| PERF-COMPAT-006 | Anthropic API compliance | 100% specification | Automated compliance test |

## TDD Implementation Requirements

### Four-Word Naming Convention
ALL function names must follow `verb_constraint_target_qualifier()` pattern:
- ✅ `download_phi4_model_with_progress()`
- ✅ `parse_claude_api_request_validate()`
- ✅ `launch_http_server_on_port_528491()`
- ✅ `generate_tokens_with_streaming_async()`
- ❌ `download_model()` (too short)
- ❌ `handle_http_request()` (too generic)

### STUB → RED → GREEN → REFACTOR Cycle
1. **STUB Phase**: Write failing test with expected interface
2. **RED Phase**: Verify test fails with clear error message
3. **GREEN Phase**: Minimal implementation to pass test
4. **REFACTOR Phase**: Improve code while maintaining test coverage

### Critical Test Categories
- **Unit Tests**: Each function with isolated testing
- **Integration Tests**: Cross-crate interaction validation
- **Performance Tests**: All contracts validated with benchmarks
- **Compliance Tests**: Anthropic API specification verification
- **Stress Tests**: Concurrent request handling validation

## Quality Assurance Requirements

### Pre-Commit Enforcement Checklist
Before ANY commit to main branch, ALL must be TRUE:
- [ ] **Four-Word Names**: All functions follow `verb_constraint_target_qualifier()` pattern
- [ ] **Tests Pass**: `cargo test --all` returns 0 failures
- [ ] **Build Passes**: `cargo build --release` succeeds without warnings
- [ ] **Zero TODOs**: No `TODO`, `STUB`, or `PLACEHOLDER` comments in production code
- [ ] **Performance Contracts**: All PERF-001 through PERF-006 tests passing
- [ ] **API Compliance**: 100% Claude API v1 specification compatibility verified
- [ ] **Memory Safety**: Valgrind/miri reports no memory issues
- [ ] **Documentation Updated**: README.md reflects current behavior

### Architecture Validation
- **Layer Separation**: L1 (p07) → L2 (p04-p06) → L3 (p01-p03) boundaries enforced
- **Trait-Based Design**: All major components implement traits for testability
- **RAII Resources**: All resources managed with automatic cleanup
- **Error Handling**: Structured errors with thiserror + anyhow context

## Implementation Strategy

### Version Compliance Strategy
- **v0.9.4**: Model download and caching (ES001, ES004)
- **v0.9.5**: Inference engine integration (ES003)
- **v0.9.6**: HTTP server and API compatibility (ES002, ES003)
- **v0.9.7**: Performance optimization (PERF-001-006)
- **v0.9.8**: Integration testing and validation
- **v0.9.9**: Documentation and deployment readiness
- **v1.0.0**: Production release with all requirements verified

### Core Implementation Notes
- **p01 CLI launcher**: Zero-config entry point with baked-in Phi-4-reasoning-plus-4bit + port 528491
- **p05 model storage**: mlx-community integration with SHA256 checksum validation
- **p04 inference engine**: Real MLX integration replacing mock implementations
- **p06 Metal GPU**: Apple Silicon acceleration through Rust↔C MLX bindings
- **p02 HTTP server**: Warp with full Claude API v1 compatibility

### Success Metrics
Each version increment delivers **EXACTLY ONE complete feature**, fully working end-to-end:
- ✅ Feature works in production binary
- ✅ All tests passing (not just new feature)
- ✅ Documentation updated (README, PRD)
- ✅ Zero TODOs, zero stubs, zero placeholders
- ✅ Pushed to origin/main
