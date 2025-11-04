# Architectural Layers Analysis

## Overview

This document analyzes the layered architecture of Pensieve as defined in CLAUDE.md and validated through ISG analysis.

## Architecture Definition (CLAUDE.md)

### Layer Rules

1. **L1 (Core)**: No external dependencies (except core/alloc)
2. **L2 (Engine)**: Depends only on L1 + framework libs (Candle, Metal)
3. **L3 (Application)**: Can depend on L1, L2, and external libs
4. **No circular dependencies** between layers
5. **pensieve-07_core** must remain minimal

## Actual Layer Implementation

### Layer 1: Core Foundation

```
┌────────────────────────────────────────────────────────┐
│                   pensieve-07_core                     │
│                    (25 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: Foundation traits and error types            │
│ Dependencies: zero (except core/alloc)                 │
│ Entities:                                              │
│   - 3 traits (InferenceProvider, etc.)                 │
│   - 6 impl blocks                                      │
│   - 4 functions                                        │
│   - 7 methods                                          │
│   - 1 enum (CoreError)                                 │
│                                                         │
│ Status: ✅ Correctly minimal (25 entities)             │
│ Compliance: ✅ No external dependencies                │
└────────────────────────────────────────────────────────┘
```

**Analysis**:
- ✅ **Size**: 25 entities is appropriately minimal
- ✅ **Focus**: Error types + core traits only
- ✅ **No external deps**: Follows L1 rule
- ✅ **Shared by all**: Foundation for L2/L3

**Key Types**:
```rust
// Core error (1 enum)
pub enum CoreError { ... }
pub type CoreResult<T> = Result<T, CoreError>;

// Foundation traits (3 traits)
pub trait InferenceProvider { ... }
pub trait TokenEncoder { ... }
pub trait ModelMetadata { ... }
```

### Layer 2: Engine & Implementation

#### pensieve-04_engine (217 entities)

```
┌────────────────────────────────────────────────────────┐
│               pensieve-04_engine                       │
│                  (217 entities)                        │
├────────────────────────────────────────────────────────┤
│ Purpose: Inference engine abstractions                 │
│ Dependencies: pensieve-07 + Candle                     │
│ Entities:                                              │
│   - 9 traits (high abstraction)                        │
│   - 93 methods                                         │
│   - 41 functions                                       │
│   - 39 impl blocks                                     │
│   - 26 structs                                         │
│                                                         │
│ Status: ⚠️  High complexity (217 entities)             │
│ Compliance: ✅ Depends only on L1 + Candle             │
└────────────────────────────────────────────────────────┘
```

**9 Traits** suggest comprehensive abstraction:
- High flexibility for implementations
- Supports multiple inference backends
- Enables future MLX migration

**Concerns**:
- 217 entities is substantial for an abstraction layer
- May contain implementation details better suited to L3

#### pensieve-05_models (339 entities) ⚠️ LARGEST

```
┌────────────────────────────────────────────────────────┐
│               pensieve-05_models                       │
│                  (339 entities)                        │
├────────────────────────────────────────────────────────┤
│ Purpose: Model loading and management                  │
│ Dependencies: pensieve-07 + file I/O                   │
│ Entities:                                              │
│   - 184 methods (HIGHEST)                              │
│   - 63 impl blocks (HIGHEST)                           │
│   - 41 functions                                       │
│   - 33 structs                                         │
│   - 1 trait                                            │
│                                                         │
│ Status: 🚨 TOO LARGE - violates cohesion               │
│ Compliance: ❓ May have L3 responsibilities            │
└────────────────────────────────────────────────────────┘
```

**Red Flags**:
- 🚨 **339 entities** - 56% larger than next largest
- 🚨 **184 methods** - suggests multiple responsibilities
- 🚨 **63 impl blocks** - excessive trait implementations

**Recommended Refactoring**:
```
pensieve-05_models (339) → Split into:
  ├─ pensieve-05a_model_loading (100-150 entities)
  │    - Safetensors loading
  │    - Model file parsing
  │    - Validation
  │
  ├─ pensieve-05b_tokenization (50-100 entities)
  │    - Tokenizer loading
  │    - Encode/decode
  │    - Vocabulary management
  │
  └─ pensieve-05c_model_metadata (50-100 entities)
       - Model config
       - Architecture info
       - Capability detection
```

#### pensieve-06_metal (167 entities)

```
┌────────────────────────────────────────────────────────┐
│               pensieve-06_metal                        │
│                  (167 entities)                        │
├────────────────────────────────────────────────────────┤
│ Purpose: Metal GPU acceleration (macOS only)           │
│ Dependencies: pensieve-07, pensieve-04 + Metal         │
│ Entities:                                              │
│   - 85 methods                                         │
│   - 35 impl blocks                                     │
│   - 19 functions                                       │
│   - 14 structs                                         │
│   - 4 traits                                           │
│                                                         │
│ Status: ✅ Appropriate size for GPU layer              │
│ Compliance: ⚠️  Depends on L2 (pensieve-04)            │
└────────────────────────────────────────────────────────┘
```

**Dependency Concern**:
- pensieve-06 depends on pensieve-04 (both L2)
- **Acceptable** if dependency is trait-based
- ⚠️ **Check**: Ensure no concrete type coupling

### Layer 3: Application

#### pensieve-01 (CLI) - 54 entities

```
┌────────────────────────────────────────────────────────┐
│                 pensieve-01 (CLI)                      │
│                    (54 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: Command-line interface                        │
│ Dependencies: ALL L1/L2 crates + clap                  │
│ Entities:                                              │
│   - 16 methods (arg parsing)                           │
│   - 15 functions (commands)                            │
│   - 7 structs (config)                                 │
│   - 7 impl blocks                                      │
│   - 4 enums (Commands)                                 │
│                                                         │
│ Status: ✅ Lean application layer                      │
│ Compliance: ✅ Can depend on all layers                │
└────────────────────────────────────────────────────────┘
```

**Analysis**:
- ✅ **54 entities** - appropriately sized CLI
- ✅ **4 enums** - likely Commands + error types
- ✅ **Thin layer** - delegates to L2

#### pensieve-02 (HTTP) - 60 entities

```
┌────────────────────────────────────────────────────────┐
│               pensieve-02 (HTTP API)                   │
│                    (60 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: HTTP server with SSE streaming                │
│ Dependencies: ALL L1/L2 + Warp, pensieve-03            │
│ Entities:                                              │
│   - 21 methods (handlers)                              │
│   - 17 functions (routes)                              │
│   - 9 impl blocks                                      │
│   - 7 structs (server state)                           │
│   - 3 modules                                          │
│                                                         │
│ Status: ✅ Focused HTTP layer                          │
│ Compliance: ✅ Depends on L1/L2 + L3 (pensieve-03)     │
└────────────────────────────────────────────────────────┘
```

**Analysis**:
- ✅ **60 entities** - lean HTTP server
- ✅ **SSE streaming** - async architecture
- ✅ **Delegates to L2** - thin orchestration

#### pensieve-03 (API Models) - 41 entities

```
┌────────────────────────────────────────────────────────┐
│            pensieve-03 (API Models)                    │
│                    (41 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: Anthropic API compatibility types             │
│ Dependencies: pensieve-07, pensieve-05 + serde         │
│ Entities:                                              │
│   - 15 functions (serialization)                       │
│   - 6 structs (request/response)                       │
│   - 6 methods (validation)                             │
│   - 5 enums (roles, errors)                            │
│   - 4 modules                                          │
│                                                         │
│ Status: ✅ Minimal API layer                           │
│ Compliance: ✅ Appropriate dependencies                │
└────────────────────────────────────────────────────────┘
```

**Analysis**:
- ✅ **41 entities** - lean data model layer
- ✅ **5 enums** - well-defined types
- ✅ **No business logic** - pure data models

### External Layer: Python Bridge

```
┌────────────────────────────────────────────────────────┐
│              python_bridge (MLX)                       │
│                    (46 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: MLX framework integration                     │
│ Dependencies: MLX, mlx-lm (Python)                     │
│ Entities:                                              │
│   - 35 functions                                       │
│   - 8 classes                                          │
│   - 3 methods                                          │
│                                                         │
│ Status: ✅ External integration layer                  │
│ Performance: ⚠️  16.85 TPS (target: 25+ TPS)           │
└────────────────────────────────────────────────────────┘
```

**Future Migration**:
- Current: Subprocess communication (high overhead)
- Target: Native Rust-MLX bindings
- Impact: Replace python_bridge with pensieve-04 impl

## Out-of-Layer Crates

### pensieve-08_claude_core (49 entities)

```
┌────────────────────────────────────────────────────────┐
│          pensieve-08_claude_core                       │
│                    (49 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: Claude API integration (unclear)              │
│ Position: ❓ Not defined in CLAUDE.md layers           │
│ Entities:                                              │
│   - 14 methods                                         │
│   - 11 functions                                       │
│   - 7 enums (errors)                                   │
│   - 7 modules                                          │
│   - 6 structs                                          │
└────────────────────────────────────────────────────────┘
```

**Questions**:
- ❓ Purpose unclear - Claude integration or proxy?
- ❓ Should it be in L3 (application logic)?
- ❓ Relationship to pensieve-09?

### pensieve-09-anthropic-proxy (139 entities)

```
┌────────────────────────────────────────────────────────┐
│        pensieve-09-anthropic-proxy                     │
│                   (139 entities)                       │
├────────────────────────────────────────────────────────┤
│ Purpose: Anthropic API proxy (unclear)                 │
│ Position: ❓ Not defined in CLAUDE.md layers           │
│ Entities:                                              │
│   - 71 functions (MANY)                                │
│   - 30 methods                                         │
│   - 14 impl blocks                                     │
│   - 10 modules                                         │
│   - 8 structs                                          │
└────────────────────────────────────────────────────────┘
```

**Concerns**:
- 🚨 **71 functions** - high for proxy layer
- ❓ **Relationship to pensieve-02** unclear
- ❓ **Should proxy be separate** or integrated?

**Recommendations**:
1. Clarify purpose vs pensieve-02 (HTTP server)
2. Consider merging into pensieve-02 if redundant
3. If needed, document as separate L3 crate

## Layer Compliance Matrix

| Crate | Layer | Size | Depends On | Status |
|-------|-------|------|------------|--------|
| pensieve-07 | L1 | 25 | core/alloc | ✅ Compliant |
| pensieve-04 | L2 | 217 | L1, Candle | ✅ Compliant |
| pensieve-05 | L2 | 339 | L1 | ⚠️  Too large |
| pensieve-06 | L2 | 167 | L1, L2, Metal | ⚠️  L2→L2 dependency |
| pensieve-01 | L3 | 54 | L1, L2, clap | ✅ Compliant |
| pensieve-02 | L3 | 60 | L1, L2, L3, Warp | ✅ Compliant |
| pensieve-03 | L3 | 41 | L1, L2, serde | ✅ Compliant |
| pensieve-08 | ❓ | 49 | ❓ | ❓ Undefined |
| pensieve-09 | ❓ | 139 | ❓ | ❓ Undefined |
| python_bridge | External | 46 | MLX | ✅ External layer |

## Architectural Patterns

### Pattern 1: Trait-Based Abstraction (L2)

**pensieve-04**: 9 traits provide flexibility
```rust
// L2 defines traits
trait InferenceEngine { ... }

// L2 implementations
impl InferenceEngine for CandleEngine { ... }
impl InferenceEngine for MLXEngine { ... }  // future

// L3 uses trait
fn run_inference(engine: &dyn InferenceEngine) { ... }
```

**Benefits**:
- ✅ Decouples L3 from L2 implementations
- ✅ Enables multiple backends
- ✅ Testable with mock implementations

### Pattern 2: Error Propagation (L1 → L3)

```rust
// L1: Core error type
enum CoreError { ... }

// L2: Wraps with context
enum EngineError {
    Core(CoreError),
    ModelLoadFailed,
}

// L3: Wraps with API error
enum ApiError {
    Engine(EngineError),
    InvalidRequest,
}
```

**Benefits**:
- ✅ Type-safe error handling
- ✅ Maintains error context
- ✅ Layer-specific error details

### Pattern 3: Dependency Injection (L3)

```rust
// L3 creates engine from L2
fn start_server(config: Config) -> Result<(), Error> {
    let engine = create_engine(&config)?;  // L2
    let server = HttpServer::new(engine);  // L3
    server.run().await
}
```

## Architectural Violations

### Violation 1: pensieve-05 Size

**Issue**: 339 entities violates cohesion principle

**Evidence**:
- 184 methods (too many responsibilities)
- 63 impl blocks (too many traits)
- 56% larger than next biggest (pensieve-04: 217)

**Impact**:
- Hard to maintain
- Increased compile time
- Unclear boundaries

**Recommendation**: Split into 3 crates (loading, tokenization, metadata)

### Violation 2: pensieve-06 → pensieve-04 (L2→L2)

**Issue**: pensieve-06 depends on pensieve-04 (both L2)

**Acceptable if**:
- ✅ Dependency is trait-based only
- ✅ No concrete type coupling

**Check Required**:
```bash
# Verify pensieve-06 only imports traits from pensieve-04
grep "use pensieve_04" pensieve-06/src/**/*.rs
```

### Violation 3: Undefined Layer Position (08, 09)

**Issue**: pensieve-08 and pensieve-09 not in CLAUDE.md layer model

**Impact**:
- Unclear architecture
- Potential for circular dependencies
- Maintenance confusion

**Recommendation**: Define layer position or remove if obsolete

## Complexity Distribution

```
Total entities: 1,137

Layer 1 (Core):           25 entities ( 2.2%)  ✅ Minimal
Layer 2 (Engine):        723 entities (63.6%)  ⚠️  Large
  - pensieve-04:         217 (19.1%)
  - pensieve-05:         339 (29.8%) 🚨 Too large
  - pensieve-06:         167 (14.7%)
Layer 3 (Application):   155 entities (13.6%)  ✅ Lean
  - pensieve-01:          54 ( 4.7%)
  - pensieve-02:          60 ( 5.3%)
  - pensieve-03:          41 ( 3.6%)
Undefined (08, 09):      188 entities (16.5%)  ❓
External (Python):        46 entities ( 4.0%)  ✅
```

**Analysis**:
- ✅ L1 is appropriately minimal (2.2%)
- ⚠️  L2 is 64% of codebase (expected for engine layer)
- 🚨 pensieve-05 alone is 30% (needs splitting)
- ❓ 16.5% undefined position (needs clarification)

## Recommended Architectural Changes

### Priority 1: Split pensieve-05

```
Current:
  pensieve-05: 339 entities (30% of codebase)

Proposed:
  pensieve-05a_loading: ~120 entities
  pensieve-05b_tokenization: ~90 entities
  pensieve-05c_metadata: ~80 entities
  Savings: Better cohesion, clearer boundaries
```

### Priority 2: Clarify 08/09 Position

```
Option A: Merge into existing L3 crates
  pensieve-08 → pensieve-03 (if API-related)
  pensieve-09 → pensieve-02 (if proxy-related)

Option B: Define new layer
  Add "Proxy Layer" between L2 and L3
  Document purpose and dependencies

Option C: Remove if obsolete
  If no longer used, remove entirely
```

### Priority 3: Verify L2→L2 Dependencies

```bash
# Check pensieve-06 imports from pensieve-04
# Ensure only trait imports, no concrete types

cargo tree -p pensieve-06 --depth 1
```

## Layer Health Score

| Layer | Score | Status |
|-------|-------|--------|
| L1 (Core) | 95/100 | ✅ Excellent |
| L2 (Engine) | 70/100 | ⚠️  Good but needs cleanup |
| L3 (Application) | 90/100 | ✅ Very Good |
| Undefined | 40/100 | ❓ Needs clarification |
| Overall | 75/100 | ⚠️  Good, fixable issues |

**Strengths**:
- ✅ L1 is minimal and focused
- ✅ L3 is lean and delegates well
- ✅ Trait-based abstraction in L2

**Weaknesses**:
- 🚨 pensieve-05 too large (339 entities)
- ❓ Undefined position for 08, 09
- ⚠️  L2→L2 dependency needs verification

---

*Architectural analysis based on 1,137 entities across 10 crates*
