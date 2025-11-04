# Module Dependencies Analysis

## Crate Dependency Graph

This analysis visualizes the dependencies between Pensieve workspace crates based on the Interface Signature Graph.

### Layered Architecture (CLAUDE.md Definition)

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Application                                        │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ │ pensieve-01 │  │ pensieve-02 │  │ pensieve-03 │         │
│ │   (CLI)     │  │ (HTTP API)  │  │(API Models) │         │
│ │  54 entities│  │ 60 entities │  │ 41 entities │         │
│ └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                           ↓ depends on
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Engine & Implementation                            │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ │ pensieve-04 │  │ pensieve-05 │  │ pensieve-06 │         │
│ │  (Engine)   │  │  (Models)   │  │   (Metal)   │         │
│ │ 217 entities│  │ 339 entities│  │ 167 entities│         │
│ └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                           ↓ depends on
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Core Foundation                                    │
│ ┌─────────────────────────────────────────────────────────┐│
│ │               pensieve-07 (Core)                         ││
│ │          Foundation traits & error types                 ││
│ │                  25 entities                             ││
│ └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

External Layer:
┌─────────────────────────────────────────────────────────────┐
│ Python Bridge: python_bridge/ (46 entities)                 │
│   - MLX inference integration                               │
│   - Apple Silicon GPU acceleration                          │
└─────────────────────────────────────────────────────────────┘

Supporting Crates:
┌─────────────────────────────────────────────────────────────┐
│ pensieve-08_claude_core (49 entities)                       │
│ pensieve-09-anthropic-proxy (139 entities)                  │
└─────────────────────────────────────────────────────────────┘
```

### Expected Dependency Rules (from CLAUDE.md)

1. **L1 crates** have zero external dependencies (except core/alloc)
2. **L2 crates** depend only on L1 crates + framework libs (Candle, Metal)
3. **L3 crates** can depend on any L1/L2 crates and external libs
4. **Never** create circular dependencies between layers
5. Keep **pensieve-07_core** as minimal foundation

### Complexity Metrics by Crate

| Crate | Total | Functions | Methods | Structs | Enums | Traits | Impl Blocks |
|-------|-------|-----------|---------|---------|-------|--------|-------------|
| **pensieve-05** (Models) | 339 | 41 | 184 | 33 | 2 | 1 | 63 |
| **pensieve-04** (Engine) | 217 | 41 | 93 | 26 | 1 | 9 | 39 |
| **pensieve-06** (Metal) | 167 | 19 | 85 | 14 | 5 | 4 | 35 |
| **pensieve-09** (Proxy) | 139 | 71 | 30 | 8 | 3 | 1 | 14 |
| **pensieve-02** (HTTP) | 60 | 17 | 21 | 7 | 1 | 1 | 9 |
| **pensieve-01** (CLI) | 54 | 15 | 16 | 7 | 4 | 1 | 7 |
| **pensieve-08** (Claude) | 49 | 11 | 14 | 6 | 7 | 1 | 4 |
| **python_bridge** | 46 | 35 | 3 | 0 | 0 | 0 | 0 |
| **pensieve-03** (API) | 41 | 15 | 6 | 6 | 5 | 0 | 3 |
| **pensieve-07** (Core) | 25 | 4 | 7 | 0 | 1 | 3 | 6 |

### Key Observations

#### 1. Largest Crate: pensieve-05 (Models)

- **339 entities** - significantly larger than others
- **184 methods** - suggests complex model handling logic
- **63 impl blocks** - extensive trait implementations
- **Recommendation**: Consider splitting into:
  - Model loading/serialization
  - Model inference interfaces
  - Model metadata handling

#### 2. Core Layer: pensieve-07

- **Only 25 entities** - good adherence to "minimal foundation" principle
- **3 traits, 6 impl blocks** - trait-based design
- **4 functions** - minimal functionality
- **Status**: ✓ Correctly positioned as minimal foundation

#### 3. Engine Layer Complexity

- **pensieve-04**: 217 entities (9 traits) - abstraction layer
- **pensieve-06**: 167 entities (4 traits) - Metal GPU impl
- **Observation**: High trait count in pensieve-04 suggests good abstraction

#### 4. Application Layer

- **pensieve-01** (CLI): 54 entities - lean CLI
- **pensieve-02** (HTTP): 60 entities - focused HTTP server
- **pensieve-03** (API): 41 entities - minimal API models
- **Status**: ✓ Appropriately sized for application logic

### Dependency Rule Compliance

Based on ISG analysis, we need to verify:

**L1 → No dependencies** (pensieve-07)
- ✓ Should have no external crate dependencies
- ⚠️ Need to verify via Cargo.toml analysis

**L2 → Depends only on L1**
- pensieve-04, pensieve-05, pensieve-06 should only depend on pensieve-07
- ⚠️ ISG edge analysis needed to verify no L2→L2 or L2→L3 dependencies

**L3 → Can depend on L1/L2**
- pensieve-01, pensieve-02, pensieve-03 can depend on any lower layer
- ✓ Expected to have widest dependency surface

### Inter-Module Call Patterns

Based on the 708,143 total edges in ISGLevel00-edges.json:

- **2,462 edges** are Pensieve-specific (Rust code)
- Most edges are external dependency calls (Cargo deps, stdlib)
- **Low inter-crate coupling** suggested by low Pensieve-specific edge count

### Potential Architecture Issues

#### Issue 1: pensieve-05 Size

**Concern**: 339 entities in pensieve-05 (Models) suggests potential violation of Single Responsibility Principle

**Impact**:
- Harder to maintain
- Increased compile times
- Reduced modularity

**Recommendation**:
- Extract model loading into separate crate
- Separate GGUF/safetensors handling
- Create pensieve-05a (loading), pensieve-05b (inference)

#### Issue 2: Supporting Crates (08, 09)

**Concern**: pensieve-08_claude_core and pensieve-09-anthropic-proxy don't fit the layered model

**Questions**:
- Where do they fit in L1/L2/L3 hierarchy?
- Do they violate dependency rules?
- Should they be integrated or remain separate?

**Action**: Review architectural position and dependencies

### Visualization Tools

To visualize the dependency graph:

1. **Export as Graphviz DOT**:
```python
import json
with open('ISGLevel00-edges.json') as f:
    data = json.load(f)
# Filter for pensieve edges
# Generate DOT format
```

2. **Use ISG data for cycle detection**:
```python
# Check for circular dependencies
# Verify layer compliance
```

3. **Generate dependency matrix**:
```
        01  02  03  04  05  06  07
  01 [  -   ?   ?   ?   ?   ?   ? ]
  02 [  ?   -   ?   ?   ?   ?   ? ]
  ...
```

## Recommended Actions

1. **Verify Cargo.toml dependencies** match ISG analysis
2. **Check for circular dependencies** using ISG edges
3. **Validate layer compliance** - ensure L2 doesn't depend on L3
4. **Consider splitting pensieve-05** if it violates cohesion
5. **Document pensieve-08, 09** positioning in architecture

## Query Examples

To explore dependencies further:

```bash
# Find all dependencies from pensieve-01
cat ISGLevel00-edges.json | jq '.edges[] | select(.from_key | contains("pensieve-01"))'

# Count dependencies by crate
cat ISGLevel00-edges.json | jq '.edges[] | select(.from_key | contains("pensieve")) | .from_key' | sort | uniq -c

# Find cross-layer dependencies (potential violations)
# Requires custom script to parse crate names and validate against layer rules
```

---

*Analysis based on 708,143 edges and 1,137 Pensieve entities*
