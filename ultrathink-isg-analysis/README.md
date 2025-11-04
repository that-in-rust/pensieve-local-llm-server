# Pensieve ISG Analysis - Ultrathink

**Generated**: 2025-11-04
**Tool**: [Parseltongue](https://github.com/that-in-rust/parseltongue) ISG Explorer
**Codebase**: Pensieve Local LLM Server v0.1.0

## What is This?

This directory contains a comprehensive **Interface Signature Graph (ISG)** analysis of the Pensieve codebase. ISG analysis provides deep insights into code structure, dependencies, and architecture by analyzing function signatures, types, and relationships.

## Quick Start

1. **Read the overview**: Start with [00-overview.md](./00-overview.md)
2. **Explore visualizations**: Read the numbered markdown files in order
3. **Query raw data**: Use the JSON files with your own analysis tools
4. **Run custom queries**: Connect to the RocksDB database

## Contents

### Documentation

| File | Description | Size |
|------|-------------|------|
| [00-overview.md](./00-overview.md) | Analysis summary and codebase statistics | Overview |
| [01-module-dependencies.md](./01-module-dependencies.md) | Crate dependency graph and compliance check | Deep dive |
| [02-data-flow.md](./02-data-flow.md) | Request flow and data transformation pipeline | Deep dive |
| [03-public-api-surface.md](./03-public-api-surface.md) | Public interfaces and API stability | Reference |
| [04-architectural-layers.md](./04-architectural-layers.md) | Layer architecture and violation detection | Deep dive |

### Raw Data

| File | Size | Entities/Edges | Description |
|------|------|----------------|-------------|
| **ISGLevel00-edges.json** | 160 MB | 708,143 edges | Pure edge list for graph visualization |
| **ISGLevel01-entities.json** | 148 MB | 180,320 entities | Entity catalog with dependencies |
| **ISGLevel02-types.json** | 161 MB | 180,320 entities | Full type system information |
| **analysis-summary.json** | ~50 KB | Summary | High-level statistics |
| **pensieve-isg.db/** | 175 MB | Full state | RocksDB database for queries |

## Key Findings

### Architecture Health: 75/100 ⚠️

**Strengths**:
- ✅ **L1 (Core)** is minimal and well-designed (25 entities)
- ✅ **L3 (Application)** is lean and focused (155 entities)
- ✅ **Trait-based abstraction** in L2 enables flexibility

**Issues Found**:
- 🚨 **pensieve-05 is too large** (339 entities, 30% of codebase)
  - Recommendation: Split into 3 crates (loading, tokenization, metadata)
- ❓ **Undefined layer position** for pensieve-08 and pensieve-09 (188 entities)
  - Recommendation: Clarify purpose or merge into existing layers
- ⚠️ **L2→L2 dependency** (pensieve-06 → pensieve-04)
  - Status: Acceptable if trait-based, needs verification

### Codebase Statistics

- **Total entities**: 1,137 Rust/Python entities
- **Total edges**: 708,143 dependency relationships
- **Crates analyzed**: 10 workspace crates
- **Largest crate**: pensieve-05 (339 entities) - 🚨 Too large
- **Most complex**: pensieve-04 (217 entities, 9 traits)
- **Most lean**: pensieve-07 (25 entities) - ✅ Correctly minimal

## Use Cases

### 1. Architecture Validation

Verify that your code follows the layered architecture:

```bash
# Check layer compliance
cat analysis-summary.json | jq '.crate_stats'

# Verify no circular dependencies
cat ISGLevel00-edges.json | jq '.edges[] | select(.from_key | contains("pensieve"))'
```

**See**: [04-architectural-layers.md](./04-architectural-layers.md)

### 2. Dependency Analysis

Understand which crates depend on each other:

```bash
# Find all dependencies of pensieve-01
cat ISGLevel00-edges.json | jq '.edges[] | select(.from_key | contains("pensieve-01"))'

# Find reverse dependencies (who depends on pensieve-07?)
cat ISGLevel00-edges.json | jq '.edges[] | select(.to_key | contains("pensieve-07"))'
```

**See**: [01-module-dependencies.md](./01-module-dependencies.md)

### 3. API Surface Analysis

Identify public interfaces and assess API stability:

```bash
# Find all public entities
cat ISGLevel01-entities.json | jq '.entities[] | select(.is_public == true)'

# Find all traits (abstraction points)
cat ISGLevel02-types.json | jq '.entities[] | select(.entity_type == "trait")'
```

**See**: [03-public-api-surface.md](./03-public-api-surface.md)

### 4. Data Flow Tracing

Trace how data flows through the system:

```bash
# Find all function calls from HTTP server
cat ISGLevel00-edges.json | jq '.edges[] | select(.from_key | contains("pensieve-02")) | select(.edge_type == "Calls")'
```

**See**: [02-data-flow.md](./02-data-flow.md)

### 5. Refactoring Planning

Identify code that needs refactoring:

```bash
# Find large modules (potential split candidates)
cat ISGLevel01-entities.json | jq '[.entities[] | select(.file_path | contains("pensieve-05"))] | length'

# Find complex functions (high method count)
cat analysis-summary.json | jq '.crate_stats | to_entries | map({crate: .key, methods: .value.types.method}) | sort_by(.methods)'
```

**See**: [04-architectural-layers.md](./04-architectural-layers.md)

## Regenerating the Analysis

If the codebase changes, regenerate the analysis:

```bash
# Step 1: Ingest codebase into CozoDB
./parseltongue pt01-folder-to-cozodb-streamer . \
  --db rocksdb:ultrathink-isg-analysis/pensieve-isg.db \
  --verbose

# Step 2: Export Level 0 (edges)
./parseltongue pt02-level00 \
  --where-clause "ALL" \
  --db rocksdb:ultrathink-isg-analysis/pensieve-isg.db \
  --output ultrathink-isg-analysis/ISGLevel00-edges.json \
  --verbose

# Step 3: Export Level 1 (entities)
./parseltongue pt02-level01 \
  --include-code 0 \
  --where-clause "ALL" \
  --db rocksdb:ultrathink-isg-analysis/pensieve-isg.db \
  --output ultrathink-isg-analysis/ISGLevel01-entities.json \
  --verbose

# Step 4: Export Level 2 (type system)
./parseltongue pt02-level02 \
  --include-code 0 \
  --where-clause "ALL" \
  --db rocksdb:ultrathink-isg-analysis/pensieve-isg.db \
  --output ultrathink-isg-analysis/ISGLevel02-types.json \
  --verbose
```

**Duration**: ~3 minutes for full analysis

## Advanced Queries

### Find Circular Dependencies

```bash
python3 << 'EOF'
import json
from collections import defaultdict

with open('ultrathink-isg-analysis/ISGLevel00-edges.json') as f:
    data = json.load(f)

# Build adjacency graph
graph = defaultdict(set)
for edge in data['edges']:
    if 'pensieve-' in edge['from_key'] and 'pensieve-' in edge['to_key']:
        from_crate = edge['from_key'].split('/')[1] if '/' in edge['from_key'] else None
        to_crate = edge['to_key'].split('/')[1] if '/' in edge['to_key'] else None
        if from_crate and to_crate and from_crate.startswith('pensieve-'):
            graph[from_crate].add(to_crate)

# Detect cycles (simplified)
for crate in graph:
    for dep in graph[crate]:
        if crate in graph.get(dep, set()):
            print(f"Circular: {crate} ↔ {dep}")
EOF
```

### Find Most Connected Entities

```bash
cat ISGLevel01-entities.json | jq '.entities[] | select(.file_path | contains("pensieve")) | {key: .isgl1_key, forward: (.forward_deps | length), reverse: (.reverse_deps | length), total: ((.forward_deps | length) + (.reverse_deps | length))} | select(.total > 10)' | jq -s 'sort_by(.total) | reverse | .[:10]'
```

### Find Unused Code

```bash
# Find entities with no incoming edges
cat ISGLevel01-entities.json | jq '.entities[] | select(.file_path | contains("pensieve")) | select((.reverse_deps | length) == 0) | {key: .isgl1_key, type: .entity_type}'
```

## Visualization Tools

### Generate Graphviz DOT

```python
import json

with open('ISGLevel00-edges.json') as f:
    data = json.load(f)

# Filter Pensieve edges
pensieve_edges = [e for e in data['edges'] if 'pensieve-' in e['from_key'] and 'pensieve-' in e['to_key']]

# Generate DOT
print("digraph Pensieve {")
for edge in pensieve_edges[:100]:  # Limit for visibility
    print(f'  "{edge["from_key"][:30]}" -> "{edge["to_key"][:30]}";')
print("}")
```

Save and render with:
```bash
dot -Tpng pensieve-graph.dot -o pensieve-graph.png
```

### Interactive Visualization (Recommended)

Use tools like:
- **Gephi**: Import JSON as network graph
- **Neo4j**: Import as graph database
- **D3.js**: Web-based interactive visualization
- **Cytoscape**: Scientific network visualization

## Files Inventory

```
ultrathink-isg-analysis/
├── README.md                    # This file
├── 00-overview.md               # Analysis summary
├── 01-module-dependencies.md    # Dependency analysis
├── 02-data-flow.md              # Data flow tracing
├── 03-public-api-surface.md     # API documentation
├── 04-architectural-layers.md   # Architecture validation
├── ISGLevel00-edges.json        # 160 MB - Edge list
├── ISGLevel01-entities.json     # 148 MB - Entity catalog
├── ISGLevel02-types.json        # 161 MB - Type system
├── analysis-summary.json        # 50 KB - Statistics
└── pensieve-isg.db/             # 175 MB - RocksDB database
    ├── data/                    # Database files
    ├── manifest                 # Manifest
    └── ...
```

**Total size**: ~650 MB

## Recommendations

Based on this analysis, we recommend:

### High Priority

1. **Split pensieve-05** (339 entities → 3 crates)
   - Improves maintainability
   - Reduces compile times
   - Clarifies responsibilities

2. **Clarify pensieve-08/09 purpose**
   - Document layer position
   - Merge or separate as appropriate
   - Update CLAUDE.md

### Medium Priority

3. **Verify L2→L2 dependencies**
   - Check pensieve-06 → pensieve-04
   - Ensure trait-based coupling only

4. **Document architectural patterns**
   - Update CLAUDE.md with findings
   - Add architecture diagrams

### Low Priority

5. **Optimize build graph**
   - Parallelize crate compilation
   - Reduce unnecessary dependencies

6. **Add architectural tests**
   - Enforce layer rules at compile time
   - Detect circular dependencies in CI

## Related Resources

- **Parseltongue**: https://github.com/that-in-rust/parseltongue
- **ISG Methodology**: Interface Signature Graphs for code analysis
- **CLAUDE.md**: Project architectural guidelines
- **Pensieve README**: Main project documentation

## Questions?

For questions about this analysis:
1. Review the detailed markdown files
2. Query the raw JSON data
3. Consult CLAUDE.md for architectural context
4. Regenerate analysis if codebase has changed significantly

---

*Generated by Claude Code with Parseltongue ISG Explorer*
*Analysis represents codebase state as of 2025-11-04*
*Last ingestion: 181 seconds | 16,249 files | 180,320 entities*
