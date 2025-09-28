# Chunked Analysis Workflow

This example demonstrates advanced chunked analysis for large codebases using code-ingest's intelligent file chunking capabilities.

## Overview

We'll analyze a large repository (like the Rust compiler) using chunked analysis to handle files that are too large for standard analysis while maintaining contextual understanding through L1 and L2 context windows.

## Prerequisites

- PostgreSQL with sufficient storage (>1GB recommended)
- code-ingest installed
- High-memory system (8GB+ RAM recommended for large repositories)

## Step 1: Ingest Large Repository

### Target Repository Selection

For this example, we'll use the Rust compiler repository, which contains many large files:

```bash
# Ingest the Rust compiler repository
./target/release/code-ingest ingest https://github.com/rust-lang/rust \
  --db-path /Users/username/desktop/RustCompilerDB \
  --include "*.rs,*.md,*.toml" \
  --exclude "tests/ui/*,src/test/*,*.lock" \
  --max-concurrency 16 \
  --batch-size 2000
```

Expected output:
```
Starting repository ingestion...
Repository: https://github.com/rust-lang/rust
Processing 15,847 files...
Large files detected: 234 files > 1000 lines
Ingestion completed: INGEST_20250928160000
Duration: 8 minutes 23 seconds
```

### Analyze File Size Distribution

```sql
-- Connect to database
psql postgresql://localhost:5432/RustCompilerDB

-- Analyze file size distribution
SELECT 
  CASE 
    WHEN line_count < 100 THEN 'Small (< 100 lines)'
    WHEN line_count < 500 THEN 'Medium (100-500 lines)'
    WHEN line_count < 1000 THEN 'Large (500-1000 lines)'
    WHEN line_count < 2000 THEN 'Very Large (1000-2000 lines)'
    ELSE 'Huge (> 2000 lines)'
  END as size_category,
  COUNT(*) as file_count,
  AVG(line_count) as avg_lines,
  MAX(line_count) as max_lines
FROM "INGEST_20250928160000" 
WHERE extension = 'rs' 
  AND line_count IS NOT NULL
GROUP BY size_category
ORDER BY AVG(line_count);
```

Expected results:
```
    size_category     | file_count | avg_lines | max_lines
----------------------+------------+-----------+-----------
 Small (< 100 lines)  |       8234 |        45 |        99
 Medium (100-500 lines)|      4567 |       287 |       499
 Large (500-1000 lines)|       892 |       734 |       999
 Very Large (1000-2000)|       234 |      1456 |      1999
 Huge (> 2000 lines)  |        67 |      3245 |      8934
```

## Step 2: Identify Chunking Candidates

### Find Files Requiring Chunking

```sql
-- Identify large files that would benefit from chunking
SELECT 
  filepath,
  line_count,
  file_size_bytes,
  CEIL(line_count::FLOAT / 500) as chunks_needed_500,
  CEIL(line_count::FLOAT / 300) as chunks_needed_300
FROM "INGEST_20250928160000" 
WHERE extension = 'rs' 
  AND line_count > 500
ORDER BY line_count DESC
LIMIT 20;
```

### Analyze Module Complexity

```sql
-- Find the most complex modules
SELECT 
  SPLIT_PART(filepath, '/', 1) as top_module,
  SPLIT_PART(filepath, '/', 2) as sub_module,
  COUNT(*) as file_count,
  AVG(line_count) as avg_complexity,
  MAX(line_count) as max_complexity,
  SUM(line_count) as total_lines
FROM "INGEST_20250928160000" 
WHERE extension = 'rs' 
  AND line_count > 300
GROUP BY top_module, sub_module
ORDER BY total_lines DESC
LIMIT 15;
```

## Step 3: Generate Chunked Analysis Tables

### Create 500-Line Chunks for Detailed Analysis

```bash
# Generate chunked analysis with 500-line chunks
code-ingest generate-hierarchical-tasks INGEST_20250928160000 \
  --chunks 500 \
  --levels 4 \
  --groups 8 \
  --output rust_compiler_chunked_500.md \
  --db-path /Users/username/desktop/RustCompilerDB
```

This creates:
- Chunked table: `INGEST_20250928160000_500`
- Content files with L1/L2 context
- Hierarchical task structure

### Create 300-Line Chunks for Fine-Grained Analysis

```bash
# Generate fine-grained chunked analysis
code-ingest generate-hierarchical-tasks INGEST_20250928160000 \
  --chunks 300 \
  --levels 3 \
  --groups 6 \
  --prompt-file .kiro/steering/rust-compiler-analysis.md \
  --output rust_compiler_chunked_300.md \
  --db-path /Users/username/desktop/RustCompilerDB
```

### Verify Chunked Table Creation

```sql
-- Verify chunked tables were created
SELECT 
  table_name,
  (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_name LIKE 'INGEST_20250928160000_%'
ORDER BY table_name;

-- Analyze chunk distribution in 500-line table
SELECT 
  filepath,
  COUNT(*) as chunk_count,
  MIN(chunk_start_line) as first_line,
  MAX(chunk_end_line) as last_line,
  MAX(chunk_end_line) - MIN(chunk_start_line) + 1 as total_lines
FROM "INGEST_20250928160000_500"
GROUP BY filepath
ORDER BY chunk_count DESC
LIMIT 10;
```

## Step 4: Advanced Chunked Analysis Queries

### Context Window Analysis

```sql
-- Analyze context window effectiveness
SELECT 
  filepath,
  chunk_number,
  LENGTH(content) as chunk_size,
  LENGTH(content_l1) as l1_context_size,
  LENGTH(content_l2) as l2_context_size,
  CASE 
    WHEN LENGTH(content_l1) > LENGTH(content) * 2 THEN 'Rich Context'
    WHEN LENGTH(content_l1) > LENGTH(content) THEN 'Good Context'
    ELSE 'Limited Context'
  END as context_quality
FROM "INGEST_20250928160000_500"
WHERE filepath LIKE '%compiler%'
ORDER BY filepath, chunk_number
LIMIT 20;
```

### Cross-Chunk Pattern Analysis

```sql
-- Find patterns that span multiple chunks
WITH chunk_patterns AS (
  SELECT 
    filepath,
    chunk_number,
    CASE 
      WHEN content LIKE '%impl %' AND content LIKE '%{%' AND content NOT LIKE '%}%' THEN 'impl_start'
      WHEN content LIKE '%}%' AND content NOT LIKE '%impl %' THEN 'impl_end'
      WHEN content LIKE '%fn %' AND content LIKE '%{%' AND content NOT LIKE '%}%' THEN 'fn_start'
      WHEN content LIKE '%struct %' THEN 'struct_def'
      WHEN content LIKE '%enum %' THEN 'enum_def'
      ELSE 'other'
    END as pattern_type
  FROM "INGEST_20250928160000_500"
  WHERE filepath LIKE '%compiler%'
)
SELECT 
  filepath,
  pattern_type,
  COUNT(*) as occurrence_count,
  array_agg(chunk_number ORDER BY chunk_number) as chunk_numbers
FROM chunk_patterns
WHERE pattern_type != 'other'
GROUP BY filepath, pattern_type
ORDER BY filepath, occurrence_count DESC;
```

### Complexity Distribution Across Chunks

```sql
-- Analyze complexity distribution within files
SELECT 
  filepath,
  chunk_number,
  chunk_start_line,
  chunk_end_line,
  -- Complexity indicators
  (LENGTH(content) - LENGTH(REPLACE(content, 'fn ', ''))) / 3 as function_count,
  (LENGTH(content) - LENGTH(REPLACE(content, 'impl ', ''))) / 5 as impl_count,
  (LENGTH(content) - LENGTH(REPLACE(content, 'match ', ''))) / 6 as match_count,
  (LENGTH(content) - LENGTH(REPLACE(content, 'unsafe ', ''))) / 7 as unsafe_count
FROM "INGEST_20250928160000_500"
WHERE filepath IN (
  SELECT filepath 
  FROM "INGEST_20250928160000_500" 
  GROUP BY filepath 
  HAVING COUNT(*) > 5
)
ORDER BY filepath, chunk_number;
```

## Step 5: Create Specialized Analysis Prompts

### Compiler-Specific Analysis Prompt

```bash
mkdir -p .kiro/steering
cat > .kiro/steering/rust-compiler-analysis.md << 'EOF'
# Rust Compiler Chunked Analysis

Analyze the provided code chunk with focus on compiler internals:

## L1: Code Structure Analysis (Current Chunk)
- Identify the primary purpose of this code section
- Analyze function signatures and their roles
- Evaluate data structures and their relationships
- Note any compiler-specific patterns or idioms

## L2: Contextual Understanding (L1 Context)
- How does this chunk relate to surrounding code?
- Identify dependencies and data flow
- Analyze the broader algorithmic approach
- Note any cross-chunk patterns or state management

## L3: Architectural Insights (L2 Context)
- Understand the module's role in the compiler pipeline
- Identify design patterns and architectural decisions
- Analyze performance considerations and optimizations
- Evaluate error handling and edge case management

## L4: Compiler Engineering Insights
- Assess code generation strategies
- Identify optimization opportunities
- Evaluate maintainability and extensibility
- Note innovative approaches or clever solutions

## Specific Focus Areas:
- **AST Processing**: How abstract syntax trees are manipulated
- **Type System**: Type checking and inference mechanisms
- **Code Generation**: Translation to lower-level representations
- **Error Reporting**: User-facing error message generation
- **Performance**: Compilation speed and memory usage optimizations

Provide specific examples from the code and explain the engineering rationale.
EOF
```

### Performance-Focused Analysis Prompt

```bash
cat > .kiro/steering/performance-analysis.md << 'EOF'
# Performance-Focused Chunked Analysis

Analyze code chunks for performance characteristics:

## Memory Usage Analysis
- Identify allocation patterns and memory hotspots
- Evaluate use of collections and their efficiency
- Analyze lifetime management and borrowing patterns
- Note potential memory leaks or excessive cloning

## Computational Complexity
- Assess algorithmic complexity of key functions
- Identify nested loops and recursive patterns
- Evaluate data structure access patterns
- Note potential optimization opportunities

## Concurrency and Parallelism
- Identify thread-safe vs thread-unsafe code
- Analyze synchronization primitives usage
- Evaluate async/await patterns and efficiency
- Note potential race conditions or deadlocks

## I/O and System Interactions
- Analyze file system and network operations
- Evaluate caching strategies and effectiveness
- Identify blocking vs non-blocking operations
- Note resource management patterns

Focus on measurable performance impacts and provide specific optimization recommendations.
EOF
```

## Step 6: Execute Chunked Analysis Workflow

### Generate Content Files with Context

The chunked analysis automatically generates three types of content files for each chunk:

```bash
# Example for chunk 1 of a large file
ls -la .raw_data_202509/INGEST_20250928160000_500_1_*

# Content files structure:
# INGEST_20250928160000_500_1_Content.txt     - Current chunk (A)
# INGEST_20250928160000_500_1_Content_L1.txt  - ±1 chunk context (B)
# INGEST_20250928160000_500_1_Content_L2.txt  - ±2 chunk context (C)
```

### Analyze Context Effectiveness

```sql
-- Measure context window effectiveness
SELECT 
  filepath,
  chunk_number,
  LENGTH(content) as base_size,
  LENGTH(content_l1) - LENGTH(content) as l1_additional,
  LENGTH(content_l2) - LENGTH(content_l1) as l2_additional,
  ROUND(
    (LENGTH(content_l1)::FLOAT / LENGTH(content) - 1) * 100, 2
  ) as l1_expansion_percent,
  ROUND(
    (LENGTH(content_l2)::FLOAT / LENGTH(content_l1) - 1) * 100, 2
  ) as l2_expansion_percent
FROM "INGEST_20250928160000_500"
WHERE filepath LIKE '%rustc_middle%'
ORDER BY l1_expansion_percent DESC
LIMIT 10;
```

## Step 7: Advanced Chunked Queries

### Find Related Chunks Across Files

```sql
-- Find chunks that reference similar concepts
WITH concept_chunks AS (
  SELECT 
    filepath,
    chunk_number,
    content,
    CASE 
      WHEN content LIKE '%TypeckResults%' THEN 'type_checking'
      WHEN content LIKE '%HIR%' OR content LIKE '%hir::%' THEN 'hir_processing'
      WHEN content LIKE '%MIR%' OR content LIKE '%mir::%' THEN 'mir_processing'
      WHEN content LIKE '%LLVM%' OR content LIKE '%codegen%' THEN 'code_generation'
      WHEN content LIKE '%diagnostic%' OR content LIKE '%error%' THEN 'error_handling'
      ELSE 'other'
    END as concept
  FROM "INGEST_20250928160000_500"
  WHERE content ~ '(TypeckResults|HIR|MIR|LLVM|codegen|diagnostic)'
)
SELECT 
  concept,
  COUNT(*) as chunk_count,
  COUNT(DISTINCT filepath) as file_count,
  array_agg(DISTINCT SPLIT_PART(filepath, '/', 2)) as modules
FROM concept_chunks
WHERE concept != 'other'
GROUP BY concept
ORDER BY chunk_count DESC;
```

### Analyze Function Boundaries Across Chunks

```sql
-- Find functions that span multiple chunks
WITH function_analysis AS (
  SELECT 
    filepath,
    chunk_number,
    chunk_start_line,
    chunk_end_line,
    (LENGTH(content) - LENGTH(REPLACE(content, 'fn ', ''))) / 3 as fn_starts,
    (LENGTH(content) - LENGTH(REPLACE(content, '}', ''))) as brace_closes,
    content LIKE '%fn %' as has_fn_start,
    content LIKE '%{%' as has_open_brace,
    content LIKE '%}%' as has_close_brace
  FROM "INGEST_20250928160000_500"
)
SELECT 
  filepath,
  chunk_number,
  chunk_start_line,
  chunk_end_line,
  CASE 
    WHEN has_fn_start AND has_open_brace AND NOT has_close_brace THEN 'Function Start'
    WHEN NOT has_fn_start AND has_close_brace THEN 'Function End'
    WHEN has_fn_start AND has_open_brace AND has_close_brace THEN 'Complete Function'
    ELSE 'Function Body'
  END as function_boundary_type
FROM function_analysis
WHERE has_fn_start OR has_close_brace
ORDER BY filepath, chunk_number;
```

### Performance Hotspot Analysis

```sql
-- Identify performance-critical chunks
SELECT 
  filepath,
  chunk_number,
  chunk_start_line,
  chunk_end_line,
  -- Performance indicators
  (LENGTH(content) - LENGTH(REPLACE(content, 'Vec::', ''))) / 5 as vec_usage,
  (LENGTH(content) - LENGTH(REPLACE(content, 'HashMap', ''))) / 7 as hashmap_usage,
  (LENGTH(content) - LENGTH(REPLACE(content, 'clone()', ''))) / 7 as clone_usage,
  (LENGTH(content) - LENGTH(REPLACE(content, 'collect()', ''))) / 9 as collect_usage,
  -- Complexity indicators
  (LENGTH(content) - LENGTH(REPLACE(content, 'for ', ''))) / 4 as loop_count,
  (LENGTH(content) - LENGTH(REPLACE(content, 'match ', ''))) / 6 as match_count
FROM "INGEST_20250928160000_500"
WHERE filepath LIKE '%compiler%'
  AND (content LIKE '%Vec::%' 
       OR content LIKE '%HashMap%' 
       OR content LIKE '%clone()%'
       OR content LIKE '%collect()%')
ORDER BY (vec_usage + hashmap_usage + clone_usage + collect_usage) DESC
LIMIT 20;
```

## Step 8: Comparative Analysis Between Chunk Sizes

### Compare 300 vs 500 Line Chunks

```sql
-- Compare chunk granularity effectiveness
WITH chunk_comparison AS (
  SELECT 
    '300_lines' as chunk_type,
    COUNT(*) as total_chunks,
    AVG(chunk_end_line - chunk_start_line + 1) as avg_chunk_size,
    COUNT(DISTINCT filepath) as files_chunked
  FROM "INGEST_20250928160000_300"
  
  UNION ALL
  
  SELECT 
    '500_lines' as chunk_type,
    COUNT(*) as total_chunks,
    AVG(chunk_end_line - chunk_start_line + 1) as avg_chunk_size,
    COUNT(DISTINCT filepath) as files_chunked
  FROM "INGEST_20250928160000_500"
)
SELECT * FROM chunk_comparison;

-- Find optimal chunk size for specific files
SELECT 
  c300.filepath,
  COUNT(c300.chunk_number) as chunks_300,
  COUNT(c500.chunk_number) as chunks_500,
  CASE 
    WHEN COUNT(c300.chunk_number) < COUNT(c500.chunk_number) * 2 THEN 'Prefer 500-line chunks'
    WHEN COUNT(c300.chunk_number) > COUNT(c500.chunk_number) * 3 THEN 'Prefer 300-line chunks'
    ELSE 'Either size works'
  END as recommendation
FROM "INGEST_20250928160000_300" c300
FULL OUTER JOIN "INGEST_20250928160000_500" c500 ON c300.filepath = c500.filepath
GROUP BY c300.filepath
ORDER BY chunks_300 DESC
LIMIT 15;
```

## Step 9: Export and Reporting

### Generate Chunked Analysis Report

```bash
# Export chunk analysis summary
code-ingest export INGEST_20250928160000_500 \
  --query "
    SELECT 
      filepath,
      COUNT(*) as chunk_count,
      MIN(chunk_start_line) as start_line,
      MAX(chunk_end_line) as end_line,
      AVG(LENGTH(content)) as avg_chunk_size,
      AVG(LENGTH(content_l1)) as avg_l1_size,
      AVG(LENGTH(content_l2)) as avg_l2_size
    FROM INGEST_20250928160000_500
    GROUP BY filepath
    ORDER BY chunk_count DESC
  " \
  --format csv \
  --output chunked_analysis_summary.csv \
  --db-path /Users/username/desktop/RustCompilerDB
```

### Create Performance Hotspot Report

```bash
# Export performance analysis
code-ingest export INGEST_20250928160000_500 \
  --query "
    SELECT 
      filepath,
      chunk_number,
      chunk_start_line,
      chunk_end_line,
      (LENGTH(content) - LENGTH(REPLACE(content, 'clone()', ''))) / 7 as clone_usage,
      (LENGTH(content) - LENGTH(REPLACE(content, 'Vec::', ''))) / 5 as vec_usage,
      (LENGTH(content) - LENGTH(REPLACE(content, 'HashMap', ''))) / 7 as hashmap_usage
    FROM INGEST_20250928160000_500
    WHERE (LENGTH(content) - LENGTH(REPLACE(content, 'clone()', ''))) > 0
       OR (LENGTH(content) - LENGTH(REPLACE(content, 'Vec::', ''))) > 0
       OR (LENGTH(content) - LENGTH(REPLACE(content, 'HashMap', ''))) > 0
    ORDER BY (clone_usage + vec_usage + hashmap_usage) DESC
  " \
  --format json \
  --output performance_hotspots.json \
  --db-path /Users/username/desktop/RustCompilerDB
```

## Step 10: Automated Chunked Analysis Pipeline

### Create Analysis Pipeline Script

```bash
cat > chunked_analysis_pipeline.sh << 'EOF'
#!/bin/bash

REPO_URL="$1"
DB_PATH="$2"
CHUNK_SIZE="${3:-500}"
ANALYSIS_NAME="${4:-analysis}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <repo_url> <db_path> [chunk_size] [analysis_name]"
    exit 1
fi

echo "Starting chunked analysis pipeline..."
echo "Repository: $REPO_URL"
echo "Database: $DB_PATH"
echo "Chunk Size: $CHUNK_SIZE"

# Step 1: Ingest repository
echo "Step 1: Ingesting repository..."
./target/release/code-ingest ingest "$REPO_URL" \
  --db-path "$DB_PATH" \
  --include "*.rs,*.md,*.toml" \
  --exclude "tests/*,target/*,*.lock" \
  --max-concurrency 16

# Step 2: Get latest table name
LATEST_TABLE=$(psql -d "$(basename "$DB_PATH")" -t -c "
  SELECT table_name 
  FROM ingestion_meta 
  ORDER BY created_at DESC 
  LIMIT 1
" | xargs)

echo "Latest table: $LATEST_TABLE"

# Step 3: Analyze file sizes and recommend chunking
echo "Step 2: Analyzing file sizes..."
LARGE_FILES=$(psql -d "$(basename "$DB_PATH")" -t -c "
  SELECT COUNT(*) 
  FROM \"$LATEST_TABLE\" 
  WHERE line_count > $CHUNK_SIZE
" | xargs)

echo "Found $LARGE_FILES files requiring chunking"

if [ "$LARGE_FILES" -gt 0 ]; then
    # Step 4: Generate chunked analysis
    echo "Step 3: Generating chunked analysis..."
    code-ingest generate-hierarchical-tasks "$LATEST_TABLE" \
      --chunks "$CHUNK_SIZE" \
      --levels 4 \
      --groups 8 \
      --output "${ANALYSIS_NAME}_chunked_${CHUNK_SIZE}.md" \
      --db-path "$DB_PATH"
    
    # Step 5: Generate analysis reports
    echo "Step 4: Generating reports..."
    
    # Chunk distribution report
    psql -d "$(basename "$DB_PATH")" -c "
      COPY (
        SELECT 
          filepath,
          COUNT(*) as chunk_count,
          MIN(chunk_start_line) as start_line,
          MAX(chunk_end_line) as end_line
        FROM \"${LATEST_TABLE}_${CHUNK_SIZE}\"
        GROUP BY filepath
        ORDER BY chunk_count DESC
      ) TO STDOUT WITH CSV HEADER
    " > "${ANALYSIS_NAME}_chunk_distribution.csv"
    
    echo "Chunked analysis complete!"
    echo "Task file: ${ANALYSIS_NAME}_chunked_${CHUNK_SIZE}.md"
    echo "Report: ${ANALYSIS_NAME}_chunk_distribution.csv"
else
    echo "No files require chunking. Use standard analysis instead."
fi
EOF

chmod +x chunked_analysis_pipeline.sh
```

### Usage Examples

```bash
# Analyze Rust compiler with 500-line chunks
./chunked_analysis_pipeline.sh \
  "https://github.com/rust-lang/rust" \
  "/data/rust_compiler_db" \
  500 \
  "rust_compiler"

# Analyze Tokio with 300-line chunks
./chunked_analysis_pipeline.sh \
  "https://github.com/tokio-rs/tokio" \
  "/data/tokio_db" \
  300 \
  "tokio_analysis"

# Analyze local large project
./chunked_analysis_pipeline.sh \
  "/path/to/large/project" \
  "/data/local_project_db" \
  400 \
  "local_project"
```

## Expected Outcomes

After completing this chunked analysis workflow, you should have:

1. **Intelligent file chunking** with context preservation
2. **Multi-granularity analysis** (300 and 500-line chunks)
3. **Context-aware content files** with L1/L2 windows
4. **Performance hotspot identification** across chunks
5. **Cross-chunk pattern analysis** for architectural insights
6. **Automated pipeline** for repeatable analysis
7. **Comparative analysis** between different chunk sizes

## Performance Considerations

### Memory Usage

- **500-line chunks**: ~2-3x memory usage vs standard analysis
- **300-line chunks**: ~3-4x memory usage vs standard analysis
- **Context windows**: Additional 2-5x storage for L1/L2 content

### Processing Time

- **Chunking overhead**: 20-40% additional processing time
- **Context generation**: 30-60% additional time for L1/L2 creation
- **Query performance**: Minimal impact with proper indexing

### Storage Requirements

- **Base ingestion**: 10-50MB for typical repository
- **Chunked tables**: 3-5x base size with context
- **Content files**: 2-4x base size for generated files

## Best Practices

1. **Choose appropriate chunk size**: 300-500 lines for most code
2. **Monitor memory usage**: Use streaming for very large repositories
3. **Leverage context windows**: L1 for local context, L2 for architectural understanding
4. **Focus on large files**: Only chunk files >500 lines
5. **Use specialized prompts**: Tailor analysis to chunk granularity
6. **Compare chunk sizes**: Test different sizes for optimal analysis depth

This chunked analysis workflow enables deep understanding of large codebases while maintaining contextual awareness and systematic analysis structure.