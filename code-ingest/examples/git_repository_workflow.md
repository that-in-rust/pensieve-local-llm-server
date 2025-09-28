# Git Repository Analysis Workflow

This example demonstrates a complete workflow for analyzing a GitHub repository using code-ingest.

## Overview

We'll analyze the popular `serde` Rust serialization library to understand its architecture, identify patterns, and generate systematic analysis tasks.

## Prerequisites

- PostgreSQL running locally
- code-ingest installed
- Internet connection for GitHub access

## Step 1: Repository Ingestion

### Basic Ingestion

```bash
# Ingest the serde repository
./target/release/code-ingest ingest https://github.com/serde-rs/serde \
  --db-path /Users/username/desktop/SerdeAnalysisDB

# Check ingestion results
code-ingest list --db-path /Users/username/desktop/SerdeAnalysisDB
```

Expected output:
```
Ingestion ID: 1
Repository: https://github.com/serde-rs/serde
Table: INGEST_20250928143022
Files Processed: 247
Start Time: 2025-09-28 14:30:22 UTC
Duration: 45 seconds
```

### Advanced Ingestion with Filtering

```bash
# Ingest only Rust source files and documentation
./target/release/code-ingest ingest https://github.com/serde-rs/serde \
  --db-path /Users/username/desktop/SerdeAnalysisDB \
  --include "*.rs,*.md,*.toml" \
  --exclude "target/*,*.lock,tests/fixtures/*" \
  --max-concurrency 8
```

## Step 2: Basic Analysis Queries

### Repository Overview

```sql
-- Connect to database
psql postgresql://localhost:5432/SerdeAnalysisDB

-- File type distribution
SELECT 
  file_type,
  extension,
  COUNT(*) as file_count,
  AVG(line_count) as avg_lines,
  SUM(file_size_bytes) as total_bytes
FROM "INGEST_20250928143022" 
GROUP BY file_type, extension
ORDER BY file_count DESC;
```

Expected results:
```
 file_type   | extension | file_count | avg_lines | total_bytes
-------------+-----------+------------+-----------+-------------
 direct_text | rs        |        156 |       245 |     1847392
 direct_text | md        |         23 |        87 |      156743
 direct_text | toml      |         12 |        34 |       23847
 direct_text | yml       |          8 |        45 |       12983
```

### Code Complexity Analysis

```sql
-- Find the most complex Rust files
SELECT 
  filepath,
  line_count,
  word_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'fn ', ''))) / 3 as function_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'impl ', ''))) / 5 as impl_count
FROM "INGEST_20250928143022" 
WHERE extension = 'rs' 
  AND line_count > 100
ORDER BY line_count DESC
LIMIT 10;
```

### Pattern Analysis

```sql
-- Find serialization patterns
SELECT 
  filepath,
  line_count,
  CASE 
    WHEN content_text LIKE '%#[derive(Serialize%' THEN 'Serializable'
    WHEN content_text LIKE '%#[derive(Deserialize%' THEN 'Deserializable'
    WHEN content_text LIKE '%Serializer%' THEN 'Serializer'
    WHEN content_text LIKE '%Deserializer%' THEN 'Deserializer'
    ELSE 'Other'
  END as pattern_type
FROM "INGEST_20250928143022" 
WHERE extension = 'rs'
  AND (content_text LIKE '%Serialize%' OR content_text LIKE '%Deserialize%')
ORDER BY pattern_type, line_count DESC;
```

## Step 3: Generate Hierarchical Analysis Tasks

### Basic Task Generation

```bash
# Generate analysis tasks for systematic review
code-ingest generate-hierarchical-tasks INGEST_20250928143022 \
  --levels 4 \
  --groups 7 \
  --output serde_analysis_tasks.md \
  --db-path /Users/username/desktop/SerdeAnalysisDB
```

This creates a structured task list:

```markdown
# Serde Repository Analysis Tasks

- [ ] 1. Analyze INGEST_20250928143022 row 1
  - **Content**: `.raw_data_202509/INGEST_20250928143022_1_Content.txt` as A
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md`
  - **Output**: `gringotts/WorkArea/INGEST_20250928143022_1.md`

- [ ] 1.1. Analyze INGEST_20250928143022 row 2
  - **Content**: `.raw_data_202509/INGEST_20250928143022_2_Content.txt` as A
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md`
  - **Output**: `gringotts/WorkArea/INGEST_20250928143022_2.md`
```

### Advanced Chunked Analysis

For large files in the repository:

```bash
# Generate chunked analysis for detailed review
code-ingest generate-hierarchical-tasks INGEST_20250928143022 \
  --chunks 300 \
  --levels 3 \
  --groups 5 \
  --prompt-file .kiro/steering/serde-analysis-prompt.md \
  --output serde_chunked_tasks.md \
  --db-path /Users/username/desktop/SerdeAnalysisDB
```

This creates the chunked table `INGEST_20250928143022_300` and generates tasks with L1/L2 context:

```markdown
- [ ] 1. Analyze INGEST_20250928143022_300 row 1
  - **Content**: 
    - A: `.raw_data_202509/INGEST_20250928143022_300_1_Content.txt`
    - B: `.raw_data_202509/INGEST_20250928143022_300_1_Content_L1.txt` 
    - C: `.raw_data_202509/INGEST_20250928143022_300_1_Content_L2.txt`
  - **Prompt**: `.kiro/steering/serde-analysis-prompt.md`
  - **Output**: `gringotts/WorkArea/INGEST_20250928143022_300_1.md`
```

## Step 4: Custom Analysis Prompt

Create a specialized analysis prompt for serde:

```bash
mkdir -p .kiro/steering
cat > .kiro/steering/serde-analysis-prompt.md << 'EOF'
# Serde Architecture Analysis

Analyze the provided code with focus on:

## L1: Implementation Patterns
- Serialization/deserialization trait implementations
- Error handling strategies
- Generic type usage and bounds
- Macro usage and code generation

## L2: Design Decisions
- API design principles
- Performance optimizations
- Memory management strategies
- Backward compatibility considerations

## L3: Architecture Insights
- Module organization and dependencies
- Abstraction layers and interfaces
- Extension points and customization
- Integration with Rust ecosystem

## L4: Strategic Observations
- Innovation in serialization approaches
- Lessons for library design
- Patterns applicable to other domains
- Evolution of Rust idioms

Provide specific examples from the code and explain the reasoning behind design choices.
EOF
```

## Step 5: Execute Analysis Workflow

### Using Kiro IDE Integration

1. Open the generated task file in Kiro:
   ```bash
   code serde_analysis_tasks.md
   ```

2. Click "Start task" next to each task item to execute the analysis

3. Review generated analysis files in `gringotts/WorkArea/`

### Manual Analysis Execution

For each task, the content files are automatically generated:

```bash
# Example content files for task 1
ls -la .raw_data_202509/INGEST_20250928143022_1_*

# View the content
cat .raw_data_202509/INGEST_20250928143022_1_Content.txt
```

## Step 6: Advanced Queries and Insights

### Architectural Analysis

```sql
-- Analyze module structure
SELECT 
  SPLIT_PART(filepath, '/', 1) as module,
  SPLIT_PART(filepath, '/', 2) as submodule,
  COUNT(*) as file_count,
  AVG(line_count) as avg_complexity
FROM "INGEST_20250928143022" 
WHERE extension = 'rs'
GROUP BY SPLIT_PART(filepath, '/', 1), SPLIT_PART(filepath, '/', 2)
ORDER BY file_count DESC;
```

### Dependency Analysis

```sql
-- Find external dependencies
SELECT 
  filepath,
  COUNT(*) as use_statements
FROM "INGEST_20250928143022" 
WHERE extension = 'rs'
  AND content_text ~ 'use [a-z_]+::'
GROUP BY filepath
ORDER BY use_statements DESC;
```

### Error Handling Patterns

```sql
-- Analyze error handling approaches
SELECT 
  filepath,
  line_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Result<', ''))) / 8 as result_usage,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Option<', ''))) / 8 as option_usage,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, '?', ''))) as error_propagation
FROM "INGEST_20250928143022" 
WHERE extension = 'rs'
  AND (content_text LIKE '%Result<%' OR content_text LIKE '%Option<%')
ORDER BY result_usage DESC;
```

## Step 7: Export and Documentation

### Export Analysis Results

```bash
# Export key findings to CSV
code-ingest export INGEST_20250928143022 \
  --query "SELECT filepath, line_count, file_type FROM INGEST_20250928143022 WHERE extension = 'rs'" \
  --format csv \
  --output serde_rust_files.csv \
  --db-path /Users/username/desktop/SerdeAnalysisDB

# Export full content for specific files
code-ingest export INGEST_20250928143022 \
  --query "SELECT filepath, content_text FROM INGEST_20250928143022 WHERE filepath LIKE '%lib.rs'" \
  --format json \
  --output serde_lib_files.json \
  --db-path /Users/username/desktop/SerdeAnalysisDB
```

### Generate Summary Report

```sql
-- Create comprehensive summary
SELECT 
  'Repository Summary' as metric,
  COUNT(*) as value
FROM "INGEST_20250928143022"
UNION ALL
SELECT 
  'Total Lines of Code',
  SUM(line_count)
FROM "INGEST_20250928143022" 
WHERE extension = 'rs'
UNION ALL
SELECT 
  'Average File Size (lines)',
  AVG(line_count)::INTEGER
FROM "INGEST_20250928143022" 
WHERE extension = 'rs'
UNION ALL
SELECT 
  'Largest File (lines)',
  MAX(line_count)
FROM "INGEST_20250928143022" 
WHERE extension = 'rs';
```

## Expected Outcomes

After completing this workflow, you should have:

1. **Complete repository ingestion** with 200+ files processed
2. **Structured analysis tasks** for systematic code review
3. **Chunked analysis** for large files with contextual information
4. **Architectural insights** from SQL queries
5. **Exportable data** for further analysis or reporting
6. **Generated content files** ready for LLM analysis

## Performance Metrics

Typical performance for the serde repository:

- **Ingestion time**: 30-60 seconds
- **Files processed**: ~250 files
- **Database size**: ~15MB
- **Task generation**: <5 seconds
- **Query response time**: <100ms for most queries

## Next Steps

1. **Execute the generated tasks** using Kiro or manual analysis
2. **Analyze patterns** across multiple repositories for comparison
3. **Create custom queries** for specific research questions
4. **Export findings** for documentation or presentation
5. **Iterate on analysis prompts** to improve insight quality

This workflow provides a systematic approach to understanding any Rust codebase and can be adapted for other programming languages and analysis goals.