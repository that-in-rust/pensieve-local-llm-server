# Local Folder Analysis Workflow

This example demonstrates analyzing a local codebase using code-ingest's folder ingestion capabilities.

## Overview

We'll analyze a local Rust project to understand its structure, identify technical debt, and generate systematic refactoring tasks.

## Prerequisites

- PostgreSQL running locally
- code-ingest installed
- A local Rust project to analyze

## Step 1: Prepare Local Project

For this example, we'll use a hypothetical local project structure:

```
/Users/dev/my-rust-project/
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── models/
│   │   ├── mod.rs
│   │   ├── user.rs
│   │   └── product.rs
│   ├── services/
│   │   ├── mod.rs
│   │   ├── auth.rs
│   │   └── database.rs
│   └── utils/
│       ├── mod.rs
│       └── helpers.rs
├── tests/
│   ├── integration_tests.rs
│   └── unit_tests.rs
├── docs/
│   ├── README.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── Cargo.toml
├── Cargo.lock
└── .gitignore
```

## Step 2: Local Folder Ingestion

### Basic Local Ingestion

```bash
# Ingest the entire local project
./target/release/code-ingest ingest /Users/dev/my-rust-project \
  --folder-flag Y \
  --db-path /Users/dev/desktop/LocalProjectDB
```

Expected output:
```
Starting local folder ingestion...
Processing directory: /Users/dev/my-rust-project
Found 47 files to process
Ingestion completed successfully
Table created: INGEST_20250928150000
Files processed: 47
Duration: 12 seconds
```

### Selective Ingestion with Filters

```bash
# Ingest only source code and documentation
./target/release/code-ingest ingest /Users/dev/my-rust-project \
  --folder-flag Y \
  --db-path /Users/dev/desktop/LocalProjectDB \
  --include "*.rs,*.md,*.toml" \
  --exclude "target/*,*.lock,.git/*,node_modules/*"
```

### Performance-Optimized Ingestion

```bash
# High-performance ingestion for large codebases
./target/release/code-ingest ingest /Users/dev/my-rust-project \
  --folder-flag Y \
  --db-path /Users/dev/desktop/LocalProjectDB \
  --max-concurrency 12 \
  --batch-size 2000 \
  --enable-monitoring
```

## Step 3: Project Structure Analysis

### Directory Structure Overview

```sql
-- Connect to the database
psql postgresql://localhost:5432/LocalProjectDB

-- Analyze directory structure
SELECT 
  CASE 
    WHEN filepath LIKE 'src/%' THEN 'Source Code'
    WHEN filepath LIKE 'tests/%' THEN 'Tests'
    WHEN filepath LIKE 'docs/%' THEN 'Documentation'
    WHEN filepath LIKE 'examples/%' THEN 'Examples'
    WHEN filepath IN ('Cargo.toml', 'Cargo.lock', '.gitignore', 'README.md') THEN 'Project Root'
    ELSE 'Other'
  END as category,
  COUNT(*) as file_count,
  AVG(line_count) as avg_lines,
  SUM(file_size_bytes) as total_bytes
FROM "INGEST_20250928150000"
GROUP BY category
ORDER BY file_count DESC;
```

Expected results:
```
   category    | file_count | avg_lines | total_bytes
---------------+------------+-----------+-------------
 Source Code   |         23 |       156 |      234567
 Tests         |         12 |        89 |       67890
 Documentation |          8 |        45 |       23456
 Project Root  |          4 |        67 |       12345
```

### Module Complexity Analysis

```sql
-- Analyze module complexity
SELECT 
  SPLIT_PART(filepath, '/', 2) as module,
  COUNT(*) as file_count,
  AVG(line_count) as avg_complexity,
  MAX(line_count) as max_complexity,
  SUM(line_count) as total_lines
FROM "INGEST_20250928150000" 
WHERE filepath LIKE 'src/%' 
  AND extension = 'rs'
GROUP BY SPLIT_PART(filepath, '/', 2)
ORDER BY total_lines DESC;
```

### Code Quality Metrics

```sql
-- Identify potential technical debt
SELECT 
  filepath,
  line_count,
  word_count,
  -- Calculate complexity indicators
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'TODO', ''))) / 4 as todo_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'FIXME', ''))) / 5 as fixme_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'unwrap()', ''))) / 8 as unwrap_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'clone()', ''))) / 7 as clone_count
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
  AND (content_text LIKE '%TODO%' 
       OR content_text LIKE '%FIXME%' 
       OR content_text LIKE '%unwrap()%'
       OR content_text LIKE '%clone()%')
ORDER BY (todo_count + fixme_count + unwrap_count) DESC;
```

## Step 4: Generate Refactoring Tasks

### Basic Task Generation for Code Review

```bash
# Generate systematic code review tasks
code-ingest generate-hierarchical-tasks INGEST_20250928150000 \
  --levels 3 \
  --groups 6 \
  --output local_project_review_tasks.md \
  --db-path /Users/dev/desktop/LocalProjectDB
```

### Focused Analysis with Custom Prompt

Create a refactoring-focused analysis prompt:

```bash
mkdir -p .kiro/steering
cat > .kiro/steering/refactoring-analysis-prompt.md << 'EOF'
# Local Project Refactoring Analysis

Analyze the provided code for refactoring opportunities:

## Code Quality Assessment
- Identify code smells and anti-patterns
- Evaluate error handling strategies
- Assess function and module complexity
- Review naming conventions and clarity

## Technical Debt Analysis
- Find TODO/FIXME comments and their context
- Identify overuse of unwrap() and clone()
- Locate duplicated code patterns
- Assess test coverage gaps

## Architecture Improvements
- Evaluate module organization and dependencies
- Identify opportunities for better abstraction
- Suggest design pattern applications
- Recommend performance optimizations

## Refactoring Recommendations
- Prioritize refactoring tasks by impact and effort
- Suggest specific code improvements
- Identify breaking change considerations
- Recommend incremental refactoring steps

Focus on actionable improvements that enhance maintainability and performance.
EOF
```

### Generate Chunked Analysis for Large Files

```bash
# Generate detailed analysis for complex files
code-ingest generate-hierarchical-tasks INGEST_20250928150000 \
  --chunks 200 \
  --levels 3 \
  --groups 4 \
  --prompt-file .kiro/steering/refactoring-analysis-prompt.md \
  --output local_project_detailed_tasks.md \
  --db-path /Users/dev/desktop/LocalProjectDB
```

## Step 5: Specialized Analysis Queries

### Dependency Analysis

```sql
-- Analyze internal dependencies
SELECT 
  filepath,
  COUNT(*) as internal_imports
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
  AND content_text ~ 'use crate::'
GROUP BY filepath
ORDER BY internal_imports DESC;

-- Find external dependencies
SELECT 
  REGEXP_REPLACE(
    SUBSTRING(content_text FROM 'use ([a-z_][a-z0-9_]*)::', 1, 1), 
    '::.*', ''
  ) as external_crate,
  COUNT(*) as usage_count
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
  AND content_text ~ 'use [a-z_][a-z0-9_]*::'
  AND content_text !~ 'use (std|crate|super|self)::'
GROUP BY external_crate
ORDER BY usage_count DESC;
```

### Test Coverage Analysis

```sql
-- Analyze test distribution
SELECT 
  CASE 
    WHEN filepath LIKE 'tests/%' THEN 'Integration Tests'
    WHEN filepath LIKE 'src/%' AND content_text LIKE '%#[test]%' THEN 'Unit Tests'
    WHEN filepath LIKE 'src/%' AND content_text LIKE '%#[cfg(test)]%' THEN 'Inline Tests'
    ELSE 'No Tests'
  END as test_type,
  COUNT(*) as file_count,
  AVG(line_count) as avg_lines
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
GROUP BY test_type
ORDER BY file_count DESC;

-- Find untested modules
SELECT 
  filepath,
  line_count,
  CASE 
    WHEN content_text LIKE '%#[test]%' OR content_text LIKE '%#[cfg(test)]%' THEN 'Has Tests'
    ELSE 'No Tests'
  END as test_status
FROM "INGEST_20250928150000" 
WHERE filepath LIKE 'src/%' 
  AND extension = 'rs'
  AND filepath NOT LIKE '%/mod.rs'
ORDER BY test_status, line_count DESC;
```

### Error Handling Assessment

```sql
-- Analyze error handling patterns
SELECT 
  filepath,
  line_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Result<', ''))) / 8 as result_usage,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'unwrap()', ''))) / 8 as unwrap_usage,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'expect(', ''))) / 8 as expect_usage,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, '?', ''))) as error_propagation
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
  AND filepath LIKE 'src/%'
ORDER BY unwrap_usage DESC;
```

## Step 6: Performance and Security Analysis

### Performance Hotspots

```sql
-- Identify potential performance issues
SELECT 
  filepath,
  line_count,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'clone()', ''))) / 7 as clone_usage,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'String::from', ''))) / 12 as string_alloc,
  (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Vec::new()', ''))) / 10 as vec_alloc,
  CASE 
    WHEN content_text LIKE '%async%' AND content_text LIKE '%await%' THEN 'Async'
    ELSE 'Sync'
  END as async_usage
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
  AND filepath LIKE 'src/%'
ORDER BY (clone_usage + string_alloc + vec_alloc) DESC;
```

### Security Considerations

```sql
-- Find potential security issues
SELECT 
  filepath,
  line_count,
  CASE 
    WHEN content_text LIKE '%unsafe%' THEN 'Uses Unsafe Code'
    WHEN content_text LIKE '%std::process::Command%' THEN 'System Commands'
    WHEN content_text LIKE '%std::fs::%' THEN 'File System Access'
    WHEN content_text LIKE '%std::net::%' THEN 'Network Operations'
    WHEN content_text LIKE '%password%' OR content_text LIKE '%secret%' THEN 'Credential Handling'
    ELSE 'Standard Code'
  END as security_category
FROM "INGEST_20250928150000" 
WHERE extension = 'rs'
  AND (content_text LIKE '%unsafe%' 
       OR content_text LIKE '%Command%'
       OR content_text LIKE '%std::fs::%'
       OR content_text LIKE '%std::net::%'
       OR content_text LIKE '%password%'
       OR content_text LIKE '%secret%')
ORDER BY security_category, line_count DESC;
```

## Step 7: Generate Improvement Roadmap

### Export Technical Debt Report

```bash
# Export technical debt analysis
code-ingest export INGEST_20250928150000 \
  --query "
    SELECT 
      filepath,
      line_count,
      (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'TODO', ''))) / 4 as todo_count,
      (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'FIXME', ''))) / 5 as fixme_count,
      (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'unwrap()', ''))) / 8 as unwrap_count
    FROM INGEST_20250928150000 
    WHERE extension = 'rs'
    ORDER BY (todo_count + fixme_count + unwrap_count) DESC
  " \
  --format csv \
  --output technical_debt_report.csv \
  --db-path /Users/dev/desktop/LocalProjectDB
```

### Create Refactoring Task List

```bash
# Generate prioritized refactoring tasks
code-ingest generate-hierarchical-tasks INGEST_20250928150000 \
  --levels 4 \
  --groups 5 \
  --prompt-file .kiro/steering/refactoring-analysis-prompt.md \
  --output refactoring_roadmap.md \
  --db-path /Users/dev/desktop/LocalProjectDB
```

## Step 8: Continuous Monitoring Setup

### Create Analysis Script

```bash
cat > analyze_project.sh << 'EOF'
#!/bin/bash

PROJECT_PATH="/Users/dev/my-rust-project"
DB_PATH="/Users/dev/desktop/LocalProjectDB"
TIMESTAMP=$(date +%Y%m%d%H%M%S)

echo "Starting project analysis at $TIMESTAMP"

# Re-ingest project to capture latest changes
./target/release/code-ingest ingest "$PROJECT_PATH" \
  --folder-flag Y \
  --db-path "$DB_PATH" \
  --include "*.rs,*.md,*.toml" \
  --exclude "target/*,*.lock"

# Get the latest table name
LATEST_TABLE=$(psql -d LocalProjectDB -t -c "
  SELECT table_name 
  FROM ingestion_meta 
  ORDER BY created_at DESC 
  LIMIT 1
" | xargs)

echo "Latest ingestion table: $LATEST_TABLE"

# Generate updated analysis tasks
code-ingest generate-hierarchical-tasks "$LATEST_TABLE" \
  --levels 3 \
  --groups 5 \
  --prompt-file .kiro/steering/refactoring-analysis-prompt.md \
  --output "analysis_${TIMESTAMP}.md" \
  --db-path "$DB_PATH"

echo "Analysis complete. Tasks generated in analysis_${TIMESTAMP}.md"
EOF

chmod +x analyze_project.sh
```

### Set up Automated Analysis

```bash
# Add to crontab for daily analysis
echo "0 9 * * * /path/to/analyze_project.sh" | crontab -
```

## Step 9: Integration with Development Workflow

### Pre-commit Analysis

```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# Quick analysis of changed files
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rs$')

if [ -n "$CHANGED_FILES" ]; then
    echo "Analyzing changed Rust files..."
    
    # Create temporary directory for changed files
    TEMP_DIR=$(mktemp -d)
    
    for file in $CHANGED_FILES; do
        cp "$file" "$TEMP_DIR/"
    done
    
    # Quick ingestion and analysis
    ./target/release/code-ingest ingest "$TEMP_DIR" \
      --folder-flag Y \
      --db-path /tmp/precommit_analysis \
      --include "*.rs"
    
    # Check for common issues
    ISSUES=$(psql -d precommit_analysis -t -c "
      SELECT COUNT(*) 
      FROM (SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'INGEST_%' ORDER BY table_name DESC LIMIT 1) t,
           LATERAL (SELECT * FROM t.table_name) data
      WHERE data.content_text LIKE '%unwrap()%' 
         OR data.content_text LIKE '%TODO%'
         OR data.content_text LIKE '%FIXME%'
    " | xargs)
    
    if [ "$ISSUES" -gt 0 ]; then
        echo "Warning: Found $ISSUES potential issues in changed files"
        echo "Consider reviewing TODO/FIXME comments and unwrap() usage"
    fi
    
    rm -rf "$TEMP_DIR"
fi
EOF

chmod +x .git/hooks/pre-commit
```

## Expected Outcomes

After completing this workflow, you should have:

1. **Complete local project analysis** with all source files ingested
2. **Technical debt assessment** with quantified metrics
3. **Refactoring roadmap** with prioritized tasks
4. **Performance and security insights** from specialized queries
5. **Automated monitoring** for continuous code quality tracking
6. **Integration with development workflow** for proactive analysis

## Performance Metrics

Typical performance for a medium-sized Rust project:

- **Ingestion time**: 5-30 seconds (depending on project size)
- **Files processed**: 20-200 files
- **Database size**: 5-50MB
- **Analysis query time**: <50ms for most queries
- **Task generation**: <3 seconds

## Best Practices

1. **Regular Analysis**: Run analysis after major changes or weekly
2. **Incremental Improvements**: Focus on high-impact, low-effort refactoring
3. **Team Collaboration**: Share analysis results and discuss findings
4. **Automated Integration**: Use pre-commit hooks and CI integration
5. **Trend Tracking**: Monitor metrics over time to track improvement

## Troubleshooting

### Common Issues with Local Ingestion

```bash
# Permission issues
chmod -R 755 /path/to/project

# Large file handling
./target/release/code-ingest ingest /path/to/project \
  --folder-flag Y \
  --max-file-size 10MB \
  --exclude "*.log,*.tmp,target/*"

# Memory issues with large projects
./target/release/code-ingest ingest /path/to/project \
  --folder-flag Y \
  --batch-size 500 \
  --max-concurrency 4
```

This workflow provides a comprehensive approach to analyzing and improving local codebases using systematic data-driven insights.