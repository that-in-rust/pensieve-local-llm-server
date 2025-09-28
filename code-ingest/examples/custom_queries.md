# Custom Queries Example

Advanced SQL queries for specialized code analysis tasks.

## 🎯 Objective

Learn to write powerful custom SQL queries for:
- Complex code pattern analysis
- Cross-file relationship discovery
- Performance bottleneck identification
- Code quality metrics calculation
- Custom reporting and dashboards

## 📊 Advanced Query Patterns

### 1. Code Complexity Analysis

#### Cyclomatic Complexity Estimation
```sql
-- Estimate complexity based on control flow keywords
SELECT 
    filepath,
    line_count,
    -- Count decision points
    (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'if ', ''))) / 3 +
    (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'while ', ''))) / 6 +
    (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'for ', ''))) / 4 +
    (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'match ', ''))) / 6 +
    (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'case ', ''))) / 5 as complexity_score,
    -- Calculate complexity per line
    ROUND(
        ((LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'if ', ''))) / 3 +
         (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'while ', ''))) / 6 +
         (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'for ', ''))) / 4 +
         (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'match ', ''))) / 6 +
         (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'case ', ''))) / 5) * 100.0 / NULLIF(line_count, 0),
        2
    ) as complexity_per_line
FROM INGEST_20240928143022 
WHERE extension IN ('rs', 'py', 'js', 'java', 'go')
  AND line_count > 10
ORDER BY complexity_score DESC
LIMIT 20;
```

#### Function Density Analysis
```sql
-- Analyze function density and average function size
WITH function_stats AS (
    SELECT 
        filepath,
        line_count,
        -- Count functions (adjust regex for different languages)
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'fn ', ''))) / 3 as fn_count_rust,
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'def ', ''))) / 4 as fn_count_python,
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'function ', ''))) / 9 as fn_count_js
    FROM INGEST_20240928143022 
    WHERE file_type = 'direct_text'
      AND line_count > 0
)
SELECT 
    filepath,
    line_count,
    CASE 
        WHEN filepath LIKE '%.rs' THEN fn_count_rust
        WHEN filepath LIKE '%.py' THEN fn_count_python
        WHEN filepath LIKE '%.js' THEN fn_count_js
        ELSE 0
    END as function_count,
    CASE 
        WHEN filepath LIKE '%.rs' AND fn_count_rust > 0 THEN ROUND(line_count::float / fn_count_rust, 2)
        WHEN filepath LIKE '%.py' AND fn_count_python > 0 THEN ROUND(line_count::float / fn_count_python, 2)
        WHEN filepath LIKE '%.js' AND fn_count_js > 0 THEN ROUND(line_count::float / fn_count_js, 2)
        ELSE NULL
    END as avg_lines_per_function
FROM function_stats
WHERE CASE 
    WHEN filepath LIKE '%.rs' THEN fn_count_rust
    WHEN filepath LIKE '%.py' THEN fn_count_python
    WHEN filepath LIKE '%.js' THEN fn_count_js
    ELSE 0
END > 0
ORDER BY avg_lines_per_function DESC NULLS LAST;
```

### 2. Dependency and Import Analysis

#### Cross-Module Dependency Graph
```sql
-- Create dependency relationships for visualization
WITH module_dependencies AS (
    SELECT 
        REGEXP_REPLACE(filepath, '/[^/]+$', '') as source_module,
        filepath as source_file,
        -- Extract imported modules (Rust example)
        REGEXP_SPLIT_TO_TABLE(
            REGEXP_REPLACE(content_text, '.*use crate::([a-zA-Z_:]+).*', '\1', 'g'),
            '\n'
        ) as target_module
    FROM INGEST_20240928143022 
    WHERE content_text ~ 'use crate::'
      AND extension = 'rs'
),
cleaned_deps AS (
    SELECT DISTINCT
        source_module,
        TRIM(target_module) as target_module
    FROM module_dependencies
    WHERE TRIM(target_module) != ''
      AND TRIM(target_module) ~ '^[a-zA-Z_:]+$'
)
SELECT 
    source_module,
    target_module,
    COUNT(*) as dependency_count,
    -- Calculate dependency strength
    CASE 
        WHEN COUNT(*) >= 5 THEN 'Strong'
        WHEN COUNT(*) >= 2 THEN 'Medium'
        ELSE 'Weak'
    END as dependency_strength
FROM cleaned_deps
GROUP BY source_module, target_module
ORDER BY dependency_count DESC;
```

#### External Dependency Usage
```sql
-- Analyze external crate/library usage
WITH external_deps AS (
    SELECT 
        filepath,
        -- Extract external crate names
        REGEXP_MATCHES(content_text, 'use ([a-zA-Z_][a-zA-Z0-9_]*)::', 'g') as crate_matches
    FROM INGEST_20240928143022 
    WHERE content_text ~ 'use [a-zA-Z_][a-zA-Z0-9_]*::'
      AND extension = 'rs'
),
crate_usage AS (
    SELECT 
        filepath,
        (crate_matches)[1] as crate_name
    FROM external_deps
    WHERE (crate_matches)[1] NOT IN ('std', 'core', 'alloc', 'crate')
)
SELECT 
    crate_name,
    COUNT(DISTINCT filepath) as files_using,
    COUNT(*) as total_imports,
    ROUND(COUNT(DISTINCT filepath) * 100.0 / (
        SELECT COUNT(DISTINCT filepath) 
        FROM INGEST_20240928143022 
        WHERE extension = 'rs'
    ), 2) as adoption_percentage,
    STRING_AGG(DISTINCT filepath, ', ' ORDER BY filepath) as example_files
FROM crate_usage
GROUP BY crate_name
HAVING COUNT(DISTINCT filepath) >= 2
ORDER BY files_using DESC, total_imports DESC;
```

### 3. Code Quality Metrics

#### Technical Debt Indicators
```sql
-- Identify potential technical debt
SELECT 
    filepath,
    line_count,
    -- Count TODO/FIXME comments
    (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'TODO', ''))) / 4 as todo_count,
    (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'FIXME', ''))) / 5 as fixme_count,
    (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'HACK', ''))) / 4 as hack_count,
    -- Count potentially problematic patterns
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.unwrap()', ''))) / 9 as unwrap_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.expect(', ''))) / 8 as expect_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'panic!(', ''))) / 7 as panic_count,
    -- Calculate debt score
    ((LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'TODO', ''))) / 4 +
     (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'FIXME', ''))) / 5 +
     (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'HACK', ''))) / 4 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.unwrap()', ''))) / 9 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'panic!(', ''))) / 7) as debt_score
FROM INGEST_20240928143022 
WHERE extension IN ('rs', 'py', 'js', 'java', 'go')
  AND file_type = 'direct_text'
HAVING ((LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'TODO', ''))) / 4 +
        (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'FIXME', ''))) / 5 +
        (LENGTH(UPPER(content_text)) - LENGTH(REPLACE(UPPER(content_text), 'HACK', ''))) / 4 +
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.unwrap()', ''))) / 9 +
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'panic!(', ''))) / 7) > 0
ORDER BY debt_score DESC;
```

#### Code Duplication Detection
```sql
-- Find potential code duplication based on similar line patterns
WITH line_hashes AS (
    SELECT 
        filepath,
        line_count,
        -- Create a simple hash of common patterns
        MD5(REGEXP_REPLACE(content_text, '\s+', ' ', 'g')) as content_hash,
        -- Extract function signatures for comparison
        ARRAY_AGG(DISTINCT 
            SUBSTRING(content_text FROM 'fn [a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)')
        ) as function_signatures
    FROM INGEST_20240928143022 
    WHERE extension = 'rs'
      AND line_count BETWEEN 50 AND 500  -- Focus on medium-sized files
    GROUP BY filepath, line_count, content_text
),
similar_files AS (
    SELECT 
        a.filepath as file_a,
        b.filepath as file_b,
        a.line_count as lines_a,
        b.line_count as lines_b,
        -- Calculate similarity based on function signatures
        ARRAY_LENGTH(
            ARRAY(SELECT UNNEST(a.function_signatures) INTERSECT SELECT UNNEST(b.function_signatures)),
            1
        ) as common_functions
    FROM line_hashes a
    JOIN line_hashes b ON a.content_hash = b.content_hash
    WHERE a.filepath < b.filepath  -- Avoid duplicates
)
SELECT 
    file_a,
    file_b,
    lines_a,
    lines_b,
    common_functions,
    CASE 
        WHEN common_functions >= 5 THEN 'High Duplication'
        WHEN common_functions >= 2 THEN 'Medium Duplication'
        ELSE 'Low Duplication'
    END as duplication_level
FROM similar_files
WHERE common_functions > 0
ORDER BY common_functions DESC;
```

### 4. Performance Analysis Queries

#### Memory Allocation Patterns
```sql
-- Analyze memory allocation patterns (Rust example)
SELECT 
    filepath,
    line_count,
    -- Count allocation-related patterns
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Vec::new()', ''))) / 10 as vec_new_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.clone()', ''))) / 8 as clone_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Box::new(', ''))) / 9 as box_new_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'String::from(', ''))) / 13 as string_from_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.to_string()', ''))) / 12 as to_string_count,
    -- Calculate allocation intensity
    ((LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Vec::new()', ''))) / 10 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.clone()', ''))) / 8 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Box::new(', ''))) / 9 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'String::from(', ''))) / 13 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.to_string()', ''))) / 12) as allocation_score,
    -- Allocation per line ratio
    ROUND(
        ((LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Vec::new()', ''))) / 10 +
         (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.clone()', ''))) / 8 +
         (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Box::new(', ''))) / 9 +
         (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'String::from(', ''))) / 13 +
         (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.to_string()', ''))) / 12) * 100.0 / NULLIF(line_count, 0),
        2
    ) as allocations_per_100_lines
FROM INGEST_20240928143022 
WHERE extension = 'rs'
  AND file_type = 'direct_text'
  AND line_count > 10
HAVING ((LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Vec::new()', ''))) / 10 +
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.clone()', ''))) / 8 +
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Box::new(', ''))) / 9 +
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'String::from(', ''))) / 13 +
        (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.to_string()', ''))) / 12) > 0
ORDER BY allocations_per_100_lines DESC;
```

#### Async/Concurrency Patterns
```sql
-- Analyze async and concurrency usage
SELECT 
    filepath,
    line_count,
    -- Count async patterns
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'async fn', ''))) / 8 as async_fn_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.await', ''))) / 6 as await_count,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'tokio::', ''))) / 7 as tokio_usage,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Arc<', ''))) / 4 as arc_usage,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Mutex<', ''))) / 6 as mutex_usage,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'RwLock<', ''))) / 7 as rwlock_usage,
    -- Calculate concurrency complexity
    ((LENGTH(content_text) - LENGTH(REPLACE(content_text, 'async fn', ''))) / 8 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.await', ''))) / 6 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Arc<', ''))) / 4 +
     (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Mutex<', ''))) / 6) as concurrency_score
FROM INGEST_20240928143022 
WHERE extension = 'rs'
  AND (content_text ~ '(async fn|\.await|Arc<|Mutex<|RwLock<|tokio::)')
ORDER BY concurrency_score DESC;
```

### 5. Security Analysis Queries

#### Comprehensive Security Scan
```sql
-- Multi-layered security analysis
WITH security_patterns AS (
    SELECT 
        filepath,
        line_count,
        -- Credential patterns
        CASE WHEN content_text ~* '(password|secret|key|token)\s*[=:]\s*["\'][^"\']{8,}["\']' THEN 1 ELSE 0 END as has_hardcoded_secrets,
        -- Crypto patterns
        CASE WHEN content_text ~* '(md5|sha1|des|rc4)' THEN 1 ELSE 0 END as uses_weak_crypto,
        -- Injection patterns
        CASE WHEN content_text ~* '(query|execute).*\+.*[a-zA-Z_]' THEN 1 ELSE 0 END as potential_sql_injection,
        -- Unsafe patterns
        CASE WHEN content_text ~* '(eval|exec|system|shell_exec)\s*\(' THEN 1 ELSE 0 END as uses_unsafe_exec,
        -- Input validation
        CASE WHEN content_text ~* '(validate|sanitize|escape)' THEN 1 ELSE 0 END as has_input_validation,
        -- Authentication
        CASE WHEN content_text ~* '(auth|login|permission|role)' THEN 1 ELSE 0 END as has_auth_code,
        -- Error handling
        CASE WHEN content_text ~* '(try|catch|except|error|result)' THEN 1 ELSE 0 END as has_error_handling
    FROM INGEST_20240928143022 
    WHERE file_type = 'direct_text'
)
SELECT 
    filepath,
    line_count,
    has_hardcoded_secrets,
    uses_weak_crypto,
    potential_sql_injection,
    uses_unsafe_exec,
    has_input_validation,
    has_auth_code,
    has_error_handling,
    -- Calculate security risk score
    (has_hardcoded_secrets * 10 + 
     uses_weak_crypto * 8 + 
     potential_sql_injection * 9 + 
     uses_unsafe_exec * 7 - 
     has_input_validation * 2 - 
     has_error_handling * 1) as security_risk_score,
    -- Security assessment
    CASE 
        WHEN (has_hardcoded_secrets * 10 + uses_weak_crypto * 8 + potential_sql_injection * 9 + uses_unsafe_exec * 7 - has_input_validation * 2 - has_error_handling * 1) >= 15 THEN 'High Risk'
        WHEN (has_hardcoded_secrets * 10 + uses_weak_crypto * 8 + potential_sql_injection * 9 + uses_unsafe_exec * 7 - has_input_validation * 2 - has_error_handling * 1) >= 8 THEN 'Medium Risk'
        WHEN (has_hardcoded_secrets * 10 + uses_weak_crypto * 8 + potential_sql_injection * 9 + uses_unsafe_exec * 7 - has_input_validation * 2 - has_error_handling * 1) > 0 THEN 'Low Risk'
        ELSE 'Minimal Risk'
    END as risk_level
FROM security_patterns
WHERE (has_hardcoded_secrets + uses_weak_crypto + potential_sql_injection + uses_unsafe_exec) > 0
ORDER BY security_risk_score DESC;
```

### 6. Project Health Dashboard

#### Comprehensive Project Metrics
```sql
-- Create a project health dashboard
WITH project_stats AS (
    SELECT 
        COUNT(*) as total_files,
        COUNT(DISTINCT extension) as language_count,
        SUM(line_count) as total_lines,
        AVG(line_count) as avg_file_size,
        MAX(line_count) as largest_file_size,
        COUNT(*) FILTER (WHERE line_count > 500) as large_files,
        COUNT(*) FILTER (WHERE line_count < 50) as small_files
    FROM INGEST_20240928143022 
    WHERE file_type = 'direct_text'
),
quality_stats AS (
    SELECT 
        COUNT(*) FILTER (WHERE content_text ~* '(todo|fixme|hack)') as files_with_debt,
        COUNT(*) FILTER (WHERE content_text ~ '(test|spec)') as test_files,
        COUNT(*) FILTER (WHERE content_text ~* '(doc|comment|/\*|\*)') as documented_files,
        COUNT(*) FILTER (WHERE content_text ~* '(unsafe|unwrap|panic)') as risky_files
    FROM INGEST_20240928143022 
    WHERE file_type = 'direct_text'
),
security_stats AS (
    SELECT 
        COUNT(*) FILTER (WHERE content_text ~* '(password|secret|key)\s*[=:]') as files_with_secrets,
        COUNT(*) FILTER (WHERE content_text ~* '(md5|sha1|des)') as files_with_weak_crypto,
        COUNT(*) FILTER (WHERE content_text ~* 'auth') as auth_related_files
    FROM INGEST_20240928143022 
    WHERE file_type = 'direct_text'
)
SELECT 
    -- Project Size Metrics
    'Project Size' as category,
    json_build_object(
        'total_files', p.total_files,
        'total_lines', p.total_lines,
        'languages', p.language_count,
        'avg_file_size', ROUND(p.avg_file_size, 0),
        'largest_file', p.largest_file_size
    ) as metrics
FROM project_stats p

UNION ALL

SELECT 
    'File Distribution' as category,
    json_build_object(
        'large_files_500plus', p.large_files,
        'small_files_under50', p.small_files,
        'medium_files', p.total_files - p.large_files - p.small_files,
        'large_file_percentage', ROUND(p.large_files * 100.0 / p.total_files, 1)
    ) as metrics
FROM project_stats p

UNION ALL

SELECT 
    'Code Quality' as category,
    json_build_object(
        'files_with_technical_debt', q.files_with_debt,
        'test_files', q.test_files,
        'documented_files', q.documented_files,
        'risky_files', q.risky_files,
        'test_coverage_estimate', ROUND(q.test_files * 100.0 / p.total_files, 1)
    ) as metrics
FROM project_stats p, quality_stats q

UNION ALL

SELECT 
    'Security Assessment' as category,
    json_build_object(
        'files_with_secrets', s.files_with_secrets,
        'weak_crypto_files', s.files_with_weak_crypto,
        'auth_files', s.auth_related_files,
        'security_risk_percentage', ROUND((s.files_with_secrets + s.files_with_weak_crypto) * 100.0 / p.total_files, 1)
    ) as metrics
FROM project_stats p, security_stats s;
```

## 🔧 Query Optimization Tips

### Performance Optimization

1. **Use Indexes Effectively**
```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_extension ON INGEST_20240928143022(extension);
CREATE INDEX idx_file_type ON INGEST_20240928143022(file_type);
CREATE INDEX idx_line_count ON INGEST_20240928143022(line_count);

-- Use indexes in WHERE clauses
SELECT * FROM INGEST_20240928143022 
WHERE extension = 'rs' AND line_count > 100;
```

2. **Limit Result Sets**
```sql
-- Always use LIMIT for exploratory queries
SELECT * FROM INGEST_20240928143022 
WHERE content_text LIKE '%pattern%'
LIMIT 100;

-- Use pagination for large results
SELECT * FROM INGEST_20240928143022 
ORDER BY line_count DESC
LIMIT 50 OFFSET 100;
```

3. **Optimize Text Searches**
```sql
-- Use full-text search instead of LIKE when possible
SELECT * FROM INGEST_20240928143022 
WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'function authentication');

-- Instead of:
-- WHERE content_text LIKE '%function%' AND content_text LIKE '%authentication%'
```

### Query Debugging

```sql
-- Use EXPLAIN ANALYZE to understand query performance
EXPLAIN ANALYZE 
SELECT filepath, line_count 
FROM INGEST_20240928143022 
WHERE extension = 'rs' 
ORDER BY line_count DESC;

-- Check query execution time
\timing on
SELECT COUNT(*) FROM INGEST_20240928143022;
\timing off
```

## 📊 Custom Reporting Templates

### Weekly Code Analysis Report
```sql
-- Template for regular code analysis reports
WITH report_data AS (
    SELECT 
        DATE_TRUNC('week', created_at) as week,
        COUNT(*) as files_analyzed,
        SUM(line_count) as lines_analyzed,
        COUNT(DISTINCT extension) as languages_covered
    FROM INGEST_20240928143022 
    GROUP BY DATE_TRUNC('week', created_at)
)
SELECT 
    week,
    files_analyzed,
    lines_analyzed,
    languages_covered,
    LAG(files_analyzed) OVER (ORDER BY week) as prev_week_files,
    ROUND(
        (files_analyzed - LAG(files_analyzed) OVER (ORDER BY week)) * 100.0 / 
        NULLIF(LAG(files_analyzed) OVER (ORDER BY week), 0), 
        2
    ) as growth_percentage
FROM report_data
ORDER BY week DESC;
```

These custom queries provide powerful tools for deep code analysis and can be adapted for specific project needs and programming languages.