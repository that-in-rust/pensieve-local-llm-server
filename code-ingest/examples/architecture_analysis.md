# Architecture Analysis Example

Use code-ingest to understand and document the architecture of unfamiliar codebases.

## 🎯 Objective

Systematically analyze a codebase to understand:
- Overall system architecture and design patterns
- Module dependencies and relationships
- Data flow and component interactions
- Key abstractions and interfaces
- Technology stack and frameworks used

## 🏗️ Architecture Discovery Workflow

### Step 1: High-Level Structure Analysis

#### Project Structure Overview
```bash
# Get an overview of the project structure
code-ingest sql "
  SELECT 
    SPLIT_PART(filepath, '/', 1) as top_level_dir,
    COUNT(*) as file_count,
    COUNT(DISTINCT extension) as extension_variety,
    SUM(line_count) as total_lines
  FROM INGEST_20240928143022 
  WHERE file_type = 'direct_text'
  GROUP BY SPLIT_PART(filepath, '/', 1)
  ORDER BY file_count DESC
" --db-path ./analysis
```

#### Module Organization
```bash
# Analyze module structure (for Rust projects)
code-ingest sql "
  SELECT 
    filepath,
    filename,
    line_count,
    CASE 
      WHEN filename = 'mod.rs' THEN 'Module Root'
      WHEN filename = 'lib.rs' THEN 'Library Root'
      WHEN filename = 'main.rs' THEN 'Binary Entry'
      ELSE 'Implementation'
    END as module_type
  FROM INGEST_20240928143022 
  WHERE extension = 'rs' 
    AND (filename IN ('mod.rs', 'lib.rs', 'main.rs') OR filepath LIKE '%/mod.rs')
  ORDER BY module_type, line_count DESC
" --db-path ./analysis
```

### Step 2: Dependency Analysis

#### External Dependencies
```bash
# Find configuration files that define dependencies
code-ingest sql "
  SELECT filepath, filename, content_text
  FROM INGEST_20240928143022 
  WHERE filename IN ('Cargo.toml', 'package.json', 'requirements.txt', 'pom.xml', 'build.gradle')
  ORDER BY filename
" --db-path ./analysis
```

#### Internal Module Dependencies
```bash
# Analyze internal imports and dependencies (Rust example)
code-ingest sql "
  SELECT 
    filepath,
    COUNT(*) as import_count,
    string_agg(DISTINCT 
      SUBSTRING(content_text FROM 'use crate::([a-zA-Z_:]+)' FOR '#'), 
      ', '
    ) as imported_modules
  FROM INGEST_20240928143022 
  WHERE content_text ~ 'use crate::'
    AND extension = 'rs'
  GROUP BY filepath
  ORDER BY import_count DESC
  LIMIT 20
" --db-path ./analysis
```

### Step 3: Design Pattern Identification

#### Common Design Patterns
```bash
# Look for common design patterns
code-ingest sql "
  SELECT 
    filepath,
    CASE 
      WHEN content_text ~ 'trait.*\{' THEN 'Trait Definition'
      WHEN content_text ~ 'impl.*for.*\{' THEN 'Trait Implementation'
      WHEN content_text ~ 'struct.*Builder' THEN 'Builder Pattern'
      WHEN content_text ~ 'enum.*\{' THEN 'Enum/State Machine'
      WHEN content_text ~ 'fn new\(' THEN 'Constructor Pattern'
      WHEN content_text ~ 'async fn' THEN 'Async Pattern'
    END as pattern_type,
    line_count
  FROM INGEST_20240928143022 
  WHERE extension = 'rs'
    AND (
      content_text ~ 'trait.*\{' OR
      content_text ~ 'impl.*for.*\{' OR
      content_text ~ 'struct.*Builder' OR
      content_text ~ 'enum.*\{' OR
      content_text ~ 'fn new\(' OR
      content_text ~ 'async fn'
    )
  ORDER BY pattern_type, line_count DESC
" --db-path ./analysis
```

#### Error Handling Patterns
```bash
# Analyze error handling approaches
code-ingest sql "
  SELECT 
    filepath,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Result<', ''))) / 7 as result_usage,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, 'Option<', ''))) / 7 as option_usage,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '.unwrap()', ''))) / 9 as unwrap_usage,
    (LENGTH(content_text) - LENGTH(REPLACE(content_text, '?', ''))) as question_mark_usage
  FROM INGEST_20240928143022 
  WHERE extension = 'rs'
    AND content_text ~ '(Result<|Option<|\\.unwrap\\(\\)|\\?)'
  ORDER BY (result_usage + option_usage) DESC
  LIMIT 15
" --db-path ./analysis
```

### Step 4: Data Flow Analysis

#### Database Interactions
```bash
# Find database-related code
code-ingest sql "
  SELECT 
    filepath,
    CASE 
      WHEN lower(content_text) LIKE '%sqlx::%' THEN 'SQLx Database'
      WHEN lower(content_text) LIKE '%diesel::%' THEN 'Diesel ORM'
      WHEN lower(content_text) LIKE '%sea_orm::%' THEN 'SeaORM'
      WHEN lower(content_text) ~ 'select.*from.*where' THEN 'Raw SQL'
      WHEN lower(content_text) LIKE '%redis%' THEN 'Redis Cache'
      WHEN lower(content_text) LIKE '%mongodb%' THEN 'MongoDB'
    END as db_type,
    line_count
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(sqlx|diesel|sea_orm|select.*from|redis|mongodb)'
    AND file_type = 'direct_text'
  ORDER BY db_type, line_count DESC
" --db-path ./analysis
```

#### API and Network Patterns
```bash
# Identify API and networking code
code-ingest sql "
  SELECT 
    filepath,
    CASE 
      WHEN content_text ~ 'axum::' THEN 'Axum Web Framework'
      WHEN content_text ~ 'warp::' THEN 'Warp Web Framework'
      WHEN content_text ~ 'actix_web::' THEN 'Actix Web Framework'
      WHEN content_text ~ 'reqwest::' THEN 'HTTP Client (reqwest)'
      WHEN content_text ~ 'hyper::' THEN 'Hyper HTTP'
      WHEN content_text ~ 'serde::' THEN 'Serialization (Serde)'
      WHEN content_text ~ 'tokio::' THEN 'Async Runtime (Tokio)'
    END as framework_type,
    COUNT(*) as usage_count
  FROM INGEST_20240928143022 
  WHERE content_text ~ '(axum::|warp::|actix_web::|reqwest::|hyper::|serde::|tokio::)'
    AND extension = 'rs'
  GROUP BY filepath, framework_type
  ORDER BY framework_type, usage_count DESC
" --db-path ./analysis
```

### Step 5: Comprehensive Architecture Analysis

#### Prepare Architecture Analysis
```bash
code-ingest query-prepare "
  SELECT filepath, content_text, line_count, extension
  FROM INGEST_20240928143022 
  WHERE (
    filename IN ('mod.rs', 'lib.rs', 'main.rs') OR
    filepath LIKE '%/mod.rs' OR
    line_count > 200 OR
    content_text ~ '(trait|impl|struct|enum).*\{'
  )
  AND file_type = 'direct_text'
  ORDER BY 
    CASE 
      WHEN filename = 'main.rs' THEN 1
      WHEN filename = 'lib.rs' THEN 2
      WHEN filename = 'mod.rs' THEN 3
      ELSE 4
    END,
    line_count DESC
" --temp-path ./architecture-temp.txt \
  --tasks-file ./architecture-tasks.md \
  --output-table QUERYRESULT_architecture_analysis \
  --db-path ./analysis
```

#### Create Architecture Analysis Prompt
```bash
cat > architecture-analysis-prompt.md << 'EOF'
# Architecture Analysis Prompt

Analyze the provided code files to understand the system architecture. For each file, provide:

## 1. Architectural Role
- What role does this component play in the overall system?
- Is this a core module, utility, interface, or implementation?
- How does it fit into the larger architecture?

## 2. Design Patterns
- What design patterns are implemented?
- Are there traits, interfaces, or abstract base classes?
- How is dependency injection or inversion of control handled?
- What creational, structural, or behavioral patterns are used?

## 3. Data Structures and Models
- What are the key data structures defined?
- How is data modeled and organized?
- What are the relationships between different data types?
- Are there domain models, DTOs, or value objects?

## 4. Component Interactions
- How does this component interact with others?
- What are the input/output interfaces?
- What dependencies does it have?
- What services does it provide to other components?

## 5. Technology Integration
- What external libraries or frameworks are used?
- How are third-party dependencies integrated?
- What protocols or standards are implemented?
- Are there any platform-specific considerations?

## 6. Scalability and Performance
- What performance considerations are evident?
- How is concurrency handled?
- Are there any scalability patterns?
- What resource management strategies are used?

## 7. Error Handling and Reliability
- How are errors handled and propagated?
- What reliability patterns are implemented?
- Are there retry mechanisms or circuit breakers?
- How is system resilience achieved?

For each file, provide:
- **Component Type**: Library, service, model, controller, etc.
- **Key Responsibilities**: Primary functions and purposes
- **Dependencies**: What it depends on
- **Interfaces**: What it exposes to other components
- **Patterns**: Design patterns and architectural styles used
- **Notes**: Important architectural decisions or trade-offs

Format as structured markdown with clear sections and architectural diagrams where helpful.
EOF
```

#### Generate Architecture Tasks
```bash
code-ingest generate-tasks \
  --sql "SELECT * FROM QUERYRESULT_architecture_analysis ORDER BY line_count DESC" \
  --prompt-file architecture-analysis-prompt.md \
  --output-table QUERYRESULT_detailed_architecture \
  --tasks-file ./detailed-architecture-tasks.md \
  --db-path ./analysis
```

## 📊 Architecture Visualization Queries

### Component Dependency Graph
```bash
# Create data for dependency visualization
code-ingest sql "
  WITH module_imports AS (
    SELECT 
      REGEXP_REPLACE(filepath, '/[^/]+$', '') as module_path,
      filepath,
      ARRAY_AGG(DISTINCT 
        SUBSTRING(content_text FROM 'use crate::([a-zA-Z_:]+)' FOR '#')
      ) as imports
    FROM INGEST_20240928143022 
    WHERE content_text ~ 'use crate::'
      AND extension = 'rs'
    GROUP BY module_path, filepath
  )
  SELECT 
    module_path as source_module,
    UNNEST(imports) as target_module,
    COUNT(*) as dependency_strength
  FROM module_imports
  WHERE UNNEST(imports) IS NOT NULL
  GROUP BY module_path, UNNEST(imports)
  ORDER BY dependency_strength DESC
" --db-path ./analysis
```

### Layer Analysis
```bash
# Identify architectural layers
code-ingest sql "
  SELECT 
    CASE 
      WHEN filepath LIKE '%/api/%' OR filepath LIKE '%/web/%' THEN 'Presentation Layer'
      WHEN filepath LIKE '%/service/%' OR filepath LIKE '%/business/%' THEN 'Business Layer'
      WHEN filepath LIKE '%/repository/%' OR filepath LIKE '%/dao/%' THEN 'Data Access Layer'
      WHEN filepath LIKE '%/model/%' OR filepath LIKE '%/entity/%' THEN 'Data Model Layer'
      WHEN filepath LIKE '%/config/%' OR filepath LIKE '%/settings/%' THEN 'Configuration Layer'
      WHEN filepath LIKE '%/util/%' OR filepath LIKE '%/helper/%' THEN 'Utility Layer'
      ELSE 'Core Layer'
    END as architectural_layer,
    COUNT(*) as file_count,
    SUM(line_count) as total_lines,
    AVG(line_count) as avg_file_size
  FROM INGEST_20240928143022 
  WHERE file_type = 'direct_text'
    AND extension IN ('rs', 'py', 'js', 'java', 'go')
  GROUP BY architectural_layer
  ORDER BY total_lines DESC
" --db-path ./analysis
```

### Technology Stack Summary
```bash
# Comprehensive technology stack analysis
code-ingest sql "
  WITH tech_patterns AS (
    SELECT 
      filepath,
      CASE 
        WHEN content_text ~ 'use axum' THEN 'Web Framework: Axum'
        WHEN content_text ~ 'use warp' THEN 'Web Framework: Warp'
        WHEN content_text ~ 'use actix_web' THEN 'Web Framework: Actix'
        WHEN content_text ~ 'use sqlx' THEN 'Database: SQLx'
        WHEN content_text ~ 'use diesel' THEN 'ORM: Diesel'
        WHEN content_text ~ 'use redis' THEN 'Cache: Redis'
        WHEN content_text ~ 'use serde' THEN 'Serialization: Serde'
        WHEN content_text ~ 'use tokio' THEN 'Async Runtime: Tokio'
        WHEN content_text ~ 'use clap' THEN 'CLI: Clap'
        WHEN content_text ~ 'use log' THEN 'Logging: Log'
        WHEN content_text ~ 'use tracing' THEN 'Tracing: Tracing'
      END as technology
    FROM INGEST_20240928143022 
    WHERE extension = 'rs'
      AND content_text ~ 'use (axum|warp|actix_web|sqlx|diesel|redis|serde|tokio|clap|log|tracing)'
  )
  SELECT 
    technology,
    COUNT(DISTINCT filepath) as files_using,
    ROUND(COUNT(DISTINCT filepath) * 100.0 / (SELECT COUNT(DISTINCT filepath) FROM INGEST_20240928143022 WHERE extension = 'rs'), 2) as percentage_adoption
  FROM tech_patterns
  WHERE technology IS NOT NULL
  GROUP BY technology
  ORDER BY files_using DESC
" --db-path ./analysis
```

## 🎯 Architecture Documentation Generation

### Create Architecture Summary
```bash
# Generate comprehensive architecture summary
code-ingest sql "
  SELECT 
    'Project Overview' as section,
    json_build_object(
      'total_files', COUNT(*),
      'total_lines', SUM(line_count),
      'languages', COUNT(DISTINCT extension),
      'avg_file_size', ROUND(AVG(line_count), 2)
    ) as metrics
  FROM INGEST_20240928143022 
  WHERE file_type = 'direct_text'
  
  UNION ALL
  
  SELECT 
    'File Type Distribution' as section,
    json_object_agg(extension, file_count) as metrics
  FROM (
    SELECT extension, COUNT(*) as file_count
    FROM INGEST_20240928143022 
    WHERE extension IS NOT NULL
    GROUP BY extension
    ORDER BY file_count DESC
    LIMIT 10
  ) ext_stats
  
  UNION ALL
  
  SELECT 
    'Largest Components' as section,
    json_object_agg(filepath, line_count) as metrics
  FROM (
    SELECT filepath, line_count
    FROM INGEST_20240928143022 
    WHERE line_count IS NOT NULL
    ORDER BY line_count DESC
    LIMIT 10
  ) large_files
" --db-path ./analysis
```

### Export Architecture Documentation
```bash
# Store architecture analysis results
code-ingest store-result \
  --output-table QUERYRESULT_detailed_architecture \
  --result-file ./architecture-analysis-complete.txt \
  --original-query "Comprehensive architecture analysis" \
  --db-path ./analysis

# Export individual component analyses
code-ingest print-to-md \
  --table QUERYRESULT_detailed_architecture \
  --sql "SELECT * FROM QUERYRESULT_detailed_architecture ORDER BY created_at" \
  --prefix architecture-component \
  --location ./architecture-docs/ \
  --db-path ./analysis
```

## 📈 Architecture Evolution Tracking

### Compare Architecture Over Time
```bash
# If you have multiple ingestions of the same project
code-ingest sql "
  SELECT 
    table_name,
    COUNT(*) as total_files,
    SUM(line_count) as total_lines,
    COUNT(DISTINCT extension) as language_count
  FROM (
    SELECT 'INGEST_20240901120000' as table_name, * FROM INGEST_20240901120000
    UNION ALL
    SELECT 'INGEST_20240928143022' as table_name, * FROM INGEST_20240928143022
  ) combined
  WHERE file_type = 'direct_text'
  GROUP BY table_name
  ORDER BY table_name
" --db-path ./analysis
```

## 🔧 Architecture Analysis Best Practices

### 1. Start with Structure
- Understand the directory organization
- Identify entry points (main.rs, lib.rs, etc.)
- Map out module hierarchies
- Look for configuration files

### 2. Identify Patterns
- Look for common design patterns
- Understand error handling approaches
- Identify abstraction layers
- Find dependency injection patterns

### 3. Trace Data Flow
- Follow data from input to output
- Understand transformation points
- Identify storage mechanisms
- Map API boundaries

### 4. Document Findings
- Create architectural diagrams
- Document key decisions and trade-offs
- Identify areas for improvement
- Track architectural evolution

This systematic approach to architecture analysis helps you quickly understand complex codebases and make informed decisions about modifications or extensions.