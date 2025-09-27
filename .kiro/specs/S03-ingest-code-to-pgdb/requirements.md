# Requirements Document

## Introduction

**The Problem:** Developers waste hours trying to understand unfamiliar codebases. Current tools (GitHub search, grep, IDE search) are too slow and shallow for semantic code understanding at scale.

**The Solution:** A Rust-based code ingestion system that clones GitHub repos, intelligently processes all file types, and stores structured data in a personal PostgreSQL database for lightning-fast semantic search.

**Success Metric:** Reduce time to understand a new codebase from hours to minutes.

## Core User Journeys

### Journey 1: New Developer Onboarding - "I need to understand this codebase by tomorrow"

**Context:** Sarah just joined a startup. She's been assigned to fix a critical bug in a 50,000-line Rust microservice she's never seen. The team lead said "just dive in" but she's overwhelmed.

**Current Pain:** Spending 4+ hours just figuring out where authentication logic lives, how database connections work, what the main data flow is.

**User Journey:**
1. **Discovery Phase** - Sarah finds the GitHub repo URL from team Slack
2. **Setup Anxiety** - "Do I need to install PostgreSQL? Will this mess up my system?"
   - Runs: `code-ingest pg-start` → Gets clear instructions for local PostgreSQL setup
   - Runs: `brew install postgresql && brew services start postgresql`
   - Sets workspace: `mkdir ~/work-analysis && cd ~/work-analysis`
3. **First Ingestion** - "Let me get this entire codebase into something searchable"
   - Runs: `code-ingest https://github.com/company/auth-service --db-path ~/work-analysis/db`
   - Waits 2 minutes while system clones and processes 847 files
   - Sees: "✅ Ingested 847 files into table INGEST_20250927143022"
4. **Exploration** - "What did I just ingest? What's in here?"
   - Runs: `code-ingest list-tables --db-path ~/work-analysis/db`
   - Sees: `ingestion_meta`, `INGEST_20250927143022`
   - Runs: `code-ingest sample --db-path ~/work-analysis/db --table INGEST_20250927143022 --limit 3`
   - Sees sample of file paths, extensions, line counts
5. **Targeted Analysis** - "Where's the authentication code?"
   - Runs: `code-ingest sql "SELECT filepath, content_text FROM INGEST_20250927143022 WHERE content_text LIKE '%authenticate%' OR filepath LIKE '%auth%'" --db-path ~/work-analysis/db`
   - Gets 12 relevant files printed to terminal
6. **Deep Dive with IDE Tasks** - "Help me understand these auth files"
   - Runs: `code-ingest query-prepare "SELECT filepath, content_text FROM INGEST_20250927143022 WHERE content_text LIKE '%authenticate%'" --db-path ~/work-analysis/db --temp-path ~/work-analysis/auth-temp.txt --tasks-file ~/work-analysis/auth-tasks.md --output-table QUERYRESULT_auth_analysis`
   - System writes query results to `auth-temp.txt` and creates structured task markdown in `auth-tasks.md`
   - Sarah opens `auth-tasks.md` in Kiro IDE, which shows organized analysis tasks
   - IDE executes tasks systematically, reading from `auth-temp.txt` and applying analysis prompts
   - When analysis completes, Sarah runs: `code-ingest store-result --db-path ~/work-analysis/db --output-table QUERYRESULT_auth_analysis --result-file ~/work-analysis/auth-result.txt --original-query "SELECT..."`
   - Results stored in `QUERYRESULT_auth_analysis` table for future reference

**Outcome:** Sarah understands the auth system in 45 minutes instead of 4 hours. She can now focus on the actual bug.

### Journey 2: Code Review Preparation - "I need to systematically analyze this large PR"

**Context:** Marcus is a senior engineer reviewing a 2,000-line PR that refactors the entire data processing pipeline. He needs to ensure no regressions and understand the new architecture.

**Current Pain:** Jumping between GitHub files, losing context, missing subtle changes in logic flow.

**User Journey:**
1. **PR Context** - Marcus has the PR branch locally at `~/code/data-pipeline-refactor`
2. **Local Analysis Setup**
   - Runs: `code-ingest ~/code/data-pipeline-refactor --db-path ~/reviews/pr-analysis/db`
   - System processes local folder, creates `INGEST_20250927151500` table
3. **Systematic Review Planning** - "I need to review this methodically"
   - Runs: `code-ingest generate-tasks --sql "SELECT filepath, content_text FROM INGEST_20250927151500 WHERE extension IN ('.rs', '.toml')" --prompt-file ~/reviews/prompts/code-review.md --output-table QUERYRESULT_pr_review --tasks-file ~/reviews/pr-tasks.md --db-path ~/reviews/pr-analysis/db`
   - Gets structured 7-part task breakdown for systematic review
4. **Batch Analysis** - "Let me analyze each section systematically"
   - Runs analysis for each task group, storing results in organized tables
   - `code-ingest query-prepare "SELECT filepath, content_text FROM INGEST_20250927151500 WHERE filepath LIKE 'src/pipeline/%'" --output-table QUERYRESULT_pipeline_review --db-path ~/reviews/pr-analysis/db --temp-path ~/reviews/temp.txt --tasks-file ~/reviews/pipeline-tasks.md`
5. **Export for Documentation** - "I need individual files for detailed comments"
   - Runs: `code-ingest print-to-md --db-path ~/reviews/pr-analysis/db --table QUERYRESULT_pipeline_review --sql "SELECT * FROM QUERYRESULT_pipeline_review" --prefix review-notes --location ~/reviews/md-files/`
   - Gets individual markdown files for each analysis result

**Outcome:** Marcus completes a thorough, systematic review with documented analysis for each component.

### Journey 3: Architecture Documentation - "I need to document this legacy system"

**Context:** Elena inherited a 5-year-old Rust service with zero documentation. New team members keep asking the same questions. She needs to create comprehensive architecture docs.

**Current Pain:** Manually reading through thousands of files, trying to understand relationships, missing important patterns.

**User Journey:**
1. **Legacy System Analysis**
   - Runs: `code-ingest ~/legacy-systems/payment-processor --db-path ~/docs/legacy-analysis/db`
   - Processes 1,247 files across multiple modules
2. **Pattern Discovery** - "What are the main architectural patterns?"
   - Creates specialized prompts for different analysis types:
     - `~/docs/prompts/find-patterns.md` - Identifies design patterns
     - `~/docs/prompts/data-flow.md` - Maps data flow
     - `~/docs/prompts/dependencies.md` - Analyzes dependencies
3. **Systematic Documentation Generation**
   - Generates task structure for comprehensive analysis
   - Runs multiple analysis sessions, each focused on different aspects
   - Stores results in organized tables: `QUERYRESULT_patterns_analysis`, `QUERYRESULT_dataflow_analysis`, `QUERYRESULT_deps_analysis`
4. **Knowledge Base Creation**
   - Exports all analysis results as individual markdown files
   - Organizes into documentation structure
   - Creates searchable knowledge base for team

**Outcome:** Elena creates comprehensive documentation that reduces new developer onboarding from weeks to days.

### Journey 4: Security Audit - "I need to find all potential security issues"

**Context:** David is conducting a security audit of multiple microservices before a major product launch. He needs to systematically identify potential vulnerabilities.

**User Journey:**
1. **Multi-Repository Ingestion**
   - Ingests 5 different services into separate timestamped tables
   - Each service gets its own analysis workspace
2. **Security-Focused Analysis**
   - Creates security-specific prompts and task structures
   - Runs systematic analysis looking for common vulnerability patterns
   - Stores findings in security-focused output tables
3. **Cross-Service Analysis**
   - Uses SQL queries to find patterns across all ingested services
   - Identifies common security anti-patterns
   - Documents findings for remediation

**Outcome:** David completes comprehensive security audit with documented findings and remediation recommendations.

## Requirements

### Requirement 1: Dual Input Sources with Absolute Path Database

**User Story:** As a developer, I want to ingest code from either GitHub repositories or local folders into a PostgreSQL database at a specified absolute path, so that I can analyze any codebase regardless of its source.

#### Acceptance Criteria

1. WHEN I provide a GitHub repository URL THEN the system SHALL clone the repository and wait for download completion before processing
2. WHEN I provide an absolute folder path THEN the system SHALL process the existing local folder directly
3. WHEN ingestion starts THEN the system SHALL create a timestamped table `INGEST_YYYYMMDDHHMMSS` and record reference in `ingestion_meta` table
4. WHEN I specify `--db-path` THEN the system SHALL use that absolute path for the PostgreSQL database location
5. WHEN processing completes THEN the system SHALL update the `ingestion_meta` record with completion status and file count

### Requirement 2: File System Processing and Three-Type Classification

**User Story:** As a developer, I want the system to intelligently process repository files and classify them into three distinct types, so that appropriate processing is applied to each file category.

#### Acceptance Criteria

1. WHEN processing a repository THEN the system SHALL recursively traverse all directories
2. WHEN encountering files THEN the system SHALL classify them into three types:
   - **Type 1 (Direct Text)**: Files that are already text-based (.md, .txt, .py, .js, .rs, .java, .c, .cpp, .h, .json, .yaml, .xml, .html, .css, .sql, .sh, etc.)
   - **Type 2 (Convertible to Text)**: Files that can be converted to text via terminal commands (.pdf, .docx, .xlsx, .pptx, .odt, etc.)
   - **Type 3 (Non-Text Relevant)**: Files that cannot be meaningfully converted to text (.jpg, .png, .gif, .mp4, .zip, .exe, .bin, etc.)
3. WHEN encountering .gitignore files THEN the system SHALL respect ignore patterns by default
4. WHEN I provide custom include/exclude patterns THEN the system SHALL apply those filters
5. WHEN a file exceeds the maximum size limit THEN the system SHALL skip that file and log the decision
6. WHEN processing symlinks THEN the system SHALL handle them safely without infinite loops

### Requirement 3: PostgreSQL Storage Schema with Three-Type Processing

**User Story:** As a developer, I want code content stored in a structured PostgreSQL database with timestamped tables and proper handling of all three file types, so that I can track ingestion history and query file-level metrics.

#### Acceptance Criteria

1. WHEN starting an ingestion query THEN the system SHALL create a timestamped table in format `INGEST_YYYYMMDDHHMMSS`
2. WHEN processing Type 1 files (direct text) THEN the system SHALL store: filepath, filename, extension, file_size_bytes, line_count, word_count, token_count, content_text, file_type='direct_text'
3. WHEN processing Type 2 files (convertible) THEN the system SHALL store: filepath, filename, extension, file_size_bytes, line_count, word_count, token_count, content_text, conversion_command, file_type='convertible'
4. WHEN processing Type 3 files (non-text) THEN the system SHALL store: filepath, filename, extension, file_size_bytes, NULL for text metrics, file_type='non_text'
5. WHEN a query (GitHub repo ingestion) starts THEN the system SHALL record in `ingestion_meta` table: ingestion_id, repo_url, local_path, start_timestamp_unix, table_name
6. WHEN the ingestion ends THEN the system SHALL update the `ingestion_meta` record with end_timestamp_unix and total_files_processed
7. WHEN storing all files THEN the system SHALL include common fields: created_at, ingestion_id (foreign key), relative_path, absolute_path

### Database Schema ER Diagram

```mermaid
erDiagram
    INGESTION_META {
        bigserial ingestion_id PK
        varchar repo_url
        varchar local_path
        bigint start_timestamp_unix
        bigint end_timestamp_unix
        varchar table_name
        integer total_files_processed
        timestamp created_at
    }
    
    INGEST_YYYYMMDDHHMMSS {
        bigserial file_id PK
        bigint ingestion_id FK
        varchar filepath
        varchar filename
        varchar extension
        bigint file_size_bytes
        integer line_count
        integer word_count
        integer token_count
        text content_text
        varchar file_type
        varchar conversion_command
        varchar relative_path
        varchar absolute_path
        timestamp created_at
    }
    
    QUERYRESULT_CUSTOM_NAME {
        bigserial analysis_id PK
        text sql_query
        varchar prompt_file_path
        text llm_result
        timestamp created_at
    }
    
    INGESTION_META ||--o{ INGEST_YYYYMMDDHHMMSS : "has many files"
```

### Requirement 4: Content Indexing and Search

**User Story:** As a developer, I want to search through ingested code efficiently, so that I can find relevant code snippets and understand codebase structure.

#### Acceptance Criteria

1. WHEN content is stored THEN the system SHALL create full-text search indexes on file content
2. WHEN I search for code patterns THEN the system SHALL return ranked results with file paths and line numbers
3. WHEN searching across multiple repositories THEN the system SHALL support repository-scoped queries
4. WHEN retrieving search results THEN the system SHALL include sufficient context around matches
5. WHEN querying by file type or path patterns THEN the system SHALL support structured metadata queries

### Requirement 5: Incremental Updates

**User Story:** As a developer, I want to update previously ingested repositories efficiently, so that I can keep my code database current without re-processing unchanged files.

#### Acceptance Criteria

1. WHEN re-ingesting a repository THEN the system SHALL detect which files have changed since last ingestion
2. WHEN files are unchanged THEN the system SHALL skip re-processing those files
3. WHEN files are deleted from the repository THEN the system SHALL mark them as deleted in the database
4. WHEN new files are added THEN the system SHALL process and store only the new files
5. WHEN tracking changes THEN the system SHALL maintain a history of file modifications

### Requirement 6: CLI Interface with Direct SQL Access

**User Story:** As a developer, I want to run SQL queries directly against the ingested code database and get clean terminal output, so that I can pipe results to LLM tools for analysis.

#### Acceptance Criteria

1. WHEN using the CLI THEN the system SHALL provide commands for ingesting and managing repositories
2. WHEN running SQL queries THEN the system SHALL execute raw SQL against the PostgreSQL database and return results
3. WHEN outputting query results THEN the system SHALL format results as clean, readable text suitable for terminal display
4. WHEN operations are long-running THEN the system SHALL provide progress feedback and status updates
5. WHEN errors occur THEN the system SHALL return structured error responses with actionable messages

### Requirement 7: Performance and Personal Use Optimization

**User Story:** As a developer, I want the system to handle large repositories efficiently on my personal machine, so that I can quickly ingest and search through multiple codebases.

#### Acceptance Criteria

1. WHEN ingesting large repositories THEN the system SHALL complete processing within reasonable time limits for personal use
2. WHEN running multiple ingestion operations THEN the system SHALL handle concurrent operations safely on a single machine
3. WHEN storing large amounts of content THEN the system SHALL optimize database storage and query performance for personal PostgreSQL instances
4. WHEN memory usage grows THEN the system SHALL implement proper cleanup and resource management for desktop environments
5. WHEN processing fails THEN the system SHALL provide retry mechanisms and graceful degradation

### Requirement 8: SQL Query Output for LLM Consumption

**User Story:** As a developer, I want to run SQL queries against the code database and get terminal output that can be piped directly to LLM tools with custom prompts, so that I can analyze both original code and stored AI insights.

#### Acceptance Criteria

1. WHEN I run a SQL query THEN the system SHALL output results in clean, readable terminal format suitable for LLM processing
2. WHEN piping to LLM tools THEN the system SHALL format output without extra formatting or colors that interfere with text processing
3. WHEN querying file contents THEN the system SHALL output filepath and content in structured format: `FILE: path/to/file.rs\n\n[content]\n\n---\n`
4. WHEN querying analysis results THEN the system SHALL output both original code and AI analysis in structured format for comparison
5. WHEN applying prompts THEN the system SHALL support: `code-ingest sql "SELECT ..." | llm -f custom-prompt.md`

### Requirement 9: PostgreSQL Database Setup and Connection

**User Story:** As a developer, I want clear instructions and scripts for setting up PostgreSQL and connecting the code ingestion system to my database, so that I can get started quickly without database administration complexity.

#### Acceptance Criteria

1. WHEN setting up the system THEN the system SHALL provide setup scripts or clear documentation for PostgreSQL installation
2. WHEN connecting to PostgreSQL THEN the system SHALL accept DATABASE_URL environment variable in standard PostgreSQL format: `postgresql://user:password@host:port/database`
3. WHEN the database doesn't exist THEN the system SHALL provide scripts to create the required database and initial schema
4. WHEN running for the first time THEN the system SHALL automatically create required tables (`ingestion_meta` and initial schema)
5. WHEN PostgreSQL is not running THEN the system SHALL provide clear error messages with troubleshooting guidance

### Requirement 10: Query-to-Temp File and Task Structure for IDE Integration

**User Story:** As a developer, I want to run SQL queries against ingested code, write results to temporary files, and generate task structures that my IDE can execute, so that I can trigger systematic LLM analysis through my development environment.

#### Acceptance Criteria

1. WHEN I run query-prepare THEN the system SHALL execute: `code-ingest query-prepare "SQL_QUERY" --db-path /abs/path/db --temp-path /abs/path/temp.txt --tasks-file /abs/path/tasks.md --output-table my_analysis`
2. WHEN executing query-prepare THEN the system SHALL write SQL results to the specified `--temp-path` absolute file location
3. WHEN temp file is ready THEN the system SHALL generate structured task markdown at `--tasks-file` that the IDE can execute as a series of analysis tasks
4. WHEN IDE processes the tasks THEN it can read the temp file content and execute LLM analysis according to the task structure
5. WHEN analysis results are ready THEN the system SHALL provide: `code-ingest store-result --db-path /abs/path/db --output-table my_analysis --result-file /abs/path/result.txt --original-query "SQL_QUERY"`

### Requirement 11: Kiro-Compatible Task Generation with Simple 7-Part Division

**User Story:** As a developer, I want to generate structured markdown task files that divide all analysis tasks into 7 main groups with Kiro-compatible numbering, so that LLMs can work through large datasets in organized batches.

#### Acceptance Criteria

1. WHEN I run task generation THEN the system SHALL execute: `code-ingest generate-tasks --sql "SELECT * FROM INGEST_20250927120000" --prompt-file analyze.md --output-table QUERYRESULT_results --tasks-file /abs/path/tasks.md --db-path /abs/path/db`
2. WHEN generating tasks THEN the system SHALL first create the complete list of all individual file/chunk analysis tasks
3. WHEN dividing into groups THEN the system SHALL divide the total task count by 7 and distribute evenly (e.g., 35 tasks ÷ 7 = 5 tasks per group)
4. WHEN structuring with Kiro numbering THEN the system SHALL format as: `- [ ] 1. Task Group 1` (main task) followed by `- [ ] 1.1`, `- [ ] 1.2`, `- [ ] 1.3`, etc. (sub-tasks)
5. WHEN creating the final structure THEN the system SHALL produce 7 main tasks each containing their allocated sub-tasks in sequential order

### Requirement 12: Print-to-MD Files Script for Individual Row Export

**User Story:** As a developer, I want to export query results as individual markdown files with sequential naming, so that I can process each database row as a separate file for detailed analysis.

#### Acceptance Criteria

1. WHEN I run print-to-md THEN the system SHALL execute: `code-ingest print-to-md --db-path /abs/path/db --table table_name --sql "SELECT * FROM table WHERE condition" --prefix analysis --location /abs/path/md-files/`
2. WHEN executing the query THEN the system SHALL create individual MD files named: `PREFIX-00001.md`, `PREFIX-00002.md`, ..., `PREFIX-N.md` where N is the number of rows returned
3. WHEN writing MD files THEN the system SHALL save each file to the specified `--location` absolute path directory
4. WHEN formatting content THEN the system SHALL structure each MD file with the row data in readable markdown format
5. WHEN processing completes THEN the system SHALL report the total number of files created and their location

### Requirement 13: PostgreSQL Database Exploration and Management

**User Story:** As a developer, I want built-in commands to explore my PostgreSQL database, list tables, and sample data, so that I can understand my ingested data structure without external tools.

#### Acceptance Criteria

1. WHEN I want to explore the database THEN the system SHALL provide: `code-ingest db-info --db-path /abs/path/db` to show database connection status and basic info
2. WHEN I want to list tables THEN the system SHALL provide: `code-ingest list-tables --db-path /abs/path/db` to show all available tables
3. WHEN I want to sample data THEN the system SHALL provide: `code-ingest sample --db-path /abs/path/db --table table_name --limit 5` to show first N rows
4. WHEN I want table schema THEN the system SHALL provide: `code-ingest describe --db-path /abs/path/db --table table_name` to show column names and types
5. WHEN I want to start PostgreSQL THEN the system SHALL provide guidance: `code-ingest pg-start` with instructions for starting local PostgreSQL service

### Requirement 14: Personal Configuration and Setup

**User Story:** As a developer, I want to easily configure the system for my personal development environment, so that it works seamlessly with my local PostgreSQL setup.

#### Acceptance Criteria

1. WHEN setting up the system THEN the system SHALL support configuration via environment variables and local config files
2. WHEN connecting to PostgreSQL THEN the system SHALL work with standard local PostgreSQL installations (localhost:5432)
3. WHEN processing different file types THEN the system SHALL allow custom file type handlers for personal preferences
4. WHEN extending functionality THEN the system SHALL provide simple configuration options for common use cases
5. WHEN monitoring the system THEN the system SHALL provide basic logging and status information for personal debugging