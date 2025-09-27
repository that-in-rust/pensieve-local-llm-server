# Requirements Document

## Introduction

**The Problem:** Developers waste hours trying to understand unfamiliar codebases. Current tools (GitHub search, grep, IDE search) are too slow and shallow for semantic code understanding at scale.

**The Solution:** A Rust-based code ingestion system that clones GitHub repos, intelligently processes all file types, and stores structured data in a personal PostgreSQL database for lightning-fast semantic search.

**Success Metric:** Reduce time to understand a new codebase from hours to minutes.

## Core User Journey

### Initial Setup
1. Developer installs PostgreSQL locally or uses Docker: `brew install postgresql` or `docker run -d postgres`
2. Sets DATABASE_URL: `export DATABASE_URL=postgresql://postgres@localhost:5432/code_ingest`
3. Runs setup: `code-ingest setup` (creates database and tables)

### Daily Usage
4. Developer encounters unfamiliar codebase (new job, open source contribution, debugging)
5. Runs: `code-ingest https://github.com/user/repo --local-path ./repos`
6. System clones repo, processes all files, stores in PostgreSQL with full-text search
7. Developer runs SQL query: `code-ingest sql "SELECT filepath, content_text FROM files_20250927120000 WHERE content_text LIKE '%auth%'"`
8. Gets clean terminal output that can be piped to LLM: `code-ingest sql "..." | llm -f analyze-code.md`
9. LLM receives SQL results and analyzes using the provided prompt template

## Requirements

### Requirement 1: Repository Cloning and Local Storage

**User Story:** As a developer, I want to ingest code from GitHub repositories to a user-specified local folder, so that I can have persistent access to the codebase for analysis.

#### Acceptance Criteria

1. WHEN I provide a GitHub repository URL and local folder path THEN the system SHALL clone the repository to that specified location
2. WHEN I provide a GitHub Personal Access Token THEN the system SHALL authenticate and access private repositories
3. WHEN I specify a branch, tag, or commit hash THEN the system SHALL checkout that specific reference
4. WHEN the local folder already exists THEN the system SHALL either update the existing clone or create a new timestamped subdirectory
5. IF the repository does not exist or is inaccessible THEN the system SHALL return a clear error message

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

1. WHEN starting an ingestion query THEN the system SHALL create a timestamped table in format `files_YYYYMMDDHHMMSS`
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
    
    FILES_YYYYMMDDHHMMSS {
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
    
    INGESTION_META ||--o{ FILES_YYYYMMDDHHMMSS : "has many files"
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

**User Story:** As a developer, I want to run SQL queries against the code database and get terminal output that can be piped directly to LLM tools, so that I can analyze code using custom prompts.

#### Acceptance Criteria

1. WHEN I run a SQL query THEN the system SHALL output results in clean, readable terminal format
2. WHEN piping to LLM tools THEN the system SHALL format output without extra formatting or colors that interfere with text processing
3. WHEN querying file contents THEN the system SHALL output filepath and content in a structured way that LLMs can parse
4. WHEN results are large THEN the system SHALL support result limiting and pagination through SQL LIMIT clauses
5. WHEN combining with prompt files THEN the system SHALL produce output that works seamlessly with `llm -f prompt.md` workflows

### Requirement 9: PostgreSQL Database Setup and Connection

**User Story:** As a developer, I want clear instructions and scripts for setting up PostgreSQL and connecting the code ingestion system to my database, so that I can get started quickly without database administration complexity.

#### Acceptance Criteria

1. WHEN setting up the system THEN the system SHALL provide setup scripts or clear documentation for PostgreSQL installation
2. WHEN connecting to PostgreSQL THEN the system SHALL accept DATABASE_URL environment variable in standard PostgreSQL format: `postgresql://user:password@host:port/database`
3. WHEN the database doesn't exist THEN the system SHALL provide scripts to create the required database and initial schema
4. WHEN running for the first time THEN the system SHALL automatically create required tables (`ingestion_meta` and initial schema)
5. WHEN PostgreSQL is not running THEN the system SHALL provide clear error messages with troubleshooting guidance

### Requirement 10: Personal Configuration and Setup

**User Story:** As a developer, I want to easily configure the system for my personal development environment, so that it works seamlessly with my local PostgreSQL setup.

#### Acceptance Criteria

1. WHEN setting up the system THEN the system SHALL support configuration via environment variables and local config files
2. WHEN connecting to PostgreSQL THEN the system SHALL work with standard local PostgreSQL installations (localhost:5432)
3. WHEN processing different file types THEN the system SHALL allow custom file type handlers for personal preferences
4. WHEN extending functionality THEN the system SHALL provide simple configuration options for common use cases
5. WHEN monitoring the system THEN the system SHALL provide basic logging and status information for personal debugging