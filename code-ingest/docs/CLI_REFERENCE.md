# CLI Reference

Complete reference for all `code-ingest` commands and options.

## Global Options

```
code-ingest [OPTIONS] <COMMAND>

Options:
  --db-path <PATH>     Database path for PostgreSQL connection [default: ./analysis]
  -h, --help          Print help information
  -V, --version       Print version information
```

## Commands

### `ingest` - Ingest Repository or Folder

Ingest a GitHub repository or local folder into PostgreSQL for analysis.

```bash
code-ingest ingest [OPTIONS] <SOURCE>
```

#### Arguments
- `<SOURCE>` - GitHub repository URL or local folder path

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--token <TOKEN>` - GitHub personal access token (or set GITHUB_TOKEN env var)
- `--clone-path <PATH>` - Temporary clone path [default: /tmp/code-ingest]
- `--max-file-size <BYTES>` - Maximum file size to process [default: 10MB]
- `--max-concurrency <N>` - Maximum concurrent file processing [default: CPU cores]

#### Examples

```bash
# Ingest public GitHub repository
code-ingest ingest https://github.com/rust-lang/mdBook

# Ingest private repository with token
code-ingest ingest https://github.com/private/repo --token ghp_xxxxxxxxxxxx

# Ingest local folder
code-ingest ingest /path/to/your/project --db-path ./my-analysis

# Ingest with custom settings
code-ingest ingest https://github.com/large/repo \
  --max-concurrency 4 \
  --max-file-size 5242880 \
  --db-path ./analysis
```

#### Output
```
🚀 Starting ingestion...
📁 Source: https://github.com/rust-lang/mdBook
🗄️  Database: ./analysis
⏳ Cloning repository...
✅ Repository cloned successfully
📊 Processing files...
[████████████████████████████████] 847/847 files (100%)
✅ Ingestion completed successfully!

📈 Results:
   Table: INGEST_20240928143022
   Files processed: 847
   Duration: 2m 15s
   Throughput: 6.3 files/sec

🔍 Next steps:
   code-ingest list-tables --db-path ./analysis
   code-ingest sql "SELECT COUNT(*) FROM INGEST_20240928143022" --db-path ./analysis
```

---

### `sql` - Execute SQL Query

Execute raw SQL queries against ingested data.

```bash
code-ingest sql [OPTIONS] <QUERY>
```

#### Arguments
- `<QUERY>` - SQL query to execute

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--limit <N>` - Limit number of results [default: 100]
- `--offset <N>` - Offset for pagination [default: 0]
- `--format <FORMAT>` - Output format: table, json, csv [default: table]

#### Examples

```bash
# Count files by type
code-ingest sql "SELECT file_type, COUNT(*) FROM INGEST_20240928143022 GROUP BY file_type"

# Find authentication-related code
code-ingest sql "SELECT filepath, filename FROM INGEST_20240928143022 WHERE content_text LIKE '%authenticate%'" --limit 10

# Full-text search
code-ingest sql "SELECT filepath FROM INGEST_20240928143022 WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'database connection')"

# Export as JSON
code-ingest sql "SELECT * FROM INGEST_20240928143022 WHERE extension = 'rs'" --format json --limit 5

# Pagination
code-ingest sql "SELECT filepath, line_count FROM INGEST_20240928143022 ORDER BY line_count DESC" --limit 20 --offset 40
```

#### Output Formats

**Table Format (default):**
```
┌─────────────────────────────────┬──────────────┬───────────┐
│ filepath                        │ filename     │ file_type │
├─────────────────────────────────┼──────────────┼───────────┤
│ src/main.rs                     │ main.rs      │ direct_text │
│ src/lib.rs                      │ lib.rs       │ direct_text │
│ Cargo.toml                      │ Cargo.toml   │ direct_text │
└─────────────────────────────────┴──────────────┴───────────┘
```

**JSON Format:**
```json
[
  {
    "filepath": "src/main.rs",
    "filename": "main.rs",
    "file_type": "direct_text"
  }
]
```

---

### `list-tables` - List Ingestion Tables

List all available ingestion tables with metadata.

```bash
code-ingest list-tables [OPTIONS]
```

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--format <FORMAT>` - Output format: table, json [default: table]

#### Examples

```bash
# List all tables
code-ingest list-tables

# List with custom database path
code-ingest list-tables --db-path ./my-analysis

# JSON output
code-ingest list-tables --format json
```

#### Output
```
📊 Ingestion Tables

┌─────────────────────────┬─────────────────────────────────────┬───────────────┬──────────────┬─────────────────────┐
│ Table Name              │ Repository URL                      │ Files         │ Created      │ Duration            │
├─────────────────────────┼─────────────────────────────────────┼───────────────┼──────────────┼─────────────────────┤
│ INGEST_20240928143022   │ https://github.com/rust-lang/mdBook │ 847           │ 2 hours ago  │ 2m 15s              │
│ INGEST_20240928120000   │ /local/project                      │ 234           │ 5 hours ago  │ 45s                 │
│ INGEST_20240927180000   │ https://github.com/tokio-rs/tokio   │ 1,234         │ 1 day ago    │ 5m 30s              │
└─────────────────────────┴─────────────────────────────────────┴───────────────┴──────────────┴─────────────────────┘

💡 Use 'code-ingest sample --table <TABLE_NAME>' to explore table contents
```

---

### `sample` - Sample Table Data

Show sample rows from an ingestion table.

```bash
code-ingest sample [OPTIONS] --table <TABLE>
```

#### Options
- `--table <TABLE>` - Table name to sample (required)
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--limit <N>` - Number of sample rows [default: 5]
- `--columns <COLS>` - Comma-separated column names [default: all]

#### Examples

```bash
# Sample 5 rows from table
code-ingest sample --table INGEST_20240928143022

# Sample specific columns
code-ingest sample --table INGEST_20240928143022 --columns "filepath,filename,line_count" --limit 10

# Sample with custom database
code-ingest sample --table INGEST_20240928143022 --db-path ./my-analysis
```

#### Output
```
📋 Sample from INGEST_20240928143022 (5 rows)

┌─────────────────────────────────┬──────────────┬───────────┬──────────────┬────────────┐
│ filepath                        │ filename     │ extension │ file_type    │ line_count │
├─────────────────────────────────┼──────────────┼───────────┼──────────────┼────────────┤
│ src/main.rs                     │ main.rs      │ rs        │ direct_text  │ 45         │
│ src/lib.rs                      │ lib.rs       │ rs        │ direct_text  │ 123        │
│ Cargo.toml                      │ Cargo.toml   │ toml      │ direct_text  │ 18         │
│ README.md                       │ README.md    │ md        │ direct_text  │ 67         │
│ docs/guide.pdf                  │ guide.pdf    │ pdf       │ convertible  │ NULL       │
└─────────────────────────────────┴──────────────┴───────────┴──────────────┴────────────┘
```

---

### `describe` - Describe Table Schema

Show the schema and statistics for an ingestion table.

```bash
code-ingest describe [OPTIONS] --table <TABLE>
```

#### Options
- `--table <TABLE>` - Table name to describe (required)
- `--db-path <PATH>` - Database path [default: ./analysis]

#### Examples

```bash
# Describe table schema
code-ingest describe --table INGEST_20240928143022

# Describe with custom database
code-ingest describe --table INGEST_20240928143022 --db-path ./my-analysis
```

#### Output
```
📊 Table Schema: INGEST_20240928143022

┌─────────────────┬──────────────────┬─────────┬─────────────┬─────────────────────────────────┐
│ Column          │ Type             │ Null    │ Default     │ Description                     │
├─────────────────┼──────────────────┼─────────┼─────────────┼─────────────────────────────────┤
│ file_id         │ bigint           │ NO      │ nextval()   │ Primary key                     │
│ ingestion_id    │ bigint           │ NO      │             │ Foreign key to ingestion_meta   │
│ filepath        │ varchar          │ NO      │             │ Full file path                  │
│ filename        │ varchar          │ NO      │             │ File name only                  │
│ extension       │ varchar          │ YES     │             │ File extension                  │
│ file_size_bytes │ bigint           │ NO      │             │ File size in bytes              │
│ line_count      │ integer          │ YES     │             │ Number of lines                 │
│ word_count      │ integer          │ YES     │             │ Number of words                 │
│ token_count     │ integer          │ YES     │             │ Estimated token count           │
│ content_text    │ text             │ YES     │             │ File content (if text)          │
│ file_type       │ varchar          │ NO      │             │ Classification type             │
│ created_at      │ timestamp        │ NO      │ now()       │ Creation timestamp              │
└─────────────────┴──────────────────┴─────────┴─────────────┴─────────────────────────────────┘

📈 Table Statistics:
   Total rows: 847
   Total size: 15.2 MB
   Average file size: 18.4 KB
   File types:
     - direct_text: 789 (93.2%)
     - convertible: 12 (1.4%)
     - non_text: 46 (5.4%)

🔍 Indexes:
   - PRIMARY KEY (file_id)
   - FOREIGN KEY (ingestion_id)
   - GIN index on content_text (full-text search)
```

---

### `query-prepare` - Prepare Query for IDE Analysis

Execute a query and prepare results for IDE-based analysis workflow.

```bash
code-ingest query-prepare [OPTIONS] <QUERY> --temp-path <PATH> --tasks-file <PATH> --output-table <TABLE>
```

#### Arguments
- `<QUERY>` - SQL query to execute

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--temp-path <PATH>` - Temporary file for query results (required)
- `--tasks-file <PATH>` - Markdown file for analysis tasks (required)
- `--output-table <TABLE>` - Output table name for storing analysis results (required)

#### Examples

```bash
# Prepare authentication analysis
code-ingest query-prepare \
  "SELECT filepath, content_text FROM INGEST_20240928143022 WHERE content_text LIKE '%auth%'" \
  --temp-path ./auth-temp.txt \
  --tasks-file ./auth-tasks.md \
  --output-table QUERYRESULT_auth_analysis

# Prepare security analysis
code-ingest query-prepare \
  "SELECT filepath, content_text FROM INGEST_20240928143022 WHERE content_text ~* '(password|secret|token|key)'" \
  --temp-path ./security-temp.txt \
  --tasks-file ./security-tasks.md \
  --output-table QUERYRESULT_security_analysis
```

#### Output
```
🔍 Preparing query for IDE analysis...
📊 Query executed: 23 results found
📝 Results written to: ./auth-temp.txt
📋 Tasks file created: ./auth-tasks.md
🗄️  Output table created: QUERYRESULT_auth_analysis

🎯 Next steps:
   1. Open ./auth-tasks.md in your IDE
   2. Execute the analysis tasks
   3. Store results with: code-ingest store-result --output-table QUERYRESULT_auth_analysis --result-file <result.txt> --original-query "<query>"
```

---

### `store-result` - Store Analysis Results

Store analysis results back to the database with traceability.

```bash
code-ingest store-result [OPTIONS] --output-table <TABLE> --result-file <PATH> --original-query <QUERY>
```

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--output-table <TABLE>` - Output table name (required)
- `--result-file <PATH>` - File containing analysis results (required)
- `--original-query <QUERY>` - Original SQL query for traceability (required)

#### Examples

```bash
# Store authentication analysis results
code-ingest store-result \
  --output-table QUERYRESULT_auth_analysis \
  --result-file ./auth-result.txt \
  --original-query "SELECT filepath, content_text FROM INGEST_20240928143022 WHERE content_text LIKE '%auth%'"

# Store with custom database
code-ingest store-result \
  --db-path ./my-analysis \
  --output-table QUERYRESULT_security_analysis \
  --result-file ./security-result.txt \
  --original-query "SELECT * FROM INGEST_20240928143022 WHERE content_text ~* 'password'"
```

#### Output
```
💾 Storing analysis results...
📊 Result file: ./auth-result.txt (2.3 KB)
🗄️  Output table: QUERYRESULT_auth_analysis
✅ Results stored successfully!

🔍 Query stored results:
   code-ingest sql "SELECT * FROM QUERYRESULT_auth_analysis" --db-path ./analysis
```

---

### `generate-tasks` - Generate Batch Analysis Tasks

Generate structured analysis tasks for systematic code review.

```bash
code-ingest generate-tasks [OPTIONS] --sql <QUERY> --prompt-file <PATH> --output-table <TABLE> --tasks-file <PATH>
```

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--sql <QUERY>` - SQL query to select files for analysis (required)
- `--prompt-file <PATH>` - Markdown file with analysis prompts (required)
- `--output-table <TABLE>` - Output table for results (required)
- `--tasks-file <PATH>` - Generated tasks markdown file (required)

#### Examples

```bash
# Generate comprehensive analysis tasks
code-ingest generate-tasks \
  --sql "SELECT * FROM INGEST_20240928143022 WHERE extension IN ('rs', 'py', 'js')" \
  --prompt-file ./prompts/code-review.md \
  --output-table QUERYRESULT_comprehensive_analysis \
  --tasks-file ./comprehensive-tasks.md

# Generate security-focused tasks
code-ingest generate-tasks \
  --sql "SELECT * FROM INGEST_20240928143022 WHERE content_text ~* '(crypto|hash|encrypt)'" \
  --prompt-file ./prompts/security-review.md \
  --output-table QUERYRESULT_security_review \
  --tasks-file ./security-tasks.md
```

#### Output
```
📋 Generating analysis tasks...
🔍 Query executed: 234 files selected
📝 Prompt file: ./prompts/code-review.md
🎯 Dividing into 7 task groups (33-34 files each)
📄 Tasks file created: ./comprehensive-tasks.md
🗄️  Output table created: QUERYRESULT_comprehensive_analysis

📊 Task breakdown:
   - Group 1: Files 1-34 (Authentication modules)
   - Group 2: Files 35-67 (Database operations)
   - Group 3: Files 68-100 (API endpoints)
   - Group 4: Files 101-134 (Utility functions)
   - Group 5: Files 135-167 (Configuration)
   - Group 6: Files 168-200 (Tests)
   - Group 7: Files 201-234 (Documentation)

🎯 Next steps:
   Open ./comprehensive-tasks.md in your IDE and execute tasks systematically
```

---

### `print-to-md` - Export Query Results as Markdown Files

Export query results as individual markdown files for detailed analysis.

```bash
code-ingest print-to-md [OPTIONS] --table <TABLE> --sql <QUERY> --prefix <PREFIX> --location <PATH>
```

#### Options
- `--db-path <PATH>` - Database path [default: ./analysis]
- `--table <TABLE>` - Table name to query (required)
- `--sql <QUERY>` - SQL query to execute (required)
- `--prefix <PREFIX>` - Prefix for generated filenames (required)
- `--location <PATH>` - Directory for output files (required)

#### Examples

```bash
# Export security analysis results
code-ingest print-to-md \
  --table QUERYRESULT_security_analysis \
  --sql "SELECT * FROM QUERYRESULT_security_analysis WHERE risk_level = 'high'" \
  --prefix security-finding \
  --location ./security-reports/

# Export code review results
code-ingest print-to-md \
  --table QUERYRESULT_code_review \
  --sql "SELECT * FROM QUERYRESULT_code_review ORDER BY complexity_score DESC" \
  --prefix code-review \
  --location ./review-reports/
```

#### Output
```
📄 Exporting query results to markdown files...
🔍 Query executed: 15 results found
📁 Output location: ./security-reports/
📝 Generated files:
   - security-finding-00001.md
   - security-finding-00002.md
   - security-finding-00003.md
   ...
   - security-finding-00015.md

✅ Export completed: 15 files created
💡 Each file contains one row of results in readable markdown format
```

---

### `pg-start` - PostgreSQL Setup Assistant

Interactive guide for setting up PostgreSQL for code-ingest.

```bash
code-ingest pg-start
```

#### Output
```
🐘 PostgreSQL Setup Assistant

📊 System Information:
   OS: macOS 14.0 (Darwin)
   Architecture: arm64
   Shell: zsh

🔍 Checking PostgreSQL installation...
❌ PostgreSQL not found in PATH

📦 Step 1: PostgreSQL Installation

For macOS (recommended):
   brew install postgresql@15
   brew services start postgresql

For Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install postgresql-15 postgresql-contrib
   sudo systemctl start postgresql
   sudo systemctl enable postgresql

For Windows:
   Download from: https://www.postgresql.org/download/windows/

🔧 Step 2: Database Setup

After installation, create a database:
   createdb code_analysis

🔐 Step 3: Connection Configuration

Set environment variable (optional):
   export DATABASE_URL="postgresql://username:password@localhost:5432/code_analysis"

Or use --db-path flag:
   code-ingest ingest <repo> --db-path postgresql://localhost/code_analysis

🧪 Step 4: Test Connection

Verify setup:
   code-ingest list-tables --db-path postgresql://localhost/code_analysis

📚 Need help? Visit: https://docs.code-ingest.dev/setup
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | None |
| `GITHUB_TOKEN` | GitHub personal access token | None |
| `CODE_INGEST_MAX_CONCURRENCY` | Maximum concurrent file processing | CPU cores |
| `CODE_INGEST_MAX_MEMORY_MB` | Maximum memory usage in MB | 1024 |
| `RUST_LOG` | Logging level (error, warn, info, debug, trace) | info |

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Database connection error |
| 3 | Git operation error |
| 4 | File processing error |
| 5 | Invalid arguments |

## Configuration File

Create `~/.config/code-ingest/config.toml` for default settings:

```toml
[database]
default_path = "postgresql://localhost/code_analysis"
max_connections = 10
connection_timeout = 30

[processing]
max_concurrency = 8
max_file_size_mb = 10
max_memory_mb = 1024

[github]
# token = "ghp_xxxxxxxxxxxx"  # Uncomment and set your token

[output]
default_format = "table"
default_limit = 100
```

## Tips and Best Practices

### Performance Optimization
- Use `--max-concurrency` to match your system capabilities
- Set appropriate `--max-file-size` to avoid processing huge files
- Use database connection pooling for multiple operations
- Create indexes on frequently queried columns

### Query Optimization
- Use `LIMIT` for large result sets
- Leverage full-text search indexes for content queries
- Use `EXPLAIN ANALYZE` to optimize complex queries
- Consider materialized views for repeated analysis

### Workflow Integration
- Use `query-prepare` for systematic analysis workflows
- Store analysis results with `store-result` for traceability
- Export findings with `print-to-md` for documentation
- Automate with shell scripts for batch processing

### Security Considerations
- Store GitHub tokens securely (environment variables)
- Use read-only database users for query operations
- Sanitize file paths in queries to prevent injection
- Regularly audit stored analysis results