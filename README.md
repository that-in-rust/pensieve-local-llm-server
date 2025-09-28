# Pensieve
part of github.com/that-in-rust/ org
Rust-based code analysis system. Ingests repositories into PostgreSQL, extracts patterns with semantic search, stores insights in queryable format.

## Quick Start

```bash
# Build
cargo build --release

# Setup database  
./target/release/code-ingest pg-start

# Ingest repository
./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path ./analysis

# Generate analysis tasks
./target/release/code-ingest generate-tasks \
  --sql "SELECT * FROM INGEST_20250928062949" \
  --output-table QUERYRESULT_analysis \
  --tasks-file ./tasks.md \
  --db-path ./analysis

# Execute tasks in IDE, results stored in QUERYRESULT_analysis table
```

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Development](#-development)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## Features

- **Enhanced Code Ingestion**: GitHub repos → PostgreSQL tables with multi-scale context windows
- **Hierarchical Analysis**: Automatic parent_filepath calculation and content aggregation
- **Pattern Analysis**: Extract code patterns and architectural insights with semantic search ready
- **Multi-Scale Context**: File → Module → System context windows populated during ingestion
- **Task Workflow**: Generate analysis tasks, execute in IDE, store results in QUERYRESULT tables
- **Export System**: Markdown files, JSON data, visualizations

### File Type Support

| Type | Extensions | Processing |
|------|------------|------------|
| **DirectText** | `.rs`, `.py`, `.js`, `.ts`, `.md`, `.txt`, `.json`, `.yaml`, `.sql`, `.sh`, `.c`, `.cpp`, `.java`, `.go`, `.rb`, `.php`, `.html`, `.css`, `.xml` | Direct content extraction with metrics |
| **Convertible** | `.pdf`, `.docx`, `.xlsx`, `.pptx` | External tool conversion (pdftotext, pandoc) |
| **Binary** | `.jpg`, `.png`, `.gif`, `.mp4`, `.exe`, `.bin`, `.zip` | Metadata-only storage |

### Performance Contracts
- **Ingestion Speed**: >100 files/second for typical repositories
- **Query Response**: <1 second for 10,000+ file repositories  
- **Memory Usage**: Constant memory regardless of repository size
- **Concurrent Processing**: Utilizes all available CPU cores

## 🛠️ Installation

### Prerequisites

1. **Rust** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **PostgreSQL** (12+)
   ```bash
   # macOS
   brew install postgresql
   brew services start postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   
   # Windows
   # Download from https://www.postgresql.org/download/windows/
   ```

3. **Optional Conversion Tools**
   ```bash
   # For PDF processing
   sudo apt-get install poppler-utils  # Linux
   brew install poppler                 # macOS
   
   # For Office document processing
   sudo apt-get install pandoc         # Linux
   brew install pandoc                 # macOS
   ```

### Install Code Ingest

#### From Crates.io (Recommended)
```bash
cargo install code-ingest
```

#### From Source
```bash
git clone https://github.com/that-in-rust/code-ingest.git
cd code-ingest/code-ingest
cargo install --path .
```

#### Pre-built Binaries
Download from [GitHub Releases](https://github.com/that-in-rust/code-ingest/releases)

### Database Setup

```bash
# Get setup guidance
code-ingest pg-start

# Create database (if needed)
createdb code_analysis

# Set environment variable (optional)
export DATABASE_URL="postgresql://username:password@localhost:5432/code_analysis"
```

## 📖 Usage

### Basic Ingestion

#### GitHub Repository
```bash
# Public repository
code-ingest ingest https://github.com/rust-lang/mdBook --db-path ./analysis

# Private repository (requires GitHub token)
export GITHUB_TOKEN="your_token_here"
code-ingest ingest https://github.com/private/repo --db-path ./analysis
```

#### Local Folder
```bash
# Ingest local codebase
code-ingest ingest /path/to/your/project --db-path ./analysis
```

### Querying Data

#### Basic SQL Queries
```bash
# List all Rust files
code-ingest sql "SELECT filepath, line_count FROM INGEST_20240928143022 WHERE extension = 'rs'" --db-path ./analysis

# Search for authentication code
code-ingest sql "SELECT filepath, content_text FROM INGEST_20240928143022 WHERE content_text LIKE '%authenticate%'" --db-path ./analysis

# Full-text search
code-ingest sql "SELECT filepath FROM INGEST_20240928143022 WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'database connection')" --db-path ./analysis
```

#### Table Management
```bash
# List all ingestion tables
code-ingest list-tables --db-path ./analysis

# Sample data from a table
code-ingest sample --table INGEST_20240928143022 --limit 5 --db-path ./analysis

# Describe table structure
code-ingest describe --table INGEST_20240928143022 --db-path ./analysis
```

### IDE Integration Workflow

#### 1. Prepare Query for Analysis
```bash
code-ingest query-prepare \
  "SELECT filepath, content_text FROM INGEST_20240928143022 WHERE content_text LIKE '%auth%'" \
  --db-path ./analysis \
  --temp-path ./auth-temp.txt \
  --tasks-file ./auth-tasks.md \
  --output-table QUERYRESULT_auth_analysis
```

#### 2. Execute Analysis Tasks
Open `auth-tasks.md` in your IDE (Kiro) and execute the generated analysis tasks.

#### 3. Store Analysis Results
```bash
code-ingest store-result \
  --db-path ./analysis \
  --output-table QUERYRESULT_auth_analysis \
  --result-file ./auth-result.txt \
  --original-query "SELECT filepath, content_text FROM INGEST_20240928143022 WHERE content_text LIKE '%auth%'"
```

### Advanced Usage

#### Batch Task Generation
```bash
code-ingest generate-tasks \
  --sql "SELECT * FROM INGEST_20240928143022" \
  --prompt-file analyze.md \
  --output-table QUERYRESULT_results \
  --tasks-file ./tasks.md \
  --db-path ./analysis
```

#### Export to Markdown Files
```bash
code-ingest print-to-md \
  --db-path ./analysis \
  --table QUERYRESULT_results \
  --sql "SELECT * FROM QUERYRESULT_results WHERE analysis_type = 'security'" \
  --prefix security-analysis \
  --location ./md-files/
```

## 📊 Performance

### Benchmarks

| Operation | Performance | Notes |
|-----------|-------------|-------|
| File Classification | 40,000 ops/sec | Extension-based classification |
| Text Processing | 150 files/sec | 1KB average file size |
| Database Insertion | 500 files/sec | PostgreSQL with indexes |
| Query Response | <100ms | 10,000 file repository |
| Memory Usage | <50MB | Constant regardless of repo size |

### Performance Tuning

#### Database Optimization
```sql
-- Increase shared_buffers for better performance
-- Add to postgresql.conf:
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB
```

#### Concurrent Processing
```bash
# Adjust concurrency based on your system
export CODE_INGEST_MAX_CONCURRENCY=8  # Default: CPU cores
export CODE_INGEST_MAX_MEMORY_MB=512   # Default: 1GB
```

## 🏗️ Architecture

### System Overview

```mermaid
graph TD
    CLI[CLI Interface] --> Core[Core Engine]
    Core --> Clone[Git Clone Module]
    Core --> Process[File Processor]
    Core --> Store[PostgreSQL Store]
    Core --> Query[Query Engine]
    
    Clone --> FS[File System]
    Process --> Classify[File Classifier]
    Process --> Extract[Content Extractor]
    Store --> PG[(PostgreSQL)]
    Query --> PG
    
    subgraph "File Processing Pipeline"
        Classify --> DirectText[Type 1: Direct Text]
        Classify --> Convertible[Type 2: Convertible]
        Classify --> Binary[Type 3: Binary]
        
        DirectText --> TextMetrics[Line/Word/Token Count]
        Convertible --> Convert[External Conversion]
        Binary --> MetaOnly[Metadata Only]
    end
```

### Enhanced Database Schema

```sql
-- Metadata table for tracking ingestions
CREATE TABLE ingestion_meta (
    ingestion_id BIGSERIAL PRIMARY KEY,
    repo_url VARCHAR NOT NULL,
    local_path VARCHAR NOT NULL,
    start_timestamp_unix BIGINT NOT NULL,
    end_timestamp_unix BIGINT,
    table_name VARCHAR NOT NULL,
    total_files_processed INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enhanced timestamped ingestion tables (INGEST_YYYYMMDDHHMMSS)
CREATE TABLE INGEST_YYYYMMDDHHMMSS (
    file_id BIGSERIAL PRIMARY KEY,
    ingestion_id BIGINT REFERENCES ingestion_meta(ingestion_id),
    filepath VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    extension VARCHAR,
    file_size_bytes BIGINT NOT NULL,
    line_count INTEGER,
    word_count INTEGER,
    token_count INTEGER,
    content_text TEXT,
    file_type VARCHAR NOT NULL CHECK (file_type IN ('direct_text', 'convertible', 'non_text')),
    conversion_command VARCHAR,
    relative_path VARCHAR NOT NULL,
    absolute_path VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Multi-scale context columns for knowledge arbitrage
    parent_filepath VARCHAR,          -- Calculated: go back by 1 slash
    l1_window_content TEXT,           -- Directory-level concatenation  
    l2_window_content TEXT,           -- System-level concatenation
    ast_patterns JSONB                -- Pattern matches for semantic search
);

-- Enhanced indexes
CREATE INDEX idx_content_fts ON INGEST_YYYYMMDDHHMMSS 
USING gin(to_tsvector('english', content_text));
CREATE INDEX idx_parent_filepath ON INGEST_YYYYMMDDHHMMSS(parent_filepath);
CREATE INDEX idx_ast_patterns ON INGEST_YYYYMMDDHHMMSS USING gin(ast_patterns);
```

### Key Components

- **File Classifier**: Extension-based three-type classification
- **Content Extractor**: Parallel processing with external tool integration
- **Database Manager**: Connection pooling and schema management
- **Query Engine**: SQL execution with result formatting
- **Git Clone Manager**: Repository cloning with authentication

## 🔧 Development

### Building from Source

```bash
git clone https://github.com/that-in-rust/code-ingest.git
cd code-ingest/code-ingest

# Install dependencies
cargo build

# Run tests
cargo test

# Run with sample data
cargo run -- ingest ./test-data --db-path ./test-analysis
```

### Testing

```bash
# Unit tests
cargo test --lib

# Integration tests (requires PostgreSQL)
export TEST_DATABASE_URL="postgresql://postgres:password@localhost:5432/code_ingest_test"
cargo test --tests

# Property-based tests
cargo test property_

# Performance benchmarks
cargo bench

# Coverage report
cargo llvm-cov --html
```

### Project Structure

```
code-ingest/
├── src/
│   ├── cli/           # Command-line interface
│   ├── core/          # Core ingestion engine
│   ├── database/      # PostgreSQL operations
│   ├── processing/    # File processing pipeline
│   ├── ingestion/     # Batch processing and Git operations
│   ├── error.rs       # Error handling
│   └── lib.rs         # Library root
├── tests/             # Integration tests
├── benches/           # Performance benchmarks
└── examples/          # Usage examples
```

## 🐛 Troubleshooting

### Common Issues

#### PostgreSQL Connection Failed
```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify database exists
psql -h localhost -U postgres -l

# Create database if missing
createdb code_analysis
```

#### GitHub Authentication Failed
```bash
# Set GitHub token
export GITHUB_TOKEN="your_personal_access_token"

# Verify token has repo access
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

#### Conversion Tools Missing
```bash
# Install PDF processing
sudo apt-get install poppler-utils  # Linux
brew install poppler               # macOS

# Install Office document processing
sudo apt-get install pandoc       # Linux
brew install pandoc              # macOS
```

#### Performance Issues
```bash
# Check system resources
htop

# Monitor PostgreSQL performance
SELECT * FROM pg_stat_activity;

# Adjust concurrency
export CODE_INGEST_MAX_CONCURRENCY=4  # Reduce if system is overloaded
```

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug
code-ingest ingest https://github.com/example/repo --db-path ./analysis

# Trace SQL queries
export RUST_LOG=sqlx=debug
```

### Performance Profiling

```bash
# Profile with perf (Linux)
perf record --call-graph dwarf cargo run --release -- ingest large-repo --db-path ./analysis
perf report

# Memory profiling with valgrind
valgrind --tool=massif cargo run --release -- ingest repo --db-path ./analysis
```

## 📚 Examples

Comprehensive examples for various use cases:

- **[Enhanced Ingestion](examples/xsv-enhanced-ingestion/)** - Multi-scale context windows with XSV codebase example
- **[Basic Usage](examples/basic_usage.md)** - Getting started with repository ingestion and basic queries
- **[Security Analysis](examples/security_analysis.md)** - Finding security vulnerabilities and sensitive information
- **[Architecture Analysis](examples/architecture_analysis.md)** - Understanding codebase structure and design patterns
- **[IDE Integration](examples/ide_integration.md)** - Systematic analysis workflows with IDE support
- **[Custom Queries](examples/custom_queries.md)** - Advanced SQL patterns for specialized analysis

See the [examples directory](examples/) for complete tutorials and use case scenarios.

## 📖 Documentation

### User Documentation
- **[CLI Reference](docs/CLI_REFERENCE.md)** - Complete command reference and options
- **[Performance Guide](docs/PERFORMANCE.md)** - Benchmarks, optimization, and system requirements
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Developer Documentation
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Architecture, contributing, and extending the system
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[Changelog](CHANGELOG.md)** - Version history and release notes

### Quick Installation
```bash
# Automated installation (recommended)
curl -sSL https://raw.githubusercontent.com/that-in-rust/code-ingest/main/install.sh | bash

# Or install manually
cargo install code-ingest
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `cargo install cargo-llvm-cov cargo-audit`
4. Make your changes and add tests
5. Run the test suite: `cargo test`
6. Submit a pull request

### Code Style

- Follow Rust standard formatting: `cargo fmt`
- Pass all lints: `cargo clippy`
- Maintain test coverage >90%
- Add documentation for public APIs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SQLx](https://github.com/launchbadge/sqlx) for excellent PostgreSQL integration
- [Tokio](https://tokio.rs/) for async runtime
- [Clap](https://clap.rs/) for CLI argument parsing
- [Git2](https://github.com/rust-lang/git2-rs) for Git operations

## 📞 Support

- 📖 [Documentation](https://docs.rs/code-ingest)
- 🐛 [Issue Tracker](https://github.com/that-in-rust/code-ingest/issues)
- 💬 [Discussions](https://github.com/that-in-rust/code-ingest/discussions)
- 📧 Email: support@that-in-rust.com

---

**Made with ❤️ by the Code Ingest Team**