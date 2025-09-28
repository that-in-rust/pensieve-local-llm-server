# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of code-ingest
- Complete documentation suite
- Comprehensive test coverage
- Performance benchmarks
- Example usage scenarios

## [0.1.0] - 2024-09-28

### Added
- **Core Ingestion Engine**
  - GitHub repository cloning with authentication support
  - Local folder processing capability
  - Three-type file classification (DirectText, Convertible, Binary)
  - Parallel file processing with configurable concurrency
  - Streaming architecture for constant memory usage

- **Database Integration**
  - PostgreSQL storage with timestamped tables
  - Full-text search indexing
  - Ingestion metadata tracking
  - Schema management and migrations

- **File Processing Pipeline**
  - Direct text extraction for 20+ file types
  - External tool integration (pdftotext, pandoc) for document conversion
  - Binary file metadata extraction
  - Content metrics calculation (lines, words, tokens)

- **CLI Interface**
  - `ingest` command for repository/folder ingestion
  - `sql` command for direct SQL query execution
  - `list-tables` command for ingestion table management
  - `sample` command for data exploration
  - `pg-start` command for PostgreSQL setup guidance

- **IDE Integration Workflow**
  - `query-prepare` command for systematic analysis preparation
  - `store-result` command for analysis result persistence
  - `generate-tasks` command for batch analysis task creation
  - `print-to-md` command for result export

- **Performance Features**
  - >100 files/second processing throughput
  - <1 second query response for 10K+ file repositories
  - Constant memory usage regardless of repository size
  - Concurrent processing utilizing all CPU cores

- **Error Handling**
  - Structured error hierarchy with actionable messages
  - Graceful degradation for missing dependencies
  - Resume capability for interrupted ingestions
  - Comprehensive logging and debugging support

- **Testing Infrastructure**
  - Unit tests with >90% coverage
  - Integration tests for end-to-end workflows
  - Property-based tests for file classification
  - Performance benchmarks and regression detection
  - Continuous integration with automated testing

- **Documentation**
  - Comprehensive README with quick start guide
  - Complete CLI reference documentation
  - Developer guide for contributors
  - Performance benchmarks and system requirements
  - Troubleshooting guide for common issues
  - Usage examples for various scenarios:
    - Basic repository analysis
    - Security vulnerability scanning
    - Architecture documentation
    - IDE integration workflows
    - Custom SQL query patterns

- **Installation and Setup**
  - Automated installation script for multiple platforms
  - Docker support for containerized deployment
  - Homebrew formula for macOS installation
  - Comprehensive setup documentation

### Technical Details

- **Languages**: Rust 1.70+
- **Database**: PostgreSQL 12+
- **Dependencies**: 
  - `clap` for CLI argument parsing
  - `tokio` for async runtime
  - `sqlx` for PostgreSQL integration
  - `git2` for repository operations
  - `serde` for serialization
  - `anyhow`/`thiserror` for error handling

- **Architecture**: 
  - Layered architecture with clear separation of concerns
  - Trait-based dependency injection for testability
  - RAII resource management
  - Streaming processing pipeline

- **Performance Contracts**:
  - File processing: >100 files/second for typical text files
  - Query response: <1 second for repositories with 10,000+ files
  - Memory usage: <100MB peak during ingestion
  - Database growth: <2x source repository size including indexes

### Known Limitations

- Binary file content extraction not supported (metadata only)
- Limited to PostgreSQL database backend
- External tool dependencies for document conversion
- Single-machine deployment only (no distributed processing)

### Security Considerations

- GitHub token handling through environment variables
- SQL injection prevention through parameterized queries
- Input validation for all user-provided data
- Secure temporary file handling

## [0.0.1] - 2024-09-01

### Added
- Initial project structure
- Basic file classification prototype
- PostgreSQL schema design
- Core CLI framework

---

## Release Notes

### Version 0.1.0 - "Foundation Release"

This initial release establishes code-ingest as a comprehensive solution for transforming GitHub repositories into queryable PostgreSQL databases. The system is designed for developers who need to quickly understand unfamiliar codebases through systematic analysis.

**Key Highlights:**
- **Production Ready**: Comprehensive error handling, logging, and recovery mechanisms
- **High Performance**: Optimized for speed with streaming processing and parallel execution
- **Developer Friendly**: Rich CLI interface with extensive documentation and examples
- **Extensible**: Clean architecture supporting future enhancements and integrations
- **Well Tested**: >90% test coverage with multiple testing strategies

**Use Cases Supported:**
- New developer onboarding and codebase exploration
- Security auditing and vulnerability assessment
- Architecture documentation and system understanding
- Code quality analysis and technical debt identification
- Research and academic analysis of software projects

**Getting Started:**
```bash
# Install code-ingest
cargo install code-ingest

# Set up PostgreSQL
code-ingest pg-start

# Ingest your first repository
code-ingest ingest https://github.com/rust-lang/mdBook --db-path ./analysis

# Start exploring
code-ingest sql "SELECT COUNT(*) FROM INGEST_*" --db-path ./analysis
```

For detailed usage instructions, see the [README](README.md) and [examples](examples/) directory.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/code-ingest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/code-ingest/discussions)