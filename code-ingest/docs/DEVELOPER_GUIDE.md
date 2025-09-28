# Developer Guide

Complete guide for developers who want to contribute to or extend code-ingest.

## 📋 Table of Contents

- [Development Setup](#-development-setup)
- [Architecture Overview](#-architecture-overview)
- [Code Organization](#-code-organization)
- [Building and Testing](#-building-and-testing)
- [Contributing Guidelines](#-contributing-guidelines)
- [Extending the System](#-extending-the-system)

## 🛠️ Development Setup

### Prerequisites

1. **Rust Toolchain** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   rustup update
   ```

2. **PostgreSQL** (12+)
   ```bash
   # macOS
   brew install postgresql@15
   brew services start postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql-15 postgresql-contrib libpq-dev
   sudo systemctl start postgresql
   ```

3. **Development Tools**
   ```bash
   # Code formatting and linting
   rustup component add rustfmt clippy
   
   # Coverage reporting
   cargo install cargo-llvm-cov
   
   # Security auditing
   cargo install cargo-audit
   ```

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/your-org/code-ingest.git
cd code-ingest/code-ingest

# Build in debug mode
cargo build

# Build optimized release
cargo build --release

# Run tests
cargo test

# Check code formatting
cargo fmt --check

# Run linter
cargo clippy -- -D warnings
```

## 🏗️ Architecture Overview

### High-Level Architecture

The system follows a layered architecture with clear separation of concerns:

1. **CLI Layer**: Command-line interface and user interaction
2. **Core Engine**: Main business logic and orchestration
3. **Processing Layer**: File classification and content extraction
4. **Database Layer**: PostgreSQL operations and schema management
5. **Git Layer**: Repository cloning and authentication

### Key Design Patterns

- **Repository Pattern**: Database operations abstracted behind traits
- **Strategy Pattern**: Different file processors for different file types
- **Builder Pattern**: Configuration objects with validation
- **RAII**: Automatic resource cleanup with Drop implementations

## 📁 Code Organization

```
code-ingest/
├── src/
│   ├── cli/                    # Command-line interface
│   ├── core/                   # Core business logic
│   ├── database/               # Database operations
│   ├── processing/             # File processing pipeline
│   ├── ingestion/              # Repository ingestion
│   ├── error.rs               # Error types and handling
│   ├── lib.rs                 # Library root
│   └── main.rs                # Binary entry point
├── tests/                      # Integration tests
├── benches/                    # Performance benchmarks
└── docs/                       # Documentation
```

## 🔨 Building and Testing

### Testing Strategy

```bash
# Run all unit tests
cargo test --lib

# Run integration tests (requires PostgreSQL)
export TEST_DATABASE_URL="postgresql://postgres@localhost/code_ingest_test"
cargo test --tests

# Run benchmarks
cargo bench

# Generate coverage report
cargo llvm-cov --html
```

### Code Quality

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-targets --all-features -- -D warnings

# Security audit
cargo audit
```

## 🤝 Contributing Guidelines

### Code Style

- Follow standard Rust naming conventions
- Use comprehensive documentation comments
- Include examples in documentation
- Write tests for all new functionality
- Maintain >90% test coverage

### Pull Request Process

1. Fork and create feature branch
2. Make changes with tests and documentation
3. Run quality checks (fmt, clippy, tests)
4. Submit pull request with clear description

## 🔧 Extending the System

### Adding New File Types

1. Update `FileClassifier` with new extensions
2. Add processing logic in appropriate processor
3. Update database schema if needed
4. Add comprehensive tests

### Adding New CLI Commands

1. Define command structure in `cli/mod.rs`
2. Implement command handler
3. Add integration tests
4. Update documentation

### Performance Considerations

- Use streaming processing for large datasets
- Implement proper resource cleanup
- Batch database operations
- Use structured concurrency patterns

This guide provides the essential information for developers to get started with contributing to code-ingest.