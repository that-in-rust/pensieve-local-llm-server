# Contributing to Code Ingest

Thank you for your interest in contributing to code-ingest! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up development environment** (see [Developer Guide](docs/DEVELOPER_GUIDE.md))
4. **Create a feature branch** from `main`
5. **Make your changes** with tests and documentation
6. **Submit a pull request**

## 📋 Development Setup

### Prerequisites

- Rust 1.70+ with `rustfmt` and `clippy`
- PostgreSQL 12+ for testing
- Git for version control

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/code-ingest.git
cd code-ingest/code-ingest

# Install development dependencies
cargo install cargo-llvm-cov cargo-audit

# Set up test database
createdb code_ingest_test
export TEST_DATABASE_URL="postgresql://postgres@localhost/code_ingest_test"

# Run tests to verify setup
cargo test
```

## 🎯 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes** - Fix issues in existing functionality
- **New features** - Add new capabilities to the system
- **Documentation** - Improve or add documentation
- **Performance improvements** - Optimize existing code
- **Tests** - Add or improve test coverage
- **Examples** - Create usage examples and tutorials

### Finding Work

- Check [GitHub Issues](https://github.com/that-in-rust/code-ingest/issues) for open tasks
- Look for issues labeled `good first issue` for newcomers
- Issues labeled `help wanted` are great for contributors
- Feel free to propose new features by opening an issue first

## 📝 Development Guidelines

### Code Style

We follow standard Rust conventions:

```bash
# Format code before committing
cargo fmt

# Check for linting issues
cargo clippy --all-targets --all-features -- -D warnings

# Run security audit
cargo audit
```

### Testing Requirements

All contributions must include appropriate tests:

```bash
# Run unit tests
cargo test --lib

# Run integration tests
cargo test --tests

# Check test coverage (aim for >90%)
cargo llvm-cov --html

# Run property-based tests
cargo test property_
```

### Documentation Standards

- All public APIs must have documentation comments
- Include examples in documentation where helpful
- Update relevant documentation files for new features
- Add entries to examples/ directory for significant features

Example documentation:

```rust
/// Classifies a file based on its extension and content.
///
/// This function determines whether a file should be processed as direct text,
/// converted from another format, or treated as binary data only.
///
/// # Arguments
///
/// * `path` - The file path to classify
///
/// # Returns
///
/// Returns the `FileType` classification for the file.
///
/// # Examples
///
/// ```rust
/// use code_ingest::processing::classify_file;
/// use std::path::Path;
///
/// let file_type = classify_file(Path::new("example.rs"));
/// assert_eq!(file_type, FileType::DirectText);
/// ```
pub fn classify_file(path: &Path) -> FileType {
    // Implementation
}
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create an issue** for significant changes to discuss the approach
2. **Write tests** for your changes
3. **Update documentation** as needed
4. **Run the full test suite** and ensure it passes
5. **Check code formatting** and linting

### Pull Request Checklist

- [ ] Code follows Rust style guidelines (`cargo fmt`)
- [ ] All tests pass (`cargo test`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation is updated for public API changes
- [ ] Examples are added for new features
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes and motivation

### Commit Message Format

Use clear, descriptive commit messages:

```
feat: add support for YAML file processing

- Implement YAML file classification
- Add yaml-rust dependency for parsing
- Include comprehensive tests for YAML processing
- Update documentation with YAML examples

Closes #123
```

Commit types:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### Review Process

1. **Automated checks** run on all PRs (CI, tests, linting)
2. **Code review** by maintainers
3. **Feedback incorporation** - address review comments
4. **Final approval** and merge by maintainers

## 🧪 Testing Guidelines

### Test Categories

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_file_classification() {
        let classifier = FileClassifier::new();
        assert_eq!(
            classifier.classify_file(Path::new("test.rs")), 
            FileType::DirectText
        );
    }
    
    #[tokio::test]
    async fn test_async_processing() {
        let processor = FileProcessor::new();
        let result = processor.process_file(Path::new("test.rs")).await;
        assert!(result.is_ok());
    }
}
```

#### Integration Tests
```rust
// tests/integration_test.rs
use code_ingest::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_end_to_end_ingestion() {
    let temp_dir = TempDir::new().unwrap();
    let config = IngestionConfig::builder()
        .source("./test-data")
        .db_path(temp_dir.path().join("test.db"))
        .build()
        .unwrap();
    
    let engine = IngestionEngine::new(config).await.unwrap();
    let result = engine.ingest().await.unwrap();
    
    assert!(result.files_processed > 0);
}
```

#### Property-Based Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_file_path_handling(path in ".*") {
        // Test that file path handling is robust
        let result = normalize_file_path(&path);
        prop_assert!(result.is_ok() || is_expected_error(&result));
    }
}
```

### Performance Tests

Include benchmarks for performance-critical code:

```rust
// benches/processing_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use code_ingest::processing::*;

fn benchmark_file_processing(c: &mut Criterion) {
    let processor = FileProcessor::new();
    let test_files = generate_test_files(1000);
    
    c.bench_function("process_files", |b| {
        b.iter(|| {
            for file in &test_files {
                black_box(processor.process_file(black_box(file)));
            }
        })
    });
}

criterion_group!(benches, benchmark_file_processing);
criterion_main!(benches);
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if it's already fixed
3. **Gather system information** (OS, Rust version, PostgreSQL version)
4. **Create a minimal reproduction** if possible

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened, including error messages.

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Rust version: [e.g. 1.73.0]
- PostgreSQL version: [e.g. 15.4]
- code-ingest version: [e.g. 0.1.0]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing issues** for similar requests
2. **Consider the scope** - does it fit the project goals?
3. **Think about implementation** - how might it work?

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Additional context**
Any other context, mockups, or examples.
```

## 🏗️ Architecture Guidelines

### Design Principles

1. **Modularity** - Keep components focused and loosely coupled
2. **Testability** - Design for easy testing with dependency injection
3. **Performance** - Maintain streaming processing and constant memory usage
4. **Error Handling** - Use structured errors with proper context
5. **Documentation** - Code should be self-documenting with good names

### Adding New Features

#### File Type Support

To add support for a new file type:

1. Update `FileClassifier` with new extensions
2. Add processing logic in appropriate processor
3. Update database schema if needed
4. Add comprehensive tests
5. Update documentation and examples

#### CLI Commands

To add a new CLI command:

1. Define command structure in `cli/mod.rs`
2. Implement command handler
3. Add integration tests
4. Update CLI reference documentation
5. Add usage examples

#### Database Backends

To add a new database backend:

1. Implement the `DatabaseBackend` trait
2. Add connection management
3. Implement schema operations
4. Add comprehensive tests
5. Update configuration documentation

## 📊 Performance Considerations

### Performance Requirements

- Maintain >100 files/second processing throughput
- Keep memory usage constant regardless of repository size
- Ensure query response times <1 second for typical repositories
- Preserve streaming processing architecture

### Benchmarking

Always benchmark performance-critical changes:

```bash
# Run benchmarks before changes
cargo bench --bench performance_benchmarks > baseline.txt

# Make your changes

# Run benchmarks after changes
cargo bench --bench performance_benchmarks > modified.txt

# Compare results
cargo bench --bench performance_benchmarks -- --baseline baseline
```

## 🔒 Security Guidelines

### Security Considerations

- Validate all user inputs
- Use parameterized queries for database operations
- Avoid unsafe code unless absolutely necessary
- Handle secrets securely (no hardcoding)
- Follow secure coding practices

### Security Review

Security-sensitive changes require additional review:

- Authentication and authorization code
- Input validation and sanitization
- Cryptographic operations
- External command execution
- File system operations

## 📚 Resources

### Documentation

- [Developer Guide](docs/DEVELOPER_GUIDE.md) - Comprehensive development guide
- [CLI Reference](docs/CLI_REFERENCE.md) - Complete command reference
- [Performance Guide](docs/PERFORMANCE.md) - Performance optimization
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Examples

- [Basic Usage](examples/basic_usage.md) - Getting started
- [Security Analysis](examples/security_analysis.md) - Security use cases
- [Architecture Analysis](examples/architecture_analysis.md) - Understanding codebases
- [IDE Integration](examples/ide_integration.md) - IDE workflow integration

### Community

- [GitHub Discussions](https://github.com/that-in-rust/code-ingest/discussions) - Questions and ideas
- [Issue Tracker](https://github.com/that-in-rust/code-ingest/issues) - Bug reports and feature requests
- [Discord](https://discord.gg/code-ingest) - Real-time chat

## 📄 License

By contributing to code-ingest, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors are recognized in:

- [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- GitHub contributor statistics
- Release notes for significant contributions
- Project documentation for major features

Thank you for contributing to code-ingest! 🎉