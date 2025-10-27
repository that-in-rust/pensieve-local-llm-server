# Pensieve Local LLM Server

A modular local LLM server built with independent Rust crates, following idiomatic Rust patterns for crate organization and workspace management.

## Architecture

This project uses a Cargo workspace with 7 independent crates:

- **pensieve-01**: Core foundation and configuration
- **pensieve-02**: Model loading and management
- **pensieve-03**: Inference engine
- **pensieve-04**: API server and HTTP interface
- **pensieve-05**: Prompt engineering and templating
- **pensieve-06**: Memory and conversation management
- **pensieve-07**: Monitoring and metrics

Each crate is designed to be:
- **Independent**: Can be used standalone outside this workspace
- **Focused**: Has a single, well-defined responsibility
- **Testable**: Comprehensive test coverage
- **Documented**: Clear API documentation

## Development

```bash
# Build all crates
cargo build

# Run tests for all crates
cargo test

# Run a specific crate
cargo run -p pensieve-04

# Check code formatting
cargo fmt --check

# Run clippy lints
cargo clippy -- -D warnings
```

## Usage

Each crate can be used independently in your own projects by adding it to your `Cargo.toml`:

```toml
[dependencies]
pensieve-04 = { version = "0.1.0", git = "https://github.com/amuldotexe/pensieve" }
```

## License

MIT OR Apache-2.0