# Pensieve CLI Tool

A simple command-line tool designed to quickly ingest text files into a clean, deduplicated database for LLM processing.

## Project Structure

```
src/
├── lib.rs          # Main library entry point with module exports
├── main.rs         # CLI application entry point
├── cli.rs          # Command-line interface and argument parsing
├── types.rs        # Core data structures and type definitions
├── errors.rs       # Comprehensive error handling hierarchy
├── scanner.rs      # File system scanning and metadata extraction
├── extractor.rs    # Content extraction from various file formats
└── database.rs     # Database operations and schema management
```

## Core Data Structures

- **FileMetadata**: Comprehensive file information including hash, timestamps, and processing status
- **ProcessingStatus**: Tracks file processing state (Pending, Processed, Error, etc.)
- **DuplicateStatus**: Manages file-level deduplication (Unique, Canonical, Duplicate)
- **Paragraph**: Content chunks with metadata for LLM processing
- **ParagraphSource**: Links paragraphs to their source files

## Dependencies

- **clap**: CLI argument parsing with derive macros
- **sqlx**: Async SQLite database operations
- **tokio**: Async runtime for concurrent processing
- **sha2**: SHA-256 hashing for content deduplication
- **walkdir**: Recursive directory traversal
- **mime_guess**: File type detection
- **thiserror/anyhow**: Structured error handling
- **chrono**: Date/time handling
- **uuid**: Unique identifier generation

## Usage

```bash
# Show help
cargo run -- --help

# Initialize database
cargo run -- init --database pensieve.db

# Process directory
cargo run -- --input /path/to/files --database pensieve.db

# Dry run mode
cargo run -- --input /path/to/files --database pensieve.db --dry-run

# Verbose output
cargo run -- --input /path/to/files --database pensieve.db --verbose
```

## Architecture

The tool follows a layered architecture:

1. **CLI Layer**: Argument parsing and user interface
2. **Orchestration Layer**: Coordinates scanning, extraction, and storage
3. **Processing Layer**: File type detection, content extraction, deduplication
4. **Storage Layer**: SQLite database operations and schema management

## Development Status

This is the initial project structure implementation (Task 1). Core interfaces and data structures are defined, with placeholder implementations for future tasks.

## Requirements Addressed

- **5.1**: CLI interface with clap for argument parsing
- **5.3**: Basic error handling with clear messages
- **5.5**: Native Rust binary with required dependencies