# Implementation Plan

- [ ] 1. Set up project structure and core interfaces
  - Create Cargo.toml with required dependencies (clap, sqlx, tokio, sha2, walkdir, mime_guess, thiserror, anyhow)
  - Define core data structures (FileMetadata, ProcessingStatus, DuplicateStatus, error types)
  - Create module structure (cli, scanner, extractor, database, errors)
  - _Requirements: 5.1, 5.3, 5.5_

- [ ] 2. Implement CLI interface and argument parsing
  - Create CLI struct with clap derive for input directory and database path arguments
  - Add --help flag with basic usage instructions
  - Implement argument validation (directory exists, database path writable)
  - Add basic error messages for missing arguments
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 3. Create database schema and connection management
  - Implement SQLite database initialization with WAL mode
  - Create all tables (files, paragraphs, paragraph_sources, processing_errors) with proper indexes
  - Write database connection pool management
  - Add database migration support for schema evolution
  - _Requirements: 1.7, 2.3, 2.4_

- [ ] 4. Implement file type detection and filtering system
  - Create file type classification (Tier 1 native, Tier 2 external, binary exclusions)
  - Implement MIME type detection using magic number analysis
  - Write file extension mapping for supported formats
  - Add binary file detection to skip unsupported formats
  - _Requirements: 1.2, 3.1, 3.2_

- [ ] 5. Build metadata scanning and hashing engine
  - Implement parallel directory traversal using walkdir and rayon
  - Create SHA-256 hash calculation for file content with buffered I/O
  - Extract complete file metadata (size, dates, permissions, path components)
  - Add progress reporting for metadata scanning phase
  - _Requirements: 1.1, 1.3, 1.6, 5.4_

- [ ] 6. Implement file-level deduplication logic
  - Create duplicate detection by comparing SHA-256 hashes
  - Implement duplicate group assignment with canonical file marking
  - Write duplicate status tracking (unique, canonical, duplicate)
  - Add duplicate statistics reporting (unique files found, duplicates identified)
  - _Requirements: 1.4, 1.5, 1.6_

- [ ] 7. Create delta processing for incremental updates
  - Implement file change detection by comparing modification dates and sizes
  - Create logic to identify new, modified, and deleted files
  - Add soft delete marking for removed files
  - Write incremental processing queue management
  - _Requirements: 2.1, 2.2_

- [ ] 8. Build native content extraction for Tier 1 formats
  - Implement text file reader with encoding detection (UTF-8, UTF-16, Latin-1)
  - Create HTML content extractor with tag removal and optional Markdown conversion
  - Add structured format parsers (JSON, YAML, TOML) for clean text extraction
  - Write source code comment and string extraction
  - _Requirements: 2.1, 3.1, 3.2_

- [ ] 9. Implement external tool orchestration for Tier 2 formats
  - Create external tool configuration system (command templates, timeouts)
  - Implement subprocess execution with timeout handling
  - Add tool availability checking at startup
  - Write graceful degradation when tools are missing
  - _Requirements: 2.1, 3.1, 3.2_

- [ ] 10. Create content processing and paragraph splitting
  - Implement content splitting by double newlines into paragraphs
  - Add paragraph hash calculation for deduplication
  - Create paragraph validation (minimum length, character filtering)
  - Write token count estimation for paragraphs
  - _Requirements: 4.1, 4.2, 4.4_

- [ ] 11. Build paragraph-level deduplication system
  - Implement paragraph hash comparison for duplicate detection
  - Create many-to-many relationship tracking (paragraph_sources table)
  - Add source file reference with byte offset tracking
  - Write unique paragraph storage with provenance information
  - _Requirements: 4.2, 4.3, 4.5_

- [ ] 12. Implement comprehensive error handling and logging
  - Create structured error hierarchy with thiserror for all failure modes
  - Add error recovery logic for non-fatal failures (skip bad files, continue processing)
  - Implement error logging to both console and database
  - Write progress reporting with error counts and processing statistics
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 5.4_

- [ ] 13. Add progress reporting and statistics
  - Implement real-time progress indicators for both scanning and processing phases
  - Create processing statistics (files processed, paragraphs stored, deduplication rates)
  - Add performance metrics (files/sec, MB processed, estimated completion time)
  - Write final summary report with token counts and efficiency metrics
  - _Requirements: 1.6, 2.2, 5.4_

- [ ] 14. Create integration tests for complete workflows
  - Write end-to-end test with sample directory structure and various file types
  - Test metadata scanning phase with duplicate detection
  - Test content processing phase with paragraph deduplication
  - Verify database consistency and proper error handling
  - _Requirements: 1.1, 2.1, 4.1, 5.1_

- [ ] 15. Add performance optimization and memory management
  - Implement batch database operations for improved throughput
  - Add memory usage monitoring and bounded processing queues
  - Create connection pooling for database operations
  - Optimize parallel processing thread counts based on system resources
  - _Requirements: 5.5, 2.2_

- [ ] 16. Implement configuration file support
  - Create TOML configuration file for external tool commands and settings
  - Add configuration validation and default value handling
  - Implement CLI argument override of configuration settings
  - Write configuration file generation command for initial setup
  - _Requirements: 5.1, 5.2_

- [ ] 17. Add comprehensive unit tests for core components
  - Test file type detection with various file samples and edge cases
  - Test hash calculation consistency and collision handling
  - Test content extraction for each supported format with sample files
  - Test deduplication logic with various duplicate scenarios
  - _Requirements: 1.2, 1.3, 4.2, 4.3_

- [ ] 18. Create command-line help and documentation
  - Implement detailed --help output with usage examples
  - Add error message improvements with actionable suggestions
  - Create README with installation and usage instructions
  - Write troubleshooting guide for common issues
  - _Requirements: 5.2, 5.3_

- [ ] 19. Implement final integration and performance validation
  - Test complete pipeline with large directory structures (>10k files)
  - Validate performance contracts (processing speed, memory usage)
  - Test incremental processing with file modifications
  - Verify database integrity after interruptions and restarts
  - _Requirements: 5.5, 2.2, 3.1_

- [ ] 20. Add production readiness features
  - Implement graceful shutdown handling with progress preservation
  - Add database backup and recovery mechanisms
  - Create logging configuration with different verbosity levels
  - Write deployment documentation and system requirements
  - _Requirements: 3.1, 5.4, 5.5_