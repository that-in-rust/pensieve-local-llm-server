# Implementation Plan


## IMPORTANT FOR VISUALS AND DIAGRAMS

ALL DIAGRAMS WILL BE IN MERMAID ONLY TO ENSURE EASE WITH GITHUB - DO NOT SKIP THAT

- [x] 1. Set up project structure and core interfaces
  - Create Cargo.toml with required dependencies (clap, sqlx, tokio, sha2, walkdir, mime_guess, thiserror, anyhow)
  - Define core data structures (FileMetadata, ProcessingStatus, DuplicateStatus, error types)
  - Create module structure (cli, scanner, extractor, database, errors)
  - _Requirements: 5.1, 5.3, 5.5_

- [x] 2. Implement CLI interface and argument parsing
  - Create CLI struct with clap derive for input directory and database path arguments
  - Add --help flag with basic usage instructions
  - Implement argument validation (directory exists, database path writable)
  - Add basic error messages for missing arguments
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 3. Create database schema and connection management
  - Implement SQLite database initialization with WAL mode
  - Create all tables (files, paragraphs, errors) with proper indexes
  - Write database connection management
  - Add basic database setup and table creation
  - _Requirements: 1.7, 2.3_

- [x] 4. Implement file type detection and filtering system
  - Create file type classification for supported text formats
  - Implement basic MIME type detection using file extensions
  - Write file extension mapping for all supported formats from Requirements 1.2
  - Add binary file detection to skip unsupported formats
  - _Requirements: 1.2, 3.1, 3.2_

- [x] 5. Build metadata scanning and hashing engine
  - Implement parallel directory traversal using walkdir and rayon
  - Create SHA-256 hash calculation for file content with buffered I/O
  - Extract complete file metadata (size, dates, permissions, path components)
  - Add progress reporting for metadata scanning phase
  - _Requirements: 1.1, 1.3, 1.6, 5.4_

- [x] 6. Implement file-level deduplication logic
  - Create duplicate detection by comparing SHA-256 hashes
  - Implement duplicate group assignment with canonical file marking
  - Write duplicate status tracking (unique, canonical, duplicate)
  - Add duplicate statistics reporting (unique files found, duplicates identified)
  - _Requirements: 1.4, 1.5, 1.6_



- [x] 7. Build native content extraction for all supported formats
  - Implement text file reader with encoding detection (UTF-8, Latin-1)
  - Create HTML content extractor with basic tag removal
  - Add basic PDF text extraction using native Rust crates
  - Implement basic DOCX text extraction using ZIP and XML parsing
  - Add structured format parsers (JSON, YAML, TOML) for clean text extraction
  - Write source code reader (treat as plain text)
  - _Requirements: 2.1, 3.1, 3.2, 5.5_

- [x] 8. Implement missing database methods for paragraph processing
  - Complete insert_paragraph method with proper SQL insertion
  - Complete insert_paragraph_source method for file-paragraph relationships
  - Complete get_paragraph_by_hash method for deduplication checks
  - Complete insert_error method for processing error tracking
  - Add batch operations for efficient paragraph storage
  - _Requirements: 4.2, 4.3, 4.5_

- [x] 9. Integrate content processing and paragraph splitting into main workflow
  - Integrate existing ContentProcessor into CLI workflow after metadata scanning
  - Connect content extraction with paragraph splitting and storage
  - Add paragraph processing to unique files after deduplication
  - Update file records with token counts after content processing
  - _Requirements: 4.1, 4.2, 4.4_

- [x] 10. Integrate paragraph-level deduplication system
  - Connect existing database paragraph methods with content processing
  - Implement paragraph deduplication logic in main workflow
  - Add paragraph-to-file relationship tracking via paragraph_sources table
  - Ensure only unique paragraphs are stored with proper file references
  - _Requirements: 4.2, 4.3, 4.5_

- [x] 11. Implement basic error handling and logging
  - Create structured error hierarchy with thiserror for all failure modes
  - Add error recovery logic for non-fatal failures (skip bad files, continue processing)
  - Implement error logging to both console and database
  - Write basic progress reporting with file and paragraph counts
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 5.4_

- [x] 12. Add basic progress reporting and statistics
  - Implement simple progress indicators for both scanning and processing phases
  - Create processing statistics (files processed, paragraphs stored, duplicates found)
  - Write final summary report with counts and basic metrics
  - _Requirements: 1.6, 2.2, 5.4_

- [x] 13. Create integration tests for complete workflows
  - Write end-to-end test with sample directory structure and various file types
  - Test metadata scanning phase with duplicate detection
  - Test content processing phase with paragraph deduplication
  - Verify database consistency and proper error handling
  - _Requirements: 1.1, 2.1, 4.1, 5.1_

- [x] 14. Add basic performance optimization
  - Implement batch database operations for improved throughput
  - Add basic memory management for large files
  - Optimize parallel processing for directory traversal
  - _Requirements: 5.5, 2.2_

- [x] 15. Add unit tests for core components
  - Test file type detection with various file samples and edge cases
  - Test hash calculation consistency and performance
  - Test content extraction for each supported format with sample files
  - Test deduplication logic with various duplicate scenarios
  - _Requirements: 1.2, 1.3, 4.2, 4.3_

- [x] 16. Create command-line help and documentation
  - Implement basic --help output with usage instructions
  - Add clear error messages for common issues
  - Create README with installation and usage instructions
  - _Requirements: 5.2, 5.3_

- [x] 17. Complete content processing integration in CLI workflow
  - Add Phase 4 content processing to CLI after metadata storage
  - Integrate ExtractionManager to process unique files only
  - Connect paragraph splitting, deduplication, and storage
  - Update file records with token counts and processing status
  - Add content processing progress reporting and statistics
  - _Requirements: 2.1, 4.1, 4.2, 4.3, 4.5_

- [x] 18. Implement final integration and validation
  - Test complete pipeline with sample directory structures
  - Verify database consistency and proper error handling
  - Test with various file types and duplicate scenarios
  - Validate basic performance and memory usage
  - _Requirements: 5.5, 2.2, 3.1_

- [x] 19. Complete database migration system implementation
  - Implement MigrationManager struct with version tracking
  - Add migration methods for schema updates and data transformations
  - Create migration files for schema evolution
  - Test migration system with version upgrades
  - _Requirements: Database schema evolution support_

- [x] 20. Fix failing unit test in deduplication module
  - Fix canonical file selection logic to properly handle path comparison
  - Ensure deterministic canonical file selection based on path length and alphabetical order
  - Update test expectations to match corrected logic
  - _Requirements: 1.4, 1.5_

- [x] 21. Implement missing CLI subcommands
  - Complete `pensieve check-deps` command for dependency verification
  - Complete `pensieve config` command for configuration file generation
  - Add proper error handling and user feedback for all subcommands
  - Note: `pensieve init` is already implemented and working
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 22. Add comprehensive end-to-end integration tests
  - Create integration test for complete CLI workflow from directory scan to final statistics
  - Test paragraph-level deduplication across multiple files
  - Verify database consistency after full processing pipeline
  - Test error recovery and partial processing scenarios
  - _Requirements: 1.1, 2.1, 4.1, 4.2, 4.3, 5.1_

- [x] 23. Create README and documentation - take help of (.kiro/steering/mermaid-troubleshooting.md;.kiro/steering/mermaid-syntax-guide.md;kiro/steering/mermaid-status-report.md;.kiro/steering/mermaid-design-patterns.md)
  - Write comprehensive README with installation instructions
  - Document CLI usage with examples
  - Add troubleshooting guide for common issues
  - Document supported file formats and processing capabilities
  - Include Mermaid diagrams showing system architecture and workflow
  - _Requirements: 5.2, 5.3_

- [ ] 24. Repository cleanup and organization
  - Run .kiro/tree-with-wc.sh to analyze repository structure
  - Clean up any unnecessary files or directories - instead of deleting place them in zzzzArchive folder
  - Ensure repository looks professional for GitHub presentation
  - Organize examples and test data appropriately
  - Can we make the README minimalistic x Minto Pyramid Principle - starting from essence at the top and then adding details and lower layers x also all the mermaid diagrams should follow guidance of steering docs x .kiro/steering/mermaid-design-patterns.md x .kiro/steering/mermaid-status-report.md x .kiro/steering/mermaid-syntax-guide.md x .kiro/steering/mermaid-troubleshooting.md