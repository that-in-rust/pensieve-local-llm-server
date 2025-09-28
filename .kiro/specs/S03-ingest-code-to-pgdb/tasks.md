# Implementation Plan

- [x] 1. Set up Rust project structure and core dependencies
  - Create new Rust binary project with Cargo.toml
  - Add core dependencies: clap, tokio, sqlx, anyhow, thiserror, serde, chrono
  - Set up basic CLI structure with clap derive macros
  - Create module structure: cli, core, processing, database, git
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement PostgreSQL database foundation
- [x] 2.1 Create database connection and schema management
  - Implement database pool creation with sqlx::PgPool
  - Create schema migration system for ingestion_meta table
  - Add functions to create timestamped INGEST_YYYYMMDDHHMMSS tables
  - Write tests for database connection and table creation
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 2.2 Implement core data models and serialization
  - Define IngestionMeta, IngestedFile, and QueryResult structs with sqlx::FromRow
  - Add serde serialization for JSON handling
  - Create database insert/update operations for ingestion tracking
  - Write unit tests for data model serialization and database operations
  - _Requirements: 3.1, 3.2_

- [x] 3. Build file classification system
- [x] 3.1 Implement three-type file classifier
  - Create FileClassifier with hardcoded extension mappings
  - Implement classify_file method for DirectText, Convertible, Binary types
  - Add FileMetadata and ProcessedFile data structures
  - Write comprehensive tests for all file type classifications
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3.2 Create content extraction pipeline
  - Implement ContentExtractor with extract_content method
  - Add direct text reading with encoding detection
  - Create conversion handlers for PDF (pdftotext), DOCX (pandoc), XLSX
  - Implement line count, word count, and basic token counting
  - Write tests for each extraction type and error handling
  - _Requirements: 2.2, 2.3, 2.4_

- [x] 4. Implement Git repository cloning
- [x] 4.1 Create Git clone manager with authentication
  - Implement GitCloneManager using git2-rs crate
  - Add GitHub token authentication support
  - Create clone_repo method with error handling for network issues
  - Add cleanup functionality for temporary clone directories
  - Write tests for public and private repository cloning
  - _Requirements: 1.1, 1.3_

- [x] 4.2 Add file discovery and traversal
  - Implement recursive directory walking with ignore pattern support
  - Add .gitignore parsing and pattern matching
  - Create file stream for parallel processing
  - Handle symlinks safely without infinite loops
  - Write tests for directory traversal and pattern filtering
  - _Requirements: 2.1, 2.5, 2.6_

- [x] 5. Build core ingestion engine
- [x] 5.1 Implement parallel file processing pipeline
  - Create IngestionEngine with async file processing
  - Add progress reporting with mpsc channels
  - Implement parallel processing using tokio tasks
  - Add memory management and resource cleanup
  - Write integration tests for full ingestion workflow
  - _Requirements: 1.1, 1.2, 6.1, 6.2_

- [x] 5.2 Add ingestion tracking and metadata management
  - Implement start_ingestion_record and complete_ingestion_record methods
  - Add timestamp tracking and file count statistics
  - Create error recovery and partial ingestion handling
  - Add ingestion result reporting with table names and metrics
  - Write tests for ingestion metadata lifecycle
  - _Requirements: 3.5, 3.6, 1.5_

- [x] 6. Implement CLI commands for ingestion
- [x] 6.1 Create ingest command with progress feedback
  - Implement CLI ingest command with repo URL and database path arguments
  - Add progress indicators and status updates during ingestion
  - Create clear success/failure messages with next steps
  - Add GitHub token handling from environment or CLI argument
  - Write end-to-end tests for ingest command
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [x] 6.2 Add PostgreSQL setup guidance
  - Implement pg-start command with setup instructions
  - Add database connection testing and troubleshooting
  - Create clear error messages for common PostgreSQL issues
  - Add automatic database creation if it doesn't exist
  - Write tests for setup guidance and connection validation
  - _Requirements: 1.4_

- [x] 7. Build SQL query interface
- [x] 7.1 Implement direct SQL execution
  - Create sql command that executes raw SQL queries
  - Add result formatting and pagination for large result sets
  - Implement query timeout and error handling
  - Add helpful error messages for common SQL mistakes
  - Write tests for various SQL query types and edge cases
  - _Requirements: 4.1, 4.2, 4.5_

- [x] 7.2 Add table management commands
  - Implement list-tables command showing all ingestion tables
  - Create sample command for exploring table structure
  - Add table metadata display (row counts, creation dates)
  - Implement table cleanup and management utilities
  - Write tests for table listing and sampling functionality
  - _Requirements: 4.3, 4.4_

- [x] 8. Create IDE integration workflow
- [x] 8.1 Implement query-prepare command
  - Create query-prepare command that executes SQL and writes to temp file
  - Add structured task generation for IDE analysis workflows
  - Implement output table creation for storing analysis results
  - Add clear documentation and examples for IDE integration
  - Write tests for query preparation and task file generation
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 8.2 Add result storage functionality
  - Implement store-result command for persisting analysis findings
  - Add traceability between original queries and analysis results
  - Create result validation and error handling
  - Add cleanup options for temporary files and failed workflows
  - Write tests for result storage and traceability
  - _Requirements: 5.2, 5.4, 5.5_

- [x] 9. Add performance optimizations and monitoring
- [x] 9.1 Implement streaming and memory management
  - Add streaming file processing to maintain constant memory usage
  - Implement parallel processing using all available CPU cores
  - Add progress indicators and estimated completion times
  - Create graceful degradation for resource-constrained environments
  - Write performance tests validating memory and CPU usage
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 9.2 Add resume capability and error recovery
  - Implement ingestion resume from last successful file
  - Add partial ingestion cleanup and recovery
  - Create comprehensive error reporting and logging
  - Add retry mechanisms for transient failures
  - Write tests for interruption handling and resume functionality
  - _Requirements: 6.5_

- [x] 10. Create comprehensive error handling
- [x] 10.1 Implement structured error hierarchy
  - Create CodeIngestError enum with all error types using thiserror
  - Add context-rich error messages with actionable suggestions
  - Implement error recovery strategies for common failure modes
  - Add error logging and debugging information
  - Write tests for all error conditions and recovery paths
  - _Requirements: 1.5, 4.5, 5.5_

- [x] 10.2 Add user-friendly error messages and help
  - Create helpful error messages for common user mistakes
  - Add troubleshooting guides for setup and configuration issues
  - Implement command help and usage examples
  - Add validation for user inputs and configuration
  - Write tests for error message clarity and helpfulness
  - _Requirements: 1.4, 1.5_

- [x] 11. Write comprehensive tests and documentation
- [x] 11.1 Create unit and integration test suite
  - Write unit tests for all core components with >90% coverage
  - Create integration tests for end-to-end workflows
  - Add performance tests validating speed and memory contracts
  - Implement property-based tests for file classification accuracy
  - Set up continuous integration with automated testing
  - _Requirements: All requirements validation_

- [x] 11.2 Add documentation and examples
  - Create comprehensive README with installation and usage instructions
  - Add CLI help documentation and command examples
  - Create troubleshooting guide for common issues
  - Add performance benchmarks and system requirements
  - Write developer documentation for extending the system
  - _Requirements: User experience and adoption_

- [-] 12. Package and distribute
- [x] 12.1 Create release artifacts
  - Set up cargo build for optimized release binaries
  - Create installation scripts for different platforms
  - Add version management and release automation
  - Create distribution packages (homebrew, apt, etc.)
  - Write installation and upgrade documentation
  - _Requirements: User adoption and deployment_