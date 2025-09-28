# Implementation Plan

- [x] 1. Set up project structure and core data models
  - Create Rust project with proper Cargo.toml dependencies (sqlx, tokio, clap, serde, thiserror, anyhow)
  - Define core data structures: IngestionSource, ChunkMetadata, TaskHierarchy, GenerationConfig
  - Implement basic error types with thiserror for structured error handling
  - _Requirements: 1.1, 2.1, 3.1, 4.1_

- [x] 2. Implement database foundation and schema management
  - [x] 2.1 Create database connection and migration system
    - Write database connection utilities with SQLx connection pooling
    - Implement schema creation for base tables with proper indexing
    - Create migration system for table versioning and updates
    - _Requirements: 1.2, 3.3_

  - [x] 2.2 Implement base table operations
    - Write CRUD operations for basic file ingestion tables
    - Implement row counting and querying functionality for task generation
    - Create database transaction handling for batch operations
    - _Requirements: 1.1, 1.2_

  - [x] 2.3 Implement chunked table creation and management
    - Write dynamic table creation for chunked tables with naming pattern `<TableName_ChunkSize>`
    - Implement chunked table schema with additional columns (chunk_number, chunk_start_line, etc.)
    - Create table validation and existence checking utilities
    - _Requirements: 2.1, 2.3_

- [x] 3. Build ingestion engine for git repositories and local folders
  - [x] 3.1 Implement git repository ingestion
    - Create GitIngestionProvider with repository cloning and file extraction
    - Implement file content reading and metadata collection from git repositories
    - Write database insertion logic for git-sourced files
    - _Requirements: 3.3, 4.3_

  - [x] 3.2 Implement local folder ingestion
    - Create FolderIngestionProvider with recursive directory traversal
    - Implement absolute path validation and accessibility checking
    - Write file system integration for local file processing with same database schema
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 3.3 Create unified ingestion interface
    - Implement IngestionEngine that coordinates git and folder providers
    - Write source validation logic for both git URLs and local paths
    - Create error handling and recovery strategies for ingestion failures
    - _Requirements: 3.1, 3.4, 4.4_

- [-] 4. Develop chunking engine for large file processing
  - [-] 4.1 Implement core chunking algorithm
    - Write LOC-based file splitting logic with configurable chunk sizes
    - Implement chunk boundary detection that preserves code structure
    - Create ChunkData structures with proper metadata (start_line, end_line, chunk_number)
    - _Requirements: 2.2, 2.3_

  - [ ] 4.2 Build context generation system
    - Implement L1 context generation (previous + current + next chunk concatenation)
    - Write L2 context generation (±2 chunks concatenation logic)
    - Create context boundary handling for first/last chunks in files
    - _Requirements: 2.4, 2.5_

  - [ ] 4.3 Integrate chunking with database operations
    - Write chunked table population with proper foreign key relationships
    - Implement batch insertion for chunked data to optimize performance
    - Create size threshold checking (LOC < chunk_size → single chunk processing)
    - _Requirements: 2.1, 2.6_

- [ ] 5. Create task generation and hierarchical structure builder
  - [ ] 5.1 Implement task hierarchy calculation
    - Write algorithm to distribute database rows across hierarchical levels
    - Implement task numbering system (1.1, 1.2, 2.1, etc.) with proper formatting
    - Create task distribution logic that balances groups across levels
    - _Requirements: 1.1, 4.1_

  - [ ] 5.2 Build task structure generation
    - Implement Task and TaskLevel data structures with proper relationships
    - Write task description generation with content file references
    - Create prompt file integration and reference generation
    - _Requirements: 1.3, 4.2_

  - [ ] 5.3 Create markdown task list writer
    - Implement markdown formatting with proper checkbox syntax and indentation
    - Write hierarchical task rendering with content, prompt, and output references
    - Create file output handling with proper path management
    - _Requirements: 1.1, 1.4_

- [ ] 6. Implement content file generation system
  - [ ] 6.1 Create basic content file generator
    - Write ContentGenerator for three-tier file creation (A, L1, L2)
    - Implement file naming conventions with proper path handling
    - Create content extraction from database rows with proper formatting
    - _Requirements: 1.2, 1.4_

  - [ ] 6.2 Implement chunked content file generation
    - Write chunked content file creation with modified naming patterns
    - Implement context file generation using chunking engine output
    - Create content file validation and error handling
    - _Requirements: 2.1, 2.4, 2.5_

  - [ ] 6.3 Build output directory management
    - Implement `.raw_data_202509/` directory creation and management
    - Write file cleanup and organization utilities
    - Create output path validation and conflict resolution
    - _Requirements: 1.2, 2.1_

- [ ] 7. Develop command-line interface and configuration
  - [ ] 7.1 Implement basic command structure
    - Create CLI using clap with subcommands for ingest and generate-hierarchical-tasks
    - Implement parameter validation for levels, groups, output files
    - Write help text and usage examples for all commands
    - _Requirements: 4.1, 4.4_

  - [ ] 7.2 Add advanced command options
    - Implement --chunks parameter with validation and chunking engine integration
    - Write --prompt-file parameter handling with file existence validation
    - Create --folder-flag parameter for local folder ingestion mode
    - _Requirements: 2.1, 3.1, 4.2_

  - [ ] 7.3 Build configuration management
    - Implement configuration file support for default parameters
    - Write parameter precedence handling (CLI > config file > defaults)
    - Create configuration validation and error reporting
    - _Requirements: 4.1, 4.2, 4.4_

- [ ] 8. Add comprehensive error handling and logging
  - [ ] 8.1 Implement structured error handling
    - Create comprehensive error hierarchy with proper error context
    - Write error recovery strategies for database, filesystem, and network errors
    - Implement user-friendly error messages with actionable suggestions
    - _Requirements: 4.4_

  - [ ] 8.2 Add logging and monitoring
    - Implement structured logging with different verbosity levels
    - Write progress reporting for long-running operations (ingestion, chunking)
    - Create performance metrics collection for optimization
    - _Requirements: 1.1, 2.1, 3.1_

- [ ] 9. Create comprehensive test suite
  - [ ] 9.1 Write unit tests for core components
    - Create tests for chunking algorithm with various file sizes and chunk configurations
    - Write tests for task hierarchy generation with different level/group combinations
    - Implement tests for database operations including table creation and data insertion
    - _Requirements: 1.1, 2.1, 2.2, 2.3_

  - [ ] 9.2 Build integration tests
    - Write end-to-end tests for complete ingestion → generation workflows
    - Create tests for both git repository and local folder ingestion paths
    - Implement performance tests for large codebase processing
    - _Requirements: 1.1, 2.1, 3.1, 3.3_

  - [ ] 9.3 Add property-based and stress tests
    - Implement property-based tests for chunking content preservation
    - Write stress tests for concurrent database operations and large file processing
    - Create edge case tests for boundary conditions and error scenarios
    - _Requirements: 2.2, 2.4, 2.5_

- [ ] 10. Optimize performance and finalize system
  - [ ] 10.1 Implement performance optimizations
    - Add database query optimization with proper indexing and connection pooling
    - Write memory-efficient streaming for large file processing
    - Implement concurrent processing for parallel file ingestion and chunking
    - _Requirements: 2.1, 3.1_

  - [ ] 10.2 Create documentation and examples
    - Write comprehensive README with installation and usage instructions
    - Create example workflows for common use cases (git repos, local folders, chunked analysis)
    - Implement inline code documentation with proper rustdoc formatting
    - _Requirements: 4.4_

  - [ ] 10.3 Build final integration and validation
    - Create end-to-end validation with real-world codebases
    - Write integration with existing `.kiro/steering/spec-S04-steering-doc-analysis.md` workflow
    - Implement final testing with `gringotts/WorkArea/` output validation
    - _Requirements: 1.3, 1.4, 2.1_