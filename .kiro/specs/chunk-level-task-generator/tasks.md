# Implementation Plan

- [x] 1. Create core error types and data models
  - Create `TaskGeneratorError` enum with structured error handling using `thiserror`
  - Create `ChunkedFile`, `TaskGenerationResult`, and `ContentFiles` data models
  - Write unit tests for error type conversions and data model serialization
  - _Requirements: 3.1, 3.2_

- [x] 2. Implement DatabaseService for table operations
  - ✅ Create `DatabaseService` struct with connection pool management
  - ✅ Implement `validate_table()` method to check table existence and schema
  - ✅ Implement `query_rows()` method to fetch `IngestedFile` records from tables
  - ✅ Implement `create_chunked_table()` method for chunk-level mode
  - ✅ Write unit tests for database operations with mock database
  - _Requirements: 1.1, 2.1, 2.2_
  - **Implementation**: `code-ingest/src/tasks/database_service.rs` with Arc<PgPool> connection management, comprehensive table validation, IngestedFile querying, and chunked table creation. Includes 15+ unit tests and integration test support.

- [x] 3. Implement ChunkingService for file processing logic
  - Create `ChunkingService` struct with chunking algorithms
  - Implement `apply_chunking_rules()` method: copy small files, chunk large files
  - Implement L1 (row+next) and L2 (row+next+next2) concatenation logic
  - Implement `process_with_chunking()` method to populate chunked table
  - Write unit tests for chunking logic with various file sizes
  - _Requirements: 2.2, 2.3, 2.4, 2.5_

- [x] 4. Implement ContentFileWriter for file generation
  - ✅ Create `ContentFileWriter` struct with async file I/O operations
  - ✅ Implement `write_content_files()` method to create content, contentL1, contentL2 files
  - ✅ Implement `write_row_files()` method for individual row processing
  - ✅ Use `tokio::fs` for async file operations with proper error handling
  - ✅ Write unit tests for file creation and content validation
  - _Requirements: 1.1, 2.6_
  - **Implementation**: `code-ingest/src/tasks/content_file_writer.rs` with async file I/O using tokio::fs, configurable naming patterns, concurrent processing, comprehensive error handling, and extensive unit tests. Supports both individual row processing and batch operations with L1/L2 context generation.

- [x] 5. Implement TaskListGenerator for task file creation
  - Create `TaskListGenerator` struct for generating task lists in txt format
  - Implement `generate_task_list()` method that references content files by row number
  - Create task list format that's compatible with existing task processing workflows
  - Write unit tests for task list generation and format validation
  - _Requirements: 1.2, 2.7_

- [x] 6. Implement main ChunkLevelTaskGenerator coordinator
  - Create `ChunkLevelTaskGenerator` struct that orchestrates all services
  - Implement `execute()` method with file-level and chunk-level mode logic
  - Add input validation for table names and chunk sizes
  - Implement proper resource cleanup and error propagation
  - Write unit tests for both execution modes
  - _Requirements: 1.1, 1.2, 2.1, 2.6, 2.7_

- [x] 7. Add CLI command integration
  - Add `ChunkLevelTaskGenerator` command to CLI enum in `cli/mod.rs`
  - Implement `execute_chunk_level_task_generator()` method in CLI handler
  - Add command-line argument parsing for table name, chunk size, and output directory
  - Add help text and usage examples for the new command
  - _Requirements: 1.1, 2.1_

- [x] 8. Write integration tests for end-to-end workflows
  - Create integration test for file-level mode (no chunk size)
  - Create integration test for chunk-level mode (with chunk size)
  - Test error handling for invalid table names and chunk sizes
  - Test content file creation and task list generation
  - Verify chunked table creation and population
  - _Requirements: 1.1, 1.2, 2.1, 2.6, 2.7, 3.1, 3.2_

- [x] 9. Remove ExtractContent command and related code
  - Remove `ExtractContent` variant from CLI `Commands` enum
  - Remove `execute_extract_content()` method from CLI implementation
  - Remove any associated test files for ExtractContent functionality
  - Update CLI help text and documentation
  - _Requirements: 4.1, 4.2_

- [-] 10. Mark existing task generators as deprecated
  - Add deprecation warnings to `GenerateHierarchicalTasks` command
  - Update help text to recommend using `chunk-level-task-generator` instead
  - Add deprecation comments to related task generation modules
  - Document migration path from old to new command
  - _Requirements: 5.1_

- [ ] 11. Test new command with existing database tables
  - Test `chunk-level-task-generator` with real ingestion tables
  - Validate content file generation matches expected format
  - Verify task list format is compatible with existing workflows
  - Test performance with large tables (>1000 rows)
  - Document any issues or edge cases discovered
  - _Requirements: 1.1, 1.2, 2.1, 2.6, 2.7_

- [ ] 12. Clean up deprecated task generation code (after validation)
  - Remove `GenerateHierarchicalTasks` command from CLI enum
  - Remove complex task generation modules from `tasks/` directory
  - Remove associated test files for deprecated functionality
  - Preserve any shared utilities that might be needed by other commands
  - Update documentation and help text
  - _Requirements: 5.2, 5.3_