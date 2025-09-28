# Implementation Plan

- [x] 1. Set up database query engine and row counting
  - Create DatabaseQueryEngine struct with connection pool
  - Implement count_rows method with SQL execution
  - Add table validation and error handling
  - Write unit tests for SQL operations
  - _Requirements: Requirement 1_

- [ ] 2. Implement content extraction with A/B/C file generation
- [x] 2.1 Create ContentExtractor core structure
  - Define ContentExtractor struct with database pool and output directory
  - Implement extract_all_rows method to query database rows
  - Create ContentTriple struct for A/B/C file references
  - Write unit tests for basic extraction logic
  - _Requirements: Requirement 1_

- [x] 2.2 Implement L1 context generation
  - Create generate_l1_context method for immediate file context
  - Extract directory structure and related files
  - Handle import/include relationship analysis
  - Write tests for L1 context accuracy
  - _Requirements: Requirement 1_

- [x] 2.3 Implement L2 context generation
  - Create generate_l2_context method for architectural context
  - Extract package/crate structure information
  - Analyze cross-module relationships and patterns
  - Write tests for L2 context completeness
  - _Requirements: Requirement 1_

- [ ] 2.4 Implement file creation and management
  - Create create_content_files method with proper file I/O
  - Ensure .raw_data_202509 directory structure creation
  - Handle file naming: TableName_RowNum_Content[_L1|_L2].txt
  - Add error handling for file system operations
  - _Requirements: Requirement 1_

- [x] 3. Extend task division to 4-level hierarchy
- [x] 3.1 Create HierarchicalTaskDivider structure
  - Define HierarchicalTaskDivider with levels and groups_per_level
  - Create TaskHierarchy and TaskLevel data structures
  - Implement basic hierarchy creation framework
  - Write unit tests for structure creation
  - _Requirements: Requirement 2_

- [x] 3.2 Implement recursive task distribution algorithm
  - Create distribute_across_levels method for mathematical division
  - Handle remainder distribution across first groups at each level
  - Ensure 7 groups per level constraint enforcement
  - Write tests for various row count scenarios (35, 100, 1000 rows)
  - _Requirements: Requirement 2_

- [x] 3.3 Create hierarchical task numbering system
  - Implement task ID generation (1.2.3.4 format)
  - Create HierarchicalTaskGroup with proper nesting
  - Handle sub-group creation and parent-child relationships
  - Write tests for ID uniqueness and hierarchy integrity
  - _Requirements: Requirement 2_

- [x] 4. Implement L1-L8 analysis task generation
- [x] 4.1 Create AnalysisTask structure
  - Define AnalysisTask with content files, prompt, and output references
  - Implement AnalysisStage enum for 4-stage analysis process
  - Create task metadata with table name and row number
  - Write unit tests for task structure validation
  - _Requirements: Requirement 3_

- [x] 4.2 Implement L1L8MarkdownGenerator
  - Create L1L8MarkdownGenerator with prompt file and output directory
  - Implement generate_hierarchical_markdown for full task structure
  - Create create_analysis_task method for individual task formatting
  - Add format_l1l8_analysis_instructions for methodology integration
  - _Requirements: Requirement 3_

- [x] 4.3 Generate proper task format with A/B/C references
  - Format tasks with Content, Prompt, and Output sections
  - Include proper file path references to A/B/C content files
  - Add L1-L8 analysis stage instructions
  - Ensure gringotts/WorkArea output path generation
  - _Requirements: Requirement 3_

- [x] 5. Create CLI commands and integration
- [x] 5.1 Implement count-rows command
  - Add count-rows subcommand to CLI parser
  - Integrate with DatabaseQueryEngine for row counting
  - Add proper error handling and user feedback
  - Write integration tests for command execution
  - _Requirements: Requirement 1_

- [x] 5.2 Implement extract-content command
  - Add extract-content subcommand with table name and output directory
  - Integrate with ContentExtractor for A/B/C file generation
  - Add progress reporting for large table processing
  - Write integration tests for content extraction workflow
  - _Requirements: Requirement 1_

- [x] 5.3 Implement generate-hierarchical-tasks command
  - Add generate-hierarchical-tasks subcommand with full parameter set
  - Integrate all components: database → content → hierarchy → markdown
  - Add validation for table existence and content file availability
  - Write end-to-end integration tests for complete workflow
  - _Requirements: Requirement 2, Requirement 3_

- [x] 6. Add error handling and performance optimization
- [x] 6.1 Implement comprehensive error types
  - Create TaskError enum with specific error variants
  - Add context-rich error messages with actionable guidance
  - Implement proper error propagation through async operations
  - Write tests for error handling scenarios
  - _Requirements: All requirements_

- [x] 6.2 Optimize for large table processing
  - Add streaming/batching for large row counts (10,000+ rows)
  - Implement memory-efficient content processing
  - Add progress reporting and cancellation support
  - Write performance tests for large dataset scenarios
  - _Requirements: All requirements_

- [x] 7. Integration testing and validation
- [x] 7.1 Create end-to-end workflow tests
  - Test complete pipeline: count → extract → generate tasks
  - Validate output file structure and content accuracy
  - Test with real INGEST_* table data
  - Verify L1-L8 analysis methodology integration
  - _Requirements: All requirements_

- [x] 7.2 Validate generated task execution
  - Test generated markdown files in Kiro task system
  - Verify task numbering and hierarchy navigation
  - Validate content file references and accessibility
  - Test analysis workflow with actual L1-L8 execution
  - _Requirements: All requirements_