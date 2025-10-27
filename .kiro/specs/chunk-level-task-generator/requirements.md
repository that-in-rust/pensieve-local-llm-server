# Requirements Document

## Introduction

Create a simple `chunk-level-task-generator` command that replaces the complex existing task generation system with two modes: file-level (no chunk size) and chunk-level (with chunk size).

## Requirements

### Requirement 1: File-Level Mode

**User Story:** As a developer, I want to run `chunk-level-task-generator <table_name>` to generate content files and task list for each database row.

#### Acceptance Criteria

1. WHEN I run `chunk-level-task-generator <table_name>` THEN the system SHALL create `content_<row>.txt`, `contentL1_<row>.txt`, `contentL2_<row>.txt` files
2. WHEN generating task list THEN the system SHALL create a txt file referencing content files by row number

### Requirement 2: Chunk-Level Mode

**User Story:** As a developer, I want to run `chunk-level-task-generator <table_name> <chunk_size>` to process large files with chunking.

#### Acceptance Criteria

1. WHEN I run with chunk size THEN the system SHALL create `<TableName>_<ChunkSize>` table
2. WHEN file LOC < chunk size THEN the system SHALL copy row unchanged
3. WHEN file LOC >= chunk size THEN the system SHALL break into multiple rows with L1 (row+next) and L2 (row+next+next2) concatenation
4. WHEN chunked table created THEN the system SHALL generate content files and task list as in file-level mode

### Requirement 3: Error Handling

**User Story:** As a developer, I want clear error messages for invalid inputs.

#### Acceptance Criteria

1. WHEN table missing or chunk size invalid THEN the system SHALL return clear error messages
2. WHEN database/file operations fail THEN the system SHALL provide actionable error messages

### Requirement 4: Clean Up ExtractContent Command

**User Story:** As a developer, I want to remove the ExtractContent command that does nothing useful.

#### Acceptance Criteria

1. WHEN cleaning up THEN the system SHALL remove ExtractContent command from CLI
2. WHEN removing ExtractContent THEN the system SHALL remove associated code and tests

### Requirement 5: Clean Up Existing Task Generators

**User Story:** As a developer, I want to safely remove complex task generator code after validating the new command.

#### Acceptance Criteria

1. WHEN new command is tested THEN the system SHALL mark existing task generators as deprecated
2. WHEN cleanup phase begins THEN the system SHALL remove GenerateHierarchicalTasks and related complex task generation code
3. WHEN removing old code THEN the system SHALL preserve any shared utilities that might be needed