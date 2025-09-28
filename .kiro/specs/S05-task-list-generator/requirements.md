# Requirements Document

## Introduction

Generate structured task lists from ingested codebase data to enable systematic analysis. Transform database query results into Kiro-compatible markdown files with hierarchical task numbering. Support both basic file-level analysis and advanced chunked analysis for large files, with flexible ingestion from git repositories or local folders.

## Example Scenario

**Input:** Query `INGEST_20250928101039` table returns 35 files from xsv repository  
**Output:** Markdown file with 7 task groups, each containing ~5 analysis tasks  
**Source:** Database table with columns: file_id, filepath, filename, extension, line_count  
**Prompt:** Use `.kiro/steering/spec-S04-steering-doc-analysis.md` for analysis

Format e.g. of final task

- [ ] 5. Analyze INGEST_20250928101039 row 35 
  - **Content**: `.raw_data_202509/INGEST_20250928101039_35_Content.txt` as A + `.raw_data_202509/INGEST_20250928101039_35_Content_L1.txt` as B + `.raw_data_202509/INGEST_20250928101039_35_Content_L2.txt` as C
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in cotext of C ; A in context B & C
  - **Output**: `gringotts/WorkArea/INGEST_20250928101039_35.md`


# Strict Scope

- the Table_NAME is known
- a terminal command with code-ingest tool to find SQL count of rows in TableName
- a terminal command to then loop through all of the rows to create 3 files each in the folder .raw_data_202509
  - 1. TableName_RowNum_Content.md as A
  - 2. TableName_RowNum_Content_L1.md as B
  - 3. TableName_RowNum_Content_L2.md as C
- a terminal command to generate the following list of tasks for each row -  knowing that the whole task list needs to just has 7 first level tasks and 7 second level tasks and so on till 4 levels

- [ ] 5. Analyze <TableName> row <row_iterator> 
  - **Content**: `.raw_data_202509/<TableName>_<row_iterator>_Content.txt` as A + `.raw_data_202509/<TableName>_<row_iterator>_Content_L1.txt` as B + `.raw_data_<TableName>_<row_iterator>_Content_L2.txt` as C
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in cotext of C ; A in context B & C
  - **Output**: `gringotts/WorkArea/<TableName>_<row_iterator>.md`


## So final task generation command looks like this
``` bash
code-ingest generate-hierarchical-tasks INGEST_20250928101039 --levels 4 --groups 7 --output INGEST_20250928101039_tasks.md
```


# Advanced Scope - Advanced Task Generator


## Advanced commands

### 1. Feature Request 1:

current version of generate-hierarchical-tasks command:
code-ingest generate-hierarchical-tasks <TableName> --levels <UserOptionLevels> --groups <UserOptionGroupCount> --output TableName_tasks.md --prompt-file <UserOptionPromptFilePathFileName>

new requirement:
code-ingest generate-hierarchical-tasks <TableName> --chunks <UserOptionChunkSizeInLOC> --levels <UserOptionLevels> --groups <UserOptionGroupCount> --output TableName_tasks.md --prompt-file <UserOptionPromptFilePathFileName>

When User sends the above command, following things happen
- a new table is created named <TableName_UserOptionChunkSizeInLOC>
  - Check if LOC < <UserOptionChunkSizeInLOC> then row processing same as <TableName>
  - If LOC >= <UserOptionChunkSizeInLOC> then
    - the row is broken into chunks of <UserOptionChunkSizeInLOC> lines
    - variables will be like following
      - filepath remains same
      - parent_filepath remains same
      - filename remains same
      - new column called Chunk_Number based on <UserOptionChunkSizeInLOC>
      - content is according to what is there in the given chunk
      - content L1 is concatenation of previous row content and next row content
      - content L2 is concatenation of previous 2 rows of content and next 2 rows of content


- [ ] 5. Analyze <TableName> row <row_iterator> 
  - **Content**: `.raw_data_202509/<TableName_UserOptionChunkSizeInLOC>_<row_iterator>_Content.txt` as A + `.raw_data_202509/<TableName>_<row_iterator>_Content_L1.txt` as B + `.raw_data_<TableName>_<row_iterator>_Content_L2.txt` as C
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in cotext of C ; A in context B & C
  - **Output**: `gringotts/WorkArea/<TableName_UserOptionChunkSizeInLOC>_<row_iterator>.md`

### 2. Feature Request 2:


#### Current ingestion command looks like this


``` bash
./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path /Users/neetipatni/desktop/PensieveDB01
```
#### Advanced ingestion command looks like this

``` bash
./target/release/code-ingest ingest /Users/neetipatni/Desktop/refGitHub --folder-flag Y --db-path /Users/neetipatni/desktop/PensieveDB01

```
so essentially the format is
``` bash
./target/release/code-ingest ingest <AbsoluteFolderPathUserOption> --folder-flag Y --db-path <AbsoluteDatabaseFolderPath>

```

## Requirements

### Requirement 1: Basic Task Generation

**User Story:** As a codebase analyst, I want to generate hierarchical task lists from database tables, so that I can systematically analyze code files with structured workflows.

#### Acceptance Criteria

1. WHEN I run `code-ingest generate-hierarchical-tasks <TableName> --levels <N> --groups <M> --output <filename>` THEN the system SHALL generate a markdown file with N hierarchical levels and M groups per level
2. WHEN processing database rows THEN the system SHALL create three content files per row: Content.txt, Content_L1.txt, and Content_L2.txt in `.raw_data_202509/` directory
3. WHEN generating tasks THEN each task SHALL reference the analysis prompt from `.kiro/steering/spec-S04-steering-doc-analysis.md`
4. WHEN task execution completes THEN output SHALL be written to `gringotts/WorkArea/<TableName>_<row_iterator>.md`

### Requirement 2: Chunked Analysis for Large Files

**User Story:** As a codebase analyst, I want to break large files into manageable chunks, so that I can analyze complex codebases without overwhelming context windows.

#### Acceptance Criteria

1. WHEN I run `code-ingest generate-hierarchical-tasks <TableName> --chunks <LOC> --levels <N> --groups <M> --output <filename> --prompt-file <path>` THEN the system SHALL create a new table `<TableName_LOC>` with chunked data
2. WHEN a file has LOC >= chunk size THEN the system SHALL split the file into chunks of specified LOC size with sequential chunk numbers
3. WHEN creating chunked content THEN the system SHALL preserve filepath, parent_filepath, and filename while adding Chunk_Number column
4. WHEN generating L1 context THEN the system SHALL concatenate previous and next chunk content
5. WHEN generating L2 context THEN the system SHALL concatenate content from 2 chunks before and 2 chunks after current chunk
6. WHEN a file has LOC < chunk size THEN the system SHALL process it as a single chunk identical to original table processing

### Requirement 3: Local Folder Ingestion

**User Story:** As a codebase analyst, I want to ingest code from local folders, so that I can analyze private repositories and local codebases without requiring git hosting.

#### Acceptance Criteria

1. WHEN I run `code-ingest ingest <AbsoluteFolderPath> --folder-flag Y --db-path <DatabasePath>` THEN the system SHALL recursively process all files in the specified folder
2. WHEN processing local folders THEN the system SHALL validate that the absolute path exists and is accessible
3. WHEN ingesting from folders THEN the system SHALL maintain the same database schema as git repository ingestion
4. WHEN folder ingestion completes THEN the system SHALL create the same table structure as repository-based ingestion

### Requirement 4: Enhanced Command Interface

**User Story:** As a codebase analyst, I want flexible command options, so that I can customize analysis workflows for different project types and sizes.

#### Acceptance Criteria

1. WHEN using basic generation THEN the system SHALL support `--levels`, `--groups`, and `--output` parameters
2. WHEN using chunked analysis THEN the system SHALL additionally support `--chunks` and `--prompt-file` parameters  
3. WHEN using local ingestion THEN the system SHALL support `--folder-flag Y` parameter for folder-based processing
4. WHEN invalid parameters are provided THEN the system SHALL display helpful error messages with correct usage examples