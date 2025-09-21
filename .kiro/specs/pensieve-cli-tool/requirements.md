# Requirements Document

## Introduction

The Pensieve is a simple command-line tool designed to quickly ingest text files into a clean, deduplicated database for LLM processing. The tool processes text-based files including but not limited to: .txt, .md, .rs, .py, .js, .ts, .html, .css, .json, .xml, .yaml, .yml, .toml, .ini, .cfg, .log, .csv, .tsv, .xls, .xlsx, .sql, .sh, .bat, .ps1, .dockerfile, .gitignore, .env, .properties, .conf, .c, .cpp, .h, .hpp, .java, .go, .php, .rb, .swift, .kt, .scala, .clj, .hs, .elm, .lua, .pl, .r, .m, .tex, .bib, .org, .rst, .adoc, .wiki, .pdf, .doc, .docx, .odt, .rtf, .pages, .epub, .mobi, .azw, .azw3, .fb2, .lit, .pdb, .tcr, .prc, and other readable text formats. The tool excludes binary files such as images (.jpg, .png, .gif, .bmp, .svg), videos (.mp4, .avi, .mov), audio files (.mp3, .wav, .flac), archives (.zip, .tar, .gz, .rar), executables (.exe, .bin, .app), and compiled libraries (.dll, .so, .dylib). It removes duplicate content to create an efficient corpus that maximizes token usage when querying LLMs for insights, ideas, or analysis.
This MVP focuses on getting content into a queryable format quickly, without complex features or optimization. The tool is exclusively written in Rust for performance and reliability.
## Requirements

### Requirement 1

**User Story:** As a developer, I want to create a metadata table of contents first, so that I can identify duplicate files by content hash before processing and efficiently remove duplicates at the file level.
#### Acceptance Criteria

1. WHEN I run the CLI tool with a directory of files THEN the system SHALL first scan all files and create a metadata table of contents
2. WHEN scanning files THEN the system SHALL process .txt, .md, .rs, .py, .js, .ts, .html, .css, .json, .xml, .yaml, .yml, .toml, .ini, .cfg, .log, .csv, .tsv, .xls, .xlsx, .sql, .sh, .bat, .ps1, .dockerfile, .gitignore, .env, .properties, .conf, .c, .cpp, .h, .hpp, .java, .go, .php, .rb, .swift, .kt, .scala, .clj, .hs, .elm, .lua, .pl, .r, .m, .tex, .bib, .org, .rst, .adoc, .wiki, .pdf, .doc, .docx, .odt, .rtf, .pages, .epub, .mobi, .azw, .azw3, .fb2, .lit, .pdb, .tcr, .prc, and other readable text files while excluding binary formats like images, videos, audio, archives, and executables
3. WHEN creating the metadata TOC THEN the system SHALL calculate file hash, size, creation date, and modification date for each file
4. WHEN the metadata scan finds duplicate file hashes THEN it SHALL identify which files have identical content
5. WHEN duplicate files are identified THEN the system SHALL mark them for exclusion from content processing
6. WHEN the metadata TOC is complete THEN the system SHALL show how many unique files were found and how many duplicates were identified
7. WHEN I specify an output database THEN the system SHALL store the metadata TOC in a files table with full_filepath, folder_path, filename, file_extension, file_type (file/folder), size, hash, creation_date, modification_date, access_date, permissions, depth_level, relative_path, is_hidden, is_symlink, symlink_target, duplicate_status, and duplicate_group_id

1. WHEN I run the CLI tool with a directory of files THEN the system SHALL first scan all files and create a metadata table of contents
2. WHEN scanning files THEN the system SHALL process .txt, .md, .rs, .py, .js, .ts, .html, .css, .json, .xml, .yaml, .yml, .toml, .ini, .cfg, .log, .csv, .tsv, .xls, .xlsx, .sql, .sh, .bat, .ps1, .dockerfile, .gitignore, .env, .properties, .conf, .c, .cpp, .h, .hpp, .java, .go, .php, .rb, .swift, .kt, .scala, .clj, .hs, .elm, .lua, .pl, .r, .m, .tex, .bib, .org, .rst, .adoc, .wiki, .pdf, .doc, .docx, .odt, .rtf, .pages, .epub, .mobi, .azw, .azw3, .fb2, .lit, .pdb, .tcr, .prc, and other readable text files while excluding binary formats like images, videos, audio, archives, and executables
3. WHEN creating the metadata TOC THEN the system SHALL calculate file hash, size, creation date, and modification date for each file
4. WHEN the metadata scan finds duplicate file hashes THEN it SHALL identify which files have identical content
5. WHEN duplicate files are identified THEN the system SHALL mark them for exclusion from content processing
6. WHEN the metadata TOC is complete THEN the system SHALL show how many unique files were found and how many duplicates were identified
7. WHEN I specify an output database THEN the system SHALL store the metadata TOC in a files table with full_filepath, folder_path, filename, file_extension, file_type (file/folder), size, hash, creation_date, modification_date, access_date, permissions, depth_level, relative_path, is_hidden, is_symlink, symlink_target, duplicate_status, and duplicate_group_id
### Requirement 2

**User Story:** As a developer, I want to quickly ingest my unique text files into a structured database with metadata, so that I can efficiently query them with an LLM without wasting tokens on duplicates.
#### Acceptance Criteria

1. WHEN the metadata TOC is complete THEN the system SHALL process only unique files for content extraction
2. WHEN processing completes THEN the system SHALL show me how many files and paragraphs were processed
3. WHEN storing content THEN the system SHALL create a paragraphs table linked to the files table
4. WHEN storing files THEN the system SHALL update the files table with token count after content processing
### Requirement 3

**User Story:** As a developer, I want basic error handling, so that a few bad files don't stop my entire ingestion.
#### Acceptance Criteria

1. WHEN the system can't read a file during metadata scanning THEN it SHALL skip it and continue with others
2. WHEN the system can't read a file during content processing THEN it SHALL skip it and continue with others
3. WHEN the database path is invalid THEN it SHALL show a clear error message
4. WHEN the input directory doesn't exist THEN it SHALL report the error clearly
### Requirement 4

**User Story:** As a developer, I want paragraph-level deduplication with file metadata tracking, so that I don't waste LLM tokens on repetitive content and can trace content back to source files.
#### Acceptance Criteria

1. WHEN processing unique files THEN the system SHALL split content by double newlines into paragraphs
2. WHEN a paragraph is duplicate THEN the system SHALL skip it
3. WHEN a paragraph is unique THEN the system SHALL store it in the paragraphs table with file reference
4. WHEN storing file metadata THEN the system SHALL calculate and store estimated token count after content processing
5. WHEN querying the database THEN I SHALL get only unique content with source file information for LLM processing

### Requirement 5

**User Story:** As a developer, I want a simple CLI interface built in Rust, so that I can get started immediately without complex setup.
#### Acceptance Criteria

1. WHEN I run the tool THEN it SHALL accept input directory and database path as arguments
2. WHEN I run with --help THEN it SHALL show basic usage instructions
3. WHEN arguments are missing THEN it SHALL show what's required
4. WHEN processing THEN it SHALL show basic progress information for both metadata scanning and content processing phases
5. WHEN the tool runs THEN it SHALL be compiled as a native Rust binary with no external runtime dependencies


This document provides a revised, high-fidelity specification for the Pensieve project, addressing the explicit requirement for DOCX, PDF, and HTML support while maintaining the performance and portability goals. It utilizes industry-standard User Journeys (UJ) and detailed Functional Requirements (FR).

### 1\. The Architectural Solution: Hybrid Extraction Architecture (HEA)

The original requirement (R5.5) for a "native Rust binary with no external runtime dependencies" conflicts with robustly parsing complex formats like PDF and DOCX. Pure Rust implementations often lack the fidelity of established tools (e.g., Pandoc, Apache Tika).

We resolve this using a **Hybrid Extraction Architecture (HEA)**, also known as Opportunistic Orchestration:

1.  **Core Rust Binary:** Pensieve remains a self-contained, high-performance Rust binary. It handles orchestration, scanning, deduplication, database management, and native parsing of Tier 1 formats.
2.  **Native Extraction (Tier 1):** Pensieve natively parses formats with robust Rust support: HTML, Markdown, source code, and structured data (JSON, YAML).
3.  **Optional External Orchestration (Tier 2):** For complex formats (PDF, DOCX, ODT, ePUB), Pensieve will opportunistically check the host system for configured external conversion tools.
      * If found, Pensieve executes the tool as a subprocess to convert the file to plain text.
      * If not found, Pensieve skips the file and logs the missing dependency, maintaining its self-contained execution capability.

This ensures the tool is easy to deploy while leveraging the best available tools for data fidelity when present.

-----

### 2\. User Journeys and Functional Requirements

Requirements are structured by User Journey (UJ), with detailed Functional Requirements (FR) nested within each, ensuring traceability between user needs and system functionality.

#### UJ1: System Configuration and Verification

**User Story:** As a Data Scientist, I want to configure Pensieve for my environment and verify that external tools are available for complex documents, ensuring reliable data ingestion.

**Functional Requirements (FR):**

  * **FR 1.1: Configuration Management**
      * 1.1.1: The system SHALL load configuration settings from a `pensieve.toml` file. CLI arguments SHALL override the configuration file settings.
      * 1.1.2: The system SHALL provide a command `pensieve init` to generate a default `pensieve.toml` template.
      * 1.1.3: Configurable parameters SHALL include: `tokenizer_model` (e.g., `cl100k_base`), `chunk_size`, `chunk_overlap`, and `thread_count`.
  * **FR 1.2: External Dependency Configuration (Orchestration)**
      * 1.2.1: The `pensieve.toml` SHALL include a `[converters]` section allowing users to map file extensions to external command templates. This provides maximum flexibility.
          * *Example:* `pdf = "pdftotext {input} -"`
          * *Example:* `docx = "pandoc -f docx -t plain {input}"`
      * 1.2.2: The system SHALL allow specifying explicit paths to binaries in the configuration.
  * **FR 1.3: Dependency Verification**
      * 1.3.1: WHEN the user executes `pensieve check-dependencies`, THEN the system SHALL verify the configuration and attempt to locate the configured external binaries (in the specified path or system PATH).
      * 1.3.2: The system SHALL output a status report indicating which tools were found and which formats are therefore enabled (e.g., `[✓] Pandoc found. DOCX support enabled.`).

#### UJ2: High-Performance Metadata Scanning and Delta Processing

**User Story:** As a Developer, I want the system to rapidly scan the input directory, identify file types robustly, perform file-level deduplication, and only process changed files (delta processing), minimizing ingestion time.

**Functional Requirements (FR):**

  * **FR 2.1: Directory Traversal and Filtering**
      * 2.1.1: The system SHALL implement parallel directory traversal (e.g., using Rust's `rayon`).
      * 2.1.2: The system SHALL respect exclusion rules found in `.gitignore` and a project-specific `.pensieveignore` file by default.
  * **FR 2.2: Robust File Type Identification**
      * 2.2.1: The system SHALL identify Tier 1 (Native) and Tier 2 (Orchestrated) formats.
      * 2.2.2: The system SHALL use both file extensions and **MIME type sniffing** (magic number analysis) to verify file types and reliably detect binary files, mitigating risks from mislabeled files.
  * **FR 2.3: Metadata Extraction and Hashing**
      * 2.3.1: The system SHALL calculate a SHA-256 hash of the file content.
      * 2.3.2: Hashing SHALL use buffered I/O or memory mapping to handle files larger than available memory.
  * **FR 2.4: File-Level Deduplication**
      * 2.4.1: The system SHALL identify files with identical SHA-256 hashes.
      * 2.4.2: Duplicate files SHALL be recorded in the `files` table but marked with `is_canonical = FALSE` and assigned a `duplicate_group_id`.
      * 2.4.3: Only the canonical file (`is_canonical = TRUE`) SHALL proceed to the Content Extraction phase.
  * **FR 2.5: Incremental Processing (Delta Updates)**
      * 2.5.1: The system SHALL compare the current file scan (path, modification date, size) against the existing database state.
      * 2.5.2: Unchanged files SHALL be skipped. New and Modified files SHALL be queued.
      * 2.5.3: Files deleted from the filesystem SHALL be marked with `processing_status = 'Deleted'` in the database (soft delete).

#### UJ3: High-Fidelity Content Extraction and Normalization (HTML, PDF, DOCX)

**User Story:** As a Data Scientist, I want the system to accurately extract clean, normalized text from diverse formats (PDF, DOCX, HTML), preserving semantic structure where possible, so that the resulting corpus is high-quality.

**Functional Requirements (FR):**

  * **FR 3.1: Native HTML Extraction and Conditioning**
      * 3.1.1: The system SHALL parse HTML natively within the Rust binary using a compliant HTML5 parser.
      * 3.1.2: It SHALL remove non-content elements (scripts, styles, navigation, headers, footers).
      * 3.1.3: It SHALL provide an option (default: ON) to convert the main content body into Markdown (e.g., using `html2md`) to preserve semantic structure (headings, lists, tables).
  * **FR 3.2: Orchestrated PDF/DOCX Extraction**
      * 3.2.1: For Tier 2 formats, the system SHALL attempt to use the configured external tools (FR 1.2).
      * 3.2.2: WHEN an external tool is executed, THEN the system SHALL capture its STDOUT (extracted text) and STDERR (for logging).
      * 3.2.3: The system SHALL implement configurable timeouts (e.g., 120s) for external processes to prevent hanging on corrupted files.
      * 3.2.4: IF the external tool fails (non-zero exit code) or is not found, THEN the system SHALL log the error and update the file's `processing_status` to `Error` or `Skipped_Dependency`.
  * **FR 3.3: Text Normalization**
      * 3.3.1: All extracted text SHALL be normalized using Unicode normalization (NFKC) to standardize characters.
      * 3.3.2: The system SHALL normalize whitespace (collapsing multiple spaces/newlines, trimming).

#### UJ4: Intelligent Chunking and Global Deduplication

**User Story:** As an NLP Engineer, I want the system to segment the text into contextually coherent chunks using precise tokenization and perform global deduplication, maximizing the information density of the input tokens.

**Functional Requirements (FR):**

  * **FR 4.1: Context-Aware Chunking Strategies**
      * 4.1.1: The system SHALL replace the naive "double newline" splitting.
      * 4.1.2: The default strategy SHALL be **Recursive Character Splitting**.
      * 4.1.3: For structured content (like the Markdown generated in FR 3.1.3), the system SHALL utilize a **Structure-Aware Splitter** that prioritizes splitting at section headings.
  * **FR 4.2: Precise Tokenization**
      * 4.2.1: The system SHALL use the specified tokenizer (e.g., `tiktoken` library) to calculate the exact token count.
      * 4.2.2: Chunking size calculations SHALL be based on token count, not character count.
  * **FR 4.3: Global Deduplication and Provenance (M:N Model)**
      * 4.3.1: The system SHALL calculate a SHA-256 hash for each chunk. Unique chunks SHALL be stored once in the `chunks` table.
      * 4.3.2: The system SHALL implement a Many-to-Many data model using a junction table (`chunk_sources`).
      * 4.3.3: For every chunk, the system SHALL record the provenance in `chunk_sources`, linking the `chunk_id` to the `file_id`, along with the start/end index (byte offset) and the `chunking_strategy` used.

#### UJ5: Operational Control, Monitoring, and Reliability

**User Story:** As a DevOps engineer, I want robust error handling, clear progress indicators, transactional integrity, and efficient performance, so that I can reliably integrate Pensieve into automated pipelines.

**Functional Requirements (FR):**

  * **FR 5.1: CLI Usability**
      * 5.1.1: The CLI SHALL support a `--dry-run` mode that simulates the ingestion without modifying the database.
      * 5.1.2: The CLI SHALL support a `--force-reprocess` mode to ignore delta checks (FR 2.5).
  * **FR 5.2: Robust Error Handling and Logging**
      * 5.2.1: The system SHALL NOT crash due to non-fatal errors (I/O, parsing, external tool failures).
      * 5.2.2: All non-fatal errors SHALL be logged to the console (stderr) AND recorded in the database `errors` table for review.
  * **FR 5.3: Progress and Metrics Reporting**
      * 5.3.1: The system SHALL display real-time progress indicators: Files/sec, MB processed, current file path, error count, overall deduplication rate (%), and Estimated Time of Arrival (ETA).
  * **FR 5.4: Reliability and Integrity (NFR)**
      * 5.4.1: Database operations SHALL be transactional (e.g., SQLite WAL mode). The database must remain uncorrupted if the process is interrupted.
      * 5.4.2: Database insertions SHALL be optimized using batch transactions to maximize throughput.
  * **FR 5.5: Performance and Scalability (NFR)**
      * 5.5.1: The system SHALL utilize all available CPU cores by default for parallelizable tasks.
      * 5.5.2: The system SHALL be memory efficient, capable of processing a 500GB corpus without exceeding 16GB of RAM.

### 3\. Data Model Definition (SQL Schema)

```sql
-- Stores metadata for all files found in the source directory.
CREATE TABLE IF NOT EXISTS files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_filepath TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL, -- SHA-256 of file content
    size_bytes INTEGER NOT NULL,
    modification_date TIMESTAMP,
    file_extension TEXT,
    mime_type TEXT, -- From MIME sniffing (FR 2.2.2)
    is_canonical BOOLEAN NOT NULL, -- FR 2.4.2
    duplicate_group_id INTEGER, -- FR 2.4.2
    estimated_tokens INTEGER,
    processing_status TEXT CHECK(processing_status IN ('Pending', 'Processed', 'Error', 'Skipped_Binary', 'Skipped_Dependency', 'Deleted'))
);

-- Stores unique content chunks generated during processing.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    tokenizer_model TEXT NOT NULL -- e.g., 'cl100k_base'
);

-- Junction table (Many-to-Many). Links unique chunks to their source files and locations. (FR 4.3.2)
CREATE TABLE IF NOT EXISTS chunk_sources (
    chunk_id INTEGER,
    file_id INTEGER,
    start_index INTEGER NOT NULL, -- Byte offset start
    end_index INTEGER NOT NULL,   -- Byte offset end
    chunking_strategy TEXT NOT NULL, -- e.g., 'Recursive_512_50' or 'Markdown_Aware'
    PRIMARY KEY (chunk_id, file_id, start_index),
    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files (file_id) ON DELETE CASCADE
);

-- Stores non-fatal errors encountered during processing. (FR 5.2.2)
CREATE TABLE IF NOT EXISTS errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_type TEXT NOT NULL, -- e.g., 'ExtractionFailed', 'Permissions', 'MissingDependency'
    error_message TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files (file_id) ON DELETE CASCADE
);
```