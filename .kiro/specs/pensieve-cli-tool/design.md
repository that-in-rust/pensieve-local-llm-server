# Design Document



## IMPORTANT FOR VISUALS AND DIAGRAMS

ALL DIAGRAMS WILL BE IN MERMAID ONLY TO ENSURE EASE WITH GITHUB - DO NOT SKIP THAT


## Overview

The Pensieve CLI tool is a simple, high-performance Rust application designed to quickly ingest text files into a clean, deduplicated database for LLM processing. The system employs a straightforward two-phase approach: metadata scanning with file-level deduplication, followed by content extraction with simple paragraph-based processing. This MVP design focuses on getting content into a queryable format quickly without complex features or optimization.

The system prioritizes simplicity and reliability, using native Rust parsing for supported text formats while maintaining a self-contained binary with no external runtime dependencies as specified in Requirement 5.5.

### Core Design Principles

1. **Simplicity First**: MVP approach focused on getting content into queryable format quickly
2. **Native Rust Implementation**: Self-contained binary with no external runtime dependencies (Requirement 5.5)
3. **Two-Level Deduplication**: File-level (by hash) and paragraph-level (by content hash) to eliminate redundancy
4. **Basic Error Handling**: Skip problematic files and continue processing (Requirement 3)
5. **Simple Content Processing**: Split content by double newlines into paragraphs (Requirement 4.1)
6. **Progress Reporting**: Show basic progress for both metadata scanning and content processing phases (Requirement 5.4)

## Architecture

### High-Level Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                       │
├─────────────────────────────────────────────────────────────┤
│                 Processing Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Metadata Scanner │  │ Content         │  │ Paragraph    │ │
│  │                 │  │ Extractor       │  │ Processor    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ SQLite Database │  │ File System     │                   │
│  │ Manager         │  │ Operations      │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Processing Flow

1. **Initialization Phase**
   - Parse CLI arguments and validate paths
   - Initialize database connection and create tables
   - Load configuration and verify external dependencies

2. **Metadata Scanning Phase**
   - Parallel directory traversal with file type detection
   - Calculate SHA-256 hashes for content-based deduplication
   - Store metadata in files table with duplicate marking
   - Generate progress reports and statistics



3. **Content Extraction Phase**
   - Process only unique files for content extraction (Requirement 2.1)
   - Extract text using native Rust parsing for supported formats
   - Split content into paragraphs by double newlines (Requirement 4.1)
   - Perform paragraph-level deduplication (Requirements 4.2, 4.3)
   - Store unique paragraphs with file references in paragraphs table

## Components and Interfaces

### Core Data Types

```rust
// File metadata representation
#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub full_filepath: PathBuf,
    pub folder_path: PathBuf,
    pub filename: String,
    pub file_extension: Option<String>,
    pub file_type: FileType,
    pub size: u64,
    pub hash: String, // SHA-256
    pub creation_date: DateTime<Utc>,
    pub modification_date: DateTime<Utc>,
    pub access_date: DateTime<Utc>,
    pub permissions: u32,
    pub depth_level: u32,
    pub relative_path: PathBuf,
    pub is_hidden: bool,
    pub is_symlink: bool,
    pub symlink_target: Option<PathBuf>,
    pub duplicate_status: DuplicateStatus,
    pub duplicate_group_id: Option<Uuid>,
}

// Processing status tracking
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStatus {
    Pending,
    Processed,
    Error,
    SkippedBinary,
    SkippedDependency,
    Deleted,
}

// File type classification
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    File,
    Directory,
}

// Duplicate status for deduplication
#[derive(Debug, Clone, PartialEq)]
pub enum DuplicateStatus {
    Unique,
    Canonical,    // First occurrence of duplicate content
    Duplicate,    // Subsequent occurrences
}
```

### File Type Detection System

The system uses native Rust parsing for all supported text formats as specified in Requirements 1.2:

**Supported Text Formats (Native Processing)**:
- Text files: .txt, .md, .rst, .org
- Source code: .rs, .py, .js, .ts, .java, .go, .c, .cpp, .h, .hpp, .php, .rb, .swift, .kt, .scala, .clj, .hs, .elm, .lua, .pl, .r, .m
- Configuration: .json, .yaml, .yml, .toml, .ini, .cfg, .env, .properties, .conf
- Web: .html, .css, .xml
- Scripts: .sh, .bat, .ps1, .dockerfile, .gitignore
- Data: .csv, .tsv, .log, .sql
- Documentation: .adoc, .wiki, .tex, .bib
- Spreadsheets: .xls, .xlsx (basic text extraction)
- Documents: .pdf, .doc, .docx, .odt, .rtf, .pages
- E-books: .epub, .mobi, .azw, .azw3, .fb2, .lit, .pdb, .tcr, .prc

**Binary Exclusions**:
- Images: .jpg, .png, .gif, .bmp, .svg
- Videos: .mp4, .avi, .mov, .mkv
- Audio: .mp3, .wav, .flac, .ogg
- Archives: .zip, .tar, .gz, .rar, .7z
- Executables: .exe, .bin, .app, .dmg
- Libraries: .dll, .so, .dylib

**Note**: For MVP, complex formats like PDF and DOCX will use basic text extraction methods available in Rust crates, maintaining the self-contained binary requirement.

### Content Extraction Interfaces

```rust
// Simple trait for content extraction
pub trait ContentExtractor: Send + Sync {
    fn extract(&self, file_path: &Path) -> Result<String, ExtractionError>;
    fn supported_extensions(&self) -> &[&str];
}

// Native text file extractor for simple formats
pub struct TextExtractor;

// Basic HTML extractor that strips tags
pub struct HtmlExtractor;

// Simple PDF text extractor using Rust crates
pub struct PdfExtractor;

// Basic DOCX extractor using Rust crates
pub struct DocxExtractor;
```

### Database Schema Design

The database schema supports the complete workflow with proper relationships and indexing, following the requirements for simple paragraph-based processing:

```sql
-- File metadata with comprehensive tracking (Requirements 1.7)
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_filepath TEXT NOT NULL UNIQUE,
    folder_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_extension TEXT,
    file_type TEXT NOT NULL CHECK(file_type IN ('file', 'folder')),
    size INTEGER NOT NULL,
    hash TEXT NOT NULL, -- SHA-256 of file content
    creation_date TIMESTAMP,
    modification_date TIMESTAMP,
    access_date TIMESTAMP,
    permissions INTEGER,
    depth_level INTEGER NOT NULL,
    relative_path TEXT NOT NULL,
    is_hidden BOOLEAN NOT NULL DEFAULT FALSE,
    is_symlink BOOLEAN NOT NULL DEFAULT FALSE,
    symlink_target TEXT,
    duplicate_status TEXT NOT NULL CHECK(duplicate_status IN ('unique', 'canonical', 'duplicate')),
    duplicate_group_id TEXT,
    processing_status TEXT NOT NULL DEFAULT 'pending' 
        CHECK(processing_status IN ('pending', 'processed', 'error', 'skipped_binary', 'skipped_dependency', 'deleted')),
    estimated_tokens INTEGER, -- Updated after content processing (Requirements 2.4)
    processed_at TIMESTAMP,
    error_message TEXT,
    mime_type TEXT -- From MIME sniffing for robust file type detection
);

-- Simple paragraph storage linked to files (Requirements 2.3, 4.3)
CREATE TABLE paragraphs (
    paragraph_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL, -- SHA-256 for deduplication (Requirements 4.2)
    paragraph_index INTEGER NOT NULL, -- Position within the file
    estimated_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
);

-- Processing errors for debugging and monitoring
CREATE TABLE errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    error_type TEXT NOT NULL, -- e.g., 'ExtractionFailed', 'Permissions', 'MissingDependency'
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX idx_files_hash ON files(hash);
CREATE INDEX idx_files_duplicate_group ON files(duplicate_group_id);
CREATE INDEX idx_files_processing_status ON files(processing_status);
CREATE INDEX idx_files_modification_date ON files(modification_date);
CREATE INDEX idx_files_mime_type ON files(mime_type);
CREATE INDEX idx_paragraphs_hash ON paragraphs(content_hash);
CREATE INDEX idx_paragraphs_file ON paragraphs(file_id);
```

## Simple Paragraph Processing System

### Content Splitting Strategy

Following the MVP requirements, the system implements simple paragraph-based content processing:

**Paragraph Splitting (Requirements 4.1)**:
- Split content by double newlines (`\n\n`) to identify paragraph boundaries
- Each paragraph becomes a separate record in the paragraphs table
- Simple and reliable approach suitable for most text content

**Deduplication Strategy (Requirements 4.2, 4.3)**:
- Calculate SHA-256 hash for each paragraph
- Skip paragraphs that already exist in the database
- Store only unique paragraphs with file reference for traceability

### Content Processing Interface

```rust
// Simple paragraph processor
pub struct ParagraphProcessor {
    min_paragraph_length: usize,
    max_paragraph_length: usize,
}

impl ParagraphProcessor {
    pub fn new() -> Self {
        Self {
            min_paragraph_length: 10,   // Skip very short paragraphs
            max_paragraph_length: 10000, // Split very long paragraphs
        }
    }
    
    pub fn split_content(&self, content: &str) -> Vec<String> {
        content
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty() && p.len() >= self.min_paragraph_length)
            .map(|p| p.to_string())
            .collect()
    }
    
    pub fn calculate_hash(&self, content: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}
```

### Token Estimation

```rust
// Simple token estimation for paragraph content
pub struct TokenEstimator;

impl TokenEstimator {
    pub fn estimate_tokens(&self, text: &str) -> usize {
        // Simple estimation: ~4 characters per token for English text
        // This is a rough approximation suitable for MVP
        (text.len() as f64 / 4.0).ceil() as usize
    }
    
    pub fn estimate_words(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}
```

## Data Models

### File Processing Pipeline

The system processes files through a well-defined pipeline:

1. **Discovery**: Recursive directory traversal with parallel processing
2. **Classification**: MIME type detection and extension-based filtering
3. **Hashing**: SHA-256 calculation for content-based deduplication
4. **Metadata Storage**: Complete file information persistence
5. **Content Extraction**: Text extraction using native Rust parsing
6. **Content Processing**: Simple paragraph splitting and deduplication
7. **Storage**: Unique paragraph persistence with file references

### Deduplication Strategy

**File-Level Deduplication**:
- SHA-256 hash calculation for entire file content
- First occurrence marked as "canonical" 
- Subsequent occurrences marked as "duplicate" with group ID
- Only canonical files proceed to content processing

**Content-Level Deduplication (Simple Paragraph Model)**:
- Split content by double newlines into paragraphs (Requirements 4.1)
- Calculate SHA-256 hash for each paragraph
- Skip duplicate paragraphs during processing (Requirements 4.2)
- Store unique paragraphs once in paragraphs table with file reference (Requirements 4.3)
- Enable traceability from any paragraph back to source file

### Native Content Extraction

The system uses only native Rust libraries for content extraction to maintain the self-contained binary requirement:

**Text Extraction Strategy**:
- **Plain Text**: Direct file reading with encoding detection
- **HTML**: Basic tag stripping using native HTML parsing
- **PDF**: Simple text extraction using Rust PDF libraries (e.g., `pdf-extract`)
- **DOCX**: Basic text extraction using Rust ZIP and XML parsing
- **JSON/YAML/TOML**: Parse and extract string values
- **CSV**: Extract all text content from cells
- **Source Code**: Read as plain text (comments and strings included)

**Encoding Handling**:
- UTF-8 detection and conversion
- Fallback to Latin-1 for legacy files
- Skip files with unrecognizable encodings

**Error Handling**:
- Skip files that cannot be processed
- Log extraction failures for user awareness
- Continue processing other files (Requirement 3)

## CLI Interface Design

### Command Structure

The CLI provides a simple interface aligned with Requirement 5:

```bash
# Basic usage (Requirements 5.1)
pensieve <input_directory> <database_path>

# Help and version (Requirements 5.2, 5.3)
pensieve --help                 # Show basic usage instructions
pensieve --version              # Show version information

# Example usage
pensieve /path/to/documents ./pensieve.db
```

### Progress Reporting

Basic progress information as specified in Requirement 5.4:

```
Pensieve v1.0.0 - Text Ingestion Tool

Phase 1: Metadata Scanning
Scanning files... 15,432 files found
Duplicates identified: 3,421 files
Unique files to process: 12,011

Phase 2: Content Processing
Processing files... 8,234 / 12,011 (68%)
Paragraphs created: 145,678
Errors: 12

Summary:
✓ Files processed: 12,011
✓ Paragraphs stored: 95,847
✓ Duplicates skipped: 49,831
✓ Processing complete
```

### Error Handling and User Feedback

Clear, actionable error messages as specified in Requirements 3:

```bash
# Missing arguments
$ pensieve
Error: Missing required arguments
Usage: pensieve <input_directory> <database_path>
Run 'pensieve --help' for more information

# Invalid directory
$ pensieve /nonexistent ./db.sqlite
Error: Input directory '/nonexistent' does not exist
Please check the path and try again

# Invalid database path
$ pensieve ./docs /readonly/db.sqlite
Error: Cannot write to database path '/readonly/db.sqlite'
Permission denied. Please choose a writable location

# File processing errors
Warning: Could not process 'document.pdf': Unsupported format
12 files skipped due to processing errors
See error details in database for troubleshooting
```

## Error Handling

### Error Hierarchy

The system defines a comprehensive error hierarchy for different failure modes:

```rust
#[derive(Error, Debug)]
pub enum PensieveError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("File processing error: {file_path} - {cause}")]
    FileProcessing { file_path: PathBuf, cause: String },
    
    #[error("External tool error: {tool} - {message}")]
    ExternalTool { tool: String, message: String },
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },
}

#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error("Unsupported file type: {extension}")]
    UnsupportedType { extension: String },
    
    #[error("External tool not found: {tool}")]
    ToolNotFound { tool: String },
    
    #[error("External tool timeout: {tool} after {timeout:?}")]
    ToolTimeout { tool: String, timeout: Duration },
    
    #[error("Content too large: {size} bytes (max: {max})")]
    ContentTooLarge { size: u64, max: u64 },
    
    #[error("Encoding error: {0}")]
    Encoding(String),
}
```

### Error Recovery Strategy

1. **Non-Fatal Errors**: Individual file failures don't stop processing
2. **Error Logging**: All errors recorded in database and console
3. **Retry Logic**: Transient failures (network, temporary locks) get retried
4. **Graceful Degradation**: Missing external tools result in skipped files, not crashes
5. **Progress Preservation**: Partial progress is saved and can be resumed

## Testing Strategy

### Unit Testing

**Core Components**:
- File type detection with various file samples
- Hash calculation consistency and performance
- Content extraction for each supported format
- Deduplication logic with edge cases
- Database operations with transaction safety

**Property-Based Testing**:
- File path handling across different operating systems
- Hash collision detection (theoretical but important)
- Content splitting boundary conditions
- Unicode handling in various encodings

### Integration Testing

**End-to-End Workflows**:
- Complete ingestion pipeline with sample directory structures
- Delta processing with file modifications
- External tool integration with mock tools
- Database consistency after interruptions
- Performance testing with large file sets

**Error Scenario Testing**:
- Corrupted files and malformed content
- Missing external dependencies
- Database connection failures
- Disk space exhaustion
- Permission denied scenarios

### Performance Testing

**Benchmarks**:
- File scanning rate (files per second)
- Hash calculation throughput (MB/s)
- Database insertion performance (records per second)
- Memory usage patterns under load
- Concurrent processing efficiency

**Scalability Testing**:
- Large directory structures (>100k files)
- Large individual files (>1GB)
- Deep directory nesting (>20 levels)
- High duplicate ratios (>90% duplicates)
- Mixed file type distributions

### Contract Testing

**Performance Contracts**:
```rust
#[test]
fn test_file_scanning_performance() {
    // Must process at least 1000 files per second on standard hardware
    let start = Instant::now();
    let result = scan_directory(&test_dir_1000_files).await.unwrap();
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_secs(1));
    assert_eq!(result.len(), 1000);
}

#[test]
fn test_memory_usage_bounds() {
    // Memory usage must not exceed 16GB for 500GB corpus
    let initial_memory = get_memory_usage();
    process_large_corpus(&test_corpus_500gb).await.unwrap();
    let peak_memory = get_peak_memory_usage();
    
    assert!(peak_memory - initial_memory < 16 * 1024 * 1024 * 1024);
}
```

## Design Decisions and Rationales

### 1. SQLite as Primary Database

**Decision**: Use SQLite with WAL mode for data persistence

**Rationale**:
- **Simplicity**: Single file database, no server setup required
- **Performance**: Excellent for read-heavy workloads with batch writes
- **Reliability**: ACID transactions with WAL mode for crash safety
- **Portability**: Database file can be easily moved between systems
- **Tooling**: Rich ecosystem of tools for analysis and debugging

**Trade-offs**: Limited concurrent write performance, but acceptable for CLI tool usage patterns

### 2. Two-Phase Processing Architecture

**Decision**: Separate metadata scanning from content extraction

**Rationale**:
- **Efficiency**: File-level deduplication eliminates redundant content processing
- **Resumability**: Metadata phase can complete independently, enabling incremental processing
- **Progress Tracking**: Clear separation allows for better user feedback
- **Error Isolation**: Metadata failures don't affect content processing and vice versa

**Trade-offs**: Slightly more complex implementation, but significant performance benefits

### 3. Native-Only Processing Architecture

**Decision**: Use native Rust parsing for all supported formats

**Rationale**:
- **Self-Contained**: Meets Requirement 5.5 for no external runtime dependencies
- **Simplicity**: Single binary deployment with no configuration needed
- **Reliability**: No external tool failures or missing dependencies
- **MVP Focus**: Basic text extraction sufficient for initial version

**Trade-offs**: Lower extraction fidelity for complex formats, but maintains simplicity and meets requirements

### 4. Content-Hash Based Deduplication

**Decision**: Use SHA-256 hashes for both file and paragraph deduplication

**Rationale**:
- **Accuracy**: Cryptographic hash eliminates false positives
- **Performance**: Single hash calculation serves both deduplication and integrity checking
- **Deterministic**: Same content always produces same hash across runs
- **Collision Resistance**: SHA-256 provides sufficient collision resistance for practical use

**Trade-offs**: Slightly slower than non-cryptographic hashes, but negligible for file sizes involved

### 5. Simple Paragraph-Based Processing

**Decision**: Implement simple paragraph splitting by double newlines for MVP

**Rationale**:
- **Simplicity**: Easy to understand and implement for MVP requirements
- **Reliability**: Double newline splitting is predictable and works across content types
- **Performance**: Fast processing without complex tokenization overhead
- **MVP Focus**: Meets requirements without over-engineering for initial version

**Implementation**:
- **Single Strategy**: Split content by double newlines (`\n\n`) (Requirements 4.1)
- **Simple Deduplication**: SHA-256 hash comparison for duplicate detection (Requirements 4.2)
- **Basic Token Estimation**: Simple character-based estimation (~4 chars per token)
- **File Traceability**: Each paragraph linked to source file (Requirements 4.3)

**Trade-offs**: Less sophisticated than advanced chunking but sufficient for MVP and easier to implement correctly

### 6. Simple Processing Model

**Decision**: Process all files on each run without delta detection for MVP

**Rationale**:
- **Simplicity**: Reduces implementation complexity for initial version
- **Reliability**: No state management or change detection edge cases
- **MVP Focus**: Get basic functionality working first
- **File-Level Deduplication**: Still avoids processing duplicate files

**Trade-offs**: Slower subsequent runs, but acceptable for MVP and can be optimized later

### 7. Native-Only Content Extraction

**Decision**: Use only native Rust libraries for content extraction

**Rationale**:
- **Self-Contained Binary**: Meets Requirement 5.5 for no external runtime dependencies
- **Simplicity**: Reduces complexity and deployment requirements
- **Reliability**: No external tool failures or version compatibility issues
- **Portability**: Works on any system where Rust binary can run

**Implementation**:
- **Native Libraries**: Use Rust crates for PDF, DOCX, HTML parsing
- **Basic Extraction**: Focus on getting text content rather than perfect formatting
- **Error Handling**: Skip files that cannot be processed natively

**Trade-offs**: Lower fidelity extraction compared to specialized tools, but meets MVP requirements and maintains simplicity

### 8. Simple Paragraph Deduplication Model

**Decision**: Implement simple paragraph storage with file references

**Rationale**:
- **MVP Simplicity**: Direct relationship between paragraphs and files is easier to understand
- **Requirements Compliance**: Matches the specified paragraphs table structure (Requirements 2.3)
- **Sufficient Traceability**: Each paragraph knows its source file for LLM processing (Requirements 4.5)
- **Performance**: Simple schema reduces complexity and improves query performance

**Implementation**:
- **Paragraphs Table**: Stores unique paragraphs with file references
- **Direct Relationship**: Each paragraph belongs to exactly one file
- **Hash-Based Deduplication**: Skip paragraphs with duplicate content hashes
- **Source Tracking**: Maintain file_id and paragraph_index for traceability

**Trade-offs**: Less sophisticated than M:N model but meets MVP requirements and is much simpler to implement

This design provides a simple, reliable foundation for the Pensieve CLI tool MVP while addressing all requirements specified in the requirements document. The focus on native Rust processing and simple paragraph-based content organization ensures the tool can be delivered quickly as a self-contained binary while still providing effective deduplication and LLM-ready content storage.