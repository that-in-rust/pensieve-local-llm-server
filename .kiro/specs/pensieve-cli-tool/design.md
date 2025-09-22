# Design Document

## Overview

The Pensieve CLI tool is a high-performance Rust application designed to efficiently ingest text files into a clean, deduplicated database optimized for LLM processing. The system employs a two-phase approach: metadata scanning with file-level deduplication, followed by content extraction with intelligent chunking and global deduplication. This design maximizes token efficiency while maintaining complete traceability from content back to source files.

The system implements a **Hybrid Extraction Architecture (HEA)** that combines native Rust parsing for simple formats with opportunistic orchestration of external tools for complex formats like PDF and DOCX, ensuring both performance and high-fidelity content extraction.

### Core Design Principles

1. **Performance First**: Native Rust implementation with parallel processing and efficient memory usage
2. **Deduplication at Multiple Levels**: File-level (by hash) and content-level (by chunk) to eliminate redundancy
3. **Hybrid Extraction Architecture**: Native parsing for Tier 1 formats, external tool orchestration for Tier 2 formats (PDF, DOCX)
4. **Robust Error Handling**: Graceful degradation when individual files fail, comprehensive error logging
5. **Incremental Processing**: Delta updates to avoid reprocessing unchanged files
6. **Intelligent Chunking**: Context-aware chunking with precise tokenization for optimal LLM processing
7. **Configuration-Driven**: Flexible external tool configuration with graceful degradation

## Architecture

### High-Level Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                       │
├─────────────────────────────────────────────────────────────┤
│                 Orchestration Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Metadata Scanner │  │ Content Processor│  │ Delta Manager│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Processing Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ File Type       │  │ Content         │  │ Deduplication│ │
│  │ Detection       │  │ Extraction      │  │ Engine       │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ SQLite Database │  │ File System     │  │ External     │ │
│  │ Manager         │  │ Operations      │  │ Tools        │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
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

3. **Delta Processing Phase**
   - Compare current scan against existing database state
   - Identify new, modified, and deleted files
   - Queue only changed files for content processing

4. **Content Extraction Phase**
   - Process only unique, changed files (Requirements 2.1)
   - Extract text using native parsers or external tools
   - Split content into paragraphs by double newlines (Requirements 4.1)
   - Perform paragraph-level deduplication (Requirements 4.2, 4.3)
   - Store unique paragraphs with file references

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

The system uses a two-tier approach for robust file type identification:

**Tier 1 (Native Processing)**:
- Text files: .txt, .md, .rst, .org
- Source code: .rs, .py, .js, .ts, .java, .go, .c, .cpp, .h, .hpp, .php, .rb, .swift, .kt, .scala, .clj, .hs, .elm, .lua, .pl, .r, .m
- Configuration: .json, .yaml, .yml, .toml, .ini, .cfg, .env, .properties, .conf
- Web: .html, .css, .xml
- Scripts: .sh, .bat, .ps1, .dockerfile, .gitignore
- Data: .csv, .tsv, .log, .sql
- Documentation: .adoc, .wiki, .tex, .bib
- Spreadsheets: .xls, .xlsx (basic text extraction)

**Tier 2 (External Tool Processing)**:
- PDF documents: .pdf
- Microsoft Office: .doc, .docx
- OpenDocument: .odt, .ods
- E-books: .epub, .mobi, .azw, .azw3, .fb2, .lit, .pdb, .tcr, .prc
- Rich text: .rtf, .pages

**Binary Exclusions**:
- Images: .jpg, .png, .gif, .bmp, .svg
- Videos: .mp4, .avi, .mov, .mkv
- Audio: .mp3, .wav, .flac, .ogg
- Archives: .zip, .tar, .gz, .rar, .7z
- Executables: .exe, .bin, .app, .dmg
- Libraries: .dll, .so, .dylib

### Content Extraction Interfaces

```rust
// Trait for content extraction strategies
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    async fn extract(&self, file_path: &Path) -> Result<String, ExtractionError>;
    fn supported_extensions(&self) -> &[&str];
    fn requires_external_tool(&self) -> bool;
}

// Native text file extractor
pub struct NativeTextExtractor;

// External tool orchestrator
pub struct ExternalToolExtractor {
    tool_path: PathBuf,
    args_template: String,
    timeout: Duration,
}

// HTML content extractor with cleaning
pub struct HtmlExtractor {
    preserve_structure: bool,
    convert_to_markdown: bool,
}
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
5. **Delta Analysis**: Comparison with existing database state
6. **Content Extraction**: Text extraction using appropriate strategy
7. **Content Processing**: Intelligent chunking and global deduplication
8. **Storage**: Unique chunk persistence with complete provenance tracking

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

### External Tool Integration

For complex document formats, the system orchestrates external tools through a flexible configuration system:

**Configuration-Driven Approach**:
```toml
# pensieve.toml configuration file
[general]
tokenizer_model = "cl100k_base"
chunk_size = 512
chunk_overlap = 50
thread_count = 0  # 0 = auto-detect

[converters]
pdf = "pdftotext {input} -"
docx = "pandoc -f docx -t plain {input}"
odt = "pandoc -f odt -t plain {input}"
epub = "pandoc -f epub -t plain {input}"

[tool_paths]
pdftotext = "/usr/bin/pdftotext"  # Optional explicit paths
pandoc = "/usr/local/bin/pandoc"

[timeouts]
pdf = 120
docx = 60
default = 30
```

**Dependency Management**:
- `pensieve init` command generates default configuration template
- `pensieve check-dependencies` verifies tool availability
- Status report shows which formats are enabled
- Graceful degradation when tools are missing

**Processing Strategy**:
- Check tool availability at startup
- Skip files if required tool is missing (mark as 'Skipped_Dependency')
- Log missing dependencies for user awareness
- Continue processing other supported formats
- Configurable timeouts prevent hanging on corrupted files
- Capture STDOUT (extracted text) and STDERR (for logging)

## CLI Interface Design

### Command Structure

The CLI provides a simple, intuitive interface aligned with Requirements 5:

```bash
# Basic usage
pensieve <input_directory> <database_path>

# With options
pensieve /path/to/documents ./pensieve.db --config ./pensieve.toml

# Utility commands
pensieve init                    # Generate default configuration
pensieve check-dependencies     # Verify external tool availability
pensieve --help                 # Show usage instructions
pensieve --version              # Show version information

# Advanced options
pensieve /docs ./db.sqlite \
    --dry-run \                 # Simulate without modifying database
    --force-reprocess \         # Ignore delta checks, reprocess all
    --config ./custom.toml \    # Custom configuration file
    --threads 8 \               # Override thread count
    --verbose                   # Detailed progress output
```

### Progress Reporting

Real-time progress indicators as specified in Requirements FR 5.3:

```
Pensieve v1.0.0 - Text Ingestion Tool

Phase 1: Metadata Scanning
├─ Files scanned: 15,432 / 20,000 (77%)
├─ Processing rate: 1,247 files/sec
├─ Data processed: 2.3 GB / 3.1 GB
├─ Duplicates found: 3,421 (22.1%)
├─ Errors: 12
└─ ETA: 00:03:45

Phase 2: Content Extraction  
├─ Files processed: 8,234 / 12,011 (68%)
├─ Chunks created: 145,678
├─ Deduplication rate: 34.2%
├─ Token count: 12.4M tokens
├─ Processing rate: 234 files/sec
└─ ETA: 00:12:33

Summary:
✓ Files processed: 12,011
✓ Unique chunks: 95,847
✓ Total tokens: 12.4M
✓ Deduplication savings: 34.2%
✓ Processing time: 00:16:18
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

# Missing external tools
Warning: pdftotext not found in PATH
PDF files will be skipped. Install poppler-utils to enable PDF processing
Run 'pensieve check-dependencies' for detailed tool status
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

### 3. Hybrid Extraction Architecture

**Decision**: Native parsing for simple formats, external tools for complex formats

**Rationale**:
- **Performance**: Native Rust parsing is faster and more reliable for text formats
- **Fidelity**: External tools (Pandoc, pdftotext) provide better quality for complex formats
- **Maintainability**: Avoid reimplementing complex document parsers
- **Flexibility**: Users can configure their preferred tools

**Trade-offs**: External dependencies, but graceful degradation when tools are missing

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

### 6. Incremental Processing with Delta Detection

**Decision**: Compare file metadata to detect changes and avoid reprocessing

**Rationale**:
- **Performance**: Dramatically reduces processing time for subsequent runs
- **User Experience**: Faster iteration cycles during development
- **Resource Efficiency**: Avoids redundant computation and I/O
- **Scalability**: Enables processing of very large corpora over time

**Trade-offs**: Additional complexity in change detection logic, but essential for practical use

### 7. Configuration-Driven External Tool Integration

**Decision**: Implement flexible TOML-based configuration for external tools

**Rationale**:
- **User Control**: Users can specify their preferred tools and configurations
- **Flexibility**: Support for custom command templates and tool paths
- **Maintainability**: No need to hardcode tool configurations in source code
- **Extensibility**: Easy to add support for new tools without code changes

**Implementation**:
- **Configuration File**: `pensieve.toml` with sections for converters, tool paths, and timeouts
- **Command Templates**: Flexible template system with `{input}` placeholder
- **Dependency Checking**: `check-dependencies` command verifies tool availability
- **Graceful Degradation**: Missing tools result in skipped files, not failures

**Trade-offs**: Requires configuration management but provides maximum flexibility and user control

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

This design provides a robust, performant, and maintainable foundation for the Pensieve CLI tool while addressing all requirements specified in the requirements document. The focus on simple paragraph-based processing ensures the MVP can be delivered quickly while still providing effective deduplication and LLM-ready content organization.