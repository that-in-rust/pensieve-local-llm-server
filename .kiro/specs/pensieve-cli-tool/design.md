# Design Document

## Overview

The Pensieve CLI tool is a high-performance Rust application designed to efficiently ingest text files into a clean, deduplicated database optimized for LLM processing. The system employs a two-phase approach: metadata scanning with file-level deduplication, followed by content extraction with paragraph-level deduplication. This design maximizes token efficiency while maintaining complete traceability from content back to source files.

### Core Design Principles

1. **Performance First**: Native Rust implementation with parallel processing and efficient memory usage
2. **Deduplication at Multiple Levels**: File-level (by hash) and content-level (by paragraph) to eliminate redundancy
3. **Comprehensive Format Support**: Native parsing for simple formats, external tool orchestration for complex formats (PDF, DOCX)
4. **Robust Error Handling**: Graceful degradation when individual files fail
5. **Incremental Processing**: Delta updates to avoid reprocessing unchanged files

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
   - Process only unique, changed files
   - Extract text using native parsers or external tools
   - Split content into paragraphs and deduplicate
   - Store unique paragraphs with source file references

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
- Source code: .rs, .py, .js, .ts, .java, .go, .c, .cpp, .h, .hpp
- Configuration: .json, .yaml, .yml, .toml, .ini, .cfg, .env
- Web: .html, .css, .xml
- Scripts: .sh, .bat, .ps1
- Data: .csv, .tsv, .log
- Documentation: .adoc, .wiki, .tex, .bib

**Tier 2 (External Tool Processing)**:
- PDF documents: .pdf
- Microsoft Office: .docx, .xlsx
- OpenDocument: .odt, .ods
- E-books: .epub, .mobi, .azw, .azw3
- Rich text: .rtf

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

The database schema supports the complete workflow with proper relationships and indexing:

```sql
-- File metadata with comprehensive tracking
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_filepath TEXT NOT NULL UNIQUE,
    folder_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_extension TEXT,
    file_type TEXT NOT NULL CHECK(file_type IN ('file', 'folder')),
    size INTEGER NOT NULL,
    hash TEXT NOT NULL,
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
    estimated_tokens INTEGER,
    processed_at TIMESTAMP,
    error_message TEXT
);

-- Unique content paragraphs
CREATE TABLE paragraphs (
    paragraph_id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many relationship between paragraphs and source files
CREATE TABLE paragraph_sources (
    paragraph_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    paragraph_index INTEGER NOT NULL, -- Position within the file
    byte_offset_start INTEGER NOT NULL,
    byte_offset_end INTEGER NOT NULL,
    PRIMARY KEY (paragraph_id, file_id, paragraph_index),
    FOREIGN KEY (paragraph_id) REFERENCES paragraphs(paragraph_id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
);

-- Processing errors for debugging and monitoring
CREATE TABLE processing_errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX idx_files_hash ON files(hash);
CREATE INDEX idx_files_duplicate_group ON files(duplicate_group_id);
CREATE INDEX idx_files_processing_status ON files(processing_status);
CREATE INDEX idx_files_modification_date ON files(modification_date);
CREATE INDEX idx_paragraphs_hash ON paragraphs(content_hash);
CREATE INDEX idx_paragraph_sources_file ON paragraph_sources(file_id);
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
7. **Content Processing**: Paragraph splitting and deduplication
8. **Storage**: Unique content persistence with source tracking

### Deduplication Strategy

**File-Level Deduplication**:
- SHA-256 hash calculation for entire file content
- First occurrence marked as "canonical"
- Subsequent occurrences marked as "duplicate" with group ID
- Only canonical files proceed to content processing

**Content-Level Deduplication**:
- Split extracted text by double newlines (paragraph boundaries)
- Calculate SHA-256 hash for each paragraph
- Store unique paragraphs once in paragraphs table
- Track all source locations in paragraph_sources junction table

### External Tool Integration

For complex document formats, the system orchestrates external tools:

**Configuration-Driven Approach**:
```toml
[extractors.pdf]
command = "pdftotext"
args = ["{input}", "-"]
timeout = 120
required = false

[extractors.docx]
command = "pandoc"
args = ["-f", "docx", "-t", "plain", "{input}"]
timeout = 60
required = false
```

**Graceful Degradation**:
- Check tool availability at startup
- Skip files if required tool is missing
- Log missing dependencies for user awareness
- Continue processing other supported formats

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

### 5. Paragraph-Level Content Splitting

**Decision**: Split content by double newlines for paragraph boundaries

**Rationale**:
- **Simplicity**: Universal pattern across text formats
- **LLM Compatibility**: Paragraph-sized chunks are optimal for most LLM contexts
- **Deduplication Granularity**: Balances between too fine (sentences) and too coarse (files)
- **Semantic Coherence**: Paragraphs typically contain coherent thoughts

**Trade-offs**: May not be optimal for all document structures, but provides good general-purpose chunking

### 6. Incremental Processing with Delta Detection

**Decision**: Compare file metadata to detect changes and avoid reprocessing

**Rationale**:
- **Performance**: Dramatically reduces processing time for subsequent runs
- **User Experience**: Faster iteration cycles during development
- **Resource Efficiency**: Avoids redundant computation and I/O
- **Scalability**: Enables processing of very large corpora over time

**Trade-offs**: Additional complexity in change detection logic, but essential for practical use

This design provides a robust, performant, and maintainable foundation for the Pensieve CLI tool while addressing all requirements specified in the requirements document.