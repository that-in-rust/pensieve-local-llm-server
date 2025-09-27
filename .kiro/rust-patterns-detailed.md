# Detailed Rust Code Ingestion Patterns

## Layer 1: Core Domain Types (L1 - Core Language Features)

### Type-Safe File System Representation
```rust
// ✅ Newtype pattern prevents confusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepoId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub Uuid);

// ✅ Make invalid states unrepresentable
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    DirectText,      // .rs, .py, .js, .md, .txt
    Convertible { command: String }, // .pdf, .docx via pandoc
    NonText,         // .jpg, .png, .bin
}

// ✅ State machine for processing stages
#[derive(Debug)]
pub enum FileNode<State> {
    Unprocessed { path: PathBuf, size: u64 },
    Processed { 
        path: PathBuf, 
        content: Option<String>,
        metrics: FileMetrics,
        file_type: FileType,
    },
    _phantom: PhantomData<State>,
}

#[derive(Debug, Clone)]
pub struct FileMetrics {
    pub size_bytes: u64,
    pub line_count: Option<u32>,
    pub word_count: Option<u32>,
    pub token_count: Option<u32>,
}
```

### Repository Cloning with Executable Contracts
```rust
/// Repository cloning with performance contract
/// 
/// # Preconditions
/// - Valid GitHub URL format
/// - Local path is writable
/// - Network connectivity available
/// 
/// # Postconditions  
/// - Repository cloned to specified path
/// - Returns Ok(CloneResult) with commit hash
/// - Cleanup on failure (RAII)
/// 
/// # Performance Contract
/// - Completes within 30s for repos <100MB
/// - Uses partial clone for repos >500MB
/// 
/// # Error Conditions
/// - CloneError::NetworkTimeout for network issues
/// - CloneError::AuthenticationFailed for private repos
/// - CloneError::InvalidUrl for malformed URLs
pub async fn clone_repository(
    url: &str,
    local_path: &Path,
    config: CloneConfig,
) -> Result<CloneResult, CloneError>;
```

## Performance Contracts and Testing

### Test-Validated Performance Claims
```rust
#[tokio::test]
async fn test_file_processing_performance_contract() {
    let processor = create_test_processor().await;
    
    // Load test data: 1000 Rust files
    let test_files = generate_test_files(1000).await;
    
    let start = Instant::now();
    let results = processor.process_files_batch(test_files).await.unwrap();
    let elapsed = start.elapsed();
    
    // Validate performance contract: 1000 files/second minimum
    assert!(elapsed < Duration::from_secs(1), 
            "Processing took {:?}, expected <1s for 1000 files", elapsed);
    assert_eq!(results.len(), 1000);
}
```

## Rust Ecosystem Integration

### Essential Crates
```rust
// Git operations
git2 = "0.18"           // Git repository operations
tempfile = "3.8"        // Temporary directory management

// Async runtime
tokio = { version = "1.0", features = ["full"] }
tokio-stream = "0.1"    // Async stream processing

// Database
sqlx = { version = "0.7", features = ["postgres", "runtime-tokio-rustls", "macros", "uuid", "chrono"] }

// Error handling
thiserror = "1.0"       // Structured errors
anyhow = "1.0"          // Application error context

// CLI
clap = { version = "4.0", features = ["derive"] }

// File processing
ignore = "0.4"          // .gitignore pattern matching
walkdir = "2.4"         // Directory traversal
```

This file contains the detailed implementation patterns referenced in the steering document.