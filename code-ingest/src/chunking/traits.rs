//! Trait definitions for file chunking operations
//! 
//! Following dependency injection principles - all components depend on traits

use std::path::{Path, PathBuf};
use async_trait::async_trait;
use crate::chunking::{ChunkingResult, ChunkingError};

/// Configuration for chunking operations
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Size of each chunk in MB
    pub chunk_size_mb: f64,
    /// Whether to validate chunks after creation
    pub validate_chunks: bool,
    /// Whether to clean up chunks on validation failure
    pub cleanup_on_failure: bool,
    /// Custom output directory (None = use default)
    pub output_dir: Option<PathBuf>,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size_mb: crate::chunking::DEFAULT_CHUNK_SIZE_MB,
            validate_chunks: true,
            cleanup_on_failure: true,
            output_dir: None,
        }
    }
}

impl ChunkingConfig {
    /// Create a new configuration with specified chunk size
    pub fn with_chunk_size(chunk_size_mb: f64) -> ChunkingResult<Self> {
        if chunk_size_mb <= 0.0 || chunk_size_mb > 10.0 {
            return Err(ChunkingError::invalid_input(
                format!("Chunk size must be between 0.1 and 10.0 MB, got {}", chunk_size_mb)
            ));
        }
        
        Ok(Self {
            chunk_size_mb,
            ..Default::default()
        })
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> ChunkingResult<()> {
        if self.chunk_size_mb <= 0.0 {
            return Err(ChunkingError::configuration("Chunk size must be positive"));
        }
        
        if self.chunk_size_mb > 10.0 {
            return Err(ChunkingError::configuration("Chunk size too large (max 10MB)"));
        }
        
        Ok(())
    }
}

/// Information about a chunked file
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// Path to the original file
    pub original_file: PathBuf,
    /// Paths to the generated chunks
    pub chunk_paths: Vec<PathBuf>,
    /// Size of each chunk in bytes
    pub chunk_size_bytes: u64,
    /// Total size of original file
    pub original_size_bytes: u64,
    /// Whether chunks were validated
    pub validated: bool,
}

impl ChunkInfo {
    /// Get the number of chunks created
    pub fn chunk_count(&self) -> usize {
        self.chunk_paths.len()
    }
    
    /// Check if the file was actually chunked (more than 1 chunk)
    pub fn is_chunked(&self) -> bool {
        self.chunk_paths.len() > 1
    }
}

/// Trait for file chunking operations
/// 
/// # Contract
/// 
/// ## Preconditions
/// - Input file must exist and be readable
/// - Configuration must be valid
/// - Sufficient disk space for chunks
/// 
/// ## Postconditions
/// - Returns ChunkInfo with paths to created chunks
/// - Original file remains unchanged
/// - Chunks are created in specified or default directory
/// - If validation enabled, chunks are verified against original
/// 
/// ## Error Conditions
/// - ChunkingError::InvalidInput if file doesn't exist or config invalid
/// - ChunkingError::Io if file operations fail
/// - ChunkingError::ValidationFailed if chunk validation fails
#[async_trait]
pub trait FileChunker: Send + Sync {
    /// Chunk a file according to the configuration
    async fn chunk_file(
        &self,
        file_path: &Path,
        config: &ChunkingConfig,
    ) -> ChunkingResult<ChunkInfo>;
    
    /// Check if a file should be chunked based on size
    async fn should_chunk(&self, file_path: &Path) -> ChunkingResult<bool>;
    
    /// Get the estimated number of chunks for a file
    async fn estimate_chunks(&self, file_path: &Path, config: &ChunkingConfig) -> ChunkingResult<usize>;
}

/// Trait for validating chunked files
/// 
/// # Contract
/// 
/// ## Preconditions
/// - Original file must exist and be readable
/// - All chunk files must exist and be readable
/// 
/// ## Postconditions
/// - Returns true if chunks exactly reconstruct the original file
/// - Returns false if validation fails
/// 
/// ## Error Conditions
/// - ChunkingError::Io if file operations fail
/// - ChunkingError::ValidationFailed if chunks are corrupted
#[async_trait]
pub trait ChunkValidator: Send + Sync {
    /// Validate that chunks correctly represent the original file
    async fn validate_chunks(
        &self,
        original_file: &Path,
        chunk_paths: &[PathBuf],
    ) -> ChunkingResult<bool>;
    
    /// Get detailed validation information
    async fn validate_with_details(
        &self,
        original_file: &Path,
        chunk_paths: &[PathBuf],
    ) -> ChunkingResult<ValidationDetails>;
}

/// Detailed validation results
#[derive(Debug, Clone)]
pub struct ValidationDetails {
    /// Whether validation passed
    pub valid: bool,
    /// Original file checksum
    pub original_checksum: String,
    /// Combined chunks checksum
    pub chunks_checksum: String,
    /// Size comparison
    pub size_match: bool,
    /// Individual chunk information
    pub chunk_details: Vec<ChunkValidationInfo>,
}

/// Information about individual chunk validation
#[derive(Debug, Clone)]
pub struct ChunkValidationInfo {
    /// Path to the chunk
    pub path: PathBuf,
    /// Size in bytes
    pub size: u64,
    /// Whether the chunk file exists and is readable
    pub accessible: bool,
    /// Checksum of the chunk
    pub checksum: String,
}