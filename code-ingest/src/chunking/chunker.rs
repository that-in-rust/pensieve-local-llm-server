//! Implementation of FileChunker using txt-sectumsempra
//! 
//! This module provides the concrete implementation of file chunking
//! using the txt-sectumsempra library.

use std::path::{Path, PathBuf};
use async_trait::async_trait;
use tokio::fs;
use tracing::{debug, info, warn, instrument};

use crate::chunking::{
    ChunkingResult, ChunkingError,
    traits::{FileChunker, ChunkingConfig, ChunkInfo},
    TSVECTOR_SAFE_LIMIT_BYTES,
};

/// File chunker implementation using txt-sectumsempra
/// 
/// This implementation follows the RAII pattern and provides
/// automatic resource management for chunked files.
#[derive(Debug, Clone)]
pub struct TxtSectumsempraChunker {
    /// Whether to enable detailed logging
    verbose: bool,
}

impl TxtSectumsempraChunker {
    /// Create a new chunker instance
    pub fn new() -> Self {
        Self {
            verbose: false,
        }
    }
    
    /// Create a new chunker with verbose logging
    pub fn with_verbose(verbose: bool) -> Self {
        Self {
            verbose,
        }
    }
    
    /// Convert txt-sectumsempra errors to our error type
    fn convert_error(file_path: &Path, error: txt_sectumsempra::ChunkError) -> ChunkingError {
        ChunkingError::chunking_failed(
            file_path.to_path_buf(),
            format!("txt-sectumsempra error: {}", error)
        )
    }
}

impl Default for TxtSectumsempraChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FileChunker for TxtSectumsempraChunker {
    #[instrument(skip(self), fields(file = %file_path.display()))]
    async fn chunk_file(
        &self,
        file_path: &Path,
        config: &ChunkingConfig,
    ) -> ChunkingResult<ChunkInfo> {
        // Validate preconditions
        config.validate()?;
        
        if !file_path.exists() {
            return Err(ChunkingError::invalid_input(
                format!("File does not exist: {}", file_path.display())
            ));
        }
        
        if !file_path.is_file() {
            return Err(ChunkingError::invalid_input(
                format!("Path is not a file: {}", file_path.display())
            ));
        }
        
        // Get file metadata
        let metadata = fs::metadata(file_path).await?;
        let file_size = metadata.len();
        
        debug!(
            "Chunking file: {} (size: {} bytes, chunk_size: {} MB)",
            file_path.display(),
            file_size,
            config.chunk_size_mb
        );
        
        // Check if chunking is actually needed
        if !self.should_chunk(file_path).await? {
            info!("File is small enough, no chunking needed");
            return Ok(ChunkInfo {
                original_file: file_path.to_path_buf(),
                chunk_paths: vec![file_path.to_path_buf()],
                chunk_size_bytes: file_size,
                original_size_bytes: file_size,
                validated: true, // No validation needed for single file
            });
        }
        
        // Perform chunking using txt-sectumsempra
        // Note: txt-sectumsempra is synchronous, so we run it in a blocking task
        let file_path_owned = file_path.to_path_buf();
        let chunk_size_mb = config.chunk_size_mb;
        
        let chunk_paths = tokio::task::spawn_blocking(move || {
            txt_sectumsempra::Chunker::split_file(&file_path_owned, chunk_size_mb)
        })
        .await
        .map_err(|e| ChunkingError::chunking_failed(
            file_path.to_path_buf(),
            format!("Task join error: {}", e)
        ))?
        .map_err(|e| Self::convert_error(file_path, e))?;
        
        info!("Created {} chunks for file {}", chunk_paths.len(), file_path.display());
        
        // Calculate chunk size in bytes
        let chunk_size_bytes = (config.chunk_size_mb * 1024.0 * 1024.0) as u64;
        
        let chunk_info = ChunkInfo {
            original_file: file_path.to_path_buf(),
            chunk_paths,
            chunk_size_bytes,
            original_size_bytes: file_size,
            validated: false, // Will be set by validator if validation is performed
        };
        
        Ok(chunk_info)
    }
    
    #[instrument(skip(self), fields(file = %file_path.display()))]
    async fn should_chunk(&self, file_path: &Path) -> ChunkingResult<bool> {
        if !file_path.exists() {
            return Err(ChunkingError::invalid_input(
                format!("File does not exist: {}", file_path.display())
            ));
        }
        
        let metadata = fs::metadata(file_path).await?;
        let file_size = metadata.len();
        
        let should_chunk = file_size > TSVECTOR_SAFE_LIMIT_BYTES;
        
        debug!(
            "File size check: {} bytes, threshold: {} bytes, should_chunk: {}",
            file_size,
            TSVECTOR_SAFE_LIMIT_BYTES,
            should_chunk
        );
        
        Ok(should_chunk)
    }
    
    #[instrument(skip(self), fields(file = %file_path.display()))]
    async fn estimate_chunks(
        &self,
        file_path: &Path,
        config: &ChunkingConfig,
    ) -> ChunkingResult<usize> {
        config.validate()?;
        
        if !self.should_chunk(file_path).await? {
            return Ok(1);
        }
        
        let metadata = fs::metadata(file_path).await?;
        let file_size = metadata.len();
        let chunk_size_bytes = (config.chunk_size_mb * 1024.0 * 1024.0) as u64;
        
        let estimated_chunks = ((file_size + chunk_size_bytes - 1) / chunk_size_bytes) as usize;
        
        debug!(
            "Estimated {} chunks for file {} (size: {} bytes, chunk_size: {} bytes)",
            estimated_chunks,
            file_path.display(),
            file_size,
            chunk_size_bytes
        );
        
        Ok(estimated_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio::io::AsyncWriteExt;
    
    #[tokio::test]
    async fn test_should_chunk_small_file() {
        let chunker = TxtSectumsempraChunker::new();
        
        // Create a small test file
        let mut temp_file = NamedTempFile::new().unwrap();
        let small_content = "a".repeat(1000); // 1KB
        temp_file.write_all(small_content.as_bytes()).unwrap();
        
        let should_chunk = chunker.should_chunk(temp_file.path()).await.unwrap();
        assert!(!should_chunk, "Small file should not be chunked");
    }
    
    #[tokio::test]
    async fn test_should_chunk_large_file() {
        let chunker = TxtSectumsempraChunker::new();
        
        // Create a large test file
        let mut temp_file = tokio::fs::File::create("test_large_file.txt").await.unwrap();
        let large_content = "a".repeat(1_000_000); // 1MB
        temp_file.write_all(large_content.as_bytes()).await.unwrap();
        temp_file.sync_all().await.unwrap();
        drop(temp_file);
        
        let should_chunk = chunker.should_chunk(Path::new("test_large_file.txt")).await.unwrap();
        assert!(should_chunk, "Large file should be chunked");
        
        // Cleanup
        tokio::fs::remove_file("test_large_file.txt").await.unwrap();
    }
    
    #[tokio::test]
    async fn test_estimate_chunks() {
        let chunker = TxtSectumsempraChunker::new();
        let config = ChunkingConfig::with_chunk_size(0.5).unwrap(); // 0.5MB chunks
        
        // Create a test file
        let mut temp_file = tokio::fs::File::create("test_estimate_file.txt").await.unwrap();
        let content = "a".repeat(1_000_000); // 1MB
        temp_file.write_all(content.as_bytes()).await.unwrap();
        temp_file.sync_all().await.unwrap();
        drop(temp_file);
        
        let estimated = chunker.estimate_chunks(Path::new("test_estimate_file.txt"), &config).await.unwrap();
        assert_eq!(estimated, 2, "1MB file with 0.5MB chunks should estimate 2 chunks");
        
        // Cleanup
        tokio::fs::remove_file("test_estimate_file.txt").await.unwrap();
    }
    
    #[tokio::test]
    async fn test_config_validation() {
        // Valid config
        let config = ChunkingConfig::with_chunk_size(1.0).unwrap();
        assert!(config.validate().is_ok());
        
        // Invalid config - too small
        let result = ChunkingConfig::with_chunk_size(0.0);
        assert!(result.is_err());
        
        // Invalid config - too large
        let result = ChunkingConfig::with_chunk_size(15.0);
        assert!(result.is_err());
    }
}