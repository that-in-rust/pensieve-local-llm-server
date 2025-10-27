//! Chunk validation implementation using checksum verification
//! 
//! This module provides validation of chunked files to ensure data integrity.

use std::path::{Path, PathBuf};
use async_trait::async_trait;
use tokio::fs;
use tracing::{debug, instrument};
use sha2::{Sha256, Digest};

use crate::chunking::{
    ChunkingResult, ChunkingError,
    traits::{ChunkValidator, ValidationDetails, ChunkValidationInfo},
};

/// Checksum-based chunk validator
/// 
/// This validator uses SHA-256 checksums to verify that the concatenated
/// chunks exactly match the original file.
#[derive(Debug, Clone, Default)]
pub struct ChecksumValidator {
    /// Buffer size for reading files
    buffer_size: usize,
}

impl ChecksumValidator {
    /// Create a new validator with default buffer size
    pub fn new() -> Self {
        Self {
            buffer_size: 8192, // 8KB buffer
        }
    }
    
    /// Create a validator with custom buffer size
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        Self {
            buffer_size,
        }
    }
    
    /// Calculate SHA-256 checksum of a file
    async fn calculate_file_checksum(&self, file_path: &Path) -> ChunkingResult<String> {
        let mut file = fs::File::open(file_path).await?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; self.buffer_size];
        
        use tokio::io::AsyncReadExt;
        
        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Calculate combined checksum of multiple chunk files
    async fn calculate_chunks_checksum(&self, chunk_paths: &[PathBuf]) -> ChunkingResult<String> {
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; self.buffer_size];
        
        use tokio::io::AsyncReadExt;
        
        for chunk_path in chunk_paths {
            let mut file = fs::File::open(chunk_path).await
                .map_err(|e| ChunkingError::validation_failed(
                    chunk_path.clone(),
                    format!("Cannot open chunk file: {}", e)
                ))?;
            
            loop {
                let bytes_read = file.read(&mut buffer).await?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Get information about a single chunk
    async fn get_chunk_info(&self, chunk_path: &Path) -> ChunkValidationInfo {
        let accessible = chunk_path.exists() && chunk_path.is_file();
        
        let (size, checksum) = if accessible {
            match fs::metadata(chunk_path).await {
                Ok(metadata) => {
                    let size = metadata.len();
                    let checksum = self.calculate_file_checksum(chunk_path).await
                        .unwrap_or_else(|_| "error".to_string());
                    (size, checksum)
                }
                Err(_) => (0, "error".to_string()),
            }
        } else {
            (0, "inaccessible".to_string())
        };
        
        ChunkValidationInfo {
            path: chunk_path.to_path_buf(),
            size,
            accessible,
            checksum,
        }
    }
}

#[async_trait]
impl ChunkValidator for ChecksumValidator {
    #[instrument(skip(self), fields(original = %original_file.display(), chunks = chunk_paths.len()))]
    async fn validate_chunks(
        &self,
        original_file: &Path,
        chunk_paths: &[PathBuf],
    ) -> ChunkingResult<bool> {
        // Validate preconditions
        if !original_file.exists() {
            return Err(ChunkingError::validation_failed(
                original_file.to_path_buf(),
                "Original file does not exist"
            ));
        }
        
        if chunk_paths.is_empty() {
            return Err(ChunkingError::invalid_input("No chunk paths provided"));
        }
        
        // Check that all chunks exist
        for chunk_path in chunk_paths {
            if !chunk_path.exists() {
                return Err(ChunkingError::validation_failed(
                    chunk_path.clone(),
                    "Chunk file does not exist"
                ));
            }
        }
        
        debug!("Validating {} chunks against original file", chunk_paths.len());
        
        // Use txt-sectumsempra's built-in validation
        // This is more efficient than our manual checksum calculation
        let original_file_owned = original_file.to_path_buf();
        let chunk_paths_owned = chunk_paths.to_vec();
        
        let is_valid = tokio::task::spawn_blocking(move || {
            txt_sectumsempra::Chunker::validate(&original_file_owned, &chunk_paths_owned)
        })
        .await
        .map_err(|e| ChunkingError::validation_failed(
            original_file.to_path_buf(),
            format!("Task join error: {}", e)
        ))?
        .map_err(|e| ChunkingError::validation_failed(
            original_file.to_path_buf(),
            format!("txt-sectumsempra validation error: {}", e)
        ))?;
        
        debug!("Validation result: {}", is_valid);
        Ok(is_valid)
    }
    
    #[instrument(skip(self), fields(original = %original_file.display(), chunks = chunk_paths.len()))]
    async fn validate_with_details(
        &self,
        original_file: &Path,
        chunk_paths: &[PathBuf],
    ) -> ChunkingResult<ValidationDetails> {
        // Calculate checksums
        let original_checksum = self.calculate_file_checksum(original_file).await?;
        let chunks_checksum = self.calculate_chunks_checksum(chunk_paths).await?;
        
        // Check size match
        let original_size = fs::metadata(original_file).await?.len();
        let mut total_chunk_size = 0u64;
        
        let mut chunk_details = Vec::new();
        for chunk_path in chunk_paths {
            let chunk_info = self.get_chunk_info(chunk_path).await;
            total_chunk_size += chunk_info.size;
            chunk_details.push(chunk_info);
        }
        
        let size_match = original_size == total_chunk_size;
        let checksum_match = original_checksum == chunks_checksum;
        let valid = size_match && checksum_match;
        
        debug!(
            "Detailed validation: size_match={}, checksum_match={}, valid={}",
            size_match, checksum_match, valid
        );
        
        Ok(ValidationDetails {
            valid,
            original_checksum,
            chunks_checksum,
            size_match,
            chunk_details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio::io::AsyncWriteExt;
    
    #[tokio::test]
    async fn test_calculate_file_checksum() {
        let validator = ChecksumValidator::new();
        
        // Create a test file with known content
        let mut temp_file = tokio::fs::File::create("test_checksum.txt").await.unwrap();
        temp_file.write_all(b"hello world").await.unwrap();
        temp_file.sync_all().await.unwrap();
        drop(temp_file);
        
        let checksum = validator.calculate_file_checksum(Path::new("test_checksum.txt")).await.unwrap();
        
        // SHA-256 of "hello world"
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert_eq!(checksum, expected);
        
        // Cleanup
        tokio::fs::remove_file("test_checksum.txt").await.unwrap();
    }
    
    #[tokio::test]
    async fn test_validate_single_chunk() {
        let validator = ChecksumValidator::new();
        
        // Create a test file
        let mut temp_file = tokio::fs::File::create("test_single.txt").await.unwrap();
        temp_file.write_all(b"test content").await.unwrap();
        temp_file.sync_all().await.unwrap();
        drop(temp_file);
        
        // Validate against itself (single chunk case)
        let chunks = vec![PathBuf::from("test_single.txt")];
        let is_valid = validator.validate_chunks(Path::new("test_single.txt"), &chunks).await.unwrap();
        assert!(is_valid, "File should validate against itself");
        
        // Cleanup
        tokio::fs::remove_file("test_single.txt").await.unwrap();
    }
    
    #[tokio::test]
    async fn test_validation_details() {
        let validator = ChecksumValidator::new();
        
        // Create a test file
        let mut temp_file = tokio::fs::File::create("test_details.txt").await.unwrap();
        temp_file.write_all(b"detailed test").await.unwrap();
        temp_file.sync_all().await.unwrap();
        drop(temp_file);
        
        let chunks = vec![PathBuf::from("test_details.txt")];
        let details = validator.validate_with_details(Path::new("test_details.txt"), &chunks).await.unwrap();
        
        assert!(details.valid, "Validation should pass");
        assert!(details.size_match, "Sizes should match");
        assert_eq!(details.original_checksum, details.chunks_checksum, "Checksums should match");
        assert_eq!(details.chunk_details.len(), 1, "Should have one chunk detail");
        
        // Cleanup
        tokio::fs::remove_file("test_details.txt").await.unwrap();
    }
}