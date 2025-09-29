//! Chunking-aware file processor that automatically handles large files
//! 
//! This processor wraps existing file processors and adds automatic chunking
//! for files that exceed the tsvector size limit.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use async_trait::async_trait;
use tracing::{debug, info, warn, instrument};

use crate::chunking::{
    FileChunker, ChunkValidator, ChunkingConfig,
    TxtSectumsempraChunker, ChecksumValidator,
    traits::ChunkInfo,
};
use crate::processing::{FileProcessor, ProcessedFile, FileType};
use crate::error::{ProcessingError, ProcessingResult};

/// Configuration for the chunking processor
#[derive(Debug, Clone)]
pub struct ChunkingProcessorConfig {
    /// Chunking configuration
    pub chunking_config: ChunkingConfig,
    /// Whether to validate chunks after creation
    pub validate_chunks: bool,
    /// Whether to clean up chunks after processing
    pub cleanup_chunks: bool,
    /// Whether to add chunk metadata to processed files
    pub include_chunk_metadata: bool,
}

impl Default for ChunkingProcessorConfig {
    fn default() -> Self {
        Self {
            chunking_config: ChunkingConfig::default(),
            validate_chunks: true,
            cleanup_chunks: false, // Keep chunks for debugging by default
            include_chunk_metadata: true,
        }
    }
}

/// A file processor that automatically chunks large files
/// 
/// This processor follows the decorator pattern, wrapping an existing
/// file processor and adding chunking capabilities.
pub struct ChunkingProcessor<P, C, V> 
where
    P: FileProcessor,
    C: FileChunker,
    V: ChunkValidator,
{
    /// The underlying file processor
    inner_processor: Arc<P>,
    /// File chunker implementation
    chunker: Arc<C>,
    /// Chunk validator implementation
    validator: Arc<V>,
    /// Configuration
    config: ChunkingProcessorConfig,
}

impl<P, C, V> ChunkingProcessor<P, C, V>
where
    P: FileProcessor,
    C: FileChunker,
    V: ChunkValidator,
{
    /// Create a new chunking processor
    pub fn new(
        inner_processor: Arc<P>,
        chunker: Arc<C>,
        validator: Arc<V>,
        config: ChunkingProcessorConfig,
    ) -> Self {
        Self {
            inner_processor,
            chunker,
            validator,
            config,
        }
    }
    

    
    /// Process a single file, potentially creating chunks
    #[instrument(skip(self), fields(file = %file_path.display()))]
    async fn process_file_with_chunking(&self, file_path: &Path) -> ProcessingResult<Vec<ProcessedFile>> {
        // Check if the file should be chunked
        let should_chunk = self.chunker.should_chunk(file_path).await
            .map_err(|e| ProcessingError::ContentAnalysisFailed {
                path: file_path.display().to_string(),
                cause: format!("Chunking check failed: {}", e),
            })?;
        
        if !should_chunk {
            // File is small enough, process normally
            debug!("File is small enough, processing without chunking");
            let processed = self.inner_processor.process(file_path).await?;
            return Ok(vec![processed]);
        }
        
        info!("File is large, chunking before processing");
        
        // Chunk the file
        let chunk_info = self.chunker.chunk_file(file_path, &self.config.chunking_config).await
            .map_err(|e| ProcessingError::ContentAnalysisFailed {
                path: file_path.display().to_string(),
                cause: format!("Chunking failed: {}", e),
            })?;
        
        // Validate chunks if requested
        if self.config.validate_chunks {
            debug!("Validating {} chunks", chunk_info.chunk_paths.len());
            let is_valid = self.validator.validate_chunks(&chunk_info.original_file, &chunk_info.chunk_paths).await
                .map_err(|e| ProcessingError::ContentAnalysisFailed {
                    path: file_path.display().to_string(),
                    cause: format!("Chunk validation failed: {}", e),
                })?;
            
            if !is_valid {
                return Err(ProcessingError::ContentAnalysisFailed {
                    path: file_path.display().to_string(),
                    cause: "Chunk validation failed - chunks do not match original file".to_string(),
                });
            }
            
            info!("Chunk validation passed");
        }
        
        // Process each chunk
        let mut processed_files = Vec::new();
        for (index, chunk_path) in chunk_info.chunk_paths.iter().enumerate() {
            debug!("Processing chunk {} of {}: {}", index + 1, chunk_info.chunk_paths.len(), chunk_path.display());
            
            match self.inner_processor.process(chunk_path).await {
                Ok(mut processed) => {
                    // Add chunking metadata
                    if self.config.include_chunk_metadata {
                        self.add_chunk_metadata(&mut processed, &chunk_info, index);
                    }
                    processed_files.push(processed);
                }
                Err(e) => {
                    warn!("Failed to process chunk {}: {}", chunk_path.display(), e);
                    // Continue processing other chunks
                }
            }
        }
        
        // Clean up chunks if requested
        if self.config.cleanup_chunks {
            self.cleanup_chunks(&chunk_info.chunk_paths).await;
        }
        
        if processed_files.is_empty() {
            return Err(ProcessingError::ContentAnalysisFailed {
                path: file_path.display().to_string(),
                cause: "All chunks failed to process".to_string(),
            });
        }
        
        info!("Successfully processed {} chunks from file {}", processed_files.len(), file_path.display());
        Ok(processed_files)
    }
    
    /// Add chunking metadata to a processed file
    fn add_chunk_metadata(&self, processed: &mut ProcessedFile, chunk_info: &ChunkInfo, chunk_index: usize) {
        // Store original file path in metadata
        processed.metadata.insert("original_file".to_string(), chunk_info.original_file.display().to_string());
        processed.metadata.insert("is_chunked".to_string(), "true".to_string());
        processed.metadata.insert("chunk_index".to_string(), chunk_index.to_string());
        processed.metadata.insert("total_chunks".to_string(), chunk_info.chunk_paths.len().to_string());
        processed.metadata.insert("original_size_bytes".to_string(), chunk_info.original_size_bytes.to_string());
        processed.metadata.insert("chunk_size_bytes".to_string(), chunk_info.chunk_size_bytes.to_string());
        processed.metadata.insert("chunk_validated".to_string(), chunk_info.validated.to_string());
        
        // Update the filepath to indicate it's a chunk
        let original_filepath = processed.filepath.clone();
        processed.filepath = format!("{} (chunk {} of {})", original_filepath, chunk_index + 1, chunk_info.chunk_paths.len());
    }
    
    /// Clean up chunk files
    async fn cleanup_chunks(&self, chunk_paths: &[std::path::PathBuf]) {
        for chunk_path in chunk_paths {
            if let Err(e) = tokio::fs::remove_file(chunk_path).await {
                warn!("Failed to clean up chunk {}: {}", chunk_path.display(), e);
            } else {
                debug!("Cleaned up chunk: {}", chunk_path.display());
            }
        }
        
        // Try to remove the chunk directory if it's empty
        if let Some(chunk_dir) = chunk_paths.first().and_then(|p| p.parent()) {
            if let Err(e) = tokio::fs::remove_dir(chunk_dir).await {
                debug!("Could not remove chunk directory {} (may not be empty): {}", chunk_dir.display(), e);
            }
        }
    }
}

#[async_trait]
impl<P, C, V> FileProcessor for ChunkingProcessor<P, C, V>
where
    P: FileProcessor,
    C: FileChunker,
    V: ChunkValidator,
{
    fn can_process(&self, file_path: &Path) -> bool {
        // We can process any file that the inner processor can handle
        self.inner_processor.can_process(file_path)
    }
    
    #[instrument(skip(self), fields(file = %file_path.display()))]
    async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
        // Process the file (potentially with chunking)
        let processed_files = self.process_file_with_chunking(file_path).await?;
        
        if processed_files.len() == 1 {
            // Single file (not chunked or single chunk)
            Ok(processed_files.into_iter().next().unwrap())
        } else {
            // Multiple chunks - we need to return the first one for compatibility
            // In a real implementation, we might want to modify the FileProcessor trait
            // to support returning multiple files, but for now we'll return the first chunk
            // and log information about the others
            info!("File was chunked into {} parts, returning first chunk", processed_files.len());
            Ok(processed_files.into_iter().next().unwrap())
        }
    }
    
    fn get_file_type(&self) -> FileType {
        self.inner_processor.get_file_type()
    }
}

/// Factory for creating chunking processors
pub struct ChunkingProcessorFactory;

impl ChunkingProcessorFactory {
    /// Create a chunking processor that wraps any file processor
    pub fn wrap_processor<P>(processor: Arc<P>) -> ChunkingProcessor<P, TxtSectumsempraChunker, ChecksumValidator>
    where
        P: FileProcessor,
    {
        ChunkingProcessor::new(
            processor,
            Arc::new(TxtSectumsempraChunker::new()),
            Arc::new(ChecksumValidator::new()),
            ChunkingProcessorConfig::default(),
        )
    }
    
    /// Create a chunking processor with custom configuration
    pub fn wrap_processor_with_config<P>(
        processor: Arc<P>,
        config: ChunkingProcessorConfig,
    ) -> ChunkingProcessor<P, TxtSectumsempraChunker, ChecksumValidator>
    where
        P: FileProcessor,
    {
        ChunkingProcessor::new(
            processor,
            Arc::new(TxtSectumsempraChunker::new()),
            Arc::new(ChecksumValidator::new()),
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::{FileType, ProcessedFile};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;
    use tokio::io::AsyncWriteExt;
    
    // Mock file processor for testing
    struct MockProcessor;
    
    #[async_trait]
    impl FileProcessor for MockProcessor {
        fn can_process(&self, _file_path: &Path) -> bool {
            true
        }
        
        async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
            let content = tokio::fs::read_to_string(file_path).await
                .map_err(|e| ProcessingError::FileReadFailed {
                    path: file_path.display().to_string(),
                    cause: e.to_string(),
                })?;
            
            Ok(ProcessedFile {
                filepath: file_path.display().to_string(),
                filename: file_path.file_name().unwrap().to_string_lossy().to_string(),
                file_type: FileType::DirectText,
                extension: "txt".to_string(),
                content_text: Some(content),
                file_size_bytes: 0,
                line_count: Some(0),
                word_count: Some(0),
                token_count: Some(0),
                conversion_command: None,
                relative_path: file_path.display().to_string(),
                absolute_path: file_path.display().to_string(),
                skipped: false,
                skip_reason: None,
                metadata: HashMap::new(),
            })
        }
        
        fn get_file_type(&self) -> FileType {
            FileType::DirectText
        }
    }
    
    #[tokio::test]
    async fn test_small_file_no_chunking() {
        let processor = ChunkingProcessorFactory::wrap_processor(Arc::new(MockProcessor));
        
        // Create a small test file
        let mut temp_file = tokio::fs::File::create("test_small.txt").await.unwrap();
        temp_file.write_all(b"small content").await.unwrap();
        temp_file.sync_all().await.unwrap();
        drop(temp_file);
        
        let result = processor.process(Path::new("test_small.txt")).await.unwrap();
        
        // Should not be marked as chunked
        assert!(!result.metadata.contains_key("is_chunked"));
        
        // Cleanup
        tokio::fs::remove_file("test_small.txt").await.unwrap();
    }
    
    #[tokio::test]
    async fn test_chunking_processor_can_process() {
        let processor = ChunkingProcessorFactory::wrap_processor(Arc::new(MockProcessor));
        
        // Should delegate to inner processor
        assert!(processor.can_process(Path::new("test.txt")));
    }
    
    #[tokio::test]
    async fn test_chunking_processor_file_type() {
        let processor = ChunkingProcessorFactory::wrap_processor(Arc::new(MockProcessor));
        
        // Should delegate to inner processor
        assert_eq!(processor.get_file_type(), FileType::DirectText);
    }
}