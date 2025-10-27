//! Core engine module for code ingestion
//! 
//! This module provides the main ingestion engine and core functionality
//! for processing repositories and local folders into PostgreSQL databases.

// Re-export the main ingestion engine and related types
pub use crate::ingestion::{
    IngestionEngine, IngestionConfig, IngestionOperationResult, SourceType,
    IngestionStatistics, IngestionRecord, IngestionStatus,
};

// Re-export git functionality for direct access
pub use crate::ingestion::git_cloner::{GitCloner, CloneConfig, CloneResult};

// Re-export batch processing functionality
pub use crate::ingestion::batch_processor::{BatchProcessor, BatchConfig, BatchStats, BatchProgress};

// Re-export folder processing functionality
pub use crate::ingestion::folder_processor::{FolderProcessor, FolderConfig, FolderResult};

/// Core engine facade that provides a simplified interface to the ingestion system
pub struct CoreEngine {
    ingestion_engine: IngestionEngine,
}

impl CoreEngine {
    /// Create a new core engine with default configuration
    pub fn new(
        database: std::sync::Arc<crate::database::Database>,
        file_processor: std::sync::Arc<dyn crate::processing::FileProcessor>,
    ) -> Self {
        let config = IngestionConfig::default();
        let ingestion_engine = IngestionEngine::new(config, database, file_processor);
        
        Self {
            ingestion_engine,
        }
    }
    
    /// Create a new core engine with custom configuration
    pub fn with_config(
        config: IngestionConfig,
        database: std::sync::Arc<crate::database::Database>,
        file_processor: std::sync::Arc<dyn crate::processing::FileProcessor>,
    ) -> Self {
        let ingestion_engine = IngestionEngine::new(config, database, file_processor);
        
        Self {
            ingestion_engine,
        }
    }
    
    /// Ingest from a source (Git repository or local folder)
    pub async fn ingest(
        &self,
        source: &str,
        progress_callback: Option<Box<dyn Fn(BatchProgress) + Send + Sync>>,
    ) -> crate::error::IngestionResult<IngestionOperationResult> {
        self.ingestion_engine.ingest_source(source, progress_callback).await
    }
    
    /// Get statistics for a completed ingestion
    pub async fn get_statistics(
        &self,
        ingestion_id: i64,
    ) -> crate::error::IngestionResult<IngestionStatistics> {
        self.ingestion_engine.get_ingestion_statistics(ingestion_id).await
    }
    
    /// List all ingestion records
    pub async fn list_ingestions(&self) -> crate::error::IngestionResult<Vec<IngestionRecord>> {
        self.ingestion_engine.list_ingestions().await
    }
    
    /// Request graceful shutdown
    pub fn shutdown(&self) {
        self.ingestion_engine.request_shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::{FileType, ProcessedFile};
    use crate::error::{ProcessingResult};
    use std::path::Path;

    // Mock file processor for testing
    struct MockFileProcessor;

    #[async_trait::async_trait]
    impl crate::processing::FileProcessor for MockFileProcessor {
        fn can_process(&self, _file_path: &Path) -> bool {
            true
        }

        async fn process(&self, file_path: &Path) -> ProcessingResult<ProcessedFile> {
            Ok(ProcessedFile {
                filepath: file_path.display().to_string(),
                filename: file_path.file_name().unwrap().to_str().unwrap().to_string(),
                extension: file_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_string(),
                file_size_bytes: 1024,
                line_count: Some(10),
                word_count: Some(50),
                token_count: Some(100),
                content_text: Some("Mock content".to_string()),
                file_type: FileType::DirectText,
                conversion_command: None,
                relative_path: file_path.display().to_string(),
                absolute_path: file_path.display().to_string(),
                skipped: false,
                skip_reason: None,
            })
        }

        fn get_file_type(&self) -> FileType {
            FileType::DirectText
        }
    }

    #[test]
    fn test_core_engine_creation() {
        // This test validates that the core engine can be created
        // In a real test, we would use a mock database
        let config = IngestionConfig::default();
        assert!(config.cleanup_cloned_repos);
        assert_eq!(config.max_ingestion_time, std::time::Duration::from_secs(1800));
    }

    #[test]
    fn test_re_exports() {
        // Test that all re-exports are accessible
        let _config = IngestionConfig::default();
        let _clone_config = CloneConfig::default();
        let _batch_config = BatchConfig::default();
        let _folder_config = FolderConfig::default();
        
        // Test enum variants
        let _source_type = SourceType::GitRepository;
        let _status = IngestionStatus::InProgress;
    }
}