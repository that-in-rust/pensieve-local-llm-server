pub mod classifier;
pub mod text_processor;
pub mod converter;
pub mod binary_processor;
pub mod pipeline;
pub mod streaming;
pub mod performance;

// Re-export main types and traits for convenience
pub use classifier::{FileClassifier};
pub use text_processor::{TextProcessor, TextProcessorConfig};
pub use converter::{Converter, ConverterConfig};
pub use binary_processor::{BinaryProcessor, BinaryProcessorConfig};
pub use pipeline::{ContentExtractionPipeline, PipelineConfig, ProcessingStats};
pub use streaming::{StreamingProcessor, StreamingConfig, StreamingProgress, StreamingStats};
pub use performance::{PerformanceMonitor, PerformanceConfig, PerformanceSnapshot, PerformanceStats, OptimizationRecommendation};

use serde::{Deserialize, Serialize};
use std::path::Path;

/// File type classification based on processing requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileType {
    /// Type 1: Direct text files that can be read as-is
    DirectText,
    /// Type 2: Files that can be converted to text via external commands
    Convertible,
    /// Type 3: Non-text files that cannot be meaningfully converted
    NonText,
}

impl FileType {
    /// Convert to database storage string
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::DirectText => "direct_text",
            FileType::Convertible => "convertible",
            FileType::NonText => "non_text",
        }
    }

    /// Parse from database storage string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "direct_text" => Some(FileType::DirectText),
            "convertible" => Some(FileType::Convertible),
            "non_text" => Some(FileType::NonText),
            _ => None,
        }
    }
}

/// Processed file data ready for database storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedFile {
    pub filepath: String,
    pub filename: String,
    pub extension: String,
    pub file_size_bytes: i64,
    pub line_count: Option<i32>,
    pub word_count: Option<i32>,
    pub token_count: Option<i32>,
    pub content_text: Option<String>,
    pub file_type: FileType,
    pub conversion_command: Option<String>,
    pub relative_path: String,
    pub absolute_path: String,
    pub skipped: bool,
    pub skip_reason: Option<String>,
}

/// Trait for processing different file types
#[async_trait::async_trait]
pub trait FileProcessor: Send + Sync {
    /// Check if this processor can handle the given file
    fn can_process(&self, file_path: &Path) -> bool;
    
    /// Process the file and return processed data
    async fn process(&self, file_path: &Path) -> crate::error::ProcessingResult<ProcessedFile>;
    
    /// Get the file type this processor handles
    fn get_file_type(&self) -> FileType;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_type_string_conversion() {
        assert_eq!(FileType::DirectText.as_str(), "direct_text");
        assert_eq!(FileType::Convertible.as_str(), "convertible");
        assert_eq!(FileType::NonText.as_str(), "non_text");

        assert_eq!(FileType::from_str("direct_text"), Some(FileType::DirectText));
        assert_eq!(FileType::from_str("convertible"), Some(FileType::Convertible));
        assert_eq!(FileType::from_str("non_text"), Some(FileType::NonText));
        assert_eq!(FileType::from_str("invalid"), None);
    }

    #[test]
    fn test_file_type_serialization() {
        let file_type = FileType::DirectText;
        let serialized = serde_json::to_string(&file_type).unwrap();
        let deserialized: FileType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(file_type, deserialized);
    }

    #[test]
    fn test_processed_file_creation() {
        let processed_file = ProcessedFile {
            filepath: "src/main.rs".to_string(),
            filename: "main.rs".to_string(),
            extension: "rs".to_string(),
            file_size_bytes: 1024,
            line_count: Some(50),
            word_count: Some(200),
            token_count: Some(180),
            content_text: Some("fn main() {}".to_string()),
            file_type: FileType::DirectText,
            conversion_command: None,
            relative_path: "src/main.rs".to_string(),
            absolute_path: "/home/user/project/src/main.rs".to_string(),
            skipped: false,
            skip_reason: None,
        };

        assert_eq!(processed_file.file_type, FileType::DirectText);
        assert_eq!(processed_file.extension, "rs");
        assert!(processed_file.content_text.is_some());
        assert!(processed_file.conversion_command.is_none());
        assert!(!processed_file.skipped);
    }
}
