//! Error types for file chunking operations
//! 
//! Following structured error handling principles with thiserror

use std::path::PathBuf;
use thiserror::Error;

/// Result type for chunking operations
pub type ChunkingResult<T> = Result<T, ChunkingError>;

/// Structured error hierarchy for file chunking operations
#[derive(Error, Debug)]
pub enum ChunkingError {
    /// File system I/O errors
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Invalid input parameters
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    /// File validation errors
    #[error("File validation failed for {file}: {reason}")]
    ValidationFailed { file: PathBuf, reason: String },
    
    /// Chunking operation failed
    #[error("Chunking failed for {file}: {reason}")]
    ChunkingFailed { file: PathBuf, reason: String },
    
    /// External library errors
    #[error("External chunking library error: {0}")]
    ExternalLibrary(String),
    
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },
}

impl ChunkingError {
    /// Create an invalid input error
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }
    
    /// Create a validation failed error
    pub fn validation_failed(file: PathBuf, reason: impl Into<String>) -> Self {
        Self::ValidationFailed {
            file,
            reason: reason.into(),
        }
    }
    
    /// Create a chunking failed error
    pub fn chunking_failed(file: PathBuf, reason: impl Into<String>) -> Self {
        Self::ChunkingFailed {
            file,
            reason: reason.into(),
        }
    }
    
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }
}