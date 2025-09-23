use thiserror::Error;
use std::path::PathBuf;

/// Comprehensive error handling for validation framework
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Pensieve binary not found at path: {path}")]
    PensieveBinaryNotFound { path: PathBuf },
    
    #[error("Target directory not accessible: {path} - {cause}")]
    DirectoryNotAccessible { path: PathBuf, cause: String },
    
    #[error("Pensieve process crashed: {exit_code} - {stderr}")]
    PensieveCrashed { exit_code: i32, stderr: String },
    
    #[error("Validation timeout after {seconds}s")]
    ValidationTimeout { seconds: u64 },
    
    #[error("Resource limit exceeded: {resource} - {limit}")]
    ResourceLimitExceeded { resource: String, limit: String },
    
    #[error("Report generation failed: {cause}")]
    ReportGenerationFailed { cause: String },
    
    #[error("File system error: {0}")]
    FileSystem(#[from] std::io::Error),
    
    #[error("Permission denied accessing: {path}")]
    PermissionDenied { path: PathBuf },
    
    #[error("Symlink chain too deep: {path} (max depth: {max_depth})")]
    SymlinkChainTooDeep { path: PathBuf, max_depth: usize },
    
    #[error("Invalid file path: {path}")]
    InvalidPath { path: PathBuf },
    
    #[error("File type detection failed: {path} - {cause}")]
    FileTypeDetectionFailed { path: PathBuf, cause: String },
    
    #[error("Configuration error: {field} - {message}")]
    ConfigurationError { field: String, message: String },
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Process monitoring error: {0}")]
    ProcessMonitoring(String),
    
    #[error("Analysis error: {0}")]
    Analysis(String),
}

pub type Result<T> = std::result::Result<T, ValidationError>;