//! Error types and handling for the Pensieve CLI tool

use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Main error type for the Pensieve application
pub type Result<T> = std::result::Result<T, PensieveError>;

/// Comprehensive error hierarchy for all failure modes
#[derive(Error, Debug)]
pub enum PensieveError {
    /// I/O related errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Database operation errors
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// File processing errors
    #[error("File processing error: {file_path} - {cause}")]
    FileProcessing { file_path: PathBuf, cause: String },

    /// External tool execution errors
    #[error("External tool error: {tool} - {message}")]
    ExternalTool { tool: String, message: String },

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Validation errors
    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },

    /// CLI argument parsing errors
    #[error("CLI argument error: {0}")]
    CliArgument(String),

    /// Directory traversal errors
    #[error("Directory traversal error: {path} - {cause}")]
    DirectoryTraversal { path: PathBuf, cause: String },

    /// Hash calculation errors
    #[error("Hash calculation error: {file_path} - {cause}")]
    HashCalculation { file_path: PathBuf, cause: String },

    /// Content extraction errors
    #[error("Content extraction error: {0}")]
    ContentExtraction(#[from] ExtractionError),

    /// Database migration errors
    #[error("Database migration error: {0}")]
    Migration(String),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid data format errors
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Specific errors for content extraction operations
#[derive(Error, Debug)]
pub enum ExtractionError {
    /// Unsupported file type
    #[error("Unsupported file type: {extension}")]
    UnsupportedType { extension: String },

    /// External tool not found
    #[error("External tool not found: {tool}")]
    ToolNotFound { tool: String },

    /// External tool execution timeout
    #[error("External tool timeout: {tool} after {timeout:?}")]
    ToolTimeout { tool: String, timeout: Duration },

    /// Content too large to process
    #[error("Content too large: {size} bytes (max: {max})")]
    ContentTooLarge { size: u64, max: u64 },

    /// Text encoding errors
    #[error("Encoding error: {0}")]
    Encoding(String),

    /// File format parsing errors
    #[error("Format parsing error: {format} - {cause}")]
    FormatParsing { format: String, cause: String },

    /// External tool execution failed
    #[error("Tool execution failed: {tool} - exit code {code}")]
    ToolExecutionFailed { tool: String, code: i32 },
}

/// Database-specific error types
#[derive(Error, Debug)]
pub enum DatabaseError {
    /// Connection establishment failed
    #[error("Database connection failed: {0}")]
    ConnectionFailed(String),

    /// Query execution failed
    #[error("Query execution failed: {query} - {cause}")]
    QueryFailed { query: String, cause: String },

    /// Transaction failed
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    /// Schema migration failed
    #[error("Schema migration failed: {version} - {cause}")]
    MigrationFailed { version: String, cause: String },

    /// Constraint violation
    #[error("Constraint violation: {constraint} - {details}")]
    ConstraintViolation { constraint: String, details: String },
}

/// File scanning specific errors
#[derive(Error, Debug)]
pub enum ScanError {
    /// Permission denied accessing path
    #[error("Permission denied: {path}")]
    PermissionDenied { path: PathBuf },

    /// Path does not exist
    #[error("Path not found: {path}")]
    PathNotFound { path: PathBuf },

    /// Invalid path format
    #[error("Invalid path: {path} - {reason}")]
    InvalidPath { path: PathBuf, reason: String },

    /// File metadata extraction failed
    #[error("Metadata extraction failed: {path} - {cause}")]
    MetadataFailed { path: PathBuf, cause: String },

    /// Directory traversal interrupted
    #[error("Traversal interrupted: {path}")]
    TraversalInterrupted { path: PathBuf },
}

impl From<DatabaseError> for PensieveError {
    fn from(err: DatabaseError) -> Self {
        match err {
            DatabaseError::ConnectionFailed(msg) => {
                PensieveError::Configuration(format!("Database connection: {}", msg))
            }
            DatabaseError::QueryFailed { query, cause } => {
                PensieveError::Configuration(format!("Query '{}' failed: {}", query, cause))
            }
            DatabaseError::TransactionFailed(msg) => {
                PensieveError::Configuration(format!("Transaction failed: {}", msg))
            }
            DatabaseError::MigrationFailed { version, cause } => {
                PensieveError::Migration(format!("Migration {} failed: {}", version, cause))
            }
            DatabaseError::ConstraintViolation { constraint, details } => {
                PensieveError::Validation {
                    field: constraint,
                    message: details,
                }
            }
        }
    }
}

impl From<ScanError> for PensieveError {
    fn from(err: ScanError) -> Self {
        match err {
            ScanError::PermissionDenied { path } => PensieveError::DirectoryTraversal {
                path,
                cause: "Permission denied".to_string(),
            },
            ScanError::PathNotFound { path } => PensieveError::DirectoryTraversal {
                path,
                cause: "Path not found".to_string(),
            },
            ScanError::InvalidPath { path, reason } => PensieveError::DirectoryTraversal {
                path,
                cause: reason,
            },
            ScanError::MetadataFailed { path, cause } => PensieveError::FileProcessing {
                file_path: path,
                cause,
            },
            ScanError::TraversalInterrupted { path } => PensieveError::DirectoryTraversal {
                path,
                cause: "Traversal interrupted".to_string(),
            },
        }
    }
}

/// Helper trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    /// Add context with file path
    fn with_file_context(self, path: &std::path::Path) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<PensieveError>,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let base_error = e.into();
            match base_error {
                PensieveError::Io(io_err) => {
                    PensieveError::Configuration(format!("{}: {}", f(), io_err))
                }
                other => other,
            }
        })
    }

    fn with_file_context(self, path: &std::path::Path) -> Result<T> {
        self.map_err(|e| {
            let base_error = e.into();
            match base_error {
                PensieveError::Io(io_err) => PensieveError::FileProcessing {
                    file_path: path.to_path_buf(),
                    cause: io_err.to_string(),
                },
                other => other,
            }
        })
    }
}