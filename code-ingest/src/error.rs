use thiserror::Error;
use tracing::{error, warn, debug};

/// Main error type for the code ingestion system with actionable error messages
#[derive(Error, Debug)]
pub enum CodeIngestError {
    #[error("Git operation failed: {message}\n💡 Suggestion: {suggestion}")]
    Git { 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Database operation failed: {message}\n💡 Suggestion: {suggestion}")]
    Database { 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("File system operation failed: {message}\n💡 Suggestion: {suggestion}")]
    FileSystem { 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("File processing failed: {file_path} - {message}\n💡 Suggestion: {suggestion}")]
    FileProcessing { 
        file_path: String, 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Content conversion failed: {tool} - {message}\n💡 Suggestion: {suggestion}")]
    ConversionFailed { 
        tool: String, 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Configuration error: {message}\n💡 Suggestion: {suggestion}")]
    Configuration { 
        message: String, 
        suggestion: String,
    },

    #[error("GitHub authentication failed: {message}\n💡 Suggestion: {suggestion}")]
    Authentication { 
        message: String, 
        suggestion: String,
    },

    #[error("Repository not accessible: {url}\n💡 Suggestion: {suggestion}")]
    RepositoryNotFound { 
        url: String, 
        suggestion: String,
    },

    #[error("PostgreSQL connection failed: {message}\n💡 Suggestion: {suggestion}")]
    DatabaseConnection { 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Network operation failed: {message}\n💡 Suggestion: {suggestion}")]
    Network { 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Validation failed: {field} - {message}\n💡 Suggestion: {suggestion}")]
    Validation { 
        field: String, 
        message: String, 
        suggestion: String,
    },

    #[error("Resource limit exceeded: {resource} - {message}\n💡 Suggestion: {suggestion}")]
    ResourceLimit { 
        resource: String, 
        message: String, 
        suggestion: String,
    },

    #[error("Task generation failed: {message}\n💡 Suggestion: {suggestion}")]
    TaskGeneration { 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Query execution failed: {query} - {message}\n💡 Suggestion: {suggestion}")]
    QueryExecution { 
        query: String, 
        message: String, 
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Permission denied: {path}\n💡 Suggestion: {suggestion}")]
    PermissionDenied { 
        path: String, 
        suggestion: String,
    },

    #[error("Timeout occurred: {operation} took longer than {timeout_seconds}s\n💡 Suggestion: {suggestion}")]
    Timeout { 
        operation: String, 
        timeout_seconds: u64, 
        suggestion: String,
    },
}

/// Legacy system error for backward compatibility
#[derive(Error, Debug)]
pub enum SystemError {
    #[error("Ingestion error: {0}")]
    Ingestion(#[from] IngestionError),

    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),

    #[error("Processing error: {0}")]
    Processing(#[from] ProcessingError),

    #[error("Task generation error: {0}")]
    TaskGeneration(#[from] TaskError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Code ingest error: {0}")]
    CodeIngest(#[from] CodeIngestError),
}

/// Errors related to repository ingestion and file discovery
#[derive(Error, Debug, Clone)]
pub enum IngestionError {
    #[error("Git clone failed: {repo_url} - {cause}")]
    GitCloneFailed { repo_url: String, cause: String },

    #[error("Local path not found: {path}")]
    LocalPathNotFound { path: String },

    #[error("Permission denied: {path}")]
    PermissionDenied { path: String },

    #[error("Repository too large: {size_mb}MB exceeds limit")]
    RepositoryTooLarge { size_mb: u64 },

    #[error("Invalid repository URL: {url}")]
    InvalidRepositoryUrl { url: String },

    #[error("Authentication failed for repository: {repo_url}")]
    AuthenticationFailed { repo_url: String },

    #[error("Network error during clone: {cause}")]
    NetworkError { cause: String },

    #[error("File system error: {path} - {cause}")]
    FileSystemError { path: String, cause: String },

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Database error: {cause}")]
    DatabaseError { cause: String },

    #[error("Git error: {cause}")]
    GitError { cause: String },

    #[error("Processing error: {cause}")]
    ProcessingError { cause: String },

    #[error("Authentication error: {message}")]
    AuthenticationError { message: String },
}

/// Errors related to database operations
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {url} - {cause}")]
    ConnectionFailed { url: String, cause: String },

    #[error("Table creation failed: {table_name} - {cause}")]
    TableCreationFailed { table_name: String, cause: String },

    #[error("Query execution failed: {query} - {cause}")]
    QueryFailed { query: String, cause: String },

    #[error("Transaction failed: {cause}")]
    TransactionFailed { cause: String },

    #[error("Migration failed: {cause}")]
    MigrationFailed { cause: String },

    #[error("Database not found: {path}")]
    DatabaseNotFound { path: String },

    #[error("Invalid database URL: {url}")]
    InvalidDatabaseUrl { url: String },

    #[error("Batch insertion failed: {cause}")]
    BatchInsertionFailed { cause: String },
}

/// Errors related to file processing and content extraction
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("File read failed: {path} - {cause}")]
    FileReadFailed { path: String, cause: String },

    #[error("Encoding detection failed: {path}")]
    EncodingDetectionFailed { path: String },

    #[error("Content conversion failed: {path} - {cause}")]
    ContentConversionFailed { path: String, cause: String },

    #[error("External command failed: {command} - {cause}")]
    ExternalCommandFailed { command: String, cause: String },

    #[error("File too large: {path} - {size_mb}MB exceeds limit")]
    FileTooLarge { path: String, size_mb: u64 },

    #[error("Unsupported file type: {path} - {extension}")]
    UnsupportedFileType { path: String, extension: String },

    #[error("Content analysis failed: {path} - {cause}")]
    ContentAnalysisFailed { path: String, cause: String },
}

/// Errors related to task generation and markdown creation
#[derive(Error, Debug)]
pub enum TaskError {
    #[error("Task division failed: {total_tasks} tasks cannot be divided into {groups} groups")]
    TaskDivisionFailed { total_tasks: usize, groups: usize },

    #[error("Markdown generation failed: {cause}")]
    MarkdownGenerationFailed { cause: String },

    #[error("Task file creation failed: {path} - {cause}")]
    TaskFileCreationFailed { path: String, cause: String },

    #[error("Invalid task configuration: {cause}")]
    InvalidTaskConfiguration { cause: String },

    #[error("Query result processing failed: {cause}")]
    QueryResultProcessingFailed { cause: String },

    #[error("Chunk analysis failed: {cause}")]
    ChunkAnalysisFailed { cause: String },
}

/// Result type aliases for common operations
pub type CodeIngestResult<T> = Result<T, CodeIngestError>;
pub type SystemResult<T> = Result<T, SystemError>;
pub type IngestionResult<T> = Result<T, IngestionError>;
pub type DatabaseResult<T> = Result<T, DatabaseError>;
pub type ProcessingResult<T> = Result<T, ProcessingError>;
pub type TaskResult<T> = Result<T, TaskError>;

/// Error recovery strategies and helper functions
impl CodeIngestError {
    /// Create a Git error with actionable suggestions
    pub fn git_error(message: impl Into<String>, source: Option<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        let message = message.into();
        let suggestion = match message.as_str() {
            msg if msg.contains("authentication") || msg.contains("401") => {
                "Set your GitHub token: export GITHUB_TOKEN=your_token_here or use --token flag".to_string()
            },
            msg if msg.contains("not found") || msg.contains("404") => {
                "Check the repository URL and ensure it exists and is accessible".to_string()
            },
            msg if msg.contains("network") || msg.contains("timeout") => {
                "Check your internet connection and try again. Use --retry flag for automatic retries".to_string()
            },
            msg if msg.contains("permission") || msg.contains("403") => {
                "Ensure you have access to this repository. For private repos, set GITHUB_TOKEN".to_string()
            },
            _ => "Check the repository URL and your network connection".to_string(),
        };
        
        error!("Git operation failed: {}", message);
        debug!("Git error source: {:?}", source);
        
        Self::Git { message, suggestion, source }
    }

    /// Create a database error with actionable suggestions
    pub fn database_error(message: impl Into<String>, source: Option<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        let message = message.into();
        let suggestion = match message.as_str() {
            msg if msg.contains("connection") || msg.contains("connect") => {
                "Start PostgreSQL: 'brew services start postgresql' or 'sudo systemctl start postgresql'".to_string()
            },
            msg if msg.contains("database") && msg.contains("does not exist") => {
                "Create the database: 'createdb your_database_name' or run 'code-ingest pg-start' for setup help".to_string()
            },
            msg if msg.contains("permission") || msg.contains("authentication") => {
                "Check your database credentials and permissions in the connection string".to_string()
            },
            msg if msg.contains("syntax") => {
                "Check your SQL query syntax. Use 'code-ingest sql --help' for examples".to_string()
            },
            msg if msg.contains("table") && msg.contains("does not exist") => {
                "Run ingestion first to create tables, or check table name with 'code-ingest list-tables'".to_string()
            },
            _ => "Check your database connection and try 'code-ingest pg-start' for setup help".to_string(),
        };
        
        error!("Database operation failed: {}", message);
        debug!("Database error source: {:?}", source);
        
        Self::Database { message, suggestion, source }
    }

    /// Create a file system error with actionable suggestions
    pub fn file_system_error(message: impl Into<String>, source: Option<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        let message = message.into();
        let suggestion = match message.as_str() {
            msg if msg.contains("permission") || msg.contains("denied") => {
                "Check file permissions: 'ls -la path' and ensure you have read/write access".to_string()
            },
            msg if msg.contains("not found") || msg.contains("No such file") => {
                "Verify the path exists and is spelled correctly".to_string()
            },
            msg if msg.contains("space") || msg.contains("disk full") => {
                "Free up disk space or choose a different location with more available space".to_string()
            },
            msg if msg.contains("too many") => {
                "Close some files or increase system file descriptor limits".to_string()
            },
            _ => "Check the file path and your system permissions".to_string(),
        };
        
        error!("File system operation failed: {}", message);
        debug!("File system error source: {:?}", source);
        
        Self::FileSystem { message, suggestion, source }
    }

    /// Create a file processing error with actionable suggestions
    pub fn file_processing_error(
        file_path: impl Into<String>, 
        message: impl Into<String>, 
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let file_path = file_path.into();
        let message = message.into();
        let suggestion = match message.as_str() {
            msg if msg.contains("encoding") => {
                "File may have unusual encoding. Try converting to UTF-8 first".to_string()
            },
            msg if msg.contains("too large") => {
                "File exceeds size limit. Use --max-file-size flag to increase limit or exclude large files".to_string()
            },
            msg if msg.contains("binary") => {
                "Binary files are stored as metadata only. This is expected behavior".to_string()
            },
            msg if msg.contains("conversion") => {
                "Install required tools: 'brew install pandoc poppler-utils' or check tool availability".to_string()
            },
            _ => "Check if the file is readable and not corrupted".to_string(),
        };
        
        warn!("File processing failed for {}: {}", file_path, message);
        debug!("File processing error source: {:?}", source);
        
        Self::FileProcessing { file_path, message, suggestion, source }
    }

    /// Create a conversion error with actionable suggestions
    pub fn conversion_error(
        tool: impl Into<String>, 
        message: impl Into<String>, 
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let tool = tool.into();
        let message = message.into();
        let suggestion = match tool.as_str() {
            "pdftotext" => "Install poppler-utils: 'brew install poppler' or 'sudo apt-get install poppler-utils'".to_string(),
            "pandoc" => "Install pandoc: 'brew install pandoc' or visit https://pandoc.org/installing.html".to_string(),
            "xlsx2csv" => "Install xlsx2csv: 'pip install xlsx2csv' or use alternative Excel processing".to_string(),
            _ => format!("Install or check the '{}' tool and ensure it's in your PATH", tool),
        };
        
        error!("Conversion failed with {}: {}", tool, message);
        debug!("Conversion error source: {:?}", source);
        
        Self::ConversionFailed { tool, message, suggestion, source }
    }

    /// Create a configuration error with actionable suggestions
    pub fn configuration_error(message: impl Into<String>) -> Self {
        let message = message.into();
        let suggestion = match message.as_str() {
            msg if msg.contains("database") => {
                "Check your database connection string format: postgresql://user:pass@host:port/db".to_string()
            },
            msg if msg.contains("path") => {
                "Ensure all paths are absolute and directories exist".to_string()
            },
            msg if msg.contains("token") => {
                "Set GITHUB_TOKEN environment variable or use --token flag".to_string()
            },
            _ => "Check your configuration values and command line arguments".to_string(),
        };
        
        error!("Configuration error: {}", message);
        
        Self::Configuration { message, suggestion }
    }

    /// Check if this error is recoverable with retry
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::Timeout { .. } => true,
            Self::Database { message, .. } if message.contains("connection") => true,
            Self::Git { message, .. } if message.contains("network") || message.contains("timeout") => true,
            _ => false,
        }
    }

    /// Get suggested retry delay in seconds
    pub fn retry_delay_seconds(&self) -> u64 {
        match self {
            Self::Network { .. } => 5,
            Self::Timeout { .. } => 10,
            Self::Database { .. } => 3,
            Self::Git { .. } => 5,
            _ => 1,
        }
    }

    /// Log error with appropriate level
    pub fn log_error(&self) {
        match self {
            Self::FileProcessing { .. } => warn!("{}", self),
            Self::ConversionFailed { .. } => warn!("{}", self),
            Self::Validation { .. } => warn!("{}", self),
            _ => error!("{}", self),
        }
    }
}

/// Convert from external error types to CodeIngestError
impl From<sqlx::Error> for CodeIngestError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::Database(ref db_err) => {
                CodeIngestError::database_error(
                    format!("Database query failed: {}", db_err),
                    None,
                )
            },
            sqlx::Error::Configuration(ref config_err) => {
                CodeIngestError::database_error(
                    format!("Database configuration error: {}", config_err),
                    None,
                )
            },
            sqlx::Error::Io(ref io_err) => {
                CodeIngestError::database_error(
                    format!("Database connection I/O error: {}", io_err),
                    None,
                )
            },
            _ => CodeIngestError::database_error(
                format!("Database operation failed: {}", err),
                None,
            ),
        }
    }
}

impl From<git2::Error> for CodeIngestError {
    fn from(err: git2::Error) -> Self {
        match err.class() {
            git2::ErrorClass::Net => {
                CodeIngestError::Network {
                    message: format!("Git network error: {}", err.message()),
                    suggestion: "Check your internet connection and try again".to_string(),
                    source: None,
                }
            },
            git2::ErrorClass::Http => {
                CodeIngestError::git_error(
                    format!("Git HTTP authentication failed: {}", err.message()),
                    None,
                )
            },
            _ => CodeIngestError::git_error(
                format!("Git operation failed: {}", err.message()),
                None,
            ),
        }
    }
}

impl From<std::io::Error> for CodeIngestError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::NotFound => {
                CodeIngestError::file_system_error(
                    format!("File or directory not found: {}", err),
                    None,
                )
            },
            std::io::ErrorKind::PermissionDenied => {
                CodeIngestError::PermissionDenied {
                    path: "unknown".to_string(),
                    suggestion: "Check file permissions and ensure you have read/write access".to_string(),
                }
            },
            std::io::ErrorKind::TimedOut => {
                CodeIngestError::Timeout {
                    operation: "I/O operation".to_string(),
                    timeout_seconds: 30,
                    suggestion: "Try again or increase timeout limits".to_string(),
                }
            },
            _ => CodeIngestError::file_system_error(
                format!("I/O error: {}", err),
                None,
            ),
        }
    }
}

impl From<walkdir::Error> for CodeIngestError {
    fn from(err: walkdir::Error) -> Self {
        let path = err.path().map_or("unknown".to_string(), |p| p.display().to_string());
        
        CodeIngestError::file_processing_error(
            path,
            format!("Directory traversal error: {}", err),
            None,
        )
    }
}

/// Legacy conversions for backward compatibility
impl From<sqlx::Error> for DatabaseError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::Database(db_err) => DatabaseError::QueryFailed {
                query: "unknown".to_string(),
                cause: db_err.to_string(),
            },
            sqlx::Error::Configuration(config_err) => DatabaseError::InvalidDatabaseUrl {
                url: config_err.to_string(),
            },
            sqlx::Error::Io(io_err) => DatabaseError::ConnectionFailed {
                url: "unknown".to_string(),
                cause: io_err.to_string(),
            },
            _ => DatabaseError::QueryFailed {
                query: "unknown".to_string(),
                cause: err.to_string(),
            },
        }
    }
}

impl From<git2::Error> for IngestionError {
    fn from(err: git2::Error) -> Self {
        match err.class() {
            git2::ErrorClass::Net => IngestionError::NetworkError {
                cause: err.message().to_string(),
            },
            git2::ErrorClass::Http => IngestionError::AuthenticationFailed {
                repo_url: "unknown".to_string(),
            },
            _ => IngestionError::GitCloneFailed {
                repo_url: "unknown".to_string(),
                cause: err.message().to_string(),
            },
        }
    }
}

impl From<walkdir::Error> for ProcessingError {
    fn from(err: walkdir::Error) -> Self {
        ProcessingError::FileReadFailed {
            path: err.path().map_or("unknown".to_string(), |p| p.display().to_string()),
            cause: err.to_string(),
        }
    }
}

impl From<DatabaseError> for IngestionError {
    fn from(err: DatabaseError) -> Self {
        match err {
            DatabaseError::ConnectionFailed { url, cause } => IngestionError::NetworkError { cause },
            DatabaseError::QueryFailed { query: _, cause } => IngestionError::NetworkError { cause },
            _ => IngestionError::NetworkError { 
                cause: format!("Database error: {}", err) 
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_code_ingest_error_git_suggestions() {
        let auth_error = CodeIngestError::git_error("authentication failed", None);
        assert!(auth_error.to_string().contains("Set your GitHub token"));
        
        let not_found_error = CodeIngestError::git_error("repository not found", None);
        assert!(not_found_error.to_string().contains("Check the repository URL"));
        
        let network_error = CodeIngestError::git_error("network timeout", None);
        assert!(network_error.to_string().contains("Check your internet connection"));
    }

    #[test]
    fn test_code_ingest_error_database_suggestions() {
        let connection_error = CodeIngestError::database_error("connection refused", None);
        assert!(connection_error.to_string().contains("Start PostgreSQL"));
        
        let db_not_exist_error = CodeIngestError::database_error("database does not exist", None);
        assert!(db_not_exist_error.to_string().contains("Create the database"));
        
        let syntax_error = CodeIngestError::database_error("syntax error", None);
        assert!(syntax_error.to_string().contains("Check your SQL query syntax"));
    }

    #[test]
    fn test_code_ingest_error_file_system_suggestions() {
        let permission_error = CodeIngestError::file_system_error("permission denied", None);
        assert!(permission_error.to_string().contains("Check file permissions"));
        
        let not_found_error = CodeIngestError::file_system_error("file not found", None);
        assert!(not_found_error.to_string().contains("Verify the path exists"));
        
        let disk_full_error = CodeIngestError::file_system_error("no space left on device", None);
        assert!(disk_full_error.to_string().contains("Free up disk space"));
    }

    #[test]
    fn test_code_ingest_error_file_processing_suggestions() {
        let encoding_error = CodeIngestError::file_processing_error(
            "/path/file.txt", 
            "encoding error", 
            None
        );
        assert!(encoding_error.to_string().contains("unusual encoding"));
        
        let size_error = CodeIngestError::file_processing_error(
            "/path/large.txt", 
            "file too large", 
            None
        );
        assert!(size_error.to_string().contains("--max-file-size"));
        
        let binary_error = CodeIngestError::file_processing_error(
            "/path/file.bin", 
            "binary file", 
            None
        );
        assert!(binary_error.to_string().contains("metadata only"));
    }

    #[test]
    fn test_code_ingest_error_conversion_suggestions() {
        let pdf_error = CodeIngestError::conversion_error("pdftotext", "command not found", None);
        assert!(pdf_error.to_string().contains("Install poppler-utils"));
        
        let pandoc_error = CodeIngestError::conversion_error("pandoc", "not found", None);
        assert!(pandoc_error.to_string().contains("Install pandoc"));
        
        let xlsx_error = CodeIngestError::conversion_error("xlsx2csv", "missing", None);
        assert!(xlsx_error.to_string().contains("pip install xlsx2csv"));
    }

    #[test]
    fn test_code_ingest_error_configuration_suggestions() {
        let db_config_error = CodeIngestError::configuration_error("invalid database URL");
        assert!(db_config_error.to_string().contains("postgresql://user:pass@host:port/db"));
        
        let path_config_error = CodeIngestError::configuration_error("invalid path");
        assert!(path_config_error.to_string().contains("absolute"));
        
        let token_config_error = CodeIngestError::configuration_error("missing token");
        assert!(token_config_error.to_string().contains("GITHUB_TOKEN"));
    }

    #[test]
    fn test_error_recovery_strategies() {
        let network_error = CodeIngestError::Network {
            message: "connection timeout".to_string(),
            suggestion: "retry".to_string(),
            source: None,
        };
        assert!(network_error.is_recoverable());
        assert_eq!(network_error.retry_delay_seconds(), 5);
        
        let timeout_error = CodeIngestError::Timeout {
            operation: "clone".to_string(),
            timeout_seconds: 30,
            suggestion: "retry".to_string(),
        };
        assert!(timeout_error.is_recoverable());
        assert_eq!(timeout_error.retry_delay_seconds(), 10);
        
        let config_error = CodeIngestError::Configuration {
            message: "invalid config".to_string(),
            suggestion: "fix config".to_string(),
        };
        assert!(!config_error.is_recoverable());
    }

    #[test]
    fn test_external_error_conversions_to_code_ingest_error() {
        // Test sqlx::Error conversion
        let sqlx_error = sqlx::Error::Configuration("Invalid connection string".into());
        let code_ingest_error: CodeIngestError = sqlx_error.into();
        match code_ingest_error {
            CodeIngestError::Database { message, suggestion, .. } => {
                assert!(message.contains("configuration error"));
                assert!(suggestion.contains("PostgreSQL") || suggestion.contains("database"));
            }
            _ => panic!("Expected CodeIngestError::Database"),
        }

        // Test git2::Error conversion
        let git_error = git2::Error::from_str("Authentication failed");
        let code_ingest_error: CodeIngestError = git_error.into();
        match code_ingest_error {
            CodeIngestError::Git { message, suggestion, .. } => {
                assert!(message.contains("Authentication failed"));
                assert!(suggestion.contains("repository URL") || suggestion.contains("network"));
            }
            _ => panic!("Expected CodeIngestError::Git"),
        }

        // Test std::io::Error conversion
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let code_ingest_error: CodeIngestError = io_error.into();
        match code_ingest_error {
            CodeIngestError::FileSystem { message, suggestion, .. } => {
                assert!(message.contains("not found"));
                assert!(suggestion.contains("Verify the path"));
            }
            _ => panic!("Expected CodeIngestError::FileSystem"),
        }

        // Test permission denied conversion
        let permission_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
        let code_ingest_error: CodeIngestError = permission_error.into();
        match code_ingest_error {
            CodeIngestError::PermissionDenied { path: _, suggestion } => {
                assert!(suggestion.contains("permissions"));
            }
            _ => panic!("Expected CodeIngestError::PermissionDenied"),
        }
    }

    #[test]
    fn test_error_display_with_suggestions() {
        let error = CodeIngestError::Git {
            message: "Repository not found".to_string(),
            suggestion: "Check the URL".to_string(),
            source: None,
        };
        let display = error.to_string();
        assert!(display.contains("Git operation failed"));
        assert!(display.contains("Repository not found"));
        assert!(display.contains("💡 Suggestion: Check the URL"));
    }

    #[test]
    fn test_error_source_chaining() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "Original error");
        let source = Some(Box::new(io_error) as Box<dyn std::error::Error + Send + Sync>);
        
        let code_ingest_error = CodeIngestError::FileSystem {
            message: "File operation failed".to_string(),
            suggestion: "Check the path".to_string(),
            source,
        };
        
        // Test that source is preserved
        assert!(code_ingest_error.source().is_some());
        let source_error = code_ingest_error.source().unwrap();
        assert!(source_error.to_string().contains("Original error"));
    }

    #[test]
    fn test_result_type_aliases_with_code_ingest_error() {
        fn test_code_ingest_result() -> CodeIngestResult<String> {
            Ok("success".to_string())
        }

        fn test_code_ingest_error_result() -> CodeIngestResult<i32> {
            Err(CodeIngestError::configuration_error("test error"))
        }

        assert!(test_code_ingest_result().is_ok());
        assert!(test_code_ingest_error_result().is_err());
    }

    // Legacy tests for backward compatibility
    #[test]
    fn test_system_error_from_ingestion_error() {
        let ingestion_error = IngestionError::LocalPathNotFound {
            path: "/nonexistent/path".to_string(),
        };
        let system_error: SystemError = ingestion_error.into();

        match system_error {
            SystemError::Ingestion(IngestionError::LocalPathNotFound { path }) => {
                assert_eq!(path, "/nonexistent/path");
            }
            _ => panic!("Expected SystemError::Ingestion"),
        }
    }

    #[test]
    fn test_system_error_from_database_error() {
        let database_error = DatabaseError::ConnectionFailed {
            url: "postgresql://localhost:5432/test".to_string(),
            cause: "Connection refused".to_string(),
        };
        let system_error: SystemError = database_error.into();

        match system_error {
            SystemError::Database(DatabaseError::ConnectionFailed { url, cause }) => {
                assert_eq!(url, "postgresql://localhost:5432/test");
                assert_eq!(cause, "Connection refused");
            }
            _ => panic!("Expected SystemError::Database"),
        }
    }

    #[test]
    fn test_system_error_from_processing_error() {
        let processing_error = ProcessingError::FileTooLarge {
            path: "/large/file.txt".to_string(),
            size_mb: 1024,
        };
        let system_error: SystemError = processing_error.into();

        match system_error {
            SystemError::Processing(ProcessingError::FileTooLarge { path, size_mb }) => {
                assert_eq!(path, "/large/file.txt");
                assert_eq!(size_mb, 1024);
            }
            _ => panic!("Expected SystemError::Processing"),
        }
    }

    #[test]
    fn test_system_error_from_task_error() {
        let task_error = TaskError::TaskDivisionFailed {
            total_tasks: 5,
            groups: 7,
        };
        let system_error: SystemError = task_error.into();

        match system_error {
            SystemError::TaskGeneration(TaskError::TaskDivisionFailed {
                total_tasks,
                groups,
            }) => {
                assert_eq!(total_tasks, 5);
                assert_eq!(groups, 7);
            }
            _ => panic!("Expected SystemError::TaskGeneration"),
        }
    }

    #[test]
    fn test_system_error_from_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let system_error: SystemError = io_error.into();

        match system_error {
            SystemError::Io(_) => {
                // Success - IO error was properly converted
            }
            _ => panic!("Expected SystemError::Io"),
        }
    }

    #[test]
    fn test_error_display_messages() {
        let ingestion_error = IngestionError::GitCloneFailed {
            repo_url: "https://github.com/user/repo".to_string(),
            cause: "Network timeout".to_string(),
        };
        assert_eq!(
            ingestion_error.to_string(),
            "Git clone failed: https://github.com/user/repo - Network timeout"
        );

        let database_error = DatabaseError::TableCreationFailed {
            table_name: "INGEST_20250927143022".to_string(),
            cause: "Permission denied".to_string(),
        };
        assert_eq!(
            database_error.to_string(),
            "Table creation failed: INGEST_20250927143022 - Permission denied"
        );

        let processing_error = ProcessingError::UnsupportedFileType {
            path: "/path/to/file.unknown".to_string(),
            extension: "unknown".to_string(),
        };
        assert_eq!(
            processing_error.to_string(),
            "Unsupported file type: /path/to/file.unknown - unknown"
        );

        let task_error = TaskError::MarkdownGenerationFailed {
            cause: "Invalid template".to_string(),
        };
        assert_eq!(
            task_error.to_string(),
            "Markdown generation failed: Invalid template"
        );
    }

    #[test]
    fn test_error_context_propagation() {
        // Test that errors can be chained and context is preserved
        let root_cause = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access denied");
        let processing_error = ProcessingError::FileReadFailed {
            path: "/protected/file.txt".to_string(),
            cause: root_cause.to_string(),
        };
        let system_error: SystemError = processing_error.into();

        let error_chain = format!("{}", system_error);
        assert!(error_chain.contains("Processing error"));
        assert!(error_chain.contains("File read failed"));
        assert!(error_chain.contains("/protected/file.txt"));
        assert!(error_chain.contains("Access denied"));
    }

    #[test]
    fn test_result_type_aliases() {
        // Test that result type aliases work correctly
        fn test_system_result() -> SystemResult<String> {
            Ok("success".to_string())
        }

        fn test_ingestion_result() -> IngestionResult<i32> {
            Err(IngestionError::LocalPathNotFound {
                path: "/test".to_string(),
            })
        }

        fn test_database_result() -> DatabaseResult<bool> {
            Ok(true)
        }

        fn test_processing_result() -> ProcessingResult<Vec<u8>> {
            Err(ProcessingError::FileTooLarge {
                path: "/large".to_string(),
                size_mb: 100,
            })
        }

        fn test_task_result() -> TaskResult<()> {
            Ok(())
        }

        assert!(test_system_result().is_ok());
        assert!(test_ingestion_result().is_err());
        assert!(test_database_result().is_ok());
        assert!(test_processing_result().is_err());
        assert!(test_task_result().is_ok());
    }

    #[test]
    fn test_external_error_conversions() {
        // Test conversion from sqlx::Error
        let sqlx_error = sqlx::Error::Configuration("Invalid connection string".into());
        let db_error: DatabaseError = sqlx_error.into();
        match db_error {
            DatabaseError::InvalidDatabaseUrl { url } => {
                assert!(url.contains("Invalid connection string"));
            }
            _ => panic!("Expected InvalidDatabaseUrl"),
        }

        // Test that we can convert from git2::Error
        let git_error = git2::Error::from_str("Test git error");
        let ingestion_error: IngestionError = git_error.into();
        match ingestion_error {
            IngestionError::GitCloneFailed { repo_url: _, cause } => {
                assert!(cause.contains("Test git error"));
            }
            _ => panic!("Expected GitCloneFailed"),
        }
    }
}