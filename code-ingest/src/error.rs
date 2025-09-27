use thiserror::Error;

/// Top-level system error that encompasses all error types
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
}

/// Errors related to repository ingestion and file discovery
#[derive(Error, Debug)]
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
pub type SystemResult<T> = Result<T, SystemError>;
pub type IngestionResult<T> = Result<T, IngestionError>;
pub type DatabaseResult<T> = Result<T, DatabaseError>;
pub type ProcessingResult<T> = Result<T, ProcessingError>;
pub type TaskResult<T> = Result<T, TaskError>;

/// Convert from external error types to our error types
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

#[cfg(test)]
mod tests {
    use super::*;

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