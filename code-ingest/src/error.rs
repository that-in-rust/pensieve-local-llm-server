use thiserror::Error;
use tracing::{error, warn, debug, info, instrument};
use std::time::Duration;


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

    #[error("Batch processing failed: processed {processed}/{total} items - {message}\n💡 Suggestion: {suggestion}")]
    BatchProcessingFailed {
        processed: usize,
        total: usize,
        message: String,
        suggestion: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
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

    #[error("Table not found: {table_name}")]
    TableNotFound { table_name: String },

    #[error("Invalid table name: {table_name} - {cause}")]
    InvalidTableName { table_name: String, cause: String },
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

    #[error("Chunking failed: {reason}")]
    ChunkingFailed { reason: String },
}

/// Comprehensive errors for task generation and markdown creation with actionable guidance
#[derive(Error, Debug)]
pub enum TaskError {
    #[error("Task division failed: {total_tasks} tasks cannot be divided into {groups} groups\n💡 Suggestion: {suggestion}")]
    TaskDivisionFailed { 
        total_tasks: usize, 
        groups: usize,
        suggestion: String,
    },

    #[error("Markdown generation failed: {cause}\n💡 Suggestion: {suggestion}")]
    MarkdownGenerationFailed { 
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Task file creation failed: {path} - {cause}\n💡 Suggestion: {suggestion}")]
    TaskFileCreationFailed { 
        path: String, 
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Invalid task configuration: {cause}\n💡 Suggestion: {suggestion}")]
    InvalidTaskConfiguration { 
        cause: String,
        suggestion: String,
    },

    #[error("Query result processing failed: {cause}\n💡 Suggestion: {suggestion}")]
    QueryResultProcessingFailed { 
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Chunk analysis failed: {cause}\n💡 Suggestion: {suggestion}")]
    ChunkAnalysisFailed { 
        cause: String,
        suggestion: String,
    },

    #[error("Content extraction failed for table '{table_name}' row {row}: {cause}\n💡 Suggestion: {suggestion}")]
    ContentExtractionFailed { 
        table_name: String,
        row: usize, 
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Hierarchical division failed: {cause}\n💡 Suggestion: {suggestion}")]
    HierarchicalDivisionFailed { 
        cause: String,
        suggestion: String,
    },

    #[error("L1/L2 context generation failed for file '{filepath}': {cause}\n💡 Suggestion: {suggestion}")]
    ContextGenerationFailed { 
        filepath: String,
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Database query engine operation failed: {operation} - {cause}\n💡 Suggestion: {suggestion}")]
    DatabaseQueryEngineFailed {
        operation: String,
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Task hierarchy creation failed: levels={levels}, groups_per_level={groups_per_level} - {cause}\n💡 Suggestion: {suggestion}")]
    TaskHierarchyFailed {
        levels: usize,
        groups_per_level: usize,
        cause: String,
        suggestion: String,
    },

    #[error("L1L8 markdown generation failed: {cause}\n💡 Suggestion: {suggestion}")]
    L1L8MarkdownFailed {
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Async operation timeout: {operation} exceeded {timeout_seconds}s\n💡 Suggestion: {suggestion}")]
    AsyncTimeout {
        operation: String,
        timeout_seconds: u64,
        suggestion: String,
    },

    #[error("Async operation cancelled: {operation}\n💡 Suggestion: {suggestion}")]
    AsyncCancelled {
        operation: String,
        suggestion: String,
    },

    #[error("Memory limit exceeded during {operation}: used {used_mb}MB, limit {limit_mb}MB\n💡 Suggestion: {suggestion}")]
    MemoryLimitExceeded {
        operation: String,
        used_mb: usize,
        limit_mb: usize,
        suggestion: String,
    },

    #[error("Batch processing failed: processed {processed}/{total} items - {cause}\n💡 Suggestion: {suggestion}")]
    BatchProcessingFailed {
        processed: usize,
        total: usize,
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Progress reporting failed: {cause}\n💡 Suggestion: {suggestion}")]
    ProgressReportingFailed {
        cause: String,
        suggestion: String,
    },

    #[error("Large table processing failed: table '{table_name}' with {row_count} rows - {cause}\n💡 Suggestion: {suggestion}")]
    LargeTableProcessingFailed {
        table_name: String,
        row_count: usize,
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Streaming operation failed: {operation} - {cause}\n💡 Suggestion: {suggestion}")]
    StreamingFailed {
        operation: String,
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Ingestion source validation failed: {source_name} - {cause}\n💡 Suggestion: {suggestion}")]
    IngestionSourceValidationFailed {
        source_name: String,
        cause: String,
        suggestion: String,
    },

    #[error("Chunk metadata creation failed: {filepath} - {cause}\n💡 Suggestion: {suggestion}")]
    ChunkMetadataCreationFailed {
        filepath: String,
        cause: String,
        suggestion: String,
        #[source] 
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Task hierarchy validation failed: levels={levels}, groups={groups} - {cause}\n💡 Suggestion: {suggestion}")]
    TaskHierarchyValidationFailed {
        levels: u8,
        groups: u8,
        cause: String,
        suggestion: String,
    },

    #[error("Generation config validation failed: {field} - {cause}\n💡 Suggestion: {suggestion}")]
    GenerationConfigValidationFailed {
        field: String,
        cause: String,
        suggestion: String,
    },
}

/// Result type aliases for common operations
pub type CodeIngestResult<T> = Result<T, CodeIngestError>;
pub type SystemResult<T> = Result<T, SystemError>;
pub type IngestionResult<T> = Result<T, IngestionError>;
pub type DatabaseResult<T> = Result<T, DatabaseError>;
pub type ProcessingResult<T> = Result<T, ProcessingError>;
pub type TaskResult<T> = Result<T, TaskError>;

/// Enhanced task error creation helpers with actionable suggestions
impl TaskError {
    /// Create a task division error with actionable suggestions
    pub fn task_division_failed(total_tasks: usize, groups: usize) -> Self {
        let suggestion = if total_tasks < groups {
            format!("Reduce group count to {} or fewer, or increase the number of tasks", total_tasks)
        } else if groups == 0 {
            "Group count must be greater than 0".to_string()
        } else {
            "Check task generation logic and ensure proper task distribution".to_string()
        };

        error!("Task division failed: {} tasks cannot be divided into {} groups", total_tasks, groups);
        
        Self::TaskDivisionFailed { 
            total_tasks, 
            groups, 
            suggestion 
        }
    }

    /// Create a content extraction error with actionable suggestions
    pub fn content_extraction_failed(
        table_name: impl Into<String>,
        row: usize,
        cause: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let table_name = table_name.into();
        let cause = cause.into();
        
        let suggestion = if cause.contains("connection") {
            "Check database connection and retry the operation".to_string()
        } else if cause.contains("timeout") {
            "Increase timeout settings or process smaller batches".to_string()
        } else if cause.contains("memory") {
            "Reduce batch size or increase available memory".to_string()
        } else if cause.contains("permission") {
            format!("Check read permissions for table '{}'", table_name)
        } else {
            format!("Verify table '{}' exists and row {} is accessible", table_name, row)
        };

        error!("Content extraction failed for table '{}' row {}: {}", table_name, row, cause);
        debug!("Content extraction error source: {:?}", source);
        
        Self::ContentExtractionFailed { 
            table_name, 
            row, 
            cause, 
            suggestion, 
            source 
        }
    }

    /// Create a context generation error with actionable suggestions
    pub fn context_generation_failed(
        filepath: impl Into<String>,
        cause: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let filepath = filepath.into();
        let cause = cause.into();
        
        let suggestion = if cause.contains("encoding") {
            "Check file encoding and ensure it's UTF-8 compatible".to_string()
        } else if cause.contains("size") {
            "File may be too large for context generation, consider chunking".to_string()
        } else if cause.contains("parse") {
            "File content may be corrupted or in an unexpected format".to_string()
        } else {
            "Verify file exists and is readable, check file format".to_string()
        };

        warn!("Context generation failed for file '{}': {}", filepath, cause);
        debug!("Context generation error source: {:?}", source);
        
        Self::ContextGenerationFailed { 
            filepath, 
            cause, 
            suggestion, 
            source 
        }
    }

    /// Create a database query engine error with actionable suggestions
    pub fn database_query_engine_failed(
        operation: impl Into<String>,
        cause: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let operation = operation.into();
        let cause = cause.into();
        
        let suggestion = match operation.as_str() {
            "count_rows" => "Check table name and database connection".to_string(),
            "table_exists" => "Verify database connection and table permissions".to_string(),
            "validate_table" => "Ensure table exists and you have read permissions".to_string(),
            _ => "Check database connection and query syntax".to_string(),
        };

        error!("Database query engine operation '{}' failed: {}", operation, cause);
        debug!("Database query engine error source: {:?}", source);
        
        Self::DatabaseQueryEngineFailed { 
            operation, 
            cause, 
            suggestion, 
            source 
        }
    }

    /// Create a task hierarchy error with actionable suggestions
    pub fn task_hierarchy_failed(
        levels: usize,
        groups_per_level: usize,
        cause: impl Into<String>
    ) -> Self {
        let cause = cause.into();
        
        let suggestion = if levels == 0 {
            "Levels must be greater than 0".to_string()
        } else if groups_per_level == 0 {
            "Groups per level must be greater than 0".to_string()
        } else if levels > 10 {
            "Consider reducing hierarchy levels for better usability (recommended: 2-4 levels)".to_string()
        } else if groups_per_level > 20 {
            "Consider reducing groups per level for better organization (recommended: 5-10 groups)".to_string()
        } else {
            "Check hierarchy parameters and ensure they match your task count".to_string()
        };

        error!("Task hierarchy creation failed: levels={}, groups_per_level={} - {}", levels, groups_per_level, cause);
        
        Self::TaskHierarchyFailed { 
            levels, 
            groups_per_level, 
            cause, 
            suggestion 
        }
    }

    /// Create an async timeout error with actionable suggestions
    pub fn async_timeout(operation: impl Into<String>, timeout_seconds: u64) -> Self {
        let operation = operation.into();
        
        let suggestion = match operation.as_str() {
            op if op.contains("database") => "Increase database timeout or optimize query performance".to_string(),
            op if op.contains("file") => "Increase file I/O timeout or process smaller files".to_string(),
            op if op.contains("network") => "Check network connection and increase network timeout".to_string(),
            _ => format!("Increase timeout from {}s or optimize the operation", timeout_seconds),
        };

        warn!("Async operation '{}' timed out after {}s", operation, timeout_seconds);
        
        Self::AsyncTimeout { 
            operation, 
            timeout_seconds, 
            suggestion 
        }
    }

    /// Create a memory limit exceeded error with actionable suggestions
    pub fn memory_limit_exceeded(
        operation: impl Into<String>,
        used_mb: usize,
        limit_mb: usize
    ) -> Self {
        let operation = operation.into();
        
        let suggestion = if used_mb > limit_mb * 2 {
            format!("Memory usage ({} MB) is very high. Consider processing in smaller batches or increasing memory limit to {} MB", used_mb, used_mb + 100)
        } else {
            format!("Increase memory limit from {} MB to {} MB or process in smaller batches", limit_mb, limit_mb * 2)
        };

        error!("Memory limit exceeded during '{}': used {} MB, limit {} MB", operation, used_mb, limit_mb);
        
        Self::MemoryLimitExceeded { 
            operation, 
            used_mb, 
            limit_mb, 
            suggestion 
        }
    }

    /// Create a large table processing error with actionable suggestions
    pub fn large_table_processing_failed(
        table_name: impl Into<String>,
        row_count: usize,
        cause: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let table_name = table_name.into();
        let cause = cause.into();
        
        let suggestion = if row_count > 100_000 {
            format!("Table has {} rows. Use streaming with batch size 1000-5000 for better performance", row_count)
        } else if row_count > 10_000 {
            format!("Table has {} rows. Consider using batch processing with size 500-1000", row_count)
        } else {
            "Enable progress reporting and consider parallel processing".to_string()
        };

        error!("Large table processing failed for '{}' with {} rows: {}", table_name, row_count, cause);
        debug!("Large table processing error source: {:?}", source);
        
        Self::LargeTableProcessingFailed { 
            table_name, 
            row_count, 
            cause, 
            suggestion, 
            source 
        }
    }

    /// Check if this error is recoverable with retry
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::AsyncTimeout { .. } => true,
            Self::DatabaseQueryEngineFailed { cause, .. } if cause.contains("connection") => true,
            Self::ContentExtractionFailed { cause, .. } if cause.contains("timeout") || cause.contains("connection") => true,
            Self::StreamingFailed { cause, .. } if cause.contains("network") || cause.contains("timeout") => true,
            Self::BatchProcessingFailed { cause, .. } if cause.contains("timeout") => true,
            _ => false,
        }
    }

    /// Get suggested retry delay in seconds
    pub fn retry_delay_seconds(&self) -> u64 {
        match self {
            Self::AsyncTimeout { .. } => 10,
            Self::DatabaseQueryEngineFailed { .. } => 5,
            Self::ContentExtractionFailed { .. } => 3,
            Self::StreamingFailed { .. } => 5,
            Self::BatchProcessingFailed { .. } => 2,
            _ => 1,
        }
    }

    /// Create an ingestion source validation error
    pub fn ingestion_source_validation_failed(
        source_name: impl Into<String>,
        cause: impl Into<String>
    ) -> Self {
        let source_name = source_name.into();
        let cause = cause.into();
        
        let suggestion = if cause.contains("URL") {
            "Check the Git repository URL format and ensure it's accessible".to_string()
        } else if cause.contains("path") || cause.contains("directory") {
            "Verify the local folder path exists and is readable".to_string()
        } else if cause.contains("permission") {
            "Check file and directory permissions".to_string()
        } else {
            "Validate the ingestion source configuration".to_string()
        };

        error!("Ingestion source validation failed for '{}': {}", source_name, cause);
        
        Self::IngestionSourceValidationFailed { 
            source_name, 
            cause, 
            suggestion 
        }
    }

    /// Create a chunk metadata creation error
    pub fn chunk_metadata_creation_failed(
        filepath: impl Into<String>,
        cause: impl Into<String>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>
    ) -> Self {
        let filepath = filepath.into();
        let cause = cause.into();
        
        let suggestion = if cause.contains("encoding") {
            "Check file encoding and ensure it's UTF-8 compatible".to_string()
        } else if cause.contains("size") {
            "File may be too large, consider adjusting chunk size".to_string()
        } else if cause.contains("lines") {
            "Verify file line count and chunk boundaries".to_string()
        } else {
            "Check file accessibility and format".to_string()
        };

        error!("Chunk metadata creation failed for '{}': {}", filepath, cause);
        debug!("Chunk metadata creation error source: {:?}", source);
        
        Self::ChunkMetadataCreationFailed { 
            filepath, 
            cause, 
            suggestion, 
            source 
        }
    }

    /// Create a task hierarchy validation error
    pub fn task_hierarchy_validation_failed(
        levels: u8,
        groups: u8,
        cause: impl Into<String>
    ) -> Self {
        let cause = cause.into();
        
        let suggestion = if levels == 0 {
            "Hierarchy levels must be at least 1".to_string()
        } else if groups == 0 {
            "Groups per level must be at least 1".to_string()
        } else if levels > 10 {
            "Consider reducing hierarchy levels (recommended: 2-4 levels)".to_string()
        } else if groups > 20 {
            "Consider reducing groups per level (recommended: 5-10 groups)".to_string()
        } else {
            "Check hierarchy configuration parameters".to_string()
        };

        error!("Task hierarchy validation failed: levels={}, groups={} - {}", levels, groups, cause);
        
        Self::TaskHierarchyValidationFailed { 
            levels, 
            groups, 
            cause, 
            suggestion 
        }
    }

    /// Create a generation config validation error
    pub fn generation_config_validation_failed(
        field: impl Into<String>,
        cause: impl Into<String>
    ) -> Self {
        let field = field.into();
        let cause = cause.into();
        
        let suggestion = match field.as_str() {
            "table_name" => "Provide a valid database table name".to_string(),
            "levels" => "Set hierarchy levels to a value between 1 and 10".to_string(),
            "groups" => "Set groups per level to a value between 1 and 20".to_string(),
            "chunk_size" => "Set chunk size to a positive integer".to_string(),
            "output_file" => "Provide a valid output file path".to_string(),
            _ => "Check the configuration parameter value".to_string(),
        };

        error!("Generation config validation failed for field '{}': {}", field, cause);
        
        Self::GenerationConfigValidationFailed { 
            field, 
            cause, 
            suggestion 
        }
    }

    /// Log error with appropriate level
    pub fn log_error(&self) {
        match self {
            Self::ProgressReportingFailed { .. } => debug!("{}", self),
            Self::ContextGenerationFailed { .. } => warn!("{}", self),
            Self::AsyncCancelled { .. } => info!("{}", self),
            _ => error!("{}", self),
        }
    }
}

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
            DatabaseError::ConnectionFailed { url: _, cause } => IngestionError::NetworkError { cause },
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
    fn test_task_error_creation_helpers() {
        // Test task division error
        let division_error = TaskError::task_division_failed(5, 7);
        match division_error {
            TaskError::TaskDivisionFailed { total_tasks, groups, suggestion } => {
                assert_eq!(total_tasks, 5);
                assert_eq!(groups, 7);
                assert!(suggestion.contains("Reduce group count"));
            }
            _ => panic!("Expected TaskDivisionFailed"),
        }

        // Test content extraction error
        let extraction_error = TaskError::content_extraction_failed(
            "test_table", 
            42, 
            "connection timeout", 
            None
        );
        match extraction_error {
            TaskError::ContentExtractionFailed { table_name, row, cause, suggestion, .. } => {
                assert_eq!(table_name, "test_table");
                assert_eq!(row, 42);
                assert!(cause.contains("connection timeout"));
                assert!(suggestion.contains("database connection"));
            }
            _ => panic!("Expected ContentExtractionFailed"),
        }

        // Test context generation error
        let context_error = TaskError::context_generation_failed(
            "/path/to/file.rs",
            "encoding error",
            None
        );
        match context_error {
            TaskError::ContextGenerationFailed { filepath, cause, suggestion, .. } => {
                assert_eq!(filepath, "/path/to/file.rs");
                assert!(cause.contains("encoding error"));
                assert!(suggestion.contains("UTF-8"));
            }
            _ => panic!("Expected ContextGenerationFailed"),
        }

        // Test database query engine error
        let db_error = TaskError::database_query_engine_failed(
            "count_rows",
            "table not found",
            None
        );
        match db_error {
            TaskError::DatabaseQueryEngineFailed { operation, cause, suggestion, .. } => {
                assert_eq!(operation, "count_rows");
                assert!(cause.contains("table not found"));
                assert!(suggestion.contains("table name"));
            }
            _ => panic!("Expected DatabaseQueryEngineFailed"),
        }

        // Test task hierarchy error
        let hierarchy_error = TaskError::task_hierarchy_failed(0, 7, "invalid levels");
        match hierarchy_error {
            TaskError::TaskHierarchyFailed { levels, groups_per_level, cause, suggestion } => {
                assert_eq!(levels, 0);
                assert_eq!(groups_per_level, 7);
                assert!(cause.contains("invalid levels"));
                assert!(suggestion.contains("greater than 0"));
            }
            _ => panic!("Expected TaskHierarchyFailed"),
        }

        // Test async timeout error
        let timeout_error = TaskError::async_timeout("database_query", 30);
        match timeout_error {
            TaskError::AsyncTimeout { operation, timeout_seconds, suggestion } => {
                assert_eq!(operation, "database_query");
                assert_eq!(timeout_seconds, 30);
                assert!(suggestion.contains("database timeout"));
            }
            _ => panic!("Expected AsyncTimeout"),
        }

        // Test memory limit exceeded error
        let memory_error = TaskError::memory_limit_exceeded("content_processing", 512, 256);
        match memory_error {
            TaskError::MemoryLimitExceeded { operation, used_mb, limit_mb, suggestion } => {
                assert_eq!(operation, "content_processing");
                assert_eq!(used_mb, 512);
                assert_eq!(limit_mb, 256);
                assert!(suggestion.contains("512 MB"));
            }
            _ => panic!("Expected MemoryLimitExceeded"),
        }

        // Test large table processing error
        let large_table_error = TaskError::large_table_processing_failed(
            "huge_table",
            150_000,
            "memory exhausted",
            None
        );
        match large_table_error {
            TaskError::LargeTableProcessingFailed { table_name, row_count, cause, suggestion, .. } => {
                assert_eq!(table_name, "huge_table");
                assert_eq!(row_count, 150_000);
                assert!(cause.contains("memory exhausted"));
                assert!(suggestion.contains("streaming"));
            }
            _ => panic!("Expected LargeTableProcessingFailed"),
        }
    }

    #[test]
    fn test_task_error_recovery_strategies() {
        // Test recoverable errors
        let timeout_error = TaskError::async_timeout("operation", 30);
        assert!(timeout_error.is_recoverable());
        assert_eq!(timeout_error.retry_delay_seconds(), 10);

        let db_connection_error = TaskError::database_query_engine_failed(
            "query", 
            "connection failed", 
            None
        );
        assert!(db_connection_error.is_recoverable());
        assert_eq!(db_connection_error.retry_delay_seconds(), 5);

        let extraction_timeout_error = TaskError::content_extraction_failed(
            "table", 
            1, 
            "timeout occurred", 
            None
        );
        assert!(extraction_timeout_error.is_recoverable());
        assert_eq!(extraction_timeout_error.retry_delay_seconds(), 3);

        // Test non-recoverable errors
        let config_error = TaskError::task_hierarchy_failed(0, 7, "invalid config");
        assert!(!config_error.is_recoverable());

        let context_error = TaskError::context_generation_failed(
            "file.rs", 
            "parse error", 
            None
        );
        assert!(!context_error.is_recoverable());
    }

    #[test]
    fn test_task_error_suggestions_quality() {
        // Test that suggestions are actionable and specific
        
        // Division error with too few tasks
        let division_error = TaskError::task_division_failed(3, 7);
        let error_string = division_error.to_string();
        assert!(error_string.contains("Reduce group count to 3"));

        // Division error with zero groups
        let zero_groups_error = TaskError::task_division_failed(10, 0);
        let error_string = zero_groups_error.to_string();
        assert!(error_string.contains("Group count must be greater than 0"));

        // Context generation with encoding issue
        let encoding_error = TaskError::context_generation_failed(
            "file.txt", 
            "encoding detection failed", 
            None
        );
        let error_string = encoding_error.to_string();
        assert!(error_string.contains("UTF-8 compatible"));

        // Memory error with high usage
        let high_memory_error = TaskError::memory_limit_exceeded("processing", 1024, 256);
        let error_string = high_memory_error.to_string();
        assert!(error_string.contains("very high"));
        assert!(error_string.contains("1124 MB")); // 1024 + 100

        // Large table with very high row count
        let huge_table_error = TaskError::large_table_processing_failed(
            "massive_table",
            500_000,
            "processing failed",
            None
        );
        let error_string = huge_table_error.to_string();
        assert!(error_string.contains("streaming"));
        assert!(error_string.contains("1000-5000"));
    }

    #[test]
    fn test_task_error_display_format() {
        let error = TaskError::content_extraction_failed(
            "INGEST_20250928101039",
            35,
            "database connection timeout",
            None
        );
        
        let error_string = error.to_string();
        
        // Check that error contains all expected components
        assert!(error_string.contains("Content extraction failed"));
        assert!(error_string.contains("INGEST_20250928101039"));
        assert!(error_string.contains("row 35"));
        assert!(error_string.contains("database connection timeout"));
        assert!(error_string.contains("💡 Suggestion:"));
        assert!(error_string.contains("database connection"));
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
            suggestion: "Reduce group count or increase tasks".to_string(),
        };
        let system_error: SystemError = task_error.into();

        match system_error {
            SystemError::TaskGeneration(TaskError::TaskDivisionFailed {
                total_tasks,
                groups,
                suggestion: _,
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
            suggestion: "Check template format and try again".to_string(),
            source: None,
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
/// Error recovery strategies for different types of failures
#[derive(Debug, Clone)]
pub struct ErrorRecoveryStrategy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub recoverable_errors: Vec<String>,
}

impl Default for ErrorRecoveryStrategy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            recoverable_errors: vec![
                "connection".to_string(),
                "timeout".to_string(),
                "network".to_string(),
                "temporary".to_string(),
            ],
        }
    }
}

impl ErrorRecoveryStrategy {
    /// Create a strategy for database operations
    pub fn database() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 1.5,
            recoverable_errors: vec![
                "connection".to_string(),
                "timeout".to_string(),
                "deadlock".to_string(),
                "lock".to_string(),
            ],
        }
    }

    /// Create a strategy for network operations
    pub fn network() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            recoverable_errors: vec![
                "network".to_string(),
                "timeout".to_string(),
                "dns".to_string(),
                "connection".to_string(),
                "502".to_string(),
                "503".to_string(),
                "504".to_string(),
            ],
        }
    }

    /// Create a strategy for filesystem operations
    pub fn filesystem() -> Self {
        Self {
            max_retries: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            recoverable_errors: vec![
                "busy".to_string(),
                "locked".to_string(),
                "temporary".to_string(),
            ],
        }
    }

    /// Calculate delay for retry attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let delay = self.base_delay.as_millis() as f64 * self.backoff_multiplier.powi(attempt as i32);
        let delay = Duration::from_millis(delay as u64);
        std::cmp::min(delay, self.max_delay)
    }

    /// Check if error is recoverable based on error message
    pub fn is_recoverable(&self, error_message: &str) -> bool {
        let message_lower = error_message.to_lowercase();
        self.recoverable_errors.iter().any(|pattern| message_lower.contains(pattern))
    }
}

/// Enhanced error context with recovery information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub retry_count: u32,
    pub recovery_strategy: Option<ErrorRecoveryStrategy>,
    pub user_context: Option<String>,
    pub system_context: Option<String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            component: component.into(),
            timestamp: chrono::Utc::now(),
            retry_count: 0,
            recovery_strategy: None,
            user_context: None,
            system_context: None,
        }
    }

    pub fn with_recovery_strategy(mut self, strategy: ErrorRecoveryStrategy) -> Self {
        self.recovery_strategy = Some(strategy);
        self
    }

    pub fn with_user_context(mut self, context: impl Into<String>) -> Self {
        self.user_context = Some(context.into());
        self
    }

    pub fn with_system_context(mut self, context: impl Into<String>) -> Self {
        self.system_context = Some(context.into());
        self
    }

    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    pub fn should_retry(&self, error_message: &str) -> bool {
        if let Some(strategy) = &self.recovery_strategy {
            self.retry_count < strategy.max_retries && strategy.is_recoverable(error_message)
        } else {
            false
        }
    }

    pub fn next_retry_delay(&self) -> Option<Duration> {
        self.recovery_strategy.as_ref().map(|strategy| strategy.calculate_delay(self.retry_count))
    }
}

/// Retry executor with exponential backoff
pub struct RetryExecutor {
    strategy: ErrorRecoveryStrategy,
}

impl RetryExecutor {
    pub fn new(strategy: ErrorRecoveryStrategy) -> Self {
        Self { strategy }
    }

    /// Execute operation with retry logic
    #[instrument(skip(self, operation))]
    pub async fn execute<F, Fut, T, E>(&self, mut operation: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut attempt = 0;
        
        loop {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    let error_message = error.to_string();
                    
                    if attempt >= self.strategy.max_retries || !self.strategy.is_recoverable(&error_message) {
                        error!("Operation failed after {} attempts: {}", attempt + 1, error_message);
                        return Err(error);
                    }
                    
                    let delay = self.strategy.calculate_delay(attempt);
                    warn!("Operation failed (attempt {}), retrying in {:?}: {}", 
                          attempt + 1, delay, error_message);
                    
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }
}

/// Error aggregator for batch operations
#[derive(Debug, Default)]
pub struct ErrorAggregator {
    errors: Vec<(String, Box<dyn std::error::Error + Send + Sync>)>,
    warnings: Vec<String>,
    context: Option<String>,
}

impl ErrorAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn add_error(&mut self, operation: impl Into<String>, error: Box<dyn std::error::Error + Send + Sync>) {
        let operation = operation.into();
        error!("Error in operation '{}': {}", operation, error);
        self.errors.push((operation, error));
    }

    pub fn add_warning(&mut self, warning: impl Into<String>) {
        let warning = warning.into();
        warn!("Warning: {}", warning);
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    pub fn into_result<T>(self, success_value: T) -> Result<T, CodeIngestError> {
        if self.has_errors() {
            let error_summary = self.errors.iter()
                .map(|(op, err)| format!("{}: {}", op, err))
                .collect::<Vec<_>>()
                .join("; ");
            
            let suggestion = if self.errors.len() == 1 {
                "Check the specific error above and retry the operation".to_string()
            } else {
                format!("Multiple errors occurred ({}). Check logs and retry failed operations individually", self.errors.len())
            };

            Err(CodeIngestError::BatchProcessingFailed {
                processed: 0, // This should be set by the caller
                total: self.errors.len(),
                message: error_summary,
                suggestion,
                source: None,
            })
        } else {
            if self.has_warnings() {
                info!("Operation completed with {} warnings", self.warnings.len());
            }
            Ok(success_value)
        }
    }

    pub fn log_summary(&self) {
        if self.has_errors() || self.has_warnings() {
            let context = self.context.as_deref().unwrap_or("batch operation");
            info!("Summary for {}: {} errors, {} warnings", 
                  context, self.error_count(), self.warning_count());
        }
    }
}

/// Circuit breaker for preventing cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    recovery_timeout: Duration,
    failure_count: std::sync::atomic::AtomicU32,
    last_failure_time: std::sync::Mutex<Option<std::time::Instant>>,
    state: std::sync::Mutex<CircuitBreakerState>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitBreakerState {
    Closed,  // Normal operation
    Open,    // Failing fast
    HalfOpen, // Testing recovery
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
            failure_count: std::sync::atomic::AtomicU32::new(0),
            last_failure_time: std::sync::Mutex::new(None),
            state: std::sync::Mutex::new(CircuitBreakerState::Closed),
        }
    }

    /// Execute operation through circuit breaker
    #[instrument(skip(self, operation))]
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display,
    {
        // Check if circuit is open
        {
            let mut state = self.state.lock().unwrap();
            match *state {
                CircuitBreakerState::Open => {
                    let last_failure = self.last_failure_time.lock().unwrap();
                    if let Some(last_time) = *last_failure {
                        if last_time.elapsed() > self.recovery_timeout {
                            *state = CircuitBreakerState::HalfOpen;
                            debug!("Circuit breaker transitioning to half-open state");
                        } else {
                            debug!("Circuit breaker is open, failing fast");
                            return Err(CircuitBreakerError::CircuitOpen);
                        }
                    }
                }
                CircuitBreakerState::HalfOpen => {
                    debug!("Circuit breaker in half-open state, testing operation");
                }
                CircuitBreakerState::Closed => {
                    // Normal operation
                }
            }
        }

        // Execute operation
        match operation().await {
            Ok(result) => {
                // Success - reset failure count and close circuit
                self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
                *self.state.lock().unwrap() = CircuitBreakerState::Closed;
                debug!("Operation succeeded, circuit breaker closed");
                Ok(result)
            }
            Err(error) => {
                // Failure - increment count and potentially open circuit
                let failures = self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                *self.last_failure_time.lock().unwrap() = Some(std::time::Instant::now());
                
                if failures >= self.failure_threshold {
                    *self.state.lock().unwrap() = CircuitBreakerState::Open;
                    warn!("Circuit breaker opened after {} failures", failures);
                }
                
                Err(CircuitBreakerError::OperationFailed(error))
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum CircuitBreakerError<E> {
    #[error("Circuit breaker is open, failing fast")]
    CircuitOpen,
    #[error("Operation failed: {0}")]
    OperationFailed(E),
}

/// Enhanced error reporting with structured context
pub struct ErrorReporter {
    component: String,
}

impl ErrorReporter {
    pub fn new(component: impl Into<String>) -> Self {
        Self {
            component: component.into(),
        }
    }

    /// Report error with full context
    #[instrument(skip(self, error))]
    pub fn report_error<E>(&self, error: &E, context: &ErrorContext) -> String
    where
        E: std::error::Error,
    {
        let error_id = uuid::Uuid::new_v4();
        
        error!(
            error_id = %error_id,
            component = %self.component,
            operation = %context.operation,
            retry_count = context.retry_count,
            "Error occurred: {}", error
        );

        // Log error chain
        let mut source = error.source();
        let mut chain_depth = 0;
        while let Some(err) = source {
            chain_depth += 1;
            debug!(
                error_id = %error_id,
                chain_depth = chain_depth,
                "Error source: {}", err
            );
            source = err.source();
        }

        // Log context information
        if let Some(user_context) = &context.user_context {
            debug!(error_id = %error_id, "User context: {}", user_context);
        }
        
        if let Some(system_context) = &context.system_context {
            debug!(error_id = %error_id, "System context: {}", system_context);
        }

        format!("Error ID: {} - {}", error_id, error)
    }

    /// Report warning with context
    #[instrument(skip(self))]
    pub fn report_warning(&self, message: &str, context: Option<&ErrorContext>) {
        if let Some(ctx) = context {
            warn!(
                component = %self.component,
                operation = %ctx.operation,
                "Warning: {}", message
            );
        } else {
            warn!(component = %self.component, "Warning: {}", message);
        }
    }

    /// Report info with context
    #[instrument(skip(self))]
    pub fn report_info(&self, message: &str, context: Option<&ErrorContext>) {
        if let Some(ctx) = context {
            info!(
                component = %self.component,
                operation = %ctx.operation,
                "Info: {}", message
            );
        } else {
            info!(component = %self.component, "Info: {}", message);
        }
    }
}

/// Convenience macros for error handling with context
#[macro_export]
macro_rules! with_error_context {
    ($operation:expr, $component:expr, $result:expr) => {{
        let context = $crate::error::ErrorContext::new($operation, $component);
        match $result {
            Ok(value) => Ok(value),
            Err(error) => {
                let reporter = $crate::error::ErrorReporter::new($component);
                let error_id = reporter.report_error(&error, &context);
                Err(error)
            }
        }
    }};
}

#[macro_export]
macro_rules! retry_with_backoff {
    ($strategy:expr, $operation:expr) => {{
        let executor = $crate::error::RetryExecutor::new($strategy);
        executor.execute($operation).await
    }};
}

/// Performance metrics for error analysis
#[derive(Debug, Default)]
pub struct ErrorMetrics {
    pub total_errors: std::sync::atomic::AtomicU64,
    pub errors_by_type: std::sync::Mutex<std::collections::HashMap<String, u64>>,
    pub recovery_attempts: std::sync::atomic::AtomicU64,
    pub successful_recoveries: std::sync::atomic::AtomicU64,
}

impl ErrorMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_error(&self, error_type: &str) {
        self.total_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let mut errors_by_type = self.errors_by_type.lock().unwrap();
        *errors_by_type.entry(error_type.to_string()).or_insert(0) += 1;
    }

    pub fn record_recovery_attempt(&self) {
        self.recovery_attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_successful_recovery(&self) {
        self.successful_recoveries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_error_rate(&self) -> f64 {
        let total = self.total_errors.load(std::sync::atomic::Ordering::Relaxed);
        let attempts = self.recovery_attempts.load(std::sync::atomic::Ordering::Relaxed);
        
        if attempts > 0 {
            total as f64 / attempts as f64
        } else {
            0.0
        }
    }

    pub fn get_recovery_rate(&self) -> f64 {
        let successful = self.successful_recoveries.load(std::sync::atomic::Ordering::Relaxed);
        let attempts = self.recovery_attempts.load(std::sync::atomic::Ordering::Relaxed);
        
        if attempts > 0 {
            successful as f64 / attempts as f64
        } else {
            0.0
        }
    }

    #[instrument(skip(self))]
    pub fn log_metrics(&self) {
        let total_errors = self.total_errors.load(std::sync::atomic::Ordering::Relaxed);
        let recovery_attempts = self.recovery_attempts.load(std::sync::atomic::Ordering::Relaxed);
        let successful_recoveries = self.successful_recoveries.load(std::sync::atomic::Ordering::Relaxed);
        
        info!(
            total_errors = total_errors,
            recovery_attempts = recovery_attempts,
            successful_recoveries = successful_recoveries,
            error_rate = self.get_error_rate(),
            recovery_rate = self.get_recovery_rate(),
            "Error metrics summary"
        );

        let errors_by_type = self.errors_by_type.lock().unwrap();
        for (error_type, count) in errors_by_type.iter() {
            debug!(error_type = %error_type, count = count, "Error type frequency");
        }
    }
}