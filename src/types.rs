//! Core data types and structures for the Pensieve CLI tool

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Comprehensive file metadata representation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Complete file path
    pub full_filepath: PathBuf,
    /// Directory containing the file
    pub folder_path: PathBuf,
    /// File name with extension
    pub filename: String,
    /// File extension (without the dot)
    pub file_extension: Option<String>,
    /// File type classification
    pub file_type: FileType,
    /// File size in bytes
    pub size: u64,
    /// SHA-256 hash of file content
    pub hash: String,
    /// File creation timestamp
    pub creation_date: DateTime<Utc>,
    /// File modification timestamp
    pub modification_date: DateTime<Utc>,
    /// File access timestamp
    pub access_date: DateTime<Utc>,
    /// File permissions (Unix-style)
    pub permissions: u32,
    /// Directory depth level
    pub depth_level: u32,
    /// Path relative to scan root
    pub relative_path: PathBuf,
    /// Whether file is hidden
    pub is_hidden: bool,
    /// Whether file is a symbolic link
    pub is_symlink: bool,
    /// Target of symbolic link (if applicable)
    pub symlink_target: Option<PathBuf>,
    /// Duplicate status for deduplication
    pub duplicate_status: DuplicateStatus,
    /// Group ID for duplicate files
    pub duplicate_group_id: Option<Uuid>,
    /// Current processing status
    pub processing_status: ProcessingStatus,
    /// Estimated token count after processing
    pub estimated_tokens: Option<u32>,
    /// Processing timestamp
    pub processed_at: Option<DateTime<Utc>>,
    /// Error message if processing failed
    pub error_message: Option<String>,
}

/// File type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileType {
    /// Regular file
    File,
    /// Directory
    Directory,
}

/// Processing status tracking for files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// File is queued for processing
    Pending,
    /// File has been successfully processed
    Processed,
    /// File processing encountered an error
    Error,
    /// File was skipped because it's binary
    SkippedBinary,
    /// File was skipped due to missing external dependency
    SkippedDependency,
    /// File was deleted from filesystem
    Deleted,
}

/// Duplicate status for file-level deduplication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DuplicateStatus {
    /// File has unique content
    Unique,
    /// First occurrence of duplicate content (canonical)
    Canonical,
    /// Subsequent occurrence of duplicate content
    Duplicate,
}

/// Unique identifier for files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileId(pub Uuid);

impl FileId {
    /// Generate a new unique file ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for FileId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for FileId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<FileId> for Uuid {
    fn from(id: FileId) -> Self {
        id.0
    }
}

/// Unique identifier for paragraphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParagraphId(pub Uuid);

impl ParagraphId {
    /// Generate a new unique paragraph ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ParagraphId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for ParagraphId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<ParagraphId> for Uuid {
    fn from(id: ParagraphId) -> Self {
        id.0
    }
}

/// Content paragraph with metadata
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paragraph {
    /// Unique paragraph identifier
    pub id: ParagraphId,
    /// SHA-256 hash of content for deduplication
    pub content_hash: String,
    /// Actual text content
    pub content: String,
    /// Estimated token count
    pub estimated_tokens: u32,
    /// Word count
    pub word_count: u32,
    /// Character count
    pub char_count: u32,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Link between paragraphs and their source files
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParagraphSource {
    /// Paragraph identifier
    pub paragraph_id: ParagraphId,
    /// Source file identifier
    pub file_id: FileId,
    /// Position within the file (0-based)
    pub paragraph_index: u32,
    /// Byte offset where paragraph starts
    pub byte_offset_start: u64,
    /// Byte offset where paragraph ends
    pub byte_offset_end: u64,
}

/// Processing error information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingError {
    /// Unique error identifier
    pub id: Uuid,
    /// Associated file (if applicable)
    pub file_id: Option<FileId>,
    /// Error type classification
    pub error_type: String,
    /// Human-readable error message
    pub error_message: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// When the error occurred
    pub occurred_at: DateTime<Utc>,
}

impl Default for FileMetadata {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            full_filepath: PathBuf::new(),
            folder_path: PathBuf::new(),
            filename: String::new(),
            file_extension: None,
            file_type: FileType::File,
            size: 0,
            hash: String::new(),
            creation_date: now,
            modification_date: now,
            access_date: now,
            permissions: 0,
            depth_level: 0,
            relative_path: PathBuf::new(),
            is_hidden: false,
            is_symlink: false,
            symlink_target: None,
            duplicate_status: DuplicateStatus::Unique,
            duplicate_group_id: None,
            processing_status: ProcessingStatus::Pending,
            estimated_tokens: None,
            processed_at: None,
            error_message: None,
        }
    }
}

impl std::fmt::Display for ProcessingStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingStatus::Pending => write!(f, "pending"),
            ProcessingStatus::Processed => write!(f, "processed"),
            ProcessingStatus::Error => write!(f, "error"),
            ProcessingStatus::SkippedBinary => write!(f, "skipped_binary"),
            ProcessingStatus::SkippedDependency => write!(f, "skipped_dependency"),
            ProcessingStatus::Deleted => write!(f, "deleted"),
        }
    }
}

impl std::fmt::Display for DuplicateStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DuplicateStatus::Unique => write!(f, "unique"),
            DuplicateStatus::Canonical => write!(f, "canonical"),
            DuplicateStatus::Duplicate => write!(f, "duplicate"),
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileType::File => write!(f, "file"),
            FileType::Directory => write!(f, "directory"),
        }
    }
}

/// Dependency check information for CLI subcommands
#[derive(Debug, Clone)]
pub struct DependencyCheck {
    /// Name of the dependency
    pub name: &'static str,
    /// Description of what this dependency provides
    pub description: &'static str,
    /// Type of dependency check to perform
    pub check_type: DependencyType,
    /// Whether this dependency is required for basic functionality
    pub required: bool,
    /// Current status of the dependency
    pub status: DependencyStatus,
}

/// Type of dependency to check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyType {
    /// Library or crate dependency
    Library,
    /// System access requirement (file system, network, etc.)
    SystemAccess,
}

/// Status of a dependency check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyStatus {
    /// Dependency is available and working
    Available,
    /// Dependency is missing or not accessible
    Missing,
    /// Error occurred while checking dependency
    Error(String),
    /// Status is unknown or not yet checked
    Unknown,
}