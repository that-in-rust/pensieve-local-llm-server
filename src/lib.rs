//! Pensieve - A CLI tool for ingesting text files into a deduplicated database for LLM processing
//!
//! This library provides the core functionality for scanning directories, extracting content,
//! and storing deduplicated text data optimized for LLM token efficiency.

pub mod cli;
pub mod database;
pub mod errors;
pub mod extractor;
pub mod scanner;
pub mod types;

// Re-export commonly used types
pub use errors::{PensieveError, Result};
pub use types::{DuplicateStatus, FileMetadata, ProcessingStatus};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        errors::{PensieveError, Result},
        types::{DuplicateStatus, FileMetadata, ProcessingStatus},
    };
}