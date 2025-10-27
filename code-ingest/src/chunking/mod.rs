//! File chunking module for handling large files that exceed tsvector limits
//! 
//! This module provides automatic file chunking capabilities to handle files
//! that are too large for PostgreSQL's tsvector field (1MB limit).
//! 
//! # Architecture
//! 
//! Following L1→L2→L3 layered architecture:
//! - L1 Core: Traits and error types
//! - L2 Standard: File operations and validation
//! - L3 External: txt-sectumsempra integration

pub mod error;
pub mod traits;
pub mod chunker;
pub mod validator;

pub use error::{ChunkingError, ChunkingResult};
pub use traits::{FileChunker, ChunkValidator, ChunkingConfig};
pub use chunker::TxtSectumsempraChunker;
pub use validator::ChecksumValidator;

/// Default chunk size in MB (safe margin under 1MB tsvector limit)
pub const DEFAULT_CHUNK_SIZE_MB: f64 = 0.8;

/// Size threshold for determining if a file should be chunked
pub const TSVECTOR_SAFE_LIMIT_BYTES: u64 = 800_000; // 800KB