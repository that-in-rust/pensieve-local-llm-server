//! Database operations module
//! 
//! This module provides PostgreSQL connection management, schema operations,
//! and core database functionality for the code ingestion system.

pub mod connection;
pub mod operations;
pub mod query_executor;
pub mod result_storage;
pub mod schema;
pub mod temp_file_manager;

pub use connection::Database;
pub use operations::{QueryResult, BatchInsertResult};
pub use query_executor::{QueryExecutor, QueryConfig, FormattedQueryOutput};
pub use result_storage::{ResultStorage, StorageConfig, ResultMetadata, StorageResult};
pub use schema::{SchemaManager, TableType};
pub use temp_file_manager::{TempFileManager, TempFileConfig, TempFileMetadata, TempFileResult};
