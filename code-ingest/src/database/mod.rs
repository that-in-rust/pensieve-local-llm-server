//! Database operations module
//! 
//! This module provides PostgreSQL connection management, schema operations,
//! and core database functionality for the code ingestion system.

pub mod connection;
pub mod exploration;
pub mod export;
pub mod operations;
pub mod query_executor;
pub mod result_storage;
pub mod schema;
pub mod setup;
pub mod temp_file_manager;

pub use connection::Database;
pub use exploration::{DatabaseExplorer, DatabaseInfo, TableListItem, TableSample, TableSchema};
pub use export::{DatabaseExporter, ExportConfig, ExportResult};
pub use operations::{DatabaseOperations, QueryResult, BatchInsertResult};
pub use query_executor::{QueryExecutor, QueryConfig, FormattedQueryOutput};
pub use result_storage::{ResultStorage, StorageConfig, ResultMetadata, StorageResult};
pub use schema::{SchemaManager, TableType};
pub use setup::{PostgreSQLSetup, SystemInfo, ConnectionTest};
pub use temp_file_manager::{TempFileManager, TempFileConfig, TempFileMetadata, TempFileResult};
