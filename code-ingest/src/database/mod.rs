//! Database operations module
//! 
//! This module provides PostgreSQL connection management, schema operations,
//! and core database functionality for the code ingestion system.

pub mod connection;
pub mod schema;
pub mod operations;

pub use connection::Database;
pub use schema::{SchemaManager, TableType};
pub use operations::{QueryResult, BatchInsertResult};
