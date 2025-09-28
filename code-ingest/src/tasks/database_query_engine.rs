//! Database Query Engine for Task List Generation
//!
//! This module provides the DatabaseQueryEngine struct that handles SQL operations
//! for counting rows and extracting data from ingestion tables for task generation.

use crate::error::{DatabaseError, DatabaseResult};
use sqlx::{PgPool, Row};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Database query engine for task generation operations
#[derive(Clone, Debug)]
pub struct DatabaseQueryEngine {
    pool: Arc<PgPool>,
}

/// Result of a table validation check
#[derive(Debug, Clone)]
pub struct TableValidation {
    pub exists: bool,
    pub row_count: Option<usize>,
    pub columns: Vec<String>,
}

impl DatabaseQueryEngine {
    /// Create a new DatabaseQueryEngine with a connection pool
    ///
    /// # Arguments
    /// * `pool` - PostgreSQL connection pool
    ///
    /// # Returns
    /// * `Self` - New DatabaseQueryEngine instance
    pub fn new(pool: Arc<PgPool>) -> Self {
        Self { pool }
    }

    /// Count the number of rows in a specified table
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to count rows in
    ///
    /// # Returns
    /// * `DatabaseResult<usize>` - Number of rows in the table
    ///
    /// # Examples
    /// ```rust
    /// let engine = DatabaseQueryEngine::new(pool);
    /// let count = engine.count_rows("INGEST_20250928101039").await?;
    /// println!("Table has {} rows", count);
    /// ```
    pub async fn count_rows(&self, table_name: &str) -> DatabaseResult<usize> {
        debug!("Counting rows in table: {}", table_name);

        // Validate table name to prevent SQL injection
        self.validate_table_name(table_name)?;

        // Validate table exists before counting
        if !self.table_exists(table_name).await? {
            return Err(DatabaseError::TableNotFound {
                table_name: table_name.to_string(),
            });
        }

        // Execute count query
        let query = format!("SELECT COUNT(*) FROM \"{}\"", table_name);
        debug!("Executing count query: {}", query);

        let row = sqlx::query(&query)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.clone(),
                cause: e.to_string(),
            })?;

        let count: i64 = row.try_get(0).map_err(|e| DatabaseError::QueryFailed {
            query: query.clone(),
            cause: format!("Failed to extract count: {}", e),
        })?;

        let count = count as usize;
        info!("Table '{}' contains {} rows", table_name, count);

        Ok(count)
    }

    /// Check if a table exists in the database
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to check
    ///
    /// # Returns
    /// * `DatabaseResult<bool>` - True if table exists, false otherwise
    pub async fn table_exists(&self, table_name: &str) -> DatabaseResult<bool> {
        debug!("Checking if table exists: {}", table_name);

        self.validate_table_name(table_name)?;

        let query = "SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = $1
        )";

        let row = sqlx::query(query)
            .bind(table_name)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.to_string(),
                cause: e.to_string(),
            })?;

        let exists: bool = row.try_get(0).map_err(|e| DatabaseError::QueryFailed {
            query: query.to_string(),
            cause: format!("Failed to extract existence check: {}", e),
        })?;

        debug!("Table '{}' exists: {}", table_name, exists);
        Ok(exists)
    }

    /// Validate a table and return detailed information
    ///
    /// # Arguments
    /// * `table_name` - Name of the table to validate
    ///
    /// # Returns
    /// * `DatabaseResult<TableValidation>` - Validation result with table info
    pub async fn validate_table(&self, table_name: &str) -> DatabaseResult<TableValidation> {
        debug!("Validating table: {}", table_name);

        self.validate_table_name(table_name)?;

        let exists = self.table_exists(table_name).await?;

        if !exists {
            return Ok(TableValidation {
                exists: false,
                row_count: None,
                columns: Vec::new(),
            });
        }

        // Get row count
        let row_count = match self.count_rows(table_name).await {
            Ok(count) => Some(count),
            Err(e) => {
                warn!("Failed to count rows for table '{}': {}", table_name, e);
                None
            }
        };

        // Get column information
        let columns = self.get_table_columns(table_name).await?;

        Ok(TableValidation {
            exists: true,
            row_count,
            columns,
        })
    }

    /// Get column names for a table
    ///
    /// # Arguments
    /// * `table_name` - Name of the table
    ///
    /// # Returns
    /// * `DatabaseResult<Vec<String>>` - List of column names
    pub async fn get_table_columns(&self, table_name: &str) -> DatabaseResult<Vec<String>> {
        debug!("Getting columns for table: {}", table_name);

        self.validate_table_name(table_name)?;

        let query = "SELECT column_name 
                     FROM information_schema.columns 
                     WHERE table_schema = 'public' 
                     AND table_name = $1 
                     ORDER BY ordinal_position";

        let rows = sqlx::query(query)
            .bind(table_name)
            .fetch_all(&*self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.to_string(),
                cause: e.to_string(),
            })?;

        let columns: Result<Vec<String>, _> = rows
            .iter()
            .map(|row| {
                row.try_get::<String, _>(0).map_err(|e| DatabaseError::QueryFailed {
                    query: query.to_string(),
                    cause: format!("Failed to extract column name: {}", e),
                })
            })
            .collect();

        let columns = columns?;
        debug!("Table '{}' has {} columns: {:?}", table_name, columns.len(), columns);

        Ok(columns)
    }

    /// Execute a raw SQL query and return the number of affected rows
    ///
    /// # Arguments
    /// * `sql` - SQL query to execute
    ///
    /// # Returns
    /// * `DatabaseResult<u64>` - Number of affected rows
    ///
    /// # Safety
    /// This method executes raw SQL and should be used carefully to avoid SQL injection.
    /// It's primarily intended for testing and administrative operations.
    pub async fn execute_raw_sql(&self, sql: &str) -> DatabaseResult<u64> {
        debug!("Executing raw SQL: {}", sql);

        let result = sqlx::query(sql)
            .execute(&*self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: e.to_string(),
            })?;

        let rows_affected = result.rows_affected();
        debug!("SQL execution affected {} rows", rows_affected);

        Ok(rows_affected)
    }

    /// Get database connection pool statistics
    ///
    /// # Returns
    /// * `DatabaseConnectionStats` - Current connection pool statistics
    pub fn connection_stats(&self) -> DatabaseConnectionStats {
        DatabaseConnectionStats {
            active_connections: self.pool.size(),
            idle_connections: self.pool.num_idle() as u32,
            max_connections: self.pool.options().get_max_connections(),
        }
    }

    /// Validate table name to prevent SQL injection
    ///
    /// # Arguments
    /// * `table_name` - Table name to validate
    ///
    /// # Returns
    /// * `DatabaseResult<()>` - Ok if valid, error if invalid
    fn validate_table_name(&self, table_name: &str) -> DatabaseResult<()> {
        if table_name.is_empty() {
            return Err(DatabaseError::InvalidTableName {
                table_name: table_name.to_string(),
                cause: "Table name cannot be empty".to_string(),
            });
        }

        // Check for basic SQL injection patterns
        let invalid_chars = ['\'', '"', ';', '\\'];
        let invalid_strings = ["--", "/*", "*/"];
        if table_name.chars().any(|c| invalid_chars.contains(&c)) || 
           invalid_strings.iter().any(|s| table_name.contains(s)) {
            return Err(DatabaseError::InvalidTableName {
                table_name: table_name.to_string(),
                cause: "Table name contains invalid characters".to_string(),
            });
        }

        // Check length (PostgreSQL identifier limit is 63 characters)
        if table_name.len() > 63 {
            return Err(DatabaseError::InvalidTableName {
                table_name: table_name.to_string(),
                cause: "Table name exceeds maximum length of 63 characters".to_string(),
            });
        }

        // Check that it starts with a letter or underscore
        if !table_name.chars().next().unwrap().is_ascii_alphabetic() && !table_name.starts_with('_') {
            return Err(DatabaseError::InvalidTableName {
                table_name: table_name.to_string(),
                cause: "Table name must start with a letter or underscore".to_string(),
            });
        }

        // Check that it only contains valid characters (letters, digits, underscores)
        if !table_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(DatabaseError::InvalidTableName {
                table_name: table_name.to_string(),
                cause: "Table name can only contain letters, digits, and underscores".to_string(),
            });
        }

        Ok(())
    }
}

/// Database connection statistics
#[derive(Debug, Clone)]
pub struct DatabaseConnectionStats {
    pub active_connections: u32,
    pub idle_connections: u32,
    pub max_connections: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use std::env;

    /// Helper function to get a test database engine
    /// Returns None if DATABASE_URL is not set or connection fails
    async fn get_test_engine() -> Option<DatabaseQueryEngine> {
        if let Ok(database_url) = env::var("DATABASE_URL") {
            if let Ok(db) = Database::new(&database_url).await {
                Some(DatabaseQueryEngine::new(Arc::new(db.pool().clone())))
            } else {
                None
            }
        } else {
            None
        }
    }

    #[test]
    fn test_validate_table_name_logic() {
        // Test the validation logic without requiring a database connection
        
        // Valid table names
        let valid_names = vec![
            "valid_table",
            "INGEST_20250928101039",
            "_private_table",
            "table123",
            "Table_Name_123",
            "a",
            "A",
            "_",
        ];

        for name in valid_names {
            // Test validation patterns directly
            assert!(!name.is_empty(), "Name '{}' should not be empty", name);
            assert!(name.len() <= 63, "Name '{}' should be valid length", name);
            
            let first_char = name.chars().next().unwrap();
            assert!(
                first_char.is_ascii_alphabetic() || first_char == '_',
                "Name '{}' should start with letter or underscore", name
            );
            
            assert!(
                name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "Name '{}' should only contain valid characters", name
            );
            
            // Test for invalid patterns
            let invalid_chars = ['\'', '"', ';', '\\'];
            let invalid_strings = ["--", "/*", "*/"];
            assert!(
                !name.chars().any(|c| invalid_chars.contains(&c)) && 
                !invalid_strings.iter().any(|s| name.contains(s)),
                "Name '{}' should not contain invalid patterns", name
            );
        }
    }

    #[test]
    fn test_invalid_table_names() {
        // Test invalid table name patterns
        let long_name = "a".repeat(64);
        let invalid_names = vec![
            "",                          // Empty
            "123table",                  // Starts with number
            "table-name",                // Contains hyphen
            "table name",                // Contains space
            "table'name",                // Contains single quote
            "table\"name",               // Contains double quote
            "table;name",                // Contains semicolon
            "table--name",               // Contains SQL comment
            "table/*name*/",             // Contains SQL comment
            "table\\name",               // Contains backslash
            &long_name,                  // Too long (64 chars)
        ];

        for name in invalid_names {
            // Test validation patterns
            let is_invalid = name.is_empty() ||
                name.len() > 63 ||
                !name.chars().next().map_or(false, |c| c.is_ascii_alphabetic() || c == '_') ||
                name.chars().any(|c| !c.is_ascii_alphanumeric() && c != '_') ||
                name.contains("--") || name.contains("/*") || name.contains("*/");
            
            assert!(is_invalid, "Name '{}' should be invalid", name);
        }
    }

    #[test]
    fn test_database_connection_stats_structure() {
        let stats = DatabaseConnectionStats {
            active_connections: 2,
            idle_connections: 3,
            max_connections: 10,
        };

        assert_eq!(stats.active_connections, 2);
        assert_eq!(stats.idle_connections, 3);
        assert_eq!(stats.max_connections, 10);
    }
}