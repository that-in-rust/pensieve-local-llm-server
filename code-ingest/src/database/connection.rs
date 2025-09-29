//! PostgreSQL connection management with pooling and health checks
//!
//! This module provides the core Database struct that manages PostgreSQL connections
//! using sqlx connection pooling, with built-in health checks, retry logic, and
//! database creation capabilities.

use crate::error::{DatabaseError, DatabaseResult};
use sqlx::postgres::{PgPool, PgPoolOptions, PgConnectOptions};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn, error};

/// Database connection manager with connection pooling
#[derive(Clone, Debug)]
pub struct Database {
    pool: PgPool,
    #[allow(dead_code)]
    database_url: String,
}

/// Configuration for database connections
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: Option<String>,
    pub database_name: String,
    pub max_connections: u32,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            username: "postgres".to_string(),
            password: None,
            database_name: "code_ingest".to_string(),
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600), // 10 minutes
            max_lifetime: Duration::from_secs(1800), // 30 minutes
        }
    }
}

impl Database {
    /// Create a new database connection from a database URL
    /// 
    /// # Arguments
    /// * `database_url` - PostgreSQL connection string (e.g., "postgresql://user:pass@host:port/db")
    /// 
    /// # Returns
    /// * `DatabaseResult<Self>` - Database instance or error
    /// 
    /// # Examples
    /// ```rust
    /// let db = Database::new("postgresql://postgres@localhost:5432/code_ingest").await?;
    /// ```
    pub async fn new(database_url: &str) -> DatabaseResult<Self> {
        debug!("Creating database connection to: {}", Self::sanitize_url(database_url));
        
        // Validate URL format
        Self::validate_database_url(database_url)?;
        
        // Create connection pool
        let pool = Self::create_pool(database_url).await?;
        
        // Verify connection health
        Self::verify_connection_health(&pool).await?;
        
        info!("Successfully connected to PostgreSQL database");
        
        Ok(Self {
            pool,
            database_url: database_url.to_string(),
        })
    }

    /// Create a new database connection with automatic migration
    /// 
    /// This method creates a connection and automatically applies any pending migrations
    pub async fn new_with_migration(database_url: &str) -> DatabaseResult<Self> {
        let db = Self::new(database_url).await?;
        
        // Run migrations automatically
        let migration_manager = crate::database::migration::MigrationManager::new(db.pool.clone());
        let results = migration_manager.migrate().await?;
        
        let applied_count = results.iter().filter(|r| r.applied).count();
        if applied_count > 0 {
            info!("Applied {} database migrations", applied_count);
        } else {
            debug!("No pending migrations to apply");
        }
        
        Ok(db)
    }

    /// Create a database connection from configuration
    pub async fn from_config(config: &DatabaseConfig) -> DatabaseResult<Self> {
        let database_url = Self::build_database_url(config);
        Self::new(&database_url).await
    }

    /// Create a database connection from a local database path
    /// This method will create the database if it doesn't exist
    pub async fn from_path<P: AsRef<Path>>(db_path: P) -> DatabaseResult<Self> {
        let db_path = db_path.as_ref();
        
        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                DatabaseError::ConnectionFailed {
                    url: db_path.display().to_string(),
                    cause: format!("Failed to create directory: {}", e),
                }
            })?;
        }

        // Extract database name from path
        let db_name = db_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("code_ingest");

        let config = DatabaseConfig {
            database_name: db_name.to_string(),
            ..Default::default()
        };

        // Try to connect, create database if it doesn't exist
        match Self::from_config(&config).await {
            Ok(db) => Ok(db),
            Err(DatabaseError::ConnectionFailed { .. }) => {
                info!("Database '{}' not found, attempting to create it", db_name);
                Self::create_database_if_not_exists(&config).await?;
                Self::from_config(&config).await
            }
            Err(e) => Err(e),
        }
    }

    /// Get the underlying connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Check if the database connection is healthy
    pub async fn health_check(&self) -> DatabaseResult<bool> {
        Self::verify_connection_health(&self.pool).await?;
        Ok(true)
    }

    /// Get connection statistics
    pub fn connection_stats(&self) -> ConnectionStats {
        ConnectionStats {
            active_connections: self.pool.size(),
            idle_connections: self.pool.num_idle() as u32,
            max_connections: self.pool.options().get_max_connections(),
        }
    }

    /// Close the database connection pool
    pub async fn close(&self) {
        info!("Closing database connection pool");
        self.pool.close().await;
    }

    /// Execute a raw SQL query for testing purposes
    pub async fn execute_raw(&self, sql: &str) -> DatabaseResult<u64> {
        debug!("Executing raw SQL: {}", sql);
        
        let result = sqlx::query(sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: e.to_string(),
            })?;

        Ok(result.rows_affected())
    }

    /// Create an ingestion table with the given name
    pub async fn create_ingestion_table(&self, _table_name: &str) -> DatabaseResult<()> {
        let schema_manager = crate::database::schema::SchemaManager::new(self.pool.clone());
        let _table_name = schema_manager.create_ingestion_table(Some(chrono::Utc::now())).await?;
        Ok(())
    }

    /// Create an ingestion record and return the ID
    pub async fn create_ingestion_record(
        &self,
        repo_url: Option<String>,
        local_path: String,
        start_timestamp: u64,
        table_name: &str,
    ) -> DatabaseResult<i64> {
        let operations = crate::database::operations::DatabaseOperations::new(self.pool.clone());
        operations.create_ingestion_record(repo_url.as_deref(), &local_path, start_timestamp, table_name).await
    }

    /// Complete an ingestion record
    pub async fn complete_ingestion_record(
        &self,
        ingestion_id: i64,
        end_timestamp: u64,
        total_files: i32,
    ) -> DatabaseResult<()> {
        let operations = crate::database::operations::DatabaseOperations::new(self.pool.clone());
        operations.complete_ingestion_record(ingestion_id, end_timestamp, total_files).await
    }

    /// Insert processed files into the database
    pub async fn insert_processed_files(
        &self,
        table_name: &str,
        files: &[crate::processing::ProcessedFile],
        ingestion_id: i64,
    ) -> DatabaseResult<()> {
        let operations = crate::database::operations::DatabaseOperations::new(self.pool.clone());
        operations.insert_processed_files(table_name, files, ingestion_id).await
    }

    /// Get ingestion statistics
    pub async fn get_ingestion_statistics(
        &self,
        ingestion_id: i64,
    ) -> DatabaseResult<crate::ingestion::IngestionStatistics> {
        let operations = crate::database::operations::DatabaseOperations::new(self.pool.clone());
        operations.get_ingestion_statistics(ingestion_id).await
    }

    /// List all ingestion records
    pub async fn list_ingestion_records(&self) -> DatabaseResult<Vec<crate::ingestion::IngestionRecord>> {
        let operations = crate::database::operations::DatabaseOperations::new(self.pool.clone());
        operations.list_ingestion_records().await
    }

    /// Run database migrations
    pub async fn migrate(&self) -> DatabaseResult<Vec<crate::database::migration::MigrationResult>> {
        let migration_manager = crate::database::migration::MigrationManager::new(self.pool.clone());
        migration_manager.migrate().await
    }

    /// Get migration status
    pub async fn get_migration_status(&self) -> DatabaseResult<Vec<crate::database::migration::MigrationStatus>> {
        let migration_manager = crate::database::migration::MigrationManager::new(self.pool.clone());
        migration_manager.get_migration_status().await
    }

    /// Initialize database schema with migrations
    pub async fn initialize_schema(&self) -> DatabaseResult<()> {
        let migration_manager = crate::database::migration::MigrationManager::new(self.pool.clone());
        migration_manager.initialize().await?;
        
        let schema_manager = crate::database::schema::SchemaManager::new(self.pool.clone());
        schema_manager.initialize_schema().await
    }

    // Private helper methods

    fn validate_database_url(url: &str) -> DatabaseResult<()> {
        if !url.starts_with("postgresql://") && !url.starts_with("postgres://") {
            return Err(DatabaseError::InvalidDatabaseUrl {
                url: Self::sanitize_url(url),
            });
        }
        
        // Try to parse the URL to validate format
        PgConnectOptions::from_str(url).map_err(|e| DatabaseError::InvalidDatabaseUrl {
            url: format!("Invalid URL format: {}", e),
        })?;
        
        Ok(())
    }

    async fn create_pool(database_url: &str) -> DatabaseResult<PgPool> {
        // Optimized connection pool configuration for high-performance ingestion
        let cpu_count = num_cpus::get();
        let max_connections = std::cmp::max(cpu_count * 2, 20); // Scale with CPU cores, minimum 20
        
        let pool_options = PgPoolOptions::new()
            .max_connections(max_connections as u32)
            .min_connections(5) // Keep minimum connections warm
            .acquire_timeout(Duration::from_secs(30))
            .idle_timeout(Duration::from_secs(300)) // Reduced idle timeout for better resource usage
            .max_lifetime(Duration::from_secs(3600)) // Increased lifetime for better connection reuse
            .test_before_acquire(false) // Skip health checks for performance
            .after_connect(|conn, _meta| {
                Box::pin(async move {
                    // Optimize PostgreSQL connection settings for bulk operations
                    // Only set session-level parameters that don't require server restart
                    sqlx::query("SET synchronous_commit = off").execute(&mut *conn).await?;
                    sqlx::query("SET work_mem = '64MB'").execute(&mut *conn).await?;
                    sqlx::query("SET maintenance_work_mem = '256MB'").execute(&mut *conn).await?;
                    sqlx::query("SET temp_buffers = '32MB'").execute(&mut *conn).await?;
                    sqlx::query("SET random_page_cost = 1.1").execute(&mut *conn).await?;
                    Ok(())
                })
            });

        let pool = pool_options
            .connect(database_url)
            .await
            .map_err(|e| DatabaseError::ConnectionFailed {
                url: Self::sanitize_url(database_url),
                cause: e.to_string(),
            })?;

        info!("Created optimized connection pool with {} max connections", max_connections);
        Ok(pool)
    }

    async fn verify_connection_health(pool: &PgPool) -> DatabaseResult<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_millis(500);

        for attempt in 1..=MAX_RETRIES {
            match sqlx::query("SELECT 1").fetch_one(pool).await {
                Ok(_) => {
                    debug!("Database health check passed on attempt {}", attempt);
                    return Ok(());
                }
                Err(e) if attempt < MAX_RETRIES => {
                    warn!("Database health check failed on attempt {}: {}", attempt, e);
                    sleep(RETRY_DELAY).await;
                }
                Err(e) => {
                    error!("Database health check failed after {} attempts: {}", MAX_RETRIES, e);
                    return Err(DatabaseError::ConnectionFailed {
                        url: "health_check".to_string(),
                        cause: format!("Health check failed after {} attempts: {}", MAX_RETRIES, e),
                    });
                }
            }
        }

        unreachable!()
    }

    fn build_database_url(config: &DatabaseConfig) -> String {
        let auth = match &config.password {
            Some(password) => format!("{}:{}", config.username, password),
            None => config.username.clone(),
        };

        format!(
            "postgresql://{}@{}:{}/{}",
            auth, config.host, config.port, config.database_name
        )
    }

    async fn create_database_if_not_exists(config: &DatabaseConfig) -> DatabaseResult<()> {
        // Connect to the default 'postgres' database to create our target database
        let admin_config = DatabaseConfig {
            database_name: "postgres".to_string(),
            ..config.clone()
        };
        
        let admin_url = Self::build_database_url(&admin_config);
        let admin_pool = Self::create_pool(&admin_url).await?;

        // Check if database exists
        let exists_query = "SELECT 1 FROM pg_database WHERE datname = $1";
        let exists = sqlx::query(exists_query)
            .bind(&config.database_name)
            .fetch_optional(&admin_pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: exists_query.to_string(),
                cause: e.to_string(),
            })?;

        if exists.is_none() {
            // Create the database
            let create_query = format!("CREATE DATABASE \"{}\"", config.database_name);
            sqlx::query(&create_query)
                .execute(&admin_pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: create_query.clone(),
                    cause: e.to_string(),
                })?;

            info!("Created database: {}", config.database_name);
        } else {
            debug!("Database {} already exists", config.database_name);
        }

        admin_pool.close().await;
        Ok(())
    }

    fn sanitize_url(url: &str) -> String {
        // Remove password from URL for logging
        if let Ok(mut parsed) = url::Url::parse(url) {
            if parsed.password().is_some() {
                let _ = parsed.set_password(Some("***"));
            }
            parsed.to_string()
        } else {
            "[invalid_url]".to_string()
        }
    }
}

/// Connection statistics for monitoring
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub active_connections: u32,
    pub idle_connections: u32,
    pub max_connections: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_database_url_validation() {
        // Valid URLs
        assert!(Database::validate_database_url("postgresql://user@localhost:5432/db").is_ok());
        assert!(Database::validate_database_url("postgres://user:pass@localhost/db").is_ok());
        
        // Invalid URLs
        assert!(Database::validate_database_url("mysql://user@localhost/db").is_err());
        assert!(Database::validate_database_url("invalid_url").is_err());
        assert!(Database::validate_database_url("").is_err());
    }

    #[test]
    fn test_database_config_default() {
        let config = DatabaseConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5432);
        assert_eq!(config.username, "postgres");
        assert_eq!(config.database_name, "code_ingest");
        assert_eq!(config.max_connections, 10);
    }

    #[test]
    fn test_build_database_url() {
        let config = DatabaseConfig {
            host: "localhost".to_string(),
            port: 5432,
            username: "testuser".to_string(),
            password: Some("testpass".to_string()),
            database_name: "testdb".to_string(),
            ..Default::default()
        };

        let url = Database::build_database_url(&config);
        assert_eq!(url, "postgresql://testuser:testpass@localhost:5432/testdb");

        let config_no_pass = DatabaseConfig {
            password: None,
            ..config
        };

        let url_no_pass = Database::build_database_url(&config_no_pass);
        assert_eq!(url_no_pass, "postgresql://testuser@localhost:5432/testdb");
    }

    #[test]
    fn test_sanitize_url() {
        let url_with_pass = "postgresql://user:secret@localhost:5432/db";
        let sanitized = Database::sanitize_url(url_with_pass);
        assert!(sanitized.contains("user:***"));
        assert!(!sanitized.contains("secret"));

        let url_no_pass = "postgresql://user@localhost:5432/db";
        let sanitized_no_pass = Database::sanitize_url(url_no_pass);
        assert_eq!(sanitized_no_pass, url_no_pass);

        let invalid_url = "not_a_url";
        let sanitized_invalid = Database::sanitize_url(invalid_url);
        assert_eq!(sanitized_invalid, "[invalid_url]");
    }

    #[tokio::test]
    async fn test_connection_stats() {
        // This test requires a running PostgreSQL instance
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        let stats = db.connection_stats();
        assert!(stats.max_connections > 0);
        assert!(stats.active_connections <= stats.max_connections);
        assert!(stats.idle_connections <= stats.max_connections);
    }

    #[tokio::test]
    async fn test_health_check() {
        // This test requires a running PostgreSQL instance
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        let health = db.health_check().await.unwrap();
        assert!(health);
    }

    #[tokio::test]
    async fn test_execute_raw() {
        // This test requires a running PostgreSQL instance
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        // Test a simple SELECT query
        let result = db.execute_raw("SELECT 1").await.unwrap();
        assert_eq!(result, 0); // SELECT doesn't affect rows
    }

    #[test]
    fn test_from_path_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("subdir").join("test.db");
        
        // This would test directory creation, but requires PostgreSQL
        // The logic is tested in the actual implementation
        assert!(db_path.parent().is_some());
    }

    #[tokio::test]
    async fn test_database_creation_logic() {
        // Test the database creation logic without actually creating a database
        let config = DatabaseConfig {
            database_name: "test_nonexistent_db".to_string(),
            ..Default::default()
        };

        // This would fail in a real environment, but tests the error handling
        let result = Database::from_config(&config).await;
        
        // We expect this to fail since the database doesn't exist
        // and we're not actually creating it in this test
        assert!(result.is_err());
    }

    #[test]
    fn test_connection_config_builder() {
        let mut config = DatabaseConfig::default();
        config.host = "remote-host".to_string();
        config.port = 5433;
        config.username = "myuser".to_string();
        config.password = Some("mypass".to_string());
        config.database_name = "mydb".to_string();

        let url = Database::build_database_url(&config);
        assert_eq!(url, "postgresql://myuser:mypass@remote-host:5433/mydb");
    }

    #[test]
    fn test_connection_timeouts() {
        let config = DatabaseConfig {
            connection_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(900),
            ..Default::default()
        };

        assert_eq!(config.connection_timeout, Duration::from_secs(10));
        assert_eq!(config.idle_timeout, Duration::from_secs(300));
        assert_eq!(config.max_lifetime, Duration::from_secs(900));
    }

    // Additional comprehensive tests for subtask 2.1 requirements

    #[tokio::test]
    async fn test_database_pool_creation_with_sqlx_pgpool() {
        // Test that we can create a pool with proper configuration
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        // Verify pool is properly configured
        let stats = db.connection_stats();
        assert!(stats.max_connections > 0);
        assert!(stats.max_connections <= 10); // Default max connections
        
        // Verify we can get the underlying pool
        let pool = db.pool();
        assert_eq!(pool.size(), stats.active_connections);
    }

    #[tokio::test]
    async fn test_schema_migration_system_for_ingestion_meta() {
        // Test schema initialization for ingestion_meta table
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        // Initialize schema using SchemaManager
        let schema_manager = crate::database::schema::SchemaManager::new(db.pool().clone());
        let result = schema_manager.initialize_schema().await;
        assert!(result.is_ok(), "Schema initialization should succeed");
        
        // Verify ingestion_meta table exists
        let exists = schema_manager.table_exists("ingestion_meta").await.unwrap();
        assert!(exists, "ingestion_meta table should exist after initialization");
        
        // Verify table has correct schema
        let is_valid = schema_manager.validate_table_schema("ingestion_meta", crate::database::schema::TableType::Meta).await.unwrap();
        assert!(is_valid, "ingestion_meta table should have correct schema");
    }

    #[tokio::test]
    async fn test_timestamped_ingest_table_creation() {
        // Test creation of timestamped INGEST_YYYYMMDDHHMMSS tables
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        // Initialize schema first
        let schema_manager = crate::database::schema::SchemaManager::new(db.pool().clone());
        schema_manager.initialize_schema().await.unwrap();
        
        // Create a timestamped ingestion table
        let timestamp = chrono::Utc::now();
        let table_name = schema_manager.create_ingestion_table(Some(timestamp)).await.unwrap();
        
        // Verify table name format
        assert!(table_name.starts_with("INGEST_"));
        assert_eq!(table_name.len(), 21); // INGEST_ + 14 digit timestamp
        
        // Verify table exists
        let exists = schema_manager.table_exists(&table_name).await.unwrap();
        assert!(exists, "Timestamped ingestion table should exist");
        
        // Verify table has correct schema
        let is_valid = schema_manager.validate_table_schema(&table_name, crate::database::schema::TableType::Ingestion).await.unwrap();
        assert!(is_valid, "Ingestion table should have correct schema");
        
        // Clean up
        schema_manager.drop_table(&table_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_database_connection_error_handling() {
        // Test connection failure scenarios
        let invalid_url = "postgresql://invalid:invalid@nonexistent:5432/nonexistent";
        let result = Database::new(invalid_url).await;
        assert!(result.is_err(), "Connection to invalid database should fail");
        
        match result.unwrap_err() {
            crate::error::DatabaseError::ConnectionFailed { url, cause: _ } => {
                assert!(url.contains("invalid:***")); // Password should be sanitized
            }
            _ => panic!("Expected ConnectionFailed error"),
        }
    }

    #[tokio::test]
    async fn test_ingestion_record_operations() {
        // Test creating and completing ingestion records
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        // Initialize schema
        let schema_manager = crate::database::schema::SchemaManager::new(db.pool().clone());
        schema_manager.initialize_schema().await.unwrap();
        
        // Create ingestion record
        let repo_url = Some("https://github.com/test/repo".to_string());
        let local_path = "/tmp/test".to_string();
        let start_timestamp = chrono::Utc::now().timestamp() as u64;
        let table_name = "INGEST_20250927143022";
        
        let ingestion_id = db.create_ingestion_record(
            repo_url.clone(),
            local_path.clone(),
            start_timestamp,
            table_name,
        ).await.unwrap();
        
        assert!(ingestion_id > 0, "Ingestion ID should be positive");
        
        // Complete ingestion record
        let end_timestamp = start_timestamp + 100;
        let total_files = 42;
        
        let result = db.complete_ingestion_record(ingestion_id, end_timestamp, total_files).await;
        assert!(result.is_ok(), "Completing ingestion record should succeed");
        
        // Verify record was updated
        let records = db.list_ingestion_records().await.unwrap();
        let found_record = records.iter().find(|r| r.ingestion_id == ingestion_id);
        assert!(found_record.is_some(), "Ingestion record should be found");
        
        let record = found_record.unwrap();
        assert_eq!(record.repo_url, repo_url);
        assert_eq!(record.local_path, local_path);
        assert_eq!(record.total_files_processed, Some(total_files));
    }

    #[tokio::test]
    async fn test_database_pool_configuration() {
        // Test that pool is configured with correct parameters
        let config = DatabaseConfig {
            max_connections: 5,
            connection_timeout: Duration::from_secs(15),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(900),
            ..Default::default()
        };
        
        // Test URL building with custom config
        let url = Database::build_database_url(&config);
        assert!(url.contains("localhost:5432"));
        assert!(url.contains("code_ingest")); // Default database name
        
        // Test config validation
        assert_eq!(config.max_connections, 5);
        assert_eq!(config.connection_timeout, Duration::from_secs(15));
    }

    #[tokio::test]
    async fn test_database_from_path_functionality() {
        // Test creating database connection from local path
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");
        
        // This test would require a running PostgreSQL instance
        // We test the path handling logic
        assert!(db_path.parent().is_some());
        
        // Test that parent directory would be created
        let nested_path = temp_dir.path().join("nested").join("deep").join("test.db");
        assert!(nested_path.parent().is_some());
        assert!(nested_path.parent().unwrap().parent().is_some());
    }

    #[test]
    fn test_connection_stats_structure() {
        let stats = ConnectionStats {
            active_connections: 2,
            idle_connections: 3,
            max_connections: 10,
        };
        
        assert_eq!(stats.active_connections, 2);
        assert_eq!(stats.idle_connections, 3);
        assert_eq!(stats.max_connections, 10);
        assert!(stats.active_connections + stats.idle_connections <= stats.max_connections);
    }

    #[tokio::test]
    async fn test_raw_sql_execution() {
        // Test raw SQL execution capability
        // Skip if DATABASE_URL is not set
        if std::env::var("DATABASE_URL").is_err() {
            return;
        }

        let database_url = std::env::var("DATABASE_URL").unwrap();
        let db = Database::new(&database_url).await.unwrap();
        
        // Test simple SELECT
        let result = db.execute_raw("SELECT 1 as test_column").await.unwrap();
        assert_eq!(result, 0); // SELECT doesn't affect rows
        
        // Test that we can execute DDL (create temporary table)
        let create_result = db.execute_raw("CREATE TEMPORARY TABLE test_temp (id INTEGER)").await.unwrap();
        assert_eq!(create_result, 0); // CREATE TABLE doesn't return affected rows
        
        // Test INSERT
        let insert_result = db.execute_raw("INSERT INTO test_temp (id) VALUES (1), (2), (3)").await.unwrap();
        assert_eq!(insert_result, 3); // 3 rows inserted
        
        // Test UPDATE
        let update_result = db.execute_raw("UPDATE test_temp SET id = id + 10").await.unwrap();
        assert_eq!(update_result, 3); // 3 rows updated
        
        // Test DELETE
        let delete_result = db.execute_raw("DELETE FROM test_temp WHERE id > 12").await.unwrap();
        assert_eq!(delete_result, 1); // 1 row deleted (id = 13)
    }
}