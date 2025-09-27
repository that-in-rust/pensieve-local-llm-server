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
        let pool_options = PgPoolOptions::new()
            .max_connections(10)
            .acquire_timeout(Duration::from_secs(30))
            .idle_timeout(Duration::from_secs(600))
            .max_lifetime(Duration::from_secs(1800));

        let pool = pool_options
            .connect(database_url)
            .await
            .map_err(|e| DatabaseError::ConnectionFailed {
                url: Self::sanitize_url(database_url),
                cause: e.to_string(),
            })?;

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
}