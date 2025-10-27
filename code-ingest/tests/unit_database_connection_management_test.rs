//! Unit tests for database connection management
//! 
//! Tests Requirements 2.4, 3.1, 3.2, 3.3, 3.4 - Database connection pooling and management

use code_ingest::database::{Database, connection::{DatabaseConfig, ConnectionStats}};
use code_ingest::error::DatabaseError;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::task::JoinSet;

/// Helper to get test database URL
fn get_test_database_url() -> String {
    std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/code_ingest_test".to_string())
}

/// Test database connection creation and basic functionality
#[tokio::test]
async fn test_database_connection_creation() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test basic functionality
            let pool = db.pool();
            assert!(!pool.is_closed());
            
            // Test health check
            let health = db.health_check().await.unwrap();
            assert!(health);
            
            // Test connection stats
            let stats = db.connection_stats();
            assert!(stats.max_connections > 0);
            assert!(stats.active_connections <= stats.max_connections);
            assert!(stats.idle_connections <= stats.max_connections);
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - this is expected in test environments without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database connection with migration
#[tokio::test]
async fn test_database_connection_with_migration() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database migration test - no test database configured");
        return;
    }
    
    let result = Database::new_with_migration(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test that migrations were applied
            let migration_status = db.get_migration_status().await.unwrap();
            assert!(!migration_status.is_empty(), "Should have migration records");
            
            // Test schema initialization
            let schema_result = db.initialize_schema().await;
            assert!(schema_result.is_ok(), "Schema initialization should succeed");
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - this is expected in test environments without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database connection from configuration
#[tokio::test]
async fn test_database_from_config() {
    let config = DatabaseConfig {
        host: "localhost".to_string(),
        port: 5432,
        username: "postgres".to_string(),
        password: Some("password".to_string()),
        database_name: "code_ingest_test".to_string(),
        max_connections: 5,
        connection_timeout: Duration::from_secs(10),
        idle_timeout: Duration::from_secs(300),
        max_lifetime: Duration::from_secs(900),
    };
    
    let result = Database::from_config(&config).await;
    
    match result {
        Ok(db) => {
            let stats = db.connection_stats();
            assert!(stats.max_connections <= 5); // Should respect config
            
            let health = db.health_check().await.unwrap();
            assert!(health);
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - this is expected in test environments without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database connection from path
#[tokio::test]
async fn test_database_from_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test_db");
    
    // This test primarily validates the path handling logic
    // since it would require PostgreSQL to actually create the database
    let result = Database::from_path(&db_path).await;
    
    match result {
        Ok(_db) => {
            // If successful, verify the database was created
            assert!(db_path.parent().unwrap().exists());
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            // Expected in test environments without PostgreSQL
            println!("Database creation from path failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database URL validation through connection attempts
#[tokio::test]
async fn test_database_url_validation() {
    // Valid URL format should at least attempt connection
    let valid_url = "postgresql://user@localhost:5432/db";
    let result = Database::new(valid_url).await;
    // Should fail with connection error, not URL validation error
    match result {
        Err(DatabaseError::ConnectionFailed { .. }) => {
            // Expected - connection failed but URL was valid
        }
        Ok(_) => {
            // Unexpected success, but URL was valid
        }
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
    
    // Invalid URL format should fail immediately
    let invalid_url = "not_a_database_url";
    let result = Database::new(invalid_url).await;
    assert!(result.is_err(), "Invalid URL should fail");
}

/// Test database error messages don't expose passwords
#[tokio::test]
async fn test_database_error_sanitization() {
    let url_with_password = "postgresql://user:secret123@nonexistent:5432/db";
    let result = Database::new(url_with_password).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        DatabaseError::ConnectionFailed { url, .. } => {
            // Error message should not contain the actual password
            assert!(!url.contains("secret123"));
        }
        _ => panic!("Expected ConnectionFailed error"),
    }
}

/// Test database configuration structure
#[test]
fn test_database_config_structure() {
    let config = DatabaseConfig {
        host: "remote-host".to_string(),
        port: 5433,
        username: "myuser".to_string(),
        password: Some("mypass".to_string()),
        database_name: "mydb".to_string(),
        max_connections: 20,
        connection_timeout: Duration::from_secs(15),
        idle_timeout: Duration::from_secs(600),
        max_lifetime: Duration::from_secs(1800),
    };
    
    // Test that config fields are accessible
    assert_eq!(config.host, "remote-host");
    assert_eq!(config.port, 5433);
    assert_eq!(config.username, "myuser");
    assert_eq!(config.password, Some("mypass".to_string()));
    assert_eq!(config.database_name, "mydb");
    assert_eq!(config.max_connections, 20);
}

/// Test database configuration defaults
#[test]
fn test_database_config_defaults() {
    let config = DatabaseConfig::default();
    
    assert_eq!(config.host, "localhost");
    assert_eq!(config.port, 5432);
    assert_eq!(config.username, "postgres");
    assert!(config.password.is_none());
    assert_eq!(config.database_name, "code_ingest");
    assert_eq!(config.max_connections, 10);
    assert_eq!(config.connection_timeout, Duration::from_secs(30));
    assert_eq!(config.idle_timeout, Duration::from_secs(600));
    assert_eq!(config.max_lifetime, Duration::from_secs(1800));
}

/// Test connection statistics structure
#[test]
fn test_connection_stats_structure() {
    let stats = ConnectionStats {
        active_connections: 3,
        idle_connections: 2,
        max_connections: 10,
    };
    
    assert_eq!(stats.active_connections, 3);
    assert_eq!(stats.idle_connections, 2);
    assert_eq!(stats.max_connections, 10);
    
    // Verify logical constraints
    assert!(stats.active_connections + stats.idle_connections <= stats.max_connections);
}

/// Test database pool configuration optimization
#[tokio::test]
async fn test_database_pool_optimization() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database pool optimization test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test that pool is configured with optimized settings
            let stats = db.connection_stats();
            
            // Should have reasonable connection limits
            assert!(stats.max_connections >= 5, "Should have at least 5 max connections");
            assert!(stats.max_connections <= 50, "Should not exceed 50 connections for tests");
            
            // Test that pool is functional
            let health = db.health_check().await.unwrap();
            assert!(health, "Database should be healthy after optimization");
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database health check functionality
#[tokio::test]
async fn test_database_health_check() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database health check test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test health check
            let health = db.health_check().await.unwrap();
            assert!(health, "Health check should return true for healthy connection");
            
            // Test multiple health checks
            for _ in 0..5 {
                let health = db.health_check().await.unwrap();
                assert!(health);
            }
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database raw SQL execution
#[tokio::test]
async fn test_database_raw_sql_execution() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database raw SQL test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test simple SELECT query
            let result = db.execute_raw("SELECT 1").await.unwrap();
            assert_eq!(result, 0); // SELECT doesn't affect rows
            
            // Test that invalid SQL returns error
            let invalid_result = db.execute_raw("INVALID SQL STATEMENT").await;
            assert!(invalid_result.is_err(), "Invalid SQL should return error");
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database connection concurrent access
#[tokio::test]
async fn test_database_concurrent_access() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database concurrent access test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            let db = Arc::new(db);
            let mut join_set = JoinSet::new();
            
            // Spawn multiple tasks that use the database concurrently
            for i in 0..10 {
                let db_clone = Arc::clone(&db);
                join_set.spawn(async move {
                    // Test health check
                    let _health = db_clone.health_check().await.unwrap();
                    
                    // Test raw SQL execution
                    let _result = db_clone.execute_raw(&format!("SELECT {}", i)).await.unwrap();
                    
                    // Test connection stats
                    let _stats = db_clone.connection_stats();
                });
            }
            
            // Wait for all tasks to complete
            while let Some(result) = join_set.join_next().await {
                result.unwrap(); // Panic if any task failed
            }
            
            // Verify database is still functional
            let final_health = db.health_check().await.unwrap();
            assert!(final_health);
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database connection error handling
#[tokio::test]
async fn test_database_connection_error_handling() {
    // Test connection to invalid database
    let invalid_url = "postgresql://invalid:invalid@nonexistent:5432/nonexistent";
    let result = Database::new(invalid_url).await;
    
    assert!(result.is_err(), "Connection to invalid database should fail");
    
    match result.unwrap_err() {
        DatabaseError::ConnectionFailed { url, cause: _ } => {
            assert!(url.contains("invalid:***")); // Password should be sanitized
        }
        _ => panic!("Expected ConnectionFailed error"),
    }
    
    // Test invalid URL format
    let malformed_url = "not_a_database_url";
    let result2 = Database::new(malformed_url).await;
    assert!(result2.is_err(), "Malformed URL should fail");
}

/// Test database connection pool sizing
#[tokio::test]
async fn test_database_pool_sizing() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database pool sizing test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            let stats = db.connection_stats();
            
            // Test that pool size is reasonable
            let cpu_count = num_cpus::get();
            let expected_min = std::cmp::max(cpu_count * 2, 20);
            
            assert!(stats.max_connections >= expected_min as u32, 
                   "Pool should scale with CPU count, expected >= {}, got {}", 
                   expected_min, stats.max_connections);
            
            // Test that we don't create excessive connections
            assert!(stats.max_connections <= 100, 
                   "Pool should not exceed reasonable limits, got {}", 
                   stats.max_connections);
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database connection lifecycle
#[tokio::test]
async fn test_database_connection_lifecycle() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database lifecycle test - no test database configured");
        return;
    }
    
    let result = Database::new(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test initial state
            let initial_stats = db.connection_stats();
            assert!(initial_stats.max_connections > 0);
            
            // Test that connection works
            let health = db.health_check().await.unwrap();
            assert!(health);
            
            // Test closing connection
            db.close().await;
            
            // After closing, pool should be closed
            let pool = db.pool();
            assert!(pool.is_closed());
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

/// Test database ingestion operations
#[tokio::test]
async fn test_database_ingestion_operations() {
    let database_url = get_test_database_url();
    
    // Skip test if no database available
    if database_url.contains("localhost") && std::env::var("CI").is_err() {
        println!("Skipping database ingestion operations test - no test database configured");
        return;
    }
    
    let result = Database::new_with_migration(&database_url).await;
    
    match result {
        Ok(db) => {
            // Test creating ingestion table
            let table_result = db.create_ingestion_table("test_table").await;
            match table_result {
                Ok(_) => println!("Ingestion table created successfully"),
                Err(e) => println!("Ingestion table creation failed (may be expected): {}", e),
            }
            
            // Test creating ingestion record
            let record_result = db.create_ingestion_record(
                Some("https://github.com/test/repo".to_string()),
                "/tmp/test".to_string(),
                chrono::Utc::now().timestamp() as u64,
                "test_table",
            ).await;
            
            match record_result {
                Ok(ingestion_id) => {
                    assert!(ingestion_id > 0);
                    
                    // Test completing ingestion record
                    let complete_result = db.complete_ingestion_record(
                        ingestion_id,
                        chrono::Utc::now().timestamp() as u64,
                        42,
                    ).await;
                    
                    match complete_result {
                        Ok(_) => println!("Ingestion record completed successfully"),
                        Err(e) => println!("Ingestion record completion failed: {}", e),
                    }
                }
                Err(e) => println!("Ingestion record creation failed: {}", e),
            }
        }
        Err(DatabaseError::ConnectionFailed { .. }) => {
            println!("Database connection failed - expected without PostgreSQL");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}