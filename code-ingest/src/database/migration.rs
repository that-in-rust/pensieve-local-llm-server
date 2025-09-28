//! Database migration system for schema versioning and updates
//!
//! This module provides a comprehensive migration system that tracks schema versions,
//! applies migrations in order, and ensures database consistency across deployments.

use crate::error::{DatabaseError, DatabaseResult};
use sqlx::PgPool;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};

/// Migration manager for database schema versioning
pub struct MigrationManager {
    pool: PgPool,
    migrations: HashMap<String, Migration>,
}

/// Represents a single database migration
#[derive(Debug, Clone)]
pub struct Migration {
    pub version: String,
    pub description: String,
    pub sql: String,
    pub checksum: String,
}

/// Migration execution result
#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub version: String,
    pub applied: bool,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// Migration status information
#[derive(Debug, Clone)]
pub struct MigrationStatus {
    pub version: String,
    pub applied_at: Option<chrono::DateTime<chrono::Utc>>,
    pub checksum: String,
    pub is_pending: bool,
}

impl MigrationManager {
    /// Create a new migration manager
    pub fn new(pool: PgPool) -> Self {
        let mut manager = Self {
            pool,
            migrations: HashMap::new(),
        };
        
        // Register built-in migrations
        manager.register_builtin_migrations();
        manager
    }

    /// Initialize the migration system by creating the schema_migrations table
    pub async fn initialize(&self) -> DatabaseResult<()> {
        info!("Initializing migration system");
        
        let create_migrations_table = r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                checksum VARCHAR(64) NOT NULL
            )
        "#;

        sqlx::query(create_migrations_table)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::MigrationFailed {
                cause: format!("Failed to create schema_migrations table: {}", e),
            })?;

        debug!("Migration system initialized");
        Ok(())
    }

    /// Apply all pending migrations
    pub async fn migrate(&self) -> DatabaseResult<Vec<MigrationResult>> {
        info!("Starting database migration");
        
        // Ensure migration system is initialized
        self.initialize().await?;
        
        // Get current migration status
        let applied_migrations = self.get_applied_migrations().await?;
        let mut results = Vec::new();
        
        // Sort migrations by version for ordered execution
        let mut sorted_migrations: Vec<_> = self.migrations.values().collect();
        sorted_migrations.sort_by(|a, b| a.version.cmp(&b.version));
        
        for migration in sorted_migrations {
            if applied_migrations.contains_key(&migration.version) {
                // Check if checksum matches
                let applied_checksum = &applied_migrations[&migration.version];
                if applied_checksum != &migration.checksum {
                    warn!(
                        "Migration {} checksum mismatch: expected {}, found {}",
                        migration.version, migration.checksum, applied_checksum
                    );
                    return Err(DatabaseError::MigrationFailed {
                        cause: format!(
                            "Migration {} has been modified after application", 
                            migration.version
                        ),
                    });
                }
                
                debug!("Migration {} already applied, skipping", migration.version);
                results.push(MigrationResult {
                    version: migration.version.clone(),
                    applied: false,
                    execution_time_ms: 0,
                    error: None,
                });
                continue;
            }
            
            // Apply the migration
            let start_time = std::time::Instant::now();
            match self.apply_migration(migration).await {
                Ok(()) => {
                    let execution_time = start_time.elapsed().as_millis() as u64;
                    info!(
                        "Applied migration {} in {}ms", 
                        migration.version, execution_time
                    );
                    
                    results.push(MigrationResult {
                        version: migration.version.clone(),
                        applied: true,
                        execution_time_ms: execution_time,
                        error: None,
                    });
                }
                Err(e) => {
                    error!("Failed to apply migration {}: {}", migration.version, e);
                    results.push(MigrationResult {
                        version: migration.version.clone(),
                        applied: false,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        error: Some(e.to_string()),
                    });
                    return Err(e);
                }
            }
        }
        
        info!("Database migration completed successfully");
        Ok(results)
    }

    /// Get the status of all migrations
    pub async fn get_migration_status(&self) -> DatabaseResult<Vec<MigrationStatus>> {
        let applied_migrations = self.get_applied_migrations().await?;
        let mut statuses = Vec::new();
        
        for migration in self.migrations.values() {
            let (applied_at, is_pending) = if let Some(checksum) = applied_migrations.get(&migration.version) {
                // Migration is applied, get the timestamp
                let timestamp = self.get_migration_timestamp(&migration.version).await?;
                (timestamp, checksum != &migration.checksum)
            } else {
                (None, true)
            };
            
            statuses.push(MigrationStatus {
                version: migration.version.clone(),
                applied_at,
                checksum: migration.checksum.clone(),
                is_pending,
            });
        }
        
        // Sort by version
        statuses.sort_by(|a, b| a.version.cmp(&b.version));
        Ok(statuses)
    }

    /// Register a custom migration
    pub fn register_migration(&mut self, migration: Migration) {
        debug!("Registering migration: {}", migration.version);
        self.migrations.insert(migration.version.clone(), migration);
    }

    /// Load migrations from a directory
    pub async fn load_migrations_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> DatabaseResult<usize> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(0);
        }
        
        let mut loaded_count = 0;
        let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
            DatabaseError::MigrationFailed {
                cause: format!("Failed to read migrations directory: {}", e),
            }
        })?;
        
        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            DatabaseError::MigrationFailed {
                cause: format!("Failed to read directory entry: {}", e),
            }
        })? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("sql") {
                if let Some(file_name) = path.file_stem().and_then(|s| s.to_str()) {
                    let sql = tokio::fs::read_to_string(&path).await.map_err(|e| {
                        DatabaseError::MigrationFailed {
                            cause: format!("Failed to read migration file {}: {}", path.display(), e),
                        }
                    })?;
                    
                    let checksum = Self::calculate_checksum(&sql);
                    let migration = Migration {
                        version: file_name.to_string(),
                        description: format!("Migration from file: {}", path.display()),
                        sql,
                        checksum,
                    };
                    
                    self.register_migration(migration);
                    loaded_count += 1;
                }
            }
        }
        
        info!("Loaded {} migrations from {}", loaded_count, dir.display());
        Ok(loaded_count)
    }

    /// Rollback the last applied migration (use with caution)
    pub async fn rollback_last(&self) -> DatabaseResult<Option<String>> {
        warn!("Attempting to rollback last migration");
        
        // Get the most recently applied migration
        let query = r#"
            SELECT version FROM schema_migrations 
            ORDER BY applied_at DESC 
            LIMIT 1
        "#;
        
        let row: Option<(String,)> = sqlx::query_as(query)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.to_string(),
                cause: e.to_string(),
            })?;
        
        if let Some((version,)) = row {
            // Remove the migration record
            let delete_query = "DELETE FROM schema_migrations WHERE version = $1";
            sqlx::query(delete_query)
                .bind(&version)
                .execute(&self.pool)
                .await
                .map_err(|e| DatabaseError::QueryFailed {
                    query: delete_query.to_string(),
                    cause: e.to_string(),
                })?;
            
            warn!("Rolled back migration: {}", version);
            Ok(Some(version))
        } else {
            info!("No migrations to rollback");
            Ok(None)
        }
    }

    // Private helper methods

    fn register_builtin_migrations(&mut self) {
        // Register the initial schema migration
        let initial_migration = Migration {
            version: "001_initial_schema".to_string(),
            description: "Initial schema with ingestion_meta table".to_string(),
            sql: include_str!("../../migrations/001_initial_schema.sql").to_string(),
            checksum: Self::calculate_checksum(include_str!("../../migrations/001_initial_schema.sql")),
        };
        self.register_migration(initial_migration);

        // Register enhanced ingestion tables migration
        let enhanced_migration = Migration {
            version: "002_enhanced_ingestion_tables".to_string(),
            description: "Enhanced ingestion table schema with multi-scale context".to_string(),
            sql: include_str!("../../migrations/002_enhanced_ingestion_tables.sql").to_string(),
            checksum: Self::calculate_checksum(include_str!("../../migrations/002_enhanced_ingestion_tables.sql")),
        };
        self.register_migration(enhanced_migration);

        // Register chunked table support migration
        let chunked_migration = Migration {
            version: "003_chunked_table_support".to_string(),
            description: "Chunked table schema support for large file processing".to_string(),
            sql: include_str!("../../migrations/003_chunked_table_support.sql").to_string(),
            checksum: Self::calculate_checksum(include_str!("../../migrations/003_chunked_table_support.sql")),
        };
        self.register_migration(chunked_migration);
    }

    async fn get_applied_migrations(&self) -> DatabaseResult<HashMap<String, String>> {
        let query = "SELECT version, checksum FROM schema_migrations";
        
        let rows: Vec<(String, String)> = sqlx::query_as(query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.to_string(),
                cause: e.to_string(),
            })?;
        
        Ok(rows.into_iter().collect())
    }

    async fn get_migration_timestamp(&self, version: &str) -> DatabaseResult<Option<chrono::DateTime<chrono::Utc>>> {
        let query = "SELECT applied_at FROM schema_migrations WHERE version = $1";
        
        let row: Option<(chrono::DateTime<chrono::Utc>,)> = sqlx::query_as(query)
            .bind(version)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: query.to_string(),
                cause: e.to_string(),
            })?;
        
        Ok(row.map(|(timestamp,)| timestamp))
    }

    async fn apply_migration(&self, migration: &Migration) -> DatabaseResult<()> {
        debug!("Applying migration: {}", migration.version);
        
        // Start a transaction
        let mut tx = self.pool.begin().await.map_err(|e| {
            DatabaseError::TransactionFailed {
                cause: format!("Failed to start migration transaction: {}", e),
            }
        })?;
        
        // Execute the migration SQL
        sqlx::query(&migration.sql)
            .execute(&mut *tx)
            .await
            .map_err(|e| DatabaseError::MigrationFailed {
                cause: format!("Migration {} failed: {}", migration.version, e),
            })?;
        
        // Record the migration as applied
        let record_query = r#"
            INSERT INTO schema_migrations (version, checksum) 
            VALUES ($1, $2)
            ON CONFLICT (version) DO UPDATE SET 
                checksum = EXCLUDED.checksum,
                applied_at = NOW()
        "#;
        
        sqlx::query(record_query)
            .bind(&migration.version)
            .bind(&migration.checksum)
            .execute(&mut *tx)
            .await
            .map_err(|e| DatabaseError::MigrationFailed {
                cause: format!("Failed to record migration {}: {}", migration.version, e),
            })?;
        
        // Commit the transaction
        tx.commit().await.map_err(|e| {
            DatabaseError::TransactionFailed {
                cause: format!("Failed to commit migration transaction: {}", e),
            }
        })?;
        
        Ok(())
    }

    fn calculate_checksum(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl Migration {
    /// Create a new migration
    pub fn new(version: String, description: String, sql: String) -> Self {
        let checksum = MigrationManager::calculate_checksum(&sql);
        Self {
            version,
            description,
            sql,
            checksum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_pool() -> Option<PgPool> {
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    #[tokio::test]
    async fn test_migration_manager_creation() {
        if let Some(pool) = create_test_pool() {
            let manager = MigrationManager::new(pool);
            assert!(!manager.migrations.is_empty());
            assert!(manager.migrations.contains_key("001_initial_schema"));
        }
    }

    #[tokio::test]
    async fn test_migration_initialization() {
        if let Some(pool) = create_test_pool() {
            let manager = MigrationManager::new(pool);
            let result = manager.initialize().await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_migration_execution() {
        if let Some(pool) = create_test_pool() {
            let manager = MigrationManager::new(pool);
            
            // Initialize and run migrations
            let results = manager.migrate().await.unwrap();
            assert!(!results.is_empty());
            
            // Check that migrations were applied
            let status = manager.get_migration_status().await.unwrap();
            assert!(!status.is_empty());
            
            // Verify some migrations are no longer pending
            let applied_count = status.iter().filter(|s| !s.is_pending).count();
            assert!(applied_count > 0);
        }
    }

    #[tokio::test]
    async fn test_custom_migration_registration() {
        if let Some(pool) = create_test_pool() {
            let mut manager = MigrationManager::new(pool);
            
            let custom_migration = Migration::new(
                "999_test_migration".to_string(),
                "Test migration".to_string(),
                "SELECT 1;".to_string(),
            );
            
            manager.register_migration(custom_migration);
            assert!(manager.migrations.contains_key("999_test_migration"));
        }
    }

    #[tokio::test]
    async fn test_migration_checksum_calculation() {
        let sql = "CREATE TABLE test (id INTEGER);";
        let checksum1 = MigrationManager::calculate_checksum(sql);
        let checksum2 = MigrationManager::calculate_checksum(sql);
        assert_eq!(checksum1, checksum2);
        
        let different_sql = "CREATE TABLE test2 (id INTEGER);";
        let checksum3 = MigrationManager::calculate_checksum(different_sql);
        assert_ne!(checksum1, checksum3);
    }

    #[tokio::test]
    async fn test_migration_from_directory() {
        if let Some(pool) = create_test_pool() {
            let mut manager = MigrationManager::new(pool);
            
            // Create a temporary directory with a test migration
            let temp_dir = TempDir::new().unwrap();
            let migration_file = temp_dir.path().join("004_test_migration.sql");
            tokio::fs::write(&migration_file, "-- Test migration\nSELECT 1;").await.unwrap();
            
            let loaded_count = manager.load_migrations_from_dir(temp_dir.path()).await.unwrap();
            assert_eq!(loaded_count, 1);
            assert!(manager.migrations.contains_key("004_test_migration"));
        }
    }

    #[test]
    fn test_migration_creation() {
        let migration = Migration::new(
            "test_version".to_string(),
            "Test description".to_string(),
            "SELECT 1;".to_string(),
        );
        
        assert_eq!(migration.version, "test_version");
        assert_eq!(migration.description, "Test description");
        assert_eq!(migration.sql, "SELECT 1;");
        assert!(!migration.checksum.is_empty());
    }

    #[tokio::test]
    async fn test_migration_status_tracking() {
        if let Some(pool) = create_test_pool() {
            let manager = MigrationManager::new(pool);
            
            // Initialize and get status
            manager.initialize().await.unwrap();
            let status = manager.get_migration_status().await.unwrap();
            
            // Should have at least the built-in migrations
            assert!(status.len() >= 3);
            
            // Check that status contains expected migrations
            let versions: Vec<&str> = status.iter().map(|s| s.version.as_str()).collect();
            assert!(versions.contains(&"001_initial_schema"));
            assert!(versions.contains(&"002_enhanced_ingestion_tables"));
            assert!(versions.contains(&"003_chunked_table_support"));
        }
    }

    #[tokio::test]
    async fn test_migration_idempotency() {
        if let Some(pool) = create_test_pool() {
            let manager = MigrationManager::new(pool);
            
            // Run migrations twice
            let results1 = manager.migrate().await.unwrap();
            let results2 = manager.migrate().await.unwrap();
            
            // Second run should skip already applied migrations
            let applied_count1 = results1.iter().filter(|r| r.applied).count();
            let applied_count2 = results2.iter().filter(|r| r.applied).count();
            
            // First run should apply migrations, second should skip them
            assert!(applied_count1 >= applied_count2);
        }
    }
}