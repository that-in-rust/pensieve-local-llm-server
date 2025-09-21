//! Database operations and schema management

use crate::prelude::*;
use crate::types::{FileMetadata, Paragraph, ParagraphSource, ProcessingError};
use sqlx::{SqlitePool, Sqlite, Transaction};
use std::path::Path;
use std::time::Duration;

/// Database manager for SQLite operations with connection pooling
#[derive(Clone)]
pub struct Database {
    /// SQLite connection pool with optimized settings
    pool: SqlitePool,
}

impl Database {
    /// Create new database connection with optimized pool settings
    pub async fn new(database_path: &Path) -> Result<Self> {
        let database_url = format!("sqlite:{}", database_path.display());
        
        // Create connection pool with optimized settings for CLI tool usage
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(10) // Reasonable for CLI tool
            .min_connections(1)  // Keep at least one connection alive
            .acquire_timeout(Duration::from_secs(30))
            .idle_timeout(Duration::from_secs(600)) // 10 minutes
            .max_lifetime(Duration::from_secs(1800)) // 30 minutes
            .connect(&database_url)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        let db = Self { pool };
        
        // Configure SQLite for optimal performance and safety
        db.configure_sqlite().await?;
        
        // Ensure schema version table exists
        db.create_schema_version_table().await?;
        
        Ok(db)
    }

    /// Configure SQLite settings for optimal performance and safety
    async fn configure_sqlite(&self) -> Result<()> {
        // Enable WAL mode for better concurrency and crash safety
        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        // Enable foreign key constraints
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        // Optimize for performance
        sqlx::query("PRAGMA synchronous = NORMAL") // Balance safety and performance
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        sqlx::query("PRAGMA cache_size = -64000") // 64MB cache
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        sqlx::query("PRAGMA temp_store = MEMORY") // Use memory for temp tables
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        sqlx::query("PRAGMA mmap_size = 268435456") // 256MB memory map
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Initialize database schema with migration support
    pub async fn initialize_schema(&self) -> Result<()> {
        // Create schema version table first
        self.create_schema_version_table().await?;
        
        // Run migrations to ensure schema is up to date
        let migration_manager = MigrationManager::new(self.clone());
        migration_manager.migrate().await?;
        
        Ok(())
    }

    /// Create schema version tracking table
    async fn create_schema_version_table(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        // Insert initial version if table is empty
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM schema_version")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        if count == 0 {
            sqlx::query(
                "INSERT INTO schema_version (version, description) VALUES (0, 'Initial schema')"
            )
            .execute(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        }

        Ok(())
    }

    /// Create files table with comprehensive metadata tracking
    pub async fn create_files_table(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                full_filepath TEXT NOT NULL UNIQUE,
                folder_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_extension TEXT,
                file_type TEXT NOT NULL CHECK(file_type IN ('file', 'directory')),
                size INTEGER NOT NULL CHECK(size >= 0),
                hash TEXT NOT NULL,
                creation_date TIMESTAMP,
                modification_date TIMESTAMP,
                access_date TIMESTAMP,
                permissions INTEGER,
                depth_level INTEGER NOT NULL CHECK(depth_level >= 0),
                relative_path TEXT NOT NULL,
                is_hidden BOOLEAN NOT NULL DEFAULT FALSE,
                is_symlink BOOLEAN NOT NULL DEFAULT FALSE,
                symlink_target TEXT,
                duplicate_status TEXT NOT NULL DEFAULT 'unique' 
                    CHECK(duplicate_status IN ('unique', 'canonical', 'duplicate')),
                duplicate_group_id TEXT,
                processing_status TEXT NOT NULL DEFAULT 'pending' 
                    CHECK(processing_status IN ('pending', 'processed', 'error', 'skipped_binary', 'skipped_dependency', 'deleted')),
                estimated_tokens INTEGER CHECK(estimated_tokens >= 0),
                processed_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Create paragraphs table for deduplicated content storage
    pub async fn create_paragraphs_table(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS paragraphs (
                paragraph_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL CHECK(length(content) > 0),
                estimated_tokens INTEGER NOT NULL CHECK(estimated_tokens > 0),
                word_count INTEGER NOT NULL CHECK(word_count >= 0),
                char_count INTEGER NOT NULL CHECK(char_count > 0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Create paragraph_sources table for many-to-many relationships
    pub async fn create_paragraph_sources_table(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS paragraph_sources (
                paragraph_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                paragraph_index INTEGER NOT NULL CHECK(paragraph_index >= 0),
                byte_offset_start INTEGER NOT NULL CHECK(byte_offset_start >= 0),
                byte_offset_end INTEGER NOT NULL CHECK(byte_offset_end > byte_offset_start),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (paragraph_id, file_id, paragraph_index),
                FOREIGN KEY (paragraph_id) REFERENCES paragraphs(paragraph_id) ON DELETE CASCADE,
                FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Create processing_errors table for error tracking and debugging
    pub async fn create_processing_errors_table(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS processing_errors (
                error_id TEXT PRIMARY KEY,
                file_id TEXT,
                error_type TEXT NOT NULL CHECK(length(error_type) > 0),
                error_message TEXT NOT NULL CHECK(length(error_message) > 0),
                stack_trace TEXT,
                occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE SET NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Create comprehensive database indexes for optimal query performance
    pub async fn create_indexes(&self) -> Result<()> {
        let indexes = vec![
            // Files table indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash)",
            "CREATE INDEX IF NOT EXISTS idx_files_duplicate_group ON files(duplicate_group_id) WHERE duplicate_group_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_files_processing_status ON files(processing_status)",
            "CREATE INDEX IF NOT EXISTS idx_files_modification_date ON files(modification_date)",
            "CREATE INDEX IF NOT EXISTS idx_files_file_extension ON files(file_extension) WHERE file_extension IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_files_size ON files(size)",
            "CREATE INDEX IF NOT EXISTS idx_files_duplicate_status ON files(duplicate_status)",
            "CREATE INDEX IF NOT EXISTS idx_files_processed_at ON files(processed_at) WHERE processed_at IS NOT NULL",
            
            // Paragraphs table indexes
            "CREATE INDEX IF NOT EXISTS idx_paragraphs_hash ON paragraphs(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_paragraphs_tokens ON paragraphs(estimated_tokens)",
            "CREATE INDEX IF NOT EXISTS idx_paragraphs_created_at ON paragraphs(created_at)",
            
            // Paragraph sources table indexes for joins
            "CREATE INDEX IF NOT EXISTS idx_paragraph_sources_file ON paragraph_sources(file_id)",
            "CREATE INDEX IF NOT EXISTS idx_paragraph_sources_paragraph ON paragraph_sources(paragraph_id)",
            "CREATE INDEX IF NOT EXISTS idx_paragraph_sources_index ON paragraph_sources(paragraph_index)",
            
            // Processing errors table indexes
            "CREATE INDEX IF NOT EXISTS idx_processing_errors_file ON processing_errors(file_id) WHERE file_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_processing_errors_type ON processing_errors(error_type)",
            "CREATE INDEX IF NOT EXISTS idx_processing_errors_occurred_at ON processing_errors(occurred_at)",
            
            // Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_files_status_type ON files(processing_status, file_type)",
            "CREATE INDEX IF NOT EXISTS idx_files_duplicate_hash ON files(duplicate_status, hash)",
        ];

        for index_sql in indexes {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| PensieveError::Database(e))?;
        }

        Ok(())
    }

    /// Insert file metadata
    pub async fn insert_file(&self, _metadata: &FileMetadata) -> Result<()> {
        // TODO: Implement file insertion
        // This will be implemented in a later task
        Ok(())
    }

    /// Update file metadata
    pub async fn update_file(&self, _metadata: &FileMetadata) -> Result<()> {
        // TODO: Implement file update
        // This will be implemented in a later task
        Ok(())
    }

    /// Get file by path
    pub async fn get_file_by_path(&self, _path: &Path) -> Result<Option<FileMetadata>> {
        // TODO: Implement file retrieval
        // This will be implemented in a later task
        Ok(None)
    }

    /// Insert paragraph
    pub async fn insert_paragraph(&self, _paragraph: &Paragraph) -> Result<()> {
        // TODO: Implement paragraph insertion
        // This will be implemented in a later task
        Ok(())
    }

    /// Insert paragraph source link
    pub async fn insert_paragraph_source(&self, _source: &ParagraphSource) -> Result<()> {
        // TODO: Implement paragraph source insertion
        // This will be implemented in a later task
        Ok(())
    }

    /// Get paragraph by hash
    pub async fn get_paragraph_by_hash(&self, _hash: &str) -> Result<Option<Paragraph>> {
        // TODO: Implement paragraph retrieval
        // This will be implemented in a later task
        Ok(None)
    }

    /// Insert processing error
    pub async fn insert_error(&self, _error: &ProcessingError) -> Result<()> {
        // TODO: Implement error insertion
        // This will be implemented in a later task
        Ok(())
    }

    /// Get processing statistics
    pub async fn get_statistics(&self) -> Result<DatabaseStatistics> {
        // TODO: Implement statistics calculation
        // This will be implemented in a later task
        Ok(DatabaseStatistics::default())
    }

    /// Begin transaction for batch operations
    pub async fn begin_transaction(&self) -> Result<Transaction<'_, Sqlite>> {
        self.pool.begin().await
            .map_err(|e| PensieveError::Database(e))
    }



    /// Get connection pool statistics
    pub fn pool_stats(&self) -> PoolStats {
        PoolStats {
            size: self.pool.size(),
            idle: self.pool.num_idle(),
        }
    }

    /// Check if database connection is healthy
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        Ok(())
    }

    /// Get current schema version
    pub async fn get_schema_version(&self) -> Result<u32> {
        let version: i64 = sqlx::query_scalar(
            "SELECT MAX(version) FROM schema_version"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(version as u32)
    }

    /// Close database connection pool gracefully
    pub async fn close(self) {
        self.pool.close().await;
    }
}

/// Database statistics for monitoring and reporting
#[derive(Debug, Default)]
pub struct DatabaseStatistics {
    /// Total number of files
    pub total_files: u64,
    /// Number of unique files (non-duplicates)
    pub unique_files: u64,
    /// Number of duplicate files
    pub duplicate_files: u64,
    /// Total number of paragraphs
    pub total_paragraphs: u64,
    /// Total estimated tokens
    pub total_tokens: u64,
    /// Number of processing errors
    pub error_count: u64,
    /// Files by processing status
    pub files_by_status: std::collections::HashMap<String, u64>,
}

/// Connection pool statistics
#[derive(Debug)]
pub struct PoolStats {
    /// Total number of connections in the pool
    pub size: u32,
    /// Number of idle connections
    pub idle: usize,
}

/// Database migration manager for schema evolution
pub struct MigrationManager {
    /// Database connection
    database: Database,
}

impl MigrationManager {
    /// Create new migration manager
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Run all pending migrations
    pub async fn migrate(&self) -> Result<()> {
        let current_version = self.get_current_version().await?;
        let target_version = self.get_target_version();

        if current_version >= target_version {
            return Ok(()); // Already up to date
        }

        // Run migrations sequentially
        for version in (current_version + 1)..=target_version {
            self.apply_migration(version).await?;
        }

        Ok(())
    }

    /// Get current schema version from database
    async fn get_current_version(&self) -> Result<u32> {
        // Ensure schema_version table exists first
        self.database.create_schema_version_table().await?;
        self.database.get_schema_version().await
    }

    /// Get target schema version (latest available)
    fn get_target_version(&self) -> u32 {
        // Current target version - increment when adding new migrations
        1
    }

    /// Apply a specific migration
    async fn apply_migration(&self, version: u32) -> Result<()> {
        match version {
            1 => self.migrate_to_v1().await?,
            _ => return Err(PensieveError::Migration(
                format!("Unknown migration version: {}", version)
            )),
        }

        // Record successful migration
        sqlx::query(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)"
        )
        .bind(version as i64)
        .bind(self.get_migration_description(version))
        .execute(&self.database.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Migration to version 1: Create initial schema
    async fn migrate_to_v1(&self) -> Result<()> {
        // Create all tables
        self.database.create_files_table().await?;
        self.database.create_paragraphs_table().await?;
        self.database.create_paragraph_sources_table().await?;
        self.database.create_processing_errors_table().await?;
        
        // Create all indexes
        self.database.create_indexes().await?;

        Ok(())
    }

    /// Get description for a migration version
    fn get_migration_description(&self, version: u32) -> &'static str {
        match version {
            1 => "Create initial schema with files, paragraphs, paragraph_sources, and processing_errors tables",
            _ => "Unknown migration",
        }
    }

    /// Rollback to a specific version (for development/testing)
    pub async fn rollback_to(&self, target_version: u32) -> Result<()> {
        let current_version = self.get_current_version().await?;
        
        if target_version >= current_version {
            return Ok(()); // Nothing to rollback
        }

        // For now, we only support rollback to version 0 (drop all tables)
        if target_version == 0 {
            self.drop_all_tables().await?;
            
            // Remove version records
            sqlx::query("DELETE FROM schema_version WHERE version > 0")
                .execute(&self.database.pool)
                .await
                .map_err(|e| PensieveError::Database(e))?;
        } else {
            return Err(PensieveError::Migration(
                format!("Rollback to version {} not supported", target_version)
            ));
        }

        Ok(())
    }

    /// Drop all tables (for complete reset)
    async fn drop_all_tables(&self) -> Result<()> {
        let tables = vec![
            "DROP TABLE IF EXISTS paragraph_sources",
            "DROP TABLE IF EXISTS processing_errors", 
            "DROP TABLE IF EXISTS paragraphs",
            "DROP TABLE IF EXISTS files",
        ];

        for drop_sql in tables {
            sqlx::query(drop_sql)
                .execute(&self.database.pool)
                .await
                .map_err(|e| PensieveError::Database(e))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;


    async fn create_test_database() -> Result<(Database, NamedTempFile)> {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path();
        let db = Database::new(db_path).await?;
        Ok((db, temp_file))
    }

    #[tokio::test]
    async fn test_database_creation_and_initialization() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        
        // Test schema initialization
        db.initialize_schema().await.unwrap();
        
        // Verify schema version
        let version = db.get_schema_version().await.unwrap();
        assert_eq!(version, 1);
        
        // Test health check
        db.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_wal_mode_enabled() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        
        // Check that WAL mode is enabled
        let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        
        assert_eq!(journal_mode.to_lowercase(), "wal");
    }

    #[tokio::test]
    async fn test_foreign_keys_enabled() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        
        // Check that foreign keys are enabled
        let foreign_keys: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        
        assert_eq!(foreign_keys, 1);
    }

    #[tokio::test]
    async fn test_migration_system() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Initial version should be 0
        let initial_version = migration_manager.get_current_version().await.unwrap();
        assert_eq!(initial_version, 0);
        
        // Run migrations
        migration_manager.migrate().await.unwrap();
        
        // Version should now be 1
        let final_version = migration_manager.get_current_version().await.unwrap();
        assert_eq!(final_version, 1);
        
        // Running migrations again should be idempotent
        migration_manager.migrate().await.unwrap();
        let version_after_second_run = migration_manager.get_current_version().await.unwrap();
        assert_eq!(version_after_second_run, 1);
    }

    #[tokio::test]
    async fn test_table_creation() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        db.initialize_schema().await.unwrap();
        
        // Verify all tables exist by querying their structure
        let tables = vec!["files", "paragraphs", "paragraph_sources", "processing_errors", "schema_version"];
        
        for table in tables {
            let count: i64 = sqlx::query_scalar(&format!(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'", 
                table
            ))
            .fetch_one(&db.pool)
            .await
            .unwrap();
            
            assert_eq!(count, 1, "Table {} should exist", table);
        }
    }

    #[tokio::test]
    async fn test_indexes_creation() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        db.initialize_schema().await.unwrap();
        
        // Verify that indexes were created
        let index_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        .fetch_one(&db.pool)
        .await
        .unwrap();
        
        // Should have created multiple indexes
        assert!(index_count > 10, "Should have created multiple indexes, found {}", index_count);
    }

    #[tokio::test]
    async fn test_transaction_support() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        db.initialize_schema().await.unwrap();
        
        // Test manual transaction
        let mut tx = db.begin_transaction().await.unwrap();
        
        sqlx::query("INSERT INTO schema_version (version, description) VALUES (999, 'test')")
            .execute(&mut *tx)
            .await
            .unwrap();
        
        tx.commit().await.unwrap();
        
        // Verify the insert was committed
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM schema_version WHERE version = 999")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let stats = db.pool_stats();
        
        // Pool should have at least one connection
        assert!(stats.size > 0);
        assert!(stats.idle <= stats.size as usize);
    }

    #[tokio::test]
    async fn test_rollback_functionality() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Apply migrations
        migration_manager.migrate().await.unwrap();
        
        // Verify tables exist
        let table_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('files', 'paragraphs')"
        )
        .fetch_one(&db.pool)
        .await
        .unwrap();
        assert_eq!(table_count, 2);
        
        // Rollback to version 0
        migration_manager.rollback_to(0).await.unwrap();
        
        // Verify tables are dropped
        let table_count_after: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('files', 'paragraphs')"
        )
        .fetch_one(&db.pool)
        .await
        .unwrap();
        assert_eq!(table_count_after, 0);
    }
}