//! Database operations and schema management

use crate::prelude::*;
use crate::types::{FileMetadata, Paragraph, ParagraphSource, ProcessingError, FileType, DuplicateStatus, ProcessingStatus, ParagraphId, FileId};
use sqlx::{SqlitePool, Sqlite, Transaction};
use std::path::{Path, PathBuf};
use std::time::Duration;
use chrono::{DateTime, Utc};

/// Database manager for SQLite operations with connection pooling
#[derive(Clone)]
pub struct Database {
    /// SQLite connection pool with optimized settings
    pool: SqlitePool,
}

impl Database {
    /// Create new database connection with optimized pool settings
    pub async fn new(database_path: &Path) -> Result<Self> {
        // Ensure the parent directory exists
        if let Some(parent) = database_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| PensieveError::Io(e))?;
            }
        }
        
        let database_url = format!("sqlite://{}?mode=rwc", database_path.display());
        
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
    pub async fn insert_file(&self, metadata: &FileMetadata) -> Result<()> {
        let file_id = uuid::Uuid::new_v4().to_string();
        let full_filepath = metadata.full_filepath.to_string_lossy().to_string();
        let folder_path = metadata.folder_path.to_string_lossy().to_string();
        let file_type = metadata.file_type.to_string();
        let size = metadata.size as i64;
        let permissions = metadata.permissions as i64;
        let depth_level = metadata.depth_level as i64;
        let relative_path = metadata.relative_path.to_string_lossy().to_string();
        let symlink_target = metadata.symlink_target.as_ref().map(|p| p.to_string_lossy().to_string());
        let duplicate_status = metadata.duplicate_status.to_string();
        let duplicate_group_id = metadata.duplicate_group_id.map(|id| id.to_string());
        let processing_status = metadata.processing_status.to_string();
        let estimated_tokens = metadata.estimated_tokens.map(|t| t as i64);
        
        sqlx::query!(
            r#"
            INSERT INTO files (
                file_id, full_filepath, folder_path, filename, file_extension, file_type,
                size, hash, creation_date, modification_date, access_date, permissions,
                depth_level, relative_path, is_hidden, is_symlink, symlink_target,
                duplicate_status, duplicate_group_id, processing_status, estimated_tokens,
                processed_at, error_message
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            "#,
            file_id,
            full_filepath,
            folder_path,
            metadata.filename,
            metadata.file_extension,
            file_type,
            size,
            metadata.hash,
            metadata.creation_date,
            metadata.modification_date,
            metadata.access_date,
            permissions,
            depth_level,
            relative_path,
            metadata.is_hidden,
            metadata.is_symlink,
            symlink_target,
            duplicate_status,
            duplicate_group_id,
            processing_status,
            estimated_tokens,
            metadata.processed_at,
            metadata.error_message
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(())
    }

    /// Update file processing status and token count
    pub async fn update_file_processing_status(
        &self,
        file_path: &std::path::Path,
        status: ProcessingStatus,
        estimated_tokens: Option<u32>,
        error_message: Option<String>,
    ) -> Result<()> {
        let path_str = file_path.to_string_lossy().to_string();
        let status_str = status.to_string();
        let tokens = estimated_tokens.map(|t| t as i64);
        let processed_at = if status == ProcessingStatus::Processed {
            Some(chrono::Utc::now())
        } else {
            None
        };
        
        sqlx::query!(
            r#"
            UPDATE files SET
                processing_status = ?,
                estimated_tokens = ?,
                processed_at = ?,
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE full_filepath = ?
            "#,
            status_str,
            tokens,
            processed_at,
            error_message,
            path_str
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(())
    }

    /// Update file metadata
    pub async fn update_file(&self, metadata: &FileMetadata) -> Result<()> {
        let folder_path = metadata.folder_path.to_string_lossy().to_string();
        let file_type = metadata.file_type.to_string();
        let size = metadata.size as i64;
        let permissions = metadata.permissions as i64;
        let depth_level = metadata.depth_level as i64;
        let relative_path = metadata.relative_path.to_string_lossy().to_string();
        let symlink_target = metadata.symlink_target.as_ref().map(|p| p.to_string_lossy().to_string());
        let duplicate_status = metadata.duplicate_status.to_string();
        let duplicate_group_id = metadata.duplicate_group_id.map(|id| id.to_string());
        let processing_status = metadata.processing_status.to_string();
        let estimated_tokens = metadata.estimated_tokens.map(|t| t as i64);
        let full_filepath = metadata.full_filepath.to_string_lossy().to_string();
        
        sqlx::query!(
            r#"
            UPDATE files SET
                folder_path = ?, filename = ?, file_extension = ?, file_type = ?,
                size = ?, hash = ?, creation_date = ?, modification_date = ?, access_date = ?,
                permissions = ?, depth_level = ?, relative_path = ?, is_hidden = ?,
                is_symlink = ?, symlink_target = ?, duplicate_status = ?, duplicate_group_id = ?,
                processing_status = ?, estimated_tokens = ?, processed_at = ?, error_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE full_filepath = ?
            "#,
            folder_path,
            metadata.filename,
            metadata.file_extension,
            file_type,
            size,
            metadata.hash,
            metadata.creation_date,
            metadata.modification_date,
            metadata.access_date,
            permissions,
            depth_level,
            relative_path,
            metadata.is_hidden,
            metadata.is_symlink,
            symlink_target,
            duplicate_status,
            duplicate_group_id,
            processing_status,
            estimated_tokens,
            metadata.processed_at,
            metadata.error_message,
            full_filepath
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(())
    }

    /// Get file ID by path
    pub async fn get_file_id_by_path(&self, path: &Path) -> Result<Option<FileId>> {
        let path_str = path.to_string_lossy().to_string();
        
        let row = sqlx::query!(
            "SELECT file_id FROM files WHERE full_filepath = ?",
            path_str
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        if let Some(row) = row {
            let file_id = row.file_id.as_ref()
                .and_then(|id_str| uuid::Uuid::parse_str(id_str).ok())
                .map(FileId);
            Ok(file_id)
        } else {
            Ok(None)
        }
    }

    /// Get file by path
    pub async fn get_file_by_path(&self, path: &Path) -> Result<Option<FileMetadata>> {
        let path_str = path.to_string_lossy().to_string();
        
        let row = sqlx::query!(
            r#"
            SELECT file_id, full_filepath, folder_path, filename, file_extension, file_type,
                   size, hash, creation_date, modification_date, access_date, permissions,
                   depth_level, relative_path, is_hidden, is_symlink, symlink_target,
                   duplicate_status, duplicate_group_id, processing_status, estimated_tokens,
                   processed_at, error_message
            FROM files 
            WHERE full_filepath = ?
            "#,
            path_str
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        if let Some(row) = row {
            let duplicate_status = match row.duplicate_status.as_str() {
                "unique" => DuplicateStatus::Unique,
                "canonical" => DuplicateStatus::Canonical,
                "duplicate" => DuplicateStatus::Duplicate,
                _ => DuplicateStatus::Unique,
            };
            
            let processing_status = match row.processing_status.as_str() {
                "pending" => ProcessingStatus::Pending,
                "processed" => ProcessingStatus::Processed,
                "error" => ProcessingStatus::Error,
                "skipped_binary" => ProcessingStatus::SkippedBinary,
                "skipped_dependency" => ProcessingStatus::SkippedDependency,
                "deleted" => ProcessingStatus::Deleted,
                _ => ProcessingStatus::Pending,
            };
            
            let file_type = match row.file_type.as_str() {
                "file" => FileType::File,
                "directory" => FileType::Directory,
                _ => FileType::File,
            };
            
            // Convert NaiveDateTime to DateTime<Utc>
            let creation_date = row.creation_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let modification_date = row.modification_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let access_date = row.access_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let processed_at = row.processed_at
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
            
            let metadata = FileMetadata {
                full_filepath: PathBuf::from(row.full_filepath),
                folder_path: PathBuf::from(row.folder_path),
                filename: row.filename,
                file_extension: row.file_extension,
                file_type,
                size: row.size as u64,
                hash: row.hash,
                creation_date,
                modification_date,
                access_date,
                permissions: row.permissions.unwrap_or(0) as u32,
                depth_level: row.depth_level as u32,
                relative_path: PathBuf::from(row.relative_path),
                is_hidden: row.is_hidden,
                is_symlink: row.is_symlink,
                symlink_target: row.symlink_target.map(PathBuf::from),
                duplicate_status,
                duplicate_group_id: row.duplicate_group_id.as_ref().and_then(|s| uuid::Uuid::parse_str(s).ok()),
                processing_status,
                estimated_tokens: row.estimated_tokens.map(|t| t as u32),
                processed_at,
                error_message: row.error_message,
            };
            
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    /// Insert paragraph with deduplication check
    pub async fn insert_paragraph(&self, paragraph: &Paragraph) -> Result<()> {
        let paragraph_id = paragraph.id.0.to_string();
        let estimated_tokens = paragraph.estimated_tokens as i64;
        let word_count = paragraph.word_count as i64;
        let char_count = paragraph.char_count as i64;
        
        sqlx::query!(
            r#"
            INSERT INTO paragraphs (
                paragraph_id, content_hash, content, estimated_tokens, 
                word_count, char_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_hash) DO NOTHING
            "#,
            paragraph_id,
            paragraph.content_hash,
            paragraph.content,
            estimated_tokens,
            word_count,
            char_count,
            paragraph.created_at
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(())
    }

    /// Insert paragraph source link for file-paragraph relationships
    pub async fn insert_paragraph_source(&self, source: &ParagraphSource) -> Result<()> {
        let paragraph_id = source.paragraph_id.0.to_string();
        let file_id = source.file_id.0.to_string();
        let paragraph_index = source.paragraph_index as i64;
        let byte_offset_start = source.byte_offset_start as i64;
        let byte_offset_end = source.byte_offset_end as i64;
        
        sqlx::query!(
            r#"
            INSERT INTO paragraph_sources (
                paragraph_id, file_id, paragraph_index, 
                byte_offset_start, byte_offset_end, created_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(paragraph_id, file_id, paragraph_index) DO NOTHING
            "#,
            paragraph_id,
            file_id,
            paragraph_index,
            byte_offset_start,
            byte_offset_end
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(())
    }

    /// Get paragraph by hash for deduplication checks
    pub async fn get_paragraph_by_hash(&self, hash: &str) -> Result<Option<Paragraph>> {
        let row = sqlx::query!(
            r#"
            SELECT paragraph_id, content_hash, content, estimated_tokens,
                   word_count, char_count, created_at
            FROM paragraphs 
            WHERE content_hash = ?
            "#,
            hash
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        if let Some(row) = row {
            let paragraph_id_str = row.paragraph_id.ok_or_else(|| 
                PensieveError::InvalidData("Paragraph ID is null".to_string()))?;
            let paragraph_id = uuid::Uuid::parse_str(&paragraph_id_str)
                .map_err(|e| PensieveError::InvalidData(format!("Invalid paragraph ID: {}", e)))?;
            
            // Convert NaiveDateTime to DateTime<Utc>
            let created_at = row.created_at
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            
            let paragraph = Paragraph {
                id: ParagraphId(paragraph_id),
                content_hash: row.content_hash,
                content: row.content,
                estimated_tokens: row.estimated_tokens as u32,
                word_count: row.word_count as u32,
                char_count: row.char_count as u32,
                created_at,
            };
            
            Ok(Some(paragraph))
        } else {
            Ok(None)
        }
    }

    /// Insert processing error for tracking and debugging
    pub async fn insert_error(&self, error: &ProcessingError) -> Result<()> {
        let error_id = error.id.to_string();
        let file_id = error.file_id.map(|id| id.0.to_string());
        
        sqlx::query!(
            r#"
            INSERT INTO processing_errors (
                error_id, file_id, error_type, error_message, 
                stack_trace, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            "#,
            error_id,
            file_id,
            error.error_type,
            error.error_message,
            error.stack_trace,
            error.occurred_at
        )
        .execute(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(())
    }

    /// Get processing statistics
    pub async fn get_statistics(&self) -> Result<DatabaseStatistics> {
        // Get total file counts
        let total_files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        
        // Get unique files (unique + canonical)
        let unique_files: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM files WHERE duplicate_status IN ('unique', 'canonical')"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        // Get duplicate files
        let duplicate_files: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM files WHERE duplicate_status = 'duplicate'"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        // Get paragraph count (unique paragraphs stored)
        let total_paragraphs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM paragraphs")
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?
            .unwrap_or(0);
        
        // Get total tokens from paragraphs (more accurate than file estimates)
        let total_tokens: i64 = sqlx::query_scalar(
            "SELECT COALESCE(SUM(estimated_tokens), 0) FROM paragraphs"
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?
        .unwrap_or(0);
        
        // Get error count
        let error_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM processing_errors")
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?
            .unwrap_or(0);
        
        // Get files by status using raw query to avoid type issues
        let status_rows = sqlx::query_as::<_, (String, i64)>(
            "SELECT processing_status, COUNT(*) as count FROM files GROUP BY processing_status"
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let mut files_by_status = std::collections::HashMap::new();
        for (status, count) in status_rows {
            files_by_status.insert(status, count as u64);
        }
        
        Ok(DatabaseStatistics {
            total_files: total_files as u64,
            unique_files: unique_files as u64,
            duplicate_files: duplicate_files as u64,
            total_paragraphs: total_paragraphs as u64,
            total_tokens: total_tokens as u64,
            error_count: error_count as u64,
            files_by_status,
        })
    }

    /// Get paragraph-level deduplication statistics
    pub async fn get_paragraph_statistics(&self) -> Result<ParagraphStatistics> {
        // Get total paragraph sources (all paragraph instances across files)
        let total_paragraph_sources: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM paragraph_sources")
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?
            .unwrap_or(0);
        
        // Get unique paragraphs count
        let unique_paragraphs: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM paragraphs")
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?
            .unwrap_or(0);
        
        // Get total tokens from unique paragraphs
        let total_tokens: i64 = sqlx::query_scalar(
            "SELECT COALESCE(SUM(estimated_tokens), 0) FROM paragraphs"
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?
        .unwrap_or(0);
        
        // Get average paragraph length
        let avg_paragraph_length: f64 = sqlx::query_scalar(
            "SELECT COALESCE(AVG(char_count), 0.0) FROM paragraphs"
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?
        .unwrap_or(0.0);
        
        // Calculate deduplication metrics
        let deduplicated_paragraphs = if total_paragraph_sources > unique_paragraphs {
            total_paragraph_sources - unique_paragraphs
        } else {
            0
        };
        
        let deduplication_rate = if total_paragraph_sources > 0 {
            (deduplicated_paragraphs as f64 / total_paragraph_sources as f64) * 100.0
        } else {
            0.0
        };
        
        Ok(ParagraphStatistics {
            total_paragraph_instances: total_paragraph_sources as u64,
            unique_paragraphs: unique_paragraphs as u64,
            deduplicated_paragraphs: deduplicated_paragraphs as u64,
            deduplication_rate,
            total_tokens: total_tokens as u64,
            average_paragraph_length: avg_paragraph_length,
        })
    }

    /// Begin transaction for batch operations
    pub async fn begin_transaction(&self) -> Result<Transaction<'_, Sqlite>> {
        self.pool.begin().await
            .map_err(|e| PensieveError::Database(e))
    }

    /// Insert multiple files in a batch transaction
    pub async fn insert_files_batch(&self, files: &[FileMetadata]) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }

        let mut tx = self.begin_transaction().await?;
        
        for metadata in files {
            let file_id = uuid::Uuid::new_v4().to_string();
            let full_filepath = metadata.full_filepath.to_string_lossy().to_string();
            let folder_path = metadata.folder_path.to_string_lossy().to_string();
            let file_type = metadata.file_type.to_string();
            let size = metadata.size as i64;
            let permissions = metadata.permissions as i64;
            let depth_level = metadata.depth_level as i64;
            let relative_path = metadata.relative_path.to_string_lossy().to_string();
            let symlink_target = metadata.symlink_target.as_ref().map(|p| p.to_string_lossy().to_string());
            let duplicate_status = metadata.duplicate_status.to_string();
            let duplicate_group_id = metadata.duplicate_group_id.map(|id| id.to_string());
            let processing_status = metadata.processing_status.to_string();
            let estimated_tokens = metadata.estimated_tokens.map(|t| t as i64);
            
            sqlx::query!(
                r#"
                INSERT INTO files (
                    file_id, full_filepath, folder_path, filename, file_extension, file_type,
                    size, hash, creation_date, modification_date, access_date, permissions,
                    depth_level, relative_path, is_hidden, is_symlink, symlink_target,
                    duplicate_status, duplicate_group_id, processing_status, estimated_tokens,
                    processed_at, error_message
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                "#,
                file_id,
                full_filepath,
                folder_path,
                metadata.filename,
                metadata.file_extension,
                file_type,
                size,
                metadata.hash,
                metadata.creation_date,
                metadata.modification_date,
                metadata.access_date,
                permissions,
                depth_level,
                relative_path,
                metadata.is_hidden,
                metadata.is_symlink,
                symlink_target,
                duplicate_status,
                duplicate_group_id,
                processing_status,
                estimated_tokens,
                metadata.processed_at,
                metadata.error_message
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        }
        
        tx.commit().await.map_err(|e| PensieveError::Database(e))?;
        Ok(())
    }

    /// Insert multiple paragraphs in a batch transaction for efficient storage
    pub async fn insert_paragraphs_batch(&self, paragraphs: &[Paragraph]) -> Result<()> {
        if paragraphs.is_empty() {
            return Ok(());
        }

        let mut tx = self.begin_transaction().await?;
        
        for paragraph in paragraphs {
            let paragraph_id = paragraph.id.0.to_string();
            let estimated_tokens = paragraph.estimated_tokens as i64;
            let word_count = paragraph.word_count as i64;
            let char_count = paragraph.char_count as i64;
            
            sqlx::query!(
                r#"
                INSERT INTO paragraphs (
                    paragraph_id, content_hash, content, estimated_tokens, 
                    word_count, char_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO NOTHING
                "#,
                paragraph_id,
                paragraph.content_hash,
                paragraph.content,
                estimated_tokens,
                word_count,
                char_count,
                paragraph.created_at
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        }
        
        tx.commit().await.map_err(|e| PensieveError::Database(e))?;
        Ok(())
    }

    /// Insert multiple paragraph sources in a batch transaction
    pub async fn insert_paragraph_sources_batch(&self, sources: &[ParagraphSource]) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }

        let mut tx = self.begin_transaction().await?;
        
        for source in sources {
            let paragraph_id = source.paragraph_id.0.to_string();
            let file_id = source.file_id.0.to_string();
            let paragraph_index = source.paragraph_index as i64;
            let byte_offset_start = source.byte_offset_start as i64;
            let byte_offset_end = source.byte_offset_end as i64;
            
            sqlx::query!(
                r#"
                INSERT INTO paragraph_sources (
                    paragraph_id, file_id, paragraph_index, 
                    byte_offset_start, byte_offset_end, created_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(paragraph_id, file_id, paragraph_index) DO NOTHING
                "#,
                paragraph_id,
                file_id,
                paragraph_index,
                byte_offset_start,
                byte_offset_end
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        }
        
        tx.commit().await.map_err(|e| PensieveError::Database(e))?;
        Ok(())
    }

    /// Insert multiple processing errors in a batch transaction
    pub async fn insert_errors_batch(&self, errors: &[ProcessingError]) -> Result<()> {
        if errors.is_empty() {
            return Ok(());
        }

        let mut tx = self.begin_transaction().await?;
        
        for error in errors {
            let error_id = error.id.to_string();
            let file_id = error.file_id.map(|id| id.0.to_string());
            
            sqlx::query!(
                r#"
                INSERT INTO processing_errors (
                    error_id, file_id, error_type, error_message, 
                    stack_trace, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                "#,
                error_id,
                file_id,
                error.error_type,
                error.error_message,
                error.stack_trace,
                error.occurred_at
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| PensieveError::Database(e))?;
        }
        
        tx.commit().await.map_err(|e| PensieveError::Database(e))?;
        Ok(())
    }

    /// Get files by hash for duplicate detection
    pub async fn get_files_by_hash(&self, hash: &str) -> Result<Vec<FileMetadata>> {
        let rows = sqlx::query!(
            r#"
            SELECT file_id, full_filepath, folder_path, filename, file_extension, file_type,
                   size, hash, creation_date, modification_date, access_date, permissions,
                   depth_level, relative_path, is_hidden, is_symlink, symlink_target,
                   duplicate_status, duplicate_group_id, processing_status, estimated_tokens,
                   processed_at, error_message
            FROM files 
            WHERE hash = ? AND hash != ''
            ORDER BY full_filepath
            "#,
            hash
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let mut files = Vec::new();
        for row in rows {
            let duplicate_status = match row.duplicate_status.as_str() {
                "unique" => DuplicateStatus::Unique,
                "canonical" => DuplicateStatus::Canonical,
                "duplicate" => DuplicateStatus::Duplicate,
                _ => DuplicateStatus::Unique,
            };
            
            let processing_status = match row.processing_status.as_str() {
                "pending" => ProcessingStatus::Pending,
                "processed" => ProcessingStatus::Processed,
                "error" => ProcessingStatus::Error,
                "skipped_binary" => ProcessingStatus::SkippedBinary,
                "skipped_dependency" => ProcessingStatus::SkippedDependency,
                "deleted" => ProcessingStatus::Deleted,
                _ => ProcessingStatus::Pending,
            };
            
            let file_type = match row.file_type.as_str() {
                "file" => FileType::File,
                "directory" => FileType::Directory,
                _ => FileType::File,
            };
            
            // Convert NaiveDateTime to DateTime<Utc>
            let creation_date = row.creation_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let modification_date = row.modification_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let access_date = row.access_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let processed_at = row.processed_at
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
            
            let metadata = FileMetadata {
                full_filepath: PathBuf::from(row.full_filepath),
                folder_path: PathBuf::from(row.folder_path),
                filename: row.filename,
                file_extension: row.file_extension,
                file_type,
                size: row.size as u64,
                hash: row.hash,
                creation_date,
                modification_date,
                access_date,
                permissions: row.permissions.unwrap_or(0) as u32,
                depth_level: row.depth_level as u32,
                relative_path: PathBuf::from(row.relative_path),
                is_hidden: row.is_hidden,
                is_symlink: row.is_symlink,
                symlink_target: row.symlink_target.map(PathBuf::from),
                duplicate_status,
                duplicate_group_id: row.duplicate_group_id.as_ref().and_then(|s| uuid::Uuid::parse_str(s).ok()),
                processing_status,
                estimated_tokens: row.estimated_tokens.map(|t| t as u32),
                processed_at,
                error_message: row.error_message,
            };
            
            files.push(metadata);
        }
        
        Ok(files)
    }

    /// Get duplicate statistics
    pub async fn get_duplicate_statistics(&self) -> Result<DuplicateStatistics> {
        // Get duplicate groups
        let duplicate_groups: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT duplicate_group_id) FROM files WHERE duplicate_group_id IS NOT NULL"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        // Get files by duplicate status
        let unique_files: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM files WHERE duplicate_status = 'unique'"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let canonical_files: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM files WHERE duplicate_status = 'canonical'"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let duplicate_files: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM files WHERE duplicate_status = 'duplicate'"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        // Calculate space savings
        let total_size: i64 = sqlx::query_scalar(
            "SELECT COALESCE(SUM(size), 0) FROM files"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let duplicate_size: i64 = sqlx::query_scalar(
            "SELECT COALESCE(SUM(size), 0) FROM files WHERE duplicate_status = 'duplicate'"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        Ok(DuplicateStatistics {
            duplicate_groups: duplicate_groups as u64,
            unique_files: unique_files as u64,
            canonical_files: canonical_files as u64,
            duplicate_files: duplicate_files as u64,
            total_size: total_size as u64,
            duplicate_size: duplicate_size as u64,
            space_savings: duplicate_size as u64,
        })
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

    /// Get access to the database pool for advanced operations
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
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

/// Duplicate detection statistics
#[derive(Debug, Default)]
pub struct DuplicateStatistics {
    /// Number of duplicate groups
    pub duplicate_groups: u64,
    /// Number of unique files
    pub unique_files: u64,
    /// Number of canonical files (first in each duplicate group)
    pub canonical_files: u64,
    /// Number of duplicate files
    pub duplicate_files: u64,
    /// Total size of all files
    pub total_size: u64,
    /// Total size of duplicate files
    pub duplicate_size: u64,
    /// Space savings from deduplication
    pub space_savings: u64,
}

/// Paragraph-level deduplication statistics
#[derive(Debug, Default)]
pub struct ParagraphStatistics {
    /// Total paragraph instances across all files
    pub total_paragraph_instances: u64,
    /// Number of unique paragraphs stored
    pub unique_paragraphs: u64,
    /// Number of deduplicated paragraph instances
    pub deduplicated_paragraphs: u64,
    /// Deduplication rate as percentage
    pub deduplication_rate: f64,
    /// Total tokens in unique paragraphs
    pub total_tokens: u64,
    /// Average paragraph length in characters
    pub average_paragraph_length: f64,
}

/// Migration information structure
#[derive(Debug, Clone)]
pub struct Migration {
    /// Migration version number
    pub version: u32,
    /// Human-readable description
    pub description: String,
    /// SQL statements to apply the migration
    pub up_sql: Vec<String>,
    /// SQL statements to rollback the migration (optional)
    pub down_sql: Option<Vec<String>>,
    /// Data transformation function (optional)
    pub data_transform: Option<fn(&Database) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>>,
}

/// Database migration manager for schema evolution
pub struct MigrationManager {
    /// Database connection
    database: Database,
    /// Available migrations
    migrations: Vec<Migration>,
}

impl MigrationManager {
    /// Create new migration manager with all available migrations
    pub fn new(database: Database) -> Self {
        let migrations = Self::get_all_migrations();
        Self { database, migrations }
    }

    /// Create migration manager with migrations loaded from files
    pub fn from_files(database: Database, migrations_dir: &std::path::Path) -> Result<Self> {
        let migrations = Self::load_migrations_from_files(migrations_dir)?;
        Ok(Self { database, migrations })
    }

    /// Load migrations from SQL files in a directory
    fn load_migrations_from_files(migrations_dir: &std::path::Path) -> Result<Vec<Migration>> {
        use std::fs;
        
        if !migrations_dir.exists() {
            return Ok(Self::get_all_migrations()); // Fallback to hardcoded migrations
        }

        let mut migrations = Vec::new();
        let mut entries: Vec<_> = fs::read_dir(migrations_dir)
            .map_err(|e| PensieveError::Io(e))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| PensieveError::Io(e))?;

        // Sort by filename to ensure proper order
        entries.sort_by_key(|entry| entry.file_name());

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("sql") {
                let filename = path.file_name()
                    .and_then(|s| s.to_str())
                    .ok_or_else(|| PensieveError::InvalidData("Invalid filename".to_string()))?;

                // Extract version from filename (e.g., "001_initial_schema.sql" -> 1)
                let version = filename.split('_').next()
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| PensieveError::InvalidData(
                        format!("Cannot extract version from filename: {}", filename)
                    ))?;

                let content = fs::read_to_string(&path)
                    .map_err(|e| PensieveError::Io(e))?;

                // Extract description from SQL comments
                let description = Self::extract_description_from_sql(&content)
                    .unwrap_or_else(|| format!("Migration {}", version));

                // Split SQL content into individual statements
                let up_sql = Self::parse_sql_statements(&content);

                // Try to load rollback SQL if it exists
                let rollback_path = migrations_dir.join("rollback")
                    .join(format!("{}_rollback.sql", filename.trim_end_matches(".sql")));
                
                let down_sql = if rollback_path.exists() {
                    let rollback_content = fs::read_to_string(&rollback_path)
                        .map_err(|e| PensieveError::Io(e))?;
                    Some(Self::parse_sql_statements(&rollback_content))
                } else {
                    None
                };

                migrations.push(Migration {
                    version,
                    description,
                    up_sql,
                    down_sql,
                    data_transform: None, // File-based migrations don't support data transforms yet
                });
            }
        }

        // Sort by version
        migrations.sort_by_key(|m| m.version);

        if migrations.is_empty() {
            // Fallback to hardcoded migrations if no files found
            Ok(Self::get_all_migrations())
        } else {
            Ok(migrations)
        }
    }

    /// Extract description from SQL comments
    fn extract_description_from_sql(content: &str) -> Option<String> {
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("-- Description:") {
                return Some(line.trim_start_matches("-- Description:").trim().to_string());
            }
        }
        None
    }

    /// Parse SQL content into individual statements
    fn parse_sql_statements(content: &str) -> Vec<String> {
        let mut statements = Vec::new();
        let mut current_statement = String::new();
        let _in_comment = false;
        
        for line in content.lines() {
            let line = line.trim();
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("--") {
                continue;
            }
            
            // Add line to current statement
            if !current_statement.is_empty() {
                current_statement.push(' ');
            }
            current_statement.push_str(line);
            
            // Check if statement is complete (ends with semicolon)
            if line.ends_with(';') {
                let statement = current_statement.trim().to_string();
                if !statement.is_empty() {
                    statements.push(statement);
                }
                current_statement.clear();
            }
        }
        
        // Add any remaining statement
        if !current_statement.trim().is_empty() {
            statements.push(current_statement.trim().to_string());
        }
        
        statements
    }

    /// Get all available migrations in order
    fn get_all_migrations() -> Vec<Migration> {
        vec![
            Migration {
                version: 1,
                description: "Create initial schema with files, paragraphs, paragraph_sources, and processing_errors tables".to_string(),
                up_sql: vec![
                    // Files table
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
                    "#.to_string(),
                    
                    // Paragraphs table
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
                    "#.to_string(),
                    
                    // Paragraph sources table
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
                    "#.to_string(),
                    
                    // Processing errors table
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
                    "#.to_string(),
                ],
                down_sql: Some(vec![
                    "DROP TABLE IF EXISTS paragraph_sources".to_string(),
                    "DROP TABLE IF EXISTS processing_errors".to_string(),
                    "DROP TABLE IF EXISTS paragraphs".to_string(),
                    "DROP TABLE IF EXISTS files".to_string(),
                ]),
                data_transform: None,
            },
            
            Migration {
                version: 2,
                description: "Add performance indexes for common queries".to_string(),
                up_sql: vec![
                    // Files table indexes
                    "CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_duplicate_group ON files(duplicate_group_id) WHERE duplicate_group_id IS NOT NULL".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_processing_status ON files(processing_status)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_modification_date ON files(modification_date)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_file_extension ON files(file_extension) WHERE file_extension IS NOT NULL".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_size ON files(size)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_duplicate_status ON files(duplicate_status)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_processed_at ON files(processed_at) WHERE processed_at IS NOT NULL".to_string(),
                    
                    // Paragraphs table indexes
                    "CREATE INDEX IF NOT EXISTS idx_paragraphs_hash ON paragraphs(content_hash)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_paragraphs_tokens ON paragraphs(estimated_tokens)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_paragraphs_created_at ON paragraphs(created_at)".to_string(),
                    
                    // Paragraph sources table indexes
                    "CREATE INDEX IF NOT EXISTS idx_paragraph_sources_file ON paragraph_sources(file_id)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_paragraph_sources_paragraph ON paragraph_sources(paragraph_id)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_paragraph_sources_index ON paragraph_sources(paragraph_index)".to_string(),
                    
                    // Processing errors table indexes
                    "CREATE INDEX IF NOT EXISTS idx_processing_errors_file ON processing_errors(file_id) WHERE file_id IS NOT NULL".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_processing_errors_type ON processing_errors(error_type)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_processing_errors_occurred_at ON processing_errors(occurred_at)".to_string(),
                    
                    // Composite indexes for common query patterns
                    "CREATE INDEX IF NOT EXISTS idx_files_status_type ON files(processing_status, file_type)".to_string(),
                    "CREATE INDEX IF NOT EXISTS idx_files_duplicate_hash ON files(duplicate_status, hash)".to_string(),
                ],
                down_sql: Some(vec![
                    "DROP INDEX IF EXISTS idx_files_hash".to_string(),
                    "DROP INDEX IF EXISTS idx_files_duplicate_group".to_string(),
                    "DROP INDEX IF EXISTS idx_files_processing_status".to_string(),
                    "DROP INDEX IF EXISTS idx_files_modification_date".to_string(),
                    "DROP INDEX IF EXISTS idx_files_file_extension".to_string(),
                    "DROP INDEX IF EXISTS idx_files_size".to_string(),
                    "DROP INDEX IF EXISTS idx_files_duplicate_status".to_string(),
                    "DROP INDEX IF EXISTS idx_files_processed_at".to_string(),
                    "DROP INDEX IF EXISTS idx_paragraphs_hash".to_string(),
                    "DROP INDEX IF EXISTS idx_paragraphs_tokens".to_string(),
                    "DROP INDEX IF EXISTS idx_paragraphs_created_at".to_string(),
                    "DROP INDEX IF EXISTS idx_paragraph_sources_file".to_string(),
                    "DROP INDEX IF EXISTS idx_paragraph_sources_paragraph".to_string(),
                    "DROP INDEX IF EXISTS idx_paragraph_sources_index".to_string(),
                    "DROP INDEX IF EXISTS idx_processing_errors_file".to_string(),
                    "DROP INDEX IF EXISTS idx_processing_errors_type".to_string(),
                    "DROP INDEX IF EXISTS idx_processing_errors_occurred_at".to_string(),
                    "DROP INDEX IF EXISTS idx_files_status_type".to_string(),
                    "DROP INDEX IF EXISTS idx_files_duplicate_hash".to_string(),
                ]),
                data_transform: None,
            },
            
            Migration {
                version: 3,
                description: "Add MIME type support and enhanced file metadata".to_string(),
                up_sql: vec![
                    "CREATE INDEX IF NOT EXISTS idx_files_mime_type ON files(mime_type) WHERE mime_type IS NOT NULL".to_string(),
                ],
                down_sql: Some(vec![
                    "DROP INDEX IF EXISTS idx_files_mime_type".to_string(),
                    // Note: SQLite doesn't support DROP COLUMN, so we'd need to recreate the table
                    // For now, we'll leave the column but document this limitation
                ]),
                data_transform: Some(|db: &Database| {
                    let pool = db.pool().clone();
                    Box::pin(async move {
                        // Example data transformation: populate MIME types for existing files
                        // This would analyze existing files and set their MIME types
                        println!("Running data transformation for MIME type population...");
                        
                        // In a real implementation, this would:
                        // 1. Query all files without MIME types
                        // 2. Analyze their extensions or content
                        // 3. Update the mime_type column
                        
                        // For now, just update common extensions
                        let updates = vec![
                            ("UPDATE files SET mime_type = 'text/plain' WHERE file_extension IN ('txt', 'md', 'rst')", "text files"),
                            ("UPDATE files SET mime_type = 'text/html' WHERE file_extension = 'html'", "HTML files"),
                            ("UPDATE files SET mime_type = 'application/json' WHERE file_extension = 'json'", "JSON files"),
                            ("UPDATE files SET mime_type = 'text/x-python' WHERE file_extension = 'py'", "Python files"),
                            ("UPDATE files SET mime_type = 'text/x-rust' WHERE file_extension = 'rs'", "Rust files"),
                        ];
                        
                        for (sql, description) in updates {
                            let affected = sqlx::query(sql)
                                .execute(&pool)
                                .await
                                .map_err(|e| PensieveError::Database(e))?
                                .rows_affected();
                            
                            if affected > 0 {
                                println!("Updated {} {} with MIME types", affected, description);
                            }
                        }
                        
                        Ok(())
                    })
                }),
            },
        ]
    }

    /// Run all pending migrations
    pub async fn migrate(&self) -> Result<()> {
        let current_version = self.get_current_version().await?;
        let target_version = self.get_target_version();

        if current_version >= target_version {
            return Ok(()); // Already up to date
        }

        println!("Running migrations from version {} to {}", current_version, target_version);

        // Run migrations sequentially in a transaction
        let mut tx = self.database.pool.begin().await.map_err(|e| PensieveError::Database(e))?;

        for version in (current_version + 1)..=target_version {
            println!("Applying migration {}", version);
            self.apply_migration_in_transaction(&mut tx, version).await?;
        }

        tx.commit().await.map_err(|e| PensieveError::Database(e))?;
        println!("All migrations completed successfully");

        Ok(())
    }

    /// Get current schema version from database
    pub async fn get_current_version(&self) -> Result<u32> {
        // Ensure schema_version table exists first
        self.database.create_schema_version_table().await?;
        self.database.get_schema_version().await
    }

    /// Get target schema version (latest available)
    pub fn get_target_version(&self) -> u32 {
        self.migrations.iter().map(|m| m.version).max().unwrap_or(0)
    }

    /// Apply a specific migration within a transaction
    async fn apply_migration_in_transaction(
        &self, 
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>, 
        version: u32
    ) -> Result<()> {
        let migration = self.migrations.iter()
            .find(|m| m.version == version)
            .ok_or_else(|| PensieveError::Migration(
                format!("Migration version {} not found", version)
            ))?;

        // Special handling for migration 3 (MIME type column)
        if version == 3 {
            // Check if mime_type column already exists
            let column_exists = self.check_column_exists_in_transaction(tx, "files", "mime_type").await?;
            
            if !column_exists {
                // Add the column only if it doesn't exist
                sqlx::query("ALTER TABLE files ADD COLUMN mime_type TEXT")
                    .execute(&mut **tx)
                    .await
                    .map_err(|e| PensieveError::Migration(
                        format!("Failed to add mime_type column: {}", e)
                    ))?;
            }
            
            // Always create the index (it's safe to run multiple times)
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_files_mime_type ON files(mime_type) WHERE mime_type IS NOT NULL")
                .execute(&mut **tx)
                .await
                .map_err(|e| PensieveError::Migration(
                    format!("Failed to create mime_type index: {}", e)
                ))?;
        } else {
            // Execute all SQL statements for other migrations
            for sql in &migration.up_sql {
                sqlx::query(sql)
                    .execute(&mut **tx)
                    .await
                    .map_err(|e| PensieveError::Migration(
                        format!("Failed to execute migration {} SQL: {} - Error: {}", version, sql, e)
                    ))?;
            }
        }

        // Record successful migration
        sqlx::query(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)"
        )
        .bind(version as i64)
        .bind(&migration.description)
        .execute(&mut **tx)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Check if a column exists in a table within a transaction
    async fn check_column_exists_in_transaction(
        &self, 
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        table_name: &str, 
        column_name: &str
    ) -> Result<bool> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info(?) WHERE name = ?"
        )
        .bind(table_name)
        .bind(column_name)
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(count > 0)
    }

    /// Apply a specific migration (legacy method for compatibility)
    async fn apply_migration(&self, version: u32) -> Result<()> {
        let migration = self.migrations.iter()
            .find(|m| m.version == version)
            .ok_or_else(|| PensieveError::Migration(
                format!("Migration version {} not found", version)
            ))?;

        // Special handling for migration 3 (MIME type column)
        if version == 3 {
            // Check if mime_type column already exists
            let column_exists = self.check_column_exists("files", "mime_type").await?;
            
            if !column_exists {
                // Add the column only if it doesn't exist
                sqlx::query("ALTER TABLE files ADD COLUMN mime_type TEXT")
                    .execute(&self.database.pool)
                    .await
                    .map_err(|e| PensieveError::Migration(
                        format!("Failed to add mime_type column: {}", e)
                    ))?;
            }
        } else {
            // Execute all SQL statements for other migrations
            for sql in &migration.up_sql {
                sqlx::query(sql)
                    .execute(&self.database.pool)
                    .await
                    .map_err(|e| PensieveError::Migration(
                        format!("Failed to execute migration {} SQL: {} - Error: {}", version, sql, e)
                    ))?;
            }
        }

        // For migration 3, always create the index (it's safe to run multiple times)
        if version == 3 {
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_files_mime_type ON files(mime_type) WHERE mime_type IS NOT NULL")
                .execute(&self.database.pool)
                .await
                .map_err(|e| PensieveError::Migration(
                    format!("Failed to create mime_type index: {}", e)
                ))?;
        }

        // Execute data transformation if present
        if let Some(transform_fn) = migration.data_transform {
            transform_fn(&self.database).await?;
        }

        // Record successful migration
        sqlx::query(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)"
        )
        .bind(version as i64)
        .bind(&migration.description)
        .execute(&self.database.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Check if a column exists in a table
    async fn check_column_exists(&self, table_name: &str, column_name: &str) -> Result<bool> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pragma_table_info(?) WHERE name = ?"
        )
        .bind(table_name)
        .bind(column_name)
        .fetch_one(&self.database.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(count > 0)
    }

    /// Get migration history
    pub async fn get_migration_history(&self) -> Result<Vec<(u32, String, DateTime<Utc>)>> {
        let rows = sqlx::query!(
            "SELECT version, description, applied_at FROM schema_version ORDER BY version"
        )
        .fetch_all(&self.database.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        let mut history = Vec::new();
        for row in rows {
            let version = row.version as u32;
            let description = row.description;
            let applied_at = row.applied_at
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| Utc::now());
            
            history.push((version, description, applied_at));
        }

        Ok(history)
    }

    /// Check if a specific migration has been applied
    pub async fn is_migration_applied(&self, version: u32) -> Result<bool> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM schema_version WHERE version = ?"
        )
        .bind(version as i64)
        .fetch_one(&self.database.pool)
        .await
        .map_err(|e| PensieveError::Database(e))?;

        Ok(count > 0)
    }

    /// Rollback to a specific version
    pub async fn rollback_to(&self, target_version: u32) -> Result<()> {
        let current_version = self.get_current_version().await?;
        
        if target_version >= current_version {
            return Ok(()); // Nothing to rollback
        }

        println!("Rolling back from version {} to {}", current_version, target_version);

        // Rollback migrations in reverse order
        for version in ((target_version + 1)..=current_version).rev() {
            println!("Rolling back migration {}", version);
            self.rollback_migration(version).await?;
        }

        println!("Rollback completed successfully");
        Ok(())
    }

    /// Rollback a specific migration
    async fn rollback_migration(&self, version: u32) -> Result<()> {
        let migration = self.migrations.iter()
            .find(|m| m.version == version)
            .ok_or_else(|| PensieveError::Migration(
                format!("Migration version {} not found", version)
            ))?;

        // Execute rollback SQL if available
        if let Some(down_sql) = &migration.down_sql {
            for sql in down_sql {
                sqlx::query(sql)
                    .execute(&self.database.pool)
                    .await
                    .map_err(|e| PensieveError::Migration(
                        format!("Failed to rollback migration {} SQL: {} - Error: {}", version, sql, e)
                    ))?;
            }
        } else {
            return Err(PensieveError::Migration(
                format!("Migration {} does not support rollback", version)
            ));
        }

        // Remove migration record
        sqlx::query("DELETE FROM schema_version WHERE version = ?")
            .bind(version as i64)
            .execute(&self.database.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;

        Ok(())
    }

    /// Validate database schema integrity
    pub async fn validate_schema(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();

        // Check that all expected tables exist
        let expected_tables = vec!["files", "paragraphs", "paragraph_sources", "processing_errors", "schema_version"];
        
        for table in expected_tables {
            let count: i64 = sqlx::query_scalar(&format!(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'", 
                table
            ))
            .fetch_one(&self.database.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?;
            
            if count == 0 {
                issues.push(format!("Missing table: {}", table));
            }
        }

        // Check foreign key constraints
        let fk_violations: i64 = sqlx::query_scalar("PRAGMA foreign_key_check")
            .fetch_optional(&self.database.pool)
            .await
            .map_err(|e| PensieveError::Database(e))?
            .unwrap_or(0);

        if fk_violations > 0 {
            issues.push(format!("Foreign key violations found: {}", fk_violations));
        }

        // Check schema version consistency
        let current_version = self.get_current_version().await?;
        let target_version = self.get_target_version();
        
        if current_version > target_version {
            issues.push(format!(
                "Database version {} is newer than expected version {}", 
                current_version, target_version
            ));
        }

        Ok(issues)
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
        
        // Verify schema version (should be the latest version)
        let version = db.get_schema_version().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        let target_version = migration_manager.get_target_version();
        assert_eq!(version, target_version);
        
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
        
        // Version should now be the latest (3 in our case)
        let final_version = migration_manager.get_current_version().await.unwrap();
        assert_eq!(final_version, 3);
        
        // Running migrations again should be idempotent
        migration_manager.migrate().await.unwrap();
        let version_after_second_run = migration_manager.get_current_version().await.unwrap();
        assert_eq!(version_after_second_run, 3);
    }

    #[tokio::test]
    async fn test_migration_history() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Run migrations
        migration_manager.migrate().await.unwrap();
        
        // Check migration history
        let history = migration_manager.get_migration_history().await.unwrap();
        assert_eq!(history.len(), 4); // Initial version 0 + 3 migrations
        
        // Check that versions are in order
        for (i, (version, _, _)) in history.iter().enumerate() {
            assert_eq!(*version, i as u32);
        }
    }

    #[tokio::test]
    async fn test_migration_rollback() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Run all migrations
        migration_manager.migrate().await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 3);
        
        // Rollback to version 1
        migration_manager.rollback_to(1).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 1);
        
        // Check that migration 2 and 3 are no longer applied
        assert!(!migration_manager.is_migration_applied(2).await.unwrap());
        assert!(!migration_manager.is_migration_applied(3).await.unwrap());
        assert!(migration_manager.is_migration_applied(1).await.unwrap());
    }

    #[tokio::test]
    async fn test_schema_validation() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Run migrations
        migration_manager.migrate().await.unwrap();
        
        // Validate schema
        let issues = migration_manager.validate_schema().await.unwrap();
        assert!(issues.is_empty(), "Schema validation found issues: {:?}", issues);
    }

    #[tokio::test]
    async fn test_incremental_migrations() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Apply migrations one by one to test incremental updates
        
        // Start with version 0
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 0);
        
        // Apply migration 1
        migration_manager.apply_migration(1).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 1);
        assert!(migration_manager.is_migration_applied(1).await.unwrap());
        
        // Apply migration 2
        migration_manager.apply_migration(2).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 2);
        assert!(migration_manager.is_migration_applied(2).await.unwrap());
        
        // Apply migration 3
        migration_manager.apply_migration(3).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 3);
        assert!(migration_manager.is_migration_applied(3).await.unwrap());
        
        // Verify all tables and indexes exist
        let tables = vec!["files", "paragraphs", "paragraph_sources", "processing_errors"];
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
        
        // Check that MIME type column was added in migration 3
        let columns: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM pragma_table_info('files') WHERE name = 'mime_type'"
        )
        .fetch_all(&db.pool)
        .await
        .unwrap();
        assert_eq!(columns.len(), 1, "mime_type column should exist");
    }

    #[tokio::test]
    async fn test_data_transformation() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Apply first two migrations
        migration_manager.apply_migration(1).await.unwrap();
        migration_manager.apply_migration(2).await.unwrap();
        
        // Insert some test data
        sqlx::query!(
            r#"
            INSERT INTO files (
                file_id, full_filepath, folder_path, filename, file_extension, 
                file_type, size, hash, creation_date, modification_date, 
                access_date, permissions, depth_level, relative_path, 
                is_hidden, is_symlink, duplicate_status, processing_status
            ) VALUES (
                'test-id', '/test/file.py', '/test', 'file.py', 'py',
                'file', 1000, 'testhash', datetime('now'), datetime('now'),
                datetime('now'), 644, 1, 'file.py',
                0, 0, 'unique', 'pending'
            )
            "#
        )
        .execute(&db.pool)
        .await
        .unwrap();
        
        // Apply migration 3 which includes data transformation
        migration_manager.apply_migration(3).await.unwrap();
        
        // Check that MIME type was populated
        let mime_type: Option<String> = sqlx::query_scalar(
            "SELECT mime_type FROM files WHERE file_id = 'test-id'"
        )
        .fetch_one(&db.pool)
        .await
        .unwrap();
        
        assert_eq!(mime_type, Some("text/x-python".to_string()));
    }

    #[tokio::test]
    async fn test_file_based_migrations() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        
        // Test loading migrations from files
        let migrations_dir = std::path::Path::new("migrations");
        if migrations_dir.exists() {
            let migration_manager = MigrationManager::from_files(db.clone(), migrations_dir).unwrap();
            
            // Should have loaded migrations from files
            assert!(migration_manager.migrations.len() >= 3);
            
            // Run migrations
            migration_manager.migrate().await.unwrap();
            
            // Verify final version
            let version = migration_manager.get_current_version().await.unwrap();
            assert!(version >= 3);
            
            // Verify tables exist
            let tables = vec!["files", "paragraphs", "paragraph_sources", "processing_errors"];
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
    }

    #[tokio::test]
    async fn test_migration_version_upgrades() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Test upgrading from version 0 to 1
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 0);
        migration_manager.apply_migration(1).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 1);
        
        // Test upgrading from version 1 to 2
        migration_manager.apply_migration(2).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 2);
        
        // Test upgrading from version 2 to 3
        migration_manager.apply_migration(3).await.unwrap();
        assert_eq!(migration_manager.get_current_version().await.unwrap(), 3);
        
        // Test that all migrations are recorded in history
        let history = migration_manager.get_migration_history().await.unwrap();
        assert_eq!(history.len(), 4); // Version 0 + 3 migrations
        
        // Verify migration descriptions
        let descriptions: Vec<String> = history.iter().map(|(_, desc, _)| desc.clone()).collect();
        assert!(descriptions[1].contains("initial schema"));
        assert!(descriptions[2].contains("performance indexes"));
        assert!(descriptions[3].contains("MIME type"));
    }

    #[tokio::test]
    async fn test_migration_error_handling() {
        let (db, _temp_file) = create_test_database().await.unwrap();
        let migration_manager = MigrationManager::new(db.clone());
        
        // Test applying non-existent migration
        let result = migration_manager.apply_migration(999).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Migration version 999 not found"));
        
        // Test rollback of non-existent migration
        let result = migration_manager.rollback_migration(999).await;
        assert!(result.is_err());
        
        // Test rollback to higher version (should be no-op)
        migration_manager.migrate().await.unwrap();
        let current_version = migration_manager.get_current_version().await.unwrap();
        migration_manager.rollback_to(current_version + 1).await.unwrap(); // Should not error
        assert_eq!(migration_manager.get_current_version().await.unwrap(), current_version);
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

    #[tokio::test]
    async fn test_paragraph_processing_methods() {
        use crate::types::{Paragraph, ParagraphId, ParagraphSource, FileId, ProcessingError};
        use chrono::Utc;
        use uuid::Uuid;
        
        let (db, _temp_file) = create_test_database().await.unwrap();
        db.initialize_schema().await.unwrap();
        
        // Test insert_paragraph
        let paragraph = Paragraph {
            id: ParagraphId(Uuid::new_v4()),
            content_hash: "test_hash_123".to_string(),
            content: "This is a test paragraph content.".to_string(),
            estimated_tokens: 8,
            word_count: 6,
            char_count: 33,
            created_at: Utc::now(),
        };
        
        db.insert_paragraph(&paragraph).await.unwrap();
        
        // Test get_paragraph_by_hash
        let retrieved = db.get_paragraph_by_hash("test_hash_123").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved_paragraph = retrieved.unwrap();
        assert_eq!(retrieved_paragraph.content_hash, "test_hash_123");
        assert_eq!(retrieved_paragraph.content, "This is a test paragraph content.");
        assert_eq!(retrieved_paragraph.estimated_tokens, 8);
        
        // Test deduplication - inserting same hash should not create duplicate
        let duplicate_paragraph = Paragraph {
            id: ParagraphId(Uuid::new_v4()),
            content_hash: "test_hash_123".to_string(), // Same hash
            content: "Different content but same hash".to_string(),
            estimated_tokens: 10,
            word_count: 5,
            char_count: 32,
            created_at: Utc::now(),
        };
        
        db.insert_paragraph(&duplicate_paragraph).await.unwrap();
        
        // Should still return the original paragraph
        let retrieved_again = db.get_paragraph_by_hash("test_hash_123").await.unwrap();
        assert!(retrieved_again.is_some());
        let retrieved_paragraph_again = retrieved_again.unwrap();
        assert_eq!(retrieved_paragraph_again.content, "This is a test paragraph content."); // Original content
        
        // Test insert_paragraph_source - first need to create a file
        let file_id = FileId(Uuid::new_v4());
        let file_id_str = file_id.0.to_string();
        let now = Utc::now();
        
        // Insert a test file directly with specific ID to satisfy foreign key constraint
        sqlx::query!(
            r#"
            INSERT INTO files (
                file_id, full_filepath, folder_path, filename, file_extension, file_type,
                size, hash, creation_date, modification_date, access_date, permissions,
                depth_level, relative_path, is_hidden, is_symlink, symlink_target,
                duplicate_status, duplicate_group_id, processing_status, estimated_tokens,
                processed_at, error_message
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            "#,
            file_id_str,
            "/test/file.txt",
            "/test",
            "file.txt",
            "txt",
            "file",
            100i64,
            "test_file_hash",
            now,
            now,
            now,
            644i64,
            1i64,
            "file.txt",
            false,
            false,
            None::<String>,
            "unique",
            None::<String>,
            "pending",
            None::<i64>,
            None::<chrono::DateTime<Utc>>,
            None::<String>
        )
        .execute(&db.pool)
        .await
        .unwrap();
        
        let paragraph_source = ParagraphSource {
            paragraph_id: paragraph.id,
            file_id,
            paragraph_index: 0,
            byte_offset_start: 0,
            byte_offset_end: 33,
        };
        
        db.insert_paragraph_source(&paragraph_source).await.unwrap();
        
        // Test insert_error
        let processing_error = ProcessingError {
            id: Uuid::new_v4(),
            file_id: Some(file_id),
            error_type: "TestError".to_string(),
            error_message: "This is a test error message".to_string(),
            stack_trace: Some("test stack trace".to_string()),
            occurred_at: Utc::now(),
        };
        
        db.insert_error(&processing_error).await.unwrap();
        
        // Test batch operations
        let batch_paragraphs = vec![
            Paragraph {
                id: ParagraphId(Uuid::new_v4()),
                content_hash: "batch_hash_1".to_string(),
                content: "Batch paragraph 1".to_string(),
                estimated_tokens: 3,
                word_count: 3,
                char_count: 17,
                created_at: Utc::now(),
            },
            Paragraph {
                id: ParagraphId(Uuid::new_v4()),
                content_hash: "batch_hash_2".to_string(),
                content: "Batch paragraph 2".to_string(),
                estimated_tokens: 3,
                word_count: 3,
                char_count: 17,
                created_at: Utc::now(),
            },
        ];
        
        db.insert_paragraphs_batch(&batch_paragraphs).await.unwrap();
        
        // Verify batch inserts worked
        let batch_1 = db.get_paragraph_by_hash("batch_hash_1").await.unwrap();
        let batch_2 = db.get_paragraph_by_hash("batch_hash_2").await.unwrap();
        assert!(batch_1.is_some());
        assert!(batch_2.is_some());
        assert_eq!(batch_1.unwrap().content, "Batch paragraph 1");
        assert_eq!(batch_2.unwrap().content, "Batch paragraph 2");
    }
}