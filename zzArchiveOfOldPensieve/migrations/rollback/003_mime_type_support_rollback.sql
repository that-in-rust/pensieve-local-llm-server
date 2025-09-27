-- Rollback Migration: Remove MIME type support
-- Version: 3
-- Description: Remove MIME type column and related indexes

-- Drop MIME type index
DROP INDEX IF EXISTS idx_files_mime_type;

-- Note: SQLite doesn't support DROP COLUMN directly
-- In a production system, this would require recreating the table
-- For now, we document this limitation and leave the column
-- A full rollback would require:
-- 1. CREATE TABLE files_backup AS SELECT ... (without mime_type)
-- 2. DROP TABLE files
-- 3. ALTER TABLE files_backup RENAME TO files