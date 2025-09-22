-- Migration: Add MIME type support and enhanced file metadata
-- Version: 3
-- Description: Add MIME type column for better file classification

-- Add MIME type column to files table
ALTER TABLE files ADD COLUMN mime_type TEXT;

-- Create index for MIME type queries
CREATE INDEX IF NOT EXISTS idx_files_mime_type ON files(mime_type) WHERE mime_type IS NOT NULL;