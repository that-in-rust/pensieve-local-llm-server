-- Migration: Add performance indexes for common queries
-- Version: 2
-- Description: Create comprehensive indexes for optimal query performance

-- Files table indexes for common queries
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
CREATE INDEX IF NOT EXISTS idx_files_duplicate_group ON files(duplicate_group_id) WHERE duplicate_group_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_files_processing_status ON files(processing_status);
CREATE INDEX IF NOT EXISTS idx_files_modification_date ON files(modification_date);
CREATE INDEX IF NOT EXISTS idx_files_file_extension ON files(file_extension) WHERE file_extension IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_files_size ON files(size);
CREATE INDEX IF NOT EXISTS idx_files_duplicate_status ON files(duplicate_status);
CREATE INDEX IF NOT EXISTS idx_files_processed_at ON files(processed_at) WHERE processed_at IS NOT NULL;

-- Paragraphs table indexes
CREATE INDEX IF NOT EXISTS idx_paragraphs_hash ON paragraphs(content_hash);
CREATE INDEX IF NOT EXISTS idx_paragraphs_tokens ON paragraphs(estimated_tokens);
CREATE INDEX IF NOT EXISTS idx_paragraphs_created_at ON paragraphs(created_at);

-- Paragraph sources table indexes for joins
CREATE INDEX IF NOT EXISTS idx_paragraph_sources_file ON paragraph_sources(file_id);
CREATE INDEX IF NOT EXISTS idx_paragraph_sources_paragraph ON paragraph_sources(paragraph_id);
CREATE INDEX IF NOT EXISTS idx_paragraph_sources_index ON paragraph_sources(paragraph_index);

-- Processing errors table indexes
CREATE INDEX IF NOT EXISTS idx_processing_errors_file ON processing_errors(file_id) WHERE file_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_processing_errors_type ON processing_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_processing_errors_occurred_at ON processing_errors(occurred_at);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_files_status_type ON files(processing_status, file_type);
CREATE INDEX IF NOT EXISTS idx_files_duplicate_hash ON files(duplicate_status, hash);