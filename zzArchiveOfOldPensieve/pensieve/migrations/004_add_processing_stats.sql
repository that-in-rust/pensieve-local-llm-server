-- Migration: Add processing statistics table
-- Version: 4
-- Description: Add table for tracking processing statistics and performance metrics

-- Processing statistics table for performance tracking
CREATE TABLE IF NOT EXISTS processing_stats (
    stat_id TEXT PRIMARY KEY,
    processing_session_id TEXT NOT NULL,
    files_processed INTEGER NOT NULL DEFAULT 0,
    paragraphs_created INTEGER NOT NULL DEFAULT 0,
    duplicates_found INTEGER NOT NULL DEFAULT 0,
    errors_encountered INTEGER NOT NULL DEFAULT 0,
    processing_start_time TIMESTAMP NOT NULL,
    processing_end_time TIMESTAMP,
    total_processing_time_ms INTEGER,
    average_file_processing_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for querying by session
CREATE INDEX IF NOT EXISTS idx_processing_stats_session ON processing_stats(processing_session_id);

-- Index for querying by processing time
CREATE INDEX IF NOT EXISTS idx_processing_stats_start_time ON processing_stats(processing_start_time);