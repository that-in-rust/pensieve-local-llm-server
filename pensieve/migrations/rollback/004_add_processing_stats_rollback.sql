-- Rollback Migration: Remove processing statistics table
-- Version: 4
-- Description: Remove processing statistics table and related indexes

-- Drop indexes first
DROP INDEX IF EXISTS idx_processing_stats_session;
DROP INDEX IF EXISTS idx_processing_stats_start_time;

-- Drop the table
DROP TABLE IF EXISTS processing_stats;