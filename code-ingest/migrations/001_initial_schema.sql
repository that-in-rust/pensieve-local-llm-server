-- Initial schema migration for code-ingest system
-- This migration creates the base tables and indexes for the ingestion system

-- Create ingestion_meta table for tracking ingestion metadata
CREATE TABLE IF NOT EXISTS ingestion_meta (
    ingestion_id BIGSERIAL PRIMARY KEY,
    repo_url VARCHAR,
    local_path VARCHAR NOT NULL,
    start_timestamp_unix BIGINT NOT NULL,
    end_timestamp_unix BIGINT,
    table_name VARCHAR NOT NULL,
    total_files_processed INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for ingestion_meta table
CREATE INDEX IF NOT EXISTS idx_ingestion_meta_table_name ON ingestion_meta(table_name);
CREATE INDEX IF NOT EXISTS idx_ingestion_meta_repo_url ON ingestion_meta(repo_url);
CREATE INDEX IF NOT EXISTS idx_ingestion_meta_created_at ON ingestion_meta(created_at);

-- Create schema_migrations table to track applied migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    checksum VARCHAR(64) NOT NULL
);

-- Insert this migration record
INSERT INTO schema_migrations (version, checksum) 
VALUES ('001_initial_schema', 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6') 
ON CONFLICT (version) DO NOTHING;