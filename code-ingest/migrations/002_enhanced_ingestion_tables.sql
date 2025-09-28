-- Enhanced ingestion table schema with multi-scale context support
-- This migration adds support for L1-L8 extraction hierarchy and chunking

-- This migration will be applied when creating new ingestion tables
-- The schema is used as a template for INGEST_YYYYMMDDHHMMSS tables

-- Template schema for ingestion tables (not creating actual table here)
-- This is documented for reference and used by SchemaManager

/*
CREATE TABLE "INGEST_TEMPLATE" (
    file_id BIGSERIAL PRIMARY KEY,
    ingestion_id BIGINT NOT NULL,
    filepath VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    extension VARCHAR,
    file_size_bytes BIGINT NOT NULL,
    line_count INTEGER,
    word_count INTEGER,
    token_count INTEGER,
    content_text TEXT,
    file_type VARCHAR NOT NULL CHECK (file_type IN ('direct_text', 'convertible', 'non_text')),
    conversion_command VARCHAR,
    relative_path VARCHAR NOT NULL,
    absolute_path VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Multi-scale context columns for knowledge arbitrage
    parent_filepath VARCHAR,
    l1_window_content TEXT,
    l2_window_content TEXT,
    ast_patterns JSONB
);
*/

-- Insert this migration record
INSERT INTO schema_migrations (version, checksum) 
VALUES ('002_enhanced_ingestion_tables', 'b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7') 
ON CONFLICT (version) DO NOTHING;