-- Chunked table schema support for large file processing
-- This migration defines the schema template for chunked tables

-- Template schema for chunked tables (TABLENAME_CHUNKSIZE format)
-- This is documented for reference and used by SchemaManager

/*
CREATE TABLE "INGEST_TEMPLATE_CHUNKSIZE" (
    id BIGSERIAL PRIMARY KEY,
    file_id TEXT NOT NULL,
    filepath TEXT NOT NULL,
    parent_filepath TEXT NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT,
    chunk_number INTEGER NOT NULL,
    chunk_start_line INTEGER NOT NULL,
    chunk_end_line INTEGER NOT NULL,
    line_count INTEGER,
    content TEXT,
    content_l1 TEXT,  -- Context with ±1 chunk
    content_l2 TEXT,  -- Context with ±2 chunks
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT check_chunk_number_positive CHECK (chunk_number > 0),
    CONSTRAINT check_line_numbers CHECK (chunk_end_line >= chunk_start_line)
);
*/

-- Insert this migration record
INSERT INTO schema_migrations (version, checksum) 
VALUES ('003_chunked_table_support', 'c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8') 
ON CONFLICT (version) DO NOTHING;