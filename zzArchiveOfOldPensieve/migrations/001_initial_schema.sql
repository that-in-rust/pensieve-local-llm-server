-- Migration: Create initial schema with files, paragraphs, paragraph_sources, and processing_errors tables
-- Version: 1
-- Description: Initial database schema for Pensieve CLI tool

-- Files table for comprehensive metadata tracking
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
);

-- Paragraphs table for deduplicated content storage
CREATE TABLE IF NOT EXISTS paragraphs (
    paragraph_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL CHECK(length(content) > 0),
    estimated_tokens INTEGER NOT NULL CHECK(estimated_tokens > 0),
    word_count INTEGER NOT NULL CHECK(word_count >= 0),
    char_count INTEGER NOT NULL CHECK(char_count > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paragraph sources table for many-to-many relationships
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
);

-- Processing errors table for error tracking and debugging
CREATE TABLE IF NOT EXISTS processing_errors (
    error_id TEXT PRIMARY KEY,
    file_id TEXT,
    error_type TEXT NOT NULL CHECK(length(error_type) > 0),
    error_message TEXT NOT NULL CHECK(length(error_message) > 0),
    stack_trace TEXT,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE SET NULL
);