# XSV Enhanced Ingestion Example

This example demonstrates the enhanced ingestion system with multi-scale context windows.

## What Was Enhanced

The ingestion system now automatically creates 4 additional columns during ingestion:

1. **parent_filepath** - Calculated by going back 1 slash in the filepath
2. **l1_window_content** - Directory-level concatenation of all files in same parent_filepath
3. **l2_window_content** - System-level concatenation of all files in same grandfather directory
4. **ast_patterns** - JSONB column ready for semantic search patterns

## Step-by-Step Process

### Step 1: Enhanced Schema Implementation

Modified `code-ingest/src/database/schema.rs` to include 4 new columns in the ingestion table creation.

### Step 2: Enhanced Data Population

Modified `code-ingest/src/database/operations.rs` to:
- Calculate parent_filepath during insertion
- Populate window content after all files are inserted using SQL aggregation

### Step 3: Test Ingestion

```bash
./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path /Users/neetipatni/desktop/PensieveDB01
```

**Result**: Table `INGEST_20250928073857` with 59 files and 19 columns (15 original + 4 new)

### Step 4: Verification

Verified the multi-scale context windows work correctly:

#### Hierarchical Structure
- `./xsv` (root level)
- `./xsv/ci` (CI scripts) 
- `./xsv/src` (main source)
- `./xsv/src/cmd` (21 command modules)
- `./xsv/tests` (test files)

#### Multi-Scale Context Verified
- **Individual**: Each file has its own content_text
- **L1 (Module)**: Directory-level concatenation with `--- FILE SEPARATOR ---`
- **L2 (System)**: System-level concatenation with `--- MODULE SEPARATOR ---`

## Example Files

### Verification Data
- `xsv-verification-*.md` - Shows all 59 files have both L1 and L2 content populated

### L1 Window Content Example  
- `xsv-l1-content-00001.md` - Shows concatenated content of all 21 files from `./xsv/src/cmd/` directory (96.71 KB)

### L2 Window Content Examples
- `xsv-l2-content-00001.md` - L2 content for cmd files (96.75 KB)
- `xsv-l2-main-00001.md` - L2 content for main.rs (115.75 KB, includes more system files)

## Key Insights

1. **No Truncation**: All content properly concatenated with separators
2. **Correct Hierarchical Logic**: Different L2 content based on system-level grouping
3. **Performance**: 59 files processed in 1.56s with multi-scale context population
4. **Ready for Analysis**: Database now supports triple-comparison analysis (Individual ↔ L1 ↔ L2)

## Database Schema

The enhanced schema includes:

```sql
CREATE TABLE INGEST_YYYYMMDDHHMMSS (
    -- Original 15 columns
    file_id BIGSERIAL PRIMARY KEY,
    ingestion_id BIGINT,
    filepath VARCHAR NOT NULL,
    filename VARCHAR NOT NULL,
    extension VARCHAR,
    file_size_bytes BIGINT NOT NULL,
    line_count INTEGER,
    word_count INTEGER,
    token_count INTEGER,
    content_text TEXT,
    file_type VARCHAR NOT NULL,
    conversion_command VARCHAR,
    relative_path VARCHAR NOT NULL,
    absolute_path VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- New 4 columns for multi-scale context
    parent_filepath VARCHAR,          -- Calculated: go back by 1 slash
    l1_window_content TEXT,           -- Directory-level concatenation  
    l2_window_content TEXT,           -- System-level concatenation
    ast_patterns JSONB                -- Pattern matches for semantic search
);
```

This enhancement enables systematic knowledge arbitrage analysis with immediate access to multi-scale context for any file.