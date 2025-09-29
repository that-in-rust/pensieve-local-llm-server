# Code Ingest Documentation

This directory contains comprehensive documentation for the Code Ingest project, focusing on the core ingestion engine that transforms codebases into queryable PostgreSQL databases.

## 📁 Directory Structure

### `/analysis/` - Technical Analysis & Design Documents
- **`task_volume_analysis.md`** - Analysis of specialized task generation features (Kiro-specific)
- **`windowed_task_system_design.md`** - Advanced task management system (experimental)
- **`generic_task_generator_analysis.md`** - Task generator improvements (specialized use case)
- **`fix_demonstration.md`** - Problem analysis and solution demonstration
- **`COMMIT_SUMMARY.md`** - Summary of major commits and changes
- **`FINAL_VALIDATION_SUMMARY.md`** - Complete validation results and success metrics

### `/testing/` - Test Files & Validation Scripts
- **`test_windowed_system.rs`** - Demonstration of advanced task features
- **`test_generic_task_generator.rs`** - Validation of specialized generators
- **`test_final_format_validation.rs`** - Format compatibility validation
- **`test_reference_format_match.rs`** - Reference format matching tests
- **`test_simple_generator.rs`** - Basic format validation tests

## 🎯 Core Focus: Database Ingestion

### Primary Value: PostgreSQL Integration
- **Issue**: Manual codebase analysis is time-consuming and inconsistent
- **Solution**: Automated ingestion into queryable PostgreSQL databases
- **Result**: 100+ files/second processing with full-text search and metadata analysis

### Technical Implementation
1. **Ingestion Engine** - High-performance file processing and database storage
2. **Multi-Scale Context** - Directory and system-level relationship mapping
3. **SQL Interface** - Full-text search and metadata queries
4. **Performance Optimization** - Efficient batch processing and indexing

### Results
- ✅ **High Performance**: 100+ files/second ingestion speed
- ✅ **Full-Text Search**: Indexed content for pattern matching
- ✅ **Metadata Queries**: File types, sizes, complexity analysis
- ✅ **Relationship Mapping**: Directory and system context preservation

## 🚀 Primary Usage Examples

### Ingest GitHub Repository
```bash
code-ingest ingest https://github.com/user/repo --db-path ./analysis
```

### Ingest Local Codebase
```bash
code-ingest ingest /path/to/code --folder-flag --db-path ./analysis
```

### Query Your Data
```bash
# Find async functions
code-ingest sql "SELECT filepath FROM TABLE_NAME WHERE content_text LIKE '%async fn%'" --db-path ./analysis

# Analyze file complexity
code-ingest sql "SELECT extension, AVG(line_count) FROM TABLE_NAME GROUP BY extension" --db-path ./analysis
```

## 📊 Performance Metrics

- **Ingestion Speed**: 100+ files/second processing
- **Database Efficiency**: Optimized PostgreSQL schema with full-text indexing
- **Memory Usage**: Constant ~10-25MB regardless of repository size
- **Query Performance**: Sub-second response times for most queries

## 📝 Note on Task Generation

The task generation features documented in this directory are **specialized tools** designed for specific IDE integration (Kiro). The core value of Code Ingest is the **database ingestion engine** that works universally with any PostgreSQL-compatible analysis workflow.

## 🔗 Related Files

- **Main README**: `../README.md` - Project overview and quick start
- **Long Form README**: `../READMELongForm20250929.md` - Comprehensive documentation with examples
- **Source Code**: `../code-ingest/src/` - Implementation source code
- **Specifications**: `../.kiro/specs/` - Task specifications and examples