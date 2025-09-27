# GitIngest Code Ingestion Patterns

## Core Architecture Insights

Based on analysis of the GitIngest codebase, here are the key patterns for ingesting code from GitHub repositories:

### 1. Repository Cloning Strategy
- **Partial Clone Support**: Use `git clone --filter=blob:none` for large repos
- **Branch/Tag/Commit Resolution**: Support specific refs, not just main branch
- **Authentication**: Handle GitHub PAT tokens for private repos
- **Submodule Support**: Optional recursive submodule cloning
- **Timeout Handling**: Async operations with configurable timeouts

### 2. File System Traversal
- **Recursive Directory Walking**: Process directories depth-first
- **Pattern Matching**: Support .gitignore and custom include/exclude patterns
- **Size Limits**: Configurable max file size and total size limits
- **File Type Detection**: Handle text vs binary files appropriately
- **Symlink Handling**: Process symlinks safely without infinite loops

### 3. Content Processing
- **Text Extraction**: Read file contents with encoding detection
- **Token Counting**: Estimate LLM tokens using tiktoken
- **Output Formatting**: Generate structured text output for LLM consumption
- **Statistics Tracking**: File counts, sizes, directory depth metrics
- **Error Handling**: Graceful handling of unreadable files

### 4. Key Data Structures
```python
# Core node representation
FileSystemNode:
  - name: str
  - type: FILE | DIRECTORY
  - size: int
  - content: str (for files)
  - children: List[FileSystemNode] (for directories)
  - path_str: str (relative path)

# Query configuration
IngestionQuery:
  - slug: str (repo identifier)
  - local_path: Path
  - subpath: str (subdirectory focus)
  - max_file_size: int
  - include_patterns: List[str]
  - ignore_patterns: List[str]
```

### 5. Performance Optimizations
- **Lazy Loading**: Only read file contents when needed
- **Streaming**: Process large repositories incrementally
- **Caching**: Cache cloned repositories for repeated access
- **Parallel Processing**: Async file operations where possible
- **Memory Management**: Cleanup temporary files and directories

### 6. PostgreSQL Integration Considerations
For our use case of ingesting to PostgreSQL:
- **Chunking Strategy**: Split large files into manageable chunks
- **Metadata Storage**: Store file paths, sizes, types in structured tables
- **Content Indexing**: Use PostgreSQL full-text search capabilities
- **Relationship Mapping**: Track file dependencies and imports
- **Version Tracking**: Handle repository updates and diffs

## Implementation Priorities (Shreyas Doshi Style)

### P0 (Must Have)
1. **Basic GitHub Clone**: Clone public repos to temp directory
2. **File Traversal**: Walk directory tree with basic filtering
3. **PostgreSQL Storage**: Store file content and metadata in DB
4. **Simple Query Interface**: Basic search and retrieval

### P1 (Should Have)
1. **Private Repo Support**: GitHub PAT token authentication
2. **Pattern Filtering**: .gitignore and custom include/exclude
3. **Content Chunking**: Handle large files efficiently
4. **Incremental Updates**: Detect and process only changed files

### P2 (Nice to Have)
1. **Submodule Support**: Recursive repository processing
2. **Advanced Search**: Semantic search with embeddings
3. **Diff Tracking**: Version history and change detection
4. **Performance Optimization**: Caching and parallel processing

## Key Terminal Commands for Analysis
```bash
# Find core implementation files
grep -n "def ingest" /path/to/gitingest/file.txt
sed -n 'start,end p' /path/to/file.txt
grep -A 10 -B 5 "pattern" /path/to/file.txt

# Analyze file structure
head -100 /path/to/file.txt
tail -100 /path/to/file.txt
wc -l /path/to/file.txt
```

This steering document captures the essential patterns from GitIngest that we should incorporate into our PostgreSQL-based code ingestion system.