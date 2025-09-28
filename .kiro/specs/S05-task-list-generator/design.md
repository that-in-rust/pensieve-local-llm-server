# Design Document

## Overview

The Task List Generator creates systematic codebase analysis workflows by transforming database rows into hierarchical analysis tasks with multi-layered contextual content files. The system enables knowledge arbitrage through structured L1-L8 analysis of mature codebases.

## Architecture

### Core Components

1. **ContentExtractor** - Extracts and contextualizes database content into A/B/C files
2. **HierarchicalTaskDivider** - Creates 4-level task hierarchies with 7 groups per level
3. **DatabaseQueryEngine** - Handles SQL operations for row counting and content extraction
4. **L1L8MarkdownGenerator** - Generates analysis tasks with L1-L8 methodology references

### Data Flow

```
Database Table → ContentExtractor → A/B/C Files → HierarchicalTaskDivider → 4-Level Tasks → Markdown Output
```

## Components and Interfaces

### ContentExtractor

```rust
pub struct ContentExtractor {
    db_pool: Arc<sqlx::PgPool>,
    output_dir: PathBuf,
}

impl ContentExtractor {
    pub async fn extract_all_rows(&self, table_name: &str) -> TaskResult<Vec<ContentTriple>>;
    pub async fn create_content_files(&self, row: &QueryResultRow, row_num: usize) -> TaskResult<ContentTriple>;
    fn generate_l1_context(&self, content: &str, metadata: &RowMetadata) -> String;
    fn generate_l2_context(&self, content: &str, metadata: &RowMetadata) -> String;
}

pub struct ContentTriple {
    pub content_a: PathBuf,    // Raw content
    pub content_b: PathBuf,    // L1 context
    pub content_c: PathBuf,    // L2 context
    pub row_number: usize,
}
```

### HierarchicalTaskDivider

```rust
pub struct HierarchicalTaskDivider {
    levels: usize,
    groups_per_level: usize,
}

impl HierarchicalTaskDivider {
    pub fn new(levels: usize, groups_per_level: usize) -> Self;
    pub fn create_hierarchy(&self, content_triples: Vec<ContentTriple>) -> TaskResult<TaskHierarchy>;
    fn distribute_across_levels(&self, items: Vec<ContentTriple>, current_level: usize) -> Vec<TaskLevel>;
}

pub struct TaskHierarchy {
    pub levels: Vec<TaskLevel>,
    pub total_tasks: usize,
}

pub struct TaskLevel {
    pub level: usize,
    pub groups: Vec<HierarchicalTaskGroup>,
}

pub struct HierarchicalTaskGroup {
    pub id: String,           // e.g., "1.2.3"
    pub title: String,
    pub tasks: Vec<AnalysisTask>,
    pub sub_groups: Vec<HierarchicalTaskGroup>,
}
```

### DatabaseQueryEngine

```rust
pub struct DatabaseQueryEngine {
    pool: Arc<sqlx::PgPool>,
}

impl DatabaseQueryEngine {
    pub async fn count_rows(&self, table_name: &str) -> TaskResult<usize>;
    pub async fn get_all_rows(&self, table_name: &str) -> TaskResult<Vec<QueryResultRow>>;
    pub async fn validate_table_exists(&self, table_name: &str) -> TaskResult<bool>;
}
```

### L1L8MarkdownGenerator

```rust
pub struct L1L8MarkdownGenerator {
    prompt_file: PathBuf,
    output_dir: PathBuf,
}

impl L1L8MarkdownGenerator {
    pub fn generate_hierarchical_markdown(&self, hierarchy: &TaskHierarchy, table_name: &str) -> TaskResult<String>;
    fn create_analysis_task(&self, content_triple: &ContentTriple, task_id: &str, table_name: &str) -> String;
    fn format_l1l8_analysis_instructions(&self) -> String;
}
```

## Data Models

### AnalysisTask

```rust
pub struct AnalysisTask {
    pub id: String,                    // e.g., "1.2.3.4"
    pub table_name: String,
    pub row_number: usize,
    pub content_files: ContentTriple,
    pub prompt_file: PathBuf,
    pub output_file: PathBuf,
    pub analysis_stages: Vec<AnalysisStage>,
}

pub enum AnalysisStage {
    AnalyzeA,                         // A alone
    AnalyzeAInContextB,               // A in context of B
    AnalyzeBInContextC,               // B in context of C
    AnalyzeAInContextBC,              // A in context of B & C
}
```

### CLI Commands

```rust
pub enum TaskCommand {
    CountRows { table_name: String },
    ExtractContent { 
        table_name: String, 
        output_dir: PathBuf 
    },
    GenerateHierarchicalTasks { 
        table_name: String, 
        levels: usize, 
        groups_per_level: usize,
        output_file: PathBuf 
    },
}
```

## Error Handling

```rust
#[derive(Error, Debug)]
pub enum TaskError {
    #[error("Table {table_name} not found")]
    TableNotFound { table_name: String },
    
    #[error("Content extraction failed for row {row}: {cause}")]
    ContentExtractionFailed { row: usize, cause: String },
    
    #[error("Hierarchical division failed: {cause}")]
    HierarchicalDivisionFailed { cause: String },
    
    #[error("L1/L2 context generation failed: {cause}")]
    ContextGenerationFailed { cause: String },
}
```

## Testing Strategy

### Unit Tests
- ContentExtractor: Test A/B/C file generation with mock database rows
- HierarchicalTaskDivider: Test 4-level distribution with various row counts
- DatabaseQueryEngine: Test SQL operations with test database
- L1L8MarkdownGenerator: Test markdown format compliance

### Integration Tests
- End-to-end workflow: Database → Content → Tasks → Markdown
- CLI command execution with real database tables
- File system operations and directory creation

### Performance Tests
- Large table processing (10,000+ rows)
- Memory usage during hierarchical task creation
- File I/O performance for content extraction

## Implementation Notes

### Context Generation Strategy

**L1 Context (Immediate):**
- Same directory files
- Import/include relationships
- Module-level dependencies

**L2 Context (Architectural):**
- Package/crate structure
- Cross-module relationships
- Architectural patterns and constraints

### Hierarchical Distribution Algorithm

For N rows and 7 groups per level across 4 levels:
- Level 1: 7 main groups
- Level 2: 7 sub-groups per main group (49 total)
- Level 3: 7 sub-sub-groups per sub-group (343 total)
- Level 4: Individual tasks distributed across leaf groups

Mathematical distribution ensures even task allocation while maintaining the 7-group structure at each level.