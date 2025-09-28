# Design Document

## Overview

The S04 Knowledge Arbitrage system implements a revolutionary multi-scale context window approach to extract L1-L8 insights from the burnt-sushi/xsv codebase. This system transforms traditional file-by-file analysis into hierarchical knowledge extraction that mirrors how expert programmers understand code architecture.

The design leverages our existing code-ingest infrastructure, enhances it with semantic search capabilities via ast-grep, and implements a triple-comparison analysis framework to systematically extract decades of engineering wisdom for The Horcrux Codex.

## Architecture

### System Overview

```mermaid
graph TD
    A[XSV Repository] --> B[Code-Ingest Tool]
    B --> C[INGEST_20250928062949 Table]
    C --> D[Database Enhancement Tasks]
    D --> E[Enhanced Source Table]
    E --> F[L1-L8 Analysis Tasks]
    F --> G[QUERYRESULT_xsv_knowledge_arbitrage Table]
    G --> H[Export Tasks]
    H --> I[Horcrux Codex Dataset]
    H --> J[Mermaid Visualizations]
    H --> K[Markdown Reports]
    
    subgraph "Task-Based Workflow"
        D --> D1[Task: Add parent_filepath]
        D --> D2[Task: Add l1_window_content]
        D --> D3[Task: Add l2_window_content]
        D --> D4[Task: Add ast_patterns]
        F --> F1[Task: L1-L3 Extraction]
        F --> F2[Task: L4-L6 Extraction]
        F --> F3[Task: L7-L8 Extraction]
        F --> F4[Task: Triple-Comparison]
        H --> H1[Task: Generate Horcrux Entries]
        H --> H2[Task: Generate Visualizations]
    end
    
    subgraph "Database Tables"
        C --> C1[Source: INGEST_20250928062949]
        G --> G1[Results: QUERYRESULT_xsv_knowledge_arbitrage]
        G --> G2[Metadata: analysis_meta]
    end
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant R as XSV Repository
    participant CI as Code-Ingest
    participant DB as PostgreSQL
    participant MSE as Multi-Scale Engine
    participant AST as ast-grep
    participant L18 as L1-L8 Extractor
    participant OUT as Output Generator
    
    R->>CI: Repository URL
    CI->>DB: Ingest 59 files
    DB->>MSE: Raw file content
    MSE->>DB: Enhanced schema with context windows
    DB->>AST: Semantic pattern queries
    AST->>DB: Structured pattern matches
    DB->>L18: Multi-scale context data
    L18->>OUT: Knowledge arbitrage insights
    OUT->>OUT: Generate visualizations & reports
```

## Components and Interfaces

### 1. Database Enhancement Module

**Purpose**: Transform flat file storage into hierarchical context-aware database

**Interface**:
```rust
pub trait DatabaseEnhancer {
    async fn add_parent_filepath_column(&self) -> Result<()>;
    async fn populate_l1_window_content(&self) -> Result<()>;
    async fn populate_l2_window_content(&self) -> Result<()>;
    async fn add_ast_patterns_column(&self) -> Result<()>;
    async fn verify_enhancement_integrity(&self) -> Result<EnhancementReport>;
}
```

**Implementation Strategy**:
- Simple path logic: go back by 1 slash for parent_filepath
- Window functions for content aggregation
- Deterministic ordering: `ORDER BY parent_filepath, filepath`
- Analytics-first design accepting redundancy

### 2. Semantic Search Integration

**Purpose**: Replace text-based grep with AST-aware pattern matching

**Interface**:
```rust
pub trait SemanticSearcher {
    async fn extract_patterns(&self, pattern: &str, language: &str) -> Result<Vec<AstMatch>>;
    async fn find_optimization_patterns(&self) -> Result<OptimizationReport>;
    async fn extract_architectural_patterns(&self) -> Result<ArchitectureReport>;
    async fn identify_language_limitations(&self) -> Result<LanguageLimitationReport>;
}
```

**Key Patterns for XSV**:
```bash
# L1 Micro-optimizations
ast-grep run -p 'unsafe { $$$ }' -l rust
ast-grep run -p 'Vec::with_capacity($CAP)' -l rust
ast-grep run -p 'BufReader::new($READER)' -l rust

# L2 Design patterns
ast-grep run -p 'impl<$GENERICS> $TRAIT for $TYPE { $$$ }' -l rust
ast-grep run -p 'Result<$OK, $ERR>' -l rust

# L3 Micro-library opportunities
ast-grep run -p 'pub struct $NAME { $$$ }' -l rust
ast-grep run -p 'pub fn $NAME($ARGS) -> $RET' -l rust
```

### 3. Triple-Comparison Analysis Engine

**Purpose**: Systematic multi-scale pattern recognition

**Interface**:
```rust
pub trait TripleComparator {
    async fn compare_individual_vs_module(&self, file_id: i64) -> Result<ComparisonInsights>;
    async fn compare_individual_vs_system(&self, file_id: i64) -> Result<ComparisonInsights>;
    async fn compare_module_vs_system(&self, parent_path: &str) -> Result<ComparisonInsights>;
    async fn generate_scaling_analysis(&self) -> Result<ScalingReport>;
}
```

**Analysis Framework**:
1. **Individual vs L1**: How file patterns compose within modules
2. **Individual vs L2**: How file patterns relate to system architecture  
3. **L1 vs L2**: How module patterns scale to system-wide principles

### 4. L1-L8 Extraction Engine

**Purpose**: Systematic knowledge arbitrage across all extraction levels

**Interface**:
```rust
pub trait KnowledgeExtractor {
    // Tactical Implementation (L1-L3)
    async fn extract_micro_optimizations(&self) -> Result<L1Report>;
    async fn extract_design_patterns(&self) -> Result<L2Report>;
    async fn identify_micro_libraries(&self) -> Result<L3Report>;
    
    // Strategic Architecture (L4-L6)
    async fn identify_macro_opportunities(&self) -> Result<L4Report>;
    async fn analyze_architecture_decisions(&self) -> Result<L5Report>;
    async fn examine_hardware_interaction(&self) -> Result<L6Report>;
    
    // Foundational Evolution (L7-L8)
    async fn identify_language_limitations(&self) -> Result<L7Report>;
    async fn perform_intent_archaeology(&self) -> Result<L8Report>;
}
```

### 5. Multi-Persona Expert Council

**Purpose**: Challenge assumptions and ensure comprehensive analysis

**Interface**:
```rust
pub trait ExpertCouncil {
    async fn activate_domain_expert(&self, context: &AnalysisContext) -> Result<DomainInsights>;
    async fn activate_strategic_analyst(&self, context: &AnalysisContext) -> Result<StrategyInsights>;
    async fn activate_implementation_specialist(&self, context: &AnalysisContext) -> Result<ImplInsights>;
    async fn activate_ux_advocate(&self, context: &AnalysisContext) -> Result<UXInsights>;
    async fn activate_skeptical_engineer(&self, context: &AnalysisContext) -> Result<SkepticalChallenges>;
    async fn synthesize_council_insights(&self, all_insights: Vec<Insight>) -> Result<RefinedInsights>;
}
```

## Data Models

### Enhanced Database Schema

```sql
-- Enhanced ingestion table with multi-scale context
CREATE TABLE enhanced_xsv_analysis (
    -- Original columns from INGEST_20250928062949
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
    relative_path VARCHAR NOT NULL,
    absolute_path VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Multi-scale context enhancements
    parent_filepath VARCHAR NOT NULL,
    l1_window_content TEXT,           -- Directory-level concatenation
    l2_window_content TEXT,           -- System-level concatenation
    
    -- Semantic search results
    ast_patterns JSONB,               -- ast-grep pattern matches
    semantic_metadata JSONB,          -- Extracted meta-variables
    
    -- Analysis results
    l1_insights JSONB,                -- Micro-optimization insights
    l2_insights JSONB,                -- Design pattern insights
    l3_insights JSONB,                -- Micro-library opportunities
    l4_insights JSONB,                -- Macro-library opportunities
    l5_insights JSONB,                -- Architecture decisions
    l6_insights JSONB,                -- Hardware interaction patterns
    l7_insights JSONB,                -- Language limitations
    l8_insights JSONB,                -- Intent archaeology
    
    -- Comparison analysis
    triple_comparison_results JSONB,  -- Individual↔L1↔L2 comparisons
    scaling_patterns JSONB,           -- How patterns scale across levels
    
    -- Metadata
    analysis_timestamp TIMESTAMP DEFAULT NOW(),
    analysis_version VARCHAR DEFAULT '1.0'
);

-- Indexes for efficient querying
CREATE INDEX idx_parent_filepath ON enhanced_xsv_analysis(parent_filepath);
CREATE INDEX idx_extension ON enhanced_xsv_analysis(extension);
CREATE INDEX idx_ast_patterns ON enhanced_xsv_analysis USING gin(ast_patterns);
CREATE INDEX idx_l1_insights ON enhanced_xsv_analysis USING gin(l1_insights);
```

### Knowledge Arbitrage Output Schema

```rust
#[derive(Serialize, Deserialize, Debug)]
pub struct KnowledgeArbitrageOutput {
    pub metadata: AnalysisMetadata,
    pub tactical_implementation: TacticalInsights,    // L1-L3
    pub strategic_architecture: StrategyInsights,     // L4-L6
    pub foundational_evolution: FoundationInsights,   // L7-L8
    pub cross_scale_patterns: CrossScalePatterns,
    pub horcrux_codex_entries: Vec<HorcruxEntry>,
    pub visualizations: Vec<MermaidDiagram>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AnalysisMetadata {
    pub xsv_commit_hash: String,
    pub analysis_timestamp: DateTime<Utc>,
    pub database_table: String,
    pub files_analyzed: u32,
    pub extraction_completeness: ExtractionCompleteness,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TacticalInsights {
    pub micro_optimizations: Vec<MicroOptimization>,  // L1
    pub design_patterns: Vec<DesignPattern>,          // L2
    pub micro_libraries: Vec<MicroLibraryOpportunity>, // L3
}
```

## Error Handling

### Robust Analysis Pipeline

```rust
#[derive(Error, Debug)]
pub enum AnalysisError {
    #[error("Database enhancement failed: {cause}")]
    DatabaseEnhancement { cause: String },
    
    #[error("ast-grep pattern failed: {pattern} - {cause}")]
    SemanticSearchFailed { pattern: String, cause: String },
    
    #[error("L{level} extraction failed: {cause}")]
    ExtractionFailed { level: u8, cause: String },
    
    #[error("Triple comparison failed for file {file_id}: {cause}")]
    ComparisonFailed { file_id: i64, cause: String },
    
    #[error("Expert council synthesis failed: {cause}")]
    CouncilSynthesisFailed { cause: String },
    
    #[error("Output generation failed: {format} - {cause}")]
    OutputGenerationFailed { format: String, cause: String },
}

pub type AnalysisResult<T> = Result<T, AnalysisError>;
```

### Graceful Degradation Strategy

1. **Partial Analysis**: Continue with available data if some files fail
2. **Pattern Fallback**: Use text search if ast-grep patterns fail
3. **Context Reconstruction**: Rebuild missing context windows from available data
4. **Insight Validation**: Cross-validate insights across multiple extraction methods

## Testing Strategy

### Test Plan for Multi-Scale Analysis

#### Unit Tests
```rust
#[tokio::test]
async fn test_parent_filepath_calculation() {
    assert_eq!(calculate_parent_filepath("./xsv/src/cmd/sort.rs"), "./xsv/src/cmd");
    assert_eq!(calculate_parent_filepath("./xsv/README.md"), "./xsv");
    assert_eq!(calculate_parent_filepath("README.md"), "README.md");
}

#[tokio::test]
async fn test_l1_window_content_generation() {
    let files = vec![
        ("./xsv/src/cmd/sort.rs", "sort code"),
        ("./xsv/src/cmd/join.rs", "join code"),
    ];
    let l1_content = generate_l1_window_content(&files).await.unwrap();
    assert!(l1_content.contains("sort code"));
    assert!(l1_content.contains("join code"));
    assert!(l1_content.contains("--- FILE SEPARATOR ---"));
}

#[tokio::test]
async fn test_ast_grep_pattern_extraction() {
    let pattern = "struct $NAME { $$$ }";
    let results = extract_ast_patterns(pattern, "rust", "./test-data/").await.unwrap();
    assert!(!results.is_empty());
    assert!(results[0].meta_variables.contains_key("NAME"));
}
```

#### Integration Tests
```rust
#[tokio::test]
async fn test_complete_xsv_analysis_pipeline() {
    // Test the entire pipeline from database enhancement to output generation
    let analyzer = XSVAnalyzer::new(test_database_url()).await.unwrap();
    
    // Enhance database schema
    analyzer.enhance_database_schema().await.unwrap();
    
    // Run semantic search
    let patterns = analyzer.extract_all_patterns().await.unwrap();
    assert!(!patterns.is_empty());
    
    // Perform L1-L8 extraction
    let insights = analyzer.extract_all_insights().await.unwrap();
    assert!(insights.tactical_implementation.micro_optimizations.len() > 0);
    
    // Generate outputs
    let outputs = analyzer.generate_knowledge_arbitrage_outputs().await.unwrap();
    assert!(!outputs.horcrux_codex_entries.is_empty());
}
```

#### Performance Tests
```rust
#[tokio::test]
async fn test_analysis_performance_contracts() {
    let start = Instant::now();
    
    // L1-L3 extraction should complete in <30 seconds
    let tactical_insights = extract_tactical_insights().await.unwrap();
    assert!(start.elapsed() < Duration::from_secs(30));
    
    // Database queries should be <1 second for single file analysis
    let query_start = Instant::now();
    let file_analysis = analyze_single_file(1).await.unwrap();
    assert!(query_start.elapsed() < Duration::from_secs(1));
}
```

#### Validation Tests
```rust
#[tokio::test]
async fn test_knowledge_arbitrage_quality() {
    let insights = extract_all_insights().await.unwrap();
    
    // Verify L1 insights contain concrete optimizations
    assert!(insights.tactical_implementation.micro_optimizations
        .iter()
        .any(|opt| opt.performance_impact.is_some()));
    
    // Verify L8 insights contain historical context
    assert!(insights.foundational_evolution.intent_archaeology
        .iter()
        .any(|arch| arch.commit_references.len() > 0));
    
    // Verify cross-scale pattern consistency
    validate_cross_scale_consistency(&insights).unwrap();
}
```

### Verification Framework

#### Expert Council Validation
```rust
pub async fn validate_insights_with_expert_council(
    insights: &KnowledgeArbitrageOutput
) -> AnalysisResult<ValidationReport> {
    let mut challenges = Vec::new();
    
    // Skeptical Engineer challenges
    let skeptical_challenges = generate_skeptical_challenges(insights).await?;
    challenges.extend(skeptical_challenges);
    
    // Domain Expert validation
    let domain_validation = validate_with_domain_expert(insights).await?;
    
    // Generate fact-checkable questions
    let verification_questions = generate_verification_questions(insights).await?;
    
    Ok(ValidationReport {
        challenges,
        domain_validation,
        verification_questions,
        overall_confidence: calculate_confidence_score(insights),
    })
}
```

## Implementation Phases

### Phase 1: Database Enhancement (Week 1)
- Implement parent_filepath calculation
- Add l1_window_content and l2_window_content columns
- Populate hierarchical context data
- Verify data integrity and performance

### Phase 2: Semantic Search Integration (Week 1)
- Integrate ast-grep with existing codebase
- Implement XSV-specific pattern library
- Store semantic search results in database
- Test pattern extraction accuracy

### Phase 3: L1-L8 Extraction Engine (Week 2)
- Implement tactical implementation extraction (L1-L3)
- Implement strategic architecture analysis (L4-L6)
- Implement foundational evolution insights (L7-L8)
- Validate extraction quality

### Phase 4: Triple-Comparison Framework (Week 2)
- Implement individual↔module↔system comparisons
- Generate scaling pattern analysis
- Validate cross-scale consistency
- Optimize comparison performance

### Phase 5: Output Generation (Week 3)
- Generate Horcrux Codex training data
- Create Mermaid visualizations
- Generate markdown reports
- Implement export functionality

### Phase 6: Validation and Refinement (Week 3)
- Expert council validation
- Performance optimization
- Documentation completion
- Final quality assurance

This design provides a comprehensive foundation for transforming the XSV codebase analysis from traditional file processing into systematic Knowledge Arbitrage extraction, directly supporting the mission to achieve top-5 Rust programmer mastery through decades of accumulated engineering wisdom.