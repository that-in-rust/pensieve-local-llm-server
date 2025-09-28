//! Integration test for L1-L8 task generation workflow
//!
//! This test verifies the complete workflow from content extraction to hierarchical
//! task generation with proper A/B/C file references and L1-L8 analysis methodology.

use code_ingest::tasks::{
    ContentExtractor, ContentTriple, HierarchicalTaskDivider, L1L8MarkdownGenerator,
    AnalysisTask, AnalysisStage
};
use std::path::PathBuf;
use tempfile::TempDir;

/// Test the complete L1-L8 task generation workflow
#[tokio::test]
async fn test_complete_l1l8_task_generation_workflow() {
    // Create temporary directories for testing
    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join(".raw_data_202509");
    let work_area = temp_dir.path().join("gringotts/WorkArea");
    
    // Create test content triples (simulating ContentExtractor output)
    let content_triples = vec![
        ContentTriple {
            content_a: output_dir.join("INGEST_20250928101039_1_Content.txt"),
            content_b: output_dir.join("INGEST_20250928101039_1_Content_L1.txt"),
            content_c: output_dir.join("INGEST_20250928101039_1_Content_L2.txt"),
            row_number: 1,
            table_name: "INGEST_20250928101039".to_string(),
        },
        ContentTriple {
            content_a: output_dir.join("INGEST_20250928101039_2_Content.txt"),
            content_b: output_dir.join("INGEST_20250928101039_2_Content_L1.txt"),
            content_c: output_dir.join("INGEST_20250928101039_2_Content_L2.txt"),
            row_number: 2,
            table_name: "INGEST_20250928101039".to_string(),
        },
        ContentTriple {
            content_a: output_dir.join("INGEST_20250928101039_3_Content.txt"),
            content_b: output_dir.join("INGEST_20250928101039_3_Content_L1.txt"),
            content_c: output_dir.join("INGEST_20250928101039_3_Content_L2.txt"),
            row_number: 3,
            table_name: "INGEST_20250928101039".to_string(),
        },
    ];

    // Create hierarchical task divider
    let divider = HierarchicalTaskDivider::new(2, 2).unwrap(); // 2 levels, 2 groups for simplicity
    
    // Create task hierarchy
    let hierarchy = divider.create_hierarchy(content_triples).unwrap();
    
    // Verify hierarchy structure
    assert_eq!(hierarchy.total_tasks, 3);
    assert!(!hierarchy.levels.is_empty());

    // Create L1L8 markdown generator
    let generator = L1L8MarkdownGenerator::new(
        PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
        work_area.clone(),
    );

    // Generate hierarchical markdown
    let markdown = generator.generate_hierarchical_markdown(&hierarchy, "INGEST_20250928101039").await.unwrap();

    // Verify markdown structure and content
    assert!(markdown.contains("# L1-L8 Analysis Tasks for INGEST_20250928101039"));
    assert!(markdown.contains("## Task Generation Metadata"));
    assert!(markdown.contains("**Source Table**: `INGEST_20250928101039`"));
    assert!(markdown.contains("**Total Tasks**: 3"));
    
    // Verify L1-L8 methodology section
    assert!(markdown.contains("## L1-L8 Analysis Methodology"));
    assert!(markdown.contains("L1: Idiomatic Patterns"));
    assert!(markdown.contains("L8: The Meta-Context"));
    
    // Verify task hierarchy section
    assert!(markdown.contains("## Task Hierarchy"));
    
    // Verify proper task format with A/B/C references
    assert!(markdown.contains("Analyze INGEST_20250928101039 row"));
    assert!(markdown.contains("**Content**: "));
    assert!(markdown.contains("INGEST_20250928101039_1_Content.txt` as A"));
    assert!(markdown.contains("INGEST_20250928101039_1_Content_L1.txt` as B"));
    assert!(markdown.contains("INGEST_20250928101039_1_Content_L2.txt` as C"));
    
    // Verify prompt and output references
    assert!(markdown.contains("**Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md`"));
    assert!(markdown.contains("where you try to find insights of A alone ; A in context of B ; B in context of C ; A in context B & C"));
    assert!(markdown.contains("**Output**: `gringotts/WorkArea/INGEST_20250928101039_"));
    
    // Verify analysis stages
    assert!(markdown.contains("**Analysis Stages**:"));
    assert!(markdown.contains("Analyze A alone"));
    assert!(markdown.contains("A in context of B"));
    assert!(markdown.contains("B in context of C"));
    assert!(markdown.contains("A in context of B & C"));
    
    // Verify processing instructions
    assert!(markdown.contains("## Processing Instructions"));
    assert!(markdown.contains("### How to Execute These Tasks"));
    assert!(markdown.contains("### File Structure"));
    assert!(markdown.contains("### Analysis Commands"));

    println!("Generated markdown preview:");
    println!("{}", &markdown[..std::cmp::min(markdown.len(), 1000)]);
    println!("... (truncated)");
}

/// Test individual analysis task format compliance
#[test]
fn test_analysis_task_format_compliance() {
    let generator = L1L8MarkdownGenerator::new(
        PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
        PathBuf::from("gringotts/WorkArea"),
    );
    
    // Create test analysis task matching the requirements format
    let task = AnalysisTask {
        id: "5".to_string(),
        table_name: "INGEST_20250928101039".to_string(),
        row_number: 35,
        content_files: ContentTriple {
            content_a: PathBuf::from(".raw_data_202509/INGEST_20250928101039_35_Content.txt"),
            content_b: PathBuf::from(".raw_data_202509/INGEST_20250928101039_35_Content_L1.txt"),
            content_c: PathBuf::from(".raw_data_202509/INGEST_20250928101039_35_Content_L2.txt"),
            row_number: 35,
            table_name: "INGEST_20250928101039".to_string(),
        },
        prompt_file: PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
        output_file: PathBuf::from("gringotts/WorkArea/INGEST_20250928101039_35.md"),
        analysis_stages: vec![
            AnalysisStage::AnalyzeA,
            AnalysisStage::AnalyzeAInContextB,
            AnalysisStage::AnalyzeBInContextC,
            AnalysisStage::AnalyzeAInContextBC,
        ],
    };
    
    let task_markdown = generator.create_analysis_task(&task, "INGEST_20250928101039");
    
    // Verify exact format compliance with requirements example
    let expected_format = "- [ ] 5. Analyze INGEST_20250928101039 row 35";
    assert!(task_markdown.contains(expected_format));
    
    let expected_content = "**Content**: `.raw_data_202509/INGEST_20250928101039_35_Content.txt` as A + `.raw_data_202509/INGEST_20250928101039_35_Content_L1.txt` as B + `.raw_data_202509/INGEST_20250928101039_35_Content_L2.txt` as C";
    assert!(task_markdown.contains(expected_content));
    
    let expected_prompt = "**Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in context of C ; A in context B & C";
    assert!(task_markdown.contains(expected_prompt));
    
    let expected_output = "**Output**: `gringotts/WorkArea/INGEST_20250928101039_35.md`";
    assert!(task_markdown.contains(expected_output));
    
    println!("Generated task format:");
    println!("{}", task_markdown);
}

/// Test L1-L8 analysis stages format
#[test]
fn test_l1l8_analysis_stages_format() {
    let generator = L1L8MarkdownGenerator::new(
        PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
        PathBuf::from("gringotts/WorkArea"),
    );
    
    // Test each analysis stage format
    let stages = vec![
        (AnalysisStage::AnalyzeA, "Analyze A alone: Extract insights from raw content"),
        (AnalysisStage::AnalyzeAInContextB, "A in context of B: Understand content within immediate file context"),
        (AnalysisStage::AnalyzeBInContextC, "B in context of C: Understand immediate context within architectural context"),
        (AnalysisStage::AnalyzeAInContextBC, "A in context of B & C: Synthesize insights across all contextual layers"),
    ];
    
    for (stage, expected_description) in stages {
        let formatted = generator.format_analysis_stage(&stage);
        assert_eq!(formatted, expected_description);
    }
}

/// Test gringotts/WorkArea output path generation
#[test]
fn test_gringotts_workarea_output_path_generation() {
    let generator = L1L8MarkdownGenerator::new(
        PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
        PathBuf::from("gringotts/WorkArea"),
    );
    
    let task = AnalysisTask {
        id: "1.2.3.4".to_string(),
        table_name: "TEST_TABLE".to_string(),
        row_number: 42,
        content_files: ContentTriple {
            content_a: PathBuf::from(".raw_data_202509/TEST_TABLE_42_Content.txt"),
            content_b: PathBuf::from(".raw_data_202509/TEST_TABLE_42_Content_L1.txt"),
            content_c: PathBuf::from(".raw_data_202509/TEST_TABLE_42_Content_L2.txt"),
            row_number: 42,
            table_name: "TEST_TABLE".to_string(),
        },
        prompt_file: PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
        output_file: PathBuf::from("gringotts/WorkArea/TEST_TABLE_42.md"),
        analysis_stages: vec![AnalysisStage::AnalyzeA],
    };
    
    let task_markdown = generator.create_analysis_task(&task, "TEST_TABLE");
    
    // Verify gringotts/WorkArea output path is correctly generated
    assert!(task_markdown.contains("**Output**: `gringotts/WorkArea/TEST_TABLE_42.md`"));
}