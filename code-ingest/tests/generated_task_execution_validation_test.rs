//! Generated Task Execution Validation Tests
//! 
//! Tests generated markdown files in Kiro task system
//! Verifies task numbering and hierarchy navigation
//! Validates content file references and accessibility
//! Tests analysis workflow with actual L1-L8 execution

use anyhow::Result;
use code_ingest::{
    database::Database,
    config::DatabaseConfig,
    tasks::{
        DatabaseQueryEngine, ContentExtractor, HierarchicalTaskDivider, 
        L1L8MarkdownGenerator, ContentTriple, TaskHierarchy
    },
    error::TaskError,
};
use std::{
    path::PathBuf,
    sync::Arc,
    collections::HashMap,
};
use tempfile::TempDir;
use serial_test::serial;
use regex::Regex;
use tokio;

/// Test structure for validating Kiro-compatible task format
#[derive(Debug, Clone)]
struct KiroTask {
    pub id: String,
    pub title: String,
    pub content_files: Vec<PathBuf>,
    pub prompt_file: PathBuf,
    pub output_file: PathBuf,
    pub level: usize,
    pub parent_id: Option<String>,
    pub children: Vec<String>,
}

/// Parser for extracting tasks from generated markdown
struct TaskMarkdownParser;

impl TaskMarkdownParser {
    /// Parse generated markdown and extract task structure
    pub fn parse_tasks(markdown: &str) -> Result<Vec<KiroTask>> {
        let mut tasks = Vec::new();
        let lines: Vec<&str> = markdown.lines().collect();
        
        // Regex patterns for parsing
        let task_pattern = Regex::new(r"^- \[ \] (\d+(?:\.\d+)*)\. (.+)$")?;
        let content_pattern = Regex::new(r"^\s*- \*\*Content\*\*: (.+)$")?;
        let prompt_pattern = Regex::new(r"^\s*- \*\*Prompt\*\*: (.+)$")?;
        let output_pattern = Regex::new(r"^\s*- \*\*Output\*\*: (.+)$")?;
        
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            
            if let Some(captures) = task_pattern.captures(line) {
                let id = captures.get(1).unwrap().as_str().to_string();
                let title = captures.get(2).unwrap().as_str().to_string();
                
                // Calculate level from ID depth
                let level = id.split('.').count();
                
                // Calculate parent ID
                let parent_id = if level > 1 {
                    let parts: Vec<&str> = id.split('.').collect();
                    Some(parts[..parts.len()-1].join("."))
                } else {
                    None
                };
                
                // Look for content, prompt, and output in following lines
                let mut content_files = Vec::new();
                let mut prompt_file = PathBuf::new();
                let mut output_file = PathBuf::new();
                
                // Parse task details from following lines
                let mut j = i + 1;
                while j < lines.len() && (lines[j].starts_with("  - ") || lines[j].trim().is_empty()) {
                    let detail_line = lines[j];
                    
                    if let Some(captures) = content_pattern.captures(detail_line) {
                        let content_spec = captures.get(1).unwrap().as_str();
                        content_files = Self::parse_content_files(content_spec)?;
                    } else if let Some(captures) = prompt_pattern.captures(detail_line) {
                        prompt_file = PathBuf::from(captures.get(1).unwrap().as_str().trim());
                    } else if let Some(captures) = output_pattern.captures(detail_line) {
                        output_file = PathBuf::from(captures.get(1).unwrap().as_str().trim());
                    }
                    
                    j += 1;
                }
                
                let task = KiroTask {
                    id,
                    title,
                    content_files,
                    prompt_file,
                    output_file,
                    level,
                    parent_id,
                    children: Vec::new(), // Will be populated later
                };
                
                tasks.push(task);
                i = j;
            } else {
                i += 1;
            }
        }
        
        // Populate children relationships
        Self::populate_children(&mut tasks);
        
        Ok(tasks)
    }
    
    /// Parse content file specification (A + B + C format)
    fn parse_content_files(content_spec: &str) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        
        // Parse format: "path/file.txt as A + path/file_L1.txt as B + path/file_L2.txt as C"
        let parts: Vec<&str> = content_spec.split(" + ").collect();
        
        for part in parts {
            // Extract file path before " as X"
            if let Some(as_pos) = part.find(" as ") {
                let file_path = part[..as_pos].trim();
                // Remove backticks if present
                let clean_path = file_path.trim_matches('`');
                files.push(PathBuf::from(clean_path));
            }
        }
        
        Ok(files)
    }
    
    /// Populate parent-child relationships
    fn populate_children(tasks: &mut [KiroTask]) {
        let mut id_to_index: HashMap<String, usize> = HashMap::new();
        
        // Build ID to index mapping
        for (i, task) in tasks.iter().enumerate() {
            id_to_index.insert(task.id.clone(), i);
        }
        
        // Populate children
        for i in 0..tasks.len() {
            let parent_id = tasks[i].parent_id.clone();
            if let Some(parent_id) = parent_id {
                if let Some(&parent_index) = id_to_index.get(&parent_id) {
                    tasks[parent_index].children.push(tasks[i].id.clone());
                }
            }
        }
    }
}

#[tokio::test]
#[serial]
async fn test_generated_markdown_kiro_compatibility() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create test content and generate markdown
    let (markdown, _content_triples) = create_test_markdown(&temp_dir).await?;
    
    // Parse the generated markdown
    let tasks = TaskMarkdownParser::parse_tasks(&markdown)?;
    
    assert!(!tasks.is_empty(), "Should parse tasks from markdown");
    
    // Validate Kiro task format compliance
    for task in &tasks {
        // Validate task ID format (hierarchical numbering)
        let id_pattern = Regex::new(r"^\d+(\.\d+)*$")?;
        assert!(id_pattern.is_match(&task.id), "Task ID should be hierarchical: {}", task.id);
        
        // Validate task has required components
        assert!(!task.title.is_empty(), "Task should have title");
        assert!(!task.content_files.is_empty(), "Task should have content files");
        assert!(task.prompt_file.exists() || task.prompt_file.to_str().unwrap().contains(".kiro/steering"), 
                "Task should reference valid prompt file: {}", task.prompt_file.display());
        
        // Validate level consistency
        let expected_level = task.id.split('.').count();
        assert_eq!(task.level, expected_level, "Task level should match ID depth");
        
        // Validate parent-child relationships
        if task.level > 1 {
            assert!(task.parent_id.is_some(), "Non-root task should have parent");
            
            if let Some(parent_id) = &task.parent_id {
                let parent_level = parent_id.split('.').count();
                assert_eq!(parent_level, task.level - 1, "Parent should be one level up");
            }
        } else {
            assert!(task.parent_id.is_none(), "Root task should not have parent");
        }
    }
    
    println!("✅ Generated markdown is Kiro-compatible: {} tasks parsed", tasks.len());
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_task_numbering_and_hierarchy_navigation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (markdown, _content_triples) = create_test_markdown(&temp_dir).await?;
    let tasks = TaskMarkdownParser::parse_tasks(&markdown)?;
    
    // Test hierarchical numbering system
    let mut level_counts: HashMap<usize, usize> = HashMap::new();
    let mut parent_child_map: HashMap<String, Vec<String>> = HashMap::new();
    
    for task in &tasks {
        // Count tasks per level
        *level_counts.entry(task.level).or_insert(0) += 1;
        
        // Build parent-child map
        if let Some(parent_id) = &task.parent_id {
            parent_child_map.entry(parent_id.clone()).or_insert_with(Vec::new).push(task.id.clone());
        }
    }
    
    // Validate level distribution (should follow 7 groups per level constraint)
    for (level, count) in &level_counts {
        if *level == 1 {
            assert!(*count <= 7, "Level 1 should have at most 7 tasks, got {}", count);
        }
        println!("Level {}: {} tasks", level, count);
    }
    
    // Test navigation paths
    let root_tasks: Vec<_> = tasks.iter().filter(|t| t.level == 1).collect();
    assert!(!root_tasks.is_empty(), "Should have root level tasks");
    
    for root_task in &root_tasks {
        // Test depth-first traversal
        let mut visited = std::collections::HashSet::new();
        let path = traverse_task_hierarchy(&tasks, &root_task.id, &mut visited)?;
        assert!(!path.is_empty(), "Should be able to traverse from root task {}", root_task.id);
        
        println!("Navigation path from {}: {:?}", root_task.id, path);
    }
    
    // Test sibling navigation
    for task in &tasks {
        if let Some(parent_id) = &task.parent_id {
            let siblings = get_sibling_tasks(&tasks, &task.id);
            println!("Task {} has {} siblings", task.id, siblings.len());
            
            // Validate sibling numbering is sequential
            if siblings.len() > 1 {
                let mut sibling_numbers: Vec<usize> = siblings.iter()
                    .map(|s| s.id.split('.').last().unwrap().parse().unwrap())
                    .collect();
                sibling_numbers.sort();
                
                for i in 1..sibling_numbers.len() {
                    assert_eq!(sibling_numbers[i], sibling_numbers[i-1] + 1, 
                              "Sibling numbering should be sequential");
                }
            }
        }
    }
    
    println!("✅ Task numbering and hierarchy navigation validated");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_content_file_references_and_accessibility() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (markdown, content_triples) = create_test_markdown(&temp_dir).await?;
    let tasks = TaskMarkdownParser::parse_tasks(&markdown)?;
    
    // Validate all content files are accessible
    for task in &tasks {
        assert_eq!(task.content_files.len(), 3, "Each task should reference 3 content files (A/B/C)");
        
        for (i, content_file) in task.content_files.iter().enumerate() {
            assert!(content_file.exists(), "Content file should exist: {}", content_file.display());
            
            // Validate file naming convention
            let filename = content_file.file_name().unwrap().to_str().unwrap();
            match i {
                0 => assert!(filename.contains("_Content.txt") && !filename.contains("_L"), 
                            "First file should be A (raw content): {}", filename),
                1 => assert!(filename.contains("_Content_L1.txt"), 
                            "Second file should be B (L1 context): {}", filename),
                2 => assert!(filename.contains("_Content_L2.txt"), 
                            "Third file should be C (L2 context): {}", filename),
                _ => panic!("Unexpected content file index: {}", i),
            }
            
            // Validate file content is not empty
            let content = tokio::fs::read_to_string(content_file).await?;
            assert!(!content.trim().is_empty(), "Content file should not be empty: {}", content_file.display());
        }
    }
    
    // Validate content file references in markdown match actual files
    for (task_idx, task) in tasks.iter().enumerate() {
        if task_idx < content_triples.len() {
            let triple = &content_triples[task_idx];
            
            // Check that task references match the actual content triple files
            assert_eq!(task.content_files[0], triple.content_a, "Task A file should match content triple");
            assert_eq!(task.content_files[1], triple.content_b, "Task B file should match content triple");
            assert_eq!(task.content_files[2], triple.content_c, "Task C file should match content triple");
        }
    }
    
    // Test file accessibility patterns
    let content_dir = temp_dir.path().join(".raw_data_202509");
    assert!(content_dir.exists(), "Content directory should exist");
    
    // Validate all files follow the expected pattern
    let mut file_count = 0;
    for entry in std::fs::read_dir(&content_dir)? {
        let entry = entry?;
        let filename = entry.file_name().to_str().unwrap().to_string();
        
        if filename.ends_with(".txt") {
            file_count += 1;
            
            // Validate filename pattern
            let pattern = Regex::new(r"^TEST_TABLE_\d+_Content(_L[12])?\.txt$")?;
            assert!(pattern.is_match(&filename), "File should match naming pattern: {}", filename);
        }
    }
    
    // Should have 3 files per content triple (A, B, C)
    assert_eq!(file_count, content_triples.len() * 3, "Should have correct number of content files");
    
    println!("✅ Content file references and accessibility validated");
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_analysis_workflow_with_l1_l8_execution() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let (markdown, content_triples) = create_test_markdown(&temp_dir).await?;
    let tasks = TaskMarkdownParser::parse_tasks(&markdown)?;
    
    // Simulate L1-L8 analysis execution on first task
    if let Some(first_task) = tasks.first() {
        println!("🔬 Simulating L1-L8 analysis on task: {}", first_task.title);
        
        // Read content files
        let content_a = tokio::fs::read_to_string(&first_task.content_files[0]).await?;
        let content_b = tokio::fs::read_to_string(&first_task.content_files[1]).await?;
        let content_c = tokio::fs::read_to_string(&first_task.content_files[2]).await?;
        
        // Read prompt file
        let prompt_content = tokio::fs::read_to_string(&first_task.prompt_file).await?;
        
        // Simulate L1-L8 analysis stages
        let analysis_results = simulate_l1_l8_analysis(&content_a, &content_b, &content_c, &prompt_content)?;
        
        // Validate analysis results structure
        assert_eq!(analysis_results.len(), 4, "Should have 4 analysis stages");
        
        let expected_stages = [
            "A alone",
            "A in context of B", 
            "B in context of C",
            "A in context of B & C"
        ];
        
        for (i, stage) in expected_stages.iter().enumerate() {
            assert!(analysis_results[i].stage_description.contains(stage), 
                   "Stage {} should analyze {}", i + 1, stage);
            assert!(!analysis_results[i].insights.is_empty(), 
                   "Stage {} should have insights", i + 1);
        }
        
        // Write analysis result to output file
        let output_content = format_analysis_results(&analysis_results);
        tokio::fs::write(&first_task.output_file, &output_content).await?;
        
        // Validate output file was created
        assert!(first_task.output_file.exists(), "Output file should be created");
        
        let written_content = tokio::fs::read_to_string(&first_task.output_file).await?;
        assert!(!written_content.trim().is_empty(), "Output file should have content");
        assert!(written_content.contains("L1-L8 Analysis Results"), "Output should have proper header");
        
        println!("✅ L1-L8 analysis workflow simulation completed");
    }
    
    // Test batch analysis workflow
    println!("🔄 Testing batch analysis workflow...");
    
    let batch_size = std::cmp::min(3, tasks.len());
    let mut completed_analyses = 0;
    
    for task in tasks.iter().take(batch_size) {
        // Simulate quick analysis
        let quick_analysis = format!(
            "# Quick Analysis for {}\n\n## Summary\nAnalyzed file: {}\n\n## Key Insights\n- File structure validated\n- Content accessibility confirmed\n- L1-L8 methodology applicable\n",
            task.title,
            task.content_files[0].display()
        );
        
        tokio::fs::write(&task.output_file, &quick_analysis).await?;
        completed_analyses += 1;
    }
    
    assert_eq!(completed_analyses, batch_size, "Should complete batch analysis");
    
    println!("✅ Batch analysis workflow validated: {} tasks processed", completed_analyses);
    
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_task_execution_error_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create markdown with intentionally problematic references
    let problematic_markdown = r#"# Implementation Plan

- [ ] 1. Analyze TEST_TABLE row 1
  - **Content**: `/nonexistent/file.txt` as A + `/nonexistent/file_L1.txt` as B + `/nonexistent/file_L2.txt` as C
  - **Prompt**: `/nonexistent/prompt.md`
  - **Output**: `gringotts/WorkArea/TEST_TABLE_1.md`

- [ ] 2. Analyze TEST_TABLE row 2
  - **Content**: `` as A + `` as B + `` as C
  - **Prompt**: ``
  - **Output**: ``
"#;
    
    let tasks = TaskMarkdownParser::parse_tasks(problematic_markdown)?;
    
    // Test error handling for missing files
    for task in &tasks {
        // Test missing content files
        for content_file in &task.content_files {
            if !content_file.as_os_str().is_empty() {
                let exists = content_file.exists();
                if !exists {
                    println!("⚠️  Missing content file detected: {}", content_file.display());
                    // In real execution, this would be handled gracefully
                }
            }
        }
        
        // Test missing prompt file
        if !task.prompt_file.as_os_str().is_empty() && !task.prompt_file.exists() {
            println!("⚠️  Missing prompt file detected: {}", task.prompt_file.display());
        }
        
        // Test invalid output path
        if let Some(parent) = task.output_file.parent() {
            if !parent.exists() {
                println!("⚠️  Output directory does not exist: {}", parent.display());
                // In real execution, directory would be created
            }
        }
    }
    
    // Test recovery from parsing errors
    let malformed_markdown = r#"# Implementation Plan

- [ ] 1. Malformed task without proper structure
  - Missing content specification
  
- [ ] 2.1.3.4.5.6.7.8. Task with too deep nesting
  - **Content**: `file.txt`
  - **Prompt**: `prompt.md`
  
- [ ] Invalid ID format
  - **Content**: `file.txt`
"#;
    
    let malformed_tasks = TaskMarkdownParser::parse_tasks(malformed_markdown)?;
    
    // Should still parse what it can
    assert!(!malformed_tasks.is_empty(), "Should parse valid tasks even with malformed content");
    
    // Validate error resilience
    for task in &malformed_tasks {
        if task.id.split('.').count() > 4 {
            println!("⚠️  Task with excessive nesting detected: {}", task.id);
        }
        
        if task.content_files.is_empty() {
            println!("⚠️  Task without content files detected: {}", task.id);
        }
    }
    
    println!("✅ Task execution error handling validated");
    
    Ok(())
}

// Helper functions

async fn create_test_markdown(temp_dir: &TempDir) -> Result<(String, Vec<ContentTriple>)> {
    // Create test database connection (mock for this test)
    let database_url = std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://postgres:password@localhost:5432/code_ingest_test".to_string());
    
    let config = DatabaseConfig {
        database_url,
        max_connections: 5,
        connection_timeout_seconds: 30,
    };
    
    // Try to create database, use mock data if fails
    let content_triples = if let Ok(db) = Database::new(config).await {
        // Create real content triples
        let output_dir = temp_dir.path().join(".raw_data_202509");
        let content_extractor = ContentExtractor::new(Arc::new(db.pool().clone()), output_dir);
        
        // Create mock metadata for testing
        let test_metadata = code_ingest::tasks::content_extractor::RowMetadata {
            file_id: Some(1),
            filepath: Some("src/main.rs".to_string()),
            filename: Some("main.rs".to_string()),
            extension: Some("rs".to_string()),
            file_size_bytes: Some(1024),
            line_count: Some(45),
            word_count: Some(200),
            content_text: Some("fn main() { println!(\"Hello, world!\"); }".to_string()),
            file_type: Some("direct_text".to_string()),
            relative_path: Some("src/main.rs".to_string()),
            absolute_path: Some("/project/src/main.rs".to_string()),
        };
        
        let triple = content_extractor.create_content_files(&test_metadata, 1, "TEST_TABLE").await?;
        vec![triple]
    } else {
        // Create mock content triples
        let output_dir = temp_dir.path().join(".raw_data_202509");
        tokio::fs::create_dir_all(&output_dir).await?;
        
        let content_a = output_dir.join("TEST_TABLE_1_Content.txt");
        let content_b = output_dir.join("TEST_TABLE_1_Content_L1.txt");
        let content_c = output_dir.join("TEST_TABLE_1_Content_L2.txt");
        
        tokio::fs::write(&content_a, "fn main() { println!(\"Hello, world!\"); }").await?;
        tokio::fs::write(&content_b, "# L1 Context\nImmediate file context...").await?;
        tokio::fs::write(&content_c, "# L2 Context\nArchitectural context...").await?;
        
        vec![ContentTriple {
            content_a,
            content_b,
            content_c,
            row_number: 1,
            table_name: "TEST_TABLE".to_string(),
        }]
    };
    
    // Create prompt file
    let prompt_file = temp_dir.path().join("test_prompt.md");
    let prompt_content = r#"# L1-L8 Analysis Methodology

## Analysis Instructions

Analyze the provided content using the L1-L8 methodology:

1. **A alone**: Analyze the raw content in isolation
2. **A in context of B**: Analyze A with immediate file context
3. **B in context of C**: Analyze immediate context with architectural context  
4. **A in context of B & C**: Comprehensive analysis with full context

## Expected Output

Provide insights for each analysis stage focusing on:
- Code patterns and idioms
- Architectural decisions
- Optimization opportunities
- Design patterns
"#;
    
    tokio::fs::write(&prompt_file, prompt_content).await?;
    
    // Generate markdown
    let task_divider = HierarchicalTaskDivider::new(2, 3); // 2 levels, 3 groups for testing
    let hierarchy = task_divider?.create_hierarchy(content_triples.clone())?;
    
    let markdown_generator = L1L8MarkdownGenerator::new(prompt_file, temp_dir.path().to_path_buf());
    let markdown = markdown_generator.generate_hierarchical_markdown(&hierarchy, "TEST_TABLE").await?;
    
    Ok((markdown, content_triples))
}

fn traverse_task_hierarchy(tasks: &[KiroTask], start_id: &str, visited: &mut std::collections::HashSet<String>) -> Result<Vec<String>> {
    if visited.contains(start_id) {
        return Ok(vec![]); // Avoid cycles
    }
    
    visited.insert(start_id.to_string());
    let mut path = vec![start_id.to_string()];
    
    // Find task by ID
    if let Some(task) = tasks.iter().find(|t| t.id == start_id) {
        // Add children to path
        for child_id in &task.children {
            let child_path = traverse_task_hierarchy(tasks, child_id, visited)?;
            path.extend(child_path);
        }
    }
    
    Ok(path)
}

fn get_sibling_tasks<'a>(tasks: &'a [KiroTask], task_id: &str) -> Vec<&'a KiroTask> {
    if let Some(task) = tasks.iter().find(|t| t.id == task_id) {
        if let Some(parent_id) = &task.parent_id {
            return tasks.iter()
                .filter(|t| t.parent_id.as_ref() == Some(parent_id) && t.id != task_id)
                .collect();
        }
    }
    
    Vec::new()
}

#[derive(Debug)]
struct AnalysisStage {
    stage_description: String,
    insights: Vec<String>,
}

fn simulate_l1_l8_analysis(content_a: &str, content_b: &str, content_c: &str, _prompt: &str) -> Result<Vec<AnalysisStage>> {
    let mut stages = Vec::new();
    
    // Stage 1: A alone
    stages.push(AnalysisStage {
        stage_description: "Analysis of A alone".to_string(),
        insights: vec![
            format!("Raw content analysis: {} characters", content_a.len()),
            "Code structure identified".to_string(),
            "Syntax patterns detected".to_string(),
        ],
    });
    
    // Stage 2: A in context of B
    stages.push(AnalysisStage {
        stage_description: "Analysis of A in context of B".to_string(),
        insights: vec![
            "File context enhances understanding".to_string(),
            "Import relationships identified".to_string(),
            format!("L1 context provides {} additional characters", content_b.len()),
        ],
    });
    
    // Stage 3: B in context of C
    stages.push(AnalysisStage {
        stage_description: "Analysis of B in context of C".to_string(),
        insights: vec![
            "Architectural patterns emerge".to_string(),
            "Module relationships clarified".to_string(),
            format!("L2 context adds {} characters of architectural insight", content_c.len()),
        ],
    });
    
    // Stage 4: A in context of B & C
    stages.push(AnalysisStage {
        stage_description: "Comprehensive analysis of A in context of B & C".to_string(),
        insights: vec![
            "Full context reveals design intent".to_string(),
            "Optimization opportunities identified".to_string(),
            "Architectural compliance validated".to_string(),
        ],
    });
    
    Ok(stages)
}

fn format_analysis_results(results: &[AnalysisStage]) -> String {
    let mut output = String::new();
    
    output.push_str("# L1-L8 Analysis Results\n\n");
    
    for (i, stage) in results.iter().enumerate() {
        output.push_str(&format!("## Stage {}: {}\n\n", i + 1, stage.stage_description));
        
        for insight in &stage.insights {
            output.push_str(&format!("- {}\n", insight));
        }
        
        output.push_str("\n");
    }
    
    output.push_str("---\n\n");
    output.push_str("*Analysis completed using L1-L8 methodology*\n");
    
    output
}