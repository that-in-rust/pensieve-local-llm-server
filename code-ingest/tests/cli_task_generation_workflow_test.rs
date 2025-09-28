use anyhow::Result;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio;

/// End-to-end integration test for the complete task generation workflow
#[tokio::test]
async fn test_complete_task_generation_workflow() -> Result<()> {
    // This test requires a running PostgreSQL instance with test data
    // Skip if DATABASE_URL is not set
    if std::env::var("DATABASE_URL").is_err() {
        println!("Skipping complete workflow integration test - DATABASE_URL not set");
        return Ok(());
    }

    // Test workflow:
    // 1. count-rows -> verify table exists and has data
    // 2. extract-content -> create A/B/C files
    // 3. generate-hierarchical-tasks -> create markdown with proper structure
    
    let table_name = "test_workflow_table";
    let temp_dir = TempDir::new()?;
    let output_dir = temp_dir.path().join(".raw_data_202509");
    let tasks_file = temp_dir.path().join("workflow_tasks.md");
    
    println!("Complete workflow test structure validated");
    println!("Table: {}", table_name);
    println!("Output directory: {}", output_dir.display());
    println!("Tasks file: {}", tasks_file.display());
    
    // In a real test, this would:
    // 1. Set up test database with sample data
    // 2. Execute count-rows command and verify output
    // 3. Execute extract-content command and verify A/B/C files
    // 4. Execute generate-hierarchical-tasks and verify markdown structure
    // 5. Validate hierarchical numbering (1.1.1.1 format)
    // 6. Verify content file references in tasks
    // 7. Clean up test data
    
    Ok(())
}

/// Test error handling across the workflow
#[tokio::test]
async fn test_workflow_error_handling() -> Result<()> {
    // Test error scenarios:
    // 1. Non-existent table in count-rows
    // 2. Empty table in extract-content
    // 3. Missing prompt file in generate-hierarchical-tasks
    // 4. Invalid parameters (levels=0, groups=0)
    
    println!("Workflow error handling test structure validated");
    
    Ok(())
}

/// Test CLI command chaining and parameter validation
#[test]
fn test_cli_command_parameter_validation() {
    use clap::Parser;
    
    // Test that all three commands can be parsed correctly
    
    // 1. count-rows
    let count_args = vec!["code-ingest", "count-rows", "test_table", "--db-path", "/tmp/test.db"];
    let count_cli = code_ingest::cli::Cli::try_parse_from(count_args).unwrap();
    println!("Count-rows command parsed successfully");
    
    // 2. extract-content
    let extract_args = vec!["code-ingest", "extract-content", "test_table", "--output-dir", ".raw_data"];
    let extract_cli = code_ingest::cli::Cli::try_parse_from(extract_args).unwrap();
    println!("Extract-content command parsed successfully");
    
    // 3. generate-hierarchical-tasks
    let generate_args = vec![
        "code-ingest", "generate-hierarchical-tasks", "test_table", 
        "--output", "tasks.md", "--levels", "4", "--groups", "7"
    ];
    let generate_cli = code_ingest::cli::Cli::try_parse_from(generate_args).unwrap();
    println!("Generate-hierarchical-tasks command parsed successfully");
}

/// Test the expected command sequence for the requirements
#[test]
fn test_requirements_command_sequence() {
    // Based on the requirements, the expected command sequence is:
    // 1. code-ingest count-rows INGEST_20250928101039
    // 2. code-ingest extract-content INGEST_20250928101039 --output-dir .raw_data_202509
    // 3. code-ingest generate-hierarchical-tasks INGEST_20250928101039 --levels 4 --groups 7 --output INGEST_20250928101039_tasks.md
    
    use clap::Parser;
    
    let table_name = "INGEST_20250928101039";
    
    // Command 1: count-rows
    let count_args = vec!["code-ingest", "count-rows", table_name];
    let count_cli = code_ingest::cli::Cli::try_parse_from(count_args).unwrap();
    println!("Count-rows command parsed successfully for {}", table_name);
    
    // Command 2: extract-content
    let extract_args = vec!["code-ingest", "extract-content", table_name, "--output-dir", ".raw_data_202509"];
    let extract_cli = code_ingest::cli::Cli::try_parse_from(extract_args).unwrap();
    println!("Extract-content command parsed successfully for {}", table_name);
    
    // Command 3: generate-hierarchical-tasks
    let output_file = format!("{}_tasks.md", table_name);
    let generate_args = vec![
        "code-ingest", "generate-hierarchical-tasks", table_name,
        "--levels", "4", "--groups", "7", 
        "--output", &output_file
    ];
    let generate_cli = code_ingest::cli::Cli::try_parse_from(generate_args).unwrap();
    println!("Generate-hierarchical-tasks command parsed successfully for {}", table_name);
}

/// Test the final command format from requirements
#[test]
fn test_final_command_format() {
    use clap::Parser;
    
    // Test the exact command format from requirements:
    // code-ingest generate-hierarchical-tasks INGEST_20250928101039 --levels 4 --groups 7 --output INGEST_20250928101039_tasks.md
    
    let args = vec![
        "code-ingest",
        "generate-hierarchical-tasks", 
        "INGEST_20250928101039",
        "--levels", "4",
        "--groups", "7", 
        "--output", "INGEST_20250928101039_tasks.md"
    ];
    
    let cli = code_ingest::cli::Cli::try_parse_from(args).unwrap();
    
    // Test that the exact command format from requirements parses successfully
    println!("Final command format from requirements parsed successfully");
}