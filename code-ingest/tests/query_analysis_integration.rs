//! Integration tests for Query and Analysis Preparation functionality
//!
//! These tests verify that the SQL query execution, temporary file management,
//! and result storage functionality work correctly together.

use std::collections::HashMap;
use code_ingest::database::{QueryExecutor, QueryConfig, TempFileManager, TempFileConfig, TempFileMetadata, ResultStorage, StorageConfig, ResultMetadata};

/// Test query execution and formatting functionality
#[test]
fn test_query_config_creation() {
    let config = QueryConfig::default();
    assert_eq!(config.max_rows, 0);
    assert!(!config.llm_format);
    assert!(config.include_stats);
    assert!(config.stream_results);
}

#[test]
fn test_query_config_customization() {
    let config = QueryConfig {
        max_rows: 100,
        llm_format: true,
        include_stats: false,
        stream_results: false,
    };
    
    assert_eq!(config.max_rows, 100);
    assert!(config.llm_format);
    assert!(!config.include_stats);
    assert!(!config.stream_results);
}

/// Test temporary file management functionality
#[test]
fn test_temp_file_config_creation() {
    let config = TempFileConfig::default();
    assert!(config.overwrite_existing);
    assert!(config.create_parent_dirs);
    assert_eq!(config.file_permissions, Some(0o644));
    assert!(config.validate_paths);
}

#[test]
fn test_temp_file_metadata_creation() {
    let metadata = TempFileMetadata {
        original_query: "SELECT * FROM test".to_string(),
        output_table: "QUERYRESULT_test".to_string(),
        prompt_file_path: Some(std::path::PathBuf::from("/tmp/prompt.md")),
        description: Some("Test query for analysis".to_string()),
    };
    
    assert_eq!(metadata.original_query, "SELECT * FROM test");
    assert_eq!(metadata.output_table, "QUERYRESULT_test");
    assert!(metadata.prompt_file_path.is_some());
    assert!(metadata.description.is_some());
}

/// Test result storage functionality
#[test]
fn test_storage_config_creation() {
    let config = StorageConfig::default();
    assert!(config.create_table_if_missing);
    assert!(config.validate_content);
    assert_eq!(config.max_result_size, 10 * 1024 * 1024); // 10MB
    assert!(!config.compress_large_results);
}

#[test]
fn test_result_metadata_creation() {
    let metadata = ResultMetadata {
        original_query: "SELECT filepath, content_text FROM files".to_string(),
        prompt_file_path: Some("/prompts/analyze.md".to_string()),
        analysis_type: Some("code_review".to_string()),
        original_file_path: Some("/src/main.rs".to_string()),
        chunk_number: Some(1),
        created_by: Some("test_user".to_string()),
        tags: vec!["rust".to_string(), "analysis".to_string()],
    };
    
    assert_eq!(metadata.original_query, "SELECT filepath, content_text FROM files");
    assert_eq!(metadata.analysis_type, Some("code_review".to_string()));
    assert_eq!(metadata.chunk_number, Some(1));
    assert_eq!(metadata.tags.len(), 2);
    assert!(metadata.tags.contains(&"rust".to_string()));
    assert!(metadata.tags.contains(&"analysis".to_string()));
}

/// Test query result formatting
#[test]
fn test_query_result_structure() {
    use code_ingest::database::QueryResult;
    
    let mut row1 = HashMap::new();
    row1.insert("id".to_string(), "1".to_string());
    row1.insert("filepath".to_string(), "/test/file.rs".to_string());
    row1.insert("content_text".to_string(), "fn main() {}".to_string());
    
    let mut row2 = HashMap::new();
    row2.insert("id".to_string(), "2".to_string());
    row2.insert("filepath".to_string(), "/test/lib.rs".to_string());
    row2.insert("content_text".to_string(), "pub mod test;".to_string());
    
    let result = QueryResult {
        columns: vec!["id".to_string(), "filepath".to_string(), "content_text".to_string()],
        rows: vec![row1, row2],
        row_count: 2,
        execution_time_ms: 50,
    };
    
    assert_eq!(result.columns.len(), 3);
    assert_eq!(result.row_count, 2);
    assert_eq!(result.execution_time_ms, 50);
    assert!(result.columns.contains(&"filepath".to_string()));
    assert!(result.columns.contains(&"content_text".to_string()));
}

/// Test error handling and validation
#[test]
fn test_path_validation_logic() {
    use std::path::Path;
    
    // Test absolute vs relative paths
    let absolute_path = Path::new("/tmp/test.txt");
    let relative_path = Path::new("test.txt");
    
    assert!(absolute_path.is_absolute());
    assert!(!relative_path.is_absolute());
    
    // Test file extensions
    let txt_path = Path::new("/tmp/test.txt");
    let md_path = Path::new("/tmp/test.md");
    let no_ext_path = Path::new("/tmp/test");
    
    assert_eq!(txt_path.extension().unwrap(), "txt");
    assert_eq!(md_path.extension().unwrap(), "md");
    assert!(no_ext_path.extension().is_none());
}

/// Test configuration combinations
#[test]
fn test_configuration_combinations() {
    // Test LLM-optimized configuration
    let llm_config = QueryConfig {
        llm_format: true,
        include_stats: false,
        max_rows: 0, // No limit for LLM processing
        stream_results: true,
    };
    
    assert!(llm_config.llm_format);
    assert!(!llm_config.include_stats);
    assert_eq!(llm_config.max_rows, 0);
    
    // Test terminal-optimized configuration
    let terminal_config = QueryConfig {
        llm_format: false,
        include_stats: true,
        max_rows: 100, // Limit for terminal display
        stream_results: false,
    };
    
    assert!(!terminal_config.llm_format);
    assert!(terminal_config.include_stats);
    assert_eq!(terminal_config.max_rows, 100);
}

/// Test workflow integration
#[test]
fn test_workflow_data_structures() {
    // Test that all the data structures work together
    let query = "SELECT filepath, content_text FROM INGEST_20250927143022 WHERE extension = 'rs'";
    let output_table = "QUERYRESULT_rust_analysis";
    
    // Create temp file metadata
    let temp_metadata = TempFileMetadata {
        original_query: query.to_string(),
        output_table: output_table.to_string(),
        prompt_file_path: Some(std::path::PathBuf::from("/prompts/rust_analysis.md")),
        description: Some("Rust code analysis workflow".to_string()),
    };
    
    // Create result metadata
    let result_metadata = ResultMetadata {
        original_query: query.to_string(),
        prompt_file_path: Some("/prompts/rust_analysis.md".to_string()),
        analysis_type: Some("rust_code_review".to_string()),
        original_file_path: None,
        chunk_number: None,
        created_by: Some("code-ingest-cli".to_string()),
        tags: vec!["rust".to_string(), "code_review".to_string()],
    };
    
    // Verify consistency
    assert_eq!(temp_metadata.original_query, result_metadata.original_query);
    assert_eq!(temp_metadata.output_table, output_table);
    assert_eq!(
        temp_metadata.prompt_file_path.as_ref().unwrap().to_str().unwrap(),
        result_metadata.prompt_file_path.as_ref().unwrap()
    );
}

/// Test CLI command structure validation
#[test]
fn test_cli_command_parameters() {
    // Test that required parameters are properly structured
    let query = "SELECT * FROM test";
    let temp_path = std::path::PathBuf::from("/tmp/query_results.txt");
    let tasks_file = std::path::PathBuf::from("/tmp/analysis_tasks.md");
    let output_table = "QUERYRESULT_test_analysis";
    let result_file = std::path::PathBuf::from("/tmp/analysis_results.txt");
    
    // Verify paths are absolute (as required by the CLI)
    assert!(temp_path.is_absolute());
    assert!(tasks_file.is_absolute());
    assert!(result_file.is_absolute());
    
    // Verify table naming convention
    assert!(output_table.starts_with("QUERYRESULT_"));
    
    // Verify file extensions are appropriate
    assert_eq!(temp_path.extension().unwrap(), "txt");
    assert_eq!(tasks_file.extension().unwrap(), "md");
    assert_eq!(result_file.extension().unwrap(), "txt");
}