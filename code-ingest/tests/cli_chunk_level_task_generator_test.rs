//! Integration tests for the chunk-level-task-generator CLI command
//!
//! This module tests the CLI integration of the ChunkLevelTaskGenerator command,
//! verifying that command-line arguments are parsed correctly and the command
//! executes without errors.

use std::path::PathBuf;
use clap::Parser;

// Import the CLI structure
use code_ingest::cli::{Cli, Commands};

#[test]
fn test_chunk_level_task_generator_cli_parsing_file_level_mode() {
    // Test file-level mode (no chunk size)
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "INGEST_20250928101039",
        "--db-path",
        "/tmp/test.db",
        "--output-dir",
        "./output",
    ];
    
    let cli = Cli::try_parse_from(args).unwrap();
    
    match &cli.command {
        Some(Commands::ChunkLevelTaskGenerator { 
            table_name, 
            chunk_size, 
            db_path, 
            output_dir 
        }) => {
            assert_eq!(table_name, "INGEST_20250928101039");
            assert_eq!(*chunk_size, None);
            assert_eq!(db_path.as_ref().unwrap(), &PathBuf::from("/tmp/test.db"));
            assert_eq!(output_dir, &PathBuf::from("./output"));
        }
        _ => panic!("Expected ChunkLevelTaskGenerator command, got: {:?}", cli.command),
    }
}

#[test]
fn test_chunk_level_task_generator_cli_parsing_chunk_level_mode() {
    // Test chunk-level mode (with chunk size)
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "INGEST_20250928101039",
        "500",
        "--db-path",
        "/tmp/test.db",
    ];
    
    let cli = Cli::try_parse_from(args).unwrap();
    
    match &cli.command {
        Some(Commands::ChunkLevelTaskGenerator { 
            table_name, 
            chunk_size, 
            db_path, 
            output_dir 
        }) => {
            assert_eq!(table_name, "INGEST_20250928101039");
            assert_eq!(*chunk_size, Some(500));
            assert_eq!(db_path.as_ref().unwrap(), &PathBuf::from("/tmp/test.db"));
            assert_eq!(output_dir, &PathBuf::from(".")); // Default value
        }
        _ => panic!("Expected ChunkLevelTaskGenerator command, got: {:?}", cli.command),
    }
}

#[test]
fn test_chunk_level_task_generator_cli_parsing_minimal_args() {
    // Test with minimal arguments (only table name)
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "INGEST_20250928101039",
    ];
    
    let cli = Cli::try_parse_from(args).unwrap();
    
    match &cli.command {
        Some(Commands::ChunkLevelTaskGenerator { 
            table_name, 
            chunk_size, 
            db_path, 
            output_dir 
        }) => {
            assert_eq!(table_name, "INGEST_20250928101039");
            assert_eq!(*chunk_size, None);
            assert_eq!(*db_path, None);
            assert_eq!(output_dir, &PathBuf::from(".")); // Default value
        }
        _ => panic!("Expected ChunkLevelTaskGenerator command, got: {:?}", cli.command),
    }
}

#[test]
fn test_chunk_level_task_generator_cli_parsing_all_args() {
    // Test with all possible arguments
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "INGEST_20250928101039",
        "1000",
        "--db-path",
        "/path/to/database",
        "--output-dir",
        "/path/to/output",
    ];
    
    let cli = Cli::try_parse_from(args).unwrap();
    
    match &cli.command {
        Some(Commands::ChunkLevelTaskGenerator { 
            table_name, 
            chunk_size, 
            db_path, 
            output_dir 
        }) => {
            assert_eq!(table_name, "INGEST_20250928101039");
            assert_eq!(*chunk_size, Some(1000));
            assert_eq!(db_path.as_ref().unwrap(), &PathBuf::from("/path/to/database"));
            assert_eq!(output_dir, &PathBuf::from("/path/to/output"));
        }
        _ => panic!("Expected ChunkLevelTaskGenerator command, got: {:?}", cli.command),
    }
}

#[test]
fn test_chunk_level_task_generator_cli_parsing_invalid_chunk_size() {
    // Test with invalid chunk size (non-numeric)
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "INGEST_20250928101039",
        "invalid",
        "--db-path",
        "/tmp/test.db",
    ];
    
    let result = Cli::try_parse_from(args);
    assert!(result.is_err(), "Expected parsing to fail with invalid chunk size");
}

#[test]
fn test_chunk_level_task_generator_cli_parsing_missing_table_name() {
    // Test with missing table name (required argument)
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "--db-path",
        "/tmp/test.db",
    ];
    
    let result = Cli::try_parse_from(args);
    assert!(result.is_err(), "Expected parsing to fail with missing table name");
}

#[test]
fn test_chunk_level_task_generator_help_text() {
    // Test that help text is available for the command
    let args = vec![
        "code-ingest",
        "chunk-level-task-generator",
        "--help",
    ];
    
    let result = Cli::try_parse_from(args);
    // This should fail with a help message, not a parsing error
    assert!(result.is_err());
    
    // The error should contain help information
    let error_message = format!("{}", result.unwrap_err());
    assert!(error_message.contains("chunk-level-task-generator"));
    assert!(error_message.contains("table_name"));
    assert!(error_message.contains("chunk_size"));
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    /// Test that demonstrates the expected command usage patterns
    #[test]
    fn test_command_usage_examples() {
        // Example 1: File-level mode
        let file_level_args = vec![
            "code-ingest",
            "chunk-level-task-generator", 
            "INGEST_20250928101039",
            "--db-path",
            "./analysis.db",
        ];
        
        let cli = Cli::try_parse_from(file_level_args).unwrap();
        assert!(matches!(cli.command, Some(Commands::ChunkLevelTaskGenerator { .. })));
        
        // Example 2: Chunk-level mode
        let chunk_level_args = vec![
            "code-ingest",
            "chunk-level-task-generator",
            "INGEST_20250928101039", 
            "500",
            "--db-path",
            "./analysis.db",
            "--output-dir",
            "./task-output",
        ];
        
        let cli = Cli::try_parse_from(chunk_level_args).unwrap();
        assert!(matches!(cli.command, Some(Commands::ChunkLevelTaskGenerator { .. })));
    }
}