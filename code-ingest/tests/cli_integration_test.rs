//! Integration tests for CLI commands
//! 
//! These tests verify that the CLI commands work correctly end-to-end,
//! including argument parsing, command execution, and output formatting.

use anyhow::Result;
use std::process::Command;
use tempfile::TempDir;
use std::fs;

/// Test that the CLI binary can be built and executed
#[test]
fn test_cli_binary_help() -> Result<()> {
    let output = Command::new("cargo")
        .args(&["run", "--", "--help"])
        .current_dir(".")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    
    // Verify key CLI elements are present
    assert!(stdout.contains("A Rust-based code ingestion system for PostgreSQL"));
    assert!(stdout.contains("ingest"));
    assert!(stdout.contains("sql"));
    assert!(stdout.contains("pg-start"));
    assert!(stdout.contains("list-tables"));
    
    Ok(())
}

/// Test that the pg-start command executes without errors
#[test]
fn test_pg_start_command() -> Result<()> {
    let output = Command::new("cargo")
        .args(&["run", "--", "pg-start"])
        .current_dir(".")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    
    // Verify pg-start output contains expected sections
    assert!(stdout.contains("PostgreSQL Setup Assistant"));
    assert!(stdout.contains("System Information"));
    assert!(stdout.contains("Step 1: PostgreSQL Installation"));
    assert!(stdout.contains("DATABASE_URL"));
    
    Ok(())
}

/// Test that the ingest command handles missing database gracefully
#[test]
fn test_ingest_command_missing_database() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_folder = temp_dir.path().join("test_repo");
    fs::create_dir(&test_folder)?;
    fs::write(test_folder.join("test.txt"), "Hello, world!")?;
    
    let output = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "ingest", 
            test_folder.to_str().unwrap(),
            "--db-path", 
            "/nonexistent/database/path"
        ])
        .current_dir(".")
        .output()?;

    // Should fail gracefully with a clear error message
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    
    // Should contain helpful error information
    assert!(stderr.contains("database") || stderr.contains("connection") || stderr.contains("path"));
    
    Ok(())
}

/// Test that the CLI handles invalid commands gracefully
#[test]
fn test_invalid_command() -> Result<()> {
    let output = Command::new("cargo")
        .args(&["run", "--", "invalid-command"])
        .current_dir(".")
        .output()?;

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    
    // Should provide helpful error message
    assert!(stderr.contains("invalid-command") || stderr.contains("subcommand"));
    
    Ok(())
}

/// Test that the CLI provides version information
#[test]
fn test_version_command() -> Result<()> {
    let output = Command::new("cargo")
        .args(&["run", "--", "--version"])
        .current_dir(".")
        .output()?;

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout)?;
    
    // Should contain version information
    assert!(stdout.contains("code-ingest") && stdout.contains("0.1.0"));
    
    Ok(())
}

/// Test that individual command help works
#[test]
fn test_command_specific_help() -> Result<()> {
    let commands = ["ingest", "sql", "list-tables", "sample", "pg-start"];
    
    for command in &commands {
        let output = Command::new("cargo")
            .args(&["run", "--", command, "--help"])
            .current_dir(".")
            .output()?;

        assert!(output.status.success(), "Help for command '{}' failed", command);
        let stdout = String::from_utf8(output.stdout)?;
        
        // Each command should have its own help text
        assert!(stdout.len() > 50, "Help text for '{}' seems too short", command);
    }
    
    Ok(())
}

/// Test that the CLI handles missing required arguments appropriately
#[test]
fn test_missing_required_arguments() -> Result<()> {
    // Test commands that require arguments
    let test_cases = vec![
        vec!["sql"], // Missing query
        vec!["sample"], // Missing table
        vec!["describe"], // Missing table
        vec!["store-result"], // Missing multiple required args
    ];
    
    for args in test_cases {
        let output = Command::new("cargo")
            .args(&["run", "--"])
            .args(&args)
            .current_dir(".")
            .output()?;

        assert!(!output.status.success(), "Command {:?} should fail with missing args", args);
        
        let stderr = String::from_utf8(output.stderr)?;
        // Should provide helpful error about missing arguments
        assert!(
            stderr.contains("required") || stderr.contains("argument") || stderr.contains("missing"),
            "Error message for {:?} should mention missing arguments: {}", 
            args, 
            stderr
        );
    }
    
    Ok(())
}

/// Test that global --db-path option works
#[test]
fn test_global_db_path_option() -> Result<()> {
    let output = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "--db-path", 
            "/tmp/test.db",
            "list-tables"
        ])
        .current_dir(".")
        .output()?;

    // Should fail because database doesn't exist, but should parse arguments correctly
    assert!(!output.status.success());
    
    // The error should be about database connection, not argument parsing
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("database") || stderr.contains("connection") || stderr.contains("path"),
        "Should fail with database error, not argument error: {}", 
        stderr
    );
    
    Ok(())
}

/// Test that the CLI can handle local folder paths for ingestion
#[test]
fn test_local_folder_ingestion_argument_parsing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_folder = temp_dir.path().join("test_code");
    fs::create_dir(&test_folder)?;
    fs::write(test_folder.join("main.rs"), "fn main() { println!(\"Hello!\"); }")?;
    fs::write(test_folder.join("README.md"), "# Test Project")?;
    
    let output = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "ingest", 
            test_folder.to_str().unwrap(),
            "--db-path", 
            "/tmp/nonexistent.db"
        ])
        .current_dir(".")
        .output()?;

    // Should fail due to database connection, but arguments should parse correctly
    assert!(!output.status.success());
    
    let stderr = String::from_utf8(output.stderr)?;
    // Error should be about database, not about invalid folder path
    assert!(
        stderr.contains("database") || stderr.contains("connection"),
        "Should fail with database error, got: {}", 
        stderr
    );
    
    Ok(())
}

/// Test that GitHub URL parsing works (without actually cloning)
#[test]
fn test_github_url_argument_parsing() -> Result<()> {
    let output = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "ingest", 
            "https://github.com/rust-lang/mdBook",
            "--db-path", 
            "/tmp/nonexistent.db"
        ])
        .current_dir(".")
        .output()?;

    // Should fail due to database connection, but URL should be recognized
    assert!(!output.status.success());
    
    let stderr = String::from_utf8(output.stderr)?;
    // Should fail with database error, not URL parsing error
    assert!(
        stderr.contains("database") || stderr.contains("connection"),
        "Should fail with database error, got: {}", 
        stderr
    );
    
    Ok(())
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Test that CLI startup time is reasonable
    #[test]
    fn test_cli_startup_performance() -> Result<()> {
        let start = Instant::now();
        
        let output = Command::new("cargo")
            .args(&["run", "--", "--help"])
            .current_dir(".")
            .output()?;

        let duration = start.elapsed();
        
        assert!(output.status.success());
        
        // CLI should start reasonably quickly (allowing for compilation time in tests)
        // This is more about ensuring we don't have obvious performance regressions
        assert!(
            duration.as_secs() < 30, 
            "CLI startup took too long: {:?}", 
            duration
        );
        
        Ok(())
    }
}