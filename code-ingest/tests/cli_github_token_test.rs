//! Tests for GitHub token handling in CLI commands
//!
//! These tests verify that the CLI properly handles GitHub authentication
//! through environment variables and command-line arguments.

use std::process::Command;
use std::env;

/// Test that GitHub token can be passed via environment variable
#[test]
fn test_github_token_environment_variable() {
    let output = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "ingest", 
            "https://github.com/rust-lang/mdBook",
            "--db-path", 
            "/tmp/nonexistent.db"
        ])
        .env("GITHUB_TOKEN", "fake_token_for_testing")
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    // Should fail due to database connection, not token issues
    assert!(!output.status.success());
    
    let stderr = String::from_utf8(output.stderr).expect("Invalid UTF-8");
    // Should fail with database error, not authentication error
    assert!(
        stderr.contains("database") || stderr.contains("connection"),
        "Should fail with database error, got: {}", 
        stderr
    );
}

/// Test that the CLI accepts GitHub URLs without requiring immediate authentication
#[test]
fn test_github_url_parsing_without_token() {
    let output = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "ingest", 
            "https://github.com/rust-lang/mdBook",
            "--db-path", 
            "/tmp/nonexistent.db"
        ])
        .env_remove("GITHUB_TOKEN") // Ensure no token is set
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    // Should fail due to database connection, not URL parsing
    assert!(!output.status.success());
    
    let stderr = String::from_utf8(output.stderr).expect("Invalid UTF-8");
    // Should not fail with URL parsing error
    assert!(
        !stderr.contains("invalid") || !stderr.contains("url"),
        "Should not fail with URL parsing error, got: {}", 
        stderr
    );
}

/// Test that various GitHub URL formats are accepted
#[test]
fn test_github_url_formats() {
    let test_urls = vec![
        "https://github.com/user/repo",
        "https://github.com/user/repo.git",
        "git@github.com:user/repo.git",
        "https://gitlab.com/user/repo",
        "https://bitbucket.org/user/repo",
    ];
    
    for url in test_urls {
        let output = Command::new("cargo")
            .args(&[
                "run", 
                "--", 
                "ingest", 
                url,
                "--db-path", 
                "/tmp/nonexistent.db"
            ])
            .current_dir(".")
            .output()
            .expect("Failed to execute command");

        // Should fail due to database, not URL format
        assert!(!output.status.success(), "URL format should be accepted: {}", url);
        
        let stderr = String::from_utf8(output.stderr).expect("Invalid UTF-8");
        // Should fail with database error, not URL format error
        assert!(
            stderr.contains("database") || stderr.contains("connection"),
            "URL {} should be accepted, got error: {}", 
            url, 
            stderr
        );
    }
}

/// Test that local paths are distinguished from URLs
#[test]
fn test_local_path_vs_url_detection() {
    let test_cases = vec![
        ("/tmp/local/path", true),  // Should be treated as local path
        ("./relative/path", true),  // Should be treated as local path
        ("https://github.com/user/repo", false), // Should be treated as URL
        ("git@github.com:user/repo.git", false), // Should be treated as URL
    ];
    
    for (input, is_local) in test_cases {
        let output = Command::new("cargo")
            .args(&[
                "run", 
                "--", 
                "ingest", 
                input,
                "--db-path", 
                "/tmp/nonexistent.db"
            ])
            .current_dir(".")
            .output()
            .expect("Failed to execute command");

        // All should fail due to database or file not found, but not due to parsing
        assert!(!output.status.success());
        
        let stderr = String::from_utf8(output.stderr).expect("Invalid UTF-8");
        
        if is_local {
            // Local paths might fail with "file not found" or "database" error
            assert!(
                stderr.contains("database") || 
                stderr.contains("connection") || 
                stderr.contains("No such file") ||
                stderr.contains("not found"),
                "Local path {} should fail appropriately, got: {}", 
                input, 
                stderr
            );
        } else {
            // URLs should fail with database error (not URL parsing error)
            assert!(
                stderr.contains("database") || stderr.contains("connection"),
                "URL {} should fail with database error, got: {}", 
                input, 
                stderr
            );
        }
    }
}

/// Test that progress feedback is mentioned in help text
#[test]
fn test_ingest_command_help_mentions_progress() {
    let output = Command::new("cargo")
        .args(&["run", "--", "ingest", "--help"])
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("Invalid UTF-8");
    
    // Help should mention the ingest functionality
    assert!(stdout.contains("Ingest a GitHub repository or local folder"));
}

/// Test that database path can be provided globally or per-command
#[test]
fn test_database_path_options() {
    // Test global --db-path
    let output1 = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "--db-path", 
            "/tmp/test1.db",
            "list-tables"
        ])
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    // Test command-specific --db-path
    let output2 = Command::new("cargo")
        .args(&[
            "run", 
            "--", 
            "list-tables",
            "--db-path", 
            "/tmp/test2.db"
        ])
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    // Both should fail with database connection error (not argument parsing error)
    assert!(!output1.status.success());
    assert!(!output2.status.success());
    
    let stderr1 = String::from_utf8(output1.stderr).expect("Invalid UTF-8");
    let stderr2 = String::from_utf8(output2.stderr).expect("Invalid UTF-8");
    
    // Both should fail with database-related errors
    assert!(
        stderr1.contains("database") || stderr1.contains("connection"),
        "Global db-path should work, got: {}", 
        stderr1
    );
    assert!(
        stderr2.contains("database") || stderr2.contains("connection"),
        "Command-specific db-path should work, got: {}", 
        stderr2
    );
}

/// Test that error messages are clear and actionable
#[test]
fn test_clear_error_messages() {
    // Test missing database path
    let output = Command::new("cargo")
        .args(&["run", "--", "list-tables"])
        .env_remove("DATABASE_URL") // Ensure no DATABASE_URL is set
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("Invalid UTF-8");
    
    // Error message should be helpful and mention how to fix it
    assert!(
        stderr.contains("database") && 
        (stderr.contains("--db-path") || stderr.contains("DATABASE_URL")),
        "Error message should mention how to provide database path, got: {}", 
        stderr
    );
}

/// Test that success messages provide next steps
#[test]
fn test_help_provides_next_steps() {
    let output = Command::new("cargo")
        .args(&["run", "--", "pg-start"])
        .current_dir(".")
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("Invalid UTF-8");
    
    // pg-start should provide actionable next steps
    assert!(stdout.contains("Next Steps") || stdout.contains("next steps") || stdout.contains("Ready to Get Started") || stdout.contains("Step"));
    assert!(stdout.contains("DATABASE_URL") || stdout.contains("database"));
}