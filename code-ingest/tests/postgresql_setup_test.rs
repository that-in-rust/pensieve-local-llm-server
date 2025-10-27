//! Integration tests for PostgreSQL setup functionality
//!
//! These tests verify that the PostgreSQL setup guidance and connectivity testing
//! work correctly across different system configurations.

use code_ingest::database::{PostgreSQLSetup, SystemInfo, ConnectionTest};
use std::env;
use tokio;

/// Helper function to create a test database connection
fn get_test_database_url() -> Option<String> {
    env::var("DATABASE_URL").ok()
}

#[tokio::test]
async fn test_system_info_detection() {
    let setup = PostgreSQLSetup::new();
    let system_info = setup.get_system_info().await;
    
    // Basic system information should be detected
    assert!(!system_info.os.is_empty());
    assert!(!system_info.arch.is_empty());
    
    // OS should be one of the expected values
    assert!(
        system_info.os == "linux" || 
        system_info.os == "macos" || 
        system_info.os == "windows"
    );
    
    // Architecture should be detected
    assert!(
        system_info.arch == "x86_64" || 
        system_info.arch == "aarch64" ||
        system_info.arch == "arm" ||
        system_info.arch.contains("86")
    );
    
    // Shell information should be available (even if unknown)
    assert!(!system_info.shell.is_empty());
    
    println!("✅ System info detection test passed");
    println!("   OS: {} ({})", system_info.os, system_info.arch);
    println!("   Shell: {}", system_info.shell);
    println!("   Homebrew: {}", system_info.has_homebrew);
    println!("   APT: {}", system_info.has_apt);
    println!("   YUM: {}", system_info.has_yum);
    println!("   PostgreSQL: {}", system_info.has_psql);
}

#[tokio::test]
async fn test_setup_instructions_generation() {
    let setup = PostgreSQLSetup::new();
    let instructions = setup.generate_setup_instructions().await;
    
    // Should generate exactly 5 setup steps
    assert_eq!(instructions.len(), 5);
    
    // Verify step numbering and basic structure
    for (i, step) in instructions.iter().enumerate() {
        assert_eq!(step.step_number, i + 1);
        assert!(!step.title.is_empty());
        assert!(!step.description.is_empty());
        // Commands may be empty if PostgreSQL is already installed
        assert!(!step.troubleshooting.is_empty());
    }
    
    // Verify expected step titles
    assert!(instructions[0].title.to_lowercase().contains("install"));
    assert!(instructions[1].title.to_lowercase().contains("start") || instructions[1].title.to_lowercase().contains("service"));
    assert!(instructions[2].title.to_lowercase().contains("database") || instructions[2].title.to_lowercase().contains("create"));
    assert!(instructions[3].title.to_lowercase().contains("environment") || instructions[3].title.to_lowercase().contains("configure"));
    assert!(instructions[4].title.to_lowercase().contains("test") || instructions[4].title.to_lowercase().contains("connection"));
    
    println!("✅ Setup instructions generation test passed");
    println!("   Generated {} steps", instructions.len());
    for step in &instructions {
        println!("   Step {}: {}", step.step_number, step.title);
    }
}

#[tokio::test]
async fn test_connection_testing_with_valid_url() {
    if let Some(database_url) = get_test_database_url() {
        let setup = PostgreSQLSetup::new();
        let test_result = setup.test_connection(Some(&database_url)).await;
        
        if test_result.success {
            // If connection is successful, verify the results
            assert!(test_result.database_url.is_some());
            assert!(test_result.server_version.is_some());
            assert!(test_result.connection_time_ms.is_some());
            assert!(test_result.error_message.is_none());
            assert!(!test_result.suggestions.is_empty());
            
            // Verify server version format
            let version = test_result.server_version.unwrap();
            assert!(version.to_lowercase().contains("postgresql"));
            
            // Connection time should be reasonable
            let connection_time = test_result.connection_time_ms.unwrap();
            assert!(connection_time < 10000); // Less than 10 seconds
            
            println!("✅ Valid connection test passed");
            println!("   Database URL: {}", test_result.database_url.unwrap());
            println!("   Server Version: {}", version);
            println!("   Connection Time: {}ms", connection_time);
        } else {
            // If connection failed, verify error handling
            assert!(test_result.error_message.is_some());
            assert!(!test_result.suggestions.is_empty());
            
            println!("⚠️  Connection failed (expected if no test database)");
            println!("   Error: {}", test_result.error_message.unwrap());
            println!("   Suggestions: {}", test_result.suggestions.len());
        }
    } else {
        println!("⏭️  Skipping valid connection test (no DATABASE_URL)");
    }
}

#[tokio::test]
async fn test_connection_testing_with_invalid_url() {
    let setup = PostgreSQLSetup::new();
    
    // Test with obviously invalid URLs
    let invalid_urls = vec![
        "invalid://url",
        "postgresql://nonexistent:password@localhost:9999/nonexistent",
        "postgresql://user:pass@nonexistent.host:5432/db",
        "not-a-url-at-all",
        "",
    ];
    
    for invalid_url in &invalid_urls {
        let test_result = setup.test_connection(Some(invalid_url)).await;
        
        // Should always fail for invalid URLs
        assert!(!test_result.success);
        assert!(test_result.error_message.is_some());
        assert!(!test_result.suggestions.is_empty());
        
        // Should provide helpful suggestions
        let suggestions = &test_result.suggestions;
        assert!(suggestions.iter().any(|s| s.contains("pg-start") || s.contains("setup")));
    }
    
    println!("✅ Invalid connection test passed");
    println!("   Tested {} invalid URLs", invalid_urls.len());
}

#[tokio::test]
async fn test_connection_error_suggestions() {
    let setup = PostgreSQLSetup::new();
    
    // Test different error message patterns
    let error_patterns = vec![
        ("connection refused", "not running"),
        ("authentication failed", "username and password"),
        ("database does not exist", "createdb"),
        ("timeout", "network"),
        ("invalid url", "format"),
    ];
    
    for (error_msg, expected_suggestion) in error_patterns {
        // Test that connection test provides suggestions for different error types
        let test_result = setup.test_connection(Some(&format!("postgresql://invalid:invalid@localhost:5432/test?error={}", error_msg))).await;
        let suggestions = &test_result.suggestions;
        
        assert!(!suggestions.is_empty());
        assert!(
            suggestions.iter().any(|s| s.to_lowercase().contains(expected_suggestion)),
            "Expected suggestion containing '{}' for error '{}', got: {:?}",
            expected_suggestion, error_msg, suggestions
        );
    }
    
    println!("✅ Connection error suggestions test passed");
}

#[tokio::test]
async fn test_setup_instructions_formatting() {
    let setup = PostgreSQLSetup::new();
    let system_info = setup.get_system_info().await;
    let instructions = setup.generate_setup_instructions().await;
    
    let formatted = setup.format_setup_instructions(&instructions, &system_info);
    
    // Verify basic structure
    assert!(formatted.contains("PostgreSQL Setup Guide"));
    assert!(formatted.contains("System Information"));
    assert!(formatted.contains("Package Managers"));
    assert!(formatted.contains("Quick Reference"));
    
    // Should contain system-specific information
    assert!(formatted.contains(&system_info.os));
    assert!(formatted.contains(&system_info.arch));
    
    // Should contain all steps
    for step in &instructions {
        assert!(formatted.contains(&step.title));
        assert!(formatted.contains(&step.description));
    }
    
    // Should contain practical commands
    assert!(formatted.contains("code-ingest"));
    assert!(formatted.contains("DATABASE_URL"));
    
    println!("✅ Setup instructions formatting test passed");
    println!("   Formatted length: {} characters", formatted.len());
}

#[tokio::test]
async fn test_connection_test_formatting() {
    let setup = PostgreSQLSetup::new();
    
    // Test successful connection formatting
    let success_test = ConnectionTest {
        success: true,
        database_url: Some("postgresql://localhost:5432/test".to_string()),
        server_version: Some("PostgreSQL 14.0 on x86_64-pc-linux-gnu".to_string()),
        database_name: Some("test_db".to_string()),
        connection_time_ms: Some(25),
        error_message: None,
        suggestions: vec!["Connection successful!".to_string()],
    };
    
    let formatted_success = setup.format_connection_test(&success_test);
    
    assert!(formatted_success.contains("✅"));
    assert!(formatted_success.contains("Connection Successful"));
    assert!(formatted_success.contains("PostgreSQL 14.0"));
    assert!(formatted_success.contains("test_db"));
    assert!(formatted_success.contains("25ms"));
    assert!(formatted_success.contains("Next Steps"));
    
    // Test failed connection formatting
    let failed_test = ConnectionTest {
        success: false,
        database_url: Some("postgresql://localhost:5432/test".to_string()),
        server_version: None,
        database_name: None,
        connection_time_ms: None,
        error_message: Some("connection refused".to_string()),
        suggestions: vec![
            "PostgreSQL server is not running".to_string(),
            "Check if PostgreSQL is listening on port 5432".to_string(),
        ],
    };
    
    let formatted_failed = setup.format_connection_test(&failed_test);
    
    assert!(formatted_failed.contains("❌"));
    assert!(formatted_failed.contains("Connection Failed"));
    assert!(formatted_failed.contains("connection refused"));
    assert!(formatted_failed.contains("not running"));
    assert!(formatted_failed.contains("Suggestions"));
    
    println!("✅ Connection test formatting test passed");
}

#[tokio::test]
async fn test_platform_specific_instructions() {
    let setup = PostgreSQLSetup::new();
    
    // Test different system configurations
    let test_systems = vec![
        SystemInfo {
            os: "macos".to_string(),
            arch: "x86_64".to_string(),
            shell: "/bin/zsh".to_string(),
            has_homebrew: true,
            has_apt: false,
            has_yum: false,
            has_psql: false,
            postgresql_running: false,
        },
        SystemInfo {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            shell: "/bin/bash".to_string(),
            has_homebrew: false,
            has_apt: true,
            has_yum: false,
            has_psql: false,
            postgresql_running: false,
        },
        SystemInfo {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            shell: "/bin/bash".to_string(),
            has_homebrew: false,
            has_apt: false,
            has_yum: true,
            has_psql: false,
            postgresql_running: false,
        },
    ];
    
    for system_info in test_systems {
        let instructions = setup.generate_setup_instructions().await;
        let formatted = setup.format_setup_instructions(&instructions, &system_info);
        
        // Should contain platform-specific package manager info
        if system_info.has_homebrew {
            assert!(formatted.contains("Homebrew"));
            assert!(formatted.contains("brew"));
        }
        
        if system_info.has_apt {
            assert!(formatted.contains("APT"));
            assert!(formatted.contains("apt"));
        }
        
        if system_info.has_yum {
            assert!(formatted.contains("YUM"));
            assert!(formatted.contains("yum") || formatted.contains("dnf"));
        }
        
        // Should contain shell-specific configuration
        if system_info.shell.contains("zsh") {
            assert!(formatted.contains("zshrc"));
        } else if system_info.shell.contains("bash") {
            assert!(formatted.contains("bashrc"));
        }
    }
    
    println!("✅ Platform-specific instructions test passed");
}

#[tokio::test]
async fn test_setup_manager_default() {
    let setup1 = PostgreSQLSetup::new();
    let setup2 = PostgreSQLSetup::default();
    
    // Both should work the same way
    let info1 = setup1.get_system_info().await;
    let info2 = setup2.get_system_info().await;
    
    assert_eq!(info1.os, info2.os);
    assert_eq!(info1.arch, info2.arch);
    
    println!("✅ Setup manager default test passed");
}

#[tokio::test]
async fn test_comprehensive_setup_workflow() {
    let setup = PostgreSQLSetup::new();
    
    // Simulate complete setup workflow
    println!("🔄 Running comprehensive setup workflow test...");
    
    // Step 1: Get system info
    let system_info = setup.get_system_info().await;
    assert!(!system_info.os.is_empty());
    
    // Step 2: Generate instructions
    let instructions = setup.generate_setup_instructions().await;
    assert_eq!(instructions.len(), 5);
    
    // Step 3: Format instructions
    let formatted_instructions = setup.format_setup_instructions(&instructions, &system_info);
    assert!(formatted_instructions.len() > 1000); // Should be comprehensive
    
    // Step 4: Test connection (may fail, that's OK)
    let connection_test = setup.test_connection(None).await;
    assert!(connection_test.database_url.is_some() || !connection_test.success);
    
    // Step 5: Format connection test
    let formatted_test = setup.format_connection_test(&connection_test);
    assert!(formatted_test.contains("PostgreSQL Connection Test"));
    
    println!("✅ Comprehensive setup workflow test passed");
    println!("   System: {} {}", system_info.os, system_info.arch);
    println!("   Instructions: {} chars", formatted_instructions.len());
    println!("   Connection: {}", if connection_test.success { "✅" } else { "❌" });
}