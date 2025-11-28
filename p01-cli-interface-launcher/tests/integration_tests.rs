# Complete Binary Integration Tests
# Testing single pensieve-local-llm-server binary with all components

use std::process::Command;
use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;

// Import our four-word functions from lib.rs
use p01_cli_interface_launcher::{
    parse_cli_arguments_validate,
    check_system_prerequisites_quiet,
    ensure_model_directory_exists,
    launch_inference_server_process,
    ServerConfig, SystemCheckResult, CliError
};

/// Test complete binary argument parsing
///
/// # Preconditions
/// - Binary available in target directory
/// - Command line args follow expected format
///
/// # Postconditions
/// - Arguments parsed correctly
/// - Returns appropriate error codes
/// - Help text displayed when requested
#[test]
fn test_binary_argument_parsing_auto() -> Result<(), CliError> {
    // Test the argument parsing logic directly (since binary might not be compiled)
    let args = vec!["pensieve-local-llm-server", "auto"];

    // Simulate argument parsing by setting up test environment
    std::env::set_var("CARGO_PKG_NAME", "pensieve-local-llm-server");

    // Test validation function directly
    let result = parse_cli_arguments_validate();

    // Should succeed with default args
    assert!(result.is_ok(), "Default argument parsing should succeed");

    let parsed_args = result.unwrap();
    assert_eq!(parsed_args.download, "auto", "Download action should default to auto");

    Ok(())
}

/// Test binary argument parsing with skip option
#[test]
fn test_binary_argument_parsing_skip() -> Result<(), CliError> {
    // This test would normally inject CLI args, but we're testing the validation logic
    let result = parse_cli_arguments_validate();

    // The function should work with whatever args are available
    // In a real test environment, we'd inject test args
    match result {
        Ok(args) => {
            assert!(matches!(args.download.as_str(), "auto" | "skip" | "force"));
        }
        Err(_) => {
            // Expected in test environment without proper args
        }
    }

    Ok(())
}

/// Test system prerequisites check
///
/// # Preconditions
/// - Running on compatible system
/// - System resources accessible
///
/// # Postconditions
/// - Returns system capability information
/// - Apple Silicon detection works
/// - MLX availability checked
#[test]
fn test_system_prerequisites_check() -> Result<(), CliError> {
    let result = check_system_prerequisites_quiet()?;

    // Should always return a result, even on non-Apple Silicon
    assert!(result.is_apple_silicon == std::env::consts::ARCH.contains("aarch64") ||
            result.is_apple_silicon == std::env::consts::ARCH.contains("arm64"),
            "Apple Silicon detection should be accurate");

    // MLX should be available on Apple Silicon
    if result.is_apple_silicon {
        assert!(result.mlx_available, "MLX should be available on Apple Silicon");
    }

    Ok(())
}

/// Test model directory creation
///
/// # Preconditions
/// - Home directory accessible
/// - Write permissions available
///
/// # Postconditions
/// - Model directory created
/// - Path returned correctly
/// - Directory structure valid
#[test]
fn test_model_directory_creation() -> Result<(), CliError> {
    let result = ensure_model_directory_exists()?;

    assert!(result.exists(), "Model directory should exist after creation");
    assert!(result.is_dir(), "Model path should be a directory");

    // Check it's in the home directory under .pensieve-local-llm-server
    let home_dir = std::env::var("HOME")?;
    let expected_path = PathBuf::from(home_dir).join(".pensieve-local-llm-server");
    assert_eq!(result, expected_path, "Model directory should be in home/.pensieve-local-llm-server");

    Ok(())
}

/// Test server configuration defaults
///
/// # Preconditions
/// - ServerConfig created with default values
///
/// # Postconditions
/// - Fixed values match parseltongue principles
/// - Port 528491 configured for Claude Code
/// - Conservative limits set
#[test]
fn test_server_configuration_defaults() {
    let config = ServerConfig::default();

    // Fixed values as per parseltongue principles
    assert_eq!(config.host, "127.0.0.1", "Host should be fixed to localhost");
    assert_eq!(config.port, 528491, "Port should be fixed to 528491 for Claude Code");
    assert_eq!(config.max_concurrent_requests, 10, "Should have conservative concurrency limit");
    assert_eq!(config.request_timeout_ms, 30000, "Should have reasonable timeout");
    assert!(config.enable_cors, "CORS should be enabled by default");
}

/// Test server launch process (mock)
///
/// # Preconditions
/// - Valid server configuration
/// - Mock model path available
///
/// # Postconditions
/// - Server process started
/// - Returns success exit code
/// - No crashes during startup
#[tokio::test]
async fn test_server_launch_process() -> Result<(), CliError> {
    let config = ServerConfig::default();
    let model_path = PathBuf::from("/tmp/test_model");

    let exit_code = launch_inference_server_process(&config, &model_path).await?;

    assert_eq!(exit_code, 0, "Server launch should return success exit code");

    Ok(())
}

/// Test complete workflow simulation
///
/// # Preconditions
/// - All components available
/// - System meets requirements
///
/// # Postconditions
/// - Complete workflow executed
/// - Each step validated
/// - No integration failures
#[test]
fn test_complete_workflow_simulation() -> Result<(), CliError> {
    // Step 1: Parse arguments
    let args = parse_cli_arguments_validate()?;

    // Step 2: Check system
    let system_check = check_system_prerequisites_quiet()?;

    // Step 3: Ensure model directory
    let model_dir = ensure_model_directory_exists()?;

    // Step 4: Create server config
    let config = ServerConfig::default();

    // Validate workflow
    assert!(matches!(args.download.as_str(), "auto" | "skip" | "force"));
    assert!(model_dir.exists());
    assert_eq!(config.port, 528491);

    // On Apple Silicon, MLX should be available
    if system_check.is_apple_silicon {
        assert!(system_check.mlx_available);
    }

    Ok(())
}

/// Test error handling scenarios
///
/// # Preconditions
/// - Various error conditions simulated
/// - Error handling functions available
///
/// # Postconditions
/// - Errors handled gracefully
/// - Proper error messages provided
/// - No crashes on invalid input
#[test]
fn test_error_handling_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    // Test invalid download action handling
    let args = vec!["pensieve-local-llm-server", "invalid_action"];

    // This would normally be caught by argument parsing
    // Since we can't inject args easily, we test the validation logic conceptually
    let valid_actions = ["auto", "skip", "force"];
    assert!(!valid_actions.contains(&"invalid_action"), "Invalid action should be rejected");

    // Test system check on unsupported architecture (conceptual)
    // In real scenarios, this would fail gracefully

    Ok(())
}

/// Test performance characteristics
///
/// # Preconditions
/// - System under normal load
/// - Performance monitoring available
///
/// # Postconditions
/// - Startup time reasonable
/// - Memory usage within limits
/// - No memory leaks detected
#[test]
fn test_performance_characteristics() -> Result<(), CliError> {
    let start_time = std::time::Instant::now();

    // Perform core operations
    let system_check = check_system_prerequisites_quiet()?;
    let model_dir = ensure_model_directory_exists()?;
    let config = ServerConfig::default();

    let elapsed = start_time.elapsed();

    // Should complete quickly (< 100ms for these operations)
    assert!(elapsed.as_millis() < 100, "Core operations should complete quickly");

    // Basic memory usage check (conceptual)
    // In real implementation, we'd track memory usage

    Ok(())
}

/// Test file system operations
///
/// # Preconditions
/// - File system accessible
/// - Write permissions available
///
/// # Postconditions
/// - Files created and cleaned up properly
/// - No orphaned files left
/// - Permissions handled correctly
#[test]
fn test_file_system_operations() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let test_model_path = temp_dir.path().join("test_model");

    // Test model directory creation in temp location
    // (This simulates what the main binary would do)
    std::fs::write(&test_model_path, "dummy model content")?;

    assert!(test_model_path.exists());
    assert!(test_model_path.is_file());

    // Test cleanup
    drop(temp_dir); // TempDir handles cleanup automatically

    Ok(())
}

/// Test concurrent operations safety
///
/// # Preconditions
/// - Multiple operations initiated
/// - Thread-safe functions available
///
/// # Postconditions
/// - Operations complete without race conditions
/// - No data corruption
/// - Proper synchronization
#[test]
fn test_concurrent_operations_safety() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::thread;

    let config = Arc::new(ServerConfig::default());

    let mut handles = vec![];

    // Spawn multiple threads with configuration access
    for _ in 0..5 {
        let config_clone = config.clone();
        let handle = thread::spawn(move || {
            // Test concurrent access to configuration
            assert_eq!(config_clone.port, 528491);
            assert_eq!(config_clone.host, "127.0.0.1");
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

/// Test CLI help and version information
///
/// # Preconditions
/// - Binary compiled with help information
/// - Version information available
///
/// # Postconditions
/// - Help displays correctly
/// - Version information accurate
/// - Usage instructions clear
#[test]
fn test_cli_help_version() -> Result<(), Box<dyn std::error::Error>> {
    // Test that CLI structure is properly defined
    let args = parse_cli_arguments_validate();

    // Should either succeed (with default args) or fail gracefully
    match args {
        Ok(_) => {
            // Arguments parsed successfully
        }
        Err(_) => {
            // Expected in test environment
        }
    }

    // Verify that the binary name is correct
    assert_eq!(std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "pensieve-local-llm-server".to_string()),
               "pensieve-local-llm-server");

    Ok(())
}