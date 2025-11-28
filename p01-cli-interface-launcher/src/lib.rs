# STUB Phase: CLI Interface Tests
# These tests will fail initially - this is the RED phase of TDD

use clap::Parser;
use std::path::PathBuf;

/// CLI arguments following four-word naming convention
#[derive(Debug, Parser)]
struct CliArgs {
    /// Download action: auto (check and download), skip (use existing), force (re-download)
    #[arg(default_value = "auto")]
    download: String,
}

// Four-word function names as per parseltongue principles

/// Parse CLI arguments with validation
///
/// # Preconditions
/// - Command line args available in env::args()
/// - Args follow pensieve CLI format
///
/// # Postconditions
/// - Returns Ok(CliArgs) if valid
/// - Returns Err(CliError) if invalid
///
/// # Error Conditions
/// - Invalid argument format
/// - Invalid download action
pub fn parse_cli_arguments_validate() -> Result<CliArgs, CliError> {
    let args = CliArgs::try_parse()
        .map_err(|e| CliError::InvalidArgument(format!("Failed to parse arguments: {}", e)))?;

    // Validate download action
    if !matches!(args.download.as_str(), "auto" | "skip" | "force") {
        return Err(CliError::InvalidArgument(format!(
            "Invalid download action: {}. Must be one of: auto, skip, force",
            args.download
        )));
    }

    Ok(args)
}

/// Check system prerequisites silently
///
/// # Preconditions
/// - System resources accessible
/// - macOS environment with Apple Silicon
///
/// # Postconditions
/// - Returns SystemCheckResult with availability status
/// - Logs minimal output
///
/// # Error Conditions
/// - Unsupported architecture
/// - Insufficient system resources
pub fn check_system_prerequisites_quiet() -> Result<SystemCheckResult, CliError> {
    // Check architecture (should be Apple Silicon)
    let is_apple_silicon = std::env::consts::ARCH.contains("aarch64") ||
                           std::env::consts::ARCH.contains("arm64");

    // Check minimum memory requirement (8GB)
    let has_sufficient_memory = true; // TODO: Implement actual memory check

    // Check disk space (5GB for model)
    let has_sufficient_disk = true; // TODO: Implement actual disk space check

    // Check MLX availability (assume available on Apple Silicon)
    let mlx_available = is_apple_silicon;

    Ok(SystemCheckResult {
        is_apple_silicon,
        has_sufficient_memory,
        has_sufficient_disk,
        mlx_available,
    })
}

/// Create model directory structure
///
/// # Preconditions
/// - Write permissions in home directory
/// - Home directory environment variable set
///
/// # Postconditions
/// - Creates ~/.pensieve-local-llm-server/ directory
/// - Returns PathBuf to model directory
///
/// # Error Conditions
/// - Permission denied
/// - Invalid home directory path
pub fn ensure_model_directory_exists() -> Result<PathBuf, CliError> {
    let home_dir = std::env::var("HOME")
        .map_err(|_| CliError::DirectoryCreationFailed("HOME environment variable not set".to_string()))?;

    let model_dir = std::path::PathBuf::from(home_dir)
        .join(".pensieve-local-llm-server");

    std::fs::create_dir_all(&model_dir)
        .map_err(|e| CliError::DirectoryCreationFailed(format!(
            "Failed to create model directory {}: {}",
            model_dir.display(),
            e
        )))?;

    Ok(model_dir)
}

/// Launch inference server process
///
/// # Preconditions
/// - Valid ServerConfig provided
/// - Model available in storage
/// - Port 528491 is available
///
/// # Postconditions
/// - Server listening on port 528491
/// - Returns exit code on completion
///
/// # Error Conditions
/// - Port already in use
/// - Model loading failure
/// - Permission denied
pub async fn launch_inference_server_process(
    config: &ServerConfig,
    model_path: &PathBuf,
) -> Result<i32, CliError> {
    // For now, just return success - will implement actual server launching
    // when we integrate with p02-http-server-core
    println!("Starting server with config: {:?}", config);
    println!("Model path: {:?}", model_path);

    // Simulate server startup
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Ok(0) // Success exit code
}

// Supporting types

/// System check result
#[derive(Debug, Clone)]
pub struct SystemCheckResult {
    pub is_apple_silicon: bool,
    pub has_sufficient_memory: bool,
    pub has_sufficient_disk: bool,
    pub mlx_available: bool,
}

/// Server configuration with fixed constraints
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,           // Fixed: "127.0.0.1"
    pub port: u16,             // Fixed: 528491
    pub max_concurrent_requests: usize,  // Fixed: 10
    pub request_timeout_ms: u64,        // Fixed: 30000
    pub enable_cors: bool,              // Fixed: true
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 528491,  // Fixed port for Claude Code
            max_concurrent_requests: 10,  // Conservative limit
            request_timeout_ms: 30000,
            enable_cors: true,
        }
    }
}

/// CLI error types following parseltongue patterns
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("System check failed: {0}")]
    SystemCheckFailed(String),

    #[error("Directory creation failed: {0}")]
    DirectoryCreationFailed(String),

    #[error("Server launch failed: {0}")]
    ServerLaunchFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cli_arguments_valid_auto() -> Result<(), CliError> {
        // STUB test - will fail until implementation
        let args = vec!["pensieve", "auto"];
        // TODO: Need to inject args into parse_cli_arguments_validate
        let result = parse_cli_arguments_validate()?;
        assert_eq!(result.download, "auto");
        Ok(())
    }

    #[test]
    fn test_parse_cli_arguments_valid_skip() -> Result<(), CliError> {
        // STUB test - will fail until implementation
        let result = parse_cli_arguments_validate()?;
        // This test needs arg injection - will implement later
        Ok(())
    }

    #[test]
    fn test_ensure_model_directory_creates_directory() -> Result<(), CliError> {
        // STUB test - will fail until implementation
        let result = ensure_model_directory_exists()?;
        assert!(result.exists());
        assert!(result.is_dir());
        Ok(())
    }

    #[test]
    fn test_system_check_apple_silicon() -> Result<(), CliError> {
        // STUB test - will fail until implementation
        let result = check_system_prerequisites_quiet()?;
        assert!(result.is_apple_silicon, "Should detect Apple Silicon");
        Ok(())
    }

    #[test]
    fn test_server_config_fixed_values() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 528491, "Port should be fixed to 528491");
        assert_eq!(config.host, "127.0.0.1", "Host should be fixed to 127.0.0.1");
        assert_eq!(config.max_concurrent_requests, 10, "Should have fixed concurrency limit");
    }

    #[tokio::test]
    async fn test_launch_server_with_valid_config() -> Result<(), CliError> {
        // STUB test - will fail until implementation
        let config = ServerConfig::default();
        let model_path = PathBuf::from("/tmp/test_model");
        let result = launch_inference_server_process(&config, &model_path).await?;
        // Should return exit code 0 for successful launch
        assert_eq!(result, 0);
        Ok(())
    }
}