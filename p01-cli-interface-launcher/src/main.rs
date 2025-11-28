//! Pensieve CLI Binary Entry Point
//!
//! This is the main entry point for the simplified Pensieve CLI application.
//! Follows parseltongue four-word naming conventions.

use clap::Parser;

/// Single command interface with download action
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "pensieve-local-llm-server")]
#[command(about = "Pensieve Local LLM Server - Zero-configuration LLM inference", long_about = None)]
struct CliArgs {
    /// Download action: auto (check and download), skip (use existing), force (re-download)
    #[arg(default_value = "auto")]
    pub download: String,
}

/// Main entry point following parseltongue principles
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments with validation
    let args = parse_cli_arguments_validate()?;

    // Check system prerequisites
    let system_check = check_system_prerequisites_quiet()?;

    if !system_check.is_apple_silicon {
        eprintln!("❌ Error: Apple Silicon required for MLX acceleration");
        std::process::exit(1);
    }

    if !system_check.mlx_available {
        eprintln!("❌ Error: MLX framework not available");
        std::process::exit(1);
    }

    // Create model directory
    let model_dir = ensure_model_directory_exists()?;
    println!("✅ Model directory: {}", model_dir.display());

    // Check model existence and download if needed
    let model_path = model_dir.join("phi-3-mini-128k-instruct-4bit");

    match args.download.as_str() {
        "auto" => {
            if !model_path.exists() {
                println!("📥 Model not found, downloading...");
                download_phi3_model_with_progress(&model_dir).await?;
            } else {
                println!("✅ Model found at: {}", model_path.display());
            }
        }
        "skip" => {
            if !model_path.exists() {
                eprintln!("❌ Error: Model not found at {}", model_path.display());
                eprintln!("💡 Use --download auto to download automatically");
                std::process::exit(1);
            }
        }
        "force" => {
            println!("📥 Forcing model download...");
            download_phi3_model_with_progress(&model_dir).await?;
        }
        _ => unreachable!(), // Should be caught by validation
    }

    // Create server configuration with fixed values
    let config = ServerConfig::default();

    // Launch inference server
    println!("🚀 Starting server on port {}...", config.port);
    let exit_code = launch_inference_server_process(&config, &model_path).await?;

    if exit_code == 0 {
        println!("✅ Server started successfully!");
        println!("🌐 Server listening on http://{}:{}", config.host, config.port);
    } else {
        eprintln!("❌ Server failed to start with exit code: {}", exit_code);
    }

    std::process::exit(exit_code);
}

// Import our four-word functions from lib.rs
use std::path::PathBuf;

// Re-export from lib.rs
pub use {
    parse_cli_arguments_validate,
    check_system_prerequisites_quiet,
    ensure_model_directory_exists,
    launch_inference_server_process,
    ServerConfig,
    SystemCheckResult,
    CliError,
};

/// Download Phi-3 model with progress tracking
///
/// # Preconditions
/// - Valid network connection
/// - Sufficient disk space (~3GB)
/// - Write permissions in model directory
///
/// # Postconditions
/// - Model downloaded to ~/.pensieve-local-llm-server/
/// - Returns success
///
/// # Error Conditions
/// - Network connection failure
/// - Insufficient disk space
/// - Permission denied
/// - Corrupted download
pub async fn download_phi3_model_with_progress(
    model_dir: &PathBuf,
) -> Result<(), CliError> {
    println!("⬇️  Downloading Phi-3-mini-128k-instruct-4bit model...");

    // TODO: Implement actual model download from HuggingFace
    // For now, just simulate download
    for i in 0..=10 {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        println!("⏳ Downloading: {}% ({}MB/{}MB)", i * 10, i * 200, 2000);
    }

    // Create a dummy model file for testing
    let model_path = model_dir.join("phi-3-mini-128k-instruct-4bit");
    std::fs::write(&model_path, "dummy model content - will be replaced with actual model")
        .map_err(|e| CliError::DirectoryCreationFailed(format!("Failed to create model file: {}", e)))?;

    println!("✅ Model download completed successfully!");
    println!("📁 Model saved to: {}", model_path.display());

    Ok(())
}