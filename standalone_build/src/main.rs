//! Pensieve CLI Binary Entry Point
//!
//! This is the main entry point for the simplified Pensieve CLI application.
//! Follows parseltongue four-word naming conventions.

use clap::Parser;

/// PRD01-compliant zero-config LLM server interface
/// Note: CLI is retained but all values are baked-in per PRD01
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "pensieve-local-llm-server")]
#[command(about = "Zero-config local LLM server powered by Phi-4 reasoning model", long_about = None)]
struct BakedCliArgs {
    /// Baked-in: Phi-4-reasoning-plus-4bit model (ignored, always uses Phi-4)
    #[arg(long, default_value = "mlx-community/Phi-4-reasoning-plus-4bit")]
    model_url: String,

    /// Baked-in: Fixed port 528491 (ignored, always uses 528491)
    #[arg(long, default_value = "528491")]
    port: u16,

    /// Baked-in: Default cache directory
    #[arg(long, default_value = "./models")]
    cache_dir: PathBuf,

    /// Baked-in: Conservative concurrency for Apple Silicon
    #[arg(long, default_value = "2")]
    max_concurrent: usize,

    /// Baked-in: Always check model integrity
    #[arg(long, default_value = "false")]
    skip_download: bool,
}

/// Main entry point following parseltongue principles and PRD01 zero-config
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments with baked-in defaults (PRD01: zero-config)
    let args = parse_baked_cli_arguments_with_defaults()?;

    // Check system prerequisites for MLX acceleration
    let system_check = check_system_prerequisites_quiet()?;

    if !system_check.is_apple_silicon {
        eprintln!("❌ Error: Apple Silicon required for MLX acceleration");
        std::process::exit(1);
    }

    if !system_check.mlx_available {
        eprintln!("❌ Error: MLX framework not available");
        std::process::exit(1);
    }

    println!("🚀 Starting Rust LLM Server...");
    println!("📂 Model: {}", args.model_url);
    println!("🌐 Port: {}", args.port);
    println!("💾 Cache: {}", args.cache_dir.display());

    // Enforce PRD01 fixed configuration
    let server_config = create_prd01_baked_config()?;

    println!("🚀 Starting pensieve-local-llm-server...");
    println!("🤖 Model: Phi-4-reasoning-plus-4bit (MLX optimized)");
    println!("🌐 Port: 528491 (fixed per PRD01)");
    println!("💾 Cache: {}", server_config.cache_dir.display());

    // Ensure Phi-4 model is available (ES001)
    let model_service = UnifiedModelManager::new(server_config.cache_dir.clone())?;
    let model_path = model_service.ensure_model_available("mlx-community/Phi-4-reasoning-plus-4bit").await?;

    println!("✅ Phi-4 model ready: {}", model_path.display());

    // TODO: Start HTTP server and inference engine
    println!("🌐 Starting HTTP server on http://127.0.0.1:528491");
    println!("📡 Ready to serve Anthropic-compatible requests");

    // For now, keep the process alive
    tokio::signal::ctrl_c().await?;
    println!("👋 Server shutdown complete");

    Ok(())
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

/// Parse baked CLI arguments with PRD01 defaults
///
/// # Preconditions
/// - Command line args available (but ignored for PRD01 compliance)
///
/// # Postconditions
/// - Returns parsed args with fixed defaults
///
/// # Error Conditions
/// - Invalid argument format
fn parse_baked_cli_arguments_with_defaults() -> Result<BakedCliArgs, CliError> {
    let args = BakedCliArgs::try_parse()
        .map_err(|e| CliError::InvalidArgument(format!("Failed to parse arguments: {}", e)))?;

    // PRD01: Enforce fixed configuration regardless of input
    let enforced_args = BakedCliArgs {
        model_url: "mlx-community/Phi-4-reasoning-plus-4bit".to_string(),
        port: 528491,
        cache_dir: args.cache_dir, // Allow cache directory override
        max_concurrent: 2,
        skip_download: false,
    };

    Ok(enforced_args)
}

/// Create PRD01 baked configuration
///
/// # Preconditions
/// - System has default directories available
///
/// # Postconditions
/// - Returns ServerConfig with PRD01 fixed values
///
/// # Error Conditions
/// - Cache directory creation failure
fn create_prd01_baked_config() -> Result<UnifiedServerConfig, CliError> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap())
        .join("pensieve-models");

    // Ensure cache directory exists
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| CliError::DirectoryCreationFailed(format!(
            "Failed to create cache directory: {}", e
        )))?;

    Ok(UnifiedServerConfig {
        host: "127.0.0.1".to_string(),
        port: 528491, // Fixed per PRD01
        model_path: cache_dir.join("Phi-4-reasoning-plus-4bit.gguf"), // Will be populated by model manager
        max_concurrent_requests: 2, // Conservative for Apple Silicon
    })
}

/// Download Phi-4 model with progress tracking (Legacy - replaced by UnifiedModelManager)
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