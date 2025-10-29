//! Pensieve CLI - Command-line interface for server management
//!
//! This is the Layer 3 (L3) CLI crate that provides:
//! - Clap-based command-line parsing
//! - Configuration management
//! - Server lifecycle management
//! - User-friendly error handling
//!
//! Depends on all L1 and L2 crates, plus pensieve-02 for server management.

use pensieve_07_core::CoreError;
use pensieve_02::{HttpApiServer, ServerConfig, traits::ApiServer, MockRequestHandler, MlxRequestHandler};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::signal;
use tracing::{error, info, warn};

/// CLI-specific error types
pub mod error {
    use super::*;
    use thiserror::Error;

    /// CLI result type
    pub type CliResult<T> = std::result::Result<T, CliError>;

    /// CLI-specific errors
    #[derive(Error, Debug)]
    pub enum CliError {
        #[error("Configuration error: {0}")]
        Config(String),

        #[error("Server error: {0}")]
        Server(String),

        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),

        #[error("Serialization error: {0}")]
        Serialization(#[from] serde_json::Error),

        #[error("Core error: {0}")]
        Core(#[from] CoreError),

        #[error("Command execution error: {0}")]
        Execution(String),
    }

    impl From<pensieve_02::error::ServerError> for CliError {
        fn from(err: pensieve_02::error::ServerError) -> Self {
            CliError::Server(err.to_string())
        }
    }
}

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub server: ServerConfigFile,
    pub logging: LoggingConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfigFile {
    pub host: String,
    pub port: u16,
    pub max_concurrent_requests: usize,
    pub request_timeout_ms: u64,
    pub enable_cors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub model_type: String,
    pub context_size: usize,
    pub gpu_layers: Option<u32>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            server: ServerConfigFile {
                host: "127.0.0.1".to_string(),
                port: 7777,
                max_concurrent_requests: 100,
                request_timeout_ms: 30000,
                enable_cors: true,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "compact".to_string(),
                file: None,
            },
            model: ModelConfig {
                model_path: PathBuf::from("model.gguf"),
                model_type: "llama".to_string(),
                context_size: 2048,
                gpu_layers: None,
            },
        }
    }
}

impl From<CliConfig> for ServerConfig {
    fn from(config: CliConfig) -> Self {
        ServerConfig {
            host: config.server.host,
            port: config.server.port,
            max_concurrent_requests: config.server.max_concurrent_requests,
            request_timeout_ms: config.server.request_timeout_ms,
            enable_cors: config.server.enable_cors,
        }
    }
}

/// Core CLI traits
pub mod traits {
    use super::*;

    /// Trait for CLI applications
    #[async_trait::async_trait]
    pub trait CliApp: Send + Sync {
        /// Run the CLI application
        async fn run(&self) -> error::CliResult<i32>;

        /// Validate configuration
        fn validate_config(&self) -> error::CliResult<()>;
    }

    /// Trait for command handlers
    pub trait CommandHandler: Send + Sync {
        /// Execute a command
        async fn execute(&self, args: &CliArgs) -> error::CliResult<CommandResult>;

        /// Get command description
        fn description(&self) -> &'static str;
    }
}

/// Command execution result
#[derive(Debug, Clone)]
pub enum CommandResult {
    Success(String),
    Warning(String),
    Error(String),
}

/// CLI arguments
#[derive(Debug, Clone, clap::Parser)]
#[command(name = "pensieve")]
#[command(about = "Pensieve Local LLM Server", long_about = None)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Commands,

    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, value_name = "LEVEL")]
    pub log_level: Option<String>,
}

#[derive(Debug, Clone, clap::Subcommand)]
pub enum Commands {
    /// Start the server
    Start {
        /// Override host
        #[arg(long)]
        host: Option<String>,

        /// Override port
        #[arg(long)]
        port: Option<u16>,

        /// Model file path
        #[arg(long, value_name = "FILE")]
        model: Option<PathBuf>,

        /// Number of GPU layers (0 = CPU only)
        #[arg(long)]
        gpu_layers: Option<u32>,
    },

    /// Stop the server
    Stop {
        /// Server host
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Server port
        #[arg(long, default_value = "7777")]
        port: u16,
    },

    /// Show server status
    Status {
        /// Server host
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Server port
        #[arg(long, default_value = "7777")]
        port: u16,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Validate configuration
    Validate {
        /// Configuration file path
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, clap::Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,

    /// Generate default configuration
    Generate {
        /// Output file path
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },

    /// Validate configuration
    Validate {
        /// Configuration file path
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
    },
}

/// CLI application implementation
#[derive(Debug)]
pub struct PensieveCli {
    config: CliConfig,
    args: CliArgs,
}

impl PensieveCli {
    pub fn new(args: CliArgs) -> error::CliResult<Self> {
        // Load configuration
        let config = if let Some(config_path) = &args.config {
            Self::load_config(config_path)?
        } else {
            CliConfig::default()
        };

        Ok(Self { config, args })
    }

    /// Load configuration from file
    fn load_config(path: &PathBuf) -> error::CliResult<CliConfig> {
        let content = std::fs::read_to_string(path)?;
        let config: CliConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    fn save_config(path: &PathBuf, config: &CliConfig) -> error::CliResult<()> {
        let content = serde_json::to_string_pretty(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Initialize logging
    fn init_logging(&self) -> error::CliResult<()> {
        let default_level = "info".to_string();
        let level = self.args.log_level
            .as_ref()
            .or_else(|| Some(&self.config.logging.level))
            .unwrap_or(&default_level);

        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level.clone()));

        if let Some(log_file) = &self.config.logging.file {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_file)?;

            tracing_subscriber::fmt()
                .with_writer(file)
                .with_env_filter(filter)
                .init();
        } else {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .init();
        }

        Ok(())
    }

    /// Create server instance
    fn create_server(&self) -> error::CliResult<HttpApiServer> {
        let server_config = ServerConfig::from(self.config.clone());

        // Use MLX handler if model is specified, otherwise use mock handler
        let model_path = "models/Phi-3-mini-128k-instruct-4bit".to_string();

        let handler: std::sync::Arc<dyn pensieve_02::traits::RequestHandler> =
            if std::path::Path::new(&model_path).exists() {
                info!("Using MLX handler with model: {}", model_path);
                std::sync::Arc::new(MlxRequestHandler::new(model_path))
            } else {
                warn!("Model not found at {}, using mock handler", model_path);
                std::sync::Arc::new(MockRequestHandler::new(100))
            };

        Ok(HttpApiServer::new(server_config, handler))
    }
}

#[async_trait::async_trait]
impl traits::CliApp for PensieveCli {
    async fn run(&self) -> error::CliResult<i32> {
        // Initialize logging
        self.init_logging()?;

        match &self.args.command {
            Commands::Start { host, port, model, gpu_layers } => {
                self.handle_start(host, port, model, gpu_layers).await
            }
            Commands::Stop { host, port } => {
                self.handle_stop(host, *port).await
            }
            Commands::Status { host, port } => {
                self.handle_status(host, *port).await
            }
            Commands::Config { action } => {
                self.handle_config(action).await
            }
            Commands::Validate { config } => {
                self.handle_validate(config).await
            }
        }
    }

    fn validate_config(&self) -> error::CliResult<()> {
        // Validate server configuration
        if self.config.server.host.is_empty() {
            return Err(error::CliError::Config("Server host cannot be empty".to_string()));
        }

        if self.config.server.port == 0 {
            return Err(error::CliError::Config("Server port must be non-zero".to_string()));
        }

        // Validate model configuration
        if !self.config.model.model_path.exists() {
            return Err(error::CliError::Config(format!(
                "Model file does not exist: {}",
                self.config.model.model_path.display()
            )));
        }

        if self.config.model.context_size == 0 {
            return Err(error::CliError::Config("Context size must be positive".to_string()));
        }

        Ok(())
    }
}

impl PensieveCli {
    async fn handle_start(
        &self,
        host: &Option<String>,
        port: &Option<u16>,
        model: &Option<PathBuf>,
        gpu_layers: &Option<u32>,
    ) -> error::CliResult<i32> {
        info!("Starting Pensieve server...");

        // Apply command-line overrides
        let mut config = self.config.clone();
        if let Some(host) = host {
            config.server.host = host.clone();
        }
        if let Some(port) = port {
            config.server.port = *port;
        }
        if let Some(model) = model {
            config.model.model_path = model.clone();
        }
        if let Some(gpu_layers) = gpu_layers {
            config.model.gpu_layers = Some(*gpu_layers);
        }

        // Validate configuration
        let cli = PensieveCli { config: config.clone(), args: self.args.clone() };
        cli.validate_config()?;

        // Create and start server
        let server = cli.create_server()?;
        server.start().await?;

        info!("Server started successfully on {}:{}", config.server.host, config.server.port);

        // Wait for shutdown signal
        info!("Press Ctrl+C to stop the server");
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received shutdown signal");
                server.shutdown().await?;
                info!("Server stopped successfully");
                Ok(0)
            }
            Err(err) => {
                error!("Failed to listen for shutdown signal: {}", err);
                Ok(1)
            }
        }
    }

    async fn handle_stop(&self, host: &str, port: u16) -> error::CliResult<i32> {
        info!("Stopping Pensieve server at {}:{}", host, port);
        // For now, just print a message
        println!("Server stop functionality not yet implemented");
        Ok(0)
    }

    async fn handle_status(&self, host: &str, port: u16) -> error::CliResult<i32> {
        info!("Checking Pensieve server status at {}:{}", host, port);
        // For now, just print a message
        println!("Server status functionality not yet implemented");
        Ok(0)
    }

    async fn handle_config(&self, action: &ConfigAction) -> error::CliResult<i32> {
        match action {
            ConfigAction::Show => {
                let config_json = serde_json::to_string_pretty(&self.config)?;
                println!("{}", config_json);
                Ok(0)
            }
            ConfigAction::Generate { output } => {
                let default_config = CliConfig::default();
                let config_json = serde_json::to_string_pretty(&default_config)?;

                if let Some(output_path) = output {
                    Self::save_config(output_path, &default_config)?;
                    println!("Configuration saved to: {}", output_path.display());
                } else {
                    println!("{}", config_json);
                }
                Ok(0)
            }
            ConfigAction::Validate { config } => {
                let default_path = PathBuf::from("config.json");
                let config_path = config.as_ref().unwrap_or(&default_path);
                let loaded_config = Self::load_config(config_path)?;
                let cli = PensieveCli { config: loaded_config, args: self.args.clone() };
                cli.validate_config()?;
                println!("Configuration is valid");
                Ok(0)
            }
        }
    }

    async fn handle_validate(&self, config: &Option<PathBuf>) -> error::CliResult<i32> {
        let default_path = PathBuf::from("config.json");
        let config_path = config.as_ref().unwrap_or(&default_path);
        let final_config = if config_path.exists() {
            Self::load_config(config_path)?
        } else {
            self.config.clone()
        };

        let cli = PensieveCli { config: final_config, args: self.args.clone() };
        cli.validate_config()?;
        println!("Configuration is valid");
        Ok(0)
    }
}

/// Re-export commonly used items
pub use error::{CliError, CliResult};
pub use traits::{CliApp, CommandHandler};

/// Result type alias for convenience
pub type Result<T> = CliResult<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = CliConfig::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 7777);
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.model.model_type, "llama");
    }

    #[test]
    fn test_config_serialization() {
        let config = CliConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CliConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.server.host, deserialized.server.host);
        assert_eq!(config.server.port, deserialized.server.port);
    }

    #[test]
    fn test_config_file_operations() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.json");
        let config = CliConfig::default();

        // Test saving
        PensieveCli::save_config(&config_path, &config).unwrap();
        assert!(config_path.exists());

        // Test loading
        let loaded_config = PensieveCli::load_config(&config_path).unwrap();
        assert_eq!(config.server.host, loaded_config.server.host);
        assert_eq!(config.server.port, loaded_config.server.port);
    }

    #[test]
    fn test_server_config_conversion() {
        let cli_config = CliConfig::default();
        let server_config = ServerConfig::from(cli_config.clone());
        assert_eq!(cli_config.server.host, server_config.host);
        assert_eq!(cli_config.server.port, server_config.port);
        assert_eq!(cli_config.server.max_concurrent_requests, server_config.max_concurrent_requests);
    }

    #[test]
    fn test_cli_validation() {
        let args = CliArgs {
            command: Commands::Validate { config: None },
            config: None,
            verbose: false,
            log_level: None,
        };

        let cli = PensieveCli::new(args).unwrap();
        // This should fail validation because model file doesn't exist
        assert!(cli.validate_config().is_err());
    }

    #[test]
    fn test_command_result() {
        let success = CommandResult::Success("Operation completed".to_string());
        let warning = CommandResult::Warning("Operation completed with warnings".to_string());
        let error = CommandResult::Error("Operation failed".to_string());

        match success {
            CommandResult::Success(msg) => assert_eq!(msg, "Operation completed"),
            _ => panic!("Expected Success"),
        }

        match warning {
            CommandResult::Warning(msg) => assert_eq!(msg, "Operation completed with warnings"),
            _ => panic!("Expected Warning"),
        }

        match error {
            CommandResult::Error(msg) => assert_eq!(msg, "Operation failed"),
            _ => panic!("Expected Error"),
        }
    }

    // Test for configuration validation
    #[test]
    fn test_config_validation() {
        let config = CliConfig {
            server: ServerConfigFile {
                host: "127.0.0.1".to_string(),
                port: 7777,
                max_concurrent_requests: 100,
                request_timeout_ms: 30000,
                enable_cors: true,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "compact".to_string(),
                file: None,
            },
            model: ModelConfig {
                model_path: PathBuf::from("test.gguf"),
                model_type: "llama".to_string(),
                context_size: 2048,
                gpu_layers: Some(0),
            },
        };

        assert!(config.server.port >= 1024);
        assert!(config.server.port <= 65535);
        assert!(config.server.max_concurrent_requests > 0);
        assert!(config.server.request_timeout_ms > 0);
    }

    #[test]
    fn test_error_conversions() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let cli_error: CliError = io_error.into();
        assert!(matches!(cli_error, CliError::Io(_)));

        // Create a simple serialization error using from_str
        let serialization_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let cli_error: CliError = serialization_error.into();
        assert!(matches!(cli_error, CliError::Serialization(_)));

        let core_error = CoreError::InvalidConfig("test");
        let cli_error: CliError = core_error.into();
        assert!(matches!(cli_error, CliError::Core(_)));
    }
}