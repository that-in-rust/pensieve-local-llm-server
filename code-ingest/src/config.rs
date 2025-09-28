use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration structure for code-ingest CLI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default database path
    pub database: Option<DatabaseConfig>,
    
    /// Default ingestion settings
    pub ingestion: Option<IngestionConfig>,
    
    /// Default task generation settings
    pub task_generation: Option<TaskGenerationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Default database path
    pub default_path: Option<PathBuf>,
    
    /// Connection timeout in seconds
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Default folder flag setting
    pub default_folder_flag: Option<bool>,
    
    /// File size limit in bytes
    pub max_file_size: Option<usize>,
    
    /// File extensions to include
    pub include_extensions: Option<Vec<String>>,
    
    /// File extensions to exclude
    pub exclude_extensions: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGenerationConfig {
    /// Default number of hierarchy levels
    pub default_levels: Option<usize>,
    
    /// Default number of groups per level
    pub default_groups: Option<usize>,
    
    /// Default prompt file path
    pub default_prompt_file: Option<PathBuf>,
    
    /// Default chunk size for large files
    pub default_chunk_size: Option<usize>,
    
    /// Default output directory
    pub default_output_dir: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: Some(DatabaseConfig {
                default_path: None,
                timeout_seconds: Some(30),
            }),
            ingestion: Some(IngestionConfig {
                default_folder_flag: Some(false),
                max_file_size: Some(10 * 1024 * 1024), // 10MB
                include_extensions: None,
                exclude_extensions: Some(vec![
                    "bin".to_string(),
                    "exe".to_string(),
                    "dll".to_string(),
                    "so".to_string(),
                    "dylib".to_string(),
                ]),
            }),
            task_generation: Some(TaskGenerationConfig {
                default_levels: Some(4),
                default_groups: Some(7),
                default_prompt_file: Some(PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md")),
                default_chunk_size: None,
                default_output_dir: Some(PathBuf::from(".")),
            }),
        }
    }
}

/// Configuration manager for loading and saving configuration files
pub struct ConfigManager {
    config_path: PathBuf,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        let config_path = Self::default_config_path();
        Self { config_path }
    }
    
    /// Create a configuration manager with a custom path
    pub fn with_path(config_path: PathBuf) -> Self {
        Self { config_path }
    }
    
    /// Get the default configuration file path
    pub fn default_config_path() -> PathBuf {
        if let Some(home) = dirs::home_dir() {
            home.join(".config").join("code-ingest").join("config.toml")
        } else {
            PathBuf::from("code-ingest-config.toml")
        }
    }
    
    /// Load configuration from file, falling back to defaults if file doesn't exist
    pub fn load_config(&self) -> Result<Config> {
        if self.config_path.exists() {
            let content = std::fs::read_to_string(&self.config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            // Return default configuration if file doesn't exist
            Ok(Config::default())
        }
    }
    
    /// Save configuration to file
    pub fn save_config(&self, config: &Config) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let content = toml::to_string_pretty(config)?;
        std::fs::write(&self.config_path, content)?;
        Ok(())
    }
    
    /// Get the configuration file path
    pub fn config_path(&self) -> &PathBuf {
        &self.config_path
    }
    
    /// Check if configuration file exists
    pub fn config_exists(&self) -> bool {
        self.config_path.exists()
    }
    
    /// Create a default configuration file
    pub fn create_default_config(&self) -> Result<()> {
        let default_config = Config::default();
        self.save_config(&default_config)?;
        Ok(())
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge CLI arguments with configuration file values
/// CLI arguments take precedence over config file values
pub fn merge_config_with_cli(
    config: &Config,
    cli_db_path: Option<PathBuf>,
    cli_levels: Option<usize>,
    cli_groups: Option<usize>,
    cli_prompt_file: Option<PathBuf>,
    cli_chunks: Option<usize>,
) -> MergedConfig {
    MergedConfig {
        db_path: cli_db_path.or_else(|| {
            config.database.as_ref()?.default_path.clone()
        }),
        levels: cli_levels.or_else(|| {
            config.task_generation.as_ref()?.default_levels
        }).unwrap_or(4),
        groups: cli_groups.or_else(|| {
            config.task_generation.as_ref()?.default_groups
        }).unwrap_or(7),
        prompt_file: cli_prompt_file.or_else(|| {
            config.task_generation.as_ref()?.default_prompt_file.clone()
        }).unwrap_or_else(|| PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md")),
        chunks: cli_chunks.or_else(|| {
            config.task_generation.as_ref()?.default_chunk_size
        }),
        timeout_seconds: config.database.as_ref()
            .and_then(|db| db.timeout_seconds)
            .unwrap_or(30),
    }
}

/// Merged configuration combining CLI args and config file
#[derive(Debug, Clone)]
pub struct MergedConfig {
    pub db_path: Option<PathBuf>,
    pub levels: usize,
    pub groups: usize,
    pub prompt_file: PathBuf,
    pub chunks: Option<usize>,
    pub timeout_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.database.is_some());
        assert!(config.ingestion.is_some());
        assert!(config.task_generation.is_some());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml_str).unwrap();
        
        // Compare some key fields
        assert_eq!(
            config.database.as_ref().unwrap().timeout_seconds,
            deserialized.database.as_ref().unwrap().timeout_seconds
        );
    }

    #[test]
    fn test_config_manager() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test-config.toml");
        let manager = ConfigManager::with_path(config_path.clone());
        
        // Test loading non-existent config (should return defaults)
        let config = manager.load_config().unwrap();
        assert!(config.database.is_some());
        
        // Test saving and loading config
        manager.save_config(&config).unwrap();
        assert!(config_path.exists());
        
        let loaded_config = manager.load_config().unwrap();
        assert_eq!(
            config.database.as_ref().unwrap().timeout_seconds,
            loaded_config.database.as_ref().unwrap().timeout_seconds
        );
    }

    #[test]
    fn test_merge_config_with_cli() {
        let config = Config::default();
        
        // Test CLI precedence
        let merged = merge_config_with_cli(
            &config,
            Some(PathBuf::from("/custom/db")),
            Some(6),
            Some(10),
            Some(PathBuf::from("custom-prompt.md")),
            Some(1000),
        );
        
        assert_eq!(merged.db_path, Some(PathBuf::from("/custom/db")));
        assert_eq!(merged.levels, 6);
        assert_eq!(merged.groups, 10);
        assert_eq!(merged.prompt_file, PathBuf::from("custom-prompt.md"));
        assert_eq!(merged.chunks, Some(1000));
    }

    #[test]
    fn test_merge_config_fallback_to_defaults() {
        let config = Config::default();
        
        // Test fallback to config defaults
        let merged = merge_config_with_cli(
            &config,
            None, // No CLI db_path
            None, // No CLI levels
            None, // No CLI groups
            None, // No CLI prompt_file
            None, // No CLI chunks
        );
        
        assert_eq!(merged.levels, 4); // From config default
        assert_eq!(merged.groups, 7); // From config default
        assert_eq!(merged.timeout_seconds, 30); // From config default
    }
}