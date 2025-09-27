//! PostgreSQL setup and configuration guidance
//!
//! This module provides comprehensive setup instructions, connectivity testing,
//! and troubleshooting guidance for PostgreSQL database setup.

use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use std::env;
use std::process::Command;
use tracing::debug;

/// PostgreSQL setup manager
pub struct PostgreSQLSetup;

/// System information for setup guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub shell: String,
    pub has_homebrew: bool,
    pub has_apt: bool,
    pub has_yum: bool,
    pub has_psql: bool,
    pub postgresql_running: bool,
}

/// Database connection test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionTest {
    pub success: bool,
    pub database_url: Option<String>,
    pub server_version: Option<String>,
    pub database_name: Option<String>,
    pub connection_time_ms: Option<u64>,
    pub error_message: Option<String>,
    pub suggestions: Vec<String>,
}

/// Setup step with instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupStep {
    pub step_number: usize,
    pub title: String,
    pub description: String,
    pub commands: Vec<String>,
    pub verification: Option<String>,
    pub troubleshooting: Vec<String>,
}

impl PostgreSQLSetup {
    /// Create a new PostgreSQL setup manager
    pub fn new() -> Self {
        Self
    }

    /// Get system information for tailored setup instructions
    pub async fn get_system_info(&self) -> SystemInfo {
        debug!("Gathering system information for PostgreSQL setup");

        let os = env::consts::OS.to_string();
        let arch = env::consts::ARCH.to_string();
        let shell = env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());

        // Check for package managers
        let has_homebrew = self.command_exists("brew").await;
        let has_apt = self.command_exists("apt").await || self.command_exists("apt-get").await;
        let has_yum = self.command_exists("yum").await || self.command_exists("dnf").await;

        // Check for PostgreSQL tools
        let has_psql = self.command_exists("psql").await;
        let postgresql_running = self.check_postgresql_running().await;

        SystemInfo {
            os,
            arch,
            shell,
            has_homebrew,
            has_apt,
            has_yum,
            has_psql,
            postgresql_running,
        }
    }

    /// Generate platform-specific setup instructions
    pub async fn generate_setup_instructions(&self) -> Vec<SetupStep> {
        let system_info = self.get_system_info().await;
        let mut steps = Vec::new();

        // Step 1: Install PostgreSQL
        steps.push(self.create_installation_step(&system_info));

        // Step 2: Start PostgreSQL service
        steps.push(self.create_service_start_step(&system_info));

        // Step 3: Create database and user
        steps.push(self.create_database_setup_step(&system_info));

        // Step 4: Configure environment
        steps.push(self.create_environment_setup_step(&system_info));

        // Step 5: Test connection
        steps.push(self.create_connection_test_step(&system_info));

        steps
    }

    /// Test database connectivity
    pub async fn test_connection(&self, database_url: Option<&str>) -> ConnectionTest {
        let start_time = std::time::Instant::now();
        
        // Determine database URL to test
        let test_url = if let Some(url) = database_url {
            url.to_string()
        } else if let Ok(url) = env::var("DATABASE_URL") {
            url
        } else {
            // Try default local connection
            "postgresql://postgres:postgres@localhost:5432/postgres".to_string()
        };

        debug!("Testing PostgreSQL connection: {}", test_url);

        match PgPool::connect(&test_url).await {
            Ok(pool) => {
                // Test basic query
                match sqlx::query("SELECT version()").fetch_one(&pool).await {
                    Ok(row) => {
                        let version: String = row.get(0);
                        let connection_time_ms = start_time.elapsed().as_millis() as u64;
                        
                        // Get database name
                        let db_name = match sqlx::query("SELECT current_database()").fetch_one(&pool).await {
                            Ok(row) => Some(row.get::<String, _>(0)),
                            Err(_) => None,
                        };

                        ConnectionTest {
                            success: true,
                            database_url: Some(test_url),
                            server_version: Some(version),
                            database_name: db_name,
                            connection_time_ms: Some(connection_time_ms),
                            error_message: None,
                            suggestions: vec!["✅ Connection successful! PostgreSQL is ready to use.".to_string()],
                        }
                    }
                    Err(e) => {
                        ConnectionTest {
                            success: false,
                            database_url: Some(test_url),
                            server_version: None,
                            database_name: None,
                            connection_time_ms: None,
                            error_message: Some(format!("Query failed: {}", e)),
                            suggestions: vec![
                                "Database connection established but query failed".to_string(),
                                "Check database permissions and user access".to_string(),
                            ],
                        }
                    }
                }
            }
            Err(e) => {
                let error_msg = e.to_string();
                let suggestions = self.generate_connection_suggestions(&error_msg);
                
                ConnectionTest {
                    success: false,
                    database_url: Some(test_url),
                    server_version: None,
                    database_name: None,
                    connection_time_ms: None,
                    error_message: Some(error_msg),
                    suggestions,
                }
            }
        }
    }

    /// Format setup instructions for display
    pub fn format_setup_instructions(&self, steps: &[SetupStep], system_info: &SystemInfo) -> String {
        let mut output = String::new();
        
        output.push_str("PostgreSQL Setup Guide\n");
        output.push_str("======================\n\n");
        
        // System information
        output.push_str("🖥️  System Information:\n");
        output.push_str(&format!("   Operating System: {} ({})\n", system_info.os, system_info.arch));
        output.push_str(&format!("   Shell: {}\n", system_info.shell));
        output.push_str(&format!("   PostgreSQL installed: {}\n", if system_info.has_psql { "✅ Yes" } else { "❌ No" }));
        output.push_str(&format!("   PostgreSQL running: {}\n", if system_info.postgresql_running { "✅ Yes" } else { "❌ No" }));
        output.push_str("\n");

        // Package managers available
        output.push_str("📦 Available Package Managers:\n");
        if system_info.has_homebrew {
            output.push_str("   ✅ Homebrew (recommended for macOS)\n");
        }
        if system_info.has_apt {
            output.push_str("   ✅ APT (Ubuntu/Debian)\n");
        }
        if system_info.has_yum {
            output.push_str("   ✅ YUM/DNF (RHEL/CentOS/Fedora)\n");
        }
        if !system_info.has_homebrew && !system_info.has_apt && !system_info.has_yum {
            output.push_str("   ⚠️  No common package managers detected\n");
            output.push_str("   Please install PostgreSQL manually from https://postgresql.org/download/\n");
        }
        output.push_str("\n");

        // Setup steps
        for step in steps {
            output.push_str(&format!("## Step {}: {}\n\n", step.step_number, step.title));
            output.push_str(&format!("{}\n\n", step.description));
            
            if !step.commands.is_empty() {
                output.push_str("**Commands:**\n");
                for command in &step.commands {
                    output.push_str(&format!("```bash\n{}\n```\n\n", command));
                }
            }
            
            if let Some(verification) = &step.verification {
                output.push_str("**Verification:**\n");
                output.push_str(&format!("```bash\n{}\n```\n\n", verification));
            }
            
            if !step.troubleshooting.is_empty() {
                output.push_str("**Troubleshooting:**\n");
                for tip in &step.troubleshooting {
                    output.push_str(&format!("- {}\n", tip));
                }
                output.push_str("\n");
            }
        }

        // Quick reference
        output.push_str("## Quick Reference\n\n");
        output.push_str("**Common Commands:**\n");
        output.push_str("```bash\n");
        output.push_str("# Start PostgreSQL\n");
        if system_info.has_homebrew {
            output.push_str("brew services start postgresql\n");
        } else if system_info.has_apt {
            output.push_str("sudo systemctl start postgresql\n");
        }
        output.push_str("\n# Connect to PostgreSQL\n");
        output.push_str("psql -U postgres -d postgres\n");
        output.push_str("\n# Create database for code-ingest\n");
        output.push_str("createdb code_ingest\n");
        output.push_str("\n# Test connection\n");
        output.push_str("code-ingest db-info --db-path /path/to/db\n");
        output.push_str("```\n\n");

        // Environment variables
        output.push_str("**Environment Variables:**\n");
        output.push_str("```bash\n");
        output.push_str("# Set DATABASE_URL (replace with your credentials)\n");
        output.push_str("export DATABASE_URL=\"postgresql://username:password@localhost:5432/code_ingest\"\n");
        output.push_str("\n# Or use default local connection\n");
        output.push_str("export DATABASE_URL=\"postgresql://postgres:postgres@localhost:5432/code_ingest\"\n");
        output.push_str("```\n\n");

        output.push_str("💡 **Pro Tips:**\n");
        output.push_str("- Use `code-ingest db-info` to test your connection\n");
        output.push_str("- The DATABASE_URL can be set in your shell profile (~/.bashrc, ~/.zshrc)\n");
        output.push_str("- For development, the default PostgreSQL user is usually 'postgres'\n");
        output.push_str("- Use strong passwords for production environments\n");

        output
    }

    /// Format connection test results for display
    pub fn format_connection_test(&self, test: &ConnectionTest) -> String {
        let mut output = String::new();
        
        output.push_str("PostgreSQL Connection Test\n");
        output.push_str("==========================\n\n");
        
        if test.success {
            output.push_str("✅ **Connection Successful!**\n\n");
            
            if let Some(url) = &test.database_url {
                output.push_str(&format!("**Database URL**: `{}`\n", url));
            }
            
            if let Some(version) = &test.server_version {
                output.push_str(&format!("**Server Version**: {}\n", version));
            }
            
            if let Some(db_name) = &test.database_name {
                output.push_str(&format!("**Database Name**: {}\n", db_name));
            }
            
            if let Some(time) = test.connection_time_ms {
                output.push_str(&format!("**Connection Time**: {}ms\n", time));
            }
            
            output.push_str("\n🎉 **PostgreSQL is ready for code-ingest!**\n\n");
            output.push_str("**Next Steps:**\n");
            output.push_str("1. Try ingesting a repository: `code-ingest <repo_url> --db-path <path>`\n");
            output.push_str("2. Explore your data: `code-ingest list-tables --db-path <path>`\n");
            output.push_str("3. Run queries: `code-ingest sql \"SELECT COUNT(*) FROM ingestion_meta\" --db-path <path>`\n");
        } else {
            output.push_str("❌ **Connection Failed**\n\n");
            
            if let Some(url) = &test.database_url {
                output.push_str(&format!("**Attempted URL**: `{}`\n", url));
            }
            
            if let Some(error) = &test.error_message {
                output.push_str(&format!("**Error**: {}\n", error));
            }
            
            output.push_str("\n**Suggestions:**\n");
            for suggestion in &test.suggestions {
                output.push_str(&format!("- {}\n", suggestion));
            }
        }
        
        output
    }

    // Private helper methods

    async fn command_exists(&self, command: &str) -> bool {
        Command::new("which")
            .arg(command)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    async fn check_postgresql_running(&self) -> bool {
        // Try to connect to default PostgreSQL port
        std::net::TcpStream::connect("127.0.0.1:5432").is_ok()
    }

    fn create_installation_step(&self, system_info: &SystemInfo) -> SetupStep {
        let mut commands = Vec::new();
        let mut troubleshooting = Vec::new();

        if system_info.has_psql {
            return SetupStep {
                step_number: 1,
                title: "PostgreSQL Installation".to_string(),
                description: "✅ PostgreSQL is already installed on your system.".to_string(),
                commands: vec![],
                verification: Some("psql --version".to_string()),
                troubleshooting: vec![],
            };
        }

        let description = "Install PostgreSQL database server on your system.".to_string();

        if system_info.has_homebrew {
            commands.push("brew install postgresql".to_string());
            troubleshooting.push("If brew is slow, try: brew update && brew install postgresql".to_string());
        } else if system_info.has_apt {
            commands.push("sudo apt update".to_string());
            commands.push("sudo apt install postgresql postgresql-contrib".to_string());
            troubleshooting.push("If you get permission errors, make sure you have sudo access".to_string());
        } else if system_info.has_yum {
            commands.push("sudo yum install postgresql-server postgresql-contrib".to_string());
            commands.push("sudo postgresql-setup initdb".to_string());
            troubleshooting.push("For newer systems, try 'dnf' instead of 'yum'".to_string());
        } else {
            commands.push("# Download from https://postgresql.org/download/".to_string());
            commands.push("# Follow platform-specific installation instructions".to_string());
            troubleshooting.push("Visit https://postgresql.org/download/ for manual installation".to_string());
        }

        troubleshooting.push("Verify installation with: psql --version".to_string());

        SetupStep {
            step_number: 1,
            title: "Install PostgreSQL".to_string(),
            description,
            commands,
            verification: Some("psql --version".to_string()),
            troubleshooting,
        }
    }

    fn create_service_start_step(&self, system_info: &SystemInfo) -> SetupStep {
        let mut commands = Vec::new();
        let mut troubleshooting = Vec::new();

        let description = if system_info.postgresql_running {
            "✅ PostgreSQL service is already running.".to_string()
        } else {
            "Start the PostgreSQL database service.".to_string()
        };

        if !system_info.postgresql_running {
            if system_info.has_homebrew {
                commands.push("brew services start postgresql".to_string());
                troubleshooting.push("If service fails to start, try: brew services restart postgresql".to_string());
            } else if system_info.has_apt {
                commands.push("sudo systemctl start postgresql".to_string());
                commands.push("sudo systemctl enable postgresql".to_string());
                troubleshooting.push("Check service status: sudo systemctl status postgresql".to_string());
            } else if system_info.has_yum {
                commands.push("sudo systemctl start postgresql".to_string());
                commands.push("sudo systemctl enable postgresql".to_string());
                troubleshooting.push("For older systems, try: sudo service postgresql start".to_string());
            }
        }

        troubleshooting.push("Check if PostgreSQL is running: pg_isready".to_string());
        troubleshooting.push("View PostgreSQL logs if there are issues".to_string());

        SetupStep {
            step_number: 2,
            title: "Start PostgreSQL Service".to_string(),
            description,
            commands,
            verification: Some("pg_isready".to_string()),
            troubleshooting,
        }
    }

    fn create_database_setup_step(&self, _system_info: &SystemInfo) -> SetupStep {
        let commands = vec![
            "# Connect as postgres user".to_string(),
            "sudo -u postgres psql".to_string(),
            "".to_string(),
            "# Or create a database directly".to_string(),
            "createdb code_ingest".to_string(),
            "".to_string(),
            "# Create user (optional)".to_string(),
            "sudo -u postgres createuser --interactive".to_string(),
        ];

        let troubleshooting = vec![
            "If 'postgres' user doesn't exist, try connecting as your system user".to_string(),
            "On macOS with Homebrew, you might not need 'sudo -u postgres'".to_string(),
            "Default database is usually named 'postgres'".to_string(),
            "You can use any database name, just update your DATABASE_URL accordingly".to_string(),
        ];

        SetupStep {
            step_number: 3,
            title: "Create Database".to_string(),
            description: "Create a database for code-ingest and optionally set up a user.".to_string(),
            commands,
            verification: Some("psql -l".to_string()),
            troubleshooting,
        }
    }

    fn create_environment_setup_step(&self, system_info: &SystemInfo) -> SetupStep {
        let shell_file = if system_info.shell.contains("zsh") {
            "~/.zshrc"
        } else if system_info.shell.contains("bash") {
            "~/.bashrc"
        } else {
            "~/.profile"
        };

        let commands = vec![
            format!("# Add to {} (replace with your credentials)", shell_file),
            "export DATABASE_URL=\"postgresql://postgres:postgres@localhost:5432/code_ingest\"".to_string(),
            "".to_string(),
            "# Reload your shell configuration".to_string(),
            format!("source {}", shell_file),
            "".to_string(),
            "# Or set for current session only".to_string(),
            "export DATABASE_URL=\"postgresql://postgres:postgres@localhost:5432/code_ingest\"".to_string(),
        ];

        let troubleshooting = vec![
            "Replace 'postgres:postgres' with your actual username:password".to_string(),
            "Replace 'code_ingest' with your actual database name".to_string(),
            "Use 'localhost' for local connections, or your server IP for remote".to_string(),
            "Default PostgreSQL port is 5432".to_string(),
            "Test the URL format: postgresql://user:pass@host:port/database".to_string(),
        ];

        SetupStep {
            step_number: 4,
            title: "Configure Environment".to_string(),
            description: "Set up the DATABASE_URL environment variable for code-ingest.".to_string(),
            commands,
            verification: Some("echo $DATABASE_URL".to_string()),
            troubleshooting,
        }
    }

    fn create_connection_test_step(&self, _system_info: &SystemInfo) -> SetupStep {
        let commands = vec![
            "# Test connection with code-ingest".to_string(),
            "code-ingest db-info".to_string(),
            "".to_string(),
            "# Or test with psql directly".to_string(),
            "psql $DATABASE_URL".to_string(),
            "".to_string(),
            "# Test with specific database path".to_string(),
            "code-ingest db-info --db-path /path/to/your/database".to_string(),
        ];

        let troubleshooting = vec![
            "If connection fails, check your DATABASE_URL format".to_string(),
            "Ensure PostgreSQL service is running: pg_isready".to_string(),
            "Check firewall settings if connecting to remote database".to_string(),
            "Verify database exists: psql -l".to_string(),
            "Check user permissions if you get authentication errors".to_string(),
        ];

        SetupStep {
            step_number: 5,
            title: "Test Connection".to_string(),
            description: "Verify that code-ingest can connect to your PostgreSQL database.".to_string(),
            commands,
            verification: Some("code-ingest db-info".to_string()),
            troubleshooting,
        }
    }

    fn generate_connection_suggestions(&self, error_msg: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        let error_lower = error_msg.to_lowercase();
        
        if error_lower.contains("connection refused") {
            suggestions.push("PostgreSQL server is not running. Start it with your system's service manager.".to_string());
            suggestions.push("Check if PostgreSQL is listening on the correct port (default: 5432).".to_string());
        }
        
        if error_lower.contains("authentication failed") {
            suggestions.push("Check your username and password in the DATABASE_URL.".to_string());
            suggestions.push("Verify the user exists and has proper permissions.".to_string());
        }
        
        if error_lower.contains("database") && error_lower.contains("does not exist") {
            suggestions.push("Create the database first: createdb <database_name>".to_string());
            suggestions.push("Check the database name in your DATABASE_URL.".to_string());
        }
        
        if error_lower.contains("timeout") {
            suggestions.push("Check network connectivity to the PostgreSQL server.".to_string());
            suggestions.push("Verify firewall settings are not blocking the connection.".to_string());
        }
        
        if error_lower.contains("invalid") && error_lower.contains("url") {
            suggestions.push("Check DATABASE_URL format: postgresql://user:pass@host:port/database".to_string());
            suggestions.push("Ensure special characters in password are URL-encoded.".to_string());
        }
        
        // General suggestions
        suggestions.push("Run 'code-ingest pg-start' for detailed setup instructions.".to_string());
        suggestions.push("Check PostgreSQL logs for more detailed error information.".to_string());
        
        suggestions
    }
}

impl Default for PostgreSQLSetup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_postgresql_setup_creation() {
        let setup = PostgreSQLSetup::new();
        // Just test that we can create the setup manager
        assert!(true);
    }

    #[tokio::test]
    async fn test_system_info_gathering() {
        let setup = PostgreSQLSetup::new();
        let system_info = setup.get_system_info().await;
        
        // Basic system info should be available
        assert!(!system_info.os.is_empty());
        assert!(!system_info.arch.is_empty());
        
        // At least one of these should be true on most systems
        assert!(
            system_info.has_homebrew || 
            system_info.has_apt || 
            system_info.has_yum ||
            system_info.os == "windows"
        );
    }

    #[tokio::test]
    async fn test_setup_instructions_generation() {
        let setup = PostgreSQLSetup::new();
        let instructions = setup.generate_setup_instructions().await;
        
        // Should have all 5 steps
        assert_eq!(instructions.len(), 5);
        
        // Verify step order and titles
        assert_eq!(instructions[0].step_number, 1);
        assert!(instructions[0].title.contains("Install"));
        
        assert_eq!(instructions[1].step_number, 2);
        assert!(instructions[1].title.contains("Start"));
        
        assert_eq!(instructions[4].step_number, 5);
        assert!(instructions[4].title.contains("Test"));
    }

    #[tokio::test]
    async fn test_connection_test_with_invalid_url() {
        let setup = PostgreSQLSetup::new();
        
        // Test with obviously invalid URL
        let test = setup.test_connection(Some("invalid://url")).await;
        
        assert!(!test.success);
        assert!(test.error_message.is_some());
        assert!(!test.suggestions.is_empty());
    }

    #[test]
    fn test_connection_suggestions() {
        let setup = PostgreSQLSetup::new();
        
        // Test connection refused error
        let suggestions = setup.generate_connection_suggestions("connection refused");
        assert!(suggestions.iter().any(|s| s.contains("not running")));
        
        // Test authentication error
        let suggestions = setup.generate_connection_suggestions("authentication failed");
        assert!(suggestions.iter().any(|s| s.contains("username and password")));
        
        // Test database not found error
        let suggestions = setup.generate_connection_suggestions("database does not exist");
        assert!(suggestions.iter().any(|s| s.contains("createdb")));
    }

    #[test]
    fn test_formatting_functions() {
        let setup = PostgreSQLSetup::new();
        
        // Test system info formatting
        let system_info = SystemInfo {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            shell: "/bin/bash".to_string(),
            has_homebrew: false,
            has_apt: true,
            has_yum: false,
            has_psql: true,
            postgresql_running: true,
        };
        
        let steps = vec![
            SetupStep {
                step_number: 1,
                title: "Test Step".to_string(),
                description: "Test description".to_string(),
                commands: vec!["echo test".to_string()],
                verification: Some("test command".to_string()),
                troubleshooting: vec!["Test troubleshooting".to_string()],
            }
        ];
        
        let formatted = setup.format_setup_instructions(&steps, &system_info);
        
        assert!(formatted.contains("PostgreSQL Setup Guide"));
        assert!(formatted.contains("linux"));
        assert!(formatted.contains("✅ Yes")); // PostgreSQL installed
        assert!(formatted.contains("Test Step"));
        assert!(formatted.contains("echo test"));
    }

    #[test]
    fn test_connection_test_formatting() {
        let setup = PostgreSQLSetup::new();
        
        // Test successful connection
        let success_test = ConnectionTest {
            success: true,
            database_url: Some("postgresql://localhost:5432/test".to_string()),
            server_version: Some("PostgreSQL 14.0".to_string()),
            database_name: Some("test".to_string()),
            connection_time_ms: Some(50),
            error_message: None,
            suggestions: vec!["Connection successful!".to_string()],
        };
        
        let formatted = setup.format_connection_test(&success_test);
        assert!(formatted.contains("✅ **Connection Successful!**"));
        assert!(formatted.contains("PostgreSQL 14.0"));
        assert!(formatted.contains("50ms"));
        
        // Test failed connection
        let failed_test = ConnectionTest {
            success: false,
            database_url: Some("postgresql://localhost:5432/test".to_string()),
            server_version: None,
            database_name: None,
            connection_time_ms: None,
            error_message: Some("Connection refused".to_string()),
            suggestions: vec!["Check if PostgreSQL is running".to_string()],
        };
        
        let formatted = setup.format_connection_test(&failed_test);
        assert!(formatted.contains("❌ **Connection Failed**"));
        assert!(formatted.contains("Connection refused"));
        assert!(formatted.contains("Check if PostgreSQL is running"));
    }
}