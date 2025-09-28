use crate::error::{CodeIngestError, CodeIngestResult};
use std::path::Path;
use colored::*;

/// User-friendly help and troubleshooting system
pub struct HelpSystem;

impl HelpSystem {
    /// Show comprehensive setup guide
    pub fn show_setup_guide() {
        println!("{}", "🚀 Code Ingest Setup Guide".bright_blue().bold());
        println!();
        
        println!("{}", "1. Install PostgreSQL".bright_green().bold());
        println!("   macOS:    brew install postgresql && brew services start postgresql");
        println!("   Ubuntu:   sudo apt-get install postgresql postgresql-contrib");
        println!("   Windows:  Download from https://www.postgresql.org/download/");
        println!();
        
        println!("{}", "2. Create Database".bright_green().bold());
        println!("   createdb code_analysis");
        println!("   # Or specify custom database in connection string");
        println!();
        
        println!("{}", "3. Set Environment (Optional)".bright_green().bold());
        println!("   export DATABASE_URL=postgresql://user:pass@localhost:5432/code_analysis");
        println!("   export GITHUB_TOKEN=your_github_token_here  # For private repos");
        println!();
        
        println!("{}", "4. Basic Usage".bright_green().bold());
        println!("   # Ingest a repository");
        println!("   code-ingest ingest https://github.com/user/repo --db-path ./analysis");
        println!();
        println!("   # Query the data");
        println!("   code-ingest sql \"SELECT * FROM INGEST_20250927143022 LIMIT 5\" --db-path ./analysis");
        println!();
        println!("   # List available tables");
        println!("   code-ingest list-tables --db-path ./analysis");
        println!();
        
        println!("{}", "5. Need Help?".bright_yellow().bold());
        println!("   code-ingest --help           # Show all commands");
        println!("   code-ingest <command> --help # Show command-specific help");
        println!("   code-ingest pg-start         # PostgreSQL troubleshooting");
        println!();
    }

    /// Show PostgreSQL troubleshooting guide
    pub fn show_postgresql_troubleshooting() {
        println!("{}", "🔧 PostgreSQL Troubleshooting Guide".bright_blue().bold());
        println!();
        
        println!("{}", "Common Issues & Solutions:".bright_green().bold());
        println!();
        
        println!("{}", "❌ Connection refused / PostgreSQL not running".red());
        println!("   macOS:    brew services start postgresql");
        println!("   Linux:    sudo systemctl start postgresql");
        println!("   Windows:  Start PostgreSQL service from Services panel");
        println!();
        
        println!("{}", "❌ Database does not exist".red());
        println!("   createdb your_database_name");
        println!("   # Or use default: createdb code_analysis");
        println!();
        
        println!("{}", "❌ Authentication failed".red());
        println!("   # Check your connection string format:");
        println!("   postgresql://username:password@host:port/database");
        println!("   # Default local connection:");
        println!("   postgresql://localhost:5432/code_analysis");
        println!();
        
        println!("{}", "❌ Permission denied".red());
        println!("   # Create user with proper permissions:");
        println!("   sudo -u postgres createuser --interactive your_username");
        println!("   # Or use existing postgres user:");
        println!("   postgresql://postgres@localhost:5432/code_analysis");
        println!();
        
        println!("{}", "✅ Test Connection".bright_green().bold());
        println!("   psql postgresql://localhost:5432/code_analysis");
        println!("   # Should connect without errors");
        println!();
        
        println!("{}", "✅ Verify Installation".bright_green().bold());
        println!("   code-ingest db-info --db-path ./analysis");
        println!("   # Should show connection status and database info");
        println!();
    }

    /// Show command examples for common tasks
    pub fn show_examples() {
        println!("{}", "📚 Code Ingest Examples".bright_blue().bold());
        println!();
        
        println!("{}", "🔍 Basic Repository Analysis".bright_green().bold());
        println!("# 1. Ingest a public repository");
        println!("code-ingest ingest https://github.com/rust-lang/cargo --db-path ./rust-analysis");
        println!();
        println!("# 2. Explore what was ingested");
        println!("code-ingest list-tables --db-path ./rust-analysis");
        println!("code-ingest sample --table INGEST_20250927143022 --limit 3 --db-path ./rust-analysis");
        println!();
        println!("# 3. Find specific code patterns");
        println!("code-ingest sql \"SELECT filepath, content_text FROM INGEST_20250927143022 WHERE content_text LIKE '%async%' LIMIT 10\" --db-path ./rust-analysis");
        println!();
        
        println!("{}", "🔐 Private Repository Access".bright_green().bold());
        println!("# Set GitHub token first");
        println!("export GITHUB_TOKEN=ghp_your_token_here");
        println!("code-ingest ingest https://github.com/company/private-repo --db-path ./private-analysis");
        println!();
        
        println!("{}", "📁 Local Folder Analysis".bright_green().bold());
        println!("# Analyze local codebase");
        println!("code-ingest ingest /path/to/your/project --db-path ./local-analysis");
        println!();
        
        println!("{}", "🤖 IDE Integration Workflow".bright_green().bold());
        println!("# 1. Prepare data for systematic analysis");
        println!("code-ingest query-prepare \"SELECT filepath, content_text FROM INGEST_20250927143022 WHERE extension = 'rs'\" \\");
        println!("  --db-path ./analysis \\");
        println!("  --temp-path ./rust-files.txt \\");
        println!("  --tasks-file ./rust-analysis-tasks.md \\");
        println!("  --output-table RUST_ANALYSIS");
        println!();
        println!("# 2. (Process tasks in your IDE)");
        println!();
        println!("# 3. Store results back to database");
        println!("code-ingest store-result \\");
        println!("  --db-path ./analysis \\");
        println!("  --output-table RUST_ANALYSIS \\");
        println!("  --result-file ./analysis-results.txt \\");
        println!("  --original-query \"SELECT filepath, content_text FROM INGEST_20250927143022 WHERE extension = 'rs'\"");
        println!();
        
        println!("{}", "📊 Database Management".bright_green().bold());
        println!("# View database status");
        println!("code-ingest db-info --db-path ./analysis");
        println!();
        println!("# Clean up old tables (keep 3 most recent)");
        println!("code-ingest cleanup-tables --keep 3 --db-path ./analysis");
        println!();
        println!("# Export results as markdown files");
        println!("code-ingest print-to-md \\");
        println!("  --db-path ./analysis \\");
        println!("  --table RUST_ANALYSIS \\");
        println!("  --sql \"SELECT * FROM RUST_ANALYSIS\" \\");
        println!("  --prefix analysis \\");
        println!("  --location ./markdown-exports/");
        println!();
    }

    /// Show troubleshooting guide for common user errors
    pub fn show_troubleshooting_guide() {
        println!("{}", "🩺 Troubleshooting Common Issues".bright_blue().bold());
        println!();
        
        println!("{}", "🚫 Command Not Found".red().bold());
        println!("   Problem: 'code-ingest: command not found'");
        println!("   Solution: Install the binary or run 'cargo run --' instead of 'code-ingest'");
        println!();
        
        println!("{}", "🚫 Repository Access Denied".red().bold());
        println!("   Problem: 'Repository not accessible' or '404 Not Found'");
        println!("   Solutions:");
        println!("   • Check repository URL spelling");
        println!("   • For private repos: Set GITHUB_TOKEN environment variable");
        println!("   • Verify you have access to the repository");
        println!();
        
        println!("{}", "🚫 Database Connection Issues".red().bold());
        println!("   Problem: 'PostgreSQL connection failed'");
        println!("   Solutions:");
        println!("   • Run 'code-ingest pg-start' for detailed PostgreSQL help");
        println!("   • Check if PostgreSQL is running: 'brew services list | grep postgresql'");
        println!("   • Verify database exists: 'psql -l'");
        println!();
        
        println!("{}", "🚫 File Processing Errors".red().bold());
        println!("   Problem: 'File processing failed' or 'Conversion failed'");
        println!("   Solutions:");
        println!("   • Install required tools: 'brew install pandoc poppler'");
        println!("   • Check file permissions: 'ls -la /path/to/file'");
        println!("   • Large files are skipped automatically (this is normal)");
        println!();
        
        println!("{}", "🚫 SQL Query Errors".red().bold());
        println!("   Problem: 'Query execution failed' or 'Syntax error'");
        println!("   Solutions:");
        println!("   • Check table name: 'code-ingest list-tables --db-path ./analysis'");
        println!("   • Verify SQL syntax: Use standard PostgreSQL syntax");
        println!("   • Quote table names with spaces: \"SELECT * FROM \\\"INGEST_20250927143022\\\"\"");
        println!();
        
        println!("{}", "🚫 Out of Disk Space".red().bold());
        println!("   Problem: 'No space left on device'");
        println!("   Solutions:");
        println!("   • Free up disk space or choose different location");
        println!("   • Clean up old tables: 'code-ingest cleanup-tables --keep 2'");
        println!("   • Use smaller repositories for testing");
        println!();
        
        println!("{}", "✅ Getting Help".bright_green().bold());
        println!("   • General help: 'code-ingest --help'");
        println!("   • Command help: 'code-ingest <command> --help'");
        println!("   • Setup guide: 'code-ingest pg-start'");
        println!("   • Examples: Run with --examples flag (if implemented)");
        println!();
    }

    /// Validate user inputs and provide helpful error messages
    pub fn validate_database_path(db_path: &Path) -> CodeIngestResult<()> {
        if !db_path.exists() {
            return Err(CodeIngestError::configuration_error(
                format!("Database path does not exist: {}", db_path.display())
            ));
        }

        if !db_path.is_dir() {
            return Err(CodeIngestError::configuration_error(
                format!("Database path must be a directory: {}", db_path.display())
            ));
        }

        // Check if directory is writable
        let test_file = db_path.join(".write_test");
        if let Err(_) = std::fs::write(&test_file, "test") {
            return Err(CodeIngestError::PermissionDenied {
                path: db_path.display().to_string(),
                suggestion: "Ensure you have write permissions to the database directory".to_string(),
            });
        }
        let _ = std::fs::remove_file(&test_file);

        Ok(())
    }

    /// Validate repository URL format
    pub fn validate_repository_url(url: &str) -> CodeIngestResult<()> {
        if url.starts_with('/') || url.starts_with('.') {
            // Local path - validate it exists
            let path = Path::new(url);
            if !path.exists() {
                return Err(CodeIngestError::configuration_error(
                    format!("Local path does not exist: {}", url)
                ));
            }
            if !path.is_dir() {
                return Err(CodeIngestError::configuration_error(
                    format!("Local path must be a directory: {}", url)
                ));
            }
            return Ok(());
        }

        // GitHub URL validation
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(CodeIngestError::configuration_error(
                format!("Invalid repository URL: {}. Must be a GitHub URL (https://github.com/...) or local path", url)
            ));
        }

        if !url.contains("github.com") {
            return Err(CodeIngestError::configuration_error(
                format!("Only GitHub repositories are supported: {}", url)
            ));
        }

        // Basic GitHub URL format check
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() < 5 {
            return Err(CodeIngestError::configuration_error(
                format!("Invalid GitHub URL format: {}. Expected: https://github.com/owner/repo", url)
            ));
        }

        Ok(())
    }

    /// Validate SQL query for common issues
    pub fn validate_sql_query(query: &str) -> CodeIngestResult<()> {
        let query_lower = query.to_lowercase();
        
        // Check for dangerous operations
        if query_lower.contains("drop ") || query_lower.contains("delete ") || query_lower.contains("truncate ") {
            return Err(CodeIngestError::Validation {
                field: "SQL query".to_string(),
                message: "Destructive operations (DROP, DELETE, TRUNCATE) are not allowed".to_string(),
                suggestion: "Use SELECT queries only for data analysis".to_string(),
            });
        }

        // Check for empty query
        if query.trim().is_empty() {
            return Err(CodeIngestError::Validation {
                field: "SQL query".to_string(),
                message: "Query cannot be empty".to_string(),
                suggestion: "Provide a valid SELECT query, e.g., 'SELECT * FROM table_name LIMIT 10'".to_string(),
            });
        }

        // Suggest LIMIT for potentially large queries
        if query_lower.contains("select") && !query_lower.contains("limit") && !query_lower.contains("count(") {
            println!("{}", "💡 Tip: Consider adding LIMIT to your query to avoid large result sets".yellow());
        }

        Ok(())
    }

    /// Validate table name format
    pub fn validate_table_name(table_name: &str) -> CodeIngestResult<()> {
        if table_name.trim().is_empty() {
            return Err(CodeIngestError::Validation {
                field: "table name".to_string(),
                message: "Table name cannot be empty".to_string(),
                suggestion: "Use 'code-ingest list-tables' to see available tables".to_string(),
            });
        }

        // Check for SQL injection patterns
        if table_name.contains(';') || table_name.contains('\'') || table_name.contains('"') {
            return Err(CodeIngestError::Validation {
                field: "table name".to_string(),
                message: "Table name contains invalid characters".to_string(),
                suggestion: "Table names should only contain letters, numbers, and underscores".to_string(),
            });
        }

        Ok(())
    }

    /// Provide context-aware suggestions based on error type
    pub fn suggest_next_steps(error: &CodeIngestError) -> Vec<String> {
        match error {
            CodeIngestError::Git { .. } => vec![
                "Check your internet connection".to_string(),
                "Verify the repository URL is correct".to_string(),
                "For private repos, set GITHUB_TOKEN environment variable".to_string(),
                "Try with a public repository first to test your setup".to_string(),
            ],
            CodeIngestError::Database { .. } => vec![
                "Run 'code-ingest pg-start' for PostgreSQL setup help".to_string(),
                "Check if PostgreSQL is running: 'brew services list | grep postgresql'".to_string(),
                "Verify database exists: 'psql -l'".to_string(),
                "Test connection: 'code-ingest db-info --db-path ./analysis'".to_string(),
            ],
            CodeIngestError::FileSystem { .. } => vec![
                "Check file and directory permissions".to_string(),
                "Ensure you have enough disk space".to_string(),
                "Verify the path exists and is accessible".to_string(),
                "Try with a different location if permissions are restricted".to_string(),
            ],
            CodeIngestError::Configuration { .. } => vec![
                "Check all command line arguments are correct".to_string(),
                "Verify file paths are absolute and exist".to_string(),
                "Review the command help: 'code-ingest <command> --help'".to_string(),
                "See examples: 'code-ingest --help'".to_string(),
            ],
            _ => vec![
                "Check the error message for specific guidance".to_string(),
                "Run 'code-ingest --help' for general usage".to_string(),
                "Try the troubleshooting guide".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validate_database_path_success() {
        let temp_dir = TempDir::new().unwrap();
        let result = HelpSystem::validate_database_path(temp_dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_database_path_not_exists() {
        let result = HelpSystem::validate_database_path(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn test_validate_repository_url_github_success() {
        let result = HelpSystem::validate_repository_url("https://github.com/user/repo");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_repository_url_local_path_success() {
        let temp_dir = TempDir::new().unwrap();
        let result = HelpSystem::validate_repository_url(&temp_dir.path().display().to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_repository_url_invalid_format() {
        let result = HelpSystem::validate_repository_url("not-a-url");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid repository URL"));
    }

    #[test]
    fn test_validate_repository_url_non_github() {
        let result = HelpSystem::validate_repository_url("https://gitlab.com/user/repo");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Only GitHub repositories"));
    }

    #[test]
    fn test_validate_sql_query_success() {
        let result = HelpSystem::validate_sql_query("SELECT * FROM table LIMIT 10");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_sql_query_dangerous_operations() {
        let result = HelpSystem::validate_sql_query("DROP TABLE users");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Destructive operations"));

        let result = HelpSystem::validate_sql_query("DELETE FROM users");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_sql_query_empty() {
        let result = HelpSystem::validate_sql_query("");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_validate_table_name_success() {
        let result = HelpSystem::validate_table_name("INGEST_20250927143022");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_table_name_invalid_characters() {
        let result = HelpSystem::validate_table_name("table'; DROP TABLE users; --");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("invalid characters"));
    }

    #[test]
    fn test_validate_table_name_empty() {
        let result = HelpSystem::validate_table_name("");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_suggest_next_steps() {
        let git_error = CodeIngestError::git_error("test error", None);
        let suggestions = HelpSystem::suggest_next_steps(&git_error);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("internet connection")));

        let db_error = CodeIngestError::database_error("test error", None);
        let suggestions = HelpSystem::suggest_next_steps(&db_error);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("pg-start")));
    }

    #[test]
    fn test_error_message_clarity() {
        // Test that error messages are clear and actionable
        let git_auth_error = CodeIngestError::git_error("authentication failed", None);
        let message = git_auth_error.to_string();
        assert!(message.contains("💡 Suggestion:"));
        assert!(message.contains("GitHub token"));

        let db_connection_error = CodeIngestError::database_error("connection refused", None);
        let message = db_connection_error.to_string();
        assert!(message.contains("💡 Suggestion:"));
        assert!(message.contains("PostgreSQL"));

        let file_permission_error = CodeIngestError::PermissionDenied {
            path: "/test/path".to_string(),
            suggestion: "Check permissions".to_string(),
        };
        let message = file_permission_error.to_string();
        assert!(message.contains("💡 Suggestion:"));
        assert!(message.contains("permissions"));
    }

    #[test]
    fn test_validation_provides_helpful_suggestions() {
        // Test SQL validation provides helpful suggestions
        let result = HelpSystem::validate_sql_query("DROP TABLE users");
        assert!(result.is_err());
        let error = result.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("SELECT queries only"));

        // Test table name validation provides helpful suggestions
        let result = HelpSystem::validate_table_name("'; DROP TABLE users; --");
        assert!(result.is_err());
        let error = result.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("letters, numbers, and underscores"));

        // Test repository URL validation provides helpful suggestions
        let result = HelpSystem::validate_repository_url("not-a-url");
        assert!(result.is_err());
        let error = result.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("GitHub URL") || message.contains("local path"));
    }

    #[test]
    fn test_contextual_error_suggestions() {
        // Test that different error types get different contextual suggestions
        let git_error = CodeIngestError::git_error("network timeout", None);
        let git_suggestions = HelpSystem::suggest_next_steps(&git_error);
        assert!(git_suggestions.iter().any(|s| s.contains("internet connection")));
        assert!(git_suggestions.iter().any(|s| s.contains("GITHUB_TOKEN")));

        let fs_error = CodeIngestError::file_system_error("permission denied", None);
        let fs_suggestions = HelpSystem::suggest_next_steps(&fs_error);
        assert!(fs_suggestions.iter().any(|s| s.contains("permissions")));
        assert!(fs_suggestions.iter().any(|s| s.contains("disk space")));

        let config_error = CodeIngestError::configuration_error("invalid path");
        let config_suggestions = HelpSystem::suggest_next_steps(&config_error);
        assert!(config_suggestions.iter().any(|s| s.contains("arguments")));
        assert!(config_suggestions.iter().any(|s| s.contains("help")));
    }

    #[test]
    fn test_user_input_validation_edge_cases() {
        // Test empty inputs
        assert!(HelpSystem::validate_sql_query("").is_err());
        assert!(HelpSystem::validate_sql_query("   ").is_err());
        assert!(HelpSystem::validate_table_name("").is_err());
        assert!(HelpSystem::validate_table_name("   ").is_err());

        // Test SQL injection attempts
        assert!(HelpSystem::validate_table_name("users'; DROP TABLE passwords; --").is_err());
        assert!(HelpSystem::validate_sql_query("SELECT * FROM users; DROP TABLE passwords; --").is_err());

        // Test valid edge cases
        assert!(HelpSystem::validate_sql_query("SELECT COUNT(*) FROM table").is_ok());
        assert!(HelpSystem::validate_table_name("INGEST_20250927143022").is_ok());
        assert!(HelpSystem::validate_table_name("table_123").is_ok());
    }

    #[test]
    fn test_repository_url_validation_comprehensive() {
        // Valid GitHub URLs
        assert!(HelpSystem::validate_repository_url("https://github.com/user/repo").is_ok());
        assert!(HelpSystem::validate_repository_url("https://github.com/org/repo-name").is_ok());
        assert!(HelpSystem::validate_repository_url("https://github.com/user/repo.git").is_ok());

        // Invalid GitHub URLs
        assert!(HelpSystem::validate_repository_url("https://github.com/user").is_err());
        assert!(HelpSystem::validate_repository_url("https://github.com/").is_err());
        assert!(HelpSystem::validate_repository_url("github.com/user/repo").is_err());

        // Non-GitHub URLs
        assert!(HelpSystem::validate_repository_url("https://gitlab.com/user/repo").is_err());
        assert!(HelpSystem::validate_repository_url("https://bitbucket.org/user/repo").is_err());

        // Local paths (should work if they exist)
        let temp_dir = TempDir::new().unwrap();
        assert!(HelpSystem::validate_repository_url(&temp_dir.path().display().to_string()).is_ok());
        assert!(HelpSystem::validate_repository_url("/nonexistent/path").is_err());
    }
}