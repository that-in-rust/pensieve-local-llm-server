//! Database export functionality
//!
//! This module provides functionality to export query results as individual markdown files
//! with sequential naming and structured formatting.

use crate::error::{DatabaseError, DatabaseResult};
use crate::database::{DatabaseOperations, QueryResult};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Export manager for database query results
pub struct DatabaseExporter {
    operations: DatabaseOperations,
}

/// Configuration for export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Prefix for generated markdown files
    pub prefix: String,
    /// Directory location for exported files
    pub location: PathBuf,
    /// Maximum number of files to export (safety limit)
    pub max_files: Option<usize>,
    /// Whether to overwrite existing files
    pub overwrite_existing: bool,
    /// Template for markdown formatting
    pub markdown_template: Option<String>,
}

/// Result of an export operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Number of files successfully created
    pub files_created: usize,
    /// Number of files that failed to create
    pub files_failed: usize,
    /// Total size of exported files in bytes
    pub total_size_bytes: u64,
    /// Time taken for the export operation
    pub execution_time_ms: u64,
    /// List of created file paths
    pub created_files: Vec<PathBuf>,
    /// List of errors encountered
    pub errors: Vec<String>,
}

/// Information about an exported file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedFile {
    /// Path to the exported file
    pub file_path: PathBuf,
    /// Size of the file in bytes
    pub size_bytes: u64,
    /// Row index from the query result
    pub row_index: usize,
    /// Whether the export was successful
    pub success: bool,
    /// Error message if export failed
    pub error: Option<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            prefix: "export".to_string(),
            location: PathBuf::from("."),
            max_files: Some(1000), // Safety limit
            overwrite_existing: false,
            markdown_template: None,
        }
    }
}

impl DatabaseExporter {
    /// Create a new database exporter
    pub fn new(pool: PgPool) -> Self {
        let operations = DatabaseOperations::new(pool);
        Self { operations }
    }

    /// Export query results to individual markdown files
    pub async fn export_to_markdown_files(
        &self,
        table_name: &str,
        sql_query: &str,
        config: &ExportConfig,
    ) -> DatabaseResult<ExportResult> {
        let start_time = std::time::Instant::now();
        
        info!("Starting export to markdown files");
        debug!("Table: {}, Query: {}, Location: {}", table_name, sql_query, config.location.display());

        // Validate configuration
        self.validate_export_config(config).await?;

        // Execute the query
        let query_result = self.operations.execute_query(sql_query).await?;
        
        if query_result.rows.is_empty() {
            warn!("Query returned no results to export");
            return Ok(ExportResult {
                files_created: 0,
                files_failed: 0,
                total_size_bytes: 0,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                created_files: vec![],
                errors: vec!["No results to export".to_string()],
            });
        }

        // Check safety limits
        if let Some(max_files) = config.max_files {
            if query_result.rows.len() > max_files {
                return Err(DatabaseError::QueryFailed {
                    query: sql_query.to_string(),
                    cause: format!("Query returned {} rows, exceeding safety limit of {}", 
                                  query_result.rows.len(), max_files),
                });
            }
        }

        // Create output directory if it doesn't exist
        if !config.location.exists() {
            fs::create_dir_all(&config.location).await.map_err(|e| {
                DatabaseError::QueryFailed {
                    query: "create_directory".to_string(),
                    cause: format!("Failed to create output directory: {}", e),
                }
            })?;
        }

        // Export each row as a separate markdown file
        let mut export_results = Vec::new();
        let mut total_size_bytes = 0u64;
        let mut files_created = 0;
        let mut files_failed = 0;
        let mut created_files = Vec::new();
        let mut errors = Vec::new();

        for (index, row) in query_result.rows.iter().enumerate() {
            let file_number = index + 1;
            let filename = format!("{}-{:05}.md", config.prefix, file_number);
            let file_path = config.location.join(&filename);

            // Check if file exists and handle overwrite policy
            if file_path.exists() && !config.overwrite_existing {
                let error_msg = format!("File already exists: {}", file_path.display());
                errors.push(error_msg.clone());
                export_results.push(ExportedFile {
                    file_path: file_path.clone(),
                    size_bytes: 0,
                    row_index: index,
                    success: false,
                    error: Some(error_msg),
                });
                files_failed += 1;
                continue;
            }

            // Generate markdown content
            let markdown_content = self.generate_markdown_content(row, index, &query_result, config)?;
            
            // Write file
            match fs::write(&file_path, &markdown_content).await {
                Ok(()) => {
                    let size_bytes = markdown_content.len() as u64;
                    total_size_bytes += size_bytes;
                    files_created += 1;
                    created_files.push(file_path.clone());
                    
                    export_results.push(ExportedFile {
                        file_path,
                        size_bytes,
                        row_index: index,
                        success: true,
                        error: None,
                    });
                    
                    debug!("Created file: {} ({} bytes)", filename, size_bytes);
                }
                Err(e) => {
                    let error_msg = format!("Failed to write file {}: {}", filename, e);
                    errors.push(error_msg.clone());
                    export_results.push(ExportedFile {
                        file_path,
                        size_bytes: 0,
                        row_index: index,
                        success: false,
                        error: Some(error_msg),
                    });
                    files_failed += 1;
                }
            }
        }

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "Export completed: {} files created, {} failed, {} total bytes in {}ms",
            files_created, files_failed, total_size_bytes, execution_time_ms
        );

        Ok(ExportResult {
            files_created,
            files_failed,
            total_size_bytes,
            execution_time_ms,
            created_files,
            errors,
        })
    }

    /// Export from a specific table with a simple query
    pub async fn export_table_to_markdown(
        &self,
        table_name: &str,
        config: &ExportConfig,
    ) -> DatabaseResult<ExportResult> {
        let sql_query = format!("SELECT * FROM \"{}\"", table_name);
        self.export_to_markdown_files(table_name, &sql_query, config).await
    }

    /// Export with custom SQL query
    pub async fn export_query_to_markdown(
        &self,
        sql_query: &str,
        config: &ExportConfig,
    ) -> DatabaseResult<ExportResult> {
        self.export_to_markdown_files("custom_query", sql_query, config).await
    }

    /// Format export result for display
    pub fn format_export_result(&self, result: &ExportResult) -> String {
        let mut output = String::new();
        
        output.push_str("Export Results\n");
        output.push_str("==============\n\n");
        
        output.push_str(&format!("Files Created: {}\n", result.files_created));
        output.push_str(&format!("Files Failed: {}\n", result.files_failed));
        output.push_str(&format!("Total Size: {:.2} KB\n", result.total_size_bytes as f64 / 1024.0));
        output.push_str(&format!("Execution Time: {}ms\n", result.execution_time_ms));
        
        if !result.created_files.is_empty() {
            output.push_str("\nCreated Files:\n");
            for (i, file_path) in result.created_files.iter().enumerate() {
                if i < 10 {
                    output.push_str(&format!("  📄 {}\n", file_path.display()));
                } else if i == 10 {
                    output.push_str(&format!("  ... and {} more files\n", result.created_files.len() - 10));
                    break;
                }
            }
        }
        
        if !result.errors.is_empty() {
            output.push_str("\nErrors:\n");
            for (i, error) in result.errors.iter().enumerate() {
                if i < 5 {
                    output.push_str(&format!("  ❌ {}\n", error));
                } else if i == 5 {
                    output.push_str(&format!("  ... and {} more errors\n", result.errors.len() - 5));
                    break;
                }
            }
        }
        
        if result.files_created > 0 {
            output.push_str("\n✅ Export completed successfully!");
        } else {
            output.push_str("\n⚠️  No files were created.");
        }
        
        output
    }

    // Private helper methods

    async fn validate_export_config(&self, config: &ExportConfig) -> DatabaseResult<()> {
        // Validate prefix
        if config.prefix.is_empty() {
            return Err(DatabaseError::QueryFailed {
                query: "validate_config".to_string(),
                cause: "Export prefix cannot be empty".to_string(),
            });
        }

        // Validate prefix contains only safe characters
        if !config.prefix.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return Err(DatabaseError::QueryFailed {
                query: "validate_config".to_string(),
                cause: "Export prefix can only contain alphanumeric characters, hyphens, and underscores".to_string(),
            });
        }

        // Validate location is not a file
        if config.location.exists() && config.location.is_file() {
            return Err(DatabaseError::QueryFailed {
                query: "validate_config".to_string(),
                cause: format!("Export location is a file, not a directory: {}", config.location.display()),
            });
        }

        Ok(())
    }

    fn generate_markdown_content(
        &self,
        row: &HashMap<String, String>,
        row_index: usize,
        query_result: &QueryResult,
        config: &ExportConfig,
    ) -> DatabaseResult<String> {
        if let Some(template) = &config.markdown_template {
            // Use custom template if provided
            self.apply_template(template, row, row_index, query_result)
        } else {
            // Use default markdown formatting
            self.generate_default_markdown(row, row_index, query_result)
        }
    }

    fn generate_default_markdown(
        &self,
        row: &HashMap<String, String>,
        row_index: usize,
        query_result: &QueryResult,
    ) -> DatabaseResult<String> {
        let mut content = String::new();
        
        // Header
        content.push_str(&format!("# Export Row {}\n\n", row_index + 1));
        
        // Metadata
        content.push_str("## Metadata\n\n");
        content.push_str(&format!("- **Row Index**: {}\n", row_index + 1));
        content.push_str(&format!("- **Total Rows**: {}\n", query_result.row_count));
        content.push_str(&format!("- **Exported At**: {}\n", chrono::Utc::now().to_rfc3339()));
        content.push_str(&format!("- **Query Time**: {}ms\n\n", query_result.execution_time_ms));
        
        // Check if this looks like a file content row
        if let (Some(filepath), Some(content_text)) = (row.get("filepath"), row.get("content_text")) {
            // Format as file content
            content.push_str("## File Information\n\n");
            content.push_str(&format!("**File Path**: `{}`\n\n", filepath));
            
            if let Some(filename) = row.get("filename") {
                content.push_str(&format!("**Filename**: `{}`\n\n", filename));
            }
            
            if let Some(extension) = row.get("extension") {
                content.push_str(&format!("**Extension**: `{}`\n\n", extension));
            }
            
            if let Some(file_type) = row.get("file_type") {
                content.push_str(&format!("**File Type**: `{}`\n\n", file_type));
            }
            
            if let Some(size) = row.get("file_size_bytes") {
                if let Ok(size_num) = size.parse::<i64>() {
                    content.push_str(&format!("**Size**: {} bytes ({:.2} KB)\n\n", size_num, size_num as f64 / 1024.0));
                }
            }
            
            // File statistics
            if let (Some(lines), Some(words), Some(tokens)) = (
                row.get("line_count"), 
                row.get("word_count"), 
                row.get("token_count")
            ) {
                content.push_str("### Statistics\n\n");
                content.push_str(&format!("- **Lines**: {}\n", lines));
                content.push_str(&format!("- **Words**: {}\n", words));
                content.push_str(&format!("- **Tokens**: {}\n\n", tokens));
            }
            
            // File content
            content.push_str("## Content\n\n");
            if !content_text.is_empty() {
                // Detect language for syntax highlighting
                let language = self.detect_language_from_extension(row.get("extension"));
                content.push_str(&format!("```{}\n{}\n```\n\n", language, content_text));
            } else {
                content.push_str("*No content available*\n\n");
            }
        } else if let (Some(sql_query), Some(llm_result)) = (row.get("sql_query"), row.get("llm_result")) {
            // Format as analysis result
            content.push_str("## Analysis Information\n\n");
            
            if let Some(analysis_type) = row.get("analysis_type") {
                content.push_str(&format!("**Analysis Type**: `{}`\n\n", analysis_type));
            }
            
            if let Some(prompt_file) = row.get("prompt_file_path") {
                content.push_str(&format!("**Prompt File**: `{}`\n\n", prompt_file));
            }
            
            if let Some(original_file) = row.get("original_file_path") {
                content.push_str(&format!("**Original File**: `{}`\n\n", original_file));
            }
            
            content.push_str("### Original Query\n\n");
            content.push_str(&format!("```sql\n{}\n```\n\n", sql_query));
            
            content.push_str("### Analysis Result\n\n");
            content.push_str(&format!("{}\n\n", llm_result));
        } else {
            // Generic row formatting
            content.push_str("## Data\n\n");
            
            for (key, value) in row {
                content.push_str(&format!("**{}**: ", key.replace('_', " ").to_title_case()));
                
                // Format value based on content
                if value.len() > 200 {
                    content.push_str("\n\n```\n");
                    content.push_str(value);
                    content.push_str("\n```\n\n");
                } else if value.is_empty() {
                    content.push_str("*empty*\n\n");
                } else {
                    content.push_str(&format!("`{}`\n\n", value));
                }
            }
        }
        
        // Footer
        content.push_str("---\n\n");
        content.push_str(&format!("*Generated by code-ingest export at {}*\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        Ok(content)
    }

    fn apply_template(
        &self,
        template: &str,
        row: &HashMap<String, String>,
        row_index: usize,
        query_result: &QueryResult,
    ) -> DatabaseResult<String> {
        let mut content = template.to_string();
        
        // Replace template variables
        content = content.replace("{{ROW_INDEX}}", &(row_index + 1).to_string());
        content = content.replace("{{TOTAL_ROWS}}", &query_result.row_count.to_string());
        content = content.replace("{{TIMESTAMP}}", &chrono::Utc::now().to_rfc3339());
        content = content.replace("{{QUERY_TIME}}", &query_result.execution_time_ms.to_string());
        
        // Replace column values
        for (key, value) in row {
            let placeholder = format!("{{{{{}}}}}", key.to_uppercase());
            content = content.replace(&placeholder, value);
        }
        
        Ok(content)
    }

    fn detect_language_from_extension(&self, extension: Option<&String>) -> &str {
        match extension.map(|s| s.as_str()) {
            Some("rs") => "rust",
            Some("py") => "python",
            Some("js") => "javascript",
            Some("ts") => "typescript",
            Some("java") => "java",
            Some("cpp") | Some("cc") | Some("cxx") => "cpp",
            Some("c") => "c",
            Some("h") => "c",
            Some("go") => "go",
            Some("rb") => "ruby",
            Some("php") => "php",
            Some("cs") => "csharp",
            Some("swift") => "swift",
            Some("kt") => "kotlin",
            Some("scala") => "scala",
            Some("sh") | Some("bash") => "bash",
            Some("sql") => "sql",
            Some("json") => "json",
            Some("yaml") | Some("yml") => "yaml",
            Some("xml") => "xml",
            Some("html") => "html",
            Some("css") => "css",
            Some("md") => "markdown",
            _ => "text",
        }
    }
}

// Helper trait for string formatting
trait ToTitleCase {
    fn to_title_case(&self) -> String;
}

impl ToTitleCase for str {
    fn to_title_case(&self) -> String {
        self.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_pool() -> Option<PgPool> {
        // Only run tests if DATABASE_URL is set
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    fn create_test_row() -> HashMap<String, String> {
        let mut row = HashMap::new();
        row.insert("filepath".to_string(), "/test/example.rs".to_string());
        row.insert("filename".to_string(), "example.rs".to_string());
        row.insert("extension".to_string(), "rs".to_string());
        row.insert("file_size_bytes".to_string(), "1024".to_string());
        row.insert("line_count".to_string(), "50".to_string());
        row.insert("word_count".to_string(), "200".to_string());
        row.insert("token_count".to_string(), "180".to_string());
        row.insert("content_text".to_string(), "fn main() {\n    println!(\"Hello, world!\");\n}".to_string());
        row.insert("file_type".to_string(), "direct_text".to_string());
        row
    }

    fn create_test_query_result() -> QueryResult {
        QueryResult {
            columns: vec![
                "filepath".to_string(),
                "filename".to_string(),
                "content_text".to_string(),
            ],
            rows: vec![create_test_row()],
            row_count: 1,
            execution_time_ms: 50,
        }
    }

    #[tokio::test]
    async fn test_database_exporter_creation() {
        if let Some(pool) = create_test_pool() {
            let _exporter = DatabaseExporter::new(pool);
            // Just test that we can create the exporter
            assert!(true);
        }
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.prefix, "export");
        assert_eq!(config.location, PathBuf::from("."));
        assert_eq!(config.max_files, Some(1000));
        assert!(!config.overwrite_existing);
        assert!(config.markdown_template.is_none());
    }

    #[test]
    fn test_markdown_generation() {
        if let Some(pool) = create_test_pool() {
            let exporter = DatabaseExporter::new(pool);
            let row = create_test_row();
            let query_result = create_test_query_result();
            let config = ExportConfig::default();
            
            let markdown = exporter.generate_default_markdown(&row, 0, &query_result).unwrap();
            
            assert!(markdown.contains("# Export Row 1"));
            assert!(markdown.contains("example.rs"));
            assert!(markdown.contains("```rust"));
            assert!(markdown.contains("fn main()"));
            assert!(markdown.contains("Hello, world!"));
        }
    }

    #[test]
    fn test_language_detection() {
        if let Some(pool) = create_test_pool() {
            let exporter = DatabaseExporter::new(pool);
            
            assert_eq!(exporter.detect_language_from_extension(Some(&"rs".to_string())), "rust");
            assert_eq!(exporter.detect_language_from_extension(Some(&"py".to_string())), "python");
            assert_eq!(exporter.detect_language_from_extension(Some(&"js".to_string())), "javascript");
            assert_eq!(exporter.detect_language_from_extension(Some(&"unknown".to_string())), "text");
            assert_eq!(exporter.detect_language_from_extension(None), "text");
        }
    }

    #[test]
    fn test_title_case_conversion() {
        assert_eq!("hello world".to_title_case(), "Hello World");
        assert_eq!("file_name".to_title_case(), "File_name");
        assert_eq!("UPPERCASE".to_title_case(), "Uppercase");
        assert_eq!("".to_title_case(), "");
    }

    #[tokio::test]
    async fn test_config_validation() {
        if let Some(pool) = create_test_pool() {
            let exporter = DatabaseExporter::new(pool);
            
            // Valid config
            let valid_config = ExportConfig::default();
            assert!(exporter.validate_export_config(&valid_config).await.is_ok());
            
            // Invalid prefix (empty)
            let mut invalid_config = ExportConfig::default();
            invalid_config.prefix = "".to_string();
            assert!(exporter.validate_export_config(&invalid_config).await.is_err());
            
            // Invalid prefix (special characters)
            invalid_config.prefix = "test@file".to_string();
            assert!(exporter.validate_export_config(&invalid_config).await.is_err());
            
            // Valid prefix with allowed characters
            invalid_config.prefix = "test-file_123".to_string();
            assert!(exporter.validate_export_config(&invalid_config).await.is_ok());
        }
    }

    #[test]
    fn test_export_result_formatting() {
        if let Some(pool) = create_test_pool() {
            let exporter = DatabaseExporter::new(pool);
            
            let result = ExportResult {
                files_created: 5,
                files_failed: 1,
                total_size_bytes: 10240,
                execution_time_ms: 150,
                created_files: vec![
                    PathBuf::from("test-00001.md"),
                    PathBuf::from("test-00002.md"),
                ],
                errors: vec!["Test error".to_string()],
            };
            
            let formatted = exporter.format_export_result(&result);
            
            assert!(formatted.contains("Files Created: 5"));
            assert!(formatted.contains("Files Failed: 1"));
            assert!(formatted.contains("10.00 KB"));
            assert!(formatted.contains("150ms"));
            assert!(formatted.contains("test-00001.md"));
            assert!(formatted.contains("Test error"));
        }
    }

    #[test]
    fn test_template_application() {
        if let Some(pool) = create_test_pool() {
            let exporter = DatabaseExporter::new(pool);
            let row = create_test_row();
            let query_result = create_test_query_result();
            
            let template = "# Row {{ROW_INDEX}} of {{TOTAL_ROWS}}\n\nFile: {{FILEPATH}}\nContent: {{CONTENT_TEXT}}";
            
            let result = exporter.apply_template(template, &row, 0, &query_result).unwrap();
            
            assert!(result.contains("# Row 1 of 1"));
            assert!(result.contains("File: /test/example.rs"));
            assert!(result.contains("fn main()"));
        }
    }
}