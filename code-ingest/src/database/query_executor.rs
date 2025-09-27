//! SQL query execution and result formatting for LLM consumption
//!
//! This module provides query execution with clean terminal output formatting,
//! result formatting for LLM consumption (FILE: format), query validation,
//! and support for large result sets with streaming.

use crate::database::operations::{DatabaseOperations, QueryResult};
use crate::error::{DatabaseError, DatabaseResult};
use sqlx::PgPool;
use std::io::{self, Write};
use tracing::{debug, info, warn};

/// Query executor with formatting capabilities
pub struct QueryExecutor {
    operations: DatabaseOperations,
}

/// Configuration for query execution and formatting
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Maximum number of rows to return (0 = no limit)
    pub max_rows: usize,
    /// Whether to format output for LLM consumption
    pub llm_format: bool,
    /// Whether to include execution statistics
    pub include_stats: bool,
    /// Whether to stream large results
    pub stream_results: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            max_rows: 0, // No limit by default
            llm_format: false,
            include_stats: true,
            stream_results: true,
        }
    }
}

/// Formatted query output
#[derive(Debug, Clone)]
pub struct FormattedQueryOutput {
    pub content: String,
    pub row_count: usize,
    pub execution_time_ms: u64,
    pub truncated: bool,
}

impl QueryExecutor {
    /// Create a new query executor
    pub fn new(pool: PgPool) -> Self {
        Self {
            operations: DatabaseOperations::new(pool),
        }
    }

    /// Execute a SQL query and return formatted results for terminal output
    pub async fn execute_query_terminal(&self, sql: &str) -> DatabaseResult<FormattedQueryOutput> {
        let config = QueryConfig {
            llm_format: false,
            include_stats: true,
            ..Default::default()
        };
        
        self.execute_query_with_config(sql, &config).await
    }

    /// Execute a SQL query and return formatted results for LLM consumption
    pub async fn execute_query_llm(&self, sql: &str) -> DatabaseResult<FormattedQueryOutput> {
        let config = QueryConfig {
            llm_format: true,
            include_stats: false,
            ..Default::default()
        };
        
        self.execute_query_with_config(sql, &config).await
    }

    /// Execute a SQL query with custom configuration
    pub async fn execute_query_with_config(
        &self,
        sql: &str,
        config: &QueryConfig,
    ) -> DatabaseResult<FormattedQueryOutput> {
        debug!("Executing query with config: {:?}", config);
        
        // Validate the SQL query
        self.validate_query(sql)?;
        
        // Execute the query
        let result = self.operations.execute_query(sql).await?;
        
        // Apply row limit if specified
        let (limited_result, truncated) = self.apply_row_limit(result, config.max_rows);
        
        // Format the output based on configuration
        let content = if config.llm_format {
            self.format_for_llm(&limited_result, config)
        } else {
            self.format_for_terminal(&limited_result, config)
        };
        
        info!(
            "Query executed successfully: {} rows in {}ms{}",
            limited_result.row_count,
            limited_result.execution_time_ms,
            if truncated { " (truncated)" } else { "" }
        );
        
        Ok(FormattedQueryOutput {
            content,
            row_count: limited_result.row_count,
            execution_time_ms: limited_result.execution_time_ms,
            truncated,
        })
    }

    /// Execute a query and stream results to stdout (for large result sets)
    pub async fn execute_query_streaming(
        &self,
        sql: &str,
        config: &QueryConfig,
    ) -> DatabaseResult<usize> {
        debug!("Executing streaming query");
        
        // Validate the SQL query
        self.validate_query(sql)?;
        
        // For streaming, we'll execute in chunks
        // This is a simplified implementation - in production you'd use cursor-based pagination
        let chunk_size = 1000;
        let mut offset = 0;
        let mut total_rows = 0;
        
        loop {
            let paginated_sql = format!("{} LIMIT {} OFFSET {}", sql, chunk_size, offset);
            let result = self.operations.execute_query(&paginated_sql).await?;
            
            if result.rows.is_empty() {
                break;
            }
            
            // Format and output this chunk
            let chunk_content = if config.llm_format {
                self.format_for_llm(&result, config)
            } else {
                self.format_for_terminal(&result, config)
            };
            
            // Write to stdout
            print!("{}", chunk_content);
            io::stdout().flush().map_err(|e| DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: format!("Failed to write output: {}", e),
            })?;
            
            total_rows += result.row_count;
            offset += chunk_size;
            
            // Apply max_rows limit if specified
            if config.max_rows > 0 && total_rows >= config.max_rows {
                break;
            }
        }
        
        info!("Streaming query completed: {} total rows", total_rows);
        Ok(total_rows)
    }

    /// Validate SQL query for safety and correctness
    fn validate_query(&self, sql: &str) -> DatabaseResult<()> {
        let sql_trimmed = sql.trim().to_lowercase();
        
        // Check for empty query
        if sql_trimmed.is_empty() {
            return Err(DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: "Query cannot be empty".to_string(),
            });
        }
        
        // Check for dangerous operations (basic safety check)
        let dangerous_keywords = [
            "drop table", "drop database", "truncate", "delete from",
            "alter table", "create user", "drop user", "grant", "revoke"
        ];
        
        for keyword in &dangerous_keywords {
            if sql_trimmed.contains(keyword) {
                warn!("Potentially dangerous SQL detected: {}", keyword);
                // For now, we'll allow it but log a warning
                // In production, you might want to restrict this
            }
        }
        
        // Basic syntax validation (check for balanced quotes)
        let single_quotes = sql.chars().filter(|&c| c == '\'').count();
        let double_quotes = sql.chars().filter(|&c| c == '"').count();
        
        if single_quotes % 2 != 0 {
            return Err(DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: "Unbalanced single quotes in query".to_string(),
            });
        }
        
        if double_quotes % 2 != 0 {
            return Err(DatabaseError::QueryFailed {
                query: sql.to_string(),
                cause: "Unbalanced double quotes in query".to_string(),
            });
        }
        
        Ok(())
    }

    /// Apply row limit to query results
    fn apply_row_limit(&self, mut result: QueryResult, max_rows: usize) -> (QueryResult, bool) {
        if max_rows == 0 || result.rows.len() <= max_rows {
            return (result, false);
        }
        
        result.rows.truncate(max_rows);
        result.row_count = result.rows.len();
        (result, true)
    }

    /// Format query results for terminal display
    fn format_for_terminal(&self, result: &QueryResult, config: &QueryConfig) -> String {
        let mut output = String::new();
        
        if result.rows.is_empty() {
            output.push_str("No rows returned.\n");
            if config.include_stats {
                output.push_str(&format!("Execution time: {}ms\n", result.execution_time_ms));
            }
            return output;
        }
        
        // Calculate column widths for nice formatting
        let mut col_widths: Vec<usize> = result.columns.iter().map(|col| col.len()).collect();
        
        for row in &result.rows {
            for (i, col) in result.columns.iter().enumerate() {
                if let Some(value) = row.get(col) {
                    col_widths[i] = col_widths[i].max(value.len());
                }
            }
        }
        
        // Limit column width to prevent extremely wide output
        for width in &mut col_widths {
            *width = (*width).min(50);
        }
        
        // Header
        for (i, col) in result.columns.iter().enumerate() {
            output.push_str(&format!("{:<width$}", col, width = col_widths[i]));
            if i < result.columns.len() - 1 {
                output.push_str(" | ");
            }
        }
        output.push('\n');
        
        // Separator
        for (i, &width) in col_widths.iter().enumerate() {
            output.push_str(&"-".repeat(width));
            if i < col_widths.len() - 1 {
                output.push_str("-+-");
            }
        }
        output.push('\n');
        
        // Rows
        for row in &result.rows {
            for (i, col) in result.columns.iter().enumerate() {
                let empty_string = String::new();
                let value = row.get(col).unwrap_or(&empty_string);
                let truncated_value = if value.len() > col_widths[i] {
                    format!("{}...", &value[..col_widths[i].saturating_sub(3)])
                } else {
                    value.clone()
                };
                
                output.push_str(&format!("{:<width$}", truncated_value, width = col_widths[i]));
                if i < result.columns.len() - 1 {
                    output.push_str(" | ");
                }
            }
            output.push('\n');
        }
        
        // Statistics
        if config.include_stats {
            output.push('\n');
            output.push_str(&format!("Rows: {}\n", result.row_count));
            output.push_str(&format!("Execution time: {}ms\n", result.execution_time_ms));
        }
        
        output
    }

    /// Format query results for LLM consumption
    fn format_for_llm(&self, result: &QueryResult, _config: &QueryConfig) -> String {
        let mut output = String::new();
        
        for row in &result.rows {
            // Check if this looks like a file-based query (has filepath and content_text)
            if let (Some(filepath), Some(content)) = (row.get("filepath"), row.get("content_text")) {
                output.push_str(&format!("FILE: {}\n\n", filepath));
                output.push_str(content);
                output.push_str("\n\n---\n\n");
            } else {
                // For non-file queries, format as structured data
                for (key, value) in row {
                    output.push_str(&format!("{}: {}\n", key, value));
                }
                output.push_str("\n---\n\n");
            }
        }
        
        output
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

    fn create_test_query_result() -> QueryResult {
        let mut row1 = HashMap::new();
        row1.insert("id".to_string(), "1".to_string());
        row1.insert("name".to_string(), "test_file.rs".to_string());
        row1.insert("filepath".to_string(), "/test/file.rs".to_string());
        row1.insert("content_text".to_string(), "fn main() {}".to_string());
        
        let mut row2 = HashMap::new();
        row2.insert("id".to_string(), "2".to_string());
        row2.insert("name".to_string(), "another_file.rs".to_string());
        row2.insert("filepath".to_string(), "/test/another.rs".to_string());
        row2.insert("content_text".to_string(), "struct Test;".to_string());
        
        QueryResult {
            columns: vec!["id".to_string(), "name".to_string(), "filepath".to_string(), "content_text".to_string()],
            rows: vec![row1, row2],
            row_count: 2,
            execution_time_ms: 50,
        }
    }

    #[test]
    fn test_query_config_default() {
        let config = QueryConfig::default();
        assert_eq!(config.max_rows, 0);
        assert!(!config.llm_format);
        assert!(config.include_stats);
        assert!(config.stream_results);
    }

    #[test]
    fn test_query_validation_empty() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            
            let result = executor.validate_query("");
            assert!(result.is_err());
            
            let result = executor.validate_query("   ");
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_query_validation_quotes() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            
            // Valid queries
            assert!(executor.validate_query("SELECT * FROM test").is_ok());
            assert!(executor.validate_query("SELECT 'hello' FROM test").is_ok());
            assert!(executor.validate_query("SELECT \"column\" FROM test").is_ok());
            
            // Invalid queries (unbalanced quotes)
            assert!(executor.validate_query("SELECT 'hello FROM test").is_err());
            assert!(executor.validate_query("SELECT \"hello FROM test").is_err());
        }
    }

    #[test]
    fn test_row_limit_application() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let result = create_test_query_result();
            
            // No limit
            let (limited, truncated) = executor.apply_row_limit(result.clone(), 0);
            assert_eq!(limited.row_count, 2);
            assert!(!truncated);
            
            // Limit higher than row count
            let (limited, truncated) = executor.apply_row_limit(result.clone(), 5);
            assert_eq!(limited.row_count, 2);
            assert!(!truncated);
            
            // Limit lower than row count
            let (limited, truncated) = executor.apply_row_limit(result.clone(), 1);
            assert_eq!(limited.row_count, 1);
            assert!(truncated);
        }
    }

    #[test]
    fn test_terminal_formatting() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let result = create_test_query_result();
            let config = QueryConfig::default();
            
            let formatted = executor.format_for_terminal(&result, &config);
            
            // Should contain headers
            assert!(formatted.contains("id"));
            assert!(formatted.contains("name"));
            assert!(formatted.contains("filepath"));
            
            // Should contain data
            assert!(formatted.contains("test_file.rs"));
            assert!(formatted.contains("/test/file.rs"));
            
            // Should contain statistics
            assert!(formatted.contains("Rows: 2"));
            assert!(formatted.contains("Execution time: 50ms"));
        }
    }

    #[test]
    fn test_llm_formatting() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let result = create_test_query_result();
            let config = QueryConfig::default();
            
            let formatted = executor.format_for_llm(&result, &config);
            
            // Should contain FILE: markers
            assert!(formatted.contains("FILE: /test/file.rs"));
            assert!(formatted.contains("FILE: /test/another.rs"));
            
            // Should contain content
            assert!(formatted.contains("fn main() {}"));
            assert!(formatted.contains("struct Test;"));
            
            // Should contain separators
            assert!(formatted.contains("---"));
        }
    }

    #[test]
    fn test_empty_result_formatting() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            let empty_result = QueryResult {
                columns: vec![],
                rows: vec![],
                row_count: 0,
                execution_time_ms: 10,
            };
            let config = QueryConfig::default();
            
            let formatted = executor.format_for_terminal(&empty_result, &config);
            assert!(formatted.contains("No rows returned"));
            assert!(formatted.contains("Execution time: 10ms"));
        }
    }

    #[tokio::test]
    async fn test_query_executor_creation() {
        if let Some(pool) = create_test_pool() {
            let _executor = QueryExecutor::new(pool);
            // Just test that we can create the executor
            assert!(true);
        }
    }

    #[test]
    fn test_formatted_query_output() {
        let output = FormattedQueryOutput {
            content: "test content".to_string(),
            row_count: 5,
            execution_time_ms: 100,
            truncated: true,
        };
        
        assert_eq!(output.content, "test content");
        assert_eq!(output.row_count, 5);
        assert_eq!(output.execution_time_ms, 100);
        assert!(output.truncated);
    }

    #[test]
    fn test_query_config_customization() {
        let config = QueryConfig {
            max_rows: 100,
            llm_format: true,
            include_stats: false,
            stream_results: false,
        };
        
        assert_eq!(config.max_rows, 100);
        assert!(config.llm_format);
        assert!(!config.include_stats);
        assert!(!config.stream_results);
    }

    #[test]
    fn test_dangerous_query_detection() {
        if let Some(pool) = create_test_pool() {
            let executor = QueryExecutor::new(pool);
            
            // These should pass validation but log warnings
            assert!(executor.validate_query("SELECT * FROM test").is_ok());
            
            // Dangerous queries should still pass (with warnings) for flexibility
            // In production, you might want to restrict these
            assert!(executor.validate_query("DROP TABLE test").is_ok());
            assert!(executor.validate_query("DELETE FROM test").is_ok());
        }
    }
}