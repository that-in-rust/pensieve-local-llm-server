//! Database exploration functionality
//!
//! This module provides commands for exploring and inspecting the PostgreSQL database,
//! including listing tables, sampling data, describing schemas, and showing database info.

use crate::error::{DatabaseError, DatabaseResult};
use crate::database::{SchemaManager, TableType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row, Column};
use std::collections::HashMap;
use tracing::debug;

/// Database exploration manager
pub struct DatabaseExplorer {
    pool: PgPool,
    pub schema_manager: SchemaManager,
}

/// Information about database connection and status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    pub connection_status: String,
    pub database_name: String,
    pub server_version: String,
    pub total_tables: usize,
    pub ingestion_tables: usize,
    pub query_result_tables: usize,
    pub total_size_mb: f64,
    pub connection_time_ms: u64,
}

/// Information about a table for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableListItem {
    pub name: String,
    pub table_type: String,
    pub row_count: Option<i64>,
    pub size_mb: Option<f64>,
    pub created_at: Option<DateTime<Utc>>,
    pub description: Option<String>,
}

/// Sample data from a table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSample {
    pub table_name: String,
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, String>>,
    pub total_rows: i64,
    pub sample_size: usize,
    pub execution_time_ms: u64,
}

/// Schema information for a table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub table_name: String,
    pub table_type: String,
    pub columns: Vec<ColumnInfo>,
    pub indexes: Vec<IndexInfo>,
    pub constraints: Vec<ConstraintInfo>,
    pub row_count: Option<i64>,
    pub size_mb: Option<f64>,
}

/// Information about a table column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: String,
    pub is_nullable: bool,
    pub default_value: Option<String>,
    pub is_primary_key: bool,
    pub character_maximum_length: Option<i32>,
}

/// Information about a table index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexInfo {
    pub name: String,
    pub columns: Vec<String>,
    pub is_unique: bool,
    pub index_type: String,
}

/// Information about a table constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintInfo {
    pub name: String,
    pub constraint_type: String,
    pub columns: Vec<String>,
    pub definition: Option<String>,
}

/// Result of table cleanup operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupResult {
    pub tables_removed: usize,
    pub tables_kept: usize,
    pub space_freed_mb: f64,
    pub errors: Vec<String>,
}

/// Management recommendation for database maintenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagementRecommendation {
    pub recommendation_type: RecommendationType,
    pub title: String,
    pub description: String,
    pub action: String,
    pub priority: Priority,
}

/// Type of management recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Cleanup,
    Performance,
    Security,
    Storage,
}

/// Priority level for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    High,
    Medium,
    Low,
}

/// Result of table optimization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub tables_optimized: usize,
    pub total_tables: usize,
    pub duration_ms: u64,
    pub errors: Vec<String>,
}

impl DatabaseExplorer {
    /// Create a new database explorer
    pub fn new(pool: PgPool) -> Self {
        let schema_manager = SchemaManager::new(pool.clone());
        Self {
            pool,
            schema_manager,
        }
    }

    /// Clean up old ingestion tables
    pub async fn cleanup_old_tables(&self, keep_count: usize) -> DatabaseResult<CleanupResult> {
        debug!("Cleaning up old ingestion tables, keeping {} most recent", keep_count);

        // Get all ingestion tables sorted by creation date
        let ingestion_tables = self.schema_manager.list_tables(Some(TableType::Ingestion)).await?;
        
        if ingestion_tables.len() <= keep_count {
            return Ok(CleanupResult {
                tables_removed: 0,
                tables_kept: ingestion_tables.len(),
                space_freed_mb: 0.0,
                errors: vec![],
            });
        }

        // Sort tables by creation date (newest first)
        let mut table_info: Vec<(String, DateTime<Utc>, f64)> = Vec::new();
        for table_name in &ingestion_tables {
            if let Ok(info) = self.schema_manager.get_table_info(table_name).await {
                let (_, size_mb) = self.get_table_stats(table_name).await.unwrap_or((None, Some(0.0)));
                table_info.push((table_name.clone(), info.created_at, size_mb.unwrap_or(0.0)));
            }
        }
        
        table_info.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by date, newest first

        // Identify tables to remove (keep the newest ones)
        let tables_to_remove: Vec<_> = table_info.iter().skip(keep_count).collect();
        
        let mut removed_count = 0;
        let mut space_freed = 0.0;
        let mut errors = Vec::new();

        for (table_name, _, size_mb) in tables_to_remove {
            match self.drop_table(table_name).await {
                Ok(_) => {
                    removed_count += 1;
                    space_freed += size_mb;
                    debug!("Dropped table: {}", table_name);
                }
                Err(e) => {
                    errors.push(format!("Failed to drop table {}: {}", table_name, e));
                }
            }
        }

        Ok(CleanupResult {
            tables_removed: removed_count,
            tables_kept: table_info.len() - removed_count,
            space_freed_mb: space_freed,
            errors,
        })
    }

    /// Drop a specific table
    pub async fn drop_table(&self, table_name: &str) -> DatabaseResult<()> {
        debug!("Dropping table: {}", table_name);

        // Verify table exists
        if !self.schema_manager.table_exists(table_name).await? {
            return Err(DatabaseError::QueryFailed {
                query: format!("DROP TABLE {}", table_name),
                cause: "Table does not exist".to_string(),
            });
        }

        let drop_sql = format!("DROP TABLE \"{}\" CASCADE", table_name);
        sqlx::query(&drop_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: drop_sql,
                cause: e.to_string(),
            })?;

        Ok(())
    }

    /// Get table management recommendations
    pub async fn get_management_recommendations(&self) -> DatabaseResult<Vec<ManagementRecommendation>> {
        let mut recommendations = Vec::new();

        // Check for old ingestion tables
        let ingestion_tables = self.schema_manager.list_tables(Some(TableType::Ingestion)).await?;
        if ingestion_tables.len() > 10 {
            recommendations.push(ManagementRecommendation {
                recommendation_type: RecommendationType::Cleanup,
                title: "Too many ingestion tables".to_string(),
                description: format!("You have {} ingestion tables. Consider cleaning up old ones.", ingestion_tables.len()),
                action: format!("Run: code-ingest cleanup-tables --keep 5"),
                priority: Priority::Medium,
            });
        }

        // Check for large tables
        for table_name in &ingestion_tables {
            if let Ok((Some(row_count), Some(size_mb))) = self.get_table_stats(table_name).await {
                if size_mb > 100.0 {
                    recommendations.push(ManagementRecommendation {
                        recommendation_type: RecommendationType::Performance,
                        title: format!("Large table: {}", table_name),
                        description: format!("Table {} is {:.1} MB with {} rows", table_name, size_mb, row_count),
                        action: "Consider archiving or cleaning up this table if no longer needed".to_string(),
                        priority: Priority::Low,
                    });
                }
            }
        }

        // Check for empty tables
        for table_name in &ingestion_tables {
            if let Ok((Some(row_count), _)) = self.get_table_stats(table_name).await {
                if row_count == 0 {
                    recommendations.push(ManagementRecommendation {
                        recommendation_type: RecommendationType::Cleanup,
                        title: format!("Empty table: {}", table_name),
                        description: format!("Table {} has no data", table_name),
                        action: format!("Consider dropping: code-ingest drop-table --table {}", table_name),
                        priority: Priority::Low,
                    });
                }
            }
        }

        Ok(recommendations)
    }

    /// Vacuum and analyze tables for performance
    pub async fn optimize_tables(&self, table_names: Option<Vec<String>>) -> DatabaseResult<OptimizationResult> {
        let tables = if let Some(names) = table_names {
            names
        } else {
            self.schema_manager.list_tables(None).await?
        };

        let mut optimized_count = 0;
        let mut errors = Vec::new();
        let start_time = std::time::Instant::now();

        for table_name in &tables {
            // VACUUM ANALYZE for each table
            let vacuum_sql = format!("VACUUM ANALYZE \"{}\"", table_name);
            match sqlx::query(&vacuum_sql).execute(&self.pool).await {
                Ok(_) => {
                    optimized_count += 1;
                    debug!("Optimized table: {}", table_name);
                }
                Err(e) => {
                    errors.push(format!("Failed to optimize table {}: {}", table_name, e));
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(OptimizationResult {
            tables_optimized: optimized_count,
            total_tables: tables.len(),
            duration_ms,
            errors,
        })
    }

    /// Get comprehensive database information
    pub async fn get_database_info(&self) -> DatabaseResult<DatabaseInfo> {
        let start_time = std::time::Instant::now();
        
        debug!("Gathering database information");

        // Test connection and get basic info
        let version_query = "SELECT version()";
        let version_row = sqlx::query(version_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::ConnectionFailed {
                url: "database".to_string(),
                cause: e.to_string(),
            })?;
        
        let server_version: String = version_row.get(0);

        // Get database name
        let db_name_query = "SELECT current_database()";
        let db_name_row = sqlx::query(db_name_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: db_name_query.to_string(),
                cause: e.to_string(),
            })?;
        
        let database_name: String = db_name_row.get(0);

        // Get table counts
        let all_tables = self.schema_manager.list_tables(None).await?;
        let ingestion_tables = self.schema_manager.list_tables(Some(TableType::Ingestion)).await?;
        let query_result_tables = self.schema_manager.list_tables(Some(TableType::QueryResult)).await?;

        // Get database size
        let size_query = r#"
            SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                   pg_database_size(current_database()) as size_bytes
        "#;
        
        let size_row = sqlx::query(size_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: size_query.to_string(),
                cause: e.to_string(),
            })?;

        let size_bytes: i64 = size_row.get("size_bytes");
        let total_size_mb = size_bytes as f64 / (1024.0 * 1024.0);

        let connection_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(DatabaseInfo {
            connection_status: "Connected".to_string(),
            database_name,
            server_version: server_version.split_whitespace().take(2).collect::<Vec<_>>().join(" "),
            total_tables: all_tables.len(),
            ingestion_tables: ingestion_tables.len(),
            query_result_tables: query_result_tables.len(),
            total_size_mb,
            connection_time_ms,
        })
    }

    /// List tables with optional filtering by type
    pub async fn list_tables(&self, table_type: Option<TableType>) -> DatabaseResult<Vec<TableListItem>> {
        debug!("Listing tables with filter: {:?}", table_type);

        let tables = self.schema_manager.list_tables(table_type.clone()).await?;
        let mut table_items = Vec::new();

        for table_name in tables {
            let item = self.get_table_list_item(&table_name).await?;
            table_items.push(item);
        }

        // Sort by table type, then by name
        table_items.sort_by(|a, b| {
            a.table_type.cmp(&b.table_type)
                .then_with(|| a.name.cmp(&b.name))
        });

        Ok(table_items)
    }

    /// Get sample data from a table
    pub async fn sample_table(&self, table_name: &str, limit: usize) -> DatabaseResult<TableSample> {
        let start_time = std::time::Instant::now();
        
        debug!("Sampling table: {} (limit: {})", table_name, limit);

        // Verify table exists
        if !self.schema_manager.table_exists(table_name).await? {
            return Err(DatabaseError::QueryFailed {
                query: format!("Sample table {}", table_name),
                cause: "Table does not exist".to_string(),
            });
        }

        // Get total row count
        let count_query = format!("SELECT COUNT(*) FROM \"{}\"", table_name);
        let count_row = sqlx::query(&count_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: count_query.clone(),
                cause: e.to_string(),
            })?;
        
        let total_rows: i64 = count_row.get(0);

        // Get sample data
        let sample_query = format!("SELECT * FROM \"{}\" LIMIT {}", table_name, limit);
        let sample_rows = sqlx::query(&sample_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: sample_query.clone(),
                cause: e.to_string(),
            })?;

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        if sample_rows.is_empty() {
            return Ok(TableSample {
                table_name: table_name.to_string(),
                columns: vec![],
                rows: vec![],
                total_rows,
                sample_size: 0,
                execution_time_ms,
            });
        }

        // Extract column names
        let columns: Vec<String> = sample_rows[0]
            .columns()
            .iter()
            .map(|col| col.name().to_string())
            .collect();

        // Convert rows to HashMap format
        let mut result_rows = Vec::new();
        for row in &sample_rows {
            let mut row_map = HashMap::new();
            for (i, column) in columns.iter().enumerate() {
                let value = self.extract_column_value(&row, i)?;
                row_map.insert(column.clone(), value);
            }
            result_rows.push(row_map);
        }

        Ok(TableSample {
            table_name: table_name.to_string(),
            columns,
            rows: result_rows,
            total_rows,
            sample_size: sample_rows.len(),
            execution_time_ms,
        })
    }

    /// Get detailed schema information for a table
    pub async fn describe_table(&self, table_name: &str) -> DatabaseResult<TableSchema> {
        debug!("Describing table: {}", table_name);

        // Verify table exists
        if !self.schema_manager.table_exists(table_name).await? {
            return Err(DatabaseError::QueryFailed {
                query: format!("Describe table {}", table_name),
                cause: "Table does not exist".to_string(),
            });
        }

        // Get column information
        let columns = self.get_table_columns(table_name).await?;
        
        // Get index information
        let indexes = self.get_table_indexes(table_name).await?;
        
        // Get constraint information
        let constraints = self.get_table_constraints(table_name).await?;

        // Get row count and size
        let (row_count, size_mb) = self.get_table_stats(table_name).await?;

        // Determine table type
        let table_type = match self.schema_manager.get_table_info(table_name).await {
            Ok(info) => format!("{:?}", info.table_type),
            Err(_) => "Unknown".to_string(),
        };

        Ok(TableSchema {
            table_name: table_name.to_string(),
            table_type,
            columns,
            indexes,
            constraints,
            row_count,
            size_mb,
        })
    }

    /// Format database info for display
    pub fn format_database_info(&self, info: &DatabaseInfo) -> String {
        format!(
            r#"Database Information
====================

Connection Status: {}
Database Name: {}
Server Version: {}
Connection Time: {}ms

Table Summary:
  Total Tables: {}
  Ingestion Tables: {}
  Query Result Tables: {}
  Other Tables: {}

Storage:
  Total Size: {:.2} MB

Status: ✅ Connected and operational"#,
            info.connection_status,
            info.database_name,
            info.server_version,
            info.connection_time_ms,
            info.total_tables,
            info.ingestion_tables,
            info.query_result_tables,
            info.total_tables - info.ingestion_tables - info.query_result_tables,
            info.total_size_mb
        )
    }

    /// Format table list for display
    pub fn format_table_list(&self, tables: &[TableListItem], show_details: bool) -> String {
        if tables.is_empty() {
            return "No tables found in the database.".to_string();
        }

        let mut output = String::new();
        output.push_str("Database Tables\n");
        output.push_str("===============\n\n");

        // Group by table type
        let mut by_type: HashMap<String, Vec<&TableListItem>> = HashMap::new();
        for table in tables {
            by_type.entry(table.table_type.clone()).or_default().push(table);
        }

        for (table_type, type_tables) in by_type {
            output.push_str(&format!("{} Tables ({}):\n", table_type, type_tables.len()));
            
            for table in type_tables {
                if show_details {
                    output.push_str(&format!(
                        "  📊 {} ({}{}{})\n",
                        table.name,
                        table.row_count.map(|c| format!("{} rows", c)).unwrap_or_else(|| "? rows".to_string()),
                        table.size_mb.map(|s| format!(", {:.2} MB", s)).unwrap_or_default(),
                        table.created_at.map(|c| format!(", created {}", c.format("%Y-%m-%d"))).unwrap_or_default()
                    ));
                } else {
                    output.push_str(&format!("  📊 {}\n", table.name));
                }
            }
            output.push('\n');
        }

        output.push_str(&format!("Total: {} tables", tables.len()));
        output
    }

    /// Format table sample for display
    pub fn format_table_sample(&self, sample: &TableSample) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("Table Sample: {}\n", sample.table_name));
        output.push_str(&format!("Total Rows: {} | Sample Size: {} | Query Time: {}ms\n", 
                                sample.total_rows, sample.sample_size, sample.execution_time_ms));
        output.push_str("=".repeat(80).as_str());
        output.push('\n');

        if sample.rows.is_empty() {
            output.push_str("No data in table.\n");
            return output;
        }

        // Calculate column widths
        let mut col_widths: HashMap<String, usize> = HashMap::new();
        for col in &sample.columns {
            col_widths.insert(col.clone(), col.len().max(10));
        }

        for row in &sample.rows {
            for (col, value) in row {
                if let Some(width) = col_widths.get_mut(col) {
                    *width = (*width).max(value.len().min(50)); // Cap at 50 chars for display
                }
            }
        }

        // Print header
        for col in &sample.columns {
            let width = col_widths.get(col).unwrap_or(&10);
            output.push_str(&format!("{:<width$} | ", col, width = width));
        }
        output.push('\n');

        // Print separator
        for col in &sample.columns {
            let width = col_widths.get(col).unwrap_or(&10);
            output.push_str(&format!("{} | ", "-".repeat(*width)));
        }
        output.push('\n');

        // Print rows
        for row in &sample.rows {
            for col in &sample.columns {
                let width = col_widths.get(col).unwrap_or(&10);
                let null_string = "NULL".to_string();
                let value = row.get(col).unwrap_or(&null_string);
                let display_value = if value.len() > 50 {
                    format!("{}...", &value[..47])
                } else {
                    value.clone()
                };
                output.push_str(&format!("{:<width$} | ", display_value, width = width));
            }
            output.push('\n');
        }

        output
    }

    /// Format table schema for display
    pub fn format_table_schema(&self, schema: &TableSchema) -> String {
        let mut output = String::new();
        
        output.push_str(&format!("Table Schema: {}\n", schema.table_name));
        output.push_str(&format!("Type: {} | Rows: {} | Size: {}\n", 
                                schema.table_type,
                                schema.row_count.map(|c| c.to_string()).unwrap_or_else(|| "Unknown".to_string()),
                                schema.size_mb.map(|s| format!("{:.2} MB", s)).unwrap_or_else(|| "Unknown".to_string())));
        output.push_str("=".repeat(80).as_str());
        output.push('\n');

        // Columns section
        output.push_str("\nColumns:\n");
        output.push_str("--------\n");
        for col in &schema.columns {
            let pk_marker = if col.is_primary_key { " 🔑" } else { "" };
            let nullable = if col.is_nullable { "NULL" } else { "NOT NULL" };
            let max_len = col.character_maximum_length
                .map(|l| format!("({})", l))
                .unwrap_or_default();
            
            output.push_str(&format!(
                "  {} {}{} {} {}{}\n",
                col.name,
                col.data_type,
                max_len,
                nullable,
                col.default_value.as_ref().map(|d| format!("DEFAULT {}", d)).unwrap_or_default(),
                pk_marker
            ));
        }

        // Indexes section
        if !schema.indexes.is_empty() {
            output.push_str("\nIndexes:\n");
            output.push_str("--------\n");
            for idx in &schema.indexes {
                let unique_marker = if idx.is_unique { " (UNIQUE)" } else { "" };
                output.push_str(&format!(
                    "  {} ON ({}) {}{}\n",
                    idx.name,
                    idx.columns.join(", "),
                    idx.index_type,
                    unique_marker
                ));
            }
        }

        // Constraints section
        if !schema.constraints.is_empty() {
            output.push_str("\nConstraints:\n");
            output.push_str("------------\n");
            for constraint in &schema.constraints {
                output.push_str(&format!(
                    "  {} ({}) ON ({})\n",
                    constraint.name,
                    constraint.constraint_type,
                    constraint.columns.join(", ")
                ));
            }
        }

        output
    }

    // Private helper methods

    async fn get_table_list_item(&self, table_name: &str) -> DatabaseResult<TableListItem> {
        let table_info = self.schema_manager.get_table_info(table_name).await?;
        let (row_count, size_mb) = self.get_table_stats(table_name).await?;

        Ok(TableListItem {
            name: table_name.to_string(),
            table_type: format!("{:?}", table_info.table_type),
            row_count,
            size_mb,
            created_at: Some(table_info.created_at),
            description: self.get_table_description(&table_info.table_type),
        })
    }

    fn get_table_description(&self, table_type: &TableType) -> Option<String> {
        match table_type {
            TableType::Ingestion => Some("Stores ingested code files and metadata".to_string()),
            TableType::QueryResult => Some("Stores LLM analysis results".to_string()),
            TableType::Meta => Some("Tracks ingestion operations metadata".to_string()),
        }
    }

    async fn get_table_stats(&self, table_name: &str) -> DatabaseResult<(Option<i64>, Option<f64>)> {
        // Get row count
        let count_query = format!("SELECT COUNT(*) FROM \"{}\"", table_name);
        let row_count = match sqlx::query(&count_query).fetch_one(&self.pool).await {
            Ok(row) => Some(row.get::<i64, _>(0)),
            Err(_) => None,
        };

        // Get table size
        let size_query = format!("SELECT pg_total_relation_size('\"{}\"')", table_name);
        let size_mb = match sqlx::query(&size_query).fetch_one(&self.pool).await {
            Ok(row) => {
                let size_bytes: i64 = row.get(0);
                Some(size_bytes as f64 / (1024.0 * 1024.0))
            },
            Err(_) => None,
        };

        Ok((row_count, size_mb))
    }

    async fn get_table_columns(&self, table_name: &str) -> DatabaseResult<Vec<ColumnInfo>> {
        let columns_query = r#"
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable = 'YES' as is_nullable,
                c.column_default,
                c.character_maximum_length,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.table_name = $1 
                AND tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name
            WHERE c.table_name = $1
            ORDER BY c.ordinal_position
        "#;

        let rows = sqlx::query(columns_query)
            .bind(table_name.to_lowercase())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: columns_query.to_string(),
                cause: e.to_string(),
            })?;

        let mut columns = Vec::new();
        for row in rows {
            columns.push(ColumnInfo {
                name: row.get("column_name"),
                data_type: row.get("data_type"),
                is_nullable: row.get("is_nullable"),
                default_value: row.get("column_default"),
                is_primary_key: row.get("is_primary_key"),
                character_maximum_length: row.get("character_maximum_length"),
            });
        }

        Ok(columns)
    }

    async fn get_table_indexes(&self, table_name: &str) -> DatabaseResult<Vec<IndexInfo>> {
        let indexes_query = r#"
            SELECT 
                i.indexname as index_name,
                i.indexdef as index_definition,
                ix.indisunique as is_unique,
                am.amname as index_type,
                array_agg(a.attname ORDER BY a.attnum) as columns
            FROM pg_indexes i
            JOIN pg_class c ON c.relname = i.tablename
            JOIN pg_index ix ON ix.indexrelid = (
                SELECT oid FROM pg_class WHERE relname = i.indexname
            )
            JOIN pg_am am ON am.oid = (
                SELECT pg_class.relam FROM pg_class WHERE relname = i.indexname
            )
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(ix.indkey)
            WHERE i.tablename = $1
            GROUP BY i.indexname, i.indexdef, ix.indisunique, am.amname
            ORDER BY i.indexname
        "#;

        let rows = sqlx::query(indexes_query)
            .bind(table_name.to_lowercase())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: indexes_query.to_string(),
                cause: e.to_string(),
            })?;

        let mut indexes = Vec::new();
        for row in rows {
            let columns: Vec<String> = row.get::<Vec<String>, _>("columns");
            indexes.push(IndexInfo {
                name: row.get("index_name"),
                columns,
                is_unique: row.get("is_unique"),
                index_type: row.get("index_type"),
            });
        }

        Ok(indexes)
    }

    async fn get_table_constraints(&self, table_name: &str) -> DatabaseResult<Vec<ConstraintInfo>> {
        let constraints_query = r#"
            SELECT 
                tc.constraint_name,
                tc.constraint_type,
                array_agg(kcu.column_name) as columns,
                cc.check_clause as definition
            FROM information_schema.table_constraints tc
            LEFT JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            LEFT JOIN information_schema.check_constraints cc
                ON tc.constraint_name = cc.constraint_name
            WHERE tc.table_name = $1
            GROUP BY tc.constraint_name, tc.constraint_type, cc.check_clause
            ORDER BY tc.constraint_name
        "#;

        let rows = sqlx::query(constraints_query)
            .bind(table_name.to_lowercase())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DatabaseError::QueryFailed {
                query: constraints_query.to_string(),
                cause: e.to_string(),
            })?;

        let mut constraints = Vec::new();
        for row in rows {
            let columns: Vec<String> = row.get::<Vec<String>, _>("columns");
            constraints.push(ConstraintInfo {
                name: row.get("constraint_name"),
                constraint_type: row.get("constraint_type"),
                columns,
                definition: row.get("definition"),
            });
        }

        Ok(constraints)
    }

    fn extract_column_value(&self, row: &sqlx::postgres::PgRow, column_index: usize) -> DatabaseResult<String> {
        let column = &row.columns()[column_index];
        let column_name = column.name();

        // Handle different PostgreSQL types
        let type_name = format!("{:?}", column.type_info());
        let value = if type_name.contains("TEXT") || type_name.contains("VARCHAR") || type_name.contains("CHAR") {
            row.try_get::<Option<String>, _>(column_index)
                .map_err(|e| DatabaseError::QueryFailed {
                    query: format!("extract column {}", column_name),
                    cause: e.to_string(),
                })?
                .unwrap_or_default()
        } else if type_name.contains("INT4") || type_name.contains("INTEGER") {
            row.try_get::<Option<i32>, _>(column_index)
                .map_err(|e| DatabaseError::QueryFailed {
                    query: format!("extract column {}", column_name),
                    cause: e.to_string(),
                })?
                .map(|v| v.to_string())
                .unwrap_or_default()
        } else if type_name.contains("INT8") || type_name.contains("BIGINT") {
            row.try_get::<Option<i64>, _>(column_index)
                .map_err(|e| DatabaseError::QueryFailed {
                    query: format!("extract column {}", column_name),
                    cause: e.to_string(),
                })?
                .map(|v| v.to_string())
                .unwrap_or_default()
        } else if type_name.contains("TIMESTAMP") {
            row.try_get::<Option<DateTime<Utc>>, _>(column_index)
                .map_err(|e| DatabaseError::QueryFailed {
                    query: format!("extract column {}", column_name),
                    cause: e.to_string(),
                })?
                .map(|v| v.to_rfc3339())
                .unwrap_or_default()
        } else if type_name.contains("BOOL") {
            row.try_get::<Option<bool>, _>(column_index)
                .map_err(|e| DatabaseError::QueryFailed {
                    query: format!("extract column {}", column_name),
                    cause: e.to_string(),
                })?
                .map(|v| v.to_string())
                .unwrap_or_default()
        } else {
            // Fallback: try to get as string
            row.try_get::<Option<String>, _>(column_index)
                .map_err(|e| DatabaseError::QueryFailed {
                    query: format!("extract column {}", column_name),
                    cause: e.to_string(),
                })?
                .unwrap_or_else(|| format!("[{}]", type_name))
        };

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool() -> Option<PgPool> {
        // Only run tests if DATABASE_URL is set
        std::env::var("DATABASE_URL").ok().and_then(|url| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                PgPool::connect(&url).await.ok()
            })
        })
    }

    #[tokio::test]
    async fn test_database_explorer_creation() {
        if let Some(pool) = create_test_pool() {
            let _explorer = DatabaseExplorer::new(pool);
            // Just test that we can create the explorer
            assert!(true);
        }
    }

    #[tokio::test]
    async fn test_database_info() {
        if let Some(pool) = create_test_pool() {
            let explorer = DatabaseExplorer::new(pool);
            
            let info = explorer.get_database_info().await.unwrap();
            assert_eq!(info.connection_status, "Connected");
            assert!(!info.database_name.is_empty());
            assert!(!info.server_version.is_empty());
            assert!(info.connection_time_ms > 0);
        }
    }

    #[tokio::test]
    async fn test_list_tables() {
        if let Some(pool) = create_test_pool() {
            let explorer = DatabaseExplorer::new(pool);
            
            // Initialize schema to ensure we have some tables
            explorer.schema_manager.initialize_schema().await.unwrap();
            
            let tables = explorer.list_tables(None).await.unwrap();
            assert!(!tables.is_empty());
            
            // Should have at least the ingestion_meta table
            let meta_table = tables.iter().find(|t| t.name == "ingestion_meta");
            assert!(meta_table.is_some());
        }
    }

    #[tokio::test]
    async fn test_table_sample() {
        if let Some(pool) = create_test_pool() {
            let explorer = DatabaseExplorer::new(pool);
            
            // Initialize schema
            explorer.schema_manager.initialize_schema().await.unwrap();
            
            // Sample the ingestion_meta table (should exist but be empty)
            let sample = explorer.sample_table("ingestion_meta", 5).await.unwrap();
            assert_eq!(sample.table_name, "ingestion_meta");
            assert!(sample.columns.len() > 0);
            assert_eq!(sample.total_rows, 0); // Should be empty initially
        }
    }

    #[tokio::test]
    async fn test_describe_table() {
        if let Some(pool) = create_test_pool() {
            let explorer = DatabaseExplorer::new(pool);
            
            // Initialize schema
            explorer.schema_manager.initialize_schema().await.unwrap();
            
            // Describe the ingestion_meta table
            let schema = explorer.describe_table("ingestion_meta").await.unwrap();
            assert_eq!(schema.table_name, "ingestion_meta");
            assert!(!schema.columns.is_empty());
            
            // Should have primary key column
            let pk_column = schema.columns.iter().find(|c| c.is_primary_key);
            assert!(pk_column.is_some());
        }
    }

    #[tokio::test]
    async fn test_nonexistent_table_operations() {
        if let Some(pool) = create_test_pool() {
            let explorer = DatabaseExplorer::new(pool);
            
            // Test operations on nonexistent table
            let sample_result = explorer.sample_table("nonexistent_table", 5).await;
            assert!(sample_result.is_err());
            
            let describe_result = explorer.describe_table("nonexistent_table").await;
            assert!(describe_result.is_err());
        }
    }

    #[test]
    fn test_format_database_info() {
        // Create a mock pool for testing - we'll just test the formatting logic
        let pool = match std::env::var("DATABASE_URL") {
            Ok(url) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                match rt.block_on(sqlx::PgPool::connect(&url)) {
                    Ok(p) => p,
                    Err(_) => return, // Skip test if no database
                }
            }
            Err(_) => return, // Skip test if no DATABASE_URL
        };
        
        let explorer = DatabaseExplorer::new(pool);
        
        let info = DatabaseInfo {
            connection_status: "Connected".to_string(),
            database_name: "test_db".to_string(),
            server_version: "PostgreSQL 14.0".to_string(),
            total_tables: 10,
            ingestion_tables: 5,
            query_result_tables: 3,
            total_size_mb: 125.5,
            connection_time_ms: 25,
        };
        
        let formatted = explorer.format_database_info(&info);
        assert!(formatted.contains("Connected"));
        assert!(formatted.contains("test_db"));
        assert!(formatted.contains("PostgreSQL 14.0"));
        assert!(formatted.contains("10"));
        assert!(formatted.contains("125.5"));
    }

    #[test]
    fn test_format_table_list() {
        // Create a mock pool for testing - we'll just test the formatting logic
        let pool = match std::env::var("DATABASE_URL") {
            Ok(url) => {
                let rt = tokio::runtime::Runtime::new().unwrap();
                match rt.block_on(sqlx::PgPool::connect(&url)) {
                    Ok(p) => p,
                    Err(_) => return, // Skip test if no database
                }
            }
            Err(_) => return, // Skip test if no DATABASE_URL
        };
        
        let explorer = DatabaseExplorer::new(pool);
        
        let tables = vec![
            TableListItem {
                name: "INGEST_20250927143022".to_string(),
                table_type: "Ingestion".to_string(),
                row_count: Some(100),
                size_mb: Some(5.2),
                created_at: Some(Utc::now()),
                description: Some("Test table".to_string()),
            },
            TableListItem {
                name: "ingestion_meta".to_string(),
                table_type: "Meta".to_string(),
                row_count: Some(1),
                size_mb: Some(0.1),
                created_at: Some(Utc::now()),
                description: Some("Metadata table".to_string()),
            },
        ];
        
        let formatted = explorer.format_table_list(&tables, true);
        assert!(formatted.contains("INGEST_20250927143022"));
        assert!(formatted.contains("ingestion_meta"));
        assert!(formatted.contains("100 rows"));
        assert!(formatted.contains("Total: 2 tables"));
    }

    #[test]
    fn test_column_info_structure() {
        let column = ColumnInfo {
            name: "test_column".to_string(),
            data_type: "VARCHAR".to_string(),
            is_nullable: false,
            default_value: Some("'default'".to_string()),
            is_primary_key: true,
            character_maximum_length: Some(255),
        };
        
        assert_eq!(column.name, "test_column");
        assert_eq!(column.data_type, "VARCHAR");
        assert!(!column.is_nullable);
        assert!(column.is_primary_key);
        assert_eq!(column.character_maximum_length, Some(255));
    }
}