use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "code-ingest")]
#[command(about = "A Rust-based code ingestion system for PostgreSQL")]
#[command(version = "0.1.0")]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Database path for PostgreSQL connection
    #[arg(long, global = true)]
    db_path: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Ingest a GitHub repository or local folder
    Ingest {
        /// GitHub repository URL or local folder path
        source: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Execute SQL query and output results
    Sql {
        /// SQL query to execute
        query: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Prepare query results for IDE task processing
    QueryPrepare {
        /// SQL query to execute
        query: String,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Temporary file path for query results
        #[arg(long)]
        temp_path: PathBuf,
        /// Tasks file path for IDE integration
        #[arg(long)]
        tasks_file: PathBuf,
        /// Output table name for storing analysis results
        #[arg(long)]
        output_table: String,
    },

    /// Generate structured task markdown for systematic analysis
    GenerateTasks {
        /// SQL query to get data for task generation
        #[arg(long)]
        sql: String,
        /// Prompt file for analysis instructions
        #[arg(long)]
        prompt_file: PathBuf,
        /// Output table name for storing results
        #[arg(long)]
        output_table: String,
        /// Tasks file path for generated markdown
        #[arg(long)]
        tasks_file: PathBuf,
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Store analysis results back to database
    StoreResult {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Output table name for storing results
        #[arg(long)]
        output_table: String,
        /// Result file containing analysis output
        #[arg(long)]
        result_file: PathBuf,
        /// Original SQL query that generated the data
        #[arg(long)]
        original_query: String,
    },

    /// Export query results as individual markdown files
    PrintToMd {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to query
        #[arg(long)]
        table: String,
        /// SQL query to execute
        #[arg(long)]
        sql: String,
        /// Prefix for generated markdown files
        #[arg(long)]
        prefix: String,
        /// Location directory for markdown files
        #[arg(long)]
        location: PathBuf,
    },

    /// List all tables in the database
    ListTables {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Show sample data from a table
    Sample {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to sample
        #[arg(long)]
        table: String,
        /// Number of rows to show
        #[arg(long, default_value = "5")]
        limit: usize,
    },

    /// Show table schema information
    Describe {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
        /// Table name to describe
        #[arg(long)]
        table: String,
    },

    /// Show database connection info
    DbInfo {
        /// Database path (can also be set globally)
        #[arg(long)]
        db_path: Option<PathBuf>,
    },

    /// Show PostgreSQL setup instructions
    PgStart,
}

impl Default for Cli {
    fn default() -> Self {
        Self::new()
    }
}

impl Cli {
    pub fn new() -> Self {
        Self::parse()
    }

    pub async fn run(&self) -> Result<()> {
        match &self.command {
            Some(command) => self.execute_command(command).await,
            None => {
                // If no subcommand is provided, try to parse as ingest command
                // This handles the case: code-ingest <url> --db-path <path>
                let args: Vec<String> = std::env::args().collect();
                if args.len() >= 2 && !args[1].starts_with('-') {
                    let source = args[1].clone();
                    let db_path = self.db_path.clone();
                    self.execute_ingest(source, db_path).await
                } else {
                    println!("Code ingestion tool");
                    println!("Use --help for usage information");
                    Ok(())
                }
            }
        }
    }

    async fn execute_command(&self, command: &Commands) -> Result<()> {
        match command {
            Commands::Ingest { source, db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_ingest(source.clone(), db_path).await
            }
            Commands::Sql { query, db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_sql(query.clone(), db_path).await
            }
            Commands::QueryPrepare {
                query,
                db_path,
                temp_path,
                tasks_file,
                output_table,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_query_prepare(
                    query.clone(),
                    db_path,
                    temp_path.clone(),
                    tasks_file.clone(),
                    output_table.clone(),
                )
                .await
            }
            Commands::GenerateTasks {
                sql,
                prompt_file,
                output_table,
                tasks_file,
                db_path,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_generate_tasks(
                    sql.clone(),
                    prompt_file.clone(),
                    output_table.clone(),
                    tasks_file.clone(),
                    db_path,
                )
                .await
            }
            Commands::StoreResult {
                db_path,
                output_table,
                result_file,
                original_query,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_store_result(
                    db_path,
                    output_table.clone(),
                    result_file.clone(),
                    original_query.clone(),
                )
                .await
            }
            Commands::PrintToMd {
                db_path,
                table,
                sql,
                prefix,
                location,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_print_to_md(
                    db_path,
                    table.clone(),
                    sql.clone(),
                    prefix.clone(),
                    location.clone(),
                )
                .await
            }
            Commands::ListTables { db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_list_tables(db_path).await
            }
            Commands::Sample {
                db_path,
                table,
                limit,
            } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_sample(db_path, table.clone(), *limit).await
            }
            Commands::Describe { db_path, table } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_describe(db_path, table.clone()).await
            }
            Commands::DbInfo { db_path } => {
                let db_path = db_path.clone().or_else(|| self.db_path.clone());
                self.execute_db_info(db_path).await
            }
            Commands::PgStart => self.execute_pg_start().await,
        }
    }

    // Placeholder implementations for all commands
    async fn execute_ingest(&self, source: String, db_path: Option<PathBuf>) -> Result<()> {
        println!("Ingesting source: {}", source);
        if let Some(path) = db_path {
            println!("Database path: {}", path.display());
        }
        println!("Implementation pending - Task 4");
        Ok(())
    }

    async fn execute_sql(&self, query: String, db_path: Option<PathBuf>) -> Result<()> {
        println!("Executing SQL: {}", query);
        if let Some(path) = db_path {
            println!("Database path: {}", path.display());
        }
        println!("Implementation pending - Task 2");
        Ok(())
    }

    async fn execute_query_prepare(
        &self,
        query: String,
        db_path: Option<PathBuf>,
        temp_path: PathBuf,
        tasks_file: PathBuf,
        output_table: String,
    ) -> Result<()> {
        println!("Query prepare:");
        println!("  Query: {}", query);
        println!("  Temp path: {}", temp_path.display());
        println!("  Tasks file: {}", tasks_file.display());
        println!("  Output table: {}", output_table);
        if let Some(path) = db_path {
            println!("  Database path: {}", path.display());
        }
        println!("Implementation pending - Task 5");
        Ok(())
    }

    async fn execute_generate_tasks(
        &self,
        sql: String,
        prompt_file: PathBuf,
        output_table: String,
        tasks_file: PathBuf,
        db_path: Option<PathBuf>,
    ) -> Result<()> {
        println!("Generate tasks:");
        println!("  SQL: {}", sql);
        println!("  Prompt file: {}", prompt_file.display());
        println!("  Output table: {}", output_table);
        println!("  Tasks file: {}", tasks_file.display());
        if let Some(path) = db_path {
            println!("  Database path: {}", path.display());
        }
        println!("Implementation pending - Task 6");
        Ok(())
    }

    async fn execute_store_result(
        &self,
        db_path: Option<PathBuf>,
        output_table: String,
        result_file: PathBuf,
        original_query: String,
    ) -> Result<()> {
        println!("Store result:");
        println!("  Output table: {}", output_table);
        println!("  Result file: {}", result_file.display());
        println!("  Original query: {}", original_query);
        if let Some(path) = db_path {
            println!("  Database path: {}", path.display());
        }
        println!("Implementation pending - Task 5");
        Ok(())
    }

    async fn execute_print_to_md(
        &self,
        db_path: Option<PathBuf>,
        table: String,
        sql: String,
        prefix: String,
        location: PathBuf,
    ) -> Result<()> {
        println!("Print to MD:");
        println!("  Table: {}", table);
        println!("  SQL: {}", sql);
        println!("  Prefix: {}", prefix);
        println!("  Location: {}", location.display());
        if let Some(path) = db_path {
            println!("  Database path: {}", path.display());
        }
        println!("Implementation pending - Task 7");
        Ok(())
    }

    async fn execute_list_tables(&self, db_path: Option<PathBuf>) -> Result<()> {
        println!("List tables");
        if let Some(path) = db_path {
            println!("Database path: {}", path.display());
        }
        println!("Implementation pending - Task 7");
        Ok(())
    }

    async fn execute_sample(
        &self,
        db_path: Option<PathBuf>,
        table: String,
        limit: usize,
    ) -> Result<()> {
        println!("Sample table: {} (limit: {})", table, limit);
        if let Some(path) = db_path {
            println!("Database path: {}", path.display());
        }
        println!("Implementation pending - Task 7");
        Ok(())
    }

    async fn execute_describe(&self, db_path: Option<PathBuf>, table: String) -> Result<()> {
        println!("Describe table: {}", table);
        if let Some(path) = db_path {
            println!("Database path: {}", path.display());
        }
        println!("Implementation pending - Task 7");
        Ok(())
    }

    async fn execute_db_info(&self, db_path: Option<PathBuf>) -> Result<()> {
        println!("Database info");
        if let Some(path) = db_path {
            println!("Database path: {}", path.display());
        }
        println!("Implementation pending - Task 7");
        Ok(())
    }

    async fn execute_pg_start(&self) -> Result<()> {
        println!("PostgreSQL Setup Instructions");
        println!("============================");
        println!();
        println!("macOS (using Homebrew):");
        println!("  brew install postgresql");
        println!("  brew services start postgresql");
        println!();
        println!("Ubuntu/Debian:");
        println!("  sudo apt update");
        println!("  sudo apt install postgresql postgresql-contrib");
        println!("  sudo systemctl start postgresql");
        println!();
        println!("Create database:");
        println!("  createdb code_ingest");
        println!();
        println!("Set DATABASE_URL environment variable:");
        println!("  export DATABASE_URL=postgresql://username:password@localhost:5432/code_ingest");
        println!();
        println!("Test connection:");
        println!("  psql $DATABASE_URL");
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_parsing_ingest_command() {
        let args = vec![
            "code-ingest",
            "ingest",
            "https://github.com/test/repo",
            "--db-path",
            "/tmp/test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Ingest { source, db_path }) => {
                assert_eq!(source, "https://github.com/test/repo");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected Ingest command"),
        }
    }

    #[test]
    fn test_cli_parsing_sql_command() {
        let args = vec![
            "code-ingest",
            "sql",
            "SELECT * FROM test",
            "--db-path",
            "/tmp/test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Sql { query, db_path }) => {
                assert_eq!(query, "SELECT * FROM test");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected Sql command"),
        }
    }

    #[test]
    fn test_cli_parsing_query_prepare_command() {
        let args = vec![
            "code-ingest",
            "query-prepare",
            "SELECT * FROM test",
            "--db-path",
            "/tmp/test",
            "--temp-path",
            "/tmp/temp.txt",
            "--tasks-file",
            "/tmp/tasks.md",
            "--output-table",
            "QUERYRESULT_test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::QueryPrepare {
                query,
                db_path,
                temp_path,
                tasks_file,
                output_table,
            }) => {
                assert_eq!(query, "SELECT * FROM test");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(temp_path.to_str().unwrap(), "/tmp/temp.txt");
                assert_eq!(tasks_file.to_str().unwrap(), "/tmp/tasks.md");
                assert_eq!(output_table, "QUERYRESULT_test");
            }
            _ => panic!("Expected QueryPrepare command"),
        }
    }

    #[test]
    fn test_cli_parsing_generate_tasks_command() {
        let args = vec![
            "code-ingest",
            "generate-tasks",
            "--sql",
            "SELECT * FROM test",
            "--prompt-file",
            "/tmp/prompt.md",
            "--output-table",
            "QUERYRESULT_test",
            "--tasks-file",
            "/tmp/tasks.md",
            "--db-path",
            "/tmp/test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::GenerateTasks {
                sql,
                prompt_file,
                output_table,
                tasks_file,
                db_path,
            }) => {
                assert_eq!(sql, "SELECT * FROM test");
                assert_eq!(prompt_file.to_str().unwrap(), "/tmp/prompt.md");
                assert_eq!(output_table, "QUERYRESULT_test");
                assert_eq!(tasks_file.to_str().unwrap(), "/tmp/tasks.md");
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected GenerateTasks command"),
        }
    }

    #[test]
    fn test_cli_parsing_print_to_md_command() {
        let args = vec![
            "code-ingest",
            "print-to-md",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
            "--sql",
            "SELECT * FROM test_table",
            "--prefix",
            "analysis",
            "--location",
            "/tmp/md-files",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::PrintToMd {
                db_path,
                table,
                sql,
                prefix,
                location,
            }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(table, "test_table");
                assert_eq!(sql, "SELECT * FROM test_table");
                assert_eq!(prefix, "analysis");
                assert_eq!(location.to_str().unwrap(), "/tmp/md-files");
            }
            _ => panic!("Expected PrintToMd command"),
        }
    }

    #[test]
    fn test_cli_parsing_list_tables_command() {
        let args = vec!["code-ingest", "list-tables", "--db-path", "/tmp/test"];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::ListTables { db_path }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected ListTables command"),
        }
    }

    #[test]
    fn test_cli_parsing_sample_command() {
        let args = vec![
            "code-ingest",
            "sample",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
            "--limit",
            "10",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Sample {
                db_path,
                table,
                limit,
            }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(table, "test_table");
                assert_eq!(*limit, 10);
            }
            _ => panic!("Expected Sample command"),
        }
    }

    #[test]
    fn test_cli_parsing_describe_command() {
        let args = vec![
            "code-ingest",
            "describe",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Describe { db_path, table }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(table, "test_table");
            }
            _ => panic!("Expected Describe command"),
        }
    }

    #[test]
    fn test_cli_parsing_db_info_command() {
        let args = vec!["code-ingest", "db-info", "--db-path", "/tmp/test"];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::DbInfo { db_path }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
            }
            _ => panic!("Expected DbInfo command"),
        }
    }

    #[test]
    fn test_cli_parsing_pg_start_command() {
        let args = vec!["code-ingest", "pg-start"];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::PgStart) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected PgStart command"),
        }
    }

    #[test]
    fn test_cli_parsing_global_db_path() {
        let args = vec![
            "code-ingest",
            "--db-path",
            "/global/path",
            "list-tables",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        // Global db_path should be set
        assert_eq!(cli.db_path.as_ref().unwrap().to_str().unwrap(), "/global/path");
        
        // Command should be parsed correctly
        match &cli.command {
            Some(Commands::ListTables { .. }) => {
                // Success - command parsed correctly
            }
            _ => panic!("Expected ListTables command"),
        }
    }

    #[test]
    fn test_cli_parsing_store_result_command() {
        let args = vec![
            "code-ingest",
            "store-result",
            "--db-path",
            "/tmp/test",
            "--output-table",
            "QUERYRESULT_test",
            "--result-file",
            "/tmp/result.txt",
            "--original-query",
            "SELECT * FROM test",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::StoreResult {
                db_path,
                output_table,
                result_file,
                original_query,
            }) => {
                assert_eq!(db_path.as_ref().unwrap().to_str().unwrap(), "/tmp/test");
                assert_eq!(output_table, "QUERYRESULT_test");
                assert_eq!(result_file.to_str().unwrap(), "/tmp/result.txt");
                assert_eq!(original_query, "SELECT * FROM test");
            }
            _ => panic!("Expected StoreResult command"),
        }
    }

    #[test]
    fn test_cli_validation_missing_required_args() {
        // Test that missing required arguments cause parsing to fail
        let args = vec!["code-ingest", "ingest"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err()); // Missing SOURCE argument

        let args = vec!["code-ingest", "sql"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err()); // Missing QUERY argument

        // Note: --db-path is optional since it can be set globally or via environment
        // So we test missing required positional arguments instead
        let args = vec!["code-ingest", "sample"];
        let result = Cli::try_parse_from(args);
        assert!(result.is_err()); // Missing --table argument
    }

    #[test]
    fn test_sample_command_default_limit() {
        let args = vec![
            "code-ingest",
            "sample",
            "--db-path",
            "/tmp/test",
            "--table",
            "test_table",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match &cli.command {
            Some(Commands::Sample { limit, .. }) => {
                assert_eq!(*limit, 5); // Default value
            }
            _ => panic!("Expected Sample command"),
        }
    }
}