use pensieve_validator::{
    ValidationOrchestrator,
    MetricsCollector, ErrorCategory, ErrorSeverity, ErrorContext, SystemState,
    UXEventType, UXQualityScores, DatabaseOperation,
    cli_config::{CliConfig, ConfigError},
    directory_analyzer::DirectoryAnalyzer,
    chaos_detector::ChaosDetector,
    report_generator::ReportGenerator,
    ComparativeAnalyzer, HistoricalReportGenerator, ExportFormat,
    comparative_analyzer::SystemInfo,
};
use std::path::PathBuf;
use std::time::Duration;
use clap::{Parser, Subcommand, ValueEnum};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "pensieve-validator")]
#[command(about = "A comprehensive validation framework for pensieve and other CLI tools")]
#[command(version = "0.1.0")]
#[command(author = "Pensieve Team")]
struct Cli {
    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,
    
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
    
    /// Quiet mode (minimal output)
    #[arg(short, long, global = true)]
    quiet: bool,
    
    /// Output format
    #[arg(long, global = true, value_enum, default_value = "human")]
    output_format: OutputFormat,
    
    /// Enable progress reporting
    #[arg(long, global = true)]
    progress: bool,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Human,
    Json,
    Yaml,
}

#[derive(Subcommand)]
enum Commands {
    /// Run comprehensive validation on a directory
    Validate {
        /// Target directory to validate
        #[arg(short, long)]
        directory: PathBuf,
        
        /// Path to pensieve binary
        #[arg(short, long)]
        pensieve_binary: Option<PathBuf>,
        
        /// Output database path
        #[arg(short, long)]
        output_db: Option<PathBuf>,
        
        /// Timeout in seconds
        #[arg(short, long)]
        timeout: Option<u64>,
        
        /// Memory limit in MB
        #[arg(short, long)]
        memory_limit: Option<u64>,
        
        /// Output directory for reports
        #[arg(long)]
        output_dir: Option<PathBuf>,
        
        /// Skip chaos detection
        #[arg(long)]
        skip_chaos_detection: bool,
        
        /// Skip deduplication analysis
        #[arg(long)]
        skip_deduplication: bool,
        
        /// Skip UX analysis
        #[arg(long)]
        skip_ux_analysis: bool,
        
        /// Enable real-time monitoring
        #[arg(long)]
        real_time: bool,
    },
    
    /// Analyze directory structure and detect chaos
    AnalyzeDirectory {
        /// Target directory to analyze
        #[arg(short, long)]
        directory: PathBuf,
        
        /// Output file for analysis results
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Include detailed file analysis
        #[arg(long)]
        detailed: bool,
        
        /// Maximum depth to analyze
        #[arg(long)]
        max_depth: Option<usize>,
        
        /// Minimum file size to analyze (bytes)
        #[arg(long)]
        min_file_size: Option<u64>,
    },
    
    /// Generate reports from validation results
    GenerateReport {
        /// Input validation results file or directory
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output directory for generated reports
        #[arg(short, long)]
        output_dir: PathBuf,
        
        /// Report format
        #[arg(short, long, value_enum, default_value = "html")]
        format: ReportFormat,
        
        /// Include raw data in reports
        #[arg(long)]
        include_raw_data: bool,
        
        /// Template directory for custom report templates
        #[arg(long)]
        template_dir: Option<PathBuf>,
    },
    
    /// Compare validation results across multiple runs
    CompareRuns {
        /// Baseline validation results
        #[arg(short, long)]
        baseline: PathBuf,
        
        /// Current validation results to compare
        #[arg(short, long)]
        current: PathBuf,
        
        /// Output file for comparison report
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Threshold for significant changes (percentage)
        #[arg(long, default_value = "5.0")]
        threshold: f64,
        
        /// Focus on specific metrics
        #[arg(long)]
        focus_metrics: Option<Vec<String>>,
    },
    
    /// Establish a baseline from validation results
    EstablishBaseline {
        /// Validation results to use as baseline
        #[arg(short, long)]
        results: PathBuf,
        
        /// Baseline identifier
        #[arg(short, long)]
        baseline_id: String,
        
        /// Dataset path for the baseline
        #[arg(short, long)]
        dataset_path: PathBuf,
        
        /// System information file (JSON)
        #[arg(long)]
        system_info: Option<PathBuf>,
        
        /// Storage directory for baselines
        #[arg(long)]
        storage_dir: Option<PathBuf>,
    },
    
    /// Compare against established baseline
    CompareBaseline {
        /// Current validation results
        #[arg(short, long)]
        results: PathBuf,
        
        /// Baseline identifier to compare against
        #[arg(short, long)]
        baseline_id: String,
        
        /// Output file for comparison report
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Storage directory for baselines
        #[arg(long)]
        storage_dir: Option<PathBuf>,
    },
    
    /// Analyze historical trends across multiple validation runs
    AnalyzeTrends {
        /// Dataset path to analyze trends for
        #[arg(short, long)]
        dataset_path: PathBuf,
        
        /// Analysis period in days
        #[arg(short, long, default_value = "30")]
        period_days: i64,
        
        /// Output directory for trend analysis
        #[arg(short, long)]
        output_dir: PathBuf,
        
        /// Storage directory for historical data
        #[arg(long)]
        storage_dir: Option<PathBuf>,
    },
    
    /// Generate historical report with trend visualization
    GenerateHistoricalReport {
        /// Dataset path for the report
        #[arg(short, long)]
        dataset_path: PathBuf,
        
        /// Analysis period in days
        #[arg(short, long, default_value = "30")]
        period_days: i64,
        
        /// Output directory for the report
        #[arg(short, long)]
        output_dir: PathBuf,
        
        /// Storage directory for historical data
        #[arg(long)]
        storage_dir: Option<PathBuf>,
        
        /// Export formats
        #[arg(long, value_enum)]
        formats: Option<Vec<HistoricalReportFormat>>,
    },
    
    /// List available baselines
    ListBaselines {
        /// Storage directory for baselines
        #[arg(long)]
        storage_dir: Option<PathBuf>,
    },
    
    /// Delete a baseline
    DeleteBaseline {
        /// Baseline identifier to delete
        #[arg(short, long)]
        baseline_id: String,
        
        /// Storage directory for baselines
        #[arg(long)]
        storage_dir: Option<PathBuf>,
        
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
    
    /// Generate default configuration file
    InitConfig {
        /// Output path for configuration file
        #[arg(short, long, default_value = "pensieve-validator.toml")]
        output: PathBuf,
        
        /// Overwrite existing configuration file
        #[arg(long)]
        force: bool,
    },
    
    /// Validate configuration file
    ValidateConfig {
        /// Configuration file to validate
        #[arg(short, long, default_value = "pensieve-validator.toml")]
        config: PathBuf,
    },
    
    /// Demonstrate metrics collection capabilities
    Demo {
        /// Duration of demo in seconds
        #[arg(short, long, default_value = "10")]
        duration: u64,
        
        /// Enable real-time display
        #[arg(long)]
        real_time: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum ReportFormat {
    Html,
    Json,
    Csv,
    Markdown,
}

#[derive(Clone, ValueEnum)]
enum HistoricalReportFormat {
    Html,
    Json,
    Csv,
    Excel,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize logging based on verbosity
    init_logging(&cli)?;
    
    // Load configuration if specified
    let config = load_configuration(&cli).await?;
    
    // Clone CLI values to avoid borrow checker issues
    let verbose = cli.verbose;
    let quiet = cli.quiet;
    let output_format = cli.output_format.clone();
    let progress = cli.progress;
    
    match cli.command {
        Commands::Validate {
            directory,
            pensieve_binary,
            output_db,
            timeout,
            memory_limit,
            output_dir,
            skip_chaos_detection,
            skip_deduplication,
            skip_ux_analysis,
            real_time,
        } => {
            run_validation(
                verbose,
                quiet,
                output_format,
                progress,
                &config,
                directory,
                pensieve_binary,
                output_db,
                timeout,
                memory_limit,
                output_dir,
                skip_chaos_detection,
                skip_deduplication,
                skip_ux_analysis,
                real_time,
            ).await?;
        }
        
        Commands::AnalyzeDirectory {
            directory,
            output,
            detailed,
            max_depth,
            min_file_size,
        } => {
            run_directory_analysis(
                verbose,
                quiet,
                output_format,
                directory,
                output,
                detailed,
                max_depth,
                min_file_size,
            ).await?;
        }
        
        Commands::GenerateReport {
            input,
            output_dir,
            format,
            include_raw_data,
            template_dir,
        } => {
            run_report_generation(
                verbose,
                quiet,
                input,
                output_dir,
                format,
                include_raw_data,
                template_dir,
            ).await?;
        }
        
        Commands::CompareRuns {
            baseline,
            current,
            output,
            threshold,
            focus_metrics,
        } => {
            run_comparison(
                verbose,
                quiet,
                output_format,
                baseline,
                current,
                output,
                threshold,
                focus_metrics,
            ).await?;
        }
        
        Commands::EstablishBaseline {
            results,
            baseline_id,
            dataset_path,
            system_info,
            storage_dir,
        } => {
            run_establish_baseline(
                verbose,
                quiet,
                results,
                baseline_id,
                dataset_path,
                system_info,
                storage_dir,
            ).await?;
        }
        
        Commands::CompareBaseline {
            results,
            baseline_id,
            output,
            storage_dir,
        } => {
            run_compare_baseline(
                verbose,
                quiet,
                output_format,
                results,
                baseline_id,
                output,
                storage_dir,
            ).await?;
        }
        
        Commands::AnalyzeTrends {
            dataset_path,
            period_days,
            output_dir,
            storage_dir,
        } => {
            run_analyze_trends(
                verbose,
                quiet,
                dataset_path,
                period_days,
                output_dir,
                storage_dir,
            ).await?;
        }
        
        Commands::GenerateHistoricalReport {
            dataset_path,
            period_days,
            output_dir,
            storage_dir,
            formats,
        } => {
            run_generate_historical_report(
                verbose,
                quiet,
                dataset_path,
                period_days,
                output_dir,
                storage_dir,
                formats,
            ).await?;
        }
        
        Commands::ListBaselines {
            storage_dir,
        } => {
            run_list_baselines(
                verbose,
                quiet,
                storage_dir,
            ).await?;
        }
        
        Commands::DeleteBaseline {
            baseline_id,
            storage_dir,
            force,
        } => {
            run_delete_baseline(
                verbose,
                quiet,
                baseline_id,
                storage_dir,
                force,
            ).await?;
        }
        
        Commands::InitConfig { output, force } => {
            run_init_config(output, force)?;
        }
        
        Commands::ValidateConfig { config } => {
            run_validate_config(config)?;
        }
        
        Commands::Demo { duration, real_time } => {
            run_metrics_demo(verbose, quiet, duration, real_time).await?;
        }
    }

    Ok(())
}

/// Initialize logging based on CLI arguments
fn init_logging(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let level = if cli.quiet {
        "error"
    } else if cli.verbose {
        "debug"
    } else {
        "info"
    };
    
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
        .format_timestamp_secs()
        .init();
    
    Ok(())
}

/// Load configuration from file or use defaults
async fn load_configuration(cli: &Cli) -> Result<CliConfig, Box<dyn std::error::Error>> {
    if let Some(config_path) = &cli.config {
        if !cli.quiet {
            println!("📋 Loading configuration from: {:?}", config_path);
        }
        
        match CliConfig::load_from_file(config_path) {
            Ok(mut config) => {
                config.validate().map_err(|e| ConfigError::ValidationFailed { message: e })?;
                Ok(config)
            }
            Err(e) => {
                eprintln!("❌ Failed to load configuration: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        Ok(CliConfig::default())
    }
}

/// Run comprehensive validation
async fn run_validation(
    verbose: bool,
    quiet: bool,
    output_format: OutputFormat,
    progress: bool,
    config: &CliConfig,
    directory: PathBuf,
    pensieve_binary: Option<PathBuf>,
    output_db: Option<PathBuf>,
    timeout: Option<u64>,
    memory_limit: Option<u64>,
    output_dir: Option<PathBuf>,
    _skip_deduplication: bool,
    _skip_ux_analysis: bool,
    skip_chaos_detection: bool,
    real_time: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("🔍 Starting comprehensive validation of: {:?}", directory);
    }
    
    // Verify directory exists
    if !directory.exists() {
        eprintln!("❌ Target directory does not exist: {:?}", directory);
        std::process::exit(1);
    }
    
    // Build validation configuration
    let mut validation_config = config.to_validation_orchestrator_config();
    
    // Override with CLI arguments
    if let Some(binary) = pensieve_binary {
        validation_config.pensieve_config.binary_path = binary;
    }
    if let Some(db) = output_db {
        validation_config.pensieve_config.output_database_path = db;
    }
    if let Some(timeout_secs) = timeout {
        validation_config.pensieve_config.timeout_seconds = timeout_secs;
    }
    if let Some(memory_mb) = memory_limit {
        validation_config.pensieve_config.memory_limit_mb = memory_mb;
        validation_config.performance_thresholds.max_memory_mb = memory_mb;
    }
    
    validation_config.pensieve_config.verbose_output = verbose;
    validation_config.enable_real_time_analysis = real_time;
    
    // Create orchestrator
    let orchestrator = ValidationOrchestrator::new(validation_config);
    
    // Show progress if enabled
    let progress_handle = if progress && !quiet {
        Some(start_progress_reporting())
    } else {
        None
    };
    
    // Pre-analyze directory for chaos detection if not skipped
    let chaos_report = if !skip_chaos_detection {
        let chaos_detector = ChaosDetector::new();
        chaos_detector.detect_chaos_files(&directory)?
    } else {
        pensieve_validator::types::ChaosReport {
            files_without_extensions: Vec::new(),
            misleading_extensions: Vec::new(),
            unicode_filenames: Vec::new(),
            extremely_large_files: Vec::new(),
            zero_byte_files: Vec::new(),
            permission_issues: Vec::new(),
            symlink_chains: Vec::new(),
            corrupted_files: Vec::new(),
            unusual_characters: Vec::new(),
            deep_nesting: Vec::new(),
        }
    };
    
    // Run validation
    match orchestrator.run_comprehensive_validation(&directory, &chaos_report).await {
        Ok(results) => {
            if let Some(handle) = progress_handle {
                handle.abort();
            }
            
            if !quiet {
                println!("\n✅ Validation completed successfully!");
                print_validation_summary(&results, &output_format);
            }
            
            // Save results if output directory specified
            if let Some(output_dir) = output_dir {
                save_validation_results(&results, &output_dir, config).await?;
            }
        }
        Err(e) => {
            if let Some(handle) = progress_handle {
                handle.abort();
            }
            eprintln!("\n❌ Validation failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Run directory analysis
async fn run_directory_analysis(
    verbose: bool,
    quiet: bool,
    output_format: OutputFormat,
    directory: PathBuf,
    output: Option<PathBuf>,
    _detailed: bool,
    _max_depth: Option<usize>,
    _min_file_size: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("📁 Analyzing directory structure: {:?}", directory);
    }
    
    if !directory.exists() {
        eprintln!("❌ Directory does not exist: {:?}", directory);
        std::process::exit(1);
    }
    
    let analyzer = DirectoryAnalyzer::new();
    let chaos_detector = ChaosDetector::new();
    
    // Run analysis
    let analysis = analyzer.analyze_directory(&directory)?;
    let chaos_report = chaos_detector.detect_chaos_files(&directory)?;
    
    // Print results
    if !quiet {
        print_directory_analysis(&analysis, &chaos_report, &output_format);
    }
    
    // Save results if output specified
    if let Some(output_path) = output {
        save_directory_analysis(&analysis, &chaos_report, &output_path, &output_format)?;
        if !quiet {
            println!("💾 Analysis results saved to: {:?}", output_path);
        }
    }
    
    Ok(())
}

/// Run report generation
async fn run_report_generation(
    verbose: bool,
    quiet: bool,
    input: PathBuf,
    output_dir: PathBuf,
    format: ReportFormat,
    include_raw_data: bool,
    _template_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("📊 Generating reports from: {:?}", input);
    }
    
    if !input.exists() {
        eprintln!("❌ Input file/directory does not exist: {:?}", input);
        std::process::exit(1);
    }
    
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir)?;
    
    let report_generator = ReportGenerator::new(Default::default());
    
    // Load validation results
    let results = load_validation_results(&input)?;
    
    // Generate reports based on format
    match format {
        ReportFormat::Html => {
            let report_path = output_dir.join("validation_report.html");
            // HTML report generation would be implemented here
            std::fs::write(&report_path, "HTML report placeholder")?;
            if !quiet {
                println!("📄 HTML report generated: {:?}", report_path);
            }
        }
        ReportFormat::Json => {
            let report_path = output_dir.join("validation_report.json");
            let json_content = serde_json::to_string_pretty(&results)?;
            std::fs::write(&report_path, json_content)?;
            if !quiet {
                println!("📄 JSON report generated: {:?}", report_path);
            }
        }
        ReportFormat::Csv => {
            // CSV report generation would be implemented here
            let csv_path = output_dir.join("validation_summary.csv");
            std::fs::write(&csv_path, "CSV report placeholder")?;
            if !quiet {
                println!("📄 CSV reports generated in: {:?}", output_dir);
            }
        }
        ReportFormat::Markdown => {
            let report_path = output_dir.join("validation_report.md");
            // Markdown report generation would be implemented here
            std::fs::write(&report_path, "# Validation Report\n\nMarkdown report placeholder")?;
            if !quiet {
                println!("📄 Markdown report generated: {:?}", report_path);
            }
        }
    }
    
    Ok(())
}

/// Run comparison between validation runs
async fn run_comparison(
    verbose: bool,
    quiet: bool,
    output_format: OutputFormat,
    baseline: PathBuf,
    current: PathBuf,
    output: Option<PathBuf>,
    threshold: f64,
    focus_metrics: Option<Vec<String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("🔄 Comparing validation results:");
        println!("   Baseline: {:?}", baseline);
        println!("   Current:  {:?}", current);
    }
    
    if !baseline.exists() {
        eprintln!("❌ Baseline file does not exist: {:?}", baseline);
        std::process::exit(1);
    }
    
    if !current.exists() {
        eprintln!("❌ Current file does not exist: {:?}", current);
        std::process::exit(1);
    }
    
    // Load both result sets
    let baseline_results = load_validation_results(&baseline)?;
    let current_results = load_validation_results(&current)?;
    
    // Perform comparison
    let comparison = compare_validation_results(&baseline_results, &current_results, threshold, focus_metrics)?;
    
    // Print comparison results
    if !quiet {
        print_comparison_results(&comparison, &output_format);
    }
    
    // Save comparison if output specified
    if let Some(output_path) = output {
        save_comparison_results(&comparison, &output_path, &output_format)?;
        if !quiet {
            println!("💾 Comparison results saved to: {:?}", output_path);
        }
    }
    
    Ok(())
}

/// Initialize configuration file
fn run_init_config(output: PathBuf, force: bool) -> Result<(), Box<dyn std::error::Error>> {
    if output.exists() && !force {
        eprintln!("❌ Configuration file already exists: {:?}", output);
        eprintln!("   Use --force to overwrite");
        std::process::exit(1);
    }
    
    CliConfig::generate_default_config_file(&output)?;
    println!("✅ Default configuration file created: {:?}", output);
    println!("   Edit this file to customize validation settings");
    
    Ok(())
}

/// Validate configuration file
fn run_validate_config(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !config_path.exists() {
        eprintln!("❌ Configuration file does not exist: {:?}", config_path);
        std::process::exit(1);
    }
    
    match CliConfig::load_from_file(&config_path) {
        Ok(config) => {
            match config.validate() {
                Ok(()) => {
                    println!("✅ Configuration file is valid: {:?}", config_path);
                }
                Err(e) => {
                    eprintln!("❌ Configuration validation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to load configuration: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Start progress reporting in background
fn start_progress_reporting() -> tokio::task::JoinHandle<()> {
    tokio::spawn(async {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        let spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let mut spinner_index = 0;
        
        loop {
            interval.tick().await;
            print!("\r{} Validation in progress...", spinner_chars[spinner_index]);
            io::stdout().flush().unwrap();
            spinner_index = (spinner_index + 1) % spinner_chars.len();
        }
    })
}

/// Demonstrate the metrics collection system capabilities
async fn run_metrics_demo(verbose: bool, quiet: bool, duration_secs: u64, _real_time: bool) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use tokio::sync::Mutex;
    
    let collector = Arc::new(Mutex::new(MetricsCollector::new()));
    
    println!("📊 Initializing metrics collection...");
    
    // Start metrics collection
    let (mut metrics_rx, metrics_handle) = {
        let collector_guard = collector.lock().await;
        collector_guard.start_collection().await?
    };
    
    // Simulate various activities
    let collector_clone = Arc::clone(&collector);
    let demo_handle = tokio::spawn(async move {
        let start_time = std::time::Instant::now();
        let mut files_processed = 0u64;
        let mut memory_usage = 100u64;
        
        while start_time.elapsed().as_secs() < duration_secs {
            // Simulate processing activity
            files_processed += 10;
            memory_usage += 5;
            
            // Record performance metrics
            {
                let collector_guard = collector_clone.lock().await;
                if let Err(e) = collector_guard.record_performance(files_processed, memory_usage, 25.0) {
                    eprintln!("Error recording performance: {}", e);
                }
            }
            
            // Simulate occasional errors
            if files_processed % 100 == 0 {
                let context = ErrorContext {
                    file_path: Some(format!("/test/file_{}.txt", files_processed)),
                    operation: "file_processing".to_string(),
                    system_state: SystemState {
                        memory_usage_mb: memory_usage,
                        cpu_usage_percent: 25.0,
                        files_processed,
                        processing_speed: 10.0,
                    },
                    preceding_events: vec!["Started processing batch".to_string()],
                };
                
                let collector_guard = collector_clone.lock().await;
                if let Err(e) = collector_guard.record_error(
                    ErrorCategory::FileSystem,
                    ErrorSeverity::Medium,
                    "Simulated processing error".to_string(),
                    context,
                ) {
                    eprintln!("Error recording error: {}", e);
                }
            }
            
            // Simulate UX events
            if files_processed % 50 == 0 {
                let quality_scores = UXQualityScores {
                    clarity: 0.8,
                    actionability: 0.7,
                    completeness: 0.9,
                    user_friendliness: 0.8,
                };
                
                let collector_guard = collector_clone.lock().await;
                if let Err(e) = collector_guard.record_ux_event(
                    UXEventType::ProgressUpdate,
                    format!("Processed {} files...", files_processed),
                    quality_scores,
                ) {
                    eprintln!("Error recording UX event: {}", e);
                }
            }
            
            // Simulate database operations
            {
                let collector_guard = collector_clone.lock().await;
                if let Err(e) = collector_guard.record_database_operation(
                    DatabaseOperation::Insert,
                    Duration::from_millis(25),
                    true,
                ) {
                    eprintln!("Error recording database operation: {}", e);
                }
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    
    // Monitor real-time metrics
    let monitor_handle = tokio::spawn(async move {
        let mut update_count = 0;
        while let Some(update) = metrics_rx.recv().await {
            update_count += 1;
            if update_count % 10 == 0 {
                println!("📈 Real-time update #{}: {:.1} files/sec, {} MB memory, {:.1}% CPU, UX: {:.1}/10",
                    update_count,
                    update.performance_snapshot.current_files_per_second,
                    update.performance_snapshot.current_memory_mb,
                    update.performance_snapshot.current_cpu_percent,
                    update.ux_snapshot.current_ux_score
                );
            }
        }
    });
    
    // Wait for demo to complete
    demo_handle.await?;
    
    // Stop monitoring
    metrics_handle.abort();
    monitor_handle.abort();
    
    // Generate final report
    println!("\n📋 Generating comprehensive metrics report...");
    let report = {
        let collector_guard = collector.lock().await;
        collector_guard.generate_report()?
    };
    
    print_metrics_demo_summary(&report);
    
    Ok(())
}

fn print_metrics_demo_summary(report: &pensieve_validator::MetricsCollectionResults) {
    println!("\n🎯 METRICS COLLECTION DEMO RESULTS");
    println!("═══════════════════════════════════");
    
    // Overall assessment
    let assessment = &report.overall_assessment;
    println!("Overall Score: {:.1}%", assessment.overall_score * 100.0);
    println!("Performance Score: {:.1}%", assessment.performance_score * 100.0);
    println!("Reliability Score: {:.1}%", assessment.reliability_score * 100.0);
    println!("User Experience Score: {:.1}%", assessment.user_experience_score * 100.0);
    println!("Efficiency Score: {:.1}%", assessment.efficiency_score * 100.0);
    
    // Performance metrics
    let perf = &report.performance_metrics;
    println!("\n📈 PERFORMANCE TRACKING");
    println!("──────────────────────");
    println!("Data Points Collected: {}", perf.files_processed_per_second.len());
    println!("Memory Readings: {}", perf.memory_usage_over_time.len());
    println!("CPU Readings: {}", perf.cpu_usage_over_time.len());
    println!("Performance Consistency: {:.1}%", perf.performance_consistency_score * 100.0);
    println!("Processing Speed Analysis:");
    println!("  • Current: {:.2} files/sec", perf.processing_speed_analysis.current_files_per_second);
    println!("  • Peak: {:.2} files/sec", perf.processing_speed_analysis.peak_files_per_second);
    println!("  • Average: {:.2} files/sec", perf.processing_speed_analysis.average_files_per_second);
    println!("  • Trend: {:?}", perf.processing_speed_analysis.speed_trend);
    
    // Resource efficiency
    let efficiency = &perf.resource_efficiency_metrics;
    println!("Resource Efficiency:");
    println!("  • Memory Efficiency: {:.1}%", efficiency.memory_efficiency_score * 100.0);
    println!("  • CPU Efficiency: {:.1}%", efficiency.cpu_efficiency_score * 100.0);
    println!("  • Overall Efficiency: {:.1}%", efficiency.overall_efficiency_score * 100.0);
    
    if !efficiency.resource_waste_indicators.is_empty() {
        println!("  • Waste Indicators:");
        for indicator in &efficiency.resource_waste_indicators {
            println!("    - {}: {:.1}% waste", indicator.resource_type, indicator.waste_percentage);
        }
    }
    
    // Error tracking
    let errors = &report.error_metrics;
    println!("\n🚨 ERROR TRACKING");
    println!("────────────────");
    println!("Error Categories: {}", errors.error_categories.len());
    println!("Total Error Events: {}", errors.error_timeline.len());
    println!("Recovery Patterns: {}", errors.recovery_patterns.len());
    println!("Error Rate: {:.2}/min", errors.error_rate_analysis.errors_per_minute);
    println!("Error Trend: {:?}", errors.error_rate_analysis.error_rate_trend);
    
    for (category, stats) in &errors.error_categories {
        println!("  • {:?}: {} total ({} critical, {} recoverable)", 
                 category, stats.total_count, stats.critical_count, stats.recoverable_count);
    }
    
    // UX tracking
    let ux = &report.ux_metrics;
    println!("\n👤 USER EXPERIENCE TRACKING");
    println!("──────────────────────────");
    println!("Overall UX Score: {:.1}/10", ux.overall_ux_score);
    
    let progress = &ux.progress_reporting_quality;
    println!("Progress Reporting:");
    println!("  • Update Frequency: {:.1}%", progress.update_frequency_score * 100.0);
    println!("  • Information Completeness: {:.1}%", progress.information_completeness_score * 100.0);
    println!("  • Clarity: {:.1}%", progress.clarity_score * 100.0);
    println!("  • Progress Updates: {}", progress.progress_updates.len());
    
    let error_clarity = &ux.error_message_clarity;
    println!("Error Message Quality:");
    println!("  • Average Clarity: {:.1}%", error_clarity.average_clarity_score * 100.0);
    println!("  • Actionability: {:.1}%", error_clarity.actionability_score * 100.0);
    println!("  • Technical Jargon Score: {:.1}%", error_clarity.technical_jargon_score * 100.0);
    println!("  • Solution Guidance: {:.1}%", error_clarity.solution_guidance_score * 100.0);
    
    // Database tracking
    let db = &report.database_metrics;
    println!("\n🗄️  DATABASE TRACKING");
    println!("───────────────────");
    println!("Operation Types Tracked: {}", db.operation_timings.len());
    println!("Slow Queries Detected: {}", db.query_performance_analysis.slow_queries.len());
    println!("Bottlenecks Identified: {}", db.database_bottlenecks.len());
    
    for (operation, stats) in &db.operation_timings {
        println!("  • {:?}: {} ops, avg {:.2}ms, {:.1} ops/sec", 
                 operation, stats.total_operations, stats.average_time.as_millis(), stats.operations_per_second);
    }
    
    // Key insights
    if !assessment.key_insights.is_empty() {
        println!("\n💡 KEY INSIGHTS");
        println!("──────────────");
        for insight in &assessment.key_insights {
            println!("  • {}", insight);
        }
    }
    
    // Recommendations
    if !assessment.improvement_recommendations.is_empty() {
        println!("\n🔧 IMPROVEMENT RECOMMENDATIONS");
        println!("─────────────────────────────");
        for recommendation in &assessment.improvement_recommendations {
            println!("  • {}", recommendation);
        }
    }
    
    println!("\n✨ Demo complete! The metrics collection system successfully tracked:");
    println!("   📊 Real-time performance data");
    println!("   🚨 Error patterns and recovery");
    println!("   👤 User experience quality");
    println!("   🗄️  Database operation efficiency");
    println!("   🎯 Overall system assessment");
}

/// Save validation results to files
async fn save_validation_results(
    results: &pensieve_validator::ComprehensiveValidationResults,
    output_dir: &PathBuf,
    config: &CliConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;
    
    if config.output.enable_json_output {
        let json_path = output_dir.join("validation_results.json");
        let json_content = serde_json::to_string_pretty(results)?;
        std::fs::write(&json_path, json_content)?;
    }
    
    if config.output.enable_html_reports {
        let html_path = output_dir.join("validation_report.html");
        std::fs::write(&html_path, "HTML validation report placeholder")?;
    }
    
    if config.output.enable_csv_export {
        let csv_path = output_dir.join("validation_summary.csv");
        std::fs::write(&csv_path, "CSV validation report placeholder")?;
    }
    
    Ok(())
}

/// Load validation results from file
fn load_validation_results(path: &PathBuf) -> Result<pensieve_validator::ComprehensiveValidationResults, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let results: pensieve_validator::ComprehensiveValidationResults = serde_json::from_str(&content)?;
    Ok(results)
}

/// Print directory analysis results
fn print_directory_analysis(
    analysis: &pensieve_validator::types::DirectoryAnalysis,
    chaos_report: &pensieve_validator::types::ChaosReport,
    format: &OutputFormat,
) {
    match format {
        OutputFormat::Human => {
            println!("\n📁 DIRECTORY ANALYSIS");
            println!("═══════════════════");
            println!("Total Files: {}", analysis.total_files);
            println!("Total Directories: {}", analysis.total_directories);
            println!("Total Size: {:.2} MB", analysis.total_size_bytes as f64 / 1_048_576.0);
            
            let chaos_metrics = chaos_report.calculate_chaos_metrics(analysis.total_files);
            println!("Chaos Score: {:.1}% ({} problematic files)", 
                     chaos_metrics.chaos_percentage, chaos_metrics.problematic_file_count);
        }
        OutputFormat::Json => {
            let combined = serde_json::json!({
                "analysis": analysis,
                "chaos_report": chaos_report
            });
            println!("{}", serde_json::to_string_pretty(&combined).unwrap());
        }
        OutputFormat::Yaml => {
            // For now, use JSON format as YAML support would require additional dependency
            let combined = serde_json::json!({
                "analysis": analysis,
                "chaos_report": chaos_report
            });
            println!("{}", serde_json::to_string_pretty(&combined).unwrap());
        }
    }
}

/// Save directory analysis results
fn save_directory_analysis(
    analysis: &pensieve_validator::types::DirectoryAnalysis,
    chaos_report: &pensieve_validator::types::ChaosReport,
    output_path: &PathBuf,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let combined = serde_json::json!({
        "analysis": analysis,
        "chaos_report": chaos_report
    });
    
    match format {
        OutputFormat::Human => {
            let content = format!("Directory Analysis Results\n\
                                 Total Files: {}\n\
                                 Total Directories: {}\n\
                                 Total Size: {:.2} MB\n\
                                 Chaos Score: {:.1}%",
                                analysis.total_files,
                                analysis.total_directories,
                                analysis.total_size_bytes as f64 / 1_048_576.0,
                                chaos_report.calculate_chaos_metrics(analysis.total_files).chaos_percentage);
            std::fs::write(output_path, content)?;
        }
        OutputFormat::Json | OutputFormat::Yaml => {
            let content = serde_json::to_string_pretty(&combined)?;
            std::fs::write(output_path, content)?;
        }
    }
    
    Ok(())
}

/// Comparison results structure
#[derive(Debug, serde::Serialize)]
struct ComparisonResults {
    baseline_summary: ValidationSummary,
    current_summary: ValidationSummary,
    changes: Vec<MetricChange>,
    overall_assessment: String,
}

#[derive(Debug, serde::Serialize)]
struct ValidationSummary {
    overall_score: f64,
    performance_score: f64,
    reliability_score: f64,
    ux_score: f64,
    files_processed: u64,
    execution_time_seconds: f64,
}

#[derive(Debug, serde::Serialize)]
struct MetricChange {
    metric_name: String,
    baseline_value: f64,
    current_value: f64,
    change_percentage: f64,
    is_significant: bool,
    is_improvement: bool,
}

/// Compare validation results
fn compare_validation_results(
    baseline: &pensieve_validator::ComprehensiveValidationResults,
    current: &pensieve_validator::ComprehensiveValidationResults,
    threshold: f64,
    _focus_metrics: Option<Vec<String>>,
) -> Result<ComparisonResults, Box<dyn std::error::Error>> {
    let baseline_summary = ValidationSummary {
        overall_score: baseline.validation_assessment.overall_score,
        performance_score: baseline.metrics_collection_results.overall_assessment.performance_score,
        reliability_score: baseline.metrics_collection_results.overall_assessment.reliability_score,
        ux_score: baseline.metrics_collection_results.ux_metrics.overall_ux_score,
        files_processed: baseline.pensieve_execution_results.output_analysis.files_processed,
        execution_time_seconds: baseline.pensieve_execution_results.execution_time.as_secs_f64(),
    };
    
    let current_summary = ValidationSummary {
        overall_score: current.validation_assessment.overall_score,
        performance_score: current.metrics_collection_results.overall_assessment.performance_score,
        reliability_score: current.metrics_collection_results.overall_assessment.reliability_score,
        ux_score: current.metrics_collection_results.ux_metrics.overall_ux_score,
        files_processed: current.pensieve_execution_results.output_analysis.files_processed,
        execution_time_seconds: current.pensieve_execution_results.execution_time.as_secs_f64(),
    };
    
    let mut changes = Vec::new();
    
    // Compare key metrics
    let metrics = vec![
        ("Overall Score", baseline_summary.overall_score, current_summary.overall_score),
        ("Performance Score", baseline_summary.performance_score, current_summary.performance_score),
        ("Reliability Score", baseline_summary.reliability_score, current_summary.reliability_score),
        ("UX Score", baseline_summary.ux_score, current_summary.ux_score),
        ("Execution Time", baseline_summary.execution_time_seconds, current_summary.execution_time_seconds),
    ];
    
    for (name, baseline_val, current_val) in metrics {
        let change_pct = if baseline_val != 0.0 {
            ((current_val - baseline_val) / baseline_val) * 100.0
        } else {
            0.0
        };
        
        let is_significant = change_pct.abs() >= threshold;
        let is_improvement = match name {
            "Execution Time" => change_pct < 0.0, // Lower is better
            _ => change_pct > 0.0, // Higher is better
        };
        
        changes.push(MetricChange {
            metric_name: name.to_string(),
            baseline_value: baseline_val,
            current_value: current_val,
            change_percentage: change_pct,
            is_significant,
            is_improvement,
        });
    }
    
    let significant_changes = changes.iter().filter(|c| c.is_significant).count();
    let improvements = changes.iter().filter(|c| c.is_significant && c.is_improvement).count();
    let regressions = changes.iter().filter(|c| c.is_significant && !c.is_improvement).count();
    
    let overall_assessment = if significant_changes == 0 {
        "No significant changes detected".to_string()
    } else if improvements > regressions {
        format!("Overall improvement: {} improvements, {} regressions", improvements, regressions)
    } else if regressions > improvements {
        format!("Overall regression: {} regressions, {} improvements", regressions, improvements)
    } else {
        format!("Mixed results: {} improvements, {} regressions", improvements, regressions)
    };
    
    Ok(ComparisonResults {
        baseline_summary,
        current_summary,
        changes,
        overall_assessment,
    })
}

/// Print comparison results
fn print_comparison_results(comparison: &ComparisonResults, format: &OutputFormat) {
    match format {
        OutputFormat::Human => {
            println!("\n🔄 VALIDATION COMPARISON");
            println!("═══════════════════════");
            println!("Assessment: {}", comparison.overall_assessment);
            println!();
            
            for change in &comparison.changes {
                let icon = if change.is_significant {
                    if change.is_improvement { "📈" } else { "📉" }
                } else {
                    "➡️"
                };
                
                println!("{} {}: {:.2} → {:.2} ({:+.1}%)",
                         icon, change.metric_name, change.baseline_value, 
                         change.current_value, change.change_percentage);
            }
        }
        OutputFormat::Json | OutputFormat::Yaml => {
            println!("{}", serde_json::to_string_pretty(comparison).unwrap());
        }
    }
}

/// Save comparison results
fn save_comparison_results(
    comparison: &ComparisonResults,
    output_path: &PathBuf,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Human => {
            let mut content = format!("Validation Comparison Results\n\
                                     Assessment: {}\n\n", comparison.overall_assessment);
            
            for change in &comparison.changes {
                content.push_str(&format!("{}: {:.2} → {:.2} ({:+.1}%)\n",
                                         change.metric_name, change.baseline_value,
                                         change.current_value, change.change_percentage));
            }
            
            std::fs::write(output_path, content)?;
        }
        OutputFormat::Json | OutputFormat::Yaml => {
            let content = serde_json::to_string_pretty(comparison)?;
            std::fs::write(output_path, content)?;
        }
    }
    
    Ok(())
}

fn print_validation_summary(results: &pensieve_validator::ComprehensiveValidationResults, format: &OutputFormat) {
    match format {
        OutputFormat::Human => {
            println!("\n📊 VALIDATION SUMMARY");
            println!("═══════════════════════");
            
            // Overall assessment
            let assessment = &results.validation_assessment;
            println!("Overall Score: {:.1}%", assessment.overall_score * 100.0);
            println!("Performance: {:?}", assessment.performance_grade);
            println!("Reliability: {:?}", assessment.reliability_grade);
            println!("User Experience: {:?}", assessment.user_experience_grade);
            println!("Efficiency: {:?}", assessment.efficiency_grade);
            
            // Production readiness
            match &assessment.production_readiness {
                pensieve_validator::ProductionReadiness::Ready => {
                    println!("\n🚀 Production Status: READY");
                }
                pensieve_validator::ProductionReadiness::ReadyWithCaveats(caveats) => {
                    println!("\n⚠️  Production Status: READY WITH CAVEATS");
                    for caveat in caveats {
                        println!("   • {}", caveat);
                    }
                }
                pensieve_validator::ProductionReadiness::NotReady(blockers) => {
                    println!("\n🚫 Production Status: NOT READY");
                    for blocker in blockers {
                        println!("   • {}", blocker);
                    }
                }
            }
            
            // Critical issues
            if !assessment.critical_issues.is_empty() {
                println!("\n🚨 CRITICAL ISSUES:");
                for issue in &assessment.critical_issues {
                    println!("   • {} (Severity: {:.1})", issue.description, issue.severity);
                    println!("     Action: {}", issue.recommended_action);
                }
            }
            
            // Performance metrics
            let perf = &results.pensieve_execution_results;
            println!("\n📈 PERFORMANCE METRICS");
            println!("─────────────────────");
            println!("Execution Time: {:.2}s", perf.execution_time.as_secs_f64());
            println!("Files Processed: {}", perf.output_analysis.files_processed);
            println!("Processing Speed: {:.2} files/sec", perf.performance_metrics.files_per_second);
            println!("Peak Memory: {} MB", perf.peak_memory_mb);
            println!("Average Memory: {} MB", perf.average_memory_mb);
            println!("Peak CPU: {:.1}%", perf.cpu_usage_stats.peak_cpu_percent);
            
            // Error summary
            if perf.error_summary.total_errors > 0 {
                println!("\n⚠️  ERROR SUMMARY");
                println!("─────────────────");
                println!("Total Errors: {}", perf.error_summary.total_errors);
                println!("Critical Errors: {}", perf.error_summary.critical_errors.len());
                println!("Error Rate: {:.1}/min", perf.error_summary.error_rate_per_minute);
            }
            
            // Recommendations
            if !results.recommendations.is_empty() {
                println!("\n💡 RECOMMENDATIONS");
                println!("──────────────────");
                for rec in &results.recommendations {
                    let priority_icon = match rec.priority {
                        pensieve_validator::RecommendationPriority::Critical => "🔴",
                        pensieve_validator::RecommendationPriority::High => "🟠",
                        pensieve_validator::RecommendationPriority::Medium => "🟡",
                        pensieve_validator::RecommendationPriority::Low => "🟢",
                    };
                    println!("   {} [{}] {}", priority_icon, rec.category, rec.description);
                    println!("     Impact: {}", rec.expected_impact);
                }
            }
            
            // Metrics summary
            let metrics = &results.metrics_collection_results;
            println!("\n📊 METRICS SUMMARY");
            println!("─────────────────");
            println!("Collection Duration: {:.2}s", metrics.collection_duration.as_secs_f64());
            println!("Performance Consistency: {:.1}%", metrics.performance_metrics.performance_consistency_score * 100.0);
            println!("Resource Efficiency: {:.1}%", metrics.overall_assessment.efficiency_score * 100.0);
            println!("UX Score: {:.1}/10", metrics.ux_metrics.overall_ux_score);
            
            println!("\n✨ Validation complete! Check the detailed results above.");
        }
        OutputFormat::Json | OutputFormat::Yaml => {
            println!("{}", serde_json::to_string_pretty(results).unwrap());
        }
    }
}/
// Establish a baseline from validation results
async fn run_establish_baseline(
    verbose: bool,
    quiet: bool,
    results_path: PathBuf,
    baseline_id: String,
    dataset_path: PathBuf,
    system_info_path: Option<PathBuf>,
    storage_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("📊 Establishing baseline '{}' from: {:?}", baseline_id, results_path);
    }
    
    if !results_path.exists() {
        eprintln!("❌ Results file does not exist: {:?}", results_path);
        std::process::exit(1);
    }
    
    if !dataset_path.exists() {
        eprintln!("❌ Dataset path does not exist: {:?}", dataset_path);
        std::process::exit(1);
    }
    
    // Load validation results
    let results = load_validation_results(&results_path)?;
    
    // Load or create system info
    let system_info = if let Some(info_path) = system_info_path {
        if info_path.exists() {
            let info_content = std::fs::read_to_string(info_path)?;
            serde_json::from_str(&info_content)?
        } else {
            eprintln!("❌ System info file does not exist: {:?}", info_path);
            std::process::exit(1);
        }
    } else {
        // Create default system info
        SystemInfo {
            cpu_model: "Unknown CPU".to_string(),
            memory_gb: 8,
            storage_type: "Unknown Storage".to_string(),
            os_version: std::env::consts::OS.to_string(),
        }
    };
    
    // Create comparative analyzer
    let storage_path = storage_dir.unwrap_or_else(|| PathBuf::from("./validation_storage"));
    let mut analyzer = ComparativeAnalyzer::new(storage_path);
    
    // Establish baseline
    match analyzer.establish_baseline(baseline_id.clone(), results, dataset_path, system_info).await {
        Ok(baseline_set) => {
            if !quiet {
                println!("✅ Baseline '{}' established successfully!", baseline_id);
                println!("   Established at: {}", baseline_set.established_at.format("%Y-%m-%d %H:%M:%S UTC"));
                println!("   Confidence level: {:.1}%", baseline_set.confidence_level * 100.0);
                println!("   Sample count: {}", baseline_set.sample_count);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to establish baseline: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Compare current results against established baseline
async fn run_compare_baseline(
    verbose: bool,
    quiet: bool,
    output_format: OutputFormat,
    results_path: PathBuf,
    baseline_id: String,
    output: Option<PathBuf>,
    storage_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("🔄 Comparing results against baseline '{}'", baseline_id);
        println!("   Current results: {:?}", results_path);
    }
    
    if !results_path.exists() {
        eprintln!("❌ Results file does not exist: {:?}", results_path);
        std::process::exit(1);
    }
    
    // Load validation results
    let results = load_validation_results(&results_path)?;
    
    // Create comparative analyzer
    let storage_path = storage_dir.unwrap_or_else(|| PathBuf::from("./validation_storage"));
    let analyzer = ComparativeAnalyzer::new(storage_path);
    
    // Compare against baseline
    match analyzer.compare_against_baseline(&baseline_id, results).await {
        Ok(comparison) => {
            if !quiet {
                print_baseline_comparison(&comparison, &output_format);
            }
            
            // Save comparison if output specified
            if let Some(output_path) = output {
                save_baseline_comparison(&comparison, &output_path, &output_format)?;
                if !quiet {
                    println!("💾 Comparison results saved to: {:?}", output_path);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to compare against baseline: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Analyze historical trends
async fn run_analyze_trends(
    verbose: bool,
    quiet: bool,
    dataset_path: PathBuf,
    period_days: i64,
    output_dir: PathBuf,
    storage_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("📈 Analyzing trends for dataset: {:?}", dataset_path);
        println!("   Analysis period: {} days", period_days);
    }
    
    if !dataset_path.exists() {
        eprintln!("❌ Dataset path does not exist: {:?}", dataset_path);
        std::process::exit(1);
    }
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // Create comparative analyzer
    let storage_path = storage_dir.unwrap_or_else(|| PathBuf::from("./validation_storage"));
    let analyzer = ComparativeAnalyzer::new(storage_path);
    
    // Analyze trends
    match analyzer.analyze_historical_trends(&dataset_path, period_days).await {
        Ok(trend_analysis) => {
            if !quiet {
                print_trend_analysis(&trend_analysis);
            }
            
            // Save trend analysis
            let analysis_path = output_dir.join("trend_analysis.json");
            let analysis_json = serde_json::to_string_pretty(&trend_analysis)?;
            std::fs::write(&analysis_path, analysis_json)?;
            
            if !quiet {
                println!("💾 Trend analysis saved to: {:?}", analysis_path);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to analyze trends: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Generate historical report with trend visualization
async fn run_generate_historical_report(
    verbose: bool,
    quiet: bool,
    dataset_path: PathBuf,
    period_days: i64,
    output_dir: PathBuf,
    storage_dir: Option<PathBuf>,
    formats: Option<Vec<HistoricalReportFormat>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("📊 Generating historical report for dataset: {:?}", dataset_path);
        println!("   Analysis period: {} days", period_days);
    }
    
    if !dataset_path.exists() {
        eprintln!("❌ Dataset path does not exist: {:?}", dataset_path);
        std::process::exit(1);
    }
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // Create analyzers
    let storage_path = storage_dir.unwrap_or_else(|| PathBuf::from("./validation_storage"));
    let analyzer = ComparativeAnalyzer::new(storage_path);
    let report_generator = HistoricalReportGenerator::new(output_dir.clone());
    
    // Analyze trends
    let trend_analysis = match analyzer.analyze_historical_trends(&dataset_path, period_days).await {
        Ok(analysis) => analysis,
        Err(e) => {
            eprintln!("❌ Failed to analyze trends: {}", e);
            std::process::exit(1);
        }
    };
    
    // Get comparisons and baselines (simplified for now)
    let comparisons = Vec::new(); // Would load from storage
    let baselines = Vec::new();   // Would load from storage
    
    // Generate historical report
    match report_generator.generate_historical_report(trend_analysis, comparisons, baselines).await {
        Ok(report) => {
            if !quiet {
                println!("✅ Historical report generated successfully!");
                println!("   Report ID: {}", report.report_id);
                println!("   Generated at: {}", report.generated_at.format("%Y-%m-%d %H:%M:%S UTC"));
                println!("   Report period: {} to {}", 
                    report.report_period.start_date.format("%Y-%m-%d"),
                    report.report_period.end_date.format("%Y-%m-%d")
                );
                println!("   Data points: {}", report.report_period.total_validation_runs);
            }
            
            // Generate charts
            let charts = report_generator.generate_trend_charts(&trend_analysis).await?;
            if !quiet && !charts.is_empty() {
                println!("📈 Generated {} trend charts", charts.len());
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to generate historical report: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// List available baselines
async fn run_list_baselines(
    verbose: bool,
    quiet: bool,
    storage_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("📋 Listing available baselines...");
    }
    
    // Create comparative analyzer
    let storage_path = storage_dir.unwrap_or_else(|| PathBuf::from("./validation_storage"));
    let analyzer = ComparativeAnalyzer::new(storage_path);
    
    // List baselines
    match analyzer.list_baselines().await {
        Ok(baselines) => {
            if baselines.is_empty() {
                if !quiet {
                    println!("📭 No baselines found");
                }
            } else {
                if !quiet {
                    println!("📊 Found {} baseline(s):", baselines.len());
                    for baseline_id in &baselines {
                        println!("   • {}", baseline_id);
                    }
                } else {
                    for baseline_id in &baselines {
                        println!("{}", baseline_id);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to list baselines: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Delete a baseline
async fn run_delete_baseline(
    verbose: bool,
    quiet: bool,
    baseline_id: String,
    storage_dir: Option<PathBuf>,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("🗑️  Deleting baseline '{}'...", baseline_id);
    }
    
    // Confirm deletion if not forced
    if !force && !quiet {
        print!("Are you sure you want to delete baseline '{}'? (y/N): ", baseline_id);
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        if !input.trim().to_lowercase().starts_with('y') {
            println!("❌ Deletion cancelled");
            return Ok(());
        }
    }
    
    // Create comparative analyzer
    let storage_path = storage_dir.unwrap_or_else(|| PathBuf::from("./validation_storage"));
    let mut analyzer = ComparativeAnalyzer::new(storage_path);
    
    // Delete baseline
    match analyzer.delete_baseline(&baseline_id).await {
        Ok(()) => {
            if !quiet {
                println!("✅ Baseline '{}' deleted successfully", baseline_id);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to delete baseline: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

/// Print baseline comparison results
fn print_baseline_comparison(
    comparison: &pensieve_validator::ValidationComparison,
    output_format: &OutputFormat,
) {
    match output_format {
        OutputFormat::Human => {
            println!("\n🔄 BASELINE COMPARISON RESULTS");
            println!("═══════════════════════════════");
            println!("Comparison ID: {}", comparison.comparison_id);
            println!("Comparison Time: {}", comparison.comparison_timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
            
            println!("\n📊 BASELINE RUN");
            println!("──────────────");
            println!("Run ID: {}", comparison.baseline_run.run_id);
            println!("Timestamp: {}", comparison.baseline_run.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
            println!("Pensieve Version: {}", comparison.baseline_run.pensieve_version);
            println!("Duration: {:.2}s", comparison.baseline_run.total_duration_seconds);
            
            println!("\n📊 CURRENT RUN");
            println!("─────────────");
            println!("Run ID: {}", comparison.current_run.run_id);
            println!("Timestamp: {}", comparison.current_run.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
            println!("Pensieve Version: {}", comparison.current_run.pensieve_version);
            println!("Duration: {:.2}s", comparison.current_run.total_duration_seconds);
            
            println!("\n⚡ PERFORMANCE COMPARISON");
            println!("───────────────────────");
            let perf = &comparison.performance_comparison;
            print_performance_change("Files per Second", &perf.files_per_second_change);
            print_performance_change("Memory Usage (MB)", &perf.memory_usage_change);
            print_performance_change("Processing Time (s)", &perf.processing_time_change);
            
            println!("\n🔒 RELIABILITY COMPARISON");
            println!("────────────────────────");
            let rel = &comparison.reliability_comparison;
            println!("Crash Count Change: {:+}", rel.crash_count_change);
            print_performance_change("Error Recovery Rate", &rel.error_recovery_rate_change);
            print_performance_change("Overall Reliability", &rel.overall_reliability_change);
            
            println!("\n👤 UX COMPARISON");
            println!("───────────────");
            let ux = &comparison.ux_comparison;
            print_performance_change("Progress Reporting", &ux.progress_reporting_change);
            print_performance_change("Error Message Clarity", &ux.error_message_clarity_change);
            print_performance_change("Completion Feedback", &ux.completion_feedback_change);
            print_performance_change("Overall UX Score", &ux.overall_ux_change);
            
            println!("\n💾 DEDUPLICATION COMPARISON");
            println!("──────────────────────────");
            let dedup = &comparison.deduplication_comparison;
            print_performance_change("Storage Savings (%)", &dedup.storage_savings_change);
            print_performance_change("Processing Time (s)", &dedup.processing_time_change);
            println!("ROI Recommendation: {} → {}", 
                dedup.roi_recommendation_change.baseline_recommendation,
                dedup.roi_recommendation_change.current_recommendation
            );
            
            println!("\n📈 OVERALL ASSESSMENT");
            println!("────────────────────");
            let assessment = &comparison.overall_assessment;
            println!("Overall Trend: {:?}", assessment.overall_trend);
            println!("Summary: {}", assessment.summary);
            
            if !assessment.key_insights.is_empty() {
                println!("\n💡 Key Insights:");
                for insight in &assessment.key_insights {
                    println!("  • {}", insight);
                }
            }
            
            if !comparison.regression_alerts.is_empty() {
                println!("\n🚨 REGRESSION ALERTS");
                println!("───────────────────");
                for alert in &comparison.regression_alerts {
                    println!("  • {:?}: {} ({:.1}% degradation)", 
                        alert.severity, alert.description, alert.degradation_percentage);
                }
            }
            
            if !comparison.improvement_highlights.is_empty() {
                println!("\n🎯 IMPROVEMENT HIGHLIGHTS");
                println!("────────────────────────");
                for improvement in &comparison.improvement_highlights {
                    println!("  • {}: {} ({:.1}% improvement)", 
                        improvement.category, improvement.description, improvement.improvement_percentage);
                }
            }
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(comparison).unwrap();
            println!("{}", json);
        }
        OutputFormat::Yaml => {
            // YAML output would be implemented here
            println!("YAML output not yet implemented");
        }
    }
}

/// Print performance change details
fn print_performance_change(metric_name: &str, change: &pensieve_validator::comparative_analyzer::PerformanceChange) {
    let trend_symbol = match change.trend_direction {
        pensieve_validator::TrendDirection::Improving => "📈",
        pensieve_validator::TrendDirection::Degrading => "📉",
        pensieve_validator::TrendDirection::Stable => "➡️",
        pensieve_validator::TrendDirection::Volatile => "📊",
    };
    
    let significance_color = match change.change_significance {
        pensieve_validator::comparative_analyzer::ChangeSignificance::Major => "🔴",
        pensieve_validator::comparative_analyzer::ChangeSignificance::Significant => "🟠",
        pensieve_validator::comparative_analyzer::ChangeSignificance::Moderate => "🟡",
        pensieve_validator::comparative_analyzer::ChangeSignificance::Minor => "🟢",
        pensieve_validator::comparative_analyzer::ChangeSignificance::Negligible => "⚪",
    };
    
    println!("{} {} {}: {:.2} → {:.2} ({:+.1}%) {:?}", 
        trend_symbol, significance_color, metric_name,
        change.baseline_value, change.current_value, change.percentage_change,
        change.change_significance
    );
}

/// Print trend analysis summary
fn print_trend_analysis(analysis: &pensieve_validator::HistoricalTrendAnalysis) {
    println!("\n📈 HISTORICAL TREND ANALYSIS");
    println!("═══════════════════════════");
    println!("Analysis ID: {}", analysis.analysis_id);
    println!("Dataset: {:?}", analysis.dataset_path);
    println!("Analysis Period: {} to {} ({} days)", 
        analysis.analysis_period.start_date.format("%Y-%m-%d"),
        analysis.analysis_period.end_date.format("%Y-%m-%d"),
        analysis.analysis_period.duration_days
    );
    println!("Data Points: {}", analysis.data_points);
    
    println!("\n⚡ PERFORMANCE TRENDS");
    println!("───────────────────");
    println!("Overall Direction: {:?}", analysis.performance_trends.overall_performance_trend);
    
    println!("\n🔒 RELIABILITY TRENDS");
    println!("────────────────────");
    println!("Overall Direction: {:?}", analysis.reliability_trends.overall_reliability_trend);
    
    println!("\n👤 UX TRENDS");
    println!("───────────");
    println!("Overall Direction: {:?}", analysis.ux_trends.overall_ux_trend);
    
    println!("\n🚀 IMPROVEMENT TRAJECTORY");
    println!("────────────────────────");
    let trajectory = &analysis.improvement_trajectory;
    println!("Overall Trajectory: {:?}", trajectory.overall_trajectory);
    println!("Improvement Velocity: {:.3}/day", trajectory.improvement_velocity);
    println!("Consistency Score: {:.1}%", trajectory.consistency_score * 100.0);
    
    if !analysis.recommendations.is_empty() {
        println!("\n💡 RECOMMENDATIONS");
        println!("─────────────────");
        for recommendation in &analysis.recommendations {
            println!("  • {:?}: {}", recommendation.priority, recommendation.description);
        }
    }
}

/// Save baseline comparison results
fn save_baseline_comparison(
    comparison: &pensieve_validator::ValidationComparison,
    output_path: &PathBuf,
    output_format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match output_format {
        OutputFormat::Json => {
            let json_content = serde_json::to_string_pretty(comparison)?;
            std::fs::write(output_path, json_content)?;
        }
        OutputFormat::Yaml => {
            // YAML serialization would be implemented here
            return Err("YAML output not yet implemented".into());
        }
        OutputFormat::Human => {
            // Human-readable format saved as text
            let mut content = String::new();
            content.push_str(&format!("Baseline Comparison Report\n"));
            content.push_str(&format!("Generated: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
            content.push_str(&format!("Comparison ID: {}\n", comparison.comparison_id));
            // Add more formatted content here
            std::fs::write(output_path, content)?;
        }
    }
    Ok(())
}