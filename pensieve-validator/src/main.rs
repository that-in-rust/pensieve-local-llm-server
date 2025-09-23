use pensieve_validator::{
    ValidationOrchestrator, ValidationOrchestratorConfig, 
    PensieveConfig, MonitoringConfig, PerformanceThresholds,
    MetricsCollector, ErrorCategory, ErrorSeverity, ErrorContext, SystemState,
    UXEventType, UXQualityScores, DatabaseOperation
};
use std::path::PathBuf;
use std::time::Duration;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pensieve-validator")]
#[command(about = "A comprehensive validation framework for pensieve")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run comprehensive validation on a directory
    Validate {
        /// Target directory to validate
        #[arg(short, long)]
        directory: PathBuf,
        
        /// Path to pensieve binary
        #[arg(short, long, default_value = "pensieve")]
        pensieve_binary: PathBuf,
        
        /// Output database path
        #[arg(short, long, default_value = "validation_results.db")]
        output_db: PathBuf,
        
        /// Timeout in seconds
        #[arg(short, long, default_value = "3600")]
        timeout: u64,
        
        /// Memory limit in MB
        #[arg(short, long, default_value = "8192")]
        memory_limit: u64,
        
        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Demonstrate metrics collection capabilities
    Demo {
        /// Duration of demo in seconds
        #[arg(short, long, default_value = "10")]
        duration: u64,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Validate {
            directory,
            pensieve_binary,
            output_db,
            timeout,
            memory_limit,
            verbose,
        } => {
            println!("🔍 Starting comprehensive validation of: {:?}", directory);
            
            // Configure validation
            let config = ValidationOrchestratorConfig {
                pensieve_config: PensieveConfig {
                    binary_path: pensieve_binary,
                    timeout_seconds: timeout,
                    memory_limit_mb: memory_limit,
                    output_database_path: output_db,
                    enable_deduplication: true,
                    verbose_output: verbose,
                },
                monitoring_config: MonitoringConfig::default(),
                metrics_collection_interval_ms: 500,
                enable_real_time_analysis: true,
                performance_thresholds: PerformanceThresholds {
                    min_files_per_second: 1.0,
                    max_memory_mb: memory_limit,
                    max_cpu_percent: 80.0,
                    max_error_rate_per_minute: 5.0,
                    min_ux_score: 7.0,
                },
            };
            
            // Create orchestrator and run validation
            let orchestrator = ValidationOrchestrator::new(config);
            
            match orchestrator.run_comprehensive_validation(&directory).await {
                Ok(results) => {
                    println!("\n✅ Validation completed successfully!");
                    print_validation_summary(&results);
                }
                Err(e) => {
                    eprintln!("\n❌ Validation failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        
        Commands::Demo { duration } => {
            println!("🎯 Starting metrics collection demo for {} seconds...", duration);
            run_metrics_demo(duration).await?;
        }
    }

    Ok(())
}

/// Demonstrate the metrics collection system capabilities
async fn run_metrics_demo(duration_secs: u64) -> Result<(), Box<dyn std::error::Error>> {
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

fn print_validation_summary(results: &pensieve_validator::ComprehensiveValidationResults) {
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