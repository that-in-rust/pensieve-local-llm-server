//! # Pensieve CLI - Multi-Agent Debate Launcher
//!
//! Zero-config CLI for MoA-Lite debate system.
//!
//! ## Usage
//!
//! ```bash
//! # Interactive mode
//! pensieve
//!
//! # Single query mode
//! pensieve --query "Explain ownership in Rust"
//!
//! # Mock mode (no llama-server required)
//! pensieve --mock --query "Who is the Home Minister of India?"
//!
//! # Connect to custom llama-server
//! pensieve --llm-url http://localhost:9000
//! ```

use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use debate_orchestrator_state_machine::{
    DebateOrchestrator, HeuristicRouter, InMemoryBlackboard, MoaLiteOrchestrator, MockLlmClient,
    ResponseSource,
};
use llama_server_client_streaming::{LlamaServerClient, LlmClientConfig};
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Pensieve MoA-Lite Debate CLI
#[derive(Parser, Debug)]
#[command(name = "pensieve")]
#[command(author, version, about = "Multi-Agent Debate AI Assistant CLI")]
struct CliArgs {
    /// Single query to process (non-interactive mode)
    #[arg(short, long)]
    query: Option<String>,

    /// llama-server URL
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    llm_url: String,

    /// Use mock LLM backend (for testing)
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// Output as JSON
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Verbose output
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn setup_tracing_subscriber_verbose(verbose: bool) {
    let filter = if verbose {
        "pensieve_cli_debate_launcher=debug,debate_orchestrator_state_machine=debug"
    } else {
        "pensieve_cli_debate_launcher=info"
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();
}

/// Output result as JSON
fn output_result_as_json(query: &str, response: &str, source: ResponseSource, latency_ms: u64, proposal_count: u8) {
    let output = serde_json::json!({
        "query": query,
        "response": response,
        "metadata": {
            "source": format!("{:?}", source),
            "latency_ms": latency_ms,
            "proposal_count": proposal_count
        }
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

/// Output result as formatted text
fn output_result_as_text(response: &str, source: ResponseSource, latency_ms: u64, proposal_count: u8) {
    println!("\n{}", "=".repeat(60));
    println!("RESPONSE");
    println!("{}", "=".repeat(60));
    println!("{}", response);
    println!("{}", "-".repeat(60));
    println!(
        "Source: {:?} | Proposals: {} | Latency: {}ms",
        source, proposal_count, latency_ms
    );
    println!("{}", "=".repeat(60));
}

/// Process a single query
async fn process_single_query_async<O>(
    orchestrator: &O,
    query: &str,
    json_output: bool,
) -> Result<(), anyhow::Error>
where
    O: DebateOrchestrator,
{
    if !json_output {
        println!("\nProcessing: {}", query);
        println!("Running MoA-Lite debate (3 proposers + 1 aggregator)...\n");
    }

    let result = orchestrator
        .process_query_through_debate(query)
        .await
        .map_err(|e| anyhow::anyhow!("Debate failed: {}", e))?;

    if json_output {
        output_result_as_json(
            query,
            &result.response,
            result.source,
            result.latency_ms,
            result.proposal_count,
        );
    } else {
        output_result_as_text(
            &result.response,
            result.source,
            result.latency_ms,
            result.proposal_count,
        );
    }

    Ok(())
}

/// Run interactive REPL mode
async fn run_interactive_mode_repl<O>(orchestrator: &O, json_output: bool) -> Result<(), anyhow::Error>
where
    O: DebateOrchestrator,
{
    println!("=============================================================");
    println!("  PENSIEVE - Multi-Agent Debate AI Assistant");
    println!("  MoA-Lite: 3 Proposers + 1 Aggregator");
    println!("=============================================================");
    println!("Type your query and press Enter. Type 'quit' or 'exit' to stop.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("pensieve> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF
            break;
        }

        let query = input.trim();

        if query.is_empty() {
            continue;
        }

        if query == "quit" || query == "exit" {
            println!("Goodbye!");
            break;
        }

        if query == "help" {
            println!("\nCommands:");
            println!("  <query>  - Process query through MoA-Lite debate");
            println!("  help     - Show this help");
            println!("  quit     - Exit the program");
            println!();
            continue;
        }

        if let Err(e) = process_single_query_async(orchestrator, query, json_output).await {
            error!("Error: {}", e);
            eprintln!("Error processing query: {}", e);
        }
    }

    Ok(())
}

/// Create mock orchestrator for testing
fn create_mock_orchestrator_instance(
) -> MoaLiteOrchestrator<MockLlmClient, InMemoryBlackboard, HeuristicRouter> {
    let mock_response = r#"
Based on multi-agent debate synthesis:

The current Home Minister of India is **Amit Shah**. He has been serving as the Union Minister of Home Affairs since May 30, 2019.

Key points:
- Amit Shah is a senior leader of the Bharatiya Janata Party (BJP)
- He previously served as the President of BJP from 2014 to 2020
- As Home Minister, he oversees internal security, border management, and law enforcement
- Notable policies: Reorganization of Jammu & Kashmir, Citizenship Amendment Act
- He also holds the portfolio of Minister of Cooperation

The Ministry of Home Affairs is one of the most important ministries in the Government of India, responsible for internal security, management of para-military forces, border management, and administration of Union Territories.
"#;

    let llm_client = Arc::new(MockLlmClient::create_mock_for_testing(mock_response));
    let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
    let router = Arc::new(HeuristicRouter::create_router_with_defaults());

    MoaLiteOrchestrator::create_orchestrator_with_components(llm_client, blackboard, router)
}

/// Create real orchestrator connecting to llama-server
fn create_real_orchestrator_instance(
    llm_url: &str,
) -> Result<MoaLiteOrchestrator<LlamaServerClient, InMemoryBlackboard, HeuristicRouter>, anyhow::Error>
{
    let llm_config = LlmClientConfig::create_config_with_url(llm_url.to_string());
    let llm_client = Arc::new(
        LlamaServerClient::create_client_with_config(llm_config)
            .map_err(|e| anyhow::anyhow!("Failed to create LLM client: {}", e))?,
    );
    let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
    let router = Arc::new(HeuristicRouter::create_router_with_defaults());

    Ok(MoaLiteOrchestrator::create_orchestrator_with_components(
        llm_client,
        blackboard,
        router,
    ))
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = CliArgs::parse();

    setup_tracing_subscriber_verbose(args.verbose);

    info!(
        llm_url = %args.llm_url,
        mock = args.mock,
        "Initializing Pensieve CLI"
    );

    // Process based on mode
    let result = if args.mock {
        info!("Using mock LLM backend");
        let orchestrator = create_mock_orchestrator_instance();

        if let Some(query) = &args.query {
            process_single_query_async(&orchestrator, query, args.json).await
        } else {
            run_interactive_mode_repl(&orchestrator, args.json).await
        }
    } else {
        match create_real_orchestrator_instance(&args.llm_url) {
            Ok(orchestrator) => {
                if let Some(query) = &args.query {
                    process_single_query_async(&orchestrator, query, args.json).await
                } else {
                    run_interactive_mode_repl(&orchestrator, args.json).await
                }
            }
            Err(e) => {
                eprintln!("Failed to initialize: {}", e);
                eprintln!("\nHint: Use --mock flag to test without llama-server");
                return ExitCode::FAILURE;
            }
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}
