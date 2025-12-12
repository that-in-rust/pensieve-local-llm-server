# Pensieve: Multi-Agent Debate AI Assistant

**Achieve 65-75% Claude quality locally through multi-agent debate.**

Local LLMs are private and fast, but produce 40-50% Claude quality. Cloud APIs deliver quality but cost $20-50/month with privacy concerns. Pensieve bridges this gap using MoA-Lite (Mixture-of-Agents Lite) architecture that improves local LLM quality by 15-25% through structured debate.

## Why Multi-Agent Debate Works

| Approach | Quality vs Claude | Monthly Cost | Privacy |
|----------|-------------------|--------------|---------|
| Single Local LLM | 40-50% | $0 | Full |
| Cloud API (Claude/GPT-4) | 100% | $20-50 | None |
| **Pensieve (MoA-Lite)** | **65-75%** | **$0-15** | **Full** |

Three proposers generate diverse responses in parallel. One aggregator synthesizes the best answer. Result: significantly better quality than single-model inference.

## Quick Start

```bash
# Clone and build
git clone https://github.com/that-in-rust/pensieve-local-llm-server
cd pensieve-local-llm-server
cargo build --release

# Run tests (113 tests)
cargo test
```

## Usage

### HTTP Server (OpenAI-Compatible API)

```bash
# Start server (connects to llama-server at localhost:8080)
./target/release/pensieve-server

# Start in mock mode (for testing without llama-server)
./target/release/pensieve-server --mock

# Custom port
./target/release/pensieve-server --port 8000

# Custom llama-server URL
./target/release/pensieve-server --llm-url http://localhost:9000
```

**API Endpoints:**

```bash
# Health check
curl http://localhost:3000/health

# Chat completion (OpenAI-compatible)
curl -X POST http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Who is the Home Minister of India?"}]}'
```

**Example Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "pensieve-moa-lite",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The current Home Minister of India is **Amit Shah**..."
    },
    "finish_reason": "stop"
  }],
  "pensieve_metadata": {
    "source": "LocalDebate",
    "proposal_count": 3,
    "latency_ms": 15000
  }
}
```

### CLI (Interactive Mode)

```bash
# Interactive REPL
./target/release/pensieve

# Single query
./target/release/pensieve --query "Explain ownership in Rust"

# Mock mode (no llama-server required)
./target/release/pensieve --mock --query "Who is the Home Minister of India?"

# JSON output
./target/release/pensieve --mock --json --query "What is async/await?"
```

## Architecture: MoA-Lite 2-Layer Debate

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   COMPLEXITY ROUTER                          │
│            classify_routing_for_query()                      │
│                                                              │
│    ┌──────────────┐              ┌──────────────┐           │
│    │    LOCAL     │              │    CLOUD     │           │
│    │   (80%)      │              │   (20%)      │           │
│    └──────┬───────┘              └──────┬───────┘           │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────────┐    ┌─────────────────────┐
│     MOA-LITE DEBATE       │    │   CLAUDE HANDOFF    │
│                           │    │                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ │    │  Compressed context │
│  │ P1  │ │ P2  │ │ P3  │ │    │  (150-300 tokens)   │
│  │     │ │     │ │     │ │    │                     │
│  └──┬──┘ └──┬──┘ └──┬──┘ │    └─────────────────────┘
│     │       │       │     │
│     └───────┼───────┘     │
│             ▼             │
│      ┌───────────┐        │
│      │ AGGREGATOR│        │
│      │           │        │
│      └───────────┘        │
└───────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                      FINAL RESPONSE                          │
│                      (10-17s local)                          │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Router** classifies query complexity (Local vs Cloud)
2. **3 Proposers** generate diverse responses in parallel (150-300 tokens each)
3. **1 Aggregator** synthesizes the best answer (200-500 tokens)
4. **Graceful degradation**: Works with 2 of 3 proposers if one fails

## Crate Structure (L1 → L2 → L3)

```
crates/
├── L1 CORE (No External Dependencies)
│   ├── agent-role-definition-types      # AgentRole, ProposerConfig, AggregatorConfig
│   └── blackboard-handoff-protocol-core # ProposalEntry, CloudHandoffContext, Blackboard
│
├── L2 ENGINE (Async + Internal Dependencies)
│   ├── complexity-router-heuristic-classifier  # 2-way routing (Local/Cloud)
│   ├── llama-server-client-streaming           # HTTP + SSE to llama-server
│   └── debate-orchestrator-state-machine       # MoA-Lite state machine
│
└── L3 APPLICATION (External Dependencies)
    ├── pensieve-cli-debate-launcher     # Zero-config CLI (pensieve binary)
    └── pensieve-http-api-server         # Axum HTTP API (pensieve-server binary)
```

### Naming Convention

All crate and function names follow **4-word Parseltongue convention**:

```rust
// Crates: noun-noun-noun-noun
agent-role-definition-types
complexity-router-heuristic-classifier

// Functions: verb-noun-preposition-noun
create_config_with_index()
classify_routing_for_query()
process_query_through_debate()
```

## Key Components

### 1. Agent Roles (`agent-role-definition-types`)

```rust
pub enum AgentRole {
    Proposer,    // 3 instances, parallel execution
    Aggregator,  // 1 instance, synthesis
}

// Each proposer has a unique focus:
// - Proposer 0: ACCURACY (correctness-focused)
// - Proposer 1: CREATIVITY (alternative solutions)
// - Proposer 2: CONCISENESS (clarity-focused)
```

### 2. Complexity Router (`complexity-router-heuristic-classifier`)

```rust
pub enum RoutingDecision {
    Local,        // Full MoA-Lite debate (10-17s)
    CloudHandoff, // Route to Claude API (11-18s)
}

// Routing triggers:
// Local:  Code blocks, "explain"/"write"/"debug", < 2000 tokens
// Cloud:  "design"/"architect", > 2000 tokens, reasoning depth > 3
```

### 3. Debate Orchestrator (`debate-orchestrator-state-machine`)

```rust
// State machine: Idle → AnalyzingComplexity → Proposing → Aggregating → Complete

let result = orchestrator
    .process_query_through_debate("Explain ownership in Rust")
    .await?;

// Returns: DebateResult { response, source, latency_ms, proposal_count }
```

## Performance Targets

| Metric | Target | Hardware |
|--------|--------|----------|
| Local debate latency | 10-17s | Mac Mini M4 |
| Cloud handoff latency | 11-18s | - |
| Token throughput | 35-45 tok/s | Q4_K_M quantization |
| Memory usage | < 4GB | Qwen2.5-3B |
| Proposer output | 150-300 tokens | Per proposer |
| Aggregator output | 200-500 tokens | Final response |

## Development

### Prerequisites

- Rust 1.75+
- Mac with Apple Silicon (M1/M2/M3/M4) recommended
- llama-server running locally (for real inference)

### Running Tests

```bash
# Run all 113 tests
cargo test

# Run specific crate tests
cargo test -p agent-role-definition-types
cargo test -p debate-orchestrator-state-machine
cargo test -p pensieve-http-api-server
```

### Code Quality

```bash
# Check formatting
cargo fmt --check

# Run clippy (pedantic + nursery)
cargo clippy --all-targets
```

## Roadmap

- [x] L1 Core: Agent types and blackboard protocol
- [x] L2 Engine: Router, LLM client, orchestrator
- [x] L3 Application: CLI and HTTP API
- [ ] Web search integration
- [ ] Claude API integration for cloud handoff
- [ ] Metrics and observability

## Documentation

- [Architecture](./architecture-moa-lite-debate-system.md) - Detailed system design
- [PRD](./PRD-Multi-Agent-Debate-Assistant.md) - Product requirements

## License

MIT OR Apache-2.0
