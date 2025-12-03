# Product Requirements Document: Multi-Agent Debate AI Assistant

**Version:** 1.0
**Date:** 2025-12-03
**Status:** Draft for Implementation
**Target Platform:** Mac Mini M4 (Apple Silicon)
**Project:** pensieve-local-llm-server

---

## 1. Executive Summary

### Vision Statement
Build a high-quality local LLM coding assistant for open-source software development that achieves **65-75% of Claude/GPT-4 quality** for factual and code-related tasks through multi-agent debate architecture, while intelligently routing complex reasoning to cloud APIs when needed.

### Strategic Positioning
This system bridges the gap between fully local (private, fast, but lower quality) and fully cloud-based (expensive, latency-sensitive, privacy concerns) AI assistants by combining:
- **Local multi-agent debate** for common coding tasks (factual Q&A, code snippets, summarization)
- **Adaptive cloud routing** for complex reasoning beyond local capabilities
- **Parallel web search integration** for up-to-date information

### Key Differentiators
1. **Hybrid local-cloud architecture** - Uses local resources efficiently, cloud strategically
2. **Multi-agent debate** - Improves quality through iterative refinement
3. **Optimized for Apple Silicon** - Leverages unified memory architecture
4. **Cost-effective** - Reduces cloud API costs by 70-85% vs pure cloud
5. **Privacy-preserving** - Sensitive code stays local

### Success Metrics at a Glance
| Metric | Target |
|--------|--------|
| Quality vs Claude (code tasks) | 65-75% |
| Simple query latency | 3-5 seconds |
| Complex query latency | 11-18 seconds |
| Memory usage | <4GB |
| Token throughput (per stream) | 35-45 tok/s |
| Cloud API cost reduction | 70-85% |

---

## 2. Problem Statement

### The Open Source Developer's Dilemma

**Scenario:** A developer working on an OSS project needs AI assistance for:
- Understanding unfamiliar codebases
- Writing boilerplate code
- Debugging issues
- Researching API documentation
- Architectural decision-making

**Current Solutions and Their Failures:**

| Approach | Problems |
|----------|----------|
| **Pure Cloud (Claude/GPT-4)** | • $20-50/month subscription<br>• Latency on API calls<br>• Privacy concerns (code sent to cloud)<br>• Rate limits<br>• Requires internet connectivity |
| **Pure Local (Ollama + small model)** | • 40-50% quality vs Claude for complex tasks<br>• Poor reasoning capabilities<br>• No web search integration<br>• Struggles with multi-step problems<br>• Outdated knowledge |
| **Hybrid (Current Approaches)** | • Manual routing decisions<br>• Inefficient resource usage<br>• No quality optimization strategies<br>• Single model bottleneck |

### Why Existing Solutions Don't Address This

1. **Ollama/LM Studio** - Single model inference, no debate architecture, no adaptive routing
2. **LocalAI** - Multi-model support but no quality optimization through debate
3. **Text Generation WebUI** - Frontend focus, no intelligent routing
4. **Llama.cpp server** - Infrastructure only, no application logic

### Quantified Impact

**For a typical OSS developer (20 hours/week coding):**

| Task Type | % of Time | Current Solution | Cost/Quality Impact |
|-----------|-----------|------------------|---------------------|
| Simple code snippets | 40% | Cloud API | $15/month, overkill |
| Factual Q&A | 30% | Cloud API | $10/month, overkill |
| Complex reasoning | 20% | Cloud API | $15/month, necessary |
| Web-augmented queries | 10% | Manual search + Cloud | $5/month + time waste |

**With Multi-Agent Debate Assistant:**
- Reduce cloud API costs to $10-15/month (70% reduction)
- Improve local quality from 45% to 70% vs Claude
- Add parallel web search (reduce time by 50%)
- Keep sensitive code local (privacy win)

---

## 3. Product Vision & Goals

### Vision: "Claude-Quality Local Coding Assistant"

Enable OSS developers to have a **near-Claude-quality coding assistant** running entirely on their development machine, with strategic cloud augmentation for complex reasoning.

### Primary Goals

#### Goal 1: Quality Parity for Common Tasks
**Target:** 65-75% of Claude Sonnet quality for factual Q&A and code generation tasks

**Measured by:**
- Human evaluation on benchmark coding tasks
- Automated evaluation on HumanEval/MBPP code benchmarks
- User satisfaction surveys

#### Goal 2: Cost Efficiency
**Target:** Reduce cloud API costs by 70-85% compared to pure cloud usage

**Measured by:**
- API call volume tracking
- Cost per query analysis
- Local vs cloud routing ratio

#### Goal 3: Low Latency
**Target:**
- Simple queries: 3-5 seconds end-to-end
- Complex queries: 11-18 seconds end-to-end

**Measured by:**
- p50, p95, p99 latency metrics
- Time-to-first-token
- Time-to-complete-response

#### Goal 4: Resource Efficiency
**Target:** Run on Mac Mini M4 with <4GB memory footprint

**Measured by:**
- Peak memory usage monitoring
- Sustained memory usage over 8-hour sessions
- GPU memory allocation

#### Goal 5: Developer Experience
**Target:** Zero-config setup, seamless integration with development workflow

**Measured by:**
- Time to first successful query (<10 minutes from install)
- Configuration complexity (zero required config)
- IDE integration compatibility

### Non-Goals (Explicit Scope Boundaries)

1. **Not a training platform** - No local model fine-tuning
2. **Not a model hub** - Single optimized model, not multi-model switching
3. **Not a generic chatbot** - Optimized for coding tasks only
4. **Not a cloud replacement** - Complements cloud for complex reasoning
5. **Not cross-platform (initially)** - Apple Silicon only in v1.0

---

## 4. Target Users & Use Cases

### Primary User Persona: "Alex, the OSS Contributor"

**Demographics:**
- Software engineer, 3-7 years experience
- Contributes to 2-5 OSS projects regularly
- Works with unfamiliar codebases frequently
- Privacy-conscious about proprietary/sensitive code
- Budget-conscious ($20/month threshold for tools)

**Environment:**
- Mac Mini M4 or MacBook Pro M-series
- 32-64GB unified memory
- Fast internet (for cloud fallback)
- VS Code or similar IDE

**Pain Points:**
- Can't afford $50/month for Claude Pro + Copilot
- Doesn't want to send proprietary code to cloud
- Needs quick answers without context switching
- Frustrated by local model quality gaps

**Jobs to Be Done:**
1. Understand what unfamiliar code does
2. Generate boilerplate/repetitive code
3. Debug error messages and stack traces
4. Research library APIs and best practices
5. Make architectural decisions

### Use Case Matrix

| Use Case | Frequency | Local Quality | Cloud Quality | Target Strategy |
|----------|-----------|---------------|---------------|-----------------|
| **UC1: Code Explanation** | High (daily) | 70% | 90% | **Local debate** |
| **UC2: Simple Code Gen** | High (daily) | 65% | 85% | **Local debate** |
| **UC3: API Documentation Q&A** | Medium (weekly) | 75% | 95% | **Local + web search** |
| **UC4: Debugging Assistance** | Medium (weekly) | 60% | 85% | **Local debate** |
| **UC5: Architectural Design** | Low (monthly) | 40% | 90% | **Route to cloud** |
| **UC6: Complex Refactoring** | Low (monthly) | 45% | 88% | **Route to cloud** |
| **UC7: Test Generation** | Medium (weekly) | 70% | 85% | **Local debate** |
| **UC8: Code Review** | Medium (weekly) | 65% | 82% | **Local debate** |

### Use Case Details

#### UC1: Code Explanation
**User Story:**
As a developer, I want to understand what a complex function does so that I can modify it safely.

**Input:** Code snippet (10-100 lines)
**Expected Output:** Plain English explanation with key logic highlighted
**Quality Bar:** 70% as good as Claude
**Latency Target:** 3-5 seconds

**Example:**
```
User: "Explain what this Rust function does"
[code snippet]

Expected: "This function implements a depth-first search traversal...
The key optimization is caching visited nodes to avoid cycles..."
```

#### UC2: Simple Code Generation
**User Story:**
As a developer, I want to generate boilerplate code so that I can focus on business logic.

**Input:** Natural language description (1-3 sentences)
**Expected Output:** Syntactically correct code matching specification
**Quality Bar:** 65% as good as Claude
**Latency Target:** 4-6 seconds

**Example:**
```
User: "Write a Rust function to parse JSON config with error handling"

Expected:
```rust
use serde_json::Value;
use std::fs;

fn parse_config(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let config: Value = serde_json::from_str(&contents)?;
    Ok(config)
}
```
```

#### UC3: API Documentation Q&A (with Web Search)
**User Story:**
As a developer, I want to look up current API usage patterns so that I use libraries correctly.

**Input:** Question about library/framework (1-2 sentences)
**Expected Output:** Accurate answer with code examples and sources
**Quality Bar:** 75% as good as Claude with web search
**Latency Target:** 5-8 seconds (parallel web search)

**Example:**
```
User: "How do I use async/await with Tokio 1.x in Rust?"

Expected: "Tokio 1.x provides async runtime... [code example]
Sources: [docs.rs/tokio, official guide]"
```

#### UC8: Code Review
**User Story:**
As a developer, I want feedback on my code changes so that I can improve quality before submitting.

**Input:** Git diff or code snippet
**Expected Output:** List of issues (bugs, style, performance) with severity
**Quality Bar:** 65% as good as Claude
**Latency Target:** 5-8 seconds

---

## 5. Functional Requirements

### FR1: Multi-Agent Debate Architecture

#### FR1.1: Two-Layer MoA-Lite Configuration
**Priority:** P0 (Critical)

**Specification:**
- Implement 2-layer Mixture-of-Agents Lite architecture
- Layer 1: 3 proposer agents generate initial responses in parallel
- Layer 2: 1 aggregator agent synthesizes best answer
- Total: 4 inference passes per query

**Acceptance Criteria:**
- [ ] 3 parallel proposer inferences complete in <3 seconds
- [ ] Aggregator synthesis completes in <2 seconds
- [ ] Total latency for simple queries: 3-5 seconds
- [ ] Quality improvement of 15-25% vs single model

**Technical Details:**
```
Query → [Proposer 1, Proposer 2, Proposer 3] (parallel)
      → Aggregator(proposals)
      → Final Response
```

#### FR1.2: Prompt Engineering for Debate
**Priority:** P0 (Critical)

**Proposer Prompt Template:**
```
You are an expert coding assistant. Analyze this query and provide
a clear, concise response. Focus on accuracy and code correctness.

Query: {user_query}

Response:
```

**Aggregator Prompt Template:**
```
You are synthesizing multiple responses to create the best answer.
Review these 3 proposals and create a unified response that:
- Takes the best ideas from each
- Resolves contradictions
- Provides the most accurate and helpful answer

Proposal 1: {prop1}
Proposal 2: {prop2}
Proposal 3: {prop3}

Synthesized Response:
```

**Acceptance Criteria:**
- [ ] Prompts optimized for Qwen2.5-3B model characteristics
- [ ] Proposer outputs are 150-300 tokens
- [ ] Aggregator output is 200-500 tokens
- [ ] Measured quality improvement on benchmark tasks

#### FR1.3: Adaptive Routing Logic
**Priority:** P1 (High)

**Specification:**
Implement heuristic-based routing to decide local vs cloud:

**Local Processing Triggers:**
- Query contains code snippet
- Keywords: "explain", "write", "generate", "debug"
- Token count < 2000
- Task type: factual Q&A, simple code gen

**Cloud Routing Triggers:**
- Keywords: "design", "architect", "complex refactoring"
- Token count > 2000
- Multi-file analysis required
- Reasoning depth > 3 steps
- User explicitly requests "use cloud"

**Acceptance Criteria:**
- [ ] Routing decision made in <100ms
- [ ] Precision: >85% (correct routing decisions)
- [ ] Recall: >80% (catch complex queries)
- [ ] User override option available

**Example Routing Rules:**
```rust
enum RoutingDecision {
    Local(DebateConfig),
    Cloud(CloudProvider),
}

fn route_query(query: &Query) -> RoutingDecision {
    if query.contains_keywords(&["design", "architect"]) {
        return RoutingDecision::Cloud(CloudProvider::Claude);
    }
    if query.code_snippet_lines() < 100 && query.is_factual() {
        return RoutingDecision::Local(DebateConfig::default());
    }
    // ... more rules
}
```

### FR2: Model Management

#### FR2.1: Primary Model Configuration
**Priority:** P0 (Critical)

**Specification:**
- Model: Qwen2.5-3B-Instruct
- Quantization: Q4_K_M (optimal quality/performance trade-off)
- Format: GGUF (llama.cpp compatible)
- Size: ~1.7GB quantized
- Context window: 12K tokens total (4K per parallel slot)

**Acceptance Criteria:**
- [ ] Auto-download from HuggingFace on first run
- [ ] SHA256 checksum validation
- [ ] Model loads in <5 seconds warm start
- [ ] Model persists at `~/.cache/pensieve/models/qwen2.5-3b-q4km.gguf`

#### FR2.2: Context Management
**Priority:** P1 (High)

**Specification:**
- Total context budget: 12K tokens
- Allocation per parallel slot: 4K tokens each (3 slots)
- KV cache quantization: Q8_0 (8-bit, minimal quality loss)
- Context overflow handling: Truncate oldest, keep system prompt

**Acceptance Criteria:**
- [ ] Each proposer agent gets 4K context
- [ ] System prompt + user query + response fit in 4K
- [ ] KV cache uses <2GB memory for 3 parallel slots
- [ ] Graceful truncation with user warning

**Memory Calculation:**
```
Per-token KV cache (Qwen 2.5-3B):
= layers × 2 × kv_heads × head_dim × dtype
= 36 × 2 × 8 × 128 × 1 byte (Q8)
≈ 64 KB per token

Per slot (4K context): 64KB × 4096 = 256 MB
Total (3 slots): 256MB × 3 = 768 MB ✓
```

### FR3: Runtime Infrastructure

#### FR3.1: llama-server Configuration
**Priority:** P0 (Critical)

**Specification:**
- Runtime: llama.cpp server (latest stable)
- Parallel slots: 3 (for 3 proposer agents)
- Continuous batching: Enabled
- GPU offload: Full (all layers to Metal GPU)
- Batch size: 512 tokens
- Micro-batch size: 256 tokens

**Launch Command:**
```bash
./llama-server \
  --model ~/.cache/pensieve/models/qwen2.5-3b-q4km.gguf \
  --ctx-size 12288 \
  --parallel 3 \
  --cont-batching \
  --batch-size 512 \
  --ubatch-size 256 \
  --n-gpu-layers 999 \
  --threads 4 \
  --port 8080 \
  --metrics
```

**Acceptance Criteria:**
- [ ] Server starts in <5 seconds
- [ ] Health check endpoint responds
- [ ] All 3 slots available
- [ ] Prometheus metrics exposed at `/metrics`
- [ ] Graceful shutdown on SIGTERM

#### FR3.2: Async Request Handling
**Priority:** P0 (Critical)

**Specification:**
- Language: Rust with Tokio async runtime
- HTTP client: reqwest with streaming support
- Parallel proposer requests: tokio::join! for 3 concurrent requests
- Request timeout: 30 seconds per inference
- Retry logic: 2 retries with exponential backoff

**Implementation Pattern:**
```rust
use tokio::time::timeout;
use reqwest::Client;

async fn parallel_debate(query: &str) -> Result<String> {
    let client = Client::new();

    // Fire 3 proposer requests in parallel
    let (prop1, prop2, prop3) = tokio::join!(
        generate_proposal(&client, query, 1),
        generate_proposal(&client, query, 2),
        generate_proposal(&client, query, 3),
    );

    // Aggregate results
    let final_response = aggregate_proposals(
        prop1?, prop2?, prop3?
    ).await?;

    Ok(final_response)
}
```

**Acceptance Criteria:**
- [ ] 3 proposer requests complete in parallel
- [ ] No blocking calls in async context
- [ ] Proper error propagation
- [ ] Request cancellation on timeout

#### FR3.3: Streaming Response Protocol
**Priority:** P1 (High)

**Specification:**
- Protocol: Server-Sent Events (SSE)
- Events: `message_start`, `content_block_delta`, `message_stop`
- Token buffering: 5-10 tokens per chunk
- Backpressure handling: Pause generation if client slow

**SSE Format:**
```
event: message_start
data: {"type":"message_start","message":{"role":"assistant"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"text":"Sure, I can help"}}

event: message_stop
data: {"type":"message_stop"}
```

**Acceptance Criteria:**
- [ ] Streaming works in all proposer + aggregator phases
- [ ] User sees first token within 500ms
- [ ] Smooth token delivery (no stuttering)
- [ ] Proper SSE connection cleanup

### FR4: Web Search Integration

#### FR4.1: Parallel Web Search
**Priority:** P1 (High)

**Specification:**
- Trigger: Detect queries needing current information (date references, "latest", "current")
- Provider: Tavily API (primary) or Brave Search API (fallback)
- Parallelism: Fire web search concurrently with first proposer
- Result integration: Include top 3 search results in aggregator context

**Workflow:**
```
Query → [
  Parallel: Web Search API call,
  Parallel: Proposer 1, Proposer 2, Proposer 3
] → Aggregator(proposals + search_results) → Response
```

**Acceptance Criteria:**
- [ ] Web search adds <2 seconds latency (parallel execution)
- [ ] Search results formatted as: "Source: [title](url) - snippet"
- [ ] Max 3 search results included
- [ ] Search fails gracefully (continue without results)
- [ ] User sees source citations in response

#### FR4.2: Search Result Formatting
**Priority:** P2 (Medium)

**Specification:**
Format search results for aggregator consumption:

```
Web Search Results (3 sources):
1. [Tokio Documentation - Async/Await](https://docs.rs/tokio)
   "Tokio is a runtime for writing reliable asynchronous applications..."

2. [Rust Async Book](https://rust-lang.github.io/async-book)
   "The async/await syntax in Rust allows you to write asynchronous code..."

3. [Stack Overflow: Common Tokio Patterns](https://stackoverflow.com/...)
   "Here are the most common patterns when using Tokio 1.x..."
```

**Acceptance Criteria:**
- [ ] Citations appear in final response
- [ ] URLs are clickable
- [ ] Snippets are 50-100 words
- [ ] Deduplication of similar results

### FR5: Cloud API Handoff

#### FR5.1: Structured Handoff Protocol
**Priority:** P1 (High)

**Specification:**
When routing to cloud, send structured JSON with:
- User query (original)
- Proposer outputs (if any)
- Local model confidence score
- Task type classification
- Context (code snippets, error messages)

**Handoff Format:**
```json
{
  "query": "Design a distributed caching architecture for this service",
  "task_type": "architectural_design",
  "local_attempts": {
    "proposer_1": "...",
    "proposer_2": "...",
    "proposer_3": "..."
  },
  "confidence": 0.35,
  "context": {
    "code_snippets": [...],
    "language": "rust",
    "framework": "tokio"
  },
  "routing_reason": "Complex multi-component architecture design"
}
```

**Token Budget:** 150-300 tokens for handoff

**Acceptance Criteria:**
- [ ] Handoff preserves all user intent
- [ ] Claude receives sufficient context
- [ ] Handoff adds <200ms latency
- [ ] Structured format enables better Claude responses

#### FR5.2: Cloud Provider Configuration
**Priority:** P1 (High)

**Specification:**
- Primary: Anthropic Claude API (Sonnet 4.5)
- Fallback: OpenAI GPT-4o (if Claude unavailable)
- API key management: Environment variables
- Rate limiting: Respect provider limits
- Cost tracking: Log tokens consumed

**Acceptance Criteria:**
- [ ] API keys loaded from env vars
- [ ] Graceful fallback if primary fails
- [ ] User notified when using cloud
- [ ] Cost per query logged

### FR6: Quality Monitoring

#### FR6.1: Response Quality Metrics
**Priority:** P2 (Medium)

**Specification:**
Track quality indicators:
- Response coherence score (auto-eval)
- Code correctness (syntax validation)
- User feedback (thumbs up/down)
- Task completion rate
- Comparison to cloud baseline

**Acceptance Criteria:**
- [ ] Metrics logged per query
- [ ] Daily quality reports generated
- [ ] Alerts if quality drops below 60%
- [ ] Export to Prometheus

#### FR6.2: User Feedback Collection
**Priority:** P2 (Medium)

**Specification:**
- Inline feedback: Thumbs up/down after response
- Detailed feedback: Optional text input
- Storage: SQLite local database
- Export: CSV for analysis

**Acceptance Criteria:**
- [ ] Feedback UI in CLI/IDE plugin
- [ ] Feedback stored with query/response
- [ ] Privacy-preserving (no PII)

---

## 6. Non-Functional Requirements

### NFR1: Performance

#### NFR1.1: Latency Targets
**Priority:** P0 (Critical)

| Query Type | Time Budget | Breakdown |
|------------|-------------|-----------|
| **Simple (local)** | 3-5 seconds | Proposers: 2-3s, Aggregator: 1-2s |
| **Web-augmented** | 5-8 seconds | Web: +2s parallel, else same |
| **Complex (cloud)** | 11-18 seconds | Local attempt: 5s, Cloud: 6-13s |

**Measurement:**
- p50 latency: <5 seconds for 80% of queries
- p95 latency: <10 seconds
- p99 latency: <18 seconds

**Acceptance Criteria:**
- [ ] 95% of simple queries complete in <5 seconds
- [ ] No query times out before 30 seconds
- [ ] Latency metrics exposed via Prometheus

#### NFR1.2: Throughput Targets
**Priority:** P0 (Critical)

**Specification:**
- Per-stream token generation: 35-45 tokens/second
- Parallel streams: 3 (proposers)
- Aggregate throughput: 105-135 tokens/second during proposer phase

**Measured on:** Mac Mini M4 with 64GB unified memory

**Acceptance Criteria:**
- [ ] Each proposer generates 150-300 tokens in <7 seconds
- [ ] Aggregator generates 200-500 tokens in <10 seconds
- [ ] No thermal throttling during sustained load
- [ ] Consistent performance over 8-hour sessions

#### NFR1.3: Resource Utilization
**Priority:** P0 (Critical)

**Memory Budget:**
| Component | Allocation | Notes |
|-----------|------------|-------|
| Model weights (shared) | 1.7 GB | Q4_K_M quantized |
| KV cache (3 slots × 4K) | 768 MB | Q8_0 quantization |
| Activation buffers | 512 MB | Temporary GPU memory |
| Framework overhead | 512 MB | llama.cpp + Rust |
| **Total** | **3.5 GB** | <4GB target ✓ |

**Acceptance Criteria:**
- [ ] Peak memory usage <4GB under load
- [ ] Sustained memory usage <3.5GB
- [ ] No memory leaks over 24-hour run
- [ ] GPU memory efficiently released between queries

**GPU Utilization:**
- Target: 80-95% during inference
- Idle: <5% when not processing
- Power draw: <40W sustained (Mac Mini thermal limit)

### NFR2: Reliability

#### NFR2.1: Error Handling
**Priority:** P0 (Critical)

**Specification:**
- Graceful degradation: If 1 proposer fails, continue with 2
- Retry logic: 2 retries with exponential backoff
- Fallback: Route to cloud if local repeatedly fails
- User notification: Clear error messages

**Error Categories:**
1. **Transient:** Network timeout, GPU busy → Retry
2. **Resource:** OOM, GPU error → Reduce batch size, retry
3. **Model:** Bad output, context overflow → Truncate, retry
4. **Fatal:** Model missing, GPU unavailable → Clear error, exit

**Acceptance Criteria:**
- [ ] No crashes on transient errors
- [ ] User-friendly error messages
- [ ] Automatic recovery from transient failures
- [ ] Error rates logged to metrics

#### NFR2.2: Availability
**Priority:** P1 (High)

**Target:** 99% uptime during work hours (8am-6pm local time)

**Failure Modes:**
- llama-server crash: Auto-restart
- GPU hang: Detect + restart
- Model corruption: Re-download
- Configuration error: Validate on startup

**Acceptance Criteria:**
- [ ] Automatic restart on server crash
- [ ] Health checks every 30 seconds
- [ ] Startup validation of all components
- [ ] Uptime tracking

### NFR3: Usability

#### NFR3.1: Zero-Config First Run
**Priority:** P0 (Critical)

**Specification:**
User should run a single command and get working system:

```bash
$ pensieve-debate-assistant
[1/3] Checking system requirements... ✓ (Mac M4, 64GB RAM)
[2/3] Downloading model (1.7GB)... [=========>    ] 60% ETA 2m
[3/3] Starting server... ✓
Ready! Ask me anything at http://localhost:8080
```

**Acceptance Criteria:**
- [ ] No config file required
- [ ] All dependencies bundled
- [ ] Model auto-downloads
- [ ] First query works within 10 minutes of install

#### NFR3.2: Developer Experience
**Priority:** P1 (High)

**Specification:**
- CLI interface with REPL mode
- HTTP API compatible with OpenAI format (easy IDE integration)
- VS Code extension (future)
- JetBrains plugin (future)

**Acceptance Criteria:**
- [ ] CLI has syntax highlighting
- [ ] HTTP API documented (OpenAPI spec)
- [ ] Example clients in Python, JavaScript
- [ ] Streaming responses work in all clients

### NFR4: Security & Privacy

#### NFR4.1: Local-First Privacy
**Priority:** P0 (Critical)

**Specification:**
- Default: All processing local, no telemetry
- Cloud routing: Explicit user consent required
- API keys: Stored in OS keychain (not plain text)
- Logs: No PII, code snippets redacted

**Acceptance Criteria:**
- [ ] No data sent to external servers without consent
- [ ] API keys encrypted at rest
- [ ] Audit log of cloud API calls
- [ ] Privacy policy documented

#### NFR4.2: Code Safety
**Priority:** P1 (High)

**Specification:**
- Sandbox execution: Generated code NOT auto-executed
- Static analysis: Check generated code for obvious vulnerabilities
- User review: Always show code before suggesting execution

**Acceptance Criteria:**
- [ ] No code execution without explicit user command
- [ ] Warning on potentially dangerous operations (rm -rf, curl | sh)
- [ ] Safe by default

### NFR5: Maintainability

#### NFR5.1: Code Quality
**Priority:** P1 (High)

**Specification:**
- Language: Rust (type-safe, memory-safe)
- Testing: >80% code coverage
- Documentation: All public APIs documented
- Idiomatic Rust: Follow Rust API guidelines

**Acceptance Criteria:**
- [ ] `cargo test` passes
- [ ] `cargo clippy` has zero warnings
- [ ] `cargo fmt` enforced
- [ ] Documentation builds without warnings

#### NFR5.2: Observability
**Priority:** P1 (High)

**Specification:**
- Structured logging: tracing crate
- Metrics: Prometheus endpoint
- Tracing: OpenTelemetry support
- Diagnostics: Debug mode with verbose logs

**Acceptance Criteria:**
- [ ] All operations logged with context
- [ ] Key metrics exported (latency, throughput, errors)
- [ ] Tracing spans for distributed flow
- [ ] Diagnostic mode for troubleshooting

---

## 7. Technical Architecture Overview

### 7.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│  (CLI REPL, HTTP API Server, Future: IDE Plugins)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Orchestration Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Query     │  │   Adaptive   │  │   Response           │   │
│  │   Parser    │→ │   Router     │→ │   Synthesizer        │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│                           │                                       │
│                     ┌─────┴──────┐                               │
│                     ▼            ▼                               │
│              ┌──────────┐  ┌──────────┐                          │
│              │  Local   │  │  Cloud   │                          │
│              │  Debate  │  │  Handoff │                          │
│              └──────────┘  └──────────┘                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴───────────────────────┐
        ▼                                            ▼
┌──────────────────────────────┐       ┌───────────────────────────┐
│    Local Inference Engine    │       │    Cloud API Gateway      │
│  ┌────────────────────────┐  │       │  ┌─────────────────────┐ │
│  │  Multi-Agent Debate    │  │       │  │ Anthropic Claude    │ │
│  │  (MoA-Lite 2-layer)    │  │       │  │ API Client          │ │
│  │                        │  │       │  ├─────────────────────┤ │
│  │  Proposer 1 ┐          │  │       │  │ OpenAI GPT-4o       │ │
│  │  Proposer 2 ├─parallel │  │       │  │ API Client (backup) │ │
│  │  Proposer 3 ┘          │  │       │  └─────────────────────┘ │
│  │      ↓                 │  │       └───────────────────────────┘
│  │  Aggregator            │  │
│  └────────────────────────┘  │
│            ↓                  │
│  ┌────────────────────────┐  │
│  │  llama-server          │  │
│  │  (llama.cpp)           │  │
│  │  - Parallel slots: 3   │  │
│  │  - Continuous batching │  │
│  │  - Metal GPU offload   │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
          ↓
┌──────────────────────────────┐
│   Hardware: Mac Mini M4      │
│   - Unified Memory: 64GB     │
│   - Metal GPU Acceleration   │
│   - 273 GB/s Bandwidth       │
└──────────────────────────────┘
```

### 7.2 Component Descriptions

#### Orchestration Layer (Rust)
**Responsibilities:**
- Parse and classify incoming queries
- Route to local debate or cloud based on complexity heuristics
- Synthesize final response with citations
- Manage timeouts and retries

**Key Modules:**
- `query_router.rs` - Routing decision logic
- `debate_orchestrator.rs` - Multi-agent debate coordination
- `web_search.rs` - Tavily/Brave API integration
- `cloud_handoff.rs` - Claude API client

#### Local Inference Engine (llama.cpp + Rust)
**Responsibilities:**
- Load and manage Qwen2.5-3B model
- Handle parallel inference requests
- Manage KV cache and context windows
- Stream tokens back to orchestrator

**Key Components:**
- `llama-server` process (external binary)
- `llama_client.rs` - HTTP client to llama-server
- `prompt_templates.rs` - Debate prompts
- `token_streaming.rs` - SSE handling

#### Cloud API Gateway (Rust)
**Responsibilities:**
- Authenticate with Anthropic/OpenAI
- Format structured handoff requests
- Handle API rate limits and retries
- Track costs and usage

**Key Modules:**
- `anthropic_client.rs` - Claude API wrapper
- `openai_client.rs` - GPT-4o fallback
- `api_auth.rs` - Keychain integration
- `usage_tracker.rs` - Cost tracking

### 7.3 Data Flow: Simple Query (Local)

```
1. User Query: "Explain this Rust function"
   ↓
2. Query Parser: Extract code snippet, classify as "code_explanation"
   ↓
3. Adaptive Router: Task=simple, route_decision=Local
   ↓
4. Debate Orchestrator: Prepare 3 proposer prompts
   ↓
5. Parallel Inference (tokio::join!):
   - Proposer 1 → llama-server slot 0 → 180 tokens in 4.2s
   - Proposer 2 → llama-server slot 1 → 220 tokens in 5.1s
   - Proposer 3 → llama-server slot 2 → 195 tokens in 4.7s
   (Total wall time: 5.1s - parallel execution)
   ↓
6. Aggregator Inference:
   - Input: 3 proposals (595 tokens)
   - Output: 320 tokens in 7.8s
   ↓
7. Response Synthesizer: Format final output
   ↓
8. Stream to User: SSE, first token at 5.2s, complete at 12.9s
```

**Total Latency:** ~13 seconds (within 11-18s target for quality)

### 7.4 Data Flow: Complex Query (Cloud)

```
1. User Query: "Design a distributed caching architecture"
   ↓
2. Query Parser: No code, keywords "design", "architecture"
   ↓
3. Adaptive Router: Task=architectural_design, route_decision=Cloud
   ↓
4. Cloud Handoff:
   - Optionally run 1 proposer locally for context (3s)
   - Format structured handoff JSON (150 tokens)
   - Send to Claude API
   ↓
5. Claude Response: 800 tokens in 10.2s
   ↓
6. Response Synthesizer: Add metadata (source: cloud)
   ↓
7. Stream to User: Total 13.2s
```

### 7.5 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Application** | Rust (tokio) | Memory safety, performance, async |
| **Inference Runtime** | llama.cpp | Battle-tested, best Metal performance |
| **Model Format** | GGUF | De facto standard, wide compatibility |
| **HTTP Server** | Axum | Async, tokio-native, ergonomic |
| **HTTP Client** | reqwest | Async, streaming, production-ready |
| **Web Search** | Tavily API | LLM-optimized search results |
| **Cloud AI** | Anthropic Claude | Best-in-class reasoning |
| **Metrics** | Prometheus | Industry standard observability |
| **Logging** | tracing | Structured, async-aware |

### 7.6 Deployment Architecture

**Single Binary Deployment:**
```
pensieve-debate-assistant (Rust binary)
├── Embeds: prompt templates, routing rules
├── Downloads: Qwen2.5-3B model (1.7GB) on first run
├── Spawns: llama-server subprocess
├── Listens: HTTP server on port 8080
└── Stores: Config in ~/.config/pensieve/
```

**Directory Structure:**
```
~/.cache/pensieve/
├── models/
│   └── qwen2.5-3b-q4km.gguf (1.7GB)
├── metrics/
│   └── query_logs.db (SQLite)
└── tmp/

~/.config/pensieve/
├── config.toml (optional overrides)
└── api_keys.enc (encrypted)
```

---

## 8. Success Metrics & KPIs

### 8.1 Product Metrics

#### Primary KPI: Quality Score
**Definition:** Average quality rating vs Claude baseline on benchmark task set

**Target:** 65-75% of Claude Sonnet quality

**Measurement:**
- Automated: HumanEval, MBPP code benchmarks
- Human: Weekly blind A/B test (local vs Claude)
- Sample size: 100 queries/week

**Formula:**
```
Quality Score = (Local Debate Score / Claude Score) × 100%
```

**Success Criteria:**
- Month 1: >60% quality
- Month 3: >65% quality
- Month 6: >70% quality

#### Secondary KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Cost Savings** | 70-85% reduction | Cloud API spend vs pure cloud |
| **User Satisfaction** | >4.0/5.0 | Post-query thumbs up rate |
| **Adoption** | 100 weekly active users | Unique IPs to API server |
| **Latency (p95)** | <10 seconds | Prometheus histogram |
| **Uptime** | >99% during 8am-6pm | Health check monitoring |

### 8.2 Technical Metrics

#### Inference Performance

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Tokens/sec (per stream)** | 35-45 | <30 |
| **Proposer latency (p95)** | <6 seconds | >8 seconds |
| **Aggregator latency (p95)** | <10 seconds | >12 seconds |
| **End-to-end latency (p95)** | <10 seconds | >15 seconds |

#### Resource Utilization

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Memory usage** | 3.5GB avg | >4.5GB |
| **GPU utilization** | 80-95% during inference | <60% |
| **CPU usage** | <30% avg | >60% sustained |
| **Power draw** | <40W | >50W |

#### Reliability

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Error rate** | <2% | >5% |
| **Retry rate** | <5% | >10% |
| **Cloud fallback rate** | <15% | >25% |
| **Timeout rate** | <1% | >3% |

### 8.3 Quality Breakdown by Task Type

| Task Type | Target Quality | Measurement Method |
|-----------|---------------|---------------------|
| **Code Explanation** | 70-75% | BLEU score vs Claude reference |
| **Simple Code Gen** | 65-70% | Pass@1 on HumanEval subset |
| **API Documentation Q&A** | 75-80% | Factual accuracy (human eval) |
| **Debugging** | 60-70% | Resolution rate on known bugs |
| **Test Generation** | 70-75% | Mutation testing score |
| **Code Review** | 65-70% | Issue detection rate |

### 8.4 Business Metrics (OSS Project Context)

| Metric | Definition | Target |
|--------|------------|--------|
| **Cost per Query** | Total cost / queries served | <$0.01 |
| **Cloud API Reduction** | % queries handled locally | >75% |
| **Developer Productivity** | Time saved per dev/week | >2 hours |
| **Model Download Rate** | New installations/week | 50+ (Month 3) |

### 8.5 Monitoring Dashboard

**Real-Time Metrics (Grafana):**
- Queries per second
- p50/p95/p99 latency
- Error rate (last hour)
- GPU temperature and utilization
- Memory usage trend
- Cloud API costs (today)

**Daily Reports:**
- Total queries: Local vs Cloud breakdown
- Quality scores (automated benchmarks)
- User feedback summary
- Cost analysis
- Top error types

**Weekly Reviews:**
- Quality trend analysis
- User satisfaction survey results
- Performance regression tests
- Resource optimization opportunities

---

## 9. Risks & Mitigations

### 9.1 Technical Risks

#### Risk 1: Thermal Throttling (Mac Mini M4)
**Probability:** Medium
**Impact:** High (25% throughput loss)

**Description:**
Mac Mini M4 is known to thermally throttle under sustained GPU load after 15-30 minutes, reducing memory bandwidth from 273 GB/s to ~200 GB/s.

**Mitigation Strategy:**
1. **Immediate:** Enable High Performance mode (`pmset -a highpowermode 1`)
2. **Short-term:** Implement adaptive batch sizing (reduce when temp >90°C)
3. **Medium-term:** Monitor GPU temperature, display warning at 85°C
4. **Long-term:** Recommend Mac Studio for high-volume users

**Detection:**
```rust
// Monitor via powermetrics
if gpu_temp > 90.0 {
    reduce_batch_size();
    warn_user("GPU running hot, throttling to prevent overheating");
}
```

**Residual Risk:** Low (with mitigations)

#### Risk 2: Model Quality Insufficient (<60% vs Claude)
**Probability:** Medium
**Impact:** Critical (product not viable)

**Description:**
Qwen2.5-3B might not reach 65% quality target even with debate architecture.

**Mitigation Strategy:**
1. **Immediate:** Benchmark on HumanEval before v1.0 launch
2. **Short-term:** If <60%, upgrade to Qwen2.5-7B (requires 8GB RAM)
3. **Medium-term:** Implement 3-layer debate (4 proposers + 1 aggregator)
4. **Long-term:** Fine-tune aggregator model on high-quality examples

**Pivot Plan:**
- If 3B quality <55%: Switch to 7B model (increase memory to 6GB)
- If 7B still insufficient: Increase cloud routing threshold (more cloud calls)

**Residual Risk:** Medium (model selection is hypothesis)

#### Risk 3: Memory Budget Exceeded (>4GB)
**Probability:** Low
**Impact:** High (crashes, requires hardware upgrade)

**Description:**
KV cache or activation memory could exceed 4GB budget under load.

**Mitigation Strategy:**
1. **Immediate:** Implement memory pressure monitoring
2. **Short-term:** Reduce context from 4K to 2K per slot if needed
3. **Medium-term:** More aggressive KV cache quantization (Q4 instead of Q8)
4. **Long-term:** Implement KV cache eviction (drop old tokens)

**Detection:**
```rust
if memory_usage > 3.8_GB {
    reduce_context_window();
    alert_monitoring_system();
}
```

**Residual Risk:** Very Low (conservative budget with headroom)

#### Risk 4: llama-server Instability
**Probability:** Low
**Impact:** High (service downtime)

**Description:**
llama.cpp server could crash due to bugs, GPU driver issues, or memory errors.

**Mitigation Strategy:**
1. **Immediate:** Wrap llama-server in supervisor (auto-restart on crash)
2. **Short-term:** Health checks every 30s, restart if unhealthy
3. **Medium-term:** Pre-flight hardware validation (GPU test on startup)
4. **Long-term:** Contribute stability fixes upstream to llama.cpp

**Implementation:**
```rust
// Supervisor process
loop {
    let status = Command::new("llama-server").status();
    if status.is_err() {
        log::error!("llama-server crashed, restarting...");
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
```

**Residual Risk:** Low (auto-restart mitigates)

### 9.2 Product Risks

#### Risk 5: Insufficient User Adoption
**Probability:** Medium
**Impact:** High (project not viable)

**Description:**
Developers might not find the tool valuable enough to install and use regularly.

**Root Causes:**
- Setup complexity too high
- Quality not good enough
- Latency too slow
- Existing tools "good enough"

**Mitigation Strategy:**
1. **Immediate:** Focus on zero-config setup (install to first query <10 min)
2. **Short-term:** Target specific pain point: "local code understanding"
3. **Medium-term:** Build IDE plugins (VS Code, JetBrains) for seamless UX
4. **Long-term:** Community building, showcase success stories

**Validation:**
- Month 1: 10 beta users (hand-picked)
- Month 2: 50 users (public alpha)
- Month 3: 100 users (public beta)

**Pivot Plan:**
If adoption stalls, pivot to:
- Narrower use case (e.g., only Rust code)
- Enterprise offering (internal company deployment)
- Cloud-hosted version (trade privacy for ease)

**Residual Risk:** Medium (market validation required)

#### Risk 6: Cloud API Costs Higher Than Expected
**Probability:** Low
**Impact:** Medium (user costs increase)

**Description:**
Adaptive router might send more queries to cloud than predicted, reducing cost savings.

**Mitigation Strategy:**
1. **Immediate:** Conservative routing (prefer local unless clearly too complex)
2. **Short-term:** User-configurable cloud budget ($10/month limit)
3. **Medium-term:** Learn from user feedback (which routed queries were unnecessary)
4. **Long-term:** Fine-tune router with reinforcement learning

**Monitoring:**
```rust
// Alert if cloud routing exceeds threshold
if cloud_calls_percentage > 25.0 {
    alert_user("Cloud usage high this week: $X.XX");
    suggest_router_tuning();
}
```

**Residual Risk:** Low (user control + monitoring)

### 9.3 External Dependencies

#### Risk 7: Anthropic API Changes/Pricing
**Probability:** Medium
**Impact:** Medium (fallback required)

**Description:**
Anthropic could change API format, increase pricing, or deprecate Claude Sonnet.

**Mitigation Strategy:**
1. **Immediate:** Support multiple providers (OpenAI GPT-4o as fallback)
2. **Short-term:** Abstract cloud provider interface
3. **Medium-term:** Add Gemini, DeepSeek as additional options
4. **Long-term:** Improve local quality to reduce cloud dependency

**Interface:**
```rust
trait CloudProvider {
    async fn generate(&self, prompt: &str) -> Result<String>;
}

// Implementations: ClaudeProvider, OpenAIProvider, GeminiProvider
```

**Residual Risk:** Low (multiple fallbacks)

#### Risk 8: Web Search API Limits/Costs
**Probability:** Low
**Impact:** Low (feature degradation)

**Description:**
Tavily or Brave could rate-limit or become expensive.

**Mitigation Strategy:**
1. **Immediate:** Cache search results (1 hour TTL)
2. **Short-term:** Support multiple search providers
3. **Medium-term:** Implement local web search (Searx)
4. **Long-term:** Make web search optional feature

**Residual Risk:** Very Low (non-critical feature)

### 9.4 Risk Summary Matrix

| Risk | Probability | Impact | Mitigation Priority | Residual Risk |
|------|-------------|--------|---------------------|---------------|
| Thermal throttling | Medium | High | P1 | Low |
| Model quality <60% | Medium | Critical | P0 | Medium |
| Memory budget exceeded | Low | High | P2 | Very Low |
| llama-server instability | Low | High | P1 | Low |
| Poor user adoption | Medium | High | P0 | Medium |
| Cloud costs too high | Low | Medium | P2 | Low |
| API provider changes | Medium | Medium | P2 | Low |
| Web search limits | Low | Low | P3 | Very Low |

**Overall Risk Assessment:** MEDIUM (quality and adoption are key uncertainties)

---

## 10. MVP Scope vs Future Phases

### 10.1 MVP (v1.0) - "Core Debate Assistant"

**Timeline:** 12 weeks
**Goal:** Validate multi-agent debate quality improvement on local OSS coding tasks

#### MVP Feature Set

**In Scope:**
- ✅ 2-layer MoA-Lite architecture (3 proposers + 1 aggregator)
- ✅ Qwen2.5-3B model with Q4_K_M quantization
- ✅ llama-server runtime (3 parallel slots)
- ✅ Basic adaptive routing (heuristic-based)
- ✅ CLI REPL interface
- ✅ HTTP API (OpenAI-compatible format)
- ✅ Anthropic Claude handoff (structured JSON)
- ✅ Basic web search integration (Tavily API)
- ✅ Prometheus metrics endpoint
- ✅ Single binary deployment (Mac M4)

**Out of Scope (Future):**
- ❌ IDE plugins (VS Code, JetBrains)
- ❌ Multi-model support (only Qwen2.5-3B)
- ❌ Fine-tuning or model training
- ❌ Cross-platform (Linux, Windows)
- ❌ Distributed deployment
- ❌ Advanced routing (RL-based)
- ❌ UI (web or desktop)

#### MVP Quality Bar

| Metric | MVP Target | Stretch Goal |
|--------|------------|--------------|
| Quality vs Claude | >60% | >65% |
| Simple query latency | <8 seconds | <5 seconds |
| Complex query latency | <20 seconds | <15 seconds |
| Memory usage | <5GB | <4GB |
| Uptime | >95% | >99% |

#### MVP Success Criteria

**Launch Criteria:**
- [ ] Zero-config install works on Mac M4
- [ ] Model downloads and loads successfully
- [ ] 10 benchmark queries complete successfully
- [ ] Quality >60% on HumanEval subset (25 problems)
- [ ] 5 beta users complete 10 queries each
- [ ] No P0 bugs in issue tracker

**Definition of Success (3 months post-launch):**
- 100+ weekly active users
- >65% quality score on benchmarks
- <5% error rate
- 4.0+ user satisfaction rating
- Published blog post with results

### 10.2 Phase 2 (v1.5) - "IDE Integration"

**Timeline:** 8 weeks after v1.0
**Goal:** Seamless developer workflow integration

**Features:**
- VS Code extension with inline suggestions
- JetBrains plugin (IntelliJ, PyCharm)
- Git integration (code review on diffs)
- Codebase indexing (RAG for project context)
- Improved routing with user feedback loop

**Success Metrics:**
- 50% of users use IDE plugin
- Average 20+ queries/day per active user
- Quality improvement to 70% (via RAG context)

### 10.3 Phase 3 (v2.0) - "Advanced Intelligence"

**Timeline:** 6 months after v1.0
**Goal:** Best-in-class local coding assistant

**Features:**
- 3-layer debate architecture (more proposers)
- Upgrade to Qwen2.5-7B model (better quality)
- Learned routing (RL-based, adaptive)
- Multi-file analysis (project-level understanding)
- Fine-tuned aggregator model
- Conversation memory (multi-turn dialogue)
- Code execution sandbox (validate generated code)

**Success Metrics:**
- 75% quality vs Claude
- 1000+ weekly active users
- Featured in "awesome-llm-tools" lists
- Published research paper on debate architecture

### 10.4 Future Phases

#### Phase 4: Cross-Platform (v2.5)
- Linux support (CUDA, ROCm)
- Windows support (DirectML)
- Cloud deployment option (Docker, Kubernetes)
- Model serving for teams (1 server, N clients)

#### Phase 5: Enterprise (v3.0)
- Multi-tenant support
- SSO/LDAP authentication
- Audit logging and compliance
- On-premise deployment
- Custom model fine-tuning

### 10.5 Feature Prioritization Framework

**Priority = Impact × Feasibility / Effort**

| Feature | Impact (1-10) | Feasibility (1-10) | Effort (weeks) | Priority Score |
|---------|---------------|-------------------|----------------|----------------|
| Core debate (MVP) | 10 | 8 | 8 | 10.0 |
| Web search | 7 | 9 | 2 | 31.5 |
| VS Code plugin | 8 | 7 | 4 | 14.0 |
| RAG codebase context | 9 | 6 | 6 | 9.0 |
| 3-layer debate | 6 | 8 | 3 | 16.0 |
| Learned routing | 7 | 5 | 8 | 4.4 |
| Multi-model support | 5 | 7 | 5 | 7.0 |
| Cloud deployment | 6 | 8 | 4 | 12.0 |

**Rank Order (for sequencing):**
1. Core debate (10.0) → MVP
2. Web search (31.5) → MVP
3. 3-layer debate (16.0) → Phase 3
4. VS Code plugin (14.0) → Phase 2
5. Cloud deployment (12.0) → Phase 4
6. RAG codebase (9.0) → Phase 2
7. Multi-model (7.0) → Phase 4
8. Learned routing (4.4) → Phase 3

---

## 11. Dependencies & Assumptions

### 11.1 Technical Dependencies

#### External Software

| Dependency | Version | Purpose | License | Risk |
|------------|---------|---------|---------|------|
| **llama.cpp** | Latest stable | Inference runtime | MIT | Low (mature) |
| **Rust** | 1.75+ | Application language | MIT/Apache-2.0 | None |
| **Tokio** | 1.35+ | Async runtime | MIT | None |
| **Axum** | 0.7+ | HTTP server | MIT | Low |
| **reqwest** | 0.11+ | HTTP client | MIT/Apache-2.0 | Low |

#### External APIs

| API | Purpose | Cost | Rate Limits | Risk |
|-----|---------|------|-------------|------|
| **Anthropic Claude** | Complex reasoning | $3/$15 per 1M tokens | 50 req/min (Tier 1) | Medium (pricing) |
| **Tavily Search** | Web search | $0.50/1K searches | 1K/month free | Low (non-critical) |
| **OpenAI GPT-4o** | Fallback cloud | $2.50/$10 per 1M tokens | 500 req/min | Low (backup) |

#### Hardware Requirements

| Component | Minimum | Recommended | Critical? |
|-----------|---------|-------------|-----------|
| **CPU** | M1/M2/M3/M4 | M4 | Yes |
| **RAM** | 16GB | 32GB+ | Yes |
| **Storage** | 5GB free | 10GB+ | Yes |
| **GPU** | 7-core+ | 10-core+ | Yes |
| **OS** | macOS 13+ | macOS 14+ | Yes |

### 11.2 Model Dependencies

#### Primary Model: Qwen2.5-3B-Instruct

**Source:** HuggingFace (Qwen/Qwen2.5-3B-Instruct-GGUF)
**Format:** GGUF (Q4_K_M quantization)
**Size:** 1.7GB
**License:** Apache-2.0 (commercial use allowed)

**Assumptions:**
1. Model will be available on HuggingFace indefinitely
2. Q4_K_M quantization provides 65%+ quality vs FP16
3. Model fits code generation tasks (trained on code)
4. No legal issues with Apache-2.0 license
5. GGUF format remains standard

**Contingency:**
- If Qwen unavailable: Fallback to Llama-3.2-3B (similar size)
- If quality insufficient: Upgrade to Qwen2.5-7B (3.5GB)
- If license issues: Switch to Mistral-7B (Apache-2.0)

#### Alternative Models (Fallback)

| Model | Size | License | Quality (estimated) | Notes |
|-------|------|---------|---------------------|-------|
| Llama-3.2-3B | 1.7GB | Llama 3 | 60-65% | Meta official |
| Mistral-7B | 4.1GB | Apache-2.0 | 70-75% | Requires more RAM |
| Phi-3-Mini-4K | 2.4GB | MIT | 60-65% | Microsoft official |
| DeepSeek-Coder-1.3B | 800MB | MIT | 55-60% | Code-specialized |

### 11.3 Key Assumptions

#### Product Assumptions

1. **Users have Mac M-series devices**
   - Rationale: Target OSS developers, many use Macs
   - Risk: Excludes Linux/Windows users
   - Validation: Survey developer hardware preferences

2. **Developers value privacy over pure performance**
   - Rationale: Willingness to use local (slower) for sensitive code
   - Risk: Some prioritize speed over privacy
   - Validation: User interviews pre-launch

3. **65% quality is "good enough" for common tasks**
   - Rationale: Massive cost savings justify quality trade-off
   - Risk: Users might expect 90%+ quality
   - Validation: Beta testing with quality benchmarks

4. **Web search improves quality by 10-15%**
   - Rationale: Current information helps with API questions
   - Risk: Search results might be noisy or irrelevant
   - Validation: A/B test web search on/off

5. **Multi-agent debate improves quality by 15-25%**
   - Rationale: Research shows MoA improves accuracy
   - Risk: Gains might be smaller with small models
   - Validation: Benchmark single vs debate on HumanEval

#### Technical Assumptions

6. **llama.cpp is stable on Mac M4**
   - Rationale: Mature project with extensive testing
   - Risk: M4 is new, could have driver issues
   - Validation: Stress testing on M4 hardware

7. **3 parallel slots don't cause GPU contention**
   - Rationale: llama.cpp continuous batching handles this
   - Risk: Memory bandwidth bottleneck
   - Validation: Benchmark parallel vs sequential

8. **Memory bandwidth (273 GB/s) is sufficient**
   - Rationale: Calculations show 35-45 tok/s achievable
   - Risk: Thermal throttling reduces bandwidth
   - Validation: Sustained load testing

9. **Q4_K_M quantization quality acceptable**
   - Rationale: Research shows <5% quality loss
   - Risk: Code generation might be more sensitive
   - Validation: Benchmark Q4_K_M vs Q8_0 vs FP16

10. **Rust + Tokio has low enough overhead**
    - Rationale: Rust is zero-cost abstractions
    - Risk: Overhead from async/await
    - Validation: Profile CPU usage during inference

#### External Assumptions

11. **Anthropic API remains available and affordable**
    - Rationale: Current pricing and availability
    - Risk: Pricing increase or API changes
    - Validation: Multi-provider fallback (OpenAI, Gemini)

12. **HuggingFace model hosting continues**
    - Rationale: De facto model distribution platform
    - Risk: Model could be removed or licenses change
    - Validation: Mirror models to local CDN

13. **Apple continues Metal GPU support**
    - Rationale: Apple's commitment to ML on Silicon
    - Risk: Future macOS changes break Metal
    - Validation: Stay updated with macOS betas

### 11.4 Dependency Management Strategy

#### Versioning

- **Pin exact versions** for llama.cpp (binary)
- **Semantic versioning** for Rust dependencies
- **Automated updates** via Dependabot (security only)
- **Manual updates** for major versions (with testing)

#### Monitoring

- Track dependency CVEs (GitHub Security Advisories)
- Monthly review of dependency updates
- Automated build tests on dependency updates

#### Contingency Plans

| Dependency | Primary | Fallback 1 | Fallback 2 |
|------------|---------|------------|------------|
| Inference | llama.cpp | candle-rs | mistral.rs |
| Model | Qwen2.5-3B | Llama-3.2-3B | Mistral-7B |
| Cloud AI | Claude | GPT-4o | Gemini Pro |
| Web Search | Tavily | Brave Search | DuckDuckGo |

### 11.5 Validation Plan

**Pre-Launch Validation (Week 1-2):**
- [ ] Hardware compatibility test (M1, M2, M3, M4)
- [ ] Model quality benchmark (HumanEval subset)
- [ ] Thermal throttling test (4-hour load)
- [ ] Memory stability test (24-hour run)
- [ ] API fallback test (Claude → OpenAI)

**Post-Launch Validation (Month 1-3):**
- [ ] User survey on quality perception
- [ ] Cost analysis (cloud API spend)
- [ ] Performance profiling (identify bottlenecks)
- [ ] Failure mode analysis (crash reports)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **MoA-Lite** | Mixture-of-Agents Lite: Simplified multi-agent debate with proposers + aggregator |
| **Proposer** | Agent that generates initial response to query |
| **Aggregator** | Agent that synthesizes multiple proposer responses into final answer |
| **GGUF** | File format for quantized LLM models (llama.cpp standard) |
| **Q4_K_M** | 4-bit K-quant quantization, medium quality (0.5 bytes per parameter) |
| **KV Cache** | Key-Value cache for transformer attention (speeds up autoregressive generation) |
| **Continuous Batching** | Technique to interleave token generation across multiple requests |
| **Unified Memory** | Apple Silicon architecture where CPU/GPU share same RAM |
| **SSE** | Server-Sent Events: Protocol for streaming responses from server to client |
| **Adaptive Routing** | Decision logic to route queries to local debate vs cloud API |
| **Handoff Protocol** | Structured format for passing context to cloud API |
| **Thermal Throttling** | CPU/GPU slowing down to prevent overheating |
| **p95 Latency** | 95th percentile latency (5% of requests are slower) |

---

## Appendix B: References

### Research Papers
1. "Mixture-of-Agents Enhances Large Language Model Capabilities" (Together AI, 2024)
2. "LLM Inference on Apple Silicon: Performance Analysis" (arXiv:2511.05502)
3. "Continuous Batching for LLM Serving" (vLLM whitepaper)
4. "Flash Attention 2: Faster Attention with Better Parallelism" (Dao et al., 2023)

### Technical Documentation
- llama.cpp GitHub: https://github.com/ggml-org/llama.cpp
- Qwen2.5 Model Card: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Apple Metal Performance Shaders: https://developer.apple.com/metal/
- Anthropic Claude API: https://docs.anthropic.com/

### Prior Art
- Existing PRD01 (pensieve-local-llm-server)
- Research documents:
  - `research-max-efficiency-parallel-llm-architecture-202412011400.md`
  - `research-apple-silicon-llm-parallelism-deep-dive-202412011200.md`
  - `research-prd01-analysis-2025-11-30.md`

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-12-03 | AI (Claude) | Initial draft based on research synthesis |
| 1.0 | 2025-12-03 | AI (Claude) | Complete PRD for review |

---

**Approval Signatures:**

- [ ] Product Owner: ______________________ Date: ______
- [ ] Tech Lead: ______________________ Date: ______
- [ ] Engineering Manager: ______________________ Date: ______

**Next Steps:**
1. Review and approve PRD
2. Break down into JIRA stories
3. Architecture design document (HLD/LLD)
4. Sprint planning for MVP (12-week timeline)
5. Set up development environment
6. Begin implementation

---

*End of Product Requirements Document*
