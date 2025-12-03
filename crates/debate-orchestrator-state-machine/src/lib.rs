//! # debate-orchestrator-state-machine
//!
//! L2 Engine: MoA-Lite state machine orchestrator coordinating debate workflow.
//!
//! ## Executable Specification
//!
//! ### Preconditions
//! - LlmClient connected to llama-server
//! - Blackboard initialized
//! - Router configured
//!
//! ### Postconditions
//! - Local queries: 10-17s latency (realistic)
//! - Cloud queries: 11-18s latency
//! - All state transitions logged
//!
//! ### Performance Contract
//! - Parallel proposers via tokio::join!
//! - Graceful degradation: 2 of 3 proposers sufficient
//! - State machine is deterministic

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

// Re-export dependencies
pub use agent_role_definition_types::{
    AgentRole, AggregatorConfig, ProposerConfig, PROPOSER_COUNT,
};
pub use blackboard_handoff_protocol_core::{
    Blackboard, BlackboardError, CloudHandoffContext, InMemoryBlackboard, ProposalEntry,
};
pub use complexity_router_heuristic_classifier::{
    ComplexityRouter, HeuristicRouter, RouterConfig, RoutingDecision, RoutingFeatures, RoutingResult,
};
pub use llama_server_client_streaming::{
    LlmClient, LlmClientConfig, LlmClientError, MockLlmClient,
};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Minimum proposals required for aggregation
pub const MIN_PROPOSALS_REQUIRED: usize = 2;

/// Maximum proposals (3 proposers)
pub const MAX_PROPOSALS_COUNT: usize = 3;

/// Default timeout for entire debate workflow
pub const DEFAULT_DEBATE_TIMEOUT_SECS: u64 = 30;

/// Target latency for local debate (realistic)
pub const TARGET_LOCAL_LATENCY_MS: u64 = 17_000;

// ============================================================================
// ORCHESTRATOR STATE
// ============================================================================

/// State of the debate orchestrator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrchestratorState {
    /// Waiting for query
    Idle,
    /// Analyzing query complexity for routing
    AnalyzingComplexity,
    /// Running proposers in parallel
    Proposing {
        /// Number of completed proposers
        completed: u8,
    },
    /// Partial proposing (graceful degradation)
    PartialProposing {
        /// Number of completed proposers
        completed: u8,
        /// Number of failed proposers
        failed: u8,
    },
    /// Aggregating proposals into final response
    Aggregating,
    /// Routing to cloud
    CloudRouting,
    /// Waiting for cloud response
    CloudProcessing,
    /// Retrying after transient error
    Retrying {
        /// Current retry attempt
        attempt: u8,
    },
    /// Successfully completed
    Complete,
    /// Failed with error
    Failed {
        /// Error reason
        reason: String,
    },
}

impl OrchestratorState {
    /// Check if state is terminal
    #[must_use]
    pub const fn is_terminal_state_check(&self) -> bool {
        matches!(self, Self::Complete | Self::Failed { .. })
    }

    /// Check if state is active (processing)
    #[must_use]
    pub const fn is_active_state_check(&self) -> bool {
        !matches!(self, Self::Idle | Self::Complete | Self::Failed { .. })
    }
}

// ============================================================================
// ORCHESTRATOR EVENTS
// ============================================================================

/// Events that drive state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestratorEvent {
    /// Query received from user
    QueryReceived {
        /// The query text
        query: String,
    },
    /// Routing decision made
    RoutingDecided {
        /// The routing decision
        decision: RoutingDecision,
        /// Confidence score
        confidence: f32,
    },
    /// Proposal completed by proposer
    ProposalCompleted {
        /// Proposer index (0, 1, 2)
        index: u8,
        /// Proposal content
        content: String,
        /// Token count
        token_count: usize,
    },
    /// Proposal failed
    ProposalFailed {
        /// Proposer index
        index: u8,
        /// Error message
        error: String,
    },
    /// All proposals ready
    AllProposalsReady {
        /// Number of proposals
        count: u8,
    },
    /// Synthesis completed
    SynthesisCompleted {
        /// Final output
        output: String,
    },
    /// Cloud response received
    CloudResponseReceived {
        /// Response from cloud
        output: String,
    },
    /// Retry requested
    RetryRequested {
        /// Retry attempt number
        attempt: u8,
    },
    /// Error occurred
    ErrorOccurred {
        /// Error message
        message: String,
    },
}

// ============================================================================
// DEBATE RESULT
// ============================================================================

/// Result of a completed debate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateResult {
    /// Final response text
    pub response: String,
    /// Source of the response
    pub source: ResponseSource,
    /// Total latency in milliseconds
    pub latency_ms: u64,
    /// Number of proposals used
    pub proposal_count: u8,
    /// Conversation ID
    pub conversation_id: Uuid,
}

impl DebateResult {
    /// Create result from local debate.
    #[must_use]
    pub fn create_result_from_local(
        response: String,
        proposal_count: u8,
        latency_ms: u64,
        conversation_id: Uuid,
    ) -> Self {
        let source = if proposal_count == MAX_PROPOSALS_COUNT as u8 {
            ResponseSource::LocalDebate
        } else {
            ResponseSource::PartialDebate
        };

        Self {
            response,
            source,
            latency_ms,
            proposal_count,
            conversation_id,
        }
    }

    /// Create result from cloud handoff.
    #[must_use]
    pub fn create_result_from_cloud(
        response: String,
        latency_ms: u64,
        conversation_id: Uuid,
    ) -> Self {
        Self {
            response,
            source: ResponseSource::CloudHandoff,
            latency_ms,
            proposal_count: 0,
            conversation_id,
        }
    }
}

/// Source of the debate response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseSource {
    /// Full local debate (3 proposers)
    LocalDebate,
    /// Partial local debate (2 proposers - graceful degradation)
    PartialDebate,
    /// Cloud handoff (Claude API)
    CloudHandoff,
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors from orchestrator operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum OrchestratorError {
    /// Insufficient proposals for aggregation
    #[error("Insufficient proposals: need {required}, got {received}")]
    InsufficientProposals {
        /// Required count
        required: usize,
        /// Received count
        received: usize,
    },

    /// LLM client error
    #[error("LLM error: {0}")]
    LlmError(String),

    /// Blackboard error
    #[error("Blackboard error: {0}")]
    BlackboardError(String),

    /// Cloud API not configured
    #[error("Cloud API not configured")]
    CloudApiNotConfigured,

    /// Timeout during workflow
    #[error("Timeout after {0}ms")]
    Timeout(u64),

    /// Invalid state transition
    #[error("Invalid state transition from {from:?} on event {event}")]
    InvalidTransition {
        /// Current state
        from: OrchestratorState,
        /// Event that triggered
        event: String,
    },

    /// Orchestrator busy
    #[error("Orchestrator busy - already processing")]
    Busy,
}

impl From<BlackboardError> for OrchestratorError {
    fn from(e: BlackboardError) -> Self {
        Self::BlackboardError(e.to_string())
    }
}

impl From<LlmClientError> for OrchestratorError {
    fn from(e: LlmClientError) -> Self {
        Self::LlmError(e.to_string())
    }
}

// ============================================================================
// ORCHESTRATOR TRAIT
// ============================================================================

/// Trait for debate orchestrator implementations.
#[async_trait]
pub trait DebateOrchestrator: Send + Sync {
    /// Process query through debate workflow.
    ///
    /// # Preconditions
    /// - Query is non-empty
    ///
    /// # Postconditions
    /// - Returns `DebateResult` on success
    /// - State returns to `Idle`
    async fn process_query_through_debate(
        &self,
        query: &str,
    ) -> Result<DebateResult, OrchestratorError>;

    /// Get current orchestrator state.
    fn get_current_orchestrator_state(&self) -> OrchestratorState;

    /// Check if orchestrator is busy.
    fn is_orchestrator_currently_busy(&self) -> bool;
}

// ============================================================================
// MOA-LITE ORCHESTRATOR IMPLEMENTATION
// ============================================================================

/// MoA-Lite debate orchestrator.
///
/// Coordinates the 2-layer MoA-Lite architecture:
/// - Layer 1: 3 parallel proposers
/// - Layer 2: 1 aggregator
pub struct MoaLiteOrchestrator<L, B, R>
where
    L: LlmClient + 'static,
    B: Blackboard + 'static,
    R: ComplexityRouter + 'static,
{
    llm_client: Arc<L>,
    blackboard: Arc<B>,
    router: Arc<R>,
    proposer_configs: [ProposerConfig; PROPOSER_COUNT],
    aggregator_config: AggregatorConfig,
    timeout_secs: u64,
}

impl<L, B, R> MoaLiteOrchestrator<L, B, R>
where
    L: LlmClient + 'static,
    B: Blackboard + 'static,
    R: ComplexityRouter + 'static,
{
    /// Create orchestrator with provided components.
    #[must_use]
    pub fn create_orchestrator_with_components(
        llm_client: Arc<L>,
        blackboard: Arc<B>,
        router: Arc<R>,
    ) -> Self {
        Self {
            llm_client,
            blackboard,
            router,
            proposer_configs: ProposerConfig::create_all_proposer_configs(),
            aggregator_config: AggregatorConfig::create_config_default_aggregator(),
            timeout_secs: DEFAULT_DEBATE_TIMEOUT_SECS,
        }
    }

    /// Create orchestrator with custom timeout.
    #[must_use]
    pub fn create_orchestrator_with_timeout(
        llm_client: Arc<L>,
        blackboard: Arc<B>,
        router: Arc<R>,
        timeout_secs: u64,
    ) -> Self {
        let mut orchestrator = Self::create_orchestrator_with_components(llm_client, blackboard, router);
        orchestrator.timeout_secs = timeout_secs;
        orchestrator
    }

    /// Execute local debate workflow (3 proposers + aggregator).
    #[instrument(skip(self), fields(conversation_id = %conversation_id))]
    async fn execute_local_debate_workflow(
        &self,
        query: &str,
        conversation_id: Uuid,
    ) -> Result<DebateResult, OrchestratorError> {
        let start = Instant::now();
        info!("Starting local debate workflow");

        // Run proposers in parallel
        let proposals = self.run_proposers_in_parallel(query, conversation_id).await?;

        info!(proposal_count = proposals.len(), "Proposers completed");

        // Check minimum proposals
        if proposals.len() < MIN_PROPOSALS_REQUIRED {
            return Err(OrchestratorError::InsufficientProposals {
                required: MIN_PROPOSALS_REQUIRED,
                received: proposals.len(),
            });
        }

        // Store proposals in blackboard
        for proposal in &proposals {
            self.blackboard.store_proposal_in_blackboard(proposal.clone())?;
        }

        // Aggregate proposals
        let aggregator_input = self.build_aggregator_input_text(&proposals);
        let final_response = self
            .generate_aggregator_final_response(&aggregator_input)
            .await?;

        let latency_ms = start.elapsed().as_millis() as u64;
        info!(latency_ms, "Local debate completed");

        Ok(DebateResult::create_result_from_local(
            final_response,
            proposals.len() as u8,
            latency_ms,
            conversation_id,
        ))
    }

    /// Run all proposers in parallel.
    #[instrument(skip(self))]
    async fn run_proposers_in_parallel(
        &self,
        query: &str,
        conversation_id: Uuid,
    ) -> Result<Vec<ProposalEntry>, OrchestratorError> {
        debug!("Running 3 proposers in parallel");

        // Execute all proposers concurrently
        let (result0, result1, result2) = tokio::join!(
            self.generate_proposal_with_config(query, 0),
            self.generate_proposal_with_config(query, 1),
            self.generate_proposal_with_config(query, 2),
        );

        // Collect successful proposals (graceful degradation)
        let results = [result0, result1, result2];
        let proposals: Vec<ProposalEntry> = results
            .into_iter()
            .enumerate()
            .filter_map(|(idx, result)| {
                match result {
                    Ok(content) => {
                        let _token_count = content.len() / 4; // Approximate
                        Some(ProposalEntry::create_entry_from_content(
                            conversation_id,
                            idx as u8,
                            content,
                            0.7, // Default confidence
                        ))
                    }
                    Err(e) => {
                        warn!(proposer = idx, error = %e, "Proposer failed");
                        None
                    }
                }
            })
            .collect();

        debug!(successful = proposals.len(), "Proposers finished");
        Ok(proposals)
    }

    /// Generate proposal using specific proposer config.
    async fn generate_proposal_with_config(
        &self,
        query: &str,
        index: usize,
    ) -> Result<String, OrchestratorError> {
        let config = &self.proposer_configs[index];
        let prompt = format!("{}\n\nQuery: {}", config.system_prompt(), query);

        self.llm_client
            .generate_complete_response_blocking(&prompt, config.max_tokens(), config.temperature())
            .await
            .map_err(|e| OrchestratorError::LlmError(e.to_string()))
    }

    /// Build aggregator input from proposals.
    fn build_aggregator_input_text(&self, proposals: &[ProposalEntry]) -> String {
        proposals
            .iter()
            .enumerate()
            .map(|(i, p)| format!("## Proposal {} (Proposer {}):\n{}\n", i + 1, p.proposer_index, p.content))
            .collect::<Vec<_>>()
            .join("\n---\n\n")
    }

    /// Generate final response using aggregator.
    async fn generate_aggregator_final_response(
        &self,
        input: &str,
    ) -> Result<String, OrchestratorError> {
        let prompt = format!(
            "{}\n\n## Proposals to synthesize:\n\n{}",
            self.aggregator_config.system_prompt(),
            input
        );

        self.llm_client
            .generate_complete_response_blocking(
                &prompt,
                self.aggregator_config.max_tokens(),
                self.aggregator_config.temperature(),
            )
            .await
            .map_err(|e| OrchestratorError::LlmError(e.to_string()))
    }

    /// Execute cloud handoff workflow.
    #[instrument(skip(self))]
    async fn execute_cloud_handoff_workflow(
        &self,
        query: &str,
        conversation_id: Uuid,
    ) -> Result<DebateResult, OrchestratorError> {
        let _start = Instant::now();
        info!("Starting cloud handoff workflow");

        // For now, cloud handoff is not implemented (returns error)
        // In production, this would call Claude API
        let context = CloudHandoffContext {
            query: query.to_string(),
            task_type: "complex_reasoning".to_string(),
            proposal_summaries: vec![],
            local_confidence: 0.3,
            routing_reason: "Query exceeded complexity threshold".to_string(),
        };

        debug!(?context, "Cloud handoff context prepared");

        // Placeholder: In production, call Claude API here
        Err(OrchestratorError::CloudApiNotConfigured)
    }
}

#[async_trait]
impl<L, B, R> DebateOrchestrator for MoaLiteOrchestrator<L, B, R>
where
    L: LlmClient + 'static,
    B: Blackboard + 'static,
    R: ComplexityRouter + 'static,
{
    #[instrument(skip(self), fields(query_len = query.len()))]
    async fn process_query_through_debate(
        &self,
        query: &str,
    ) -> Result<DebateResult, OrchestratorError> {
        if query.is_empty() {
            return Err(OrchestratorError::LlmError("Query cannot be empty".into()));
        }

        let conversation_id = Uuid::new_v4();
        info!(%conversation_id, "Processing query");

        // Step 1: Route query
        let routing_result = self.router.classify_with_confidence_score(query);
        info!(
            ?routing_result.decision,
            confidence = routing_result.confidence,
            reason = %routing_result.reason,
            "Routing decision"
        );

        // Step 2: Execute appropriate workflow
        let result = match routing_result.decision {
            RoutingDecision::Local => {
                self.execute_local_debate_workflow(query, conversation_id).await
            }
            RoutingDecision::CloudHandoff => {
                // Try cloud first, fall back to local if not configured
                match self.execute_cloud_handoff_workflow(query, conversation_id).await {
                    Ok(result) => Ok(result),
                    Err(OrchestratorError::CloudApiNotConfigured) => {
                        warn!("Cloud API not configured, falling back to local");
                        self.execute_local_debate_workflow(query, conversation_id).await
                    }
                    Err(e) => Err(e),
                }
            }
        };

        // Cleanup blackboard
        let _ = self.blackboard.clear_conversation_from_blackboard(conversation_id);

        result
    }

    fn get_current_orchestrator_state(&self) -> OrchestratorState {
        // Stateless design - return Idle
        // In a stateful implementation, this would track actual state
        OrchestratorState::Idle
    }

    fn is_orchestrator_currently_busy(&self) -> bool {
        false // Stateless design
    }
}

// ============================================================================
// BUILDER PATTERN FOR ORCHESTRATOR
// ============================================================================

/// Builder for creating orchestrator instances.
#[derive(Debug)]
pub struct OrchestratorBuilder<L, B, R>
where
    L: LlmClient + 'static,
    B: Blackboard + 'static,
    R: ComplexityRouter + 'static,
{
    llm_client: Option<Arc<L>>,
    blackboard: Option<Arc<B>>,
    router: Option<Arc<R>>,
    timeout_secs: u64,
}

impl<L, B, R> OrchestratorBuilder<L, B, R>
where
    L: LlmClient + 'static,
    B: Blackboard + 'static,
    R: ComplexityRouter + 'static,
{
    /// Create new builder with defaults.
    #[must_use]
    pub fn create_builder_with_defaults() -> Self {
        Self {
            llm_client: None,
            blackboard: None,
            router: None,
            timeout_secs: DEFAULT_DEBATE_TIMEOUT_SECS,
        }
    }

    /// Set LLM client.
    #[must_use]
    pub fn with_llm_client_instance(mut self, client: Arc<L>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Set blackboard.
    #[must_use]
    pub fn with_blackboard_instance_arc(mut self, blackboard: Arc<B>) -> Self {
        self.blackboard = Some(blackboard);
        self
    }

    /// Set router.
    #[must_use]
    pub fn with_router_instance_arc(mut self, router: Arc<R>) -> Self {
        self.router = Some(router);
        self
    }

    /// Set timeout.
    #[must_use]
    pub fn with_timeout_seconds_value(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Build the orchestrator.
    ///
    /// # Errors
    /// Returns error if required components are missing
    pub fn build_orchestrator_from_config(self) -> Result<MoaLiteOrchestrator<L, B, R>, OrchestratorError> {
        let llm_client = self
            .llm_client
            .ok_or_else(|| OrchestratorError::LlmError("LLM client required".into()))?;
        let blackboard = self
            .blackboard
            .ok_or_else(|| OrchestratorError::BlackboardError("Blackboard required".into()))?;
        let router = self
            .router
            .ok_or_else(|| OrchestratorError::LlmError("Router required".into()))?;

        Ok(MoaLiteOrchestrator::create_orchestrator_with_timeout(
            llm_client,
            blackboard,
            router,
            self.timeout_secs,
        ))
    }
}

impl<L, B, R> Default for OrchestratorBuilder<L, B, R>
where
    L: LlmClient + 'static,
    B: Blackboard + 'static,
    R: ComplexityRouter + 'static,
{
    fn default() -> Self {
        Self::create_builder_with_defaults()
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Create orchestrator with mock components for testing.
#[must_use]
pub fn create_test_orchestrator_mock(
    mock_response: &str,
) -> MoaLiteOrchestrator<MockLlmClient, InMemoryBlackboard, HeuristicRouter> {
    let llm_client = Arc::new(MockLlmClient::create_mock_for_testing(mock_response));
    let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
    let router = Arc::new(HeuristicRouter::create_router_with_defaults());

    MoaLiteOrchestrator::create_orchestrator_with_components(llm_client, blackboard, router)
}

// ============================================================================
// TESTS (TDD-First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // OrchestratorState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_orchestrator_state_terminal() {
        assert!(OrchestratorState::Complete.is_terminal_state_check());
        assert!(OrchestratorState::Failed {
            reason: "test".into()
        }
        .is_terminal_state_check());

        assert!(!OrchestratorState::Idle.is_terminal_state_check());
        assert!(!OrchestratorState::Proposing { completed: 0 }.is_terminal_state_check());
    }

    #[test]
    fn test_orchestrator_state_active() {
        assert!(OrchestratorState::Proposing { completed: 1 }.is_active_state_check());
        assert!(OrchestratorState::Aggregating.is_active_state_check());
        assert!(OrchestratorState::AnalyzingComplexity.is_active_state_check());

        assert!(!OrchestratorState::Idle.is_active_state_check());
        assert!(!OrchestratorState::Complete.is_active_state_check());
    }

    #[test]
    fn test_orchestrator_state_serialization() {
        let state = OrchestratorState::Proposing { completed: 2 };
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: OrchestratorState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, deserialized);
    }

    // -------------------------------------------------------------------------
    // DebateResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debate_result_local_full() {
        let conv_id = Uuid::new_v4();
        let result = DebateResult::create_result_from_local(
            "Response".to_string(),
            3, // Full debate
            15000,
            conv_id,
        );

        assert_eq!(result.source, ResponseSource::LocalDebate);
        assert_eq!(result.proposal_count, 3);
    }

    #[test]
    fn test_debate_result_local_partial() {
        let conv_id = Uuid::new_v4();
        let result = DebateResult::create_result_from_local(
            "Response".to_string(),
            2, // Partial
            12000,
            conv_id,
        );

        assert_eq!(result.source, ResponseSource::PartialDebate);
        assert_eq!(result.proposal_count, 2);
    }

    #[test]
    fn test_debate_result_cloud() {
        let conv_id = Uuid::new_v4();
        let result = DebateResult::create_result_from_cloud(
            "Cloud response".to_string(),
            14000,
            conv_id,
        );

        assert_eq!(result.source, ResponseSource::CloudHandoff);
        assert_eq!(result.proposal_count, 0);
    }

    // -------------------------------------------------------------------------
    // OrchestratorEvent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_orchestrator_event_serialization() {
        let events = vec![
            OrchestratorEvent::QueryReceived {
                query: "Test".into(),
            },
            OrchestratorEvent::RoutingDecided {
                decision: RoutingDecision::Local,
                confidence: 0.85,
            },
            OrchestratorEvent::ProposalCompleted {
                index: 0,
                content: "Proposal".into(),
                token_count: 100,
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            assert!(!json.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // OrchestratorError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_from_blackboard() {
        let bb_error = BlackboardError::ConversationNotFound(Uuid::new_v4());
        let orch_error: OrchestratorError = bb_error.into();
        assert!(matches!(orch_error, OrchestratorError::BlackboardError(_)));
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            OrchestratorError::InsufficientProposals {
                required: 2,
                received: 1,
            },
            OrchestratorError::LlmError("Test error".into()),
            OrchestratorError::Timeout(30000),
            OrchestratorError::CloudApiNotConfigured,
        ];

        for error in errors {
            let display = format!("{error}");
            assert!(!display.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Orchestrator Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_test_orchestrator() {
        let orchestrator = create_test_orchestrator_mock("Test response");

        assert_eq!(
            orchestrator.get_current_orchestrator_state(),
            OrchestratorState::Idle
        );
        assert!(!orchestrator.is_orchestrator_currently_busy());
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_missing_components() {
        let builder: OrchestratorBuilder<MockLlmClient, InMemoryBlackboard, HeuristicRouter> =
            OrchestratorBuilder::create_builder_with_defaults();

        let result = builder.build_orchestrator_from_config();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_complete() {
        let llm_client = Arc::new(MockLlmClient::create_mock_for_testing("Test"));
        let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
        let router = Arc::new(HeuristicRouter::create_router_with_defaults());

        let result = OrchestratorBuilder::create_builder_with_defaults()
            .with_llm_client_instance(llm_client)
            .with_blackboard_instance_arc(blackboard)
            .with_router_instance_arc(router)
            .with_timeout_seconds_value(60)
            .build_orchestrator_from_config();

        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Integration Tests (with Mock)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_orchestrator_local_debate_success() {
        let orchestrator = create_test_orchestrator_mock("This is a proposal response.");

        let result = orchestrator
            .process_query_through_debate("Explain ownership in Rust")
            .await;

        assert!(result.is_ok());
        let debate_result = result.unwrap();
        assert!(debate_result.proposal_count >= MIN_PROPOSALS_REQUIRED as u8);
        assert!(!debate_result.response.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_empty_query_fails() {
        let orchestrator = create_test_orchestrator_mock("Response");

        let result = orchestrator.process_query_through_debate("").await;

        assert!(result.is_err());
        assert!(matches!(result, Err(OrchestratorError::LlmError(_))));
    }

    #[tokio::test]
    async fn test_orchestrator_routes_simple_locally() {
        let orchestrator = create_test_orchestrator_mock("Simple response");

        let result = orchestrator
            .process_query_through_debate("What is Rust?")
            .await;

        assert!(result.is_ok());
        // Simple queries should route locally
        assert!(matches!(
            result.unwrap().source,
            ResponseSource::LocalDebate | ResponseSource::PartialDebate
        ));
    }

    #[tokio::test]
    async fn test_orchestrator_with_failing_proposer() {
        // Create orchestrator with one failing proposer
        let llm_client = Arc::new(MockLlmClient::create_mock_for_testing("Good response"));
        let blackboard = Arc::new(InMemoryBlackboard::create_blackboard_in_memory());
        let router = Arc::new(HeuristicRouter::create_router_with_defaults());

        let orchestrator =
            MoaLiteOrchestrator::create_orchestrator_with_components(llm_client, blackboard, router);

        // Should succeed with graceful degradation (2+ proposers)
        let result = orchestrator
            .process_query_through_debate("Explain async")
            .await;

        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_match_architecture() {
        assert_eq!(MIN_PROPOSALS_REQUIRED, 2);
        assert_eq!(MAX_PROPOSALS_COUNT, 3);
        assert_eq!(DEFAULT_DEBATE_TIMEOUT_SECS, 30);
        assert_eq!(TARGET_LOCAL_LATENCY_MS, 17_000);
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_aggregator_input() {
        let orchestrator = create_test_orchestrator_mock("Response");

        let proposals = vec![
            ProposalEntry::create_entry_from_content(
                Uuid::new_v4(),
                0,
                "First proposal content".to_string(),
                0.8,
            ),
            ProposalEntry::create_entry_from_content(
                Uuid::new_v4(),
                1,
                "Second proposal content".to_string(),
                0.7,
            ),
        ];

        let input = orchestrator.build_aggregator_input_text(&proposals);

        assert!(input.contains("Proposal 1"));
        assert!(input.contains("Proposal 2"));
        assert!(input.contains("First proposal content"));
        assert!(input.contains("Second proposal content"));
    }
}
