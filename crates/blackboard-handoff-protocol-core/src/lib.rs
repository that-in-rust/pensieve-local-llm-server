//! # blackboard-handoff-protocol-core
//!
//! L1 Core: Proposal storage and cloud handoff protocol for MoA-Lite debate system.
//!
//! ## Executable Specification
//!
//! ### Preconditions
//! - conversation_id is valid UUID
//! - proposals count in 1..=3
//!
//! ### Postconditions
//! - ProposalEntry stores full proposal verbatim
//! - CloudHandoffContext compresses to 150-300 tokens
//!
//! ### Error Conditions
//! - BlackboardError::TokenBudgetExceeded if cloud context > 300 tokens
//! - BlackboardError::ConversationNotFound if lookup fails

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// Re-export agent types
pub use agent_role_definition_types::{
    AgentRole, AggregatorConfig, ProposerConfig, PROPOSER_COUNT,
};

// ============================================================================
// CONSTANTS (Token Budgets)
// ============================================================================

/// Minimum tokens for cloud handoff context
pub const CLOUD_HANDOFF_MIN_TOKENS: usize = 150;

/// Maximum tokens for cloud handoff context
pub const CLOUD_HANDOFF_MAX_TOKENS: usize = 300;

/// Approximate characters per token (for estimation)
pub const CHARS_PER_TOKEN: usize = 4;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur in blackboard operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BlackboardError {
    /// Token budget exceeded for cloud handoff
    #[error("Token budget exceeded: {actual} > {max}")]
    TokenBudgetExceeded {
        /// Actual token count
        actual: usize,
        /// Maximum allowed
        max: usize,
    },

    /// Conversation not found in blackboard
    #[error("Conversation not found: {0}")]
    ConversationNotFound(Uuid),

    /// Storage operation failed
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Invalid proposal count
    #[error("Invalid proposal count: expected 1-3, got {0}")]
    InvalidProposalCount(usize),
}

// ============================================================================
// PROPOSAL ENTRY (Full verbatim storage)
// ============================================================================

/// Full proposal entry for internal debate.
///
/// Stores the complete proposer output verbatim (150-300 tokens).
/// This is passed to the aggregator without compression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalEntry {
    /// Unique conversation identifier
    pub conversation_id: Uuid,
    /// Proposer instance index (0, 1, or 2)
    pub proposer_index: u8,
    /// Full proposal content (verbatim)
    pub content: String,
    /// Estimated token count
    pub token_count: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence_score: f32,
    /// Timestamp when created
    pub created_at: DateTime<Utc>,
}

impl ProposalEntry {
    /// Create proposal entry from content.
    ///
    /// # Preconditions
    /// - `proposer_index` in 0..3
    /// - `content` is non-empty
    ///
    /// # Postconditions
    /// - Returns valid `ProposalEntry` with estimated token count
    #[must_use]
    pub fn create_entry_from_content(
        conversation_id: Uuid,
        proposer_index: u8,
        content: String,
        confidence_score: f32,
    ) -> Self {
        let token_count = estimate_token_count_chars(&content);
        Self {
            conversation_id,
            proposer_index: proposer_index.min(2),
            content,
            token_count,
            confidence_score: confidence_score.clamp(0.0, 1.0),
            created_at: Utc::now(),
        }
    }

    /// Create proposal with explicit token count.
    #[must_use]
    pub fn create_entry_with_tokens(
        conversation_id: Uuid,
        proposer_index: u8,
        content: String,
        token_count: usize,
        confidence_score: f32,
    ) -> Self {
        Self {
            conversation_id,
            proposer_index: proposer_index.min(2),
            content,
            token_count,
            confidence_score: confidence_score.clamp(0.0, 1.0),
            created_at: Utc::now(),
        }
    }
}

// ============================================================================
// CLOUD HANDOFF CONTEXT (Compressed)
// ============================================================================

/// Compressed cloud handoff context (150-300 tokens total).
///
/// Used when routing to Claude API. Contains:
/// - Original query
/// - Task type classification
/// - Summarized proposals (not verbatim)
/// - Local confidence score
/// - Routing reason
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CloudHandoffContext {
    /// Original user query
    pub query: String,
    /// Task type classification
    pub task_type: String,
    /// Compressed proposal summaries
    pub proposal_summaries: Vec<String>,
    /// Average local confidence score
    pub local_confidence: f32,
    /// Reason for cloud routing
    pub routing_reason: String,
}

impl CloudHandoffContext {
    /// Build context from proposals list.
    ///
    /// # Preconditions
    /// - `proposals.len()` in 1..=3
    ///
    /// # Postconditions
    /// - Total token count <= 300
    ///
    /// # Errors
    /// - `BlackboardError::TokenBudgetExceeded` if context exceeds budget
    /// - `BlackboardError::InvalidProposalCount` if proposals empty or > 3
    pub fn build_context_from_proposals(
        query: &str,
        task_type: &str,
        proposals: &[ProposalEntry],
        routing_reason: &str,
    ) -> Result<Self, BlackboardError> {
        // Validate proposal count
        if proposals.is_empty() || proposals.len() > PROPOSER_COUNT {
            return Err(BlackboardError::InvalidProposalCount(proposals.len()));
        }

        // Compress each proposal to ~50-80 tokens
        let summaries: Vec<String> = proposals
            .iter()
            .map(|p| summarize_proposal_for_handoff(&p.content))
            .collect();

        // Calculate average confidence
        let avg_confidence = proposals
            .iter()
            .map(|p| p.confidence_score)
            .sum::<f32>()
            / proposals.len() as f32;

        let context = Self {
            query: truncate_to_token_limit(query, 50),
            task_type: task_type.to_string(),
            proposal_summaries: summaries,
            local_confidence: avg_confidence,
            routing_reason: routing_reason.to_string(),
        };

        // Validate total token budget
        let total_tokens = context.estimate_total_token_count();
        if total_tokens > CLOUD_HANDOFF_MAX_TOKENS {
            return Err(BlackboardError::TokenBudgetExceeded {
                actual: total_tokens,
                max: CLOUD_HANDOFF_MAX_TOKENS,
            });
        }

        Ok(context)
    }

    /// Estimate total token count of context.
    #[must_use]
    pub fn estimate_total_token_count(&self) -> usize {
        let query_tokens = estimate_token_count_chars(&self.query);
        let task_tokens = estimate_token_count_chars(&self.task_type);
        let summary_tokens: usize = self
            .proposal_summaries
            .iter()
            .map(|s| estimate_token_count_chars(s))
            .sum();
        let reason_tokens = estimate_token_count_chars(&self.routing_reason);

        // Add overhead for JSON structure (~20 tokens)
        query_tokens + task_tokens + summary_tokens + reason_tokens + 20
    }

    /// Convert to JSON string for API call.
    ///
    /// # Errors
    /// Returns `BlackboardError::StorageError` if serialization fails
    pub fn to_json_string_for_api(&self) -> Result<String, BlackboardError> {
        serde_json::to_string(self)
            .map_err(|e| BlackboardError::StorageError(e.to_string()))
    }
}

// ============================================================================
// BLACKBOARD TRAIT (Functional Interface)
// ============================================================================

/// Blackboard trait for storing and retrieving proposals.
///
/// Implementations should be thread-safe (Send + Sync).
pub trait Blackboard: Send + Sync {
    /// Store proposal in blackboard.
    ///
    /// # Errors
    /// - `BlackboardError::StorageError` on failure
    fn store_proposal_in_blackboard(
        &self,
        entry: ProposalEntry,
    ) -> Result<(), BlackboardError>;

    /// Get all proposals for conversation.
    ///
    /// # Errors
    /// - `BlackboardError::ConversationNotFound` if no proposals exist
    fn get_proposals_for_conversation(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<ProposalEntry>, BlackboardError>;

    /// Clear all proposals for conversation.
    ///
    /// # Errors
    /// - `BlackboardError::ConversationNotFound` if conversation doesn't exist
    fn clear_conversation_from_blackboard(
        &self,
        conversation_id: Uuid,
    ) -> Result<(), BlackboardError>;

    /// Get proposal count for conversation.
    fn count_proposals_for_conversation(&self, conversation_id: Uuid) -> usize;
}

// ============================================================================
// IN-MEMORY BLACKBOARD IMPLEMENTATION
// ============================================================================

/// In-memory blackboard implementation.
///
/// Thread-safe using `RwLock`. Suitable for single-process deployments.
#[derive(Debug, Default)]
pub struct InMemoryBlackboard {
    proposals: RwLock<HashMap<Uuid, Vec<ProposalEntry>>>,
}

impl InMemoryBlackboard {
    /// Create new in-memory blackboard.
    #[must_use]
    pub fn create_blackboard_in_memory() -> Self {
        Self {
            proposals: RwLock::new(HashMap::new()),
        }
    }

    /// Create as Arc for sharing across tasks.
    #[must_use]
    pub fn create_shared_blackboard_arc() -> Arc<Self> {
        Arc::new(Self::create_blackboard_in_memory())
    }
}

impl Blackboard for InMemoryBlackboard {
    fn store_proposal_in_blackboard(
        &self,
        entry: ProposalEntry,
    ) -> Result<(), BlackboardError> {
        let mut proposals = self
            .proposals
            .write()
            .map_err(|e| BlackboardError::StorageError(e.to_string()))?;

        proposals
            .entry(entry.conversation_id)
            .or_default()
            .push(entry);

        Ok(())
    }

    fn get_proposals_for_conversation(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<ProposalEntry>, BlackboardError> {
        let proposals = self
            .proposals
            .read()
            .map_err(|e| BlackboardError::StorageError(e.to_string()))?;

        proposals
            .get(&conversation_id)
            .cloned()
            .ok_or(BlackboardError::ConversationNotFound(conversation_id))
    }

    fn clear_conversation_from_blackboard(
        &self,
        conversation_id: Uuid,
    ) -> Result<(), BlackboardError> {
        let mut proposals = self
            .proposals
            .write()
            .map_err(|e| BlackboardError::StorageError(e.to_string()))?;

        if proposals.remove(&conversation_id).is_none() {
            return Err(BlackboardError::ConversationNotFound(conversation_id));
        }

        Ok(())
    }

    fn count_proposals_for_conversation(&self, conversation_id: Uuid) -> usize {
        self.proposals
            .read()
            .ok()
            .and_then(|p| p.get(&conversation_id).map(Vec::len))
            .unwrap_or(0)
    }
}

// ============================================================================
// PURE HELPER FUNCTIONS (No Side Effects)
// ============================================================================

/// Estimate token count from character count.
///
/// Uses simple heuristic: ~4 characters per token.
#[must_use]
pub fn estimate_token_count_chars(text: &str) -> usize {
    (text.len() + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

/// Summarize proposal for cloud handoff.
///
/// Truncates to approximately 80 tokens (~320 chars).
#[must_use]
fn summarize_proposal_for_handoff(content: &str) -> String {
    const MAX_SUMMARY_CHARS: usize = 320; // ~80 tokens

    if content.len() <= MAX_SUMMARY_CHARS {
        return content.to_string();
    }

    // Take first sentence(s) up to limit
    let truncated: String = content
        .chars()
        .take(MAX_SUMMARY_CHARS - 3)
        .collect();

    format!("{truncated}...")
}

/// Truncate text to approximate token limit.
#[must_use]
fn truncate_to_token_limit(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens * CHARS_PER_TOKEN;

    if text.len() <= max_chars {
        return text.to_string();
    }

    text.chars().take(max_chars).collect()
}

// ============================================================================
// TESTS (TDD-First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ProposalEntry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_proposal_entry_basic() {
        let conv_id = Uuid::new_v4();
        let entry = ProposalEntry::create_entry_from_content(
            conv_id,
            0,
            "This is a test proposal.".to_string(),
            0.8,
        );

        assert_eq!(entry.conversation_id, conv_id);
        assert_eq!(entry.proposer_index, 0);
        assert_eq!(entry.content, "This is a test proposal.");
        assert!((entry.confidence_score - 0.8).abs() < f32::EPSILON);
        assert!(entry.token_count > 0);
    }

    #[test]
    fn test_create_proposal_entry_clamps_index() {
        let entry = ProposalEntry::create_entry_from_content(
            Uuid::new_v4(),
            99, // Invalid index
            "Test".to_string(),
            0.5,
        );

        assert_eq!(entry.proposer_index, 2); // Clamped to max
    }

    #[test]
    fn test_create_proposal_entry_clamps_confidence() {
        let entry1 = ProposalEntry::create_entry_from_content(
            Uuid::new_v4(),
            0,
            "Test".to_string(),
            -0.5, // Below min
        );
        assert!((entry1.confidence_score - 0.0).abs() < f32::EPSILON);

        let entry2 = ProposalEntry::create_entry_from_content(
            Uuid::new_v4(),
            0,
            "Test".to_string(),
            1.5, // Above max
        );
        assert!((entry2.confidence_score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_proposal_entry_serialization() {
        let entry = ProposalEntry::create_entry_from_content(
            Uuid::new_v4(),
            1,
            "Test proposal content".to_string(),
            0.75,
        );

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: ProposalEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(entry.conversation_id, deserialized.conversation_id);
        assert_eq!(entry.proposer_index, deserialized.proposer_index);
        assert_eq!(entry.content, deserialized.content);
    }

    // -------------------------------------------------------------------------
    // CloudHandoffContext Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_cloud_handoff_context_success() {
        let conv_id = Uuid::new_v4();
        let proposals = vec![
            ProposalEntry::create_entry_from_content(
                conv_id,
                0,
                "First proposal".to_string(),
                0.8,
            ),
            ProposalEntry::create_entry_from_content(
                conv_id,
                1,
                "Second proposal".to_string(),
                0.7,
            ),
        ];

        let result = CloudHandoffContext::build_context_from_proposals(
            "Test query",
            "code_explanation",
            &proposals,
            "Complex task",
        );

        assert!(result.is_ok());
        let context = result.unwrap();
        assert_eq!(context.query, "Test query");
        assert_eq!(context.task_type, "code_explanation");
        assert_eq!(context.proposal_summaries.len(), 2);
        assert!((context.local_confidence - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_build_cloud_handoff_context_empty_proposals() {
        let result = CloudHandoffContext::build_context_from_proposals(
            "Query",
            "task",
            &[], // Empty
            "Reason",
        );

        assert!(matches!(
            result,
            Err(BlackboardError::InvalidProposalCount(0))
        ));
    }

    #[test]
    fn test_build_cloud_handoff_context_too_many_proposals() {
        let conv_id = Uuid::new_v4();
        let proposals: Vec<ProposalEntry> = (0..4)
            .map(|i| {
                ProposalEntry::create_entry_from_content(
                    conv_id,
                    i as u8,
                    format!("Proposal {i}"),
                    0.5,
                )
            })
            .collect();

        let result = CloudHandoffContext::build_context_from_proposals(
            "Query",
            "task",
            &proposals,
            "Reason",
        );

        assert!(matches!(
            result,
            Err(BlackboardError::InvalidProposalCount(4))
        ));
    }

    #[test]
    fn test_cloud_handoff_context_token_estimation() {
        let conv_id = Uuid::new_v4();
        let proposals = vec![ProposalEntry::create_entry_from_content(
            conv_id,
            0,
            "Short proposal".to_string(),
            0.8,
        )];

        let context = CloudHandoffContext::build_context_from_proposals(
            "Short query",
            "task",
            &proposals,
            "Reason",
        )
        .unwrap();

        let tokens = context.estimate_total_token_count();
        assert!(tokens > 0);
        assert!(tokens <= CLOUD_HANDOFF_MAX_TOKENS);
    }

    #[test]
    fn test_cloud_handoff_context_serialization() {
        let context = CloudHandoffContext {
            query: "Test query".to_string(),
            task_type: "code_gen".to_string(),
            proposal_summaries: vec!["Summary 1".to_string()],
            local_confidence: 0.5,
            routing_reason: "Complex".to_string(),
        };

        let json = context.to_json_string_for_api().unwrap();
        assert!(json.contains("Test query"));
        assert!(json.contains("code_gen"));
    }

    // -------------------------------------------------------------------------
    // InMemoryBlackboard Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blackboard_store_and_retrieve() {
        let blackboard = InMemoryBlackboard::create_blackboard_in_memory();
        let conv_id = Uuid::new_v4();

        let entry = ProposalEntry::create_entry_from_content(
            conv_id,
            0,
            "Test proposal".to_string(),
            0.8,
        );

        // Store
        let store_result = blackboard.store_proposal_in_blackboard(entry.clone());
        assert!(store_result.is_ok());

        // Retrieve
        let get_result = blackboard.get_proposals_for_conversation(conv_id);
        assert!(get_result.is_ok());
        let proposals = get_result.unwrap();
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].content, "Test proposal");
    }

    #[test]
    fn test_blackboard_multiple_proposals() {
        let blackboard = InMemoryBlackboard::create_blackboard_in_memory();
        let conv_id = Uuid::new_v4();

        for i in 0..3 {
            let entry = ProposalEntry::create_entry_from_content(
                conv_id,
                i,
                format!("Proposal {i}"),
                0.7,
            );
            blackboard.store_proposal_in_blackboard(entry).unwrap();
        }

        let proposals = blackboard.get_proposals_for_conversation(conv_id).unwrap();
        assert_eq!(proposals.len(), 3);

        let count = blackboard.count_proposals_for_conversation(conv_id);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_blackboard_conversation_not_found() {
        let blackboard = InMemoryBlackboard::create_blackboard_in_memory();
        let unknown_id = Uuid::new_v4();

        let result = blackboard.get_proposals_for_conversation(unknown_id);
        assert!(matches!(
            result,
            Err(BlackboardError::ConversationNotFound(_))
        ));
    }

    #[test]
    fn test_blackboard_clear_conversation() {
        let blackboard = InMemoryBlackboard::create_blackboard_in_memory();
        let conv_id = Uuid::new_v4();

        let entry = ProposalEntry::create_entry_from_content(
            conv_id,
            0,
            "Test".to_string(),
            0.5,
        );
        blackboard.store_proposal_in_blackboard(entry).unwrap();

        // Clear
        let clear_result = blackboard.clear_conversation_from_blackboard(conv_id);
        assert!(clear_result.is_ok());

        // Should not exist anymore
        let get_result = blackboard.get_proposals_for_conversation(conv_id);
        assert!(matches!(
            get_result,
            Err(BlackboardError::ConversationNotFound(_))
        ));
    }

    #[test]
    fn test_blackboard_clear_nonexistent() {
        let blackboard = InMemoryBlackboard::create_blackboard_in_memory();
        let unknown_id = Uuid::new_v4();

        let result = blackboard.clear_conversation_from_blackboard(unknown_id);
        assert!(matches!(
            result,
            Err(BlackboardError::ConversationNotFound(_))
        ));
    }

    #[test]
    fn test_blackboard_shared_arc() {
        let blackboard = InMemoryBlackboard::create_shared_blackboard_arc();
        let conv_id = Uuid::new_v4();

        let entry = ProposalEntry::create_entry_from_content(
            conv_id,
            0,
            "Test".to_string(),
            0.5,
        );

        // Use through Arc
        blackboard.store_proposal_in_blackboard(entry).unwrap();
        let count = blackboard.count_proposals_for_conversation(conv_id);
        assert_eq!(count, 1);
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_estimate_token_count_chars() {
        assert_eq!(estimate_token_count_chars(""), 0);
        assert_eq!(estimate_token_count_chars("test"), 1); // 4 chars = 1 token
        assert_eq!(estimate_token_count_chars("12345678"), 2); // 8 chars = 2 tokens
        assert_eq!(estimate_token_count_chars("123456789"), 3); // 9 chars = 3 tokens (rounded up)
    }

    #[test]
    fn test_summarize_proposal_short() {
        let short = "This is a short proposal.";
        let summary = summarize_proposal_for_handoff(short);
        assert_eq!(summary, short);
    }

    #[test]
    fn test_summarize_proposal_long() {
        let long = "a".repeat(500);
        let summary = summarize_proposal_for_handoff(&long);

        assert!(summary.len() <= 320);
        assert!(summary.ends_with("..."));
    }

    #[test]
    fn test_truncate_to_token_limit() {
        let text = "This is a test string for truncation.";
        let truncated = truncate_to_token_limit(text, 5); // 5 tokens = 20 chars

        assert!(truncated.len() <= 20);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_match_architecture() {
        // Cloud handoff: 150-300 tokens
        assert_eq!(CLOUD_HANDOFF_MIN_TOKENS, 150);
        assert_eq!(CLOUD_HANDOFF_MAX_TOKENS, 300);

        // Token estimation
        assert_eq!(CHARS_PER_TOKEN, 4);
    }
}
