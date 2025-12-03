//! # agent-role-definition-types
//!
//! L1 Core: Agent role traits and type definitions for MoA-Lite debate system.
//!
//! ## Executable Specification
//!
//! ### Preconditions
//! - None (pure type definitions)
//!
//! ### Postconditions
//! - AgentRole enum defines exactly 2 roles: Proposer, Aggregator
//! - All types implement Clone, Debug, Serialize, Deserialize
//!
//! ### Invariants
//! - Proposer temperature: 0.6-0.8
//! - Aggregator temperature: 0.4-0.6
//! - Proposer max_tokens: 150-300
//! - Aggregator max_tokens: 200-500

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};

// ============================================================================
// CONSTANTS (Token Budgets from PRD FR1.2)
// ============================================================================

/// Minimum tokens for proposer output
pub const PROPOSER_MIN_TOKENS: usize = 150;

/// Maximum tokens for proposer output
pub const PROPOSER_MAX_TOKENS: usize = 300;

/// Minimum tokens for aggregator output
pub const AGGREGATOR_MIN_TOKENS: usize = 200;

/// Maximum tokens for aggregator output
pub const AGGREGATOR_MAX_TOKENS: usize = 500;

/// Default proposer temperature
pub const PROPOSER_DEFAULT_TEMP: f32 = 0.7;

/// Default aggregator temperature
pub const AGGREGATOR_DEFAULT_TEMP: f32 = 0.5;

/// Number of proposers in MoA-Lite
pub const PROPOSER_COUNT: usize = 3;

// ============================================================================
// AGENT ROLE ENUM
// ============================================================================

/// Agent role in the MoA-Lite debate system.
///
/// MoA-Lite uses a 2-layer architecture:
/// - Layer 1: 3 Proposers generate initial responses in parallel
/// - Layer 2: 1 Aggregator synthesizes the best answer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Generates initial proposals (3 instances run in parallel)
    Proposer,
    /// Synthesizes final answer from all proposals
    Aggregator,
}

impl AgentRole {
    /// Get default temperature for role
    #[must_use]
    pub const fn default_temperature_for_role(&self) -> f32 {
        match self {
            Self::Proposer => PROPOSER_DEFAULT_TEMP,
            Self::Aggregator => AGGREGATOR_DEFAULT_TEMP,
        }
    }

    /// Get default max tokens for role
    #[must_use]
    pub const fn default_max_tokens_for_role(&self) -> usize {
        match self {
            Self::Proposer => PROPOSER_MAX_TOKENS,
            Self::Aggregator => AGGREGATOR_MAX_TOKENS,
        }
    }
}

// ============================================================================
// PROPOSER CONFIG (Immutable)
// ============================================================================

/// Configuration for a proposer agent (immutable after creation).
///
/// Each proposer has a unique index (0, 1, 2) and system prompt
/// to encourage diverse perspectives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposerConfig {
    /// Temperature for generation (0.6-0.8)
    temperature: f32,
    /// Maximum tokens to generate (150-300)
    max_tokens_output: usize,
    /// System prompt template
    system_prompt: String,
    /// Proposer instance index (0, 1, or 2)
    instance_index: u8,
}

impl ProposerConfig {
    /// Create proposer config with index.
    ///
    /// # Preconditions
    /// - `index` must be in range 0..3
    ///
    /// # Postconditions
    /// - Returns valid `ProposerConfig` with appropriate system prompt
    ///
    /// # Panics
    /// Never panics - invalid index defaults to proposer 0 prompt
    #[must_use]
    pub fn create_config_with_index(index: u8) -> Self {
        Self {
            temperature: PROPOSER_DEFAULT_TEMP,
            max_tokens_output: (PROPOSER_MIN_TOKENS + PROPOSER_MAX_TOKENS) / 2, // 225
            system_prompt: Self::system_prompt_for_index(index),
            instance_index: index.min(2), // Clamp to valid range
        }
    }

    /// Create all three proposer configs.
    ///
    /// # Postconditions
    /// - Returns array of exactly 3 `ProposerConfig`
    #[must_use]
    pub fn create_all_proposer_configs() -> [Self; PROPOSER_COUNT] {
        [
            Self::create_config_with_index(0),
            Self::create_config_with_index(1),
            Self::create_config_with_index(2),
        ]
    }

    /// Get system prompt for proposer index.
    ///
    /// Different prompts encourage diverse perspectives:
    /// - Index 0: Focus on accuracy and correctness
    /// - Index 1: Focus on creativity and alternatives
    /// - Index 2: Focus on conciseness and clarity
    #[must_use]
    fn system_prompt_for_index(index: u8) -> String {
        match index {
            0 => Self::prompt_accuracy_focused_proposer(),
            1 => Self::prompt_creativity_focused_proposer(),
            2 => Self::prompt_conciseness_focused_proposer(),
            _ => Self::prompt_accuracy_focused_proposer(),
        }
    }

    fn prompt_accuracy_focused_proposer() -> String {
        String::from(
            "You are an expert coding assistant focused on ACCURACY. \
             Analyze this query and provide a clear, technically correct response. \
             Prioritize correctness over brevity. \
             Include code examples where helpful.",
        )
    }

    fn prompt_creativity_focused_proposer() -> String {
        String::from(
            "You are an expert coding assistant focused on CREATIVE SOLUTIONS. \
             Analyze this query and provide alternative approaches. \
             Consider unconventional but valid solutions. \
             Explain trade-offs between different approaches.",
        )
    }

    fn prompt_conciseness_focused_proposer() -> String {
        String::from(
            "You are an expert coding assistant focused on CONCISENESS. \
             Analyze this query and provide a clear, brief response. \
             Get to the point quickly. \
             Include only essential code examples.",
        )
    }

    // Pure getters (no mutation)

    /// Get temperature value
    #[must_use]
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get max tokens value
    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens_output
    }

    /// Get system prompt
    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Get instance index
    #[must_use]
    pub const fn index(&self) -> u8 {
        self.instance_index
    }
}

// ============================================================================
// AGGREGATOR CONFIG (Immutable)
// ============================================================================

/// Configuration for the aggregator agent (immutable after creation).
///
/// The aggregator synthesizes multiple proposals into a single
/// high-quality response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Temperature for generation (0.4-0.6)
    temperature: f32,
    /// Maximum tokens to generate (200-500)
    max_tokens_output: usize,
    /// System prompt template
    system_prompt: String,
}

impl AggregatorConfig {
    /// Create default aggregator config.
    ///
    /// # Postconditions
    /// - Returns valid `AggregatorConfig` with synthesis prompt
    #[must_use]
    pub fn create_config_default_aggregator() -> Self {
        Self {
            temperature: AGGREGATOR_DEFAULT_TEMP,
            max_tokens_output: (AGGREGATOR_MIN_TOKENS + AGGREGATOR_MAX_TOKENS) / 2, // 350
            system_prompt: Self::default_aggregator_system_prompt(),
        }
    }

    fn default_aggregator_system_prompt() -> String {
        String::from(
            "You are synthesizing multiple responses to create the best answer. \
             Review these proposals and create a unified response that: \
             - Takes the best ideas from each proposal \
             - Resolves any contradictions \
             - Provides the most accurate and helpful answer \
             - Is well-structured and clear",
        )
    }

    // Pure getters (no mutation)

    /// Get temperature value
    #[must_use]
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get max tokens value
    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens_output
    }

    /// Get system prompt
    #[must_use]
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self::create_config_default_aggregator()
    }
}

// ============================================================================
// VALIDATION FUNCTIONS (Pure, No Side Effects)
// ============================================================================

/// Validate proposer token count is within budget.
///
/// # Preconditions
/// - `token_count` is the actual token count of proposer output
///
/// # Postconditions
/// - Returns `true` if within budget (150-300)
#[must_use]
pub const fn validate_proposer_token_budget(token_count: usize) -> bool {
    token_count >= PROPOSER_MIN_TOKENS && token_count <= PROPOSER_MAX_TOKENS
}

/// Validate aggregator token count is within budget.
///
/// # Preconditions
/// - `token_count` is the actual token count of aggregator output
///
/// # Postconditions
/// - Returns `true` if within budget (200-500)
#[must_use]
pub const fn validate_aggregator_token_budget(token_count: usize) -> bool {
    token_count >= AGGREGATOR_MIN_TOKENS && token_count <= AGGREGATOR_MAX_TOKENS
}

/// Validate temperature is within valid range for role.
///
/// # Postconditions
/// - Returns `true` if temperature is valid for the role
#[must_use]
pub fn validate_temperature_for_role(role: AgentRole, temperature: f32) -> bool {
    match role {
        AgentRole::Proposer => (0.6..=0.8).contains(&temperature),
        AgentRole::Aggregator => (0.4..=0.6).contains(&temperature),
    }
}

// ============================================================================
// TESTS (TDD-First: These were written BEFORE implementation)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // AgentRole Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_role_enum_values() {
        // Contract: Exactly 2 roles exist
        let proposer = AgentRole::Proposer;
        let aggregator = AgentRole::Aggregator;

        assert_ne!(proposer, aggregator);
    }

    #[test]
    fn test_agent_role_default_temperature() {
        assert!((AgentRole::Proposer.default_temperature_for_role() - 0.7).abs() < f32::EPSILON);
        assert!((AgentRole::Aggregator.default_temperature_for_role() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_agent_role_default_max_tokens() {
        assert_eq!(AgentRole::Proposer.default_max_tokens_for_role(), 300);
        assert_eq!(AgentRole::Aggregator.default_max_tokens_for_role(), 500);
    }

    #[test]
    fn test_agent_role_serialization() {
        let proposer = AgentRole::Proposer;
        let json = serde_json::to_string(&proposer).unwrap();
        let deserialized: AgentRole = serde_json::from_str(&json).unwrap();
        assert_eq!(proposer, deserialized);
    }

    // -------------------------------------------------------------------------
    // ProposerConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_proposer_config_index_zero() {
        let config = ProposerConfig::create_config_with_index(0);

        assert_eq!(config.index(), 0);
        assert!((config.temperature() - 0.7).abs() < f32::EPSILON);
        assert!(config.system_prompt().contains("ACCURACY"));
    }

    #[test]
    fn test_create_proposer_config_index_one() {
        let config = ProposerConfig::create_config_with_index(1);

        assert_eq!(config.index(), 1);
        assert!(config.system_prompt().contains("CREATIVE"));
    }

    #[test]
    fn test_create_proposer_config_index_two() {
        let config = ProposerConfig::create_config_with_index(2);

        assert_eq!(config.index(), 2);
        assert!(config.system_prompt().contains("CONCISENESS"));
    }

    #[test]
    fn test_create_proposer_config_invalid_index_clamped() {
        // Contract: Invalid index should be clamped, not panic
        let config = ProposerConfig::create_config_with_index(99);

        assert_eq!(config.index(), 2); // Clamped to max valid
    }

    #[test]
    fn test_create_all_proposer_configs() {
        let configs = ProposerConfig::create_all_proposer_configs();

        assert_eq!(configs.len(), 3);
        assert_eq!(configs[0].index(), 0);
        assert_eq!(configs[1].index(), 1);
        assert_eq!(configs[2].index(), 2);

        // All should have different prompts
        assert_ne!(configs[0].system_prompt(), configs[1].system_prompt());
        assert_ne!(configs[1].system_prompt(), configs[2].system_prompt());
    }

    #[test]
    fn test_proposer_config_immutability() {
        // Contract: Config is immutable (no setters)
        let config = ProposerConfig::create_config_with_index(0);

        // These are the only ways to access values - all are getters
        let _ = config.temperature();
        let _ = config.max_tokens();
        let _ = config.system_prompt();
        let _ = config.index();

        // Clone creates a new independent value
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_proposer_config_serialization() {
        let config = ProposerConfig::create_config_with_index(1);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProposerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    // -------------------------------------------------------------------------
    // AggregatorConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_aggregator_config_default() {
        let config = AggregatorConfig::create_config_default_aggregator();

        assert!((config.temperature() - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens(), 350); // (200 + 500) / 2
        assert!(config.system_prompt().contains("synthesizing"));
    }

    #[test]
    fn test_aggregator_config_default_trait() {
        let config1 = AggregatorConfig::default();
        let config2 = AggregatorConfig::create_config_default_aggregator();

        assert_eq!(config1, config2);
    }

    #[test]
    fn test_aggregator_config_serialization() {
        let config = AggregatorConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AggregatorConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config, deserialized);
    }

    // -------------------------------------------------------------------------
    // Validation Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_proposer_token_budget_valid() {
        assert!(validate_proposer_token_budget(150)); // Min
        assert!(validate_proposer_token_budget(225)); // Middle
        assert!(validate_proposer_token_budget(300)); // Max
    }

    #[test]
    fn test_validate_proposer_token_budget_invalid() {
        assert!(!validate_proposer_token_budget(0));
        assert!(!validate_proposer_token_budget(149)); // Below min
        assert!(!validate_proposer_token_budget(301)); // Above max
        assert!(!validate_proposer_token_budget(1000));
    }

    #[test]
    fn test_validate_aggregator_token_budget_valid() {
        assert!(validate_aggregator_token_budget(200)); // Min
        assert!(validate_aggregator_token_budget(350)); // Middle
        assert!(validate_aggregator_token_budget(500)); // Max
    }

    #[test]
    fn test_validate_aggregator_token_budget_invalid() {
        assert!(!validate_aggregator_token_budget(0));
        assert!(!validate_aggregator_token_budget(199)); // Below min
        assert!(!validate_aggregator_token_budget(501)); // Above max
    }

    #[test]
    fn test_validate_temperature_for_proposer() {
        assert!(validate_temperature_for_role(AgentRole::Proposer, 0.6));
        assert!(validate_temperature_for_role(AgentRole::Proposer, 0.7));
        assert!(validate_temperature_for_role(AgentRole::Proposer, 0.8));

        assert!(!validate_temperature_for_role(AgentRole::Proposer, 0.5));
        assert!(!validate_temperature_for_role(AgentRole::Proposer, 0.9));
    }

    #[test]
    fn test_validate_temperature_for_aggregator() {
        assert!(validate_temperature_for_role(AgentRole::Aggregator, 0.4));
        assert!(validate_temperature_for_role(AgentRole::Aggregator, 0.5));
        assert!(validate_temperature_for_role(AgentRole::Aggregator, 0.6));

        assert!(!validate_temperature_for_role(AgentRole::Aggregator, 0.3));
        assert!(!validate_temperature_for_role(AgentRole::Aggregator, 0.7));
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_match_prd() {
        // PRD FR1.2: Proposer outputs are 150-300 tokens
        assert_eq!(PROPOSER_MIN_TOKENS, 150);
        assert_eq!(PROPOSER_MAX_TOKENS, 300);

        // PRD FR1.2: Aggregator output is 200-500 tokens
        assert_eq!(AGGREGATOR_MIN_TOKENS, 200);
        assert_eq!(AGGREGATOR_MAX_TOKENS, 500);

        // MoA-Lite: 3 proposers
        assert_eq!(PROPOSER_COUNT, 3);
    }
}
