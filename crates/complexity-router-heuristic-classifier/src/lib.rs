//! # complexity-router-heuristic-classifier
//!
//! L2 Engine: Heuristic complexity classification for 2-way routing.
//!
//! ## Executable Specification
//!
//! ### Preconditions
//! - Query is non-empty string
//!
//! ### Postconditions
//! - Returns `RoutingDecision::Local` OR `RoutingDecision::CloudHandoff`
//! - Classification takes <10ms (no LLM calls)
//!
//! ### Routing Distribution Target
//! - ~80% Local (full MoA-Lite debate)
//! - ~20% CloudHandoff (complex reasoning)
//!
//! ### Routing Triggers (from PRD FR1.3)
//! - **Local**: Code blocks, "explain"/"write"/"debug" keywords, token count < 2000
//! - **Cloud**: "design"/"architect" keywords, token count > 2000, reasoning depth > 3

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default token threshold for cloud routing
pub const DEFAULT_TOKEN_THRESHOLD: usize = 2000;

/// Default reasoning depth threshold for cloud routing
pub const DEFAULT_DEPTH_THRESHOLD: u8 = 3;

/// Approximate characters per token (for estimation)
pub const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

// ============================================================================
// ROUTING DECISION (2-Way for MVP)
// ============================================================================

/// Routing decision for query processing.
///
/// MVP uses 2-way routing:
/// - `Local`: Full MoA-Lite debate (10-17s realistic latency)
/// - `CloudHandoff`: Route to Claude API (11-18s latency)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoutingDecision {
    /// Process locally with full MoA-Lite debate
    Local,
    /// Hand off to cloud (Claude API)
    CloudHandoff,
}

impl RoutingDecision {
    /// Check if decision routes locally
    #[must_use]
    pub const fn is_local_routing_decision(&self) -> bool {
        matches!(self, Self::Local)
    }

    /// Check if decision routes to cloud
    #[must_use]
    pub const fn is_cloud_routing_decision(&self) -> bool {
        matches!(self, Self::CloudHandoff)
    }
}

// ============================================================================
// ROUTING FEATURES (Extracted from Query)
// ============================================================================

/// Features extracted from query for routing classification.
///
/// These are computed heuristically without LLM calls (<10ms).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingFeatures {
    /// Estimated token count of query
    pub token_count: usize,
    /// Whether query contains code block (```)
    pub has_code_block: bool,
    /// Whether query contains local trigger keywords
    pub has_local_keywords: bool,
    /// Whether query contains cloud trigger keywords
    pub has_cloud_keywords: bool,
    /// Estimated reasoning depth (step-by-step indicators)
    pub reasoning_depth: u8,
    /// Query character length
    pub char_count: usize,
}

impl RoutingFeatures {
    /// Extract features from query text.
    ///
    /// # Preconditions
    /// - `query` is a valid string (can be empty)
    ///
    /// # Postconditions
    /// - Returns `RoutingFeatures` with all fields populated
    /// - Extraction completes in <10ms
    #[must_use]
    pub fn extract_features_from_query(query: &str) -> Self {
        let query_lower = query.to_lowercase();

        Self {
            token_count: estimate_token_count_heuristic(query),
            has_code_block: query.contains("```"),
            has_local_keywords: check_local_trigger_keywords(&query_lower),
            has_cloud_keywords: check_cloud_trigger_keywords(&query_lower),
            reasoning_depth: estimate_reasoning_depth_heuristic(&query_lower),
            char_count: query.len(),
        }
    }

    /// Create features for testing
    #[cfg(test)]
    pub fn create_features_for_testing(
        token_count: usize,
        has_code_block: bool,
        has_local_keywords: bool,
        has_cloud_keywords: bool,
        reasoning_depth: u8,
    ) -> Self {
        Self {
            token_count,
            has_code_block,
            has_local_keywords,
            has_cloud_keywords,
            reasoning_depth,
            char_count: token_count * CHARS_PER_TOKEN_ESTIMATE,
        }
    }
}

// ============================================================================
// ROUTING RESULT (With Confidence)
// ============================================================================

/// Routing result with confidence score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingResult {
    /// The routing decision
    pub decision: RoutingDecision,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Features used for decision
    pub features: RoutingFeatures,
    /// Reason for the decision
    pub reason: String,
}

impl RoutingResult {
    /// Create routing result with details.
    #[must_use]
    pub fn create_result_with_details(
        decision: RoutingDecision,
        confidence: f32,
        features: RoutingFeatures,
        reason: String,
    ) -> Self {
        Self {
            decision,
            confidence: confidence.clamp(0.0, 1.0),
            features,
            reason,
        }
    }
}

// ============================================================================
// ROUTER CONFIGURATION
// ============================================================================

/// Configuration for the heuristic router.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Token count threshold for cloud routing
    pub token_threshold: usize,
    /// Reasoning depth threshold for cloud routing
    pub depth_threshold: u8,
    /// Whether to prefer local for code queries
    pub prefer_local_for_code: bool,
}

impl RouterConfig {
    /// Create router config with defaults.
    #[must_use]
    pub fn create_config_with_defaults() -> Self {
        Self {
            token_threshold: DEFAULT_TOKEN_THRESHOLD,
            depth_threshold: DEFAULT_DEPTH_THRESHOLD,
            prefer_local_for_code: true,
        }
    }

    /// Create router config with thresholds.
    #[must_use]
    pub fn create_config_with_thresholds(token_threshold: usize, depth_threshold: u8) -> Self {
        Self {
            token_threshold,
            depth_threshold,
            prefer_local_for_code: true,
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self::create_config_with_defaults()
    }
}

// ============================================================================
// COMPLEXITY ROUTER TRAIT
// ============================================================================

/// Trait for complexity routing implementations.
///
/// All implementations must be:
/// - Pure (no side effects)
/// - Fast (<10ms, no LLM calls)
/// - Thread-safe (Send + Sync)
pub trait ComplexityRouter: Send + Sync {
    /// Classify routing for query.
    ///
    /// # Preconditions
    /// - `query` is a valid string
    ///
    /// # Postconditions
    /// - Returns `RoutingDecision::Local` or `RoutingDecision::CloudHandoff`
    fn classify_routing_for_query(&self, query: &str) -> RoutingDecision;

    /// Classify with confidence score.
    ///
    /// # Postconditions
    /// - Returns decision with confidence (0.0 - 1.0)
    fn classify_with_confidence_score(&self, query: &str) -> RoutingResult;

    /// Get router configuration.
    fn get_router_configuration_current(&self) -> &RouterConfig;
}

// ============================================================================
// HEURISTIC ROUTER IMPLEMENTATION
// ============================================================================

/// Heuristic-based complexity router.
///
/// Uses keyword matching and token counting for fast (<10ms) classification.
/// No LLM calls required.
#[derive(Debug, Clone, Default)]
pub struct HeuristicRouter {
    config: RouterConfig,
}

impl HeuristicRouter {
    /// Create router with default configuration.
    #[must_use]
    pub fn create_router_with_defaults() -> Self {
        Self {
            config: RouterConfig::create_config_with_defaults(),
        }
    }

    /// Create router with custom configuration.
    #[must_use]
    pub fn create_router_with_config(config: RouterConfig) -> Self {
        Self { config }
    }

    /// Evaluate features and determine routing.
    fn evaluate_features_for_routing(&self, features: &RoutingFeatures) -> (RoutingDecision, f32, String) {
        // Cloud trigger 1: Token count exceeds threshold
        if features.token_count > self.config.token_threshold {
            return (
                RoutingDecision::CloudHandoff,
                0.85,
                format!(
                    "Token count {} exceeds threshold {}",
                    features.token_count, self.config.token_threshold
                ),
            );
        }

        // Cloud trigger 2: Cloud keywords + high reasoning depth
        if features.has_cloud_keywords && features.reasoning_depth > self.config.depth_threshold {
            return (
                RoutingDecision::CloudHandoff,
                0.80,
                format!(
                    "Cloud keywords detected with reasoning depth {} > {}",
                    features.reasoning_depth, self.config.depth_threshold
                ),
            );
        }

        // Cloud trigger 3: Only cloud keywords, no local keywords
        if features.has_cloud_keywords && !features.has_local_keywords && !features.has_code_block {
            return (
                RoutingDecision::CloudHandoff,
                0.70,
                "Cloud keywords detected without local indicators".to_string(),
            );
        }

        // Local trigger 1: Code block present (strong local signal)
        if features.has_code_block {
            return (
                RoutingDecision::Local,
                0.90,
                "Code block detected - routing locally".to_string(),
            );
        }

        // Local trigger 2: Local keywords present
        if features.has_local_keywords {
            return (
                RoutingDecision::Local,
                0.85,
                "Local trigger keywords detected".to_string(),
            );
        }

        // Default: Local routing (short, simple queries)
        (
            RoutingDecision::Local,
            0.75,
            "Default routing - simple query".to_string(),
        )
    }
}

impl ComplexityRouter for HeuristicRouter {
    fn classify_routing_for_query(&self, query: &str) -> RoutingDecision {
        let features = RoutingFeatures::extract_features_from_query(query);
        let (decision, _, _) = self.evaluate_features_for_routing(&features);
        decision
    }

    fn classify_with_confidence_score(&self, query: &str) -> RoutingResult {
        let features = RoutingFeatures::extract_features_from_query(query);
        let (decision, confidence, reason) = self.evaluate_features_for_routing(&features);

        RoutingResult::create_result_with_details(decision, confidence, features, reason)
    }

    fn get_router_configuration_current(&self) -> &RouterConfig {
        &self.config
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors from routing operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RouterError {
    /// Query is empty
    #[error("Query cannot be empty")]
    EmptyQuery,

    /// Configuration is invalid
    #[error("Invalid router configuration: {0}")]
    InvalidConfiguration(String),
}

// ============================================================================
// PURE HELPER FUNCTIONS (No Side Effects)
// ============================================================================

/// Check for local trigger keywords.
///
/// PRD FR1.3 local triggers: "explain", "write", "debug", "fix", "implement"
#[must_use]
pub fn check_local_trigger_keywords(query_lower: &str) -> bool {
    const LOCAL_KEYWORDS: &[&str] = &[
        "explain",
        "write",
        "debug",
        "fix",
        "implement",
        "create",
        "code",
        "function",
        "how to",
        "example",
        "syntax",
        "error",
        "bug",
    ];

    LOCAL_KEYWORDS
        .iter()
        .any(|kw| query_lower.contains(kw))
}

/// Check for cloud trigger keywords.
///
/// PRD FR1.3 cloud triggers: "design", "architect", "compare", "trade-off"
#[must_use]
pub fn check_cloud_trigger_keywords(query_lower: &str) -> bool {
    const CLOUD_KEYWORDS: &[&str] = &[
        "design",
        "architect",
        "architecture",
        "compare",
        "trade-off",
        "tradeoff",
        "trade off",
        "distributed",
        "scalable",
        "scalability",
        "system design",
        "high level",
        "strategic",
        "evaluate options",
        "pros and cons",
    ];

    CLOUD_KEYWORDS
        .iter()
        .any(|kw| query_lower.contains(kw))
}

/// Estimate reasoning depth from query.
///
/// Counts indicators of multi-step reasoning.
#[must_use]
pub fn estimate_reasoning_depth_heuristic(query_lower: &str) -> u8 {
    const DEPTH_INDICATORS: &[&str] = &[
        "step by step",
        "first",
        "then",
        "finally",
        "next",
        "after that",
        "because",
        "therefore",
        "however",
        "alternatively",
        "on the other hand",
        "considering",
        "weighing",
        "if",
        "else",
        "when",
    ];

    let count: usize = DEPTH_INDICATORS
        .iter()
        .filter(|ind| query_lower.contains(*ind))
        .count();

    // Cap at 10 to prevent overflow
    count.min(10) as u8
}

/// Estimate token count from text.
///
/// Uses heuristic: ~4 characters per token.
#[must_use]
pub fn estimate_token_count_heuristic(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.len() + CHARS_PER_TOKEN_ESTIMATE - 1) / CHARS_PER_TOKEN_ESTIMATE
}

// ============================================================================
// TESTS (TDD-First: Written BEFORE implementation)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // RoutingDecision Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_routing_decision_variants() {
        let local = RoutingDecision::Local;
        let cloud = RoutingDecision::CloudHandoff;

        assert_ne!(local, cloud);
        assert!(local.is_local_routing_decision());
        assert!(!local.is_cloud_routing_decision());
        assert!(cloud.is_cloud_routing_decision());
        assert!(!cloud.is_local_routing_decision());
    }

    #[test]
    fn test_routing_decision_serialization() {
        let decision = RoutingDecision::Local;
        let json = serde_json::to_string(&decision).unwrap();
        let deserialized: RoutingDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(decision, deserialized);
    }

    // -------------------------------------------------------------------------
    // RoutingFeatures Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_features_empty_query() {
        let features = RoutingFeatures::extract_features_from_query("");

        assert_eq!(features.token_count, 0);
        assert!(!features.has_code_block);
        assert!(!features.has_local_keywords);
        assert!(!features.has_cloud_keywords);
        assert_eq!(features.reasoning_depth, 0);
    }

    #[test]
    fn test_extract_features_code_block() {
        let query = "Here is some code:\n```rust\nfn main() {}\n```";
        let features = RoutingFeatures::extract_features_from_query(query);

        assert!(features.has_code_block);
    }

    #[test]
    fn test_extract_features_local_keywords() {
        let queries = vec![
            "Explain how async works",
            "Write a function to sort",
            "Debug this error",
            "Fix this bug",
            "How to implement a trait",
        ];

        for query in queries {
            let features = RoutingFeatures::extract_features_from_query(query);
            assert!(
                features.has_local_keywords,
                "Expected local keywords in: {query}"
            );
        }
    }

    #[test]
    fn test_extract_features_cloud_keywords() {
        let queries = vec![
            "Design a distributed system",
            "Architect a microservice",
            "Compare REST vs GraphQL trade-offs",
            "Evaluate scalability options",
        ];

        for query in queries {
            let features = RoutingFeatures::extract_features_from_query(query);
            assert!(
                features.has_cloud_keywords,
                "Expected cloud keywords in: {query}"
            );
        }
    }

    #[test]
    fn test_extract_features_reasoning_depth() {
        let query = "First, explain the concept. Then, show an example. Finally, discuss alternatives.";
        let features = RoutingFeatures::extract_features_from_query(query);

        assert!(features.reasoning_depth >= 3);
    }

    #[test]
    fn test_extract_features_token_count() {
        let query = "This is a test query with some words."; // ~10 tokens
        let features = RoutingFeatures::extract_features_from_query(query);

        assert!(features.token_count > 0);
        assert!(features.token_count < 20);
    }

    // -------------------------------------------------------------------------
    // RouterConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_router_config_defaults() {
        let config = RouterConfig::create_config_with_defaults();

        assert_eq!(config.token_threshold, DEFAULT_TOKEN_THRESHOLD);
        assert_eq!(config.depth_threshold, DEFAULT_DEPTH_THRESHOLD);
        assert!(config.prefer_local_for_code);
    }

    #[test]
    fn test_router_config_custom() {
        let config = RouterConfig::create_config_with_thresholds(1000, 2);

        assert_eq!(config.token_threshold, 1000);
        assert_eq!(config.depth_threshold, 2);
    }

    // -------------------------------------------------------------------------
    // HeuristicRouter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_router_simple_query_routes_local() {
        let router = HeuristicRouter::create_router_with_defaults();

        let decision = router.classify_routing_for_query("What is Rust?");

        assert_eq!(decision, RoutingDecision::Local);
    }

    #[test]
    fn test_router_code_block_routes_local() {
        let router = HeuristicRouter::create_router_with_defaults();
        let query = "What does this code do?\n```rust\nfn main() { println!(\"hello\"); }\n```";

        let decision = router.classify_routing_for_query(query);

        assert_eq!(decision, RoutingDecision::Local);
    }

    #[test]
    fn test_router_local_keywords_routes_local() {
        let router = HeuristicRouter::create_router_with_defaults();
        let queries = vec![
            "Explain ownership in Rust",
            "Write a sorting function",
            "Debug this async error",
            "How to implement Display trait",
        ];

        for query in queries {
            let decision = router.classify_routing_for_query(query);
            assert_eq!(
                decision,
                RoutingDecision::Local,
                "Expected Local for: {query}"
            );
        }
    }

    #[test]
    fn test_router_cloud_keywords_routes_cloud() {
        let router = HeuristicRouter::create_router_with_defaults();

        // Cloud keywords + reasoning depth
        let query = "Design a distributed cache architecture. First consider the trade-offs, then evaluate scalability options.";
        let decision = router.classify_routing_for_query(query);

        assert_eq!(decision, RoutingDecision::CloudHandoff);
    }

    #[test]
    fn test_router_long_query_routes_cloud() {
        let router = HeuristicRouter::create_router_with_defaults();

        // Create a long query (>2000 tokens = >8000 chars)
        let long_query = "a ".repeat(5000);
        let decision = router.classify_routing_for_query(&long_query);

        assert_eq!(decision, RoutingDecision::CloudHandoff);
    }

    #[test]
    fn test_router_with_confidence() {
        let router = HeuristicRouter::create_router_with_defaults();

        let result = router.classify_with_confidence_score("Explain async/await in Rust");

        assert_eq!(result.decision, RoutingDecision::Local);
        assert!(result.confidence > 0.5);
        assert!(!result.reason.is_empty());
    }

    #[test]
    fn test_router_code_block_high_confidence() {
        let router = HeuristicRouter::create_router_with_defaults();
        let query = "```rust\nfn main() {}\n```";

        let result = router.classify_with_confidence_score(query);

        assert_eq!(result.decision, RoutingDecision::Local);
        assert!(result.confidence >= 0.9, "Code block should have high confidence");
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_check_local_trigger_keywords() {
        assert!(check_local_trigger_keywords("explain this"));
        assert!(check_local_trigger_keywords("write a function"));
        assert!(check_local_trigger_keywords("debug this error"));
        assert!(check_local_trigger_keywords("how to do this"));
        assert!(!check_local_trigger_keywords("hello world"));
    }

    #[test]
    fn test_check_cloud_trigger_keywords() {
        assert!(check_cloud_trigger_keywords("design a system"));
        assert!(check_cloud_trigger_keywords("architect the solution"));
        assert!(check_cloud_trigger_keywords("compare trade-offs"));
        assert!(check_cloud_trigger_keywords("scalable architecture"));
        assert!(!check_cloud_trigger_keywords("simple function"));
    }

    #[test]
    fn test_estimate_reasoning_depth_heuristic() {
        assert_eq!(estimate_reasoning_depth_heuristic("hello world"), 0);
        assert!(estimate_reasoning_depth_heuristic("first do this") >= 1);
        assert!(estimate_reasoning_depth_heuristic("first, then, finally") >= 3);
        assert!(estimate_reasoning_depth_heuristic("step by step analysis") >= 1);
    }

    #[test]
    fn test_estimate_token_count_heuristic() {
        assert_eq!(estimate_token_count_heuristic(""), 0);
        assert_eq!(estimate_token_count_heuristic("test"), 1); // 4 chars
        assert_eq!(estimate_token_count_heuristic("12345678"), 2); // 8 chars
        assert_eq!(estimate_token_count_heuristic("123456789"), 3); // 9 chars, rounded up
    }

    // -------------------------------------------------------------------------
    // Routing Distribution Tests (Target: 80% Local, 20% Cloud)
    // -------------------------------------------------------------------------

    #[test]
    fn test_routing_distribution_typical_queries() {
        let router = HeuristicRouter::create_router_with_defaults();

        // Typical coding queries that should route locally
        let local_queries = vec![
            "What is ownership?",
            "Explain borrowing",
            "How do I use async?",
            "Write a hello world",
            "Debug this error message",
            "```rust\nfn main() {}\n```",
            "Fix this bug",
            "Implement a trait",
            "Create a function",
            "Show me an example",
        ];

        // Complex queries that should route to cloud
        let cloud_queries = vec![
            "Design a distributed microservice architecture considering trade-offs between consistency and availability. First analyze the requirements, then evaluate options.",
            "Compare different database architectures and their scalability trade-offs for a high-traffic system",
        ];

        let local_count = local_queries
            .iter()
            .filter(|q| router.classify_routing_for_query(q) == RoutingDecision::Local)
            .count();

        let cloud_count = cloud_queries
            .iter()
            .filter(|q| router.classify_routing_for_query(q) == RoutingDecision::CloudHandoff)
            .count();

        // Most typical coding queries should route locally
        assert!(
            local_count >= 8,
            "Expected at least 80% local routing, got {}/{} local",
            local_count,
            local_queries.len()
        );

        // Complex design queries should route to cloud
        assert!(
            cloud_count >= 1,
            "Expected complex queries to route to cloud"
        );
    }

    // -------------------------------------------------------------------------
    // RoutingResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_routing_result_creation() {
        let features = RoutingFeatures::extract_features_from_query("test query");
        let result = RoutingResult::create_result_with_details(
            RoutingDecision::Local,
            0.85,
            features,
            "Test reason".to_string(),
        );

        assert_eq!(result.decision, RoutingDecision::Local);
        assert!((result.confidence - 0.85).abs() < f32::EPSILON);
        assert_eq!(result.reason, "Test reason");
    }

    #[test]
    fn test_routing_result_clamps_confidence() {
        let features = RoutingFeatures::extract_features_from_query("test");

        let result1 = RoutingResult::create_result_with_details(
            RoutingDecision::Local,
            1.5, // Above max
            features.clone(),
            "Test".to_string(),
        );
        assert!((result1.confidence - 1.0).abs() < f32::EPSILON);

        let result2 = RoutingResult::create_result_with_details(
            RoutingDecision::Local,
            -0.5, // Below min
            features,
            "Test".to_string(),
        );
        assert!((result2.confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_routing_result_serialization() {
        let features = RoutingFeatures::extract_features_from_query("test");
        let result = RoutingResult::create_result_with_details(
            RoutingDecision::Local,
            0.8,
            features,
            "Test".to_string(),
        );

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: RoutingResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.decision, deserialized.decision);
        assert!((result.confidence - deserialized.confidence).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants_match_architecture() {
        // PRD FR1.3: Token threshold ~2000
        assert_eq!(DEFAULT_TOKEN_THRESHOLD, 2000);

        // Architecture: Reasoning depth threshold 3
        assert_eq!(DEFAULT_DEPTH_THRESHOLD, 3);

        // Token estimation: ~4 chars per token
        assert_eq!(CHARS_PER_TOKEN_ESTIMATE, 4);
    }
}
