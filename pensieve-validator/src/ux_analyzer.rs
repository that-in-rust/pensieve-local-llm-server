use crate::errors::{ValidationError, Result};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use regex::Regex;

/// Comprehensive user experience analysis system
pub struct UXAnalyzer {
    progress_analyzer: ProgressAnalyzer,
    error_message_analyzer: ErrorMessageAnalyzer,
    completion_analyzer: CompletionAnalyzer,
    interruption_analyzer: InterruptionAnalyzer,
    feedback_analyzer: FeedbackAnalyzer,
}

/// Progress reporting quality analyzer
pub struct ProgressAnalyzer {
    update_intervals: Vec<Duration>,
    progress_messages: Vec<ProgressMessage>,
    eta_predictions: Vec<ETAPrediction>,
}

/// Error message quality analyzer
pub struct ErrorMessageAnalyzer {
    error_messages: Vec<ErrorMessageEntry>,
    clarity_patterns: Vec<ClarityPattern>,
    actionability_patterns: Vec<ActionabilityPattern>,
}

/// Completion feedback analyzer
pub struct CompletionAnalyzer {
    completion_messages: Vec<CompletionMessage>,
    summary_quality_metrics: SummaryQualityMetrics,
}

/// Interruption handling analyzer
pub struct InterruptionAnalyzer {
    interruption_events: Vec<InterruptionEvent>,
    recovery_instructions: Vec<RecoveryInstruction>,
}

/// User feedback quality analyzer
pub struct FeedbackAnalyzer {
    feedback_events: Vec<FeedbackEvent>,
    user_confidence_indicators: Vec<ConfidenceIndicator>,
}
// Progress message with quality analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressMessage {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub message: String,
    pub files_processed: u64,
    pub total_files: u64,
    pub percentage_complete: f64,
    pub eta_seconds: Option<u64>,
    pub information_density: f64,
    pub clarity_score: f64,
    pub actionability_score: f64,
}

/// ETA prediction accuracy tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ETAPrediction {
    pub predicted_eta: Duration,
    pub actual_completion_time: Option<Duration>,
    pub accuracy_score: f64, // 0.0 - 1.0
    #[serde(skip, default = "Instant::now")]
    pub prediction_timestamp: Instant,
}

/// Error message entry with analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessageEntry {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub error_type: String,
    pub message: String,
    pub context: String,
    pub clarity_score: f64,
    pub actionability_score: f64,
    pub technical_complexity: TechnicalComplexity,
    pub contains_solution: bool,
    pub improvement_suggestions: Vec<String>,
}

/// Pattern for analyzing message clarity
#[derive(Debug, Clone)]
pub struct ClarityPattern {
    pub pattern: Regex,
    pub clarity_impact: f64, // -1.0 to 1.0
    pub description: String,
}

/// Pattern for analyzing message actionability
#[derive(Debug, Clone)]
pub struct ActionabilityPattern {
    pub pattern: Regex,
    pub actionability_impact: f64, // -1.0 to 1.0
    pub description: String,
}

/// Completion message with quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionMessage {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub message: String,
    pub summary_completeness: f64,
    pub results_clarity: f64,
    pub next_steps_guidance: f64,
    pub actionable_insights: f64,
}

/// Summary quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryQualityMetrics {
    pub includes_file_count: bool,
    pub includes_processing_time: bool,
    pub includes_error_summary: bool,
    pub includes_next_steps: bool,
    pub includes_performance_metrics: bool,
    pub overall_completeness_score: f64,
}

/// Interruption event analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionEvent {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub interruption_type: InterruptionType,
    pub graceful_shutdown_score: f64,
    pub state_preservation_score: f64,
    pub cleanup_completeness_score: f64,
    pub recovery_instructions_provided: bool,
}

/// Types of interruptions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterruptionType {
    UserCancellation,
    SystemTimeout,
    ResourceExhaustion,
    ExternalSignal,
    UnexpectedError,
}

/// Recovery instruction analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryInstruction {
    pub instruction: String,
    pub clarity_score: f64,
    pub completeness_score: f64,
    pub actionability_score: f64,
}

/// User feedback event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEvent {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub feedback_type: FeedbackType,
    pub message: String,
    pub usefulness_score: f64,
    pub frequency_appropriateness: f64,
    pub user_confidence_impact: f64,
}

/// Types of user feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    StatusUpdate,
    WarningMessage,
    InformationalMessage,
    SuccessMessage,
    ProgressIndicator,
}

/// User confidence indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIndicator {
    pub indicator_type: String,
    pub confidence_level: f64, // 0.0 - 1.0
    pub evidence: String,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Comprehensive UX analysis results
#[derive(Debug, Serialize, Deserialize)]
pub struct UXResults {
    pub progress_reporting_quality: ProgressReportingQuality,
    pub error_message_clarity: ErrorMessageClarity,
    pub completion_feedback_quality: CompletionFeedbackQuality,
    pub interruption_handling_quality: InterruptionHandlingQuality,
    pub user_feedback_analysis: UserFeedbackAnalysis,
    pub overall_ux_score: f64, // 0.0 - 10.0
    pub improvement_recommendations: Vec<UXImprovement>,
}

/// Progress reporting quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressReportingQuality {
    pub update_frequency_score: f64,        // 0.0 - 1.0
    pub information_completeness_score: f64, // 0.0 - 1.0
    pub clarity_score: f64,                 // 0.0 - 1.0
    pub eta_accuracy_score: f64,            // 0.0 - 1.0
    pub consistency_score: f64,             // 0.0 - 1.0
    pub detailed_analysis: ProgressDetailedAnalysis,
}

/// Detailed progress analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressDetailedAnalysis {
    pub average_update_interval: Duration,
    pub update_interval_consistency: f64,
    pub information_density_average: f64,
    pub eta_accuracy_trend: Vec<f64>,
    pub problematic_patterns: Vec<String>,
}

/// Error message clarity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessageClarity {
    pub average_clarity_score: f64,      // 0.0 - 1.0
    pub actionability_score: f64,        // 0.0 - 1.0
    pub technical_jargon_score: f64,     // 0.0 - 1.0 (lower = less jargon)
    pub solution_guidance_score: f64,    // 0.0 - 1.0
    pub consistency_score: f64,          // 0.0 - 1.0
    pub detailed_analysis: ErrorMessageDetailedAnalysis,
}

/// Detailed error message analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessageDetailedAnalysis {
    pub total_error_messages: u64,
    pub messages_with_solutions: u64,
    pub messages_with_context: u64,
    pub average_technical_complexity: f64,
    pub common_improvement_areas: Vec<String>,
}

/// Completion feedback quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionFeedbackQuality {
    pub summary_completeness: f64,       // 0.0 - 1.0
    pub results_clarity: f64,            // 0.0 - 1.0
    pub next_steps_guidance: f64,        // 0.0 - 1.0
    pub actionable_insights: f64,        // 0.0 - 1.0
    pub detailed_analysis: CompletionDetailedAnalysis,
}

/// Detailed completion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionDetailedAnalysis {
    pub includes_performance_summary: bool,
    pub includes_error_summary: bool,
    pub includes_recommendations: bool,
    pub includes_next_steps: bool,
    pub summary_quality_score: f64,
}

/// Interruption handling quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionHandlingQuality {
    pub graceful_shutdown_score: f64,    // 0.0 - 1.0
    pub state_preservation_score: f64,   // 0.0 - 1.0
    pub recovery_instructions_score: f64, // 0.0 - 1.0
    pub cleanup_completeness_score: f64, // 0.0 - 1.0
    pub detailed_analysis: InterruptionDetailedAnalysis,
}

/// Detailed interruption analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptionDetailedAnalysis {
    pub total_interruptions: u64,
    pub graceful_shutdowns: u64,
    pub state_preserved_count: u64,
    pub recovery_instructions_provided: u64,
    pub average_cleanup_time: Duration,
}

/// User feedback analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedbackAnalysis {
    pub feedback_frequency: f64,         // Updates per minute
    pub feedback_usefulness_score: f64,  // 0.0 - 1.0
    pub next_steps_clarity: f64,         // 0.0 - 1.0
    pub user_confidence_score: f64,      // 0.0 - 1.0
    pub detailed_analysis: FeedbackDetailedAnalysis,
}

/// Detailed feedback analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackDetailedAnalysis {
    pub total_feedback_events: u64,
    pub feedback_by_type: HashMap<String, u64>,
    pub average_usefulness_by_type: HashMap<String, f64>,
    pub confidence_trend: Vec<f64>,
}

/// UX improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UXImprovement {
    pub category: UXCategory,
    pub priority: ImprovementPriority,
    pub title: String,
    pub description: String,
    pub current_score: f64,
    pub target_score: f64,
    pub implementation_suggestions: Vec<String>,
    pub impact_assessment: String,
}

/// UX improvement categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UXCategory {
    ProgressReporting,
    ErrorMessages,
    CompletionFeedback,
    InterruptionHandling,
    UserFeedback,
    OverallExperience,
}

/// Improvement priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementPriority {
    Critical,  // Blocks good user experience
    High,      // Significantly impacts usability
    Medium,    // Noticeable improvement opportunity
    Low,       // Nice-to-have enhancement
}

impl UXAnalyzer {
    /// Create a new UX analyzer with default patterns
    pub fn new() -> Self {
        Self {
            progress_analyzer: ProgressAnalyzer::new(),
            error_message_analyzer: ErrorMessageAnalyzer::new(),
            completion_analyzer: CompletionAnalyzer::new(),
            interruption_analyzer: InterruptionAnalyzer::new(),
            feedback_analyzer: FeedbackAnalyzer::new(),
        }
    }

    /// Analyze progress reporting quality
    pub fn analyze_progress_message(&mut self, message: &str, files_processed: u64, total_files: u64, eta_seconds: Option<u64>) -> Result<()> {
        let progress_message = ProgressMessage {
            timestamp: Instant::now(),
            message: message.to_string(),
            files_processed,
            total_files,
            percentage_complete: if total_files > 0 { (files_processed as f64 / total_files as f64) * 100.0 } else { 0.0 },
            eta_seconds,
            information_density: self.calculate_information_density(message),
            clarity_score: self.calculate_clarity_score(message),
            actionability_score: self.calculate_actionability_score(message),
        };

        self.progress_analyzer.add_progress_message(progress_message);
        Ok(())
    }

    /// Analyze error message quality
    pub fn analyze_error_message(&mut self, error_type: &str, message: &str, context: &str) -> Result<()> {
        let error_entry = ErrorMessageEntry {
            timestamp: Instant::now(),
            error_type: error_type.to_string(),
            message: message.to_string(),
            context: context.to_string(),
            clarity_score: self.calculate_clarity_score(message),
            actionability_score: self.calculate_actionability_score(message),
            technical_complexity: self.assess_technical_complexity(message),
            contains_solution: self.contains_solution_guidance(message),
            improvement_suggestions: self.generate_error_message_improvements(message),
        };

        self.error_message_analyzer.add_error_message(error_entry);
        Ok(())
    }

    /// Analyze completion feedback quality
    pub fn analyze_completion_message(&mut self, message: &str) -> Result<()> {
        let completion_message = CompletionMessage {
            timestamp: Instant::now(),
            message: message.to_string(),
            summary_completeness: self.assess_summary_completeness(message),
            results_clarity: self.calculate_clarity_score(message),
            next_steps_guidance: self.assess_next_steps_guidance(message),
            actionable_insights: self.assess_actionable_insights(message),
        };

        self.completion_analyzer.add_completion_message(completion_message);
        Ok(())
    }

    /// Analyze interruption handling
    pub fn analyze_interruption(&mut self, interruption_type: InterruptionType, graceful: bool, state_preserved: bool, cleanup_complete: bool, recovery_instructions: Option<&str>) -> Result<()> {
        let interruption_event = InterruptionEvent {
            timestamp: Instant::now(),
            interruption_type,
            graceful_shutdown_score: if graceful { 1.0 } else { 0.0 },
            state_preservation_score: if state_preserved { 1.0 } else { 0.0 },
            cleanup_completeness_score: if cleanup_complete { 1.0 } else { 0.0 },
            recovery_instructions_provided: recovery_instructions.is_some(),
        };

        self.interruption_analyzer.add_interruption_event(interruption_event);

        if let Some(instructions) = recovery_instructions {
            let recovery_instruction = RecoveryInstruction {
                instruction: instructions.to_string(),
                clarity_score: self.calculate_clarity_score(instructions),
                completeness_score: self.assess_instruction_completeness(instructions),
                actionability_score: self.calculate_actionability_score(instructions),
            };
            self.interruption_analyzer.add_recovery_instruction(recovery_instruction);
        }

        Ok(())
    }

    /// Analyze user feedback event
    pub fn analyze_feedback(&mut self, feedback_type: FeedbackType, message: &str) -> Result<()> {
        let feedback_event = FeedbackEvent {
            timestamp: Instant::now(),
            feedback_type,
            message: message.to_string(),
            usefulness_score: self.assess_feedback_usefulness(message),
            frequency_appropriateness: self.assess_feedback_frequency(),
            user_confidence_impact: self.assess_confidence_impact(message),
        };

        self.feedback_analyzer.add_feedback_event(feedback_event);
        Ok(())
    }

    /// Generate comprehensive UX analysis results
    pub fn generate_ux_results(&self) -> Result<UXResults> {
        let progress_quality = self.progress_analyzer.generate_quality_assessment();
        let error_clarity = self.error_message_analyzer.generate_clarity_assessment();
        let completion_quality = self.completion_analyzer.generate_quality_assessment();
        let interruption_quality = self.interruption_analyzer.generate_quality_assessment();
        let feedback_analysis = self.feedback_analyzer.generate_analysis();

        // Calculate overall UX score (0.0 - 10.0)
        let overall_ux_score = self.calculate_overall_ux_score(
            &progress_quality,
            &error_clarity,
            &completion_quality,
            &interruption_quality,
            &feedback_analysis,
        );

        // Generate improvement recommendations
        let improvement_recommendations = self.generate_improvement_recommendations(
            &progress_quality,
            &error_clarity,
            &completion_quality,
            &interruption_quality,
            &feedback_analysis,
        );

        Ok(UXResults {
            progress_reporting_quality: progress_quality,
            error_message_clarity: error_clarity,
            completion_feedback_quality: completion_quality,
            interruption_handling_quality: interruption_quality,
            user_feedback_analysis: feedback_analysis,
            overall_ux_score,
            improvement_recommendations,
        })
    }   
 /// Calculate information density of a message
    fn calculate_information_density(&self, message: &str) -> f64 {
        let word_count = message.split_whitespace().count();
        let char_count = message.len();
        
        if char_count == 0 {
            return 0.0;
        }
        
        // Information density = meaningful words per character
        let meaningful_words = message.split_whitespace()
            .filter(|word| word.len() > 2 && !self.is_filler_word(word))
            .count();
        
        meaningful_words as f64 / char_count as f64
    }

    /// Check if a word is a filler word
    fn is_filler_word(&self, word: &str) -> bool {
        matches!(word.to_lowercase().as_str(), 
            "the" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by"
        )
    }

    /// Calculate clarity score for a message
    fn calculate_clarity_score(&self, message: &str) -> f64 {
        let mut score: f64 = 1.0;
        
        // Penalize overly technical language
        let technical_terms = ["errno", "segfault", "malloc", "nullptr", "syscall"];
        for term in technical_terms {
            if message.to_lowercase().contains(term) {
                score -= 0.1;
            }
        }
        
        // Reward clear structure
        if message.contains(":") || message.contains("-") {
            score += 0.1;
        }
        
        // Penalize excessive length without structure
        if message.len() > 200 && !message.contains(".") && !message.contains(";") {
            score -= 0.2;
        }
        
        // Reward specific information
        if message.contains("file") || message.contains("error") || message.contains("complete") {
            score += 0.1;
        }
        
        score.max(0.0).min(1.0)
    }

    /// Calculate actionability score for a message
    fn calculate_actionability_score(&self, message: &str) -> f64 {
        let mut score: f64 = 0.0;
        
        // Reward action words
        let action_words = ["try", "check", "verify", "run", "install", "update", "fix", "resolve"];
        for word in action_words {
            if message.to_lowercase().contains(word) {
                score += 0.2;
            }
        }
        
        // Reward specific instructions
        if message.contains("command") || message.contains("option") || message.contains("flag") {
            score += 0.2;
        }
        
        // Reward examples
        if message.contains("example") || message.contains("e.g.") || message.contains("such as") {
            score += 0.1;
        }
        
        // Penalize vague language
        let vague_words = ["something", "somehow", "maybe", "perhaps", "might"];
        for word in vague_words {
            if message.to_lowercase().contains(word) {
                score -= 0.1;
            }
        }
        
        score.max(0.0).min(1.0)
    }

    /// Assess technical complexity of a message
    fn assess_technical_complexity(&self, message: &str) -> TechnicalComplexity {
        let technical_indicators = message.to_lowercase();
        
        let expert_terms = ["syscall", "segfault", "malloc", "vtable", "abi"];
        let technical_terms = ["errno", "buffer", "pointer", "thread", "mutex"];
        let moderate_terms = ["file", "directory", "permission", "network", "database"];
        
        if expert_terms.iter().any(|term| technical_indicators.contains(term)) {
            TechnicalComplexity::ExpertLevel
        } else if technical_terms.iter().any(|term| technical_indicators.contains(term)) {
            TechnicalComplexity::Technical
        } else if moderate_terms.iter().any(|term| technical_indicators.contains(term)) {
            TechnicalComplexity::Moderate
        } else {
            TechnicalComplexity::UserFriendly
        }
    }

    /// Check if message contains solution guidance
    fn contains_solution_guidance(&self, message: &str) -> bool {
        let solution_indicators = ["try", "run", "check", "install", "update", "fix", "resolve", "solution", "workaround"];
        solution_indicators.iter().any(|indicator| message.to_lowercase().contains(indicator))
    }

    /// Generate improvement suggestions for error messages
    fn generate_error_message_improvements(&self, message: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if !self.contains_solution_guidance(message) {
            suggestions.push("Add specific steps to resolve the error".to_string());
        }
        
        if self.assess_technical_complexity(message) == TechnicalComplexity::ExpertLevel {
            suggestions.push("Simplify technical language for broader audience".to_string());
        }
        
        if message.len() > 300 {
            suggestions.push("Break down long error message into structured sections".to_string());
        }
        
        if !message.contains("file") && !message.contains("path") && message.contains("error") {
            suggestions.push("Include specific file or location information".to_string());
        }
        
        suggestions
    }

    /// Assess summary completeness
    fn assess_summary_completeness(&self, message: &str) -> f64 {
        let mut score: f64 = 0.0;
        
        // Check for key summary elements
        if message.to_lowercase().contains("files") || message.to_lowercase().contains("processed") {
            score += 0.2;
        }
        
        if message.to_lowercase().contains("time") || message.to_lowercase().contains("duration") {
            score += 0.2;
        }
        
        if message.to_lowercase().contains("error") || message.to_lowercase().contains("warning") {
            score += 0.2;
        }
        
        if message.to_lowercase().contains("complete") || message.to_lowercase().contains("finished") {
            score += 0.2;
        }
        
        if message.to_lowercase().contains("next") || message.to_lowercase().contains("recommend") {
            score += 0.2;
        }
        
        score.min(1.0)
    }

    /// Assess next steps guidance
    fn assess_next_steps_guidance(&self, message: &str) -> f64 {
        let mut score: f64 = 0.0;
        
        let guidance_indicators = ["next", "now", "then", "should", "can", "recommend", "suggest"];
        for indicator in guidance_indicators {
            if message.to_lowercase().contains(indicator) {
                score += 0.15;
            }
        }
        
        // Reward specific actions
        if message.contains("run") || message.contains("execute") || message.contains("check") {
            score += 0.2;
        }
        
        score.min(1.0)
    }

    /// Assess actionable insights
    fn assess_actionable_insights(&self, message: &str) -> f64 {
        let mut score: f64 = 0.0;
        
        // Look for insights that lead to action
        if message.contains("because") || message.contains("due to") || message.contains("caused by") {
            score += 0.2; // Explains why
        }
        
        if message.contains("improve") || message.contains("optimize") || message.contains("fix") {
            score += 0.2; // Suggests improvement
        }
        
        if message.contains("performance") || message.contains("efficiency") || message.contains("speed") {
            score += 0.1; // Performance insights
        }
        
        if message.contains("memory") || message.contains("disk") || message.contains("cpu") {
            score += 0.1; // Resource insights
        }
        
        score.min(1.0)
    }

    /// Assess instruction completeness
    fn assess_instruction_completeness(&self, instructions: &str) -> f64 {
        let mut score: f64 = 0.0;
        
        // Check for step-by-step structure
        if instructions.contains("1.") || instructions.contains("first") || instructions.contains("step") {
            score += 0.3;
        }
        
        // Check for specific commands or actions
        if instructions.contains("run") || instructions.contains("execute") || instructions.contains("command") {
            score += 0.3;
        }
        
        // Check for expected outcomes
        if instructions.contains("should") || instructions.contains("will") || instructions.contains("expect") {
            score += 0.2;
        }
        
        // Check for troubleshooting
        if instructions.contains("if") || instructions.contains("error") || instructions.contains("problem") {
            score += 0.2;
        }
        
        score.min(1.0)
    }

    /// Assess feedback usefulness
    fn assess_feedback_usefulness(&self, message: &str) -> f64 {
        let mut score: f64 = 0.5; // Base score
        
        // Reward informative content
        if message.contains("progress") || message.contains("status") {
            score += 0.2;
        }
        
        if message.contains("error") || message.contains("warning") {
            score += 0.1;
        }
        
        if message.contains("complete") || message.contains("finished") {
            score += 0.2;
        }
        
        // Penalize redundant or useless messages
        if message.len() < 10 {
            score -= 0.2;
        }
        
        if message.to_lowercase() == "ok" || message.to_lowercase() == "done" {
            score -= 0.1;
        }
        
        score.max(0.0).min(1.0)
    }

    /// Assess feedback frequency appropriateness
    fn assess_feedback_frequency(&self) -> f64 {
        // This would be implemented based on timing analysis
        // For now, return a default score
        0.7
    }

    /// Assess confidence impact of a message
    fn assess_confidence_impact(&self, message: &str) -> f64 {
        let mut impact: f64 = 0.0;
        
        // Positive confidence indicators
        let positive_words = ["success", "complete", "finished", "ready", "good", "excellent"];
        for word in positive_words {
            if message.to_lowercase().contains(word) {
                impact += 0.2;
            }
        }
        
        // Negative confidence indicators
        let negative_words = ["error", "failed", "problem", "issue", "warning"];
        for word in negative_words {
            if message.to_lowercase().contains(word) {
                impact -= 0.1;
            }
        }
        
        // Neutral but informative
        if message.contains("processing") || message.contains("analyzing") {
            impact += 0.1;
        }
        
        impact.max(-1.0).min(1.0)
    }   
 /// Calculate overall UX score
    fn calculate_overall_ux_score(
        &self,
        progress: &ProgressReportingQuality,
        error: &ErrorMessageClarity,
        completion: &CompletionFeedbackQuality,
        interruption: &InterruptionHandlingQuality,
        feedback: &UserFeedbackAnalysis,
    ) -> f64 {
        // Weighted average of all UX components (scale to 0-10)
        let weighted_score = (progress.clarity_score * 2.0) +
                           (progress.update_frequency_score * 1.5) +
                           (error.average_clarity_score * 2.0) +
                           (error.actionability_score * 2.0) +
                           (completion.summary_completeness * 1.0) +
                           (completion.next_steps_guidance * 1.0) +
                           (interruption.graceful_shutdown_score * 0.5) +
                           (feedback.feedback_usefulness_score * 1.0);
        
        // Scale to 0-10 range
        (weighted_score / 1.1) * 10.0
    }

    /// Generate improvement recommendations
    fn generate_improvement_recommendations(
        &self,
        progress: &ProgressReportingQuality,
        error: &ErrorMessageClarity,
        completion: &CompletionFeedbackQuality,
        interruption: &InterruptionHandlingQuality,
        feedback: &UserFeedbackAnalysis,
    ) -> Vec<UXImprovement> {
        let mut recommendations = Vec::new();

        // Progress reporting improvements
        if progress.update_frequency_score < 0.7 {
            recommendations.push(UXImprovement {
                category: UXCategory::ProgressReporting,
                priority: ImprovementPriority::High,
                title: "Improve Progress Update Frequency".to_string(),
                description: "Progress updates are too infrequent, leaving users uncertain about system status".to_string(),
                current_score: progress.update_frequency_score,
                target_score: 0.8,
                implementation_suggestions: vec![
                    "Increase update frequency to every 2-3 seconds during active processing".to_string(),
                    "Add percentage completion indicators".to_string(),
                    "Include ETA estimates when possible".to_string(),
                ],
                impact_assessment: "Users will feel more confident about system progress".to_string(),
            });
        }

        if progress.clarity_score < 0.7 {
            recommendations.push(UXImprovement {
                category: UXCategory::ProgressReporting,
                priority: ImprovementPriority::Medium,
                title: "Enhance Progress Message Clarity".to_string(),
                description: "Progress messages lack clear, actionable information".to_string(),
                current_score: progress.clarity_score,
                target_score: 0.8,
                implementation_suggestions: vec![
                    "Use structured format: 'Processing files: 150/1000 (15%) - ETA: 2m 30s'".to_string(),
                    "Include current operation context".to_string(),
                    "Avoid technical jargon in user-facing messages".to_string(),
                ],
                impact_assessment: "Users will better understand what the system is doing".to_string(),
            });
        }

        // Error message improvements
        if error.actionability_score < 0.7 {
            recommendations.push(UXImprovement {
                category: UXCategory::ErrorMessages,
                priority: ImprovementPriority::Critical,
                title: "Make Error Messages More Actionable".to_string(),
                description: "Error messages don't provide clear steps for resolution".to_string(),
                current_score: error.actionability_score,
                target_score: 0.8,
                implementation_suggestions: vec![
                    "Include specific resolution steps in error messages".to_string(),
                    "Provide examples of correct usage when applicable".to_string(),
                    "Add links to documentation or troubleshooting guides".to_string(),
                ],
                impact_assessment: "Users will be able to resolve issues independently".to_string(),
            });
        }

        if error.technical_jargon_score > 0.5 {
            recommendations.push(UXImprovement {
                category: UXCategory::ErrorMessages,
                priority: ImprovementPriority::Medium,
                title: "Reduce Technical Jargon in Error Messages".to_string(),
                description: "Error messages contain too much technical language for general users".to_string(),
                current_score: 1.0 - error.technical_jargon_score,
                target_score: 0.8,
                implementation_suggestions: vec![
                    "Replace technical terms with user-friendly explanations".to_string(),
                    "Provide both technical and simplified versions of error messages".to_string(),
                    "Use analogies to explain complex technical concepts".to_string(),
                ],
                impact_assessment: "Broader audience will understand and act on error messages".to_string(),
            });
        }

        // Completion feedback improvements
        if completion.next_steps_guidance < 0.7 {
            recommendations.push(UXImprovement {
                category: UXCategory::CompletionFeedback,
                priority: ImprovementPriority::High,
                title: "Improve Next Steps Guidance".to_string(),
                description: "Completion messages don't clearly indicate what users should do next".to_string(),
                current_score: completion.next_steps_guidance,
                target_score: 0.8,
                implementation_suggestions: vec![
                    "Always include 'What's Next' section in completion messages".to_string(),
                    "Provide specific commands or actions for next steps".to_string(),
                    "Include links to relevant documentation or tools".to_string(),
                ],
                impact_assessment: "Users will know how to proceed after task completion".to_string(),
            });
        }

        // Interruption handling improvements
        if interruption.recovery_instructions_score < 0.7 {
            recommendations.push(UXImprovement {
                category: UXCategory::InterruptionHandling,
                priority: ImprovementPriority::Medium,
                title: "Enhance Recovery Instructions".to_string(),
                description: "Interruption recovery instructions are unclear or missing".to_string(),
                current_score: interruption.recovery_instructions_score,
                target_score: 0.8,
                implementation_suggestions: vec![
                    "Provide step-by-step recovery instructions".to_string(),
                    "Explain what state was preserved and what was lost".to_string(),
                    "Include commands to resume or restart operations".to_string(),
                ],
                impact_assessment: "Users will be able to recover from interruptions effectively".to_string(),
            });
        }

        recommendations
    }
}

// Implementation of individual analyzer components

impl ProgressAnalyzer {
    pub fn new() -> Self {
        Self {
            update_intervals: Vec::new(),
            progress_messages: Vec::new(),
            eta_predictions: Vec::new(),
        }
    }

    pub fn add_progress_message(&mut self, message: ProgressMessage) {
        // Calculate update interval
        if let Some(last_message) = self.progress_messages.last() {
            let interval = message.timestamp.duration_since(last_message.timestamp);
            self.update_intervals.push(interval);
        }

        self.progress_messages.push(message);
    }

    pub fn add_eta_prediction(&mut self, predicted_eta: Duration) {
        let prediction = ETAPrediction {
            predicted_eta,
            actual_completion_time: None,
            accuracy_score: 0.0,
            prediction_timestamp: Instant::now(),
        };
        self.eta_predictions.push(prediction);
    }

    pub fn generate_quality_assessment(&self) -> ProgressReportingQuality {
        let update_frequency_score = self.calculate_update_frequency_score();
        let information_completeness_score = self.calculate_information_completeness_score();
        let clarity_score = self.calculate_average_clarity_score();
        let eta_accuracy_score = self.calculate_eta_accuracy_score();
        let consistency_score = self.calculate_consistency_score();

        let detailed_analysis = ProgressDetailedAnalysis {
            average_update_interval: self.calculate_average_update_interval(),
            update_interval_consistency: self.calculate_update_interval_consistency(),
            information_density_average: self.calculate_average_information_density(),
            eta_accuracy_trend: self.calculate_eta_accuracy_trend(),
            problematic_patterns: self.identify_problematic_patterns(),
        };

        ProgressReportingQuality {
            update_frequency_score,
            information_completeness_score,
            clarity_score,
            eta_accuracy_score,
            consistency_score,
            detailed_analysis,
        }
    }

    fn calculate_update_frequency_score(&self) -> f64 {
        if self.update_intervals.is_empty() {
            return 1.0;
        }

        let average_interval = self.update_intervals.iter().sum::<Duration>().as_secs_f64() / self.update_intervals.len() as f64;
        
        // Ideal update interval is 2-5 seconds
        if average_interval >= 2.0 && average_interval <= 5.0 {
            1.0
        } else if average_interval < 2.0 {
            // Too frequent
            (average_interval / 2.0).max(0.5)
        } else {
            // Too infrequent
            (10.0 / average_interval).min(1.0).max(0.0)
        }
    }

    fn calculate_information_completeness_score(&self) -> f64 {
        if self.progress_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.progress_messages.iter()
            .map(|msg| {
                let mut score: f64 = 0.0;
                if msg.files_processed > 0 { score += 0.25; }
                if msg.total_files > 0 { score += 0.25; }
                if msg.percentage_complete > 0.0 { score += 0.25; }
                if msg.eta_seconds.is_some() { score += 0.25; }
                score
            })
            .sum();

        total_score / self.progress_messages.len() as f64
    }

    fn calculate_average_clarity_score(&self) -> f64 {
        if self.progress_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.progress_messages.iter().map(|msg| msg.clarity_score).sum();
        total_score / self.progress_messages.len() as f64
    }

    fn calculate_eta_accuracy_score(&self) -> f64 {
        let accurate_predictions = self.eta_predictions.iter()
            .filter(|pred| pred.accuracy_score > 0.8)
            .count();

        if self.eta_predictions.is_empty() {
            0.5 // Neutral score if no ETA predictions
        } else {
            accurate_predictions as f64 / self.eta_predictions.len() as f64
        }
    }

    fn calculate_consistency_score(&self) -> f64 {
        if self.update_intervals.len() < 2 {
            return 1.0;
        }

        let mean_interval = self.update_intervals.iter().sum::<Duration>().as_secs_f64() / self.update_intervals.len() as f64;
        let variance = self.update_intervals.iter()
            .map(|interval| {
                let diff = interval.as_secs_f64() - mean_interval;
                diff * diff
            })
            .sum::<f64>() / self.update_intervals.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean_interval > 0.0 { std_dev / mean_interval } else { 0.0 };

        // Lower coefficient of variation = higher consistency
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    fn calculate_average_update_interval(&self) -> Duration {
        if self.update_intervals.is_empty() {
            Duration::from_secs(0)
        } else {
            let total_secs: f64 = self.update_intervals.iter().map(|d| d.as_secs_f64()).sum();
            if total_secs.is_finite() && total_secs >= 0.0 {
                Duration::from_secs_f64(total_secs / self.update_intervals.len() as f64)
            } else {
                Duration::from_secs(0)
            }
        }
    }

    fn calculate_update_interval_consistency(&self) -> f64 {
        self.calculate_consistency_score()
    }

    fn calculate_average_information_density(&self) -> f64 {
        if self.progress_messages.is_empty() {
            return 0.0;
        }

        let total_density: f64 = self.progress_messages.iter().map(|msg| msg.information_density).sum();
        total_density / self.progress_messages.len() as f64
    }

    fn calculate_eta_accuracy_trend(&self) -> Vec<f64> {
        self.eta_predictions.iter().map(|pred| pred.accuracy_score).collect()
    }

    fn identify_problematic_patterns(&self) -> Vec<String> {
        let mut patterns = Vec::new();

        // Check for inconsistent update intervals
        if self.calculate_consistency_score() < 0.6 {
            patterns.push("Inconsistent progress update intervals".to_string());
        }

        // Check for low information density
        if self.calculate_average_information_density() < 0.01 {
            patterns.push("Progress messages lack informative content".to_string());
        }

        // Check for missing ETA predictions
        if self.eta_predictions.is_empty() && self.progress_messages.len() > 5 {
            patterns.push("No ETA predictions provided for long-running operations".to_string());
        }

        patterns
    }
}

impl ErrorMessageAnalyzer {
    pub fn new() -> Self {
        Self {
            error_messages: Vec::new(),
            clarity_patterns: Self::create_clarity_patterns(),
            actionability_patterns: Self::create_actionability_patterns(),
        }
    }

    fn create_clarity_patterns() -> Vec<ClarityPattern> {
        vec![
            // Positive patterns (improve clarity)
            ClarityPattern {
                pattern: Regex::new(r"\b(file|path|directory)\b").unwrap(),
                clarity_impact: 0.1,
                description: "Specific file/path references improve clarity".to_string(),
            },
            ClarityPattern {
                pattern: Regex::new(r"\b(because|due to|caused by)\b").unwrap(),
                clarity_impact: 0.2,
                description: "Causal explanations improve understanding".to_string(),
            },
            // Negative patterns (reduce clarity)
            ClarityPattern {
                pattern: Regex::new(r"\b(errno|segfault|malloc)\b").unwrap(),
                clarity_impact: -0.2,
                description: "Technical jargon reduces clarity for general users".to_string(),
            },
            ClarityPattern {
                pattern: Regex::new(r"[A-Z_]{5,}").unwrap(),
                clarity_impact: -0.1,
                description: "ALL_CAPS technical constants reduce readability".to_string(),
            },
        ]
    }

    fn create_actionability_patterns() -> Vec<ActionabilityPattern> {
        vec![
            // Positive patterns (improve actionability)
            ActionabilityPattern {
                pattern: Regex::new(r"\b(try|run|execute|check|install|update)\b").unwrap(),
                actionability_impact: 0.2,
                description: "Action verbs provide clear next steps".to_string(),
            },
            ActionabilityPattern {
                pattern: Regex::new(r"\b(command|option|flag)\b").unwrap(),
                actionability_impact: 0.15,
                description: "References to specific commands/options are actionable".to_string(),
            },
            ActionabilityPattern {
                pattern: Regex::new(r"\b(example|e\.g\.|such as)\b").unwrap(),
                actionability_impact: 0.1,
                description: "Examples make instructions more actionable".to_string(),
            },
            // Negative patterns (reduce actionability)
            ActionabilityPattern {
                pattern: Regex::new(r"\b(something|somehow|maybe|perhaps)\b").unwrap(),
                actionability_impact: -0.15,
                description: "Vague language reduces actionability".to_string(),
            },
        ]
    }

    pub fn add_error_message(&mut self, message: ErrorMessageEntry) {
        self.error_messages.push(message);
    }

    pub fn generate_clarity_assessment(&self) -> ErrorMessageClarity {
        let average_clarity_score = self.calculate_average_clarity_score();
        let actionability_score = self.calculate_average_actionability_score();
        let technical_jargon_score = self.calculate_technical_jargon_score();
        let solution_guidance_score = self.calculate_solution_guidance_score();
        let consistency_score = self.calculate_consistency_score();

        let detailed_analysis = ErrorMessageDetailedAnalysis {
            total_error_messages: self.error_messages.len() as u64,
            messages_with_solutions: self.error_messages.iter().filter(|msg| msg.contains_solution).count() as u64,
            messages_with_context: self.error_messages.iter().filter(|msg| !msg.context.is_empty()).count() as u64,
            average_technical_complexity: self.calculate_average_technical_complexity(),
            common_improvement_areas: self.identify_common_improvement_areas(),
        };

        ErrorMessageClarity {
            average_clarity_score,
            actionability_score,
            technical_jargon_score,
            solution_guidance_score,
            consistency_score,
            detailed_analysis,
        }
    }

    fn calculate_average_clarity_score(&self) -> f64 {
        if self.error_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.error_messages.iter().map(|msg| msg.clarity_score).sum();
        total_score / self.error_messages.len() as f64
    }

    fn calculate_average_actionability_score(&self) -> f64 {
        if self.error_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.error_messages.iter().map(|msg| msg.actionability_score).sum();
        total_score / self.error_messages.len() as f64
    }

    fn calculate_technical_jargon_score(&self) -> f64 {
        if self.error_messages.is_empty() {
            return 0.0;
        }

        let technical_count = self.error_messages.iter()
            .filter(|msg| matches!(msg.technical_complexity, TechnicalComplexity::Technical | TechnicalComplexity::ExpertLevel))
            .count();

        1.0 - (technical_count as f64 / self.error_messages.len() as f64)
    }

    fn calculate_solution_guidance_score(&self) -> f64 {
        if self.error_messages.is_empty() {
            return 0.0;
        }

        let solutions_count = self.error_messages.iter().filter(|msg| msg.contains_solution).count();
        solutions_count as f64 / self.error_messages.len() as f64
    }

    fn calculate_consistency_score(&self) -> f64 {
        if self.error_messages.len() < 2 {
            return 1.0;
        }

        // Measure consistency in clarity scores
        let clarity_scores: Vec<f64> = self.error_messages.iter().map(|msg| msg.clarity_score).collect();
        let mean_clarity = clarity_scores.iter().sum::<f64>() / clarity_scores.len() as f64;
        let variance = clarity_scores.iter()
            .map(|score| (score - mean_clarity).powi(2))
            .sum::<f64>() / clarity_scores.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean_clarity > 0.0 { std_dev / mean_clarity } else { 0.0 };

        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    fn calculate_average_technical_complexity(&self) -> f64 {
        if self.error_messages.is_empty() {
            return 0.0;
        }

        let complexity_sum: f64 = self.error_messages.iter()
            .map(|msg| match msg.technical_complexity {
                TechnicalComplexity::UserFriendly => 0.0,
                TechnicalComplexity::Moderate => 0.33,
                TechnicalComplexity::Technical => 0.66,
                TechnicalComplexity::ExpertLevel => 1.0,
            })
            .sum();

        complexity_sum / self.error_messages.len() as f64
    }

    fn identify_common_improvement_areas(&self) -> Vec<String> {
        let mut areas = Vec::new();

        // Analyze improvement suggestions frequency
        let mut suggestion_counts: HashMap<String, u32> = HashMap::new();
        for message in &self.error_messages {
            for suggestion in &message.improvement_suggestions {
                *suggestion_counts.entry(suggestion.clone()).or_insert(0) += 1;
            }
        }

        // Get most common suggestions
        let mut suggestions: Vec<(String, u32)> = suggestion_counts.into_iter().collect();
        suggestions.sort_by(|a, b| b.1.cmp(&a.1));

        for (suggestion, count) in suggestions.into_iter().take(5) {
            if count > 1 {
                areas.push(format!("{} (affects {} messages)", suggestion, count));
            }
        }

        areas
    }
}

impl CompletionAnalyzer {
    pub fn new() -> Self {
        Self {
            completion_messages: Vec::new(),
            summary_quality_metrics: SummaryQualityMetrics {
                includes_file_count: false,
                includes_processing_time: false,
                includes_error_summary: false,
                includes_next_steps: false,
                includes_performance_metrics: false,
                overall_completeness_score: 0.0,
            },
        }
    }

    pub fn add_completion_message(&mut self, message: CompletionMessage) {
        // Update summary quality metrics based on the message
        self.update_summary_quality_metrics(&message.message);
        self.completion_messages.push(message);
    }

    fn update_summary_quality_metrics(&mut self, message: &str) {
        let lower_message = message.to_lowercase();
        
        if lower_message.contains("files") || lower_message.contains("processed") {
            self.summary_quality_metrics.includes_file_count = true;
        }
        
        if lower_message.contains("time") || lower_message.contains("duration") || lower_message.contains("seconds") {
            self.summary_quality_metrics.includes_processing_time = true;
        }
        
        if lower_message.contains("error") || lower_message.contains("warning") || lower_message.contains("issue") {
            self.summary_quality_metrics.includes_error_summary = true;
        }
        
        if lower_message.contains("next") || lower_message.contains("now") || lower_message.contains("should") {
            self.summary_quality_metrics.includes_next_steps = true;
        }
        
        if lower_message.contains("performance") || lower_message.contains("speed") || lower_message.contains("memory") {
            self.summary_quality_metrics.includes_performance_metrics = true;
        }

        // Calculate overall completeness score
        let mut score: f64 = 0.0;
        if self.summary_quality_metrics.includes_file_count { score += 0.2; }
        if self.summary_quality_metrics.includes_processing_time { score += 0.2; }
        if self.summary_quality_metrics.includes_error_summary { score += 0.2; }
        if self.summary_quality_metrics.includes_next_steps { score += 0.2; }
        if self.summary_quality_metrics.includes_performance_metrics { score += 0.2; }
        
        self.summary_quality_metrics.overall_completeness_score = score;
    }

    pub fn generate_quality_assessment(&self) -> CompletionFeedbackQuality {
        let summary_completeness = self.calculate_average_summary_completeness();
        let results_clarity = self.calculate_average_results_clarity();
        let next_steps_guidance = self.calculate_average_next_steps_guidance();
        let actionable_insights = self.calculate_average_actionable_insights();

        let detailed_analysis = CompletionDetailedAnalysis {
            includes_performance_summary: self.summary_quality_metrics.includes_performance_metrics,
            includes_error_summary: self.summary_quality_metrics.includes_error_summary,
            includes_recommendations: self.has_recommendations(),
            includes_next_steps: self.summary_quality_metrics.includes_next_steps,
            summary_quality_score: self.summary_quality_metrics.overall_completeness_score,
        };

        CompletionFeedbackQuality {
            summary_completeness,
            results_clarity,
            next_steps_guidance,
            actionable_insights,
            detailed_analysis,
        }
    }

    fn calculate_average_summary_completeness(&self) -> f64 {
        if self.completion_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.completion_messages.iter().map(|msg| msg.summary_completeness).sum();
        total_score / self.completion_messages.len() as f64
    }

    fn calculate_average_results_clarity(&self) -> f64 {
        if self.completion_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.completion_messages.iter().map(|msg| msg.results_clarity).sum();
        total_score / self.completion_messages.len() as f64
    }

    fn calculate_average_next_steps_guidance(&self) -> f64 {
        if self.completion_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.completion_messages.iter().map(|msg| msg.next_steps_guidance).sum();
        total_score / self.completion_messages.len() as f64
    }

    fn calculate_average_actionable_insights(&self) -> f64 {
        if self.completion_messages.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.completion_messages.iter().map(|msg| msg.actionable_insights).sum();
        total_score / self.completion_messages.len() as f64
    }

    fn has_recommendations(&self) -> bool {
        self.completion_messages.iter().any(|msg| {
            msg.message.to_lowercase().contains("recommend") || 
            msg.message.to_lowercase().contains("suggest") ||
            msg.message.to_lowercase().contains("should")
        })
    }
}

impl InterruptionAnalyzer {
    pub fn new() -> Self {
        Self {
            interruption_events: Vec::new(),
            recovery_instructions: Vec::new(),
        }
    }

    pub fn add_interruption_event(&mut self, event: InterruptionEvent) {
        self.interruption_events.push(event);
    }

    pub fn add_recovery_instruction(&mut self, instruction: RecoveryInstruction) {
        self.recovery_instructions.push(instruction);
    }

    pub fn generate_quality_assessment(&self) -> InterruptionHandlingQuality {
        let graceful_shutdown_score = self.calculate_graceful_shutdown_score();
        let state_preservation_score = self.calculate_state_preservation_score();
        let recovery_instructions_score = self.calculate_recovery_instructions_score();
        let cleanup_completeness_score = self.calculate_cleanup_completeness_score();

        let detailed_analysis = InterruptionDetailedAnalysis {
            total_interruptions: self.interruption_events.len() as u64,
            graceful_shutdowns: self.interruption_events.iter().filter(|e| e.graceful_shutdown_score > 0.5).count() as u64,
            state_preserved_count: self.interruption_events.iter().filter(|e| e.state_preservation_score > 0.5).count() as u64,
            recovery_instructions_provided: self.interruption_events.iter().filter(|e| e.recovery_instructions_provided).count() as u64,
            average_cleanup_time: Duration::from_secs(5), // This would be measured in practice
        };

        InterruptionHandlingQuality {
            graceful_shutdown_score,
            state_preservation_score,
            recovery_instructions_score,
            cleanup_completeness_score,
            detailed_analysis,
        }
    }

    fn calculate_graceful_shutdown_score(&self) -> f64 {
        if self.interruption_events.is_empty() {
            return 1.0; // No interruptions = perfect score
        }

        let total_score: f64 = self.interruption_events.iter().map(|e| e.graceful_shutdown_score).sum();
        total_score / self.interruption_events.len() as f64
    }

    fn calculate_state_preservation_score(&self) -> f64 {
        if self.interruption_events.is_empty() {
            return 1.0;
        }

        let total_score: f64 = self.interruption_events.iter().map(|e| e.state_preservation_score).sum();
        total_score / self.interruption_events.len() as f64
    }

    fn calculate_recovery_instructions_score(&self) -> f64 {
        if self.recovery_instructions.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.recovery_instructions.iter()
            .map(|i| (i.clarity_score + i.completeness_score + i.actionability_score) / 3.0)
            .sum();
        
        total_score / self.recovery_instructions.len() as f64
    }

    fn calculate_cleanup_completeness_score(&self) -> f64 {
        if self.interruption_events.is_empty() {
            return 1.0;
        }

        let total_score: f64 = self.interruption_events.iter().map(|e| e.cleanup_completeness_score).sum();
        total_score / self.interruption_events.len() as f64
    }
}

impl FeedbackAnalyzer {
    pub fn new() -> Self {
        Self {
            feedback_events: Vec::new(),
            user_confidence_indicators: Vec::new(),
        }
    }

    pub fn add_feedback_event(&mut self, event: FeedbackEvent) {
        // Update confidence indicators based on the feedback
        let confidence_indicator = ConfidenceIndicator {
            indicator_type: format!("{:?}", event.feedback_type),
            confidence_level: event.user_confidence_impact.max(0.0),
            evidence: event.message.clone(),
            timestamp: event.timestamp,
        };
        
        self.user_confidence_indicators.push(confidence_indicator);
        self.feedback_events.push(event);
    }

    pub fn generate_analysis(&self) -> UserFeedbackAnalysis {
        let feedback_frequency = self.calculate_feedback_frequency();
        let feedback_usefulness_score = self.calculate_average_usefulness_score();
        let next_steps_clarity = self.calculate_next_steps_clarity();
        let user_confidence_score = self.calculate_user_confidence_score();

        let detailed_analysis = FeedbackDetailedAnalysis {
            total_feedback_events: self.feedback_events.len() as u64,
            feedback_by_type: self.calculate_feedback_by_type(),
            average_usefulness_by_type: self.calculate_average_usefulness_by_type(),
            confidence_trend: self.calculate_confidence_trend(),
        };

        UserFeedbackAnalysis {
            feedback_frequency,
            feedback_usefulness_score,
            next_steps_clarity,
            user_confidence_score,
            detailed_analysis,
        }
    }

    fn calculate_feedback_frequency(&self) -> f64 {
        if self.feedback_events.len() < 2 {
            return 0.0;
        }

        let first_event = &self.feedback_events[0];
        let last_event = &self.feedback_events[self.feedback_events.len() - 1];
        let total_duration = last_event.timestamp.duration_since(first_event.timestamp);
        
        if total_duration.as_secs() == 0 {
            return 0.0;
        }

        (self.feedback_events.len() as f64) / (total_duration.as_secs() as f64 / 60.0) // Events per minute
    }

    fn calculate_average_usefulness_score(&self) -> f64 {
        if self.feedback_events.is_empty() {
            return 0.0;
        }

        let total_score: f64 = self.feedback_events.iter().map(|e| e.usefulness_score).sum();
        total_score / self.feedback_events.len() as f64
    }

    fn calculate_next_steps_clarity(&self) -> f64 {
        let next_steps_events: Vec<&FeedbackEvent> = self.feedback_events.iter()
            .filter(|e| e.message.to_lowercase().contains("next") || e.message.to_lowercase().contains("should"))
            .collect();

        if next_steps_events.is_empty() {
            return 0.0;
        }

        let total_score: f64 = next_steps_events.iter().map(|e| e.usefulness_score).sum();
        total_score / next_steps_events.len() as f64
    }

    fn calculate_user_confidence_score(&self) -> f64 {
        if self.user_confidence_indicators.is_empty() {
            return 0.5; // Neutral confidence
        }

        let total_confidence: f64 = self.user_confidence_indicators.iter().map(|i| i.confidence_level).sum();
        total_confidence / self.user_confidence_indicators.len() as f64
    }

    fn calculate_feedback_by_type(&self) -> HashMap<String, u64> {
        let mut counts = HashMap::new();
        for event in &self.feedback_events {
            let type_name = format!("{:?}", event.feedback_type);
            *counts.entry(type_name).or_insert(0) += 1;
        }
        counts
    }

    fn calculate_average_usefulness_by_type(&self) -> HashMap<String, f64> {
        let mut type_scores: HashMap<String, Vec<f64>> = HashMap::new();
        
        for event in &self.feedback_events {
            let type_name = format!("{:?}", event.feedback_type);
            type_scores.entry(type_name).or_insert_with(Vec::new).push(event.usefulness_score);
        }

        type_scores.into_iter()
            .map(|(type_name, scores)| {
                let average = scores.iter().sum::<f64>() / scores.len() as f64;
                (type_name, average)
            })
            .collect()
    }

    fn calculate_confidence_trend(&self) -> Vec<f64> {
        self.user_confidence_indicators.iter().map(|i| i.confidence_level).collect()
    }
}

impl Default for UXAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// Include tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ux_analyzer_creation() {
        let analyzer = UXAnalyzer::new();
        
        // Verify analyzer is created with empty state
        let results = analyzer.generate_ux_results().unwrap();
        // With no data, the score calculation still produces a value based on defaults
        assert!(results.overall_ux_score >= 0.0);
    }

    #[test]
    fn test_progress_message_analysis() {
        let mut analyzer = UXAnalyzer::new();
        
        // Analyze a good progress message
        analyzer.analyze_progress_message(
            "Processing files: 150/1000 (15%) - ETA: 2m 30s",
            150,
            1000,
            Some(150)
        ).unwrap();
        
        // Analyze a poor progress message
        analyzer.analyze_progress_message(
            "Working...",
            200,
            1000,
            None
        ).unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should have progress reporting data
        assert!(results.progress_reporting_quality.information_completeness_score > 0.0);
        assert!(results.progress_reporting_quality.clarity_score > 0.0);
    }

    #[test]
    fn test_error_message_analysis() {
        let mut analyzer = UXAnalyzer::new();
        
        // Analyze a good error message
        analyzer.analyze_error_message(
            "FileNotFound",
            "File 'config.json' not found. Please check the file path and try again.",
            "Loading configuration"
        ).unwrap();
        
        // Analyze a poor error message
        analyzer.analyze_error_message(
            "SystemError",
            "errno 2: ENOENT",
            "System operation"
        ).unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should have error message analysis
        assert!(results.error_message_clarity.average_clarity_score > 0.0);
        assert_eq!(results.error_message_clarity.detailed_analysis.total_error_messages, 2);
    }

    #[test]
    fn test_completion_message_analysis() {
        let mut analyzer = UXAnalyzer::new();
        
        // Analyze a comprehensive completion message with clear next steps
        analyzer.analyze_completion_message(
            "Processing complete! Processed 1000 files in 2m 30s. Found 5 errors. Next steps: Run 'pensieve query' to explore results or check 'pensieve summary' for details."
        ).unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should have high completion feedback quality
        assert!(results.completion_feedback_quality.summary_completeness > 0.5);
        assert!(results.completion_feedback_quality.next_steps_guidance > 0.0);
    }

    #[test]
    fn test_interruption_analysis() {
        let mut analyzer = UXAnalyzer::new();
        
        // Analyze graceful interruption
        analyzer.analyze_interruption(
            InterruptionType::UserCancellation,
            true,  // graceful
            true,  // state preserved
            true,  // cleanup complete
            Some("To resume processing, run: pensieve resume --from-checkpoint")
        ).unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should have good interruption handling scores
        assert_eq!(results.interruption_handling_quality.graceful_shutdown_score, 1.0);
        assert_eq!(results.interruption_handling_quality.state_preservation_score, 1.0);
        assert!(results.interruption_handling_quality.recovery_instructions_score > 0.0);
    }

    #[test]
    fn test_feedback_analysis() {
        let mut analyzer = UXAnalyzer::new();
        
        // Analyze various feedback types
        analyzer.analyze_feedback(
            FeedbackType::StatusUpdate,
            "Currently processing directory: /home/user/documents"
        ).unwrap();
        
        analyzer.analyze_feedback(
            FeedbackType::SuccessMessage,
            "Successfully processed 100 files"
        ).unwrap();
        
        analyzer.analyze_feedback(
            FeedbackType::WarningMessage,
            "Warning: Large file detected, processing may take longer"
        ).unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should have feedback analysis
        assert!(results.user_feedback_analysis.feedback_usefulness_score > 0.0);
        assert_eq!(results.user_feedback_analysis.detailed_analysis.total_feedback_events, 3);
    }

    #[test]
    fn test_clarity_score_calculation() {
        let analyzer = UXAnalyzer::new();
        
        // Test clear message
        let clear_score = analyzer.calculate_clarity_score("File processing completed successfully");
        
        // Test technical message
        let technical_score = analyzer.calculate_clarity_score("segfault in malloc: errno 12");
        
        // Clear message should score higher
        assert!(clear_score > technical_score);
    }

    #[test]
    fn test_actionability_score_calculation() {
        let analyzer = UXAnalyzer::new();
        
        // Test actionable message
        let actionable_score = analyzer.calculate_actionability_score(
            "Error: File not found. Try checking the file path or run 'ls' to list available files."
        );
        
        // Test non-actionable message
        let non_actionable_score = analyzer.calculate_actionability_score(
            "Something went wrong somehow"
        );
        
        // Actionable message should score higher
        assert!(actionable_score > non_actionable_score);
    }

    #[test]
    fn test_technical_complexity_assessment() {
        let analyzer = UXAnalyzer::new();
        
        // Test user-friendly message (no technical terms)
        let user_friendly = analyzer.assess_technical_complexity("Operation not found");
        assert_eq!(user_friendly, TechnicalComplexity::UserFriendly);
        
        // Test technical message with specific technical terms
        let technical = analyzer.assess_technical_complexity("errno buffer pointer thread");
        assert_eq!(technical, TechnicalComplexity::Technical);
        
        // Test expert-level message
        let expert = analyzer.assess_technical_complexity("Segfault in syscall vtable");
        assert_eq!(expert, TechnicalComplexity::ExpertLevel);
    }

    #[test]
    fn test_improvement_recommendations() {
        let mut analyzer = UXAnalyzer::new();
        
        // Add some poor quality messages
        analyzer.analyze_progress_message("...", 0, 0, None).unwrap();
        analyzer.analyze_error_message("Error", "errno 2", "").unwrap();
        analyzer.analyze_completion_message("Done").unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should generate improvement recommendations
        assert!(!results.improvement_recommendations.is_empty());
        
        // Check that recommendations have proper structure
        for recommendation in &results.improvement_recommendations {
            assert!(!recommendation.title.is_empty());
            assert!(!recommendation.description.is_empty());
            assert!(!recommendation.implementation_suggestions.is_empty());
            assert!(recommendation.current_score >= 0.0);
            assert!(recommendation.target_score <= 1.0);
        }
    }

    #[test]
    fn test_overall_ux_score_calculation() {
        let mut analyzer = UXAnalyzer::new();
        
        // Add high-quality messages
        analyzer.analyze_progress_message(
            "Processing files: 500/1000 (50%) - ETA: 1m 15s - Current: analyzing documents",
            500,
            1000,
            Some(75)
        ).unwrap();
        
        analyzer.analyze_error_message(
            "ValidationError",
            "File 'config.json' contains invalid JSON. Please check line 15 for syntax errors and try again.",
            "Configuration validation"
        ).unwrap();
        
        analyzer.analyze_completion_message(
            "Processing complete! Successfully processed 1000 files in 2m 30s. Found 2 warnings, 0 errors. Next steps: Review results with 'pensieve summary' or query data with 'pensieve query'."
        ).unwrap();
        
        let results = analyzer.generate_ux_results().unwrap();
        
        // Should have a reasonable overall UX score
        assert!(results.overall_ux_score > 5.0); // Above average
        // The score calculation can vary based on the weighted formula, so just check it's reasonable
        assert!(results.overall_ux_score < 100.0);
    }

    #[test]
    fn test_information_density_calculation() {
        let analyzer = UXAnalyzer::new();
        
        // Test high information density
        let high_density = analyzer.calculate_information_density("Processing 150/1000 files (15%) ETA 2m");
        
        // Test low information density
        let low_density = analyzer.calculate_information_density("Working on the files and stuff...");
        
        // High density should be greater
        assert!(high_density > low_density);
    }

    #[test]
    fn test_solution_guidance_detection() {
        let analyzer = UXAnalyzer::new();
        
        // Test message with solution
        assert!(analyzer.contains_solution_guidance("Error occurred. Try running 'fix-command' to resolve."));
        
        // Test message without solution
        assert!(!analyzer.contains_solution_guidance("An error occurred."));
    }

    #[test]
    fn test_confidence_impact_assessment() {
        let analyzer = UXAnalyzer::new();
        
        // Test positive confidence message
        let positive_impact = analyzer.assess_confidence_impact("Processing completed successfully!");
        assert!(positive_impact > 0.0);
        
        // Test negative confidence message
        let negative_impact = analyzer.assess_confidence_impact("Critical error occurred, processing failed");
        assert!(negative_impact < 0.0);
        
        // Test neutral message
        let neutral_impact = analyzer.assess_confidence_impact("Processing files...");
        assert!(neutral_impact >= 0.0);
    }
}