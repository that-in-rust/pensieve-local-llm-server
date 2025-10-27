//! Pensieve Engine - Core inference engine
//!
//! This is the Layer 2 (L2) engine crate that provides:
//! - Inference engine traits and interfaces
//! - CPU-based model execution
//! - Token generation and sampling
//! - Resource management for computation
//!
//! Depends only on L1 (pensieve-07_core) crate.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

// Re-export from core
pub use pensieve_07_core::{
    error::{CoreError, CoreResult},
    traits::{Resource, Reset, Validate},
    Result,
};

/// Core inference engine traits
pub mod engine {
    use super::{CoreError, CoreResult, Resource, Reset, Validate};

    /// Trait for inference engines that can process models
    pub trait InferenceEngine: Resource + Reset + Validate {
        /// Input token type
        type Token: Copy + Clone + PartialEq;

        /// Output token type (usually same as input)
        type OutputToken: Copy + Clone + PartialEq;

        /// Context for the inference session
        type Context: Reset + Validate;

        /// Create a new inference context
        fn create_context(&mut self) -> CoreResult<Self::Context>;

        /// Process a single forward pass through the model
        fn forward(
            &mut self,
            context: &mut Self::Context,
            input_tokens: &[Self::Token],
        ) -> CoreResult<Self::OutputToken>;

        /// Generate the next token with sampling
        fn generate_next(
            &mut self,
            context: &mut Self::Context,
            input_tokens: &[Self::Token],
            temperature: f32,
        ) -> CoreResult<Self::OutputToken>;

        /// Check if the engine is ready for inference
        fn is_ready(&self) -> bool {
            self.is_available()
        }
    }

    /// Trait for token sampling strategies
    pub trait Sampler {
        /// Token type
        type Token: Copy + Clone;

        /// Sample the next token from logits
        fn sample(&self, logits: &[f32], temperature: f32) -> Self::Token;

        /// Validate logits input
        fn validate_logits(&self, logits: &[f32]) -> CoreResult<()> {
            if logits.is_empty() {
                return Err(CoreError::InvalidInput("empty logits"));
            }
            Ok(())
        }
    }
}

/// CPU-based inference implementation
pub mod cpu {
    use super::{
        engine::{InferenceEngine, Sampler},
        CoreError, CoreResult, Resource, Reset, Validate,
    };

    /// Simple CPU inference engine implementation
    #[derive(Debug)]
    pub struct CpuEngine {
        vocab_size: usize,
        ready: bool,
    }

    impl CpuEngine {
        /// Create a new CPU engine with specified vocabulary size
        pub fn new(vocab_size: usize) -> Self {
            Self {
                vocab_size,
                ready: false,
            }
        }

        /// Get the vocabulary size
        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    impl Resource for CpuEngine {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.ready
        }

        fn acquire(&mut self) -> CoreResult<()> {
            self.ready = true;
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            self.ready = false;
            Ok(())
        }
    }

    impl Reset for CpuEngine {
        fn reset(&mut self) {
            self.ready = false;
        }
    }

    impl Validate for CpuEngine {
        fn validate(&self) -> CoreResult<()> {
            if self.vocab_size == 0 {
                return Err(CoreError::InvalidConfig("vocab_size must be > 0"));
            }
            Ok(())
        }
    }

    /// Simple CPU inference context
    #[derive(Debug)]
    pub struct CpuContext {
        pub position: usize,
        pub ready: bool,
    }

    impl CpuContext {
        /// Create a new CPU context
        pub fn new() -> Self {
            Self {
                position: 0,
                ready: false,
            }
        }

        /// Get current position
        pub fn position(&self) -> usize {
            self.position
        }
    }

    impl Reset for CpuContext {
        fn reset(&mut self) {
            self.position = 0;
            self.ready = false;
        }
    }

    impl Validate for CpuContext {
        fn validate(&self) -> CoreResult<()> {
            if !self.ready {
                return Err(CoreError::Unavailable("context not initialized"));
            }
            Ok(())
        }
    }

    // Simple token type for CPU engine
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CpuToken(pub u32);

    // Simple sampler implementation
    #[derive(Debug)]
    pub struct CpuSampler {
        temperature: f32,
    }

    impl CpuSampler {
        /// Create a new CPU sampler
        pub fn new(temperature: f32) -> Self {
            Self { temperature }
        }
    }

    impl Sampler for CpuSampler {
        type Token = CpuToken;

        fn sample(&self, logits: &[f32], temperature: f32) -> Self::Token {
            self.validate_logits(logits).expect("Invalid logits");

            // Simple greedy sampling - just pick the max token
            let max_index = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            CpuToken(max_index as u32)
        }
    }

    // Basic implementation for InferenceEngine trait
    impl InferenceEngine for CpuEngine {
        type Token = CpuToken;
        type OutputToken = CpuToken;
        type Context = CpuContext;

        fn create_context(&mut self) -> CoreResult<Self::Context> {
            if !self.is_available() {
                return Err(CoreError::Unavailable("engine not ready"));
            }

            let mut context = CpuContext::new();
            context.ready = true;
            Ok(context)
        }

        fn forward(
            &mut self,
            context: &mut Self::Context,
            input_tokens: &[Self::Token],
        ) -> CoreResult<Self::OutputToken> {
            context.validate()?;

            if input_tokens.is_empty() {
                return Err(CoreError::InvalidInput("no input tokens"));
            }

            // Mock forward pass - in real implementation this would run the model
            context.position += input_tokens.len();

            // Return the last input token as output (mock)
            Ok(*input_tokens.last().unwrap())
        }

        fn generate_next(
            &mut self,
            context: &mut Self::Context,
            input_tokens: &[Self::Token],
            temperature: f32,
        ) -> CoreResult<Self::OutputToken> {
            context.validate()?;

            // For simplicity, just return the last input token as mock generation
            // In a real implementation, this would do proper logits computation
            if input_tokens.is_empty() {
                return Err(CoreError::InvalidInput("no input tokens"));
            }

            Ok(*input_tokens.last().unwrap())
        }
    }
}

// Re-export key types for convenience
pub use cpu::{CpuEngine, CpuSampler, CpuToken};
pub use engine::{InferenceEngine, Sampler};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuContext;

    #[test]
    fn test_cpu_engine_creation() {
        let engine = CpuEngine::new(1000);
        assert_eq!(engine.vocab_size(), 1000);
        assert!(!engine.is_available());
    }

    #[test]
    fn test_cpu_engine_resource_management() {
        let mut engine = CpuEngine::new(1000);

        // Test acquisition
        engine.acquire().unwrap();
        assert!(engine.is_available());

        // Test release
        engine.release().unwrap();
        assert!(!engine.is_available());
    }

    #[test]
    fn test_cpu_engine_validation() {
        let valid_engine = CpuEngine::new(1000);
        assert!(valid_engine.validate().is_ok());

        let invalid_engine = CpuEngine::new(0);
        assert!(invalid_engine.validate().is_err());
    }

    #[test]
    fn test_cpu_context() {
        let mut context = CpuContext::new();
        assert!(!context.ready);

        context.ready = true;
        assert!(context.validate().is_ok());

        context.reset();
        assert!(!context.ready);
    }

    #[test]
    fn test_cpu_sampler() {
        let sampler = CpuSampler::new(0.8);
        let logits = [0.1, 0.2, 0.7, 0.0];

        let token = sampler.sample(&logits, 0.8);
        assert_eq!(token, CpuToken(2)); // Should pick the max value
    }

    #[test]
    fn test_inference_engine_basic() {
        let mut engine = CpuEngine::new(1000);
        engine.acquire().unwrap();

        let mut context = engine.create_context().unwrap();
        let input_tokens = [CpuToken(1), CpuToken(2), CpuToken(3)];

        let output = engine.forward(&mut context, &input_tokens).unwrap();
        assert_eq!(output, CpuToken(3)); // Should return last input token
        assert_eq!(context.position(), 3);
    }

    #[test]
    fn test_inference_engine_generation() {
        let mut engine = CpuEngine::new(1000);
        engine.acquire().unwrap();

        let mut context = engine.create_context().unwrap();
        let input_tokens = [CpuToken(42)];

        let output = engine.generate_next(&mut context, &input_tokens, 0.8).unwrap();
        // Output should be a valid token (in this case, token 42 due to mock logic)
        assert_eq!(output, CpuToken(42));
    }
}