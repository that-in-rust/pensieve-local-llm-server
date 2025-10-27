//! Pensieve Models - Data models and GGUF interfaces
//!
//! This is the Layer 2 (L2) models crate that provides:
//! - Data model interfaces and abstractions
//! - GGUF file format parsing and handling
//! - Model metadata management
//! - Model validation and loading interfaces
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

// Import alloc types for no_std compatibility
extern crate alloc;

/// Core model traits and interfaces
pub mod model {
    use super::{CoreResult, Resource, Reset, Validate};
    

    /// Trait for model metadata
    pub trait ModelMetadata: Validate {
        /// Get model name
        fn name(&self) -> &str;

        /// Get model version
        fn version(&self) -> &str;

        /// Get model architecture
        fn architecture(&self) -> &str;

        /// Get vocabulary size
        fn vocab_size(&self) -> usize;

        /// Get context size
        fn context_size(&self) -> usize;

        /// Check if model supports inference
        fn supports_inference(&self) -> bool;
    }

    /// Trait for model loaders
    pub trait ModelLoader: Resource {
        /// Model type that this loader produces
        type Model: Resource + Reset + Validate;

        /// Metadata type for this loader
        type Metadata: ModelMetadata;

        /// Load model metadata from path
        fn load_metadata(&mut self, path: &str) -> CoreResult<Self::Metadata>;

        /// Load full model from path
        fn load_model(&mut self, path: &str) -> CoreResult<Self::Model>;

        /// Validate model file before loading
        fn validate_model_file(&self, path: &str) -> CoreResult<()>;
    }

    /// Trait for model validators
    pub trait ModelValidator {
        /// Validate model structure
        fn validate_structure(&self, model_data: &[u8]) -> CoreResult<()>;

        /// Validate model metadata consistency
        fn validate_metadata_consistency<M: ModelMetadata>(&self, metadata: &M) -> CoreResult<()>;

        /// Check model compatibility
        fn check_compatibility(&self, requirements: &ModelRequirements) -> CoreResult<bool>;
    }

    /// Model requirements for compatibility checking
    #[derive(Debug, Clone)]
    pub struct ModelRequirements {
        /// Minimum vocabulary size
        pub min_vocab_size: usize,
        /// Minimum context size
        pub min_context_size: usize,
        /// Required architectures
        pub supported_architectures: &'static [&'static str],
        /// Whether inference support is required
        pub requires_inference: bool,
    }

    impl ModelRequirements {
        /// Create basic model requirements
        pub fn new(
            min_vocab_size: usize,
            min_context_size: usize,
            supported_architectures: &'static [&'static str],
        ) -> Self {
            Self {
                min_vocab_size,
                min_context_size,
                supported_architectures,
                requires_inference: true,
            }
        }

        /// Create requirements without inference support
        pub fn without_inference(
            min_vocab_size: usize,
            min_context_size: usize,
            supported_architectures: &'static [&'static str],
        ) -> Self {
            Self {
                min_vocab_size,
                min_context_size,
                supported_architectures,
                requires_inference: false,
            }
        }
    }
}

/// GGUF format handling
pub mod gguf {
    use super::{CoreError, CoreResult, Validate};
    use crate::model::ModelMetadata;
    use alloc::{
        string::String,
        vec::Vec
    };

    /// GGUF file header information
    #[derive(Debug, Clone)]
    pub struct GgufHeader {
        /// Magic number identifying GGUF format
        pub magic: u32,
        /// Version of GGUF format
        pub version: u32,
        /// Number of tensors in the model
        pub tensor_count: u32,
        /// Number of key-value pairs
        pub kv_count: u32,
    }

    impl GgufHeader {
        /// Create a new GGUF header
        pub fn new(magic: u32, version: u32, tensor_count: u32, kv_count: u32) -> Self {
            Self {
                magic,
                version,
                tensor_count,
                kv_count,
            }
        }

        /// Get GGUF magic number
        pub const fn gguf_magic() -> u32 {
            0x46554747 // "GGUF" in little endian
        }

        /// Validate GGUF magic number
        pub fn validate_magic(&self) -> CoreResult<()> {
            if self.magic != Self::gguf_magic() {
                return Err(CoreError::InvalidInput("invalid GGUF magic number"));
            }
            Ok(())
        }
    }

    impl Validate for GgufHeader {
        fn validate(&self) -> CoreResult<()> {
            self.validate_magic()?;
            if self.version == 0 {
                return Err(CoreError::InvalidInput("invalid GGUF version"));
            }
            Ok(())
        }
    }

    /// GGUF tensor information
    #[derive(Debug, Clone)]
    pub struct GgufTensor {
        /// Tensor name
        pub name: String,
        /// Tensor dimensions
        pub dimensions: Vec<usize>,
        /// Tensor data type
        pub data_type: GgufDataType,
        /// Offset in file where tensor data starts
        pub offset: u64,
    }

    impl GgufTensor {
        /// Create a new GGUF tensor
        pub fn new(name: String, dimensions: Vec<usize>, data_type: GgufDataType, offset: u64) -> Self {
            Self {
                name,
                dimensions,
                data_type,
                offset,
            }
        }

        /// Get total number of elements in tensor
        pub fn element_count(&self) -> usize {
            self.dimensions.iter().product()
        }

        /// Validate tensor dimensions
        pub fn validate_dimensions(&self) -> CoreResult<()> {
            if self.dimensions.is_empty() {
                return Err(CoreError::InvalidInput("tensor dimensions cannot be empty"));
            }
            if self.dimensions.iter().any(|&dim| dim == 0) {
                return Err(CoreError::InvalidInput("tensor dimensions cannot be zero"));
            }
            Ok(())
        }
    }

    /// GGUF data types
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GgufDataType {
        /// 8-bit unsigned integer
        U8,
        /// 8-bit signed integer
        I8,
        /// 16-bit unsigned integer
        U16,
        /// 16-bit signed integer
        I16,
        /// 32-bit unsigned integer
        U32,
        /// 32-bit signed integer
        I32,
        /// 32-bit float
        F32,
        /// 64-bit float
        F64,
    }

    impl GgufDataType {
        /// Get size of this data type in bytes
        pub const fn size(&self) -> usize {
            match self {
                GgufDataType::U8 | GgufDataType::I8 => 1,
                GgufDataType::U16 | GgufDataType::I16 => 2,
                GgufDataType::U32 | GgufDataType::I32 | GgufDataType::F32 => 4,
                GgufDataType::F64 => 8,
            }
        }
    }

    /// GGUF model metadata implementation
    #[derive(Debug, Clone)]
    pub struct GgufMetadata {
        /// Model name
        pub name: String,
        /// Model version
        pub version: String,
        /// Model architecture
        pub architecture: String,
        /// Vocabulary size
        pub vocab_size: usize,
        /// Context size
        pub context_size: usize,
        /// Whether model supports inference
        pub supports_inference: bool,
    }

    impl GgufMetadata {
        /// Create new GGUF metadata
        pub fn new(
            name: String,
            version: String,
            architecture: String,
            vocab_size: usize,
            context_size: usize,
        ) -> Self {
            Self {
                name,
                version,
                architecture,
                vocab_size,
                context_size,
                supports_inference: true,
            }
        }

        /// Create metadata without inference support
        pub fn without_inference(
            name: String,
            version: String,
            architecture: String,
            vocab_size: usize,
            context_size: usize,
        ) -> Self {
            Self {
                name,
                version,
                architecture,
                vocab_size,
                context_size,
                supports_inference: false,
            }
        }
    }

    impl ModelMetadata for GgufMetadata {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            &self.version
        }

        fn architecture(&self) -> &str {
            &self.architecture
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn context_size(&self) -> usize {
            self.context_size
        }

        fn supports_inference(&self) -> bool {
            self.supports_inference
        }
    }

    impl Validate for GgufMetadata {
        fn validate(&self) -> CoreResult<()> {
            if self.name.is_empty() {
                return Err(CoreError::InvalidInput("model name cannot be empty"));
            }
            if self.version.is_empty() {
                return Err(CoreError::InvalidInput("model version cannot be empty"));
            }
            if self.architecture.is_empty() {
                return Err(CoreError::InvalidInput("model architecture cannot be empty"));
            }
            if self.vocab_size == 0 {
                return Err(CoreError::InvalidInput("vocabulary size must be > 0"));
            }
            if self.context_size == 0 {
                return Err(CoreError::InvalidInput("context size must be > 0"));
            }
            Ok(())
        }
    }
}

/// Simple model implementations for testing
pub mod mock {
    use super::{
        model::ModelLoader,
        gguf::GgufMetadata,
        CoreError, CoreResult, Resource, Reset, Validate,
    };
    use alloc::{
        format,
        string::{String, ToString}
    };

    /// Mock model for testing
    #[derive(Debug)]
    pub struct MockModel {
        pub name: String,
        pub loaded: bool,
    }

    impl MockModel {
        /// Create a new mock model
        pub fn new(name: String) -> Self {
            Self {
                name,
                loaded: false,
            }
        }
    }

    impl Resource for MockModel {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.loaded
        }

        fn acquire(&mut self) -> CoreResult<()> {
            self.loaded = true;
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            self.loaded = false;
            Ok(())
        }
    }

    impl Reset for MockModel {
        fn reset(&mut self) {
            self.loaded = false;
        }
    }

    impl Validate for MockModel {
        fn validate(&self) -> CoreResult<()> {
            if self.name.is_empty() {
                return Err(CoreError::InvalidInput("model name cannot be empty"));
            }
            Ok(())
        }
    }

    /// Mock model loader for testing
    #[derive(Debug)]
    pub struct MockModelLoader {
        available: bool,
    }

    impl MockModelLoader {
        /// Create a new mock model loader
        pub fn new() -> Self {
            Self { available: false }
        }
    }

    impl Resource for MockModelLoader {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.available
        }

        fn acquire(&mut self) -> CoreResult<()> {
            self.available = true;
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            self.available = false;
            Ok(())
        }
    }

    impl ModelLoader for MockModelLoader {
        type Model = MockModel;
        type Metadata = GgufMetadata;

        fn load_metadata(&mut self, path: &str) -> CoreResult<Self::Metadata> {
            if !self.is_available() {
                return Err(CoreError::Unavailable("loader not available"));
            }

            // Mock metadata loading
            let metadata = GgufMetadata::new(
                format!("model_from_{}", path),
                "1.0.0".to_string(),
                "llama".to_string(),
                32000,
                2048,
            );
            Ok(metadata)
        }

        fn load_model(&mut self, path: &str) -> CoreResult<Self::Model> {
            if !self.is_available() {
                return Err(CoreError::Unavailable("loader not available"));
            }

            // Mock model loading
            let model = MockModel::new(format!("model_from_{}", path));
            Ok(model)
        }

        fn validate_model_file(&self, path: &str) -> CoreResult<()> {
            if path.is_empty() {
                return Err(CoreError::InvalidInput("path cannot be empty"));
            }
            Ok(())
        }
    }

    impl Default for MockModelLoader {
        fn default() -> Self {
            Self::new()
        }
    }
}

// Re-export key types for convenience
pub use model::{ModelLoader, ModelMetadata, ModelRequirements, ModelValidator};
pub use gguf::{GgufDataType, GgufHeader, GgufMetadata, GgufTensor};
pub use mock::{MockModel, MockModelLoader};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{
        borrow::ToOwned,
        format,
        string::{String, ToString},
        vec::Vec,
        vec
    };

    #[test]
    fn test_model_requirements_creation() {
        let reqs = ModelRequirements::new(1000, 512, &["llama", "bert"]);
        assert_eq!(reqs.min_vocab_size, 1000);
        assert_eq!(reqs.min_context_size, 512);
        assert_eq!(reqs.supported_architectures.len(), 2);
        assert!(reqs.requires_inference);
    }

    #[test]
    fn test_model_requirements_without_inference() {
        let reqs = ModelRequirements::without_inference(1000, 512, &["llama"]);
        assert!(!reqs.requires_inference);
    }

    #[test]
    fn test_gguf_header_validation() {
        let valid_header = GgufHeader::new(
            GgufHeader::gguf_magic(),
            1,
            10,
            5,
        );
        assert!(valid_header.validate().is_ok());

        let invalid_header = GgufHeader::new(0x12345678, 1, 10, 5);
        assert!(invalid_header.validate().is_err());
    }

    #[test]
    fn test_gguf_tensor_validation() {
        let valid_tensor = GgufTensor::new(
            "test".to_string(),
            vec![1, 2, 3],
            GgufDataType::F32,
            100,
        );
        assert!(valid_tensor.validate_dimensions().is_ok());
        assert_eq!(valid_tensor.element_count(), 6);

        let invalid_tensor = GgufTensor::new(
            "test".to_string(),
            vec![1, 0, 3],
            GgufDataType::F32,
            100,
        );
        assert!(invalid_tensor.validate_dimensions().is_err());
    }

    #[test]
    fn test_gguf_data_type_sizes() {
        assert_eq!(GgufDataType::U8.size(), 1);
        assert_eq!(GgufDataType::I16.size(), 2);
        assert_eq!(GgufDataType::F32.size(), 4);
        assert_eq!(GgufDataType::F64.size(), 8);
    }

    #[test]
    fn test_gguf_metadata_validation() {
        let valid_metadata = GgufMetadata::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );
        assert!(valid_metadata.validate().is_ok());
        assert_eq!(valid_metadata.name(), "test_model");
        assert_eq!(valid_metadata.vocab_size(), 32000);
        assert!(valid_metadata.supports_inference());

        let invalid_metadata = GgufMetadata::new(
            "".to_string(),
            "1.0.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );
        assert!(invalid_metadata.validate().is_err());
    }

    #[test]
    fn test_gguf_metadata_without_inference() {
        let metadata = GgufMetadata::without_inference(
            "test_model".to_string(),
            "1.0.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );
        assert!(!metadata.supports_inference());
    }

    #[test]
    fn test_mock_model_resource_management() {
        let mut model = MockModel::new("test".to_string());
        assert!(!model.is_available());

        model.acquire().unwrap();
        assert!(model.is_available());

        model.release().unwrap();
        assert!(!model.is_available());
    }

    #[test]
    fn test_mock_model_validation() {
        let valid_model = MockModel::new("test".to_string());
        assert!(valid_model.validate().is_ok());

        let invalid_model = MockModel::new("".to_string());
        assert!(invalid_model.validate().is_err());
    }

    #[test]
    fn test_mock_model_loader() {
        let mut loader = MockModelLoader::new();

        // Test that loader is initially unavailable
        assert!(loader.load_model("test.gguf").is_err());

        // Test loader availability
        loader.acquire().unwrap();
        assert!(loader.is_available());

        // Test model loading
        let model = loader.load_model("test.gguf").unwrap();
        assert_eq!(model.name, "model_from_test.gguf");

        // Test metadata loading
        let metadata = loader.load_metadata("test.gguf").unwrap();
        assert_eq!(metadata.name(), "model_from_test.gguf");
        assert_eq!(metadata.architecture(), "llama");

        // Test model validation
        assert!(loader.validate_model_file("test.gguf").is_ok());
        assert!(loader.validate_model_file("").is_err());
    }

    #[test]
    fn test_mock_model_reset() {
        let mut model = MockModel::new("test".to_string());
        model.acquire().unwrap();
        assert!(model.is_available());

        model.reset();
        assert!(!model.is_available());
    }

    #[test]
    fn test_model_metadata_trait() {
        let metadata = GgufMetadata::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );

        // Test ModelMetadata trait methods
        assert_eq!(metadata.name(), "test_model");
        assert_eq!(metadata.version(), "1.0.0");
        assert_eq!(metadata.architecture(), "llama");
        assert_eq!(metadata.vocab_size(), 32000);
        assert_eq!(metadata.context_size(), 2048);
        assert!(metadata.supports_inference());
    }
}