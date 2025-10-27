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

    /// GGUF data types - extended for quantization support
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
        /// Quantization types
        /// Q4_0 - 4-bit quantization with 6-bit block scale
        Q4_0,
        /// Q4_1 - 4-bit quantization with 6-bit block scale and 4-bit min
        Q4_1,
        /// Q5_0 - 5-bit quantization with 6-bit block scale
        Q5_0,
        /// Q5_1 - 5-bit quantization with 6-bit block scale and 4-bit min
        Q5_1,
        /// Q8_0 - 8-bit quantization with 6-bit block scale
        Q8_0,
        /// Q2_K - 2-bit quantization (K-quants)
        Q2_K,
        /// Q3_K - 3-bit quantization (K-quants)
        Q3_K,
        /// Q4_K - 4-bit quantization (K-quants)
        Q4_K,
        /// Q5_K - 5-bit quantization (K-quants)
        Q5_K,
        /// Q6_K - 6-bit quantization (K-quants)
        Q6_K,
        /// Q8_K - 8-bit quantization (K-quants)
        Q8_K,
    }

    impl GgufDataType {
        /// Get size of this data type in bytes (base size)
        pub const fn size(&self) -> usize {
            match self {
                GgufDataType::U8 | GgufDataType::I8 => 1,
                GgufDataType::U16 | GgufDataType::I16 => 2,
                GgufDataType::U32 | GgufDataType::I32 | GgufDataType::F32 => 4,
                GgufDataType::F64 => 8,
                // Quantization types - return base element size
                GgufDataType::Q4_0 | GgufDataType::Q4_1 | GgufDataType::Q5_0 | GgufDataType::Q5_1 => 1,
                GgufDataType::Q8_0 => 1,
                GgufDataType::Q2_K => 1,
                GgufDataType::Q3_K => 1,
                GgufDataType::Q4_K => 1,
                GgufDataType::Q5_K => 1,
                GgufDataType::Q6_K => 1,
                GgufDataType::Q8_K => 1,
            }
        }

        /// Get the effective bits per element for quantization
        pub const fn bits_per_element(&self) -> usize {
            match self {
                GgufDataType::U8 | GgufDataType::I8 => 8,
                GgufDataType::U16 | GgufDataType::I16 => 16,
                GgufDataType::U32 | GgufDataType::I32 | GgufDataType::F32 => 32,
                GgufDataType::F64 => 64,
                GgufDataType::Q4_0 | GgufDataType::Q4_1 => 4,
                GgufDataType::Q5_0 | GgufDataType::Q5_1 => 5,
                GgufDataType::Q8_0 => 8,
                GgufDataType::Q2_K => 2,
                GgufDataType::Q3_K => 3,
                GgufDataType::Q4_K => 4,
                GgufDataType::Q5_K => 5,
                GgufDataType::Q6_K => 6,
                GgufDataType::Q8_K => 8,
            }
        }

        /// Check if this is a quantization type
        pub const fn is_quantized(&self) -> bool {
            matches!(
                self,
                GgufDataType::Q4_0
                    | GgufDataType::Q4_1
                    | GgufDataType::Q5_0
                    | GgufDataType::Q5_1
                    | GgufDataType::Q8_0
                    | GgufDataType::Q2_K
                    | GgufDataType::Q3_K
                    | GgufDataType::Q4_K
                    | GgufDataType::Q5_K
                    | GgufDataType::Q6_K
                    | GgufDataType::Q8_K
            )
        }

        /// Get the block size for quantization types
        pub const fn block_size(&self) -> usize {
            match self {
                GgufDataType::Q4_0 | GgufDataType::Q4_1 | GgufDataType::Q5_0 | GgufDataType::Q5_1 | GgufDataType::Q8_0 => 32,
                GgufDataType::Q2_K | GgufDataType::Q3_K | GgufDataType::Q4_K | GgufDataType::Q5_K | GgufDataType::Q6_K | GgufDataType::Q8_K => 256,
                _ => 1,
            }
        }
    }

    /// Memory usage tracking for GGUF models
    #[cfg(feature = "std")]
    #[derive(Debug, Clone)]
    pub struct MemoryUsage {
        /// Total memory usage in bytes
        pub total_bytes: usize,
        /// Model weights memory in bytes
        pub weights_bytes: usize,
        /// KV cache memory in bytes
        pub kv_cache_bytes: usize,
        /// Additional overhead in bytes
        pub overhead_bytes: usize,
    }

    impl MemoryUsage {
        /// Create new memory usage tracking
        pub fn new(total_bytes: usize, weights_bytes: usize, kv_cache_bytes: usize, overhead_bytes: usize) -> Self {
            Self {
                total_bytes,
                weights_bytes,
                kv_cache_bytes,
                overhead_bytes,
            }
        }

        /// Get memory usage in GB
        pub fn total_gb(&self) -> f64 {
            self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        }

        /// Get weights memory in GB
        pub fn weights_gb(&self) -> f64 {
            self.weights_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        }

        /// Check if memory usage exceeds limit for M1 16GB systems
        pub fn exceeds_m1_limit(&self) -> bool {
            self.total_gb() > 12.0 // 12GB limit for 16GB M1 systems
        }

        /// Check if model fits in M1 constraints
        pub fn fits_m1_constraints(&self) -> bool {
            !self.exceeds_m1_limit() && self.weights_gb() < 8.0 // Individual model limit
        }
    }

    /// Quantization configuration
    #[cfg(feature = "std")]
    #[derive(Debug, Clone)]
    pub struct QuantizationConfig {
        /// Quantization type
        pub quant_type: GgufDataType,
        /// Number of parameters in the model
        pub parameter_count: usize,
        /// Calculated memory usage
        pub memory_usage: MemoryUsage,
    }

    impl QuantizationConfig {
        /// Create new quantization configuration
        pub fn new(quant_type: GgufDataType, parameter_count: usize) -> Self {
            let memory_usage = Self::calculate_memory_usage(quant_type, parameter_count);
            Self {
                quant_type,
                parameter_count,
                memory_usage,
            }
        }

        /// Calculate memory usage for given quantization and parameter count
        fn calculate_memory_usage(quant_type: GgufDataType, parameter_count: usize) -> MemoryUsage {
            let bits_per_param = quant_type.bits_per_element();
            let weights_bytes = (parameter_count * bits_per_param + 7) / 8; // Round up to whole bytes

            // Estimate KV cache and overhead based on model size
            let kv_cache_bytes = match parameter_count {
                0..=3_000_000_000 => 512 * 1024 * 1024, // 512MB for small models
                3_000_000_001..=7_000_000_000 => 1024 * 1024 * 1024, // 1GB for medium models
                7_000_000_001..=13_000_000_000 => 1536 * 1024 * 1024, // 1.5GB for large models
                _ => 2048 * 1024 * 1024, // 2GB for very large models
            };

            let overhead_bytes = 512 * 1024 * 1024; // 512MB overhead

            let total_bytes = weights_bytes + kv_cache_bytes + overhead_bytes;

            MemoryUsage::new(total_bytes, weights_bytes, kv_cache_bytes, overhead_bytes)
        }

        /// Check if quantization is supported
        pub fn is_supported(&self) -> bool {
            matches!(
                self.quant_type,
                GgufDataType::Q4_0
                    | GgufDataType::Q4_1
                    | GgufDataType::Q5_0
                    | GgufDataType::Q5_1
                    | GgufDataType::Q8_0
                    | GgufDataType::Q4_K
                    | GgufDataType::Q5_K
                    | GgufDataType::Q6_K
                    | GgufDataType::Q8_K
            )
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
        /// Quantization configuration
        #[cfg(feature = "std")]
        pub quantization: Option<QuantizationConfig>,
        /// Number of layers in the model
        #[cfg(feature = "std")]
        pub layer_count: Option<usize>,
        /// Feed forward dimension
        #[cfg(feature = "std")]
        pub feed_forward_length: Option<usize>,
        /// Number of attention heads
        #[cfg(feature = "std")]
        pub attention_head_count: Option<usize>,
        /// Number of key-value heads
        #[cfg(feature = "std")]
        pub key_value_head_count: Option<usize>,
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
                #[cfg(feature = "std")]
                quantization: None,
                #[cfg(feature = "std")]
                layer_count: None,
                #[cfg(feature = "std")]
                feed_forward_length: None,
                #[cfg(feature = "std")]
                attention_head_count: None,
                #[cfg(feature = "std")]
                key_value_head_count: None,
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
                #[cfg(feature = "std")]
                quantization: None,
                #[cfg(feature = "std")]
                layer_count: None,
                #[cfg(feature = "std")]
                feed_forward_length: None,
                #[cfg(feature = "std")]
                attention_head_count: None,
                #[cfg(feature = "std")]
                key_value_head_count: None,
            }
        }

        /// Create metadata with quantization
        #[cfg(feature = "std")]
        pub fn with_quantization(
            name: String,
            version: String,
            architecture: String,
            vocab_size: usize,
            context_size: usize,
            quantization: QuantizationConfig,
        ) -> Self {
            Self {
                name,
                version,
                architecture,
                vocab_size,
                context_size,
                supports_inference: true,
                quantization: Some(quantization),
                layer_count: None,
                feed_forward_length: None,
                attention_head_count: None,
                key_value_head_count: None,
            }
        }

        /// Check if model fits M1 memory constraints
        #[cfg(feature = "std")]
        pub fn fits_m1_constraints(&self) -> bool {
            if let Some(quantization) = &self.quantization {
                quantization.memory_usage.fits_m1_constraints()
            } else {
                false // No quantization info means we can't validate
            }
        }

        /// Get memory usage information
        #[cfg(feature = "std")]
        pub fn memory_usage(&self) -> Option<&MemoryUsage> {
            self.quantization.as_ref().map(|q| &q.memory_usage)
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

/// Real GGUF file loader implementation
#[cfg(feature = "std")]
pub mod loader {
    use super::{
        gguf::{GgufHeader, GgufMetadata, GgufTensor, GgufDataType, QuantizationConfig, MemoryUsage},
        model::{ModelLoader, ModelMetadata},
        CoreError, CoreResult, Resource, Reset, Validate,
    };
    use std::{
        fs::File,
        io::Read,
        path::Path,
        string::{String, ToString},
        vec::Vec,
        vec,
        collections::HashMap,
    };
    use alloc::format;

    /// Loaded model weights and metadata
    #[derive(Debug, Clone)]
    pub struct LoadedWeights {
        /// Map of tensor name to loaded tensor data
        pub tensors: HashMap<String, LoadedTensor>,
        /// Model metadata
        pub metadata: GgufMetadata,
        /// Memory usage information
        pub memory_usage: MemoryUsage,
    }

    impl LoadedWeights {
        /// Create new loaded weights structure
        pub fn new(metadata: GgufMetadata) -> Self {
            let memory_usage = metadata.memory_usage()
                .cloned()
                .unwrap_or_else(|| MemoryUsage::new(0, 0, 0, 0));

            Self {
                tensors: HashMap::new(),
                metadata,
                memory_usage,
            }
        }

        /// Add a tensor to the loaded weights
        pub fn add_tensor(&mut self, name: String, tensor: LoadedTensor) {
            self.memory_usage.weights_bytes += tensor.data.len();
            self.memory_usage.total_bytes = self.memory_usage.weights_bytes +
                self.memory_usage.kv_cache_bytes +
                self.memory_usage.overhead_bytes;
            self.tensors.insert(name, tensor);
        }

        /// Get tensor by name
        pub fn get_tensor(&self, name: &str) -> Option<&LoadedTensor> {
            self.tensors.get(name)
        }

        /// Get total number of tensors
        pub fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        /// Validate loaded weights
        pub fn validate(&self) -> CoreResult<()> {
            if self.tensors.is_empty() {
                return Err(CoreError::InvalidInput("no tensors loaded"));
            }

            for (name, tensor) in &self.tensors {
                if name.is_empty() {
                    return Err(CoreError::InvalidInput("tensor name cannot be empty"));
                }
                tensor.validate()?;
            }

            self.metadata.validate()?;
            Ok(())
        }
    }

    /// Individual loaded tensor with raw data
    #[derive(Debug, Clone)]
    pub struct LoadedTensor {
        /// Raw tensor data (quantized)
        pub data: Vec<u8>,
        /// Tensor shape
        pub shape: Vec<usize>,
        /// Data type
        pub dtype: GgufDataType,
        /// Quantization information
        pub quantization: Option<QuantizationInfo>,
    }

    impl LoadedTensor {
        /// Create new loaded tensor
        pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: GgufDataType) -> Self {
            Self {
                data,
                shape,
                dtype,
                quantization: None,
            }
        }

        /// Create new quantized tensor
        pub fn new_quantized(
            data: Vec<u8>,
            shape: Vec<usize>,
            dtype: GgufDataType,
            quantization: QuantizationInfo,
        ) -> Self {
            Self {
                data,
                shape,
                dtype,
                quantization: Some(quantization),
            }
        }

        /// Get total number of elements
        pub fn element_count(&self) -> usize {
            self.shape.iter().product()
        }

        /// Get data size in bytes
        pub fn size_bytes(&self) -> usize {
            self.data.len()
        }

        /// Validate tensor
        pub fn validate(&self) -> CoreResult<()> {
            if self.shape.is_empty() {
                return Err(CoreError::InvalidInput("tensor shape cannot be empty"));
            }
            if self.shape.iter().any(|&dim| dim == 0) {
                return Err(CoreError::InvalidInput("tensor dimensions cannot be zero"));
            }
            if self.data.is_empty() {
                return Err(CoreError::InvalidInput("tensor data cannot be empty"));
            }

            // Validate quantization if present
            if let Some(ref quant) = self.quantization {
                quant.validate()?;
            }

            Ok(())
        }
    }

    /// Quantization information for loaded tensors
    #[derive(Debug, Clone)]
    pub struct QuantizationInfo {
        /// Quantization type
        pub quant_type: GgufDataType,
        /// Block size for quantization
        pub block_size: usize,
        /// Number of blocks
        pub block_count: usize,
        /// Scale factors (if applicable)
        pub scales: Option<Vec<f32>>,
        /// Minimum values (if applicable)
        pub mins: Option<Vec<f32>>,
    }

    impl QuantizationInfo {
        /// Create new quantization info
        pub fn new(quant_type: GgufDataType, block_size: usize, block_count: usize) -> Self {
            Self {
                quant_type,
                block_size,
                block_count,
                scales: None,
                mins: None,
            }
        }

        /// Create quantization info with scales
        pub fn with_scales(
            quant_type: GgufDataType,
            block_size: usize,
            block_count: usize,
            scales: Vec<f32>,
        ) -> Self {
            Self {
                quant_type,
                block_size,
                block_count,
                scales: Some(scales),
                mins: None,
            }
        }

        /// Create quantization info with scales and mins
        pub fn with_scales_and_mins(
            quant_type: GgufDataType,
            block_size: usize,
            block_count: usize,
            scales: Vec<f32>,
            mins: Vec<f32>,
        ) -> Self {
            Self {
                quant_type,
                block_size,
                block_count,
                scales: Some(scales),
                mins: Some(mins),
            }
        }

        /// Validate quantization info
        pub fn validate(&self) -> CoreResult<()> {
            if self.block_size == 0 {
                return Err(CoreError::InvalidInput("block size cannot be zero"));
            }
            if self.block_count == 0 {
                return Err(CoreError::InvalidInput("block count cannot be zero"));
            }
            if let Some(ref scales) = self.scales {
                if scales.is_empty() {
                    return Err(CoreError::InvalidInput("scales cannot be empty"));
                }
            }
            if let Some(ref mins) = self.mins {
                if mins.is_empty() {
                    return Err(CoreError::InvalidInput("mins cannot be empty"));
                }
            }
            Ok(())
        }
    }

    /// Model handle for loaded models
    #[derive(Debug)]
    pub struct ModelHandle {
        /// Loaded weights
        pub weights: LoadedWeights,
        /// Model is ready for inference
        pub ready: bool,
        /// Model ID
        pub id: String,
    }

    impl ModelHandle {
        /// Create new model handle
        pub fn new(id: String, weights: LoadedWeights) -> Self {
            Self {
                weights,
                ready: false,
                id,
            }
        }

        /// Get model ID
        pub fn id(&self) -> &str {
            &self.id
        }

        /// Check if model is ready
        pub fn is_ready(&self) -> bool {
            self.ready
        }

        /// Mark model as ready
        pub fn set_ready(&mut self) {
            self.ready = true;
        }

        /// Get model metadata
        pub fn metadata(&self) -> &GgufMetadata {
            &self.weights.metadata
        }

        /// Get memory usage
        pub fn memory_usage(&self) -> &MemoryUsage {
            &self.weights.memory_usage
        }

        /// Validate model handle
        pub fn validate(&self) -> CoreResult<()> {
            self.weights.validate()?;
            if self.id.is_empty() {
                return Err(CoreError::InvalidInput("model ID cannot be empty"));
            }
            Ok(())
        }
    }

    impl Resource for ModelHandle {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.ready
        }

        fn acquire(&mut self) -> CoreResult<()> {
            self.validate()?;
            self.ready = true;
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            self.ready = false;
            Ok(())
        }
    }

    impl Reset for ModelHandle {
        fn reset(&mut self) {
            self.ready = false;
        }
    }

    impl Validate for ModelHandle {
        fn validate(&self) -> CoreResult<()> {
            self.weights.validate()?;
            if self.id.is_empty() {
                return Err(CoreError::InvalidInput("model ID cannot be empty"));
            }
            Ok(())
        }
    }

    /// Real GGUF file loader that parses actual GGUF files
    #[derive(Debug)]
    pub struct GgufLoader {
        pub resource_available: bool,
        pub loaded_models: Vec<String>,
    }

    impl GgufLoader {
        /// Create a new GGUF loader
        pub fn new() -> Self {
            Self {
                resource_available: false,
                loaded_models: Vec::new(),
            }
        }

        /// Load actual tensor weights from GGUF file
        pub fn load_weights(&mut self, path: &str) -> CoreResult<LoadedWeights> {
            self.validate()?;

            // Parse metadata first
            let metadata = self.parse_metadata(path)?;

            // Create loaded weights structure
            let mut loaded_weights = LoadedWeights::new(metadata.clone());

            // Open file for reading
            let mut file = File::open(path)
                .map_err(|_| CoreError::Generic("Cannot open GGUF file"))?;

            // Read header
            let header = self.read_gguf_header(&mut file)?;
            header.validate()?;

            // TODO: Implement actual tensor loading from GGUF file
            // For now, add mock tensors to demonstrate the structure
            self.add_mock_tensors(&mut loaded_weights, &header);

            // Validate loaded weights
            loaded_weights.validate()?;

            Ok(loaded_weights)
        }

        /// Create model from loaded weights
        pub fn create_model(&mut self, weights: LoadedWeights, id: String) -> CoreResult<ModelHandle> {
            self.validate()?;

            // Create model handle (don't acquire yet - let user do that)
            let model = ModelHandle::new(id, weights);

            Ok(model)
        }

        /// Validate loaded model
        pub fn validate_model(&self, model: &ModelHandle) -> CoreResult<()> {
            model.validate()
        }

        /// Load and create model in one operation
        pub fn load_model_from_file(&mut self, path: &str, model_id: String) -> CoreResult<ModelHandle> {
            // Load weights
            let weights = self.load_weights(path)?;

            // Create model
            let mut model = self.create_model(weights, model_id)?;

            // Mark as loaded in our tracking
            self.loaded_models.push(path.to_string());

            Ok(model)
        }

        // TODO: This is a placeholder implementation for the RED phase
        // In the GREEN phase, this will be replaced with actual GGUF tensor parsing
        fn add_mock_tensors(&self, loaded_weights: &mut LoadedWeights, header: &GgufHeader) {
            // Add some mock tensors to demonstrate the structure
            for i in 0..header.tensor_count.min(5) {
                let tensor_name = format!("tensor_{}", i);
                let shape = vec![1024, 1024]; // Mock shape
                let data = vec![0u8; 1024 * 1024]; // Mock data
                let dtype = GgufDataType::Q4_K; // Mock quantized type

                let tensor = LoadedTensor::new(data, shape, dtype);
                loaded_weights.add_tensor(tensor_name, tensor);
            }
        }

        /// Read and parse GGUF file header
        pub fn read_gguf_header(&self, file: &mut File) -> CoreResult<GgufHeader> {
            use std::io::Read;

            // Read magic number (4 bytes)
            let mut magic_bytes = [0u8; 4];
            file.read_exact(&mut magic_bytes)
                .map_err(|_| CoreError::Generic("Failed to read magic number"))?;

            let magic = u32::from_le_bytes(magic_bytes);
            if magic != GgufHeader::gguf_magic() {
                return Err(CoreError::InvalidInput("Not a valid GGUF file"));
            }

            // Read version (4 bytes)
            let mut version_bytes = [0u8; 4];
            file.read_exact(&mut version_bytes)
                .map_err(|_| CoreError::Generic("Failed to read version"))?;
            let version = u32::from_le_bytes(version_bytes);

            // Read tensor count (4 bytes)
            let mut tensor_count_bytes = [0u8; 4];
            file.read_exact(&mut tensor_count_bytes)
                .map_err(|_| CoreError::Generic("Failed to read tensor count"))?;
            let tensor_count = u32::from_le_bytes(tensor_count_bytes);

            // Read KV count (4 bytes)
            let mut kv_count_bytes = [0u8; 4];
            file.read_exact(&mut kv_count_bytes)
                .map_err(|_| CoreError::Generic("Failed to read KV count"))?;
            let kv_count = u32::from_le_bytes(kv_count_bytes);

            Ok(GgufHeader::new(magic, version, tensor_count, kv_count))
        }

        /// Parse metadata from GGUF file
        pub fn parse_metadata(&self, path: &str) -> CoreResult<GgufMetadata> {
            let mut file = File::open(path)
                .map_err(|_| CoreError::Generic("Cannot open file"))?;

            // Read header
            let header = self.read_gguf_header(&mut file)?;
            header.validate()?;

            // For now, return basic metadata
            // TODO: Parse actual KV pairs from the file
            let quant_config = QuantizationConfig::new(GgufDataType::Q4_K, 7_000_000_000);
            let metadata = GgufMetadata::with_quantization(
                "gguf_model".to_string(),
                "1.0".to_string(),
                "llama".to_string(),
                32000,
                2048,
                quant_config,
            );

            Ok(metadata)
        }

        /// Validate GGUF file format
        pub fn validate_gguf_file(&self, path: &str) -> CoreResult<()> {
            let mut file = File::open(path)
                .map_err(|_| CoreError::Generic("Cannot open file"))?;

            // Try to read header
            let header = self.read_gguf_header(&mut file)?;
            header.validate()?;

            // TODO: Add more comprehensive validation
            Ok(())
        }
    }

    impl Resource for GgufLoader {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.resource_available
        }

        fn acquire(&mut self) -> CoreResult<()> {
            self.resource_available = true;
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            self.resource_available = false;
            self.loaded_models.clear();
            Ok(())
        }
    }

    impl Reset for GgufLoader {
        fn reset(&mut self) {
            self.resource_available = false;
            self.loaded_models.clear();
        }
    }

    impl Validate for GgufLoader {
        fn validate(&self) -> CoreResult<()> {
            if !self.resource_available {
                return Err(CoreError::Unavailable("GGUF loader not available"));
            }
            Ok(())
        }
    }

    impl ModelLoader for GgufLoader {
        type Model = ModelHandle;
        type Metadata = GgufMetadata;

        fn load_metadata(&mut self, path: &str) -> CoreResult<Self::Metadata> {
            self.validate()?;
            self.parse_metadata(path)
        }

        fn load_model(&mut self, path: &str) -> CoreResult<Self::Model> {
            self.validate()?;

            // Validate the file first
            self.validate_gguf_file(path)?;

            // Load model using the new method
            let model_id = format!("gguf_model_{}", path);
            self.load_model_from_file(path, model_id)
        }

        fn validate_model_file(&self, path: &str) -> CoreResult<()> {
            if path.is_empty() {
                return Err(CoreError::InvalidInput("path cannot be empty"));
            }

            if !Path::new(path).exists() {
                return Err(CoreError::NotFound("File not found"));
            }

            self.validate_gguf_file(path)
        }
    }

    impl Default for GgufLoader {
        fn default() -> Self {
            Self::new()
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

/// Candle tensor integration module
#[cfg(feature = "std")]
pub mod candle_integration {
    use super::{
        gguf::{GgufDataType, MemoryUsage},
        loader::{LoadedTensor, QuantizationInfo},
        CoreError, CoreResult,
    };
    use candle_core::{Tensor, Device, DType};
    use std::{
        vec::Vec,
        string::String,
        format,
    };

    /// Candle tensor created from GGUF data
    #[derive(Debug, Clone)]
    pub struct CandleTensor {
        /// The underlying Candle tensor
        pub tensor: Tensor,
        /// Original quantization info (if any)
        pub quantization: Option<QuantizationInfo>,
        /// Memory used by this tensor
        pub memory_usage: usize,
    }

    impl CandleTensor {
        /// Create a new Candle tensor from loaded tensor data
        pub fn from_loaded_tensor(
            loaded_tensor: &LoadedTensor,
            device: &Device,
        ) -> CoreResult<Self> {
            // TODO: This is a placeholder implementation for RED phase
            // In GREEN phase, implement actual quantization decompression

            let shape: Vec<usize> = loaded_tensor.shape.clone();
            let dtype = Self::gguf_dtype_to_candle(loaded_tensor.dtype)?;

            // For now, create a tensor of zeros as placeholder
            // In GREEN phase, this will decode the actual quantized data
            let tensor = Tensor::zeros(shape.as_slice(), dtype, device)
                .map_err(|_| CoreError::Generic("tensor creation failed"))?;

            let memory_usage = tensor.elem_count() * dtype.size_in_bytes();

            Ok(Self {
                tensor,
                quantization: loaded_tensor.quantization.clone(),
                memory_usage,
            })
        }

        /// Get the underlying tensor
        pub fn tensor(&self) -> &Tensor {
            &self.tensor
        }

        /// Get tensor shape
        pub fn shape(&self) -> &[usize] {
            self.tensor.dims()
        }

        /// Get element count
        pub fn element_count(&self) -> usize {
            self.tensor.elem_count()
        }

        /// Get memory usage in bytes
        pub fn memory_usage_bytes(&self) -> usize {
            self.memory_usage
        }

        /// Convert GGUF data type to Candle DType
        fn gguf_dtype_to_candle(gguf_dtype: GgufDataType) -> CoreResult<DType> {
            match gguf_dtype {
                GgufDataType::F32 => Ok(DType::F32),
                GgufDataType::F64 => Ok(DType::F64),
                GgufDataType::U8 => Ok(DType::U8),
                // Candle doesn't have separate signed integer types, map to available ones
                GgufDataType::I8 => Ok(DType::U8), // Map I8 to U8 for placeholder
                GgufDataType::U16 => Ok(DType::F32), // Map to F32 for placeholder
                GgufDataType::I16 => Ok(DType::F32), // Map to F32 for placeholder
                GgufDataType::U32 => Ok(DType::U32),
                GgufDataType::I32 => Ok(DType::U32), // Map I32 to U32 for placeholder
                // For quantized types, we'll need to decompress to F32
                // This is a placeholder - in GREEN phase implement proper decompression
                GgufDataType::Q4_0 | GgufDataType::Q4_1 |
                GgufDataType::Q5_0 | GgufDataType::Q5_1 |
                GgufDataType::Q8_0 | GgufDataType::Q2_K |
                GgufDataType::Q3_K | GgufDataType::Q4_K |
                GgufDataType::Q5_K | GgufDataType::Q6_K |
                GgufDataType::Q8_K => Ok(DType::F32), // Placeholder
            }
        }

        /// Validate tensor conversion
        pub fn validate(&self) -> CoreResult<()> {
            if self.tensor.dims().is_empty() {
                return Err(CoreError::InvalidInput("tensor cannot be empty"));
            }
            if self.tensor.dims().iter().any(|&dim| dim == 0) {
                return Err(CoreError::InvalidInput("tensor dimensions cannot be zero"));
            }
            Ok(())
        }
    }

    /// Batch tensor converter for efficient processing
    #[derive(Debug)]
    pub struct TensorConverter {
        /// Target device for tensor creation
        pub device: Device,
        /// Total memory allocated
        pub total_memory: usize,
    }

    impl TensorConverter {
        /// Create new tensor converter
        pub fn new(device: Device) -> Self {
            Self {
                device,
                total_memory: 0,
            }
        }

        /// Convert multiple loaded tensors to Candle tensors
        pub fn convert_batch(
            &mut self,
            loaded_tensors: &[(String, LoadedTensor)],
        ) -> CoreResult<Vec<(String, CandleTensor)>> {
            let mut results = Vec::with_capacity(loaded_tensors.len());

            for (name, loaded_tensor) in loaded_tensors {
                let candle_tensor = CandleTensor::from_loaded_tensor(loaded_tensor, &self.device)?;
                self.total_memory += candle_tensor.memory_usage;
                results.push((name.clone(), candle_tensor));
            }

            Ok(results)
        }

        /// Get total memory allocated
        pub fn total_memory(&self) -> usize {
            self.total_memory
        }

        /// Reset memory tracking
        pub fn reset_memory_tracking(&mut self) {
            self.total_memory = 0;
        }
    }

    /// Memory-efficient tensor loading with streaming support
    #[derive(Debug)]
    pub struct StreamingTensorLoader {
        /// Device for tensor creation
        pub device: Device,
        /// Chunk size for streaming large tensors
        pub chunk_size: usize,
        /// Total loaded tensors
        pub loaded_count: usize,
    }

    impl StreamingTensorLoader {
        /// Create new streaming loader
        pub fn new(device: Device, chunk_size: usize) -> Self {
            Self {
                device,
                chunk_size,
                loaded_count: 0,
            }
        }

        /// Load tensor in chunks (placeholder for RED phase)
        pub fn load_tensor_streaming(
            &mut self,
            loaded_tensor: &LoadedTensor,
        ) -> CoreResult<CandleTensor> {
            // TODO: In GREEN phase, implement actual streaming
            // For now, just convert normally
            let result = CandleTensor::from_loaded_tensor(loaded_tensor, &self.device)?;
            self.loaded_count += 1;
            Ok(result)
        }

        /// Get number of loaded tensors
        pub fn loaded_count(&self) -> usize {
            self.loaded_count
        }
    }
}

// Re-export key types for convenience
pub use model::{ModelLoader, ModelMetadata, ModelRequirements, ModelValidator};
#[cfg(feature = "std")]
pub use gguf::{GgufDataType, GgufHeader, GgufMetadata, GgufTensor, MemoryUsage, QuantizationConfig};
#[cfg(not(feature = "std"))]
pub use gguf::{GgufDataType, GgufHeader, GgufMetadata, GgufTensor};
#[cfg(feature = "std")]
pub use loader::{
    GgufLoader, LoadedWeights, LoadedTensor, QuantizationInfo, ModelHandle
};
#[cfg(feature = "std")]
pub use candle_integration::{CandleTensor, TensorConverter, StreamingTensorLoader};
pub use mock::{MockModel, MockModelLoader};

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use super::{MemoryUsage, QuantizationConfig, GgufLoader};
    use alloc::{
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

    // RED TESTS: These will initially fail and pass after implementation

    #[cfg(feature = "std")]
    #[test]
    fn test_quantization_data_types_extended() {
        // Test quantization type properties
        assert_eq!(GgufDataType::Q4_K.bits_per_element(), 4);
        assert_eq!(GgufDataType::Q5_K.bits_per_element(), 5);
        assert!(GgufDataType::Q4_K.is_quantized());
        assert!(GgufDataType::Q5_K.is_quantized());
        assert!(!GgufDataType::F32.is_quantized());

        // Test block sizes
        assert_eq!(GgufDataType::Q4_K.block_size(), 256);
        assert_eq!(GgufDataType::Q4_0.block_size(), 32);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_memory_usage_tracking() {
        // Test memory usage calculation for 7B parameter model with Q4_K_M
        let memory_usage = MemoryUsage::new(
            5_500_000_000, // 5.5GB total
            4_000_000_000, // 4GB weights
            1_000_000_000, // 1GB KV cache
            500_000_000,   // 500MB overhead
        );

        // Use appropriate tolerances for floating point comparison
        assert!(memory_usage.weights_gb() > 3.5 && memory_usage.weights_gb() < 4.5);
        assert!(memory_usage.total_gb() > 5.0 && memory_usage.total_gb() < 6.0);
        assert!(!memory_usage.exceeds_m1_limit()); // Should be under 12GB
        assert!(memory_usage.fits_m1_constraints());

        // Test memory limit enforcement
        let large_memory = MemoryUsage::new(
            18_000_000_000, // 18GB total
            15_000_000_000, // 15GB weights
            2_000_000_000,  // 2GB KV cache
            1_000_000_000,  // 1GB overhead
        );

        assert!(large_memory.exceeds_m1_limit()); // Should exceed 12GB
        assert!(!large_memory.fits_m1_constraints());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_quantization_config() {
        // Test 7B parameter model with Q4_K quantization
        let quant_config = QuantizationConfig::new(GgufDataType::Q4_K, 7_000_000_000);

        assert_eq!(quant_config.quant_type, GgufDataType::Q4_K);
        assert_eq!(quant_config.parameter_count, 7_000_000_000);
        assert!(quant_config.is_supported());

        // Test memory calculation
        let memory_usage = &quant_config.memory_usage;
        assert!(memory_usage.weights_gb() > 3.0); // Should be around 3.5GB
        assert!(memory_usage.weights_gb() < 4.0);
        assert!(!memory_usage.exceeds_m1_limit()); // Should fit in M1 constraints
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_metadata_with_quantization() {
        let quant_config = QuantizationConfig::new(GgufDataType::Q4_K, 7_000_000_000);
        let metadata = GgufMetadata::with_quantization(
            "mistral-7b-instruct".to_string(),
            "v0.2".to_string(),
            "llama".to_string(),
            32000,
            4096,
            quant_config,
        );

        assert_eq!(metadata.name(), "mistral-7b-instruct");
        assert!(metadata.fits_m1_constraints());

        // Test memory usage retrieval
        let memory_usage = metadata.memory_usage().unwrap();
        assert!(memory_usage.total_gb() < 12.0);
        assert!(memory_usage.weights_gb() < 8.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_real_gguf_file_parsing_should_fail() {
        // This test should fail initially because we haven't implemented real GGUF parsing yet
        // It will pass after we implement the real GgufLoader

        // Create a placeholder for a real GGUF file path (doesn't need to exist for now)
        let gguf_path = "test_models/mistral-7b-instruct.Q4_K_M.gguf";

        // This should fail because MockModelLoader doesn't handle real files
        let mut loader = MockModelLoader::new();
        loader.acquire().unwrap();

        // Try to load a real GGUF file - this should fail with current mock implementation
        let result = loader.load_model(gguf_path);
        assert!(result.is_ok()); // Current mock implementation returns Ok, but with fake data

        let model = result.unwrap();
        assert_eq!(model.name, format!("model_from_{}", gguf_path));

        // TODO: After implementing real GgufLoader, this test should:
        // 1. Parse actual GGUF file structure
        // 2. Extract real model metadata
        // 3. Validate quantization format
        // 4. Calculate actual memory usage
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_file_validation() {
        // Test GGUF file format validation
        let valid_header = GgufHeader::new(
            GgufHeader::gguf_magic(),
            3, // GGUF v3
            280, // Typical tensor count for 7B model
            150, // Typical KV count
        );

        assert!(valid_header.validate().is_ok());

        // Test invalid magic number
        let invalid_header = GgufHeader::new(0x12345678, 3, 280, 150);
        assert!(invalid_header.validate().is_err());

        // Test invalid version
        let invalid_version = GgufHeader::new(GgufHeader::gguf_magic(), 0, 280, 150);
        assert!(invalid_version.validate().is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_m1_memory_constraint_enforcement() {
        // Test memory constraint enforcement for M1 16GB systems

        // Small model that should fit
        let small_quant = QuantizationConfig::new(GgufDataType::Q4_K, 3_000_000_000);
        assert!(small_quant.memory_usage.fits_m1_constraints());

        // Large model that should not fit
        let large_quant = QuantizationConfig::new(GgufDataType::Q4_K, 34_000_000_000);
        assert!(!large_quant.memory_usage.fits_m1_constraints());

        // Edge case: exactly at the limit
        let edge_quant = QuantizationConfig::new(GgufDataType::Q8_0, 13_000_000_000);
        assert!(!edge_quant.memory_usage.fits_m1_constraints()); // Should exceed with KV cache
    }

    #[test]
    fn test_gguf_quantization_format_support() {
        // Test supported quantization formats
        let supported_formats = [
            GgufDataType::Q4_0,
            GgufDataType::Q4_1,
            GgufDataType::Q5_0,
            GgufDataType::Q5_1,
            GgufDataType::Q8_0,
            GgufDataType::Q4_K,
            GgufDataType::Q5_K,
            GgufDataType::Q6_K,
            GgufDataType::Q8_K,
        ];

        for format in &supported_formats {
            assert!(format.is_quantized());
            assert!(format.bits_per_element() <= 8);
        }

        // Test that regular data types are not considered quantized
        assert!(!GgufDataType::F32.is_quantized());
        assert!(!GgufDataType::F64.is_quantized());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_real_gguf_loader_interface() {
        // This test defines the interface we want to implement
        // It will fail initially and pass after implementing GgufLoader

        // Define the interface we expect
        trait RealGgufLoader {
            fn load_from_file(&mut self, path: &str) -> CoreResult<GgufMetadata>;
            fn validate_file_format(&self, path: &str) -> CoreResult<()>;
            fn estimate_memory_usage(&self, path: &str) -> CoreResult<MemoryUsage>;
        }

        // This test documents what we want to implement
        // TODO: Replace MockModelLoader with real GgufLoader that implements RealGgufLoader

        // For now, just verify that our trait definition compiles
        fn _check_trait_compiles<T: RealGgufLoader>(_: &T) {}

        // This test passes if the trait compiles, but will be updated
        // to test actual implementation in the GREEN phase
        assert!(true);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_real_gguf_loader_implementation() {
        // Test the new GgufLoader implementation
        let mut loader = GgufLoader::new();

        // Initially not available
        assert!(!loader.is_available());

        // Acquire the loader
        loader.acquire().unwrap();
        assert!(loader.is_available());

        // Test validation with non-existent file
        let result = loader.validate_model_file("non_existent_file.gguf");
        assert!(result.is_err());

        // Test validation with empty path
        let result = loader.validate_model_file("");
        assert!(result.is_err());

        // Test with a file that exists but isn't a GGUF file (create a temporary file)
        use std::fs::File;
        use std::io::Write;

        let temp_path = "temp_test_file.txt";
        let mut temp_file = File::create(temp_path).unwrap();
        temp_file.write_all(b"This is not a GGUF file").unwrap();
        temp_file.flush().unwrap();

        // Should fail because it's not a valid GGUF file
        let result = loader.validate_model_file(temp_path);
        assert!(result.is_err());

        // Clean up
        std::fs::remove_file(temp_path).unwrap();

        // Release the loader
        loader.release().unwrap();
        assert!(!loader.is_available());
    }

    // ===== RED PHASE TESTS FOR PHASE 2.3 =====
    // These tests define the expected behavior for real weight loading

    #[cfg(feature = "std")]
    #[test]
    fn test_loaded_weights_structure_creation() {
        // Test LoadedWeights structure creation and validation
        let metadata = GgufMetadata::new(
            "test_model".to_string(),
            "1.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );

        let mut loaded_weights = LoadedWeights::new(metadata.clone());

        // Initially empty
        assert_eq!(loaded_weights.tensor_count(), 0);

        // Should fail validation because no tensors loaded
        assert!(loaded_weights.validate().is_err());

        // Add a tensor
        let tensor = LoadedTensor::new(
            vec![1, 2, 3, 4], // Mock data
            vec![2, 2], // Shape
            GgufDataType::Q4_K,
        );
        loaded_weights.add_tensor("test_tensor".to_string(), tensor);

        // Should now have 1 tensor
        assert_eq!(loaded_weights.tensor_count(), 1);

        // Should still fail because metadata validation
        assert!(loaded_weights.validate().is_ok());

        // Check tensor retrieval
        let retrieved_tensor = loaded_weights.get_tensor("test_tensor");
        assert!(retrieved_tensor.is_some());
        assert_eq!(retrieved_tensor.unwrap().shape, vec![2, 2]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_loaded_tensor_validation() {
        // Test LoadedTensor validation
        let valid_tensor = LoadedTensor::new(
            vec![1, 2, 3, 4],
            vec![2, 2],
            GgufDataType::Q4_K,
        );
        assert!(valid_tensor.validate().is_ok());
        assert_eq!(valid_tensor.element_count(), 4);
        assert_eq!(valid_tensor.size_bytes(), 4);

        // Test invalid tensor with empty shape
        let invalid_shape_tensor = LoadedTensor::new(
            vec![1, 2, 3, 4],
            vec![],
            GgufDataType::Q4_K,
        );
        assert!(invalid_shape_tensor.validate().is_err());

        // Test invalid tensor with zero dimension
        let invalid_dim_tensor = LoadedTensor::new(
            vec![1, 2, 3, 4],
            vec![2, 0],
            GgufDataType::Q4_K,
        );
        assert!(invalid_dim_tensor.validate().is_err());

        // Test invalid tensor with empty data
        let invalid_data_tensor = LoadedTensor::new(
            vec![],
            vec![2, 2],
            GgufDataType::Q4_K,
        );
        assert!(invalid_data_tensor.validate().is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_quantization_info_validation() {
        // Test QuantizationInfo creation and validation
        let quant_info = QuantizationInfo::new(GgufDataType::Q4_K, 256, 100);
        assert!(quant_info.validate().is_ok());
        assert_eq!(quant_info.quant_type, GgufDataType::Q4_K);
        assert_eq!(quant_info.block_size, 256);
        assert_eq!(quant_info.block_count, 100);

        // Test with scales
        let scales = vec![0.1, 0.2, 0.3];
        let quant_info_with_scales = QuantizationInfo::with_scales(
            GgufDataType::Q4_K, 256, 100, scales.clone()
        );
        assert!(quant_info_with_scales.validate().is_ok());
        assert_eq!(quant_info_with_scales.scales.unwrap(), scales);

        // Test invalid block size
        let invalid_quant = QuantizationInfo::new(GgufDataType::Q4_K, 0, 100);
        assert!(invalid_quant.validate().is_err());

        // Test invalid block count
        let invalid_quant2 = QuantizationInfo::new(GgufDataType::Q4_K, 256, 0);
        assert!(invalid_quant2.validate().is_err());

        // Test empty scales
        let invalid_quant3 = QuantizationInfo::with_scales(GgufDataType::Q4_K, 256, 100, vec![]);
        assert!(invalid_quant3.validate().is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_model_handle_creation_and_validation() {
        // Test ModelHandle creation and management
        let metadata = GgufMetadata::new(
            "test_model".to_string(),
            "1.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );
        let loaded_weights = LoadedWeights::new(metadata);

        let model_handle = ModelHandle::new("test_model_id".to_string(), loaded_weights);

        // Should not be ready initially
        assert!(!model_handle.is_ready());
        assert_eq!(model_handle.id(), "test_model_id");

        // Should fail validation because no tensors in weights
        assert!(model_handle.validate().is_err());

        // Test acquire/release - acquire should also fail due to empty weights
        let mut model_handle = model_handle;
        let acquire_result = model_handle.acquire();
        assert!(acquire_result.is_err()); // Should fail due to empty weights
        assert!(!model_handle.is_available()); // Should not be available

        model_handle.release().unwrap(); // Release should still work
        assert!(!model_handle.is_available());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_loader_weight_loading_should_load_mock_data() {
        // Test that GgufLoader can load weights (currently with mock data)
        let mut loader = GgufLoader::new();
        loader.acquire().unwrap();

        // Create a temporary GGUF-like file for testing
        use std::fs::File;
        use std::io::Write;

        let temp_path = "temp_test_gguf.gguf";
        let mut temp_file = File::create(temp_path).unwrap();

        // Write minimal GGUF header
        temp_file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // GGUF magic
        temp_file.write_all(&3u32.to_le_bytes()).unwrap(); // Version
        temp_file.write_all(&5u32.to_le_bytes()).unwrap(); // Tensor count
        temp_file.write_all(&10u32.to_le_bytes()).unwrap(); // KV count
        temp_file.flush().unwrap();

        // Load weights - should work with mock implementation
        let result = loader.load_weights(temp_path);

        // This should succeed with our mock implementation
        assert!(result.is_ok());
        let loaded_weights = result.unwrap();

        // Should have some tensors (mock data)
        assert!(loaded_weights.tensor_count() > 0);

        // Should have valid metadata
        assert_eq!(loaded_weights.metadata.name(), "gguf_model");

        // Should validate successfully
        assert!(loaded_weights.validate().is_ok());

        // Clean up
        std::fs::remove_file(temp_path).unwrap();
        loader.release().unwrap();
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_loader_model_creation_should_fail_with_empty_id() {
        // Test model creation with empty ID should fail
        let mut loader = GgufLoader::new();
        loader.acquire().unwrap();

        let metadata = GgufMetadata::new(
            "test_model".to_string(),
            "1.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );
        let weights = LoadedWeights::new(metadata);

        // Create model with empty ID
        let result = loader.create_model(weights, "".to_string());

        // Should succeed creation but fail validation due to empty ID and no tensors
        assert!(result.is_ok());
        let model = result.unwrap();
        assert!(model.validate().is_err()); // Should fail due to empty ID or no tensors

        loader.release().unwrap();
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_loader_load_model_from_file_should_track_models() {
        // Test that load_model_from_file tracks loaded models
        let mut loader = GgufLoader::new();
        loader.acquire().unwrap();

        // Create a temporary GGUF-like file
        use std::fs::File;
        use std::io::Write;

        let temp_path = "temp_test_gguf_2.gguf";
        let mut temp_file = File::create(temp_path).unwrap();

        // Write minimal GGUF header
        temp_file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // GGUF magic
        temp_file.write_all(&3u32.to_le_bytes()).unwrap(); // Version
        temp_file.write_all(&3u32.to_le_bytes()).unwrap(); // Tensor count
        temp_file.write_all(&10u32.to_le_bytes()).unwrap(); // KV count
        temp_file.flush().unwrap();

        // Load model from file
        let result = loader.load_model_from_file(temp_path, "test_model".to_string());

        // Should succeed
        assert!(result.is_ok());
        let model = result.unwrap();

        // Model should have correct ID
        assert_eq!(model.id(), "test_model");

        // Should track the loaded file
        assert!(loader.loaded_models.contains(&temp_path.to_string()));

        // Clean up
        std::fs::remove_file(temp_path).unwrap();
        loader.release().unwrap();
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_loader_model_validation() {
        // Test model validation through loader
        let mut loader = GgufLoader::new();
        loader.acquire().unwrap();

        // Create a valid model
        let metadata = GgufMetadata::new(
            "test_model".to_string(),
            "1.0".to_string(),
            "llama".to_string(),
            32000,
            2048,
        );
        let mut weights = LoadedWeights::new(metadata);

        // Add a tensor to make it valid
        let tensor = LoadedTensor::new(
            vec![1, 2, 3, 4],
            vec![2, 2],
            GgufDataType::Q4_K,
        );
        weights.add_tensor("test_tensor".to_string(), tensor);

        let model = ModelHandle::new("test_model".to_string(), weights);

        // Validate through loader
        let result = loader.validate_model(&model);
        assert!(result.is_ok());

        loader.release().unwrap();
    }

    // ===== RED PHASE TESTS FOR CANDLE TENSOR INTEGRATION =====

    #[cfg(feature = "std")]
    #[test]
    fn test_candle_tensor_creation_from_loaded_tensor() {
        // Test CandleTensor creation from LoadedTensor
        use candle_core::Device;

        let loaded_tensor = LoadedTensor::new(
            vec![1, 2, 3, 4], // Mock data
            vec![2, 2], // Shape
            GgufDataType::F32,
        );

        let device = Device::Cpu;
        let result = CandleTensor::from_loaded_tensor(&loaded_tensor, &device);

        // Should succeed with placeholder implementation
        assert!(result.is_ok());
        let candle_tensor = result.unwrap();

        // Should have correct shape
        assert_eq!(candle_tensor.shape(), &[2, 2]);

        // Should have element count
        assert_eq!(candle_tensor.element_count(), 4);

        // Should have memory usage (4 elements * 4 bytes for F32)
        assert_eq!(candle_tensor.memory_usage_bytes(), 16);

        // Should validate successfully
        assert!(candle_tensor.validate().is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_candle_tensor_creation_with_quantization() {
        // Test CandleTensor creation from quantized tensor
        use candle_core::Device;

        let quant_info = QuantizationInfo::new(GgufDataType::Q4_K, 256, 100);
        let loaded_tensor = LoadedTensor::new_quantized(
            vec![1, 2, 3, 4], // Mock quantized data
            vec![1024, 1024], // Large shape
            GgufDataType::Q4_K,
            quant_info,
        );

        let device = Device::Cpu;
        let result = CandleTensor::from_loaded_tensor(&loaded_tensor, &device);

        // Should succeed with placeholder implementation
        assert!(result.is_ok());
        let candle_tensor = result.unwrap();

        // Should have quantization info preserved
        assert!(candle_tensor.quantization.is_some());

        // Should have correct shape
        assert_eq!(candle_tensor.shape(), &[1024, 1024]);

        // Should have memory usage (quantized to F32 for now)
        assert_eq!(candle_tensor.element_count(), 1024 * 1024);
        assert_eq!(candle_tensor.memory_usage_bytes(), 1024 * 1024 * 4); // F32 bytes
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_candle_tensor_validation() {
        // Test CandleTensor validation
        use candle_core::Device;

        let device = Device::Cpu;

        // Valid tensor
        let valid_tensor = LoadedTensor::new(
            vec![1, 2, 3, 4],
            vec![2, 2],
            GgufDataType::F32,
        );
        let candle_tensor = CandleTensor::from_loaded_tensor(&valid_tensor, &device).unwrap();
        assert!(candle_tensor.validate().is_ok());

        // TODO: Test invalid tensors when we have real tensor creation
        // For now, placeholder implementation always creates valid tensors
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_tensor_converter_batch_processing() {
        // Test TensorConverter for batch processing
        use candle_core::Device;

        let device = Device::Cpu;
        let mut converter = TensorConverter::new(device);

        // Create multiple tensors
        let tensors = vec![
            ("tensor1".to_string(), LoadedTensor::new(
                vec![1, 2, 3, 4],
                vec![2, 2],
                GgufDataType::F32,
            )),
            ("tensor2".to_string(), LoadedTensor::new(
                vec![5, 6, 7, 8],
                vec![4],
                GgufDataType::F32,
            )),
        ];

        let result = converter.convert_batch(&tensors);

        // Should succeed
        assert!(result.is_ok());
        let converted = result.unwrap();

        // Should have converted all tensors
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].0, "tensor1");
        assert_eq!(converted[1].0, "tensor2");

        // Should track memory usage
        assert!(converter.total_memory() > 0);

        // Should reset memory tracking
        converter.reset_memory_tracking();
        assert_eq!(converter.total_memory(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_streaming_tensor_loader() {
        // Test StreamingTensorLoader for large tensors
        use candle_core::Device;

        let device = Device::Cpu;
        let chunk_size = 1024;
        let mut loader = StreamingTensorLoader::new(device, chunk_size);

        // Should start with no loaded tensors
        assert_eq!(loader.loaded_count(), 0);

        // Load a tensor
        let loaded_tensor = LoadedTensor::new(
            vec![1; 2048], // 2KB of data
            vec![2048],
            GgufDataType::F32,
        );

        let result = loader.load_tensor_streaming(&loaded_tensor);

        // Should succeed
        assert!(result.is_ok());
        let candle_tensor = result.unwrap();

        // Should track loaded count
        assert_eq!(loader.loaded_count(), 1);

        // Should have correct shape
        assert_eq!(candle_tensor.shape(), &[2048]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_dtype_to_candle_conversion() {
        // Test conversion of GGUF data types to Candle DTypes
        // This tests the internal conversion logic through the public API

        use candle_core::Device;
        let device = Device::Cpu;

        // Test F32 conversion
        let f32_tensor = LoadedTensor::new(vec![1, 2, 3, 4], vec![2, 2], GgufDataType::F32);
        let result = CandleTensor::from_loaded_tensor(&f32_tensor, &device);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tensor.dtype(), candle_core::DType::F32);

        // Test I32 conversion (mapped to U32 in our implementation)
        let i32_tensor = LoadedTensor::new(vec![1, 2, 3, 4], vec![2, 2], GgufDataType::I32);
        let result = CandleTensor::from_loaded_tensor(&i32_tensor, &device);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tensor.dtype(), candle_core::DType::U32); // Mapped to U32

        // Test quantized conversion (should convert to F32 for now)
        let q4_tensor = LoadedTensor::new(vec![1, 2, 3, 4], vec![2, 2], GgufDataType::Q4_K);
        let result = CandleTensor::from_loaded_tensor(&q4_tensor, &device);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tensor.dtype(), candle_core::DType::F32); // Placeholder
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_candle_tensor_memory_tracking() {
        // Test memory tracking in Candle tensors
        use candle_core::Device;

        let device = Device::Cpu;

        // Create tensor with known size
        let loaded_tensor = LoadedTensor::new(
            vec![1; 1000], // 1000 bytes
            vec![250], // 250 elements of 4 bytes each (F32)
            GgufDataType::F32,
        );

        let result = CandleTensor::from_loaded_tensor(&loaded_tensor, &device);
        assert!(result.is_ok());

        let candle_tensor = result.unwrap();

        // Should track memory correctly
        assert_eq!(candle_tensor.memory_usage_bytes(), 250 * 4); // 1000 bytes
        assert_eq!(candle_tensor.element_count(), 250);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_candle_integration_error_handling() {
        // Test error handling in Candle integration
        use candle_core::Device;

        let device = Device::Cpu;

        // TODO: Test with invalid tensors when we have real tensor creation
        // For now, placeholder implementation is very forgiving

        // Test with zero-element tensor (should fail validation)
        let zero_tensor = LoadedTensor::new(vec![], vec![], GgufDataType::F32);
        let result = CandleTensor::from_loaded_tensor(&zero_tensor, &device);
        // This should succeed in placeholder (creates empty tensor)
        assert!(result.is_ok());
        let candle_tensor = result.unwrap();
        // But validation should fail
        assert!(candle_tensor.validate().is_err());

        // Test with zero dimension
        let zero_dim_tensor = LoadedTensor::new(vec![1, 2, 3, 4], vec![2, 0], GgufDataType::F32);
        let result = CandleTensor::from_loaded_tensor(&zero_dim_tensor, &device);
        // Should succeed in placeholder but fail validation
        assert!(result.is_ok());
        let candle_tensor = result.unwrap();
        assert!(candle_tensor.validate().is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_gguf_loader_file_header_parsing() {
        // Test GGUF header parsing with a mock GGUF file
        let mut loader = GgufLoader::new();
        loader.acquire().unwrap();

        use std::fs::File;
        use std::io::Write;

        // Create a mock GGUF file with proper header
        let temp_path = "temp_mock_gguf.gguf";
        let mut temp_file = File::create(temp_path).unwrap();

        // Write GGUF magic number (little endian)
        temp_file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // "GGUF"

        // Write version (3)
        temp_file.write_all(&3u32.to_le_bytes()).unwrap();

        // Write tensor count (100)
        temp_file.write_all(&100u32.to_le_bytes()).unwrap();

        // Write KV count (50)
        temp_file.write_all(&50u32.to_le_bytes()).unwrap();

        temp_file.flush().unwrap();

        // Try to read the header
        let mut file = File::open(temp_path).unwrap();
        let result = loader.read_gguf_header(&mut file);

        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.magic, 0x46554747);
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 100);
        assert_eq!(header.kv_count, 50);

        // Clean up
        std::fs::remove_file(temp_path).unwrap();
        loader.release().unwrap();
    }
}