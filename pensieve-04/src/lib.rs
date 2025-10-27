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

/// Device abstraction layer for compute backends
pub mod device {
    use super::{CoreError, CoreResult, Resource, Reset, Validate};

    #[cfg(feature = "std")]
    use candle_core::{Device, Tensor};

    #[cfg(feature = "std")]
    use std::{string::String, vec::Vec, format, println};

    /// Compute device types
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DeviceType {
        Cpu,
        Metal,
        Cuda,
    }

    /// Device information and capabilities
    #[derive(Debug, Clone)]
    pub struct DeviceInfo {
        pub device_type: DeviceType,
        #[cfg(feature = "std")]
        pub name: String,
        #[cfg(not(feature = "std"))]
        pub name: &'static str,
        pub memory_mb: Option<usize>,
        pub is_available: bool,
    }

    /// Abstraction over compute devices (CPU, GPU, etc.)
    pub trait ComputeDevice: Resource + Reset + Validate {
        /// Get device information
        fn info(&self) -> &DeviceInfo;

        /// Get the underlying Candle device
        #[cfg(feature = "std")]
        fn candle_device(&self) -> &Device;

        /// Check if device is ready for computation
        fn is_ready(&self) -> bool {
            self.info().is_available && self.is_available()
        }

        /// Allocate memory for tensors
        #[cfg(feature = "std")]
        fn allocate_tensor(&self, shape: &[usize], dtype: candle_core::DType) -> CoreResult<Tensor>;

        /// Synchronize device operations
        #[cfg(feature = "std")]
        fn synchronize(&self) -> CoreResult<()>;
    }

    /// Device manager for handling multiple compute devices
    pub trait DeviceManager: Resource + Reset + Validate {
        type Device: ComputeDevice;

        /// Create a new device manager
        fn new() -> CoreResult<Self>
        where
            Self: Sized;

        /// Get the best available device
        fn get_best_device(&self) -> CoreResult<&Self::Device>;

        /// Get device by type
        fn get_device_by_type(&self, device_type: DeviceType) -> CoreResult<&Self::Device>;

        /// List all available devices
        #[cfg(feature = "std")]
        fn list_devices(&self) -> CoreResult<Vec<&DeviceInfo>>;

        /// List all available devices (no_std version)
        #[cfg(not(feature = "std"))]
        fn list_devices(&self) -> CoreResult<usize>;

        /// Add a device to the manager
        fn add_device(&mut self, device: Self::Device) -> CoreResult<()>;
    }
}

#[cfg(feature = "std")]
pub mod concrete_devices {
    use super::{
        device::{ComputeDevice, DeviceInfo, DeviceManager, DeviceType},
        CoreError, CoreResult, Resource, Reset, Validate,
    };
    use candle_core::{Device, Tensor, DType};
    use std::{vec::Vec, string::ToString, format};

    /// CPU device implementation
    #[derive(Debug)]
    pub struct CpuDevice {
        info: DeviceInfo,
        candle_device: Device,
        resource_available: bool,
    }

    impl CpuDevice {
        /// Create a new CPU device
        pub fn new() -> Self {
            Self {
                info: DeviceInfo {
                    device_type: DeviceType::Cpu,
                    name: "CPU".to_string(),
                    memory_mb: None, // CPU doesn't have fixed memory
                    is_available: true,
                },
                candle_device: Device::Cpu,
                resource_available: true,
            }
        }
    }

    impl Resource for CpuDevice {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.resource_available
        }

        fn acquire(&mut self) -> core::result::Result<(), Self::Error> {
            self.resource_available = true;
            Ok(())
        }

        fn release(&mut self) -> core::result::Result<(), Self::Error> {
            self.resource_available = false;
            Ok(())
        }
    }

    impl Reset for CpuDevice {
        fn reset(&mut self) {
            self.resource_available = true;
        }
    }

    impl Validate for CpuDevice {
        fn validate(&self) -> CoreResult<()> {
            if !self.resource_available {
                return Err(CoreError::Unavailable("CPU device not available"));
            }
            Ok(())
        }
    }

    impl ComputeDevice for CpuDevice {
        fn info(&self) -> &DeviceInfo {
            &self.info
        }

        fn candle_device(&self) -> &Device {
            &self.candle_device
        }

        fn allocate_tensor(&self, shape: &[usize], dtype: DType) -> CoreResult<Tensor> {
            Tensor::zeros(shape, dtype, &self.candle_device)
                .map_err(|_| CoreError::Generic("tensor allocation failed"))
        }

        fn synchronize(&self) -> CoreResult<()> {
            // CPU doesn't need synchronization
            Ok(())
        }
    }

    /// Metal device implementation (macOS only)
    #[cfg(target_os = "macos")]
    #[derive(Debug)]
    pub struct MetalDevice {
        info: DeviceInfo,
        candle_device: Device,
        resource_available: bool,
    }

    #[cfg(target_os = "macos")]
    impl MetalDevice {
        /// Try to create a Metal device, fallback to None if unavailable
        pub fn try_new() -> Option<Self> {
            match Device::new_metal(0) {
                Ok(device) => Some(Self {
                    info: DeviceInfo {
                        device_type: DeviceType::Metal,
                        name: "Apple GPU".to_string(),
                        memory_mb: None, // Could query this but not essential now
                        is_available: true,
                    },
                    candle_device: device,
                    resource_available: true,
                }),
                Err(_) => None,
            }
        }
    }

    #[cfg(target_os = "macos")]
    impl Resource for MetalDevice {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.resource_available
        }

        fn acquire(&mut self) -> core::result::Result<(), Self::Error> {
            self.resource_available = true;
            Ok(())
        }

        fn release(&mut self) -> core::result::Result<(), Self::Error> {
            self.resource_available = false;
            Ok(())
        }
    }

    #[cfg(target_os = "macos")]
    impl Reset for MetalDevice {
        fn reset(&mut self) {
            self.resource_available = true;
        }
    }

    #[cfg(target_os = "macos")]
    impl Validate for MetalDevice {
        fn validate(&self) -> CoreResult<()> {
            if !self.resource_available {
                return Err(CoreError::Unavailable("Metal device not available"));
            }
            Ok(())
        }
    }

    #[cfg(target_os = "macos")]
    impl ComputeDevice for MetalDevice {
        fn info(&self) -> &DeviceInfo {
            &self.info
        }

        fn candle_device(&self) -> &Device {
            &self.candle_device
        }

        fn allocate_tensor(&self, shape: &[usize], dtype: DType) -> CoreResult<Tensor> {
            Tensor::zeros(shape, dtype, &self.candle_device)
                .map_err(|_| CoreError::Generic("tensor allocation failed"))
        }

        fn synchronize(&self) -> CoreResult<()> {
            // Metal device synchronization
            // For now, just return Ok - actual sync can be added later
            Ok(())
        }
    }

    /// Basic device manager implementation
    #[derive(Debug)]
    pub struct BasicDeviceManager {
        cpu_device: CpuDevice,
        #[cfg(target_os = "macos")]
        metal_device: Option<MetalDevice>,
    }

    impl BasicDeviceManager {
        /// Create a new device manager and detect available devices
        pub fn new() -> CoreResult<Self> {
            Ok(Self {
                cpu_device: CpuDevice::new(),
                #[cfg(target_os = "macos")]
                metal_device: MetalDevice::try_new(),
            })
        }

        /// Get the CPU device (always available)
        pub fn cpu_device(&self) -> &CpuDevice {
            &self.cpu_device
        }

        /// Get the Metal device if available
        #[cfg(target_os = "macos")]
        pub fn metal_device(&self) -> Option<&MetalDevice> {
            self.metal_device.as_ref()
        }
    }

    impl Resource for BasicDeviceManager {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            let cpu_available = self.cpu_device.is_available();
            #[cfg(target_os = "macos")]
            let metal_available = self.metal_device.as_ref().map_or(true, |d| d.is_available());
            #[cfg(not(target_os = "macos"))]
            let metal_available = true;

            cpu_available && metal_available
        }

        fn acquire(&mut self) -> core::result::Result<(), Self::Error> {
            self.cpu_device.acquire()?;
            #[cfg(target_os = "macos")]
            if let Some(ref mut device) = self.metal_device {
                device.acquire()?;
            }
            Ok(())
        }

        fn release(&mut self) -> core::result::Result<(), Self::Error> {
            self.cpu_device.release()?;
            #[cfg(target_os = "macos")]
            if let Some(ref mut device) = self.metal_device {
                device.release()?;
            }
            Ok(())
        }
    }

    impl Reset for BasicDeviceManager {
        fn reset(&mut self) {
            self.cpu_device.reset();
            #[cfg(target_os = "macos")]
            if let Some(ref mut device) = self.metal_device {
                device.reset();
            }
        }
    }

    impl Validate for BasicDeviceManager {
        fn validate(&self) -> CoreResult<()> {
            self.cpu_device.validate()?;
            #[cfg(target_os = "macos")]
            if let Some(ref device) = self.metal_device {
                device.validate()?;
            }
            Ok(())
        }
    }

    impl DeviceManager for BasicDeviceManager {
        type Device = CpuDevice;

        fn new() -> CoreResult<Self>
        where
            Self: Sized,
        {
            Self::new()
        }

        fn get_best_device(&self) -> CoreResult<&Self::Device> {
            // For now, just return the CPU device as a simple implementation
            Ok(&self.cpu_device)
        }

        fn get_device_by_type(&self, device_type: DeviceType) -> CoreResult<&Self::Device> {
            match device_type {
                DeviceType::Cpu => Ok(&self.cpu_device),
                DeviceType::Metal => Err(CoreError::NotFound("Metal device not available")),
                DeviceType::Cuda => Err(CoreError::NotFound("CUDA not supported yet")),
            }
        }

        fn list_devices(&self) -> CoreResult<Vec<&DeviceInfo>> {
            let mut devices = Vec::new();
            devices.push(&self.cpu_device.info);

            #[cfg(target_os = "macos")]
            if let Some(ref metal_device) = self.metal_device {
                devices.push(&metal_device.info);
            }

            Ok(devices)
        }

        fn add_device(&mut self, _device: Self::Device) -> CoreResult<()> {
            // For now, we don't support adding custom devices
            Err(CoreError::Unsupported("Adding custom devices not supported"))
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

    #[cfg(feature = "std")]
    use std::{string::{String, ToString}, format};

    #[test]
    fn test_device_type_equality() {
        assert_eq!(device::DeviceType::Cpu, device::DeviceType::Cpu);
        assert_ne!(device::DeviceType::Cpu, device::DeviceType::Metal);
        assert_ne!(device::DeviceType::Metal, device::DeviceType::Cuda);
    }

    #[test]
    fn test_device_info_creation() {
        let info = device::DeviceInfo {
            device_type: device::DeviceType::Cpu,
            #[cfg(feature = "std")]
            name: "Apple M1 CPU".to_string(),
            #[cfg(not(feature = "std"))]
            name: "Apple M1 CPU",
            memory_mb: Some(8192),
            is_available: true,
        };

        assert_eq!(info.device_type, device::DeviceType::Cpu);
        assert_eq!(info.name, "Apple M1 CPU");
        assert_eq!(info.memory_mb, Some(8192));
        assert!(info.is_available);
    }

    #[test]
    fn test_device_info_debug_format() {
        let info = device::DeviceInfo {
            device_type: device::DeviceType::Metal,
            #[cfg(feature = "std")]
            name: "Apple M1 GPU".to_string(),
            #[cfg(not(feature = "std"))]
            name: "Apple M1 GPU",
            memory_mb: Some(16384),
            is_available: false,
        };

        #[cfg(feature = "std")]
        {
            let debug_str = format!("{:?}", info);
            assert!(debug_str.contains("Metal"));
            assert!(debug_str.contains("Apple M1 GPU"));
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_candle_device_import() {
        // This test verifies that candle-core is properly linked
        use candle_core::Device;

        let device = Device::Cpu;
        match device {
            Device::Cpu => assert!(true), // CPU device created successfully
            Device::Metal(_) => assert!(true), // Metal device created successfully
            Device::Cuda(_) => assert!(true), // CUDA device created successfully
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_tensor_creation_failure() {
        // This test should fail initially because we haven't implemented
        // the concrete device types yet
        use candle_core::{Device, Tensor, DType};

        let device = Device::Cpu;
        let result = Tensor::zeros((2, 3), DType::F32, &device);

        // This should work with Candle, showing the integration is functional
        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.dims(), &[2, 3]);
    }

    #[test]
    fn test_device_manager_trait_exists() {
        // Verify the DeviceManager trait is properly defined
        use super::device::DeviceManager;

        // This test just verifies the trait exists and compiles
        // Concrete implementations will be added in the GREEN step
        assert!(true);
    }

    #[test]
    fn test_compute_device_trait_exists() {
        // Verify the ComputeDevice trait is properly defined
        use super::device::ComputeDevice;

        // This test just verifies the trait exists and compiles
        // Concrete implementations will be added in the GREEN step
        assert!(true);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_cpu_device_creation() {
        use super::concrete_devices::CpuDevice;
        use super::device::ComputeDevice;

        let device = CpuDevice::new();
        assert!(device.is_available());
        assert_eq!(device.info().device_type, device::DeviceType::Cpu);
        assert_eq!(device.info().name, "CPU");
        assert!(device.info().is_available);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_cpu_device_tensor_allocation() {
        use super::concrete_devices::CpuDevice;
        use super::device::ComputeDevice;
        use candle_core::DType;

        let device = CpuDevice::new();
        let result = device.allocate_tensor(&[2, 3], DType::F32);

        assert!(result.is_ok());
        let tensor = result.unwrap();
        assert_eq!(tensor.dims(), &[2, 3]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_cpu_device_validation() {
        use super::concrete_devices::CpuDevice;
        use super::Validate;

        let device = CpuDevice::new();
        assert!(device.validate().is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_basic_device_manager_creation() {
        use super::concrete_devices::BasicDeviceManager;

        let manager = BasicDeviceManager::new();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(manager.is_available());
        assert!(manager.validate().is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_device_manager_list_devices() {
        use super::concrete_devices::BasicDeviceManager;
        use super::device::DeviceManager;

        let manager = BasicDeviceManager::new().unwrap();
        let devices = manager.list_devices();

        assert!(devices.is_ok());
        let devices = devices.unwrap();
        assert!(!devices.is_empty());

        // Should have at least CPU device
        assert!(devices.iter().any(|d| d.device_type == device::DeviceType::Cpu));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_device_manager_get_best_device() {
        use super::concrete_devices::BasicDeviceManager;
        use super::device::{DeviceManager, ComputeDevice};

        let manager = BasicDeviceManager::new().unwrap();
        let device = manager.get_best_device();

        assert!(device.is_ok());
        let device = device.unwrap();
        assert_eq!(device.info().device_type, device::DeviceType::Cpu);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_device_manager_get_cpu_device() {
        use super::concrete_devices::BasicDeviceManager;
        use super::device::{DeviceManager, ComputeDevice};

        let manager = BasicDeviceManager::new().unwrap();
        let device = manager.get_device_by_type(device::DeviceType::Cpu);

        assert!(device.is_ok());
        let device = device.unwrap();
        assert_eq!(device.info().device_type, device::DeviceType::Cpu);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_device_manager_get_metal_device_should_fail() {
        use super::concrete_devices::BasicDeviceManager;
        use super::device::DeviceManager;

        let manager = BasicDeviceManager::new().unwrap();
        let device = manager.get_device_by_type(device::DeviceType::Metal);

        assert!(device.is_err());
    }

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