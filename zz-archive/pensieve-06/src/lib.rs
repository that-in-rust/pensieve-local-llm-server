//! Pensieve Metal - Metal GPU implementations
//!
//! This is the Layer 2 (L2) Metal crate that provides:
//! - Metal GPU device management
//! - GPU buffer and memory management
//! - Metal kernel compilation and execution
//! - Cross-platform GPU abstraction layer
//!
//! Depends on L1 (pensieve-07_core) and L2 (pensieve-04_engine) crates.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

// Import alloc types for no_std compatibility
extern crate alloc;

// Re-export from dependencies
pub use pensieve_07_core::{
    error::{CoreError, CoreResult},
    traits::{Resource, Reset, Validate},
    Result,
};

pub use pensieve_04_engine::{
    engine::{InferenceEngine, Sampler},
    CpuEngine, CpuSampler, CpuToken,
};

/// Core GPU device traits
pub mod device {
    use super::{CoreResult, Resource, Reset, Validate};
    use alloc::{string::String, vec::Vec};

    /// Trait for GPU device abstraction
    pub trait GpuDevice: Resource + Reset + Validate {
        /// Device identifier
        type DeviceId: Clone + PartialEq;

        /// Memory allocation type
        type Memory: Resource + Reset;

        /// Command queue for GPU operations
        type CommandQueue: Resource;

        /// Get device name
        fn name(&self) -> &str;

        /// Get device identifier
        fn device_id(&self) -> Self::DeviceId;

        /// Get total memory in bytes
        fn total_memory(&self) -> u64;

        /// Get available memory in bytes
        fn available_memory(&self) -> u64;

        /// Check if device supports specific operations
        fn supports_operation(&self, operation: GpuOperation) -> bool;

        /// Allocate GPU memory
        fn allocate_memory(&mut self, size: u64) -> CoreResult<Self::Memory>;

        /// Create command queue
        fn create_command_queue(&mut self) -> CoreResult<Self::CommandQueue>;

        /// Check if device is ready for operations
        fn is_ready(&self) -> bool {
            self.is_available()
        }
    }

    /// GPU operations that devices can support
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GpuOperation {
        /// Matrix multiplication
        MatrixMultiplication,
        /// Convolution operations
        Convolution,
        /// Attention mechanisms
        Attention,
        /// Element-wise operations
        ElementWise,
        /// Reduction operations
        Reduction,
        /// Custom kernels
        CustomKernel,
    }

    /// GPU device information
    #[derive(Debug, Clone)]
    pub struct GpuDeviceInfo {
        /// Device name
        pub name: String,
        /// Device type
        pub device_type: DeviceType,
        /// Total memory in bytes
        pub total_memory: u64,
        /// Maximum buffer size
        pub max_buffer_size: u64,
        /// Supported operations
        pub supported_operations: Vec<GpuOperation>,
    }

    /// GPU device types
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum DeviceType {
        /// Integrated GPU
        Integrated,
        /// Discrete GPU
        Discrete,
        /// Software/emulated GPU
        Software,
    }

    impl GpuDeviceInfo {
        /// Create new GPU device info
        pub fn new(
            name: String,
            device_type: DeviceType,
            total_memory: u64,
            max_buffer_size: u64,
        ) -> Self {
            Self {
                name,
                device_type,
                total_memory,
                max_buffer_size,
                supported_operations: Vec::new(),
            }
        }

        /// Add supported operation
        pub fn with_supported_operation(mut self, operation: GpuOperation) -> Self {
            self.supported_operations.push(operation);
            self
        }

        /// Check if operation is supported
        pub fn supports_operation(&self, operation: GpuOperation) -> bool {
            self.supported_operations.contains(&operation)
        }
    }
}

/// GPU buffer and memory management
pub mod buffer {
    use super::{CoreResult, Resource, Reset, Validate};
    use alloc::vec::Vec;

    /// Trait for GPU buffer management
    pub trait GpuBuffer: Resource + Reset + Validate {
        /// Buffer identifier
        type BufferId: Clone + PartialEq;

        /// Get buffer size in bytes
        fn size(&self) -> u64;

        /// Get buffer identifier
        fn buffer_id(&self) -> Self::BufferId;

        /// Check if buffer is mapped to CPU memory
        fn is_mapped(&self) -> bool;

        /// Map buffer to CPU memory
        fn map(&mut self) -> CoreResult<*mut u8>;

        /// Unmap buffer from CPU memory
        fn unmap(&mut self) -> CoreResult<()>;

        /// Copy data from host to device
        fn copy_from_host(&mut self, data: &[u8]) -> CoreResult<()>;

        /// Copy data from device to host
        fn copy_to_host(&self, offset: u64, size: u64) -> CoreResult<Vec<u8>>;

        /// Fill buffer with pattern
        fn fill(&mut self, pattern: u8) -> CoreResult<()>;
    }

    /// Buffer usage flags
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum BufferUsage {
        /// Read-only storage
        ReadOnly,
        /// Write-only storage
        WriteOnly,
        /// Read-write storage
        ReadWrite,
        /// Index buffer
        IndexBuffer,
        /// Vertex buffer
        VertexBuffer,
        /// Uniform buffer
        UniformBuffer,
    }

    /// Memory management statistics
    #[derive(Debug, Clone)]
    pub struct MemoryStats {
        /// Total allocated memory
        pub total_allocated: u64,
        /// Currently used memory
        pub used_memory: u64,
        /// Peak memory usage
        pub peak_usage: u64,
        /// Number of allocations
        pub allocation_count: usize,
    }

    impl MemoryStats {
        /// Create new memory stats
        pub fn new() -> Self {
            Self {
                total_allocated: 0,
                used_memory: 0,
                peak_usage: 0,
                allocation_count: 0,
            }
        }

        /// Get memory utilization percentage
        pub fn utilization(&self) -> f32 {
            if self.total_allocated == 0 {
                0.0
            } else {
                (self.used_memory as f32 / self.total_allocated as f32) * 100.0
            }
        }
    }

    impl Default for MemoryStats {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Metal kernel compilation and execution
pub mod kernel {
    use super::{
        buffer::GpuBuffer, CoreResult, Resource, Reset, Validate,
    };
    

    /// Trait for GPU kernel management
    pub trait GpuKernel: Resource + Reset + Validate {
        /// Kernel identifier
        type KernelId: Clone + PartialEq;

        /// Get kernel name
        fn name(&self) -> &str;

        /// Get kernel identifier
        fn kernel_id(&self) -> Self::KernelId;

        /// Check if kernel is compiled
        fn is_compiled(&self) -> bool;

        /// Compile kernel from source
        fn compile(&mut self, source: &str) -> CoreResult<()>;

        /// Execute kernel with given parameters
        fn execute<B: GpuBuffer>(
            &mut self,
            grid_size: (u32, u32, u32),
            threadgroup_size: (u32, u32, u32),
            buffers: &mut [&mut B],
        ) -> CoreResult<()>;

        /// Get kernel compilation log
        fn compilation_log(&self) -> Option<&str>;
    }

    /// Kernel execution parameters
    #[derive(Debug, Clone)]
    pub struct KernelExecutionParams {
        /// Grid dimensions
        pub grid_size: (u32, u32, u32),
        /// Threadgroup dimensions
        pub threadgroup_size: (u32, u32, u32),
        /// Execution priority
        pub priority: ExecutionPriority,
    }

    /// Kernel execution priority
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum ExecutionPriority {
        /// Low priority
        Low,
        /// Normal priority
        Normal,
        /// High priority
        High,
        /// Critical priority
        Critical,
    }

    /// Kernel source templates for common operations
    pub struct KernelTemplates;

    impl KernelTemplates {
        /// Matrix multiplication kernel
        pub fn matrix_multiply() -> &'static str {
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void matrix_multiply(
                device float *result [[buffer(0)]],
                device float *a [[buffer(1)]],
                device float *b [[buffer(2)]],
                constant uint &dim [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;

                if (row >= dim || col >= dim) return;

                float sum = 0.0;
                for (uint k = 0; k < dim; k++) {
                    sum += a[row * dim + k] * b[k * dim + col];
                }
                result[row * dim + col] = sum;
            }
            "#
        }

        /// Element-wise addition kernel
        pub fn element_add() -> &'static str {
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void element_add(
                device float *result [[buffer(0)]],
                device float *a [[buffer(1)]],
                device float *b [[buffer(2)]],
                uint index [[thread_position_in_grid]]
            ) {
                result[index] = a[index] + b[index];
            }
            "#
        }

        /// Simple attention kernel (placeholder)
        pub fn attention() -> &'static str {
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void attention(
                device float *output [[buffer(0)]],
                device float *query [[buffer(1)]],
                device float *key [[buffer(2)]],
                device float *value [[buffer(3)]],
                constant uint &seq_len [[buffer(4)]],
                constant uint &head_dim [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                // Simplified attention computation
                uint seq = gid.x;
                uint head = gid.y;

                if (seq >= seq_len) return;

                float sum = 0.0;
                for (uint i = 0; i < seq_len; i++) {
                    float q = query[seq * head_dim + head];
                    float k = key[i * head_dim + head];
                    float v = value[i * head_dim + head];

                    sum += q * k * v; // Simplified attention
                }

                output[seq * head_dim + head] = sum;
            }
            "#
        }
    }
}

/// Memory pool management
pub mod memory_pool {
    use super::{
        buffer::{BufferUsage, GpuBuffer, MemoryStats}, CoreResult, Resource, Reset, Validate,
    };
    

    /// Trait for GPU memory pool management
    pub trait GpuMemoryPool: Resource + Reset + Validate {
        /// Buffer type
        type Buffer: GpuBuffer;

        /// Allocate buffer from pool
        fn allocate_buffer(
            &mut self,
            size: u64,
            usage: BufferUsage,
        ) -> CoreResult<Self::Buffer>;

        /// Deallocate buffer back to pool
        fn deallocate_buffer(&mut self, buffer: Self::Buffer) -> CoreResult<()>;

        /// Get memory statistics
        fn memory_stats(&self) -> MemoryStats;

        /// Clear all allocations
        fn clear(&mut self) -> CoreResult<()>;

        /// Compact memory
        fn compact(&mut self) -> CoreResult<()>;
    }

    /// Memory pool configuration
    #[derive(Debug, Clone)]
    pub struct MemoryPoolConfig {
        /// Initial pool size in bytes
        pub initial_size: u64,
        /// Maximum pool size in bytes
        pub max_size: u64,
        /// Growth factor when expanding
        pub growth_factor: f32,
        /// Enable memory compaction
        pub enable_compaction: bool,
    }

    impl MemoryPoolConfig {
        /// Create default memory pool config
        pub fn default() -> Self {
            Self {
                initial_size: 64 * 1024 * 1024, // 64MB
                max_size: 1024 * 1024 * 1024,   // 1GB
                growth_factor: 1.5,
                enable_compaction: true,
            }
        }

        /// Create minimal memory pool config
        pub fn minimal() -> Self {
            Self {
                initial_size: 16 * 1024 * 1024, // 16MB
                max_size: 256 * 1024 * 1024,    // 256MB
                growth_factor: 2.0,
                enable_compaction: true,
            }
        }

        /// Create large memory pool config
        pub fn large() -> Self {
            Self {
                initial_size: 512 * 1024 * 1024, // 512MB
                max_size: 8 * 1024 * 1024 * 1024, // 8GB
                growth_factor: 1.25,
                enable_compaction: true,
            }
        }
    }

    impl Default for MemoryPoolConfig {
        fn default() -> Self {
            Self::default()
        }
    }
}

/// Mock implementations for testing
pub mod mock {
    use super::{
        buffer::{BufferUsage, GpuBuffer, MemoryStats},
        device::{DeviceType, GpuDevice, GpuOperation},
        kernel::GpuKernel,
        memory_pool::{GpuMemoryPool, MemoryPoolConfig},
        CoreError, CoreResult, Resource, Reset, Validate,
    };
    use alloc::{format, string::{String, ToString}, vec::Vec, vec};

    /// Mock GPU device for testing
    #[derive(Debug)]
    pub struct MockGpuDevice {
        name: String,
        device_type: DeviceType,
        total_memory: u64,
        used_memory: u64,
        available: bool,
    }

    impl MockGpuDevice {
        /// Create a new mock GPU device
        pub fn new(name: String, device_type: DeviceType, total_memory: u64) -> Self {
            Self {
                name,
                device_type,
                total_memory,
                used_memory: 0,
                available: false,
            }
        }

        /// Get device type
        pub fn device_type(&self) -> DeviceType {
            self.device_type
        }

        /// Create default integrated GPU
        pub fn integrated() -> Self {
            Self::new(
                "Mock Integrated GPU".to_string(),
                DeviceType::Integrated,
                1024 * 1024 * 1024, // 1GB
            )
        }

        /// Create default discrete GPU
        pub fn discrete() -> Self {
            Self::new(
                "Mock Discrete GPU".to_string(),
                DeviceType::Discrete,
                8 * 1024 * 1024 * 1024, // 8GB
            )
        }
    }

    impl Resource for MockGpuDevice {
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

    impl Reset for MockGpuDevice {
        fn reset(&mut self) {
            self.used_memory = 0;
            self.available = false;
        }
    }

    impl Validate for MockGpuDevice {
        fn validate(&self) -> CoreResult<()> {
            if self.name.is_empty() {
                return Err(CoreError::InvalidInput("device name cannot be empty"));
            }
            if self.total_memory == 0 {
                return Err(CoreError::InvalidInput("total memory must be > 0"));
            }
            if self.used_memory > self.total_memory {
                return Err(CoreError::InvalidInput("used memory exceeds total"));
            }
            Ok(())
        }
    }

    // Device identifier for mock
    #[derive(Debug, Clone, PartialEq)]
    pub struct MockDeviceId(pub String);

    impl GpuDevice for MockGpuDevice {
        type DeviceId = MockDeviceId;
        type Memory = MockGpuMemory;
        type CommandQueue = MockCommandQueue;

        fn name(&self) -> &str {
            &self.name
        }

        fn device_id(&self) -> Self::DeviceId {
            MockDeviceId(self.name.clone())
        }

        fn total_memory(&self) -> u64 {
            self.total_memory
        }

        fn available_memory(&self) -> u64 {
            self.total_memory - self.used_memory
        }

        fn supports_operation(&self, operation: GpuOperation) -> bool {
            match self.device_type {
                DeviceType::Integrated => matches!(
                    operation,
                    GpuOperation::ElementWise | GpuOperation::Reduction
                ),
                DeviceType::Discrete => true, // Supports all operations
                DeviceType::Software => matches!(
                    operation,
                    GpuOperation::ElementWise
                ),
            }
        }

        fn allocate_memory(&mut self, size: u64) -> CoreResult<Self::Memory> {
            if self.used_memory + size > self.total_memory {
                return Err(CoreError::Unavailable("insufficient GPU memory"));
            }
            self.used_memory += size;
            Ok(MockGpuMemory::new(size))
        }

        fn create_command_queue(&mut self) -> CoreResult<Self::CommandQueue> {
            Ok(MockCommandQueue::new())
        }
    }

    /// Mock GPU memory
    #[derive(Debug)]
    pub struct MockGpuMemory {
        size: u64,
        data: Vec<u8>,
    }

    impl MockGpuMemory {
        pub fn new(size: u64) -> Self {
            Self {
                size,
                data: vec![0; size as usize],
            }
        }

        /// Get memory size
        pub fn size(&self) -> u64 {
            self.size
        }

        pub fn data(&self) -> &[u8] {
            &self.data
        }

        pub fn data_mut(&mut self) -> &mut [u8] {
            &mut self.data
        }
    }

    impl Resource for MockGpuMemory {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            !self.data.is_empty()
        }

        fn acquire(&mut self) -> CoreResult<()> {
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            Ok(())
        }
    }

    impl Reset for MockGpuMemory {
        fn reset(&mut self) {
            self.data.fill(0);
        }
    }

    impl Validate for MockGpuMemory {
        fn validate(&self) -> CoreResult<()> {
            if self.size == 0 {
                return Err(CoreError::InvalidInput("memory size must be > 0"));
            }
            if self.data.len() != self.size as usize {
                return Err(CoreError::InvalidInput("data size mismatch"));
            }
            Ok(())
        }
    }

    /// Mock command queue
    #[derive(Debug)]
    pub struct MockCommandQueue {
        ready: bool,
    }

    impl MockCommandQueue {
        pub fn new() -> Self {
            Self { ready: false }
        }
    }

    impl Resource for MockCommandQueue {
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

    impl Reset for MockCommandQueue {
        fn reset(&mut self) {
            self.ready = false;
        }
    }

    impl Validate for MockCommandQueue {
        fn validate(&self) -> CoreResult<()> {
            Ok(())
        }
    }

    /// Mock GPU buffer
    #[derive(Debug)]
    pub struct MockGpuBuffer {
        id: String,
        size: u64,
        data: Vec<u8>,
        usage: BufferUsage,
        mapped: bool,
    }

    impl MockGpuBuffer {
        pub fn new(id: String, size: u64, usage: BufferUsage) -> Self {
            Self {
                id,
                size,
                data: vec![0; size as usize],
                usage,
                mapped: false,
            }
        }

        pub fn data(&self) -> &[u8] {
            &self.data
        }

        pub fn data_mut(&mut self) -> &mut [u8] {
            &mut self.data
        }
    }

    // Buffer identifier for mock
    #[derive(Debug, Clone, PartialEq)]
    pub struct MockBufferId(pub String);

    impl GpuBuffer for MockGpuBuffer {
        type BufferId = MockBufferId;

        fn size(&self) -> u64 {
            self.size
        }

        fn buffer_id(&self) -> Self::BufferId {
            MockBufferId(self.id.clone())
        }

        fn is_mapped(&self) -> bool {
            self.mapped
        }

        fn map(&mut self) -> CoreResult<*mut u8> {
            self.mapped = true;
            Ok(self.data.as_mut_ptr())
        }

        fn unmap(&mut self) -> CoreResult<()> {
            self.mapped = false;
            Ok(())
        }

        fn copy_from_host(&mut self, data: &[u8]) -> CoreResult<()> {
            if data.len() > self.data.len() {
                return Err(CoreError::InvalidInput("data too large for buffer"));
            }
            self.data[..data.len()].copy_from_slice(data);
            Ok(())
        }

        fn copy_to_host(&self, offset: u64, size: u64) -> CoreResult<Vec<u8>> {
            let start = offset as usize;
            let end = start + size as usize;
            if end > self.data.len() {
                return Err(CoreError::InvalidInput("requested range out of bounds"));
            }
            Ok(self.data[start..end].to_vec())
        }

        fn fill(&mut self, pattern: u8) -> CoreResult<()> {
            self.data.fill(pattern);
            Ok(())
        }
    }

    impl Resource for MockGpuBuffer {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            !self.data.is_empty()
        }

        fn acquire(&mut self) -> CoreResult<()> {
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            Ok(())
        }
    }

    impl Reset for MockGpuBuffer {
        fn reset(&mut self) {
            self.data.fill(0);
            self.mapped = false;
        }
    }

    impl Validate for MockGpuBuffer {
        fn validate(&self) -> CoreResult<()> {
            if self.size == 0 {
                return Err(CoreError::InvalidInput("buffer size must be > 0"));
            }
            if self.data.len() != self.size as usize {
                return Err(CoreError::InvalidInput("data size mismatch"));
            }
            Ok(())
        }
    }

    /// Mock GPU kernel
    #[derive(Debug)]
    pub struct MockGpuKernel {
        name: String,
        source: Option<String>,
        compiled: bool,
        compilation_log: Option<String>,
    }

    impl MockGpuKernel {
        pub fn new(name: String) -> Self {
            Self {
                name,
                source: None,
                compiled: false,
                compilation_log: None,
            }
        }

        pub fn from_template(name: String, template: &'static str) -> Self {
            Self {
                name,
                source: Some(template.to_string()),
                compiled: false,
                compilation_log: None,
            }
        }
    }

    // Kernel identifier for mock
    #[derive(Debug, Clone, PartialEq)]
    pub struct MockKernelId(pub String);

    impl GpuKernel for MockGpuKernel {
        type KernelId = MockKernelId;

        fn name(&self) -> &str {
            &self.name
        }

        fn kernel_id(&self) -> Self::KernelId {
            MockKernelId(self.name.clone())
        }

        fn is_compiled(&self) -> bool {
            self.compiled
        }

        fn compile(&mut self, source: &str) -> CoreResult<()> {
            self.source = Some(source.to_string());

            // Mock compilation - check if source contains basic Metal syntax
            if source.contains("kernel") && source.contains("device") {
                self.compiled = true;
                self.compilation_log = Some("Compilation successful".to_string());
                Ok(())
            } else {
                self.compiled = false;
                self.compilation_log = Some("Compilation failed: invalid Metal syntax".to_string());
                Err(CoreError::InvalidInput("invalid Metal kernel source"))
            }
        }

        fn execute<B: GpuBuffer>(
            &mut self,
            _grid_size: (u32, u32, u32),
            _threadgroup_size: (u32, u32, u32),
            _buffers: &mut [&mut B],
        ) -> CoreResult<()> {
            if !self.is_compiled() {
                return Err(CoreError::Unavailable("kernel not compiled"));
            }
            // Mock execution - just pretend it succeeds
            Ok(())
        }

        fn compilation_log(&self) -> Option<&str> {
            self.compilation_log.as_deref()
        }
    }

    impl Resource for MockGpuKernel {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            self.compiled
        }

        fn acquire(&mut self) -> CoreResult<()> {
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            Ok(())
        }
    }

    impl Reset for MockGpuKernel {
        fn reset(&mut self) {
            self.compiled = false;
            self.compilation_log = None;
        }
    }

    impl Validate for MockGpuKernel {
        fn validate(&self) -> CoreResult<()> {
            if self.name.is_empty() {
                return Err(CoreError::InvalidInput("kernel name cannot be empty"));
            }
            Ok(())
        }
    }

    /// Mock memory pool
    #[derive(Debug)]
    pub struct MockGpuMemoryPool {
        config: MemoryPoolConfig,
        allocated_size: u64,
        buffers: Vec<MockGpuBuffer>,
    }

    impl MockGpuMemoryPool {
        pub fn new(config: MemoryPoolConfig) -> Self {
            Self {
                config,
                allocated_size: 0,
                buffers: Vec::new(),
            }
        }
    }

    impl GpuMemoryPool for MockGpuMemoryPool {
        type Buffer = MockGpuBuffer;

        fn allocate_buffer(
            &mut self,
            size: u64,
            usage: BufferUsage,
        ) -> CoreResult<Self::Buffer> {
            if self.allocated_size + size > self.config.max_size {
                return Err(CoreError::Unavailable("memory pool size limit exceeded"));
            }

            let id = format!("buffer_{}", self.buffers.len());
            let buffer = MockGpuBuffer::new(id, size, usage);
            self.allocated_size += size;
            self.buffers.push(buffer.clone());
            Ok(buffer)
        }

        fn deallocate_buffer(&mut self, _buffer: Self::Buffer) -> CoreResult<()> {
            // Mock deallocation - just remove from tracking
            self.allocated_size = self.allocated_size.saturating_sub(_buffer.size());
            Ok(())
        }

        fn memory_stats(&self) -> MemoryStats {
            MemoryStats {
                total_allocated: self.config.max_size,
                used_memory: self.allocated_size,
                peak_usage: self.allocated_size,
                allocation_count: self.buffers.len(),
            }
        }

        fn clear(&mut self) -> CoreResult<()> {
            self.buffers.clear();
            self.allocated_size = 0;
            Ok(())
        }

        fn compact(&mut self) -> CoreResult<()> {
            // Mock compaction - just pretend it succeeds
            Ok(())
        }
    }

    impl Resource for MockGpuMemoryPool {
        type Error = CoreError;

        fn is_available(&self) -> bool {
            true
        }

        fn acquire(&mut self) -> CoreResult<()> {
            Ok(())
        }

        fn release(&mut self) -> CoreResult<()> {
            Ok(())
        }
    }

    impl Reset for MockGpuMemoryPool {
        fn reset(&mut self) {
            self.clear().unwrap();
        }
    }

    impl Validate for MockGpuMemoryPool {
        fn validate(&self) -> CoreResult<()> {
            if self.config.initial_size == 0 {
                return Err(CoreError::InvalidInput("initial size must be > 0"));
            }
            if self.config.max_size < self.config.initial_size {
                return Err(CoreError::InvalidInput("max size must be >= initial size"));
            }
            Ok(())
        }
    }

    impl Clone for MockGpuBuffer {
        fn clone(&self) -> Self {
            Self {
                id: format!("{}_clone", self.id),
                size: self.size,
                data: self.data.clone(),
                usage: self.usage,
                mapped: false, // Clone starts unmapped
            }
        }
    }
}

// Re-export key types for convenience
pub use device::{DeviceType, GpuDevice, GpuDeviceInfo, GpuOperation};
pub use buffer::{BufferUsage, GpuBuffer, MemoryStats};
pub use kernel::{GpuKernel, KernelExecutionParams, KernelTemplates};
pub use memory_pool::{GpuMemoryPool, MemoryPoolConfig};
pub use mock::{
    MockGpuBuffer, MockGpuDevice, MockGpuKernel, MockGpuMemoryPool, MockGpuMemory,
};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{borrow::ToOwned, format, string::{String, ToString}, vec::Vec, vec};

    #[test]
    fn test_gpu_device_creation() {
        let device = MockGpuDevice::integrated();
        assert_eq!(device.name(), "Mock Integrated GPU");
        assert_eq!(device.device_type(), DeviceType::Integrated);
        assert_eq!(device.total_memory(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_device_validation() {
        let valid_device = MockGpuDevice::integrated();
        assert!(valid_device.validate().is_ok());

        let invalid_device = MockGpuDevice::new("".to_string(), DeviceType::Integrated, 0);
        assert!(invalid_device.validate().is_err());
    }

    #[test]
    fn test_gpu_device_operations() {
        let integrated = MockGpuDevice::integrated();
        assert!(integrated.supports_operation(GpuOperation::ElementWise));
        assert!(!integrated.supports_operation(GpuOperation::MatrixMultiplication));

        let discrete = MockGpuDevice::discrete();
        assert!(discrete.supports_operation(GpuOperation::MatrixMultiplication));
        assert!(discrete.supports_operation(GpuOperation::Attention));
    }

    #[test]
    fn test_gpu_device_memory_management() {
        let mut device = MockGpuDevice::integrated();
        device.acquire().unwrap();

        let memory = device.allocate_memory(1024).unwrap();
        assert_eq!(memory.size(), 1024);
        assert_eq!(device.available_memory(), device.total_memory() - 1024);

        let too_large = device.allocate_memory(device.total_memory());
        assert!(too_large.is_err());
    }

    #[test]
    fn test_gpu_device_resource_management() {
        let mut device = MockGpuDevice::integrated();
        assert!(!device.is_available());

        device.acquire().unwrap();
        assert!(device.is_available());

        device.release().unwrap();
        assert!(!device.is_available());

        device.reset();
        assert!(!device.is_available());
    }

    #[test]
    fn test_gpu_buffer_operations() {
        let mut buffer = MockGpuBuffer::new("test".to_string(), 1024, BufferUsage::ReadWrite);

        assert!(!buffer.is_mapped());

        let data = vec![1, 2, 3, 4];
        buffer.copy_from_host(&data).unwrap();
        assert_eq!(&buffer.data()[..4], &data);

        let copied = buffer.copy_to_host(0, 4).unwrap();
        assert_eq!(copied, data);

        buffer.fill(42).unwrap();
        assert_eq!(buffer.data()[0], 42);
    }

    #[test]
    fn test_gpu_buffer_mapping() {
        let mut buffer = MockGpuBuffer::new("test".to_string(), 1024, BufferUsage::ReadWrite);

        let ptr = buffer.map().unwrap();
        assert!(!ptr.is_null());
        assert!(buffer.is_mapped());

        buffer.unmap().unwrap();
        assert!(!buffer.is_mapped());
    }

    #[test]
    fn test_gpu_kernel_compilation() {
        let mut kernel = MockGpuKernel::from_template("test".to_string(), KernelTemplates::matrix_multiply());

        assert!(!kernel.is_compiled());

        kernel.compile(KernelTemplates::matrix_multiply()).unwrap();
        assert!(kernel.is_compiled());

        let log = kernel.compilation_log().unwrap();
        assert!(log.contains("successful"));
    }

    #[test]
    fn test_gpu_kernel_compilation_failure() {
        let mut kernel = MockGpuKernel::new("test".to_string());

        let result = kernel.compile("invalid code");
        assert!(result.is_err());
        assert!(!kernel.is_compiled());

        let log = kernel.compilation_log().unwrap();
        assert!(log.contains("failed"));
    }

    #[test]
    fn test_gpu_kernel_execution() {
        let mut kernel = MockGpuKernel::from_template("test".to_string(), KernelTemplates::element_add());
        kernel.compile(KernelTemplates::element_add()).unwrap();

        let mut buffer = MockGpuBuffer::new("test".to_string(), 1024, BufferUsage::ReadWrite);

        let result = kernel.execute((1, 1, 1), (1, 1, 1), &mut [&mut buffer]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_kernel_execution_without_compilation() {
        let mut kernel = MockGpuKernel::new("test".to_string());
        let mut buffer = MockGpuBuffer::new("test".to_string(), 1024, BufferUsage::ReadWrite);

        let result = kernel.execute((1, 1, 1), (1, 1, 1), &mut [&mut buffer]);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig::default();
        assert_eq!(config.initial_size, 64 * 1024 * 1024);
        assert_eq!(config.max_size, 1024 * 1024 * 1024);
        assert_eq!(config.growth_factor, 1.5);
        assert!(config.enable_compaction);

        let minimal = MemoryPoolConfig::minimal();
        assert_eq!(minimal.initial_size, 16 * 1024 * 1024);
        assert_eq!(minimal.max_size, 256 * 1024 * 1024);
    }

    #[test]
    fn test_memory_pool_operations() {
        let mut pool = MockGpuMemoryPool::new(MemoryPoolConfig::default());

        let buffer = pool.allocate_buffer(1024, BufferUsage::ReadWrite).unwrap();
        assert_eq!(buffer.size(), 1024);

        let stats = pool.memory_stats();
        assert_eq!(stats.used_memory, 1024);
        assert_eq!(stats.allocation_count, 1);

        pool.deallocate_buffer(buffer).unwrap();
        let stats_after = pool.memory_stats();
        assert_eq!(stats_after.used_memory, 0);
        assert_eq!(stats_after.allocation_count, 1); // Still tracked
    }

    #[test]
    fn test_memory_pool_limits() {
        let mut pool = MockGpuMemoryPool::new(MemoryPoolConfig::minimal());

        // Allocate up to the limit
        let buffer1 = pool.allocate_buffer(100 * 1024 * 1024, BufferUsage::ReadWrite).unwrap();
        let buffer2 = pool.allocate_buffer(100 * 1024 * 1024, BufferUsage::ReadWrite).unwrap();

        // This should fail due to exceeding max size
        let buffer3 = pool.allocate_buffer(100 * 1024 * 1024, BufferUsage::ReadWrite);
        assert!(buffer3.is_err());

        pool.deallocate_buffer(buffer1).unwrap();
        pool.deallocate_buffer(buffer2).unwrap();
    }

    #[test]
    fn test_memory_stats_utilization() {
        let mut stats = MemoryStats::new();
        assert_eq!(stats.utilization(), 0.0);

        stats.total_allocated = 1000;
        stats.used_memory = 500;
        assert_eq!(stats.utilization(), 50.0);
    }

    #[test]
    fn test_gpu_device_info() {
        let info = GpuDeviceInfo::new(
            "Test GPU".to_string(),
            DeviceType::Discrete,
            8 * 1024 * 1024 * 1024,
            1024 * 1024 * 1024,
        )
        .with_supported_operation(GpuOperation::MatrixMultiplication)
        .with_supported_operation(GpuOperation::Attention);

        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.device_type, DeviceType::Discrete);
        assert!(info.supports_operation(GpuOperation::MatrixMultiplication));
        assert!(!info.supports_operation(GpuOperation::Convolution));
    }

    #[test]
    fn test_kernel_templates() {
        let matrix_template = KernelTemplates::matrix_multiply();
        assert!(matrix_template.contains("kernel"));
        assert!(matrix_template.contains("matrix_multiply"));

        let element_template = KernelTemplates::element_add();
        assert!(element_template.contains("element_add"));

        let attention_template = KernelTemplates::attention();
        assert!(attention_template.contains("attention"));
    }

    #[test]
    fn test_mock_gpu_buffer_clone() {
        let mut original = MockGpuBuffer::new("original".to_string(), 1024, BufferUsage::ReadWrite);
        original.copy_from_host(&[1, 2, 3, 4]).unwrap();

        let cloned = original.clone();
        assert_ne!(original.buffer_id(), cloned.buffer_id());
        assert_eq!(original.size(), cloned.size());
        assert_eq!(original.data(), cloned.data());
        assert!(!cloned.is_mapped()); // Clone starts unmapped
    }
}