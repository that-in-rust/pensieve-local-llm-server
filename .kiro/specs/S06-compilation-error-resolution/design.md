# Design Document

## Overview

This design addresses the systematic resolution of 27 compilation errors in the code-ingest Rust codebase. The errors fall into distinct categories that require different resolution strategies. The design prioritizes maintaining existing functionality while ensuring memory safety and following Rust best practices.

## Architecture

### Error Classification System

The compilation errors are categorized into five main types:

1. **Dependency API Changes** - Methods/APIs that have changed in dependency updates
2. **Ownership/Borrowing Issues** - Rust borrow checker violations
3. **Missing Imports/Types** - Unresolved symbols and missing definitions
4. **Serialization Problems** - Serde compatibility issues with system types
5. **Configuration/Implementation Gaps** - Missing method implementations

### Resolution Strategy

Each category requires a specific approach:

```rust
// Dependency API Resolution Pattern
// Before: deprecated_method()
// After: current_api_method()

// Ownership Resolution Pattern  
// Before: value moved/borrowed incorrectly
// After: proper borrowing with minimal clones

// Import Resolution Pattern
// Before: unresolved symbol
// After: correct import with minimal scope
```

## Components and Interfaces

### 1. Dependency Compatibility Layer

**Purpose**: Handle API changes in external dependencies

**Key Components**:
- `SysInfoAdapter` - Wraps sysinfo API changes
- `Git2ConfigAdapter` - Handles git2 configuration updates
- `SqlxConnectionManager` - Manages database connection borrowing

```rust
// SysInfo API Compatibility
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub active_processes: usize,
}

impl SystemMetrics {
    pub fn collect() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        Self {
            cpu_usage: system.global_cpu_info().cpu_usage(),
            memory_usage: system.used_memory(),
            active_processes: system.processes().len(),
        }
    }
}
```

### 2. Performance Monitoring Implementation

**Purpose**: Implement missing PerformanceMonitor methods

**Key Components**:
- `PerformanceMonitor` - System resource tracking
- `PerformanceConfig` - Configuration for monitoring
- `OptimizationRecommendation` - Performance suggestions

```rust
pub struct PerformanceMonitor {
    config: PerformanceConfig,
    metrics_history: VecDeque<SystemMetrics>,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub async fn start_monitoring(&self) -> Result<(), ProcessingError> {
        // Implementation for continuous monitoring
    }
    
    pub fn get_current_utilization(&self) -> Result<f64, ProcessingError> {
        // Current system utilization calculation
    }
    
    pub fn is_under_pressure(&self) -> bool {
        // Determine if system is under resource pressure
    }
    
    pub fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        // Generate performance recommendations
    }
}
```

### 3. Serialization Compatibility

**Purpose**: Handle non-serializable system types

**Key Components**:
- Custom serialization for `Instant`
- Alternative representations for system types
- Serde compatibility wrappers

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMetrics {
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub active_processes: usize,
    #[serde(with = "timestamp_serde")]
    pub timestamp: SystemTime, // Use SystemTime instead of Instant
}

mod timestamp_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::SystemTime;
    
    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        time.duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}
```

### 4. Database Connection Management

**Purpose**: Resolve SQLx connection borrowing issues

**Key Components**:
- Connection pool management
- Transaction handling
- Query execution patterns

```rust
impl DatabaseConnection {
    pub async fn optimize_connection(&mut self) -> DatabaseResult<()> {
        let mut conn = self.pool.acquire().await?;
        
        // Execute optimization queries sequentially
        sqlx::query("SET synchronous_commit = off").execute(&mut *conn).await?;
        sqlx::query("SET wal_buffers = '16MB'").execute(&mut *conn).await?;
        sqlx::query("SET checkpoint_completion_target = 0.9").execute(&mut *conn).await?;
        sqlx::query("SET shared_buffers = '256MB'").execute(&mut *conn).await?;
        sqlx::query("SET effective_cache_size = '1GB'").execute(&mut *conn).await?;
        sqlx::query("SET work_mem = '64MB'").execute(&mut *conn).await?;
        sqlx::query("SET maintenance_work_mem = '256MB'").execute(&mut *conn).await?;
        
        Ok(())
    }
}
```

## Data Models

### Error Resolution Tracking

```rust
#[derive(Debug, Clone)]
pub struct CompilationError {
    pub file_path: PathBuf,
    pub error_type: ErrorType,
    pub line_number: Option<u32>,
    pub description: String,
    pub resolution_strategy: ResolutionStrategy,
}

#[derive(Debug, Clone)]
pub enum ErrorType {
    DependencyApi,
    OwnershipBorrowing,
    MissingImport,
    TypeMismatch,
    SerializationIssue,
    MissingImplementation,
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    UpdateApiCall,
    FixBorrowing,
    AddImport,
    CorrectType,
    CustomSerialization,
    ImplementMethod,
}
```

### Performance Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub monitoring_interval: Duration,
    pub memory_threshold: f64,
    pub cpu_threshold: f64,
    pub enable_recommendations: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(5),
            memory_threshold: 0.8,
            cpu_threshold: 0.9,
            enable_recommendations: true,
        }
    }
}
```

## Error Handling

### Compilation Error Recovery

```rust
#[derive(Error, Debug)]
pub enum CompilationFixError {
    #[error("Failed to resolve dependency API: {api} - {cause}")]
    DependencyApiError { api: String, cause: String },
    
    #[error("Ownership resolution failed: {context}")]
    OwnershipError { context: String },
    
    #[error("Import resolution failed: {symbol} in {file}")]
    ImportError { symbol: String, file: String },
    
    #[error("Type correction failed: expected {expected}, found {found}")]
    TypeMismatchError { expected: String, found: String },
}
```

### Performance Monitoring Errors

```rust
#[derive(Error, Debug)]
pub enum PerformanceError {
    #[error("System metrics collection failed: {0}")]
    MetricsCollectionError(String),
    
    #[error("Performance threshold exceeded: {metric} = {value}")]
    ThresholdExceeded { metric: String, value: f64 },
    
    #[error("Monitoring initialization failed: {0}")]
    InitializationError(String),
}
```

## Testing Strategy

### Unit Testing Approach

1. **Dependency Compatibility Tests**
   - Test each adapter with mock dependencies
   - Verify API compatibility across versions
   - Validate error handling for API changes

2. **Performance Monitor Tests**
   - Test metric collection accuracy
   - Verify threshold detection
   - Test recommendation generation

3. **Serialization Tests**
   - Test custom serialization roundtrips
   - Verify compatibility with existing data
   - Test error handling for invalid data

### Integration Testing

1. **End-to-End Compilation Tests**
   - Verify successful compilation after fixes
   - Test functionality preservation
   - Validate performance characteristics

2. **Database Integration Tests**
   - Test connection management
   - Verify transaction handling
   - Test query execution patterns

### Performance Testing

1. **Resource Usage Tests**
   - Monitor memory usage during operations
   - Test CPU utilization patterns
   - Verify monitoring overhead is minimal

2. **Scalability Tests**
   - Test with large datasets
   - Verify performance under load
   - Test resource cleanup

## Implementation Phases

### Phase 1: Critical Compilation Fixes
- Fix dependency API issues (sysinfo, git2)
- Resolve ownership/borrowing errors
- Add missing imports and types

### Phase 2: Performance Monitoring Implementation
- Implement PerformanceMonitor methods
- Add PerformanceConfig and related types
- Create serialization compatibility layer

### Phase 3: Database and Concurrency Fixes
- Fix SQLx connection borrowing
- Resolve concurrent processing issues
- Implement proper resource management

### Phase 4: Warning Resolution and Cleanup
- Address unused variable warnings
- Remove unused imports
- Fix mutability warnings
- Code quality improvements