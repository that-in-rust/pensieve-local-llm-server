//! Pensieve Core - Foundation traits and error types
//!
//! This is the Layer 1 (L1) foundation crate that provides:
//! - Core traits and interfaces
//! - Error handling patterns
//! - Result/Option convenience types
//! - No-std compatible abstractions
//!
//! No external dependencies, only core library features.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

// Re-export core types for convenience
pub use core::{
    convert::{AsRef, From, Into, TryFrom, TryInto},
    fmt,
    option::Option,
};

/// Core error types for the Pensieve system
pub mod error {
    use super::fmt;

    /// Core result type for pensieve operations
    pub type CoreResult<T> = core::result::Result<T, CoreError>;

    /// Base error type for all pensieve operations
    #[derive(Debug, Clone, PartialEq)]
    pub enum CoreError {
        /// Invalid configuration or parameters
        InvalidConfig(&'static str),
        /// Resource not found
        NotFound(&'static str),
        /// Invalid input data
        InvalidInput(&'static str),
        /// Operation not supported
        Unsupported(&'static str),
        /// Resource unavailable
        Unavailable(&'static str),
        /// Generic error with message
        Generic(&'static str),
    }

    impl fmt::Display for CoreError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                CoreError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
                CoreError::NotFound(msg) => write!(f, "Not found: {}", msg),
                CoreError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
                CoreError::Unsupported(msg) => write!(f, "Unsupported operation: {}", msg),
                CoreError::Unavailable(msg) => write!(f, "Unavailable: {}", msg),
                CoreError::Generic(msg) => write!(f, "Error: {}", msg),
            }
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for CoreError {}

    // Conversion from common string types for convenience
    impl From<&str> for CoreError {
        fn from(_msg: &str) -> Self {
            CoreError::Generic("generic error")
        }
    }
}

/// Core traits for the Pensieve system
pub mod traits {
    use super::error::CoreError;

    /// Trait for types that can be validated
    pub trait Validate {
        /// Validate the current state
        fn validate(&self) -> core::result::Result<(), CoreError>;
    }

    /// Trait for types that can be reset to initial state
    pub trait Reset {
        /// Reset to initial state
        fn reset(&mut self);
    }

    /// Trait for resource management with RAII pattern
    pub trait Resource {
        /// Type of error that can occur during operations
        type Error: Into<CoreError>;

        /// Check if the resource is available/valid
        fn is_available(&self) -> bool;

        /// Acquire the resource
        fn acquire(&mut self) -> core::result::Result<(), Self::Error>;

        /// Release the resource
        fn release(&mut self) -> core::result::Result<(), Self::Error>;
    }
}

/// Result type alias for convenience
pub type Result<T> = error::CoreResult<T>;

/// Re-export commonly used items
pub use error::{CoreError, CoreResult};
pub use traits::{Resource, Reset, Validate};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_error_creation() {
        let error = CoreError::InvalidConfig("test");
        assert_eq!(error, CoreError::InvalidConfig("test"));
    }

    #[test]
    fn test_error_from_string() {
        let error: CoreError = "test message".into();
        assert!(matches!(error, CoreError::Generic(_msg)));
    }

    #[test]
    fn test_result_type_alias() {
        let success: Result<i32> = Ok(42);
        assert_eq!(success, Ok(42));

        let failure: Result<i32> = Err(CoreError::InvalidInput("bad"));
        assert!(failure.is_err());
    }

    #[test]
    fn test_trait_bounds_exist() {
        // Test that we can create structs that implement our traits
        struct TestResource {
            available: bool,
        }

        impl Resource for TestResource {
            type Error = CoreError;

            fn is_available(&self) -> bool {
                self.available
            }

            fn acquire(&mut self) -> core::result::Result<(), Self::Error> {
                self.available = true;
                Ok(())
            }

            fn release(&mut self) -> core::result::Result<(), Self::Error> {
                self.available = false;
                Ok(())
            }
        }

        impl Validate for TestResource {
            fn validate(&self) -> core::result::Result<(), CoreError> {
                if self.available {
                    Ok(())
                } else {
                    Err(CoreError::Unavailable("resource not available"))
                }
            }
        }

        impl Reset for TestResource {
            fn reset(&mut self) {
                self.available = false;
            }
        }

        let mut resource = TestResource { available: false };
        assert!(!resource.is_available());

        resource.acquire().unwrap();
        assert!(resource.is_available());

        resource.release().unwrap();
        assert!(!resource.is_available());
    }
}