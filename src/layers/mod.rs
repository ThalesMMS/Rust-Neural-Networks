//! Layer abstractions for neural networks
//!
//! This module provides the Layer trait and implementations for common layer types
//! used across different neural network architectures.

mod r#trait;
pub mod dense;
pub mod conv2d;

// Re-export the Layer trait for convenience
pub use r#trait::Layer;
pub use dense::DenseLayer;
pub use conv2d::Conv2DLayer;
