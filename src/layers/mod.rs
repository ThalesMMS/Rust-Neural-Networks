//! Layer abstractions for neural networks
//!
//! This module provides the Layer trait and implementations for common layer types
//! used across different neural network architectures.

pub mod conv2d;
pub mod dense;
pub mod dropout;
mod r#trait;

// Re-export the Layer trait for convenience
pub use conv2d::Conv2DLayer;
pub use dense::DenseLayer;
pub use dropout::DropoutLayer;
pub use r#trait::Layer;
