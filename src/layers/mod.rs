//! Layer abstractions for neural networks
//!
//! This module provides the Layer trait and implementations for common layer types
//! used across different neural network architectures.

pub mod attention;
pub mod batchnorm;
pub mod conv2d;
pub mod dense;
pub mod dropout;
pub mod layernorm;
pub mod lstm;
pub mod rnn;
pub mod transformer;
mod r#trait;

// Re-export the Layer trait for convenience
pub use attention::MultiHeadAttentionLayer;
pub use batchnorm::BatchNormLayer;
pub use conv2d::Conv2DLayer;
pub use dense::DenseLayer;
pub use dropout::DropoutLayer;
pub use layernorm::LayerNormLayer;
pub use lstm::LstmLayer;
pub use r#trait::Layer;
pub use rnn::RnnLayer;
pub use transformer::{TransformerBlock, TransformerEncoder};
