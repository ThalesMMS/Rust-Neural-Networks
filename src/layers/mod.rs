//! Layer abstractions for neural networks
//!
//! This module provides the Layer trait and implementations for common layer types
//! used across different neural network architectures.

pub mod attention;
pub mod batchnorm;
pub mod conv2d;
pub mod dense;
pub mod dropout;
pub mod gradient;
pub mod layernorm;
pub mod lstm;
pub mod pooling;
pub mod residual;
pub mod rnn;
mod r#trait;
pub mod transformer;

// Re-export the Layer trait for convenience
pub use attention::MultiHeadAttentionLayer;
pub use batchnorm::BatchNormLayer;
pub use conv2d::Conv2DLayer;
pub use dense::DenseLayer;
pub use dropout::DropoutLayer;
pub use gradient::GradientAccumulator;
pub use layernorm::LayerNormLayer;
pub use lstm::LstmLayer;
pub use pooling::GlobalAvgPoolLayer;
pub use r#trait::Layer;
pub use residual::ResidualBlock;
pub use rnn::RnnLayer;
pub use transformer::{TransformerBlock, TransformerEncoder};
