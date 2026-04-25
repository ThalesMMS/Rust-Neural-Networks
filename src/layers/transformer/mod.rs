//! Transformer block layer implementation
//!
//! This module provides a TransformerBlock that combines multi-head attention,
//! layer normalization, feed-forward network, and residual connections following
//! the "Attention is All You Need" architecture (Vaswani et al., 2017).
//!
//! # Architecture
//!
//! The TransformerBlock implements a Pre-LN (Pre-Layer Normalization) architecture:
//!
//! ```text
//! input
//!   |
//!   +--> LayerNorm --> MultiHeadAttention --> Add(residual) --> [intermediate]
//!                                              |
//!                                              +--> LayerNorm --> FFN --> Add(residual) --> output
//! ```
//!
//! Where FFN (Feed-Forward Network) is:
//! ```text
//! Dense(d_model -> d_ff) --> ReLU --> Dense(d_ff -> d_model)
//! ```
//!
//! # Pre-LN vs Post-LN
//!
//! This implementation uses Pre-LN where layer normalization is applied *before*
//! each sub-layer (attention and FFN), which provides better training stability
//! and gradient flow compared to the original Post-LN architecture.
//!
//! # References
//!
//! Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.
//! Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. ICML.

mod block;
mod encoder;

pub use block::TransformerBlock;
pub use encoder::TransformerEncoder;
