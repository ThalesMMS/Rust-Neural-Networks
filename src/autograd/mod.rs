//! Automatic differentiation engine.
//!
//! This module provides a computational graph-based automatic differentiation
//! system, enabling gradients to be computed automatically via reverse-mode AD.
//!
//! # Submodules
//!
//! - [`tensor`] — Core [`Tensor`] type with interior-mutable gradient storage.
//! - [`tape`] — [`Op`] enum, [`GradNode`] struct, and topological-sort traversal.
//! - [`ops`] — Differentiable forward operations that record gradient nodes.

pub mod ops;
pub mod tape;
pub mod tensor;

// Re-export the primary public types for convenience.
pub use ops::*;
pub use tape::{GradNode, Op};
pub use tensor::Tensor;
