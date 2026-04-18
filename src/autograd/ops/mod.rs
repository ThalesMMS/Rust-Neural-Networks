//! Differentiable operations on tensors.
//!
//! This module provides the forward-pass implementations of differentiable
//! operations. Each function:
//!
//! 1. Computes the forward result.
//! 2. Records a [`GradNode`] on the output tensor so the backward pass can
//!    propagate gradients through the computational graph via the chain rule.
//!
//! # Operations
//!
//! | Function              | Description                                      |
//! |-----------------------|--------------------------------------------------|
//! | [`tensor_add`]        | Element-wise addition of two same-shape tensors  |
//! | [`tensor_sub`]        | Element-wise subtraction of two same-shape tensors |
//! | [`tensor_mul_scalar`] | Multiply all elements by a scalar constant       |
//! | [`tensor_add_bias`]   | Add a `(1, features)` bias to a `(batch, features)` matrix |
//! | [`tensor_relu`]       | Element-wise ReLU activation                     |
//! | [`tensor_sigmoid`]    | Element-wise logistic sigmoid activation         |
//! | [`tensor_tanh`]       | Element-wise hyperbolic tangent activation       |
//! | [`tensor_softmax_cross_entropy`] | Fused softmax + cross-entropy loss (numerically stable) |
//! | [`tensor_mse_loss`]   | Mean squared error loss                          |
//!
//! # Gradient recording
//!
//! A [`GradNode`] is only recorded on the output tensor when at least one
//! input has `requires_grad = true`.  Tensors that do not participate in
//! gradient computation have `grad_node = None` and are therefore ignored
//! during backward traversal.

mod activations;
mod arithmetic;
mod backward;
mod linear;
mod losses;
mod reductions;

pub use activations::*;
pub use arithmetic::*;
pub use backward::*;
pub use linear::*;
pub use losses::*;
pub use reductions::*;

#[cfg(test)]
mod tests;
