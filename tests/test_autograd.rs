// Tests for the automatic differentiation engine.
//
// This file grows incrementally as each subtask is implemented.
// Phase 1 (subtask-1-1): tensor construction and basic accessor tests.
// Phase 2 (subtask-2-1): basic element-wise and scalar operations.
// Phase 6 (subtask-6-1): numerical gradient checking via finite differences.
// Phase 6 (subtask-6-2): XOR MLP integration test - loss decrease, numgrad, hand-coded backward.

use approx::assert_relative_eq;
use rust_neural_networks::autograd::tensor::Tensor;

#[path = "test_autograd/loss_backward.rs"]
mod loss_backward;
#[path = "test_autograd/matmul_reductions.rs"]
mod matmul_reductions;
#[path = "test_autograd/numgrad.rs"]
mod numgrad;
#[path = "test_autograd/ops.rs"]
mod ops;
#[path = "test_autograd/tensor_construction.rs"]
mod tensor_construction;
use numgrad::numerical_gradient;
#[path = "test_autograd/xor_mlp.rs"]
mod xor_mlp;
