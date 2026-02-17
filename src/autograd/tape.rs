//! Computational graph nodes for automatic differentiation.
//!
//! This module defines the [`Op`] enum and [`GradNode`] struct that together
//! represent a node in the backward computation graph. Each output tensor produced
//! by a differentiable operation stores an `Option<Box<GradNode>>` referencing the
//! operation that created it and the input tensors it consumed.
//!
//! # Overview
//!
//! - [`Op`] encodes the mathematical operation (Add, MatMul, ReLU, …).
//! - [`GradNode`] pairs an [`Op`] with the input [`Tensor`]s for that operation.
//! - [`GradNode::build_topo`] performs a depth-first traversal from a root tensor
//!   and returns all reachable tensors in **reverse topological order** (root first,
//!   leaves last), which is the correct iteration order for reverse-mode AD.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::autograd::tensor::Tensor;
//! use rust_neural_networks::autograd::tape::GradNode;
//!
//! // A leaf tensor with no creating operation.
//! let leaf = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
//! let topo = GradNode::build_topo(&leaf);
//! // Only the leaf itself is reachable.
//! assert_eq!(topo.len(), 1);
//! ```

use crate::autograd::tensor::Tensor;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Op
// ---------------------------------------------------------------------------

/// Differentiable operations recorded in the computational graph.
///
/// Each variant represents a mathematical operation whose gradient rule is
/// implemented in the backward pass. The [`MatMul`](Op::MatMul) variant stores
/// the matrix dimensions `(m, k, n)` so the backward pass can reconstruct the
/// correct shapes without re-deriving them from tensor metadata.
///
/// # Variants
///
/// * `Add` — Element-wise addition of two tensors.
/// * `Sub` — Element-wise subtraction of two tensors.
/// * `Mul` — Element-wise (Hadamard) multiplication of two tensors.
/// * `MatMul` — Matrix multiplication `A (m×k) × B (k×n) = out (m×n)`.
/// * `ReLU` — Rectified linear unit applied element-wise.
/// * `Sigmoid` — Logistic sigmoid applied element-wise.
/// * `Tanh` — Hyperbolic tangent applied element-wise.
/// * `SoftmaxCE` — Fused softmax + cross-entropy loss (scalar output).
/// * `MSE` — Mean squared error loss (scalar output).
#[derive(Debug, Clone)]
pub enum Op {
    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    Add,
    /// Element-wise subtraction: `out[i] = a[i] - b[i]`.
    Sub,
    /// Element-wise (Hadamard) product: `out[i] = a[i] * b[i]`.
    Mul,
    /// Matrix multiplication `A (m×k) × B (k×n) = out (m×n)`.
    MatMul {
        /// Number of rows in A and in the output matrix.
        m: usize,
        /// Shared inner dimension (columns of A, rows of B).
        k: usize,
        /// Number of columns in B and in the output matrix.
        n: usize,
    },
    /// Element-wise ReLU: `out[i] = max(0, x[i])`.
    ReLU,
    /// Element-wise logistic sigmoid: `out[i] = 1 / (1 + exp(-x[i]))`.
    Sigmoid,
    /// Element-wise hyperbolic tangent: `out[i] = tanh(x[i])`.
    Tanh,
    /// Fused softmax cross-entropy loss (scalar output).
    ///
    /// Inputs are `[logits, cached_softmax, labels]`.  `cached_softmax` holds the
    /// pre-computed softmax probabilities (no grad) so the backward pass can use
    /// them without recomputing.  `labels` holds integer class indices as f32.
    /// The fused formulation uses the log-sum-exp trick for numerical stability.
    SoftmaxCE,
    /// Mean squared error loss: `out = mean((pred - target)^2)` (scalar output).
    ///
    /// Inputs are `[predictions, targets]`.
    /// Backward: `grad_pred[i] = upstream * 2 * (pred[i] - target[i]) / n`.
    MSE,
    /// Sum reduction: sums all elements to a scalar (1×1) output.
    ///
    /// Backward: broadcast scalar gradient to every input element.
    Sum,
    /// Mean reduction: averages all elements to a scalar (1×1) output.
    ///
    /// Backward: broadcast `grad_out / n` to every input element, where `n` is
    /// the total number of input elements.
    Mean,
}

// ---------------------------------------------------------------------------
// GradNode
// ---------------------------------------------------------------------------

/// A node in the backward computation graph.
///
/// Each output tensor produced by a differentiable operation stores an
/// `Option<Box<GradNode>>`. The `GradNode` records which [`Op`] produced the
/// tensor and clones of all input tensors that the backward pass must propagate
/// gradients through.
///
/// Holding cloned [`Tensor`] handles (cheap `Rc` clones) keeps the input tensors
/// alive for the duration of the backward pass without copying their data.
///
/// # Fields
///
/// * `op` — The operation that produced the output tensor.
/// * `inputs` — Input tensors consumed by the operation, in the order expected
///   by the corresponding backward gradient rule.
pub struct GradNode {
    /// The operation that produced the output tensor.
    pub op: Op,
    /// Input tensors consumed by the operation.
    pub inputs: Vec<Tensor>,
}

impl GradNode {
    /// Constructs a new `GradNode` recording `op` and the given input tensors.
    ///
    /// Callers should pass cheap `Rc` clones of the input tensors so that the
    /// backward traversal can access their data and gradient buffers.
    ///
    /// # Arguments
    ///
    /// * `op` — The mathematical operation that produced the output tensor.
    /// * `inputs` — Input tensors consumed by `op`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tape::{GradNode, Op};
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    /// let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
    /// let node = GradNode::new(Op::Add, vec![a, b]);
    /// assert!(matches!(node.op, Op::Add));
    /// assert_eq!(node.inputs.len(), 2);
    /// ```
    pub fn new(op: Op, inputs: Vec<Tensor>) -> Self {
        Self { op, inputs }
    }

    /// Traverses the computation graph from `root` and returns all tensors in
    /// **reverse topological order** suitable for backward-pass iteration.
    ///
    /// "Reverse topological order" means that `root` appears **first** in the
    /// returned `Vec` and leaf tensors (tensors with no creating operation)
    /// appear **last**. Iterating front-to-back therefore propagates gradients
    /// in the correct direction: from the loss output toward the leaf parameters.
    ///
    /// The traversal handles shared nodes: if the same tensor is reachable via
    /// multiple paths (e.g. a weight used in multiple layers), it appears in the
    /// output exactly once.
    ///
    /// # Arguments
    ///
    /// * `root` — The tensor to start traversal from (typically the scalar loss).
    ///
    /// # Returns
    ///
    /// A `Vec<Tensor>` in reverse topological order: `result[0]` is `root`,
    /// `result[result.len() - 1]` is a leaf tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    /// use rust_neural_networks::autograd::tape::GradNode;
    ///
    /// // A single leaf tensor with no grad_node.
    /// let leaf = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    /// let topo = GradNode::build_topo(&leaf);
    /// assert_eq!(topo.len(), 1);
    /// ```
    pub fn build_topo(root: &Tensor) -> Vec<Tensor> {
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        build_topo_dfs(root, &mut topo, &mut visited);
        // DFS produces post-order (leaves first, root last); reverse for backward.
        topo.reverse();
        topo
    }
}

impl fmt::Debug for GradNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GradNode")
            .field("op", &self.op)
            .field("inputs.len", &self.inputs.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Traversal helpers
// ---------------------------------------------------------------------------

/// Recursive depth-first post-order traversal of the computation graph.
///
/// Visits all input tensors before appending `t` to `topo`, ensuring a valid
/// topological ordering (inputs before outputs). The `visited` set tracks tensor
/// identity by raw `Rc` pointer to handle shared tensors that appear in multiple
/// paths through the graph.
///
/// The borrow of `t.0` is released before recursing into child nodes to avoid
/// simultaneous borrows through `RefCell`.
fn build_topo_dfs(t: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<usize>) {
    // Use the raw Rc pointer as a stable unique identifier for tensor identity.
    let ptr = Rc::as_ptr(&t.0) as usize;
    if !visited.insert(ptr) {
        // Already processed; skip to avoid duplicates.
        return;
    }

    // Collect the inputs before recursing so we do not hold a borrow on
    // `t.0` across the recursive calls (which may also borrow inner nodes).
    let inputs: Vec<Tensor> =
        t.0.borrow()
            .grad_node
            .as_ref()
            .map(|node| node.inputs.clone())
            .unwrap_or_default();

    for input in &inputs {
        build_topo_dfs(input, topo, visited);
    }

    // Post-order: append after all inputs have been appended.
    topo.push(t.clone());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::tensor::Tensor;

    #[test]
    fn test_op_variants_clone_and_debug() {
        let variants: Vec<Op> = vec![
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::MatMul { m: 2, k: 3, n: 4 },
            Op::ReLU,
            Op::Sigmoid,
            Op::Tanh,
            Op::SoftmaxCE,
            Op::MSE,
        ];
        for op in &variants {
            let cloned = op.clone();
            // Clone should not panic and Debug should work.
            let _ = format!("{:?}", cloned);
        }
        // Spot-check a struct variant.
        assert!(matches!(variants[3], Op::MatMul { m: 2, k: 3, n: 4 }));
    }

    #[test]
    fn test_grad_node_new() {
        let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        let b = Tensor::from_vec(vec![3.0, 4.0], (1, 2), true);
        let node = GradNode::new(Op::Add, vec![a, b]);
        assert!(matches!(node.op, Op::Add));
        assert_eq!(node.inputs.len(), 2);
    }

    #[test]
    fn test_grad_node_debug() {
        let a = Tensor::from_vec(vec![1.0], (1, 1), true);
        let node = GradNode::new(Op::ReLU, vec![a]);
        let s = format!("{:?}", node);
        assert!(s.contains("GradNode"));
        assert!(s.contains("ReLU"));
    }

    #[test]
    fn test_build_topo_single_leaf() {
        let leaf = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        let topo = GradNode::build_topo(&leaf);
        assert_eq!(topo.len(), 1);
        // The single element should be the leaf itself.
        assert!(std::ptr::eq(topo[0].0.as_ptr(), leaf.0.as_ptr()));
    }

    #[test]
    fn test_build_topo_linear_chain() {
        // Build a 3-node linear chain: leaf → mid → root.
        let leaf = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);

        let mid = Tensor::from_vec(vec![2.0, 4.0], (1, 2), true);
        mid.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::ReLU, vec![leaf.clone()])));

        let root = Tensor::from_vec(vec![2.0, 4.0], (1, 2), true);
        root.0.borrow_mut().grad_node =
            Some(Box::new(GradNode::new(Op::Sigmoid, vec![mid.clone()])));

        let topo = GradNode::build_topo(&root);

        // Expect root, mid, leaf in that order.
        assert_eq!(topo.len(), 3);
        assert!(std::ptr::eq(topo[0].0.as_ptr(), root.0.as_ptr()));
        assert!(std::ptr::eq(topo[1].0.as_ptr(), mid.0.as_ptr()));
        assert!(std::ptr::eq(topo[2].0.as_ptr(), leaf.0.as_ptr()));
    }

    #[test]
    fn test_build_topo_diamond_shared_leaf() {
        // Diamond graph: leaf is shared between two paths.
        //   leaf
        //  /    \
        // mid1  mid2
        //  \    /
        //   root
        let leaf = Tensor::from_vec(vec![1.0], (1, 1), true);

        let mid1 = Tensor::from_vec(vec![1.0], (1, 1), true);
        mid1.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::ReLU, vec![leaf.clone()])));

        let mid2 = Tensor::from_vec(vec![1.0], (1, 1), true);
        mid2.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::Tanh, vec![leaf.clone()])));

        let root = Tensor::from_vec(vec![2.0], (1, 1), true);
        root.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Add,
            vec![mid1.clone(), mid2.clone()],
        )));

        let topo = GradNode::build_topo(&root);

        // 4 distinct tensors: root, mid1, mid2, leaf (leaf appears only once).
        assert_eq!(topo.len(), 4);
        // root must be first.
        assert!(std::ptr::eq(topo[0].0.as_ptr(), root.0.as_ptr()));
        // leaf must be last (deepest in post-order).
        assert!(std::ptr::eq(topo[3].0.as_ptr(), leaf.0.as_ptr()));
    }

    #[test]
    fn test_build_topo_two_independent_leaves() {
        // Simple binary op: root = add(a, b) with two independent leaves.
        let a = Tensor::from_vec(vec![1.0], (1, 1), true);
        let b = Tensor::from_vec(vec![2.0], (1, 1), true);

        let root = Tensor::from_vec(vec![3.0], (1, 1), true);
        root.0.borrow_mut().grad_node =
            Some(Box::new(GradNode::new(Op::Add, vec![a.clone(), b.clone()])));

        let topo = GradNode::build_topo(&root);
        // root + a + b = 3 nodes.
        assert_eq!(topo.len(), 3);
        // root first.
        assert!(std::ptr::eq(topo[0].0.as_ptr(), root.0.as_ptr()));
    }
}
