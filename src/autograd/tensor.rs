//! Core tensor type for automatic differentiation.
//!
//! This module provides the foundational [`Tensor`] type used throughout the autograd engine.
//! A [`Tensor`] wraps a [`TensorInner`] in an `Rc<RefCell<...>>` to enable shared ownership
//! and interior mutability — essential for building a computational graph where multiple
//! downstream nodes may hold references to the same input tensor.
//!
//! # Overview
//!
//! - [`TensorInner`] holds the actual data, optional gradient buffer, shape, and
//!   `requires_grad` flag.
//! - [`Tensor`] is a cheap-to-clone newtype over `Rc<RefCell<TensorInner>>`. Cloning a
//!   [`Tensor`] clones only the reference-counted pointer, not the underlying data.
//! - [`Tensor::accumulate_grad`] adds an incoming gradient slice into the tensor's gradient
//!   buffer, allocating the buffer on first use. This is called during backward traversal.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::autograd::tensor::Tensor;
//!
//! // Leaf tensor with gradient tracking
//! let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
//! assert_eq!(t.shape(), (1, 3));
//! assert!(t.requires_grad());
//! assert!(t.grad().is_none());
//!
//! // Accumulate a gradient
//! t.accumulate_grad(&[0.1, 0.2, 0.3]);
//! let g = t.grad().unwrap();
//! assert!((g[0] - 0.1).abs() < 1e-6);
//! ```

use crate::autograd::tape::GradNode;
use std::cell::RefCell;
use std::rc::Rc;

/// Inner storage for a tensor node in the computational graph.
///
/// Holds the raw data vector, optional accumulated gradient, tensor shape,
/// a flag indicating whether gradients should be computed for this tensor,
/// and an optional backward node that records the operation that produced it.
///
/// # Fields
///
/// * `data` - Flat row-major data buffer; length must equal `rows * cols`.
/// * `grad` - Accumulated gradient buffer; `None` until the first gradient is accumulated.
/// * `shape` - `(rows, cols)` dimensions of the tensor.
/// * `requires_grad` - If `true`, gradients are tracked and accumulated during backward.
/// * `grad_node` - Operation node linking this tensor back to its inputs in the graph;
///   `None` for leaf tensors that were not produced by a recorded operation.
pub struct TensorInner {
    /// Flat data buffer in row-major order; length equals `shape.0 * shape.1`.
    pub data: Vec<f32>,
    /// Accumulated gradient; `None` before any gradient has been accumulated.
    pub grad: Option<Vec<f32>>,
    /// Tensor dimensions `(rows, cols)`.
    pub shape: (usize, usize),
    /// Whether this tensor participates in gradient computation.
    pub requires_grad: bool,
    /// Backward node recording the op that produced this tensor and its inputs.
    ///
    /// `None` for leaf tensors (e.g. model weights created directly) and for
    /// tensors produced by non-differentiable operations. Set by the operation
    /// functions in `src/autograd/ops.rs` after computing the forward result.
    pub grad_node: Option<Box<GradNode>>,
}

/// A tensor in the computational graph — a reference-counted, interior-mutable node.
///
/// `Tensor` is a newtype over `Rc<RefCell<TensorInner>>`. Multiple tensors in the
/// computational graph can share ownership of the same inner node, and any of them
/// may mutate it (e.g. to accumulate gradients) via the `RefCell`.
///
/// Cloning a `Tensor` is cheap: it increments the reference count and copies the
/// thin pointer, not the underlying data.
///
/// # Example
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
///
/// let a = Tensor::zeros((2, 3));
/// assert_eq!(a.shape(), (2, 3));
/// assert_eq!(a.data(), vec![0.0f32; 6]);
/// ```
#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

impl Tensor {
    /// Creates a new leaf tensor from pre-allocated data and a shape.
    ///
    /// The tensor is created **without** gradient tracking (`requires_grad = false`).
    /// Use [`Tensor::from_vec`] to control gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat row-major data vector; must have length `shape.0 * shape.1`.
    /// * `shape` - `(rows, cols)` dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != shape.0 * shape.1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    /// assert_eq!(t.shape(), (2, 2));
    /// assert!(!t.requires_grad());
    /// ```
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Tensor::new: data length {} does not match shape ({}, {}), expected {}",
            data.len(),
            shape.0,
            shape.1,
            shape.0 * shape.1
        );
        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            grad: None,
            shape,
            requires_grad: false,
            grad_node: None,
        })))
    }

    /// Creates a tensor filled with zeros.
    ///
    /// The tensor is created **without** gradient tracking (`requires_grad = false`).
    ///
    /// # Arguments
    ///
    /// * `shape` - `(rows, cols)` dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::zeros((3, 4));
    /// assert_eq!(t.shape(), (3, 4));
    /// assert_eq!(t.data(), vec![0.0f32; 12]);
    /// assert!(!t.requires_grad());
    /// ```
    pub fn zeros(shape: (usize, usize)) -> Self {
        let n = shape.0 * shape.1;
        Tensor(Rc::new(RefCell::new(TensorInner {
            data: vec![0.0f32; n],
            grad: None,
            shape,
            requires_grad: false,
            grad_node: None,
        })))
    }

    /// Creates a tensor from a data vector with explicit gradient-tracking control.
    ///
    /// This is the primary constructor for leaf tensors that are to be trained
    /// (e.g. model weights), where `requires_grad` should be `true`.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat row-major data vector; must have length `shape.0 * shape.1`.
    /// * `shape` - `(rows, cols)` dimensions.
    /// * `requires_grad` - Whether to track and accumulate gradients for this tensor.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != shape.0 * shape.1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let w = Tensor::from_vec(vec![0.5, -0.3, 0.1], (1, 3), true);
    /// assert!(w.requires_grad());
    /// assert!(w.grad().is_none());
    /// ```
    pub fn from_vec(data: Vec<f32>, shape: (usize, usize), requires_grad: bool) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Tensor::from_vec: data length {} does not match shape ({}, {}), expected {}",
            data.len(),
            shape.0,
            shape.1,
            shape.0 * shape.1
        );
        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            grad: None,
            shape,
            requires_grad,
            grad_node: None,
        })))
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Returns a clone of the tensor's data buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![1.0, 2.0], (1, 2));
    /// assert_eq!(t.data(), vec![1.0f32, 2.0]);
    /// ```
    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().data.clone()
    }

    /// Returns a clone of the gradient buffer, or `None` if no gradient has been accumulated.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    /// assert!(t.grad().is_none());
    /// t.accumulate_grad(&[0.5, 0.5]);
    /// assert!(t.grad().is_some());
    /// ```
    pub fn grad(&self) -> Option<Vec<f32>> {
        self.0.borrow().grad.clone()
    }

    /// Returns whether gradient computation is enabled for this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let a = Tensor::zeros((2, 2));
    /// assert!(!a.requires_grad());
    ///
    /// let b = Tensor::from_vec(vec![1.0; 4], (2, 2), true);
    /// assert!(b.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    /// Returns the `(rows, cols)` shape of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::zeros((5, 7));
    /// assert_eq!(t.shape(), (5, 7));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        self.0.borrow().shape
    }

    /// Returns the total number of elements in the tensor (`rows * cols`).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::zeros((3, 4));
    /// assert_eq!(t.numel(), 12);
    /// ```
    pub fn numel(&self) -> usize {
        let s = self.0.borrow().shape;
        s.0 * s.1
    }

    // -----------------------------------------------------------------------
    // Backward helpers
    // -----------------------------------------------------------------------

    /// Adds an incoming gradient slice into this tensor's gradient buffer.
    ///
    /// If the gradient buffer has not been allocated yet, it is initialised to zeros
    /// before accumulating. This mirrors the behaviour of [`GradientAccumulator::accumulate`]
    /// but operates on the tensor's own gradient field.
    ///
    /// This method is called by the backward traversal code whenever a downstream
    /// operation propagates a gradient through this tensor.
    ///
    /// # Arguments
    ///
    /// * `grad` - Incoming gradient slice; must have the same length as [`Tensor::numel`].
    ///
    /// # Panics
    ///
    /// Panics if `grad.len()` does not match the tensor's element count.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    /// t.accumulate_grad(&[0.1, 0.2, 0.3]);
    /// t.accumulate_grad(&[0.1, 0.2, 0.3]);
    /// let g = t.grad().unwrap();
    /// assert!((g[0] - 0.2).abs() < 1e-6);
    /// assert!((g[1] - 0.4).abs() < 1e-6);
    /// assert!((g[2] - 0.6).abs() < 1e-6);
    /// ```
    pub fn accumulate_grad(&self, grad: &[f32]) {
        let mut inner = self.0.borrow_mut();
        let n = inner.shape.0 * inner.shape.1;
        assert_eq!(
            grad.len(),
            n,
            "accumulate_grad: gradient length {} does not match tensor numel {}",
            grad.len(),
            n
        );
        if inner.grad.is_none() {
            inner.grad = Some(vec![0.0f32; n]);
        }
        let g = inner.grad.as_mut().unwrap();
        for (acc, &dg) in g.iter_mut().zip(grad.iter()) {
            *acc += dg;
        }
    }

    /// Resets the gradient buffer to zero (sets all entries to 0.0).
    ///
    /// If the gradient buffer has not been allocated, this is a no-op.
    /// After calling `zero_grad`, the gradient is `Some(vec![0.0; n])` (not `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
    /// t.accumulate_grad(&[3.0, 4.0]);
    /// t.zero_grad();
    /// let g = t.grad().unwrap();
    /// assert_eq!(g, vec![0.0f32, 0.0]);
    /// ```
    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        if let Some(ref mut g) = inner.grad {
            g.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Creates a detached leaf copy of this tensor with no gradient history.
    ///
    /// The returned tensor contains the same data and shape as `self`, but has
    /// `requires_grad = false` and no backward node. This severs the connection
    /// to the computational graph, making it safe to use the data for parameter
    /// updates or other operations without accumulating gradients into the graph.
    ///
    /// This mirrors the behaviour of `Tensor.detach()` in PyTorch: the detached
    /// tensor is a completely independent leaf that never participates in backward
    /// propagation.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    /// use rust_neural_networks::autograd::ops::tensor_add;
    ///
    /// let w = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
    /// let detached = w.detach();
    ///
    /// // Detached tensor has the same values but no grad tracking.
    /// assert_eq!(detached.data(), vec![1.0f32, 2.0, 3.0]);
    /// assert_eq!(detached.shape(), (1, 3));
    /// assert!(!detached.requires_grad());
    /// assert!(detached.grad().is_none());
    ///
    /// // Operations on detached tensor do not record grad nodes.
    /// let b = Tensor::from_vec(vec![1.0, 1.0, 1.0], (1, 3), false);
    /// let out = tensor_add(&detached, &b);
    /// assert!(!out.requires_grad());
    /// ```
    pub fn detach(&self) -> Tensor {
        let inner = self.0.borrow();
        Tensor(Rc::new(RefCell::new(TensorInner {
            data: inner.data.clone(),
            grad: None,
            shape: inner.shape,
            requires_grad: false,
            grad_node: None,
        })))
    }

    // -----------------------------------------------------------------------
    // Backward pass
    // -----------------------------------------------------------------------

    /// Runs reverse-mode automatic differentiation from this tensor (the loss root).
    ///
    /// This implements the backward pass of backpropagation:
    ///
    /// 1. Initialises this tensor's gradient to all-ones — the standard seed for
    ///    a scalar loss (d(loss)/d(loss) = 1).
    /// 2. Calls [`GradNode::build_topo`] to collect all ancestor tensors in
    ///    **reverse topological order** (this tensor first, leaves last).
    /// 3. Iterates through the nodes front-to-back and, for each tensor that has
    ///    a [`GradNode`] recording its creating operation, reads the tensor's
    ///    accumulated upstream gradient and dispatches to
    ///    [`crate::autograd::ops::backward_op`] which applies the chain rule and
    ///    calls [`Tensor::accumulate_grad`] on the input tensors.
    ///
    /// After `backward` returns, every tensor in the computation graph that has
    /// `requires_grad = true` will have its gradient stored in [`Tensor::grad`].
    ///
    /// # Notes
    ///
    /// This method is designed for **scalar** root tensors (shape `(1, 1)` or any
    /// single-element shape). For non-scalar roots the gradient seed is still set
    /// to all-ones, which corresponds to summing all output gradients — this is
    /// correct for loss functions but not for general vector-valued functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::autograd::tensor::Tensor;
    /// use rust_neural_networks::autograd::ops::tensor_mse_loss;
    ///
    /// let pred = Tensor::from_vec(vec![2.0], (1, 1), true);
    /// let tgt  = Tensor::from_vec(vec![1.0], (1, 1), false);
    /// let loss = tensor_mse_loss(&pred, &tgt);
    /// loss.backward();
    ///
    /// // MSE gradient: 2 * (pred - tgt) / n = 2 * (2 - 1) / 1 = 2.0
    /// let g = pred.grad().unwrap();
    /// assert!((g[0] - 2.0).abs() < 1e-5, "grad = {}", g[0]);
    /// ```
    pub fn backward(&self) {
        use crate::autograd::ops::backward_op;

        // Step 1: Initialise this tensor's gradient to ones (loss seed).
        let n = self.numel();
        {
            let mut inner = self.0.borrow_mut();
            inner.grad = Some(vec![1.0f32; n]);
        }

        // Step 2: Collect all ancestor tensors in reverse topological order.
        let topo = GradNode::build_topo(self);

        // Step 3: Propagate gradients through each node front-to-back.
        for node in &topo {
            // Read the upstream gradient accumulated into this node.
            let upstream_grad = match node.grad() {
                Some(g) => g,
                None => continue, // No upstream gradient yet; skip.
            };

            // Extract the op and input tensor handles before releasing the borrow.
            // Cloning Tensor is cheap (Rc pointer copy); cloning Op is cheap too.
            let (op, inputs) = {
                let inner = node.0.borrow();
                match inner.grad_node.as_ref() {
                    Some(grad_node) => (grad_node.op.clone(), grad_node.inputs.clone()),
                    None => continue, // Leaf tensor: no backward operation.
                }
            };

            // Dispatch to the per-operation backward gradient rule.
            backward_op(&op, &inputs, &upstream_grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        assert_eq!(t.shape(), (2, 2));
        assert_eq!(t.data(), vec![1.0f32, 2.0, 3.0, 4.0]);
        assert!(!t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros((3, 4));
        assert_eq!(t.shape(), (3, 4));
        assert_eq!(t.numel(), 12);
        assert_eq!(t.data(), vec![0.0f32; 12]);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_tensor_from_vec_no_grad() {
        let t = Tensor::from_vec(vec![0.5, -0.3], (1, 2), false);
        assert!(!t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_from_vec_with_grad() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_tensor_accumulate_grad_first_call() {
        let t = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        assert!(t.grad().is_none());
        t.accumulate_grad(&[0.1, 0.2]);
        let g = t.grad().unwrap();
        assert!((g[0] - 0.1).abs() < 1e-6);
        assert!((g[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_accumulate_grad_twice() {
        let t = Tensor::from_vec(vec![0.0; 3], (1, 3), true);
        t.accumulate_grad(&[1.0, 2.0, 3.0]);
        t.accumulate_grad(&[0.5, 0.5, 0.5]);
        let g = t.grad().unwrap();
        assert!((g[0] - 1.5).abs() < 1e-6);
        assert!((g[1] - 2.5).abs() < 1e-6);
        assert!((g[2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_zero_grad() {
        let t = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        t.accumulate_grad(&[5.0, 6.0]);
        t.zero_grad();
        let g = t.grad().unwrap();
        assert_eq!(g, vec![0.0f32, 0.0]);
    }

    #[test]
    fn test_tensor_clone_shares_inner() {
        let a = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        let b = a.clone();
        a.accumulate_grad(&[1.0, 1.0]);
        // b shares the same inner, so it should see the gradient too
        let g = b.grad().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_tensor_new_wrong_length() {
        let _ = Tensor::new(vec![1.0, 2.0], (2, 2)); // 2 != 4
    }

    #[test]
    #[should_panic(expected = "accumulate_grad: gradient length")]
    fn test_tensor_accumulate_grad_wrong_length() {
        let t = Tensor::from_vec(vec![1.0, 2.0], (1, 2), true);
        t.accumulate_grad(&[1.0, 2.0, 3.0]); // 3 != 2
    }
}
