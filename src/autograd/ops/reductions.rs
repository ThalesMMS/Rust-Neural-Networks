use crate::autograd::tape::{GradNode, Op};
use crate::autograd::tensor::Tensor;

// ---------------------------------------------------------------------------
// tensor_sum
// ---------------------------------------------------------------------------

/// Reduces all elements of a tensor to a scalar by summation.
///
/// The output is a `(1, 1)` tensor holding the sum of all input elements.
/// This is primarily used as a loss reduction step (e.g. after computing
/// per-element losses, sum to get a single scalar before calling backward).
///
/// If the input `requires_grad`, the output records an [`Op::Sum`] node.
/// During backward, the scalar gradient is broadcast back to every element
/// of the input (`grad_a[i] = grad_out[0]`).
///
/// # Arguments
///
/// * `a` - Input tensor of any shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_sum;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
/// let s = tensor_sum(&a);
/// assert_eq!(s.shape(), (1, 1));
/// assert!((s.data()[0] - 10.0).abs() < 1e-6);
/// assert!(s.requires_grad());
/// ```
pub fn tensor_sum(a: &Tensor) -> Tensor {
    let data_a = a.data();
    let total: f32 = data_a.iter().sum();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(vec![total], (1, 1), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::Sum, vec![a.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_mean
// ---------------------------------------------------------------------------

/// Computes the arithmetic mean of all elements in `a` and returns it as a `(1, 1)` tensor.
///
/// If `a.requires_grad()` is true, the output records an `Op::Mean` autograd node so that during
/// backpropagation each input element receives `grad_out[0] / n`, where `n` is the total number
/// of elements in `a`.
///
/// # Panics
///
/// Panics if `a` is an empty tensor.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_mean;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
/// let m = tensor_mean(&a);
/// assert_eq!(m.shape(), (1, 1));
/// assert!((m.data()[0] - 2.5).abs() < 1e-6);
/// assert!(m.requires_grad());
/// ```
pub fn tensor_mean(a: &Tensor) -> Tensor {
    let data_a = a.data();
    let n = data_a.len();
    assert!(n > 0, "tensor_mean: empty tensor");
    let total: f32 = data_a.iter().sum();
    let mean = total / n as f32;

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(vec![mean], (1, 1), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(Op::Mean, vec![a.clone()])));
    }

    out
}
