use crate::autograd::tape::{GradNode, Op};
use crate::autograd::tensor::Tensor;

// ---------------------------------------------------------------------------
// tensor_matmul
// ---------------------------------------------------------------------------

/// Matrix multiplication of two 2-D tensors: `out = a @ b`.
///
/// `a` must have shape `(m, k)` and `b` must have shape `(k, n)`.  The output
/// has shape `(m, n)`.  The forward pass uses explicit nested loops for
/// educational clarity (O(m·k·n)).
///
/// If either input `requires_grad`, the output records an
/// [`Op::MatMul { m, k, n }`] node.  The backward pass uses this node to
/// compute:
///
/// ```text
/// grad_a = grad_out  @  b.T    shape (m, k)
/// grad_b = a.T       @  grad_out    shape (k, n)
/// ```
///
/// # Arguments
///
/// * `a` - Left-hand matrix of shape `(m, k)`.
/// * `b` - Right-hand matrix of shape `(k, n)`.
///
/// # Panics
///
/// Panics if the inner dimensions do not match (i.e. `a.shape().1 != b.shape().0`).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_matmul;
///
/// // 2×2 identity times any matrix leaves it unchanged.
/// let eye = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], (2, 2), false);
/// let b   = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], (2, 2), true);
/// let out = tensor_matmul(&eye, &b);
/// assert_eq!(out.shape(), (2, 2));
/// assert_eq!(out.data(), vec![3.0f32, 4.0, 5.0, 6.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k_a) = a.shape();
    let (k_b, n) = b.shape();
    assert_eq!(
        k_a, k_b,
        "tensor_matmul: inner dimension mismatch — a is ({m}, {k_a}), b is ({k_b}, {n})"
    );
    let k = k_a;

    let data_a = a.data();
    let data_b = b.data();

    // Forward: O(m·k·n) nested loops for educational clarity.
    let mut out_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += data_a[i * k + p] * data_b[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }

    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, (m, n), requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::MatMul { m, k, n },
            vec![a.clone(), b.clone()],
        )));
    }

    out
}
