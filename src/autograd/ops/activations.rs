use crate::autograd::tape::{GradNode, Op};
use crate::autograd::tensor::Tensor;

fn build_activation_out<F, C>(a: &Tensor, op: Op, compute_out: F, build_cached: C) -> Tensor
where
    F: FnOnce(&[f32]) -> Vec<f32>,
    C: FnOnce(&Tensor, (usize, usize)) -> Option<Tensor>,
{
    let shape = a.shape();
    let data_a = a.data();
    let out_data = compute_out(&data_a);
    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(out_data, shape, requires_grad);

    if requires_grad {
        let mut inputs = vec![a.clone()];
        if let Some(cached) = build_cached(&out, shape) {
            inputs.push(cached);
        }
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(op, inputs)));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_relu
// ---------------------------------------------------------------------------

/// Applies the Rectified Linear Unit (ReLU) activation element-wise.
///
/// `out[i] = max(0, a[i])`
///
/// If the input `requires_grad`, the output records an [`Op::ReLU`] node with
/// the original input tensor `a` stored as `inputs[0]`.  The backward pass
/// uses the sign of `inputs[0].data[i]` to reconstruct the activation mask:
///
/// ```text
/// grad_a[i] = grad_out[i]  if a[i] > 0
///           = 0             otherwise
/// ```
///
/// # Arguments
///
/// * `a` - Input tensor.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_relu;
///
/// let a = Tensor::from_vec(vec![-1.0, 0.0, 2.0], (1, 3), true);
/// let out = tensor_relu(&a);
/// assert_eq!(out.data(), vec![0.0f32, 0.0, 2.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_relu(a: &Tensor) -> Tensor {
    build_activation_out(
        a,
        Op::ReLU,
        |data| {
            data.iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect()
        },
        |_, _| None,
    )
}

// ---------------------------------------------------------------------------
// tensor_sigmoid
// ---------------------------------------------------------------------------

/// Computes the element-wise logistic sigmoid activation.
///
/// The output tensor has the same shape and `requires_grad` as the input. If the input
/// requires gradients, the output records an `Op::Sigmoid` autograd node whose
/// `inputs[0]` is the original input tensor and whose `inputs[1]` is a cached tensor
/// containing the sigmoid outputs (used by the backward pass to compute `s * (1 - s)`
/// without recomputation).
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_sigmoid;
///
/// let a = Tensor::from_vec(vec![0.0f32], (1, 1), true);
/// let out = tensor_sigmoid(&a);
/// assert!((out.data()[0] - 0.5).abs() < 1e-6);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_sigmoid(a: &Tensor) -> Tensor {
    build_activation_out(
        a,
        Op::Sigmoid,
        |data| data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        |out, shape| Some(Tensor::new(out.data(), shape)),
    )
}

// ---------------------------------------------------------------------------
// tensor_tanh
// ---------------------------------------------------------------------------

/// Applies the hyperbolic tangent activation element-wise.
///
/// `out[i] = tanh(a[i])`
///
/// If the input `requires_grad`, the output records an [`Op::Tanh`] node.
/// `inputs[0]` is the original input `a` (gradient flows to it) and
/// `inputs[1]` is a constant tensor holding the cached tanh output values
/// (no grad) so the backward pass avoids recomputing them:
///
/// ```text
/// grad_a[i] = grad_out[i] * (1 - t[i]^2)   where t[i] = tanh(a[i])
/// ```
///
/// # Arguments
///
/// * `a` - Input tensor.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_tanh;
///
/// let a = Tensor::from_vec(vec![0.0], (1, 1), true);
/// let out = tensor_tanh(&a);
/// assert!((out.data()[0] - 0.0).abs() < 1e-6);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_tanh(a: &Tensor) -> Tensor {
    build_activation_out(
        a,
        Op::Tanh,
        |data| data.iter().map(|&x| x.tanh()).collect(),
        |out, shape| Some(Tensor::new(out.data(), shape)),
    )
}
