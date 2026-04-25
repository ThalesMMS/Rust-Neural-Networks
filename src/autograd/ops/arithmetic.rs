use crate::autograd::tape::{GradNode, Op};
use crate::autograd::tensor::Tensor;

// ---------------------------------------------------------------------------
// tensor_add
// ---------------------------------------------------------------------------

/// Element-wise addition of two tensors: `out[i] = a[i] + b[i]`.
///
/// Both tensors must have the same shape. If either input `requires_grad`,
/// the output records an [`Op::Add`] node so gradients can be propagated
/// to both inputs during backward.
///
/// # Arguments
///
/// * `a` - First input tensor.
/// * `b` - Second input tensor (same shape as `a`).
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_add;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
/// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], (1, 3), true);
/// let out = tensor_add(&a, &b);
/// assert_eq!(out.data(), vec![5.0f32, 7.0, 9.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert_eq!(
        shape_a, shape_b,
        "tensor_add: shape mismatch {:?} vs {:?}",
        shape_a, shape_b
    );

    let data_a = a.data();
    let data_b = b.data();
    let out_data: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x + y)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, shape_a, requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node =
            Some(Box::new(GradNode::new(Op::Add, vec![a.clone(), b.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_sub
// ---------------------------------------------------------------------------

/// Element-wise subtraction of two tensors: `out[i] = a[i] - b[i]`.
///
/// Both tensors must have the same shape. If either input `requires_grad`,
/// the output records an [`Op::Sub`] node.
///
/// # Panics
///
/// Panics if `a` and `b` do not have the same shape.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_sub;
///
/// let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], (1, 3), true);
/// let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), false);
/// let out = tensor_sub(&a, &b);
/// assert_eq!(out.data(), vec![4.0f32, 5.0, 6.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_sub(a: &Tensor, b: &Tensor) -> Tensor {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert_eq!(
        shape_a, shape_b,
        "tensor_sub: shape mismatch {:?} vs {:?}",
        shape_a, shape_b
    );

    let data_a = a.data();
    let data_b = b.data();
    let out_data: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(x, y)| x - y)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let out = Tensor::from_vec(out_data, shape_a, requires_grad);

    if requires_grad {
        out.0.borrow_mut().grad_node =
            Some(Box::new(GradNode::new(Op::Sub, vec![a.clone(), b.clone()])));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_mul_scalar
// ---------------------------------------------------------------------------

/// Multiplies every element of `a` by a scalar constant: `out[i] = a[i] * scalar`.
///
/// Internally the scalar is broadcast into a constant tensor of the same shape
/// as `a` and stored as the second input of an [`Op::Mul`] node.  Because the
/// scalar tensor does not `require_grad`, the backward pass will only propagate
/// gradients to `a`.
///
/// # Arguments
///
/// * `a`      - Input tensor.
/// * `scalar` - The scalar multiplier.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_mul_scalar;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], (1, 3), true);
/// let out = tensor_mul_scalar(&a, 2.0);
/// assert_eq!(out.data(), vec![2.0f32, 4.0, 6.0]);
/// assert!(out.requires_grad());
/// ```
pub fn tensor_mul_scalar(a: &Tensor, scalar: f32) -> Tensor {
    let shape = a.shape();
    let data_a = a.data();
    let out_data: Vec<f32> = data_a.iter().map(|x| x * scalar).collect();

    let requires_grad = a.requires_grad();
    let out = Tensor::from_vec(out_data, shape, requires_grad);

    if requires_grad {
        // Create a constant tensor holding the broadcast scalar value.
        // requires_grad = false so backward will not accumulate into it.
        let scalar_tensor = Tensor::new(vec![scalar; a.numel()], shape);
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Mul,
            vec![a.clone(), scalar_tensor],
        )));
    }

    out
}

// ---------------------------------------------------------------------------
// tensor_add_bias
// ---------------------------------------------------------------------------

/// Adds a bias vector to each row of a matrix.
///
/// `a` has shape `(batch, features)` and `bias` has shape `(1, features)`.
/// The bias is broadcast across the batch dimension:
///
/// ```text
/// out[r][c] = a[r][c] + bias[c]   for r in 0..batch, c in 0..features
/// ```
///
/// The output records an [`Op::Add`] node with `inputs = [a, bias]`.  During
/// backward the gradient of `bias` is accumulated by summing `grad_out` along
/// the batch axis (standard broadcast-sum rule for the backward of `+`).
///
/// # Arguments
///
/// * `a`    - Input matrix of shape `(batch, features)`.
/// * `bias` - Bias row-vector of shape `(1, features)`.
///
/// # Panics
///
/// Panics if `bias.shape().0 != 1` or if `bias.shape().1 != a.shape().1`.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::autograd::tensor::Tensor;
/// use rust_neural_networks::autograd::ops::tensor_add_bias;
///
/// let a    = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], (2, 2), true);
/// let bias = Tensor::from_vec(vec![10.0, 20.0], (1, 2), true);
/// let out  = tensor_add_bias(&a, &bias);
/// assert_eq!(out.data(), vec![11.0f32, 22.0, 13.0, 24.0]);
/// ```
pub fn tensor_add_bias(a: &Tensor, bias: &Tensor) -> Tensor {
    let (batch, features) = a.shape();
    let (bias_rows, bias_cols) = bias.shape();
    assert_eq!(
        bias_rows, 1,
        "tensor_add_bias: bias must have exactly 1 row, got {}",
        bias_rows
    );
    assert_eq!(
        bias_cols, features,
        "tensor_add_bias: bias columns {} must match a columns {}",
        bias_cols, features
    );

    let data_a = a.data();
    let data_bias = bias.data();
    let out_data: Vec<f32> = data_a
        .chunks(features)
        .flat_map(|row| row.iter().zip(data_bias.iter()).map(|(&x, &b)| x + b))
        .collect();

    let requires_grad = a.requires_grad() || bias.requires_grad();
    let out = Tensor::from_vec(out_data, (batch, features), requires_grad);

    if requires_grad {
        // Record Op::Add with inputs [a, bias].  The backward pass detects the
        // shape difference (batch × features vs 1 × features) and sums grad_out
        // along the batch axis when propagating to bias.
        out.0.borrow_mut().grad_node = Some(Box::new(GradNode::new(
            Op::Add,
            vec![a.clone(), bias.clone()],
        )));
    }

    out
}
