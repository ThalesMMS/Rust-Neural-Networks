//! Multi-Head Attention layer implementation
//!
//! This module provides a MultiHeadAttentionLayer that implements the scaled
//! dot-product attention mechanism from "Attention is All You Need" (Vaswani et al., 2017).
//!
//! The multi-head attention mechanism allows the model to jointly attend to information
//! from different representation subspaces at different positions.

use crate::layers::Layer;
use crate::optimizers::Optimizer;
use crate::utils::rng::SimpleRng;
use std::any::Any;
use std::cell::RefCell;

/// Multi-Head Attention layer with learnable Q/K/V projection matrices.
///
/// Implements the scaled dot-product attention mechanism:
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// where Q (queries), K (keys), and V (values) are linear projections of the input.
/// With multiple heads, the input is projected to Q/K/V, split into num_heads,
/// attention is computed per head, and results are concatenated and projected.
///
/// # Architecture
///
/// For input shape [batch_size, seq_len, d_model]:
/// 1. Linear projections: Q = XW_q, K = XW_k, V = XW_v
/// 2. Split into heads: reshape to [batch, seq_len, num_heads, d_head]
/// 3. Compute attention per head: softmax(QK^T / sqrt(d_head)) * V
/// 4. Concatenate heads: reshape to [batch, seq_len, d_model]
/// 5. Output projection: output = concat(heads) * W_o
///
/// # Fields
///
/// * `d_model` - Model dimension (input/output feature size)
/// * `num_heads` - Number of attention heads
/// * `d_head` - Dimension per head (d_model / num_heads)
/// * `w_q`, `w_k`, `w_v` - Query/Key/Value projection matrices (d_model × d_model)
/// * `b_q`, `b_k`, `b_v` - Query/Key/Value biases (d_model)
/// * `w_o` - Output projection matrix (d_model × d_model)
/// * `b_o` - Output bias (d_model)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::MultiHeadAttentionLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = MultiHeadAttentionLayer::new(512, 8, &mut rng);
/// assert_eq!(layer.input_size(), 512);
/// assert_eq!(layer.output_size(), 512);
/// assert_eq!(layer.num_heads(), 8);
/// ```
pub struct MultiHeadAttentionLayer {
    d_model: usize,
    num_heads: usize,
    d_head: usize,

    // Q/K/V projection weights and biases
    w_q: Vec<f32>,
    b_q: Vec<f32>,
    w_k: Vec<f32>,
    b_k: Vec<f32>,
    w_v: Vec<f32>,
    b_v: Vec<f32>,

    // Output projection
    w_o: Vec<f32>,
    b_o: Vec<f32>,

    // Gradient accumulators (mutable interior via RefCell)
    grad_w_q: RefCell<Vec<f32>>,
    grad_b_q: RefCell<Vec<f32>>,
    grad_w_k: RefCell<Vec<f32>>,
    grad_b_k: RefCell<Vec<f32>>,
    grad_w_v: RefCell<Vec<f32>>,
    grad_b_v: RefCell<Vec<f32>>,
    grad_w_o: RefCell<Vec<f32>>,
    grad_b_o: RefCell<Vec<f32>>,

    // Cached activations for backward pass (per thread/batch)
    // These are stored as RefCell to allow interior mutability in forward pass
    cached_q: RefCell<Vec<f32>>,
    cached_k: RefCell<Vec<f32>>,
    cached_v: RefCell<Vec<f32>>,
    cached_attn_weights: RefCell<Vec<f32>>,
    cached_attn_out: RefCell<Vec<f32>>,
    cached_batch_size: RefCell<usize>,
    cached_seq_len: RefCell<usize>,
}

impl MultiHeadAttentionLayer {
    /// Creates a new multi-head attention layer with Xavier-initialized weights.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Panics
    ///
    /// Panics if d_model is not divisible by num_heads.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let layer = MultiHeadAttentionLayer::new(64, 4, &mut rng);
    /// assert_eq!(layer.num_heads(), 4);
    /// assert_eq!(layer.d_head(), 16);
    /// ```
    #[allow(clippy::manual_is_multiple_of)]
    pub fn new(d_model: usize, num_heads: usize, rng: &mut SimpleRng) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        let d_head = d_model / num_heads;

        // Xavier initialization for all weight matrices
        let limit = (6.0f32 / (d_model + d_model) as f32).sqrt();

        let mut init_weights = |size: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; size];
            for v in w.iter_mut() {
                *v = rng.gen_range_f32(-limit, limit);
            }
            w
        };

        Self {
            d_model,
            num_heads,
            d_head,

            // Initialize projection weights
            w_q: init_weights(d_model * d_model),
            b_q: vec![0.0f32; d_model],
            w_k: init_weights(d_model * d_model),
            b_k: vec![0.0f32; d_model],
            w_v: init_weights(d_model * d_model),
            b_v: vec![0.0f32; d_model],

            // Initialize output projection
            w_o: init_weights(d_model * d_model),
            b_o: vec![0.0f32; d_model],

            // Initialize gradient accumulators
            grad_w_q: RefCell::new(vec![0.0f32; d_model * d_model]),
            grad_b_q: RefCell::new(vec![0.0f32; d_model]),
            grad_w_k: RefCell::new(vec![0.0f32; d_model * d_model]),
            grad_b_k: RefCell::new(vec![0.0f32; d_model]),
            grad_w_v: RefCell::new(vec![0.0f32; d_model * d_model]),
            grad_b_v: RefCell::new(vec![0.0f32; d_model]),
            grad_w_o: RefCell::new(vec![0.0f32; d_model * d_model]),
            grad_b_o: RefCell::new(vec![0.0f32; d_model]),

            // Initialize cache (will be resized as needed)
            cached_q: RefCell::new(Vec::new()),
            cached_k: RefCell::new(Vec::new()),
            cached_v: RefCell::new(Vec::new()),
            cached_attn_weights: RefCell::new(Vec::new()),
            cached_attn_out: RefCell::new(Vec::new()),
            cached_batch_size: RefCell::new(0),
            cached_seq_len: RefCell::new(0),
        }
    }

    /// Returns the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the dimension per attention head.
    pub fn d_head(&self) -> usize {
        self.d_head
    }

    /// Returns the model dimension (d_model).
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Returns a reference to the cached attention weights from the last forward pass.
    ///
    /// The returned weights have shape `[batch_size * num_heads, seq_len, seq_len]`
    /// (flattened). This is useful for visualization of attention patterns.
    ///
    /// # Panics
    ///
    /// Panics if called before any forward pass (the cache will be empty).
    pub fn get_attention_weights(&self) -> std::cell::Ref<'_, Vec<f32>> {
        self.cached_attn_weights.borrow()
    }

    /// Softmax in-place for a single vector.
    fn softmax_inplace(v: &mut [f32]) {
        if v.is_empty() {
            return;
        }

        let mut maxv = v[0];
        for &x in v.iter().skip(1) {
            if x > maxv {
                maxv = x;
            }
        }

        let mut sum = 0.0f32;
        for x in v.iter_mut() {
            *x = (*x - maxv).exp();
            sum += *x;
        }

        if sum > 0.0 {
            let inv = 1.0f32 / sum;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    /// Compute Q/K/V projections: Q = XW_q + b_q (and similarly for K, V)
    /// Input shape: [batch_size * seq_len, d_model]
    /// Output shape: [batch_size * seq_len, d_model]
    fn compute_qkv_projections(
        &self,
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
    ) {
        let total_tokens = batch_size * seq_len;

        // Initialize outputs
        q.fill(0.0);
        k.fill(0.0);
        v.fill(0.0);

        // For each token in the batch
        for tok_idx in 0..total_tokens {
            let input_base = tok_idx * self.d_model;
            let output_base = tok_idx * self.d_model;

            // Compute Q, K, V projections for this token
            for d_out in 0..self.d_model {
                let mut sum_q = self.b_q[d_out];
                let mut sum_k = self.b_k[d_out];
                let mut sum_v = self.b_v[d_out];

                for d_in in 0..self.d_model {
                    let x = input[input_base + d_in];
                    // Weights are stored in row-major: [d_in * d_model + d_out]
                    sum_q += x * self.w_q[d_in * self.d_model + d_out];
                    sum_k += x * self.w_k[d_in * self.d_model + d_out];
                    sum_v += x * self.w_v[d_in * self.d_model + d_out];
                }

                q[output_base + d_out] = sum_q;
                k[output_base + d_out] = sum_k;
                v[output_base + d_out] = sum_v;
            }
        }
    }

    /// Compute multi-head scaled dot-product attention.
    /// Q, K, V shape: [batch_size * seq_len, d_model]
    /// Output shape: [batch_size * seq_len, d_model]
    /// Attention weights shape: [batch_size * num_heads, seq_len, seq_len]
    #[allow(clippy::too_many_arguments)]
    fn compute_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_size: usize,
        seq_len: usize,
        attn_out: &mut [f32],
        attn_weights: &mut [f32],
    ) {
        let inv_sqrt_d = 1.0f32 / (self.d_head as f32).sqrt();

        // Initialize outputs
        attn_out.fill(0.0);
        attn_weights.fill(0.0);

        // Process each head separately
        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                // For each query position
                for i in 0..seq_len {
                    let attn_row_base = ((batch * self.num_heads + head) * seq_len + i) * seq_len;

                    // Compute attention scores: Q * K^T / sqrt(d_head)
                    for j in 0..seq_len {
                        let q_base = (batch * seq_len + i) * self.d_model + head * self.d_head;
                        let k_base = (batch * seq_len + j) * self.d_model + head * self.d_head;

                        let mut score = 0.0f32;
                        for d in 0..self.d_head {
                            score += q[q_base + d] * k[k_base + d];
                        }

                        attn_weights[attn_row_base + j] = score * inv_sqrt_d;
                    }

                    // Apply softmax to get attention probabilities
                    Self::softmax_inplace(
                        &mut attn_weights[attn_row_base..attn_row_base + seq_len],
                    );

                    // Compute weighted sum of values
                    let out_base = (batch * seq_len + i) * self.d_model + head * self.d_head;
                    for j in 0..seq_len {
                        let alpha = attn_weights[attn_row_base + j];
                        let v_base = (batch * seq_len + j) * self.d_model + head * self.d_head;

                        for d in 0..self.d_head {
                            attn_out[out_base + d] += alpha * v[v_base + d];
                        }
                    }
                }
            }
        }
    }

    /// Apply output projection: output = attn_out * W_o + b_o
    fn apply_output_projection(
        &self,
        attn_out: &[f32],
        batch_size: usize,
        seq_len: usize,
        output: &mut [f32],
    ) {
        let total_tokens = batch_size * seq_len;

        output.fill(0.0);

        for tok_idx in 0..total_tokens {
            let input_base = tok_idx * self.d_model;
            let output_base = tok_idx * self.d_model;

            for d_out in 0..self.d_model {
                let mut sum = self.b_o[d_out];

                for d_in in 0..self.d_model {
                    sum += attn_out[input_base + d_in] * self.w_o[d_in * self.d_model + d_out];
                }

                output[output_base + d_out] = sum;
            }
        }
    }
}

impl Layer for MultiHeadAttentionLayer {
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        // Input shape: [batch_size * seq_len, d_model]
        // We infer seq_len from the input size
        let total_size = input.len();
        assert_eq!(
            total_size % (batch_size * self.d_model),
            0,
            "Input size must be batch_size * seq_len * d_model"
        );
        let seq_len = total_size / (batch_size * self.d_model);

        // Resize caches if needed
        let total_tokens = batch_size * seq_len;
        let qkv_size = total_tokens * self.d_model;
        let attn_size = batch_size * self.num_heads * seq_len * seq_len;

        {
            let mut q = self.cached_q.borrow_mut();
            let mut k = self.cached_k.borrow_mut();
            let mut v = self.cached_v.borrow_mut();
            let mut attn_weights = self.cached_attn_weights.borrow_mut();
            let mut attn_out = self.cached_attn_out.borrow_mut();

            q.resize(qkv_size, 0.0);
            k.resize(qkv_size, 0.0);
            v.resize(qkv_size, 0.0);
            attn_weights.resize(attn_size, 0.0);
            attn_out.resize(qkv_size, 0.0);

            *self.cached_batch_size.borrow_mut() = batch_size;
            *self.cached_seq_len.borrow_mut() = seq_len;

            // Step 1: Compute Q/K/V projections
            self.compute_qkv_projections(input, batch_size, seq_len, &mut q, &mut k, &mut v);

            // Step 2: Compute multi-head attention
            self.compute_attention(
                &q,
                &k,
                &v,
                batch_size,
                seq_len,
                &mut attn_out,
                &mut attn_weights,
            );

            // Step 3: Apply output projection
            self.apply_output_projection(&attn_out, batch_size, seq_len, output);
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        let seq_len = *self.cached_seq_len.borrow();
        let total_tokens = batch_size * seq_len;

        // Get cached values from forward pass
        let q = self.cached_q.borrow();
        let k = self.cached_k.borrow();
        let v = self.cached_v.borrow();
        let attn_weights = self.cached_attn_weights.borrow();
        let attn_out = self.cached_attn_out.borrow();

        // Gradient buffers
        let mut grad_attn_out = vec![0.0f32; total_tokens * self.d_model];
        let mut grad_q = vec![0.0f32; total_tokens * self.d_model];
        let mut grad_k = vec![0.0f32; total_tokens * self.d_model];
        let mut grad_v = vec![0.0f32; total_tokens * self.d_model];

        // Get mutable references to gradient accumulators
        let mut grad_w_o = self.grad_w_o.borrow_mut();
        let mut grad_b_o = self.grad_b_o.borrow_mut();
        let mut grad_w_q = self.grad_w_q.borrow_mut();
        let mut grad_b_q = self.grad_b_q.borrow_mut();
        let mut grad_w_k = self.grad_w_k.borrow_mut();
        let mut grad_b_k = self.grad_b_k.borrow_mut();
        let mut grad_w_v = self.grad_w_v.borrow_mut();
        let mut grad_b_v = self.grad_b_v.borrow_mut();

        // Step 1: Backprop through output projection
        // grad_attn_out and gradient for W_o, b_o
        for tok_idx in 0..total_tokens {
            let grad_out_base = tok_idx * self.d_model;
            let attn_out_base = tok_idx * self.d_model;

            for d_out in 0..self.d_model {
                let grad = grad_output[grad_out_base + d_out];
                grad_b_o[d_out] += grad;

                for d_in in 0..self.d_model {
                    // Gradient w.r.t. W_o
                    grad_w_o[d_in * self.d_model + d_out] += attn_out[attn_out_base + d_in] * grad;

                    // Gradient w.r.t. attn_out (backprop to attention layer)
                    grad_attn_out[attn_out_base + d_in] +=
                        grad * self.w_o[d_in * self.d_model + d_out];
                }
            }
        }

        // Step 2: Backprop through attention mechanism
        let inv_sqrt_d = 1.0f32 / (self.d_head as f32).sqrt();

        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                for i in 0..seq_len {
                    let attn_row_base = ((batch * self.num_heads + head) * seq_len + i) * seq_len;
                    let out_base = (batch * seq_len + i) * self.d_model + head * self.d_head;

                    // Gradient w.r.t. V (from weighted sum)
                    for j in 0..seq_len {
                        let alpha = attn_weights[attn_row_base + j];
                        let v_base = (batch * seq_len + j) * self.d_model + head * self.d_head;

                        for d in 0..self.d_head {
                            grad_v[v_base + d] += alpha * grad_attn_out[out_base + d];
                        }
                    }

                    // Gradient w.r.t. attention weights (alpha)
                    let mut grad_alpha = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let v_base = (batch * seq_len + j) * self.d_model + head * self.d_head;
                        let mut dot = 0.0f32;
                        for d in 0..self.d_head {
                            dot += grad_attn_out[out_base + d] * v[v_base + d];
                        }
                        grad_alpha[j] = dot;
                    }

                    // Backprop through softmax
                    let mut grad_scores = vec![0.0f32; seq_len];
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        let alpha = attn_weights[attn_row_base + j];
                        sum += grad_alpha[j] * alpha;
                    }
                    for j in 0..seq_len {
                        let alpha = attn_weights[attn_row_base + j];
                        grad_scores[j] = alpha * (grad_alpha[j] - sum);
                    }

                    // Gradient w.r.t. Q and K (from scores = Q * K^T / sqrt(d_head))
                    for j in 0..seq_len {
                        let q_base = (batch * seq_len + i) * self.d_model + head * self.d_head;
                        let k_base = (batch * seq_len + j) * self.d_model + head * self.d_head;
                        let ds = grad_scores[j] * inv_sqrt_d;

                        for d in 0..self.d_head {
                            grad_q[q_base + d] += ds * k[k_base + d];
                            grad_k[k_base + d] += ds * q[q_base + d];
                        }
                    }
                }
            }
        }

        // Step 3: Backprop through Q/K/V projections
        grad_input.fill(0.0);

        for tok_idx in 0..total_tokens {
            let input_base = tok_idx * self.d_model;

            for d_out in 0..self.d_model {
                let grad_q_val = grad_q[input_base + d_out];
                let grad_k_val = grad_k[input_base + d_out];
                let grad_v_val = grad_v[input_base + d_out];

                grad_b_q[d_out] += grad_q_val;
                grad_b_k[d_out] += grad_k_val;
                grad_b_v[d_out] += grad_v_val;

                for d_in in 0..self.d_model {
                    let x = input[input_base + d_in];

                    // Gradients for W_q, W_k, W_v
                    grad_w_q[d_in * self.d_model + d_out] += x * grad_q_val;
                    grad_w_k[d_in * self.d_model + d_out] += x * grad_k_val;
                    grad_w_v[d_in * self.d_model + d_out] += x * grad_v_val;

                    // Gradient w.r.t. input
                    grad_input[input_base + d_in] += grad_q_val
                        * self.w_q[d_in * self.d_model + d_out]
                        + grad_k_val * self.w_k[d_in * self.d_model + d_out]
                        + grad_v_val * self.w_v[d_in * self.d_model + d_out];
                }
            }
        }
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        let mut grad_w_q = self.grad_w_q.borrow_mut();
        let mut grad_b_q = self.grad_b_q.borrow_mut();
        let mut grad_w_k = self.grad_w_k.borrow_mut();
        let mut grad_b_k = self.grad_b_k.borrow_mut();
        let mut grad_w_v = self.grad_w_v.borrow_mut();
        let mut grad_b_v = self.grad_b_v.borrow_mut();
        let mut grad_w_o = self.grad_w_o.borrow_mut();
        let mut grad_b_o = self.grad_b_o.borrow_mut();

        // Update W_q and b_q
        for i in 0..self.w_q.len() {
            self.w_q[i] -= learning_rate * grad_w_q[i];
            grad_w_q[i] = 0.0;
        }
        for i in 0..self.b_q.len() {
            self.b_q[i] -= learning_rate * grad_b_q[i];
            grad_b_q[i] = 0.0;
        }

        // Update W_k and b_k
        for i in 0..self.w_k.len() {
            self.w_k[i] -= learning_rate * grad_w_k[i];
            grad_w_k[i] = 0.0;
        }
        for i in 0..self.b_k.len() {
            self.b_k[i] -= learning_rate * grad_b_k[i];
            grad_b_k[i] = 0.0;
        }

        // Update W_v and b_v
        for i in 0..self.w_v.len() {
            self.w_v[i] -= learning_rate * grad_w_v[i];
            grad_w_v[i] = 0.0;
        }
        for i in 0..self.b_v.len() {
            self.b_v[i] -= learning_rate * grad_b_v[i];
            grad_b_v[i] = 0.0;
        }

        // Update W_o and b_o
        for i in 0..self.w_o.len() {
            self.w_o[i] -= learning_rate * grad_w_o[i];
            grad_w_o[i] = 0.0;
        }
        for i in 0..self.b_o.len() {
            self.b_o[i] -= learning_rate * grad_b_o[i];
            grad_b_o[i] = 0.0;
        }
    }

    fn update_with_optimizer(&mut self, optimizer: &mut dyn Optimizer) {
        let mut grad_w_q = self.grad_w_q.borrow_mut();
        let mut grad_b_q = self.grad_b_q.borrow_mut();
        let mut grad_w_k = self.grad_w_k.borrow_mut();
        let mut grad_b_k = self.grad_b_k.borrow_mut();
        let mut grad_w_v = self.grad_w_v.borrow_mut();
        let mut grad_b_v = self.grad_b_v.borrow_mut();
        let mut grad_w_o = self.grad_w_o.borrow_mut();
        let mut grad_b_o = self.grad_b_o.borrow_mut();

        // Update with optimizer
        optimizer.update(&mut self.w_q, &grad_w_q);
        optimizer.update(&mut self.b_q, &grad_b_q);
        optimizer.update(&mut self.w_k, &grad_w_k);
        optimizer.update(&mut self.b_k, &grad_b_k);
        optimizer.update(&mut self.w_v, &grad_w_v);
        optimizer.update(&mut self.b_v, &grad_b_v);
        optimizer.update(&mut self.w_o, &grad_w_o);
        optimizer.update(&mut self.b_o, &grad_b_o);

        // Clear gradients
        grad_w_q.fill(0.0);
        grad_b_q.fill(0.0);
        grad_w_k.fill(0.0);
        grad_b_k.fill(0.0);
        grad_w_v.fill(0.0);
        grad_b_v.fill(0.0);
        grad_w_o.fill(0.0);
        grad_b_o.fill(0.0);
    }

    fn input_size(&self) -> usize {
        self.d_model
    }

    fn output_size(&self) -> usize {
        self.d_model
    }

    fn parameter_count(&self) -> usize {
        // 4 weight matrices (Q, K, V, O): each d_model × d_model
        // 4 bias vectors (Q, K, V, O): each d_model
        4 * (self.d_model * self.d_model + self.d_model)
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let mut rng = SimpleRng::new(42);
        let layer = MultiHeadAttentionLayer::new(64, 4, &mut rng);
        assert_eq!(layer.d_model(), 64);
        assert_eq!(layer.num_heads(), 4);
        assert_eq!(layer.d_head(), 16);
    }

    #[test]
    #[should_panic(expected = "must be divisible by")]
    fn test_invalid_heads() {
        let mut rng = SimpleRng::new(42);
        let _ = MultiHeadAttentionLayer::new(64, 5, &mut rng);
    }

    #[test]
    fn test_forward_shape() {
        let mut rng = SimpleRng::new(42);
        let layer = MultiHeadAttentionLayer::new(32, 4, &mut rng);

        let batch_size = 2;
        let seq_len = 10;
        let input = vec![1.0f32; batch_size * seq_len * 32];
        let mut output = vec![0.0f32; batch_size * seq_len * 32];

        layer.forward(&input, &mut output, batch_size);

        // Output should have same shape as input
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let layer = MultiHeadAttentionLayer::new(64, 8, &mut rng);

        // 4 matrices (Q, K, V, O): 64 * 64 each
        // 4 biases: 64 each
        let expected = 4 * (64 * 64 + 64);
        assert_eq!(layer.parameter_count(), expected);
    }
}
