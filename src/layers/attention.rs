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
use cblas::{Layout, Transpose};
use std::any::Any;
use std::cell::RefCell;

#[derive(Default)]
struct AttentionScratch {
    // [total_tokens, d_model]
    grad_attn_out: Vec<f32>,
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,

    // [batch_size * num_heads, seq_len, seq_len]
    grad_attn_weights: Vec<f32>,

    // Packed contiguous per-head matrices (reused)
    // [seq_len, d_head]
    grad_context_contig: Vec<f32>,
    v_contig: Vec<f32>,
    grad_v_contig: Vec<f32>,

    // Temporary [seq_len, seq_len] matrix for grad_scores_scaled copy
    grad_scores_mat: Vec<f32>,

    // Per-(batch, head, query) temporaries
    grad_scores: Vec<f32>,
    grad_alpha: Vec<f32>,
}

impl AttentionScratch {
    fn ensure_sizes(
        &mut self,
        total_tokens: usize,
        d_model: usize,
        d_head: usize,
        attn_size: usize,
        seq_len: usize,
    ) {
        let qkv_size = total_tokens * d_model;
        self.grad_attn_out.resize(qkv_size, 0.0);
        self.grad_q.resize(qkv_size, 0.0);
        self.grad_k.resize(qkv_size, 0.0);
        self.grad_v.resize(qkv_size, 0.0);
        self.grad_attn_weights.resize(attn_size, 0.0);

        // Packed contiguous per-head matrices for GEMM (seq_len x d_head)
        let head_mat_size = seq_len * d_head;
        self.grad_context_contig.resize(head_mat_size, 0.0);
        self.v_contig.resize(head_mat_size, 0.0);
        self.grad_v_contig.resize(head_mat_size, 0.0);

        // [seq_len, seq_len]
        self.grad_scores_mat.resize(seq_len * seq_len, 0.0);

        self.grad_scores.resize(seq_len, 0.0);
        self.grad_alpha.resize(seq_len, 0.0);
    }

    fn zero_used(&mut self) {
        self.grad_attn_out.fill(0.0);
        self.grad_q.fill(0.0);
        self.grad_k.fill(0.0);
        self.grad_v.fill(0.0);
        self.grad_attn_weights.fill(0.0);
    }
}

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

    // Scratch buffers reused across backward calls (resized on demand)
    scratch: RefCell<AttentionScratch>,
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

            scratch: RefCell::new(AttentionScratch::default()),
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

    /// Returns the trainable parameter tensors in update order.
    pub fn parameter_slices(&self) -> [&[f32]; 8] {
        [
            &self.w_q, &self.b_q, &self.w_k, &self.b_k, &self.w_v, &self.b_v, &self.w_o, &self.b_o,
        ]
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
        let m = total_tokens as i32;
        let n = self.d_model as i32;
        let k_dim = self.d_model as i32;

        // Q = input[m x d_model] * W_q[d_model x d_model] + b_q
        // Shapes (row-major):
        //   A=input:  (m=total_tokens) x (k=d_model), lda=k
        //   B=W_q:    (k=d_model) x (n=d_model),     ldb=n
        //   C=Q:      (m=total_tokens) x (n=d_model),ldc=n
        q.fill(0.0);
        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m,
                n,
                k_dim,
                1.0,
                input,
                k_dim,
                &self.w_q,
                n,
                0.0,
                q,
                n,
            );
        }
        for tok in 0..total_tokens {
            let base = tok * self.d_model;
            for d in 0..self.d_model {
                q[base + d] += self.b_q[d];
            }
        }

        // K = input[m x d_model] * W_k[d_model x d_model] + b_k
        // Shapes (row-major):
        //   A=input:  (m=total_tokens) x (k=d_model), lda=k
        //   B=W_k:    (k=d_model) x (n=d_model),     ldb=n
        //   C=K:      (m=total_tokens) x (n=d_model),ldc=n
        k.fill(0.0);
        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m,
                n,
                k_dim,
                1.0,
                input,
                k_dim,
                &self.w_k,
                n,
                0.0,
                k,
                n,
            );
        }
        for tok in 0..total_tokens {
            let base = tok * self.d_model;
            for d in 0..self.d_model {
                k[base + d] += self.b_k[d];
            }
        }

        // V = input[m x d_model] * W_v[d_model x d_model] + b_v
        // Shapes (row-major):
        //   A=input:  (m=total_tokens) x (k=d_model), lda=k
        //   B=W_v:    (k=d_model) x (n=d_model),     ldb=n
        //   C=V:      (m=total_tokens) x (n=d_model),ldc=n
        v.fill(0.0);
        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m,
                n,
                k_dim,
                1.0,
                input,
                k_dim,
                &self.w_v,
                n,
                0.0,
                v,
                n,
            );
        }
        for tok in 0..total_tokens {
            let base = tok * self.d_model;
            for d in 0..self.d_model {
                v[base + d] += self.b_v[d];
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
                let q_base = (batch * seq_len) * self.d_model + head * self.d_head;
                let k_base = (batch * seq_len) * self.d_model + head * self.d_head;

                // scores[seq_len x seq_len] = Q_h[seq_len x d_head] * K_h^T[d_head x seq_len]
                // Shapes (row-major, using strided views with lda/ldb=d_model):
                //   A=Q_h: (m=seq_len) x (k=d_head),  lda=d_model
                //   B=K_h: (n=seq_len) x (k=d_head),  but transposed -> (k=d_head) x (n=seq_len), ldb=d_model
                //   C=scores: (m=seq_len) x (n=seq_len), ldc=seq_len
                let bh = batch * self.num_heads + head;
                let scores_base = bh * seq_len * seq_len;
                let scores = &mut attn_weights[scores_base..scores_base + seq_len * seq_len];

                let q_ptr = &q[q_base..];
                let k_ptr = &k[k_base..];
                unsafe {
                    cblas::sgemm(
                        Layout::RowMajor,
                        Transpose::None,
                        Transpose::Ordinary,
                        seq_len as i32,
                        seq_len as i32,
                        self.d_head as i32,
                        inv_sqrt_d,
                        q_ptr,
                        self.d_model as i32,
                        k_ptr,
                        self.d_model as i32,
                        0.0,
                        scores,
                        seq_len as i32,
                    );
                }

                // Row-wise softmax
                for i in 0..seq_len {
                    let row_base = i * seq_len;
                    Self::softmax_inplace(&mut scores[row_base..row_base + seq_len]);
                }

                // context_h[seq_len x d_head] = alpha[seq_len x seq_len] * V_h[seq_len x d_head]
                // Shapes (row-major; V_h and output are strided views with ld*=d_model):
                //   A=alpha: (m=seq_len) x (k=seq_len), lda=seq_len
                //   B=V_h:   (k=seq_len) x (n=d_head),  ldb=d_model
                //   C=out_h: (m=seq_len) x (n=d_head),  ldc=d_model
                let v_ptr = &v[(batch * seq_len) * self.d_model + head * self.d_head..];
                let out_ptr =
                    &mut attn_out[(batch * seq_len) * self.d_model + head * self.d_head..];
                unsafe {
                    cblas::sgemm(
                        Layout::RowMajor,
                        Transpose::None,
                        Transpose::None,
                        seq_len as i32,
                        self.d_head as i32,
                        seq_len as i32,
                        1.0,
                        scores,
                        seq_len as i32,
                        v_ptr,
                        self.d_model as i32,
                        1.0,
                        out_ptr,
                        self.d_model as i32,
                    );
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
        let m = total_tokens;
        let n = self.d_model;
        let k = self.d_model;

        // output[m x n] = attn_out[m x k] * w_o[k x n]
        // Shapes (row-major):
        //   A=attn_out: (m=total_tokens) x (k=d_model), lda=k
        //   B=W_o:      (k=d_model) x (n=d_model),     ldb=n
        //   C=output:   (m=total_tokens) x (n=d_model),ldc=n
        output.fill(0.0);
        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                attn_out,
                k as i32,
                &self.w_o,
                n as i32,
                0.0,
                output,
                n as i32,
            );
        }

        // Bias: broadcast over tokens
        for tok in 0..total_tokens {
            let base = tok * self.d_model;
            for d in 0..self.d_model {
                output[base + d] += self.b_o[d];
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

        let attn_size = batch_size * self.num_heads * seq_len * seq_len;

        // Gradient + scratch buffers (reused across calls)
        let mut scratch = self.scratch.borrow_mut();
        scratch.ensure_sizes(total_tokens, self.d_model, self.d_head, attn_size, seq_len);
        scratch.zero_used();
        scratch.grad_context_contig.fill(0.0);
        scratch.v_contig.fill(0.0);
        scratch.grad_v_contig.fill(0.0);

        // We'll access scratch buffers directly via `scratch` to avoid borrow conflicts
        // with the attention-score workspace.

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
        // Shapes (RowMajor):
        //  - attn_out: [total_tokens, d_model]
        //  - W_o:     [d_model, d_model]
        //  - dY:      [total_tokens, d_model]
        //  - dW_o = attn_out^T * dY  => [d_model, d_model]
        //  - d_attn_out = dY * W_o^T => [total_tokens, d_model]

        // b_o gradient: sum over tokens
        for tok_idx in 0..total_tokens {
            let grad_out_base = tok_idx * self.d_model;
            for d_out in 0..self.d_model {
                grad_b_o[d_out] += grad_output[grad_out_base + d_out];
            }
        }

        unsafe {
            // grad_w_o += attn_out^T * grad_output
            // Shapes (row-major):
            //   A=attn_out:   (m=total_tokens) x (k=d_model), transposed -> (k=d_model) x (m=total_tokens), lda=d_model
            //   B=grad_output:(m=total_tokens) x (n=d_model), ldb=d_model
            //   C=grad_w_o:   (k=d_model) x (n=d_model), ldc=d_model
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::Ordinary,
                Transpose::None,
                self.d_model as i32,
                self.d_model as i32,
                total_tokens as i32,
                1.0,
                &attn_out,
                self.d_model as i32,
                grad_output,
                self.d_model as i32,
                1.0,
                &mut grad_w_o,
                self.d_model as i32,
            );

            // grad_attn_out = grad_output * W_o^T
            // Shapes (row-major):
            //   A=grad_output:(m=total_tokens) x (k=d_model), lda=d_model
            //   B=W_o:        (n=d_model) x (k=d_model), transposed -> (k=d_model) x (n=d_model), ldb=d_model
            //   C=grad_attn_out:(m=total_tokens) x (n=d_model), ldc=d_model
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                total_tokens as i32,
                self.d_model as i32,
                self.d_model as i32,
                1.0,
                grad_output,
                self.d_model as i32,
                &self.w_o,
                self.d_model as i32,
                0.0,
                &mut scratch.grad_attn_out,
                self.d_model as i32,
            );
        }

        // Step 2: Backprop through attention mechanism
        let inv_sqrt_d = 1.0f32 / (self.d_head as f32).sqrt();

        // We compute per-(batch, head) matrices using GEMM:
        //  - context = alpha * V
        //  - grad_alpha = grad_context * V^T
        //  - grad_V = alpha^T * grad_context
        //  - grad_Q = (grad_scores / sqrt(d_head)) * K
        //  - grad_K = (grad_scores / sqrt(d_head))^T * Q
        // where scores = Q*K^T / sqrt(d_head).
        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                let bh = batch * self.num_heads + head;

                let q_base = (batch * seq_len) * self.d_model + head * self.d_head;
                let k_base = (batch * seq_len) * self.d_model + head * self.d_head;
                let v_base = (batch * seq_len) * self.d_model + head * self.d_head;
                let grad_context_base = (batch * seq_len) * self.d_model + head * self.d_head;

                // Strides between tokens for per-head matrices inside the flattened [total_tokens, d_model]
                let row_stride = self.d_model;

                // Pointers to the start of each per-head matrix
                let q_ptr = &q[q_base..];
                let k_ptr = &k[k_base..];
                let _v_ptr = &v[v_base..];
                // We'll borrow the different scratch buffers in small scopes to satisfy Rust's
                // aliasing rules while still avoiding any per-loop allocations.

                {
                    // 1) grad_alpha = grad_context * V^T   => [seq_len, seq_len]
                    //    (RowMajor) A=[seq_len, d_head] lda=d_head
                    //               B=[seq_len, d_head] ldb=d_head, transposed
                    //               C=[seq_len, seq_len] ldc=seq_len
                    //
                    // Pack grad_context and V (strided in [total_tokens, d_model]) into
                    // contiguous [seq_len, d_head] matrices for GEMM.
                    // Fill grad_context_contig and v_contig
                    for i in 0..seq_len {
                        let src_row = grad_context_base + i * row_stride;
                        let dst_row = i * self.d_head;
                        // Avoid overlapping borrows of `scratch` by copying elementwise.
                        for j in 0..self.d_head {
                            scratch.grad_context_contig[dst_row + j] =
                                scratch.grad_attn_out[src_row + j];
                        }
                    }
                    for i in 0..seq_len {
                        let src_row = v_base + i * row_stride;
                        let dst_row = i * self.d_head;
                        scratch.v_contig[dst_row..dst_row + self.d_head]
                            .copy_from_slice(&v[src_row..src_row + self.d_head]);
                    }

                    // grad_context_contig: [seq_len, d_head]
                    // v_contig:            [seq_len, d_head]

                    let grad_alpha_range = bh * seq_len * seq_len..(bh + 1) * seq_len * seq_len;

                    // Use raw pointers to avoid borrow conflicts while still passing slices to cblas.
                    unsafe {
                        let grad_context_ptr = scratch.grad_context_contig.as_ptr();
                        let v_ptr = scratch.v_contig.as_ptr();
                        let grad_alpha_ptr =
                            scratch.grad_attn_weights[grad_alpha_range].as_mut_ptr();

                        let grad_context =
                            std::slice::from_raw_parts(grad_context_ptr, seq_len * self.d_head);
                        let v_contig = std::slice::from_raw_parts(v_ptr, seq_len * self.d_head);
                        let grad_alpha =
                            std::slice::from_raw_parts_mut(grad_alpha_ptr, seq_len * seq_len);
                        grad_alpha.fill(0.0);

                        cblas::sgemm(
                            Layout::RowMajor,
                            Transpose::None,
                            Transpose::Ordinary,
                            seq_len as i32,
                            seq_len as i32,
                            self.d_head as i32,
                            1.0,
                            grad_context,
                            self.d_head as i32,
                            v_contig,
                            self.d_head as i32,
                            0.0,
                            grad_alpha,
                            seq_len as i32,
                        );
                    }
                }
                {
                    // 2) grad_V += alpha^T * grad_context  => [seq_len, d_head]
                    //    A=alpha [seq_len, seq_len] lda=seq_len, transposed
                    //    B=grad_context [seq_len, d_head] ldb=d_head
                    //    C=grad_V [seq_len, d_head] ldc=d_head
                    scratch.grad_v_contig.fill(0.0);

                    let alpha_mat =
                        &attn_weights[bh * seq_len * seq_len..(bh + 1) * seq_len * seq_len];

                    unsafe {
                        let grad_context_ptr = scratch.grad_context_contig.as_ptr();
                        let grad_v_ptr = scratch.grad_v_contig.as_mut_ptr();

                        let grad_context =
                            std::slice::from_raw_parts(grad_context_ptr, seq_len * self.d_head);
                        let grad_v =
                            std::slice::from_raw_parts_mut(grad_v_ptr, seq_len * self.d_head);

                        cblas::sgemm(
                            Layout::RowMajor,
                            Transpose::Ordinary,
                            Transpose::None,
                            seq_len as i32,
                            self.d_head as i32,
                            seq_len as i32,
                            1.0,
                            alpha_mat,
                            seq_len as i32,
                            grad_context,
                            self.d_head as i32,
                            0.0,
                            grad_v,
                            self.d_head as i32,
                        );
                    }

                    for i in 0..seq_len {
                        let dst_row = v_base + i * row_stride;
                        let src_row = i * self.d_head;
                        // Avoid overlapping borrows of `scratch` by copying elementwise.
                        for j in 0..self.d_head {
                            scratch.grad_v[dst_row + j] = scratch.grad_v_contig[src_row + j];
                        }
                    }
                }

                {
                    // 2) Backprop softmax per row i: grad_scores_mat[i,*] becomes dL/dscores.
                    let grad_scores_mat = &mut scratch.grad_attn_weights
                        [bh * seq_len * seq_len..(bh + 1) * seq_len * seq_len];
                    for i in 0..seq_len {
                        let attn_row_base = (bh * seq_len + i) * seq_len;
                        let grad_row = &mut grad_scores_mat[i * seq_len..(i + 1) * seq_len];

                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            sum += grad_row[j] * attn_weights[attn_row_base + j];
                        }
                        for j in 0..seq_len {
                            let alpha = attn_weights[attn_row_base + j];
                            grad_row[j] = alpha * (grad_row[j] - sum);
                        }
                    }
                }

                // (grad_V computed above using packed GEMM; no additional work needed here)

                {
                    // 4) grad_scores_scaled = grad_scores_mat / sqrt(d_head), then
                    //    grad_Q += grad_scores_scaled * K  => [seq_len, d_head]
                    //    grad_K += grad_scores_scaled^T * Q => [seq_len, d_head]
                    {
                        let grad_scores_range =
                            bh * seq_len * seq_len..(bh + 1) * seq_len * seq_len;

                        unsafe {
                            let grad_scores_ptr =
                                scratch.grad_attn_weights[grad_scores_range.clone()].as_mut_ptr();
                            let grad_scores_mat =
                                std::slice::from_raw_parts_mut(grad_scores_ptr, seq_len * seq_len);

                            for val in grad_scores_mat.iter_mut() {
                                *val *= inv_sqrt_d;
                            }

                            let dst_ptr = scratch.grad_scores_mat.as_mut_ptr();
                            std::ptr::copy_nonoverlapping(
                                grad_scores_mat.as_ptr(),
                                dst_ptr,
                                seq_len * seq_len,
                            );
                        }

                        unsafe {
                            let grad_scores_ptr = scratch.grad_scores_mat.as_ptr();
                            let grad_scores =
                                std::slice::from_raw_parts(grad_scores_ptr, seq_len * seq_len);

                            // grad_Q
                            let grad_q_ptr = scratch.grad_q[q_base..].as_mut_ptr();
                            let grad_q =
                                std::slice::from_raw_parts_mut(grad_q_ptr, seq_len * row_stride);
                            cblas::sgemm(
                                Layout::RowMajor,
                                Transpose::None,
                                Transpose::None,
                                seq_len as i32,
                                self.d_head as i32,
                                seq_len as i32,
                                1.0,
                                grad_scores,
                                seq_len as i32,
                                k_ptr,
                                row_stride as i32,
                                1.0,
                                grad_q,
                                row_stride as i32,
                            );

                            // grad_K
                            let grad_k_ptr = scratch.grad_k[k_base..].as_mut_ptr();
                            let grad_k =
                                std::slice::from_raw_parts_mut(grad_k_ptr, seq_len * row_stride);
                            cblas::sgemm(
                                Layout::RowMajor,
                                Transpose::Ordinary,
                                Transpose::None,
                                seq_len as i32,
                                self.d_head as i32,
                                seq_len as i32,
                                1.0,
                                grad_scores,
                                seq_len as i32,
                                q_ptr,
                                row_stride as i32,
                                1.0,
                                grad_k,
                                row_stride as i32,
                            );
                        }
                    }
                }
            }
        }

        // Step 3: Backprop through Q/K/V projections
        // Shapes (row-major):
        //  - X:  [total_tokens, d_model]
        //  - dQ/dK/dV: [total_tokens, d_model]
        //  - dW? = X^T * d?  => [d_model, d_model]
        //  - dX  += d? * W?^T => [total_tokens, d_model]
        grad_input.fill(0.0);

        // Bias gradients: sum over tokens
        for tok_idx in 0..total_tokens {
            let base = tok_idx * self.d_model;
            for j in 0..self.d_model {
                grad_b_q[j] += scratch.grad_q[base + j];
                grad_b_k[j] += scratch.grad_k[base + j];
                grad_b_v[j] += scratch.grad_v[base + j];
            }
        }

        unsafe {
            // dWq += X^T * dQ
            // Shapes (row-major):
            //   A=X:  (m=total_tokens) x (k=d_model), transposed -> (k=d_model) x (m=total_tokens)
            //   B=dQ: (m=total_tokens) x (n=d_model)
            //   C=dWq:(k=d_model) x (n=d_model)
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::Ordinary,
                Transpose::None,
                self.d_model as i32,
                self.d_model as i32,
                total_tokens as i32,
                1.0,
                input,
                self.d_model as i32,
                &scratch.grad_q,
                self.d_model as i32,
                1.0,
                &mut grad_w_q,
                self.d_model as i32,
            );
            // dWk += X^T * dK
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::Ordinary,
                Transpose::None,
                self.d_model as i32,
                self.d_model as i32,
                total_tokens as i32,
                1.0,
                input,
                self.d_model as i32,
                &scratch.grad_k,
                self.d_model as i32,
                1.0,
                &mut grad_w_k,
                self.d_model as i32,
            );
            // dWv += X^T * dV
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::Ordinary,
                Transpose::None,
                self.d_model as i32,
                self.d_model as i32,
                total_tokens as i32,
                1.0,
                input,
                self.d_model as i32,
                &scratch.grad_v,
                self.d_model as i32,
                1.0,
                &mut grad_w_v,
                self.d_model as i32,
            );

            // dX = dQ*Wq^T + dK*Wk^T + dV*Wv^T
            // Shapes (row-major):
            //   A=dQ:  (m=total_tokens) x (k=d_model)
            //   B=Wq:  (n=d_model) x (k=d_model), transposed -> (k=d_model) x (n=d_model)
            //   C=dX:  (m=total_tokens) x (n=d_model)
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                total_tokens as i32,
                self.d_model as i32,
                self.d_model as i32,
                1.0,
                &scratch.grad_q,
                self.d_model as i32,
                &self.w_q,
                self.d_model as i32,
                0.0,
                grad_input,
                self.d_model as i32,
            );
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                total_tokens as i32,
                self.d_model as i32,
                self.d_model as i32,
                1.0,
                &scratch.grad_k,
                self.d_model as i32,
                &self.w_k,
                self.d_model as i32,
                1.0,
                grad_input,
                self.d_model as i32,
            );
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                total_tokens as i32,
                self.d_model as i32,
                self.d_model as i32,
                1.0,
                &scratch.grad_v,
                self.d_model as i32,
                &self.w_v,
                self.d_model as i32,
                1.0,
                grad_input,
                self.d_model as i32,
            );
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
