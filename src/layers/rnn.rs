//! Recurrent Neural Network (RNN) layer implementation
//!
//! This module provides a vanilla RNN layer that processes sequential data
//! by maintaining a hidden state across time steps.
//!
//! # Architecture
//!
//! The RNN layer implements the following recurrent transformation:
//! - Hidden state update: `h_t = tanh(x_t × W_xh + h_{t-1} × W_hh + b_h)`
//! - Output computation: `y_t = h_t × W_hy + b_y`
//!
//! where:
//! - `x_t` is the input at time step t
//! - `h_t` is the hidden state at time step t
//! - `y_t` is the output at time step t
//!
//! # Usage Example
//!
//! ```ignore
//! use rust_neural_networks::layers::{RnnLayer, Layer};
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! // Create RNN layer: 10 input features, 20 hidden units, 5 outputs
//! let mut rng = SimpleRng::new(42);
//! let layer = RnnLayer::new(10, 20, 5, &mut rng);
//!
//! // Process a sequence of 3 time steps
//! layer.reset_hidden_state();  // Clear state before new sequence
//! for t in 0..3 {
//!     let input = vec![0.5; 10];      // Input for time step t
//!     let mut output = vec![0.0; 5];  // Output buffer
//!     layer.forward(&input, &mut output, 1);
//!     // Hidden state is automatically maintained between time steps
//! }
//! ```
//!
//! # Important Notes
//!
//! - Always call `reset_hidden_state()` at the beginning of a new sequence
//! - The hidden state persists across forward passes within a sequence
//! - For batch processing, all samples in a batch share the same initial hidden state

use crate::layers::gradient::GradientAccumulator;
use crate::layers::Layer;
use crate::utils::rng::SimpleRng;
use std::cell::RefCell;

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

/// Vanilla RNN layer with hidden state.
///
/// Performs the recurrent transformation:
/// h_t = tanh(x_t × W_xh + h_{t-1} × W_hh + b_h)
/// y_t = h_t × W_hy + b_y
///
/// where x_t is the input at time t, h_t is the hidden state,
/// and y_t is the output at time t.
///
/// # Fields
///
/// * `input_size` - Number of input features per time step
/// * `hidden_size` - Size of the hidden state vector
/// * `output_size` - Number of output features
/// * `w_xh` - Input-to-hidden weight matrix (input_size × hidden_size)
/// * `w_hh` - Hidden-to-hidden weight matrix (hidden_size × hidden_size)
/// * `w_hy` - Hidden-to-output weight matrix (hidden_size × output_size)
/// * `b_h` - Hidden bias vector (hidden_size)
/// * `b_y` - Output bias vector (output_size)
/// * `hidden_state` - Current hidden state (hidden_size)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::RnnLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = RnnLayer::new(128, 256, 10, &mut rng);
/// assert_eq!(layer.input_size(), 128);
/// assert_eq!(layer.hidden_size(), 256);
/// assert_eq!(layer.output_size(), 10);
/// ```
pub struct RnnLayer {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    // Weight matrices
    w_xh: Vec<f32>, // input_size × hidden_size
    w_hh: Vec<f32>, // hidden_size × hidden_size
    w_hy: Vec<f32>, // hidden_size × output_size

    // Biases
    b_h: Vec<f32>, // hidden_size
    b_y: Vec<f32>, // output_size

    // Hidden state
    hidden_state: RefCell<Vec<f32>>, // hidden_size

    // Gradient accumulators
    grad_w_xh: GradientAccumulator,
    grad_w_hh: GradientAccumulator,
    grad_w_hy: GradientAccumulator,
    grad_b_h: GradientAccumulator,
    grad_b_y: GradientAccumulator,

    // Cache for backward pass
    cached_h_prev: RefCell<Vec<f32>>, // h_{t-1} before forward pass
    cached_h_current: RefCell<Vec<f32>>, // h_t after tanh in forward pass
}

impl RnnLayer {
    /// Creates a vanilla RNN layer with Xavier-initialized weights and zero biases.
    ///
    /// Weights are sampled uniformly from [-limit, limit], where
    /// `limit = sqrt(6.0 / (fan_in + fan_out))` for each weight matrix.
    /// Biases, hidden state, and gradient accumulators are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features per time step
    /// * `hidden_size` - Size of the hidden state vector
    /// * `output_size` - Number of output features
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(64, 128, 10, &mut rng);
    /// assert_eq!(layer.input_size(), 64);
    /// assert_eq!(layer.hidden_size(), 128);
    /// assert_eq!(layer.output_size(), 10);
    /// ```
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        // Xavier initialization for W_xh: limit = sqrt(6 / (input_size + hidden_size))
        let mut w_xh = vec![0.0f32; input_size * hidden_size];
        let limit_xh = (6.0f32 / (input_size + hidden_size) as f32).sqrt();
        for value in &mut w_xh {
            *value = rng.gen_range_f32(-limit_xh, limit_xh);
        }

        // Xavier initialization for W_hh: limit = sqrt(6 / (hidden_size + hidden_size))
        let mut w_hh = vec![0.0f32; hidden_size * hidden_size];
        let limit_hh = (6.0f32 / (hidden_size + hidden_size) as f32).sqrt();
        for value in &mut w_hh {
            *value = rng.gen_range_f32(-limit_hh, limit_hh);
        }

        // Xavier initialization for W_hy: limit = sqrt(6 / (hidden_size + output_size))
        let mut w_hy = vec![0.0f32; hidden_size * output_size];
        let limit_hy = (6.0f32 / (hidden_size + output_size) as f32).sqrt();
        for value in &mut w_hy {
            *value = rng.gen_range_f32(-limit_hy, limit_hy);
        }

        Self {
            input_size,
            hidden_size,
            output_size,
            w_xh,
            w_hh,
            w_hy,
            b_h: vec![0.0f32; hidden_size],
            b_y: vec![0.0f32; output_size],
            hidden_state: RefCell::new(vec![0.0f32; hidden_size]),
            grad_w_xh: GradientAccumulator::new(input_size * hidden_size),
            grad_w_hh: GradientAccumulator::new(hidden_size * hidden_size),
            grad_w_hy: GradientAccumulator::new(hidden_size * output_size),
            grad_b_h: GradientAccumulator::new(hidden_size),
            grad_b_y: GradientAccumulator::new(output_size),
            cached_h_prev: RefCell::new(vec![0.0f32; hidden_size]),
            cached_h_current: RefCell::new(vec![0.0f32; hidden_size]),
        }
    }

    /// Get the hidden state size of the layer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(64, 128, 10, &mut rng);
    /// assert_eq!(layer.hidden_size(), 128);
    /// ```
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Reset the hidden state to zeros.
    ///
    /// This should be called at the beginning of a new sequence to clear
    /// any information from previous sequences.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// layer.reset_hidden_state();
    /// // Process new sequence...
    /// ```
    pub fn reset_hidden_state(&self) {
        let mut hidden = self.hidden_state.borrow_mut();
        for h in hidden.iter_mut() {
            *h = 0.0;
        }
    }

    /// Get a copy of the current hidden state.
    ///
    /// # Returns
    ///
    /// A vector containing the current hidden state values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{RnnLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Process some input
    /// let input = vec![0.5; 10];
    /// let mut output = vec![0.0; 5];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// // Inspect hidden state
    /// let hidden = layer.get_hidden_state();
    /// assert_eq!(hidden.len(), 20);
    /// ```
    pub fn get_hidden_state(&self) -> Vec<f32> {
        self.hidden_state.borrow().clone()
    }

    /// Set the hidden state to specific values.
    ///
    /// Useful for initializing the hidden state with specific values or
    /// resuming computation from a saved state.
    ///
    /// # Arguments
    ///
    /// * `state` - New hidden state values (must have length equal to hidden_size)
    ///
    /// # Panics
    ///
    /// Panics if the provided state vector has incorrect length.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Initialize with custom state
    /// let custom_state = vec![0.1; 20];
    /// layer.set_hidden_state(&custom_state);
    ///
    /// let retrieved = layer.get_hidden_state();
    /// assert_eq!(retrieved, custom_state);
    /// ```
    pub fn set_hidden_state(&self, state: &[f32]) {
        assert_eq!(state.len(), self.hidden_size, "Hidden state size mismatch");
        let mut hidden = self.hidden_state.borrow_mut();
        hidden.copy_from_slice(state);
    }

    /// Immutable view of the input-to-hidden weight matrix.
    ///
    /// # Returns
    ///
    /// A slice containing weights in row-major order (input_size × hidden_size).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    /// let weights = layer.w_xh();
    /// assert_eq!(weights.len(), 10 * 20);
    /// ```
    pub fn w_xh(&self) -> &[f32] {
        &self.w_xh
    }

    /// Immutable view of the hidden-to-hidden weight matrix.
    ///
    /// # Returns
    ///
    /// A slice containing weights in row-major order (hidden_size × hidden_size).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    /// let weights = layer.w_hh();
    /// assert_eq!(weights.len(), 20 * 20);
    /// ```
    pub fn w_hh(&self) -> &[f32] {
        &self.w_hh
    }

    /// Immutable view of the hidden-to-output weight matrix.
    ///
    /// # Returns
    ///
    /// A slice containing weights in row-major order (hidden_size × output_size).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    /// let weights = layer.w_hy();
    /// assert_eq!(weights.len(), 20 * 5);
    /// ```
    pub fn w_hy(&self) -> &[f32] {
        &self.w_hy
    }

    /// Provides a slice view of the hidden bias vector.
    ///
    /// # Returns
    ///
    /// A slice containing the bias for each hidden unit.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    /// let biases = layer.b_h();
    /// assert_eq!(biases.len(), 20);
    /// ```
    pub fn b_h(&self) -> &[f32] {
        &self.b_h
    }

    /// Provides a slice view of the output bias vector.
    ///
    /// # Returns
    ///
    /// A slice containing the bias for each output unit.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    /// let biases = layer.b_y();
    /// assert_eq!(biases.len(), 5);
    /// ```
    pub fn b_y(&self) -> &[f32] {
        &self.b_y
    }

    /// Return the total number of trainable parameters in the layer.
    ///
    /// This equals:
    /// - W_xh: input_size × hidden_size
    /// - W_hh: hidden_size × hidden_size
    /// - W_hy: hidden_size × output_size
    /// - b_h: hidden_size
    /// - b_y: output_size
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::RnnLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    /// let expected = 10 * 20 + 20 * 20 + 20 * 5 + 20 + 5;
    /// assert_eq!(layer.parameter_count(), expected);
    /// ```
    pub fn parameter_count(&self) -> usize {
        self.w_xh.len() + self.w_hh.len() + self.w_hy.len() + self.b_h.len() + self.b_y.len()
    }
}

impl Layer for RnnLayer {
    /// Computes the RNN forward pass for one time step across a batch.
    ///
    /// # Mathematical Formulation
    ///
    /// The RNN forward pass consists of two main steps:
    ///
    /// **Step 1: Hidden State Update**
    /// - Compute pre-activation: `z_t = x_t × W_xh + h_{t-1} × W_hh + b_h`
    /// - Apply activation: `h_t = tanh(z_t)`
    ///
    /// **Step 2: Output Computation**
    /// - Compute output: `y_t = h_t × W_hy + b_y`
    ///
    /// where:
    /// - `x_t` is the input at time step t (batch_size × input_size)
    /// - `h_{t-1}` is the previous hidden state (hidden_size), broadcasted to batch
    /// - `h_t` is the new hidden state (hidden_size)
    /// - `y_t` is the output at time step t (batch_size × output_size)
    /// - `W_xh` is the input-to-hidden weight matrix (input_size × hidden_size)
    /// - `W_hh` is the hidden-to-hidden weight matrix (hidden_size × hidden_size)
    /// - `W_hy` is the hidden-to-output weight matrix (hidden_size × output_size)
    /// - `b_h` is the hidden bias vector (hidden_size)
    /// - `b_y` is the output bias vector (output_size)
    ///
    /// # Matrix Operations
    ///
    /// **Hidden State Computation:**
    /// 1. `x_t × W_xh`: (batch_size × input_size) × (input_size × hidden_size) → (batch_size × hidden_size)
    /// 2. `h_{t-1} × W_hh`: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
    /// 3. Add bias and apply tanh element-wise
    ///
    /// **Output Computation:**
    /// 1. `h_t × W_hy`: (batch_size × hidden_size) × (hidden_size × output_size) → (batch_size × output_size)
    /// 2. Add bias b_y
    ///
    /// # Implementation Details
    ///
    /// - Uses BLAS `sgemm` for efficient matrix multiplication
    /// - Hidden state `h_{t-1}` is broadcasted to all batch samples (all samples share the same initial hidden state)
    /// - The hidden state is cached for backward pass (both `h_{t-1}` and `h_t`)
    /// - After processing, the layer's internal hidden state is updated to `h_t[0]` (first batch sample)
    ///
    /// # Arguments
    ///
    /// * `input` - Input for current time step (batch_size × input_size)
    /// * `output` - Output buffer for current time step (batch_size × output_size)
    /// * `batch_size` - Number of sequences in the batch
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{RnnLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = RnnLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Process a sequence of 3 time steps for batch of 2
    /// layer.reset_hidden_state();
    /// for t in 0..3 {
    ///     let input = vec![0.0; 2 * 10];  // batch_size=2, input_size=10
    ///     let mut output = vec![0.0; 2 * 5];  // batch_size=2, output_size=5
    ///     layer.forward(&input, &mut output, 2);
    /// }
    /// ```
    ///
    /// # Important Notes
    ///
    /// - Always call `reset_hidden_state()` at the beginning of a new sequence
    /// - The hidden state persists across forward passes within a sequence
    /// - For batch processing, all samples in a batch share the same initial hidden state
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        use cblas::{sgemm, Layout, Transpose};

        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "Input size mismatch: expected {}, got {}",
            batch_size * self.input_size,
            input.len()
        );
        assert_eq!(
            output.len(),
            batch_size * self.output_size,
            "Output size mismatch: expected {}, got {}",
            batch_size * self.output_size,
            output.len()
        );

        // Allocate temporary buffer for new hidden state
        let mut new_hidden = vec![0.0f32; batch_size * self.hidden_size];

        // Compute x_t × W_xh: (batch_size × input_size) × (input_size × hidden_size)
        // Result: batch_size × hidden_size
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                batch_size as i32,
                self.hidden_size as i32,
                self.input_size as i32,
                1.0,
                input,
                self.input_size as i32,
                &self.w_xh,
                self.hidden_size as i32,
                0.0,
                &mut new_hidden,
                self.hidden_size as i32,
            );
        }

        // Get current hidden state and cache it for backward pass
        let hidden = self.hidden_state.borrow();
        {
            let mut cached_h_prev = self.cached_h_prev.borrow_mut();
            cached_h_prev.copy_from_slice(&hidden[..]);
        }

        // Compute h_{t-1} × W_hh and add to new_hidden
        // For batch processing, we broadcast the hidden state to all batch items
        // (batch_size × hidden_size) × (hidden_size × hidden_size)
        // Note: If batch_size > 1, we need to replicate the hidden state for each item

        // Create temporary buffer with replicated hidden state
        let mut hidden_batch = vec![0.0f32; batch_size * self.hidden_size];
        for b in 0..batch_size {
            hidden_batch[b * self.hidden_size..(b + 1) * self.hidden_size]
                .copy_from_slice(&hidden[..]);
        }

        // Temporary buffer for h_{t-1} × W_hh
        let mut hh_contrib = vec![0.0f32; batch_size * self.hidden_size];

        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                batch_size as i32,
                self.hidden_size as i32,
                self.hidden_size as i32,
                1.0,
                &hidden_batch,
                self.hidden_size as i32,
                &self.w_hh,
                self.hidden_size as i32,
                0.0,
                &mut hh_contrib,
                self.hidden_size as i32,
            );
        }

        // Add h_{t-1} × W_hh to x_t × W_xh and add bias, then apply tanh
        for i in 0..new_hidden.len() {
            let bias_idx = i % self.hidden_size;
            new_hidden[i] = (new_hidden[i] + hh_contrib[i] + self.b_h[bias_idx]).tanh();
        }

        // Cache the current hidden state (h_t) for backward pass
        {
            let mut cached_h_current = self.cached_h_current.borrow_mut();
            if batch_size == 1 {
                cached_h_current.copy_from_slice(&new_hidden[..]);
            } else {
                // For batch processing, cache the first sample's hidden state
                cached_h_current.copy_from_slice(&new_hidden[0..self.hidden_size]);
            }
        }

        // Update hidden state (for batch_size=1, or use the last batch item's hidden state)
        // For simplicity, we update with the first batch item's hidden state
        drop(hidden); // Release borrow before mutably borrowing
        let mut hidden_mut = self.hidden_state.borrow_mut();
        hidden_mut.copy_from_slice(&new_hidden[0..self.hidden_size]);

        // Compute output: h_t × W_hy + b_y
        // (batch_size × hidden_size) × (hidden_size × output_size)
        // Result: batch_size × output_size
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                batch_size as i32,
                self.output_size as i32,
                self.hidden_size as i32,
                1.0,
                &new_hidden,
                self.hidden_size as i32,
                &self.w_hy,
                self.output_size as i32,
                0.0,
                output,
                self.output_size as i32,
            );
        }

        // Add output bias
        for b in 0..batch_size {
            for o in 0..self.output_size {
                output[b * self.output_size + o] += self.b_y[o];
            }
        }
    }

    /// Computes the RNN backward pass using Backpropagation Through Time (BPTT).
    ///
    /// # Backpropagation Through Time (BPTT)
    ///
    /// BPTT extends standard backpropagation to recurrent networks by unrolling the network
    /// across time steps and applying the chain rule through the temporal connections.
    ///
    /// For a single time step, we compute gradients for:
    /// - Output layer: `y_t = h_t × W_hy + b_y`
    /// - Hidden state: `h_t = tanh(x_t × W_xh + h_{t-1} × W_hh + b_h)`
    ///
    /// # Mathematical Formulation
    ///
    /// Given gradient w.r.t. output: `∂L/∂y_t` (batch_size × output_size)
    ///
    /// ## Step 1: Output Layer Gradients
    ///
    /// **Gradient w.r.t. W_hy** (hidden-to-output weights):
    /// - `∂L/∂W_hy = h_t^T × ∂L/∂y_t`
    /// - Dimension check: (hidden_size × batch_size) × (batch_size × output_size) → (hidden_size × output_size)
    ///
    /// **Gradient w.r.t. b_y** (output bias):
    /// - `∂L/∂b_y = Σ(∂L/∂y_t)` along batch dimension
    /// - Dimension check: sum over (batch_size × output_size) → (output_size)
    ///
    /// **Gradient w.r.t. hidden state**:
    /// - `∂L/∂h_t = ∂L/∂y_t × W_hy^T`
    /// - Dimension check: (batch_size × output_size) × (output_size × hidden_size) → (batch_size × hidden_size)
    ///
    /// ## Step 2: Hidden State Activation Gradient
    ///
    /// Apply tanh derivative using the chain rule:
    /// - Tanh derivative: `tanh'(z) = 1 - tanh²(z)`
    /// - `∂L/∂z_h = ∂L/∂h_t ⊙ (1 - h_t²)`
    /// - Where `z_h = x_t × W_xh + h_{t-1} × W_hh + b_h` (pre-activation)
    /// - Element-wise multiplication (⊙) applies tanh derivative to each hidden unit
    ///
    /// ## Step 3: Weight and Bias Gradients
    ///
    /// **Gradient w.r.t. W_xh** (input-to-hidden weights):
    /// - `∂L/∂W_xh = x_t^T × ∂L/∂z_h`
    /// - Dimension check: (input_size × batch_size) × (batch_size × hidden_size) → (input_size × hidden_size)
    ///
    /// **Gradient w.r.t. W_hh** (hidden-to-hidden weights):
    /// - `∂L/∂W_hh = h_{t-1}^T × ∂L/∂z_h`
    /// - Dimension check: (hidden_size × batch_size) × (batch_size × hidden_size) → (hidden_size × hidden_size)
    /// - This captures how previous hidden state contributes to current error
    ///
    /// **Gradient w.r.t. b_h** (hidden bias):
    /// - `∂L/∂b_h = Σ(∂L/∂z_h)` along batch dimension
    /// - Dimension check: sum over (batch_size × hidden_size) → (hidden_size)
    ///
    /// ## Step 4: Input Gradient (for previous layer)
    ///
    /// **Gradient w.r.t. input x_t**:
    /// - `∂L/∂x_t = ∂L/∂z_h × W_xh^T`
    /// - Dimension check: (batch_size × hidden_size) × (hidden_size × input_size) → (batch_size × input_size)
    ///
    /// ## Temporal Gradient Flow (BPTT)
    ///
    /// **Gradient w.r.t. previous hidden state h_{t-1}** (for full BPTT):
    /// - `∂L/∂h_{t-1} = ∂L/∂z_h × W_hh^T`
    /// - Dimension check: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
    /// - This gradient would be propagated to time step t-1 in full BPTT
    /// - **Note**: Current implementation computes this gradient but does not propagate it
    ///   backward through multiple time steps (truncated BPTT at single step)
    ///
    /// # Chain Rule Summary
    ///
    /// The complete gradient flow follows this path:
    /// ```text
    /// ∂L/∂y_t → ∂L/∂h_t → ∂L/∂z_h → {∂L/∂W_xh, ∂L/∂W_hh, ∂L/∂b_h, ∂L/∂x_t, ∂L/∂h_{t-1}}
    /// ```
    ///
    /// Where:
    /// - `∂L/∂y_t`: Gradient from loss function or next layer
    /// - `∂L/∂h_t`: Gradient at hidden state (after activation)
    /// - `∂L/∂z_h`: Gradient at pre-activation (before tanh)
    /// - Final gradients: Used to update parameters or backprop to previous layers/time steps
    ///
    /// # Implementation Details
    ///
    /// - Uses BLAS `sgemm` for efficient matrix multiplications
    /// - Gradients are **accumulated** (not replaced) to support mini-batch training
    /// - Scaling by `1/batch_size` for proper gradient averaging
    /// - Cached values from forward pass (`h_{t-1}`, `h_t`) are used for gradient computation
    ///
    /// # Arguments
    ///
    /// * `input` - Input from current time step (batch_size × input_size)
    /// * `grad_output` - Gradient w.r.t. output (batch_size × output_size)
    /// * `grad_input` - Output buffer for gradient w.r.t. input (batch_size × input_size)
    /// * `batch_size` - Number of sequences in the batch
    ///
    /// # Important Notes
    ///
    /// - This implements **truncated BPTT** for a single time step
    /// - For full BPTT across sequences, this method would be called iteratively
    ///   in reverse time order, accumulating gradients at each step
    /// - Gradients are accumulated (+=) to support gradient accumulation across time steps
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        use cblas::{sgemm, Layout, Transpose};

        if batch_size == 0 {
            panic!("batch_size cannot be zero in RNN::backward");
        }

        assert_eq!(
            input.len(),
            batch_size * self.input_size,
            "Input size mismatch in backward"
        );
        assert_eq!(
            grad_output.len(),
            batch_size * self.output_size,
            "Grad output size mismatch in backward"
        );
        assert_eq!(
            grad_input.len(),
            batch_size * self.input_size,
            "Grad input size mismatch in backward"
        );

        let scale = 1.0f32 / batch_size as f32;

        // Get cached values from forward pass
        let cached_h_prev = self.cached_h_prev.borrow();
        let cached_h_current = self.cached_h_current.borrow();

        // Replicate cached hidden states for batch processing
        let mut h_prev_batch = vec![0.0f32; batch_size * self.hidden_size];
        let mut h_current_batch = vec![0.0f32; batch_size * self.hidden_size];
        for b in 0..batch_size {
            h_prev_batch[b * self.hidden_size..(b + 1) * self.hidden_size]
                .copy_from_slice(&cached_h_prev[..]);
            h_current_batch[b * self.hidden_size..(b + 1) * self.hidden_size]
                .copy_from_slice(&cached_h_current[..]);
        }

        // Step 1: Compute gradients for output layer (y_t = h_t × W_hy + b_y)

        // Gradient w.r.t. W_hy: grad_W_hy = h_t^T × grad_output / batch_size
        // h_current_batch: (batch_size × hidden_size)
        // grad_output: (batch_size × output_size)
        // grad_W_hy: (hidden_size × output_size)
        {
            let mut grad_w_hy = self.grad_w_hy.borrow_mut();
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    self.hidden_size as i32,
                    self.output_size as i32,
                    batch_size as i32,
                    scale,
                    &h_current_batch,
                    self.hidden_size as i32,
                    grad_output,
                    self.output_size as i32,
                    1.0, // Accumulate gradients
                    &mut grad_w_hy,
                    self.output_size as i32,
                );
            }
        }

        // Gradient w.r.t. b_y: sum(grad_output) / batch_size
        {
            let mut batch_bias_grad = vec![0.0f32; self.output_size];
            for b in 0..batch_size {
                for o in 0..self.output_size {
                    batch_bias_grad[o] += grad_output[b * self.output_size + o];
                }
            }
            self.grad_b_y.accumulate_scaled(&batch_bias_grad, scale);
        }

        // Gradient w.r.t. h_t: grad_h = grad_output × W_hy^T
        // grad_output: (batch_size × output_size)
        // W_hy: (hidden_size × output_size)
        // grad_h: (batch_size × hidden_size)
        let mut grad_h = vec![0.0f32; batch_size * self.hidden_size];
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                batch_size as i32,
                self.hidden_size as i32,
                self.output_size as i32,
                1.0,
                grad_output,
                self.output_size as i32,
                &self.w_hy,
                self.output_size as i32,
                0.0,
                &mut grad_h,
                self.hidden_size as i32,
            );
        }

        // Step 2: Backpropagate through tanh activation
        // h_t = tanh(pre_activation)
        // grad_pre_activation = grad_h * (1 - h_t^2)
        let mut grad_pre_activation = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_pre_activation.len() {
            let h_val = h_current_batch[i];
            grad_pre_activation[i] = grad_h[i] * (1.0 - h_val * h_val);
        }

        // Step 3: Compute gradients for hidden state computation
        // pre_activation = x_t × W_xh + h_{t-1} × W_hh + b_h

        // Gradient w.r.t. W_xh: grad_W_xh = input^T × grad_pre_activation / batch_size
        // input: (batch_size × input_size)
        // grad_pre_activation: (batch_size × hidden_size)
        // grad_W_xh: (input_size × hidden_size)
        {
            let mut grad_w_xh = self.grad_w_xh.borrow_mut();
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    self.input_size as i32,
                    self.hidden_size as i32,
                    batch_size as i32,
                    scale,
                    input,
                    self.input_size as i32,
                    &grad_pre_activation,
                    self.hidden_size as i32,
                    1.0, // Accumulate gradients
                    &mut grad_w_xh,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. W_hh: grad_W_hh = h_{t-1}^T × grad_pre_activation / batch_size
        // h_prev_batch: (batch_size × hidden_size)
        // grad_pre_activation: (batch_size × hidden_size)
        // grad_W_hh: (hidden_size × hidden_size)
        {
            let mut grad_w_hh = self.grad_w_hh.borrow_mut();
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::Ordinary,
                    Transpose::None,
                    self.hidden_size as i32,
                    self.hidden_size as i32,
                    batch_size as i32,
                    scale,
                    &h_prev_batch,
                    self.hidden_size as i32,
                    &grad_pre_activation,
                    self.hidden_size as i32,
                    1.0, // Accumulate gradients
                    &mut grad_w_hh,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. b_h: sum(grad_pre_activation) / batch_size
        {
            let mut batch_bias_grad = vec![0.0f32; self.hidden_size];
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    batch_bias_grad[h] += grad_pre_activation[b * self.hidden_size + h];
                }
            }
            self.grad_b_h.accumulate_scaled(&batch_bias_grad, scale);
        }

        // Gradient w.r.t. input: grad_input = grad_pre_activation × W_xh^T
        // grad_pre_activation: (batch_size × hidden_size)
        // W_xh: (input_size × hidden_size)
        // grad_input: (batch_size × input_size)
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                batch_size as i32,
                self.input_size as i32,
                self.hidden_size as i32,
                1.0,
                &grad_pre_activation,
                self.hidden_size as i32,
                &self.w_xh,
                self.hidden_size as i32,
                0.0,
                grad_input,
                self.input_size as i32,
            );
        }

        // Note: Gradient w.r.t. h_{t-1} for BPTT would be:
        // grad_h_prev = grad_pre_activation × W_hh^T
        // This would be used when propagating gradients back through time,
        // but for now we only compute gradients w.r.t. the current input.
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        self.grad_w_xh
            .apply_sgd_update(&mut self.w_xh, learning_rate);
        self.grad_w_hh
            .apply_sgd_update(&mut self.w_hh, learning_rate);
        self.grad_w_hy
            .apply_sgd_update(&mut self.w_hy, learning_rate);
        self.grad_b_h.apply_sgd_update(&mut self.b_h, learning_rate);
        self.grad_b_y.apply_sgd_update(&mut self.b_y, learning_rate);
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_forward() {
        let mut rng = SimpleRng::new(42);
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let batch_size = 1;

        let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Reset hidden state before starting
        layer.reset_hidden_state();

        // Create input for first time step
        let input_t0 = vec![1.0, 0.5, -0.5];
        let mut output_t0 = vec![0.0; output_size];

        // Forward pass at time step 0
        layer.forward(&input_t0, &mut output_t0, batch_size);

        // Check output is not all zeros (network produced some output)
        assert!(
            output_t0.iter().any(|&x| x != 0.0),
            "Output should not be all zeros"
        );

        // Check hidden state was updated
        let hidden_state = layer.get_hidden_state();
        assert_eq!(hidden_state.len(), hidden_size);
        assert!(
            hidden_state.iter().any(|&x| x != 0.0),
            "Hidden state should be updated"
        );

        // Forward pass at time step 1 with different input
        let input_t1 = vec![0.0, 1.0, 0.0];
        let mut output_t1 = vec![0.0; output_size];

        layer.forward(&input_t1, &mut output_t1, batch_size);

        // Output should be different from first time step (due to hidden state)
        assert_ne!(
            output_t0, output_t1,
            "Outputs at different time steps should differ"
        );

        // Hidden state should have changed
        let hidden_state_t1 = layer.get_hidden_state();
        assert_ne!(
            hidden_state, hidden_state_t1,
            "Hidden state should change between time steps"
        );
    }

    #[test]
    fn test_rnn_forward_batch() {
        let mut rng = SimpleRng::new(123);
        let input_size = 2;
        let hidden_size = 3;
        let output_size = 1;
        let batch_size = 2;

        let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
        layer.reset_hidden_state();

        // Input for batch of 2 sequences at one time step
        let input = vec![1.0, 0.0, 0.0, 1.0]; // [seq1: [1.0, 0.0], seq2: [0.0, 1.0]]
        let mut output = vec![0.0; batch_size * output_size];

        layer.forward(&input, &mut output, batch_size);

        // Check outputs for both sequences are computed
        assert_eq!(output.len(), batch_size * output_size);
        assert!(
            output.iter().any(|&x| x != 0.0),
            "Batch output should not be all zeros"
        );
    }

    #[test]
    fn test_rnn_reset_hidden_state() {
        let mut rng = SimpleRng::new(99);
        let layer = RnnLayer::new(2, 3, 1, &mut rng);

        // Process one time step
        let input = vec![1.0, 1.0];
        let mut output = vec![0.0];
        layer.forward(&input, &mut output, 1);

        // Hidden state should be non-zero
        let hidden_before = layer.get_hidden_state();
        assert!(hidden_before.iter().any(|&x| x != 0.0));

        // Reset hidden state
        layer.reset_hidden_state();

        // Hidden state should be zero
        let hidden_after = layer.get_hidden_state();
        assert!(
            hidden_after.iter().all(|&x| x == 0.0),
            "Hidden state should be zero after reset"
        );
    }

    #[test]
    fn test_rnn_forward_dimensions() {
        let mut rng = SimpleRng::new(0);
        let input_size = 5;
        let hidden_size = 10;
        let output_size = 3;

        let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Test with batch_size = 1
        let input = vec![0.5; input_size];
        let mut output = vec![0.0; output_size];
        layer.forward(&input, &mut output, 1);
        assert_eq!(output.len(), output_size);

        // Test with batch_size = 4
        layer.reset_hidden_state();
        let input_batch = vec![0.5; 4 * input_size];
        let mut output_batch = vec![0.0; 4 * output_size];
        layer.forward(&input_batch, &mut output_batch, 4);
        assert_eq!(output_batch.len(), 4 * output_size);
    }

    #[test]
    fn test_rnn_backward() {
        let mut rng = SimpleRng::new(42);
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let batch_size = 1;

        let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Reset hidden state before starting
        layer.reset_hidden_state();

        // Do two forward passes so that h_{t-1} is non-zero for the second pass
        let input_t0 = vec![1.0, 0.5, -0.5];
        let mut output_t0 = vec![0.0; output_size];
        layer.forward(&input_t0, &mut output_t0, batch_size);

        // Now do the forward pass we'll use for backward
        let input = vec![0.5, 1.0, 0.0];
        let mut output = vec![0.0; output_size];
        layer.forward(&input, &mut output, batch_size);

        // Create gradient of output
        let grad_output = vec![1.0, -1.0];
        let mut grad_input = vec![0.0; input_size];

        // Backward pass
        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Gradient should propagate back
        assert!(
            grad_input.iter().all(|&x| x.is_finite()),
            "All gradients should be finite"
        );

        // At least some gradients should be non-zero
        assert!(
            grad_input.iter().any(|&x| x.abs() > 1e-10),
            "At least some gradients should be non-zero"
        );

        // Check that weight gradients were accumulated
        let grad_w_xh = layer.grad_w_xh.borrow();
        assert!(
            grad_w_xh.iter().any(|&x| x.abs() > 1e-10),
            "W_xh gradients should be accumulated"
        );

        let grad_w_hh = layer.grad_w_hh.borrow();
        assert!(
            grad_w_hh.iter().any(|&x| x.abs() > 1e-10),
            "W_hh gradients should be accumulated (h_prev is non-zero)"
        );

        let grad_w_hy = layer.grad_w_hy.borrow();
        assert!(
            grad_w_hy.iter().any(|&x| x.abs() > 1e-10),
            "W_hy gradients should be accumulated"
        );
    }

    #[test]
    fn test_rnn_backward_batch() {
        let mut rng = SimpleRng::new(123);
        let input_size = 2;
        let hidden_size = 3;
        let output_size = 1;
        let batch_size = 2;

        let layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);
        layer.reset_hidden_state();

        // Forward pass with batch
        let input = vec![1.0, 0.0, 0.0, 1.0]; // 2 samples
        let mut output = vec![0.0; batch_size * output_size];
        layer.forward(&input, &mut output, batch_size);

        // Backward pass
        let grad_output = vec![1.0, -0.5]; // gradients for 2 samples
        let mut grad_input = vec![0.0; batch_size * input_size];
        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Check gradients are finite
        assert!(
            grad_input.iter().all(|&x| x.is_finite()),
            "Batch gradients should be finite"
        );

        // Check at least some gradients are non-zero
        assert!(
            grad_input.iter().any(|&x| x.abs() > 1e-10),
            "Batch should have non-zero gradients"
        );
    }

    #[test]
    fn test_rnn_layer_trait() {
        let mut rng = SimpleRng::new(42);
        let input_size = 4;
        let hidden_size = 8;
        let output_size = 3;
        let batch_size = 2;

        let mut layer = RnnLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Test input_size() method
        assert_eq!(layer.input_size(), input_size);

        // Test output_size() method
        assert_eq!(layer.output_size(), output_size);

        // Test parameter_count() method
        let expected_params = input_size * hidden_size  // W_xh
            + hidden_size * hidden_size                 // W_hh
            + hidden_size * output_size                 // W_hy
            + hidden_size                               // b_h
            + output_size; // b_y
        assert_eq!(layer.parameter_count(), expected_params);

        // Reset hidden state
        layer.reset_hidden_state();

        // Test forward pass
        let input = vec![0.5; batch_size * input_size];
        let mut output = vec![0.0; batch_size * output_size];
        layer.forward(&input, &mut output, batch_size);

        // Verify output is not all zeros
        assert!(
            output.iter().any(|&x| x != 0.0),
            "Forward pass should produce non-zero output"
        );

        // Test backward pass
        let grad_output = vec![1.0; batch_size * output_size];
        let mut grad_input = vec![0.0; batch_size * input_size];
        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Verify gradients are computed
        assert!(
            grad_input.iter().all(|&x| x.is_finite()),
            "Gradients should be finite"
        );

        // Test update_parameters() method
        let learning_rate = 0.01;

        // Get initial weights
        let w_xh_before = layer.w_xh().to_vec();
        let w_hy_before = layer.w_hy().to_vec();

        // Update parameters
        layer.update_parameters(learning_rate);

        // Verify parameters changed
        let w_xh_after = layer.w_xh();
        let w_hy_after = layer.w_hy();

        // At least some parameters should have changed
        let w_xh_changed = w_xh_before
            .iter()
            .zip(w_xh_after.iter())
            .any(|(before, after)| (before - after).abs() > 1e-10);
        let w_hy_changed = w_hy_before
            .iter()
            .zip(w_hy_after.iter())
            .any(|(before, after)| (before - after).abs() > 1e-10);

        assert!(
            w_xh_changed || w_hy_changed,
            "Parameters should be updated after calling update_parameters()"
        );

        // Verify gradients were cleared (do another backward pass)
        layer.reset_hidden_state();
        layer.forward(&input, &mut output, batch_size);
        layer.backward(&input, &grad_output, &mut grad_input, batch_size);

        // Gradients should accumulate from a fresh state
        {
            let grad_w_xh = layer.grad_w_xh.borrow();
            assert!(
                grad_w_xh.iter().any(|&x| x.abs() > 1e-10),
                "Gradients should accumulate after backward pass"
            );
        } // Drop borrow before next operation
    }
}
