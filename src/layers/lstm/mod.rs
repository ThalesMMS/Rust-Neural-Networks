//! Long Short-Term Memory (LSTM) layer implementation
//!
//! This module provides an LSTM layer with gated architecture that can learn
//! long-term dependencies in sequential data through its cell state mechanism.
//!
//! # Architecture
//!
//! The LSTM layer uses a gated architecture with three gates (forget, input, output)
//! and a cell state that allows information to flow unchanged across time steps:
//!
//! 1. **Forget gate**: Controls what information to discard from cell state
//!    - `f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)`
//! 2. **Input gate**: Controls what new information to add to cell state
//!    - `i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)`
//! 3. **Cell candidate**: New information that could be added to cell state
//!    - `c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)`
//! 4. **Cell state update**: Combines forget and input gates
//!    - `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
//! 5. **Output gate**: Controls what information from cell state to output
//!    - `o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)`
//! 6. **Hidden state update**: Filtered cell state
//!    - `h_t = o_t ⊙ tanh(c_t)`
//! 7. **Output projection**: Maps hidden state to output
//!    - `y_t = h_t × W_hy + b_y`
//!
//! where σ is sigmoid, ⊙ is element-wise multiplication.
//!
//! # Usage Example
//!
//! ```ignore
//! use rust_neural_networks::layers::{LstmLayer, Layer};
//! use rust_neural_networks::utils::rng::SimpleRng;
//!
//! // Create LSTM layer: 10 input features, 20 hidden units, 5 outputs
//! let mut rng = SimpleRng::new(42);
//! let layer = LstmLayer::new(10, 20, 5, &mut rng);
//!
//! // Process a sequence of 3 time steps
//! layer.reset_state();  // Clear hidden and cell states before new sequence
//! for t in 0..3 {
//!     let input = vec![0.5; 10];      // Input for time step t
//!     let mut output = vec![0.0; 5];  // Output buffer
//!     layer.forward(&input, &mut output, 1);
//!     // Hidden and cell states are automatically maintained between time steps
//! }
//! ```
//!
//! # Advantages Over Vanilla RNN
//!
//! - **Long-term dependencies**: Cell state provides a highway for gradients
//! - **Mitigates vanishing gradients**: Gating mechanism allows selective gradient flow
//! - **Adaptive memory**: Gates learn what to remember and forget
//!
//! # Important Notes
//!
//! - Always call `reset_state()` at the beginning of a new sequence
//! - Both hidden state and cell state persist across forward passes
//! - For batch processing, all samples in a batch share the same initial states
//! - The LSTM has 4× more parameters than a vanilla RNN of the same size due to gating

use crate::layers::gradient::GradientAccumulator;
use crate::layers::Layer;
use crate::utils::rng::SimpleRng;
use std::cell::RefCell;

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

mod backward;
mod forward;
#[cfg(test)]
mod tests;

/// LSTM layer with gated architecture for learning long-term dependencies.
///
/// The LSTM computes:
/// - Forget gate: f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)
/// - Input gate: i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)
/// - Cell candidate: c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)
/// - Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
/// - Output gate: o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)
/// - Hidden state: h_t = o_t ⊙ tanh(c_t)
/// - Output: y_t = h_t × W_hy + b_y
///
/// where σ is sigmoid, ⊙ is element-wise multiplication, x_t is input at time t,
/// h_t is hidden state, and c_t is cell state.
///
/// # Fields
///
/// * `input_size` - Number of input features per time step
/// * `hidden_size` - Size of hidden and cell state vectors
/// * `output_size` - Number of output features
///
/// ## Forget Gate Weights
/// * `w_xf` - Input-to-forget gate weights (input_size × hidden_size)
/// * `w_hf` - Hidden-to-forget gate weights (hidden_size × hidden_size)
/// * `b_f` - Forget gate biases (hidden_size)
///
/// ## Input Gate Weights
/// * `w_xi` - Input-to-input gate weights (input_size × hidden_size)
/// * `w_hi` - Hidden-to-input gate weights (hidden_size × hidden_size)
/// * `b_i` - Input gate biases (hidden_size)
///
/// ## Cell Candidate Weights
/// * `w_xc` - Input-to-cell candidate weights (input_size × hidden_size)
/// * `w_hc` - Hidden-to-cell candidate weights (hidden_size × hidden_size)
/// * `b_c` - Cell candidate biases (hidden_size)
///
/// ## Output Gate Weights
/// * `w_xo` - Input-to-output gate weights (input_size × hidden_size)
/// * `w_ho` - Hidden-to-output gate weights (hidden_size × hidden_size)
/// * `b_o` - Output gate biases (hidden_size)
///
/// ## Output Projection Weights
/// * `w_hy` - Hidden-to-output weights (hidden_size × output_size)
/// * `b_y` - Output biases (output_size)
///
/// ## State Vectors
/// * `hidden_state` - Current hidden state h_t (hidden_size)
/// * `cell_state` - Current cell state c_t (hidden_size)
///
/// # Example
///
/// ```ignore
/// use rust_neural_networks::layers::LstmLayer;
/// use rust_neural_networks::utils::SimpleRng;
///
/// let mut rng = SimpleRng::new(42);
/// let layer = LstmLayer::new(128, 256, 10, &mut rng);
/// assert_eq!(layer.input_size(), 128);
/// assert_eq!(layer.hidden_size(), 256);
/// assert_eq!(layer.output_size(), 10);
/// ```
pub struct LstmLayer {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    // Forget gate weights
    w_xf: Vec<f32>, // input_size × hidden_size
    w_hf: Vec<f32>, // hidden_size × hidden_size
    b_f: Vec<f32>,  // hidden_size

    // Input gate weights
    w_xi: Vec<f32>, // input_size × hidden_size
    w_hi: Vec<f32>, // hidden_size × hidden_size
    b_i: Vec<f32>,  // hidden_size

    // Cell candidate weights
    w_xc: Vec<f32>, // input_size × hidden_size
    w_hc: Vec<f32>, // hidden_size × hidden_size
    b_c: Vec<f32>,  // hidden_size

    // Output gate weights
    w_xo: Vec<f32>, // input_size × hidden_size
    w_ho: Vec<f32>, // hidden_size × hidden_size
    b_o: Vec<f32>,  // hidden_size

    // Output projection weights
    w_hy: Vec<f32>, // hidden_size × output_size
    b_y: Vec<f32>,  // output_size

    // State vectors
    hidden_state: RefCell<Vec<f32>>, // hidden_size
    cell_state: RefCell<Vec<f32>>,   // hidden_size

    // Gradient accumulators
    grad_w_xf: GradientAccumulator,
    grad_w_hf: GradientAccumulator,
    grad_b_f: GradientAccumulator,
    grad_w_xi: GradientAccumulator,
    grad_w_hi: GradientAccumulator,
    grad_b_i: GradientAccumulator,
    grad_w_xc: GradientAccumulator,
    grad_w_hc: GradientAccumulator,
    grad_b_c: GradientAccumulator,
    grad_w_xo: GradientAccumulator,
    grad_w_ho: GradientAccumulator,
    grad_b_o: GradientAccumulator,
    grad_w_hy: GradientAccumulator,
    grad_b_y: GradientAccumulator,

    // Cache for backward pass
    cached_h_prev: RefCell<Vec<f32>>, // h_{t-1} before forward pass
    cached_c_prev: RefCell<Vec<f32>>, // c_{t-1} before forward pass
    cached_forget_gate: RefCell<Vec<f32>>, // f_t after sigmoid
    cached_input_gate: RefCell<Vec<f32>>, // i_t after sigmoid
    cached_cell_candidate: RefCell<Vec<f32>>, // c̃_t after tanh
    cached_output_gate: RefCell<Vec<f32>>, // o_t after sigmoid
    cached_cell_state: RefCell<Vec<f32>>, // c_t after update
    cached_cell_tanh: RefCell<Vec<f32>>, // tanh(c_t)
}

impl LstmLayer {
    /// Creates an LSTM layer with Xavier-initialized weights and zero biases.
    ///
    /// All weight matrices are sampled uniformly from [-limit, limit], where
    /// `limit = sqrt(6.0 / (fan_in + fan_out))` for each matrix.
    /// Biases, state vectors, and gradient accumulators are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features per time step
    /// * `hidden_size` - Size of hidden and cell state vectors
    /// * `output_size` - Number of output features
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(64, 128, 10, &mut rng);
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
        // Xavier initialization helper function
        let init_weights =
            |size: usize, fan_in: usize, fan_out: usize, rng: &mut SimpleRng| -> Vec<f32> {
                let mut weights = vec![0.0f32; size];
                let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
                for value in &mut weights {
                    *value = rng.gen_range_f32(-limit, limit);
                }
                weights
            };

        // Initialize forget gate weights
        let w_xf = init_weights(input_size * hidden_size, input_size, hidden_size, rng);
        let w_hf = init_weights(hidden_size * hidden_size, hidden_size, hidden_size, rng);

        // Initialize input gate weights
        let w_xi = init_weights(input_size * hidden_size, input_size, hidden_size, rng);
        let w_hi = init_weights(hidden_size * hidden_size, hidden_size, hidden_size, rng);

        // Initialize cell candidate weights
        let w_xc = init_weights(input_size * hidden_size, input_size, hidden_size, rng);
        let w_hc = init_weights(hidden_size * hidden_size, hidden_size, hidden_size, rng);

        // Initialize output gate weights
        let w_xo = init_weights(input_size * hidden_size, input_size, hidden_size, rng);
        let w_ho = init_weights(hidden_size * hidden_size, hidden_size, hidden_size, rng);

        // Initialize output projection weights
        let w_hy = init_weights(hidden_size * output_size, hidden_size, output_size, rng);

        Self {
            input_size,
            hidden_size,
            output_size,
            w_xf,
            w_hf,
            b_f: vec![0.0f32; hidden_size],
            w_xi,
            w_hi,
            b_i: vec![0.0f32; hidden_size],
            w_xc,
            w_hc,
            b_c: vec![0.0f32; hidden_size],
            w_xo,
            w_ho,
            b_o: vec![0.0f32; hidden_size],
            w_hy,
            b_y: vec![0.0f32; output_size],
            hidden_state: RefCell::new(vec![0.0f32; hidden_size]),
            cell_state: RefCell::new(vec![0.0f32; hidden_size]),
            grad_w_xf: GradientAccumulator::new(input_size * hidden_size),
            grad_w_hf: GradientAccumulator::new(hidden_size * hidden_size),
            grad_b_f: GradientAccumulator::new(hidden_size),
            grad_w_xi: GradientAccumulator::new(input_size * hidden_size),
            grad_w_hi: GradientAccumulator::new(hidden_size * hidden_size),
            grad_b_i: GradientAccumulator::new(hidden_size),
            grad_w_xc: GradientAccumulator::new(input_size * hidden_size),
            grad_w_hc: GradientAccumulator::new(hidden_size * hidden_size),
            grad_b_c: GradientAccumulator::new(hidden_size),
            grad_w_xo: GradientAccumulator::new(input_size * hidden_size),
            grad_w_ho: GradientAccumulator::new(hidden_size * hidden_size),
            grad_b_o: GradientAccumulator::new(hidden_size),
            grad_w_hy: GradientAccumulator::new(hidden_size * output_size),
            grad_b_y: GradientAccumulator::new(output_size),
            cached_h_prev: RefCell::new(vec![0.0f32; hidden_size]),
            cached_c_prev: RefCell::new(vec![0.0f32; hidden_size]),
            cached_forget_gate: RefCell::new(vec![0.0f32; hidden_size]),
            cached_input_gate: RefCell::new(vec![0.0f32; hidden_size]),
            cached_cell_candidate: RefCell::new(vec![0.0f32; hidden_size]),
            cached_output_gate: RefCell::new(vec![0.0f32; hidden_size]),
            cached_cell_state: RefCell::new(vec![0.0f32; hidden_size]),
            cached_cell_tanh: RefCell::new(vec![0.0f32; hidden_size]),
        }
    }

    /// Get the hidden state size of the layer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::LstmLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(64, 128, 10, &mut rng);
    /// assert_eq!(layer.hidden_size(), 128);
    /// ```
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Reset the hidden state and cell state to zeros.
    ///
    /// This should be called at the beginning of a new sequence to clear
    /// any information from previous sequences.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// layer.reset_state();
    /// // Process new sequence...
    /// ```
    pub fn reset_state(&self) {
        let mut hidden = self.hidden_state.borrow_mut();
        for h in hidden.iter_mut() {
            *h = 0.0;
        }
        let mut cell = self.cell_state.borrow_mut();
        for c in cell.iter_mut() {
            *c = 0.0;
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
    /// use rust_neural_networks::layers::{LstmLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(10, 20, 5, &mut rng);
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

    /// Get a copy of the current cell state.
    ///
    /// The cell state represents the long-term memory of the LSTM,
    /// storing information that persists across many time steps.
    ///
    /// # Returns
    ///
    /// A vector containing the current cell state values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{LstmLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Process some input
    /// let input = vec![0.5; 10];
    /// let mut output = vec![0.0; 5];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// // Inspect cell state (long-term memory)
    /// let cell = layer.get_cell_state();
    /// assert_eq!(cell.len(), 20);
    /// ```
    pub fn get_cell_state(&self) -> Vec<f32> {
        self.cell_state.borrow().clone()
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
    /// use rust_neural_networks::layers::LstmLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Initialize with custom state
    /// let custom_state = vec![0.1; 20];
    /// layer.set_hidden_state(&custom_state);
    ///
    /// let retrieved = layer.get_hidden_state();
    /// assert_eq!(retrieved, custom_state);
    /// ```
    pub fn set_hidden_state(&self, state: &[f32]) {
        assert_eq!(
            state.len(),
            self.hidden_size,
            "Hidden state length must match hidden_size"
        );
        let mut hidden = self.hidden_state.borrow_mut();
        hidden.copy_from_slice(state);
    }

    /// Set the cell state to specific values.
    ///
    /// Useful for initializing the cell state (long-term memory) with specific
    /// values or resuming computation from a saved state.
    ///
    /// # Arguments
    ///
    /// * `state` - New cell state values (must have length equal to hidden_size)
    ///
    /// # Panics
    ///
    /// Panics if the provided state vector has incorrect length.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::LstmLayer;
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Initialize cell state with specific values
    /// let custom_cell = vec![0.2; 20];
    /// layer.set_cell_state(&custom_cell);
    ///
    /// let retrieved = layer.get_cell_state();
    /// assert_eq!(retrieved, custom_cell);
    /// ```
    pub fn set_cell_state(&self, state: &[f32]) {
        assert_eq!(
            state.len(),
            self.hidden_size,
            "Cell state length must match hidden_size"
        );
        let mut cell = self.cell_state.borrow_mut();
        cell.copy_from_slice(state);
    }
}

impl Layer for LstmLayer {
    /// Forward propagation through the LSTM layer for one time step.
    ///
    /// Processes input through all gates and updates hidden and cell states.
    /// For batch processing, each sample uses the same initial hidden/cell state.
    ///
    /// The forward pass computes all four gates (forget, input, cell, output),
    /// updates the cell state, computes the new hidden state, and projects to output.
    /// All intermediate values are cached for use in the backward pass.
    ///
    /// # Mathematical Formulation
    ///
    /// The LSTM forward pass consists of seven sequential steps:
    ///
    /// **Step 1: Forget Gate**
    /// - Controls what information to discard from cell state
    /// - `f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)`
    ///
    /// **Step 2: Input Gate**
    /// - Controls what new information to add to cell state
    /// - `i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)`
    ///
    /// **Step 3: Cell Candidate**
    /// - New information that could be added to cell state
    /// - `c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)`
    ///
    /// **Step 4: Cell State Update**
    /// - Combines forget and input gates with cell state
    /// - `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
    ///
    /// **Step 5: Output Gate**
    /// - Controls what information from cell state to output
    /// - `o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)`
    ///
    /// **Step 6: Hidden State Update**
    /// - Filtered cell state becomes new hidden state
    /// - `h_t = o_t ⊙ tanh(c_t)`
    ///
    /// **Step 7: Output Projection**
    /// - Maps hidden state to output space
    /// - `y_t = h_t × W_hy + b_y`
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (batch_size × input_size)
    /// * `output` - Output buffer (batch_size × output_size)
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Panics
    ///
    /// Panics if input or output buffers have incorrect sizes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{LstmLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(10, 20, 5, &mut rng);
    ///
    /// layer.reset_state();
    /// let input = vec![0.5; 10];
    /// let mut output = vec![0.0; 5];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// // Output and states have been updated
    /// assert!(output.iter().any(|&x| x != 0.0));
    /// ```
    fn forward(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
        self.forward_impl(input, output, batch_size);
    }

    /// Backward propagation through the LSTM layer using BPTT.
    ///
    /// Computes gradients for all gate weights and propagates gradients back through time.
    /// The backward pass must be called after a forward pass, as it uses cached values
    /// from the forward computation.
    ///
    /// Gradients are computed for all four gates, their weights and biases, and the
    /// output projection. The gradients are accumulated (not replaced) to support
    /// backpropagation through time (BPTT).
    ///
    /// # Arguments
    ///
    /// * `input` - Input from forward pass (batch_size × input_size)
    /// * `grad_output` - Gradient w.r.t. output (batch_size × output_size)
    /// * `grad_input` - Buffer for gradient w.r.t. input (batch_size × input_size)
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{LstmLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let layer = LstmLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Forward pass
    /// let input = vec![0.5; 10];
    /// let mut output = vec![0.0; 5];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// // Backward pass
    /// let grad_output = vec![1.0; 5];
    /// let mut grad_input = vec![0.0; 10];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// // Gradients have been computed
    /// assert!(grad_input.iter().any(|&x| x.abs() > 1e-10));
    /// ```
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        batch_size: usize,
    ) {
        self.backward_impl(input, grad_output, grad_input, batch_size);
    }

    /// Update LSTM parameters using gradient descent.
    ///
    /// Updates all gate weights, biases, and output projection using accumulated gradients.
    /// Clears gradient accumulators after update.
    ///
    /// This method applies gradient descent: `parameter -= learning_rate × gradient`
    /// for all 4 gates (forget, input, cell, output) plus the output projection,
    /// totaling 14 parameter groups (4 gates × 3 params + output × 2 params).
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use rust_neural_networks::layers::{LstmLayer, Layer};
    /// use rust_neural_networks::utils::rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new(42);
    /// let mut layer = LstmLayer::new(10, 20, 5, &mut rng);
    ///
    /// // Forward and backward passes accumulate gradients
    /// let input = vec![0.5; 10];
    /// let mut output = vec![0.0; 5];
    /// layer.forward(&input, &mut output, 1);
    ///
    /// let grad_output = vec![1.0; 5];
    /// let mut grad_input = vec![0.0; 10];
    /// layer.backward(&input, &grad_output, &mut grad_input, 1);
    ///
    /// // Update parameters and clear gradients
    /// layer.update_parameters(0.01);
    /// ```
    fn update_parameters(&mut self, learning_rate: f32) {
        self.update_parameters_impl(learning_rate);
    }

    /// Get the input size of the layer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(layer.input_size(), 128);
    /// ```
    fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get the output size of the layer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(layer.output_size(), 10);
    /// ```
    fn output_size(&self) -> usize {
        self.output_size
    }

    /// Get the total number of trainable parameters.
    ///
    /// Returns the sum of all weight matrix elements and bias vector elements
    /// across all four gates plus the output projection.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let count = layer.parameter_count();
    /// // For input_size=64, hidden_size=128, output_size=10:
    /// // 4 gates × (64×128 + 128×128 + 128) + (128×10 + 10) = 74,762
    /// ```
    fn parameter_count(&self) -> usize {
        // Forget gate: w_xf + w_hf + b_f
        let forget_params = self.w_xf.len() + self.w_hf.len() + self.b_f.len();
        // Input gate: w_xi + w_hi + b_i
        let input_params = self.w_xi.len() + self.w_hi.len() + self.b_i.len();
        // Cell candidate: w_xc + w_hc + b_c
        let cell_params = self.w_xc.len() + self.w_hc.len() + self.b_c.len();
        // Output gate: w_xo + w_ho + b_o
        let output_gate_params = self.w_xo.len() + self.w_ho.len() + self.b_o.len();
        // Output projection: w_hy + b_y
        let output_proj_params = self.w_hy.len() + self.b_y.len();

        forget_params + input_params + cell_params + output_gate_params + output_proj_params
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
