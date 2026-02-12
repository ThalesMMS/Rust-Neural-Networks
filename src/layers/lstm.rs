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

use crate::layers::Layer;
use crate::utils::rng::SimpleRng;
use std::cell::RefCell;

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

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

    // Gradient accumulators (mutable interior via RefCell for trait compatibility)
    grad_w_xf: RefCell<Vec<f32>>,
    grad_w_hf: RefCell<Vec<f32>>,
    grad_b_f: RefCell<Vec<f32>>,
    grad_w_xi: RefCell<Vec<f32>>,
    grad_w_hi: RefCell<Vec<f32>>,
    grad_b_i: RefCell<Vec<f32>>,
    grad_w_xc: RefCell<Vec<f32>>,
    grad_w_hc: RefCell<Vec<f32>>,
    grad_b_c: RefCell<Vec<f32>>,
    grad_w_xo: RefCell<Vec<f32>>,
    grad_w_ho: RefCell<Vec<f32>>,
    grad_b_o: RefCell<Vec<f32>>,
    grad_w_hy: RefCell<Vec<f32>>,
    grad_b_y: RefCell<Vec<f32>>,

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
            grad_w_xf: RefCell::new(vec![0.0f32; input_size * hidden_size]),
            grad_w_hf: RefCell::new(vec![0.0f32; hidden_size * hidden_size]),
            grad_b_f: RefCell::new(vec![0.0f32; hidden_size]),
            grad_w_xi: RefCell::new(vec![0.0f32; input_size * hidden_size]),
            grad_w_hi: RefCell::new(vec![0.0f32; hidden_size * hidden_size]),
            grad_b_i: RefCell::new(vec![0.0f32; hidden_size]),
            grad_w_xc: RefCell::new(vec![0.0f32; input_size * hidden_size]),
            grad_w_hc: RefCell::new(vec![0.0f32; hidden_size * hidden_size]),
            grad_b_c: RefCell::new(vec![0.0f32; hidden_size]),
            grad_w_xo: RefCell::new(vec![0.0f32; input_size * hidden_size]),
            grad_w_ho: RefCell::new(vec![0.0f32; hidden_size * hidden_size]),
            grad_b_o: RefCell::new(vec![0.0f32; hidden_size]),
            grad_w_hy: RefCell::new(vec![0.0f32; hidden_size * output_size]),
            grad_b_y: RefCell::new(vec![0.0f32; output_size]),
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
    /// where:
    /// - `x_t` is the input at time step t (batch_size × input_size)
    /// - `h_{t-1}` is the previous hidden state (hidden_size), broadcasted to batch
    /// - `c_{t-1}` is the previous cell state (hidden_size), broadcasted to batch
    /// - `f_t` is the forget gate activation (batch_size × hidden_size)
    /// - `i_t` is the input gate activation (batch_size × hidden_size)
    /// - `c̃_t` is the cell candidate (batch_size × hidden_size)
    /// - `c_t` is the new cell state (batch_size × hidden_size)
    /// - `o_t` is the output gate activation (batch_size × hidden_size)
    /// - `h_t` is the new hidden state (batch_size × hidden_size)
    /// - `y_t` is the output (batch_size × output_size)
    /// - `σ` is the sigmoid activation function
    /// - `⊙` is element-wise multiplication
    ///
    /// # Matrix Operations
    ///
    /// **Forget Gate Computation:**
    /// 1. `x_t × W_xf`: (batch_size × input_size) × (input_size × hidden_size) → (batch_size × hidden_size)
    /// 2. `h_{t-1} × W_hf`: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
    /// 3. Add bias b_f and apply sigmoid element-wise
    ///
    /// **Input Gate Computation:**
    /// 1. `x_t × W_xi`: (batch_size × input_size) × (input_size × hidden_size) → (batch_size × hidden_size)
    /// 2. `h_{t-1} × W_hi`: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
    /// 3. Add bias b_i and apply sigmoid element-wise
    ///
    /// **Cell Candidate Computation:**
    /// 1. `x_t × W_xc`: (batch_size × input_size) × (input_size × hidden_size) → (batch_size × hidden_size)
    /// 2. `h_{t-1} × W_hc`: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
    /// 3. Add bias b_c and apply tanh element-wise
    ///
    /// **Cell State Update:**
    /// 1. `f_t ⊙ c_{t-1}`: Element-wise multiply forget gate with previous cell state
    /// 2. `i_t ⊙ c̃_t`: Element-wise multiply input gate with cell candidate
    /// 3. Sum the two products to get new cell state
    ///
    /// **Output Gate Computation:**
    /// 1. `x_t × W_xo`: (batch_size × input_size) × (input_size × hidden_size) → (batch_size × hidden_size)
    /// 2. `h_{t-1} × W_ho`: (batch_size × hidden_size) × (hidden_size × hidden_size) → (batch_size × hidden_size)
    /// 3. Add bias b_o and apply sigmoid element-wise
    ///
    /// **Hidden State Update:**
    /// 1. Apply tanh to cell state: tanh(c_t)
    /// 2. Element-wise multiply with output gate: h_t = o_t ⊙ tanh(c_t)
    ///
    /// **Output Projection:**
    /// 1. `h_t × W_hy`: (batch_size × hidden_size) × (hidden_size × output_size) → (batch_size × output_size)
    /// 2. Add bias b_y
    ///
    /// # Implementation Details
    ///
    /// - Uses BLAS `sgemm` for efficient matrix multiplication in all gate computations
    /// - Hidden state `h_{t-1}` and cell state `c_{t-1}` are broadcasted to all batch samples
    /// - All gate activations and states are cached for backward pass:
    ///   - Previous states: `h_{t-1}`, `c_{t-1}`
    ///   - Gate activations: `f_t`, `i_t`, `c̃_t`, `o_t`
    ///   - New states: `c_t`, `tanh(c_t)`, `h_t`
    /// - After processing, the layer's internal states are updated to `h_t[0]` and `c_t[0]` (first batch sample)
    /// - All samples in a batch share the same initial hidden/cell state
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

        // Get current hidden and cell states and cache them for backward pass
        let hidden = self.hidden_state.borrow();
        let cell = self.cell_state.borrow();
        {
            let mut cached_h_prev = self.cached_h_prev.borrow_mut();
            cached_h_prev.copy_from_slice(&hidden[..]);
            let mut cached_c_prev = self.cached_c_prev.borrow_mut();
            cached_c_prev.copy_from_slice(&cell[..]);
        }

        // Create batched hidden and cell states for batch processing
        let mut hidden_batch = vec![0.0f32; batch_size * self.hidden_size];
        for b in 0..batch_size {
            hidden_batch[b * self.hidden_size..(b + 1) * self.hidden_size]
                .copy_from_slice(&hidden[..]);
        }

        // ========== Compute Forget Gate: f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f) ==========
        let mut forget_gate = vec![0.0f32; batch_size * self.hidden_size];

        // x_t × W_xf
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
                &self.w_xf,
                self.hidden_size as i32,
                0.0,
                &mut forget_gate,
                self.hidden_size as i32,
            );
        }

        // h_{t-1} × W_hf
        let mut hf_contrib = vec![0.0f32; batch_size * self.hidden_size];
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
                &self.w_hf,
                self.hidden_size as i32,
                0.0,
                &mut hf_contrib,
                self.hidden_size as i32,
            );
        }

        // Add bias and apply sigmoid
        for i in 0..forget_gate.len() {
            let bias_idx = i % self.hidden_size;
            let pre_activation = forget_gate[i] + hf_contrib[i] + self.b_f[bias_idx];
            forget_gate[i] = 1.0 / (1.0 + (-pre_activation).exp()); // sigmoid
        }

        // ========== Compute Input Gate: i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i) ==========
        let mut input_gate = vec![0.0f32; batch_size * self.hidden_size];

        // x_t × W_xi
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
                &self.w_xi,
                self.hidden_size as i32,
                0.0,
                &mut input_gate,
                self.hidden_size as i32,
            );
        }

        // h_{t-1} × W_hi
        let mut hi_contrib = vec![0.0f32; batch_size * self.hidden_size];
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
                &self.w_hi,
                self.hidden_size as i32,
                0.0,
                &mut hi_contrib,
                self.hidden_size as i32,
            );
        }

        // Add bias and apply sigmoid
        for i in 0..input_gate.len() {
            let bias_idx = i % self.hidden_size;
            let pre_activation = input_gate[i] + hi_contrib[i] + self.b_i[bias_idx];
            input_gate[i] = 1.0 / (1.0 + (-pre_activation).exp()); // sigmoid
        }

        // ========== Compute Cell Candidate: c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c) ==========
        let mut cell_candidate = vec![0.0f32; batch_size * self.hidden_size];

        // x_t × W_xc
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
                &self.w_xc,
                self.hidden_size as i32,
                0.0,
                &mut cell_candidate,
                self.hidden_size as i32,
            );
        }

        // h_{t-1} × W_hc
        let mut hc_contrib = vec![0.0f32; batch_size * self.hidden_size];
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
                &self.w_hc,
                self.hidden_size as i32,
                0.0,
                &mut hc_contrib,
                self.hidden_size as i32,
            );
        }

        // Add bias and apply tanh
        for i in 0..cell_candidate.len() {
            let bias_idx = i % self.hidden_size;
            cell_candidate[i] = (cell_candidate[i] + hc_contrib[i] + self.b_c[bias_idx]).tanh();
        }

        // ========== Update Cell State: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t ==========
        let mut new_cell_state = vec![0.0f32; batch_size * self.hidden_size];
        for b in 0..batch_size {
            for h in 0..self.hidden_size {
                let idx = b * self.hidden_size + h;
                new_cell_state[idx] =
                    forget_gate[idx] * cell[h] + input_gate[idx] * cell_candidate[idx];
            }
        }

        // ========== Compute Output Gate: o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o) ==========
        let mut output_gate = vec![0.0f32; batch_size * self.hidden_size];

        // x_t × W_xo
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
                &self.w_xo,
                self.hidden_size as i32,
                0.0,
                &mut output_gate,
                self.hidden_size as i32,
            );
        }

        // h_{t-1} × W_ho
        let mut ho_contrib = vec![0.0f32; batch_size * self.hidden_size];
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
                &self.w_ho,
                self.hidden_size as i32,
                0.0,
                &mut ho_contrib,
                self.hidden_size as i32,
            );
        }

        // Add bias and apply sigmoid
        for i in 0..output_gate.len() {
            let bias_idx = i % self.hidden_size;
            let pre_activation = output_gate[i] + ho_contrib[i] + self.b_o[bias_idx];
            output_gate[i] = 1.0 / (1.0 + (-pre_activation).exp()); // sigmoid
        }

        // ========== Compute Hidden State: h_t = o_t ⊙ tanh(c_t) ==========
        let mut cell_tanh = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..new_cell_state.len() {
            cell_tanh[i] = new_cell_state[i].tanh();
        }

        let mut new_hidden_state = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..new_hidden_state.len() {
            new_hidden_state[i] = output_gate[i] * cell_tanh[i];
        }

        // Cache all intermediate values for backward pass
        {
            let mut cached_forget_gate = self.cached_forget_gate.borrow_mut();
            if batch_size == 1 {
                cached_forget_gate.copy_from_slice(&forget_gate[..]);
            } else {
                cached_forget_gate.copy_from_slice(&forget_gate[0..self.hidden_size]);
            }

            let mut cached_input_gate = self.cached_input_gate.borrow_mut();
            if batch_size == 1 {
                cached_input_gate.copy_from_slice(&input_gate[..]);
            } else {
                cached_input_gate.copy_from_slice(&input_gate[0..self.hidden_size]);
            }

            let mut cached_cell_candidate = self.cached_cell_candidate.borrow_mut();
            if batch_size == 1 {
                cached_cell_candidate.copy_from_slice(&cell_candidate[..]);
            } else {
                cached_cell_candidate.copy_from_slice(&cell_candidate[0..self.hidden_size]);
            }

            let mut cached_output_gate = self.cached_output_gate.borrow_mut();
            if batch_size == 1 {
                cached_output_gate.copy_from_slice(&output_gate[..]);
            } else {
                cached_output_gate.copy_from_slice(&output_gate[0..self.hidden_size]);
            }

            let mut cached_cell_state = self.cached_cell_state.borrow_mut();
            if batch_size == 1 {
                cached_cell_state.copy_from_slice(&new_cell_state[..]);
            } else {
                cached_cell_state.copy_from_slice(&new_cell_state[0..self.hidden_size]);
            }

            let mut cached_cell_tanh = self.cached_cell_tanh.borrow_mut();
            if batch_size == 1 {
                cached_cell_tanh.copy_from_slice(&cell_tanh[..]);
            } else {
                cached_cell_tanh.copy_from_slice(&cell_tanh[0..self.hidden_size]);
            }
        }

        // Update internal states (use first batch item for batch_size > 1)
        drop(hidden);
        drop(cell);
        {
            let mut hidden_mut = self.hidden_state.borrow_mut();
            hidden_mut.copy_from_slice(&new_hidden_state[0..self.hidden_size]);

            let mut cell_mut = self.cell_state.borrow_mut();
            cell_mut.copy_from_slice(&new_cell_state[0..self.hidden_size]);
        }

        // ========== Compute Output: y_t = h_t × W_hy + b_y ==========
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                batch_size as i32,
                self.output_size as i32,
                self.hidden_size as i32,
                1.0,
                &new_hidden_state,
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
    /// # Mathematical Formulation
    ///
    /// Given gradient w.r.t. output: ∂L/∂y_t (batch_size × output_size)
    ///
    /// The backward pass reverses the forward computation through the following steps:
    ///
    /// ## Step 1: Output Projection Gradients
    ///
    /// From: `y_t = h_t × W_hy + b_y`
    ///
    /// **Weight gradients:**
    /// - ∂L/∂W_hy = h_t^T × ∂L/∂y_t
    /// - Dimension: (hidden_size × batch_size) × (batch_size × output_size) → (hidden_size × output_size)
    ///
    /// **Bias gradients:**
    /// - ∂L/∂b_y = Σ(∂L/∂y_t) along batch dimension
    /// - Dimension: sum over (batch_size × output_size) → (output_size)
    ///
    /// **Hidden state gradients:**
    /// - ∂L/∂h_t = ∂L/∂y_t × W_hy^T
    /// - Dimension: (batch_size × output_size) × (output_size × hidden_size) → (batch_size × hidden_size)
    ///
    /// ## Step 2: Hidden State Update Gradients
    ///
    /// From: `h_t = o_t ⊙ tanh(c_t)`
    ///
    /// **Output gate gradients:**
    /// - ∂L/∂o_t = ∂L/∂h_t ⊙ tanh(c_t)
    /// - Uses: ∂h_t/∂o_t = tanh(c_t)
    ///
    /// **Cell tanh gradients:**
    /// - ∂L/∂tanh(c_t) = ∂L/∂h_t ⊙ o_t
    /// - Uses: ∂h_t/∂tanh(c_t) = o_t
    ///
    /// ## Step 3: Tanh Derivative
    ///
    /// From: `tanh(c_t)`
    ///
    /// **Cell state gradients:**
    /// - ∂L/∂c_t = ∂L/∂tanh(c_t) ⊙ (1 - tanh²(c_t))
    /// - Uses: ∂tanh(x)/∂x = 1 - tanh²(x)
    ///
    /// ## Step 4: Output Gate Weight Gradients
    ///
    /// From: `o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)`
    ///
    /// **Sigmoid derivative:**
    /// - ∂L/∂(pre_o) = ∂L/∂o_t ⊙ o_t ⊙ (1 - o_t)
    /// - Uses: ∂σ(x)/∂x = σ(x) ⊙ (1 - σ(x))
    ///
    /// **Weight gradients:**
    /// - ∂L/∂W_xo = x_t^T × ∂L/∂(pre_o)
    /// - ∂L/∂W_ho = h_{t-1}^T × ∂L/∂(pre_o)
    /// - ∂L/∂b_o = Σ(∂L/∂(pre_o)) along batch
    ///
    /// ## Step 5: Cell State Update Gradients
    ///
    /// From: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
    ///
    /// **Chain rule application:**
    ///
    /// **Forget gate gradients:**
    /// - ∂L/∂f_t = ∂L/∂c_t ⊙ c_{t-1}
    /// - Uses: ∂c_t/∂f_t = c_{t-1}
    ///
    /// **Input gate gradients:**
    /// - ∂L/∂i_t = ∂L/∂c_t ⊙ c̃_t
    /// - Uses: ∂c_t/∂i_t = c̃_t
    ///
    /// **Cell candidate gradients:**
    /// - ∂L/∂c̃_t = ∂L/∂c_t ⊙ i_t
    /// - Uses: ∂c_t/∂c̃_t = i_t
    ///
    /// **Previous cell state gradients (BPTT):**
    /// - ∂L/∂c_{t-1} = ∂L/∂c_t ⊙ f_t
    /// - Uses: ∂c_t/∂c_{t-1} = f_t
    /// - Note: This gradient flows back through time for BPTT
    ///
    /// ## Step 6: Forget Gate Weight Gradients
    ///
    /// From: `f_t = σ(x_t × W_xf + h_{t-1} × W_hf + b_f)`
    ///
    /// **Sigmoid derivative:**
    /// - ∂L/∂(pre_f) = ∂L/∂f_t ⊙ f_t ⊙ (1 - f_t)
    ///
    /// **Weight gradients:**
    /// - ∂L/∂W_xf = x_t^T × ∂L/∂(pre_f)
    /// - ∂L/∂W_hf = h_{t-1}^T × ∂L/∂(pre_f)
    /// - ∂L/∂b_f = Σ(∂L/∂(pre_f)) along batch
    ///
    /// ## Step 7: Input Gate Weight Gradients
    ///
    /// From: `i_t = σ(x_t × W_xi + h_{t-1} × W_hi + b_i)`
    ///
    /// **Sigmoid derivative:**
    /// - ∂L/∂(pre_i) = ∂L/∂i_t ⊙ i_t ⊙ (1 - i_t)
    ///
    /// **Weight gradients:**
    /// - ∂L/∂W_xi = x_t^T × ∂L/∂(pre_i)
    /// - ∂L/∂W_hi = h_{t-1}^T × ∂L/∂(pre_i)
    /// - ∂L/∂b_i = Σ(∂L/∂(pre_i)) along batch
    ///
    /// ## Step 8: Cell Candidate Weight Gradients
    ///
    /// From: `c̃_t = tanh(x_t × W_xc + h_{t-1} × W_hc + b_c)`
    ///
    /// **Tanh derivative:**
    /// - ∂L/∂(pre_c) = ∂L/∂c̃_t ⊙ (1 - c̃_t²)
    ///
    /// **Weight gradients:**
    /// - ∂L/∂W_xc = x_t^T × ∂L/∂(pre_c)
    /// - ∂L/∂W_hc = h_{t-1}^T × ∂L/∂(pre_c)
    /// - ∂L/∂b_c = Σ(∂L/∂(pre_c)) along batch
    ///
    /// ## Step 9: Input and Hidden State Gradients (BPTT)
    ///
    /// **Input gradients:**
    /// - ∂L/∂x_t = ∂L/∂(pre_f) × W_xf^T + ∂L/∂(pre_i) × W_xi^T + ∂L/∂(pre_c) × W_xc^T + ∂L/∂(pre_o) × W_xo^T
    /// - Accumulates contributions from all four gates
    ///
    /// **Previous hidden state gradients (BPTT):**
    /// - ∂L/∂h_{t-1} = ∂L/∂(pre_f) × W_hf^T + ∂L/∂(pre_i) × W_hi^T + ∂L/∂(pre_c) × W_hc^T + ∂L/∂(pre_o) × W_ho^T
    /// - Flows back through time for BPTT
    /// - Note: Not explicitly computed in single-step backward, but implicitly handled through recurrence
    ///
    /// # BPTT Gradient Flow
    ///
    /// The LSTM's gating mechanism creates multiple gradient paths:
    ///
    /// 1. **Cell state path**: Gradients flow through ∂L/∂c_{t-1} = ∂L/∂c_t ⊙ f_t
    ///    - The forget gate controls gradient magnitude
    ///    - When f_t ≈ 1, gradients flow unchanged (mitigates vanishing gradients)
    ///    - When f_t ≈ 0, gradients are blocked (selective forgetting)
    ///
    /// 2. **Hidden state paths**: Gradients flow through W_hf, W_hi, W_hc, W_ho
    ///    - Four parallel paths from h_t to h_{t-1}
    ///    - Provides rich gradient signal across time steps
    ///
    /// 3. **Gate modulation**: Each gate's sigmoid derivative σ'(x) = σ(x)(1-σ(x))
    ///    - Maximum gradient at σ(x) = 0.5
    ///    - Gradients vanish when gates saturate (σ → 0 or σ → 1)
    ///
    /// # Implementation Notes
    ///
    /// - Gradients are accumulated (use `1.0` as beta in BLAS sgemm) for BPTT
    /// - All intermediate values from forward pass are cached for backward use
    /// - Batch processing: gradients averaged by `scale = 1.0 / batch_size`
    /// - Gate derivatives computed element-wise before matrix operations
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
        use cblas::{sgemm, Layout, Transpose};

        if batch_size == 0 {
            panic!("batch_size cannot be zero in LSTM::backward");
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
        let cached_c_prev = self.cached_c_prev.borrow();
        let cached_forget_gate = self.cached_forget_gate.borrow();
        let cached_input_gate = self.cached_input_gate.borrow();
        let cached_cell_candidate = self.cached_cell_candidate.borrow();
        let cached_output_gate = self.cached_output_gate.borrow();
        let cached_cell_tanh = self.cached_cell_tanh.borrow();

        // Replicate cached values for batch processing
        let mut h_prev_batch = vec![0.0f32; batch_size * self.hidden_size];
        let mut forget_gate_batch = vec![0.0f32; batch_size * self.hidden_size];
        let mut input_gate_batch = vec![0.0f32; batch_size * self.hidden_size];
        let mut cell_candidate_batch = vec![0.0f32; batch_size * self.hidden_size];
        let mut output_gate_batch = vec![0.0f32; batch_size * self.hidden_size];
        let mut cell_tanh_batch = vec![0.0f32; batch_size * self.hidden_size];

        for b in 0..batch_size {
            let start = b * self.hidden_size;
            let end = (b + 1) * self.hidden_size;
            h_prev_batch[start..end].copy_from_slice(&cached_h_prev[..]);
            forget_gate_batch[start..end].copy_from_slice(&cached_forget_gate[..]);
            input_gate_batch[start..end].copy_from_slice(&cached_input_gate[..]);
            cell_candidate_batch[start..end].copy_from_slice(&cached_cell_candidate[..]);
            output_gate_batch[start..end].copy_from_slice(&cached_output_gate[..]);
            cell_tanh_batch[start..end].copy_from_slice(&cached_cell_tanh[..]);
        }

        // ========== Step 1: Backprop through output projection (y_t = h_t × W_hy + b_y) ==========

        // Gradient w.r.t. W_hy: grad_W_hy = h_t^T × grad_output / batch_size
        // We need h_t, which we can reconstruct from o_t ⊙ tanh(c_t)
        let mut h_current_batch = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..h_current_batch.len() {
            h_current_batch[i] = output_gate_batch[i] * cell_tanh_batch[i];
        }

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

            let mut grad_b_y = self.grad_b_y.borrow_mut();
            for (acc, g) in grad_b_y.iter_mut().zip(batch_bias_grad.iter()) {
                *acc += *g * scale;
            }
        }

        // Gradient w.r.t. h_t: grad_h = grad_output × W_hy^T
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

        // ========== Step 2: Backprop through hidden state (h_t = o_t ⊙ tanh(c_t)) ==========

        // Gradient w.r.t. o_t: grad_o = grad_h ⊙ tanh(c_t)
        let mut grad_output_gate = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_output_gate.len() {
            grad_output_gate[i] = grad_h[i] * cell_tanh_batch[i];
        }

        // Gradient w.r.t. tanh(c_t): grad_cell_tanh = grad_h ⊙ o_t
        let mut grad_cell_tanh = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_cell_tanh.len() {
            grad_cell_tanh[i] = grad_h[i] * output_gate_batch[i];
        }

        // ========== Step 3: Backprop through tanh to get grad w.r.t. c_t ==========
        // tanh'(c_t) = 1 - tanh(c_t)^2
        let mut grad_cell_state = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_cell_state.len() {
            let tanh_val = cell_tanh_batch[i];
            grad_cell_state[i] = grad_cell_tanh[i] * (1.0 - tanh_val * tanh_val);
        }

        // ========== Step 4: Backprop through output gate (o_t = σ(x_t × W_xo + h_{t-1} × W_ho + b_o)) ==========

        // Backprop through sigmoid: σ'(z) = σ(z) * (1 - σ(z))
        let mut grad_output_gate_pre = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_output_gate_pre.len() {
            let o_val = output_gate_batch[i];
            grad_output_gate_pre[i] = grad_output_gate[i] * o_val * (1.0 - o_val);
        }

        // Gradient w.r.t. W_xo
        {
            let mut grad_w_xo = self.grad_w_xo.borrow_mut();
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
                    &grad_output_gate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_xo,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. W_ho
        {
            let mut grad_w_ho = self.grad_w_ho.borrow_mut();
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
                    &grad_output_gate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_ho,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. b_o
        {
            let mut batch_bias_grad = vec![0.0f32; self.hidden_size];
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    batch_bias_grad[h] += grad_output_gate_pre[b * self.hidden_size + h];
                }
            }

            let mut grad_b_o = self.grad_b_o.borrow_mut();
            for (acc, g) in grad_b_o.iter_mut().zip(batch_bias_grad.iter()) {
                *acc += *g * scale;
            }
        }

        // ========== Step 5: Backprop through cell state (c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t) ==========

        // Gradient w.r.t. f_t: grad_f = grad_c_t ⊙ c_{t-1}
        let mut grad_forget_gate = vec![0.0f32; batch_size * self.hidden_size];
        for b in 0..batch_size {
            for h in 0..self.hidden_size {
                let idx = b * self.hidden_size + h;
                grad_forget_gate[idx] = grad_cell_state[idx] * cached_c_prev[h];
            }
        }

        // Gradient w.r.t. i_t: grad_i = grad_c_t ⊙ c̃_t
        let mut grad_input_gate = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_input_gate.len() {
            grad_input_gate[i] = grad_cell_state[i] * cell_candidate_batch[i];
        }

        // Gradient w.r.t. c̃_t: grad_c_tilde = grad_c_t ⊙ i_t
        let mut grad_cell_candidate = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_cell_candidate.len() {
            grad_cell_candidate[i] = grad_cell_state[i] * input_gate_batch[i];
        }

        // ========== Step 6: Backprop through forget gate (f_t = σ(...)) ==========

        let mut grad_forget_gate_pre = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_forget_gate_pre.len() {
            let f_val = forget_gate_batch[i];
            grad_forget_gate_pre[i] = grad_forget_gate[i] * f_val * (1.0 - f_val);
        }

        // Gradient w.r.t. W_xf
        {
            let mut grad_w_xf = self.grad_w_xf.borrow_mut();
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
                    &grad_forget_gate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_xf,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. W_hf
        {
            let mut grad_w_hf = self.grad_w_hf.borrow_mut();
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
                    &grad_forget_gate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_hf,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. b_f
        {
            let mut batch_bias_grad = vec![0.0f32; self.hidden_size];
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    batch_bias_grad[h] += grad_forget_gate_pre[b * self.hidden_size + h];
                }
            }

            let mut grad_b_f = self.grad_b_f.borrow_mut();
            for (acc, g) in grad_b_f.iter_mut().zip(batch_bias_grad.iter()) {
                *acc += *g * scale;
            }
        }

        // ========== Step 7: Backprop through input gate (i_t = σ(...)) ==========

        let mut grad_input_gate_pre = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_input_gate_pre.len() {
            let i_val = input_gate_batch[i];
            grad_input_gate_pre[i] = grad_input_gate[i] * i_val * (1.0 - i_val);
        }

        // Gradient w.r.t. W_xi
        {
            let mut grad_w_xi = self.grad_w_xi.borrow_mut();
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
                    &grad_input_gate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_xi,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. W_hi
        {
            let mut grad_w_hi = self.grad_w_hi.borrow_mut();
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
                    &grad_input_gate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_hi,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. b_i
        {
            let mut batch_bias_grad = vec![0.0f32; self.hidden_size];
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    batch_bias_grad[h] += grad_input_gate_pre[b * self.hidden_size + h];
                }
            }

            let mut grad_b_i = self.grad_b_i.borrow_mut();
            for (acc, g) in grad_b_i.iter_mut().zip(batch_bias_grad.iter()) {
                *acc += *g * scale;
            }
        }

        // ========== Step 8: Backprop through cell candidate (c̃_t = tanh(...)) ==========

        let mut grad_cell_candidate_pre = vec![0.0f32; batch_size * self.hidden_size];
        for i in 0..grad_cell_candidate_pre.len() {
            let c_tilde = cell_candidate_batch[i];
            grad_cell_candidate_pre[i] = grad_cell_candidate[i] * (1.0 - c_tilde * c_tilde);
        }

        // Gradient w.r.t. W_xc
        {
            let mut grad_w_xc = self.grad_w_xc.borrow_mut();
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
                    &grad_cell_candidate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_xc,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. W_hc
        {
            let mut grad_w_hc = self.grad_w_hc.borrow_mut();
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
                    &grad_cell_candidate_pre,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_w_hc,
                    self.hidden_size as i32,
                );
            }
        }

        // Gradient w.r.t. b_c
        {
            let mut batch_bias_grad = vec![0.0f32; self.hidden_size];
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    batch_bias_grad[h] += grad_cell_candidate_pre[b * self.hidden_size + h];
                }
            }

            let mut grad_b_c = self.grad_b_c.borrow_mut();
            for (acc, g) in grad_b_c.iter_mut().zip(batch_bias_grad.iter()) {
                *acc += *g * scale;
            }
        }

        // ========== Step 9: Compute gradient w.r.t. input ==========
        // Accumulate contributions from all four gates
        grad_input.fill(0.0);

        // Contribution from forget gate: grad_input += grad_forget_gate_pre × W_xf^T
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                batch_size as i32,
                self.input_size as i32,
                self.hidden_size as i32,
                1.0,
                &grad_forget_gate_pre,
                self.hidden_size as i32,
                &self.w_xf,
                self.hidden_size as i32,
                1.0,
                grad_input,
                self.input_size as i32,
            );
        }

        // Contribution from input gate: grad_input += grad_input_gate_pre × W_xi^T
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                batch_size as i32,
                self.input_size as i32,
                self.hidden_size as i32,
                1.0,
                &grad_input_gate_pre,
                self.hidden_size as i32,
                &self.w_xi,
                self.hidden_size as i32,
                1.0,
                grad_input,
                self.input_size as i32,
            );
        }

        // Contribution from cell candidate: grad_input += grad_cell_candidate_pre × W_xc^T
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                batch_size as i32,
                self.input_size as i32,
                self.hidden_size as i32,
                1.0,
                &grad_cell_candidate_pre,
                self.hidden_size as i32,
                &self.w_xc,
                self.hidden_size as i32,
                1.0,
                grad_input,
                self.input_size as i32,
            );
        }

        // Contribution from output gate: grad_input += grad_output_gate_pre × W_xo^T
        unsafe {
            sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::Ordinary,
                batch_size as i32,
                self.input_size as i32,
                self.hidden_size as i32,
                1.0,
                &grad_output_gate_pre,
                self.hidden_size as i32,
                &self.w_xo,
                self.hidden_size as i32,
                1.0,
                grad_input,
                self.input_size as i32,
            );
        }

        // Note: Gradients w.r.t. h_{t-1} and c_{t-1} for BPTT would be computed here,
        // but for now we only compute gradients w.r.t. the current input.
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
        // Update forget gate weights
        {
            let grad_w_xf = self.grad_w_xf.borrow();
            for (w, &g) in self.w_xf.iter_mut().zip(grad_w_xf.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_w_hf = self.grad_w_hf.borrow();
            for (w, &g) in self.w_hf.iter_mut().zip(grad_w_hf.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_b_f = self.grad_b_f.borrow();
            for (b, &g) in self.b_f.iter_mut().zip(grad_b_f.iter()) {
                *b -= learning_rate * g;
            }
        }

        // Update input gate weights
        {
            let grad_w_xi = self.grad_w_xi.borrow();
            for (w, &g) in self.w_xi.iter_mut().zip(grad_w_xi.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_w_hi = self.grad_w_hi.borrow();
            for (w, &g) in self.w_hi.iter_mut().zip(grad_w_hi.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_b_i = self.grad_b_i.borrow();
            for (b, &g) in self.b_i.iter_mut().zip(grad_b_i.iter()) {
                *b -= learning_rate * g;
            }
        }

        // Update cell candidate weights
        {
            let grad_w_xc = self.grad_w_xc.borrow();
            for (w, &g) in self.w_xc.iter_mut().zip(grad_w_xc.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_w_hc = self.grad_w_hc.borrow();
            for (w, &g) in self.w_hc.iter_mut().zip(grad_w_hc.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_b_c = self.grad_b_c.borrow();
            for (b, &g) in self.b_c.iter_mut().zip(grad_b_c.iter()) {
                *b -= learning_rate * g;
            }
        }

        // Update output gate weights
        {
            let grad_w_xo = self.grad_w_xo.borrow();
            for (w, &g) in self.w_xo.iter_mut().zip(grad_w_xo.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_w_ho = self.grad_w_ho.borrow();
            for (w, &g) in self.w_ho.iter_mut().zip(grad_w_ho.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_b_o = self.grad_b_o.borrow();
            for (b, &g) in self.b_o.iter_mut().zip(grad_b_o.iter()) {
                *b -= learning_rate * g;
            }
        }

        // Update output projection weights
        {
            let grad_w_hy = self.grad_w_hy.borrow();
            for (w, &g) in self.w_hy.iter_mut().zip(grad_w_hy.iter()) {
                *w -= learning_rate * g;
            }
        }
        {
            let grad_b_y = self.grad_b_y.borrow();
            for (b, &g) in self.b_y.iter_mut().zip(grad_b_y.iter()) {
                *b -= learning_rate * g;
            }
        }

        // Clear gradient accumulators
        self.grad_w_xf.borrow_mut().fill(0.0);
        self.grad_w_hf.borrow_mut().fill(0.0);
        self.grad_b_f.borrow_mut().fill(0.0);
        self.grad_w_xi.borrow_mut().fill(0.0);
        self.grad_w_hi.borrow_mut().fill(0.0);
        self.grad_b_i.borrow_mut().fill(0.0);
        self.grad_w_xc.borrow_mut().fill(0.0);
        self.grad_w_hc.borrow_mut().fill(0.0);
        self.grad_b_c.borrow_mut().fill(0.0);
        self.grad_w_xo.borrow_mut().fill(0.0);
        self.grad_w_ho.borrow_mut().fill(0.0);
        self.grad_b_o.borrow_mut().fill(0.0);
        self.grad_w_hy.borrow_mut().fill(0.0);
        self.grad_b_y.borrow_mut().fill(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_creation() {
        let mut rng = SimpleRng::new(42);
        let layer = LstmLayer::new(64, 128, 10, &mut rng);

        assert_eq!(layer.input_size(), 64);
        assert_eq!(layer.hidden_size(), 128);
        assert_eq!(layer.output_size(), 10);

        // Verify state vectors are initialized to zero
        let hidden = layer.get_hidden_state();
        assert_eq!(hidden.len(), 128);
        assert!(hidden.iter().all(|&x| x == 0.0));

        let cell = layer.get_cell_state();
        assert_eq!(cell.len(), 128);
        assert!(cell.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_lstm_parameter_count() {
        let mut rng = SimpleRng::new(42);
        let layer = LstmLayer::new(64, 128, 10, &mut rng);

        // For each gate: input_size × hidden_size + hidden_size × hidden_size + hidden_size
        let gate_params = 64 * 128 + 128 * 128 + 128;
        // 4 gates + output projection (hidden_size × output_size + output_size)
        let expected = 4 * gate_params + (128 * 10 + 10);

        assert_eq!(layer.parameter_count(), expected);
    }

    #[test]
    fn test_lstm_state_management() {
        let mut rng = SimpleRng::new(42);
        let layer = LstmLayer::new(32, 64, 5, &mut rng);

        // Test setting and getting hidden state
        let test_hidden = vec![0.5f32; 64];
        layer.set_hidden_state(&test_hidden);
        let retrieved_hidden = layer.get_hidden_state();
        assert_eq!(retrieved_hidden, test_hidden);

        // Test setting and getting cell state
        let test_cell = vec![0.3f32; 64];
        layer.set_cell_state(&test_cell);
        let retrieved_cell = layer.get_cell_state();
        assert_eq!(retrieved_cell, test_cell);

        // Test reset
        layer.reset_state();
        let reset_hidden = layer.get_hidden_state();
        let reset_cell = layer.get_cell_state();
        assert!(reset_hidden.iter().all(|&x| x == 0.0));
        assert!(reset_cell.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "Hidden state length must match hidden_size")]
    fn test_lstm_invalid_hidden_state_length() {
        let mut rng = SimpleRng::new(42);
        let layer = LstmLayer::new(32, 64, 5, &mut rng);
        layer.set_hidden_state(&[0.0f32; 32]); // Wrong size
    }

    #[test]
    #[should_panic(expected = "Cell state length must match hidden_size")]
    fn test_lstm_invalid_cell_state_length() {
        let mut rng = SimpleRng::new(42);
        let layer = LstmLayer::new(32, 64, 5, &mut rng);
        layer.set_cell_state(&[0.0f32; 32]); // Wrong size
    }

    #[test]
    fn test_lstm_forward() {
        let mut rng = SimpleRng::new(42);
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let batch_size = 1;

        let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Reset states before starting
        layer.reset_state();

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

        // Check cell state was updated
        let cell_state = layer.get_cell_state();
        assert_eq!(cell_state.len(), hidden_size);
        assert!(
            cell_state.iter().any(|&x| x != 0.0),
            "Cell state should be updated"
        );

        // Forward pass at time step 1 with different input
        let input_t1 = vec![0.0, 1.0, 0.0];
        let mut output_t1 = vec![0.0; output_size];

        layer.forward(&input_t1, &mut output_t1, batch_size);

        // Output should be different from first time step (due to hidden/cell state)
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

        // Cell state should have changed
        let cell_state_t1 = layer.get_cell_state();
        assert_ne!(
            cell_state, cell_state_t1,
            "Cell state should change between time steps"
        );

        // Verify all outputs are finite (no NaN or Inf)
        assert!(
            output_t1.iter().all(|&x| x.is_finite()),
            "All outputs should be finite"
        );
        assert!(
            hidden_state_t1.iter().all(|&x| x.is_finite()),
            "All hidden state values should be finite"
        );
        assert!(
            cell_state_t1.iter().all(|&x| x.is_finite()),
            "All cell state values should be finite"
        );
    }

    #[test]
    fn test_lstm_backward() {
        let mut rng = SimpleRng::new(42);
        let input_size = 3;
        let hidden_size = 4;
        let output_size = 2;
        let batch_size = 1;

        let layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Reset states before starting
        layer.reset_state();

        // Do two forward passes so that h_{t-1} and c_{t-1} are non-zero for the second pass
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

        // Check that gate weight gradients were accumulated
        let grad_w_xf = layer.grad_w_xf.borrow();
        assert!(
            grad_w_xf.iter().any(|&x| x.abs() > 1e-10),
            "W_xf gradients should be accumulated"
        );

        let grad_w_xi = layer.grad_w_xi.borrow();
        assert!(
            grad_w_xi.iter().any(|&x| x.abs() > 1e-10),
            "W_xi gradients should be accumulated"
        );

        let grad_w_xc = layer.grad_w_xc.borrow();
        assert!(
            grad_w_xc.iter().any(|&x| x.abs() > 1e-10),
            "W_xc gradients should be accumulated"
        );

        let grad_w_xo = layer.grad_w_xo.borrow();
        assert!(
            grad_w_xo.iter().any(|&x| x.abs() > 1e-10),
            "W_xo gradients should be accumulated"
        );

        let grad_w_hy = layer.grad_w_hy.borrow();
        assert!(
            grad_w_hy.iter().any(|&x| x.abs() > 1e-10),
            "W_hy gradients should be accumulated"
        );
    }

    #[test]
    fn test_lstm_layer_trait() {
        let mut rng = SimpleRng::new(42);
        let input_size = 4;
        let hidden_size = 8;
        let output_size = 3;
        let batch_size = 2;

        let mut layer = LstmLayer::new(input_size, hidden_size, output_size, &mut rng);

        // Test input_size() method
        assert_eq!(layer.input_size(), input_size);

        // Test output_size() method
        assert_eq!(layer.output_size(), output_size);

        // Test parameter_count() method
        // For each of 4 gates: input_size × hidden_size + hidden_size × hidden_size + hidden_size
        let gate_params = input_size * hidden_size + hidden_size * hidden_size + hidden_size;
        // Output projection: hidden_size × output_size + output_size
        let output_params = hidden_size * output_size + output_size;
        let expected_params = 4 * gate_params + output_params;
        assert_eq!(layer.parameter_count(), expected_params);

        // Reset states
        layer.reset_state();

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

        // Verify at least some input gradients are non-zero
        assert!(
            grad_input.iter().any(|&x| x.abs() > 1e-10),
            "At least some input gradients should be non-zero"
        );

        // Test update_parameters() method
        let learning_rate = 0.01;

        // Call update parameters (this should clear gradients even if parameters don't change much)
        layer.update_parameters(learning_rate);

        // Verify gradients were cleared after update
        {
            let grad_w_xf = layer.grad_w_xf.borrow();
            assert!(
                grad_w_xf.iter().all(|&x| x == 0.0),
                "W_xf gradients should be cleared after update_parameters()"
            );
        }
        {
            let grad_w_hy = layer.grad_w_hy.borrow();
            assert!(
                grad_w_hy.iter().all(|&x| x == 0.0),
                "W_hy gradients should be cleared after update_parameters()"
            );
        }
        {
            let grad_b_y = layer.grad_b_y.borrow();
            assert!(
                grad_b_y.iter().all(|&x| x == 0.0),
                "b_y gradients should be cleared after update_parameters()"
            );
        }
        {
            let grad_w_xi = layer.grad_w_xi.borrow();
            assert!(
                grad_w_xi.iter().all(|&x| x == 0.0),
                "All gradients should be cleared after update_parameters()"
            );
        }
    }
}
