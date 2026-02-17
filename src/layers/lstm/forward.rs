//! Forward pass implementation for the LSTM layer.
//!
//! This module contains the forward propagation logic extracted from the main
//! LSTM module for better code organization.

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

use super::LstmLayer;

impl LstmLayer {
    /// Forward propagation through the LSTM layer for one time step.
    ///
    /// Processes input through all gates and updates hidden and cell states.
    /// For batch processing, each sample uses the same initial hidden/cell state.
    ///
    /// The forward pass computes all four gates (forget, input, cell, output),
    /// updates the cell state, computes the new hidden state, and projects to output.
    /// All intermediate values are cached for use in the backward pass.
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
    pub(super) fn forward_impl(&self, input: &[f32], output: &mut [f32], batch_size: usize) {
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
}
