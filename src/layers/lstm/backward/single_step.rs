use super::super::LstmLayer;

impl LstmLayer {
    /// Backward propagation through the LSTM layer.
    ///
    /// Computes gradients with respect to all parameters and the input.
    /// Accumulates gradients in the gradient accumulators for use by
    /// `update_parameters_impl`.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (batch_size × input_size)
    /// * `grad_output` - Gradient of loss w.r.t. output (batch_size × output_size)
    /// * `grad_input` - Output buffer for gradients w.r.t. input (batch_size × input_size)
    /// * `batch_size` - Number of samples in batch
    ///
    /// # Panics
    ///
    /// Panics if `batch_size == 0` or if slice lengths don't match expected sizes.
    pub(in crate::layers::lstm) fn backward_impl(
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
            self.grad_b_y.accumulate_scaled(&batch_bias_grad, scale);
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
            self.grad_b_o.accumulate_scaled(&batch_bias_grad, scale);
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
            self.grad_b_f.accumulate_scaled(&batch_bias_grad, scale);
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
            self.grad_b_i.accumulate_scaled(&batch_bias_grad, scale);
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
            self.grad_b_c.accumulate_scaled(&batch_bias_grad, scale);
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

        // ========== Step 10: Compute grad_h_prev and grad_c_prev for BPTT ==========

        // grad_c_prev = grad_cell_state ⊙ f_t (element-wise, averaged over batch)
        // From c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t, the gradient w.r.t. c_{t-1} is f_t
        {
            let mut grad_c_prev_out = self.grad_c_prev.borrow_mut();
            grad_c_prev_out.fill(0.0);
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    let idx = b * self.hidden_size + h;
                    grad_c_prev_out[h] += grad_cell_state[idx] * forget_gate_batch[idx];
                }
            }
            for h in 0..self.hidden_size {
                grad_c_prev_out[h] *= scale;
            }
        }

        // grad_h_prev = contributions from all four gate pre-activations via their W_h matrices
        // grad_h_prev = grad_forget_gate_pre × W_hf^T
        //             + grad_input_gate_pre × W_hi^T
        //             + grad_cell_candidate_pre × W_hc^T
        //             + grad_output_gate_pre × W_ho^T
        // Each gate's pre-activation depends on h_{t-1} through: h_{t-1} × W_h + ...
        {
            let mut grad_h_prev_batch = vec![0.0f32; batch_size * self.hidden_size];

            // Contribution from forget gate: grad_h_prev += grad_forget_gate_pre × W_hf^T
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    batch_size as i32,
                    self.hidden_size as i32,
                    self.hidden_size as i32,
                    1.0,
                    &grad_forget_gate_pre,
                    self.hidden_size as i32,
                    &self.w_hf,
                    self.hidden_size as i32,
                    0.0,
                    &mut grad_h_prev_batch,
                    self.hidden_size as i32,
                );
            }

            // Contribution from input gate: grad_h_prev += grad_input_gate_pre × W_hi^T
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    batch_size as i32,
                    self.hidden_size as i32,
                    self.hidden_size as i32,
                    1.0,
                    &grad_input_gate_pre,
                    self.hidden_size as i32,
                    &self.w_hi,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_h_prev_batch,
                    self.hidden_size as i32,
                );
            }

            // Contribution from cell candidate: grad_h_prev += grad_cell_candidate_pre × W_hc^T
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    batch_size as i32,
                    self.hidden_size as i32,
                    self.hidden_size as i32,
                    1.0,
                    &grad_cell_candidate_pre,
                    self.hidden_size as i32,
                    &self.w_hc,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_h_prev_batch,
                    self.hidden_size as i32,
                );
            }

            // Contribution from output gate: grad_h_prev += grad_output_gate_pre × W_ho^T
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    batch_size as i32,
                    self.hidden_size as i32,
                    self.hidden_size as i32,
                    1.0,
                    &grad_output_gate_pre,
                    self.hidden_size as i32,
                    &self.w_ho,
                    self.hidden_size as i32,
                    1.0,
                    &mut grad_h_prev_batch,
                    self.hidden_size as i32,
                );
            }

            // Average over batch and store
            let mut grad_h_prev_out = self.grad_h_prev.borrow_mut();
            grad_h_prev_out.fill(0.0);
            for b in 0..batch_size {
                for h in 0..self.hidden_size {
                    grad_h_prev_out[h] += grad_h_prev_batch[b * self.hidden_size + h];
                }
            }
            for h in 0..self.hidden_size {
                grad_h_prev_out[h] *= scale;
            }
        }
    }
}
