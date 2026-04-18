use super::super::LstmLayer;

impl LstmLayer {
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
    pub(in crate::layers::lstm) fn update_parameters_impl(&mut self, learning_rate: f32) {
        // Update forget gate weights
        self.grad_w_xf
            .apply_sgd_update(&mut self.w_xf, learning_rate);
        self.grad_w_hf
            .apply_sgd_update(&mut self.w_hf, learning_rate);
        self.grad_b_f.apply_sgd_update(&mut self.b_f, learning_rate);

        // Update input gate weights
        self.grad_w_xi
            .apply_sgd_update(&mut self.w_xi, learning_rate);
        self.grad_w_hi
            .apply_sgd_update(&mut self.w_hi, learning_rate);
        self.grad_b_i.apply_sgd_update(&mut self.b_i, learning_rate);

        // Update cell candidate weights
        self.grad_w_xc
            .apply_sgd_update(&mut self.w_xc, learning_rate);
        self.grad_w_hc
            .apply_sgd_update(&mut self.w_hc, learning_rate);
        self.grad_b_c.apply_sgd_update(&mut self.b_c, learning_rate);

        // Update output gate weights
        self.grad_w_xo
            .apply_sgd_update(&mut self.w_xo, learning_rate);
        self.grad_w_ho
            .apply_sgd_update(&mut self.w_ho, learning_rate);
        self.grad_b_o.apply_sgd_update(&mut self.b_o, learning_rate);

        // Update output projection weights
        self.grad_w_hy
            .apply_sgd_update(&mut self.w_hy, learning_rate);
        self.grad_b_y.apply_sgd_update(&mut self.b_y, learning_rate);
    }
}
