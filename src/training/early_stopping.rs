/// Outcome returned by [`EarlyStopping::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingAction {
    /// Validation loss improved by at least `min_delta`; the caller should save
    /// a model checkpoint.
    Improved,
    /// No improvement, but patience has not been exceeded; training continues.
    Continue,
    /// No improvement for `patience` consecutive epochs; training should stop.
    Stop,
}

/// Tracks the best observed validation loss and fires after `patience` epochs
/// without improvement.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::training::{EarlyStopping, EarlyStoppingAction};
///
/// let mut es = EarlyStopping::new(3, 0.001);
///
/// // First epoch always improves (starting from f32::INFINITY).
/// assert_eq!(es.check(1.0), EarlyStoppingAction::Improved);
///
/// // No improvement for 3 consecutive epochs triggers Stop.
/// assert_eq!(es.check(1.5), EarlyStoppingAction::Continue);
/// assert_eq!(es.check(1.5), EarlyStoppingAction::Continue);
/// assert_eq!(es.check(1.5), EarlyStoppingAction::Stop);
/// ```
pub struct EarlyStopping {
    /// The lowest validation loss observed so far.
    pub best_val_loss: f32,
    /// How many consecutive epochs have passed without a sufficient improvement.
    pub epochs_without_improvement: usize,
    patience: usize,
    min_delta: f32,
}

impl EarlyStopping {
    /// Constructs a new `EarlyStopping` monitor configured with the given patience and minimum improvement.
    ///
    /// The monitor starts with `best_val_loss` set to `f32::INFINITY` and `epochs_without_improvement` set to `0`.
    ///
    /// # Parameters
    ///
    /// - `patience`: number of consecutive epochs without a sufficient decrease in validation loss before the monitor signals `EarlyStoppingAction::Stop`.
    /// - `min_delta`: minimum decrease in validation loss required to count as an improvement.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::EarlyStopping;
    ///
    /// let mut es = EarlyStopping::new(3, 1e-4);
    /// assert_eq!(es.best_val_loss, f32::INFINITY);
    /// assert_eq!(es.epochs_without_improvement, 0);
    /// ```
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            best_val_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            patience,
            min_delta,
        }
    }

    /// Update internal state with the latest validation loss and decide whether training should save, continue, or stop.
    ///
    /// Returns `EarlyStoppingAction::Improved` when `val_loss` is at least `min_delta` lower than the previous best (resets the non‑improvement counter and updates `best_val_loss`), `EarlyStoppingAction::Stop` when the non‑improvement counter reaches `patience`, and `EarlyStoppingAction::Continue` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::{EarlyStopping, EarlyStoppingAction};
    ///
    /// let mut es = EarlyStopping::new(2, 0.01);
    /// assert_eq!(es.check(1.0), EarlyStoppingAction::Improved);
    /// assert_eq!(es.check(1.0), EarlyStoppingAction::Continue);
    /// assert_eq!(es.check(1.0), EarlyStoppingAction::Stop);
    /// ```
    pub fn check(&mut self, val_loss: f32) -> EarlyStoppingAction {
        if val_loss <= self.best_val_loss - self.min_delta {
            self.best_val_loss = val_loss;
            self.epochs_without_improvement = 0;
            EarlyStoppingAction::Improved
        } else {
            self.epochs_without_improvement += 1;
            if self.epochs_without_improvement >= self.patience {
                EarlyStoppingAction::Stop
            } else {
                EarlyStoppingAction::Continue
            }
        }
    }

    /// Indicates whether training has exceeded the configured patience and should stop.
    ///
    /// # Returns
    ///
    /// `true` if the number of consecutive epochs without improvement is greater than or equal to the configured patience, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::EarlyStopping;
    ///
    /// let mut es = EarlyStopping::new(3, 1e-4);
    /// assert!(!es.should_stop());
    /// es.epochs_without_improvement = 3;
    /// assert!(es.should_stop());
    /// ```
    pub fn should_stop(&self) -> bool {
        self.epochs_without_improvement >= self.patience
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_improves() {
        let mut es = EarlyStopping::new(3, 0.001);
        assert_eq!(es.check(1.0), EarlyStoppingAction::Improved);
        assert_eq!(es.best_val_loss, 1.0);
        assert_eq!(es.epochs_without_improvement, 0);
    }

    #[test]
    fn test_early_stopping_continues() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0);
        assert_eq!(es.check(1.5), EarlyStoppingAction::Continue);
        assert_eq!(es.epochs_without_improvement, 1);
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0);
        es.check(1.5);
        es.check(1.5);
        assert_eq!(es.check(1.5), EarlyStoppingAction::Stop);
        assert!(es.should_stop());
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0);
        es.check(1.5);
        assert_eq!(es.check(0.5), EarlyStoppingAction::Improved);
        assert_eq!(es.epochs_without_improvement, 0);
        assert!(!es.should_stop());
    }

    #[test]
    fn test_early_stopping_counts_exact_min_delta_as_improvement() {
        let mut es = EarlyStopping::new(3, 0.001);
        es.check(1.0);

        assert_eq!(es.check(0.999), EarlyStoppingAction::Improved);
        assert_eq!(es.best_val_loss, 0.999);
        assert_eq!(es.epochs_without_improvement, 0);
    }
}
