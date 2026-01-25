//! Tests for learning rate schedulers
//!
//! This file tests all LRScheduler implementations:
//! - StepDecay: step-based decay with gamma factor
//! - ExponentialDecay: continuous exponential decay
//! - CosineAnnealing: cosine curve decay to minimum

use approx::assert_relative_eq;
use rust_neural_networks::utils::lr_scheduler::{
    CosineAnnealing, ExponentialDecay, LRScheduler, StepDecay,
};

// ============================================================================
// StepDecay Tests
// ============================================================================

#[cfg(test)]
mod step_decay_tests {
    use super::*;

    #[test]
    fn test_step_decay_creation() {
        let scheduler = StepDecay::new(0.1, 3, 0.5);
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_step_decay_initial_lr() {
        let scheduler = StepDecay::new(0.01, 5, 0.1);
        assert_eq!(scheduler.get_lr(), 0.01);
    }

    #[test]
    fn test_step_decay_before_first_step() {
        let mut scheduler = StepDecay::new(0.1, 3, 0.5);

        // Before first step boundary, LR should remain constant
        for _ in 0..2 {
            scheduler.step();
            assert_eq!(scheduler.get_lr(), 0.1);
        }
    }

    #[test]
    fn test_step_decay_at_first_step() {
        let mut scheduler = StepDecay::new(0.1, 3, 0.5);

        // At epoch 3, LR should decay
        for _ in 0..3 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_lr(), 0.05, epsilon = 1e-6); // 0.1 * 0.5
    }

    /// Verifies that StepDecay applies successive multiplicative decays every `step_size` epochs.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut scheduler = StepDecay::new(0.1, 3, 0.5);
    ///
    /// // After 6 epochs (2 decay steps)
    /// for _ in 0..6 { scheduler.step(); }
    /// assert_relative_eq!(scheduler.get_lr(), 0.025, epsilon = 1e-6); // 0.1 * 0.5^2
    ///
    /// // After 9 epochs (3 decay steps)
    /// for _ in 0..3 { scheduler.step(); }
    /// assert_relative_eq!(scheduler.get_lr(), 0.0125, epsilon = 1e-6); // 0.1 * 0.5^3
    /// ```
    #[test]
    fn test_step_decay_multiple_steps() {
        let mut scheduler = StepDecay::new(0.1, 3, 0.5);

        // After 6 epochs (2 decay steps)
        for _ in 0..6 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_lr(), 0.025, epsilon = 1e-6); // 0.1 * 0.5^2

        // After 9 epochs (3 decay steps)
        for _ in 0..3 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_lr(), 0.0125, epsilon = 1e-6); // 0.1 * 0.5^3
    }

    #[test]
    fn test_step_decay_different_gamma() {
        let mut scheduler = StepDecay::new(0.1, 2, 0.1);

        // After 2 epochs
        for _ in 0..2 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-6); // 0.1 * 0.1
    }

    #[test]
    fn test_step_decay_reset() {
        let mut scheduler = StepDecay::new(0.1, 3, 0.5);

        // Decay LR
        for _ in 0..6 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_lr(), 0.025, epsilon = 1e-6);

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.1);

        // Verify decay works again after reset
        for _ in 0..3 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_lr(), 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_step_decay_step_size_one() {
        let mut scheduler = StepDecay::new(1.0, 1, 0.9);

        // With step_size=1, decay happens every epoch
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.9, epsilon = 1e-6);

        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.81, epsilon = 1e-6); // 0.9^2
    }

    /// Verifies that a StepDecay scheduler with a large `step_size` holds the initial learning rate until the decay boundary, then applies a single decay at that boundary.
    ///
    /// The learning rate remains at the configured `initial_lr` for epochs 0 through 99 and is multiplied by `gamma` at epoch 100.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut scheduler = StepDecay::new(0.1, 100, 0.5);
    /// for _ in 0..99 {
    ///     scheduler.step();
    ///     assert_eq!(scheduler.get_lr(), 0.1);
    /// }
    /// scheduler.step();
    /// assert_relative_eq!(scheduler.get_lr(), 0.05, epsilon = 1e-6);
    /// ```
    #[test]
    fn test_step_decay_large_step_size() {
        let mut scheduler = StepDecay::new(0.1, 100, 0.5);

        // With large step_size, LR should stay constant for many epochs
        for _ in 0..99 {
            scheduler.step();
            assert_eq!(scheduler.get_lr(), 0.1);
        }

        // At epoch 100, it should decay
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_step_decay_zero_epoch() {
        let scheduler = StepDecay::new(0.5, 5, 0.2);
        // At epoch 0 (initial), LR should be initial_lr
        assert_eq!(scheduler.get_lr(), 0.5);
    }

    #[test]
    fn test_step_decay_consistency() {
        let mut scheduler1 = StepDecay::new(0.1, 3, 0.5);
        let mut scheduler2 = StepDecay::new(0.1, 3, 0.5);

        // Same configuration should produce same results
        for _ in 0..10 {
            scheduler1.step();
            scheduler2.step();
            assert_eq!(scheduler1.get_lr(), scheduler2.get_lr());
        }
    }
}

// ============================================================================
// ExponentialDecay Tests
// ============================================================================

#[cfg(test)]
mod exponential_decay_tests {
    use super::*;

    #[test]
    fn test_exponential_decay_creation() {
        let scheduler = ExponentialDecay::new(0.1, 0.95);
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_exponential_decay_initial_lr() {
        let scheduler = ExponentialDecay::new(0.01, 0.99);
        assert_eq!(scheduler.get_lr(), 0.01);
    }

    #[test]
    fn test_exponential_decay_first_step() {
        let mut scheduler = ExponentialDecay::new(0.1, 0.95);

        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.095, epsilon = 1e-6); // 0.1 * 0.95
    }

    #[test]
    fn test_exponential_decay_multiple_steps() {
        let mut scheduler = ExponentialDecay::new(0.1, 0.95);

        // After 2 epochs
        scheduler.step();
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.09025, epsilon = 1e-6); // 0.1 * 0.95^2

        // After 5 epochs total
        for _ in 0..3 {
            scheduler.step();
        }
        let expected = 0.1 * 0.95_f32.powi(5);
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-6);
    }

    #[test]
    fn test_exponential_decay_continuous() {
        let mut scheduler = ExponentialDecay::new(1.0, 0.9);

        let mut prev_lr = scheduler.get_lr();

        // Verify LR decreases monotonically every epoch
        for _ in 0..10 {
            scheduler.step();
            let curr_lr = scheduler.get_lr();
            assert!(curr_lr < prev_lr);
            assert_relative_eq!(curr_lr / prev_lr, 0.9, epsilon = 1e-6);
            prev_lr = curr_lr;
        }
    }

    #[test]
    fn test_exponential_decay_different_gamma() {
        let mut scheduler = ExponentialDecay::new(1.0, 0.5);

        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.5, epsilon = 1e-6);

        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_exponential_decay_reset() {
        let mut scheduler = ExponentialDecay::new(0.1, 0.95);

        // Decay LR
        for _ in 0..10 {
            scheduler.step();
        }
        let decayed_lr = scheduler.get_lr();
        assert!(decayed_lr < 0.1);

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.1);

        // Verify decay works again after reset
        scheduler.step();
        assert_relative_eq!(scheduler.get_lr(), 0.095, epsilon = 1e-6);
    }

    #[test]
    fn test_exponential_decay_slow_decay() {
        let mut scheduler = ExponentialDecay::new(0.1, 0.99);

        // With high gamma (0.99), decay is very slow
        for _ in 0..10 {
            scheduler.step();
        }
        let expected = 0.1 * 0.99_f32.powi(10);
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-6);
        assert!(scheduler.get_lr() > 0.09); // Should still be close to 0.1
    }

    #[test]
    fn test_exponential_decay_fast_decay() {
        let mut scheduler = ExponentialDecay::new(0.1, 0.5);

        // With low gamma (0.5), decay is fast
        for _ in 0..5 {
            scheduler.step();
        }
        let expected = 0.1 * 0.5_f32.powi(5);
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-6);
        assert!(scheduler.get_lr() < 0.01); // Should be much smaller
    }

    #[test]
    fn test_exponential_decay_zero_epoch() {
        let scheduler = ExponentialDecay::new(0.5, 0.95);
        // At epoch 0 (initial), LR should be initial_lr
        assert_eq!(scheduler.get_lr(), 0.5);
    }

    /// Verifies that two `ExponentialDecay` schedulers constructed with the same configuration produce identical learning rate sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut a = ExponentialDecay::new(0.1, 0.95);
    /// let mut b = ExponentialDecay::new(0.1, 0.95);
    /// for _ in 0..20 {
    ///     a.step();
    ///     b.step();
    ///     assert_relative_eq!(a.get_lr(), b.get_lr(), epsilon = 1e-6);
    /// }
    /// ```
    #[test]
    fn test_exponential_decay_consistency() {
        let mut scheduler1 = ExponentialDecay::new(0.1, 0.95);
        let mut scheduler2 = ExponentialDecay::new(0.1, 0.95);

        // Same configuration should produce same results
        for _ in 0..20 {
            scheduler1.step();
            scheduler2.step();
            assert_relative_eq!(scheduler1.get_lr(), scheduler2.get_lr(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_exponential_decay_many_epochs() {
        let mut scheduler = ExponentialDecay::new(1.0, 0.95);

        // Test that LR doesn't become negative or NaN
        for _ in 0..1000 {
            scheduler.step();
            let lr = scheduler.get_lr();
            assert!(lr > 0.0);
            assert!(lr.is_finite());
        }
    }
}

// ============================================================================
// CosineAnnealing Tests
// ============================================================================

#[cfg(test)]
mod cosine_annealing_tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_cosine_annealing_creation() {
        let scheduler = CosineAnnealing::new(0.1, 0.0, 10);
        assert_eq!(scheduler.get_lr(), 0.1);
    }

    #[test]
    fn test_cosine_annealing_initial_lr() {
        let scheduler = CosineAnnealing::new(0.01, 0.001, 50);
        assert_eq!(scheduler.get_lr(), 0.01);
    }

    #[test]
    fn test_cosine_annealing_first_step() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.0, 10);

        scheduler.step();
        // At epoch 1 of 10: progress = 0.1, cos(π * 0.1) ≈ 0.951
        let expected = 0.0 + 0.5 * (0.1 - 0.0) * (1.0 + (PI * 0.1).cos());
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-6);
        assert!(scheduler.get_lr() < 0.1); // Should decrease
        assert!(scheduler.get_lr() > 0.0);
    }

    #[test]
    fn test_cosine_annealing_halfway() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.0, 10);

        // At epoch 5 (halfway), cos(π * 0.5) = 0, so LR = eta_min + 0.5 * (initial_lr - eta_min)
        for _ in 0..5 {
            scheduler.step();
        }
        let expected = 0.0 + 0.5 * (0.1 - 0.0) * (1.0 + (PI * 0.5).cos());
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-6);
        assert_relative_eq!(scheduler.get_lr(), 0.05, epsilon = 1e-6); // Midpoint
    }

    #[test]
    fn test_cosine_annealing_end_of_cycle() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.01, 10);

        // At epoch 10 (end of cycle), cos(π) = -1, so LR = eta_min
        for _ in 0..10 {
            scheduler.step();
        }
        let expected = 0.01 + 0.5 * (0.1 - 0.01) * (1.0 + (PI * 1.0).cos());
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-5);
        assert_relative_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-5); // Close to eta_min
    }

    #[test]
    fn test_cosine_annealing_beyond_cycle() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.0, 10);

        // Beyond T_max, continue using the formula (will go below eta_min)
        for _ in 0..15 {
            scheduler.step();
        }
        // At epoch 15: progress = 1.5, cos(π * 1.5) ≈ 0
        let progress = 15.0 / 10.0;
        let expected = 0.0 + 0.5 * (0.1 - 0.0) * (1.0 + (PI * progress).cos());
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_annealing_with_min_lr() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.01, 10);

        // Verify LR stays between initial_lr and eta_min for first cycle
        for i in 0..10 {
            scheduler.step();
            let lr = scheduler.get_lr();
            assert!(lr <= 0.1, "LR {} > initial_lr at epoch {}", lr, i + 1);
            // Note: LR can go slightly below eta_min due to floating point precision
            assert!(
                lr >= 0.009,
                "LR {} significantly below eta_min at epoch {}",
                lr,
                i + 1
            );
        }
    }

    #[test]
    fn test_cosine_annealing_reset() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.01, 10);

        // Decay LR
        for _ in 0..10 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.1);

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.1);

        // Verify decay works again after reset
        scheduler.step();
        let expected = 0.01 + 0.5 * (0.1 - 0.01) * (1.0 + (PI * 0.1).cos());
        assert_relative_eq!(scheduler.get_lr(), expected, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_annealing_monotonic_decrease() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.0, 10);

        let mut prev_lr = scheduler.get_lr();

        // First half of cycle should monotonically decrease
        for _ in 0..5 {
            scheduler.step();
            let curr_lr = scheduler.get_lr();
            assert!(
                curr_lr < prev_lr,
                "LR should decrease monotonically in first half"
            );
            prev_lr = curr_lr;
        }
    }

    #[test]
    fn test_cosine_annealing_smooth_curve() {
        let mut scheduler = CosineAnnealing::new(1.0, 0.0, 100);

        // Collect LR values over the cycle
        let mut lrs = vec![scheduler.get_lr()];
        for _ in 0..100 {
            scheduler.step();
            lrs.push(scheduler.get_lr());
        }

        // Verify smooth decay (no sudden jumps)
        for i in 1..lrs.len() {
            let change = (lrs[i] - lrs[i - 1]).abs();
            assert!(
                change < 0.05,
                "Large LR jump detected: {} to {}",
                lrs[i - 1],
                lrs[i]
            );
        }
    }

    /// Verifies CosineAnnealing performs rapid decay for a short cycle (T_max = 2).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut scheduler = CosineAnnealing::new(0.1, 0.01, 2);
    /// scheduler.step(); // epoch 1
    /// assert!(scheduler.get_lr() > 0.01);
    /// scheduler.step(); // epoch 2
    /// assert_relative_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-5);
    /// ```
    #[test]
    fn test_cosine_annealing_short_cycle() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.01, 2);

        // With T_max=2, decay should be fast
        scheduler.step(); // epoch 1
        assert!(scheduler.get_lr() > 0.01);

        scheduler.step(); // epoch 2
        assert_relative_eq!(scheduler.get_lr(), 0.01, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_annealing_zero_epoch() {
        let scheduler = CosineAnnealing::new(0.5, 0.1, 20);
        // At epoch 0 (initial), LR should be initial_lr
        assert_eq!(scheduler.get_lr(), 0.5);
    }

    #[test]
    fn test_cosine_annealing_consistency() {
        let mut scheduler1 = CosineAnnealing::new(0.1, 0.01, 10);
        let mut scheduler2 = CosineAnnealing::new(0.1, 0.01, 10);

        // Same configuration should produce same results
        for _ in 0..20 {
            scheduler1.step();
            scheduler2.step();
            assert_relative_eq!(scheduler1.get_lr(), scheduler2.get_lr(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cosine_annealing_different_parameters() {
        let mut scheduler1 = CosineAnnealing::new(0.1, 0.01, 10);
        let mut scheduler2 = CosineAnnealing::new(0.1, 0.001, 10);
        let mut scheduler3 = CosineAnnealing::new(0.1, 0.01, 20);

        for _ in 0..5 {
            scheduler1.step();
            scheduler2.step();
            scheduler3.step();
        }

        // Different eta_min should give different LRs
        assert!(scheduler1.get_lr() != scheduler2.get_lr());

        // Different T_max should give different LRs at same epoch
        assert!(scheduler1.get_lr() != scheduler3.get_lr());
    }

    #[test]
    fn test_cosine_annealing_eta_min_equals_initial() {
        let mut scheduler = CosineAnnealing::new(0.1, 0.1, 10);

        // If eta_min == initial_lr, LR should stay constant
        for _ in 0..10 {
            scheduler.step();
            assert_relative_eq!(scheduler.get_lr(), 0.1, epsilon = 1e-6);
        }
    }
}

// ============================================================================
// Cross-Scheduler Comparison Tests
// ============================================================================

#[cfg(test)]
mod comparison_tests {
    use super::*;

    #[test]
    fn test_all_schedulers_start_at_initial_lr() {
        let step = StepDecay::new(0.1, 5, 0.5);
        let exp = ExponentialDecay::new(0.1, 0.95);
        let cos = CosineAnnealing::new(0.1, 0.0, 10);

        assert_eq!(step.get_lr(), 0.1);
        assert_eq!(exp.get_lr(), 0.1);
        assert_eq!(cos.get_lr(), 0.1);
    }

    /// Verifies that three scheduler implementations decrease their learning rates after several steps.
    ///
    /// Confirms that ExponentialDecay and CosineAnnealing produce strictly lower learning rates after a few steps,
    /// and that StepDecay does not increase (it may remain equal until a decay boundary).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut step = StepDecay::new(0.1, 2, 0.5);
    /// let mut exp = ExponentialDecay::new(0.1, 0.95);
    /// let mut cos = CosineAnnealing::new(0.1, 0.0, 10);
    ///
    /// let initial_step = step.get_lr();
    /// let initial_exp = exp.get_lr();
    /// let initial_cos = cos.get_lr();
    ///
    /// for _ in 0..5 {
    ///     step.step();
    ///     exp.step();
    ///     cos.step();
    /// }
    ///
    /// assert!(step.get_lr() <= initial_step);
    /// assert!(exp.get_lr() < initial_exp);
    /// assert!(cos.get_lr() < initial_cos);
    /// ```
    #[test]
    fn test_all_schedulers_decrease_lr() {
        let mut step = StepDecay::new(0.1, 2, 0.5);
        let mut exp = ExponentialDecay::new(0.1, 0.95);
        let mut cos = CosineAnnealing::new(0.1, 0.0, 10);

        let initial_step = step.get_lr();
        let initial_exp = exp.get_lr();
        let initial_cos = cos.get_lr();

        for _ in 0..5 {
            step.step();
            exp.step();
            cos.step();
        }

        // After some epochs, all should have decreased (or stayed same for step before boundary)
        assert!(step.get_lr() <= initial_step);
        assert!(exp.get_lr() < initial_exp);
        assert!(cos.get_lr() < initial_cos);
    }

    #[test]
    fn test_all_schedulers_reset_works() {
        let mut step = StepDecay::new(0.1, 2, 0.5);
        let mut exp = ExponentialDecay::new(0.1, 0.95);
        let mut cos = CosineAnnealing::new(0.1, 0.0, 10);

        // Run for some epochs
        for _ in 0..10 {
            step.step();
            exp.step();
            cos.step();
        }

        // Reset all
        step.reset();
        exp.reset();
        cos.reset();

        // All should return to initial LR
        assert_eq!(step.get_lr(), 0.1);
        assert_eq!(exp.get_lr(), 0.1);
        assert_eq!(cos.get_lr(), 0.1);
    }

    /// Ensures that a StepDecay configured with step_size = 1 produces the same learning-rate sequence as an ExponentialDecay with the same gamma.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut step = StepDecay::new(0.1, 1, 0.95);
    /// let mut exp = ExponentialDecay::new(0.1, 0.95);
    /// for _ in 0..10 {
    ///     step.step();
    ///     exp.step();
    ///     assert_relative_eq!(step.get_lr(), exp.get_lr(), epsilon = 1e-6);
    /// }
    /// ```
    #[test]
    fn test_exponential_vs_step_decay_pattern() {
        let mut step = StepDecay::new(0.1, 1, 0.95); // decay every epoch
        let mut exp = ExponentialDecay::new(0.1, 0.95); // decay every epoch

        // When step_size=1, StepDecay and ExponentialDecay should be identical
        for _ in 0..10 {
            step.step();
            exp.step();
            assert_relative_eq!(step.get_lr(), exp.get_lr(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_step_decay_holds_constant_vs_exponential_continuous() {
        let mut step = StepDecay::new(0.1, 5, 0.5);
        let mut exp = ExponentialDecay::new(0.1, 0.95);

        let step_initial = step.get_lr();

        // Before first step boundary, StepDecay stays constant
        for i in 0..4 {
            step.step();
            exp.step();

            assert_eq!(
                step.get_lr(),
                step_initial,
                "StepDecay changed before boundary at epoch {}",
                i + 1
            );
            assert!(
                exp.get_lr() < 0.1,
                "ExponentialDecay should continuously decrease"
            );
        }

        // At step boundary, StepDecay changes
        step.step();
        assert!(step.get_lr() < step_initial);
    }
}