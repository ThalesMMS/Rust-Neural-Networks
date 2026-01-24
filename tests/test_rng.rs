// Tests for RNG reproducibility and distribution.
// SimpleRng implementation is copied from the main binaries for testing purposes.

use std::time::{SystemTime, UNIX_EPOCH};

// Simple RNG to avoid external dependencies (not cryptographic).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    // Create RNG with explicit seed.
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    // Reseed using the current time.
    fn reseed_from_time(&mut self) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.state = if nanos == 0 {
            0x9e3779b97f4a7c15
        } else {
            nanos
        };
    }

    // Generate a pseudo-random u32 (xorshift).
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    // Convert u32 to [0, 1) for f64.
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }

    // Convert u32 to [0, 1) for f32.
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    // Uniform sample in [low, high) for f64.
    fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    // Uniform sample in [low, high) for f32.
    fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    // Integer sample in [0, upper).
    fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reproducibility tests.
    #[test]
    fn test_rng_same_seed_produces_same_sequence() {
        let seed = 12345u64;
        let mut rng1 = SimpleRng::new(seed);
        let mut rng2 = SimpleRng::new(seed);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_rng_different_seeds_produce_different_sequences() {
        let mut rng1 = SimpleRng::new(12345);
        let mut rng2 = SimpleRng::new(54321);

        let mut different = false;
        for _ in 0..10 {
            if rng1.next_u32() != rng2.next_u32() {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_rng_zero_seed_uses_default() {
        let mut rng1 = SimpleRng::new(0);
        let mut rng2 = SimpleRng::new(0);

        // Both should use the same default seed
        for _ in 0..10 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_rng_state_advances_deterministically() {
        let seed = 42u64;
        let mut rng = SimpleRng::new(seed);

        let first = rng.next_u32();
        let second = rng.next_u32();
        let third = rng.next_u32();

        // Reset and verify same sequence
        let mut rng_reset = SimpleRng::new(seed);
        assert_eq!(rng_reset.next_u32(), first);
        assert_eq!(rng_reset.next_u32(), second);
        assert_eq!(rng_reset.next_u32(), third);
    }

    #[test]
    fn test_rng_reseed_changes_sequence() {
        let mut rng = SimpleRng::new(12345);
        let first = rng.next_u32();

        // Reseed with different value
        rng.state = 54321;
        let second = rng.next_u32();

        assert_ne!(first, second);
    }

    // Distribution tests for next_f64.
    #[test]
    fn test_next_f64_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let value = rng.next_f64();
            assert!(value >= 0.0, "Value should be >= 0.0");
            assert!(value < 1.0, "Value should be < 1.0");
        }
    }

    #[test]
    fn test_next_f64_distribution() {
        let mut rng = SimpleRng::new(42);
        let n = 10000;
        let mut count_low = 0;

        for _ in 0..n {
            let value = rng.next_f64();
            if value < 0.5 {
                count_low += 1;
            }
        }

        // With uniform distribution, expect roughly 50/50 split
        // Allow 5% deviation
        let low_ratio = count_low as f64 / n as f64;
        assert!(
            low_ratio > 0.45 && low_ratio < 0.55,
            "Low ratio {} should be close to 0.5",
            low_ratio
        );
    }

    #[test]
    fn test_next_f64_not_constant() {
        let mut rng = SimpleRng::new(42);
        let first = rng.next_f64();
        let mut all_same = true;

        for _ in 0..100 {
            if rng.next_f64() != first {
                all_same = false;
                break;
            }
        }

        assert!(!all_same, "RNG should produce varying values");
    }

    // Distribution tests for next_f32.
    #[test]
    fn test_next_f32_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let value = rng.next_f32();
            assert!(value >= 0.0, "Value should be >= 0.0");
            assert!(value < 1.0, "Value should be < 1.0");
        }
    }

    #[test]
    fn test_next_f32_distribution() {
        let mut rng = SimpleRng::new(42);
        let n = 10000;
        let mut count_low = 0;

        for _ in 0..n {
            let value = rng.next_f32();
            if value < 0.5 {
                count_low += 1;
            }
        }

        // With uniform distribution, expect roughly 50/50 split
        // Allow 5% deviation
        let low_ratio = count_low as f32 / n as f32;
        assert!(
            low_ratio > 0.45 && low_ratio < 0.55,
            "Low ratio {} should be close to 0.5",
            low_ratio
        );
    }

    // Distribution tests for gen_range_f64.
    #[test]
    fn test_gen_range_f64_within_bounds() {
        let mut rng = SimpleRng::new(42);
        let low = -5.0;
        let high = 10.0;

        for _ in 0..1000 {
            let value = rng.gen_range_f64(low, high);
            assert!(
                value >= low && value < high,
                "Value {} should be in [{}, {})",
                value,
                low,
                high
            );
        }
    }

    #[test]
    fn test_gen_range_f64_positive_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let value = rng.gen_range_f64(0.0, 1.0);
            assert!(value >= 0.0 && value < 1.0);
        }
    }

    #[test]
    fn test_gen_range_f64_negative_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let value = rng.gen_range_f64(-1.0, 0.0);
            assert!(value >= -1.0 && value < 0.0);
        }
    }

    #[test]
    fn test_gen_range_f64_symmetric_range() {
        let mut rng = SimpleRng::new(42);
        let n = 10000;
        let mut count_negative = 0;

        for _ in 0..n {
            let value = rng.gen_range_f64(-0.5, 0.5);
            if value < 0.0 {
                count_negative += 1;
            }
        }

        // Expect roughly 50/50 split, allow 5% deviation
        let neg_ratio = count_negative as f64 / n as f64;
        assert!(
            neg_ratio > 0.45 && neg_ratio < 0.55,
            "Negative ratio {} should be close to 0.5",
            neg_ratio
        );
    }

    #[test]
    fn test_gen_range_f64_large_range() {
        let mut rng = SimpleRng::new(42);
        let low = -1000.0;
        let high = 1000.0;

        for _ in 0..1000 {
            let value = rng.gen_range_f64(low, high);
            assert!(
                value >= low && value < high,
                "Value {} should be in [{}, {})",
                value,
                low,
                high
            );
        }
    }

    #[test]
    fn test_gen_range_f64_small_range() {
        let mut rng = SimpleRng::new(42);
        let low = 0.0;
        let high = 0.001;

        for _ in 0..100 {
            let value = rng.gen_range_f64(low, high);
            assert!(
                value >= low && value < high,
                "Value {} should be in [{}, {})",
                value,
                low,
                high
            );
        }
    }

    // Distribution tests for gen_range_f32.
    #[test]
    fn test_gen_range_f32_within_bounds() {
        let mut rng = SimpleRng::new(42);
        let low = -5.0f32;
        let high = 10.0f32;

        for _ in 0..1000 {
            let value = rng.gen_range_f32(low, high);
            assert!(
                value >= low && value < high,
                "Value {} should be in [{}, {})",
                value,
                low,
                high
            );
        }
    }

    #[test]
    fn test_gen_range_f32_symmetric_range() {
        let mut rng = SimpleRng::new(42);
        let n = 10000;
        let mut count_negative = 0;

        for _ in 0..n {
            let value = rng.gen_range_f32(-0.5, 0.5);
            if value < 0.0 {
                count_negative += 1;
            }
        }

        // Expect roughly 50/50 split, allow 5% deviation
        let neg_ratio = count_negative as f32 / n as f32;
        assert!(
            neg_ratio > 0.45 && neg_ratio < 0.55,
            "Negative ratio {} should be close to 0.5",
            neg_ratio
        );
    }

    // Distribution tests for gen_usize.
    #[test]
    fn test_gen_usize_zero_upper_bound() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..10 {
            assert_eq!(rng.gen_usize(0), 0);
        }
    }

    #[test]
    fn test_gen_usize_within_bounds() {
        let mut rng = SimpleRng::new(42);
        let upper = 10;

        for _ in 0..1000 {
            let value = rng.gen_usize(upper);
            assert!(value < upper, "Value {} should be < {}", value, upper);
        }
    }

    #[test]
    fn test_gen_usize_distribution() {
        let mut rng = SimpleRng::new(42);
        let upper = 4;
        let n = 10000;
        let mut counts = vec![0; upper];

        for _ in 0..n {
            let value = rng.gen_usize(upper);
            counts[value] += 1;
        }

        // Each bucket should have roughly n/upper elements
        let expected = n / upper;
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / expected as f64;
            assert!(
                ratio > 0.8 && ratio < 1.2,
                "Bucket {} has ratio {} (count: {}, expected: ~{})",
                i,
                ratio,
                count,
                expected
            );
        }
    }

    #[test]
    fn test_gen_usize_single_value() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..10 {
            assert_eq!(rng.gen_usize(1), 0);
        }
    }

    #[test]
    fn test_gen_usize_large_upper_bound() {
        let mut rng = SimpleRng::new(42);
        let upper = 1000000;

        for _ in 0..100 {
            let value = rng.gen_usize(upper);
            assert!(value < upper);
        }
    }

    // Statistical properties tests.
    #[test]
    fn test_rng_mean_convergence_f64() {
        let mut rng = SimpleRng::new(42);
        let n = 10000;
        let mut sum = 0.0;

        for _ in 0..n {
            sum += rng.next_f64();
        }

        let mean = sum / n as f64;
        // For uniform [0, 1), expected mean is 0.5
        assert!(
            mean > 0.48 && mean < 0.52,
            "Mean {} should be close to 0.5",
            mean
        );
    }

    #[test]
    fn test_rng_quartiles_f64() {
        let mut rng = SimpleRng::new(42);
        let n = 10000;
        let mut q1 = 0;
        let mut q2 = 0;
        let mut q3 = 0;
        let mut q4 = 0;

        for _ in 0..n {
            let value = rng.next_f64();
            if value < 0.25 {
                q1 += 1;
            } else if value < 0.5 {
                q2 += 1;
            } else if value < 0.75 {
                q3 += 1;
            } else {
                q4 += 1;
            }
        }

        // Each quartile should have ~25% of values
        for (i, count) in [q1, q2, q3, q4].iter().enumerate() {
            let ratio = *count as f64 / n as f64;
            assert!(
                ratio > 0.23 && ratio < 0.27,
                "Quartile {} has ratio {} (should be ~0.25)",
                i + 1,
                ratio
            );
        }
    }

    #[test]
    fn test_rng_independence() {
        let mut rng = SimpleRng::new(42);
        let n = 1000;
        let mut consecutive_increases = 0;

        let mut prev = rng.next_f64();
        for _ in 0..n {
            let curr = rng.next_f64();
            if curr > prev {
                consecutive_increases += 1;
            }
            prev = curr;
        }

        // With independent samples, expect roughly 50% increases
        let ratio = consecutive_increases as f64 / n as f64;
        assert!(
            ratio > 0.45 && ratio < 0.55,
            "Increase ratio {} should be close to 0.5 (independence test)",
            ratio
        );
    }

    // Edge case tests.
    #[test]
    fn test_rng_max_seed() {
        let mut rng = SimpleRng::new(u64::MAX);
        // Should not panic and should produce values
        for _ in 0..10 {
            let value = rng.next_f64();
            assert!(value >= 0.0 && value < 1.0);
        }
    }

    #[test]
    fn test_rng_power_of_two_seed() {
        let seeds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        for &seed in &seeds {
            let mut rng = SimpleRng::new(seed);
            // Should produce valid values
            for _ in 0..10 {
                let value = rng.next_f64();
                assert!(value >= 0.0 && value < 1.0);
            }
        }
    }

    #[test]
    fn test_reseed_from_time_changes_state() {
        let mut rng = SimpleRng::new(42);
        let first = rng.next_u32();

        // Sleep briefly to ensure time changes
        std::thread::sleep(std::time::Duration::from_millis(1));

        rng.reseed_from_time();
        let second = rng.next_u32();

        // After reseeding from time, sequence should be different
        // Note: There's a tiny chance they could be equal, but extremely unlikely
        assert_ne!(
            first, second,
            "Reseeding from time should change the sequence"
        );
    }
}
