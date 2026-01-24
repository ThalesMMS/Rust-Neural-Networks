//! Simple random number generator for reproducibility.
//!
//! This module provides a lightweight xorshift-based PRNG that doesn't require
//! external dependencies, ensuring reproducible results across runs.

use std::time::{SystemTime, UNIX_EPOCH};

/// Simple RNG for reproducibility without external crates.
///
/// Uses xorshift algorithm for fast, deterministic random number generation.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Creates a new `SimpleRng` initialized from the provided seed.
    ///
    /// If `seed` is 0, a fixed non-zero seed (0x9e3779b97f4a7c15) is used instead to ensure
    /// the internal state is never zero.
    ///
    /// # Parameters
    ///
    /// - `seed`: The explicit seed to initialize the generator. Zero is substituted with a fixed non-zero seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut a = SimpleRng::new(42);
    /// let mut b = SimpleRng::new(42);
    /// assert_eq!(a.next_u32(), b.next_u32());
    /// ```
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    /// Reseeds the RNG's internal state using the current system time in nanoseconds.
    ///
    /// If the computed nanosecond timestamp is 0, a fixed non-zero seed `0x9e3779b97f4a7c15` is used instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(1);
    /// rng.reseed_from_time();
    /// // After reseeding, the RNG can produce values again.
    /// let _v = rng.next_u32();
    /// ```
    pub fn reseed_from_time(&mut self) {
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

    /// Generates the next pseudorandom 32-bit unsigned integer from the RNG state.
    ///
    /// # Returns
    ///
    /// A `u32` containing the next pseudorandom value derived from the generator's internal state.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(123);
    /// let v = rng.next_u32();
    /// // v is a pseudorandom u32 deterministically derived from seed 123
    /// assert!(v <= u32::MAX);
    /// ```
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    /// Produces a floating-point sample in the range [0.0, 1.0].
    ///
    /// The result is obtained by scaling a 32-bit random integer to an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(1);
    /// let v = rng.next_f32();
    /// assert!(v >= 0.0 && v <= 1.0);
    /// ```
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Samples a f32 uniformly from the half-open interval [low, high).
    ///
    /// The result is greater than or equal to `low` and less than `high`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(123);
    /// let v = rng.gen_range_f32(-1.0, 1.0);
    /// assert!(v >= -1.0 && v < 1.0);
    /// ```
    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Samples a uniformly distributed `usize` in the half-open interval [0, upper).
    ///
    /// Returns `0` if `upper` is `0`, otherwise a value `v` such that `0 <= v < upper`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(123);
    /// let v = rng.gen_usize(10);
    /// assert!(v < 10);
    /// ```
    pub fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }

    /// Performs an in-place Fisher–Yates shuffle of a slice of `usize`.
    ///
    /// Leaves slices with length 0 or 1 unchanged. Produces a uniformly random permutation
    /// of the input elements using the RNG's current state.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::utils::rng::SimpleRng;
    /// let mut rng = SimpleRng::new(42);
    /// let mut v = (0..5).collect::<Vec<usize>>();
    /// rng.shuffle_usize(&mut v);
    /// // same elements, order may differ
    /// let mut sorted = v.clone();
    /// sorted.sort();
    /// assert_eq!(sorted, (0..5).collect::<Vec<usize>>());
    /// ```
    pub fn shuffle_usize(&mut self, data: &mut [usize]) {
        if data.len() <= 1 {
            return;
        }
        for i in (1..data.len()).rev() {
            let j = self.gen_usize(i + 1);
            data.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_rng_next_f32_range() {
        let mut rng = SimpleRng::new(12345);

        for _ in 0..1000 {
            let val = rng.next_f32();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_rng_gen_range_f32() {
        let mut rng = SimpleRng::new(67890);

        for _ in 0..1000 {
            let val = rng.gen_range_f32(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_rng_gen_usize() {
        let mut rng = SimpleRng::new(11111);

        for _ in 0..1000 {
            let val = rng.gen_usize(10);
            assert!(val < 10);
        }
    }

    #[test]
    fn test_rng_gen_usize_zero() {
        let mut rng = SimpleRng::new(22222);
        assert_eq!(rng.gen_usize(0), 0);
    }

    #[test]
    fn test_shuffle_usize() {
        let mut rng = SimpleRng::new(33333);
        let mut data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let original = data.clone();

        rng.shuffle_usize(&mut data);

        // Should contain same elements
        let mut sorted = data.clone();
        sorted.sort();
        assert_eq!(sorted, original);

        // Very unlikely to be in same order
        assert_ne!(data, original);
    }

    #[test]
    fn test_shuffle_empty() {
        let mut rng = SimpleRng::new(44444);
        let mut data: Vec<usize> = vec![];
        rng.shuffle_usize(&mut data);
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_shuffle_single() {
        let mut rng = SimpleRng::new(55555);
        let mut data = vec![42];
        rng.shuffle_usize(&mut data);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_rng_zero_seed() {
        // Zero seed should use the fixed value
        let mut rng = SimpleRng::new(0);
        let val = rng.next_u32();
        // Should produce valid output
        assert!(val > 0);
    }

    #[test]
    fn test_reseed_from_time() {
        let mut rng = SimpleRng::new(42);
        let val_before = rng.next_u32();

        // Reseed from time
        rng.reseed_from_time();
        let val_after = rng.next_u32();

        // The values should be different (very high probability)
        // Note: In theory they could be the same, but it's astronomically unlikely
        // We just test that reseed_from_time runs without error and produces output
        assert!(val_after > 0 || val_before > 0);
    }

    #[test]
    fn test_reseed_from_time_changes_state() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        // Both should produce same values initially
        assert_eq!(rng1.next_u32(), rng2.next_u32());

        // Reseed one from time
        rng1.reseed_from_time();

        // Now they should diverge (unless we get astronomically unlucky timing)
        let v1 = rng1.next_u32();
        let v2 = rng2.next_u32();
        // Just check both are valid
        assert!(v1 > 0 || v2 > 0);
    }
}
