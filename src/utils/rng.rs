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
    /// Create a new RNG with explicit seed (if zero, use a fixed value).
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state }
    }

    /// Reseed based on the current time.
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

    /// Basic xorshift to generate u32.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    /// Convert to [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    /// Uniform sample in [low, high).
    pub fn gen_range_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Integer sample in [0, upper).
    pub fn gen_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u32() as usize) % upper
        }
    }

    /// Fisher-Yates shuffle for usize slices.
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
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_rng_gen_range_f32() {
        let mut rng = SimpleRng::new(67890);

        for _ in 0..1000 {
            let val = rng.gen_range_f32(-1.0, 1.0);
            assert!(val >= -1.0 && val < 1.0);
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
}
