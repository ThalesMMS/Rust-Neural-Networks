use super::*;

#[test]
fn test_rng_deterministic() {
    let mut rng1 = SimpleRng::new(42);
    let mut rng2 = SimpleRng::new(42);

    for _ in 0..1000 {
        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }
}

#[test]
fn test_rng_different_seeds() {
    let mut rng1 = SimpleRng::new(1);
    let mut rng2 = SimpleRng::new(2);

    // Very unlikely to match
    let vals1: Vec<u32> = (0..10).map(|_| rng1.next_u32()).collect();
    let vals2: Vec<u32> = (0..10).map(|_| rng2.next_u32()).collect();
    assert_ne!(vals1, vals2);
}

#[test]
fn test_rng_zero_seed() {
    let mut rng = SimpleRng::new(0);
    // Should still work with default seed
    let val = rng.next_u32();
    assert!(val > 0);
}

#[test]
fn test_rng_next_f32_range() {
    let mut rng = SimpleRng::new(12345);
    for _ in 0..10000 {
        let val = rng.next_f32();
        assert!((0.0..1.0).contains(&val));
    }
}

#[test]
fn test_rng_gen_range_f32() {
    let mut rng = SimpleRng::new(67890);
    for _ in 0..10000 {
        let val = rng.gen_range_f32(-5.0, 5.0);
        assert!((-5.0..5.0).contains(&val));
    }
}

#[test]
fn test_rng_gen_range_f32_narrow() {
    let mut rng = SimpleRng::new(11111);
    for _ in 0..1000 {
        let val = rng.gen_range_f32(0.999, 1.0);
        assert!((0.999..1.0).contains(&val));
    }
}

#[test]
fn test_rng_gen_usize() {
    let mut rng = SimpleRng::new(22222);
    for _ in 0..10000 {
        let val = rng.gen_usize(100);
        assert!(val < 100);
    }
}

#[test]
fn test_rng_gen_usize_one() {
    let mut rng = SimpleRng::new(33333);
    for _ in 0..100 {
        let val = rng.gen_usize(1);
        assert_eq!(val, 0);
    }
}

#[test]
fn test_rng_gen_usize_zero() {
    let mut rng = SimpleRng::new(44444);
    let val = rng.gen_usize(0);
    assert_eq!(val, 0);
}

#[test]
fn test_rng_shuffle_preserves_elements() {
    let mut rng = SimpleRng::new(55555);
    let mut data: Vec<usize> = (0..100).collect();
    let original: Vec<usize> = data.clone();

    rng.shuffle_usize(&mut data);

    // Same elements, different order
    let mut sorted = data.clone();
    sorted.sort();
    assert_eq!(sorted, original);
    assert_ne!(data, original); // Very unlikely to be same order
}

#[test]
fn test_rng_shuffle_empty() {
    let mut rng = SimpleRng::new(66666);
    let mut data: Vec<usize> = vec![];
    rng.shuffle_usize(&mut data);
    assert!(data.is_empty());
}

#[test]
fn test_rng_shuffle_single() {
    let mut rng = SimpleRng::new(77777);
    let mut data = vec![42];
    rng.shuffle_usize(&mut data);
    assert_eq!(data, vec![42]);
}

#[test]
fn test_rng_shuffle_two() {
    let mut swapped = false;

    // Run many times, should swap at least once
    for seed in 0..100 {
        let mut rng = SimpleRng::new(seed);
        let mut data = vec![0, 1];
        rng.shuffle_usize(&mut data);
        if data == vec![1, 0] {
            swapped = true;
            break;
        }
    }
    assert!(swapped, "Shuffle should swap elements sometimes");
}

#[test]
fn test_rng_reseed_from_time() {
    let mut rng = SimpleRng::new(42);
    let val1 = rng.next_u32();

    rng.reseed_from_time();
    let val2 = rng.next_u32();

    // Both should be valid
    assert!(val1 > 0 || val2 > 0);
}

#[test]
fn test_rng_distribution_rough() {
    let mut rng = SimpleRng::new(99999);
    let mut buckets = [0u32; 10];

    for _ in 0..100000 {
        let val = rng.gen_usize(10);
        buckets[val] += 1;
    }

    // Each bucket should have roughly 10000 values (±20%)
    for &count in &buckets {
        assert!(
            count > 8000 && count < 12000,
            "Distribution seems biased: {}",
            count
        );
    }
}
