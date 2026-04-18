use super::*;

// ============================================================================
// Tensor construction tests  (subtask-1-1)
// ============================================================================

#[test]
fn tensor_construction_new() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
    assert_eq!(t.shape(), (2, 2));
    assert_eq!(t.data(), vec![1.0f32, 2.0, 3.0, 4.0]);
    assert!(!t.requires_grad());
    assert!(t.grad().is_none());
    assert_eq!(t.numel(), 4);
}

#[test]
fn tensor_construction_zeros() {
    let t = Tensor::zeros((3, 4));
    assert_eq!(t.shape(), (3, 4));
    assert_eq!(t.numel(), 12);
    assert_eq!(t.data(), vec![0.0f32; 12]);
    assert!(!t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn tensor_construction_from_vec_no_grad() {
    let t = Tensor::from_vec(vec![0.5f32, -0.3, 0.1], (1, 3), false);
    assert_eq!(t.shape(), (1, 3));
    assert!(!t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn tensor_construction_from_vec_with_grad() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), true);
    assert_eq!(t.shape(), (1, 3));
    assert!(t.requires_grad());
    assert!(t.grad().is_none());
}

#[test]
fn tensor_construction_accumulate_grad() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), true);
    assert!(t.grad().is_none());

    t.accumulate_grad(&[0.1, 0.2, 0.3]);
    let g = t
        .grad()
        .expect("gradient should be Some after first accumulate_grad");
    assert!((g[0] - 0.1).abs() < 1e-6);
    assert!((g[1] - 0.2).abs() < 1e-6);
    assert!((g[2] - 0.3).abs() < 1e-6);

    // Second accumulate adds to the existing gradient
    t.accumulate_grad(&[0.1, 0.2, 0.3]);
    let g2 = t.grad().unwrap();
    assert!((g2[0] - 0.2).abs() < 1e-6);
    assert!((g2[1] - 0.4).abs() < 1e-6);
    assert!((g2[2] - 0.6).abs() < 1e-6);
}

#[test]
fn tensor_construction_zero_grad() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), true);
    t.accumulate_grad(&[5.0, 6.0]);
    t.zero_grad();
    let g = t.grad().expect("grad should still be Some after zero_grad");
    assert_eq!(g, vec![0.0f32, 0.0]);
}

#[test]
fn tensor_construction_clone_shares_inner() {
    let a = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), true);
    let b = a.clone();
    // Accumulate via `a`; `b` shares the same inner, so must also see it.
    a.accumulate_grad(&[1.0, 1.0]);
    let g = b.grad().expect("b should share gradient with a");
    assert!((g[0] - 1.0).abs() < 1e-6);
    assert!((g[1] - 1.0).abs() < 1e-6);
}

#[test]
fn tensor_construction_row_vector() {
    let t = Tensor::new(vec![10.0, 20.0, 30.0], (1, 3));
    assert_eq!(t.shape(), (1, 3));
    assert_eq!(t.numel(), 3);
}

#[test]
fn tensor_construction_column_vector() {
    let t = Tensor::new(vec![10.0, 20.0, 30.0], (3, 1));
    assert_eq!(t.shape(), (3, 1));
    assert_eq!(t.numel(), 3);
}

#[test]
fn tensor_construction_scalar() {
    let t = Tensor::new(vec![42.0], (1, 1));
    assert_eq!(t.shape(), (1, 1));
    assert_eq!(t.numel(), 1);
    assert_eq!(t.data(), vec![42.0f32]);
}
