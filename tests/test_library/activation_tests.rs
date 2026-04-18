use super::*;
use rust_neural_networks::utils::activations::{sigmoid, sigmoid_derivative};

// Sigmoid tests
#[test]
fn test_sigmoid_zero() {
    assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
}

#[test]
fn test_sigmoid_positive() {
    let result = sigmoid(2.0f32);
    assert!(result > 0.5 && result < 1.0);
    assert_relative_eq!(result, 0.880_797_1, epsilon = 1e-6);
}

#[test]
fn test_sigmoid_negative() {
    let result = sigmoid(-2.0f32);
    assert!(result > 0.0 && result < 0.5);
    assert_relative_eq!(result, 0.119_202_92, epsilon = 1e-6);
}

#[test]
fn test_sigmoid_large_positive() {
    let result = sigmoid(100.0);
    assert_relative_eq!(result, 1.0, epsilon = 1e-10);
}

#[test]
fn test_sigmoid_large_negative() {
    let result = sigmoid(-100.0f32);
    assert_relative_eq!(result, 0.0f32, epsilon = 1e-6);
}

#[test]
fn test_sigmoid_symmetry() {
    for i in 1..20 {
        let x = i as f32 * 0.5;
        assert_relative_eq!(sigmoid(x) + sigmoid(-x), 1.0f32, epsilon = 1e-6);
    }
}

#[test]
fn test_sigmoid_monotonic() {
    let mut prev = sigmoid(-10.0f32);
    for i in -100..100 {
        let x = i as f32 / 10.0;
        let curr = sigmoid(x);
        assert!(curr >= prev, "Sigmoid should be monotonically increasing");
        prev = curr;
    }
}

// Sigmoid derivative tests
#[test]
fn test_sigmoid_derivative_at_half() {
    assert_relative_eq!(sigmoid_derivative(0.5f32), 0.25f32, epsilon = 1e-6);
}

#[test]
fn test_sigmoid_derivative_at_extremes() {
    assert_relative_eq!(sigmoid_derivative(0.0f32), 0.0f32, epsilon = 1e-6);
    assert_relative_eq!(sigmoid_derivative(1.0f32), 0.0f32, epsilon = 1e-6);
}

#[test]
fn test_sigmoid_derivative_range() {
    for i in 0..=100 {
        let x = i as f32 / 100.0;
        let deriv = sigmoid_derivative(x);
        assert!((0.0..=0.25).contains(&deriv));
    }
}

#[test]
fn test_sigmoid_derivative_symmetry() {
    for i in 0..50 {
        let x = i as f32 / 100.0;
        assert_relative_eq!(
            sigmoid_derivative(x),
            sigmoid_derivative(1.0 - x),
            epsilon = 1e-6
        );
    }
}

// ReLU tests
#[test]
fn test_relu_negative_values() {
    let mut data = vec![-5.0, -3.0, -1.0, -0.1, -0.001];
    relu_inplace(&mut data);
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_relu_zero() {
    let mut data = vec![0.0];
    relu_inplace(&mut data);
    assert_eq!(data[0], 0.0);
}

#[test]
fn test_relu_positive_values() {
    let original = vec![0.001, 0.1, 1.0, 5.0, 100.0];
    let mut data = original.clone();
    relu_inplace(&mut data);
    assert_eq!(data, original);
}

#[test]
fn test_relu_mixed() {
    let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    relu_inplace(&mut data);
    assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn test_relu_empty() {
    let mut data: Vec<f32> = vec![];
    relu_inplace(&mut data);
    assert!(data.is_empty());
}

#[test]
fn test_relu_large_array() {
    let mut data: Vec<f32> = (-500..500).map(|x| x as f32).collect();
    relu_inplace(&mut data);

    for (i, &val) in data.iter().enumerate() {
        let original = (i as i32 - 500) as f32;
        if original <= 0.0 {
            assert_eq!(val, 0.0);
        } else {
            assert_eq!(val, original);
        }
    }
}

// Softmax tests
#[test]
fn test_softmax_sum_to_one() {
    let mut data = vec![1.0, 2.0, 3.0];
    softmax_rows(&mut data, 1, 3);
    let sum: f32 = data.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
}

#[test]
fn test_softmax_all_positive() {
    let mut data = vec![1.0, 2.0, 3.0];
    softmax_rows(&mut data, 1, 3);
    assert!(data.iter().all(|&x| x > 0.0 && x < 1.0));
}

#[test]
fn test_softmax_ordering_preserved() {
    let mut data = vec![1.0, 2.0, 3.0];
    softmax_rows(&mut data, 1, 3);
    assert!(data[0] < data[1] && data[1] < data[2]);
}

#[test]
fn test_softmax_uniform_input() {
    let mut data = vec![1.0, 1.0, 1.0, 1.0];
    softmax_rows(&mut data, 1, 4);
    for &val in &data {
        assert_relative_eq!(val, 0.25, epsilon = 1e-6);
    }
}

#[test]
fn test_softmax_multiple_rows() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    softmax_rows(&mut data, 3, 3);

    // Each row should sum to 1
    for row in data.chunks(3) {
        let sum: f32 = row.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_softmax_numerical_stability_large() {
    let mut data = vec![1000.0, 1001.0, 1002.0];
    softmax_rows(&mut data, 1, 3);

    let sum: f32 = data.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
}

#[test]
fn test_softmax_numerical_stability_negative_large() {
    let mut data = vec![-1000.0, -1001.0, -1002.0];
    softmax_rows(&mut data, 1, 3);

    let sum: f32 = data.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    assert!(!data.iter().any(|&x| x.is_nan() || x.is_infinite()));
}

#[test]
fn test_softmax_single_element() {
    let mut data = vec![5.0];
    softmax_rows(&mut data, 1, 1);
    assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
}

#[test]
fn test_softmax_two_elements() {
    let mut data = vec![0.0, 0.0];
    softmax_rows(&mut data, 1, 2);
    assert_relative_eq!(data[0], 0.5, epsilon = 1e-6);
    assert_relative_eq!(data[1], 0.5, epsilon = 1e-6);
}
