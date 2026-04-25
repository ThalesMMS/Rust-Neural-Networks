//! Production persistence format tests.

use rust_neural_networks::layers::{
    BatchNormLayer, Conv2DLayer, DenseLayer, DropoutLayer, GlobalAvgPoolLayer, Layer, ResidualBlock,
};
use rust_neural_networks::persistence::{load_layers, save_layers, LayerTypeId};
use rust_neural_networks::utils::rng::SimpleRng;
use std::io::Cursor;

#[test]
fn test_dense_checkpoint_byte_layout() {
    let dense = DenseLayer::new_with_weights(2, 1, vec![1.5, -2.0], vec![0.25]);
    let layers: Vec<Box<dyn Layer>> = vec![Box::new(dense)];

    let mut data = Vec::new();
    save_layers(&mut data, &layers).unwrap();

    let mut expected = Vec::new();
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.push(u8::from(LayerTypeId::Dense));
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(&1.5f32.to_le_bytes());
    expected.extend_from_slice(&(-2.0f32).to_le_bytes());
    expected.extend_from_slice(&0.25f32.to_le_bytes());

    assert_eq!(data, expected);
}

#[test]
fn test_layer_stack_roundtrip_all_supported_types() {
    let dense_weights = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6];
    let dense_biases = vec![0.7, -0.8];
    let dense = DenseLayer::new_with_weights(3, 2, dense_weights.clone(), dense_biases.clone());

    let conv_weights = vec![0.01, 0.02, -0.03, 0.04, 0.05, -0.06, 0.07, 0.08];
    let conv_biases = vec![0.09, -0.1];
    let conv = Conv2DLayer::new_with_weights(
        1,
        2,
        2,
        0,
        1,
        4,
        4,
        conv_weights.clone(),
        conv_biases.clone(),
    );

    let bn_gamma = vec![1.0, 1.1, 1.2];
    let bn_beta = vec![0.0, -0.1, 0.2];
    let bn_running_mean = vec![0.3, 0.4, 0.5];
    let bn_running_var = vec![0.6, 0.7, 0.8];
    let bn_epsilon = 1e-3;
    let bn_momentum = 0.42;
    let batchnorm = BatchNormLayer::new_with_params(
        3,
        bn_epsilon,
        bn_momentum,
        bn_gamma.clone(),
        bn_beta.clone(),
        bn_running_mean.clone(),
        bn_running_var.clone(),
    );

    let mut rng = SimpleRng::new(123);
    let dropout = DropoutLayer::new(3, 0.25, &mut rng);
    let global_avgpool = GlobalAvgPoolLayer::new(4, 4, 2);
    let residual = ResidualBlock::new(2, 4, 2, 8, 8, &mut rng);

    let residual_in_channels = residual.in_channels();
    let residual_out_channels = residual.out_channels();
    let residual_out_height = residual.out_height();
    let residual_out_width = residual.out_width();
    let residual_has_projection = residual.has_projection_shortcut();

    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(dense),
        Box::new(conv),
        Box::new(batchnorm),
        Box::new(dropout),
        Box::new(global_avgpool),
        Box::new(residual),
    ];

    let mut data = Vec::new();
    save_layers(&mut data, &layers).unwrap();

    let mut cursor = Cursor::new(data);
    let loaded = load_layers(&mut cursor, &mut rng).unwrap();

    assert_eq!(loaded.len(), 6);

    let loaded_dense = loaded[0]
        .as_any()
        .downcast_ref::<DenseLayer>()
        .expect("layer 0 should be DenseLayer");
    assert_eq!(loaded_dense.input_size(), 3);
    assert_eq!(loaded_dense.output_size(), 2);
    assert_eq!(loaded_dense.weights(), dense_weights.as_slice());
    assert_eq!(loaded_dense.biases(), dense_biases.as_slice());

    let loaded_conv = loaded[1]
        .as_any()
        .downcast_ref::<Conv2DLayer>()
        .expect("layer 1 should be Conv2DLayer");
    assert_eq!(loaded_conv.in_channels(), 1);
    assert_eq!(loaded_conv.out_channels(), 2);
    assert_eq!(loaded_conv.kernel_size(), 2);
    assert_eq!(loaded_conv.padding(), 0);
    assert_eq!(loaded_conv.stride(), 1);
    assert_eq!(loaded_conv.input_height(), 4);
    assert_eq!(loaded_conv.input_width(), 4);
    assert_eq!(loaded_conv.weights(), conv_weights.as_slice());
    assert_eq!(loaded_conv.biases(), conv_biases.as_slice());

    let loaded_bn = loaded[2]
        .as_any()
        .downcast_ref::<BatchNormLayer>()
        .expect("layer 2 should be BatchNormLayer");
    assert_eq!(loaded_bn.output_size(), 3);
    assert_eq!(loaded_bn.gamma(), bn_gamma.as_slice());
    assert_eq!(loaded_bn.beta(), bn_beta.as_slice());
    assert_eq!(loaded_bn.running_mean(), bn_running_mean);
    assert_eq!(loaded_bn.running_var(), bn_running_var);
    assert_eq!(loaded_bn.epsilon(), bn_epsilon);
    assert_eq!(loaded_bn.momentum(), bn_momentum);

    let loaded_dropout = loaded[3]
        .as_any()
        .downcast_ref::<DropoutLayer>()
        .expect("layer 3 should be DropoutLayer");
    assert_eq!(loaded_dropout.output_size(), 3);
    assert!((loaded_dropout.drop_rate() - 0.25).abs() < f32::EPSILON);

    let loaded_gap = loaded[4]
        .as_any()
        .downcast_ref::<GlobalAvgPoolLayer>()
        .expect("layer 4 should be GlobalAvgPoolLayer");
    assert_eq!(loaded_gap.in_height(), 4);
    assert_eq!(loaded_gap.in_width(), 4);
    assert_eq!(loaded_gap.channels(), 2);

    let loaded_residual = loaded[5]
        .as_any()
        .downcast_ref::<ResidualBlock>()
        .expect("layer 5 should be ResidualBlock");
    assert_eq!(loaded_residual.in_channels(), residual_in_channels);
    assert_eq!(loaded_residual.out_channels(), residual_out_channels);
    assert_eq!(loaded_residual.out_height(), residual_out_height);
    assert_eq!(loaded_residual.out_width(), residual_out_width);
    assert_eq!(
        loaded_residual.has_projection_shortcut(),
        residual_has_projection
    );
    // ResidualBlock internal weights are intentionally not preserved because the
    // current layer API only exposes architecture metadata for checkpointing.
}
