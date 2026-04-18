#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use crate::gpu::GpuBackend;
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use crate::layers::{conv2d::Conv2DLayer, dense::DenseLayer, Layer};
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use std::sync::Arc;

/// Upgrade eligible layers in-place to use the provided GPU backend.
///
/// DenseLayer and Conv2DLayer instances receive a cloned `backend`; other layer types are left unchanged.
///
/// # Returns
///
/// The number of layers that were upgraded.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use rust_neural_networks::training::upgrade_layers_to_gpu;
///
/// // Illustrative: create or obtain a GPU backend and a mutable layer list from the crate.
/// if let Some(backend) = crate::gpu::create_gpu_backend() {
///     let upgraded = upgrade_layers_to_gpu(&mut layers, Arc::new(backend));
///     println!("Upgraded {}/{} layers to GPU", upgraded, layers.len());
/// }
/// ```
#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
pub fn upgrade_layers_to_gpu(layers: &mut [Box<dyn Layer>], backend: Arc<dyn GpuBackend>) -> usize {
    let mut count = 0;
    for layer in layers.iter_mut() {
        let any = layer.as_any_mut();
        let upgraded = if any.is::<DenseLayer>() {
            let dense = any.downcast_mut::<DenseLayer>().unwrap();
            dense.set_gpu_backend(Arc::clone(&backend));
            true
        } else if any.is::<Conv2DLayer>() {
            let conv = any.downcast_mut::<Conv2DLayer>().unwrap();
            conv.set_gpu_backend(Arc::clone(&backend));
            true
        } else {
            false
        };
        if upgraded {
            count += 1;
        }
    }
    count
}
