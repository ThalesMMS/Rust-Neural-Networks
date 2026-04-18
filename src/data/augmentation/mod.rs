//! Image data augmentation functions for training.
//!
//! This module provides common data augmentation techniques to improve model
//! generalization and prevent overfitting. Augmentations should only be applied
//! to training data, not validation or test sets.
//!
//! All functions assume pixel-interleaved RGB format (RGBRGBRGB...) as used
//! by the CIFAR-10 loader.

mod color;
mod spatial;

pub use self::color::{random_brightness, random_contrast, random_saturation};
pub use self::spatial::{random_crop, random_crop_into, random_horizontal_flip};
