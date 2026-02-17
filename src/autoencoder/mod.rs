//! Autoencoder architectures for unsupervised representation learning
//!
//! This module provides implementations for vanilla autoencoders and
//! variational autoencoders (VAEs) for dimensionality reduction and
//! generative modeling on datasets such as MNIST and CIFAR-10.
//!
//! # Modules
//!
//! - `vanilla`: Standard autoencoder with encoder-decoder architecture
//! - `vae`: Variational autoencoder with latent space sampling

pub mod vae;
pub mod vanilla;
