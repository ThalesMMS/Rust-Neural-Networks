//! Backward pass and parameter update implementation for the LSTM layer.
//!
//! This module contains the backward propagation and parameter update logic
//! extracted from the main LSTM module for better code organization.

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

mod bptt;
mod single_step;
mod update;
