//! GPU acceleration backends for neural network operations.
//!
//! This module provides GPU-accelerated implementations of matrix operations
//! and layer computations. Two backends are supported:
//!
//! - **Metal** (macOS): Enabled via the `gpu-metal` feature flag
//! - **CUDA** (NVIDIA): Enabled via the `gpu-cuda` feature flag
//!
//! # Usage
//!
//! Enable the appropriate feature in your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rust_neural_networks = { path = ".", features = ["gpu-metal"] }
//! ```

pub mod backend;
pub mod buffer;

#[cfg(feature = "gpu-metal")]
pub mod metal_backend;

#[cfg(feature = "gpu-cuda")]
pub mod cuda_backend;

pub use backend::{BackendType, GpuBackend, GpuDevice, GpuError};

use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// GpuInfo
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of detected GPU capabilities for display and decision-making.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Human-readable device name (e.g., "Apple M1 Pro", "NVIDIA RTX 4090").
    pub device_name: String,
    /// Backend type (Metal or CUDA).
    pub backend_type: BackendType,
    /// Total device memory in bytes, if known.
    pub memory_bytes: Option<u64>,
    /// Whether the device supports the operations required by this library.
    pub is_supported: bool,
}

impl std::fmt::Display for GpuInfo {
    /// Formats a `GpuInfo` for display.
    ///
    /// The output is `"<device_name> (<backend_type>)"`, appends `", <N> MB"` when
    /// `memory_bytes` is present (megabytes, rounded down), and appends
    /// `" [supported]"` or `" [unsupported]"` based on `is_supported`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::{GpuInfo, BackendType};
    ///
    /// let info = GpuInfo {
    ///     device_name: "Test GPU".to_string(),
    ///     backend_type: BackendType::Metal,
    ///     memory_bytes: Some(2 * 1024 * 1024 * 1024),
    ///     is_supported: true,
    /// };
    ///
    /// assert_eq!(format!("{}", info), "Test GPU (Metal), 2048 MB [supported]");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.device_name, self.backend_type)?;
        if let Some(mem) = self.memory_bytes {
            let mb = mem / (1024 * 1024);
            write!(f, ", {} MB", mb)?;
        }
        if self.is_supported {
            write!(f, " [supported]")
        } else {
            write!(f, " [unsupported]")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runtime GPU detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detects the first available GPU backend and returns information about that device.
///
/// Returns `Some(GpuInfo)` for the first detected device backend, or `None` if no supported GPU backend is available.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::gpu::detect_gpu;
///
/// if let Some(info) = detect_gpu() {
///     println!("Found GPU: {}", info);
/// } else {
///     println!("No GPU detected, using CPU");
/// }
/// ```
pub fn detect_gpu() -> Option<GpuInfo> {
    // Try Metal first (macOS)
    #[cfg(feature = "gpu-metal")]
    {
        if let Ok(backend) = metal_backend::MetalBackend::new() {
            let dev = backend.device_info();
            return Some(GpuInfo {
                device_name: dev.name.clone(),
                backend_type: dev.backend,
                memory_bytes: dev.memory_bytes,
                is_supported: dev.is_supported,
            });
        }
    }

    // Try CUDA (NVIDIA)
    #[cfg(feature = "gpu-cuda")]
    {
        if let Ok(backend) = cuda_backend::CudaBackend::new() {
            let dev = backend.device_info();
            return Some(GpuInfo {
                device_name: dev.name.clone(),
                backend_type: dev.backend,
                memory_bytes: dev.memory_bytes,
                is_supported: dev.is_supported,
            });
        }
    }

    None
}

/// Create and return the first available supported GPU backend.
///
/// Attempts enabled backends in priority order (Metal first when the `gpu-metal` feature is enabled,
/// then CUDA when the `gpu-cuda` feature is enabled) and returns the first backend successfully created.
///
/// # Returns
///
/// `Some(Arc<dyn GpuBackend>)` if a supported GPU backend was successfully created, `None` otherwise.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use rust_neural_networks::gpu::{create_gpu_backend, GpuBackend};
///
/// if let Some(backend) = create_gpu_backend() {
///     println!("GPU backend ready: {}", backend.device_info().name);
/// }
/// ```
pub fn create_gpu_backend() -> Option<Arc<dyn GpuBackend>> {
    #[cfg(feature = "gpu-metal")]
    {
        if let Ok(backend) = metal_backend::MetalBackend::new() {
            return Some(Arc::new(backend));
        }
    }

    #[cfg(feature = "gpu-cuda")]
    {
        if let Ok(backend) = cuda_backend::CudaBackend::new() {
            return Some(Arc::new(backend));
        }
    }

    None
}

/// Print startup GPU device diagnostics to stdout.
///
/// Displays detected device name, backend, memory (MB/GB or "unknown"), and support status. If no usable GPU is available, prints reasons and a message that the process will run in CPU-only mode.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::gpu::print_gpu_info;
///
/// print_gpu_info();
/// ```
/// Print GPU device diagnostics from an already-created backend.
///
/// Displays device name, backend type, memory, and support status using the
/// information from `backend.device_info()` without constructing a second backend.
///
/// # Examples
///
/// ```ignore
/// if let Some(backend) = create_gpu_backend() {
///     print_backend_info(&*backend);
/// }
/// ```
pub fn print_backend_info(backend: &dyn GpuBackend) {
    let dev = backend.device_info();
    println!("────────────────────────────────────────");
    println!("  GPU Device Information");
    println!("────────────────────────────────────────");
    println!("  Device:    {}", dev.name);
    println!("  Backend:   {}", dev.backend);
    if let Some(mem) = dev.memory_bytes {
        let mb = mem / (1024 * 1024);
        let gb = mb as f64 / 1024.0;
        if gb >= 1.0 {
            println!("  Memory:    {:.1} GB ({} MB)", gb, mb);
        } else {
            println!("  Memory:    {} MB", mb);
        }
    } else {
        println!("  Memory:    unknown");
    }
    println!(
        "  Status:    {}",
        if dev.is_supported {
            "supported ✓"
        } else {
            "unsupported ✗"
        }
    );
    println!("────────────────────────────────────────");
}

pub fn print_gpu_info() {
    println!("────────────────────────────────────────");
    println!("  GPU Device Information");
    println!("────────────────────────────────────────");
    match detect_gpu() {
        Some(info) => {
            println!("  Device:    {}", info.device_name);
            println!("  Backend:   {}", info.backend_type);
            if let Some(mem) = info.memory_bytes {
                let mb = mem / (1024 * 1024);
                let gb = mb as f64 / 1024.0;
                if gb >= 1.0 {
                    println!("  Memory:    {:.1} GB ({} MB)", gb, mb);
                } else {
                    println!("  Memory:    {} MB", mb);
                }
            } else {
                println!("  Memory:    unknown");
            }
            println!(
                "  Status:    {}",
                if info.is_supported {
                    "supported ✓"
                } else {
                    "unsupported ✗"
                }
            );
        }
        None => {
            let mut reasons = Vec::new();
            if !cfg!(feature = "gpu-metal") && !cfg!(feature = "gpu-cuda") {
                reasons.push("no GPU feature enabled (use --features gpu-metal or gpu-cuda)");
            } else {
                reasons.push("no compatible GPU device found");
            }
            println!("  No GPU detected: {}", reasons.join(", "));
            println!("  Running in CPU-only mode");
        }
    }
    println!("────────────────────────────────────────");
}