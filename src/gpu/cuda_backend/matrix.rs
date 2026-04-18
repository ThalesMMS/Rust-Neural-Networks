use super::CudaBackend;
use crate::gpu::backend::GpuError;
use cudarc::cublas::{sys, GemmConfig};

impl CudaBackend {
    fn checked_len(lhs: usize, rhs: usize, label: &str) -> Result<usize, GpuError> {
        lhs.checked_mul(rhs).ok_or_else(|| {
            GpuError::DimensionMismatch(format!(
                "{} overflow: {} * {} exceeds usize::MAX",
                label, lhs, rhs
            ))
        })
    }

    fn checked_i32(value: usize, label: &str) -> Result<i32, GpuError> {
        i32::try_from(value).map_err(|_| {
            GpuError::DimensionMismatch(format!("{} {} exceeds i32::MAX", label, value))
        })
    }

    /// Performs single-precision matrix multiplication, producing row-major C = A × B.
    ///
    /// The function adapts row-major inputs to cuBLAS's column-major convention so the device computes the correct row-major result. On success, `c` is overwritten with the product; failures return a `GpuError`.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the GEMM completed and `c` contains the result, `Err(GpuError)` on failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # // Setup omitted: obtain a CudaBackend and valid cuBLAS op enums in real code.
    /// # let backend = CudaBackend::new().unwrap();
    /// # let m = 2usize; let n = 2usize; let k = 2usize;
    /// # let a = vec![1.0f32; m * k];
    /// # let b = vec![1.0f32; k * n];
    /// # let mut c = vec![0.0f32; m * n];
    /// backend
    ///     .dispatch_sgemm(
    ///         sys::cublasOperation_t::CUBLAS_OP_N,
    ///         sys::cublasOperation_t::CUBLAS_OP_N,
    ///         m,
    ///         n,
    ///         k,
    ///         &a,
    ///         &b,
    ///         &mut c,
    ///     )
    ///     .unwrap();
    /// assert_eq!(c.len(), m * n);
    /// ```
    fn dispatch_sgemm(
        &self,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let c_len = Self::checked_len(m, n, "sgemm: C length")?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            c[..c_len].fill(0.0);
            return Ok(());
        }

        let stream = self.stream();

        let a_dev = stream
            .clone_htod(a)
            .map_err(|e| GpuError::TransferError(format!("Failed to copy A to device: {}", e)))?;
        let b_dev = stream
            .clone_htod(b)
            .map_err(|e| GpuError::TransferError(format!("Failed to copy B to device: {}", e)))?;
        let mut c_dev = stream.alloc_zeros::<f32>(c_len).map_err(|e| {
            GpuError::TransferError(format!("Failed to allocate C on device: {}", e))
        })?;

        // cuBLAS is column-major. To compute row-major C = A * B, we use:
        //   C^T = B^T * A^T
        // So we swap A and B and swap m/n, then transpose flags are inverted.
        //
        // For C = A * B (no transpose): cuBLAS does C_col = B_col * A_col with NN
        // For C = A^T * B: cuBLAS does C_col = B_col * A_col^T with N,T
        // For C = A * B^T: cuBLAS does C_col = B_col^T * A_col with T,N

        let effective_transa = transb;
        let effective_transb = transa;
        let lda_cublas = match effective_transa {
            sys::cublasOperation_t::CUBLAS_OP_N => n,
            _ => k,
        };
        let ldb_cublas = match effective_transb {
            sys::cublasOperation_t::CUBLAS_OP_N => k,
            _ => m,
        };
        let cfg_m = Self::checked_i32(n, "sgemm: cuBLAS m")?;
        let cfg_n = Self::checked_i32(m, "sgemm: cuBLAS n")?;
        let cfg_k = Self::checked_i32(k, "sgemm: cuBLAS k")?;
        let lda = Self::checked_i32(lda_cublas, "sgemm: lda")?;
        let ldb = Self::checked_i32(ldb_cublas, "sgemm: ldb")?;
        let ldc = Self::checked_i32(n, "sgemm: ldc")?;

        let cfg = GemmConfig {
            transa: effective_transa, // swap for row-major trick
            transb: effective_transb,
            m: cfg_m,
            n: cfg_n,
            k: cfg_k,
            alpha: 1.0f32,
            lda,
            ldb,
            beta: 0.0f32,
            ldc,
        };

        unsafe {
            self.blas
                .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
                .map_err(|e| GpuError::OperationFailed(format!("cuBLAS sgemm failed: {}", e)))?;
        }

        stream
            .memcpy_dtoh(&c_dev, &mut c[..c_len])
            .map_err(|e| GpuError::TransferError(format!("Failed to copy C from device: {}", e)))?;

        Ok(())
    }

    /// Validates input buffer sizes for a row-major C = A × B (no transposes) and dispatches the operation to the CUDA backend.
    ///
    /// This checks that `a` has at least `m * k` elements, `b` has at least `k * n` elements, and `c` has at least `m * n` elements, then calls into the GPU GEMM dispatcher.
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful dispatch and result copy back to `c`. Returns `GpuError::DimensionMismatch` if any input slice is too small; other `GpuError` variants may be returned from the underlying GPU operations.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Prepare shapes: A is m x k, B is k x n, C is m x n
    /// let m = 2usize;
    /// let n = 3usize;
    /// let k = 4usize;
    /// let a = vec![0f32; m * k];
    /// let b = vec![0f32; k * n];
    /// let mut c = vec![0f32; m * n];
    /// // `backend` is a CudaBackend instance available in scope.
    /// // backend.sgemm_impl(m, n, k, &a, &b, &mut c).unwrap();
    /// ```
    pub(super) fn sgemm_impl(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let expected_a = Self::checked_len(m, k, "sgemm: A length")?;
        let expected_b = Self::checked_len(k, n, "sgemm: B length")?;
        let expected_c = Self::checked_len(m, n, "sgemm: C length")?;
        if a.len() < expected_a || b.len() < expected_b || c.len() < expected_c {
            return Err(GpuError::DimensionMismatch(format!(
                "sgemm: a.len()={} expected {}, b.len()={} expected {}, c.len()={} expected {}",
                a.len(),
                expected_a,
                b.len(),
                expected_b,
                c.len(),
                expected_c,
            )));
        }
        self.dispatch_sgemm(
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            a,
            b,
            c,
        )
    }

    /// Validates buffers for and computes C = Aᵀ × B (single-precision) on the CUDA backend, writing the result into `c`.
    ///
    /// `m`, `n`, and `k` describe the logical matrix dimensions:
    /// - `A` is stored in row-major layout with logical shape `k × m` and is transposed before multiplication,
    /// - `B` is stored in row-major layout with shape `k × n`,
    /// - `C` is stored in row-major layout with shape `m × n`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; `Err(GpuError::DimensionMismatch)` if any input or output slice is too small, or another `GpuError` if a GPU transfer or operation fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Assume `backend` is a properly initialized CudaBackend instance.
    /// let m = 2;
    /// let n = 3;
    /// let k = 4;
    /// // A is stored in row-major layout for the transposed interpretation (k × m)
    /// let a = vec![0f32; k * m];
    /// let b = vec![0f32; k * n];
    /// let mut c = vec![0f32; m * n];
    ///
    /// // Compute C = Aᵀ × B on the GPU
    /// let _ = backend.sgemm_at_impl(m, n, k, &a, &b, &mut c);
    /// ```
    pub(super) fn sgemm_at_impl(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let expected_a = Self::checked_len(k, m, "sgemm_at: A length")?;
        let expected_b = Self::checked_len(k, n, "sgemm_at: B length")?;
        let expected_c = Self::checked_len(m, n, "sgemm_at: C length")?;
        if a.len() < expected_a || b.len() < expected_b || c.len() < expected_c {
            return Err(GpuError::DimensionMismatch(format!(
                "sgemm_at: a.len()={} expected {}, b.len()={} expected {}, c.len()={} expected {}",
                a.len(),
                expected_a,
                b.len(),
                expected_b,
                c.len(),
                expected_c,
            )));
        }
        self.dispatch_sgemm(
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            a,
            b,
            c,
        )
    }

    /// Computes the matrix product C = A × Bᵀ on the CUDA backend after validating input buffer sizes.
    ///
    /// Validates that `a` has at least `m * k` elements, `b` has at least `n * k` elements (since `b` is transposed), and `c` has at least `m * n` elements; on success dispatches the operation to the GPU.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; `Err(GpuError::DimensionMismatch)` if any input slice is too small, or another `GpuError` if the GPU operation or transfers fail.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// // Prepare dimensions
    /// let m = 2usize;
    /// let n = 3usize;
    /// let k = 4usize;
    /// // a: m x k, stored row-major
    /// let a = vec![0f32; m * k];
    /// // b: n x k, stored row-major (will be used as Bᵀ)
    /// let b = vec![0f32; n * k];
    /// // c: m x n output
    /// let mut c = vec![0f32; m * n];
    ///
    /// // `backend` is an instance of `CudaBackend` available in this context.
    /// // let backend = CudaBackend::new(...);
    /// // backend.sgemm_bt_impl(m, n, k, &a, &b, &mut c).unwrap();
    /// ```
    pub(super) fn sgemm_bt_impl(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let expected_a = Self::checked_len(m, k, "sgemm_bt: A length")?;
        let expected_b = Self::checked_len(n, k, "sgemm_bt: B length")?;
        let expected_c = Self::checked_len(m, n, "sgemm_bt: C length")?;
        if a.len() < expected_a || b.len() < expected_b || c.len() < expected_c {
            return Err(GpuError::DimensionMismatch(format!(
                "sgemm_bt: a.len()={} expected {}, b.len()={} expected {}, c.len()={} expected {}",
                a.len(),
                expected_a,
                b.len(),
                expected_b,
                c.len(),
                expected_c,
            )));
        }
        self.dispatch_sgemm(
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_T,
            m,
            n,
            k,
            a,
            b,
            c,
        )
    }
}
