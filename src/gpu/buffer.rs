//! Type-safe GPU memory buffer abstraction.
//!
//! This module provides [`GpuBuffer`], a wrapper around GPU memory that handles
//! CPU-to-GPU data transfer, and [`BufferPool`], a reuse pool that minimises
//! GPU allocation overhead during training loops.
//!
//! # Overview
//!
//! Neural network training involves many temporary buffers (activations, gradients,
//! intermediate results). Allocating and freeing GPU memory for every operation is
//! expensive. `GpuBuffer` tracks buffer metadata (length, capacity) while
//! `BufferPool` recycles freed buffers so subsequent requests of the same (or
//! smaller) size can be served without a new allocation.
//!
//! # Example
//!
//! ```
//! use rust_neural_networks::gpu::buffer::{GpuBuffer, BufferPool};
//!
//! // Create a buffer from host data
//! let data = vec![1.0f32, 2.0, 3.0, 4.0];
//! let buf = GpuBuffer::from_slice(&data);
//! assert_eq!(buf.len(), 4);
//!
//! // Read data back
//! let host = buf.to_vec();
//! assert_eq!(host, data);
//!
//! // Buffer pool for reuse
//! let mut pool = BufferPool::new();
//! let b1 = pool.acquire(128);
//! assert_eq!(b1.len(), 128);
//! pool.release(b1);
//! let b2 = pool.acquire(100); // reuses the 128-capacity buffer
//! assert!(b2.capacity() >= 100);
//! ```

use std::collections::BTreeMap;

/// A type-safe wrapper around GPU memory.
///
/// In this CPU-side implementation the data is stored in a `Vec<f32>`.
/// When a real GPU backend is active, this would wrap a device-side allocation
/// and manage host↔device transfers. The API is designed so that swapping in
/// actual GPU memory requires no changes to calling code.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::gpu::buffer::GpuBuffer;
///
/// let buf = GpuBuffer::zeros(16);
/// assert_eq!(buf.len(), 16);
/// assert!(buf.as_slice().iter().all(|&v| v == 0.0));
/// ```
pub struct GpuBuffer {
    /// Underlying storage. In a real GPU backend this would be a device pointer
    /// with a host-side shadow; here we use `Vec<f32>` for correctness testing.
    data: Vec<f32>,
    /// Logical length (may be less than `data.len()` when reused from pool).
    len: usize,
}

impl GpuBuffer {
    /// Creates a new GpuBuffer with the given logical length, initialized to zeros.
    ///
    /// The buffer's capacity equals its logical length and all elements are 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    ///
    /// let buf = GpuBuffer::zeros(8);
    /// assert_eq!(buf.len(), 8);
    /// assert_eq!(buf.to_vec(), vec![0.0f32; 8]);
    /// ```
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0f32; len],
            len,
        }
    }

    /// Creates a GpuBuffer containing a copy of `data` with logical length equal to `data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    ///
    /// let buf = GpuBuffer::from_slice(&[1.0, 2.0, 3.0]);
    /// assert_eq!(buf.len(), 3);
    /// assert_eq!(buf.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            len: data.len(),
        }
    }

    /// Creates a buffer with the specified capacity and sets its logical length to that capacity.
    ///
    /// The underlying storage is allocated for `capacity` f32 elements and all visible elements are initialized to `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::with_capacity(4);
    /// assert_eq!(buf.len(), 4);
    /// assert_eq!(buf.capacity(), 4);
    /// assert_eq!(buf.as_slice(), &[0.0_f32, 0.0, 0.0, 0.0]);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, 0.0);
        Self {
            len: capacity,
            data,
        }
    }

    /// Retrieve the logical length (number of visible elements) of the buffer.
    ///
    /// # Returns
    ///
    /// The logical length (number of visible elements) of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::zeros(4);
    /// assert_eq!(buf.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks whether the buffer has zero logical elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::zeros(0);
    /// assert!(buf.is_empty());
    ///
    /// let buf = GpuBuffer::zeros(3);
    /// assert!(!buf.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer's total allocated capacity in elements.
    ///
    /// This is the length of the underlying storage and may be greater than the logical
    /// length returned by `len()` when a buffer is reused from a pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::with_capacity(8);
    /// assert_eq!(buf.capacity(), 8);
    /// ```
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Returns an immutable slice view of the buffer's visible elements.
    ///
    /// The slice contains the first `len` elements; the underlying allocation (`capacity`) may be larger.
    ///
    /// # Returns
    ///
    /// A slice containing the buffer's visible elements (`&[f32]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::from_slice(&[1.0f32, 2.0, 3.0]);
    /// let s = buf.as_slice();
    /// assert_eq!(s, &[1.0, 2.0, 3.0]);
    /// ```
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len]
    }

    /// Get a mutable view of the buffer's visible elements.
    ///
    /// The returned slice covers the first `len` elements of the buffer (the logical length),
    /// not the full allocated capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let mut buf = GpuBuffer::zeros(4);
    /// let s = buf.as_mut_slice();
    /// s[0] = 1.0;
    /// assert_eq!(buf.to_vec()[0], 1.0);
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data[..self.len]
    }

    /// Copies the buffer's visible elements into a host `Vec<f32>`.
    ///
    /// In a real GPU backend this would perform a GPU→CPU transfer.
    /// The returned `Vec<f32>` contains the buffer's first `len` elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::from_slice(&[1.0, 2.0, 3.0]);
    /// let v = buf.to_vec();
    /// assert_eq!(v, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }

    /// Copies elements from `data` into the buffer's visible region and sets the buffer's logical length to `data.len()`.
    ///
    /// Panics if `data.len()` is greater than the buffer's capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let mut buf = GpuBuffer::with_capacity(4);
    /// buf.copy_from_slice(&[1.0_f32, 2.0]);
    /// assert_eq!(buf.len(), 2);
    /// assert_eq!(buf.as_slice(), &[1.0_f32, 2.0]);
    /// ```
    pub fn copy_from_slice(&mut self, data: &[f32]) {
        assert!(
            data.len() <= self.data.len(),
            "GpuBuffer::copy_from_slice: source length {} exceeds capacity {}",
            data.len(),
            self.data.len()
        );
        self.data[..data.len()].copy_from_slice(data);
        self.len = data.len();
    }

    /// Sets every visible element of the buffer (indices 0..len) to 0.0.
    ///
    /// This only affects the logical portion of the buffer; unused capacity beyond `len` is left unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let mut buf = GpuBuffer::from_slice(&[1.0f32, -2.5, 3.25]);
    /// buf.zero_fill();
    /// assert_eq!(buf.as_slice(), &[0.0f32, 0.0, 0.0]);
    /// ```
    pub fn zero_fill(&mut self) {
        self.data[..self.len].iter_mut().for_each(|v| *v = 0.0);
    }

    /// Adjusts the buffer's logical length.
    ///
    /// If `new_len` is less than the current logical length, the logical length is reduced without shrinking
    /// the allocated capacity. If `new_len` is greater than the allocated capacity, the underlying storage
    /// is grown and any newly allocated elements are set to `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let mut buf = GpuBuffer::zeros(2);
    /// assert_eq!(buf.len(), 2);
    ///
    /// buf.resize(4);
    /// assert_eq!(buf.len(), 4);
    /// assert!(buf.as_slice().iter().all(|&v| v == 0.0));
    ///
    /// buf.resize(1);
    /// assert_eq!(buf.len(), 1);
    /// assert!(buf.capacity() >= 1);
    /// ```
    pub fn resize(&mut self, new_len: usize) {
        if new_len > self.data.len() {
            self.data.resize(new_len, 0.0);
        }
        self.len = new_len;
    }
}

impl std::fmt::Debug for GpuBuffer {
    /// Format a debug representation of the buffer showing its logical length and allocated capacity.
    ///
    /// The formatted output is a struct-like representation named `GpuBuffer` with fields
    /// `len` (the logical length) and `capacity` (the underlying allocation size).
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::GpuBuffer;
    /// let buf = GpuBuffer::zeros(4);
    /// let s = format!("{:?}", buf);
    /// assert!(s.contains("GpuBuffer"));
    /// assert!(s.contains("len"));
    /// assert!(s.contains("capacity"));
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.len)
            .field("capacity", &self.data.len())
            .finish()
    }
}

/// A pool that recycles [`GpuBuffer`] instances to avoid repeated allocations.
///
/// Freed buffers are stored in size-bucketed bins. When a new buffer is requested,
/// the pool finds the smallest available buffer whose capacity is ≥ the requested
/// size, avoiding a fresh allocation.
///
/// # Examples
///
/// ```
/// use rust_neural_networks::gpu::buffer::BufferPool;
///
/// let mut pool = BufferPool::new();
///
/// // First acquire always allocates
/// let buf = pool.acquire(256);
/// assert_eq!(buf.len(), 256);
///
/// // Release returns it to the pool
/// pool.release(buf);
///
/// // Second acquire reuses the released buffer
/// let buf2 = pool.acquire(200);
/// assert!(buf2.capacity() >= 200);
///
/// // Clear the pool to free all cached buffers
/// pool.clear();
/// assert_eq!(pool.cached_count(), 0);
/// ```
pub struct BufferPool {
    /// Available buffers keyed by their capacity for efficient lookup.
    /// Using `BTreeMap` so we can efficiently find the smallest buffer ≥ requested size.
    available: BTreeMap<usize, Vec<GpuBuffer>>,
    /// Total number of cached buffers.
    count: usize,
    /// Maximum number of buffers to cache (prevents unbounded memory growth).
    max_cached: usize,
}

impl BufferPool {
    /// Creates a new empty buffer pool with a default cache limit of 64 buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::BufferPool;
    /// let mut pool = BufferPool::new();
    /// assert_eq!(pool.cached_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            available: BTreeMap::new(),
            count: 0,
            max_cached: 64,
        }
    }

    /// Constructs a buffer pool that caches up to the given number of buffers.
    ///
    /// The pool organizes cached buffers by their capacity and drops released buffers when the total
    /// cached count would exceed `max_cached`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::BufferPool;
    /// let pool = BufferPool::with_max_cached(8);
    /// assert_eq!(pool.cached_count(), 0);
    /// ```
    pub fn with_max_cached(max_cached: usize) -> Self {
        Self {
            available: BTreeMap::new(),
            count: 0,
            max_cached,
        }
    }

    /// Acquire a buffer whose logical length is the requested size.
    ///
    /// Reuses the smallest cached buffer with capacity >= `size` when available (it will be resized to
    /// `size` and zero-filled); otherwise allocates a new zero-initialized buffer.
    ///
    /// # Arguments
    ///
    /// * `size` - Requested logical length in number of `f32` elements.
    ///
    /// # Returns
    ///
    /// A `GpuBuffer` with logical length equal to `size` and capacity at least `size`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::BufferPool;
    /// let mut pool = BufferPool::new();
    /// let mut buf = pool.acquire(8);
    /// assert_eq!(buf.len(), 8);
    /// // Use buffer...
    /// pool.release(buf);
    /// ```
    pub fn acquire(&mut self, size: usize) -> GpuBuffer {
        // Find the smallest cached buffer with capacity >= size
        if let Some((&cap, _)) = self.available.range(size..).find(|(_, v)| !v.is_empty()) {
            let bufs = self.available.get_mut(&cap).unwrap();
            let mut buf = bufs.pop().unwrap();
            self.count -= 1;
            // Clean empty entries
            if bufs.is_empty() {
                self.available.remove(&cap);
            }
            buf.resize(size);
            buf.zero_fill();
            buf
        } else {
            GpuBuffer::zeros(size)
        }
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// If the pool already contains `max_cached` buffers, the provided buffer is dropped
    /// instead of being stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::{BufferPool, GpuBuffer};
    /// let mut pool = BufferPool::with_max_cached(1);
    /// let buf = GpuBuffer::zeros(4);
    /// pool.release(buf);
    /// assert_eq!(pool.cached_count(), 1);
    /// ```
    pub fn release(&mut self, buf: GpuBuffer) {
        if self.count >= self.max_cached {
            return; // Drop the buffer to stay within limits
        }
        let cap = buf.capacity();
        self.available.entry(cap).or_default().push(buf);
        self.count += 1;
    }

    /// Total number of buffers currently cached in the pool.
    ///
    /// # Returns
    ///
    /// The number of cached buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::{BufferPool, GpuBuffer};
    /// let mut pool = BufferPool::new();
    /// assert_eq!(pool.cached_count(), 0);
    /// pool.release(GpuBuffer::zeros(4));
    /// assert_eq!(pool.cached_count(), 1);
    /// ```
    pub fn cached_count(&self) -> usize {
        self.count
    }

    /// Empties the pool and drops all cached buffers, releasing their memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::{BufferPool, GpuBuffer};
    /// let mut pool = BufferPool::new();
    /// let buf = GpuBuffer::zeros(16);
    /// pool.release(buf);
    /// assert!(pool.cached_count() > 0);
    /// pool.clear();
    /// assert_eq!(pool.cached_count(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.available.clear();
        self.count = 0;
    }
}

impl Default for BufferPool {
    /// Creates an empty `BufferPool` configured with the crate's default maximum cached buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::gpu::buffer::BufferPool;
    /// let pool = BufferPool::default();
    /// assert_eq!(pool.cached_count(), 0);
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_zeros() {
        let buf = GpuBuffer::zeros(16);
        assert_eq!(buf.len(), 16);
        assert!(!buf.is_empty());
        assert!(buf.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_buffer_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = GpuBuffer::from_slice(&data);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn test_buffer_copy_from_slice() {
        let mut buf = GpuBuffer::zeros(8);
        buf.copy_from_slice(&[5.0, 6.0, 7.0]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.to_vec(), vec![5.0, 6.0, 7.0]);
    }

    #[test]
    #[should_panic(expected = "exceeds capacity")]
    fn test_buffer_copy_from_slice_overflow() {
        let mut buf = GpuBuffer::zeros(2);
        buf.copy_from_slice(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_buffer_zero_fill() {
        let mut buf = GpuBuffer::from_slice(&[1.0, 2.0, 3.0]);
        buf.zero_fill();
        assert_eq!(buf.to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_buffer_resize() {
        let mut buf = GpuBuffer::zeros(4);
        buf.resize(8);
        assert_eq!(buf.len(), 8);
        buf.resize(2);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_buffer_mut_slice() {
        let mut buf = GpuBuffer::zeros(3);
        let s = buf.as_mut_slice();
        s[0] = 10.0;
        s[1] = 20.0;
        s[2] = 30.0;
        assert_eq!(buf.to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_buffer_empty() {
        let buf = GpuBuffer::zeros(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_pool_acquire_release() {
        let mut pool = BufferPool::new();
        let buf = pool.acquire(128);
        assert_eq!(buf.len(), 128);
        assert_eq!(pool.cached_count(), 0);

        pool.release(buf);
        assert_eq!(pool.cached_count(), 1);

        // Reuse the 128-capacity buffer for a smaller request
        let buf2 = pool.acquire(100);
        assert!(buf2.capacity() >= 100);
        assert_eq!(buf2.len(), 100);
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn test_pool_zero_fills_on_acquire() {
        let mut pool = BufferPool::new();
        let mut buf = pool.acquire(4);
        buf.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        pool.release(buf);

        let buf2 = pool.acquire(4);
        assert!(buf2.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pool_max_cached() {
        let mut pool = BufferPool::with_max_cached(2);
        let b1 = pool.acquire(10);
        let b2 = pool.acquire(20);
        let b3 = pool.acquire(30);

        pool.release(b1);
        pool.release(b2);
        pool.release(b3); // Should be dropped (max is 2)
        assert_eq!(pool.cached_count(), 2);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = BufferPool::new();
        pool.release(GpuBuffer::zeros(10));
        pool.release(GpuBuffer::zeros(20));
        assert_eq!(pool.cached_count(), 2);
        pool.clear();
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn test_pool_default() {
        let pool = BufferPool::default();
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn test_buffer_debug() {
        let buf = GpuBuffer::zeros(16);
        let debug = format!("{:?}", buf);
        assert!(debug.contains("GpuBuffer"));
        assert!(debug.contains("16"));
    }
}