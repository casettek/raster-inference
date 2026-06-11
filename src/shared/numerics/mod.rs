pub mod det_num;
pub(crate) mod det_kernels;
/// Native-only SIMD kernels; public for benchmarks, never for guest code.
pub mod det_simd;
pub(crate) mod det_tensor;
pub mod transformer_kernels;
