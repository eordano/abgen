//! wgpu (WGSL) backend for the GPU mesh-coarsen lane: a portable
//! CUDA-independent implementation of the four mesh kernels in
//! kernel-ptx/src/lib.rs (survey / accum / pick / remap), byte-identical to
//! the CPU oracle (gpu_mesh_dispatch::CpuBackend).
//!
//! Determinism strategy (see soft.rs): all float math runs as u32 integer
//! softfloat in the shader; i64 quadric accumulation uses two-u32 carry
//! emulation over atomicAdd (order-independent because wrapping mod-2^64
//! sums and total carry counts are order-free); the packed-u64 atomicMin
//! rep-pick becomes two dispatches (min cost, then min id among cost
//! minima), which equals the lexicographic u64 minimum.

pub(crate) mod driver;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod soft;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod softmesh;

pub use driver::{mesh_kernels_available, warm_pipelines, MeshSession};

#[cfg(test)]
mod shader_tests;
#[cfg(test)]
mod soft_tests;
