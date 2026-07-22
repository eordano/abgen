//! Host driver for the WGSL mesh-coarsen kernels: the wgpu twin of
//! crate/src/gpu/cuda/mesh.rs. Session flow is identical (open uploads
//! positions/indices/edges once; survey and cluster dispatch per call and
//! read back), with two differences forced by WGSL:
//! - survey grids (dims + inv_cell per scale) are precomputed on the host
//!   with mc::scale_grid — bit-identical to both the CPU oracle (which also
//!   computes them on the host) and the CUDA kernel (which recomputes them
//!   on-device with the same correctly-rounded f32 ops);
//! - the pick stage runs as two dispatches (cost min, then id min), see
//!   shaders/mesh_coarsen.wgsl.

use crate::gpu::corelib::mesh_coarsen as mc;
use crate::gpu::wgpu::{block_on_now, gpu, Gpu};
use anyhow::{anyhow, ensure, Result};
use std::sync::OnceLock;

pub(crate) const MESH_WGSL: &str = include_str!("../shaders/mesh_coarsen.wgsl");

const WG: u64 = 256;
const NCELLS_CLUSTER_CAP: u64 = 8_388_608;
const MAX_SCALES: u64 = 64;

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

pub struct MeshPipes {
    survey: ::wgpu::ComputePipeline,
    accum: ::wgpu::ComputePipeline,
    accum_edges: ::wgpu::ComputePipeline,
    fill_ff: ::wgpu::ComputePipeline,
    pick_cost: ::wgpu::ComputePipeline,
    pick_id: ::wgpu::ComputePipeline,
    remap: ::wgpu::ComputePipeline,
}

static PIPES: OnceLock<Result<MeshPipes, String>> = OnceLock::new();

fn build_pipes(g: &Gpu) -> Result<MeshPipes, String> {
    let scope = g.device.push_error_scope(::wgpu::ErrorFilter::Validation);
    let module = g
        .device
        .create_shader_module(::wgpu::ShaderModuleDescriptor {
            label: Some("mesh-coarsen"),
            source: ::wgpu::ShaderSource::Wgsl(MESH_WGSL.into()),
        });
    let mk = |entry: &'static str| {
        g.device
            .create_compute_pipeline(&::wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    // The six entry points compile independently; building them on parallel
    // threads only overlaps driver compile time — the pipelines themselves,
    // and every result they produce, are unaffected by creation order.
    // ABGEN_WGPU_SERIAL_COMPILE=1 restores the serial path (bench A/B).
    let entries = [
        "mesh_survey",
        "mesh_accum",
        "mesh_accum_edges",
        "mesh_fill_ff",
        "mesh_pick_cost",
        "mesh_pick_id",
        "mesh_remap",
    ];
    let serial = std::env::var("ABGEN_WGPU_SERIAL_COMPILE").as_deref() == Ok("1");
    let mut built = if serial {
        entries.into_iter().map(&mk).collect::<Vec<_>>()
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = entries
                .into_iter()
                .map(|entry| s.spawn(move || mk(entry)))
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|p| std::panic::resume_unwind(p))
                })
                .collect::<Vec<_>>()
        })
    }
    .into_iter();
    let mut next = || built.next().expect("seven pipelines");
    let pipes = MeshPipes {
        survey: next(),
        accum: next(),
        accum_edges: next(),
        fill_ff: next(),
        pick_cost: next(),
        pick_id: next(),
        remap: next(),
    };
    if let Some(e) = block_on_now(scope.pop()) {
        return Err(format!("mesh WGSL pipeline validation failed: {e}"));
    }
    Ok(pipes)
}

fn pipes(g: &'static Gpu) -> Result<&'static MeshPipes, String> {
    let init = || {
        // naga's SPIR-V backend can panic on unsupported constructs (seen
        // with pointer-to-member arguments); a panic here must degrade to
        // "mesh kernels unavailable", never take the process down.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build_pipes(g)))
            .unwrap_or_else(|p| {
                let msg = p
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| String::from("panic"));
                Err(format!("mesh WGSL pipeline build panicked: {msg}"))
            })
    };
    match PIPES.get_or_init(init) {
        Ok(p) => Ok(p),
        Err(e) => Err(e.clone()),
    }
}

/// True when a wgpu adapter is present and the mesh pipelines validate.
pub fn mesh_kernels_available() -> bool {
    match gpu() {
        Ok(g) => pipes(g).is_ok(),
        Err(_) => false,
    }
}

/// Force adapter init + pipeline compile now (bench: isolates cold compile
/// cost from the first survey/cluster call).
pub fn warm_pipelines() -> Result<()> {
    let g = gpu().map_err(|e| anyhow!("{e}"))?;
    pipes(g).map_err(|e| anyhow!("{e}"))?;
    Ok(())
}

/// ABGEN_WGPU_MESH_TIMING=1 prints per-stage timings to stderr: GPU-side
/// per-dispatch times via timestamp queries when the adapter has them, plus
/// host-side encode/submit/readback walls. Purely observational — the
/// dispatch chain, submit pattern, and all math are unchanged.
fn timing_enabled() -> bool {
    std::env::var("ABGEN_WGPU_MESH_TIMING").as_deref() == Ok("1")
}

struct GpuTimer {
    qs: ::wgpu::QuerySet,
    labels: Vec<&'static str>,
}

impl GpuTimer {
    fn new(g: &Gpu) -> Option<GpuTimer> {
        if !timing_enabled()
            || !g
                .device
                .features()
                .contains(::wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let qs = g.device.create_query_set(&::wgpu::QuerySetDescriptor {
            label: Some("mesh-stage-times"),
            ty: ::wgpu::QueryType::Timestamp,
            count: 128,
        });
        Some(GpuTimer {
            qs,
            labels: Vec::new(),
        })
    }

    fn report(self, g: &Gpu, what: &str) -> Result<()> {
        use std::fmt::Write as _;
        let n = self.labels.len() as u32 * 2;
        if n == 0 {
            return Ok(());
        }
        let resolve = g.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("mesh-ts-resolve"),
            size: n as u64 * 8,
            usage: ::wgpu::BufferUsages::QUERY_RESOLVE | ::wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = g.device.create_command_encoder(&Default::default());
        enc.resolve_query_set(&self.qs, 0..n, &resolve, 0);
        let raw = readback(g, enc, &resolve, n as u64 * 8)?;
        let ticks: Vec<u64> = raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let period = g.queue.get_timestamp_period() as f64;
        let mut line = format!("wgpu-mesh-gpu {what}:");
        let mut sum = 0.0f64;
        for (i, lbl) in self.labels.iter().enumerate() {
            let ms = ticks[i * 2 + 1].wrapping_sub(ticks[i * 2]) as f64 * period / 1e6;
            sum += ms;
            let _ = write!(line, " {lbl} {ms:.3}");
        }
        let _ = write!(line, " | sum {sum:.3} ms");
        eprintln!("{line}");
        Ok(())
    }
}

fn as_bytes<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), std::mem::size_of_val(s)) }
}

fn storage_init(g: &Gpu, label: &str, data: &[u8]) -> ::wgpu::Buffer {
    use ::wgpu::util::DeviceExt;
    g.device
        .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: ::wgpu::BufferUsages::STORAGE,
        })
}

/// STORAGE buffer; WebGPU guarantees zero initialization.
fn storage_zeroed(g: &Gpu, label: &str, size: u64, extra: ::wgpu::BufferUsages) -> ::wgpu::Buffer {
    g.device.create_buffer(&::wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: ::wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

struct Params {
    mn: [f32; 3],
    inv_ext: f32,
    inv_cell: [f32; 3],
    nverts: u32,
    dims: [u32; 3],
    ntris: u32,
    nedges: u32,
    ncells: u32,
    nscales: u32,
}

fn params_bytes(p: &Params, base: u32) -> Vec<u8> {
    let words: [u32; 16] = [
        p.mn[0].to_bits(),
        p.mn[1].to_bits(),
        p.mn[2].to_bits(),
        p.inv_ext.to_bits(),
        p.inv_cell[0].to_bits(),
        p.inv_cell[1].to_bits(),
        p.inv_cell[2].to_bits(),
        p.nverts,
        p.dims[0],
        p.dims[1],
        p.dims[2],
        p.ntris,
        p.nedges,
        p.ncells,
        p.nscales,
        base,
    ];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Per-cluster-call buffers whose size depends on the grid's cell count.
/// Pooled with grow-on-demand: reused across cluster calls within a session
/// and reallocated only when a call needs more cells. Every element a kernel
/// can read is explicitly re-initialized each call (cell_q via clear_buffer,
/// best_cost/best_id via the exact 0xff pattern), so reuse cannot leak bytes
/// between calls.
struct CellBufs {
    cap_cells: u64,
    cell_q: ::wgpu::Buffer,
    best_cost: ::wgpu::Buffer,
    best_id: ::wgpu::Buffer,
}

pub struct MeshSession {
    g: &'static Gpu,
    pipes: &'static MeshPipes,
    d_pos: ::wgpu::Buffer,
    d_idx: ::wgpu::Buffer,
    d_edges: ::wgpu::Buffer,
    d_grids: ::wgpu::Buffer,
    d_counts: ::wgpu::Buffer,
    d_vert_cost: ::wgpu::Buffer,
    d_out: ::wgpu::Buffer,
    staging: ::wgpu::Buffer,
    cells: Option<CellBufs>,
    nverts: usize,
    ntris: usize,
    nedges: usize,
    mn: [f32; 3],
    ext: [f32; 3],
}

/// Dispatch `total` threads of `pipeline`, chunking by the device's
/// max_compute_workgroups_per_dimension via the params `base` field.
fn run_stage(
    g: &Gpu,
    enc: &mut ::wgpu::CommandEncoder,
    pipeline: &::wgpu::ComputePipeline,
    params: &Params,
    total: u64,
    bufs: &[(u32, &::wgpu::Buffer)],
    timer: &mut Option<GpuTimer>,
    label: &'static str,
) {
    use ::wgpu::util::DeviceExt;
    let max_wg = g.device.limits().max_compute_workgroups_per_dimension as u64;
    let chunk = max_wg * WG;
    let layout = pipeline.get_bind_group_layout(0);
    let mut base = 0u64;
    while base < total {
        let n = (total - base).min(chunk);
        let params_buf = g
            .device
            .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                label: Some("mesh-params"),
                contents: &params_bytes(params, base as u32),
                usage: ::wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![::wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buf.as_entire_binding(),
        }];
        for (binding, buf) in bufs {
            entries.push(::wgpu::BindGroupEntry {
                binding: *binding,
                resource: buf.as_entire_binding(),
            });
        }
        let bg = g.device.create_bind_group(&::wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let ts_index = timer.as_mut().and_then(|t| {
            let i = t.labels.len() as u32;
            if i * 2 + 1 >= 128 {
                return None;
            }
            t.labels.push(label);
            Some(i)
        });
        let timestamp_writes = match (timer.as_ref(), ts_index) {
            (Some(t), Some(i)) => Some(::wgpu::ComputePassTimestampWrites {
                query_set: &t.qs,
                beginning_of_pass_write_index: Some(i * 2),
                end_of_pass_write_index: Some(i * 2 + 1),
            }),
            _ => None,
        };
        let mut pass = enc.begin_compute_pass(&::wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(WG) as u32, 1, 1);
        drop(pass);
        base += n;
    }
}

/// Run `work` under validation + OOM error scopes so wgpu errors surface as
/// Err (feeding the meshopt fallback) instead of the default panic handler.
fn scoped<T>(g: &Gpu, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let oom = g.device.push_error_scope(::wgpu::ErrorFilter::OutOfMemory);
    let val = g.device.push_error_scope(::wgpu::ErrorFilter::Validation);
    let out = work();
    let val_err = block_on_now(val.pop());
    let oom_err = block_on_now(oom.pop());
    if let Some(e) = val_err {
        return Err(anyhow!("wgpu mesh validation error: {e}"));
    }
    if let Some(e) = oom_err {
        return Err(anyhow!("wgpu mesh out of memory: {e}"));
    }
    out
}

/// One-shot readback via a fresh staging buffer (timer resolve path only;
/// data readbacks go through the pooled session staging buffer).
fn readback(g: &Gpu, enc: ::wgpu::CommandEncoder, src: &::wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
    let staging = g.device.create_buffer(&::wgpu::BufferDescriptor {
        label: Some("mesh-staging"),
        size,
        usage: ::wgpu::BufferUsages::MAP_READ | ::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    readback_via(g, enc, src, &staging, size)
}

/// Copy `size` bytes of `src` into `staging` after the encoded work, submit,
/// wait, and return the bytes. Transport only: the same finished GPU buffer
/// is copied after the same submit regardless of which staging buffer (fresh
/// or session-pooled) carries it; a pooled buffer's stale tail beyond `size`
/// is never mapped.
fn readback_via(
    g: &Gpu,
    enc: ::wgpu::CommandEncoder,
    src: &::wgpu::Buffer,
    staging: &::wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>> {
    let t0 = std::time::Instant::now();
    let mut enc = enc;
    enc.copy_buffer_to_buffer(src, 0, staging, 0, size);
    g.queue.submit([enc.finish()]);
    let submit_ms = ms(t0);
    let t1 = std::time::Instant::now();
    let slice = staging.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(::wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    g.device
        .poll(::wgpu::PollType::wait_indefinitely())
        .map_err(|e| anyhow!("wgpu poll failed: {e:?}"))?;
    rx.recv()
        .map_err(|_| anyhow!("map_async callback dropped"))?
        .map_err(|e| anyhow!("mesh readback map failed: {e:?}"))?;
    let wait_ms = ms(t1);
    let t2 = std::time::Instant::now();
    let out = slice
        .get_mapped_range()
        .map_err(|e| anyhow!("mesh readback range failed: {e:?}"))?
        .to_vec();
    staging.unmap();
    if timing_enabled() {
        eprintln!(
            "  wgpu-mesh readback: submit {submit_ms:.2} ms, gpu-wait {wait_ms:.2} ms, copy {:.2} ms ({size} B)",
            ms(t2)
        );
    }
    Ok(out)
}

impl MeshSession {
    pub fn open(positions: &[[f32; 3]], indices: &[u32], edges: &[u32]) -> Result<MeshSession> {
        let g = gpu().map_err(|e| anyhow!("{e}"))?;
        let pipes = pipes(g).map_err(|e| anyhow!("{e}"))?;
        ensure!(indices.len() % 3 == 0, "indices not a multiple of 3");
        ensure!(edges.len() % 3 == 0, "edge triples malformed");
        ensure!(!positions.is_empty() && !indices.is_empty(), "empty mesh");
        let (mn, ext) = crate::gpu_mesh_dispatch::bounds_of(positions)
            .ok_or_else(|| anyhow!("degenerate mesh extent"))?;
        let limits = g.device.limits();
        let max_threads = limits.max_compute_workgroups_per_dimension as u64 * WG;
        ensure!(
            (positions.len() as u64) <= max_threads
                && (indices.len() as u64 / 3) <= max_threads
                && (edges.len() as u64 / 3) <= max_threads,
            "mesh too large for one wgpu dispatch chain"
        );
        let pos_bytes = std::mem::size_of_val(positions) as u64;
        ensure!(
            pos_bytes.max(indices.len() as u64 * 4)
                <= limits.max_storage_buffer_binding_size,
            "mesh buffers exceed wgpu storage binding limit"
        );
        let t0 = std::time::Instant::now();
        let d_pos = storage_init(g, "mesh-positions", as_bytes(positions));
        let d_idx = storage_init(g, "mesh-indices", as_bytes(indices));
        let d_edges = if edges.is_empty() {
            storage_zeroed(g, "mesh-edges-empty", 4, ::wgpu::BufferUsages::empty())
        } else {
            storage_init(g, "mesh-edges", as_bytes(edges))
        };
        // Session-lifetime pools, sized once here. Contents are re-initialized
        // (or fully overwritten) at each use; see survey_inner/cluster_inner.
        let d_grids = storage_zeroed(g, "mesh-grids", MAX_SCALES * 32, ::wgpu::BufferUsages::COPY_DST);
        let d_counts = storage_zeroed(
            g,
            "mesh-counts",
            MAX_SCALES * 4,
            ::wgpu::BufferUsages::COPY_SRC | ::wgpu::BufferUsages::COPY_DST,
        );
        // vert_cost needs no per-call re-init: mesh_pick_cost writes every
        // element < nverts before mesh_pick_id reads any of them.
        let d_vert_cost = storage_zeroed(
            g,
            "mesh-vert-cost",
            positions.len() as u64 * 4,
            ::wgpu::BufferUsages::empty(),
        );
        let d_out = storage_zeroed(
            g,
            "mesh-out-tris",
            indices.len() as u64 * 4,
            ::wgpu::BufferUsages::COPY_SRC | ::wgpu::BufferUsages::COPY_DST,
        );
        let staging = g.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("mesh-staging"),
            size: (indices.len() as u64 * 4).max(MAX_SCALES * 4),
            usage: ::wgpu::BufferUsages::MAP_READ | ::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if timing_enabled() {
            eprintln!(
                "  wgpu-mesh open: upload {:.2} ms ({} B)",
                ms(t0),
                std::mem::size_of_val(positions)
                    + std::mem::size_of_val(indices)
                    + std::mem::size_of_val(edges)
            );
        }
        Ok(MeshSession {
            g,
            pipes,
            d_pos,
            d_idx,
            d_edges,
            d_grids,
            d_counts,
            d_vert_cost,
            d_out,
            staging,
            cells: None,
            nverts: positions.len(),
            ntris: indices.len() / 3,
            nedges: edges.len() / 3,
            mn,
            ext,
        })
    }

    pub fn extent(&self) -> [f32; 3] {
        self.ext
    }

    fn params(&self, scale: Option<f32>, nscales: u32) -> (Params, u64) {
        let (dims, inv_cell, ncells) = match scale {
            Some(s) => {
                let (dims, inv_cell) = mc::scale_grid(self.ext, s);
                let ncells = mc::ncells_of(dims);
                (dims, inv_cell, ncells)
            }
            None => ([1, 1, 1], [0.0; 3], 1),
        };
        (
            Params {
                mn: self.mn,
                inv_ext: 1.0 / mc::max_ext(self.ext),
                inv_cell,
                nverts: self.nverts as u32,
                dims,
                ntris: self.ntris as u32,
                nedges: self.nedges as u32,
                ncells: ncells as u32,
                nscales,
            },
            ncells,
        )
    }

    pub fn survey(&mut self, scales: &[f32]) -> Result<Vec<u32>> {
        ensure!(
            !scales.is_empty() && scales.len() <= 64,
            "bad survey ladder"
        );
        let g = self.g;
        scoped(g, || self.survey_inner(scales))
    }

    fn survey_inner(&self, scales: &[f32]) -> Result<Vec<u32>> {
        let g = self.g;
        // host-side grids, bit-identical to the CPU oracle's
        let mut grid_words: Vec<u32> = Vec::with_capacity(scales.len() * 8);
        for &s in scales {
            let (dims, inv_cell) = mc::scale_grid(self.ext, s);
            grid_words.extend_from_slice(&[
                dims[0],
                dims[1],
                dims[2],
                0,
                inv_cell[0].to_bits(),
                inv_cell[1].to_bits(),
                inv_cell[2].to_bits(),
                0,
            ]);
        }
        let t0 = std::time::Instant::now();
        g.queue
            .write_buffer(&self.d_grids, 0, as_bytes(&grid_words));
        let (params, _) = self.params(None, scales.len() as u32);
        let bufs_ms = ms(t0);
        let t1 = std::time::Instant::now();
        let mut timer = GpuTimer::new(g);
        let mut enc = g.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.d_counts, 0, None);
        run_stage(
            g,
            &mut enc,
            &self.pipes.survey,
            &params,
            self.ntris as u64,
            &[
                (1, &self.d_pos),
                (2, &self.d_idx),
                (4, &self.d_grids),
                (5, &self.d_counts),
            ],
            &mut timer,
            "survey",
        );
        if timing_enabled() {
            eprintln!(
                "  wgpu-mesh survey: bufs {bufs_ms:.2} ms, encode {:.2} ms (ntris {}, nscales {})",
                ms(t1),
                self.ntris,
                scales.len()
            );
        }
        let raw = readback_via(g, enc, &self.d_counts, &self.staging, scales.len() as u64 * 4)?;
        if let Some(t) = timer {
            t.report(g, "survey")?;
        }
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn cluster(&mut self, scale: f32) -> Result<Vec<u32>> {
        let g = self.g;
        scoped(g, || self.cluster_inner(scale))
    }

    fn ensure_cells(&mut self, ncells: u64) {
        let g = self.g;
        if self.cells.as_ref().map(|c| c.cap_cells < ncells).unwrap_or(true) {
            self.cells = Some(CellBufs {
                cap_cells: ncells,
                cell_q: storage_zeroed(g, "mesh-cell-q", ncells * 80, ::wgpu::BufferUsages::COPY_DST),
                best_cost: storage_zeroed(g, "mesh-best-cost", ncells * 4, ::wgpu::BufferUsages::empty()),
                best_id: storage_zeroed(g, "mesh-best-id", ncells * 4, ::wgpu::BufferUsages::empty()),
            });
        }
    }

    fn cluster_inner(&mut self, scale: f32) -> Result<Vec<u32>> {
        let g = self.g;
        let (params, ncells) = self.params(Some(scale), 0);
        ensure!(
            ncells <= NCELLS_CLUSTER_CAP,
            "cluster grid too large: {ncells} cells at scale {scale}"
        );
        let limits = g.device.limits();
        let need = (ncells * 80).max(self.ntris as u64 * 12);
        ensure!(
            need <= limits.max_storage_buffer_binding_size && need <= limits.max_buffer_size,
            "cluster buffers ({need} B) exceed wgpu device limits"
        );
        let t0 = std::time::Instant::now();
        self.ensure_cells(ncells);
        let cells = self.cells.as_ref().expect("cells ensured above");
        let bufs_ms = ms(t0);
        let t1 = std::time::Instant::now();
        let mut timer = GpuTimer::new(g);
        let mut enc = g.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&cells.cell_q, 0, Some(ncells * 80));
        enc.clear_buffer(&self.d_out, 0, None);
        run_stage(
            g,
            &mut enc,
            &self.pipes.fill_ff,
            &params,
            ncells,
            &[(7, &cells.best_cost), (9, &cells.best_id)],
            &mut timer,
            "fill_ff",
        );
        run_stage(
            g,
            &mut enc,
            &self.pipes.accum,
            &params,
            self.ntris as u64,
            &[(1, &self.d_pos), (2, &self.d_idx), (6, &cells.cell_q)],
            &mut timer,
            "accum",
        );
        if self.nedges > 0 {
            run_stage(
                g,
                &mut enc,
                &self.pipes.accum_edges,
                &params,
                self.nedges as u64,
                &[(1, &self.d_pos), (3, &self.d_edges), (6, &cells.cell_q)],
                &mut timer,
                "accum_edges",
            );
        }
        run_stage(
            g,
            &mut enc,
            &self.pipes.pick_cost,
            &params,
            self.nverts as u64,
            &[
                (1, &self.d_pos),
                (6, &cells.cell_q),
                (7, &cells.best_cost),
                (8, &self.d_vert_cost),
            ],
            &mut timer,
            "pick_cost",
        );
        run_stage(
            g,
            &mut enc,
            &self.pipes.pick_id,
            &params,
            self.nverts as u64,
            &[
                (1, &self.d_pos),
                (7, &cells.best_cost),
                (8, &self.d_vert_cost),
                (9, &cells.best_id),
            ],
            &mut timer,
            "pick_id",
        );
        run_stage(
            g,
            &mut enc,
            &self.pipes.remap,
            &params,
            self.ntris as u64,
            &[
                (1, &self.d_pos),
                (2, &self.d_idx),
                (9, &cells.best_id),
                (10, &self.d_out),
            ],
            &mut timer,
            "remap",
        );
        if timing_enabled() {
            eprintln!(
                "  wgpu-mesh cluster: bufs {bufs_ms:.2} ms, encode {:.2} ms (ntris {}, ncells {ncells})",
                ms(t1),
                self.ntris
            );
        }
        let raw = readback_via(g, enc, &self.d_out, &self.staging, self.ntris as u64 * 12)?;
        if let Some(t) = timer {
            t.report(g, "cluster")?;
        }
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}
