use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gpu::corelib::mesh_coarsen as mc;
use crate::lodgen::model::{LodModel, LodPrimitive};
use crate::lodgen::simplify_meshopt::apportion;

pub const LADDER: usize = 32;
const NCELLS_CLUSTER_CAP: u64 = 8_388_608;
const PRIM_MIN_TRIS: usize = 10_000;
const DEFAULT_COARSEN_K: u64 = 4;
const DEFAULT_MIN_TOTAL_TRIS: usize = 200_000;

static COARSEN_WARNED: AtomicBool = AtomicBool::new(false);
static DIRECT_WARNED: AtomicBool = AtomicBool::new(false);
static MODE_WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshMode {
    Coarsen,
    Direct,
    Off,
}

pub fn parse_mesh_mode(raw: Option<&str>) -> std::result::Result<MeshMode, String> {
    match raw {
        None | Some("") | Some("coarsen") => Ok(MeshMode::Coarsen),
        Some("direct") => Ok(MeshMode::Direct),
        Some("off") | Some("0") => Ok(MeshMode::Off),
        Some(other) => Err(format!(
            "unknown ABGEN_GPU_MESH_MODE value {other:?} (expected coarsen|direct|off)"
        )),
    }
}

fn mesh_mode() -> MeshMode {
    let raw = std::env::var("ABGEN_GPU_MESH_MODE").ok();
    match parse_mesh_mode(raw.as_deref()) {
        Ok(m) => m,
        Err(e) => {
            if !MODE_WARNED.swap(true, Ordering::Relaxed) {
                eprintln!("warn: {e}; gpu mesh path disabled");
            }
            MeshMode::Off
        }
    }
}

fn env_u64(name: &str, default: u64, lo: u64, hi: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .map(|v: u64| v.clamp(lo, hi))
        .unwrap_or(default)
}

fn coarsen_k() -> u64 {
    env_u64("ABGEN_GPU_MESH_COARSEN_K", DEFAULT_COARSEN_K, 2, 64)
}

fn min_total_tris() -> usize {
    env_u64(
        "ABGEN_GPU_MESH_MIN_TRIS",
        DEFAULT_MIN_TOTAL_TRIS as u64,
        0,
        u64::MAX >> 1,
    ) as usize
}

fn boundary_enabled() -> bool {
    crate::clihelp::env_bool("ABGEN_GPU_MESH_BOUNDARY", true)
}

pub trait CoarsenBackend {
    fn survey(&mut self, scales: &[f32]) -> Result<Vec<u32>>;
    fn cluster(&mut self, scale: f32) -> Result<Vec<u32>>;
}

pub struct CpuBackend<'a> {
    positions: &'a [[f32; 3]],
    indices: &'a [u32],
    edges: &'a [u32],
    mn: [f32; 3],
    ext: [f32; 3],
}

pub fn bounds_of(positions: &[[f32; 3]]) -> Option<([f32; 3], [f32; 3])> {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    if !mn[0].is_finite() || !mx[0].is_finite() {
        return None;
    }
    let ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    if mc::max_ext(ext) <= 0.0 {
        return None;
    }
    Some((mn, ext))
}

impl<'a> CpuBackend<'a> {
    pub fn new(
        positions: &'a [[f32; 3]],
        indices: &'a [u32],
        edges: &'a [u32],
    ) -> Option<CpuBackend<'a>> {
        let (mn, ext) = bounds_of(positions)?;
        Some(CpuBackend {
            positions,
            indices,
            edges,
            mn,
            ext,
        })
    }

    pub fn extent(&self) -> [f32; 3] {
        self.ext
    }
}

impl CoarsenBackend for CpuBackend<'_> {
    fn survey(&mut self, scales: &[f32]) -> Result<Vec<u32>> {
        let grids: Vec<([u32; 3], [f32; 3])> = scales
            .iter()
            .map(|&s| mc::scale_grid(self.ext, s))
            .collect();
        let mn = self.mn;
        let counts = self
            .indices
            .par_chunks(3 * 4096)
            .map(|chunk| {
                let mut local = vec![0u32; grids.len()];
                for t in chunk.chunks_exact(3) {
                    let a = self.positions[t[0] as usize];
                    let b = self.positions[t[1] as usize];
                    let c = self.positions[t[2] as usize];
                    for (s, &(dims, inv_cell)) in grids.iter().enumerate() {
                        let ca = mc::cell_index(a, mn, inv_cell, dims);
                        let cb = mc::cell_index(b, mn, inv_cell, dims);
                        let cc = mc::cell_index(c, mn, inv_cell, dims);
                        if mc::tri_survives(ca, cb, cc) {
                            local[s] += 1;
                        }
                    }
                }
                local
            })
            .reduce(
                || vec![0u32; grids.len()],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        Ok(counts)
    }

    fn cluster(&mut self, scale: f32) -> Result<Vec<u32>> {
        let (dims, inv_cell) = mc::scale_grid(self.ext, scale);
        let ncells = mc::ncells_of(dims);
        ensure!(
            ncells <= NCELLS_CLUSTER_CAP,
            "cluster grid too large: {ncells} cells at scale {scale}"
        );
        let ncells = ncells as usize;
        let inv_ext = 1.0 / mc::max_ext(self.ext);
        let mut cell_q = vec![0i64; ncells * 10];
        for t in self.indices.chunks_exact(3) {
            let a = self.positions[t[0] as usize];
            let b = self.positions[t[1] as usize];
            let c = self.positions[t[2] as usize];
            if let Some(q) = mc::accum_tri_quadric(a, b, c, self.mn, inv_ext) {
                for &p in &[a, b, c] {
                    let base = mc::cell_index(p, self.mn, inv_cell, dims) as usize * 10;
                    for i in 0..10 {
                        cell_q[base + i] = cell_q[base + i].wrapping_add(q[i]);
                    }
                }
            }
        }
        for e in self.edges.chunks_exact(3) {
            let u = self.positions[e[0] as usize];
            let v = self.positions[e[1] as usize];
            let w = self.positions[e[2] as usize];
            if let Some(q) = mc::accum_edge_quadric(u, v, w, self.mn, inv_ext) {
                for &p in &[u, v] {
                    let base = mc::cell_index(p, self.mn, inv_cell, dims) as usize * 10;
                    for i in 0..10 {
                        cell_q[base + i] = cell_q[base + i].wrapping_add(q[i]);
                    }
                }
            }
        }
        let mut best = vec![u64::MAX; ncells];
        for (vid, &p) in self.positions.iter().enumerate() {
            let cid = mc::cell_index(p, self.mn, inv_cell, dims) as usize;
            let q: &[i64; 10] = cell_q[cid * 10..cid * 10 + 10].try_into().unwrap();
            let cost = mc::eval_cost_fp(q, mc::norm_pos(p, self.mn, inv_ext));
            let cand = mc::pack_cost_id(cost, vid as u32);
            if cand < best[cid] {
                best[cid] = cand;
            }
        }
        let ntris = self.indices.len() / 3;
        let mut out = vec![0u32; ntris * 3];
        for (t, tri) in self.indices.chunks_exact(3).enumerate() {
            let a = self.positions[tri[0] as usize];
            let b = self.positions[tri[1] as usize];
            let c = self.positions[tri[2] as usize];
            let ca = mc::cell_index(a, self.mn, inv_cell, dims);
            let cb = mc::cell_index(b, self.mn, inv_cell, dims);
            let cc = mc::cell_index(c, self.mn, inv_cell, dims);
            if mc::tri_survives(ca, cb, cc) {
                out[t * 3] = (best[ca as usize] & 0xffff_ffff) as u32;
                out[t * 3 + 1] = (best[cb as usize] & 0xffff_ffff) as u32;
                out[t * 3 + 2] = (best[cc as usize] & 0xffff_ffff) as u32;
            } else {
                out[t * 3] = mc::CULLED;
            }
        }
        Ok(out)
    }
}

fn scale_cluster_cap(ext: [f32; 3]) -> f32 {
    let (mut lo, mut hi) = (2.0f32, mc::MAX_SCALE);
    if mc::ncells_of(mc::scale_grid(ext, hi).0) <= NCELLS_CLUSTER_CAP {
        return hi;
    }
    for _ in 0..24 {
        let mid = (lo + hi) / 2.0;
        if mc::ncells_of(mc::scale_grid(ext, mid).0) <= NCELLS_CLUSTER_CAP {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn ladder_geometric(smax: f32) -> Vec<f32> {
    let smax = smax.max(2.0);
    let ratio = (smax / 2.0).max(1.0);
    (0..LADDER)
        .map(|i| 2.0 * ratio.powf(i as f32 / (LADDER - 1) as f32))
        .collect()
}

fn ladder_linear(lo: f32, hi: f32) -> Vec<f32> {
    (1..=LADDER)
        .map(|i| lo + (hi - lo) * i as f32 / (LADDER + 1) as f32)
        .collect()
}

pub fn choose_scale<B: CoarsenBackend>(b: &mut B, ext: [f32; 3], target: usize) -> Result<f32> {
    let smax = scale_cluster_cap(ext);
    let s1 = ladder_geometric(smax);
    let c1 = b.survey(&s1)?;
    let mut best: (u32, f32) = (0, 1.0);
    let mut bracket_lo = 1.0f32;
    let mut bracket_hi: Option<f32> = Some(s1[0]);
    for (i, (&s, &c)) in s1.iter().zip(c1.iter()).enumerate() {
        if c as usize <= target {
            if c > best.0 || (c == best.0 && s < best.1) {
                best = (c, s);
            }
            bracket_lo = s;
            bracket_hi = if i + 1 < s1.len() {
                Some(s1[i + 1])
            } else {
                None
            };
        }
    }
    if let Some(hi) = bracket_hi {
        let s2 = ladder_linear(bracket_lo, hi);
        let c2 = b.survey(&s2)?;
        for (&s, &c) in s2.iter().zip(c2.iter()) {
            if c as usize <= target && (c > best.0 || (c == best.0 && s < best.1)) {
                best = (c, s);
            }
        }
    }
    Ok(best.1)
}

pub struct CoarsenOut {
    pub tris: Vec<u32>,
    pub scale: f32,
    pub survivors: usize,
}

pub fn coarsen_tris_at<B: CoarsenBackend>(b: &mut B, scale: f32) -> Result<CoarsenOut> {
    let raw = b.cluster(scale)?;
    let mut tris = Vec::new();
    for t in raw.chunks_exact(3) {
        if t[0] != mc::CULLED {
            tris.extend_from_slice(t);
        }
    }
    let survivors = tris.len() / 3;
    Ok(CoarsenOut {
        tris,
        scale,
        survivors,
    })
}

pub fn coarsen_tris<B: CoarsenBackend>(
    b: &mut B,
    ext: [f32; 3],
    target: usize,
) -> Result<CoarsenOut> {
    let scale = choose_scale(b, ext, target)?;
    let out = coarsen_tris_at(b, scale)?;
    ensure!(
        out.survivors <= target,
        "cluster overshot: {} tris > target {target} at scale {scale}",
        out.survivors
    );
    Ok(out)
}

pub fn boundary_edges(indices: &[u32]) -> Vec<u32> {
    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(indices.len());
    for t in indices.chunks_exact(3) {
        for e in 0..3 {
            let u = t[e];
            let v = t[(e + 1) % 3];
            let w = t[(e + 2) % 3];
            let key = if u < v {
                ((u as u64) << 32) | v as u64
            } else {
                ((v as u64) << 32) | u as u64
            };
            keys.push((key, w));
        }
    }
    keys.par_sort_unstable();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j].0 == keys[i].0 {
            j += 1;
        }
        if j - i == 1 {
            let (key, w) = keys[i];
            out.push((key >> 32) as u32);
            out.push((key & 0xffff_ffff) as u32);
            out.push(w);
        }
        i = j;
    }
    out
}

pub fn rebuild_prim(src: &LodPrimitive, tris: &[u32]) -> LodPrimitive {
    let mut map: HashMap<u32, u32> = HashMap::with_capacity(tris.len());
    let mut order: Vec<u32> = Vec::with_capacity(tris.len());
    let mut indices = Vec::with_capacity(tris.len());
    for &orig in tris {
        let next = order.len() as u32;
        let id = *map.entry(orig).or_insert_with(|| {
            order.push(orig);
            next
        });
        indices.push(id);
    }
    let gather3 = |v: &Vec<[f32; 3]>| -> Vec<[f32; 3]> {
        if v.is_empty() {
            Vec::new()
        } else {
            order.iter().map(|&i| v[i as usize]).collect()
        }
    };
    let gather2 = |v: &Vec<[f32; 2]>| -> Vec<[f32; 2]> {
        if v.is_empty() {
            Vec::new()
        } else {
            order.iter().map(|&i| v[i as usize]).collect()
        }
    };
    let gather4 = |v: &Vec<[f32; 4]>| -> Vec<[f32; 4]> {
        if v.is_empty() {
            Vec::new()
        } else {
            order.iter().map(|&i| v[i as usize]).collect()
        }
    };
    LodPrimitive {
        positions: gather3(&src.positions),
        normals: gather3(&src.normals),
        uvs: gather2(&src.uvs),
        tangents: gather4(&src.tangents),
        colors: gather4(&src.colors),
        indices,
        material: src.material,
    }
}

pub fn coarsen_prim_cpu(prim: &LodPrimitive, target: u64) -> Result<(LodPrimitive, f32)> {
    let target = target as usize;
    if prim.indices.len() / 3 <= target {
        return Ok((prim.clone(), 0.0));
    }
    let edges = if boundary_enabled() {
        boundary_edges(&prim.indices)
    } else {
        Vec::new()
    };
    let mut b = CpuBackend::new(&prim.positions, &prim.indices, &edges)
        .ok_or_else(|| anyhow::anyhow!("degenerate primitive extent"))?;
    let ext = b.extent();
    let out = coarsen_tris(&mut b, ext, target)?;
    Ok((rebuild_prim(prim, &out.tris), out.scale))
}

/// Which GPU backend serves the mesh lane. CUDA stays the default; wgpu is
/// used only on explicit ABGEN_GPU_BACKEND=wgpu (portable, CUDA-free), so
/// auto behavior is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshGpu {
    Cuda,
    Wgpu,
}

fn mesh_gpu_backend() -> Option<MeshGpu> {
    if crate::gpu::wgpu_backend_selected() {
        crate::gpu::wgpu_mesh::mesh_kernels_available().then_some(MeshGpu::Wgpu)
    } else {
        crate::gpu::cuda::mesh::mesh_kernels_available().then_some(MeshGpu::Cuda)
    }
}

fn coarsen_prim_gpu(prim: &LodPrimitive, target: u64, which: MeshGpu) -> Result<(LodPrimitive, f32)> {
    let target = target as usize;
    if prim.indices.len() / 3 <= target {
        return Ok((prim.clone(), 0.0));
    }
    let edges = if boundary_enabled() {
        boundary_edges(&prim.indices)
    } else {
        Vec::new()
    };
    let out = match which {
        MeshGpu::Cuda => {
            let mut s =
                crate::gpu::cuda::mesh::MeshSession::open(&prim.positions, &prim.indices, &edges)?;
            let ext = s.extent();
            coarsen_tris(&mut s, ext, target)?
        }
        MeshGpu::Wgpu => {
            let mut s =
                crate::gpu::wgpu_mesh::MeshSession::open(&prim.positions, &prim.indices, &edges)?;
            let ext = s.extent();
            coarsen_tris(&mut s, ext, target)?
        }
    };
    Ok((rebuild_prim(prim, &out.tris), out.scale))
}

impl CoarsenBackend for crate::gpu::cuda::mesh::MeshSession {
    fn survey(&mut self, scales: &[f32]) -> Result<Vec<u32>> {
        crate::gpu::cuda::mesh::MeshSession::survey(self, scales)
    }
    fn cluster(&mut self, scale: f32) -> Result<Vec<u32>> {
        crate::gpu::cuda::mesh::MeshSession::cluster(self, scale)
    }
}

impl CoarsenBackend for crate::gpu::wgpu_mesh::MeshSession {
    fn survey(&mut self, scales: &[f32]) -> Result<Vec<u32>> {
        crate::gpu::wgpu_mesh::MeshSession::survey(self, scales)
    }
    fn cluster(&mut self, scale: f32) -> Result<Vec<u32>> {
        crate::gpu::wgpu_mesh::MeshSession::cluster(self, scale)
    }
}

pub fn coarsen_model_with(
    model: &LodModel,
    cap: u64,
    k: u64,
    min_tris: usize,
    coarsen: &mut dyn FnMut(&LodPrimitive, u64) -> Result<(LodPrimitive, f32)>,
) -> Option<LodModel> {
    let budget = cap.saturating_mul(k).max(cap);
    if budget == 0 {
        return None;
    }
    let total = model.total_tris();
    if total <= min_tris.max(budget as usize) {
        return None;
    }
    let counts: Vec<usize> = model
        .primitives
        .iter()
        .map(|p| p.indices.len() / 3)
        .collect();
    let targets = apportion(&counts, budget);
    let t0 = std::time::Instant::now();
    let mut prims = Vec::with_capacity(model.primitives.len());
    let mut scales = Vec::new();
    for (prim, &target) in model.primitives.iter().zip(targets.iter()) {
        let count = prim.indices.len() / 3;
        if count <= target as usize || count < PRIM_MIN_TRIS {
            prims.push(prim.clone());
            continue;
        }
        match coarsen(prim, target) {
            Ok((p, scale)) => {
                scales.push(scale);
                prims.push(p);
            }
            Err(e) => {
                if !COARSEN_WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!("warn: gpu mesh coarsen failed, using meshopt directly: {e:#}");
                }
                return None;
            }
        }
    }
    let out = LodModel {
        root_name: model.root_name.clone(),
        primitives: prims,
        materials: model.materials.clone(),
        images: model.images.clone(),
        log: Vec::new(),
    };
    eprintln!(
        "abgen-gpu: mesh coarsen {} -> {} tris (budget {budget}, scales {:?}) in {} ms",
        total,
        out.total_tris(),
        scales,
        t0.elapsed().as_millis()
    );
    Some(out)
}

pub fn coarsen_for_finish(model: &LodModel, cap: u64) -> Option<LodModel> {
    if !crate::gpu_dispatch::enabled() || mesh_mode() != MeshMode::Coarsen {
        return None;
    }
    let Some(which) = mesh_gpu_backend() else {
        if !COARSEN_WARNED.swap(true, Ordering::Relaxed) {
            eprintln!("warn: gpu mesh kernels unavailable; meshopt lane unchanged");
        }
        return None;
    };
    coarsen_model_with(
        model,
        cap,
        coarsen_k(),
        min_total_tris(),
        &mut |prim, target| coarsen_prim_gpu(prim, target, which),
    )
}

pub fn direct_prim_cpu(prim: &LodPrimitive, target_tris: u64) -> Result<LodPrimitive> {
    coarsen_prim_cpu(prim, target_tris).map(|(p, _)| p)
}

pub fn direct_decimate(prim: &LodPrimitive, target_tris: u64) -> Option<(LodPrimitive, bool)> {
    if !crate::gpu_dispatch::enabled() || mesh_mode() != MeshMode::Direct {
        return None;
    }
    let Some(which) = mesh_gpu_backend() else {
        if !DIRECT_WARNED.swap(true, Ordering::Relaxed) {
            eprintln!("warn: gpu mesh kernels unavailable; meshopt lane unchanged");
        }
        return None;
    };
    if prim.indices.len() / 3 < PRIM_MIN_TRIS {
        return None;
    }
    match coarsen_prim_gpu(prim, target_tris, which) {
        Ok((p, _)) => Some((p, true)),
        Err(e) => {
            if !DIRECT_WARNED.swap(true, Ordering::Relaxed) {
                eprintln!("warn: gpu mesh direct decimate failed, falling back to meshopt: {e:#}");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lodgen::emit::emit_glb;
    use crate::lodgen::model::{AlphaClass, LodMaterial};

    fn grid_prim(n: u32, flat: bool) -> LodPrimitive {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let x = i as f32 / n as f32;
                let z = j as f32 / n as f32;
                let y = if flat {
                    0.0
                } else {
                    0.05 * ((x * 12.0).sin() + (z * 12.0).cos())
                        * (1.0 + 0.3 * ((x * 5.0 + z * 7.0).sin()))
                };
                positions.push([x * 10.0, y, z * 10.0]);
                normals.push([0.0, 1.0, 0.0]);
                uvs.push([x, z]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                let b = a + 1;
                let c = a + n + 1;
                let d = c + 1;
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        LodPrimitive {
            positions,
            normals,
            uvs,
            indices,
            material: 0,
            ..Default::default()
        }
    }

    fn model_of(prim: LodPrimitive) -> LodModel {
        LodModel {
            root_name: "grid".to_string(),
            primitives: vec![prim],
            materials: vec![LodMaterial {
                name: "m".to_string(),
                class: AlphaClass::Opaque,
                base_color: [1.0, 1.0, 1.0, 1.0],
                cutoff: 0.5,
                image: None,
                double_sided: false,
            }],
            images: Vec::new(),
            log: Vec::new(),
        }
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(parse_mesh_mode(None).unwrap(), MeshMode::Coarsen);
        assert_eq!(parse_mesh_mode(Some("")).unwrap(), MeshMode::Coarsen);
        assert_eq!(parse_mesh_mode(Some("coarsen")).unwrap(), MeshMode::Coarsen);
        assert_eq!(parse_mesh_mode(Some("direct")).unwrap(), MeshMode::Direct);
        assert_eq!(parse_mesh_mode(Some("off")).unwrap(), MeshMode::Off);
        assert_eq!(parse_mesh_mode(Some("0")).unwrap(), MeshMode::Off);
        assert!(parse_mesh_mode(Some("wat")).is_err());
    }

    #[test]
    fn boundary_edges_of_grid() {
        let prim = grid_prim(4, true);
        let edges = boundary_edges(&prim.indices);
        assert_eq!(edges.len() % 3, 0);
        assert_eq!(edges.len() / 3, 16);
        for e in edges.chunks_exact(3) {
            assert!(e[0] < e[1]);
            assert!((e[2] as usize) < prim.positions.len());
        }
    }

    #[test]
    fn survey_counts_match_cluster_survivors() {
        let prim = grid_prim(32, false);
        let edges = boundary_edges(&prim.indices);
        let mut b = CpuBackend::new(&prim.positions, &prim.indices, &edges).unwrap();
        let scales = [3.7f32, 8.0, 16.5, 25.0];
        let counts = b.survey(&scales).unwrap();
        for (&s, &c) in scales.iter().zip(counts.iter()) {
            let raw = b.cluster(s).unwrap();
            let survivors = raw.chunks_exact(3).filter(|t| t[0] != mc::CULLED).count();
            assert_eq!(survivors as u32, c, "scale {s}");
        }
    }

    #[test]
    fn fixed_point_accumulation_is_order_independent() {
        let prim = grid_prim(32, false);
        let ntris = prim.indices.len() / 3;
        let mut perm: Vec<usize> = (0..ntris).collect();
        let mut state = 0x9e3779b97f4a7c15u64;
        for i in (1..ntris).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            perm.swap(i, j);
        }
        let mut shuffled = prim.clone();
        shuffled.indices = perm
            .iter()
            .flat_map(|&t| prim.indices[t * 3..t * 3 + 3].to_vec())
            .collect();
        let (a, sa) = coarsen_prim_cpu(&prim, 200).unwrap();
        let (b, sb) = coarsen_prim_cpu(&shuffled, 200).unwrap();
        assert_eq!(sa, sb);
        let key = |p: &LodPrimitive| {
            let mut tris: Vec<[[f32; 3]; 3]> = p
                .indices
                .chunks_exact(3)
                .map(|t| {
                    [
                        p.positions[t[0] as usize],
                        p.positions[t[1] as usize],
                        p.positions[t[2] as usize],
                    ]
                })
                .collect();
            tris.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
            tris
        };
        assert_eq!(key(&a), key(&b));
        let (a2, _) = coarsen_prim_cpu(&prim, 200).unwrap();
        assert_eq!(
            emit_glb(&model_of(a.clone())).unwrap(),
            emit_glb(&model_of(a2)).unwrap()
        );
    }

    #[test]
    fn direct_lands_at_or_under_target_and_keeps_streams_valid() {
        let prim = grid_prim(32, false);
        for target in [60u64, 200, 450] {
            let out = direct_prim_cpu(&prim, target).unwrap();
            let tris = out.indices.len() / 3;
            assert!(tris <= target as usize, "target {target}: {tris}");
            assert!(tris > 0, "target {target}");
            assert_eq!(out.indices.len() % 3, 0);
            assert_eq!(out.normals.len(), out.positions.len());
            assert_eq!(out.uvs.len(), out.positions.len());
            assert!(out.tangents.is_empty());
            assert!(out.colors.is_empty());
            let max = *out.indices.iter().max().unwrap() as usize;
            assert!(max < out.positions.len());
            let mut q = out.clone();
            assert_eq!(q.compact_orphans(), 0);
        }
    }

    #[test]
    fn winding_is_preserved_on_flat_grid() {
        let prim = grid_prim(24, true);
        let out = direct_prim_cpu(&prim, 100).unwrap();
        for t in out.indices.chunks_exact(3) {
            let a = out.positions[t[0] as usize];
            let b = out.positions[t[1] as usize];
            let c = out.positions[t[2] as usize];
            let n = mc::cross3(mc::sub3(b, a), mc::sub3(c, a));
            assert!(n[1] > 0.0, "flipped triangle {t:?}");
        }
    }

    #[test]
    fn rebuild_keeps_original_attributes_by_id() {
        let prim = grid_prim(16, false);
        let (out, _) = coarsen_prim_cpu(&prim, 100).unwrap();
        for (i, p) in out.positions.iter().enumerate() {
            let orig = prim
                .positions
                .iter()
                .position(|q| q == p)
                .expect("rep vertex must be an original vertex");
            assert_eq!(out.normals[i], prim.normals[orig]);
            assert_eq!(out.uvs[i], prim.uvs[orig]);
        }
    }

    #[test]
    fn coarsen_then_meshopt_finish_lands_in_the_prod_window() {
        let model = model_of(grid_prim(128, false));
        assert_eq!(model.total_tris(), 32768);
        let coarse = coarsen_model_with(&model, 500, 4, 0, &mut coarsen_prim_cpu).unwrap();
        assert!(coarse.total_tris() <= 2000);
        assert!(coarse.total_tris() > 500);
        let (out, report) =
            crate::lodgen::simplify_meshopt::simplify_model(&coarse, 500, true).unwrap();
        assert!(report.tris_after <= 500, "{}", report.tris_after);
        assert!(report.tris_after >= 400, "{}", report.tris_after);
        let orphans = out
            .primitives
            .iter()
            .map(|p| {
                let mut q = p.clone();
                q.compact_orphans()
            })
            .sum::<usize>();
        assert_eq!(orphans, 0);
    }

    #[test]
    fn coarsen_model_gating_thresholds() {
        let model = model_of(grid_prim(32, false));
        assert!(coarsen_model_with(&model, 500, 4, 200_000, &mut coarsen_prim_cpu).is_none());
        assert!(coarsen_model_with(&model, 0, 4, 0, &mut coarsen_prim_cpu).is_none());
        assert!(coarsen_model_with(&model, 2048, 4, 0, &mut coarsen_prim_cpu).is_none());
    }

    #[test]
    fn seam_is_inert_without_arming() {
        let prim = grid_prim(32, false);
        assert!(direct_decimate(&prim, 100).is_none());
        let model = model_of(prim);
        assert!(coarsen_for_finish(&model, 500).is_none());
    }
}
