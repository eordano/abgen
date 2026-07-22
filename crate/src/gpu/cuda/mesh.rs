use super::*;
use crate::gpu::corelib::mesh_coarsen as mc;

const BLOCK: u32 = 256;
const NCELLS_CLUSTER_CAP: u64 = 8_388_608;

pub fn mesh_kernels_available() -> bool {
    match gpu() {
        Ok(g) => g.has_mesh,
        Err(_) => false,
    }
}

pub struct MeshSession {
    g: &'static Gpu,
    d_pos: DevPtr,
    d_idx: DevPtr,
    d_edges: DevPtr,
    nverts: usize,
    ntris: usize,
    nedges: usize,
    mn: [f32; 3],
    ext: [f32; 3],
}

fn bounds(positions: &[[f32; 3]]) -> Result<([f32; 3], [f32; 3])> {
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    ensure!(mn[0].is_finite() && mx[0].is_finite(), "empty mesh");
    let ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    ensure!(mc::max_ext(ext) > 0.0, "degenerate mesh extent");
    Ok((mn, ext))
}

fn as_bytes<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), std::mem::size_of_val(s)) }
}

impl MeshSession {
    pub fn open(positions: &[[f32; 3]], indices: &[u32], edges: &[u32]) -> Result<MeshSession> {
        let g = gpu()?;
        ensure!(g.has_mesh, "mesh kernels not present in loaded PTX");
        ensure!(indices.len() % 3 == 0, "indices not a multiple of 3");
        ensure!(edges.len() % 3 == 0, "edge triples malformed");
        let (mn, ext) = bounds(positions)?;
        unsafe {
            g.check((g.ctx_set_current)(g.ctx))?;
            let d_pos = g.alloc_upload(as_bytes(positions))?;
            let d_idx = match g.alloc_upload(as_bytes(indices)) {
                Ok(d) => d,
                Err(e) => {
                    let _ = (g.mem_free)(d_pos);
                    return Err(e);
                }
            };
            let d_edges = if edges.is_empty() {
                0
            } else {
                match g.alloc_upload(as_bytes(edges)) {
                    Ok(d) => d,
                    Err(e) => {
                        let _ = (g.mem_free)(d_pos);
                        let _ = (g.mem_free)(d_idx);
                        return Err(e);
                    }
                }
            };
            Ok(MeshSession {
                g,
                d_pos,
                d_idx,
                d_edges,
                nverts: positions.len(),
                ntris: indices.len() / 3,
                nedges: edges.len() / 3,
                mn,
                ext,
            })
        }
    }

    pub fn extent(&self) -> [f32; 3] {
        self.ext
    }

    pub fn survey(&mut self, scales: &[f32]) -> Result<Vec<u32>> {
        ensure!(
            !scales.is_empty() && scales.len() <= 64,
            "bad survey ladder"
        );
        let g = self.g;
        let params = mc::SurveyParams {
            mn: self.mn,
            ntris: self.ntris as u32,
            ext: self.ext,
            nscales: scales.len() as u32,
        };
        unsafe {
            g.check((g.ctx_set_current)(g.ctx))?;
            let d_params = g.alloc_upload(as_bytes(std::slice::from_ref(&params)))?;
            let run = || -> Result<(DevPtr, DevPtr)> {
                let d_scales = g.alloc_upload(as_bytes(scales))?;
                let d_counts = g.alloc_dev(scales.len() * 4, "mesh-counts").inspect_err(|_| {
                    let _ = (g.mem_free)(d_scales);
                })?;
                Ok((d_scales, d_counts))
            };
            let (d_scales, d_counts) = run().inspect_err(|_| {
                let _ = (g.mem_free)(d_params);
            })?;
            let work = || -> Result<Vec<u32>> {
                g.check((g.memset_d8)(d_counts, 0, scales.len() * 4))?;
                let grid = (self.ntris as u32).div_ceil(BLOCK).max(1);
                let mut args = [d_params, self.d_pos, self.d_idx, d_scales, d_counts];
                g.launch_u64s(g.func_mesh_survey, grid, BLOCK, &mut args)?;
                g.check((g.ctx_synchronize)())?;
                let mut counts = vec![0u32; scales.len()];
                g.check((g.memcpy_dtoh)(
                    counts.as_mut_ptr().cast(),
                    d_counts,
                    scales.len() * 4,
                ))?;
                Ok(counts)
            };
            let out = work();
            for d in [d_params, d_scales, d_counts] {
                let _ = (g.mem_free)(d);
            }
            out
        }
    }

    pub fn cluster(&mut self, scale: f32) -> Result<Vec<u32>> {
        let g = self.g;
        let (dims, inv_cell) = mc::scale_grid(self.ext, scale);
        let ncells = mc::ncells_of(dims);
        ensure!(
            ncells <= NCELLS_CLUSTER_CAP,
            "cluster grid too large: {ncells} cells at scale {scale}"
        );
        let params = mc::CoarsenParams {
            mn: self.mn,
            inv_ext: 1.0 / mc::max_ext(self.ext),
            inv_cell,
            nverts: self.nverts as u32,
            dims,
            ntris: self.ntris as u32,
            nedges: self.nedges as u32,
            ncells: ncells as u32,
            pad: [0, 0],
        };
        unsafe {
            g.check((g.ctx_set_current)(g.ctx))?;
            let mut d_q: DevPtr = 0;
            let mut d_best: DevPtr = 0;
            let mut d_out: DevPtr = 0;
            let mut d_params: DevPtr = 0;
            let alloc = |p: &mut DevPtr, bytes: usize| -> Result<()> {
                *p = g.alloc_dev(bytes, "mesh-cluster")?;
                Ok(())
            };
            let work = |d_q: &mut DevPtr,
                        d_best: &mut DevPtr,
                        d_out: &mut DevPtr,
                        d_params: &mut DevPtr|
             -> Result<Vec<u32>> {
                alloc(d_q, ncells as usize * 80)?;
                alloc(d_best, ncells as usize * 8)?;
                alloc(d_out, self.ntris * 12)?;
                *d_params = g.alloc_upload(as_bytes(std::slice::from_ref(&params)))?;
                g.check((g.memset_d8)(*d_q, 0, ncells as usize * 80))?;
                g.check((g.memset_d8)(*d_best, 0xff, ncells as usize * 8))?;
                let tri_grid = (self.ntris as u32).div_ceil(BLOCK).max(1);
                let vert_grid = (self.nverts as u32).div_ceil(BLOCK).max(1);
                let mut a1 = [*d_params, self.d_pos, self.d_idx, *d_q];
                g.launch_u64s(g.func_mesh_accum, tri_grid, BLOCK, &mut a1)?;
                if self.nedges > 0 {
                    let edge_grid = (self.nedges as u32).div_ceil(BLOCK).max(1);
                    let mut a2 = [*d_params, self.d_pos, self.d_edges, *d_q];
                    g.launch_u64s(g.func_mesh_accum_edges, edge_grid, BLOCK, &mut a2)?;
                }
                let mut a3 = [*d_params, self.d_pos, *d_q, *d_best];
                g.launch_u64s(g.func_mesh_pick, vert_grid, BLOCK, &mut a3)?;
                let mut a4 = [*d_params, self.d_pos, self.d_idx, *d_best, *d_out];
                g.launch_u64s(g.func_mesh_remap, tri_grid, BLOCK, &mut a4)?;
                g.check((g.ctx_synchronize)())?;
                let mut out = vec![0u32; self.ntris * 3];
                g.check((g.memcpy_dtoh)(
                    out.as_mut_ptr().cast(),
                    *d_out,
                    self.ntris * 12,
                ))?;
                Ok(out)
            };
            let out = work(&mut d_q, &mut d_best, &mut d_out, &mut d_params);
            for d in [d_q, d_best, d_out, d_params] {
                if d != 0 {
                    let _ = (g.mem_free)(d);
                }
            }
            out
        }
    }
}

impl Drop for MeshSession {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.g.ctx_set_current)(self.g.ctx);
            for d in [self.d_pos, self.d_idx, self.d_edges] {
                if d != 0 {
                    let _ = (self.g.mem_free)(d);
                }
            }
        }
    }
}
