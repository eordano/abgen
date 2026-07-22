use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::time::Instant;

use crate::gpu_mesh_dispatch as gmd;
use crate::lodgen::model::LodPrimitive;

fn grid_prim(quads: u32) -> LodPrimitive {
    let n = quads + 1;
    let mut positions = Vec::with_capacity((n * n) as usize);
    let mut normals = Vec::with_capacity((n * n) as usize);
    let mut uvs = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 / quads as f32;
            let z = j as f32 / quads as f32;
            let y =
                (x * 37.0).sin() * 0.05 + (z * 29.0).cos() * 0.05 + ((x * z) * 91.0).sin() * 0.02;
            positions.push([x * 16.0, y * 16.0, z * 16.0]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x, z]);
        }
    }
    let mut indices = Vec::with_capacity((quads * quads * 6) as usize);
    for j in 0..quads {
        for i in 0..quads {
            let a = j * n + i;
            let b = a + 1;
            let c = a + n;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
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

fn meshopt_simplify(prim: &LodPrimitive, cap: usize) -> Result<(LodPrimitive, f32, f64)> {
    let bytes = meshopt::typed_to_bytes(&prim.positions);
    let adapter = meshopt::VertexDataAdapter::new(bytes, 12, 0)
        .map_err(|e| anyhow!("meshopt vertex adapter: {e}"))?;
    let mut err = 0.0f32;
    let t = Instant::now();
    let indices = meshopt::simplify(
        &prim.indices,
        &adapter,
        cap * 3,
        1.0,
        meshopt::SimplifyOptions::empty(),
        Some(&mut err),
    );
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut out = LodPrimitive {
        positions: prim.positions.clone(),
        normals: prim.normals.clone(),
        uvs: prim.uvs.clone(),
        tangents: prim.tangents.clone(),
        colors: prim.colors.clone(),
        indices,
        material: prim.material,
    };
    out.compact_orphans();
    Ok((out, err, ms))
}

fn tri_areas(prim: &LodPrimitive) -> Vec<f64> {
    prim.indices
        .chunks_exact(3)
        .map(|t| {
            let a = prim.positions[t[0] as usize];
            let b = prim.positions[t[1] as usize];
            let c = prim.positions[t[2] as usize];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            0.5 * ((n[0] as f64).powi(2) + (n[1] as f64).powi(2) + (n[2] as f64).powi(2)).sqrt()
        })
        .collect()
}

fn sample_surface(prim: &LodPrimitive, count: usize, seed: u64) -> Vec<[f32; 3]> {
    let areas = tri_areas(prim);
    let mut cum = Vec::with_capacity(areas.len());
    let mut acc = 0.0f64;
    for a in &areas {
        acc += a;
        cum.push(acc);
    }
    let total = acc.max(1e-30);
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let r = next() * total;
            let t = cum.partition_point(|&c| c < r).min(areas.len() - 1);
            let (mut u, mut v) = (next() as f32, next() as f32);
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let idx = &prim.indices[t * 3..t * 3 + 3];
            let a = prim.positions[idx[0] as usize];
            let b = prim.positions[idx[1] as usize];
            let c = prim.positions[idx[2] as usize];
            let w = 1.0 - u - v;
            [
                a[0] * w + b[0] * u + c[0] * v,
                a[1] * w + b[1] * u + c[1] * v,
                a[2] * w + b[2] * u + c[2] * v,
            ]
        })
        .collect()
}

fn point_tri_dist2(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let sub = |x: [f32; 3], y: [f32; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let dot = |x: [f32; 3], y: [f32; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        let q = [ab[0] * t, ab[1] * t, ab[2] * t];
        let d = sub(ap, q);
        return dot(d, d);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        let q = [ac[0] * t, ac[1] * t, ac[2] * t];
        let d = sub(ap, q);
        return dot(d, d);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub(c, b);
        let q = [b[0] + bc[0] * t, b[1] + bc[1] * t, b[2] + bc[2] * t];
        let d = sub(p, q);
        return dot(d, d);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    let d = sub(p, q);
    dot(d, d)
}

fn rms_dist(samples: &[[f32; 3]], target: &LodPrimitive) -> f64 {
    let tris: Vec<[[f32; 3]; 3]> = target
        .indices
        .chunks_exact(3)
        .map(|t| {
            [
                target.positions[t[0] as usize],
                target.positions[t[1] as usize],
                target.positions[t[2] as usize],
            ]
        })
        .collect();
    let sum: f64 = samples
        .par_iter()
        .map(|&p| {
            let mut best = f32::INFINITY;
            for t in &tris {
                let d = point_tri_dist2(p, t[0], t[1], t[2]);
                if d < best {
                    best = d;
                }
            }
            best as f64
        })
        .sum();
    (sum / samples.len() as f64).sqrt()
}

pub fn cmd_mesh(args: &[String]) -> Result<i32> {
    let mut quads = 575u32;
    let mut cap = 32_000usize;
    let mut k = 4u64;
    let mut samples = 4096usize;
    let mut use_gpu = false;
    let mut repeats = 1usize;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--quads" => {
                i += 1;
                quads = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--quads needs a number"))?;
            }
            "--cap" => {
                i += 1;
                cap = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--cap needs a number"))?;
            }
            "--k" => {
                i += 1;
                k = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--k needs a number"))?;
            }
            "--samples" => {
                i += 1;
                samples = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--samples needs a number"))?;
            }
            "--repeats" => {
                i += 1;
                repeats = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .filter(|&r| r >= 1)
                    .ok_or_else(|| anyhow!("--repeats needs a number >= 1"))?;
            }
            "--gpu" => use_gpu = true,
            other => return Err(anyhow!("unknown flag {other}")),
        }
        i += 1;
    }
    let prim = grid_prim(quads);
    let src_tris = prim.indices.len() / 3;
    println!(
        "mesh A/B: {} tris / {} verts, cap {cap}, coarsen k {k}",
        src_tris,
        prim.positions.len()
    );
    let pts = sample_surface(&prim, samples, 42);

    let (a_out, a_err, a_ms) = meshopt_simplify(&prim, cap)?;
    let a_rms = rms_dist(&pts, &a_out);
    println!(
        "A meshopt-only:        {:>7} tris  {:>8.1} ms  meshopt_err {:.5}  sampled_rms {:.5}",
        a_out.indices.len() / 3,
        a_ms,
        a_err,
        a_rms
    );

    let coarse_target = (cap as u64).saturating_mul(k) as usize;
    let t0 = Instant::now();
    let edges = gmd::boundary_edges(&prim.indices);
    let edges_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let mut backend = gmd::CpuBackend::new(&prim.positions, &prim.indices, &edges)
        .ok_or_else(|| anyhow!("degenerate bench mesh"))?;
    let ext = backend.extent();
    let scale = gmd::choose_scale(&mut backend, ext, coarse_target)?;
    let survey_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    let out = gmd::coarsen_tris_at(&mut backend, scale)?;
    let cluster_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let t3 = Instant::now();
    let coarse = gmd::rebuild_prim(&prim, &out.tris);
    let rebuild_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let (b_out, b_err, b_finish_ms) = meshopt_simplify(&coarse, cap)?;
    let b_rms = rms_dist(&pts, &b_out);
    println!(
        "B coarsen+finish(cpu): {:>7} tris  {:>8.1} ms  meshopt_err {:.5}  sampled_rms {:.5}",
        b_out.indices.len() / 3,
        edges_ms + survey_ms + cluster_ms + rebuild_ms + b_finish_ms,
        b_err,
        b_rms
    );
    println!(
        "  coarse {} tris at scale {:.2}; edges {:.1} ms, survey {:.1} ms, cluster {:.1} ms, rebuild {:.1} ms, finish {:.1} ms",
        coarse.indices.len() / 3,
        scale,
        edges_ms,
        survey_ms,
        cluster_ms,
        rebuild_ms,
        b_finish_ms
    );

    let t4 = Instant::now();
    let c_out = gmd::direct_prim_cpu(&prim, cap as u64)?;
    let c_ms = t4.elapsed().as_secs_f64() * 1000.0;
    let c_rms = rms_dist(&pts, &c_out);
    println!(
        "C direct-to-cap(cpu):  {:>7} tris  {:>8.1} ms  meshopt_err       -  sampled_rms {:.5}",
        c_out.indices.len() / 3,
        c_ms,
        c_rms
    );

    let t5 = Instant::now();
    let bytes = meshopt::typed_to_bytes(&prim.positions);
    let adapter = meshopt::VertexDataAdapter::new(bytes, 12, 0)
        .map_err(|e| anyhow!("meshopt vertex adapter: {e}"))?;
    let sloppy = meshopt::simplify_sloppy(&prim.indices, &adapter, cap * 3, 1.0, None);
    let d_ms = t5.elapsed().as_secs_f64() * 1000.0;
    let mut d_out = LodPrimitive {
        positions: prim.positions.clone(),
        indices: sloppy,
        material: prim.material,
        ..Default::default()
    };
    d_out.compact_orphans();
    let d_rms = rms_dist(&pts, &d_out);
    println!(
        "D meshopt-sloppy:      {:>7} tris  {:>8.1} ms  meshopt_err       -  sampled_rms {:.5}",
        d_out.indices.len() / 3,
        d_ms,
        d_rms
    );

    let mut identical = true;
    if use_gpu {
        let wgpu_lane = crate::gpu::wgpu_backend_selected();
        let backend_name = if wgpu_lane { "wgpu" } else { "cuda" };
        let t_ready = Instant::now();
        let available = if wgpu_lane {
            match crate::gpu::wgpu_mesh::warm_pipelines() {
                Ok(()) => true,
                Err(e) => {
                    println!("wgpu mesh pipelines unavailable: {e:#}");
                    false
                }
            }
        } else {
            crate::gpu::cuda::mesh::mesh_kernels_available()
        };
        let ready_ms = t_ready.elapsed().as_secs_f64() * 1000.0;
        if !available {
            println!("gpu lanes skipped: {backend_name} mesh kernels unavailable");
            return Ok(2);
        }
        println!("gpu backend {backend_name}: kernels ready in {ready_ms:.1} ms (cold pipeline build)");
        let cpu_coarsen = (scale, out.tris.as_slice());
        let ok = if wgpu_lane {
            run_gpu_lanes::<crate::gpu::wgpu_mesh::MeshSession>(
                &prim, &edges, &pts, cap, coarse_target, repeats, cpu_coarsen, &c_out,
            )?
        } else {
            run_gpu_lanes::<crate::gpu::cuda::mesh::MeshSession>(
                &prim, &edges, &pts, cap, coarse_target, repeats, cpu_coarsen, &c_out,
            )?
        };
        identical &= ok;
    }
    Ok(if identical { 0 } else { 1 })
}

trait BenchSession: gmd::CoarsenBackend + Sized {
    fn open_bench(prim: &LodPrimitive, edges: &[u32]) -> Result<Self>;
    fn ext(&self) -> [f32; 3];
}

impl BenchSession for crate::gpu::cuda::mesh::MeshSession {
    fn open_bench(prim: &LodPrimitive, edges: &[u32]) -> Result<Self> {
        Self::open(&prim.positions, &prim.indices, edges)
    }
    fn ext(&self) -> [f32; 3] {
        self.extent()
    }
}

impl BenchSession for crate::gpu::wgpu_mesh::MeshSession {
    fn open_bench(prim: &LodPrimitive, edges: &[u32]) -> Result<Self> {
        Self::open(&prim.positions, &prim.indices, edges)
    }
    fn ext(&self) -> [f32; 3] {
        self.extent()
    }
}

struct LaneRep {
    open_ms: f64,
    survey_ms: f64,
    cluster_ms: f64,
    rebuild_ms: f64,
    scale: f32,
    tris: Vec<u32>,
}

fn lane_once<B: BenchSession>(
    prim: &LodPrimitive,
    edges: &[u32],
    target: usize,
) -> Result<(LaneRep, LodPrimitive)> {
    let t0 = Instant::now();
    let mut s = B::open_bench(prim, edges)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let ext = s.ext();
    let scale = gmd::choose_scale(&mut s, ext, target)?;
    let survey_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    let out = gmd::coarsen_tris_at(&mut s, scale)?;
    let cluster_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let t3 = Instant::now();
    let coarse = gmd::rebuild_prim(prim, &out.tris);
    let rebuild_ms = t3.elapsed().as_secs_f64() * 1000.0;
    Ok((
        LaneRep {
            open_ms,
            survey_ms,
            cluster_ms,
            rebuild_ms,
            scale,
            tris: out.tris,
        },
        coarse,
    ))
}

fn lane_repeat<B: BenchSession>(
    label: &str,
    prim: &LodPrimitive,
    edges: &[u32],
    target: usize,
    repeats: usize,
) -> Result<(LaneRep, LodPrimitive)> {
    let mut best: Option<(LaneRep, LodPrimitive)> = None;
    for rep in 0..repeats {
        let (r, coarse) = lane_once::<B>(prim, edges, target)?;
        let total = r.open_ms + r.survey_ms + r.cluster_ms + r.rebuild_ms;
        println!(
            "  {label} rep {rep}: open {:.1} survey {:.1} cluster {:.1} rebuild {:.1} total {:.1} ms",
            r.open_ms, r.survey_ms, r.cluster_ms, r.rebuild_ms, total
        );
        let better = best
            .as_ref()
            .map(|(b, _)| {
                total < b.open_ms + b.survey_ms + b.cluster_ms + b.rebuild_ms
            })
            .unwrap_or(true);
        if let Some((b, _)) = &best {
            if r.scale != b.scale || r.tris != b.tris {
                return Err(anyhow!("{label}: nondeterministic output across repeats"));
            }
        }
        if better {
            best = Some((r, coarse));
        }
    }
    Ok(best.expect("repeats >= 1"))
}

#[allow(clippy::too_many_arguments)]
fn run_gpu_lanes<B: BenchSession>(
    prim: &LodPrimitive,
    edges: &[u32],
    pts: &[[f32; 3]],
    cap: usize,
    coarse_target: usize,
    repeats: usize,
    cpu_coarsen: (f32, &[u32]),
    cpu_direct: &LodPrimitive,
) -> Result<bool> {
    let mut identical = true;
    let (scale, cpu_tris) = cpu_coarsen;

    let (g, gcoarse) = lane_repeat::<B>("coarsen", prim, edges, coarse_target, repeats)?;
    let g_coarsen_ms = g.open_ms + g.survey_ms + g.cluster_ms + g.rebuild_ms;
    let (g_out, g_err, g_finish_ms) = meshopt_simplify(&gcoarse, cap)?;
    let g_rms = rms_dist(pts, &g_out);
    if g.scale == scale && g.tris == cpu_tris {
        println!(
            "  gpu-vs-cpu coarsen: IDENTICAL ({} tris at scale {:.4})",
            g.tris.len() / 3,
            g.scale
        );
    } else {
        identical = false;
        let mismatched = g
            .tris
            .iter()
            .zip(cpu_tris.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "  gpu-vs-cpu coarsen: DIVERGED  scale cpu {:.6} gpu {:.6}  tris cpu {} gpu {}  mismatched u32 {}",
            scale,
            g.scale,
            cpu_tris.len() / 3,
            g.tris.len() / 3,
            mismatched
        );
    }
    println!(
        "B' coarsen+finish(gpu): {:>6} tris  {:>8.1} ms  meshopt_err {:.5}  sampled_rms {:.5}  (gpu coarsen {:.1} ms incl. transfers, finish {:.1} ms)",
        g_out.indices.len() / 3,
        g_coarsen_ms + g_finish_ms,
        g_err,
        g_rms,
        g_coarsen_ms,
        g_finish_ms
    );

    let (d, gd) = lane_repeat::<B>("direct", prim, edges, cap, repeats)?;
    let gd_ms = d.open_ms + d.survey_ms + d.cluster_ms + d.rebuild_ms;
    let gd_rms = rms_dist(pts, &gd);
    println!(
        "C' direct-to-cap(gpu):  {:>6} tris  {:>8.1} ms  meshopt_err       -  sampled_rms {:.5}",
        gd.indices.len() / 3,
        gd_ms,
        gd_rms
    );
    if gd.indices == cpu_direct.indices && gd.positions == cpu_direct.positions {
        println!("  gpu-vs-cpu direct:  IDENTICAL");
    } else {
        identical = false;
        println!(
            "  gpu-vs-cpu direct:  DIVERGED  tris cpu {} gpu {}  scale cpu - gpu {:.6}",
            cpu_direct.indices.len() / 3,
            gd.indices.len() / 3,
            d.scale
        );
    }
    Ok(identical)
}
