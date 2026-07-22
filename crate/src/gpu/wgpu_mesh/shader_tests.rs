//! mesh_coarsen.wgsl validation and execution tests.
//!
//! - naga parse+validate runs on any box (no GPU needed) and is the same
//!   front end wgpu uses at pipeline creation.
//! - constant-drift checks parse the WGSL source text and pin its bit
//!   constants to the Rust mirror (which soft_tests pins to hardware).
//! - execution tests need a wgpu adapter (lavapipe suffices) and assert
//!   BYTE-identity of survey/cluster against the CPU oracle
//!   (gpu_mesh_dispatch::CpuBackend). They skip when no adapter exists;
//!   set ABGEN_GPU_REQUIRE_WGPU=1 to make skips fail.

use super::driver::{mesh_kernels_available, MeshSession, MESH_WGSL};
use super::softmesh as sm;
use crate::gpu_mesh_dispatch::{boundary_edges, CoarsenBackend, CpuBackend};

#[test]
fn mesh_wgsl_validates_under_naga() {
    let module = naga::front::wgsl::parse_str(MESH_WGSL)
        .unwrap_or_else(|e| panic!("mesh_coarsen.wgsl parse error:\n{}", e.emit_to_string(MESH_WGSL)));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    let info = validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("mesh_coarsen.wgsl validation error: {e:?}"));
    let entries: Vec<&str> = module
        .entry_points
        .iter()
        .map(|ep| ep.name.as_str())
        .collect();
    for want in [
        "mesh_survey",
        "mesh_accum",
        "mesh_accum_edges",
        "mesh_pick_cost",
        "mesh_pick_id",
        "mesh_remap",
    ] {
        assert!(entries.contains(&want), "missing entry point {want}");
    }
    let _ = info;
}

fn wgsl_const(name: &str) -> u32 {
    for line in MESH_WGSL.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix(&format!("const {name}: u32 = 0x")) else {
            continue;
        };
        let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        return u32::from_str_radix(&hex, 16).unwrap();
    }
    panic!("constant {name} not found in mesh_coarsen.wgsl");
}

#[test]
fn mesh_wgsl_constants_match_mirror() {
    assert_eq!(wgsl_const("C_F32_EPS20"), sm::C_F32_EPS20);
    assert_eq!(wgsl_const("C_F32_HALF"), sm::C_F32_HALF);
    assert_eq!(
        wgsl_const("C_F32_BOUNDARY_WEIGHT"),
        sm::C_F32_BOUNDARY_WEIGHT
    );
    assert_eq!(wgsl_const("C_F64_HALF_HI"), sm::C_F64_HALF.0);
    assert_eq!(wgsl_const("C_F64_TWO_HI"), sm::C_F64_TWO.0);
    assert_eq!(wgsl_const("C_F64_QUADRIC_FP_HI"), sm::C_F64_QUADRIC_FP.0);
    assert_eq!(
        wgsl_const("C_F64_INV_QUADRIC_FP_HI"),
        sm::C_F64_INV_QUADRIC_FP.0
    );
    assert_eq!(wgsl_const("C_F64_COST_FP_HI"), sm::C_F64_COST_FP.0);
    assert_eq!(wgsl_const("C_F64_NINE_E18_HI"), sm::C_F64_NINE_E18.0);
    assert_eq!(wgsl_const("C_F64_NINE_E18_LO"), sm::C_F64_NINE_E18.1);
    assert_eq!(wgsl_const("C_F64_U32_MAX_HI"), sm::C_F64_U32_MAX.0);
    assert_eq!(wgsl_const("C_F64_U32_MAX_LO"), sm::C_F64_U32_MAX.1);
    assert_eq!(wgsl_const("C_I64_SAT_POS_HI"), sm::C_I64_SAT_POS.0);
    assert_eq!(wgsl_const("C_I64_SAT_POS_LO"), sm::C_I64_SAT_POS.1);
    assert_eq!(wgsl_const("C_I64_SAT_NEG_HI"), sm::C_I64_SAT_NEG.0);
    assert_eq!(wgsl_const("C_I64_SAT_NEG_LO"), sm::C_I64_SAT_NEG.1);
    // the zero-of-f64 arguments passed inline in the shader rely on lo=0
    assert_eq!(sm::C_F64_HALF.1, 0);
    assert_eq!(sm::C_F64_TWO.1, 0);
    assert_eq!(sm::C_F64_QUADRIC_FP.1, 0);
    assert_eq!(sm::C_F64_INV_QUADRIC_FP.1, 0);
    assert_eq!(sm::C_F64_COST_FP.1, 0);
}

// ---------------------------------------------------------------------------
// execution (needs an adapter; lavapipe works)
// ---------------------------------------------------------------------------

fn require() -> bool {
    std::env::var("ABGEN_GPU_REQUIRE_WGPU").as_deref() == Ok("1")
}

fn session_or_skip(
    test: &str,
    positions: &[[f32; 3]],
    indices: &[u32],
    edges: &[u32],
) -> Option<MeshSession> {
    if !mesh_kernels_available() {
        if require() {
            panic!("{test}: no wgpu mesh kernels (ABGEN_GPU_REQUIRE_WGPU=1)");
        }
        eprintln!("{test}: SKIP no wgpu adapter / mesh pipelines");
        return None;
    }
    Some(MeshSession::open(positions, indices, edges).expect("open mesh session"))
}

fn grid_mesh(n: u32, flat: bool) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
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
    (positions, indices)
}

#[test]
fn wgpu_mesh_survey_and_cluster_byte_identical_to_cpu_oracle() {
    for (n, flat) in [(32u32, false), (48, true), (17, false)] {
        let (positions, indices) = grid_mesh(n, flat);
        let edges = boundary_edges(&indices);
        let Some(mut gpu) = session_or_skip(
            "wgpu_mesh_survey_and_cluster_byte_identical_to_cpu_oracle",
            &positions,
            &indices,
            &edges,
        ) else {
            return;
        };
        let mut cpu = CpuBackend::new(&positions, &indices, &edges).unwrap();
        let scales = [2.0f32, 3.7, 8.0, 16.5, 25.0, 100.0, 288.0];
        let t0 = std::time::Instant::now();
        let gpu_counts = gpu.survey(&scales).expect("gpu survey");
        let cpu_counts = cpu.survey(&scales).expect("cpu survey");
        assert_eq!(gpu_counts, cpu_counts, "survey n={n} flat={flat}");
        for &s in &scales {
            let g = gpu.cluster(s).expect("gpu cluster");
            let c = cpu.cluster(s).expect("cpu cluster");
            assert_eq!(
                g.len(),
                c.len(),
                "cluster len n={n} flat={flat} scale={s}"
            );
            if let Some(i) = g.iter().zip(c.iter()).position(|(a, b)| a != b) {
                panic!(
                    "cluster n={n} flat={flat} scale={s}: first divergence at word {i}: gpu {:#010x} cpu {:#010x}",
                    g[i], c[i]
                );
            }
        }
        eprintln!(
            "wgpu mesh byte-identity n={n} flat={flat}: survey+{} clusters in {} ms",
            scales.len(),
            t0.elapsed().as_millis()
        );
    }
}

#[test]
fn wgpu_mesh_no_edges_still_matches() {
    let (positions, indices) = grid_mesh(24, false);
    let Some(mut gpu) = session_or_skip(
        "wgpu_mesh_no_edges_still_matches",
        &positions,
        &indices,
        &[],
    ) else {
        return;
    };
    let mut cpu = CpuBackend::new(&positions, &indices, &[]).unwrap();
    for s in [3.0f32, 9.5] {
        assert_eq!(
            gpu.cluster(s).expect("gpu"),
            cpu.cluster(s).expect("cpu"),
            "scale {s}"
        );
    }
}

#[test]
fn wgpu_mesh_full_coarsen_flow_matches_cpu() {
    use crate::gpu_mesh_dispatch::coarsen_tris;
    let (positions, indices) = grid_mesh(128, false); // 32768 tris
    let edges = boundary_edges(&indices);
    let Some(mut gpu) = session_or_skip(
        "wgpu_mesh_full_coarsen_flow_matches_cpu",
        &positions,
        &indices,
        &edges,
    ) else {
        return;
    };
    let mut cpu = CpuBackend::new(&positions, &indices, &edges).unwrap();
    let ext = gpu.extent();
    assert_eq!(ext, cpu.extent());
    for target in [500usize, 2000] {
        let t0 = std::time::Instant::now();
        let go = coarsen_tris(&mut gpu, ext, target).expect("gpu coarsen");
        let tg = t0.elapsed();
        let t1 = std::time::Instant::now();
        let co = coarsen_tris(&mut cpu, ext, target).expect("cpu coarsen");
        let tc = t1.elapsed();
        assert_eq!(go.scale, co.scale, "target {target}: chosen scale");
        assert_eq!(go.survivors, co.survivors, "target {target}: survivors");
        assert_eq!(go.tris, co.tris, "target {target}: surviving tris");
        eprintln!(
            "full coarsen 32768 tris -> target {target}: {} survivors at scale {}; wgpu {} ms vs cpu {} ms",
            go.survivors,
            go.scale,
            tg.as_millis(),
            tc.as_millis()
        );
    }
}

/// End-to-end seam: ABGEN_GPU_BACKEND=wgpu routes direct_decimate through
/// the wgpu mesh session (after BC7 qualification arms the backend), and
/// the result matches the CPU coarsen path byte-for-byte. Ignored because
/// it mutates process env and resolves the global backend: run alone, e.g.
/// on lavapipe with the pinned 64-bit vulkan-loader.
#[test]
#[ignore = "mutates env + global backend resolution; run alone with --ignored"]
fn wgpu_mesh_dispatch_env_routing_end_to_end() {
    use crate::gpu_mesh_dispatch::{direct_decimate, direct_prim_cpu};
    use crate::lodgen::model::LodPrimitive;
    std::env::set_var("ABGEN_GPU", "1");
    std::env::set_var("ABGEN_GPU_BACKEND", "wgpu");
    std::env::set_var("ABGEN_GPU_MESH_MODE", "direct");
    if let Err(e) = crate::enable_gpu() {
        panic!("enable_gpu under ABGEN_GPU_BACKEND=wgpu failed: {e}");
    }
    let (positions, indices) = grid_mesh(128, false); // 32768 tris >= PRIM_MIN_TRIS
    let normals = vec![[0.0f32, 1.0, 0.0]; positions.len()];
    let uvs = vec![[0.0f32, 0.0]; positions.len()];
    let prim = LodPrimitive {
        positions,
        normals,
        uvs,
        indices,
        material: 0,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let (gpu_out, used_gpu) =
        direct_decimate(&prim, 500).expect("direct_decimate must take the wgpu lane");
    assert!(used_gpu);
    let cpu_out = direct_prim_cpu(&prim, 500).expect("cpu path");
    assert_eq!(gpu_out.indices, cpu_out.indices, "indices diverge");
    assert_eq!(gpu_out.positions, cpu_out.positions, "positions diverge");
    assert_eq!(gpu_out.normals, cpu_out.normals, "normals diverge");
    assert_eq!(gpu_out.uvs, cpu_out.uvs, "uvs diverge");
    eprintln!(
        "env-routed wgpu direct_decimate: 32768 -> {} tris in {} ms, byte-identical to CPU",
        gpu_out.indices.len() / 3,
        t0.elapsed().as_millis()
    );
}
