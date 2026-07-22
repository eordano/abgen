#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
#[path = "core/mod.rs"]
mod abgen_gpu_core;

use crate::abgen_gpu_core::bc7::{encode_group, group_signature, OptTables, Params, GROUP_WIDTH};
use crate::abgen_gpu_core::mesh_coarsen::{
    accum_edge_quadric, accum_tri_quadric, cell_index, eval_cost_fp, norm_pos, pack_cost_id,
    scale_grid, tri_survives, CoarsenParams, SurveyParams, CULLED,
};
use crate::abgen_gpu_core::mips::{
    box_halve_cell, box_halve_dims, level_block_dims, linearize_pixel, quantize_pack_block,
    HalveItem, LinItem, PackItem,
};
use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
use core::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

unsafe fn global_id() -> usize {
    (_block_idx_x() as usize) * (_block_dim_x() as usize) + (_thread_idx_x() as usize)
}

unsafe fn find_item(prefix: *const u64, n_items: usize, gid: u64) -> usize {
    let mut lo = 0usize;
    let mut hi = n_items;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if *prefix.add(mid) <= gid {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn bc7_encode_groups(
    blocks: *const u8,
    num_blocks: usize,
    params: *const Params,
    tables: *const OptTables,
    out: *mut u8,
) {
    let gidx =
        (_block_idx_x() as usize) * (_block_dim_x() as usize) + (_thread_idx_x() as usize);
    let num_groups = num_blocks.div_ceil(GROUP_WIDTH);
    if gidx >= num_groups {
        return;
    }
    let start = gidx * GROUP_WIDTH;
    let n = if num_blocks - start < GROUP_WIDTH {
        num_blocks - start
    } else {
        GROUP_WIDTH
    };
    let src = core::slice::from_raw_parts(blocks.add(start * 64), n * 64);
    let mut enc = [[0u8; 16]; GROUP_WIDTH];
    encode_group(src, n, &*params, &*tables, &mut enc);
    let dst = core::slice::from_raw_parts_mut(out.add(start * 16), n * 16);
    for k in 0..n {
        dst[k * 16..(k + 1) * 16].copy_from_slice(&enc[k]);
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn bc7_group_sigs(
    blocks: *const u8,
    num_blocks: usize,
    sigs: *mut u8,
) {
    let gidx = global_id();
    let num_groups = num_blocks.div_ceil(GROUP_WIDTH);
    if gidx >= num_groups {
        return;
    }
    let start = gidx * GROUP_WIDTH;
    let n = if num_blocks - start < GROUP_WIDTH {
        num_blocks - start
    } else {
        GROUP_WIDTH
    };
    let src = core::slice::from_raw_parts(blocks.add(start * 64), n * 64);
    *sigs.add(gidx) = group_signature(src, n);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn bc7_encode_groups_perm(
    blocks: *const u8,
    num_blocks: usize,
    perm: *const u32,
    params: *const Params,
    tables: *const OptTables,
    out: *mut u8,
) {
    let i = global_id();
    let num_groups = num_blocks.div_ceil(GROUP_WIDTH);
    if i >= num_groups {
        return;
    }
    let gidx = *perm.add(i) as usize;
    let start = gidx * GROUP_WIDTH;
    let n = if num_blocks - start < GROUP_WIDTH {
        num_blocks - start
    } else {
        GROUP_WIDTH
    };
    let src = core::slice::from_raw_parts(blocks.add(start * 64), n * 64);
    let mut enc = [[0u8; 16]; GROUP_WIDTH];
    encode_group(src, n, &*params, &*tables, &mut enc);
    let dst = core::slice::from_raw_parts_mut(out.add(start * 16), n * 16);
    for k in 0..n {
        dst[k * 16..(k + 1) * 16].copy_from_slice(&enc[k]);
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn bc7_group_sigs_desc(
    blocks: *const u8,
    descs: *const u64,
    num_groups: usize,
    sigs: *mut u8,
) {
    let gidx = global_id();
    if gidx >= num_groups {
        return;
    }
    let d = *descs.add(gidx);
    let n = (d & 0xf) as usize;
    let start = (d >> 8) as usize;
    let src = core::slice::from_raw_parts(blocks.add(start * 64), n * 64);
    *sigs.add(gidx) = group_signature(src, n);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn bc7_encode_groups_desc(
    blocks: *const u8,
    descs: *const u64,
    num_groups: usize,
    params4: *const Params,
    tables: *const OptTables,
    out: *mut u8,
) {
    let gidx = global_id();
    if gidx >= num_groups {
        return;
    }
    let d = *descs.add(gidx);
    let n = (d & 0xf) as usize;
    let bucket = ((d >> 4) & 0xf) as usize;
    let start = (d >> 8) as usize;
    let src = core::slice::from_raw_parts(blocks.add(start * 64), n * 64);
    let mut enc = [[0u8; 16]; GROUP_WIDTH];
    encode_group(src, n, &*params4.add(bucket), &*tables, &mut enc);
    let dst = core::slice::from_raw_parts_mut(out.add(start * 16), n * 16);
    for k in 0..n {
        dst[k * 16..(k + 1) * 16].copy_from_slice(&enc[k]);
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn blockify_linearize(
    items: *const LinItem,
    prefix: *const u64,
    n_items: usize,
    total: usize,
    base: *const u8,
    pyr: *mut f32,
) {
    let gid = global_id();
    if gid >= total {
        return;
    }
    let idx = find_item(prefix, n_items, gid as u64);
    let it = &*items.add(idx);
    let p = gid as u64 - *prefix.add(idx);
    let src = core::slice::from_raw_parts(base.add(((it.base_px + p) * 4) as usize), 4);
    let dst = core::slice::from_raw_parts_mut(pyr.add(((it.pyr_px + p) * 4) as usize), 4);
    linearize_pixel(src, it.srgb != 0, dst);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn blockify_quantize_pack(
    items: *const PackItem,
    prefix: *const u64,
    n_items: usize,
    total: usize,
    pyr: *const f32,
    blocks: *mut u8,
) {
    let gid = global_id();
    if gid >= total {
        return;
    }
    let idx = find_item(prefix, n_items, gid as u64);
    let it = &*items.add(idx);
    let lb = gid as u64 - *prefix.add(idx);
    let w = it.w as usize;
    let h = it.h as usize;
    let (bw, _) = level_block_dims(w, h);
    let bx = (lb as usize) % bw;
    let by = (lb as usize) / bw;
    let level = core::slice::from_raw_parts(pyr.add((it.lvl_px * 4) as usize), w * h * 4);
    let out = core::slice::from_raw_parts_mut(blocks.add(((it.blk_off + lb) * 64) as usize), 64);
    quantize_pack_block(level, w, h, it.srgb != 0, bx, by, out);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn blockify_halve(
    items: *const HalveItem,
    prefix: *const u64,
    n_items: usize,
    total: usize,
    pyr: *mut f32,
) {
    let gid = global_id();
    if gid >= total {
        return;
    }
    let idx = find_item(prefix, n_items, gid as u64);
    let it = &*items.add(idx);
    let np = gid as u64 - *prefix.add(idx);
    let w = it.w as usize;
    let h = it.h as usize;
    let (nw, _) = box_halve_dims(w, h);
    let nx = (np as usize) % nw;
    let ny = (np as usize) / nw;
    let src = core::slice::from_raw_parts(pyr.add((it.src_px * 4) as usize) as *const f32, w * h * 4);
    let mut cell = [0f32; 4];
    box_halve_cell(src, w, h, nx, ny, &mut cell);
    let dst = core::slice::from_raw_parts_mut(pyr.add(((it.dst_px + np) * 4) as usize), 4);
    dst.copy_from_slice(&cell);
}

unsafe fn atom_add_i64(p: *mut i64, v: i64) {
    AtomicI64::from_ptr(p).fetch_add(v, Ordering::Relaxed);
}

unsafe fn atom_add_u32(p: *mut u32, v: u32) {
    AtomicU32::from_ptr(p).fetch_add(v, Ordering::Relaxed);
}

unsafe fn atom_min_u64(p: *mut u64, v: u64) {
    AtomicU64::from_ptr(p).fetch_min(v, Ordering::Relaxed);
}

unsafe fn load_pos(positions: *const f32, i: usize) -> [f32; 3] {
    [
        *positions.add(i * 3),
        *positions.add(i * 3 + 1),
        *positions.add(i * 3 + 2),
    ]
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mesh_survey(
    params: *const SurveyParams,
    positions: *const f32,
    indices: *const u32,
    scales: *const f32,
    counts: *mut u32,
) {
    let p = &*params;
    let t = global_id();
    if t >= p.ntris as usize {
        return;
    }
    let a = load_pos(positions, *indices.add(t * 3) as usize);
    let b = load_pos(positions, *indices.add(t * 3 + 1) as usize);
    let c = load_pos(positions, *indices.add(t * 3 + 2) as usize);
    let mut s = 0usize;
    while s < p.nscales as usize {
        let (dims, inv_cell) = scale_grid(p.ext, *scales.add(s));
        let ca = cell_index(a, p.mn, inv_cell, dims);
        let cb = cell_index(b, p.mn, inv_cell, dims);
        let cc = cell_index(c, p.mn, inv_cell, dims);
        if tri_survives(ca, cb, cc) {
            atom_add_u32(counts.add(s), 1);
        }
        s += 1;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mesh_accum(
    params: *const CoarsenParams,
    positions: *const f32,
    indices: *const u32,
    cell_q: *mut i64,
) {
    let p = &*params;
    let t = global_id();
    if t >= p.ntris as usize {
        return;
    }
    let ia = *indices.add(t * 3) as usize;
    let ib = *indices.add(t * 3 + 1) as usize;
    let ic = *indices.add(t * 3 + 2) as usize;
    let a = load_pos(positions, ia);
    let b = load_pos(positions, ib);
    let c = load_pos(positions, ic);
    if let Some(q) = accum_tri_quadric(a, b, c, p.mn, p.inv_ext) {
        let cells = [
            cell_index(a, p.mn, p.inv_cell, p.dims),
            cell_index(b, p.mn, p.inv_cell, p.dims),
            cell_index(c, p.mn, p.inv_cell, p.dims),
        ];
        let mut k = 0usize;
        while k < 3 {
            let base = cells[k] as usize * 10;
            let mut i = 0usize;
            while i < 10 {
                atom_add_i64(cell_q.add(base + i), q[i]);
                i += 1;
            }
            k += 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mesh_accum_edges(
    params: *const CoarsenParams,
    positions: *const f32,
    edges: *const u32,
    cell_q: *mut i64,
) {
    let p = &*params;
    let e = global_id();
    if e >= p.nedges as usize {
        return;
    }
    let u = load_pos(positions, *edges.add(e * 3) as usize);
    let v = load_pos(positions, *edges.add(e * 3 + 1) as usize);
    let w = load_pos(positions, *edges.add(e * 3 + 2) as usize);
    if let Some(q) = accum_edge_quadric(u, v, w, p.mn, p.inv_ext) {
        let cells = [
            cell_index(u, p.mn, p.inv_cell, p.dims),
            cell_index(v, p.mn, p.inv_cell, p.dims),
        ];
        let mut k = 0usize;
        while k < 2 {
            let base = cells[k] as usize * 10;
            let mut i = 0usize;
            while i < 10 {
                atom_add_i64(cell_q.add(base + i), q[i]);
                i += 1;
            }
            k += 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mesh_pick(
    params: *const CoarsenParams,
    positions: *const f32,
    cell_q: *const i64,
    best: *mut u64,
) {
    let p = &*params;
    let vid = global_id();
    if vid >= p.nverts as usize {
        return;
    }
    let pos = load_pos(positions, vid);
    let cid = cell_index(pos, p.mn, p.inv_cell, p.dims) as usize;
    let mut q = [0i64; 10];
    let mut i = 0usize;
    while i < 10 {
        q[i] = *cell_q.add(cid * 10 + i);
        i += 1;
    }
    let cost = eval_cost_fp(&q, norm_pos(pos, p.mn, p.inv_ext));
    atom_min_u64(best.add(cid), pack_cost_id(cost, vid as u32));
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mesh_remap(
    params: *const CoarsenParams,
    positions: *const f32,
    indices: *const u32,
    best: *const u64,
    out_tris: *mut u32,
) {
    let p = &*params;
    let t = global_id();
    if t >= p.ntris as usize {
        return;
    }
    let a = load_pos(positions, *indices.add(t * 3) as usize);
    let b = load_pos(positions, *indices.add(t * 3 + 1) as usize);
    let c = load_pos(positions, *indices.add(t * 3 + 2) as usize);
    let ca = cell_index(a, p.mn, p.inv_cell, p.dims);
    let cb = cell_index(b, p.mn, p.inv_cell, p.dims);
    let cc = cell_index(c, p.mn, p.inv_cell, p.dims);
    if !tri_survives(ca, cb, cc) {
        *out_tris.add(t * 3) = CULLED;
        return;
    }
    *out_tris.add(t * 3) = (*best.add(ca as usize) & 0xffff_ffff) as u32;
    *out_tris.add(t * 3 + 1) = (*best.add(cb as usize) & 0xffff_ffff) as u32;
    *out_tris.add(t * 3 + 2) = (*best.add(cc as usize) & 0xffff_ffff) as u32;
}
