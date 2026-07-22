//! Softfloat transliteration of kernel-ptx core/mesh_coarsen.rs: the exact
//! computation the WGSL mesh kernels perform, expressed on the host for
//! testing. Every f32/f64 value is a u32 bit pattern (f64/i64 as (hi, lo)
//! pairs); the arithmetic goes through soft.rs only, so results are
//! bit-defined regardless of host FPU behavior — and must equal the
//! hardware-float core functions bit-for-bit (tests enforce this).

use super::soft::*;

pub type V3 = [u32; 3];
/// i64/f64 as (hi, lo) bit pairs.
pub type P64 = (u32, u32);

// f32 constants (bit patterns; soft_constants test pins them to literals).
pub const C_F32_ZERO: u32 = 0x0000_0000;
pub const C_F32_HALF: u32 = 0x3f00_0000;
pub const C_F32_EPS20: u32 = 0x1e3c_e508; // 1e-20f32
pub const C_F32_BOUNDARY_WEIGHT: u32 = 0x4080_0000; // 4.0

// f64 constants as (hi, lo).
pub const C_F64_ZERO: P64 = (0, 0);
pub const C_F64_HALF: P64 = (0x3fe0_0000, 0);
pub const C_F64_TWO: P64 = (0x4000_0000, 0);
pub const C_F64_QUADRIC_FP: P64 = (0x41d0_0000, 0); // 2^30
pub const C_F64_INV_QUADRIC_FP: P64 = (0x3e10_0000, 0); // 2^-30
pub const C_F64_COST_FP: P64 = (0x41b0_0000, 0); // 2^28
pub const C_F64_NINE_E18: P64 = (0x43df_399b, 0x1438_a100); // 9.0e18
pub const C_F64_NEG_NINE_E18: P64 = (0xc3df_399b, 0x1438_a100);
pub const C_F64_U32_MAX: P64 = (0x41ef_ffff, 0xffe0_0000); // 4294967295.0

// i64 saturation values of core mesh_coarsen::fp().
pub const C_I64_SAT_POS: P64 = (0x7ce6_6c50, 0xe284_0000); // 9e18
pub const C_I64_SAT_NEG: P64 = (0x8319_93af, 0x1d7c_0000); // -9e18

#[inline]
pub fn ssub3(a: V3, b: V3) -> V3 {
    [
        f32_sub(a[0], b[0]),
        f32_sub(a[1], b[1]),
        f32_sub(a[2], b[2]),
    ]
}

#[inline]
pub fn scross3(a: V3, b: V3) -> V3 {
    [
        f32_sub(f32_mul(a[1], b[2]), f32_mul(a[2], b[1])),
        f32_sub(f32_mul(a[2], b[0]), f32_mul(a[0], b[2])),
        f32_sub(f32_mul(a[0], b[1]), f32_mul(a[1], b[0])),
    ]
}

#[inline]
pub fn sdot3(a: V3, b: V3) -> u32 {
    f32_add(
        f32_add(f32_mul(a[0], b[0]), f32_mul(a[1], b[1])),
        f32_mul(a[2], b[2]),
    )
}

#[inline]
pub fn slen3(a: V3) -> u32 {
    f32_sqrt(sdot3(a, a))
}

#[inline]
pub fn scell_axis(v: u32, mn: u32, inv_cell: u32, dim: u32) -> u32 {
    let t = f32_mul(f32_sub(v, mn), inv_cell);
    if f32_le(t, C_F32_ZERO) {
        return 0;
    }
    let c = f32_trunc_u32(t);
    if c >= dim {
        dim - 1
    } else {
        c
    }
}

#[inline]
pub fn scell_index(p: V3, mn: V3, inv_cell: V3, dims: [u32; 3]) -> u32 {
    let cx = scell_axis(p[0], mn[0], inv_cell[0], dims[0]);
    let cy = scell_axis(p[1], mn[1], inv_cell[1], dims[1]);
    let cz = scell_axis(p[2], mn[2], inv_cell[2], dims[2]);
    (cz * dims[1] + cy) * dims[0] + cx
}

#[inline]
pub fn snorm_pos(p: V3, mn: V3, inv_ext: u32) -> V3 {
    [
        f32_mul(f32_sub(p[0], mn[0]), inv_ext),
        f32_mul(f32_sub(p[1], mn[1]), inv_ext),
        f32_mul(f32_sub(p[2], mn[2]), inv_ext),
    ]
}

pub fn stri_plane(a: V3, b: V3, c: V3) -> Option<([u32; 4], u32)> {
    let n = scross3(ssub3(b, a), ssub3(c, a));
    let twice_area = slen3(n);
    if f32_le(twice_area, C_F32_EPS20) {
        return None;
    }
    let u = [
        f32_div(n[0], twice_area),
        f32_div(n[1], twice_area),
        f32_div(n[2], twice_area),
    ];
    let plane = [u[0], u[1], u[2], f32_neg(sdot3(u, a))];
    Some((plane, f32_mul(C_F32_HALF, twice_area)))
}

pub fn sedge_plane(a: V3, b: V3, face_n: V3) -> Option<[u32; 4]> {
    let perp = scross3(ssub3(b, a), face_n);
    let l = slen3(perp);
    if f32_le(l, C_F32_EPS20) {
        return None;
    }
    let u = [f32_div(perp[0], l), f32_div(perp[1], l), f32_div(perp[2], l)];
    Some([u[0], u[1], u[2], f32_neg(sdot3(u, a))])
}

/// mesh_coarsen::fp on an f64 pair -> saturating fixed-point i64 pair.
pub fn sfp(x: P64) -> P64 {
    let r = if f64_ge(x.0, x.1, C_F64_ZERO.0, C_F64_ZERO.1) {
        f64_add(x.0, x.1, C_F64_HALF.0, C_F64_HALF.1)
    } else {
        f64_sub(x.0, x.1, C_F64_HALF.0, C_F64_HALF.1)
    };
    if f64_ge(r.0, r.1, C_F64_NINE_E18.0, C_F64_NINE_E18.1) {
        C_I64_SAT_POS
    } else if f64_le(r.0, r.1, C_F64_NEG_NINE_E18.0, C_F64_NEG_NINE_E18.1) {
        C_I64_SAT_NEG
    } else {
        f64_trunc_i64(r.0, r.1)
    }
}

pub fn splane_quadric_fp(p: [u32; 4], weight: u32) -> [P64; 10] {
    let wt = f32_to_f64(weight);
    let w = f64_mul(wt.0, wt.1, C_F64_QUADRIC_FP.0, C_F64_QUADRIC_FP.1);
    let a = f32_to_f64(p[0]);
    let b = f32_to_f64(p[1]);
    let c = f32_to_f64(p[2]);
    let d = f32_to_f64(p[3]);
    let m = |x: P64, y: P64| -> P64 {
        let wx = f64_mul(w.0, w.1, x.0, x.1);
        f64_mul(wx.0, wx.1, y.0, y.1)
    };
    [
        sfp(m(a, a)),
        sfp(m(a, b)),
        sfp(m(a, c)),
        sfp(m(a, d)),
        sfp(m(b, b)),
        sfp(m(b, c)),
        sfp(m(b, d)),
        sfp(m(c, c)),
        sfp(m(c, d)),
        sfp(m(d, d)),
    ]
}

pub fn seval_cost_fp(q: &[P64; 10], p: V3) -> u32 {
    let x = f32_to_f64(p[0]);
    let y = f32_to_f64(p[1]);
    let z = f32_to_f64(p[2]);
    let qf = |i: usize| i64_to_f64(q[i].0, q[i].1);
    let mul = |a: P64, b: P64| f64_mul(a.0, a.1, b.0, b.1);
    let add = |a: P64, b: P64| f64_add(a.0, a.1, b.0, b.1);
    let two = |t: P64| mul(C_F64_TWO, t);
    let mut s = mul(mul(qf(0), x), x);
    s = add(s, two(mul(mul(qf(1), x), y)));
    s = add(s, two(mul(mul(qf(2), x), z)));
    s = add(s, two(mul(qf(3), x)));
    s = add(s, mul(mul(qf(4), y), y));
    s = add(s, two(mul(mul(qf(5), y), z)));
    s = add(s, two(mul(qf(6), y)));
    s = add(s, mul(mul(qf(7), z), z));
    s = add(s, two(mul(qf(8), z)));
    s = add(s, qf(9));
    // s / QUADRIC_FP == s * 2^-30: the reciprocal of an exact power of two is
    // exact, so both are RN of the same real value, hence bit-identical.
    let cost = mul(s, C_F64_INV_QUADRIC_FP);
    if f64_le(cost.0, cost.1, C_F64_ZERO.0, C_F64_ZERO.1) {
        return 0;
    }
    let r = mul(cost, C_F64_COST_FP);
    if f64_ge(r.0, r.1, C_F64_U32_MAX.0, C_F64_U32_MAX.1) {
        u32::MAX
    } else {
        f64_trunc_u32(r.0, r.1)
    }
}

pub fn saccum_tri_quadric(pa: V3, pb: V3, pc: V3, mn: V3, inv_ext: u32) -> Option<[P64; 10]> {
    let a = snorm_pos(pa, mn, inv_ext);
    let b = snorm_pos(pb, mn, inv_ext);
    let c = snorm_pos(pc, mn, inv_ext);
    let (plane, area) = stri_plane(a, b, c)?;
    Some(splane_quadric_fp(plane, area))
}

pub fn saccum_edge_quadric(pu: V3, pv: V3, pw: V3, mn: V3, inv_ext: u32) -> Option<[P64; 10]> {
    let a = snorm_pos(pu, mn, inv_ext);
    let b = snorm_pos(pv, mn, inv_ext);
    let c = snorm_pos(pw, mn, inv_ext);
    let (face, _) = stri_plane(a, b, c)?;
    let plane = sedge_plane(a, b, [face[0], face[1], face[2]])?;
    let e = ssub3(b, a);
    Some(splane_quadric_fp(
        plane,
        f32_mul(sdot3(e, e), C_F32_BOUNDARY_WEIGHT),
    ))
}
