use super::sqrtf;

pub const QUADRIC_FP: f64 = 1_073_741_824.0;
pub const COST_FP: f64 = 268_435_456.0;
pub const BOUNDARY_WEIGHT: f32 = 4.0;
pub const MAX_SCALE: f32 = 1024.0;
pub const MAX_AXIS_DIM: u32 = 1024;
pub const CULLED: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SurveyParams {
    pub mn: [f32; 3],
    pub ntris: u32,
    pub ext: [f32; 3],
    pub nscales: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CoarsenParams {
    pub mn: [f32; 3],
    pub inv_ext: f32,
    pub inv_cell: [f32; 3],
    pub nverts: u32,
    pub dims: [u32; 3],
    pub ntris: u32,
    pub nedges: u32,
    pub ncells: u32,
    pub pad: [u32; 2],
}

#[inline]
pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn len3(a: [f32; 3]) -> f32 {
    sqrtf(dot3(a, a))
}

#[inline]
pub fn max_ext(ext: [f32; 3]) -> f32 {
    let mut m = ext[0];
    if ext[1] > m {
        m = ext[1];
    }
    if ext[2] > m {
        m = ext[2];
    }
    m
}

#[inline]
pub fn axis_dim(ext_k: f32, mx: f32, scale: f32) -> u32 {
    if ext_k <= 0.0 || mx <= 0.0 {
        return 1;
    }
    let d = (scale * ext_k / mx) as u32;
    if d < 1 {
        1
    } else if d > MAX_AXIS_DIM {
        MAX_AXIS_DIM
    } else {
        d
    }
}

#[inline]
pub fn scale_grid(ext: [f32; 3], scale: f32) -> ([u32; 3], [f32; 3]) {
    let mx = max_ext(ext);
    let dims = [
        axis_dim(ext[0], mx, scale),
        axis_dim(ext[1], mx, scale),
        axis_dim(ext[2], mx, scale),
    ];
    let inv = |k: usize| {
        if ext[k] > 0.0 {
            dims[k] as f32 / ext[k]
        } else {
            0.0
        }
    };
    (dims, [inv(0), inv(1), inv(2)])
}

#[inline]
pub fn ncells_of(dims: [u32; 3]) -> u64 {
    dims[0] as u64 * dims[1] as u64 * dims[2] as u64
}

#[inline]
pub fn cell_axis(v: f32, mn: f32, inv_cell: f32, dim: u32) -> u32 {
    let t = (v - mn) * inv_cell;
    if t <= 0.0 {
        return 0;
    }
    let c = t as u32;
    if c >= dim {
        dim - 1
    } else {
        c
    }
}

#[inline]
pub fn cell_index(p: [f32; 3], mn: [f32; 3], inv_cell: [f32; 3], dims: [u32; 3]) -> u32 {
    let cx = cell_axis(p[0], mn[0], inv_cell[0], dims[0]);
    let cy = cell_axis(p[1], mn[1], inv_cell[1], dims[1]);
    let cz = cell_axis(p[2], mn[2], inv_cell[2], dims[2]);
    (cz * dims[1] + cy) * dims[0] + cx
}

#[inline]
pub fn tri_survives(ca: u32, cb: u32, cc: u32) -> bool {
    ca != cb && cb != cc && ca != cc
}

#[inline]
pub fn norm_pos(p: [f32; 3], mn: [f32; 3], inv_ext: f32) -> [f32; 3] {
    [
        (p[0] - mn[0]) * inv_ext,
        (p[1] - mn[1]) * inv_ext,
        (p[2] - mn[2]) * inv_ext,
    ]
}

#[inline]
pub fn tri_plane(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<([f32; 4], f32)> {
    let n = cross3(sub3(b, a), sub3(c, a));
    let twice_area = len3(n);
    if twice_area <= 1e-20 {
        return None;
    }
    let u = [n[0] / twice_area, n[1] / twice_area, n[2] / twice_area];
    Some(([u[0], u[1], u[2], -dot3(u, a)], 0.5 * twice_area))
}

#[inline]
pub fn edge_plane(a: [f32; 3], b: [f32; 3], face_n: [f32; 3]) -> Option<[f32; 4]> {
    let perp = cross3(sub3(b, a), face_n);
    let l = len3(perp);
    if l <= 1e-20 {
        return None;
    }
    let u = [perp[0] / l, perp[1] / l, perp[2] / l];
    Some([u[0], u[1], u[2], -dot3(u, a)])
}

#[inline]
pub fn fp(x: f64) -> i64 {
    let r = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
    if r >= 9.0e18 {
        9_000_000_000_000_000_000
    } else if r <= -9.0e18 {
        -9_000_000_000_000_000_000
    } else {
        r as i64
    }
}

#[inline]
pub fn plane_quadric_fp(p: [f32; 4], weight: f32) -> [i64; 10] {
    let w = weight as f64 * QUADRIC_FP;
    let a = p[0] as f64;
    let b = p[1] as f64;
    let c = p[2] as f64;
    let d = p[3] as f64;
    [
        fp(w * a * a),
        fp(w * a * b),
        fp(w * a * c),
        fp(w * a * d),
        fp(w * b * b),
        fp(w * b * c),
        fp(w * b * d),
        fp(w * c * c),
        fp(w * c * d),
        fp(w * d * d),
    ]
}

#[inline]
pub fn eval_cost_fp(q: &[i64; 10], p: [f32; 3]) -> u32 {
    let x = p[0] as f64;
    let y = p[1] as f64;
    let z = p[2] as f64;
    let s = q[0] as f64 * x * x
        + 2.0 * (q[1] as f64 * x * y)
        + 2.0 * (q[2] as f64 * x * z)
        + 2.0 * (q[3] as f64 * x)
        + q[4] as f64 * y * y
        + 2.0 * (q[5] as f64 * y * z)
        + 2.0 * (q[6] as f64 * y)
        + q[7] as f64 * z * z
        + 2.0 * (q[8] as f64 * z)
        + q[9] as f64;
    let cost = s / QUADRIC_FP;
    if cost <= 0.0 {
        return 0;
    }
    let r = cost * COST_FP;
    if r >= 4_294_967_295.0 {
        u32::MAX
    } else {
        r as u32
    }
}

#[inline]
pub fn pack_cost_id(cost: u32, id: u32) -> u64 {
    ((cost as u64) << 32) | id as u64
}

#[inline]
pub fn accum_tri_quadric(
    pa: [f32; 3],
    pb: [f32; 3],
    pc: [f32; 3],
    mn: [f32; 3],
    inv_ext: f32,
) -> Option<[i64; 10]> {
    let a = norm_pos(pa, mn, inv_ext);
    let b = norm_pos(pb, mn, inv_ext);
    let c = norm_pos(pc, mn, inv_ext);
    let (plane, area) = tri_plane(a, b, c)?;
    Some(plane_quadric_fp(plane, area))
}

#[inline]
pub fn accum_edge_quadric(
    pu: [f32; 3],
    pv: [f32; 3],
    pw: [f32; 3],
    mn: [f32; 3],
    inv_ext: f32,
) -> Option<[i64; 10]> {
    let a = norm_pos(pu, mn, inv_ext);
    let b = norm_pos(pv, mn, inv_ext);
    let c = norm_pos(pw, mn, inv_ext);
    let (face, _) = tri_plane(a, b, c)?;
    let plane = edge_plane(a, b, [face[0], face[1], face[2]])?;
    let e = sub3(b, a);
    Some(plane_quadric_fp(plane, dot3(e, e) * BOUNDARY_WEIGHT))
}
