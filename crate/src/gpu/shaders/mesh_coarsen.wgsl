// wgpu mesh-coarsen kernels: WGSL port of kernel-ptx mesh_survey /
// mesh_accum / mesh_accum_edges / mesh_pick (split in two) / mesh_remap.
//
// Byte-identity contract: output must be bit-equal to the CPU oracle
// (gpu_mesh_dispatch::CpuBackend), which computes in host IEEE f32/f64.
// WGSL has no f64/i64 and gives no exactness guarantees for native f32
// (fma contraction is allowed, denormals may flush, div/sqrt are loosely
// rounded), so every float operation below is integer softfloat on u32
// bit patterns - a transliteration of crate/src/gpu/wgpu_mesh/soft.rs and
// softmesh.rs (round-to-nearest-even; f64 and i64 travel as hi/lo pairs).
// Tests pin the Rust mirror to hardware IEEE and this file to the mirror.
//
// Determinism of the atomics:
// - survey counts: u32 atomicAdd, commutative mod 2^32.
// - quadric accumulation: wrapping i64 sums emulated as two u32 atomicAdds
//   with a carry derived from the returned previous value. The lo-word sum
//   and the total number of carries (= floor(sum of lo-terms / 2^32)) are
//   both order-independent, so the final hi:lo pair equals the sequential
//   wrapping sum for every interleaving.
// - rep pick: the CUDA u64 atomicMin over pack(cost,id) becomes two passes
//   (atomicMin of cost, then atomicMin of id among cost-minima), which is
//   exactly the lexicographic = packed-u64 minimum.

struct MeshParams {
    mn0: u32,
    mn1: u32,
    mn2: u32,
    inv_ext: u32,
    ic0: u32,
    ic1: u32,
    ic2: u32,
    nverts: u32,
    d0: u32,
    d1: u32,
    d2: u32,
    ntris: u32,
    nedges: u32,
    ncells: u32,
    nscales: u32,
    base: u32,
}

struct GridEntry {
    d0: u32,
    d1: u32,
    d2: u32,
    pad0: u32,
    ic0: u32,
    ic1: u32,
    ic2: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: MeshParams;
@group(0) @binding(1) var<storage, read> positions: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> edges: array<u32>;
@group(0) @binding(4) var<storage, read> grids: array<GridEntry>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cell_q: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> best_cost: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> vert_cost: array<u32>;
@group(0) @binding(9) var<storage, read_write> best_id: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> out_tris: array<u32>;

// ---------------------------------------------------------------------------
// 64-bit helpers ((hi, lo) pairs)
// ---------------------------------------------------------------------------

struct U64 {
    hi: u32,
    lo: u32,
}

fn u64_new(hi: u32, lo: u32) -> U64 {
    return U64(hi, lo);
}

fn mul32x32(a: u32, b: u32) -> U64 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let mid = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    let lo = (ll & 0xffffu) | (mid << 16u);
    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return U64(hi, lo);
}

fn add64(ah: u32, al: u32, bh: u32, bl: u32) -> U64 {
    let lo = al + bl;
    var carry = 0u;
    if (lo < al) {
        carry = 1u;
    }
    return U64(ah + bh + carry, lo);
}

fn sub64(ah: u32, al: u32, bh: u32, bl: u32) -> U64 {
    let lo = al - bl;
    var borrow = 0u;
    if (al < bl) {
        borrow = 1u;
    }
    return U64(ah - bh - borrow, lo);
}

fn neg64(h: u32, l: u32) -> U64 {
    return sub64(0u, 0u, h, l);
}

fn shl64(h: u32, l: u32, n: u32) -> U64 {
    if (n == 0u) {
        return U64(h, l);
    } else if (n < 32u) {
        return U64((h << n) | (l >> (32u - n)), l << n);
    } else if (n < 64u) {
        return U64(l << (n - 32u), 0u);
    }
    return U64(0u, 0u);
}

fn shr64(h: u32, l: u32, n: u32) -> U64 {
    if (n == 0u) {
        return U64(h, l);
    } else if (n < 32u) {
        return U64(h >> n, (l >> n) | (h << (32u - n)));
    } else if (n < 64u) {
        return U64(0u, h >> (n - 32u));
    }
    return U64(0u, 0u);
}

fn shr64_jam(h: u32, l: u32, n: u32) -> U64 {
    if (n == 0u) {
        return U64(h, l);
    } else if (n < 32u) {
        let lost = l & ((1u << n) - 1u);
        var s = shr64(h, l, n);
        if (lost != 0u) {
            s.lo = s.lo | 1u;
        }
        return s;
    } else if (n < 64u) {
        var mask = 0u;
        if (n > 32u) {
            mask = (1u << (n - 32u)) - 1u;
        }
        let lost = l | (h & mask);
        var r = U64(0u, h >> (n - 32u));
        if (lost != 0u) {
            r.lo = r.lo | 1u;
        }
        return r;
    }
    var r = U64(0u, 0u);
    if (h != 0u || l != 0u) {
        r.lo = 1u;
    }
    return r;
}

fn shr32_jam(m: u32, n: u32) -> u32 {
    if (n == 0u) {
        return m;
    } else if (n < 32u) {
        var jam = 0u;
        if ((m & ((1u << n) - 1u)) != 0u) {
            jam = 1u;
        }
        return (m >> n) | jam;
    } else if (m != 0u) {
        return 1u;
    }
    return 0u;
}

fn clz64(h: u32, l: u32) -> u32 {
    if (h != 0u) {
        return countLeadingZeros(h);
    }
    return 32u + countLeadingZeros(l);
}

fn ge64(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    return ah > bh || (ah == bh && al >= bl);
}

// ---------------------------------------------------------------------------
// f32 softfloat (bits in u32)
// ---------------------------------------------------------------------------

const F32_QNAN: u32 = 0x7fc00000u;
const F32_INF: u32 = 0x7f800000u;

fn f32_is_nan(x: u32) -> bool {
    return (x & 0x7fffffffu) > F32_INF;
}

struct Unpack32 {
    e: i32,
    m: u32,
}

fn f32_norm_sub(f: u32) -> Unpack32 {
    let shift = countLeadingZeros(f) - 8u;
    return Unpack32(1 - i32(shift), f << shift);
}

fn f32_unpack(x: u32) -> Unpack32 {
    let e = (x >> 23u) & 0xffu;
    let f = x & 0x007fffffu;
    if (e == 0u) {
        return f32_norm_sub(f);
    }
    return Unpack32(i32(e), f | 0x00800000u);
}

// RNE pack; value = m * 2^(e-153), m < 2^27 with GRS at bits 2..0.
fn f32_round_pack(sign: u32, e_in: i32, m_in: u32) -> u32 {
    var e = e_in;
    var m = m_in;
    if (e <= 0) {
        m = shr32_jam(m, u32(1 - e));
        e = 1;
    } else if (e > 254) {
        return (sign << 31u) | F32_INF;
    }
    let round_bits = m & 7u;
    var mr = (m + 4u) >> 3u;
    if (round_bits == 4u) {
        mr = mr & ~1u;
    }
    return (sign << 31u) + (u32(e - 1) << 23u) + mr;
}

fn f32_add(a: u32, b: u32) -> u32 {
    let ea = (a >> 23u) & 0xffu;
    let eb = (b >> 23u) & 0xffu;
    let fa = a & 0x007fffffu;
    let fb = b & 0x007fffffu;
    if (ea == 0xffu) {
        if (fa != 0u || (eb == 0xffu && (fb != 0u || a != b))) {
            return F32_QNAN;
        }
        return a;
    }
    if (eb == 0xffu) {
        if (fb != 0u) {
            return F32_QNAN;
        }
        return b;
    }
    if ((a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u) {
        return a & b;
    }
    let ua = f32_unpack(a);
    let ub = f32_unpack(b);
    let sa = a >> 31u;
    let sb = b >> 31u;
    let ma = ua.m << 3u;
    let mb = ub.m << 3u;
    let swap = ub.e > ua.e || (ub.e == ua.e && mb > ma);
    var eh = ua.e;
    var mh = ma;
    var sh = sa;
    var el = ub.e;
    var ml = mb;
    if (swap) {
        eh = ub.e;
        mh = mb;
        sh = sb;
        el = ua.e;
        ml = ma;
    }
    ml = shr32_jam(ml, u32(eh - el));
    if (sa == sb) {
        var m = mh + ml;
        var e = eh;
        if (m >= (1u << 27u)) {
            m = (m >> 1u) | (m & 1u);
            e = e + 1;
        }
        return f32_round_pack(sh, e, m);
    }
    let m = mh - ml;
    if (m == 0u) {
        return 0u;
    }
    let shift = 26 - (31 - i32(countLeadingZeros(m)));
    return f32_round_pack(sh, eh - shift, m << u32(shift));
}

fn f32_sub(a: u32, b: u32) -> u32 {
    return f32_add(a, b ^ 0x80000000u);
}

fn f32_neg(a: u32) -> u32 {
    return a ^ 0x80000000u;
}

fn f32_mul(a: u32, b: u32) -> u32 {
    let s = (a ^ b) >> 31u;
    let ea = (a >> 23u) & 0xffu;
    let eb = (b >> 23u) & 0xffu;
    if (ea == 0xffu || eb == 0xffu) {
        if (f32_is_nan(a) || f32_is_nan(b)
            || (ea == 0xffu && (b & 0x7fffffffu) == 0u)
            || (eb == 0xffu && (a & 0x7fffffffu) == 0u)) {
            return F32_QNAN;
        }
        return (s << 31u) | F32_INF;
    }
    if ((a & 0x7fffffffu) == 0u || (b & 0x7fffffffu) == 0u) {
        return s << 31u;
    }
    let ua = f32_unpack(a);
    let ub = f32_unpack(b);
    let p = mul32x32(ua.m, ub.m);
    // product in [2^46, 2^48)
    if (p.hi >= (1u << 15u)) {
        let m = shr64_jam(p.hi, p.lo, 21u);
        return f32_round_pack(s, ua.e + ub.e - 126, m.lo);
    }
    let m = shr64_jam(p.hi, p.lo, 20u);
    return f32_round_pack(s, ua.e + ub.e - 127, m.lo);
}

fn f32_div(a: u32, b: u32) -> u32 {
    let s = (a ^ b) >> 31u;
    if (f32_is_nan(a) || f32_is_nan(b)) {
        return F32_QNAN;
    }
    let ea = (a >> 23u) & 0xffu;
    let eb = (b >> 23u) & 0xffu;
    let a_zero = (a & 0x7fffffffu) == 0u;
    let b_zero = (b & 0x7fffffffu) == 0u;
    if (ea == 0xffu) {
        if (eb == 0xffu) {
            return F32_QNAN;
        }
        return (s << 31u) | F32_INF;
    }
    if (eb == 0xffu) {
        return s << 31u;
    }
    if (b_zero) {
        if (a_zero) {
            return F32_QNAN;
        }
        return (s << 31u) | F32_INF;
    }
    if (a_zero) {
        return s << 31u;
    }
    let ua = f32_unpack(a);
    let ub = f32_unpack(b);
    var ma = ua.m;
    let mb = ub.m;
    var e = ua.e - ub.e + 127;
    if (ma < mb) {
        ma = ma << 1u;
        e = e - 1;
    }
    var rem = ma - mb;
    var q = 1u;
    for (var i = 0u; i < 26u; i = i + 1u) {
        q = q << 1u;
        rem = rem << 1u;
        if (rem >= mb) {
            rem = rem - mb;
            q = q | 1u;
        }
    }
    var sticky = 0u;
    if (rem != 0u) {
        sticky = 1u;
    }
    return f32_round_pack(s, e, q | sticky);
}

fn f32_sqrt(a: u32) -> u32 {
    if ((a & 0x7fffffffu) == 0u) {
        return a;
    }
    if (f32_is_nan(a) || (a >> 31u) == 1u) {
        return F32_QNAN;
    }
    if (a == F32_INF) {
        return a;
    }
    let ua = f32_unpack(a);
    var mx = ua.m;
    var ex = ua.e - 150;
    if ((ex & 1) != 0) {
        mx = mx << 1u;
        ex = ex - 1;
    }
    let n = shl64(0u, mx, 28u);
    var xh = n.hi;
    var xl = n.lo;
    var ch = 0u;
    var cl = 0u;
    var dh = 1u << 20u;
    var dl = 0u;
    for (var i = 0u; i < 27u; i = i + 1u) {
        let t = add64(ch, cl, dh, dl);
        if (ge64(xh, xl, t.hi, t.lo)) {
            let r = sub64(xh, xl, t.hi, t.lo);
            xh = r.hi;
            xl = r.lo;
            let q = shr64(ch, cl, 1u);
            let n2 = add64(q.hi, q.lo, dh, dl);
            ch = n2.hi;
            cl = n2.lo;
        } else {
            let q = shr64(ch, cl, 1u);
            ch = q.hi;
            cl = q.lo;
        }
        let d2 = shr64(dh, dl, 2u);
        dh = d2.hi;
        dl = d2.lo;
    }
    let root = cl;
    var sticky = 0u;
    if (xh != 0u || xl != 0u) {
        sticky = 1u;
    }
    let half = (ex - 28) / 2;
    if (root >= (1u << 26u)) {
        return f32_round_pack(0u, half + 153, root | sticky);
    }
    return f32_round_pack(0u, half + 152, (root << 1u) | sticky);
}

// Rust `a <= b` semantics (NaN false, -0 == +0).
fn f32_le(a: u32, b: u32) -> bool {
    if (f32_is_nan(a) || f32_is_nan(b)) {
        return false;
    }
    if ((a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u) {
        return true;
    }
    var ka = a | 0x80000000u;
    if ((a >> 31u) == 1u) {
        ka = ~a;
    }
    var kb = b | 0x80000000u;
    if ((b >> 31u) == 1u) {
        kb = ~b;
    }
    return ka <= kb;
}

// Rust `x as u32`: truncate toward zero, saturate, NaN -> 0.
fn f32_trunc_u32(x: u32) -> u32 {
    if (f32_is_nan(x) || (x >> 31u) == 1u) {
        return 0u;
    }
    let e = (x >> 23u) & 0xffu;
    if (e < 127u) {
        return 0u;
    }
    let k = e - 127u;
    if (k >= 32u) {
        return 0xffffffffu;
    }
    let m = (x & 0x007fffffu) | 0x00800000u;
    if (k <= 23u) {
        return m >> (23u - k);
    }
    return m << (k - 23u);
}

// Exact widening conversion (Rust `x as f64`).
fn f32_to_f64(x: u32) -> U64 {
    let s = x >> 31u;
    let e = (x >> 23u) & 0xffu;
    let f = x & 0x007fffffu;
    if (e == 0xffu) {
        if (f != 0u) {
            return U64(0x7ff80000u, 0u);
        }
        return U64((s << 31u) | 0x7ff00000u, 0u);
    }
    if ((x & 0x7fffffffu) == 0u) {
        return U64(s << 31u, 0u);
    }
    let u = f32_unpack(x);
    let frac = u.m & 0x007fffffu;
    let be = u32(u.e + 896);
    return U64((s << 31u) | (be << 20u) | (frac >> 3u), frac << 29u);
}

// ---------------------------------------------------------------------------
// f64 softfloat ((hi, lo) pairs)
// ---------------------------------------------------------------------------

fn f64_is_nan(h: u32, l: u32) -> bool {
    let m = h & 0x7fffffffu;
    return m > 0x7ff00000u || (m == 0x7ff00000u && l != 0u);
}

fn f64_is_zero(h: u32, l: u32) -> bool {
    return (h & 0x7fffffffu) == 0u && l == 0u;
}

struct Unpack64 {
    e: i32,
    mh: u32,
    ml: u32,
}

fn f64_norm_sub(fh: u32, fl: u32) -> Unpack64 {
    let shift = clz64(fh, fl) - 11u;
    let m = shl64(fh, fl, shift);
    return Unpack64(1 - i32(shift), m.hi, m.lo);
}

fn f64_unpack(h: u32, l: u32) -> Unpack64 {
    let e = (h >> 20u) & 0x7ffu;
    let fh = h & 0x000fffffu;
    if (e == 0u) {
        return f64_norm_sub(fh, l);
    }
    return Unpack64(i32(e), fh | 0x00100000u, l);
}

// RNE pack; value = M * 2^(e-1078), M < 2^56 with GRS at bits 2..0.
fn f64_round_pack(sign: u32, e_in: i32, mh_in: u32, ml_in: u32) -> U64 {
    var e = e_in;
    var mh = mh_in;
    var ml = ml_in;
    if (e <= 0) {
        let j = shr64_jam(mh, ml, u32(1 - e));
        mh = j.hi;
        ml = j.lo;
        e = 1;
    } else if (e > 2046) {
        return U64((sign << 31u) | 0x7ff00000u, 0u);
    }
    let round_bits = ml & 7u;
    let a = add64(mh, ml, 0u, 4u);
    let r = shr64(a.hi, a.lo, 3u);
    var rl = r.lo;
    if (round_bits == 4u) {
        rl = rl & ~1u;
    }
    return U64((sign << 31u) + (u32(e - 1) << 20u) + r.hi, rl);
}

fn f64_add(ah: u32, al: u32, bh: u32, bl: u32) -> U64 {
    let ea = (ah >> 20u) & 0x7ffu;
    let eb = (bh >> 20u) & 0x7ffu;
    if (ea == 0x7ffu) {
        if (f64_is_nan(ah, al) || (eb == 0x7ffu && (f64_is_nan(bh, bl) || ah != bh))) {
            return U64(0x7ff80000u, 0u);
        }
        return U64(ah, al);
    }
    if (eb == 0x7ffu) {
        if (f64_is_nan(bh, bl)) {
            return U64(0x7ff80000u, 0u);
        }
        return U64(bh, bl);
    }
    if (f64_is_zero(ah, al) && f64_is_zero(bh, bl)) {
        return U64(ah & bh, 0u);
    }
    let ua = f64_unpack(ah, al);
    let ub = f64_unpack(bh, bl);
    let sa = ah >> 31u;
    let sb = bh >> 31u;
    let ma = shl64(ua.mh, ua.ml, 3u);
    let mb = shl64(ub.mh, ub.ml, 3u);
    let swap = ub.e > ua.e || (ub.e == ua.e && !ge64(ma.hi, ma.lo, mb.hi, mb.lo));
    var eh = ua.e;
    var hh = ma.hi;
    var hl = ma.lo;
    var sh = sa;
    var el = ub.e;
    var lh = mb.hi;
    var ll = mb.lo;
    if (swap) {
        eh = ub.e;
        hh = mb.hi;
        hl = mb.lo;
        sh = sb;
        el = ua.e;
        lh = ma.hi;
        ll = ma.lo;
    }
    let low = shr64_jam(lh, ll, u32(eh - el));
    lh = low.hi;
    ll = low.lo;
    if (sa == sb) {
        var m = add64(hh, hl, lh, ll);
        var e = eh;
        if (ge64(m.hi, m.lo, 0x01000000u, 0u)) {
            let lowbit = m.lo & 1u;
            let q = shr64(m.hi, m.lo, 1u);
            m = U64(q.hi, q.lo | lowbit);
            e = e + 1;
        }
        return f64_round_pack(sh, e, m.hi, m.lo);
    }
    let m = sub64(hh, hl, lh, ll);
    if (m.hi == 0u && m.lo == 0u) {
        return U64(0u, 0u);
    }
    let shift = 55 - (63 - i32(clz64(m.hi, m.lo)));
    let n = shl64(m.hi, m.lo, u32(shift));
    return f64_round_pack(sh, eh - shift, n.hi, n.lo);
}

fn f64_sub(ah: u32, al: u32, bh: u32, bl: u32) -> U64 {
    return f64_add(ah, al, bh ^ 0x80000000u, bl);
}

// Shift a 128-bit value right by n (33..63) into a 64-bit pair with jam.
fn shr128_to64_jam(p3: u32, p2: u32, p1: u32, p0: u32, n: u32) -> U64 {
    let k = n - 32u;
    let rl = (p1 >> k) | (p2 << (32u - k));
    let rh = (p2 >> k) | (p3 << (32u - k));
    let lost = p0 | (p1 & ((1u << k) - 1u));
    var jam = 0u;
    if (lost != 0u) {
        jam = 1u;
    }
    return U64(rh, rl | jam);
}

fn f64_mul(ah: u32, al: u32, bh: u32, bl: u32) -> U64 {
    let s = (ah ^ bh) >> 31u;
    let ea = (ah >> 20u) & 0x7ffu;
    let eb = (bh >> 20u) & 0x7ffu;
    if (ea == 0x7ffu || eb == 0x7ffu) {
        if (f64_is_nan(ah, al) || f64_is_nan(bh, bl)
            || (ea == 0x7ffu && f64_is_zero(bh, bl))
            || (eb == 0x7ffu && f64_is_zero(ah, al))) {
            return U64(0x7ff80000u, 0u);
        }
        return U64((s << 31u) | 0x7ff00000u, 0u);
    }
    if (f64_is_zero(ah, al) || f64_is_zero(bh, bl)) {
        return U64(s << 31u, 0u);
    }
    let ua = f64_unpack(ah, al);
    let ub = f64_unpack(bh, bl);
    let p00 = mul32x32(ua.ml, ub.ml);
    let p01 = mul32x32(ua.ml, ub.mh);
    let p10 = mul32x32(ua.mh, ub.ml);
    let p11 = mul32x32(ua.mh, ub.mh);
    let p0 = p00.lo;
    let c1 = add64(0u, p00.hi, 0u, p01.lo);
    let c2 = add64(c1.hi, c1.lo, 0u, p10.lo);
    let p1 = c2.lo;
    let c3 = add64(0u, p01.hi, 0u, p10.hi);
    let c4 = add64(c3.hi, c3.lo, 0u, p11.lo);
    let c5 = add64(c4.hi, c4.lo, 0u, c2.hi);
    let p2 = c5.lo;
    let p3 = p11.hi + c5.hi;
    // P = p3:p2:p1:p0 in [2^104, 2^106)
    if (p3 >= (1u << 9u)) {
        let m = shr128_to64_jam(p3, p2, p1, p0, 50u);
        return f64_round_pack(s, ua.e + ub.e - 1022, m.hi, m.lo);
    }
    let m = shr128_to64_jam(p3, p2, p1, p0, 49u);
    return f64_round_pack(s, ua.e + ub.e - 1023, m.hi, m.lo);
}

// Rust `a <= b` for f64 (NaN false, -0 == +0).
fn f64_le(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    if (f64_is_nan(ah, al) || f64_is_nan(bh, bl)) {
        return false;
    }
    if (f64_is_zero(ah, al) && f64_is_zero(bh, bl)) {
        return true;
    }
    var kah = ah | 0x80000000u;
    var kal = al;
    if ((ah >> 31u) == 1u) {
        kah = ~ah;
        kal = ~al;
    }
    var kbh = bh | 0x80000000u;
    var kbl = bl;
    if ((bh >> 31u) == 1u) {
        kbh = ~bh;
        kbl = ~bl;
    }
    return kah < kbh || (kah == kbh && kal <= kbl);
}

fn f64_ge(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    return f64_le(bh, bl, ah, al);
}

// Rust `x as f64` for i64 pairs; RNE.
fn i64_to_f64(h: u32, l: u32) -> U64 {
    if (h == 0u && l == 0u) {
        return U64(0u, 0u);
    }
    let sign = h >> 31u;
    var mh = h;
    var ml = l;
    if (sign == 1u) {
        let m = neg64(h, l);
        mh = m.hi;
        ml = m.lo;
    }
    let msb = 63 - i32(clz64(mh, ml));
    var n = U64(mh, ml);
    if (msb <= 55) {
        n = shl64(mh, ml, u32(55 - msb));
    } else {
        n = shr64_jam(mh, ml, u32(msb - 55));
    }
    return f64_round_pack(sign, msb + 1023, n.hi, n.lo);
}

// Rust `x as i64`: truncate toward zero, saturate, NaN -> 0.
fn f64_trunc_i64(h: u32, l: u32) -> U64 {
    if (f64_is_nan(h, l)) {
        return U64(0u, 0u);
    }
    let sign = h >> 31u;
    let e = (h >> 20u) & 0x7ffu;
    if (e < 1023u) {
        return U64(0u, 0u);
    }
    let k = e - 1023u;
    if (k >= 63u) {
        if (sign == 0u) {
            return U64(0x7fffffffu, 0xffffffffu);
        }
        return U64(0x80000000u, 0u);
    }
    let fh = (h & 0x000fffffu) | 0x00100000u;
    var v = U64(fh, l);
    if (k <= 52u) {
        v = shr64(fh, l, 52u - k);
    } else {
        v = shl64(fh, l, k - 52u);
    }
    if (sign == 1u) {
        return neg64(v.hi, v.lo);
    }
    return v;
}

// Rust `x as u32` for f64.
fn f64_trunc_u32(h: u32, l: u32) -> u32 {
    if (f64_is_nan(h, l) || (h >> 31u) == 1u) {
        return 0u;
    }
    let e = (h >> 20u) & 0x7ffu;
    if (e < 1023u) {
        return 0u;
    }
    let k = e - 1023u;
    if (k >= 32u) {
        return 0xffffffffu;
    }
    let fh = (h & 0x000fffffu) | 0x00100000u;
    let v = shr64(fh, l, 52u - k);
    return v.lo;
}

// ---------------------------------------------------------------------------
// mesh core (softmesh.rs transliteration)
// ---------------------------------------------------------------------------

const C_F32_ZERO: u32 = 0x00000000u;
const C_F32_HALF: u32 = 0x3f000000u;
const C_F32_EPS20: u32 = 0x1e3ce508u; // 1e-20f32
const C_F32_BOUNDARY_WEIGHT: u32 = 0x40800000u; // 4.0

const C_F64_HALF_HI: u32 = 0x3fe00000u;
const C_F64_TWO_HI: u32 = 0x40000000u;
const C_F64_QUADRIC_FP_HI: u32 = 0x41d00000u; // 2^30
const C_F64_INV_QUADRIC_FP_HI: u32 = 0x3e100000u; // 2^-30
const C_F64_COST_FP_HI: u32 = 0x41b00000u; // 2^28
const C_F64_NINE_E18_HI: u32 = 0x43df399bu;
const C_F64_NINE_E18_LO: u32 = 0x1438a100u;
const C_F64_U32_MAX_HI: u32 = 0x41efffffu;
const C_F64_U32_MAX_LO: u32 = 0xffe00000u;

const C_I64_SAT_POS_HI: u32 = 0x7ce66c50u;
const C_I64_SAT_POS_LO: u32 = 0xe2840000u;
const C_I64_SAT_NEG_HI: u32 = 0x831993afu;
const C_I64_SAT_NEG_LO: u32 = 0x1d7c0000u;

const CULLED: u32 = 0xffffffffu;

struct V3 {
    x: u32,
    y: u32,
    z: u32,
}

fn load_pos(i: u32) -> V3 {
    return V3(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
}

fn params_mn() -> V3 {
    return V3(P.mn0, P.mn1, P.mn2);
}

fn ssub3(a: V3, b: V3) -> V3 {
    return V3(f32_sub(a.x, b.x), f32_sub(a.y, b.y), f32_sub(a.z, b.z));
}

fn scross3(a: V3, b: V3) -> V3 {
    return V3(
        f32_sub(f32_mul(a.y, b.z), f32_mul(a.z, b.y)),
        f32_sub(f32_mul(a.z, b.x), f32_mul(a.x, b.z)),
        f32_sub(f32_mul(a.x, b.y), f32_mul(a.y, b.x)),
    );
}

fn sdot3(a: V3, b: V3) -> u32 {
    return f32_add(
        f32_add(f32_mul(a.x, b.x), f32_mul(a.y, b.y)),
        f32_mul(a.z, b.z),
    );
}

fn slen3(a: V3) -> u32 {
    return f32_sqrt(sdot3(a, a));
}

fn scell_axis(v: u32, mn: u32, inv_cell: u32, dim: u32) -> u32 {
    let t = f32_mul(f32_sub(v, mn), inv_cell);
    if (f32_le(t, C_F32_ZERO)) {
        return 0u;
    }
    let c = f32_trunc_u32(t);
    if (c >= dim) {
        return dim - 1u;
    }
    return c;
}

fn scell_index(p: V3, mn: V3, ic: V3, d0: u32, d1: u32, d2: u32) -> u32 {
    let cx = scell_axis(p.x, mn.x, ic.x, d0);
    let cy = scell_axis(p.y, mn.y, ic.y, d1);
    let cz = scell_axis(p.z, mn.z, ic.z, d2);
    return (cz * d1 + cy) * d0 + cx;
}

fn snorm_pos(p: V3, mn: V3, inv_ext: u32) -> V3 {
    return V3(
        f32_mul(f32_sub(p.x, mn.x), inv_ext),
        f32_mul(f32_sub(p.y, mn.y), inv_ext),
        f32_mul(f32_sub(p.z, mn.z), inv_ext),
    );
}

struct Plane {
    ok: bool,
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    area: u32,
}

fn stri_plane(a: V3, b: V3, c: V3) -> Plane {
    let n = scross3(ssub3(b, a), ssub3(c, a));
    let twice_area = slen3(n);
    if (f32_le(twice_area, C_F32_EPS20)) {
        return Plane(false, 0u, 0u, 0u, 0u, 0u);
    }
    let u = V3(
        f32_div(n.x, twice_area),
        f32_div(n.y, twice_area),
        f32_div(n.z, twice_area),
    );
    return Plane(
        true,
        u.x,
        u.y,
        u.z,
        f32_neg(sdot3(u, a)),
        f32_mul(C_F32_HALF, twice_area),
    );
}

fn sedge_plane(a: V3, b: V3, face_n: V3) -> Plane {
    let perp = scross3(ssub3(b, a), face_n);
    let l = slen3(perp);
    if (f32_le(l, C_F32_EPS20)) {
        return Plane(false, 0u, 0u, 0u, 0u, 0u);
    }
    let u = V3(f32_div(perp.x, l), f32_div(perp.y, l), f32_div(perp.z, l));
    return Plane(true, u.x, u.y, u.z, f32_neg(sdot3(u, a)), 0u);
}

// mesh_coarsen::fp: saturating f64 -> fixed-point i64.
fn sfp(x: U64) -> U64 {
    var r: U64;
    if (f64_ge(x.hi, x.lo, 0u, 0u)) {
        r = f64_add(x.hi, x.lo, C_F64_HALF_HI, 0u);
    } else {
        r = f64_sub(x.hi, x.lo, C_F64_HALF_HI, 0u);
    }
    if (f64_ge(r.hi, r.lo, C_F64_NINE_E18_HI, C_F64_NINE_E18_LO)) {
        return U64(C_I64_SAT_POS_HI, C_I64_SAT_POS_LO);
    }
    if (f64_le(r.hi, r.lo, C_F64_NINE_E18_HI ^ 0x80000000u, C_F64_NINE_E18_LO)) {
        return U64(C_I64_SAT_NEG_HI, C_I64_SAT_NEG_LO);
    }
    return f64_trunc_i64(r.hi, r.lo);
}

struct Quad {
    ok: bool,
    q: array<U64, 10>,
}

fn wxy(w: U64, x: U64, y: U64) -> U64 {
    let wx = f64_mul(w.hi, w.lo, x.hi, x.lo);
    return f64_mul(wx.hi, wx.lo, y.hi, y.lo);
}

fn splane_quadric_fp(p0: u32, p1: u32, p2: u32, p3: u32, weight: u32) -> array<U64, 10> {
    let wt = f32_to_f64(weight);
    let w = f64_mul(wt.hi, wt.lo, C_F64_QUADRIC_FP_HI, 0u);
    let a = f32_to_f64(p0);
    let b = f32_to_f64(p1);
    let c = f32_to_f64(p2);
    let d = f32_to_f64(p3);
    var q: array<U64, 10>;
    q[0] = sfp(wxy(w, a, a));
    q[1] = sfp(wxy(w, a, b));
    q[2] = sfp(wxy(w, a, c));
    q[3] = sfp(wxy(w, a, d));
    q[4] = sfp(wxy(w, b, b));
    q[5] = sfp(wxy(w, b, c));
    q[6] = sfp(wxy(w, b, d));
    q[7] = sfp(wxy(w, c, c));
    q[8] = sfp(wxy(w, c, d));
    q[9] = sfp(wxy(w, d, d));
    return q;
}

fn seval_cost_fp(q: array<U64, 10>, p: V3) -> u32 {
    let x = f32_to_f64(p.x);
    let y = f32_to_f64(p.y);
    let z = f32_to_f64(p.z);
    var s: U64;
    var t: U64;
    var qf: U64;
    qf = i64_to_f64(q[0].hi, q[0].lo);
    t = f64_mul(qf.hi, qf.lo, x.hi, x.lo);
    s = f64_mul(t.hi, t.lo, x.hi, x.lo);
    qf = i64_to_f64(q[1].hi, q[1].lo);
    t = f64_mul(qf.hi, qf.lo, x.hi, x.lo);
    t = f64_mul(t.hi, t.lo, y.hi, y.lo);
    t = f64_mul(C_F64_TWO_HI, 0u, t.hi, t.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[2].hi, q[2].lo);
    t = f64_mul(qf.hi, qf.lo, x.hi, x.lo);
    t = f64_mul(t.hi, t.lo, z.hi, z.lo);
    t = f64_mul(C_F64_TWO_HI, 0u, t.hi, t.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[3].hi, q[3].lo);
    t = f64_mul(qf.hi, qf.lo, x.hi, x.lo);
    t = f64_mul(C_F64_TWO_HI, 0u, t.hi, t.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[4].hi, q[4].lo);
    t = f64_mul(qf.hi, qf.lo, y.hi, y.lo);
    t = f64_mul(t.hi, t.lo, y.hi, y.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[5].hi, q[5].lo);
    t = f64_mul(qf.hi, qf.lo, y.hi, y.lo);
    t = f64_mul(t.hi, t.lo, z.hi, z.lo);
    t = f64_mul(C_F64_TWO_HI, 0u, t.hi, t.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[6].hi, q[6].lo);
    t = f64_mul(qf.hi, qf.lo, y.hi, y.lo);
    t = f64_mul(C_F64_TWO_HI, 0u, t.hi, t.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[7].hi, q[7].lo);
    t = f64_mul(qf.hi, qf.lo, z.hi, z.lo);
    t = f64_mul(t.hi, t.lo, z.hi, z.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[8].hi, q[8].lo);
    t = f64_mul(qf.hi, qf.lo, z.hi, z.lo);
    t = f64_mul(C_F64_TWO_HI, 0u, t.hi, t.lo);
    s = f64_add(s.hi, s.lo, t.hi, t.lo);
    qf = i64_to_f64(q[9].hi, q[9].lo);
    s = f64_add(s.hi, s.lo, qf.hi, qf.lo);
    // s / 2^30 == s * 2^-30 (exact reciprocal, same RN result)
    let cost = f64_mul(s.hi, s.lo, C_F64_INV_QUADRIC_FP_HI, 0u);
    if (f64_le(cost.hi, cost.lo, 0u, 0u)) {
        return 0u;
    }
    let r = f64_mul(cost.hi, cost.lo, C_F64_COST_FP_HI, 0u);
    if (f64_ge(r.hi, r.lo, C_F64_U32_MAX_HI, C_F64_U32_MAX_LO)) {
        return 0xffffffffu;
    }
    return f64_trunc_u32(r.hi, r.lo);
}

fn saccum_tri_quadric(pa: V3, pb: V3, pc: V3, mn: V3, inv_ext: u32) -> Quad {
    let a = snorm_pos(pa, mn, inv_ext);
    let b = snorm_pos(pb, mn, inv_ext);
    let c = snorm_pos(pc, mn, inv_ext);
    let pl = stri_plane(a, b, c);
    var out: Quad;
    out.ok = pl.ok;
    if (pl.ok) {
        out.q = splane_quadric_fp(pl.p0, pl.p1, pl.p2, pl.p3, pl.area);
    }
    return out;
}

fn saccum_edge_quadric(pu: V3, pv: V3, pw: V3, mn: V3, inv_ext: u32) -> Quad {
    let a = snorm_pos(pu, mn, inv_ext);
    let b = snorm_pos(pv, mn, inv_ext);
    let c = snorm_pos(pw, mn, inv_ext);
    var out: Quad;
    out.ok = false;
    let face = stri_plane(a, b, c);
    if (!face.ok) {
        return out;
    }
    let pl = sedge_plane(a, b, V3(face.p0, face.p1, face.p2));
    if (!pl.ok) {
        return out;
    }
    let e = ssub3(b, a);
    out.ok = true;
    out.q = splane_quadric_fp(
        pl.p0,
        pl.p1,
        pl.p2,
        pl.p3,
        f32_mul(sdot3(e, e), C_F32_BOUNDARY_WEIGHT),
    );
    return out;
}

// Order-independent wrapping i64 accumulate: the lo-word sum mod 2^32 and
// the total carry count floor(sum_lo / 2^32) are both order-free, so the
// final pair equals the sequential wrapping i64 sum.
fn atom_add_i64(cell: u32, term: u32, v: U64) {
    let idx = (cell * 10u + term) * 2u;
    let prev = atomicAdd(&cell_q[idx], v.lo);
    var hi = v.hi;
    if (prev + v.lo < prev) {
        hi = hi + 1u;
    }
    atomicAdd(&cell_q[idx + 1u], hi);
}

fn accum_quad_to_cell(cell: u32, q: array<U64, 10>) {
    for (var i = 0u; i < 10u; i = i + 1u) {
        atom_add_i64(cell, i, q[i]);
    }
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn mesh_survey(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = P.base + gid.x;
    if (t >= P.ntris) {
        return;
    }
    let mn = params_mn();
    let a = load_pos(indices[t * 3u]);
    let b = load_pos(indices[t * 3u + 1u]);
    let c = load_pos(indices[t * 3u + 2u]);
    for (var s = 0u; s < P.nscales; s = s + 1u) {
        let g = grids[s];
        let ic = V3(g.ic0, g.ic1, g.ic2);
        let ca = scell_index(a, mn, ic, g.d0, g.d1, g.d2);
        let cb = scell_index(b, mn, ic, g.d0, g.d1, g.d2);
        let cc = scell_index(c, mn, ic, g.d0, g.d1, g.d2);
        if (ca != cb && cb != cc && ca != cc) {
            atomicAdd(&counts[s], 1u);
        }
    }
}

@compute @workgroup_size(256)
fn mesh_accum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = P.base + gid.x;
    if (t >= P.ntris) {
        return;
    }
    let mn = params_mn();
    let ic = V3(P.ic0, P.ic1, P.ic2);
    let a = load_pos(indices[t * 3u]);
    let b = load_pos(indices[t * 3u + 1u]);
    let c = load_pos(indices[t * 3u + 2u]);
    var quad = saccum_tri_quadric(a, b, c, mn, P.inv_ext);
    if (!quad.ok) {
        return;
    }
    let ca = scell_index(a, mn, ic, P.d0, P.d1, P.d2);
    let cb = scell_index(b, mn, ic, P.d0, P.d1, P.d2);
    let cc = scell_index(c, mn, ic, P.d0, P.d1, P.d2);
    accum_quad_to_cell(ca, quad.q);
    accum_quad_to_cell(cb, quad.q);
    accum_quad_to_cell(cc, quad.q);
}

@compute @workgroup_size(256)
fn mesh_accum_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = P.base + gid.x;
    if (e >= P.nedges) {
        return;
    }
    let mn = params_mn();
    let ic = V3(P.ic0, P.ic1, P.ic2);
    let u = load_pos(edges[e * 3u]);
    let v = load_pos(edges[e * 3u + 1u]);
    let w = load_pos(edges[e * 3u + 2u]);
    var quad = saccum_edge_quadric(u, v, w, mn, P.inv_ext);
    if (!quad.ok) {
        return;
    }
    let cu = scell_index(u, mn, ic, P.d0, P.d1, P.d2);
    let cv = scell_index(v, mn, ic, P.d0, P.d1, P.d2);
    accum_quad_to_cell(cu, quad.q);
    accum_quad_to_cell(cv, quad.q);
}

// Init fill for the pick stages: every word of best_cost/best_id becomes
// the exact constant 0xffffffff. One disjoint write per cell, no arithmetic;
// replaces the host-side 0xff memset+upload.
@compute @workgroup_size(256)
fn mesh_fill_ff(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = P.base + gid.x;
    if (i >= P.ncells) {
        return;
    }
    atomicStore(&best_cost[i], 0xffffffffu);
    atomicStore(&best_id[i], 0xffffffffu);
}

@compute @workgroup_size(256)
fn mesh_pick_cost(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid = P.base + gid.x;
    if (vid >= P.nverts) {
        return;
    }
    let mn = params_mn();
    let ic = V3(P.ic0, P.ic1, P.ic2);
    let pos = load_pos(vid);
    let cid = scell_index(pos, mn, ic, P.d0, P.d1, P.d2);
    var q: array<U64, 10>;
    for (var i = 0u; i < 10u; i = i + 1u) {
        let idx = (cid * 10u + i) * 2u;
        q[i] = U64(atomicLoad(&cell_q[idx + 1u]), atomicLoad(&cell_q[idx]));
    }
    let cost = seval_cost_fp(q, snorm_pos(pos, mn, P.inv_ext));
    vert_cost[vid] = cost;
    atomicMin(&best_cost[cid], cost);
}

@compute @workgroup_size(256)
fn mesh_pick_id(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid = P.base + gid.x;
    if (vid >= P.nverts) {
        return;
    }
    let mn = params_mn();
    let ic = V3(P.ic0, P.ic1, P.ic2);
    let pos = load_pos(vid);
    let cid = scell_index(pos, mn, ic, P.d0, P.d1, P.d2);
    if (vert_cost[vid] == atomicLoad(&best_cost[cid])) {
        atomicMin(&best_id[cid], vid);
    }
}

@compute @workgroup_size(256)
fn mesh_remap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = P.base + gid.x;
    if (t >= P.ntris) {
        return;
    }
    let mn = params_mn();
    let ic = V3(P.ic0, P.ic1, P.ic2);
    let a = load_pos(indices[t * 3u]);
    let b = load_pos(indices[t * 3u + 1u]);
    let c = load_pos(indices[t * 3u + 2u]);
    let ca = scell_index(a, mn, ic, P.d0, P.d1, P.d2);
    let cb = scell_index(b, mn, ic, P.d0, P.d1, P.d2);
    let cc = scell_index(c, mn, ic, P.d0, P.d1, P.d2);
    if (!(ca != cb && cb != cc && ca != cc)) {
        out_tris[t * 3u] = CULLED;
        return;
    }
    out_tris[t * 3u] = atomicLoad(&best_id[ca]);
    out_tris[t * 3u + 1u] = atomicLoad(&best_id[cb]);
    out_tris[t * 3u + 2u] = atomicLoad(&best_id[cc]);
}
