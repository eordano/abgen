//! Integer-only IEEE-754 float emulation: the host mirror of the WGSL
//! softfloat in shaders/mesh_coarsen.wgsl.
//!
//! Why: WGSL has no f64, no i64, and its native f32 gives no byte-exactness
//! guarantees (fma contraction is permitted, denormals may flush, div/sqrt
//! are not correctly rounded). The mesh coarsen core (kernel-ptx
//! core/mesh_coarsen.rs) is defined in terms of host IEEE f32/f64 arithmetic,
//! so the wgpu mesh kernels perform every float operation in u32 integer
//! arithmetic (round-to-nearest-even softfloat). Integer ops are exact on
//! every backend, which makes GPU output bit-equal to the CPU oracle by
//! construction.
//!
//! Every function uses only u32 operations plus i32 exponent bookkeeping
//! (64-bit values travel as (hi, lo) pairs) so the WGSL port is a mechanical
//! transliteration. Tests check this mirror against hardware IEEE arithmetic;
//! a shader test checks the WGSL against the mirror.
//!
//! NaN payloads are canonical (0x7fc00000 / 0x7ff8000000000000) rather than
//! propagated: NaN-free inputs (the mesh contract, shared with the CUDA lane)
//! never observe the difference.

pub const F32_QNAN: u32 = 0x7fc0_0000;
pub const F32_INF: u32 = 0x7f80_0000;
pub const F64_QNAN: (u32, u32) = (0x7ff8_0000, 0);

#[inline]
fn clz32(x: u32) -> u32 {
    x.leading_zeros()
}

#[inline]
pub fn mul32x32(a: u32, b: u32) -> (u32, u32) {
    let a0 = a & 0xffff;
    let a1 = a >> 16;
    let b0 = b & 0xffff;
    let b1 = b >> 16;
    let ll = a0.wrapping_mul(b0);
    let lh = a0.wrapping_mul(b1);
    let hl = a1.wrapping_mul(b0);
    let hh = a1.wrapping_mul(b1);
    let mid = (ll >> 16)
        .wrapping_add(lh & 0xffff)
        .wrapping_add(hl & 0xffff);
    let lo = (ll & 0xffff) | (mid << 16);
    let hi = hh
        .wrapping_add(lh >> 16)
        .wrapping_add(hl >> 16)
        .wrapping_add(mid >> 16);
    (hi, lo)
}

#[inline]
pub fn add64(ah: u32, al: u32, bh: u32, bl: u32) -> (u32, u32) {
    let lo = al.wrapping_add(bl);
    let carry = if lo < al { 1u32 } else { 0 };
    (ah.wrapping_add(bh).wrapping_add(carry), lo)
}

#[inline]
pub fn sub64(ah: u32, al: u32, bh: u32, bl: u32) -> (u32, u32) {
    let lo = al.wrapping_sub(bl);
    let borrow = if al < bl { 1u32 } else { 0 };
    (ah.wrapping_sub(bh).wrapping_sub(borrow), lo)
}

#[inline]
pub fn neg64(h: u32, l: u32) -> (u32, u32) {
    sub64(0, 0, h, l)
}

#[inline]
pub fn shl64(h: u32, l: u32, n: u32) -> (u32, u32) {
    if n == 0 {
        (h, l)
    } else if n < 32 {
        ((h << n) | (l >> (32 - n)), l << n)
    } else if n < 64 {
        (l << (n - 32), 0)
    } else {
        (0, 0)
    }
}

#[inline]
pub fn shr64(h: u32, l: u32, n: u32) -> (u32, u32) {
    if n == 0 {
        (h, l)
    } else if n < 32 {
        (h >> n, (l >> n) | (h << (32 - n)))
    } else if n < 64 {
        (0, h >> (n - 32))
    } else {
        (0, 0)
    }
}

/// Right shift with the shifted-out bits jammed (OR-ed) into bit 0.
#[inline]
pub fn shr64_jam(h: u32, l: u32, n: u32) -> (u32, u32) {
    if n == 0 {
        (h, l)
    } else if n < 32 {
        let lost = l & ((1u32 << n) - 1);
        let (sh, sl) = shr64(h, l, n);
        (sh, sl | if lost != 0 { 1 } else { 0 })
    } else if n < 64 {
        let lost = l | (h & if n == 32 { 0 } else { (1u32 << (n - 32)) - 1 });
        (0, (h >> (n - 32)) | if lost != 0 { 1 } else { 0 })
    } else {
        (0, if h != 0 || l != 0 { 1 } else { 0 })
    }
}

#[inline]
pub fn shr32_jam(m: u32, n: u32) -> u32 {
    if n == 0 {
        m
    } else if n < 32 {
        (m >> n) | if m & ((1u32 << n) - 1) != 0 { 1 } else { 0 }
    } else if m != 0 {
        1
    } else {
        0
    }
}

#[inline]
pub fn clz64(h: u32, l: u32) -> u32 {
    if h != 0 {
        clz32(h)
    } else {
        32 + clz32(l)
    }
}

#[inline]
pub fn ge64(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    ah > bh || (ah == bh && al >= bl)
}

// ---------------------------------------------------------------------------
// f32 (bits in u32)
// ---------------------------------------------------------------------------

#[inline]
fn f32_is_nan(x: u32) -> bool {
    (x & 0x7fff_ffff) > F32_INF
}

/// Normalize a subnormal fraction: (adjusted biased exponent, significand
/// with bit 23 set).
#[inline]
fn f32_norm_sub(f: u32) -> (i32, u32) {
    let shift = clz32(f) - 8;
    (1 - shift as i32, f << shift)
}

#[inline]
fn f32_unpack(x: u32) -> (i32, u32) {
    let e = (x >> 23) & 0xff;
    let f = x & 0x007f_ffff;
    if e == 0 {
        f32_norm_sub(f)
    } else {
        (e as i32, f | 0x0080_0000)
    }
}

/// RNE pack. `m` carries 3 GRS bits at bits 2..0 and (when normal) the
/// leading 1 at bit 26; the represented value is m * 2^(e-153). m < 2^27.
/// Packs additively so subnormal results, round-up to normal, mantissa
/// carry, and round-up to infinity all fall out of the arithmetic.
fn f32_round_pack(sign: u32, e: i32, m: u32) -> u32 {
    let mut e = e;
    let mut m = m;
    if e <= 0 {
        m = shr32_jam(m, (1 - e) as u32);
        e = 1;
    } else if e > 254 {
        return (sign << 31) | F32_INF;
    }
    let round_bits = m & 7;
    let mut mr = (m + 4) >> 3;
    if round_bits == 4 {
        mr &= !1u32;
    }
    (sign << 31)
        .wrapping_add(((e - 1) as u32) << 23)
        .wrapping_add(mr)
}

pub fn f32_add(a: u32, b: u32) -> u32 {
    let ea = (a >> 23) & 0xff;
    let eb = (b >> 23) & 0xff;
    let fa = a & 0x007f_ffff;
    let fb = b & 0x007f_ffff;
    if ea == 0xff {
        if fa != 0 || (eb == 0xff && (fb != 0 || a != b)) {
            return F32_QNAN;
        }
        return a;
    }
    if eb == 0xff {
        if fb != 0 {
            return F32_QNAN;
        }
        return b;
    }
    if (a & 0x7fff_ffff) == 0 && (b & 0x7fff_ffff) == 0 {
        return a & b;
    }
    let (ea_i, ma) = f32_unpack(a);
    let (eb_i, mb) = f32_unpack(b);
    let sa = a >> 31;
    let sb = b >> 31;
    let ma = ma << 3;
    let mb = mb << 3;
    let swap = eb_i > ea_i || (eb_i == ea_i && mb > ma);
    let (eh, mh, sh, el, ml) = if swap {
        (eb_i, mb, sb, ea_i, ma)
    } else {
        (ea_i, ma, sa, eb_i, mb)
    };
    let ml = shr32_jam(ml, (eh - el) as u32);
    if sa == sb {
        let mut m = mh + ml;
        let mut e = eh;
        if m >= 1 << 27 {
            m = (m >> 1) | (m & 1);
            e += 1;
        }
        f32_round_pack(sh, e, m)
    } else {
        let m = mh - ml;
        if m == 0 {
            return 0;
        }
        let shift = 26 - (31 - clz32(m) as i32);
        f32_round_pack(sh, eh - shift, m << shift as u32)
    }
}

#[inline]
pub fn f32_sub(a: u32, b: u32) -> u32 {
    f32_add(a, b ^ 0x8000_0000)
}

#[inline]
pub fn f32_neg(a: u32) -> u32 {
    a ^ 0x8000_0000
}

pub fn f32_mul(a: u32, b: u32) -> u32 {
    let s = (a ^ b) >> 31;
    let ea = (a >> 23) & 0xff;
    let eb = (b >> 23) & 0xff;
    if ea == 0xff || eb == 0xff {
        if f32_is_nan(a)
            || f32_is_nan(b)
            || (ea == 0xff && (b & 0x7fff_ffff) == 0)
            || (eb == 0xff && (a & 0x7fff_ffff) == 0)
        {
            return F32_QNAN;
        }
        return (s << 31) | F32_INF;
    }
    if (a & 0x7fff_ffff) == 0 || (b & 0x7fff_ffff) == 0 {
        return s << 31;
    }
    let (ea_i, ma) = f32_unpack(a);
    let (eb_i, mb) = f32_unpack(b);
    let (ph, pl) = mul32x32(ma, mb);
    // product in [2^46, 2^48)
    if ph >= 1 << 15 {
        let (_, m) = shr64_jam(ph, pl, 21);
        f32_round_pack(s, ea_i + eb_i - 126, m)
    } else {
        let (_, m) = shr64_jam(ph, pl, 20);
        f32_round_pack(s, ea_i + eb_i - 127, m)
    }
}

pub fn f32_div(a: u32, b: u32) -> u32 {
    let s = (a ^ b) >> 31;
    if f32_is_nan(a) || f32_is_nan(b) {
        return F32_QNAN;
    }
    let ea = (a >> 23) & 0xff;
    let eb = (b >> 23) & 0xff;
    let a_zero = (a & 0x7fff_ffff) == 0;
    let b_zero = (b & 0x7fff_ffff) == 0;
    if ea == 0xff {
        if eb == 0xff {
            return F32_QNAN;
        }
        return (s << 31) | F32_INF;
    }
    if eb == 0xff {
        return s << 31;
    }
    if b_zero {
        if a_zero {
            return F32_QNAN;
        }
        return (s << 31) | F32_INF;
    }
    if a_zero {
        return s << 31;
    }
    let (ea_i, ma) = f32_unpack(a);
    let (eb_i, mb) = f32_unpack(b);
    // normalize the ratio into [1, 2), then long-divide 26 fraction bits
    let mut ma = ma;
    let mut e = ea_i - eb_i + 127;
    if ma < mb {
        ma <<= 1;
        e -= 1;
    }
    let mut rem = ma - mb;
    let mut q = 1u32;
    let mut i = 0;
    while i < 26 {
        q <<= 1;
        rem <<= 1;
        if rem >= mb {
            rem -= mb;
            q |= 1;
        }
        i += 1;
    }
    // q = floor(ma * 2^26 / mb) in [2^26, 2^27)
    let sticky = if rem != 0 { 1u32 } else { 0 };
    f32_round_pack(s, e, q | sticky)
}

pub fn f32_sqrt(a: u32) -> u32 {
    if (a & 0x7fff_ffff) == 0 {
        return a; // +-0
    }
    if f32_is_nan(a) || a >> 31 == 1 {
        return F32_QNAN;
    }
    if a == F32_INF {
        return a;
    }
    let (ea_i, ma) = f32_unpack(a);
    let mut mx = ma;
    let mut ex = ea_i - 150; // value = mx * 2^ex
    if ex & 1 != 0 {
        mx <<= 1;
        ex -= 1;
    }
    // N = mx << 28 in [2^51, 2^53); value = N * 2^(ex-28)
    let (nh, nl) = shl64(0, mx, 28);
    // root = floor(sqrt(N)) in [2^25, 2^27)
    let mut xh = nh;
    let mut xl = nl;
    let mut ch = 0u32;
    let mut cl = 0u32;
    let mut dh = 1u32 << 20; // 2^52
    let mut dl = 0u32;
    let mut i = 0;
    while i < 27 {
        let (th, tl) = add64(ch, cl, dh, dl);
        if ge64(xh, xl, th, tl) {
            let (rh, rl) = sub64(xh, xl, th, tl);
            xh = rh;
            xl = rl;
            let (qh, ql) = shr64(ch, cl, 1);
            let (nh2, nl2) = add64(qh, ql, dh, dl);
            ch = nh2;
            cl = nl2;
        } else {
            let (qh, ql) = shr64(ch, cl, 1);
            ch = qh;
            cl = ql;
        }
        let (eh2, el2) = shr64(dh, dl, 2);
        dh = eh2;
        dl = el2;
        i += 1;
    }
    let root = cl;
    let sticky = if xh != 0 || xl != 0 { 1u32 } else { 0 };
    let half = (ex - 28) / 2;
    if root >= 1 << 26 {
        f32_round_pack(0, half + 153, root | sticky)
    } else {
        f32_round_pack(0, half + 152, (root << 1) | sticky)
    }
}

/// Rust `a <= b` semantics (NaN -> false, -0 == +0).
pub fn f32_le(a: u32, b: u32) -> bool {
    if f32_is_nan(a) || f32_is_nan(b) {
        return false;
    }
    if (a & 0x7fff_ffff) == 0 && (b & 0x7fff_ffff) == 0 {
        return true;
    }
    let ka = if a >> 31 == 1 { !a } else { a | 0x8000_0000 };
    let kb = if b >> 31 == 1 { !b } else { b | 0x8000_0000 };
    ka <= kb
}

/// Rust `x as u32` semantics: truncate toward zero, saturate, NaN -> 0.
pub fn f32_trunc_u32(x: u32) -> u32 {
    if f32_is_nan(x) || x >> 31 == 1 {
        return 0;
    }
    let e = (x >> 23) & 0xff;
    if e < 127 {
        return 0;
    }
    let k = e - 127;
    if k >= 32 {
        return u32::MAX;
    }
    let m = (x & 0x007f_ffff) | 0x0080_0000;
    if k <= 23 {
        m >> (23 - k)
    } else {
        m << (k - 23)
    }
}

/// Exact widening conversion, matching Rust `x as f64`.
pub fn f32_to_f64(x: u32) -> (u32, u32) {
    let s = x >> 31;
    let e = (x >> 23) & 0xff;
    let f = x & 0x007f_ffff;
    if e == 0xff {
        if f != 0 {
            return F64_QNAN;
        }
        return ((s << 31) | 0x7ff0_0000, 0);
    }
    if (x & 0x7fff_ffff) == 0 {
        return (s << 31, 0);
    }
    let (e_i, m) = f32_unpack(x);
    let frac = m & 0x007f_ffff;
    let be = (e_i + 896) as u32;
    ((s << 31) | (be << 20) | (frac >> 3), frac << 29)
}

// ---------------------------------------------------------------------------
// f64 (bits as (hi, lo) pair)
// ---------------------------------------------------------------------------

#[inline]
fn f64_is_nan(h: u32, l: u32) -> bool {
    let m = h & 0x7fff_ffff;
    m > 0x7ff0_0000 || (m == 0x7ff0_0000 && l != 0)
}

#[inline]
fn f64_is_zero(h: u32, l: u32) -> bool {
    (h & 0x7fff_ffff) == 0 && l == 0
}

#[inline]
fn f64_norm_sub(fh: u32, fl: u32) -> (i32, u32, u32) {
    let shift = clz64(fh, fl) - 11;
    let (mh, ml) = shl64(fh, fl, shift);
    (1 - shift as i32, mh, ml)
}

#[inline]
fn f64_unpack(h: u32, l: u32) -> (i32, u32, u32) {
    let e = (h >> 20) & 0x7ff;
    let fh = h & 0x000f_ffff;
    if e == 0 {
        f64_norm_sub(fh, l)
    } else {
        (e as i32, fh | 0x0010_0000, l)
    }
}

/// RNE pack. Significand pair carries 3 GRS bits at bits 2..0 and (when
/// normal) the leading 1 at pair bit 55; the value is M * 2^(e-1078).
/// M < 2^56.
fn f64_round_pack(sign: u32, e: i32, mh: u32, ml: u32) -> (u32, u32) {
    let mut e = e;
    let (mut mh, mut ml) = (mh, ml);
    if e <= 0 {
        let (jh, jl) = shr64_jam(mh, ml, (1 - e) as u32);
        mh = jh;
        ml = jl;
        e = 1;
    } else if e > 2046 {
        return ((sign << 31) | 0x7ff0_0000, 0);
    }
    let round_bits = ml & 7;
    let (ah, al) = add64(mh, ml, 0, 4);
    let (rh, mut rl) = shr64(ah, al, 3);
    if round_bits == 4 {
        rl &= !1u32;
    }
    (
        (sign << 31)
            .wrapping_add(((e - 1) as u32) << 20)
            .wrapping_add(rh),
        rl,
    )
}

pub fn f64_add(ah: u32, al: u32, bh: u32, bl: u32) -> (u32, u32) {
    let ea = (ah >> 20) & 0x7ff;
    let eb = (bh >> 20) & 0x7ff;
    if ea == 0x7ff {
        if f64_is_nan(ah, al) || (eb == 0x7ff && (f64_is_nan(bh, bl) || ah != bh)) {
            return F64_QNAN;
        }
        return (ah, al);
    }
    if eb == 0x7ff {
        if f64_is_nan(bh, bl) {
            return F64_QNAN;
        }
        return (bh, bl);
    }
    if f64_is_zero(ah, al) && f64_is_zero(bh, bl) {
        return (ah & bh, 0);
    }
    let (ea_i, mah, mal) = f64_unpack(ah, al);
    let (eb_i, mbh, mbl) = f64_unpack(bh, bl);
    let sa = ah >> 31;
    let sb = bh >> 31;
    let (mah, mal) = shl64(mah, mal, 3);
    let (mbh, mbl) = shl64(mbh, mbl, 3);
    let swap = eb_i > ea_i || (eb_i == ea_i && !ge64(mah, mal, mbh, mbl));
    let (eh, hh, hl, sh, el, lh, ll) = if swap {
        (eb_i, mbh, mbl, sb, ea_i, mah, mal)
    } else {
        (ea_i, mah, mal, sa, eb_i, mbh, mbl)
    };
    let (lh, ll) = shr64_jam(lh, ll, (eh - el) as u32);
    if sa == sb {
        let (mut mh, mut ml) = add64(hh, hl, lh, ll);
        let mut e = eh;
        if ge64(mh, ml, 0x0100_0000, 0) {
            // >= 2^56
            let low = ml & 1;
            let (qh, ql) = shr64(mh, ml, 1);
            mh = qh;
            ml = ql | low;
            e += 1;
        }
        f64_round_pack(sh, e, mh, ml)
    } else {
        let (mh, ml) = sub64(hh, hl, lh, ll);
        if mh == 0 && ml == 0 {
            return (0, 0);
        }
        let shift = 55 - (63 - clz64(mh, ml) as i32);
        let (nh, nl) = shl64(mh, ml, shift as u32);
        f64_round_pack(sh, eh - shift, nh, nl)
    }
}

#[inline]
pub fn f64_sub(ah: u32, al: u32, bh: u32, bl: u32) -> (u32, u32) {
    f64_add(ah, al, bh ^ 0x8000_0000, bl)
}

/// Shift a 128-bit value right by n (33..63) into a 64-bit pair with jam.
/// The value must fit 64 bits after the shift.
#[inline]
fn shr128_to64_jam(p3: u32, p2: u32, p1: u32, p0: u32, n: u32) -> (u32, u32) {
    let k = n - 32; // 1..31
    let rl = (p1 >> k) | (p2 << (32 - k));
    let rh = (p2 >> k) | (p3 << (32 - k));
    let lost = p0 | (p1 & ((1u32 << k) - 1));
    (rh, rl | if lost != 0 { 1 } else { 0 })
}

pub fn f64_mul(ah: u32, al: u32, bh: u32, bl: u32) -> (u32, u32) {
    let s = (ah ^ bh) >> 31;
    let ea = (ah >> 20) & 0x7ff;
    let eb = (bh >> 20) & 0x7ff;
    if ea == 0x7ff || eb == 0x7ff {
        if f64_is_nan(ah, al)
            || f64_is_nan(bh, bl)
            || (ea == 0x7ff && f64_is_zero(bh, bl))
            || (eb == 0x7ff && f64_is_zero(ah, al))
        {
            return F64_QNAN;
        }
        return ((s << 31) | 0x7ff0_0000, 0);
    }
    if f64_is_zero(ah, al) || f64_is_zero(bh, bl) {
        return (s << 31, 0);
    }
    let (ea_i, mah, mal) = f64_unpack(ah, al);
    let (eb_i, mbh, mbl) = f64_unpack(bh, bl);
    // 53x53 -> 106-bit product; limbs a1:a0 x b1:b0
    let (p00h, p00l) = mul32x32(mal, mbl);
    let (p01h, p01l) = mul32x32(mal, mbh);
    let (p10h, p10l) = mul32x32(mah, mbl);
    let (p11h, p11l) = mul32x32(mah, mbh);
    let p0 = p00l;
    let (c1, p1a) = add64(0, p00h, 0, p01l);
    let (c2, p1) = add64(c1, p1a, 0, p10l);
    let (c3, p2a) = add64(0, p01h, 0, p10h);
    let (c4, p2b) = add64(c3, p2a, 0, p11l);
    let (c5, p2) = add64(c4, p2b, 0, c2);
    let p3 = p11h.wrapping_add(c5);
    // P = p3:p2:p1:p0 in [2^104, 2^106)
    if p3 >= 1 << 9 {
        let (mh, ml) = shr128_to64_jam(p3, p2, p1, p0, 50);
        f64_round_pack(s, ea_i + eb_i - 1022, mh, ml)
    } else {
        let (mh, ml) = shr128_to64_jam(p3, p2, p1, p0, 49);
        f64_round_pack(s, ea_i + eb_i - 1023, mh, ml)
    }
}

/// Rust `a <= b` for f64 pairs (NaN -> false, -0 == +0).
pub fn f64_le(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    if f64_is_nan(ah, al) || f64_is_nan(bh, bl) {
        return false;
    }
    if f64_is_zero(ah, al) && f64_is_zero(bh, bl) {
        return true;
    }
    let (kah, kal) = if ah >> 31 == 1 {
        (!ah, !al)
    } else {
        (ah | 0x8000_0000, al)
    };
    let (kbh, kbl) = if bh >> 31 == 1 {
        (!bh, !bl)
    } else {
        (bh | 0x8000_0000, bl)
    };
    kah < kbh || (kah == kbh && kal <= kbl)
}

/// Rust `a >= b` for f64 pairs.
pub fn f64_ge(ah: u32, al: u32, bh: u32, bl: u32) -> bool {
    f64_le(bh, bl, ah, al)
}

/// Rust `x as f64` for i64 (hi, lo) pairs; RNE.
pub fn i64_to_f64(h: u32, l: u32) -> (u32, u32) {
    if h == 0 && l == 0 {
        return (0, 0);
    }
    let sign = h >> 31;
    let (mh, ml) = if sign == 1 { neg64(h, l) } else { (h, l) };
    let msb = 63 - clz64(mh, ml) as i32;
    let (nh, nl) = if msb <= 55 {
        shl64(mh, ml, (55 - msb) as u32)
    } else {
        shr64_jam(mh, ml, (msb - 55) as u32)
    };
    f64_round_pack(sign, msb + 1023, nh, nl)
}

/// Rust `x as i64` semantics for an f64 pair: truncate toward zero,
/// saturate, NaN -> 0. Returns an i64 as (hi, lo).
pub fn f64_trunc_i64(h: u32, l: u32) -> (u32, u32) {
    if f64_is_nan(h, l) {
        return (0, 0);
    }
    let sign = h >> 31;
    let e = (h >> 20) & 0x7ff;
    if e < 1023 {
        return (0, 0);
    }
    let k = e - 1023;
    if k >= 63 {
        // magnitude >= 2^63
        if sign == 0 {
            return (0x7fff_ffff, 0xffff_ffff);
        }
        return (0x8000_0000, 0);
    }
    let fh = (h & 0x000f_ffff) | 0x0010_0000;
    let fl = l;
    let (vh, vl) = if k <= 52 {
        shr64(fh, fl, 52 - k)
    } else {
        shl64(fh, fl, k - 52)
    };
    if sign == 1 {
        neg64(vh, vl)
    } else {
        (vh, vl)
    }
}

/// Rust `x as u32` semantics for an f64 pair.
pub fn f64_trunc_u32(h: u32, l: u32) -> u32 {
    if f64_is_nan(h, l) || h >> 31 == 1 {
        return 0;
    }
    let e = (h >> 20) & 0x7ff;
    if e < 1023 {
        return 0;
    }
    let k = e - 1023;
    if k >= 32 {
        return u32::MAX;
    }
    let fh = (h & 0x000f_ffff) | 0x0010_0000;
    let (_, vl) = shr64(fh, l, 52 - k);
    vl
}
