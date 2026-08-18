use std::sync::OnceLock;

const BLOCK_SIZE: usize = 8;
const PIXELS_PER_BLOCK: usize = 16;

#[inline]
fn pack_565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = ((r as u16) >> 3) & 0x1F;
    let g6 = ((g as u16) >> 2) & 0x3F;
    let b5 = ((b as u16) >> 3) & 0x1F;
    (r5 << 11) | (g6 << 5) | b5
}

#[inline]
fn unpack_565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1F) as u8;
    let g = ((c >> 5) & 0x3F) as u8;
    let b = (c & 0x1F) as u8;

    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

// NOTE(merge dxt1m): o02's planar/branchless encode_block restructure was
// ported and parity-verified here, but benched 1.5-5.4% SLOWER than this
// original on Graviton2 on top of the o03 LUT+NEON base (the compiler already
// turns the palette loop into fully-unrolled csel code held in registers, so
// the planar transpose only added work). Reverted to the original; see
// ~/results/dxt1m.json.
fn encode_block(rgba: &[u8; 64]) -> [u8; BLOCK_SIZE] {
    let mut pix = [[0u8; 3]; 16];
    for i in 0..16 {
        pix[i] = [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]];
    }

    let mut mean = [0f32; 3];
    for p in &pix {
        for k in 0..3 {
            mean[k] += p[k] as f32;
        }
    }
    for k in 0..3 {
        mean[k] /= 16.0;
    }

    let mut cov = [[0f32; 3]; 3];
    for p in &pix {
        let d = [
            p[0] as f32 - mean[0],
            p[1] as f32 - mean[1],
            p[2] as f32 - mean[2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += d[a] * d[b];
            }
        }
    }
    let mut axis = [1f32, 1f32, 1f32];
    for _ in 0..6 {
        let mut n = [0f32; 3];
        for a in 0..3 {
            for b in 0..3 {
                n[a] += cov[a][b] * axis[b];
            }
        }
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if mag < 1e-6 {
            axis = [1.0, 1.0, 1.0];
            break;
        }
        axis = [n[0] / mag, n[1] / mag, n[2] / mag];
    }

    let mut min_dot = f32::INFINITY;
    let mut max_dot = f32::NEG_INFINITY;
    let mut min_i = 0usize;
    let mut max_i = 0usize;
    for (i, p) in pix.iter().enumerate() {
        let d = (p[0] as f32 - mean[0]) * axis[0]
            + (p[1] as f32 - mean[1]) * axis[1]
            + (p[2] as f32 - mean[2]) * axis[2];
        if d < min_dot {
            min_dot = d;
            min_i = i;
        }
        if d > max_dot {
            max_dot = d;
            max_i = i;
        }
    }
    let mut c0 = pack_565(pix[max_i][0], pix[max_i][1], pix[max_i][2]);
    let mut c1 = pack_565(pix[min_i][0], pix[min_i][1], pix[min_i][2]);

    if c0 == c1 {
        if c1 > 0 {
            c1 -= 1;
        } else {
            c0 += 1;
        }
    }
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }

    let ep0 = unpack_565(c0);
    let ep1 = unpack_565(c1);
    let palette: [[u8; 3]; 4] = [
        ep0,
        ep1,
        [
            ((2u16 * ep0[0] as u16 + ep1[0] as u16) / 3) as u8,
            ((2u16 * ep0[1] as u16 + ep1[1] as u16) / 3) as u8,
            ((2u16 * ep0[2] as u16 + ep1[2] as u16) / 3) as u8,
        ],
        [
            ((ep0[0] as u16 + 2u16 * ep1[0] as u16) / 3) as u8,
            ((ep0[1] as u16 + 2u16 * ep1[1] as u16) / 3) as u8,
            ((ep0[2] as u16 + 2u16 * ep1[2] as u16) / 3) as u8,
        ],
    ];

    let mut bits = 0u32;
    for (i, p) in pix.iter().enumerate() {
        let mut best = 0u32;
        let mut best_err = i32::MAX;
        for (k, pc) in palette.iter().enumerate() {
            let dr = p[0] as i32 - pc[0] as i32;
            let dg = p[1] as i32 - pc[1] as i32;
            let db = p[2] as i32 - pc[2] as i32;
            let e = dr * dr + dg * dg + db * db;
            if e < best_err {
                best_err = e;
                best = k as u32;
            }
        }
        bits |= best << (2 * i);
    }

    let mut out = [0u8; BLOCK_SIZE];
    out[0] = (c0 & 0xFF) as u8;
    out[1] = ((c0 >> 8) & 0xFF) as u8;
    out[2] = (c1 & 0xFF) as u8;
    out[3] = ((c1 >> 8) & 0xFF) as u8;
    out[4] = (bits & 0xFF) as u8;
    out[5] = ((bits >> 8) & 0xFF) as u8;
    out[6] = ((bits >> 16) & 0xFF) as u8;
    out[7] = ((bits >> 24) & 0xFF) as u8;
    out
}

fn pad_to_block_size(rgba: &[u8], w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let pw = (w + 3) & !3;
    let ph = (h + 3) & !3;
    if pw == w && ph == h {
        return (rgba.to_vec(), w, h);
    }

    let mut out = vec![0u8; pw * ph * 4];
    for y in 0..ph {
        let sy = y % h;
        for x in 0..pw {
            let sx = x % w;
            let s = (sy * w + sx) * 4;
            let d = (y * pw + x) * 4;
            out[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    (out, pw, ph)
}

fn srgb_to_linear_u8(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        crate::detmath::powf((s + 0.055) / 1.055, 2.4)
    }
}

fn linear_to_srgb_u8(lin: f32) -> u8 {
    let lin = lin.clamp(0.0, 1.0);
    let s = if lin <= 0.0031308 {
        12.92 * lin
    } else {
        1.055 * crate::detmath::powf(lin, 1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Exact per-byte table of `srgb_to_linear_u8`; identical by construction.
fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| srgb_to_linear_u8(i as u8)))
}

/// `t[k]` = smallest f32 in [0.0, 1.0] for which `linear_to_srgb_u8` returns
/// >= k, found by binary search over the (order-preserving) bit patterns of
/// positive floats against the exact scalar function. `linear_to_srgb_u8` is
/// monotone non-decreasing on [0, 1] (verified exhaustively over every f32 in
/// that range by `linear_to_srgb_fast_matches_scalar_exhaustive`), so a table
/// lookup reproduces the scalar function bit-for-bit.
fn linear_to_srgb_thresholds() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 256];
        for (k, slot) in t.iter_mut().enumerate().skip(1) {
            // Invariant: f(from_bits(lo)) < k <= f(from_bits(hi)).
            let mut lo = 0u32; // 0.0 -> 0
            let mut hi = 0x3F80_0000u32; // 1.0 -> 255
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if (linear_to_srgb_u8(f32::from_bits(mid)) as usize) < k {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            *slot = f32::from_bits(hi);
        }
        t
    })
}

/// Bit-identical replacement for `linear_to_srgb_u8` (see
/// `linear_to_srgb_thresholds`): returns the largest k with `t[k] <= lin`.
fn linear_to_srgb_u8_fast(lin: f32, t: &[f32; 256]) -> u8 {
    let lin = lin.clamp(0.0, 1.0);
    let mut k = 0usize;
    let mut step = 128usize;
    while step > 0 {
        let n = k + step;
        if t[n & 0xFF] <= lin {
            k = n;
        }
        step >>= 1;
    }
    k as u8
}

fn box_halve_rgba(arr: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if w > 1 && h > 1 {
            return box_halve_rgba_neon2x2(arr, w, h);
        }
    }
    box_halve_rgba_scalar(arr, w, h)
}

/// 2x2 box filter, one output RGBA pixel per float32x4 lane group. Matches the
/// scalar accumulation order exactly: ((p00 + p01) + p10) + p11, then / 4.0.
#[cfg(target_arch = "aarch64")]
fn box_halve_rgba_neon2x2(arr: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    use std::arch::aarch64::*;
    debug_assert!(w > 1 && h > 1);
    let nw = w / 2;
    let nh = h / 2;
    let row_stride = w * 4;
    let mut out = vec![0f32; nh * nw * 4];
    unsafe {
        let denom = vdupq_n_f32(4.0);
        for ny in 0..nh {
            let r0 = (ny * 2) * row_stride;
            let r1 = r0 + row_stride;
            let ob = ny * nw * 4;
            for nx in 0..nw {
                let x0 = nx * 8;
                let p00 = vld1q_f32(arr.as_ptr().add(r0 + x0));
                let p01 = vld1q_f32(arr.as_ptr().add(r0 + x0 + 4));
                let p10 = vld1q_f32(arr.as_ptr().add(r1 + x0));
                let p11 = vld1q_f32(arr.as_ptr().add(r1 + x0 + 4));
                let s = vaddq_f32(vaddq_f32(vaddq_f32(p00, p01), p10), p11);
                vst1q_f32(out.as_mut_ptr().add(ob + nx * 4), vdivq_f32(s, denom));
            }
        }
    }
    (out, nw, nh)
}

fn box_halve_rgba_scalar(arr: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let c = 4usize;
    let nh = (h / 2).max(1);
    let nw = (w / 2).max(1);
    let fh = if h > 1 { 2 } else { 1 };
    let fw = if w > 1 { 2 } else { 1 };
    let denom = (fh * fw) as f32;
    let mut out = vec![0f32; nh * nw * c];
    let row_stride = w * c;
    for ny in 0..nh {
        for nx in 0..nw {
            for ch in 0..c {
                let mut acc = 0f32;
                for dy in 0..fh {
                    for dx in 0..fw {
                        let y = ny * fh + dy;
                        let x = nx * fw + dx;
                        acc += arr[y * row_stride + x * c + ch];
                    }
                }
                out[(ny * nw + nx) * c + ch] = acc / denom;
            }
        }
    }
    (out, nw, nh)
}

// `as u8` saturates (negative -> 0, > 255 -> 255, NaN -> 0), which is exactly
// what the branchy original computed.
#[inline]
fn round_half_up_u8(v: f32) -> u8 {
    (v + 0.5).floor() as u8
}

#[cfg(not(target_arch = "aarch64"))]
fn round_half_up_u8_slice(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = round_half_up_u8(*s);
    }
}

/// NEON lane-parallel `round_half_up_u8`: floor(v + 0.5) clamped to [0, 255],
/// identical semantics per lane (vrndmq is floor; values are integral before
/// the truncating convert).
#[cfg(target_arch = "aarch64")]
fn round_half_up_u8_slice(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    let mut i = 0usize;
    {
        use std::arch::aarch64::*;
        let n = src.len();
        unsafe {
            let half = vdupq_n_f32(0.5);
            let zero = vdupq_n_f32(0.0);
            let hi = vdupq_n_f32(255.0);
            while i + 8 <= n {
                let a = vld1q_f32(src.as_ptr().add(i));
                let b = vld1q_f32(src.as_ptr().add(i + 4));
                let a = vminq_f32(vmaxq_f32(vrndmq_f32(vaddq_f32(a, half)), zero), hi);
                let b = vminq_f32(vmaxq_f32(vrndmq_f32(vaddq_f32(b, half)), zero), hi);
                let a = vcvtq_u32_f32(a);
                let b = vcvtq_u32_f32(b);
                let w = vcombine_u16(vqmovn_u32(a), vqmovn_u32(b));
                vst1_u8(dst.as_mut_ptr().add(i), vqmovn_u16(w));
                i += 8;
            }
        }
    }
    for j in i..src.len() {
        dst[j] = round_half_up_u8(src[j]);
    }
}

pub fn encode_dxt1_mip_chain(
    rgba: &[u8],
    width: u32,
    height: u32,
    mip_count: Option<i32>,
    flip: bool,
    srgb: bool,
) -> (Vec<u8>, i32) {
    let params = [
        mip_count.map(i64::from).unwrap_or(-1),
        flip as i64,
        srgb as i64,
    ];
    crate::texencode_cache::get_or_encode(
        crate::texencode_cache::Kind::Dxt1,
        rgba,
        width,
        height,
        &params,
        || {
            Some(encode_dxt1_mip_chain_uncached(
                rgba, width, height, mip_count, flip, srgb,
            ))
        },
    )
    .expect("dxt1 encode closure always returns Some")
}

fn encode_dxt1_mip_chain_uncached(
    rgba: &[u8],
    width: u32,
    height: u32,
    mip_count: Option<i32>,
    flip: bool,
    srgb: bool,
) -> (Vec<u8>, i32) {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(rgba.len(), w * h * 4);
    let flipped: Vec<u8> = if flip {
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            let src = &rgba[(h - 1 - y) * w * 4..(h - 1 - y) * w * 4 + w * 4];
            out[y * w * 4..y * w * 4 + w * 4].copy_from_slice(src);
        }
        out
    } else {
        rgba.to_vec()
    };

    let mip_count = mip_count.unwrap_or_else(|| {
        let m = width.max(height).max(1) as f64;
        (crate::detmath::log2(m).floor() as i32) + 1
    });

    let mut cur: Vec<f32> = vec![0f32; w * h * 4];
    if srgb {
        let lut = srgb_to_linear_lut();
        for i in 0..(w * h) {
            cur[i * 4] = lut[flipped[i * 4] as usize];
            cur[i * 4 + 1] = lut[flipped[i * 4 + 1] as usize];
            cur[i * 4 + 2] = lut[flipped[i * 4 + 2] as usize];
            cur[i * 4 + 3] = flipped[i * 4 + 3] as f32;
        }
    } else {
        for (dst, src) in cur.iter_mut().zip(flipped.iter()) {
            *dst = *src as f32;
        }
    }
    let mut cw = w;
    let mut ch = h;

    let mut parts: Vec<u8> = Vec::new();
    for m in 0..mip_count {
        let mut level = vec![0u8; cw * ch * 4];
        if srgb {
            let t = linear_to_srgb_thresholds();
            for i in 0..(cw * ch) {
                level[i * 4] = linear_to_srgb_u8_fast(cur[i * 4], t);
                level[i * 4 + 1] = linear_to_srgb_u8_fast(cur[i * 4 + 1], t);
                level[i * 4 + 2] = linear_to_srgb_u8_fast(cur[i * 4 + 2], t);
                level[i * 4 + 3] = round_half_up_u8(cur[i * 4 + 3]);
            }
        } else {
            round_half_up_u8_slice(&cur, &mut level);
        }
        let (padded, pw, ph) = pad_to_block_size(&level, cw, ch);
        let bw = pw / 4;
        let bh = ph / 4;
        let row_bytes = pw * 4;
        for by in 0..bh {
            for bx in 0..bw {
                let mut block = [0u8; PIXELS_PER_BLOCK * 4];
                let base = by * 4 * row_bytes + bx * 16;
                for r in 0..4 {
                    let start = base + r * row_bytes;
                    block[r * 16..r * 16 + 16].copy_from_slice(&padded[start..start + 16]);
                }
                let enc = encode_block(&block);
                parts.extend_from_slice(&enc);
            }
        }
        if m < mip_count - 1 {
            let (next, nw, nh) = box_halve_rgba(&cur, cw, ch);
            cur = next;
            cw = nw;
            ch = nh;
        }
    }
    (parts, mip_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_block_is_8_bytes() {
        let mut rgba = vec![0u8; 16 * 4];
        for i in 0..16 {
            rgba[i * 4] = 0xAA;
            rgba[i * 4 + 1] = 0x55;
            rgba[i * 4 + 2] = 0x33;
            rgba[i * 4 + 3] = 0xFF;
        }
        let (data, mips) = encode_dxt1_mip_chain(&rgba, 4, 4, Some(1), false, false);
        assert_eq!(data.len(), 8);
        assert_eq!(mips, 1);

        assert!(data.iter().any(|&b| b != 0));
    }

    #[test]
    fn mip_chain_byte_count_matches_block_math() {
        let rgba = vec![0xFFu8; 8 * 8 * 4];
        let (data, mips) = encode_dxt1_mip_chain(&rgba, 8, 8, None, false, false);
        assert_eq!(mips, 4);
        assert_eq!(data.len(), 7 * 8);
    }

    #[test]
    fn mip_chain_512_matches_prod_byte_count() {
        let rgba = vec![0x80u8; 512 * 512 * 4];
        let (data, mips) = encode_dxt1_mip_chain(&rgba, 512, 512, None, false, false);
        assert_eq!(mips, 10);
        assert_eq!(data.len(), 174_776);
    }

    #[test]
    fn always_4_color_mode() {
        let mut rgba = vec![0u8; 16 * 4];
        for i in 0..16 {
            rgba[i * 4] = (i * 16) as u8;
            rgba[i * 4 + 1] = (i * 16) as u8;
            rgba[i * 4 + 2] = (i * 16) as u8;
            rgba[i * 4 + 3] = 0xFF;
        }
        let (data, _) = encode_dxt1_mip_chain(&rgba, 4, 4, Some(1), false, false);
        let c0 = u16::from_le_bytes([data[0], data[1]]);
        let c1 = u16::from_le_bytes([data[2], data[3]]);
        assert!(c0 >= c1, "block must be in 4-color mode (c0 >= c1)");
    }

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    #[test]
    fn linear_to_srgb_fast_matches_scalar_sampled() {
        let t = linear_to_srgb_thresholds();
        // Every threshold, its predecessor ulp, and its successor ulp.
        for k in 1..256usize {
            let bits = t[k].to_bits();
            for b in [bits.wrapping_sub(1), bits, bits + 1] {
                let x = f32::from_bits(b);
                assert_eq!(
                    linear_to_srgb_u8(x),
                    linear_to_srgb_u8_fast(x, t),
                    "mismatch at bits {b:#010x}"
                );
            }
        }
        // Out-of-range and edge values.
        for x in [-1.0f32, -0.0, 0.0, 1.0, 1.5, f32::MIN_POSITIVE, 1e-30] {
            assert_eq!(linear_to_srgb_u8(x), linear_to_srgb_u8_fast(x, t));
        }
        // Random samples across [0, 1].
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2_000_000 {
            let b = (xorshift(&mut s) as u32) % 0x3F80_0001;
            let x = f32::from_bits(b);
            assert_eq!(
                linear_to_srgb_u8(x),
                linear_to_srgb_u8_fast(x, t),
                "mismatch at bits {b:#010x}"
            );
        }
    }

    /// Proves bit-identity of the threshold lookup for every possible input:
    /// all f32 bit patterns in [0.0, 1.0] (inputs outside that range clamp
    /// identically in both paths). ~1.07e9 powf calls; run explicitly with
    /// `cargo test --release -- --ignored linear_to_srgb_fast_matches_scalar_exhaustive`.
    #[test]
    #[ignore]
    fn linear_to_srgb_fast_matches_scalar_exhaustive() {
        let nthreads = 4u32;
        let mut handles = Vec::new();
        for tid in 0..nthreads {
            handles.push(std::thread::spawn(move || {
                let t = linear_to_srgb_thresholds();
                let mut bad = 0u64;
                let mut bits = tid;
                while bits <= 0x3F80_0000 {
                    let x = f32::from_bits(bits);
                    if linear_to_srgb_u8(x) != linear_to_srgb_u8_fast(x, t) {
                        if bad < 8 {
                            eprintln!("mismatch at bits {bits:#010x}");
                        }
                        bad += 1;
                    }
                    bits += nthreads;
                }
                bad
            }));
        }
        let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 0, "{total} mismatching f32 inputs in [0,1]");
    }

    #[test]
    fn box_halve_matches_scalar_bitexact() {
        let mut s = 0xDEAD_BEEF_CAFE_F00Du64;
        for (w, h) in [
            (2usize, 2usize),
            (3, 3),
            (4, 4),
            (5, 7),
            (16, 16),
            (17, 9),
            (2, 64),
            (64, 2),
            (33, 31),
        ] {
            let mut arr = vec![0f32; w * h * 4];
            for v in arr.iter_mut() {
                // Mix of linear-light range values and full u8-range values.
                let r = xorshift(&mut s);
                *v = if r & 1 == 0 {
                    ((r >> 8) & 0xFF) as f32
                } else {
                    ((r >> 8) & 0xFFFFFF) as f32 / 16_777_215.0
                };
            }
            let (a, aw, ah) = box_halve_rgba(&arr, w, h);
            let (b, bw, bh) = box_halve_rgba_scalar(&arr, w, h);
            assert_eq!((aw, ah), (bw, bh));
            assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{w}x{h} idx {i}");
            }
        }
    }

    #[test]
    fn round_half_up_slice_matches_scalar() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut src = vec![0f32; 4099];
        for v in src.iter_mut() {
            let r = xorshift(&mut s);
            *v = match r % 5 {
                0 => -1.5,
                1 => 256.5,
                2 => ((r >> 8) & 0xFF) as f32 + 0.5,
                3 => 255.0,
                _ => ((r >> 8) & 0x3FFFF) as f32 / 256.0 - 1.0,
            };
        }
        src[0] = 0.0;
        src[1] = -0.0;
        src[2] = 254.5;
        src[3] = 255.5;
        let mut dst = vec![0u8; src.len()];
        round_half_up_u8_slice(&src, &mut dst);
        for (i, v) in src.iter().enumerate() {
            assert_eq!(dst[i], round_half_up_u8(*v), "idx {i} val {v}");
        }
    }

    #[test]
    fn srgb_lut_matches_scalar() {
        let lut = srgb_to_linear_lut();
        for c in 0..=255u8 {
            assert_eq!(lut[c as usize].to_bits(), srgb_to_linear_u8(c).to_bits());
        }
    }

    /// o02's independent parity gate: SHA256 over a second fixed corpus
    /// (different generator/sizes than tests/dxt1_corpus_hash.rs); the hash was
    /// recorded against the original scalar encoder and must not move.
    #[test]
    fn parity_corpus_hash() {
        struct R(u64);
        impl R {
            fn next(&mut self) -> u32 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                (x >> 32) as u32
            }
        }
        let cases: &[(usize, usize, Option<i32>, bool, bool)] = &[
            (4, 4, Some(1), false, false),
            (4, 4, None, true, true),
            (5, 7, None, false, true),
            (1, 1, None, false, false),
            (3, 9, Some(2), true, false),
            (16, 16, None, true, true),
            (64, 64, None, false, true),
            (128, 32, None, true, false),
            (33, 17, None, false, true),
            (96, 96, None, false, true),
        ];
        let mut hash = crate::hashes::Sha256::new();
        let mut rng = R(0x0DD0_BEEF_CAFE_F00D);
        for (ci, &(w, h, mips, flip, srgb)) in cases.iter().enumerate() {
            let mut rgba = vec![0u8; w * h * 4];
            match ci % 3 {
                0 => {
                    for b in rgba.iter_mut() {
                        *b = (rng.next() & 0xFF) as u8;
                    }
                }
                1 => {
                    for (i, b) in rgba.iter_mut().enumerate() {
                        *b = [0u8, 255, 128, 1, 254, 127][(i / 64) % 6];
                    }
                }
                _ => {
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) * 4;
                            rgba[i] = ((x * 255) / w.max(1)) as u8;
                            rgba[i + 1] = ((y * 255) / h.max(1)) as u8;
                            rgba[i + 2] = (rng.next() & 0xFF) as u8;
                            rgba[i + 3] = if (x + y) % 7 == 0 { 0 } else { 255 };
                        }
                    }
                }
            }
            let (data, m) = encode_dxt1_mip_chain(&rgba, w as u32, h as u32, mips, flip, srgb);
            hash.update(&(m as i64).to_le_bytes());
            hash.update(&(data.len() as u64).to_le_bytes());
            hash.update(&data);
        }
        let d = hash.finalize();
        let hex: String = d.iter().map(|b| format!("{b:02x}")).collect();
        println!("PARITY_HASH {hex}");
        assert_eq!(
            hex,
            "ab454890f11ebab06f6346a0ebd887a1dfe9ebb61d82cab0eedc5b2675f002cd"
        );
    }
}
