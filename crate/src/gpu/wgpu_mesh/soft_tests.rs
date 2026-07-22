//! Softfloat mirror vs hardware IEEE-754. Any divergence here would break
//! byte-identity of the wgpu mesh lane, so these tests are deliberately
//! heavy on edge cases (subnormals, ties, overflow, cancellation).

use super::soft::*;
use super::softmesh::*;
use crate::gpu::corelib::mesh_coarsen as mc;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = self.0;
        x ^ (x >> 33)
    }

    fn u32(&mut self) -> u32 {
        (self.next() >> 16) as u32
    }
}

const F32_SPECIALS: [u32; 16] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3f80_0000, // 1
    0xbf80_0000, // -1
    0x0000_0001, // min subnormal
    0x8000_0001,
    0x007f_ffff, // max subnormal
    0x0080_0000, // min normal
    0x7f7f_ffff, // max finite
    0xff7f_ffff,
    0x7f80_0000, // inf
    0xff80_0000,
    0x7fc0_0000, // qnan
    0x3f7f_ffff, // just under 1
    0x3400_0000, // 2^-23
    0x4b7f_ffff, // large odd int
];

fn f32_cases(n: usize) -> Vec<(u32, u32)> {
    let mut rng = Rng(0x5eed_f32a);
    let mut v = Vec::with_capacity(n + F32_SPECIALS.len() * F32_SPECIALS.len());
    for &a in &F32_SPECIALS {
        for &b in &F32_SPECIALS {
            v.push((a, b));
        }
    }
    for i in 0..n {
        let a = rng.u32();
        let b = match i % 4 {
            0 => rng.u32(),
            1 => {
                // nearby exponent: heavy cancellation and alignment cases
                let d = (rng.u32() % 5) as i32 - 2;
                let ea = ((a >> 23) & 0xff) as i32;
                let eb = (ea + d).clamp(0, 254) as u32;
                (rng.u32() & 0x807f_ffff) | (eb << 23)
            }
            2 => a ^ 0x8000_0000 ^ (rng.u32() & 0xff), // near-negation
            3 => (rng.u32() & 0x807f_ffff) | (((rng.u32() % 40) + 107) << 23),
            _ => unreachable!(),
        };
        v.push((a, b));
    }
    v
}

fn check2(op: &str, a: u32, b: u32, got: u32, want: u32) {
    let both_nan =
        f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan();
    assert!(
        got == want || both_nan,
        "{op}({a:#010x}, {b:#010x}): got {got:#010x} want {want:#010x}"
    );
}

#[test]
fn soft_f32_add_matches_hardware() {
    for (a, b) in f32_cases(120_000) {
        let want = (f32::from_bits(a) + f32::from_bits(b)).to_bits();
        check2("add", a, b, f32_add(a, b), want);
        let wsub = (f32::from_bits(a) - f32::from_bits(b)).to_bits();
        check2("sub", a, b, f32_sub(a, b), wsub);
    }
}

#[test]
fn soft_f32_mul_matches_hardware() {
    for (a, b) in f32_cases(120_000) {
        let want = (f32::from_bits(a) * f32::from_bits(b)).to_bits();
        check2("mul", a, b, f32_mul(a, b), want);
    }
}

#[test]
fn soft_f32_div_matches_hardware() {
    for (a, b) in f32_cases(120_000) {
        let want = (f32::from_bits(a) / f32::from_bits(b)).to_bits();
        check2("div", a, b, f32_div(a, b), want);
    }
}

#[test]
fn soft_f32_sqrt_matches_hardware() {
    for &a in &F32_SPECIALS {
        let want = f32::from_bits(a).sqrt().to_bits();
        check2("sqrt", a, 0, f32_sqrt(a), want);
    }
    let mut rng = Rng(0x5eed_5c47);
    for _ in 0..200_000 {
        let a = rng.u32();
        let want = f32::from_bits(a).sqrt().to_bits();
        check2("sqrt", a, 0, f32_sqrt(a), want);
    }
    // every exponent, boundary mantissas
    for e in 0..=255u32 {
        for f in [0u32, 1, 0x7f_fffe, 0x7f_ffff, 0x40_0000, 0x2a_aaaa] {
            let a = (e << 23) | f;
            let want = f32::from_bits(a).sqrt().to_bits();
            check2("sqrt", a, 0, f32_sqrt(a), want);
        }
    }
}

#[test]
fn soft_f32_compare_and_trunc_match_hardware() {
    for (a, b) in f32_cases(80_000) {
        let fa = f32::from_bits(a);
        let fb = f32::from_bits(b);
        assert_eq!(f32_le(a, b), fa <= fb, "le({a:#010x}, {b:#010x})");
        assert_eq!(f32_trunc_u32(a), fa as u32, "trunc({a:#010x})");
        let wide = (fa as f64).to_bits();
        let got = f32_to_f64(a);
        let want = ((wide >> 32) as u32, wide as u32);
        if !(fa.is_nan() && f64::from_bits(((got.0 as u64) << 32) | got.1 as u64).is_nan()) {
            assert_eq!(got, want, "to_f64({a:#010x})");
        }
    }
}

fn f64_pair(x: f64) -> (u32, u32) {
    let b = x.to_bits();
    ((b >> 32) as u32, b as u32)
}

fn pair_f64(p: (u32, u32)) -> f64 {
    f64::from_bits(((p.0 as u64) << 32) | p.1 as u64)
}

const F64_SPECIALS: [u64; 16] = [
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0000, // 1
    0xbff0_0000_0000_0000,
    0x0000_0000_0000_0001, // min subnormal
    0x000f_ffff_ffff_ffff, // max subnormal
    0x0010_0000_0000_0000, // min normal
    0x7fef_ffff_ffff_ffff, // max finite
    0xffef_ffff_ffff_ffff,
    0x7ff0_0000_0000_0000, // inf
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0000, // qnan
    0x3fef_ffff_ffff_ffff, // just under 1
    0x4340_0000_0000_0000, // 2^53
    0x43df_3776_1f00_0000, // 9e18
    0x41d0_0000_0000_0000, // 2^30
];

fn f64_cases(n: usize) -> Vec<(u64, u64)> {
    let mut rng = Rng(0x5eed_f64b);
    let mut v = Vec::new();
    for &a in &F64_SPECIALS {
        for &b in &F64_SPECIALS {
            v.push((a, b));
        }
    }
    for i in 0..n {
        let a = rng.next();
        let b = match i % 4 {
            0 => rng.next(),
            1 => {
                let d = (rng.next() % 7) as i64 - 3;
                let ea = ((a >> 52) & 0x7ff) as i64;
                let eb = (ea + d).clamp(0, 2046) as u64;
                (rng.next() & 0x800f_ffff_ffff_ffff) | (eb << 52)
            }
            2 => a ^ 0x8000_0000_0000_0000 ^ (rng.next() & 0xffff),
            3 => (rng.next() & 0x800f_ffff_ffff_ffff) | ((rng.next() % 200 + 923) << 52),
            _ => unreachable!(),
        };
        v.push((a, b));
    }
    v
}

fn check64(op: &str, a: u64, b: u64, got: (u32, u32), want: f64) {
    let gotf = pair_f64(got);
    assert!(
        gotf.to_bits() == want.to_bits() || (gotf.is_nan() && want.is_nan()),
        "{op}({a:#018x}, {b:#018x}): got {:#018x} want {:#018x}",
        gotf.to_bits(),
        want.to_bits()
    );
}

#[test]
fn soft_f64_add_matches_hardware() {
    for (a, b) in f64_cases(120_000) {
        let (ah, al) = ((a >> 32) as u32, a as u32);
        let (bh, bl) = ((b >> 32) as u32, b as u32);
        check64(
            "add64",
            a,
            b,
            f64_add(ah, al, bh, bl),
            f64::from_bits(a) + f64::from_bits(b),
        );
        check64(
            "sub64",
            a,
            b,
            f64_sub(ah, al, bh, bl),
            f64::from_bits(a) - f64::from_bits(b),
        );
    }
}

#[test]
fn soft_f64_mul_matches_hardware() {
    for (a, b) in f64_cases(120_000) {
        let (ah, al) = ((a >> 32) as u32, a as u32);
        let (bh, bl) = ((b >> 32) as u32, b as u32);
        check64(
            "mul64",
            a,
            b,
            f64_mul(ah, al, bh, bl),
            f64::from_bits(a) * f64::from_bits(b),
        );
    }
}

#[test]
fn soft_f64_compare_matches_hardware() {
    for (a, b) in f64_cases(60_000) {
        let (ah, al) = ((a >> 32) as u32, a as u32);
        let (bh, bl) = ((b >> 32) as u32, b as u32);
        let fa = f64::from_bits(a);
        let fb = f64::from_bits(b);
        assert_eq!(f64_le(ah, al, bh, bl), fa <= fb, "le64({a:#x}, {b:#x})");
        assert_eq!(f64_ge(ah, al, bh, bl), fa >= fb, "ge64({a:#x}, {b:#x})");
    }
}

#[test]
fn soft_i64_f64_conversions_match_hardware() {
    let mut rng = Rng(0x5eed_c095);
    let mut ints: Vec<i64> = vec![
        0,
        1,
        -1,
        i64::MAX,
        i64::MIN,
        i64::MAX - 1,
        (1 << 53) + 1,
        -(1 << 53) - 1,
        (1 << 53) + 2,
        9_000_000_000_000_000_000,
        -9_000_000_000_000_000_000,
    ];
    for _ in 0..120_000 {
        let sh = rng.next() % 64;
        ints.push((rng.next() >> sh) as i64);
    }
    for &x in &ints {
        let h = ((x as u64) >> 32) as u32;
        let l = x as u64 as u32;
        let got = i64_to_f64(h, l);
        assert_eq!(
            pair_f64(got).to_bits(),
            (x as f64).to_bits(),
            "i64_to_f64({x})"
        );
    }
    for (a, _) in f64_cases(80_000) {
        let (ah, al) = ((a >> 32) as u32, a as u32);
        let f = f64::from_bits(a);
        let (gh, gl) = f64_trunc_i64(ah, al);
        let got = (((gh as u64) << 32) | gl as u64) as i64;
        assert_eq!(got, f as i64, "f64_trunc_i64({a:#018x})");
        assert_eq!(f64_trunc_u32(ah, al), f as u32, "f64_trunc_u32({a:#018x})");
    }
}

#[test]
fn soft_constants_match_rust_literals() {
    assert_eq!(C_F32_HALF, 0.5f32.to_bits());
    assert_eq!(C_F32_EPS20, 1e-20f32.to_bits());
    assert_eq!(C_F32_BOUNDARY_WEIGHT, mc::BOUNDARY_WEIGHT.to_bits());
    assert_eq!(f64_pair(0.5), C_F64_HALF);
    assert_eq!(f64_pair(2.0), C_F64_TWO);
    assert_eq!(f64_pair(mc::QUADRIC_FP), C_F64_QUADRIC_FP);
    assert_eq!(f64_pair(1.0 / mc::QUADRIC_FP), C_F64_INV_QUADRIC_FP);
    assert_eq!(f64_pair(mc::COST_FP), C_F64_COST_FP);
    assert_eq!(f64_pair(9.0e18), C_F64_NINE_E18);
    assert_eq!(f64_pair(-9.0e18), C_F64_NEG_NINE_E18);
    assert_eq!(f64_pair(4_294_967_295.0), C_F64_U32_MAX);
    let sat = 9_000_000_000_000_000_000u64;
    assert_eq!(C_I64_SAT_POS, ((sat >> 32) as u32, sat as u32));
    let nsat = (-9_000_000_000_000_000_000i64) as u64;
    assert_eq!(C_I64_SAT_NEG, ((nsat >> 32) as u32, nsat as u32));
}

fn v3(p: [f32; 3]) -> V3 {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

fn gen_pos(rng: &mut Rng) -> [f32; 3] {
    let f = |r: &mut Rng| {
        let raw = (r.u32() % 2_000_000) as f32 / 100.0 - 10_000.0;
        raw
    };
    [f(rng), f(rng), f(rng)]
}

#[test]
fn softmesh_matches_core_functions() {
    let mut rng = Rng(0x0abe_11e5);
    let mn = [-3.0f32, 0.25, -10_000.0];
    let inv_ext = 1.0f32 / 17.3;
    let inv_cell = [3.7f32, 0.9, 121.0];
    let dims = [37u32, 5, 1000];
    for i in 0..40_000 {
        let a = gen_pos(&mut rng);
        let b = gen_pos(&mut rng);
        let c = gen_pos(&mut rng);
        // cell index
        assert_eq!(
            scell_index(v3(a), v3(mn), inv_cell.map(f32::to_bits), dims),
            mc::cell_index(a, mn, inv_cell, dims),
            "cell_index case {i}"
        );
        // tri plane on normalized coords
        let na = mc::norm_pos(a, mn, inv_ext);
        let nb = mc::norm_pos(b, mn, inv_ext);
        let nc = mc::norm_pos(c, mn, inv_ext);
        assert_eq!(
            snorm_pos(v3(a), v3(mn), inv_ext.to_bits()),
            v3(na),
            "norm_pos case {i}"
        );
        let want = mc::tri_plane(na, nb, nc);
        let got = stri_plane(v3(na), v3(nb), v3(nc));
        match (got, want) {
            (None, None) => {}
            (Some((gp, ga)), Some((wp, wa))) => {
                assert_eq!(gp, [wp[0].to_bits(), wp[1].to_bits(), wp[2].to_bits(), wp[3].to_bits()], "tri_plane case {i}");
                assert_eq!(ga, wa.to_bits(), "tri_area case {i}");
            }
            (g, w) => panic!("tri_plane case {i}: got {g:?} want-some={}", w.is_some()),
        }
        // full tri quadric
        let wq = mc::accum_tri_quadric(a, b, c, mn, inv_ext);
        let gq = saccum_tri_quadric(v3(a), v3(b), v3(c), v3(mn), inv_ext.to_bits());
        match (gq, wq) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                for k in 0..10 {
                    let wu = w[k] as u64;
                    assert_eq!(
                        g[k],
                        ((wu >> 32) as u32, wu as u32),
                        "tri quadric case {i} term {k}"
                    );
                }
            }
            (g, w) => panic!("tri_quadric case {i}: got-some={} want-some={}", g.is_some(), w.is_some()),
        }
        // edge quadric
        let wq = mc::accum_edge_quadric(a, b, c, mn, inv_ext);
        let gq = saccum_edge_quadric(v3(a), v3(b), v3(c), v3(mn), inv_ext.to_bits());
        match (gq, wq) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                for k in 0..10 {
                    let wu = w[k] as u64;
                    assert_eq!(
                        g[k],
                        ((wu >> 32) as u32, wu as u32),
                        "edge quadric case {i} term {k}"
                    );
                }
            }
            (g, w) => panic!("edge_quadric case {i}: got-some={} want-some={}", g.is_some(), w.is_some()),
        }
        // cost eval against the accumulated quadric (plus a perturbed one)
        if let Some(w) = wq {
            let mut q = w;
            for (k, item) in q.iter_mut().enumerate() {
                *item = item.wrapping_mul(3).wrapping_add(k as i64);
            }
            let qp: [(u32, u32); 10] =
                core::array::from_fn(|k| (((q[k] as u64) >> 32) as u32, q[k] as u64 as u32));
            let p = mc::norm_pos(a, mn, inv_ext);
            assert_eq!(
                seval_cost_fp(&qp, v3(p)),
                mc::eval_cost_fp(&q, p),
                "eval_cost case {i}"
            );
        }
    }
}

#[test]
fn softmesh_fp_saturation_matches_core() {
    for x in [
        0.0f64,
        -0.0,
        0.4999,
        0.5,
        -0.5,
        1e30,
        -1e30,
        8.9e18,
        9.1e18,
        -9.1e18,
        f64::INFINITY,
        f64::NEG_INFINITY,
        123456789.123,
    ] {
        let want = mc::fp(x) as u64;
        let got = sfp(f64_pair(x));
        assert_eq!(
            got,
            ((want >> 32) as u32, want as u32),
            "fp({x})"
        );
    }
}

#[test]
#[ignore = "large randomized sweep; run explicitly in release"]
fn soft_f32_f64_big_sweep() {
    let mut rng = Rng(0xb16_5eed);
    for _ in 0..20_000_000u64 {
        let a = rng.u32();
        let b = rng.u32();
        check2("add", a, b, f32_add(a, b), (f32::from_bits(a) + f32::from_bits(b)).to_bits());
        check2("mul", a, b, f32_mul(a, b), (f32::from_bits(a) * f32::from_bits(b)).to_bits());
        check2("div", a, b, f32_div(a, b), (f32::from_bits(a) / f32::from_bits(b)).to_bits());
        check2("sqrt", a, 0, f32_sqrt(a), f32::from_bits(a).sqrt().to_bits());
        let x = rng.next();
        let y = rng.next();
        let (xh, xl) = ((x >> 32) as u32, x as u32);
        let (yh, yl) = ((y >> 32) as u32, y as u32);
        check64("add64", x, y, f64_add(xh, xl, yh, yl), f64::from_bits(x) + f64::from_bits(y));
        check64("mul64", x, y, f64_mul(xh, xl, yh, yl), f64::from_bits(x) * f64::from_bits(y));
    }
}
