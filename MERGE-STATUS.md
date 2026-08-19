# abgen arm64 optimization — merge dashboard

Local branch: `feat/arm64-neon` (from 1ff8df0 via benchbox/integration dd3c480).
Updated: 2026-08-19 (merge agent live).

## Gen-1 results

| id | title | speedup | parity | state |
|----|-------|---------|--------|-------|
| o01 | DXT1 NEON encode_block | +2.1% over dxt1m (88.1% vs scalar) | bit-identical | MERGED 6451635 (encode_block module + tests grafted onto dxt1m; overlapping sRGB/box-halve parts dropped) |
| o02 | DXT1 scalar restructure + sRGB LUTs | 86.3% | bit-identical | superseded by dxt1m (its SHA256 gate + branchless round kept inside dxt1m) |
| o03 | DXT1 sRGB LUTs + NEON box-halve/round | 87.8% | bit-identical | superseded by dxt1m (taken wholesale into it) |
| o04 | BC5/BC4 NEON channel encode + box-halve | 83.0% (bc5 5.9x) | bit-identical | MERGED 16c8040 |
| o05 | BC7 estimator NEON (est_wasm128 port) | 19.8% (bc7/slow) | bit-identical | superseded by o08 (same port + partition drivers, −24.5%) |
| o06 | BC7 evaluate.rs NEON | 5.07% (bc7/basic) | bit-identical | MERGED via integration dd3c480 |
| o07 | BC7 color.rs NEON (dist + LSQ) | 0.67% basic, +0.23% slow | bit-identical | MERGED via integration dd3c480 |
| o08 | BC7 partition/estimate NEON (est_neon.rs) | 24.5% (bc7/slow), 14.2% basic | bit-identical | MERGED via integration dd3c480 |
| o09 | resize.rs NEON f64x2 + row-wise vertical + LUTs | 45.5% (resize/box) | bit-identical | MERGED via integration dd3c480 |
| o10 | target-cpu flags experiment | no-win | bit-identical | SKIPPED (config; see flags below) |
| o11 | LTO experiment | no-win | bit-identical | SKIPPED (config; keep default release profile) |
| o12 | PGO pipeline | no-win (bc7 +4.3%) | bit-identical | SKIPPED (net negative for dominant kernel + CI cost) |
| o13 | third-party C flags (libjpeg -O2→-O3 aarch64-linux) | 3.0% (jpeg decode) | bit-identical | MERGED via integration dd3c480 |
| o14 | BC7 scalar unit-weight specialization of evaluate.rs loops | 3.2% (pre-o06 baseline) | bit-identical | SKIPPED — superseded by o06's NEON port of the same selector loops; win does not stack on aarch64 (NEON path covers the unit-weight case, perceptual stays weighted either way); patch conflicts inside o06's gating regions |
| o15 | BC7 alpha selector + mip box_halve NEON | 7.9% alpha, 34.9% mip chain | bit-identical | MERGED via integration dd3c480 |
| o16 | e2e profile (encode_share 0.90) + NEON LZ4HC hash batch | 2.5% (lz4) | bit-identical | MERGED via integration dd3c480 |
| dxt1m | DXT1 unification: o03 wholesale + o02 gates; o02 encode_block restructure rejected (+5.4% regression) | 8.2x vs scalar | bit-identical | MERGED 2bdbceb |

## Gen-2 results (g01–g08, ~/results/gen2/, diffed vs integration)

None arrived yet.

## Gen-3 results (h01–h08, ~/results/gen3/, diffed vs integration)

None arrived yet (launched ~30-60 min behind gen-2).

## Recommended build flags (from o10/o11/o12)

- o10: do NOT enable `-C target-cpu=neoverse-n1` (bc7 +2.3%, dxt1 +4.4% regressions) or
  `neoverse-512tvb` (SIGILL on Graviton2 — emits SVE). `+lse` is safe but measured noise-level.
- o11: keep the default release profile — fat LTO+cu1 and thin LTO are noise on kernels,
  thin LTO regresses resize/box by 8%, and both double clean-build time.
- o12: PGO not recommended as-is — dxt1/resize gain 3-4% but bc7 (dominant kernel) regresses
  4.3%, plus two uncached builds + profile staleness against the PARITY.md gate.
- o13 (merged): third-party C/C++ on aarch64-linux gets `-mcpu=neoverse-n1` (crunch/draco,
  measured neutral) and libjpeg9c lifted -O2→-O3 (the actual 3% jpeg-decode win).

## Cumulative kernel state on the merged branch (box numbers)

- dxt1: 703 ms → 83.4 ms on 1024² (8.4x) [dxt1m + o01 NEON encode_block]
- bc5: 82.0 ms → 14.0 ms on 1024² (5.9x) [o04]
- bc7: basic 203 → ~156 ms est., slow 212 → ~146 ms est. (o08 −14.2%/−24.5%, o15 −5.6%/−0.9%,
  o06 −5.1%/−3.9%, o07 −0.7%/+0.2%, stacking measured only pairwise — box verify will re-bench)
- bc7 mip chain: 26.4 → 17.2 ms (−34.9%) [o15]
- resize: box 97.5 → 67.0 ms (−31%), premul 88.1 → 77.6 ms (−12%) [o09]
- lz4 compress_hc: −2.5% [o16]; jpeg decode: −3.0% [o13]

## Log

- 2bdbceb dxt1m merged (Cargo.toml [[test]] conflict: kept all entries).
- 6451635 o01 NEON encode_block grafted onto dxt1m per orchestrator instruction; local
  `cargo check --no-default-features -p abgen` passes (needs nix cmake on PATH for draco).
- 16c8040 o04 merged (kernels.rs criterion_group conflict: kept all bench groups).
- o14 applied experimentally, 3 conflict regions inside o06 NEON gates in evaluate.rs → reverted, skipped.
- Pushed merged branch to box as `merged-shade`; verification build/test pinned to cores 36-39 pending.
