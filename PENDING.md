# PENDING — arm64 (Graviton) optimization of abgen-lambda

Handoff doc. Goal: make the Lambda conversion path fast and cheap on arm64, and
settle **arm64 vs x86_64** for the Lambda fleet with measurements, not vibes.

## Why this exists

The texture encoders — the CPU-bound heart of a conversion — carry ~149
x86-gated SIMD sections (`crate/src/dxt1_pure.rs`, `crate/src/bc7_pure/`,
`crate/src/bc5_pure.rs`) and **zero `aarch64`/NEON paths**. On Graviton they run
scalar. Lambda arm64 is 20 % cheaper per GB-second, but the arithmetic inverts
fast: `arm_cost / x86_cost = 0.8 × (1 + encode_share × (slowdown − 1))`. At a
2× scalar penalty, arm64 is the *more expensive* arch once encoding exceeds
25 % of runtime — and encoding is why the function has 6 vCPUs at all.

Memory sizing is NOT the lever: Lambda cost = memory × seconds and vCPUs scale
with memory, so the parallel part of a job costs the same at every size; only
serial time is taxed by more memory. Keep 10 GB; fix the arch and the kernels.

## Current state (as of this commit)

- Branch `feat/lambda-converter`, commit `66544f2`: nix-built `packages.lambdaImage`
  (Dockerfiles deleted), single version source (`[workspace.package]` in root
  Cargo.toml), CI cache overhaul, dead-code drops.
- **Uncommitted WIP in the tree** (commit it before handing off): server
  feature-gate refactor — `crate` gains a `server` feature (axum/sqlx/tokio
  stack), `abgen-lambda` builds with `default-features = false`. Lock already
  validates (`cargo metadata --locked` passes offline against the crane vendor
  dir).
- E2E is proven: the nix image converts a scene end-to-end in docker against a
  mock catalyst (see appendix), exit 0, bundles + manifests on disk.
- Bench lane (criterion `kernels.rs`, `abgen-bench` bin) was deliberately
  deleted in `66544f2`. For this work, resurrect it on a throwaway branch:
  `git checkout 66544f2^ -- crate/benches crate/src/bin/abgen-bench.rs` and
  re-add the manifest entries + criterion dev-dep — or measure through
  `abgen-lambda --once` timings only.

## AWS resources ALREADY CREATED (account 175651002275)

Created via the API on 2026-08-19; reuse or delete:

- IAM role `abgen-bench-ssm` (trust: ec2.amazonaws.com) with
  `AmazonSSMManagedInstanceCore` attached.
- Instance profile `abgen-bench-ssm` (role attached).
- **No instance, no bucket, nothing billable is running.**

The account credentials on the dev box are a **root access key** — replace with
an IAM user or SSO before doing more of this.

## The plan

1. **Instance**: `c6g.16xlarge` (64 vCPU) in **eu-west-1** — c6g is Graviton2,
   the same silicon Lambda arm64 runs on; c7g/c8g measure prettier and lie.
   AL2023 arm64, 100 GB gp3, the `abgen-bench-ssm` instance profile (SSM) or a
   keypair + SSH. Stop it when idle (~$2.2/h on-demand).
2. **Repo**: push `feat/lambda-converter` (including the WIP commit) and clone
   on the instance. Toolchain: rustup `1.97.1` (matches `rust-toolchain.toml`)
   or Determinate nix + `nix build .#lambdaImage` (flake already does
   aarch64-linux).
3. **Baseline before touching code** (all on the instance):
   - Per-encoder scalar throughput (Mpix/s) for DXT1, BC5, BC7 on real texture
     data; same numbers on any x86 AVX2 box for the reference ratio.
   - `perf record` a full `--once` conversion of a texture-heavy scene (use the
     mock-catalyst appendix with bigger GLBs, or a real catalyst entity) — get
     the true encode share of runtime, which decides how much the kernels
     matter (see the cost formula above).
   - Free lever first: `RUSTFLAGS="-C target-cpu=neoverse-n1"` (Lambda
     Graviton2) — measure what autovectorization alone recovers.
4. **NEON ports**, in payoff order, in `std::arch::aarch64` intrinsics
   mirroring the existing SSE structure (`#[cfg(target_arch = "aarch64")]`
   beside the x86 gates, scalar stays the fallback):
   1. `dxt1_pure` — most bundles are DXT1; color-endpoint search + 4-color
      palette distance are straight lane-parallel ports.
   2. `bc5_pure` — normal maps; two independent single-channel BC4 problems,
      trivially NEON-able.
   3. `bc7_pure` — the mode tree is the hard one; port the inner
      endpoint/index scoring loops, keep mode selection scalar first.
   - **Hard constraint: bit-identical output vs scalar.** Builds are
     reproducible and parity-gated (PARITY.md, `ci/artifact-hashes/`); a NEON
     path that changes a single encoded byte breaks the reproducibility story.
     Add exhaustive block-level tests (NEON vs scalar over random + edge-case
     blocks) plus a corpus diff before/after.
   - Check the C side too: libjpeg-turbo already has NEON (fine); crunch
     (DXT5Crunched) is scalar-ish C++ — profile before investing there.
5. **Decide the arch**: same scenes, both arches, cost = billed-duration ×
   price. If arm64 with NEON lands ≥ ~20 % cheaper than x86 (it should, if the
   ports reach even half of AVX2 throughput), keep Graviton. If not, flip the
   Lambda to x86_64 — one-line runner change in `lambda-image.yml`
   (`ubuntu-24.04-arm` → `ubuntu-22.04`) since the flake builds both.
6. **Confirm on real Lambda**: push both arch images to ECR, deploy two
   functions, replay the same SQS events, compare billed duration × price.
   Independent of arch, two more paydays spotted during the cost review:
   - `lambda/src/convert.rs` calls `texencode_cache::clear()` after every job —
     warm invocations re-encode everything. Persist per-instance (drop the
     clear, add an LRU cap) or content-address into S3.
   - Route small entities (wearables/emotes) to a second ~2 GB function via
     SQS filter; scenes keep 10 GB.

## Access notes (dev-box quirks)

- The dev machine's per-app firewall (OpenSnitch) blocks outbound for `aws`,
  `cargo`, `docker` pulls, and git-over-https; `curl` and `nix` are allowed.
  AWS is reachable via `curl --aws-sigv4` if you must work from this box.
  SSH goes through the jump host: `ssh vpn -p 42042 -i ~/.ssh/yubi-777`, then
  ProxyJump to the instance.
- `cargo` offline tricks: point `CARGO_HOME` at a config whose source
  replacement is the crane vendor dir (`/nix/store/*-vendor-cargo-deps`).

## Cleanup checklist when done

- [ ] Terminate the instance; delete the keypair/SG if created.
- [ ] Delete instance profile + role `abgen-bench-ssm` (detach
      `AmazonSSMManagedInstanceCore` first) and any transfer bucket.
- [ ] Rotate/retire the root access key on the dev box.
- [ ] Delete this file once the arch decision is merged.

## Appendix — hermetic E2E (mock catalyst)

Runs the real container end-to-end offline; ~40 lines, no AWS. Layout:

```
mock/h-glb          # any real GLB (e.g. crate/abgen-wasm/test/fixtures/jpeg-quad.glb)
mock/h-scenejson    # {"main":"bin/game.js","scene":{"base":"0,0","parcels":["0,0"]}}
mock/h-gamejs       # "// noop"
mock/<entityId>     # entity doc: {"id":"<entityId>","type":"scene","content":
                    #   [{"file":"scene.json","hash":"h-scenejson"},
                    #    {"file":"bin/game.js","hash":"h-gamejs"},
                    #    {"file":"models/quad.glb","hash":"h-glb"}],
                    #  "metadata":{...scene.json...}}
```

Serve `GET /contents/<name>` from that dir on 127.0.0.1:8123 (return 404 for
`/contents/*/active-entities`, `[]` for `POST /entities/active`), then:

```
nix build .#lambdaImage && docker load < result
docker run --rm --network host -v $PWD/tmp:/tmp \
  -v $PWD/event.json:/event.json:ro abgen-lambda:0.16.2 --once /event.json
# event.json: {"entityId":"<entityId>","contentServerUrl":"http://127.0.0.1:8123"}
```

Expect exit 0, a summary JSON with `"exitCode": 0` per platform, and bundles +
`<platform>.manifest.json` under `tmp/abgen-out/<entityId>/`. Fatten `mock/`
with big textured GLBs to turn this into the benchmark driver.
