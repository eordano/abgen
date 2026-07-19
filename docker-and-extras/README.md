# abgen-on-docker — thin vast.ai runner

A small, abgen-agnostic base image for the **pull-not-build** GPU pipeline. It
carries only the tooling to boot on a vast GPU box, take an ssh session, then
pull or `nix build` the abgen GPU image, warm it, and rsync the slot cache
back. It does **not** contain abgen — that is fetched at runtime, so the image
is reusable across every abgen sha.

Published to `ghcr.io/eordano/abgen-on-docker` by the docker-and-extras workflow.
exactly what you enter in vast's image field). vast.ai has no registry of its
own; instances pull the image from Docker Hub / GHCR at creation time, so the
image must live on one of those. A `docker save` tarball cannot be loaded into
a vast instance (the instance *is* the container).

## What's inside

- ssh (sshd), rsync, git, bash, coreutils — the runner contract.
- cmake, pkg-config, gcc/g++, make — the bare (non-nix) build fallback.
- nix with flakes enabled (Determinate installer, public caches only).
- docker CLI (client, no daemon) — for the `docker pull`/`docker run` path.
- CUDA runtime base + Vulkan loader (`libvulkan1`), with `VK_ICD_FILENAMES`
  pointed at the NVIDIA runtime-driver ICD so the wgpu backend finds a device.

Env baked so **sshd sessions inherit it** (OCI `config.Env` is not inherited by
sshd) — written to `/etc/environment` (PAM, `ssh host cmd`) and
`/etc/profile.d/00-runner.sh` (login shells): `VK_ICD_FILENAMES`,
`CMAKE_POLICY_VERSION_MINIMUM=3.5` (cmake4 rejects draco's old CMakeLists),
`LIBRARY_PATH` (cuda stubs + glibc iconv dir, a *link-time* path), and nix on
`PATH`.

**`LD_LIBRARY_PATH` is baked EMPTY on purpose** (ENV, `/etc/environment`,
`/etc/profile.d`), and set to empty to override the CUDA base image's own value.
Leaking a system/multiarch or CUDA lib dir onto the *runtime* loader path shadows
nix's own OpenSSL (libs2n needs `OPENSSL_3.4.0`) and glibc (`GLIBC_PRIVATE`) and
crashes `nix` itself. The system libc is already resolved via `/etc/ld.so.cache`,
so nothing is lost. The warm flow (colmena `abgen-image.nix`) sets
`LD_LIBRARY_PATH` **per command** instead: cleared for `nix`, and for the abgen
GPU server a small `culibs/` dir of symlinks to the injected `libcuda.so.1` +
`libnvidia-*` (plus `/usr/local/cuda/lib64`) — never the system libc dir.

## Build + publish

```bash
docker build -t ghcr.io/eordano/abgen-on-docker:latest docker-and-extras/
docker login                                   # Docker Hub
docker push ghcr.io/eordano/abgen-on-docker:latest
```

The GitHub Actions workflow `.github/workflows/docker-and-extras.yml` does the
same on push to the `docker-and-extras` branch (needs repo secrets
`DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`).

## Booting on vast

Create the instance with image `ghcr.io/eordano/abgen-on-docker`, request a GPU,
and inject your ssh pubkey via the `PUBLIC_KEY` env (the entrypoint appends it
to `/root/.ssh/authorized_keys`, generates host keys, starts the nix daemon,
then runs `sshd -D`). Ensure the GPU is exposed with the `graphics` capability
so the Vulkan ICD is mounted.

## Two runtime paths for abgen

**Path A — nix-direct (recommended).** vast instances are unprivileged
containers and usually **cannot** run docker-in-docker, so build/run abgen from
its flake with nix (no daemon socket needed):

```bash
nix build 'github:eordano/abgen#<gpu-package>'   # substituted from public cache
./result/bin/abgen ...                           # run + warm on the GPU
```

**Path B — docker pull (fallback).** Only if the box actually grants a usable
docker daemon (privileged / mounted socket):

```bash
docker pull  <abgen-gpu-image>
docker run --gpus all --rm <abgen-gpu-image> ...
```

Because privileged DinD is the exception on vast, **Path A is the primary route
for the pull-not-build flow**; Path B is a fallback for boxes that expose the
host daemon.

## Warm + sync back

After warming the hot set against the local abgen server, push the slot cache
to your collector over ssh, e.g.:

```bash
rsync -a --info=progress2 /data/cache/ <user>@<collector-host>:<dest>/
```
