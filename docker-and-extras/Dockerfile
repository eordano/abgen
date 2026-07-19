# syntax=docker/dockerfile:1

# Thin vast.ai base runner for the abgen "pull-not-build" GPU pipeline.
#
# It carries ONLY the tooling needed to boot on a vast GPU box, accept an ssh
# session from vast's proxy, then pull+run (or nix-build) the abgen GPU image,
# warm it, and rsync the slot cache back. It deliberately does NOT contain
# abgen itself — that is pulled or built at runtime, so this image stays small
# and reusable across every abgen sha.

# CUDA *runtime* base so vast's GPU injection + libcuda "just work". Override
# CUDA_IMAGE to match the box's driver if needed (e.g. 12.2.2 for driver 535).
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# --- OS tooling --------------------------------------------------------------
# bash/coreutils/openssh/rsync/git   = the runner contract (ssh in, sync back).
# cmake/pkg-config/gcc/g++/make      = the bare (non-nix) build fallback.
# libvulkan1                         = the Vulkan loader; its silent absence is
#                                      a known GPU trap for the wgpu backend.
# vulkan-tools                       = vulkaninfo, to diagnose ICD injection.
# ca-certificates/curl/xz/gnupg/sudo = installer + registry plumbing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash coreutils \
        openssh-server openssh-client \
        rsync git \
        cmake pkg-config gcc g++ make \
        libvulkan1 vulkan-tools \
        ca-certificates curl xz-utils gnupg sudo \
    && rm -rf /var/lib/apt/lists/*

# --- docker CLI (client only, no daemon) ------------------------------------
# For `docker pull` + `docker run` of the abgen GPU image against the HOST
# daemon. Privileged docker-in-docker is usually unavailable on vast, so the
# nix-build path is the primary route and this is the documented fallback.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin \
    && rm -rf /var/lib/apt/lists/*

# --- nix (flakes) ------------------------------------------------------------
# Determinate installer, container mode (--init none). Enables the flake
# features abgen's flake.nix needs; substituters stay on the public
# cache.nixos.org + FlakeHub defaults. The nix daemon is started at runtime by
# the entrypoint (there is no init system in the container).
RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix \
      | sh -s -- install linux \
          --init none \
          --no-confirm \
          --extra-conf "experimental-features = nix-command flakes" \
          --extra-conf "max-jobs = auto" \
          --extra-conf "cores = 0" \
          --extra-conf "trusted-users = root" \
    && grep -q 'experimental-features' /etc/nix/nix.conf

# --- baked environment -------------------------------------------------------
# This ENV block only reaches processes that inherit the OCI config.Env
# (e.g. `docker run <img> <cmd>`). sshd sessions do NOT inherit it, so the same
# values are also written to /etc/environment (PAM, non-login `ssh host cmd`)
# and /etc/profile.d (login shells) below.
#
# VK_ICD_FILENAMES : the NVIDIA runtime driver ICD the container toolkit mounts
#                    when NVIDIA_DRIVER_CAPABILITIES includes `graphics`.
# CMAKE_POLICY_... : cmake4 rejects draco's old CMakeLists; force the 3.5 floor
#                    for any runtime `nix build`/`cargo` of abgen.
# LIBRARY_PATH     : cuda stubs (link against libcuda at build time; the real
#                    driver is injected at run time) + the multiarch dir where
#                    glibc's iconv lives (belt-and-suspenders for the documented
#                    libiconv-on-LIBRARY_PATH build need). This is the *link-time*
#                    search path (ld/collect2), not the runtime loader path.
# LD_LIBRARY_PATH  : baked EMPTY on purpose, and set to empty here to OVERRIDE the
#                    CUDA base image's own value. Any system/multiarch dir on the
#                    *runtime* loader path shadows nix's own OpenSSL (libs2n needs
#                    OPENSSL_3.4.0) and glibc (GLIBC_PRIVATE), so `nix` itself
#                    crashes (libs2n.so.1: OPENSSL_3.4.0 not found /
#                    __nptl_change_stack_perm GLIBC_PRIVATE). The system libc is
#                    already found via /etc/ld.so.cache, so nothing is lost by
#                    clearing it. The warm flow (colmena) sets LD_LIBRARY_PATH per
#                    command instead: cleared for `nix`, and a culibs/ dir
#                    (symlinks to the injected libcuda.so.1 + libnvidia-*) plus
#                    /usr/local/cuda/lib64 for the abgen GPU server — never the
#                    system libc dir.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu \
    LD_LIBRARY_PATH= \
    PATH=/nix/var/nix/profiles/default/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin

RUN { \
      echo 'PATH="/nix/var/nix/profiles/default/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin"'; \
      echo 'VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"'; \
      echo 'CMAKE_POLICY_VERSION_MINIMUM="3.5"'; \
      echo 'LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu"'; \
      echo 'LD_LIBRARY_PATH='; \
    } > /etc/environment

RUN mkdir -p /etc/profile.d && { \
      echo '# baked by docker-and-extras runner image'; \
      echo 'export PATH="/nix/var/nix/profiles/default/bin:${PATH}:/usr/local/cuda/bin"'; \
      echo 'export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"'; \
      echo 'export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"'; \
      echo 'export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"'; \
      echo '# LD_LIBRARY_PATH stays EMPTY (see the Dockerfile ENV note): leaking the'; \
      echo '# system/CUDA libc dirs onto the runtime loader path shadows nix own'; \
      echo '# OpenSSL/glibc and crashes nix. The warm flow sets it per command.'; \
      echo 'export LD_LIBRARY_PATH='; \
      echo 'if [ -e /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh; fi'; \
    } > /etc/profile.d/00-runner.sh

# --- sshd + entrypoint -------------------------------------------------------
RUN mkdir -p /run/sshd /etc/ssh/sshd_config.d \
    && { \
        echo 'PermitRootLogin prohibit-password'; \
        echo 'PasswordAuthentication no'; \
        echo 'PubkeyAuthentication yes'; \
        echo 'UsePAM yes'; \
    } > /etc/ssh/sshd_config.d/99-runner.conf

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

LABEL org.opencontainers.image.title="docker-and-extras" \
      org.opencontainers.image.description="Thin vast.ai base runner (ssh+nix+docker+rsync+cuda/vulkan) for the abgen pull-not-build GPU pipeline. Does not contain abgen." \
      org.opencontainers.image.source="https://github.com/eordano/abgen"

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
