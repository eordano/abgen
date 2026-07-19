#!/usr/bin/env bash
# Entrypoint for the docker-and-extras vast runner.
#   1. seed sshd host keys + accept an injected authorized_keys
#   2. start the nix daemon (no init system in the container)
#   3. exec the operator's command, or sshd -D so vast's proxy can connect
set -euo pipefail

# --- sshd host keys ----------------------------------------------------------
mkdir -p /run/sshd
ssh-keygen -A >/dev/null 2>&1 || true

# --- injected authorized_keys ------------------------------------------------
# vast passes the operator pubkey via env (PUBLIC_KEY or SSH_PUBLIC_KEY); a
# mounted file at /run/secrets/authorized_keys is also honored.
mkdir -p /root/.ssh
chmod 700 /root/.ssh
for k in "${PUBLIC_KEY:-}" "${SSH_PUBLIC_KEY:-}"; do
  [ -n "$k" ] && printf '%s\n' "$k" >> /root/.ssh/authorized_keys
done
if [ -f /run/secrets/authorized_keys ]; then
  cat /run/secrets/authorized_keys >> /root/.ssh/authorized_keys
fi
if [ -f /root/.ssh/authorized_keys ]; then
  sort -u -o /root/.ssh/authorized_keys /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# --- nix daemon --------------------------------------------------------------
# Needed for a runtime `nix build`/`nix copy` of abgen as root. Skip if already
# running.
daemon=/nix/var/nix/profiles/default/bin/nix-daemon
if [ -x "$daemon" ] && [ ! -S /nix/var/nix/daemon-socket/socket ]; then
  "$daemon" >/var/log/nix-daemon.log 2>&1 &
fi

# --- hand off ----------------------------------------------------------------
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
exec /usr/sbin/sshd -D -e
