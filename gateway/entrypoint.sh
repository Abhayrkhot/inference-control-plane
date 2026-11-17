#!/bin/sh
# Forwards :8000 in this pod to TARGET_HOST:TARGET_PORT (the vLLM job) through LOGIN_HOST.
# If the tunnel drops, ssh exits and Kubernetes restarts the pod. The pod's readiness probe
# calls vLLM's /health through the tunnel.
set -eu
: "${TARGET_HOST:?}" "${TARGET_PORT:?}" "${LOGIN_HOST:?}" "${SSH_USER:?}"
KEY_DIR=${KEY_DIR:-/etc/icp-ssh}

echo "gateway: :8000 -> ${SSH_USER}@${LOGIN_HOST} -> ${TARGET_HOST}:${TARGET_PORT}"
exec ssh -N \
  -i "$KEY_DIR/id_ed25519" \
  -o IdentitiesOnly=yes \
  -o UserKnownHostsFile="$KEY_DIR/known_hosts" \
  -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "0.0.0.0:8000:${TARGET_HOST}:${TARGET_PORT}" \
  "${SSH_USER}@${LOGIN_HOST}"
