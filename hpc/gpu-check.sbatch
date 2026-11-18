#!/bin/bash
# Checks that vllm.sif can use a GPU on this cluster, then serves a small model and sends it
# one completion. Run it once after building the image:
#   cd /scratch/$USER/icp && sbatch gpu-check.sbatch [model]
#
# vLLM images are built for a newer CUDA than many cluster drivers. Datacenter GPUs can still
# run them through CUDA's forward-compatibility libraries (/usr/local/cuda/compat in the
# image); this script tries the image as-is first, then with those libraries, and reports
# which one worked.
#SBATCH --job-name=icp-gpu-check
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100@80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=icp-gpu-check-%j.out
set -uo pipefail

MODEL=${1:-Qwen/Qwen2.5-1.5B-Instruct}
ROOT=/scratch/$USER/icp
SIF=$ROOT/vllm.sif
export HF_HUB_CACHE=$ROOT/hf/hub
# Apptainer mounts your home directory, so Python would load packages from ~/.local (for
# example a PyTorch installed for coursework) ahead of the image's own. Ignore them.
export PYTHONNOUSERSITE=1
# Keep vLLM and Triton caches on scratch rather than in the small home quota.
export XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton
mkdir -p "$HF_HUB_CACHE" "$TRITON_CACHE_DIR"
# Compute nodes reach the internet through an HTTP proxy (http_proxy is set); requests to
# this node's own server must not go through it.
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}

in_image() {  # in_image <plain|compat> command...
  local mode=$1; shift
  apptainer exec --nv --bind "$ROOT" "$SIF" bash -c '
    if [ "$0" = compat ]; then export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
    exec "$@"' "$mode" "$@"
}

echo "== host: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "== image CUDA: $(in_image plain bash -c 'echo $CUDA_VERSION'); compat libs: $(in_image plain bash -c 'ls /usr/local/cuda/compat 2>/dev/null | tr "\n" " "')"

CHECK='import torch; print("torch", torch.__version__, "cuda", torch.version.cuda); x = torch.ones(1).cuda(); print("gpu ok:", torch.cuda.get_device_name(0))'
MODE=""
for mode in plain compat; do
  echo "== torch on GPU, mode=$mode"
  if in_image "$mode" python3 -c "$CHECK"; then MODE=$mode; break; fi
done
if [ -z "$MODE" ]; then echo "RESULT: the image cannot use the GPU in either mode"; exit 1; fi
echo "RESULT: GPU works with mode=$MODE"

PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
LOG=$ROOT/icp-gpu-check-${SLURM_JOB_ID}-vllm.log
echo "== serving $MODEL on port $PORT (log: $LOG)"
START=$(date +%s)
in_image "$MODE" vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name check >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for _ in $(seq 1 180); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "RESULT: vllm exited during startup"; tail -40 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "RESULT: not healthy after 15 min"; tail -40 "$LOG"; exit 1; }
echo "ready after $(( $(date +%s) - START ))s (includes the model download on the first run)"

echo "== completion"
curl -s "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model": "check", "prompt": "The capital of France is", "max_tokens": 8}'
echo
echo "== metrics"
curl -s "localhost:$PORT/metrics" | grep -E '^vllm:(num_requests_(running|waiting)|generation_tokens_total|e2e_request_latency_seconds_count)' | head -6
echo "RESULT: served $MODEL with mode=$MODE"
