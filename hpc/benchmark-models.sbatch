#!/bin/bash
# Benchmarks several models one after another on ONE GPU, from the GPU node itself (no SSH
# tunnel or laptop in the path), so they are compared on identical hardware. For each model:
# start vLLM, wait until healthy, warm up, run streamed load tests at each concurrency,
# stop vLLM, wait for the GPU memory to be released.
#
#   cp hpc/benchmark-models.sbatch scripts/load_gen.py -> /scratch/$USER/icp/bench/
#   cd /scratch/$USER/icp/bench && sbatch benchmark-models.sbatch
#   (override with e.g. MODELS="Qwen/Qwen2.5-7B-Instruct" CONCURRENCY="1 8" sbatch ...)
#
# Weights must already be in the scratch cache (hpc/download-model.sbatch); the job runs
# offline. Results: /scratch/$USER/icp/bench/<job id>/results.jsonl, a vLLM log per model,
# and gpu.csv (utilization and memory every 5 s).
#SBATCH --job-name=icp-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=bench-%j.out
set -uo pipefail

ROOT=/scratch/$USER/icp
SIF=$ROOT/vllm.sif
HERE=$ROOT/bench
OUT=$HERE/$SLURM_JOB_ID
mkdir -p "$OUT"

read -r -a MODELS <<<"${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct openai/gpt-oss-120b}"
read -r -a CONCURRENCY <<<"${CONCURRENCY:-1 16 64}"
DURATION=${DURATION:-60}
MAX_TOKENS=${MAX_TOKENS:-128}

export HF_HUB_CACHE=$ROOT/hf/hub HF_HUB_OFFLINE=1   # pre-downloaded; never wait on the network
export PYTHONNOUSERSITE=1                          # don't let ~/.local shadow the image's packages
export XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}

# vLLM's CUDA 13 image needs driver >= 580; on older drivers use its compatibility libraries.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)
export CUDA_COMPAT=0
if [ -n "$DRIVER" ] && [ "$DRIVER" -lt 580 ]; then CUDA_COMPAT=1; fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
NODE=$(hostname -s)
echo "node $NODE, $GPU, driver $DRIVER, cuda compat $CUDA_COMPAT"

in_image() {
  apptainer exec --nv --bind "$ROOT" "$SIF" bash -c \
    'if [ "$CUDA_COMPAT" = 1 ]; then export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi; exec "$@"' _ "$@"
}

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 >"$OUT/gpu.csv" 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
URL=http://127.0.0.1:$PORT

for model in "${MODELS[@]}"; do
  name=$(basename "$model" | tr '[:upper:].' '[:lower:]-')
  log=$OUT/$name-vllm.log
  echo "== $model ($(date +%H:%M:%S))"
  started=$(date +%s)
  in_image vllm serve "$model" --host 127.0.0.1 --port "$PORT" --served-model-name bench \
    --max-model-len 8192 >"$log" 2>&1 &
  server=$!

  ready=""
  for _ in $(seq 1 240); do  # up to 20 minutes
    curl -sf -m 5 "$URL/health" >/dev/null && { ready=1; break; }
    kill -0 "$server" 2>/dev/null || break
    sleep 5
  done
  if [ -z "$ready" ]; then
    echo "   did not start; last lines of $log:"; tail -15 "$log"
    kill "$server" 2>/dev/null; wait "$server" 2>/dev/null; continue
  fi
  startup=$(( $(date +%s) - started ))
  weights=$(grep -o "Model loading took [0-9.]* GiB" "$log" | grep -o "[0-9.]*" | head -1)
  kv=$(grep -o "Available KV cache memory: [0-9.]* GiB" "$log" | grep -o "[0-9.]*" | head -1)
  echo "   ready in ${startup}s (weights ${weights:-?} GiB, KV cache ${kv:-?} GiB)"

  # Warm-up, not recorded: first requests pay for CUDA graph capture and cache setup.
  in_image python3 "$HERE/load_gen.py" --url "$URL" --model bench --concurrency 4 --duration 15 \
    --max-tokens 32 --stream --json >/dev/null

  for c in "${CONCURRENCY[@]}"; do
    in_image python3 "$HERE/load_gen.py" --url "$URL" --model bench --concurrency "$c" \
      --duration "$DURATION" --max-tokens "$MAX_TOKENS" --stream --json |
      MODEL="$model" GPU="$GPU" NODE="$NODE" STARTUP="$startup" WEIGHTS="${weights:-}" KV="${kv:-}" python3 -c '
import json, os, sys
row = json.loads(sys.stdin.read())
num = lambda k: float(os.environ[k]) if os.environ.get(k) else None
row.update(model=os.environ["MODEL"], gpu=os.environ["GPU"], node=os.environ["NODE"],
           startup_s=num("STARTUP"), weights_gib=num("WEIGHTS"), kv_cache_gib=num("KV"))
print(json.dumps(row))' | tee -a "$OUT/results.jsonl"
  done

  kill "$server" 2>/dev/null; wait "$server" 2>/dev/null
  for _ in $(seq 1 30); do  # let the GPU memory drain before the next model
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 2000 ] && break
    sleep 2
  done
done
echo "done ($(date +%H:%M:%S)): $OUT/results.jsonl"
