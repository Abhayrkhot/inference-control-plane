#!/bin/bash
# Downloads Hugging Face models into the scratch cache the Slurm backend reads from, so a
# rollout of a large model spends its startup time loading weights, not downloading them.
#   cd /scratch/$USER/icp && sbatch download-model.sbatch openai/gpt-oss-120b [more models...]
# Runs on a CPU node; compute nodes reach Hugging Face through the cluster's HTTP proxy.
#SBATCH --job-name=icp-download
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=icp-download-%j.out
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: sbatch download-model.sbatch <model> [model...]"; exit 2; }
ROOT=/scratch/$USER/icp
export HF_HUB_CACHE=$ROOT/hf/hub
export PYTHONNOUSERSITE=1  # use the image's huggingface_hub, not one in ~/.local
mkdir -p "$HF_HUB_CACHE"

for model in "$@"; do
  echo "== $model (started $(date +%H:%M:%S))"
  apptainer exec --bind "$ROOT" "$ROOT/vllm.sif" python3 - "$model" <<'EOF'
import sys
import time

from huggingface_hub import snapshot_download

model = sys.argv[1]
start = time.time()
# Skip the duplicate checkpoint formats some repos also ship (gpt-oss: original/ and metal/).
path = snapshot_download(model, ignore_patterns=["original/*", "metal/*", "*.pth", "*.pt", "*.gguf"], max_workers=8)
print(f"downloaded {model} in {time.time() - start:.0f}s to {path}", flush=True)
EOF
done
echo "== cache size: $(du -sh --apparent-size "$HF_HUB_CACHE" | cut -f1)"
