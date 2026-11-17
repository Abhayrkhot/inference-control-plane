#!/bin/bash
# Converts the vLLM Docker image into an Apptainer image in scratch. Run once per vLLM version.
#   sbatch hpc/pull-vllm-sif.sbatch v0.x.y      (pin a real tag; check `sinfo` for a CPU partition)
#SBATCH --job-name=icp-pull-vllm
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=icp-pull-vllm-%j.out
set -euo pipefail

TAG=${1:?usage: sbatch pull-vllm-sif.sbatch <vllm image tag>}
ROOT=/scratch/$USER/icp
export APPTAINER_CACHEDIR=$ROOT/.apptainer-cache
export APPTAINER_TMPDIR=$ROOT/.apptainer-tmp
mkdir -p "$ROOT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

module load apptainer 2>/dev/null || true
apptainer pull --force "$ROOT/vllm.sif" "docker://vllm/vllm-openai:${TAG}"
echo "wrote $ROOT/vllm.sif"
