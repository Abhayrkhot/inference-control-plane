#!/bin/bash
# Converts the vLLM Docker image into an Apptainer image in scratch. Run once per vLLM version.
#   cd /scratch/$USER/icp && sbatch pull-vllm-sif.sbatch v0.x.y    (pin a real tag)
#
# The image is unpacked on the node's local disk (/tmp). Unpacking it on /scratch, a network
# filesystem, takes hours because the image is hundreds of thousands of small files. Downloaded
# layers are cached in scratch, so a rerun does not download them again.
#SBATCH --job-name=icp-pull-vllm
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=icp-pull-vllm-%j.out
set -euo pipefail

TAG=${1:?usage: sbatch pull-vllm-sif.sbatch <vllm image tag>}
ROOT=/scratch/$USER/icp
export APPTAINER_CACHEDIR=$ROOT/.apptainer-cache
export APPTAINER_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/icp-apptainer-XXXXXX")
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
mkdir -p "$ROOT" "$APPTAINER_CACHEDIR"
echo "unpacking in $APPTAINER_TMPDIR on $(hostname) ($(df -h "$APPTAINER_TMPDIR" | awk 'NR==2 {print $4}') free)"

module load apptainer 2>/dev/null || true
apptainer pull --force "$APPTAINER_TMPDIR/vllm.sif" "docker://vllm/vllm-openai:${TAG}"
cp "$APPTAINER_TMPDIR/vllm.sif" "$ROOT/vllm.sif.partial"
mv "$ROOT/vllm.sif.partial" "$ROOT/vllm.sif"
echo "wrote $ROOT/vllm.sif ($(du -h "$ROOT/vllm.sif" | cut -f1))"

# Remove temp files left on scratch by older versions of this script, which unpacked there.
rm -rf "$ROOT/.apptainer-tmp"
