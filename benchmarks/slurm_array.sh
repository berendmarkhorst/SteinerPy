#!/bin/bash
#SBATCH --job-name=steinerpy-bench
#SBATCH --array=0-99%16          # 100 instances, 16 concurrent (adjust to your set)
#SBATCH --time=01:00:00          # hard wall-clock per instance (per task)
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --output=benchmarks/results/slurm-%A_%a.out

# One SteinLib instance per array task -> hard per-instance wall-clock isolation
# and trivial parallelism. Build an instance list first, e.g.:
#   ls benchmarks/data/B/*.stp > benchmarks/instances.txt
#
# Each task solves its instance and appends to a shared, resumable CSV.

set -euo pipefail

INSTANCE_LIST="${INSTANCE_LIST:-benchmarks/instances.txt}"
TIME_LIMIT="${TIME_LIMIT:-3300}"   # leave headroom under the SBATCH --time
OUT="${OUT:-benchmarks/results/hpc.csv}"

INSTANCE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$INSTANCE_LIST")
if [[ -z "$INSTANCE" ]]; then
    echo "No instance for task ${SLURM_ARRAY_TASK_ID}"; exit 0
fi

DIR=$(dirname "$INSTANCE")
BASENAME=$(basename "$INSTANCE")

# Run a single-instance directory view by symlinking into a temp dir, or simply
# point --instances at the parent and rely on the resumable skip for the rest.
python -m benchmarks.run_benchmarks \
    --instances "$DIR" \
    --config both --full \
    --time-limit "$TIME_LIMIT" \
    --out "$OUT" \
    --jobs 1
