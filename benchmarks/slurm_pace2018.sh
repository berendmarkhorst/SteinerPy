#!/bin/bash
#SBATCH --job-name=steinerpy-pace2018
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --output=pace2018-%A_%a.log

# Two 12-worker Gurobi shards, two HiGHS shards, one heuristic shard.
# Memory, rather than CPU, determines the allocation: 12 workers * 8 GiB.
# Shared WLS reservations enforce 24 sessions INCLUDING expired-process tokens.
set -euo pipefail
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
BENCH_ROOT="${BENCH_ROOT:-$HOME/steinerpy-benchmarks}"
OUTPUT="${OUTPUT:-$BENCH_ROOT/results/pace2018}"
cd "$BENCH_ROOT/source"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=(gurobi gurobi highs highs heuristic)
INDEX="${SLURM_ARRAY_TASK_ID:?Submit as a job array}"
CONFIG="${CONFIG:-${CONFIGS[$INDEX]}}"
SHARDS=2
SHARD=$((INDEX % 2))
if [[ "$INDEX" -eq 4 ]]; then SHARDS=1; SHARD=0; fi
EXTRA=()
if [[ -n "${FIRST:-}" ]]; then EXTRA+=(--first "$FIRST"); fi

exec "$BENCH_ROOT/.venv/bin/python" -m benchmarks.pace2018 \
    --dataset "$BENCH_ROOT/pace2018" --output "$OUTPUT" \
    --config "$CONFIG" --tracks "${TRACKS:-1,2,3}" \
    --jobs "${JOBS:-12}" --shard-count "$SHARDS" --shard-index "$SHARD" \
    --time-limit "${TIME_LIMIT:-1800}" --memory-gb 8 "${EXTRA[@]}"
