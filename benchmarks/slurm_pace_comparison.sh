#!/bin/bash
#SBATCH --job-name=steinerpy-followup
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:45:00
#SBATCH --output=followup-%j.log

# 48 * 15 + 48 * 120 worker-seconds / 3 = 36 minutes, plus startup overhead.
set -euo pipefail
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0 Rust/1.79.0-GCCcore-13.3.0
BENCH_ROOT="${BENCH_ROOT:-$HOME/steinerpy-benchmarks}"
PILOT_ROOT="${PILOT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the prepared pilot directory}}"
cd "$PILOT_ROOT/harness"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$BENCH_ROOT/.venv/bin/python" -m benchmarks.pace_comparison \
    --dataset "$BENCH_ROOT/pace2018" --solver-root "$BENCH_ROOT/competitors/wata" \
    --baseline "$PILOT_ROOT/baseline" --candidate "$PILOT_ROOT/candidate" \
    --output "$PILOT_ROOT/results-$SLURM_JOB_ID"
