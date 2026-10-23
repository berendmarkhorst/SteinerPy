#!/bin/bash
#SBATCH --job-name=steinerpy-connectivity
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=00:40:00
#SBATCH --output=connectivity-%j.log

# Rome's minimum allocation is 16 cores / 28 GiB. Three 8 GiB workers fit.
# 48 * 120 seconds / 3 workers = 32 minutes, plus supervisor/startup time.
set -euo pipefail
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
BENCH_ROOT="${BENCH_ROOT:-$HOME/steinerpy-benchmarks}"
# Some sites strip custom exported variables from the submitted environment.
PILOT_ROOT="${PILOT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the prepared pilot directory}}"
cd "$PILOT_ROOT/harness"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$BENCH_ROOT/.venv/bin/python" -m benchmarks.connectivity_pilot \
    --dataset "$BENCH_ROOT/pace2018" \
    --baseline "$PILOT_ROOT/baseline" --candidate "$PILOT_ROOT/candidate" \
    --output "$PILOT_ROOT/results-$SLURM_JOB_ID" --jobs 3
