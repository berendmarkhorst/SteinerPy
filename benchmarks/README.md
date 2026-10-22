# SteinerPy benchmarks

Compares SteinerPy's solvers against **published SteinLib optima**, and measures
the effect of the opt-in **dual-ascent accelerator** (`dual_ascent=True`).

This directory is **not** part of the installable package (it ships in neither
the wheel nor the sdist). Benchmark infrastructure tests live in `tests/`;
the expensive benchmark sweeps are not collected by `pytest`.

## PACE 2018

The [public report](reports/pace2018/README.md) covers all three applicable
undirected Steiner-tree tracks, with licensed Gurobi, open-source HiGHS, and
SteinerPy's heuristic mode shown separately. The accompanying CSV includes
every requested instance, including pending runs, failures and timeouts.

For the Snellius sweep, prepare a separate checkout and environment:

```bash
mkdir -p "$HOME/steinerpy-benchmarks"
cd "$HOME/steinerpy-benchmarks"
git clone https://github.com/berendmarkhorst/SteinerPy.git source
git -C source checkout 7d4d8b8e058c08a8f4215ca0fb446ce4d35f059c
git clone https://github.com/PACE-challenge/SteinerTree-PACE-2018-instances.git pace2018
git -C pace2018 checkout 4df73cea9c311faea7d03e6d6bffa8733c34a1aa
module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
python -m venv .venv
.venv/bin/pip install -e ./source \
  networkx==3.7 numpy==2.5.4 scipy==1.18.1 highspy==1.15.1 gurobipy==13.0.3
sbatch source/benchmarks/slurm_pace2018.sh
```

The five array tasks run two Gurobi shards, two HiGHS shards, and one heuristic
shard, with 12 isolated workers per task. Each instance has a hard 1,800-second
wall deadline, an 8 GiB address-space limit and one solver/BLAS thread. Budget
for five allocations of 64 CPUs and 100 GiB, each capped at 24 hours. Use the
usual Slurm account/partition overrides for a different allocation.

Provide your own Gurobi license through `GRB_LICENSE_FILE` (default
`$HOME/gurobi.lic`). No credentials belong in this repository. All Gurobi
benchmark processes share **24 slots** under `$HOME/.cache/steinerpy-wls`,
including a 310-second reservation after closing a licensed environment.
This leaves six sessions under a 30-session WLS license for other use;
other applications must fit within that headroom. Keep the same slot directory
across arrays. The guard requires POSIX file locking and runs on Linux.

Use explicit exports for overrides, for example:

```bash
sbatch --export=ALL,FIRST=1,TIME_LIMIT=60,OUTPUT="$HOME/steinerpy-benchmarks/results/smoke" \
  source/benchmarks/slurm_pace2018.sh
```

Do not change the source checkout or its environment while a sweep is running.
The measured revision above had `STEINERPY_DECOMPOSE` and
`STEINERPY_DW_MAX_TERMINALS` unset; keep these unset when reproducing that
revision. Later runner revisions explicitly pin decomposition off and the
few-terminal DP threshold to ten. Use a new output directory when changing
code, dependencies or limits. Restarting the same command resumes completed
cases only when their recorded provenance matches.

Each case has `request.json`, phase checkpoints, `worker.log` and `result.json`.
Results contain original-edge certificates, costs, proof status, checksums,
versions, peak RSS and phase timings. The checker rejects disconnected/cyclic
outputs, incorrect original costs and false optimum claims. A wall timeout
without a returned, checked solution is scored as missing even if the solver
internally found an incumbent.

Generate the report from a current checkout containing `pace_report.py`:

```bash
python -m benchmarks.pace_report \
  --results /path/to/results/pace2018 \
  --comparison-results /path/to/results/composition-ablation \
  --output benchmarks/reports/pace2018
```

All five manifests are required. Pending cases are explicit, and aggregate
quality scores are withheld until the corresponding cohort has finished.
Published PACE participant results are historical context; they were not
rerun on this machine and do not establish a timing speedup or ranking.

### Same-hardware comparison sample

The report also contains a separate, completed sample on instances 001–003 of
each track. It compares specialized HiGHS, the generic composition model and
the original wata&sigma submission with 90-second deadlines. It is too small
to establish a general competition ranking. Reproduce it on a compute node,
using the same dataset, Python environment and CPU model:

```bash
for config in highs composition-highs; do
  python -m benchmarks.pace2018 --dataset /path/to/pace2018 \
    --output /path/to/results/composition-ablation --config "$config" \
    --first 3 --jobs 3 --time-limit 90 --memory-gb 8
done

git clone https://github.com/wata-orz/steiner_tree.git /path/to/wata
git -C /path/to/wata checkout 2700bdc3033d0d01c5d3648832b4535fceb70991
cp benchmarks/reports/pace2018/wata-Cargo.lock /path/to/wata/Cargo.lock
module load Rust/1.79.0-GCCcore-13.3.0
cargo build --manifest-path /path/to/wata/Cargo.toml --release --locked -j 4
python -m benchmarks.pace_wata --dataset /path/to/pace2018 \
  --solver-root /path/to/wata --output /path/to/results/composition-ablation
```

The participant binary uses its supplied tree decomposition for Track 2;
SteinerPy currently ignores that extra structure. Its build used GCC 13.3.0
and Snellius's Rust 1.79.0 module (the exact compiler identification and binary
hashes appear in the CSV). The runner does not modify the participant's code.
Omit `--comparison-results` when exporting only the full SteinerPy sweep.

## Layout

```
benchmarks/
  stp_parser.py     read_stp(path) -> (nx.Graph, terminal_groups)   # SteinLib .stp
  optima.py         known SteinLib optima (B-series inline; CSV override)
  run_benchmarks.py CLI runner (headless, resumable, parallel)
  slurm_array.sh    example Slurm job array (one instance per task)
  data/             .stp instances (gitignored)
  results/          generated CSVs (gitignored)
```

## Getting instances

Download SteinLib instances from <https://steinlib.zib.de> (e.g. the `B` testset)
and unpack the `.stp` files into `benchmarks/data/B/`. The `B` series (sparse
random graphs, |V| = 50–100) is the default smoke set and has proven optima built
into `optima.py`.

## Running

```bash
# baseline vs dual-ascent on the B series, validate against known optima
python -m benchmarks.run_benchmarks \
    --instances benchmarks/data/B --config both \
    --time-limit 300 --out benchmarks/results/B.csv

# parallel sweep
python -m benchmarks.run_benchmarks --instances benchmarks/data/B \
    --config both --jobs 8 --out benchmarks/results/B.csv

# large categories on HPC (one instance per Slurm task, hard wall-clock per task)
sbatch benchmarks/slurm_array.sh
```

The runner is **resumable**: instances already present in the output CSV are
skipped and rows are appended, so a preempted sweep is restarted with the same
command. Per-instance solve time is bounded by `--time-limit`; for hard
wall-clock isolation on a cluster, the Slurm array runs one instance per task.

### Dreyfus-Wagner memory benchmark

`benchmark_dreyfus_memory.py` compares the few-terminal dynamic program across
source checkouts with fixed generated graphs. Every repetition runs in a fresh
process so peak RSS is isolated, and the CSV records the Python, NetworkX, and
SciPy versions used:

```bash
python benchmarks/benchmark_dreyfus_memory.py \
  --source-root /path/to/checkout --label candidate \
  --k 6,7,8,9,10 --sizes 300,1200 --densities 3,8 --repeats 3 \
  --output candidate.csv
```

### Phase-1 research benchmarks

`benchmark_phase1.py` measures the experimental PCSTP reductions, primal
portfolio, and HiGHS cut purger. Every configuration and repetition runs in a
fresh process, with fixed seeds and thread counts; the CSV includes dependency
versions, commit, phase timings, cut/reduction counters, objective and gap, and
isolated peak RSS. Compare the candidate with a detached checkout of current
main, for example:

```bash
python benchmarks/benchmark_phase1.py --feature pc \
  --source-root /tmp/steinerpy-main --label main --configs none,pcd \
  --threads 1 --repeats 3 --output /tmp/phase1-pc.csv
python benchmarks/benchmark_phase1.py --feature pc \
  --source-root . --label candidate \
  --configs none,pcd,pcd+trd,pcd+trd+nodes \
  --threads 1 --repeats 3 --output /tmp/phase1-pc.csv --append
```

Use `--feature primal` or `--feature cut` for the other sweeps. The built-in
suite is deterministic and intended as a smoke/microbenchmark; robust default
decisions still require representative SteinLib/PCSPG suites. A configuration
is marked `MISMATCH` unless every compared run has the same proven objective.

For the primal and PC reduction sweeps, `--instances` accepts either a
directory or a quoted glob. Standard SteinLib instances are used with
`--feature primal`; PCSPG instances (with `TP` prize records) are used with
`--feature pc`. For example:

```bash
python benchmarks/benchmark_phase1.py --feature primal \
  --source-root . --label candidate --instances benchmarks/data/B \
  --configs baseline,local,implied,portfolio --repeats 1 \
  --output /tmp/phase1-B.csv
python benchmarks/benchmark_phase1.py --feature pc \
  --source-root . --label candidate \
  --instances 'benchmarks/data/PCSPG-JMP/K100.*.stp' \
  --configs none,pcd,pcd+trd,pcd+trd+nodes --repeats 1 \
  --output /tmp/phase1-PCSPG.csv
```

Known B-series optima are recorded in every row and a zero-gap disagreement is
reported as `WRONG_OPT`. All instance/configuration objective or certificate
disagreements are still rejected as `MISMATCH`.

## Output columns

`instance, nodes, edges, terminals, opt, base_obj, base_rt, base_gap,
da_obj, da_rt, da_gap, speedup, n_fixed, status`

* `opt` — published SteinLib optimum (ground truth).
* `n_fixed` — variables eliminated by reduced-cost fixing (model shrinkage).
* `status` — `ok`, `MISMATCH` (baseline ≠ DA — a correctness bug), `WRONG_OPT`
  (solved-to-optimality objective ≠ published optimum), or `error:<Type>`.

## Example result (SteinLib `B` series, HiGHS, this machine)

All 18 `B` instances are solved to the **published optimum** by both the baseline
and the dual-ascent accelerator (`base_obj == da_obj == opt`, `status == ok`).
Reduced-cost fixing helps most when the dual-ascent bound is tight:

| instance | \|V\| | \|E\| | \|T\| | opt | base_rt | da_rt | n_fixed | speedup |
|----------|------|------|------|-----|---------|-------|---------|---------|
| b01      | 50   | 63   | 9    | 82  | 0.079s  | 0.001s| 113     | **79×** (solved by DA alone) |
| b05      | 50   | 100  | 13   | 61  | 0.315s  | 0.167s| 255     | 1.89×   |
| b04      | 50   | 100  | 9    | 59  | 0.307s  | 0.289s| 156     | 1.06×   |
| b08      | 75   | 94   | 19   | 104 | 0.099s  | 0.104s| 22      | 0.95×   |
| …        |      |      |      |     |         |       | 0       | ~1×     |

When the bound is loose (`n_fixed = 0`) the accelerator adds a small overhead and
falls back to the baseline result — never changing the optimum. Numbers vary by
machine; regenerate with the command above.

## Reference

SteinLib publishes the proven optima used here as ground truth. **SCIP-Jack**
(Gamrath et al., <https://scipjack.zib.de>) is the state-of-the-art specialised
solver and the external reference point for runtime; a general-MIP solver like
HiGHS is not expected to match its wall-clock. The goal of this harness is to
measure SteinerPy's standing against the optima and quantify how much the
dual-ascent accelerator shrinks the model and speeds up the exact solve.
