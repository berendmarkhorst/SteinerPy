# SteinerPy benchmarks

Compares SteinerPy's solvers against **published SteinLib optima**, and measures
the effect of the opt-in **dual-ascent accelerator** (`dual_ascent=True`).

This directory is **not** part of the installable package (it ships in neither
the wheel nor the sdist) and is not collected by the default `pytest` run.

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
