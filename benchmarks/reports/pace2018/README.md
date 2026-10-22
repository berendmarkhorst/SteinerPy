# SteinerPy on PACE 2018

Completed snapshot generated 2026-10-22T07:10:38.381346+00:00.

This benchmark attempts the official few-terminal, low-treewidth and heuristic tracks. Track 2 is solved as a graph problem; SteinerPy does not consume its supplied tree decompositions. The [organizers' dataset](https://github.com/PACE-challenge/SteinerTree-PACE-2018-instances) contains 200, 200 and 199 instances respectively; Track 3 instance 058 was withdrawn.

## Measurements

Each configuration runs once per instance in a fresh process with one solver/BLAS thread, an 8 GiB address-space limit, and a 1,800-second wall deadline covering imports, parsing, preprocessing, model construction, solving and validation. Default classic reductions and the few-terminal dynamic program remain enabled. Dual-ascent acceleration, decomposition, and experimental local-search/implied-profit options are disabled. Heuristic mode is the existing single-pass `exact=False` algorithm; it can finish before the deadline.

The Gurobi runs are licensed; HiGHS and heuristic mode are the open-source alternatives. All costs and connectivity are checked on original edges. A feasible incumbent is not counted as an optimum unless its value matches a published optimum or the solver proves it. Timeouts produce no scored solution when the public solve call has not returned by the deadline.

| Track | Configuration | Finished / expected | Valid trees | Proven/reference optima | Timeouts | Errors / resource exits | Median wall, valid (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | gurobi | 200/200 | 179 | 152 | 13 | 8 | 3.206 |
| 1 | heuristic | 200/200 | 200 | 16 | 0 | 0 | 1.513 |
| 1 | highs | 200/200 | 132 | 132 | 43 | 25 | 2.405 |
| 2 | gurobi | 200/200 | 187 | 126 | 6 | 7 | 12.626 |
| 2 | heuristic | 200/200 | 198 | 4 | 1 | 1 | 2.106 |
| 2 | highs | 200/200 | 129 | 126 | 15 | 56 | 12.220 |
| 3 | gurobi | 199/199 | 44 | 18 | 114 | 41 | 776.402 |
| 3 | heuristic | 199/199 | 132 | 0 | 1 | 66 | 14.545 |
| 3 | highs | 199/199 | 12 | 11 | 132 | 55 | 293.705 |

Median times cover valid returned solutions only; configurations can solve different subsets. Use the per-instance CSV for paired timing comparisons. `solve_seconds` measures the whole `get_solution` call (including model construction); `model_build_seconds` times its classic model builders, and `solver_seconds` is the library-reported solver runtime.

## Original private instances

| Track | Configuration | Finished / expected | Optima | Quality index (%) |
|---|---|---:|---:|---:|
| 1 | gurobi | 100/100 | 72 | 89.989 |
| 1 | heuristic | 100/100 | 8 | 95.456 |
| 1 | highs | 100/100 | 61 | 61.000 |
| 2 | gurobi | 100/100 | 57 | 90.000 |
| 2 | heuristic | 100/100 | 3 | 96.568 |
| 2 | highs | 100/100 | 60 | 63.000 |
| 3 | gurobi | 99/99 | 8 | 17.984 |
| 3 | heuristic | 99/99 | 0 | 62.932 |
| 3 | highs | 99/99 | 5 | 5.051 |

Quality is `100 × mean(reference_upper / objective)` over the full requested private set, with zero for missing/invalid solutions. A completed sweep is required before publishing this score. The upper reference is not necessarily an optimum for Track 3.

## Historical PACE participants

The following are published 2018 results, not reruns on Snellius. [Source: official PACE report, sections 4.1–4.3](https://pacechallenge.org/files/PACE18-report.pdf).

| Track | Place | Team | Published private-set result |
|---|---:|---|---:|
| 1 | 1 | wata&sigma | 95/100 solved |
| 1 | 2 | Jagiellonian | 94/100 solved |
| 1 | 3 | reko (SCIP-Jack) | 93/100 solved |
| 2 | 1 | reko (SCIP-Jack) | 92/100 solved |
| 2 | 2 | wata&sigma | 77/100 solved |
| 2 | 3 | Tom | 58/100 solved |
| 3 | 1 | CIMAT | 99.93% quality |
| 3 | 2 | reko (SCIP-Jack) | 99.85% quality |
| 3 | 3 | MJG | 99.80% quality |

The exact-track solved counts provide historical context on the same private instances. They do not establish a competition ranking or speedup: hardware, software and resource limits differ. Track 3's current archive omits one formerly private instance and its reference bounds need not be the same as the 2018 scoring bounds, so its quality index is not directly interchangeable with the published competition score. The table above does not contain rerun measurements.

## Same-hardware comparison sample

This separate experiment uses instances 001–003 from each track, one run per method, with a 90-second full wall deadline, one thread and an 8 GiB per-process address-space limit. It compares the classic specialized HiGHS model, the generic composition model (`Problem(..., node_costs={})`), and the original MIT-licensed [wata&sigma submission](https://github.com/wata-orz/steiner_tree). The external binary and checker run inside the same deadline; the binary's reported runtime is separately available as `solve_seconds`. Its peak RSS is the maximum individual-process RSS, not summed parent/child memory. All output edges are checked.

| Track | Method | Finished / 3 | Valid trees | Optima | Timeouts | Errors | Median wall, valid (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | highs | 3/3 | 3 | 3 | 0 | 0 | 1.604 |
| 1 | composition-highs | 3/3 | 1 | 1 | 2 | 0 | 0.904 |
| 1 | wata | 3/3 | 3 | 3 | 0 | 0 | 0.403 |
| 2 | highs | 3/3 | 3 | 3 | 0 | 0 | 1.504 |
| 2 | composition-highs | 3/3 | 3 | 0 | 0 | 0 | 88.876 |
| 2 | wata | 3/3 | 3 | 3 | 0 | 0 | 0.303 |
| 3 | highs | 3/3 | 0 | 0 | 3 | 0 | — |
| 3 | composition-highs | 3/3 | 0 | 0 | 3 | 0 | — |
| 3 | wata | 3/3 | 3 | 3 | 0 | 0 | 1.304 |

These nine cases are a fixed engineering sample, not a representative ranking of the full competition. [Per-instance comparison data](comparison.csv) records source revisions, compiler, binary checksums and matching dataset checksums. Keep this sample's 90-second results separate from the full sweep's 1,800-second results.

## Reproducibility

```json
{
  "commit": "7d4d8b8e058c08a8f4215ca0fb446ce4d35f059c",
  "cpu": "AMD EPYC 7H12 64-Core Processor",
  "dataset_commit": "4df73cea9c311faea7d03e6d6bffa8733c34a1aa",
  "memory_gb": 8.0,
  "packages": {
    "gurobipy": "13.0.3",
    "highspy": "1.15.1",
    "networkx": "3.7",
    "numpy": "2.5.4",
    "scipy": "1.18.1"
  },
  "python": "3.12.3",
  "time_limit": 1800.0
}
```

## Where wall deadlines expired

| Configuration | Phase | Instances |
|---|---|---:|
| gurobi | solve | 133 |
| heuristic | solve | 2 |
| highs | solve | 190 |

## Rejected results and resource failures

| Configuration | Phase | Reason | Instances |
|---|---|---|---:|
| gurobi | solve | GurobiError: Out of memory | 5 |
| gurobi | solve | RuntimeError: Exact solve stopped before finding a connectivity-valid feasible incumbent. | 46 |
| gurobi | validate | ValueError: objective below published lower bound | 2 |
| gurobi | validate | ValueError: reported objective differs from original edge costs | 2 |
| gurobi | validate | ValueError: solution is not a tree spanning all terminals | 1 |
| heuristic | solve | MemoryError | 67 |
| highs | solve | MemoryError | 42 |
| highs | solve | RuntimeError: Exact solve stopped before finding a connectivity-valid feasible incumbent. | 94 |

The [per-instance results](results.csv) include checksums, versions, objective/proof distinctions, memory and phase timings. Raw JSON and original-edge solution certificates remain in the HPC run directory. The runner and Slurm script live in `benchmarks/pace2018.py` and `benchmarks/slurm_pace2018.sh`.

[Findings and proposed follow-up work](findings.md) connect these measurements to the package's modularity and performance goals.
