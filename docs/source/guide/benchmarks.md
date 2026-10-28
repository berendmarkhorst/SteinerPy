# Benchmarks

These measurements describe specific datasets, configurations, and machines.
Use them to understand the tested workloads; runtime on your inputs may differ.

## Scope and scale

The [PACE 2018 report](https://github.com/berendmarkhorst/SteinerPy/blob/main/benchmarks/reports/pace2018/README.md)
measures all three undirected Steiner-tree tracks on Snellius, with HiGHS,
Gurobi and heuristic configurations reported separately. It includes per-instance
CSV data, failures, resource limits and historical participant results, including
SCIP-Jack. Historical results use different hardware and software; they do not
establish a measured speedup or a current competition ranking. Check the report's
completion status before interpreting aggregate scores.

The smaller SteinLib experiments below compare solution quality, proof status
and runtime with Python baselines. They do not measure SCIP-Jack's runtime.

## Setup

Two scripts at the repository root benchmark SteinerPy against the standard baselines on literature instances, scoring every method on one optimality-gap axis against published / proven optima:

- [`benchmark_compare.py`](https://github.com/berendmarkhorst/SteinerPy/blob/main/benchmark_compare.py) — Steiner tree, vs `networkx` `approximation.steiner_tree` (Kou and Mehlhorn), on the bundled SteinLib C-series (the `B` and `D` tiers are bundled too).
- [`benchmark_pcstp.py`](https://github.com/berendmarkhorst/SteinerPy/blob/main/benchmark_pcstp.py) — Prize-collecting Steiner (PCSPG), vs [`pcst_fast`](https://github.com/fraenkel-lab/pcst_fast) (Goemans–Williamson 2-approx), on the DIMACS 2014 JMP set.

The numbers below were produced on an Apple M4, Python 3.13.4, networkx 3.6.1, Gurobi 13.0.2; absolute times are machine-dependent, the gaps are not.
Each table is generated verbatim by the script's `--markdown` flag — see [Reproducing](#reproducing) below.

## Steiner tree — SteinerPy vs NetworkX

*Instances: SteinLib `C` (20 files, 500 nodes, up to 12 500 edges / 250 terminals) · solver: gurobi · time limit: 60s/solve.*

| Method | Avg time (s) | Avg gap % | # optimal |
|--------|-------------:|----------:|----------:|
| NetworkX-kou | 0.139 | 6.04 | 1/20 |
| NetworkX-mehlhorn | 0.010 | 6.06 | 1/20 |
| SteinerPy-exact | 9.585 | 0.00 | 19/20 |
| SteinerPy-exact+DA | 9.049 | 0.00 | 20/20 |
| SteinerPy-heuristic | 0.245 | 4.45 | 2/20 |

The plain exact solve returned no feasible solution for `c18` within the limit;
its averages therefore do not cover the same returned-solution set as the
dual-ascent run. The gap column measures error against reference optima, not
the solver-reported certificate. Rounded zero does not imply a completed proof.

## Prize-collecting Steiner (PCSPG) — SteinerPy vs pcst_fast

*Instances: DIMACS 2014 JMP `K100/P100` (6 files) · solver: gurobi · time limit: 30s.*

| Method | Avg time (s) | Avg gap % | # optimal |
|--------|-------------:|----------:|----------:|
| SteinerPy-exact | 0.165 | 0.00 | 6/6 |
| SteinerPy-heuristic | 0.008 | 2.94 | 2/6 |
| pcst_fast | 0.000 | 2.30 | 3/6 |

The zero runtime for `pcst_fast` is rounded to three decimal places.
The exact run uses the prize transformation; heuristic results do not all match
the reference optima. These are six-instance results, not a general ranking.

(reproducing)=
## Reproducing

```shell
# Steiner tree vs networkx (exact + heuristic) on the C tier — the table above
python benchmark_compare.py --markdown --instances benchmarks/data/C --time-limit 60

# Heuristics only (no Gurobi license needed; scales to large instances)
python benchmark_compare.py --markdown --heuristics-only --solver highs

# PCSPG vs pcst_fast on the bundled JMP set
python benchmark_pcstp.py --markdown

# Any directory of SteinLib/PCSPG .stp files (default is the B tier)
python benchmark_compare.py --markdown --instances path/to/instances
```

Drop the `--markdown` flag for the annotated fixed-width console report.
Both scripts default to Gurobi; pass `--solver highs` for the always-available HiGHS backend.
