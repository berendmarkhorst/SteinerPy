# Benchmarks

## Scope and scale

SteinerPy is a **Python / NetworkX library**, and it is honest about where that puts it.
Its goal is *breadth and accessibility* — a unified, pip-installable, MIT-licensed API that solves many Steiner variants exactly (with a certified optimality gap), and drops straight into a Python data pipeline.
Its sweet spot is **small-to-medium instances** (up to roughly a few thousand nodes/edges), rapid prototyping, and research that needs to move across variants quickly.

It is **not** a performance competitor to dedicated C/C++ exact solvers such as [SCIP-Jack](https://scipjack.zib.de) (the DIMACS 2014 / PACE 2018 winner).
On the hard SteinLib / DIMACS families — instances with tens of thousands of nodes/edges, or the notoriously difficult PUC/hypercube sets — SCIP-Jack is faster by orders of magnitude, thanks to a decade of C engineering, a far deeper reduction-test arsenal, and SCIP's full branch-and-cut.
The benchmarks below use the **mid-size** SteinLib `C` tier (500 nodes, up to 12 500 edges) — still well short of a large-scale exact race, but heavy enough that the dual-ascent acceleration and reductions earn their keep; the point is *solution quality and a certified gap versus fast heuristics*, not raw scale.
If you need to solve million-edge instances to proven optimality, reach for SCIP-Jack; if you want exactness, breadth, and Python ergonomics on tractable instances, SteinerPy is built for you.

## Setup

Two scripts at the repository root benchmark SteinerPy against the standard baselines on **literature instances** (not random graphs), scoring every method on one optimality-gap axis against published / proven optima:

- [`benchmark_compare.py`](https://github.com/berendmarkhorst/SteinerPy/blob/main/benchmark_compare.py) — Steiner tree, vs `networkx` `approximation.steiner_tree` (Kou and Mehlhorn), on the bundled SteinLib C-series (the `B` and `D` tiers are bundled too).
- [`benchmark_pcstp.py`](https://github.com/berendmarkhorst/SteinerPy/blob/main/benchmark_pcstp.py) — Prize-collecting Steiner (PCSPG), vs [`pcst_fast`](https://github.com/fraenkel-lab/pcst_fast) (Goemans–Williamson 2-approx), on the DIMACS 2014 JMP set.

The numbers below were produced on an **Apple M4, Python 3.13.4, networkx 3.6.1, Gurobi 13.0.2**; absolute times are machine-dependent, the gaps are not.
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

Average exact-vs-DA speedup (SteinerPy-exact / SteinerPy-exact+DA): **2.62×**.

- **SteinerPy-exact+DA** solves all 20 to proven optimality. Crucially, on `c18` (500 nodes, 12 500 edges, 83 terminals) the plain exact solve finds no feasible solution within 60 s, while the dual-ascent accelerator (`dual_ascent=True`) proves optimality — it doesn't just speed up easy instances, it makes a hard one solvable (and is ~2.6× faster on average).
- **NetworkX-kou / -mehlhorn** are the fastest but ≈6% off on average and solve only 1/20 to optimality, with no bound to tell you so.
- **SteinerPy-heuristic** (`exact=False`, no ILP) stays in NetworkX's speed class (~0.25 s), beats both NetworkX heuristics on quality (4.45% vs ~6%), *and* reports a **certified** gap — bounding how far it could be from the optimum (0 = proven optimal), something `networkx.steiner_tree` cannot provide.

## Prize-collecting Steiner (PCSPG) — SteinerPy vs pcst_fast

*Instances: DIMACS 2014 JMP `K100/P100` (6 files) · solver: gurobi · time limit: 30s.*

| Method | Avg time (s) | Avg gap % | # optimal |
|--------|-------------:|----------:|----------:|
| SteinerPy-exact | 0.165 | 0.00 | 6/6 |
| SteinerPy-heuristic | 0.008 | 2.94 | 2/6 |
| pcst_fast | 0.000 | 2.30 | 3/6 |

`SteinerPy-exact` (`pc_transform=True`) proves optimality on all six in well under a second; `pcst_fast` is essentially instantaneous but ≈2–3% off with no certificate, and `SteinerPy-heuristic` matches its speed class while reporting a certified gap.
(The bundled `K400/P400` instances are larger; run the full set yourself to see the time-limit behaviour.)

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
