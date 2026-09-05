# SteinerPy profiling results — 2026-09-05

Gurobi profiling identifies cut separation as the main bottleneck on the tested terminal-heavy instances. This is work performed by SteinerPy during Gurobi callbacks, not evidence of a difficult branch-and-bound search. Target the graph/separation implementation before a general C++ rewrite.

## Method

- Source commit: `594cef7a41d90246e1d12dc8c7a4d52c51d943b0`; library code unchanged.
- Local macOS ARM64, 10 logical CPUs, Python 3.13.4, Gurobi 13.0.2, SciPy 1.18.0, NetworkX 3.6.1.
- Nine local SteinLib instances: b18, c01, c03, c05, c15, d01, d03, d05, d15. These are a diagnostic sample, not a comprehensive solver benchmark.
- Gurobi uses four threads and a 20-second solve limit. Normal separation uses its default eight threads. Default preprocessing is enabled.
- `default`: dual ascent and DA graph reductions off. `accelerated`: both `dual_ascent=True` and `da_reduce=True`.
- Wall time includes the input graph copy, problem construction/preprocessing, model construction, solving/callbacks, and solution reconstruction. It excludes imports and STP parsing. Lightweight phase timers remain enabled.
- Detailed cProfile runs use serial separation to capture the complete call tree. Profiler-inflated timings are kept separate from ordinary wall-time measurements. The serial switch also sets reduction/ascent workers to one; reduction multiprocessing is below its 1,500-node threshold in this sample, and ascent already defaults to one worker.
- All recorded solves returned the known optimum in the repository reference table, gap zero, and Gurobi optimal status. Every solve used one branch-and-bound node. This does not establish behavior on harder instances.
- SCIP-Jack was not benchmarked: no `scipjack` or `scip` executable was found on PATH. These measurements identify SteinerPy bottlenecks; they do not quantify a SCIP-Jack speed ratio.

## Ordinary wall-time measurements

Times below are individual runs; short runs can include first-use effects. Separation time is included within solving time and total wall time.

| Instance | Nodes / edges / terminals | Default total | Construction/preprocessing | Model build | Separation | Accelerated total |
|---|---:|---:|---:|---:|---:|---:|
| b18 | 100 / 200 / 50 | 0.180 s | 0.051 s | 0.011 s | 0.102 s | 0.125 s |
| c01 | 500 / 625 / 5 | 0.105 s | 0.020 s | 0.014 s | 0.039 s | 0.027 s |
| c03 | 500 / 625 / 83 | 0.386 s | 0.034 s | 0.010 s | 0.317 s | 0.505 s |
| c05 | 500 / 625 / 250 | 0.947 s | 0.091 s | 0.014 s | 0.814 s | 0.850 s |
| c15 | 500 / 2500 / 250 | 1.806 s | 0.236 s | 0.040 s | 1.469 s | 1.966 s |
| d01 | 1000 / 1250 / 5 | 0.302 s | 0.021 s | 0.029 s | 0.113 s | 0.040 s |
| d03 | 1000 / 1250 / 167 | 0.997 s | 0.058 s | 0.029 s | 0.858 s | 1.334 s |
| d05 | 1000 / 1250 / 500 | 3.101 s | 0.311 s | 0.041 s | 2.671 s | 3.428 s |
| d15 | 1000 / 5000 / 500 | 6.576 s | 0.839 s | 0.115 s | 5.422 s | 6.386 s |

On c05, c15, d05, and d15, separation accounts for 81–86% of default total wall time. Model construction accounts for only about 1–2%. Gurobi runtime includes callbacks: treating its reported runtime as time spent solely in the native optimizer would misdiagnose the bottleneck.

## Detailed call profiles

For the serial, default d15 run (11.076 s with cProfile; 5.698 s without it in the first serial run):

- 18 separation rounds × 499 non-root terminals = **8,982 minimum-cut calls**.
- The entire Gurobi callback consumes 7.972 s; separation consumes 7.906 s of that.
- `min_cut_scipy` consumes 6.655 s cumulatively. Sparse compressed-matrix constructors account for 2.887 s cumulatively across the profile, with 179,658 total calls. These figures overlap; do not add them.
- `cut_arcs` consumes 1.163 s across 7,135 calls.
- Preprocessing consumes 2.822 s, including 1.624 s in `_bottleneck_from_mst` and 1.009 s in `long_edge_deletions`.
- The c05 default profile independently shows the same separation pattern: 2,739 minimum-cut calls and 54,791 sparse compressed-matrix constructor calls.

The code constructs residual matrices, transposes/converts them, traverses both residual directions, and creates Python sets for every terminal, even before its caller checks whether the cut is violated. Native maximum-flow routines therefore do not eliminate the surrounding allocation and Python overhead. cProfile cannot reliably separate all native-kernel work from its Python caller, so this report does not claim a pure-Python versus C percentage.

## Repeated thread comparison

c05 and d15 default solves were measured three times per setting, in separate sequential processes. Gurobi stayed at four threads; the effective change here is eight versus one separation threads.

| Instance | Default separation, median total (range) | Serial separation, median total (range) | Speedup |
|---|---:|---:|---:|
| c05 | 0.970 s (0.947–0.989) | 0.742 s (0.710–0.849) | 1.31× |
| d15 | 6.637 s (6.576–6.935) | 5.742 s (5.698–5.907) | 1.16× |

The code creates a fresh thread pool each separation round. The observed slowdown with eight threads is consistent with dispatch, allocation and Python/GIL overhead; the experiment does not isolate those individual causes. It does not imply serial separation is best for all graph sizes.

## Dual-ascent results

On the five terminal-heavy instances, enabling both accelerators removed no additional graph nodes/edges beyond the default heavy preprocessing. It sometimes reduced separation rounds, but the extra dual-ascent computation offset some or all of that benefit. Two dual-ascent calls were recorded per accelerated run: one during graph reduction and another before model construction. Cache/reuse could avoid duplication when graph and terminal state are unchanged.

For fewer-terminal d01, acceleration reduced the post-preprocessing graph from 273 nodes / 506 edges to 14 nodes / 18 edges; total runtime fell from 0.302 s to 0.040 s. On c01 it fell from 0.105 s to 0.027 s. These short, single-run ratios are illustrative, not stable benchmark estimates. Acceleration is useful, but not uniformly beneficial.

## Recommended optimization order

1. **Reduce separation work.** For integer candidates, investigate reachability-based connectivity separation instead of running maximum flow for every terminal. Keep the required correctness checks; fractional solutions still need suitable cut separation.
2. **Avoid unnecessary residual processing.** Check the returned flow value before constructing source/back-side sets when no violated cut can be emitted. Reuse buffers and reduce sparse-matrix conversions. A native implementation of the entire separation pipeline is a better target than merely replacing a maximum-flow kernel that is already compiled.
3. **Tune parallelism to task size.** Test a serial default for small/medium instances or reuse a pool with an evidence-based threshold. The existing `STEINERPY_SEP_THREADS=1` setting enables this experiment without changing library code.
4. **Replace all-pairs terminal bottleneck storage.** `_bottleneck_from_mst` traverses a terminal MST from every terminal, taking quadratic time and storage. Preprocessed tree-path maximum queries can avoid this scaling.
5. **Reuse DA results and enable extra reductions adaptively.** Avoid repeating unchanged computations; evaluate reduction benefit against cost. Harder instances may still require stronger reductions, heuristics, and branch-and-cut integration.

No production optimization was applied in this profiling task. The measurements justify where to investigate first, not a promised speedup from unimplemented changes.

## Reproduce and inspect

Run from the repository root with its existing virtual environment:

```sh
.venv/bin/python benchmarks/profiling/profile_solver.py --tag rerun --limit 20
.venv/bin/python benchmarks/profiling/profile_solver.py --tag rerun-serial --serial --limit 20
.venv/bin/python benchmarks/profiling/profile_solver.py --tag rerun-detail --serial --profile --instances c05 d15 --limit 30
.venv/bin/python benchmarks/profiling/profile_solver.py --tag rerun-fewer --instances c01 c03 d01 d03 --limit 20
```

Each output directory contains environment metadata and raw `results.jsonl`. Profile runs also contain `.prof` files readable with `pstats`/SnakeViz and `.txt` reports sorted by cumulative and self time. Use a new tag to retain previous results. Phase timers can nest: constructor includes DA reduction; solve includes separation; total dual-ascent time includes calls from both preprocessing and solving. Do not sum overlapping timers.
