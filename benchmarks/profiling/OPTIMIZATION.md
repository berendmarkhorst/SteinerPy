# Faster connectivity cut separation

This branch targets the dominant bottleneck measured in [the initial profile](REPORT.md): repeated per-terminal cut separation inside Gurobi callbacks. It keeps the existing mathematical formulation and maximum-flow fallback for terminals that cannot be certified cheaply.

## Implementation

1. Build a capacity-filtered graph once per source group and traverse it from the root. If every arc on a path has capacity at least the largest active group demand minus the existing separation tolerance, every root/terminal cut on that path has enough capacity. Skip maximum flow for those reached terminals. This is valid for fractional as well as integer candidates; failure to find such a path is not treated as disconnection.
2. For remaining terminals, use the maximum-flow value before allocating and traversing the residual graph. Satisfied flows need no residual partition. When back cuts are disabled, omit the reverse traversal entirely.
3. Extract cut arcs with NumPy membership masks instead of Python traversal over all outgoing arcs in a large source partition.
4. Default to serial separation, matching the repeated profiling evidence. `STEINERPY_SEP_THREADS` still enables a user-selected worker count, independently of MIP solver threads. Certified terminals are removed before thread dispatch.
5. Fix the existing NetworkX fallback to complete the maximum flow before taking source-side reachability. An intermediate preflow can retain excess at internal nodes and yield a non-minimum source partition. A dedicated regression test covers this case.

## Validation

- Full local test suite: 542 passed, 26 skipped (Python 3.13.4). Includes licensed Gurobi tests and the existing HiGHS/variant tests.
- New tests compare returned cuts against independent NetworkX minimum cuts on 20 deterministic random graphs per configuration. They cover directed/disconnected graphs, multiple terminal groups, integer/fractional capacities, numerical tolerances, back cuts on/off, explicit threaded execution, and the NetworkX fallback.
- Additional tests check that satisfied paths skip maximum flow, split fractional flows still invoke it, satisfied flows skip residual traversal, arbitrary node labels remain supported, and separation thread overrides remain available.
- 108 benchmark solves: nine instances × two configurations × two revisions × three repeats. Every run returned the known optimum with gap zero and Gurobi optimal status.

## Repeated performance measurements

Baseline: clean checkout of `594cef7a41d90246e1d12dc8c7a4d52c51d943b0`. Both revisions used the same installed Python and solver environment, identical input files, four Gurobi threads, and a 20-second solver limit. Revision order alternated across repeats; runs were sequential. No cProfile instrumentation was active in these timings, only lightweight phase timers. Wall time includes graph copying, preprocessing, model construction, solving/callbacks, and reconstruction; imports and file parsing are excluded.

Environment: macOS ARM64, 10 logical CPUs, Python 3.13.4, Gurobi 13.0.2, SciPy 1.18.0, NetworkX 3.6.1.

`default` uses normal heavy preprocessing with dual ascent and DA graph reduction disabled. `accelerated` enables both. Times below are medians of three runs. The comparison includes the intentional default change from eight separation threads to one.

| Instance | Default before | Default after | Speedup | Accelerated before | Accelerated after | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| b18 | 0.129 s | 0.080 s | 1.61× | 0.102 s | 0.063 s | 1.60× |
| c01 | 0.097 s | 0.076 s | 1.27× | 0.027 s | 0.025 s | 1.05× |
| c03 | 0.402 s | 0.245 s | 1.64× | 0.494 s | 0.361 s | 1.37× |
| c05 | 0.944 s | 0.570 s | 1.65× | 0.832 s | 0.571 s | 1.46× |
| c15 | 1.858 s | 1.184 s | 1.57× | 2.045 s | 1.257 s | 1.63× |
| d01 | 0.313 s | 0.251 s | 1.25× | 0.040 s | 0.039 s | 1.03× |
| d03 | 1.035 s | 0.679 s | 1.52× | 1.329 s | 0.997 s | 1.33× |
| d05 | 3.156 s | 2.016 s | 1.57× | 3.319 s | 2.414 s | 1.37× |
| d15 | 6.671 s | 3.370 s | 1.98× | 6.527 s | 3.856 s | 1.69× |

The largest measured benefit is on terminal-heavy d15. Results on very short solves are noisy and should not be generalized. These are easy-to-moderate local SteinLib cases, all solved at the root node; performance on large, difficult branch-and-bound instances remains unmeasured. The branch does not establish parity with SCIP-Jack.

## Reproduce

Use the same virtual environment and instance data for both checkouts. `profile_solver.py` is new on this branch, so copy it to the baseline checkout before running there. The script imports SteinerPy from its containing checkout.

```sh
# Run in each checkout, using the same Python executable:
python benchmarks/profiling/profile_solver.py --tag comparison-1 --instances b18 c01 c03 c05 c15 d01 d03 d05 d15 --limit 20
# Repeat with new tags and alternate revision order. For detailed call profiles:
python benchmarks/profiling/profile_solver.py --tag detail --serial --profile --instances c05 d15 --limit 30
```

Run directories contain raw JSON and metadata. Binary profiles and machine-local output directories are ignored by git. [comparison.csv](comparison.csv) contains all 108 timing observations, objectives, and gaps used above. Phase timers overlap: solve time includes separation, and constructor time can include DA graph reduction.

## Follow-up work

The subsequent [follow-up](FOLLOWUP.md) replaces the all-pairs terminal
bottleneck table and reuses unchanged preprocessing dual-ascent results. It is
benchmarked separately against this first cut-separation optimization.
