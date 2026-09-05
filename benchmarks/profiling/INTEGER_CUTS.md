# Third profiling pass: integer connectivity cuts and long-edge reduction

For measurements after integration with current main, see [PR validation](PR_VALIDATION.md).

This pass compares against `5ca239aa82f402d0b44da0e7a24951d118a4d5a3`, which already includes both earlier optimization rounds. These are additional improvements on `perf/reduce-cut-separation-overhead`.

## What the new profile found

On d15/default, cProfile attributed 3.841 s of a 5.427 s run to cut separation, including 4,977 calls to `min_cut_scipy`. Long-edge reduction accounted for another 0.986 s, mostly in Python's per-vertex shortest-path search. A separate diagnostic classified 17 of 18 separation rounds as integral: those rounds consumed 2.165 s, versus 0.041 s for the fractional round. Profile and diagnostic times come from different runs and must not be added together.

Most integer disconnections do not need a maximum-flow computation. They can be exposed by traversing selected arcs and checking the capacity of the resulting partition. SteinerPy already uses compiled SciPy graph routines; avoiding thousands of unnecessary calls and their sparse-matrix allocations is the main opportunity here.

## Changes

- **Checked connectivity certificates for integer candidates.** Reuse the root-reachability traversal, examine disconnected weak components and reverse-reachable terminal sets, and deduplicate identical cuts within each group/round. Every proposed cut is checked against the existing scaled capacities and creep tolerance. Uncertified demands fall back to maximum flow. This preserves valid separation even when many tiny capacities add up to a satisfied demand.
- **Retain minimum cuts at Gurobi LP nodes.** The fast partitions are valid connectivity inequalities but need not be minimum cuts. Gurobi uses them for incumbent checks only; LP-node callbacks retain minimum-cut separation even if relaxation values happen to be integral. HiGHS's iterative integer separation can use the certificates as well.
- **Compiled, cost-bounded long-edge searches.** Build one shared weighted CSR graph per reduction pass and call SciPy Dijkstra with a distance limit. Enable this on graphs with at least 128 nodes and average degree at least four, only when `max_settle` covers all vertices. Other cases retain the Python search and its work cap. No new dependency or compiled extension is introduced.

A preliminary reachability-only policy increased c13/accelerated from approximately 2.55 s to 4.31 s by enlarging Gurobi's search tree. Keeping LP minimum cuts and adding whole-component certificates brought that case back close to baseline. The final repeated measurements below include this case; cheaper separation does not guarantee a smaller search tree.

## Measurement protocol

13 SteinLib instances, two configurations, three runs per revision, alternating revision order in sequential processes: 156 full solves. The baseline ran from a clean worktree with the same interpreter and instance data. Four Gurobi threads, default serial separation, 20 s solver limit. No cProfile during timing comparisons. Times include graph copying, preprocessing, model construction, callbacks/solving and reconstruction, but exclude imports and file parsing. Phase timers can overlap and must not be summed.

Environment: macOS 26.6.2 ARM64, 10 logical CPUs, Python 3.13.4, Gurobi 13.0.2, SciPy 1.18.0, NetworkX 3.6.1. `default` disables dual ascent; `accelerated` sets both `dual_ascent=True` and `da_reduce=True`. Both retain heavy preprocessing. The name `accelerated` denotes the configuration, not a promise that it is faster for each instance.

All **156 runs** matched the known optimum within 1e-6, returned zero gap, and finished with Gurobi optimal status. Reduced node and edge counts matched between revisions. Times below are medians of three runs.

| Instance | Default before | Default after | Speedup | DA before | DA after | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| b18 | 0.080 s | 0.039 s | 2.02× | 0.055 s | 0.035 s | 1.54× |
| c01 | 0.079 s | 0.077 s | 1.03× | 0.026 s | 0.026 s | 1.00× |
| c03 | 0.235 s | 0.073 s | 3.22× | 0.314 s | 0.153 s | 2.05× |
| c05 | 0.468 s | 0.094 s | 4.99× | 0.406 s | 0.227 s | 1.79× |
| c08 | 1.218 s | 0.535 s | 2.28× | 0.920 s | 0.683 s | 1.35× |
| c13 | 3.136 s | 2.088 s | 1.50× | 2.458 s | 2.569 s | 0.96× |
| c15 | 1.092 s | 0.243 s | 4.50× | 1.119 s | 0.333 s | 3.36× |
| d01 | 0.248 s | 0.258 s | 0.96× | 0.038 s | 0.039 s | 0.98× |
| d03 | 0.646 s | 0.224 s | 2.89× | 0.845 s | 0.529 s | 1.60× |
| d05 | 1.719 s | 0.286 s | 6.01× | 1.814 s | 1.792 s | 1.01× |
| d08 | 6.397 s | 5.597 s | 1.14× | 5.657 s | 4.212 s | 1.34× |
| d13 | 19.109 s | 5.591 s | 3.42× | 6.806 s | 3.545 s | 1.92× |
| d15 | 2.928 s | 0.629 s | 4.65× | 3.142 s | 1.752 s | 1.79× |

**Limits and search-tree changes:** c13/accelerated is 4.5% slower (2.458 → 2.569 s), with node count increasing from 1 to 53. d01/default is 4.3% slower (0.248 → 0.258 s), with 1 → 50 nodes. d08/default improves despite 1 → 39 nodes. Conversely, c13/default drops from 47 to 11 nodes and d13/default from 44 to 14. These node counts were stable across all three repeats. d05/accelerated is essentially unchanged; tiny accelerated c01/d01 cases also show little benefit. This is why both configurations and branching cases are retained in the table.

## Profile confirmation and remaining costs

The separate cProfile rerun confirms that the expensive operations were removed. These cumulative profile times are instrumented and must not be treated as unprofiled wall-time speedups.

| Case / operation | Before | After |
|---|---:|---:|
| d15/default max-flow calls | 4,977 | 37 |
| d15/default separation | 3.841 s | 0.347 s |
| d15/default long-edge reduction | 0.986 s | 0.138 s |
| c05/default max-flow calls | 2,047 | 0 |
| c05/default separation | 0.985 s | 0.088 s |
| d15/accelerated max-flow calls | 4,212 | 1,479 |
| d15/accelerated separation | 3.529 s | 1.505 s |

On d15/default, the remaining separation cost is mostly constructing and checking connectivity certificates (0.299 s under cProfile). On d15/accelerated, fractional/LP separation still costs 1.505 s and dual ascent 1.095 s. Unprofiled final medians also attribute about 4.58 s of d08/default’s 5.60 s and 4.27 s of d13/default’s 5.59 s to separation. Reusing residual flow state across sinks and reducing certificate allocations are plausible next targets, but neither is implemented or claimed as a measured gain here. Changes to cut selection must continue to be judged by whole-solve time and search-tree behavior.

## Validation

Full local suite: **578 passed, 26 skipped**. Separation tests use independent NetworkX minimum cuts on random directed and disconnected graphs, multiple terminal groups, fractional/integer demands, tolerance boundaries, and serial/threaded execution. Additional tests check certificate deduplication, capacity-check fallback, empty arc sets, and explicitly requested minimum cuts. Native long-edge tests compare deletions against both the Python implementation and independent NetworkX shortest paths, including zero costs, equal-cost paths, a custom weight attribute, and the size/density/work-cap gates.

These measurements support improvements on the tested workloads, not a general speed guarantee or parity with SCIPJack. SCIPJack was not rebenchmarked in this pass.

## Reproduce

Run the following command three times per revision with distinct tags, alternating revision order. Use commit `5ca239a` for the baseline and the commit containing this report for the new implementation, with the same virtual environment and benchmark data.

```sh
python benchmarks/profiling/profile_solver.py --tag timing-1 \
  --instances b18 c01 c03 c05 c08 c13 c15 d01 d03 d05 d08 d13 d15 --limit 20
python benchmarks/profiling/profile_solver.py --tag profile --serial --profile \
  --instances c05 d15 --limit 30
python -m pytest tests/ -q --cov=steinerpy --cov-report=term-missing
```

All final timing observations, solver statuses, gaps and node counts are in [third-comparison.csv](third-comparison.csv). Compact before/after cProfile observations are in [third-profile-summary.json](third-profile-summary.json). Machine-local detailed profiles remain in the ignored profiling output directories.
