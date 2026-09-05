# Follow-up: smaller bottleneck index and reuse of dual ascent

This follow-up is measured against commit `9ccb5f686ef3d51d209f1386ccefa2754205767b`, which already includes the cut-separation improvements. Benefits here are additional to the [first optimization](OPTIMIZATION.md).

## What changed

- **Lazy terminal bottleneck index.** Replace a traversal from every terminal and a stored distance for every pair with a binary-lifting tree-path maximum index. Construction and storage are O(T log T); queries are O(log T), with no growing pair cache. Lazy mapping rows retain the existing `bott[a][b]` and `.get()` behavior, including disconnected forests and missing terminals. The reduction criteria and resulting distances are unchanged.
- **One-use preprocessing handoff.** When the last DA reduction pass removes nothing, save its dual-ascent result for the solve. A linear snapshot comparison checks graph structure, costs, weight attribute, terminal groups, roots, and edge/arc ordering before reuse. The entry is consumed once, including when stale. A result from before a graph-changing final pass is never retained. This avoids the repeated expensive ascent without trusting graph identity alone.

## Full solves

Nine SteinLib instances, both default and accelerated configurations, three repeats per revision, alternating revision order in sequential processes: **108 solves**. Every run returned the known optimum with zero gap and Gurobi optimal status. The reduced node and edge counts remained the same between revisions.

Environment: macOS ARM64 with 10 logical CPUs, Python 3.13.4, Gurobi 13.0.2 using four solver threads, SciPy 1.18.0, NetworkX 3.6.1. Both revisions use serial separation by default. The solve limit is 20 seconds. Times exclude imports/file parsing and include graph copying, preprocessing, model construction, solving/callbacks, and reconstruction. Lightweight phase timers are enabled; cProfile is not.

`accelerated` means `dual_ascent=True, da_reduce=True`; `default` uses neither. Heavy reductions are enabled in both. All times below are medians of three runs.

| Instance | Default before | Default after | Accelerated before | Accelerated after | Accelerated time reduction |
|---|---:|---:|---:|---:|---:|
| b18 | 0.082 s | 0.077 s | 0.064 s | 0.055 s | 14.1% |
| c01 | 0.078 s | 0.078 s | 0.026 s | 0.026 s | -1.4% |
| c03 | 0.253 s | 0.237 s | 0.362 s | 0.321 s | 11.5% |
| c05 | 0.545 s | 0.486 s | 0.567 s | 0.415 s | 26.8% |
| c15 | 1.170 s | 1.137 s | 1.269 s | 1.125 s | 11.4% |
| d01 | 0.253 s | 0.254 s | 0.039 s | 0.039 s | 0.8% |
| d03 | 0.689 s | 0.646 s | 0.981 s | 0.849 s | 13.5% |
| d05 | 2.002 s | 1.752 s | 2.384 s | 1.845 s | 22.6% |
| d15 | 3.325 s | 3.005 s | 3.809 s | 3.277 s | 14.0% |

Construction/preprocessing in the default configuration:

| Instance | Before | After | Speedup |
|---|---:|---:|---:|
| c05 | 0.087 s | 0.019 s | 4.65× |
| d05 | 0.320 s | 0.053 s | 6.04× |
| d15 | 0.841 s | 0.485 s | 1.74× |

Dual ascent takes roughly half its former time on the terminal-heavy accelerated cases, because the solve consumes the result computed during preprocessing. There are still two calls to the public function; the second returns the validated stored result instead of doing another ascent.

## Isolated index construction and memory

The reproducible `benchmark_bottlenecks.py` uses weighted random trees (seed 42), three timing repeats, and 10,000 deterministic pair queries. Peak Python allocations are measured in a separate tracemalloc pass so memory measurement does not inflate the reported times. Input graph storage is excluded; memory figures are for construction of the bottleneck representation.

| Terminals | Old build | New build | Old peak allocations | New peak allocations | Old / new time for 10,000 queries |
|---|---:|---:|---:|---:|---:|
| 512 | 0.0918 s | 0.0006 s | 9.54 MB | 0.13 MB | 0.0010 / 0.0070 s |
| 2,048 | 1.5463 s | 0.0024 s | 151.40 MB | 0.55 MB | 0.0027 / 0.0078 s |

This trades constant-time dictionary lookups for logarithmic path queries. Individual queries are slower, but removing quadratic construction and storage is a substantial gain on terminal-heavy graphs. The full-solve measurements above include that tradeoff; the isolated build-time ratio is not an end-to-end solver speedup.

## Validation and limits

- Full local suite: **563 passed, 26 skipped**.
- New tests compare every indexed pair against explicit NetworkX tree paths on ten random trees, test forests and missing/isolated terminals, and exercise a 4,096-node path beyond the recursion limit while checking subquadratic storage.
- Reuse tests cover consumption only once, changed costs, terminal groups, roots, nodes, edges, arc ordering, and weight attributes, plus a final graph-changing reduction pass.
- No new compiled extension or dependency is required.
- Very small problems show little benefit and some sub-millisecond variation. These local SteinLib instances were all solved at the root node; performance on difficult branch-and-bound instances remains unmeasured.

## Reproduce

Use the same virtual environment and benchmark data with both revisions. The baseline is available by checking out commit `9ccb5f6`. Copy the new microbenchmark script into that checkout to run the same workload.

```sh
python benchmarks/profiling/profile_solver.py --tag followup-1 --instances b18 c01 c03 c05 c15 d01 d03 d05 d15 --limit 20
python benchmarks/profiling/benchmark_bottlenecks.py --sizes 512 2048 --repeats 3 --queries 10000
```

Repeat the full-solve command with new tags three times per revision and alternate the order. Raw observations are in [followup-comparison.csv](followup-comparison.csv), [bottleneck-before.jsonl](bottleneck-before.jsonl), and [bottleneck-after.jsonl](bottleneck-after.jsonl). Constructor, dual-ascent and separation timers can overlap and should not be summed.
