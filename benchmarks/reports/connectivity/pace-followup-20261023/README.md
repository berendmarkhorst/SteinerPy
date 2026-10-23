# Connectivity policy and PACE follow-up

Issue #104 follows the [earlier Snellius pilot](../snellius-pilot-20261023/README.md).
This experiment separates the general-model switching policy from the normal
public API's comparison with a PACE participant. It uses baseline
`d982be2f56905db602fd86818b4ac5f23edf3dd5`, candidate
`80055b308ab19ba344f03b84918abbdf7dd47bc0`, and harness
`1b980bbf422a634798aed249507b188bc796ae73`.

## Connectivity policy

The candidate allows the initial model two seconds instead of one before adding
stronger connectivity constraints. It only probes when more than four seconds
remain, preserving a useful retry budget. Time limits, checked incumbents and
proof status retain their existing semantics. This changes the scheduling of
equivalent formulations, not which feature combinations or selections are valid.

The paired Snellius experiment covers twelve combined-feature inputs, each
repeated twice per policy: **48 workers**. Both policies prove all 24 optima and
agree on every objective. After retrieval, every original node/edge selection
was revalidated; regenerated input hashes and cross-version proof checks pass.

Median full worker wall seconds are below. Each pair has the same proved
objective in both repetitions. The common whole-worker deadline is fifteen
seconds, including startup, model construction and certificate checks.

| Case | Optimum | One-second probe (s) | Two-second probe (s) |
|---|---:|---:|---:|
| Signed rewards 40 | 19 | 1.604 | 0.903 |
| Signed rewards 41 | 18 | 3.608 | 1.805 |
| Signed rewards 42 | 24 | 1.504 | 0.852 |
| Signed rewards 43 | 19 | 1.604 | 1.604 |
| Signed rewards 44 | 19 | 4.107 | 1.904 |
| Signed rewards 45 | 19 | 1.604 | 1.604 |
| Resource tree 42 | 96 | 2.505 | 3.206 |
| Resource tree 43 | 94 | 5.862 | 6.710 |
| Resource tree 44 | 87 | 5.510 | 6.064 |
| Resource tree 45 | 99 | 5.709 | 6.310 |
| Groups/forest 42 | 25 | 1.505 | 1.504 |
| Groups/forest 44 | 22 | 1.303 | 1.303 |

The measured signed-rewards regression is removed: seed 44 takes **1.904 instead
of 4.107 seconds**. Seed 41 also benefits. Both now finish on the initial model
instead of restarting with a larger model. The cost is **0.55–0.85 seconds** on
each resource case, which still requires strengthening. Resource seed 43 remains
fast at 6.710 seconds; the preceding experiment's pre-strengthening baseline
took 102.254 seconds, under a different deadline and on another node. That
historical number is context, not a new paired speedup.

The total measured optimization time across all 24 runs changes only from
50.705 to 49.528 seconds. Do not claim a broad speedup from the larger change in
total worker time (72.852 to 67.538 seconds): first-run startup costs affect seeds
40–42, while their native optimization measurements are mostly unchanged except
for seed 41. The policy trades a small delay on harder cases for avoiding
unnecessary restarts on near-finished cases; it is not faster on every input.

## Focused local diagnostic of the classic solver

An additional bounded Windows check uses Track 1/101, the same frozen candidate,
HiGHS 1.15.1, one thread, a fifteen-second whole-worker cap and two alternating
repetitions. The default public API is compared with existing constructor options;
no additional solver code is changed. All six observations are retained in
[classic-probe.jsonl](classic-probe.jsonl), with the exact options and environment
in [classic-probe-manifest.json](classic-probe-manifest.json).

| Constructor options | Both repetitions | Median wall (s) |
|---|---|---:|
| Default | No connectivity-valid incumbent returned | 13.977 |
| `dual_ascent=True` | Valid cost 1,601,646, not proven optimal | 14.078 |
| `dual_ascent=True, primal_local_search=True, implied_profit=True` | Whole-worker timeout | 15.078 |

The published optimum is 1,601,190, so the dual-ascent result is approximately
0.0285% above it. Its reported solver gap is approximately 0.0468%, not zero.
Both returned original-edge certificates were revalidated after the run. The
larger primal portfolio is not automatically better under the same deadline.

Reproduce these calls using `Problem(graph, terminals=terminals, preprocess=True,
**options).get_solution(time_limit=remaining, solver="highs", threads=1,
decompose=False)`, with a fresh worker, the common external deadline/checker and
two seconds reserved for validation. The temporary diagnostic harness changes
only the constructor options; the manifest pins the production and shared
benchmark revisions. The Windows memory limit is not enforced. These six local
measurements guide the next investigation and are not pooled with HPC results.

## Matched PACE comparison

All 48 PACE workers finished: twelve inputs, two methods and two repetitions.
The normal public `Problem` API with HiGHS reaches **eight known optima in
24 runs**, compared with **eighteen for wata&sigma**. That is four versus nine
distinct cases, with the same outcome in both repetitions. Every optimal result
matches the published exact reference and has a checked original tree.

| Outcome | SteinerPy normal API / HiGHS | wata&sigma |
|---|---:|---:|
| Proven and reference-matching optimum | 8 | 18 |
| Whole-worker timeout, no scored solution | 12 | 6 |
| Returned without a connectivity-valid incumbent | 4 | 0 |
| Rejected original-tree certificate | 0 | 0 |

Every SteinerPy worker selects the existing `SteinerProblem` strategy; none is
forced through the general composition model. Consequently the connectivity
probe change is not exercised in this part. This measures the classic default
HiGHS path, including reductions and the few-terminal DP where eligible. The
explicit DP ceiling is ten terminals; dual ascent and decomposition are off.

The table reports median whole-worker seconds. A timeout or no-incumbent duration
is not a time to optimum. The same 120-second cap applies to both methods,
including startup and original-certificate validation.

| Track / input | Terminals | SteinerPy seconds / outcome | wata&sigma seconds / outcome |
|---|---:|---|---|
| 1 / 001 | 4 | 0.703 / optimum 503 | 0.303 / optimum 503 |
| 1 / 050 | 10 | 2.457 / optimum 2016 | 0.903 / optimum 2016 |
| 1 / 101 | 16 | 118.570 / no valid incumbent | 0.452 / optimum 1601190 |
| 1 / 150 | 24 | 120.073 / timeout | 1.804 / optimum 2173 |
| 2 / 001 | 25 | 1.704 / optimum 1086 | 0.352 / optimum 1086 |
| 2 / 050 | 724 | 13.270 / optimum 4069628 | 0.452 / optimum 4069628 |
| 2 / 101 | 1000 | 120.030 / timeout | 120.032 / timeout |
| 2 / 150 | 71 | 119.035 / no valid incumbent | 0.954 / optimum 7102320 |
| 3 / 001 | 16 | 120.107 / timeout | 0.804 / optimum 2256 |
| 3 / 050 | 84 | 120.070 / timeout | 17.285 / optimum 8401094 |
| 3 / 101 | 363 | 120.054 / timeout | 120.086 / timeout |
| 3 / 150 | 1184 | 120.357 / timeout | 120.082 / timeout |

The four returned errors are the two repetitions each of Track 1/101 and
Track 2/150. They report `Exact solve stopped before finding a connectivity-valid
feasible incumbent.` They do not represent invalid trees or false proofs. Hard
timeouts take precedence when the worker fails to exit before the outer deadline,
even if a partial progress file already records that message.

The PACE reference remains substantially ahead in this sample. Track 2/050 is a
like-for-like example: both methods prove the same optimum, with median end-to-end
times 13.270 versus 0.452 seconds, approximately 29 times apart. Ratios on the
subsecond examples include Python startup overhead and should not be interpreted
as algorithm-only speedups. No timing ratio is assigned to unfinished solves.

### What to work on next

Track 1/101 has 311 nodes and sixteen terminals. SteinerPy spends approximately
0.41 seconds parsing/importing, 0.07 preprocessing and 0.25 building its model,
then nearly the remaining allowance in solving without returning a valid tree.
Track 2/050 spends about 0.31 seconds building its model but roughly twelve
seconds in the solve call. These measurements point to the classic optimization
path as the next target; they do not suggest a large cost from constructing the
modular API object.

First benchmark the existing dual-ascent option on the failing classic cases,
recording incumbent quality, bounds, cut rounds and optimization time. The
focused local diagnostic above already returns a checked near-optimum where
the default returns nothing. Preserve a checked incumbent across deadline exits
and ensure acceleration work shares the total time budget. Do not enable the
larger primal portfolio or raise the exponential DP cutoff merely because one
option helped one case. Better pruned DP for more terminals and tree-decomposition
methods remain possible later algorithmic improvements, guided by those profiles.

This experiment reruns the original
[wata&sigma solver](https://github.com/wata-orz/steiner_tree), revision
`2700bdc3033d0d01c5d3648832b4535fceb70991`, built with Rust 1.79.0. It does not
rerun SCIP-Jack or Gurobi, and the twelve fixed inputs do not establish a full
PACE ranking. The exact-track PACE budget was thirty minutes; this bounded
comparison uses two. Track 3 is an exact-solver diagnostic here, not a reproduction
of its heuristic leaderboard. Compare algorithms under matched limits before
generalizing to other hardware, budgets, feature combinations or solver backends.

## Environment and retained observations

The Snellius job is **28477305**, on `tcn42`: AMD EPYC 7H12, Linux, Python 3.12.3,
native HiGHS 1.15.1, NetworkX 3.7, NumPy 2.5.4 and SciPy 1.18.1. Three case
sequences run concurrently, with one solver/BLAS thread and an enforced 8 GiB
address-space cap per worker. Both revisions use clean pinned checkouts.
No Gurobi/WLS sessions are opened.

All **96 requested workers completed** in a 17-minute-45-second allocation,
from 13:03:20 to 13:21:05 Europe/Oslo (11:03:20–11:21:05 UTC). The runner writes
both complete result sets, then exits nonzero because of the four no-incumbent
returns. Consequently [Slurm accounting](accounting.txt) says `FAILED` / `1:0`;
the job was not interrupted and no requested run is missing. The allocation used
4.73 core-hours, below its 45-minute cap; this is not an invoice.

[manifest.json](manifest.json) records the fixed inputs, revisions, resource
limits, environment and reference binary hashes. [combined.jsonl](combined.jsonl)
retains all 48 policy observations, including certificates, native solver status,
phase timings and memory. Paths are made portable and the full hostname is
removed; no result is discarded.

[pace.jsonl](pace.jsonl) retains all 48 PACE observations. All 96 worker files,
identities, source revisions and resource settings match the manifest. After
retrieval all twelve PACE input hashes and all **74 returned HPC certificates**
were rechecked: 48 combined-feature selections and 26 PACE trees. Objective,
reference/proof consistency and cross-version input checks pass. The raw worker
directories and logs remain locally under
`benchmarks/results/strengthening-followup/archive-28477305` and on Snellius
under `~/steinerpy-benchmarks/pace-followup-20261023`.

The separate Windows/HiGHS 1.15.1 screen also reaches all 48 proved optima; its
[manifest](local-manifest.json) and [observations](local-screen.jsonl) are retained.
It uses the same generated input hashes and fifteen-second cap, with no enforced
Linux memory limit. Other development checks overlapped part of that local run,
so it screens correctness and is not used for controlled timing claims. Do not
pool it with Snellius timings.

The [experiment instructions](../../../pace_comparison.md) describe the fixed
cohort and reproducible submission. Keep the fifteen-second policy experiment,
120-second matched PACE comparison, local screening and six-run local diagnostic
separate when interpreting these results.
