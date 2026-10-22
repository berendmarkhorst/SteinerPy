# General-formulation profiling for issue #92

The slow 48-node example spends about **98% of its time inside HiGHS**, not in
resource filtering. The best formulation depends on the requested features.
This experiment supports improving the general flow model and making measured,
feature-specific choices; it does not support replacing every general solve
with the cut model.

This PR adds diagnostic code and measurements only. Public dispatch, algorithms,
feature semantics, enumeration and the published PACE baseline are unchanged.
It is independent of the mandatory-terminal optimization in PR #95.

## Method and compatibility

Each measurement runs in a fresh process with one HiGHS thread. The benchmark
validates the normalized specification, then constructs the named strategy
explicitly. This separates resource filtering from public dispatch, which can
otherwise change the formulation when `preprocess=True` is supplied.

| Configuration | Implementation | Compatibility in this experiment |
|---|---|---|
| `flow` | `FeatureComposition`, filtering off | All fourteen original-space feature combinations below |
| `flow-filtered` | Same `FeatureComposition`, filtering on | Same features; removes edges individually forbidden by resource bounds |
| `cuts` | `GroupComposition`, filtering off | Only the five explicitly listed cases: mandatory/optional node costs, undirected groups with one terminal tree, rooted directed groups, and degree/hop limits |

The cut strategy is **not** forced onto forests, terminal leaves, rewards,
node-resource limits or penalties. Compatibility is checked before a worker
starts; it is not inferred from a finite objective. The cut strategy's
enumeration delegates to the general original-space enumeration, preserving
tied solutions. This experiment measures single solutions, not enumeration.
Direct class selection is confined to the benchmark; `Problem.implementation`
remains the public way to inspect dispatch.

The broader method matrix and classic-path measurements are in
[the dispatch report](../dispatch/README.md). Classic preprocessing/DP applies
only to the compatible single-tree dispatch described there, not to these
forced general models.

The deterministic graphs use seed 42, integer edge costs 1–9, and nonzero costs
on optional nodes, except the mandatory-cost case. The two cost-only cases have
48 nodes and 140 edges; the other cases have 24 nodes and 66 undirected edges
(132 arcs for the directed case). Five repetitions alternate configuration
order. Each strategy receives a five-second solver limit and each subprocess
has a 30-second wall limit. Model construction is outside the solver limit;
backend deadlines do not constitute a strict whole-call limit.

Environment: Windows 11 build 26100, Intel64 Family 6 Model 181 Stepping 0,
Python 3.12.15, HiGHS 1.12.0, NetworkX 3.6.1 and SciPy 1.18.0. Initial measurement
commit: `299ece4`, based on merged `main` at `e65ce9b`. The follow-up budget and
uninstrumented measurements use `b13f91c`; solver code is identical. Later
changes strengthen result comparison/tests without changing timed solving.

## Measured results

Median whole-call wall time in milliseconds, including construction and result
decoding. A dash means the cut model was not eligible for that case.

| Combined features | Flow | Flow + filtering | Cuts | Outcome |
|---|---:|---:|---:|---|
| Mandatory terminal costs, 48 nodes | 5091.6 | 5094.6 | 448.5 | Flow: unproven cost 98; cuts: proved cost 96 |
| Optional node costs, 48 nodes | 229.2 | 225.0 | 205.7 | All prove cost 15 |
| Groups + terminal tree + node costs | 216.1 | 215.2 | 123.0 | All prove cost 21 |
| Forest + node costs | 69.2 | 70.6 | — | All prove cost 21 |
| Groups + forest + node costs | 429.2 | 428.6 | — | All prove cost 25 |
| Direction + rooted groups + terminal tree + node costs | 52.3 | 51.0 | 120.4 | All prove cost 21 |
| Degree + hops + node costs | 61.2 | 62.4 | 94.8 | All prove cost 17 |
| Terminal leaves + degree + node costs | 42.5 | 42.4 | — | All prove cost 17 |
| Prizes + forest + node costs | 293.5 | 300.1 | — | All prove objective 27 |
| Prizes + edge/node budgets + node costs | 201.0 | 197.3 | — | All prove objective 22 |
| Group coverage + edge/node budgets, tight mandatory demands | 42.0 | 38.2 | — | Both prove infeasibility |
| Group coverage + edge/node budgets, feasible demands | 100.1 | 95.7 | — | Both prove coverage 2, then minimum cost 11 |
| Penalty cost + missed-terminal budget + hops + node costs | 35.2 | 35.8 | — | All prove objective 7 |
| Signed node rewards + root + edge/node budgets + degree | 111.0 | 113.1 | — | Both prove reward 24 |

Every successful result is decoded and cost-checked on the original graph
after timing. Across configurations/repetitions, proved objectives must agree;
partial incumbents cannot beat those optima. Coverage comparisons also check
the secondary minimum cost. A proved infeasibility cannot coexist with a
validated feasible result. Different outcomes at a time limit are reported
explicitly, not treated as an equivalent-work speedup.

The initial 155 runs produced 135 proved optima, ten valid but unproven results,
and ten infeasible budget solves. Those last ten initially had a generic
`error` label; their recorded HiGHS status is `Infeasible`. They remain unchanged
in the raw file. The harness now distinguishes proved infeasibility from errors.
Twenty subsequent budget runs confirm ten infeasibilities and prove ten feasible
two-stage solutions. Forty uninstrumented checks yield thirty optima and ten
unproven results. Thus all **215 runs** are retained, including the failures to
produce a feasible solution; no timeout or unexpected solver failure occurred.

## Where the time goes

The timers use exclusive accounting: time spent in a nested phase is removed
from its parent. Their per-run sum equals the measured wall time. Instrumented
functions are restored on exit, including on exceptions.

| Phase | What is timed |
|---|---|
| Preprocessing | Consumption of the resource-filtering iterator; materialized once by the diagnostic wrapper |
| Compilation | Node splitting, group terminal attachment and construction of the cut solver's problem view |
| Model build | Initial variables/constraints and feature-model configuration |
| Optimization | HiGHS `solve`, including native presolve, LP/MIP solving and interface entry |
| Separation | Python/native cut detection outside backend solves |
| Decoding | Solver selections mapped/checked in original graph coordinates |
| Other | Normalization, objective/setup work, cut insertion and orchestration outside named hooks, instrumentation setup/teardown |

For the hard mandatory-cost case, the flow model takes about **77 ms** to build,
**5007 ms** to solve, and less than **0.04 ms** to filter. Filtering leaves this
model unchanged: its first solve has **3562 columns and 4242 rows** either way.
The cut formulation starts with **1024 columns and 984 rows**, adds cuts over
21 backend solves, and proves optimality in about **448 ms** overall. Its median
phase times are approximately 50 ms building, 191 ms solving, 119 ms separating
cuts and 88 ms in other orchestration. Dimensions are recorded after each
backend solve; these are formulations of the same original problem.

For other cases:

- Groups + forest: about 405 ms of 429 ms is backend solving.
- Prizes + forest: about 278 ms of 294 ms is backend solving.
- Directed groups: flow model construction is about 23 ms and solving 26 ms;
  cut separation/orchestration makes the cut formulation slower overall.
- Node-resource filtering reduces the prize/budget model from 491 to 407
  columns and from 610 to 514 rows. Construction falls from about 14 to 12 ms;
  total time improves only modestly on this example.
- Feasible coverage uses two backend solves, first maximizing coverage and then
  minimizing cost. Both phases are included in the optimization total.

The **40 uninstrumented checks** use the same inputs, settings and five
repetitions for mandatory costs, degree/hops and prizes/forest. They confirm the
same objectives and proof outcomes. Profiled/uninstrumented median time ratios
range from **0.959 to 1.035**. They were separate batches, so timing drift is
included; these figures are a sanity check, not a precise overhead estimate.
No public performance claim depends on a few-percent timing difference.

Peak RSS is recorded for the whole worker, including imports and the extra
validation after timing. It is not model-only allocation or a memory cap.

## Evidence and reproduction

- [Initial 155 profiled runs](profiled.json)
- [20 budget checks with explicit infeasibility reporting](coverage.json)
- [40 checks without instrumentation](uninstrumented.json)

From the repository root, using the recorded interpreter/dependencies:

```bash
# Reproduce the initial thirteen cases at commit 299ece4:
python -m benchmarks.profile_composition --repeats 5 --output profiled.json

# Reproduce the follow-ups at b13f91c:
python -m benchmarks.profile_composition --repeats 5 \
  --cases coverage-budget coverage-feasible --output coverage.json
python -m benchmarks.profile_composition --no-profile --repeats 5 \
  --cases mandatory-costs degree-hops prize-forest --output uninstrumented.json
```

The current default includes all fourteen cases. Output is saved after each
worker so earlier observations survive a later failure. Tests cover nested
phase accounting, temporary-hook restoration, native solve capture, filtering,
infeasibility, compatibility rejection and both objective directions, including
coverage's secondary cost.

The full local suite passes **3,711 tests, with 29 skipped**, including nine new
profiler regressions. This branch starts from `main` and does not include PR #95.

## Follow-up decision

Keep compatibility-based selection; **do not switch every composition to cuts**.
The cut model is about 1.76x faster for the measured undirected groups, but about
2.30x slower for directed groups and 1.55x slower for degree/hops. These tradeoffs
are substantially larger than the instrumentation sanity-check variation.

The next optimization experiment should target the general flow formulation:
measure its LP bound and try stronger connectivity constraints on the hard
tree/group/forest cases, preserving every active feature. Assess construction
overhead separately on small directed/leaf examples. Benchmark any candidate
against these preserved results, with the existing median regression allowance
of 10% or 2 ms (whichever is larger) and no loss of feasibility/proof quality.

This is a small synthetic sample with one graph seed. It neither ranks SteinerPy
against PACE participants nor proves a universally fastest method. No HPC jobs,
Gurobi benchmark sessions, new solver defaults or release were requested.
