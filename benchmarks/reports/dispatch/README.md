# Compatible dispatch: first measured rule

This implements the first selection rule for [issue #92](https://github.com/berendmarkhorst/SteinerPy/issues/92).
An otherwise ordinary undirected single-tree problem with inactive composed
features can use `SteinerProblem`, including its existing few-terminal dynamic
program. Validation happens before selection. `problem.implementation` exposes
the chosen object; `problem.variant` remains `None` for composed input.

| Requested features | Single-solution implementation | Preprocessing / enumeration |
|---|---|---|
| Ordinary legacy calls, including ordinary forests | Existing dispatch, unchanged | Existing legacy restrictions |
| One undirected terminal set; absent or all-zero node costs; no other active feature | `SteinerProblem`, adapted to the composed result format | Composition's default `preprocess=False` is retained; explicit `True` permits classic reductions for one solution |
| Same eligible input, requesting all optima | `FeatureComposition` on the original specification | Original-space enumeration preserves ties, including zero-cost alternatives, regardless of single-solution reductions |
| Composed forests, groups, direction, explicit roots, paid nodes, rewards, hops, degree limits, terminal leaves, or active budgets/penalties | Existing composed strategy | Existing feature-aware preprocessing and enumeration |
| Explicit specialized variants | Existing variant selection | Existing restrictions |

Zero prizes, zero weights, and zero budgets are **not** treated as absent:
they can change the objective or feasible set. Empty terminal-leaf lists and
`None` limits can be inactive, but invalid mappings or unknown options still
raise before constructing a solver. Results preserve original nodes, edge costs,
the composed result class and enumeration behavior. Solver choice is forwarded;
HiGHS remains the default and there is no fallback to a licensed backend.

## Local measurements

Eight deterministic 12-node, 18-edge examples, five repetitions per version,
one HiGHS thread and a five-second solve limit. Every measurement uses a fresh
process. Old and new versions alternate within each case, reversing their order
on alternate repetitions. Imports and graph generation precede the timer.
Construction includes preprocessing; solving includes model construction and
result validation. These phases are recorded separately in the raw data, but
model-building time is not isolated from solving. Peak RSS covers the worker's
entire process, including imports; it is not a memory limit.

Environment: Windows 11 build 26100, Intel64 Family 6 Model 181 Stepping 0,
Python 3.12.15, HiGHS 1.12.0, NetworkX 3.6.1 and SciPy 1.18.0. Baseline
`2c8a961` has identical solver code to merged baseline `8c8d6db`; candidate
`d1da004` contains the final tree-only rule. This isolates dispatch from the
separate incumbent-cleanup change in PR #93.

Median construction-plus-solve times:

| Example | Before (ms) | After (ms) | Before / after | Peak RSS before / after (MiB) |
|---|---:|---:|---:|---:|
| Ordinary tree | 10.987 | 10.983 | 1.00 | 70.2 / 70.4 |
| Ordinary forest | 49.075 | 50.052 | 0.98 | 74.0 / 73.9 |
| Zero-cost nodes, tree, preprocessing | 38.189 | 10.559 | 3.62 | 74.1 / 70.3 |
| Zero-cost nodes, forest | 19.864 | 19.273 | 1.03 | 73.0 / 73.1 |
| Zero-cost nodes, tree, no preprocessing | 52.085 | 2.105 | 24.74 | 73.7 / 70.1 |
| Paid nodes + degree + hops | 32.114 | 31.693 | 1.01 | 74.0 / 74.2 |
| Groups + budget + degree | 37.572 | 38.288 | 0.98 | 73.7 / 73.8 |
| Prizes + forest + paid nodes + hops | 20.769 | 20.898 | 0.99 | 73.2 / 73.2 |

All 80 runs returned matching optimal objectives and zero gaps, with no failures.
All eight examples pass the predeclared engineering threshold: at most 10% or
2 ms increase in the median, whichever is larger. This small repeated sample
supports the rule; it does not guarantee faster performance on every instance.
Larger representative suites and finer profiling remain work for issue #92.

The [paired raw results](paired.json) are the final comparison. Earlier
[baseline](baseline.json) and [initial candidate](initial-candidate.json)
measurements exposed a forest regression (18.171 ms to 43.317 ms), which is why
forests were excluded from the rule. A later [unpaired candidate run](candidate.json)
showed increasing wall times even on unchanged paths; the paired experiment
addresses that timing drift. All records are retained rather than silently
replacing the earlier measurements.

## Reproduce

From the candidate checkout, with the same interpreter and dependencies:

```bash
python benchmarks/benchmark_dispatch.py --source-root . \
  --compare-root /path/to/baseline-checkout --repeats 5 --output paired.json
```

The harness checks matching objective/gap pairs and bounds each worker at
30 seconds, including startup. It uses no HPC resources or Gurobi sessions.
The compatibility regressions cover all-zero and absent costs, forests,
non-neutral features, validation order, solver/dependency forwarding, original
graph snapshots, isolated terminals and tied-optimum enumeration. The two
follow-up PRs together pass 3,702 local tests (29 skipped), including positional
solver-argument compatibility.

The PACE composition benchmark now constructs its generic strategy explicitly,
so changing public dispatch cannot silently turn that comparison into another
classic solve. The published PACE baseline is unchanged.
