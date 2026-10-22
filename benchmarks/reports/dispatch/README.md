# Compatible solver selection

These selection rules implement part of [issue #92](https://github.com/berendmarkhorst/SteinerPy/issues/92).
An otherwise ordinary undirected single-tree problem with inactive composed
features can use `SteinerProblem`, including its existing few-terminal dynamic
program. Validation happens before selection. `problem.implementation` exposes
the chosen object; `problem.variant` remains `None` for composed input.

| Requested features | Single-solution implementation | Preprocessing / enumeration |
|---|---|---|
| Ordinary legacy calls, including ordinary forests | Existing dispatch, unchanged | Existing legacy restrictions |
| One undirected terminal set; absent or all-zero node costs; no other active feature | `SteinerProblem`, adapted to the composed result format | Composition's default `preprocess=False` is retained; explicit `True` permits classic reductions for one solution |
| Same single-tree input, with nonzero costs only on mandatory terminals | `SteinerProblem`; original-space decoding adds every mandatory cost and rescales finite gaps | Same preprocessing and enumeration rules as the inactive-cost case |
| Same eligible input, requesting all optima | `FeatureComposition` on the original specification | Original-space enumeration preserves ties, including zero-cost alternatives, regardless of single-solution reductions |
| Composed forests, groups, direction, explicit roots, costs on optional nodes, rewards, hops, degree limits, terminal leaves, or active budgets/penalties | Existing composed strategy | Existing feature-aware preprocessing and enumeration |
| Explicit specialized variants | Existing variant selection | Existing restrictions |

Zero prizes, zero weights, and zero budgets are **not** treated as absent:
they can change the objective or feasible set. Empty terminal-leaf lists and
`None` limits can be inactive, but invalid mappings or unknown options still
raise before constructing a solver. Results preserve original nodes, edge costs,
the composed result class and enumeration behavior. Solver choice is forwarded;
HiGHS remains the default and there is no fallback to a licensed backend.

## First comparison: inactive features

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

## Second comparison: mandatory terminal costs

A nonzero node cost is constant when the node must be selected in every feasible
solution. The second rule permits these costs in an otherwise ordinary single
tree. It still validates and reports all original costs. Both objective bounds
shift by the same amount, so finite relative gaps are rescaled to preserve the
absolute bound difference; unknown gaps remain unknown. Enumeration uses the
original specification, including terminal fees, to retain tied optima.

The comparison uses merged `main` at `e65ce9b` (including PRs #93 and #94) and
candidate `2509155`, on the same environment described above. Five alternating
paired repetitions cover the original eight controls plus four new cases. The
new graphs have either 12 nodes, 18 edges and 3 terminals, or 48 nodes, 140 edges
and 12 terminals. Node fees are `2 + node % 5` for mandatory terminals and zero
elsewhere. Construction, solving and peak RSS retain their earlier definitions.
The five-second solve limit is passed identically to both versions; it is not
a hard wall-clock limit on construction or every internal solving phase.
The harness caps each worker at 30 seconds.

Median total times and median whole-process peak RSS:

| Case | Before (ms) | After (ms) | Proven optima before / after | RSS before / after (MiB) |
|---|---:|---:|---:|---:|
| Ordinary tree | 7.456 | 7.771 | 5 / 5 | 70.6 / 70.6 |
| Ordinary forest | 38.646 | 38.579 | 5 / 5 | 74.4 / 74.4 |
| Zero-cost tree, preprocessing | 8.029 | 7.809 | 5 / 5 | 70.5 / 70.7 |
| Zero-cost forest | 14.916 | 15.513 | 5 / 5 | 73.3 / 73.4 |
| Zero-cost tree, no preprocessing | 1.663 | 1.558 | 5 / 5 | 70.6 / 70.6 |
| Paid nodes + degree + hops | 23.537 | 23.326 | 5 / 5 | 74.4 / 74.4 |
| Groups + budget + degree | 28.985 | 28.985 | 5 / 5 | 74.0 / 74.2 |
| Prizes + forest + paid nodes + hops | 15.707 | 15.047 | 5 / 5 | 73.4 / 73.6 |
| Terminal fees, 12 nodes, preprocessing | 29.399 | 8.238 | 5 / 5 | 74.4 / 70.8 |
| Terminal fees, 12 nodes, no preprocessing | 38.125 | 1.593 | 5 / 5 | 74.0 / 70.5 |
| Terminal fees, 48 nodes, preprocessing | 5091.887 | 39.758 | 0 / 5 | 157.8 / 71.8 |
| Terminal fees, 48 nodes, no preprocessing | 452.641 | 266.244 | 5 / 5 | 81.1 / 78.0 |

All 120 runs returned selections accepted by the original-space feature checker.
The candidate proved optimality in all 60 runs; the baseline did so in 55. Both
versions agree whenever both prove optimality. The baseline's five remaining
runs returned cost 98 with an unknown gap; the candidate proved cost 96. These
are different solve outcomes, so their elapsed-time ratio is not a speedup for
equivalent completed work. For the three new cases completed by both versions,
median speedups are **3.57x, 23.93x and 1.70x**. All eight unchanged controls meet
the previously declared 10% or 2 ms median regression allowance.

The initial attempt stopped when the larger preprocessed baseline did not prove
optimality. The harness now retains validated partial incumbents with explicit
`feasible` status and `null` for an unknown gap. It checks that a partial
incumbent cannot beat a proved optimum from either version, respecting whether
the case minimizes costs or maximizes coverage. The initial aborted attempt is
not part of the repeated measurements. Baseline preprocessing also determines
which general formulation is selected (`FeatureComposition` versus
`GroupComposition`); the large difference between its two settings merits
separate profiling.

[Raw paired measurements](terminal-costs-paired.json) include per-run status,
objective, gap, construction/solve times, memory, versions and commit hashes.
Reproduce with the updated harness at the candidate commit:

```bash
python benchmarks/benchmark_dispatch.py --source-root . \
  --compare-root /path/to/e65ce9b-checkout --repeats 5 \
  --output terminal-costs-paired.json
```

Validation: **3,739 passed, 29 skipped** locally. The 37 additional compatibility
checks cover independent edge-subset oracles, the general formulation, dynamic
programming and MIP paths, preprocessing, constant and fractional objective
gaps, unknown gaps, singleton terminals, snapshots, tied enumeration and active
feature exclusions. Existing HiGHS/Gurobi tests also require the infeasibility
diagnostic preserved by the adapter.

No new HPC runs or PACE result changes. This remains a small deterministic
sample, not a guarantee of faster solving on every graph. Issue #92 stays open
for broader representative combinations and separate preprocessing/model-build/
optimization profiling before further rules are accepted.
