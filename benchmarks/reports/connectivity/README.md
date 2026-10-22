# Adaptive connectivity strengthening

The general model keeps its existing formulation for easy solves. When a solve
has more than two seconds available, it initially gives that formulation one
second. Only a native solver time-limit status triggers the stronger formulation,
using the remainder of the same deadline. Interrupts, errors, infeasibility and
completed proofs do not trigger a retry. Short time allowances use the existing
model directly.

This is a follow-up to [issue #92](https://github.com/berendmarkhorst/SteinerPy/issues/92)
and the [phase profiles](../composition-profile/README.md). Public dispatch,
features, result types and explicit backend choice are preserved. Ordinary PACE
calls normally use the classic solver, so a classic-only PACE rerun would not
measure this change.

## Why these constraints are valid

Every feasible original selection is reachable from its active component roots.
A multi-source traversal therefore supplies a directed spanning forest: each
selected non-root has one parent, and roots have none. Continuous auxiliary
parent variables encode this witness:

- Incoming parents at node `v` sum to `selected[v] - active_root[v]`.
- A directed parent arc cannot exceed its selected original arc.
- For an undirected edge, its two parent directions together cannot exceed the
  selected edge.

These constraints imply that there are at least as many selected edges as selected
nodes minus active roots. Their local constraints also exclude fractional
solutions that satisfy that count alone. On a unit-cost triangle with all three
nodes required, the old relaxation permits half of each edge (cost 1.5); the
stronger relaxation requires cost 2.

The original flow constraints remain necessary and remain in the model. Parent
arcs alone could contain disconnected cycles. Original selections can still
include cycles and additional edges: the witness uses only a subset. Thus the
change preserves forests with separate roots, sharing between terminal sets and
groups, direction, optional roots, empty selections, zero-cost ties, rewards,
budgets, penalties, degree/hop bounds and terminal leaves. No acyclicity
restriction is imposed on the original selection.

If the first solve has an incumbent, its values and objective are retained before
modifying the model. If the retry returns no incumbent or a worse one, the earlier
selection is returned with an unknown gap and checked in original coordinates.
Coverage's second objective and enumeration retain their existing deadlines and
proof checks. Construction time can still overrun a backend deadline; this does
not introduce a strict whole-call timeout.

## Experiment design

Baseline: merged main `9c980eca523c21101a5dd624901426a98d3cde56`.
Final candidate: `dd77f5c` (production solver last changed in `11b9c9c`).
Both use Python 3.12.15, HiGHS 1.12.0, NetworkX 3.6.1 and SciPy 1.18.0 on the same
Windows 11 machine. All workers run sequentially in fresh processes with one
solver/BLAS thread. Paired configuration order alternates between repetitions.

The comparison uses the existing fourteen feature cases, now with seeds 42, 43
and 44. Each version has three MIP repetitions per case/seed, with a five-second
solver limit and a 30-second subprocess limit. LP bounds are measured in separate
processes and excluded from MIP timing. The LP diagnostic explicitly adds the
stronger constraints even when an adaptive MIP run would finish without them.
LP numbers use the canonical minimization objective: prize offsets are omitted,
and rewards/coverage are negated.

Returned selections are decoded and cost-checked on the original graph. Proofs
must agree across versions/repetitions; partial solutions cannot beat proved
optima. Coverage's secondary cost is checked. Optimal LP bounds must not worsen
after strengthening or exceed a valid integer solution. LP infeasibility cannot
coexist with a valid selection. Peak RSS covers the whole worker, including
imports and validation, rather than just model allocation.

The preset timing allowance is a median increase of at most 10% or 2 ms, whichever
is larger, assessed separately for each seed/case. Different proof outcomes are
reported separately from comparisons of time to the same proven result.

## Repeated adaptive results

[All 336 observations](adaptive.json) comprise 252 MIP runs and 84 separate LP
diagnostics. Both checkouts were clean when this batch started. Each version has
126 MIP runs: the baseline returns 111 optima, nine unproven incumbents, three
infeasibilities and three numerical errors; the candidate returns 120 optima,
three infeasibilities and the same three numerical errors. There are no worker
timeouts or new failures.

The hard 48-node, twelve-terminal case shows the intended effect:

| Seed | LP lower bound before / after | Baseline incumbent (unproven) | Candidate optimum | Median seconds before / after | Peak RSS MiB before / after |
|---|---:|---:|---:|---:|---:|
| 42 | 80 / 96 | 98 | 96 | 5.101 / 1.281 | 160.9 / 104.9 |
| 43 | 82.5 / 90 | 90 | 90 | 5.118 / 1.309 | 149.5 / 114.6 |
| 44 | 75.5 / 85 | 90 | 85 | 5.107 / 1.269 | 131.6 / 102.5 |

The candidate proves all nine repeated optima within the limit. These are
improved proof outcomes, not a speedup ratio for equivalent completed work.
For seed 42, model building increases from about 79 to 84 ms, while backend
optimization drops from about 5,011 to 1,184 ms. Its model changes from 3,562
columns / 4,242 rows to 3,842 columns / 4,430 rows for the second solve.

All **38 unchanged completed case/seed pairs** (37 optimal, one infeasible)
pass the preset median regression tolerance. Their candidate/baseline time
ratios are between 0.940 and 1.043. They finish without activating strengthening.
The remaining pair is the existing numerical error, retained separately; it is
not counted as a successful solve or speedup. The LP checks pass, including the
stronger LP proving infeasibility of the seed-42 tight coverage budget.

### Hard combination through the public API

A follow-up uses 48 nodes and twelve mandatory terminals together with node
costs, a node-resource budget, maximum degree three and at most 24 selected edges.
`Problem(...)` selects `FeatureComposition` for this input on both versions;
it cannot use the ordinary classic fast path. A regression assertion checks that
dispatch explicitly. The benchmark still names the same strategy directly for
controlled comparison.

[These 24 additional observations](resource-tree.json) contain six LP diagnostics
and eighteen MIP solves, three repetitions per seed/version. The baseline proves
none of its nine results; the candidate proves all nine:

| Seed | Baseline incumbent costs (unproven) | Candidate optimum | Median seconds before / after |
|---|---|---:|---:|
| 42 | 97 | 96 | 5.134 / 1.264 |
| 43 | 96-97 | 94 | 5.118 / 4.977 |
| 44 | 87 | 87 | 5.109 / 3.712 |

Candidate `f0de560` adds only this benchmark case and its dispatch assertion;
production solver code is unchanged from `11b9c9c`. Its dirty metadata reflects
the untracked report directory present during measurement, not uncommitted solver
changes. Seed 43 is close to the deadline, so proof completion can vary on other
machines. Overall, the repeated adaptive experiments retain **360 observations
across fifteen feature combinations**, in addition to the rejected initial screen.

## Rejected always-on formulation

[The initial screen](always-on-screen.json) contains one MIP pair and one LP pair
for all 42 case/seed combinations. It applies the parent constraints during
construction, before the adaptive policy was implemented. It was an uncommitted
exploratory snapshot; its raw metadata records that fact. The final harness's
`--always-strengthen` option reproduces that construction policy.

This formulation proves the three hard mandatory-cost examples in roughly
0.26-0.29 seconds, where the baseline times out without proof. However, some easy
cases regress: for example, seed-42 terminal leaves increase from about 42 to
73 ms, seed-43 direction from 58 to 83 ms, and seed-44 groups/forest from 410 to
561 ms. These were sufficient reasons to reject unconditional use. A stronger LP
does not guarantee a faster integer solve.

## Validation and known limitation

The full local suite passes **3,781 tests, with 29 skipped**. New checks cover the
triangle LP bound, feasible cyclic/multi-root/optional-root/empty selections on
both HiGHS and Gurobi, incumbent fallback and its reset, native interruption and
deadline behavior, and consistency of benchmark bounds. Existing independent
subset-oracle and tied-enumeration tests cover the broader feature combinations.
The subsequent benchmark-only public-dispatch assertion also passes (eleven
profiler tests in its targeted run).

The expanded sample exposes an existing HiGHS numerical rejection on the
seed-43 prize/forest case. The native canonical objective is
`2.999999797884911`, while its feasible original selection costs exactly `3`.
Both versions retain the objective-consistency guard; these runs remain failures.
[Issue #97](https://github.com/berendmarkhorst/SteinerPy/issues/97) tracks this
separately. The harness exits unsuccessfully when any worker errors, even though
it saves and checks all other observations.

These small synthetic graphs do not establish universal speedups, large-instance
memory behavior or Gurobi performance. Gurobi is covered by correctness tests;
performance measurements here use HiGHS. No HPC jobs or release were started.

## Reproduce

Check out the baseline separately, then run from the candidate checkout:

```bash
python benchmarks/benchmark_connectivity.py \
  --compare-root /path/to/main-at-9c980ec --output adaptive.json

# Reproduce the rejected formulation's policy (one repetition per seed):
python benchmarks/benchmark_connectivity.py --always-strengthen --repeats 1 \
  --compare-root /path/to/main-at-9c980ec --output always-on-screen.json
```

Results are saved after every worker, including errors. `--configs flow-filtered`
exercises the same formulation with resource filtering enabled.
The original repeated batch used `dd77f5c`, whose default has fourteen cases.
At `f0de560`, `--cases resource-tree` reproduces the follow-up; the default now
includes all fifteen cases.

## Proposed small Snellius pilot

After review, compare the frozen baseline and candidate on twelve fixed cases:
six forced-general PACE cases (001 and 002 from each track), four combined-feature
cases (resource tree at seeds 42/43, groups/forest at seed 44, signed rewards at
seed 44), and two classic-path controls (Track 1/001 and Track 2/001).

With two repetitions, one HiGHS thread, an 8 GiB worker cap and a 120-second
whole-worker deadline, 48 runs have at most **1.6 solver-core hours** of worker
runtime. Actual charged allocation time can be higher; the existing 64-CPU,
24-hour full-sweep Slurm script should not be reused unchanged. Keep concurrent
workers low and right-size the allocation before submission. HiGHS needs no WLS
sessions. If Gurobi is added later, retain the existing shared 24-reservation
guard, below the user's 30-session license limit.

This is a proposed engineering comparison, not a new PACE ranking. No submission
has been made.
