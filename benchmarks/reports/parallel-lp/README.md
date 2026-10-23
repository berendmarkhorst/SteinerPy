# HiGHS parallel LP stall (issue #85)

The four-node group/node-cost/degree example in [issue #85](https://github.com/berendmarkhorst/SteinerPy/issues/85)
stalls in its fourth preliminary LP relaxation with HiGHS 1.12.0, 1.13.1 and
1.14.0. The LP consumes the remaining three-second allowance without proving
infeasibility. Allowing HiGHS to choose the LP algorithm resolves the example in
about 0.02 seconds. HiGHS 1.15.0 and 1.15.1 already finish quickly with the
original settings.

The fix temporarily changes `parallel="on"` to `parallel="choose"` only for
preliminary LP cut rounds, and only before HiGHS 1.15. It restores the setting
even if the LP raises an exception. Both the forest/group and directed SAP cut
loops use the same helper. Thread counts, MIP parallelism, explicit serial
settings, Gurobi, feasibility checks and optimality checks retain their existing
behavior. This workaround can be removed when the minimum HiGHS version is 1.15.

## Paired measurements

Measured locally on 2026-10-23: Windows 11, 14 logical CPUs, Python 3.12.15,
NetworkX 3.6.1. Each solve runs sequentially in a fresh subprocess. Solver
threads are automatic (`0`); OpenMP/BLAS environment thread limits are `1`.
Each solver gets three seconds and each subprocess has a twelve-second hard
timeout. Baseline/candidate order alternates across three repetitions. Imports
and graph construction are excluded from solve timings; model construction is
included. Native LP/MIP calls are also timed separately.

Baseline is merged main `35e911557a93c1e71fbc223a2e52787a4ee4d785`; the final
candidate is `9007009626d34c81e80a0471af41dcc8bd6de47e`. Both checkouts were clean.
The version matrix uses the exact issue example, including its zero-cost edge.

| HiGHS | Median seconds before | Median seconds after | Before / after outcome (each 3 runs) |
|---|---:|---:|---|
| 1.12.0 | 3.0077 | 0.0168 | No valid incumbent / proven infeasible |
| 1.13.1 | 3.0078 | 0.0181 | No valid incumbent / proven infeasible |
| 1.14.0 | 3.0106 | 0.0180 | No valid incumbent / proven infeasible |
| 1.15.0 | 0.0184 | 0.0178 | Proven infeasible / proven infeasible |
| 1.15.1 | 0.0196 | 0.0198 | Proven infeasible / proven infeasible |

The old time-limited runs and new completed proofs are different outcomes, so
their ratio is not a measure of general solver speedup.

Controls include the same infeasible graph with positive edge costs, a feasible
degree-two variant (optimum 9), and a seeded 40-node graph used for classic,
forest and directed SAP solves. The latter disable preprocessing and
Dreyfus-Wagner; classic/forest also disable decomposition. Returned edges,
connectivity and costs are checked in original coordinates; the composed
feasible case also checks degrees and node costs. Proven costs must agree
between paired runs. The harness defines every graph and seed.

| Control | HiGHS 1.12.0 before / after (seconds) | HiGHS 1.15.1 before / after (seconds) |
|---|---:|---:|
| Positive-cost infeasible | 0.0183 / 0.0168 | 0.0178 / 0.0162 |
| Feasible degree-two composition | 0.0270 / 0.0259 | 0.0266 / 0.0244 |
| Classic | 0.2751 / 0.2106 | 0.2185 / 0.2180 |
| Forest (time-limited) | 3.1144 / 3.0761 | 3.0591 / 3.0714 |
| Directed SAP | 0.1241 / 0.1085 | 0.1133 / 0.1120 |

The 1.12.0 controls use the earlier ungated candidate `8027740`; its solver
behavior on 1.12 is the same as the final candidate. The final version matrix
rechecks 1.12 with the version gate. The 1.15.1 controls use the final candidate.
Positive controls prove infeasibility; feasible composition, classic and SAP
prove optimality in every run. Forest reaches the time limit: on 1.12 the
baseline returns no valid incumbent in three runs, while the candidate returns
two unproven incumbents; on final 1.15.1 each returns one unproven incumbent.
These forest timings do not establish a speed improvement.

## Why limit the workaround to older HiGHS?

The initial ungated candidate also changed LP settings on HiGHS 1.15.1. Its
three-run SAP control was slower (0.1104 to 0.1404 seconds median); a nine-run
follow-up was closer (0.1249 to 0.1288). Together the twelve runs had medians
0.1174 and 0.1334 seconds. These short runs are noisy, but there was no clear
benefit to changing a version that already resolves the reported stall. The
final gate preserves all original solver settings on 1.15 and later. The final
controls show similar timings, without establishing a universal performance
guarantee.

[HiGHS' parallelism documentation](https://github.com/ERGO-Code/HiGHS/blob/master/docs/src/parallel.md)
describes differences between serial and parallel simplex and cautions that
parallel LP solving is useful for particular matrix shapes. Upstream
[HiGHS #2991](https://github.com/ERGO-Code/HiGHS/issues/2991) reports a similar
parallel-simplex stall on older versions; we have not established that its
native root cause is identical to this example.

## Data and reproduction

[metadata.json](metadata.json) records source commits, environment and limits for
all nine batches. [observations.jsonl](observations.jsonl) retains all 192 paired
observations, including the ungated trial and follow-up. Each line includes the
native HiGHS version, outcome and individual LP/MIP settings, statuses and times.
There were no subprocess timeouts or unexpected worker failures.

Run the complete six-case comparison from the candidate checkout:

```bash
git worktree add --detach benchmarks/results/issue85/baseline 35e9115
python benchmarks/benchmark_parallel_lp.py --compare-root benchmarks/results/issue85/baseline --output benchmarks/results/issue85/recheck.json
```

Use `--highs-path PATH` for an isolated `highspy` installation and `--repeats N`
to change repetitions. To reproduce a version-matrix-only batch, import the
benchmark module, set `CASES = ("issue85",)` and call `main()` with the same
arguments. Every worker records its actual native version.

The tests cover the version gate, restoration after an exception, both LP cut
loops with and without preliminary rounds, and bounded fresh-process copies of
the original example with automatic, one and two solver threads. Versions older
than 1.12 were not measured. These small local cases do not establish PACE or HPC
performance; no Snellius jobs were submitted for this investigation.
