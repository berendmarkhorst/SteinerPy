# Snellius assessment of current main

The bounded pilot completed on 23 October 2026 as Slurm job **28473801** on
`tcn14`, with exit code zero. It compares baseline
`9c980eca523c21101a5dd624901426a98d3cde56` with main after PRs #98, #101 and #103,
`a11d90eec231349640691f721a7fdca87b1578cb`. Both use the same harness,
`85b1235d1239896a3e463432ea936b610ac84866`, and clean source checkouts.

Current main reaches **20 known or solver-proven optima in 24 runs**, versus
**12** for the baseline. These are twelve cases repeated twice: ten distinct
cases reach an optimum with current main, versus six with the baseline.
Current main has eighteen solver proofs and two matches to a published optimum
without a solver proof. There are substantial gains on selected general-model
cases, but also a reproducible slowdown on the signed-rewards case.

## Conditions

- AMD EPYC 7H12, Linux, Python 3.12.3, native HiGHS **1.15.1**, NetworkX 3.7,
  NumPy 2.5.4 and SciPy 1.18.1. Both revisions use the same environment.
- Twelve fixed cases, two revisions, two repetitions: **48 fresh workers**.
  Each case runs baseline/candidate, then candidate/baseline. Up to three case
  sequences run concurrently on the same compute node.
- One solver/BLAS thread per worker; a **120-second whole-worker deadline**
  includes imports, construction, solving and certificate validation. The inner
  allowance reserves time for startup and validation. Linux enforces an 8 GiB
  address-space cap per worker. No Gurobi/WLS sessions are opened.
- The 16-core, 28-GiB allocation ran for **15 minutes 45 seconds**, from
  10:00:23 to 10:16:08 UTC (12:00:23 to 12:16:08 Europe/Oslo), below its
  40-minute cap. This is 4.2 allocated core-hours, not a billing statement.
  Worker wall times sum to 2,501.3 seconds because workers overlap.
- Six PACE inputs are instances 001/002 from each track, forcing the general
  formulation. Four combined-feature cases exercise the public API. Two classic
  controls use Track 1/001 and Track 2/001. The pilot measures these selected
  paths; the forced-general cases are not default classic API timings.

The first submission, **28473699**, failed after eleven seconds before any
benchmark worker ran: the site did not forward `PILOT_ROOT`. The harness now
falls back to Slurm's submission directory. Both attempts are retained in
[accounting.txt](accounting.txt); only the corrected job contributes observations.

## Outcomes

| Outcome | Baseline (24 runs) | Current main (24 runs) |
|---|---:|---:|
| Solver proves optimality | 12 | 18 |
| Matches published optimum, without solver proof | 0 | 2 |
| Valid PACE solution above known optimum | 4 | 0 |
| Valid, unproven combined-feature solution | 2 | 0 |
| Whole-worker timeout, no scored result | 6 | 4 |
| Validation failure, crash or memory-limit failure | 0 | 0 |

The following times are median full worker wall seconds over both repetitions.
Each listed outcome occurs in both repetitions. An unfinished run's duration is
not its time to optimum; the raw observations do not record when an incumbent
was first found. Timeout durations slightly exceed 120 seconds because of
supervisor polling and process termination.

| Case | Baseline seconds / result | Current main seconds / result |
|---|---|---|
| General Track 1/001 | 1.304 / optimum 503 proved | 0.802 / optimum 503 proved |
| General Track 1/002 | 120.107 / timeout | 15.973 / optimum 111 proved |
| General Track 2/001 | 118.714 / cost 1116, unproven | 118.776 / known optimum 1086, no solver proof |
| General Track 2/002 | 119.981 / cost 628, unproven | 32.092 / optimum 626 proved |
| General Track 3/001 | 120.108 / timeout | 120.117 / timeout |
| General Track 3/002 | 120.122 / timeout | 120.110 / timeout |
| Resource tree, seed 42 | 118.448 / cost 97, unproven | 2.557 / optimum 96 proved |
| Resource tree, seed 43 | 102.254 / optimum 94 proved | 5.859 / optimum 94 proved |
| Groups and forest, seed 44 | 1.354 / optimum 22 proved | 1.303 / optimum 22 proved |
| Signed rewards, seed 44 | 1.904 / optimum 19 proved | 4.108 / optimum 19 proved |
| Classic Track 1/001 | 0.703 / optimum 503 proved | 0.602 / optimum 503 proved |
| Classic Track 2/001 | 1.653 / optimum 1086 proved | 1.704 / optimum 1086 proved |

The strongest like-for-like speedup is resource-tree seed 43: **17.45 times
faster** to the same solver-proven optimum, using the ratio of median worker
times. Seed 42 also improves solution quality and finishes with a proof, while
the baseline exhausts its solver allowance without a proof. General Track 1/002
and Track 2/002 gain proofs within the cap. Track 2/001 reaches the known optimum
but still cannot prove it; both Track 3 cases remain unresolved at the deadline.

Classic controls and groups/forest remain close in absolute time. Two repeats,
shared-node scheduling, import overhead and roughly 0.1-second supervisor polling
do not justify interpreting the small timing differences as robust speedups.
For example, baseline General Track 1/001 takes 1.805 seconds on its first run
and 0.802 on its second; its median alone overstates the apparent advantage.

## The signed-rewards regression

Signed-rewards seed 44 is **2.16 times slower**, from 1.904 to 4.108 seconds,
while both revisions prove objective 19. The phase measurements and native
solver records identify the changed execution path:

- Baseline solves the original 222-column, 314-row model in one call. Median
  optimization time is 1.268 seconds; model construction is about 0.020 seconds.
- Current main reaches the one-second probe limit, then adds connectivity
  constraints and solves a 352-column, 403-row model. Median total optimization
  time is 3.450 seconds; model construction is only about 0.025 seconds.
- Both repetitions follow that same two-call path. The extra time is primarily
  optimization, not Python model construction or decoding.

This suggests that the fixed one-second strengthening trigger is too early for
this case on Snellius. The earlier local assessment did not show this slowdown;
all four generated-case input hashes match between the two environments.
Changing the trigger remains a follow-up experiment, not a validated fix.
The stronger model is valuable elsewhere: resource-tree seed 43 drops from
907 branch-and-bound nodes in the baseline to one node in the strengthened
second solve, after an unfinished probe.

## Validation and retained data

All 48 requested identities, limits, revisions and individual result files were
checked against the batch and manifest. Cross-version proof and input checks
pass. After retrieval, all **38 completed solution certificates** were rechecked
without running an optimizer: 22 PACE trees against their original edges,
terminals, costs and published references; sixteen combined-feature selections
against regenerated graphs and the feature validator. This rechecks feasibility
and objective accounting; solver optimality claims still rely on the solver's
reported proof status and the consistency checks.

The official dataset revision is
`4df73cea9c311faea7d03e6d6bffa8733c34a1aa`. All six PACE byte hashes match the
LF-normalized hashes in the earlier local assessment. Generated input hashes
also match across revisions, repetitions and the local assessment, despite the
different NetworkX versions. No failed or timed-out observation is discarded.

[manifest.json](manifest.json) records the experiment and dependencies.
[observations.jsonl](observations.jsonl) contains every result, including original
edge/node certificates, proof flags, timings, memory, phase measurements and
native solver records where available. Absolute source paths and the full host
name were removed, and dataset paths were made relative. The unsanitized worker
directories and batch log remain in the downloaded local results and on Snellius
under `~/steinerpy-benchmarks/connectivity-pilot-20261023/results-28473801` and
`connectivity-28473801.log` in its parent directory.

Use the [pilot instructions](../../../connectivity_pilot.md) with the pinned
revisions above and a new output directory to reproduce the experiment. The
[local 15-second assessment](../local-pilot-20261023/README.md) remains separate;
its different hardware and deadline must not be pooled with this run.

## Interpretation and next steps

The selected general-model improvements survive a controlled before/after
comparison on Snellius. They do not establish a universal speedup, an effect of
modularity itself, or parity with PACE winners. This pilot uses HiGHS and a
two-minute deadline; the exact PACE tracks allowed thirty minutes, and no
competitor or Gurobi configuration was rerun here. The historical
[PACE comparison](../../pace2018/README.md) measures older SteinerPy revisions.

The next focused change should investigate the strengthening trigger, preserving
the large resource-case gains while avoiding the signed-rewards regression.
Validate any proposed policy on these cases and additional seeds before another
small matched HPC comparison. After that, a current-main versus PACE-winner
comparison under identical conditions can address competitive performance;
this twelve-case revision comparison cannot answer that question.
